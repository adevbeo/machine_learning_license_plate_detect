from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import DetectionConfig, HOGConfig
from .features import HOGFeatureExtractor
from .io_utils import clamp_bbox, resize_for_detection
from .svm_utils import load_metadata, load_svm, positive_margins
from .windows import Window, build_window_sizes, iter_sliding_windows


@dataclass(frozen=True)
class Detection:
    bbox: tuple[int, int, int, int]
    score: float
    confidence: float
    crop: np.ndarray
    windows_scanned: int

    @property
    def box(self) -> np.ndarray:
        x, y, w, h = self.bbox
        return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32)


def confidence_from_margin(margin: float) -> float:
    value = max(-50.0, min(50.0, float(margin)))
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


class SlidingWindowDetector:
    def __init__(
        self,
        svm: cv2.ml_SVM,
        extractor: HOGFeatureExtractor,
        raw_score_multiplier: float,
    ) -> None:
        self.svm = svm
        self.extractor = extractor
        self.raw_score_multiplier = float(raw_score_multiplier)

    @classmethod
    def from_model_file(
        cls,
        model_path: Path,
        metadata_path: Path | None = None,
    ) -> "SlidingWindowDetector":
        svm = load_svm(model_path)
        metadata, _ = load_metadata(model_path, metadata_path)
        hog_config = HOGConfig.from_dict(metadata.get("hog_config"))
        raw_multiplier = float(metadata.get("raw_score_multiplier", -1.0))
        return cls(svm, HOGFeatureExtractor(hog_config), raw_multiplier)

    def detect(self, image: np.ndarray, config: DetectionConfig | None = None) -> Detection | None:
        config = config or DetectionConfig()
        resized, scale = resize_for_detection(image, config.max_width)
        image_h, image_w = resized.shape[:2]
        sizes = build_window_sizes((image_h, image_w), config.window_heights, config.aspect_ratios)
        total_windows = 0
        best: Detection | None = None

        batch_windows: list[Window] = []
        batch_crops: list[np.ndarray] = []

        def flush() -> None:
            nonlocal best, total_windows
            if not batch_windows:
                return

            features = self.extractor.extract_batch(batch_crops)
            margins = positive_margins(self.svm, features, self.raw_score_multiplier)
            for window, margin in zip(batch_windows, margins):
                total_windows += 1
                candidate = self._candidate_from_window(image, resized, scale, window, float(margin), total_windows)
                if best is None or candidate.score > best.score:
                    best = candidate

            batch_windows.clear()
            batch_crops.clear()

        for window in iter_sliding_windows((image_h, image_w), sizes, config.stride_ratio):
            crop = resized[window.y : window.y + window.h, window.x : window.x + window.w]
            if crop.size == 0:
                continue
            batch_windows.append(window)
            batch_crops.append(crop)
            if len(batch_windows) >= max(1, config.batch_size):
                flush()
        flush()

        if best is None or best.score < config.score_threshold:
            return None
        return Detection(best.bbox, best.score, best.confidence, best.crop, total_windows)

    def _candidate_from_window(
        self,
        original: np.ndarray,
        resized: np.ndarray,
        scale: float,
        window: Window,
        margin: float,
        windows_scanned: int,
    ) -> Detection:
        if scale != 1.0:
            inv_scale = 1.0 / scale
            x = int(round(window.x * inv_scale))
            y = int(round(window.y * inv_scale))
            w = int(round(window.w * inv_scale))
            h = int(round(window.h * inv_scale))
            x, y, w, h = clamp_bbox(x, y, w, h, original.shape[1], original.shape[0])
            crop = original[y : y + h, x : x + w]
            return Detection((x, y, w, h), margin, confidence_from_margin(margin), crop, windows_scanned)

        x, y, w, h = clamp_bbox(window.x, window.y, window.w, window.h, resized.shape[1], resized.shape[0])
        crop = original[y : y + h, x : x + w]
        return Detection((x, y, w, h), margin, confidence_from_margin(margin), crop, windows_scanned)


def draw_detection(image: np.ndarray, detection: Detection | None) -> np.ndarray:
    if image.ndim == 2:
        debug = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        debug = image.copy()

    if detection is None:
        return debug

    x, y, w, h = detection.bbox
    cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
    label = f"SVM {detection.confidence:.2f}"
    cv2.putText(debug, label, (x, max(16, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return debug
