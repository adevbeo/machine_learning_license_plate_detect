from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.pipeline import Pipeline

from .config import DetectionConfig, HOGConfig
from .features import HOGFeatureExtractor
from .io_utils import clamp_bbox, resize_for_detection
from .nms import non_max_suppression
from .svm_utils import get_decision_scores, load_metadata, load_sklearn_model
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
    """Chuyển SVM decision score sang confidence [0, 1] bằng sigmoid."""
    value = max(-50.0, min(50.0, float(margin)))
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


class SlidingWindowDetector:
    """Phát hiện biển số bằng sliding window + sklearn SVM.

    Luồng:
    1. Resize ảnh về max_width.
    2. Sinh các window kích thước đa dạng (multi-scale).
    3. Batch-extract HOG features và tính decision score bằng sklearn pipeline.
    4. Thu thập tất cả candidates có score > threshold.
    5. Áp dụng NMS để loại bỏ box trùng lặp.
    6. Trả về danh sách detection (tối đa top_k).
    """

    def __init__(
        self,
        pipeline: Pipeline,
        extractor: HOGFeatureExtractor,
    ) -> None:
        self.pipeline = pipeline
        self.extractor = extractor

    @classmethod
    def from_model_file(
        cls,
        model_path: Path,
        metadata_path: Path | None = None,
    ) -> "SlidingWindowDetector":
        pipeline = load_sklearn_model(model_path)
        metadata, _ = load_metadata(model_path, metadata_path)
        hog_config = HOGConfig.from_dict(metadata.get("hog_config"))
        return cls(pipeline, HOGFeatureExtractor(hog_config))

    def detect(
        self,
        image: np.ndarray,
        config: DetectionConfig | None = None,
    ) -> list[Detection]:
        """Phát hiện biển số trong ảnh.

        Returns:
            Danh sách Detection sau NMS, sắp xếp theo score giảm dần.
            Trống nếu không tìm thấy biển số nào vượt threshold.
        """
        config = config or DetectionConfig()
        resized, scale = resize_for_detection(image, config.max_width)
        image_h, image_w = resized.shape[:2]
        sizes = build_window_sizes((image_h, image_w), config.window_heights, config.aspect_ratios)
        total_windows = 0
        candidates: list[Detection] = []

        batch_windows: list[Window] = []
        batch_crops: list[np.ndarray] = []

        def flush() -> None:
            nonlocal total_windows
            if not batch_windows:
                return

            features = self.extractor.extract_batch(batch_crops)
            scores = get_decision_scores(self.pipeline, features)

            for window, score in zip(batch_windows, scores):
                total_windows += 1
                if float(score) < config.score_threshold:
                    continue
                det = self._make_detection(
                    original=image,
                    scale=scale,
                    window=window,
                    score=float(score),
                    windows_scanned=total_windows,
                )
                candidates.append(det)

            batch_windows.clear()
            batch_crops.clear()

        for window in iter_sliding_windows((image_h, image_w), sizes, config.stride_ratio):
            crop = resized[window.y: window.y + window.h, window.x: window.x + window.w]
            if crop.size == 0:
                continue
            batch_windows.append(window)
            batch_crops.append(crop)
            if len(batch_windows) >= max(1, config.batch_size):
                flush()
        flush()

        return non_max_suppression(
            candidates,
            iou_threshold=config.nms.iou_threshold,
            top_k=config.nms.top_k,
        )

    def detect_best(
        self,
        image: np.ndarray,
        config: DetectionConfig | None = None,
    ) -> Detection | None:
        """Shortcut: trả về detection tốt nhất duy nhất (hoặc None).

        Tương đương với ``top_k=1`` trong detect().
        """
        results = self.detect(image, config)
        return results[0] if results else None

    def _make_detection(
        self,
        original: np.ndarray,
        scale: float,
        window: Window,
        score: float,
        windows_scanned: int,
    ) -> Detection:
        if scale != 1.0:
            inv_scale = 1.0 / scale
            x = int(round(window.x * inv_scale))
            y = int(round(window.y * inv_scale))
            w = int(round(window.w * inv_scale))
            h = int(round(window.h * inv_scale))
        else:
            x, y, w, h = window.x, window.y, window.w, window.h

        x, y, w, h = clamp_bbox(x, y, w, h, original.shape[1], original.shape[0])
        crop = original[y: y + h, x: x + w]
        return Detection(
            bbox=(x, y, w, h),
            score=score,
            confidence=confidence_from_margin(score),
            crop=crop,
            windows_scanned=windows_scanned,
        )


def draw_detection(image: np.ndarray, detection: Detection | None) -> np.ndarray:
    """Vẽ bbox lên ảnh (1 detection duy nhất)."""
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


def draw_detections(image: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Vẽ nhiều bbox lên ảnh với màu phân biệt theo rank."""
    if image.ndim == 2:
        debug = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        debug = image.copy()

    # Màu: xanh lá cho detection tốt nhất, xanh dương cho phần còn lại
    colors = [(0, 255, 0)] + [(255, 180, 0)] * max(0, len(detections) - 1)

    for i, det in enumerate(detections):
        color = colors[i] if i < len(colors) else (128, 128, 128)
        x, y, w, h = det.bbox
        cv2.rectangle(debug, (x, y), (x + w, y + h), color, 2)
        label = f"#{i + 1} {det.confidence:.2f}"
        cv2.putText(debug, label, (x, max(16, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return debug
