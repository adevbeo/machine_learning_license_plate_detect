from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline

from .config import DetectionConfig, HOGConfig
from .features import HOGFeatureExtractor
from .io_utils import clamp_bbox, read_image, resize_for_detection
from .svm_utils import get_decision_scores, load_metadata, load_sklearn_model
from .windows import Window, build_window_sizes, iter_sliding_windows


@dataclass(frozen=True)
class HardNegativeSample:
    source_image: Path
    bbox: tuple[int, int, int, int]
    score: float
    crop: np.ndarray
    feature: np.ndarray


def _sample_from_window(
    original: np.ndarray,
    image_path: Path,
    scale: float,
    window: Window,
    score: float,
    feature: np.ndarray,
) -> HardNegativeSample:
    if scale != 1.0:
        inv_scale = 1.0 / scale
        x = int(round(window.x * inv_scale))
        y = int(round(window.y * inv_scale))
        w = int(round(window.w * inv_scale))
        h = int(round(window.h * inv_scale))
    else:
        x, y, w, h = window.x, window.y, window.w, window.h

    x, y, w, h = clamp_bbox(x, y, w, h, original.shape[1], original.shape[0])
    crop = original[y : y + h, x : x + w]
    return HardNegativeSample(
        source_image=image_path,
        bbox=(x, y, w, h),
        score=score,
        crop=crop,
        feature=feature.astype(np.float32, copy=True),
    )


def mine_hard_negative_samples(
    negative_image_paths: list[Path],
    pipeline: Pipeline,
    extractor: HOGFeatureExtractor,
    config: DetectionConfig | None = None,
    score_threshold: float = 0.0,
    max_per_image: int = 5,
    verbose: bool = True,
) -> list[HardNegativeSample]:
    """Collect high-scoring false positives from images known to contain no plates."""
    config = config or DetectionConfig()
    collected: list[HardNegativeSample] = []
    batch_size = max(1, config.batch_size)

    for img_idx, img_path in enumerate(negative_image_paths, start=1):
        try:
            image = read_image(img_path)
        except Exception as exc:
            if verbose:
                print(f"[SKIP] {img_path}: {exc}", file=sys.stderr)
            continue

        resized, scale = resize_for_detection(image, config.max_width)
        image_h, image_w = resized.shape[:2]
        sizes = build_window_sizes((image_h, image_w), config.window_heights, config.aspect_ratios)

        best_for_image: list[HardNegativeSample] = []
        batch_windows: list[Window] = []
        batch_crops: list[np.ndarray] = []

        def flush() -> None:
            if not batch_windows:
                return

            features = extractor.extract_batch(batch_crops)
            scores = get_decision_scores(pipeline, features)
            for window, score, feature in zip(batch_windows, scores, features):
                score_value = float(score)
                if score_value <= score_threshold:
                    continue
                best_for_image.append(
                    _sample_from_window(image, img_path, scale, window, score_value, feature)
                )

            if len(best_for_image) > max_per_image:
                best_for_image.sort(key=lambda sample: sample.score, reverse=True)
                del best_for_image[max_per_image:]

            batch_windows.clear()
            batch_crops.clear()

        for window in iter_sliding_windows((image_h, image_w), sizes, config.stride_ratio):
            crop = resized[window.y : window.y + window.h, window.x : window.x + window.w]
            if crop.size == 0:
                continue
            batch_windows.append(window)
            batch_crops.append(crop)
            if len(batch_windows) >= batch_size:
                flush()
        flush()

        best_for_image.sort(key=lambda sample: sample.score, reverse=True)
        selected = best_for_image[:max_per_image]
        collected.extend(selected)

        if verbose:
            print(
                f"[{img_idx}/{len(negative_image_paths)}] {img_path.name}: "
                f"+{len(selected)} hard negatives (total={len(collected)})",
                file=sys.stderr,
            )

    return collected


def mine_hard_negatives(
    negative_image_paths: list[Path],
    pipeline: Pipeline,
    extractor: HOGFeatureExtractor,
    config: DetectionConfig | None = None,
    score_threshold: float = 0.0,
    max_per_image: int = 5,
    verbose: bool = True,
) -> np.ndarray:
    """Backward-compatible helper returning raw HOG features only."""
    samples = mine_hard_negative_samples(
        negative_image_paths=negative_image_paths,
        pipeline=pipeline,
        extractor=extractor,
        config=config,
        score_threshold=score_threshold,
        max_per_image=max_per_image,
        verbose=verbose,
    )
    if not samples:
        return np.empty((0, extractor.feature_length), dtype=np.float32)
    return np.vstack([sample.feature for sample in samples]).astype(np.float32)


def load_detector_for_mining(
    model_path: Path,
    metadata_path: Path | None = None,
) -> tuple[Pipeline, HOGFeatureExtractor]:
    pipeline = load_sklearn_model(model_path)
    metadata, _ = load_metadata(model_path, metadata_path)
    hog_config = HOGConfig.from_dict(metadata.get("hog_config"))
    extractor = HOGFeatureExtractor(hog_config)
    return pipeline, extractor
