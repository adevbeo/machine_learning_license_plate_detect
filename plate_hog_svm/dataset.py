from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .features import HOGFeatureExtractor
from .io_utils import clamp_bbox, read_image


PATH_COLUMNS = ("candidate_path", "path", "image_path", "image", "plate_path", "file")
POSITIVE_LABELS = {"1", "true", "pos", "positive", "plate", "bien_so", "license_plate"}
NEGATIVE_LABELS = {"0", "-1", "false", "neg", "negative", "background", "non_plate", "not_plate"}


@dataclass(frozen=True)
class DatasetStats:
    samples: int
    positives: int
    negatives: int
    skipped: int


def parse_label(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized in POSITIVE_LABELS:
        return 1
    if normalized in NEGATIVE_LABELS:
        return -1
    return None


def resolve_sample_path(labels_path: Path, value: str) -> Path:
    path = Path(value.strip())
    if path.is_absolute() or path.exists():
        return path

    for candidate in (labels_path.parent / path, Path.cwd() / path):
        if candidate.exists():
            return candidate
    return labels_path.parent / path


def row_image_path(labels_path: Path, row: dict[str, str]) -> Path | None:
    for column in PATH_COLUMNS:
        value = (row.get(column) or "").strip()
        if value:
            return resolve_sample_path(labels_path, value)
    return None


def crop_from_bbox(image: np.ndarray, row: dict[str, str]) -> np.ndarray:
    try:
        x = int(float(row.get("x", "")))
        y = int(float(row.get("y", "")))
        w = int(float(row.get("w", "")))
        h = int(float(row.get("h", "")))
    except ValueError:
        return image

    if w <= 0 or h <= 0:
        return image

    image_h, image_w = image.shape[:2]
    x, y, w, h = clamp_bbox(x, y, w, h, image_w, image_h)
    return image[y : y + h, x : x + w]


def load_labeled_samples(
    labels_path: Path,
    extractor: HOGFeatureExtractor,
    require_both_classes: bool = True,
) -> tuple[np.ndarray, np.ndarray, DatasetStats]:
    samples: list[np.ndarray] = []
    labels: list[int] = []
    skipped = 0

    with labels_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        if "label" not in fieldnames:
            raise ValueError("CSV must contain a label column")
        if not fieldnames.intersection(PATH_COLUMNS):
            expected = ", ".join(PATH_COLUMNS)
            raise ValueError(f"CSV must contain one image path column: {expected}")

        for row in reader:
            label = parse_label(row.get("label", ""))
            image_path = row_image_path(labels_path, row)
            if label is None or image_path is None:
                skipped += 1
                continue

            try:
                image = crop_from_bbox(read_image(image_path), row)
                samples.append(extractor.extract(image))
                labels.append(label)
            except Exception as exc:
                print(f"[SKIP] {image_path}: {exc}", file=sys.stderr)
                skipped += 1

    if not samples:
        raise ValueError("No labeled samples found. Fill label with 1 for plate and 0 for non-plate.")

    y = np.array(labels, dtype=np.int32)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == -1))
    if require_both_classes and (positives == 0 or negatives == 0):
        raise ValueError(f"Need both classes to train. positives={positives}, negatives={negatives}")

    x = np.vstack(samples).astype(np.float32)
    return x, y, DatasetStats(len(y), positives, negatives, skipped)
