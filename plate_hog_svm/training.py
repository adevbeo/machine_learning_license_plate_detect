from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import HOGConfig
from .svm_utils import metadata_path_for_model, predict_labels, raw_scores, save_metadata


def limit_negatives(
    x: np.ndarray,
    y: np.ndarray,
    max_negatives: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if max_negatives <= 0:
        return x, y

    rng = random.Random(seed)
    positive_indices = [index for index, label in enumerate(y) if label == 1]
    negative_indices = [index for index, label in enumerate(y) if label == -1]
    if len(negative_indices) <= max_negatives:
        return x, y

    rng.shuffle(negative_indices)
    kept = sorted(positive_indices + negative_indices[:max_negatives])
    return x[kept], y[kept]


def stratified_split(
    y: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for label in (-1, 1):
        indices = [index for index, value in enumerate(y) if value == label]
        rng.shuffle(indices)
        if len(indices) <= 1 or val_ratio <= 0:
            train_indices.extend(indices)
            continue

        val_count = max(1, int(round(len(indices) * val_ratio)))
        val_count = min(val_count, len(indices) - 1)
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return np.array(train_indices, dtype=np.int32), np.array(val_indices, dtype=np.int32)


def train_linear_svm(x_train: np.ndarray, y_train: np.ndarray, c_value: float) -> cv2.ml_SVM:
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(float(c_value))
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 5000, 1e-5))
    svm.train(x_train, cv2.ml.ROW_SAMPLE, y_train.reshape(-1, 1))
    return svm


def evaluate_classifier(svm: cv2.ml_SVM, x_val: np.ndarray, y_val: np.ndarray) -> dict[str, float | int]:
    if len(y_val) == 0:
        return {"samples": 0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0}

    predictions = predict_labels(svm, x_val)
    tp = int(np.sum((predictions == 1) & (y_val == 1)))
    tn = int(np.sum((predictions == -1) & (y_val == -1)))
    fp = int(np.sum((predictions == 1) & (y_val == -1)))
    fn = int(np.sum((predictions == -1) & (y_val == 1)))

    accuracy = (tp + tn) / float(max(1, len(y_val)))
    precision = tp / float(max(1, tp + fp))
    recall = tp / float(max(1, tp + fn))
    return {
        "samples": int(len(y_val)),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def raw_score_multiplier_for_positive_class(svm: cv2.ml_SVM, x: np.ndarray, y: np.ndarray) -> float:
    raw = raw_scores(svm, x)
    positive_mean = float(np.mean(raw[y == 1]))
    negative_mean = float(np.mean(raw[y == -1]))
    return 1.0 if positive_mean > negative_mean else -1.0


def build_metadata(
    svm: cv2.ml_SVM,
    x: np.ndarray,
    y: np.ndarray,
    hog_config: HOGConfig,
    metrics: dict[str, float | int],
) -> dict[str, Any]:
    return {
        "model_type": "opencv_linear_svm",
        "positive_label": 1,
        "negative_label": -1,
        "raw_score_multiplier": raw_score_multiplier_for_positive_class(svm, x, y),
        "hog_config": hog_config.to_dict(),
        "samples": int(len(y)),
        "positives": int(np.sum(y == 1)),
        "negatives": int(np.sum(y == -1)),
        "metrics": metrics,
    }


def save_model_and_metadata(
    svm: cv2.ml_SVM,
    model_path: Path,
    metadata: dict[str, Any],
    metadata_path: Path | None = None,
) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    svm.save(str(model_path))
    path = metadata_path or metadata_path_for_model(model_path)
    save_metadata(path, metadata)
    return path
