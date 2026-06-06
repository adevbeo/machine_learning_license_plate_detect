from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def metadata_path_for_model(model_path: Path) -> Path:
    return model_path.with_suffix(".json")


def load_svm(model_path: Path) -> cv2.ml_SVM:
    if not model_path.exists():
        raise FileNotFoundError(f"SVM model not found: {model_path}")
    svm = cv2.ml.SVM_load(str(model_path))
    if svm is None or not svm.isTrained():
        raise ValueError(f"SVM model is not trained or cannot be loaded: {model_path}")
    return svm


def save_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_metadata(model_path: Path, metadata_path: Path | None = None) -> tuple[dict[str, Any], bool]:
    path = metadata_path or metadata_path_for_model(model_path)
    if not path.exists():
        return {}, False
    return json.loads(path.read_text(encoding="utf-8")), True


def predict_labels(svm: cv2.ml_SVM, samples: np.ndarray) -> np.ndarray:
    if len(samples) == 0:
        return np.empty((0,), dtype=np.int32)
    _, predictions = svm.predict(samples)
    return predictions.reshape(-1).astype(np.int32)


def raw_scores(svm: cv2.ml_SVM, samples: np.ndarray) -> np.ndarray:
    if len(samples) == 0:
        return np.empty((0,), dtype=np.float32)
    _, raw = svm.predict(samples, flags=cv2.ml.StatModel_RAW_OUTPUT)
    return raw.reshape(-1).astype(np.float32)


def positive_margins(svm: cv2.ml_SVM, samples: np.ndarray, raw_score_multiplier: float) -> np.ndarray:
    return raw_scores(svm, samples) * float(raw_score_multiplier)
