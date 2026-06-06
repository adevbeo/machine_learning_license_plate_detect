from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.pipeline import Pipeline


def metadata_path_for_model(model_path: Path) -> Path:
    return model_path.with_suffix(".json")


def save_sklearn_model(pipeline: Pipeline, model_path: Path) -> None:
    """Lưu sklearn Pipeline bằng joblib."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)


def load_sklearn_model(model_path: Path) -> Pipeline:
    """Load sklearn Pipeline từ file joblib."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    pipeline = joblib.load(model_path)
    if not isinstance(pipeline, Pipeline):
        raise ValueError(f"Loaded object is not a sklearn Pipeline: {type(pipeline)}")
    return pipeline


def save_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_metadata(model_path: Path, metadata_path: Path | None = None) -> tuple[dict[str, Any], bool]:
    path = metadata_path or metadata_path_for_model(model_path)
    if not path.exists():
        return {}, False
    return json.loads(path.read_text(encoding="utf-8")), True


def get_decision_scores(pipeline: Pipeline, samples: np.ndarray) -> np.ndarray:
    """Trả về raw decision function scores từ sklearn pipeline.

    Score dương = dự đoán là plate (class 1).
    Score âm = dự đoán là non-plate (class -1).
    """
    if len(samples) == 0:
        return np.empty((0,), dtype=np.float32)
    scores = pipeline.decision_function(samples)
    scores = scores.reshape(-1).astype(np.float32)

    estimator = pipeline.steps[-1][1]
    classes = getattr(estimator, "classes_", None)
    if classes is None or len(classes) != 2:
        return scores

    classes = np.asarray(classes)
    if classes[1] == 1:
        return scores
    if classes[0] == 1:
        return -scores
    raise ValueError(f"Cannot determine positive class from estimator classes: {classes.tolist()}")


def predict_labels(pipeline: Pipeline, samples: np.ndarray) -> np.ndarray:
    """Trả về nhãn dự đoán: 1 (plate) hoặc -1 (non-plate)."""
    if len(samples) == 0:
        return np.empty((0,), dtype=np.int32)
    predictions = pipeline.predict(samples)
    return predictions.reshape(-1).astype(np.int32)
