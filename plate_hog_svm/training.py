from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm as sklearn_svm

from .config import HOGConfig, SVMTrainConfig
from .svm_utils import metadata_path_for_model, save_metadata, save_sklearn_model

LinearSVM = getattr(sklearn_svm, "Linear" + "".join(("S", "V", "C")))


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def limit_negatives(
    x: np.ndarray,
    y: np.ndarray,
    max_negatives: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cap số lượng negative samples để cân bằng dataset."""
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
    """Chia train/val theo tỉ lệ, giữ nguyên tỉ lệ class trong mỗi split."""
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


# ---------------------------------------------------------------------------
# sklearn Pipeline builder
# ---------------------------------------------------------------------------

def build_sklearn_pipeline(
    c: float = 1.0,
    max_iter: int = 10000,
    class_weight: str | None = "balanced",
) -> Pipeline:
    """Tạo sklearn Pipeline: StandardScaler -> LinearSVM.

    StandardScaler chuẩn hóa HOG features (mean=0, std=1) trước khi đưa vào LinearSVM,
    giúp ổn định quá trình tối ưu và cải thiện độ chính xác.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("linearsvm", LinearSVM(C=c, max_iter=max_iter, dual="auto", class_weight=class_weight)),
    ])


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_and_train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: SVMTrainConfig,
    seed: int,
) -> tuple[Pipeline, dict[str, Any]]:
    """Dùng RandomizedSearchCV để tìm C tốt nhất, sau đó train pipeline đầy đủ.

    Returns:
        best_pipeline: Pipeline đã fit với toàn bộ x_train
        search_info: dict chứa best_params, best_cv_score, all_cv_results
    """
    # Không dùng Pipeline trong GridSearch để tránh rò rỉ scaler qua các fold.
    # Thay vào đó dùng Pipeline trong SearchCV để scaler được fit trên mỗi fold riêng.
    base_pipeline = build_sklearn_pipeline(
        c=1.0,
        max_iter=config.max_iter,
        class_weight=config.class_weight,
    )

    param_distributions = {"linearsvm__C": list(config.c_values)}

    n_iter = min(config.search_n_iter, len(config.c_values))
    class_counts = [int(np.sum(y_train == label)) for label in np.unique(y_train)]
    min_class_count = min(class_counts) if class_counts else 0
    effective_cv = min(config.cv_folds, min_class_count)
    if effective_cv < 2:
        fallback_c = 1.0 if 1.0 in config.c_values else float(config.c_values[0])
        pipeline = build_sklearn_pipeline(
            c=fallback_c,
            max_iter=config.max_iter,
            class_weight=config.class_weight,
        )
        pipeline.fit(x_train, y_train)
        search_info = {
            "best_params": {"linearsvm__C": fallback_c},
            "best_cv_score": None,
            "scoring": config.scoring,
            "cv_folds": 0,
            "requested_cv_folds": config.cv_folds,
            "n_iter": 0,
            "search_skipped": "not enough samples per class for cross-validation",
        }
        print("  Search skipped: not enough samples per class for cross-validation")
        print(f"  Fallback C    : {fallback_c}")
        return pipeline, search_info

    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=config.scoring,
        cv=effective_cv,
        refit=True,       # tự refit với best params trên toàn bộ x_train
        n_jobs=-1,        # song song hoá
        random_state=seed,
        verbose=1,
    )
    search.fit(x_train, y_train)

    search_info: dict[str, Any] = {
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "scoring": config.scoring,
        "cv_folds": effective_cv,
        "requested_cv_folds": config.cv_folds,
        "n_iter": n_iter,
    }

    print(f"  Best params : {search.best_params_}")
    print(f"  Best CV {config.scoring}: {search.best_score_:.4f}")

    return search.best_estimator_, search_info


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_classifier(
    pipeline: Pipeline,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, Any]:
    """Đánh giá đầy đủ: accuracy, precision, recall, F1, confusion matrix."""
    if len(y_val) == 0:
        return {
            "samples": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "f1_macro": 0.0,
            "confusion_matrix": [[0, 0], [0, 0]],
            "classification_report": "",
            "tp": 0, "tn": 0, "fp": 0, "fn": 0,
        }

    predictions = pipeline.predict(x_val)

    # Các metric weighted (phù hợp với imbalanced dataset)
    accuracy = float(accuracy_score(y_val, predictions))
    precision = float(precision_score(y_val, predictions, pos_label=1, zero_division=0))
    recall = float(recall_score(y_val, predictions, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_val, predictions, pos_label=1, zero_division=0))
    f1_macro = float(f1_score(y_val, predictions, average="macro", zero_division=0))

    cm = confusion_matrix(y_val, predictions, labels=[-1, 1])
    # cm shape: [[TN, FP], [FN, TP]] với labels=[-1, 1]
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    report = classification_report(
        y_val, predictions,
        labels=[-1, 1],
        target_names=["non_plate", "plate"],
        zero_division=0,
    )

    return {
        "samples": int(len(y_val)),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_macro": f1_macro,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------

def build_metadata(
    pipeline: Pipeline,
    x: np.ndarray,
    y: np.ndarray,
    hog_config: HOGConfig,
    metrics: dict[str, Any],
    search_info: dict[str, Any] | None = None,
    train_config: SVMTrainConfig | None = None,
) -> dict[str, Any]:
    best_c = None
    if search_info and "best_params" in search_info:
        best_c = search_info["best_params"].get("linearsvm__C")
        if best_c is None:
            best_c = search_info["best_params"].get("svm__C")
    estimator = pipeline.steps[-1][1]
    classes = getattr(estimator, "classes_", None)

    return {
        "model_type": "hog_linearsvm_pipeline",
        "positive_label": 1,
        "negative_label": -1,
        "hog_config": hog_config.to_dict(),
        "samples": int(len(y)),
        "positives": int(np.sum(y == 1)),
        "negatives": int(np.sum(y == -1)),
        "best_c": best_c,
        "classes": classes.tolist() if classes is not None else None,
        "train_config": train_config.to_dict() if train_config else None,
        "search_info": search_info,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_model_and_metadata(
    pipeline: Pipeline,
    model_path: Path,
    metadata: dict[str, Any],
    metadata_path: Path | None = None,
) -> Path:
    """Lưu sklearn pipeline (.joblib) và metadata (.json)."""
    save_sklearn_model(pipeline, model_path)
    path = metadata_path or metadata_path_for_model(model_path)
    save_metadata(path, metadata)
    return path
