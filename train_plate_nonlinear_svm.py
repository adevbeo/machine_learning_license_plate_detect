from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from plate_hog_svm.dataset import load_labeled_samples
from plate_hog_svm.features import HOGFeatureExtractor
from plate_hog_svm.svm_utils import metadata_path_for_model, save_metadata, save_sklearn_model
from plate_hog_svm.training import evaluate_classifier, limit_negatives, stratified_split
from plate_hog_svm.visualization import (
    load_labeled_preview_samples,
    save_feature_embedding_plot,
    save_hog_visualization_grid,
)


def parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one value")
    return values


def parse_gamma_list(value: str) -> list[str | float]:
    values: list[str | float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item in {"scale", "auto"}:
            values.append(item)
        else:
            values.append(float(item))
    if not values:
        raise ValueError("expected at least one gamma value")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train HOG + non-linear SVM using sklearn SVC with RBF kernel.",
    )
    parser.add_argument("--labels", default="outputs/labels_clean_final.csv", help="Clean labels CSV.")
    parser.add_argument("--model", default="models/plate_rbf_svm.joblib", help="Output model path.")
    parser.add_argument("--metadata", default="", help="Output metadata JSON. Defaults to model path with .json.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-negatives", type=int, default=0, help="Cap negative samples. 0 = keep all.")
    parser.add_argument("--c-values", default="0.1,1,10,100", help="Comma-separated SVC C values.")
    parser.add_argument("--gamma-values", default="scale,0.001,0.01,0.1", help="Comma-separated SVC gamma values.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds.")
    parser.add_argument("--scoring", default="f1", choices=["f1", "f1_macro", "precision", "recall", "accuracy"])
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--feature-plot", default="outputs/nonlinear/visualize/hog_feature_embedding.png")
    parser.add_argument("--hog-visualization", default="outputs/nonlinear/visualize/hog_visualization.png")
    parser.add_argument("--hog-visualization-limit", type=int, default=12)
    return parser.parse_args()


def print_metrics(metrics: dict[str, Any]) -> None:
    print(f"\n{'=' * 60}")
    print(f"  Validation Metrics ({metrics['samples']} samples)")
    print(f"{'=' * 60}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}  (plate class)")
    print(f"  Recall    : {metrics['recall']:.4f}  (plate class)")
    print(f"  F1        : {metrics['f1']:.4f}  (plate class)")
    print(f"  F1 macro  : {metrics['f1_macro']:.4f}")
    cm = metrics["confusion_matrix"]
    print("\n  Confusion Matrix (rows=actual, cols=predicted):")
    print("             non_plate  plate")
    print(f"  non_plate  {cm[0][0]:>9}  {cm[0][1]:>5}")
    print(f"  plate      {cm[1][0]:>9}  {cm[1][1]:>5}")
    print(f"  -> TP={metrics['tp']}  TN={metrics['tn']}  FP={metrics['fp']}  FN={metrics['fn']}")
    print(f"{'=' * 60}")


def main() -> int:
    args = parse_args()
    labels_path = Path(args.labels)
    if not labels_path.exists():
        print(f"Labels CSV not found: {labels_path}", file=sys.stderr)
        return 1

    try:
        c_values = parse_float_list(args.c_values)
        gamma_values = parse_gamma_list(args.gamma_values)
    except ValueError as exc:
        print(f"Invalid search values: {exc}", file=sys.stderr)
        return 1

    extractor = HOGFeatureExtractor()

    print(f"\n[1/5] Loading HOG features from {labels_path} ...")
    try:
        x, y, stats = load_labeled_samples(labels_path, extractor)
    except Exception as exc:
        print(f"Dataset load failed: {exc}", file=sys.stderr)
        return 1
    x, y = limit_negatives(x, y, args.max_negatives, args.seed)
    print(f"      Total: {len(y)}  (pos={int(np.sum(y == 1))}, neg={int(np.sum(y == -1))}, skip={stats.skipped})")

    if args.hog_visualization:
        print(f"[2/5] Writing HOG visualization to {args.hog_visualization} ...")
        samples = load_labeled_preview_samples([labels_path], max_samples=args.hog_visualization_limit)
        written = save_hog_visualization_grid(samples, extractor, Path(args.hog_visualization))
        print(f"      HOG visualization samples: {written}")

    if args.feature_plot:
        print(f"[3/5] Writing PCA feature plot to {args.feature_plot} ...")
        plot_info = save_feature_embedding_plot(
            x=x,
            y=y,
            output_path=Path(args.feature_plot),
            method="pca",
            dims=2,
            sample_size=2000,
            seed=args.seed,
        )
        variance = ", ".join(f"{value:.4f}" for value in plot_info.get("explained_variance_ratio", []))
        print(f"      Samples: {plot_info['samples']}")
        print(f"      PCA var: {variance}")

    train_idx, val_idx = stratified_split(y, args.val_ratio, args.seed)
    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]
    print(f"\n[4/5] Train/val split: train={len(train_idx)}, val={len(val_idx)}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        (
            "rbfsvm",
            SVC(
                kernel="rbf",
                class_weight=None if args.class_weight == "none" else args.class_weight,
                cache_size=1024,
            ),
        ),
    ])

    min_class_count = min(int(np.sum(y_train == -1)), int(np.sum(y_train == 1)))
    effective_cv = min(args.cv_folds, min_class_count)
    if effective_cv < 2:
        print("Training failed: not enough samples per class for CV", file=sys.stderr)
        return 1

    param_grid = {
        "rbfsvm__C": c_values,
        "rbfsvm__gamma": gamma_values,
    }
    print(f"      C candidates    : {c_values}")
    print(f"      gamma candidates: {gamma_values}")
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=args.scoring,
        cv=effective_cv,
        refit=True,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(x_train, y_train)
    print(f"      Best params : {search.best_params_}")
    print(f"      Best CV {args.scoring}: {search.best_score_:.4f}")

    metrics = evaluate_classifier(search.best_estimator_, x_val, y_val)
    print_metrics(metrics)

    model_path = Path(args.model)
    metadata_path = Path(args.metadata) if args.metadata else metadata_path_for_model(model_path)
    estimator = search.best_estimator_.steps[-1][1]
    metadata = {
        "model_type": "hog_rbf_svm_pipeline",
        "positive_label": 1,
        "negative_label": -1,
        "hog_config": extractor.config.to_dict(),
        "samples": int(len(y)),
        "positives": int(np.sum(y == 1)),
        "negatives": int(np.sum(y == -1)),
        "kernel": "rbf",
        "best_c": search.best_params_.get("rbfsvm__C"),
        "best_gamma": search.best_params_.get("rbfsvm__gamma"),
        "classes": estimator.classes_.tolist(),
        "train_config": {
            "c_values": c_values,
            "gamma_values": gamma_values,
            "cv_folds": effective_cv,
            "requested_cv_folds": args.cv_folds,
            "scoring": args.scoring,
            "class_weight": None if args.class_weight == "none" else args.class_weight,
        },
        "search_info": {
            "best_params": search.best_params_,
            "best_cv_score": float(search.best_score_),
            "scoring": args.scoring,
            "cv_folds": effective_cv,
        },
        "metrics": metrics,
    }

    print(f"\n[5/5] Saving model ...")
    save_sklearn_model(search.best_estimator_, model_path)
    save_metadata(metadata_path, metadata)
    print(f"      Model    : {model_path}")
    print(f"      Metadata : {metadata_path}")
    print("\nDone!\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
