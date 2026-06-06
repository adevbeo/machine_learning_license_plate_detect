from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from plate_hog_svm.dataset import load_labeled_samples
from plate_hog_svm.features import HOGFeatureExtractor
from plate_hog_svm.training import (
    build_metadata,
    evaluate_classifier,
    limit_negatives,
    save_model_and_metadata,
    stratified_split,
    train_linear_svm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Class 1: preprocess image patches, extract HOG, and train a plate/non-plate SVM.",
    )
    parser.add_argument("--labels", default="outputs/candidates/labels.csv", help="CSV with image path columns and label.")
    parser.add_argument("--model", default="models/plate_svm.yml", help="Output OpenCV SVM model path.")
    parser.add_argument("--metadata", default="", help="Optional metadata JSON path. Defaults to model path with .json.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio per class.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split/subsampling.")
    parser.add_argument("--c", type=float, default=1.0, help="Linear SVM C value.")
    parser.add_argument("--max-negatives", type=int, default=0, help="Optional cap for negative samples. 0 keeps all.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels_path = Path(args.labels)
    if not labels_path.exists():
        print(f"Labels CSV not found: {labels_path}", file=sys.stderr)
        return 1

    extractor = HOGFeatureExtractor()
    try:
        x, y, stats = load_labeled_samples(labels_path, extractor)
        x, y = limit_negatives(x, y, args.max_negatives, args.seed)
        train_indices, val_indices = stratified_split(y, args.val_ratio, args.seed)
        svm = train_linear_svm(x[train_indices], y[train_indices], args.c)
        metrics = evaluate_classifier(svm, x[val_indices], y[val_indices])
        metadata = build_metadata(svm, x, y, extractor.config, metrics)
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1

    model_path = Path(args.model)
    metadata_path = Path(args.metadata) if args.metadata else None
    written_metadata = save_model_and_metadata(svm, model_path, metadata, metadata_path)

    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == -1))
    print(f"Samples: {len(y)} total, {positives} positive, {negatives} negative")
    if stats.skipped:
        print(f"Skipped rows: {stats.skipped}")
    print(f"Train: {len(train_indices)} | Val: {len(val_indices)}")
    if metrics["samples"]:
        print(
            "Val: "
            f"accuracy={metrics['accuracy']:.4f}, "
            f"precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, "
            f"tp={metrics['tp']}, tn={metrics['tn']}, fp={metrics['fp']}, fn={metrics['fn']}"
        )
    else:
        print("Val: skipped because validation split is empty")
    print(f"Raw score multiplier: {metadata['raw_score_multiplier']}")
    print(f"Model: {model_path}")
    print(f"Metadata: {written_metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
