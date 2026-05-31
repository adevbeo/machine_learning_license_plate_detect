from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

from detect_plates import read_image


HOG_SIZE = (96, 64)


def parse_label(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "pos", "positive", "plate", "bien_so"}:
        return 1
    if normalized in {"0", "-1", "false", "neg", "negative", "background", "non_plate", "not_plate"}:
        return -1
    return None


def resolve_candidate_path(labels_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path

    from_labels_dir = labels_path.parent / path
    if from_labels_dir.exists():
        return from_labels_dir

    return path


def extract_hog(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, HOG_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    hog = cv2.HOGDescriptor(
        _winSize=HOG_SIZE,
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    features = hog.compute(gray)
    return features.reshape(-1).astype(np.float32)


def load_dataset(labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
    samples: list[np.ndarray] = []
    labels: list[int] = []
    skipped = 0

    with labels_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "candidate_path" not in (reader.fieldnames or []) or "label" not in (reader.fieldnames or []):
            raise ValueError("CSV must contain candidate_path and label columns")

        for row in reader:
            label = parse_label(row.get("label", ""))
            if label is None:
                skipped += 1
                continue

            candidate_path = resolve_candidate_path(labels_path, row["candidate_path"])
            try:
                image = read_image(candidate_path)
            except Exception as exc:
                print(f"[SKIP] {candidate_path}: {exc}", file=sys.stderr)
                skipped += 1
                continue

            samples.append(extract_hog(image))
            labels.append(label)

    if skipped:
        print(f"Skipped unlabeled/unreadable rows: {skipped}")
    if not samples:
        raise ValueError("No labeled samples found. Fill label with 1 for plate and 0 for non-plate.")

    x = np.vstack(samples).astype(np.float32)
    y = np.array(labels, dtype=np.int32)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == -1))
    if positives == 0 or negatives == 0:
        raise ValueError(f"Need both classes to train. positives={positives}, negatives={negatives}")

    return x, y


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


def train_svm(x_train: np.ndarray, y_train: np.ndarray, c_value: float) -> cv2.ml_SVM:
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setC(c_value)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 5000, 1e-5))
    svm.train(x_train, cv2.ml.ROW_SAMPLE, y_train.reshape(-1, 1))
    return svm


def evaluate(svm: cv2.ml_SVM, x_val: np.ndarray, y_val: np.ndarray) -> dict[str, float | int]:
    if len(y_val) == 0:
        return {"samples": 0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0}

    _, raw_predictions = svm.predict(x_val)
    predictions = raw_predictions.reshape(-1).astype(np.int32)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a classical HOG + SVM plate/non-plate classifier.")
    parser.add_argument("--labels", default="outputs/candidates/labels.csv", help="Candidate CSV with reviewed labels.")
    parser.add_argument("--model", default="models/plate_svm.yml", help="Output OpenCV SVM model path.")
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

    try:
        x, y = load_dataset(labels_path)
        x, y = limit_negatives(x, y, args.max_negatives, args.seed)
        train_indices, val_indices = stratified_split(y, args.val_ratio, args.seed)
        svm = train_svm(x[train_indices], y[train_indices], args.c)
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1

    metrics = evaluate(svm, x[val_indices], y[val_indices])
    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    svm.save(str(model_path))

    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == -1))
    print(f"Samples: {len(y)} total, {positives} positive, {negatives} negative")
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
    print(f"Model: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
