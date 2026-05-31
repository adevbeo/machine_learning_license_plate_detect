from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np

from detect_plates import read_image
from plate_ocr import CHARSET, hog_feature


def parse_label(value: str) -> str:
    label = value.strip().upper()
    if len(label) == 1 and label in CHARSET:
        return label
    return ""


def resolve_path(labels_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return labels_path.parent / path


def load_dataset(labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[str] = []
    skipped = 0

    with labels_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "char_path" not in (reader.fieldnames or []) or "label" not in (reader.fieldnames or []):
            raise ValueError("CSV must contain char_path and label columns")

        for row in reader:
            label = parse_label(row.get("label", ""))
            if not label:
                skipped += 1
                continue

            char_path = resolve_path(labels_path, row["char_path"])
            try:
                image = read_image(char_path)
            except Exception as exc:
                print(f"[SKIP] {char_path}: {exc}", file=sys.stderr)
                skipped += 1
                continue

            features.append(hog_feature(image))
            labels.append(label)

    if skipped:
        print(f"Skipped unlabeled/unreadable rows: {skipped}")
    if not features:
        raise ValueError("No labeled character samples found")

    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        raise ValueError("Need at least two character classes to train")

    return np.vstack(features).astype(np.float32), np.array(labels)


def evaluate_knn(features: np.ndarray, labels: np.ndarray, val_ratio: float, seed: int, k: int) -> dict[str, float | int]:
    if val_ratio <= 0:
        return {"samples": 0, "accuracy": 0.0}

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for label in sorted(set(labels)):
        indices = [index for index, value in enumerate(labels) if value == label]
        rng.shuffle(indices)
        if len(indices) <= 1:
            train_indices.extend(indices)
            continue
        val_count = max(1, int(round(len(indices) * val_ratio)))
        val_count = min(val_count, len(indices) - 1)
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    if not val_indices:
        return {"samples": 0, "accuracy": 0.0}

    train_x = features[train_indices]
    train_y = labels[train_indices]
    val_x = features[val_indices]
    val_y = labels[val_indices]
    correct = 0

    for sample, expected in zip(val_x, val_y):
        distances = np.linalg.norm(train_x - sample.reshape(1, -1), axis=1)
        order = np.argsort(distances)[: min(k, len(distances))]
        votes: dict[str, list[float]] = {}
        for index in order:
            votes.setdefault(str(train_y[index]), []).append(float(distances[index]))
        predicted = sorted(votes.items(), key=lambda item: float(np.mean(item[1])))[0][0]
        correct += int(predicted == expected)

    return {"samples": len(val_indices), "accuracy": correct / float(max(1, len(val_indices)))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a HOG KNN OCR model from reviewed character crops.")
    parser.add_argument("--labels", default="outputs/characters.csv", help="Character CSV with reviewed labels.")
    parser.add_argument("--model", default="models/char_knn.npz", help="Output character model path.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio per class.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--k", type=int, default=7, help="K used for validation preview.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels_path = Path(args.labels)
    if not labels_path.exists():
        print(f"Labels CSV not found: {labels_path}", file=sys.stderr)
        return 1

    try:
        features, labels = load_dataset(labels_path)
        metrics = evaluate_knn(features, labels, args.val_ratio, args.seed, args.k)
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1

    model_path = Path(args.model)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(model_path, features=features, labels=labels)

    counts = {label: int(np.sum(labels == label)) for label in sorted(set(labels))}
    print(f"Samples: {len(labels)}")
    print(f"Classes: {counts}")
    if metrics["samples"]:
        print(f"Val: accuracy={metrics['accuracy']:.4f}, samples={metrics['samples']}")
    else:
        print("Val: skipped because validation split is empty")
    print(f"Model: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
