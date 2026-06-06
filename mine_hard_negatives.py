from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from plate_hog_svm.config import DetectionConfig
from plate_hog_svm.hard_negatives import load_detector_for_mining, mine_hard_negative_samples
from plate_hog_svm.io_utils import iter_images, write_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hard Negative Mining: scan images known to contain no plates, "
            "collect high-scoring false-positive windows, and save them as "
            "negative crops plus a labels CSV for retraining."
        ),
    )
    parser.add_argument(
        "--negatives",
        required=True,
        help="Folder or file containing background/negative images with no license plates.",
    )
    parser.add_argument(
        "--model",
        default="models/plate_svm.joblib",
        help="Trained sklearn Pipeline (.joblib).",
    )
    parser.add_argument(
        "--metadata",
        default="",
        help="Model metadata JSON path. Defaults to model path + .json.",
    )
    parser.add_argument(
        "--output",
        default="outputs/hard_negatives",
        help="Directory to save hard negative crops, labels.csv, and feature backup.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help=(
            "Only keep windows with SVM score > threshold. "
            "Use 0.0 to collect every false positive; use a higher value for more confident false positives."
        ),
    )
    parser.add_argument(
        "--max-per-image",
        type=int,
        default=5,
        help="Max hard negatives to collect per image.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=900,
        help="Max image width before scanning.",
    )
    parser.add_argument(
        "--stride-ratio",
        type=float,
        default=0.25,
        help="Sliding window stride ratio.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Windows scored per batch during mining.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of images to scan. 0 means all.",
    )
    return parser.parse_args()


def safe_stem(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def main() -> int:
    args = parse_args()

    try:
        pipeline, extractor = load_detector_for_mining(
            Path(args.model),
            Path(args.metadata) if args.metadata else None,
        )
    except Exception as exc:
        print(f"Cannot load model: {exc}", file=sys.stderr)
        return 1

    negative_paths = iter_images(Path(args.negatives))
    if args.limit:
        negative_paths = negative_paths[: args.limit]
    if not negative_paths:
        print(f"No images found in {args.negatives}", file=sys.stderr)
        return 1

    config = DetectionConfig(
        max_width=args.max_width,
        stride_ratio=args.stride_ratio,
        batch_size=args.batch_size,
    )

    print(f"Scanning {len(negative_paths)} images for hard negatives ...")
    print(f"Score threshold: {args.score_threshold}, max per image: {args.max_per_image}")

    samples = mine_hard_negative_samples(
        negative_image_paths=negative_paths,
        pipeline=pipeline,
        extractor=extractor,
        config=config,
        score_threshold=args.score_threshold,
        max_per_image=args.max_per_image,
        verbose=True,
    )

    output_dir = Path(args.output)
    crops_dir = output_dir / "crops"
    labels_path = output_dir / "labels.csv"
    features_path = output_dir / "hard_neg_features.npy"
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    if not samples:
        print(f"\nNo hard negatives found at threshold={args.score_threshold}.")
        print("Try lowering --score-threshold.")
        return 0

    rows: list[dict[str, object]] = []
    for index, sample in enumerate(samples):
        crop_path = crops_dir / f"{index:05d}_{safe_stem(sample.source_image.stem)}.png"
        write_image(crop_path, sample.crop)
        x, y, w, h = sample.bbox
        rows.append(
            {
                "candidate_path": str(crop_path),
                "source_image": str(sample.source_image),
                "score": round(sample.score, 6),
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "label": 0,
            }
        )

    with labels_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["candidate_path", "source_image", "score", "x", "y", "w", "h", "label"],
        )
        writer.writeheader()
        writer.writerows(rows)

    np.save(str(features_path), np.vstack([sample.feature for sample in samples]).astype(np.float32))

    print(f"\nTotal hard negatives: {len(samples)}")
    print(f"Crops      : {crops_dir}")
    print(f"Labels CSV : {labels_path}")
    print(f"Features   : {features_path}")
    print("\nRetrain example:")
    print(f"  python train_plate_classifier.py --labels <base_labels.csv> --hard-negatives {labels_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
