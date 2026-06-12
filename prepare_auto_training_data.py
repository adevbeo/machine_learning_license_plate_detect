from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from plate_hog_svm.io_utils import clamp_bbox, iter_images, read_image, write_image


FIELDNAMES = ["image", "x", "y", "w", "h", "label"]


@dataclass(frozen=True)
class PlateProposal:
    bbox: tuple[int, int, int, int]
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automatically build a weakly-labeled training CSV from vehicle images. "
            "It uses OpenCV plate proposals to create positive crops and samples "
            "non-overlapping windows as negative crops. The generated CSV can train "
            "the HOG + LinearSVM model."
        ),
    )
    parser.add_argument("--input", default="images/train", help="Folder or image file used to build training data.")
    parser.add_argument("--output", default="outputs/labels.csv", help="Output labels CSV.")
    parser.add_argument("--preview-output", default="", help="Optional folder for proposal preview images. Empty string disables.")
    parser.add_argument("--limit", type=int, default=0, help="Limit images for quick experiments. 0 = all.")
    parser.add_argument("--sample-size", type=int, default=0, help="Randomly sample this many images. 0 = all.")
    parser.add_argument("--max-width", type=int, default=700, help="Resize width used only for proposal search.")
    parser.add_argument("--negatives-per-image", type=int, default=3, help="Number of negative windows per positive.")
    parser.add_argument("--min-score", type=float, default=0.25, help="Minimum proposal score to keep a positive.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for negative sampling.")
    parser.add_argument("--quiet", action="store_true", help="Print only the final summary.")
    return parser.parse_args()


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve())).replace("\\", "/")
    except ValueError:
        return str(path)


def resize_for_search(image: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    if max_width <= 0 or w <= max_width:
        return image.copy(), 1.0
    scale = max_width / float(w)
    resized = cv2.resize(image, (max_width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    return resized, scale


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    min_value = float(np.min(image))
    max_value = float(np.max(image))
    if max_value <= min_value:
        return np.zeros_like(image, dtype=np.uint8)
    normalized = (image - min_value) * (255.0 / (max_value - min_value))
    return normalized.astype(np.uint8)


def build_candidate_masks(gray: np.ndarray) -> list[np.ndarray]:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 7, 35, 35)

    masks: list[np.ndarray] = []
    for rect_kernel in ((17, 5), (25, 7), (35, 9)):
        blackhat = cv2.morphologyEx(
            gray,
            cv2.MORPH_BLACKHAT,
            cv2.getStructuringElement(cv2.MORPH_RECT, rect_kernel),
        )
        grad_x = cv2.Sobel(blackhat, cv2.CV_32F, 1, 0, ksize=3)
        grad_x = normalize_to_uint8(np.abs(grad_x))
        grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
        closed = cv2.morphologyEx(
            grad_x,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, rect_kernel),
        )
        _, thresh = cv2.threshold(closed, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        thresh = cv2.morphologyEx(
            thresh,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
            iterations=1,
        )
        thresh = cv2.erode(thresh, None, iterations=1)
        thresh = cv2.dilate(thresh, None, iterations=2)
        masks.append(thresh)

    edges = cv2.Canny(gray, 70, 180)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5)),
        iterations=1,
    )
    masks.append(edges)
    return masks


def score_candidate(
    gray: np.ndarray,
    edges: np.ndarray,
    bbox: tuple[int, int, int, int],
    contour_area: float,
) -> float:
    image_h, image_w = gray.shape[:2]
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return 0.0

    ratio = w / float(h)
    if ratio < 1.8 or ratio > 6.5:
        return 0.0
    if w < image_w * 0.08 or h < image_h * 0.035:
        return 0.0
    if w > image_w * 0.85 or h > image_h * 0.35:
        return 0.0

    crop = gray[y : y + h, x : x + w]
    crop_edges = edges[y : y + h, x : x + w]
    if crop.size == 0:
        return 0.0

    area = w * h
    area_ratio = area / float(image_w * image_h)
    if area_ratio < 0.003 or area_ratio > 0.18:
        return 0.0

    aspect_score = math.exp(-abs(ratio - 3.8) / 1.6)
    rectangularity = max(0.0, min(1.0, contour_area / float(area)))
    edge_density = float(np.mean(crop_edges > 0))
    contrast = min(1.0, float(np.std(crop)) / 65.0)
    center_x = (x + w / 2.0) / image_w
    center_y = (y + h / 2.0) / image_h
    horizontal_score = math.exp(-abs(center_x - 0.5) / 0.45)
    vertical_score = 0.85 + 0.15 * math.exp(-abs(center_y - 0.58) / 0.35)
    size_score = math.exp(-abs(math.log(max(area_ratio, 1e-6) / 0.035)) / 1.4)

    return (
        0.28 * aspect_score
        + 0.18 * rectangularity
        + 0.18 * min(1.0, edge_density / 0.18)
        + 0.14 * contrast
        + 0.12 * size_score
        + 0.06 * horizontal_score
        + 0.04 * vertical_score
    )


def merge_duplicate_proposals(proposals: list[PlateProposal]) -> list[PlateProposal]:
    merged: list[PlateProposal] = []
    for proposal in sorted(proposals, key=lambda item: item.score, reverse=True):
        if all(iou(proposal.bbox, kept.bbox) < 0.55 for kept in merged):
            merged.append(proposal)
        if len(merged) >= 10:
            break
    return merged


def find_plate_proposals(image: np.ndarray, max_width: int = 700) -> list[PlateProposal]:
    resized, scale = resize_for_search(image, max_width)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 180)
    image_h, image_w = gray.shape[:2]
    proposals: list[PlateProposal] = []

    for mask in build_candidate_masks(gray):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            pad_x = max(2, int(round(w * 0.05)))
            pad_y = max(2, int(round(h * 0.12)))
            x, y, w, h = clamp_bbox(
                x - pad_x,
                y - pad_y,
                w + 2 * pad_x,
                h + 2 * pad_y,
                image_w,
                image_h,
            )
            score = score_candidate(gray, edges, (x, y, w, h), cv2.contourArea(contour))
            if score <= 0:
                continue

            if scale != 1.0:
                inv = 1.0 / scale
                ox = int(round(x * inv))
                oy = int(round(y * inv))
                ow = int(round(w * inv))
                oh = int(round(h * inv))
            else:
                ox, oy, ow, oh = x, y, w, h
            ox, oy, ow, oh = clamp_bbox(ox, oy, ow, oh, image.shape[1], image.shape[0])
            proposals.append(PlateProposal((ox, oy, ow, oh), score))

    return merge_duplicate_proposals(proposals)


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / float(max(1, union))


def sample_negative_boxes(
    image_shape: tuple[int, int],
    positive_box: tuple[int, int, int, int],
    count: int,
    rng: random.Random,
) -> list[tuple[int, int, int, int]]:
    image_h, image_w = image_shape
    _, _, pos_w, pos_h = positive_box
    boxes: list[tuple[int, int, int, int]] = []
    aspect_ratios = (2.4, 2.8, 3.6, 4.5)

    attempts = 0
    while len(boxes) < count and attempts < count * 80:
        attempts += 1
        height_scale = rng.uniform(0.75, 1.6)
        h = int(round(pos_h * height_scale))
        h = max(18, min(h, max(18, int(image_h * 0.28))))
        ratio = rng.choice(aspect_ratios)
        w = int(round(h * ratio))
        w = max(32, min(w, max(32, int(image_w * 0.8))))
        if w >= image_w or h >= image_h:
            continue

        x = rng.randint(0, image_w - w)
        y = rng.randint(0, image_h - h)
        box = (x, y, w, h)
        if iou(box, positive_box) > 0.08:
            continue
        if any(iou(box, other) > 0.45 for other in boxes):
            continue
        boxes.append(box)

    return boxes


def draw_preview(
    image: np.ndarray,
    positive: PlateProposal,
    negatives: list[tuple[int, int, int, int]],
) -> np.ndarray:
    preview = image.copy()
    x, y, w, h = positive.bbox
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        preview,
        f"plate {positive.score:.2f}",
        (x, max(16, y - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    for i, (nx, ny, nw, nh) in enumerate(negatives, start=1):
        cv2.rectangle(preview, (nx, ny), (nx + nw, ny + nh), (0, 0, 255), 1)
        cv2.putText(
            preview,
            f"neg {i}",
            (nx, max(16, ny - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return preview


def main() -> int:
    args = parse_args()
    paths = iter_images(Path(args.input))
    rng = random.Random(args.seed)
    if args.sample_size:
        paths = rng.sample(paths, min(args.sample_size, len(paths)))
    elif args.limit:
        paths = paths[: args.limit]
    if not paths:
        print(f"No images found: {args.input}", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    preview_dir = Path(args.preview_output) if args.preview_output else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    misses: list[Path] = []

    for index, image_path in enumerate(paths, start=1):
        try:
            image = read_image(image_path)
        except Exception as exc:
            print(f"[SKIP] {image_path}: {exc}", file=sys.stderr)
            continue

        proposals = find_plate_proposals(image, max_width=args.max_width)
        if not proposals or proposals[0].score < args.min_score:
            misses.append(image_path)
            if not args.quiet:
                print(f"[MISS] {index}/{len(paths)} {image_path.name}: no strong plate proposal")
            continue

        positive = proposals[0]
        negatives = sample_negative_boxes(
            image.shape[:2],
            positive.bbox,
            args.negatives_per_image,
            rng,
        )

        image_value = relative_path(image_path)
        x, y, w, h = positive.bbox
        rows.append({"image": image_value, "x": x, "y": y, "w": w, "h": h, "label": 1})
        for nx, ny, nw, nh in negatives:
            rows.append({"image": image_value, "x": nx, "y": ny, "w": nw, "h": nh, "label": 0})

        if preview_dir:
            preview_path = preview_dir / f"{image_path.stem}_auto_labels.png"
            write_image(preview_path, draw_preview(image, positive, negatives))

        if not args.quiet:
            print(
                f"[OK  ] {index}/{len(paths)} {image_path.name}: "
                f"plate score={positive.score:.3f}, negatives={len(negatives)}"
            )

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    positives = sum(1 for row in rows if row["label"] == 1)
    negatives = sum(1 for row in rows if row["label"] == 0)
    print("\nDone.")
    print(f"Images scanned : {len(paths)}")
    print(f"Positive boxes : {positives}")
    print(f"Negative boxes : {negatives}")
    print(f"Missed images  : {len(misses)}")
    print(f"Labels CSV     : {output_path}")
    if preview_dir:
        print(f"Preview labels : {preview_dir}")
    if positives == 0 or negatives == 0:
        print("Not enough generated labels to train. Try lowering --min-score.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
