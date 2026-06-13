from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from plate_hog_svm.config import DetectionConfig, NMSConfig
from plate_hog_svm.detector import Detection, SlidingWindowDetector, draw_detection, draw_detections
from plate_hog_svm.io_utils import iter_images, read_image, unique_output_path, write_image
from plate_hog_svm.report import write_detection_csv, write_html_report


def parse_float_tuple(value: str, default: tuple[float, ...]) -> tuple[float, ...]:
    if not value.strip():
        return default
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def parse_int_tuple(value: str, default: tuple[int, ...]) -> tuple[int, ...]:
    if not value.strip():
        return default
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def row_for_miss(image_path: Path, error: str = "") -> dict[str, object]:
    return {
        "image": str(image_path),
        "found": 0,
        "score": "",
        "confidence": "",
        "x": "",
        "y": "",
        "w": "",
        "h": "",
        "windows_scanned": "",
        "plate_text": "",
        "compact_text": "",
        "ocr_confidence": "",
        "char_count": "",
        "plate_path": "",
        "annotated_path": "",
        "error": error,
    }


def row_for_detection(
    image_path: Path,
    detection: Detection,
    plate_path: str,
    annotated_path: str,
) -> dict[str, object]:
    return {
        "image": str(image_path),
        "found": 1,
        "score": round(detection.score, 4),
        "confidence": round(detection.confidence, 4),
        "x": detection.bbox[0],
        "y": detection.bbox[1],
        "w": detection.bbox[2],
        "h": detection.bbox[3],
        "windows_scanned": detection.windows_scanned,
        "plate_text": "",
        "compact_text": "",
        "ocr_confidence": "",
        "char_count": "",
        "plate_path": plate_path,
        "annotated_path": annotated_path,
        "error": "",
    }


def flat_output_path(base_dir: Path, image_path: Path, suffix: str, extension: str = ".png") -> Path:
    return base_dir / f"{image_path.stem}{suffix}{extension}"


def process_images(args: argparse.Namespace) -> int:
    images = iter_images(Path(args.input))
    if args.sample_size:
        rng = random.Random(args.seed)
        images = rng.sample(images, min(args.sample_size, len(images)))
    elif args.limit:
        images = images[: args.limit]
    if not images:
        print(f"No images found in {args.input}", file=sys.stderr)
        return 1

    try:
        detector = SlidingWindowDetector.from_model_file(
            Path(args.model),
            Path(args.metadata) if args.metadata else None,
        )
    except Exception as exc:
        print(f"Cannot load model: {exc}", file=sys.stderr)
        return 1

    default_config = DetectionConfig()
    nms_config = NMSConfig(
        iou_threshold=args.iou_threshold,
        top_k=args.top_k,
    )
    config = DetectionConfig(
        max_width=args.max_width,
        aspect_ratios=parse_float_tuple(args.aspect_ratios, default_config.aspect_ratios),
        window_heights=parse_int_tuple(args.window_heights, default_config.window_heights),
        stride_ratio=args.stride_ratio,
        score_threshold=args.score_threshold,
        batch_size=args.batch_size,
        nms=nms_config,
    )

    plates_dir = Path(args.plates_output)
    output_dir = Path(args.output)
    rows: list[dict[str, object]] = []
    found_count = 0

    for image_path in images:
        try:
            image = read_image(image_path)
            detections = detector.detect(image, config)
        except Exception as exc:
            rows.append(row_for_miss(image_path, str(exc)))
            print(f"[ERROR] {image_path}: {exc}", file=sys.stderr)
            continue

        if not detections:
            rows.append(row_for_miss(image_path))
            if not args.quiet:
                print(f"[MISS ] {image_path}")
            continue

        found_count += 1
        best = detections[0]

        if args.flat_output:
            plate_path_obj = flat_output_path(plates_dir, image_path, "_plate")
        else:
            plate_path_obj = unique_output_path(plates_dir, image_path, "_plate")
        write_image(plate_path_obj, best.crop)
        plate_path_str = str(plate_path_obj)

        if args.top_k > 1:
            annotated = draw_detections(image, detections, args.bbox_label)
        else:
            annotated = draw_detection(image, best, args.bbox_label)
        if args.flat_output:
            annotated_path_obj = flat_output_path(output_dir, image_path, args.annotated_suffix)
        else:
            annotated_path_obj = unique_output_path(output_dir, image_path, "_linearsvm")
        write_image(annotated_path_obj, annotated)
        annotated_path_str = str(annotated_path_obj)

        rows.append(row_for_detection(image_path, best, plate_path_str, annotated_path_str))

        if not args.quiet:
            det_info = f"score={best.score:.3f} conf={best.confidence:.3f}"
            if len(detections) > 1:
                det_info += f" (+{len(detections) - 1} more)"
            print(f"[FOUND] {image_path}  {det_info}")

    write_detection_csv(Path(args.report), rows)
    print(f"\nDone. Found plates in {found_count}/{len(images)} images.")
    print(f"Plates      : {plates_dir}")
    print(f"Output      : {output_dir}")
    print(f"Report CSV  : {args.report}")
    if args.html_report:
        write_html_report(Path(args.html_report), rows, [])
        print(f"HTML report : {args.html_report}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Class 2: Multi-scale sliding window scan, HOG feature extraction, "
            "LinearSVM scoring, NMS, and best plate window selection."
        ),
    )
    parser.add_argument("--input", default="images", help="Image file or directory to process.")
    parser.add_argument("--model", default="models/plate_svm.joblib", help="Sklearn Pipeline (.joblib).")
    parser.add_argument("--metadata", default="", help="Metadata JSON path. Defaults to model path + .json.")
    parser.add_argument("--plates-output", default="outputs/plates", help="Directory for cropped plate windows.")
    parser.add_argument("--output", default="outputs/output", help="Directory for images annotated with LinearSVM bbox.")
    parser.add_argument("--bbox-label", default="LinearSVM", help="Text label drawn above the detection bbox.")
    parser.add_argument("--annotated-suffix", default="_linearsvm", help="Filename suffix for annotated output images.")
    parser.add_argument("--report", default="outputs/detections.csv", help="CSV report path.")
    parser.add_argument("--html-report", default="outputs/report.html", help="HTML report path.")
    parser.add_argument("--max-width", type=int, default=900, help="Resize images to this width before scanning.")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Minimum LinearSVM decision score to accept a window as plate candidate.",
    )
    parser.add_argument("--stride-ratio", type=float, default=0.25, help="Sliding window step as fraction of window size.")
    parser.add_argument(
        "--aspect-ratios",
        default="",
        help="Comma-separated window aspect ratios (width/height). Default: 2.4,2.8,3.6,4.5",
    )
    parser.add_argument(
        "--window-heights",
        default="",
        help="Comma-separated window heights in pixels after max-width resize.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Windows scored per LinearSVM batch.")
    # NMS params
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.3,
        help="IoU threshold for NMS to suppress overlapping boxes (0-1, default 0.3).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        help="Max detections returned per image (default 1). Use >1 for multi-plate images.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images processed (for quick testing).")
    parser.add_argument("--sample-size", type=int, default=0, help="Randomly sample this many images. 0 = disabled.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --sample-size.")
    parser.add_argument("--flat-output", action="store_true", help="Save output images directly under output folders.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-image status output.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(process_images(parse_args()))
