from __future__ import annotations

import argparse
import sys
from pathlib import Path

from plate_hog_svm.config import DetectionConfig
from plate_hog_svm.detector import Detection, SlidingWindowDetector, draw_detection
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


def detect_plate(
    image,
    max_width: int = 900,
    model_path: str | Path = "models/plate_svm.yml",
    score_threshold: float = 0.0,
) -> Detection | None:
    detector = SlidingWindowDetector.from_model_file(Path(model_path))
    config = DetectionConfig(max_width=max_width, score_threshold=score_threshold)
    return detector.detect(image, config)


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
        "debug_path": "",
        "error": error,
    }


def process_images(args: argparse.Namespace) -> int:
    images = iter_images(Path(args.input))
    if args.limit:
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
        print(f"Cannot load SVM model: {exc}", file=sys.stderr)
        return 1

    default_config = DetectionConfig()
    config = DetectionConfig(
        max_width=args.max_width,
        aspect_ratios=parse_float_tuple(args.aspect_ratios, default_config.aspect_ratios),
        window_heights=parse_int_tuple(args.window_heights, default_config.window_heights),
        stride_ratio=args.stride_ratio,
        score_threshold=args.score_threshold,
        batch_size=args.batch_size,
    )

    output_dir = Path(args.output)
    debug_dir = Path(args.debug_output) if args.debug_output else None
    rows: list[dict[str, object]] = []
    found = 0

    for image_path in images:
        try:
            image = read_image(image_path)
            detection = detector.detect(image, config)
        except Exception as exc:
            rows.append(row_for_miss(image_path, str(exc)))
            print(f"[ERROR] {image_path}: {exc}", file=sys.stderr)
            continue

        plate_path = ""
        debug_path = ""
        bbox: tuple[int | str, int | str, int | str, int | str] = ("", "", "", "")
        score: float | str = ""
        confidence: float | str = ""
        windows_scanned: int | str = ""

        if detection is not None:
            found += 1
            bbox = detection.bbox
            score = round(detection.score, 4)
            confidence = round(detection.confidence, 4)
            windows_scanned = detection.windows_scanned
            plate_path_obj = unique_output_path(output_dir, image_path, "_plate")
            write_image(plate_path_obj, detection.crop)
            plate_path = str(plate_path_obj)

        if debug_dir:
            debug = draw_detection(image, detection)
            debug_path_obj = unique_output_path(debug_dir, image_path, "_debug")
            write_image(debug_path_obj, debug)
            debug_path = str(debug_path_obj)

        rows.append(
            {
                "image": str(image_path),
                "found": 1 if detection else 0,
                "score": score,
                "confidence": confidence,
                "x": bbox[0],
                "y": bbox[1],
                "w": bbox[2],
                "h": bbox[3],
                "windows_scanned": windows_scanned,
                "plate_text": "",
                "compact_text": "",
                "ocr_confidence": "",
                "char_count": "",
                "plate_path": plate_path,
                "debug_path": debug_path,
                "error": "",
            }
        )

        if not args.quiet:
            status = "FOUND" if detection else "MISS"
            print(f"[{status}] {image_path}")

    write_detection_csv(Path(args.report), rows)
    print(f"Done. Found {found}/{len(images)} plates.")
    print(f"Plate crops: {output_dir}")
    print(f"Report: {args.report}")
    if debug_dir:
        print(f"Debug images: {debug_dir}")
    if args.html_report:
        write_html_report(Path(args.html_report), rows, [])
        print(f"HTML report: {args.html_report}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Class 2: scan each image with sliding windows, extract HOG, and choose the highest-confidence SVM plate window.",
    )
    parser.add_argument("--input", default="images", help="Image file or directory to process.")
    parser.add_argument("--model", default="models/plate_svm.yml", help="Trained OpenCV SVM model path.")
    parser.add_argument("--metadata", default="", help="Optional model metadata JSON path.")
    parser.add_argument("--output", default="outputs/plates", help="Directory for cropped plate windows.")
    parser.add_argument("--debug-output", default="outputs/debug", help="Directory for annotated images. Empty disables debug images.")
    parser.add_argument("--report", default="outputs/detections.csv", help="CSV report path.")
    parser.add_argument("--html-report", default="outputs/report.html", help="HTML report path. Empty disables HTML export.")
    parser.add_argument("--max-width", type=int, default=900, help="Resize large images to this width before scanning.")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="Minimum positive SVM margin to accept a plate.")
    parser.add_argument("--stride-ratio", type=float, default=0.25, help="Sliding window step as a fraction of window size.")
    parser.add_argument(
        "--aspect-ratios",
        default="",
        help="Comma-separated window aspect ratios. Default focuses on horizontal plates.",
    )
    parser.add_argument(
        "--window-heights",
        default="",
        help="Comma-separated window heights in pixels after max-width resize.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Number of windows scored per SVM batch.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for quick testing.")
    parser.add_argument("--quiet", action="store_true", help="Do not print one status line per image.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(process_images(parse_args()))
