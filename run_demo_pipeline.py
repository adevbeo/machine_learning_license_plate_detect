from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the full ML demo pipeline: auto-generate weak labels, train HOG+LinearSVM, "
            "then detect/crop plates from validation images."
        ),
    )
    parser.add_argument("--train-input", default="images/train", help="Training images.")
    parser.add_argument("--detect-input", default="images/val", help="Images used for detection demo.")
    parser.add_argument("--labels", default="outputs/labels.csv", help="Generated labels CSV.")
    parser.add_argument("--model", default="models/plate_svm.joblib", help="Trained model path.")
    parser.add_argument("--plates-output", default="outputs/plates", help="Detected plate crop folder.")
    parser.add_argument("--annotated-output", default="outputs/output", help="Images annotated with LinearSVM bbox.")
    parser.add_argument("--report", default="outputs/detections.csv", help="Detection CSV report.")
    parser.add_argument("--html-report", default="outputs/report.html", help="Detection HTML report.")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Legacy shortcut: use the same random image count for train and detect.",
    )
    parser.add_argument("--train-sample-size", type=int, default=100, help="Random image count used to build ML training labels.")
    parser.add_argument("--detect-sample-size", type=int, default=20, help="Random image count used for demo detection output.")
    parser.add_argument("--detect-limit", type=int, default=0, help="First-N validation images. 0 = use random sample.")
    parser.add_argument("--label-limit", type=int, default=0, help="Limit training images for quick tests. 0 = all.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for image sampling. Default = new random sample each run.")
    parser.add_argument("--keep-output", action="store_true", help="Keep old files in output folders.")
    parser.add_argument("--skip-labels", action="store_true", help="Skip auto label generation.")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training.")
    parser.add_argument("--score-threshold", type=float, default=0.0, help="LinearSVM detection threshold.")
    parser.add_argument("--no-visualize", action="store_true", help="Skip HOG visualization and PCA scatter images.")
    parser.add_argument("--verbose", action="store_true", help="Show per-image progress from helper scripts.")
    return parser.parse_args()


def print_compact_output(output: str) -> None:
    prefixes = (
        "Images scanned",
        "Positive boxes",
        "Negative boxes",
        "Missed images",
        "Labels CSV",
        "Best params",
        "Accuracy",
        "Precision",
        "Recall",
        "F1        ",
        "F1 macro",
        "Model",
        "Metadata",
        "Done. Found plates",
        "Plates",
        "Output",
        "Report CSV",
        "HTML report",
    )
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if line.startswith(prefixes):
            print(f"  {line}")


def remove_path(path: str | Path) -> None:
    target = Path(path)
    if not target.exists():
        return
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()


def clean_demo_outputs(args: argparse.Namespace) -> None:
    if args.keep_output:
        return

    remove_path(args.plates_output)
    remove_path(args.annotated_output)
    remove_path(args.report)
    remove_path(args.html_report)

    demo_root = Path(args.report).parent
    remove_path(demo_root / "visualize")
    remove_path(demo_root / "demo_20")
    remove_path(demo_root / "diagnostics")
    remove_path(demo_root / "debug")
    if not args.skip_labels:
        remove_path(args.labels)
        remove_path(demo_root / "label_debug")
        remove_path(demo_root / "label_preview")


def run_step(title: str, command: list[str], verbose: bool = False) -> None:
    print(f"\n{title}", flush=True)
    if verbose:
        print(" ".join(command), flush=True)
        completed = subprocess.run(command, check=False)
    else:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print_compact_output(completed.stdout or "")
    if completed.returncode != 0:
        if not verbose and completed.stdout:
            print(completed.stdout)
        raise SystemExit(completed.returncode)


def main() -> int:
    args = parse_args()
    python = sys.executable
    seed = args.seed if args.seed is not None else random.SystemRandom().randint(1, 2_147_483_647)
    train_sample_size = args.sample_size or args.train_sample_size
    detect_sample_size = args.sample_size or args.detect_sample_size
    visualize_enabled = not args.no_visualize

    clean_demo_outputs(args)
    print(f"Random seed: {seed}")
    print(f"Train sample: {train_sample_size} images | Detect sample: {detect_sample_size} images")

    if not args.skip_labels:
        label_command = [
            python,
            "prepare_auto_training_data.py",
            "--input",
            args.train_input,
            "--output",
            args.labels,
            "--preview-output",
            "",
            "--seed",
            str(seed),
        ]
        if train_sample_size:
            label_command.extend(["--sample-size", str(train_sample_size)])
        elif args.label_limit:
            label_command.extend(["--limit", str(args.label_limit)])
        if not args.verbose:
            label_command.append("--quiet")
        run_step("[1/3] Auto-generate weak training labels", label_command, args.verbose)

    if not args.skip_train:
        train_command = [
            python,
            "train_plate_classifier.py",
            "--labels",
            args.labels,
            "--model",
            args.model,
            "--feature-plot",
            "outputs/visualize/hog_feature_embedding.png" if visualize_enabled else "",
            "--hog-visualization",
            "outputs/visualize/hog_visualization.png" if visualize_enabled else "",
            "--max-iter",
            "50000",
        ]
        run_step("[2/3] Train HOG + LinearSVM model", train_command, args.verbose)

    detect_command = [
        python,
        "detect_plates.py",
        "--input",
        args.detect_input,
        "--model",
        args.model,
        "--output",
        args.annotated_output,
        "--plates-output",
        args.plates_output,
        "--score-threshold",
        str(args.score_threshold),
        "--report",
        args.report,
        "--html-report",
        args.html_report,
        "--seed",
        str(seed),
        "--flat-output",
    ]
    if detect_sample_size and not args.detect_limit:
        detect_command.extend(["--sample-size", str(detect_sample_size)])
    elif args.detect_limit:
        detect_command.extend(["--limit", str(args.detect_limit)])
    if not args.verbose:
        detect_command.append("--quiet")
    run_step("[3/3] Detect and crop plates", detect_command, args.verbose)

    print("\nOutputs:")
    print(f"  Labels CSV : {Path(args.labels)}")
    print(f"  Model      : {Path(args.model)}")
    print(f"  Plates     : {Path(args.plates_output)}")
    print(f"  Output     : {Path(args.annotated_output)}")
    print(f"  CSV report : {Path(args.report)}")
    print(f"  HTML report: {Path(args.html_report)}")
    if visualize_enabled:
        print("  HOG visual : outputs/visualize/hog_visualization.png")
        print("  PCA plot   : outputs/visualize/hog_feature_embedding.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
