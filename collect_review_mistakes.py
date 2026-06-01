from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path


COPY_COLUMNS = (
    ("image", "originals", "original_copy"),
    ("plate_path", "plates", "plate_copy"),
    ("debug_path", "debug", "debug_copy"),
)


def resolve_input_path(raw_value: str, base_dir: Path) -> Path | None:
    raw_value = raw_value.strip()
    if not raw_value:
        return None

    path = Path(raw_value)
    candidates = [path] if path.is_absolute() else [base_dir / path, Path.cwd() / path, path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def unique_copy_path(output_dir: Path, index: int, kind: str, source: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{index:05d}_{kind}_{source.stem}"
    suffix = source.suffix or ".png"
    candidate = output_dir / f"{stem}{suffix}"
    counter = 2
    while candidate.exists():
        candidate = output_dir / f"{stem}_{counter}{suffix}"
        counter += 1
    return candidate


def is_wrong_row(row: dict[str, str]) -> bool:
    status = (row.get("review_status") or row.get("status") or "").strip().lower()
    return status == "wrong" if status else True


def collect_mistakes(review_csv: Path, output_dir: Path, base_dir: Path, include_all: bool) -> int:
    if not review_csv.exists():
        print(f"Review CSV not found: {review_csv}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    copied_rows: list[dict[str, str]] = []
    copy_count = 0

    with review_csv.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for index, row in enumerate(reader, start=1):
            if not include_all and not is_wrong_row(row):
                continue

            output_row = dict(row)
            missing_files: list[str] = []
            for source_column, subdir, output_column in COPY_COLUMNS:
                source_path = resolve_input_path(row.get(source_column, ""), base_dir)
                if source_path is None:
                    output_row[output_column] = ""
                    continue
                if not source_path.exists():
                    output_row[output_column] = ""
                    missing_files.append(f"{source_column}:{source_path}")
                    continue

                destination = unique_copy_path(output_dir / subdir, index, source_column, source_path)
                shutil.copy2(source_path, destination)
                output_row[output_column] = str(destination)
                copy_count += 1

            output_row["missing_files"] = "; ".join(missing_files)
            copied_rows.append(output_row)

    output_columns = fieldnames[:]
    for _, _, output_column in COPY_COLUMNS:
        if output_column not in output_columns:
            output_columns.append(output_column)
    if "missing_files" not in output_columns:
        output_columns.append("missing_files")

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(copied_rows)

    print(f"Rows collected: {len(copied_rows)}")
    print(f"Files copied: {copy_count}")
    print(f"Output: {output_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy manually marked wrong plate-review rows into one folder for correction.",
    )
    parser.add_argument("--review", required=True, help="CSV exported from report.html, usually review_wrong.csv.")
    parser.add_argument("--output", default="outputs/review_wrong", help="Folder to collect wrong originals/crops/debug images.")
    parser.add_argument("--base-dir", default=".", help="Base directory for resolving relative paths in the review CSV.")
    parser.add_argument("--include-all", action="store_true", help="Copy all rows instead of only review_status=wrong rows.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(
        collect_mistakes(
            review_csv=Path(args.review),
            output_dir=Path(args.output),
            base_dir=Path(args.base_dir).resolve(),
            include_all=args.include_all,
        )
    )
