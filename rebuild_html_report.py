from __future__ import annotations

import argparse
import csv
from pathlib import Path

from detect_plates import write_html_report


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild the HTML review page from existing CSV outputs.")
    parser.add_argument("--report", default="outputs/detections.csv", help="Existing detections CSV path.")
    parser.add_argument("--characters", default="outputs/characters.csv", help="Existing characters CSV path.")
    parser.add_argument("--html-report", default="outputs/report.html", help="HTML report path to write.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rows = read_csv(Path(args.report))
    character_rows = read_csv(Path(args.characters))
    write_html_report(Path(args.html_report), rows, character_rows)
    print(f"HTML report rebuilt: {args.html_report}")
