from __future__ import annotations

import csv
import html
import os
from pathlib import Path


DETECTION_FIELDNAMES = [
    "image",
    "found",
    "score",
    "confidence",
    "x",
    "y",
    "w",
    "h",
    "windows_scanned",
    "plate_text",
    "compact_text",
    "ocr_confidence",
    "char_count",
    "plate_path",
    "annotated_path",
    "error",
]


def write_detection_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DETECTION_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def html_text(value: object) -> str:
    return html.escape(str(value or ""))


def html_attr(value: object) -> str:
    return html.escape(str(value or ""), quote=True)


def image_href(path_value: object, report_path: Path) -> str:
    raw = str(path_value or "").strip()
    if not raw:
        return ""
    path = Path(raw)
    try:
        href = os.path.relpath(path, report_path.parent)
    except ValueError:
        href = str(path)
    return Path(href).as_posix()


def render_image(label: str, path_value: object, report_path: Path) -> str:
    href = image_href(path_value, report_path)
    if not href:
        return f'<figure class="empty"><figcaption>{html_text(label)}</figcaption></figure>'
    return (
        f'<figure><img src="{html_attr(href)}" alt="{html_attr(label)}" loading="lazy">'
        f"<figcaption>{html_text(label)}</figcaption></figure>"
    )


def write_html_report(
    report_path: Path,
    rows: list[dict[str, object]],
    character_rows: list[dict[str, object]] | None = None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    total = len(rows)
    found = sum(1 for row in rows if str(row.get("found", "")).strip() in {"1", "true", "True"})
    cards: list[str] = []

    for index, row in enumerate(rows, start=1):
        bbox = ", ".join(str(row.get(name) or "") for name in ("x", "y", "w", "h")).strip(", ")
        status = "FOUND" if str(row.get("found", "")).strip() in {"1", "true", "True"} else "MISS"
        cards.append(
            f"""
            <article class="card">
              <header>
                <strong>{index}. {html_text(status)}</strong>
                <span>score {html_text(row.get("score"))} | confidence {html_text(row.get("confidence"))}</span>
              </header>
              <p>{html_text(row.get("image"))}</p>
              <p>BBox: {html_text(bbox)} | Windows: {html_text(row.get("windows_scanned"))}</p>
              <div class="media">
                {render_image("Original", row.get("image"), report_path)}
                {render_image("LinearSVM bbox", row.get("annotated_path"), report_path)}
                {render_image("Plate crop", row.get("plate_path"), report_path)}
              </div>
              <div class="review">
                <label><input type="radio" name="row-{index}" value="correct"> Correct</label>
                <label><input type="radio" name="row-{index}" value="wrong"> Wrong</label>
                <input type="text" placeholder="Actual plate when wrong">
              </div>
            </article>
            """
        )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HOG + LinearSVM Plate Detection Report</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f6f7f9; color: #18202a; }}
    header.top {{ position: sticky; top: 0; background: #ffffff; border-bottom: 1px solid #d9dee7; padding: 16px 24px; z-index: 1; }}
    h1 {{ margin: 0 0 6px; font-size: 24px; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 18px; display: grid; gap: 16px; }}
    .card {{ background: #ffffff; border: 1px solid #d9dee7; border-radius: 6px; padding: 14px; }}
    .card header {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; }}
    .card p {{ color: #596474; margin: 8px 0; overflow-wrap: anywhere; }}
    .media {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
    figure {{ margin: 0; border: 1px solid #d9dee7; border-radius: 6px; overflow: hidden; background: #ffffff; }}
    figure.empty {{ min-height: 180px; display: grid; place-items: center; }}
    img {{ width: 100%; height: 260px; object-fit: contain; display: block; background: #101820; }}
    figcaption {{ border-top: 1px solid #d9dee7; color: #596474; font-size: 13px; padding: 8px; }}
    .review {{ display: flex; gap: 12px; flex-wrap: wrap; align-items: center; margin-top: 12px; }}
    input[type="text"] {{ min-width: 240px; padding: 7px 8px; border: 1px solid #bcc4d1; border-radius: 4px; }}
    @media (max-width: 900px) {{ .media {{ grid-template-columns: 1fr; }} img {{ height: auto; max-height: 320px; }} }}
  </style>
</head>
<body>
  <header class="top">
    <h1>HOG + LinearSVM Plate Detection</h1>
    <div>Total: <b>{total}</b> | Found: <b>{found}</b> | Miss: <b>{total - found}</b></div>
  </header>
  <main>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    report_path.write_text(html_doc, encoding="utf-8")
