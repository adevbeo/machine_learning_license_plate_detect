from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import IMAGE_EXTENSIONS


def read_image(path: Path | str) -> np.ndarray:
    image_path = Path(path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is not None:
        return image

    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    return image


def write_image(path: Path | str, image: np.ndarray) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if cv2.imwrite(str(output_path), image):
        return

    ok, encoded = cv2.imencode(output_path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"Cannot encode image: {output_path}")
    encoded.tofile(str(output_path))


def iter_images(input_path: Path | str) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_EXTENSIONS else []
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def resize_for_detection(image: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    if max_width <= 0 or width <= max_width:
        return image.copy(), 1.0

    scale = max_width / float(width)
    resized = cv2.resize(image, (max_width, max(1, int(round(height * scale)))), interpolation=cv2.INTER_AREA)
    return resized, scale


def clamp_bbox(
    x: int,
    y: int,
    w: int,
    h: int,
    image_w: int,
    image_h: int,
) -> tuple[int, int, int, int]:
    x = max(0, min(int(x), max(0, image_w - 1)))
    y = max(0, min(int(y), max(0, image_h - 1)))
    w = max(1, min(int(w), max(1, image_w - x)))
    h = max(1, min(int(h), max(1, image_h - y)))
    return x, y, w, h


def unique_output_path(base_dir: Path, image_path: Path, suffix: str, extension: str = ".png") -> Path:
    stem_path = image_path.with_suffix("")
    parts = list(stem_path.parts)
    if stem_path.is_absolute():
        parts = [part.replace(":", "") for part in parts if part != stem_path.anchor]
    return base_dir.joinpath(*parts).with_name(f"{image_path.stem}{suffix}{extension}")
