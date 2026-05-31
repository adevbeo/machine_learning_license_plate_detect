from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from plate_ocr import (
    PlateOCRResult,
    load_char_model,
    plate_layout_score,
    recognize_plate,
    recognize_plate_easyocr_text,
    segment_plate_characters,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
HAAR_CASCADE_FILES = (
    ("haar_plate", "haarcascade_russian_plate_number.xml"),
    ("haar_plate_16", "haarcascade_license_plate_rus_16stages.xml"),
)

CANDIDATE_FIELDNAMES = [
    "image",
    "candidate_path",
    "candidate_index",
    "source",
    "score",
    "x",
    "y",
    "w",
    "h",
    "aspect",
    "area_ratio",
    "center_x",
    "center_y",
    "char_count",
    "char_density",
    "text_rows",
    "contrast",
    "edge_density",
    "blue_ratio",
    "suggested_label",
    "label",
]

DETECTION_FIELDNAMES = [
    "image",
    "found",
    "score",
    "x",
    "y",
    "w",
    "h",
    "plate_text",
    "compact_text",
    "ocr_confidence",
    "char_count",
    "plate_path",
    "error",
]

CHARACTER_FIELDNAMES = [
    "image",
    "plate_path",
    "char_path",
    "line_index",
    "char_index",
    "char",
    "confidence",
    "x",
    "y",
    "w",
    "h",
    "label",
]


@dataclass(frozen=True)
class Detection:
    box: np.ndarray
    bbox: tuple[int, int, int, int]
    score: float
    crop: np.ndarray
    debug_mask: np.ndarray


@dataclass(frozen=True)
class CharacterComponent:
    x: int
    y: int
    w: int
    h: int
    area: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is not None:
        return image

    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if cv2.imwrite(str(path), image):
        return

    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Cannot encode image: {path}")
    encoded.tofile(str(path))


def iter_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in IMAGE_EXTENSIONS else []
    return sorted(p for p in input_path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def resize_for_detection(image: np.ndarray, max_width: int) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    if width <= max_width:
        return image.copy(), 1.0

    scale = max_width / float(width)
    resized = cv2.resize(image, (max_width, int(height * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    image = image.astype("float32")
    min_value = float(image.min())
    max_value = float(image.max())
    if math.isclose(max_value, min_value):
        return np.zeros(image.shape, dtype=np.uint8)
    image = 255.0 * (image - min_value) / (max_value - min_value)
    return image.astype(np.uint8)


def build_plate_mask(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    wide_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(15, w // 18) | 1, max(3, h // 90) | 1),
    )
    square_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(5, w // 80) | 1, max(5, w // 80) | 1),
    )

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, wide_kernel)
    grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad_x = np.absolute(grad_x)
    grad_x = normalize_to_uint8(grad_x)
    grad_x = cv2.GaussianBlur(grad_x, (5, 5), 0)
    grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, wide_kernel)
    _, text_mask = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    light = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, square_kernel)
    _, light_mask = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    edges = cv2.Canny(enhanced, 80, 180)
    edges = cv2.dilate(edges, wide_kernel, iterations=1)

    mask = cv2.bitwise_or(text_mask, cv2.bitwise_and(text_mask, light_mask))
    mask = cv2.bitwise_or(mask, edges)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, wide_kernel, iterations=2)
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask


def order_points(points: np.ndarray) -> np.ndarray:
    points = points.astype("float32")
    result = np.zeros((4, 2), dtype="float32")
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(-1)
    result[0] = points[np.argmin(sums)]
    result[2] = points[np.argmax(sums)]
    result[1] = points[np.argmin(diffs)]
    result[3] = points[np.argmax(diffs)]
    return result


def clamp_bbox(x: int, y: int, w: int, h: int, image_w: int, image_h: int) -> tuple[int, int, int, int]:
    x = max(0, min(x, image_w - 1))
    y = max(0, min(y, image_h - 1))
    w = max(1, min(w, image_w - x))
    h = max(1, min(h, image_h - y))
    return x, y, w, h


def padded_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    pad_x_ratio: float = 0.08,
    pad_y_ratio: float = 0.16,
) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    image_h, image_w = image_shape
    pad_x = int(w * pad_x_ratio)
    pad_y = int(h * pad_y_ratio)
    return clamp_bbox(x - pad_x, y - pad_y, w + 2 * pad_x, h + 2 * pad_y, image_w, image_h)


def contour_candidate_boxes(mask: np.ndarray) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[np.ndarray] = []
    for contour in contours:
        if cv2.contourArea(contour) < 80:
            continue
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        boxes.append(order_points(box))
    return boxes


def haar_plate_candidate_boxes(gray: np.ndarray) -> list[tuple[str, np.ndarray]]:
    image_h, image_w = gray.shape[:2]
    candidates: list[tuple[str, np.ndarray]] = []
    for source, filename in HAAR_CASCADE_FILES:
        cascade_path = str(Path(cv2.data.haarcascades) / filename)
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            continue
        rects = cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(max(30, image_w // 12), max(12, image_h // 25)),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        for x, y, w, h in rects:
            x, y, w, h = padded_bbox((int(x), int(y), int(w), int(h)), (image_h, image_w), 0.08, 0.18)
            aspect = w / float(max(1, h))
            area_ratio = (w * h) / float(max(1, image_w * image_h))
            if not 1.4 <= aspect <= 7.5:
                continue
            if not 0.006 <= area_ratio <= 0.42:
                continue
            candidates.append((source, bbox_to_box((x, y, w, h))))
    return candidates


def bbox_to_box(bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = bbox
    return np.array(
        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
        dtype=np.float32,
    )


def make_text_binary(gray: np.ndarray) -> np.ndarray:
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    masks = []
    for block_size, constant in ((15, 5), (25, 9), (35, 11)):
        masks.append(
            cv2.adaptiveThreshold(
                enhanced,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                block_size,
                constant,
            )
        )
    return cv2.bitwise_or(masks[0], cv2.bitwise_or(masks[1], masks[2]))


def find_character_components(gray: np.ndarray) -> list[CharacterComponent]:
    image_h, image_w = gray.shape[:2]
    binary = make_text_binary(gray)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components: list[CharacterComponent] = []

    for label in range(1, num_labels):
        x, y, w, h, area = (int(value) for value in stats[label])
        if area < 8 or area > image_h * image_w * 0.06:
            continue
        if h < max(8, image_h * 0.025) or h > max(45, image_h * 0.62):
            continue
        if w < 2 or w > max(28, image_w * 0.24):
            continue

        aspect = h / float(max(1, w))
        extent = area / float(max(1, w * h))
        if aspect < 0.65 or aspect > 8.0:
            continue
        if extent < 0.16 or extent > 0.88:
            continue

        components.append(CharacterComponent(x, y, w, h, area))

    return components


def vertical_overlap(first: CharacterComponent, second: CharacterComponent) -> float:
    overlap = max(0, min(first.y2, second.y2) - max(first.y, second.y))
    return overlap / float(max(1, min(first.h, second.h)))


def horizontal_gap(first: CharacterComponent, second: CharacterComponent) -> int:
    if first.x <= second.x:
        return max(0, second.x - first.x2)
    return max(0, first.x - second.x2)


def component_bbox(components: list[CharacterComponent]) -> tuple[int, int, int, int]:
    x1 = min(component.x for component in components)
    y1 = min(component.y for component in components)
    x2 = max(component.x2 for component in components)
    y2 = max(component.y2 for component in components)
    return x1, y1, x2 - x1, y2 - y1


def connected_component_groups(components: list[CharacterComponent]) -> list[list[CharacterComponent]]:
    parent = list(range(len(components)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for i, first in enumerate(components):
        for j in range(i + 1, len(components)):
            second = components[j]
            height_ratio = min(first.h, second.h) / float(max(first.h, second.h))
            same_line = vertical_overlap(first, second) >= 0.35 or abs(first.cy - second.cy) <= max(first.h, second.h) * 0.55
            close_enough = horizontal_gap(first, second) <= max(14, int(max(first.h, second.h) * 2.4))
            if height_ratio >= 0.62 and same_line and close_enough:
                union(i, j)

    groups_by_root: dict[int, list[CharacterComponent]] = {}
    for index, component in enumerate(components):
        groups_by_root.setdefault(find(index), []).append(component)
    return list(groups_by_root.values())


def plausible_line_group(group: list[CharacterComponent], image_shape: tuple[int, int]) -> bool:
    image_h, image_w = image_shape
    if len(group) < 3:
        return False

    x, y, w, h = component_bbox(group)
    if w < image_w * 0.08 or h < image_h * 0.025:
        return False
    if w > image_w * 0.95 or h > image_h * 0.35:
        return False

    aspect = w / float(max(1, h))
    if not 1.0 <= aspect <= 8.5:
        return False

    heights = np.array([component.h for component in group], dtype=np.float32)
    if float(heights.std()) / float(max(1.0, heights.mean())) > 0.55:
        return False

    return True


def merge_line_groups(line_groups: list[list[CharacterComponent]], image_shape: tuple[int, int]) -> list[list[CharacterComponent]]:
    merged = list(line_groups)
    for i, first in enumerate(line_groups):
        x1, y1, w1, h1 = component_bbox(first)
        for second in line_groups[i + 1 :]:
            x2, y2, w2, h2 = component_bbox(second)
            top, bottom = (y1, y1 + h1), (y2, y2 + h2)
            vertical_gap = max(0, max(top[0], bottom[0]) - min(top[1], bottom[1]))
            horizontal_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
            overlap_ratio = horizontal_overlap / float(max(1, min(w1, w2)))
            center_delta = abs((x1 + w1 / 2.0) - (x2 + w2 / 2.0))

            if vertical_gap > max(h1, h2) * 1.7:
                continue
            if overlap_ratio < 0.25 and center_delta > max(w1, w2) * 0.45:
                continue

            combined = first + second
            if len(combined) >= 5:
                merged.append(combined)
    return merged


def character_group_candidate_boxes(gray: np.ndarray) -> list[np.ndarray]:
    image_h, image_w = gray.shape[:2]
    components = find_character_components(gray)
    if not components:
        return []

    line_groups = [
        group
        for group in connected_component_groups(components)
        if plausible_line_group(group, (image_h, image_w))
    ]
    groups = merge_line_groups(line_groups, (image_h, image_w))

    boxes: list[np.ndarray] = []
    seen: set[tuple[int, int, int, int]] = set()
    for group in groups:
        bbox = component_bbox(group)
        x, y, w, h = bbox
        aspect = w / float(max(1, h))
        if not 0.9 <= aspect <= 7.8:
            continue
        if w * h < image_w * image_h * 0.002:
            continue
        key = (x // 3, y // 3, w // 3, h // 3)
        if key in seen:
            continue
        seen.add(key)
        boxes.append(bbox_to_box(bbox))

    return boxes


def two_line_candidate_boxes(gray: np.ndarray) -> list[np.ndarray]:
    image_h, image_w = gray.shape[:2]
    components = find_character_components(gray)
    if not components:
        return []

    line_groups = [
        group
        for group in connected_component_groups(components)
        if plausible_line_group(group, (image_h, image_w))
    ]

    boxes: list[np.ndarray] = []
    seen: set[tuple[int, int, int, int]] = set()
    for group in line_groups:
        if len(group) < 4:
            continue

        x, y, w, h = component_bbox(group)
        aspect = w / float(max(1, h))
        center_y = (y + h / 2.0) / float(image_h)
        if not 1.2 <= aspect <= 5.5:
            continue
        if center_y < 0.48:
            continue

        variants = (
            (0.22, 1.55, 1.44, 2.85),
            (0.35, 1.85, 1.70, 3.35),
        )
        for pad_left, up, width_mul, height_mul in variants:
            candidate_w = int(round(w * width_mul))
            candidate_h = int(round(h * height_mul))
            candidate_x = int(round(x - w * pad_left))
            candidate_y = int(round(y - h * up))
            bbox = clamp_bbox(candidate_x, candidate_y, candidate_w, candidate_h, image_w, image_h)
            bx, by, bw, bh = bbox
            candidate_aspect = bw / float(max(1, bh))
            if not 0.95 <= candidate_aspect <= 2.4:
                continue
            if bw * bh < image_w * image_h * 0.015:
                continue
            key = (bx // 3, by // 3, bw // 3, bh // 3)
            if key in seen:
                continue
            seen.add(key)
            boxes.append(bbox_to_box(bbox))

    return boxes


def bright_region_candidate_boxes(gray: np.ndarray) -> list[np.ndarray]:
    image_h, image_w = gray.shape[:2]
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, bright = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)),
        iterations=2,
    )
    bright = cv2.erode(bright, None, iterations=1)
    bright = cv2.dilate(bright, None, iterations=1)

    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[np.ndarray] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_w * image_h * 0.005:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / float(max(1, h))
        area_ratio = (w * h) / float(image_w * image_h)
        fill_ratio = area / float(max(1, w * h))

        if not 0.75 <= aspect <= 7.8:
            continue
        if area_ratio > 0.92:
            continue
        if fill_ratio < 0.28:
            continue
        if w < image_w * 0.10 or h < image_h * 0.05:
            continue

        boxes.append(bbox_to_box((x, y, w, h)))

    return boxes


def blue_plate_candidate_boxes(image: np.ndarray) -> list[np.ndarray]:
    image_h, image_w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    masks = [
        cv2.inRange(hsv, np.array([95, 80, 25]), np.array([125, 255, 255])),
        cv2.inRange(hsv, np.array([90, 50, 20]), np.array([135, 255, 255])),
    ]

    boxes: list[np.ndarray] = []
    seen: set[tuple[int, int, int, int]] = set()

    def add_bbox(bbox: tuple[int, int, int, int]) -> None:
        x, y, w, h = padded_bbox(bbox, (image_h, image_w), pad_x_ratio=0.08, pad_y_ratio=0.25)
        aspect = w / float(max(1, h))
        area_ratio = (w * h) / float(image_w * image_h)
        if not 1.7 <= aspect <= 8.5:
            return
        if area_ratio < 0.0015 or area_ratio > 0.18:
            return
        if w < image_w * 0.07 or h < image_h * 0.025:
            return
        if w > image_w * 0.70 or h > image_h * 0.35:
            return
        if (y + h / 2.0) / float(image_h) < 0.20:
            return
        key = (x // 3, y // 3, w // 3, h // 3)
        if key in seen:
            return
        seen.add(key)
        boxes.append(bbox_to_box((x, y, w, h)))

    for mask in masks:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5)),
            iterations=1,
        )
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            fill_ratio = area / float(max(1, w * h))
            if area < image_w * image_h * 0.0006:
                continue
            if fill_ratio < 0.08:
                continue
            add_bbox((x, y, w, h))

        projection = (mask > 0).sum(axis=1).astype(np.float32)
        projection = cv2.blur(projection.reshape(-1, 1), (1, 9)).reshape(-1)
        for threshold in (30.0, 40.0):
            active = projection > threshold
            start: int | None = None
            bands: list[tuple[int, int]] = []
            for index, is_active in enumerate(active):
                if is_active and start is None:
                    start = index
                elif not is_active and start is not None:
                    if index - start >= image_h * 0.025:
                        bands.append((start, index))
                    start = None
            if start is not None and len(active) - start >= image_h * 0.025:
                bands.append((start, len(active)))

            for y1, y2 in bands:
                y1 = max(0, y1 - 8)
                y2 = min(image_h, y2 + 8)
                columns = np.where((mask[y1:y2] > 0).any(axis=0))[0]
                if columns.size == 0:
                    continue
                x1 = max(0, int(columns.min()) - 8)
                x2 = min(image_w, int(columns.max()) + 9)
                add_bbox((x1, y1, x2 - x1, y2 - y1))

    return boxes


def edge_candidate_boxes(gray: np.ndarray) -> list[np.ndarray]:
    image_h, image_w = gray.shape[:2]
    blurred = cv2.bilateralFilter(gray, 7, 55, 55)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[np.ndarray] = []
    seen: set[tuple[int, int, int, int]] = set()
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_w * image_h * 0.002 or area > image_w * image_h * 0.70:
            continue

        box = order_points(cv2.boxPoints(cv2.minAreaRect(contour)))
        x, y, w, h = cv2.boundingRect(box.astype(np.int32))
        aspect = w / float(max(1, h))
        area_ratio = (w * h) / float(image_w * image_h)
        center_y = (y + h / 2.0) / float(image_h)

        if not 0.55 <= aspect <= 8.0:
            continue
        if w < image_w * 0.08 or h < image_h * 0.035:
            continue
        if center_y < 0.33 and area_ratio < 0.18:
            continue

        key = (x // 4, y // 4, w // 4, h // 4)
        if key in seen:
            continue
        seen.add(key)
        boxes.append(box)

    return boxes


def bbox_area(bbox: tuple[int, int, int, int]) -> int:
    return bbox[2] * bbox[3]


def bbox_iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    x1, y1, w1, h1 = first
    x2, y2, w2, h2 = second
    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    intersection = iw * ih
    union = bbox_area(first) + bbox_area(second) - intersection
    return intersection / float(max(1, union))


def dedupe_candidate_boxes(candidates: list[tuple[str, np.ndarray]], iou_threshold: float = 0.88) -> list[tuple[str, np.ndarray]]:
    deduped: list[tuple[str, np.ndarray]] = []
    seen_bboxes: list[tuple[int, int, int, int]] = []
    for source, box in candidates:
        bbox = cv2.boundingRect(box.astype(np.int32))
        if any(bbox_iou(bbox, seen) >= iou_threshold for seen in seen_bboxes):
            continue
        seen_bboxes.append(bbox)
        deduped.append((source, box))
    return deduped


def primary_candidate_boxes(image: np.ndarray, gray: np.ndarray, mask: np.ndarray) -> list[tuple[str, np.ndarray]]:
    candidates: list[tuple[str, np.ndarray]] = []
    candidates.extend(haar_plate_candidate_boxes(gray))
    for source, boxes in (
        ("blue", blue_plate_candidate_boxes(image)),
        ("char_group", character_group_candidate_boxes(gray)),
        ("two_line", two_line_candidate_boxes(gray)),
        ("bright_region", bright_region_candidate_boxes(gray)),
        ("morph_contour", contour_candidate_boxes(mask)),
    ):
        candidates.extend((source, box) for box in boxes)
    return dedupe_candidate_boxes(candidates)


def fallback_candidate_boxes(gray: np.ndarray) -> list[tuple[str, np.ndarray]]:
    return dedupe_candidate_boxes([("edge_fallback", box) for box in edge_candidate_boxes(gray)])


def candidate_feature_dict(
    image: np.ndarray,
    gray: np.ndarray,
    mask: np.ndarray,
    box: np.ndarray,
    source: str,
) -> tuple[dict[str, float | int | str], np.ndarray]:
    score, bbox, crop = score_candidate(image, gray, box, mask)
    x, y, w, h = bbox
    roi_gray = gray[y : y + h, x : x + w]
    roi_mask = mask[y : y + h, x : x + w]
    roi_color = image[y : y + h, x : x + w]

    if roi_gray.size == 0:
        features = {
            "source": source,
            "score": round(score, 4),
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "aspect": 0.0,
            "area_ratio": 0.0,
            "char_density": 0.0,
            "char_count": 0,
            "text_rows": 0,
            "contrast": 0.0,
            "edge_density": 0.0,
            "blue_ratio": 0.0,
            "center_x": 0.0,
            "center_y": 0.0,
        }
        return features, crop

    image_h, image_w = gray.shape[:2]
    char_density_value, char_count = character_density(roi_gray)
    hsv_roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv_roi, np.array([90, 50, 20]), np.array([135, 255, 255]))
    features = {
        "source": source,
        "score": round(score, 4),
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "aspect": round(w / float(max(1, h)), 4),
        "area_ratio": round((w * h) / float(max(1, image_w * image_h)), 6),
        "char_density": round(char_density_value, 6),
        "char_count": int(char_count),
        "text_rows": int(estimate_text_rows(roi_gray)),
        "contrast": round(float(np.percentile(roi_gray, 90) - np.percentile(roi_gray, 10)), 4),
        "edge_density": round(float(cv2.countNonZero(roi_mask)) / float(max(1, w * h)), 6),
        "blue_ratio": round(float(cv2.countNonZero(blue_mask)) / float(max(1, w * h)), 6),
        "center_x": round((x + w / 2.0) / float(max(1, image_w)), 6),
        "center_y": round((y + h / 2.0) / float(max(1, image_h)), 6),
    }
    return features, crop


def ranked_candidate_records(
    image: np.ndarray,
    max_width: int,
    include_fallback: bool = True,
) -> list[tuple[dict[str, float | int | str], np.ndarray]]:
    resized, _ = resize_for_detection(image, max_width=max_width)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    mask = build_plate_mask(gray)

    primary_candidates = dedupe_candidate_boxes(primary_candidate_boxes(resized, gray, mask), iou_threshold=0.78)
    primary_records = [
        candidate_feature_dict(resized, gray, mask, box, source) for source, box in primary_candidates
    ]
    primary_records.sort(key=lambda item: float(item[0]["score"]), reverse=True)

    fallback_records: list[tuple[dict[str, float | int | str], np.ndarray]] = []
    if include_fallback:
        fallback_candidates = dedupe_candidate_boxes(fallback_candidate_boxes(gray), iou_threshold=0.78)
        fallback_records = [
            candidate_feature_dict(resized, gray, mask, box, source) for source, box in fallback_candidates
        ]
        fallback_records.sort(key=lambda item: float(item[0]["score"]), reverse=True)

    if any(float(record[0]["score"]) > 0 for record in primary_records):
        return primary_records + fallback_records
    return fallback_records + primary_records


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value)


def export_candidate_records(
    image_path: Path,
    image: np.ndarray,
    export_dir: Path,
    max_width: int,
    limit: int,
    pseudo_positive_threshold: float,
) -> list[dict[str, str | int | float]]:
    records = ranked_candidate_records(image, max_width=max_width, include_fallback=True)
    if limit > 0:
        records = records[:limit]

    rows: list[dict[str, str | int | float]] = []
    for index, (features, crop) in enumerate(records):
        if crop.size == 0:
            continue

        source = str(features["source"])
        candidate_path = unique_output_path(
            export_dir,
            image_path,
            f"_cand_{index:02d}_{safe_name(source)}",
        )
        write_image(candidate_path, crop)

        score = float(features["score"])
        row: dict[str, str | int | float] = {
            "image": str(image_path),
            "candidate_path": str(candidate_path),
            "candidate_index": index,
            "suggested_label": 1 if index == 0 and score >= pseudo_positive_threshold else "",
            "label": "",
        }
        row.update(features)
        rows.append(row)

    return rows


def rectangularity(box: np.ndarray, contour_area: float) -> float:
    x, y, w, h = cv2.boundingRect(box.astype(np.int32))
    bbox_area = max(1, w * h)
    return min(1.0, contour_area / bbox_area)


def character_density(gray_roi: np.ndarray) -> tuple[float, int]:
    if gray_roi.size == 0:
        return 0.0, 0

    roi = cv2.resize(gray_roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    roi = cv2.GaussianBlur(roi, (3, 3), 0)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4)).apply(roi)
    blackhat = cv2.morphologyEx(
        clahe,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)),
    )
    _, blackhat = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    masks = [
        cv2.adaptiveThreshold(
            roi,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            25,
            15,
        ),
        cv2.adaptiveThreshold(
            clahe,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            25,
            7,
        ),
        cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1],
        blackhat,
    ]

    best_density = 0.0
    best_count = 0
    for threshold in masks:
        threshold = cv2.morphologyEx(
            threshold,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        )
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(threshold, connectivity=8)
        roi_h, roi_w = threshold.shape[:2]
        char_like = 0
        char_area = 0
        for label in range(1, num_labels):
            x, y, w, h, area = stats[label]
            if area < 8:
                continue
            if h < roi_h * 0.10 or h > roi_h * 0.98:
                continue
            if w < roi_w * 0.005 or w > roi_w * 0.35:
                continue
            if area / float(max(1, w * h)) < 0.12:
                continue
            char_like += 1
            char_area += int(area)

        density = char_area / float(max(1, roi_h * roi_w))
        if (char_like, density) > (best_count, best_density):
            best_count = char_like
            best_density = density

    return best_density, best_count


def estimate_text_rows(gray_roi: np.ndarray) -> int:
    if gray_roi.size == 0:
        return 0

    roi = cv2.resize(gray_roi, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4)).apply(roi)
    blackhat = cv2.morphologyEx(
        clahe,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)),
    )
    _, blackhat = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    masks = [
        cv2.adaptiveThreshold(
            clahe,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            25,
            7,
        ),
        cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1],
        blackhat,
    ]

    best_rows = 0
    for mask in masks:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
        )
        roi_h, roi_w = mask.shape[:2]
        projection = (mask > 0).sum(axis=1).astype(np.float32)
        projection = cv2.blur(projection.reshape(-1, 1), (1, 9)).reshape(-1)
        threshold = max(float(projection.max()) * 0.25, roi_w * 0.04)
        active = projection > threshold

        bands: list[tuple[int, int]] = []
        start: int | None = None
        for index, is_active in enumerate(active):
            if is_active and start is None:
                start = index
            elif not is_active and start is not None:
                if index - start >= roi_h * 0.08:
                    bands.append((start, index))
                start = None
        if start is not None and len(active) - start >= roi_h * 0.08:
            bands.append((start, len(active)))

        merged: list[tuple[int, int]] = []
        for band in bands:
            if merged and band[0] - merged[-1][1] <= roi_h * 0.06:
                merged[-1] = (merged[-1][0], band[1])
            else:
                merged.append(band)

        best_rows = max(best_rows, len(merged))

    return best_rows


def score_candidate(
    image: np.ndarray,
    gray: np.ndarray,
    box: np.ndarray,
    mask: np.ndarray,
) -> tuple[float, tuple[int, int, int, int], np.ndarray]:
    image_h, image_w = gray.shape[:2]
    x, y, w, h = cv2.boundingRect(box.astype(np.int32))
    x, y, w, h = padded_bbox((x, y, w, h), (image_h, image_w), pad_x_ratio=0.12, pad_y_ratio=0.28)

    if w < image_w * 0.08 or h < image_h * 0.05:
        return -1.0, (x, y, w, h), image[y : y + h, x : x + w]

    aspect = w / float(max(1, h))
    if not 0.60 <= aspect <= 7.2:
        return -1.0, (x, y, w, h), image[y : y + h, x : x + w]

    roi_gray = gray[y : y + h, x : x + w]
    roi_mask = mask[y : y + h, x : x + w]
    roi_color = image[y : y + h, x : x + w]

    density, char_count = character_density(roi_gray)
    if char_count < 3:
        return -1.0, (x, y, w, h), roi_color

    area_ratio = (w * h) / float(image_w * image_h)
    if (w > image_w * 0.98 or h > image_h * 0.90 or area_ratio > 0.92) and char_count < 6:
        return -1.0, (x, y, w, h), roi_color
    if h > image_h * 0.62 and char_count < 5:
        return -1.0, (x, y, w, h), roi_color

    edge_density = float(cv2.countNonZero(roi_mask)) / float(max(1, w * h))
    fill = rectangularity(box, float(w * h))
    contrast = float(np.percentile(roi_gray, 90) - np.percentile(roi_gray, 10))
    contrast_score = min(contrast / 150.0, 1.0)
    light_score = min(float(np.percentile(roi_gray, 75)) / 210.0, 1.0)
    hsv_roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv_roi, np.array([90, 50, 20]), np.array([135, 255, 255]))
    blue_ratio = cv2.countNonZero(blue_mask) / float(max(1, w * h))
    if aspect >= 2.2 and blue_ratio >= 0.18:
        blue_score = min(blue_ratio / 0.18, 1.0)
    elif aspect >= 1.7 and blue_ratio >= 0.24:
        blue_score = min(blue_ratio / 0.24, 1.0)
    else:
        blue_score = 0.0

    if aspect >= 2.0:
        aspect_score = 1.0 - min(abs(aspect - 4.2) / 4.2, 1.0)
    else:
        aspect_score = 1.0 - min(abs(aspect - 1.45) / 1.45, 1.0)

    text_rows = estimate_text_rows(roi_gray) if aspect < 2.2 and char_count >= 5 else 1
    multi_line_score = 1.0 if text_rows >= 2 and aspect < 2.0 else 0.0

    center_x = (x + w / 2.0) / image_w
    center_y = (y + h / 2.0) / image_h
    center_score = 1.0 - min(abs(center_x - 0.5) / 0.5, 1.0)
    vertical_score = 1.0 - min(abs(center_y - 0.62) / 0.62, 1.0)
    char_score = min(char_count / 7.0, 1.0)
    density_score = min(density / 0.22, 1.0)
    edge_score = min(edge_density / 0.55, 1.0)
    area_score = 1.0 - min(abs(area_ratio - 0.045) / 0.12, 1.0)
    oversize_penalty = max(0.0, min((area_ratio - 0.30) / 0.25, 1.0))
    low_contrast_penalty = max(0.0, min((45.0 - contrast) / 15.0, 1.0))
    weak_one_line_penalty = 1.0 if aspect > 2.0 and contrast < 115.0 and blue_ratio < 0.12 else 0.0
    small_area_penalty = max(0.0, min((0.012 - area_ratio) / 0.008, 1.0))
    bottom_edge_penalty = 1.0 if center_y > 0.88 and area_ratio < 0.055 else 0.0
    off_center_small_penalty = 1.0 if area_ratio < 0.085 and abs(center_x - 0.5) > 0.28 else 0.0

    score = (
        2.5 * char_score
        + 1.6 * aspect_score
        + 1.4 * density_score
        + 1.0 * edge_score
        + 0.7 * area_score
        + 0.4 * center_score
        + 0.3 * vertical_score
        + 0.2 * fill
        + 0.6 * contrast_score
        + 0.3 * light_score
        + 1.2 * multi_line_score
        + 1.8 * blue_score
        - 2.5 * oversize_penalty
        - 2.2 * low_contrast_penalty
        - 2.0 * weak_one_line_penalty
        - 2.4 * small_area_penalty
        - 2.5 * bottom_edge_penalty
        - 2.4 * off_center_small_penalty
    )
    return score, (x, y, w, h), roi_color


def refine_bbox_from_characters(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    image_h, image_w = image.shape[:2]
    x, y, w, h = bbox
    crop = image[y : y + h, x : x + w]
    segments = segment_plate_characters(crop)
    if len(segments) < 5:
        return bbox

    x1 = min(segment.bbox[0] for segment in segments)
    y1 = min(segment.bbox[1] for segment in segments)
    x2 = max(segment.bbox[0] + segment.bbox[2] for segment in segments)
    y2 = max(segment.bbox[1] + segment.bbox[3] for segment in segments)
    char_heights = np.array([segment.bbox[3] for segment in segments], dtype=np.float32)
    median_h = float(np.median(char_heights))
    line_count = len({segment.line_index for segment in segments})
    area_ratio = (w * h) / float(max(1, image_w * image_h))

    if line_count == 2 and area_ratio < 0.65:
        return bbox

    if line_count == 1:
        pad_x = int(round(median_h * 0.85))
        pad_y = int(round(median_h * 0.45))
    elif area_ratio > 0.65:
        pad_x = int(round(median_h * 0.45))
        pad_y = int(round(median_h * 0.25))
    else:
        pad_x = int(round(median_h * 0.80))
        pad_y = int(round(median_h * 0.80))
    local = clamp_bbox(
        x1 - pad_x,
        y1 - pad_y,
        (x2 - x1) + 2 * pad_x,
        (y2 - y1) + 2 * pad_y,
        w,
        h,
    )

    lx, ly, lw, lh = local
    if lw * lh < w * h * 0.08:
        return bbox
    aspect = lw / float(max(1, lh))
    if not 0.65 <= aspect <= 7.5:
        return bbox

    return clamp_bbox(x + lx, y + ly, lw, lh, image_w, image_h)


def detect_plate(image: np.ndarray, max_width: int = 900) -> Detection | None:
    resized, scale = resize_for_detection(image, max_width=max_width)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    mask = build_plate_mask(gray)

    def score_candidates(candidates: list[tuple[str, np.ndarray]]) -> list[tuple[float, tuple[int, int, int, int], np.ndarray, np.ndarray]]:
        scored_candidates: list[tuple[float, tuple[int, int, int, int], np.ndarray, np.ndarray]] = []
        for source, candidate_box in candidates:
            score, bbox, crop = score_candidate(resized, gray, candidate_box, mask)
            if score > 0:
                layout_score, layout_count, layout_counts = plate_layout_score(crop)
                source_bonus = 0.35 if source in {"char_group", "blue"} else 0.0
                bbox_aspect = bbox[2] / float(max(1, bbox[3]))
                if source.startswith("haar") and layout_count >= 7:
                    source_bonus += 2.6
                    if len(layout_counts) == 1 and bbox_aspect >= 2.0:
                        source_bonus += 0.8
                if (
                    source == "two_line"
                    and len(layout_counts) == 2
                    and 2 <= layout_counts[0] <= 4
                    and 4 <= layout_counts[1] <= 6
                ):
                    source_bonus += 1.2
                if len(layout_counts) == 2 and bbox[3] < resized.shape[0] * 0.12:
                    source_bonus -= 2.4
                if source == "char_group" and bbox_aspect < 2.0 and len(layout_counts) == 1:
                    source_bonus -= 1.2
                if source == "two_line" and layout_count < 7:
                    source_bonus -= 0.5
                if layout_count >= 7:
                    segments = segment_plate_characters(crop)
                    if segments:
                        crop_h, crop_w = crop.shape[:2]
                        sx1 = min(segment.bbox[0] for segment in segments)
                        sy1 = min(segment.bbox[1] for segment in segments)
                        sx2 = max(segment.bbox[0] + segment.bbox[2] for segment in segments)
                        sy2 = max(segment.bbox[1] + segment.bbox[3] for segment in segments)
                        margin_x = min(sx1, crop_w - sx2) / float(max(1, crop_w))
                        margin_y = min(sy1, crop_h - sy2) / float(max(1, crop_h))
                        if margin_x < 0.025:
                            source_bonus -= 2.2
                        elif margin_x < 0.06:
                            source_bonus -= 0.9
                        if margin_y < 0.025:
                            source_bonus -= 1.0
                score = score + layout_score + source_bonus
                if score > 0:
                    scored_candidates.append((score, bbox, crop, candidate_box))
        return scored_candidates

    scored = score_candidates(primary_candidate_boxes(resized, gray, mask))
    if not scored:
        scored = score_candidates(fallback_candidate_boxes(gray))
        if not scored:
            return None

    score, bbox, _, box = max(scored, key=lambda item: item[0])
    bbox = refine_bbox_from_characters(resized, bbox)
    box = bbox_to_box(bbox)

    if scale != 1.0:
        inv_scale = 1.0 / scale
        x, y, w, h = bbox
        full_bbox = (
            int(round(x * inv_scale)),
            int(round(y * inv_scale)),
            int(round(w * inv_scale)),
            int(round(h * inv_scale)),
        )
        full_bbox = clamp_bbox(*full_bbox, image.shape[1], image.shape[0])
        full_box = box * inv_scale
    else:
        full_bbox = bbox
        full_box = box

    x, y, w, h = full_bbox
    crop = image[y : y + h, x : x + w]
    return Detection(full_box.astype(np.int32), full_bbox, score, crop, mask)


def draw_detection(image: np.ndarray, detection: Detection | None, plate_text: str = "") -> np.ndarray:
    debug = image.copy()
    if detection is None:
        return debug

    x, y, w, h = detection.bbox
    cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.polylines(debug, [detection.box.reshape((-1, 1, 2))], True, (0, 180, 255), 2)
    label = f"{plate_text} {detection.score:.2f}".strip() if plate_text else f"plate {detection.score:.2f}"
    cv2.putText(debug, label, (x, max(16, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return debug


def unique_output_path(base_dir: Path, image_path: Path, suffix: str) -> Path:
    stem_path = image_path.with_suffix("")
    parts = list(stem_path.parts)
    if stem_path.is_absolute():
        parts = [part.replace(":", "") for part in parts if part != stem_path.anchor]
    return base_dir.joinpath(*parts).with_name(f"{image_path.stem}{suffix}.png")


def save_character_crops(
    image_path: Path,
    plate_path: str,
    ocr: PlateOCRResult,
    chars_dir: Path,
) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for global_index, character in enumerate(ocr.characters):
        suffix = (
            f"_char_{global_index:02d}_"
            f"l{character.line_index}_c{character.char_index}_{safe_name(character.char)}"
        )
        char_path = unique_output_path(chars_dir, image_path, suffix)
        write_image(char_path, character.crop)
        x, y, w, h = character.bbox
        rows.append(
            {
                "image": str(image_path),
                "plate_path": plate_path,
                "char_path": str(char_path),
                "line_index": character.line_index,
                "char_index": character.char_index,
                "char": character.char,
                "confidence": round(character.confidence, 4),
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "label": "",
            }
        )
    return rows


def process_images(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    output_dir = Path(args.output)
    debug_dir = Path(args.debug_output) if args.debug_output else None
    chars_dir = Path(args.chars_output) if args.chars_output else None
    char_report = Path(args.char_report) if args.char_report else Path(args.report).with_name("characters.csv")
    char_model = load_char_model(Path(args.char_model)) if args.char_model else None
    candidate_dir = Path(args.export_candidates) if args.export_candidates else None
    candidate_report = (
        Path(args.candidate_report)
        if args.candidate_report
        else candidate_dir / "labels.csv"
        if candidate_dir
        else None
    )
    images = iter_images(input_path)
    if args.limit:
        images = images[: args.limit]

    if not images:
        print(f"No images found in {input_path}", file=sys.stderr)
        return 1

    rows: list[dict[str, str | int | float]] = []
    candidate_rows: list[dict[str, str | int | float]] = []
    character_rows: list[dict[str, str | int | float]] = []
    found = 0
    for image_path in images:
        try:
            image = read_image(image_path)
            detection = detect_plate(image, max_width=args.max_width)
        except Exception as exc:
            rows.append(
                {
                    "image": str(image_path),
                    "found": 0,
                    "score": 0.0,
                    "x": "",
                    "y": "",
                    "w": "",
                    "h": "",
                    "plate_text": "",
                    "compact_text": "",
                    "ocr_confidence": 0.0,
                    "char_count": 0,
                    "plate_path": "",
                    "error": str(exc),
                }
            )
            print(f"[ERROR] {image_path}: {exc}", file=sys.stderr)
            continue

        if candidate_dir is not None:
            try:
                candidate_rows.extend(
                    export_candidate_records(
                        image_path=image_path,
                        image=image,
                        export_dir=candidate_dir,
                        max_width=args.max_width,
                        limit=args.candidates_per_image,
                        pseudo_positive_threshold=args.pseudo_positive_threshold,
                    )
                )
            except Exception as exc:
                print(f"[CANDIDATE_ERROR] {image_path}: {exc}", file=sys.stderr)

        plate_path = ""
        bbox: tuple[int | str, int | str, int | str, int | str] = ("", "", "", "")
        score = 0.0
        plate_text = ""
        compact_text = ""
        ocr_confidence = 0.0
        char_count = 0
        if detection is not None:
            found += 1
            plate_path_obj = unique_output_path(output_dir, image_path, "_plate")
            write_image(plate_path_obj, detection.crop)
            plate_path = str(plate_path_obj)
            bbox = detection.bbox
            score = detection.score
            if not args.skip_ocr:
                try:
                    ocr = recognize_plate(detection.crop, char_model)
                    plate_text = ocr.text
                    compact_text = ocr.compact_text
                    ocr_confidence = ocr.confidence
                    char_count = len(ocr.characters)
                    if args.ocr_engine in {"easyocr", "hybrid"}:
                        easy_text, easy_compact, easy_confidence = recognize_plate_easyocr_text(detection.crop)
                        use_easyocr = args.ocr_engine == "easyocr" or (
                            bool(easy_text)
                            and easy_confidence >= 0.45
                            and (ocr_confidence < 0.90 or len(compact_text) not in {8, 9})
                        )
                        if use_easyocr:
                            plate_text = easy_text
                            compact_text = easy_compact
                            ocr_confidence = easy_confidence
                    if chars_dir is not None:
                        character_rows.extend(save_character_crops(image_path, plate_path, ocr, chars_dir))
                except Exception as exc:
                    print(f"[OCR_ERROR] {image_path}: {exc}", file=sys.stderr)

        if debug_dir:
            debug = draw_detection(image, detection, plate_text)
            debug_path = unique_output_path(debug_dir, image_path, "_debug")
            write_image(debug_path, debug)

        rows.append(
            {
                "image": str(image_path),
                "found": 1 if detection else 0,
                "score": round(score, 4),
                "x": bbox[0],
                "y": bbox[1],
                "w": bbox[2],
                "h": bbox[3],
                "plate_text": plate_text,
                "compact_text": compact_text,
                "ocr_confidence": round(ocr_confidence, 4),
                "char_count": char_count,
                "plate_path": plate_path,
                "error": "",
            }
        )
        status = "FOUND" if detection else "MISS"
        if not args.quiet:
            print(f"[{status}] {image_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.report)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=DETECTION_FIELDNAMES,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Found {found}/{len(images)} plates.")
    print(f"Plate crops: {output_dir}")
    print(f"Report: {csv_path}")
    if debug_dir:
        print(f"Debug images: {debug_dir}")
    if chars_dir is not None and not args.skip_ocr:
        chars_dir.mkdir(parents=True, exist_ok=True)
        char_report.parent.mkdir(parents=True, exist_ok=True)
        with char_report.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CHARACTER_FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(character_rows)
        print(f"Character crops: {chars_dir}")
        print(f"Character report: {char_report}")
    if candidate_dir is not None and candidate_report is not None:
        candidate_report.parent.mkdir(parents=True, exist_ok=True)
        with candidate_report.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CANDIDATE_FIELDNAMES, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(candidate_rows)
        print(f"Candidate crops: {candidate_dir}")
        print(f"Candidate labels CSV: {candidate_report}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and crop vehicle license plates using classical image processing.",
    )
    parser.add_argument("--input", default="images", help="Image file or directory to process.")
    parser.add_argument("--output", default="outputs/plates", help="Directory for cropped plates.")
    parser.add_argument("--debug-output", default="outputs/debug", help="Directory for annotated images.")
    parser.add_argument("--report", default="outputs/detections.csv", help="CSV report path.")
    parser.add_argument("--chars-output", default="outputs/chars", help="Directory for extracted character crops. Empty disables saving chars.")
    parser.add_argument("--char-report", default="outputs/characters.csv", help="CSV report path for extracted characters.")
    parser.add_argument("--char-model", default="", help="Optional trained character KNN model (.npz) for OCR.")
    parser.add_argument(
        "--ocr-engine",
        choices=["template", "easyocr", "hybrid"],
        default="hybrid",
        help="OCR engine. hybrid uses the local template OCR and replaces it with EasyOCR when EasyOCR gives a stronger plate-format result.",
    )
    parser.add_argument("--skip-ocr", action="store_true", help="Only detect/crop plates, do not segment or recognize characters.")
    parser.add_argument("--max-width", type=int, default=900, help="Resize large images to this width before detection.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit for quick testing.")
    parser.add_argument(
        "--export-candidates",
        default="",
        help="Optional directory for cropped candidate regions used to build training data.",
    )
    parser.add_argument(
        "--candidate-report",
        default="",
        help="Optional CSV path for candidate labels. Defaults to <export-candidates>/labels.csv.",
    )
    parser.add_argument(
        "--candidates-per-image",
        type=int,
        default=8,
        help="Number of top ranked candidates to export per image. Use 0 to export all candidates.",
    )
    parser.add_argument(
        "--pseudo-positive-threshold",
        type=float,
        default=6.0,
        help="Mark the top candidate as suggested_label=1 when its score reaches this threshold.",
    )
    parser.add_argument("--quiet", action="store_true", help="Do not print one status line per image.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(process_images(parse_args()))
