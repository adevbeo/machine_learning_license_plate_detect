from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - OpenCV template fallback still works.
    Image = None
    ImageDraw = None
    ImageFont = None


DIGITS = "0123456789"
LETTERS = "ABCDEFGHKLMNPRSTUVXYZ"
CHARSET = DIGITS + LETTERS
CHAR_SIZE = (32, 48)
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\bahnschrift.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\impact.ttf",
    r"C:\Windows\Fonts\GOTHICB.TTF",
    r"C:\Windows\Fonts\tahomabd.ttf",
    r"C:\Windows\Fonts\consolab.ttf",
]


@dataclass(frozen=True)
class CharacterSegment:
    line_index: int
    char_index: int
    bbox: tuple[int, int, int, int]
    crop: np.ndarray


@dataclass(frozen=True)
class RecognizedCharacter:
    line_index: int
    char_index: int
    bbox: tuple[int, int, int, int]
    char: str
    confidence: float
    crop: np.ndarray


@dataclass(frozen=True)
class PlateOCRResult:
    text: str
    compact_text: str
    confidence: float
    characters: list[RecognizedCharacter]


CharModel = dict[str, np.ndarray]
OCR_ALLOWED = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-."
DIGIT_NORMALIZE = str.maketrans(
    {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "7", "S": "5", "B": "8", "G": "6", "T": "7", "J": "3"}
)
LETTER_NORMALIZE = {"0": "A", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G", "8": "B", "I": "A", "L": "A"}


def resize_plate_for_ocr(image: np.ndarray, target_height: int = 150) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        return image, 1.0

    scale = target_height / float(height)
    if scale <= 0:
        return image, 1.0
    resized = cv2.resize(image, (max(1, int(round(width * scale))), target_height), interpolation=cv2.INTER_CUBIC)
    return resized, scale


def normalize_binary_char(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask.copy()

    if gray.size == 0:
        return np.zeros((CHAR_SIZE[1], CHAR_SIZE[0]), dtype=np.uint8)

    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if cv2.countNonZero(binary) > binary.size * 0.55:
        binary = cv2.bitwise_not(binary)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num_labels > 2:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = int(areas.max()) if areas.size else 0
        cleaned = np.zeros_like(binary)
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= max(8, int(largest * 0.30)):
                cleaned[labels == label] = 255
        if cv2.countNonZero(cleaned) > 0:
            binary = cleaned

    points = cv2.findNonZero(binary)
    if points is None:
        return np.zeros((CHAR_SIZE[1], CHAR_SIZE[0]), dtype=np.uint8)

    x, y, w, h = cv2.boundingRect(points)
    roi = binary[y : y + h, x : x + w]
    canvas_w, canvas_h = CHAR_SIZE
    pad = 4
    scale = min((canvas_w - 2 * pad) / float(max(1, w)), (canvas_h - 2 * pad) / float(max(1, h)))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    x0 = (canvas_w - new_w) // 2
    y0 = (canvas_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def ocr_masks(gray: np.ndarray) -> list[np.ndarray]:
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(4, 4)).apply(gray)
    blurred = cv2.GaussianBlur(clahe, (3, 3), 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    masks = [
        cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            25,
            9,
        ),
        cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            7,
        ),
        cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1],
        cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            25,
            5,
        ),
        cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1],
    ]

    blackhat = cv2.morphologyEx(
        blurred,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5)),
    )
    masks.append(cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])

    cleaned: list[np.ndarray] = []
    for mask in masks:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        cleaned.append(mask)
    return cleaned


def extract_component_bboxes(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    height, width = mask.shape[:2]
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components: list[tuple[int, int, int, int, int]] = []

    for label in range(1, num_labels):
        x, y, w, h, area = (int(value) for value in stats[label])
        if area < max(18, width * height * 0.00045):
            continue
        if h < height * 0.13 or h > height * 0.82:
            continue
        if w < width * 0.008 or w > width * 0.28:
            continue
        if x <= 1 or y <= 1 or x + w >= width - 1 or y + h >= height - 1:
            if h > height * 0.55 or w > width * 0.12:
                continue

        aspect = h / float(max(1, w))
        extent = area / float(max(1, w * h))
        if not 0.75 <= aspect <= 7.5:
            continue
        if not 0.12 <= extent <= 0.88:
            continue

        components.append((x, y, w, h, area))

    return components


def split_wide_components(
    components: list[tuple[int, int, int, int, int]],
    median_width: float,
) -> list[tuple[int, int, int, int, int]]:
    if median_width <= 0:
        return components

    split: list[tuple[int, int, int, int, int]] = []
    for x, y, w, h, area in components:
        width_ratio = w / median_width
        pieces = max(2, int(round(width_ratio))) if width_ratio >= 1.25 else 1
        if pieces <= 1 or pieces > 3 or w / float(max(1, h)) < 0.42:
            split.append((x, y, w, h, area))
            continue

        piece_w = w // pieces
        for index in range(pieces):
            px = x + index * piece_w
            pw = piece_w if index < pieces - 1 else x + w - px
            split.append((px, y, pw, h, max(1, area // pieces)))
    return split


def cluster_lines(components: list[tuple[int, int, int, int, int]]) -> list[list[tuple[int, int, int, int, int]]]:
    if not components:
        return []

    heights = np.array([h for _, _, _, h, _ in components], dtype=np.float32)
    median_height = float(np.median(heights))
    components = sorted(components, key=lambda item: item[1] + item[3] / 2.0)

    lines: list[list[tuple[int, int, int, int, int]]] = []
    for component in components:
        cy = component[1] + component[3] / 2.0
        placed = False
        for line in lines:
            line_cy = float(np.mean([item[1] + item[3] / 2.0 for item in line]))
            if abs(cy - line_cy) <= median_height * 0.75:
                line.append(component)
                placed = True
                break
        if not placed:
            lines.append([component])

    filtered = [sorted(line, key=lambda item: item[0]) for line in lines if len(line) >= 2]
    filtered.sort(key=lambda line: float(np.mean([item[1] + item[3] / 2.0 for item in line])))
    return filtered[:2]


def line_gap_filter(line: list[tuple[int, int, int, int, int]]) -> list[tuple[int, int, int, int, int]]:
    if len(line) <= 2:
        return line

    widths = np.array([w for _, _, w, _, _ in line], dtype=np.float32)
    heights = np.array([h for _, _, _, h, _ in line], dtype=np.float32)
    median_width = float(np.median(widths))
    median_height = float(np.median(heights))
    filtered: list[tuple[int, int, int, int, int]] = []

    for component in line:
        x, y, w, h, area = component
        if h < median_height * 0.45:
            continue
        if w < max(2.0, median_width * 0.22) and h < median_height * 0.75:
            continue
        if area < median_width * median_height * 0.08:
            continue
        filtered.append(component)

    return filtered


def trim_plate_line_counts(lines: list[list[tuple[int, int, int, int, int]]]) -> list[list[tuple[int, int, int, int, int]]]:
    if len(lines) != 2:
        return lines

    trimmed: list[list[tuple[int, int, int, int, int]]] = []
    expected_counts = (3, 5)
    for line, expected in zip(lines, expected_counts):
        current = list(line)
        while len(current) > expected:
            widths = np.array([item[2] for item in current], dtype=np.float32)
            median_width = float(np.median(widths))
            edge_indices = [0, len(current) - 1]
            narrow_edges = [
                index for index in edge_indices if current[index][2] <= median_width * 0.72
            ]
            remove_index = narrow_edges[0] if narrow_edges else min(edge_indices, key=lambda index: current[index][4])
            current.pop(remove_index)
        trimmed.append(current)
    return trimmed


def score_lines(lines: list[list[tuple[int, int, int, int, int]]], image_shape: tuple[int, int]) -> float:
    height, width = image_shape
    if not lines:
        return -1.0

    total = sum(len(line) for line in lines)
    if total < 4 or total > 10:
        return -1.0

    line_counts = [len(line) for line in lines]
    count_score = 1.0 - min(abs(total - 8) / 5.0, 1.0)
    if len(lines) == 2:
        top_ok = 2 <= line_counts[0] <= 4
        bottom_ok = 4 <= line_counts[1] <= 6
        layout_score = 1.2 if top_ok and bottom_ok else 0.2
    else:
        layout_score = 1.0 if 7 <= total <= 9 else 0.4

    heights = np.array([h for line in lines for _, _, _, h, _ in line], dtype=np.float32)
    widths = np.array([w for line in lines for _, _, w, _, _ in line], dtype=np.float32)
    height_consistency = 1.0 - min(float(heights.std()) / float(max(1.0, heights.mean())), 1.0)
    width_consistency = 1.0 - min(float(widths.std()) / float(max(1.0, widths.mean() * 1.6)), 1.0)

    x1 = min(x for line in lines for x, _, _, _, _ in line)
    x2 = max(x + w for line in lines for x, _, w, _, _ in line)
    y1 = min(y for line in lines for _, y, _, _, _ in line)
    y2 = max(y + h for line in lines for _, y, _, h, _ in line)
    coverage_x = (x2 - x1) / float(max(1, width))
    coverage_y = (y2 - y1) / float(max(1, height))
    coverage_score = 1.0 if 0.35 <= coverage_x <= 0.92 and 0.18 <= coverage_y <= 0.85 else 0.2

    return 2.0 * count_score + layout_score + height_consistency + 0.5 * width_consistency + coverage_score


def segment_plate_characters(plate_image: np.ndarray) -> list[CharacterSegment]:
    if plate_image.size == 0:
        return []

    resized, scale = resize_plate_for_ocr(plate_image)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    best_score = -1.0
    best_mask: np.ndarray | None = None
    best_lines: list[list[tuple[int, int, int, int, int]]] = []

    for mask in ocr_masks(gray):
        components = extract_component_bboxes(mask)
        if not components:
            continue

        widths = np.array([w for _, _, w, _, _ in components], dtype=np.float32)
        components = split_wide_components(components, float(np.median(widths)))
        lines = [line_gap_filter(line) for line in cluster_lines(components)]
        lines = [line for line in lines if len(line) >= 2]
        lines = trim_plate_line_counts(lines)
        score = score_lines(lines, gray.shape[:2])
        if score > best_score:
            best_score = score
            best_lines = lines
            best_mask = mask

    if best_mask is None or not best_lines:
        return []

    segments: list[CharacterSegment] = []
    for line_index, line in enumerate(best_lines):
        for char_index, (x, y, w, h, _) in enumerate(line):
            pad_x = max(1, int(round(w * 0.12)))
            pad_y = max(1, int(round(h * 0.10)))
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(best_mask.shape[1], x + w + pad_x)
            y2 = min(best_mask.shape[0], y + h + pad_y)
            crop = best_mask[y1:y2, x1:x2]

            original_bbox = (
                int(round(x1 / scale)),
                int(round(y1 / scale)),
                max(1, int(round((x2 - x1) / scale))),
                max(1, int(round((y2 - y1) / scale))),
            )
            segments.append(CharacterSegment(line_index, char_index, original_bbox, crop))

    return segments


def segment_lines(segments: list[CharacterSegment]) -> list[list[CharacterSegment]]:
    lines: list[list[CharacterSegment]] = []
    for segment in segments:
        while len(lines) <= segment.line_index:
            lines.append([])
        lines[segment.line_index].append(segment)
    return [sorted(line, key=lambda item: item.bbox[0]) for line in lines if line]


def plate_layout_score(plate_image: np.ndarray) -> tuple[float, int, list[int]]:
    segments = segment_plate_characters(plate_image)
    lines = segment_lines(segments)
    counts = [len(line) for line in lines]
    total = sum(counts)
    if total == 0:
        return -2.0, 0, counts

    score = 0.0
    if len(lines) == 1:
        score += 2.4 if 7 <= total <= 9 else -1.0
    elif len(lines) == 2:
        top, bottom = counts
        if 2 <= top <= 4 and 4 <= bottom <= 6 and 7 <= total <= 9:
            score += 2.8
        else:
            score -= 1.4
    else:
        score -= 2.0

    if total < 7:
        score -= 3.5
    if total < 5:
        score -= 1.5
    if total > 9:
        score -= min(2.5, (total - 9) * 0.45)

    if lines:
        heights = np.array([segment.bbox[3] for line in lines for segment in line], dtype=np.float32)
        if heights.size > 1:
            score += 0.6 * (1.0 - min(float(heights.std()) / float(max(1.0, heights.mean())), 1.0))

    return score, total, counts


def hog_feature(mask: np.ndarray) -> np.ndarray:
    normalized = normalize_binary_char(mask)
    hog = cv2.HOGDescriptor(
        _winSize=CHAR_SIZE,
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    feature = hog.compute(normalized).reshape(-1).astype(np.float32)
    norm = float(np.linalg.norm(feature))
    if norm > 1e-6:
        feature /= norm
    pixels = (normalized.reshape(-1).astype(np.float32) / 255.0)
    return np.concatenate([feature, pixels])


def render_template(char: str, font: int, scale: float, thickness: int, shift_x: int, shift_y: int, rotate: float) -> np.ndarray:
    canvas = np.zeros((72, 52), dtype=np.uint8)
    (text_w, text_h), baseline = cv2.getTextSize(char, font, scale, thickness)
    x = (canvas.shape[1] - text_w) // 2 + shift_x
    y = (canvas.shape[0] + text_h) // 2 + shift_y
    cv2.putText(canvas, char, (x, y), font, scale, 255, thickness, cv2.LINE_AA)
    if not math.isclose(rotate, 0.0):
        matrix = cv2.getRotationMatrix2D((canvas.shape[1] / 2.0, canvas.shape[0] / 2.0), rotate, 1.0)
        canvas = cv2.warpAffine(canvas, matrix, (canvas.shape[1], canvas.shape[0]), flags=cv2.INTER_LINEAR)
    return normalize_binary_char(canvas)


def render_ttf_template(char: str, font_path: Path, size: int, shift_x: int, shift_y: int, rotate: float, x_scale: float) -> np.ndarray:
    if Image is None or ImageDraw is None or ImageFont is None:
        return np.zeros((CHAR_SIZE[1], CHAR_SIZE[0]), dtype=np.uint8)

    image = Image.new("L", (64, 88), 0)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), size=size)
    bbox = draw.textbbox((0, 0), char, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (image.width - text_w) // 2 - bbox[0] + shift_x
    y = (image.height - text_h) // 2 - bbox[1] + shift_y
    draw.text((x, y), char, fill=255, font=font)

    if not math.isclose(x_scale, 1.0):
        scaled_w = max(1, int(round(image.width * x_scale)))
        image = image.resize((scaled_w, image.height), Image.Resampling.BICUBIC)
        canvas = Image.new("L", (64, 88), 0)
        paste_x = (canvas.width - image.width) // 2
        canvas.paste(image, (paste_x, 0))
        image = canvas

    if not math.isclose(rotate, 0.0):
        image = image.rotate(rotate, resample=Image.Resampling.BICUBIC, fillcolor=0)

    return normalize_binary_char(np.array(image, dtype=np.uint8))


@lru_cache(maxsize=1)
def template_features() -> tuple[np.ndarray, np.ndarray]:
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
    ]
    labels: list[str] = []
    features: list[np.ndarray] = []

    font_paths = [Path(path) for path in FONT_CANDIDATES if Path(path).exists()][:4]
    for font_path in font_paths:
        for char in CHARSET:
            for size in (48, 56, 64):
                for shift_x, shift_y in ((0, 0), (-2, 0), (2, 0)):
                    for rotate in (-4.0, 0.0, 4.0):
                        for x_scale in (0.85, 1.0):
                            template = render_ttf_template(char, font_path, size, shift_x, shift_y, rotate, x_scale)
                            features.append(hog_feature(template))
                            labels.append(char)

                            if size == 56 and rotate == 0.0 and shift_x == 0 and shift_y == 0:
                                eroded = cv2.erode(template, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
                                dilated = cv2.dilate(template, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
                                features.append(hog_feature(eroded))
                                labels.append(char)
                                features.append(hog_feature(dilated))
                                labels.append(char)

    if not font_paths:
        for char in CHARSET:
            for font in fonts:
                for scale in (1.15, 1.35, 1.55):
                    for thickness in (2, 3):
                        for shift_x, shift_y in ((0, 0), (-2, 0), (2, 0), (0, -2), (0, 2)):
                            for rotate in (-5.0, 0.0, 5.0):
                                template = render_template(char, font, scale, thickness, shift_x, shift_y, rotate)
                                features.append(hog_feature(template))
                                labels.append(char)

    return np.vstack(features).astype(np.float32), np.array(labels)


def load_char_model(path: Path) -> CharModel:
    data = np.load(path, allow_pickle=False)
    return {
        "features": data["features"].astype(np.float32),
        "labels": data["labels"].astype(str),
    }


def classify_with_model(query: np.ndarray, allowed: str, model: CharModel, k: int = 7) -> tuple[str, float] | None:
    features = model["features"]
    labels = model["labels"].astype(str)
    allowed_mask = np.array([label in allowed for label in labels], dtype=bool)
    if not bool(np.any(allowed_mask)):
        return None

    allowed_features = features[allowed_mask]
    allowed_labels = labels[allowed_mask]
    distances = np.linalg.norm(allowed_features - query.reshape(1, -1), axis=1)
    order = np.argsort(distances)[: max(k, min(20, len(distances)))]

    votes: dict[str, list[float]] = {}
    for index in order:
        label = str(allowed_labels[index])
        votes.setdefault(label, []).append(float(distances[index]))
    ranked = sorted(votes.items(), key=lambda item: float(np.mean(item[1])))
    if not ranked:
        return None

    best_label, best_distances = ranked[0]
    best_distance = float(np.mean(best_distances))
    second_distance = float(np.mean(ranked[1][1])) if len(ranked) > 1 else best_distance + 1.0
    confidence = max(0.0, min(1.0, (second_distance - best_distance + 0.10) / 0.45))
    return best_label, confidence


def digit_shape_override(mask: np.ndarray, predicted: str, confidence: float) -> tuple[str, float]:
    if predicted not in DIGITS:
        return predicted, confidence

    normalized = normalize_binary_char(mask)
    height, width = normalized.shape[:2]
    points = cv2.findNonZero(normalized)
    if points is None:
        return predicted, confidence
    _, _, foreground_w, foreground_h = cv2.boundingRect(points)
    foreground_aspect = foreground_w / float(max(1, foreground_h))

    contours, hierarchy = cv2.findContours(normalized, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    holes = 0
    if hierarchy is not None:
        for index, item in enumerate(hierarchy[0]):
            if item[3] != -1 and cv2.contourArea(contours[index]) > 8:
                holes += 1

    def density(y1: float, y2: float, x1: float, x2: float) -> float:
        roi = normalized[int(y1 * height) : int(y2 * height), int(x1 * width) : int(x2 * width)]
        return cv2.countNonZero(roi) / float(max(1, roi.size))

    lower_left = density(0.55, 0.85, 0.02, 0.42)
    middle = density(0.40, 0.60, 0.20, 0.82)
    top = density(0.05, 0.28, 0.05, 0.95)
    if predicted == "8" and holes <= 1 and lower_left > 0.52 and middle > 0.68 and confidence < 0.90:
        return "6", max(confidence, 0.74)
    if predicted == "0" and holes >= 1 and lower_left < 0.22 and density(0.10, 0.45, 0.02, 0.42) > 0.35:
        return "9", max(confidence, 0.74)
    if predicted == "0" and holes == 1 and middle > 0.64 and lower_left > 0.38 and confidence < 0.88:
        return "6", max(confidence, 0.74)
    if predicted == "8" and holes == 0 and confidence < 0.85:
        return "0", max(confidence, 0.70)
    if predicted == "0" and holes >= 1 and middle > 0.62 and lower_left < 0.32:
        return "9", max(confidence, 0.72)
    if predicted == "5" and holes >= 1:
        return "6", max(confidence, 0.70)
    if predicted == "6" and holes == 0 and lower_left < 0.35:
        return "5", max(confidence, 0.70)
    if predicted == "1" and top > 0.24 and confidence < 0.95:
        return "7", max(confidence, 0.70)
    if predicted == "1" and foreground_aspect >= 0.42 and confidence < 0.92:
        return "3", max(confidence, 0.70)
    return predicted, confidence


def letter_shape_override(mask: np.ndarray, predicted: str, confidence: float) -> tuple[str, float]:
    if predicted not in LETTERS:
        return predicted, confidence

    normalized = normalize_binary_char(mask)
    height, width = normalized.shape[:2]

    def density(y1: float, y2: float, x1: float, x2: float) -> float:
        roi = normalized[int(y1 * height) : int(y2 * height), int(x1 * width) : int(x2 * width)]
        return cv2.countNonZero(roi) / float(max(1, roi.size))

    left = density(0.18, 0.85, 0.02, 0.35)
    right = density(0.18, 0.85, 0.65, 0.98)
    middle = density(0.38, 0.62, 0.12, 0.88)
    lower_right = density(0.55, 0.85, 0.55, 0.98)
    top = density(0.05, 0.30, 0.10, 0.90)
    center = density(0.10, 0.90, 0.35, 0.65)
    if predicted == "T" and middle > 0.45 and left > 0.25 and right < 0.30:
        return "F", max(confidence, 0.72)
    if predicted == "L" and center > 0.65 and top > 0.20:
        return "A", max(confidence, 0.72)
    if predicted == "N" and left > 0.35 and right > 0.35 and middle > 0.50:
        return "H", max(confidence, 0.72)
    return predicted, confidence


def classify_character(mask: np.ndarray, allowed: str, model: CharModel | None = None) -> tuple[str, float]:
    query = hog_feature(mask)
    if model is not None:
        result = classify_with_model(query, allowed, model)
        if result is not None:
            if set(allowed).issubset(set(DIGITS)):
                return digit_shape_override(mask, result[0], result[1])
            if set(allowed).issubset(set(LETTERS)):
                return letter_shape_override(mask, result[0], result[1])
            return result

    features, labels = template_features()
    allowed_mask = np.array([label in allowed for label in labels], dtype=bool)
    allowed_features = features[allowed_mask]
    allowed_labels = labels[allowed_mask]

    distances = np.linalg.norm(allowed_features - query.reshape(1, -1), axis=1)
    order = np.argsort(distances)[:15]
    votes: dict[str, list[float]] = {}
    for index in order:
        label = str(allowed_labels[index])
        votes.setdefault(label, []).append(float(distances[index]))

    ranked = sorted(votes.items(), key=lambda item: float(np.mean(item[1])))
    best_label, best_distances = ranked[0]
    best_distance = float(np.mean(best_distances))
    second_distance = float(np.mean(ranked[1][1])) if len(ranked) > 1 else best_distance + 1.0
    confidence = max(0.0, min(1.0, (second_distance - best_distance + 0.15) / 0.6))
    if set(allowed).issubset(set(DIGITS)):
        return digit_shape_override(mask, best_label, confidence)
    if set(allowed).issubset(set(LETTERS)):
        return letter_shape_override(mask, best_label, confidence)
    return best_label, confidence


def allowed_chars_for_position(lines: list[list[CharacterSegment]], line_index: int, char_index: int) -> str:
    if len(lines) == 2:
        if line_index == 0:
            if char_index in (0, 1):
                return DIGITS
            if char_index == 2:
                return LETTERS
            return DIGITS + LETTERS
        return DIGITS

    flat_index = char_index
    total = len(lines[0]) if lines else 0
    if total >= 7:
        if flat_index in (0, 1):
            return DIGITS
        if flat_index == 2:
            return LETTERS
        return DIGITS
    return CHARSET


def format_plate_text(lines: list[list[str]]) -> tuple[str, str]:
    compact_lines = ["".join(line) for line in lines if line]
    compact = "".join(compact_lines)
    if not compact:
        return "", ""

    if len(compact_lines) == 2:
        top, bottom = compact_lines[0][:3], compact_lines[1]
        if len(top) >= 3 and len(bottom) >= 5:
            bottom = bottom[:5]
            return f"{top}-{bottom[:3]}.{bottom[3:]}", top + bottom
        if len(top) >= 3 and len(bottom) >= 4:
            bottom = bottom[:4]
            return f"{top}-{bottom}", top + bottom
        if len(top) >= 3 and bottom:
            return f"{top}-{bottom}", top + bottom
        return top, compact

    line = compact_lines[0]
    if len(line) == 8 and line[3:7] == "0000":
        line = line[:3] + line[4:]
        return f"{line[:3]}-{line[3:]}", line
    if len(line) >= 8:
        line = line[:8]
        return f"{line[:3]}-{line[3:6]}.{line[6:]}", line
    if len(line) == 7:
        return f"{line[:3]}-{line[3:]}", compact
    if len(line) >= 6:
        return f"{line[:3]}-{line[3:]}", compact
    return line, compact


def clean_ocr_text(value: str) -> str:
    return "".join(char for char in value.upper() if char in OCR_ALLOWED)


def normalize_digit_text(value: str) -> str:
    return clean_ocr_text(value).translate(DIGIT_NORMALIZE).replace("-", "").replace(".", "")


def normalize_top_text(value: str) -> str:
    clean = clean_ocr_text(value).replace("-", "").replace(".", "")
    if len(clean) < 3:
        return clean
    first = normalize_digit_text(clean[:2])[:2]
    third = clean[2]
    third = LETTER_NORMALIZE.get(third, third)
    if first == "80":
        first = "30"
    return first + third


def format_compact_plate(compact: str) -> tuple[str, str]:
    compact = clean_ocr_text(compact).replace("-", "").replace(".", "")
    if len(compact) == 7 and compact[:2].isdigit() and compact[2:].isdigit():
        top = normalize_digit_text(compact[:2])[:2] + "A"
        bottom = normalize_digit_text(compact[2:])[:5]
        if len(bottom) == 5:
            return f"{top}-{bottom[:3]}.{bottom[3:]}", top + bottom
    if len(compact) == 8:
        top = normalize_top_text(compact[:3])
        bottom = normalize_digit_text(compact[3:])[:5]
        if len(top) == 3 and len(bottom) == 5 and bottom[:4] == "0000":
            bottom = bottom[1:]
            return f"{top}-{bottom}", top + bottom
    if len(compact) >= 8:
        top = normalize_top_text(compact[:3])
        bottom = normalize_digit_text(compact[3:])[:5]
        if len(top) == 3 and len(bottom) >= 5:
            return f"{top}-{bottom[:3]}.{bottom[3:5]}", top + bottom[:5]
    if len(compact) == 7:
        top = normalize_top_text(compact[:3])
        bottom = normalize_digit_text(compact[3:])[:4]
        if len(top) == 3 and len(bottom) == 4:
            return f"{top}-{bottom}", top + bottom
    return "", ""


@lru_cache(maxsize=1)
def easyocr_reader():
    import easyocr

    return easyocr.Reader(["en"], gpu=False, verbose=False)


def recognize_plate_easyocr_text(plate_image: np.ndarray, try_rotations: bool = False) -> tuple[str, str, float]:
    candidates: list[tuple[str, str, float]] = []

    variants = [plate_image]
    if try_rotations and plate_image.size:
        height, width = plate_image.shape[:2]
        center = (width / 2.0, height / 2.0)
        for angle in (-10.0, -8.0, -5.0, -3.0, 3.0, 5.0, 8.0, 10.0):
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            variants.append(
                cv2.warpAffine(
                    plate_image,
                    matrix,
                    (width, height),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE,
                )
            )

    for variant in variants:
        text, compact, confidence = recognize_plate_easyocr_text_once(variant)
        if text:
            candidates.append((text, compact, confidence))

    if not candidates:
        return "", "", 0.0

    def rank(item: tuple[str, str, float]) -> tuple[int, float]:
        _, compact, confidence = item
        valid_length = 1 if len(compact) in {7, 8} else 0
        return valid_length, confidence

    return max(candidates, key=rank)


def recognize_plate_easyocr_text_once(plate_image: np.ndarray) -> tuple[str, str, float]:
    try:
        results = easyocr_reader().readtext(
            plate_image,
            allowlist=OCR_ALLOWED,
            detail=1,
            paragraph=False,
            decoder="beamsearch",
        )
    except Exception:
        return "", "", 0.0

    parsed: list[tuple[float, float, str, float]] = []
    for box, text, confidence in results:
        clean = clean_ocr_text(str(text))
        if not clean:
            continue
        points = np.array(box, dtype=np.float32)
        parsed.append((float(points[:, 1].mean()), float(points[:, 0].mean()), clean, float(confidence)))

    if not parsed:
        return "", "", 0.0

    parsed.sort(key=lambda item: (item[0], item[1]))
    if len(parsed) >= 2:
        top = normalize_top_text(parsed[0][2])
        bottom = normalize_digit_text("".join(item[2] for item in parsed[1:]))[:5]
        if len(top) >= 3 and len(bottom) >= 5:
            text, compact = format_plate_text([list(top[:3]), list(bottom[:5])])
            confidence = float(np.mean([item[3] for item in parsed]))
            return text, compact, confidence

    joined = "".join(item[2] for item in parsed)
    text, compact = format_compact_plate(joined)
    confidence = float(np.mean([item[3] for item in parsed]))
    return text, compact, confidence if text else 0.0


def recognize_plate(plate_image: np.ndarray, model: CharModel | None = None) -> PlateOCRResult:
    segments = segment_plate_characters(plate_image)
    grouped = segment_lines(segments)

    recognized: list[RecognizedCharacter] = []
    text_lines: list[list[str]] = []
    confidences: list[float] = []
    for line_index, line in enumerate(grouped):
        text_line: list[str] = []
        for char_index, segment in enumerate(line):
            allowed = allowed_chars_for_position(grouped, line_index, char_index)
            char, confidence = classify_character(segment.crop, allowed, model)
            recognized.append(
                RecognizedCharacter(
                    line_index=line_index,
                    char_index=char_index,
                    bbox=segment.bbox,
                    char=char,
                    confidence=confidence,
                    crop=segment.crop,
                )
            )
            text_line.append(char)
            confidences.append(confidence)
        text_lines.append(text_line)

    text, compact = format_plate_text(text_lines)
    confidence = float(np.mean(confidences)) if confidences else 0.0
    return PlateOCRResult(text=text, compact_text=compact, confidence=confidence, characters=recognized)
