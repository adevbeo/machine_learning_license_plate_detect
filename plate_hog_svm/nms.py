from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Chỉ import lúc type-checking để tránh circular import tại runtime.
    # nms.py được import bởi detector.py, nên không thể import Detection tại top-level.
    from .detector import Detection


@dataclass(frozen=True)
class _ScoredBox:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    detection: Any  # Thực tế là Detection, dùng Any để tránh circular import


def _bbox_to_scored(det: Any) -> _ScoredBox:
    x, y, w, h = det.bbox
    return _ScoredBox(int(x), int(y), int(x + w), int(y + h), float(det.score), det)


def compute_iou(a: _ScoredBox, b: _ScoredBox) -> float:
    """Tính Intersection over Union giữa 2 box theo tọa độ (x1, y1, x2, y2)."""
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    area_a = max(0, a.x2 - a.x1) * max(0, a.y2 - a.y1)
    area_b = max(0, b.x2 - b.x1) * max(0, b.y2 - b.y1)
    union_area = area_a + area_b - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def non_max_suppression(
    detections: list[Any],
    iou_threshold: float = 0.3,
    top_k: int = 1,
) -> list[Any]:
    """Áp dụng Non-Maximum Suppression lên danh sách Detection.

    Thuật toán:
    1. Sắp xếp detection theo score giảm dần.
    2. Duyệt từng detection: nếu chưa bị suppress thì giữ lại.
    3. Suppress các detection còn lại có IoU > iou_threshold với detection vừa giữ.
    4. Dừng khi đã giữ được top_k detection.

    Args:
        detections: Danh sách Detection (hoặc bất kỳ object nào có `.bbox` và `.score`).
        iou_threshold: Ngưỡng IoU để loại box trùng lặp (0–1).
        top_k: Số detection tối đa trả về.

    Returns:
        Danh sách tối đa top_k detection sau NMS, sắp xếp theo score giảm dần.
    """
    if not detections:
        return []

    scored = sorted(
        [_bbox_to_scored(d) for d in detections],
        key=lambda s: s.score,
        reverse=True,
    )

    kept: list[Any] = []
    suppressed = [False] * len(scored)

    for i, box in enumerate(scored):
        if suppressed[i]:
            continue
        kept.append(box.detection)
        if len(kept) >= top_k:
            break
        for j in range(i + 1, len(scored)):
            if not suppressed[j] and compute_iou(box, scored[j]) > iou_threshold:
                suppressed[j] = True

    return kept
