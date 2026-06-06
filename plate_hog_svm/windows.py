from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Window:
    x: int
    y: int
    w: int
    h: int

    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


def build_window_sizes(
    image_shape: tuple[int, int],
    heights: tuple[int, ...],
    aspect_ratios: tuple[float, ...],
) -> list[tuple[int, int]]:
    image_h, image_w = image_shape
    sizes: set[tuple[int, int]] = set()
    for height in heights:
        h = int(round(height))
        if h < 12 or h > image_h:
            continue
        for ratio in aspect_ratios:
            w = int(round(h * ratio))
            if w < 16 or w > image_w:
                continue
            sizes.add((w, h))
    return sorted(sizes, key=lambda item: (item[1], item[0]))


def iter_sliding_windows(
    image_shape: tuple[int, int],
    sizes: list[tuple[int, int]],
    stride_ratio: float,
) -> Iterator[Window]:
    image_h, image_w = image_shape
    for w, h in sizes:
        step_x = max(4, int(round(w * stride_ratio)))
        step_y = max(4, int(round(h * stride_ratio)))
        y_values = list(range(0, max(1, image_h - h + 1), step_y))
        x_values = list(range(0, max(1, image_w - w + 1), step_x))
        if y_values and y_values[-1] != image_h - h:
            y_values.append(image_h - h)
        if x_values and x_values[-1] != image_w - w:
            x_values.append(image_w - w)

        for y in y_values:
            for x in x_values:
                yield Window(x, y, w, h)
