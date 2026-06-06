from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class HOGConfig:
    window_size: tuple[int, int] = (96, 64)
    block_size: tuple[int, int] = (16, 16)
    block_stride: tuple[int, int] = (8, 8)
    cell_size: tuple[int, int] = (8, 8)
    bins: int = 9
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (4, 4)
    blur_kernel: tuple[int, int] = (3, 3)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "HOGConfig":
        if not value:
            return cls()

        def tuple2(name: str, default: tuple[int, int]) -> tuple[int, int]:
            raw = value.get(name, default)
            return int(raw[0]), int(raw[1])

        return cls(
            window_size=tuple2("window_size", cls.window_size),
            block_size=tuple2("block_size", cls.block_size),
            block_stride=tuple2("block_stride", cls.block_stride),
            cell_size=tuple2("cell_size", cls.cell_size),
            bins=int(value.get("bins", cls.bins)),
            clahe_clip_limit=float(value.get("clahe_clip_limit", cls.clahe_clip_limit)),
            clahe_tile_grid_size=tuple2("clahe_tile_grid_size", cls.clahe_tile_grid_size),
            blur_kernel=tuple2("blur_kernel", cls.blur_kernel),
        )


@dataclass(frozen=True)
class DetectionConfig:
    max_width: int = 900
    aspect_ratios: tuple[float, ...] = (2.4, 2.8, 3.6, 4.5)
    window_heights: tuple[int, ...] = (32, 40, 48, 64, 80, 96)
    stride_ratio: float = 0.25
    score_threshold: float = 0.0
    batch_size: int = 256
