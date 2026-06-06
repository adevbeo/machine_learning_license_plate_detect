from __future__ import annotations

import cv2
import numpy as np

from .config import HOGConfig


class HOGFeatureExtractor:
    def __init__(self, config: HOGConfig | None = None) -> None:
        self.config = config or HOGConfig()
        self.descriptor = cv2.HOGDescriptor(
            _winSize=self.config.window_size,
            _blockSize=self.config.block_size,
            _blockStride=self.config.block_stride,
            _cellSize=self.config.cell_size,
            _nbins=self.config.bins,
        )

    @property
    def feature_length(self) -> int:
        return int(self.descriptor.getDescriptorSize())

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        if image.size == 0:
            raise ValueError("Cannot extract HOG from an empty image")

        if image.ndim == 2:
            gray = image.copy()
        elif image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray = cv2.resize(gray, self.config.window_size, interpolation=cv2.INTER_AREA)
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid_size,
        )
        gray = clahe.apply(gray)
        if self.config.blur_kernel[0] > 1 and self.config.blur_kernel[1] > 1:
            gray = cv2.GaussianBlur(gray, self.config.blur_kernel, 0)
        return gray

    def extract(self, image: np.ndarray) -> np.ndarray:
        gray = self.preprocess(image)
        feature = self.descriptor.compute(gray)
        if feature is None:
            raise ValueError("OpenCV HOGDescriptor returned no features")
        return feature.reshape(-1).astype(np.float32)

    def extract_batch(self, images: list[np.ndarray]) -> np.ndarray:
        if not images:
            return np.empty((0, self.feature_length), dtype=np.float32)
        return np.vstack([self.extract(image) for image in images]).astype(np.float32)
