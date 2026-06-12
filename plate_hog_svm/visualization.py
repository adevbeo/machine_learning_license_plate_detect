from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .dataset import crop_from_bbox, parse_label, row_image_path
from .features import HOGFeatureExtractor
from .io_utils import read_image


@dataclass(frozen=True)
class VisualizationSample:
    image_path: Path
    label: int
    image: np.ndarray


def _load_pyplot() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def load_labeled_preview_samples(
    labels_paths: Sequence[Path],
    max_samples: int = 12,
    verbose: bool = True,
) -> list[VisualizationSample]:
    """Load a small, balanced set of labeled crops for HOG visualization."""
    if max_samples <= 0:
        return []

    target_per_class = max(1, math.ceil(max_samples / 2))
    samples_by_label: dict[int, list[VisualizationSample]] = {1: [], -1: []}

    for labels_path in labels_paths:
        if not labels_path.exists():
            continue
        with labels_path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = parse_label(row.get("label", ""))
                if label not in samples_by_label:
                    continue
                if len(samples_by_label[label]) >= target_per_class:
                    continue

                image_path = row_image_path(labels_path, row)
                if image_path is None:
                    continue
                try:
                    image = crop_from_bbox(read_image(image_path), row)
                except Exception as exc:
                    if verbose:
                        print(f"[SKIP HOG VIS] {image_path}: {exc}", file=sys.stderr)
                    continue
                samples_by_label[label].append(VisualizationSample(image_path, label, image))

                total = sum(len(items) for items in samples_by_label.values())
                if total >= max_samples:
                    break

    samples = samples_by_label[1] + samples_by_label[-1]
    return samples[:max_samples]


def skimage_hog_visualization(
    image: np.ndarray,
    extractor: HOGFeatureExtractor,
) -> tuple[np.ndarray, np.ndarray]:
    """Return preprocessed grayscale image and skimage HOG visualization image."""
    from skimage.exposure import rescale_intensity
    from skimage.feature import hog

    gray = extractor.preprocess(image)
    config = extractor.config
    cells_per_block = (
        max(1, int(config.block_size[1] / config.cell_size[1])),
        max(1, int(config.block_size[0] / config.cell_size[0])),
    )
    pixels_per_cell = (
        max(1, int(config.cell_size[1])),
        max(1, int(config.cell_size[0])),
    )

    _, hog_image = hog(
        gray,
        orientations=config.bins,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        visualize=True,
        feature_vector=True,
    )
    hog_image = rescale_intensity(hog_image, in_range=(0, np.max(hog_image) or 1.0))
    return gray, hog_image


def save_hog_visualization_grid(
    samples: Sequence[VisualizationSample],
    extractor: HOGFeatureExtractor,
    output_path: Path,
) -> int:
    """Save a matplotlib figure showing source crops next to HOG images."""
    if not samples:
        return 0

    plt = _load_pyplot()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = len(samples)
    fig, axes = plt.subplots(rows, 2, figsize=(8, max(2.2, rows * 2.2)))
    if rows == 1:
        axes = np.array([axes])

    for row_index, sample in enumerate(samples):
        gray, hog_image = skimage_hog_visualization(sample.image, extractor)
        label_name = "plate" if sample.label == 1 else "non_plate"

        axes[row_index, 0].imshow(gray, cmap="gray")
        axes[row_index, 0].set_title(f"{label_name}: {sample.image_path.name}", fontsize=9)
        axes[row_index, 0].axis("off")

        axes[row_index, 1].imshow(hog_image, cmap="magma")
        axes[row_index, 1].set_title("skimage HOG visualize=True", fontsize=9)
        axes[row_index, 1].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return len(samples)


def _stratified_sample_indices(
    y: np.ndarray,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    if max_samples <= 0 or len(y) <= max_samples:
        return np.arange(len(y), dtype=np.int32)

    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    labels = sorted(np.unique(y).tolist())
    per_class = max(1, max_samples // max(1, len(labels)))

    for label in labels:
        indices = np.flatnonzero(y == label)
        take = min(len(indices), per_class)
        if take:
            selected.append(rng.choice(indices, size=take, replace=False))

    merged = np.concatenate(selected) if selected else np.empty((0,), dtype=np.int32)
    if len(merged) < max_samples:
        remaining = np.setdiff1d(np.arange(len(y), dtype=np.int32), merged, assume_unique=False)
        extra_count = min(len(remaining), max_samples - len(merged))
        if extra_count:
            merged = np.concatenate([merged, rng.choice(remaining, size=extra_count, replace=False)])

    rng.shuffle(merged)
    return merged.astype(np.int32)


def reduce_hog_features(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "pca",
    dims: int = 2,
    sample_size: int = 2000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Scale HOG features and reduce them to 2D/3D for visual inspection."""
    method = method.lower().strip()
    if method not in {"pca", "tsne"}:
        raise ValueError("method must be 'pca' or 'tsne'")
    if dims not in {2, 3}:
        raise ValueError("dims must be 2 or 3")
    if len(y) < 2:
        raise ValueError("Need at least 2 samples to draw feature embedding")

    indices = _stratified_sample_indices(y, sample_size, seed)
    x_plot = x[indices]
    y_plot = y[indices]
    x_scaled = StandardScaler().fit_transform(x_plot)

    info: dict[str, Any] = {
        "method": method,
        "dims": dims,
        "samples": int(len(indices)),
        "sample_size": int(sample_size),
    }

    if method == "pca":
        reducer = PCA(n_components=dims, random_state=seed)
        embedding = reducer.fit_transform(x_scaled)
        info["explained_variance_ratio"] = [
            float(value) for value in reducer.explained_variance_ratio_
        ]
    else:
        if len(indices) < 4:
            raise ValueError("Need at least 4 samples for t-SNE")
        perplexity = min(30.0, max(2.0, (len(indices) - 1) / 3.0))
        tsne_kwargs: dict[str, Any] = {
            "n_components": dims,
            "perplexity": perplexity,
            "init": "pca",
            "learning_rate": "auto",
            "random_state": seed,
        }
        try:
            reducer = TSNE(max_iter=1000, **tsne_kwargs)
        except TypeError:
            reducer = TSNE(n_iter=1000, **tsne_kwargs)
        embedding = reducer.fit_transform(x_scaled)
        info["perplexity"] = float(perplexity)

    return embedding.astype(np.float32), y_plot.astype(np.int32), info


def save_feature_embedding_plot(
    x: np.ndarray,
    y: np.ndarray,
    output_path: Path,
    method: str = "pca",
    dims: int = 2,
    sample_size: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Save a PCA/t-SNE scatter plot of HOG features before LinearSVM training."""
    embedding, labels, info = reduce_hog_features(
        x=x,
        y=y,
        method=method,
        dims=dims,
        sample_size=sample_size,
        seed=seed,
    )

    plt = _load_pyplot()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(8, 6))
    if dims == 3:
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = fig.add_subplot(111)

    class_styles = {
        -1: ("non_plate", "#dc2626"),
        1: ("plate", "#2563eb"),
    }
    for label, (name, color) in class_styles.items():
        mask = labels == label
        if not np.any(mask):
            continue
        if dims == 3:
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                embedding[mask, 2],
                s=16,
                c=color,
                alpha=0.75,
                label=name,
                edgecolors="none",
            )
        else:
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                s=18,
                c=color,
                alpha=0.75,
                label=name,
                edgecolors="none",
            )

    method_name = info["method"].upper()
    title = f"HOG feature embedding projected by {method_name} ({dims}D)"
    ax.set_title(title)

    if info["method"] == "pca":
        ax.set_xlabel("PCA component 1 - main HOG variation")
        ax.set_ylabel("PCA component 2 - second HOG variation")
    else:
        ax.set_xlabel("t-SNE dimension 1 - HOG neighborhood map")
        ax.set_ylabel("t-SNE dimension 2 - HOG neighborhood map")

    if dims == 3:
        if info["method"] == "pca":
            ax.set_zlabel("PCA component 3 - third HOG variation")
        else:
            ax.set_zlabel("t-SNE dimension 3 - HOG neighborhood map")
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.4, alpha=0.35)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    info["output_path"] = str(output_path)
    return info
