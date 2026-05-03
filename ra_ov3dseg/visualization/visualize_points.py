from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ra_ov3dseg.utils.io import ensure_dir


def build_color_palette(num_classes: int) -> np.ndarray:
    cmap = plt.get_cmap("gist_ncar")
    colors = cmap(np.linspace(0.0, 1.0, max(num_classes, 1)))[:, :3]
    return (colors * 255.0).astype(np.uint8)


def labels_to_colors(label_indices: np.ndarray, num_classes: int) -> np.ndarray:
    palette = build_color_palette(num_classes)
    colors = np.full((label_indices.shape[0], 3), 160, dtype=np.uint8)
    valid_mask = label_indices >= 0
    if np.any(valid_mask):
        colors[valid_mask] = palette[label_indices[valid_mask] % max(num_classes, 1)]
    return colors


def save_point_cloud_ply(
    point_xyz: np.ndarray,
    label_indices: np.ndarray,
    output_path: str | Path,
    valid_mask: np.ndarray | None = None,
    max_points: int | None = None,
    num_classes: int | None = None,
) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    if valid_mask is None:
        valid_mask = np.ones(point_xyz.shape[0], dtype=bool)
    keep_indices = np.nonzero(valid_mask)[0]
    if max_points is not None and keep_indices.shape[0] > max_points:
        keep_indices = np.linspace(0, keep_indices.shape[0] - 1, num=max_points, dtype=np.int64)
        keep_indices = np.nonzero(valid_mask)[0][keep_indices]

    point_xyz = point_xyz[keep_indices]
    label_indices = label_indices[keep_indices]

    if num_classes is None:
        num_classes = int(np.max(label_indices[label_indices >= 0]) + 1) if np.any(label_indices >= 0) else 1
    colors = labels_to_colors(label_indices, num_classes=num_classes)

    with output_path.open("w", encoding="utf-8") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {point_xyz.shape[0]}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")
        for point, color in zip(point_xyz, colors):
            file.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )

    return output_path


def save_bev_prediction_plot(
    point_xyz: np.ndarray,
    label_indices: np.ndarray,
    output_path: str | Path,
    valid_mask: np.ndarray | None = None,
    max_points: int | None = 40000,
    num_classes: int | None = None,
) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    if valid_mask is None:
        valid_mask = np.ones(point_xyz.shape[0], dtype=bool)
    keep_indices = np.nonzero(valid_mask)[0]
    if max_points is not None and keep_indices.shape[0] > max_points:
        keep_indices = np.linspace(0, keep_indices.shape[0] - 1, num=max_points, dtype=np.int64)
        keep_indices = np.nonzero(valid_mask)[0][keep_indices]

    point_xyz = point_xyz[keep_indices]
    label_indices = label_indices[keep_indices]

    if num_classes is None:
        num_classes = int(np.max(label_indices[label_indices >= 0]) + 1) if np.any(label_indices >= 0) else 1
    colors = labels_to_colors(label_indices, num_classes=num_classes).astype(np.float32) / 255.0

    fig, ax = plt.subplots(figsize=(8, 8), dpi=180)
    ax.scatter(
        point_xyz[:, 0],
        point_xyz[:, 1],
        c=colors,
        s=0.8,
        alpha=0.8,
        linewidths=0.0,
    )
    ax.set_title("Zero-shot Point Prediction (BEV)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
