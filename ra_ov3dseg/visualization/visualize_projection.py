from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ra_ov3dseg.utils.io import ensure_dir


def save_projection_overlay(
    image_path: str | Path,
    uv: np.ndarray,
    depth: np.ndarray,
    valid_mask: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    """将投影点绘制到相机图像上，并按深度上色。"""

    image_path = Path(image_path)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    with Image.open(image_path) as image:
        image_np = np.asarray(image.convert("RGB"))

    valid_uv = uv[valid_mask]
    valid_depth = depth[valid_mask]

    figure_width = max(image_np.shape[1] / 220.0, 6.0)
    figure_height = max(image_np.shape[0] / 220.0, 4.0)
    fig, ax = plt.subplots(figsize=(figure_width, figure_height), dpi=150)
    ax.imshow(image_np)

    if valid_uv.shape[0] > 0:
        depth_min = float(np.nanmin(valid_depth))
        depth_max = float(np.nanmax(valid_depth))
        if abs(depth_max - depth_min) < 1e-6:
            depth_max = depth_min + 1e-6

        scatter = ax.scatter(
            valid_uv[:, 0],
            valid_uv[:, 1],
            c=valid_depth,
            cmap="turbo_r",
            s=3.0,
            alpha=0.85,
            linewidths=0.0,
        )
        scatter.set_clim(depth_min, depth_max)
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Depth (m)")

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
