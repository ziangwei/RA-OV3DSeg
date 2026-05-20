from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ra_ov3dseg.utils.io import ensure_dir

for _numpy_alias, _numpy_value in {"Inf": np.inf, "Infinity": np.inf, "NaN": np.nan}.items():
    if not hasattr(np, _numpy_alias):
        setattr(np, _numpy_alias, _numpy_value)


def _prepare_overlay_points(
    uv: np.ndarray,
    depth: np.ndarray,
    valid_mask: np.ndarray,
    min_depth: float,
    max_depth: float | None,
    max_points: int | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    mask = valid_mask.astype(bool).copy()
    mask &= np.isfinite(uv[:, 0]) & np.isfinite(uv[:, 1]) & np.isfinite(depth)
    mask &= depth >= float(min_depth)
    if max_depth is not None:
        mask &= depth <= float(max_depth)

    selected_uv = uv[mask]
    selected_depth = depth[mask]
    original_count = int(selected_uv.shape[0])

    if selected_uv.shape[0] == 0:
        return selected_uv, selected_depth, original_count

    # 先按深度从远到近排序，再在最终绘制时让近处点压住远处点。
    order = np.argsort(selected_depth)[::-1]
    selected_uv = selected_uv[order]
    selected_depth = selected_depth[order]

    if max_points is not None and selected_uv.shape[0] > max_points:
        keep_indices = np.linspace(0, selected_uv.shape[0] - 1, num=max_points, dtype=np.int64)
        selected_uv = selected_uv[keep_indices]
        selected_depth = selected_depth[keep_indices]

    return selected_uv, selected_depth, original_count


def save_projection_overlay(
    image_path: str | Path,
    uv: np.ndarray,
    depth: np.ndarray,
    valid_mask: np.ndarray,
    output_path: str | Path,
    title: str,
    min_depth: float = 0.0,
    max_depth: float | None = None,
    max_points: int | None = None,
    point_size: float = 3.0,
    alpha: float = 0.85,
) -> dict[str, float | int]:
    """将投影点绘制到相机图像上，并按深度上色。

    这个函数主要服务于人工 sanity check：
    1. 支持只看近距离点，避免远处稠密点把整张图糊住。
    2. 支持限制绘制点数，让图更容易肉眼判断几何是否对齐。
    3. 默认让近处点覆盖在远处点上，更容易观察车辆、路面、路沿等局部区域。
    """

    image_path = Path(image_path)
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    with Image.open(image_path) as image:
        image_np = np.asarray(image.convert("RGB"))

    valid_uv, valid_depth, original_count = _prepare_overlay_points(
        uv=uv,
        depth=depth,
        valid_mask=valid_mask,
        min_depth=min_depth,
        max_depth=max_depth,
        max_points=max_points,
    )

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
            s=point_size,
            alpha=alpha,
            linewidths=0.0,
        )
        scatter.set_clim(depth_min, depth_max)
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Depth (m)")

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    return {
        "drawn_points": int(valid_uv.shape[0]),
        "candidate_points": int(original_count),
        "min_depth": float(np.nanmin(valid_depth)) if valid_depth.shape[0] > 0 else -1.0,
        "max_depth": float(np.nanmax(valid_depth)) if valid_depth.shape[0] > 0 else -1.0,
    }
