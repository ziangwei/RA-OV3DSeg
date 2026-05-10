from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PointAugmentationConfig:
    rotation_z_max_rad: float = 3.141592653589793
    flip_x_prob: float = 0.5
    flip_y_prob: float = 0.5
    scale_min: float = 0.95
    scale_max: float = 1.05
    dropout_prob: float = 0.1


def augment_point_xyz(
    point_xyz: np.ndarray,
    rng: np.random.Generator,
    config: PointAugmentationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply basic LiDAR-only augmentation and return kept-point mask.

    This should be used for supervised 3D training. It should not be combined
    with nonzero 2D feature/logit distillation because image-projected teacher
    signals correspond to the original, unaugmented point coordinates.
    """

    if point_xyz.ndim != 2 or point_xyz.shape[1] != 3:
        raise ValueError(f"point_xyz must have shape [N, 3], got {point_xyz.shape}")
    augmented = point_xyz.astype(np.float32, copy=True)

    max_angle = float(max(config.rotation_z_max_rad, 0.0))
    if max_angle > 0:
        angle = float(rng.uniform(-max_angle, max_angle))
        cos_a = np.float32(np.cos(angle))
        sin_a = np.float32(np.sin(angle))
        x = augmented[:, 0].copy()
        y = augmented[:, 1].copy()
        augmented[:, 0] = x * cos_a - y * sin_a
        augmented[:, 1] = x * sin_a + y * cos_a

    if rng.random() < float(config.flip_x_prob):
        augmented[:, 0] *= -1.0
    if rng.random() < float(config.flip_y_prob):
        augmented[:, 1] *= -1.0

    scale_min = float(config.scale_min)
    scale_max = float(config.scale_max)
    if scale_max < scale_min:
        raise ValueError("scale_max must be >= scale_min")
    if scale_min > 0 and scale_max > 0 and scale_max != 1.0:
        scale = float(rng.uniform(scale_min, scale_max))
        augmented *= np.float32(scale)

    dropout_prob = float(np.clip(config.dropout_prob, 0.0, 0.95))
    keep_mask = np.ones(augmented.shape[0], dtype=bool)
    if dropout_prob > 0 and augmented.shape[0] > 1:
        keep_mask = rng.random(augmented.shape[0]) >= dropout_prob
        if not np.any(keep_mask):
            keep_mask[int(rng.integers(0, augmented.shape[0]))] = True

    return augmented, keep_mask
