from __future__ import annotations

import numpy as np


def distance_weight(
    distances: np.ndarray,
    max_distance: float = 60.0,
    min_weight: float = 0.1,
) -> np.ndarray:
    """Distance reliability: close points are trusted more than far points."""

    normalized = np.clip(distances.astype(np.float32) / max_distance, 0.0, 1.0)
    return ((1.0 - normalized) * (1.0 - min_weight) + min_weight).astype(np.float32)


def boundary_weight(
    uv: np.ndarray,
    image_widths: np.ndarray,
    image_heights: np.ndarray,
    valid_masks: np.ndarray,
    margin_ratio: float = 0.05,
) -> np.ndarray:
    """Point reliability from distance to image borders.

    Points near image borders are less reliable because small calibration or sync errors can move
    them out of the image or onto a wrong semantic region.
    """

    num_cameras, num_points, _ = uv.shape
    per_camera_boundary = np.zeros((num_cameras, num_points), dtype=np.float32)

    for camera_idx in range(num_cameras):
        valid = valid_masks[camera_idx].astype(bool)
        if not np.any(valid):
            continue

        width = max(float(image_widths[camera_idx]), 1.0)
        height = max(float(image_heights[camera_idx]), 1.0)
        margin = max(min(width, height) * float(margin_ratio), 1.0)

        x = uv[camera_idx, :, 0]
        y = uv[camera_idx, :, 1]
        distance_to_border = np.minimum.reduce([x, y, width - 1.0 - x, height - 1.0 - y])
        per_camera_boundary[camera_idx, valid] = np.clip(distance_to_border[valid] / margin, 0.0, 1.0)

    visible_count = valid_masks.astype(np.int32).sum(axis=0)
    boundary_sum = per_camera_boundary.sum(axis=0)
    weights = np.zeros(num_points, dtype=np.float32)
    valid_points = visible_count > 0
    weights[valid_points] = boundary_sum[valid_points] / visible_count[valid_points].astype(np.float32)
    return weights


def geometric_weight(
    visible_camera_count: np.ndarray,
    boundary_weights: np.ndarray,
    max_cameras: int = 6,
    min_visible_weight: float = 0.5,
) -> np.ndarray:
    """Geometric reliability from visibility count and image-boundary safety."""

    visible_camera_count = visible_camera_count.astype(np.float32)
    valid_mask = visible_camera_count > 0
    visibility_bonus = np.clip(visible_camera_count / float(max(max_cameras, 1)), 0.0, 1.0)
    visibility_weight = min_visible_weight + (1.0 - min_visible_weight) * visibility_bonus
    return (valid_mask.astype(np.float32) * visibility_weight * np.clip(boundary_weights, 0.0, 1.0)).astype(
        np.float32
    )


def semantic_weight(
    max_similarity: np.ndarray,
    min_similarity: float = 0.0,
    max_similarity_value: float = 0.35,
) -> np.ndarray:
    """Semantic reliability from teacher/text similarity.

    Dense CLIP cosine scores can be low in absolute value, so the default upper bound is modest
    for MVP-v2 and should be tuned after observing real distributions.
    """

    denom = max(max_similarity_value - min_similarity, 1e-6)
    normalized = (max_similarity.astype(np.float32) - min_similarity) / denom
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def compute_point_reliability(
    distances: np.ndarray,
    visible_camera_count: np.ndarray,
    boundary_weights: np.ndarray,
    max_similarity: np.ndarray,
    max_distance: float = 60.0,
    min_distance_weight: float = 0.1,
    semantic_min_similarity: float = 0.0,
    semantic_max_similarity: float = 0.35,
) -> dict[str, np.ndarray]:
    distance_weights = distance_weight(
        distances,
        max_distance=max_distance,
        min_weight=min_distance_weight,
    )
    geometric_weights = geometric_weight(
        visible_camera_count,
        boundary_weights,
    )
    semantic_weights = semantic_weight(
        max_similarity,
        min_similarity=semantic_min_similarity,
        max_similarity_value=semantic_max_similarity,
    )
    reliability_weights = distance_weights * geometric_weights * semantic_weights
    return {
        "distance_weight": distance_weights.astype(np.float32),
        "geometric_weight": geometric_weights.astype(np.float32),
        "semantic_weight": semantic_weights.astype(np.float32),
        "reliability_weight": reliability_weights.astype(np.float32),
    }


def point_reliability(
    distances: np.ndarray,
    valid_mask: np.ndarray,
    visible_camera_count: np.ndarray,
    max_similarity: np.ndarray,
) -> np.ndarray:
    """Backward-compatible wrapper for the initial MVP-v2 formula."""

    return compute_point_reliability(
        distances=distances,
        visible_camera_count=visible_camera_count,
        boundary_weights=valid_mask.astype(np.float32),
        max_similarity=max_similarity,
    )["reliability_weight"]
