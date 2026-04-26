from __future__ import annotations

import numpy as np


def distance_weight(distances: np.ndarray, max_distance: float = 60.0, min_weight: float = 0.1) -> np.ndarray:
    """根据距离生成一个简单的衰减权重。"""

    normalized = np.clip(distances / max_distance, 0.0, 1.0)
    return (1.0 - normalized) * (1.0 - min_weight) + min_weight


def geometric_weight(valid_mask: np.ndarray, visible_camera_count: np.ndarray) -> np.ndarray:
    """根据投影是否有效以及被多少个相机看到，生成几何权重。"""

    visibility_bonus = np.clip(visible_camera_count.astype(np.float32) / 6.0, 0.0, 1.0)
    return valid_mask.astype(np.float32) * (0.5 + 0.5 * visibility_bonus)


def semantic_weight(max_similarity: np.ndarray) -> np.ndarray:
    """根据 teacher 与文本的最大相似度生成语义权重。"""

    return np.clip(max_similarity.astype(np.float32), 0.0, 1.0)


def point_reliability(
    distances: np.ndarray,
    valid_mask: np.ndarray,
    visible_camera_count: np.ndarray,
    max_similarity: np.ndarray,
) -> np.ndarray:
    """MVP-v2 预留：组合 point-level reliability weight。"""

    return (
        distance_weight(distances)
        * geometric_weight(valid_mask, visible_camera_count)
        * semantic_weight(max_similarity)
    )
