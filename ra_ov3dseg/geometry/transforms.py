from __future__ import annotations

from typing import Sequence

import numpy as np
from pyquaternion import Quaternion


def quaternion_to_rotation_matrix(rotation: Sequence[float]) -> np.ndarray:
    """将 nuScenes 的四元数转换为 3x3 旋转矩阵。"""

    return Quaternion(rotation).rotation_matrix.astype(np.float32)


def transform_points(
    points: np.ndarray,
    rotation: Sequence[float],
    translation: Sequence[float],
) -> np.ndarray:
    """对点做正向刚体变换。

    这里统一使用 `N x 3` 的行向量表示点。
    如果列向量形式是 `p_out = R @ p_in + t`，
    那么行向量形式对应为 `p_out = p_in @ R.T + t`。
    """

    rotation_matrix = quaternion_to_rotation_matrix(rotation)
    translation_vector = np.asarray(translation, dtype=np.float32).reshape(1, 3)
    return points @ rotation_matrix.T + translation_vector


def inverse_transform_points(
    points: np.ndarray,
    rotation: Sequence[float],
    translation: Sequence[float],
) -> np.ndarray:
    """对点做逆向刚体变换。

    已知正向关系为 `p_out = p_in @ R.T + t`，
    则逆变换为 `p_in = (p_out - t) @ R`。
    """

    rotation_matrix = quaternion_to_rotation_matrix(rotation)
    translation_vector = np.asarray(translation, dtype=np.float32).reshape(1, 3)
    return (points - translation_vector) @ rotation_matrix
