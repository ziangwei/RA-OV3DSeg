from __future__ import annotations

import numpy as np


def cosine_similarity(features: np.ndarray, prototypes: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """计算特征与类别原型之间的余弦相似度。"""

    features_norm = features / np.clip(np.linalg.norm(features, axis=-1, keepdims=True), eps, None)
    prototypes_norm = prototypes / np.clip(np.linalg.norm(prototypes, axis=-1, keepdims=True), eps, None)
    return features_norm @ prototypes_norm.T
