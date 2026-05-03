from __future__ import annotations

import numpy as np

from ra_ov3dseg.evaluation.metrics import cosine_similarity


def l2_normalize(features: np.ndarray, axis: int = -1, eps: float = 1e-6) -> np.ndarray:
    norms = np.linalg.norm(features, axis=axis, keepdims=True)
    return features / np.clip(norms, eps, None)


def zero_shot_logits(point_features: np.ndarray, text_embeddings: np.ndarray) -> np.ndarray:
    """计算点特征与文本特征之间的余弦相似度矩阵。"""

    return cosine_similarity(point_features, text_embeddings)


def zero_shot_predict(point_features: np.ndarray, text_embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """用最近邻余弦相似度做 zero-shot 分类。"""

    logits = zero_shot_logits(point_features, text_embeddings)
    pred_indices = np.argmax(logits, axis=1)
    pred_scores = logits[np.arange(logits.shape[0]), pred_indices]
    return pred_indices.astype(np.int32), pred_scores.astype(np.float32)
