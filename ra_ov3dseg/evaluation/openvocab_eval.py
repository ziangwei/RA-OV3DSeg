from __future__ import annotations

import numpy as np

from ra_ov3dseg.evaluation.metrics import cosine_similarity


def zero_shot_predict(point_features: np.ndarray, text_embeddings: np.ndarray) -> np.ndarray:
    """MVP-v1 预留：用最近邻余弦相似度做 zero-shot 分类。"""

    logits = cosine_similarity(point_features, text_embeddings)
    return np.argmax(logits, axis=1)
