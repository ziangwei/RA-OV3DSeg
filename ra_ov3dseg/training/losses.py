from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_ce_loss(logits: torch.Tensor, train_labels: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
    valid = train_labels != ignore_index
    if not torch.any(valid):
        return logits.sum() * 0.0
    return F.cross_entropy(logits, train_labels, ignore_index=ignore_index)


def cosine_distillation_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    weights: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    valid = valid_mask & torch.isfinite(weights) & (weights > 0)
    if not torch.any(valid):
        return student_features.sum() * 0.0

    student = F.normalize(student_features[valid], dim=-1, eps=eps)
    teacher = F.normalize(teacher_features[valid], dim=-1, eps=eps)
    point_loss = 1.0 - torch.sum(student * teacher, dim=-1)
    valid_weights = weights[valid].float()
    return torch.sum(point_loss * valid_weights) / torch.clamp(valid_weights.sum(), min=eps)


def dense_logit_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    weights: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reliability-weighted KL distillation from dense 2D teacher logits to 3D point logits."""
    valid = valid_mask & torch.isfinite(weights) & (weights > 0)
    valid = valid & torch.isfinite(teacher_logits).all(dim=-1) & torch.isfinite(student_logits).all(dim=-1)
    if not torch.any(valid):
        return student_logits.sum() * 0.0

    temp = max(float(temperature), eps)
    student_log_probs = F.log_softmax(student_logits[valid] / temp, dim=-1)
    teacher_probs = F.softmax(teacher_logits[valid] / temp, dim=-1)
    point_kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1) * (temp * temp)
    valid_weights = weights[valid].float()
    return torch.sum(point_kl * valid_weights) / torch.clamp(valid_weights.sum(), min=eps)


def text_prototype_alignment_loss(
    student_features: torch.Tensor,
    train_labels: torch.Tensor,
    text_prototypes: torch.Tensor,
    valid_mask: torch.Tensor,
    ignore_index: int = -100,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Supervised cosine loss that pulls point embeddings toward class text prototypes.

    `train_labels` must index rows in `text_prototypes`. This should be applied only
    to supervised base-class points, while novel/ignore labels stay `ignore_index`.
    """

    valid = valid_mask & (train_labels != ignore_index)
    valid = valid & (train_labels >= 0) & (train_labels < text_prototypes.shape[0])
    if not torch.any(valid):
        return student_features.sum() * 0.0

    student = F.normalize(student_features[valid], dim=-1, eps=eps)
    target = F.normalize(text_prototypes[train_labels[valid]], dim=-1, eps=eps)
    return (1.0 - torch.sum(student * target, dim=-1)).mean()
