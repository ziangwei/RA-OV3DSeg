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
