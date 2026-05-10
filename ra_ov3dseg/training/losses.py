from __future__ import annotations

import torch
import torch.nn.functional as F


def supervised_ce_loss(
    logits: torch.Tensor,
    train_labels: torch.Tensor,
    ignore_index: int = -100,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    valid = train_labels != ignore_index
    if not torch.any(valid):
        return logits.sum() * 0.0
    return F.cross_entropy(logits, train_labels, ignore_index=ignore_index, weight=class_weights)


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    """Gradient of the Lovasz extension with respect to sorted errors."""

    num_pixels = gt_sorted.numel()
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted.float()).cumsum(0)
    jaccard = 1.0 - intersection / torch.clamp(union, min=1.0)
    if num_pixels > 1:
        jaccard[1:num_pixels] = jaccard[1:num_pixels] - jaccard[: num_pixels - 1]
    return jaccard


def lovasz_softmax_loss(
    logits: torch.Tensor,
    train_labels: torch.Tensor,
    ignore_index: int = -100,
    classes: str = "present",
) -> torch.Tensor:
    """Multi-class Lovasz-Softmax loss for point segmentation logits.

    This implementation follows the standard flat multi-class formulation and
    is intended as an IoU-oriented supplement to CE for imbalanced lidarseg data.
    """

    valid = train_labels != ignore_index
    if not torch.any(valid):
        return logits.sum() * 0.0

    probs = F.softmax(logits[valid], dim=1)
    labels = train_labels[valid].long()
    num_classes = probs.shape[1]
    losses = []
    for class_idx in range(num_classes):
        fg = (labels == class_idx).float()
        if classes == "present" and torch.sum(fg) == 0:
            continue
        errors = (fg - probs[:, class_idx]).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def dice_loss(
    logits: torch.Tensor,
    train_labels: torch.Tensor,
    ignore_index: int = -100,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Soft Dice loss over classes present in the current batch."""

    valid = train_labels != ignore_index
    if not torch.any(valid):
        return logits.sum() * 0.0

    probs = F.softmax(logits[valid], dim=1)
    labels = train_labels[valid].long()
    num_classes = probs.shape[1]
    losses = []
    for class_idx in range(num_classes):
        target = (labels == class_idx).float()
        if torch.sum(target) == 0:
            continue
        pred = probs[:, class_idx]
        numerator = 2.0 * torch.sum(pred * target) + eps
        denominator = torch.sum(pred) + torch.sum(target) + eps
        losses.append(1.0 - numerator / denominator)
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


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
