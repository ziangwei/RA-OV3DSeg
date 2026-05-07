from __future__ import annotations

import numpy as np


def cosine_similarity(features: np.ndarray, prototypes: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """计算特征与类别原型之间的余弦相似度。"""

    features_norm = features / np.clip(np.linalg.norm(features, axis=-1, keepdims=True), eps, None)
    prototypes_norm = prototypes / np.clip(np.linalg.norm(prototypes, axis=-1, keepdims=True), eps, None)
    return features_norm @ prototypes_norm.T


def segmentation_intersections_unions(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    num_classes: int,
    valid_gt_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-class intersection/union/counts for point semantic segmentation.

    `pred_labels == -1` is treated as invalid prediction: it still contributes to
    the GT count and union of the true class, but not to any predicted class.
    """

    pred_labels = pred_labels.astype(np.int64)
    gt_labels = gt_labels.astype(np.int64)
    if pred_labels.shape[0] != gt_labels.shape[0]:
        raise ValueError(f"pred/gt length mismatch: pred={pred_labels.shape[0]}, gt={gt_labels.shape[0]}")

    if valid_gt_mask is None:
        valid_gt_mask = (gt_labels >= 0) & (gt_labels < num_classes)
    else:
        valid_gt_mask = valid_gt_mask.astype(bool) & (gt_labels >= 0) & (gt_labels < num_classes)

    intersections = np.zeros(num_classes, dtype=np.int64)
    unions = np.zeros(num_classes, dtype=np.int64)
    gt_counts = np.zeros(num_classes, dtype=np.int64)
    for class_id in range(num_classes):
        gt_mask = valid_gt_mask & (gt_labels == class_id)
        pred_mask = valid_gt_mask & (pred_labels == class_id)
        intersections[class_id] = int(np.sum(gt_mask & pred_mask))
        unions[class_id] = int(np.sum(gt_mask | pred_mask))
        gt_counts[class_id] = int(np.sum(gt_mask))
    return intersections, unions, gt_counts


def safe_iou(intersections: np.ndarray, unions: np.ndarray) -> np.ndarray:
    intersections = intersections.astype(np.float64)
    unions = unions.astype(np.float64)
    ious = np.full(intersections.shape, np.nan, dtype=np.float64)
    valid = unions > 0
    ious[valid] = intersections[valid] / unions[valid]
    return ious


def mean_iou_for_ids(ious: np.ndarray, class_ids: np.ndarray | list[int]) -> float:
    class_ids = np.asarray(class_ids, dtype=np.int64)
    if class_ids.shape[0] == 0:
        return float("nan")
    values = ious[class_ids]
    values = values[np.isfinite(values)]
    if values.shape[0] == 0:
        return float("nan")
    return float(values.mean())


def confusion_matrix(
    pred_labels: np.ndarray,
    gt_labels: np.ndarray,
    num_classes: int,
    valid_gt_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Rows are GT labels, columns are predicted labels. Invalid predictions are skipped."""

    pred_labels = pred_labels.astype(np.int64)
    gt_labels = gt_labels.astype(np.int64)
    if valid_gt_mask is None:
        valid_gt_mask = (gt_labels >= 0) & (gt_labels < num_classes)
    else:
        valid_gt_mask = valid_gt_mask.astype(bool) & (gt_labels >= 0) & (gt_labels < num_classes)
    valid_pred = (pred_labels >= 0) & (pred_labels < num_classes)
    valid = valid_gt_mask & valid_pred
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (gt_labels[valid], pred_labels[valid]), 1)
    return matrix
