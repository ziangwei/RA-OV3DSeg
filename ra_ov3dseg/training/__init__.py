"""Training helpers for dry-run and future model training."""

from .labels import (
    NUSCENES_LIDARSEG_OFFICIAL_CLASS_NAMES,
    NUSCENES_RAW_TO_OFFICIAL_16,
    ClassSplit,
    build_class_split,
    map_labels_for_base_ce,
    map_official_16_for_ce,
    map_raw_lidarseg_to_official_16,
)
from .augmentations import PointAugmentationConfig, augment_point_xyz


_LAZY_EXPORTS = {
    "cosine_distillation_loss": ("ra_ov3dseg.training.losses", "cosine_distillation_loss"),
    "dense_logit_distillation_loss": ("ra_ov3dseg.training.losses", "dense_logit_distillation_loss"),
    "dice_loss": ("ra_ov3dseg.training.losses", "dice_loss"),
    "lovasz_softmax_loss": ("ra_ov3dseg.training.losses", "lovasz_softmax_loss"),
    "supervised_ce_loss": ("ra_ov3dseg.training.losses", "supervised_ce_loss"),
    "PrecomputedPointFeatureDataset": ("ra_ov3dseg.training.precomputed_dataset", "PrecomputedPointFeatureDataset"),
    "collate_point_feature_samples": ("ra_ov3dseg.training.precomputed_dataset", "collate_point_feature_samples"),
    "find_missing_dense_point_files": ("ra_ov3dseg.training.precomputed_dataset", "find_missing_dense_point_files"),
    "find_missing_precomputed_files": ("ra_ov3dseg.training.precomputed_dataset", "find_missing_precomputed_files"),
    "RawLidarsegDataset": ("ra_ov3dseg.training.raw_lidarseg_dataset", "RawLidarsegDataset"),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module_name, attribute_name = _LAZY_EXPORTS[name]
    value = getattr(importlib.import_module(module_name), attribute_name)
    globals()[name] = value
    return value

__all__ = [
    "ClassSplit",
    "NUSCENES_LIDARSEG_OFFICIAL_CLASS_NAMES",
    "NUSCENES_RAW_TO_OFFICIAL_16",
    "build_class_split",
    "map_labels_for_base_ce",
    "map_official_16_for_ce",
    "map_raw_lidarseg_to_official_16",
    "PointAugmentationConfig",
    "augment_point_xyz",
    "cosine_distillation_loss",
    "dense_logit_distillation_loss",
    "dice_loss",
    "lovasz_softmax_loss",
    "supervised_ce_loss",
    "PrecomputedPointFeatureDataset",
    "RawLidarsegDataset",
    "collate_point_feature_samples",
    "find_missing_dense_point_files",
    "find_missing_precomputed_files",
]
