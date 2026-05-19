"""Training helpers retained for reliability-aware distillation."""

from .labels import (
    NUSCENES_LIDARSEG_OFFICIAL_CLASS_NAMES,
    NUSCENES_RAW_TO_OFFICIAL_16,
    ClassSplit,
    build_class_split,
    map_labels_for_base_ce,
    map_official_16_for_ce,
    map_raw_lidarseg_to_official_16,
)


_LAZY_EXPORTS = {
    "cosine_distillation_loss": ("ra_ov3dseg.training.losses", "cosine_distillation_loss"),
    "dense_logit_distillation_loss": ("ra_ov3dseg.training.losses", "dense_logit_distillation_loss"),
    "dice_loss": ("ra_ov3dseg.training.losses", "dice_loss"),
    "lovasz_softmax_loss": ("ra_ov3dseg.training.losses", "lovasz_softmax_loss"),
    "supervised_ce_loss": ("ra_ov3dseg.training.losses", "supervised_ce_loss"),
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
    "cosine_distillation_loss",
    "dense_logit_distillation_loss",
    "dice_loss",
    "lovasz_softmax_loss",
    "supervised_ce_loss",
]
