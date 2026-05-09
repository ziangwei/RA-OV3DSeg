from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


CLIP_PATCH_BASELINE = "clip_patch_baseline"
CLIPSEG_DENSE = "clipseg_dense"
GROUPVIT_DENSE = "groupvit_dense"
SUPPORTED_TEACHERS = (
    CLIP_PATCH_BASELINE,
    CLIPSEG_DENSE,
    GROUPVIT_DENSE,
)


@dataclass(frozen=True)
class TeacherSpec:
    name: str
    role: str
    feature_granularity: str
    is_baseline: bool
    description: str


def describe_teacher(teacher_backend: str) -> TeacherSpec:
    if teacher_backend == CLIP_PATCH_BASELINE:
        return TeacherSpec(
            name=teacher_backend,
            role="mvp_baseline",
            feature_granularity="coarse_patch",
            is_baseline=True,
            description=(
                "CLIP/SigLIP patch-token baseline. It verifies the open-vocabulary "
                "2D-to-3D pipeline, but it is too coarse to be the final dense teacher."
            ),
        )

    if teacher_backend == CLIPSEG_DENSE:
        return TeacherSpec(
            name=teacher_backend,
            role="runnable_dense_teacher",
            feature_granularity="dense_class_logits",
            is_baseline=True,
            description=(
                "Runnable dense open-vocabulary segmentation teacher based on CLIPSeg. "
                "It produces per-class dense logits and is stronger than CLIP patch tokens, "
                "but remains a practical baseline before the GroupViT teacher."
            ),
        )

    if teacher_backend == GROUPVIT_DENSE:
        return TeacherSpec(
            name=teacher_backend,
            role="transformers_native_dense_teacher",
            feature_granularity="dense_class_logits",
            is_baseline=False,
            description=(
                "Transformers-native GroupViT zero-shot semantic segmentation teacher. "
                "It runs inside the RA-OV3DSeg environment through Hugging Face Transformers."
            ),
        )

    raise ValueError(f"Unknown teacher_backend={teacher_backend}. Supported: {SUPPORTED_TEACHERS}")


def build_image_teacher(
    teacher_backend: str,
    model_name: str,
    device: str = "auto",
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
):
    """Build an image teacher for 2D feature extraction.

    `clip_patch_baseline` is used by the MVP feature-assignment path. Dense
    teachers write class-logit maps and are handled by
    `scripts/extract_dense_teacher_logits.py`.
    """

    if teacher_backend == CLIP_PATCH_BASELINE:
        from ra_ov3dseg.models.image_encoder import ImageEncoder

        return ImageEncoder(
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

    if teacher_backend == CLIPSEG_DENSE:
        raise NotImplementedError(
            "clipseg_dense is a dense-logit teacher. Use scripts/extract_dense_teacher_logits.py "
            "instead of scripts/extract_2d_features.py."
        )

    if teacher_backend == GROUPVIT_DENSE:
        raise NotImplementedError(
            "groupvit_dense is a dense-logit teacher. Use scripts/extract_dense_teacher_logits.py "
            "instead of scripts/extract_2d_features.py."
        )

    raise ValueError(f"Unknown teacher_backend={teacher_backend}. Supported: {SUPPORTED_TEACHERS}")
