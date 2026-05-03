from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


CLIP_PATCH_BASELINE = "clip_patch_baseline"
OPENSEG_DENSE = "openseg_dense"
GROUNDED_SAM_MASK = "grounded_sam_mask"
SUPPORTED_TEACHERS = (CLIP_PATCH_BASELINE, OPENSEG_DENSE, GROUNDED_SAM_MASK)


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

    if teacher_backend == OPENSEG_DENSE:
        return TeacherSpec(
            name=teacher_backend,
            role="main_dense_teacher",
            feature_granularity="dense_pixel",
            is_baseline=False,
            description=(
                "Planned main teacher: dense open-vocabulary pixel-level features or logits. "
                "This is the preferred route for point-level 2D-to-3D distillation."
            ),
        )

    if teacher_backend == GROUNDED_SAM_MASK:
        return TeacherSpec(
            name=teacher_backend,
            role="high_quality_mask_teacher",
            feature_granularity="open_vocab_mask",
            is_baseline=False,
            description=(
                "Planned mask-level teacher for higher-quality pseudo labels. It is heavier "
                "than dense feature extraction and should be introduced after the sparse 3D "
                "student is stable."
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

    `clip_patch_baseline` is the only runnable teacher in the current code. The
    dense teacher names are registered now so configs and outputs no longer imply
    that CLIP patch tokens are the final method.
    """

    if teacher_backend == CLIP_PATCH_BASELINE:
        from ra_ov3dseg.models.image_encoder import ImageEncoder

        return ImageEncoder(
            model_name=model_name,
            device=device,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

    if teacher_backend == OPENSEG_DENSE:
        raise NotImplementedError(
            "openseg_dense is the intended main dense teacher, but it is not implemented yet. "
            "Use --teacher_backend clip_patch_baseline for MVP smoke tests, then add the "
            "OpenSeg/OVSeg adapter in the dense-teacher stage."
        )

    if teacher_backend == GROUNDED_SAM_MASK:
        raise NotImplementedError(
            "grounded_sam_mask is reserved for a heavier mask pseudo-label pipeline. "
            "It should not block the current sparse 3D student work."
        )

    raise ValueError(f"Unknown teacher_backend={teacher_backend}. Supported: {SUPPORTED_TEACHERS}")
