# RA-OV3DSeg

Reliability-Aware Open-Vocabulary 3D Semantic Segmentation on nuScenes-lidarseg.

## Status

This project is closed as a bounded weak-teacher distillation study. The final
Stage 4 control experiment did **not** validate the proposed reliability score:
random same-scale filtering matched or beat the full reliability formula. No
more GPU experiments are planned.

The useful result is diagnostic rather than a positive method claim: SAM2 +
SigLIP projected pseudo-labels are weak on nuScenes-LiDARSeg, selective
supervision can reduce some damage from noisy labels, but the implemented
distance/geometric/semantic reliability formula is not empirically supported by
the final control pilot.

## Headline Results

| Result | Value | Notes |
|---|---:|---|
| Pointcept SpUNet closed-set val mIoU | 0.7432 | Reproduced nuScenes-LiDARSeg baseline |
| SigLIP text-aligned OV head val mIoU | 0.7465 | Frozen-backbone head training, no closed-set drop |
| SAM2+SigLIP projected teacher mIoU | 0.1022 | 128-sample official16 diagnostic |
| Semantic top-confidence teacher mIoU | 0.3149 / 0.3159 | top-20% / top-40%, background excluded |
| Threshold pilot best | 0.4554 | 128-cache pilot, threshold 0.9 |
| Component/control pilot | failed | full=0.0619, random=0.0793, max component drop=0.0023 |

## Final Interpretation

The project should be described as an engineering and experimental diagnosis of
weak 2D-to-3D open-vocabulary distillation:

1. A strong 3D Pointcept baseline and text-aligned head are stable.
2. The SAM2+SigLIP teacher is useful for analysis but too noisy for clean dense
   supervision.
3. Strict filtering can help compared with using every pseudo-label, but the
   final random/component controls do not support the proposed reliability
   formula as a validated method.
4. The honest stopping point is a negative/limited finding, not a SOTA model.

## Repository Map

- `docs/ROADMAP.md`: current closure decision and stage history.
- `docs/EXPERIMENT_RECAP.md`: durable experiment ledger.
- `docs/INTERVIEW_PREP.md`: interview framing and limitations.
- `scripts/`: preprocessing, teacher extraction, reliability computation, and
  pilot runners.
- `ra_ov3dseg/`: project-owned dataset, projection, model, reliability, and
  training utilities.

## Setup

```bash
bash scripts/setup_env.sh
bash scripts/sanity_check.sh
```

The original training server used separate environments:

- `ra-ov3dseg` for Pointcept/SpUNet training.
- `ra-teacher` for SAM2 teacher extraction.

## Closure Artifacts

- Threshold pilot table:
  `outputs/pointcept/reliability_pilot/pilot_threshold_summary.tsv`
- Component/control pilot table:
  `outputs/pointcept/reliability_component_pilot/pilot_component_summary.tsv`
- Final tracked recap:
  `docs/EXPERIMENT_RECAP.md`
