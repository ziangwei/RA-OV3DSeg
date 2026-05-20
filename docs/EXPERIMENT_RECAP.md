# RA-OV3DSeg Experiment Recap

This file is the durable ledger of experiments. Each experiment ends with
its `RunConclusion` block, which automatically appends one row to the table
below via `RunConclusion.append_to_recap()`.

## Result Ledger

| Date | Stage | Experiment | Status | Primary Metric | Notes |
|---|---|---|---|---|---|
| 2026-05-14 | stage-baseline | eval_baseline_fast | success | val_miou=0.7432 | fast validation via Pointcept SemSegEvaluator; skips PreciseEvaluator |
| 2026-05-20 | stage-ov-head | train_ov_head | success | val_miou=0.7465 | frozen-backbone SigLIP prototype head; final eval mIoU=0.7449 |
| 2026-05-20 | stage-teacher | eval_dense_teacher_pseudo_labels | success | teacher_miou=0.1148 | SAM2+SigLIP official16 5-sample diagnostic; coverage=0.4209 |
| 2026-05-20 | stage-teacher | eval_dense_teacher_pseudo_labels | success | teacher_miou=0.1376 | SAM2+SigLIP official16 32-sample diagnostic; coverage=0.5464; semantic top20/top40 mIoU=0.3002/0.3386 |
| 2026-05-20 | stage-teacher | eval_dense_teacher_pseudo_labels | success | teacher_miou=0.1022 | SAM2+SigLIP official16 128-sample diagnostic; coverage=0.5578; semantic top20/top40 mIoU=0.3149/0.3159 |
| 2026-05-20 | stage-reliability | compute_reliability | success | reliability_mean=0.4990 | 128-sample rank-calibrated cache; semantic_score_ratio=0.4196; high>=0.5 ratio=0.4990 |

## Carryover Knowledge From Prototype Phase

This is the only narrative content allowed in this file. It captures lessons
learned in the 1-week prototype phase before the migration.

### Lessons that still apply

1. CLIP patch features are insufficient for fine-grained outdoor lidarseg.
2. GroupViT projected pseudo-labels reach only about 0.02 mIoU; not strong
   enough to drive distillation without filtering.
3. Hand-implementing sparse-conv backbones in 1 week did not match published
   numbers; pip-installed Pointcept SpUNet is the chosen direction.
4. Official nuScenes-lidarseg 16-class mapping is the correct label space
   (not raw 32-class).
5. Prediction coverage = 1.0 must be verified; voxelization that drops
   points causes evaluation artifacts.
6. Confidence ranking must exclude ignore/background pseudo-labels when the
   question is semantic teacher quality; otherwise high-confidence background
   points can hide useful semantic subsets.

### Lessons that did NOT generalize

Filled as new lessons supersede old ones.
