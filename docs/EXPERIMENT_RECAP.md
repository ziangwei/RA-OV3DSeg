# RA-OV3DSeg Experiment Recap

Last updated: 2026-05-11

This document is the durable experiment record for RA-OV3DSeg. It is meant to
replace ad-hoc terminal logs and large `outputs/` folders when we only need to
remember the technical conclusion.

## Current Position

RA-OV3DSeg started as a reliability-aware open-vocabulary 3D segmentation
project, but the work exposed a stricter dependency order:

```text
1. nuScenes IO / projection correctness
2. reliable closed-set 3D supervised baseline
3. text-aligned 3D embedding interface
4. dense open-vocabulary 2D teacher
5. reliability-aware 2D-to-3D distillation
6. arbitrary text-query 3D inference
```

The current credible state is step 2 moving into step 3. The strongest trusted
result so far is the V16a official-16 supervised cylinder baseline:

```text
V16a, 1024 train / 512 eval / 30 epochs
best_eval_miou = 0.415858
prediction_coverage = 0.998260
```

This is not yet a final open-vocabulary result. It is the corrected closed-set
baseline that makes later open-vocabulary claims meaningful.

## Technical Route

### Final Target

The intended final model should expose a text-query interface:

```text
LiDAR points
  -> 3D backbone
  -> point embeddings
  -> cosine(point_embedding, text_embedding(class_name))
  -> arbitrary text-class prediction
```

The closed-set classifier is an auxiliary training and debugging head, not the
final interface.

### Current Backbone Direction

The in-house toy/debug backbones were useful for plumbing, but not enough for a
credible segmentation baseline. The current backbone direction is:

```text
Pointcept SpUNet v1m1, vendored minimally under third_party/pointcept_spunet/
  -> RA-OV3DSeg adapter
  -> existing RA train / predict / eval scripts
```

Constraint: one repository and one conda environment. No separate Pointcept repo
runtime, no external export/import workflow.

### Current Teacher Direction

Early CLIP/SigLIP patch features and GroupViT/CLIPSeg-style dense teachers were
useful for interface checks, but they were not strong enough to support a final
claim by themselves. The next teacher step should happen only after the closed-
set supervised baseline is clearly credible.

## Result Ledger

| Stage | Purpose | Scale | Key Result | Decision |
|---|---:|---:|---:|---|
| MVP-v0 | nuScenes mini IO, 6-camera projection, overlay visualization | mini | Passed | Keep as data/projection sanity check |
| MVP-v1 | CLIP patch feature extraction, 2D-to-3D assignment, zero-shot baseline | mini | Passed as interface; weak semantics | Interface only, not final teacher |
| MVP-v2 | Point-level reliability score interface | mini | Passed | Keep formula/API for later distillation |
| V4 | Debug MLP training harness | mini / tiny | Passed | Training IO validated only |
| V5 | First sparse 3D student | mini / tiny | Passed | Useful scaffold, not strong baseline |
| V9 | Isolated trainval mini protocol | 128 train / 128 eval | Passed | Established reproducible experiment wrapper/log layout |
| V10 | Open-vocabulary inference interface | 128 eval | all_mIoU `0.000360`, novel `0.0` | Interface exists, embeddings not aligned |
| V11 | Text-aligned embedding loss | 128 train / 128 eval | all_mIoU `0.064718`, base `0.087831`, novel `0.0` | Alignment helps but is far from enough |
| V12 | GroupViT dense teacher path | 128-scale path | Runnable, not a strong final teacher | Do not spend more compute until 3D baseline is fixed |
| V13 | Teacher/backbone diagnostics | smoke / small | Passed diagnostic wrapper | Redirected work to supervised 3D baseline |
| V14 | Improved supervised Cartesian ResUNet | 128 / 512 subsets | `0.2170` at 128 eval; `0.2328` at 512 eval | Still weak; not enough |
| V15 | Cylinder-style supervised baseline | 1024 train / 1024 eval / 30 epochs | best mIoU `0.247641`, coverage `0.947942` | Suspicious: coverage and label-space issues |
| pre-V16 | Sanity checks | 128 samples | point/label count pass; coordinate pass; loss mask pass; coverage pass; split warning | Found label-space/split issue |
| V16a | Official-16 cylinder supervised baseline | 1024 train / 512 eval / 30 epochs | best mIoU `0.415858`, coverage `0.998260` | First credible supervised baseline |
| V17 old | Pointcept SpUNet initial adapter | 128 train / 128 eval / 10 epochs | all_mIoU `0.186444`, coverage `0.996193` | Invalid comparison: extra embedding bottleneck |
| V17 headfix | Pointcept SpUNet direct supervised head | smoke | Verification passed | Next default 128 run pending |

## Important Lessons

1. The early poor numbers were not only model-capacity failures. Label-space and
   coverage bugs materially affected evaluation.

2. The project should not claim open-vocabulary quality from CLIP patch tokens.
   CLIP patch assignment was an engineering scaffold.

3. Strong closed-set supervision must come before reliability-aware distillation.
   If the 3D student cannot learn lidarseg labels directly, teacher distillation
   will not rescue the project.

4. Official nuScenes-lidarseg 16-class evaluation is the current baseline space.
   Raw 32-class experiments are useful for debugging but are not the main
   benchmark.

5. V17 must be evaluated only after the direct supervised head fix. The old V17
   128 number is recorded as a negative result, not a benchmark.

## Interview Framing

Concise project story:

```text
I built an end-to-end research pipeline for reliability-aware open-vocabulary
3D semantic segmentation on nuScenes-lidarseg. The project includes nuScenes
data validation, LiDAR-camera projection, 2D feature assignment, point-level
reliability scoring, sparse-conv 3D training, open-vocabulary text-query
inference, and reproducible experiment launchers. During development I found
that open-vocabulary distillation was premature because the closed-set 3D
student baseline was not credible. I then added raw-lidarseg supervised training,
fixed label-space and coverage bugs, moved to the official 16-class space, and
raised the supervised baseline from about 24.8 mIoU to about 41.6 mIoU on a
1024/512 subset. The current work is integrating a minimally vendored Pointcept
SpUNet backbone inside the same repository and environment before reintroducing
reliability-aware open-vocabulary distillation.
```

What to emphasize:

- Reproducible research engineering, not just model code.
- Geometry correctness: LiDAR to camera projection and point-feature assignment.
- Diagnosis discipline: separating teacher weakness, backbone weakness, label
  mapping bugs, and evaluation coverage issues.
- Single-repo integration discipline for third-party backbone code.

What not to overclaim:

- Do not claim final open-vocabulary performance yet.
- Do not claim GroupViT/CLIPSeg teacher quality is sufficient.
- Do not report the old V17 `0.186` number as the Pointcept baseline.

## Output Retention Guide

Raw data and code are separate:

- Keep `data/nuscenes/`.
- Keep this repository and tracked docs.
- Delete `outputs/` artifacts once the key result is recorded here and the
  checkpoint is no longer needed.

Keep only these if storage is tight:

```text
outputs/experiments/trainval_v16a_official16_cylinder_1024_r120/
  training/cylinder_spconv_unet_best.pt          # optional: best current baseline checkpoint
  training/train_summary.json                    # small, useful
  evaluation3d/batch_3d_eval_summary.json        # small, useful
  compact_summary.json                           # small, useful

outputs/experiments/trainval_v17_pointcept_spunet_*_headfix/
  training/train_summary.json
  evaluation3d/batch_3d_eval_summary.json
  compact_summary.json
  training/pointcept_spunet_best.pt              # keep only if it becomes the best baseline
```

Safe-to-delete categories after recording metrics:

```text
outputs/features2d/
outputs/point_features/
outputs/projections/
outputs/reliability/
outputs/zero_shot/
outputs/visualizations/
outputs/dense_teacher_logits/
outputs/dense_point_logits/
outputs/voxelization/
outputs/training_dryrun/
outputs/training_v*/
outputs/predictions3d/
outputs/evaluation3d/
```

Safe-to-delete old experiment dirs after this recap:

```text
outputs/experiments/trainval_v9_*
outputs/experiments/trainval_v10_*
outputs/experiments/trainval_v11_*
outputs/experiments/trainval_v12_*
outputs/experiments/trainval_v13_*
outputs/experiments/trainval_v14_*
outputs/experiments/trainval_v15_*
outputs/experiments/trainval_v17_pointcept_spunet_128/          # old non-headfix run
outputs/experiments/trainval_v17_pointcept_spunet_smoke/        # old non-headfix smoke
```

Do not delete a directory if you plan to resume that exact experiment. In that
case, keep its `precompute/`, `training/`, and logs.

## Next Decisions

1. Run V17 headfix default 128 and compare against V16a at a similar scale.

2. If V17 headfix is clearly better at 128, scale it to 1024 train / 512 eval.

3. If V17 still underperforms V16a, debug the adapter against Pointcept's recipe
   before adding any teacher or open-vocabulary experiment.

4. Re-enter open-vocabulary only after a supervised baseline is strong enough to
   make distillation meaningful.
