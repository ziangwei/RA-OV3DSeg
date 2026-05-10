# Running Experiments

All V9+ experiments should be launched through bash wrappers, not by pasting long
Python commands into the terminal. The wrapper records stdout/stderr, command
arguments, GPU snapshot, and disk snapshot into `outputs/logs/`.

## V9 Trainval Subset

Smoke test:

```bash
bash scripts/run_v9_trainval_experiment.sh --profile smoke
```

Small trainval subset:

```bash
bash scripts/run_v9_trainval_experiment.sh --profile small
```

Explicit server paths:

```bash
bash scripts/run_v9_trainval_experiment.sh \
  --profile small \
  --dataroot /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg/data/nuscenes \
  --cache_dir /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/huggingface_cache
```

## Logs

Logs are written to:

```text
outputs/logs/v9_<experiment_name>_<timestamp>.log
outputs/logs/v9_<experiment_name>_latest.log
```

Experiment artifacts are isolated under:

```text
outputs/experiments/<experiment_name>/
  precompute/
    projections/
    features2d/
    point_features/
    zero_shot/
    reliability/
    dense_teacher_logits/
    dense_point_logits/
  training/
  predictions3d/
  evaluation3d/
```

Do not reuse root-level MVP folders such as `outputs/point_features/` for trainval
experiments. Those folders may contain mini artifacts with the same
`sample_0000` names, which can cause label/point count mismatches.

Watch progress:

```bash
tail -f outputs/logs/v9_trainval_v9_8_latest.log
```

Search important milestones:

```bash
grep -E "RUN|PASS|FAIL|training|dense_teacher|eval|exit_status" outputs/logs/*.log
```

## Resume Behavior

The launcher passes `--skip_existing` by default. If a previous run finished some
precompute artifacts, rerunning the same profile reuses them and continues from
the missing stage.

Use `--no_skip_existing` only when intentionally regenerating artifacts.

## V10 Open-Vocabulary Evaluation

Run open-vocabulary prediction with the V9 isolated checkpoint:

```bash
bash scripts/run_v10_open_vocab_eval.sh
```

This writes:

```text
outputs/experiments/trainval_v10_open_vocab_128/
  open_vocab_predictions3d/
  open_vocab_evaluation3d/
outputs/logs/v10_trainval_v10_open_vocab_128_<timestamp>.log
outputs/logs/v10_trainval_v10_open_vocab_128_latest.log
```

Watch progress:

```bash
tail -f outputs/logs/v10_trainval_v10_open_vocab_128_latest.log
```

Verify:

```bash
python scripts/verify_mvp_outputs.py \
  --stage v10 \
  --sample_idx 128 \
  --outputs_dir outputs \
  --experiment_dir outputs/experiments/trainval_v10_open_vocab_128 \
  --output_dir outputs/verification
```

Arbitrary text query mode:

```bash
bash scripts/run_v10_open_vocab_eval.sh \
  --experiment_name trainval_v10_custom_queries \
  --class_names_csv "driveable surface,sidewalk,car,truck,pedestrian,vegetation" \
  --skip_eval
```

`--skip_eval` is required for arbitrary labels unless they exactly map to
nuScenes-lidarseg names.

## V10+ Rule

New training or evaluation stages should follow the same pattern:

- Add one `scripts/run_v*_*.sh` wrapper.
- Put logs under `outputs/logs/`.
- Keep experiment artifacts under `outputs/experiments/<experiment_name>/`.
- Make the Python script configurable, but make the bash wrapper the normal
  server entrypoint.

## V11 Text-Aligned Embedding Training

V10 proved the open-vocabulary inference interface exists, but the point
embeddings were not yet aligned to text. V11 warm-starts from the V9 sparse
student checkpoint and adds a base-class text-prototype alignment loss.

Default run:

```bash
bash scripts/run_v11_text_aligned_training.sh
```

This reuses:

```text
outputs/experiments/trainval_v9_128_isolated/precompute/
outputs/experiments/trainval_v9_128_isolated/training/sparse_unet_spconv_latest.pt
```

and writes:

```text
outputs/experiments/trainval_v11_text_align_128/
  training/
  open_vocab_predictions3d/
  open_vocab_evaluation3d/
outputs/logs/v11_trainval_v11_text_align_128_<timestamp>.log
outputs/logs/v11_trainval_v11_text_align_128_latest.log
```

Watch progress:

```bash
tail -f outputs/logs/v11_trainval_v11_text_align_128_latest.log
```

Verify:

```bash
python scripts/verify_mvp_outputs.py \
  --stage v11 \
  --sample_idx 128 \
  --outputs_dir outputs \
  --experiment_dir outputs/experiments/trainval_v11_text_align_128 \
  --output_dir outputs/verification
```

Quick smoke before a full V11 run:

```bash
bash scripts/run_v11_text_aligned_training.sh \
  --experiment_name trainval_v11_text_align_8 \
  --train_max_samples 8 \
  --eval_max_samples 8 \
  --epochs 2
```

If the first run needs to download the CLIP text model, omit
`--local_files_only`. After the model is cached, add `--local_files_only` for
stable offline reruns.

## V12 GroupViT Dense Teacher

The recommended V12 path uses `groupvit_dense`, which is available through
Hugging Face Transformers and runs in the existing `ra-ov3dseg` environment.
It does not require a second repository or a second conda environment.

Default run:

```bash
bash scripts/run_v12_groupvit_teacher_training.sh --local_files_only
```

If the GroupViT model is not cached yet, omit `--local_files_only` for the first
run:

```bash
bash scripts/run_v12_groupvit_teacher_training.sh
```

Smoke run:

```bash
bash scripts/run_v12_groupvit_teacher_training.sh \
  --experiment_name trainval_v12_groupvit_smoke \
  --train_max_samples 8 \
  --eval_max_samples 8 \
  --epochs 2
```

This writes:

```text
outputs/experiments/trainval_v12_groupvit_128/
  precompute/dense_teacher_logits/
  precompute/dense_point_logits/
  training/
  open_vocab_predictions3d/
  open_vocab_evaluation3d/
outputs/logs/v12_trainval_v12_groupvit_128_<timestamp>.log
outputs/logs/v12_trainval_v12_groupvit_128_latest.log
```

Verify:

```bash
python scripts/verify_mvp_outputs.py \
  --stage v12 \
  --sample_idx 128 \
  --outputs_dir outputs \
  --experiment_dir outputs/experiments/trainval_v12_groupvit_128 \
  --output_dir outputs/verification
```

## V13 Diagnostics

V13 is not another teacher sweep. It answers two gating questions before more
open-vocabulary training:

- How good is the projected dense 2D teacher by itself?
- How high can a stronger 3D student go under full lidarseg supervision?

Smoke run:

```bash
bash scripts/run_v13_diagnostics.sh \
  --experiment_name trainval_v13_diagnostics_smoke \
  --train_max_samples 8 \
  --eval_max_samples 8 \
  --epochs 2 \
  --sparse_base_channels 16
```

Default 128-sample diagnostic:

```bash
bash scripts/run_v13_diagnostics.sh
```

This writes:

```text
outputs/experiments/trainval_v13_diagnostics_128/
  teacher_quality/
  supervised_training/
  supervised_predictions3d/
  supervised_evaluation3d/
outputs/logs/v13_trainval_v13_diagnostics_128_<timestamp>.log
outputs/logs/v13_trainval_v13_diagnostics_128_latest.log
```

Key files to inspect:

```text
outputs/experiments/trainval_v13_diagnostics_128/teacher_quality/batch_teacher_pseudo_eval_summary.json
outputs/experiments/trainval_v13_diagnostics_128/supervised_training/train_summary.json
outputs/experiments/trainval_v13_diagnostics_128/supervised_evaluation3d/batch_3d_eval_summary.json
```

## V14 Supervised ResUNet Recipe

V14 improves the closed-set 3D upper-bound recipe before more open-vocabulary
distillation:

- Uses raw nuScenes LiDAR + lidarseg labels directly; it does not require V9
  point-feature/reliability precompute caches.
- Computes class frequencies and inverse-sqrt class weights.
- Trains `spconv_resunet` with weighted CE plus optional Lovasz/Dice losses.
- Enables basic LiDAR augmentations.
- Evaluates during training and saves `spconv_resunet_best.pt` by eval mIoU.

Smoke run:

```bash
bash scripts/run_v14_supervised_resunet.sh \
  --experiment_name trainval_v14_supervised_resunet_smoke \
  --train_max_samples 8 \
  --eval_max_samples 8 \
  --epochs 2 \
  --sparse_base_channels 16
```

Default 128-sample run:

```bash
bash scripts/run_v14_supervised_resunet.sh
```

Verify:

```bash
python scripts/verify_mvp_outputs.py \
  --stage v14 \
  --outputs_dir outputs \
  --experiment_dir outputs/experiments/trainval_v14_supervised_resunet_128 \
  --output_dir outputs/verification
```

Key files:

```text
outputs/experiments/trainval_v14_supervised_resunet_128/class_frequencies.json
outputs/experiments/trainval_v14_supervised_resunet_128/training/train_summary.json
outputs/experiments/trainval_v14_supervised_resunet_128/training/spconv_resunet_best.pt
outputs/experiments/trainval_v14_supervised_resunet_128/evaluation3d/batch_3d_eval_summary.json
```

## V15 Cylinder Baseline

V15 is the current supervised baseline direction. It directly replaces the
Cartesian `spconv_resunet` with `cylinder_spconv_unet`, a Cylinder3D-style
spconv backbone that uses raw LiDAR intensity and cylindrical voxelization.

Smoke run:

```bash
bash scripts/run_v15_cylinder_baseline.sh \
  --experiment_name trainval_v15_cylinder_smoke \
  --train_max_samples 8 \
  --eval_start_idx 128 \
  --eval_max_samples 8 \
  --epochs 2 \
  --sparse_base_channels 16
```

Default 1024-sample run:

```bash
bash scripts/run_v15_cylinder_baseline.sh
```

Verify:

```bash
python scripts/verify_mvp_outputs.py \
  --stage v15 \
  --outputs_dir outputs \
  --experiment_dir outputs/experiments/trainval_v15_cylinder_1024 \
  --output_dir outputs/verification
```

Key files:

```text
outputs/experiments/trainval_v15_cylinder_1024/class_frequencies.json
outputs/experiments/trainval_v15_cylinder_1024/training/train_summary.json
outputs/experiments/trainval_v15_cylinder_1024/training/cylinder_spconv_unet_best.pt
outputs/experiments/trainval_v15_cylinder_1024/evaluation3d/batch_3d_eval_summary.json
```
