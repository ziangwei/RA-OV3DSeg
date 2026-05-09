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

## V12 External Dense Teacher

V12 does not run CAT-Seg/OpenSeg inside this repository. The external teacher
should run in its own environment and export canonical dense logits:

```text
outputs/external_teachers/catseg_dense/
  sample_0000_dense_teacher_logits.npz
  sample_0001_dense_teacher_logits.npz
  ...
```

Required `.npz` keys:

```text
sample_idx
sample_token
teacher_backend
model_name
camera_names
camera_available
image_widths
image_heights
class_names
prompts
dense_logits
```

`dense_logits` may be either `(camera, class, height, width)` or
`(camera, height, width, class)`. `class_names` must start with the 32
nuScenes-lidarseg names in `configs/nuscenes_lidarseg_class_names.txt`.
See `docs/EXTERNAL_DENSE_TEACHER_FORMAT.md` for the full contract.
For a CAT-Seg-specific server workflow, see `docs/CATSEG_SERVER_EXPORT.md`.

First produce the manifest for the external teacher environment:

```bash
bash scripts/run_v12_external_teacher_training.sh --manifest_only
```

The manifest is written to:

```text
outputs/experiments/trainval_v12_external_teacher_128/external_teacher_manifest/
```

After the external teacher has written the canonical logits, run V12:

```bash
bash scripts/run_v12_external_teacher_training.sh \
  --external_dense_teacher_dir outputs/external_teachers/catseg_dense \
  --local_files_only
```

This writes:

```text
outputs/experiments/trainval_v12_external_teacher_128/
  external_teacher_check/
  precompute/dense_point_logits/
  training/
  open_vocab_predictions3d/
  open_vocab_evaluation3d/
outputs/logs/v12_trainval_v12_external_teacher_128_<timestamp>.log
outputs/logs/v12_trainval_v12_external_teacher_128_latest.log
```

Verify:

```bash
python scripts/verify_mvp_outputs.py \
  --stage v12 \
  --sample_idx 128 \
  --outputs_dir outputs \
  --experiment_dir outputs/experiments/trainval_v12_external_teacher_128 \
  --output_dir outputs/verification
```
