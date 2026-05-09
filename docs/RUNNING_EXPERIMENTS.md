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
