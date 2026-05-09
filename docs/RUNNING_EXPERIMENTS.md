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

## V10+ Rule

New training or evaluation stages should follow the same pattern:

- Add one `scripts/run_v*_*.sh` wrapper.
- Put logs under `outputs/logs/`.
- Keep experiment artifacts under `outputs/experiments/<experiment_name>/`.
- Make the Python script configurable, but make the bash wrapper the normal
  server entrypoint.
