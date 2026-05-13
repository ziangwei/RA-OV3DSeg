# RA-OV3DSeg Roadmap

> **Current Stage**: stage-baseline
> **Last Updated**: 2026-05-13

This is the live operational plan. The static execution plan is in
`EXECUTION_PLAN.md` and must not be modified. New decisions and adjustments
that occur during execution go in this file.

## Current Stage

**stage-baseline**: reproduce Pointcept SpUNet nuScenes-lidarseg validation
mIoU >= 0.70 using Pointcept's own training launcher and recipe. See
`EXECUTION_PLAN.md` Section 4.

## Next Experiment

First Stage 1 experiment: Pointcept preprocessing plus a baseline launcher
smoke run. The smoke run must prove that data loading, Pointcept launcher
execution, log capture, mIoU parsing, and `RunConclusion` emission work before
any full trainval baseline run. `SMOKE=1` creates tiny Pointcept train/val pkl
files under `outputs/pointcept/smoke_data/` so evaluation does not traverse the
full 6019-sample split.

Server discovery commands:

```bash
find third_party/Pointcept -name "preprocess_*.py" -path "*nuscenes*"
ls third_party/Pointcept/configs/nuscenes
grep -rn "data_root" third_party/Pointcept/configs/nuscenes/*spunet*.py
```

Expected preprocessing command:

```bash
python scripts/preprocess_nuscenes_trainval_only.py \
  --dataset_root "$PWD/data/nuscenes" \
  --output_root "$PWD/data/nuscenes_pointcept_processed" \
  --max_sweeps 1 \
  --with_camera

ln -sfn "$PWD/data/nuscenes" "$PWD/data/nuscenes_pointcept_processed/raw"
```

Record the confirmed config path and processed data root here before full
training:
- Pointcept SpUNet config: TBD on server
- nuScenes raw root: `$PWD/data/nuscenes` on server
- Pointcept processed root: `$PWD/data/nuscenes_pointcept_processed` on server
- Smoke log: `outputs/logs/train_baseline_<timestamp>.log`

Do not run the setup in the current Windows base Python 3.13 environment.
Create or activate a clean Python 3.10 CUDA environment first, then run the
setup script from Git Bash, WSL, or the target Linux training server.

Pointcept commit `d74c646db6abec569d0f23e0c34e7ddfce142789` may not expose
`setup.py` or `pyproject.toml`. In that case `scripts/setup_env.sh` registers
`third_party/Pointcept` through `pointcept.pth` in the active Python
environment instead of running `pip install -e`.

Server environment target observed during Phase 0:
- `numpy==1.26.4`
- `opencv-python-headless==4.8.1.78`
- PyG extension wheels from
  `https://data.pyg.org/whl/torch-2.1.0+cu121.html`.
- `pointops` is intentionally not installed. Stage 1-5 use SpUNet through
  spconv, and pointops is only needed for Pointcept PTv3-style paths.

## Stage History

### stage-baseline (in progress)
- Goal: reproduce Pointcept SpUNet nuScenes-lidarseg val mIoU >= 0.70.
- Status: preprocessing/smoke preparation in progress.
- Next check: smoke run `scripts/train_baseline.sh` and inspect
  `RunConclusion` before authorizing full 50 epoch training.

### phase-0 (complete)
- Goal: clean repo, install Pointcept, pass sanity check.
- Status: complete. Server setup and sanity check passed on 2026-05-13.
- Retrospective: no stop condition fired. Pointcept is registered through
  `pointcept.pth`; `pointops` is intentionally skipped because this project
  uses SpUNet through spconv, not PTv3 paths.

## Pivots and Adjustments

If Stage 3 or Stage 4 triggers a project pivot, document it here with date,
trigger, and new framing.
