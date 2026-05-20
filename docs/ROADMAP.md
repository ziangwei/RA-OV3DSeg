# RA-OV3DSeg Roadmap

> **Current Stage**: stage-teacher
> **Last Updated**: 2026-05-20

This is the live operational plan. The static execution plan is in
`EXECUTION_PLAN.md` and must not be modified. New decisions and adjustments
that occur during execution go in this file.

## Current Stage

**stage-teacher**: build a SAM2 + SigLIP dense open-vocabulary teacher,
project its pseudo-labels to nuScenes LiDAR points, and check whether the
teacher is strong enough for Stage 4 distillation. See `EXECUTION_PLAN.md`
Section 6.

## Next Experiment

Next Stage 3 experiment: implement a SAM2 + SigLIP teacher wrapper and run a
diagnostic extraction/evaluation on a small nuScenes subset. The gate is
projected teacher mIoU >= 0.10; if the teacher remains weak, record the pivot
decision before Stage 4.

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
- Smoke log: `outputs/logs/train_baseline_20260513_111528.log`
- Fast val log: `outputs/logs/eval_baseline_fast_20260514_004248.log`
- Fast val result: mIoU 0.7432 / mAcc 0.8095 / allAcc 0.9321

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

### stage-teacher (in progress)
- Goal: produce dense teacher pseudo-labels from SAM2 masks classified by
  SigLIP text prototypes, then project them to LiDAR points.
- Status: branch created on 2026-05-20.
- Next check: inspect current projection and dense-teacher utilities, add SAM2
  dependency path, and implement a small diagnostic extraction script.

### stage-ov-head (complete)
- Goal: replace the closed-set head with a SigLIP prototype cosine head while
  preserving most closed-set performance.
- Status: gate passed on 2026-05-20. Best val mIoU was 0.7465, with final eval
  at 0.7449. This is above the 0.6632 threshold and does not drop from the
  Stage 1 baseline.
- Retrospective: no stop condition fired. The result should be framed as
  matching the closed-set baseline within noise, not as a meaningful
  improvement over Stage 1.

### stage-baseline (complete)
- Goal: reproduce Pointcept SpUNet nuScenes-lidarseg val mIoU >= 0.70.
- Status: accepted. Fast full-val gate passed on 2026-05-14 with mIoU 0.7432.
- C2 check-in: baseline is credible for Stage 2. It uses Pointcept SpUNet's
  own recipe and evaluator; routine runs skip the slow `PreciseEvaluator`.
- Retrospective: no stop condition fired. The final checkpoint is preserved as
  `outputs/checkpoints/closed_set_baseline.pt` on the server. The Stage 2
  threshold is 0.6632 mIoU, computed as 0.7432 - 0.08.

### phase-0 (complete)
- Goal: clean repo, install Pointcept, pass sanity check.
- Status: complete. Server setup and sanity check passed on 2026-05-13.
- Retrospective: no stop condition fired. Pointcept is registered through
  `pointcept.pth`; `pointops` is intentionally skipped because this project
  uses SpUNet through spconv, not PTv3 paths.

## Pivots and Adjustments

If Stage 3 or Stage 4 triggers a project pivot, document it here with date,
trigger, and new framing.
