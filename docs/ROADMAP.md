# RA-OV3DSeg Roadmap

> **Current Stage**: stage-ov-head
> **Last Updated**: 2026-05-20

This is the live operational plan. The static execution plan is in
`EXECUTION_PLAN.md` and must not be modified. New decisions and adjustments
that occur during execution go in this file.

## Current Stage

**stage-ov-head**: replace the closed-set classifier with a SigLIP
text-prototype cosine head while keeping closed-set mIoU within 0.08 of the
Stage 1 baseline. See `EXECUTION_PLAN.md` Section 5.

## Next Experiment

Next Stage 2 experiment: implement the text-prototype OV head and run a smoke
fine-tune from `outputs/checkpoints/closed_set_baseline.pt`. The acceptance
threshold is val mIoU >= 0.6632, computed from the Stage 1 baseline
0.7432 - 0.08.

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

### stage-ov-head (in progress)
- Goal: replace the closed-set head with a SigLIP prototype cosine head while
  preserving most closed-set performance.
- Status: OV head module and text prototype cache script implemented locally.
- Next check: cache 16 class text prototypes on the server, then write a smoke
  fine-tune launcher before any full fine-tune.

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
