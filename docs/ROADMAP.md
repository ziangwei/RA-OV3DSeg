# RA-OV3DSeg Roadmap

> **Current Stage**: phase-0
> **Last Updated**: 2026-05-13

This is the live operational plan. The static execution plan is in
`EXECUTION_PLAN.md` and must not be modified. New decisions and adjustments
that occur during execution go in this file.

## Current Stage

**phase-0**: foundation reset. See `EXECUTION_PLAN.md` Section 3.

## Next Experiment

Open the Stage 1 C1 check-in before starting full baseline training:
confirm the Pointcept SpUNet config path, nuScenes trainval dataroot, target
launcher command, logging directory, and expected runtime. Stage 1 begins only
after this check-in.

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

### phase-0 (complete)
- Goal: clean repo, install Pointcept, pass sanity check.
- Status: complete. Server setup and sanity check passed on 2026-05-13.
- Retrospective: no stop condition fired. Pointcept is registered through
  `pointcept.pth`; `pointops` is intentionally skipped because this project
  uses SpUNet through spconv, not PTv3 paths.

## Pivots and Adjustments

If Stage 3 or Stage 4 triggers a project pivot, document it here with date,
trigger, and new framing.
