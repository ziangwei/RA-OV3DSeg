# RA-OV3DSeg Roadmap

> **Current Stage**: phase-0
> **Last Updated**: 2026-05-12

This is the live operational plan. The static execution plan is in
`EXECUTION_PLAN.md` and must not be modified. New decisions and adjustments
that occur during execution go in this file.

## Current Stage

**phase-0**: foundation reset. See `EXECUTION_PLAN.md` Section 3.

## Next Experiment

Run `scripts/setup_env.sh` in a clean conda environment, then run
`scripts/sanity_check.sh`. On success, the `RunConclusion` expectation for
later stage scripts is a final parseable block with `status=success` and
`gate_passed=yes`. Phase 0 itself is complete only after the environment
provisions Pointcept at the pinned commit and the sanity check passes.

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
- `pointops` must be compiled from `third_party/Pointcept/libs/pointops` on a
  CUDA build/compute node. `scripts/setup_env.sh` does this only when
  `INSTALL_POINTOPS=1` is set.

## Stage History

### phase-0 (in progress)
- Goal: clean repo, install Pointcept, pass sanity check.
- Status: in progress.
- Retrospective: filled if a stop condition fires.

## Pivots and Adjustments

If Stage 3 or Stage 4 triggers a project pivot, document it here with date,
trigger, and new framing.
