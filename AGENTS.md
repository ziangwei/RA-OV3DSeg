# Codex Working Rules for RA-OV3DSeg

This file is read by Codex at the start of every session. Read it before any
other action.

## Stage Discipline
- The current stage is declared on line 1-3 of `docs/ROADMAP.md`.
- Work that does not match the current stage's deliverables is forbidden.
- A stage is "complete" only when all five conditions in `EXECUTION_PLAN.md`
  Section 2.1 are met.

## Document Discipline
- Only three docs/* files are allowed: ROADMAP.md, EXPERIMENT_RECAP.md,
  INTERVIEW_PREP.md.
- No `*_PLAN.md`, `*_STATUS.md`, `*_NOTES.md`, `*_TODO.md`.
- New ideas go to `docs/ROADMAP.md`. Results go to
  `docs/EXPERIMENT_RECAP.md` via `RunConclusion.append_to_recap()`.
  Narrative goes to `docs/INTERVIEW_PREP.md`.
- `EXECUTION_PLAN.md` is read-only except for Section 11 checkboxes.

## Script Naming
- Semantic names only.
- Forbidden patterns: `run_v*.sh`, `stage*_run.sh`, `experiment_*.py`,
  `v[0-9]+_*.sh`.

## Third-Party Code
- `third_party/Pointcept/` is gitignored, managed by `scripts/setup_env.sh`.
- Do not modify any file under `third_party/Pointcept/`.
- Customization wraps Pointcept from within `ra_ov3dseg/`.
- Do not use wide Pointcept imports such as `from pointcept.models import *`.
  Import the exact submodule needed, for example
  `pointcept.models.sparse_unet.spconv_unet_v1m1_base`, so the SpUNet path
  does not trigger optional Pointcept CUDA extensions such as pointops.
- If Pointcept package initialization imports optional pointops anyway, do not
  install or compile pointops for this project. For import-only sanity checks,
  inject a local `sys.modules["pointops"]` stub before importing SpUNet.

## RunConclusion
- Every training, evaluation, and extraction script ends with a
  `RunConclusion.print_block()` call.
- The block is the last thing printed. No trailing output.
- Failure cases also produce a RunConclusion with appropriate `status`.

## Stage Cleanup
- At stage completion, remove intermediate debug scripts and
  `outputs/scratch/*` artifacts.
- Run `git status`; only intentional files allowed.

## Stop Conditions
- Each stage in `EXECUTION_PLAN.md` defines a stop condition.
- On trigger: stop training, write a "Stage Retrospective" subsection to
  `docs/ROADMAP.md`, wait for user check-in.
- Gate numbers cannot be lowered. Stop conditions cannot be redefined.

## Forbidden
- New top-level planning docs.
- Modifying `EXECUTION_PLAN.md` outside Section 11.
- Modifying any file under `third_party/Pointcept/`.
- Version numbers (`v18`, `v19`) in file names.
- A second 3D backbone alongside Pointcept SpUNet.
- Skipping ablation tables in Stage 4.
- Declaring a stage complete before all five Section 2.1 conditions are met.
