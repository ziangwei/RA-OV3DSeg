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

## Stage History

### phase-0 (in progress)
- Goal: clean repo, install Pointcept, pass sanity check.
- Status: in progress.
- Retrospective: filled if a stop condition fires.

## Pivots and Adjustments

If Stage 3 or Stage 4 triggers a project pivot, document it here with date,
trigger, and new framing.
