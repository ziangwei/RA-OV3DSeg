# RA-OV3DSeg Interview Preparation

This document is maintained as a side-effect of normal stage work. Every
stage completion appends to relevant sections below. The final form after
Stage 5 is interview-ready.

## Elevator Pitch (30 seconds)

Rewritten at each stage to reflect current project state. Phase 0 version
is a placeholder.

> Placeholder: RA-OV3DSeg is a project on reliability-aware open-vocabulary
> 3D semantic segmentation on outdoor LiDAR scenes. Status: foundation
> reset in progress.

## Long Pitch (3 minutes)

Filled in at Stage 5. Structure: motivation, method, results, limitations,
future work.

## Decision Log

| Decision | Chose | Rejected | Why |
|---|---|---|---|

## Headline Results

| Metric | Value | Context |
|---|---|---|
| Closed-set val mIoU (Pointcept SpUNet) | TBD | full nuScenes-lidarseg trainval |
| Text-aligned OV head closed-set drop | TBD | vs closed-set baseline |
| SAM2+SigLIP teacher projected mIoU | TBD | on diagnostic split |
| Best reliability threshold | TBD | from ablation |
| OV-query retrieval@5 | TBD | on hand-curated benchmark |

## Anticipated Questions & Answers

### Q: Why didn't you implement your own LiDAR backbone?
A: filled at Stage 1.

### Q: Why SAM2 + SigLIP instead of CAT-Seg or OpenSeg?
A: filled at Stage 3.

### Q: Your novel-class mIoU is low. Is OV actually working?
A: filled at Stage 4. This is the project's sharpest question and must be
answered honestly.

### Q: How does your reliability score differ from a confidence threshold?
A: filled at Stage 4.

### Q: Why only nuScenes? What about SemanticKITTI / Waymo?
A: filled at Stage 5.

## Honest Limitations

One bullet per limitation. Filled as discovered.

## What I Would Do With 3 More Months

Filled at Stage 5.
