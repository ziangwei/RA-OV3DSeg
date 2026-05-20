# RA-OV3DSeg Interview Preparation

This document is maintained as a side-effect of normal stage work. Every
stage completion appends to relevant sections below. The final form after
Stage 5 is interview-ready.

## Elevator Pitch (30 seconds)

RA-OV3DSeg is a reliability-aware open-vocabulary 3D semantic segmentation
project for outdoor LiDAR scenes. The current foundation is a Pointcept SpUNet
closed-set baseline on nuScenes-lidarseg, which reaches 0.7432 val mIoU and
provides the checkpoint for the upcoming text-aligned head replacement.

## Long Pitch (3 minutes)

Filled in at Stage 5. Structure: motivation, method, results, limitations,
future work.

## Decision Log

| Decision | Chose | Rejected | Why |
|---|---|---|---|
| 3D backbone choice | Pointcept SpUNet v1m1 | Self-written Cylinder3D-style backbone; vendored partial SpUNet adapter | The prototype showed that sparse-conv recipe details dominate quality. Pointcept gives a mature published recipe, stable checkpoint format, and reproducible 70+ mIoU without maintaining a second backbone. |

## Headline Results

| Metric | Value | Context |
|---|---|---|
| Closed-set val mIoU (Pointcept SpUNet) | 0.7432 | full nuScenes-lidarseg val, fast Pointcept SemSegEvaluator |
| Text-aligned OV head closed-set drop | -0.0033 | best mIoU 0.7465 vs Stage 1 0.7432 |
| SAM2+SigLIP teacher projected mIoU | TBD | on diagnostic split |
| Best reliability threshold | TBD | from ablation |
| OV-query retrieval@5 | TBD | on hand-curated benchmark |

## Anticipated Questions & Answers

### Q: Why didn't you implement your own LiDAR backbone?
A: The prototype proved that implementing the backbone was not the research
bottleneck. A self-written sparse-conv baseline underperformed and consumed
time on adapter details, class mapping, and CUDA recipe issues. Using
Pointcept SpUNet keeps the 3D backbone close to a known recipe, reaches 0.7432
val mIoU, and lets the project focus on the actual contribution: text-aligned
heads, teacher signals, and reliability weighting.

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
