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
| SAM2+SigLIP teacher projected mIoU | 0.1022 | 128-sample official16 diagnostic; coverage 0.5578 |
| SAM2+SigLIP semantic top-confidence mIoU | 0.3149 / 0.3159 | top-20% / top-40% after excluding background/ignore from ranking |
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
A: SAM2 provides model-agnostic image masks, while SigLIP supplies the
text-image similarity needed to map those masks into the nuScenes lidarseg
class space. The first implementation failed because background competed with
semantic classes and raw lidarseg names such as `driveable_surface` were poor
text prompts. After removing background from mask classification and using
driving-scene phrases, the 5-sample projected teacher diagnostic reached 0.1148
mIoU, above the 0.10 Stage 3 gate. A larger 32-sample cross-scene diagnostic
then reached 0.1376 mIoU with 0.5464 coverage, and the planned 128-sample
diagnostic reached 0.1022 mIoU with 0.5578 coverage. This is still a weak
teacher, so Stage 4 must treat it as noisy pseudo-label supervision rather
than as ground truth.

### Q: Your novel-class mIoU is low. Is OV actually working?
A: filled at Stage 4. This is the project's sharpest question and must be
answered honestly.

### Q: How does your reliability score differ from a confidence threshold?
A: Confidence alone is not enough. In Stage 3, naive confidence ranking was
misleading because background/ignore predictions dominated high-confidence
points; 58.22% of score-valid points were excluded on the 128-sample
diagnostic when ranking only semantic pseudo-labels. After that correction,
semantic top-20% and top-40% subsets reached 0.3149 and 0.3159 mIoU,
suggesting that confidence is useful only as one component of a reliability
score, not as the whole method.

### Q: Why only nuScenes? What about SemanticKITTI / Waymo?
A: filled at Stage 5.

## Honest Limitations

- The SAM2+SigLIP teacher is useful but weak: raw projected mIoU is 0.1022 on
  the current 128-sample diagnostic, so Stage 4 must validate reliability
  filtering rather than assume pseudo-labels are clean supervision.
- Teacher confidence is not automatically calibrated for 3D distillation.
  Background/ignore predictions can be very confident and must be handled
  separately from semantic pseudo-label ranking.

## What I Would Do With 3 More Months

Filled at Stage 5.
