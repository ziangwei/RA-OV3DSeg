# RA-OV3DSeg Interview Preparation

This document is maintained as a side-effect of normal stage work. Every
stage completion appends to relevant sections below. The final form after
Stage 5 is interview-ready.

## Elevator Pitch (30 seconds)

RA-OV3DSeg is a reliability-aware open-vocabulary 3D semantic segmentation
project for outdoor LiDAR scenes. It uses Pointcept SpUNet as a strong 3D
baseline, a SigLIP text-aligned head, and a SAM2+SigLIP 2D teacher projected
into LiDAR space to study when weak open-vocabulary pseudo-labels are useful
or harmful for 3D distillation.

## Long Pitch (3 minutes)

Filled in at Stage 5. Structure: motivation, method, results, limitations,
future work.

## Decision Log

| Decision | Chose | Rejected | Why |
|---|---|---|---|
| 3D backbone choice | Pointcept SpUNet v1m1 | Self-written Cylinder3D-style backbone; vendored partial SpUNet adapter | The prototype showed that sparse-conv recipe details dominate quality. Pointcept gives a mature published recipe, stable checkpoint format, and reproducible 70+ mIoU without maintaining a second backbone. |
| Stage 4 threshold scale | Rank-calibrated reliability weights for threshold ablation, while preserving raw multiplicative weights | Lowering the fixed threshold grid to match raw scores | The raw product is useful diagnostically but lives around 0.0-0.2 on the 128-sample cache, so the fixed [0.3, 0.5, 0.7, 0.9] grid would zero out supervision. Rank calibration keeps the planned grid meaningful without lowering the gates. |
| Project closure scope | Close Stage 4 after the 128-cache component/control pilot and document honestly | Full 6019-cache ablations, new teacher families, and a large manual OV-query benchmark | The current teacher is weak enough that further scale has poor cost/benefit. The component/control pilot answered the key question: the proposed reliability formula did not beat random/sparse controls, so the project should be framed as a bounded weak-teacher distillation study rather than a SOTA segmentation method. |

## Headline Results

| Metric | Value | Context |
|---|---|---|
| Closed-set val mIoU (Pointcept SpUNet) | 0.7432 | full nuScenes-lidarseg val, fast Pointcept SemSegEvaluator |
| Text-aligned OV head closed-set drop | -0.0033 | best mIoU 0.7465 vs Stage 1 0.7432 |
| SAM2+SigLIP teacher projected mIoU | 0.1022 | 128-sample official16 diagnostic; coverage 0.5578 |
| SAM2+SigLIP semantic top-confidence mIoU | 0.3149 / 0.3159 | top-20% / top-40% after excluding background/ignore from ranking |
| Rank-calibrated reliability cache | mean 0.4990 | 128 samples; high>=0.5 ratio 0.4990; semantic score ratio 0.4196 |
| Reliability distillation smoke | pass | threshold 0.5; distill_valid_ratio 0.1190 after Pointcept GridSample |
| Best reliability threshold | 0.9 pilot | threshold sweep only; 0.4554 vs 0.3636 at threshold 0.0, later contradicted by component/control pilot |
| Reliability component/control pilot | gate failed | full=0.0619, random=0.0793, uniform=0.0683, max component drop=0.0023 |
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
A: The intended score combined distance, geometric visibility, and semantic
confidence, but the final pilot did not validate that formula. Earlier
diagnostics showed that high-confidence semantic subsets were cleaner than the
raw teacher, and the threshold sweep suggested that strict selective
supervision could help. However, the component/control pilot showed that
`full` reliability did not beat a random same-scale control. The honest answer
is that selective filtering matters for weak teachers, but this specific
reliability formula is not yet proven better than simpler controls.

### Q: Did the stronger teacher solve pseudo-label quality?
A: No. It solved the first-order failure mode but did not produce clean dense
supervision. SAM2+SigLIP reached the Stage 3 gate at 0.1022 raw projected mIoU
on the planned 128-sample diagnostic split, which is enough to proceed but not
enough to trust all pseudo-labels. The project should now be judged by whether
Stage 4 reliability weighting improves over unfiltered distillation.

### Q: Isn't this just filtering out a bad teacher?
A: Yes, that is the final bounded conclusion. The threshold pilot showed that
using fewer high-ranked teacher labels can outperform using all labels, but
the component/control pilot did not show that the proposed reliability formula
is the reason. `full` reliability reached 0.0619 mIoU, while the random
same-scale control reached 0.0793 mIoU, and component removal caused only a
0.0023 maximum drop. I would present this as a weak-teacher diagnostic and a
negative/limited method result, not as a validated reliability method.

### Q: Why only nuScenes? What about SemanticKITTI / Waymo?
A: filled at Stage 5.

## Honest Limitations

- The SAM2+SigLIP teacher is useful but weak: raw projected mIoU is 0.1022 on
  the current 128-sample diagnostic, so Stage 4 must validate reliability
  filtering rather than assume pseudo-labels are clean supervision.
- Teacher confidence is not automatically calibrated for 3D distillation.
  Background/ignore predictions can be very confident and must be handled
  separately from semantic pseudo-label ranking.
- The final component/control pilot did not validate the proposed reliability
  formula. The defensible claim is about diagnosing and bounding weak-teacher
  distillation rather than proposing a broadly validated reliability method.

## What I Would Do With 3 More Months

Filled at Stage 5.
