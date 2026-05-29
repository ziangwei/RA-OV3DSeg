# RA-OV3DSeg Interview Preparation

This document records the final interview framing after closing Stage 4 as a
negative/limited weak-teacher distillation study.

## Elevator Pitch (30 seconds)

RA-OV3DSeg is a reliability-aware open-vocabulary 3D semantic segmentation
project for outdoor LiDAR scenes. It uses Pointcept SpUNet as a strong 3D
baseline, a SigLIP text-aligned head, and a SAM2+SigLIP 2D teacher projected
into LiDAR space to study when weak open-vocabulary pseudo-labels are useful
or harmful for 3D distillation.

## Long Pitch (3 minutes)

RA-OV3DSeg started as an attempt to build open-vocabulary 3D semantic
segmentation for outdoor LiDAR by distilling 2D open-vocabulary teachers into a
Pointcept SpUNet student. The first part of the project established a reliable
base: Pointcept SpUNet reached 0.7432 val mIoU on nuScenes-LiDARSeg, and a
SigLIP text-prototype head preserved closed-set performance at roughly the same
level. I then built a SAM2+SigLIP mask-then-classify teacher, projected the
six-camera outputs into LiDAR space, and measured the teacher directly. The raw
teacher was weak at 0.1022 mIoU on a 128-sample diagnostic, but semantic
high-confidence subsets were much cleaner, around 0.315 mIoU. That motivated a
reliability-weighted distillation stage. The threshold pilot suggested strict
filtering could help, but the final component/control pilot showed the proposed
distance/geometric/semantic reliability formula did not beat a random same-scale
control. I closed the project as a weak-teacher distillation diagnostic: the
pipeline is real and the negative result is informative, but the method itself
is not validated as a positive contribution.

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
| OV-query retrieval@5 | not pursued | stopped after Stage 4 control failure |

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
A: Not in the strong sense originally intended. The text-aligned head can
preserve closed-set performance, and the 2D teacher produces some cleaner
semantic subsets, but the final reliability distillation evidence is not enough
to claim robust open-vocabulary 3D segmentation. I would frame the result as an
honest diagnostic of weak 2D teacher transfer, not as a successful OV model.

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
A: The project was stopped after the Stage 4 control experiment because the
core reliability formulation was not validated. Extending to SemanticKITTI or
Waymo would only make sense after replacing the weak teacher or developing a
stronger control-beating reliability signal. Scaling datasets before that would
increase cost without fixing the core failure mode.

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

1. Replace the teacher first, not the student: evaluate SAM3, CAT-Seg, or
   Grounded-SAM-style teachers on the same 128-sample projected mIoU diagnostic.
2. Add strict controls before training: random same keep-ratio, confidence-only,
   and per-class balanced filtering.
3. Only run full distillation if the teacher/control diagnostic is positive.
4. Keep Pointcept as the 3D backbone and spend effort on teacher quality,
   calibration, and clean ablation design.
