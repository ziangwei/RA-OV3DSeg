# Stage 3 Teacher Comparison

Date: 2026-05-20
Stage: `stage-teacher`
Label space: official nuScenes-lidarseg 16 semantic classes, with void ignored.

## Summary

SAM2+SigLIP passes the Stage 3 teacher gate on the planned 128-sample
diagnostic split, but only narrowly in raw projected mIoU. The important signal
is not that the full pseudo-label field is clean; it is that semantic
confidence filtering exposes a much cleaner subset for Stage 4 reliability
distillation.

## Comparison Table

| Teacher | Status | Diagnostic scope | Projected mIoU | Coverage | Notes |
|---|---|---:|---:|---:|---|
| CLIP patch | rejected baseline | prototype carryover | not accepted | not accepted | Patch tokens were too coarse for point-level outdoor segmentation and are retained only as a smoke-test baseline. |
| CLIPSeg | runnable weak dense baseline | prototype carryover | not accepted | not accepted | Dense class-logit path worked, but no tracked Stage 3 comparable diagnostic result was accepted as the final teacher. |
| GroupViT | rejected dense teacher | prototype carryover | 0.0195 | not recorded | Too weak for direct distillation; Stage 3 gate was set to 0.10, about 5x this value. |
| SAM2+SigLIP | accepted Stage 3 teacher candidate | 5-sample diagnostic | 0.1148 | 0.4209 | First post-fix smoke diagnostic after removing background competition and improving class prompts. |
| SAM2+SigLIP | accepted Stage 3 teacher candidate | 32-sample cross-scene diagnostic | 0.1376 | 0.5464 | Semantic top-20/top-40 confidence mIoU was 0.3002/0.3386 after excluding background/ignore from ranking. |
| SAM2+SigLIP | accepted Stage 3 teacher candidate | 128-sample planned diagnostic | 0.1022 | 0.5578 | Semantic top-20/top-40 confidence mIoU was 0.3149/0.3159; 58.22% of score-valid points were excluded from semantic ranking as background/ignore. |

## Stage 4 Handoff

- Do not train Stage 4 as if the full SAM2+SigLIP pseudo-label map were clean.
- Use reliability filtering as the main hypothesis: semantic high-confidence
  subsets are much cleaner than the raw teacher field.
- Keep background/ignore out of semantic confidence ranking, but report the
  excluded ratio because it explains why naive confidence-only filtering is
  misleading.
- Stage 4 should compare non-zero reliability thresholds against
  threshold=0 and run the planned component ablation table.

## Source Artifacts

- `outputs/teacher_quality/sam2_siglip_stage3_32_semconf/batch_teacher_pseudo_eval_summary.json`
- `outputs/teacher_quality/sam2_siglip_stage3_32_semconf/batch_teacher_pseudo_eval.npz`
- `outputs/teacher_quality/sam2_siglip_stage3_128/batch_teacher_pseudo_eval_summary.json`
- `outputs/teacher_quality/sam2_siglip_stage3_128/batch_teacher_pseudo_eval.npz`
