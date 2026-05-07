# RA-OV3DSeg Roadmap

This project should not treat CLIP patch features as the final teacher.

## Correct Mainline

The intended research pipeline is:

```text
dense open-vocabulary 2D teacher
  -> pixel-level or mask-level semantic features/logits
  -> LiDAR-to-camera projection sampling
  -> reliability-aware 2D-to-3D distillation
  -> sparse-conv 3D student
  -> text-embedding open-vocabulary inference
```

## Teacher Backends

`clip_patch_baseline`

- Status: runnable.
- Role: MVP baseline only.
- Limitation: coarse ViT patch tokens are not reliable enough for fine point-level distillation.

`openseg_dense`

- Status: planned.
- Role: main dense teacher.
- Target: pixel-level open-vocabulary features or logits sampled at projected LiDAR locations.

`grounded_sam_mask`

- Status: planned.
- Role: high-quality mask pseudo-label teacher.
- Target: later-stage refinement, not the first production teacher.

## Backbone Backends

`debug_point_mlp`

- Status: runnable.
- Role: training harness only.
- Limitation: no local geometry context, not a final 3D segmentation model.

`sparse_unet_spconv`

- Status: implemented as an MVP-v5 SparseUNet-Lite adapter.
- Role: main 3D student backbone.
- Target: voxelize points, run sparse U-Net, gather voxel features back to points, keep CE + reliability distillation losses.

## Milestones

V5 should implement the sparse 3D student first:

```text
raw points
  -> voxelization
  -> spconv SparseUNet-Lite
  -> point-wise feature gather
  -> base CE + reliability distillation
```

V6 should replace the baseline CLIP patch teacher with a dense teacher:

```text
image
  -> dense open-vocabulary teacher
  -> dense feature/logit map
  -> projected point sampling
```

V6-A implements `clipseg_dense` as a runnable dense class-logit teacher:

```text
camera image + class prompts
  -> CLIPSeg dense class logits
  -> projected point sampling
  -> point-level dense teacher logits
```

V7 connects those dense point logits to sparse 3D training:

```text
point-level dense teacher logits
  -> base-class logit selection
  -> reliability/confidence-weighted KL distillation
  -> sparse_unet_spconv student logits
```

V8 closes the mini evaluation loop:

```text
trained sparse_unet_spconv checkpoint
  -> point-level lidarseg predictions
  -> BEV/PLY visualization
  -> base / novel / all mIoU against lidarseg labels
```

`openseg_dense` remains the preferred final dense teacher direction once a stable
checkpoint and dependency path are selected.

The final experiments should compare:

- CLIP patch baseline.
- Dense teacher without reliability.
- Dense teacher with reliability-aware distillation.
- Base-only supervised 3D student.
