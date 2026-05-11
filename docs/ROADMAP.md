# RA-OV3DSeg Roadmap

This project should not treat CLIP patch features as the final teacher, and it
should not treat a fixed 32-way classifier as the final open-vocabulary model.

For the current experiment history, trusted metrics, and output cleanup rules,
see `docs/EXPERIMENT_RECAP.md`.

## Correct Mainline

The intended research pipeline is:

```text
dense open-vocabulary 2D teacher
  -> pixel-level or mask-level text-aligned semantic features/logits
  -> LiDAR-to-camera projection sampling
  -> reliability-aware 2D-to-3D distillation
  -> sparse-conv 3D student point embeddings
  -> arbitrary text prompts at inference
  -> cosine(point_embedding, text_embedding) open-vocabulary classification
```

The student should keep two roles separate:

```text
3D backbone
  -> point embedding head        # final open-vocabulary inference path
  -> base classifier head        # auxiliary CE loss on base classes only
```

The 32-class lidarseg classifier/logit distillation used in the MVP is an
engineering scaffold for supervised debugging and teacher-signal validation. It
is not the final open-vocabulary interface.

## Data Requirement

The dataset requirement remains nuScenes-lidarseg:

- `v1.0-mini` for smoke tests.
- `v1.0-trainval` keyframe blobs for larger experiments.
- `nuScenes-lidarseg-all-v1.0.tar.bz2` for point-level labels.

No DriveLM data is required for the core RA-OV3DSeg method. Full non-keyframe
sweeps are not required by the current pipeline.

## Teacher Backends

`clip_patch_baseline`

- Status: runnable.
- Role: MVP baseline only.
- Limitation: coarse ViT patch tokens are not reliable enough for fine point-level distillation.

`clipseg_dense`

- Status: runnable.
- Role: weak dense-logit baseline.
- Limitation: useful for pipeline checks, not a final teacher for nuScenes-lidarseg.

`groupvit_dense`

- Status: runnable.
- Role: current single-environment dense open-vocabulary teacher.
- Limitation: V12 shows it is not strong enough to trust blindly; it needs teacher-quality gating.

## Backbone Backends

`debug_point_mlp`

- Status: runnable.
- Role: training harness only.
- Limitation: no local geometry context, not a final 3D segmentation model.

`sparse_unet_spconv`

- Status: implemented as an MVP-v5 SparseUNet-Lite adapter.
- Role: compact 3D student backbone.
- Target: voxelize points, run sparse U-Net, gather voxel features back to points, keep CE + reliability distillation losses.

`spconv_resunet`

- Status: implemented for V13 diagnostics.
- Role: stronger in-repository 3D student for supervised upper-bound checks.
- Target: determine whether the current bottleneck is 3D capacity before spending more compute on open-vocabulary teacher distillation.

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

V7 connects those dense point logits to sparse 3D training as a temporary closed-set scaffold:

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

V9 turns the single-sample loop into a small mini protocol:

```text
shared precompute cache
  -> train split range
  -> eval split range
  -> aggregate mini mIoU summary
```

V10 should restore the open-vocabulary inference path as a first-class target:

```text
trained 3D point embeddings
  -> arbitrary class_names_csv / class_names_path
  -> text encoder
  -> cosine similarity prediction
  -> base / novel / all mIoU against lidarseg when labels are available
```

V10 implementation status:

- `scripts/predict_3d_open_vocab.py` loads a trained 3D checkpoint, computes
  point embeddings, encodes arbitrary text classes, and predicts by cosine
  similarity.
- `scripts/run_v10_open_vocab_eval.sh` runs the default full lidarseg 32-class
  text-query evaluation using the V9 isolated checkpoint and precomputed point
  features.
- Closed-set classifier logits are no longer used for V10 prediction; they remain
  an auxiliary training scaffold.

V11 addresses the main V10 failure mode:

```text
V9 checkpoint
  -> supervised base-class text prototype alignment loss
  -> text-aligned 3D point embeddings
  -> V10-style arbitrary text inference/eval
```

V11 is still not the final teacher upgrade. It is a controlled check that the
3D embedding head can be pulled into the same CLIP/SigLIP text space before
replacing the patch teacher with a dense open-vocabulary model.

After V11, teacher upgrade should focus on improving text-aligned dense teacher
quality, not on adding new datasets.

`groupvit_dense` is the current dense teacher direction because it runs through
the same RA-OV3DSeg environment and Hugging Face cache.

V12 introduces a Transformers-native dense teacher first:

```text
nuScenes camera images
  -> GroupViT zero-shot dense segmentation
  -> projected point-level teacher logits
  -> sparse 3D student with reliability + text alignment
  -> open-vocabulary point embedding inference
```

This keeps the workflow inside the existing RA-OV3DSeg repository and conda
environment.

V13 stops adding more weak-teacher training and adds two diagnostic gates:

```text
projected dense teacher logits
  -> direct teacher pseudo-label mIoU against lidarseg

full lidarseg supervision
  -> spconv_resunet closed-set upper-bound mIoU
```

Decision rule:

- If teacher pseudo-label mIoU is poor, reliability-aware distillation cannot fix the teacher alone.
- If supervised `spconv_resunet` is poor, the 3D student capacity or training recipe must be fixed first.
- Only if both gates are acceptable should the project spend more compute on open-vocabulary distillation.

V14 focuses on the second V13 failure mode: the 3D closed-set upper bound is not
strong enough yet.

```text
spconv_resunet
  -> raw LiDAR + lidarseg labels, no 2D precompute dependency
  -> class-balanced CE
  -> Lovasz/Dice optional IoU losses
  -> basic LiDAR augmentation
  -> in-training eval
  -> best checkpoint by eval mIoU
```

V14 deliberately does not add a new teacher. A stronger teacher should only be
introduced after the 3D supervised recipe has a credible baseline.

V15 stops treating the in-house Cartesian ResUNet as the final backbone and
switches the supervised baseline to a Cylinder3D-style sparse backbone:

```text
raw LiDAR xyz + intensity
  -> cylindrical voxel grid [radius, azimuth, z]
  -> asymmetric spconv residual U-Net
  -> point-level lidarseg logits
```

This is the current default path for making the closed-set baseline credible
before any further open-vocabulary or reliability-distillation claims.

V16a fixes the major V15 interpretation problems:

```text
raw LiDAR xyz + intensity
  -> official nuScenes-lidarseg 16-class mapping
  -> expanded point-cloud range for near-complete prediction coverage
  -> cylinder_spconv_unet supervised baseline
```

V16a is the first credible supervised baseline in this project. The 1024 train /
512 eval run reached about `0.4159` mIoU with about `0.9983` prediction coverage.

V17 integrates a mature vendored Pointcept SpUNet backbone without switching
repositories or environments:

```text
third_party/pointcept_spunet/
  -> ra_ov3dseg.models.pointcept_spunet_adapter
  -> existing train_3d_segmentor / predict_3d_segmentor / eval_lidarseg scripts
```

Only the headfixed V17 runs should be used for decisions. The first V17 adapter
had an extra embedding bottleneck and is recorded only as a negative result.

The final experiments should compare:

- CLIP patch baseline.
- Dense teacher without reliability.
- Dense teacher with reliability-aware distillation.
- Base-only supervised 3D student.
- Closed-set auxiliary classifier vs text-embedding open-vocabulary inference.
