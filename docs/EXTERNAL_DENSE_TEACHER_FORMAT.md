# External Dense Teacher Format

V12 consumes dense open-vocabulary 2D teacher outputs from external projects
such as CAT-Seg or OpenSeg. Those projects should write one `.npz` per nuScenes
sample.

## File Name

```text
sample_XXXX_dense_teacher_logits.npz
```

Example:

```text
outputs/external_teachers/catseg_dense/sample_0000_dense_teacher_logits.npz
```

## Required Keys

```text
sample_idx                 int scalar
sample_token               string scalar
teacher_backend            string scalar, e.g. catseg_dense
model_name                 string scalar
camera_names               string array, shape (6,)
camera_available           bool array, shape (6,)
image_widths               int array, shape (6,)
image_heights              int array, shape (6,)
class_names                string array, shape (C,)
prompts                    string array, shape (C,)
dense_logits               float array, shape (6, C, H, W) or (6, H, W, C)
```

`camera_names` must be exactly:

```text
CAM_FRONT
CAM_FRONT_LEFT
CAM_FRONT_RIGHT
CAM_BACK
CAM_BACK_LEFT
CAM_BACK_RIGHT
```

`class_names` must start with the 32 nuScenes-lidarseg names from:

```text
configs/nuscenes_lidarseg_class_names.txt
```

Extra classes may be appended after those 32 names, but the current V12 training
uses the first 32 for lidarseg-compatible dense-logit distillation.

## Validation

```bash
python scripts/check_external_dense_teacher_logits.py \
  --start_idx 0 \
  --max_samples 128 \
  --dense_teacher_dir outputs/external_teachers/catseg_dense \
  --projection_dir outputs/experiments/trainval_v9_128_isolated/precompute/projections \
  --output_dir outputs/external_teacher_checks/catseg_dense_128
```

## Point Assignment

```bash
python scripts/assign_dense_logits_to_points.py \
  --start_idx 0 \
  --max_samples 128 \
  --projection_dir outputs/experiments/trainval_v9_128_isolated/precompute/projections \
  --dense_teacher_dir outputs/external_teachers/catseg_dense \
  --output_dir outputs/experiments/trainval_v12_external_teacher_128/precompute/dense_point_logits
```
