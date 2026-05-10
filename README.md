# RA-OV3DSeg

## Project Goal

RA-OV3DSeg targets **Reliability-Aware Open-Vocabulary 3D Semantic Segmentation** for outdoor driving scenes.

The final model should not be a fixed 32-way nuScenes classifier. The main output should be a 3D point embedding aligned with a text embedding space. At inference time, users provide arbitrary class names, the text encoder embeds those names, and each 3D point is classified by cosine similarity between the point embedding and the text embeddings.

The closed-set classifier used in the MVP is only an auxiliary training/evaluation head:

- `point_embedding_head`: main open-vocabulary output for arbitrary text classes.
- `base_classifier_head`: auxiliary head for base-class CE loss during training.
- `dense_logit_head/distillation`: optional teacher supervision for known prompt sets, not the final open-vocabulary interface.

CLIP/SigLIP patch features are treated as an MVP baseline only, not as the final 2D teacher. The intended mainline is a dense open-vocabulary 2D teacher that runs inside this repository through Hugging Face Transformers. V12 uses `groupvit_dense` to provide pixel-level class logits for projected LiDAR points.

Teacher registry:

- `clip_patch_baseline`: runnable smoke-test baseline.
- `clipseg_dense`: runnable dense-logit baseline.
- `groupvit_dense`: recommended V12 dense open-vocabulary teacher; runs in the RA-OV3DSeg environment.

3D backbone registry:

- `debug_point_mlp`: runnable training harness only.
- `sparse_unet_spconv`: compact sparse-conv student used by V5-V12.
- `spconv_resunet`: stronger in-repository sparse ResUNet for V13/V14 supervised upper-bound checks; V14 can train directly from raw LiDAR + lidarseg without 2D precompute caches.

See `docs/ROADMAP.md` for the corrected project path.

## Data Scope

No extra dataset is required for the next stage beyond nuScenes-lidarseg:

- `v1.0-mini` for smoke tests.
- `v1.0-trainval` keyframe blobs for larger training/eval.
- `nuScenes-lidarseg-all-v1.0.tar.bz2` for point-level semantic labels.

We do not need nuScenes full non-keyframe sweeps for the current method. DriveLM QA data is not required for RA-OV3DSeg unless a later VLM-reasoning extension is explicitly added.

`RA-OV3DSeg` 是 `Reliability-Aware Open-Vocabulary 3D Segmentation` 的代码仓库。

当前阶段只实现 MVP-v0，目标是先在 `nuScenes v1.0-mini` 上验证以下链路是否成立：

1. 正确读取 `nuScenes` sample。
2. 正确读取 `LIDAR_TOP` 和 6 个相机。
3. 将 LiDAR 点云投影到 6 张相机图像。
4. 保存投影中间结果与 overlay 图，便于人工检查。

当前不包含完整训练、完整 `trainval` 支持、分布式训练或大规模特征缓存。

## 目录结构

```text
RA-OV3DSeg/
  README.md
  requirements.txt
  .gitignore

  configs/
    base_novel_split.yaml
    nuscenes.yaml
    nuscenes_mini.yaml

  scripts/
    check_nuscenes_sample.py
    project_lidar_to_cameras.py
    visualize_projection.py
    extract_2d_features.py
    assign_2d_features_to_points.py
    zero_shot_eval.py
    compute_reliability.py
    dry_run_training_step.py
    train_3d_segmentor.py
    verify_mvp_outputs.py

  ra_ov3dseg/
    datasets/
      nuscenes_dataset.py
      nuscenes_mini_dataset.py

    geometry/
      transforms.py
      projection.py

    models/
      image_encoder.py
      text_encoder.py
      point_feature_assigner.py
      reliability.py

    evaluation/
      metrics.py
      openvocab_eval.py

    visualization/
      visualize_projection.py
      visualize_points.py

    utils/
      config.py
      io.py
      logger.py
```

## 安装

建议先在服务器环境安装依赖：

```bash
pip install -r requirements.txt
```

## MVP-v0 用法

## 服务器侧数据准备

本仓库不负责任何本地数据下载逻辑。建议你在服务器端单独准备 `nuScenes` 数据目录，然后再运行本仓库脚本。

可以直接参考模板脚本：

```bash
bash scripts/server_prepare_nuscenes_mini.sh /path/to/nuscenes
```

如果你只想手动执行命令，最小下载链路是：

```bash
mkdir -p /path/to/nuscenes/downloads
cd /path/to/nuscenes/downloads

wget -c https://www.nuscenes.org/data/v1.0-mini.tgz
wget -c https://www.nuscenes.org/data/nuScenes-lidarseg-mini-v1.0.tar.bz2

tar -xf v1.0-mini.tgz -C /path/to/nuscenes
tar -xf nuScenes-lidarseg-mini-v1.0.tar.bz2 -C /path/to/nuscenes
```

解压后建议至少确认目录类似：

```text
/path/to/nuscenes/
  samples/
  sweeps/
  maps/
  lidarseg/
    v1.0-mini/
  v1.0-mini/
```

### 1. 检查单个 sample

```bash
python scripts/check_nuscenes_sample.py \
  --dataroot /path/to/nuscenes \
  --version v1.0-mini \
  --sample_idx 0
```

也支持小批量检查：

```bash
python scripts/check_nuscenes_sample.py \
  --dataroot /path/to/nuscenes \
  --version v1.0-mini \
  --start_idx 0 \
  --max_samples 8 \
  --output_dir outputs/checks
```

### 2. 计算 LiDAR 到 6 个相机的投影

```bash
python scripts/project_lidar_to_cameras.py \
  --dataroot /path/to/nuscenes \
  --version v1.0-mini \
  --sample_idx 0 \
  --output_dir outputs/projections
```

也支持小批量投影：

```bash
python scripts/project_lidar_to_cameras.py \
  --dataroot /path/to/nuscenes \
  --version v1.0-mini \
  --start_idx 0 \
  --max_samples 8 \
  --output_dir outputs/projections
```

脚本会输出：

- `outputs/projections/sample_0000_projection.npz`
- `outputs/projections/sample_0000_projection_summary.json`

### 3. 保存 overlay 可视化结果

```bash
python scripts/visualize_projection.py \
  --dataroot /path/to/nuscenes \
  --version v1.0-mini \
  --sample_idx 0 \
  --projection_npz outputs/projections/sample_0000_projection.npz \
  --output_dir outputs/visualizations
```

也支持直接按投影目录批量可视化：

```bash
python scripts/visualize_projection.py \
  --dataroot /path/to/nuscenes \
  --version v1.0-mini \
  --start_idx 0 \
  --max_samples 8 \
  --projection_dir outputs/projections \
  --output_dir outputs/visualizations
```

脚本会输出 6 张相机 overlay 图，以及一份可视化清单 JSON。

## 说明

- 不要把原始数据放进仓库。
- 所有脚本都支持命令行参数。
- 当前只面向 `v1.0-mini` 做链路验证。
- 中间结果统一写到 `outputs/` 下。

## MVP-v1 用法

MVP-v1 需要 PyTorch。由于服务器的 CUDA / 驱动环境各不相同，建议你先按服务器实际环境安装合适的 `torch`，
然后再执行：

```bash
pip install -r requirements.txt
```

### 1. 提取 2D 图像 patch features

```bash
python scripts/extract_2d_features.py \
  --dataroot /path/to/nuscenes \
  --version v1.0-mini \
  --sample_idx 0 \
  --teacher_backend clip_patch_baseline \
  --model_name openai/clip-vit-base-patch16 \
  --cache_dir /path/to/huggingface_cache \
  --output_dir outputs/features2d
```

### 2. 根据投影结果把 2D features 赋给 3D 点

```bash
python scripts/assign_2d_features_to_points.py \
  --sample_idx 0 \
  --projection_dir outputs/projections \
  --image_feature_dir outputs/features2d \
  --output_dir outputs/point_features
```

### 3. 做最小 zero-shot baseline

```bash
python scripts/zero_shot_eval.py \
  --sample_idx 0 \
  --point_feature_dir outputs/point_features \
  --class_names_path configs/nuscenes_lidarseg_class_names.txt \
  --cache_dir /path/to/huggingface_cache \
  --output_dir outputs/zero_shot
```

zero-shot 脚本会输出：

- 预测结果 `.npz`
- 预测摘要 `.json`
- 一份彩色 `.ply` 点云
- 一张俯视图 `BEV` 预测可视化

### 4. 一键检查 MVP 输出规格

```bash
python scripts/verify_mvp_outputs.py \
  --sample_idx 0 \
  --outputs_dir outputs \
  --stage v1 \
  --output_dir outputs/verification
```

脚本只读取现有输出，不重新跑模型。它会检查投影、2D feature、point feature、zero-shot 结果和可视化文件，
并生成：

```text
outputs/verification/sample_0000_mvp_verify_summary.json
```

## MVP-v2 用法

MVP-v2 只计算 point-level reliability score，不开始训练。

```bash
python scripts/compute_reliability.py \
  --sample_idx 0 \
  --projection_dir outputs/projections \
  --zero_shot_dir outputs/zero_shot \
  --output_dir outputs/reliability
```

然后验证 v2 输出：

```bash
python scripts/verify_mvp_outputs.py \
  --sample_idx 0 \
  --outputs_dir outputs \
  --stage v2 \
  --output_dir outputs/verification
```

## MVP-v3 用法

MVP-v3 仍然不是正式训练，只做一次单帧 dry-run，验证训练所需接口可以闭环：

- lidarseg label 能读取并映射到 base-class train ids
- novel / ignore classes 在 CE loss 中被 ignore
- 最小 point MLP 能 forward
- CE loss 和 reliability-weighted cosine distillation loss 能 backward

```bash
python scripts/dry_run_training_step.py \
  --dataroot /path/to/nuscenes \
  --version v1.0-mini \
  --sample_idx 0 \
  --point_feature_dir outputs/point_features \
  --reliability_dir outputs/reliability \
  --class_names_path configs/nuscenes_lidarseg_class_names.txt \
  --split_config configs/base_novel_split.yaml \
  --device cpu \
  --output_dir outputs/training_dryrun
```

验证 v3 输出：

```bash
python scripts/verify_mvp_outputs.py \
  --sample_idx 0 \
  --outputs_dir outputs \
  --stage v3 \
  --output_dir outputs/verification
```

## MVP-v4 用法

MVP-v4 开始做小规模真实训练，主入口是通用 3D segmentor trainer。当前可运行 backbone 是 `debug_point_mlp`，它只用于验证训练工程闭环，不是最终实验模型：

- 输入：3D point xyz。
- 监督：base classes 用 lidarseg CE loss，novel / ignore classes 在 CE 中忽略。
- 蒸馏：用已经缓存的 CLIP/SigLIP point features 做 reliability-weighted cosine distillation。
- 输出：`outputs/training_v4/train_summary.json` 和 `point_mlp_latest.pt`。

训练前必须先对同一批 `sample_idx` 跑完 v0-v2，确保 `outputs/point_features/` 和 `outputs/reliability/` 中存在对应 `.npz`。后续正式模型会在同一个入口下接 `sparse_unet_spconv` / SPVCNN 类 backbone。如果只是 smoke test，可以加 `--skip_missing_precomputed` 跳过缺少预计算输出的 sample。

单卡 H100 / A100 / V100：

```bash
python scripts/train_3d_segmentor.py \
  --dataroot ${NUSCENES_ROOT} \
  --version v1.0-mini \
  --start_idx 0 \
  --max_samples 8 \
  --backbone debug_point_mlp \
  --point_feature_dir outputs/point_features \
  --reliability_dir outputs/reliability \
  --device cuda \
  --epochs 2 \
  --batch_size 1 \
  --max_points 20000 \
  --skip_missing_precomputed \
  --amp \
  --output_dir outputs/training_v4
```

多卡 DDP：

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_3d_segmentor.py \
  --dataroot ${NUSCENES_ROOT} \
  --version v1.0-mini \
  --start_idx 0 \
  --max_samples 8 \
  --backbone debug_point_mlp \
  --point_feature_dir outputs/point_features \
  --reliability_dir outputs/reliability \
  --device cuda \
  --epochs 2 \
  --batch_size 1 \
  --max_points 20000 \
  --skip_missing_precomputed \
  --amp \
  --output_dir outputs/training_v4
```

验证 V4 训练产物：

```bash
python scripts/verify_mvp_outputs.py \
  --sample_idx 0 \
  --outputs_dir outputs \
  --stage v4 \
  --output_dir outputs/verification
```

## MVP-v5 Sparse 3D Student

MVP-v5 adds the first non-debug 3D student: `--backbone sparse_unet_spconv`.
It voxelizes LiDAR points, runs a compact spconv SparseUNet-style backbone, gathers voxel features back to points, and reuses the existing base CE + reliability-weighted distillation losses.

Install the spconv wheel that matches your PyTorch CUDA build:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
pip install -r requirements-spconv-cu118.txt
# or, for a CUDA 12.x PyTorch build:
# pip install -r requirements-spconv-cu120.txt
```

Check voxelization first:

```bash
python scripts/check_voxelization.py \
  --sample_idx 0 \
  --point_feature_dir outputs/point_features \
  --voxel_size 0.2 0.2 0.2 \
  --point_cloud_range -54 -54 -5 54 54 3 \
  --output_dir outputs/voxelization
```

Train the sparse student on mini:

```bash
python scripts/train_3d_segmentor.py \
  --dataroot ${NUSCENES_ROOT} \
  --version v1.0-mini \
  --sample_idx 0 \
  --backbone sparse_unet_spconv \
  --point_feature_dir outputs/point_features \
  --reliability_dir outputs/reliability \
  --device cuda \
  --epochs 5 \
  --batch_size 1 \
  --max_points 50000 \
  --voxel_size 0.2 0.2 0.2 \
  --point_cloud_range -54 -54 -5 54 54 3 \
  --sparse_base_channels 32 \
  --amp \
  --output_dir outputs/training_v5
```

Verify V5:

```bash
python scripts/verify_mvp_outputs.py \
  --sample_idx 0 \
  --outputs_dir outputs \
  --stage v5 \
  --training_v5_dir outputs/training_v5 \
  --output_dir outputs/verification
```

## MVP-v6 Dense Teacher

V6-A adds a runnable dense teacher path using `clipseg_dense`. It generates dense class logits for each camera image, then samples those logits at projected LiDAR point locations.

Extract dense teacher logits:

```bash
python scripts/extract_dense_teacher_logits.py \
  --dataroot /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg/data/nuscenes \
  --version v1.0-mini \
  --sample_idx 0 \
  --teacher_backend clipseg_dense \
  --model_name CIDAS/clipseg-rd64-refined \
  --cache_dir /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/huggingface_cache \
  --device cuda \
  --class_names_path configs/nuscenes_lidarseg_class_names.txt \
  --prompt_batch_size 8 \
  --output_dir outputs/dense_teacher_logits
```

Assign dense logits to points:

```bash
python scripts/assign_dense_logits_to_points.py \
  --sample_idx 0 \
  --projection_dir outputs/projections \
  --dense_teacher_dir outputs/dense_teacher_logits \
  --output_dir outputs/dense_point_logits
```

Verify V6:

```bash
python scripts/verify_mvp_outputs.py \
  --sample_idx 0 \
  --outputs_dir outputs \
  --stage v6 \
  --dense_teacher_dir outputs/dense_teacher_logits \
  --dense_point_dir outputs/dense_point_logits \
  --output_dir outputs/verification
```

## nuScenes trainval 数据准备

V4 接口仍然可以先用 mini 跑通。正式实验才需要 `v1.0-trainval`。流式下载脚本会按“下载一个压缩包 -> 解压 -> 删除压缩包”的方式控制峰值空间：

```bash
bash scripts/server_prepare_nuscenes_trainval_streaming.sh \
  --dataroot ${NUSCENES_ROOT} \
  --download_dir ${NUSCENES_ROOT}/downloads_trainval
```

如果官网直链失败，先手动从 nuScenes 官网下载到 `${NUSCENES_ROOT}/downloads_trainval`，然后：

```bash
bash scripts/server_prepare_nuscenes_trainval_streaming.sh \
  --dataroot ${NUSCENES_ROOT} \
  --download_dir ${NUSCENES_ROOT}/downloads_trainval \
  --skip_download
```

清理 trainval 数据默认只 dry-run，真正删除需要 `--yes`：

```bash
bash scripts/server_cleanup_nuscenes_trainval.sh \
  --dataroot ${NUSCENES_ROOT} \
  --download_dir ${NUSCENES_ROOT}/downloads_trainval

bash scripts/server_cleanup_nuscenes_trainval.sh \
  --dataroot ${NUSCENES_ROOT} \
  --download_dir ${NUSCENES_ROOT}/downloads_trainval \
  --yes
```

## Compact Trainval For RA-OV3DSeg

For the current RA-OV3DSeg pipeline, use `Keyframe blobs only`, not the full 29GB-per-part blob archives.
The current pipeline reads nuScenes annotated `sample` keyframes: `samples/CAM_*`, `samples/LIDAR_TOP`, `maps`, `v1.0-trainval`, and `lidarseg/v1.0-trainval`.
It does not use non-keyframe `sweeps/`.

Recommended compact trainval command for the LRZ server:

```bash
bash scripts/server_prepare_nuscenes_trainval_compact.sh \
  --dataroot /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg/data/nuscenes \
  --download_dir /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg/data/nuscenes/downloads_trainval \
  --first_part 1 \
  --num_parts 10 \
  --blob_type keyframes \
  --min_free_gb 35
```

If direct downloads fail, manually download the required archives from the nuScenes download page into the same `downloads_trainval` directory, then rerun:

```bash
bash scripts/server_prepare_nuscenes_trainval_compact.sh \
  --dataroot /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg/data/nuscenes \
  --download_dir /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg/data/nuscenes/downloads_trainval \
  --skip_download \
  --first_part 1 \
  --num_parts 10 \
  --blob_type keyframes
```

Approximate storage planning:

- Full blobs for all 10 trainval parts: roughly hundreds of GB and unnecessary for the current keyframe pipeline.
- Keyframe blobs for all 10 trainval parts: roughly 40-50GB plus metadata/lidarseg.
- With `--keep_archives` off, `wget -c` resumes interrupted archive downloads and extracted archives are deleted after each part.
- Radar and sweeps are excluded during `tar` extraction by default, so they do not temporarily consume quota.
- Extraction progress is tracked by marker files in `downloads_trainval/.extracted_compact_*`; rerunning the script skips already extracted parts.

## MVP-v7 Dense-Logit Distillation

V7 connects the V6 point-level dense teacher logits to the sparse 3D student training loop.
Use `--teacher_mode hybrid` to train with base-class CE, legacy feature distillation, and dense-logit KL distillation together.

Single-H100 mini smoke run:

```bash
python scripts/train_3d_segmentor.py \
  --dataroot /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg/data/nuscenes \
  --version v1.0-mini \
  --sample_idx 0 \
  --backbone sparse_unet_spconv \
  --teacher_mode hybrid \
  --student_output_space all_lidarseg \
  --point_feature_dir outputs/point_features \
  --reliability_dir outputs/reliability \
  --dense_point_dir outputs/dense_point_logits \
  --device cuda \
  --epochs 5 \
  --batch_size 1 \
  --max_points 50000 \
  --voxel_size 0.2 0.2 0.2 \
  --point_cloud_range -54 -54 -5 54 54 3 \
  --sparse_base_channels 32 \
  --ce_weight 1.0 \
  --distill_weight 1.0 \
  --dense_logit_weight 1.0 \
  --dense_temperature 1.0 \
  --amp \
  --output_dir outputs/training_v7
```

Verify V7:

```bash
python scripts/verify_mvp_outputs.py \
  --sample_idx 0 \
  --outputs_dir outputs \
  --stage v7 \
  --training_v7_dir outputs/training_v7 \
  --dense_teacher_dir outputs/dense_teacher_logits \
  --dense_point_dir outputs/dense_point_logits \
  --output_dir outputs/verification
```

## MVP-v8 3D Prediction And Lidarseg Eval

V8 loads a trained `train_3d_segmentor.py` checkpoint, predicts point-level lidarseg labels, then evaluates against nuScenes lidarseg labels.

Predict one mini sample:

```bash
python scripts/predict_3d_segmentor.py \
  --checkpoint outputs/training_v7/sparse_unet_spconv_latest.pt \
  --sample_idx 0 \
  --point_feature_dir outputs/point_features \
  --device cuda \
  --output_dir outputs/predictions3d
```

Evaluate one mini sample:

```bash
python scripts/eval_lidarseg.py \
  --dataroot /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg/data/nuscenes \
  --version v1.0-mini \
  --sample_idx 0 \
  --prediction_dir outputs/predictions3d \
  --class_names_path configs/nuscenes_lidarseg_class_names.txt \
  --split_config configs/base_novel_split.yaml \
  --output_dir outputs/evaluation3d
```

Verify V8:

```bash
python scripts/verify_mvp_outputs.py \
  --sample_idx 0 \
  --outputs_dir outputs \
  --stage v8 \
  --prediction_dir outputs/predictions3d \
  --evaluation_dir outputs/evaluation3d \
  --output_dir outputs/verification
```

## MVP-v9 Mini Experiment Protocol

V9 runs a small multi-sample protocol on `v1.0-mini`: shared precompute cache, train on one sample range, predict/evaluate on another range, and write one experiment summary.

```bash
python scripts/run_mini_experiment.py \
  --dataroot /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg/data/nuscenes \
  --version v1.0-mini \
  --outputs_dir outputs \
  --experiment_dir outputs/experiments/mini_v9 \
  --train_start_idx 0 \
  --train_max_samples 32 \
  --eval_start_idx 32 \
  --eval_max_samples 32 \
  --cache_dir /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/huggingface_cache \
  --precompute_device cuda \
  --train_device cuda \
  --teacher_mode hybrid \
  --student_output_space all_lidarseg \
  --epochs 10 \
  --batch_size 1 \
  --num_workers 4 \
  --max_points 50000 \
  --sparse_base_channels 32 \
  --amp \
  --skip_existing
```

Verify V9:

```bash
python scripts/verify_mvp_outputs.py \
  --outputs_dir outputs \
  --stage v9 \
  --experiment_dir outputs/experiments/mini_v9 \
  --output_dir outputs/verification
```

Main result file:

```text
outputs/experiments/mini_v9/summary.json
```
