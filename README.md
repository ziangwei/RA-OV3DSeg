# RA-OV3DSeg

## Project Route Correction

CLIP/SigLIP patch features are now treated as an MVP baseline only, not as the final 2D teacher. The intended mainline is a dense open-vocabulary 2D teacher, such as `openseg_dense`, that provides pixel-level semantic features or logits for projected LiDAR points.

Teacher registry:

- `clip_patch_baseline`: runnable smoke-test baseline.
- `openseg_dense`: planned main dense teacher.
- `grounded_sam_mask`: planned high-quality mask pseudo-label teacher.

3D backbone registry:

- `debug_point_mlp`: runnable training harness only.
- `sparse_unet_spconv`: planned V5 sparse-conv 3D student.

See `docs/ROADMAP.md` for the corrected project path.

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
