# RA-OV3DSeg

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
    nuscenes.yaml
    nuscenes_mini.yaml

  scripts/
    check_nuscenes_sample.py
    project_lidar_to_cameras.py
    visualize_projection.py
    extract_2d_features.py
    assign_2d_features_to_points.py
    zero_shot_eval.py
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
