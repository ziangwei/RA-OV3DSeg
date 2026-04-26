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
    nuscenes_mini.yaml

  scripts/
    check_nuscenes_sample.py
    project_lidar_to_cameras.py
    visualize_projection.py

  ra_ov3dseg/
    datasets/
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

### 2. 计算 LiDAR 到 6 个相机的投影

```bash
python scripts/project_lidar_to_cameras.py \
  --dataroot /path/to/nuscenes \
  --version v1.0-mini \
  --sample_idx 0 \
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

脚本会输出 6 张相机 overlay 图，以及一份可视化清单 JSON。

## 说明

- 不要把原始数据放进仓库。
- 所有脚本都支持命令行参数。
- 当前只面向 `v1.0-mini` 做链路验证。
- 中间结果统一写到 `outputs/` 下。
