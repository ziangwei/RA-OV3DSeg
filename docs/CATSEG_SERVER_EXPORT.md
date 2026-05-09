# CAT-Seg Server Export For V12

This document describes how to run CAT-Seg outside the RA-OV3DSeg environment
and export canonical dense teacher logits for V12.

## 1. Generate The RA-OV3DSeg Manifest

Run this inside the RA-OV3DSeg environment:

```bash
cd /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg

bash scripts/run_v12_external_teacher_training.sh --manifest_only
```

The manifest path is:

```text
outputs/experiments/trainval_v12_external_teacher_128/external_teacher_manifest/train_0_128.jsonl
```

## 2. Prepare A Separate CAT-Seg Environment

CAT-Seg is a Detectron2 project. Keep it outside the RA-OV3DSeg conda env.
The CAT-Seg README recommends Python 3.8, PyTorch 1.13.1, CUDA 11.7, and
Detectron2-compatible torchvision.

Example server setup:

```bash
cd /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test

git clone https://github.com/cvlab-kaist/CAT-Seg.git
cd CAT-Seg

/dss/dssmcmlfs01/pn39qo/pn39qo-dss-0000/di97fer/miniconda3/bin/conda create -n catseg python=3.8 -y
source /dss/dssmcmlfs01/pn39qo/pn39qo-dss-0000/di97fer/miniconda3/etc/profile.d/conda.sh
conda activate catseg

conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia -y
pip install -r requirements.txt
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

If your cluster image cannot compile Detectron2, use the installation route
recommended by the cluster or Detectron2 docs for the active CUDA/PyTorch pair.

## 3. Download CAT-Seg Weights

```bash
mkdir -p checkpoints
wget -c -O checkpoints/model_base.pth \
  https://huggingface.co/spaces/hamacojr/CAT-Seg-weights/resolve/main/model_base.pth
```

For the larger model:

```bash
wget -c -O checkpoints/model_large.pth \
  https://huggingface.co/spaces/hamacojr/CAT-Seg-weights/resolve/main/model_large.pth
```

Start with `model_base.pth`. The large model is heavier and should wait until
the base export path is working.

## 4. Copy Or Reference The Export Adapter

The adapter lives in RA-OV3DSeg:

```text
RA-OV3DSeg/tools/export_catseg_dense_logits.py
```

Run it from the CAT-Seg repo root:

```bash
cd /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/CAT-Seg
conda activate catseg

python /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg/tools/export_catseg_dense_logits.py \
  --catseg_root . \
  --manifest_jsonl /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg/outputs/experiments/trainval_v12_external_teacher_128/external_teacher_manifest/train_0_128.jsonl \
  --output_dir /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg/outputs/external_teachers/catseg_dense \
  --config_file configs/vitb_384.yaml \
  --weights checkpoints/model_base.pth \
  --device cuda \
  --model_name catseg_vitb_384 \
  --logit_height 180 \
  --logit_width 320 \
  --dtype float16 \
  --skip_existing
```

The output should look like:

```text
RA-OV3DSeg/outputs/external_teachers/catseg_dense/
  sample_0000_dense_teacher_logits.npz
  sample_0001_dense_teacher_logits.npz
  ...
```

The default `180x320` saved map is intentional. Full-resolution nuScenes logits
would be too large for 128 samples.

## 5. Validate And Train V12

Return to RA-OV3DSeg:

```bash
cd /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg
conda activate ra-ov3dseg

bash scripts/run_v12_external_teacher_training.sh \
  --external_dense_teacher_dir outputs/external_teachers/catseg_dense \
  --local_files_only
```

Verify after the run:

```bash
python scripts/verify_mvp_outputs.py \
  --stage v12 \
  --sample_idx 128 \
  --outputs_dir outputs \
  --experiment_dir outputs/experiments/trainval_v12_external_teacher_128 \
  --output_dir outputs/verification
```

## Smoke Export

To debug CAT-Seg export before running all 128 records:

```bash
python /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg/tools/export_catseg_dense_logits.py \
  --catseg_root . \
  --manifest_jsonl /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg/outputs/experiments/trainval_v12_external_teacher_128/external_teacher_manifest/train_0_128.jsonl \
  --output_dir /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg/outputs/external_teachers/catseg_dense_smoke \
  --config_file configs/vitb_384.yaml \
  --weights checkpoints/model_base.pth \
  --device cuda \
  --max_records 2 \
  --skip_existing
```

Then validate only those two records:

```bash
cd /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg

python scripts/check_external_dense_teacher_logits.py \
  --start_idx 0 \
  --max_samples 2 \
  --dense_teacher_dir outputs/external_teachers/catseg_dense_smoke \
  --projection_dir outputs/experiments/trainval_v9_128_isolated/precompute/projections \
  --output_dir outputs/external_teacher_checks/catseg_dense_smoke
```
