#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATAROOT="${PROJECT_ROOT}/data/nuscenes"
OUTPUTS_DIR="${PROJECT_ROOT}/outputs"
EXPERIMENT_NAME="trainval_v17_pointcept_spunet_128"
TRAIN_START_IDX=0
TRAIN_MAX_SAMPLES=128
EVAL_START_IDX=128
EVAL_MAX_SAMPLES=128
EPOCHS=10
BATCH_SIZE=1
NUM_WORKERS=4
MAX_POINTS=50000
SPARSE_BASE_CHANNELS=32
FEATURE_DIM=128
LR=0.001
WEIGHT_DECAY=0.005
LOVASZ_WEIGHT=1.0
DICE_WEIGHT=0.0
EVAL_EVERY=2
DEVICE="cuda"
NPROC_PER_NODE=1
AMP=1
AUGMENT=1
SKIP_EXISTING=0
SKIP_CLASS_FREQ=0
SKIP_TRAIN=0
SKIP_PREDICT=0
SKIP_EVAL=0

VOXEL_SIZE=0.05
RANGE_XY=120.0
RANGE_Z_MIN=-10.0
RANGE_Z_MAX=10.0

usage() {
  cat <<'EOF'
Run V17 vendored Pointcept SpUNet official-16 nuScenes lidarseg baseline.

Task type:
  - Class-frequency: CPU, a few minutes.
  - Training/prediction: GPU if --device cuda, usually H100.
  - Evaluation: CPU-light plus file IO.

Expected runtime:
  - Smoke: --train_max_samples 8 --eval_max_samples 8 --epochs 2 --sparse_base_channels 16 -> 10-25 min on one H100.
  - Default: 128 train / 128 eval / 10 epochs -> about 30-90 min on one H100.
  - 1024 train / 512 eval / 30 epochs -> several hours on one H100.

Main purpose:
  Test a mature vendored Pointcept SpConv SparseUNet backend inside RA-OV3DSeg's
  own train/predict/eval pipeline, without switching repo or conda env.

Examples:
  bash scripts/run_v17_pointcept_spunet.sh \
    --experiment_name trainval_v17_pointcept_spunet_smoke \
    --train_max_samples 8 \
    --eval_start_idx 128 \
    --eval_max_samples 8 \
    --epochs 2 \
    --sparse_base_channels 16

  bash scripts/run_v17_pointcept_spunet.sh

Options:
  --dataroot PATH
  --outputs_dir PATH
  --experiment_name NAME
  --train_start_idx N
  --train_max_samples N
  --eval_start_idx N
  --eval_max_samples N
  --epochs N
  --batch_size N
  --num_workers N
  --max_points N
  --sparse_base_channels N
  --feature_dim N
  --lr LR
  --weight_decay WD
  --lovasz_weight W
  --dice_weight W
  --eval_every N
  --device auto|cpu|cuda
  --nproc_per_node N
  --voxel_size M
  --range_xy M
  --range_z_min Z
  --range_z_max Z
  --no_amp
  --no_augment
  --skip_existing
  --skip_class_freq
  --skip_train
  --skip_predict
  --skip_eval
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataroot) DATAROOT="$2"; shift 2 ;;
    --outputs_dir) OUTPUTS_DIR="$2"; shift 2 ;;
    --experiment_name) EXPERIMENT_NAME="$2"; shift 2 ;;
    --train_start_idx) TRAIN_START_IDX="$2"; shift 2 ;;
    --train_max_samples) TRAIN_MAX_SAMPLES="$2"; shift 2 ;;
    --eval_start_idx) EVAL_START_IDX="$2"; shift 2 ;;
    --eval_max_samples) EVAL_MAX_SAMPLES="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --num_workers) NUM_WORKERS="$2"; shift 2 ;;
    --max_points) MAX_POINTS="$2"; shift 2 ;;
    --sparse_base_channels) SPARSE_BASE_CHANNELS="$2"; shift 2 ;;
    --feature_dim) FEATURE_DIM="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --weight_decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --lovasz_weight) LOVASZ_WEIGHT="$2"; shift 2 ;;
    --dice_weight) DICE_WEIGHT="$2"; shift 2 ;;
    --eval_every) EVAL_EVERY="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --nproc_per_node) NPROC_PER_NODE="$2"; shift 2 ;;
    --voxel_size) VOXEL_SIZE="$2"; shift 2 ;;
    --range_xy) RANGE_XY="$2"; shift 2 ;;
    --range_z_min) RANGE_Z_MIN="$2"; shift 2 ;;
    --range_z_max) RANGE_Z_MAX="$2"; shift 2 ;;
    --no_amp) AMP=0; shift ;;
    --no_augment) AUGMENT=0; shift ;;
    --skip_existing) SKIP_EXISTING=1; shift ;;
    --skip_class_freq) SKIP_CLASS_FREQ=1; shift ;;
    --skip_train) SKIP_TRAIN=1; shift ;;
    --skip_predict) SKIP_PREDICT=1; shift ;;
    --skip_eval) SKIP_EVAL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[ERROR] unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

EXPERIMENT_DIR="${OUTPUTS_DIR}/experiments/${EXPERIMENT_NAME}"
TRAINING_DIR="${EXPERIMENT_DIR}/training"
PREDICTION_DIR="${EXPERIMENT_DIR}/predictions3d"
EVALUATION_DIR="${EXPERIMENT_DIR}/evaluation3d"
LOG_DIR="${OUTPUTS_DIR}/logs"
CLASS_NAMES_PATH="${PROJECT_ROOT}/configs/nuscenes_lidarseg_class_names.txt"
SPLIT_CONFIG="${PROJECT_ROOT}/configs/all_lidarseg_supervised_split.yaml"
CLASS_FREQ_JSON="${EXPERIMENT_DIR}/class_frequencies.json"

mkdir -p "${TRAINING_DIR}" "${PREDICTION_DIR}" "${EVALUATION_DIR}" "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/v17_${EXPERIMENT_NAME}_${TIMESTAMP}.log"
LATEST_LOG="${LOG_DIR}/v17_${EXPERIMENT_NAME}_latest.log"
ln -sfn "$(basename "${LOG_FILE}")" "${LATEST_LOG}"

run_step() {
  local step_name="$1"
  shift
  echo "[INFO] ========== STEP START: ${step_name} ==========" | tee -a "${LOG_FILE}"
  printf '[INFO] command:' | tee -a "${LOG_FILE}"
  printf ' %q' "$@" | tee -a "${LOG_FILE}"
  echo | tee -a "${LOG_FILE}"
  set +e
  "$@" 2>&1 | tee -a "${LOG_FILE}"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ "${status}" != "0" ]]; then
    echo "[ERROR] ========== STEP FAIL: ${step_name} | status=${status} ==========" | tee -a "${LOG_FILE}"
    exit "${status}"
  fi
  echo "[INFO] ========== STEP DONE: ${step_name} ==========" | tee -a "${LOG_FILE}"
}

CLASS_FREQ_COMMAND=(
  python "${PROJECT_ROOT}/scripts/compute_class_frequencies.py"
  --dataroot "${DATAROOT}"
  --version v1.0-trainval
  --start_idx "${TRAIN_START_IDX}"
  --max_samples "${TRAIN_MAX_SAMPLES}"
  --class_names_path "${CLASS_NAMES_PATH}"
  --split_config "${SPLIT_CONFIG}"
  --output_json "${CLASS_FREQ_JSON}"
)

TRAIN_CORE=(
  "${PROJECT_ROOT}/scripts/train_3d_segmentor.py"
  --dataroot "${DATAROOT}"
  --version v1.0-trainval
  --start_idx "${TRAIN_START_IDX}"
  --max_samples "${TRAIN_MAX_SAMPLES}"
  --data_source raw_lidarseg
  --teacher_mode feature_distill
  --student_output_space official_lidarseg_16
  --class_names_path "${CLASS_NAMES_PATH}"
  --split_config "${SPLIT_CONFIG}"
  --backbone pointcept_spunet
  --device "${DEVICE}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --max_points "${MAX_POINTS}"
  --voxel_size "${VOXEL_SIZE}" "${VOXEL_SIZE}" "${VOXEL_SIZE}"
  --point_cloud_range "-${RANGE_XY}" "-${RANGE_XY}" "${RANGE_Z_MIN}" "${RANGE_XY}" "${RANGE_XY}" "${RANGE_Z_MAX}"
  --sparse_base_channels "${SPARSE_BASE_CHANNELS}"
  --feature_dim "${FEATURE_DIM}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --ce_weight 1.0
  --class_weights_path "${CLASS_FREQ_JSON}"
  --lovasz_weight "${LOVASZ_WEIGHT}"
  --dice_weight "${DICE_WEIGHT}"
  --distill_weight 0.0
  --dense_logit_weight 0.0
  --text_align_weight 0.0
  --eval_start_idx "${EVAL_START_IDX}"
  --eval_max_samples "${EVAL_MAX_SAMPLES}"
  --eval_every "${EVAL_EVERY}"
  --output_dir "${TRAINING_DIR}"
)
if [[ "${AMP}" == "1" ]]; then
  TRAIN_CORE+=(--amp)
fi
if [[ "${AUGMENT}" == "1" ]]; then
  TRAIN_CORE+=(--augment)
fi
if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  TRAIN_COMMAND=(torchrun --standalone "--nproc_per_node=${NPROC_PER_NODE}" "${TRAIN_CORE[@]}")
else
  TRAIN_COMMAND=(python "${TRAIN_CORE[@]}")
fi

CHECKPOINT="${TRAINING_DIR}/pointcept_spunet_best.pt"
PREDICT_COMMAND=(
  python "${PROJECT_ROOT}/scripts/predict_3d_segmentor.py"
  --checkpoint "${CHECKPOINT}"
  --input_source raw_lidarseg
  --dataroot "${DATAROOT}"
  --version v1.0-trainval
  --start_idx "${EVAL_START_IDX}"
  --max_samples "${EVAL_MAX_SAMPLES}"
  --class_names_path "${CLASS_NAMES_PATH}"
  --device "${DEVICE}"
  --output_dir "${PREDICTION_DIR}"
)
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  PREDICT_COMMAND+=(--skip_existing)
fi

EVAL_COMMAND=(
  python "${PROJECT_ROOT}/scripts/eval_lidarseg.py"
  --dataroot "${DATAROOT}"
  --version v1.0-trainval
  --start_idx "${EVAL_START_IDX}"
  --max_samples "${EVAL_MAX_SAMPLES}"
  --prediction_dir "${PREDICTION_DIR}"
  --prediction_file_template "sample_{sample_idx:04d}_3d_predictions.npz"
  --class_names_path "${CLASS_NAMES_PATH}"
  --split_config "${SPLIT_CONFIG}"
  --label_space official16
  --output_dir "${EVALUATION_DIR}"
)
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  EVAL_COMMAND+=(--skip_existing)
fi

SUMMARY_COMMAND=(
  python "${PROJECT_ROOT}/scripts/print_experiment_summary.py"
  --stage v17_pointcept_spunet
  --experiment_dir "${EXPERIMENT_DIR}"
  --output_json "${EXPERIMENT_DIR}/compact_summary.json"
)

{
  echo "[INFO] V17 Pointcept SpUNet baseline started at $(date -Is)"
  echo "[INFO] task_type=CPU class-frequency + GPU train/predict + CPU eval"
  echo "[INFO] expected_runtime=default 128 train/eval, 10 epochs: about 30-90 min on one H100"
  echo "[INFO] gpu_required=yes for training when --device cuda"
  echo "[INFO] project_root=${PROJECT_ROOT}"
  echo "[INFO] experiment_dir=${EXPERIMENT_DIR}"
  echo "[INFO] train_range=start:${TRAIN_START_IDX} max:${TRAIN_MAX_SAMPLES}"
  echo "[INFO] eval_range=start:${EVAL_START_IDX} max:${EVAL_MAX_SAMPLES}"
  echo "[INFO] backbone=pointcept_spunet"
  echo "[INFO] label_space=official_lidarseg_16"
  echo "[INFO] voxel_size=${VOXEL_SIZE}"
  echo "[INFO] point_cloud_range=[-${RANGE_XY},-${RANGE_XY},${RANGE_Z_MIN},${RANGE_XY},${RANGE_XY},${RANGE_Z_MAX}]"
  echo "[INFO] log_file=${LOG_FILE}"
  echo "[INFO] disk snapshot:"
  df -h "${PROJECT_ROOT}" "${OUTPUTS_DIR}" "${DATAROOT}" || true
  echo "[INFO] nvidia-smi snapshot:"
  nvidia-smi || true
} 2>&1 | tee "${LOG_FILE}"

if [[ "${SKIP_CLASS_FREQ}" != "1" ]]; then
  run_step "compute_class_frequencies" "${CLASS_FREQ_COMMAND[@]}"
fi
if [[ ! -f "${CLASS_FREQ_JSON}" ]]; then
  echo "[ERROR] class frequency json not found: ${CLASS_FREQ_JSON}" | tee -a "${LOG_FILE}" >&2
  exit 1
fi
if [[ "${SKIP_TRAIN}" != "1" ]]; then
  run_step "train_v17_pointcept_spunet" "${TRAIN_COMMAND[@]}"
fi
if [[ "${SKIP_PREDICT}" != "1" ]]; then
  if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "[ERROR] best checkpoint not found: ${CHECKPOINT}" | tee -a "${LOG_FILE}" >&2
    exit 1
  fi
  run_step "predict_v17_best_checkpoint" "${PREDICT_COMMAND[@]}"
fi
if [[ "${SKIP_EVAL}" != "1" ]]; then
  run_step "eval_v17_best_checkpoint_official16" "${EVAL_COMMAND[@]}"
fi

{
  echo "[INFO] V17 Pointcept SpUNet baseline finished at $(date -Is)"
  echo "[INFO] exit_status=0"
  echo "[INFO] log_file=${LOG_FILE}"
  echo "[INFO] latest_log=${LATEST_LOG}"
  echo "[INFO] final compact conclusion follows; paste RUN_CONCLUSION if needed:"
} 2>&1 | tee -a "${LOG_FILE}"

set +e
"${SUMMARY_COMMAND[@]}" 2>&1 | tee -a "${LOG_FILE}"
SUMMARY_STATUS=${PIPESTATUS[0]}
set -e
if [[ "${SUMMARY_STATUS}" != "0" ]]; then
  echo "[WARN] compact conclusion failed | status=${SUMMARY_STATUS}" | tee -a "${LOG_FILE}" >&2
fi

exit 0
