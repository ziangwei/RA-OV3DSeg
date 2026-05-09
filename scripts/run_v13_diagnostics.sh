#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATAROOT="${PROJECT_ROOT}/data/nuscenes"
OUTPUTS_DIR="${PROJECT_ROOT}/outputs"
SOURCE_EXPERIMENT_NAME="trainval_v9_128_isolated"
TEACHER_EXPERIMENT_NAME="trainval_v12_groupvit_128"
EXPERIMENT_NAME="trainval_v13_diagnostics_128"
TRAIN_START_IDX=0
TRAIN_MAX_SAMPLES=128
EVAL_START_IDX=128
EVAL_MAX_SAMPLES=128
EPOCHS=20
BATCH_SIZE=1
NUM_WORKERS=4
MAX_POINTS=50000
SPARSE_BASE_CHANNELS=32
LR=0.0003
WEIGHT_DECAY=0.0001
DEVICE="cuda"
NPROC_PER_NODE=1
AMP=1
SKIP_EXISTING=1
SKIP_TEACHER_EVAL=0
SKIP_UPPER_BOUND=0
SKIP_PREDICT=0
SKIP_EVAL=0

usage() {
  cat <<'EOF'
Run V13 diagnostics:
  1. Evaluate projected dense teacher pseudo labels directly against lidarseg.
  2. Train a stronger in-repository spconv_resunet with full lidarseg supervision.

This separates the two failure modes:
  - weak 2D teacher
  - weak 3D student capacity

Examples:
  bash scripts/run_v13_diagnostics.sh

  bash scripts/run_v13_diagnostics.sh \
    --experiment_name trainval_v13_diagnostics_smoke \
    --train_max_samples 8 \
    --eval_max_samples 8 \
    --epochs 2 \
    --sparse_base_channels 16

Options:
  --dataroot PATH
  --outputs_dir PATH
  --source_experiment_name NAME
  --teacher_experiment_name NAME
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
  --lr LR
  --weight_decay WD
  --device auto|cpu|cuda
  --nproc_per_node N
  --no_amp
  --no_skip_existing
  --skip_teacher_eval
  --skip_upper_bound
  --skip_predict
  --skip_eval
  -h, --help

Logs:
  outputs/logs/v13_<experiment_name>_<timestamp>.log
  outputs/logs/v13_<experiment_name>_latest.log
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataroot)
      DATAROOT="$2"
      shift 2
      ;;
    --outputs_dir)
      OUTPUTS_DIR="$2"
      shift 2
      ;;
    --source_experiment_name)
      SOURCE_EXPERIMENT_NAME="$2"
      shift 2
      ;;
    --teacher_experiment_name)
      TEACHER_EXPERIMENT_NAME="$2"
      shift 2
      ;;
    --experiment_name)
      EXPERIMENT_NAME="$2"
      shift 2
      ;;
    --train_start_idx)
      TRAIN_START_IDX="$2"
      shift 2
      ;;
    --train_max_samples)
      TRAIN_MAX_SAMPLES="$2"
      shift 2
      ;;
    --eval_start_idx)
      EVAL_START_IDX="$2"
      shift 2
      ;;
    --eval_max_samples)
      EVAL_MAX_SAMPLES="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch_size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --num_workers)
      NUM_WORKERS="$2"
      shift 2
      ;;
    --max_points)
      MAX_POINTS="$2"
      shift 2
      ;;
    --sparse_base_channels)
      SPARSE_BASE_CHANNELS="$2"
      shift 2
      ;;
    --lr)
      LR="$2"
      shift 2
      ;;
    --weight_decay)
      WEIGHT_DECAY="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --nproc_per_node)
      NPROC_PER_NODE="$2"
      shift 2
      ;;
    --no_amp)
      AMP=0
      shift
      ;;
    --no_skip_existing)
      SKIP_EXISTING=0
      shift
      ;;
    --skip_teacher_eval)
      SKIP_TEACHER_EVAL=1
      shift
      ;;
    --skip_upper_bound)
      SKIP_UPPER_BOUND=1
      shift
      ;;
    --skip_predict)
      SKIP_PREDICT=1
      shift
      ;;
    --skip_eval)
      SKIP_EVAL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

SOURCE_EXPERIMENT_DIR="${OUTPUTS_DIR}/experiments/${SOURCE_EXPERIMENT_NAME}"
TEACHER_EXPERIMENT_DIR="${OUTPUTS_DIR}/experiments/${TEACHER_EXPERIMENT_NAME}"
EXPERIMENT_DIR="${OUTPUTS_DIR}/experiments/${EXPERIMENT_NAME}"
POINT_FEATURE_DIR="${SOURCE_EXPERIMENT_DIR}/precompute/point_features"
RELIABILITY_DIR="${SOURCE_EXPERIMENT_DIR}/precompute/reliability"
DENSE_POINT_DIR="${TEACHER_EXPERIMENT_DIR}/precompute/dense_point_logits"
TEACHER_QUALITY_DIR="${EXPERIMENT_DIR}/teacher_quality"
TRAINING_DIR="${EXPERIMENT_DIR}/supervised_training"
PREDICTION_DIR="${EXPERIMENT_DIR}/supervised_predictions3d"
EVALUATION_DIR="${EXPERIMENT_DIR}/supervised_evaluation3d"
LOG_DIR="${OUTPUTS_DIR}/logs"
SPLIT_CONFIG="${PROJECT_ROOT}/configs/all_lidarseg_supervised_split.yaml"
CLASS_NAMES_PATH="${PROJECT_ROOT}/configs/nuscenes_lidarseg_class_names.txt"

mkdir -p "${TEACHER_QUALITY_DIR}" "${TRAINING_DIR}" "${PREDICTION_DIR}" "${EVALUATION_DIR}" "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/v13_${EXPERIMENT_NAME}_${TIMESTAMP}.log"
LATEST_LOG="${LOG_DIR}/v13_${EXPERIMENT_NAME}_latest.log"

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

TEACHER_EVAL_COMMAND=(
  python "${PROJECT_ROOT}/scripts/eval_dense_teacher_pseudo_labels.py"
  --dataroot "${DATAROOT}"
  --version v1.0-trainval
  --start_idx "${TRAIN_START_IDX}"
  --max_samples "${TRAIN_MAX_SAMPLES}"
  --dense_point_dir "${DENSE_POINT_DIR}"
  --class_names_path "${CLASS_NAMES_PATH}"
  --split_config "${PROJECT_ROOT}/configs/base_novel_split.yaml"
  --output_dir "${TEACHER_QUALITY_DIR}"
)
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  TEACHER_EVAL_COMMAND+=(--skip_existing)
fi

TRAIN_CORE=(
  "${PROJECT_ROOT}/scripts/train_3d_segmentor.py"
  --dataroot "${DATAROOT}"
  --version v1.0-trainval
  --start_idx "${TRAIN_START_IDX}"
  --max_samples "${TRAIN_MAX_SAMPLES}"
  --point_feature_dir "${POINT_FEATURE_DIR}"
  --reliability_dir "${RELIABILITY_DIR}"
  --teacher_mode feature_distill
  --student_output_space all_lidarseg
  --class_names_path "${CLASS_NAMES_PATH}"
  --split_config "${SPLIT_CONFIG}"
  --backbone spconv_resunet
  --device "${DEVICE}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --max_points "${MAX_POINTS}"
  --sparse_base_channels "${SPARSE_BASE_CHANNELS}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --ce_weight 1.0
  --distill_weight 0.0
  --dense_logit_weight 0.0
  --text_align_weight 0.0
  --output_dir "${TRAINING_DIR}"
)
if [[ "${AMP}" == "1" ]]; then
  TRAIN_CORE+=(--amp)
fi
if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  TRAIN_COMMAND=(torchrun --standalone "--nproc_per_node=${NPROC_PER_NODE}" "${TRAIN_CORE[@]}")
else
  TRAIN_COMMAND=(python "${TRAIN_CORE[@]}")
fi

PREDICT_COMMAND=(
  python "${PROJECT_ROOT}/scripts/predict_3d_segmentor.py"
  --checkpoint "${TRAINING_DIR}/spconv_resunet_latest.pt"
  --start_idx "${EVAL_START_IDX}"
  --max_samples "${EVAL_MAX_SAMPLES}"
  --point_feature_dir "${POINT_FEATURE_DIR}"
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
  --output_dir "${EVALUATION_DIR}"
)
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  EVAL_COMMAND+=(--skip_existing)
fi

{
  echo "[INFO] V13 diagnostics started at $(date -Is)"
  echo "[INFO] project_root=${PROJECT_ROOT}"
  echo "[INFO] source_experiment_dir=${SOURCE_EXPERIMENT_DIR}"
  echo "[INFO] teacher_experiment_dir=${TEACHER_EXPERIMENT_DIR}"
  echo "[INFO] experiment_dir=${EXPERIMENT_DIR}"
  echo "[INFO] train_range=start:${TRAIN_START_IDX} max:${TRAIN_MAX_SAMPLES}"
  echo "[INFO] eval_range=start:${EVAL_START_IDX} max:${EVAL_MAX_SAMPLES}"
  echo "[INFO] spconv_resunet base_channels=${SPARSE_BASE_CHANNELS}"
  echo "[INFO] log_file=${LOG_FILE}"
  echo "[INFO] disk snapshot:"
  df -h "${PROJECT_ROOT}" "${OUTPUTS_DIR}" "${DATAROOT}" || true
  echo "[INFO] nvidia-smi snapshot:"
  nvidia-smi || true
} 2>&1 | tee "${LOG_FILE}"

for required_dir in "${POINT_FEATURE_DIR}" "${RELIABILITY_DIR}"; do
  if [[ ! -d "${required_dir}" ]]; then
    echo "[ERROR] required directory not found: ${required_dir}" | tee -a "${LOG_FILE}" >&2
    exit 1
  fi
done
if [[ "${SKIP_TEACHER_EVAL}" != "1" && ! -d "${DENSE_POINT_DIR}" ]]; then
  echo "[ERROR] dense teacher point directory not found: ${DENSE_POINT_DIR}" | tee -a "${LOG_FILE}" >&2
  echo "[ERROR] run V12 first or pass --skip_teacher_eval." | tee -a "${LOG_FILE}" >&2
  exit 1
fi

if [[ "${SKIP_TEACHER_EVAL}" != "1" ]]; then
  run_step "eval_projected_dense_teacher_quality" "${TEACHER_EVAL_COMMAND[@]}"
fi
if [[ "${SKIP_UPPER_BOUND}" != "1" ]]; then
  run_step "train_spconv_resunet_supervised_upper_bound" "${TRAIN_COMMAND[@]}"
fi
if [[ "${SKIP_PREDICT}" != "1" ]]; then
  run_step "predict_spconv_resunet_supervised" "${PREDICT_COMMAND[@]}"
fi
if [[ "${SKIP_EVAL}" != "1" ]]; then
  run_step "eval_spconv_resunet_supervised" "${EVAL_COMMAND[@]}"
fi

{
  echo "[INFO] V13 diagnostics finished at $(date -Is)"
  echo "[INFO] exit_status=0"
  echo "[INFO] log_file=${LOG_FILE}"
  echo "[INFO] latest_log=${LATEST_LOG}"
} 2>&1 | tee -a "${LOG_FILE}"

ln -sfn "$(basename "${LOG_FILE}")" "${LATEST_LOG}"
exit 0
