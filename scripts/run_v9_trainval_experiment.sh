#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATAROOT="${PROJECT_ROOT}/data/nuscenes"
CACHE_DIR="/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/huggingface_cache"
OUTPUTS_DIR="${PROJECT_ROOT}/outputs"
PROFILE="smoke"
EXPERIMENT_NAME=""
TRAIN_START_IDX=0
TRAIN_MAX_SAMPLES=""
EVAL_START_IDX=""
EVAL_MAX_SAMPLES=""
EPOCHS=""
BATCH_SIZE=1
NUM_WORKERS=4
MAX_POINTS=50000
SPARSE_BASE_CHANNELS=32
NPROC_PER_NODE=1
PRECOMPUTE_DEVICE="cuda"
TRAIN_DEVICE="cuda"
TEACHER_MODE="hybrid"
STUDENT_OUTPUT_SPACE="all_lidarseg"
LOCAL_FILES_ONLY=0
SKIP_EXISTING=1
DRY_RUN=0
AMP=1

usage() {
  cat <<'EOF'
Run V9 trainval subset experiment with persistent logging.

Default profiles:
  smoke    train 8 samples, eval 8 samples, 3 epochs
  small    train 128 samples, eval 128 samples, 10 epochs

Examples:
  bash scripts/run_v9_trainval_experiment.sh --profile smoke

  bash scripts/run_v9_trainval_experiment.sh \
    --profile small \
    --dataroot /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg/data/nuscenes \
    --cache_dir /dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/huggingface_cache

Options:
  --profile smoke|small|custom
  --experiment_name NAME
  --dataroot PATH
  --cache_dir PATH
  --outputs_dir PATH
  --train_start_idx N
  --train_max_samples N
  --eval_start_idx N
  --eval_max_samples N
  --epochs N
  --batch_size N
  --num_workers N
  --max_points N
  --sparse_base_channels N
  --nproc_per_node N
  --precompute_device auto|cpu|cuda
  --train_device auto|cpu|cuda
  --teacher_mode feature_distill|dense_logit_distill|hybrid
  --student_output_space auto|base|all_lidarseg
  --local_files_only
  --no_skip_existing
  --no_amp
  --dry_run
  -h, --help

Logs:
  outputs/logs/v9_<experiment_name>_<timestamp>.log
  outputs/logs/v9_<experiment_name>_latest.log
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --experiment_name)
      EXPERIMENT_NAME="$2"
      shift 2
      ;;
    --dataroot)
      DATAROOT="$2"
      shift 2
      ;;
    --cache_dir)
      CACHE_DIR="$2"
      shift 2
      ;;
    --outputs_dir)
      OUTPUTS_DIR="$2"
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
    --nproc_per_node)
      NPROC_PER_NODE="$2"
      shift 2
      ;;
    --precompute_device)
      PRECOMPUTE_DEVICE="$2"
      shift 2
      ;;
    --train_device)
      TRAIN_DEVICE="$2"
      shift 2
      ;;
    --teacher_mode)
      TEACHER_MODE="$2"
      shift 2
      ;;
    --student_output_space)
      STUDENT_OUTPUT_SPACE="$2"
      shift 2
      ;;
    --local_files_only)
      LOCAL_FILES_ONLY=1
      shift
      ;;
    --no_skip_existing)
      SKIP_EXISTING=0
      shift
      ;;
    --no_amp)
      AMP=0
      shift
      ;;
    --dry_run)
      DRY_RUN=1
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

case "${PROFILE}" in
  smoke)
    TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-8}"
    EVAL_START_IDX="${EVAL_START_IDX:-8}"
    EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-8}"
    EPOCHS="${EPOCHS:-3}"
    ;;
  small)
    TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-128}"
    EVAL_START_IDX="${EVAL_START_IDX:-128}"
    EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-128}"
    EPOCHS="${EPOCHS:-10}"
    ;;
  custom)
    if [[ -z "${TRAIN_MAX_SAMPLES}" || -z "${EVAL_START_IDX}" || -z "${EVAL_MAX_SAMPLES}" || -z "${EPOCHS}" ]]; then
      echo "[ERROR] custom profile requires --train_max_samples, --eval_start_idx, --eval_max_samples, and --epochs." >&2
      exit 2
    fi
    ;;
  *)
    echo "[ERROR] --profile must be smoke, small, or custom." >&2
    exit 2
    ;;
esac

if [[ -z "${EXPERIMENT_NAME}" ]]; then
  if [[ "${PROFILE}" == "smoke" ]]; then
    EXPERIMENT_NAME="trainval_v9_8"
  elif [[ "${PROFILE}" == "small" ]]; then
    EXPERIMENT_NAME="trainval_v9_128"
  else
    EXPERIMENT_NAME="trainval_v9_custom_${TRAIN_MAX_SAMPLES}"
  fi
fi

LOG_DIR="${OUTPUTS_DIR}/logs"
EXPERIMENT_DIR="${OUTPUTS_DIR}/experiments/${EXPERIMENT_NAME}"
mkdir -p "${LOG_DIR}" "${EXPERIMENT_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/v9_${EXPERIMENT_NAME}_${TIMESTAMP}.log"
LATEST_LOG="${LOG_DIR}/v9_${EXPERIMENT_NAME}_latest.log"

COMMAND=(
  python "${PROJECT_ROOT}/scripts/run_mini_experiment.py"
  --dataroot "${DATAROOT}"
  --version v1.0-trainval
  --outputs_dir "${OUTPUTS_DIR}"
  --experiment_dir "${EXPERIMENT_DIR}"
  --train_start_idx "${TRAIN_START_IDX}"
  --train_max_samples "${TRAIN_MAX_SAMPLES}"
  --eval_start_idx "${EVAL_START_IDX}"
  --eval_max_samples "${EVAL_MAX_SAMPLES}"
  --cache_dir "${CACHE_DIR}"
  --precompute_device "${PRECOMPUTE_DEVICE}"
  --train_device "${TRAIN_DEVICE}"
  --teacher_mode "${TEACHER_MODE}"
  --student_output_space "${STUDENT_OUTPUT_SPACE}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --max_points "${MAX_POINTS}"
  --sparse_base_channels "${SPARSE_BASE_CHANNELS}"
  --nproc_per_node "${NPROC_PER_NODE}"
)

if [[ "${AMP}" == "1" ]]; then
  COMMAND+=(--amp)
fi
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  COMMAND+=(--skip_existing)
fi
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  COMMAND+=(--local_files_only)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  COMMAND+=(--dry_run)
fi

{
  echo "[INFO] V9 launcher started at $(date -Is)"
  echo "[INFO] project_root=${PROJECT_ROOT}"
  echo "[INFO] dataroot=${DATAROOT}"
  echo "[INFO] cache_dir=${CACHE_DIR}"
  echo "[INFO] outputs_dir=${OUTPUTS_DIR}"
  echo "[INFO] experiment_dir=${EXPERIMENT_DIR}"
  echo "[INFO] log_file=${LOG_FILE}"
  echo "[INFO] profile=${PROFILE}"
  echo "[INFO] train_range=${TRAIN_START_IDX}:${TRAIN_MAX_SAMPLES}"
  echo "[INFO] eval_range=${EVAL_START_IDX}:${EVAL_MAX_SAMPLES}"
  echo "[INFO] command:"
  printf '  %q' "${COMMAND[@]}"
  echo
  echo "[INFO] nvidia-smi snapshot:"
  nvidia-smi || true
  echo "[INFO] disk snapshot:"
  df -h "${PROJECT_ROOT}" "${DATAROOT}" || true
  echo "[INFO] start running experiment"
} 2>&1 | tee "${LOG_FILE}"

set +e
"${COMMAND[@]}" 2>&1 | tee -a "${LOG_FILE}"
STATUS=${PIPESTATUS[0]}
set -e

{
  echo "[INFO] V9 launcher finished at $(date -Is)"
  echo "[INFO] exit_status=${STATUS}"
  echo "[INFO] log_file=${LOG_FILE}"
  echo "[INFO] latest_log=${LATEST_LOG}"
} 2>&1 | tee -a "${LOG_FILE}"

ln -sfn "$(basename "${LOG_FILE}")" "${LATEST_LOG}"

exit "${STATUS}"
