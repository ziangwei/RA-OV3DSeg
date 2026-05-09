#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATAROOT="${PROJECT_ROOT}/data/nuscenes"
CACHE_DIR="/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/huggingface_cache"
OUTPUTS_DIR="${PROJECT_ROOT}/outputs"
SOURCE_EXPERIMENT_NAME="trainval_v9_128_isolated"
EXPERIMENT_NAME="trainval_v11_text_align_128"
INIT_CHECKPOINT=""
USE_INIT_CHECKPOINT=1
POINT_FEATURE_DIR=""
RELIABILITY_DIR=""
DENSE_POINT_DIR=""
TRAIN_START_IDX=0
TRAIN_MAX_SAMPLES=128
EVAL_START_IDX=128
EVAL_MAX_SAMPLES=128
EPOCHS=5
BATCH_SIZE=1
NUM_WORKERS=4
MAX_POINTS=50000
SPARSE_BASE_CHANNELS=32
LR=0.0003
WEIGHT_DECAY=0.0001
TEXT_ALIGN_WEIGHT=1.0
CE_WEIGHT=1.0
DISTILL_WEIGHT=1.0
DENSE_LOGIT_WEIGHT=1.0
TEXT_MODEL_NAME="openai/clip-vit-base-patch16"
TEXT_PROMPT_TEMPLATE="a {} in a driving scene"
DEVICE="cuda"
NPROC_PER_NODE=1
LOCAL_FILES_ONLY=0
AMP=1
SKIP_TRAIN=0
SKIP_PREDICT=0
SKIP_EVAL=0
SKIP_EXISTING=1

usage() {
  cat <<'EOF'
Run V11 text-aligned 3D training and open-vocabulary evaluation.

Default behavior:
  - Reuses V9 isolated precompute artifacts.
  - Warm-starts from V9 sparse_unet_spconv checkpoint.
  - Adds supervised base-class text prototype alignment loss.
  - Runs open-vocabulary prediction/evaluation on the V9 eval split.

Examples:
  bash scripts/run_v11_text_aligned_training.sh

  bash scripts/run_v11_text_aligned_training.sh \
    --experiment_name trainval_v11_text_align_128 \
    --text_align_weight 1.0 \
    --epochs 5

Options:
  --dataroot PATH
  --cache_dir PATH
  --outputs_dir PATH
  --source_experiment_name NAME
  --experiment_name NAME
  --init_checkpoint PATH
  --no_init_checkpoint
  --point_feature_dir PATH
  --reliability_dir PATH
  --dense_point_dir PATH
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
  --text_align_weight W
  --ce_weight W
  --distill_weight W
  --dense_logit_weight W
  --text_model_name NAME
  --text_prompt_template TEMPLATE
  --device auto|cpu|cuda
  --nproc_per_node N
  --local_files_only
  --no_amp
  --no_skip_existing
  --skip_train
  --skip_predict
  --skip_eval
  -h, --help

Logs:
  outputs/logs/v11_<experiment_name>_<timestamp>.log
  outputs/logs/v11_<experiment_name>_latest.log
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --source_experiment_name)
      SOURCE_EXPERIMENT_NAME="$2"
      shift 2
      ;;
    --experiment_name)
      EXPERIMENT_NAME="$2"
      shift 2
      ;;
    --init_checkpoint)
      INIT_CHECKPOINT="$2"
      USE_INIT_CHECKPOINT=1
      shift 2
      ;;
    --no_init_checkpoint)
      USE_INIT_CHECKPOINT=0
      shift
      ;;
    --point_feature_dir)
      POINT_FEATURE_DIR="$2"
      shift 2
      ;;
    --reliability_dir)
      RELIABILITY_DIR="$2"
      shift 2
      ;;
    --dense_point_dir)
      DENSE_POINT_DIR="$2"
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
    --text_align_weight)
      TEXT_ALIGN_WEIGHT="$2"
      shift 2
      ;;
    --ce_weight)
      CE_WEIGHT="$2"
      shift 2
      ;;
    --distill_weight)
      DISTILL_WEIGHT="$2"
      shift 2
      ;;
    --dense_logit_weight)
      DENSE_LOGIT_WEIGHT="$2"
      shift 2
      ;;
    --text_model_name)
      TEXT_MODEL_NAME="$2"
      shift 2
      ;;
    --text_prompt_template)
      TEXT_PROMPT_TEMPLATE="$2"
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
    --local_files_only)
      LOCAL_FILES_ONLY=1
      shift
      ;;
    --no_amp)
      AMP=0
      shift
      ;;
    --no_skip_existing)
      SKIP_EXISTING=0
      shift
      ;;
    --skip_train)
      SKIP_TRAIN=1
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
EXPERIMENT_DIR="${OUTPUTS_DIR}/experiments/${EXPERIMENT_NAME}"
TRAINING_DIR="${EXPERIMENT_DIR}/training"
PREDICTION_DIR="${EXPERIMENT_DIR}/open_vocab_predictions3d"
EVALUATION_DIR="${EXPERIMENT_DIR}/open_vocab_evaluation3d"
LOG_DIR="${OUTPUTS_DIR}/logs"

if [[ -z "${INIT_CHECKPOINT}" ]]; then
  INIT_CHECKPOINT="${SOURCE_EXPERIMENT_DIR}/training/sparse_unet_spconv_latest.pt"
fi
if [[ -z "${POINT_FEATURE_DIR}" ]]; then
  POINT_FEATURE_DIR="${SOURCE_EXPERIMENT_DIR}/precompute/point_features"
fi
if [[ -z "${RELIABILITY_DIR}" ]]; then
  RELIABILITY_DIR="${SOURCE_EXPERIMENT_DIR}/precompute/reliability"
fi
if [[ -z "${DENSE_POINT_DIR}" ]]; then
  DENSE_POINT_DIR="${SOURCE_EXPERIMENT_DIR}/precompute/dense_point_logits"
fi

mkdir -p "${TRAINING_DIR}" "${PREDICTION_DIR}" "${EVALUATION_DIR}" "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/v11_${EXPERIMENT_NAME}_${TIMESTAMP}.log"
LATEST_LOG="${LOG_DIR}/v11_${EXPERIMENT_NAME}_latest.log"

if [[ "${SKIP_TRAIN}" != "1" ]]; then
  for required_dir in "${POINT_FEATURE_DIR}" "${RELIABILITY_DIR}" "${DENSE_POINT_DIR}"; do
    if [[ ! -d "${required_dir}" ]]; then
      {
        echo "[ERROR] required precompute directory not found: ${required_dir}"
        echo "[ERROR] run V9 first, or pass explicit --point_feature_dir/--reliability_dir/--dense_point_dir."
      } 2>&1 | tee -a "${LOG_FILE}" >&2
      exit 1
    fi
  done
  if [[ "${USE_INIT_CHECKPOINT}" == "1" && ! -f "${INIT_CHECKPOINT}" ]]; then
    {
      echo "[ERROR] init checkpoint not found: ${INIT_CHECKPOINT}"
      echo "[ERROR] run V9 first, pass --init_checkpoint, or use --no_init_checkpoint for a cold-start ablation."
    } 2>&1 | tee -a "${LOG_FILE}" >&2
    exit 1
  fi
fi

TRAIN_CORE=(
  "${PROJECT_ROOT}/scripts/train_3d_segmentor.py"
  --dataroot "${DATAROOT}"
  --version v1.0-trainval
  --start_idx "${TRAIN_START_IDX}"
  --max_samples "${TRAIN_MAX_SAMPLES}"
  --point_feature_dir "${POINT_FEATURE_DIR}"
  --reliability_dir "${RELIABILITY_DIR}"
  --dense_point_dir "${DENSE_POINT_DIR}"
  --teacher_mode hybrid
  --student_output_space all_lidarseg
  --backbone sparse_unet_spconv
  --device "${DEVICE}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --num_workers "${NUM_WORKERS}"
  --max_points "${MAX_POINTS}"
  --sparse_base_channels "${SPARSE_BASE_CHANNELS}"
  --lr "${LR}"
  --weight_decay "${WEIGHT_DECAY}"
  --ce_weight "${CE_WEIGHT}"
  --distill_weight "${DISTILL_WEIGHT}"
  --dense_logit_weight "${DENSE_LOGIT_WEIGHT}"
  --text_align_weight "${TEXT_ALIGN_WEIGHT}"
  --text_model_name "${TEXT_MODEL_NAME}"
  --text_prompt_template "${TEXT_PROMPT_TEMPLATE}"
  --cache_dir "${CACHE_DIR}"
  --output_dir "${TRAINING_DIR}"
)
if [[ "${USE_INIT_CHECKPOINT}" == "1" ]]; then
  TRAIN_CORE+=(--init_checkpoint "${INIT_CHECKPOINT}")
fi
if [[ "${AMP}" == "1" ]]; then
  TRAIN_CORE+=(--amp)
fi
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  TRAIN_CORE+=(--local_files_only)
fi

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  TRAIN_COMMAND=(torchrun --standalone "--nproc_per_node=${NPROC_PER_NODE}" "${TRAIN_CORE[@]}")
else
  TRAIN_COMMAND=(python "${TRAIN_CORE[@]}")
fi

PREDICT_COMMAND=(
  python "${PROJECT_ROOT}/scripts/predict_3d_open_vocab.py"
  --checkpoint "${TRAINING_DIR}/sparse_unet_spconv_latest.pt"
  --start_idx "${EVAL_START_IDX}"
  --max_samples "${EVAL_MAX_SAMPLES}"
  --point_feature_dir "${POINT_FEATURE_DIR}"
  --class_names_path "${PROJECT_ROOT}/configs/nuscenes_lidarseg_class_names.txt"
  --lidarseg_class_names_path "${PROJECT_ROOT}/configs/nuscenes_lidarseg_class_names.txt"
  --text_model_name "${TEXT_MODEL_NAME}"
  --cache_dir "${CACHE_DIR}"
  --prompt_template "${TEXT_PROMPT_TEMPLATE}"
  --device "${DEVICE}"
  --output_dir "${PREDICTION_DIR}"
)
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  PREDICT_COMMAND+=(--local_files_only)
fi
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
  --prediction_file_template "sample_{sample_idx:04d}_open_vocab_predictions.npz"
  --class_names_path "${PROJECT_ROOT}/configs/nuscenes_lidarseg_class_names.txt"
  --output_dir "${EVALUATION_DIR}"
)
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  EVAL_COMMAND+=(--skip_existing)
fi

{
  echo "[INFO] V11 launcher started at $(date -Is)"
  echo "[INFO] project_root=${PROJECT_ROOT}"
  echo "[INFO] source_experiment_dir=${SOURCE_EXPERIMENT_DIR}"
  echo "[INFO] experiment_dir=${EXPERIMENT_DIR}"
  echo "[INFO] init_checkpoint=${INIT_CHECKPOINT}"
  echo "[INFO] point_feature_dir=${POINT_FEATURE_DIR}"
  echo "[INFO] reliability_dir=${RELIABILITY_DIR}"
  echo "[INFO] dense_point_dir=${DENSE_POINT_DIR}"
  echo "[INFO] text_align_weight=${TEXT_ALIGN_WEIGHT}"
  echo "[INFO] train_range=start:${TRAIN_START_IDX} max:${TRAIN_MAX_SAMPLES}"
  echo "[INFO] eval_range=start:${EVAL_START_IDX} max:${EVAL_MAX_SAMPLES}"
  echo "[INFO] log_file=${LOG_FILE}"
  echo "[INFO] train command:"
  printf '  %q' "${TRAIN_COMMAND[@]}"
  echo
  echo "[INFO] predict command:"
  printf '  %q' "${PREDICT_COMMAND[@]}"
  echo
  echo "[INFO] eval command:"
  printf '  %q' "${EVAL_COMMAND[@]}"
  echo
  echo "[INFO] disk snapshot:"
  df -h "${PROJECT_ROOT}" "${OUTPUTS_DIR}" "${DATAROOT}" || true
  echo "[INFO] nvidia-smi snapshot:"
  nvidia-smi || true
} 2>&1 | tee "${LOG_FILE}"

STATUS=0
if [[ "${SKIP_TRAIN}" != "1" ]]; then
  set +e
  "${TRAIN_COMMAND[@]}" 2>&1 | tee -a "${LOG_FILE}"
  STATUS=${PIPESTATUS[0]}
  set -e
fi

if [[ "${STATUS}" == "0" && "${SKIP_PREDICT}" != "1" ]]; then
  set +e
  "${PREDICT_COMMAND[@]}" 2>&1 | tee -a "${LOG_FILE}"
  STATUS=${PIPESTATUS[0]}
  set -e
fi

if [[ "${STATUS}" == "0" && "${SKIP_EVAL}" != "1" ]]; then
  set +e
  "${EVAL_COMMAND[@]}" 2>&1 | tee -a "${LOG_FILE}"
  STATUS=${PIPESTATUS[0]}
  set -e
fi

{
  echo "[INFO] V11 launcher finished at $(date -Is)"
  echo "[INFO] exit_status=${STATUS}"
  echo "[INFO] log_file=${LOG_FILE}"
  echo "[INFO] latest_log=${LATEST_LOG}"
} 2>&1 | tee -a "${LOG_FILE}"

ln -sfn "$(basename "${LOG_FILE}")" "${LATEST_LOG}"
exit "${STATUS}"
