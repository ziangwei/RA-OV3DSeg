#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATAROOT="${PROJECT_ROOT}/data/nuscenes"
CACHE_DIR="/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/huggingface_cache"
OUTPUTS_DIR="${PROJECT_ROOT}/outputs"
SOURCE_EXPERIMENT_NAME="trainval_v9_128_isolated"
EXPERIMENT_NAME="trainval_v10_open_vocab_128"
CHECKPOINT=""
POINT_FEATURE_DIR=""
START_IDX=128
MAX_SAMPLES=128
CLASS_NAMES_PATH="${PROJECT_ROOT}/configs/nuscenes_lidarseg_class_names.txt"
CLASS_NAMES_CSV=""
TEXT_MODEL_NAME=""
PROMPT_TEMPLATE="a {} in a driving scene"
DEVICE="cuda"
LOCAL_FILES_ONLY=0
SKIP_EXISTING=1
SAVE_SIMILARITIES=0
SAVE_POINT_EMBEDDINGS=0
SKIP_PREDICT=0
SKIP_EVAL=0

usage() {
  cat <<'EOF'
Run V10 open-vocabulary 3D evaluation with persistent logging.

Default behavior:
  - Uses V9 isolated checkpoint.
  - Uses V9 isolated precomputed point features.
  - Queries the full nuScenes-lidarseg 32-class text list.
  - Evaluates against lidarseg labels.

Examples:
  bash scripts/run_v10_open_vocab_eval.sh

  bash scripts/run_v10_open_vocab_eval.sh \
    --source_experiment_name trainval_v9_128_isolated \
    --experiment_name trainval_v10_open_vocab_128

  bash scripts/run_v10_open_vocab_eval.sh \
    --class_names_csv "driveable surface,sidewalk,car,truck,pedestrian,vegetation" \
    --skip_eval

Options:
  --dataroot PATH
  --cache_dir PATH
  --outputs_dir PATH
  --source_experiment_name NAME
  --experiment_name NAME
  --checkpoint PATH
  --point_feature_dir PATH
  --start_idx N
  --max_samples N
  --class_names_path PATH
  --class_names_csv CSV
  --text_model_name NAME
  --prompt_template TEMPLATE
  --device auto|cpu|cuda
  --local_files_only
  --no_skip_existing
  --save_similarities
  --save_point_embeddings
  --skip_predict
  --skip_eval
  -h, --help

Logs:
  outputs/logs/v10_<experiment_name>_<timestamp>.log
  outputs/logs/v10_<experiment_name>_latest.log
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
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --point_feature_dir)
      POINT_FEATURE_DIR="$2"
      shift 2
      ;;
    --start_idx)
      START_IDX="$2"
      shift 2
      ;;
    --max_samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --class_names_path)
      CLASS_NAMES_PATH="$2"
      shift 2
      ;;
    --class_names_csv)
      CLASS_NAMES_CSV="$2"
      shift 2
      ;;
    --text_model_name)
      TEXT_MODEL_NAME="$2"
      shift 2
      ;;
    --prompt_template)
      PROMPT_TEMPLATE="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
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
    --save_similarities)
      SAVE_SIMILARITIES=1
      shift
      ;;
    --save_point_embeddings)
      SAVE_POINT_EMBEDDINGS=1
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
if [[ -z "${CHECKPOINT}" ]]; then
  CHECKPOINT="${SOURCE_EXPERIMENT_DIR}/training/sparse_unet_spconv_latest.pt"
fi
if [[ -z "${POINT_FEATURE_DIR}" ]]; then
  POINT_FEATURE_DIR="${SOURCE_EXPERIMENT_DIR}/precompute/point_features"
fi
PREDICTION_DIR="${EXPERIMENT_DIR}/open_vocab_predictions3d"
EVALUATION_DIR="${EXPERIMENT_DIR}/open_vocab_evaluation3d"
LOG_DIR="${OUTPUTS_DIR}/logs"
mkdir -p "${PREDICTION_DIR}" "${EVALUATION_DIR}" "${LOG_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/v10_${EXPERIMENT_NAME}_${TIMESTAMP}.log"
LATEST_LOG="${LOG_DIR}/v10_${EXPERIMENT_NAME}_latest.log"

PREDICT_COMMAND=(
  python "${PROJECT_ROOT}/scripts/predict_3d_open_vocab.py"
  --checkpoint "${CHECKPOINT}"
  --start_idx "${START_IDX}"
  --max_samples "${MAX_SAMPLES}"
  --point_feature_dir "${POINT_FEATURE_DIR}"
  --class_names_path "${CLASS_NAMES_PATH}"
  --lidarseg_class_names_path "${PROJECT_ROOT}/configs/nuscenes_lidarseg_class_names.txt"
  --cache_dir "${CACHE_DIR}"
  --prompt_template "${PROMPT_TEMPLATE}"
  --device "${DEVICE}"
  --output_dir "${PREDICTION_DIR}"
)
if [[ -n "${CLASS_NAMES_CSV}" ]]; then
  PREDICT_COMMAND+=(--class_names_csv "${CLASS_NAMES_CSV}")
fi
if [[ -n "${TEXT_MODEL_NAME}" ]]; then
  PREDICT_COMMAND+=(--text_model_name "${TEXT_MODEL_NAME}")
fi
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  PREDICT_COMMAND+=(--local_files_only)
fi
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  PREDICT_COMMAND+=(--skip_existing)
fi
if [[ "${SAVE_SIMILARITIES}" == "1" ]]; then
  PREDICT_COMMAND+=(--save_similarities)
fi
if [[ "${SAVE_POINT_EMBEDDINGS}" == "1" ]]; then
  PREDICT_COMMAND+=(--save_point_embeddings)
fi

EVAL_COMMAND=(
  python "${PROJECT_ROOT}/scripts/eval_lidarseg.py"
  --dataroot "${DATAROOT}"
  --version v1.0-trainval
  --start_idx "${START_IDX}"
  --max_samples "${MAX_SAMPLES}"
  --prediction_dir "${PREDICTION_DIR}"
  --prediction_file_template "sample_{sample_idx:04d}_open_vocab_predictions.npz"
  --class_names_path "${PROJECT_ROOT}/configs/nuscenes_lidarseg_class_names.txt"
  --output_dir "${EVALUATION_DIR}"
)
if [[ "${SKIP_EXISTING}" == "1" ]]; then
  EVAL_COMMAND+=(--skip_existing)
fi

{
  echo "[INFO] V10 launcher started at $(date -Is)"
  echo "[INFO] project_root=${PROJECT_ROOT}"
  echo "[INFO] dataroot=${DATAROOT}"
  echo "[INFO] source_experiment_dir=${SOURCE_EXPERIMENT_DIR}"
  echo "[INFO] experiment_dir=${EXPERIMENT_DIR}"
  echo "[INFO] checkpoint=${CHECKPOINT}"
  echo "[INFO] point_feature_dir=${POINT_FEATURE_DIR}"
  echo "[INFO] prediction_dir=${PREDICTION_DIR}"
  echo "[INFO] evaluation_dir=${EVALUATION_DIR}"
  echo "[INFO] log_file=${LOG_FILE}"
  echo "[INFO] range=start:${START_IDX} max:${MAX_SAMPLES}"
  echo "[INFO] predict command:"
  printf '  %q' "${PREDICT_COMMAND[@]}"
  echo
  if [[ "${SKIP_EVAL}" != "1" ]]; then
    echo "[INFO] eval command:"
    printf '  %q' "${EVAL_COMMAND[@]}"
    echo
  fi
  echo "[INFO] nvidia-smi snapshot:"
  nvidia-smi || true
  echo "[INFO] start running V10"
} 2>&1 | tee "${LOG_FILE}"

STATUS=0
if [[ "${SKIP_PREDICT}" != "1" ]]; then
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
  echo "[INFO] V10 launcher finished at $(date -Is)"
  echo "[INFO] exit_status=${STATUS}"
  echo "[INFO] log_file=${LOG_FILE}"
  echo "[INFO] latest_log=${LATEST_LOG}"
} 2>&1 | tee -a "${LOG_FILE}"

ln -sfn "$(basename "${LOG_FILE}")" "${LATEST_LOG}"
exit "${STATUS}"
