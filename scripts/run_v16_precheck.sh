#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DATAROOT="${PROJECT_ROOT}/data/nuscenes"
VERSION="v1.0-trainval"
START_IDX=0
MAX_SAMPLES=128
SAMPLE_IDX=""
MAX_POINTS=50000
OUTPUT_DIR="${PROJECT_ROOT}/outputs/reports"
STRICT=0

usage() {
  cat <<'EOF'
Run the pre-V16 sanity check before vendoring a mature backbone.

This check verifies:
  - LiDAR point count == lidarseg label count
  - current raw 32-class training split vs official 16-class nuScenes lidarseg mapping
  - V15 cylindrical range coverage
  - ignore-index CE masking
  - raw dataset coordinate consistency

Examples:
  bash scripts/run_v16_precheck.sh

  bash scripts/run_v16_precheck.sh \
    --dataroot /abs/path/to/RA-OV3DSeg/data/nuscenes \
    --max_samples 32

Options:
  --dataroot PATH
  --version VERSION
  --sample_idx N
  --start_idx N
  --max_samples N
  --max_points N
  --output_dir PATH
  --strict
  -h, --help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataroot) DATAROOT="$2"; shift 2 ;;
    --version) VERSION="$2"; shift 2 ;;
    --sample_idx) SAMPLE_IDX="$2"; shift 2 ;;
    --start_idx) START_IDX="$2"; shift 2 ;;
    --max_samples) MAX_SAMPLES="$2"; shift 2 ;;
    --max_points) MAX_POINTS="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    --strict) STRICT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "[ERROR] unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

LOG_DIR="${PROJECT_ROOT}/outputs/logs"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/v16_precheck_${TIMESTAMP}.log"
LATEST_LOG="${LOG_DIR}/v16_precheck_latest.log"
ln -sfn "$(basename "${LOG_FILE}")" "${LATEST_LOG}"

COMMAND=(
  python "${PROJECT_ROOT}/scripts/pre_v16_sanity_check.py"
  --dataroot "${DATAROOT}"
  --version "${VERSION}"
  --start_idx "${START_IDX}"
  --max_samples "${MAX_SAMPLES}"
  --max_points "${MAX_POINTS}"
  --output_dir "${OUTPUT_DIR}"
)
if [[ -n "${SAMPLE_IDX}" ]]; then
  COMMAND+=(--sample_idx "${SAMPLE_IDX}")
fi
if [[ "${STRICT}" == "1" ]]; then
  COMMAND+=(--strict)
fi

{
  echo "[INFO] V16 precheck started at $(date -Is)"
  echo "[INFO] project_root=${PROJECT_ROOT}"
  echo "[INFO] dataroot=${DATAROOT}"
  echo "[INFO] version=${VERSION}"
  echo "[INFO] start_idx=${START_IDX}"
  echo "[INFO] max_samples=${MAX_SAMPLES}"
  echo "[INFO] max_points=${MAX_POINTS}"
  echo "[INFO] output_dir=${OUTPUT_DIR}"
  echo "[INFO] log_file=${LOG_FILE}"
  echo "[INFO] disk snapshot:"
  df -h "${PROJECT_ROOT}" "${DATAROOT}" || true
  printf '[INFO] command:'
  printf ' %q' "${COMMAND[@]}"
  echo
} 2>&1 | tee "${LOG_FILE}"

set +e
"${COMMAND[@]}" 2>&1 | tee -a "${LOG_FILE}"
STATUS=${PIPESTATUS[0]}
set -e

{
  echo "[INFO] V16 precheck finished at $(date -Is)"
  echo "[INFO] exit_status=${STATUS}"
  echo "[INFO] log_file=${LOG_FILE}"
  echo "[INFO] latest_log=${LATEST_LOG}"
} 2>&1 | tee -a "${LOG_FILE}"

exit "${STATUS}"
