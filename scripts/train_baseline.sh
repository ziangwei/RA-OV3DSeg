#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POINTCEPT_DIR="${PROJECT_ROOT}/third_party/Pointcept"
LOG_DIR="${PROJECT_ROOT}/outputs/logs"
POINTCEPT_OUT_DIR="${PROJECT_ROOT}/outputs/pointcept"
CHECKPOINT_DIR="${PROJECT_ROOT}/outputs/checkpoints"
EXPERIMENT_NAME="train_baseline"
STAGE="stage-baseline"
SMOKE="${SMOKE:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
POINTCEPT_SWEEPS="${POINTCEPT_SWEEPS:-1}"
SMOKE_TRAIN_SAMPLES="${SMOKE_TRAIN_SAMPLES:-8}"
SMOKE_VAL_SAMPLES="${SMOKE_VAL_SAMPLES:-4}"
DISABLE_PRECISE_EVAL="${DISABLE_PRECISE_EVAL:-1}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}" "${POINTCEPT_OUT_DIR}" "${CHECKPOINT_DIR}"

timestamp="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}_${timestamp}.log"
ln -sfn "$(basename "${LOG_FILE}")" "${LOG_DIR}/${EXPERIMENT_NAME}_latest.log"

if [ ! -d "${POINTCEPT_DIR}" ]; then
  echo "[ERROR] Pointcept checkout not found at ${POINTCEPT_DIR}" | tee "${LOG_FILE}"
  exit 1
fi

CONFIG_PATH="${POINTCEPT_CONFIG:-}"
if [ -z "${CONFIG_PATH}" ]; then
  CONFIG_PATH="$(find "${POINTCEPT_DIR}/configs/nuscenes" -maxdepth 1 -type f -name "*spunet*.py" | sort | head -n 1 || true)"
fi
if [ -z "${CONFIG_PATH}" ] || [ ! -f "${CONFIG_PATH}" ]; then
  echo "[ERROR] Pointcept SpUNet config not found. Set POINTCEPT_CONFIG=/abs/path/to/config.py" | tee "${LOG_FILE}"
  exit 1
fi

SAVE_PATH="${POINTCEPT_SAVE_PATH:-${POINTCEPT_OUT_DIR}/train_baseline}"
if [ "${SMOKE}" = "1" ]; then
  SAVE_PATH="${POINTCEPT_SAVE_PATH:-${POINTCEPT_OUT_DIR}/train_baseline_smoke}"
fi

TRAIN_SCRIPT="${POINTCEPT_DIR}/tools/train.py"
if [ ! -f "${TRAIN_SCRIPT}" ]; then
  echo "[ERROR] Pointcept launcher not found at ${TRAIN_SCRIPT}" | tee "${LOG_FILE}"
  exit 1
fi

options=(
  "save_path=${SAVE_PATH}"
  "enable_wandb=False"
)

DATA_ROOT="${POINTCEPT_DATA_ROOT:-}"
if [ "${SMOKE}" = "1" ]; then
  if [ -z "${DATA_ROOT}" ]; then
    echo "[ERROR] SMOKE=1 requires POINTCEPT_DATA_ROOT pointing at processed train/val pkl files" | tee -a "${LOG_FILE}"
    exit 1
  fi
  SMOKE_DATA_ROOT="${POINTCEPT_SMOKE_DATA_ROOT:-${POINTCEPT_OUT_DIR}/smoke_data}"
  python "${PROJECT_ROOT}/scripts/make_nuscenes_smoke_infos.py" \
    --source_root "${DATA_ROOT}" \
    --output_root "${SMOKE_DATA_ROOT}" \
    --max_sweeps "${POINTCEPT_SWEEPS}" \
    --train_samples "${SMOKE_TRAIN_SAMPLES}" \
    --val_samples "${SMOKE_VAL_SAMPLES}" | tee -a "${LOG_FILE}"
  DATA_ROOT="${SMOKE_DATA_ROOT}"
fi

if [ -n "${DATA_ROOT}" ]; then
  options+=(
    "data_root=${DATA_ROOT}"
    "data.train.data_root=${DATA_ROOT}"
    "data.val.data_root=${DATA_ROOT}"
    "data.test.data_root=${DATA_ROOT}"
  )
fi

if [ "${SMOKE}" = "1" ]; then
  options+=("epoch=1" "eval_epoch=1" "batch_size=1" "batch_size_val=1")
fi

if [ -n "${POINTCEPT_SWEEPS}" ]; then
  options+=(
    "data.train.sweeps=${POINTCEPT_SWEEPS}"
    "data.val.sweeps=${POINTCEPT_SWEEPS}"
    "data.test.sweeps=${POINTCEPT_SWEEPS}"
  )
fi

if [ -n "${POINTCEPT_OPTIONS:-}" ]; then
  # shellcheck disable=SC2206
  extra_options=(${POINTCEPT_OPTIONS})
  options+=("${extra_options[@]}")
fi

train_args=(
  "--config-file" "${CONFIG_PATH}"
  "--num-gpus" "${NUM_GPUS}"
  "--options" "${options[@]}"
)

{
  echo "[INFO] experiment=${EXPERIMENT_NAME} stage=${STAGE} smoke=${SMOKE}"
  echo "[INFO] started=$(date -Is)"
  echo "[INFO] config=${CONFIG_PATH}"
  echo "[INFO] save_path=${SAVE_PATH}"
  echo "[INFO] num_gpus=${NUM_GPUS}"
  if [ -n "${DATA_ROOT}" ]; then
    echo "[INFO] data_root=${DATA_ROOT}"
  fi
  echo "[INFO] pointcept_sweeps=${POINTCEPT_SWEEPS}"
  echo "[INFO] disable_precise_eval=${DISABLE_PRECISE_EVAL}"
  echo "[INFO] options=${options[*]}"
  nvidia-smi || true
  df -h "${PROJECT_ROOT}" || true
} | tee "${LOG_FILE}"

set +e
PYTHONPATH="${PROJECT_ROOT}:${POINTCEPT_DIR}:${PYTHONPATH:-}" python - "${TRAIN_SCRIPT}" "${train_args[@]}" <<'PY' 2>&1 | tee -a "${LOG_FILE}"
from __future__ import annotations

import runpy
import os
import sys
import types

train_script = sys.argv[1]
sys.argv = [train_script] + sys.argv[2:]

# SpUNet training does not need pointops, but Pointcept package initialization
# imports optional hooks that reference it.
sys.modules.setdefault("pointops", types.ModuleType("pointops"))

if os.environ.get("DISABLE_PRECISE_EVAL", "1") == "1":
    import pointcept.engines.defaults as defaults

    default_config_parser = defaults.default_config_parser

    def default_config_parser_without_precise_eval(file_path, options):
        cfg = default_config_parser(file_path, options)
        cfg.hooks = [
            hook
            for hook in cfg.hooks
            if not (
                isinstance(hook, dict)
                and hook.get("type") == "PreciseEvaluator"
            )
        ]
        return cfg

    defaults.default_config_parser = default_config_parser_without_precise_eval

runpy.run_path(train_script, run_name="__main__")
PY
run_exit=${PIPESTATUS[0]}
set -e

CHECKPOINT=""
if [ "${SMOKE}" != "1" ] && [ -f "${SAVE_PATH}/model/model_best.pth" ]; then
  CHECKPOINT="${CHECKPOINT_DIR}/closed_set_baseline.pt"
  cp "${SAVE_PATH}/model/model_best.pth" "${CHECKPOINT}"
elif [ -f "${SAVE_PATH}/model/model_best.pth" ]; then
  CHECKPOINT="${SAVE_PATH}/model/model_best.pth"
fi

python - "${LOG_FILE}" "${run_exit}" "${STAGE}" "${EXPERIMENT_NAME}" "${SMOKE}" "${CHECKPOINT}" <<'PY' | tee -a "${LOG_FILE}"
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

from ra_ov3dseg.utils.run_conclusion import RunConclusion

log_path = Path(sys.argv[1])
run_exit = int(sys.argv[2])
stage = sys.argv[3]
experiment = sys.argv[4]
smoke = sys.argv[5] == "1"
checkpoint = sys.argv[6] or None

text = log_path.read_text(encoding="utf-8", errors="replace")
metric_values: list[float] = []
for line in text.splitlines():
    match = re.search(r"(?:^|\s)mIoU\s+([-+]?(?:\d*\.\d+|\d+))", line)
    if match is None:
        continue
    value = float(match.group(1))
    if value > 1.0 and value <= 100.0:
        value = value / 100.0
    if 0.0 <= value <= 1.0:
        metric_values.append(value)

best_miou = max(metric_values) if metric_values else 0.0
if smoke:
    gate = "smoke run completes, log is captured, and RunConclusion is emitted"
    gate_passed = run_exit == 0
else:
    gate = "full nuScenes-lidarseg val mIoU >= 0.70"
    gate_passed = run_exit == 0 and best_miou >= 0.70

status = "success" if run_exit == 0 else "failed"
notes = "smoke run" if smoke else "full baseline run"
if run_exit == 0 and not metric_values:
    notes += "; no mIoU parsed from Pointcept log"
elif run_exit != 0:
    notes += f"; Pointcept launcher exited with code {run_exit}"

conclusion = RunConclusion(
    stage=stage,
    experiment=experiment,
    status=status,
    gate=gate,
    gate_passed=gate_passed,
    primary_metric_name="val_miou",
    primary_metric_value=best_miou,
    secondary={},
    runtime_seconds=0.0,
    checkpoint=checkpoint,
    artifacts=[str(log_path)],
    next_step=(
        "inspect smoke log and request owner confirmation before full baseline"
        if smoke
        else "proceed to Stage 1 acceptance review if gate passed"
    ),
    notes=notes,
)
conclusion.append_to_recap(Path("docs/EXPERIMENT_RECAP.md"))
conclusion.print_block()
PY

exit "${run_exit}"
