#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POINTCEPT_DIR="${PROJECT_ROOT}/third_party/Pointcept"
LOG_DIR="${PROJECT_ROOT}/outputs/logs"
POINTCEPT_OUT_DIR="${PROJECT_ROOT}/outputs/pointcept"
CHECKPOINT_DIR="${PROJECT_ROOT}/outputs/checkpoints"
EXPERIMENT_NAME="train_ov_head"
STAGE="stage-ov-head"
SMOKE="${SMOKE:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
POINTCEPT_SWEEPS="${POINTCEPT_SWEEPS:-1}"
SMOKE_TRAIN_SAMPLES="${SMOKE_TRAIN_SAMPLES:-8}"
SMOKE_VAL_SAMPLES="${SMOKE_VAL_SAMPLES:-4}"
DISABLE_PRECISE_EVAL="${DISABLE_PRECISE_EVAL:-1}"
OV_HEAD_GATE_MIOU="${OV_HEAD_GATE_MIOU:-0.6632}"
FREEZE_BACKBONE="${FREEZE_BACKBONE:-1}"
OV_HEAD_BACKBONE_OUT_CHANNELS="${OV_HEAD_BACKBONE_OUT_CHANNELS:-96}"
OV_HEAD_TEMPERATURE="${OV_HEAD_TEMPERATURE:-0.07}"
OV_HEAD_TRAINABLE_TEMPERATURE="${OV_HEAD_TRAINABLE_TEMPERATURE:-1}"

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

DATA_ROOT="${POINTCEPT_DATA_ROOT:-}"
if [ -z "${DATA_ROOT}" ]; then
  echo "[ERROR] Set POINTCEPT_DATA_ROOT to the Pointcept processed nuScenes root" | tee "${LOG_FILE}"
  exit 1
fi

TEXT_PROTOTYPES="${POINTCEPT_TEXT_PROTOTYPES:-${PROJECT_ROOT}/outputs/text_prototypes/nuscenes_siglip_16.npz}"
if [ ! -f "${TEXT_PROTOTYPES}" ]; then
  echo "[ERROR] text prototype cache not found at ${TEXT_PROTOTYPES}" | tee "${LOG_FILE}"
  echo "[ERROR] run: python scripts/cache_text_prototypes.py" | tee -a "${LOG_FILE}"
  exit 1
fi

BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-${CHECKPOINT_DIR}/closed_set_baseline.pt}"
if [ ! -f "${BASELINE_CHECKPOINT}" ]; then
  echo "[ERROR] Stage 1 baseline checkpoint not found at ${BASELINE_CHECKPOINT}" | tee "${LOG_FILE}"
  exit 1
fi

SAVE_PATH="${POINTCEPT_SAVE_PATH:-${POINTCEPT_OUT_DIR}/train_ov_head}"
if [ "${SMOKE}" = "1" ]; then
  SAVE_PATH="${POINTCEPT_SAVE_PATH:-${POINTCEPT_OUT_DIR}/train_ov_head_smoke}"
  SMOKE_DATA_ROOT="${POINTCEPT_SMOKE_DATA_ROOT:-${POINTCEPT_OUT_DIR}/smoke_data_ov_head}"
  python "${PROJECT_ROOT}/scripts/make_nuscenes_smoke_infos.py" \
    --source_root "${DATA_ROOT}" \
    --output_root "${SMOKE_DATA_ROOT}" \
    --max_sweeps "${POINTCEPT_SWEEPS}" \
    --train_samples "${SMOKE_TRAIN_SAMPLES}" \
    --val_samples "${SMOKE_VAL_SAMPLES}" | tee -a "${LOG_FILE}"
  DATA_ROOT="${SMOKE_DATA_ROOT}"
fi

TRAIN_SCRIPT="${POINTCEPT_DIR}/tools/train.py"
if [ ! -f "${TRAIN_SCRIPT}" ]; then
  echo "[ERROR] Pointcept launcher not found at ${TRAIN_SCRIPT}" | tee "${LOG_FILE}"
  exit 1
fi

options=(
  "save_path=${SAVE_PATH}"
  "enable_wandb=False"
  "data_root=${DATA_ROOT}"
  "data.train.data_root=${DATA_ROOT}"
  "data.val.data_root=${DATA_ROOT}"
  "data.test.data_root=${DATA_ROOT}"
  "data.train.sweeps=${POINTCEPT_SWEEPS}"
  "data.val.sweeps=${POINTCEPT_SWEEPS}"
  "data.test.sweeps=${POINTCEPT_SWEEPS}"
)

if [ "${SMOKE}" = "1" ]; then
  options+=("epoch=1" "eval_epoch=1" "batch_size=1" "batch_size_val=1")
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
  echo "[INFO] data_root=${DATA_ROOT}"
  echo "[INFO] text_prototypes=${TEXT_PROTOTYPES}"
  echo "[INFO] baseline_checkpoint=${BASELINE_CHECKPOINT}"
  echo "[INFO] freeze_backbone=${FREEZE_BACKBONE}"
  echo "[INFO] gate_miou=${OV_HEAD_GATE_MIOU}"
  echo "[INFO] disable_precise_eval=${DISABLE_PRECISE_EVAL}"
  echo "[INFO] options=${options[*]}"
  nvidia-smi || true
  df -h "${PROJECT_ROOT}" || true
} | tee "${LOG_FILE}"

set +e
PYTHONPATH="${PROJECT_ROOT}:${POINTCEPT_DIR}:${PYTHONPATH:-}" \
DISABLE_PRECISE_EVAL="${DISABLE_PRECISE_EVAL}" \
FREEZE_BACKBONE="${FREEZE_BACKBONE}" \
OV_HEAD_BACKBONE_OUT_CHANNELS="${OV_HEAD_BACKBONE_OUT_CHANNELS}" \
OV_HEAD_TEMPERATURE="${OV_HEAD_TEMPERATURE}" \
OV_HEAD_TRAINABLE_TEMPERATURE="${OV_HEAD_TRAINABLE_TEMPERATURE}" \
python - \
  "${TRAIN_SCRIPT}" \
  "${TEXT_PROTOTYPES}" \
  "${BASELINE_CHECKPOINT}" \
  "${train_args[@]}" <<'PY' 2>&1 | tee -a "${LOG_FILE}"
from __future__ import annotations

import copy
import os
import runpy
import sys
import types

train_script = sys.argv[1]
text_prototypes = sys.argv[2]
baseline_checkpoint = sys.argv[3]
sys.argv = [train_script] + sys.argv[4:]

sys.modules.setdefault("pointops", types.ModuleType("pointops"))

import pointcept.engines.defaults as defaults
import ra_ov3dseg.models.ov_head  # noqa: F401

default_config_parser = defaults.default_config_parser


def _get_attr(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def default_config_parser_with_ov_head(file_path, options):
    cfg = default_config_parser(file_path, options)
    if os.environ.get("DISABLE_PRECISE_EVAL", "1") == "1":
        cfg.hooks = [
            hook
            for hook in cfg.hooks
            if not (isinstance(hook, dict) and hook.get("type") == "PreciseEvaluator")
        ]

    original_model = cfg.model
    backbone = copy.deepcopy(_get_attr(original_model, "backbone"))
    if backbone is None:
        raise ValueError("Pointcept config model has no backbone field.")
    criteria = copy.deepcopy(_get_attr(original_model, "criteria"))
    cfg.model = dict(
        type="RAOVHeadSegmentor",
        backbone=backbone,
        criteria=criteria,
        text_prototypes_path=text_prototypes,
        backbone_weight_path=baseline_checkpoint,
        backbone_out_channels=int(os.environ.get("OV_HEAD_BACKBONE_OUT_CHANNELS", "96")),
        temperature=float(os.environ.get("OV_HEAD_TEMPERATURE", "0.07")),
        trainable_temperature=os.environ.get("OV_HEAD_TRAINABLE_TEMPERATURE", "1") == "1",
        use_projection=True,
        freeze_backbone=os.environ.get("FREEZE_BACKBONE", "1") == "1",
    )
    return cfg


defaults.default_config_parser = default_config_parser_with_ov_head
runpy.run_path(train_script, run_name="__main__")
PY
run_exit=${PIPESTATUS[0]}
set -e

CHECKPOINT=""
if [ "${SMOKE}" != "1" ] && [ -f "${SAVE_PATH}/model/model_best.pth" ]; then
  CHECKPOINT="${CHECKPOINT_DIR}/ov_head_aligned.pt"
  cp "${SAVE_PATH}/model/model_best.pth" "${CHECKPOINT}"
elif [ -f "${SAVE_PATH}/model/model_best.pth" ]; then
  CHECKPOINT="${SAVE_PATH}/model/model_best.pth"
fi

python - "${LOG_FILE}" "${run_exit}" "${STAGE}" "${EXPERIMENT_NAME}" "${SMOKE}" "${CHECKPOINT}" "${OV_HEAD_GATE_MIOU}" <<'PY' | tee -a "${LOG_FILE}"
from __future__ import annotations

import re
import sys
from pathlib import Path

from ra_ov3dseg.utils.run_conclusion import RunConclusion

log_path = Path(sys.argv[1])
run_exit = int(sys.argv[2])
stage = sys.argv[3]
experiment = sys.argv[4]
smoke = sys.argv[5] == "1"
checkpoint = sys.argv[6] or None
gate_miou = float(sys.argv[7])

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
    gate = "OV head smoke run completes, log is captured, and RunConclusion is emitted"
    gate_passed = run_exit == 0
else:
    gate = f"OV head val mIoU >= {gate_miou:.4f}"
    gate_passed = run_exit == 0 and best_miou >= gate_miou

status = "success" if run_exit == 0 else "failed"
notes = "smoke run; backbone frozen" if smoke else "OV head fine-tune; backbone frozen"
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
        "inspect smoke log before full OV head fine-tune"
        if smoke
        else "proceed to Stage 2 acceptance review if gate passed"
    ),
    notes=notes,
)
conclusion.append_to_recap(Path("docs/EXPERIMENT_RECAP.md"))
conclusion.print_block()
PY

exit "${run_exit}"
