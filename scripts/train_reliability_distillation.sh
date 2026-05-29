#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POINTCEPT_DIR="${PROJECT_ROOT}/third_party/Pointcept"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/outputs/logs}"
POINTCEPT_OUT_DIR="${PROJECT_ROOT}/outputs/pointcept"
CHECKPOINT_DIR="${PROJECT_ROOT}/outputs/checkpoints"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-train_reliability_distillation}"
STAGE="stage-reliability"
SMOKE="${SMOKE:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
POINTCEPT_SWEEPS="${POINTCEPT_SWEEPS:-1}"
SMOKE_TRAIN_SAMPLES="${SMOKE_TRAIN_SAMPLES:-8}"
SMOKE_VAL_SAMPLES="${SMOKE_VAL_SAMPLES:-4}"
DISABLE_PRECISE_EVAL="${DISABLE_PRECISE_EVAL:-1}"
FREEZE_BACKBONE="${FREEZE_BACKBONE:-0}"
OV_HEAD_BACKBONE_OUT_CHANNELS="${OV_HEAD_BACKBONE_OUT_CHANNELS:-96}"
OV_HEAD_TEMPERATURE="${OV_HEAD_TEMPERATURE:-0.07}"
OV_HEAD_TRAINABLE_TEMPERATURE="${OV_HEAD_TRAINABLE_TEMPERATURE:-1}"
FORCE_FP32_BACKBONE="${FORCE_FP32_BACKBONE:-1}"
RELIABILITY_THRESHOLD="${RELIABILITY_THRESHOLD:-0.5}"
RELIABILITY_COMPONENT_MODE="${RELIABILITY_COMPONENT_MODE:-full}"
RELIABILITY_COMPONENT_CALIBRATION="${RELIABILITY_COMPONENT_CALIBRATION:-rank}"
DISTILL_LOSS_WEIGHT="${DISTILL_LOSS_WEIGHT:-1.0}"
CE_LOSS_WEIGHT="${CE_LOSS_WEIGHT:-1.0}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-2.0}"
RELIABILITY_STRICT="${RELIABILITY_STRICT:-1}"
RELIABILITY_EPOCHS="${RELIABILITY_EPOCHS:-20}"
RELIABILITY_EVAL_EPOCH="${RELIABILITY_EVAL_EPOCH:-1}"
PROMOTE_RELIABILITY_BEST="${PROMOTE_RELIABILITY_BEST:-0}"

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

OV_HEAD_CHECKPOINT="${OV_HEAD_CHECKPOINT:-${CHECKPOINT_DIR}/ov_head_aligned.pt}"
if [ ! -f "${OV_HEAD_CHECKPOINT}" ]; then
  echo "[ERROR] Stage 2 OV head checkpoint not found at ${OV_HEAD_CHECKPOINT}" | tee "${LOG_FILE}"
  exit 1
fi
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT:-${CHECKPOINT_DIR}/closed_set_baseline.pt}"
if [ ! -f "${BASELINE_CHECKPOINT}" ]; then
  echo "[ERROR] Stage 1 baseline checkpoint not found at ${BASELINE_CHECKPOINT}" | tee "${LOG_FILE}"
  exit 1
fi

RELIABILITY_DIR="${RELIABILITY_DIR:-${PROJECT_ROOT}/outputs/reliability/sam2_siglip_stage4_128_rank}"
DENSE_POINT_DIR="${DENSE_POINT_DIR:-${PROJECT_ROOT}/outputs/dense_point_logits/sam2_siglip_stage3_128}"
if [ ! -d "${RELIABILITY_DIR}" ]; then
  echo "[ERROR] reliability cache dir not found at ${RELIABILITY_DIR}" | tee "${LOG_FILE}"
  exit 1
fi
if [ ! -d "${DENSE_POINT_DIR}" ]; then
  echo "[ERROR] dense point teacher dir not found at ${DENSE_POINT_DIR}" | tee "${LOG_FILE}"
  exit 1
fi

SAVE_PATH="${POINTCEPT_SAVE_PATH:-${POINTCEPT_OUT_DIR}/train_reliability_distillation_${RELIABILITY_COMPONENT_MODE}_t${RELIABILITY_THRESHOLD//./p}}"
if [ "${SMOKE}" = "1" ]; then
  SAVE_PATH="${POINTCEPT_SAVE_PATH:-${POINTCEPT_OUT_DIR}/train_reliability_distillation_smoke}"
  SMOKE_DATA_ROOT="${POINTCEPT_SMOKE_DATA_ROOT:-${POINTCEPT_OUT_DIR}/smoke_data_reliability}"
  smoke_args=(
    --source_root "${DATA_ROOT}"
    --output_root "${SMOKE_DATA_ROOT}"
    --max_sweeps "${POINTCEPT_SWEEPS}"
    --train_samples "${SMOKE_TRAIN_SAMPLES}"
    --val_samples "${SMOKE_VAL_SAMPLES}"
  )
  if [ -n "${RELIABILITY_SAMPLE_INDEX_MANIFEST:-}" ]; then
    smoke_args+=(--sample_indices_path "${RELIABILITY_SAMPLE_INDEX_MANIFEST}")
  fi
  if [ -n "${RELIABILITY_DIR:-}" ]; then
    smoke_args+=(--cache_reliability_dir "${RELIABILITY_DIR}")
  fi
  python "${PROJECT_ROOT}/scripts/make_nuscenes_smoke_infos.py" "${smoke_args[@]}" | tee -a "${LOG_FILE}"
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
  "enable_amp=False"
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
else
  options+=("epoch=${RELIABILITY_EPOCHS}" "eval_epoch=${RELIABILITY_EVAL_EPOCH}")
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
  echo "[INFO] ov_head_checkpoint=${OV_HEAD_CHECKPOINT}"
  echo "[INFO] baseline_checkpoint=${BASELINE_CHECKPOINT}"
  echo "[INFO] reliability_dir=${RELIABILITY_DIR}"
  echo "[INFO] dense_point_dir=${DENSE_POINT_DIR}"
  echo "[INFO] reliability_threshold=${RELIABILITY_THRESHOLD}"
  echo "[INFO] reliability_component_mode=${RELIABILITY_COMPONENT_MODE}"
  echo "[INFO] reliability_component_calibration=${RELIABILITY_COMPONENT_CALIBRATION}"
  echo "[INFO] distill_loss_weight=${DISTILL_LOSS_WEIGHT}"
  echo "[INFO] ce_loss_weight=${CE_LOSS_WEIGHT}"
  echo "[INFO] distill_temperature=${DISTILL_TEMPERATURE}"
  echo "[INFO] freeze_backbone=${FREEZE_BACKBONE}"
  echo "[INFO] strict=${RELIABILITY_STRICT}"
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
FORCE_FP32_BACKBONE="${FORCE_FP32_BACKBONE}" \
BASELINE_CHECKPOINT="${BASELINE_CHECKPOINT}" \
RELIABILITY_DIR="${RELIABILITY_DIR}" \
DENSE_POINT_DIR="${DENSE_POINT_DIR}" \
RELIABILITY_SAMPLE_INDEX_MANIFEST="${RELIABILITY_SAMPLE_INDEX_MANIFEST:-}" \
RELIABILITY_THRESHOLD="${RELIABILITY_THRESHOLD}" \
RELIABILITY_COMPONENT_MODE="${RELIABILITY_COMPONENT_MODE}" \
RELIABILITY_COMPONENT_CALIBRATION="${RELIABILITY_COMPONENT_CALIBRATION}" \
RELIABILITY_STRICT="${RELIABILITY_STRICT}" \
DISTILL_LOSS_WEIGHT="${DISTILL_LOSS_WEIGHT}" \
CE_LOSS_WEIGHT="${CE_LOSS_WEIGHT}" \
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE}" \
python - \
  "${TRAIN_SCRIPT}" \
  "${TEXT_PROTOTYPES}" \
  "${OV_HEAD_CHECKPOINT}" \
  "${train_args[@]}" <<'PY' 2>&1 | tee -a "${LOG_FILE}"
from __future__ import annotations

import copy
import os
import runpy
import sys
import types

train_script = sys.argv[1]
text_prototypes = sys.argv[2]
ov_head_checkpoint = sys.argv[3]
sys.argv = [train_script] + sys.argv[4:]

sys.modules.setdefault("pointops", types.ModuleType("pointops"))

import pointcept.engines.defaults as defaults
import ra_ov3dseg.models.ov_head  # noqa: F401
import ra_ov3dseg.training.reliability_teacher  # noqa: F401

default_config_parser = defaults.default_config_parser
TEACHER_KEYS = ("teacher_logits", "teacher_valid_mask", "reliability_weight")


def _get_attr(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _set_attr(obj, key, value):
    if isinstance(obj, dict):
        obj[key] = value
    else:
        setattr(obj, key, value)


def _merge_sequence(existing, extra):
    if existing is None:
        values = []
        original_tuple = True
    else:
        values = list(existing)
        original_tuple = isinstance(existing, tuple)
    for item in extra:
        if item not in values:
            values.append(item)
    return tuple(values) if original_tuple else values


def _patch_transform_keys(transforms):
    for transform in transforms:
        transform_type = _get_attr(transform, "type", "")
        if transform_type == "Collect":
            keys = _get_attr(transform, "keys", None)
            _set_attr(transform, "keys", _merge_sequence(keys, TEACHER_KEYS))


def _prepend_reliability_transform(train_cfg):
    transforms = list(_get_attr(train_cfg, "transform", []))
    reliability_transform = dict(
        type="RALoadReliabilityTeacher",
        reliability_dir=os.environ["RELIABILITY_DIR"],
        dense_point_dir=os.environ["DENSE_POINT_DIR"],
        sample_index_manifest=os.environ.get("RELIABILITY_SAMPLE_INDEX_MANIFEST") or None,
        teacher_class_start=0,
        teacher_num_classes=16,
        component_mode=os.environ.get("RELIABILITY_COMPONENT_MODE", "full"),
        component_calibration=os.environ.get("RELIABILITY_COMPONENT_CALIBRATION", "rank"),
        strict=os.environ.get("RELIABILITY_STRICT", "1") == "1",
        max_coord_error=0.05,
    )
    transforms.insert(0, reliability_transform)
    _patch_transform_keys(transforms)
    _set_attr(train_cfg, "transform", transforms)


def default_config_parser_with_reliability(file_path, options):
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
        type="RAOVReliabilitySegmentor",
        backbone=backbone,
        criteria=criteria,
        text_prototypes_path=text_prototypes,
        backbone_weight_path=os.environ.get("BASELINE_CHECKPOINT") or None,
        model_weight_path=ov_head_checkpoint,
        backbone_out_channels=int(os.environ.get("OV_HEAD_BACKBONE_OUT_CHANNELS", "96")),
        temperature=float(os.environ.get("OV_HEAD_TEMPERATURE", "0.07")),
        trainable_temperature=os.environ.get("OV_HEAD_TRAINABLE_TEMPERATURE", "1") == "1",
        use_projection=True,
        freeze_backbone=os.environ.get("FREEZE_BACKBONE", "0") == "1",
        force_fp32_backbone=os.environ.get("FORCE_FP32_BACKBONE", "1") == "1",
        ce_loss_weight=float(os.environ.get("CE_LOSS_WEIGHT", "1.0")),
        distill_loss_weight=float(os.environ.get("DISTILL_LOSS_WEIGHT", "1.0")),
        distill_temperature=float(os.environ.get("DISTILL_TEMPERATURE", "2.0")),
        reliability_threshold=float(os.environ.get("RELIABILITY_THRESHOLD", "0.5")),
        require_teacher=True,
    )
    _prepend_reliability_transform(cfg.data.train)
    return cfg


defaults.default_config_parser = default_config_parser_with_reliability
runpy.run_path(train_script, run_name="__main__")
PY
run_exit=${PIPESTATUS[0]}
set -e

CHECKPOINT=""
checkpoint_tag="${RELIABILITY_COMPONENT_MODE}_t${RELIABILITY_THRESHOLD//./p}"
if [ "${SMOKE}" != "1" ] && [ -f "${SAVE_PATH}/model/model_best.pth" ]; then
  CHECKPOINT="${CHECKPOINT_DIR}/ov_reliability_${checkpoint_tag}.pt"
  cp "${SAVE_PATH}/model/model_best.pth" "${CHECKPOINT}"
  if [ "${PROMOTE_RELIABILITY_BEST}" = "1" ]; then
    cp "${SAVE_PATH}/model/model_best.pth" "${CHECKPOINT_DIR}/ov_reliability_best.pt"
  fi
elif [ -f "${SAVE_PATH}/model/model_best.pth" ]; then
  CHECKPOINT="${SAVE_PATH}/model/model_best.pth"
fi

python - "${LOG_FILE}" "${run_exit}" "${STAGE}" "${EXPERIMENT_NAME}" "${SMOKE}" "${CHECKPOINT}" "${RELIABILITY_THRESHOLD}" "${RELIABILITY_COMPONENT_MODE}" <<'PY' | tee -a "${LOG_FILE}"
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
threshold = float(sys.argv[7])
component_mode = sys.argv[8]

text = log_path.read_text(encoding="utf-8", errors="replace")
metric_values: list[float] = []
for line in text.splitlines():
    match = re.search(r"(?:^|\s)mIoU\s+([-+]?(?:\d*\.\d+|\d+))", line)
    if match is None:
        match = re.search(r"Val result:\s*mIoU/mAcc/allAcc\s+([-+]?(?:\d*\.\d+|\d+))", line)
    if match is None:
        match = re.search(r"Best mIoU:\s*([-+]?(?:\d*\.\d+|\d+))", line)
    if match is None:
        continue
    value = float(match.group(1))
    if value > 1.0 and value <= 100.0:
        value = value / 100.0
    if 0.0 <= value <= 1.0:
        metric_values.append(value)

best_miou = max(metric_values) if metric_values else 0.0
stats_match = re.search(
    r"distill_valid_ratio=([-+]?(?:\d*\.\d+|\d+)).*distill_mean_weight=([-+]?(?:\d*\.\d+|\d+))",
    text,
)
secondary = {"threshold": threshold}
if stats_match is not None:
    secondary["distill_valid_ratio"] = float(stats_match.group(1))
    secondary["distill_mean_weight"] = float(stats_match.group(2))

if smoke:
    gate = "Stage 4 reliability smoke run completes with teacher fields loaded"
    gate_passed = run_exit == 0 and stats_match is not None
else:
    gate = "Stage 4 reliability run completes and records val mIoU for ablation"
    gate_passed = run_exit == 0 and bool(metric_values)

status = "success" if run_exit == 0 else "failed"
notes = f"component={component_mode}; threshold={threshold:.3f}"
if run_exit == 0 and not metric_values and not smoke:
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
    secondary=secondary,
    runtime_seconds=0.0,
    checkpoint=checkpoint,
    artifacts=[str(log_path)],
    next_step=(
        "inspect smoke log, then launch threshold ablation"
        if smoke and gate_passed
        else "add this run to the Stage 4 ablation table"
        if gate_passed
        else "fix reliability distillation wiring before continuing"
    ),
    notes=notes,
)
conclusion.append_to_recap(Path("docs/EXPERIMENT_RECAP.md"))
conclusion.print_block()
PY

exit "${run_exit}"
