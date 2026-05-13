#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POINTCEPT_DIR="${PROJECT_ROOT}/third_party/Pointcept"
LOG_DIR="${PROJECT_ROOT}/outputs/logs"
POINTCEPT_OUT_DIR="${PROJECT_ROOT}/outputs/pointcept"
EXPERIMENT_NAME="eval_baseline_fast"
STAGE="stage-baseline"
POINTCEPT_SWEEPS="${POINTCEPT_SWEEPS:-1}"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_DIR}" "${POINTCEPT_OUT_DIR}"

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

SAVE_PATH="${POINTCEPT_SAVE_PATH:-${POINTCEPT_OUT_DIR}/train_baseline}"
WEIGHT_PATH="${POINTCEPT_WEIGHT:-${SAVE_PATH}/model/model_best.pth}"
if [ ! -f "${WEIGHT_PATH}" ]; then
  echo "[ERROR] checkpoint not found at ${WEIGHT_PATH}" | tee "${LOG_FILE}"
  exit 1
fi

{
  echo "[INFO] experiment=${EXPERIMENT_NAME} stage=${STAGE}"
  echo "[INFO] started=$(date -Is)"
  echo "[INFO] config=${CONFIG_PATH}"
  echo "[INFO] save_path=${SAVE_PATH}"
  echo "[INFO] data_root=${DATA_ROOT}"
  echo "[INFO] pointcept_sweeps=${POINTCEPT_SWEEPS}"
  echo "[INFO] weight=${WEIGHT_PATH}"
  nvidia-smi || true
  df -h "${PROJECT_ROOT}" || true
} | tee "${LOG_FILE}"

set +e
PYTHONPATH="${PROJECT_ROOT}:${POINTCEPT_DIR}:${PYTHONPATH:-}" python - \
  "${CONFIG_PATH}" \
  "${DATA_ROOT}" \
  "${SAVE_PATH}" \
  "${WEIGHT_PATH}" \
  "${POINTCEPT_SWEEPS}" \
  "${LOG_FILE}" \
  "${STAGE}" \
  "${EXPERIMENT_NAME}" <<'PY' 2>&1 | tee -a "${LOG_FILE}"
from __future__ import annotations

import sys
import types
from collections import OrderedDict
from pathlib import Path

sys.modules.setdefault("pointops", types.ModuleType("pointops"))

import torch
from torch.utils.data import DataLoader

from pointcept.datasets import build_dataset, collate_fn
from pointcept.engines.defaults import default_config_parser, default_setup
from pointcept.engines.hooks.evaluator import SemSegEvaluator
from pointcept.models import build_model
from pointcept.utils.events import EventStorage
from pointcept.utils.logger import get_root_logger
from ra_ov3dseg.utils.run_conclusion import RunConclusion


class _FastValTrainer:
    pass


def load_weight(model, weight_path: Path) -> None:
    checkpoint = torch.load(
        weight_path,
        map_location=lambda storage, loc: storage.cuda(),
        weights_only=False,
    )
    state_dict = checkpoint.get("state_dict", checkpoint)
    weight = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]
        weight[key] = value
    load_state_info = model.load_state_dict(weight, strict=True)
    if load_state_info[0] or load_state_info[1]:
        raise RuntimeError(f"checkpoint load mismatch: {load_state_info}")


config_path = sys.argv[1]
data_root = sys.argv[2]
save_path = sys.argv[3]
weight_path = Path(sys.argv[4])
pointcept_sweeps = int(sys.argv[5])
log_path = Path(sys.argv[6])
stage = sys.argv[7]
experiment = sys.argv[8]

options = {
    "save_path": save_path,
    "enable_wandb": False,
    "resume": True,
    "evaluate": True,
    "data_root": data_root,
    "data.train.data_root": data_root,
    "data.val.data_root": data_root,
    "data.test.data_root": data_root,
    "data.train.sweeps": pointcept_sweeps,
    "data.val.sweeps": pointcept_sweeps,
    "data.test.sweeps": pointcept_sweeps,
}
cfg = default_setup(default_config_parser(config_path, options))
logger = get_root_logger(log_file=str(log_path), file_mode="a")
logger.info("=> Building model for fast validation ...")
model = build_model(cfg.model).cuda()
load_weight(model, weight_path)
logger.info("=> Loaded checkpoint: %s", weight_path)

logger.info("=> Building validation dataset & dataloader ...")
val_data = build_dataset(cfg.data.val)
val_loader = DataLoader(
    val_data,
    batch_size=cfg.batch_size_val_per_gpu,
    shuffle=False,
    num_workers=cfg.num_worker_per_gpu,
    pin_memory=True,
    collate_fn=collate_fn,
)

trainer = _FastValTrainer()
trainer.cfg = cfg
trainer.model = model
trainer.val_loader = val_loader
trainer.logger = logger
trainer.writer = None
trainer.epoch = 0
trainer.comm_info = {}
trainer.best_metric_value = -float("inf")

status = "success"
notes = "fast validation via Pointcept SemSegEvaluator; skips PreciseEvaluator"
try:
    with EventStorage() as storage:
        trainer.storage = storage
        evaluator = SemSegEvaluator()
        evaluator.trainer = trainer
        evaluator.eval()
except Exception as exc:
    status = "failed"
    notes += f"; {type(exc).__name__}: {exc}"
    metric = 0.0
else:
    metric = float(trainer.comm_info.get("current_metric_value", 0.0))

conclusion = RunConclusion(
    stage=stage,
    experiment=experiment,
    status=status,
    gate="full nuScenes-lidarseg fast val mIoU >= 0.70",
    gate_passed=status == "success" and metric >= 0.70,
    primary_metric_name="val_miou",
    primary_metric_value=metric,
    secondary={},
    runtime_seconds=0.0,
    checkpoint=str(weight_path),
    artifacts=[str(log_path)],
    next_step="use this checkpoint for Stage 1 acceptance if gate passed",
    notes=notes,
)
conclusion.append_to_recap(Path("docs/EXPERIMENT_RECAP.md"))
conclusion.print_block()
if status != "success":
    raise SystemExit(1)
PY
run_exit=${PIPESTATUS[0]}
set -e

exit "${run_exit}"
