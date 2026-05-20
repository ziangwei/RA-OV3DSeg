#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

STAGE="stage-reliability"
EXPERIMENT_NAME="pilot_reliability_threshold_sweep"
POINTCEPT_SWEEPS="${POINTCEPT_SWEEPS:-1}"
THRESHOLDS="${THRESHOLDS:-0.0 0.3 0.5 0.7 0.9}"
PILOT_TRAIN_SAMPLES="${PILOT_TRAIN_SAMPLES:-128}"
PILOT_VAL_SAMPLES="${PILOT_VAL_SAMPLES:-32}"
PILOT_EPOCHS="${PILOT_EPOCHS:-5}"
PILOT_EVAL_EPOCH="${PILOT_EVAL_EPOCH:-1}"
PILOT_OVERWRITE="${PILOT_OVERWRITE:-1}"
PILOT_SUBSET_ROOT="${PILOT_SUBSET_ROOT:-${PROJECT_ROOT}/outputs/pointcept/reliability_subset_128}"
PILOT_OUT_ROOT="${PILOT_OUT_ROOT:-${PROJECT_ROOT}/outputs/pointcept/reliability_pilot}"
PILOT_LOG_DIR="${PILOT_LOG_DIR:-${PROJECT_ROOT}/outputs/logs/reliability_pilot}"
POINTCEPT_DATA_ROOT="${POINTCEPT_DATA_ROOT:-${PROJECT_ROOT}/data/nuscenes_pointcept_processed}"
RELIABILITY_SAMPLE_INDEX_MANIFEST="${RELIABILITY_SAMPLE_INDEX_MANIFEST:-${PROJECT_ROOT}/outputs/diagnostics/stage3_teacher_indices_128.json}"
RELIABILITY_DIR="${RELIABILITY_DIR:-${PROJECT_ROOT}/outputs/reliability/sam2_siglip_stage4_128_rank}"
DENSE_POINT_DIR="${DENSE_POINT_DIR:-${PROJECT_ROOT}/outputs/dense_point_logits/sam2_siglip_stage3_128}"

mkdir -p "${PILOT_OUT_ROOT}" "${PILOT_LOG_DIR}"

safe_rm_path() {
  local target="$1"
  case "${target}" in
    "${PROJECT_ROOT}/outputs/"*)
      rm -rf "${target}"
      ;;
    *)
      echo "[pilot][ERROR] refusing to remove path outside outputs/: ${target}" >&2
      exit 2
      ;;
  esac
}

if [ "${PILOT_OVERWRITE}" = "1" ]; then
  safe_rm_path "${PILOT_SUBSET_ROOT}"
fi

echo "[pilot] building cached subset: ${PILOT_SUBSET_ROOT}"
python "${PROJECT_ROOT}/scripts/make_nuscenes_smoke_infos.py" \
  --source_root "${POINTCEPT_DATA_ROOT}" \
  --output_root "${PILOT_SUBSET_ROOT}" \
  --max_sweeps "${POINTCEPT_SWEEPS}" \
  --train_samples "${PILOT_TRAIN_SAMPLES}" \
  --val_samples "${PILOT_VAL_SAMPLES}" \
  --sample_indices_path "${RELIABILITY_SAMPLE_INDEX_MANIFEST}"

python - \
  "${PILOT_SUBSET_ROOT}" \
  "${RELIABILITY_DIR}" \
  "${DENSE_POINT_DIR}" \
  "${POINTCEPT_SWEEPS}" <<'PY'
from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

subset_root = Path(sys.argv[1])
reliability_dir = Path(sys.argv[2])
dense_point_dir = Path(sys.argv[3])
sweeps = sys.argv[4]

def load_infos(split: str):
    path = subset_root / "info" / f"nuscenes_infos_{sweeps}sweeps_{split}.pkl"
    with path.open("rb") as f:
        return pickle.load(f)

train_infos = load_infos("train")
val_infos = load_infos("val")
missing_names = []
missing_cache = []
sample_indices = []
for split, infos in (("train", train_infos), ("val", val_infos)):
    for row, info in enumerate(infos):
        name = str(info.get("lidar_token", ""))
        match = re.fullmatch(r"sample_(\d+)", name)
        if match is None:
            missing_names.append(f"{split}[{row}]={name}")
            continue
        sample_idx = int(match.group(1))
        sample_indices.append(sample_idx)
        reliability = reliability_dir / f"sample_{sample_idx:04d}_reliability.npz"
        dense = dense_point_dir / f"sample_{sample_idx:04d}_dense_point_logits.npz"
        if not reliability.exists() or not dense.exists():
            missing_cache.append(f"sample_{sample_idx:04d}")

if missing_names:
    raise SystemExit(
        "[pilot][ERROR] subset contains non-cache-resolvable names: "
        + ", ".join(missing_names[:8])
    )
if missing_cache:
    raise SystemExit(
        "[pilot][ERROR] subset references samples without cache: "
        + ", ".join(sorted(set(missing_cache))[:8])
    )
if not train_infos or not val_infos:
    raise SystemExit("[pilot][ERROR] subset train/val split is empty")

print(
    f"[pilot] preflight ok: train={len(train_infos)} val={len(val_infos)} "
    f"unique_cached_samples={len(set(sample_indices))}"
)
PY

SUMMARY_TSV="${PILOT_OUT_ROOT}/pilot_threshold_summary.tsv"
if [ "${PILOT_OVERWRITE}" = "1" ]; then
  rm -f "${SUMMARY_TSV}"
fi
printf "threshold\tstatus\tgate_passed\tval_miou\tdistill_valid_ratio\tdistill_mean_weight\tlog\n" > "${SUMMARY_TSV}"

run_status="success"
failed_threshold=""
failed_log=""

for threshold in ${THRESHOLDS}; do
  tag="${threshold/./p}"
  run_name="pilot_reliability_t${tag}"
  run_dir="${PILOT_OUT_ROOT}/${run_name}"
  console_log="${PILOT_LOG_DIR}/${run_name}.console.log"
  inner_log_dir="${PILOT_LOG_DIR}/${run_name}_inner"
  if [ "${PILOT_OVERWRITE}" = "1" ]; then
    safe_rm_path "${run_dir}"
    safe_rm_path "${inner_log_dir}"
    safe_rm_path "${PROJECT_ROOT}/outputs/pointcept/${run_name}"
    rm -f "${console_log}"
    rm -f "${PROJECT_ROOT}/outputs/logs/${run_name}_"*.log "${PROJECT_ROOT}/outputs/logs/${run_name}_latest.log"
  fi

  echo "[pilot] threshold=${threshold} running; full log: ${console_log}"
  set +e
  EXPERIMENT_NAME="${run_name}" \
  LOG_DIR="${inner_log_dir}" \
  POINTCEPT_DATA_ROOT="${PILOT_SUBSET_ROOT}" \
  POINTCEPT_SAVE_PATH="${run_dir}" \
  RELIABILITY_SAMPLE_INDEX_MANIFEST="${RELIABILITY_SAMPLE_INDEX_MANIFEST}" \
  RELIABILITY_DIR="${RELIABILITY_DIR}" \
  DENSE_POINT_DIR="${DENSE_POINT_DIR}" \
  RELIABILITY_THRESHOLD="${threshold}" \
  RELIABILITY_EPOCHS="${PILOT_EPOCHS}" \
  RELIABILITY_EVAL_EPOCH="${PILOT_EVAL_EPOCH}" \
  POINTCEPT_SWEEPS="${POINTCEPT_SWEEPS}" \
  bash "${PROJECT_ROOT}/scripts/train_reliability_distillation.sh" > "${console_log}" 2>&1
  exit_code=$?
  set -e

  parsed=$(python - "${console_log}" "${threshold}" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
threshold = sys.argv[2]
text = log_path.read_text(encoding="utf-8", errors="replace")
block_match = list(re.finditer(r"========== RUN_CONCLUSION ==========(.*?)====================================", text, re.S))
block = block_match[-1].group(1) if block_match else ""

def field(name: str, default: str = "-") -> str:
    match = re.search(rf"^{name}:\s*(.+)$", block, re.M)
    return match.group(1).strip() if match else default

status = field("status", "missing")
gate_passed = field("gate_passed", "no")
metric_match = re.search(r"primary_metric:\s*val_miou\s*=\s*([-+]?(?:\d*\.\d+|\d+))", block)
val_miou = metric_match.group(1) if metric_match else "0.0000"
secondary = field("  secondary", "")
valid_match = re.search(r"distill_valid_ratio=([-+]?(?:\d*\.\d+|\d+))", secondary)
weight_match = re.search(r"distill_mean_weight=([-+]?(?:\d*\.\d+|\d+))", secondary)
valid = valid_match.group(1) if valid_match else "0.0000"
weight = weight_match.group(1) if weight_match else "0.0000"
print("\t".join([threshold, status, gate_passed, val_miou, valid, weight, str(log_path)]))
PY
)
  printf "%s\n" "${parsed}" >> "${SUMMARY_TSV}"

  IFS=$'\t' read -r parsed_threshold parsed_status parsed_gate parsed_miou parsed_valid parsed_weight parsed_log <<< "${parsed}"
  echo "[pilot] threshold=${parsed_threshold} status=${parsed_status} val_miou=${parsed_miou} distill_valid_ratio=${parsed_valid}"

  if [ "${exit_code}" -ne 0 ]; then
    run_status="failed"
    failed_threshold="${threshold}"
    failed_log="${console_log}"
    echo "[pilot][ERROR] threshold=${threshold} failed. Last 60 log lines:"
    tail -n 60 "${console_log}" || true
    break
  fi
done

python - "${SUMMARY_TSV}" "${run_status}" "${failed_threshold}" "${failed_log}" <<'PY'
from __future__ import annotations

import csv
import sys
from pathlib import Path

from ra_ov3dseg.utils.run_conclusion import RunConclusion

summary_path = Path(sys.argv[1])
run_status = sys.argv[2]
failed_threshold = sys.argv[3]
failed_log = sys.argv[4]

rows = list(csv.DictReader(summary_path.open("r", encoding="utf-8"), delimiter="\t"))
completed = [row for row in rows if row["status"] == "success"]
best = max(completed, key=lambda row: float(row["val_miou"])) if completed else None
best_miou = float(best["val_miou"]) if best else 0.0
best_threshold = float(best["threshold"]) if best else 0.0
artifacts = [str(summary_path)]
if failed_log:
    artifacts.append(failed_log)

print("[pilot] summary table:")
print("threshold  status   val_miou  distill_valid_ratio  distill_mean_weight")
for row in rows:
    print(
        f"{row['threshold']:>9}  {row['status']:<7}  {row['val_miou']:>8}  "
        f"{row['distill_valid_ratio']:>20}  {row['distill_mean_weight']:>19}"
    )

notes = (
    f"best_threshold={best_threshold:.3f}; completed_runs={len(completed)}/{len(rows)}"
    if run_status == "success"
    else f"failed_threshold={failed_threshold}; completed_runs={len(completed)}/{len(rows)}"
)
conclusion = RunConclusion(
    stage="stage-reliability",
    experiment="pilot_reliability_threshold_sweep",
    status="success" if run_status == "success" else "failed",
    gate="128-cache pilot threshold sweep completes; not the final Stage 4 gate",
    gate_passed=run_status == "success" and len(completed) == len(rows) and len(rows) > 0,
    primary_metric_name="best_pilot_val_miou",
    primary_metric_value=best_miou,
    secondary={"best_threshold": best_threshold},
    runtime_seconds=0.0,
    checkpoint=None,
    artifacts=artifacts,
    next_step=(
        "inspect pilot table and decide whether to generate a larger teacher cache"
        if run_status == "success"
        else "fix pilot failure before launching more runs"
    ),
    notes=notes,
)
conclusion.append_to_recap(Path("docs/EXPERIMENT_RECAP.md"))
conclusion.print_block()
PY

if [ "${run_status}" != "success" ]; then
  exit 1
fi
