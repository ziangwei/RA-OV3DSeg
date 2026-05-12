#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POINTCEPT_DIR="${PROJECT_ROOT}/third_party/Pointcept"
POINTCEPT_COMMIT="d74c646db6abec569d0f23e0c34e7ddfce142789"
POINTCEPT_REPO="https://github.com/Pointcept/Pointcept.git"

echo "[setup_env] project_root=${PROJECT_ROOT}"
echo "[setup_env] pointcept_commit=${POINTCEPT_COMMIT}"

mkdir -p "${PROJECT_ROOT}/third_party"
if [ ! -d "${POINTCEPT_DIR}/.git" ]; then
  echo "[setup_env] cloning Pointcept..."
  git clone "${POINTCEPT_REPO}" "${POINTCEPT_DIR}"
fi

cd "${POINTCEPT_DIR}"
git fetch
git checkout "${POINTCEPT_COMMIT}"

cd "${PROJECT_ROOT}"
if [ -f "${POINTCEPT_DIR}/requirements.txt" ]; then
  echo "[setup_env] Pointcept requirements found at third_party/Pointcept/requirements.txt"
  echo "[setup_env] review requirements-pointcept.txt if this install fails in your CUDA/PyTorch environment"
fi

echo "[setup_env] installing Pointcept base requirements..."
pip install -r requirements-pointcept.txt

if [ -f "${POINTCEPT_DIR}/setup.py" ] || [ -f "${POINTCEPT_DIR}/pyproject.toml" ]; then
  echo "[setup_env] pip install -e third_party/Pointcept --no-deps ..."
  pip install -e "${POINTCEPT_DIR}" --no-deps
else
  echo "[setup_env] Pointcept has no setup.py/pyproject.toml; registering via .pth ..."
  POINTCEPT_DIR="${POINTCEPT_DIR}" python - <<'PY'
from __future__ import annotations

import os
import site
import sysconfig
from pathlib import Path

pointcept_dir = Path(os.environ["POINTCEPT_DIR"]).resolve()
site_dirs: list[Path] = []
try:
    site_dirs.extend(Path(path) for path in site.getsitepackages())
except AttributeError:
    pass
purelib = sysconfig.get_paths().get("purelib")
if purelib:
    site_dirs.append(Path(purelib))

seen: set[Path] = set()
for site_dir in site_dirs:
    site_dir = site_dir.resolve()
    if site_dir in seen:
        continue
    seen.add(site_dir)
    if not site_dir.exists():
        continue
    pth_path = site_dir / "ra_ov3dseg_pointcept.pth"
    try:
        pth_path.write_text(f"{pointcept_dir}\n", encoding="utf-8")
    except OSError:
        continue
    print(f"[setup_env] wrote {pth_path}")
    break
else:
    raise RuntimeError("Could not write Pointcept .pth into the active Python environment.")
PY
fi

echo "[setup_env] installing RA-OV3DSeg extras..."
pip install -r requirements.txt

echo "[setup_env] OK. Run scripts/sanity_check.sh to verify."
