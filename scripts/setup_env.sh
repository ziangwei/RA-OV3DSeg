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

echo "[setup_env] pip install -e third_party/Pointcept --no-deps ..."
cd "${POINTCEPT_DIR}"
pip install -e . --no-deps
cd "${PROJECT_ROOT}"

echo "[setup_env] installing RA-OV3DSeg extras..."
pip install -r requirements.txt

echo "[setup_env] OK. Run scripts/sanity_check.sh to verify."
