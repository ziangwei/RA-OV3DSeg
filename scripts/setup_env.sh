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
python -m pip install -r requirements-pointcept.txt

# Step 3: register Pointcept on sys.path via a .pth file
echo "[setup_env] registering Pointcept on sys.path..."
SITE_PACKAGES=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
echo "${POINTCEPT_DIR}" > "${SITE_PACKAGES}/pointcept.pth"
python -c "import pointcept; print('[setup_env] pointcept loaded from:', pointcept.__file__)"

if [ "${INSTALL_POINTOPS:-0}" = "1" ]; then
  if [ ! -f "${POINTCEPT_DIR}/libs/pointops/setup.py" ]; then
    echo "[setup_env] ERROR: ${POINTCEPT_DIR}/libs/pointops/setup.py not found" >&2
    exit 1
  fi
  echo "[setup_env] building Pointcept pointops extension..."
  cd "${POINTCEPT_DIR}/libs/pointops"
  python setup.py install
  cd "${PROJECT_ROOT}"
  python -c "import pointops; print('[setup_env] pointops loaded from:', pointops.__file__)"
else
  echo "[setup_env] pointops is not built by default."
  echo "[setup_env] On a CUDA build/compute node, run:"
  echo "[setup_env]   INSTALL_POINTOPS=1 bash scripts/setup_env.sh"
fi

echo "[setup_env] installing RA-OV3DSeg extras..."
python -m pip install -r requirements.txt

echo "[setup_env] OK. Run scripts/sanity_check.sh to verify."
