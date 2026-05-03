#!/usr/bin/env bash

set -euo pipefail

DATAROOT="${NUSCENES_ROOT:-/data/sets/nuscenes}"
DOWNLOAD_DIR=""
YES=0
REMOVE_DOWNLOADS=1

usage() {
  cat <<'EOF'
Safely remove nuScenes v1.0-trainval data from a server dataroot.

Default mode is dry-run. Add --yes to actually delete files.
This removes shared full-dataset folders samples/sweeps/maps as well as v1.0-trainval
and lidarseg/v1.0-trainval, so v1.0-mini in the same dataroot may no longer be runnable.

Usage:
  bash scripts/server_cleanup_nuscenes_trainval.sh \
    --dataroot /path/to/nuscenes \
    --download_dir /path/to/nuscenes/downloads_trainval \
    --yes

Options:
  --dataroot PATH       nuScenes root. Default: ${NUSCENES_ROOT:-/data/sets/nuscenes}
  --download_dir PATH   Archive staging directory. Default: DATAROOT/downloads_trainval
  --keep_downloads      Keep DOWNLOAD_DIR.
  --yes                 Actually delete. Without this, only prints targets.
  -h, --help            Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataroot)
      DATAROOT="$2"
      shift 2
      ;;
    --download_dir)
      DOWNLOAD_DIR="$2"
      shift 2
      ;;
    --keep_downloads)
      REMOVE_DOWNLOADS=0
      shift
      ;;
    --yes)
      YES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "${DOWNLOAD_DIR}" ]]; then
  DOWNLOAD_DIR="${DATAROOT}/downloads_trainval"
fi

if [[ -z "${DATAROOT}" || "${DATAROOT}" == "/" ]]; then
  echo "[ERROR] Refusing to operate on empty dataroot or /." >&2
  exit 1
fi

case "${DATAROOT}" in
  *nuscenes*|*nuScenes*) ;;
  *)
    echo "[ERROR] Refusing to delete because dataroot does not look like a nuScenes path: ${DATAROOT}" >&2
    echo "[ERROR] Rename/use a path containing 'nuscenes' or delete manually after checking." >&2
    exit 1
    ;;
esac

TARGETS=(
  "${DATAROOT}/v1.0-trainval"
  "${DATAROOT}/samples"
  "${DATAROOT}/sweeps"
  "${DATAROOT}/maps"
  "${DATAROOT}/lidarseg/v1.0-trainval"
)

if [[ "${REMOVE_DOWNLOADS}" == "1" ]]; then
  TARGETS+=("${DOWNLOAD_DIR}")
fi

echo "[INFO] cleanup targets:"
for target in "${TARGETS[@]}"; do
  if [[ -e "${target}" ]]; then
    du -sh "${target}" 2>/dev/null || true
    echo "  ${target}"
  else
    echo "  ${target} [missing]"
  fi
done

if [[ "${YES}" != "1" ]]; then
  echo "[DRY-RUN] No files deleted. Re-run with --yes to delete."
  exit 0
fi

echo "[WARN] deleting trainval data in 5 seconds. Press Ctrl+C to abort."
sleep 5

for target in "${TARGETS[@]}"; do
  if [[ -e "${target}" ]]; then
    echo "[INFO] removing ${target}"
    rm -rf "${target}"
  fi
done

echo "[INFO] cleanup done."
df -h "${DATAROOT}" || true
