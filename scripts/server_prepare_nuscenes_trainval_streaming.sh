#!/usr/bin/env bash

set -euo pipefail

DATAROOT="${NUSCENES_ROOT:-/data/sets/nuscenes}"
DOWNLOAD_DIR=""
WITH_LIDARSEG=1
KEEP_ARCHIVES=0
SKIP_DOWNLOAD=0

usage() {
  cat <<'EOF'
Prepare nuScenes v1.0-trainval with controlled peak disk usage.

This script downloads one archive, extracts it, then deletes the archive by default.
If official direct links fail, download the files manually from https://www.nuscenes.org/download
into --download_dir and rerun with --skip_download.

Usage:
  bash scripts/server_prepare_nuscenes_trainval_streaming.sh \
    --dataroot /path/to/nuscenes \
    --download_dir /path/to/nuscenes/downloads_trainval

Options:
  --dataroot PATH       nuScenes root. Default: ${NUSCENES_ROOT:-/data/sets/nuscenes}
  --download_dir PATH   Archive staging directory. Default: DATAROOT/downloads_trainval
  --no_lidarseg         Do not download/extract nuScenes-lidarseg.
  --keep_archives       Keep .tgz/.tar.bz2 archives after extraction.
  --skip_download       Only extract archives that already exist in --download_dir.
  -h, --help            Show this help.

Expected final layout:
  DATAROOT/
    maps/
    samples/
    sweeps/
    lidarseg/v1.0-trainval/
    v1.0-trainval/
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
    --no_lidarseg)
      WITH_LIDARSEG=0
      shift
      ;;
    --keep_archives)
      KEEP_ARCHIVES=1
      shift
      ;;
    --skip_download)
      SKIP_DOWNLOAD=1
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

mkdir -p "${DATAROOT}" "${DOWNLOAD_DIR}"

echo "[INFO] dataroot=${DATAROOT}"
echo "[INFO] download_dir=${DOWNLOAD_DIR}"
echo "[INFO] keep_archives=${KEEP_ARCHIVES}"
echo "[INFO] skip_download=${SKIP_DOWNLOAD}"
df -h "${DATAROOT}" || true

TRAINVAL_FILES=(
  "v1.0-trainval_meta.tgz"
  "v1.0-trainval01_blobs.tgz"
  "v1.0-trainval02_blobs.tgz"
  "v1.0-trainval03_blobs.tgz"
  "v1.0-trainval04_blobs.tgz"
  "v1.0-trainval05_blobs.tgz"
  "v1.0-trainval06_blobs.tgz"
  "v1.0-trainval07_blobs.tgz"
  "v1.0-trainval08_blobs.tgz"
  "v1.0-trainval09_blobs.tgz"
  "v1.0-trainval10_blobs.tgz"
)
LIDARSEG_FILE="nuScenes-lidarseg-all-v1.0.tar.bz2"

BASE_URLS=()
if [[ -n "${NUSCENES_BASE_URL:-}" ]]; then
  BASE_URLS+=("${NUSCENES_BASE_URL}")
fi
BASE_URLS+=(
  "https://www.nuscenes.org/data"
  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0"
  "https://motional-nuscenes.s3.amazonaws.com/public/v1.0"
)

download_one() {
  local file="$1"
  local archive="${DOWNLOAD_DIR}/${file}"
  local partial="${archive}.part"

  if [[ "${SKIP_DOWNLOAD}" == "1" ]]; then
    if [[ ! -s "${archive}" ]]; then
      echo "[ERROR] --skip_download set but archive is missing: ${archive}" >&2
      exit 1
    fi
    echo "[INFO] using existing archive: ${archive}"
    return
  fi

  if [[ -s "${archive}" ]]; then
    echo "[INFO] archive already exists: ${archive}"
    return
  fi

  for base_url in "${BASE_URLS[@]}"; do
    local url="${base_url%/}/${file}"
    echo "[INFO] downloading ${url}"
    if wget -c -O "${partial}" "${url}"; then
      mv "${partial}" "${archive}"
      return
    fi
    echo "[WARN] failed: ${url}"
  done

  echo "[ERROR] Could not download ${file}." >&2
  echo "[ERROR] Download it manually from https://www.nuscenes.org/download into ${DOWNLOAD_DIR}, then rerun with --skip_download." >&2
  exit 1
}

extract_one() {
  local file="$1"
  local archive="${DOWNLOAD_DIR}/${file}"
  local marker="${DOWNLOAD_DIR}/.extracted_${file}"

  if [[ -f "${marker}" ]]; then
    echo "[INFO] already extracted, skip: ${file}"
    return
  fi

  download_one "${file}"

  echo "[INFO] extracting ${archive} -> ${DATAROOT}"
  tar -xf "${archive}" -C "${DATAROOT}"
  touch "${marker}"

  if [[ "${KEEP_ARCHIVES}" != "1" ]]; then
    echo "[INFO] deleting archive to reduce peak disk: ${archive}"
    rm -f "${archive}"
  fi
  df -h "${DATAROOT}" || true
}

for file in "${TRAINVAL_FILES[@]}"; do
  extract_one "${file}"
done

if [[ "${WITH_LIDARSEG}" == "1" ]]; then
  extract_one "${LIDARSEG_FILE}"
fi

echo "[INFO] done. Verify with:"
cat <<EOF
python scripts/check_nuscenes_sample.py \\
  --dataroot "${DATAROOT}" \\
  --version v1.0-trainval \\
  --sample_idx 0
EOF
