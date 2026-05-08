#!/usr/bin/env bash

set -euo pipefail

DATAROOT="/dss/dssfs05/pn39qo/pn39qo-dss-0001/di97fer/projects_for_test/RA-OV3DSeg/data/nuscenes"
DOWNLOAD_DIR=""
FIRST_PART=1
NUM_PARTS=10
BLOB_TYPE="keyframes"
WITH_LIDARSEG=1
KEEP_ARCHIVES=0
KEEP_SWEEPS=0
KEEP_RADAR=0
SKIP_DOWNLOAD=0
MIN_FREE_GB=35
TAR_EXCLUDES=()

usage() {
  cat <<'EOF'
Prepare a compact nuScenes v1.0-trainval layout for RA-OV3DSeg.

This keeps only what the current project pipeline needs:
  - v1.0-trainval metadata
  - maps
  - samples/CAM_*
  - samples/LIDAR_TOP
  - lidarseg/v1.0-trainval labels

By default it deletes archives after extraction and removes sweeps/radar data after
each blob archive, which reduces peak and final disk usage.
By default it downloads keyframe blobs only, because the current pipeline uses
nuScenes annotated sample keyframes, not non-keyframe sweeps.

Usage:
  bash scripts/server_prepare_nuscenes_trainval_compact.sh \
    --dataroot /path/to/RA-OV3DSeg/data/nuscenes \
    --download_dir /path/to/RA-OV3DSeg/data/nuscenes/downloads_trainval

Options:
  --dataroot PATH       nuScenes root.
  --download_dir PATH   Archive staging directory. Default: DATAROOT/downloads_trainval
  --first_part N        First trainval blob index, 1-10. Default: 1
  --num_parts N         Number of trainval blob archives to process. Default: 10
  --blob_type TYPE      keyframes or full. Default: keyframes
  --no_lidarseg         Do not download/extract nuScenes-lidarseg.
  --keep_archives       Keep .tgz/.tar.bz2 archives after extraction.
  --keep_sweeps         Keep sweeps/. Not needed for current RA-OV3DSeg pipeline.
  --keep_radar          Keep samples/RADAR_* and sweeps/RADAR_*.
  --skip_download       Only extract existing archives from --download_dir.
  --min_free_gb GB      Warn if free disk drops below this after each step. Default: 35
  -h, --help            Show this help.

Official direct links may fail depending on nuScenes auth/session behavior. If that
happens, manually download the listed archives from https://www.nuscenes.org/download
into --download_dir, then rerun with --skip_download.
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
    --first_part)
      FIRST_PART="$2"
      shift 2
      ;;
    --num_parts)
      NUM_PARTS="$2"
      shift 2
      ;;
    --blob_type)
      BLOB_TYPE="$2"
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
    --keep_sweeps)
      KEEP_SWEEPS=1
      shift
      ;;
    --keep_radar)
      KEEP_RADAR=1
      shift
      ;;
    --skip_download)
      SKIP_DOWNLOAD=1
      shift
      ;;
    --min_free_gb)
      MIN_FREE_GB="$2"
      shift 2
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

if [[ "${FIRST_PART}" -lt 1 || "${FIRST_PART}" -gt 10 ]]; then
  echo "[ERROR] --first_part must be in [1, 10]." >&2
  exit 2
fi
LAST_PART=$((FIRST_PART + NUM_PARTS - 1))
if [[ "${LAST_PART}" -gt 10 ]]; then
  echo "[ERROR] first_part + num_parts - 1 must be <= 10." >&2
  exit 2
fi
if [[ "${BLOB_TYPE}" != "keyframes" && "${BLOB_TYPE}" != "full" ]]; then
  echo "[ERROR] --blob_type must be keyframes or full." >&2
  exit 2
fi

mkdir -p "${DATAROOT}" "${DOWNLOAD_DIR}"

echo "[INFO] dataroot=${DATAROOT}"
echo "[INFO] download_dir=${DOWNLOAD_DIR}"
echo "[INFO] parts=${FIRST_PART}-${LAST_PART}"
echo "[INFO] blob_type=${BLOB_TYPE}"
echo "[INFO] with_lidarseg=${WITH_LIDARSEG}"
echo "[INFO] keep_archives=${KEEP_ARCHIVES}"
echo "[INFO] keep_sweeps=${KEEP_SWEEPS}"
echo "[INFO] keep_radar=${KEEP_RADAR}"
echo "[INFO] skip_download=${SKIP_DOWNLOAD}"
df -h "${DATAROOT}" || true

BASE_URLS=()
if [[ -n "${NUSCENES_BASE_URL:-}" ]]; then
  BASE_URLS+=("${NUSCENES_BASE_URL}")
fi
BASE_URLS+=(
  "https://d36yt3mvayqw5m.cloudfront.net/public/v1.0"
  "https://motional-nuscenes.s3.amazonaws.com/public/v1.0"
  "https://www.nuscenes.org/data"
)

free_gb() {
  df -BG "${DATAROOT}" | awk 'NR==2 {gsub("G", "", $4); print $4}'
}

check_free_space() {
  local free
  free="$(free_gb || echo 0)"
  echo "[INFO] free_disk_gb=${free}"
  if [[ "${free}" -lt "${MIN_FREE_GB}" ]]; then
    echo "[WARN] Free disk below ${MIN_FREE_GB}GB. Consider stopping, deleting archives, or using fewer parts." >&2
  fi
}

validate_archive() {
  local file="$1"
  local archive="${DOWNLOAD_DIR}/${file}"
  local magic=""

  if [[ ! -s "${archive}" ]]; then
    echo "[ERROR] archive is missing or empty: ${archive}" >&2
    return 1
  fi

  case "${file}" in
    *.tgz|*.tar.gz)
      magic="$(head -c 2 "${archive}" | od -An -tx1 | tr -d ' \n')"
      if [[ "${magic}" != "1f8b" ]]; then
        echo "[WARN] downloaded file is not a gzip archive: ${archive}" >&2
        echo "[WARN] this usually means the URL returned an HTML page/login/download page." >&2
        echo "[WARN] first bytes:" >&2
        head -c 240 "${archive}" >&2 || true
        echo >&2
        return 1
      fi
      ;;
    *.tar.bz2)
      magic="$(head -c 3 "${archive}" | od -An -tc | tr -d ' \n')"
      if [[ "${magic}" != "BZh" ]]; then
        echo "[WARN] downloaded file is not a bzip2 archive: ${archive}" >&2
        echo "[WARN] first bytes:" >&2
        head -c 240 "${archive}" >&2 || true
        echo >&2
        return 1
      fi
      ;;
  esac
  return 0
}

validate_archive_or_die() {
  local file="$1"
  local archive="${DOWNLOAD_DIR}/${file}"
  if ! validate_archive "${file}"; then
    echo "[ERROR] invalid archive: ${archive}" >&2
    echo "[ERROR] If all automatic URLs fail, manually download ${file} into ${DOWNLOAD_DIR}, then rerun with --skip_download." >&2
    exit 1
  fi
}

download_one() {
  local file="$1"
  local archive="${DOWNLOAD_DIR}/${file}"
  local partial="${archive}.part"

  if [[ "${SKIP_DOWNLOAD}" == "1" ]]; then
    if [[ ! -s "${archive}" ]]; then
      echo "[ERROR] --skip_download set but archive is missing: ${archive}" >&2
      exit 1
    fi
    validate_archive_or_die "${file}"
    echo "[INFO] using existing archive: ${archive}"
    return
  fi

  if [[ -s "${archive}" ]]; then
    validate_archive_or_die "${file}"
    echo "[INFO] archive already exists: ${archive}"
    return
  fi

  for base_url in "${BASE_URLS[@]}"; do
    local url="${base_url%/}/${file}"
    echo "[INFO] downloading ${url}"
    if wget -c -O "${partial}" "${url}"; then
      mv "${partial}" "${archive}"
      if validate_archive "${file}"; then
        return
      fi
      echo "[WARN] invalid archive from ${url}; deleting and trying next mirror"
      rm -f "${archive}" "${partial}"
    fi
    echo "[WARN] failed: ${url}"
  done

  echo "[ERROR] Could not download ${file}." >&2
  echo "[ERROR] Download it manually from https://www.nuscenes.org/download into ${DOWNLOAD_DIR}, then rerun with --skip_download." >&2
  exit 1
}

build_tar_excludes() {
  TAR_EXCLUDES=()
  if [[ "${KEEP_SWEEPS}" != "1" ]]; then
    TAR_EXCLUDES+=("--exclude=sweeps")
    TAR_EXCLUDES+=("--exclude=sweeps/*")
  fi
  if [[ "${KEEP_RADAR}" != "1" ]]; then
    TAR_EXCLUDES+=("--exclude=samples/RADAR_*")
    TAR_EXCLUDES+=("--exclude=sweeps/RADAR_*")
  fi
}

remove_unneeded_for_ra_ov3dseg() {
  if [[ "${KEEP_SWEEPS}" != "1" && -d "${DATAROOT}/sweeps" ]]; then
    echo "[INFO] removing sweeps/ because current pipeline uses only keyframe samples"
    rm -rf "${DATAROOT}/sweeps"
  fi

  if [[ "${KEEP_RADAR}" != "1" ]]; then
    echo "[INFO] removing radar folders because current pipeline uses cameras + LIDAR_TOP"
    rm -rf "${DATAROOT}"/samples/RADAR_* 2>/dev/null || true
    rm -rf "${DATAROOT}"/sweeps/RADAR_* 2>/dev/null || true
  fi
}

extract_one() {
  local file="$1"
  local compact_cleanup="${2:-1}"
  local archive="${DOWNLOAD_DIR}/${file}"
  local marker="${DOWNLOAD_DIR}/.extracted_compact_${file}"

  if [[ -f "${marker}" ]]; then
    echo "[INFO] already extracted, skip: ${file}"
    return
  fi

  download_one "${file}"
  echo "[INFO] extracting ${archive} -> ${DATAROOT}"
  build_tar_excludes
  if [[ "${#TAR_EXCLUDES[@]}" -gt 0 ]]; then
    echo "[INFO] tar excludes: ${TAR_EXCLUDES[*]}"
  fi
  tar "${TAR_EXCLUDES[@]}" -xf "${archive}" -C "${DATAROOT}"
  touch "${marker}"

  if [[ "${compact_cleanup}" == "1" ]]; then
    remove_unneeded_for_ra_ov3dseg
  fi

  if [[ "${KEEP_ARCHIVES}" != "1" ]]; then
    echo "[INFO] deleting archive: ${archive}"
    rm -f "${archive}"
  fi

  du -sh "${DATAROOT}" 2>/dev/null || true
  check_free_space
}

extract_one "v1.0-trainval_meta.tgz" 0

for part in $(seq "${FIRST_PART}" "${LAST_PART}"); do
  if [[ "${BLOB_TYPE}" == "keyframes" ]]; then
    printf -v file "v1.0-trainval%02d_keyframes.tgz" "${part}"
  else
    printf -v file "v1.0-trainval%02d_blobs.tgz" "${part}"
  fi
  extract_one "${file}" 1
done

if [[ "${WITH_LIDARSEG}" == "1" ]]; then
  extract_one "nuScenes-lidarseg-all-v1.0.tar.bz2" 0
fi

echo "[INFO] final compact layout check:"
for target in \
  "${DATAROOT}/v1.0-trainval" \
  "${DATAROOT}/maps" \
  "${DATAROOT}/samples/CAM_FRONT" \
  "${DATAROOT}/samples/CAM_FRONT_LEFT" \
  "${DATAROOT}/samples/CAM_FRONT_RIGHT" \
  "${DATAROOT}/samples/CAM_BACK" \
  "${DATAROOT}/samples/CAM_BACK_LEFT" \
  "${DATAROOT}/samples/CAM_BACK_RIGHT" \
  "${DATAROOT}/samples/LIDAR_TOP" \
  "${DATAROOT}/lidarseg/v1.0-trainval"; do
  if [[ -e "${target}" ]]; then
    du -sh "${target}" 2>/dev/null || true
  else
    echo "[WARN] missing: ${target}"
  fi
done

echo "[INFO] done. Verify with:"
cat <<EOF
python scripts/check_nuscenes_sample.py \\
  --dataroot "${DATAROOT}" \\
  --version v1.0-trainval \\
  --sample_idx 0
EOF
