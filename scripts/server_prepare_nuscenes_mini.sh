#!/usr/bin/env bash

set -euo pipefail

DATAROOT="${1:-/data/sets/nuscenes}"
DOWNLOAD_DIR="${DATAROOT}/downloads"

echo "[INFO] dataroot: ${DATAROOT}"
mkdir -p "${DOWNLOAD_DIR}"

echo "[INFO] download dir: ${DOWNLOAD_DIR}"
cd "${DOWNLOAD_DIR}"

# 如果之前下载过不完整数据，可以先人工确认后删除。
# 示例：
#   rm -rf "${DATAROOT}/samples" \
#          "${DATAROOT}/sweeps" \
#          "${DATAROOT}/maps" \
#          "${DATAROOT}/lidarseg" \
#          "${DATAROOT}/v1.0-mini"

echo "[INFO] downloading nuScenes mini split..."
wget -c https://www.nuscenes.org/data/v1.0-mini.tgz

echo "[INFO] downloading nuScenes lidarseg mini labels..."
wget -c https://www.nuscenes.org/data/nuScenes-lidarseg-mini-v1.0.tar.bz2

echo "[INFO] extracting v1.0-mini.tgz ..."
tar -xf v1.0-mini.tgz -C "${DATAROOT}"

echo "[INFO] extracting nuScenes-lidarseg-mini-v1.0.tar.bz2 ..."
tar -xf nuScenes-lidarseg-mini-v1.0.tar.bz2 -C "${DATAROOT}"

echo "[INFO] done. expected layout:"
cat <<EOF
${DATAROOT}/
  samples/
  sweeps/
  maps/
  lidarseg/
    v1.0-mini/
  v1.0-mini/
EOF
