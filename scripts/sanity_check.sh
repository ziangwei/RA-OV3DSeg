#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

python - <<'PY'
import torch
import pointcept
import ra_ov3dseg
from ra_ov3dseg.models.reliability import compute_point_reliability
from ra_ov3dseg.utils.run_conclusion import RunConclusion

print("[sanity] torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("[sanity] pointcept:", pointcept.__file__)
print("[sanity] ra_ov3dseg:", ra_ov3dseg.__version__)
print("[sanity] reliability import:", compute_point_reliability.__name__)
print("[sanity] RunConclusion import:", RunConclusion.__name__)
PY

python - <<'PY'
from pointcept.models.sparse_unet.spconv_unet_v1m1_base import SpUNetBase

m = SpUNetBase(in_channels=4, num_classes=16).eval()
print("[sanity] Pointcept SpUNet built:", sum(p.numel() for p in m.parameters()), "params")
PY

echo "[sanity] all checks passed"
