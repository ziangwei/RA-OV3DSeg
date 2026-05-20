# RA-OV3DSeg Execution Plan

Authored: 2026-05-12
Authoring assistant: Claude (Cowork)
Owner: Ziang

> **Status of this document**: Single-source execution plan for the project's
> migration off self-written backbones and into a Pointcept-backed,
> reliability-aware open-vocabulary 3D segmentation pipeline. This document is
> static after Phase 0. Operational decisions made during Stage 1-5 go into
> `docs/ROADMAP.md`. Experimental results go into `docs/EXPERIMENT_RECAP.md`.
> Narrative for interviews goes into `docs/INTERVIEW_PREP.md`. **Codex must
> not append to or modify this file** except for one-line status checkmarks in
> §11 (Execution Checklist).

---

## 0. How To Read This Document

This document is for two audiences:

1. **Ziang (the project owner)** uses it to brief Codex and to spot-check
   progress at the defined check-in points (§9).
2. **Codex** uses it as the operating manual. Codex starts by reading
   §2 (Working Rules), then executes Phase 0 (§3), then proceeds Stage by
   Stage (§4 through §8). At no point does Codex skip ahead.

Reading order for first execution:
- §1 (target), §2 (rules), §3 (Phase 0), §10 (document inventory).
- Then begin Phase 0 work.
- After Phase 0, read §4 only. Do not preview §5-§8 until §4 is complete.

Everything that says "TBD" or "fill in" is a slot Codex must populate during
execution. Everything that says "fixed" cannot be changed.

---

## 1. Project Target and End State

### 1.1 What the project is

`RA-OV3DSeg` = **Reliability-Aware Open-Vocabulary 3D Semantic Segmentation**
on outdoor LiDAR scenes (nuScenes-lidarseg).

The technical pipeline is:

```
LiDAR points
  -> 3D sparse-conv backbone (Pointcept SpUNet, pip-installed)
  -> point embeddings
  -> reliability-aware distillation from a dense open-vocabulary 2D teacher
     (SAM2 + SigLIP, projected via 6-camera LiDAR-to-image projection)
  -> at inference: cosine(point_embedding, text_embedding(class_name))
  -> arbitrary text-class 3D segmentation
```

### 1.2 What "done" looks like

The project is "done" (in the sense of being ready for portfolio / interview
walkthrough) when ALL of the following exist:

1. A `checkpoints/closed_set_baseline.pt` with reproducible val mIoU ≥ 0.70 on
   nuScenes-lidarseg trainval (Pointcept SpUNet under its own recipe).
2. A `checkpoints/ov_reliability_best.pt` trained with SAM2+SigLIP teacher and
   reliability-weighted distillation.
3. Two ablation tables (reliability threshold + reliability component) with at
   least one positive finding in each.
4. An OV-query demo script that takes a sample token + free-form text query and
   produces a colored point cloud heatmap.
5. A rewritten top-level `README.md` that an interviewer can read in 10 minutes.
6. A `docs/INTERVIEW_PREP.md` filled with decisions, results, anticipated
   questions, and honest limitations.

### 1.3 What this project is NOT trying to do

- Beat published SOTA on nuScenes-lidarseg closed-set mIoU.
- Demonstrate strong zero-shot novel-class mIoU. (Existing OV teachers are too
  weak for the lidarseg taxonomy; the project's contribution is reliability
  *filtering* under weak teachers, not pretending teachers are strong.)
- Validate on multiple datasets. nuScenes-lidarseg only.
- Train custom backbones. Pointcept SpUNet is the only backbone.

---

## 2. Codex Working Rules

These rules are duplicated into `AGENTS.md` for Codex to read on every session
start. The version in `AGENTS.md` is the live copy; this section is a snapshot.

### 2.1 Stage discipline

- The current stage is declared on the first line of `docs/ROADMAP.md`.
- Work that does not match the current stage's deliverables is forbidden.
- A stage is "complete" only when ALL of:
  - Its numeric gate (in this document, §4-§8) has been met.
  - A `RunConclusion` block was printed with `status=success` and
    `gate_passed=yes`.
  - `docs/EXPERIMENT_RECAP.md` ledger has been updated.
  - `docs/INTERVIEW_PREP.md` Decision Log and Headline Results have been
    updated.
  - File cleanup (§2.6) has been performed.

### 2.2 Document discipline

- Only three long-form documents are allowed in `docs/`:
  - `docs/ROADMAP.md` — forward planning, current stage, next experiment.
  - `docs/EXPERIMENT_RECAP.md` — append-only ledger of experiments.
  - `docs/INTERVIEW_PREP.md` — narrative, decisions, results, limitations.
- Codex MUST NOT create `*_PLAN.md`, `*_STATUS.md`, `*_NOTES.md`,
  `*_TODO.md`, or any new long-form planning file.
- Codex MUST NOT modify this `EXECUTION_PLAN.md` except for ticking checkboxes
  in §11.
- New ideas go to the appropriate section of `docs/ROADMAP.md`.
- Result rows go to `docs/EXPERIMENT_RECAP.md` (appended via
  `RunConclusion.append_to_recap()`).

### 2.3 Script naming

- Scripts use **semantic names**, not version numbers.
  - Good: `train_baseline.sh`, `extract_sam2_teacher.py`,
    `ablate_reliability_threshold.sh`.
  - Forbidden: `run_v18_*.sh`, `stage4_run.sh`, `experiment_42.py`.
- `scripts/` MUST NOT contain version-numbered scripts.

### 2.4 Third-party code

- `third_party/Pointcept/` is `.gitignore`d and managed by `scripts/setup_env.sh`.
- Codex MUST NOT modify any file under `third_party/Pointcept/`.
- Any customization or fix wraps Pointcept from within `ra_ov3dseg/`.
- The Pointcept commit hash is pinned in `scripts/setup_env.sh`. Bumping it
  requires explicit user approval and an entry in `docs/EXPERIMENT_RECAP.md`.

### 2.5 RunConclusion requirement

- Every training, evaluation, and extraction script MUST construct a
  `RunConclusion` (see `ra_ov3dseg/utils/run_conclusion.py`) and call
  `.print_block()` as its last action.
- The block is the last thing printed to stdout. Nothing after it.
- Failure cases (exceptions, gate failures, OOMs) also produce a
  `RunConclusion` with `status` set appropriately.

### 2.6 Cleanup at stage completion

When a stage is marked complete, Codex MUST:

- Delete any scripts under `scripts/` that were used only for intermediate
  debugging during this stage and are not referenced by `docs/ROADMAP.md` or
  `README.md`.
- Delete any `outputs/scratch/*` artifacts that are not the final stage
  deliverables.
- Run `git status` and confirm the working tree contains only intentional
  files. Stray `__pycache__`, `.ipynb_checkpoints`, `.DS_Store` must be
  removed and `.gitignore`d if recurring.

### 2.7 Stop conditions

- Each stage has a defined stop condition in §4-§8.
- When a stop condition triggers, Codex MUST stop further training and add a
  "Stage Retrospective" subsection to `docs/ROADMAP.md` for that stage. The
  user must review the retrospective before any continuation.
- Gate numbers cannot be lowered. Stop conditions cannot be redefined.

### 2.8 Forbidden actions

- Creating new top-level planning docs.
- Modifying `EXECUTION_PLAN.md` outside §11.
- Modifying anything under `third_party/`.
- Adding version numbers (`v18`, `v19`, ...) to file names.
- Adding a second 3D backbone alongside Pointcept SpUNet.
- Skipping ablation tables in Stage 4.
- Claiming a stage is complete before all five conditions in §2.1 are met.

---

## 3. Phase 0 — Foundation Reset

Time budget: 0.5 day.

Phase 0 ends with a clean repo, Pointcept installed, sanity check passing.

### 3.1 Freeze the current state (10 minutes)

```bash
cd D:\project\codex\RA-OV3DSeg
git status                          # confirm clean working tree first
git add -A && git commit -m "snapshot: pre-cleanup state" || true
git tag -a prototype-v1-archive -m "Frozen 1-week prototype before migration"
git push origin prototype-v1-archive  # if remote exists
```

Any time later, `git checkout prototype-v1-archive` recovers the pre-migration
state. This tag is the safety net.

### 3.2 Open the migration branch

```bash
git checkout -b migration/foundation-reset
```

All Phase 0 work happens on this branch. It merges to `main` only after
sanity_check passes.

### 3.3 Aggressive deletion

Codex executes the following `git rm`s. Each line corresponds to noise from
the prototype phase that will be replaced or is no longer needed.

```bash
# Superseded planning documents (kept in git history via the tag)
git rm RA_OV3DSeg_V16_V18_PLAN.md
git rm RA_OV3DSeg_ACTION_PLAN.md
git rm PROJECT_STATUS_FOR_REVIEW.md

# Self-written backbones (replaced by Pointcept)
git rm ra_ov3dseg/models/sparse_unet_spconv.py
git rm ra_ov3dseg/models/spconv_resunet.py
git rm ra_ov3dseg/models/cylinder_spconv_unet.py
git rm ra_ov3dseg/models/cylindrical_voxelization.py
git rm ra_ov3dseg/models/voxelization.py
git rm ra_ov3dseg/models/point_mlp.py
git rm ra_ov3dseg/models/pointcept_spunet_adapter.py

# Vendored Pointcept SpUNet (replaced by pip-installed Pointcept)
git rm -r third_party/pointcept_spunet

# Deprecated diagnostic and version-numbered scripts
git rm scripts/check_voxelization.py
git rm scripts/dry_run_training_step.py
git rm scripts/pre_v16_sanity_check.py
git rm scripts/verify_mvp_outputs.py
git rm scripts/check_teacher_backend.py
git rm scripts/check_nuscenes_sample.py
git rm scripts/check_nuscenes_sample_light.py
git rm scripts/train_point_mlp.py
git rm scripts/train_3d_segmentor.py            # to be replaced by train_baseline.py
git rm scripts/predict_3d_segmentor.py          # to be replaced by predict.py
git rm scripts/run_v9_trainval_experiment.sh
git rm scripts/run_v10_open_vocab_eval.sh
git rm scripts/run_v11_text_aligned_training.sh
git rm scripts/run_v12_groupvit_teacher_training.sh
git rm scripts/run_v13_diagnostics.sh
git rm scripts/run_v14_supervised_resunet.sh
git rm scripts/run_v15_cylinder_baseline.sh
git rm scripts/run_v16_precheck.sh
git rm scripts/run_v16a_official16_cylinder.sh
git rm scripts/run_v17_pointcept_spunet.sh
git rm scripts/run_mini_experiment.py
git rm scripts/print_experiment_summary.py      # replaced by RunConclusion utility
git rm scripts/server_prepare_nuscenes_mini.sh
git rm scripts/server_prepare_nuscenes_trainval_streaming.sh
git rm scripts/server_prepare_nuscenes_trainval_compact.sh
git rm scripts/server_cleanup_nuscenes_trainval.sh

# Stale docs (will be rewritten in 3.10)
git rm docs/ROADMAP.md
git rm docs/EXPERIMENT_RECAP.md
git rm docs/RUNNING_EXPERIMENTS.md

# Stale README (rewritten at Stage 5)
git rm README.md
```

The following files survive Phase 0. Codex MUST verify each still exists
after the deletes; missing files indicate the deletion list was wrong and
must be reported, not silently accepted.

```
ra_ov3dseg/__init__.py
ra_ov3dseg/datasets/__init__.py
ra_ov3dseg/datasets/nuscenes_mini_dataset.py
ra_ov3dseg/geometry/__init__.py
ra_ov3dseg/geometry/projection.py
ra_ov3dseg/geometry/transforms.py
ra_ov3dseg/models/__init__.py
ra_ov3dseg/models/reliability.py
ra_ov3dseg/models/text_encoder.py
ra_ov3dseg/models/image_encoder.py
ra_ov3dseg/models/point_feature_assigner.py
ra_ov3dseg/models/clipseg_dense_teacher.py
ra_ov3dseg/models/groupvit_dense_teacher.py
ra_ov3dseg/models/teacher_registry.py
ra_ov3dseg/models/segmentor_factory.py          (will be simplified, not deleted)
ra_ov3dseg/evaluation/__init__.py
ra_ov3dseg/evaluation/metrics.py
ra_ov3dseg/evaluation/openvocab_eval.py
ra_ov3dseg/utils/__init__.py
ra_ov3dseg/utils/config.py
ra_ov3dseg/utils/io.py
ra_ov3dseg/utils/logger.py
ra_ov3dseg/visualization/__init__.py
ra_ov3dseg/visualization/visualize_points.py
ra_ov3dseg/visualization/visualize_projection.py
ra_ov3dseg/training/                            (entire directory, will be reorganized)
scripts/project_lidar_to_cameras.py
scripts/visualize_projection.py
scripts/compute_reliability.py
scripts/extract_2d_features.py
scripts/extract_dense_teacher_logits.py
scripts/assign_2d_features_to_points.py
scripts/assign_dense_logits_to_points.py
scripts/eval_dense_teacher_pseudo_labels.py
scripts/eval_lidarseg.py
scripts/compute_class_frequencies.py
scripts/predict_3d_open_vocab.py
scripts/zero_shot_eval.py
configs/                                        (the whole directory)
```

Cleanup `__pycache__`:

```bash
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete
```

Commit:

```bash
git add -A
git commit -m "cleanup: archive prototype-phase backbones, scripts, and planning docs"
```

### 3.4 New repository layout (target state)

After Phase 0, the repo looks like this:

```
RA-OV3DSeg/
├── .gitignore                                  # includes third_party/Pointcept/
├── AGENTS.md                                   # Codex working rules
├── EXECUTION_PLAN.md                           # this file (static)
├── README.md                                   # placeholder until Stage 5
├── requirements.txt
├── requirements-pointcept.txt                  # torch/spconv pins for Pointcept
├── configs/                                    # nuscenes class names, splits, prompts
├── docs/
│   ├── ROADMAP.md                              # forward-looking, current stage
│   ├── EXPERIMENT_RECAP.md                     # backward-looking ledger
│   └── INTERVIEW_PREP.md                       # narrative, decisions, Q&A
├── ra_ov3dseg/
│   ├── __init__.py
│   ├── datasets/                               # nuScenes IO
│   ├── geometry/                               # LiDAR-camera projection
│   ├── models/
│   │   ├── reliability.py                      # core contribution
│   │   ├── text_encoder.py
│   │   ├── clipseg_dense_teacher.py            # baseline teachers
│   │   ├── groupvit_dense_teacher.py
│   │   ├── sam2_siglip_teacher.py              # added in Stage 3
│   │   ├── pointcept_backbone.py               # added in Stage 1 (thin import wrapper)
│   │   └── ov_head.py                          # added in Stage 2
│   ├── training/                               # losses, augmentations, trainer entry
│   ├── evaluation/
│   ├── visualization/
│   └── utils/
│       └── run_conclusion.py                   # added in Phase 0
├── scripts/
│   ├── setup_env.sh                            # Pointcept install
│   ├── sanity_check.sh                         # 30-second smoke test
│   ├── train_baseline.sh                       # Stage 1
│   ├── train_ov_head.sh                        # Stage 2
│   ├── extract_sam2_teacher.sh                 # Stage 3
│   ├── ablate_reliability.sh                   # Stage 4
│   ├── demo_ov_query.sh                        # Stage 5
│   ├── project_lidar_to_cameras.py             # carry-overs from prototype
│   ├── compute_reliability.py
│   ├── eval_dense_teacher_pseudo_labels.py
│   ├── eval_lidarseg.py
│   └── ... (other carry-overs, all semantically named)
├── third_party/
│   ├── .gitkeep                                # so the directory is tracked
│   └── Pointcept/                              # gitignored; cloned by setup_env.sh
└── outputs/                                    # gitignored; experiment artifacts
    ├── checkpoints/
    ├── logs/
    ├── results/
    └── scratch/
```

### 3.5 `.gitignore` content

Codex writes the following to `.gitignore` (replacing any prior contents):

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.egg-info/
.pytest_cache/
.ipynb_checkpoints/
.coverage

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Project artifacts
outputs/
checkpoints/
*.pth
*.pt

# Pointcept lives inside the repo but is not part of our git history
third_party/Pointcept/

# Data
data/
*.npz
*.npy
*.ply
*.pcd
```

### 3.6 `scripts/setup_env.sh`

This script provisions the environment. It is idempotent — running twice is
safe.

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POINTCEPT_DIR="${PROJECT_ROOT}/third_party/Pointcept"
POINTCEPT_COMMIT="d74c646db6abec569d0f23e0c34e7ddfce142789"
POINTCEPT_REPO="https://github.com/Pointcept/Pointcept.git"

echo "[setup_env] project_root=${PROJECT_ROOT}"
echo "[setup_env] pointcept_commit=${POINTCEPT_COMMIT}"

# Step 1: clone Pointcept inside the repo (gitignored)
mkdir -p "${PROJECT_ROOT}/third_party"
if [ ! -d "${POINTCEPT_DIR}/.git" ]; then
  echo "[setup_env] cloning Pointcept..."
  git clone "${POINTCEPT_REPO}" "${POINTCEPT_DIR}"
fi
cd "${POINTCEPT_DIR}"
git fetch
git checkout "${POINTCEPT_COMMIT}"

# Step 2: install Pointcept's required base deps FIRST (torch, spconv, etc.)
cd "${PROJECT_ROOT}"
echo "[setup_env] installing Pointcept base requirements..."
pip install -r requirements-pointcept.txt

# Step 3: install Pointcept itself as editable package
echo "[setup_env] pip install -e third_party/Pointcept ..."
cd "${POINTCEPT_DIR}"
pip install -e . --no-deps   # avoid pulling fresh torch/spconv versions
cd "${PROJECT_ROOT}"

# Step 4: install RA-OV3DSeg extras (transformers, sam2, etc.)
echo "[setup_env] installing ra-ov3dseg extras..."
pip install -r requirements.txt

echo "[setup_env] OK. Run scripts/sanity_check.sh to verify."
```

Codex creates this file and makes it executable. The Pointcept commit hash
is fixed in this document and cannot be changed without user approval.

### 3.7 `requirements-pointcept.txt`

Pinned versions matching Pointcept's nuScenes recipe. Codex creates this file
with content like (Codex must consult Pointcept's `install.md` or
`requirements.txt` from the pinned commit and copy the exact versions):

```
# Generated to match Pointcept commit d74c646db6abec569d0f23e0c34e7ddfce142789
torch==2.1.0
torchvision==0.16.0
spconv-cu120==2.3.6
numpy<2.0
scipy
addict
einops
ftfy
regex
plyfile
SharedArray
tensorboardx
yapf
termcolor
timm
# (verify against third_party/Pointcept/requirements.txt after cloning)
```

After Pointcept is cloned in §3.6, Codex must verify these pins against
`third_party/Pointcept/requirements.txt` and reconcile any differences.

### 3.8 `requirements.txt`

```
# RA-OV3DSeg extras (assumes Pointcept env already set up)
nuscenes-devkit==1.1.11
transformers==4.46.0
open-clip-torch==2.24.0
ftfy
pyyaml
matplotlib
tqdm
Pillow
# Stage 3 will add: segment-anything-2, etc. (add via append, do not regenerate)
```

### 3.9 `scripts/sanity_check.sh`

A 30-second smoke test that verifies the entire stack is wired correctly.

```bash
#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

python -c "
import torch
import pointcept
import ra_ov3dseg
from ra_ov3dseg.models.reliability import compute_reliability_weight  # adjust to real export
from ra_ov3dseg.utils.run_conclusion import RunConclusion
print('[sanity] torch:', torch.__version__, 'cuda:', torch.cuda.is_available())
print('[sanity] pointcept:', pointcept.__file__)
print('[sanity] ra_ov3dseg: OK')
print('[sanity] RunConclusion import: OK')
"

# Forward pass smoke test on Pointcept SpUNet
python -c "
import torch
from pointcept.models.sparse_unet import SpUNetBase  # adjust to actual import path
m = SpUNetBase(in_channels=4, num_classes=16).eval()
print('[sanity] Pointcept SpUNet built: ', sum(p.numel() for p in m.parameters()), 'params')
"

echo "[sanity] all checks passed"
```

Codex must adjust import paths if the actual Pointcept module layout differs
from what's written above. The point of this script is to fail loudly when
the environment is broken.

### 3.10 `ra_ov3dseg/utils/run_conclusion.py`

Codex creates this file verbatim (modulo minor style):

```python
"""Standardized run conclusion block.

Every training, evaluation, and extraction script MUST emit a RunConclusion
block as its last action. The block is parseable (one key per line) and
human-readable.

Codex must not invent additional fields. Use `notes` for one-line caveats.
"""

from __future__ import annotations

import dataclasses
import datetime
import json
from pathlib import Path
from typing import Literal

Status = Literal["success", "failed", "stopped_by_gate", "crashed"]


@dataclasses.dataclass
class RunConclusion:
    stage: str                                 # "stage-baseline", "stage-ov-head", ...
    experiment: str                            # semantic name, no version numbers
    status: Status
    gate: str                                  # human description
    gate_passed: bool
    primary_metric_name: str
    primary_metric_value: float
    secondary: dict[str, float]                # extra metrics, may be empty
    runtime_seconds: float
    checkpoint: str | None
    artifacts: list[str]
    next_step: str                             # one sentence
    notes: str = "-"

    def print_block(self) -> None:
        lines = [
            "========== RUN_CONCLUSION ==========",
            f"stage:             {self.stage}",
            f"experiment:        {self.experiment}",
            f"status:            {self.status}",
            f"gate:              {self.gate}",
            f"gate_passed:       {'yes' if self.gate_passed else 'no'}",
            "result:",
            f"  primary_metric:  {self.primary_metric_name} = {self.primary_metric_value:.4f}",
        ]
        if self.secondary:
            sec = ", ".join(f"{k}={v:.4f}" for k, v in self.secondary.items())
            lines.append(f"  secondary:       {sec}")
        else:
            lines.append("  secondary:       -")
        lines.append(f"runtime:           {self._format_runtime()}")
        lines.append(f"checkpoint:        {self.checkpoint or '-'}")
        if self.artifacts:
            lines.append(f"artifacts:         {self.artifacts[0]}")
            for art in self.artifacts[1:]:
                lines.append(f"                   {art}")
        else:
            lines.append("artifacts:         -")
        lines.append(f"next_step:         {self.next_step}")
        lines.append(f"notes:             {self.notes}")
        lines.append("====================================")
        for line in lines:
            print(line)

    def _format_runtime(self) -> str:
        secs = int(self.runtime_seconds)
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        return f"{h}h {m:02d}m {s:02d}s"

    def append_to_recap(self, recap_path: Path) -> None:
        """Append one row to docs/EXPERIMENT_RECAP.md's ledger table.

        The ledger row format is:
        | <date> | <stage> | <experiment> | <status> | <primary>=<value> | <notes> |
        """
        row = " | ".join([
            datetime.date.today().isoformat(),
            self.stage,
            self.experiment,
            self.status,
            f"{self.primary_metric_name}={self.primary_metric_value:.4f}",
            (self.notes or "-").replace("\n", " ")[:80],
        ])
        recap_path = Path(recap_path)
        with recap_path.open("a", encoding="utf-8") as f:
            f.write(f"| {row} |\n")

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2)
```

### 3.11 Document reset

Codex creates the three docs with the following starting content. Templates
are in Appendix A.

```bash
# docs/ROADMAP.md         from Appendix A.1
# docs/EXPERIMENT_RECAP.md from Appendix A.2
# docs/INTERVIEW_PREP.md   from Appendix A.3
# AGENTS.md               from Appendix A.4
```

### 3.12 Placeholder `README.md`

Until Stage 5, `README.md` is minimal:

```markdown
# RA-OV3DSeg

Reliability-Aware Open-Vocabulary 3D Semantic Segmentation on nuScenes-lidarseg.

This project is currently under active development. See:

- `docs/ROADMAP.md` for the current stage and plan.
- `docs/EXPERIMENT_RECAP.md` for experimental results.
- `EXECUTION_PLAN.md` for the multi-stage execution plan.

## Setup

```
bash scripts/setup_env.sh
bash scripts/sanity_check.sh
```
```

### 3.13 Phase 0 acceptance criteria

Phase 0 is complete when ALL of the following hold:

- [ ] `prototype-v1-archive` tag exists and is pushed.
- [ ] `migration/foundation-reset` branch contains all Phase 0 commits.
- [ ] Repository layout matches §3.4.
- [ ] `bash scripts/setup_env.sh` completes without error in a clean conda env.
- [ ] `bash scripts/sanity_check.sh` prints "[sanity] all checks passed".
- [ ] `docs/ROADMAP.md`, `docs/EXPERIMENT_RECAP.md`, `docs/INTERVIEW_PREP.md`,
      `AGENTS.md` exist with template content.
- [ ] `ra_ov3dseg/utils/run_conclusion.py` exists and is importable.
- [ ] `git status` is clean.

Once all checked, merge to `main`:

```bash
git checkout main
git merge --no-ff migration/foundation-reset
git tag -a phase0-complete -m "Foundation reset done; Pointcept env up; sanity OK"
```

Then proceed to Stage 1.

---

## 4. Stage 1 — Closed-Set Baseline (Pointcept SpUNet)

Time budget: 2-3 days.
Branch: `stage-1-baseline`.

### 4.1 Goal

Reproduce Pointcept SpUNet's nuScenes-lidarseg val mIoU using Pointcept's
**own** training config, with no modifications. This calibrates the
environment, data pipeline, and evaluation code against a known-good
reference.

### 4.2 Steps

1. Branch from `main`: `git checkout -b stage-1-baseline`.
2. Update `docs/ROADMAP.md` to declare current stage = `stage-baseline`.
3. Find Pointcept's nuScenes SpUNet config at
   `third_party/Pointcept/configs/nuscenes/semseg-spunet-v1m1-0-base.py` (path
   may vary; Codex verifies). Do NOT modify this config.
4. Write a thin wrapper `scripts/train_baseline.sh` that calls Pointcept's
   official launcher with this config. The wrapper's only added value is:
   - Computes a runtime, captures GPU info, redirects logs to `outputs/logs/`.
   - At the end, parses Pointcept's best val mIoU from its log and constructs
     a `RunConclusion` block.
5. Run on full nuScenes-lidarseg trainval, 50 epochs (or whatever Pointcept's
   default is — do not deviate). The training is what Pointcept's recipe
   says, not what we invent.
6. Capture the final checkpoint to `outputs/checkpoints/closed_set_baseline.pt`.
7. Run `scripts/eval_lidarseg.py` (carry-over) on the produced checkpoint to
   independently verify mIoU using our evaluation code. Both numbers (Pointcept
   internal eval, our eval) should match within ±1 mIoU. Significant divergence
   indicates an eval bug.

### 4.3 Gate

**Primary metric**: full nuScenes-lidarseg trainval val mIoU.
**Gate**: `≥ 0.70`.

### 4.4 Stop condition

If val mIoU < 0.65 after a full training run, STOP. Do not retry with random
modifications. Likely causes (in priority order):

1. Data path / nuScenes version mismatch.
2. Pointcept's class-mapping vs our `configs/nuscenes_lidarseg_class_names.txt`
   disagree.
3. spconv / CUDA mismatch causing silent numerical issues.
4. Pointcept commit pinned at a regressed point.

Codex must write a "Stage Retrospective" subsection to `docs/ROADMAP.md`
identifying the root cause before proceeding.

### 4.5 Deliverables

- `outputs/checkpoints/closed_set_baseline.pt`
- `outputs/logs/train_baseline_<timestamp>.log` containing the
  `RunConclusion` block.
- A new row in `docs/EXPERIMENT_RECAP.md`'s ledger.
- `docs/INTERVIEW_PREP.md` Decision Log entry: "3D backbone choice".
- `docs/INTERVIEW_PREP.md` Headline Results row updated.

### 4.6 INTERVIEW_PREP updates required

Decision Log row:
- Decision: "Use Pointcept SpUNet via pip-installed editable package"
- Chose: "Pointcept SpUNet v1m1"
- Rejected: "Self-written Cylinder3D-style (0.42 mIoU at 1024 samples),
  vendored partial SpUNet (V17 adapter contract drift)"
- Why: "Mature published recipe; single-env via pip -e; reproducible 70+ mIoU."

Headline Results: fill in actual closed-set val mIoU.

Anticipated Q: "Why not implement your own backbone?" Answer: prototype-phase
showed adapter / recipe complexity dominates; single-env pip-install is
simpler and reaches published numbers. **Codex writes this answer in concrete
terms based on actual numbers from this stage.**

### 4.7 Cleanup at stage completion

- Remove any intermediate `outputs/scratch/baseline_*` directories.
- Confirm `outputs/checkpoints/closed_set_baseline.pt` is the only artifact
  kept from this stage.
- Merge `stage-1-baseline` to `main`, tag `stage1-baseline-complete`.

---

## 5. Stage 2 — OV Head Replacement

Time budget: 2 days.
Branch: `stage-2-ov-head`.

### 5.1 Goal

Replace the closed-set classification head with a text-embedding cosine
similarity head. Verify that text-aligned point embeddings can match the
closed-set baseline within a small drop, **without any new teacher signal
yet**. This isolates the head-swap effect from any distillation effect.

### 5.2 Steps

1. Load `outputs/checkpoints/closed_set_baseline.pt`.
2. Encode the 16 official nuScenes-lidarseg class names with SigLIP
   (`google/siglip-base-patch16-224`) using prompt template
   `"a photo of a {class}"`. Cache the resulting `[16, embed_dim]` matrix.
3. Implement `ra_ov3dseg/models/ov_head.py`:
   - Wraps the Pointcept backbone, exposes `point_embeddings` of dimension
     `embed_dim` (insert an optional projection layer if backbone output
     differs from SigLIP embed dim).
   - Forward returns `logits = cosine(point_emb, text_prototype) / temperature`.
4. Short fine-tune: 5-10 epochs, **backbone frozen**, only the projection
   layer (and learnable temperature) are trainable. CE loss on official 16
   classes with `ignore_index` for noise.
5. Evaluate on full val split with `scripts/eval_lidarseg.py`.
6. Save `outputs/checkpoints/ov_head_aligned.pt`.

### 5.3 Gate

**Primary metric**: val mIoU drop relative to Stage 1.
**Gate**: drop ≤ 0.08 (i.e., if Stage 1 = 0.72, Stage 2 must ≥ 0.64).

### 5.4 Stop condition

If drop > 0.12, STOP. The head replacement is not working as expected. Likely
causes:

1. Projection layer too small / over-regularized.
2. Wrong prompt template (try variants).
3. SigLIP vs CLIP space mismatch; try CLIP ViT-L/14.

Document in Stage Retrospective before proceeding.

### 5.5 Deliverables

- `outputs/checkpoints/ov_head_aligned.pt`
- `outputs/results/stage2_head_swap.md`: closed-set mIoU before/after, per-class.
- EXPERIMENT_RECAP row.
- INTERVIEW_PREP Decision Log entry: "OV head architecture".

### 5.6 Cleanup

- Remove old text-encoding cache if a different model is finally chosen.
- Merge, tag `stage2-ov-head-complete`.

---

## 6. Stage 3 — Stronger Teacher (SAM2 + SigLIP)

Time budget: 3-4 days.
Branch: `stage-3-teacher`.

### 6.1 Goal

Build a 2D dense open-vocabulary teacher stronger than GroupViT
(prototype's projected mIoU = 0.0195) using mask-then-classify with SAM2 +
SigLIP. Cache projected per-point pseudo-labels for use in Stage 4.

### 6.2 Steps

1. Install SAM2:
   - Append `segment-anything-2` to `requirements.txt`.
   - Verify import. If installation in the existing env fails, fall back to
     `facebook/sam-vit-huge` (SAM v1) and note the fallback in INTERVIEW_PREP.
2. Implement `ra_ov3dseg/models/sam2_siglip_teacher.py`:
   - SAM2 automatic mask generation for one image.
   - For each mask: crop bbox with 10% padding, fill outside-mask with image mean,
     encode with SigLIP, cosine vs pre-encoded 16-class + "background" text
     prototypes, assign argmax class.
   - Overlap resolution: smallest containing mask wins.
3. Write `scripts/extract_sam2_teacher.py`:
   - Processes 6 cameras per sample.
   - Caches per-camera dense class maps and confidence to
     `outputs/teacher_caches/sam2_siglip/<sample_token>/CAM_*.npz`.
4. Reuse `scripts/assign_dense_logits_to_points.py` (carry-over) to project
   per-camera class maps to 3D points using existing projection caches.
5. Evaluate teacher quality:
   - Reuse `scripts/eval_dense_teacher_pseudo_labels.py` (carry-over) on
     the same diagnostic split used for GroupViT (~128 samples) to allow
     direct comparison.
6. Update teacher comparison table in `docs/INTERVIEW_PREP.md`.

### 6.3 Gate

**Primary metric**: SAM2+SigLIP projected pseudo-label all-class mIoU on the
diagnostic split.
**Gate**: `≥ 0.10` (5× GroupViT's 0.0195).

### 6.4 Stop condition (and project pivot)

If after reasonable iteration (different prompt templates, different SigLIP
sizes, SAM2 parameter tuning) the teacher still cannot beat 0.05:

- STOP teacher work.
- Update `docs/ROADMAP.md` with a "Pivot" section declaring the project's
  framing changes from:
  > "Reliability-Aware Open-Vocabulary 3D Semantic Segmentation"

  to:
  > "Reliability-Aware Filtering for 2D-to-3D Distillation under Weak
  > Open-Vocabulary Teachers"

- INTERVIEW_PREP Elevator Pitch is rewritten.
- Stage 4 proceeds anyway: reliability filtering on weak teachers is still
  the main contribution, just with more honest framing.

### 6.5 Deliverables

- `outputs/teacher_caches/sam2_siglip/` cached pseudo-labels.
- `outputs/results/teacher_comparison.md` with 4 rows:
  CLIP patch / CLIPSeg / GroupViT / SAM2+SigLIP.
- EXPERIMENT_RECAP row.
- INTERVIEW_PREP teacher comparison table updated, with discussion of why
  SAM2+SigLIP is stronger (or isn't).

### 6.6 Cleanup

- Optionally keep CLIPSeg/GroupViT teacher caches small enough for the
  comparison table; otherwise delete after recording numbers in
  EXPERIMENT_RECAP.
- Merge, tag `stage3-teacher-complete`.

---

## 7. Stage 4 — Reliability-Aware Distillation (Core Experiment)

Time budget: 4-5 days.
Branch: `stage-4-reliability`.

### 7.1 Goal

Train an OV student that uses the Stage 2 text-aligned head AND the Stage 3
teacher, with reliability-weighted KL distillation. Produce two ablation
tables that constitute the project's main empirical contribution.

### 7.2 Steps

1. From `outputs/checkpoints/ov_head_aligned.pt`, initialize. Backbone
   unfrozen this time; head warm-started from Stage 2.
2. Loss = `λ_ce * CE(student_logits, gt_labels) + λ_kl * reliability_weight * KL(student_logits, teacher_logits)`.
3. Reliability weight: `w_point = w_distance * w_geometric * w_semantic`,
   per the existing `ra_ov3dseg/models/reliability.py`.
4. **Threshold ablation** (5 runs, 20 epochs each):
   thresholds = [0.0, 0.3, 0.5, 0.7, 0.9].
   Points below threshold get zero distillation weight.
5. **Component ablation** (5 runs, 20 epochs each, using best threshold from
   step 4):
   - full (all three components)
   - no distance
   - no geometric
   - no semantic
   - uniform (all weights = 1)
6. Each run produces a `RunConclusion` and appends to ledger.
7. Aggregate into two tables in `outputs/results/reliability_ablation.md`
   and `outputs/results/reliability_components.md`.
8. Generate two plots:
   - `outputs/plots/reliability_threshold_sweep.png`
   - `outputs/plots/reliability_components.png`
   matplotlib only, no seaborn.

### 7.3 Gates

Two gates, both must pass:

**Gate A (threshold)**: at least one non-zero threshold strictly outperforms
threshold=0 by ≥ 0.005 mIoU. (Otherwise reliability filtering has no effect.)

**Gate B (components)**: removing at least one reliability component causes
≥ 0.005 mIoU drop vs full. (Otherwise the multiplicative form is unjustified.)

### 7.4 Stop condition

If both gates fail, reliability filtering is not empirically supported.
This is itself a valid scientific result and must be honestly reported:

- INTERVIEW_PREP gets a new Limitation: "Our reliability weighting did not
  produce measurable gains in this experimental setup."
- The project's framing pivots to "characterization of weak teacher OV
  distillation, with negative findings on reliability filtering."

This pivot is honest and defensible, but it's an absorbed loss. The Stage 4
work still produces the deliverables; only the narrative changes.

### 7.5 Deliverables

- `outputs/checkpoints/ov_reliability_best.pt` (best configuration).
- `outputs/results/reliability_ablation.md`
- `outputs/results/reliability_components.md`
- `outputs/plots/reliability_threshold_sweep.png`
- `outputs/plots/reliability_components.png`
- EXPERIMENT_RECAP rows (10 total, one per ablation run).
- INTERVIEW_PREP: Decision Log, Headline Results, Anticipated Q&A all updated.

### 7.6 Cleanup

- Keep only the best ablation checkpoint; delete the other 9.
- Merge, tag `stage4-reliability-complete`.

---

## 8. Stage 5 — OV-Query Demo and Final README

Time budget: 2-3 days.
Branch: `stage-5-demo`.

### 8.1 Goal

Build an interview-ready demo where any natural-language query produces a
3D point cloud heatmap. Write the final README that pulls everything
together.

### 8.2 Steps

1. Curate `configs/ov_queries.yaml` with 30-50 queries that go beyond
   lidarseg taxonomy:
   - "a child on the road"
   - "police car"
   - "parked truck"
   - "wet road surface"
   - "construction worker"
   - "bicycle rack"
   - ... (50 total)
2. For 10-20 nuScenes val samples, manually annotate point-level GT masks
   for 5-10 queries each. This is the only manual labeling in the project.
   Store under `configs/ov_query_annotations/`.
3. Implement `scripts/eval_ov_queries.py`:
   - For each (query, sample): cosine similarity between point embeddings
     and the query's text embedding; rank points; compute retrieval@k for
     k ∈ {1, 3, 5} against GT.
4. Implement `scripts/demo_ov_query.py`:
   - Inputs: `--sample_token`, `--query "text"`.
   - Outputs: `outputs/demos/<sample>_<slug>.ply` colored by similarity,
     and a top-1% mask.
5. Rewrite `README.md`:
   - One paragraph pitch.
   - Headline result table.
   - Pipeline diagram (ASCII or `docs/pipeline.png`).
   - Repo structure (top 2 levels).
   - Reproduction commands (5-10 lines).
   - Honest Limitations section.
6. Final pass on `docs/INTERVIEW_PREP.md`:
   - Rewrite Elevator Pitch.
   - Confirm all Decision Log rows reference actual decisions made.
   - Confirm Headline Results table is fully populated.
   - Confirm at least 5 Anticipated Questions with concrete answers.
   - Confirm Limitations section is honest.

### 8.3 Gate

- README readable in ≤ 10 minutes (subjective; user confirms).
- Demo script runs end-to-end on a fresh (sample, query) pair without
  hand-editing.
- Retrieval@5 reported for the full query benchmark.

### 8.4 Stop condition

None. This stage is documentation and packaging — no model training to fail.

### 8.5 Deliverables

- `configs/ov_queries.yaml`
- `configs/ov_query_annotations/*.json`
- `scripts/eval_ov_queries.py`, `scripts/demo_ov_query.py`
- `outputs/results/ov_query_retrieval.md`
- `outputs/demos/example_*.ply` (3-5 example outputs)
- `README.md` rewritten
- `docs/INTERVIEW_PREP.md` finalized

### 8.6 Cleanup

- Delete `outputs/scratch/` entirely.
- Confirm `git status` clean except for tracked files.
- Merge, tag `stage5-demo-complete` AND `v1.0` (project-ready release).

---

## 9. User Check-In Protocol

Ziang checks in with Claude (Cowork) at these moments. Each check-in is a
focused conversation, not a project-wide review.

| Check-in | After | Purpose | Time |
|---|---|---|---|
| C1 | Phase 0 complete | Verify repo structure / env are clean | 5-10 min |
| C2 | Stage 1 complete | Validate baseline mIoU is credible | 10 min |
| C3 | Stage 3 complete | **Critical decision**: teacher strong enough to keep OV framing, or pivot to weak-teacher framing? | 20-30 min |
| C4 | Stage 4 complete | Interpret ablation tables; decide how to frame the contribution | 30 min |
| C5 | Stage 5 complete | Review README + INTERVIEW_PREP for interview readiness | 20 min |

Between check-ins, Ziang does NOT need to interact with Claude. Codex
proceeds autonomously within the bounds of this document.

If a stop condition fires between check-ins, Codex must:
1. Stop work.
2. Update `docs/ROADMAP.md` with a Stage Retrospective.
3. Wait for the user to initiate an out-of-band check-in.

---

## 10. Document Inventory

After Phase 0, the project contains exactly these documents:

| Path | Owner | Read by | Modified by |
|---|---|---|---|
| `EXECUTION_PLAN.md` | Claude | Codex, Ziang | Locked except §11 |
| `AGENTS.md` | Claude | Codex | Locked |
| `docs/ROADMAP.md` | Codex (template by Claude) | Codex, Ziang | Codex on every stage |
| `docs/EXPERIMENT_RECAP.md` | Codex | Codex, Ziang | Codex on every experiment |
| `docs/INTERVIEW_PREP.md` | Codex | Ziang (mainly) | Codex on every stage |
| `README.md` | Codex (placeholder at Phase 0; rewritten at Stage 5) | Everyone | Codex at Stage 5 only |

No other long-form documents are permitted. Short README files inside
subdirectories (e.g., `configs/README.md` if needed) are allowed only if
strictly necessary and remain under 50 lines.

---

## 11. Execution Checklist

Codex ticks each item with `[x]` only when truly complete. Ziang inspects
this list at each check-in.

### Phase 0
- [x] `prototype-v1-archive` tag created and pushed
- [x] `migration/foundation-reset` branch contains cleanup commits
- [x] `.gitignore` updated
- [x] `scripts/setup_env.sh` written and runnable
- [x] `requirements-pointcept.txt` and `requirements.txt` written
- [x] `scripts/sanity_check.sh` written and passing
- [x] `ra_ov3dseg/utils/run_conclusion.py` written and imported successfully
- [x] `docs/ROADMAP.md`, `docs/EXPERIMENT_RECAP.md`, `docs/INTERVIEW_PREP.md` created from templates
- [x] `AGENTS.md` created
- [x] Placeholder `README.md` written
- [x] `phase0-complete` tag

### Stage 1 — Closed-Set Baseline
- [x] Branch `stage-1-baseline` created
- [x] `scripts/train_baseline.sh` written, calls Pointcept's own launcher
- [x] Full training run completed
- [ ] Independent eval via `scripts/eval_lidarseg.py` matches within ±1 mIoU
- [x] `outputs/checkpoints/closed_set_baseline.pt` produced
- [x] Gate: val mIoU ≥ 0.70
- [x] EXPERIMENT_RECAP updated
- [x] INTERVIEW_PREP Decision Log + Headline Results updated
- [x] `stage1-baseline-complete` tag

### Stage 2 — OV Head Replacement
- [x] Branch `stage-2-ov-head` created
- [x] `ra_ov3dseg/models/ov_head.py` implemented
- [x] Text prototype matrix cached
- [x] Short fine-tune completed (backbone frozen)
- [x] `outputs/checkpoints/ov_head_aligned.pt` produced
- [x] Gate: drop ≤ 0.08 vs Stage 1
- [x] EXPERIMENT_RECAP updated
- [x] INTERVIEW_PREP updated
- [ ] `stage2-ov-head-complete` tag

### Stage 3 — SAM2 + SigLIP Teacher
- [ ] Branch `stage-3-teacher` created
- [ ] `ra_ov3dseg/models/sam2_siglip_teacher.py` implemented
- [ ] `scripts/extract_sam2_teacher.py` implemented
- [ ] Teacher cache produced for diagnostic split
- [ ] Teacher comparison table populated
- [ ] Gate: projected mIoU ≥ 0.10 (or pivot decision recorded)
- [ ] EXPERIMENT_RECAP updated
- [ ] INTERVIEW_PREP updated
- [ ] `stage3-teacher-complete` tag

### Stage 4 — Reliability Distillation
- [ ] Branch `stage-4-reliability` created
- [ ] Threshold ablation: 5 runs complete
- [ ] Component ablation: 5 runs complete
- [ ] Both ablation tables and plots produced
- [ ] Gate A and Gate B passed (or pivot recorded)
- [ ] `outputs/checkpoints/ov_reliability_best.pt` produced
- [ ] EXPERIMENT_RECAP updated (10 rows)
- [ ] INTERVIEW_PREP updated
- [ ] `stage4-reliability-complete` tag

### Stage 5 — Demo & README
- [ ] Branch `stage-5-demo` created
- [ ] `configs/ov_queries.yaml` curated (30-50 queries)
- [ ] Manual annotation done for ~10-20 samples
- [ ] `scripts/eval_ov_queries.py` and `scripts/demo_ov_query.py` written
- [ ] Retrieval@k benchmark run
- [ ] `README.md` rewritten
- [ ] `INTERVIEW_PREP.md` finalized
- [ ] `stage5-demo-complete` and `v1.0` tags

---

## Appendix A: File Templates

### A.1 `docs/ROADMAP.md` template

```markdown
# RA-OV3DSeg Roadmap

> **Current Stage**: phase-0
> **Last Updated**: 2026-05-12

This is the live operational plan. The static execution plan is in
`EXECUTION_PLAN.md` and must not be modified. New decisions and adjustments
that occur during execution go in this file.

## Current Stage

**phase-0**: foundation reset. See `EXECUTION_PLAN.md` §3.

## Next Experiment

(Codex fills in: what is about to be run, with what parameters, expected
runtime, and what RunConclusion fields will look like on success.)

## Stage History

### phase-0 (in progress)
- Goal: clean repo, install Pointcept, pass sanity check.
- Status: (in progress / complete)
- Retrospective: (filled if stop condition fired)

## Pivots and Adjustments

(If Stage 3 or Stage 4 triggers a project pivot, document it here with date,
trigger, and new framing.)
```

### A.2 `docs/EXPERIMENT_RECAP.md` template

```markdown
# RA-OV3DSeg Experiment Recap

This file is the durable ledger of experiments. Each experiment ends with
its `RunConclusion` block, which automatically appends one row to the table
below via `RunConclusion.append_to_recap()`.

## Result Ledger

| Date | Stage | Experiment | Status | Primary Metric | Notes |
|---|---|---|---|---|---|

## Carryover Knowledge From Prototype Phase

This is the only narrative content allowed in this file. It captures lessons
learned in the 1-week prototype phase before the migration.

### Lessons that still apply

1. CLIP patch features are insufficient for fine-grained outdoor lidarseg.
2. GroupViT projected pseudo-labels reach only ~0.02 mIoU; not strong enough
   to drive distillation without filtering.
3. Hand-implementing sparse-conv backbones in 1 week did not match published
   numbers; pip-installed Pointcept SpUNet is the chosen direction.
4. Official nuScenes-lidarseg 16-class mapping is the correct label space
   (not raw 32-class).
5. Prediction coverage = 1.0 must be verified; voxelization that drops
   points causes evaluation artifacts.

### Lessons that did NOT generalize

(Codex fills in as new lessons supersede old ones.)
```

### A.3 `docs/INTERVIEW_PREP.md` template

```markdown
# RA-OV3DSeg Interview Preparation

This document is maintained as a side-effect of normal stage work. Every
stage completion appends to relevant sections below. The final form (after
Stage 5) is interview-ready.

## Elevator Pitch (30 seconds)

(Rewritten at each stage to reflect current project state. Phase 0 version
is a placeholder.)

> Placeholder: RA-OV3DSeg is a project on reliability-aware open-vocabulary
> 3D semantic segmentation on outdoor LiDAR scenes. Status: foundation
> reset in progress.

## Long Pitch (3 minutes)

(Filled in at Stage 5. Structure: motivation → method → results →
limitations → future work.)

## Decision Log

| Decision | Chose | Rejected | Why |
|---|---|---|---|

## Headline Results

| Metric | Value | Context |
|---|---|---|
| Closed-set val mIoU (Pointcept SpUNet) | TBD | full nuScenes-lidarseg trainval |
| Text-aligned OV head closed-set drop | TBD | vs closed-set baseline |
| SAM2+SigLIP teacher projected mIoU | TBD | on diagnostic split |
| Best reliability threshold | TBD | from ablation |
| OV-query retrieval@5 | TBD | on hand-curated benchmark |

## Anticipated Questions & Answers

### Q: Why didn't you implement your own LiDAR backbone?
A: (filled at Stage 1)

### Q: Why SAM2 + SigLIP instead of CAT-Seg or OpenSeg?
A: (filled at Stage 3)

### Q: Your novel-class mIoU is low. Is OV actually working?
A: (filled at Stage 4. This is the project's sharpest question and must
be answered honestly.)

### Q: How does your reliability score differ from a confidence threshold?
A: (filled at Stage 4)

### Q: Why only nuScenes? What about SemanticKITTI / Waymo?
A: (filled at Stage 5)

## Honest Limitations

(One bullet per limitation. Filled as discovered.)

## What I Would Do With 3 More Months

(Filled at Stage 5.)
```

### A.4 `AGENTS.md` template

```markdown
# Codex Working Rules for RA-OV3DSeg

This file is read by Codex at the start of every session. Read it before any
other action.

## Stage Discipline
- The current stage is declared on line 1-3 of `docs/ROADMAP.md`.
- Work that does not match the current stage's deliverables is forbidden.
- A stage is "complete" only when all five conditions in `EXECUTION_PLAN.md`
  §2.1 are met.

## Document Discipline
- Only three docs/* files are allowed: ROADMAP.md, EXPERIMENT_RECAP.md,
  INTERVIEW_PREP.md.
- No `*_PLAN.md`, `*_STATUS.md`, `*_NOTES.md`, `*_TODO.md`.
- New ideas → `docs/ROADMAP.md`. Results → `docs/EXPERIMENT_RECAP.md` via
  `RunConclusion.append_to_recap()`. Narrative → `docs/INTERVIEW_PREP.md`.
- `EXECUTION_PLAN.md` is read-only except for §11 checkboxes.

## Script Naming
- Semantic names only.
- Forbidden patterns: `run_v*.sh`, `stage*_run.sh`, `experiment_*.py`,
  `v[0-9]+_*.sh`.

## Third-Party Code
- `third_party/Pointcept/` is gitignored, managed by `scripts/setup_env.sh`.
- Do not modify any file under `third_party/`.
- Customization wraps Pointcept from within `ra_ov3dseg/`.

## RunConclusion
- Every training, evaluation, and extraction script ends with a
  `RunConclusion.print_block()` call.
- The block is the last thing printed. No trailing output.
- Failure cases also produce a RunConclusion with appropriate `status`.

## Stage Cleanup
- At stage completion, remove intermediate debug scripts and
  `outputs/scratch/*` artifacts.
- Run `git status`; only intentional files allowed.

## Stop Conditions
- Each stage in `EXECUTION_PLAN.md` defines a stop condition.
- On trigger: stop training, write a "Stage Retrospective" subsection to
  `docs/ROADMAP.md`, wait for user check-in.
- Gate numbers cannot be lowered. Stop conditions cannot be redefined.

## Forbidden
- New top-level planning docs.
- Modifying `EXECUTION_PLAN.md` outside §11.
- Anything under `third_party/`.
- Version numbers (`v18`, `v19`) in file names.
- A second 3D backbone alongside Pointcept SpUNet.
- Skipping ablation tables in Stage 4.
- Declaring a stage complete before all five §2.1 conditions are met.
```

### A.5 Bash wrapper skeleton for stage scripts

All `scripts/<semantic>_*.sh` follow this skeleton. Codex uses it as the
template for new wrapper scripts.

```bash
#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

EXPERIMENT_NAME="<semantic-name>"
STAGE="<stage-name>"
LOG_DIR="${PROJECT_ROOT}/outputs/logs"
LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${LOG_DIR}"
ln -sfn "$(basename ${LOG_FILE})" "${LOG_DIR}/${EXPERIMENT_NAME}_latest.log"

{
  echo "[INFO] experiment=${EXPERIMENT_NAME} stage=${STAGE}"
  echo "[INFO] started=$(date -Is)"
  nvidia-smi || true
  df -h "${PROJECT_ROOT}" || true
} | tee "${LOG_FILE}"

# Run the actual python entry. It is the python script's responsibility to
# print a RunConclusion block as its last action.
python "${PROJECT_ROOT}/scripts/<python_entry>.py" \
  --stage "${STAGE}" \
  --experiment "${EXPERIMENT_NAME}" \
  "$@" 2>&1 | tee -a "${LOG_FILE}"

# Bash-level RunConclusion fallback (used only if python entry crashed
# before printing its own block):
if ! grep -q "RUN_CONCLUSION" "${LOG_FILE}"; then
  python - <<EOF | tee -a "${LOG_FILE}"
from ra_ov3dseg.utils.run_conclusion import RunConclusion
RunConclusion(
    stage="${STAGE}",
    experiment="${EXPERIMENT_NAME}",
    status="crashed",
    gate="(see EXECUTION_PLAN.md for this stage's gate)",
    gate_passed=False,
    primary_metric_name="-",
    primary_metric_value=0.0,
    secondary={},
    runtime_seconds=0.0,
    checkpoint=None,
    artifacts=["${LOG_FILE}"],
    next_step="inspect log and fix root cause before retrying",
    notes="python entry crashed without producing RunConclusion",
).print_block()
EOF
fi
```

---

## Appendix B: Carryover From Prototype Phase

The following knowledge from the prototype phase informs Stage 1+ work. It
is captured in `docs/EXPERIMENT_RECAP.md` § "Carryover Knowledge" so Codex
sees it during normal operation. This appendix is the authoritative source.

### B.1 Known-good components (preserved from prototype)
- nuScenes IO: `ra_ov3dseg/datasets/nuscenes_mini_dataset.py` works.
- LiDAR-camera projection: `ra_ov3dseg/geometry/projection.py` works,
  produces coverage ≈ 1.0 when point cloud range is wide enough.
- Reliability formula: `w_point = w_distance * w_geometric * w_semantic`
  defined in `ra_ov3dseg/models/reliability.py`. Untested empirically;
  Stage 4 tests it.
- 6-camera dense teacher extraction pipeline: works for CLIPSeg and GroupViT.
- Open-vocabulary inference plumbing: `scripts/predict_3d_open_vocab.py`
  exists from prototype; needs to be re-pointed at Pointcept backbone in
  Stage 2.

### B.2 Known-bad approaches (do not repeat)
- Vendoring partial Pointcept source (V17). Adapter contract drifts
  indefinitely.
- Hand-implementing Cylinder3D-style backbone (V15/V16a). 0.42 mIoU at 1024
  samples is the ceiling we observed.
- Treating CLIP patch tokens as a final teacher. Coarse and weak.
- Treating GroupViT as a final teacher. 0.0195 projected mIoU is too low to
  drive distillation directly.
- Defining "open vocabulary" as "the 32 lidarseg class names encoded as
  text." That's still closed-set with a different head.

### B.3 Reference numbers
- Pointcept SpUNet on nuScenes-lidarseg val (published): ~73-76 mIoU.
- Cylinder3D paper: ~76-77 mIoU.
- Prototype V16a: 0.4159 mIoU at 1024/512/30ep (not comparable to above due
  to scale; kept as a sanity floor).

---

End of EXECUTION_PLAN.md.
