# RA-OV3DSeg Roadmap

> **Current Stage**: stage-reliability
> **Last Updated**: 2026-05-20

This is the live operational plan. The static execution plan is in
`EXECUTION_PLAN.md` and must not be modified. New decisions and adjustments
that occur during execution go in this file.

## Current Stage

**stage-reliability**: train the OV student with the Stage 2 text-aligned
head and the Stage 3 SAM2+SigLIP teacher, then run reliability threshold and
component ablations. See `EXECUTION_PLAN.md` Section 7.

## Next Experiment

Next Stage 4 experiment: rerun the 128-cache pilot threshold sweep through
`scripts/pilot_reliability_threshold_sweep.sh`. The runner rebuilds a
cache-checked Pointcept subset, overwrites prior pilot outputs by default,
hides long Pointcept logs in `outputs/logs/reliability_pilot/`, and prints a
short summary table plus a final RunConclusion. Treat the pilot as a direction
check, not the final Stage 4 gate; only generate a larger/full train cache if
the pilot shows useful threshold signal.

Server discovery commands:

```bash
find third_party/Pointcept -name "preprocess_*.py" -path "*nuscenes*"
ls third_party/Pointcept/configs/nuscenes
grep -rn "data_root" third_party/Pointcept/configs/nuscenes/*spunet*.py
```

Expected preprocessing command:

```bash
python scripts/preprocess_nuscenes_trainval_only.py \
  --dataset_root "$PWD/data/nuscenes" \
  --output_root "$PWD/data/nuscenes_pointcept_processed" \
  --max_sweeps 1 \
  --with_camera

ln -sfn "$PWD/data/nuscenes" "$PWD/data/nuscenes_pointcept_processed/raw"
```

Record the confirmed config path and processed data root here before full
training:
- Pointcept SpUNet config: TBD on server
- nuScenes raw root: `$PWD/data/nuscenes` on server
- Pointcept processed root: `$PWD/data/nuscenes_pointcept_processed` on server
- Smoke log: `outputs/logs/train_baseline_20260513_111528.log`
- Fast val log: `outputs/logs/eval_baseline_fast_20260514_004248.log`
- Fast val result: mIoU 0.7432 / mAcc 0.8095 / allAcc 0.9321

Do not run the setup in the current Windows base Python 3.13 environment.
Create or activate a clean Python 3.10 CUDA environment first, then run the
setup script from Git Bash, WSL, or the target Linux training server.

Pointcept commit `d74c646db6abec569d0f23e0c34e7ddfce142789` may not expose
`setup.py` or `pyproject.toml`. In that case `scripts/setup_env.sh` registers
`third_party/Pointcept` through `pointcept.pth` in the active Python
environment instead of running `pip install -e`.

Server environment target observed during Phase 0:
- `numpy==1.26.4`
- `opencv-python-headless==4.8.1.78`
- PyG extension wheels from
  `https://data.pyg.org/whl/torch-2.1.0+cu121.html`.
- `pointops` is intentionally not installed. Stage 1-5 use SpUNet through
  spconv, and pointops is only needed for Pointcept PTv3-style paths.

## Stage History

### stage-reliability (in progress)
- Goal: test whether reliability filtering turns the weak SAM2+SigLIP teacher
  into useful 3D distillation supervision.
- Status: branch created on 2026-05-20 after `stage3-teacher-complete`.
- First task: update/verify reliability computation so it consumes
  SAM2+SigLIP dense point pseudo-labels and excludes background/ignore from
  semantic confidence ranking.
- Reliability smoke result: 128-sample cache count matched expected outputs,
  and semantic_score_ratio mean was 0.4196, matching the Stage 3 excluded-ratio
  diagnosis. Raw multiplicative reliability weights were healthy but lived on
  a 0.0-0.2009 scale, so the fixed Stage 4 thresholds [0.3, 0.5, 0.7, 0.9]
  would zero out all non-zero-threshold runs if applied directly.
- Adjustment: keep the raw multiplicative product as `reliability_weight_raw`,
  and use rank-calibrated `reliability_weight` in [0, 1] for threshold
  ablations. This preserves the planned threshold grid without lowering gates.
- Rank-calibrated retry result: 128 summaries and 128 npz files matched;
  semantic_score_ratio mean stayed 0.4196; reliability mean was 0.4990;
  high_reliability_ratio mean at threshold 0.5 was 0.4990; quantiles were
  [0.0000, 0.2486, 0.4991, 0.7496, 0.8999, 0.9500, 0.9900, 1.0000].
- Training wiring added: `scripts/train_reliability_distillation.sh` loads
  the Stage 2 OV head checkpoint, injects reliability teacher fields through a
  Pointcept transform, slices SAM2+SigLIP's appended background class away, and
  applies reliability-weighted dense KL alongside supervised CE.
- Reliability distillation smoke passed on the server: threshold 0.5 produced
  val_mIoU 0.5686 on the smoke subset, with distill_valid_ratio 0.1190 and
  distill_mean_weight 0.7338. This validates batch-field propagation and
  Pointcept point-index alignment after GridSample, but the smoke mIoU is not
  a Stage 4 quality result.
- Hidden constraint: the current teacher/reliability cache covers only the 128
  diagnostic samples. Full train-split ablations require a larger/full cache;
  using the 128 cache with full train data would either fail in strict mode or
  silently reduce distillation to a tiny fraction of batches in non-strict mode.
- Pilot runner adjustment: generated subset info files now rewrite Pointcept's
  `lidar_token`/`name` to `sample_XXXX`, so reliability lookup uses explicit
  sample ids instead of fragile coordinate fingerprints. The runner preflights
  cache coverage before training and redirects verbose Pointcept output away
  from tmux.
- Planned gate: at least one non-zero reliability threshold beats threshold=0
  by >= 0.005 mIoU, and at least one component removal hurts by >= 0.005 mIoU.

### stage-teacher (complete)
- Goal: produce dense teacher pseudo-labels from SAM2 masks classified by
  SigLIP text prototypes, then project them to LiDAR points.
- Status: complete. Tag `stage3-teacher-complete` created on 2026-05-20.
- Latest diagnostic: 128 cross-scene samples passed on 2026-05-20 with
  projected teacher mIoU 0.1022 and prediction coverage 0.5578. Semantic-only
  confidence ranking gave top-20% mIoU 0.3149 and top-40% mIoU 0.3159.
- Previous 32-sample diagnostic passed with projected teacher mIoU 0.1376 and
  prediction coverage 0.5464. Semantic-only confidence ranking gave top-20%
  mIoU 0.3002 and top-40% mIoU 0.3386.
- Previous 5-sample diagnostic passed with projected teacher mIoU 0.1148 and
  prediction coverage 0.4209. Per-sample mIoU was 0.1367, 0.1583, 0.1037,
  0.1132, and 0.0775.
- Teacher comparison artifact: `outputs/results/teacher_comparison.md`.
- Teacher environment: use a separate `ra-teacher` env with PyTorch/TorchVision
  2.5/0.20 and `requirements-teacher.txt`; keep the Pointcept training env on
  its validated torch/spconv stack.
- Adjustment: the initial SAM2+SigLIP attempt collapsed to background and
  construction_vehicle at 0.0024 mIoU. Removing background from mask
  classification and replacing raw class names with driving-scene phrases fixed
  the immediate failure.
- Adjustment: naive top-confidence ranking was misleading because
  background/ignore predictions dominated high-confidence points. The
  diagnostic now excludes ignore/background labels for semantic confidence
  ranking while still reporting the excluded ratio; on the 128-sample run,
  58.22% of score-valid points were excluded from semantic ranking.
- Decision: do not treat full SAM2+SigLIP pseudo-labels as clean supervision.
  Stage 4 should test whether reliability filtering can exploit the cleaner
  semantic subset while downweighting noisy/background-dominated regions.
- Retrospective: no stop condition fired. The raw teacher only narrowly clears
  the 0.10 gate, but semantic confidence subsets are much cleaner, which is
  sufficient to justify Stage 4 reliability ablations.

### stage-ov-head (complete)
- Goal: replace the closed-set head with a SigLIP prototype cosine head while
  preserving most closed-set performance.
- Status: gate passed on 2026-05-20. Best val mIoU was 0.7465, with final eval
  at 0.7449. This is above the 0.6632 threshold and does not drop from the
  Stage 1 baseline.
- Retrospective: no stop condition fired. The result should be framed as
  matching the closed-set baseline within noise, not as a meaningful
  improvement over Stage 1.

### stage-baseline (complete)
- Goal: reproduce Pointcept SpUNet nuScenes-lidarseg val mIoU >= 0.70.
- Status: accepted. Fast full-val gate passed on 2026-05-14 with mIoU 0.7432.
- C2 check-in: baseline is credible for Stage 2. It uses Pointcept SpUNet's
  own recipe and evaluator; routine runs skip the slow `PreciseEvaluator`.
- Retrospective: no stop condition fired. The final checkpoint is preserved as
  `outputs/checkpoints/closed_set_baseline.pt` on the server. The Stage 2
  threshold is 0.6632 mIoU, computed as 0.7432 - 0.08.

### phase-0 (complete)
- Goal: clean repo, install Pointcept, pass sanity check.
- Status: complete. Server setup and sanity check passed on 2026-05-13.
- Retrospective: no stop condition fired. Pointcept is registered through
  `pointcept.pth`; `pointops` is intentionally skipped because this project
  uses SpUNet through spconv, not PTv3 paths.

## Pivots and Adjustments

If Stage 3 or Stage 4 triggers a project pivot, document it here with date,
trigger, and new framing.
