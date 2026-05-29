# RA-OV3DSeg Roadmap

> **Current Stage**: stage-reliability
> **Last Updated**: 2026-05-29

This is the live operational plan. The static execution plan is in
`EXECUTION_PLAN.md` and must not be modified. New decisions and adjustments
that occur during execution go in this file.

## Current Stage

**stage-reliability**: train the OV student with the Stage 2 text-aligned
head and the Stage 3 SAM2+SigLIP teacher, then run reliability threshold and
component ablations. See `EXECUTION_PLAN.md` Section 7.

## Next Experiment

Stage 4 is closed for model work. Do not expand to a full 6019-sample teacher
cache, do not switch to another teacher family, and do not start the original
full 10-run 20-epoch ablation plan unless the user explicitly reopens the
scope.

Final Stage 4 closure result: the 128-cache component/control pilot completed
on 2026-05-29 and did **not** support the reliability formula. `full`
reliability reached 0.0619 mIoU, while the random same-scale control reached
0.0793 mIoU. Removing components did not produce a meaningful drop from
`full` (`max_drop_vs_full=0.0023`, below the 0.005 gate). The remaining work is
documentation only: frame Stage 4 as a weak-teacher distillation diagnostic,
not a positive method claim.

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
  `lidar_token`/`name` to `sample_<idx>`, so reliability lookup uses explicit
  sample ids instead of fragile coordinate fingerprints. The runner preflights
  cache coverage and raw LiDAR/cache coordinate alignment before training,
  redirects verbose Pointcept output away from tmux, and shows tracebacks only
  when `PILOT_VERBOSE_FAILURE=1`.
- Token-namespace fix: generated subset info now writes the manifest
  `sample_token` back into each matched row and preserves Pointcept's original
  tokens under `original_*`, because Pointcept info tokens can be LiDAR
  sample_data identifiers rather than nuScenes sample identifiers.
- Reliability lookup now treats only `sample_token` as a token fallback and
  does not trust generic `token`; pilot preflight reports token namespace
  differences as warnings while using point count and raw/cache coordinate
  equality as the hard alignment gate.
- Timestamp fallback now uses only exact microsecond timestamps. Integer-second
  matching is forbidden because nuScenes has multiple samples per second and
  can map Pointcept info rows to the wrong teacher cache.
- Pilot subset generation is cache-backed: for reliability pilot runs,
  `make_nuscenes_smoke_infos.py` writes subset LiDAR `.bin` files from
  reliability-cache `point_xyz` and matching lidarseg labels into
  `outputs/pointcept/reliability_subset_128/raw/`. This makes the Pointcept
  student coordinates and teacher cache share one point contract instead of
  depending on an external raw symlink.
- 128-cache threshold pilot passed on 2026-05-21: all 5 thresholds completed,
  best pilot val_mIoU was 0.4554 at threshold 0.9. This is a real positive
  reliability signal, but not the final Stage 4 gate because it uses the
  diagnostic cache-backed subset rather than a larger/full train cache.
  Threshold sweep table: t=0.0 -> 0.3636 mIoU, valid=0.2571; t=0.3 ->
  0.3155, valid=0.2835; t=0.5 -> 0.3659, valid=0.1648; t=0.7 -> 0.3946,
  valid=0.1054; t=0.9 -> 0.4554, valid=0.0258. Strict filtering beats the
  unfiltered teacher by +0.0918 mIoU in the pilot.
- Planned gate: at least one non-zero reliability threshold beats threshold=0
  by >= 0.005 mIoU, and at least one component removal hurts by >= 0.005 mIoU.
- Closure adjustment on 2026-05-28: the remaining work is intentionally capped
  at a 128-cache component/control pilot plus documentation. This keeps the
  project honest and prevents full-scale compute spend on a weak teacher whose
  main value is diagnostic rather than state-of-the-art performance.
- Component/control pilot completed on 2026-05-29: `full` 0.0619 mIoU,
  `random` 0.0793, `uniform` 0.0683, `no_distance` 0.0596, `no_geometric`
  0.0793, `no_semantic` 0.0613. Gate failed: `full` did not beat `random`
  (`random_gap_vs_full=-0.0174`) and component removal barely hurt
  (`max_drop_vs_full=0.0023`). Conclusion: the reliability formula is not
  empirically supported by the closure pilot; the defensible result is that
  weak 2D-to-3D teachers need selective/controlled supervision, but this
  particular reliability score is not validated.

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

- 2026-05-28: Stage 4 closure pivot. Trigger: the SAM2+SigLIP teacher is weak
  enough that full-scale distillation would have poor cost/benefit, and the
  strongest scientific risk is that reliability filtering may simply be a
  sparse-supervision control. New framing: finish a bounded pilot study of
  weak 2D-to-3D open-vocabulary teacher distillation, including random and
  component controls, then stop model work and document the limitations.
- 2026-05-29: Stage 4 closure result. The bounded component/control pilot
  failed the reliability-specific gate because random sparse filtering matched
  or beat the proposed score. New framing: negative/limited finding on this
  reliability formulation, with useful engineering lessons about teacher
  weakness, cache alignment, and control ablations.
