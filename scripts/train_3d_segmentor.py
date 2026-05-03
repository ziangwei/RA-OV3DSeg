from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_ov3dseg.datasets.nuscenes_dataset import NuScenesDataset  # noqa: E402
from ra_ov3dseg.models.segmentor_factory import (  # noqa: E402
    DEBUG_BACKBONE,
    SUPPORTED_BACKBONES,
    build_segmentor,
    describe_backbone,
)
from ra_ov3dseg.training.labels import build_class_split  # noqa: E402
from ra_ov3dseg.training.losses import cosine_distillation_loss, supervised_ce_loss  # noqa: E402
from ra_ov3dseg.training.precomputed_dataset import (  # noqa: E402
    IGNORE_INDEX,
    PrecomputedPointFeatureDataset,
    collate_point_feature_samples,
    label_hist,
)
from ra_ov3dseg.utils.io import ensure_dir, save_json  # noqa: E402
from ra_ov3dseg.utils.logger import setup_logger  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generic MVP-v4 3D segmentor trainer with single-GPU and torchrun DDP support."
    )
    parser.add_argument("--dataroot", required=True, type=str, help="nuScenes dataroot.")
    parser.add_argument("--version", default="v1.0-mini", type=str, help="nuScenes version.")
    parser.add_argument("--sample_idx", default=None, type=int, help="Train on one sample index.")
    parser.add_argument("--start_idx", default=0, type=int, help="First sample index when sample_idx is not set.")
    parser.add_argument("--max_samples", default=1, type=int, help="Number of samples to use. Ignored by --all_samples.")
    parser.add_argument("--all_samples", action="store_true", help="Use all samples from start_idx to dataset end.")
    parser.add_argument("--point_feature_dir", default="outputs/point_features", type=str)
    parser.add_argument("--reliability_dir", default="outputs/reliability", type=str)
    parser.add_argument("--class_names_path", default="configs/nuscenes_lidarseg_class_names.txt", type=str)
    parser.add_argument("--split_config", default="configs/base_novel_split.yaml", type=str)
    parser.add_argument("--output_dir", default="outputs/training_v4", type=str)
    parser.add_argument(
        "--backbone",
        default=DEBUG_BACKBONE,
        choices=list(SUPPORTED_BACKBONES),
        help="3D backbone. debug_point_mlp is only a smoke-test model; sparse_unet_spconv is the planned V5 adapter.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Training device.")
    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--batch_size", default=1, type=int, help="Samples per process per optimizer step.")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--max_points", default=20000, type=int, help="Subsample points per sample; <=0 keeps all.")
    parser.add_argument("--hidden_dim", default=128, type=int, help="Debug MLP hidden dim.")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--ce_weight", default=1.0, type=float)
    parser.add_argument("--distill_weight", default=1.0, type=float)
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--save_every", default=1, type=int, help="Save checkpoint every N epochs on rank 0.")
    parser.add_argument("--no_shuffle", action="store_true", help="Disable training shuffle.")
    parser.add_argument("--ddp_backend", default="auto", choices=["auto", "nccl", "gloo"])
    return parser


def import_torch():
    try:
        import torch
        from torch.nn.parallel import DistributedDataParallel
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler
    except ImportError as exc:
        raise ImportError("train_3d_segmentor.py requires PyTorch. Install a CUDA-matched torch build first.") from exc
    return torch, DistributedDataParallel, DataLoader, DistributedSampler


def setup_distributed(torch_module, ddp_backend: str) -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    backend = ddp_backend
    if backend == "auto":
        backend = "nccl" if torch_module.cuda.is_available() else "gloo"

    if distributed:
        import torch.distributed as dist

        if backend == "nccl" and not torch_module.cuda.is_available():
            raise RuntimeError("DDP backend nccl requires CUDA.")
        dist.init_process_group(backend=backend)

    return {
        "distributed": distributed,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "backend": backend,
    }


def resolve_device(torch_module, requested: str, local_rank: int):
    if requested == "cpu":
        return torch_module.device("cpu")
    if requested == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("--device cuda was requested but CUDA is not available.")
        torch_module.cuda.set_device(local_rank)
        return torch_module.device("cuda", local_rank)
    if torch_module.cuda.is_available():
        torch_module.cuda.set_device(local_rank)
        return torch_module.device("cuda", local_rank)
    return torch_module.device("cpu")


def is_main_process(distributed_state: dict[str, Any]) -> bool:
    return int(distributed_state["rank"]) == 0


def numpy_batch_to_torch(batch: dict[str, Any], torch_module, device) -> dict[str, Any]:
    return {
        "sample_indices": batch["sample_indices"],
        "sample_tokens": batch["sample_tokens"],
        "point_xyz": torch_module.from_numpy(batch["point_xyz"]).to(device, non_blocking=True),
        "teacher_features": torch_module.from_numpy(batch["teacher_features"]).to(device, non_blocking=True),
        "teacher_valid_mask": torch_module.from_numpy(batch["teacher_valid_mask"]).bool().to(device, non_blocking=True),
        "reliability_weight": torch_module.from_numpy(batch["reliability_weight"]).float().to(device, non_blocking=True),
        "train_labels": torch_module.from_numpy(batch["train_labels"]).long().to(device, non_blocking=True),
        "raw_labels": batch["raw_labels"],
    }


def compute_grad_norm(model) -> float:
    grad_norm_sq = 0.0
    for param in model.parameters():
        if param.grad is not None:
            grad_norm_sq += float(param.grad.detach().float().pow(2).sum().cpu().item())
    return math.sqrt(grad_norm_sq)


def reduce_epoch_stats(stats: dict[str, float], torch_module, device, distributed: bool) -> dict[str, float]:
    keys = [
        "steps",
        "points",
        "base_supervised_points",
        "distill_points",
        "total_loss_sum",
        "ce_loss_sum",
        "distill_loss_sum",
        "grad_norm_sum",
    ]
    tensor = torch_module.tensor([float(stats[key]) for key in keys], dtype=torch_module.float64, device=device)
    if distributed:
        import torch.distributed as dist

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    values = tensor.detach().cpu().tolist()
    reduced = dict(zip(keys, values))
    steps = max(reduced["steps"], 1.0)
    reduced["avg_total_loss"] = reduced["total_loss_sum"] / steps
    reduced["avg_ce_loss"] = reduced["ce_loss_sum"] / steps
    reduced["avg_distill_loss"] = reduced["distill_loss_sum"] / steps
    reduced["avg_grad_norm"] = reduced["grad_norm_sum"] / steps
    return reduced


def save_checkpoint(
    path: Path,
    torch_module,
    model,
    optimizer,
    epoch: int,
    args: argparse.Namespace,
    class_split,
    metrics: dict[str, Any],
    sample_indices: list[int],
    distributed_state: dict[str, Any],
    backbone_spec,
) -> None:
    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "metrics": metrics,
        "sample_indices": sample_indices,
        "distributed": distributed_state,
        "backbone": backbone_spec.__dict__,
        "class_split": {
            "class_names": class_split.class_names,
            "base_class_names": class_split.base_class_names,
            "novel_class_names": class_split.novel_class_names,
            "ignore_class_names": class_split.ignore_class_names,
            "train_id_to_label_id": class_split.train_id_to_label_id.tolist(),
            "label_id_to_train_id": class_split.label_id_to_train_id.tolist(),
        },
    }
    torch_module.save(checkpoint, path)


def main() -> int:
    args = build_parser().parse_args()
    logger = setup_logger("train_3d_segmentor")
    torch, DistributedDataParallel, DataLoader, DistributedSampler = import_torch()

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")

    backbone_spec = describe_backbone(args.backbone)
    distributed_state = setup_distributed(torch, args.ddp_backend)
    device = resolve_device(torch, args.device, int(distributed_state["local_rank"]))
    main_process = is_main_process(distributed_state)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    output_dir = ensure_dir(args.output_dir)
    dataset = NuScenesDataset(args.dataroot, version=args.version, verbose=main_process)
    max_samples = None if args.all_samples else args.max_samples
    sample_indices = dataset.resolve_sample_indices(
        sample_idx=args.sample_idx,
        start_idx=args.start_idx,
        max_samples=max_samples,
    )
    class_split = build_class_split(args.class_names_path, args.split_config)
    max_points = None if args.max_points <= 0 else args.max_points
    train_dataset = PrecomputedPointFeatureDataset(
        nuscenes_dataset=dataset,
        sample_indices=sample_indices,
        point_feature_dir=args.point_feature_dir,
        reliability_dir=args.reliability_dir,
        class_split=class_split,
        max_points=max_points,
        seed=args.seed,
        ignore_index=IGNORE_INDEX,
    )

    first_sample = train_dataset[0]
    feature_dim = int(first_sample["teacher_features"].shape[1])
    raw_hist = label_hist(first_sample["raw_labels"], class_split.class_names)

    sampler = None
    if distributed_state["distributed"]:
        sampler = DistributedSampler(
            train_dataset,
            num_replicas=int(distributed_state["world_size"]),
            rank=int(distributed_state["rank"]),
            shuffle=not args.no_shuffle,
            seed=args.seed,
            drop_last=False,
        )

    loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None and not args.no_shuffle),
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_point_feature_samples,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = build_segmentor(
        backbone=args.backbone,
        input_dim=3,
        hidden_dim=args.hidden_dim,
        feature_dim=feature_dim,
        num_classes=class_split.num_train_classes,
    ).to(device)
    if distributed_state["distributed"]:
        device_ids = [int(distributed_state["local_rank"])] if device.type == "cuda" else None
        model = DistributedDataParallel(model, device_ids=device_ids)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))

    if main_process:
        logger.info(
            "train start | backbone=%s | role=%s | version=%s | samples=%d | device=%s | ddp=%s | world_size=%d",
            backbone_spec.backbone,
            backbone_spec.role,
            args.version,
            len(sample_indices),
            device,
            distributed_state["distributed"],
            distributed_state["world_size"],
        )
        if backbone_spec.is_debug_model:
            logger.warning("debug backbone in use: %s", backbone_spec.description)

    epoch_logs: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        local_stats = {
            "steps": 0.0,
            "points": 0.0,
            "base_supervised_points": 0.0,
            "distill_points": 0.0,
            "total_loss_sum": 0.0,
            "ce_loss_sum": 0.0,
            "distill_loss_sum": 0.0,
            "grad_norm_sum": 0.0,
        }

        for batch in loader:
            torch_batch = numpy_batch_to_torch(batch, torch, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=bool(args.amp and device.type == "cuda")):
                outputs = model(torch_batch["point_xyz"])
                ce_loss = supervised_ce_loss(outputs["logits"], torch_batch["train_labels"], ignore_index=IGNORE_INDEX)
                distill_loss = cosine_distillation_loss(
                    student_features=outputs["point_features"],
                    teacher_features=torch_batch["teacher_features"],
                    weights=torch_batch["reliability_weight"],
                    valid_mask=torch_batch["teacher_valid_mask"],
                )
                total_loss = args.ce_weight * ce_loss + args.distill_weight * distill_loss

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = compute_grad_norm(model)
            scaler.step(optimizer)
            scaler.update()

            base_points = int(torch.sum(torch_batch["train_labels"] != IGNORE_INDEX).detach().cpu().item())
            distill_points = int(
                torch.sum(
                    torch_batch["teacher_valid_mask"]
                    & torch.isfinite(torch_batch["reliability_weight"])
                    & (torch_batch["reliability_weight"] > 0.0)
                )
                .detach()
                .cpu()
                .item()
            )
            num_points = int(torch_batch["point_xyz"].shape[0])

            local_stats["steps"] += 1.0
            local_stats["points"] += float(num_points)
            local_stats["base_supervised_points"] += float(base_points)
            local_stats["distill_points"] += float(distill_points)
            local_stats["total_loss_sum"] += float(total_loss.detach().cpu().item())
            local_stats["ce_loss_sum"] += float(ce_loss.detach().cpu().item())
            local_stats["distill_loss_sum"] += float(distill_loss.detach().cpu().item())
            local_stats["grad_norm_sum"] += float(grad_norm)

        epoch_stats = reduce_epoch_stats(
            local_stats,
            torch_module=torch,
            device=device,
            distributed=bool(distributed_state["distributed"]),
        )
        epoch_log = {
            "epoch": epoch,
            "steps": int(epoch_stats["steps"]),
            "points": int(epoch_stats["points"]),
            "base_supervised_points": int(epoch_stats["base_supervised_points"]),
            "distill_points": int(epoch_stats["distill_points"]),
            "avg_total_loss": float(epoch_stats["avg_total_loss"]),
            "avg_ce_loss": float(epoch_stats["avg_ce_loss"]),
            "avg_distill_loss": float(epoch_stats["avg_distill_loss"]),
            "avg_grad_norm": float(epoch_stats["avg_grad_norm"]),
        }
        epoch_logs.append(epoch_log)

        if main_process:
            logger.info(
                "epoch=%d | loss=%.6f | ce=%.6f | distill=%.6f | points=%d | base_points=%d | distill_points=%d",
                epoch,
                epoch_log["avg_total_loss"],
                epoch_log["avg_ce_loss"],
                epoch_log["avg_distill_loss"],
                epoch_log["points"],
                epoch_log["base_supervised_points"],
                epoch_log["distill_points"],
            )
            if args.save_every > 0 and (epoch % args.save_every == 0 or epoch == args.epochs):
                save_checkpoint(
                    output_dir / f"{args.backbone}_epoch_{epoch:04d}.pt",
                    torch,
                    model,
                    optimizer,
                    epoch,
                    args,
                    class_split,
                    epoch_log,
                    sample_indices,
                    distributed_state,
                    backbone_spec,
                )

    if main_process:
        latest_path = output_dir / f"{args.backbone}_latest.pt"
        final_metrics = epoch_logs[-1] if epoch_logs else {}
        save_checkpoint(
            latest_path,
            torch,
            model,
            optimizer,
            args.epochs,
            args,
            class_split,
            final_metrics,
            sample_indices,
            distributed_state,
            backbone_spec,
        )
        summary = {
            "status": "pass",
            "version": args.version,
            "device": str(device),
            "distributed": distributed_state,
            "backbone": backbone_spec.__dict__,
            "num_samples": len(sample_indices),
            "sample_indices": sample_indices,
            "epochs_completed": args.epochs,
            "batch_size_per_process": args.batch_size,
            "max_points_per_sample": max_points,
            "feature_dim": feature_dim,
            "num_base_train_classes": class_split.num_train_classes,
            "base_classes": class_split.base_class_names,
            "novel_classes": class_split.novel_class_names,
            "ignore_classes": class_split.ignore_class_names,
            "first_sample_raw_label_hist": raw_hist,
            "loss_weights": {
                "ce_weight": args.ce_weight,
                "distill_weight": args.distill_weight,
            },
            "epoch_logs": epoch_logs,
            "latest_checkpoint": str(latest_path),
        }
        summary_path = output_dir / "train_summary.json"
        save_json(summary_path, summary)
        logger.info("training PASS | latest=%s | summary=%s", latest_path, summary_path)

    if distributed_state["distributed"]:
        import torch.distributed as dist

        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
