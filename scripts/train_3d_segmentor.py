from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ra_ov3dseg.datasets.nuscenes_dataset import NuScenesDataset  # noqa: E402
from ra_ov3dseg.models.segmentor_factory import (  # noqa: E402
    DEBUG_BACKBONE,
    SPCONV_BACKBONES,
    SUPPORTED_BACKBONES,
    build_segmentor,
    describe_backbone,
)
from ra_ov3dseg.training.labels import build_class_split  # noqa: E402
from ra_ov3dseg.training.losses import (  # noqa: E402
    cosine_distillation_loss,
    dense_logit_distillation_loss,
    dice_loss,
    lovasz_softmax_loss,
    supervised_ce_loss,
    text_prototype_alignment_loss,
)
from ra_ov3dseg.training.precomputed_dataset import (  # noqa: E402
    IGNORE_INDEX,
    PrecomputedPointFeatureDataset,
    collate_point_feature_samples,
    find_missing_dense_point_files,
    find_missing_precomputed_files,
    label_hist,
)
from ra_ov3dseg.training.raw_lidarseg_dataset import RawLidarsegDataset  # noqa: E402
from ra_ov3dseg.utils.io import ensure_dir, save_json  # noqa: E402
from ra_ov3dseg.utils.io import load_json  # noqa: E402
from ra_ov3dseg.utils.logger import setup_logger  # noqa: E402
from ra_ov3dseg.evaluation.metrics import mean_iou_for_ids, safe_iou, segmentation_intersections_unions  # noqa: E402
from ra_ov3dseg.training.augmentations import PointAugmentationConfig  # noqa: E402


FEATURE_DISTILL_MODE = "feature_distill"
DENSE_LOGIT_DISTILL_MODE = "dense_logit_distill"
HYBRID_TEACHER_MODE = "hybrid"
TEACHER_MODES = (FEATURE_DISTILL_MODE, DENSE_LOGIT_DISTILL_MODE, HYBRID_TEACHER_MODE)
STUDENT_OUTPUT_SPACES = ("auto", "base", "all_lidarseg")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generic MVP-v4/v5 3D segmentor trainer with single-GPU and torchrun DDP support."
    )
    parser.add_argument("--dataroot", required=True, type=str, help="nuScenes dataroot.")
    parser.add_argument("--version", default="v1.0-mini", type=str, help="nuScenes version.")
    parser.add_argument("--sample_idx", default=None, type=int, help="Train on one sample index.")
    parser.add_argument("--start_idx", default=0, type=int, help="First sample index when sample_idx is not set.")
    parser.add_argument("--max_samples", default=1, type=int, help="Number of samples to use. Ignored by --all_samples.")
    parser.add_argument("--all_samples", action="store_true", help="Use all samples from start_idx to dataset end.")
    parser.add_argument("--point_feature_dir", default="outputs/point_features", type=str)
    parser.add_argument("--reliability_dir", default="outputs/reliability", type=str)
    parser.add_argument("--dense_point_dir", default="outputs/dense_point_logits", type=str)
    parser.add_argument(
        "--data_source",
        default="precomputed",
        choices=["precomputed", "raw_lidarseg"],
        help="precomputed uses MVP point-feature caches; raw_lidarseg reads LiDAR/lidarseg directly.",
    )
    parser.add_argument(
        "--teacher_mode",
        default=FEATURE_DISTILL_MODE,
        choices=list(TEACHER_MODES),
        help=(
            "feature_distill uses point_features cosine loss; dense_logit_distill uses V6 dense point logits; "
            "hybrid uses both."
        ),
    )
    parser.add_argument(
        "--skip_missing_precomputed",
        action="store_true",
        help="Skip samples missing point_features/reliability npz instead of failing before training.",
    )
    parser.add_argument("--class_names_path", default="configs/nuscenes_lidarseg_class_names.txt", type=str)
    parser.add_argument("--split_config", default="configs/base_novel_split.yaml", type=str)
    parser.add_argument("--output_dir", default="outputs/training_v4", type=str)
    parser.add_argument(
        "--init_checkpoint",
        default=None,
        type=str,
        help="Optional checkpoint to initialize model weights before training.",
    )
    parser.add_argument(
        "--backbone",
        default=DEBUG_BACKBONE,
        choices=list(SUPPORTED_BACKBONES),
        help="3D backbone. debug_point_mlp is only a smoke-test model; spconv_resunet is the stronger upper-bound check.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Training device.")
    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--batch_size", default=1, type=int, help="Samples per process per optimizer step.")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--max_points", default=20000, type=int, help="Subsample points per sample; <=0 keeps all.")
    parser.add_argument("--hidden_dim", default=128, type=int, help="Debug MLP hidden dim.")
    parser.add_argument("--feature_dim", default=512, type=int, help="Point embedding dim for raw_lidarseg training.")
    parser.add_argument("--voxel_size", default=(0.2, 0.2, 0.2), nargs=3, type=float, metavar=("VX", "VY", "VZ"))
    parser.add_argument(
        "--point_cloud_range",
        default=(-54.0, -54.0, -5.0, 54.0, 54.0, 3.0),
        nargs=6,
        type=float,
        metavar=("X_MIN", "Y_MIN", "Z_MIN", "X_MAX", "Y_MAX", "Z_MAX"),
    )
    parser.add_argument("--sparse_base_channels", default=32, type=int, help="Base channels for sparse_unet_spconv.")
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--ce_weight", default=1.0, type=float)
    parser.add_argument("--class_weights_path", default=None, type=str, help="Optional JSON from compute_class_frequencies.py.")
    parser.add_argument("--distill_weight", default=1.0, type=float)
    parser.add_argument("--dense_logit_weight", default=1.0, type=float)
    parser.add_argument("--lovasz_weight", default=0.0, type=float, help="Weight for Lovasz-Softmax supervised loss.")
    parser.add_argument("--dice_weight", default=0.0, type=float, help="Weight for soft Dice supervised loss.")
    parser.add_argument(
        "--text_align_weight",
        default=0.0,
        type=float,
        help="Weight for supervised point embedding -> class text prototype cosine loss.",
    )
    parser.add_argument("--dense_temperature", default=1.0, type=float)
    parser.add_argument("--text_model_name", default="openai/clip-vit-base-patch16", type=str)
    parser.add_argument("--text_prompt_template", default="a {} in a driving scene", type=str)
    parser.add_argument("--cache_dir", default=None, type=str, help="Hugging Face cache dir for text prototypes.")
    parser.add_argument("--local_files_only", action="store_true", help="Load text encoder from local cache only.")
    parser.add_argument(
        "--student_output_space",
        default="auto",
        choices=list(STUDENT_OUTPUT_SPACES),
        help="auto uses all_lidarseg for dense/hybrid teacher modes and base for feature_distill.",
    )
    parser.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    parser.add_argument("--augment", action="store_true", help="Enable supervised LiDAR point augmentations.")
    parser.add_argument("--aug_rotation_z", default=3.141592653589793, type=float)
    parser.add_argument("--aug_flip_x_prob", default=0.5, type=float)
    parser.add_argument("--aug_flip_y_prob", default=0.5, type=float)
    parser.add_argument("--aug_scale_min", default=0.95, type=float)
    parser.add_argument("--aug_scale_max", default=1.05, type=float)
    parser.add_argument("--aug_dropout_prob", default=0.1, type=float)
    parser.add_argument("--eval_start_idx", default=None, type=int, help="Optional eval start sample index for in-training mIoU.")
    parser.add_argument("--eval_max_samples", default=0, type=int, help="Number of eval samples for in-training mIoU.")
    parser.add_argument("--eval_every", default=0, type=int, help="Evaluate and update best checkpoint every N epochs.")
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


def torch_load_checkpoint(torch_module, checkpoint_path: Path) -> dict[str, Any]:
    try:
        return torch_module.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch_module.load(checkpoint_path, map_location="cpu")


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
        "point_batch_indices": torch_module.from_numpy(batch["point_batch_indices"]).long().to(device, non_blocking=True),
        "teacher_features": torch_module.from_numpy(batch["teacher_features"]).to(device, non_blocking=True),
        "teacher_valid_mask": torch_module.from_numpy(batch["teacher_valid_mask"]).bool().to(device, non_blocking=True),
        "reliability_weight": torch_module.from_numpy(batch["reliability_weight"]).float().to(device, non_blocking=True),
        "dense_teacher_logits": torch_module.from_numpy(batch["dense_teacher_logits"]).float().to(device, non_blocking=True),
        "dense_teacher_valid_mask": torch_module.from_numpy(batch["dense_teacher_valid_mask"])
        .bool()
        .to(device, non_blocking=True),
        "dense_teacher_confidence": torch_module.from_numpy(batch["dense_teacher_confidence"])
        .float()
        .to(device, non_blocking=True),
        "train_labels": torch_module.from_numpy(batch["train_labels"]).long().to(device, non_blocking=True),
        "all_class_train_labels": torch_module.from_numpy(batch["all_class_train_labels"])
        .long()
        .to(device, non_blocking=True),
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
        "dense_distill_points",
        "total_loss_sum",
        "ce_loss_sum",
        "lovasz_loss_sum",
        "dice_loss_sum",
        "distill_loss_sum",
        "dense_logit_loss_sum",
        "text_align_loss_sum",
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
    reduced["avg_lovasz_loss"] = reduced["lovasz_loss_sum"] / steps
    reduced["avg_dice_loss"] = reduced["dice_loss_sum"] / steps
    reduced["avg_distill_loss"] = reduced["distill_loss_sum"] / steps
    reduced["avg_dense_logit_loss"] = reduced["dense_logit_loss_sum"] / steps
    reduced["avg_text_align_loss"] = reduced["text_align_loss_sum"] / steps
    reduced["avg_grad_norm"] = reduced["grad_norm_sum"] / steps
    return reduced


def build_text_prototypes(args: argparse.Namespace, class_names: list[str], torch_module, device):
    if args.text_align_weight <= 0.0:
        return None
    from ra_ov3dseg.models.text_encoder import TextEncoder

    encoder = TextEncoder(
        model_name=args.text_model_name,
        device=str(device),
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    text_result = encoder.encode_texts(
        class_names,
        prompt_template=args.text_prompt_template,
        normalize=True,
    )
    return torch_module.from_numpy(text_result["text_embeddings"]).float().to(device)


def load_class_weights(args: argparse.Namespace, num_output_classes: int, torch_module, device):
    if args.class_weights_path is None:
        return None, {}
    weight_path = Path(args.class_weights_path).expanduser().resolve()
    if not weight_path.exists():
        raise FileNotFoundError(f"class weights json not found: {weight_path}")
    data = load_json(weight_path)
    key = "raw_class_weights" if num_output_classes == len(data.get("class_names", [])) else "train_class_weights"
    weights = data.get(key)
    if weights is None:
        raise ValueError(f"class weights json missing key: {key}")
    if len(weights) != num_output_classes:
        raise ValueError(f"{key} length mismatch: weights={len(weights)}, num_output_classes={num_output_classes}")
    tensor = torch_module.as_tensor(weights, dtype=torch_module.float32, device=device)
    return tensor, {"path": str(weight_path), "key": key, "num_weights": int(tensor.numel())}


def map_output_predictions_to_lidarseg(pred_indices: np.ndarray, student_output_space: str, class_split) -> np.ndarray:
    if student_output_space == "all_lidarseg":
        return pred_indices.astype(np.int64)
    mapped = np.full(pred_indices.shape, -1, dtype=np.int64)
    train_to_label = np.asarray(class_split.train_id_to_label_id, dtype=np.int64)
    valid = (pred_indices >= 0) & (pred_indices < train_to_label.shape[0])
    mapped[valid] = train_to_label[pred_indices[valid]]
    return mapped


def finite_float_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def evaluate_model_on_loader(
    model,
    loader,
    torch_module,
    device,
    class_split,
    ce_label_key: str,
    student_output_space: str,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    num_classes = class_split.num_classes
    total_intersections = np.zeros(num_classes, dtype=np.int64)
    total_unions = np.zeros(num_classes, dtype=np.int64)
    total_gt_counts = np.zeros(num_classes, dtype=np.int64)
    total_points = 0
    total_valid_pred_points = 0

    with torch_module.no_grad():
        for batch in loader:
            torch_batch = numpy_batch_to_torch(batch, torch_module, device)
            outputs = model(torch_batch)
            model_valid_mask = outputs.get(
                "model_valid_mask",
                torch_module.ones_like(torch_batch[ce_label_key], dtype=torch_module.bool),
            )
            pred_output = torch_module.argmax(outputs["logits"], dim=1).detach().cpu().numpy().astype(np.int64)
            model_valid_np = model_valid_mask.detach().cpu().numpy().astype(bool)
            pred_labels = map_output_predictions_to_lidarseg(pred_output, student_output_space, class_split)
            pred_labels[~model_valid_np] = -1

            if student_output_space == "all_lidarseg":
                gt_labels = torch_batch["all_class_train_labels"].detach().cpu().numpy().astype(np.int64)
            else:
                gt_labels = np.asarray(batch["raw_labels"], dtype=np.int64)
                valid_base = torch_batch["train_labels"].detach().cpu().numpy().astype(np.int64) != IGNORE_INDEX
                gt_labels[~valid_base] = IGNORE_INDEX
            valid_gt_mask = gt_labels != IGNORE_INDEX
            intersections, unions, gt_counts = segmentation_intersections_unions(
                pred_labels=pred_labels,
                gt_labels=gt_labels,
                num_classes=num_classes,
                valid_gt_mask=valid_gt_mask,
            )
            total_intersections += intersections
            total_unions += unions
            total_gt_counts += gt_counts
            total_points += int(gt_labels.shape[0])
            total_valid_pred_points += int(np.sum((pred_labels >= 0) & (pred_labels < num_classes)))

    if was_training:
        model.train()
    ious = safe_iou(total_intersections, total_unions)
    eval_ids = np.asarray(class_split.base_label_ids, dtype=np.int64)
    all_miou = mean_iou_for_ids(ious, eval_ids)
    base_miou = mean_iou_for_ids(ious, class_split.base_label_ids)
    novel_miou = (
        float("nan")
        if class_split.novel_label_ids.shape[0] == 0
        else mean_iou_for_ids(ious, class_split.novel_label_ids)
    )
    return {
        "all_miou": finite_float_or_none(all_miou),
        "base_miou": finite_float_or_none(base_miou),
        "novel_miou": finite_float_or_none(novel_miou),
        "num_points": int(total_points),
        "num_valid_pred_points": int(total_valid_pred_points),
        "prediction_coverage": float(total_valid_pred_points / max(total_points, 1)),
        "per_class_iou": [
            None if not np.isfinite(float(value)) else float(value)
            for value in ious.tolist()
        ],
        "gt_counts": total_gt_counts.astype(int).tolist(),
    }


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
    if args.dense_temperature <= 0:
        raise ValueError("--dense_temperature must be positive.")
    if args.text_align_weight < 0:
        raise ValueError("--text_align_weight must be non-negative.")
    if args.lovasz_weight < 0 or args.dice_weight < 0:
        raise ValueError("--lovasz_weight and --dice_weight must be non-negative.")
    if args.feature_dim <= 0:
        raise ValueError("--feature_dim must be positive.")
    if args.data_source == "raw_lidarseg" and (args.distill_weight > 0.0 or args.dense_logit_weight > 0.0):
        raise ValueError(
            "raw_lidarseg reads only LiDAR/lidarseg and cannot use 2D feature or dense-logit distillation. "
            "Set --distill_weight 0.0 and --dense_logit_weight 0.0."
        )
    if args.augment and (args.distill_weight > 0.0 or args.dense_logit_weight > 0.0):
        raise ValueError(
            "--augment changes point coordinates and must not be combined with nonzero "
            "--distill_weight or --dense_logit_weight because 2D teacher signals are tied to original projections."
        )
    if args.eval_every < 0:
        raise ValueError("--eval_every must be >= 0.")
    if args.eval_every > 0 and (args.eval_start_idx is None or args.eval_max_samples <= 0):
        raise ValueError("--eval_every requires --eval_start_idx and --eval_max_samples > 0.")

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
    if args.data_source == "precomputed":
        missing_precomputed = find_missing_precomputed_files(
            sample_indices=sample_indices,
            point_feature_dir=args.point_feature_dir,
            reliability_dir=args.reliability_dir,
        )
        if missing_precomputed:
            missing_sample_indices = {int(item["sample_idx"]) for item in missing_precomputed}
            if args.skip_missing_precomputed:
                original_count = len(sample_indices)
                sample_indices = [idx for idx in sample_indices if idx not in missing_sample_indices]
                if not sample_indices:
                    raise FileNotFoundError(
                        "All requested samples are missing precomputed point features or reliability outputs."
                    )
                if main_process:
                    logger.warning(
                        "skipping %d/%d samples missing precomputed outputs; first_missing_sample=%s",
                        len(missing_precomputed),
                        original_count,
                        missing_precomputed[0]["sample_idx"],
                    )
            else:
                first_missing = missing_precomputed[0]
                raise FileNotFoundError(
                    "Training requires precomputed outputs from MVP-v1/v2 for every requested sample. "
                    f"Missing {len(missing_precomputed)} sample(s); first missing sample_idx="
                    f"{first_missing['sample_idx']}:\n{first_missing['missing_files']}\n"
                    "Either precompute that sample range first, lower --max_samples, pass "
                    "--skip_missing_precomputed for a smoke test, or use --data_source raw_lidarseg "
                    "for supervised closed-set training."
                )
    load_dense_logits = (
        args.data_source == "precomputed"
        and args.teacher_mode in {DENSE_LOGIT_DISTILL_MODE, HYBRID_TEACHER_MODE}
        and args.dense_logit_weight > 0.0
    )
    if load_dense_logits:
        missing_dense = find_missing_dense_point_files(
            sample_indices=sample_indices,
            dense_point_dir=args.dense_point_dir,
        )
        if missing_dense:
            first_missing = missing_dense[0]
            raise FileNotFoundError(
                "Dense-logit distillation requires V6 dense point logits for every requested sample. "
                f"Missing {len(missing_dense)} sample(s); first missing sample_idx="
                f"{first_missing['sample_idx']}:\n{first_missing['missing_files']}\n"
                "Precompute V6 first or lower --max_samples/--sample_idx."
            )
    class_split = build_class_split(args.class_names_path, args.split_config)
    base_text_prototypes = build_text_prototypes(args, class_split.base_class_names, torch, device)
    student_output_space = args.student_output_space
    if student_output_space == "auto":
        student_output_space = "all_lidarseg" if load_dense_logits else "base"
    num_output_classes = (
        class_split.num_classes if student_output_space == "all_lidarseg" else class_split.num_train_classes
    )
    ce_label_key = "all_class_train_labels" if student_output_space == "all_lidarseg" else "train_labels"
    max_points = None if args.max_points <= 0 else args.max_points
    augmentation_config = PointAugmentationConfig(
        rotation_z_max_rad=float(args.aug_rotation_z),
        flip_x_prob=float(args.aug_flip_x_prob),
        flip_y_prob=float(args.aug_flip_y_prob),
        scale_min=float(args.aug_scale_min),
        scale_max=float(args.aug_scale_max),
        dropout_prob=float(args.aug_dropout_prob),
    )
    if args.data_source == "raw_lidarseg":
        train_dataset = RawLidarsegDataset(
            nuscenes_dataset=dataset,
            sample_indices=sample_indices,
            class_split=class_split,
            max_points=max_points,
            seed=args.seed,
            ignore_index=IGNORE_INDEX,
            augment=args.augment,
            augmentation_config=augmentation_config,
            feature_dim=args.feature_dim,
        )
    else:
        train_dataset = PrecomputedPointFeatureDataset(
            nuscenes_dataset=dataset,
            sample_indices=sample_indices,
            point_feature_dir=args.point_feature_dir,
            reliability_dir=args.reliability_dir,
            class_split=class_split,
            dense_point_dir=args.dense_point_dir,
            load_dense_logits=load_dense_logits,
            dense_logit_space=student_output_space,
            max_points=max_points,
            seed=args.seed,
            ignore_index=IGNORE_INDEX,
            augment=args.augment,
            augmentation_config=augmentation_config,
        )

    eval_loader = None
    eval_sample_indices: list[int] = []
    if args.eval_every > 0:
        eval_sample_indices = dataset.resolve_sample_indices(
            sample_idx=None,
            start_idx=int(args.eval_start_idx),
            max_samples=int(args.eval_max_samples),
        )
        if args.data_source == "precomputed":
            missing_eval_precomputed = find_missing_precomputed_files(
                sample_indices=eval_sample_indices,
                point_feature_dir=args.point_feature_dir,
                reliability_dir=args.reliability_dir,
            )
            if missing_eval_precomputed:
                first_missing = missing_eval_precomputed[0]
                raise FileNotFoundError(
                    "In-training eval requires precomputed point features/reliability for eval samples. "
                    f"Missing {len(missing_eval_precomputed)} sample(s); first missing sample_idx="
                    f"{first_missing['sample_idx']}:\n{first_missing['missing_files']}"
                )
        if args.data_source == "raw_lidarseg":
            eval_dataset = RawLidarsegDataset(
                nuscenes_dataset=dataset,
                sample_indices=eval_sample_indices,
                class_split=class_split,
                max_points=max_points,
                seed=args.seed,
                ignore_index=IGNORE_INDEX,
                augment=False,
                feature_dim=args.feature_dim,
            )
        else:
            eval_dataset = PrecomputedPointFeatureDataset(
                nuscenes_dataset=dataset,
                sample_indices=eval_sample_indices,
                point_feature_dir=args.point_feature_dir,
                reliability_dir=args.reliability_dir,
                class_split=class_split,
                dense_point_dir=args.dense_point_dir,
                load_dense_logits=False,
                dense_logit_space=student_output_space,
                max_points=max_points,
                seed=args.seed,
                ignore_index=IGNORE_INDEX,
                augment=False,
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
    if args.eval_every > 0:
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=None,
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
        num_classes=num_output_classes,
        voxel_size=tuple(args.voxel_size),
        point_cloud_range=tuple(args.point_cloud_range),
        sparse_base_channels=args.sparse_base_channels,
    ).to(device)
    if args.init_checkpoint is not None:
        init_checkpoint_path = Path(args.init_checkpoint).expanduser().resolve()
        if not init_checkpoint_path.exists():
            raise FileNotFoundError(f"init checkpoint not found: {init_checkpoint_path}")
        init_checkpoint = torch_load_checkpoint(torch, init_checkpoint_path)
        model.load_state_dict(init_checkpoint["model_state_dict"], strict=True)
        if main_process:
            logger.info("initialized model from checkpoint: %s", init_checkpoint_path)
    if distributed_state["distributed"]:
        device_ids = [int(distributed_state["local_rank"])] if device.type == "cuda" else None
        model = DistributedDataParallel(
            model,
            device_ids=device_ids,
            find_unused_parameters=(args.backbone in SPCONV_BACKBONES),
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    use_feature_distill = args.teacher_mode in {FEATURE_DISTILL_MODE, HYBRID_TEACHER_MODE} and args.distill_weight > 0.0
    use_dense_logit_distill = (
        args.teacher_mode in {DENSE_LOGIT_DISTILL_MODE, HYBRID_TEACHER_MODE} and args.dense_logit_weight > 0.0
    )
    use_text_align = bool(args.text_align_weight > 0.0 and base_text_prototypes is not None)
    class_weights, class_weight_info = load_class_weights(args, num_output_classes, torch, device)
    best_eval_miou = -math.inf
    best_checkpoint_path = ""
    best_eval_metrics: dict[str, Any] = {}

    if main_process:
        logger.info(
            (
                "train start | backbone=%s | role=%s | data_source=%s | teacher_mode=%s | output_space=%s | "
                "text_align=%s | version=%s | samples=%d | device=%s | ddp=%s | world_size=%d"
            ),
            backbone_spec.backbone,
            backbone_spec.role,
            args.data_source,
            args.teacher_mode,
            student_output_space,
            use_text_align,
            args.version,
            len(sample_indices),
            device,
            distributed_state["distributed"],
            distributed_state["world_size"],
        )
        if backbone_spec.is_debug_model:
            logger.warning("debug backbone in use: %s", backbone_spec.description)
        if class_weight_info:
            logger.info("class weights enabled | %s", class_weight_info)
        if args.augment:
            logger.info("point augmentation enabled | %s", augmentation_config)

    epoch_logs: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        local_stats = {
            "steps": 0.0,
            "points": 0.0,
            "base_supervised_points": 0.0,
            "distill_points": 0.0,
            "dense_distill_points": 0.0,
            "total_loss_sum": 0.0,
            "ce_loss_sum": 0.0,
            "lovasz_loss_sum": 0.0,
            "dice_loss_sum": 0.0,
            "distill_loss_sum": 0.0,
            "dense_logit_loss_sum": 0.0,
            "text_align_loss_sum": 0.0,
            "grad_norm_sum": 0.0,
        }

        for batch in loader:
            torch_batch = numpy_batch_to_torch(batch, torch, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(torch_batch)
                model_valid_mask = outputs.get(
                    "model_valid_mask",
                    torch.ones_like(torch_batch[ce_label_key], dtype=torch.bool),
                )
                effective_train_labels = torch_batch[ce_label_key].clone()
                effective_train_labels[~model_valid_mask] = IGNORE_INDEX
                ce_loss = supervised_ce_loss(
                    outputs["logits"],
                    effective_train_labels,
                    ignore_index=IGNORE_INDEX,
                    class_weights=class_weights,
                )
                if args.lovasz_weight > 0.0:
                    lovasz_loss = lovasz_softmax_loss(
                        outputs["logits"],
                        effective_train_labels,
                        ignore_index=IGNORE_INDEX,
                    )
                else:
                    lovasz_loss = outputs["logits"].sum() * 0.0
                if args.dice_weight > 0.0:
                    supervised_dice_loss = dice_loss(
                        outputs["logits"],
                        effective_train_labels,
                        ignore_index=IGNORE_INDEX,
                    )
                else:
                    supervised_dice_loss = outputs["logits"].sum() * 0.0
                if use_feature_distill:
                    distill_loss = cosine_distillation_loss(
                        student_features=outputs["point_features"],
                        teacher_features=torch_batch["teacher_features"],
                        weights=torch_batch["reliability_weight"],
                        valid_mask=torch_batch["teacher_valid_mask"] & model_valid_mask,
                    )
                else:
                    distill_loss = outputs["point_features"].sum() * 0.0

                if use_dense_logit_distill:
                    dense_weights = torch_batch["reliability_weight"] * torch_batch["dense_teacher_confidence"]
                    dense_logit_loss = dense_logit_distillation_loss(
                        student_logits=outputs["logits"],
                        teacher_logits=torch_batch["dense_teacher_logits"],
                        weights=dense_weights,
                        valid_mask=torch_batch["dense_teacher_valid_mask"] & model_valid_mask,
                        temperature=args.dense_temperature,
                    )
                else:
                    dense_logit_loss = outputs["logits"].sum() * 0.0

                if use_text_align:
                    text_align_loss = text_prototype_alignment_loss(
                        student_features=outputs["point_features"],
                        train_labels=torch_batch["train_labels"],
                        text_prototypes=base_text_prototypes,
                        valid_mask=model_valid_mask,
                        ignore_index=IGNORE_INDEX,
                    )
                else:
                    text_align_loss = outputs["point_features"].sum() * 0.0

                total_loss = (
                    args.ce_weight * ce_loss
                    + args.lovasz_weight * lovasz_loss
                    + args.dice_weight * supervised_dice_loss
                    + args.distill_weight * distill_loss
                    + args.dense_logit_weight * dense_logit_loss
                    + args.text_align_weight * text_align_loss
                )

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = compute_grad_norm(model)
            scaler.step(optimizer)
            scaler.update()

            output_valid_mask = outputs.get(
                "model_valid_mask",
                torch.ones_like(torch_batch[ce_label_key], dtype=torch.bool),
            )
            base_points = int(
                torch.sum((torch_batch[ce_label_key] != IGNORE_INDEX) & output_valid_mask).detach().cpu().item()
            )
            distill_points = int(
                torch.sum(
                    torch_batch["teacher_valid_mask"]
                    & output_valid_mask
                    & torch.isfinite(torch_batch["reliability_weight"])
                    & (torch_batch["reliability_weight"] > 0.0)
                )
                .detach()
                .cpu()
                .item()
            )
            if not use_feature_distill:
                distill_points = 0
            dense_distill_points = int(
                torch.sum(
                    torch_batch["dense_teacher_valid_mask"]
                    & output_valid_mask
                    & torch.isfinite(torch_batch["reliability_weight"])
                    & torch.isfinite(torch_batch["dense_teacher_confidence"])
                    & ((torch_batch["reliability_weight"] * torch_batch["dense_teacher_confidence"]) > 0.0)
                )
                .detach()
                .cpu()
                .item()
            )
            if not use_dense_logit_distill:
                dense_distill_points = 0
            num_points = int(torch_batch["point_xyz"].shape[0])

            local_stats["steps"] += 1.0
            local_stats["points"] += float(num_points)
            local_stats["base_supervised_points"] += float(base_points)
            local_stats["distill_points"] += float(distill_points)
            local_stats["dense_distill_points"] += float(dense_distill_points)
            local_stats["total_loss_sum"] += float(total_loss.detach().cpu().item())
            local_stats["ce_loss_sum"] += float(ce_loss.detach().cpu().item())
            local_stats["lovasz_loss_sum"] += float(lovasz_loss.detach().cpu().item())
            local_stats["dice_loss_sum"] += float(supervised_dice_loss.detach().cpu().item())
            local_stats["distill_loss_sum"] += float(distill_loss.detach().cpu().item())
            local_stats["dense_logit_loss_sum"] += float(dense_logit_loss.detach().cpu().item())
            local_stats["text_align_loss_sum"] += float(text_align_loss.detach().cpu().item())
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
            "dense_distill_points": int(epoch_stats["dense_distill_points"]),
            "avg_total_loss": float(epoch_stats["avg_total_loss"]),
            "avg_ce_loss": float(epoch_stats["avg_ce_loss"]),
            "avg_lovasz_loss": float(epoch_stats["avg_lovasz_loss"]),
            "avg_dice_loss": float(epoch_stats["avg_dice_loss"]),
            "avg_distill_loss": float(epoch_stats["avg_distill_loss"]),
            "avg_dense_logit_loss": float(epoch_stats["avg_dense_logit_loss"]),
            "avg_text_align_loss": float(epoch_stats["avg_text_align_loss"]),
            "avg_grad_norm": float(epoch_stats["avg_grad_norm"]),
        }
        epoch_logs.append(epoch_log)

        if main_process:
            logger.info(
                (
                    "epoch=%d | loss=%.6f | ce=%.6f | lovasz=%.6f | dice=%.6f | "
                    "feature_distill=%.6f | dense_logit=%.6f | text_align=%.6f | "
                    "points=%d | base_points=%d | feature_points=%d | dense_points=%d"
                ),
                epoch,
                epoch_log["avg_total_loss"],
                epoch_log["avg_ce_loss"],
                epoch_log["avg_lovasz_loss"],
                epoch_log["avg_dice_loss"],
                epoch_log["avg_distill_loss"],
                epoch_log["avg_dense_logit_loss"],
                epoch_log["avg_text_align_loss"],
                epoch_log["points"],
                epoch_log["base_supervised_points"],
                epoch_log["distill_points"],
                epoch_log["dense_distill_points"],
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
        if eval_loader is not None and epoch % args.eval_every == 0:
            eval_metrics = evaluate_model_on_loader(
                model=model,
                loader=eval_loader,
                torch_module=torch,
                device=device,
                class_split=class_split,
                ce_label_key=ce_label_key,
                student_output_space=student_output_space,
            )
            epoch_log["eval"] = eval_metrics
            all_miou_value = eval_metrics.get("all_miou")
            current_miou = float(all_miou_value) if all_miou_value is not None else float("nan")
            if main_process:
                logger.info(
                    "eval epoch=%d | all_miou=%s | base_miou=%s | novel_miou=%s | coverage=%.6f",
                    epoch,
                    eval_metrics.get("all_miou"),
                    eval_metrics.get("base_miou"),
                    eval_metrics.get("novel_miou"),
                    float(eval_metrics.get("prediction_coverage", 0.0)),
                )
                if math.isfinite(current_miou) and current_miou > best_eval_miou:
                    best_eval_miou = current_miou
                    best_checkpoint = output_dir / f"{args.backbone}_best.pt"
                    best_eval_metrics = dict(eval_metrics)
                    best_eval_metrics["epoch"] = epoch
                    best_checkpoint_path = str(best_checkpoint)
                    save_checkpoint(
                        best_checkpoint,
                        torch,
                        model,
                        optimizer,
                        epoch,
                        args,
                        class_split,
                        best_eval_metrics,
                        sample_indices,
                        distributed_state,
                        backbone_spec,
                    )
                    logger.info("new best checkpoint | epoch=%d | all_miou=%.6f | path=%s", epoch, current_miou, best_checkpoint)

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
            "data_source": args.data_source,
            "teacher_mode": args.teacher_mode,
            "student_output_space": student_output_space,
            "num_samples": len(sample_indices),
            "sample_indices": sample_indices,
            "epochs_completed": args.epochs,
            "batch_size_per_process": args.batch_size,
            "max_points_per_sample": max_points,
            "voxel_size": list(args.voxel_size),
            "point_cloud_range": list(args.point_cloud_range),
            "sparse_base_channels": args.sparse_base_channels,
            "feature_dim": feature_dim,
            "dense_point_dir": str(Path(args.dense_point_dir).expanduser().resolve()),
            "init_checkpoint": str(Path(args.init_checkpoint).expanduser().resolve()) if args.init_checkpoint else "",
            "load_dense_logits": load_dense_logits,
            "dense_temperature": args.dense_temperature,
            "num_output_classes": num_output_classes,
            "num_base_train_classes": class_split.num_train_classes,
            "base_classes": class_split.base_class_names,
            "novel_classes": class_split.novel_class_names,
            "ignore_classes": class_split.ignore_class_names,
            "first_sample_raw_label_hist": raw_hist,
            "loss_weights": {
                "ce_weight": args.ce_weight,
                "lovasz_weight": args.lovasz_weight,
                "dice_weight": args.dice_weight,
                "distill_weight": args.distill_weight,
                "dense_logit_weight": args.dense_logit_weight,
                "text_align_weight": args.text_align_weight,
            },
            "class_weights": class_weight_info,
            "augmentation": {
                "enabled": args.augment,
                "rotation_z_max_rad": args.aug_rotation_z,
                "flip_x_prob": args.aug_flip_x_prob,
                "flip_y_prob": args.aug_flip_y_prob,
                "scale_min": args.aug_scale_min,
                "scale_max": args.aug_scale_max,
                "dropout_prob": args.aug_dropout_prob,
            },
            "eval_during_training": {
                "enabled": eval_loader is not None,
                "eval_every": args.eval_every,
                "eval_sample_indices": eval_sample_indices,
                "best_eval_miou": None if not math.isfinite(best_eval_miou) else float(best_eval_miou),
                "best_eval_metrics": best_eval_metrics,
                "best_checkpoint": best_checkpoint_path,
            },
            "text_alignment": {
                "enabled": use_text_align,
                "text_model_name": args.text_model_name,
                "text_prompt_template": args.text_prompt_template,
                "cache_dir": args.cache_dir or "",
                "local_files_only": args.local_files_only,
                "num_text_prototypes": int(base_text_prototypes.shape[0]) if base_text_prototypes is not None else 0,
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
