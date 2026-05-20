from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ra_ov3dseg.models.text_encoder import prettify_label_name, resolve_device


def _normalize_torch(torch_module, tensor, eps: float = 1e-6):
    return tensor / torch_module.clamp(torch_module.linalg.norm(tensor, dim=-1, keepdim=True), min=eps)


def _pad_xywh_bbox(
    bbox: list[float],
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    x, y, w, h = bbox
    pad = padding_ratio * max(float(w), float(h))
    x0 = max(int(np.floor(x - pad)), 0)
    y0 = max(int(np.floor(y - pad)), 0)
    x1 = min(int(np.ceil(x + w + pad)), image_width)
    y1 = min(int(np.ceil(y + h + pad)), image_height)
    if x1 <= x0 or y1 <= y0:
        return 0, 0, image_width, image_height
    return x0, y0, x1, y1


def _resize_channel_first(array: np.ndarray, output_height: int, output_width: int) -> np.ndarray:
    if output_height <= 0 or output_width <= 0:
        return array
    channels = []
    for channel in array:
        image = Image.fromarray(channel.astype(np.float32), mode="F")
        resized = image.resize((output_width, output_height), resample=Image.Resampling.BILINEAR)
        channels.append(np.asarray(resized, dtype=np.float32))
    return np.stack(channels, axis=0)


class SAM2SigLIPTeacher:
    """Mask-then-classify teacher: SAM2 automatic masks classified by SigLIP."""

    def __init__(
        self,
        sam_model_id: str = "facebook/sam2.1-hiera-small",
        siglip_model_name: str = "google/siglip-base-patch16-224",
        device: str = "auto",
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        points_per_side: int = 24,
        points_per_batch: int = 64,
        pred_iou_thresh: float = 0.80,
        stability_score_thresh: float = 0.92,
        min_mask_region_area: int = 100,
        crop_padding_ratio: float = 0.10,
        classification_batch_size: int = 16,
        logit_temperature: float = 0.07,
        output_height: int = 450,
        output_width: int = 800,
        background_class_name: str = "background",
    ) -> None:
        try:
            import torch
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2_hf
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "SAM2SigLIPTeacher requires `sam2`, `torch`, and `transformers`. "
                "Install SAM2 separately in a compatible environment before running Stage 3 extraction."
            ) from exc

        self.torch = torch
        self.device = resolve_device(torch, device)
        self.sam_model_id = sam_model_id
        self.siglip_model_name = siglip_model_name
        self.cache_dir = str(Path(cache_dir).expanduser().resolve()) if cache_dir is not None else None
        self.local_files_only = local_files_only
        self.crop_padding_ratio = float(crop_padding_ratio)
        self.classification_batch_size = int(classification_batch_size)
        self.logit_temperature = float(logit_temperature)
        self.output_height = int(output_height)
        self.output_width = int(output_width)
        self.background_class_name = background_class_name

        sam_model = build_sam2_hf(sam_model_id, device=str(self.device))
        self.mask_generator = SAM2AutomaticMaskGenerator(
            sam_model,
            points_per_side=points_per_side,
            points_per_batch=points_per_batch,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
            output_mode="binary_mask",
        )
        self.processor = AutoProcessor.from_pretrained(
            siglip_model_name,
            cache_dir=self.cache_dir,
            local_files_only=local_files_only,
        )
        self.siglip = AutoModel.from_pretrained(
            siglip_model_name,
            cache_dir=self.cache_dir,
            local_files_only=local_files_only,
            use_safetensors=True,
        )
        if not hasattr(self.siglip, "get_image_features") or not hasattr(self.siglip, "get_text_features"):
            raise ValueError(f"Model `{siglip_model_name}` must expose get_image_features and get_text_features.")
        self.siglip.eval()
        self.siglip.to(self.device)

    def build_prompts(self, class_names: list[str], prompt_template: str) -> list[str]:
        return [prompt_template.format(prettify_label_name(class_name)) for class_name in class_names]

    def encode_text_prototypes(self, class_names: list[str], prompt_template: str):
        prompts = self.build_prompts(class_names, prompt_template)
        inputs = self.processor(text=prompts, padding=True, truncation=True, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            features = self.siglip.get_text_features(**inputs).float()
            features = _normalize_torch(self.torch, features)
        return features, prompts

    def _masked_crop(self, image_np: np.ndarray, mask: np.ndarray, bbox: list[float]) -> Image.Image:
        height, width = image_np.shape[:2]
        x0, y0, x1, y1 = _pad_xywh_bbox(bbox, width, height, self.crop_padding_ratio)
        crop = image_np[y0:y1, x0:x1].copy()
        crop_mask = mask[y0:y1, x0:x1].astype(bool)
        if crop_mask.shape[:2] != crop.shape[:2]:
            raise ValueError("crop mask shape mismatch")
        mean_color = crop[crop_mask].mean(axis=0) if np.any(crop_mask) else crop.reshape(-1, 3).mean(axis=0)
        crop[~crop_mask] = mean_color.astype(crop.dtype)
        return Image.fromarray(crop, mode="RGB")

    def _encode_crops(self, crops: list[Image.Image]):
        image_features = []
        for start in range(0, len(crops), self.classification_batch_size):
            crop_batch = crops[start : start + self.classification_batch_size]
            inputs = self.processor(images=crop_batch, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self.torch.no_grad():
                features = self.siglip.get_image_features(**inputs).float()
                features = _normalize_torch(self.torch, features)
            image_features.append(features)
        if not image_features:
            return None
        return self.torch.cat(image_features, dim=0)

    def encode_image_logits(
        self,
        image_path: str | Path,
        class_names: list[str],
        prompt_template: str = "a photo of a {}",
    ) -> dict[str, Any]:
        image_path = Path(image_path)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            original_width, original_height = image.size
            image_np = np.asarray(image, dtype=np.uint8)

        output_class_names = list(class_names) + [self.background_class_name]
        text_features, prompts = self.encode_text_prototypes(output_class_names, prompt_template)
        masks = self.mask_generator.generate(image_np)
        masks = sorted(masks, key=lambda item: float(item.get("area", 0.0)), reverse=True)

        num_classes = len(output_class_names)
        dense_logits = np.full((num_classes, original_height, original_width), -10.0, dtype=np.float32)
        dense_logits[-1] = 10.0
        dense_confidence = np.zeros((original_height, original_width), dtype=np.float32)
        dense_mask_area = np.zeros((original_height, original_width), dtype=np.float32)
        dense_pred_label = np.full((original_height, original_width), num_classes - 1, dtype=np.int16)

        crops = [self._masked_crop(image_np, ann["segmentation"].astype(bool), ann["bbox"]) for ann in masks]
        image_features = self._encode_crops(crops)
        mask_logits = []
        if image_features is not None:
            sims = image_features @ text_features.T
            logits = sims / max(self.logit_temperature, 1e-6)
            mask_logits = logits.detach().cpu().numpy().astype(np.float32)

        for ann, logits in zip(masks, mask_logits):
            mask = ann["segmentation"].astype(bool)
            if not np.any(mask):
                continue
            pred_idx = int(np.argmax(logits))
            confidence = float(self.torch.softmax(self.torch.from_numpy(logits), dim=0).max().item())
            dense_logits[:, mask] = logits[:, None]
            dense_confidence[mask] = confidence
            dense_mask_area[mask] = float(ann.get("area", int(mask.sum())))
            dense_pred_label[mask] = pred_idx

        dense_logits = _resize_channel_first(dense_logits, self.output_height, self.output_width)
        if self.output_height > 0 and self.output_width > 0:
            dense_confidence = np.asarray(
                Image.fromarray(dense_confidence, mode="F").resize(
                    (self.output_width, self.output_height),
                    resample=Image.Resampling.BILINEAR,
                ),
                dtype=np.float32,
            )
            dense_pred_label = np.asarray(
                Image.fromarray(dense_pred_label.astype(np.int32), mode="I").resize(
                    (self.output_width, self.output_height),
                    resample=Image.Resampling.NEAREST,
                ),
                dtype=np.int16,
            )

        return {
            "dense_logits": dense_logits.astype(np.float32),
            "dense_confidence": dense_confidence.astype(np.float32),
            "dense_pred_label": dense_pred_label.astype(np.int16),
            "prompts": prompts,
            "class_names": output_class_names,
            "metadata": {
                "image_path": str(image_path),
                "original_width": int(original_width),
                "original_height": int(original_height),
                "logit_width": int(dense_logits.shape[-1]),
                "logit_height": int(dense_logits.shape[-2]),
                "num_classes": int(dense_logits.shape[0]),
                "num_masks": int(len(masks)),
                "sam_model_id": self.sam_model_id,
                "siglip_model_name": self.siglip_model_name,
                "teacher_backend": "sam2_siglip",
                "teacher_feature_granularity": "dense_class_logits",
            },
        }
