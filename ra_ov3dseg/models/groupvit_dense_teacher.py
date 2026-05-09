from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ra_ov3dseg.models.image_encoder import _resolve_device
from ra_ov3dseg.models.text_encoder import prettify_label_name


class GroupViTDenseTeacher:
    """Transformers-native GroupViT zero-shot dense segmentation teacher.

    This is the pragmatic V12 path: it keeps teacher extraction inside the
    existing RA-OV3DSeg environment. GroupViT supports zero-shot segmentation
    over arbitrary text categories via `output_segmentation=True`.
    """

    def __init__(
        self,
        model_name: str = "nvidia/groupvit-gcc-yfcc",
        device: str = "auto",
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        output_height: int = 0,
        output_width: int = 0,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = str(Path(cache_dir).expanduser().resolve()) if cache_dir is not None else None
        self.local_files_only = local_files_only
        self.output_height = int(output_height)
        self.output_width = int(output_width)

        try:
            import torch
            import torch.nn.functional as F
            from transformers import AutoProcessor, GroupViTModel
        except ImportError as exc:
            raise ImportError(
                "GroupViTDenseTeacher requires torch and transformers. It should run in the existing "
                "ra-ov3dseg environment after installing requirements.txt."
            ) from exc

        self.torch = torch
        self.functional = F
        self.device = _resolve_device(torch, device)
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            local_files_only=local_files_only,
        )
        try:
            self.model = GroupViTModel.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                local_files_only=local_files_only,
                use_safetensors=True,
            )
        except (OSError, ValueError):
            self.model = GroupViTModel.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                local_files_only=local_files_only,
            )
        self.model.eval()
        self.model.to(self.device)

    def build_prompts(self, class_names: list[str], prompt_template: str | None = None) -> list[str]:
        if prompt_template is None:
            return [prettify_label_name(class_name) for class_name in class_names]
        return [prompt_template.format(prettify_label_name(class_name)) for class_name in class_names]

    def encode_image_logits(
        self,
        image_path: str | Path,
        class_names: list[str],
        prompt_template: str | None = None,
        prompt_batch_size: int = 0,
    ) -> dict[str, Any]:
        del prompt_batch_size  # GroupViT consumes all query classes in one forward pass.

        image_path = Path(image_path)
        image = Image.open(image_path).convert("RGB")
        prompts = self.build_prompts(class_names, prompt_template=prompt_template)
        inputs = self.processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model(**inputs, output_segmentation=True, return_dict=True)
            if not hasattr(outputs, "segmentation_logits") or outputs.segmentation_logits is None:
                raise RuntimeError(f"GroupViT model `{self.model_name}` did not return segmentation_logits.")
            logits = outputs.segmentation_logits

        if logits.ndim != 4:
            raise RuntimeError(f"Expected GroupViT segmentation logits with shape (B,C,H,W), got {tuple(logits.shape)}")
        logits = logits[0].float()
        if self.output_height > 0 and self.output_width > 0:
            logits = self.functional.interpolate(
                logits.unsqueeze(0),
                size=(self.output_height, self.output_width),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        logits_np = logits.detach().cpu().numpy().astype(np.float32)
        return {
            "dense_logits": logits_np,
            "prompts": prompts,
            "metadata": {
                "original_width": int(image.width),
                "original_height": int(image.height),
                "logit_width": int(logits_np.shape[-1]),
                "logit_height": int(logits_np.shape[-2]),
                "num_classes": int(logits_np.shape[0]),
                "model_name": self.model_name,
            },
        }
