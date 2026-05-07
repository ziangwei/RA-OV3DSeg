from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ra_ov3dseg.models.text_encoder import prettify_label_name


def _resolve_device(torch_module, device: str) -> str:
    if device == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    return device


class CLIPSegDenseTeacher:
    """Dense open-vocabulary teacher using CLIPSeg class prompts.

    CLIPSeg predicts one binary mask/logit map per text prompt. We run all class
    prompts for each camera image and store class-wise dense logits. The logits are
    later sampled at projected LiDAR point locations.
    """

    def __init__(
        self,
        model_name: str = "CIDAS/clipseg-rd64-refined",
        device: str = "auto",
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = str(Path(cache_dir).expanduser().resolve()) if cache_dir is not None else None
        self.local_files_only = local_files_only

        try:
            import torch
            from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
        except ImportError as exc:
            raise ImportError(
                "CLIPSegDenseTeacher requires torch and transformers. "
                "Install the project requirements and a CUDA-matched PyTorch build."
            ) from exc

        self.torch = torch
        self.device = _resolve_device(torch, device)
        self.processor = CLIPSegProcessor.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            local_files_only=local_files_only,
        )
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            local_files_only=local_files_only,
            use_safetensors=True,
        )
        self.model.eval()
        self.model.to(self.device)

    def build_prompts(self, class_names: list[str], prompt_template: str) -> list[str]:
        return [prompt_template.format(prettify_label_name(class_name)) for class_name in class_names]

    def encode_image_logits(
        self,
        image_path: str | Path,
        class_names: list[str],
        prompt_template: str = "a {} in a driving scene",
        prompt_batch_size: int = 8,
    ) -> dict[str, Any]:
        image_path = Path(image_path)
        prompts = self.build_prompts(class_names, prompt_template)
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            original_width, original_height = image.size
            logits_chunks = []

            for start in range(0, len(prompts), prompt_batch_size):
                prompt_chunk = prompts[start : start + prompt_batch_size]
                images = [image] * len(prompt_chunk)
                inputs = self.processor(
                    text=prompt_chunk,
                    images=images,
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in inputs.items()}
                with self.torch.no_grad():
                    outputs = self.model(**inputs)
                logits = outputs.logits
                if logits.ndim == 2:
                    logits = logits.unsqueeze(0)
                logits_chunks.append(logits.detach().float().cpu())

        dense_logits = self.torch.cat(logits_chunks, dim=0).numpy().astype(np.float32)
        metadata = {
            "image_path": str(image_path),
            "original_width": int(original_width),
            "original_height": int(original_height),
            "logit_height": int(dense_logits.shape[-2]),
            "logit_width": int(dense_logits.shape[-1]),
            "num_classes": int(dense_logits.shape[0]),
            "model_name": self.model_name,
            "cache_dir": self.cache_dir or "",
            "teacher_backend": "clipseg_dense",
            "teacher_feature_granularity": "dense_class_logits",
        }
        return {
            "dense_logits": dense_logits,
            "prompts": prompts,
            "metadata": metadata,
        }
