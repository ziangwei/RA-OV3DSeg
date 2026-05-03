from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class PointMLP(nn.Module):
    """Minimal point-wise MLP for MVP-v3 dry-run.

    This is not the final 3D backbone. It only verifies that the training inputs, label mapping,
    CE loss, and reliability-weighted distillation loss are wired correctly.
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 128,
        feature_dim: int = 512,
        num_classes: int = 14,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.feature_head = nn.Linear(hidden_dim, feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, batch_or_point_xyz) -> dict[str, torch.Tensor]:
        if isinstance(batch_or_point_xyz, dict):
            point_xyz = batch_or_point_xyz["point_xyz"]
        else:
            point_xyz = batch_or_point_xyz

        hidden = self.backbone(point_xyz)
        point_features = F.normalize(self.feature_head(hidden), dim=-1)
        logits = self.classifier(point_features)
        return {
            "point_features": point_features,
            "logits": logits,
            "model_valid_mask": torch.ones(point_xyz.shape[0], dtype=torch.bool, device=point_xyz.device),
        }
