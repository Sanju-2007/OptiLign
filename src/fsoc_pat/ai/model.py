"""Lightweight CNN model for beacon detection.

Architecture: BeaconNet — a small, fast CNN that predicts:
  1. presence: P(beacon visible) ∈ [0, 1]
  2. centroid: (u, v) pixel coordinates

Design constraints:
  - < 100K parameters (must run at >> 100 FPS)
  - Input: (1, H, W) grayscale normalized to [0, 1]
  - No pretrained weights — trained from scratch on synthetic data
  - Deterministic inference when seeded

Architecture:
  Input (1, 120, 160)  — downsampled 4x from 480x640
  → Conv2d(1, 16, 5, stride=2, pad=2) + BN + ReLU     → (16, 60, 80)
  → Conv2d(16, 32, 3, stride=2, pad=1) + BN + ReLU    → (32, 30, 40)
  → Conv2d(32, 64, 3, stride=2, pad=1) + BN + ReLU    → (64, 15, 20)
  → Conv2d(64, 64, 3, stride=2, pad=1) + BN + ReLU    → (64, 8, 10)
  → AdaptiveAvgPool2d(1)                                → (64, 1, 1)
  → Flatten                                              → (64,)
  → FC(64, 32) + ReLU
  → FC(32, 3)  → [presence_logit, u_norm, v_norm]
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    """BeaconNet configuration.

    Attributes:
        input_h: Input height (after downsampling).
        input_w: Input width (after downsampling).
        base_channels: Number of channels in first conv layer.
    """

    input_h: int = 120
    input_w: int = 160
    base_channels: int = 16


class BeaconNet(nn.Module):
    """Lightweight CNN for beacon detection.

    Outputs:
        presence_logit: Scalar logit for P(beacon present).
        u_norm: Normalized u ∈ [0, 1] (multiply by image width).
        v_norm: Normalized v ∈ [0, 1] (multiply by image height).
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        cfg = config or ModelConfig()
        self._config = cfg

        c = cfg.base_channels

        self.features = nn.Sequential(
            nn.Conv2d(1, c, 5, stride=2, padding=2),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),

            nn.Conv2d(c, c * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(c * 4, c * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )

        self.head = nn.Sequential(
            nn.Linear(c * 4, c * 2),
            nn.ReLU(inplace=True),
            nn.Linear(c * 2, 3),  # [presence_logit, u_norm, v_norm]
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor (B, 1, H, W) float32 in [0, 1].

        Returns:
            Dictionary with 'presence_logit', 'u_norm', 'v_norm'.
        """
        feat = self.features(x).flatten(1)
        out = self.head(feat)

        return {
            "presence_logit": out[:, 0],
            "u_norm": torch.sigmoid(out[:, 1]),
            "v_norm": torch.sigmoid(out[:, 2]),
        }

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BeaconNetLoss(nn.Module):
    """Combined loss for presence classification + centroid regression.

    Loss = λ_cls * BCE(presence) + λ_reg * MSE(centroid) * presence_mask

    Centroid loss is only computed for samples where the beacon is present.
    """

    def __init__(
        self,
        cls_weight: float = 1.0,
        reg_weight: float = 5.0,
    ) -> None:
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        target_present: torch.Tensor,
        target_u: torch.Tensor,
        target_v: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            pred: Model output dict.
            target_present: (B,) float32, 1.0 if beacon present.
            target_u: (B,) float32, normalized u in [0, 1].
            target_v: (B,) float32, normalized v in [0, 1].

        Returns:
            Dict with 'total', 'cls', 'reg' losses.
        """
        # Classification loss
        cls_loss = F.binary_cross_entropy_with_logits(
            pred["presence_logit"], target_present,
        )

        # Regression loss (only where beacon present)
        mask = target_present > 0.5
        if mask.any():
            u_loss = F.mse_loss(pred["u_norm"][mask], target_u[mask])
            v_loss = F.mse_loss(pred["v_norm"][mask], target_v[mask])
            reg_loss = u_loss + v_loss
        else:
            reg_loss = torch.tensor(0.0, device=cls_loss.device)

        total = self.cls_weight * cls_loss + self.reg_weight * reg_loss

        return {"total": total, "cls": cls_loss, "reg": reg_loss}
