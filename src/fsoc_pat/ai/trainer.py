"""Training infrastructure for BeaconNet.

Provides a self-contained Trainer class that:
  1. Generates a synthetic dataset
  2. Creates PyTorch DataLoaders
  3. Trains BeaconNet with Adam optimizer
  4. Evaluates on validation set each epoch
  5. Saves best model checkpoint

Usage:
    trainer = Trainer(intrinsics, config)
    metrics = trainer.train()
    trainer.save_model("models/beaconnet_v1.pt")
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from fsoc_pat.ai.dataset import DatasetConfig, DatasetGenerator, Sample
from fsoc_pat.ai.model import BeaconNet, BeaconNetLoss, ModelConfig
from fsoc_pat.geometry.camera_model import CameraIntrinsics


class BeaconDataset(Dataset):
    """PyTorch Dataset wrapper for beacon detection samples."""

    def __init__(
        self,
        samples: list[Sample],
        input_h: int = 120,
        input_w: int = 160,
        image_w: int = 640,
        image_h: int = 480,
    ) -> None:
        self._samples = samples
        self._input_h = input_h
        self._input_w = input_w
        self._image_w = image_w
        self._image_h = image_h

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self._samples[idx]

        # Resize and normalize
        resized = cv2.resize(
            sample.image,
            (self._input_w, self._input_h),
            interpolation=cv2.INTER_AREA,
        )
        image = torch.from_numpy(resized.astype(np.float32) / 255.0).unsqueeze(0)

        present = 1.0 if sample.present else 0.0

        if sample.present and not math.isnan(sample.u):
            u_norm = sample.u / self._image_w
            v_norm = sample.v / self._image_h
        else:
            u_norm = 0.5
            v_norm = 0.5

        return {
            "image": image,
            "present": torch.tensor(present, dtype=torch.float32),
            "u_norm": torch.tensor(u_norm, dtype=torch.float32),
            "v_norm": torch.tensor(v_norm, dtype=torch.float32),
        }


@dataclass
class TrainConfig:
    """Training configuration.

    Attributes:
        num_samples: Number of training samples to generate.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        weight_decay: L2 regularization.
        model_config: Model architecture config.
        dataset_config: Dataset generation config.
        seed: Training seed.
    """

    num_samples: int = 500
    epochs: int = 10
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    model_config: ModelConfig | None = None
    dataset_config: DatasetConfig | None = None
    seed: int = 42


@dataclass
class TrainMetrics:
    """Training metrics per epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float  # Presence classification accuracy
    val_centroid_error_px: float  # Mean pixel error on positives


class Trainer:
    """End-to-end training pipeline for BeaconNet."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        config: TrainConfig | None = None,
    ) -> None:
        self._intrinsics = intrinsics
        self._config = config or TrainConfig()
        self._model: BeaconNet | None = None
        self._best_state: dict | None = None

        torch.manual_seed(self._config.seed)

    def train(self) -> list[TrainMetrics]:
        """Generate data, train model, return metrics.

        Returns:
            List of TrainMetrics for each epoch.
        """
        cfg = self._config

        # Generate dataset
        ds_config = cfg.dataset_config or DatasetConfig(
            num_samples=cfg.num_samples, seed=cfg.seed,
        )
        gen = DatasetGenerator(self._intrinsics, ds_config)
        all_samples = gen.generate()
        train_samples, val_samples, _ = gen.split(all_samples)

        model_config = cfg.model_config or ModelConfig()

        # Create datasets
        train_ds = BeaconDataset(
            train_samples,
            input_h=model_config.input_h,
            input_w=model_config.input_w,
            image_w=self._intrinsics.width,
            image_h=self._intrinsics.height,
        )
        val_ds = BeaconDataset(
            val_samples,
            input_h=model_config.input_h,
            input_w=model_config.input_w,
            image_w=self._intrinsics.width,
            image_h=self._intrinsics.height,
        )

        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

        # Model, loss, optimizer
        self._model = BeaconNet(model_config)
        criterion = BeaconNetLoss()
        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        # Training loop
        metrics: list[TrainMetrics] = []
        best_val_loss = float("inf")

        for epoch in range(cfg.epochs):
            # Train
            self._model.train()
            total_train_loss = 0.0
            n_train = 0

            for batch in train_loader:
                optimizer.zero_grad()
                pred = self._model(batch["image"])
                losses = criterion(
                    pred, batch["present"], batch["u_norm"], batch["v_norm"],
                )
                losses["total"].backward()
                optimizer.step()

                total_train_loss += losses["total"].item() * len(batch["image"])
                n_train += len(batch["image"])

            avg_train_loss = total_train_loss / max(n_train, 1)

            # Validate
            val_metrics = self._evaluate(val_loader, criterion)

            epoch_metrics = TrainMetrics(
                epoch=epoch + 1,
                train_loss=avg_train_loss,
                val_loss=val_metrics["loss"],
                val_accuracy=val_metrics["accuracy"],
                val_centroid_error_px=val_metrics["centroid_error_px"],
            )
            metrics.append(epoch_metrics)

            # Save best
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                self._best_state = {
                    k: v.clone() for k, v in self._model.state_dict().items()
                }

        # Load best weights
        if self._best_state is not None:
            self._model.load_state_dict(self._best_state)

        return metrics

    def _evaluate(
        self,
        loader: DataLoader,
        criterion: BeaconNetLoss,
    ) -> dict[str, float]:
        """Evaluate model on a data loader."""
        self._model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        centroid_errors = []

        with torch.no_grad():
            for batch in loader:
                pred = self._model(batch["image"])
                losses = criterion(
                    pred, batch["present"], batch["u_norm"], batch["v_norm"],
                )
                total_loss += losses["total"].item() * len(batch["image"])

                # Classification accuracy
                pred_present = (torch.sigmoid(pred["presence_logit"]) > 0.5).float()
                correct += (pred_present == batch["present"]).sum().item()
                total += len(batch["image"])

                # Centroid error for positive samples
                mask = batch["present"] > 0.5
                if mask.any():
                    du = (pred["u_norm"][mask] - batch["u_norm"][mask]) * self._intrinsics.width
                    dv = (pred["v_norm"][mask] - batch["v_norm"][mask]) * self._intrinsics.height
                    err = torch.sqrt(du ** 2 + dv ** 2)
                    centroid_errors.extend(err.cpu().numpy().tolist())

        return {
            "loss": total_loss / max(total, 1),
            "accuracy": correct / max(total, 1),
            "centroid_error_px": float(np.mean(centroid_errors)) if centroid_errors else 0.0,
        }

    def save_model(self, path: str | Path) -> None:
        """Save the best model weights."""
        if self._model is None:
            raise RuntimeError("No trained model to save. Call train() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), str(path))

    @property
    def model(self) -> BeaconNet | None:
        """Trained model (or None if not yet trained)."""
        return self._model
