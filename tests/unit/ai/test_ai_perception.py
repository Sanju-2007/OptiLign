"""Tests for AI perception modules — model, dataset, detector, hybrid, trainer."""

import math
import os

import cv2
import numpy as np
import pytest
import torch

from fsoc_pat.ai.model import BeaconNet, BeaconNetLoss, ModelConfig
from fsoc_pat.ai.dataset import DatasetConfig, DatasetGenerator, Sample
from fsoc_pat.ai.ai_detector import AIDetector
from fsoc_pat.ai.hybrid_detector import HybridDetector, HybridConfig
from fsoc_pat.ai.trainer import TrainConfig, Trainer
from fsoc_pat.core.state import CameraState
from fsoc_pat.core.types import AngularState, Orientation, Position3D
from fsoc_pat.geometry.camera_model import CameraIntrinsics
from fsoc_pat.perception.detection_types import DetectionStatus


@pytest.fixture
def intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics.from_fov(640, 480, math.radians(60))


# ─── BeaconNet Model Tests ────────────────────────────────

class TestBeaconNet:

    def test_output_shape(self) -> None:
        """Model should produce correct output shapes."""
        model = BeaconNet()
        x = torch.randn(4, 1, 120, 160)
        out = model(x)

        assert out["presence_logit"].shape == (4,)
        assert out["u_norm"].shape == (4,)
        assert out["v_norm"].shape == (4,)

    def test_output_range(self) -> None:
        """u_norm and v_norm should be in [0, 1]."""
        model = BeaconNet()
        x = torch.randn(8, 1, 120, 160)
        out = model(x)

        assert (out["u_norm"] >= 0).all()
        assert (out["u_norm"] <= 1).all()
        assert (out["v_norm"] >= 0).all()
        assert (out["v_norm"] <= 1).all()

    def test_parameter_count(self) -> None:
        """BeaconNet should have < 100K parameters."""
        model = BeaconNet()
        count = model.count_parameters()
        assert count < 100_000, f"Model has {count} params, expected < 100K"

    def test_deterministic(self) -> None:
        """Same input → same output in eval mode."""
        model = BeaconNet()
        model.eval()
        x = torch.randn(1, 1, 120, 160)

        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)

        assert torch.allclose(out1["presence_logit"], out2["presence_logit"])
        assert torch.allclose(out1["u_norm"], out2["u_norm"])

    def test_batch_independence(self) -> None:
        """Each sample in a batch should be independent (in eval mode)."""
        model = BeaconNet()
        model.eval()

        x1 = torch.randn(1, 1, 120, 160)
        x2 = torch.randn(1, 1, 120, 160)
        x_batch = torch.cat([x1, x2], dim=0)

        with torch.no_grad():
            out_single1 = model(x1)
            out_batch = model(x_batch)

        assert torch.allclose(
            out_single1["presence_logit"],
            out_batch["presence_logit"][:1],
            atol=1e-5,
        )


# ─── Loss Function Tests ─────────────────────────────────

class TestBeaconNetLoss:

    def test_loss_computes(self) -> None:
        """Loss should compute without error."""
        model = BeaconNet()
        criterion = BeaconNetLoss()

        x = torch.randn(4, 1, 120, 160)
        pred = model(x)

        target_present = torch.tensor([1.0, 1.0, 0.0, 0.0])
        target_u = torch.tensor([0.5, 0.3, 0.5, 0.5])
        target_v = torch.tensor([0.5, 0.7, 0.5, 0.5])

        losses = criterion(pred, target_present, target_u, target_v)

        assert "total" in losses
        assert "cls" in losses
        assert "reg" in losses
        assert losses["total"].item() > 0

    def test_loss_backprop(self) -> None:
        """Loss should support backpropagation."""
        model = BeaconNet()
        criterion = BeaconNetLoss()

        x = torch.randn(2, 1, 120, 160)
        pred = model(x)
        losses = criterion(
            pred,
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.5, 0.5]),
            torch.tensor([0.5, 0.5]),
        )
        losses["total"].backward()

        # Gradients should exist
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None
                break


# ─── Dataset Generator Tests ─────────────────────────────

class TestDatasetGenerator:

    def test_generates_correct_count(self, intrinsics: CameraIntrinsics) -> None:
        """Should generate the requested number of samples."""
        gen = DatasetGenerator(intrinsics, DatasetConfig(num_samples=50, seed=1))
        samples = gen.generate()
        assert len(samples) == 50

    def test_positive_negative_ratio(self, intrinsics: CameraIntrinsics) -> None:
        """Negative fraction should be approximately correct."""
        gen = DatasetGenerator(
            intrinsics, DatasetConfig(num_samples=100, fraction_negative=0.2, seed=1),
        )
        samples = gen.generate()
        n_neg = sum(1 for s in samples if not s.present)
        assert n_neg == 20  # Exact, since we generate deterministically

    def test_split_no_leakage(self, intrinsics: CameraIntrinsics) -> None:
        """Train/val/test splits should have no overlap."""
        gen = DatasetGenerator(intrinsics, DatasetConfig(num_samples=100, seed=1))
        samples = gen.generate()
        train, val, test = gen.split(samples)

        assert len(train) + len(val) + len(test) == len(samples)
        # No overlap (check by index)
        train_ids = {id(s) for s in train}
        val_ids = {id(s) for s in val}
        test_ids = {id(s) for s in test}
        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0

    def test_image_shape(self, intrinsics: CameraIntrinsics) -> None:
        """Generated images should have correct shape."""
        gen = DatasetGenerator(intrinsics, DatasetConfig(num_samples=5, seed=1))
        samples = gen.generate()
        for s in samples:
            assert s.image.shape == (480, 640)
            assert s.image.dtype == np.uint8

    def test_deterministic(self, intrinsics: CameraIntrinsics) -> None:
        """Same seed → same dataset."""
        gen1 = DatasetGenerator(intrinsics, DatasetConfig(num_samples=10, seed=42))
        gen2 = DatasetGenerator(intrinsics, DatasetConfig(num_samples=10, seed=42))
        s1 = gen1.generate()
        s2 = gen2.generate()

        for a, b in zip(s1, s2):
            assert a.present == b.present
            np.testing.assert_array_equal(a.image, b.image)


# ─── AI Detector Tests ───────────────────────────────────

class TestAIDetector:

    def test_untrained_runs(self) -> None:
        """Untrained detector should still produce a result."""
        detector = AIDetector(confidence_threshold=0.5)
        image = np.zeros((480, 640), dtype=np.uint8)
        result = detector.detect(image)

        assert result.frame_index == 1
        assert result.processing_time_ms > 0

    def test_frame_index_increments(self) -> None:
        """Frame index should increment."""
        detector = AIDetector()
        image = np.zeros((480, 640), dtype=np.uint8)

        r1 = detector.detect(image)
        r2 = detector.detect(image)

        assert r1.frame_index == 1
        assert r2.frame_index == 2

    def test_reset(self) -> None:
        detector = AIDetector()
        image = np.zeros((480, 640), dtype=np.uint8)
        detector.detect(image)
        detector.reset()

        r = detector.detect(image)
        assert r.frame_index == 1


# ─── Hybrid Detector Tests ───────────────────────────────

class TestHybridDetector:

    def test_synthetic_blob(self) -> None:
        """Hybrid should detect a clear synthetic blob."""
        image = np.zeros((480, 640), dtype=np.uint8)
        cv2.circle(image, (300, 200), 8, 200, -1)

        hybrid = HybridDetector()
        result = hybrid.detect(image)

        # At minimum, classical should detect it
        assert result.best is not None
        assert abs(result.best.u - 300) < 15
        assert abs(result.best.v - 200) < 15

    def test_empty_image(self) -> None:
        """Empty image should produce a result (untrained model may detect false positives)."""
        image = np.zeros((480, 640), dtype=np.uint8)
        hybrid = HybridDetector()
        result = hybrid.detect(image)

        # With an untrained AI model, we only verify it runs without error
        # and produces a valid Detection object
        assert result.frame_index == 1
        assert result.processing_time_ms >= 0

    def test_frame_index(self) -> None:
        """Frame index should increment."""
        image = np.zeros((480, 640), dtype=np.uint8)
        hybrid = HybridDetector()

        r1 = hybrid.detect(image)
        r2 = hybrid.detect(image)

        assert r1.frame_index == 1
        assert r2.frame_index == 2


# ─── Trainer Integration Tests ───────────────────────────

class TestTrainer:

    def test_quick_train(self, intrinsics: CameraIntrinsics) -> None:
        """A quick training run should complete without error."""
        config = TrainConfig(
            num_samples=30,
            epochs=2,
            batch_size=8,
            seed=42,
        )
        trainer = Trainer(intrinsics, config)
        metrics = trainer.train()

        assert len(metrics) == 2
        assert metrics[0].train_loss > 0
        assert 0 <= metrics[0].val_accuracy <= 1

    def test_model_accessible_after_train(self, intrinsics: CameraIntrinsics) -> None:
        """Model should be accessible after training."""
        config = TrainConfig(num_samples=20, epochs=1, batch_size=10)
        trainer = Trainer(intrinsics, config)
        trainer.train()

        assert trainer.model is not None
        assert trainer.model.count_parameters() > 0

    def test_save_load(self, intrinsics: CameraIntrinsics, tmp_path) -> None:
        """Saved model should be loadable."""
        config = TrainConfig(num_samples=20, epochs=1, batch_size=10)
        trainer = Trainer(intrinsics, config)
        trainer.train()

        model_path = tmp_path / "test_model.pt"
        trainer.save_model(model_path)

        assert model_path.exists()

        # Load into a detector
        detector = AIDetector(model_path=str(model_path))
        image = np.zeros((480, 640), dtype=np.uint8)
        result = detector.detect(image)
        assert result.frame_index == 1
