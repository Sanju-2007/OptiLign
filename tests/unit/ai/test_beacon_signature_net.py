"""Unit tests for BeaconSignatureNet and MLBeaconTracker."""

import math
from pathlib import Path
import numpy as np
import pytest
import torch

from fsoc_pat.ai.beacon_signature_net import (
    BeaconSignatureNet,
    MLBeaconTracker,
    SignatureModelConfig,
    generate_beacon_pulse_pattern,
    generate_synthetic_training_batch,
    train_beacon_signature_model,
)


class TestBeaconSignatureNet:
    """Test suite for neural network architecture and inference."""

    def test_model_forward_shapes(self) -> None:
        """Model produces expected tensor shapes."""
        config = SignatureModelConfig(seq_len=36)
        model = BeaconSignatureNet(config)

        batch_size = 4
        int_seq = torch.randn(batch_size, 2, 36)
        pos_seq = torch.randn(batch_size, 3, 36)

        out = model(int_seq, pos_seq)

        assert out["signature_logit"].shape == (batch_size, 1)
        assert out["signature_prob"].shape == (batch_size, 1)
        assert out["pred_position"].shape == (batch_size, 2)
        assert out["pred_velocity"].shape == (batch_size, 2)

        # Probabilities must be in [0, 1]
        assert (out["signature_prob"] >= 0.0).all()
        assert (out["signature_prob"] <= 1.0).all()

        # Positions must be in [0, 1]
        assert (out["pred_position"] >= 0.0).all()
        assert (out["pred_position"] <= 1.0).all()

    def test_parameter_count(self) -> None:
        """Model must be lightweight (< 100K parameters) for real-time operation."""
        model = BeaconSignatureNet()
        count = model.count_parameters()
        assert count < 100_000, f"Model has {count} parameters, expected < 100K"

    def test_pulse_pattern_generator(self) -> None:
        """Ground truth generator outputs the 3 distinct pulses in 1.2s cycle."""
        t_samples = np.array([0.05, 0.20, 0.35, 0.90])
        pat = generate_beacon_pulse_pattern(t_samples, phase_offset=0.0)

        assert pat[0] == 1.0  # Strobe 1
        assert pat[1] == 0.0  # Gap 1
        assert pat[2] == 1.0  # Strobe 2
        assert pat[3] == 0.0  # Sync Interval

    def test_synthetic_batch_generator(self) -> None:
        """Batch generator yields valid batches."""
        ints, poses, sig_labels, target_pos = generate_synthetic_training_batch(
            batch_size=16, seq_len=36
        )
        assert ints.shape == (16, 2, 36)
        assert poses.shape == (16, 3, 36)
        assert sig_labels.shape == (16, 1)
        assert target_pos.shape == (16, 2)

    def test_training_convergence(self, tmp_path: Path) -> None:
        """Model trains quickly and saves weights."""
        model_path = tmp_path / "test_sig_model.pt"
        model, metrics = train_beacon_signature_model(
            model_path=model_path,
            epochs=5,
            batches_per_epoch=8,
            batch_size=32,
        )
        assert model_path.is_file()
        assert metrics["signature_accuracy"] > 0.85
        assert metrics["final_loss"] < 0.25

    def test_tracker_inference(self) -> None:
        """MLBeaconTracker evaluates rolling temporal sequences."""
        tracker = MLBeaconTracker()
        # Feed 36 frames of high intensity (strobe pulse)
        res = None
        for _ in range(40):
            res = tracker.update(
                intensity=240.0,
                is_visible=True,
                u_px=320.0,
                v_px=240.0,
            )
        assert res is not None
        assert "signature_prob" in res
        assert "predicted_u" in res
        assert "predicted_v" in res
        assert 0.0 <= res["predicted_u"] <= 640.0
        assert 0.0 <= res["predicted_v"] <= 480.0
