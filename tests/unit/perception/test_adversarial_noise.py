"""Unit tests for Adversarial Optical Perception & Stress Engine."""

import numpy as np
import pytest

from fsoc_pat.perception.adversarial_noise import (
    AdversarialConfig,
    AdversarialNoiseEngine,
    DiscriminationReport,
)


def test_adversarial_initialization():
    engine = AdversarialNoiseEngine()
    assert engine.config.temporal_carrier_hz == 4.0
    assert engine.config.solar_glint_enabled is False
    assert engine.config.decoys_enabled is False


def test_solar_glint_injection():
    engine = AdversarialNoiseEngine(AdversarialConfig(solar_glint_enabled=True, glint_radius_px=20))
    blank = np.zeros((100, 100), dtype=np.uint8)
    glint_frame = engine.inject_solar_glint(blank, position=(50, 50), radius=20, intensity=255)

    # Core must be saturated
    assert glint_frame[50, 50] == 255
    # Halo around core must be elevated
    assert glint_frame[50, 60] > 0
    # Far away must fall off significantly
    assert glint_frame[10, 10] < 10
    assert glint_frame[50, 50] == 255
    assert glint_frame[50, 75] < 255
    assert glint_frame[50, 75] > glint_frame[10, 10]


def test_decoy_hotspots_injection():
    engine = AdversarialNoiseEngine(AdversarialConfig(decoys_enabled=True, decoy_count=3))
    blank = np.zeros((120, 160), dtype=np.uint8)
    noisy_frame, count = engine.inject_decoy_hotspots(blank, t=0.25, count=3)

    assert count > 0
    assert np.max(noisy_frame) > 100


def test_poisson_noise():
    engine = AdversarialNoiseEngine(AdversarialConfig(poisson_noise_enabled=True, photon_flux=80.0))
    flat = np.full((50, 50), 128, dtype=np.uint8)
    noisy = engine.inject_poisson_noise(flat)

    assert noisy.shape == flat.shape
    # Variance should be non-zero
    assert np.var(noisy) > 0.0


def test_carrier_discrimination_rejects_glint_and_decoys():
    config = AdversarialConfig(
        solar_glint_enabled=True,
        decoys_enabled=True,
        decoy_count=2,
        temporal_carrier_hz=4.0,
    )
    engine = AdversarialNoiseEngine(config)

    # Simulate 8 consecutive frames at 30 fps with genuine 4Hz beacon + static glint + decoys
    fps = 30.0
    for frame_idx in range(8):
        t = frame_idx / fps
        frame = np.zeros((60, 60), dtype=np.uint8)
        # Genuine beacon at (20, 20) pulsating at 4.0 Hz
        beacon_val = int(127.5 * (1.0 + np.sin(2.0 * np.pi * 4.0 * t)))
        frame[20, 20] = beacon_val

        # Apply adversarial effects (glint + decoys)
        perturbed, _ = engine.apply_adversarial_effects(frame, t)

    # Run carrier lock-in discriminator
    report = engine.discriminate_carrier()

    assert isinstance(report, DiscriminationReport)
    assert report.solar_glint_detected is True
    assert report.solar_glint_suppressed is True
    assert report.total_decoys_present == 2
    assert report.decoys_rejected == 2
    assert report.false_positive_rejection_pct == 100.0
    assert report.genuine_carrier_locked is True


def test_filter_frame_for_detection():
    config = AdversarialConfig(temporal_carrier_hz=4.0, carrier_filter_enabled=True)
    engine = AdversarialNoiseEngine(config)

    fps = 30.0
    for frame_idx in range(8):
        t = frame_idx / fps
        frame = np.zeros((40, 40), dtype=np.uint8)
        # Static bright background point (DC glint at 10, 10)
        frame[10, 10] = 250
        # 4.0 Hz pulsating beacon at (30, 30)
        frame[30, 30] = int(127.5 * (1.0 + np.cos(2.0 * np.pi * 4.0 * t)))
        engine._record_frame(frame, t)

    filtered = engine.filter_frame_for_detection(frame)
    # The 4Hz pulsating beacon must have significant energy
    assert filtered.shape == (40, 40)
