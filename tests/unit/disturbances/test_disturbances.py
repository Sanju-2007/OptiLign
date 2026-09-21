"""Unit tests for platform jitter, atmospheric turbulence, and occlusions."""

import math
import pytest

from fsoc_pat.disturbances.atmospheric import AtmosphericConfig, AtmosphericTurbulence
from fsoc_pat.disturbances.occlusion import OcclusionConfig, OcclusionModel
from fsoc_pat.disturbances.platform_jitter import JitterConfig, PlatformJitter


def test_platform_jitter():
    jitter = PlatformJitter(JitterConfig(random_std_rad=0.001, seed=42))

    pan_offsets = []
    tilt_offsets = []
    for step in range(100):
        t = step * 0.01
        dp, dt = jitter.compute(t, 0.01)
        pan_offsets.append(dp)
        tilt_offsets.append(dt)

    assert any(abs(p) > 0.0001 for p in pan_offsets)
    assert any(abs(t) > 0.0001 for t in tilt_offsets)
    # Ensure reasonable bounded amplitude (< 10 mrad)
    assert max(map(abs, pan_offsets)) < 0.01
    assert max(map(abs, tilt_offsets)) < 0.01


def test_atmospheric_turbulence():
    turb = AtmosphericTurbulence(
        AtmosphericConfig(beam_wander_std_rad=0.0005, scintillation_index=0.3, seed=42)
    )

    scales = []
    for _ in range(50):
        wp, wt, scale = turb.compute(0.02)
        scales.append(scale)
        assert abs(wp) < 0.005
        assert abs(wt) < 0.005

    # Scintillation scale should stay positive and vary around 1.0
    assert all(s > 0.0 for s in scales)
    assert min(scales) < 1.0 or max(scales) > 1.0


def test_occlusion_model():
    model = OcclusionModel(OcclusionConfig(intervals=[(1.0, 2.0)], transition_duration_s=0.1))

    # Before occlusion
    assert not model.is_occluded(0.5)
    assert model.transmission(0.5) == 1.0

    # During occlusion
    assert model.is_occluded(1.5)
    assert model.transmission(1.5) == 0.0

    # After occlusion
    assert not model.is_occluded(2.5)
    assert model.transmission(2.5) == 1.0
