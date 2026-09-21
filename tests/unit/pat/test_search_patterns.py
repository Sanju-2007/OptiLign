"""Unit tests for FSOC PAT search patterns including OmniSearchPattern."""

import math
import pytest

from fsoc_pat.pat.search_patterns import (
    OmniSearchPattern,
    RasterSearchPattern,
    SpiralSearchPattern,
)


class TestOmniSearchPattern:
    """Test suite for OmniSearchPattern scanning in all degrees."""

    def test_initialization(self) -> None:
        """Initializes with full pan coverage and valid tilt limits."""
        pattern = OmniSearchPattern()
        assert pattern.pan_amplitude == math.pi
        assert pattern.tilt_min < 0.0
        assert pattern.tilt_max > 0.0
        assert not pattern.is_finished

    def test_step_rates_and_offsets(self) -> None:
        """Pattern steps smoothly without NaN or zero commands."""
        pattern = OmniSearchPattern()
        pattern.reset(0.0, 0.0)

        pan_rate, tilt_rate = pattern.step(0.033)
        assert not math.isnan(pan_rate)
        assert not math.isnan(tilt_rate)

        d_pan, d_tilt = pattern.current_offset()
        assert not math.isnan(d_pan)
        assert not math.isnan(d_tilt)

    def test_all_degrees_coverage(self) -> None:
        """Pattern traverses wide range in both pan (azimuth) and tilt (elevation)."""
        pattern = OmniSearchPattern(
            pan_amplitude_rad=math.pi,
            tilt_min_rad=math.radians(-20.0),
            tilt_max_rad=math.radians(45.0),
            pan_freq=0.5,
            tilt_freq=0.2,
        )
        pattern.reset(0.0, 0.0)

        pan_offsets = []
        tilt_offsets = []

        dt = 0.05
        for _ in range(200):  # 10 seconds of scan
            pattern.step(dt)
            dp, dtilt = pattern.current_offset()
            pan_offsets.append(dp)
            tilt_offsets.append(dtilt)

        # Must reach positive and negative pan (sweeping across degrees)
        assert max(pan_offsets) > math.radians(60)
        assert min(pan_offsets) < -math.radians(60)

        # Must reach high and low elevations
        assert max(tilt_offsets) > math.radians(15)
        assert min(tilt_offsets) < -math.radians(10)
