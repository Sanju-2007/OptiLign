"""Data structures, configuration, and telemetry for predictive anti-jitter feedforward.

Defines the mathematical and physical parameters for:
  - Harmonic vibration modes (frequency, amplitude, phase)
  - Latency horizon & dead-time compensation
  - Predictive feedforward injection into Fast Steering Mirrors (FSM)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass
class HarmonicMode:
    """A detected mechanical vibration harmonic mode.

    Attributes:
        frequency_hz: Vibration resonance frequency in Hertz.
        amplitude_urad: Peak angular deflection amplitude in micro-radians.
        phase_rad: Estimated instantaneous phase angle in radians.
        snr_db: Power spectral density prominence over noise floor in dB.
    """

    frequency_hz: float
    amplitude_urad: float
    phase_rad: float
    snr_db: float


@dataclass
class PredictiveJitterConfig:
    """Predictive feedforward compensator configuration.

    Attributes:
        enabled: Whether predictive feedforward is actively injected.
        latency_horizon_s: Total loop dead-time to anticipate (s).
            Default 0.015s (15 ms: exposure + readout + compute + lag).
        sampling_rate_hz: Control loop execution frequency in Hz (e.g. 500 Hz).
        num_harmonics: Maximum number of dominant harmonic peaks to track.
        min_freq_hz: Lower frequency cutoff for vibration tracking (Hz).
        max_freq_hz: Upper frequency cutoff for vibration tracking (Hz).
        adaptation_rate: Learning rate for Recursive Least Squares (RLS) / LMS tracker.
        feedforward_gain: Scaling factor for injected feedforward command (0.0 to 1.0).
        history_buffer_len: Number of past angular jitter samples stored for spectral analysis.
    """

    enabled: bool = True
    latency_horizon_s: float = 0.015  # 15 ms system dead-time
    sampling_rate_hz: float = 500.0   # 500 Hz
    num_harmonics: int = 4
    min_freq_hz: float = 5.0
    max_freq_hz: float = 200.0
    adaptation_rate: float = 0.15
    feedforward_gain: float = 0.95
    history_buffer_len: int = 128



@dataclass
class JitterTelemetry:
    """Instantaneous telemetry for predictive jitter compensation.

    Attributes:
        timestamp_s: Simulation timestamp in seconds.
        raw_jitter_pan_urad: Platform disturbance jitter in pan before compensation (urad).
        raw_jitter_tilt_urad: Platform disturbance jitter in tilt before compensation (urad).
        raw_jitter_rms_urad: Total radial RMS disturbance jitter (urad).
        residual_jitter_rms_urad: Net line-of-sight jitter after feedforward cancellation (urad).
        jitter_rejection_db: Suppression ratio in decibels (20*log10(raw / residual)).
        feedforward_pan_urad: Injected feedforward command in pan (urad).
        feedforward_tilt_urad: Injected feedforward command in tilt (urad).
        phase_lead_deg: Phase advance angle applied over dead-time horizon (deg).
        tracked_frequencies_hz: Dominant vibration resonance frequencies identified online (Hz).
        feedforward_active: Whether predictive feedforward is actively engaged.
    """

    timestamp_s: float
    raw_jitter_pan_urad: float
    raw_jitter_tilt_urad: float
    raw_jitter_rms_urad: float
    residual_jitter_rms_urad: float
    jitter_rejection_db: float
    feedforward_pan_urad: float
    feedforward_tilt_urad: float
    phase_lead_deg: float
    tracked_frequencies_hz: list[float]
    feedforward_active: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert telemetry to serializable dictionary."""
        return {
            "timestamp_s": round(self.timestamp_s, 4),
            "raw_jitter_pan_urad": round(self.raw_jitter_pan_urad, 2),
            "raw_jitter_tilt_urad": round(self.raw_jitter_tilt_urad, 2),
            "raw_jitter_rms_urad": round(self.raw_jitter_rms_urad, 2),
            "residual_jitter_rms_urad": round(self.residual_jitter_rms_urad, 2),
            "jitter_rejection_db": round(self.jitter_rejection_db, 2),
            "feedforward_pan_urad": round(self.feedforward_pan_urad, 2),
            "feedforward_tilt_urad": round(self.feedforward_tilt_urad, 2),
            "phase_lead_deg": round(self.phase_lead_deg, 1),
            "tracked_frequencies_hz": [round(f, 1) for f in self.tracked_frequencies_hz],
            "feedforward_active": self.feedforward_active,
        }
