"""Platform angular vibration and jitter model.

Simulates structural vibrations, engine harmonics, and wind buffeting
affecting the base mount of the optical tracking system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class HarmonicTone:
    """A discrete sinusoidal vibration tone (e.g. rotor or motor harmonic)."""
    frequency_hz: float
    amplitude_rad: float
    phase_rad: float = 0.0


@dataclass
class JitterConfig:
    """Configuration for base platform jitter.

    Attributes:
        tones_pan: Harmonic tones in pan axis.
        tones_tilt: Harmonic tones in tilt axis.
        random_std_rad: RMS random broadband noise standard deviation (rad).
        correlation_time_s: Correlation time constant for colored noise (s).
        seed: Random seed for reproducibility.
    """

    tones_pan: list[HarmonicTone] = field(
        default_factory=lambda: [
            HarmonicTone(frequency_hz=12.0, amplitude_rad=0.0015),
            HarmonicTone(frequency_hz=28.0, amplitude_rad=0.0008),
        ]
    )
    tones_tilt: list[HarmonicTone] = field(
        default_factory=lambda: [
            HarmonicTone(frequency_hz=15.0, amplitude_rad=0.0012),
            HarmonicTone(frequency_hz=35.0, amplitude_rad=0.0006),
        ]
    )
    random_std_rad: float = 0.0008  # 0.8 mrad RMS random jitter
    correlation_time_s: float = 0.05
    seed: int = 42


class PlatformJitter:
    """Platform vibration generator combining deterministic tones and colored noise."""

    def __init__(self, config: JitterConfig | None = None) -> None:
        self.config = config or JitterConfig()
        self._rng = np.random.default_rng(self.config.seed)

        # Markov colored noise states for pan and tilt
        self._noise_pan = 0.0
        self._noise_tilt = 0.0

    def compute(self, t: float, dt: float) -> tuple[float, float]:
        """Compute instantaneous (delta_pan, delta_tilt) jitter angles in radians.

        Args:
            t: Current simulation time (s).
            dt: Timestep (s).

        Returns:
            Tuple of (delta_pan, delta_tilt) angular offsets in radians.
        """
        cfg = self.config

        # 1. Deterministic harmonic tones
        harm_pan = sum(
            tone.amplitude_rad * math.sin(2.0 * math.pi * tone.frequency_hz * t + tone.phase_rad)
            for tone in cfg.tones_pan
        )
        harm_tilt = sum(
            tone.amplitude_rad * math.sin(2.0 * math.pi * tone.frequency_hz * t + tone.phase_rad)
            for tone in cfg.tones_tilt
        )

        # 2. First-order Markov colored random noise
        if dt > 0.0 and cfg.correlation_time_s > 0.0:
            decay = math.exp(-dt / cfg.correlation_time_s)
            innovation_std = cfg.random_std_rad * math.sqrt(max(0.0, 1.0 - decay * decay))
            self._noise_pan = self._noise_pan * decay + float(self._rng.normal(0.0, innovation_std))
            self._noise_tilt = self._noise_tilt * decay + float(self._rng.normal(0.0, innovation_std))
        else:
            self._noise_pan = float(self._rng.normal(0.0, cfg.random_std_rad))
            self._noise_tilt = float(self._rng.normal(0.0, cfg.random_std_rad))

        total_pan_jitter = harm_pan + self._noise_pan
        total_tilt_jitter = harm_tilt + self._noise_tilt
        return total_pan_jitter, total_tilt_jitter
