"""Atmospheric turbulence models for free-space optical communication (FSOC).

Simulates:
  - Beam Wander: Random angle-of-arrival fluctuations (rad)
  - Scintillation: Log-normal intensity fading (normalized scale)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class AtmosphericConfig:
    """Atmospheric turbulence parameters.

    Attributes:
        cn2: Refractive index structure parameter (m^(-2/3)), e.g. 1e-14 (moderate).
        link_distance_m: Propagation distance (m).
        aperture_diameter_m: Receiver aperture diameter (m).
        beam_wander_std_rad: RMS beam wander angle (rad). Default ~0.5 mrad.
        scintillation_index: Intensity variance sigma_I^2 [0, 1.2]. 0 = no fading.
        temporal_correlation_s: Turbulence temporal correlation time (s).
        seed: Random seed.
    """

    cn2: float = 1e-14
    link_distance_m: float = 100.0
    aperture_diameter_m: float = 0.05
    beam_wander_std_rad: float = 0.0005  # 0.5 mrad RMS wander
    scintillation_index: float = 0.25    # Moderate fading
    temporal_correlation_s: float = 0.02 # 20 ms correlation time
    seed: int = 42


class AtmosphericTurbulence:
    """Simulates beam wander and scintillation intensity fluctuations."""

    def __init__(self, config: AtmosphericConfig | None = None) -> None:
        self.config = config or AtmosphericConfig()
        self._rng = np.random.default_rng(self.config.seed)

        # Log-amplitude variance for log-normal scintillation
        # sigma_I^2 = exp(4 * sigma_chi^2) - 1 => sigma_chi^2 = 0.25 * ln(1 + sigma_I^2)
        s_i = max(0.0, min(1.5, self.config.scintillation_index))
        self._sigma_chi = math.sqrt(0.25 * math.log(1.0 + s_i))

        self._wander_pan = 0.0
        self._wander_tilt = 0.0
        self._chi = 0.0

    def compute(self, dt: float) -> tuple[float, float, float]:
        """Compute atmospheric perturbation for the current timestep.

        Args:
            dt: Timestep (s).

        Returns:
            Tuple of:
              - wander_pan: Beam wander angular deflection in pan (rad).
              - wander_tilt: Beam wander angular deflection in tilt (rad).
              - intensity_scale: Multiplicative factor for beacon intensity (>0).
        """
        cfg = self.config

        # 1. Beam Wander (colored Gaussian process)
        if dt > 0.0 and cfg.temporal_correlation_s > 0.0:
            decay = math.exp(-dt / cfg.temporal_correlation_s)
            innov_std = cfg.beam_wander_std_rad * math.sqrt(max(0.0, 1.0 - decay * decay))
            self._wander_pan = self._wander_pan * decay + float(self._rng.normal(0.0, innov_std))
            self._wander_tilt = self._wander_tilt * decay + float(self._rng.normal(0.0, innov_std))

            # 2. Scintillation log-amplitude chi (colored Gaussian process)
            innov_chi = self._sigma_chi * math.sqrt(max(0.0, 1.0 - decay * decay))
            self._chi = self._chi * decay + float(self._rng.normal(0.0, innov_chi))
        else:
            self._wander_pan = float(self._rng.normal(0.0, cfg.beam_wander_std_rad))
            self._wander_tilt = float(self._rng.normal(0.0, cfg.beam_wander_std_rad))
            self._chi = float(self._rng.normal(0.0, self._sigma_chi))

        # Log-normal intensity I = I_0 * exp(2*chi - 2*sigma_chi^2) to preserve mean = 1
        intensity_scale = math.exp(2.0 * self._chi - 2.0 * (self._sigma_chi ** 2))
        intensity_scale = max(0.05, min(3.0, intensity_scale))

        return self._wander_pan, self._wander_tilt, intensity_scale
