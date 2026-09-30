"""Atmospheric turbulence phase screen generator based on Kolmogorov theory.

Implements Noll's theoretical Zernike covariance spectrum and temporal
autoregressive colored noise driven by Taylor's frozen flow hypothesis:
  - Calculates Fried atmospheric coherence parameter (r_0)
  - Calculates Greenwood frequency (f_G)
  - Synthesizes dynamic 2D turbulent wavefronts with exact (D / r_0)^(5/3) energy scaling
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from fsoc_pat.ao.ao_types import WavefrontMap
from fsoc_pat.ao.zernike import ZernikeBasis, noll_to_nm


@dataclass
class KolmogorovTurbulenceConfig:
    """Atmospheric turbulence optical parameters.

    Attributes:
        cn2: Refractive index structure parameter (m^(-2/3)). E.g., 1e-15 (weak) to 5e-14 (moderate/strong).
        link_distance_m: Propagation distance through turbulent path (m).
        pupil_diameter_m: Receiver telescope aperture diameter D (m).
        wavelength_m: Laser carrier wavelength lambda (m). Default 1550 nm.
        wind_speed_m_s: Transverse wind speed across optical path (m/s).
        seed: Random seed for reproducible turbulence realizations.
    """

    cn2: float = 2.5e-14
    link_distance_m: float = 1000.0
    pupil_diameter_m: float = 0.15
    wavelength_m: float = 1550e-9
    wind_speed_m_s: float = 5.0
    seed: int = 42


def compute_fried_parameter(cn2: float, link_distance_m: float, wavelength_m: float) -> float:
    """Calculate the Fried atmospheric coherence length r_0 (m).

    Formula (plane wave, horizontal path):
      r_0 = (0.423 * k^2 * C_n^2 * L)^(-3/5)
      where k = 2*pi / lambda

    Args:
        cn2: Refractive index structure constant (m^(-2/3)).
        link_distance_m: Path length (m).
        wavelength_m: Carrier wavelength (m).

    Returns:
        Fried parameter r_0 in meters. Returns 100.0 m for vacuum (cn2 <= 0).
    """
    if cn2 <= 0.0 or link_distance_m <= 0.0:
        return 100.0  # Vacuum equivalent

    k = 2.0 * math.pi / wavelength_m
    val = 0.423 * (k ** 2) * cn2 * link_distance_m
    return float(val ** (-0.6))


def compute_greenwood_frequency(r_0_m: float, wind_speed_m_s: float) -> float:
    """Calculate the Greenwood frequency f_G (Hz).

    Formula:
      f_G = 0.43 * (v_wind / r_0)

    Args:
        r_0_m: Fried coherence parameter (m).
        wind_speed_m_s: Transverse atmospheric wind speed (m/s).

    Returns:
        Greenwood frequency in Hertz.
    """
    if r_0_m <= 0.0:
        return 1.0
    return float(0.43 * (wind_speed_m_s / r_0_m))


def noll_theoretical_variance(j: int, d_over_r0: float) -> float:
    """Calculate theoretical Noll modal variance <a_j^2> for Kolmogorov turbulence.

    Formula:
      <a_j^2> = c_j * (D / r_0)^(5/3)

    Theoretical coefficients c_j from Noll (1976):
      Piston (j=1): Arbitrary phase reference (0 in AO)
      Tip/Tilt (j=2, 3): ~0.448
      Defocus (j=4): ~0.0232
      Astigmatism (j=5, 6): ~0.0232
      Coma (j=7, 8): ~0.00619
      Trefoil (j=9, 10): ~0.00619
      Spherical (j=11): ~0.00245
      Higher orders (n >= 4): c_j ≈ 0.046 * (n + 1)^(-11/3)

    Args:
        j: 1-based Noll index.
        d_over_r0: Ratio of pupil diameter to Fried parameter (D / r_0).

    Returns:
        Theoretical variance in radians squared.
    """
    if j == 1:
        return 0.0  # Piston has no optical effect in imaging/coupling

    n, _ = noll_to_nm(j)

    if n == 1:
        c_j = 0.448
    elif n == 2:
        c_j = 0.0232
    elif n == 3:
        c_j = 0.00619
    elif n == 4:
        c_j = 0.00245
    elif n == 5:
        c_j = 0.00123
    else:
        c_j = 0.046 * ((n + 1) ** (-11.0 / 3.0))

    scale = max(0.0, d_over_r0) ** (5.0 / 3.0)
    return float(c_j * scale)


class KolmogorovPhaseScreenGenerator:
    """Synthesizes dynamic, temporally correlated atmospheric phase screens.

    Follows Taylor's frozen flow hypothesis using an autoregressive Markov
    process on the Zernike modal expansion vector.
    """

    def __init__(
        self,
        config: KolmogorovTurbulenceConfig | None = None,
        basis: ZernikeBasis | None = None,
        max_modes: int = 22,
    ) -> None:
        self.config = config or KolmogorovTurbulenceConfig()
        self.basis = basis or ZernikeBasis(resolution=64, max_modes=max_modes)
        self.max_modes = max_modes
        self._rng = np.random.default_rng(self.config.seed)

        # Compute atmospheric turbulence constants
        self.r_0 = compute_fried_parameter(
            self.config.cn2, self.config.link_distance_m, self.config.wavelength_m
        )
        self.d_over_r0 = self.config.pupil_diameter_m / max(1e-4, self.r_0)
        self.f_greenwood = compute_greenwood_frequency(self.r_0, self.config.wind_speed_m_s)

        # Compute standard deviations for all modes
        self.modal_stds = np.zeros(self.max_modes, dtype=np.float64)
        for i in range(self.max_modes):
            j = i + 1
            var_j = noll_theoretical_variance(j, self.d_over_r0)
            self.modal_stds[i] = math.sqrt(var_j)

        # Current modal coefficients state [a_1, ..., a_J]
        self._current_coeffs = np.zeros(self.max_modes, dtype=np.float64)
        self.reset()

    def reset(self) -> None:
        """Reset the phase screen generator with a fresh random realization."""
        # Draw initial Gaussian modal coefficients matching Kolmogorov spectrum
        self._current_coeffs = self._rng.normal(0.0, self.modal_stds, size=self.max_modes)
        self._current_coeffs[0] = 0.0  # Piston set to zero

    def step(self, dt: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Advance atmospheric turbulence by timestep dt and return current phase.

        Uses autoregressive Gauss-Markov temporal correlation:
          rho_t = exp(-dt / tau_atm)
          a_{t+1} = rho_t * a_t + sqrt(1 - rho_t^2) * w_t

        Args:
            dt: Timestep in seconds.

        Returns:
            Tuple of:
              - 2D phase map array of shape (resolution, resolution) in radians.
              - 1D array of current Zernike modal amplitudes in radians.
        """
        if dt > 0.0 and self.f_greenwood > 0.0:
            # Atmospheric correlation time scale tau ≈ 1 / (2 * pi * f_G)
            tau_atm = 1.0 / (2.0 * math.pi * self.f_greenwood)
            rho_t = math.exp(-dt / tau_atm)
            innov_scale = math.sqrt(max(0.0, 1.0 - rho_t * rho_t))

            # Innovations drawn from stationary distribution
            innovations = self._rng.normal(0.0, self.modal_stds * innov_scale, size=self.max_modes)
            self._current_coeffs = rho_t * self._current_coeffs + innovations
            self._current_coeffs[0] = 0.0  # Zero piston
        else:
            self._current_coeffs = self._rng.normal(0.0, self.modal_stds, size=self.max_modes)
            self._current_coeffs[0] = 0.0

        # Synthesize 2D phase map
        phase_map = self.basis.synthesize(self._current_coeffs)
        return phase_map, self._current_coeffs.copy()

    def generate_static_screen(self) -> WavefrontMap:
        """Generate a single static turbulent wavefront snapshot."""
        phase_map, _ = self.step(0.0)
        return WavefrontMap(
            phase=phase_map,
            pupil_mask=self.basis.pupil_mask,
            diameter_m=self.config.pupil_diameter_m,
            wavelength_m=self.config.wavelength_m,
        )
