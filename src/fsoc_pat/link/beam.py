"""Gaussian beam diffraction and propagation optics.

Computes exact TEM00 / M^2 beam waist expansion, Rayleigh range,
divergence angles, and spatial irradiance distributions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from fsoc_pat.link.link_types import LaserConfig


class GaussianBeam:
    """Gaussian beam propagation model."""

    def __init__(self, config: LaserConfig | None = None) -> None:
        self.config = config or LaserConfig()

    @property
    def rayleigh_range_m(self) -> float:
        """Rayleigh range z_R = pi * w0^2 / (M^2 * lambda)."""
        cfg = self.config
        w0 = cfg.beam_waist_w0_m
        return (math.pi * (w0 ** 2)) / (cfg.m2_factor * cfg.wavelength_m)

    @property
    def divergence_half_angle_rad(self) -> float:
        """Far-field divergence half-angle theta_div = (M^2 * lambda) / (pi * w0)."""
        cfg = self.config
        return (cfg.m2_factor * cfg.wavelength_m) / (math.pi * cfg.beam_waist_w0_m)

    @property
    def divergence_full_angle_mrad(self) -> float:
        """Full divergence angle in milliradians."""
        return 2.0 * self.divergence_half_angle_rad * 1000.0

    def beam_radius_at(self, z_m: float) -> float:
        """Compute 1/e^2 beam spot radius w(z) at propagation distance z.

        w(z) = w0 * sqrt(1 + (z / z_R)^2)

        Args:
            z_m: Propagation distance along optical axis (m).

        Returns:
            Beam radius in meters.
        """
        z_r = self.rayleigh_range_m
        return self.config.beam_waist_w0_m * math.sqrt(1.0 + (z_m / z_r) ** 2)

    def peak_irradiance(self, z_m: float, power_w: float | None = None) -> float:
        """Compute peak on-axis irradiance I_0(z) in W/m^2.

        I_0(z) = 2 * P / (pi * w(z)^2)

        Args:
            z_m: Propagation distance (m).
            power_w: Optical power arriving at distance z (W). Defaults to launched power.
        """
        if power_w is None:
            power_w = self.config.power_w * self.config.tx_efficiency

        w_z = self.beam_radius_at(z_m)
        return (2.0 * power_w) / (math.pi * (w_z ** 2))

    def irradiance_at(self, r_m: float, z_m: float, power_w: float | None = None) -> float:
        """Compute irradiance I(r, z) at radial offset r from beam axis.

        I(r, z) = I_0(z) * exp(-2 * r^2 / w(z)^2)

        Args:
            r_m: Radial distance from optical beam center (m).
            z_m: Propagation distance (m).
            power_w: Transmitted power (W).

        Returns:
            Irradiance in W/m^2.
        """
        w_z = self.beam_radius_at(z_m)
        i_0 = self.peak_irradiance(z_m, power_w)
        return i_0 * math.exp(-2.0 * (r_m ** 2) / (w_z ** 2))

    def enclosed_power(self, r_aperture_m: float, z_m: float, power_w: float | None = None) -> float:
        """Compute total optical power collected within a centered aperture of radius r.

        P_enc = P * (1 - exp(-2 * r^2 / w(z)^2))

        Args:
            r_aperture_m: Aperture radius (m).
            z_m: Distance (m).
            power_w: Incident beam power (W).

        Returns:
            Enclosed power in Watts.
        """
        if power_w is None:
            power_w = self.config.power_w * self.config.tx_efficiency

        w_z = self.beam_radius_at(z_m)
        fraction = 1.0 - math.exp(-2.0 * (r_aperture_m ** 2) / (w_z ** 2))
        return power_w * fraction
