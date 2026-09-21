"""Unit tests for Gaussian beam propagation."""

import math
import pytest

from fsoc_pat.link.beam import GaussianBeam
from fsoc_pat.link.link_types import LaserConfig


def test_gaussian_beam_rayleigh_range():
    cfg = LaserConfig(wavelength_m=1550e-9, beam_waist_w0_m=0.01, m2_factor=1.0)
    beam = GaussianBeam(cfg)

    # z_R = pi * w0^2 / lambda = pi * 0.0001 / 1.55e-6 ~ 202.68 m
    expected_zr = (math.pi * (0.01 ** 2)) / 1550e-9
    assert math.isclose(beam.rayleigh_range_m, expected_zr, rel_tol=1e-5)


def test_gaussian_beam_expansion():
    cfg = LaserConfig(wavelength_m=1550e-9, beam_waist_w0_m=0.01, m2_factor=1.0)
    beam = GaussianBeam(cfg)

    # At waist z=0: w(0) == w0
    assert math.isclose(beam.beam_radius_at(0.0), 0.01, rel_tol=1e-6)

    # At z = z_R: w(z_R) == w0 * sqrt(2)
    zr = beam.rayleigh_range_m
    assert math.isclose(beam.beam_radius_at(zr), 0.01 * math.sqrt(2.0), rel_tol=1e-5)

    # Far field divergence: for large z, w(z) ~ theta_div * z
    z_far = 10000.0  # 10 km
    theta_div = beam.divergence_half_angle_rad
    assert math.isclose(beam.beam_radius_at(z_far), theta_div * z_far, rel_tol=1e-3)


def test_gaussian_beam_irradiance():
    cfg = LaserConfig(power_w=1.0, tx_efficiency=1.0, beam_waist_w0_m=0.01)
    beam = GaussianBeam(cfg)

    # Peak on-axis irradiance: 2 * P / (pi * w0^2) at z=0
    i0 = beam.peak_irradiance(0.0)
    expected_i0 = 2.0 / (math.pi * (0.01 ** 2))
    assert math.isclose(i0, expected_i0, rel_tol=1e-5)

    # Off-axis irradiance at r = w(z): I(w) = I_0 * exp(-2) ~ I_0 * 0.1353
    i_waist = beam.irradiance_at(r_m=0.01, z_m=0.0)
    assert math.isclose(i_waist, i0 * math.exp(-2.0), rel_tol=1e-5)

    # Enclosed power within r = w(0): 1 - exp(-2) ~ 86.5% of total power
    p_enc = beam.enclosed_power(0.01, 0.0)
    assert math.isclose(p_enc, 1.0 * (1.0 - math.exp(-2.0)), rel_tol=1e-4)
