"""Comprehensive unit test suite for Adaptive Optics (AO) & Wavefront Reconstruction.

Validates:
  - Zernike polynomial evaluation, Noll indices, orthonormality, and modal decomposition
  - Kolmogorov atmospheric turbulence physics (Fried parameter r_0, Noll variances)
  - Shack-Hartmann Wavefront Sensor (SH-WFS) slope measurement
  - 37-actuator Deformable Mirror (DM) geometry, stroke limits, and phase shift
  - SVD wavefront reconstruction (modal and zonal)
  - Single-Mode Fiber (SMF) mode-overlap integral calculation
  - Closed-loop AO convergence (Strehl ratio improvement & WFE reduction)
  - Tri-Stage Pointing & Wavefront Controller (Gimbal + FSM + AO)
  - AO Benchmark Engine
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from fsoc_pat.ao.ao_benchmark import AOBenchmarkConfig, AOBenchmarkEngine
from fsoc_pat.ao.ao_system import AdaptiveOpticsSystem, compute_smf_coupling_efficiency
from fsoc_pat.ao.ao_types import (
    AOConfig,
    DeformableMirrorConfig,
    SHWFSConfig,
    WavefrontMap,
    ZernikeIndex,
)
from fsoc_pat.ao.deformable_mirror import DeformableMirror
from fsoc_pat.ao.kolmogorov import (
    KolmogorovPhaseScreenGenerator,
    KolmogorovTurbulenceConfig,
    compute_fried_parameter,
    compute_greenwood_frequency,
    noll_theoretical_variance,
)
from fsoc_pat.ao.reconstructor import WavefrontReconstructor
from fsoc_pat.ao.shack_hartmann import ShackHartmannSensor
from fsoc_pat.ao.zernike import (
    ZernikeBasis,
    noll_to_nm,
    radial_polynomial,
    zernike_noll,
    zernike_polynomial,
)
from fsoc_pat.control.tri_stage import TriStageController
from fsoc_pat.core.types import AngularError
from fsoc_pat.link.budget import FSOCLinkBudget


class TestZernikePolynomials:
    """Tests for Zernike polynomial evaluation and modal decomposition."""

    def test_noll_indexing_mapping(self) -> None:
        """Verify Noll indexing matches standard optical conventions."""
        assert noll_to_nm(1) == (0, 0)   # Piston
        assert noll_to_nm(2) == (1, 1)   # Tip X
        assert noll_to_nm(3) == (1, -1)  # Tilt Y
        assert noll_to_nm(4) == (2, 0)   # Defocus
        assert noll_to_nm(5) == (2, -2)  # Astigmatism 45 deg
        assert noll_to_nm(6) == (2, 2)   # Astigmatism 0 deg
        assert noll_to_nm(11) == (4, 0)  # Primary Spherical

    def test_radial_polynomial_boundaries(self) -> None:
        """Verify radial polynomials evaluate correctly at center and edge."""
        rho_center = np.array([0.0])
        rho_edge = np.array([1.0])

        # R_0^0(rho) = 1
        assert float(radial_polynomial(0, 0, rho_center)[0]) == pytest.approx(1.0)
        assert float(radial_polynomial(0, 0, rho_edge)[0]) == pytest.approx(1.0)

        # R_2^0(rho) = 2*rho^2 - 1 => -1 at center, +1 at edge
        assert float(radial_polynomial(2, 0, rho_center)[0]) == pytest.approx(-1.0)
        assert float(radial_polynomial(2, 0, rho_edge)[0]) == pytest.approx(1.0)

        # R_1^1(rho) = rho => 0 at center, +1 at edge
        assert float(radial_polynomial(1, 1, rho_center)[0]) == pytest.approx(0.0)
        assert float(radial_polynomial(1, 1, rho_edge)[0]) == pytest.approx(1.0)

    def test_zernike_orthonormality(self) -> None:
        """Verify discrete inner product orthogonality over circular aperture."""
        basis = ZernikeBasis(resolution=64, max_modes=11)
        pupil = basis.pupil_mask
        da = 4.0 / (64 * 64)  # Pixel area in [-1, 1] square

        # Test orthogonality between Defocus (j=4) and Primary Spherical (j=11)
        z4 = basis.get_mode_2d(4)[pupil]
        z11 = basis.get_mode_2d(11)[pupil]

        # Inner product should be close to zero
        inner_prod = np.sum(z4 * z11) * da
        assert abs(inner_prod) < 0.15

        # Normalization of Defocus should be close to area pi (~3.14)
        norm_z4 = np.sum(z4 * z4) * da
        assert norm_z4 == pytest.approx(np.pi, rel=0.10)

    def test_basis_decomposition_and_synthesis(self) -> None:
        """Verify decomposing synthesized modal phase recovers input coefficients."""
        basis = ZernikeBasis(resolution=64, max_modes=15)
        # Apply known modal amplitudes: 0.8 rad Defocus, -0.5 rad Coma X, 0.4 rad Astigmatism
        input_coeffs = np.zeros(15, dtype=np.float64)
        input_coeffs[3] = 0.80   # j=4 Defocus
        input_coeffs[4] = 0.40   # j=5 Astigmatism
        input_coeffs[7] = -0.50  # j=8 Coma X

        phase_map = basis.synthesize(input_coeffs)
        recovered_coeffs = basis.decompose(phase_map)

        assert recovered_coeffs[3] == pytest.approx(0.80, abs=0.03)
        assert recovered_coeffs[4] == pytest.approx(0.40, abs=0.03)
        assert recovered_coeffs[7] == pytest.approx(-0.50, abs=0.03)
        # Unexcited modes should be near zero
        assert abs(recovered_coeffs[1]) < 0.05
        assert abs(recovered_coeffs[10]) < 0.05


class TestKolmogorovTurbulence:
    """Tests for Kolmogorov atmospheric turbulence and phase screen generator."""

    def test_fried_parameter_scaling(self) -> None:
        """Verify Fried parameter scales inversely with Cn^2 and path length."""
        # Baseline
        r0_base = compute_fried_parameter(cn2=1e-14, link_distance_m=1000.0, wavelength_m=1550e-9)
        assert 0.01 < r0_base < 0.50  # Typically 5 to 15 cm

        # Stronger turbulence => smaller coherence diameter
        r0_strong = compute_fried_parameter(cn2=1e-13, link_distance_m=1000.0, wavelength_m=1550e-9)
        assert r0_strong < r0_base

        # Vacuum => large r0
        r0_vac = compute_fried_parameter(cn2=0.0, link_distance_m=1000.0, wavelength_m=1550e-9)
        assert r0_vac >= 100.0

    def test_greenwood_frequency_scaling(self) -> None:
        """Verify Greenwood frequency increases with wind speed and turbulence."""
        r0 = 0.05  # 5 cm
        fg_5 = compute_greenwood_frequency(r0, wind_speed_m_s=5.0)
        fg_10 = compute_greenwood_frequency(r0, wind_speed_m_s=10.0)
        assert fg_10 == pytest.approx(2.0 * fg_5)
        assert fg_5 > 0.0

    def test_noll_modal_variances(self) -> None:
        """Verify Noll theoretical variances follow Kolmogorov decay with radial order."""
        d_over_r0 = 2.0
        var_tilt = noll_theoretical_variance(2, d_over_r0)      # Tip/Tilt (n=1)
        var_defocus = noll_theoretical_variance(4, d_over_r0)   # Defocus (n=2)
        var_coma = noll_theoretical_variance(8, d_over_r0)      # Coma (n=3)
        var_spherical = noll_theoretical_variance(11, d_over_r0)# Spherical (n=4)

        # Higher radial orders contain progressively lower turbulent energy
        assert var_tilt > var_defocus
        assert var_defocus > var_coma
        assert var_coma > var_spherical

    def test_phase_screen_generator(self) -> None:
        """Verify dynamic phase screen generation produces valid WavefrontMap."""
        gen = KolmogorovPhaseScreenGenerator(
            KolmogorovTurbulenceConfig(cn2=2.5e-14, link_distance_m=500.0, seed=123),
            max_modes=15,
        )
        screen = gen.generate_static_screen()
        assert isinstance(screen, WavefrontMap)
        assert screen.rms_wfe_rad > 0.1
        assert screen.rms_wfe_nm > 20.0
        assert 0.0 < screen.strehl_ratio < 1.0


class TestShackHartmannSensor:
    """Tests for Shack-Hartmann sensor subaperture layout and slope sensing."""

    def test_subaperture_geometry(self) -> None:
        """Verify sensor creates correct active subapertures covering circular pupil."""
        cfg = SHWFSConfig(grid_dim=8, subaperture_threshold=0.5)
        shwfs = ShackHartmannSensor(cfg, resolution=64)
        assert len(shwfs.subapertures) == 64
        # Corner subapertures outside circular pupil should be inactive
        assert 40 <= shwfs.num_active <= 56
        assert shwfs.num_slopes == 2 * shwfs.num_active


    def test_tilt_sensing(self) -> None:
        """Verify sensor responds with uniform slopes to linear wavefront tilt."""
        cfg = SHWFSConfig(grid_dim=8, measurement_noise_std_rad=0.0)
        shwfs = ShackHartmannSensor(cfg, resolution=64)

        # Pure X-tilt phase: phi(x, y) = 2.0 * x
        lin = np.linspace(-1.0, 1.0, 64)
        x_grid, _ = np.meshgrid(lin, lin)
        tilt_phase = 2.0 * x_grid

        slopes = shwfs.measure_slopes(tilt_phase, add_noise=False)
        sx = slopes[: shwfs.num_active]
        sy = slopes[shwfs.num_active :]

        # X-slopes should be positive constant, Y-slopes near zero
        assert np.mean(sx) == pytest.approx(2.0, rel=0.10)
        assert abs(np.mean(sy)) < 0.05


class TestDeformableMirror:
    """Tests for 37-actuator Deformable Mirror."""

    def test_actuator_layout_count(self) -> None:
        """Verify 37 actuators arranged in concentric rings."""
        dm = DeformableMirror(DeformableMirrorConfig(num_actuators=37), resolution=64)
        assert dm.num_actuators == 37
        assert len(dm.actuators) == 37
        # Center actuator at (0, 0)
        assert dm.actuators[0].x == 0.0 and dm.actuators[0].y == 0.0

    def test_stroke_limits_and_saturation(self) -> None:
        """Verify actuator command clamping to max stroke."""
        cfg = DeformableMirrorConfig(max_stroke_um=3.5, loop_gain=1.0, leak_factor=1.0)
        dm = DeformableMirror(cfg, resolution=64)

        # Command excessive stroke
        large_cmd = np.full(37, 10.0)
        dm.apply_delta_commands(large_cmd)
        assert np.all(dm.commands_um <= 3.5)
        assert np.all(dm.commands_um >= -3.5)

    def test_reflective_optical_phase_shift(self) -> None:
        """Verify double-pass reflective optical phase shift OPD = 2 * delta_z."""
        dm = DeformableMirror(resolution=64, wavelength_m=1550e-9)
        # Push center actuator by 0.775 um (quarter-wave physical displacement)
        # Optical path difference OPD = 2 * 0.775 um = 1.55 um (exactly 1 wave = 2*pi rad)
        cmds = np.zeros(37)
        cmds[0] = 0.775
        dm.set_commands(cmds)

        phase_rad = dm.get_optical_phase_radians()
        center_phase = phase_rad[32, 32]
        assert center_phase == pytest.approx(2.0 * np.pi, rel=0.10)


class TestWavefrontReconstructor:
    """Tests for SVD Interaction Matrix calibration and modal/zonal reconstruction."""

    def test_modal_reconstruction_accuracy(self) -> None:
        """Verify reconstructor recovers known Zernike mode amplitudes from slopes."""
        basis = ZernikeBasis(resolution=64, max_modes=12)
        shwfs = ShackHartmannSensor(SHWFSConfig(grid_dim=8, measurement_noise_std_rad=0.0), resolution=64)
        dm = DeformableMirror(resolution=64)

        recon = WavefrontReconstructor(shwfs, basis, dm, max_modes=12)

        # Apply pure Defocus (j=4) with amplitude 1.2 rad
        defocus_phase = 1.2 * basis.get_mode_2d(4)
        slopes = shwfs.measure_slopes(defocus_phase, add_noise=False)

        recovered_modes = recon.reconstruct_zernike_modes(slopes)
        assert recovered_modes[3] == pytest.approx(1.2, rel=0.15)


class TestAdaptiveOpticsClosedLoop:
    """Tests for complete closed-loop AO system and SMF coupling."""

    def test_smf_coupling_flat_vs_aberrated(self) -> None:
        """Verify flat wavefront achieves near-optimal coupling while aberrated collapses."""
        pupil = np.hypot(*np.meshgrid(np.linspace(-1, 1, 64), np.linspace(-1, 1, 64))) <= 1.0

        # Flat unaberrated wavefront
        flat_phase = np.zeros((64, 64), dtype=np.float64)
        coupling_flat = compute_smf_coupling_efficiency(flat_phase, pupil)
        assert coupling_flat == pytest.approx(0.8145, rel=0.05)

        # Strongly aberrated wavefront (RMS = 2.0 rad)
        rng = np.random.default_rng(42)
        aberrated_phase = rng.normal(0.0, 2.0, size=(64, 64))
        coupling_aberrated = compute_smf_coupling_efficiency(aberrated_phase, pupil)
        assert coupling_aberrated < 0.10
        assert coupling_aberrated < coupling_flat

    def test_closed_loop_convergence(self) -> None:
        """Verify AO closed-loop reduces WFE RMS and increases Strehl ratio over time."""
        ao_sys = AdaptiveOpticsSystem(AOConfig(grid_resolution=64, max_zernike_modes=15), seed=42)

        # Create static turbulent phase with mixed modes
        turb_coeffs = np.zeros(15)
        turb_coeffs[3] = 0.90   # Defocus
        turb_coeffs[4] = 0.60   # Astigmatism
        turb_coeffs[7] = -0.50  # Coma
        turb_phase = ao_sys.basis.synthesize(turb_coeffs)

        # Initial step
        telem_initial = ao_sys.step(turb_phase, dt=0.002, timestamp_s=0.0)
        assert telem_initial.strehl_ratio_pre < 0.40

        # Run 25 closed-loop cycles
        for step in range(1, 26):
            telem = ao_sys.step(turb_phase, dt=0.002, timestamp_s=step * 0.002)

        # After convergence, residual WFE must be substantially lower and Strehl higher
        assert telem.corrected_wfe_rms_rad < telem_initial.raw_wfe_rms_rad * 0.45
        assert telem.strehl_ratio_post > telem_initial.strehl_ratio_pre * 2.0
        assert telem.smf_coupling_post > telem_initial.smf_coupling_pre
        assert telem.ao_optical_gain_db > 2.0

    def test_tri_stage_controller(self) -> None:
        """Verify TriStageController coordinates Gimbal, FSM, and AO."""
        tri_ctrl = TriStageController()
        tri_ctrl.reset()

        error = AngularError(horizontal=0.001, vertical=-0.0008, total=0.00128)
        cmd, state = tri_ctrl.update(error, dt=0.02, turbulent_phase_rad=None, timestamp_s=0.0)

        assert abs(cmd.pan_rate) > 0.0
        assert abs(cmd.tilt_rate) > 0.0
        assert state.fine_error_urad < state.coarse_error_urad
        assert state.strehl_ratio == 1.0

    def test_link_budget_with_ao_coupling(self) -> None:
        """Verify FSOCLinkBudget properly includes Strehl and SMF coupling."""
        budget = FSOCLinkBudget()
        # Degraded link without AO (Strehl = 0.08, coupling = 6%)
        res_no_ao = budget.evaluate(distance_m=1000.0, pointing_error_rad=1e-5, strehl_ratio=0.08, smf_coupling_efficiency=0.06)

        # Restored link with AO (Strehl = 0.85, coupling = 72%)
        res_with_ao = budget.evaluate(distance_m=1000.0, pointing_error_rad=1e-5, strehl_ratio=0.85, smf_coupling_efficiency=0.72)

        # AO link should have higher received power and larger link margin
        assert res_with_ao.rx_power_w > res_no_ao.rx_power_w * 5.0
        assert res_with_ao.link_margin_db > res_no_ao.link_margin_db + 8.0
        assert res_with_ao.ao_gain_db > 0.0

    def test_ao_benchmark_engine_run(self) -> None:
        """Verify AOBenchmarkEngine executes and produces metrics."""
        cfg = AOBenchmarkConfig(duration_s=0.05, dt=0.005, cn2=1e-14, link_distance_m=500.0)
        engine = AOBenchmarkEngine(cfg)
        metrics, telemetry, ao_sys = engine.run()

        assert len(telemetry) == 10
        assert metrics.r0_cm > 0.0
        assert metrics.greenwood_freq_hz > 0.0
        assert metrics.post_ao_mean_strehl >= metrics.pre_ao_mean_strehl
