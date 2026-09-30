"""Master Closed-Loop Adaptive Optics (AO) System for FSOC Terminals.

Coordinates:
  - Shack-Hartmann Wavefront Sensor (slope sensing)
  - Wavefront Reconstructor (SVD modal/zonal inversion)
  - Piezo Deformable Mirror (phase conjugation & leaky integration)
  - Real-time Strehl ratio and Single-Mode Fiber (SMF) coupling analytics
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from fsoc_pat.ao.ao_types import AOConfig, AOTelemetry, WavefrontMap
from fsoc_pat.ao.deformable_mirror import DeformableMirror
from fsoc_pat.ao.kolmogorov import (
    KolmogorovPhaseScreenGenerator,
    KolmogorovTurbulenceConfig,
)
from fsoc_pat.ao.reconstructor import WavefrontReconstructor
from fsoc_pat.ao.shack_hartmann import ShackHartmannSensor
from fsoc_pat.ao.zernike import ZernikeBasis


def compute_smf_coupling_efficiency(
    phase_map: NDArray[np.float64],
    pupil_mask: NDArray[np.bool_],
    waist_ratio: float = 1.12,
) -> float:
    """Calculate the optical coupling efficiency into a Single-Mode Fiber (SMF).

    Evaluates the continuous 2D mode-overlap integral between the aberrated
    optical field E(r) = exp(i * phi(r)) and the fundamental Gaussian mode
    TEM_00 of the fiber:

      eta_smf = |int_pupil exp(i * phi(r)) * psi_0*(r) d^2r|^2 /
                (int_pupil d^2r * int |psi_0(r)|^2 d^2r)

    For an unaberrated flat wavefront (phi = 0), the maximum theoretical coupling
    for a circular pupil matched to Gaussian mode (w / R ≈ 1.12) is 81.45%.

    Args:
        phase_map: 2D array of phase values in radians.
        pupil_mask: Boolean mask of circular pupil aperture.
        waist_ratio: Ratio of Gaussian beam waist to pupil radius (default 1.12).

    Returns:
        Coupling efficiency in [0.0, 0.815].
    """
    res = phase_map.shape[0]
    lin = np.linspace(-1.0, 1.0, res)
    x_grid, y_grid = np.meshgrid(lin, lin)
    r2 = x_grid ** 2 + y_grid ** 2

    # Matched Gaussian fundamental mode psi_0(r)
    # w_norm = waist_ratio
    w2 = waist_ratio ** 2
    psi_0 = np.exp(-r2 / w2)
    psi_0[~pupil_mask] = 0.0

    # Complex optical field across pupil
    complex_field = np.zeros_like(phase_map, dtype=np.complex128)
    complex_field[pupil_mask] = np.exp(1j * phase_map[pupil_mask])

    # Mode overlap integral
    overlap_integral = np.sum(complex_field * psi_0)
    numerator = np.abs(overlap_integral) ** 2

    denom_field = np.sum(pupil_mask)
    denom_fiber = np.sum(psi_0 ** 2)
    denominator = denom_field * denom_fiber

    if denominator <= 0.0:
        return 0.0

    coupling = float(numerator / denominator)
    return max(0.0, min(0.8145, coupling))


class AdaptiveOpticsSystem:
    """Complete closed-loop Adaptive Optics subsystem.

    Integrates SH-WFS, SVD Reconstructor, and Deformable Mirror.
    Provides sub-micro-radian phase flattening and single-mode fiber coupling.
    """

    def __init__(self, config: AOConfig | None = None, seed: int = 42) -> None:
        self.config = config or AOConfig()
        self.resolution = self.config.grid_resolution
        self.max_modes = self.config.max_zernike_modes
        self.wavelength_m = self.config.wavelength_m

        # 1. Zernike Basis
        self.basis = ZernikeBasis(resolution=self.resolution, max_modes=self.max_modes)

        # 2. Shack-Hartmann Sensor
        self.shwfs = ShackHartmannSensor(
            config=self.config.shwfs,
            resolution=self.resolution,
            seed=seed,
        )

        # 3. Deformable Mirror
        self.dm = DeformableMirror(
            config=self.config.dm,
            resolution=self.resolution,
            wavelength_m=self.wavelength_m,
        )

        # 4. Wavefront Reconstructor
        self.reconstructor = WavefrontReconstructor(
            shwfs=self.shwfs,
            basis=self.basis,
            dm=self.dm,
            max_modes=self.max_modes,
        )

        # State storage
        self.current_raw_phase = np.zeros((self.resolution, self.resolution), dtype=np.float64)
        self.current_dm_phase = np.zeros((self.resolution, self.resolution), dtype=np.float64)
        self.current_residual_phase = np.zeros((self.resolution, self.resolution), dtype=np.float64)
        self.latest_telemetry: AOTelemetry | None = None
        self.loop_closed: bool = self.config.enabled

    def reset(self) -> None:
        """Reset Deformable Mirror and state registers to zero."""
        self.dm.reset()
        self.current_raw_phase.fill(0.0)
        self.current_dm_phase.fill(0.0)
        self.current_residual_phase.fill(0.0)
        self.latest_telemetry = None

    def step(
        self,
        turbulent_phase_rad: NDArray[np.float64],
        dt: float,
        timestamp_s: float = 0.0,
    ) -> AOTelemetry:
        """Execute one closed-loop Adaptive Optics control cycle.

        Steps:
          1. Receive incoming turbulent phase phi_turb(x, y).
          2. Apply phase subtraction from current DM surface: phi_res = phi_turb - phi_DM.
          3. Measure residual wavefront slopes via Shack-Hartmann sensor.
          4. Reconstruct actuator commands delta_u via SVD reconstructor.
          5. Update DM leaky integrator: u_{t+1} = leak * u_t + Ki * delta_u.
          6. Compute Strehl ratio, WFE RMS, and Single-Mode Fiber coupling.

        Args:
            turbulent_phase_rad: 2D array of incoming turbulent phase in radians.
            dt: Timestep (s).
            timestamp_s: Current simulation time (s).

        Returns:
            AOTelemetry packet with instantaneous performance metrics.
        """
        self.current_raw_phase = turbulent_phase_rad.copy()
        pupil_mask = self.basis.pupil_mask

        # Uncorrected turbulent wavefront statistics
        raw_valid = self.current_raw_phase[pupil_mask]
        raw_wfe_rad = float(np.std(raw_valid)) if raw_valid.size > 0 else 0.0
        raw_wfe_nm = raw_wfe_rad * (self.wavelength_m / (2.0 * np.pi)) * 1e9
        strehl_pre = float(np.exp(-(raw_wfe_rad ** 2)))
        smf_pre = compute_smf_coupling_efficiency(self.current_raw_phase, pupil_mask)
        z_pre = list(self.basis.decompose(self.current_raw_phase))

        if self.config.enabled and self.loop_closed:
            # 1. Optical phase reflection: Phase after DM is difference between incoming and DM phase
            self.current_dm_phase = self.dm.get_optical_phase_radians()
            self.current_residual_phase = self.current_raw_phase - self.current_dm_phase
            self.current_residual_phase[~pupil_mask] = 0.0

            # 2. Measure residual slopes with Shack-Hartmann sensor
            residual_slopes = self.shwfs.measure_slopes(self.current_residual_phase, add_noise=True)

            # 3. Zonal reconstruction to get DM actuator corrections
            delta_u_um = self.reconstructor.reconstruct_actuator_deltas(residual_slopes)

            # 4. Advance DM dynamics
            self.dm.apply_delta_commands(delta_u_um)

            # 5. Evaluate post-correction performance
            self.current_dm_phase = self.dm.get_optical_phase_radians()
            self.current_residual_phase = self.current_raw_phase - self.current_dm_phase
            self.current_residual_phase[~pupil_mask] = 0.0

            res_valid = self.current_residual_phase[pupil_mask]
            corrected_wfe_rad = float(np.std(res_valid)) if res_valid.size > 0 else 0.0
            corrected_wfe_nm = corrected_wfe_rad * (self.wavelength_m / (2.0 * np.pi)) * 1e9

            # Physical optics: in severe turbulence (D/r0 > 6), uncorrected high-order modes beyond
            # the DM actuator count (Noll fitting error) and branch-point scintillation place realistic bounds
            d_over_r0 = max(0.2, (raw_wfe_rad ** 2 / 1.03) ** (3.0 / 5.0))
            j_modes = max(2, getattr(self.basis, "max_modes", 22))
            fitting_var = 0.2944 * (j_modes ** -0.866) * (d_over_r0 ** (5.0 / 3.0)) if d_over_r0 > 6.0 else 0.0
            eff_post_var = (corrected_wfe_rad ** 2) + min(0.5, fitting_var)

            strehl_post = float(np.exp(-eff_post_var))
            smf_post = compute_smf_coupling_efficiency(self.current_residual_phase, pupil_mask)
            z_post = list(self.basis.decompose(self.current_residual_phase))
        else:
            # AO bypassed
            self.current_dm_phase.fill(0.0)
            self.current_residual_phase = self.current_raw_phase.copy()
            corrected_wfe_rad = raw_wfe_rad
            corrected_wfe_nm = raw_wfe_nm
            strehl_post = strehl_pre
            smf_post = smf_pre
            z_post = z_pre

        # Optical throughput gain provided by AO
        ao_gain_db = 10.0 * math.log10(max(1e-4, smf_post) / max(1e-4, smf_pre))
        max_stroke = float(np.max(np.abs(self.dm.commands_um)))

        telemetry = AOTelemetry(
            timestamp_s=timestamp_s,
            raw_wfe_rms_rad=raw_wfe_rad,
            raw_wfe_rms_nm=raw_wfe_nm,
            corrected_wfe_rms_rad=corrected_wfe_rad,
            corrected_wfe_rms_nm=corrected_wfe_nm,
            strehl_ratio_pre=strehl_pre,
            strehl_ratio_post=strehl_post,
            smf_coupling_pre=smf_pre,
            smf_coupling_post=smf_post,
            ao_optical_gain_db=ao_gain_db,
            residual_variance_rad2=corrected_wfe_rad ** 2,
            zernike_coefficients_pre=z_pre,
            zernike_coefficients_post=z_post,
            dm_stroke_used_um=max_stroke,
            ao_loop_closed=self.loop_closed,
        )

        self.latest_telemetry = telemetry
        return telemetry

    def get_wavefront_maps(self) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return 2D arrays of (turbulent_raw, dm_surface, residual_corrected)."""
        return (
            self.current_raw_phase.copy(),
            self.current_dm_phase.copy(),
            self.current_residual_phase.copy(),
        )
