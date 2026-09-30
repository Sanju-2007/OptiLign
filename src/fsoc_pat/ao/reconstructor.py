"""Wavefront reconstructor using Interaction Matrix calibration and SVD pseudo-inversion.

Calculates the optical mapping between:
  1. Shack-Hartmann subaperture slope measurements s
  2. Zernike polynomial modal amplitudes a (Modal Reconstruction)
  3. Deformable mirror actuator displacement commands u (Zonal Reconstruction)

Uses Singular Value Decomposition (SVD) with condition number regularized
pseudo-inverse to reject unobservable modes and measurement noise.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from fsoc_pat.ao.deformable_mirror import DeformableMirror
from fsoc_pat.ao.shack_hartmann import ShackHartmannSensor
from fsoc_pat.ao.zernike import ZernikeBasis


class WavefrontReconstructor:
    """Wavefront reconstructor mapping slope vectors to modal/zonal corrections."""

    def __init__(
        self,
        shwfs: ShackHartmannSensor,
        basis: ZernikeBasis,
        dm: DeformableMirror,
        max_modes: int = 22,
        regularization: float = 1e-4,
    ) -> None:
        self.shwfs = shwfs
        self.basis = basis
        self.dm = dm
        self.max_modes = max_modes
        self.regularization = regularization

        # 1. Build Modal Interaction Matrix D_modal: shape (num_slopes, max_modes - 1)
        # Note: Mode 1 (Piston) has zero gradient, so we skip it (indices 2 to max_modes)
        self.reconstructed_modes = list(range(2, max_modes + 1))
        self.num_reconstructed_modes = len(self.reconstructed_modes)

        self.d_modal = self._calibrate_modal_interaction_matrix()
        self.r_modal = self._compute_regularized_reconstructor(self.d_modal)

        # 2. Build Zonal Interaction Matrix D_zonal: shape (num_slopes, num_actuators)
        self.d_zonal = self._calibrate_zonal_interaction_matrix()
        self.r_zonal = self._compute_regularized_reconstructor(self.d_zonal)

    def _calibrate_modal_interaction_matrix(self) -> NDArray[np.float64]:
        """Calibrate response of SH-WFS to unit Zernike modes.

        Applies each mode Z_j (j=2..J) with unit 1 radian RMS amplitude
        and measures the resulting slope vector.
        """
        num_slopes = self.shwfs.num_slopes
        d_mat = np.zeros((num_slopes, self.num_reconstructed_modes), dtype=np.float64)

        for col_idx, j in enumerate(self.reconstructed_modes):
            mode_phase = self.basis.get_mode_2d(j)
            slopes = self.shwfs.measure_slopes(mode_phase, add_noise=False)
            d_mat[:, col_idx] = slopes

        return d_mat

    def _calibrate_zonal_interaction_matrix(self) -> NDArray[np.float64]:
        """Calibrate response of SH-WFS to individual DM actuators.

        Pushes each actuator k by delta_u = 0.5 um and measures slope response.
        """
        num_slopes = self.shwfs.num_slopes
        num_act = self.dm.num_actuators
        d_mat = np.zeros((num_slopes, num_act), dtype=np.float64)

        delta_push_um = 0.5
        for k in range(num_act):
            test_cmd = np.zeros(num_act, dtype=np.float64)
            test_cmd[k] = delta_push_um
            self.dm.set_commands(test_cmd)
            dm_phase = self.dm.get_optical_phase_radians()
            slopes = self.shwfs.measure_slopes(dm_phase, add_noise=False)
            d_mat[:, k] = slopes / delta_push_um

        self.dm.reset()
        return d_mat

    def _compute_regularized_reconstructor(self, d_mat: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the regularized Moore-Penrose pseudo-inverse using SVD.

        D = U * S * V^T
        R = V * S_inv * U^T
        where singular values s_i below threshold are attenuated via Tikhonov:
        s_inv_i = s_i / (s_i^2 + alpha^2)
        """
        u, s, vt = np.linalg.svd(d_mat, full_matrices=False)
        max_s = float(np.max(s)) if s.size > 0 else 1.0
        alpha = self.regularization * max_s

        s_inv = s / (s ** 2 + alpha ** 2)
        r_mat = vt.T @ np.diag(s_inv) @ u.T
        return r_mat

    def reconstruct_zernike_modes(self, slopes: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reconstruct Zernike modal amplitudes from SH-WFS slope measurements.

        Args:
            slopes: 1D array of measured slopes.

        Returns:
            1D array of reconstructed Zernike coefficients [a_1=0, a_2, ..., a_J] in radians.
        """
        estimated_active = self.r_modal @ slopes
        full_coeffs = np.zeros(self.max_modes, dtype=np.float64)
        full_coeffs[1 : 1 + len(estimated_active)] = estimated_active
        return full_coeffs

    def reconstruct_actuator_deltas(self, slopes: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reconstruct DM actuator displacement updates (um) from slopes.

        Args:
            slopes: 1D array of measured slopes.

        Returns:
            1D array of actuator command deltas in micrometers.
        """
        delta_u_um = self.r_zonal @ slopes
        return delta_u_um
