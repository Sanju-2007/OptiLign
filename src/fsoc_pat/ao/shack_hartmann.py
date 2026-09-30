"""Shack-Hartmann Wavefront Sensor (SH-WFS) optical simulation.

Divides the telescope aperture into an array of micro-lenslet subapertures.
For each subaperture:
  - Integrates local wavefront phase gradients (dphi/dx, dphi/dy)
  - Computes focal spot centroid displacements (delta_x, delta_y)
  - Adds photon/detector readout noise
  - Generates the standard wavefront slope measurement vector s
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from fsoc_pat.ao.ao_types import SHWFSConfig


@dataclass
class Subaperture:
    """A single Shack-Hartmann micro-lenslet subaperture.

    Attributes:
        index: Sequential 0-based index.
        row: Grid row (0 to grid_dim - 1).
        col: Grid column (0 to grid_dim - 1).
        center_x: Normalized center x-coordinate in [-1, 1].
        center_y: Normalized center y-coordinate in [-1, 1].
        pixel_indices_y: Grid row slice or boolean mask for pixels in this subaperture.
        pixel_indices_x: Grid col slice or boolean mask for pixels in this subaperture.
        illumination_fraction: Fraction of subaperture area within the circular pupil.
        is_active: Whether illumination fraction exceeds threshold.
    """

    index: int
    row: int
    col: int
    center_x: float
    center_y: float
    r_slice: slice
    c_slice: slice
    illumination_fraction: float
    is_active: bool


class ShackHartmannSensor:
    """Shack-Hartmann Wavefront Sensor (SH-WFS).

    Simulates the lenslet array, spot centroiding, and slope vector generation.
    """

    def __init__(
        self,
        config: SHWFSConfig | None = None,
        resolution: int = 64,
        seed: int = 42,
    ) -> None:
        self.config = config or SHWFSConfig()
        self.resolution = resolution
        self._rng = np.random.default_rng(seed)

        # Build subaperture geometry
        self.subapertures: list[Subaperture] = []
        self._build_subapertures()

        self.active_subapertures = [s for s in self.subapertures if s.is_active]
        self.num_active = len(self.active_subapertures)
        self.num_slopes = 2 * self.num_active  # (x and y slopes)

    def _build_subapertures(self) -> None:
        """Map micro-lenslet subapertures to the 2D simulation pixel grid."""
        dim = self.config.grid_dim
        sub_px = self.resolution // dim
        lin = np.linspace(-1.0, 1.0, self.resolution)
        x_grid, y_grid = np.meshgrid(lin, lin)
        pupil_mask = np.hypot(x_grid, y_grid) <= 1.0

        sub_idx = 0
        for r in range(dim):
            r_start = r * sub_px
            r_end = (r + 1) * sub_px if r < dim - 1 else self.resolution
            r_slice = slice(r_start, r_end)

            for c in range(dim):
                c_start = c * sub_px
                c_end = (c + 1) * sub_px if c < dim - 1 else self.resolution
                c_slice = slice(c_start, c_end)

                sub_mask = pupil_mask[r_slice, c_slice]
                total_px = (r_end - r_start) * (c_end - c_start)
                fraction = float(np.sum(sub_mask)) / float(max(1, total_px))

                cx = float(np.mean(x_grid[r_slice, c_slice]))
                cy = float(np.mean(y_grid[r_slice, c_slice]))

                is_active = fraction >= self.config.subaperture_threshold
                self.subapertures.append(
                    Subaperture(
                        index=sub_idx,
                        row=r,
                        col=c,
                        center_x=cx,
                        center_y=cy,
                        r_slice=r_slice,
                        c_slice=c_slice,
                        illumination_fraction=fraction,
                        is_active=is_active,
                    )
                )
                sub_idx += 1

    def measure_slopes(
        self,
        phase_map: NDArray[np.float64],
        add_noise: bool = True,
    ) -> NDArray[np.float64]:
        """Compute the average wavefront gradient (slopes) across all active subapertures.

        Args:
            phase_map: 2D array of phase values (rad) of shape (resolution, resolution).
            add_noise: Whether to inject sensor detector readout noise.

        Returns:
            1D array of slopes of length 2 * num_active:
              slopes = [s_x_1, s_x_2, ..., s_x_K, s_y_1, s_y_2, ..., s_y_K] in radians.
        """
        # Phase gradient using finite differences in normalized coordinates [-1, 1]
        lin = np.linspace(-1.0, 1.0, self.resolution)
        dx = float(lin[1] - lin[0])
        grad_y, grad_x = np.gradient(phase_map, dx, dx)

        sx_list: list[float] = []
        sy_list: list[float] = []

        for sub in self.active_subapertures:
            gx_sub = grad_x[sub.r_slice, sub.c_slice]
            gy_sub = grad_y[sub.r_slice, sub.c_slice]

            # Mean slope across subaperture
            sx = float(np.mean(gx_sub))
            sy = float(np.mean(gy_sub))

            if add_noise and self.config.measurement_noise_std_rad > 0.0:
                noise_x = float(self._rng.normal(0.0, self.config.measurement_noise_std_rad))
                noise_y = float(self._rng.normal(0.0, self.config.measurement_noise_std_rad))
                sx += noise_x
                sy += noise_y

            sx_list.append(sx)
            sy_list.append(sy)

        return np.concatenate([np.array(sx_list), np.array(sy_list)])

    def slopes_to_spot_displacements(
        self, slopes: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Convert slope measurements to physical focal spot centroid displacements (m).

        Formula:
          Delta_x = f_lenslet * s_x * (lambda / (2 * pi))
          Delta_y = f_lenslet * s_y * (lambda / (2 * pi))

        Args:
            slopes: Slope vector of length 2 * num_active.

        Returns:
            Tuple of (delta_x_um, delta_y_um) arrays in micrometers.
        """
        k = 2.0 * math.pi / self.config.wavelength_m
        half = self.num_active
        sx = slopes[:half]
        sy = slopes[half:]

        # Angular tilt in radians theta = slope / k
        theta_x = sx / k
        theta_y = sy / k

        delta_x_m = self.config.lenslet_focal_length_m * theta_x
        delta_y_m = self.config.lenslet_focal_length_m * theta_y

        return delta_x_m * 1e6, delta_y_m * 1e6
