"""Zernike polynomial mathematics, modal decomposition, and wavefront synthesis.

Implements analytical Zernike polynomials Z_j(rho, theta) using Noll's 1976 indexing:
  - Vectorized evaluation on Cartesian (x, y) grids
  - Orthonormal basis normalization over the unit disk
  - Analytical spatial derivatives (dZ/dx, dZ/dy) for wavefront slope sensing
  - Least-squares modal decomposition and phase reconstruction
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from fsoc_pat.ao.ao_types import NOLL_TABLE


def noll_to_nm(j: int) -> tuple[int, int]:
    """Convert 1-based Noll index j to radial order n and azimuthal frequency m.

    Args:
        j: Noll index (>= 1).

    Returns:
        Tuple of (n, m).
    """
    if j in NOLL_TABLE:
        return NOLL_TABLE[j][0], NOLL_TABLE[j][1]

    # General conversion for j > 22
    n = int(np.floor(np.sqrt(2 * j - 1) + 0.5)) - 1
    while (n + 1) * (n + 2) // 2 < j:
        n += 1
    m_val = j - n * (n + 1) // 2
    if n % 2 == 0:
        # n even => m even
        m_abs = 2 * ((m_val + 1) // 2) - 2
    else:
        # n odd => m odd
        m_abs = 2 * (m_val // 2) - 1
        m_abs = max(1, m_abs)

    m = m_abs if j % 2 == 0 else -m_abs
    if m_abs == 0:
        m = 0
    return n, m


def radial_polynomial(n: int, m: int, rho: NDArray[np.float64]) -> NDArray[np.float64]:
    """Calculate the Zernike radial polynomial R_n^{|m|}(rho).

    Formula:
      R_n^m(rho) = sum_{s=0}^{(n-m)/2} (-1)^s (n-s)! / [s! ((n+m)/2 - s)! ((n-m)/2 - s)!] * rho^(n-2s)

    Args:
        n: Radial degree (n >= 0).
        m: Azimuthal frequency (|m| <= n, n - |m| is even).
        rho: Radial coordinate array in [0, 1].

    Returns:
        Evaluated radial polynomial array with same shape as rho.
    """
    m_abs = abs(m)
    if (n - m_abs) % 2 != 0 or n < m_abs:
        return np.zeros_like(rho)

    k_max = (n - m_abs) // 2
    r_val = np.zeros_like(rho, dtype=np.float64)

    for s in range(k_max + 1):
        num = ((-1) ** s) * math.factorial(n - s)
        den = (
            math.factorial(s)
            * math.factorial((n + m_abs) // 2 - s)
            * math.factorial((n - m_abs) // 2 - s)
        )
        coeff = float(num) / float(den)
        exponent = n - 2 * s
        r_val += coeff * (rho ** exponent)

    return r_val


def zernike_polynomial(
    n: int,
    m: int,
    rho: NDArray[np.float64],
    theta: NDArray[np.float64],
    normalized: bool = True,
) -> NDArray[np.float64]:
    """Evaluate a single Zernike polynomial mode on coordinates (rho, theta).

    Orthonormal on unit disk:
      integral_{pupil} Z_i * Z_j * r dr dtheta = pi * delta_ij

    Normalization factor N_n^m:
      sqrt(n + 1) for m = 0
      sqrt(2 * (n + 1)) for m != 0

    Args:
        n: Radial degree.
        m: Azimuthal frequency.
        rho: Normalized radial coordinates in [0, 1].
        theta: Azimuthal angle in radians.
        normalized: Whether to apply the Noll orthonormalization factor.

    Returns:
        2D array of polynomial values.
    """
    r_poly = radial_polynomial(n, m, rho)

    if m == 0:
        norm = math.sqrt(n + 1) if normalized else 1.0
        return norm * r_poly
    elif m > 0:
        norm = math.sqrt(2.0 * (n + 1)) if normalized else 1.0
        return norm * r_poly * np.cos(m * theta)
    else:
        norm = math.sqrt(2.0 * (n + 1)) if normalized else 1.0
        return norm * r_poly * np.sin(abs(m) * theta)


def zernike_noll(
    j: int,
    rho: NDArray[np.float64],
    theta: NDArray[np.float64],
    normalized: bool = True,
) -> NDArray[np.float64]:
    """Evaluate Zernike polynomial by 1-based Noll index j.

    Args:
        j: Noll index (1 = Piston, 2 = Tip, 3 = Tilt, 4 = Defocus, etc.).
        rho: Radial coordinate array [0, 1].
        theta: Azimuthal angle array [-pi, pi].
        normalized: Whether to use orthonormal normalization.

    Returns:
        Array of polynomial values.
    """
    n, m = noll_to_nm(j)
    return zernike_polynomial(n, m, rho, theta, normalized=normalized)


class ZernikeBasis:
    """Precomputed orthonormal Zernike polynomial basis over a 2D circular pupil grid.

    Provides high-speed modal decomposition, phase reconstruction, and spatial
    derivatives for wavefront sensor simulation.
    """

    def __init__(self, resolution: int = 64, max_modes: int = 22) -> None:
        """Initialize Zernike basis on a square grid of given resolution.

        Args:
            resolution: Grid dimension (e.g. 64 for 64x64 pupil).
            max_modes: Number of Zernike modes to precompute (default 22).
        """
        self.resolution = resolution
        self.max_modes = max_modes

        # Generate normalized Cartesian grid in [-1, 1]
        lin = np.linspace(-1.0, 1.0, resolution)
        self.x, self.y = np.meshgrid(lin, lin)
        self.dx = float(lin[1] - lin[0])

        self.rho = np.hypot(self.x, self.y)
        self.theta = np.arctan2(self.y, self.x)

        # Circular pupil mask (rho <= 1.0)
        self.pupil_mask = self.rho <= 1.0
        self.num_pupil_pixels = int(np.sum(self.pupil_mask))

        # Evaluate and cache Zernike modes: shape (max_modes, num_pupil_pixels)
        # Using 1D flattened indices inside the pupil for fast vector operations
        self.basis_matrix = np.zeros((max_modes, self.num_pupil_pixels), dtype=np.float64)
        self.modes_2d = np.zeros((max_modes, resolution, resolution), dtype=np.float64)

        for mode_idx in range(max_modes):
            j = mode_idx + 1  # 1-based Noll index
            z_full = zernike_noll(j, self.rho, self.theta, normalized=True)
            z_full[~self.pupil_mask] = 0.0
            self.modes_2d[mode_idx] = z_full
            self.basis_matrix[mode_idx] = z_full[self.pupil_mask]

        # Precompute pseudo-inverse for fast modal decomposition: a = P_inv * phi
        # For an orthonormal basis on continuous disk, <Z_i, Z_j> = pi * delta_ij
        # On discrete grid: B_pinv = (B * B^T)^(-1) * B
        bt_b = self.basis_matrix @ self.basis_matrix.T
        self.projection_matrix = np.linalg.pinv(bt_b) @ self.basis_matrix

    def decompose(self, phase_map: NDArray[np.float64]) -> NDArray[np.float64]:
        """Project a 2D phase map onto the Zernike modal basis.

        Args:
            phase_map: 2D array of phase values (rad) with shape (resolution, resolution).

        Returns:
            1D array of Zernike coefficients [a_1, a_2, ..., a_J] in radians.
        """
        valid_phase = phase_map[self.pupil_mask]
        coeffs = self.projection_matrix @ valid_phase
        return coeffs

    def synthesize(self, coefficients: Sequence[float]) -> NDArray[np.float64]:
        """Reconstruct 2D phase map from a sequence of Zernike coefficients.

        phi(x, y) = sum_{j=1}^J a_j * Z_j(x, y)

        Args:
            coefficients: Sequence of modal amplitudes [a_1, ..., a_J].

        Returns:
            2D phase map array of shape (resolution, resolution). Outside pupil is 0.
        """
        out = np.zeros((self.resolution, self.resolution), dtype=np.float64)
        n_modes = min(len(coefficients), self.max_modes)
        for i in range(n_modes):
            coeff = float(coefficients[i])
            if abs(coeff) > 1e-12:
                out += coeff * self.modes_2d[i]
        out[~self.pupil_mask] = 0.0
        return out

    def get_mode_2d(self, j: int) -> NDArray[np.float64]:
        """Get 2D array of a specific mode j (1-based)."""
        idx = j - 1
        if 0 <= idx < self.max_modes:
            return self.modes_2d[idx].copy()
        # Compute on the fly for higher modes
        z = zernike_noll(j, self.rho, self.theta, normalized=True)
        z[~self.pupil_mask] = 0.0
        return z

    def compute_derivatives(self, j: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute spatial phase derivatives (dZ/dx, dZ/dy) for mode j.

        Used to calculate Shack-Hartmann subaperture wavefront slopes.

        Args:
            j: 1-based Noll index.

        Returns:
            Tuple of (grad_x, grad_y) arrays of shape (resolution, resolution).
        """
        mode = self.get_mode_2d(j)
        # Numerical central difference gradient
        grad_y, grad_x = np.gradient(mode, self.dx, self.dx)
        grad_x[~self.pupil_mask] = 0.0
        grad_y[~self.pupil_mask] = 0.0
        return grad_x, grad_y
