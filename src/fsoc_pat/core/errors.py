"""Custom exceptions and numerical validation utilities.

All numerical failures must fail loudly — never silently continue
with NaN, Inf, or invalid matrix states. This module provides the
exception hierarchy and validation functions to enforce that rule.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class OptiLignError(Exception):
    """Base exception for all OptiLign errors."""


class ConfigurationError(OptiLignError):
    """Invalid or inconsistent configuration."""


class NumericalInstabilityError(OptiLignError):
    """Numerical instability detected (NaN, Inf, divergence)."""


class CovarianceError(NumericalInstabilityError):
    """Covariance matrix is not valid (not PSD, contains NaN, etc.)."""


class SimulationError(OptiLignError):
    """Error during simulation execution."""


class CalibrationError(OptiLignError):
    """Camera calibration or geometric configuration error."""


class ModelLoadError(OptiLignError):
    """Failed to load an AI model or external resource."""


class ControllerError(OptiLignError):
    """Controller computation failure (instability, infeasible MPC, etc.)."""


class DetectorError(OptiLignError):
    """Detector computation failure."""


# ---------------------------------------------------------------------------
# Numerical validation utilities
# ---------------------------------------------------------------------------

def check_finite(value: float, name: str) -> None:
    """Raise NumericalInstabilityError if value is NaN or Inf.

    Args:
        value: Scalar value to check.
        name: Descriptive name for error messages.

    Raises:
        NumericalInstabilityError: If value is not finite.
    """
    if not math.isfinite(value):
        raise NumericalInstabilityError(f"{name} is not finite: {value}")


def check_array_finite(arr: NDArray[np.float64], name: str) -> None:
    """Raise NumericalInstabilityError if array contains NaN or Inf.

    Args:
        arr: Array to check.
        name: Descriptive name for error messages.

    Raises:
        NumericalInstabilityError: If any element is not finite.
    """
    if not np.all(np.isfinite(arr)):
        bad_indices = np.argwhere(~np.isfinite(arr))
        raise NumericalInstabilityError(
            f"{name} contains non-finite values at indices: {bad_indices.tolist()}"
        )


def check_matrix_dimensions(
    matrix: NDArray[np.float64],
    expected_shape: tuple[int, ...],
    name: str,
) -> None:
    """Validate matrix dimensions.

    Args:
        matrix: Matrix to check.
        expected_shape: Expected shape tuple.
        name: Descriptive name for error messages.

    Raises:
        NumericalInstabilityError: If dimensions don't match.
    """
    if matrix.shape != expected_shape:
        raise NumericalInstabilityError(
            f"{name}: expected shape {expected_shape}, got {matrix.shape}"
        )


def check_positive_definite(matrix: NDArray[np.float64], name: str) -> None:
    """Validate that a matrix is symmetric positive semi-definite.

    Uses Cholesky decomposition as the primary check. Falls back to
    eigenvalue check if Cholesky fails.

    Args:
        matrix: Square matrix to check.
        name: Descriptive name for error messages.

    Raises:
        CovarianceError: If the matrix is not PSD.
    """
    check_array_finite(matrix, name)

    n = matrix.shape[0]
    if matrix.shape != (n, n):
        raise CovarianceError(f"{name}: expected square matrix, got shape {matrix.shape}")

    # Check symmetry (within tolerance)
    if not np.allclose(matrix, matrix.T, atol=1e-10):
        max_asym = float(np.max(np.abs(matrix - matrix.T)))
        raise CovarianceError(
            f"{name}: matrix is not symmetric (max asymmetry: {max_asym:.2e})"
        )

    # Check positive semi-definiteness via eigenvalues
    eigenvalues = np.linalg.eigvalsh(matrix)
    min_eigenvalue = float(np.min(eigenvalues))
    if min_eigenvalue < -1e-10:
        raise CovarianceError(
            f"{name}: matrix is not PSD (min eigenvalue: {min_eigenvalue:.2e})"
        )


def symmetrize(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Force a matrix to be symmetric by averaging with its transpose.

    This is a standard numerical stabilization technique for covariance
    matrices that may accumulate small asymmetries from floating-point
    arithmetic.

    Args:
        matrix: Square matrix.

    Returns:
        Symmetrized matrix.
    """
    return (matrix + matrix.T) / 2.0


def check_positive(value: float, name: str) -> None:
    """Raise ConfigurationError if value is not positive.

    Args:
        value: Scalar value to check.
        name: Descriptive name for error messages.
    """
    if value <= 0:
        raise ConfigurationError(f"{name} must be positive, got {value}")


def check_non_negative(value: float, name: str) -> None:
    """Raise ConfigurationError if value is negative.

    Args:
        value: Scalar value to check.
        name: Descriptive name for error messages.
    """
    if value < 0:
        raise ConfigurationError(f"{name} must be non-negative, got {value}")
