"""Rotation matrices and Euler angle conversions.

Convention:
    - All angles in radians.
    - ZYX intrinsic rotation order: R = Rz(yaw) · Ry(pitch) · Rx(roll).
    - This is equivalent to XYZ extrinsic rotation order.
    - Elementary rotations are right-handed (positive = counter-clockwise
      when looking along the positive axis toward the origin).

Numerical stability:
    - Euler extraction uses atan2 for quadrant-correct results.
    - Gimbal lock near pitch = ±π/2 is handled with a tolerance check.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from fsoc_pat.core.errors import NumericalInstabilityError


def rotation_matrix_x(angle: float) -> NDArray[np.float64]:
    """Elementary rotation about the X-axis.

    Args:
        angle: Rotation angle (rad). Positive = CCW when viewed from +X.

    Returns:
        3×3 rotation matrix.
    """
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0,   c,  -s],
        [0.0,   s,   c],
    ], dtype=np.float64)


def rotation_matrix_y(angle: float) -> NDArray[np.float64]:
    """Elementary rotation about the Y-axis.

    Args:
        angle: Rotation angle (rad). Positive = CCW when viewed from +Y.

    Returns:
        3×3 rotation matrix.
    """
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [ c, 0.0,  s],
        [0.0, 1.0, 0.0],
        [-s, 0.0,  c],
    ], dtype=np.float64)


def rotation_matrix_z(angle: float) -> NDArray[np.float64]:
    """Elementary rotation about the Z-axis.

    Args:
        angle: Rotation angle (rad). Positive = CCW when viewed from +Z.

    Returns:
        3×3 rotation matrix.
    """
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [c,  -s, 0.0],
        [s,   c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def euler_to_rotation_matrix(
    roll: float,
    pitch: float,
    yaw: float,
) -> NDArray[np.float64]:
    """Convert ZYX intrinsic Euler angles to a rotation matrix.

    The rotation transforms vectors from the rotated frame to the
    reference frame: v_ref = R · v_body.

    Composition order: R = Rz(yaw) · Ry(pitch) · Rx(roll)

    Args:
        roll: Rotation about X-axis (rad).
        pitch: Rotation about Y-axis (rad).
        yaw: Rotation about Z-axis (rad).

    Returns:
        3×3 rotation matrix.
    """
    return rotation_matrix_z(yaw) @ rotation_matrix_y(pitch) @ rotation_matrix_x(roll)


def rotation_matrix_to_euler(R: NDArray[np.float64]) -> tuple[float, float, float]:
    """Extract ZYX intrinsic Euler angles from a rotation matrix.

    Handles gimbal lock (pitch near ±π/2) with a tolerance-based
    fallback that sets roll = 0 and extracts yaw from the remaining
    degrees of freedom.

    Args:
        R: 3×3 rotation matrix.

    Returns:
        Tuple of (roll, pitch, yaw) in radians.

    Raises:
        NumericalInstabilityError: If R is not a valid rotation matrix.
    """
    if R.shape != (3, 3):
        raise NumericalInstabilityError(
            f"Expected 3×3 rotation matrix, got shape {R.shape}"
        )

    # Check that R is orthogonal: R^T R ≈ I
    det = np.linalg.det(R)
    if abs(det - 1.0) > 1e-6:
        raise NumericalInstabilityError(
            f"Rotation matrix determinant = {det:.6f}, expected 1.0"
        )

    # Extract pitch from R[2,0] = -sin(pitch)
    sin_pitch = -R[2, 0]
    sin_pitch = float(np.clip(sin_pitch, -1.0, 1.0))

    GIMBAL_LOCK_THRESHOLD = 1.0 - 1e-6

    if abs(sin_pitch) > GIMBAL_LOCK_THRESHOLD:
        # Gimbal lock: pitch ≈ ±π/2
        # Set roll = 0, extract yaw from remaining elements
        pitch = math.copysign(math.pi / 2.0, sin_pitch)
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    else:
        pitch = math.asin(sin_pitch)
        cos_pitch = math.cos(pitch)
        roll = math.atan2(R[2, 1] / cos_pitch, R[2, 2] / cos_pitch)
        yaw = math.atan2(R[1, 0] / cos_pitch, R[0, 0] / cos_pitch)

    return roll, pitch, yaw


def is_valid_rotation_matrix(
    R: NDArray[np.float64],
    tol: float = 1e-6,
) -> bool:
    """Check if a matrix is a valid rotation matrix (det=1, orthogonal).

    Args:
        R: Matrix to check.
        tol: Tolerance for numerical comparison.

    Returns:
        True if R is a valid rotation matrix.
    """
    if R.shape != (3, 3):
        return False
    identity_check = np.allclose(R @ R.T, np.eye(3), atol=tol)
    det_check = abs(np.linalg.det(R) - 1.0) < tol
    return bool(identity_check and det_check)
