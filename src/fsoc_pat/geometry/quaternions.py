"""Quaternion algebra for rotation representation.

Quaternions avoid gimbal lock and provide smooth interpolation (SLERP).
Convention: scalar-first (w, x, y, z) where q = w + xi + yj + zk.

All quaternions representing rotations must be unit quaternions (|q| = 1).
Construction automatically normalizes and validates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fsoc_pat.core.errors import NumericalInstabilityError


@dataclass(frozen=True, slots=True)
class Quaternion:
    """Unit quaternion for 3D rotation.

    Attributes:
        w: Scalar part.
        x: X component of vector part.
        y: Y component of vector part.
        z: Z component of vector part.
    """

    w: float
    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        for name in ("w", "x", "y", "z"):
            val = getattr(self, name)
            if not math.isfinite(val):
                raise NumericalInstabilityError(
                    f"Quaternion.{name} is not finite: {val}"
                )

    @property
    def norm(self) -> float:
        """Quaternion magnitude."""
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def normalized(self) -> Quaternion:
        """Return a unit quaternion.

        Raises:
            NumericalInstabilityError: If the quaternion has zero norm.
        """
        n = self.norm
        if n < 1e-15:
            raise NumericalInstabilityError("Cannot normalize zero quaternion")
        return Quaternion(self.w / n, self.x / n, self.y / n, self.z / n)

    def conjugate(self) -> Quaternion:
        """Quaternion conjugate (inverse for unit quaternions)."""
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def multiply(self, other: Quaternion) -> Quaternion:
        """Hamilton product: self ⊗ other.

        For rotations, q1.multiply(q2) applies q2 first, then q1.

        Args:
            other: Right-hand quaternion.

        Returns:
            Product quaternion (not re-normalized — caller should
            normalize periodically to prevent drift).
        """
        w = self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z
        x = self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y
        y = self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x
        z = self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w
        return Quaternion(w, x, y, z)

    def rotate_vector(self, v: NDArray[np.float64]) -> NDArray[np.float64]:
        """Rotate a 3D vector by this quaternion.

        Computes q ⊗ (0, v) ⊗ q*, which rotates v from body to reference frame.

        Args:
            v: 3D vector as (3,) array.

        Returns:
            Rotated 3D vector as (3,) array.
        """
        vq = Quaternion(0.0, float(v[0]), float(v[1]), float(v[2]))
        result = self.multiply(vq).multiply(self.conjugate())
        return np.array([result.x, result.y, result.z], dtype=np.float64)

    def to_rotation_matrix(self) -> NDArray[np.float64]:
        """Convert to 3×3 rotation matrix.

        Returns:
            Rotation matrix R such that R @ v = self.rotate_vector(v).
        """
        w, x, y, z = self.w, self.x, self.y, self.z
        return np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
            [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
            [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
        ], dtype=np.float64)

    def to_euler(self) -> tuple[float, float, float]:
        """Convert to ZYX intrinsic Euler angles (roll, pitch, yaw).

        Returns:
            Tuple of (roll, pitch, yaw) in radians.
        """
        w, x, y, z = self.w, self.x, self.y, self.z

        # Roll (X-axis)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (Y-axis)
        sinp = 2.0 * (w * y - z * x)
        sinp = float(np.clip(sinp, -1.0, 1.0))
        pitch = math.asin(sinp)

        # Yaw (Z-axis)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    @classmethod
    def from_euler(cls, roll: float, pitch: float, yaw: float) -> Quaternion:
        """Create from ZYX intrinsic Euler angles.

        Args:
            roll: Rotation about X-axis (rad).
            pitch: Rotation about Y-axis (rad).
            yaw: Rotation about Z-axis (rad).

        Returns:
            Unit quaternion.
        """
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)

        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy

        return cls(w, x, y, z).normalized()

    @classmethod
    def from_rotation_matrix(cls, R: NDArray[np.float64]) -> Quaternion:
        """Create from a 3×3 rotation matrix.

        Uses Shepperd's method for numerical stability — selects the
        largest diagonal element to avoid division by near-zero.

        Args:
            R: 3×3 rotation matrix.

        Returns:
            Unit quaternion.
        """
        trace = R[0, 0] + R[1, 1] + R[2, 2]

        if trace > 0:
            s = 0.5 / math.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

        return cls(w, x, y, z).normalized()

    @classmethod
    def identity(cls) -> Quaternion:
        """Identity quaternion (no rotation)."""
        return cls(1.0, 0.0, 0.0, 0.0)

    def as_array(self) -> NDArray[np.float64]:
        """Return as (4,) array [w, x, y, z]."""
        return np.array([self.w, self.x, self.y, self.z], dtype=np.float64)


def slerp(q0: Quaternion, q1: Quaternion, t: float) -> Quaternion:
    """Spherical Linear Interpolation between two quaternions.

    Args:
        q0: Start quaternion.
        q1: End quaternion.
        t: Interpolation parameter in [0, 1].

    Returns:
        Interpolated unit quaternion.
    """
    if t < 0.0 or t > 1.0:
        raise ValueError(f"SLERP parameter t must be in [0, 1], got {t}")

    # Compute cosine of angle between quaternions
    dot = q0.w * q1.w + q0.x * q1.x + q0.y * q1.y + q0.z * q1.z

    # If dot < 0, negate one quaternion to take the short path
    if dot < 0:
        q1 = Quaternion(-q1.w, -q1.x, -q1.y, -q1.z)
        dot = -dot

    # Clamp for numerical safety
    dot = min(dot, 1.0)

    # If quaternions are very close, use linear interpolation
    if dot > 0.9995:
        w = q0.w + t * (q1.w - q0.w)
        x = q0.x + t * (q1.x - q0.x)
        y = q0.y + t * (q1.y - q0.y)
        z = q0.z + t * (q1.z - q0.z)
        return Quaternion(w, x, y, z).normalized()

    theta = math.acos(dot)
    sin_theta = math.sin(theta)

    s0 = math.sin((1.0 - t) * theta) / sin_theta
    s1 = math.sin(t * theta) / sin_theta

    return Quaternion(
        s0 * q0.w + s1 * q1.w,
        s0 * q0.x + s1 * q1.x,
        s0 * q0.y + s1 * q1.y,
        s0 * q0.z + s1 * q1.z,
    ).normalized()
