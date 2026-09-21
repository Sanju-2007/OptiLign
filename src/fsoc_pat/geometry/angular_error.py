"""Angular pointing error computation.

Converts pixel-space measurements into physically meaningful angular
errors. This is a critical module — raw pixel error must NEVER be used
directly as a physical pointing error.

Pipeline:
    pixel → camera-frame ray → angular offset from boresight

Uses atan2 for numerical stability (not small-angle approximation).

Units:
    All angular outputs are in radians.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from fsoc_pat.core.types import AngularError, PixelCoord
from fsoc_pat.geometry.camera_model import CameraIntrinsics
from fsoc_pat.geometry.projection import pixel_to_ray


def compute_angular_error_from_pixel(
    pixel: PixelCoord,
    intrinsics: CameraIntrinsics,
) -> AngularError:
    """Compute angular error between boresight and a pixel measurement.

    The boresight corresponds to the principal point (cx, cy) on the
    image plane. This function computes the angular offset of the
    given pixel from that boresight.

    Uses atan2 for each axis independently, then computes total
    angular separation from the 3D ray angle.

    Args:
        pixel: Detected target pixel coordinate.
        intrinsics: Camera intrinsic parameters.

    Returns:
        AngularError with horizontal, vertical, and total components (rad).
    """
    # Normalized image coordinates relative to principal point
    xn = (pixel.u - intrinsics.cx) / intrinsics.fx
    yn = (pixel.v - intrinsics.cy) / intrinsics.fy

    # Angular offsets using atan for exact projection model
    # Horizontal: angle between boresight and the projection on the XY plane
    horizontal = math.atan(xn)

    # Vertical: angle between boresight and the projection on the XZ plane
    vertical = math.atan(yn)

    # Total angular separation: angle between boresight [1,0,0] and ray [1, xn, yn]
    # Using the dot-product formula:
    # cos(theta) = (1) / sqrt(1 + xn² + yn²)
    total = math.atan(math.sqrt(xn * xn + yn * yn))

    return AngularError(
        horizontal=horizontal,
        vertical=vertical,
        total=total,
    )


def compute_angular_separation(
    vec_a: NDArray[np.float64],
    vec_b: NDArray[np.float64],
) -> float:
    """Compute angular separation between two 3D unit vectors.

    Uses the numerically stable formula:
        theta = atan2(|a × b|, a · b)

    This avoids the acos instability near 0 and π.

    Args:
        vec_a: First unit vector, shape (3,).
        vec_b: Second unit vector, shape (3,).

    Returns:
        Angular separation in radians, in [0, π].
    """
    cross = np.cross(vec_a, vec_b)
    sin_angle = float(np.linalg.norm(cross))
    cos_angle = float(np.dot(vec_a, vec_b))

    return math.atan2(sin_angle, cos_angle)


def compute_angular_error_from_vectors(
    boresight: NDArray[np.float64],
    los: NDArray[np.float64],
) -> AngularError:
    """Compute angular error from boresight and LOS vectors in world frame.

    Decomposes the angular error into horizontal (azimuth) and vertical
    (elevation) components relative to the boresight direction.

    Args:
        boresight: Camera boresight unit vector in world frame, shape (3,).
        los: Line-of-sight unit vector to target in world frame, shape (3,).

    Returns:
        AngularError with horizontal, vertical, and total components (rad).
    """
    total = compute_angular_separation(boresight, los)

    # Decompose into horizontal and vertical components
    # Project both vectors onto a local frame defined by the boresight
    # For small angles, this is approximately the cross product components
    # For large angles, we use the full formulation

    cross = np.cross(boresight, los)
    cross_norm = float(np.linalg.norm(cross))

    if cross_norm < 1e-12:
        # Vectors are parallel or anti-parallel
        return AngularError(horizontal=0.0, vertical=0.0, total=total)

    # Build a local reference frame from the boresight
    # X = boresight, Z = up approximation, Y = Z × X
    # Use world up [0, 0, -1] (Z is down in our convention, so up = -Z)
    world_up = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    # Handle case where boresight is parallel to world up
    if abs(np.dot(boresight, world_up)) > 0.999:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    y_local = np.cross(world_up, boresight)
    y_norm = np.linalg.norm(y_local)
    if y_norm > 1e-12:
        y_local /= y_norm
    z_local = np.cross(boresight, y_local)

    # Project the LOS-boresight angular difference onto local axes
    # The cross product direction tells us the rotation axis
    # Components along y_local → vertical error
    # Components along z_local → horizontal error
    sin_total = math.sin(total)
    if sin_total < 1e-12:
        return AngularError(horizontal=0.0, vertical=0.0, total=total)

    rotation_axis = cross / cross_norm
    horizontal = math.asin(
        float(np.clip(np.dot(rotation_axis, z_local) * sin_total, -1.0, 1.0))
    )
    vertical = math.asin(
        float(np.clip(np.dot(rotation_axis, y_local) * sin_total, -1.0, 1.0))
    )

    return AngularError(horizontal=horizontal, vertical=vertical, total=total)
