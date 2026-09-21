"""3D-to-pixel projection and pixel-to-ray inverse projection.

Implements the core pinhole camera model:

Forward:
    3D world point
    → camera frame (via transforms)
    → normalized coordinates (x/z, y/z)
    → pixel coordinates (u = fx·xn + cx, v = fy·yn + cy)

Inverse:
    Pixel (u, v)
    → normalized coordinates ((u-cx)/fx, (v-cy)/fy)
    → camera-frame ray direction (xn, yn, 1) normalized

Handles:
    - Points behind the camera (z <= 0): returns None
    - Points outside FOV: returns PixelCoord with is_in_fov flag
    - FOV boundary conditions
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fsoc_pat.core.state import CameraState
from fsoc_pat.core.types import NormalizedCoord, PixelCoord
from fsoc_pat.geometry.camera_model import CameraIntrinsics
from fsoc_pat.geometry.transforms import world_to_camera_from_state


def project_point_to_pixel(
    point_world: NDArray[np.float64],
    camera_state: CameraState,
    intrinsics: CameraIntrinsics,
) -> PixelCoord | None:
    """Project a 3D world point onto the image plane.

    Pipeline: world → camera frame → normalized → pixel

    Args:
        point_world: 3D point in world frame, shape (3,).
        camera_state: Camera position and orientation.
        intrinsics: Camera intrinsic parameters.

    Returns:
        PixelCoord if the point is in front of the camera, else None.
        Note: The returned pixel may be outside the image bounds (outside FOV).
    """
    # Transform to camera frame
    point_camera = world_to_camera_from_state(point_world, camera_state)

    # Camera convention: X = forward (optical axis)
    # Point must be in front of the camera (positive X)
    if point_camera[0] <= 0:
        return None

    # Normalized image coordinates
    # In our convention: X = forward, Y = right, Z = down
    # Normalized coords: xn = Y/X (horizontal), yn = Z/X (vertical)
    xn = point_camera[1] / point_camera[0]
    yn = point_camera[2] / point_camera[0]

    # Pixel coordinates
    u = intrinsics.fx * xn + intrinsics.cx
    v = intrinsics.fy * yn + intrinsics.cy

    return PixelCoord(u=float(u), v=float(v))


def project_to_normalized(
    point_camera: NDArray[np.float64],
) -> NormalizedCoord | None:
    """Project a camera-frame point to normalized coordinates.

    Args:
        point_camera: 3D point in camera frame, shape (3,).

    Returns:
        NormalizedCoord if point is in front of camera, else None.
    """
    if point_camera[0] <= 0:
        return None

    return NormalizedCoord(
        x=float(point_camera[1] / point_camera[0]),
        y=float(point_camera[2] / point_camera[0]),
    )


def pixel_to_ray(
    pixel: PixelCoord,
    intrinsics: CameraIntrinsics,
) -> NDArray[np.float64]:
    """Convert a pixel coordinate to a camera-frame ray direction.

    The ray originates at the camera center and passes through the
    pixel on the image plane. Returned as a unit vector in camera frame.

    Convention: X = forward, Y = right, Z = down

    Args:
        pixel: Pixel coordinate (u, v).
        intrinsics: Camera intrinsic parameters.

    Returns:
        Unit ray direction in camera frame, shape (3,).
    """
    # Normalized image coordinates
    xn = (pixel.u - intrinsics.cx) / intrinsics.fx
    yn = (pixel.v - intrinsics.cy) / intrinsics.fy

    # Ray in camera frame: (1, xn, yn) normalized
    ray = np.array([1.0, xn, yn], dtype=np.float64)
    norm = np.linalg.norm(ray)
    if norm > 0:
        ray /= norm

    return ray


def is_in_fov(pixel: PixelCoord, intrinsics: CameraIntrinsics) -> bool:
    """Check whether a pixel coordinate is within the camera FOV.

    Args:
        pixel: Pixel coordinate.
        intrinsics: Camera intrinsic parameters.

    Returns:
        True if the pixel is within the image bounds.
    """
    return (
        0 <= pixel.u < intrinsics.width
        and 0 <= pixel.v < intrinsics.height
    )


def is_point_visible(
    point_world: NDArray[np.float64],
    camera_state: CameraState,
    intrinsics: CameraIntrinsics,
) -> bool:
    """Check if a 3D world point is visible in the camera.

    A point is visible if it is in front of the camera AND within
    the image bounds.

    Args:
        point_world: 3D point in world frame, shape (3,).
        camera_state: Camera state.
        intrinsics: Camera intrinsic parameters.

    Returns:
        True if the point projects to a valid in-FOV pixel.
    """
    pixel = project_point_to_pixel(point_world, camera_state, intrinsics)
    if pixel is None:
        return False
    return is_in_fov(pixel, intrinsics)


def normalized_fov_position(
    pixel: PixelCoord,
    intrinsics: CameraIntrinsics,
) -> tuple[float, float]:
    """Compute the normalized position within the FOV.

    Returns values in [-1, 1] where (0, 0) is the image center
    and ±1 are the FOV edges.

    Args:
        pixel: Pixel coordinate.
        intrinsics: Camera intrinsic parameters.

    Returns:
        Tuple (nx, ny) where nx, ny ∈ [-1, 1].
    """
    nx = 2.0 * (pixel.u - intrinsics.cx) / intrinsics.width
    ny = 2.0 * (pixel.v - intrinsics.cy) / intrinsics.height
    return float(nx), float(ny)
