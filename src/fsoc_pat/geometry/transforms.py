"""Coordinate frame transformations.

Transforms points between the frame hierarchy:
    WORLD → PLATFORM → GIMBAL → CAMERA

Convention (from frames.py):
    X = Forward, Y = Right, Z = Down
    ZYX intrinsic Euler rotations

The camera orientation is composed from:
    1. Platform orientation (roll, pitch, yaw) relative to WORLD
    2. Gimbal pan/tilt relative to PLATFORM

Pan rotates about the platform Z-axis (yaw).
Tilt rotates about the resulting Y-axis (pitch).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fsoc_pat.core.state import CameraState
from fsoc_pat.core.types import AngularState, Orientation, Position3D
from fsoc_pat.geometry.rotations import euler_to_rotation_matrix, rotation_matrix_y, rotation_matrix_z


def world_to_camera_rotation(
    orientation: Orientation,
    angular_state: AngularState,
) -> NDArray[np.float64]:
    """Compute the full world-to-camera rotation matrix.

    Composes platform orientation with gimbal pan/tilt:
        R_world_to_camera = R_gimbal · R_platform

    where:
        R_platform = R_zyx(roll, pitch, yaw)^T  (world→platform)
        R_gimbal   = R_tilt · R_pan              (platform→camera)

    Pan is about the platform Z-axis (yaw-like).
    Tilt is about the resulting Y-axis (pitch-like).

    Args:
        orientation: Platform orientation (roll, pitch, yaw) in radians.
        angular_state: Gimbal pan/tilt state.

    Returns:
        3×3 rotation matrix from world frame to camera frame.
    """
    # Platform body-to-world: R = Rz(yaw) · Ry(pitch) · Rx(roll)
    # World-to-platform: R^T
    R_platform = euler_to_rotation_matrix(
        orientation.roll, orientation.pitch, orientation.yaw
    ).T

    # Gimbal rotation: pan about Z, then tilt about Y
    R_pan = rotation_matrix_z(angular_state.pan).T
    R_tilt = rotation_matrix_y(angular_state.tilt).T
    R_gimbal = R_tilt @ R_pan

    return R_gimbal @ R_platform


def world_to_camera(
    point_world: NDArray[np.float64],
    camera_position: Position3D,
    orientation: Orientation,
    angular_state: AngularState,
) -> NDArray[np.float64]:
    """Transform a point from world frame to camera frame.

    Args:
        point_world: 3D point in world frame (3,).
        camera_position: Camera position in world frame.
        orientation: Platform orientation.
        angular_state: Gimbal state.

    Returns:
        3D point in camera frame (3,).
    """
    R = world_to_camera_rotation(orientation, angular_state)
    # Translate to camera origin, then rotate
    translated = point_world - camera_position.as_array()
    return R @ translated


def camera_to_world(
    point_camera: NDArray[np.float64],
    camera_position: Position3D,
    orientation: Orientation,
    angular_state: AngularState,
) -> NDArray[np.float64]:
    """Transform a point from camera frame to world frame.

    Inverse of world_to_camera.

    Args:
        point_camera: 3D point in camera frame (3,).
        camera_position: Camera position in world frame.
        orientation: Platform orientation.
        angular_state: Gimbal state.

    Returns:
        3D point in world frame (3,).
    """
    R = world_to_camera_rotation(orientation, angular_state)
    # R is orthogonal, so R^-1 = R^T
    return R.T @ point_camera + camera_position.as_array()


def world_to_camera_from_state(
    point_world: NDArray[np.float64],
    camera_state: CameraState,
) -> NDArray[np.float64]:
    """Transform a point from world to camera frame using a CameraState.

    Convenience wrapper over world_to_camera.

    Args:
        point_world: 3D point in world frame (3,).
        camera_state: Complete camera state.

    Returns:
        3D point in camera frame (3,).
    """
    return world_to_camera(
        point_world,
        camera_state.position,
        camera_state.orientation,
        camera_state.angular_state,
    )


def camera_to_world_from_state(
    point_camera: NDArray[np.float64],
    camera_state: CameraState,
) -> NDArray[np.float64]:
    """Transform a point from camera to world frame using a CameraState.

    Args:
        point_camera: 3D point in camera frame (3,).
        camera_state: Complete camera state.

    Returns:
        3D point in world frame (3,).
    """
    return camera_to_world(
        point_camera,
        camera_state.position,
        camera_state.orientation,
        camera_state.angular_state,
    )


def boresight_in_world(
    orientation: Orientation,
    angular_state: AngularState,
) -> NDArray[np.float64]:
    """Get the camera boresight direction in world frame.

    The boresight is the +X direction in camera frame.

    Args:
        orientation: Platform orientation.
        angular_state: Gimbal state.

    Returns:
        Unit vector (3,) in world frame pointing along the camera boresight.
    """
    R = world_to_camera_rotation(orientation, angular_state)
    # Camera +X in camera frame is [1, 0, 0]
    # In world frame: R^T @ [1, 0, 0] = first column of R^T = first row of R
    # But actually R^T transforms camera→world, so:
    boresight_camera = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    boresight_world = R.T @ boresight_camera
    # Ensure unit vector
    norm = np.linalg.norm(boresight_world)
    if norm > 0:
        boresight_world /= norm
    return boresight_world
