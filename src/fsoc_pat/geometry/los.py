"""Line-of-sight (LOS) geometry.

Computes the line-of-sight vector from camera to target and the
camera boresight direction. These are the fundamental geometric
quantities for angular error computation.

All vectors are unit vectors in the world frame.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from fsoc_pat.core.types import AngularState, Orientation, Position3D
from fsoc_pat.geometry.transforms import boresight_in_world


def compute_los_vector(
    camera_position: Position3D,
    target_position: Position3D,
) -> NDArray[np.float64]:
    """Compute the line-of-sight unit vector from camera to target.

    Args:
        camera_position: Camera position in world frame (m).
        target_position: Target position in world frame (m).

    Returns:
        Unit LOS vector in world frame, shape (3,).
        Points from camera toward target.

    Raises:
        ValueError: If camera and target are co-located.
    """
    delta = target_position.as_array() - camera_position.as_array()
    distance = float(np.linalg.norm(delta))

    if distance < 1e-12:
        raise ValueError(
            "Camera and target are co-located — LOS is undefined"
        )

    return delta / distance


def compute_boresight(
    orientation: Orientation,
    angular_state: AngularState,
) -> NDArray[np.float64]:
    """Compute camera boresight direction in world frame.

    The boresight is the +X direction in camera frame, transformed
    to world coordinates.

    Args:
        orientation: Platform orientation.
        angular_state: Gimbal pan/tilt state.

    Returns:
        Unit boresight vector in world frame, shape (3,).
    """
    return boresight_in_world(orientation, angular_state)


def los_range(
    camera_position: Position3D,
    target_position: Position3D,
) -> float:
    """Compute the range (distance) from camera to target.

    Args:
        camera_position: Camera position (m).
        target_position: Target position (m).

    Returns:
        Range in meters.
    """
    return camera_position.distance_to(target_position)
