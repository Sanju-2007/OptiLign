"""Motion models for Kalman filter state transition.

Provides F (state transition) and Q (process noise) matrices for:
  1. Constant Position (CP)  — state = [x, y]
  2. Constant Velocity (CV)  — state = [x, vx, y, vy]
  3. Constant Acceleration (CA) — state = [x, vx, ax, y, vy, ay]

All models work in 2D pixel space (u, v). The state vector is
interleaved: [u-components, v-components].

Units:
    Position: pixels
    Velocity: pixels/s
    Acceleration: pixels/s²
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray


class MotionModelType(Enum):
    """Available motion model types."""

    CONSTANT_POSITION = auto()
    CONSTANT_VELOCITY = auto()
    CONSTANT_ACCELERATION = auto()


@dataclass(frozen=True)
class MotionModel:
    """A motion model providing F and Q matrices.

    Attributes:
        F: State transition matrix (n, n).
        Q: Process noise covariance (n, n).
        H: Measurement matrix (m, n).
        state_dim: State dimension n.
        meas_dim: Measurement dimension m.
        name: Human-readable name.
    """

    F: NDArray[np.float64]
    Q: NDArray[np.float64]
    H: NDArray[np.float64]
    state_dim: int
    meas_dim: int
    name: str


def constant_position_model(
    dt: float,
    q_pos: float = 1.0,
) -> MotionModel:
    """Constant position model (no motion assumed).

    State: [u, v]
    Measurement: [u, v]

    Args:
        dt: Timestep (s).
        q_pos: Position process noise variance (px²).

    Returns:
        MotionModel with F, Q, H.
    """
    F = np.eye(2, dtype=np.float64)
    Q = q_pos * np.eye(2, dtype=np.float64)
    H = np.eye(2, dtype=np.float64)

    return MotionModel(F=F, Q=Q, H=H, state_dim=2, meas_dim=2,
                       name="Constant Position")


def constant_velocity_model(
    dt: float,
    q_vel: float = 10.0,
) -> MotionModel:
    """Constant velocity model.

    State: [u, vu, v, vv]
    Measurement: [u, v]

    Uses discrete white noise acceleration model for Q.

    Args:
        dt: Timestep (s).
        q_vel: Velocity process noise power spectral density (px²/s³).

    Returns:
        MotionModel with F, Q, H.
    """
    # State transition: constant velocity in each axis
    F = np.array([
        [1, dt, 0, 0],
        [0, 1,  0, 0],
        [0, 0,  1, dt],
        [0, 0,  0, 1],
    ], dtype=np.float64)

    # Process noise (piecewise white noise acceleration)
    dt2 = dt * dt
    dt3 = dt2 * dt
    q_block = q_vel * np.array([
        [dt3 / 3, dt2 / 2],
        [dt2 / 2, dt],
    ], dtype=np.float64)

    Q = np.zeros((4, 4), dtype=np.float64)
    Q[0:2, 0:2] = q_block
    Q[2:4, 2:4] = q_block

    # Measurement: observe position only
    H = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
    ], dtype=np.float64)

    return MotionModel(F=F, Q=Q, H=H, state_dim=4, meas_dim=2,
                       name="Constant Velocity")


def constant_acceleration_model(
    dt: float,
    q_acc: float = 50.0,
) -> MotionModel:
    """Constant acceleration model.

    State: [u, vu, au, v, vv, av]
    Measurement: [u, v]

    Args:
        dt: Timestep (s).
        q_acc: Acceleration jerk noise variance (px²/s⁵).

    Returns:
        MotionModel with F, Q, H.
    """
    dt2 = dt * dt

    # Per-axis state transition
    f_block = np.array([
        [1, dt, dt2 / 2],
        [0, 1,  dt],
        [0, 0,  1],
    ], dtype=np.float64)

    F = np.zeros((6, 6), dtype=np.float64)
    F[0:3, 0:3] = f_block
    F[3:6, 3:6] = f_block

    # Process noise (white jerk model)
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    dt5 = dt4 * dt
    q_block = q_acc * np.array([
        [dt5 / 20, dt4 / 8, dt3 / 6],
        [dt4 / 8,  dt3 / 3, dt2 / 2],
        [dt3 / 6,  dt2 / 2, dt],
    ], dtype=np.float64)

    Q = np.zeros((6, 6), dtype=np.float64)
    Q[0:3, 0:3] = q_block
    Q[3:6, 3:6] = q_block

    # Measurement: observe position only
    H = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
    ], dtype=np.float64)

    return MotionModel(F=F, Q=Q, H=H, state_dim=6, meas_dim=2,
                       name="Constant Acceleration")


def get_motion_model(
    model_type: MotionModelType,
    dt: float,
    **kwargs,
) -> MotionModel:
    """Factory to get a motion model by type.

    Args:
        model_type: Which model to use.
        dt: Timestep (s).
        **kwargs: Model-specific parameters.

    Returns:
        MotionModel instance.
    """
    if model_type == MotionModelType.CONSTANT_POSITION:
        return constant_position_model(dt, **kwargs)
    elif model_type == MotionModelType.CONSTANT_VELOCITY:
        return constant_velocity_model(dt, **kwargs)
    elif model_type == MotionModelType.CONSTANT_ACCELERATION:
        return constant_acceleration_model(dt, **kwargs)
    else:
        raise ValueError(f"Unknown motion model type: {model_type}")
