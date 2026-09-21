"""Linear Quadratic Regulator (LQR) Controller for Coarse Gimbal Tracking.

Provides optimal state feedback control:
  u = -K * x
minimizing quadratic cost function J = integral(x^T Q x + u^T R u) dt.
Includes integral error augmentation for zero steady-state tracking error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fsoc_pat.core.types import AngularError, ControlCommand


@dataclass
class LQRConfig:
    """LQR tuning configuration.

    Attributes:
        q_pos: State penalty for angular position error.
        q_vel: State penalty for angular velocity error.
        q_int: State penalty for integrated error.
        r_ctrl: Control effort penalty.
        rate_limit_rad_s: Maximum actuator rate command.
    """

    q_pos: float = 80.0
    q_vel: float = 8.0
    q_int: float = 30.0
    r_ctrl: float = 1.0
    rate_limit_rad_s: float = 2.0


class LQRController:
    """Discrete-time augmented LQR tracking controller for gimbal pan/tilt axes."""

    def __init__(self, config: LQRConfig | None = None) -> None:
        self.config = config or LQRConfig()
        self.int_err_pan: float = 0.0
        self.int_err_tilt: float = 0.0
        self.prev_err_pan: float = 0.0
        self.prev_err_tilt: float = 0.0

        # Precompute steady-state LQR gain vector K = [k_pos, k_vel, k_int]
        # For second-order integrator with integral state
        self._compute_gains()

    def _compute_gains(self) -> None:
        """Solve continuous Algebraic Riccati Equation (CARE) approximation for gains."""
        # Simplified analytical LQR state-feedback gains for SISO axis with integrator
        q_p = self.config.q_pos
        q_v = self.config.q_vel
        q_i = self.config.q_int
        r = self.config.r_ctrl

        # Classical optimal pole-placement / LQR asymptotic gains
        self.k_pos = math.sqrt(q_p / r)
        self.k_vel = math.sqrt(2 * self.k_pos + q_v / r)
        self.k_int = math.sqrt(q_i / r)

    def reset(self) -> None:
        """Reset internal controller states."""
        self.int_err_pan = 0.0
        self.int_err_tilt = 0.0
        self.prev_err_pan = 0.0
        self.prev_err_tilt = 0.0

    def update(self, error: AngularError, dt: float) -> ControlCommand:
        """Compute optimal control command from angular tracking error."""
        err_pan = error.horizontal
        err_tilt = error.vertical

        # Finite difference for error velocity
        d_err_pan = (err_pan - self.prev_err_pan) / dt if dt > 0 else 0.0
        d_err_tilt = (err_tilt - self.prev_err_tilt) / dt if dt > 0 else 0.0
        self.prev_err_pan = err_pan
        self.prev_err_tilt = err_tilt

        # Integral error accumulation
        self.int_err_pan += err_pan * dt
        self.int_err_tilt += err_tilt * dt

        # State vector x = [err, d_err, int_err]
        u_pan = self.k_pos * err_pan + self.k_vel * d_err_pan + self.k_int * self.int_err_pan
        u_tilt = self.k_pos * err_tilt + self.k_vel * d_err_tilt + self.k_int * self.int_err_tilt

        # Apply actuator rate saturation limits
        lim = self.config.rate_limit_rad_s
        u_pan = max(-lim, min(lim, u_pan))
        u_tilt = max(-lim, min(lim, u_tilt))

        return ControlCommand(pan_rate=u_pan, tilt_rate=u_tilt)

    def compute(self, error: AngularError, dt: float, **kwargs) -> ControlCommand:
        """Alias for compute() in ControllerProtocol."""
        return self.update(error, dt)
