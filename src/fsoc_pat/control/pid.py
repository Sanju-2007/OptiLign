"""PID Controller for gimbal pointing.

Full-featured PID with:
  - Derivative filter (low-pass to reduce noise amplification)
  - Anti-windup (clamping + back-calculation)
  - Output saturation
  - Independent pan/tilt channels

Implements ControllerProtocol.

This controller operates on AngularError (rad) and produces
ControlCommand (rad/s) for the gimbal. It NEVER accesses
ground truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from fsoc_pat.core.types import AngularError, ControlCommand


@dataclass
class PIDConfig:
    """PID controller configuration.

    Attributes:
        kp: Proportional gain (rad/s per rad).
        ki: Integral gain (rad/s per rad·s).
        kd: Derivative gain (rad/s per rad/s).
        d_filter_tau: Derivative low-pass filter time constant (s).
            0 = no filter.
        output_limit: Maximum output rate (rad/s). Applied symmetrically.
        integral_limit: Maximum integral accumulator (rad·s).
        anti_windup: Whether to apply anti-windup clamping.
    """

    kp: float = 5.0
    ki: float = 0.5
    kd: float = 0.3
    d_filter_tau: float = 0.02
    output_limit: float = 2.0  # rad/s
    integral_limit: float = 1.0  # rad·s
    anti_windup: bool = True


class PIDController:
    """Dual-axis PID controller for gimbal pointing.

    Maintains separate PID states for pan (horizontal) and
    tilt (vertical) channels.

    Usage:
        pid = PIDController(config)
        cmd = pid.compute(angular_error, dt)
        gimbal.command(cmd.pan_rate, cmd.tilt_rate)
    """

    def __init__(self, config: PIDConfig | None = None) -> None:
        self._config = config or PIDConfig()

        # Pan channel state
        self._integral_pan: float = 0.0
        self._prev_error_pan: float = 0.0
        self._filtered_d_pan: float = 0.0

        # Tilt channel state
        self._integral_tilt: float = 0.0
        self._prev_error_tilt: float = 0.0
        self._filtered_d_tilt: float = 0.0

        self._first_call = True

    def compute(
        self,
        error: AngularError,
        dt: float,
        **kwargs,
    ) -> ControlCommand:
        """Compute gimbal rate command from angular error.

        Args:
            error: Angular pointing error (rad).
            dt: Control timestep (s).
            **kwargs: Ignored (for protocol compatibility).

        Returns:
            ControlCommand with pan/tilt rates (rad/s).
        """
        cfg = self._config
        timestamp = kwargs.get("timestamp", 0.0)

        pan_rate = self._compute_channel(
            error.horizontal, dt,
            self._integral_pan, self._prev_error_pan, self._filtered_d_pan,
            is_pan=True,
        )

        tilt_rate = self._compute_channel(
            error.vertical, dt,
            self._integral_tilt, self._prev_error_tilt, self._filtered_d_tilt,
            is_pan=False,
        )

        self._first_call = False

        return ControlCommand(
            pan_rate=pan_rate,
            tilt_rate=tilt_rate,
            timestamp=timestamp,
        )

    def _compute_channel(
        self,
        error: float,
        dt: float,
        integral: float,
        prev_error: float,
        filtered_d: float,
        is_pan: bool,
    ) -> float:
        """Compute PID output for a single axis.

        Returns:
            Rate command (rad/s).
        """
        cfg = self._config

        if dt <= 0:
            return 0.0

        # Proportional
        p_term = cfg.kp * error

        # Integral with anti-windup
        integral += error * dt
        if cfg.anti_windup:
            integral = np.clip(integral, -cfg.integral_limit, cfg.integral_limit)
        i_term = cfg.ki * integral

        # Derivative with filter
        if self._first_call:
            raw_d = 0.0
        else:
            raw_d = (error - prev_error) / dt

        if cfg.d_filter_tau > 0:
            alpha = dt / (cfg.d_filter_tau + dt)
            filtered_d = alpha * raw_d + (1 - alpha) * filtered_d
        else:
            filtered_d = raw_d

        d_term = cfg.kd * filtered_d

        # Total output
        output = p_term + i_term + d_term

        # Saturation
        output = float(np.clip(output, -cfg.output_limit, cfg.output_limit))

        # Store state
        if is_pan:
            self._integral_pan = integral
            self._prev_error_pan = error
            self._filtered_d_pan = filtered_d
        else:
            self._integral_tilt = integral
            self._prev_error_tilt = error
            self._filtered_d_tilt = filtered_d

        return output

    def reset(self) -> None:
        """Reset controller internal state."""
        self._integral_pan = 0.0
        self._prev_error_pan = 0.0
        self._filtered_d_pan = 0.0
        self._integral_tilt = 0.0
        self._prev_error_tilt = 0.0
        self._filtered_d_tilt = 0.0
        self._first_call = True

    @property
    def integral_state(self) -> tuple[float, float]:
        """Current integral accumulator (pan, tilt)."""
        return (self._integral_pan, self._integral_tilt)
