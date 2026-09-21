"""Gimbal dynamics model.

Models a realistic pan-tilt actuator with:
- Rate limits (max angular velocity)
- Acceleration limits
- Position limits (joint stops)
- Actuator lag (first-order response)
- Deadband (minimum command threshold)
- Mechanical noise

The key engineering distinction: commanded state ≠ actual state.
The controller commands a desired rate; the gimbal responds with
its physical dynamics. This delay and filtering is essential for
realistic control evaluation.

Units:
    Angles: radians
    Angular rates: radians/second
    Angular acceleration: radians/second²
    Time: seconds
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from fsoc_pat.core.types import AngularState


@dataclass
class GimbalDynamics:
    """Physical parameters of the pan-tilt gimbal actuator.

    Attributes:
        pan_rate_limit: Maximum pan angular velocity (rad/s).
        tilt_rate_limit: Maximum tilt angular velocity (rad/s).
        angular_accel_limit: Maximum angular acceleration (rad/s²).
            Applied to both axes. Set to large value to disable.
        pan_min: Minimum pan angle (rad). Set to -π for unrestricted.
        pan_max: Maximum pan angle (rad). Set to +π for unrestricted.
        tilt_min: Minimum tilt angle (rad).
        tilt_max: Maximum tilt angle (rad).
        actuator_time_constant: First-order lag time constant (s).
            Smaller = faster response. 0 = instant (no lag).
        deadband: Minimum command magnitude to produce motion (rad/s).
            Commands below this threshold produce zero motion.
        noise_std: Gaussian noise standard deviation on position (rad).
            Models mechanical imprecision.
    """

    pan_rate_limit: float = 1.0       # rad/s
    tilt_rate_limit: float = 1.0      # rad/s
    angular_accel_limit: float = 10.0  # rad/s²
    pan_min: float = -math.pi
    pan_max: float = math.pi
    tilt_min: float = -math.pi / 2.0
    tilt_max: float = math.pi / 2.0
    actuator_time_constant: float = 0.05  # s (20 Hz bandwidth)
    deadband: float = 0.001  # rad/s
    noise_std: float = 0.0   # rad

    def __post_init__(self) -> None:
        if self.pan_rate_limit <= 0:
            raise ValueError(f"pan_rate_limit must be positive: {self.pan_rate_limit}")
        if self.tilt_rate_limit <= 0:
            raise ValueError(f"tilt_rate_limit must be positive: {self.tilt_rate_limit}")
        if self.angular_accel_limit <= 0:
            raise ValueError(
                f"angular_accel_limit must be positive: {self.angular_accel_limit}"
            )


@dataclass
class VirtualGimbal:
    """Virtual pan-tilt gimbal with realistic dynamics.

    Maintains internal state (actual position, actual rate) and
    applies dynamics when stepping with commanded rates.

    Attributes:
        dynamics: Physical gimbal parameters.
    """

    dynamics: GimbalDynamics = field(default_factory=GimbalDynamics)

    # Internal state
    _pan: float = field(default=0.0, init=False, repr=False)
    _tilt: float = field(default=0.0, init=False, repr=False)
    _pan_rate: float = field(default=0.0, init=False, repr=False)
    _tilt_rate: float = field(default=0.0, init=False, repr=False)
    _rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0),
        init=False, repr=False,
    )

    def set_seed(self, seed: int) -> None:
        """Set random seed for noise generation."""
        self._rng = np.random.default_rng(seed)

    def set_position(self, pan: float, tilt: float) -> None:
        """Set initial gimbal position (bypasses dynamics).

        Use only during initialization — not during simulation.

        Args:
            pan: Initial pan angle (rad).
            tilt: Initial tilt angle (rad).
        """
        self._pan = self._clamp_pan(pan)
        self._tilt = self._clamp_tilt(tilt)
        self._pan_rate = 0.0
        self._tilt_rate = 0.0

    @property
    def state(self) -> AngularState:
        """Current actual gimbal state."""
        return AngularState(
            pan=self._pan,
            tilt=self._tilt,
            pan_rate=self._pan_rate,
            tilt_rate=self._tilt_rate,
        )

    def step(
        self,
        commanded_pan_rate: float,
        commanded_tilt_rate: float,
        dt: float,
    ) -> AngularState:
        """Advance gimbal state by one timestep.

        Applies: deadband → rate limiting → acceleration limiting →
        first-order lag → position integration → position clamping →
        noise.

        Args:
            commanded_pan_rate: Desired pan rate (rad/s).
            commanded_tilt_rate: Desired tilt rate (rad/s).
            dt: Timestep (s).

        Returns:
            New actual AngularState.
        """
        # Apply deadband
        cmd_pan = self._apply_deadband(commanded_pan_rate)
        cmd_tilt = self._apply_deadband(commanded_tilt_rate)

        # Apply rate limits
        cmd_pan = self._clamp(cmd_pan, -self.dynamics.pan_rate_limit,
                              self.dynamics.pan_rate_limit)
        cmd_tilt = self._clamp(cmd_tilt, -self.dynamics.tilt_rate_limit,
                               self.dynamics.tilt_rate_limit)

        # Apply acceleration limits
        cmd_pan = self._apply_accel_limit(self._pan_rate, cmd_pan, dt)
        cmd_tilt = self._apply_accel_limit(self._tilt_rate, cmd_tilt, dt)

        # First-order lag filter: rate tracks command with time constant
        tau = self.dynamics.actuator_time_constant
        if tau > 0 and dt > 0:
            alpha = dt / (tau + dt)
            self._pan_rate += alpha * (cmd_pan - self._pan_rate)
            self._tilt_rate += alpha * (cmd_tilt - self._tilt_rate)
        else:
            self._pan_rate = cmd_pan
            self._tilt_rate = cmd_tilt

        # Integrate position
        self._pan += self._pan_rate * dt
        self._tilt += self._tilt_rate * dt

        # Apply position limits
        self._pan = self._clamp_pan(self._pan)
        self._tilt = self._clamp_tilt(self._tilt)

        # Zero rate at limits (prevents integrator buildup at stops)
        if self._pan <= self.dynamics.pan_min or self._pan >= self.dynamics.pan_max:
            self._pan_rate = 0.0
        if self._tilt <= self.dynamics.tilt_min or self._tilt >= self.dynamics.tilt_max:
            self._tilt_rate = 0.0

        # Apply mechanical noise
        if self.dynamics.noise_std > 0:
            self._pan += float(self._rng.normal(0, self.dynamics.noise_std))
            self._tilt += float(self._rng.normal(0, self.dynamics.noise_std))
            self._pan = self._clamp_pan(self._pan)
            self._tilt = self._clamp_tilt(self._tilt)

        return self.state

    def reset(self) -> None:
        """Reset gimbal to home position."""
        self._pan = 0.0
        self._tilt = 0.0
        self._pan_rate = 0.0
        self._tilt_rate = 0.0

    def _apply_deadband(self, cmd: float) -> float:
        """Zero out commands below the deadband threshold."""
        if abs(cmd) < self.dynamics.deadband:
            return 0.0
        return cmd

    def _apply_accel_limit(
        self, current_rate: float, desired_rate: float, dt: float,
    ) -> float:
        """Limit rate change to maximum acceleration."""
        if dt <= 0:
            return desired_rate
        delta = desired_rate - current_rate
        max_delta = self.dynamics.angular_accel_limit * dt
        if abs(delta) > max_delta:
            delta = math.copysign(max_delta, delta)
        return current_rate + delta

    def _clamp_pan(self, angle: float) -> float:
        return self._clamp(angle, self.dynamics.pan_min, self.dynamics.pan_max)

    def _clamp_tilt(self, angle: float) -> float:
        return self._clamp(angle, self.dynamics.tilt_min, self.dynamics.tilt_max)

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))
