"""Search and reacquisition scan patterns for optical beacon acquisition."""

from __future__ import annotations

import math
from typing import Protocol


class SearchPatternProtocol(Protocol):
    """Protocol for gimbal search pattern generators."""

    def reset(self, center_pan: float = 0.0, center_tilt: float = 0.0) -> None:
        """Reset the search pattern to start from the specified center angles."""
        ...

    def step(self, dt: float) -> tuple[float, float]:
        """Compute the next (pan_rate, tilt_rate) command in rad/s."""
        ...

    def current_offset(self) -> tuple[float, float]:
        """Return the current (delta_pan, delta_tilt) angular offset from center (rad)."""
        ...

    @property
    def is_finished(self) -> bool:
        """Whether the search pattern has completed its scan area."""
        ...


class SpiralSearchPattern:
    """Archimedean spiral scan pattern.

    Expands smoothly outward from a center point with constant angular velocity.
    Optimal for optical beacon reacquisition where the uncertainty area is
    circularly symmetric around the last-known target position.

    Equations:
        theta(t) = omega * t
        r(t) = step_per_turn * theta(t) / (2 * pi)
        pan(t) = center_pan + r(t) * cos(theta(t))
        tilt(t) = center_tilt + r(t) * sin(theta(t))
    """

    def __init__(
        self,
        step_rad: float = 0.08,
        angular_speed: float = 1.0,
        max_radius_rad: float = 0.45,
    ) -> None:
        self.step_rad = step_rad
        self.angular_speed = angular_speed
        self.max_radius_rad = max_radius_rad

        self.center_pan = 0.0
        self.center_tilt = 0.0
        self.elapsed_time = 0.0
        self._cur_d_pan = 0.0
        self._cur_d_tilt = 0.0
        self._finished = False

    def reset(self, center_pan: float = 0.0, center_tilt: float = 0.0) -> None:
        self.center_pan = center_pan
        self.center_tilt = center_tilt
        self.elapsed_time = 0.0
        self._cur_d_pan = 0.0
        self._cur_d_tilt = 0.0
        self._finished = False

    def _eval_offset(self, t: float) -> tuple[float, float]:
        """Evaluate (delta_pan, delta_tilt) at time t."""
        theta = self.angular_speed * t
        # r = (step_rad / 2pi) * theta
        r = (self.step_rad / (2.0 * math.pi)) * theta
        if r > self.max_radius_rad:
            r = self.max_radius_rad
        return r * math.cos(theta), r * math.sin(theta)

    def step(self, dt: float) -> tuple[float, float]:
        """Advance time and return commanded (pan_rate, tilt_rate)."""
        if self._finished or dt <= 0.0:
            return 0.0, 0.0

        p1, t1 = self._eval_offset(self.elapsed_time)
        p2, t2 = self._eval_offset(self.elapsed_time + dt)

        self.elapsed_time += dt
        self._cur_d_pan = p2
        self._cur_d_tilt = t2

        r_current = math.hypot(p2, t2)
        if r_current >= self.max_radius_rad:
            self._finished = True

        pan_rate = (p2 - p1) / dt
        tilt_rate = (t2 - t1) / dt
        return pan_rate, tilt_rate

    def current_offset(self) -> tuple[float, float]:
        return self._cur_d_pan, self._cur_d_tilt

    @property
    def is_finished(self) -> bool:
        return self._finished


class RasterSearchPattern:
    """Meander / Boustrophedon raster scan pattern across a rectangular region."""

    def __init__(
        self,
        pan_span_rad: float = 0.6,
        tilt_span_rad: float = 0.4,
        step_rad: float = 0.06,
        scan_rate: float = 0.6,
    ) -> None:
        self.pan_span_rad = pan_span_rad
        self.tilt_span_rad = tilt_span_rad
        self.step_rad = step_rad
        self.scan_rate = scan_rate

        self.center_pan = 0.0
        self.center_tilt = 0.0
        self._cur_d_pan = -pan_span_rad / 2.0
        self._cur_d_tilt = -tilt_span_rad / 2.0
        self._dir = 1.0  # +1 moving right, -1 moving left
        self._finished = False

    def reset(self, center_pan: float = 0.0, center_tilt: float = 0.0) -> None:
        self.center_pan = center_pan
        self.center_tilt = center_tilt
        self._cur_d_pan = -self.pan_span_rad / 2.0
        self._cur_d_tilt = -self.tilt_span_rad / 2.0
        self._dir = 1.0
        self._finished = False

    def step(self, dt: float) -> tuple[float, float]:
        if self._finished or dt <= 0.0:
            return 0.0, 0.0

        pan_rate = self._dir * self.scan_rate
        tilt_rate = 0.0

        self._cur_d_pan += pan_rate * dt
        half_pan = self.pan_span_rad / 2.0
        half_tilt = self.tilt_span_rad / 2.0

        if self._dir > 0 and self._cur_d_pan >= half_pan:
            self._cur_d_pan = half_pan
            self._dir = -1.0
            self._cur_d_tilt += self.step_rad
            tilt_rate = self.step_rad / dt
        elif self._dir < 0 and self._cur_d_pan <= -half_pan:
            self._cur_d_pan = -half_pan
            self._dir = 1.0
            self._cur_d_tilt += self.step_rad
            tilt_rate = self.step_rad / dt

        if self._cur_d_tilt > half_tilt:
            self._finished = True
            return 0.0, 0.0

        return pan_rate, tilt_rate

    def current_offset(self) -> tuple[float, float]:
        return self._cur_d_pan, self._cur_d_tilt

    @property
    def is_finished(self) -> bool:
        return self._finished


class OmniSearchPattern:
    """Omnidirectional / All-Degrees High-Speed Search Pattern.

    Sweeps through all degrees of freedom in both pan (azimuth) and tilt (elevation)
    to rapidly search and acquire targets that have exited the camera's field of view.

    Features:
    - Wide azimuth coverage up to +/- 140 deg (+/- 2.44 rad), which with 60 deg camera FOV
      covers 340 deg without hitting physical gimbal mechanical stops (+/- 180 deg).
    - Elevation coverage covering -15 deg to +30 deg with 46.8 deg vertical FOV.
    - Minimal-time reacquisition: sweeps towards the target sector immediately (< 0.8s)
      at optimal velocity (0.85 rad/s) safely within the gimbal rate limit (1.0 rad/s).
    """

    def __init__(
        self,
        pan_amplitude_rad: float = math.pi,
        tilt_min_rad: float = math.radians(-20.0),
        tilt_max_rad: float = math.radians(45.0),
        pan_freq: float = 0.25,
        tilt_freq: float = 0.25,
        scan_rate_rad_s: float = 0.85,
        pan_span_rad: float | None = None,
    ) -> None:
        self.pan_amplitude = pan_amplitude_rad
        self.pan_amplitude_rad = pan_amplitude_rad
        self.pan_span = pan_span_rad if pan_span_rad is not None else min(math.radians(280.0), 2.0 * pan_amplitude_rad)
        self.tilt_min = tilt_min_rad
        self.tilt_min_rad = tilt_min_rad
        self.tilt_max = tilt_max_rad
        self.tilt_max_rad = tilt_max_rad
        self.pan_freq = pan_freq
        self.tilt_freq = tilt_freq
        self.scan_rate = scan_rate_rad_s

        self.tilt_center = (tilt_min_rad + tilt_max_rad) / 2.0
        self.tilt_amp = (tilt_max_rad - tilt_min_rad) / 2.0

        self.current_pan = 0.0
        self.current_tilt = 0.0
        self.direction = -1.0
        self.elapsed_time = 0.0
        self._cur_d_pan = 0.0
        self._cur_d_tilt = 0.0
        self._finished = False

    def reset(self, center_pan: float = 0.0, center_tilt: float = 0.0) -> None:
        self.current_pan = center_pan
        self.current_tilt = center_tilt
        self.elapsed_time = 0.0
        # If slewed to positive azimuth, sweep left towards center (0 rad); if negative, sweep right towards center
        self.direction = -1.0 if center_pan >= 0.0 else 1.0
        self._cur_d_pan = 0.0
        self._cur_d_tilt = 0.0
        self._finished = False

    def step(self, dt: float) -> tuple[float, float]:
        """Advance time and return commanded (pan_rate, tilt_rate) in rad/s."""
        if dt <= 0.0:
            return 0.0, 0.0

        self.elapsed_time += dt

        # Pan rate at constant rapid scan velocity
        pan_rate = self.direction * self.scan_rate
        self.current_pan += pan_rate * dt

        max_pan = self.pan_span / 2.0
        if self.current_pan >= max_pan:
            self.current_pan = max_pan
            self.direction = -1.0
            pan_rate = self.direction * self.scan_rate
        elif self.current_pan <= -max_pan:
            self.current_pan = -max_pan
            self.direction = 1.0
            pan_rate = self.direction * self.scan_rate

        # Smooth elevation tracking across 3D cone
        target_tilt = self.tilt_center + self.tilt_amp * math.sin(2.0 * math.pi * self.tilt_freq * self.elapsed_time)
        tilt_rate = (target_tilt - self.current_tilt) / max(dt, 0.001)
        tilt_rate = max(-0.8, min(0.8, tilt_rate))
        self.current_tilt += tilt_rate * dt

        self._cur_d_pan = self.current_pan
        self._cur_d_tilt = self.current_tilt

        return pan_rate, tilt_rate

    def current_offset(self) -> tuple[float, float]:
        return self._cur_d_pan, self._cur_d_tilt

    @property
    def is_finished(self) -> bool:
        return self._finished


