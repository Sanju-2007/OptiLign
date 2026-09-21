"""Target trajectory generators.

Each trajectory implements TrajectoryProtocol and provides analytical
position(t), velocity(t), and acceleration(t). Numerical differentiation
of noisy positions is never used when analytical derivatives exist.

All trajectories are independent of rendering and detection.
Trajectory generation is a pure mathematical function of time.

Units:
    position: meters
    velocity: meters/second
    acceleration: meters/second²
    time: seconds
    angles: radians
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from fsoc_pat.core.types import Acceleration3D, Position3D, Velocity3D


@dataclass(frozen=True)
class StationaryTrajectory:
    """Fixed-position target.

    Attributes:
        pos: Static position (m).
    """

    pos: Position3D

    def position(self, t: float) -> Position3D:
        return self.pos

    def velocity(self, t: float) -> Velocity3D:
        return Velocity3D(0.0, 0.0, 0.0)

    def acceleration(self, t: float) -> Acceleration3D:
        return Acceleration3D(0.0, 0.0, 0.0)


@dataclass(frozen=True)
class LinearTrajectory:
    """Constant-velocity linear motion.

    p(t) = p0 + v * t

    Attributes:
        p0: Initial position (m).
        vel: Constant velocity (m/s).
    """

    p0: Position3D
    vel: Velocity3D

    def position(self, t: float) -> Position3D:
        return Position3D(
            self.p0.x + self.vel.x * t,
            self.p0.y + self.vel.y * t,
            self.p0.z + self.vel.z * t,
        )

    def velocity(self, t: float) -> Velocity3D:
        return self.vel

    def acceleration(self, t: float) -> Acceleration3D:
        return Acceleration3D(0.0, 0.0, 0.0)


@dataclass(frozen=True)
class ConstantAccelerationTrajectory:
    """Constant-acceleration motion.

    p(t) = p0 + v0 * t + 0.5 * a * t²
    v(t) = v0 + a * t

    Attributes:
        p0: Initial position (m).
        v0: Initial velocity (m/s).
        accel: Constant acceleration (m/s²).
    """

    p0: Position3D
    v0: Velocity3D
    accel: Acceleration3D

    def position(self, t: float) -> Position3D:
        return Position3D(
            self.p0.x + self.v0.x * t + 0.5 * self.accel.x * t * t,
            self.p0.y + self.v0.y * t + 0.5 * self.accel.y * t * t,
            self.p0.z + self.v0.z * t + 0.5 * self.accel.z * t * t,
        )

    def velocity(self, t: float) -> Velocity3D:
        return Velocity3D(
            self.v0.x + self.accel.x * t,
            self.v0.y + self.accel.y * t,
            self.v0.z + self.accel.z * t,
        )

    def acceleration(self, t: float) -> Acceleration3D:
        return self.accel


@dataclass(frozen=True)
class CircularTrajectory:
    """Circular motion in a plane.

    The target orbits a center point at a given radius and angular rate.
    Motion is in the XY-plane at height center.z unless a different
    normal is configured.

    p(t) = center + R * [cos(ω*t + φ), sin(ω*t + φ), 0]

    Attributes:
        center: Center of the circle (m).
        radius: Orbit radius (m).
        angular_rate: Angular velocity (rad/s). Positive = CCW.
        phase: Initial phase angle (rad).
    """

    center: Position3D
    radius: float
    angular_rate: float
    phase: float = 0.0

    def position(self, t: float) -> Position3D:
        theta = self.angular_rate * t + self.phase
        return Position3D(
            self.center.x + self.radius * math.cos(theta),
            self.center.y + self.radius * math.sin(theta),
            self.center.z,
        )

    def velocity(self, t: float) -> Velocity3D:
        theta = self.angular_rate * t + self.phase
        return Velocity3D(
            -self.radius * self.angular_rate * math.sin(theta),
            self.radius * self.angular_rate * math.cos(theta),
            0.0,
        )

    def acceleration(self, t: float) -> Acceleration3D:
        theta = self.angular_rate * t + self.phase
        w2r = self.angular_rate * self.angular_rate * self.radius
        return Acceleration3D(
            -w2r * math.cos(theta),
            -w2r * math.sin(theta),
            0.0,
        )


@dataclass(frozen=True)
class SinusoidalTrajectory:
    """Sinusoidal oscillation in each axis independently.

    p_i(t) = center_i + amplitude_i * sin(frequency_i * t + phase_i)

    Attributes:
        center: Mean position (m).
        amplitude: Amplitude in each axis (m).
        frequency: Angular frequency in each axis (rad/s).
        phase: Phase offset in each axis (rad).
    """

    center: Position3D
    amplitude: Position3D  # reusing as (ax, ay, az)
    frequency: Position3D  # reusing as (fx, fy, fz) in rad/s
    phase: Position3D = Position3D(0.0, 0.0, 0.0)

    def position(self, t: float) -> Position3D:
        return Position3D(
            self.center.x + self.amplitude.x * math.sin(self.frequency.x * t + self.phase.x),
            self.center.y + self.amplitude.y * math.sin(self.frequency.y * t + self.phase.y),
            self.center.z + self.amplitude.z * math.sin(self.frequency.z * t + self.phase.z),
        )

    def velocity(self, t: float) -> Velocity3D:
        return Velocity3D(
            self.amplitude.x * self.frequency.x
            * math.cos(self.frequency.x * t + self.phase.x),
            self.amplitude.y * self.frequency.y
            * math.cos(self.frequency.y * t + self.phase.y),
            self.amplitude.z * self.frequency.z
            * math.cos(self.frequency.z * t + self.phase.z),
        )

    def acceleration(self, t: float) -> Acceleration3D:
        return Acceleration3D(
            -self.amplitude.x * self.frequency.x**2
            * math.sin(self.frequency.x * t + self.phase.x),
            -self.amplitude.y * self.frequency.y**2
            * math.sin(self.frequency.y * t + self.phase.y),
            -self.amplitude.z * self.frequency.z**2
            * math.sin(self.frequency.z * t + self.phase.z),
        )


@dataclass(frozen=True)
class SpiralTrajectory:
    """Expanding or contracting spiral in the XY-plane.

    r(t) = r0 + expansion_rate * t
    p(t) = center + r(t) * [cos(ω*t), sin(ω*t), 0]

    Attributes:
        center: Spiral center (m).
        initial_radius: Starting radius (m).
        expansion_rate: Radial expansion rate (m/s). Can be negative.
        angular_rate: Angular velocity (rad/s).
    """

    center: Position3D
    initial_radius: float
    expansion_rate: float
    angular_rate: float

    def _radius(self, t: float) -> float:
        return self.initial_radius + self.expansion_rate * t

    def position(self, t: float) -> Position3D:
        r = self._radius(t)
        theta = self.angular_rate * t
        return Position3D(
            self.center.x + r * math.cos(theta),
            self.center.y + r * math.sin(theta),
            self.center.z,
        )

    def velocity(self, t: float) -> Velocity3D:
        r = self._radius(t)
        theta = self.angular_rate * t
        rdot = self.expansion_rate
        w = self.angular_rate
        return Velocity3D(
            rdot * math.cos(theta) - r * w * math.sin(theta),
            rdot * math.sin(theta) + r * w * math.cos(theta),
            0.0,
        )

    def acceleration(self, t: float) -> Acceleration3D:
        r = self._radius(t)
        theta = self.angular_rate * t
        rdot = self.expansion_rate
        w = self.angular_rate
        return Acceleration3D(
            -2 * rdot * w * math.sin(theta) - r * w * w * math.cos(theta),
            2 * rdot * w * math.cos(theta) - r * w * w * math.sin(theta),
            0.0,
        )


@dataclass
class ManeuverTrajectory:
    """Piecewise trajectory with sudden direction changes.

    Segments are defined by time breakpoints and velocity vectors.
    At each breakpoint, the velocity instantly changes to the new value.

    Attributes:
        p0: Initial position (m).
        segments: List of (time, velocity) tuples. Times must be sorted.
    """

    p0: Position3D
    segments: list[tuple[float, Velocity3D]]

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("ManeuverTrajectory requires at least one segment")
        times = [s[0] for s in self.segments]
        if times != sorted(times):
            raise ValueError("Segment times must be sorted")

    def _active_segment(self, t: float) -> int:
        """Find the index of the active segment at time t."""
        for i in range(len(self.segments) - 1, -1, -1):
            if t >= self.segments[i][0]:
                return i
        return 0

    def position(self, t: float) -> Position3D:
        # Integrate piecewise: sum displacement from each segment
        pos = self.p0.as_array().copy()

        for i, (seg_time, seg_vel) in enumerate(self.segments):
            if i + 1 < len(self.segments):
                next_time = self.segments[i + 1][0]
            else:
                next_time = t + 1  # beyond current time

            if t <= seg_time:
                break

            dt = min(t, next_time) - seg_time
            pos += seg_vel.as_array() * dt

        return Position3D.from_array(pos)

    def velocity(self, t: float) -> Velocity3D:
        idx = self._active_segment(t)
        return self.segments[idx][1]

    def acceleration(self, t: float) -> Acceleration3D:
        # Constant velocity within each segment
        return Acceleration3D(0.0, 0.0, 0.0)


@dataclass
class RandomTrajectory:
    """Bounded random-walk trajectory (seeded).

    Generates a smooth random trajectory by pre-computing waypoints
    and interpolating with cubic splines. The trajectory stays within
    a configurable bounding box.

    Attributes:
        center: Center of the bounding region (m).
        extent: Half-size of the bounding box in each axis (m).
        speed: Approximate average speed (m/s).
        seed: Random seed for reproducibility.
        num_waypoints: Number of waypoints to generate per 10 seconds.
    """

    center: Position3D
    extent: Position3D  # half-sizes
    speed: float
    seed: int = 42
    num_waypoints: int = 10
    _duration: float = 60.0  # default max duration for pre-computation

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._generate_waypoints()

    def _generate_waypoints(self) -> None:
        """Pre-compute random waypoints and time-parametrize them."""
        n = max(3, int(self.num_waypoints * self._duration / 10.0))
        times = np.linspace(0.0, self._duration, n)

        cx, cy, cz = self.center.x, self.center.y, self.center.z
        ex, ey, ez = self.extent.x, self.extent.y, self.extent.z

        # Random positions within bounding box
        xs = cx + self._rng.uniform(-ex, ex, n)
        ys = cy + self._rng.uniform(-ey, ey, n)
        zs = cz + self._rng.uniform(-ez, ez, n)

        # Start at center
        xs[0], ys[0], zs[0] = cx, cy, cz

        self._times = times
        self._xs = xs
        self._ys = ys
        self._zs = zs

    def position(self, t: float) -> Position3D:
        t = max(0.0, min(t, self._duration))
        x = float(np.interp(t, self._times, self._xs))
        y = float(np.interp(t, self._times, self._ys))
        z = float(np.interp(t, self._times, self._zs))
        return Position3D(x, y, z)

    def velocity(self, t: float) -> Velocity3D:
        # Numerical derivative with small dt
        dt = 1e-4
        p1 = self.position(t)
        p2 = self.position(t + dt)
        return Velocity3D(
            (p2.x - p1.x) / dt,
            (p2.y - p1.y) / dt,
            (p2.z - p1.z) / dt,
        )

    def acceleration(self, t: float) -> Acceleration3D:
        dt = 1e-4
        v1 = self.velocity(t)
        v2 = self.velocity(t + dt)
        return Acceleration3D(
            (v2.x - v1.x) / dt,
            (v2.y - v1.y) / dt,
            (v2.z - v1.z) / dt,
        )


# ---------------------------------------------------------------------------
# Trajectory factory
# ---------------------------------------------------------------------------

TRAJECTORY_REGISTRY: dict[str, type] = {
    "stationary": StationaryTrajectory,
    "linear": LinearTrajectory,
    "constant_acceleration": ConstantAccelerationTrajectory,
    "circular": CircularTrajectory,
    "sinusoidal": SinusoidalTrajectory,
    "spiral": SpiralTrajectory,
    "maneuver": ManeuverTrajectory,
    "random": RandomTrajectory,
}
