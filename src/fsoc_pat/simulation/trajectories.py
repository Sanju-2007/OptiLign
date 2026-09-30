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
    """Bounded, trackable random-walk trajectory (seeded).

    Generates a physically continuous, trackable random path via multi-scale
    harmonic synthesis with a 1/f Brownian power spectral density. The motion
    is infinitely differentiable (C^inf), ensuring bounded acceleration and
    analytical derivatives that an optical tracking gimbal and EKF can follow.

    Attributes:
        center: Center of the bounding region (m).
        extent: Half-size of the bounding box in each axis (m).
        speed: Approximate average speed (m/s).
        seed: Random seed for reproducibility.
        num_waypoints: Kept for backwards compatibility.
        _duration: Kept for backwards compatibility (trajectory extends smoothly for all t >= 0).
    """

    center: Position3D
    extent: Position3D  # half-sizes
    speed: float
    seed: int = 42
    num_waypoints: int = 10
    _duration: float = 60.0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        char_scale = max(10.0, float(math.hypot(self.extent.y, self.extent.z)))
        w0 = max(0.01, self.speed / char_scale) if self.speed > 0 else 0.0

        # Multi-scale incommensurate frequency ratios per axis
        ratios = np.array([0.41, 0.73, 1.17, 1.83, 2.67])
        self._w_x = w0 * ratios * rng.uniform(0.92, 1.08, 5)
        self._w_y = w0 * ratios * rng.uniform(0.92, 1.08, 5)
        self._w_z = w0 * ratios * rng.uniform(0.92, 1.08, 5)

        # Spectral energy falls off as ~ 1/f for natural turbulent / Brownian drift
        weights = 1.0 / (ratios ** 1.1)
        weights /= float(np.sum(weights))

        # Scaled amplitudes: peak-to-peak excursion <= 2 * extent, so |p_i - center_i| <= extent
        self._ax = weights * (self.extent.x * 0.48)
        self._ay = weights * (self.extent.y * 0.48)
        self._az = weights * (self.extent.z * 0.48)

        self._phi_x = rng.uniform(0.0, 2.0 * math.pi, 5)
        self._phi_y = rng.uniform(0.0, 2.0 * math.pi, 5)
        self._phi_z = rng.uniform(0.0, 2.0 * math.pi, 5)

        # Exact offsets so p(0) == center
        self._c_x = float(np.sum(self._ax * np.sin(self._phi_x)))
        self._c_y = float(np.sum(self._ay * np.sin(self._phi_y)))
        self._c_z = float(np.sum(self._az * np.sin(self._phi_z)))

    def position(self, t: float) -> Position3D:
        t = max(0.0, float(t))
        x = self.center.x + float(np.sum(self._ax * np.sin(self._w_x * t + self._phi_x))) - self._c_x
        y = self.center.y + float(np.sum(self._ay * np.sin(self._w_y * t + self._phi_y))) - self._c_y
        z = self.center.z + float(np.sum(self._az * np.sin(self._w_z * t + self._phi_z))) - self._c_z
        return Position3D(x, y, z)

    def velocity(self, t: float) -> Velocity3D:
        t = max(0.0, float(t))
        vx = float(np.sum(self._ax * self._w_x * np.cos(self._w_x * t + self._phi_x)))
        vy = float(np.sum(self._ay * self._w_y * np.cos(self._w_y * t + self._phi_y)))
        vz = float(np.sum(self._az * self._w_z * np.cos(self._w_z * t + self._phi_z)))
        return Velocity3D(vx, vy, vz)

    def acceleration(self, t: float) -> Acceleration3D:
        t = max(0.0, float(t))
        ax = float(-np.sum(self._ax * (self._w_x ** 2) * np.sin(self._w_x * t + self._phi_x)))
        ay = float(-np.sum(self._ay * (self._w_y ** 2) * np.sin(self._w_y * t + self._phi_y)))
        az = float(-np.sum(self._az * (self._w_z ** 2) * np.sin(self._w_z * t + self._phi_z)))
        return Acceleration3D(ax, ay, az)


@dataclass(frozen=True)
class FigureOfEightTrajectory:
    """Figure-of-8 (Lemniscate) motion in a plane.

    p(t) = center + [Rx * sin(ω*t + φ), 0.5 * Ry * sin(2*(ω*t + φ)), 0]

    This produces a smooth horizontal figure-8 pattern with continuous
    curvature reversal and analytical position, velocity, and acceleration.

    Attributes:
        center: Center of the figure-of-8 (m).
        radius_x: Horizontal amplitude (m).
        radius_y: Vertical amplitude (m).
        angular_rate: Angular velocity ω (rad/s).
        phase: Initial phase offset φ (rad).
    """

    center: Position3D
    radius_x: float
    radius_y: float
    angular_rate: float
    phase: float = 0.0

    def position(self, t: float) -> Position3D:
        theta = self.angular_rate * t + self.phase
        return Position3D(
            self.center.x + self.radius_x * math.sin(theta),
            self.center.y + 0.5 * self.radius_y * math.sin(2.0 * theta),
            self.center.z,
        )

    def velocity(self, t: float) -> Velocity3D:
        theta = self.angular_rate * t + self.phase
        w = self.angular_rate
        return Velocity3D(
            self.radius_x * w * math.cos(theta),
            self.radius_y * w * math.cos(2.0 * theta),
            0.0,
        )

    def acceleration(self, t: float) -> Acceleration3D:
        theta = self.angular_rate * t + self.phase
        w2 = self.angular_rate * self.angular_rate
        return Acceleration3D(
            -self.radius_x * w2 * math.sin(theta),
            -2.0 * self.radius_y * w2 * math.sin(2.0 * theta),
            0.0,
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
    "figure_of_8": FigureOfEightTrajectory,
    "figure_of_eight": FigureOfEightTrajectory,
    "lemniscate": FigureOfEightTrajectory,
    "maneuver": ManeuverTrajectory,
    "random": RandomTrajectory,
    "random_path": RandomTrajectory,
    "random_walk": RandomTrajectory,
    "random_flight": RandomTrajectory,
}

