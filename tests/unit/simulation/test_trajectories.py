"""Tests for trajectory generators."""

import math

import numpy as np
import pytest

from fsoc_pat.core.types import Acceleration3D, Position3D, Velocity3D
from fsoc_pat.simulation.trajectories import (
    CircularTrajectory,
    ConstantAccelerationTrajectory,
    LinearTrajectory,
    SinusoidalTrajectory,
    StationaryTrajectory,
)


class TestStationaryTrajectory:
    def test_position_constant(self) -> None:
        traj = StationaryTrajectory(pos=Position3D(10.0, 20.0, 30.0))
        for t in [0, 1, 5, 100]:
            p = traj.position(t)
            assert p.x == 10.0
            assert p.y == 20.0
            assert p.z == 30.0

    def test_velocity_zero(self) -> None:
        traj = StationaryTrajectory(pos=Position3D(0, 0, 0))
        v = traj.velocity(5.0)
        assert v.magnitude == 0.0

    def test_acceleration_zero(self) -> None:
        traj = StationaryTrajectory(pos=Position3D(0, 0, 0))
        a = traj.acceleration(5.0)
        assert a.magnitude == 0.0


class TestLinearTrajectory:
    def test_position_at_t0(self) -> None:
        traj = LinearTrajectory(
            p0=Position3D(10.0, 0.0, 0.0),
            vel=Velocity3D(5.0, 0.0, 0.0),
        )
        p = traj.position(0.0)
        assert p.x == pytest.approx(10.0)

    def test_position_at_t1(self) -> None:
        traj = LinearTrajectory(
            p0=Position3D(0.0, 0.0, 0.0),
            vel=Velocity3D(3.0, 4.0, 0.0),
        )
        p = traj.position(2.0)
        assert p.x == pytest.approx(6.0)
        assert p.y == pytest.approx(8.0)

    def test_velocity_constant(self) -> None:
        vel = Velocity3D(1.0, 2.0, 3.0)
        traj = LinearTrajectory(p0=Position3D(0, 0, 0), vel=vel)
        for t in [0, 1, 10, 100]:
            v = traj.velocity(t)
            assert v.x == vel.x
            assert v.y == vel.y
            assert v.z == vel.z

    def test_acceleration_zero(self) -> None:
        traj = LinearTrajectory(
            p0=Position3D(0, 0, 0), vel=Velocity3D(5, 0, 0),
        )
        a = traj.acceleration(5.0)
        assert a.magnitude == 0.0


class TestConstantAccelerationTrajectory:
    def test_position_formula(self) -> None:
        """p(t) = p0 + v0*t + 0.5*a*t²"""
        traj = ConstantAccelerationTrajectory(
            p0=Position3D(0, 0, 0),
            v0=Velocity3D(10, 0, 0),
            accel=Acceleration3D(2, 0, 0),
        )
        t = 3.0
        p = traj.position(t)
        expected = 0 + 10 * 3 + 0.5 * 2 * 9  # = 39
        assert p.x == pytest.approx(expected)

    def test_velocity_formula(self) -> None:
        """v(t) = v0 + a*t"""
        traj = ConstantAccelerationTrajectory(
            p0=Position3D(0, 0, 0),
            v0=Velocity3D(5, 0, 0),
            accel=Acceleration3D(3, 0, 0),
        )
        v = traj.velocity(2.0)
        assert v.x == pytest.approx(11.0)


class TestCircularTrajectory:
    def test_stays_on_circle(self) -> None:
        """Distance from center should equal radius at all times."""
        center = Position3D(50.0, 50.0, 0.0)
        radius = 10.0
        traj = CircularTrajectory(
            center=center, radius=radius, angular_rate=1.0,
        )
        for t in np.linspace(0, 10, 50):
            p = traj.position(float(t))
            dist = center.distance_to(p)
            assert dist == pytest.approx(radius, abs=1e-10)

    def test_velocity_tangent(self) -> None:
        """Velocity should be perpendicular to radius vector."""
        center = Position3D(0.0, 0.0, 0.0)
        traj = CircularTrajectory(
            center=center, radius=5.0, angular_rate=2.0,
        )
        for t in [0.5, 1.0, 2.5]:
            p = traj.position(t)
            v = traj.velocity(t)
            # Radius vector
            r = np.array([p.x - center.x, p.y - center.y])
            vel = np.array([v.x, v.y])
            dot = np.dot(r, vel)
            assert dot == pytest.approx(0.0, abs=1e-8)

    def test_velocity_magnitude(self) -> None:
        """Speed = |ω| × r."""
        radius, omega = 10.0, 0.5
        traj = CircularTrajectory(
            center=Position3D(0, 0, 0), radius=radius, angular_rate=omega,
        )
        v = traj.velocity(1.0)
        assert v.magnitude == pytest.approx(abs(omega) * radius, abs=1e-10)

    def test_acceleration_centripetal(self) -> None:
        """Acceleration magnitude = ω² × r (centripetal)."""
        radius, omega = 8.0, 1.5
        traj = CircularTrajectory(
            center=Position3D(0, 0, 0), radius=radius, angular_rate=omega,
        )
        a = traj.acceleration(2.0)
        assert a.magnitude == pytest.approx(omega**2 * radius, abs=1e-10)

    def test_derivative_consistency(self) -> None:
        """Numerical derivative of position ≈ analytical velocity."""
        traj = CircularTrajectory(
            center=Position3D(0, 0, 0), radius=5.0, angular_rate=1.0,
        )
        t = 2.0
        dt = 1e-6
        p1 = traj.position(t)
        p2 = traj.position(t + dt)
        numerical_v = Velocity3D(
            (p2.x - p1.x) / dt,
            (p2.y - p1.y) / dt,
            (p2.z - p1.z) / dt,
        )
        analytical_v = traj.velocity(t)
        assert numerical_v.x == pytest.approx(analytical_v.x, abs=1e-4)
        assert numerical_v.y == pytest.approx(analytical_v.y, abs=1e-4)


class TestSinusoidalTrajectory:
    def test_returns_to_center(self) -> None:
        """At t=0 with zero phase, position should be at center."""
        center = Position3D(10.0, 20.0, 0.0)
        traj = SinusoidalTrajectory(
            center=center,
            amplitude=Position3D(5.0, 3.0, 0.0),
            frequency=Position3D(1.0, 2.0, 0.0),
        )
        p = traj.position(0.0)
        assert p.x == pytest.approx(center.x)
        assert p.y == pytest.approx(center.y)

    def test_max_displacement(self) -> None:
        """At peak, displacement = amplitude."""
        traj = SinusoidalTrajectory(
            center=Position3D(0, 0, 0),
            amplitude=Position3D(5.0, 0.0, 0.0),
            frequency=Position3D(1.0, 0.0, 0.0),
        )
        # Peak at t = π/(2ω) = π/2
        p = traj.position(math.pi / 2)
        assert p.x == pytest.approx(5.0, abs=1e-10)
