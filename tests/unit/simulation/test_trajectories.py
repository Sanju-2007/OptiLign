"""Tests for trajectory generators."""

import math

import numpy as np
import pytest

from fsoc_pat.core.types import Acceleration3D, Position3D, Velocity3D
from fsoc_pat.simulation.trajectories import (
    CircularTrajectory,
    ConstantAccelerationTrajectory,
    FigureOfEightTrajectory,
    LinearTrajectory,
    RandomTrajectory,
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


class TestFigureOfEightTrajectory:
    def test_origin_crossing(self) -> None:
        """At t=0, trajectory should be at center."""
        center = Position3D(10.0, -20.0, 50.0)
        traj = FigureOfEightTrajectory(
            center=center, radius_x=15.0, radius_y=8.0, angular_rate=0.5,
        )
        p0 = traj.position(0.0)
        assert p0.x == pytest.approx(center.x, abs=1e-10)
        assert p0.y == pytest.approx(center.y, abs=1e-10)
        assert p0.z == pytest.approx(center.z, abs=1e-10)

    def test_bounding_box(self) -> None:
        """Trajectory must remain within [center - radius, center + radius]."""
        center = Position3D(0.0, 0.0, 100.0)
        rx, ry = 25.0, 12.0
        traj = FigureOfEightTrajectory(
            center=center, radius_x=rx, radius_y=ry, angular_rate=1.2,
        )
        for t in np.linspace(0.0, 10.0, 100):
            p = traj.position(float(t))
            assert abs(p.x - center.x) <= rx + 1e-10
            assert abs(p.y - center.y) <= 0.5 * ry + 1e-10
            assert p.z == center.z

    def test_derivative_consistency(self) -> None:
        """Numerical derivative of position ≈ analytical velocity, and velocity ≈ accel."""
        traj = FigureOfEightTrajectory(
            center=Position3D(5.0, 10.0, 200.0),
            radius_x=20.0, radius_y=15.0, angular_rate=0.8, phase=0.2,
        )
        t = 1.75
        dt = 1e-6
        p1 = traj.position(t)
        p2 = traj.position(t + dt)
        num_v = Velocity3D((p2.x - p1.x) / dt, (p2.y - p1.y) / dt, (p2.z - p1.z) / dt)
        ana_v = traj.velocity(t)
        assert num_v.x == pytest.approx(ana_v.x, abs=1e-4)
        assert num_v.y == pytest.approx(ana_v.y, abs=1e-4)

        v1 = traj.velocity(t)
        v2 = traj.velocity(t + dt)
        num_a = Acceleration3D((v2.x - v1.x) / dt, (v2.y - v1.y) / dt, (v2.z - v1.z) / dt)
        ana_a = traj.acceleration(t)
        assert num_a.x == pytest.approx(ana_a.x, abs=1e-3)
        assert num_a.y == pytest.approx(ana_a.y, abs=1e-3)

    def test_registry_lookup(self) -> None:
        """Trajectory should be registered under figure_of_8 and lemniscate."""
        from fsoc_pat.simulation.trajectories import TRAJECTORY_REGISTRY
        assert "figure_of_8" in TRAJECTORY_REGISTRY
        assert "lemniscate" in TRAJECTORY_REGISTRY
        assert TRAJECTORY_REGISTRY["figure_of_8"] is FigureOfEightTrajectory


class TestRandomTrajectory:
    def test_origin_crossing(self) -> None:
        """At t=0, trajectory should be at center."""
        center = Position3D(100.0, -50.0, 25.0)
        traj = RandomTrajectory(
            center=center,
            extent=Position3D(10.0, 30.0, 15.0),
            speed=24.0,
            seed=42,
        )
        p0 = traj.position(0.0)
        assert p0.x == pytest.approx(center.x, abs=1e-10)
        assert p0.y == pytest.approx(center.y, abs=1e-10)
        assert p0.z == pytest.approx(center.z, abs=1e-10)

    def test_bounding_box(self) -> None:
        """Trajectory must remain strictly within [center - extent, center + extent]."""
        center = Position3D(500.0, 20.0, -10.0)
        extent = Position3D(20.0, 45.0, 25.0)
        traj = RandomTrajectory(center=center, extent=extent, speed=35.0, seed=123)

        for t in np.linspace(0.0, 200.0, 200):
            p = traj.position(float(t))
            assert abs(p.x - center.x) <= extent.x + 1e-9
            assert abs(p.y - center.y) <= extent.y + 1e-9
            assert abs(p.z - center.z) <= extent.z + 1e-9

    def test_derivative_consistency(self) -> None:
        """Analytical velocity and acceleration should match numerical finite differences."""
        traj = RandomTrajectory(
            center=Position3D(0.0, 0.0, 0.0),
            extent=Position3D(15.0, 25.0, 18.0),
            speed=20.0,
            seed=99,
        )
        t = 4.25
        dt = 1e-6
        p1 = traj.position(t)
        p2 = traj.position(t + dt)
        num_v = Velocity3D((p2.x - p1.x) / dt, (p2.y - p1.y) / dt, (p2.z - p1.z) / dt)
        ana_v = traj.velocity(t)
        assert num_v.x == pytest.approx(ana_v.x, abs=1e-4)
        assert num_v.y == pytest.approx(ana_v.y, abs=1e-4)
        assert num_v.z == pytest.approx(ana_v.z, abs=1e-4)

        v1 = traj.velocity(t)
        v2 = traj.velocity(t + dt)
        num_a = Acceleration3D((v2.x - v1.x) / dt, (v2.y - v1.y) / dt, (v2.z - v1.z) / dt)
        ana_a = traj.acceleration(t)
        assert num_a.x == pytest.approx(ana_a.x, abs=1e-3)
        assert num_a.y == pytest.approx(ana_a.y, abs=1e-3)
        assert num_a.z == pytest.approx(ana_a.z, abs=1e-3)

    def test_registry_lookup(self) -> None:
        """Trajectory should be registered under random, random_path, and random_walk."""
        from fsoc_pat.simulation.trajectories import TRAJECTORY_REGISTRY
        assert "random" in TRAJECTORY_REGISTRY
        assert "random_path" in TRAJECTORY_REGISTRY
        assert "random_walk" in TRAJECTORY_REGISTRY
        assert TRAJECTORY_REGISTRY["random"] is RandomTrajectory
        assert TRAJECTORY_REGISTRY["random_path"] is RandomTrajectory


