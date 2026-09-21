"""Tests for gimbal dynamics."""

import math

import pytest

from fsoc_pat.core.types import AngularState
from fsoc_pat.simulation.dynamics import GimbalDynamics, VirtualGimbal


class TestGimbalDynamics:
    """Unit tests for VirtualGimbal."""

    def test_initial_state_zero(self) -> None:
        gimbal = VirtualGimbal()
        state = gimbal.state
        assert state.pan == 0.0
        assert state.tilt == 0.0
        assert state.pan_rate == 0.0
        assert state.tilt_rate == 0.0

    def test_set_position(self) -> None:
        gimbal = VirtualGimbal()
        gimbal.set_position(0.5, -0.3)
        assert gimbal.state.pan == pytest.approx(0.5)
        assert gimbal.state.tilt == pytest.approx(-0.3)

    def test_step_moves_gimbal(self) -> None:
        """Commanding a rate should cause the gimbal to move."""
        gimbal = VirtualGimbal(dynamics=GimbalDynamics(
            actuator_time_constant=0.0,  # No lag
            noise_std=0.0,
        ))
        dt = 0.01
        # Command a constant pan rate for many steps
        for _ in range(100):
            gimbal.step(0.5, 0.0, dt)

        # After 100 steps at 0.5 rad/s with dt=0.01 → ~0.5 rad
        assert gimbal.state.pan == pytest.approx(0.5, abs=0.05)

    def test_rate_limiting(self) -> None:
        """Commanded rate exceeding limit should be clamped."""
        dynamics = GimbalDynamics(
            pan_rate_limit=0.5,
            actuator_time_constant=0.0,
            noise_std=0.0,
        )
        gimbal = VirtualGimbal(dynamics=dynamics)

        # Command far beyond the limit
        gimbal.step(10.0, 0.0, 0.01)
        assert abs(gimbal.state.pan_rate) <= dynamics.pan_rate_limit + 1e-10

    def test_position_limits(self) -> None:
        """Gimbal should not exceed position limits."""
        dynamics = GimbalDynamics(
            pan_min=-0.5,
            pan_max=0.5,
            actuator_time_constant=0.0,
            noise_std=0.0,
        )
        gimbal = VirtualGimbal(dynamics=dynamics)

        # Drive past the limit
        for _ in range(200):
            gimbal.step(1.0, 0.0, 0.01)

        assert gimbal.state.pan <= dynamics.pan_max + 1e-10

    def test_deadband(self) -> None:
        """Commands below deadband should produce no motion."""
        dynamics = GimbalDynamics(
            deadband=0.01,
            actuator_time_constant=0.0,
            noise_std=0.0,
        )
        gimbal = VirtualGimbal(dynamics=dynamics)

        # Command below deadband
        gimbal.step(0.005, 0.005, 0.01)
        assert gimbal.state.pan == pytest.approx(0.0)
        assert gimbal.state.tilt == pytest.approx(0.0)

    def test_actuator_lag(self) -> None:
        """With a time constant, actual rate lags commanded rate."""
        dynamics = GimbalDynamics(
            actuator_time_constant=0.1,  # 100ms lag
            noise_std=0.0,
        )
        gimbal = VirtualGimbal(dynamics=dynamics)

        # First step: command 1.0 rad/s, actual rate should be << 1.0
        gimbal.step(1.0, 0.0, 0.01)
        assert gimbal.state.pan_rate < 0.5  # Should be much less than commanded

    def test_acceleration_limit(self) -> None:
        """Rate change per step should be bounded by accel limit."""
        dynamics = GimbalDynamics(
            angular_accel_limit=5.0,  # rad/s²
            actuator_time_constant=0.0,
            noise_std=0.0,
        )
        gimbal = VirtualGimbal(dynamics=dynamics)
        dt = 0.01

        gimbal.step(10.0, 0.0, dt)
        # Max delta = 5.0 * 0.01 = 0.05 rad/s from zero
        assert gimbal.state.pan_rate <= 5.0 * dt + 1e-10

    def test_commanded_not_equal_actual(self) -> None:
        """Core engineering invariant: commanded ≠ actual under dynamics."""
        dynamics = GimbalDynamics(
            actuator_time_constant=0.1,
            noise_std=0.0,
        )
        gimbal = VirtualGimbal(dynamics=dynamics)

        commanded_rate = 1.0
        gimbal.step(commanded_rate, 0.0, 0.01)

        # Actual rate should differ from commanded due to lag
        assert gimbal.state.pan_rate != pytest.approx(commanded_rate)

    def test_reset(self) -> None:
        gimbal = VirtualGimbal()
        gimbal.step(0.5, 0.3, 0.1)
        gimbal.reset()
        assert gimbal.state.pan == 0.0
        assert gimbal.state.tilt == 0.0
        assert gimbal.state.pan_rate == 0.0

    def test_bidirectional_motion(self) -> None:
        """Gimbal should move in both directions."""
        gimbal = VirtualGimbal(dynamics=GimbalDynamics(
            actuator_time_constant=0.0, noise_std=0.0,
        ))
        for _ in range(50):
            gimbal.step(0.5, 0.0, 0.01)
        assert gimbal.state.pan > 0

        for _ in range(100):
            gimbal.step(-0.5, 0.0, 0.01)
        assert gimbal.state.pan < 0

    def test_invalid_rate_limit(self) -> None:
        with pytest.raises(ValueError):
            GimbalDynamics(pan_rate_limit=0.0)
        with pytest.raises(ValueError):
            GimbalDynamics(pan_rate_limit=-1.0)
