"""Tests for estimation and control (Phase 4)."""

import math

import numpy as np
import pytest

from fsoc_pat.core.types import AngularError, ControlCommand
from fsoc_pat.estimation.kalman_filter import KalmanConfig, KalmanFilter
from fsoc_pat.estimation.motion_models import (
    MotionModelType,
    constant_acceleration_model,
    constant_position_model,
    constant_velocity_model,
    get_motion_model,
)
from fsoc_pat.control.pid import PIDConfig, PIDController
from fsoc_pat.control.closed_loop import ClosedLoopConfig, ClosedLoopEngine


# ─── Motion Model Tests ──────────────────────────────────

class TestMotionModels:

    def test_cp_dimensions(self) -> None:
        m = constant_position_model(0.05)
        assert m.F.shape == (2, 2)
        assert m.Q.shape == (2, 2)
        assert m.H.shape == (2, 2)
        assert m.state_dim == 2
        assert m.meas_dim == 2

    def test_cv_dimensions(self) -> None:
        m = constant_velocity_model(0.05)
        assert m.F.shape == (4, 4)
        assert m.Q.shape == (4, 4)
        assert m.H.shape == (2, 4)
        assert m.state_dim == 4
        assert m.meas_dim == 2

    def test_ca_dimensions(self) -> None:
        m = constant_acceleration_model(0.05)
        assert m.F.shape == (6, 6)
        assert m.Q.shape == (6, 6)
        assert m.H.shape == (2, 6)
        assert m.state_dim == 6
        assert m.meas_dim == 2

    def test_cv_propagation(self) -> None:
        """CV model should propagate position by velocity * dt."""
        dt = 0.1
        m = constant_velocity_model(dt)
        # state = [u=100, vu=10, v=200, vv=-5]
        x = np.array([100.0, 10.0, 200.0, -5.0])
        x_next = m.F @ x
        np.testing.assert_allclose(x_next[0], 101.0)  # 100 + 10*0.1
        np.testing.assert_allclose(x_next[1], 10.0)    # velocity unchanged
        np.testing.assert_allclose(x_next[2], 199.5)   # 200 + (-5)*0.1
        np.testing.assert_allclose(x_next[3], -5.0)

    def test_q_symmetric_positive_definite(self) -> None:
        """Q should be symmetric and PSD."""
        for model_fn in [constant_position_model, constant_velocity_model, constant_acceleration_model]:
            m = model_fn(0.05)
            np.testing.assert_allclose(m.Q, m.Q.T, atol=1e-15)
            eigvals = np.linalg.eigvalsh(m.Q)
            assert np.all(eigvals >= -1e-10)

    def test_factory(self) -> None:
        m = get_motion_model(MotionModelType.CONSTANT_VELOCITY, 0.05)
        assert m.state_dim == 4


# ─── Kalman Filter Tests ─────────────────────────────────

class TestKalmanFilter:

    def test_initialize(self) -> None:
        kf = KalmanFilter()
        kf.initialize(np.array([320.0, 240.0]))
        assert kf.initialized
        u, v = kf.estimated_position
        assert abs(u - 320.0) < 1
        assert abs(v - 240.0) < 1

    def test_predict_propagates(self) -> None:
        """Predict should move state according to model."""
        kf = KalmanFilter()
        kf.initialize(np.array([320.0, 240.0]))
        x_before = kf.state.copy()
        kf.predict()
        # Position should stay roughly the same (velocity initialized to 0)
        u, v = kf.estimated_position
        assert abs(u - 320.0) < 5
        assert abs(v - 240.0) < 5

    def test_update_corrects(self) -> None:
        """Update with a measurement should pull state toward it."""
        kf = KalmanFilter()
        kf.initialize(np.array([320.0, 240.0]))
        kf.predict()
        kf.update(np.array([325.0, 245.0]))
        u, v = kf.estimated_position
        # Should be pulled toward the measurement
        assert u > 320.0
        assert v > 240.0

    def test_convergence_on_stationary(self) -> None:
        """KF should converge to true position with repeated measurements."""
        kf = KalmanFilter(KalmanConfig(r_std=5.0, q_process=1.0))
        true_pos = np.array([300.0, 200.0])
        kf.initialize(true_pos + np.array([20.0, -15.0]))  # Start offset

        for _ in range(50):
            kf.predict()
            noise = np.random.randn(2) * 2.0
            kf.update(true_pos + noise)

        u, v = kf.estimated_position
        assert abs(u - 300.0) < 5.0
        assert abs(v - 200.0) < 5.0

    def test_predict_ahead(self) -> None:
        """Multi-step prediction should extrapolate."""
        kf = KalmanFilter()
        kf.initialize(np.array([100.0, 100.0]))
        # Feed some velocity signal
        for i in range(20):
            kf.predict()
            kf.update(np.array([100.0 + i * 2.0, 100.0]))

        ahead = kf.predict_ahead(5)
        # Should extrapolate forward (u should be > current)
        u_now = kf.estimated_position[0]
        assert ahead[0] > u_now

    def test_gating_rejects_outlier(self) -> None:
        """Large outlier should be rejected by Mahalanobis gate."""
        kf = KalmanFilter(KalmanConfig(gate_threshold=9.21))
        kf.initialize(np.array([320.0, 240.0]))

        # Normal updates
        for _ in range(10):
            kf.predict()
            kf.update(np.array([320.0, 240.0]) + np.random.randn(2) * 1.0)

        # Send wild outlier
        kf.predict()
        kf.update(np.array([1000.0, 1000.0]))

        # Position should NOT have jumped to 1000
        u, v = kf.estimated_position
        assert u < 500
        assert v < 500
        assert kf.gated_count > 0

    def test_reset(self) -> None:
        kf = KalmanFilter()
        kf.initialize(np.array([320.0, 240.0]))
        assert kf.initialized
        kf.reset()
        assert not kf.initialized

    def test_uninitialized_predict_raises(self) -> None:
        kf = KalmanFilter()
        with pytest.raises(RuntimeError):
            kf.predict()


# ─── PID Controller Tests ────────────────────────────────

class TestPIDController:

    def test_zero_error_zero_output(self) -> None:
        """Zero error should produce zero output."""
        pid = PIDController()
        cmd = pid.compute(AngularError(0.0, 0.0, 0.0), dt=0.02)
        assert cmd.pan_rate == 0.0
        assert cmd.tilt_rate == 0.0

    def test_positive_error_positive_output(self) -> None:
        """Positive error should produce positive control output."""
        pid = PIDController(PIDConfig(kp=5.0, ki=0.0, kd=0.0))
        err = AngularError(horizontal=0.1, vertical=0.05, total=0.112)
        cmd = pid.compute(err, dt=0.02)
        assert cmd.pan_rate > 0
        assert cmd.tilt_rate > 0

    def test_proportional_scaling(self) -> None:
        """Output should scale with Kp."""
        pid_low = PIDController(PIDConfig(kp=1.0, ki=0.0, kd=0.0))
        pid_high = PIDController(PIDConfig(kp=10.0, ki=0.0, kd=0.0))

        err = AngularError(horizontal=0.05, vertical=0.0, total=0.05)
        cmd_low = pid_low.compute(err, dt=0.02)
        cmd_high = pid_high.compute(err, dt=0.02)

        assert abs(cmd_high.pan_rate) > abs(cmd_low.pan_rate)

    def test_output_saturation(self) -> None:
        """Output should be limited to output_limit."""
        pid = PIDController(PIDConfig(kp=100.0, ki=0.0, kd=0.0, output_limit=2.0))
        err = AngularError(horizontal=1.0, vertical=1.0, total=1.414)
        cmd = pid.compute(err, dt=0.02)

        assert abs(cmd.pan_rate) <= 2.0
        assert abs(cmd.tilt_rate) <= 2.0

    def test_integral_accumulates(self) -> None:
        """Repeated errors should build integral term."""
        pid = PIDController(PIDConfig(kp=0.0, ki=5.0, kd=0.0, anti_windup=False))
        err = AngularError(horizontal=0.1, vertical=0.0, total=0.1)

        cmd1 = pid.compute(err, dt=0.02)
        cmd2 = pid.compute(err, dt=0.02)
        cmd3 = pid.compute(err, dt=0.02)

        # Each step adds more integral → larger output
        assert abs(cmd3.pan_rate) > abs(cmd1.pan_rate)

    def test_anti_windup(self) -> None:
        """Anti-windup should clamp the integral term."""
        pid = PIDController(PIDConfig(
            kp=0.0, ki=10.0, kd=0.0,
            integral_limit=0.5, anti_windup=True, output_limit=100.0,
        ))
        err = AngularError(horizontal=1.0, vertical=0.0, total=1.0)

        for _ in range(100):
            cmd = pid.compute(err, dt=0.1)

        # Integral should be clamped → output bounded
        assert abs(cmd.pan_rate) <= 10.0 * 0.5 + 0.1  # ki * limit + margin

    def test_reset(self) -> None:
        pid = PIDController()
        pid.compute(AngularError(0.1, 0.1, 0.14), dt=0.02)
        pid.reset()
        assert pid.integral_state == (0.0, 0.0)


# ─── Closed-Loop Engine Tests ────────────────────────────

class TestClosedLoopEngine:

    def test_runs_without_error(self) -> None:
        """Closed loop should complete without crash."""
        config = ClosedLoopConfig(duration_s=1.0, dt=0.05)
        engine = ClosedLoopEngine(config)
        result = engine.run()
        assert len(result.snapshots) == 20  # 1.0 / 0.05

    def test_detections_happen(self) -> None:
        """Most frames should have detections."""
        config = ClosedLoopConfig(duration_s=2.0, dt=0.05)
        engine = ClosedLoopEngine(config)
        result = engine.run()
        assert result.detection_rate > 0.5

    def test_error_stays_bounded(self) -> None:
        """Tracking error should remain bounded (not diverge)."""
        config = ClosedLoopConfig(duration_s=3.0, dt=0.02)
        engine = ClosedLoopEngine(config)
        result = engine.run()

        errors = result.gt_errors
        valid = [e for e in errors if not math.isnan(e)]

        # Steady-state errors (last 50%) should be bounded
        n = len(valid)
        if n > 20:
            late_errors = valid[n // 2:]
            mean_late = np.mean(late_errors)
            max_late = np.max(late_errors)
            # Tracking error should not diverge — mean < 1 rad
            assert mean_late < 1.0, f"Mean late error {mean_late:.3f} rad diverged"
            assert max_late < 2.0, f"Max late error {max_late:.3f} rad diverged"

    def test_gimbal_moves(self) -> None:
        """Gimbal should move in response to tracking error."""
        config = ClosedLoopConfig(duration_s=2.0, dt=0.02)
        engine = ClosedLoopEngine(config)
        result = engine.run()

        # Check that gimbal commands are non-zero at some point
        pan_cmds = [s.pan_cmd for s in result.snapshots]
        tilt_cmds = [s.tilt_cmd for s in result.snapshots]
        assert max(abs(c) for c in pan_cmds) > 0.01
        # tilt may be small if target doesn't move vertically

    def test_rms_error_finite(self) -> None:
        """RMS error should be finite."""
        config = ClosedLoopConfig(duration_s=1.0, dt=0.05)
        engine = ClosedLoopEngine(config)
        result = engine.run()
        assert not math.isnan(result.rms_gt_error_mrad)
        assert result.rms_gt_error_mrad >= 0
