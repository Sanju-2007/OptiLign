"""Integration tests for PAT State Machine reacquisition after signal occlusion."""

import math
import pytest
import numpy as np

from fsoc_pat.control.pid import PIDConfig, PIDController
from fsoc_pat.core.state import CameraState
from fsoc_pat.core.types import AngularError, AngularState, Orientation, Position3D
from fsoc_pat.disturbances.occlusion import OcclusionConfig, OcclusionModel
from fsoc_pat.estimation.kalman_filter import KalmanConfig, KalmanFilter
from fsoc_pat.geometry.angular_error import compute_angular_error_from_pixel
from fsoc_pat.geometry.camera_model import CameraIntrinsics
from fsoc_pat.geometry.projection import project_point_to_pixel
from fsoc_pat.pat.pat_state_machine import PATStateMachine
from fsoc_pat.pat.pat_types import PATConfig, PATState
from fsoc_pat.perception.classical_detector import ClassicalDetectorConfig
from fsoc_pat.perception.pipeline import PerceptionConfig, PerceptionPipeline
from fsoc_pat.simulation.beacon import BeaconProperties
from fsoc_pat.simulation.dynamics import VirtualGimbal
from fsoc_pat.simulation.trajectories import CircularTrajectory


from fsoc_pat.perception.preprocessing import PreprocessConfig
from fsoc_pat.perception.renderer import RenderConfig


def test_pat_closed_loop_occlusion_and_reacquisition():
    """Verify that when an occlusion occurs during tracking:
    1. System transitions from TRACKING to REACQUISITION.
    2. Gimbal does NOT run away to joint limits.
    3. Target is reacquired when the occlusion clears.
    """
    intrinsics = CameraIntrinsics.from_fov(640, 480, math.radians(60))
    traj = CircularTrajectory(
        center=Position3D(100.0, 0.0, 0.0),
        radius=5.0,
        angular_rate=0.2,
    )
    gimbal = VirtualGimbal()
    pipeline = PerceptionPipeline(
        intrinsics,
        PerceptionConfig(
            render=RenderConfig(noise_std=6.0, shot_noise=True, num_stars=5, seed=42),
            detector=ClassicalDetectorConfig(
                min_brightness=60.0,
                confidence_threshold=0.3,
                preprocess=PreprocessConfig(threshold_value=60),
            ),
        ),
    )
    kf = KalmanFilter(KalmanConfig(r_std=1.5, q_process=10.0))
    pid = PIDController(PIDConfig(kp=4.0, ki=0.2, kd=0.15, output_limit=1.0))
    pat_sm = PATStateMachine(
        PATConfig(
            loss_misses_threshold=3,
            coast_time_s=0.2,
            confirm_detections_threshold=2,
            acquisition_lock_error_rad=0.08,
        )
    )
    # Occlusion occurs between t=1.0s and t=2.0s
    occlusion = OcclusionModel(OcclusionConfig(intervals=[(1.0, 2.0)], transition_duration_s=0.05))

    pat_sm.set_state(PATState.SEARCH)
    dt = 0.02
    sim_time = 0.0

    state_history = []
    gimbal_pans = []
    gimbal_tilts = []

    for step in range(160):  # 3.2 seconds total
        t = step * dt
        sim_time = t

        # Target ground truth
        target_pos = traj.position(t).as_array()

        # Camera state
        g_state = gimbal.state
        cam_state = CameraState(
            position=Position3D(0, 0, 0),
            orientation=Orientation(yaw=g_state.pan, pitch=g_state.tilt, roll=0.0),
            angular_state=AngularState(pan_rate=g_state.pan_rate, tilt_rate=g_state.tilt_rate),
        )

        # Occlusion check: modulate beacon intensity
        is_blocked = occlusion.is_occluded(t)
        beacon_intensity = 0.0 if is_blocked else 240.0
        beacon = BeaconProperties(intensity=beacon_intensity, size_px=8.0)

        # Perception
        detection = pipeline.process_frame(cam_state, [(target_pos, beacon)], timestamp=t)
        detected = (detection.detected or (detection.best is not None and detection.confidence >= 0.3)) and not is_blocked

        # Estimation & Angular error
        if detected and detection.best is not None:
            meas = np.array([detection.best.u, detection.best.v])
            if not kf.initialized:
                kf.initialize(meas, dt)
            else:
                kf.predict(dt)
                kf.update(meas)
        elif kf.initialized:
            kf.predict(dt)

        if kf.initialized and detected:
            from fsoc_pat.core.types import PixelCoord
            est_u, est_v = kf.estimated_position
            ang_err = compute_angular_error_from_pixel(PixelCoord(est_u, est_v), intrinsics)
        else:
            ang_err = AngularError(horizontal=0.0, vertical=0.0, total=0.0)

        # PID compute
        pid_cmd = pid.compute(ang_err, dt, timestamp=t) if detected else None

        # Supervisory State Machine step
        pan_rate_cmd, tilt_rate_cmd = pat_sm.step(
            dt=dt,
            detected=detected,
            angular_error=ang_err,
            gimbal_pan=g_state.pan,
            gimbal_tilt=g_state.tilt,
            pid_cmd=pid_cmd,
            timestamp=t,
        )

        # Step gimbal
        gimbal.step(pan_rate_cmd, tilt_rate_cmd, dt)

        state_history.append(pat_sm.state)
        gimbal_pans.append(math.degrees(g_state.pan))
        gimbal_tilts.append(math.degrees(g_state.tilt))

    # Assertions:
    # 1. State machine reached TRACKING before occlusion
    assert any(s == PATState.TRACKING for s in state_history[:50])

    # 2. State machine transitioned to REACQUISITION during occlusion (around t=1.2 - 2.0s)
    assert any(s == PATState.REACQUISITION for s in state_history[50:110])

    # 3. CRITICAL ANTI-RUNAWAY: Gimbal angles never reached joint stops (stayed bounded within ±40 deg)
    assert max(map(abs, gimbal_pans)) < 45.0
    assert max(map(abs, gimbal_tilts)) < 45.0

    # 4. State machine reacquired and locked track after occlusion cleared
    assert any(s == PATState.TRACKING for s in state_history[110:])
