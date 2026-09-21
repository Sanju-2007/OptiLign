"""Closed-loop simulation engine.

Implements the full PAT loop:
  Simulate → Render → Detect → Estimate → Control → Gimbal → Next Frame

This is the integration layer that connects Phases 1-4:
  Phase 1: SimulationEngine (ground truth, gimbal dynamics)
  Phase 2: PerceptionPipeline (render + detect)
  Phase 3: AI/Hybrid detection (optional)
  Phase 4: KalmanFilter (estimation) + PIDController (control)

The closed loop runs at the simulation timestep. Each step:
  1. World advances (target moves, gimbal updates)
  2. Camera renders the scene
  3. Detector finds the beacon
  4. Estimator predicts + updates from detection
  5. Angular error computed from estimate (NOT ground truth)
  6. Controller computes gimbal command
  7. Gimbal receives command

Ground truth is logged separately for performance evaluation
but is NEVER fed to the estimator or controller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from fsoc_pat.core.state import CameraState
from fsoc_pat.core.types import AngularError, AngularState, ControlCommand, Orientation, Position3D
from fsoc_pat.control.pid import PIDConfig, PIDController
from fsoc_pat.estimation.kalman_filter import KalmanConfig, KalmanFilter
from fsoc_pat.estimation.motion_models import MotionModelType
from fsoc_pat.geometry.angular_error import compute_angular_error_from_pixel
from fsoc_pat.geometry.camera_model import CameraIntrinsics
from fsoc_pat.geometry.projection import project_point_to_pixel
from fsoc_pat.pat.pat_state_machine import PATStateMachine
from fsoc_pat.pat.pat_types import PATConfig, PATState
from fsoc_pat.perception.classical_detector import ClassicalDetectorConfig
from fsoc_pat.perception.detection_types import Detection
from fsoc_pat.perception.pipeline import PerceptionConfig, PerceptionPipeline
from fsoc_pat.perception.preprocessing import PreprocessConfig
from fsoc_pat.perception.renderer import RenderConfig
from fsoc_pat.link.budget import FSOCLinkBudget
from fsoc_pat.link.link_types import LaserConfig
from fsoc_pat.simulation.beacon import BeaconProperties
from fsoc_pat.simulation.dynamics import VirtualGimbal
from fsoc_pat.simulation.trajectories import CircularTrajectory


@dataclass
class ClosedLoopConfig:
    """Configuration for the closed-loop simulation.

    Attributes:
        duration_s: Simulation duration (s).
        dt: Timestep (s).
        target_trajectory: Target trajectory type.
        target_center: Target trajectory center (m).
        target_radius: Trajectory radius (m).
        target_angular_rate: Trajectory angular rate (rad/s).
        beacon: Beacon properties.
        intrinsics: Camera intrinsics (auto-created if None).
        pid: PID controller config.
        kalman: Kalman filter config.
        perception: Perception pipeline config.
        seed: Random seed.
    """

    duration_s: float = 5.0
    dt: float = 0.02  # 50 Hz
    target_center: tuple[float, float, float] = (100.0, 0.0, 0.0)
    target_radius: float = 10.0
    target_angular_rate: float = 1.0
    beacon: BeaconProperties = field(default_factory=lambda: BeaconProperties(intensity=230.0, size_px=8.0))
    pid: PIDConfig = field(default_factory=PIDConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    perception: PerceptionConfig | None = None
    pat: PATConfig | None = None
    seed: int = 42


@dataclass
class ClosedLoopSnapshot:
    """State at a single timestep.

    Attributes:
        time: Simulation time (s).
        gt_u: Ground truth pixel u.
        gt_v: Ground truth pixel v.
        det_u: Detected pixel u (NaN if no detection).
        det_v: Detected pixel v (NaN if no detection).
        est_u: Estimated pixel u.
        est_v: Estimated pixel v.
        error_h: Horizontal angular error (rad) from estimate.
        error_v: Vertical angular error (rad) from estimate.
        error_total: Total angular error (rad) from estimate.
        gt_error_total: Total angular error from ground truth (rad).
        pan_cmd: Commanded pan rate (rad/s).
        tilt_cmd: Commanded tilt rate (rad/s).
        gimbal_pan: Actual gimbal pan angle (rad).
        gimbal_tilt: Actual gimbal tilt angle (rad).
        confidence: Detection confidence.
        detected: Whether detection succeeded.
    """

    time: float
    gt_u: float
    gt_v: float
    det_u: float
    det_v: float
    est_u: float
    est_v: float
    error_h: float
    error_v: float
    error_total: float
    gt_error_total: float
    pan_cmd: float
    tilt_cmd: float
    gimbal_pan: float
    gimbal_tilt: float
    confidence: float
    detected: bool
    pat_state: str = "TRACKING"
    rx_power_dbm: float = -40.0
    pointing_loss_db: float = 0.0
    link_margin_db: float = 0.0
    ber: float = 1e-12


@dataclass
class ClosedLoopResult:
    """Complete closed-loop simulation result."""

    snapshots: list[ClosedLoopSnapshot]
    config: ClosedLoopConfig

    @property
    def times(self) -> list[float]:
        return [s.time for s in self.snapshots]

    @property
    def gt_errors(self) -> list[float]:
        return [s.gt_error_total for s in self.snapshots]

    @property
    def est_errors(self) -> list[float]:
        return [s.error_total for s in self.snapshots]

    @property
    def detection_rate(self) -> float:
        if not self.snapshots:
            return 0.0
        return sum(1 for s in self.snapshots if s.detected) / len(self.snapshots)

    @property
    def mean_gt_error_mrad(self) -> float:
        errors = [e for e in self.gt_errors if not math.isnan(e)]
        return float(np.mean(errors)) * 1000 if errors else float("nan")

    @property
    def rms_gt_error_mrad(self) -> float:
        errors = [e for e in self.gt_errors if not math.isnan(e)]
        return float(np.sqrt(np.mean(np.array(errors) ** 2))) * 1000 if errors else float("nan")


class ClosedLoopEngine:
    """Full closed-loop PAT simulation.

    Connects all subsystems into the complete tracking loop.
    """

    def __init__(self, config: ClosedLoopConfig | None = None) -> None:
        self._config = config or ClosedLoopConfig()

    def run(self) -> ClosedLoopResult:
        """Execute the full closed-loop simulation.

        Returns:
            ClosedLoopResult with per-frame snapshots.
        """
        cfg = self._config

        # Build subsystems
        intrinsics = CameraIntrinsics.from_fov(640, 480, math.radians(60))

        perception_config = cfg.perception or PerceptionConfig(
            render=RenderConfig(
                noise_std=8.0, shot_noise=True, num_stars=10,
                enable_bloom=True, seed=cfg.seed,
            ),
            detector=ClassicalDetectorConfig(
                min_area_px=3.0, max_area_px=800.0,
                min_circularity=0.2, min_brightness=60.0,
                confidence_threshold=0.3,
                preprocess=PreprocessConfig(
                    blur_kernel=5, threshold_method="global",
                    threshold_value=60, morph_open_kernel=2, morph_close_kernel=2,
                ),
            ),
        )
        pipeline = PerceptionPipeline(intrinsics, perception_config)
        kf = KalmanFilter(cfg.kalman)
        pid = PIDController(cfg.pid)
        gimbal = VirtualGimbal()
        pat_sm = PATStateMachine(cfg.pat) if cfg.pat is not None else None
        if pat_sm is not None:
            pat_sm.set_state(PATState.TRACKING)

        link_budget = FSOCLinkBudget(LaserConfig(power_w=1.0, beam_waist_w0_m=0.001))

        # Target trajectory
        traj = CircularTrajectory(
            center=Position3D(*cfg.target_center),
            radius=cfg.target_radius,
            angular_rate=cfg.target_angular_rate,
        )

        n_steps = int(cfg.duration_s / cfg.dt)
        snapshots: list[ClosedLoopSnapshot] = []

        for i in range(n_steps):
            t = i * cfg.dt

            # 1. Target ground truth
            target_pos = traj.position(t).as_array()

            # 2. Camera state (includes gimbal orientation)
            gimbal_state = gimbal.state
            camera_state = CameraState(
                position=Position3D(0, 0, 0),
                orientation=Orientation(
                    yaw=gimbal_state.pan,
                    pitch=gimbal_state.tilt,
                    roll=0.0,
                ),
                angular_state=AngularState(
                    pan_rate=gimbal_state.pan_rate,
                    tilt_rate=gimbal_state.tilt_rate,
                ),
            )

            # 3. Ground truth projection
            gt_pixel = project_point_to_pixel(target_pos, camera_state, intrinsics)
            gt_u = gt_pixel.u if gt_pixel else float("nan")
            gt_v = gt_pixel.v if gt_pixel else float("nan")

            # Ground truth angular error (for evaluation only)
            if gt_pixel is not None:
                gt_ang_err = compute_angular_error_from_pixel(gt_pixel, intrinsics)
                gt_error_total = gt_ang_err.total
            else:
                gt_error_total = float("nan")

            # 4. Perception (render + detect)
            detection = pipeline.process_frame(
                camera_state, [(target_pos, cfg.beacon)], timestamp=t,
            )

            det_u = float("nan")
            det_v = float("nan")

            if detection.detected and detection.best is not None:
                det_u = detection.best.u
                det_v = detection.best.v

            # 5. Estimation
            if detection.detected and detection.best is not None:
                meas = np.array([detection.best.u, detection.best.v])
                if not kf.initialized:
                    kf.initialize(meas, cfg.dt)
                else:
                    kf.predict(cfg.dt)
                    kf.update(meas)
            elif kf.initialized:
                kf.predict(cfg.dt)

            est_u, est_v = kf.estimated_position

            # 6. Control — compute error from ESTIMATED position (not GT)
            if kf.initialized:
                from fsoc_pat.core.types import PixelCoord
                est_pixel = PixelCoord(u=est_u, v=est_v)
                ang_err = compute_angular_error_from_pixel(est_pixel, intrinsics)
            else:
                ang_err = AngularError(horizontal=0.0, vertical=0.0, total=0.0)

            # Feed error to PID controller (invert vertical error because positive vertical error
            # in Z-down camera coordinates requires negative gimbal tilt rate to center target)
            ctrl_err = AngularError(horizontal=ang_err.horizontal, vertical=-ang_err.vertical, total=ang_err.total)
            cmd = pid.compute(ctrl_err, cfg.dt, timestamp=t)

            # 7. Gimbal — step with commanded rates (or PAT state machine)
            if pat_sm is not None:
                pan_rate_cmd, tilt_rate_cmd = pat_sm.step(
                    dt=cfg.dt,
                    detected=detection.detected,
                    angular_error=ang_err,
                    gimbal_pan=gimbal_state.pan,
                    gimbal_tilt=gimbal_state.tilt,
                    pid_cmd=cmd,
                    timestamp=t,
                )
            else:
                pan_rate_cmd, tilt_rate_cmd = cmd.pan_rate, cmd.tilt_rate

            gimbal.step(pan_rate_cmd, tilt_rate_cmd, cfg.dt)

            dist_to_tgt = float(np.linalg.norm(target_pos))
            eval_err = gt_error_total if not math.isnan(gt_error_total) else 0.05
            link_res = link_budget.evaluate(dist_to_tgt, eval_err)

            snapshots.append(ClosedLoopSnapshot(
                time=t,
                gt_u=gt_u, gt_v=gt_v,
                det_u=det_u, det_v=det_v,
                est_u=est_u, est_v=est_v,
                error_h=ang_err.horizontal,
                error_v=ang_err.vertical,
                error_total=ang_err.total,
                gt_error_total=gt_error_total,
                pan_cmd=cmd.pan_rate,
                tilt_cmd=cmd.tilt_rate,
                gimbal_pan=gimbal_state.pan,
                gimbal_tilt=gimbal_state.tilt,
                confidence=detection.confidence,
                detected=detection.detected,
                pat_state=pat_sm.state.name if pat_sm is not None else "TRACKING",
                rx_power_dbm=link_res.rx_power_dbm,
                pointing_loss_db=link_res.pointing_loss_db,
                link_margin_db=link_res.link_margin_db,
                ber=link_res.ber,
            ))

        return ClosedLoopResult(snapshots=snapshots, config=cfg)
