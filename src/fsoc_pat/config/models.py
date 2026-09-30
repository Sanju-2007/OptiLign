"""Configuration data models.

All tunable engineering parameters are represented as typed dataclasses.
Every parameter has a name, unit, default, and meaning. No hidden
magic numbers — every threshold and constant is explicitly configured.

Units follow SI conventions (see core/types.py).
Design assumptions are clearly labeled in defaults.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class ScenarioConfig:
    """Top-level scenario parameters.

    Attributes:
        name: Scenario identifier.
        duration_s: Simulation duration (s).
        timestep_s: Simulation timestep (s).
        seed: Random seed for reproducibility.
    """

    name: str = "default"
    duration_s: float = 10.0
    timestep_s: float = 0.01  # 100 Hz
    seed: int = 42


@dataclass
class CameraConfig:
    """Camera and gimbal configuration.

    Attributes:
        width: Image width (px).
        height: Image height (px).
        hfov_rad: Horizontal field of view (rad).
        vfov_rad: Vertical field of view (rad). 0 = derive from hfov.
        fps: Frame rate (Hz).
        latency_s: Measurement latency (s). Design assumption.
        position: Camera position in world frame [x, y, z] (m).
        pan_rate_limit: Max pan rate (rad/s).
        tilt_rate_limit: Max tilt rate (rad/s).
        angular_accel_limit: Max angular acceleration (rad/s²).
        actuator_time_constant: First-order lag time constant (s).
    """

    width: int = 640
    height: int = 480
    hfov_rad: float = math.radians(60.0)
    vfov_rad: float = 0.0  # 0 = derive from hfov assuming square pixels
    fps: float = 30.0
    latency_s: float = 0.0
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    pan_rate_limit: float = 1.0       # rad/s
    tilt_rate_limit: float = 1.0      # rad/s
    angular_accel_limit: float = 10.0  # rad/s²
    actuator_time_constant: float = 0.05  # s
    monochrome: bool = True


@dataclass
class TargetConfig:
    """Target and trajectory configuration.

    Attributes:
        trajectory_type: Trajectory type name (see trajectories.py).
        position: Initial/center position [x, y, z] (m).
        velocity: Velocity vector [vx, vy, vz] (m/s). For linear/const-accel.
        acceleration: Acceleration vector (m/s²). For const-accel.
        radius: Orbit radius (m). For circular trajectory.
        angular_rate: Angular rate (rad/s). For circular trajectory.
        amplitude: Oscillation amplitude [ax, ay, az] (m). For sinusoidal.
        frequency: Oscillation frequency [fx, fy, fz] (rad/s). For sinusoidal.
        beacon_intensity: Beacon peak brightness (0-255).
        beacon_size_px: Beacon apparent size (px).
    """

    trajectory_type: str = "circular"
    position: tuple[float, float, float] = (100.0, 0.0, 0.0)
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    acceleration: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 10.0       # m
    angular_rate: float = 0.5  # rad/s
    amplitude: tuple[float, float, float] = (5.0, 5.0, 0.0)
    frequency: tuple[float, float, float] = (1.0, 0.5, 0.0)
    beacon_intensity: float = 220.0
    beacon_size_px: float = 5.0


@dataclass
class DetectorConfig:
    """Perception / detector configuration.

    Attributes:
        mode: Detection mode: "classical", "ai", "hybrid".
        confidence_threshold: Minimum confidence for valid detection [0, 1].
        brightness_threshold: Min brightness for candidate (0-255).
        min_area_px: Minimum candidate area (px²).
        max_area_px: Maximum candidate area (px²).
    """

    mode: str = "classical"
    confidence_threshold: float = 0.5
    brightness_threshold: float = 100.0
    min_area_px: float = 4.0
    max_area_px: float = 500.0


@dataclass
class EstimatorConfig:
    """State estimator configuration.

    Attributes:
        estimator_type: Filter type: "kf", "ekf", "ukf".
        process_noise: Process noise standard deviation (px/s²).
        measurement_noise: Measurement noise std dev (px).
        initial_covariance: Initial state covariance diagonal value.
    """

    estimator_type: str = "kf"
    process_noise: float = 1.0
    measurement_noise: float = 2.0
    initial_covariance: float = 100.0


@dataclass
class ControllerConfig:
    """Controller configuration.

    Attributes:
        controller_type: Controller type: "pid", "adaptive_pid", "lqr", "mpc".
        kp: Proportional gain.
        ki: Integral gain.
        kd: Derivative gain.
        derivative_filter_tau: Derivative low-pass filter time constant (s).
        integral_limit: Anti-windup integral saturation limit.
        output_limit: Maximum controller output (rad/s).
    """

    controller_type: str = "pid"
    kp: float = 2.0
    ki: float = 0.1
    kd: float = 0.5
    derivative_filter_tau: float = 0.02
    integral_limit: float = 1.0
    output_limit: float = 1.0


@dataclass
class DisturbanceConfig:
    """Disturbance engine configuration.

    All disturbances are individually configurable and can be
    enabled/disabled independently.

    Attributes:
        measurement_noise_std: Measurement noise std dev (px).
        vibration_amplitude: Vibration amplitude (rad).
        vibration_frequency: Vibration frequency (Hz).
        jitter_std: Random jitter std dev (rad).
        frame_drop_rate: Fraction of frames dropped [0, 1].
        latency_s: Additional processing latency (s).
        atmospheric_attenuation: Atmospheric intensity attenuation [0, 1].
    """

    measurement_noise_std: float = 0.0
    vibration_amplitude: float = 0.0
    vibration_frequency: float = 0.0
    jitter_std: float = 0.0
    frame_drop_rate: float = 0.0
    latency_s: float = 0.0
    atmospheric_attenuation: float = 0.0


@dataclass
class LinkConfig:
    """FSOC link configuration.

    Attributes:
        beam_divergence_rad: Transmit beam half-angle divergence (rad).
        acceptance_angle_rad: Receiver acceptance half-angle (rad).
    """

    beam_divergence_rad: float = math.radians(0.1)   # 0.1° half-angle
    acceptance_angle_rad: float = math.radians(0.05)  # 0.05° half-angle


@dataclass
class ComplianceThresholds:
    """Formal acceptance testing thresholds for PAT verification.

    Attributes:
        acquisition_time_max_s: Maximum initial acquisition time (s) [<= 2.0 s].
        tracking_error_max_px: Maximum tracking error in image plane (px) [<= 10.0 px].
        target_loss_max_pct: Maximum allowed target loss (%) [< 5.0 %].
        reacquisition_time_max_s: Maximum re-acquisition time (s) [<= 1.0 s].
        processing_fps_min: Minimum processing frame rate (Hz) [>= 20.0 FPS].
        consecutive_misses_for_loss: Number of consecutive missed frames to flag target lost [3].
    """

    acquisition_time_max_s: float = 2.0
    tracking_error_max_px: float = 10.0
    target_loss_max_pct: float = 5.0
    reacquisition_time_max_s: float = 1.0
    processing_fps_min: float = 20.0
    consecutive_misses_for_loss: int = 3


@dataclass
class ExperimentConfig:
    """Complete experiment configuration.

    Aggregates all subsystem configurations into a single object.
    This is the primary input to the simulation engine.
    """

    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    disturbances: DisturbanceConfig = field(default_factory=DisturbanceConfig)
    link: LinkConfig = field(default_factory=LinkConfig)
    compliance: ComplianceThresholds = field(default_factory=ComplianceThresholds)
    schema_version: str = "0.1.0"
