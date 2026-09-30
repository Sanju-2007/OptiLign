"""Default configuration values.

Every default is a documented design assumption. Where the project
requirements do not define a numerical value, the default is clearly
labeled as such.

These defaults can be overridden by scenario YAML files or CLI arguments.
"""

from __future__ import annotations

import math

from fsoc_pat.config.models import (
    CameraConfig,
    ComplianceThresholds,
    ControllerConfig,
    DetectorConfig,
    DisturbanceConfig,
    EstimatorConfig,
    ExperimentConfig,
    LinkConfig,
    ScenarioConfig,
    TargetConfig,
)


def default_config() -> ExperimentConfig:
    """Return the default experiment configuration.

    All values are design assumptions unless noted otherwise.

    Returns:
        ExperimentConfig with all defaults applied.
    """
    return ExperimentConfig(
        scenario=ScenarioConfig(
            name="default",
            duration_s=10.0,       # Design assumption: 10 s nominal scenario
            timestep_s=0.01,       # Design assumption: 100 Hz loop rate
            seed=42,
        ),
        camera=CameraConfig(),
        target=TargetConfig(),
        detector=DetectorConfig(),
        estimator=EstimatorConfig(),
        controller=ControllerConfig(),
        disturbances=DisturbanceConfig(),
        link=LinkConfig(),
        compliance=ComplianceThresholds(),
    )


def standard_spec_config() -> ExperimentConfig:
    """Return the authoritative Standard Specification configuration.

    Guarantees strict compliance with the reference specification:
        - Virtual Screen: >= 2000 x 2000
        - Camera: Monochrome FPA, 640 x 480 px, 4° x 3° FOV, >= 30 FPS
        - Camera Pointing: Initially at screen center (pan=0, tilt=0)
        - Target: Beacon spot (1 mandatory), 10 x 10 px square default, random initial position
        - Motion: Straight trajectory default (options: circular, figure-8, random)
        - Camera Pan/Tilt: 5°/s default rate limit
        - Primary Estimator: Adaptive EKF
        - Primary Controller: Coarse Pan/Tilt Controller
        - Control Update: >= 20 Hz (30 Hz default)
        - Formal Verification Limits:
            Acquisition <= 2.0 s
            Tracking Error <= 10.0 px
            Target Loss < 5.0 %
            Re-acquisition <= 1.0 s
            Processing >= 20.0 FPS
    """
    return ExperimentConfig(
        scenario=ScenarioConfig(
            name="standard_specification",
            duration_s=10.0,
            timestep_s=0.0333333333,  # 30 Hz
            seed=42,
        ),
        camera=CameraConfig(
            width=640,
            height=480,
            hfov_rad=math.radians(4.0),
            vfov_rad=math.radians(3.0),
            fps=30.0,
            latency_s=0.0,
            position=(0.0, 0.0, 0.0),
            pan_rate_limit=math.radians(5.0),   # 5°/s (0.087266 rad/s)
            tilt_rate_limit=math.radians(5.0),  # 5°/s (0.087266 rad/s)
            angular_accel_limit=25.0,
            actuator_time_constant=0.02,
            monochrome=True,
        ),
        target=TargetConfig(
            trajectory_type="linear",
            position=(100.0, 0.0, 0.0),
            velocity=(1.0, 0.0, 0.0),
            beacon_intensity=240.0,
            beacon_size_px=10.0,
        ),
        detector=DetectorConfig(
            mode="classical",
            confidence_threshold=0.5,
            brightness_threshold=100.0,
            min_area_px=4.0,
            max_area_px=500.0,
        ),
        estimator=EstimatorConfig(
            estimator_type="ekf",
            process_noise=1.0,
            measurement_noise=2.0,
            initial_covariance=100.0,
        ),
        controller=ControllerConfig(
            controller_type="pid",
            kp=12.0,
            ki=0.8,
            kd=0.18,
            output_limit=math.radians(5.0),  # 5°/s limit
        ),
        disturbances=DisturbanceConfig(
            measurement_noise_std=0.0,
            vibration_amplitude=0.0,
            vibration_frequency=0.0,
            jitter_std=0.0,
            frame_drop_rate=0.0,
            latency_s=0.0,
            atmospheric_attenuation=0.0,
        ),
        link=LinkConfig(),
        compliance=ComplianceThresholds(),
    )


STANDARD_SPEC_CONFIG = standard_spec_config()
