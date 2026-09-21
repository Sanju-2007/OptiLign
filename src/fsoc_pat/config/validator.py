"""Configuration validator.

Validates all configuration values before simulation starts.
Catches invalid FOV, negative rate limits, inconsistent parameters,
and other misconfigurations that would cause silent failures.
"""

from __future__ import annotations

import math

from fsoc_pat.config.models import ExperimentConfig
from fsoc_pat.core.errors import ConfigurationError


def validate_config(config: ExperimentConfig) -> None:
    """Validate an experiment configuration.

    Checks all subsystem configs for consistency and validity.
    Raises ConfigurationError on the first invalid parameter found.

    Args:
        config: Configuration to validate.

    Raises:
        ConfigurationError: If any parameter is invalid.
    """
    _validate_scenario(config)
    _validate_camera(config)
    _validate_target(config)
    _validate_controller(config)
    _validate_link(config)


def _validate_scenario(config: ExperimentConfig) -> None:
    sc = config.scenario

    if sc.duration_s <= 0:
        raise ConfigurationError(f"Scenario duration must be positive: {sc.duration_s}s")

    if sc.timestep_s <= 0:
        raise ConfigurationError(f"Timestep must be positive: {sc.timestep_s}s")

    if sc.timestep_s > sc.duration_s:
        raise ConfigurationError(
            f"Timestep ({sc.timestep_s}s) cannot exceed duration ({sc.duration_s}s)"
        )


def _validate_camera(config: ExperimentConfig) -> None:
    cam = config.camera

    if cam.width <= 0 or cam.height <= 0:
        raise ConfigurationError(
            f"Image dimensions must be positive: {cam.width}×{cam.height}"
        )

    if cam.hfov_rad <= 0 or cam.hfov_rad >= math.pi:
        raise ConfigurationError(
            f"HFOV must be in (0, π) rad: {cam.hfov_rad} "
            f"({math.degrees(cam.hfov_rad):.1f}°)"
        )

    if cam.vfov_rad != 0 and (cam.vfov_rad <= 0 or cam.vfov_rad >= math.pi):
        raise ConfigurationError(
            f"VFOV must be in (0, π) rad or 0 for auto: {cam.vfov_rad}"
        )

    if cam.fps <= 0:
        raise ConfigurationError(f"FPS must be positive: {cam.fps}")

    if cam.latency_s < 0:
        raise ConfigurationError(f"Latency must be non-negative: {cam.latency_s}s")

    if cam.pan_rate_limit <= 0:
        raise ConfigurationError(f"Pan rate limit must be positive: {cam.pan_rate_limit}")

    if cam.tilt_rate_limit <= 0:
        raise ConfigurationError(f"Tilt rate limit must be positive: {cam.tilt_rate_limit}")


def _validate_target(config: ExperimentConfig) -> None:
    tgt = config.target

    valid_types = {
        "stationary", "linear", "constant_acceleration",
        "circular", "sinusoidal", "spiral", "maneuver", "random",
    }
    if tgt.trajectory_type.lower() not in valid_types:
        raise ConfigurationError(
            f"Unknown trajectory type: '{tgt.trajectory_type}'. "
            f"Valid: {sorted(valid_types)}"
        )

    if tgt.trajectory_type.lower() == "circular" and tgt.radius <= 0:
        raise ConfigurationError(f"Circular trajectory radius must be positive: {tgt.radius}")

    if tgt.beacon_intensity < 0 or tgt.beacon_intensity > 255:
        raise ConfigurationError(
            f"Beacon intensity must be in [0, 255]: {tgt.beacon_intensity}"
        )

    if tgt.beacon_size_px <= 0:
        raise ConfigurationError(f"Beacon size must be positive: {tgt.beacon_size_px}")


def _validate_controller(config: ExperimentConfig) -> None:
    ctrl = config.controller

    valid_types = {"pid", "adaptive_pid", "lqr", "mpc"}
    if ctrl.controller_type.lower() not in valid_types:
        raise ConfigurationError(
            f"Unknown controller type: '{ctrl.controller_type}'. "
            f"Valid: {sorted(valid_types)}"
        )

    if ctrl.output_limit <= 0:
        raise ConfigurationError(f"Output limit must be positive: {ctrl.output_limit}")


def _validate_link(config: ExperimentConfig) -> None:
    lnk = config.link

    if lnk.beam_divergence_rad <= 0:
        raise ConfigurationError(
            f"Beam divergence must be positive: {lnk.beam_divergence_rad}"
        )

    if lnk.acceptance_angle_rad <= 0:
        raise ConfigurationError(
            f"Acceptance angle must be positive: {lnk.acceptance_angle_rad}"
        )
