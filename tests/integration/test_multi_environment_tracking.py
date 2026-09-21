"""Integration tests for Multi-Environment Tracking with Dual-Stage FSM and Custom Telemetry."""

import math
from pathlib import Path
import pytest

from fsoc_pat.environments.builder import VirtualEnvironmentBuilder
from fsoc_pat.environments.presets import (
    create_airborne_uav_environment,
    create_leo_downlink_environment,
    create_maritime_environment,
    create_space_isl_environment,
    create_urban_rooftop_environment,
)
from fsoc_pat.environments.runner import EnvironmentSimulationRunner


@pytest.mark.parametrize("preset_factory", [
    create_space_isl_environment,
    create_leo_downlink_environment,
    create_urban_rooftop_environment,
])
def test_environment_runner_execution_and_accuracy(preset_factory, tmp_path: Path):
    env = preset_factory()
    csv_path = tmp_path / f"{env.name}_telemetry.csv"

    runner = EnvironmentSimulationRunner(
        environment=env,
        duration_s=1.0,  # 50 steps
        dt=0.02,
        seed=123,
    )

    results = runner.run(csv_output_path=csv_path)

    assert results.environment_name == env.name
    assert results.steps_executed == 50
    assert csv_path.exists()
    assert results.fine_rms_urad < results.coarse_rms_urad  # FSM must improve tracking error
    assert results.fine_rms_urad < 15.0  # Ultra-accurate pointing threshold (< 15 micro-radians in turbulence)


def test_custom_environment_with_custom_telemetry_channels(tmp_path: Path):
    # User creates their own custom virtual environment
    custom_env = (
        VirtualEnvironmentBuilder("User_Stratospheric_Balloon")
        .with_platform(altitude_m=22000.0, sway_amplitude_deg=0.02)
        .with_disturbances(sensor_read_noise_std=3.0)
        .with_optical_payload(laser_power_mw=1500.0)
        .with_custom_telemetry_channel("balloon_lift_gas_temp_c", unit="C", default_value=-45.0)
        .build()
    )

    csv_path = tmp_path / "balloon_telemetry.csv"
    runner = EnvironmentSimulationRunner(custom_env, duration_s=0.5, dt=0.02)
    results = runner.run(csv_output_path=csv_path)

    assert results.steps_executed == 25
    assert csv_path.exists()

    with open(csv_path) as f:
        header = f.readline().strip().split(",")
    assert "balloon_lift_gas_temp_c" in header
    assert "fine_error_urad" in header
