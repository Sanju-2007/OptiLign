"""Unit tests for VirtualEnvironmentBuilder, Presets, and Serialization."""

import json
from pathlib import Path
import pytest

from fsoc_pat.environments.builder import VirtualEnvironmentBuilder
from fsoc_pat.environments.models import AtmosphereModel, EnvironmentDefinition
from fsoc_pat.environments.presets import (
    create_airborne_uav_environment,
    create_leo_downlink_environment,
    create_maritime_environment,
    create_space_isl_environment,
    create_urban_rooftop_environment,
    get_environment_preset,
    list_environment_presets,
)


def test_builder_constructs_valid_environment():
    builder = VirtualEnvironmentBuilder("Custom_Testing_Env", "Unit test description")
    env = (
        builder.with_platform(altitude_m=200.0, sway_amplitude_deg=0.1)
        .with_atmosphere(model=AtmosphereModel.URBAN_TURBULENCE, cn2=1e-13)
        .with_optical_payload(laser_power_mw=750.0, fsm_bandwidth_hz=900.0)
        .with_custom_telemetry_channel("my_custom_flux", unit="W/m2")
        .build()
    )

    assert env.name == "Custom_Testing_Env"
    assert env.platform.base_altitude_m == 200.0
    assert env.atmosphere.model == AtmosphereModel.URBAN_TURBULENCE
    assert env.atmosphere.cn2 == 1e-13
    assert env.payload.laser_power_mw == 750.0
    assert env.payload.fsm_bandwidth_hz == 900.0
    assert "my_custom_flux" in env.telemetry_schema.channels


def test_presets_creation():
    presets = list_environment_presets()
    assert "space_isl" in presets
    assert "leo_downlink" in presets
    assert "airborne_uav" in presets
    assert "maritime" in presets
    assert "urban_rooftop" in presets

    space_env = get_environment_preset("space_isl")
    assert space_env.atmosphere.model == AtmosphereModel.VACUUM_SPACE
    assert space_env.platform.base_altitude_m > 100_000.0

    maritime_env = get_environment_preset("maritime")
    assert maritime_env.atmosphere.model == AtmosphereModel.MARITIME_HAZE


def test_json_serialization_roundtrip(tmp_path: Path):
    env = create_leo_downlink_environment()
    json_path = tmp_path / "leo_env.json"
    env.save_json(json_path)

    assert json_path.exists()
    loaded = EnvironmentDefinition.load_json(json_path)

    assert loaded.name == env.name
    assert loaded.atmosphere.cn2 == env.atmosphere.cn2
    assert loaded.payload.laser_power_mw == env.payload.laser_power_mw
    assert "zenith_angle_deg" in loaded.telemetry_schema.channels
