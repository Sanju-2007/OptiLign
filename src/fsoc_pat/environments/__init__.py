"""Virtual Environments package for custom scenarios, presets, and builder."""

from fsoc_pat.environments.builder import VirtualEnvironmentBuilder
from fsoc_pat.environments.models import (
    AtmosphereModel,
    AtmosphericChannelConfig,
    DisturbanceProfileConfig,
    EnvironmentDefinition,
    OpticalPayloadConfig,
    PlatformDynamicsConfig,
    PlatformType,
)
from fsoc_pat.environments.presets import (
    create_airborne_uav_environment,
    create_leo_downlink_environment,
    create_maritime_environment,
    create_space_isl_environment,
    create_urban_rooftop_environment,
    get_environment_preset,
    list_environment_presets,
)
from fsoc_pat.environments.runner import EnvironmentRunResults, EnvironmentSimulationRunner

__all__ = [
    "PlatformType",
    "AtmosphereModel",
    "PlatformDynamicsConfig",
    "AtmosphericChannelConfig",
    "DisturbanceProfileConfig",
    "OpticalPayloadConfig",
    "EnvironmentDefinition",
    "VirtualEnvironmentBuilder",
    "create_space_isl_environment",
    "create_leo_downlink_environment",
    "create_airborne_uav_environment",
    "create_maritime_environment",
    "create_urban_rooftop_environment",
    "get_environment_preset",
    "list_environment_presets",
    "EnvironmentRunResults",
    "EnvironmentSimulationRunner",
]
