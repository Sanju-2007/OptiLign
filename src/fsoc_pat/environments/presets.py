"""Standard High-Fidelity Virtual Environment Presets.

Includes tailored scenarios:
  1. SpaceISL: Inter-Satellite Link (vacuum, solar glints, micro-vibrations)
  2. LEODownlink: LEO Satellite to Ground Optical Station (Hufnagel-Valley Cn2)
  3. AirborneUAV: UAV-to-Ground Link (aerodynamic buffeting, motor vibration)
  4. Maritime: Shipboard FSOC Terminal (wave swell, sea spray, low fog)
  5. UrbanRooftop: City-scale Building-to-Building FSOC (sway, street clutter)
"""

from __future__ import annotations

from fsoc_pat.environments.builder import VirtualEnvironmentBuilder
from fsoc_pat.environments.models import AtmosphereModel, EnvironmentDefinition


def create_space_isl_environment() -> EnvironmentDefinition:
    """Create Space Inter-Satellite Link (ISL) environment."""
    return (
        VirtualEnvironmentBuilder(
            name="Deep_Space_ISL",
            description="Inter-Satellite Optical Link in LEO/MEO orbit (vacuum, high solar glare, reaction-wheel micro-vibrations).",
        )
        .with_platform(
            altitude_m=550_000.0,
            sway_amplitude_deg=0.001,
            sway_freq_hz=0.05,
            jitter_frequencies_hz=[60.0, 150.0, 300.0],
            jitter_amplitudes_urad=[4.0, 1.5, 0.8],
            wind_gust_buffet_std_urad=0.0,
        )
        .with_atmosphere(
            model=AtmosphereModel.VACUUM_SPACE,
            cn2=0.0,
            wind_speed_mps=0.0,
            extinction_coefficient_db_km=0.0,
            cloud_occlusion_prob=0.0,
            scintillation_index=0.0,
        )
        .with_disturbances(
            sensor_read_noise_std=2.0,
            background_clutter_count=40,  # Deep space starfield & cosmic background
            solar_glint_probability=0.15,  # Solar array reflections
            thermal_drift_rate_urad_s=0.1,
        )
        .with_optical_payload(
            laser_wavelength_nm=1550.0,
            laser_power_mw=2000.0,
            beam_divergence_mrad=0.08,  # Narrow diffraction-limited space beam
            rx_aperture_diameter_m=0.25,
            fsm_bandwidth_hz=1000.0,
        )
        .with_custom_telemetry_channel("orbit_beta_angle_deg", unit="deg", description="Sun orbit beta angle")
        .with_custom_telemetry_channel("reaction_wheel_rpm", unit="rpm", description="Reaction wheel speed")
        .build()
    )


def create_leo_downlink_environment() -> EnvironmentDefinition:
    """Create LEO Satellite Downlink to Ground Station environment."""
    return (
        VirtualEnvironmentBuilder(
            name="LEO_Downlink",
            description="LEO Satellite to Optical Ground Station pass (Hufnagel-Valley optical turbulence, high slewing rate).",
        )
        .with_platform(
            altitude_m=120.0,  # Ground telescope dome elevation
            sway_amplitude_deg=0.01,
            sway_freq_hz=0.1,
            jitter_frequencies_hz=[15.0, 35.0, 80.0],
            jitter_amplitudes_urad=[8.0, 4.0, 2.0],
            wind_gust_buffet_std_urad=3.5,
        )
        .with_atmosphere(
            model=AtmosphereModel.CLEAR_SKY,
            cn2=1.5e-14,  # Ground layer turbulence
            wind_speed_mps=5.0,
            extinction_coefficient_db_km=0.45,
            zenith_angle_deg=55.0,
            cloud_occlusion_prob=0.04,
            scintillation_index=0.12,
        )
        .with_disturbances(
            sensor_read_noise_std=5.0,
            background_clutter_count=20,
            solar_glint_probability=0.05,
            thermal_drift_rate_urad_s=0.4,
        )
        .with_optical_payload(
            laser_wavelength_nm=1550.0,
            laser_power_mw=1000.0,
            beam_divergence_mrad=0.5,
            rx_aperture_diameter_m=0.30,
            fsm_bandwidth_hz=800.0,
        )
        .with_custom_telemetry_channel("zenith_angle_deg", unit="deg", description="Slant path zenith angle")
        .with_custom_telemetry_channel("fried_parameter_r0_cm", unit="cm", description="Instantaneous Fried parameter")
        .build()
    )


def create_airborne_uav_environment() -> EnvironmentDefinition:
    """Create Airborne UAV-to-Ground or UAV-to-UAV environment."""
    return (
        VirtualEnvironmentBuilder(
            name="Airborne_UAV",
            description="Airborne UAV optical link (motor vibration harmonics, airframe buffeting, agile maneuvers).",
        )
        .with_platform(
            altitude_m=1500.0,
            sway_amplitude_deg=0.4,  # Airframe roll/yaw oscillations
            sway_freq_hz=0.6,
            jitter_frequencies_hz=[45.0, 90.0, 180.0],  # Motor / propeller harmonics
            jitter_amplitudes_urad=[40.0, 25.0, 12.0],
            wind_gust_buffet_std_urad=18.0,
        )
        .with_atmosphere(
            model=AtmosphereModel.STRATOSPHERE,
            cn2=8e-16,
            wind_speed_mps=15.0,
            extinction_coefficient_db_km=0.3,
            cloud_occlusion_prob=0.08,
            scintillation_index=0.08,
        )
        .with_disturbances(
            sensor_read_noise_std=8.0,
            background_clutter_count=15,
            solar_glint_probability=0.10,
            thermal_drift_rate_urad_s=1.2,
        )
        .with_optical_payload(
            laser_wavelength_nm=850.0,
            laser_power_mw=500.0,
            beam_divergence_mrad=1.5,
            rx_aperture_diameter_m=0.10,
            fsm_bandwidth_hz=1000.0,
        )
        .with_custom_telemetry_channel("uav_airspeed_mps", unit="m/s", description="UAV forward airspeed")
        .with_custom_telemetry_channel("airframe_load_factor_g", unit="g", description="Normal load factor")
        .build()
    )


def create_maritime_environment() -> EnvironmentDefinition:
    """Create Maritime Vessel FSOC environment."""
    return (
        VirtualEnvironmentBuilder(
            name="Maritime_Vessel",
            description="Shipboard FSOC terminal with ocean swell roll/pitch/heave, sea spray, and low-altitude mist.",
        )
        .with_platform(
            altitude_m=18.0,  # Ship mast height
            sway_amplitude_deg=1.2,  # Ship wave roll
            sway_freq_hz=0.18,  # Typical ocean wave period ~5.5s
            jitter_frequencies_hz=[12.0, 28.0, 60.0],  # Ship diesel propulsion vibration
            jitter_amplitudes_urad=[35.0, 18.0, 8.0],
            wind_gust_buffet_std_urad=12.0,
        )
        .with_atmosphere(
            model=AtmosphereModel.MARITIME_HAZE,
            cn2=4e-14,  # Strong boundary layer marine turbulence
            wind_speed_mps=10.0,
            extinction_coefficient_db_km=1.8,  # Sea aerosol attenuation
            cloud_occlusion_prob=0.12,
            scintillation_index=0.25,
        )
        .with_disturbances(
            sensor_read_noise_std=6.0,
            background_clutter_count=25,  # Sea surface specular sun reflections
            solar_glint_probability=0.25,
            thermal_drift_rate_urad_s=0.8,
        )
        .with_optical_payload(
            laser_wavelength_nm=1550.0,
            laser_power_mw=1200.0,
            beam_divergence_mrad=2.0,
            rx_aperture_diameter_m=0.12,
            fsm_bandwidth_hz=600.0,
        )
        .with_custom_telemetry_channel("sea_state", unit="Douglas", description="Sea state scale")
        .with_custom_telemetry_channel("ship_roll_deg", unit="deg", description="Ship instantaneous roll angle")
        .build()
    )


def create_urban_rooftop_environment() -> EnvironmentDefinition:
    """Create Urban Rooftop Building-to-Building FSOC environment."""
    return (
        VirtualEnvironmentBuilder(
            name="Urban_Rooftop",
            description="Building rooftop optical terminal (building sway, HVAC thermal turbulence plumes, city light clutter).",
        )
        .with_platform(
            altitude_m=45.0,  # 15-story building rooftop
            sway_amplitude_deg=0.04,  # Building wind sway
            sway_freq_hz=0.25,
            jitter_frequencies_hz=[20.0, 60.0, 120.0],  # HVAC machinery micro-tremors
            jitter_amplitudes_urad=[15.0, 6.0, 2.0],
            wind_gust_buffet_std_urad=6.0,
        )
        .with_atmosphere(
            model=AtmosphereModel.URBAN_TURBULENCE,
            cn2=2.5e-14,  # Urban thermal plume turbulence
            wind_speed_mps=4.5,
            extinction_coefficient_db_km=0.8,
            cloud_occlusion_prob=0.03,
            scintillation_index=0.18,
        )
        .with_disturbances(
            sensor_read_noise_std=4.5,
            background_clutter_count=50,  # Street lights, windows, vehicle headlights
            solar_glint_probability=0.08,
            thermal_drift_rate_urad_s=0.6,
        )
        .with_optical_payload(
            laser_wavelength_nm=1550.0,
            laser_power_mw=600.0,
            beam_divergence_mrad=1.0,
            rx_aperture_diameter_m=0.15,
            fsm_bandwidth_hz=800.0,
        )
        .with_custom_telemetry_channel("building_sway_mm", unit="mm", description="Rooftop structural sway displacement")
        .with_custom_telemetry_channel("ambient_temperature_c", unit="C", description="Urban ambient temperature")
        .build()
    )


PRESET_FACTORIES = {
    "space_isl": create_space_isl_environment,
    "leo_downlink": create_leo_downlink_environment,
    "airborne_uav": create_airborne_uav_environment,
    "maritime": create_maritime_environment,
    "urban_rooftop": create_urban_rooftop_environment,
}


def get_environment_preset(name: str) -> EnvironmentDefinition:
    """Retrieve an environment preset by name."""
    normalized = name.lower().replace("-", "_").strip()
    if normalized in PRESET_FACTORIES:
        return PRESET_FACTORIES[normalized]()
    raise ValueError(f"Unknown environment preset: '{name}'. Available: {list(PRESET_FACTORIES.keys())}")


def list_environment_presets() -> list[str]:
    """List all available environment preset keys."""
    return list(PRESET_FACTORIES.keys())
