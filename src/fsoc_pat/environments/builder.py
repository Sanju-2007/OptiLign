"""Fluent Builder for Custom Virtual Simulation Environments.

Enables users to programmatically construct, parameterize, and customize
complete FSOC environments and their telemetry schemas.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from fsoc_pat.environments.models import (
    AtmosphereModel,
    AtmosphericChannelConfig,
    DisturbanceProfileConfig,
    EnvironmentDefinition,
    OpticalPayloadConfig,
    PlatformDynamicsConfig,
    PlatformType,
)
from fsoc_pat.telemetry.channel import TelemetryChannel, TelemetryDataType, TelemetrySchema


class VirtualEnvironmentBuilder:
    """Builder pattern for creating tailored virtual simulation environments.

    Example:
        env = (
            VirtualEnvironmentBuilder("LEO_Maritime_Pass")
            .with_platform(altitude_m=50.0, sway_amplitude_deg=0.8, sway_freq_hz=0.15)
            .with_atmosphere(model=AtmosphereModel.MARITIME_HAZE, cn2=5e-14, wind_speed_mps=7.5)
            .with_optical_payload(laser_power_mw=800.0, fsm_bandwidth_hz=1000.0)
            .with_custom_telemetry_channel("wave_height_m", unit="m", description="Instantaneous wave height")
            .build()
        )
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description
        self.platform = PlatformDynamicsConfig()
        self.atmosphere = AtmosphericChannelConfig()
        self.disturbances = DisturbanceProfileConfig()
        self.payload = OpticalPayloadConfig()
        self.telemetry_schema = TelemetrySchema.standard_fsoc_schema()

    def with_platform(
        self,
        altitude_m: float | None = None,
        sway_amplitude_deg: float | None = None,
        sway_freq_hz: float | None = None,
        jitter_frequencies_hz: Sequence[float] | None = None,
        jitter_amplitudes_urad: Sequence[float] | None = None,
        wind_gust_buffet_std_urad: float | None = None,
    ) -> VirtualEnvironmentBuilder:
        """Configure platform structural and dynamic motion properties."""
        if altitude_m is not None:
            self.platform.base_altitude_m = altitude_m
        if sway_amplitude_deg is not None:
            self.platform.sway_amplitude_deg = sway_amplitude_deg
        if sway_freq_hz is not None:
            self.platform.sway_frequency_hz = sway_freq_hz
        if jitter_frequencies_hz is not None:
            self.platform.jitter_frequencies_hz = list(jitter_frequencies_hz)
        if jitter_amplitudes_urad is not None:
            self.platform.jitter_amplitudes_urad = list(jitter_amplitudes_urad)
        if wind_gust_buffet_std_urad is not None:
            self.platform.wind_gust_buffet_std_urad = wind_gust_buffet_std_urad
        return self

    def with_atmosphere(
        self,
        model: AtmosphereModel | None = None,
        cn2: float | None = None,
        wind_speed_mps: float | None = None,
        extinction_coefficient_db_km: float | None = None,
        zenith_angle_deg: float | None = None,
        cloud_occlusion_prob: float | None = None,
        scintillation_index: float | None = None,
    ) -> VirtualEnvironmentBuilder:
        """Configure atmospheric propagation and optical turbulence parameters."""
        if model is not None:
            self.atmosphere.model = model
        if cn2 is not None:
            self.atmosphere.cn2 = cn2
        if wind_speed_mps is not None:
            self.atmosphere.wind_speed_mps = wind_speed_mps
        if extinction_coefficient_db_km is not None:
            self.atmosphere.extinction_coefficient_db_km = extinction_coefficient_db_km
        if zenith_angle_deg is not None:
            self.atmosphere.zenith_angle_deg = zenith_angle_deg
        if cloud_occlusion_prob is not None:
            self.atmosphere.cloud_occlusion_prob = cloud_occlusion_prob
        if scintillation_index is not None:
            self.atmosphere.scintillation_index = scintillation_index
        return self

    def with_disturbances(
        self,
        sensor_read_noise_std: float | None = None,
        background_clutter_count: int | None = None,
        solar_glint_probability: float | None = None,
        thermal_drift_rate_urad_s: float | None = None,
    ) -> VirtualEnvironmentBuilder:
        """Configure optical noise, background clutter, and thermal drift."""
        if sensor_read_noise_std is not None:
            self.disturbances.sensor_read_noise_std = sensor_read_noise_std
        if background_clutter_count is not None:
            self.disturbances.background_clutter_count = background_clutter_count
        if solar_glint_probability is not None:
            self.disturbances.solar_glint_probability = solar_glint_probability
        if thermal_drift_rate_urad_s is not None:
            self.disturbances.thermal_drift_rate_urad_s = thermal_drift_rate_urad_s
        return self

    def with_optical_payload(
        self,
        laser_wavelength_nm: float | None = None,
        laser_power_mw: float | None = None,
        beam_divergence_mrad: float | None = None,
        rx_aperture_diameter_m: float | None = None,
        fsm_max_deflection_urad: float | None = None,
        fsm_bandwidth_hz: float | None = None,
    ) -> VirtualEnvironmentBuilder:
        """Configure laser transceiver and fine-stage FSM optics."""
        if laser_wavelength_nm is not None:
            self.payload.laser_wavelength_nm = laser_wavelength_nm
        if laser_power_mw is not None:
            self.payload.laser_power_mw = laser_power_mw
        if beam_divergence_mrad is not None:
            self.payload.beam_divergence_mrad = beam_divergence_mrad
        if rx_aperture_diameter_m is not None:
            self.payload.rx_aperture_diameter_m = rx_aperture_diameter_m
            # Recompute diffraction limit: 1.22 * lambda / D
            lam_m = (self.payload.laser_wavelength_nm or 1550.0) * 1e-9
            self.payload.diffraction_limit_urad = 1.22 * lam_m / rx_aperture_diameter_m * 1e6
        if fsm_max_deflection_urad is not None:
            self.payload.fsm_max_deflection_urad = fsm_max_deflection_urad
        if fsm_bandwidth_hz is not None:
            self.payload.fsm_bandwidth_hz = fsm_bandwidth_hz
        return self

    def with_telemetry_schema(self, schema: TelemetrySchema) -> VirtualEnvironmentBuilder:
        """Replace the telemetry schema with a custom schema."""
        self.telemetry_schema = schema
        return self

    def with_custom_telemetry_channel(
        self,
        name: str,
        unit: str = "",
        data_type: TelemetryDataType = TelemetryDataType.FLOAT,
        description: str = "",
        decimation: int = 1,
        precision: int = 4,
        default_value: Any = 0.0,
    ) -> VirtualEnvironmentBuilder:
        """Add a specific custom telemetry channel to this environment."""
        self.telemetry_schema.add_channel(
            name=name,
            unit=unit,
            data_type=data_type,
            description=description,
            decimation=decimation,
            precision=precision,
            default_value=default_value,
        )
        return self

    def build(self) -> EnvironmentDefinition:
        """Build and validate the complete EnvironmentDefinition."""
        return EnvironmentDefinition(
            name=self.name,
            description=self.description,
            platform=self.platform,
            atmosphere=self.atmosphere,
            disturbances=self.disturbances,
            payload=self.payload,
            telemetry_schema=self.telemetry_schema,
        )
