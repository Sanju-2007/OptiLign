"""Virtual Environment Data Models and Specifications.

Provides rich engineering representations of diverse operating environments
ranging from deep space inter-satellite links to airborne UAVs and urban rooftops.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from fsoc_pat.telemetry.channel import TelemetrySchema


class PlatformType(str, Enum):
    """Type of host platform carrying the FSOC terminal."""
    GROUND_ROOFTOP = "ground_rooftop"
    GROUND_OPTICAL_STATION = "ground_optical_station"
    AIRBORNE_UAV = "airborne_uav"
    AIRBORNE_AIRCRAFT = "airborne_aircraft"
    SATELLITE_LEO = "satellite_leo"
    SATELLITE_GEO = "satellite_geo"
    MARITIME_VESSEL = "maritime_vessel"


class AtmosphereModel(str, Enum):
    """Atmospheric channel propagation model."""
    VACUUM_SPACE = "vacuum_space"
    CLEAR_SKY = "clear_sky"
    URBAN_TURBULENCE = "urban_turbulence"
    MARITIME_HAZE = "maritime_haze"
    DENSE_FOG = "dense_fog"
    STRATOSPHERE = "stratosphere"


@dataclass
class PlatformDynamicsConfig:
    """Dynamic motion and structural vibration of the host platform.

    Attributes:
        base_altitude_m: Base elevation above sea level.
        sway_amplitude_deg: Low-frequency platform sway/roll amplitude.
        sway_frequency_hz: Dominant sway frequency (e.g., ship roll 0.1-0.3 Hz, building 0.2 Hz).
        jitter_frequencies_hz: Discrete structural vibration harmonics (e.g., reaction wheels, motors).
        jitter_amplitudes_urad: Amplitudes of vibration harmonics in micro-radians.
        wind_gust_buffet_std_urad: Random aerodynamic buffet jitter standard deviation.
    """

    base_altitude_m: float = 20.0
    sway_amplitude_deg: float = 0.05
    sway_frequency_hz: float = 0.2
    jitter_frequencies_hz: list[float] = field(default_factory=lambda: [25.0, 50.0, 120.0])
    jitter_amplitudes_urad: list[float] = field(default_factory=lambda: [15.0, 8.0, 3.0])
    wind_gust_buffet_std_urad: float = 5.0


@dataclass
class AtmosphericChannelConfig:
    """Free-space optical atmospheric channel characteristics.

    Attributes:
        model: Macro atmospheric regime.
        cn2: Refractive index structure constant (m^(-2/3)), measuring optical turbulence.
        wind_speed_mps: Crosswind velocity driving Greenwood frequency.
        extinction_coefficient_db_km: Specific atmospheric attenuation (dB/km).
        zenith_angle_deg: Angle from zenith for slant-path propagation.
        cloud_occlusion_prob: Probability of cloud/smoke blockage events per second.
        scintillation_index: Normalized intensity variance sigma_I^2.
    """

    model: AtmosphereModel = AtmosphereModel.CLEAR_SKY
    cn2: float = 1e-15  # Nominal ground/boundary layer turbulence
    wind_speed_mps: float = 3.0
    extinction_coefficient_db_km: float = 0.5  # Clear day at 1550nm
    zenith_angle_deg: float = 45.0
    cloud_occlusion_prob: float = 0.02
    scintillation_index: float = 0.05


@dataclass
class DisturbanceProfileConfig:
    """Sensor, environmental, and background optical noise disturbances.

    Attributes:
        sensor_read_noise_std: Camera/FPA read noise in digital counts.
        background_clutter_count: Number of false celestial or city light glints.
        solar_glint_probability: Probability of strong specular solar reflections.
        thermal_drift_rate_urad_s: Boresight thermal drift rate.
    """

    sensor_read_noise_std: float = 4.0
    background_clutter_count: int = 15
    solar_glint_probability: float = 0.05
    thermal_drift_rate_urad_s: float = 0.5


@dataclass
class OpticalPayloadConfig:
    """FSOC transceiver optical architecture and parameters.

    Attributes:
        laser_wavelength_nm: Laser operating wavelength (e.g. 1550.0 nm or 850.0 nm).
        laser_power_mw: Beacon emitter average optical power.
        beam_divergence_mrad: Beacon 1/e^2 half-angle divergence.
        rx_aperture_diameter_m: Telescope receiver clear aperture diameter.
        fsm_max_deflection_urad: Fast Steering Mirror peak angular deflection range.
        fsm_bandwidth_hz: FSM closed-loop actuation bandwidth.
        diffraction_limit_urad: Telescope diffraction limit lambda / D (computed or overridden).
    """

    laser_wavelength_nm: float = 1550.0
    laser_power_mw: float = 500.0
    beam_divergence_mrad: float = 1.2
    rx_aperture_diameter_m: float = 0.15
    fsm_max_deflection_urad: float = 5000.0  # +/- 5 mrad
    fsm_bandwidth_hz: float = 800.0  # 800 Hz fine-stage response
    diffraction_limit_urad: float = 12.6  # 1.22 * 1550nm / 0.15m ~= 12.6 urad


@dataclass
class EnvironmentDefinition:
    """Complete, self-contained definition of a virtual simulation environment.

    Attributes:
        name: Name of the environment.
        description: Description of mission conditions.
        platform: Host platform dynamic configuration.
        atmosphere: Atmospheric channel parameters.
        disturbances: Noise, clutter, and thermal drift parameters.
        payload: Optical transceiver and fine-stage hardware parameters.
        telemetry_schema: Schema describing which channels to capture.
    """

    name: str
    description: str = ""
    platform: PlatformDynamicsConfig = field(default_factory=PlatformDynamicsConfig)
    atmosphere: AtmosphericChannelConfig = field(default_factory=AtmosphericChannelConfig)
    disturbances: DisturbanceProfileConfig = field(default_factory=DisturbanceProfileConfig)
    payload: OpticalPayloadConfig = field(default_factory=OpticalPayloadConfig)
    telemetry_schema: TelemetrySchema = field(default_factory=TelemetrySchema.standard_fsoc_schema)

    def to_dict(self) -> dict[str, Any]:
        """Convert environment definition to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "platform": asdict(self.platform),
            "atmosphere": {
                **asdict(self.atmosphere),
                "model": self.atmosphere.model.value,
            },
            "disturbances": asdict(self.disturbances),
            "payload": asdict(self.payload),
            "telemetry_channels": list(self.telemetry_schema.channels.keys()),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize environment to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, filepath: str | Path) -> None:
        """Save environment definition to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())

    to_json_file = save_json

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvironmentDefinition:
        """Construct environment from dictionary."""
        plat_data = data.get("platform", {})
        platform = PlatformDynamicsConfig(**plat_data) if plat_data else PlatformDynamicsConfig()

        atmos_data = data.get("atmosphere", {})
        if atmos_data and "model" in atmos_data:
            atmos_data["model"] = AtmosphereModel(atmos_data["model"])
            atmosphere = AtmosphericChannelConfig(**atmos_data)
        else:
            atmosphere = AtmosphericChannelConfig()

        dist_data = data.get("disturbances", {})
        disturbances = DisturbanceProfileConfig(**dist_data) if dist_data else DisturbanceProfileConfig()

        pay_data = data.get("payload", {})
        payload = OpticalPayloadConfig(**pay_data) if pay_data else OpticalPayloadConfig()

        # Telemetry schema
        telemetry_schema = TelemetrySchema.standard_fsoc_schema()
        custom_chans = data.get("telemetry_channels", [])
        for ch in custom_chans:
            if ch not in telemetry_schema.channels:
                telemetry_schema.add_channel(ch)

        return cls(
            name=data.get("name", "Custom_Environment"),
            description=data.get("description", ""),
            platform=platform,
            atmosphere=atmosphere,
            disturbances=disturbances,
            payload=payload,
            telemetry_schema=telemetry_schema,
        )

    @classmethod
    def load_json(cls, filepath: str | Path) -> EnvironmentDefinition:
        """Load environment definition from a JSON file."""
        with open(filepath) as f:
            data = json.load(f)
        return cls.from_dict(data)

    from_json_file = load_json
