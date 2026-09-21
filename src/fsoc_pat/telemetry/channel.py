"""Custom Telemetry Channel and Schema Definitions.

Allows users to define, configure, and dynamically extract custom telemetry
channels during simulation runs across arbitrary virtual environments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


class TelemetryDataType(str, Enum):
    """Supported data types for telemetry channels."""
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    STRING = "string"
    VECTOR2 = "vector2"
    VECTOR3 = "vector3"


@dataclass
class TelemetryChannel:
    """Definition of a single telemetry channel.

    Attributes:
        name: Unique channel identifier (e.g. "target_los_az_rad", "fsm_deflection_urad").
        unit: Physical engineering unit (e.g. "urad", "m/s", "dBm", "m").
        data_type: Channel data type.
        description: Human-readable documentation for reports and GUI.
        decimation: Decimation factor (1 = log every tick, 2 = every 2nd tick).
        default_value: Value logged when the channel is undefined or missing in a step.
        precision: Decimal precision for numeric formatting in text outputs.
    """

    name: str
    unit: str = ""
    data_type: TelemetryDataType = TelemetryDataType.FLOAT
    description: str = ""
    decimation: int = 1
    default_value: Any = 0.0
    precision: int = 6

    def format_value(self, value: Any) -> str:
        """Format value according to channel specification."""
        if value is None:
            return ""
        if self.data_type == TelemetryDataType.FLOAT:
            try:
                return f"{float(value):.{self.precision}f}"
            except (ValueError, TypeError):
                return str(value)
        elif self.data_type == TelemetryDataType.INT:
            try:
                return str(int(value))
            except (ValueError, TypeError):
                return str(value)
        elif self.data_type == TelemetryDataType.BOOL:
            return "1" if bool(value) else "0"
        elif self.data_type in (TelemetryDataType.VECTOR2, TelemetryDataType.VECTOR3):
            if isinstance(value, (list, tuple)):
                return "[" + ", ".join(f"{float(x):.{self.precision}f}" for x in value) + "]"
        return str(value)


@dataclass
class TelemetrySchema:
    """User-configured telemetry schema defining all active channels.

    Attributes:
        channels: Dictionary of registered channels keyed by name.
        sampling_frequency_hz: Nominal sample rate.
    """

    channels: dict[str, TelemetryChannel] = field(default_factory=dict)
    sampling_frequency_hz: float = 50.0

    def add_channel(
        self,
        name: str,
        unit: str = "",
        data_type: TelemetryDataType = TelemetryDataType.FLOAT,
        description: str = "",
        decimation: int = 1,
        default_value: Any = 0.0,
        precision: int = 6,
    ) -> TelemetryChannel:
        """Register a new channel in the schema."""
        channel = TelemetryChannel(
            name=name,
            unit=unit,
            data_type=data_type,
            description=description,
            decimation=max(1, decimation),
            default_value=default_value,
            precision=precision,
        )
        self.channels[name] = channel
        return channel

    def get_channel(self, name: str) -> TelemetryChannel | None:
        """Retrieve channel definition by name."""
        return self.channels.get(name)

    @classmethod
    def standard_fsoc_schema(cls, sampling_frequency_hz: float = 50.0) -> TelemetrySchema:
        """Create a comprehensive standard FSOC coarse + fine pointing schema."""
        schema = cls(sampling_frequency_hz=sampling_frequency_hz)

        # Time and frame
        schema.add_channel("time_s", unit="s", description="Simulation elapsed time", precision=4)
        schema.add_channel("step", data_type=TelemetryDataType.INT, description="Simulation step counter")

        # Ground truth kinematics
        schema.add_channel("target_range_m", unit="m", description="Target 3D slant range", precision=2)
        schema.add_channel("target_azimuth_rad", unit="rad", description="Target true azimuth in world frame")
        schema.add_channel("target_elevation_rad", unit="rad", description="Target true elevation in world frame")

        # Coarse Gimbal
        schema.add_channel("gimbal_pan_rad", unit="rad", description="Gimbal actual pan angle")
        schema.add_channel("gimbal_tilt_rad", unit="rad", description="Gimbal actual tilt angle")
        schema.add_channel("gimbal_rate_pan_rad_s", unit="rad/s", description="Gimbal pan angular velocity")
        schema.add_channel("gimbal_rate_tilt_rad_s", unit="rad/s", description="Gimbal tilt angular velocity")

        # Fine Steering Mirror (FSM)
        schema.add_channel("fsm_pan_urad", unit="urad", description="FSM active pan deflection", precision=3)
        schema.add_channel("fsm_tilt_urad", unit="urad", description="FSM active tilt deflection", precision=3)
        schema.add_channel("fsm_offload_active", data_type=TelemetryDataType.BOOL, description="FSM offloading to gimbal active")

        # Tracking Errors
        schema.add_channel("coarse_error_urad", unit="urad", description="Coarse gimbal residual pointing error", precision=2)
        schema.add_channel("fine_error_urad", unit="urad", description="Total fine pointing error after FSM correction", precision=3)
        schema.add_channel("in_fine_lock", data_type=TelemetryDataType.BOOL, description="Fine alignment within optical diffraction limit")

        # Perception & Estimation
        schema.add_channel("beacon_detected", data_type=TelemetryDataType.BOOL, description="Beacon detected by sensor")
        schema.add_channel("detection_confidence", unit="0-1", description="Perception detection confidence score", precision=4)
        schema.add_channel("est_pixel_u", unit="px", description="Estimated beacon center pixel U", precision=2)
        schema.add_channel("est_pixel_v", unit="px", description="Estimated beacon center pixel V", precision=2)

        # FSOC Link Budget & Optical Physics
        schema.add_channel("received_power_dbm", unit="dBm", description="Optical received power at detector", precision=2)
        schema.add_channel("link_snr_db", unit="dB", description="Optical communication link SNR", precision=2)
        schema.add_channel("bit_error_rate", description="Estimated communication Bit Error Rate (BER)", precision=8)
        schema.add_channel("pat_state", data_type=TelemetryDataType.STRING, description="Supervisory PAT state machine mode")

        return schema
