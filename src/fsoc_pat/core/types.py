"""Domain value objects for the OptiLign simulation platform.

All types use SI units internally:
- Position: meters (m)
- Velocity: meters/second (m/s)
- Acceleration: meters/second² (m/s²)
- Angles: radians (rad)
- Angular velocity: radians/second (rad/s)
- Time: seconds (s)
- Pixels: pixels (px)

These are immutable data containers. They carry no business logic
beyond construction-time validation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Self

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Spatial types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Position3D:
    """3D position in meters.

    Attributes:
        x: X-coordinate (m).
        y: Y-coordinate (m).
        z: Z-coordinate (m).
    """

    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        for name in ("x", "y", "z"):
            _check_finite(getattr(self, name), f"Position3D.{name}")

    def as_array(self) -> NDArray[np.float64]:
        """Return as (3,) numpy array."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> Self:
        """Construct from a (3,) array."""
        if arr.shape != (3,):
            raise ValueError(f"Expected shape (3,), got {arr.shape}")
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))

    def distance_to(self, other: Position3D) -> float:
        """Euclidean distance to another position (m)."""
        return float(np.linalg.norm(self.as_array() - other.as_array()))

    def __add__(self, other: object) -> Position3D:
        if isinstance(other, (Velocity3D, Position3D)):
            return Position3D(self.x + other.x, self.y + other.y, self.z + other.z)
        return NotImplemented

    def __sub__(self, other: object) -> Position3D:
        if isinstance(other, Position3D):
            return Position3D(self.x - other.x, self.y - other.y, self.z - other.z)
        return NotImplemented


@dataclass(frozen=True, slots=True)
class Velocity3D:
    """3D velocity in meters/second.

    Attributes:
        x: X-component (m/s).
        y: Y-component (m/s).
        z: Z-component (m/s).
    """

    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        for name in ("x", "y", "z"):
            _check_finite(getattr(self, name), f"Velocity3D.{name}")

    def as_array(self) -> NDArray[np.float64]:
        """Return as (3,) numpy array."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> Self:
        """Construct from a (3,) array."""
        if arr.shape != (3,):
            raise ValueError(f"Expected shape (3,), got {arr.shape}")
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))

    @property
    def magnitude(self) -> float:
        """Speed magnitude (m/s)."""
        return float(np.linalg.norm(self.as_array()))

    def scaled(self, factor: float) -> Velocity3D:
        """Return velocity scaled by a factor (e.g. dt)."""
        return Velocity3D(self.x * factor, self.y * factor, self.z * factor)


@dataclass(frozen=True, slots=True)
class Acceleration3D:
    """3D acceleration in meters/second².

    Attributes:
        x: X-component (m/s²).
        y: Y-component (m/s²).
        z: Z-component (m/s²).
    """

    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        for name in ("x", "y", "z"):
            _check_finite(getattr(self, name), f"Acceleration3D.{name}")

    def as_array(self) -> NDArray[np.float64]:
        """Return as (3,) numpy array."""
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: NDArray[np.float64]) -> Self:
        """Construct from a (3,) array."""
        if arr.shape != (3,):
            raise ValueError(f"Expected shape (3,), got {arr.shape}")
        return cls(float(arr[0]), float(arr[1]), float(arr[2]))

    @property
    def magnitude(self) -> float:
        """Acceleration magnitude (m/s²)."""
        return float(np.linalg.norm(self.as_array()))


# ---------------------------------------------------------------------------
# Angular types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Orientation:
    """3D orientation as Euler angles in radians.

    Convention: ZYX intrinsic rotations (yaw → pitch → roll).

    Attributes:
        roll: Rotation about X-axis (rad).
        pitch: Rotation about Y-axis (rad).
        yaw: Rotation about Z-axis (rad).
    """

    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    def __post_init__(self) -> None:
        for name in ("roll", "pitch", "yaw"):
            _check_finite(getattr(self, name), f"Orientation.{name}")


@dataclass(frozen=True, slots=True)
class AngularState:
    """Pan-tilt gimbal state.

    Attributes:
        pan: Pan angle (rad). Positive = right.
        tilt: Tilt angle (rad). Positive = up.
        pan_rate: Pan angular velocity (rad/s).
        tilt_rate: Tilt angular velocity (rad/s).
    """

    pan: float = 0.0
    tilt: float = 0.0
    pan_rate: float = 0.0
    tilt_rate: float = 0.0

    def __post_init__(self) -> None:
        for name in ("pan", "tilt", "pan_rate", "tilt_rate"):
            _check_finite(getattr(self, name), f"AngularState.{name}")


@dataclass(frozen=True, slots=True)
class AngularError:
    """Angular pointing error in radians.

    Attributes:
        horizontal: Horizontal (azimuth) error (rad). Positive = target right of boresight.
        vertical: Vertical (elevation) error (rad). Positive = target above boresight.
        total: Total angular separation (rad). Always non-negative.
    """

    horizontal: float
    vertical: float
    total: float

    def __post_init__(self) -> None:
        for name in ("horizontal", "vertical", "total"):
            _check_finite(getattr(self, name), f"AngularError.{name}")
        if self.total < 0:
            raise ValueError(f"AngularError.total must be >= 0, got {self.total}")

    @property
    def horizontal_deg(self) -> float:
        """Horizontal error in degrees."""
        return math.degrees(self.horizontal)

    @property
    def vertical_deg(self) -> float:
        """Vertical error in degrees."""
        return math.degrees(self.vertical)

    @property
    def total_deg(self) -> float:
        """Total angular error in degrees."""
        return math.degrees(self.total)

    @property
    def total_arcsec(self) -> float:
        """Total angular error in arcseconds."""
        return math.degrees(self.total) * 3600.0


# ---------------------------------------------------------------------------
# Pixel types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PixelCoord:
    """Image-plane pixel coordinate.

    Attributes:
        u: Horizontal pixel coordinate (px). 0 = left edge.
        v: Vertical pixel coordinate (px). 0 = top edge.
    """

    u: float
    v: float

    def __post_init__(self) -> None:
        _check_finite(self.u, "PixelCoord.u")
        _check_finite(self.v, "PixelCoord.v")

    def as_array(self) -> NDArray[np.float64]:
        """Return as (2,) array."""
        return np.array([self.u, self.v], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class NormalizedCoord:
    """Normalized image coordinate (dimensionless).

    Origin at principal point, units = focal-length-normalized.

    Attributes:
        x: Horizontal normalized coordinate.
        y: Vertical normalized coordinate.
    """

    x: float
    y: float


# ---------------------------------------------------------------------------
# Time types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SimulationTime:
    """Simulation timestamp.

    Attributes:
        time_s: Simulation time in seconds.
        step: Discrete step index (0-based).
    """

    time_s: float
    step: int

    def __post_init__(self) -> None:
        _check_finite(self.time_s, "SimulationTime.time_s")
        if self.time_s < 0:
            raise ValueError(f"SimulationTime.time_s must be >= 0, got {self.time_s}")
        if self.step < 0:
            raise ValueError(f"SimulationTime.step must be >= 0, got {self.step}")


# ---------------------------------------------------------------------------
# Control types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ControlCommand:
    """Gimbal control command.

    Attributes:
        pan_rate: Commanded pan angular velocity (rad/s).
        tilt_rate: Commanded tilt angular velocity (rad/s).
        timestamp: Time at which the command was generated (s).
    """

    pan_rate: float
    tilt_rate: float
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        _check_finite(self.pan_rate, "ControlCommand.pan_rate")
        _check_finite(self.tilt_rate, "ControlCommand.tilt_rate")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DetectorMode(Enum):
    """Beacon detection mode."""

    CLASSICAL = auto()
    AI = auto()
    HYBRID = auto()


class EstimatorType(Enum):
    """State estimator type."""

    KF = auto()
    EKF = auto()
    UKF = auto()


class ControllerType(Enum):
    """Controller type."""

    PID = auto()
    ADAPTIVE_PID = auto()
    LQR = auto()
    MPC = auto()


class PATState(Enum):
    """PAT state machine states."""

    IDLE = auto()
    SEARCH = auto()
    ACQUISITION = auto()
    CONFIRMATION = auto()
    TRACKING = auto()
    PREDICTIVE_TRACKING = auto()
    DEGRADED_TRACKING = auto()
    BEACON_LOST = auto()
    REACQUISITION = auto()
    VERIFICATION = auto()
    FAILURE = auto()


class LinkState(Enum):
    """FSOC link state."""

    AVAILABLE = auto()
    DEGRADED = auto()
    LOST = auto()


class TrajectoryType(Enum):
    """Available trajectory types."""

    STATIONARY = auto()
    LINEAR = auto()
    CONSTANT_VELOCITY = auto()
    CONSTANT_ACCELERATION = auto()
    CIRCULAR = auto()
    SINUSOIDAL = auto()
    SPIRAL = auto()
    RANDOM = auto()
    PIECEWISE = auto()
    MANEUVER = auto()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_finite(value: float, name: str) -> None:
    """Raise ValueError if value is NaN or Inf."""
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
