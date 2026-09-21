"""PAT State Machine data types and configurations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class PATState(Enum):
    """Pointing, Acquisition, and Tracking operational states."""

    STANDBY = auto()        # Gimbal held stationary / awaiting command
    SEARCH = auto()         # Scanning wide uncertainty area for beacon
    ACQUISITION = auto()    # Beacon found; centering into fine FOV
    TRACKING = auto()       # Closed-loop tracking locked on beacon
    REACQUISITION = auto()  # Beacon temporarily lost; coasting + spiral search
    ML_SEARCH = auto()      # Omnidirectional search across all degrees with ML beacon signature classification
    FAULT = auto()          # Safety limit hit or hardware fault


@dataclass
class PATStateData:
    """Snapshot of active PAT state machine telemetry.

    Attributes:
        state: Active operational state.
        state_time: Elapsed time in current state (s).
        consecutive_detections: Count of consecutive frames target was detected.
        consecutive_misses: Count of consecutive frames target was missed.
        last_known_pan: Gimbal pan angle (rad) at last valid detection.
        last_known_tilt: Gimbal tilt angle (rad) at last valid detection.
        last_known_time: Simulation timestamp (s) of last valid detection.
        search_active: Whether a search pattern is currently generating commands.
        status_message: Human-readable status description.
    """

    state: PATState = PATState.STANDBY
    state_time: float = 0.0
    consecutive_detections: int = 0
    consecutive_misses: int = 0
    last_known_pan: float = 0.0
    last_known_tilt: float = 0.0
    last_known_time: float = -1.0
    search_active: bool = False
    status_message: str = "Initialized"


@dataclass
class PATConfig:
    """Supervisory parameters for the PAT state machine.

    Attributes:
        confirm_detections_threshold: Consecutive detections required to lock track.
        loss_misses_threshold: Consecutive misses to trigger reacquisition.
        coast_time_s: Duration to coast on Kalman filter before active spiral search.
        reacquisition_timeout_s: Max time in reacquisition before falling back to wide SEARCH.
        acquisition_lock_error_rad: Angular error threshold to transition ACQUISITION -> TRACKING.
        spiral_step_rad: Radial spacing between successive spiral turns (rad).
        spiral_angular_speed: Angular rotation rate along spiral trajectory (rad/s).
        spiral_max_radius_rad: Maximum radius of the spiral search pattern (rad).
        raster_pan_span_rad: Total horizontal scan width for raster search (rad).
        raster_tilt_span_rad: Total vertical scan height for raster search (rad).
        raster_step_rad: Vertical line step for raster search (rad).
    """

    confirm_detections_threshold: int = 2
    loss_misses_threshold: int = 4
    coast_time_s: float = 0.5
    reacquisition_timeout_s: float = 8.0
    acquisition_lock_error_rad: float = 0.05
    spiral_step_rad: float = 0.08
    spiral_angular_speed: float = 0.8
    spiral_max_radius_rad: float = 0.45
    raster_pan_span_rad: float = 0.6
    raster_tilt_span_rad: float = 0.4
    raster_step_rad: float = 0.06
