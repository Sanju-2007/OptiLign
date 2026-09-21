"""Supervisory Pointing, Acquisition, and Tracking (PAT) State Machine.

Coordinates state transitions between SEARCH, ACQUISITION, TRACKING,
and REACQUISITION, preventing gimbal runaway and orchestrating re-lock.
"""

from __future__ import annotations

import math
from typing import Any

from fsoc_pat.core.types import AngularError, ControlCommand
from fsoc_pat.pat.pat_types import PATConfig, PATState, PATStateData
from fsoc_pat.pat.search_patterns import SearchPatternProtocol, SpiralSearchPattern


class PATStateMachine:
    """Supervisory state machine for FSOC PAT operations.

    Controls transition between wide search, centering acquisition,
    fine tracking, and spiral reacquisition upon signal dropout.
    """

    def __init__(
        self,
        config: PATConfig | None = None,
        search_pattern: SearchPatternProtocol | None = None,
    ) -> None:
        self.config = config or PATConfig()
        self.search_pattern = search_pattern or SpiralSearchPattern(
            step_rad=self.config.spiral_step_rad,
            angular_speed=self.config.spiral_angular_speed,
            max_radius_rad=self.config.spiral_max_radius_rad,
        )

        self.data = PATStateData()
        self._consecutive_detections = 0
        self._consecutive_misses = 0

    @property
    def state(self) -> PATState:
        """Current operational state."""
        return self.data.state

    def set_state(self, new_state: PATState, reason: str = "") -> None:
        """Force or trigger state transition."""
        if self.data.state != new_state:
            self.data.state = new_state
            self.data.state_time = 0.0
            self.data.status_message = reason or f"Entered {new_state.name}"

            if new_state in (PATState.REACQUISITION, PATState.ML_SEARCH):
                # Reset search pattern centered at last known position or 0
                self.search_pattern.reset(self.data.last_known_pan, self.data.last_known_tilt)
            elif new_state == PATState.SEARCH:
                self.search_pattern.reset(0.0, 0.0)

    def step(
        self,
        dt: float,
        detected: bool,
        angular_error: AngularError | None,
        gimbal_pan: float,
        gimbal_tilt: float,
        pid_cmd: ControlCommand | None = None,
        timestamp: float = 0.0,
    ) -> tuple[float, float]:
        """Execute one state machine step and return commanded (pan_rate, tilt_rate).

        Args:
            dt: Timestep (s).
            detected: Whether the beacon was detected in the current frame.
            angular_error: Current estimated angular error (if target is tracked).
            gimbal_pan: Current gimbal pan angle (rad).
            gimbal_tilt: Current gimbal tilt angle (rad).
            pid_cmd: Control command computed by the closed-loop PID controller.
            timestamp: Simulation time (s).

        Returns:
            Tuple of (pan_rate_cmd, tilt_rate_cmd) in rad/s.
        """
        self.data.state_time += dt

        # Update detection history counters
        if detected:
            self._consecutive_detections += 1
            self._consecutive_misses = 0
            self.data.last_known_pan = gimbal_pan
            self.data.last_known_tilt = gimbal_tilt
            self.data.last_known_time = timestamp
        else:
            self._consecutive_misses += 1
            self._consecutive_detections = 0

        self.data.consecutive_detections = self._consecutive_detections
        self.data.consecutive_misses = self._consecutive_misses

        # Execute state-dependent transition logic & action
        if self.data.state == PATState.STANDBY:
            return self._handle_standby(detected)

        elif self.data.state == PATState.SEARCH:
            return self._handle_search(dt, detected, pid_cmd)

        elif self.data.state == PATState.ACQUISITION:
            return self._handle_acquisition(dt, detected, angular_error, pid_cmd)

        elif self.data.state == PATState.TRACKING:
            return self._handle_tracking(dt, detected, pid_cmd)

        elif self.data.state == PATState.REACQUISITION:
            return self._handle_reacquisition(dt, detected, pid_cmd)

        elif self.data.state == PATState.ML_SEARCH:
            return self._handle_ml_search(dt, detected, pid_cmd)

        elif self.data.state == PATState.FAULT:
            return 0.0, 0.0

        return 0.0, 0.0

    def _handle_standby(self, detected: bool) -> tuple[float, float]:
        if detected and self._consecutive_detections >= 1:
            self.set_state(PATState.ACQUISITION, "Target detected in standby")
        return 0.0, 0.0

    def _handle_search(
        self,
        dt: float,
        detected: bool,
        pid_cmd: ControlCommand | None = None,
    ) -> tuple[float, float]:
        self.data.search_active = True
        if detected:
            if self._consecutive_detections >= self.config.confirm_detections_threshold:
                self.set_state(PATState.ACQUISITION, "Target detected in search")
                if pid_cmd is not None:
                    return pid_cmd.pan_rate, pid_cmd.tilt_rate
            return 0.0, 0.0

        # Loop search pattern if completed so search scan continues
        if self.search_pattern.is_finished:
            self.search_pattern.reset(0.0, 0.0)

        # Continue search scan
        return self.search_pattern.step(dt)

    def _handle_acquisition(
        self,
        dt: float,
        detected: bool,
        angular_error: AngularError | None,
        pid_cmd: ControlCommand | None,
    ) -> tuple[float, float]:
        self.data.search_active = False

        if self._consecutive_misses >= self.config.loss_misses_threshold:
            self.set_state(PATState.REACQUISITION, "Lost target during acquisition")
            return 0.0, 0.0

        # Check if error is small enough to transition to TRACKING
        if detected and angular_error is not None:
            if angular_error.total <= self.config.acquisition_lock_error_rad:
                self.set_state(PATState.TRACKING, "Target centered — track locked")

        # Drive with PID
        if pid_cmd is not None and detected:
            return pid_cmd.pan_rate, pid_cmd.tilt_rate
        return 0.0, 0.0

    def _handle_tracking(
        self,
        dt: float,
        detected: bool,
        pid_cmd: ControlCommand | None,
    ) -> tuple[float, float]:
        self.data.search_active = False

        # Loss of track check
        if self._consecutive_misses >= self.config.loss_misses_threshold:
            self.set_state(PATState.REACQUISITION, f"Target lost for {self._consecutive_misses} frames")
            return 0.0, 0.0

        # If temporarily missed frames during beacon strobe off or brief cloud occlusion,
        # use motion model / EKF coasting feedforward if available
        if not detected:
            if pid_cmd is not None:
                return pid_cmd.pan_rate, pid_cmd.tilt_rate
            return 0.0, 0.0

        if pid_cmd is None:
            return 0.0, 0.0

        # Closed-loop tracking
        return pid_cmd.pan_rate, pid_cmd.tilt_rate

    def _handle_reacquisition(
        self,
        dt: float,
        detected: bool,
        pid_cmd: ControlCommand | None = None,
    ) -> tuple[float, float]:
        # If beacon re-detected and confirmed, transition back to TRACKING
        if detected:
            if self._consecutive_detections >= self.config.confirm_detections_threshold:
                self.set_state(PATState.TRACKING, "Target reacquired — track restored")
                if pid_cmd is not None:
                    return pid_cmd.pan_rate, pid_cmd.tilt_rate
            return 0.0, 0.0

        # Check for reacquisition timeout
        if self.data.state_time >= self.config.reacquisition_timeout_s:
            self.set_state(PATState.SEARCH, "Reacquisition timed out — resetting to wide search")
            return 0.0, 0.0

        # Coasting phase before spiral scan (applies smooth rate decay if feedforward command exists)
        if self.data.state_time < self.config.coast_time_s:
            self.data.search_active = False
            if pid_cmd is not None:
                decay = max(0.0, 1.0 - (self.data.state_time / max(0.001, self.config.coast_time_s)))
                return pid_cmd.pan_rate * decay, pid_cmd.tilt_rate * decay
            return 0.0, 0.0

        # Active spiral search around last-known position
        self.data.search_active = True
        if self.search_pattern.is_finished:
            self.search_pattern.reset(self.data.last_known_pan, self.data.last_known_tilt)
        return self.search_pattern.step(dt)

    def _handle_ml_search(
        self,
        dt: float,
        detected: bool,
        pid_cmd: ControlCommand | None = None,
    ) -> tuple[float, float]:
        """Omnidirectional all-degrees search powered by ML beacon signature recognition."""
        self.data.search_active = True

        # When beacon signature is verified and detected, immediately lock track
        if detected:
            self.set_state(PATState.TRACKING, "Target reacquired via ML beacon signature")
            if pid_cmd is not None:
                return pid_cmd.pan_rate, pid_cmd.tilt_rate
            return 0.0, 0.0

        # Step omnidirectional all-degrees search pattern
        return self.search_pattern.step(dt)

