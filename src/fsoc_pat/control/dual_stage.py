"""Dual-Stage Pointing Control: Coarse Gimbal + Fast Steering Mirror (FSM).

Implements aerospace dual-stage PAT architecture:
  - Stage 1 (Coarse): 2-DOF Gimbal with large travel range (+/- 180 deg pan, +/- 90 deg tilt),
    moderate bandwidth (10-50 Hz), high inertia.
  - Stage 2 (Fine): Piezo/voice-coil Fast Steering Mirror (FSM) with micro-radian
    precision, high bandwidth (500-1000 Hz), and fine travel range (+/- 3-5 mrad).
  - Offloading Loop: Continuous DC momentum offloading from FSM to Gimbal to prevent
    FSM actuator saturation during sustained target motion.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from fsoc_pat.control.pid import PIDConfig, PIDController
from fsoc_pat.core.types import AngularError, ControlCommand


@dataclass
class FSMConfig:
    """Fast Steering Mirror actuator configuration.

    Attributes:
        max_deflection_urad: Maximum mirror deflection in micro-radians (+/-).
        bandwidth_hz: Actuator closed-loop bandwidth (Hz).
        damping_ratio: Mechanical resonance damping factor.
        resolution_urad: Minimum achievable angular step (sub-micro-radian).
        offload_gain: Gain for offloading low-frequency deflection to coarse gimbal.
        offload_threshold_urad: Deflection threshold before offloading activates.
    """

    max_deflection_urad: float = 5000.0  # +/- 5 mrad
    bandwidth_hz: float = 800.0  # 800 Hz
    damping_ratio: float = 0.707
    resolution_urad: float = 0.05  # 50 nano-radians
    offload_gain: float = 2.5
    offload_threshold_urad: float = 300.0  # Start offload when FSM passes 300 urad


class FastSteeringMirror:
    """Piezoelectric / Voice-Coil Fast Steering Mirror (FSM) model.

    Provides ultra-fast, high-accuracy beam deflection to cancel residual
    jitter, atmospheric beam wander, and gimbal lag.
    """

    def __init__(self, config: FSMConfig | None = None) -> None:
        self.config = config or FSMConfig()
        self.pan_urad: float = 0.0
        self.tilt_urad: float = 0.0
        self.commanded_pan_urad: float = 0.0
        self.commanded_tilt_urad: float = 0.0

    def reset(self) -> None:
        """Reset FSM mirror deflection to optical center."""
        self.pan_urad = 0.0
        self.tilt_urad = 0.0
        self.commanded_pan_urad = 0.0
        self.commanded_tilt_urad = 0.0

    def step(self, cmd_pan_urad: float, cmd_tilt_urad: float, dt: float) -> tuple[float, float]:
        """Advance FSM dynamics by timestep dt.

        Args:
            cmd_pan_urad: Commanded pan deflection in micro-radians.
            cmd_tilt_urad: Commanded tilt deflection in micro-radians.
            dt: Timestep in seconds.

        Returns:
            Tuple of actual (pan_urad, tilt_urad).
        """
        # Clamp commanded deflections to physical mirror travel limits
        max_def = self.config.max_deflection_urad
        cmd_pan_clamped = max(-max_def, min(max_def, cmd_pan_urad))
        cmd_tilt_clamped = max(-max_def, min(max_def, cmd_tilt_urad))

        self.commanded_pan_urad = cmd_pan_clamped
        self.commanded_tilt_urad = cmd_tilt_clamped

        # First-order / high-bandwidth actuator response: dx/dt = omega * (cmd - x)
        omega = 2.0 * math.pi * self.config.bandwidth_hz
        alpha = 1.0 - math.exp(-omega * dt)

        self.pan_urad += alpha * (cmd_pan_clamped - self.pan_urad)
        self.tilt_urad += alpha * (cmd_tilt_clamped - self.tilt_urad)

        # Apply actuator resolution / quantization
        if self.config.resolution_urad > 0.0:
            res = self.config.resolution_urad
            self.pan_urad = round(self.pan_urad / res) * res
            self.tilt_urad = round(self.tilt_urad / res) * res

        return self.pan_urad, self.tilt_urad


@dataclass
class DualStagePointingState:
    """Instantaneous pointing telemetry for dual-stage control.

    Attributes:
        coarse_error_rad: Pointing error before FSM compensation (rad).
        coarse_error_urad: Coarse pointing error in micro-radians.
        fsm_pan_urad: FSM actual pan deflection (urad).
        fsm_tilt_urad: FSM actual tilt deflection (urad).
        fine_error_rad: Net optical pointing error after FSM correction (rad).
        fine_error_urad: Net optical pointing error in micro-radians.
        offload_active: True if FSM is currently offloading bias to coarse gimbal.
    """

    coarse_error_rad: float
    coarse_error_urad: float
    fsm_pan_urad: float
    fsm_tilt_urad: float
    fine_error_rad: float
    fine_error_urad: float
    offload_active: bool


class DualStageController:
    """Cooperating Dual-Stage Coarse Gimbal + Fine FSM Pointing Controller.

    - Gimbal loop tracks low-frequency target trajectory.
    - FSM loop cancels high-frequency residual error and platform jitter.
    - Offload loop desaturates FSM by steering gimbal towards FSM deflection center.
    """

    def __init__(
        self,
        gimbal_controller: PIDController | None = None,
        fsm_config: FSMConfig | None = None,
        fsm_p_gain: float = 1.0,
        fsm_i_gain: float = 0.05,
    ) -> None:
        self.gimbal_ctrl = gimbal_controller or PIDController(
            PIDConfig(kp=8.0, ki=2.0, kd=0.1, output_limit=2.0)
        )
        self.fsm = FastSteeringMirror(fsm_config)
        self.fsm_p_gain = fsm_p_gain
        self.fsm_i_gain = fsm_i_gain
        self._fsm_int_pan = 0.0
        self._fsm_int_tilt = 0.0

    def reset(self) -> None:
        """Reset both coarse and fine controllers."""
        self.gimbal_ctrl.reset()
        self.fsm.reset()
        self._fsm_int_pan = 0.0
        self._fsm_int_tilt = 0.0

    def compute(
        self,
        error: AngularError,
        dt: float,
    ) -> tuple[ControlCommand, DualStagePointingState]:
        """Compute coordinated coarse gimbal command and fine FSM deflection.

        Args:
            error: Estimated target angular error from boresight (rad).
            dt: Control timestep in seconds.

        Returns:
            Tuple of (coarse_gimbal_command, pointing_telemetry).
        """
        coarse_err_h_rad = error.horizontal
        coarse_err_v_rad = error.vertical
        coarse_total_rad = error.total
        coarse_total_urad = coarse_total_rad * 1e6

        # --- 1. Fine FSM Loop ---
        # The FSM needs to steer by the residual error to steer the beam onto target
        target_fsm_pan_urad = coarse_err_h_rad * 1e6 * self.fsm_p_gain
        target_fsm_tilt_urad = coarse_err_v_rad * 1e6 * self.fsm_p_gain

        # Integral action on residual fine error with anti-windup to prevent integral overshoot
        int_max = 500.0  # micro-radians max integral trim
        self._fsm_int_pan = max(-int_max, min(int_max, self._fsm_int_pan + (coarse_err_h_rad * 1e6 - self.fsm.pan_urad) * dt))
        self._fsm_int_tilt = max(-int_max, min(int_max, self._fsm_int_tilt + (coarse_err_v_rad * 1e6 - self.fsm.tilt_urad) * dt))
        cmd_fsm_pan = target_fsm_pan_urad + self.fsm_i_gain * self._fsm_int_pan
        cmd_fsm_tilt = target_fsm_tilt_urad + self.fsm_i_gain * self._fsm_int_tilt

        fsm_actual_pan, fsm_actual_tilt = self.fsm.step(cmd_fsm_pan, cmd_fsm_tilt, dt)

        # Net optical pointing error after FSM correction:
        # Fine error = Coarse error - FSM deflection (in radians)
        fine_err_h_rad = coarse_err_h_rad - (fsm_actual_pan * 1e-6)
        fine_err_v_rad = coarse_err_v_rad - (fsm_actual_tilt * 1e-6)
        fine_total_rad = math.sqrt(fine_err_h_rad**2 + fine_err_v_rad**2)
        fine_total_urad = fine_total_rad * 1e6

        # --- 2. FSM Momentum Offload to Coarse Gimbal ---
        # If FSM deflection is large, add an offload bias to the coarse gimbal
        # so the gimbal absorbs the DC offset and centers the FSM.
        # Implements progressive quadratic boost when approaching stroke limit (>50% stroke).
        offload_active = False
        offload_pan_rad_s = 0.0
        offload_tilt_rad_s = 0.0
        thresh = self.fsm.config.offload_threshold_urad
        max_stroke = self.fsm.config.max_deflection_urad
        warn_stroke = 0.5 * max_stroke

        if abs(fsm_actual_pan) > thresh:
            offload_active = True
            gain_pan = self.fsm.config.offload_gain
            if abs(fsm_actual_pan) > warn_stroke:
                norm_excess = min(1.0, (abs(fsm_actual_pan) - warn_stroke) / max(1.0, max_stroke - warn_stroke))
                gain_pan *= (1.0 + 3.0 * (norm_excess ** 2))
            offload_pan_rad_s = (fsm_actual_pan * 1e-6) * gain_pan

        if abs(fsm_actual_tilt) > thresh:
            offload_active = True
            gain_tilt = self.fsm.config.offload_gain
            if abs(fsm_actual_tilt) > warn_stroke:
                norm_excess = min(1.0, (abs(fsm_actual_tilt) - warn_stroke) / max(1.0, max_stroke - warn_stroke))
                gain_tilt *= (1.0 + 3.0 * (norm_excess ** 2))
            offload_tilt_rad_s = (fsm_actual_tilt * 1e-6) * gain_tilt

        # --- 3. Coarse Gimbal Command ---
        gimbal_cmd = self.gimbal_ctrl.compute(error, dt)
        coordinated_gimbal_cmd = ControlCommand(
            pan_rate=gimbal_cmd.pan_rate + offload_pan_rad_s,
            tilt_rate=gimbal_cmd.tilt_rate + offload_tilt_rad_s,
        )

        state = DualStagePointingState(
            coarse_error_rad=coarse_total_rad,
            coarse_error_urad=coarse_total_urad,
            fsm_pan_urad=fsm_actual_pan,
            fsm_tilt_urad=fsm_actual_tilt,
            fine_error_rad=fine_total_rad,
            fine_error_urad=fine_total_urad,
            offload_active=offload_active,
        )

        return coordinated_gimbal_cmd, state

    def update(
        self,
        error: AngularError,
        dt: float,
    ) -> tuple[ControlCommand, DualStagePointingState]:
        """Alias for compute()."""
        return self.compute(error, dt)
