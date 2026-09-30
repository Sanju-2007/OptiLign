"""Tri-Stage Pointing & Wavefront Correction Architecture.

Integrates:
  - Stage 1: Coarse Gimbal (mechanical, large angular range, moderate bandwidth)
  - Stage 2: Fast Steering Mirror (FSM) (piezo tip-tilt, sub-micro-radian precision, high bandwidth)
  - Stage 3: Adaptive Optics (AO) Deformable Mirror (higher-order Zernike wavefront correction,
             recovering Strehl ratio and Single-Mode Fiber coupling through deep turbulence)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from fsoc_pat.ao.ao_system import AdaptiveOpticsSystem
from fsoc_pat.ao.ao_types import AOConfig, AOTelemetry
from fsoc_pat.control.dual_stage import DualStageController, DualStagePointingState, FSMConfig
from fsoc_pat.control.pid import PIDController
from fsoc_pat.core.types import AngularError, ControlCommand


@dataclass
class TriStagePointingState:
    """Instantaneous telemetry for the complete tri-stage pointing & wavefront system.

    Attributes:
        coarse_error_rad: Pointing error before FSM/AO (rad).
        coarse_error_urad: Coarse pointing error (urad).
        fsm_pan_urad: FSM pan deflection (urad).
        fsm_tilt_urad: FSM tilt deflection (urad).
        fine_error_rad: Net optical pointing error after FSM tip-tilt correction (rad).
        fine_error_urad: Net optical pointing error (urad).
        offload_active: Whether FSM momentum offloading to coarse gimbal is engaged.
        strehl_ratio: Instantaneous Strehl ratio after AO wavefront correction (0 to 1).
        smf_coupling: Single-mode fiber coupling efficiency (0 to 1).
        residual_wfe_nm: Residual wavefront error (nm RMS).
        ao_gain_db: Optical throughput gain provided by AO over uncorrected link (dB).
        ao_telemetry: Full AOTelemetry record if AO is active.
    """

    coarse_error_rad: float
    coarse_error_urad: float
    fsm_pan_urad: float
    fsm_tilt_urad: float
    fine_error_rad: float
    fine_error_urad: float
    offload_active: bool
    strehl_ratio: float
    smf_coupling: float
    residual_wfe_nm: float
    ao_gain_db: float
    ao_telemetry: AOTelemetry | None = None


class TriStageController:
    """Master Tri-Stage Pointing & Wavefront Controller (Gimbal + FSM + AO).

    Coordinates:
      1. Coarse gimbal for line-of-sight tracking within camera FOV
      2. Fine FSM for sub-micro-radian tip-tilt jitter rejection
      3. Adaptive Optics Deformable Mirror for higher-order phase flattening
    """

    def __init__(
        self,
        dual_stage: DualStageController | None = None,
        ao_system: AdaptiveOpticsSystem | None = None,
        ao_config: AOConfig | None = None,
    ) -> None:
        self.dual_stage = dual_stage or DualStageController()
        self.ao = ao_system or AdaptiveOpticsSystem(ao_config or AOConfig())

    def reset(self) -> None:
        """Reset all three control stages."""
        self.dual_stage.reset()
        self.ao.reset()

    def update(
        self,
        error: AngularError,
        dt: float,
        turbulent_phase_rad: NDArray[np.float64] | None = None,
        timestamp_s: float = 0.0,
    ) -> tuple[ControlCommand, TriStagePointingState]:
        """Advance all 3 pointing & wavefront stages.

        Args:
            error: Angular error from boresight (rad).
            dt: Timestep (s).
            turbulent_phase_rad: Optional 2D incoming turbulent phase map (rad).
                                 If None, zero phase is assumed.
            timestamp_s: Simulation timestamp (s).

        Returns:
            Tuple of (coarse_gimbal_command, tri_stage_telemetry).
        """
        # 1 & 2: Dual-Stage Coarse Gimbal + FSM Tip-Tilt
        gimbal_cmd, ds_state = self.dual_stage.compute(error, dt)

        # 3: Stage 3 Adaptive Optics Deformable Mirror Correction
        ao_telem: AOTelemetry | None = None
        if turbulent_phase_rad is not None:
            ao_telem = self.ao.step(turbulent_phase_rad, dt, timestamp_s)
            strehl = ao_telem.strehl_ratio_post
            coupling = ao_telem.smf_coupling_post
            wfe_nm = ao_telem.corrected_wfe_rms_nm
            gain_db = ao_telem.ao_optical_gain_db
        else:
            strehl = 1.0
            coupling = 0.8145
            wfe_nm = 0.0
            gain_db = 0.0

        tri_state = TriStagePointingState(
            coarse_error_rad=ds_state.coarse_error_rad,
            coarse_error_urad=ds_state.coarse_error_urad,
            fsm_pan_urad=ds_state.fsm_pan_urad,
            fsm_tilt_urad=ds_state.fsm_tilt_urad,
            fine_error_rad=ds_state.fine_error_rad,
            fine_error_urad=ds_state.fine_error_urad,
            offload_active=ds_state.offload_active,
            strehl_ratio=strehl,
            smf_coupling=coupling,
            residual_wfe_nm=wfe_nm,
            ao_gain_db=gain_db,
            ao_telemetry=ao_telem,
        )

        return gimbal_cmd, tri_state
