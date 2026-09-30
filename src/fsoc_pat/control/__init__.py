"""Control package — PID, LQR, Dual-Stage, and Tri-Stage PAT controllers."""

from fsoc_pat.control.closed_loop import (
    ClosedLoopConfig,
    ClosedLoopEngine,
    ClosedLoopResult,
    ClosedLoopSnapshot,
)
from fsoc_pat.control.dual_stage import (
    DualStageController,
    DualStagePointingState,
    FSMConfig,
    FastSteeringMirror,
)
from fsoc_pat.control.lqr import LQRConfig, LQRController
from fsoc_pat.control.pid import PIDConfig, PIDController
from fsoc_pat.control.tri_stage import TriStageController, TriStagePointingState

__all__ = [
    "PIDConfig",
    "PIDController",
    "LQRConfig",
    "LQRController",
    "FSMConfig",
    "FastSteeringMirror",
    "DualStageController",
    "DualStagePointingState",
    "TriStageController",
    "TriStagePointingState",
    "ClosedLoopConfig",
    "ClosedLoopEngine",
    "ClosedLoopResult",
    "ClosedLoopSnapshot",
]
