"""OptiLign Pointing, Acquisition, and Tracking (PAT) supervisory control.

Provides:
  - PAT State Machine (STANDBY, SEARCH, ACQUISITION, TRACKING, REACQUISITION)
  - Search and reacquisition patterns (Archimedean spiral, raster scan)
  - Coasting and anti-windup supervisory guards
"""

from fsoc_pat.pat.pat_types import PATConfig, PATState, PATStateData
from fsoc_pat.pat.pat_state_machine import PATStateMachine
from fsoc_pat.pat.search_patterns import (
    OmniSearchPattern,
    RasterSearchPattern,
    SearchPatternProtocol,
    SpiralSearchPattern,
)

__all__ = [
    "OmniSearchPattern",
    "PATConfig",
    "PATState",
    "PATStateData",
    "PATStateMachine",
    "RasterSearchPattern",
    "SearchPatternProtocol",
    "SpiralSearchPattern",
]
