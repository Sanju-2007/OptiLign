"""Unit tests for the PAT State Machine and search patterns."""

import math
import pytest

from fsoc_pat.core.types import AngularError, ControlCommand
from fsoc_pat.pat.pat_state_machine import PATStateMachine
from fsoc_pat.pat.pat_types import PATConfig, PATState
from fsoc_pat.pat.search_patterns import RasterSearchPattern, SpiralSearchPattern


def test_spiral_search_pattern():
    spiral = SpiralSearchPattern(step_rad=0.08, angular_speed=1.0, max_radius_rad=0.4)
    spiral.reset(center_pan=0.1, center_tilt=-0.05)

    assert not spiral.is_finished

    # Step through pattern
    dt = 0.05
    pan_rates = []
    tilt_rates = []
    for _ in range(50):
        pr, tr = spiral.step(dt)
        pan_rates.append(pr)
        tilt_rates.append(tr)

    assert any(abs(r) > 0.01 for r in pan_rates)
    d_pan, d_tilt = spiral.current_offset()
    radius = math.hypot(d_pan, d_tilt)
    assert radius > 0.0
    assert radius <= 0.4 + 1e-4


def test_raster_search_pattern():
    raster = RasterSearchPattern(pan_span_rad=0.4, tilt_span_rad=0.2, step_rad=0.05, scan_rate=0.5)
    raster.reset(center_pan=0.0, center_tilt=0.0)

    dt = 0.05
    for _ in range(30):
        pr, tr = raster.step(dt)

    d_pan, d_tilt = raster.current_offset()
    assert abs(d_pan) <= 0.25
    assert abs(d_tilt) <= 0.15


def test_pat_state_machine_initial_standby():
    sm = PATStateMachine()
    assert sm.state == PATState.STANDBY

    # Stepping without detection remains in standby
    pr, tr = sm.step(0.02, detected=False, angular_error=None, gimbal_pan=0.0, gimbal_tilt=0.0)
    assert pr == 0.0 and tr == 0.0
    assert sm.state == PATState.STANDBY


def test_pat_state_machine_search_to_acquisition_and_track():
    cfg = PATConfig(confirm_detections_threshold=2, acquisition_lock_error_rad=0.05)
    sm = PATStateMachine(config=cfg)
    sm.set_state(PATState.SEARCH)

    # Frame 1: Detection
    sm.step(0.02, detected=True, angular_error=None, gimbal_pan=0.0, gimbal_tilt=0.0)
    assert sm.state == PATState.SEARCH  # Needs 2 consecutive detections

    # Frame 2: Confirmed detection -> ACQUISITION
    sm.step(0.02, detected=True, angular_error=None, gimbal_pan=0.0, gimbal_tilt=0.0)
    assert sm.state == PATState.ACQUISITION

    # While in ACQUISITION, large error stays in ACQUISITION
    ang_err_large = AngularError(horizontal=0.1, vertical=0.1, total=0.14)
    pid_cmd = ControlCommand(pan_rate=0.2, tilt_rate=0.2)
    sm.step(0.02, detected=True, angular_error=ang_err_large, gimbal_pan=0.0, gimbal_tilt=0.0, pid_cmd=pid_cmd)
    assert sm.state == PATState.ACQUISITION

    # Small error transitions to TRACKING
    ang_err_small = AngularError(horizontal=0.01, vertical=0.01, total=0.014)
    sm.step(0.02, detected=True, angular_error=ang_err_small, gimbal_pan=0.0, gimbal_tilt=0.0, pid_cmd=pid_cmd)
    assert sm.state == PATState.TRACKING


def test_pat_state_machine_reacquisition_on_loss_and_relock():
    cfg = PATConfig(loss_misses_threshold=3, coast_time_s=0.1, confirm_detections_threshold=2)
    sm = PATStateMachine(config=cfg)
    sm.set_state(PATState.TRACKING)

    pid_cmd = ControlCommand(pan_rate=0.1, tilt_rate=-0.1)

    # In TRACKING with detection
    sm.step(0.02, detected=True, angular_error=None, gimbal_pan=0.2, gimbal_tilt=-0.1, pid_cmd=pid_cmd)
    assert sm.state == PATState.TRACKING

    # Miss 1: coasts
    sm.step(0.02, detected=False, angular_error=None, gimbal_pan=0.2, gimbal_tilt=-0.1)
    assert sm.state == PATState.TRACKING

    # Miss 2: coasts
    sm.step(0.02, detected=False, angular_error=None, gimbal_pan=0.2, gimbal_tilt=-0.1)
    assert sm.state == PATState.TRACKING

    # Miss 3: triggers REACQUISITION
    sm.step(0.02, detected=False, angular_error=None, gimbal_pan=0.2, gimbal_tilt=-0.1)
    assert sm.state == PATState.REACQUISITION

    # Coasting phase in REACQUISITION (t_state < coast_time_s=0.1s)
    pr, tr = sm.step(0.02, detected=False, angular_error=None, gimbal_pan=0.2, gimbal_tilt=-0.1)
    assert pr == 0.0 and tr == 0.0

    # Advance past coasting time -> spiral search begins
    for _ in range(5):
        pr, tr = sm.step(0.02, detected=False, angular_error=None, gimbal_pan=0.2, gimbal_tilt=-0.1)
    assert abs(pr) > 0.0 or abs(tr) > 0.0

    # Target re-detected during spiral -> re-locks to TRACKING
    sm.step(0.02, detected=True, angular_error=None, gimbal_pan=0.2, gimbal_tilt=-0.1)
    assert sm.state == PATState.REACQUISITION  # Needs 2 detections
    sm.step(0.02, detected=True, angular_error=None, gimbal_pan=0.2, gimbal_tilt=-0.1)
    assert sm.state == PATState.TRACKING
