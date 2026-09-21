"""Unit tests for Fast Steering Mirror (FSM) and DualStageController."""

import math
import pytest
import numpy as np

from fsoc_pat.control.dual_stage import DualStageController, FSMConfig, FastSteeringMirror
from fsoc_pat.core.types import AngularError


def test_fsm_actuator_dynamics():
    cfg = FSMConfig(max_deflection_urad=1000.0, bandwidth_hz=500.0, resolution_urad=0.1)
    fsm = FastSteeringMirror(cfg)

    # Step command
    pan, tilt = fsm.step(500.0, -300.0, dt=0.005)
    assert 0.0 < pan <= 500.0
    assert -300.0 <= tilt < 0.0

    # Let it settle over multiple steps
    for _ in range(50):
        pan, tilt = fsm.step(500.0, -300.0, dt=0.002)

    assert abs(pan - 500.0) < 1.0
    assert abs(tilt - (-300.0)) < 1.0


def test_fsm_saturation():
    cfg = FSMConfig(max_deflection_urad=2000.0)
    fsm = FastSteeringMirror(cfg)

    # Over-range command
    for _ in range(50):
        pan, tilt = fsm.step(5000.0, -5000.0, dt=0.01)

    assert pan <= 2000.0
    assert tilt >= -2000.0


def test_dual_stage_controller_sub_microradian_cancellation():
    controller = DualStageController()
    controller.reset()

    # Small residual error (e.g. 15 micro-radians residual jitter)
    err = AngularError(horizontal=15e-6, vertical=-8e-6, total=math.hypot(15e-6, -8e-6))

    # Update loop for 10 steps
    for _ in range(10):
        cmd, state = controller.update(err, dt=0.01)

    # Fine error after FSM steering should be greatly reduced (< 1 urad)
    assert state.coarse_error_urad > 10.0
    assert state.fine_error_urad < 1.0
    assert abs(state.fsm_pan_urad - 15.0) < 1.0
    assert abs(state.fsm_tilt_urad - (-8.0)) < 1.0


def test_dual_stage_momentum_offloading():
    # Configure low offload threshold to trigger offload
    cfg = FSMConfig(offload_threshold_urad=200.0, offload_gain=1.5)
    controller = DualStageController(fsm_config=cfg)
    controller.reset()

    # Sustained large error (e.g. 500 urad)
    err = AngularError(horizontal=500e-6, vertical=0.0, total=500e-6)

    for _ in range(5):
        cmd, state = controller.update(err, dt=0.02)

    assert state.offload_active is True
    # Offload command should add to coarse gimbal velocity command
    assert cmd.pan_rate > 0.0
