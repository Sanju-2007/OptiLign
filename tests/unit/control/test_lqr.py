"""Unit tests for LQRController."""

import math
import pytest

from fsoc_pat.control.lqr import LQRConfig, LQRController
from fsoc_pat.core.types import AngularError


def test_lqr_controller_response():
    ctrl = LQRController(LQRConfig(q_pos=100.0, q_vel=10.0, q_int=20.0, rate_limit_rad_s=1.5))
    ctrl.reset()

    # Step error
    err = AngularError(horizontal=0.05, vertical=-0.03, total=math.hypot(0.05, -0.03))
    cmd = ctrl.compute(err, dt=0.02)

    assert cmd.pan_rate > 0.0
    assert cmd.tilt_rate < 0.0
    assert abs(cmd.pan_rate) <= 1.5
    assert abs(cmd.tilt_rate) <= 1.5


def test_lqr_integral_action():
    ctrl = LQRController()
    ctrl.reset()

    err = AngularError(horizontal=0.01, vertical=0.0, total=0.01)
    # Prime initial step so derivative spike passes
    ctrl.compute(err, dt=0.05)
    cmd_baseline = ctrl.compute(err, dt=0.05)
    for _ in range(10):
        cmd_later = ctrl.compute(err, dt=0.05)

    assert cmd_later.pan_rate > cmd_baseline.pan_rate
