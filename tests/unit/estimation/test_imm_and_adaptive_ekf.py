"""Unit tests for Adaptive EKF and Interacting Multiple Model (IMM) Filter."""

import math
import pytest
import numpy as np

from fsoc_pat.estimation.adaptive_ekf import AdaptiveEKF, AdaptiveEKFConfig
from fsoc_pat.estimation.imm import IMMFilter
from fsoc_pat.perception.detection_types import Candidate, Detection, DetectionStatus


def make_detection(frame_index: int, u: float, v: float, conf: float = 0.95) -> Detection:
    cand = Candidate(
        u=u,
        v=v,
        area_px=10.0,
        peak_intensity=200.0,
        mean_intensity=150.0,
    )
    return Detection(
        status=DetectionStatus.VALID,
        candidates=[cand],
        best=cand,
        confidence=conf,
        frame_index=frame_index,
    )


def test_adaptive_ekf_prediction_and_update():
    ekf = AdaptiveEKF()
    ekf.reset()

    # Seed with initial measurement at center (320, 240)
    det1 = make_detection(1, 320.0, 240.0, 0.95)
    ekf.update(det1)

    u, v = ekf.estimated_pixel
    assert abs(u - 320.0) < 0.1
    assert abs(v - 240.0) < 0.1

    # Predict step
    pu, pv = ekf.predict(dt=0.05)
    assert abs(pu - 320.0) < 0.1

    # Update with moving beacon (322, 241)
    det2 = make_detection(2, 322.0, 241.0, 0.90)
    u2, v2 = ekf.update(det2)

    assert 320.0 < u2 <= 322.0
    vel_u, vel_v = ekf.estimated_velocity_px_s
    assert vel_u > 0.0


def test_adaptive_ekf_outlier_rejection():
    ekf = AdaptiveEKF()
    ekf.reset()

    # Establish steady track around (320, 240)
    for i in range(10):
        ekf.predict(0.02)
        ekf.update(make_detection(i, 320.0, 240.0, 0.99))

    # Send a wild outlier (e.g. false solar glint at 50, 50)
    outlier = make_detection(11, 50.0, 50.0, 0.99)
    ekf.predict(0.02)
    u_filt, v_filt = ekf.update(outlier)

    # State should reject the outlier and remain near (320, 240)
    assert abs(u_filt - 320.0) < 2.0
    assert abs(v_filt - 240.0) < 2.0


def test_imm_filter_maneuver_adaptation():
    imm = IMMFilter()
    imm.reset()

    # Steady trajectory
    for i in range(15):
        imm.predict(0.02)
        imm.update(make_detection(i, 300.0 + i * 2.0, 200.0, 0.95))

    est_u, est_v = imm.estimated_pixel
    assert abs(est_u - 328.0) < 1.0

    # Sharp turn / acceleration
    for i in range(10):
        imm.predict(0.02)
        imm.update(make_detection(15 + i, 328.0 + (i**2) * 1.5, 200.0 + (i**2) * 1.0, 0.95))

    probs = imm.mode_probabilities
    assert len(probs) == 3
    # Accelerating/maneuver probability should increase
    assert probs[1] + probs[2] > 0.1
