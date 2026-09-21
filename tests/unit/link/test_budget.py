"""Unit tests for FSOC link budget analysis."""

import math
import pytest

from fsoc_pat.link.budget import FSOCLinkBudget
from fsoc_pat.link.link_types import LaserConfig, LinkState, OpticalReceiverConfig


def test_link_budget_perfect_alignment():
    budget = FSOCLinkBudget()
    res = budget.evaluate(distance_m=100.0, pointing_error_rad=0.0)

    # Pointing loss is zero when perfectly aligned
    assert res.pointing_loss_db == 0.0
    assert res.rx_power_w > 0.0
    assert res.rx_power_dbm > -40.0
    assert res.link_margin_db > 0.0
    assert res.state == LinkState.NOMINAL
    assert res.ber < 1e-6


def test_link_budget_pointing_error_penalty():
    budget = FSOCLinkBudget()

    # Zero error vs 0.1 mrad error
    res_aligned = budget.evaluate(distance_m=200.0, pointing_error_rad=0.0)
    res_offset = budget.evaluate(distance_m=200.0, pointing_error_rad=0.0001)

    assert res_offset.pointing_loss_db > 0.0
    assert res_offset.rx_power_w < res_aligned.rx_power_w
    assert res_offset.rx_power_dbm < res_aligned.rx_power_dbm
    assert res_offset.link_margin_db < res_aligned.link_margin_db


def test_link_budget_severe_misalignment_outage():
    budget = FSOCLinkBudget()
    # 5 mrad pointing error (far outside divergence)
    res_bad = budget.evaluate(distance_m=500.0, pointing_error_rad=0.005)

    assert res_bad.pointing_loss_db > 10.0
    assert res_bad.state == LinkState.OUTAGE
    assert res_bad.link_margin_db < 0.0
    assert res_bad.ber > 0.01


def test_link_budget_distance_scaling():
    budget = FSOCLinkBudget()
    res_100m = budget.evaluate(distance_m=100.0, pointing_error_rad=0.0)
    res_1000m = budget.evaluate(distance_m=1000.0, pointing_error_rad=0.0)

    # Longer distance means higher free space loss & lower received power
    assert res_1000m.free_space_loss_db > res_100m.free_space_loss_db
    assert res_1000m.atmospheric_loss_db > res_100m.atmospheric_loss_db
    assert res_1000m.rx_power_w < res_100m.rx_power_w
