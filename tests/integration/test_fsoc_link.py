"""Integration test for closed-loop PAT coupled with optical communication link evaluation."""

import math
import numpy as np
import pytest

from fsoc_pat.control.closed_loop import ClosedLoopConfig, ClosedLoopEngine
from fsoc_pat.link.budget import FSOCLinkBudget
from fsoc_pat.link.link_types import LinkState


def test_closed_loop_with_fsoc_link_evaluation():
    """Verify that closed-loop PAT tracking yields realistic communication link metrics."""
    config = ClosedLoopConfig(
        duration_s=2.0,
        dt=0.02,
        target_center=(100.0, 0.0, 0.0),
        target_radius=0.1,
        target_angular_rate=0.1,
        seed=42,
    )
    engine = ClosedLoopEngine(config)
    result = engine.run()

    assert len(result.snapshots) > 0

    link_budget = FSOCLinkBudget()
    link_results = []

    for s in result.snapshots:
        # Distance to target is ~100m
        dist = 100.0
        pointing_err = s.error_total

        link_res = link_budget.evaluate(dist, pointing_err)
        link_results.append(link_res)

        assert link_res.rx_power_w > 0.0
        assert not math.isnan(link_res.rx_power_dbm)
        assert not math.isnan(link_res.link_margin_db)

    # When tracking is active and error is low, link should be operational
    good_frames = [r for r in link_results if r.pointing_error_rad < 0.002]
    if good_frames:
        assert any(r.state in (LinkState.NOMINAL, LinkState.DEGRADED) for r in good_frames)
        assert any(r.link_margin_db > 0.0 for r in good_frames)
