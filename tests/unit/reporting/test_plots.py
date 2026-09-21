"""Unit tests for Phase 7 plotting module."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pytest

from fsoc_pat.reporting.plots import generate_mission_summary_plot


def test_generate_mission_summary_plot(tmp_path: Path):
    """Verify that multi-panel telemetry plot renders and saves without error."""
    records = []
    for i in range(25):
        records.append({
            "time": i * 0.02,
            "gt_error_total": 0.001 + 0.0001 * i,
            "error_total": 0.0009 + 0.0001 * i,
            "gimbal_pan": 0.02 * i,
            "gimbal_tilt": -0.01 * i,
            "pat_state": "TRACKING" if i < 15 else "REACQUISITION",
            "detected": i < 15,
            "rx_power_dbm": -30.0 if i < 15 else -80.0,
            "link_margin_db": 8.0 if i < 15 else -42.0,
            "ber": 1e-12 if i < 15 else 1.0,
        })

    out_png = tmp_path / "plot.png"
    fig = generate_mission_summary_plot(records, output_path=out_png, scenario_name="Test Run")

    assert fig is not None
    assert out_png.exists()
    assert out_png.stat().st_size > 1000  # File has nonzero bytes


def test_empty_records_plot(tmp_path: Path):
    """Verify graceful handling when no telemetry is provided."""
    fig = generate_mission_summary_plot([], output_path=tmp_path / "empty.png")
    assert fig is not None
    assert (tmp_path / "empty.png").exists()
