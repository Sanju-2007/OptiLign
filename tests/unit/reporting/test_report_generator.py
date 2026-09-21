"""Unit tests for Phase 7 report generator and metrics calculation."""

import json
from pathlib import Path

import pytest

from fsoc_pat.reporting.report_generator import (
    OpticalLinkMetrics,
    PATStateMetrics,
    PointingMetrics,
    ReportMetrics,
    SimulationReportGenerator,
)


@pytest.fixture
def sample_telemetry_records():
    records = []
    # Create 50 synthetic records
    for i in range(50):
        t = i * 0.02
        err = 0.001 + 0.0002 * (i % 5)
        # Introduce dropout between 20 and 30
        if 20 <= i <= 30:
            state = "REACQUISITION"
            det = False
            rx_pwr = -100.0
            margin = -50.0
            err = 0.005
        else:
            state = "TRACKING"
            det = True
            rx_pwr = -28.0
            margin = 10.0

        records.append({
            "time": t,
            "gt_error_total": err,
            "error_total": err * 0.95,
            "gimbal_pan": 0.01 * i,
            "gimbal_tilt": -0.005 * i,
            "pat_state": state,
            "detected": det,
            "rx_power_dbm": rx_pwr,
            "pointing_loss_db": 1.2,
            "link_margin_db": margin,
            "ber": 1e-12 if det else 1.0,
        })
    return records


class TestSimulationReportGenerator:
    """Test suite for SimulationReportGenerator."""

    def test_metrics_computation(self, sample_telemetry_records):
        gen = SimulationReportGenerator(sample_telemetry_records, scenario_name="Test Scenario")
        m = gen.metrics

        assert m.scenario_name == "Test Scenario"
        assert m.total_steps == 50
        assert m.duration_s == pytest.approx(49 * 0.02, rel=1e-3)

        # Pointing
        assert m.pointing.mean_radial_error_mrad > 0.0
        assert m.pointing.rms_radial_error_mrad >= m.pointing.mean_radial_error_mrad * 0.8
        assert m.pointing.p95_error_mrad >= m.pointing.median_error_mrad

        # State machine
        assert m.pat_state.percent_in_tracking > 0.0
        assert m.pat_state.percent_in_reacquisition > 0.0
        assert m.pat_state.reacquisition_events_count == 1
        assert m.pat_state.mean_time_to_reacquire_s > 0.0

        # Optical link
        assert m.link.mean_rx_power_dbm < 0.0
        assert 0.0 <= m.link.link_availability_pct <= 100.0
        assert m.link.outage_steps_count == 11

    def test_markdown_generation(self, sample_telemetry_records, tmp_path: Path):
        gen = SimulationReportGenerator(sample_telemetry_records)
        md_file = tmp_path / "report.md"
        content = gen.generate_markdown_report(md_file)

        assert md_file.exists()
        assert "# OptiLign FSOC PAT Mission — Verification & Telemetry Report" in content
        assert "Mean Radial Error (MRE)" in content
        assert "FSOC Optical Communications Link Budget" in content

    def test_html_generation(self, sample_telemetry_records, tmp_path: Path):
        gen = SimulationReportGenerator(sample_telemetry_records)
        html_file = tmp_path / "report.html"
        content = gen.generate_html_report(html_file)

        assert html_file.exists()
        assert "<!DOCTYPE html>" in content
        assert "Mean Radial Error" in content
        assert "PAT State Machine Timeline" in content
        assert "Link Availability" in content

    def test_json_summary_export(self, sample_telemetry_records, tmp_path: Path):
        gen = SimulationReportGenerator(sample_telemetry_records)
        json_file = tmp_path / "summary.json"
        content = gen.generate_json_summary(json_file)

        assert json_file.exists()
        data = json.loads(content)
        assert "pointing" in data
        assert "pat_state" in data
        assert "link" in data
        assert data["total_steps"] == 50
