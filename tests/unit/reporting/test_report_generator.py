"""Unit tests for Phase 7 report generator and metrics calculation."""

import json
import math
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
        assert "performance" in data
        assert "ps04" in data
        assert data["total_steps"] == 50

    def test_ps04_pixel_and_timing_metrics(self):
        """Verify explicit PS-04 pixel errors, actual processing FPS, and pass/fail thresholds."""
        records = []
        for i in range(100):
            t = i * 0.033
            # first 5 frames: initial search/acq
            is_acq = i >= 5
            det = is_acq
            state = "TRACKING" if is_acq else "ACQUISITION"
            records.append({
                "time": t,
                "detected": det,
                "pat_state": state,
                "det_u": 322.5 + (0.2 if i % 2 == 0 else -0.2),
                "det_v": 241.0,
                "gt_u": 320.0,
                "gt_v": 240.0,
                "centroid_error_px": 2.7,
                "tracking_error_px": 2.7,
                "processing_time_ms": 18.5,
                "gt_error_total": 0.002,
                "error_total": 0.002,
                "rx_power_dbm": -25.0,
                "link_margin_db": 12.0,
                "ber": 1e-12,
            })

        gen = SimulationReportGenerator(records, scenario_name="PS-04 Baseline Mission")
        m = gen.metrics
        k = m.performance

        # PS-04 Pixel Tracking
        assert k.average_centroid_error_px == pytest.approx(2.7, abs=0.1)
        assert k.rmse_centroid_error_px == pytest.approx(2.7, abs=0.1)
        assert k.maximum_centroid_error_px <= 10.0
        assert k.ps04_tracking_pass is True

        # Input FPS vs Actual Processing FPS
        assert k.input_fps == pytest.approx(30.0, abs=1.0)
        assert k.actual_processing_fps == pytest.approx(1000.0 / 18.5, abs=2.0)
        assert k.mean_processing_time_ms == pytest.approx(18.5, abs=0.1)
        assert k.ps04_processing_pass is True

        # Acquisition time: acquired at frame 5 (t = 5 * 0.033 = 0.165 s)
        assert k.acquisition_time_s == pytest.approx(0.165, abs=0.05)
        assert k.ps04_acquisition_pass is True

        # Lock retention and loss
        assert k.target_loss_pct <= 5.0
        assert k.ps04_loss_pass is True

        # Overall verdict
        assert k.ps04_overall_compliant is True
        assert m.ps04.overall_pass is True

        # Check Markdown and HTML output for PS-04 section
        md = gen.generate_markdown_report()
        assert "## OPTILIGN — PS-04 PERFORMANCE REPORT" in md
        assert "Average Centroid Error" in md
        assert "Actual Processing FPS" in md
        assert "PS-04 LIMITS COMPLIANCE SUMMARY" in md

        html = gen.generate_html_report()
        assert "PS-04 Headline Performance Deliverables" in html
        assert "PS-04 Competition Requirements Compliance Benchmark" in html
        assert "PS-04 COMPLIANT (PASS)" in html

    def test_detailed_metrics_and_sections(self):
        """Verify expanded actuator kinematics, disturbances, perception, and link breakdown."""
        records = []
        for i in range(100):
            t = i * 0.02
            # Simulate dynamic trajectory and transient cloud between 20 and 30
            is_cloud = (20 <= i <= 30)
            det = not is_cloud
            state = "REACQUISITION" if is_cloud else "TRACKING"
            cloud_trans = 0.01 if is_cloud else 1.0

            records.append({
                "time": t,
                "gt_u": 320.0 + 5.0 * math.cos(i * 0.1),
                "gt_v": 240.0 + 3.0 * math.sin(i * 0.1),
                "det_u": 320.5 + 5.0 * math.cos(i * 0.1) if det else float("nan"),
                "det_v": 240.3 + 3.0 * math.sin(i * 0.1) if det else float("nan"),
                "est_u": 320.4 + 5.0 * math.cos(i * 0.1),
                "est_v": 240.2 + 3.0 * math.sin(i * 0.1),
                "error_total": 0.003,
                "gt_error_total": 0.0032,
                "centroid_error_px": 0.58 if det else float("nan"),
                "tracking_error_px": 5.8,
                "processing_time_ms": 19.2,
                "gimbal_pan": 0.05 * math.sin(i * 0.05),
                "gimbal_tilt": 0.03 * math.cos(i * 0.05),
                "gimbal_pan_rate": 0.025,
                "gimbal_tilt_rate": 0.015,
                "confidence": 0.94 if det else 0.0,
                "detected": det,
                "pat_state": state,
                "distance_m": 150.0,
                "jitter_pan_urad": 25.0,
                "jitter_tilt_urad": 22.0,
                "wander_pan_urad": 12.0,
                "wander_tilt_urad": 10.0,
                "scintillation_factor": 0.04,
                "cloud_transmissivity": cloud_trans,
                "rx_power_dbm": -100.0 if is_cloud else -18.5,
                "pointing_loss_db": 0.42,
                "link_margin_db": -62.0 if is_cloud else 19.5,
                "snr_db": 0.0 if is_cloud else 32.5,
                "q_factor": 0.0 if is_cloud else 21.0,
                "ber": 0.5 if is_cloud else 1e-15,
            })

        gen = SimulationReportGenerator(records, scenario_name="High-Fidelity Aerospace Test")
        m = gen.metrics

        # 1. Pointing and Sub-pixel verification
        assert m.pointing.average_centroid_error_px <= 2.0
        assert m.pointing.subpixel_accuracy_achieved is True
        assert m.pointing.horizontal_centroid_error_px > 0.0

        # 2. Gimbal Actuator Kinematics
        assert m.gimbal.pan_travel_range_deg > 0.0
        assert m.gimbal.tilt_travel_range_deg > 0.0
        assert m.gimbal.pan_slew_margin_pct > 50.0
        assert m.gimbal.tilt_slew_margin_pct > 50.0
        assert m.gimbal.rate_saturation_count == 0

        # 3. Perception Fidelity
        assert m.perception.total_frames == 100
        assert m.perception.detected_frames == 89
        assert m.perception.detection_rate_pct == pytest.approx(89.0, abs=0.5)
        assert m.perception.mean_confidence > 0.8
        assert m.perception.subpixel_localization_verified is True

        # 4. PAT Reacquisition Log
        assert m.pat_state.reacquisition_events_count == 1
        assert len(m.pat_state.reacquisition_events_log) == 1
        ev = m.pat_state.reacquisition_events_log[0]
        assert ev["event_id"] == 1
        assert ev["duration_s"] > 0.0

        # 5. Disturbances and Channel
        assert m.disturbances.rms_base_jitter_urad > 0.0
        assert isinstance(m.disturbances.jitter_rejection_ratio_db, float)
        assert m.disturbances.cloud_events_count == 1
        assert m.disturbances.peak_cloud_attenuation_db > 10.0

        # 6. Optical Link Locked vs Overall Separation
        assert m.link.locked_mean_rx_power_dbm == pytest.approx(-18.5, abs=1.0)
        assert m.link.locked_mean_link_margin_db == pytest.approx(19.5, abs=1.0)
        assert m.link.locked_mean_pointing_loss_db == pytest.approx(0.42, abs=0.1)
        assert m.link.locked_mean_snr_db > 20.0
        assert m.link.locked_mean_ber <= 1e-6
        assert m.link.fec_pass is True

        # 7. Complete Report Content Checks
        md = gen.generate_markdown_report()
        assert "## 1. Scenario & Physical Simulation Profile" in md
        assert "## 4. 2-DOF Gimbal Actuator Kinematics & Slew Profile" in md
        assert "## 7. Environmental Disturbances & Channel Scintillation" in md
        assert "## 8. End-to-End FSOC Optical Communications Link Budget" in md

        html = gen.generate_html_report()
        assert "2-DOF Gimbal Actuator Kinematics & Slew Profile" in html
        assert "End-to-End FSOC Optical Communications Link Budget" in html
        assert "Environmental Disturbances & Channel Turbulence" in html

