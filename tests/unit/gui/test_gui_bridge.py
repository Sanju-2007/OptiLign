"""Unit tests for OptiLign GUI bridge and telemetry server."""

import math
import time
import pytest

from fsoc_pat.gui.gui_bridge import SimulationBridge
from fsoc_pat.gui.server import create_server


def test_simulation_bridge_initialization():
    bridge = SimulationBridge(target_fps=30.0)
    # Simulator does NOT run until explicitly started by operator or environment creation
    assert bridge.running is False
    assert bridge.sim_time == 0.0
    assert bridge.step_count == 0
    assert bridge.control_mode == "auto_track"
    assert bridge.scenario_name == "Urban_Link"


def test_simulation_bridge_clock_not_running_before_start():
    bridge = SimulationBridge(target_fps=30.0)
    assert bridge.running is False
    bridge.step()
    assert bridge.sim_time == 0.0
    assert bridge.step_count == 0

    bridge.start_simulation()
    assert bridge.running is True
    bridge.step()
    assert bridge.step_count == 1
    assert bridge.sim_time > 0.0


def test_simulation_bridge_step():
    bridge = SimulationBridge(target_fps=30.0, auto_start=True)
    bridge.step()

    assert bridge.step_count == 1
    assert bridge.sim_time > 0.0
    assert len(bridge.error_history_time) == 1
    assert len(bridge.error_history_yaw) == 1
    assert len(bridge.error_history_pitch) == 1


def test_simulation_bridge_telemetry_packet():
    bridge = SimulationBridge(target_fps=30.0, auto_start=True)
    for _ in range(5):
        bridge.step()

    packet = bridge.get_telemetry_packet()

    # Core metadata
    assert "time_formatted" in packet
    assert "battery_pct" in packet
    assert "signal_strength_dbm" in packet
    assert "weather" in packet
    assert "temperature_c" in packet
    assert "wind_speed_mps" in packet
    assert "link_margin_db" in packet

    # Tracking status
    assert "target_detected" in packet
    assert "tracking_mode" in packet
    assert "confidence_pct" in packet
    assert "target_position_px" in packet
    assert "target_offset_px" in packet
    assert "estimated_distance_m" in packet
    assert "link_status" in packet
    assert "fps" in packet

    # Camera position
    assert "camera_position" in packet
    assert "x" in packet["camera_position"]
    assert "yaw_deg" in packet["camera_position"]
    assert "pitch_deg" in packet["camera_position"]

    # History & Logs
    assert "history" in packet
    assert "times" in packet["history"]
    assert "logs" in packet
    assert len(packet["logs"]) > 0


def test_simulation_bridge_control_modes():
    bridge = SimulationBridge(target_fps=30.0, auto_start=True)

    bridge.set_control_mode("manual")
    assert bridge.control_mode == "manual"

    bridge.set_manual_rates(0.8, -0.4)
    assert bridge._manual_pan_rate == 0.8
    assert bridge._manual_tilt_rate == -0.4

    bridge.step()

    bridge.recenter_camera()
    assert bridge.gimbal.state.pan == 0.0
    assert bridge.gimbal.state.tilt == 0.0

    bridge.set_control_mode("auto_track")
    assert bridge.control_mode == "auto_track"


def test_simulation_bridge_scenarios():
    bridge = SimulationBridge(target_fps=30.0)
    bridge.set_scenario("High_Altitude")
    assert bridge.scenario_name == "High_Altitude"
    assert bridge.target_name == "HighAlt_UAV"

    bridge.set_scenario("Fast_Flyby")
    assert bridge.scenario_name == "Fast_Flyby"
    assert bridge.target_name == "Fast_Drone"


def test_server_creation():
    bridge = SimulationBridge(target_fps=30.0)
    server = create_server(host="127.0.0.1", port=9090, bridge=bridge)
    assert server is not None
    server.server_close()


def test_simulation_bridge_custom_environment():
    bridge = SimulationBridge(target_fps=30.0)

    # Configure custom environment with fog, solar glare, multi-targets, and custom distance
    env_config = {
        "name": "Foggy_Harbor_Link",
        "weather": "Fog",
        "fog_severity": 0.8,
        "solar_glare": True,
        "temperature_c": -5.0,
        "distance_m": 3200.0,
        "num_targets": 2,
        "trajectory": "evasive",
        "laser_power_mw": 1000.0,
        "wavelength_nm": 1550.0,
    }

    res = bridge.load_custom_environment(env_config)
    assert res["status"] == "ok"
    assert bridge.scenario_name == "Foggy_Harbor_Link"
    assert bridge.weather == "Fog"
    assert bridge.fog_severity == 0.8
    assert bridge.solar_glare is True
    assert bridge.temperature_c == -5.0
    assert bridge.target_distance_m == 3200.0
    assert bridge.num_targets == 2

    # Step simulation in custom environment
    for _ in range(5):
        bridge.step()

    packet = bridge.get_telemetry_packet()
    assert packet["scenario"] == "Foggy_Harbor_Link"
    assert packet["fog_severity"] == 0.8
    assert packet["solar_glare"] is True
    assert packet["num_targets"] == 2
    assert len(packet["secondary_targets"]) == 1  # 1 secondary distractor

    # Check reports data
    rep = bridge.get_reports_data()
    assert rep["environment_name"] == "Foggy_Harbor_Link"
    assert rep["fog_severity_pct"] == 80.0
    assert rep["distance_m"] == 3200.0
    assert rep["num_targets"] == 2
    assert rep["total_frames"] >= 5


def test_recent_environments_list():
    bridge = SimulationBridge(target_fps=30.0)
    recent = bridge.get_recent_environments()
    assert len(recent) >= 5
    assert any(e["name"] == "Deep Space ISL" for e in recent)
    assert any(e["name"] == "Maritime Vessel" for e in recent)


def test_simulation_bridge_six_targets_and_telemetry_tweak():
    bridge = SimulationBridge(target_fps=30.0, auto_start=True)

    # 1. Test setting 6 targets
    res = bridge.update_telemetry_parameters({
        "num_targets": 6,
        "fog_severity": 0.65,
        "weather": "Fog",
    })
    assert res["status"] == "ok"
    assert bridge.num_targets == 6
    assert bridge.fog_severity == 0.65
    assert bridge.weather == "Fog"

    # Step simulation
    for _ in range(5):
        bridge.step()

    packet = bridge.get_telemetry_packet()
    assert packet["num_targets"] == 6
    # 5 secondary distractor targets for a 6-target swarm
    assert len(packet["secondary_targets"]) == 5

    # 2. Test clamping to maximum 6 targets
    bridge.update_telemetry_parameters({"num_targets": 10})
    assert bridge.num_targets == 6

    # 3. Test on-the-fly tweaking to 3 targets and clear fog
    bridge.update_telemetry_parameters({"num_targets": 3, "fog_severity": 0.0})
    assert bridge.num_targets == 3
    assert bridge.fog_severity == 0.0
    for _ in range(3):
        bridge.step()
    packet3 = bridge.get_telemetry_packet()
    assert packet3["num_targets"] == 3
    assert len(packet3["secondary_targets"]) == 2


def test_simulation_bridge_beacon_signature_and_dynamic_target_motion():
    bridge = SimulationBridge(target_fps=30.0)

    # Configure custom environment at long slant distance (e.g. 25000m)
    bridge.load_custom_environment({
        "name": "Slant_Space_ISL",
        "distance_m": 25000.0,
        "num_targets": 3,
        "trajectory": "sinusoidal",
    })

    # Step across multiple time steps and verify beacon signature pulses
    seen_states = set()
    y_positions = []
    for _ in range(40):
        bridge.step()
        packet = bridge.get_telemetry_packet()
        assert "beacon_signature" in packet
        sig = packet["beacon_signature"]
        assert sig["sequence"] == ["ON", "OFF", "ON", "OFF", "ON"]
        assert sig["state"] in ("ON", "OFF")
        assert isinstance(sig["is_on"], bool)
        seen_states.add(sig["state"])
        y_positions.append(packet["target_3d_position"][1])

    # Both ON and OFF states must be modulated over time
    assert "ON" in seen_states
    assert "OFF" in seen_states

    # Target cross-track movement must have prominent dynamic range scaled by distance (not stuck near 0)
    y_span = max(y_positions) - min(y_positions)
    assert y_span > 1000.0  # At 25km, cross-track amplitude ~ 25000 * 0.14 = 3500m, span > 1000m


def test_simulation_bridge_ml_out_of_fov_and_random_flight():
    bridge = SimulationBridge(target_fps=30.0)
    bridge.start_simulation()

    # 1. Random flight toggle
    assert bridge.random_flight_enabled is True
    assert bridge.toggle_random_flight() is False
    assert bridge.toggle_random_flight() is True

    # 2. Slew away / out of FOV trigger
    res = bridge.trigger_out_of_fov_test()
    assert res["status"] == "ok"
    assert bridge._is_out_of_fov is True
    from fsoc_pat.pat.pat_types import PATState
    assert bridge.pat_sm.state == PATState.ML_SEARCH

    # Step simulation during all-degrees search
    for _ in range(15):
        bridge.step()

    packet = bridge.get_telemetry_packet()
    assert "ml_confidence_pct" in packet
    assert "ml_predicted_position" in packet
    assert "ml_search_active" in packet
    assert packet["ml_search_active"] is True
    assert packet["random_flight_enabled"] is True


def test_simulation_bridge_detailed_individual_targets_and_domain():
    """Verify that domain, individual targets, laser power, and platform sway are ingested and streamed."""
    bridge = SimulationBridge(target_fps=30.0)
    bridge.start_simulation()

    individual_targets = [
        {"id": "Alpha_Lead", "role": "primary", "trajectory": "circular", "range_m": 250.0, "speed_mps": 30.0, "alt_offset_m": 0.0, "wavelength_nm": 1550, "power_mw": 1000},
        {"id": "Beta_Wing", "role": "escort", "trajectory": "circular", "range_m": 280.0, "speed_mps": 30.0, "alt_offset_m": 15.0, "wavelength_nm": 1064, "power_mw": 500},
        {"id": "Gamma_Decoy", "role": "decoy", "trajectory": "sinusoidal", "range_m": 310.0, "speed_mps": 35.0, "alt_offset_m": -10.0, "wavelength_nm": 850, "power_mw": 250},
    ]

    res = bridge.load_custom_environment({
        "name": "Maritime_Tactical_Link",
        "domain": "maritime",
        "platform_type": "ship",
        "platform_sway_deg": 1.5,
        "platform_sway_freq_hz": 0.25,
        "elevation_m": 12.0,
        "weather": "Sea Spray & Mist",
        "fog_severity": 0.35,
        "solar_glare": False,
        "solar_angle_deg": 45.0,
        "turbulence_scale": 4.0,
        "temperature_c": 18.0,
        "wind_speed_mps": 15.0,
        "num_targets": 3,
        "individual_targets": individual_targets,
        "distance_m": 250.0,
        "wavelength_nm": 1550,
        "laser_power_mw": 1000,
    })

    assert res["status"] == "ok"
    assert bridge.domain == "maritime"
    assert bridge.platform_type == "ship"
    assert bridge.platform_sway_deg == 1.5
    assert bridge.num_targets == 3
    assert bridge.wavelength_nm == 1550
    assert bridge.laser_power_mw == 1000
    assert len(bridge.individual_targets) == 3

    for _ in range(5):
        bridge.step()

    packet = bridge.get_telemetry_packet()
    assert packet["domain"] == "maritime"
    assert packet["platform_type"] == "ship"
    assert packet["platform_sway_deg"] == 1.5
    assert packet["num_targets"] == 3
    assert packet["wavelength_nm"] == 1550
    assert packet["laser_power_mw"] == 1000
    assert len(packet["individual_targets"]) == 3
    assert packet["individual_targets"][0]["id"] == "Alpha_Lead"
    assert packet["individual_targets"][1]["alt_offset_m"] == 15.0
    assert packet["individual_targets"][2]["alt_offset_m"] == -10.0


def test_dual_stage_controller_sub_microradian_and_reports_data():
    """Verify dual-stage coarse/fine controller and comprehensive reports analytics data generation."""
    bridge = SimulationBridge(target_fps=30.0)
    bridge.start_simulation()

    # Step simulation across tracking states
    for _ in range(15):
        bridge.step()

    packet = bridge.get_telemetry_packet()
    assert "dual_stage" in packet
    ds = packet["dual_stage"]
    assert "coarse_yaw_deg" in ds
    assert "coarse_pitch_deg" in ds
    assert "fsm_yaw_urad" in ds
    assert "fsm_pitch_urad" in ds
    assert "net_optical_error_urad" in ds
    assert "diffraction_limit_urad" in ds
    assert "locked_fine" in ds

    # Coarse error and fine optical error histories should be populating
    assert len(bridge.error_history_coarse_yaw) > 0
    assert len(bridge.error_history_yaw) > 0

    # Test Reports Data endpoint
    rep = bridge.get_reports_data()
    assert "status" in rep and rep["status"] == "ok"
    assert "generated_at" in rep
    assert "kpis" in rep
    kpis = rep["kpis"]
    assert "uptime_s" in kpis
    assert "availability_pct" in kpis
    assert "avg_error_urad" in kpis
    assert "max_error_urad" in kpis
    assert "jitter_floor_urad" in kpis
    assert "current_state" in kpis
    assert "margin_db" in kpis
    assert "ber" in kpis

    assert "timeseries" in rep
    ts = rep["timeseries"]
    for key in (
        "time",
        "coarse_error_urad",
        "fine_error_urad",
        "rx_power_dbm",
        "link_margin_db",
        "fog_extinction_db",
        "pat_states",
        "ber_exponent",
        "rytov_variance",
    ):
        assert key in ts
        assert isinstance(ts[key], list)

    assert "subsystem_audit" in rep
    audit = rep["subsystem_audit"]
    assert "gimbal_coarse" in audit
    assert "fsm_fine" in audit
    assert "ai_tracking_core" in audit
    assert "atmospheric_channel" in audit
    assert "link_budget_engine" in audit

    assert "recent_events" in rep
    assert isinstance(rep["recent_events"], list)
    assert len(rep["recent_events"]) > 0

    # Dense fog test with optical beacon light
    bridge.update_telemetry_parameters({"fog_severity": 0.95, "weather": "Dense Fog"})
    bridge.step()
    fog_packet = bridge.get_telemetry_packet()
    assert fog_packet["fog_severity"] == 0.95
    assert fog_packet["weather"] == "Dense Fog"
    assert "beacon_signature" in fog_packet
    assert fog_packet["beacon_signature"]["is_on"] is not None




