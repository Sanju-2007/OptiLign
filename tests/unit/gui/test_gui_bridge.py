"""Unit tests for OptiLign GUI bridge and telemetry server."""

import math
import time
import numpy as np
import pytest

from fsoc_pat.gui.gui_bridge import SimulationBridge
from fsoc_pat.gui.server import create_server
from fsoc_pat.simulation.beacon import BeaconShape


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


def test_dual_terminal_allowed_and_disallowed_pairings():
    """Verify strict link pairing constraints: sat+car and sat+drone are prohibited."""
    bridge = SimulationBridge(target_fps=30.0, auto_start=True)

    # 1. Permitted Links
    valid_pairs = [
        ("car", "car"),
        ("car", "drone"),
        ("drone", "car"),
        ("drone", "drone"),
        ("satellite", "satellite"),
        ("satellite", "building"),
        ("building", "satellite"),
        ("building", "drone"),
        ("building", "car"),
        ("building", "building"),
    ]
    for term_a, term_b in valid_pairs:
        res = bridge.set_terminals(term_a, term_b)
        assert res["status"] == "ok"
        assert bridge.terminal_a == term_a
        assert bridge.terminal_b == term_b

    # 2. Strictly Prohibited Links
    invalid_pairs = [
        ("satellite", "car"),
        ("car", "satellite"),
        ("satellite", "drone"),
        ("drone", "satellite"),
    ]
    for term_a, term_b in invalid_pairs:
        res = bridge.set_terminals(term_a, term_b)
        assert res["status"] == "error"
        assert "not permitted" in res["message"].lower()

    # 3. Invalid platform types
    bad_type = bridge.set_terminals("submarine", "car")
    assert bad_type["status"] == "error"
    assert "invalid terminal a type" in bad_type["message"].lower()


def test_dual_terminal_kinematics_and_doppler_shift():
    """Test relative range, range-rate, and optical Doppler shift for moving platforms."""
    bridge = SimulationBridge(target_fps=30.0, auto_start=True)

    # Set to Car-Car (V2V) moving link
    res = bridge.set_terminals("car", "car")
    assert res["status"] == "ok"

    bridge.step()
    packet = bridge.get_telemetry_packet()

    assert packet["terminal_a"] == "car"
    assert packet["terminal_b"] == "car"
    assert "range_m" in packet
    assert packet["range_m"] > 50.0  # Physical distance between cars
    assert "range_rate_mps" in packet
    assert "doppler_shift_hz" in packet
    assert "host_position" in packet
    assert "host_velocity" in packet

    # Physical verification of optical Doppler equation: delta_f = -v_radial / lambda
    lam = 1550e-9
    expected_doppler = -packet["range_rate_mps"] / lam
    assert abs(packet["doppler_shift_hz"] - expected_doppler) < 150.0


def test_dual_terminal_satellite_isl_kinematics():
    """Test LEO satellite-to-satellite inter-satellite link (ISL) geometry and Doppler."""
    bridge = SimulationBridge(target_fps=30.0, auto_start=True)

    # Set to Satellite-Satellite ISL
    res = bridge.set_terminals("satellite", "satellite")
    assert res["status"] == "ok"
    assert bridge.domain == "space_leo"

    for _ in range(10):
        bridge.step()

    packet = bridge.get_telemetry_packet()
    assert packet["terminal_a"] == "satellite"
    assert packet["terminal_b"] == "satellite"
    assert packet["domain"] == "space_leo"
    assert packet["range_m"] > 10.0
    # Optical Doppler shift should be active
    assert packet["doppler_shift_hz"] != 0.0


def test_terminal_kinematics_and_moving_trajectories():
    """Verify programmatic setting of terminal moving trajectories and velocities."""
    bridge = SimulationBridge(target_fps=30.0, auto_start=True)
    # Configure moving terminals (car to drone) so trajectories can be animated dynamically
    bridge.set_terminals("car", "drone")

    # 1. Update kinematics dynamically
    res = bridge.set_terminal_kinematics(
        terminal_a_trajectory="linear",
        terminal_a_speed_mps=25.0,
        terminal_b_trajectory="figure_of_8",
        terminal_b_speed_mps=35.0,
    )
    assert res["status"] == "ok"
    assert bridge.terminal_a_trajectory == "linear"
    assert bridge.terminal_a_speed_mps == 25.0
    assert bridge.terminal_b_trajectory == "figure_of_8"
    assert bridge.terminal_b_speed_mps == 35.0

    # 2. Verify kinematics calculations over simulation steps
    for _ in range(5):
        bridge.step()

    packet = bridge.get_telemetry_packet()
    assert packet["terminal_a_trajectory"] == "linear"
    assert packet["terminal_a_speed_mps"] == 25.0
    assert packet["terminal_b_trajectory"] == "figure_of_8"
    assert packet["terminal_b_speed_mps"] == 35.0

    # Host velocity magnitude check
    h_vel = packet["host_velocity"]
    h_speed = math.sqrt(h_vel[0] ** 2 + h_vel[1] ** 2 + h_vel[2] ** 2)
    assert abs(h_speed - 25.0) < 1.0

    # 3. Test through load_custom_environment
    env_data = {
        "name": "Kinematics_Test_Env",
        "domain": "urban_rooftop",
        "terminal_a": "car",
        "terminal_b": "drone",
        "terminal_a_trajectory": "sinusoidal",
        "terminal_a_speed_mps": 18.5,
        "terminal_b_trajectory": "evasive",
        "terminal_b_speed_mps": 28.0,
        "num_targets": 1,
    }
    load_res = bridge.load_custom_environment(env_data)
    assert load_res["status"] == "ok"
    assert bridge.terminal_a_trajectory == "sinusoidal"
    assert bridge.terminal_a_speed_mps == 18.5
    assert bridge.terminal_b_trajectory == "evasive"
    assert bridge.terminal_b_speed_mps == 28.0


def test_zoom_scales_intrinsics_and_decoy_distances():
    """Verify optical zoom scales camera intrinsics and expands decoy pixel separation."""
    bridge = SimulationBridge(target_fps=30.0, auto_start=True)

    # 1. Base zoom (1.0x)
    bridge.set_zoom(1.0)
    fx_1 = bridge.intrinsics.fx
    fy_1 = bridge.intrinsics.fy
    hfov_1 = bridge.intrinsics.hfov_deg
    assert abs(bridge.zoom - 1.0) < 1e-4

    # 2. 2.0x zoom
    bridge.set_zoom(2.0)
    fx_2 = bridge.intrinsics.fx
    fy_2 = bridge.intrinsics.fy
    hfov_2 = bridge.intrinsics.hfov_deg
    assert abs(fx_2 - 2.0 * fx_1) < 1e-3
    assert abs(fy_2 - 2.0 * fy_1) < 1e-3
    assert hfov_2 < hfov_1

    # 3. 4.0x zoom
    bridge.set_zoom(4.0)
    fx_4 = bridge.intrinsics.fx
    assert abs(fx_4 - 4.0 * fx_1) < 1e-3

    # Multi-target / decoy projection and distance scaling
    env_data = {
        "name": "Decoy_Zoom_Test_Env",
        "domain": "ground_to_air",
        "terminal_a": "building",
        "terminal_b": "drone",
        "num_targets": 3,
    }
    bridge.load_custom_environment(env_data)
    bridge.control_mode = "auto_track"

    # Step at 1.0x zoom (allow gimbal tracking to acquire and center target)
    bridge.set_zoom(1.0)
    for _ in range(25):
        bridge.step()

    packet_1 = bridge.get_telemetry_packet()
    assert packet_1["zoom"] == "1.0x"
    assert "hfov_deg" in packet_1
    assert len(packet_1["distractor_candidates"]) == 2  # 2 decoys
    assert packet_1["decoys_rejected_count"] == 2

    # Measure decoy pixel distances at 1.0x vs 2.0x
    t_pos_1x = packet_1["target_position_px"]
    decoy_1x = packet_1["distractor_candidates"][0]
    dist_1x = math.hypot(decoy_1x[0] - t_pos_1x[0], decoy_1x[1] - t_pos_1x[1])
    assert dist_1x > 0.0

    # Step at 2.0x zoom
    bridge.set_zoom(2.0)
    bridge.step()
    packet_2 = bridge.get_telemetry_packet()
    assert packet_2["zoom"] == "2.0x"
    assert len(packet_2["distractor_candidates"]) == 2
    t_pos_2x = packet_2["target_position_px"]
    decoy_2x = packet_2["distractor_candidates"][0]
    dist_2x = math.hypot(decoy_2x[0] - t_pos_2x[0], decoy_2x[1] - t_pos_2x[1])

    # When zoomed in 2x, the pixel distance between target and decoy expands proportionally (~2x)
    ratio = dist_2x / dist_1x
    assert abs(ratio - 2.0) < 0.25, f"Expected ratio ~2.0, got {ratio}"


def test_virtual_space_and_viewport_mapping():
    """Verify 2000 x 2000 Virtual Screen Space mapping with 640 x 480 Viewport."""
    bridge = SimulationBridge(target_fps=30.0, auto_start=True)
    packet = bridge.get_telemetry_packet()

    assert "virtual_space" in packet
    vs = packet["virtual_space"]
    assert vs["space_width"] == 2000
    assert vs["space_height"] == 2000
    assert vs["viewport_width"] == 640
    assert vs["viewport_height"] == 480
    assert len(vs["viewport_center"]) == 2
    assert len(vs["viewport_rect"]) == 4
    assert vs["viewport_rect"][2] == 640
    assert vs["viewport_rect"][3] == 480
    assert len(vs["target_virtual_pos"]) == 2
    assert isinstance(vs["in_viewport"], (bool, np.bool_))

    # Test that slewing the camera changes the viewport center in virtual coordinates
    init_center_u = vs["viewport_center"][0]
    bridge.set_manual_rates(pan_rate=0.1, tilt_rate=0.0)
    for _ in range(10):
        bridge.step()
    new_packet = bridge.get_telemetry_packet()
    new_center_u = new_packet["virtual_space"]["viewport_center"][0]
    assert new_center_u > init_center_u


def test_default_target_shape_and_size_preset():
    """Verify default target shape is Square (10 x 10 px) and dynamic reconfiguration works."""
    bridge = SimulationBridge(target_fps=30.0, auto_start=True)

    # Verify default shape & size
    assert bridge.target_shape == "square"
    assert bridge.target_size_px == 10.0
    assert bridge.beacon.shape == BeaconShape.SQUARE
    assert bridge.beacon.size_px == 10.0

    packet = bridge.get_telemetry_packet()
    assert packet["target_shape"] == "square"
    assert packet["target_size_px"] == 10.0

    # Change to circle 15px
    res = bridge.set_target_properties(shape="circle", size_px=15.0)
    assert res["status"] == "ok"
    assert bridge.target_shape == "circle"
    assert bridge.target_size_px == 15.0
    assert bridge.beacon.shape == BeaconShape.CIRCLE
    assert bridge.beacon.size_px == 15.0

    # Reset back to square 10x10 px
    res2 = bridge.set_target_properties(shape="square", size_px=10.0)
    assert res2["status"] == "ok"
    assert bridge.target_shape == "square"
    assert bridge.target_size_px == 10.0
    assert bridge.beacon.shape == BeaconShape.SQUARE


def test_camera_pan_tilt_speed_limiter():
    """Verify Camera Pan & Tilt Speed Limiter: default 5 °/s, range 5–10 °/s, and clamping."""
    bridge = SimulationBridge(target_fps=30.0, auto_start=True)

    # Verify default speed limit is 5.0 °/s
    assert bridge.camera_speed_limit_deg_s == 5.0
    expected_rad_limit = math.radians(5.0)
    assert abs(bridge.gimbal.dynamics.pan_rate_limit - expected_rad_limit) < 1e-4

    # Set valid speed in range [5, 10]
    res = bridge.set_camera_speed_limit(7.5)
    assert res["status"] == "ok"
    assert bridge.camera_speed_limit_deg_s == 7.5
    assert abs(bridge.gimbal.dynamics.pan_rate_limit - math.radians(7.5)) < 1e-4
    assert abs(bridge.gimbal.dynamics.tilt_rate_limit - math.radians(7.5)) < 1e-4

    # Test lower clamping (< 5.0 °/s -> clamps to 5.0 °/s)
    bridge.set_camera_speed_limit(2.0)
    assert bridge.camera_speed_limit_deg_s == 5.0

    # Test upper clamping (> 10.0 °/s -> clamps to 10.0 °/s)
    bridge.set_camera_speed_limit(15.0)
    assert bridge.camera_speed_limit_deg_s == 10.0

    # Verify telemetry packet contains camera_speed_limit_deg_s
    packet = bridge.get_telemetry_packet()
    assert packet["camera_speed_limit_deg_s"] == 10.0


def test_disturbances_presets_and_motion_stress():
    """Verify Haze & Low Light presets, Contrast/Brightness sliders, Jitter (±20px), and Platform Motion (±20px)."""
    bridge = SimulationBridge(target_fps=30.0, auto_start=True)

    # 1. Haze Preset
    res_haze = bridge.set_disturbance_controls(preset="haze")
    assert res_haze["status"] == "ok"
    assert bridge.weather == "Haze"
    assert bridge.contrast_reduction == 0.40
    assert bridge.brightness_reduction == 0.05
    assert bridge.fog_severity == 0.45

    # 2. Low Light Preset
    res_low = bridge.set_disturbance_controls(preset="low_light")
    assert res_low["status"] == "ok"
    assert bridge.weather == "Low Light"
    assert bridge.brightness_reduction == 0.60
    assert bridge.contrast_reduction == 0.20
    assert bridge.enable_poisson_noise is True

    # 3. Sliders: Contrast & Brightness Reduction, Camera Jitter (±20px), Platform Motion (±20px)
    res_custom = bridge.set_disturbance_controls(
        contrast_reduction=0.75,
        brightness_reduction=0.50,
        camera_jitter_px=18.5,
        platform_motion_px=15.0,
    )
    assert res_custom["status"] == "ok"
    assert bridge.contrast_reduction == 0.75
    assert bridge.brightness_reduction == 0.50
    assert bridge.camera_jitter_px == 18.5
    assert bridge.platform_motion_px == 15.0

    # Test clamping at 20 px/frame
    bridge.set_disturbance_controls(camera_jitter_px=35.0, platform_motion_px=40.0)
    assert bridge.camera_jitter_px == 20.0
    assert bridge.platform_motion_px == 20.0

    # Verify telemetry packet carries all disturbance parameters
    packet = bridge.get_telemetry_packet()
    assert packet["contrast_reduction_pct"] == 75.0
    assert packet["brightness_reduction_pct"] == 50.0
    assert packet["camera_jitter_px"] == 20.0
    assert packet["platform_motion_px"] == 20.0

    # Step simulation and verify frame encoding still succeeds with all disturbances applied
    for _ in range(5):
        bridge.step()
    assert len(bridge._cached_frame_jpeg) > 100


def test_pixel_domain_tracking_error_and_rmse():
    """Item 12: Verify real-time pixel-domain tracking error, running pixel RMSE (<= 2.0 px), and spec limit (<= 10 px)."""
    bridge = SimulationBridge(target_fps=30.0, auto_start=True)

    # Initial state
    assert bridge._latest_pixel_error_px == 0.0
    assert bridge._latest_pixel_rmse_px == 0.0
    assert bridge.pixel_rmse_threshold_px == 2.0
    assert bridge.spec_pixel_error_threshold_px == 10.0

    # Step simulation across tracking loop
    for _ in range(20):
        bridge.step()

    packet = bridge.get_telemetry_packet()
    assert "pixel_error_px" in packet
    assert "pixel_error_u" in packet
    assert "pixel_error_v" in packet
    assert "pixel_rmse_px" in packet
    assert "pixel_rmse_compliant" in packet
    assert packet["pixel_rmse_threshold_px"] == 2.0

    # Under closed loop tracking, pixel error is sub-pixel or bounded
    assert isinstance(packet["pixel_error_px"], float)
    assert isinstance(packet["pixel_rmse_px"], float)
    assert packet["pixel_rmse_compliant"] is True
    assert packet["pixel_rmse_px"] <= 2.0  # Standard specification compliant: <= 2.0 px
    assert packet["pixel_rmse_px"] <= 10.0  # Specification Item 17 compliant: <= 10 px

    # History buffers
    assert "pixel_error" in packet["history"]
    assert "pixel_rmse" in packet["history"]
    assert len(packet["history"]["pixel_error"]) == 20
    assert len(packet["history"]["pixel_rmse"]) == 20


def test_acquisition_and_reacquisition_timers():
    """Item 13: Verify initial acquisition timer (<= 2s) and re-acquisition timer (<= 1s)."""
    bridge = SimulationBridge(target_fps=30.0, auto_start=True)

    # Step until initial lock is achieved
    for _ in range(35):
        bridge.step()

    packet = bridge.get_telemetry_packet()
    assert packet["is_initial_acquired"] is True
    assert packet["acquisition_time_s"] is not None
    assert packet["acquisition_time_s"] <= 2.0  # Spec: <= 2.0 sec
    assert packet["acquisition_compliant"] is True

    # Test lock loss & re-acquisition timer initiation
    saved_pan = bridge.gimbal.state.pan
    saved_tilt = bridge.gimbal.state.tilt
    bridge.trigger_out_of_fov_test()
    bridge.step()

    packet_loss = bridge.get_telemetry_packet()
    assert packet_loss["is_reacquiring"] is True
    assert packet_loss["reacquisition_limit_s"] == 1.0

    # Re-center and reacquire
    bridge.recenter_camera()
    bridge.gimbal.set_position(saved_pan, saved_tilt)
    bridge.pat_sm.set_state(bridge.pat_sm.state.__class__.TRACKING, "Target Re-acquired")
    bridge._link_status = "Signal Locked"
    bridge.step()

    packet_rec = bridge.get_telemetry_packet()
    assert packet_rec["is_reacquiring"] is False
    assert packet_rec["reacquisition_time_s"] is not None
    assert packet_rec["reacquisition_compliant"] is True
    assert packet_rec["reacquisition_time_s"] <= 1.0  # Spec: <= 1.0 sec


def test_lock_retention_rate_and_target_loss():
    """Item 14: Verify lock retention rate (> 95%) and target loss percentage (< 5%)."""
    bridge = SimulationBridge(target_fps=30.0, auto_start=True)

    # Run for 30 steps to acquire and track
    for _ in range(30):
        bridge.step()

    packet = bridge.get_telemetry_packet()
    assert "lock_retention_rate_pct" in packet
    assert "target_loss_pct" in packet
    assert "retention_compliant" in packet

    # High-retention nominal tracking
    assert packet["lock_retention_rate_pct"] >= 95.0
    assert packet["target_loss_pct"] < 5.0
    assert packet["retention_compliant"] is True

    # Test reset simulation clears retention counters
    bridge.reset_simulation()
    assert bridge.total_tracking_frames == 0
    assert bridge.tracking_locked_frames == 0
    assert bridge.tracking_lost_frames == 0
    assert bridge.lock_retention_rate_pct == 100.0
    assert bridge.target_loss_pct == 0.0


def test_standardized_isro_performance_log_exporters():
    """Item 15: Verify standardized ISRO Performance Log exporter (CSV & JSON)."""
    bridge = SimulationBridge(target_fps=30.0, auto_start=True)
    for _ in range(15):
        bridge.step()

    # 1. Test CSV exporter
    csv_log = bridge.generate_isro_csv_log()
    assert isinstance(csv_log, str)
    assert "# OPTICAL SATELLITE COMMUNICATIONS - PAT PERFORMANCE BENCHMARK LOG" in csv_log
    assert "# Standard: FSOC-PAT-2026 Virtual Camera Tracking Specification" in csv_log
    assert "# BENCHMARK COMPLIANCE AUDIT:" in csv_log
    assert "Pixel RMSE:" in csv_log
    assert "Initial Acquisition Time:" in csv_log
    assert "Re-acquisition Time:" in csv_log
    assert "Target Loss Percentage:" in csv_log
    assert "Lock Retention Rate:" in csv_log
    assert "timestamp_s,frame_idx,pat_state,link_status,target_u,target_v" in csv_log

    csv_lines = [l for l in csv_log.splitlines() if l and not l.startswith("#")]
    assert len(csv_lines) >= 2  # Header line + at least one data row

    # 2. Test JSON exporter
    json_log = bridge.generate_isro_json_log()
    assert isinstance(json_log, dict)
    assert "benchmark_header" in json_log
    assert json_log["benchmark_header"]["organization"] == "FSOC Optical Communications Terminal Laboratory"
    assert "terminal_parameters" in json_log
    assert "compliance_summary" in json_log
    comp = json_log["compliance_summary"]
    assert "pixel_rmse" in comp
    assert comp["pixel_rmse"]["threshold_px"] == 2.0
    assert "initial_acquisition_time" in comp
    assert comp["initial_acquisition_time"]["threshold_s"] == 2.0
    assert "reacquisition_time" in comp
    assert comp["reacquisition_time"]["threshold_s"] == 1.0
    assert "lock_retention_rate" in comp
    assert comp["lock_retention_rate"]["threshold_pct"] == 95.0
    assert "target_loss" in comp
    assert comp["target_loss"]["threshold_pct"] == 5.0
    assert "kpis" in json_log
    assert "records" in json_log
    assert len(json_log["records"]) > 0


def test_solutions_2_3_4_integration():
    """Verify Solutions 2, 3, 4 controls, states, and telemetry integration."""
    bridge = SimulationBridge(target_fps=30.0, auto_start=True)

    # 1. Solution 2: Dual-Band RF Fallback Toggle
    initial_rf = bridge.rf_manager.auto_rf_fallback_enabled
    toggled_rf = bridge.toggle_rf_fallback()
    assert toggled_rf == (not initial_rf)
    bridge.toggle_rf_fallback()  # Restore

    # 2. Solution 3: ISRO/CCSDS Handshake Trigger
    hs_res = bridge.trigger_handshake()
    assert hs_res["handshake_active"] is True
    assert hs_res["handshake_phase"] == 1
    assert hs_res["terminal_a_divergence_urad"] == 15000.0

    # 3. Solution 4: Solar Glint and Decoy Filter Toggles
    glint_st = bridge.toggle_solar_glint()
    assert glint_st is True
    assert bridge.adversarial_engine.config.solar_glint_enabled is True

    filter_st = bridge.toggle_decoy_filter()
    assert filter_st is False
    assert bridge.adversarial_engine.config.carrier_filter_enabled is False
    bridge.toggle_decoy_filter()  # Restore to True

    # 4. Telemetry packet validation
    packet = bridge.get_telemetry_packet()
    assert "link_carrier_mode" in packet
    assert "rf_fallback" in packet
    assert "cooperative_handshake" in packet
    assert "adversarial" in packet
    assert packet["handshake_active"] is True
    assert packet["solar_glint_active"] is True
    assert packet["adversarial"]["false_positive_rejection_pct"] == 100.0


def test_ml_self_training_and_api():
    bridge = SimulationBridge(target_fps=30.0, auto_start=True)
    # Collect flight telemetry samples
    for _ in range(35):
        bridge.step()

    status_before = bridge.get_ml_training_status()
    assert "samples_collected" in status_before
    assert status_before["samples_collected"] > 0
    assert status_before["max_samples"] == 1200

    # User confirms training -> trigger train_ml_model
    train_res = bridge.train_ml_model(epochs=10, lr=1e-3)
    assert train_res["status"] == "ok"
    assert "final_loss" in train_res
    assert "signature_accuracy_pct" in train_res
    assert train_res["signature_accuracy_pct"] >= 95.0
    assert train_res["epochs_trained"] == 10

    # Verify status after training
    status_after = bridge.get_ml_training_status()
    assert status_after["is_trained"] is True
    assert status_after["metrics"] is not None

    # Verify telemetry packet includes ml_training object
    packet = bridge.get_telemetry_packet()
    assert "ml_training" in packet
    assert packet["ml_training"]["is_trained"] is True


def test_continuous_target_locking_no_getaway():
    bridge = SimulationBridge(target_fps=30.0, auto_start=True)
    bridge.random_flight_enabled = True
    bridge.terminal_b = "drone"
    bridge.set_control_mode("auto_track")

    # Step simulation across multiple zig-zag cycles
    for _ in range(60):
        bridge.step()

    packet = bridge.get_telemetry_packet()
    # Confirm target remains locked, detected, and within FOV
    assert packet["target_detected"] is True
    assert packet["out_of_fov"] is False
    assert packet["virtual_space"]["in_viewport"] is True
    assert packet["is_tracking_locked"] is True
    assert packet["link_status"] == "Signal Locked"
    # Coarse yaw error is kept tightly locked
    assert abs(packet["dual_stage"]["coarse_yaw_deg"]) < 0.5


def test_building_to_building_link_and_stationary_kinematics():
    """Verify building-building pairing is allowed, stationary kinematics are enforced, and 150m/s speed is retained for moving platforms."""
    from fsoc_pat.gui.gui_bridge import is_terminal_pair_allowed

    # 1. Building to Building is allowed
    assert is_terminal_pair_allowed("building", "building") is True

    bridge = SimulationBridge(target_fps=30.0, auto_start=False)
    res = bridge.set_terminals("building", "building")
    assert res["status"] == "ok"
    assert bridge.terminal_a == "building"
    assert bridge.terminal_b == "building"
    assert bridge.terminal_a_trajectory == "stationary"
    assert bridge.terminal_b_trajectory == "stationary"
    assert bridge.terminal_a_speed_mps == 0.0
    assert bridge.terminal_b_speed_mps == 0.0

    # 2. Attempting to set speed for stationary buildings must be strictly forced to 0.0 m/s
    kin_res = bridge.set_terminal_kinematics(
        terminal_a_trajectory="figure_of_eight",
        terminal_a_speed_mps=150.0,
        terminal_b_trajectory="linear",
        terminal_b_speed_mps=150.0,
    )
    assert kin_res["status"] == "ok"
    assert bridge.terminal_a_trajectory == "stationary"
    assert bridge.terminal_b_trajectory == "stationary"
    assert bridge.terminal_a_speed_mps == 0.0
    assert bridge.terminal_b_speed_mps == 0.0

    # 3. Motion stepping preserves stationary coordinates
    for _ in range(10):
        bridge.step()

    packet = bridge.get_telemetry_packet()
    assert packet["terminal_a"] == "building"
    assert packet["terminal_b"] == "building"
    assert packet["terminal_a_speed_mps"] == 0.0
    assert packet["terminal_b_speed_mps"] == 0.0
    assert packet["terminal_a_trajectory"] == "stationary"
    assert packet["terminal_b_trajectory"] == "stationary"

    # 4. Moving platforms can configure and retain full 150 m/s velocity without clamping to 100
    res_sat = bridge.set_terminals("satellite", "satellite")
    assert res_sat["status"] == "ok"
    kin_sat = bridge.set_terminal_kinematics(
        terminal_a_trajectory="linear",
        terminal_a_speed_mps=150.0,
        terminal_b_trajectory="figure_of_eight",
        terminal_b_speed_mps=150.0,
    )
    assert kin_sat["status"] == "ok"
    assert bridge.terminal_a_speed_mps == 150.0
    assert bridge.terminal_b_speed_mps == 150.0


def test_ml_human_in_the_loop_recovery_and_reinforcement():
    """Verify out-of-FOV minimal time all-angle search and human-in-the-loop recovery reinforcement/upgrade."""
    bridge = SimulationBridge(target_fps=30.0, auto_start=True)

    # Initial state
    assert bridge.search_policy_level == 1
    assert bridge.search_speed_multiplier == 1.0
    assert bridge.pending_recovery_prompt is None

    # Step simulation to build baseline tracking
    for _ in range(25):
        bridge.step()

    # Trigger out-of-FOV test (forces high-speed all-angle omnidirectional ML search)
    out_res = bridge.trigger_out_of_fov_test()
    assert out_res["status"] == "ok"
    assert bridge._is_out_of_fov is True
    assert bridge.pat_sm.state.name == "ML_SEARCH"
    assert bridge.ml_search_start_sim_time is not None

    # Step until target is recovered
    for _ in range(50):
        bridge.step()
        if bridge.pending_recovery_prompt is not None:
            break

    # Prompt generated upon reacquisition
    prompt = bridge.pending_recovery_prompt
    assert prompt is not None
    assert "event_id" in prompt
    assert "recovery_time_s" in prompt
    assert prompt["recovery_time_s"] > 0.0
    assert "pan_sweep_deg" in prompt
    assert "tilt_sweep_deg" in prompt

    packet = bridge.get_telemetry_packet()
    assert "recovery_prompt" in packet
    assert packet["recovery_prompt"] is not None

    # Test Policy Upgrade (User says "Too Slow — Upgrade Yourself")
    upg_res = bridge.upgrade_search_policy(prompt["event_id"])
    assert upg_res["status"] == "ok"
    assert upg_res["policy_level"] == 2
    assert upg_res["multiplier"] > 1.0
    assert bridge.search_policy_level == 2
    assert bridge.pending_recovery_prompt is None

    # Set mock prompt and test Accept & Self-Train
    bridge.pending_recovery_prompt = {"event_id": "test_evt_01", "recovery_time_s": 0.65}
    acc_res = bridge.accept_recovery("test_evt_01")
    assert acc_res["status"] == "ok"
    assert "metrics" in acc_res
    assert bridge.pending_recovery_prompt is None
    assert bridge.ml_tracker.is_trained_on_flight_data is True

