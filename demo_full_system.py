#!/usr/bin/env python3
"""OptiLign Master End-to-End Mission Demonstration.

Integrates and demonstrates all 7 project phases concurrently:
  Phase 1: 3D Trajectory, Coordinate Transforms & 2-DOF Virtual Gimbal
  Phase 2: Synthetic Photometric Sensor, Noise & Classical Centroiding
  Phase 3: BeaconNet Deep Learning Perception & Hybrid Fusion
  Phase 4: Extended Kalman Filter (EKF) & Feedforward PID Gimbal Control
  Phase 5: Disturbances (Jitter + Atmosphere + Cloud Dropout) & PAT Reacquisition
  Phase 6: Gaussian Beam Diffraction Optics & FSOC Link Budget Analysis
  Phase 7: Automated Mission Telemetry Reporting (HTML, Markdown, JSON, Plots)
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

# Phase 1 & 4
from fsoc_pat.control.pid import PIDConfig, PIDController
from fsoc_pat.core.state import CameraState
from fsoc_pat.core.types import AngularError, AngularState, Orientation, PixelCoord, Position3D
from fsoc_pat.estimation.kalman_filter import KalmanConfig, KalmanFilter
from fsoc_pat.estimation.motion_models import MotionModelType
from fsoc_pat.geometry.angular_error import compute_angular_error_from_pixel
from fsoc_pat.geometry.camera_model import CameraIntrinsics
from fsoc_pat.geometry.projection import project_point_to_pixel
from fsoc_pat.simulation.beacon import BeaconProperties
from fsoc_pat.simulation.dynamics import VirtualGimbal
from fsoc_pat.simulation.trajectories import CircularTrajectory

# Phase 2 & 3
from fsoc_pat.ai.hybrid_detector import HybridConfig, HybridDetector
from fsoc_pat.perception.classical_detector import ClassicalDetectorConfig
from fsoc_pat.perception.pipeline import PerceptionConfig, PerceptionPipeline
from fsoc_pat.perception.preprocessing import PreprocessConfig
from fsoc_pat.perception.renderer import RenderConfig, SyntheticRenderer

# Phase 5
from fsoc_pat.disturbances.atmospheric import AtmosphericConfig, AtmosphericTurbulence
from fsoc_pat.disturbances.occlusion import OcclusionConfig, OcclusionModel
from fsoc_pat.disturbances.platform_jitter import JitterConfig, PlatformJitter
from fsoc_pat.pat.pat_state_machine import PATStateMachine
from fsoc_pat.pat.pat_types import PATConfig, PATState

# Phase 6
from fsoc_pat.link.budget import FSOCLinkBudget
from fsoc_pat.link.link_types import LaserConfig, OpticalReceiverConfig

# Phase 7
from fsoc_pat.reporting.plots import generate_mission_summary_plot
from fsoc_pat.reporting.report_generator import SimulationReportGenerator


def run_master_mission() -> None:
    print("=" * 78)
    print("       OPTILIGN — MASTER END-TO-END FSOC MISSION SIMULATION (PHASES 1-7)")
    print("=" * 78)

    t_wall_start = time.perf_counter()

    # Mission Configuration
    duration_s = 8.0
    dt = 0.02  # 50 Hz control loop
    n_steps = int(duration_s / dt)
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"▶ Initializing Mission Simulation...")
    print(f"  • Mission Duration: {duration_s:.1f} s ({n_steps} steps @ {dt*1000:.0f} ms)")
    print(f"  • Target Trajectory: Circular Aerospace Maneuver at range ~150 m")

    # 1. Trajectory & Sensor
    trajectory = CircularTrajectory(
        center=Position3D(100.0, 0.0, 0.0),
        radius=10.0,
        angular_rate=0.5,
    )
    intrinsics = CameraIntrinsics.from_fov(640, 480, math.radians(60.0))
    beacon = BeaconProperties(intensity=240.0, size_px=8.0)

    # 2. Perception (Phase 2 & 3: Synthetic Renderer + Hybrid BeaconNet Detector)
    render_cfg = RenderConfig(
        noise_std=10.0,
        shot_noise=True,
        num_stars=20,
        enable_bloom=True,
        seed=101,
    )
    renderer = SyntheticRenderer(intrinsics, render_cfg)
    model_path = "models/beaconnet_v1.pt" if Path("models/beaconnet_v1.pt").exists() else None
    hybrid_detector = HybridDetector(HybridConfig(
        ai_model_path=model_path,
        ai_confidence_threshold=0.75,
        classical_config=ClassicalDetectorConfig(
            min_area_px=3.0,
            max_area_px=900.0,
            min_brightness=60.0,
            confidence_threshold=0.35,
        ),
        classical_weight=0.8,
        ai_weight=0.2,
        agreement_threshold_px=25.0,
    ))

    # 3. Estimation & Control (Phase 4: EKF + PID + 2-DOF Gimbal)
    kf = KalmanFilter(KalmanConfig(motion_model=MotionModelType.CONSTANT_VELOCITY, q_process=15.0, r_std=2.0, gate_threshold=36.0))
    pid = PIDController(PIDConfig(kp=7.5, ki=1.0, kd=0.20, output_limit=2.0))
    gimbal = VirtualGimbal()

    # Initialize gimbal pointed directly at target initial position
    init_tgt = trajectory.position(0.0).as_array()
    init_pan = math.atan2(init_tgt[1], init_tgt[0])
    init_tilt = math.atan2(init_tgt[2], math.hypot(init_tgt[0], init_tgt[1]))
    gimbal.set_position(init_pan, init_tilt)

    # 4. Disturbances & Supervisory State Machine (Phase 5)
    jitter = PlatformJitter(JitterConfig(
        random_std_rad=0.0001,
        correlation_time_s=0.05,
        seed=123,
    ))

    atmosphere = AtmosphericTurbulence(AtmosphericConfig(
        cn2=5e-14,
        link_distance_m=100.0,
        beam_wander_std_rad=0.0003,
        scintillation_index=0.2,
        seed=456,
    ))

    # Cloud disturbance dropout between t = 2.4s and t = 2.7s (0.30 seconds transient outage)
    cloud = OcclusionModel(OcclusionConfig(
        intervals=[(2.4, 2.7)],
        transition_duration_s=0.05,
    ))

    pat_sm = PATStateMachine(PATConfig(
        loss_misses_threshold=3,
        coast_time_s=0.3,
        confirm_detections_threshold=2,
        acquisition_lock_error_rad=0.08,
        spiral_step_rad=0.07,
        spiral_angular_speed=1.2,
    ))
    pat_sm.set_state(PATState.TRACKING)

    # 5. Optical Communications Link Budget (Phase 6)
    # Divergence ~ 6.2 mrad, Tx power = 1.0 W, sensitivity = -38 dBm
    link_budget = FSOCLinkBudget(
        laser=LaserConfig(power_w=1.0, wavelength_m=1550e-9, beam_waist_w0_m=0.00008),
        receiver=OpticalReceiverConfig(aperture_diameter_m=0.10, detector_sensitivity_dbm=-38.0),
    )

    print("  ✓ Phase 1: 3D Target Kinematics & 2-DOF Gimbal Loaded")
    print("  ✓ Phase 2 & 3: Synthetic Sensor, Shot Noise, BeaconNet AI Hybrid Detector Loaded")
    print("  ✓ Phase 4: Extended Kalman Filter & PID Actuator Controller Loaded")
    print("  ✓ Phase 5: Satellite Jitter, Turbulence, Cloud Occlusion [2.4s-3.8s], PAT State Machine Loaded")
    print("  ✓ Phase 6: Gaussian Beam Diffraction Optics & Dynamic FSOC Link Budget Loaded")
    print("\n▶ Executing Closed-Loop Mission Simulation...")

    snapshots: list[dict] = []

    for i in range(n_steps):
        t = i * dt

        t_step_start = time.perf_counter()

        # Ground truth target position
        target_pos = trajectory.position(t).as_array()
        dist_m = float(np.linalg.norm(target_pos))

        # 1. Disturbances (Platform vibration jitter & Atmospheric beam wander)
        jit_pan, jit_tilt = jitter.compute(t, dt)
        wander_pan, wander_tilt, scint_factor = atmosphere.compute(dt)
        opt_trans = cloud.transmission(t)

        # 2. Camera Orientation (Gimbal pointing + instantaneous base jitter + beam wander)
        g_state = gimbal.state
        cam_state = CameraState(
            position=Position3D(0, 0, 0),
            orientation=Orientation(
                yaw=g_state.pan + jit_pan + wander_pan,
                pitch=g_state.tilt + jit_tilt + wander_tilt,
                roll=0.0,
            ),
            angular_state=AngularState(pan_rate=g_state.pan_rate, tilt_rate=g_state.tilt_rate),
        )

        # Ground truth projection
        gt_pixel = project_point_to_pixel(target_pos, cam_state, intrinsics)
        if gt_pixel is not None:
            gt_err = compute_angular_error_from_pixel(gt_pixel, intrinsics)
            gt_err_rad = gt_err.total
            gt_u, gt_v = gt_pixel.u, gt_pixel.v
        else:
            gt_err_rad = float("nan")
            gt_u, gt_v = float("nan"), float("nan")

        # Atmospheric scintillation and cloud transmission effects on beacon
        if opt_trans <= 0.05:
            scint_factor = 0.0
        eff_intensity = max(0.0, min(255.0, beacon.intensity * opt_trans * scint_factor))
        effective_beacon = BeaconProperties(
            intensity=eff_intensity,
            size_px=beacon.size_px,
        )

        # Render sensor frame
        img = renderer.render(cam_state, [(target_pos, effective_beacon)])

        # Detect beacon (Hybrid BeaconNet + Classical)
        det = hybrid_detector.detect(img)
        beacon_detected = det.detected and (opt_trans > 0.05) and (det.confidence >= 0.40)

        det_u = det.best.u if beacon_detected and det.best else float("nan")
        det_v = det.best.v if beacon_detected and det.best else float("nan")

        # Estimation (EKF)
        prev_state = pat_sm.state
        if beacon_detected and det.best is not None:
            meas = np.array([det.best.u, det.best.v])
            if not kf.initialized or prev_state == PATState.REACQUISITION:
                kf.initialize(meas, dt)
            else:
                kf.predict(dt)
                kf.update(meas)
        elif kf.initialized:
            kf.predict(dt)

        est_u, est_v = kf.estimated_position
        if kf.initialized and not math.isnan(est_u):
            est_px = PixelCoord(u=est_u, v=est_v)
            ang_err = compute_angular_error_from_pixel(est_px, intrinsics)
        else:
            ang_err = AngularError(horizontal=0.0, vertical=0.0, total=0.0)

        # PID Control Command
        # Invert vertical error because Z-down camera convention means positive vertical error
        # requires negative gimbal tilt rate to center the target
        ctrl_err = AngularError(horizontal=ang_err.horizontal, vertical=-ang_err.vertical, total=ang_err.total)
        if beacon_detected:
            pid_cmd = pid.compute(ctrl_err, dt, timestamp=t)
        else:
            pid.reset()
            pid_cmd = None

        # Supervisory PAT State Machine
        pan_rate, tilt_rate = pat_sm.step(
            dt=dt,
            detected=beacon_detected,
            angular_error=ang_err,
            gimbal_pan=g_state.pan,
            gimbal_tilt=g_state.tilt,
            pid_cmd=pid_cmd,
            timestamp=t,
        )

        # Step virtual gimbal dynamics
        gimbal.step(pan_rate, tilt_rate, dt)

        # Optical Communications Link Budget Evaluation
        eval_err = gt_err_rad if not math.isnan(gt_err_rad) else 0.005
        # Link budget under true physical pointing error
        link_res = link_budget.evaluate(dist_m, eval_err)

        # Cloud occlusion introduces extinction attenuation
        cloud_loss_db = -10.0 * math.log10(max(1e-10, opt_trans)) if opt_trans > 0.0 else 100.0
        if opt_trans > 0.05:
            rx_pwr = max(-100.0, link_res.rx_power_dbm - cloud_loss_db)
            link_marg = rx_pwr - (-38.0)
            snr_db = max(0.0, link_res.snr_db - cloud_loss_db)
            q_factor = math.sqrt(10.0 ** (snr_db / 10.0)) / 2.0
            ber = 0.5 * math.erfc(q_factor / math.sqrt(2.0)) if q_factor < 8.0 else 1e-15
        else:
            rx_pwr = -100.0
            link_marg = -62.0
            snr_db = 0.0
            q_factor = 0.0
            ber = 0.5  # Signal blocked by cloud

        t_step_end = time.perf_counter()
        step_proc_ms = (t_step_end - t_step_start) * 1000.0

        if not math.isnan(det_u) and not math.isnan(gt_u):
            c_err_px = math.hypot(det_u - gt_u, det_v - gt_v)
        elif not math.isnan(est_u) and not math.isnan(gt_u):
            c_err_px = math.hypot(est_u - gt_u, est_v - gt_v)
        else:
            c_err_px = float("nan")

        t_err_px = math.hypot(gt_u - 320.0, gt_v - 240.0) if not math.isnan(gt_u) else float("nan")

        snapshots.append({
            "time": round(t, 4),
            "gt_u": round(gt_u, 2),
            "gt_v": round(gt_v, 2),
            "det_u": round(det_u, 2),
            "det_v": round(det_v, 2),
            "est_u": round(est_u, 2),
            "est_v": round(est_v, 2),
            "error_h": round(ang_err.horizontal, 6),
            "error_v": round(ang_err.vertical, 6),
            "error_total": round(ang_err.total, 6),
            "gt_error_total": round(gt_err_rad, 6),
            "centroid_error_px": round(c_err_px, 2) if not math.isnan(c_err_px) else None,
            "tracking_error_px": round(t_err_px, 2) if not math.isnan(t_err_px) else None,
            "processing_time_ms": round(step_proc_ms, 2),
            "pan_cmd": round(pan_rate, 5),
            "tilt_cmd": round(tilt_rate, 5),
            "gimbal_pan": round(g_state.pan, 5),
            "gimbal_tilt": round(g_state.tilt, 5),
            "gimbal_pan_rate": round(g_state.pan_rate, 5),
            "gimbal_tilt_rate": round(g_state.tilt_rate, 5),
            "confidence": round(det.confidence if beacon_detected else 0.0, 3),
            "detected": beacon_detected,
            "pat_state": pat_sm.state.name,
            "target_x": round(float(target_pos[0]), 2),
            "target_y": round(float(target_pos[1]), 2),
            "target_z": round(float(target_pos[2]), 2),
            "distance_m": round(dist_m, 2),
            "jitter_pan_urad": round(jit_pan * 1e6, 2),
            "jitter_tilt_urad": round(jit_tilt * 1e6, 2),
            "wander_pan_urad": round(wander_pan * 1e6, 2),
            "wander_tilt_urad": round(wander_tilt * 1e6, 2),
            "scintillation_factor": round(scint_factor, 3),
            "cloud_transmissivity": round(opt_trans, 3),
            "cloud_loss_db": round(cloud_loss_db, 2),
            "free_space_loss_db": round(link_res.free_space_loss_db, 2),
            "atmospheric_loss_db": round(link_res.atmospheric_loss_db, 2),
            "rx_power_dbm": round(rx_pwr, 2),
            "pointing_loss_db": round(link_res.pointing_loss_db, 2),
            "link_margin_db": round(link_marg, 2),
            "snr_db": round(snr_db, 2),
            "q_factor": round(q_factor, 2),
            "ber": ber,
        })

    t_wall = time.perf_counter() - t_wall_start
    print(f"\n[OK] Simulation completed in {t_wall:.2f} s wall time.")

    # 6. Export Telemetry & Generate Reports (Phase 7)
    print("\n▶ Generating Phase 7 Automated Mission Reports...")
    import csv
    csv_file = out_dir / "mission_telemetry.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(snapshots[0].keys()))
        writer.writeheader()
        writer.writerows(snapshots)
    print(f"  ✓ Exported CSV Telemetry: {csv_file}")

    rep_gen = SimulationReportGenerator(snapshots, scenario_name="OptiLign Aerospace FSOC PAT Mission")
    m = rep_gen.metrics

    # JSON Summary
    json_file = out_dir / "mission_summary.json"
    rep_gen.generate_json_summary(json_file)
    print(f"  ✓ Exported JSON Summary: {json_file}")

    # Markdown Report
    md_file = out_dir / "mission_report.md"
    rep_gen.generate_markdown_report(md_file)
    print(f"  ✓ Exported Markdown Report: {md_file}")

    # HTML Interactive Report
    html_file = out_dir / "mission_report.html"
    rep_gen.generate_html_report(html_file, title="OptiLign FSOC PAT Mission Verification Report")
    print(f"  ✓ Exported Interactive HTML Report: {html_file}")

    # 6-Panel Telemetry Plot
    png_file = Path("demo_full_system.png")
    generate_mission_summary_plot(snapshots, output_path=png_file, scenario_name="OptiLign FSOC PAT Mission")
    print(f"  ✓ Exported 6-Panel Telemetry Plot: {png_file}")

    # Copy plot to brain artifact directory as well
    artifact_dir = Path("/Users/sanju/.gemini/antigravity-ide/brain/f9737e30-4833-44bb-86f3-2b91df19d396")
    if artifact_dir.exists():
        import shutil
        shutil.copy(png_file, artifact_dir / "demo_full_system.png")
        shutil.copy(md_file, artifact_dir / "demo_mission_report.md")

    print("\n" + "=" * 78)
    print("                    MISSION RESULTS EXECUTIVE SUMMARY")
    print("=" * 78)
    print(f"  • Pointing Accuracy (MRE):      {m.pointing.mean_radial_error_mrad:.3f} mrad ({m.pointing.mean_error_deg:.4f}°)")
    print(f"  • Pointing Jitter (RMS):        {m.pointing.rms_radial_error_mrad:.3f} mrad")
    print(f"  • 95th Percentile Error:        {m.pointing.p95_error_mrad:.3f} mrad")
    print(f"  • Closed-Loop Tracking Duty:    {m.pat_state.percent_in_tracking:.1f}%")
    print(f"  • Reacquisition Time (TTR):     {m.pat_state.mean_time_to_reacquire_s:.2f} s")
    print(f"  • Optical Link Availability:    {m.link.link_availability_pct:.1f}% (Margin >= 3.0 dB)")
    print(f"  • Mean Link Margin:             {m.link.mean_link_margin_db:.2f} dB")
    print(f"  • Mean Received Power:          {m.link.mean_rx_power_dbm:.2f} dBm")
    print(f"  • Mean Bit Error Rate (BER):    {m.link.mean_ber:.2e}")
    print("=" * 78)


if __name__ == "__main__":
    run_master_mission()
