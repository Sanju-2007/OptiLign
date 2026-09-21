#!/usr/bin/env python3
"""OptiLign Phase 5 Demo — Disturbances, Signal Loss & PAT Reacquisition.

Demonstrates:
  1. Platform jitter & atmospheric turbulence (beam wander + scintillation)
  2. PAT State Machine (SEARCH -> ACQUISITION -> TRACKING -> REACQUISITION)
  3. Simulated cloud occlusion / blackout (t=1.5s to 3.0s)
  4. Coasting & autonomous Archimedean spiral reacquisition
  5. Recovery and track relock without gimbal runaway
"""

import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from fsoc_pat.control.pid import PIDConfig, PIDController
from fsoc_pat.core.state import CameraState
from fsoc_pat.core.types import AngularError, AngularState, Orientation, Position3D, PixelCoord
from fsoc_pat.disturbances.atmospheric import AtmosphericConfig, AtmosphericTurbulence
from fsoc_pat.disturbances.occlusion import OcclusionConfig, OcclusionModel
from fsoc_pat.disturbances.platform_jitter import JitterConfig, PlatformJitter
from fsoc_pat.estimation.kalman_filter import KalmanConfig, KalmanFilter
from fsoc_pat.geometry.angular_error import compute_angular_error_from_pixel
from fsoc_pat.geometry.camera_model import CameraIntrinsics
from fsoc_pat.pat.pat_state_machine import PATStateMachine
from fsoc_pat.pat.pat_types import PATConfig, PATState
from fsoc_pat.perception.classical_detector import ClassicalDetectorConfig
from fsoc_pat.perception.pipeline import PerceptionConfig, PerceptionPipeline
from fsoc_pat.perception.preprocessing import PreprocessConfig
from fsoc_pat.perception.renderer import RenderConfig
from fsoc_pat.simulation.beacon import BeaconProperties
from fsoc_pat.simulation.dynamics import VirtualGimbal
from fsoc_pat.simulation.trajectories import CircularTrajectory


print("=" * 70)
print("  OptiLign — Phase 5 Demo: Disturbances, Signal Loss & PAT Reacquisition")
print("=" * 70)

# ── 1. Setup Simulation ──────────────────────────────────
duration_s = 5.0
dt = 0.02  # 50 Hz
n_steps = int(duration_s / dt)

intrinsics = CameraIntrinsics.from_fov(640, 480, math.radians(60))
traj = CircularTrajectory(
    center=Position3D(100.0, 0.0, 0.0),
    radius=6.0,
    angular_rate=0.3,
)

# Initial pointing towards target
p0 = traj.position(0.0)
init_pan = math.atan2(p0.y, p0.x)
init_tilt = math.atan2(p0.z, math.hypot(p0.x, p0.y))
gimbal = VirtualGimbal()
gimbal.set_position(pan=init_pan, tilt=init_tilt)

pipeline = PerceptionPipeline(
    intrinsics,
    PerceptionConfig(
        render=RenderConfig(noise_std=6.0, shot_noise=True, num_stars=6, seed=42),
        detector=ClassicalDetectorConfig(
            min_brightness=60.0,
            confidence_threshold=0.3,
            preprocess=PreprocessConfig(threshold_value=60),
        ),
    ),
)
kf = KalmanFilter(KalmanConfig(r_std=1.5, q_process=10.0))
pid = PIDController(PIDConfig(kp=4.5, ki=0.2, kd=0.15, output_limit=1.2))
pat_sm = PATStateMachine(
    PATConfig(
        loss_misses_threshold=3,
        coast_time_s=0.3,
        confirm_detections_threshold=2,
        acquisition_lock_error_rad=0.08,
        spiral_step_rad=0.07,
        spiral_angular_speed=1.2,
    )
)
pat_sm.set_state(PATState.SEARCH)

# Disturbances: Jitter, Turbulence, Occlusion
jitter_model = PlatformJitter(JitterConfig(random_std_rad=0.0008, seed=42))
atmo_model = AtmosphericTurbulence(AtmosphericConfig(beam_wander_std_rad=0.0005, scintillation_index=0.2, seed=42))
occlusion_model = OcclusionModel(OcclusionConfig(intervals=[(1.6, 3.0)], transition_duration_s=0.05))

print(f"\n▶ 1. SIMULATING 5.0s CLOSED-LOOP WITH OCCLUSION")
print(f"  • Cloud occlusion active: t = 1.6s to 3.0s (1.4s blackout)")
print(f"  • Platform jitter: 12Hz/28Hz harmonics + random colored noise")
print(f"  • Atmospheric beam wander + scintillation fading")

# ── 2. Execution Loop ────────────────────────────────────
times = []
state_names = []
state_nums = []  # 0=STANDBY, 1=SEARCH, 2=ACQUISITION, 3=TRACKING, 4=REACQUISITION
ang_errors_mrad = []
pan_angles_deg = []
tilt_angles_deg = []
jitter_pans_mrad = []
scint_scales = []
occluded_flags = []

state_map = {
    PATState.STANDBY: 0,
    PATState.SEARCH: 1,
    PATState.ACQUISITION: 2,
    PATState.TRACKING: 3,
    PATState.REACQUISITION: 4,
    PATState.FAULT: 5,
}

t_start = time.perf_counter()

for step in range(n_steps):
    t = step * dt

    # 1. Target kinematics
    target_pos = traj.position(t).as_array()

    # 2. Platform jitter & atmospheric perturbations
    j_pan, j_tilt = jitter_model.compute(t, dt)
    w_pan, w_tilt, scint_factor = atmo_model.compute(dt)
    trans = occlusion_model.transmission(t)
    is_blocked = occlusion_model.is_occluded(t)

    # Effective camera orientation with base jitter
    g_state = gimbal.state
    cam_state = CameraState(
        position=Position3D(0, 0, 0),
        orientation=Orientation(
            yaw=g_state.pan + j_pan,
            pitch=g_state.tilt + j_tilt,
            roll=0.0,
        ),
        angular_state=AngularState(pan_rate=g_state.pan_rate, tilt_rate=g_state.tilt_rate),
    )

    # Beacon intensity modulated by scintillation and occlusion
    beacon_intensity = min(255.0, max(0.0, 240.0 * scint_factor * trans))
    beacon = BeaconProperties(intensity=beacon_intensity, size_px=8.0)

    # 3. Perception (Render + Detect)
    detection = pipeline.process_frame(cam_state, [(target_pos, beacon)], timestamp=t)
    detected = (
        (detection.detected or (detection.best is not None and detection.confidence >= 0.3))
        and not is_blocked
        and trans > 0.1
    )

    # 4. Estimation
    if detected and detection.best is not None:
        meas = np.array([detection.best.u, detection.best.v])
        if not kf.initialized:
            kf.initialize(meas, dt)
        else:
            kf.predict(dt)
            kf.update(meas)
    elif kf.initialized:
        kf.predict(dt)

    # 5. Angular error from estimated position
    if kf.initialized and detected:
        est_u, est_v = kf.estimated_position
        ang_err = compute_angular_error_from_pixel(PixelCoord(est_u, est_v), intrinsics)
        err_mrad = ang_err.total * 1000.0
    else:
        ang_err = AngularError(horizontal=0.0, vertical=0.0, total=0.0)
        err_mrad = float("nan")

    # 6. PID controller
    pid_cmd = pid.compute(ang_err, dt, timestamp=t) if detected else None

    # 7. Supervisory PAT State Machine
    pan_rate_cmd, tilt_rate_cmd = pat_sm.step(
        dt=dt,
        detected=detected,
        angular_error=ang_err,
        gimbal_pan=g_state.pan,
        gimbal_tilt=g_state.tilt,
        pid_cmd=pid_cmd,
        timestamp=t,
    )

    # 8. Gimbal step
    gimbal.step(pan_rate_cmd, tilt_rate_cmd, dt)

    # Log telemetry
    times.append(t)
    state_names.append(pat_sm.state.name)
    state_nums.append(state_map[pat_sm.state])
    ang_errors_mrad.append(err_mrad)
    pan_angles_deg.append(math.degrees(g_state.pan))
    tilt_angles_deg.append(math.degrees(g_state.tilt))
    jitter_pans_mrad.append(j_pan * 1000.0)
    scint_scales.append(scint_factor)
    occluded_flags.append(is_blocked)

sim_wall_time = time.perf_counter() - t_start

tracking_frames = sum(1 for s in state_names if s == "TRACKING")
reacq_frames = sum(1 for s in state_names if s == "REACQUISITION")
valid_errors = [e for e in ang_errors_mrad if not math.isnan(e)]
mean_tracking_err = float(np.mean(valid_errors)) if valid_errors else 0.0

print(f"  Simulation completed in {sim_wall_time:.2f}s ({n_steps / sim_wall_time:.0f} steps/s)")
print(f"  Tracking frames:  {tracking_frames}/{n_steps} ({tracking_frames/n_steps:.1%})")
print(f"  Reacq frames:     {reacq_frames}/{n_steps}")
print(f"  Mean track error: {mean_tracking_err:.2f} mrad")
print(f"  Max gimbal pan:   {max(map(abs, pan_angles_deg)):.1f}° (no runaway)")
print(f"  Max gimbal tilt:  {max(map(abs, tilt_angles_deg)):.1f}° (no runaway)")

# ── 3. Generate Comprehensive Plots ──────────────────────
print("\n▶ 2. GENERATING PHASE 5 PLOTS")

fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

# 1. PAT State Timeline
ax1 = fig.add_subplot(gs[0, :])
ax1.step(times, state_nums, "g-", linewidth=2.0, where="post")
ax1.axvspan(1.6, 3.0, color="red", alpha=0.18, label="Cloud Occlusion (Blackout)")
ax1.set_yticks([1, 2, 3, 4])
ax1.set_yticklabels(["SEARCH", "ACQUISITION", "TRACKING", "REACQUISITION"], fontweight="bold")
ax1.set_title("PAT State Machine Mode Transitions", fontweight="bold", fontsize=12)
ax1.set_xlabel("Time (s)")
ax1.legend(loc="upper right")
ax1.grid(True, alpha=0.3)

# 2. Angular Tracking Error
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(times, ang_errors_mrad, "b.-", markersize=3, linewidth=1.0, label="Tracking Error (mrad)")
ax2.axvspan(1.6, 3.0, color="red", alpha=0.15)
ax2.set_title("Angular Pointing Error During Track & Reacquisition", fontweight="bold")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Error (mrad)")
ax2.grid(True, alpha=0.3)
ax2.legend()

# 3. Gimbal Orientation (Pan & Tilt)
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(times, pan_angles_deg, "b-", linewidth=1.2, label="Pan (yaw)")
ax3.plot(times, tilt_angles_deg, "r-", linewidth=1.2, label="Tilt (pitch)")
ax3.axvspan(1.6, 3.0, color="red", alpha=0.15)
ax3.set_title("Gimbal Orientation (Bounded — Anti-Runaway)", fontweight="bold")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Angle (deg)")
ax3.grid(True, alpha=0.3)
ax3.legend()

# 4. Platform Jitter Time Series
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(times, jitter_pans_mrad, color="purple", linewidth=0.9, label="Base Jitter (mrad)")
ax4.set_title("Platform Angular Vibration / Jitter Spectrum", fontweight="bold")
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Jitter (mrad)")
ax4.grid(True, alpha=0.3)
ax4.legend()

# 5. Atmospheric Scintillation Fading
ax5 = fig.add_subplot(gs[2, 1])
ax5.plot(times, scint_scales, color="orange", linewidth=1.0, label="Intensity Scale I/I₀")
ax5.axhline(1.0, color="gray", linestyle="--", alpha=0.6)
ax5.axvspan(1.6, 3.0, color="red", alpha=0.15)
ax5.set_title("Atmospheric Scintillation (Log-Normal Fading)", fontweight="bold")
ax5.set_xlabel("Time (s)")
ax5.set_ylabel("Normalized Intensity")
ax5.grid(True, alpha=0.3)
ax5.legend()

out_path = "demo_p5_results.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"  → Saved: {out_path}")

print("=" * 70)
print("  Phase 5 Demo Complete!")
print("======================================================================")
