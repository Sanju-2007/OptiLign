#!/usr/bin/env python3
"""OptiLign Phase 1 Demo — What the system can do right now.

This script demonstrates every Phase 1 capability:
  1. Trajectory generation (5 types)
  2. Camera intrinsics & projection
  3. Gimbal dynamics (commanded ≠ actual)
  4. Full simulation engine run
  5. Ground-truth CSV export
  6. Angular error analysis
  7. Matplotlib visualization of results
"""

import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ──────────────────────────────────────────────────────────────
# 1. TRAJECTORY SHOWCASE
# ──────────────────────────────────────────────────────────────
from fsoc_pat.core.types import Position3D, Velocity3D, Acceleration3D
from fsoc_pat.simulation.trajectories import (
    StationaryTrajectory,
    LinearTrajectory,
    CircularTrajectory,
    SinusoidalTrajectory,
    SpiralTrajectory,
)

print("=" * 70)
print("  OptiLign — Phase 1 Demonstration")
print("  FSOC AI-Assisted Virtual Camera Coarse PAT Simulator")
print("=" * 70)

print("\n▶ 1. TRAJECTORY GENERATION")
print("  OptiLign supports 8 trajectory types with analytical derivatives.")
print("  Demonstrating 5 of them:\n")

trajectories = {
    "Stationary": StationaryTrajectory(pos=Position3D(100, 0, 0)),
    "Linear": LinearTrajectory(p0=Position3D(100, -20, 0), vel=Velocity3D(0, 4, 0)),
    "Circular": CircularTrajectory(center=Position3D(100, 0, 0), radius=10, angular_rate=1.0),
    "Sinusoidal": SinusoidalTrajectory(
        center=Position3D(100, 0, 0),
        amplitude=Position3D(0, 15, 5),
        frequency=Position3D(0, 0.8, 1.2),
    ),
    "Spiral": SpiralTrajectory(
        center=Position3D(100, 0, 0), initial_radius=2, expansion_rate=1.5, angular_rate=2.0,
    ),
}

fig_traj, axes_traj = plt.subplots(1, 5, figsize=(22, 4), subplot_kw={"aspect": "equal"})
times = np.linspace(0, 10, 500)

for idx, (name, traj) in enumerate(trajectories.items()):
    ys, zs = [], []
    for t in times:
        p = traj.position(float(t))
        ys.append(p.y)
        zs.append(p.z)
    axes_traj[idx].plot(ys, zs, linewidth=1.5, color=plt.cm.viridis(idx / 5))
    axes_traj[idx].plot(ys[0], zs[0], "go", markersize=8, label="start")
    axes_traj[idx].plot(ys[-1], zs[-1], "rs", markersize=8, label="end")
    axes_traj[idx].set_title(name, fontsize=12, fontweight="bold")
    axes_traj[idx].set_xlabel("Y (m)")
    axes_traj[idx].set_ylabel("Z (m)")
    axes_traj[idx].grid(True, alpha=0.3)
    axes_traj[idx].legend(fontsize=8)
    print(f"  ✓ {name:15s} — t=0: ({traj.position(0).x:.1f}, {traj.position(0).y:.1f}, {traj.position(0).z:.1f})m")

fig_traj.suptitle("Target Trajectory Types (Y-Z plane, 10s)", fontsize=14, fontweight="bold")
fig_traj.tight_layout()
fig_traj.savefig("demo_trajectories.png", dpi=150, bbox_inches="tight")
print(f"\n  → Saved: demo_trajectories.png")

# ──────────────────────────────────────────────────────────────
# 2. CAMERA INTRINSICS & PROJECTION
# ──────────────────────────────────────────────────────────────
from fsoc_pat.geometry.camera_model import CameraIntrinsics
from fsoc_pat.geometry.projection import project_point_to_pixel, is_in_fov, pixel_to_ray
from fsoc_pat.core.state import CameraState
from fsoc_pat.core.types import Orientation, AngularState, PixelCoord

print("\n▶ 2. CAMERA MODEL & PROJECTION")
intrinsics = CameraIntrinsics.from_fov(640, 480, math.radians(60))
print(f"  Camera: {intrinsics.width}×{intrinsics.height}px, "
      f"HFOV={intrinsics.hfov_deg:.1f}°, VFOV={intrinsics.vfov_deg:.1f}°")
print(f"  Focal lengths: fx={intrinsics.fx:.1f}px, fy={intrinsics.fy:.1f}px")
print(f"  Principal point: ({intrinsics.cx:.1f}, {intrinsics.cy:.1f})")
print(f"  Angular resolution: {intrinsics.pixel_angular_resolution * 1000:.2f} mrad/px")

cam_state = CameraState(
    position=Position3D(0, 0, 0),
    orientation=Orientation(),
    angular_state=AngularState(),
)

# Project a grid of points
print("\n  Projection examples (target at 100m range):")
test_points = [
    ("On boresight",     [100, 0, 0]),
    ("10m right",        [100, 10, 0]),
    ("10m left",         [100, -10, 0]),
    ("10m down (Z+)",    [100, 0, 10]),
    ("10m right + down", [100, 10, 10]),
    ("Behind camera",    [-100, 0, 0]),
]
for label, pt in test_points:
    px = project_point_to_pixel(np.array(pt, dtype=np.float64), cam_state, intrinsics)
    if px:
        fov = "IN FOV" if is_in_fov(px, intrinsics) else "OUTSIDE FOV"
        print(f"    {label:20s} → pixel ({px.u:7.1f}, {px.v:7.1f})  [{fov}]")
    else:
        print(f"    {label:20s} → NOT VISIBLE (behind camera)")

# ──────────────────────────────────────────────────────────────
# 3. GIMBAL DYNAMICS DEMO
# ──────────────────────────────────────────────────────────────
from fsoc_pat.simulation.dynamics import GimbalDynamics, VirtualGimbal

print("\n▶ 3. GIMBAL DYNAMICS (commanded ≠ actual)")
dynamics = GimbalDynamics(
    pan_rate_limit=1.0,
    tilt_rate_limit=1.0,
    angular_accel_limit=10.0,
    actuator_time_constant=0.05,
    deadband=0.001,
    noise_std=0.0,
)
gimbal = VirtualGimbal(dynamics=dynamics)

dt = 0.01
cmd_rates, actual_rates, positions = [], [], []
commanded = 0.8  # rad/s

for i in range(300):
    t = i * dt
    # Step command at t=0.5s, reverse at t=1.5s
    if t < 0.5:
        cmd = 0.0
    elif t < 1.5:
        cmd = commanded
    else:
        cmd = -commanded

    gimbal.step(cmd, 0.0, dt)
    cmd_rates.append(cmd)
    actual_rates.append(gimbal.state.pan_rate)
    positions.append(gimbal.state.pan)

fig_gimbal, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
t_arr = np.arange(300) * dt

ax1.plot(t_arr, cmd_rates, "r--", label="Commanded rate", linewidth=2)
ax1.plot(t_arr, actual_rates, "b-", label="Actual rate", linewidth=2)
ax1.set_ylabel("Pan rate (rad/s)")
ax1.set_title("Gimbal Dynamics — Commanded vs Actual", fontweight="bold")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(t_arr, [math.degrees(p) for p in positions], "g-", linewidth=2)
ax2.set_ylabel("Pan position (deg)")
ax2.set_xlabel("Time (s)")
ax2.grid(True, alpha=0.3)

fig_gimbal.tight_layout()
fig_gimbal.savefig("demo_gimbal.png", dpi=150, bbox_inches="tight")
print(f"  Actuator lag: {dynamics.actuator_time_constant * 1000:.0f}ms time constant")
print(f"  Rate limit: ±{dynamics.pan_rate_limit} rad/s")
print(f"  Accel limit: {dynamics.angular_accel_limit} rad/s²")
print(f"  Peak actual rate reached: {max(actual_rates):.3f} rad/s (commanded {commanded})")
print(f"  → Saved: demo_gimbal.png")

# ──────────────────────────────────────────────────────────────
# 4. FULL ENGINE RUN
# ──────────────────────────────────────────────────────────────
from fsoc_pat.config.models import ExperimentConfig, ScenarioConfig, TargetConfig, CameraConfig
from fsoc_pat.simulation.engine import SimulationEngine

print("\n▶ 4. FULL SIMULATION ENGINE")
config = ExperimentConfig(
    scenario=ScenarioConfig(name="demo", duration_s=10.0, timestep_s=0.01, seed=42),
    camera=CameraConfig(width=640, height=480, hfov_rad=math.radians(60)),
    target=TargetConfig(
        trajectory_type="circular",
        position=(100.0, 0.0, 0.0),
        radius=10.0,
        angular_rate=1.0,
    ),
)

engine = SimulationEngine(config)
engine.initialize()
results = engine.run()

print(f"  Scenario: {config.scenario.name}")
print(f"  Duration: {results.duration_s:.1f}s, Steps: {results.total_steps}")
print(f"  Wall time: {results.wall_time_s * 1000:.1f}ms ({results.total_steps / results.wall_time_s:.0f} steps/s)")
print(f"  Real-time factor: {results.duration_s / results.wall_time_s:.0f}×")
print(f"  Ground truth snapshots: {len(results.ground_truth)}")

# ──────────────────────────────────────────────────────────────
# 5. ANGULAR ERROR ANALYSIS
# ──────────────────────────────────────────────────────────────
print("\n▶ 5. ANGULAR ERROR ANALYSIS")

times_gt = []
errors_h, errors_v, errors_total = [], [], []
pixels_u, pixels_v = [], []
target_y, target_z = [], []
in_fov_count = 0

for snap in results.ground_truth.snapshots:
    times_gt.append(snap.time.time_s)
    target_y.append(snap.target.position.y)
    target_z.append(snap.target.position.z)

    if snap.true_angular_error is not None:
        errors_h.append(math.degrees(snap.true_angular_error.horizontal))
        errors_v.append(math.degrees(snap.true_angular_error.vertical))
        errors_total.append(math.degrees(snap.true_angular_error.total))
    else:
        errors_h.append(0)
        errors_v.append(0)
        errors_total.append(0)

    if snap.true_pixel is not None:
        pixels_u.append(snap.true_pixel.u)
        pixels_v.append(snap.true_pixel.v)
    else:
        pixels_u.append(float("nan"))
        pixels_v.append(float("nan"))

    if snap.target_in_fov:
        in_fov_count += 1

print(f"  Mean angular error:  {np.mean(errors_total):.4f}°")
print(f"  Max angular error:   {max(errors_total):.4f}°")
print(f"  Min angular error:   {min(errors_total):.4f}°")
print(f"  Frames in FOV:       {in_fov_count}/{len(results.ground_truth)} "
      f"({100 * in_fov_count / len(results.ground_truth):.1f}%)")

# ──────────────────────────────────────────────────────────────
# 6. COMPREHENSIVE VISUALIZATION
# ──────────────────────────────────────────────────────────────
print("\n▶ 6. GENERATING COMPREHENSIVE PLOTS")

fig = plt.figure(figsize=(18, 14))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# Target trajectory in world Y-Z
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(target_y, target_z, "b-", linewidth=1.5)
ax1.plot(target_y[0], target_z[0], "go", markersize=10, label="Start")
ax1.plot(target_y[-1], target_z[-1], "rs", markersize=10, label="End")
ax1.set_title("Target Trajectory (World Y-Z)", fontweight="bold")
ax1.set_xlabel("Y (m)")
ax1.set_ylabel("Z (m)")
ax1.set_aspect("equal")
ax1.grid(True, alpha=0.3)
ax1.legend()

# Pixel trace on image plane
ax2 = fig.add_subplot(gs[0, 1])
scatter = ax2.scatter(pixels_u, pixels_v, c=times_gt, cmap="viridis", s=2, alpha=0.7)
ax2.set_xlim(0, 640)
ax2.set_ylim(480, 0)  # Image convention: y increases downward
ax2.set_title("Pixel Trace on Image Plane", fontweight="bold")
ax2.set_xlabel("u (px)")
ax2.set_ylabel("v (px)")
ax2.axhline(240, color="gray", linestyle="--", alpha=0.3)
ax2.axvline(320, color="gray", linestyle="--", alpha=0.3)
plt.colorbar(scatter, ax=ax2, label="Time (s)")
ax2.grid(True, alpha=0.3)

# Angular error over time
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(times_gt, errors_total, "r-", linewidth=1.5, label="Total")
ax3.plot(times_gt, errors_h, "b-", linewidth=1, alpha=0.7, label="Horizontal")
ax3.plot(times_gt, errors_v, "g-", linewidth=1, alpha=0.7, label="Vertical")
ax3.set_title("Angular Error vs Time", fontweight="bold")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Error (deg)")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Pixel U over time
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(times_gt, pixels_u, "b-", linewidth=1)
ax4.axhline(320, color="red", linestyle="--", alpha=0.5, label="Center")
ax4.set_title("Pixel U (Horizontal) vs Time", fontweight="bold")
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("u (px)")
ax4.legend()
ax4.grid(True, alpha=0.3)

# Pixel V over time
ax5 = fig.add_subplot(gs[1, 1])
ax5.plot(times_gt, pixels_v, "g-", linewidth=1)
ax5.axhline(240, color="red", linestyle="--", alpha=0.5, label="Center")
ax5.set_title("Pixel V (Vertical) vs Time", fontweight="bold")
ax5.set_xlabel("Time (s)")
ax5.set_ylabel("v (px)")
ax5.legend()
ax5.grid(True, alpha=0.3)

# Target speed
speeds = [snap.target.velocity.magnitude for snap in results.ground_truth.snapshots]
ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(times_gt, speeds, "m-", linewidth=1.5)
ax6.set_title("Target Speed vs Time", fontweight="bold")
ax6.set_xlabel("Time (s)")
ax6.set_ylabel("Speed (m/s)")
ax6.grid(True, alpha=0.3)

# Angular error histogram
ax7 = fig.add_subplot(gs[2, 0])
ax7.hist(errors_total, bins=40, color="coral", edgecolor="black", alpha=0.8)
ax7.axvline(np.mean(errors_total), color="red", linestyle="--", linewidth=2, label=f"Mean={np.mean(errors_total):.3f}°")
ax7.set_title("Angular Error Distribution", fontweight="bold")
ax7.set_xlabel("Error (deg)")
ax7.set_ylabel("Count")
ax7.legend()
ax7.grid(True, alpha=0.3)

# Error components phase plot
ax8 = fig.add_subplot(gs[2, 1])
ax8.plot(errors_h, errors_v, "b-", linewidth=0.8, alpha=0.6)
ax8.plot(errors_h[0], errors_v[0], "go", markersize=10)
ax8.set_title("Error Phase Plot (H vs V)", fontweight="bold")
ax8.set_xlabel("Horizontal Error (deg)")
ax8.set_ylabel("Vertical Error (deg)")
ax8.set_aspect("equal")
ax8.grid(True, alpha=0.3)

# Summary text
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis("off")
summary = (
    f"OptiLign Phase 1 — Simulation Summary\n"
    f"{'─' * 40}\n\n"
    f"Scenario:       {config.scenario.name}\n"
    f"Duration:       {results.duration_s:.1f}s\n"
    f"Timestep:       {config.scenario.timestep_s * 1000:.0f}ms\n"
    f"Steps:          {results.total_steps}\n"
    f"Wall time:      {results.wall_time_s * 1000:.1f}ms\n"
    f"RT factor:      {results.duration_s / results.wall_time_s:.0f}×\n\n"
    f"Camera:         {intrinsics.width}×{intrinsics.height}\n"
    f"HFOV:           {intrinsics.hfov_deg:.1f}°\n"
    f"Trajectory:     Circular\n"
    f"Radius:         {config.target.radius:.0f}m\n"
    f"ω:              {config.target.angular_rate:.1f} rad/s\n\n"
    f"Mean error:     {np.mean(errors_total):.4f}°\n"
    f"Max error:      {max(errors_total):.4f}°\n"
    f"FOV coverage:   {100 * in_fov_count / len(results.ground_truth):.1f}%\n"
    f"Speed:          {np.mean(speeds):.1f} m/s"
)
ax9.text(0.05, 0.95, summary, transform=ax9.transAxes, fontsize=10,
         verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

fig.suptitle("OptiLign — FSOC Coarse PAT Simulator — Phase 1 Results",
             fontsize=16, fontweight="bold", y=0.98)
fig.savefig("demo_results.png", dpi=150, bbox_inches="tight")
print(f"  → Saved: demo_results.png")

# ──────────────────────────────────────────────────────────────
# 7. CSV EXPORT
# ──────────────────────────────────────────────────────────────
print("\n▶ 7. CSV EXPORT")
output_dir = "demo_output"
results.export_csv(output_dir)
csv_path = os.path.join(output_dir, "ground_truth.csv")
line_count = sum(1 for _ in open(csv_path))
print(f"  Exported {line_count - 1} data rows to {csv_path}")
print(f"  Columns: time_s, step, target_xyz, target_vxyz, camera_pan/tilt,")
print(f"           pixel_uv, angular_error_hvt, in_fov")

# ──────────────────────────────────────────────────────────────
# 8. QUATERNION DEMO
# ──────────────────────────────────────────────────────────────
from fsoc_pat.geometry.quaternions import Quaternion, slerp

print("\n▶ 8. QUATERNION ALGEBRA")
q1 = Quaternion.from_euler(0.0, 0.0, 0.0)
q2 = Quaternion.from_euler(0.0, 0.0, math.radians(90))
print(f"  q1 (identity):  ({q1.w:.3f}, {q1.x:.3f}, {q1.y:.3f}, {q1.z:.3f})")
print(f"  q2 (90° yaw):   ({q2.w:.3f}, {q2.x:.3f}, {q2.y:.3f}, {q2.z:.3f})")

mid = slerp(q1, q2, 0.5)
_, _, yaw_mid = mid.to_euler()
print(f"  SLERP(0.5):     yaw = {math.degrees(yaw_mid):.1f}° (expected 45°)")

# ──────────────────────────────────────────────────────────────
# DONE
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  Phase 1 Demo Complete!")
print(f"  Generated files: demo_trajectories.png, demo_gimbal.png, demo_results.png")
print(f"  CSV output:      {csv_path}")
print("=" * 70)
