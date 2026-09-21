#!/usr/bin/env python3
"""OptiLign Phase 4 Demo — Estimation + Closed-Loop Control.

Demonstrates:
  1. Kalman Filter tracking (CV model)
  2. PID controller with derivative filter + anti-windup
  3. Full closed-loop: Simulate → Render → Detect → Estimate → Control → Gimbal
  4. Performance analysis: detection accuracy, tracking error, gimbal response
"""

import math
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from fsoc_pat.control.closed_loop import ClosedLoopConfig, ClosedLoopEngine
from fsoc_pat.control.pid import PIDConfig
from fsoc_pat.estimation.kalman_filter import KalmanConfig
from fsoc_pat.estimation.motion_models import MotionModelType


print("=" * 70)
print("  OptiLign — Phase 4 Demo: Estimation + Closed-Loop Control")
print("=" * 70)

# ── 1. Closed-Loop Simulation ─────────────────────────────
print("\n▶ 1. RUNNING CLOSED-LOOP SIMULATION")

config = ClosedLoopConfig(
    duration_s=5.0,
    dt=0.02,  # 50 Hz
    target_center=(100.0, 0.0, 0.0),
    target_radius=10.0,
    target_angular_rate=1.0,
    pid=PIDConfig(kp=5.0, ki=0.5, kd=0.3, d_filter_tau=0.02, output_limit=2.0),
    kalman=KalmanConfig(
        motion_model=MotionModelType.CONSTANT_VELOCITY,
        r_std=2.0, q_process=10.0, gate_threshold=9.21,
    ),
    seed=42,
)

engine = ClosedLoopEngine(config)
t0 = time.perf_counter()
result = engine.run()
t_sim = time.perf_counter() - t0

print(f"  Duration:         {config.duration_s}s at {1/config.dt:.0f} Hz")
print(f"  Frames:           {len(result.snapshots)}")
print(f"  Detection rate:   {result.detection_rate:.1%}")
print(f"  Mean GT error:    {result.mean_gt_error_mrad:.2f} mrad")
print(f"  RMS GT error:     {result.rms_gt_error_mrad:.2f} mrad")
print(f"  Simulation time:  {t_sim:.1f}s")

# ── 2. Plots ──────────────────────────────────────────────
print("\n▶ 2. GENERATING PLOTS")

snaps = result.snapshots
times = [s.time for s in snaps]

fig = plt.figure(figsize=(18, 16))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# --- Angular tracking error ---
ax1 = fig.add_subplot(gs[0, :2])
gt_errs = [s.gt_error_total * 1000 for s in snaps]  # mrad
est_errs = [s.error_total * 1000 for s in snaps]
ax1.plot(times, gt_errs, "b-", linewidth=0.8, alpha=0.8, label="GT error")
ax1.plot(times, est_errs, "r-", linewidth=0.8, alpha=0.6, label="Estimated error")
ax1.set_title("Angular Tracking Error", fontweight="bold")
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Error (mrad)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- Detection vs estimation pixel U ---
ax2 = fig.add_subplot(gs[0, 2])
det_us = [s.det_u for s in snaps]
est_us = [s.est_u for s in snaps]
gt_us = [s.gt_u for s in snaps]
ax2.plot(times, gt_us, "k--", linewidth=0.6, alpha=0.5, label="GT")
ax2.plot(times, det_us, "b.", markersize=1.5, alpha=0.6, label="Detection")
ax2.plot(times, est_us, "r-", linewidth=0.8, alpha=0.8, label="KF Estimate")
ax2.set_title("Pixel U: GT vs Detection vs KF", fontweight="bold")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("u (px)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# --- Gimbal angles ---
ax3 = fig.add_subplot(gs[1, 0])
pans = [math.degrees(s.gimbal_pan) for s in snaps]
tilts = [math.degrees(s.gimbal_tilt) for s in snaps]
ax3.plot(times, pans, "b-", linewidth=1, label="Pan (yaw)")
ax3.plot(times, tilts, "r-", linewidth=1, label="Tilt (pitch)")
ax3.set_title("Gimbal Angles", fontweight="bold")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Angle (deg)")
ax3.legend()
ax3.grid(True, alpha=0.3)

# --- Control commands ---
ax4 = fig.add_subplot(gs[1, 1])
pan_cmds = [s.pan_cmd for s in snaps]
tilt_cmds = [s.tilt_cmd for s in snaps]
ax4.plot(times, pan_cmds, "b-", linewidth=0.8, alpha=0.8, label="Pan rate cmd")
ax4.plot(times, tilt_cmds, "r-", linewidth=0.8, alpha=0.8, label="Tilt rate cmd")
ax4.set_title("Control Commands", fontweight="bold")
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Rate (rad/s)")
ax4.legend()
ax4.grid(True, alpha=0.3)

# --- Confidence ---
ax5 = fig.add_subplot(gs[1, 2])
confs = [s.confidence for s in snaps]
detected_mask = [s.detected for s in snaps]
ax5.plot(times, confs, "g-", linewidth=0.8)
ax5.scatter(
    [t for t, d in zip(times, detected_mask) if not d],
    [c for c, d in zip(confs, detected_mask) if not d],
    color="red", s=10, zorder=5, label="Missed",
)
ax5.set_title("Detection Confidence", fontweight="bold")
ax5.set_xlabel("Time (s)")
ax5.set_ylabel("Confidence")
ax5.set_ylim(0, 1.05)
ax5.legend()
ax5.grid(True, alpha=0.3)

# --- Pixel trajectory ---
ax6 = fig.add_subplot(gs[2, 0])
valid_gt = [(s.gt_u, s.gt_v) for s in snaps if not math.isnan(s.gt_u)]
valid_est = [(s.est_u, s.est_v) for s in snaps]
if valid_gt:
    ax6.plot([p[0] for p in valid_gt], [p[1] for p in valid_gt],
             "k--", linewidth=0.6, alpha=0.5, label="GT")
ax6.plot([p[0] for p in valid_est], [p[1] for p in valid_est],
         "r-", linewidth=0.8, alpha=0.8, label="KF Estimate")
ax6.set_title("Pixel Trajectory (u, v)", fontweight="bold")
ax6.set_xlabel("u (px)")
ax6.set_ylabel("v (px)")
ax6.legend()
ax6.set_xlim(0, 640)
ax6.set_ylim(480, 0)
ax6.grid(True, alpha=0.3)

# --- Error histogram ---
ax7 = fig.add_subplot(gs[2, 1])
valid_errs = [e for e in gt_errs if not math.isnan(e)]
ax7.hist(valid_errs, bins=40, color="steelblue", edgecolor="black", alpha=0.8)
ax7.axvline(np.mean(valid_errs), color="red", linestyle="--", label=f"Mean: {np.mean(valid_errs):.1f} mrad")
ax7.set_title("GT Error Distribution", fontweight="bold")
ax7.set_xlabel("Error (mrad)")
ax7.set_ylabel("Count")
ax7.legend()
ax7.grid(True, alpha=0.3)

# --- Summary ---
ax8 = fig.add_subplot(gs[2, 2])
ax8.axis("off")
summary = "\n".join([
    "Phase 4 — Closed-Loop Summary",
    "─" * 35,
    "",
    f"Duration:       {config.duration_s}s",
    f"Rate:           {1/config.dt:.0f} Hz",
    f"Frames:         {len(snaps)}",
    "",
    f"Detection rate: {result.detection_rate:.1%}",
    f"Mean GT error:  {result.mean_gt_error_mrad:.2f} mrad",
    f"RMS GT error:   {result.rms_gt_error_mrad:.2f} mrad",
    "",
    f"KF model:       Constant Velocity",
    f"KF meas noise:  {config.kalman.r_std} px",
    f"PID gains:      Kp={config.pid.kp} Ki={config.pid.ki} Kd={config.pid.kd}",
    f"Output limit:   {config.pid.output_limit} rad/s",
    "",
    f"Sim wall time:  {t_sim:.1f}s",
])
ax8.text(0.05, 0.95, summary, transform=ax8.transAxes, fontsize=9,
         verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

fig.suptitle("OptiLign — Phase 4: Closed-Loop PAT Tracking",
             fontsize=16, fontweight="bold", y=0.98)
fig.savefig("demo_p4_results.png", dpi=150, bbox_inches="tight")
print("  → Saved: demo_p4_results.png")

print("\n" + "=" * 70)
print("  Phase 4 Demo Complete!")
print("=" * 70)
