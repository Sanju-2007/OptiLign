#!/usr/bin/env python3
"""OptiLign Phase 2 Demo — Perception Pipeline.

Demonstrates:
  1. Synthetic image rendering (beacon on dark background with noise)
  2. Classical detection (contour → centroid)
  3. Detection accuracy vs ground truth
  4. Multi-frame confidence buildup
  5. Tracking a moving target across frames
"""

import math
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from fsoc_pat.core.state import CameraState
from fsoc_pat.core.types import AngularState, Orientation, Position3D
from fsoc_pat.geometry.camera_model import CameraIntrinsics
from fsoc_pat.geometry.projection import project_point_to_pixel
from fsoc_pat.perception.classical_detector import ClassicalDetectorConfig
from fsoc_pat.perception.pipeline import PerceptionConfig, PerceptionPipeline
from fsoc_pat.perception.preprocessing import PreprocessConfig
from fsoc_pat.perception.renderer import RenderConfig
from fsoc_pat.simulation.beacon import BeaconProperties
from fsoc_pat.simulation.trajectories import CircularTrajectory

print("=" * 70)
print("  OptiLign — Phase 2 Demonstration: Perception Pipeline")
print("=" * 70)

# Setup
intrinsics = CameraIntrinsics.from_fov(640, 480, math.radians(60))
camera_state = CameraState(
    position=Position3D(0, 0, 0),
    orientation=Orientation(),
    angular_state=AngularState(),
)
beacon = BeaconProperties(intensity=230.0, size_px=8.0)

config = PerceptionConfig(
    render=RenderConfig(
        noise_std=8.0, shot_noise=True, num_stars=15,
        enable_bloom=True, seed=42,
    ),
    detector=ClassicalDetectorConfig(
        min_area_px=3.0,
        max_area_px=800.0,
        min_circularity=0.2,
        min_brightness=60.0,
        confidence_threshold=0.3,
        preprocess=PreprocessConfig(
            blur_kernel=5,
            threshold_method="global",
            threshold_value=60,
            morph_open_kernel=2,
            morph_close_kernel=2,
        ),
    ),
)
pipeline = PerceptionPipeline(intrinsics, config)

# ── Demo 1: Single frame detection ────────────────────────
print("\n▶ 1. SINGLE FRAME DETECTION")
target_pos = np.array([100.0, 5.0, -2.0])

detection = pipeline.process_frame(
    camera_state, [(target_pos, beacon)], timestamp=0.0,
)

gt_pixel = project_point_to_pixel(target_pos, camera_state, intrinsics)
print(f"  Ground truth pixel:  ({gt_pixel.u:.1f}, {gt_pixel.v:.1f})")
if detection.detected:
    print(f"  Detected pixel:      ({detection.best.u:.1f}, {detection.best.v:.1f})")
    print(f"  Error:               ({abs(detection.best.u - gt_pixel.u):.1f}, "
          f"{abs(detection.best.v - gt_pixel.v):.1f}) px")
    print(f"  Confidence:          {detection.confidence:.3f}")
    print(f"  Candidates found:    {detection.num_candidates}")
    print(f"  Processing time:     {detection.processing_time_ms:.2f} ms")
else:
    print("  NOT DETECTED")

# Save rendered frame with annotations
frame = pipeline.last_frame.copy()
frame_annotated = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
if gt_pixel:
    cv2.drawMarker(frame_annotated, (int(gt_pixel.u), int(gt_pixel.v)),
                   (0, 255, 0), cv2.MARKER_CROSS, 15, 1)
if detection.detected:
    cv2.circle(frame_annotated, (int(detection.best.u), int(detection.best.v)),
               12, (0, 0, 255), 1)
    cv2.putText(frame_annotated, f"conf={detection.confidence:.2f}",
                (int(detection.best.u) + 15, int(detection.best.v) - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

cv2.imwrite("demo_p2_single_frame.png", frame_annotated)
print(f"  → Saved: demo_p2_single_frame.png")

# ── Demo 2: Moving target tracking ───────────────────────
print("\n▶ 2. TRACKING A MOVING TARGET (100 frames)")

pipeline.reset()
traj = CircularTrajectory(
    center=Position3D(100.0, 0.0, 0.0), radius=10.0, angular_rate=1.0,
)

gt_us, gt_vs = [], []
det_us, det_vs = [], []
confidences = []
errors = []
times = []

for i in range(100):
    t = i * 0.05  # 20 FPS
    pos = traj.position(t)
    target_pos = pos.as_array()

    det = pipeline.process_frame(
        camera_state, [(target_pos, beacon)], timestamp=t,
    )

    gt_px = project_point_to_pixel(target_pos, camera_state, intrinsics)
    if gt_px:
        gt_us.append(gt_px.u)
        gt_vs.append(gt_px.v)
    else:
        gt_us.append(float("nan"))
        gt_vs.append(float("nan"))

    if det.detected:
        det_us.append(det.best.u)
        det_vs.append(det.best.v)
        confidences.append(det.confidence)
        if gt_px:
            err = math.sqrt((det.best.u - gt_px.u)**2 + (det.best.v - gt_px.v)**2)
            errors.append(err)
        else:
            errors.append(float("nan"))
    else:
        det_us.append(float("nan"))
        det_vs.append(float("nan"))
        confidences.append(0.0)
        errors.append(float("nan"))

    times.append(t)

valid_errors = [e for e in errors if not math.isnan(e)]
print(f"  Frames processed:    {len(times)}")
print(f"  Detections:          {sum(1 for u in det_us if not math.isnan(u))}/100")
print(f"  Mean pixel error:    {np.nanmean(valid_errors):.2f} px")
print(f"  Max pixel error:     {np.nanmax(valid_errors):.2f} px")
print(f"  Mean confidence:     {np.mean(confidences):.3f}")

# ── Demo 3: Comprehensive visualization ──────────────────
print("\n▶ 3. GENERATING PLOTS")

fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

# Rendered frame
ax1 = fig.add_subplot(gs[0, 0])
ax1.imshow(frame, cmap="gray", vmin=0, vmax=255)
if detection.detected:
    ax1.plot(detection.best.u, detection.best.v, "r+", markersize=15, markeredgewidth=2)
if gt_pixel:
    ax1.plot(gt_pixel.u, gt_pixel.v, "gx", markersize=12, markeredgewidth=2)
ax1.set_title("Rendered Frame (green=GT, red=detected)", fontweight="bold")
ax1.set_xlabel("u (px)")
ax1.set_ylabel("v (px)")

# Pixel trace: GT vs detected
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(gt_us, gt_vs, "g-", linewidth=2, label="Ground Truth", alpha=0.8)
ax2.plot(det_us, det_vs, "r.", markersize=4, label="Detected", alpha=0.7)
ax2.set_xlim(0, 640)
ax2.set_ylim(480, 0)
ax2.set_title("Pixel Trace: GT vs Detected", fontweight="bold")
ax2.set_xlabel("u (px)")
ax2.set_ylabel("v (px)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Pixel error over time
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(times, errors, "b-", linewidth=1.5)
ax3.axhline(np.nanmean(valid_errors), color="red", linestyle="--",
            label=f"Mean={np.nanmean(valid_errors):.2f}px")
ax3.set_title("Detection Error vs Time", fontweight="bold")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Pixel error (px)")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Confidence over time
ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(times, confidences, "m-", linewidth=1.5)
ax4.set_title("Detection Confidence vs Time", fontweight="bold")
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Confidence [0,1]")
ax4.set_ylim(0, 1.05)
ax4.grid(True, alpha=0.3)

# Error histogram
ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(valid_errors, bins=30, color="coral", edgecolor="black", alpha=0.8)
ax5.axvline(np.nanmean(valid_errors), color="red", linestyle="--", linewidth=2,
            label=f"Mean={np.nanmean(valid_errors):.2f}px")
ax5.set_title("Detection Error Distribution", fontweight="bold")
ax5.set_xlabel("Error (px)")
ax5.set_ylabel("Count")
ax5.legend()
ax5.grid(True, alpha=0.3)

# Summary
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")
summary = (
    f"OptiLign Phase 2 — Perception Summary\n"
    f"{'─' * 40}\n\n"
    f"Camera:          640×480, HFOV=60°\n"
    f"Beacon:          I={beacon.intensity:.0f}, size={beacon.size_px:.0f}px\n"
    f"Noise:           σ=8, shot=yes, stars=15\n"
    f"Trajectory:      Circular, R=10m, ω=1\n\n"
    f"Detection rate:  {sum(1 for u in det_us if not math.isnan(u))}/100 "
    f"({100*sum(1 for u in det_us if not math.isnan(u))/100:.0f}%)\n"
    f"Mean error:      {np.nanmean(valid_errors):.2f} px\n"
    f"Max error:       {np.nanmax(valid_errors):.2f} px\n"
    f"Mean confidence: {np.mean(confidences):.3f}\n"
    f"Ang. resolution: {intrinsics.pixel_angular_resolution*1000:.2f} mrad/px\n"
    f"Error (angular): {np.nanmean(valid_errors)*intrinsics.pixel_angular_resolution*1000:.2f} mrad\n"
)
ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
         verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

fig.suptitle("OptiLign — Phase 2: Perception Pipeline Results",
             fontsize=16, fontweight="bold", y=0.98)
fig.savefig("demo_p2_results.png", dpi=150, bbox_inches="tight")
print(f"  → Saved: demo_p2_results.png")

# ── Demo 4: Frame montage ────────────────────────────────
print("\n▶ 4. FRAME MONTAGE (4 sample frames)")

pipeline.reset()
montage_frames = []
sample_times = [0.0, 1.5, 3.0, 4.5]

for t in sample_times:
    pos = traj.position(t)
    det = pipeline.process_frame(
        camera_state, [(pos.as_array(), beacon)], timestamp=t,
    )
    f = pipeline.last_frame.copy()
    f_bgr = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)

    gt_px = project_point_to_pixel(pos.as_array(), camera_state, intrinsics)
    if gt_px:
        cv2.drawMarker(f_bgr, (int(gt_px.u), int(gt_px.v)),
                       (0, 255, 0), cv2.MARKER_CROSS, 12, 1)
    if det.detected:
        cv2.circle(f_bgr, (int(det.best.u), int(det.best.v)), 10, (0, 0, 255), 1)

    cv2.putText(f_bgr, f"t={t:.1f}s", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    montage_frames.append(f_bgr)

# Stitch 2x2
top = np.hstack(montage_frames[:2])
bottom = np.hstack(montage_frames[2:])
montage = np.vstack([top, bottom])
cv2.imwrite("demo_p2_montage.png", montage)
print(f"  → Saved: demo_p2_montage.png")

print("\n" + "=" * 70)
print("  Phase 2 Demo Complete!")
print("  Files: demo_p2_single_frame.png, demo_p2_results.png, demo_p2_montage.png")
print("=" * 70)
