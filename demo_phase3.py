#!/usr/bin/env python3
"""OptiLign Phase 3 Demo — AI-Assisted Perception.

Demonstrates:
  1. Synthetic dataset generation with domain randomization
  2. BeaconNet training (lightweight CNN)
  3. AI detector inference
  4. Hybrid detector (classical + AI fusion)
  5. Benchmark: Classical vs AI vs Hybrid
"""

import math
import time

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from fsoc_pat.ai.ai_detector import AIDetector
from fsoc_pat.ai.dataset import DatasetConfig, DatasetGenerator
from fsoc_pat.ai.hybrid_detector import HybridDetector, HybridConfig
from fsoc_pat.ai.model import BeaconNet
from fsoc_pat.ai.trainer import TrainConfig, Trainer
from fsoc_pat.core.state import CameraState
from fsoc_pat.core.types import AngularState, Orientation, Position3D
from fsoc_pat.geometry.camera_model import CameraIntrinsics
from fsoc_pat.geometry.projection import project_point_to_pixel
from fsoc_pat.perception.classical_detector import ClassicalDetector, ClassicalDetectorConfig
from fsoc_pat.perception.preprocessing import PreprocessConfig
from fsoc_pat.perception.renderer import RenderConfig, SyntheticRenderer
from fsoc_pat.simulation.beacon import BeaconProperties
from fsoc_pat.simulation.trajectories import CircularTrajectory

print("=" * 70)
print("  OptiLign — Phase 3 Demo: AI-Assisted Perception")
print("=" * 70)

intrinsics = CameraIntrinsics.from_fov(640, 480, math.radians(60))
camera_state = CameraState(
    position=Position3D(0, 0, 0),
    orientation=Orientation(),
    angular_state=AngularState(),
)

# ── 1. Dataset Generation ─────────────────────────────────
print("\n▶ 1. DATASET GENERATION")

ds_config = DatasetConfig(num_samples=200, fraction_negative=0.15, seed=42)
gen = DatasetGenerator(intrinsics, ds_config)
t0 = time.perf_counter()
samples = gen.generate()
t_gen = time.perf_counter() - t0

train, val, test = gen.split(samples)
n_pos = sum(1 for s in samples if s.present)
n_neg = sum(1 for s in samples if not s.present)

print(f"  Total samples:    {len(samples)} ({t_gen:.1f}s)")
print(f"  Positive/Neg:     {n_pos}/{n_neg}")
print(f"  Train/Val/Test:   {len(train)}/{len(val)}/{len(test)}")

# ── 2. Model Architecture ─────────────────────────────────
print("\n▶ 2. BEACONNET ARCHITECTURE")

model = BeaconNet()
params = model.count_parameters()
print(f"  Parameters:       {params:,} (<100K constraint)")
print(f"  Input:            (1, 120, 160) grayscale")
print(f"  Output:           presence_logit, u_norm, v_norm")

# ── 3. Training ───────────────────────────────────────────
print("\n▶ 3. TRAINING BeaconNet")

train_config = TrainConfig(
    num_samples=300,
    epochs=15,
    batch_size=16,
    lr=1e-3,
    seed=42,
)
trainer = Trainer(intrinsics, train_config)
t0 = time.perf_counter()
metrics = trainer.train()
t_train = time.perf_counter() - t0

print(f"  Training time:    {t_train:.1f}s")
print(f"  Epochs:           {len(metrics)}")
print(f"  Final train loss: {metrics[-1].train_loss:.4f}")
print(f"  Final val loss:   {metrics[-1].val_loss:.4f}")
print(f"  Val accuracy:     {metrics[-1].val_accuracy:.1%}")
print(f"  Val centroid err: {metrics[-1].val_centroid_error_px:.1f} px")

# Save model
model_path = "models/beaconnet_v1.pt"
trainer.save_model(model_path)
print(f"  → Saved: {model_path}")

# ── 4. Benchmark: Classical vs AI vs Hybrid ───────────────
print("\n▶ 4. BENCHMARK: Classical vs AI vs Hybrid")

# Setup detectors
classical = ClassicalDetector(ClassicalDetectorConfig(
    min_area_px=3.0, max_area_px=800.0, min_circularity=0.2,
    min_brightness=60.0, confidence_threshold=0.3,
    preprocess=PreprocessConfig(
        blur_kernel=5, threshold_method="global",
        threshold_value=60, morph_open_kernel=2, morph_close_kernel=2,
    ),
))
ai = AIDetector(
    model_path=model_path, confidence_threshold=0.5,
)
hybrid = HybridDetector(HybridConfig(
    ai_model_path=model_path, ai_confidence_threshold=0.5,
    classical_config=ClassicalDetectorConfig(
        min_area_px=3.0, max_area_px=800.0, min_circularity=0.2,
        min_brightness=60.0, confidence_threshold=0.3,
        preprocess=PreprocessConfig(
            blur_kernel=5, threshold_method="global",
            threshold_value=60, morph_open_kernel=2, morph_close_kernel=2,
        ),
    ),
))

# Generate test frames
beacon = BeaconProperties(intensity=230.0, size_px=8.0)
renderer = SyntheticRenderer(
    intrinsics,
    RenderConfig(noise_std=10.0, shot_noise=True, num_stars=10,
                 enable_bloom=True, seed=99),
)
traj = CircularTrajectory(
    center=Position3D(100.0, 0.0, 0.0), radius=10.0, angular_rate=1.0,
)

results = {
    "Classical": {"errors": [], "confs": [], "times": [], "dets": 0},
    "AI": {"errors": [], "confs": [], "times": [], "dets": 0},
    "Hybrid": {"errors": [], "confs": [], "times": [], "dets": 0},
}

N_FRAMES = 80
for i in range(N_FRAMES):
    t = i * 0.05
    pos = traj.position(t)
    target_pos = pos.as_array()
    frame = renderer.render_single_target(camera_state, target_pos, beacon)
    gt_px = project_point_to_pixel(target_pos, camera_state, intrinsics)

    for name, detector in [("Classical", classical), ("AI", ai), ("Hybrid", hybrid)]:
        t0 = time.perf_counter()
        det = detector.detect(frame, timestamp=t)
        dt_ms = (time.perf_counter() - t0) * 1000

        results[name]["times"].append(dt_ms)
        if det.detected and det.best is not None and gt_px is not None:
            results[name]["dets"] += 1
            err = math.sqrt((det.best.u - gt_px.u)**2 + (det.best.v - gt_px.v)**2)
            results[name]["errors"].append(err)
            results[name]["confs"].append(det.confidence)

print(f"\n  {'Detector':<12} {'Det%':>8} {'Mean Err':>10} {'Max Err':>10} {'Avg Time':>10}")
print(f"  {'─'*12} {'─'*8} {'─'*10} {'─'*10} {'─'*10}")
for name, data in results.items():
    det_pct = 100 * data["dets"] / N_FRAMES
    mean_err = np.mean(data["errors"]) if data["errors"] else float("nan")
    max_err = np.max(data["errors"]) if data["errors"] else float("nan")
    avg_time = np.mean(data["times"])
    print(f"  {name:<12} {det_pct:>7.0f}% {mean_err:>9.2f}px {max_err:>9.2f}px {avg_time:>8.2f}ms")

# ── 5. Generate plots ────────────────────────────────────
print("\n▶ 5. GENERATING PLOTS")

fig = plt.figure(figsize=(18, 14))
gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

# Training curves
ax1 = fig.add_subplot(gs[0, 0])
epochs = [m.epoch for m in metrics]
ax1.plot(epochs, [m.train_loss for m in metrics], "b-o", label="Train", markersize=3)
ax1.plot(epochs, [m.val_loss for m in metrics], "r-o", label="Val", markersize=3)
ax1.set_title("Training Loss", fontweight="bold")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(epochs, [m.val_accuracy for m in metrics], "g-o", markersize=3)
ax2.set_title("Val Classification Accuracy", fontweight="bold")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_ylim(0, 1.05)
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(epochs, [m.val_centroid_error_px for m in metrics], "m-o", markersize=3)
ax3.set_title("Val Centroid Error (px)", fontweight="bold")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Mean error (px)")
ax3.grid(True, alpha=0.3)

# Benchmark: error comparison
ax4 = fig.add_subplot(gs[1, 0])
colors = {"Classical": "dodgerblue", "AI": "coral", "Hybrid": "mediumseagreen"}
for name, data in results.items():
    if data["errors"]:
        ax4.plot(data["errors"], "-", color=colors[name], label=name, alpha=0.8, linewidth=1)
ax4.set_title("Detection Error per Frame", fontweight="bold")
ax4.set_xlabel("Frame")
ax4.set_ylabel("Pixel error (px)")
ax4.legend()
ax4.grid(True, alpha=0.3)

# Benchmark: confidence
ax5 = fig.add_subplot(gs[1, 1])
for name, data in results.items():
    if data["confs"]:
        ax5.plot(data["confs"], "-", color=colors[name], label=name, alpha=0.8, linewidth=1)
ax5.set_title("Detection Confidence per Frame", fontweight="bold")
ax5.set_xlabel("Frame")
ax5.set_ylabel("Confidence")
ax5.set_ylim(0, 1.05)
ax5.legend()
ax5.grid(True, alpha=0.3)

# Benchmark: speed
ax6 = fig.add_subplot(gs[1, 2])
names = list(results.keys())
avg_times = [np.mean(results[n]["times"]) for n in names]
bars = ax6.bar(names, avg_times, color=[colors[n] for n in names], edgecolor="black")
ax6.set_title("Average Processing Time", fontweight="bold")
ax6.set_ylabel("Time (ms)")
for bar, val in zip(bars, avg_times):
    ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f"{val:.2f}ms", ha="center", va="bottom", fontsize=9)
ax6.grid(True, alpha=0.3, axis="y")

# Error histograms
ax7 = fig.add_subplot(gs[2, 0])
for name, data in results.items():
    if data["errors"]:
        ax7.hist(data["errors"], bins=20, color=colors[name], alpha=0.6,
                 label=name, edgecolor="black")
ax7.set_title("Error Distribution", fontweight="bold")
ax7.set_xlabel("Pixel error (px)")
ax7.set_ylabel("Count")
ax7.legend()
ax7.grid(True, alpha=0.3)

# Detection rate comparison
ax8 = fig.add_subplot(gs[2, 1])
det_rates = [100 * results[n]["dets"] / N_FRAMES for n in names]
bars2 = ax8.bar(names, det_rates, color=[colors[n] for n in names], edgecolor="black")
ax8.set_title("Detection Rate", fontweight="bold")
ax8.set_ylabel("Detection rate (%)")
ax8.set_ylim(0, 110)
for bar, val in zip(bars2, det_rates):
    ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f"{val:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax8.grid(True, alpha=0.3, axis="y")

# Summary
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis("off")
summary_lines = [
    "OptiLign Phase 3 — AI Benchmark Summary",
    "─" * 42,
    "",
    f"BeaconNet params:  {params:,}",
    f"Training samples:  {train_config.num_samples}",
    f"Training epochs:   {train_config.epochs}",
    f"Training time:     {t_train:.1f}s",
    "",
]
for name in names:
    d = results[name]
    me = np.mean(d["errors"]) if d["errors"] else float("nan")
    summary_lines.append(
        f"{name:<12} {d['dets']}/{N_FRAMES} det  "
        f"err={me:.2f}px  "
        f"t={np.mean(d['times']):.2f}ms"
    )
summary = "\n".join(summary_lines)
ax9.text(0.05, 0.95, summary, transform=ax9.transAxes, fontsize=9,
         verticalalignment="top", fontfamily="monospace",
         bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

fig.suptitle("OptiLign — Phase 3: AI-Assisted Perception Results",
             fontsize=16, fontweight="bold", y=0.98)
fig.savefig("demo_p3_results.png", dpi=150, bbox_inches="tight")
print("  → Saved: demo_p3_results.png")

print("\n" + "=" * 70)
print("  Phase 3 Demo Complete!")
print("=" * 70)
