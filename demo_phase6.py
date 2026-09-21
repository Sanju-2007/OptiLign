#!/usr/bin/env python3
"""OptiLign Phase 6 Demo — FSOC Optical Alignment & Link Budget Analysis.

Demonstrates:
  1. Fundamental Gaussian beam propagation (waist, Rayleigh range, divergence)
  2. Spatial irradiance profile and aperture coupling
  3. Pointing error vs link margin degradation curves
  4. Dynamic closed-loop optical link budget (Rx Power, SNR, BER, Margin)
"""

import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

from fsoc_pat.control.closed_loop import ClosedLoopConfig, ClosedLoopEngine
from fsoc_pat.link.beam import GaussianBeam
from fsoc_pat.link.budget import FSOCLinkBudget
from fsoc_pat.link.link_types import (
    AtmospherePathConfig,
    LaserConfig,
    LinkState,
    OpticalReceiverConfig,
)


print("=" * 70)
print("  OptiLign — Phase 6 Demo: FSOC Optical Alignment & Link Budget")
print("=" * 70)

# ── 1. Optical System Configuration ──────────────────────
laser = LaserConfig(
    wavelength_m=1550e-9,  # 1550 nm C-band
    power_w=0.5,           # 500 mW (27 dBm)
    beam_waist_w0_m=0.00008, # 80 um waist (divergence ~ 6.8 mrad matched to coarse tracking)
    m2_factor=1.1,
    tx_efficiency=0.85,
)
rx = OpticalReceiverConfig(
    aperture_diameter_m=0.10,        # 10 cm collector
    rx_efficiency=0.70,
    detector_sensitivity_dbm=-38.0,  # -38 dBm for 1 Gbps at BER 10^-9
    data_rate_bps=1e9,               # 1 Gbps
)
atmo = AtmospherePathConfig(attenuation_db_per_km=0.5)

budget_calc = FSOCLinkBudget(laser=laser, receiver=rx, atmosphere=atmo)
beam = budget_calc.beam

print(f"\n▶ 1. GAUSSIAN BEAM OPTICAL PARAMETERS")
print(f"  • Wavelength (λ):         {laser.wavelength_m * 1e9:.1f} nm")
print(f"  • Transmit Power (P_tx):   {laser.power_w * 1000:.0f} mW ({laser.power_dbm:.1f} dBm)")
print(f"  • Beam Waist (w_0):        {laser.beam_waist_w0_m * 1000:.2f} mm")
print(f"  • Rayleigh Range (z_R):    {beam.rayleigh_range_m:.2f} m")
print(f"  • Divergence Half-Angle:   {beam.divergence_half_angle_rad * 1000:.3f} mrad")
print(f"  • Divergence Full-Angle:   {beam.divergence_full_angle_mrad:.3f} mrad")
print(f"  • Receiver Aperture (D):   {rx.aperture_diameter_m * 100:.1f} cm")
print(f"  • Receiver Sensitivity:    {rx.detector_sensitivity_dbm:.1f} dBm (at 1 Gbps)")

# ── 2. Parametric Analysis: Error vs Margin ──────────────
print(f"\n▶ 2. PARAMETRIC ANALYSIS: POINTING ERROR VS LINK MARGIN")
ranges_m = [100.0, 300.0, 500.0, 1000.0]
errors_mrad = np.linspace(0.0, 2.5, 100)

margin_curves = {}
for r in ranges_m:
    margins = []
    for err in errors_mrad:
        res = budget_calc.evaluate(distance_m=r, pointing_error_rad=err / 1000.0)
        margins.append(res.link_margin_db)
    margin_curves[r] = margins

# ── 3. Dynamic Closed-Loop Tracking Simulation ───────────
print(f"\n▶ 3. RUNNING CLOSED-LOOP LINK SIMULATION (5.0s at 50 Hz)")
sim_config = ClosedLoopConfig(
    duration_s=5.0,
    dt=0.02,
    target_center=(100.0, 0.0, 0.0),
    target_radius=1.5,
    target_angular_rate=0.4,
    seed=42,
)
engine = ClosedLoopEngine(sim_config)
t0 = time.perf_counter()
sim_result = engine.run()
sim_duration = time.perf_counter() - t0

times = [s.time for s in sim_result.snapshots]
pointing_errs_mrad = [s.error_total * 1000.0 for s in sim_result.snapshots]
rx_powers_dbm = []
link_margins_db = []
bers = []
states = []

for s in sim_result.snapshots:
    eval_res = budget_calc.evaluate(distance_m=100.0, pointing_error_rad=s.error_total)
    rx_powers_dbm.append(eval_res.rx_power_dbm)
    link_margins_db.append(eval_res.link_margin_db)
    bers.append(eval_res.ber)
    states.append(eval_res.state.name)

nominal_pct = (sum(1 for st in states if st == "NOMINAL") / len(states)) * 100.0
avg_margin = float(np.mean(link_margins_db))
min_margin = float(np.min(link_margins_db))

print(f"  Simulation completed in {sim_duration:.2f}s")
print(f"  Mean Link Margin:         {avg_margin:.2f} dB")
print(f"  Min Link Margin:          {min_margin:.2f} dB")
print(f"  Nominal Link Uptime:      {nominal_pct:.1f}%")

# ── 4. Generate Comprehensive Visualizations ─────────────
print(f"\n▶ 4. GENERATING PHASE 6 ANALYSIS PLOTS")

fig = plt.figure(figsize=(18, 14))
gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

# 1. 2D Gaussian Beam Irradiance Cross-Section at 100m
ax1 = fig.add_subplot(gs[0, 0])
w_100 = beam.beam_radius_at(100.0)
grid_span = w_100 * 2.2
x = np.linspace(-grid_span, grid_span, 150)
y = np.linspace(-grid_span, grid_span, 150)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
I = beam.peak_irradiance(100.0) * np.exp(-2.0 * (R**2) / (w_100**2))

im = ax1.imshow(
    I,
    extent=[-grid_span * 100, grid_span * 100, -grid_span * 100, grid_span * 100],
    cmap="inferno",
    origin="lower",
)
circ_waist = plt.Circle((0, 0), w_100 * 100, color="cyan", fill=False, linestyle="--", linewidth=1.5, label="1/e² Beam Waist (w)")
circ_rx = plt.Circle((0, 0), (rx.aperture_diameter_m / 2) * 100, color="white", fill=False, linestyle="-", linewidth=1.8, label="Rx Aperture (10cm)")
ax1.add_patch(circ_waist)
ax1.add_patch(circ_rx)
ax1.set_title("Gaussian Beam Irradiance Cross-Section at 100m", fontweight="bold")
ax1.set_xlabel("Horizontal Offset (cm)")
ax1.set_ylabel("Vertical Offset (cm)")
ax1.legend(loc="upper right", fontsize=8.5)
plt.colorbar(im, ax=ax1, label="Irradiance (W/m²)")

# 2. Pointing Error vs Link Margin Parametric Curves
ax2 = fig.add_subplot(gs[0, 1])
colors = ["#22c55e", "#38bdf8", "#f59e0b", "#ef4444"]
for (r, margins), col in zip(margin_curves.items(), colors):
    ax2.plot(errors_mrad, margins, color=col, linewidth=1.8, label=f"Range = {r:.0f}m")
ax2.axhline(3.0, color="gray", linestyle="--", alpha=0.7, label="Nominal Threshold (3 dB)")
ax2.axhline(0.0, color="red", linestyle=":", alpha=0.7, label="Outage Threshold (0 dB)")
ax2.set_title("Optical Link Margin vs Pointing Error", fontweight="bold")
ax2.set_xlabel("Pointing Error (mrad)")
ax2.set_ylabel("Link Margin (dB)")
ax2.set_ylim(-15, 45)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=8.5)

# 3. Dynamic Closed-Loop Pointing Error & Received Power
ax3 = fig.add_subplot(gs[1, :])
ax3_twin = ax3.twinx()

p1 = ax3.plot(times, pointing_errs_mrad, "b-", linewidth=1.2, label="Pointing Error (mrad)")
p2 = ax3_twin.plot(times, rx_powers_dbm, "r-", linewidth=1.2, label="Received Power P_rx (dBm)")
ax3_twin.axhline(rx.detector_sensitivity_dbm, color="red", linestyle="--", alpha=0.6, label="Sensitivity (-38 dBm)")

ax3.set_title("Closed-Loop Tracking Performance & Received Optical Power", fontweight="bold")
ax3.set_xlabel("Time (s)")
ax3.set_ylabel("Pointing Error (mrad)", color="blue")
ax3_twin.set_ylabel("Rx Optical Power (dBm)", color="red")
ax3.grid(True, alpha=0.3)

# Combine legends
lines = p1 + p2
labels = [l.get_label() for l in lines]
ax3.legend(lines, labels, loc="upper right")

# 4. Optical Link Margin Over Time
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(times, link_margins_db, color="green", linewidth=1.5, label="Link Margin (dB)")
ax4.axhline(3.0, color="darkorange", linestyle="--", alpha=0.8, label="3 dB Safe Buffer")
ax4.set_title("Dynamic Link Margin Over Time", fontweight="bold")
ax4.set_xlabel("Time (s)")
ax4.set_ylabel("Margin (dB)")
ax4.grid(True, alpha=0.3)
ax4.legend()

# 5. Theoretical Bit Error Rate (BER) Log Plot
ax5 = fig.add_subplot(gs[2, 1])
ax5.semilogy(times, bers, color="purple", linewidth=1.5, label="1 Gbps OOK BER")
ax5.axhline(1e-9, color="red", linestyle="--", alpha=0.7, label="Telecom Target (10⁻⁹)")
ax5.set_title("Communication Bit Error Rate (BER)", fontweight="bold")
ax5.set_xlabel("Time (s)")
ax5.set_ylabel("Bit Error Rate (BER)")
ax5.grid(True, alpha=0.3)
ax5.set_ylim(1e-15, 1e-1)
ax5.legend()

out_plot = "demo_p6_results.png"
plt.savefig(out_plot, dpi=150, bbox_inches="tight")
print(f"  → Saved: {out_plot}")

print("=" * 70)
print("  Phase 6 Demo Complete!")
print("======================================================================")
