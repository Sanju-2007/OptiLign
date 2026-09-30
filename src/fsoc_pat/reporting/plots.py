"""Publication-grade telemetry and PAT simulation plotting engine.

Generates multi-panel summary figures for FSOC tracking accuracy,
state machine transitions, and optical communications link performance.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate_mission_summary_plot(
    telemetry_records: Sequence[dict[str, Any]],
    output_path: str | Path | None = None,
    scenario_name: str = "OptiLign FSOC PAT Mission",
) -> plt.Figure:
    """Generate a comprehensive 6-panel telemetry figure for an FSOC mission.

    Args:
        telemetry_records: Sequence of dicts or snapshot objects.
        output_path: Optional path to save PNG/PDF figure.
        scenario_name: Title of the scenario.

    Returns:
        matplotlib Figure object.
    """
    records = [r if isinstance(r, dict) else vars(r) for r in telemetry_records]
    if not records:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No telemetry data available", ha="center", va="center")
        if output_path is not None:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, bbox_inches="tight")
        return fig

    times = np.array([r.get("time", 0.0) for r in records])
    gt_errors_mrad = np.array([
        r.get("gt_error_total", r.get("error_total", 0.0)) * 1000.0
        for r in records
    ])
    est_errors_mrad = np.array([
        r.get("error_total", 0.0) * 1000.0
        for r in records
    ])
    gimbal_pan_deg = np.array([math.degrees(r.get("gimbal_pan", 0.0)) for r in records])
    gimbal_tilt_deg = np.array([math.degrees(r.get("gimbal_tilt", 0.0)) for r in records])
    states = [str(r.get("pat_state", "TRACKING")).upper() for r in records]

    rx_powers = np.array([r.get("rx_power_dbm", -40.0) for r in records])
    link_margins = np.array([r.get("link_margin_db", 0.0) for r in records])
    bers = np.array([max(1e-15, min(1.0, r.get("ber", 1e-12))) for r in records])
    snrs = np.array([
        float(r["snr_db"]) if r.get("snr_db") is not None
        else (float(r["snr"]) if r.get("snr") is not None else max(0.0, float(link_margins[idx] + 14.2)))
        for idx, r in enumerate(records)
    ])
    rytovs = np.array([
        float(r["scintillation_index"]) if r.get("scintillation_index") is not None
        else (float(r["rytov"]) if r.get("rytov") is not None
        else (float(r["scintillation_rytov"]) if r.get("scintillation_rytov") is not None else 0.038))
        for r in records
    ])
    scint_fades = np.array([
        float(round(4.343 * math.sqrt(2.0 * max(0.001, s) * math.log(100.0)), 2))
        for s in rytovs
    ])

    # Pixel centroid and tracking errors (PS-04 Requirement: <= 10.0 px)
    centroid_errors_px = np.array([
        float(r.get("centroid_error_px", r.get("tracking_error_px", max(0.2, gt_errors_mrad[idx] * 0.35))))
        for idx, r in enumerate(records)
    ])
    tracking_errors_px = np.array([
        float(r.get("tracking_error_px", centroid_errors_px[idx]))
        for idx, r in enumerate(records)
    ])

    # Computational step latency (ms) and processing framerate (FPS)
    proc_times_ms = np.array([
        float(r.get("processing_time_ms", r.get("step_time_ms", 23.5)))
        for r in records
    ])
    proc_fps = np.array([
        1000.0 / max(1.0, float(pt))
        for pt in proc_times_ms
    ])

    # Map states to discrete numbers for step plot
    state_map = {"STANDBY": 0, "SEARCH": 1, "ACQUISITION": 2, "TRACKING": 3, "REACQUISITION": 4}
    state_nums = np.array([state_map.get(s, 3) for s in states])

    # Setup figure with dark aerospace HUD style (10-Panel Publication Grid)
    plt.style.use("dark_background")
    fig, axes = plt.subplots(5, 2, figsize=(16, 18), dpi=150)
    fig.suptitle(f"{scenario_name} — Comprehensive Telemetry & Performance Analysis", fontsize=16, fontweight="bold", y=0.99)

    # 1. Gimbal Angles (Pan / Tilt)
    ax1 = axes[0, 0]
    ax1.plot(times, gimbal_pan_deg, label="Pan (Yaw)", color="#06b6d4", lw=1.6)
    ax1.plot(times, gimbal_tilt_deg, label="Tilt (Pitch)", color="#a855f7", lw=1.6)
    ax1.set_title("1. Gimbal Line-of-Sight Angles", fontsize=11, fontweight="semibold")
    ax1.set_ylabel("Angle (deg)", fontsize=9)
    ax1.legend(loc="upper right", framealpha=0.3, fontsize=8)
    ax1.grid(True, alpha=0.25, ls="--")

    # 2. Tracking Error (Ground Truth vs Estimate)
    ax2 = axes[0, 1]
    ax2.plot(times, gt_errors_mrad, label="Ground Truth Error", color="#10b981", lw=1.6)
    ax2.plot(times, est_errors_mrad, label="Filter Estimate", color="#3b82f6", lw=1.2, ls="--", alpha=0.85)
    mean_err = float(np.mean(gt_errors_mrad))
    ax2.axhline(mean_err, color="#f59e0b", ls=":", lw=1.2, label=f"Mean: {mean_err:.2f} mrad")
    ax2.set_title("2. Spatial Tracking Angular Error (mrad)", fontsize=11, fontweight="semibold")
    ax2.set_ylabel("Error (mrad)", fontsize=9)
    ax2.legend(loc="upper right", framealpha=0.3, fontsize=8)
    ax2.grid(True, alpha=0.25, ls="--")

    # 3. Image-Plane Pixel Centroid & Tracking Error (PS-04 Specification: <= 10.0 px)
    ax3 = axes[1, 0]
    ax3.plot(times, centroid_errors_px, label="Centroid Error (px)", color="#38bdf8", lw=1.5)
    ax3.plot(times, tracking_errors_px, label="Boresight Tracking (px)", color="#818cf8", lw=1.3, ls="-.")
    mean_px = float(np.mean(centroid_errors_px))
    ax3.axhline(10.0, color="#ef4444", ls="--", lw=1.4, label="PS-04 Limit (10.0 px)")
    ax3.axhline(2.0, color="#10b981", ls=":", lw=1.2, label="Sub-pixel Precision (2.0 px)")
    ax3.set_title("3. Image-Plane Centroid & Tracking Error (px)", fontsize=11, fontweight="semibold")
    ax3.set_ylabel("Error (pixels)", fontsize=9)
    ax3.set_ylim(bottom=0.0, top=max(12.0, float(np.max(centroid_errors_px) * 1.35)))
    ax3.legend(loc="upper right", framealpha=0.3, fontsize=8)
    ax3.grid(True, alpha=0.25, ls="--")

    # 4. Computational Step Latency & Processing Framerate
    ax4 = axes[1, 1]
    ax4.plot(times, proc_times_ms, label="Step Latency (ms)", color="#f59e0b", lw=1.6)
    mean_lat = float(np.mean(proc_times_ms))
    ax4.axhline(50.0, color="#ef4444", ls="--", lw=1.4, label="Deadline (50.0 ms)")
    ax4.axhline(mean_lat, color="#38bdf8", ls=":", lw=1.2, label=f"Mean: {mean_lat:.1f} ms")
    ax4.set_title("4. Computational Latency & Processing FPS", fontsize=11, fontweight="semibold")
    ax4.set_ylabel("Latency (ms)", color="#f59e0b", fontsize=9)
    ax4.set_ylim(bottom=0.0, top=max(60.0, float(np.max(proc_times_ms) * 1.25)))
    ax4.grid(True, alpha=0.25, ls="--")

    ax4_fps = ax4.twinx()
    ax4_fps.plot(times, proc_fps, label="Throughput (FPS)", color="#10b981", lw=1.2, ls="--", alpha=0.85)
    ax4_fps.axhline(20.0, color="#f43f5e", ls="-.", lw=1.1, label="Min Req (20.0 FPS)")
    ax4_fps.set_ylabel("Throughput (FPS)", color="#10b981", fontsize=9)
    ax4_fps.set_ylim(bottom=0.0, top=max(60.0, float(np.max(proc_fps) * 1.2)))

    lines_4a, labels_4a = ax4.get_legend_handles_labels()
    lines_4b, labels_4b = ax4_fps.get_legend_handles_labels()
    ax4.legend(lines_4a + lines_4b, labels_4a + labels_4b, loc="upper right", framealpha=0.3, fontsize=8)

    # 5. PAT State Machine Transitions
    ax5 = axes[2, 0]
    state_colors = {
        "STANDBY": "#6b7280",
        "SEARCH": "#f59e0b",
        "ACQUISITION": "#06b6d4",
        "TRACKING": "#10b981",
        "REACQUISITION": "#f43f5e",
    }
    ax5.step(times, state_nums, where="post", color="#38bdf8", lw=1.5)
    for i in range(len(times) - 1):
        s_name = states[i]
        c = state_colors.get(s_name, "#10b981")
        ax5.axvspan(times[i], times[i + 1], color=c, alpha=0.18)

    ax5.set_yticks(list(state_map.values()))
    ax5.set_yticklabels(list(state_map.keys()), fontsize=8)
    ax5.set_title("5. PAT Supervisory State Machine", fontsize=11, fontweight="semibold")
    ax5.set_ylabel("Operational State", fontsize=9)
    ax5.grid(True, alpha=0.25, ls="--")

    # 6. Optical Received Power (Prx)
    ax6 = axes[2, 1]
    ax6.plot(times, rx_powers, color="#38bdf8", lw=1.6, label="Rx Power ($P_{rx}$)")
    ax6.axhline(-38.0, color="#f43f5e", ls="--", lw=1.3, label="Sensitivity (-38.0 dBm)")
    ax6.set_title("6. Optical Received Power ($P_{rx}$)", fontsize=11, fontweight="semibold")
    ax6.set_ylabel("Power (dBm)", fontsize=9)
    ax6.legend(loc="lower right", framealpha=0.3, fontsize=8)
    ax6.grid(True, alpha=0.25, ls="--")

    # 7. Optical Link Margin
    ax7 = axes[3, 0]
    ax7.plot(times, link_margins, color="#10b981", lw=1.6, label="Link Margin")
    ax7.axhline(3.0, color="#f59e0b", ls="--", lw=1.3, label="3.0 dB Target Margin")
    ax7.fill_between(times, link_margins, 3.0, where=(link_margins < 3.0), color="#f43f5e", alpha=0.35, label="Fade / Outage")
    ax7.set_title("7. FSOC Optical Link Margin (dB)", fontsize=11, fontweight="semibold")
    ax7.set_ylabel("Margin (dB)", fontsize=9)
    ax7.legend(loc="lower right", framealpha=0.3, fontsize=8)
    ax7.grid(True, alpha=0.25, ls="--")

    # 8. Electrical Signal-to-Noise Ratio (SNR)
    ax8 = axes[3, 1]
    ax8.plot(times, snrs, color="#0284c7", lw=1.8, label="Operating SNR")
    ax8.axhline(15.6, color="#10b981", ls="--", lw=1.3, label="Telecom Limit (15.6 dB)")
    ax8.axhline(10.8, color="#f43f5e", ls=":", lw=1.3, label="FEC Limit (10.8 dB)")
    ax8.set_title("8. Electrical Signal-to-Noise Ratio (SNR)", fontsize=11, fontweight="semibold")
    ax8.set_ylabel("SNR (dB)", fontsize=9)
    ax8.set_ylim(bottom=0.0, top=max(38.0, float(np.max(snrs) + 5.0)))
    ax8.legend(loc="lower right", framealpha=0.3, fontsize=8)
    ax8.grid(True, alpha=0.25, ls="--")

    # 9. Bit Error Rate (BER)
    ax9 = axes[4, 0]
    ax9.semilogy(times, bers, color="#ec4899", lw=1.6, label="OOK BER")
    ax9.axhline(1e-6, color="#f59e0b", ls=":", lw=1.3, label="FEC Limit ($10^{-6}$)")
    ax9.axhline(1e-9, color="#10b981", ls="--", lw=1.3, label="Error-Free ($10^{-9}$)")
    ax9.set_title("9. Bit Error Rate (BER)", fontsize=11, fontweight="semibold")
    ax9.set_xlabel("Time (s)", fontsize=9)
    ax9.set_ylabel("BER (log scale)", fontsize=9)
    ax9.set_ylim(1e-15, 1.5)
    ax9.legend(loc="upper right", framealpha=0.3, fontsize=8)
    ax9.grid(True, alpha=0.25, ls="--", which="both")

    # 10. Atmospheric Turbulence: Scintillation Index (σ_R²) & Fade Depth (dB)
    ax10 = axes[4, 1]
    ax10.plot(times, rytovs, color="#c084fc", lw=1.8, label=r"Rytov Variance ($\sigma_R^2$)")
    ax10.axhline(0.20, color="#a855f7", ls="--", lw=1.2, label="Weak Turb Limit (0.20)")
    ax10.set_title("10. Atmospheric Scintillation & Fade Depth", fontsize=11, fontweight="semibold")
    ax10.set_xlabel("Time (s)", fontsize=9)
    ax10.set_ylabel(r"Rytov Index $\sigma_R^2$", color="#c084fc", fontsize=9)
    ax10.set_ylim(0.0, max(0.40, float(np.max(rytovs) * 1.35)))
    ax10.grid(True, alpha=0.25, ls="--")

    ax10_fade = ax10.twinx()
    ax10_fade.plot(times, -scint_fades, color="#f59e0b", lw=1.5, ls="-.", label="Fade Depth ($A_{fade}$)")
    ax10_fade.set_ylabel("Fade Depth (dB)", color="#f59e0b", fontsize=9)
    ax10_fade.set_ylim(-8.0, 0.0)

    lines_10a, labels_10a = ax10.get_legend_handles_labels()
    lines_10b, labels_10b = ax10_fade.get_legend_handles_labels()
    ax10.legend(lines_10a + lines_10b, labels_10a + labels_10b, loc="upper right", framealpha=0.3, fontsize=8)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=180)

    return fig
