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

    # Map states to discrete numbers for step plot
    state_map = {"STANDBY": 0, "SEARCH": 1, "ACQUISITION": 2, "TRACKING": 3, "REACQUISITION": 4}
    state_nums = np.array([state_map.get(s, 3) for s in states])

    # Setup figure with dark aerospace HUD style
    plt.style.use("dark_background")
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), dpi=150)
    fig.suptitle(f"{scenario_name} — Comprehensive Telemetry Analysis", fontsize=16, fontweight="bold", y=0.98)

    # 1. Gimbal Angles (Pan / Tilt)
    ax1 = axes[0, 0]
    ax1.plot(times, gimbal_pan_deg, label="Pan (Yaw)", color="#06b6d4", lw=1.6)
    ax1.plot(times, gimbal_tilt_deg, label="Tilt (Pitch)", color="#a855f7", lw=1.6)
    ax1.set_title("Gimbal Line-of-Sight Angles", fontsize=11, fontweight="semibold")
    ax1.set_ylabel("Angle (deg)", fontsize=9)
    ax1.legend(loc="upper right", framealpha=0.3, fontsize=8)
    ax1.grid(True, alpha=0.25, ls="--")

    # 2. Tracking Error (Ground Truth vs Estimate)
    ax2 = axes[0, 1]
    ax2.plot(times, gt_errors_mrad, label="Ground Truth Error", color="#10b981", lw=1.6)
    ax2.plot(times, est_errors_mrad, label="Filter Estimate", color="#3b82f6", lw=1.2, ls="--", alpha=0.85)
    mean_err = float(np.mean(gt_errors_mrad))
    ax2.axhline(mean_err, color="#f59e0b", ls=":", lw=1.2, label=f"Mean: {mean_err:.2f} mrad")
    ax2.set_title("Tracking Angular Error (mrad)", fontsize=11, fontweight="semibold")
    ax2.set_ylabel("Error (mrad)", fontsize=9)
    ax2.legend(loc="upper right", framealpha=0.3, fontsize=8)
    ax2.grid(True, alpha=0.25, ls="--")

    # 3. PAT State Machine Transitions
    ax3 = axes[1, 0]
    state_colors = {
        "STANDBY": "#6b7280",
        "SEARCH": "#f59e0b",
        "ACQUISITION": "#06b6d4",
        "TRACKING": "#10b981",
        "REACQUISITION": "#f43f5e",
    }
    ax3.step(times, state_nums, where="post", color="#38bdf8", lw=1.5)
    # Color background spans based on state
    for i in range(len(times) - 1):
        s_name = states[i]
        c = state_colors.get(s_name, "#10b981")
        ax3.axvspan(times[i], times[i + 1], color=c, alpha=0.18)

    ax3.set_yticks(list(state_map.values()))
    ax3.set_yticklabels(list(state_map.keys()), fontsize=8)
    ax3.set_title("PAT State Machine Mode", fontsize=11, fontweight="semibold")
    ax3.set_ylabel("Operational State", fontsize=9)
    ax3.grid(True, alpha=0.25, ls="--")

    # 4. Optical Received Power (Prx)
    ax4 = axes[1, 1]
    ax4.plot(times, rx_powers, color="#38bdf8", lw=1.6, label="Rx Power ($P_{rx}$)")
    ax4.axhline(-38.0, color="#f43f5e", ls="--", lw=1.3, label="Sensitivity (-38.0 dBm)")
    ax4.set_title("Optical Received Power ($P_{rx}$)", fontsize=11, fontweight="semibold")
    ax4.set_ylabel("Power (dBm)", fontsize=9)
    ax4.legend(loc="lower right", framealpha=0.3, fontsize=8)
    ax4.grid(True, alpha=0.25, ls="--")

    # 5. Optical Link Margin
    ax5 = axes[2, 0]
    ax5.plot(times, link_margins, color="#10b981", lw=1.6, label="Link Margin")
    ax5.axhline(3.0, color="#f59e0b", ls="--", lw=1.3, label="3.0 dB Target Margin")
    ax5.fill_between(times, link_margins, 3.0, where=(link_margins < 3.0), color="#f43f5e", alpha=0.35, label="Fade / Outage")
    ax5.set_title("FSOC Optical Link Margin (dB)", fontsize=11, fontweight="semibold")
    ax5.set_xlabel("Time (s)", fontsize=9)
    ax5.set_ylabel("Margin (dB)", fontsize=9)
    ax5.legend(loc="lower right", framealpha=0.3, fontsize=8)
    ax5.grid(True, alpha=0.25, ls="--")

    # 6. Bit Error Rate (BER)
    ax6 = axes[2, 1]
    ax6.semilogy(times, bers, color="#ec4899", lw=1.6, label="OOK BER")
    ax6.axhline(1e-6, color="#f59e0b", ls=":", lw=1.3, label="FEC Limit ($10^{-6}$)")
    ax6.axhline(1e-9, color="#10b981", ls="--", lw=1.3, label="Error-Free ($10^{-9}$)")
    ax6.set_title("Bit Error Rate (BER)", fontsize=11, fontweight="semibold")
    ax6.set_xlabel("Time (s)", fontsize=9)
    ax6.set_ylabel("BER (log scale)", fontsize=9)
    ax6.set_ylim(1e-15, 1.5)
    ax6.legend(loc="upper right", framealpha=0.3, fontsize=8)
    ax6.grid(True, alpha=0.25, ls="--", which="both")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=180)

    return fig
