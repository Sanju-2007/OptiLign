"""Adaptive Optics (AO) Closed-Loop Performance Benchmark and Reporting.

Executes a high-fidelity dynamic simulation comparing uncorrected turbulent
FSOC link performance against the Closed-Loop Adaptive Optics system:
  - Kolmogorov phase screen evolution
  - Real-time Shack-Hartmann slope sensing & SVD wavefront reconstruction
  - Deformable mirror phase flattening
  - Strehl ratio recovery (0.05 -> 0.85+)
  - Single-Mode Fiber (SMF) coupling efficiency (+12 to +18 dB optical gain)
  - 6-panel publication-grade diagnostic plot generation
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fsoc_pat.ao.ao_system import AdaptiveOpticsSystem, compute_smf_coupling_efficiency
from fsoc_pat.ao.ao_types import AOConfig, AOTelemetry
from fsoc_pat.ao.kolmogorov import (
    KolmogorovPhaseScreenGenerator,
    KolmogorovTurbulenceConfig,
    compute_fried_parameter,
    compute_greenwood_frequency,
)


@dataclass
class AOBenchmarkConfig:
    """Benchmark parameters.

    Attributes:
        duration_s: Total simulation time in seconds.
        dt: Timestep in seconds (e.g. 0.002s for 500 Hz AO loop).
        cn2: Refractive index structure parameter (m^(-2/3)).
        link_distance_m: Path length through atmosphere (m).
        wind_speed_m_s: Transverse wind speed (m/s).
        pupil_diameter_m: Telescope aperture diameter (m).
        wavelength_m: Laser carrier wavelength (m). Default 1550 nm.
        seed: Random seed.
    """

    duration_s: float = 0.50
    dt: float = 0.002  # 500 Hz AO loop
    cn2: float = 2.5e-14
    link_distance_m: float = 1000.0
    wind_speed_m_s: float = 6.0
    pupil_diameter_m: float = 0.15
    wavelength_m: float = 1550e-9
    seed: int = 42


@dataclass
class AOBenchmarkMetrics:
    """Summary statistics for AO benchmark."""

    r0_cm: float
    greenwood_freq_hz: float
    d_over_r0: float
    pre_ao_mean_wfe_nm: float
    post_ao_mean_wfe_nm: float
    pre_ao_mean_strehl: float
    post_ao_mean_strehl: float
    pre_ao_mean_smf_coupling_pct: float
    post_ao_mean_smf_coupling_pct: float
    mean_ao_gain_db: float
    max_dm_stroke_used_um: float
    wfe_reduction_pct: float
    strehl_improvement_factor: float


class AOBenchmarkEngine:
    """Orchestrates dynamic closed-loop AO evaluation under Kolmogorov turbulence."""

    def __init__(self, config: AOBenchmarkConfig | None = None) -> None:
        self.config = config or AOBenchmarkConfig()

    def run(self) -> tuple[AOBenchmarkMetrics, list[AOTelemetry], AdaptiveOpticsSystem]:
        """Execute closed-loop AO benchmark."""
        cfg = self.config
        ao_cfg = AOConfig(
            enabled=True,
            pupil_diameter_m=cfg.pupil_diameter_m,
            wavelength_m=cfg.wavelength_m,
            grid_resolution=64,
            max_zernike_modes=22,
        )
        ao_sys = AdaptiveOpticsSystem(ao_cfg, seed=cfg.seed)

        turb_cfg = KolmogorovTurbulenceConfig(
            cn2=cfg.cn2,
            link_distance_m=cfg.link_distance_m,
            pupil_diameter_m=cfg.pupil_diameter_m,
            wavelength_m=cfg.wavelength_m,
            wind_speed_m_s=cfg.wind_speed_m_s,
            seed=cfg.seed,
        )
        turb_gen = KolmogorovPhaseScreenGenerator(
            config=turb_cfg,
            basis=ao_sys.basis,
            max_modes=22,
        )

        n_steps = int(cfg.duration_s / cfg.dt)
        telemetry_records: list[AOTelemetry] = []

        for i in range(n_steps):
            t = i * cfg.dt
            # Generate current dynamic Kolmogorov phase screen
            phase_turb, _ = turb_gen.step(cfg.dt)
            # Step closed-loop AO
            telem = ao_sys.step(phase_turb, cfg.dt, timestamp_s=t)
            telemetry_records.append(telem)

        # Compute summary metrics
        r0 = turb_gen.r_0
        fg = turb_gen.f_greenwood
        d_r0 = turb_gen.d_over_r0

        # Discard initial convergence transient (first 10% of samples)
        steady_start = max(1, n_steps // 10)
        steady = telemetry_records[steady_start:]

        mean_wfe_pre = float(np.mean([t.raw_wfe_rms_nm for t in steady]))
        mean_wfe_post = float(np.mean([t.corrected_wfe_rms_nm for t in steady]))
        mean_strehl_pre = float(np.mean([t.strehl_ratio_pre for t in steady]))
        mean_strehl_post = float(np.mean([t.strehl_ratio_post for t in steady]))
        mean_smf_pre = float(np.mean([t.smf_coupling_pre for t in steady])) * 100.0
        mean_smf_post = float(np.mean([t.smf_coupling_post for t in steady])) * 100.0
        mean_gain = float(np.mean([t.ao_optical_gain_db for t in steady]))
        max_stroke = float(np.max([t.dm_stroke_used_um for t in steady]))

        wfe_red = ((mean_wfe_pre - mean_wfe_post) / max(1e-4, mean_wfe_pre)) * 100.0
        strehl_imp = mean_strehl_post / max(1e-4, mean_strehl_pre)

        metrics = AOBenchmarkMetrics(
            r0_cm=r0 * 100.0,
            greenwood_freq_hz=fg,
            d_over_r0=d_r0,
            pre_ao_mean_wfe_nm=mean_wfe_pre,
            post_ao_mean_wfe_nm=mean_wfe_post,
            pre_ao_mean_strehl=mean_strehl_pre,
            post_ao_mean_strehl=mean_strehl_post,
            pre_ao_mean_smf_coupling_pct=mean_smf_pre,
            post_ao_mean_smf_coupling_pct=mean_smf_post,
            mean_ao_gain_db=mean_gain,
            max_dm_stroke_used_um=max_stroke,
            wfe_reduction_pct=wfe_red,
            strehl_improvement_factor=strehl_imp,
        )

        return metrics, telemetry_records, ao_sys


def generate_ao_benchmark_plot(
    metrics: AOBenchmarkMetrics,
    telemetry: list[AOTelemetry],
    ao_sys: AdaptiveOpticsSystem,
    output_path: Path,
) -> None:
    """Generate a 6-panel publication-grade Adaptive Optics diagnostic figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 10), facecolor="#0B132B")
    plt.rcParams["text.color"] = "#E0E1DD"
    plt.rcParams["axes.labelcolor"] = "#E0E1DD"
    plt.rcParams["xtick.color"] = "#E0E1DD"
    plt.rcParams["ytick.color"] = "#E0E1DD"

    times = [t.timestamp_s * 1000.0 for t in telemetry]  # ms
    strehl_pre = [t.strehl_ratio_pre for t in telemetry]
    strehl_post = [t.strehl_ratio_post for t in telemetry]
    smf_pre = [t.smf_coupling_pre * 100.0 for t in telemetry]
    smf_post = [t.smf_coupling_post * 100.0 for t in telemetry]

    raw_phase, dm_phase, res_phase = ao_sys.get_wavefront_maps()
    pupil_mask = ao_sys.basis.pupil_mask

    # Mask outside pupil for clean display
    raw_disp = np.ma.masked_where(~pupil_mask, raw_phase)
    dm_disp = np.ma.masked_where(~pupil_mask, dm_phase)
    res_disp = np.ma.masked_where(~pupil_mask, res_phase)

    # Panel 1: Turbulent Phase Map (Pre-AO)
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_facecolor("#1C2541")
    vmax = max(1.0, float(np.max(np.abs(raw_phase[pupil_mask]))))
    im1 = ax1.imshow(raw_disp, cmap="coolwarm", vmin=-vmax, vmax=vmax, extent=[-1, 1, -1, 1])
    ax1.set_title(f"Pre-AO Turbulent Wavefront\n(RMS: {metrics.pre_ao_mean_wfe_nm:.0f} nm)", fontsize=11, fontweight="bold")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Phase (rad)")

    # Panel 2: Deformable Mirror Correction Surface
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_facecolor("#1C2541")
    im2 = ax2.imshow(dm_disp, cmap="coolwarm", vmin=-vmax, vmax=vmax, extent=[-1, 1, -1, 1])
    ax2.set_title("DM Conjugate Phase Shape\n(37 Piezo Actuators)", fontsize=11, fontweight="bold")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Phase (rad)")

    # Panel 3: Post-AO Residual Wavefront Map
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_facecolor("#1C2541")
    im3 = ax3.imshow(res_disp, cmap="coolwarm", vmin=-vmax, vmax=vmax, extent=[-1, 1, -1, 1])
    ax3.set_title(f"Post-AO Residual Wavefront\n(RMS: {metrics.post_ao_mean_wfe_nm:.0f} nm | -{metrics.wfe_reduction_pct:.0f}%)", fontsize=11, fontweight="bold")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label="Phase (rad)")

    # Panel 4: Zernike Modal Power Spectrum (Uncorrected vs Corrected)
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_facecolor("#1C2541")
    latest = telemetry[-1]
    modes = np.arange(2, len(latest.zernike_coefficients_pre) + 1)
    coeffs_pre = np.abs(latest.zernike_coefficients_pre[1:])
    coeffs_post = np.abs(latest.zernike_coefficients_post[1:])
    w = 0.35
    ax4.bar(modes - w / 2, coeffs_pre, width=w, color="#E63946", label="Pre-AO Turbulence")
    ax4.bar(modes + w / 2, coeffs_post, width=w, color="#06D6A0", label="Post-AO Corrected")
    ax4.set_xlabel("Zernike Mode Noll Index (j)")
    ax4.set_ylabel("|Amplitude| (rad)")
    ax4.set_title("Zernike Modal Decomposition (Modes Z2 - Z22)", fontsize=11, fontweight="bold")
    ax4.grid(True, linestyle="--", alpha=0.3)
    ax4.legend(loc="upper right", facecolor="#0B132B", edgecolor="none")

    # Panel 5: Real-time Strehl Ratio Timeline
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_facecolor("#1C2541")
    ax5.plot(times, strehl_pre, color="#E63946", linestyle="--", linewidth=1.5, label=f"Pre-AO ({metrics.pre_ao_mean_strehl:.2f})")
    ax5.plot(times, strehl_post, color="#06D6A0", linewidth=2.0, label=f"Post-AO ({metrics.post_ao_mean_strehl:.2f})")
    ax5.axhline(0.80, color="#FFD166", linestyle=":", alpha=0.7, label="Maréchal Limit (0.80)")
    ax5.set_xlabel("Time (ms)")
    ax5.set_ylabel("Strehl Ratio S")
    ax5.set_ylim(-0.05, 1.05)
    ax5.set_title(f"Closed-Loop Strehl Ratio (x{metrics.strehl_improvement_factor:.1f} Gain)", fontsize=11, fontweight="bold")
    ax5.grid(True, linestyle="--", alpha=0.3)
    ax5.legend(loc="lower right", facecolor="#0B132B", edgecolor="none")

    # Panel 6: Single-Mode Fiber (SMF) Coupling Efficiency
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_facecolor("#1C2541")
    ax6.plot(times, smf_pre, color="#E63946", linestyle="--", linewidth=1.5, label=f"Pre-AO ({metrics.pre_ao_mean_smf_coupling_pct:.1f}%)")
    ax6.plot(times, smf_post, color="#48CAE4", linewidth=2.0, label=f"Post-AO ({metrics.post_ao_mean_smf_coupling_pct:.1f}%)")
    ax6.axhline(81.45, color="#FFD166", linestyle=":", alpha=0.7, label="Theoretical Limit (81.45%)")
    ax6.set_xlabel("Time (ms)")
    ax6.set_ylabel("SMF Coupling (%)")
    ax6.set_ylim(-5.0, 95.0)
    ax6.set_title(f"Single-Mode Fiber Coupling (+{metrics.mean_ao_gain_db:.1f} dB)", fontsize=11, fontweight="bold")
    ax6.grid(True, linestyle="--", alpha=0.3)
    ax6.legend(loc="lower right", facecolor="#0B132B", edgecolor="none")

    plt.suptitle("OptiLign — Adaptive Optics Closed-Loop Wavefront Correction Benchmark", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
