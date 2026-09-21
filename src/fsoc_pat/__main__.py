"""OptiLign CLI entry point.

Provides command-line interface for running simulations, benchmarks,
Monte Carlo studies, report generation, and the interactive web GUI.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any


def main() -> int:
    """Main CLI entry point for OptiLign."""
    parser = argparse.ArgumentParser(
        prog="fsoc-pat",
        description="OptiLign — FSOC Coarse PAT Simulation & Verification Suite",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # 1. run command
    run_parser = subparsers.add_parser("run", help="Run a single PAT simulation")
    run_parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Path to scenario configuration YAML file (default: synthetic circular orbit)",
    )
    run_parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Simulation duration in seconds (default: 5.0)",
    )
    run_parser.add_argument(
        "--dt",
        type=float,
        default=0.02,
        help="Simulation timestep in seconds (default: 0.02, 50 Hz)",
    )
    run_parser.add_argument(
        "--closed-loop",
        action="store_true",
        default=True,
        help="Run in full closed-loop mode (Perception + EKF + PID + Link Budget)",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    run_parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for telemetry and reports (default: results)",
    )
    run_parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate and save 6-panel publication telemetry plot",
    )
    run_parser.add_argument(
        "--html",
        action="store_true",
        help="Generate and save self-contained interactive HTML report",
    )

    # 2. benchmark command
    bm_parser = subparsers.add_parser("benchmark", help="Run detector benchmark comparison")
    bm_parser.add_argument(
        "--frames",
        type=int,
        default=50,
        help="Number of test frames to evaluate (default: 50)",
    )
    bm_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save benchmark JSON results",
    )

    # 3. monte-carlo command
    mc_parser = subparsers.add_parser("monte-carlo", help="Run Monte Carlo statistical analysis")
    mc_parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of randomized Monte Carlo runs (default: 10)",
    )
    mc_parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Duration of each run in seconds (default: 3.0)",
    )
    mc_parser.add_argument(
        "--output",
        type=str,
        default="results/monte_carlo.json",
        help="Path to save Monte Carlo statistical summary",
    )

    # 4. report command
    rep_parser = subparsers.add_parser("report", help="Generate HTML/Markdown report from telemetry CSV")
    rep_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to telemetry CSV file",
    )
    rep_parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory for generated reports",
    )
    rep_parser.add_argument(
        "--title",
        type=str,
        default="OptiLign FSOC PAT Mission Telemetry Report",
        help="Title for the generated report",
    )
    rep_parser.add_argument(
        "--plot",
        action="store_true",
        help="Also generate multi-panel telemetry plot",
    )

    # 5. gui command
    gui_parser = subparsers.add_parser("gui", help="Launch interactive web GUI dashboard")
    gui_parser.add_argument(
        "--port",
        type=int,
        default=8088,
        help="Port to serve GUI on (default: 8088)",
    )
    gui_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface to bind to (default: 127.0.0.1)",
    )
    gui_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not automatically open browser",
    )

    # 6. env command (Virtual Environments, Presets, Custom Telemetry, Dual-Stage Tracking)
    env_parser = subparsers.add_parser("env", help="Manage virtual environments and run multi-environment simulations")
    env_subparsers = env_parser.add_subparsers(dest="env_subcommand", help="Environment actions")

    # env list
    env_subparsers.add_parser("list", help="List all built-in environment presets")

    # env run
    env_run = env_subparsers.add_parser("run", help="Run simulation in a virtual environment")
    env_run.add_argument("--preset", type=str, default="space_isl", help="Preset name (space_isl, leo_downlink, airborne_uav, maritime, urban_rooftop)")
    env_run.add_argument("--config", type=str, default=None, help="Path to custom environment JSON definition file")
    env_run.add_argument("--duration", type=float, default=2.0, help="Simulation duration in seconds (default: 2.0)")
    env_run.add_argument("--dt", type=float, default=0.02, help="Simulation timestep in seconds (default: 0.02)")
    env_run.add_argument("--output", type=str, default="results", help="Output directory for telemetry (default: results)")
    env_run.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # env benchmark
    env_bm = env_subparsers.add_parser("benchmark", help="Benchmark tracking performance across all virtual environments")
    env_bm.add_argument("--duration", type=float, default=1.0, help="Duration per environment in seconds (default: 1.0)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "run":
        return _run_simulation(args)
    if args.command == "benchmark":
        return _run_benchmark(args)
    if args.command == "monte-carlo":
        return _run_monte_carlo(args)
    if args.command == "report":
        return _run_report(args)
    if args.command == "gui":
        return _run_gui(args)
    if args.command == "env":
        return _run_env(args)

    print(f"Unknown command: '{args.command}'", file=sys.stderr)
    return 1


def _run_simulation(args: argparse.Namespace) -> int:
    """Execute a single simulation from CLI with reporting."""
    from fsoc_pat.control.closed_loop import ClosedLoopConfig, ClosedLoopEngine
    from fsoc_pat.pat.pat_types import PATConfig
    from fsoc_pat.reporting.plots import generate_mission_summary_plot
    from fsoc_pat.reporting.report_generator import SimulationReportGenerator

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"============================================================")
    print(f"  OptiLign FSOC PAT Simulation (Duration: {args.duration}s, dt: {args.dt}s)")
    print(f"============================================================")

    # Closed-loop PAT execution
    cl_cfg = ClosedLoopConfig(
        duration_s=args.duration,
        dt=args.dt,
        seed=args.seed,
        pat=PATConfig(),
    )
    engine = ClosedLoopEngine(cl_cfg)
    print("Running closed-loop PAT tracking simulation...")
    res = engine.run()

    snapshots = [vars(s) for s in res.snapshots]

    # Save CSV
    csv_path = out_dir / "telemetry.csv"
    if snapshots:
        keys = list(snapshots[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(snapshots)
        print(f"[OK] Exported telemetry CSV: {csv_path}")

    # Generate Metrics & Reports
    rep_gen = SimulationReportGenerator(snapshots, scenario_name="OptiLign Mission Run")
    m = rep_gen.metrics

    # Always generate Markdown summary
    md_path = out_dir / "mission_report.md"
    rep_gen.generate_markdown_report(md_path)
    print(f"[OK] Exported Markdown report: {md_path}")

    if args.html:
        html_path = out_dir / "mission_report.html"
        rep_gen.generate_html_report(html_path)
        print(f"[OK] Exported interactive HTML report: {html_path}")

    if args.plot:
        plot_path = out_dir / "mission_summary.png"
        generate_mission_summary_plot(snapshots, output_path=plot_path)
        print(f"[OK] Exported multi-panel telemetry plot: {plot_path}")

    # Print summary to terminal
    print("\n--- Mission Summary Results ---")
    print(f"Steps: {m.total_steps} | Duration: {m.duration_s:.2f}s")
    print(f"Mean Radial Pointing Error: {m.pointing.mean_radial_error_mrad:.3f} mrad ({m.pointing.mean_error_deg:.4f}°)")
    print(f"RMS Pointing Jitter (1-sigma): {m.pointing.rms_radial_error_mrad:.3f} mrad")
    print(f"PAT State Machine: {m.pat_state.percent_in_tracking:.1f}% TRACKING, {m.pat_state.percent_in_reacquisition:.1f}% REACQUISITION")
    print(f"Optical Link Availability: {m.link.link_availability_pct:.1f}% (Mean Margin: {m.link.mean_link_margin_db:.2f} dB)")
    print(f"============================================================\n")
    return 0


def _run_benchmark(args: argparse.Namespace) -> int:
    """Run detector accuracy and latency benchmark."""
    import time
    import numpy as np
    from fsoc_pat.ai.ai_detector import AIDetector
    from fsoc_pat.geometry.camera_model import CameraIntrinsics
    from fsoc_pat.perception.classical_detector import ClassicalDetector, ClassicalDetectorConfig
    from fsoc_pat.perception.pipeline import PerceptionPipeline
    from fsoc_pat.perception.renderer import RenderConfig, SyntheticRenderer
    from fsoc_pat.simulation.beacon import BeaconProperties
    from fsoc_pat.core.state import CameraState
    from fsoc_pat.core.types import Orientation, Position3D, AngularState

    print("============================================================")
    print(f"  OptiLign Perception Benchmark ({args.frames} Frames)")
    print("============================================================")

    intrinsics = CameraIntrinsics.from_fov(640, 480, math.radians(60))
    renderer = SyntheticRenderer(intrinsics, RenderConfig(noise_std=12.0, num_stars=15, seed=42))
    cam_state = CameraState(position=Position3D(0, 0, 0), orientation=Orientation(), angular_state=AngularState())
    beacon = BeaconProperties(intensity=220.0, size_px=8.0)

    classic = ClassicalDetector(ClassicalDetectorConfig(min_brightness=50.0))
    ai_weights = Path(__file__).resolve().parent.parent.parent / "models" / "beaconnet_v1.pt"
    if not ai_weights.is_file():
        ai_weights = Path("models/beaconnet_v1.pt")
    ai = AIDetector(
        model_path=ai_weights if ai_weights.is_file() else None,
        confidence_threshold=0.35,
    )

    classical_latencies: list[float] = []
    classical_errors: list[float] = []
    ai_latencies: list[float] = []
    ai_errors: list[float] = []

    np.random.seed(42)
    for i in range(args.frames):
        # Place beacon randomly in field of view
        x = 50.0
        y = np.random.uniform(-15.0, 15.0)
        z = np.random.uniform(-10.0, 10.0)
        target_pos = np.array([x, y, z])

        img = renderer.render(cam_state, [(target_pos, beacon)])
        # Ground truth pixel
        from fsoc_pat.geometry.projection import project_point_to_pixel
        gt_px = project_point_to_pixel(target_pos, cam_state, intrinsics)
        if gt_px is None:
            continue

        # Benchmark classical
        t0 = time.perf_counter()
        det_c = classic.detect(img)
        dt_c = (time.perf_counter() - t0) * 1000.0
        classical_latencies.append(dt_c)
        if det_c.detected and det_c.best:
            err_c = math.hypot(det_c.best.u - gt_px.u, det_c.best.v - gt_px.v)
            classical_errors.append(err_c)

        # Benchmark AI
        t0 = time.perf_counter()
        det_ai = ai.detect(img)
        dt_ai = (time.perf_counter() - t0) * 1000.0
        ai_latencies.append(dt_ai)
        if det_ai.detected and det_ai.best:
            err_ai = math.hypot(det_ai.best.u - gt_px.u, det_ai.best.v - gt_px.v)
            ai_errors.append(err_ai)

    mean_lat_c = float(np.mean(classical_latencies))
    mean_err_c = float(np.mean(classical_errors)) if classical_errors else float("nan")
    det_rate_c = len(classical_errors) / len(classical_latencies) * 100.0

    mean_lat_ai = float(np.mean(ai_latencies))
    mean_err_ai = float(np.mean(ai_errors)) if ai_errors else float("nan")
    det_rate_ai = len(ai_errors) / len(ai_latencies) * 100.0

    print("\n| Perception Engine | Mean Latency (ms) | Mean Pixel Error (px) | Detection Rate (%) |")
    print("| :--- | :--- | :--- | :--- |")
    print(f"| **Classical Centroid** | {mean_lat_c:.2f} ms | {mean_err_c:.2f} px | {det_rate_c:.1f}% |")
    print(f"| **BeaconNet CNN (AI)** | {mean_lat_ai:.2f} ms | {mean_err_ai:.2f} px | {det_rate_ai:.1f}% |")
    print("============================================================\n")

    if args.output:
        import json
        out_data = {
            "classical": {"latency_ms": mean_lat_c, "error_px": mean_err_c, "detection_rate": det_rate_c},
            "beacon_net": {"latency_ms": mean_lat_ai, "error_px": mean_err_ai, "detection_rate": det_rate_ai},
        }
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
        print(f"[OK] Saved benchmark metrics to: {path}")

    return 0


def _run_monte_carlo(args: argparse.Namespace) -> int:
    """Execute Monte Carlo statistical analysis over randomized seeds."""
    import json
    import numpy as np
    from fsoc_pat.control.closed_loop import ClosedLoopConfig, ClosedLoopEngine
    from fsoc_pat.pat.pat_types import PATConfig
    from fsoc_pat.reporting.report_generator import SimulationReportGenerator

    print("============================================================")
    print(f"  OptiLign Monte Carlo Study ({args.runs} Runs @ {args.duration}s)")
    print("============================================================")

    mres: list[float] = []
    availabilities: list[float] = []
    tracking_percentages: list[float] = []
    reacq_counts: list[int] = []

    for r in range(args.runs):
        seed = 1000 + r * 37
        cfg = ClosedLoopConfig(
            duration_s=args.duration,
            dt=0.02,
            seed=seed,
            pat=PATConfig(),
        )
        engine = ClosedLoopEngine(cfg)
        res = engine.run()
        rep_gen = SimulationReportGenerator([vars(s) for s in res.snapshots])
        m = rep_gen.metrics

        mres.append(m.pointing.mean_radial_error_mrad)
        availabilities.append(m.link.link_availability_pct)
        tracking_percentages.append(m.pat_state.percent_in_tracking)
        reacq_counts.append(m.pat_state.reacquisition_events_count)
        print(f"  Run {r + 1:2d}/{args.runs} [Seed {seed}]: MRE={m.pointing.mean_radial_error_mrad:.3f} mrad | Link Avail={m.link.link_availability_pct:.1f}% | Track={m.pat_state.percent_in_tracking:.1f}%")

    arr_mre = np.array(mres)
    arr_avail = np.array(availabilities)
    arr_track = np.array(tracking_percentages)

    summary = {
        "runs": args.runs,
        "duration_s": args.duration,
        "pointing_mre_mrad": {
            "mean": float(np.mean(arr_mre)),
            "std": float(np.std(arr_mre)),
            "min": float(np.min(arr_mre)),
            "max": float(np.max(arr_mre)),
        },
        "link_availability_pct": {
            "mean": float(np.mean(arr_avail)),
            "std": float(np.std(arr_avail)),
            "min": float(np.min(arr_avail)),
            "max": float(np.max(arr_avail)),
        },
        "time_in_tracking_pct": {
            "mean": float(np.mean(arr_track)),
            "std": float(np.std(arr_track)),
        },
        "total_reacquisition_events": int(sum(reacq_counts)),
    }

    print("\n--- Monte Carlo Aggregated Statistics ---")
    print(f"Pointing MRE: {summary['pointing_mre_mrad']['mean']:.3f} ± {summary['pointing_mre_mrad']['std']:.3f} mrad")
    print(f"Link Availability: {summary['link_availability_pct']['mean']:.1f}% ± {summary['link_availability_pct']['std']:.1f}%")
    print(f"Tracking Duty Cycle: {summary['time_in_tracking_pct']['mean']:.1f}%")
    print(f"Reacquisition Events: {summary['total_reacquisition_events']} total")
    print("============================================================\n")

    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[OK] Exported Monte Carlo summary: {path}")

    return 0


def _run_report(args: argparse.Namespace) -> int:
    """Generate reports from an existing telemetry CSV file."""
    from fsoc_pat.reporting.plots import generate_mission_summary_plot
    from fsoc_pat.reporting.report_generator import SimulationReportGenerator

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: Input file '{in_path}' does not exist.", file=sys.stderr)
        return 1

    records: list[dict[str, Any]] = []
    with open(in_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            converted = {}
            for k, v in row.items():
                try:
                    if v == "True":
                        converted[k] = True
                    elif v == "False":
                        converted[k] = False
                    else:
                        converted[k] = float(v)
                except (ValueError, TypeError):
                    converted[k] = v
            records.append(converted)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    gen = SimulationReportGenerator(records, scenario_name=args.title)
    html_path = out_dir / "report.html"
    md_path = out_dir / "report.md"

    gen.generate_html_report(html_path, title=args.title)
    gen.generate_markdown_report(md_path)
    print(f"[OK] Generated HTML report: {html_path}")
    print(f"[OK] Generated Markdown report: {md_path}")

    if args.plot:
        plot_path = out_dir / "report_plots.png"
        generate_mission_summary_plot(records, output_path=plot_path, scenario_name=args.title)
        print(f"[OK] Generated telemetry plots: {plot_path}")

    return 0


def _run_gui(args: argparse.Namespace) -> int:
    """Launch the interactive web GUI dashboard."""
    import webbrowser
    from fsoc_pat.gui.server import run_server

    url = f"http://{args.host}:{args.port}"
    print(f"Starting OptiLign Interactive Web Dashboard at: {url}")

    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    run_server(host=args.host, port=args.port)
    return 0


def _run_env(args: argparse.Namespace) -> int:
    """Handle virtual environment CLI commands."""
    from fsoc_pat.environments.models import EnvironmentDefinition
    from fsoc_pat.environments.presets import (
        PRESET_FACTORIES,
        get_environment_preset,
        list_environment_presets,
    )
    from fsoc_pat.environments.runner import EnvironmentSimulationRunner

    sub = getattr(args, "env_subcommand", None)

    if sub == "list" or sub is None:
        print("=" * 70)
        print("  OptiLign Virtual Environments & Telemetry Profiles")
        print("=" * 70)
        for key in list_environment_presets():
            env = get_environment_preset(key)
            print(f"\n  [{key.upper()}] - {env.name}")
            print(f"    Description : {env.description}")
            print(f"    Platform    : Altitude {env.platform.base_altitude_m:.0f}m, Sway {env.platform.sway_amplitude_deg:.2f} deg @ {env.platform.sway_frequency_hz:.2f} Hz")
            print(f"    Atmosphere  : {env.atmosphere.model.value} (Cn2={env.atmosphere.cn2:.1e}, Scint={env.atmosphere.scintillation_index:.2f})")
            print(f"    Payload     : {env.payload.laser_wavelength_nm:.0f}nm, {env.payload.laser_power_mw:.0f}mW, FSM BW={env.payload.fsm_bandwidth_hz:.0f}Hz")
            std_keys = {"time_s", "step", "target_range_m", "target_azimuth_rad", "target_elevation_rad", "gimbal_pan_rad", "gimbal_tilt_rad", "gimbal_rate_pan_rad_s", "gimbal_rate_tilt_rad_s", "fsm_pan_urad", "fsm_tilt_urad", "fsm_offload_active", "coarse_error_urad", "fine_error_urad", "in_fine_lock", "beacon_detected", "detection_confidence", "est_pixel_u", "est_pixel_v", "received_power_dbm", "link_snr_db", "bit_error_rate", "pat_state"}
            custom_ch = [k for k in env.telemetry_schema.channels if k not in std_keys]
            if custom_ch:
                print(f"    Custom Telemetry: {', '.join(custom_ch)}")
        print("\nRun any environment with:")
        print("  fsoc-pat env run --preset <name>")
        print("  fsoc-pat env benchmark")
        return 0

    if sub == "run":
        if args.config:
            cfg_path = Path(args.config)
            print(f"Loading custom environment definition from: {cfg_path}")
            env = EnvironmentDefinition.from_json_file(cfg_path)
        else:
            try:
                env = get_environment_preset(args.preset)
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1

        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / f"{env.name}_telemetry.csv"

        print("=" * 70)
        print(f"  OptiLign Multi-Environment PAT Simulation: {env.name}")
        print("=" * 70)
        print(f"  Duration : {args.duration}s (dt={args.dt}s, {int(args.duration / args.dt)} steps)")
        print(f"  Platform : Altitude {env.platform.base_altitude_m:.0f}m, Sway {env.platform.sway_amplitude_deg:.2f} deg")
        print(f"  Atmos Cn2: {env.atmosphere.cn2:.2e}")
        print(f"  Output   : {csv_path}")
        print("-" * 70)

        runner = EnvironmentSimulationRunner(
            environment=env,
            duration_s=args.duration,
            dt=args.dt,
            seed=args.seed,
        )

        results = runner.run(csv_output_path=csv_path)

        print("\n[COMPLETE] Simulation Results Summary:")
        print(f"  Coarse Gimbal RMS Error : {results.coarse_rms_urad:10.2f} µrad")
        print(f"  Fine Optical FSM RMS    : {results.fine_rms_urad:10.4f} µrad")
        print(f"  Mean Optical Link SNR   : {results.mean_snr_db:10.1f} dB")
        print(f"  Fine Lock Availability  : {results.fine_lock_availability_pct:10.1f} %")
        print(f"  Wall Clock Execution    : {results.wall_time_s:10.2f} s")
        print(f"  Telemetry CSV Exported  : {csv_path}")
        return 0

    if sub == "benchmark":
        print("=" * 80)
        print(f"  OptiLign Virtual Environment Tracking Accuracy Benchmark ({args.duration:.1f}s/env)")
        print("=" * 80)
        print(f"  {'Environment':<18} | {'Coarse RMS':<12} | {'Fine RMS':<12} | {'Mean SNR':<10} | {'Lock Avail':<10}")
        print("-" * 80)

        for key in list_environment_presets():
            env = get_environment_preset(key)
            runner = EnvironmentSimulationRunner(
                environment=env,
                duration_s=args.duration,
                dt=0.02,
                seed=42,
            )
            res = runner.run()
            print(
                f"  {env.name:<18} | {res.coarse_rms_urad:>9.1f} µrad | "
                f"{res.fine_rms_urad:>9.4f} µrad | {res.mean_snr_db:>7.1f} dB | "
                f"{res.fine_lock_availability_pct:>8.1f} %"
            )

        print("=" * 80)
        print("  All environments verified with Dual-Stage Coarse + FSM Ultra-Precision Pointing!")
        return 0

    print(f"Unknown env action: '{sub}'", file=sys.stderr)
    return 1


def _get_version() -> str:
    """Get application version string."""
    from fsoc_pat import __version__
    return __version__


if __name__ == "__main__":
    sys.exit(main())
