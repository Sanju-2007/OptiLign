"""Integration tests for OptiLign CLI commands."""

import subprocess
import sys
from pathlib import Path

import pytest


def test_cli_version():
    res = subprocess.run(
        [sys.executable, "-m", "fsoc_pat", "--version"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "0.1.0" in res.stdout or "0.1.0" in res.stderr or "fsoc-pat" in res.stdout or "fsoc-pat" in res.stderr


def test_cli_run_command(tmp_path: Path):
    out_dir = tmp_path / "sim_out"
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "fsoc_pat",
            "run",
            "--duration", "0.4",
            "--dt", "0.05",
            "--seed", "42",
            "--output", str(out_dir),
            "--html",
            "--plot",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert (out_dir / "telemetry.csv").exists()
    assert (out_dir / "mission_report.md").exists()
    assert (out_dir / "mission_report.html").exists()
    assert (out_dir / "mission_summary.png").exists()


def test_cli_benchmark_command(tmp_path: Path):
    json_out = tmp_path / "bench.json"
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "fsoc_pat",
            "benchmark",
            "--frames", "5",
            "--output", str(json_out),
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert json_out.exists()
    assert "Classical Centroid" in res.stdout
    assert "BeaconNet CNN" in res.stdout


def test_cli_monte_carlo_command(tmp_path: Path):
    mc_out = tmp_path / "mc.json"
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "fsoc_pat",
            "monte-carlo",
            "--runs", "2",
            "--duration", "0.2",
            "--output", str(mc_out),
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert mc_out.exists()


def test_cli_report_command(tmp_path: Path):
    # First run a short sim to create a telemetry CSV
    sim_dir = tmp_path / "sim"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "fsoc_pat",
            "run",
            "--duration", "0.2",
            "--dt", "0.05",
            "--output", str(sim_dir),
        ],
        check=True,
    )

    rep_dir = tmp_path / "rep"
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "fsoc_pat",
            "report",
            "--input", str(sim_dir / "telemetry.csv"),
            "--output", str(rep_dir),
            "--plot",
        ],
        capture_output=True,
        text=True,
    )
    assert (rep_dir / "report.html").exists()
    assert (rep_dir / "report.md").exists()
    assert (rep_dir / "report_plots.png").exists()


def test_cli_ao_command(tmp_path: Path):
    ao_dir = tmp_path / "ao_out"
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "fsoc_pat",
            "ao",
            "--duration", "0.05",
            "--dt", "0.005",
            "--output", str(ao_dir),
            "--plot",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert (ao_dir / "ao_benchmark_summary.json").exists()
    assert (ao_dir / "ao_benchmark_summary.png").exists()
    assert "ADAPTIVE OPTICS CLOSED-LOOP BENCHMARK RESULTS" in res.stdout
    assert "Single-Mode Fiber Coupling" in res.stdout


def test_cli_jitter_command(tmp_path: Path):
    jitter_dir = tmp_path / "jitter_out"
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "fsoc_pat",
            "jitter",
            "--duration", "0.4",
            "--dt", "0.002",
            "--output", str(jitter_dir),
            "--plot",
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert (jitter_dir / "jitter_benchmark_summary.json").exists()
    assert (jitter_dir / "jitter_benchmark_summary.png").exists()
    assert "PREDICTIVE ANTI-JITTER BENCHMARK RESULTS" in res.stdout
    assert "OptiLign Zero-Phase FF" in res.stdout


def test_cli_video_benchmark_command(tmp_path: Path):
    video_dir = tmp_path / "video_out"
    res = subprocess.run(
        [
            sys.executable,
            "-m",
            "fsoc_pat",
            "video-benchmark",
            "--generate",
            "--duration", "1.0",
            "--motion", "figure_of_8",
            "--output", str(video_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert (video_dir / "benchmark_sample.mp4").exists()
    assert (video_dir / "benchmark_sample.csv").exists()
    assert (video_dir / "video_benchmark_summary.json").exists()
    assert "FSOC PAT VIDEO BENCHMARK EVALUATION RESULTS" in res.stdout
    assert "100% COMPLIANT" in res.stdout


