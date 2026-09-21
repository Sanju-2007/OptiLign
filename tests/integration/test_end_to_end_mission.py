"""Integration test for full end-to-end 7-phase mission simulation."""

import subprocess
import sys
from pathlib import Path

import pytest


def test_master_end_to_end_mission_execution():
    """Verify that demo_full_system.py runs to completion and produces expected artifacts."""
    script = Path("demo_full_system.py")
    assert script.exists()

    res = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"Error: {res.stderr}"

    # Verify all expected outputs
    assert Path("demo_full_system.png").exists()
    assert Path("results/mission_telemetry.csv").exists()
    assert Path("results/mission_summary.json").exists()
    assert Path("results/mission_report.md").exists()
    assert Path("results/mission_report.html").exists()

    # Check key console output
    assert "OPTILIGN — MASTER END-TO-END FSOC MISSION SIMULATION" in res.stdout
    assert "MISSION RESULTS EXECUTIVE SUMMARY" in res.stdout
    assert "Pointing Accuracy (MRE):" in res.stdout
    assert "Optical Link Availability:" in res.stdout
