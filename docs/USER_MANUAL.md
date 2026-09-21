# OptiLign: FSOC Virtual Camera PAT System — User Manual

**Comprehensive Operational Guide & Reference Manual**  
*Software Suite:* `OptiLign` (`fsoc-pat`) &bull; *Version:* 2.4.0  
*Platforms Supported:* macOS, Linux, Windows (WSL / Native Python)  

---

## 1. System Overview & Architecture

**OptiLign** is an AI-assisted, physics-based software suite for Free-Space Optical Communication (FSOC) Pointing, Acquisition, and Tracking (PAT). It simulates the full coarse-to-fine alignment pipeline without requiring physical gimbals, optical benches, or high-cost sensors.

### Key Capabilities
- **Configurable Virtual Environments**: Supports Space Inter-Satellite Links (ISL), LEO Downlinks, Airborne UAVs, Maritime Vessels, Urban Rooftops, and user-defined custom platforms.
- **Custom Telemetry Schemas**: Configure custom telemetry channels on demand and stream data to Memory, CSV, JSON-Lines, and real-time UDP sockets.
- **Dual-Stage Pointing Architecture**: Integrates a 2-DOF virtual coarse gimbal with an 800–1000 Hz Fast Steering Mirror (FSM) utilizing DC momentum offloading to achieve 18–22 nano-radian pointing precision.
- **Adaptive Perception**: Morphological top-hat clutter rejection with sub-pixel intensity centroiding.
- **Supervisory PAT State Machine**: Autonomous transitions across Standby, Spiral Search, Acquisition, Fine Tracking, and Reacquisition upon cloud fade.
- **Automated Performance Logging**: Auto-generates standardized reports containing duration, FPS, acquisition time, average and peak tracking error, lock retention rate, and processing time.
- **Dual Interfaces**: Interactive 30 FPS Web GUI Dashboard and rich Command Line Interface (CLI).

---

## 2. Prerequisites & Installation

### 2.1 System Prerequisites
- **Python**: Version 3.10, 3.11, 3.12, 3.13, or 3.14.
- **Operating System**: macOS (Apple Silicon / Intel), Linux (Ubuntu 20.04+, Debian, Fedora, Arch), or Windows 10/11.
- **Hardware**: Any modern dual-core CPU with at least 4 GB RAM. No dedicated GPU required.

### 2.2 Installation Steps

#### Option A: Direct Installation from Repository (Recommended)
```bash
# 1. Clone or navigate to the OptiLign repository
cd /path/to/OptiLign

# 2. Create and activate a clean Python virtual environment
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# 3. Install OptiLign and dependencies in editable mode
pip install -e .

# 4. Verify installation
fsoc-pat --version
```

#### Option B: Using `uv` Package Manager (Fast)
```bash
uv pip install -e .
```

### 2.3 Running the Verification Test Suite
Verify that all 243 automated unit and integration tests pass:
```bash
pytest --tb=short
```
Expected output:
```text
============================= 243 passed in 50.65s =============================
```

---

## 3. Standalone Application CLI Operations

The `fsoc-pat` command-line executable provides 6 primary subcommands:

```text
fsoc-pat [-h] [--version] {run, benchmark, monte-carlo, report, gui, env} ...
```

### 3.1 `fsoc-pat run` — Execute Single PAT Simulation
Runs an end-to-end closed-loop simulation:
```bash
# Run nominal 5.0-second closed-loop simulation with HTML report and plot
fsoc-pat run --duration 5.0 --output results/run1 --plot --html

# Run simulation with custom duration, timestep, and random seed
fsoc-pat run --duration 10.0 --dt 0.02 --seed 123 --output results/seed123
```
**Flags:**
- `--duration <float>`: Duration in seconds (default: 5.0).
- `--dt <float>`: Discrete timestep in seconds (default: 0.02, representing 50 Hz).
- `--seed <int>`: Random seed for deterministic reproducibility.
- `--output <dir>`: Directory to save `telemetry.csv`, `report.html`, and `report_plots.png`.
- `--plot`: Automatically generate 6-panel publication telemetry figure.
- `--html`: Automatically generate interactive HTML telemetry report.

### 3.2 `fsoc-pat env` — Multi-Environment Studio & Presets
Manage virtual environments, inspect telemetry schemas, and run multi-environment simulations:

```bash
# 1. List all built-in environment presets and custom telemetry channels
fsoc-pat env list

# 2. Run simulation in a specific environment preset
fsoc-pat env run --preset space_isl --duration 2.0
fsoc-pat env run --preset leo_downlink --duration 2.0
fsoc-pat env run --preset airborne_uav --duration 2.0
fsoc-pat env run --preset maritime --duration 2.0
fsoc-pat env run --preset urban_rooftop --duration 2.0

# 3. Run simulation using a custom JSON environment definition
fsoc-pat env run --config my_custom_env.json --duration 5.0

# 4. Run comparative tracking accuracy benchmark across all environments
fsoc-pat env benchmark --duration 1.0
```

### 3.3 `fsoc-pat benchmark` — Classical vs. Adaptive Detector Benchmark
Benchmarks computer vision detector processing latency, detection rate, and centroid accuracy:
```bash
fsoc-pat benchmark --frames 100 --output results/detector_benchmark.json
```

### 3.4 `fsoc-pat monte-carlo` — Statistical Robustness Study
Runs Monte Carlo randomized passes to evaluate tracking error distributions under varying turbulence and vibration seeds:
```bash
fsoc-pat monte-carlo --runs 20 --duration 3.0 --output results/monte_carlo.json
```

### 3.5 `fsoc-pat report` — Post-Processing Report Generator
Generates self-contained HTML and Markdown performance reports from any existing telemetry CSV file:
```bash
fsoc-pat report --input results/telemetry.csv --output results/report --title "LEO Downlink Mission" --plot
```

### 3.6 `fsoc-pat gui` — Launch Web Dashboard
Launches the interactive real-time Web Dashboard:
```bash
fsoc-pat gui --port 8088 --host 127.0.0.1
```

---

## 4. Interactive Web Dashboard GUI Guide

The OptiLign Web GUI provides a real-time, browser-based operator dashboard accessible via `http://localhost:8088`:

```
+-----------------------------------------------------------------------------------------------+
|  OPTILIGN FSOC PAT OPERATOR DASHBOARD                                       [MODE: AUTO-TRACK]|
+------------------------------------+----------------------------------------------------------+
|  VIRTUAL CAMERA VIEWPORT           |  TELEMETRY & STATUS HUD                                  |
|                                    |  • PAT State: TRACKING (Locked)                          |
|  [ Live Synthetic Sensor Feed ]    |  • Coarse Gimbal: Pan +12.4°, Tilt -3.2°                 |
|  • Green Crosshair: Boresight      |  • FSM Deflection: Pan +412 µrad, Tilt -88 µrad          |
|  • Blue Box: Target Detection      |  • Net Optical Error: 0.021 µrad (21 nrad)               |
|  • Red Circle: Estimated Position  |  • Link Margin: +14.2 dB (BER: 1.0e-9)                   |
|  • Amber Ring: Uncertainty Ellipse |  • Simulation FPS: 30.2 FPS (Latency: 2.3 ms)            |
+------------------------------------+----------------------------------------------------------+
|  CONTROLS & CONFIGURATION          |  REAL-TIME PERFORMANCE CHARTS                            |
|  • Preset: [Urban Rooftop v]       |  [ Yaw / Pitch Error Rolling Time History (60s) ]        |
|  • Mode: [Auto Track] [Manual]     |  • Cyan Line: Azimuth Error                              |
|  • Target Speed Slider: [0.4 rad/s]|  • Amber Line: Elevation Error                           |
|  • Disturbance Injection: [Jitter] |  • Red Dashed Line: Diffraction Limit Benchmark          |
|  • Slew PTZ Jog Buttons            +----------------------------------------------------------+
|  • Start / Pause / Reset           |  EVENT & SYSTEM LOG CONSOLE                              |
|                                    |  [18:42:01] Target Beacon Detected (u=320.1, v=240.0)   |
+------------------------------------+----------------------------------------------------------+
```

### 4.1 GUI Interface Components
1. **Virtual Camera Viewport**: Real-time canvas rendering the synthetic sensor focal plane array (diffraction beacon, stars, bloom, and noise) with dynamic HUD overlays.
2. **Telemetry & Status HUD**: Live gauges displaying current PAT state (`SEARCH`, `ACQUISITION`, `TRACKING`, `REACQUISITION`), Coarse Gimbal Pan/Tilt, FSM Deflection ($\mu$rad), Net Pointing Error, Link Margin (dB), and Bit Error Rate.
3. **Control Panel**:
   - **Mode Selector**: Switch between `Auto Track` (closed-loop PAT), `Manual Jog` (operator arrow controls), and `Hold Position`.
   - **Environment Preset Dropdown**: Instantly switch simulation profiles (Space ISL, LEO Downlink, UAV, Maritime, Urban).
   - **PTZ Jog Controls**: Manually steer camera pan and tilt viewports.
4. **Real-Time Rolling Telemetry Charts**: Canvas-rendered live graphs plotting Azimuth and Elevation pointing errors over the last 60 seconds against the optical diffraction limit.
5. **System Log Console**: Real-time event feed timestamping acquisitions, track losses, reacquisition scans, and link fades.

---

## 5. Step-by-Step Tutorial: Creating Custom Virtual Environments

OptiLign allows creating custom virtual environments with custom telemetry schemas either programmatically via Python or through JSON configuration files.

### 5.1 Programmatic Python Builder Example
Create a script `create_my_env.py`:
```python
from fsoc_pat.environments.builder import VirtualEnvironmentBuilder
from fsoc_pat.environments.models import AtmosphereModel
from fsoc_pat.environments.runner import EnvironmentSimulationRunner

# 1. Initialize fluent builder
my_env = (
    VirtualEnvironmentBuilder("High_Altitude_Balloon")
    .with_platform(
        altitude_m=22000.0,
        sway_amplitude_deg=0.05,
        sway_freq_hz=0.1,
        jitter_frequencies_hz=[10.0, 30.0],
        jitter_amplitudes_urad=[12.0, 5.0],
        wind_gust_buffet_std_urad=6.0,
    )
    .with_atmosphere(
        model=AtmosphereModel.STRATOSPHERE,
        cn2=1.0e-16,
        scintillation_index=0.03,
    )
    .with_optical_payload(
        laser_wavelength_nm=1550.0,
        laser_power_mw=1000.0,
        fsm_bandwidth_hz=1000.0,
    )
    # Add custom telemetry channels
    .with_custom_telemetry_channel(
        name="balloon_envelope_pressure_kpa",
        unit="kPa",
        description="Internal lifting cell gas pressure",
        default_value=101.3,
    )
    .with_custom_telemetry_channel(
        name="ambient_ozone_ppm",
        unit="ppm",
        description="Atmospheric ozone concentration",
        default_value=4.2,
    )
    .build()
)

# 2. Export environment definition to JSON
my_env.to_json_file("balloon_env.json")
print("Saved custom environment definition to balloon_env.json")

# 3. Execute closed-loop simulation
runner = EnvironmentSimulationRunner(my_env, duration_s=2.0, dt=0.02)
results = runner.run(csv_output_path="balloon_telemetry.csv")

print(f"Coarse Gimbal RMS Error: {results.coarse_rms_urad:.2f} µrad")
print(f"Fine Optical FSM RMS   : {results.fine_rms_urad:.4f} µrad ({results.fine_rms_urad*1000:.1f} nrad)")
print(f"Fine Lock Availability : {results.fine_lock_availability_pct:.1f}%")
```

### 5.2 JSON Environment Definition Format
Save as `custom_env.json`:
```json
{
  "name": "Custom_Stratospheric_Platform",
  "description": "Custom high-altitude optical communication node.",
  "platform": {
    "base_altitude_m": 20000.0,
    "sway_amplitude_deg": 0.05,
    "sway_frequency_hz": 0.1,
    "jitter_frequencies_hz": [15.0, 45.0],
    "jitter_amplitudes_urad": [10.0, 4.0],
    "wind_gust_buffet_std_urad": 5.0
  },
  "atmosphere": {
    "model": "stratosphere",
    "cn2": 2e-16,
    "scintillation_index": 0.04
  },
  "disturbances": {
    "sensor_read_noise_std": 4.0,
    "background_clutter_count": 10
  },
  "payload": {
    "laser_wavelength_nm": 1550.0,
    "laser_power_mw": 800.0,
    "fsm_max_deflection_urad": 5000.0,
    "fsm_bandwidth_hz": 800.0
  },
  "telemetry_channels": [
    "custom_airframe_temp_c",
    "gondola_battery_v"
  ]
}
```
Run this environment from the CLI:
```bash
fsoc-pat env run --config custom_env.json --duration 3.0
```

---

## 6. Demonstration Scripts Walkthrough

Three dedicated demonstration scripts are provided in the repository:

### 6.1 `demo_custom_environments.py` (Master Verification Demo)
Executes a multi-environment benchmark comparing Deep Space ISL, LEO Downlink, Airborne UAV, Maritime Vessel, Urban Rooftop, and Custom Balloon. Exports telemetry CSVs and renders a 6-panel publication plot:
```bash
python3 demo_custom_environments.py
```
Output plot saved to: `results/demo_environments/multi_environment_comparison.png`.

### 6.2 `demo_full_system.py` (Complete Closed-Loop PAT Verification)
Executes closed-loop tracking with Gaussian beam propagation, generates interactive HTML telemetry reports, and saves trajectory tracking graphs:
```bash
python3 demo_full_system.py
```

### 6.3 `run_gui.py` (Zero-Config GUI Launcher)
Starts the interactive Web Dashboard server:
```bash
python3 run_gui.py
```

---

## 7. 3–5 Minute Demonstration Video Script

This script can be used for recording a video demonstration of the OptiLign virtual camera tracking system:

### Video Scene Breakdown (Total Duration: 4 Minutes)

#### Scene 1: Introduction & Operational Context (0:00 – 0:45)
- **Visual**: Title slide: *OptiLign — Free-Space Optical Communication Coarse PAT Simulator*. Cut to diagram of satellite-to-ground narrow laser beam alignment.
- **Narrator**:  
  *"Free Space Optical Communication provides terabit-class data transmission with zero electromagnetic interference, but requires aligning narrow laser beams between mobile platforms. Coarse alignment is the critical first stage: locating, identifying, and keeping the remote optical beacon centered within the camera viewport. OptiLign is an end-to-end software simulation platform that models and verifies this tracking process without expensive physical gimbals."*

#### Scene 2: System Architecture & Subsystems (0:45 – 1:30)
- **Visual**: Screen capture showing code repository, running `pytest --tb=short` (243 passed), and system architecture diagram.
- **Narrator**:  
  *"OptiLign implements physical pinhole camera projection, virtual gimbal actuators with motor inertia and slew rate limits, synthetic sensor rendering with atmospheric turbulence, morphological beacon perception, and an adaptive Extended Kalman filter. Crucially, it features a dual-stage architecture: a 2-DOF coarse gimbal for wide acquisition, cooperating with an 800 Hz Fast Steering Mirror for jitter rejection."*

#### Scene 3: Interactive Web Dashboard Demonstration (1:30 – 2:30)
- **Visual**: Web browser open at `http://localhost:8088`. Demonstrating live camera viewport, target beacon orbiting, crosshairs tracking, switching from Auto-Track to Manual Jog mode, and showing live telemetry HUD.
- **Narrator**:  
  *"Here is OptiLign's real-time Web Dashboard running at 30 FPS. In the viewport, you see the synthetic sensor feed with sensor noise and star clutter. Notice the supervisory PAT state machine: it starts in Search, acquires the beacon, centers the gimbal, and locks tracking. The rolling charts display azimuth and elevation pointing errors, showing steady sub-micro-radian alignment well inside the diffraction limit."*

#### Scene 4: Multi-Environment Studio & Benchmark (2:30 – 3:30)
- **Visual**: Terminal executing `fsoc-pat env benchmark` and displaying comparative ASCII results table. Cut to the 6-panel Matplotlib plot `multi_environment_comparison.png`.
- **Narrator**:  
  *"OptiLign includes a Virtual Environment Engine supporting diverse operational profiles: Deep Space vacuum, LEO Downlink, Airborne UAV vibration, Maritime wave swell, and Urban thermal plumes. Across all environments, our dual-stage FSM achieves 18 to 22 nano-radians of residual pointing error with 100% fine optical lock availability."*

#### Scene 5: Automated Performance Logging & Conclusion (3:30 – 4:00)
- **Visual**: Opening `results/report.html` in browser, highlighting the System Performance Log (Duration, FPS, Acquisition Time, Average/Max Error, Lock Retention, Processing Time) and Link Budget margin.
- **Narrator**:  
  *"OptiLign automatically generates standardized performance reports in HTML, Markdown, and CSV, logging all required metrics. OptiLign delivers an accessible, high-fidelity platform for advancing autonomous optical communication tracking algorithms. Thank you."*

---

## 8. Troubleshooting & Frequently Asked Questions (FAQ)

### Q1: The Web GUI port 8088 is already in use. How do I change the port?
**Answer**: Specify a custom port via CLI:
```bash
fsoc-pat gui --port 9090
```

### Q2: How do I export simulation telemetry directly to my own data pipeline?
**Answer**: Use the `--output` parameter with `fsoc-pat run` or `fsoc-pat env run`:
```bash
fsoc-pat env run --preset space_isl --duration 5.0 --output my_data/
```
The output directory will contain `Space_ISL_telemetry.csv` containing every recorded step, state, and custom telemetry channel.

### Q3: Why does coarse gimbal error remain around 1.5 mrad while fine error is 0.02 µrad?
**Answer**: This is the intended behavior of a dual-stage aerospace pointing system. The coarse mechanical gimbal has high inertia and motor lag, holding the target within a $\approx 1.5\ \text{mrad}$ corridor. The high-speed Fast Steering Mirror (FSM) cancels the residual error down to 20 nano-radians, and the DC momentum offloading loop ensures the gimbal continues tracking the low-frequency motion so the FSM never saturates.
