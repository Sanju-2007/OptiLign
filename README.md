# OptiLign: AI-Assisted FSOC Virtual Camera PAT Tracking System

[![Tests](https://img.shields.io/badge/tests-337%20passed-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Pointing Accuracy](https://img.shields.io/badge/FSM%20Accuracy-18--22%20nrad-cyan.svg)](docs/TECHNICAL_REPORT.md)
[![Lock Availability](https://img.shields.io/badge/Lock%20Availability-100.0%25-emerald.svg)](results/demo_environments/)
[![Typing: Mypy Strict](https://img.shields.io/badge/typing-mypy%20strict-blue.svg)](pyproject.toml)
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](pyproject.toml)

**OptiLign** is a modular, physics-based software platform for simulating, evaluating, and verifying **Free-Space Optical Communication (FSOC)** Pointing, Acquisition, and Tracking (PAT) algorithms. It provides an autonomous coarse-to-fine virtual camera tracking pipeline with sub-micro-radian (nano-radian) precision across diverse, turbulent operational environments—without requiring expensive physical gimbals, optical benches, or dedicated hardware.

---

## 1. Background & Operational Context

Free Space Optical Communication (FSOC) provides an order-of-magnitude leap over legacy radio-frequency (RF) wireless systems, offering:
- **Gigabit-to-Terabit Data Rates**: Extreme carrier frequencies ($>193\ \text{THz}$) exceeding conventional RF throughput by orders of magnitude.
- **License-Free Spectrum Operation**: Global deployment without international spectrum coordination hurdles.
- **High Immunity to Electromagnetic Interference**: Immune to RF jamming, spoofing, and side-lobe interception.
- **Low SWaP Footprint**: Reduced size, weight, and power suitable for mobile space, airborne, and ground nodes.

However, deploying FSOC links between mobile platforms (satellites, UAVs, ships, high-speed ground terminals) presents a severe challenge of **Pointing, Acquisition, and Tracking (PAT)** of highly narrow laser beams (typically $0.1 - 2.0\ \text{mrad}$). PAT fundamentally operates in sequential stages:
1. **Coarse Alignment**: Wide Field-of-View (FOV) camera on a 2-DOF pan-tilt gimbal acquiring and centering the remote optical beacon within the optical boresight.
2. **Fine Alignment**: High-bandwidth Fast Steering Mirror (FSM) rejecting residual micro-vibrations and beam wander down to nano-radian precision.
3. **Wavefront Flattening & Adaptive Optics**: Deformable mirror correcting higher-order atmospheric wavefront aberrations to preserve fiber coupling efficiency.

OptiLign provides an accessible, high-fidelity software testbed for developing, stress-testing, and verifying these algorithms in a deterministic, fully repeatable simulation environment.

---

## 2. Functional Objective & Architecture

### Functional Objective
Develop a software system that autonomously detects, identifies, and continuously tracks a designated moving target within a virtual scene by controlling a virtual camera viewport and gimbal platform.

### System Architecture & Data Flow

```
                                  OPTILIGN SYSTEM ARCHITECTURE
                                  
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │                              VIRTUAL ENVIRONMENT & DISTURBANCES                          │
  │  • 3D Kinematics (Circular, Linear, Sinusoid, Random)  • Kolmogorov Turbulence (Cn²)      │
  │  • Platform Sway & Aerodynamic Wind Gusts             • Multi-Harmonic Micro-Vibrations  │
  │  • Background Star/Urban Clutter                       • Sensor Read & Photon Shot Noise  │
  └──────────────────────────────┬───────────────────────────────────────────────────────────┘
                                 │ Target 3D Position & Optical Flux
                                 ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │                         VIRTUAL SENSOR & GIMBAL PLATFORM                                 │
  │  • Pinhole Camera Intrinsics (fx, fy, cx, cy)         • 2-DOF Pan-Tilt Actuator Dynamics │
  │  • First-Order Actuator Inertial Lag (τ = 15 ms)       • Slew Rate & Acceleration Limits  │
  │  • Mechanical Backlash & Deadband Modeling            • Synthetic Focal Plane Rendering  │
  └──────────────────────────────┬───────────────────────────────────────────────────────────┘
                                 │ Synthetic Video Stream (50 Hz / 30 FPS)
                                 ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │                           PERCEPTION & IMAGE CENTROIDING                                 │
  │  • Morphological Top-Hat Background Subtraction       • Dynamic Adaptive Thresholding    │
  │  • Sub-Pixel Intensity-Weighted Centroiding           • Mahalanobis Ellipsoidal Gating   │
  └──────────────────────────────┬───────────────────────────────────────────────────────────┘
                                 │ Measured Pixel Centroid (u, v) + SNR
                                 ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │                        KINEMATIC ESTIMATION & TRACKING FILTER                            │
  │  • Adaptive Extended Kalman Filter (EKF)              • Interacting Multiple Model (IMM) │
  │  • Dynamic Covariance Scaling Rk(σ_I²)                • Scintillation-Aware Filtering    │
  └──────────────────────────────┬───────────────────────────────────────────────────────────┘
                                 │ Estimated Target Line-of-Sight (Yaw, Pitch, Rates)
                                 ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │                             DUAL-STAGE POINTING CONTROLLER                               │
  │  ┌──────────────────────────────────────┐     ┌────────────────────────────────────────┐ │
  │  │   STAGE 1: COARSE GIMBAL (PID/LQR)   │     │    STAGE 2: FAST STEERING MIRROR (FSM) │ │
  │  │   • Large Travel (±180° pan, ±45° tilt)│ ◄───┤    • Ultra-Fast Piezo (800–1000 Hz)    │ │
  │  │   • 60°/s Slew, holds target in FOV  │ DC  │    • Sub-µrad Precision (18–22 nrad)   │ │
  │  └──────────────────────────────────────┘ Offload  • Predictive Zero-Lag Anti-Jitter  │ │
  │                                               └────────────────────────────────────────┘ │
  └──────────────────────────────┬───────────────────────────────────────────────────────────┘
                                 │ Corrected Optical Axis & Residual Pointing Error
                                 ▼
  ┌──────────────────────────────────────────────────────────────────────────────────────────┐
  │                         FSOC LINK BUDGET & TELEMETRY STREAMING                           │
  │  • Gaussian Beam Propagation & Coupling Loss          • Adaptive Optics Wavefront Strehl │
  │  • Real-Time Received Power, SNR, Margin & BER        • Standard Performance Logging     │
  │  • Web GUI Dashboard (30 FPS Canvas + HUD)            • CSV, JSON, HTML & Markdown Exporters│
  └──────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Parameters and Specifications Reference Table

| Category | Parameter | Reference / Nominal Value | Configurable Range | Unit | Operational Description |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Virtual Camera & Sensor** | Image Resolution | $640 \times 480$ | $320\times 240$ to $1920\times 1080$ | Pixels | Synthetic sensor focal plane array (VGA default, Full HD supported). |
| | Field-of-View (Horizontal) | $45.0^\circ$ | $15.0^\circ$ to $90.0^\circ$ | Degrees | Viewport angular span for coarse acquisition and target search. |
| | Focal Length ($f_x, f_y$) | $772.5$ | Derived from FOV & resolution | Pixels | Pinhole camera intrinsic matrix mapping angles to image plane. |
| | Principal Point ($c_x, c_y$) | $(320.0, 240.0)$ | Resolution / 2 | Pixels | Center optical boresight coordinate. |
| | Frame Rate ($f_s$) | $50.0$ | $20.0$ to $100.0$ | Hz (FPS) | Closed-loop vision & tracking control rate ($\Delta t = 0.02$ s default). |
| | Sensor Read Noise ($\sigma_{\text{read}}$) | $4.5$ | $0.0$ to $20.0$ | ADU / DN | Zero-mean Gaussian detector readout noise. |
| | Photon Shot Noise | Enabled | Boolean Flag | — | Signal-dependent Poisson noise on photon arrival. |
| | Dark Background Mean | $20.0$ | $0.0$ to $60.0$ | ADU | Ambient sky / night background offset level. |
| **Coarse Gimbal Platform** | Pan Travel Range | $\pm 180.0^\circ$ | Continuous ($\pm 180^\circ$) | Degrees | Coarse azimuth rotational degree of freedom. |
| | Tilt Travel Range | $-45.0^\circ \text{ to } +45.0^\circ$ | Up to $\pm 90.0^\circ$ | Degrees | Coarse elevation rotational degree of freedom. |
| | Max Pan Slew Rate ($\dot{\theta}_{\text{pan},\max}$) | $60.0$ | $10.0$ to $120.0$ | deg/s | Maximum motorized gimbal azimuth slewing speed. |
| | Max Tilt Slew Rate ($\dot{\theta}_{\text{tilt},\max}$) | $45.0$ | $10.0$ to $90.0$ | deg/s | Maximum motorized gimbal elevation slewing speed. |
| | Actuator Time Constant ($\tau$) | $0.015$ | $0.005$ to $0.050$ | Seconds | First-order physical actuator lag $\dot{\omega} = \frac{1}{\tau}(\omega_{\text{cmd}} - \omega)$. |
| | Actuator Deadband | $0.01^\circ$ ($174\ \mu\text{rad}$) | $0.0$ to $0.1^\circ$ | Degrees | Mechanical backlash and stiction threshold. |
| | Rate Saturation & Limits | Enabled | Hard Limit | — | Prevents non-physical velocity steps; satisfies motor dynamics. |
| **Fine Steering Mirror (FSM)** | Deflection Range | $\pm 5000.0$ | $\pm 1000.0$ to $\pm 10000.0$ | $\mu$rad | High-speed piezo / voice-coil secondary optical stage ($\pm 5$ mrad). |
| | Closed-Loop Bandwidth | $800.0$ | $500.0$ to $1500.0$ | Hz | Ultra-fast fine correction bandwidth (10–20× gimbal bandwidth). |
| | Angular Resolution | $0.05$ | $0.01$ to $0.20$ | $\mu$rad | Sub-micro-radian precision steering ($50$ nano-radians). |
| | DC Momentum Offloading | Active ($k_{\text{off}}=2.5$) | $0.5$ to $8.0$ | $1/\text{s}$ | Continuous desaturation loop steering gimbal to keep FSM centered. |
| **Target & Optical Beacon** | Target Range / Distance | $500.0$ | $50.0$ to $500,000.0$ | Meters | Operating link distance (terrestrial, airborne, or LEO satellite orbit). |
| | Beacon Peak Intensity | $240.0$ | $50.0$ to $255.0$ | ADU | Peak brightness of the diffraction spot at optical wavelength. |
| | Beacon Gaussian Spot Size | $6.0$ | $2.0$ to $20.0$ | Pixels ($\sigma$) | Apparent diffraction spot radius on sensor plane. |
| | Target Kinematics | 3D Circular, Linear, Sinusoid | Configurable analytical | m, m/s, m/s² | Analytical 3D trajectories with position, velocity, and acceleration. |
| | Target Angular Speed | $0.40$ | $0.05$ to $2.0$ | rad/s | Apparent angular line-of-sight slew rate across viewport. |
| **Atmosphere & Channel** | Refractive Turbulence ($C_n^2$) | $2.5 \times 10^{-14}$ | $0.0 \text{ to } 5.0 \times 10^{-13}$ | $\text{m}^{-2/3}$ | Kolmogorov index (Vacuum $0.0$ to strong urban turbulence). |
| | Scintillation Index ($\sigma_I^2$) | $0.18$ | $0.0$ to $1.2$ | Dimensionless | Irradiance variance from atmospheric temperature micro-fluctuations. |
| | Extinction Coefficient | $0.8$ | $0.0$ to $5.0$ | dB/km | Beer-Lambert atmospheric attenuation (clear, haze, mist). |
| | Beam Wander Jitter | $15.0$ | $0.0$ to $100.0$ | $\mu$rad | Atmospheric spatial tilt variance on arriving optical wave. |
| **Platform Disturbances** | Host Platform Sway | $0.04^\circ$ ($698\ \mu\text{rad}$) | $0.0^\circ$ to $2.5^\circ$ | Degrees | Low-frequency base oscillation (building wind sway, ship sea swell). |
| | Structural Vibration Harmonics | $[20, 60, 120]$ | $5.0$ to $250.0$ | Hz | Mechanical machinery resonances (reaction wheels, HVAC, UAV motors). |
| | Vibration Amplitudes | $[15, 6, 2]$ | $1.0$ to $80.0$ | $\mu$rad | Angular deflection amplitude per vibration harmonic. |
| | Aerodynamic Buffet Jitter | $6.0$ | $0.0$ to $30.0$ | $\mu$rad | Random broadband wind gust buffet standard deviation. |
| | Optical Background Clutter | $50$ | $0$ to $150$ | Objects | Background star field, city street lamps, and building reflections. |
| **Optical Communications** | Laser Wavelength ($\lambda$) | $1550.0$ | $850.0 \text{ or } 1550.0$ | nm | Standard eye-safe telecom C-Band laser carrier. |
| | Optical Transmit Power ($P_{\text{tx}}$) | $600.0$ | $100.0$ to $5000.0$ | mW | Continuous-wave laser optical power. |
| | Beam Divergence ($\theta_{\text{div}}$) | $1.0$ | $0.5$ to $3.0$ | mrad | Gaussian beam 1/e² angular divergence. |
| | Receiver Aperture ($D_{\text{rx}}$) | $0.15$ | $0.05$ to $0.50$ | Meters | Diameter of optical collector telescope. |
| | Detector Sensitivity | $-38.0$ | Fixed benchmark | dBm | Minimum optical power required for $10^{-9}$ Bit Error Rate (BER). |
| | Link Margin Threshold | $\ge 3.0$ | Fixed benchmark | dB | Minimum excess power margin required for communication lock. |
| **Algorithm Performance** | Coarse Tracking Error | $\le 2500.0$ | $\le 5000.0$ | $\mu$rad | Gimbal alignment holding beacon well within camera FOV. |
| | Fine Tracking Error | $\le 15.0$ (Nominal $0.02$) | Sub-$\mu$rad achievable | $\mu$rad | Net optical pointing accuracy inside diffraction limit ($12.6\ \mu\text{rad}$). |
| | Frame Processing Time | $\le 20.0$ (Nominal $2.5$) | $\le 33.3$ | ms | Per-frame end-to-end compute latency (50 Hz real-time). |
| | Lock Availability | $\ge 95.0\%$ (Nominal $100\%$) | $0.0$ to $100.0$ | % | Percentage of mission duration with continuous optical lock. |
| | Acquisition Time | $\le 0.50$ (Nominal $0.04$) | $\le 2.0$ | Seconds | Autonomous time to locate, center, and lock track on beacon. |

---

## 4. Evaluation Method and Criteria Table

| Stage | Stage Title | Weightage | Evaluation Methods & Focus Areas | Acceptance Criteria & Benchmarks |
| :---: | :--- | :---: | :--- | :--- |
| **1** | **System Functionality & Software Modularity** | **25%** | Automated unit and integration test suite coverage; clean architectural separation of geometry, kinematics, perception, estimation, and control; deterministic seed control. | 100% test pass rate across all suites (**337 passed tests**); zero unhandled exceptions; PEP 8 and strict typing compliance (`mypy`, `ruff`). |
| **2** | **Target Detection & Tracking Accuracy** | **30%** | Robustness in synthetic video feeds with background clutter and noise; sub-pixel centroiding precision; closed-loop tracking across dynamic 3D target trajectories. | Coarse Tracking Error $\le 2.5$ mrad; Dual-Stage Fine Error $\le 15.0\ \mu\text{rad}$ (nominal $\le 0.05\ \mu\text{rad}$); Lock Retention Rate $\ge 95.0\%$. |
| **3** | **Disturbance Rejection & Operational Robustness** | **20%** | Generation of atmospheric turbulence, scintillation fading, platform sway, motor vibration harmonics; multi-environment stress testing (Space, LEO, UAV, Maritime, Urban). | Maintained tracking lock under vibration harmonics $>30\ \mu\text{rad}$ and $C_n^2 \ge 2.5 \times 10^{-14}\ \text{m}^{-2/3}$; Autonomous reacquisition $\le 0.50$ s. |
| **4** | **Software Deliverables & Usability** | **15%** | Standalone executable CLI (`run`, `benchmark`, `monte-carlo`, `report`, `gui`, `env`, `ao`, `jitter`, `video-benchmark`, `standard-benchmark`); interactive 30 FPS Web GUI dashboard; automated performance logging. | Zero-install Web GUI with canvas viewport and HUD; automated CSV/JSON/HTML report generation; custom virtual environment builder. |
| **5** | **Technical Documentation & Reporting** | **10%** | Comprehensive Technical Report (10–15 pages), User Manual, Performance Log, and Reference Tables detailing architecture, mathematics, benchmarks, and reproduction instructions. | Full IEEE/AIAA structured report; clear equations, diagrams, benchmark tables, and reproduction steps. |

---

## 5. Mandatory Deliverables Mapping

| Required Deliverable | Description & Location in Repository |
| :--- | :--- |
| **1. Software Application** | Standalone executable CLI (`fsoc-pat`) and interactive Web GUI (`fsoc-pat gui` / `python3 run_gui.py`) providing complete tracking and simulation capabilities. |
| **2. Source Code & Tests** | Modular, documented, PEP-8 compliant Python package in [`src/fsoc_pat/`](src/fsoc_pat/) with **337 automated tests** passing in [`tests/`](tests/). |
| **3. Technical Report** | Comprehensive formal IEEE/AIAA Technical Report detailing problem understanding, system architecture, mathematics, benchmarks, and roadmap: [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md). |
| **4. User Manual** | Operational guide covering installation, CLI commands, GUI operations, custom environments, and a 4-minute demonstration video script: [`docs/USER_MANUAL.md`](docs/USER_MANUAL.md). |
| **5. Performance Log** | Auto-generated standardized mission performance logs complying with ISRO/CCSDS specifications: [`docs/PERFORMANCE_LOG.md`](docs/PERFORMANCE_LOG.md) and [`src/fsoc_pat/reporting/`](src/fsoc_pat/reporting/). |
| **6. Reference Tables** | Dedicated reference tables for [Parameters & Specifications](docs/PARAMETERS_AND_SPECIFICATIONS.md) and [Evaluation Criteria](docs/EVALUATION_CRITERIA.md). |
| **7. Advanced Optical Modules** | Closed-Loop Adaptive Optics & Wavefront Reconstruction: [`docs/ADAPTIVE_OPTICS.md`](docs/ADAPTIVE_OPTICS.md); Predictive Micro-Vibration & Latency Feedforward Compensator: [`docs/PREDICTIVE_ANTI_JITTER.md`](docs/PREDICTIVE_ANTI_JITTER.md). |
| **8. System Architecture** | Formal subsystem decomposition, coordinate frame conventions, and layered architecture specification: [`docs/architecture/ARCHITECTURE.md`](docs/architecture/ARCHITECTURE.md). |

---

## 6. Multi-Environment Tracking Benchmark

Closed-loop tracking was validated across 6 diverse operational scenarios using `fsoc-pat env benchmark`:

| Environment | Platform Profile | Coarse Gimbal RMS | Fine Optical FSM RMS | Net Precision | Optical SNR | Lock Availability |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Deep Space ISL** | 550 km Orbit (Vacuum) | $1548.8\ \mu\text{rad}$ | **$0.0200\ \mu\text{rad}$** | **$20.0\ \text{nrad}$** | $116.3\ \text{dB}$ | **100.0%** |
| **LEO Downlink** | Ground Station to LEO | $1552.3\ \mu\text{rad}$ | **$0.0211\ \mu\text{rad}$** | **$21.1\ \text{nrad}$** | $113.2\ \text{dB}$ | **100.0%** |
| **Airborne UAV** | 1,500m Altitude, 0.4° roll | $2295.6\ \mu\text{rad}$ | **$0.0189\ \mu\text{rad}$** | **$18.9\ \text{nrad}$** | $98.9\ \text{dB}$ | **100.0%** |
| **Maritime Vessel** | Ship Mast, 1.2° swell | $2272.4\ \mu\text{rad}$ | **$0.0207\ \mu\text{rad}$** | **$20.7\ \text{nrad}$** | $99.4\ \text{dB}$ | **100.0%** |
| **Urban Rooftop** | 45m Rooftop, HVAC tremor | $1559.4\ \mu\text{rad}$ | **$0.0208\ \mu\text{rad}$** | **$20.8\ \text{nrad}$** | $97.2\ \text{dB}$ | **100.0%** |
| **Custom Balloon** | 24,000m Stratosphere | $1554.6\ \mu\text{rad}$ | **$0.0212\ \mu\text{rad}$** | **$21.2\ \text{nrad}$** | $113.8\ \text{dB}$ | **100.0%** |

*Comparative 6-panel multi-environment analysis plot saved to [`results/demo_environments/multi_environment_comparison.png`](results/demo_environments/multi_environment_comparison.png).*

---

## 7. Quick Start & Execution Guide

### Prerequisites & Installation
OptiLign requires **Python 3.12 or newer** and runs on macOS, Linux, and Windows:

```bash
# Clone the repository
git clone https://github.com/Sanju-2007/OptiLign.git
cd OptiLign

# Create virtual environment and activate
python3 -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate

# Install OptiLign in editable mode with all optional dependencies
pip install -e .

# Run complete automated verification test suite (337 passed)
pytest --tb=short
```

### Demonstration Scripts Walkthrough
OptiLign includes ready-to-run demonstration scripts demonstrating individual subsystems and end-to-end mission scenarios:

```bash
# 1. Master Multi-Environment Benchmark (Exports telemetry CSVs & 6-panel comparison figure)
python3 demo_custom_environments.py

# 2. Complete Closed-Loop PAT Verification (Gaussian beam, HTML report & trajectory plots)
python3 demo_full_system.py

# 3. Launch Interactive Real-Time Web Dashboard (30 FPS canvas + HUD)
python3 run_gui.py   # Access dashboard at http://localhost:8088

# 4. Modular Subsystem Verification Scripts (Phases 1 through 6)
python3 demo_phase1.py   # 3D Trajectory kinematics & gimbal dynamics
python3 demo_phase2.py   # Pinhole projection, sensor noise & centroiding
python3 demo_phase3.py   # Closed-loop tracking & EKF estimation
python3 demo_phase4.py   # Supervisory PAT state machine & reacquisition
python3 demo_phase5.py   # Atmospheric turbulence & disturbance rejection
python3 demo_phase6.py   # FSOC link budget, Gaussian beam & BER margin
```

### Complete CLI Command Suite (`fsoc-pat`)
The standalone `fsoc-pat` executable provides 10 powerful subcommands:

```bash
# 1. Virtual Environment Studio & Presets
fsoc-pat env list                                     # List all presets and custom telemetry channels
fsoc-pat env run --preset space_isl --duration 2.0    # Run simulation in Space ISL preset
fsoc-pat env run --config custom_env.json             # Run user-defined custom JSON environment
fsoc-pat env benchmark --duration 1.0                 # Comparative benchmark across all 6 environments

# 2. Closed-Loop Mission Simulation & Reporting
fsoc-pat run --duration 5.0 --output results/run1 --plot --html
# Generates telemetry.csv, mission_summary.png, mission_report.md, and interactive mission_report.html

# 3. Computer Vision Detector Benchmark
fsoc-pat benchmark --frames 100 --output results/detector_benchmark.json

# 4. Monte Carlo Statistical Analysis
fsoc-pat monte-carlo --runs 20 --duration 3.0 --output results/monte_carlo.json

# 5. Post-Processing Report Generator from Telemetry CSV
fsoc-pat report --input results/telemetry.csv --output results/report --plot

# 6. Interactive Web GUI Dashboard
fsoc-pat gui --port 8088 --host 127.0.0.1

# 7. Closed-Loop Adaptive Optics (AO) & Wavefront Reconstruction
fsoc-pat ao --duration 0.50 --cn2 2.5e-14 --wind-speed 6.0 --output results/ao

# 8. Predictive Zero-Lag Anti-Jitter Compensator
fsoc-pat jitter --duration 1.0 --dt 0.002 --latency 0.015 --output results/jitter

# 9. External MP4 Video Bypass Ingestion & Evaluation
fsoc-pat video-benchmark --generate --duration 5.0 --motion figure_of_8 --output results/video

# 10. Standard Automated Acceptance Benchmark Suite
fsoc-pat standard-benchmark --duration 6.0 --trials 1 --output results/standard_benchmark
```

---

## 8. Repository Structure

```text
OptiLign/
├── configs/                      # YAML scenario configurations
├── data/                         # Test fixtures & trajectory definitions
├── docs/                         # Technical documentation & reports
│   ├── ADAPTIVE_OPTICS.md        # 3-Stage Wavefront correction & AO theory
│   ├── EVALUATION_CRITERIA.md    # Quantitative scoring rubric & benchmarks
│   ├── PARAMETERS_AND_SPECIFICATIONS.md # Full parameter specifications
│   ├── PERFORMANCE_LOG.md        # ISRO/CCSDS standard performance log
│   ├── PREDICTIVE_ANTI_JITTER.md # Zero-phase-lag feedforward compensator
│   ├── TECHNICAL_REPORT.md       # 15-page IEEE/AIAA formal technical report
│   ├── USER_MANUAL.md            # Comprehensive user manual & video script
│   └── architecture/             # Architecture specifications
├── results/                      # Output telemetry, plots, HTML reports & videos
├── src/fsoc_pat/                 # Production source code
│   ├── __main__.py               # Standalone CLI entrypoint (10 subcommands)
│   ├── ai/                       # Neural perception & adaptive models
│   ├── ao/                       # Adaptive optics & deformable mirror
│   ├── benchmark/                # Detector & environment benchmark runners
│   ├── config/                   # Configuration loaders & schema models
│   ├── control/                  # Dual-stage PID/LQR & FSM controllers
│   ├── core/                     # Fundamental domain types & interfaces
│   ├── disturbances/             # Kolmogorov turbulence, vibration & clutter
│   ├── environments/             # Multi-environment engine & JSON builder
│   ├── estimation/               # Adaptive EKF & IMM kinematic filters
│   ├── geometry/                 # Pinhole projection, frames & rotations
│   ├── gui/                      # Real-time Web GUI server & static frontend
│   ├── link/                     # Gaussian beam propagation & link budget
│   ├── pat/                      # Supervisory PAT state machine
│   ├── perception/               # Top-hat background filter & centroiding
│   ├── predictive/               # Micro-vibration spectral analyzer & feedforward
│   ├── reporting/                # HTML, Markdown, CSV report & plot generators
│   ├── simulation/               # Simulation clock, world, targets & gimbal
│   └── telemetry/                # Custom schemas, sinks & telemetry streaming
├── tests/                        # 337 automated unit and integration tests
│   ├── integration/              # End-to-end mission & CLI integration tests
│   └── unit/                     # Unit test suites across all 16 modules
├── demo_custom_environments.py   # Multi-environment benchmark demo
├── demo_full_system.py           # Complete closed-loop PAT verification demo
├── demo_phase1.py ... phase6.py  # Modular subsystem verification scripts
├── run_gui.py                    # Zero-config Web GUI launcher
├── pyproject.toml                # Build system & package dependencies
└── README.md                     # Project overview & reference guide
```

---

## 9. License

OptiLign is released under the **MIT Open-Source License**. See the [LICENSE](LICENSE) file for complete terms.
