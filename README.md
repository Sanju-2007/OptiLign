# OptiLign: AI-Assisted FSOC Virtual Camera PAT Tracking System

[![Tests](https://img.shields.io/badge/tests-243%20passed-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Pointing Accuracy](https://img.shields.io/badge/FSM%20Accuracy-18--22%20nrad-cyan.svg)](docs/TECHNICAL_REPORT.md)
[![Lock Availability](https://img.shields.io/badge/Lock%20Availability-100.0%25-emerald.svg)](results/demo_environments/)

**OptiLign** is a modular software platform for simulating, evaluating, and verifying **Free-Space Optical Communication (FSOC)** Pointing, Acquisition, and Tracking (PAT) algorithms. It implements an autonomous coarse-to-fine camera tracking pipeline with sub-micro-radian (nano-radian) precision across diverse, highly turbulent operational environments.

---

## 1. Background & Operational Context

Free Space Optical Communication (FSOC) offers unprecedented advantages for next-generation mobile networks, including:
- **Gigabit-to-Terabit Data Rates**: Extreme carrier frequencies ($>193\ \text{THz}$) exceeding conventional RF throughput by orders of magnitude.
- **License-Free Spectrum Operation**: Global deployment without international spectrum coordination hurdles.
- **High Immunity to Electromagnetic Interference**: Immune to RF jamming, spoofing, and side-lobe interception.
- **Low SWaP Footprint**: Reduced size, weight, and power suitable for mobile space, airborne, and ground nodes.

However, deploying FSOC links between mobile platforms (satellites, UAVs, ships, high-speed ground terminals) presents a severe challenge of **Pointing, Acquisition, and Tracking (PAT)** of highly narrow laser beams (typically $0.1 - 2.0\ \text{mrad}$). PAT typically happens in two stages: **coarse alignment** and **fine alignment**. 

Coarse alignment is one of the key challenges of PAT, where the transmitting terminal must first locate and maintain the remote terminal within its camera Field-of-View (FOV). Developing and testing such algorithms on real hardware requires expensive cameras, pan-tilt mechanisms, and optical components & equipment. A software-based virtual camera tracking system provides an accessible, high-fidelity platform for algorithm development and validation.

---

## 2. Functional Objective & Expected Solution

### Functional Objective
Develop a software system that autonomously detects, identifies, and continuously tracks a designated moving target within a virtual scene by controlling a virtual camera viewport and gimbal platform.

### Expected Solution
OptiLign provides an AI-assisted camera tracking system capable of automatically detecting and continuously tracking a moving optical beacon in a simulated video stream while controlling a virtual pan-tilt camera:
- **Configurable Virtual Environments**: Multi-environment engine supporting Deep Space ISL, LEO Downlink, Airborne UAV, Maritime Vessel, Urban Rooftop, and user-defined custom platforms.
- **One or More Moving Targets**: Analytical 3D trajectories (circular passes, linear transits, sinusoidal maneuvers) with ground truth position, velocity, and acceleration.
- **Movable Virtual Camera**: 2-DOF pan-tilt camera viewport with physical actuator lag ($\tau=0.015\ \text{s}$), rate limiting, and acceleration limits.
- **Automatic Beacon Detection**: Computer vision perception pipeline utilizing morphological top-hat background subtraction and sub-pixel intensity-weighted centroiding.
- **Continuous Tracking**: Adaptive Extended Kalman Filter (EKF) and Interacting Multiple Model (IMM) filter with dynamic measurement covariance scaling ($R_k$) coupled to atmospheric scintillation.
- **Virtual Camera & FSM Control**: Dual-stage pointing combining a coarse mechanical gimbal with an 800–1000 Hz Fast Steering Mirror (FSM) utilizing DC momentum offloading.
- **Disturbance Modeling**: Injects atmospheric turbulence ($C_n^2$), scintillation fading, platform vibration harmonics (8–180 Hz), aerodynamic wind buffet, sensor read/shot noise, and optical clutter.
- **Real-Time Performance Display**: Interactive 30 FPS Web Dashboard with canvas viewport, HUD gauges, rolling error charts, and auto-generated performance logs.

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
| **1** | **System Functionality & Software Modularity** | **25%** | Automated unit and integration test suite coverage; clean architectural separation of geometry, kinematics, perception, estimation, and control; deterministic seed control. | 100% test pass rate across all suites ($>200$ tests); zero unhandled exceptions; PEP 8 and strict typing compliance. |
| **2** | **Target Detection & Tracking Accuracy** | **30%** | Robustness in synthetic video feeds with background clutter and noise; sub-pixel centroiding precision; closed-loop tracking across dynamic 3D target trajectories. | Coarse Tracking Error $\le 2.5$ mrad; Dual-Stage Fine Error $\le 15.0\ \mu\text{rad}$ (nominal $\le 0.05\ \mu\text{rad}$); Lock Retention Rate $\ge 95.0\%$. |
| **3** | **Disturbance Rejection & Operational Robustness** | **20%** | Generation of atmospheric turbulence, scintillation fading, platform sway, motor vibration harmonics; multi-environment stress testing (Space, LEO, UAV, Maritime, Urban). | Maintained tracking lock under vibration harmonics $>30\ \mu\text{rad}$ and $C_n^2 \ge 2.5 \times 10^{-14}\ \text{m}^{-2/3}$; Autonomous reacquisition $\le 0.50$ s. |
| **4** | **Software Deliverables & Usability** | **15%** | Standalone executable CLI (`run`, `benchmark`, `monte-carlo`, `report`, `gui`, `env`); interactive 30 FPS Web GUI dashboard; automated performance logging. | Zero-install Web GUI with canvas viewport and HUD; automated CSV/JSON/HTML report generation; custom virtual environment builder. |
| **5** | **Technical Documentation & Reporting** | **10%** | Comprehensive Technical Report (10–15 pages) and User Manual detailing architecture, mathematics, benchmarks, and step-by-step reproduction instructions. | Full IEEE/AIAA structured report; clear equations, diagrams, benchmark tables, and reproduction steps. |

---

## 5. Mandatory Deliverables Mapping

| Required Deliverable | Description & Location in Repository |
| :--- | :--- |
| **1. Software Application** | Standalone executable CLI (`fsoc-pat`) and interactive Web GUI (`fsoc-pat gui` / `python3 run_gui.py`) providing full tracking and simulation capabilities. |
| **2. Source Code** | Modular, documented, PEP-8 compliant Python package located in [`src/fsoc_pat/`](src/fsoc_pat/) with 243 automated tests in [`tests/`](tests/). |
| **3. Technical Report** | 10–15 page formal IEEE/AIAA Technical Report detailing problem understanding, system architecture, mathematics, benchmarks, and roadmap: [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md). |
| **4. User Manual** | Comprehensive operational guide covering installation, CLI commands, GUI operations, custom environments, and a 4-minute video demonstration script: [`docs/USER_MANUAL.md`](docs/USER_MANUAL.md). |
| **5. Performance Log** | Auto-generated standardized mission performance logs (CSV, JSON, HTML, Markdown) containing simulation duration, FPS, acquisition time, average and maximum tracking error, lock retention rate, and processing time: [`src/fsoc_pat/reporting/`](src/fsoc_pat/reporting/). |
| **6. Reference Tables** | Dedicated reference tables for [Parameters & Specifications](docs/PARAMETERS_AND_SPECIFICATIONS.md) and [Evaluation Criteria](docs/EVALUATION_CRITERIA.md). |

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

---

## 7. Quick Start & Execution Guide

### Installation
```bash
# Clone the repository
cd OptiLign

# Create virtual environment and install in editable mode
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Run test suite (243 passed)
pytest --tb=short
```

### Run Demonstration Scripts
```bash
# 1. Master Multi-Environment Verification Demo (Generates 6-panel plot)
python3 demo_custom_environments.py

# 2. Complete Closed-Loop PAT Verification (Gaussian beam & HTML report)
python3 demo_full_system.py

# 3. Launch Interactive Web Dashboard
python3 run_gui.py   # Access at http://localhost:8088
```

### CLI Command Summary (`fsoc-pat`)
```bash
# List all built-in environment presets and custom telemetry channels
fsoc-pat env list

# Run simulation in a specific environment preset
fsoc-pat env run --preset space_isl --duration 2.0

# Benchmark tracking accuracy across all environments
fsoc-pat env benchmark --duration 1.0

# Run nominal closed-loop simulation with HTML report and plot
fsoc-pat run --duration 5.0 --output results/run1 --plot --html

# Monte Carlo randomized disturbance study
fsoc-pat monte-carlo --runs 20 --duration 3.0 --output results/mc.json

# Launch interactive Web GUI on custom port
fsoc-pat gui --port 8088
```

---

## 8. License

OptiLign is released under the MIT Open-Source License. See [LICENSE](LICENSE) for details.
