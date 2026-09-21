# OptiLign: Evaluation Method and Criteria

## Overview
The solutions developed for the Free Space Optical Communication (FSOC) Virtual Camera Tracking and Pointing, Acquisition, and Tracking (PAT) challenge are evaluated using a multi-layered, quantitative evaluation method. The criteria assess end-to-end system fidelity, algorithmic precision, mathematical rigor, robustness under simulated environmental disturbances, software modularity, and completeness of deliverables.

---

## Evaluation Method and Criteria Table

| Stage | Stage Title | Weightage | Evaluation Methods & Focus Areas | Acceptance Criteria & Benchmarks | Verification Mechanism |
| :---: | :--- | :---: | :--- | :--- | :--- |
| **1** | **System Functionality & Software Modularity** | **25%** | • Automated unit and integration test suite coverage.<br>• Clean architectural separation of geometry, kinematics, perception, estimation, and control subsystems.<br>• Non-blocking concurrency and real-time execution capability.<br>• Deterministic simulation reproducibility via random seed control. | • **100% test pass rate** across all unit and integration test suites ($>200$ automated tests).<br>• Zero unhandled exceptions or memory leaks.<br>• Strict compliance with PEP 8 and modular interface protocols (`typing.Protocol`). | Executed via `pytest --cov=src` and static analysis (`mypy`, `ruff`). |
| **2** | **Target Detection & Tracking Accuracy** | **30%** | • Detection robustness in synthetic video feeds with background clutter and sensor noise.<br>• Center-of-mass sub-pixel centroiding precision.<br>• Closed-loop tracking accuracy across dynamic 3D target trajectories (circular, linear, sinusoidal).<br>• Pointing error metrics: Mean Radial Error (MRE), RMS jitter, and peak error. | • **Coarse Tracking Error $\le 2.5$ mrad** ($\le 0.14^\circ$), maintaining beacon within camera FOV.<br>• **Dual-Stage Fine Error $\le 15.0\ \mu\text{rad}$** (Empirical: $0.018 - 0.022\ \mu\text{rad}$, well below diffraction limit).<br>• **Lock Retention Rate $\ge 95.0\%$** throughout entire tracking pass. | Automated evaluation in `fsoc-pat run --closed-loop` and integration tests. |
| **3** | **Disturbance Rejection & Operational Robustness** | **20%** | • Disturbance generation fidelity: Kolmogorov atmospheric turbulence, scintillation fading, platform sway, motor vibration harmonics, and wind gusts.<br>• Performance across diverse operational regimes: Deep Space ISL, LEO Downlink, Airborne UAV, Maritime Vessel, and Urban Rooftop.<br>• Outlier rejection via Mahalanobis ellipsoidal gating.<br>• Autonomous reacquisition speed following signal dropout. | • Stable tracking lock maintained under multi-harmonic vibration ($>30\ \mu\text{rad}$) and turbulence ($C_n^2 \ge 2.5 \times 10^{-14}\ \text{m}^{-2/3}$).<br>• **Autonomous Reacquisition Time $\le 0.50$ s** upon target reappearance.<br>• Zero tracking divergence or integrator windup. | Automated stress-test benchmark via `fsoc-pat env benchmark`. |
| **4** | **Software Deliverables & Usability** | **15%** | • Standalone CLI application with rich subcommands (`run`, `benchmark`, `monte-carlo`, `report`, `gui`, `env`).<br>• Interactive real-time Web Dashboard (30 FPS canvas rendering, pan-tilt-zoom controls, telemetry charts, HUD overlays).<br>• Automated generation of performance reports in CSV, JSON, HTML, and Markdown.<br>• Flexible Virtual Environment Builder with custom telemetry logging. | • Fully functional CLI with clean `--help` documentation.<br>• Responsive, zero-install Web GUI accessible via browser (`http://localhost:8088`).<br>• Auto-generated HTML performance logs with complete KPIs, link budgets, and state timelines. | Interactive GUI manual test & automated CLI validation (`python3 -m fsoc_pat --version`). |
| **5** | **Technical Documentation & Reporting** | **10%** | • Comprehensive Technical Report (10–15 pages) detailing problem understanding, system architecture, mathematical formulations, tracking algorithms, and benchmark analysis.<br>• Detailed User Manual covering prerequisites, installation, CLI usage, GUI operations, configuration schemas, and video script. | • Rigorous, IEEE/AIAA-style technical documentation.<br>• Complete equations (Gaussian beams, Hufnagel-Valley, EKF, PID/LQR, FSM).<br>• Clear operational guides with step-by-step reproduction instructions. | Document review of `docs/TECHNICAL_REPORT.md` and `docs/USER_MANUAL.md`. |

---

## Detailed Scoring Rubric (100 Points Total)

### 1. System Functionality & Software Architecture (25 Points)
- **1.1 Modular Pipeline Design (10 pts)**: Clear separation into core packages (`core`, `geometry`, `simulation`, `perception`, `estimation`, `control`, `pat`, `link`, `environments`, `telemetry`, `reporting`, `gui`).
- **1.2 Actuator & Physics Realism (8 pts)**: Realistic gimbal mechanics with first-order inertial lag ($\tau=0.015\text{s}$), slew rate saturation ($60^\circ/\text{s}$ pan, $45^\circ/\text{s}$ tilt), acceleration limits, and deadband.
- **1.3 Automated Test Suite (7 pts)**: Comprehensive automated unit and integration tests verifying all subsystems under nominal and boundary conditions.

### 2. Detection & Tracking Accuracy (30 Points)
- **2.1 Beacon Centroiding & Clutter Rejection (10 pts)**: Morphological background top-hat filtering, intensity thresholding, and sub-pixel intensity-weighted centroiding under high star/urban clutter.
- **2.2 Kinematic State Estimation (10 pts)**: Adaptive Extended Kalman Filter (EKF) and Interacting Multiple Model (IMM) filter with dynamic measurement noise scaling $R_k$ based on scintillation index.
- **2.3 Dual-Stage Pointing Performance (10 pts)**: Coarse gimbal tracking coupled with Fast Steering Mirror (FSM) fine stage achieving sub-micro-radian (nano-radian) residual optical error.

### 3. Environmental Disturbances & Robustness (20 Points)
- **3.1 Realistic Atmospheric & Platform Disturbances (8 pts)**: Implementation of Kolmogorov-Hufnagel-Valley refractive index structure parameter ($C_n^2$), log-normal scintillation, multi-harmonic mechanical vibrations, and aerodynamic wind gusts.
- **3.2 Multi-Environment Simulation Engine (7 pts)**: Built-in validated environments (Space ISL, LEO Downlink, Airborne UAV, Maritime Vessel, Urban Rooftop) and custom environment builder.
- **3.3 Supervisory PAT State Machine & Reacquisition (5 pts)**: Autonomous transitions between Standby, Spiral Search, Centering Acquisition, Fine Tracking, and Reacquisition upon cloud/building fade.

### 4. Usability, Deliverables, & Performance Logging (15 Points)
- **4.1 Interactive Web GUI Dashboard (6 pts)**: High-performance browser interface with simulated camera viewport, target crosshairs, 3D pan/tilt orientation, PTZ controls, and live telemetry graphs.
- **4.2 CLI & Standalone Application (5 pts)**: Command-line interface supporting batch simulations, Monte Carlo analyses, and preset benchmarks.
- **4.3 Automated Performance Logging (4 pts)**: Automated generation of standardized performance reports containing duration, FPS, acquisition time, average/max tracking error, lock retention rate, and processing time.

### 5. Technical Documentation (10 Points)
- **5.1 Technical Report Quality (6 pts)**: 10–15 page formal report detailing mathematics, system architecture, empirical benchmark analysis, and future improvements.
- **5.2 User Manual & Demonstration Guide (4 pts)**: Clear installation guide, operation workflows, configuration schemas, and 3–5 minute video presentation script.
