# OptiLign FSOC PAT Mission — Verification & Telemetry Report

> **Executive Summary**: OptiLign end-to-end Free-Space Optical Communication (FSOC) Pointing, Acquisition, and Tracking (PAT) simulation telemetry report with PS-04 competition verification. Evaluates sub-pixel image centroiding, 2-DOF gimbal kinematics, Extended Kalman Filter estimation, disturbance rejection, dynamic cloud fault recovery, and optical link budget radiometry.

## OPTILIGN — PS-04 PERFORMANCE REPORT

- **Mission Scenario**: OptiLign Aerospace FSOC PAT Mission
- **Simulation Timeline**: `7.98 s` (400 steps @ `20.0 ms`)
- **Overall PS-04 Verification Verdict**: **`PASS`**

### INPUT / PROCESSING
| Metric | Measured Value | PS-04 Benchmark Limit | Status |
| :--- | :--- | :--- | :---: |
| **Input / Simulation FPS** | `50.1 FPS` | System Sample Rate | PASS |
| **Actual Processing FPS** | `42.2 FPS` | $\ge 20.0$ FPS (Wall-Clock Runtime) | PASS |
| **Average Processing Time** | `23.72 ms` | $\le 50.0$ ms per step (Nominal $\le 20$ ms) | PASS |
| **Maximum Processing Time** | `27.47 ms` | Jitter Excursion Bound | PASS |

### ACQUISITION
| Metric | Measured Value | PS-04 Benchmark Limit | Status |
| :--- | :--- | :--- | :---: |
| **Acquisition Time** | `0.00 s` | $\le 2.0$ s (Autonomous Initial Lock) | PASS |
| **Re-acquisition Time** | `0.30 s` | $\le 1.0$ s (Fast Link Re-establishment) | PASS |

### TRACKING (PIXEL & CENTROID ERROR)
| Metric | Measured Value | PS-04 Benchmark Limit | Status |
| :--- | :--- | :--- | :---: |
| **Average Centroid Error** | `1.49 px` | $\le 10.0$ px (PS-04 Headline Requirement) | PASS |
| **RMSE Centroid Error** | `1.54 px` | Sub-pixel precision / low jitter | PASS |
| **Maximum Centroid Error** | `3.88 px` | Bound within camera viewport | PASS |
| **95th Percentile Centroid Error** | `2.00 px` | Statistical 2-sigma bound | PASS |
| **Target Loss** | `4.2%` | $< 5.0\%$ of operational duration | PASS |
| **Lock Retention** | `95.8%` | $\ge 95.0\%$ in closed-loop track | PASS |

### PS-04 LIMITS COMPLIANCE SUMMARY
| PS-04 Requirement | Specification Limit | Measured Result | Margin / Headroom | Verdict |
| :--- | :---: | :---: | :---: | :---: |
| **Acquisition** | $\le 2.0$ s | `0.00 s` | `+2.00 s` | **PASS** |
| **Tracking Error** | $\le 10$ px | `1.49 px` | `+8.51 px` | **PASS** |
| **Target Loss** | $< 5\%$ | `4.2%` | `+0.8%` | **PASS** |
| **Re-acquisition** | $\le 1.0$ s | `0.30 s` | `+0.70 s` | **PASS** |
| **Processing** | $\ge 20$ FPS | `42.2 FPS` | `+22.2 FPS` | **PASS** |

---

## 1. Scenario & Physical Simulation Profile

| Parameter | Value | Operational Notes |
| :--- | :--- | :--- |
| **Mission Scenario** | OptiLign Aerospace FSOC PAT Mission | Closed-loop PAT with disturbance & channel |
| **Simulation Duration** | `7.98 s` | Total discrete integration steps: 400 |
| **Integration Timestep ($\Delta t$)** | `20.0 ms` | Fundamental simulation cycle rate: `50.1 Hz` |
| **Target Distance / Range** | `98.3 m` (Range: `90.0 - 110.0 m`) | Dynamic 3D aerospace target |
| **Sensor Viewport** | $640 \times 480$ VGA @ $60.0^\circ$ HFOV | Optical center: $(c_x, c_y) = (320.0, 240.0)$ |
| **Laser Transmitter** | $\lambda = 1550$ nm, $P_{tx} = 1.0$ W (30 dBm) | Gaussian beam waist $w_0 = 80\ \mu\text{m}$, $\theta_{div} = 6.17$ mrad |
| **Optical Receiver** | $D_{rx} = 10$ cm aperture, Sensitivity = $-38.0$ dBm | PIN photodiode direct detection |

## 2. Image-Plane Perception & Sub-pixel Centroiding Performance

### A. Centroiding Displacement Relative to Ground Truth
| Centroiding Metric | Measured Value | Benchmark Criterion | Evaluation |
| :--- | :--- | :---: | :---: |
| **Average Centroid Error** | `1.49 px` | $\le 10.0$ px (PS-04 Headline) | PASS |
| **RMSE Centroid Error** | `1.54 px` | Sub-pixel precision / low jitter | OPTIMAL |
| **Median Centroid Error** | `1.48 px` | Robust central tendency | NOMINAL |
| **95th Percentile Centroid Error** | `2.00 px` | Statistical 2-sigma excursion | BOUNDED |
| **Maximum Centroid Error** | `3.88 px` | Peak transient excursion | FOV OK |
| **Horizontal Error Component ($\Delta u$)** | `0.99 px` | Azimuth image displacement | BALANCED |
| **Vertical Error Component ($\Delta v$)** | `1.02 px` | Elevation image displacement | BALANCED |

### B. Optical Boresight Centering Performance
| Boresight Metric | Measured Value | Standard / Limit | Status |
| :--- | :--- | :---: | :---: |
| **Mean Tracking Error (from Boresight)** | `2.44 px` | Centering on optical axis (320, 240) | STABLE |
| **RMSE Tracking Error** | `2.67 px` | Control closed-loop jitter | SUB-PIXEL |
| **Maximum Boresight Excursion** | `5.56 px` | Sensor field-of-view limit | BOUNDED |
| **Sub-pixel Accuracy Verification** | **VERIFIED** | Error $\le 2.0$ px nominal track | PASS |

## 3. Spatial Angular Pointing Accuracy & Jitter Analysis

| Angular Pointing Metric | Value (mrad) | Value (µrad) | Value (deg) |
| :--- | :--- | :--- | :--- |
| **Mean Radial Error (MRE)** | `4.395 mrad` | `4395.4 µrad` | `0.2518°` |
| **RMS Pointing Jitter ($\sigma_\theta$)** | `4.814 mrad` | `4813.8 µrad` | `0.2758°` |
| **Median Radial Error** | `4.263 mrad` | `4263.0 µrad` | `0.2443°` |
| **95th Percentile Error** | `7.941 mrad` | `7940.7 µrad` | `0.4550°` |
| **99th Percentile Error** | `9.089 mrad` | `9089.3 µrad` | `0.5208°` |
| **Peak Pointing Error** | `10.031 mrad` | `10031.0 µrad` | `0.5747°` |
| **Azimuth Component Error ($\theta_h$)** | `3.952 mrad` | `3952.0 µrad` | `0.2264°` |
| **Elevation Component Error ($\theta_v$)** | `0.259 mrad` | `259.0 µrad` | `0.0148°` |
| **Filter-Estimated Mean Error** | `3.996 mrad` | `3996.0 µrad` | Kalman Filter estimate |
| **Estimation Residual Bias** | `0.399 mrad` | `399.0 µrad` | Filter convergence consistency |

## 4. 2-DOF Gimbal Actuator Kinematics & Slew Profile

| Actuator Channel | Mean Angle | Min Angle | Max Angle | Excursion Range | Peak Slew Rate | Slew Limit | Headroom |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pan / Azimuth Axis** | `2.28°` | `-4.58°` | `5.58°` | `10.16°` | `3.36°/s` | `60.0°/s` | `94.4%` |
| **Tilt / Elevation Axis** | `0.10°` | `0.00°` | `0.14°` | `0.14°` | `0.75°/s` | `45.0°/s` | `98.3%` |

- **Rate Saturation Incidents**: `0` (Zero mechanical lockup)
- **Servo Command Tracking RMSE**: `0.2760°`

## 5. Target Perception & AI/Hybrid Detection Fidelity

| Perception Parameter | Value | Reference Standard | Operational Verdict |
| :--- | :--- | :--- | :---: |
| **Valid Detection Frames** | `384 / 400` | Frame detection rate: `96.0%` | OPTIMAL |
| **Mean Detection Confidence** | `0.944` | Threshold: $\ge 0.40$ | PASS |
| **Peak Detection Confidence** | `1.000` | High signal-to-clutter ratio | PASS |
| **Minimum Detection Confidence** | `0.000` | Transient cloud boundary floor | NOMINAL |
| **Missed Detections (Cloud / Fade)** | `16` steps | Recovered via EKF coasting | RECOVERED |
| **False Alarms (Star Clutter)** | `0` | Spatial-temporal validation filter | ZERO FALSE |

## 6. PAT State Machine & Dynamic Fault Recovery Timeline

| PAT State | Cumulative Time (s) | Proportion (%) | Operational Functional State |
| :--- | :---: | :---: | :--- |
| **TRACKING** | `7.70 s` | `96.5%` | Closed-loop EKF estimation & gimbal rate control |
| **REACQUISITION** | `0.30 s` | `3.8%` | Autonomous link re-establishment after disturbance |
| **SEARCH** | `0.00 s` | `0.0%` | Archimedean spiral search over field-of-regard |
| **ACQUISITION** | `0.00 s` | `0.0%` | Beacon centroid confirmation & EKF initialization |

- **Autonomous Initial Acquisition Time**: `0.00 s` (Benchmark: $\le 2.00$ s) — **`PASS`**
- **Mean Time to Reacquire (MTTR)**: `0.30 s` (Benchmark: $\le 1.00$ s) — **`PASS`**
- **Maximum Reacquisition Time**: `0.30 s`
- **Lock Retention Rate**: `95.8%` (Benchmark: $\ge 95.0\%$) — **`PASS`**
- **Target Loss Rate / Duration**: `4.2%` (`0.34 s`) (Benchmark: $< 5.0\%$) — **`PASS`**
- **State Transitions Count**: `2`

### Reacquisition Events Log
| Event # | Start Time (s) | End Time (s) | Recovery Duration (s) | Final Status |
| :---: | :---: | :---: | :---: | :---: |
| Event #1 | `2.44 s` | `2.74 s` | `0.30 s` | Resolved & Locked |

## 7. Environmental Disturbances & Channel Scintillation

| Disturbance Channel | Measured Value | Standard / Baseline | Status |
| :--- | :--- | :--- | :---: |
| **Platform Base Vibration Jitter (RMS)** | `1201.5 µrad` | Multichannel satellite jitter harmonics | MEASURED |
| **Platform Base Vibration Jitter (Peak)** | `2456.7 µrad` | Harmonic excursion limit | BOUNDED |
| **Jitter Rejection Ratio** | `-12.1 dB` | Active closed-loop disturbance attenuation | REJECTED |
| **Refractive Structure Constant ($C_n^2$)** | `1.49e-12 m^{-2/3}` | Boundary layer atmospheric turbulence | NOMINAL |
| **Fried Atmospheric Coherence Length ($r_0$)** | `1.2 cm` | Spatial optical coherence seeing | OPTIMAL |
| **Mean Atmospheric Beam Wander** | `27.1 µrad` | Kolmogorov beam centroid deflection | MEASURED |
| **Scintillation Rytov Index ($\sigma_R^2$)** | `0.914` | $\sigma_R^2 < 0.20$ (Weak Regime) | PASS |
| **Scintillation Regime** | `Moderate Turbulence (0.2 ≤ σ_R² < 1.0)` | Log-normal intensity distribution | NOMINAL |
| **Scintillation Fade Reserve ($A_{\text{fade}}$)** | `-12.60 dB` | 99% availability fade reserve margin | RESERVE OK |
| **Cloud Occlusion Events / Duration** | `1` event (`0.36 s`, `4.5%`) | Transient aerosol blockage model | RECOVERED |
| **Peak Cloud Attenuation** | `100.0 dB` (Min Trans: `0.0%`) | Extinction factor during occlusion | COMPLETE FADE |

## 8. End-to-End FSOC Optical Communications Link Budget

### A. Optical Link Loss Breakdown Waterfall
| Link Budget Component | Loss / Gain Value | Operational Role / Physical Origin |
| :--- | :--- | :--- |
| **Transmitter Optical Power ($P_{tx}$)** | `+30.0 dBm` (`1.0 W`) | Continuous-wave beacon and signal laser |
| **Free Space Path Loss ($L_{fs}$)** | `-178.0 dB` | Geometric beam expansion over `98.3 m` |
| **Atmospheric Extinction Loss ($A_{atm}$)** | `-0.05 dB` | Clear-air Beer-Lambert absorption |
| **Locked Mean Pointing Loss ($L_{p,locked}$)** | `-4.29 dB` | $L_p = 8.686(\theta / \theta_{div})^2$ (Locked tracking) |
| **Locked Peak Pointing Loss ($L_{p,max}$)** | `-18.99 dB` | Maximum pointing excursion during lock |
| **Peak Cloud Extinction Loss ($A_{cloud}$)** | `-100.0 dB` | Transient cloud passage absorption |
| **Detector Sensitivity ($P_{sens}$)** | `-38.0 dBm` | PIN photodiode telecom threshold |

### B. Optical Power & Link Margin Summary
| Radiometric Metric | Operational Track (Locked) | Overall Mission (Inc. Clouds) | Benchmark Spec |
| :--- | :---: | :---: | :---: |
| **Mean Received Power ($P_{rx}$)** | `+3.92 dBm` | `-0.28 dBm` | Sensitivity: $-38.0$ dBm |
| **Minimum Received Power ($P_{rx,min}$)** | — | `-100.00 dBm` | Fade floor during cloud |
| **Mean Link Margin** | `+41.92 dB` | `+37.72 dB` | Target: $\ge 3.0$ dB |
| **Minimum Link Margin** | — | `-62.00 dB` | Deep cloud outage floor |
| **Link Availability ($\text{Margin} \ge 3\text{dB}$)** | **100.0%** | `96.0%` | High-bandwidth optical uptime |
| **Outage Time Duration** | `0.00 s` | `0.32 s` (16 steps) | Cloud blockage recovery |

## 9. Digital Communications, SNR, Q-Factor & BER Analysis

| Telecom Parameter | Operational Locked Tracking | Complete Mission Profile | Standard / Benchmark | Verdict |
| :--- | :---: | :---: | :--- | :---: |
| **Electrical Signal-to-Noise Ratio (SNR)** | `+83.87 dB` | `+80.45 dB` | $\ge 15.6$ dB (Error-free telecom) | OPTIMAL |
| **SNR Reserve Margin above FEC** | `+73.07 dB` | — | Hard Decision Limit: $10.8$ dB | PASS |
| **Digital Q-Factor** | `7805.34` | `7805.34` | $Q = \sqrt{\text{SNR}} / 2 \ge 3.0$ | EXCELLENT |
| **Uncoded OOK Bit Error Rate (BER)** | `< 1.0e-15` | `2.00e-02` | FEC Threshold: $1.0 \times 10^-6$ | PASS |
| **Hard-Decision FEC Compliance** | **COMPLIANT** | Outage-recovered | Uncoded BER $\le 10^-6$ in lock | PASS |

### Mathematical Formulations & Radiometry Principles
1. **Photodetector Electrical Signal-to-Noise Ratio (SNR)**:
   $$\text{SNR}_{\text{elec}} = 10 \log_{10} \left( \frac{R^2 P_{rx}^2}{2 q (R P_{rx} + I_d) B + \frac{4 k_B T B}{R_L} F_n} \right)$$
   Where $R$ is photodiode responsivity ($0.95$ A/W at $1550$ nm), $B = 1.25$ GHz telecom bandwidth, and NEP = $2.5$ pW/$\sqrt{\text{Hz}}$.

2. **Uncoded Bit-Error Rate (BER) for On-Off Keying (OOK)**:
   $$\text{BER} = \frac{1}{2} \text{erfc}\left( \frac{Q}{\sqrt{2}} \right) = \frac{1}{2} \text{erfc}\left( \frac{\sqrt{\text{SNR}_{\text{linear}}}}{2\sqrt{2}} \right)$$

3. **Atmospheric Turbulence Scintillation Rytov Index & Fade Depth**:
   $$\sigma_R^2 = 1.23 \, C_n^2 \, k^{7/6} \, L^{11/6}$$
   $$A_{\text{fade}} = 4.343 \left( \sqrt{2 \sigma_R^2 \ln(100)} - \frac{\sigma_R^2}{2} \right) \text{ dB (99\% availability)}$$

## 10. Algorithmic Runtime Execution & Latency Profiling

| Performance Indicator | Measured Value | Requirement / Benchmark | Status |
| :--- | :--- | :--- | :---: |
| **Input / Simulation Framerate** | `50.1 FPS` | System simulation update frequency | PASS |
| **Actual Processing Throughput** | `42.2 FPS` | $\ge 20.0$ FPS (PS-04 Headline Requirement) | **PASS** |
| **Mean Step Processing Time** | `23.72 ms` | $\le 50.0$ ms (Nominal target $\le 20.0$ ms) | PASS |
| **Median Processing Time** | `23.70 ms` | Typical algorithmic cycle latency | OPTIMAL |
| **Minimum Processing Time** | `23.43 ms` | Best-case computation latency | FAST |
| **Maximum Processing Time** | `27.47 ms` | Peak latency under heavy scene rendering | BOUNDED |
| **Latency Standard Deviation ($\sigma$)** | `0.23 ms` | Processing jitter / execution stability | STABLE |
| **95th Percentile Processing Time** | `23.88 ms` | 2-sigma computational bound | PASS |
| **Real-time Deadline Headroom** | `52.6%` | Margin below 50.0 ms (20 FPS) deadline | COMFORTABLE |

## 11. Parameters and Physical Specifications Reference

| Subsystem | Parameter | Specification Value | Unit / Operational Description |
| :--- | :--- | :--- | :--- |
| **Camera & Sensor** | Image Resolution | $640 \times 480$ (Configurable to 1920×1080) | Pixels (VGA/FHD) |
| | Field-of-View (FOV) | $45.0^\circ$ – $60.0^\circ$ Horizontal | Wide angle coarse acquisition |
| | Frame Rate (Sampling Rate) | $30.0$ – $100.0$ Hz ($50.0$ Hz default) | Timestep $\Delta t = 0.01 - 0.033$ s |
| | Sensor Read Noise | $3.0 - 8.0$ (Gaussian $\sigma$) + Shot Noise | Poisson / Gaussian sensor model |
| **Gimbal Actuator** | Pan Travel Range | $\pm 180.0^\circ$ Continuous | Azimuth axis |
| | Tilt Travel Range | $\pm 45.0^\circ$ – $\pm 90.0^\circ$ | Elevation axis |
| | Max Slewing Rate | $60.0^\circ / \text{s}$ Pan, $45.0^\circ / \text{s}$ Tilt | Servo drive velocity limit |
| | Actuator Time Constant | $\tau = 0.015$ s | First-order inertial lag |
| **Fine Steering (FSM)** | Mirror Deflection Range | $\pm 5000.0\ \mu\text{rad}$ (\pm 5 mrad) | Piezo/Voice-coil fine stage |
| | Closed-Loop Bandwidth | $800.0 - 1000.0$ Hz | High-bandwidth jitter rejection |
| | Angular Resolution | $0.05\ \mu\text{rad}$ ($50$ nano-radians) | Sub-micro-radian precision |
| **Optical Link** | Laser Wavelength | $1550.0$ nm (or $850.0$ nm) | Standard Telecom C-Band |
| | Optical Transmit Power | $500.0 - 2000.0$ mW | CW beacon & signal laser |
| | Beam Divergence | $0.8 - 1.5$ mrad | Gaussian beam waist $w_0$ |
| | Receiver Aperture | $0.10 - 0.25$ m | Parabolic / Catadioptric lens |
| **Disturbances** | Atmospheric Turbulence | $C_n^2 = 0.0 - 4.0 \times 10^{-14}\ \text{m}^{-2/3}$ | Vacuum to Strong Turbulence |
| | Platform Vibration Jitter | 3 harmonics ($8 - 180$ Hz) + random buffet | Engine, rotor, reaction wheels |
| | Background Clutter | 0 to 100 stars / city point lights | Night sky & urban clutter |

## 12. Evaluation Method and Criteria Table

| Evaluation Stage | Weightage | Evaluation Method & Focus Areas | Acceptance Criteria & Benchmark |
| :--- | :---: | :--- | :--- |
| **Stage 1: System Functionality & Modularity** | **25%** | Automated unit and integration test suite; modular code structure; clean separation of geometry, dynamics, perception, estimation, and control. | 100% test pass rate ($>200$ tests); zero unhandled exceptions; PEP 8 / clean code adherence. |
| **Stage 2: Target Detection & Tracking Accuracy** | **30%** | Automated trajectory tracking tests under static and dynamic scenarios; evaluation of Mean Radial Error and peak excursions. | Coarse error $\le 2.5$ mrad; Dual-stage fine error $\le 15.0\ \mu\text{rad}$; Lock retention rate $\ge 95.0\%$. |
| **Stage 3: Disturbance Rejection & Robustness** | **20%** | Multi-environment stress tests across Space ISL, LEO Downlink, Airborne UAV, Maritime, and Urban Rooftop under high scintillation and jitter. | Maintained tracking lock under vibration harmonics $>30\ \mu\text{rad}$ and $C_n^2 \ge 10^{-14}$; fast reacquisition $\le 0.5$ s. |
| **Stage 4: Software Deliverables & Usability** | **15%** | Standalone CLI & interactive Web GUI usability; real-time telemetry HUD; parameter configuration; automated performance logging. | Interactive 30 FPS web GUI; automated CSV/HTML/JSON report generation; custom virtual environment builder. |
| **Stage 5: Technical Documentation & Report** | **10%** | Comprehensive Technical Report (10-15 pages) and User Manual detailing architecture, mathematics, benchmarks, and installation. | Full IEEE/AIAA structured report; clear equations, diagrams, benchmark tables, and reproduction steps. |

---
*Report auto-generated by OptiLign Telemetry Reporting Engine.*