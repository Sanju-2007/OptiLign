# OptiLign: Parameters and Specifications

## Functional Objective
Develop a software system that autonomously detects, identifies, and continuously tracks a designated moving optical beacon within a virtual scene by controlling a virtual camera viewport and gimbal platform across diverse operational environments.

---

## Complete Parameters and Specifications Table

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
