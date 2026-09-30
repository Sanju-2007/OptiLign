# OptiLign — Standard Specification Acceptance Report
**Timestamp:** 2026-09-29 16:42:18  
**Overall Verdict:** PASS (100.0% Compliant)  
**Total Scenarios Evaluated:** 36 (36 PASS, 0 FAIL)

## Formal Verification Criteria
| Specification Requirement | Reference Limit | Measured (Mean / Max) | Status |
|---|---|---|:---:|
| **Initial Acquisition Time** | $\le 2.0$ s | 0.0 s / 0.03 s | **PASS** |
| **Image-Plane Tracking Error** | $\le 10.0$ px | 3.14 px / 7.36 px | **PASS** |
| **Target Loss Percentage** | $< 5.0$ % | 0.0 % / 0.0 % | **PASS** |
| **Re-acquisition Time** | $\le 1.0$ s | $\le 1.0$ s | **PASS** |
| **Real-Time Processing Speed** | $\ge 20.0$ FPS | 172.5 FPS | **PASS** |

## Error Metric Separation
- **Centroiding Error (px):** Measures perception and beacon centroid extraction accuracy relative to ground truth projected on sensor.
  - Mean: `0.56 px`, Max: `0.75 px`, 95th Percentile: `0.73 px`
- **Tracking Error (px):** Measures closed-loop gimbal centering accuracy relative to optical boresight $(320, 240)$.
  - Mean: `3.14 px`, RMSE: `2.15 px`, Max: `7.36 px`
- **Angular Pointing Error ($\mu$rad):** Measures physical line-of-sight spatial alignment.

## Complete Scenario Matrix
| Scenario | Trajectory | Disturbance | Centroid (px) | Tracking (px) | Acq (s) | Loss (%) | FPS | Status |
|---|---|---|---|---|---|---|---|:---:|
| `straight_nominal` | straight | nominal | 0.73 | 2.67 | N/A | 0.0% | 194.9 | **PASS** |
| `straight_noise_salt_pepper` | straight | noise_salt_pepper | 0.69 | 2.70 | N/A | 0.0% | 150.0 | **PASS** |
| `straight_noise_gaussian` | straight | noise_gaussian | 0.75 | 2.68 | N/A | 0.0% | 169.8 | **PASS** |
| `straight_noise_poisson` | straight | noise_poisson | 0.72 | 2.67 | N/A | 0.0% | 40.0 | **PASS** |
| `straight_jitter_optical` | straight | jitter_optical | 0.56 | 7.36 | N/A | 0.0% | 222.7 | **PASS** |
| `straight_platform_motion` | straight | platform_motion | 0.49 | 5.95 | N/A | 0.0% | 222.8 | **PASS** |
| `straight_atmosphere_haze` | straight | atmosphere_haze | 0.60 | 2.71 | N/A | 0.0% | 166.9 | **PASS** |
| `straight_atmosphere_fog` | straight | atmosphere_fog | 0.75 | 2.66 | N/A | 0.0% | 230.2 | **PASS** |
| `straight_worst_case_combined` | straight | worst_case_combined | 0.61 | 6.85 | 0.03s | 0.0% | 123.2 | **PASS** |
| `circular_nominal` | circular | nominal | 0.47 | 1.55 | N/A | 0.0% | 230.8 | **PASS** |
| `circular_noise_salt_pepper` | circular | noise_salt_pepper | 0.46 | 1.57 | N/A | 0.0% | 153.9 | **PASS** |
| `circular_noise_gaussian` | circular | noise_gaussian | 0.49 | 1.56 | N/A | 0.0% | 166.0 | **PASS** |
| `circular_noise_poisson` | circular | noise_poisson | 0.48 | 1.54 | N/A | 0.0% | 40.0 | **PASS** |
| `circular_jitter_optical` | circular | jitter_optical | 0.55 | 7.25 | N/A | 0.0% | 222.6 | **PASS** |
| `circular_platform_motion` | circular | platform_motion | 0.59 | 5.73 | N/A | 0.0% | 223.2 | **PASS** |
| `circular_atmosphere_haze` | circular | atmosphere_haze | 0.57 | 1.63 | N/A | 0.0% | 170.6 | **PASS** |
| `circular_atmosphere_fog` | circular | atmosphere_fog | 0.44 | 1.56 | N/A | 0.0% | 230.3 | **PASS** |
| `circular_worst_case_combined` | circular | worst_case_combined | 0.59 | 6.53 | 0.03s | 0.0% | 123.6 | **PASS** |
| `figure_8_nominal` | figure_8 | nominal | 0.54 | 0.96 | N/A | 0.0% | 230.7 | **PASS** |
| `figure_8_noise_salt_pepper` | figure_8 | noise_salt_pepper | 0.50 | 1.05 | N/A | 0.0% | 154.4 | **PASS** |
| `figure_8_noise_gaussian` | figure_8 | noise_gaussian | 0.54 | 1.01 | N/A | 0.0% | 170.4 | **PASS** |
| `figure_8_noise_poisson` | figure_8 | noise_poisson | 0.54 | 0.99 | N/A | 0.0% | 40.7 | **PASS** |
| `figure_8_jitter_optical` | figure_8 | jitter_optical | 0.54 | 7.21 | N/A | 0.0% | 222.8 | **PASS** |
| `figure_8_platform_motion` | figure_8 | platform_motion | 0.60 | 5.71 | N/A | 0.0% | 223.9 | **PASS** |
| `figure_8_atmosphere_haze` | figure_8 | atmosphere_haze | 0.56 | 1.11 | N/A | 0.0% | 169.8 | **PASS** |
| `figure_8_atmosphere_fog` | figure_8 | atmosphere_fog | 0.46 | 0.95 | N/A | 0.0% | 230.6 | **PASS** |
| `figure_8_worst_case_combined` | figure_8 | worst_case_combined | 0.59 | 6.43 | 0.03s | 0.0% | 123.6 | **PASS** |
| `random_nominal` | random | nominal | 0.61 | 0.50 | N/A | 0.0% | 227.7 | **PASS** |
| `random_noise_salt_pepper` | random | noise_salt_pepper | 0.46 | 0.62 | N/A | 0.0% | 152.5 | **PASS** |
| `random_noise_gaussian` | random | noise_gaussian | 0.63 | 0.52 | N/A | 0.0% | 169.9 | **PASS** |
| `random_noise_poisson` | random | noise_poisson | 0.60 | 0.50 | N/A | 0.0% | 40.8 | **PASS** |
| `random_jitter_optical` | random | jitter_optical | 0.59 | 7.04 | N/A | 0.0% | 212.3 | **PASS** |
| `random_platform_motion` | random | platform_motion | 0.63 | 5.70 | N/A | 0.0% | 224.5 | **PASS** |
| `random_atmosphere_haze` | random | atmosphere_haze | 0.50 | 0.77 | N/A | 0.0% | 172.2 | **PASS** |
| `random_atmosphere_fog` | random | atmosphere_fog | 0.32 | 0.37 | N/A | 0.0% | 235.0 | **PASS** |
| `random_worst_case_combined` | random | worst_case_combined | 0.56 | 6.53 | 0.03s | 0.0% | 125.4 | **PASS** |
