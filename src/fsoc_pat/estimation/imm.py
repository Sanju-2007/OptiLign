"""Interacting Multiple Model (IMM) Estimator for Maneuvering Target Tracking.

Maintains multiple dynamic models running in parallel:
  - Model 1: Low-noise Constant Velocity (CV) for steady cruising
  - Model 2: Constant Acceleration (CA) for orbital/slewing motion
  - Model 3: High-noise Maneuver model for sudden dynamic shifts

Computes real-time mode probabilities based on Gaussian measurement
likelihoods and combines state estimates for optimal tracking accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from fsoc_pat.estimation.adaptive_ekf import AdaptiveEKF, AdaptiveEKFConfig
from fsoc_pat.perception.detection_types import Detection


@dataclass
class IMMConfig:
    """Configuration for IMM filter."""

    # Markov mode transition probability matrix (3x3)
    transition_matrix: NDArray[np.float64] = field(default_factory=lambda: np.array([
        [0.90, 0.08, 0.02],
        [0.05, 0.90, 0.05],
        [0.05, 0.15, 0.80],
    ], dtype=np.float64))
    initial_probabilities: list[float] = field(default_factory=lambda: [0.70, 0.25, 0.05])


class IMMFilter:
    """Interacting Multiple Model filter for seamless tracking during maneuvers."""

    def __init__(self, config: IMMConfig | None = None) -> None:
        self.config = config or IMMConfig()

        # Instantiate 3 parallel sub-filters with different process noise dynamics
        # Model 0: Low noise CV/CA
        self.filters = [
            AdaptiveEKF(AdaptiveEKFConfig(q_accel_std=50.0, base_r_std=1.2, gate_threshold=25.0)),
            # Model 1: Medium noise CA
            AdaptiveEKF(AdaptiveEKFConfig(q_accel_std=300.0, base_r_std=1.5, gate_threshold=25.0)),
            # Model 2: High agility maneuver model
            AdaptiveEKF(AdaptiveEKFConfig(q_accel_std=1000.0, base_r_std=2.0, gate_threshold=35.0)),
        ]

        self.mu = np.array(self.config.initial_probabilities, dtype=np.float64)
        self.mu /= np.sum(self.mu)
        self.num_models = len(self.filters)
        self.combined_state = np.zeros(6, dtype=np.float64)

    def reset(self) -> None:
        """Reset all sub-filters and mode probabilities."""
        for f in self.filters:
            f.reset()
        self.mu = np.array(self.config.initial_probabilities, dtype=np.float64)
        self.mu /= np.sum(self.mu)
        self.combined_state.fill(0.0)

    def predict(self, dt: float) -> tuple[float, float]:
        """Perform IMM model mixing and propagate all sub-filters."""
        Pi = self.config.transition_matrix

        # 1. Interaction / Mixing
        # c_bar_j = sum_i(Pi_ij * mu_i)
        c_bar = np.dot(self.mu, Pi)

        # Mixing probabilities mu_{i|j} = (Pi_ij * mu_i) / c_bar_j
        mixed_states = []
        mixed_covs = []

        for j in range(self.num_models):
            if c_bar[j] > 1e-12:
                mu_ij = (Pi[:, j] * self.mu) / c_bar[j]
            else:
                mu_ij = np.ones(self.num_models) / self.num_models

            # Mixed state for filter j
            x_mix = np.zeros(6, dtype=np.float64)
            for i in range(self.num_models):
                x_mix += mu_ij[i] * self.filters[i].state
            mixed_states.append(x_mix)

            # Mixed covariance
            P_mix = np.zeros((6, 6), dtype=np.float64)
            for i in range(self.num_models):
                dx = (self.filters[i].state - x_mix).reshape(-1, 1)
                P_mix += mu_ij[i] * (self.filters[i].cov + dx @ dx.T)
            mixed_covs.append(P_mix)

        # Update sub-filter states with mixed estimates
        for j in range(self.num_models):
            self.filters[j].state = mixed_states[j]
            self.filters[j].cov = mixed_covs[j]

        # 2. Model-conditional Predictions
        preds = [f.predict(dt) for f in self.filters]

        # Combine predictions weighted by predicted mode probabilities c_bar
        u_pred = sum(c_bar[j] * preds[j][0] for j in range(self.num_models))
        v_pred = sum(c_bar[j] * preds[j][1] for j in range(self.num_models))

        return float(u_pred), float(v_pred)

    def update(
        self,
        detection: Detection | None,
        scintillation_index: float = 0.0,
    ) -> tuple[float, float]:
        if detection is None:
            self._combine_states()
            return float(self.combined_state[0]), float(self.combined_state[3])

        # Check if valid coordinates exist
        has_pos = False
        if hasattr(detection, "pixel_position") and detection.pixel_position is not None:
            has_pos = True
        elif hasattr(detection, "best") and detection.best is not None:
            has_pos = True
        elif hasattr(detection, "centroid_x") and not math.isnan(float(detection.centroid_x)):
            has_pos = True

        if not has_pos:
            self._combine_states()
            return float(self.combined_state[0]), float(self.combined_state[3])

        likelihoods = np.zeros(self.num_models, dtype=np.float64)

        for j in range(self.num_models):
            self.filters[j].update(detection, scintillation_index)

            # Compute Gaussian likelihood: L_j = N(y_j; 0, S_j)
            y = self.filters[j].innovation
            S = self.filters[j].innovation_cov
            try:
                det_S = np.linalg.det(S)
                if det_S > 1e-12:
                    S_inv = np.linalg.inv(S)
                    exponent = -0.5 * float(y.T @ S_inv @ y)
                    likelihoods[j] = (1.0 / (2.0 * math.pi * math.sqrt(det_S))) * math.exp(max(-50.0, exponent))
                else:
                    likelihoods[j] = 1e-6
            except np.linalg.LinAlgError:
                likelihoods[j] = 1e-6

        # Update mode probabilities: mu_j = L_j * c_bar_j / sum(...)
        Pi = self.config.transition_matrix
        c_bar = np.dot(self.mu, Pi)
        numerator = likelihoods * c_bar
        denom = np.sum(numerator)

        if denom > 1e-15:
            self.mu = numerator / denom
        else:
            self.mu = np.ones(self.num_models, dtype=np.float64) / self.num_models

        # Combine states
        self._combine_states()
        return float(self.combined_state[0]), float(self.combined_state[3])

    def _combine_states(self) -> None:
        """Combine states across sub-filters using current mode probabilities."""
        self.combined_state = np.zeros(6, dtype=np.float64)
        for j in range(self.num_models):
            self.combined_state += self.mu[j] * self.filters[j].state

    @property
    def estimated_pixel(self) -> tuple[float, float]:
        """Combined estimated (u, v) pixel."""
        return float(self.combined_state[0]), float(self.combined_state[3])

    @property
    def mode_probabilities(self) -> list[float]:
        """Current mode probabilities [P(cruising), P(accelerating), P(maneuver)]."""
        return [float(p) for p in self.mu]
