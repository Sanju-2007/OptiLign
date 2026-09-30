"""Dead-Time Predictor & Zero-Phase-Lag Horizon Extrapolator.

Projects line-of-sight mechanical platform disturbances across the system
dead-time horizon tau_lag (exposure + read + compute + actuator lag):
  1. Analytical multi-harmonic forward extrapolation (delta_phi = 2*pi*f*tau)
  2. Gauss-Markov autoregressive extrapolation for stochastic wind gusts
  3. Synthesizes anti-phase feedforward commands for Fast Steering Mirrors (FSM)
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from fsoc_pat.predictive.predictive_types import HarmonicMode, PredictiveJitterConfig


class DeadTimePredictor:
    """Predicts future disturbance displacement across dead-time latency tau_lag."""

    def __init__(self, config: PredictiveJitterConfig | None = None) -> None:
        self.config = config or PredictiveJitterConfig()
        self.tau = self.config.latency_horizon_s
        self.gain = self.config.feedforward_gain

        # Internal phase accumulators for seamless continuous synthesis
        self._last_t: float = 0.0

    def reset(self) -> None:
        """Reset internal predictor state."""
        self._last_t = 0.0

    def predict_forward(
        self,
        harmonics: Sequence[HarmonicMode],
        recent_samples: Sequence[float] | np.ndarray | float,
        timestamp_s: float = 0.0,
        dt: float = 0.002,
        correlation_time_s: float = 0.05,
    ) -> tuple[float, float]:
        """Extrapolate disturbance displacement across the latency horizon tau.

        Uses multi-harmonic least-squares regression over the recent sample buffer
        to achieve zero-phase-lag dead-time projection without phase drift.

        Args:
            harmonics: Identified dominant harmonic modes for this axis.
            recent_samples: Recent sample buffer or single current measurement (urad).
            timestamp_s: Current simulation timestamp t (s).
            dt: Discrete timestep in seconds.
            correlation_time_s: Autoregressive correlation time for broadband jitter.

        Returns:
            Tuple of:
              - predicted_future_displacement_urad: Expected disturbance at t + tau (urad).
              - feedforward_command_urad: Counter-acting FSM deflection command (urad).
        """
        tau = self.tau

        if isinstance(recent_samples, (int, float)):
            samples = [float(recent_samples)]
            current_raw_urad = float(recent_samples)
        else:
            samples = list(recent_samples)
            current_raw_urad = float(samples[-1]) if samples else 0.0

        M = min(64, len(samples))
        if harmonics and M >= 32:
            arr = np.asarray(samples[-M:], dtype=np.float64)[::-1]
            delta_t = -np.arange(M, dtype=np.float64) * dt

            cols = []
            future_row = []
            now_row = []

            for mode in harmonics:
                omega = 2.0 * math.pi * mode.frequency_hz
                cols.append(np.cos(omega * delta_t))
                cols.append(np.sin(omega * delta_t))
                future_row.extend([math.cos(omega * tau), math.sin(omega * tau)])
                now_row.extend([1.0, 0.0])

            cols.append(np.ones(M, dtype=np.float64))
            future_row.append(1.0)
            now_row.append(1.0)

            A = np.column_stack(cols)
            coeffs, _, _, _ = np.linalg.lstsq(A, arr, rcond=None)

            harmonic_future_urad = float(np.dot(future_row, coeffs))
            harmonic_now_urad = float(np.dot(now_row, coeffs))

            broadband_residual = current_raw_urad - harmonic_now_urad
            decay = math.exp(-tau / max(1e-4, correlation_time_s)) if correlation_time_s > 0 else 0.0
            broadband_future_urad = broadband_residual * decay

            pred_future_urad = harmonic_future_urad + broadband_future_urad
        else:
            decay = math.exp(-tau / max(1e-4, correlation_time_s)) if correlation_time_s > 0 else 0.0
            pred_future_urad = current_raw_urad * decay

        ff_cmd_urad = -self.gain * pred_future_urad
        return pred_future_urad, ff_cmd_urad

    def compute_average_phase_lead_deg(self, harmonics: Sequence[HarmonicMode]) -> float:
        """Compute the energy-weighted phase lead advance in degrees."""
        if not harmonics:
            return 0.0

        total_weight = 0.0
        weighted_lead = 0.0

        for h in harmonics:
            lead_deg = (360.0 * h.frequency_hz * self.tau) % 360.0
            weight = h.amplitude_urad ** 2
            weighted_lead += lead_deg * weight
            total_weight += weight

        if total_weight <= 0.0:
            return 0.0

        return weighted_lead / total_weight
