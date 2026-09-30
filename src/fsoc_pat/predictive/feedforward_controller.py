"""Cooperative Predictive Feedforward + Feedback Anti-Jitter Compensator.

Combines:
  - Real-time harmonic frequency tracking via FFT peak picking
  - Multi-harmonic horizon dead-time projection across system lag
  - Anti-phase feedforward injection into Fast Steering Mirror (FSM)
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from fsoc_pat.predictive.dead_time_predictor import DeadTimePredictor
from fsoc_pat.predictive.predictive_types import JitterTelemetry, PredictiveJitterConfig
from fsoc_pat.predictive.spectral_analyzer import HarmonicSpectralAnalyzer


class PredictiveFeedforwardCompensator:
    """Master predictive anti-jitter feedforward controller."""

    def __init__(self, config: PredictiveJitterConfig | None = None) -> None:
        self.config = config or PredictiveJitterConfig()
        self.analyzer = HarmonicSpectralAnalyzer(self.config)
        self.predictor = DeadTimePredictor(self.config)

        self.delay_steps = max(1, int(round(self.config.latency_horizon_s * self.config.sampling_rate_hz)))
        # Align prediction horizon with discrete delay steps
        self.predictor.tau = self.delay_steps / self.config.sampling_rate_hz
        self._cmd_queue_pan: list[float] = [0.0] * self.delay_steps
        self._cmd_queue_tilt: list[float] = [0.0] * self.delay_steps

        self.latest_telemetry: JitterTelemetry | None = None
        self._raw_history: list[float] = []
        self._residual_history: list[float] = []

    def reset(self) -> None:
        """Reset internal history, spectral analyzer, and predictor."""
        self.analyzer.reset()
        self.predictor.reset()
        self._cmd_queue_pan = [0.0] * self.delay_steps
        self._cmd_queue_tilt = [0.0] * self.delay_steps
        self.latest_telemetry = None
        self._raw_history.clear()
        self._residual_history.clear()

    def update(
        self,
        raw_jitter_pan_urad: float,
        raw_jitter_tilt_urad: float,
        timestamp_s: float,
        dt: float,
    ) -> tuple[float, float, JitterTelemetry]:
        """Compute predictive feedforward anti-jitter commands for FSM.

        Args:
            raw_jitter_pan_urad: Measured platform jitter disturbance in pan (urad).
            raw_jitter_tilt_urad: Measured platform jitter disturbance in tilt (urad).
            timestamp_s: Current simulation timestamp t (s).
            dt: Timestep in seconds.

        Returns:
            Tuple of:
              - ff_cmd_pan_urad: Injected feedforward command for FSM pan (urad).
              - ff_cmd_tilt_urad: Injected feedforward command for FSM tilt (urad).
              - telemetry: Complete JitterTelemetry record.
        """
        # 1. Update online spectral analyzer with current jitter measurements
        pan_modes, tilt_modes = self.analyzer.update(raw_jitter_pan_urad, raw_jitter_tilt_urad)

        if not self.config.enabled:
            # Feedforward bypassed
            raw_rms = math.hypot(raw_jitter_pan_urad, raw_jitter_tilt_urad)
            telem = JitterTelemetry(
                timestamp_s=timestamp_s,
                raw_jitter_pan_urad=raw_jitter_pan_urad,
                raw_jitter_tilt_urad=raw_jitter_tilt_urad,
                raw_jitter_rms_urad=raw_rms,
                residual_jitter_rms_urad=raw_rms,
                jitter_rejection_db=0.0,
                feedforward_pan_urad=0.0,
                feedforward_tilt_urad=0.0,
                phase_lead_deg=0.0,
                tracked_frequencies_hz=[],
                feedforward_active=False,
            )
            self.latest_telemetry = telem
            return 0.0, 0.0, telem

        # 2. Extrapolate future disturbance and generate anti-phase commands across tau
        pan_samples = self.analyzer.get_recent_samples("pan")
        tilt_samples = self.analyzer.get_recent_samples("tilt")

        _, ff_cmd_pan = self.predictor.predict_forward(
            pan_modes, pan_samples, timestamp_s, dt=dt
        )
        _, ff_cmd_tilt = self.predictor.predict_forward(
            tilt_modes, tilt_samples, timestamp_s, dt=dt
        )

        # 3. Enqueue command into transport latency pipeline
        self._cmd_queue_pan.append(ff_cmd_pan)
        self._cmd_queue_tilt.append(ff_cmd_tilt)
        applied_pan = self._cmd_queue_pan.pop(0)
        applied_tilt = self._cmd_queue_tilt.pop(0)

        # 4. Calculate residual pointing jitter after physical mirror deflection
        res_pan = raw_jitter_pan_urad + applied_pan
        res_tilt = raw_jitter_tilt_urad + applied_tilt

        raw_rms = math.hypot(raw_jitter_pan_urad, raw_jitter_tilt_urad)
        res_rms = math.hypot(res_pan, res_tilt)

        # Track rolling history for steady-state rejection ratio
        self._raw_history.append(raw_rms)
        self._residual_history.append(res_rms)
        if len(self._raw_history) > 100:
            self._raw_history.pop(0)
            self._residual_history.pop(0)

        mean_raw = float(np.mean(self._raw_history)) if self._raw_history else raw_rms
        mean_res = float(np.mean(self._residual_history)) if self._residual_history else res_rms

        # Jitter rejection ratio in dB (positive = attenuation)
        rejection_db = 20.0 * math.log10(max(1.0, mean_raw) / max(0.01, mean_res))

        # Extract tracked frequencies for telemetry
        tracked_freqs = [m.frequency_hz for m in pan_modes]
        for m in tilt_modes:
            if not any(abs(m.frequency_hz - f) < 2.0 for f in tracked_freqs):
                tracked_freqs.append(m.frequency_hz)
        tracked_freqs.sort()

        phase_lead = self.predictor.compute_average_phase_lead_deg(pan_modes + tilt_modes)

        telem = JitterTelemetry(
            timestamp_s=timestamp_s,
            raw_jitter_pan_urad=raw_jitter_pan_urad,
            raw_jitter_tilt_urad=raw_jitter_tilt_urad,
            raw_jitter_rms_urad=raw_rms,
            residual_jitter_rms_urad=res_rms,
            jitter_rejection_db=rejection_db,
            feedforward_pan_urad=applied_pan,
            feedforward_tilt_urad=applied_tilt,
            phase_lead_deg=phase_lead,
            tracked_frequencies_hz=tracked_freqs,
            feedforward_active=True,
        )

        self.latest_telemetry = telem
        return ff_cmd_pan, ff_cmd_tilt, telem
