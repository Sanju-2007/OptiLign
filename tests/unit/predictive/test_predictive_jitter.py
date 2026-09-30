"""Unit tests for Predictive Micro-Vibration & Latency Feedforward Compensator."""

import math
from pathlib import Path
import numpy as np
import pytest

from fsoc_pat.predictive.dead_time_predictor import DeadTimePredictor
from fsoc_pat.predictive.feedforward_controller import PredictiveFeedforwardCompensator
from fsoc_pat.predictive.predictive_benchmark import (
    JitterBenchmarkConfig,
    JitterBenchmarkEngine,
    generate_jitter_benchmark_plot,
)
from fsoc_pat.predictive.predictive_types import (
    HarmonicMode,
    JitterTelemetry,
    PredictiveJitterConfig,
)
from fsoc_pat.predictive.spectral_analyzer import HarmonicSpectralAnalyzer


class TestHarmonicSpectralAnalyzer:
    def test_single_tone_detection(self):
        fs = 500.0
        analyzer = HarmonicSpectralAnalyzer(
            PredictiveJitterConfig(history_buffer_len=128, sampling_rate_hz=fs)
        )
        f0 = 45.0
        amp0 = 20.0
        dt = 1.0 / fs

        pan_modes = []
        tilt_modes = []
        for i in range(128):
            t = i * dt
            sig_pan = amp0 * math.sin(2.0 * math.pi * f0 * t)
            sig_tilt = (amp0 * 0.5) * math.cos(2.0 * math.pi * f0 * t)
            pan_modes, tilt_modes = analyzer.update(sig_pan, sig_tilt)

        assert len(pan_modes) >= 1
        assert abs(pan_modes[0].frequency_hz - f0) < 1.0
        assert pan_modes[0].amplitude_urad > 15.0
        assert pan_modes[0].snr_db > 10.0

        assert len(tilt_modes) >= 1
        assert abs(tilt_modes[0].frequency_hz - f0) < 1.0

    def test_multi_tone_detection(self):
        fs = 500.0
        analyzer = HarmonicSpectralAnalyzer(
            PredictiveJitterConfig(history_buffer_len=128, sampling_rate_hz=fs, num_harmonics=3)
        )
        freqs = [25.0, 65.0, 110.0]
        dt = 1.0 / fs

        pan_modes = []
        for i in range(128):
            t = i * dt
            sig = sum(15.0 * math.sin(2.0 * math.pi * f * t) for f in freqs)
            pan_modes, _ = analyzer.update(sig, sig)

        detected_freqs = [m.frequency_hz for m in pan_modes]
        for f in freqs:
            assert any(abs(df - f) < 2.0 for df in detected_freqs), f"Mode {f} Hz not found in {detected_freqs}"

    def test_reset_clears_buffer_and_modes(self):
        analyzer = HarmonicSpectralAnalyzer()
        for i in range(70):
            analyzer.update(10.0, 5.0)
        assert len(analyzer.get_recent_samples("pan")) > 0
        analyzer.reset()
        assert len(analyzer.get_recent_samples("pan")) == 0
        assert len(analyzer.pan_harmonics) == 0


class TestDeadTimePredictor:
    def test_least_squares_forward_prediction(self):
        fs = 500.0
        dt = 1.0 / fs
        tau = 0.010  # 10 ms lag
        f0 = 50.0
        amp = 18.0
        omega = 2.0 * math.pi * f0

        predictor = DeadTimePredictor(PredictiveJitterConfig(latency_horizon_s=tau, feedforward_gain=1.0))
        harmonics = [HarmonicMode(frequency_hz=f0, amplitude_urad=amp, phase_rad=0.0, snr_db=30.0)]

        # Generate 64 past samples up to current time t_now
        t_now = 0.500
        samples = [amp * math.cos(omega * (t_now - n * dt) + 0.35) for n in range(63, -1, -1)]

        pred_future, ff_cmd = predictor.predict_forward(harmonics, samples, timestamp_s=t_now, dt=dt)
        actual_future = amp * math.cos(omega * (t_now + tau) + 0.35)

        # Accuracy should be within 0.1 urad
        assert abs(pred_future - actual_future) < 0.1
        assert abs(ff_cmd - (-actual_future)) < 0.1

    def test_phase_lead_calculation(self):
        tau = 0.015
        predictor = DeadTimePredictor(PredictiveJitterConfig(latency_horizon_s=tau))
        harmonics = [
            HarmonicMode(frequency_hz=60.0, amplitude_urad=10.0, phase_rad=0.0, snr_db=20.0)
        ]
        lead_deg = predictor.compute_average_phase_lead_deg(harmonics)
        expected_deg = (360.0 * 60.0 * 0.015) % 360.0  # 324 deg
        assert abs(lead_deg - expected_deg) < 0.01


class TestPredictiveFeedforwardCompensator:
    def test_bypass_mode(self):
        comp = PredictiveFeedforwardCompensator(PredictiveJitterConfig(enabled=False))
        ff_pan, ff_tilt, telem = comp.update(12.0, -8.0, 0.1, 0.002)
        assert ff_pan == 0.0
        assert ff_tilt == 0.0
        assert not telem.feedforward_active

    def test_jitter_rejection_performance(self):
        cfg = PredictiveJitterConfig(
            latency_horizon_s=0.014,
            sampling_rate_hz=500.0,
            feedforward_gain=0.98,
            history_buffer_len=128,
        )
        comp = PredictiveFeedforwardCompensator(cfg)
        dt = 1.0 / cfg.sampling_rate_hz
        f0 = 40.0
        amp = 20.0

        for i in range(250):
            t = i * dt
            raw = amp * math.sin(2.0 * math.pi * f0 * t)
            comp.update(raw, raw, t, dt)

        telem = comp.latest_telemetry
        assert telem is not None
        assert telem.feedforward_active
        # Over 250 steps, rejection should be substantial (> 12 dB)
        assert telem.jitter_rejection_db > 12.0


class TestJitterBenchmarkEngine:
    def test_benchmark_run_and_plot(self, tmp_path: Path):
        cfg = JitterBenchmarkConfig(
            duration_s=0.4,
            dt=0.002,
            latency_horizon_s=0.014,
            harmonic_freqs_hz=(30.0, 70.0),
            harmonic_amps_urad=(20.0, 10.0),
            wind_buffet_std_urad=1.5,
            seed=42,
        )
        engine = JitterBenchmarkEngine(cfg)
        metrics, records = engine.run()

        assert metrics.raw_jitter_rms_urad > 10.0
        assert metrics.feedforward_rejection_db > 12.0
        assert metrics.net_improvement_over_feedback_db > 12.0
        assert len(records) == int(cfg.duration_s / cfg.dt)

        plot_path = tmp_path / "jitter_test.png"
        generate_jitter_benchmark_plot(metrics, records, plot_path)
        assert plot_path.exists()
        assert plot_path.stat().st_size > 50000
