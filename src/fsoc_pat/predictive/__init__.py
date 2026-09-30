"""Predictive Micro-Vibration & Latency Feedforward Anti-Jitter Subsystem.

Exports:
  - HarmonicMode, PredictiveJitterConfig, JitterTelemetry
  - HarmonicSpectralAnalyzer
  - DeadTimePredictor
  - PredictiveFeedforwardCompensator
  - JitterBenchmarkEngine, JitterBenchmarkConfig, JitterBenchmarkMetrics, generate_jitter_benchmark_plot
"""

from fsoc_pat.predictive.dead_time_predictor import DeadTimePredictor
from fsoc_pat.predictive.feedforward_controller import PredictiveFeedforwardCompensator
from fsoc_pat.predictive.predictive_benchmark import (
    JitterBenchmarkConfig,
    JitterBenchmarkEngine,
    JitterBenchmarkMetrics,
    generate_jitter_benchmark_plot,
)
from fsoc_pat.predictive.predictive_types import (
    HarmonicMode,
    JitterTelemetry,
    PredictiveJitterConfig,
)
from fsoc_pat.predictive.spectral_analyzer import HarmonicSpectralAnalyzer

__all__ = [
    "HarmonicMode",
    "PredictiveJitterConfig",
    "JitterTelemetry",
    "HarmonicSpectralAnalyzer",
    "DeadTimePredictor",
    "PredictiveFeedforwardCompensator",
    "JitterBenchmarkConfig",
    "JitterBenchmarkEngine",
    "JitterBenchmarkMetrics",
    "generate_jitter_benchmark_plot",
]
