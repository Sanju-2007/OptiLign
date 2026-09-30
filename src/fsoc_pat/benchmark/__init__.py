"""Official Pointing, Acquisition, and Tracking (PAT) Benchmark Engine.

Provides automated compliance verification against reference tracking specifications:
- 4 Trajectories: Straight, Circular, Figure-8, Random
- Disturbance conditions: Noise, Jitter, Platform Motion, Weather
- Performance limits: Acquisition <= 2s, Tracking <= 10px, Loss < 5%, Reacq <= 1s, Speed >= 20 FPS
"""

from fsoc_pat.benchmark.standard_benchmark import (
    StandardBenchmarkConfig,
    StandardBenchmarkEngine,
    StandardBenchmarkResult,
    run_full_standard_benchmark,
)

__all__ = [
    "StandardBenchmarkConfig",
    "StandardBenchmarkEngine",
    "StandardBenchmarkResult",
    "run_full_standard_benchmark",
]
