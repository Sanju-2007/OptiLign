"""Tests for the simulation engine."""

import pytest

from fsoc_pat.config.models import (
    CameraConfig,
    ExperimentConfig,
    ScenarioConfig,
    TargetConfig,
)
from fsoc_pat.simulation.engine import SimulationEngine


class TestSimulationEngine:
    """Unit tests for SimulationEngine."""

    @pytest.fixture
    def nominal_config(self) -> ExperimentConfig:
        """Short nominal scenario for testing."""
        return ExperimentConfig(
            scenario=ScenarioConfig(
                name="test_nominal",
                duration_s=1.0,
                timestep_s=0.01,
                seed=42,
            ),
            target=TargetConfig(
                trajectory_type="circular",
                position=(100.0, 0.0, 0.0),
                radius=10.0,
                angular_rate=0.5,
            ),
        )

    def test_initialize_succeeds(self, nominal_config: ExperimentConfig) -> None:
        engine = SimulationEngine(nominal_config)
        engine.initialize()

    def test_run_produces_results(self, nominal_config: ExperimentConfig) -> None:
        engine = SimulationEngine(nominal_config)
        engine.initialize()
        results = engine.run()

        assert results.total_steps == 100  # 1.0s / 0.01s
        assert results.duration_s == pytest.approx(1.0)
        assert results.wall_time_s > 0

    def test_ground_truth_recorded(self, nominal_config: ExperimentConfig) -> None:
        engine = SimulationEngine(nominal_config)
        engine.initialize()
        results = engine.run()

        # Should have steps + 1 snapshots (initial + each step)
        assert len(results.ground_truth) == 101

    def test_ground_truth_timestamps_increase(
        self, nominal_config: ExperimentConfig,
    ) -> None:
        engine = SimulationEngine(nominal_config)
        engine.initialize()
        results = engine.run()

        timestamps = results.ground_truth.timestamps
        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1]

    def test_deterministic_same_seed(self) -> None:
        """Same seed + config → same ground truth."""
        config = ExperimentConfig(
            scenario=ScenarioConfig(duration_s=0.5, timestep_s=0.01, seed=123),
            target=TargetConfig(
                trajectory_type="circular",
                position=(100.0, 0.0, 0.0),
                radius=5.0,
                angular_rate=1.0,
            ),
        )

        engine1 = SimulationEngine(config)
        engine1.initialize()
        r1 = engine1.run()

        engine2 = SimulationEngine(config)
        engine2.initialize()
        r2 = engine2.run()

        assert r1.total_steps == r2.total_steps
        assert r1.final_angular_error_rad == pytest.approx(
            r2.final_angular_error_rad, abs=1e-12,
        )

        # Compare a few snapshots
        for i in [0, 10, 25]:
            s1 = r1.ground_truth[i]
            s2 = r2.ground_truth[i]
            assert s1.target.position.x == pytest.approx(s2.target.position.x)
            assert s1.target.position.y == pytest.approx(s2.target.position.y)

    def test_run_without_initialize_raises(self) -> None:
        config = ExperimentConfig(
            scenario=ScenarioConfig(duration_s=0.1, timestep_s=0.01, seed=1),
        )
        engine = SimulationEngine(config)
        with pytest.raises(RuntimeError, match="initialized"):
            engine.run()

    def test_stationary_target(self) -> None:
        """Stationary target should project to same pixel every frame."""
        config = ExperimentConfig(
            scenario=ScenarioConfig(duration_s=0.5, timestep_s=0.01, seed=1),
            target=TargetConfig(
                trajectory_type="stationary",
                position=(100.0, 0.0, 0.0),
            ),
        )
        engine = SimulationEngine(config)
        engine.initialize()
        results = engine.run()

        # All pixels should be the same (target on boresight)
        pixels = [
            s.true_pixel for s in results.ground_truth.snapshots
            if s.true_pixel is not None
        ]
        assert len(pixels) > 0
        for px in pixels:
            assert px.u == pytest.approx(pixels[0].u, abs=1e-6)
            assert px.v == pytest.approx(pixels[0].v, abs=1e-6)

    def test_linear_target(self) -> None:
        config = ExperimentConfig(
            scenario=ScenarioConfig(duration_s=0.5, timestep_s=0.01, seed=1),
            target=TargetConfig(
                trajectory_type="linear",
                position=(100.0, 0.0, 0.0),
                velocity=(0.0, 5.0, 0.0),
            ),
        )
        engine = SimulationEngine(config)
        engine.initialize()
        results = engine.run()

        # Target moves in Y → pixel u should change over time
        first_px = results.ground_truth[0].true_pixel
        last_px = results.ground_truth[-1].true_pixel
        assert first_px is not None
        assert last_px is not None
        assert last_px.u != pytest.approx(first_px.u, abs=1.0)
