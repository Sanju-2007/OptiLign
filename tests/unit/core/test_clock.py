"""Tests for the simulation clock."""

import pytest

from fsoc_pat.core.clock import ClockMode, SimulationClock


class TestSimulationClock:
    """Unit tests for SimulationClock."""

    def test_initial_state(self) -> None:
        clock = SimulationClock(dt=0.01)
        assert clock.time_s == 0.0
        assert clock.step == 0
        assert clock.dt == 0.01

    def test_tick_advances_time(self) -> None:
        clock = SimulationClock(dt=0.01)
        t = clock.tick()
        assert t.time_s == pytest.approx(0.01)
        assert t.step == 1

    def test_multiple_ticks(self) -> None:
        clock = SimulationClock(dt=0.1)
        for _ in range(10):
            clock.tick()
        assert clock.time_s == pytest.approx(1.0)
        assert clock.step == 10

    def test_reset(self) -> None:
        clock = SimulationClock(dt=0.01)
        clock.tick()
        clock.tick()
        clock.reset()
        assert clock.time_s == 0.0
        assert clock.step == 0

    def test_now_property(self) -> None:
        clock = SimulationClock(dt=0.05)
        clock.tick()
        now = clock.now
        assert now.time_s == pytest.approx(0.05)
        assert now.step == 1

    def test_invalid_dt_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            SimulationClock(dt=0.0)
        with pytest.raises(ValueError, match="positive"):
            SimulationClock(dt=-0.01)

    def test_offline_mode_does_not_block(self) -> None:
        """OFFLINE mode should advance instantly without sleeping."""
        import time

        clock = SimulationClock(dt=1.0, mode=ClockMode.OFFLINE)
        clock.start()
        start = time.monotonic()
        for _ in range(10):
            clock.tick()
        elapsed = time.monotonic() - start
        # 10 ticks of 1s each should take << 1s in offline mode
        assert elapsed < 0.5

    def test_speed_factor_validation(self) -> None:
        clock = SimulationClock(dt=0.01)
        with pytest.raises(ValueError, match="positive"):
            clock.set_speed_factor(0)
        with pytest.raises(ValueError, match="positive"):
            clock.set_speed_factor(-1.0)
