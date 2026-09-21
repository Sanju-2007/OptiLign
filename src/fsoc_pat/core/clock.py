"""Simulation clock — manages simulation time independently of wall-clock and GUI time.

The clock supports multiple operating modes:
- OFFLINE: Deterministic, as-fast-as-possible stepping (default for headless).
- REALTIME: Simulation time matches wall-clock time.
- ACCELERATED: Simulation time runs at a configurable multiple of wall-clock time.
- SLOWED: Simulation time runs slower than wall-clock time.

The GUI must never determine physical simulation timing. The clock is the sole
authority on simulation time progression.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto

from fsoc_pat.core.types import SimulationTime


class ClockMode(Enum):
    """Clock operating mode."""

    OFFLINE = auto()       # Deterministic, no wall-clock coupling
    REALTIME = auto()      # 1:1 with wall-clock
    ACCELERATED = auto()   # Faster than wall-clock
    SLOWED = auto()        # Slower than wall-clock


@dataclass
class SimulationClock:
    """Manages simulation time progression.

    Attributes:
        dt: Simulation timestep (s). Must be positive.
        mode: Clock operating mode.
        speed_factor: Time scaling factor. 1.0 = real-time, 2.0 = 2× speed.
            Only applies in ACCELERATED/SLOWED modes.
    """

    dt: float
    mode: ClockMode = ClockMode.OFFLINE
    speed_factor: float = 1.0

    # Internal state — not constructor arguments
    _time_s: float = field(default=0.0, init=False, repr=False)
    _step: int = field(default=0, init=False, repr=False)
    _wall_start: float = field(default=0.0, init=False, repr=False)
    _running: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.dt <= 0:
            raise ValueError(f"Clock dt must be positive, got {self.dt}")
        if self.speed_factor <= 0:
            raise ValueError(f"speed_factor must be positive, got {self.speed_factor}")

    @property
    def time_s(self) -> float:
        """Current simulation time (s)."""
        return self._time_s

    @property
    def step(self) -> int:
        """Current step index."""
        return self._step

    @property
    def now(self) -> SimulationTime:
        """Current simulation time as a SimulationTime object."""
        return SimulationTime(time_s=self._time_s, step=self._step)

    @property
    def wall_elapsed_s(self) -> float:
        """Wall-clock time elapsed since start (s)."""
        if not self._running:
            return 0.0
        return time.monotonic() - self._wall_start

    def start(self) -> None:
        """Start the clock. Records wall-clock reference."""
        self._wall_start = time.monotonic()
        self._running = True

    def tick(self) -> SimulationTime:
        """Advance one simulation timestep.

        In REALTIME/ACCELERATED/SLOWED modes, this blocks until the
        appropriate wall-clock time has elapsed. In OFFLINE mode, it
        advances immediately.

        Returns:
            The new SimulationTime after the tick.
        """
        if self.mode != ClockMode.OFFLINE and self._running:
            self._wait_for_wall_clock()

        self._time_s += self.dt
        self._step += 1
        return self.now

    def reset(self) -> None:
        """Reset simulation time to zero."""
        self._time_s = 0.0
        self._step = 0
        self._running = False

    def set_speed_factor(self, factor: float) -> None:
        """Change the time scaling factor.

        Args:
            factor: New speed factor. Must be positive.
        """
        if factor <= 0:
            raise ValueError(f"speed_factor must be positive, got {factor}")
        self.speed_factor = factor

    def _wait_for_wall_clock(self) -> None:
        """Block until wall-clock matches the next simulation time.

        Uses the speed_factor to determine the effective rate.
        For REALTIME mode, speed_factor is treated as 1.0.
        """
        effective_factor = 1.0 if self.mode == ClockMode.REALTIME else self.speed_factor
        target_wall_time = (self._time_s + self.dt) / effective_factor
        elapsed = time.monotonic() - self._wall_start
        sleep_duration = target_wall_time - elapsed

        if sleep_duration > 0:
            time.sleep(sleep_duration)
