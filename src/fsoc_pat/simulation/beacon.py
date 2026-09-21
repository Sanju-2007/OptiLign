"""Beacon properties model.

Defines the optical and physical characteristics of a beacon target
that the perception system uses for identification. These properties
are separate from the target's trajectory.

Units:
    intensity: arbitrary units (0-255 for rendering)
    size: pixels (apparent size in image)
    blink_frequency: Hz
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class BeaconShape(Enum):
    """Beacon appearance shape."""

    POINT = auto()      # Sub-pixel or near-pixel point source
    CIRCLE = auto()     # Circular (Gaussian-like) spot
    SQUARE = auto()     # Square beacon (for testing)
    RING = auto()       # Ring/annular pattern


@dataclass(frozen=True, slots=True)
class BeaconProperties:
    """Optical and physical properties of a beacon target.

    Attributes:
        intensity: Peak brightness (0-255). Used for rendering.
        size_px: Apparent diameter in pixels at reference range.
        shape: Beacon shape type.
        color_channel: Which color channel(s) the beacon emits in.
            0 = grayscale/white, 1 = red, 2 = green, 3 = blue.
        blink_frequency_hz: Blinking frequency (Hz). 0 = continuous.
        blink_duty_cycle: Fraction of time beacon is ON [0, 1].
            1.0 = always on.
        spectral_id: Identifier for spectral signature matching.
            Design assumption: integer ID for simple matching.
    """

    intensity: float = 220.0
    size_px: float = 5.0
    shape: BeaconShape = BeaconShape.CIRCLE
    color_channel: int = 0
    blink_frequency_hz: float = 0.0
    blink_duty_cycle: float = 1.0
    spectral_id: int = 1

    def __post_init__(self) -> None:
        if not (0.0 <= self.intensity <= 255.0):
            raise ValueError(f"Intensity must be in [0, 255], got {self.intensity}")
        if self.size_px <= 0:
            raise ValueError(f"Size must be positive, got {self.size_px}")
        if not (0.0 <= self.blink_duty_cycle <= 1.0):
            raise ValueError(
                f"Blink duty cycle must be in [0, 1], got {self.blink_duty_cycle}"
            )
        if self.blink_frequency_hz < 0:
            raise ValueError(
                f"Blink frequency must be >= 0, got {self.blink_frequency_hz}"
            )

    def is_visible_at_time(self, t: float) -> bool:
        """Whether the beacon is in its ON state at time t.

        Args:
            t: Simulation time (s).

        Returns:
            True if the beacon is emitting at time t.
        """
        if self.blink_frequency_hz <= 0 or self.blink_duty_cycle >= 1.0:
            return True

        period = 1.0 / self.blink_frequency_hz
        phase = (t % period) / period  # [0, 1)
        return phase < self.blink_duty_cycle
