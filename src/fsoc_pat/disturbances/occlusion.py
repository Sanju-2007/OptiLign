"""Occlusion and line-of-sight dropout models.

Simulates cloud obscuration, building/mast blockages, or severe
glare dropouts where the beacon disappears from the optical sensor.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OcclusionConfig:
    """Configuration for line-of-sight occlusions.

    Attributes:
        intervals: Explicit list of (start_time_s, end_time_s) blackout windows.
        transition_duration_s: Edge fading duration into/out of occlusion (s).
    """

    intervals: list[tuple[float, float]] = field(
        default_factory=lambda: [(2.0, 3.5)]  # Default 1.5s occlusion from t=2.0 to 3.5s
    )
    transition_duration_s: float = 0.05


class OcclusionModel:
    """Evaluates optical visibility and transmission through occlusions."""

    def __init__(self, config: OcclusionConfig | None = None) -> None:
        self.config = config or OcclusionConfig()

    def is_occluded(self, t: float) -> bool:
        """Return True if the target is completely blocked at time t."""
        for start_t, end_t in self.config.intervals:
            if start_t <= t <= end_t:
                return True
        return False

    def transmission(self, t: float) -> float:
        """Compute optical transmission factor in [0.0, 1.0].

        Returns 1.0 in clear sky, 0.0 during full occlusion, with a smooth
        linear ramp during the transition edge.
        """
        cfg = self.config
        trans = 1.0

        for start_t, end_t in cfg.intervals:
            edge = cfg.transition_duration_s

            if t < start_t - edge or t > end_t + edge:
                continue

            if start_t <= t <= end_t:
                return 0.0

            # Fading in (leading edge)
            if start_t - edge <= t < start_t:
                factor = (start_t - t) / edge
                trans = min(trans, factor)

            # Fading out (trailing edge)
            elif end_t < t <= end_t + edge:
                factor = (t - end_t) / edge
                trans = min(trans, factor)

        return max(0.0, min(1.0, trans))
