"""Confidence engine — multi-factor detection confidence scoring.

Evaluates detection reliability using multiple independent criteria:
  1. Brightness contrast (signal-to-background ratio)
  2. Shape consistency (circularity, aspect ratio)
  3. Temporal consistency (detection persistence across frames)
  4. Spatial consistency (position stability across frames)
  5. Size consistency (area stability)

Each factor produces a score in [0, 1]. The composite confidence
is a weighted combination.

This module operates on DETECTION output only — never ground truth.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from fsoc_pat.perception.detection_types import Candidate, Detection, DetectionStatus


@dataclass
class ConfidenceConfig:
    """Configuration for the confidence engine.

    Attributes:
        brightness_weight: Weight for brightness score.
        shape_weight: Weight for shape score.
        temporal_weight: Weight for temporal consistency.
        spatial_weight: Weight for spatial consistency.
        size_weight: Weight for size consistency.
        history_length: Number of frames to keep for temporal analysis.
        max_pixel_drift: Maximum expected pixel drift between frames (px).
        min_confidence: Floor confidence value.
    """

    brightness_weight: float = 0.30
    shape_weight: float = 0.25
    temporal_weight: float = 0.20
    spatial_weight: float = 0.15
    size_weight: float = 0.10
    history_length: int = 10
    max_pixel_drift: float = 30.0
    min_confidence: float = 0.0


class ConfidenceEngine:
    """Multi-factor detection confidence scorer.

    Maintains a rolling history of detections to evaluate
    temporal and spatial consistency. First few frames have
    reduced temporal/spatial scores until history is built.
    """

    def __init__(self, config: ConfidenceConfig | None = None) -> None:
        self._config = config or ConfidenceConfig()
        self._history: deque[Detection] = deque(
            maxlen=self._config.history_length,
        )

    def evaluate(self, detection: Detection) -> float:
        """Compute composite confidence for a detection.

        Args:
            detection: Current frame detection.

        Returns:
            Confidence score in [0, 1].
        """
        cfg = self._config

        if not detection.detected or detection.best is None:
            self._history.append(detection)
            return cfg.min_confidence

        candidate = detection.best

        # Individual scores
        brightness = self._brightness_score(candidate)
        shape = self._shape_score(candidate)
        temporal = self._temporal_score()
        spatial = self._spatial_score(candidate)
        size = self._size_score(candidate)

        # Weighted composite
        total_weight = (
            cfg.brightness_weight + cfg.shape_weight + cfg.temporal_weight
            + cfg.spatial_weight + cfg.size_weight
        )

        composite = (
            cfg.brightness_weight * brightness
            + cfg.shape_weight * shape
            + cfg.temporal_weight * temporal
            + cfg.spatial_weight * spatial
            + cfg.size_weight * size
        ) / total_weight

        # Record in history
        self._history.append(detection)

        return float(np.clip(composite, cfg.min_confidence, 1.0))

    def _brightness_score(self, candidate: Candidate) -> float:
        """Score based on signal brightness vs expected background."""
        # Peak intensity normalized: 0 at 0, 1.0 at 255
        peak_norm = candidate.peak_intensity / 255.0

        # Contrast: mean vs background assumption (~20)
        contrast = max(0, candidate.mean_intensity - 20) / 235.0

        return 0.6 * peak_norm + 0.4 * contrast

    def _shape_score(self, candidate: Candidate) -> float:
        """Score based on shape consistency (circularity)."""
        # Circularity is already [0, 1]
        return candidate.circularity

    def _temporal_score(self) -> float:
        """Score based on detection persistence across recent frames.

        More consecutive detections → higher score.
        """
        if len(self._history) == 0:
            return 0.5  # Prior on first frame

        # Count consecutive detections from most recent
        consecutive = 0
        for det in reversed(self._history):
            if det.detected:
                consecutive += 1
            else:
                break

        # Normalize by history length
        return min(consecutive / max(len(self._history), 1), 1.0)

    def _spatial_score(self, candidate: Candidate) -> float:
        """Score based on position stability across frames.

        Small drift between frames → higher confidence.
        Large jumps → lower confidence (possible false positive).
        """
        cfg = self._config

        # Find last valid detection
        last_pos = None
        for det in reversed(self._history):
            if det.detected and det.best is not None:
                last_pos = (det.best.u, det.best.v)
                break

        if last_pos is None:
            return 0.5  # No history → neutral

        # Compute pixel drift
        du = candidate.u - last_pos[0]
        dv = candidate.v - last_pos[1]
        drift = math.sqrt(du * du + dv * dv)

        # Score: 1.0 if drift < threshold, decays exponentially
        if drift <= cfg.max_pixel_drift:
            return 1.0
        else:
            excess = drift - cfg.max_pixel_drift
            return math.exp(-excess / cfg.max_pixel_drift)

    def _size_score(self, candidate: Candidate) -> float:
        """Score based on area stability across frames."""
        # Find last valid detection area
        last_area = None
        for det in reversed(self._history):
            if det.detected and det.best is not None:
                last_area = det.best.area_px
                break

        if last_area is None or last_area < 1:
            return 0.5  # No history → neutral

        # Ratio-based score
        ratio = candidate.area_px / last_area
        if ratio > 1:
            ratio = 1.0 / ratio  # Make symmetric

        return ratio  # 1.0 = identical, decays to 0

    def reset(self) -> None:
        """Reset confidence engine state."""
        self._history.clear()
