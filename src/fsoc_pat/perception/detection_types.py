"""Detection data types.

Structured results from the perception pipeline. These are the
typed outputs that flow from detector → estimator.

Ground truth is NEVER part of these types — they represent what
the perception system *measures*, not what is *true*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class DetectionStatus(Enum):
    """Status of a detection attempt."""

    NO_DETECTION = auto()       # No candidate found
    LOW_CONFIDENCE = auto()     # Candidate found but below threshold
    VALID = auto()              # Valid detection above threshold
    MULTIPLE = auto()           # Multiple candidates — ambiguous


@dataclass(frozen=True, slots=True)
class Candidate:
    """A single beacon candidate from the detection pipeline.

    Attributes:
        u: Centroid X pixel coordinate (px).
        v: Centroid Y pixel coordinate (px).
        area_px: Detected area (px²).
        peak_intensity: Peak brightness (0-255).
        mean_intensity: Mean brightness within candidate region (0-255).
        circularity: Shape circularity [0, 1]. 1 = perfect circle.
        bounding_box: (x, y, w, h) of the bounding rectangle (px).
    """

    u: float
    v: float
    area_px: float
    peak_intensity: float
    mean_intensity: float
    circularity: float = 0.0
    bounding_box: tuple[int, int, int, int] = (0, 0, 0, 0)


@dataclass(frozen=True, slots=True)
class Detection:
    """Result from the detection pipeline for a single frame.

    Attributes:
        status: Detection status.
        candidates: List of all candidates found.
        best: The selected best candidate, or None.
        confidence: Detection confidence [0, 1].
        frame_index: Which frame this detection corresponds to.
        timestamp: Simulation time of this detection (s).
        processing_time_ms: Time taken for detection (ms).
    """

    status: DetectionStatus
    candidates: list[Candidate] = field(default_factory=list)
    best: Candidate | None = None
    confidence: float = 0.0
    frame_index: int = 0
    timestamp: float = 0.0
    processing_time_ms: float = 0.0

    @property
    def detected(self) -> bool:
        """Whether a valid detection was made."""
        return (
            self.status in (DetectionStatus.VALID, DetectionStatus.MULTIPLE)
            and self.best is not None
        )

    @property
    def pixel_position(self) -> tuple[float, float] | None:
        """Detected centroid (u, v) or None."""
        if self.best is not None:
            return (self.best.u, self.best.v)
        return None

    @property
    def num_candidates(self) -> int:
        """Number of candidates found."""
        return len(self.candidates)
