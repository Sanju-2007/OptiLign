"""Classical beacon detector.

Pipeline:
  1. Preprocess → binary mask
  2. Find contours
  3. Filter candidates by area, circularity, brightness
  4. Compute sub-pixel centroid (weighted moments)
  5. Rank candidates by composite score
  6. Return best detection with confidence

This detector uses ONLY image data — it NEVER accesses ground truth.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from fsoc_pat.perception.detection_types import Candidate, Detection, DetectionStatus
from fsoc_pat.perception.preprocessing import PreprocessConfig, Preprocessor


@dataclass
class ClassicalDetectorConfig:
    """Configuration for the classical detector.

    Attributes:
        min_area_px: Minimum contour area (px²).
        max_area_px: Maximum contour area (px²).
        min_circularity: Minimum circularity [0, 1].
        min_brightness: Minimum peak brightness for candidates (0-255).
        confidence_threshold: Minimum confidence to report VALID [0, 1].
        use_weighted_centroid: Use intensity-weighted moments for centroid.
        preprocess: Preprocessing configuration.
    """

    min_area_px: float = 4.0
    max_area_px: float = 500.0
    min_circularity: float = 0.3
    min_brightness: float = 80.0
    confidence_threshold: float = 0.5
    use_weighted_centroid: bool = True
    preprocess: PreprocessConfig | None = None


class ClassicalDetector:
    """Classical computer vision beacon detector.

    Uses morphological preprocessing, contour analysis, and
    shape/brightness filtering to detect beacon candidates.
    Sub-pixel accuracy is achieved using intensity-weighted moments.
    """

    def __init__(self, config: ClassicalDetectorConfig | None = None) -> None:
        self._config = config or ClassicalDetectorConfig()
        self._preprocessor = Preprocessor(self._config.preprocess)
        self._frame_index = 0

    def detect(
        self,
        image: NDArray[np.uint8],
        timestamp: float = 0.0,
    ) -> Detection:
        """Detect beacon candidates in an image.

        Args:
            image: Grayscale (H, W) or BGR (H, W, 3) uint8 image.
            timestamp: Simulation time for this frame (s).

        Returns:
            Detection result with candidates and confidence.
        """
        t_start = time.perf_counter()
        cfg = self._config
        self._frame_index += 1

        # Ensure grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Step 1: Preprocess
        binary = self._preprocessor.process(image)

        # Step 2: Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        # Step 3: Filter and score candidates
        candidates: list[Candidate] = []
        for contour in contours:
            candidate = self._evaluate_contour(contour, gray)
            if candidate is not None:
                candidates.append(candidate)

        # Step 4: Select best candidate
        processing_time = (time.perf_counter() - t_start) * 1000

        if not candidates:
            return Detection(
                status=DetectionStatus.NO_DETECTION,
                candidates=[],
                best=None,
                confidence=0.0,
                frame_index=self._frame_index,
                timestamp=timestamp,
                processing_time_ms=processing_time,
            )

        # Rank by composite score
        scored = [(self._score_candidate(c), c) for c in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)

        best_score, best_candidate = scored[0]

        # Determine status
        if best_score >= cfg.confidence_threshold:
            status = DetectionStatus.VALID
            if len(candidates) > 1:
                status = DetectionStatus.MULTIPLE
        else:
            status = DetectionStatus.LOW_CONFIDENCE

        return Detection(
            status=status,
            candidates=candidates,
            best=best_candidate,
            confidence=best_score,
            frame_index=self._frame_index,
            timestamp=timestamp,
            processing_time_ms=processing_time,
        )

    def _evaluate_contour(
        self,
        contour: NDArray,
        gray: NDArray[np.uint8],
    ) -> Candidate | None:
        """Evaluate a single contour as a beacon candidate.

        Returns None if the contour fails any filter.
        """
        cfg = self._config

        # Area filter
        area = cv2.contourArea(contour)
        if area < cfg.min_area_px or area > cfg.max_area_px:
            return None

        # Circularity filter
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * math.pi * area / (perimeter * perimeter)
        else:
            circularity = 0.0

        if circularity < cfg.min_circularity:
            return None

        # Bounding box
        x, y, w, h = cv2.boundingRect(contour)

        # Extract ROI for brightness analysis
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        roi_pixels = gray[mask > 0]

        if len(roi_pixels) == 0:
            return None

        peak_intensity = float(np.max(roi_pixels))
        mean_intensity = float(np.mean(roi_pixels))

        # Brightness filter
        if peak_intensity < cfg.min_brightness:
            return None

        # Compute centroid
        if cfg.use_weighted_centroid:
            u, v = self._weighted_centroid(contour, gray, mask)
        else:
            M = cv2.moments(contour)
            if M["m00"] > 0:
                u = M["m10"] / M["m00"]
                v = M["m01"] / M["m00"]
            else:
                u = float(x + w / 2)
                v = float(y + h / 2)

        return Candidate(
            u=u,
            v=v,
            area_px=area,
            peak_intensity=peak_intensity,
            mean_intensity=mean_intensity,
            circularity=circularity,
            bounding_box=(x, y, w, h),
        )

    def _weighted_centroid(
        self,
        contour: NDArray,
        gray: NDArray[np.uint8],
        mask: NDArray[np.uint8],
    ) -> tuple[float, float]:
        """Compute intensity-weighted sub-pixel centroid.

        Uses image moments weighted by pixel intensity for sub-pixel
        accuracy. Falls back to geometric centroid if weights are zero.
        """
        # Weighted moments using the intensity image
        moments = cv2.moments(gray, binaryImage=False)

        # But we only want moments within the contour mask
        # Use masked image for weighted moments
        masked_gray = gray.copy()
        masked_gray[mask == 0] = 0

        M = cv2.moments(masked_gray)

        if M["m00"] > 0:
            u = M["m10"] / M["m00"]
            v = M["m01"] / M["m00"]
        else:
            # Fallback to geometric centroid
            M_geom = cv2.moments(contour)
            if M_geom["m00"] > 0:
                u = M_geom["m10"] / M_geom["m00"]
                v = M_geom["m01"] / M_geom["m00"]
            else:
                x, y, w, h = cv2.boundingRect(contour)
                u = float(x + w / 2)
                v = float(y + h / 2)

        return float(u), float(v)

    def _score_candidate(self, candidate: Candidate) -> float:
        """Compute composite confidence score for a candidate.

        Score components:
          - Brightness: higher = better (normalized)
          - Circularity: higher = better
          - Size: prefer expected beacon size (penalty for too large/small)

        Returns:
            Confidence score in [0, 1].
        """
        # Brightness score: peak / 255
        brightness_score = min(candidate.peak_intensity / 255.0, 1.0)

        # Circularity score: already in [0, 1]
        circ_score = candidate.circularity

        # Size score: prefer area near expected beacon size
        # Gaussian penalty around expected area (~20-80 px²)
        expected_area = 30.0
        area_sigma = 50.0
        area_score = math.exp(
            -((candidate.area_px - expected_area) ** 2) / (2 * area_sigma ** 2)
        )

        # Composite: weighted average
        score = (
            0.4 * brightness_score
            + 0.35 * circ_score
            + 0.25 * area_score
        )

        return float(np.clip(score, 0.0, 1.0))

    def reset(self) -> None:
        """Reset detector state."""
        self._frame_index = 0
        self._preprocessor.reset()
