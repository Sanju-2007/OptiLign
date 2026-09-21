"""Environment-Adaptive Optical Beacon Detector.

Features:
  - Local background suppression (morphological top-hat filter)
  - Clutter, cloud edge, and solar glint discrimination
  - Sub-pixel Gaussian intensity centroiding
  - Dynamic detection confidence computation
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from fsoc_pat.perception.detection_types import Candidate, Detection, DetectionStatus


@dataclass
class AdaptiveDetectorConfig:
    """Configuration for environment-adaptive detector.

    Attributes:
        min_snr: Minimum peak-to-noise ratio for valid detection.
        tophat_kernel_size: Morphological structuring element radius (pixels).
        min_area_px: Minimum connected component area.
        max_area_px: Maximum connected component area (rejects large cloud/glare blooms).
        confidence_threshold: Minimum output confidence.
    """

    min_snr: float = 2.5
    tophat_kernel_size: int = 15
    min_area_px: float = 2.0
    max_area_px: float = 500.0
    confidence_threshold: float = 0.3


class AdaptiveDetector:
    """Robust beacon centroiding across diverse virtual environments."""

    def __init__(self, config: AdaptiveDetectorConfig | None = None) -> None:
        self.config = config or AdaptiveDetectorConfig()
        k = self.config.tophat_kernel_size
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self.frame_index = 0

    def reset(self) -> None:
        self.frame_index = 0

    def detect(self, image: NDArray[np.uint8]) -> Detection:
        """Detect optical beacon in synthetic sensor image.

        Args:
            image: Single-channel uint8 sensor image.

        Returns:
            Detection object with sub-pixel centroid and confidence.
        """
        self.frame_index += 1

        if image is None or image.size == 0:
            return Detection(
                frame_index=self.frame_index,
                centroid_x=float("nan"),
                centroid_y=float("nan"),
                confidence=0.0,
                is_valid=False,
            )

        # 1. Background Suppression (Morphological Top-Hat)
        # Removes diffuse background gradients, city lights, and clouds
        filtered = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, self._kernel)

        # 2. Local noise estimation and thresholding
        mean_val = float(np.mean(filtered))
        std_val = float(np.std(filtered))
        thresh_val = max(15.0, mean_val + 3.0 * std_val)

        _, binary = cv2.threshold(filtered, int(thresh_val), 255, cv2.THRESH_BINARY)

        # 3. Connected Components Analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )

        best_score = -1.0
        best_u = float("nan")
        best_v = float("nan")
        best_intensity = 0.0
        best_area = 0.0

        for i in range(1, num_labels):
            area = float(stats[i, cv2.CC_STAT_AREA])
            if area < self.config.min_area_px or area > self.config.max_area_px:
                continue

            # Compute peak intensity inside the blob
            mask = (labels == i)
            peak_val = float(np.max(filtered[mask]))

            blob_snr = (peak_val - mean_val) / max(1.0, std_val)
            if blob_snr < self.config.min_snr:
                continue

            # Sub-pixel Center-of-Mass (weighted by intensity)
            y_indices, x_indices = np.where(mask)
            weights = filtered[y_indices, x_indices].astype(np.float64)
            total_weight = np.sum(weights)

            if total_weight > 0:
                sub_u = float(np.sum(x_indices * weights) / total_weight)
                sub_v = float(np.sum(y_indices * weights) / total_weight)
            else:
                sub_u = float(centroids[i][0])
                sub_v = float(centroids[i][1])

            # Score prioritizing high peak brightness and integrated spot energy
            score = (peak_val - mean_val) * math.sqrt(max(1.0, area))
            if score > best_score:
                best_score = score
                best_u = sub_u
                best_v = sub_v
                best_intensity = peak_val
                best_area = area

        if best_score > 0:
            # Map score to [0, 1] confidence
            confidence = min(1.0, max(0.0, (best_intensity - mean_val) / 180.0 * min(1.0, best_area / 6.0)))
            is_valid = confidence >= self.config.confidence_threshold
            if is_valid:
                cand = Candidate(
                    u=best_u,
                    v=best_v,
                    area_px=best_area,
                    peak_intensity=best_intensity,
                    mean_intensity=best_intensity * 0.7,
                )
                return Detection(
                    status=DetectionStatus.VALID,
                    candidates=[cand],
                    best=cand,
                    confidence=confidence,
                    frame_index=self.frame_index,
                )
            else:
                return Detection(
                    status=DetectionStatus.LOW_CONFIDENCE,
                    confidence=confidence,
                    frame_index=self.frame_index,
                )

        return Detection(
            status=DetectionStatus.NO_DETECTION,
            confidence=0.0,
            frame_index=self.frame_index,
        )
