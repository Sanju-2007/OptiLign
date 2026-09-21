"""Hybrid detector — fuses classical CV and AI detections.

Fusion strategy:
  1. Run classical detector → Detection_classical
  2. Run AI detector → Detection_ai
  3. Fuse based on agreement and confidence:
     - Both agree (within threshold): weighted-average centroid, boosted confidence
     - Only classical: use classical (AI may have low confidence)
     - Only AI: use AI (classical may have missed due to noise)
     - Neither: no detection

This provides robustness that neither detector alone achieves.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fsoc_pat.ai.ai_detector import AIDetector
from fsoc_pat.ai.model import ModelConfig
from fsoc_pat.perception.classical_detector import ClassicalDetector, ClassicalDetectorConfig
from fsoc_pat.perception.detection_types import Candidate, Detection, DetectionStatus


@dataclass
class HybridConfig:
    """Configuration for the hybrid detector.

    Attributes:
        classical_weight: Weight for classical detector in fusion [0, 1].
        ai_weight: Weight for AI detector in fusion [0, 1].
        agreement_threshold_px: Max pixel distance for "agreement".
        confidence_boost: Confidence boost when both detectors agree.
        classical_config: Classical detector config.
        ai_model_path: Path to trained AI model weights.
        ai_model_config: AI model architecture config.
        ai_confidence_threshold: AI detector threshold.
    """

    classical_weight: float = 0.5
    ai_weight: float = 0.5
    agreement_threshold_px: float = 15.0
    confidence_boost: float = 0.15
    classical_config: ClassicalDetectorConfig | None = None
    ai_model_path: str | None = None
    ai_model_config: ModelConfig | None = None
    ai_confidence_threshold: float = 0.5


class HybridDetector:
    """Hybrid classical + AI beacon detector.

    Fuses results from both detectors for improved accuracy
    and robustness. Falls back gracefully if either detector
    is unavailable or produces no result.
    """

    def __init__(self, config: HybridConfig | None = None) -> None:
        cfg = config or HybridConfig()
        self._config = cfg
        self._classical = ClassicalDetector(cfg.classical_config)
        self._ai = AIDetector(
            model_path=cfg.ai_model_path,
            model_config=cfg.ai_model_config,
            confidence_threshold=cfg.ai_confidence_threshold,
        )
        self._frame_index = 0

    def detect(
        self,
        image: NDArray[np.uint8],
        timestamp: float = 0.0,
    ) -> Detection:
        """Run hybrid detection on an image.

        Args:
            image: Grayscale (H, W) or BGR (H, W, 3) uint8.
            timestamp: Frame timestamp (s).

        Returns:
            Fused Detection result.
        """
        self._frame_index += 1
        cfg = self._config

        # Get image dimensions
        if len(image.shape) == 3:
            h, w = image.shape[:2]
        else:
            h, w = image.shape

        # Run both detectors
        det_classical = self._classical.detect(image, timestamp)
        det_ai = self._ai.detect(image, timestamp, image_width=w, image_height=h)

        # Fusion logic
        c_det = det_classical.detected and det_classical.best is not None
        a_det = det_ai.detected and det_ai.best is not None

        if c_det and a_det:
            # Both detected — check agreement
            dist = math.sqrt(
                (det_classical.best.u - det_ai.best.u) ** 2
                + (det_classical.best.v - det_ai.best.v) ** 2
            )

            if dist <= cfg.agreement_threshold_px:
                # Agreement — weighted average
                w_c = cfg.classical_weight
                w_a = cfg.ai_weight
                w_total = w_c + w_a

                fused_u = (w_c * det_classical.best.u + w_a * det_ai.best.u) / w_total
                fused_v = (w_c * det_classical.best.v + w_a * det_ai.best.v) / w_total

                fused_confidence = min(
                    (w_c * det_classical.confidence + w_a * det_ai.confidence) / w_total
                    + cfg.confidence_boost,
                    1.0,
                )

                candidate = Candidate(
                    u=fused_u,
                    v=fused_v,
                    area_px=det_classical.best.area_px,
                    peak_intensity=det_classical.best.peak_intensity,
                    mean_intensity=det_classical.best.mean_intensity,
                    circularity=det_classical.best.circularity,
                    bounding_box=det_classical.best.bounding_box,
                )

                # Ensure fused candidate is the primary candidate
                other_cands = [c for c in det_classical.candidates if c != det_classical.best]
                fused_candidates = [candidate] + other_cands

                return Detection(
                    status=DetectionStatus.VALID,
                    candidates=fused_candidates,
                    best=candidate,
                    confidence=fused_confidence,
                    frame_index=self._frame_index,
                    timestamp=timestamp,
                    processing_time_ms=(
                        det_classical.processing_time_ms + det_ai.processing_time_ms
                    ),
                )
            else:
                # Disagreement — prioritize confident classical detection for sub-pixel accuracy
                if det_classical.confidence >= 0.40:
                    return self._wrap(det_classical, timestamp)
                elif det_classical.confidence >= det_ai.confidence:
                    return self._wrap(det_classical, timestamp)
                else:
                    return self._wrap(det_ai, timestamp)

        elif c_det:
            return self._wrap(det_classical, timestamp)
        elif a_det:
            return self._wrap(det_ai, timestamp)
        else:
            return Detection(
                status=DetectionStatus.NO_DETECTION,
                candidates=[],
                best=None,
                confidence=max(det_classical.confidence, det_ai.confidence),
                frame_index=self._frame_index,
                timestamp=timestamp,
                processing_time_ms=(
                    det_classical.processing_time_ms + det_ai.processing_time_ms
                ),
            )

    def _wrap(self, det: Detection, timestamp: float) -> Detection:
        """Re-wrap a detection with the hybrid frame index."""
        return Detection(
            status=det.status,
            candidates=det.candidates,
            best=det.best,
            confidence=det.confidence,
            frame_index=self._frame_index,
            timestamp=timestamp,
            processing_time_ms=det.processing_time_ms,
        )

    def reset(self) -> None:
        """Reset both detectors."""
        self._classical.reset()
        self._ai.reset()
        self._frame_index = 0
