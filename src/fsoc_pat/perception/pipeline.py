"""Perception pipeline — end-to-end image-based beacon detection.

Wires together: Renderer → Preprocessor → Detector → Confidence Engine.

This is the main entry point for the perception subsystem. It takes
world state and produces structured detections that flow into the
estimator. It NEVER exposes ground truth to downstream modules.

Usage:
    pipeline = PerceptionPipeline(intrinsics, config)
    detection = pipeline.process_frame(camera_state, target_position, beacon, t)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pathlib import Path

from fsoc_pat.core.state import CameraState
from fsoc_pat.geometry.camera_model import CameraIntrinsics
from fsoc_pat.perception.classical_detector import ClassicalDetector, ClassicalDetectorConfig
from fsoc_pat.perception.confidence import ConfidenceConfig, ConfidenceEngine
from fsoc_pat.perception.detection_types import Detection, DetectionStatus
from fsoc_pat.perception.renderer import RenderConfig, SyntheticRenderer
from fsoc_pat.simulation.beacon import BeaconProperties


@dataclass
class PerceptionConfig:
    """Configuration for the full perception pipeline.

    Attributes:
        render: Renderer configuration.
        detector: Detector configuration.
        confidence: Confidence engine configuration.
        detector_type: Detector backend: "classical", "hybrid", "ai", or "adaptive".
        ai_model_path: Path to trained BeaconNet weights (for hybrid or ai mode).
        ai_confidence_threshold: Minimum presence probability for AI detections.
    """

    render: RenderConfig | None = None
    detector: ClassicalDetectorConfig | None = None
    confidence: ConfidenceConfig | None = None
    detector_type: str = "classical"
    ai_model_path: str | Path | None = None
    ai_confidence_threshold: float = 0.35


class PerceptionPipeline:
    """End-to-end perception pipeline.

    Produces detection results from world state by:
    1. Rendering a synthetic camera image
    2. Running beacon detection (Classical, Hybrid Classical+AI, or Adaptive)
    3. Scoring detection confidence

    The pipeline exposes the rendered frame for visualization
    but never leaks ground truth into detections.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        config: PerceptionConfig | None = None,
    ) -> None:
        cfg = config or PerceptionConfig()
        self._config = cfg
        self._intrinsics = intrinsics
        self._renderer = SyntheticRenderer(intrinsics, cfg.render)

        # Initialize detector based on configured detector_type
        det_type = str(cfg.detector_type).lower().strip()
        if det_type in ("hybrid", "ai_hybrid", "deep_hybrid"):
            # Discover model weights
            weights_path = cfg.ai_model_path
            if weights_path is None:
                default_path = Path(__file__).resolve().parent.parent.parent.parent / "models" / "beaconnet_v1.pt"
                if default_path.is_file():
                    weights_path = default_path

            if weights_path is not None and Path(weights_path).is_file():
                try:
                    from fsoc_pat.ai.hybrid_detector import HybridConfig, HybridDetector
                    self._detector = HybridDetector(
                        HybridConfig(
                            classical_config=cfg.detector,
                            ai_model_path=str(weights_path),
                            ai_confidence_threshold=cfg.ai_confidence_threshold,
                        )
                    )
                except Exception:
                    self._detector = ClassicalDetector(cfg.detector)
            else:
                self._detector = ClassicalDetector(cfg.detector)

        elif det_type in ("ai", "beaconnet", "cnn"):
            weights_path = cfg.ai_model_path
            if weights_path is None:
                default_path = Path(__file__).resolve().parent.parent.parent.parent / "models" / "beaconnet_v1.pt"
                if default_path.is_file():
                    weights_path = default_path
            try:
                from fsoc_pat.ai.ai_detector import AIDetector
                self._detector = AIDetector(
                    model_path=weights_path,
                    confidence_threshold=cfg.ai_confidence_threshold,
                )
            except Exception:
                self._detector = ClassicalDetector(cfg.detector)

        elif det_type in ("adaptive", "environment_adaptive"):
            try:
                from fsoc_pat.perception.adaptive_detector import AdaptiveDetector
                self._detector = AdaptiveDetector()
            except Exception:
                self._detector = ClassicalDetector(cfg.detector)

        else:
            self._detector = ClassicalDetector(cfg.detector)

        self._confidence = ConfidenceEngine(cfg.confidence)
        self._last_frame: NDArray[np.uint8] | None = None
        self._frame_count = 0

    def process_frame(
        self,
        camera_state: CameraState,
        target_positions: list[tuple[NDArray[np.float64], BeaconProperties]],
        timestamp: float = 0.0,
    ) -> Detection:
        """Run perception on a single frame.

        Args:
            camera_state: Current camera state.
            target_positions: List of (world_pos, beacon) tuples.
            timestamp: Simulation time (s).

        Returns:
            Detection result with confidence.
        """
        self._frame_count += 1

        # 1. Render
        frame = self._renderer.render(camera_state, target_positions)
        self._last_frame = frame

        # 2. Detect
        try:
            detection = self._detector.detect(frame, timestamp)
        except TypeError:
            detection = self._detector.detect(frame)

        # 3. Confidence scoring
        confidence = self._confidence.evaluate(detection)
        if detection.confidence > 0.0:
            confidence = max(confidence, detection.confidence)

        # Return detection with updated confidence
        return Detection(
            status=detection.status,
            candidates=detection.candidates,
            best=detection.best,
            confidence=confidence,
            frame_index=self._frame_count,
            timestamp=timestamp,
            processing_time_ms=detection.processing_time_ms,
        )

    def process_image(
        self,
        image: NDArray[np.uint8],
        timestamp: float = 0.0,
    ) -> Detection:
        """Run detection on a pre-rendered image.

        Use when you have an image from an external source
        (e.g., recorded footage or hardware camera).

        Args:
            image: Grayscale (H, W) or BGR (H, W, 3) uint8.
            timestamp: Simulation time (s).

        Returns:
            Detection result with confidence.
        """
        self._frame_count += 1
        self._last_frame = image

        try:
            detection = self._detector.detect(image, timestamp)
        except TypeError:
            detection = self._detector.detect(image)

        confidence = self._confidence.evaluate(detection)
        if detection.confidence > 0.0:
            confidence = max(confidence, detection.confidence)

        return Detection(
            status=detection.status,
            candidates=detection.candidates,
            best=detection.best,
            confidence=confidence,
            frame_index=self._frame_count,
            timestamp=timestamp,
            processing_time_ms=detection.processing_time_ms,
        )

    @property
    def last_frame(self) -> NDArray[np.uint8] | None:
        """Most recently rendered frame (for visualization)."""
        return self._last_frame

    @property
    def frame_count(self) -> int:
        """Number of frames processed."""
        return self._frame_count

    def reset(self) -> None:
        """Reset all pipeline state."""
        if hasattr(self._detector, "reset"):
            self._detector.reset()
        self._confidence.reset()
        self._last_frame = None
        self._frame_count = 0

