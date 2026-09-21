"""AI-based beacon detector.

Wraps BeaconNet for inference within the detection pipeline.
Accepts raw grayscale frames, preprocesses (resize, normalize),
runs the CNN, and returns structured Detection results.

This module NEVER accesses ground truth — it operates purely
on image pixels, just like the classical detector.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from fsoc_pat.ai.model import BeaconNet, ModelConfig
from fsoc_pat.perception.detection_types import Candidate, Detection, DetectionStatus


class AIDetector:
    """CNN-based beacon detector.

    Uses BeaconNet to predict beacon presence and centroid
    from camera images. Faster than classical for large images
    and more robust to noise/clutter.

    Usage:
        detector = AIDetector()
        detection = detector.detect(frame)
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        model_config: ModelConfig | None = None,
        confidence_threshold: float = 0.5,
        device: str = "cpu",
    ) -> None:
        """Initialize AI detector.

        Args:
            model_path: Path to saved model weights. None = untrained.
            model_config: Model architecture config.
            confidence_threshold: Minimum presence probability to report detection.
            device: Torch device ("cpu" or "cuda").
        """
        self._config = model_config or ModelConfig()
        self._threshold = confidence_threshold
        self._device = torch.device(device)
        self._frame_index = 0

        # Initialize model
        self._model = BeaconNet(self._config)
        if model_path is not None:
            state_dict = torch.load(
                str(model_path), map_location=self._device, weights_only=True,
            )
            self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        self._model.eval()

    def detect(
        self,
        image: NDArray[np.uint8],
        timestamp: float = 0.0,
        image_width: int = 640,
        image_height: int = 480,
    ) -> Detection:
        """Detect beacon in an image using the CNN.

        Args:
            image: Grayscale (H, W) or BGR (H, W, 3) uint8.
            timestamp: Frame timestamp (s).
            image_width: Original image width for denormalization.
            image_height: Original image height for denormalization.

        Returns:
            Detection result.
        """
        t_start = time.perf_counter()
        self._frame_index += 1

        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Preprocess: resize + normalize
        resized = cv2.resize(
            gray,
            (self._config.input_w, self._config.input_h),
            interpolation=cv2.INTER_AREA,
        )
        tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)
        tensor = tensor.unsqueeze(0).unsqueeze(0).to(self._device)  # (1, 1, H, W)

        # Inference
        with torch.no_grad():
            output = self._model(tensor)

        # Post-process
        presence_prob = torch.sigmoid(output["presence_logit"]).item()
        u_norm = output["u_norm"].item()
        v_norm = output["v_norm"].item()

        # Denormalize to original image coordinates
        u = u_norm * image_width
        v = v_norm * image_height

        processing_time = (time.perf_counter() - t_start) * 1000

        if presence_prob >= self._threshold:
            candidate = Candidate(
                u=u, v=v,
                area_px=0.0,  # CNN doesn't estimate area
                peak_intensity=0.0,
                mean_intensity=0.0,
                circularity=0.0,
            )
            return Detection(
                status=DetectionStatus.VALID,
                candidates=[candidate],
                best=candidate,
                confidence=presence_prob,
                frame_index=self._frame_index,
                timestamp=timestamp,
                processing_time_ms=processing_time,
            )
        else:
            return Detection(
                status=DetectionStatus.NO_DETECTION,
                candidates=[],
                best=None,
                confidence=presence_prob,
                frame_index=self._frame_index,
                timestamp=timestamp,
                processing_time_ms=processing_time,
            )

    def reset(self) -> None:
        """Reset detector state."""
        self._frame_index = 0
