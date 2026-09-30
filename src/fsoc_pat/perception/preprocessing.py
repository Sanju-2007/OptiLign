"""Image preprocessing pipeline.

Prepares raw camera frames for detection. Operations:
  1. Grayscale conversion (if needed)
  2. Background subtraction (optional temporal or spatial)
  3. Gaussian blur / denoising
  4. Adaptive or global thresholding
  5. Morphological operations (open/close)

Each step is independently configurable and bypassable.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class PreprocessConfig:
    """Preprocessing pipeline configuration.

    Attributes:
        blur_kernel: Gaussian blur kernel size (must be odd). 0 = disabled.
        denoise_strength: Non-local means denoising strength. 0 = disabled.
        threshold_method: "global", "adaptive", or "otsu".
        threshold_value: Global threshold value (0-255). Used if method = "global".
        adaptive_block_size: Block size for adaptive thresholding (must be odd).
        adaptive_c: Constant subtracted from mean in adaptive thresh.
        morph_open_kernel: Morphological opening kernel size. 0 = disabled.
        morph_close_kernel: Morphological closing kernel size. 0 = disabled.
        background_subtraction: Whether to apply background subtraction.
    """

    blur_kernel: int = 3
    median_kernel: int = 3
    denoise_strength: float = 0.0
    threshold_method: str = "adaptive"
    threshold_value: int = 128
    adaptive_block_size: int = 11
    adaptive_c: float = 5.0
    morph_open_kernel: int = 3
    morph_close_kernel: int = 3
    background_subtraction: bool = False


class Preprocessor:
    """Image preprocessing pipeline.

    Transforms raw camera frames into binary masks suitable for
    contour-based detection.
    """

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self._config = config or PreprocessConfig()
        self._background: NDArray[np.float64] | None = None
        self._frame_count = 0

    def process(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Run the full preprocessing pipeline.

        Args:
            frame: Input frame (H, W) uint8 grayscale or (H, W, 3) BGR.

        Returns:
            Binary mask (H, W) uint8 where 255 = foreground.
        """
        cfg = self._config

        # Ensure grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Background subtraction
        if cfg.background_subtraction:
            gray = self._subtract_background(gray)

        # Median filter (removes Salt & Pepper impulse noise while preserving spot edges)
        if cfg.median_kernel > 0:
            ksize = cfg.median_kernel | 1  # ensure odd
            gray = cv2.medianBlur(gray, ksize)

        # Gaussian blur
        if cfg.blur_kernel > 0:
            ksize = cfg.blur_kernel | 1  # ensure odd
            gray = cv2.GaussianBlur(gray, (ksize, ksize), 0)

        # Non-local means denoising
        if cfg.denoise_strength > 0:
            gray = cv2.fastNlMeansDenoising(
                gray, None, cfg.denoise_strength, 7, 21,
            )

        # Thresholding
        if cfg.threshold_method == "adaptive":
            block = cfg.adaptive_block_size | 1  # ensure odd
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                block,
                -cfg.adaptive_c,  # negative C → selects bright spots
            )
        elif cfg.threshold_method in ("snr", "adaptive_snr"):
            mean_val = float(np.mean(gray))
            std_val = float(np.std(gray))
            thresh_val = max(50.0, min(235.0, mean_val + 3.2 * std_val))
            _, binary = cv2.threshold(
                gray, thresh_val, 255, cv2.THRESH_BINARY,
            )
        elif cfg.threshold_method == "otsu":
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
        else:  # global
            _, binary = cv2.threshold(
                gray, cfg.threshold_value, 255, cv2.THRESH_BINARY,
            )

        # Morphological operations
        if cfg.morph_open_kernel > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (cfg.morph_open_kernel, cfg.morph_open_kernel),
            )
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        if cfg.morph_close_kernel > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (cfg.morph_close_kernel, cfg.morph_close_kernel),
            )
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return binary

    def _subtract_background(self, frame: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Simple running-average background subtraction."""
        alpha = 0.05  # Learning rate

        if self._background is None:
            self._background = frame.astype(np.float64)
            return frame

        # Update background model
        self._background = (1 - alpha) * self._background + alpha * frame.astype(np.float64)

        # Subtract
        diff = frame.astype(np.float64) - self._background
        result = np.clip(np.abs(diff), 0, 255).astype(np.uint8)
        return result

    def reset(self) -> None:
        """Reset preprocessor state (background model)."""
        self._background = None
        self._frame_count = 0
