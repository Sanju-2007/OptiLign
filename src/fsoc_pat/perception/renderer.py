"""Synthetic image renderer.

Generates realistic virtual camera frames by compositing:
  1. Background (dark space / gradient / texture)
  2. Stars / clutter (random bright points)
  3. Beacon target (Gaussian spot at projected pixel)
  4. Sensor noise (Gaussian + shot noise)

This is a PURE RENDERING module — it reads world state and produces
images. It does NOT perform any detection or estimation.

Output: uint8 grayscale images (H, W).

Units:
    Pixel coordinates: px
    Intensity: 0-255
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from fsoc_pat.core.state import CameraState
from fsoc_pat.geometry.camera_model import CameraIntrinsics
from fsoc_pat.geometry.projection import is_in_fov, project_point_to_pixel
from fsoc_pat.simulation.beacon import BeaconProperties, BeaconShape


@dataclass
class RenderConfig:
    """Configuration for the synthetic image renderer.

    Attributes:
        background_mean: Mean background brightness (0-255).
        background_std: Background spatial variation std.
        noise_std: Gaussian sensor noise std.
        shot_noise: Whether to apply Poisson shot noise.
        num_stars: Number of random star/clutter points.
        star_intensity_range: (min, max) intensity for stars.
        star_size_range: (min, max) pixel radius for stars.
        enable_bloom: Add bloom/glow around bright sources.
        bloom_radius: Bloom kernel radius (px).
        seed: Random seed for reproducibility.
    """

    background_mean: float = 15.0
    background_std: float = 5.0
    noise_std: float = 10.0
    shot_noise: bool = True
    num_stars: int = 20
    star_intensity_range: tuple[float, float] = (40.0, 120.0)
    star_size_range: tuple[float, float] = (1.0, 2.0)
    enable_bloom: bool = True
    bloom_radius: int = 5
    seed: int = 42


class SyntheticRenderer:
    """Generates synthetic camera frames.

    Creates realistic images of a beacon target against a dark
    background with stars, noise, and optional bloom effects.

    Usage:
        renderer = SyntheticRenderer(intrinsics, config)
        frame = renderer.render(camera_state, targets)
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        config: RenderConfig | None = None,
    ) -> None:
        self._intrinsics = intrinsics
        self._config = config or RenderConfig()
        self._rng = np.random.default_rng(self._config.seed)

        # Pre-generate star positions (fixed for a given seed)
        self._star_positions: list[tuple[int, int, float, float]] = []
        self._generate_stars()

    def _generate_stars(self) -> None:
        """Generate random star positions and properties."""
        cfg = self._config
        W, H = self._intrinsics.width, self._intrinsics.height

        for _ in range(cfg.num_stars):
            x = int(self._rng.integers(0, W))
            y = int(self._rng.integers(0, H))
            intensity = float(self._rng.uniform(*cfg.star_intensity_range))
            radius = float(self._rng.uniform(*cfg.star_size_range))
            self._star_positions.append((x, y, intensity, radius))

    def render(
        self,
        camera_state: CameraState,
        target_positions: list[tuple[NDArray[np.float64], BeaconProperties]],
    ) -> NDArray[np.uint8]:
        """Render a synthetic camera frame.

        Args:
            camera_state: Current camera state.
            target_positions: List of (world_position_3d, beacon_props) tuples.

        Returns:
            Grayscale image as (H, W) uint8 array.
        """
        H = self._intrinsics.height
        W = self._intrinsics.width
        cfg = self._config

        # 1. Background
        frame = self._render_background(H, W)

        # 2. Stars / clutter
        self._render_stars(frame)

        # 3. Beacon targets
        for world_pos, beacon in target_positions:
            self._render_beacon(frame, camera_state, world_pos, beacon)

        # 4. Bloom effect
        if cfg.enable_bloom:
            frame = self._apply_bloom(frame)

        # 5. Sensor noise
        frame = self._apply_noise(frame)

        return frame

    def render_single_target(
        self,
        camera_state: CameraState,
        target_world_pos: NDArray[np.float64],
        beacon: BeaconProperties,
    ) -> NDArray[np.uint8]:
        """Convenience: render a frame with a single target.

        Args:
            camera_state: Camera state.
            target_world_pos: Target position in world frame (3,).
            beacon: Beacon properties.

        Returns:
            Grayscale image (H, W) uint8.
        """
        return self.render(camera_state, [(target_world_pos, beacon)])

    def _render_background(self, H: int, W: int) -> NDArray[np.float64]:
        """Create background with spatial variation."""
        cfg = self._config
        bg = self._rng.normal(cfg.background_mean, cfg.background_std, (H, W))
        return np.clip(bg, 0, 255)

    def _render_stars(self, frame: NDArray[np.float64]) -> None:
        """Draw random star points on the frame."""
        for x, y, intensity, radius in self._star_positions:
            if radius < 1.5:
                # Sub-pixel star: just brighten the pixel
                if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
                    frame[y, x] = min(frame[y, x] + intensity, 255.0)
            else:
                cv2.circle(
                    frame,
                    (x, y),
                    int(radius),
                    float(intensity),
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )

    def _render_beacon(
        self,
        frame: NDArray[np.float64],
        camera_state: CameraState,
        world_pos: NDArray[np.float64],
        beacon: BeaconProperties,
    ) -> None:
        """Draw a beacon on the frame at its projected pixel position."""
        pixel = project_point_to_pixel(world_pos, camera_state, self._intrinsics)
        if pixel is None:
            return

        u, v = pixel.u, pixel.v
        H, W = frame.shape[:2]

        # Allow rendering slightly outside FOV for partial visibility
        margin = beacon.size_px * 3
        if u < -margin or u > W + margin or v < -margin or v > H + margin:
            return

        # Draw beacon based on shape
        if beacon.shape == BeaconShape.POINT:
            self._draw_gaussian_spot(frame, u, v, max(1.0, beacon.size_px / 2),
                                     beacon.intensity)

        elif beacon.shape == BeaconShape.CIRCLE:
            self._draw_gaussian_spot(frame, u, v, beacon.size_px / 2,
                                     beacon.intensity)

        elif beacon.shape == BeaconShape.SQUARE:
            half = int(beacon.size_px / 2)
            x1 = max(0, int(u) - half)
            y1 = max(0, int(v) - half)
            x2 = min(W, int(u) + half)
            y2 = min(H, int(v) + half)
            frame[y1:y2, x1:x2] = np.clip(
                frame[y1:y2, x1:x2] + beacon.intensity, 0, 255,
            )

        elif beacon.shape == BeaconShape.RING:
            cv2.circle(
                frame,
                (int(u), int(v)),
                int(beacon.size_px / 2),
                float(beacon.intensity),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    def _draw_gaussian_spot(
        self,
        frame: NDArray[np.float64],
        cx: float,
        cy: float,
        sigma: float,
        intensity: float,
    ) -> None:
        """Draw a Gaussian spot centered at (cx, cy).

        Creates a realistic beacon appearance with smooth falloff.
        """
        H, W = frame.shape[:2]

        # Compute patch bounds (3σ radius)
        radius = int(math.ceil(3 * sigma))
        x_min = max(0, int(cx) - radius)
        x_max = min(W, int(cx) + radius + 1)
        y_min = max(0, int(cy) - radius)
        y_max = min(H, int(cy) + radius + 1)

        if x_min >= x_max or y_min >= y_max:
            return

        # Create coordinate grids
        ys, xs = np.mgrid[y_min:y_max, x_min:x_max]
        xs = xs.astype(np.float64)
        ys = ys.astype(np.float64)

        # Gaussian kernel
        dx = xs - cx
        dy = ys - cy
        kernel = intensity * np.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma))

        # Add to frame
        frame[y_min:y_max, x_min:x_max] = np.clip(
            frame[y_min:y_max, x_min:x_max] + kernel, 0, 255,
        )

    def _apply_bloom(self, frame: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply bloom/glow effect around bright pixels."""
        cfg = self._config

        # Extract bright regions
        bright_mask = (frame > 150).astype(np.float64)
        bright_only = frame * bright_mask

        # Blur the bright regions
        ksize = cfg.bloom_radius * 2 + 1
        bloom = cv2.GaussianBlur(bright_only, (ksize, ksize), cfg.bloom_radius / 2)

        # Add bloom at reduced intensity
        result = frame + bloom * 0.3
        return np.clip(result, 0, 255)

    def _apply_noise(self, frame: NDArray[np.float64]) -> NDArray[np.uint8]:
        """Apply sensor noise and convert to uint8.

        Adds:
          - Gaussian read noise
          - Poisson shot noise (if enabled)
        """
        cfg = self._config

        # Gaussian read noise
        if cfg.noise_std > 0:
            noise = self._rng.normal(0, cfg.noise_std, frame.shape)
            frame = frame + noise

        # Poisson shot noise
        if cfg.shot_noise:
            frame_clipped = np.clip(frame, 0, 255)
            # Scale to reasonable photon counts
            photons = frame_clipped * 0.5
            noisy = self._rng.poisson(np.maximum(photons, 0))
            frame = frame_clipped + (noisy - photons) * 0.5

        return np.clip(frame, 0, 255).astype(np.uint8)
