"""Synthetic dataset generator for AI training.

Generates labelled training data using the SyntheticRenderer with
domain randomization. Each sample consists of:
  - Rendered image (H, W) uint8
  - Label: (u, v, present) — centroid pixel + visibility flag

Domain randomization varies:
  - Target position (range, offset)
  - Beacon brightness and size
  - Background noise level
  - Number of clutter stars
  - Camera orientation jitter

This prevents the CNN from overfitting to a narrow visual domain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from fsoc_pat.core.state import CameraState
from fsoc_pat.core.types import AngularState, Orientation, Position3D
from fsoc_pat.geometry.camera_model import CameraIntrinsics
from fsoc_pat.geometry.projection import project_point_to_pixel
from fsoc_pat.perception.renderer import RenderConfig, SyntheticRenderer
from fsoc_pat.simulation.beacon import BeaconProperties, BeaconShape


@dataclass(frozen=True, slots=True)
class Sample:
    """A single training sample.

    Attributes:
        image: Rendered frame (H, W) uint8.
        u: Ground-truth centroid u (px). NaN if not visible.
        v: Ground-truth centroid v (px). NaN if not visible.
        present: Whether a beacon is visible in the image.
        metadata: Extra info (range, intensity, noise_std, etc.).
    """

    image: NDArray[np.uint8]
    u: float
    v: float
    present: bool
    metadata: dict = field(default_factory=dict)


@dataclass
class DatasetConfig:
    """Configuration for the dataset generator.

    Attributes:
        num_samples: Total samples to generate.
        fraction_negative: Fraction of samples with no beacon.
        range_min: Minimum target range (m).
        range_max: Maximum target range (m).
        offset_max: Maximum lateral offset (m).
        intensity_range: (min, max) beacon intensity.
        size_range: (min, max) beacon size (px).
        noise_range: (min, max) sensor noise std.
        star_range: (min, max) number of clutter stars.
        seed: Random seed.
    """

    num_samples: int = 1000
    fraction_negative: float = 0.15
    range_min: float = 50.0
    range_max: float = 500.0
    offset_max: float = 40.0
    intensity_range: tuple[float, float] = (100.0, 250.0)
    size_range: tuple[float, float] = (3.0, 15.0)
    noise_range: tuple[float, float] = (3.0, 20.0)
    star_range: tuple[int, int] = (0, 30)
    seed: int = 42


class DatasetGenerator:
    """Generates randomized training datasets.

    Uses domain randomization to create diverse samples that
    prevent overfitting. Splits are deterministic given the seed.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        config: DatasetConfig | None = None,
    ) -> None:
        self._intrinsics = intrinsics
        self._config = config or DatasetConfig()
        self._rng = np.random.default_rng(self._config.seed)

    def generate(self) -> list[Sample]:
        """Generate the full dataset.

        Returns:
            List of Sample objects.
        """
        cfg = self._config
        samples: list[Sample] = []

        n_negative = int(cfg.num_samples * cfg.fraction_negative)
        n_positive = cfg.num_samples - n_negative

        # Positive samples (beacon visible)
        for i in range(n_positive):
            sample = self._generate_positive(i)
            samples.append(sample)

        # Negative samples (no beacon)
        for i in range(n_negative):
            sample = self._generate_negative(n_positive + i)
            samples.append(sample)

        # Shuffle deterministically
        indices = self._rng.permutation(len(samples))
        return [samples[i] for i in indices]

    def split(
        self,
        samples: list[Sample],
        train_frac: float = 0.7,
        val_frac: float = 0.15,
    ) -> tuple[list[Sample], list[Sample], list[Sample]]:
        """Split dataset into train/val/test with no leakage.

        Args:
            samples: Full dataset.
            train_frac: Fraction for training.
            val_frac: Fraction for validation.

        Returns:
            (train, val, test) sample lists.
        """
        n = len(samples)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        train = samples[:n_train]
        val = samples[n_train:n_train + n_val]
        test = samples[n_train + n_val:]

        return train, val, test

    def _generate_positive(self, idx: int) -> Sample:
        """Generate a sample with a visible beacon."""
        cfg = self._config

        # Randomize parameters
        range_m = float(self._rng.uniform(cfg.range_min, cfg.range_max))
        y_offset = float(self._rng.uniform(-cfg.offset_max, cfg.offset_max))
        z_offset = float(self._rng.uniform(-cfg.offset_max * 0.5, cfg.offset_max * 0.5))
        intensity = float(self._rng.uniform(*cfg.intensity_range))
        size_px = float(self._rng.uniform(*cfg.size_range))
        noise_std = float(self._rng.uniform(*cfg.noise_range))
        num_stars = int(self._rng.integers(*cfg.star_range))

        # Create renderer with randomized noise
        render_config = RenderConfig(
            noise_std=noise_std,
            shot_noise=True,
            num_stars=num_stars,
            enable_bloom=self._rng.random() > 0.3,
            seed=int(self._rng.integers(0, 100000)),
        )
        renderer = SyntheticRenderer(self._intrinsics, render_config)

        # Camera at origin, looking forward
        camera_state = CameraState(
            position=Position3D(0, 0, 0),
            orientation=Orientation(),
            angular_state=AngularState(),
        )

        # Target position
        target_pos = np.array([range_m, y_offset, z_offset])
        beacon = BeaconProperties(intensity=intensity, size_px=size_px)

        # Render
        frame = renderer.render_single_target(camera_state, target_pos, beacon)

        # Ground truth pixel
        gt_pixel = project_point_to_pixel(target_pos, camera_state, self._intrinsics)

        if gt_pixel is not None:
            u, v = gt_pixel.u, gt_pixel.v
            present = True
        else:
            u, v = float("nan"), float("nan")
            present = False

        return Sample(
            image=frame,
            u=u, v=v,
            present=present,
            metadata={
                "range_m": range_m,
                "intensity": intensity,
                "size_px": size_px,
                "noise_std": noise_std,
                "num_stars": num_stars,
                "index": idx,
            },
        )

    def _generate_negative(self, idx: int) -> Sample:
        """Generate a sample with no beacon (negative example)."""
        cfg = self._config

        noise_std = float(self._rng.uniform(*cfg.noise_range))
        num_stars = int(self._rng.integers(*cfg.star_range))

        render_config = RenderConfig(
            noise_std=noise_std,
            shot_noise=True,
            num_stars=num_stars,
            enable_bloom=False,
            seed=int(self._rng.integers(0, 100000)),
        )
        renderer = SyntheticRenderer(self._intrinsics, render_config)

        camera_state = CameraState(
            position=Position3D(0, 0, 0),
            orientation=Orientation(),
            angular_state=AngularState(),
        )

        # Render empty scene
        frame = renderer.render(camera_state, [])

        return Sample(
            image=frame,
            u=float("nan"), v=float("nan"),
            present=False,
            metadata={"noise_std": noise_std, "num_stars": num_stars, "index": idx},
        )

    def save_to_disk(
        self,
        samples: list[Sample],
        output_dir: str | Path,
    ) -> None:
        """Save dataset to disk as images + CSV labels.

        Args:
            samples: List of samples.
            output_dir: Output directory.
        """
        out = Path(output_dir)
        img_dir = out / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        labels = []
        for i, sample in enumerate(samples):
            fname = f"{i:06d}.png"
            cv2.imwrite(str(img_dir / fname), sample.image)
            labels.append(f"{fname},{sample.u:.2f},{sample.v:.2f},{int(sample.present)}")

        with open(out / "labels.csv", "w") as f:
            f.write("filename,u,v,present\n")
            for line in labels:
                f.write(line + "\n")
