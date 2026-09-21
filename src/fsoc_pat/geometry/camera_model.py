"""Camera intrinsics model.

Relates the physical camera parameters (sensor size, focal length, FOV)
to the image-plane coordinate system. Supports both pinhole and optional
lens distortion models.

Key relationships (pinhole model):
    fx = width / (2 · tan(hfov / 2))
    fy = height / (2 · tan(vfov / 2))
    hfov = 2 · atan(width / (2 · fx))
    vfov = 2 · atan(height / (2 · fy))

Units:
    - Image dimensions: pixels
    - Focal length: pixels
    - FOV: radians
    - Principal point: pixels
    - Distortion coefficients: dimensionless
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from fsoc_pat.core.errors import CalibrationError, ConfigurationError


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Camera intrinsic parameters.

    Attributes:
        width: Image width (px).
        height: Image height (px).
        fx: Horizontal focal length (px).
        fy: Vertical focal length (px).
        cx: Principal point X (px). Default = width / 2.
        cy: Principal point Y (px). Default = height / 2.
        distortion_coeffs: Optional radial/tangential distortion
            coefficients (k1, k2, p1, p2, k3). Empty tuple = no distortion.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float = 0.0
    cy: float = 0.0
    distortion_coeffs: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ConfigurationError(f"Image width must be positive, got {self.width}")
        if self.height <= 0:
            raise ConfigurationError(f"Image height must be positive, got {self.height}")
        if self.fx <= 0:
            raise CalibrationError(f"Focal length fx must be positive, got {self.fx}")
        if self.fy <= 0:
            raise CalibrationError(f"Focal length fy must be positive, got {self.fy}")

        # Default principal point to image center if not explicitly set
        if self.cx == 0.0 and self.cy == 0.0:
            object.__setattr__(self, "cx", self.width / 2.0)
            object.__setattr__(self, "cy", self.height / 2.0)

    @classmethod
    def from_fov(
        cls,
        width: int,
        height: int,
        hfov_rad: float,
        vfov_rad: float | None = None,
        cx: float | None = None,
        cy: float | None = None,
        distortion_coeffs: tuple[float, ...] = (),
    ) -> CameraIntrinsics:
        """Create intrinsics from field-of-view angles.

        If vfov is not provided, it is derived from hfov assuming
        square pixels (same angular resolution in both axes).

        Args:
            width: Image width (px).
            height: Image height (px).
            hfov_rad: Horizontal field of view (rad). Must be in (0, π).
            vfov_rad: Vertical field of view (rad). If None, derived from hfov.
            cx: Principal point X (px). Default = width / 2.
            cy: Principal point Y (px). Default = height / 2.
            distortion_coeffs: Distortion coefficients.

        Returns:
            CameraIntrinsics instance.

        Raises:
            ConfigurationError: If FOV is out of valid range.
        """
        if hfov_rad <= 0 or hfov_rad >= math.pi:
            raise ConfigurationError(
                f"HFOV must be in (0, π) radians, got {hfov_rad:.4f} "
                f"({math.degrees(hfov_rad):.2f}°)"
            )

        fx = width / (2.0 * math.tan(hfov_rad / 2.0))

        if vfov_rad is not None:
            if vfov_rad <= 0 or vfov_rad >= math.pi:
                raise ConfigurationError(
                    f"VFOV must be in (0, π) radians, got {vfov_rad:.4f}"
                )
            fy = height / (2.0 * math.tan(vfov_rad / 2.0))
        else:
            # Square pixels: same angular resolution
            fy = fx

        return cls(
            width=width,
            height=height,
            fx=fx,
            fy=fy,
            cx=cx if cx is not None else 0.0,
            cy=cy if cy is not None else 0.0,
            distortion_coeffs=distortion_coeffs,
        )

    @classmethod
    def from_focal_length(
        cls,
        width: int,
        height: int,
        fx: float,
        fy: float | None = None,
        cx: float | None = None,
        cy: float | None = None,
        distortion_coeffs: tuple[float, ...] = (),
    ) -> CameraIntrinsics:
        """Create intrinsics from focal length values.

        Args:
            width: Image width (px).
            height: Image height (px).
            fx: Horizontal focal length (px).
            fy: Vertical focal length (px). If None, fy = fx (square pixels).
            cx: Principal point X (px). Default = width / 2.
            cy: Principal point Y (px). Default = height / 2.
            distortion_coeffs: Distortion coefficients.

        Returns:
            CameraIntrinsics instance.
        """
        return cls(
            width=width,
            height=height,
            fx=fx,
            fy=fy if fy is not None else fx,
            cx=cx if cx is not None else 0.0,
            cy=cy if cy is not None else 0.0,
            distortion_coeffs=distortion_coeffs,
        )

    @property
    def hfov_rad(self) -> float:
        """Horizontal field of view (rad)."""
        return 2.0 * math.atan(self.width / (2.0 * self.fx))

    @property
    def vfov_rad(self) -> float:
        """Vertical field of view (rad)."""
        return 2.0 * math.atan(self.height / (2.0 * self.fy))

    @property
    def hfov_deg(self) -> float:
        """Horizontal field of view (degrees)."""
        return math.degrees(self.hfov_rad)

    @property
    def vfov_deg(self) -> float:
        """Vertical field of view (degrees)."""
        return math.degrees(self.vfov_rad)

    @property
    def intrinsic_matrix(self) -> NDArray[np.float64]:
        """3×3 camera intrinsic matrix K.

        K = [[fx, 0, cx],
             [0, fy, cy],
             [0,  0,  1]]
        """
        return np.array([
            [self.fx,    0.0, self.cx],
            [   0.0, self.fy, self.cy],
            [   0.0,    0.0,     1.0],
        ], dtype=np.float64)

    @property
    def has_distortion(self) -> bool:
        """Whether distortion coefficients are defined."""
        return len(self.distortion_coeffs) > 0 and any(
            c != 0.0 for c in self.distortion_coeffs
        )

    @property
    def pixel_angular_resolution(self) -> float:
        """Average angular resolution per pixel (rad/px).

        Design assumption: This is an average across the sensor.
        """
        return (self.hfov_rad / self.width + self.vfov_rad / self.height) / 2.0
