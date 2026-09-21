"""Tests for angular error computation."""

import math

import numpy as np
import pytest

from fsoc_pat.core.types import PixelCoord
from fsoc_pat.geometry.angular_error import (
    compute_angular_error_from_pixel,
    compute_angular_separation,
)
from fsoc_pat.geometry.camera_model import CameraIntrinsics


@pytest.fixture
def intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics.from_fov(640, 480, math.radians(60.0))


class TestAngularErrorFromPixel:
    """Tests for pixel-based angular error computation."""

    def test_center_pixel_zero_error(self, intrinsics: CameraIntrinsics) -> None:
        """Center pixel (principal point) → zero angular error."""
        pixel = PixelCoord(u=intrinsics.cx, v=intrinsics.cy)
        err = compute_angular_error_from_pixel(pixel, intrinsics)
        assert err.horizontal == pytest.approx(0.0, abs=1e-10)
        assert err.vertical == pytest.approx(0.0, abs=1e-10)
        assert err.total == pytest.approx(0.0, abs=1e-10)

    def test_right_pixel_positive_horizontal(self, intrinsics: CameraIntrinsics) -> None:
        """Pixel to the right of center → positive horizontal error."""
        pixel = PixelCoord(u=intrinsics.cx + 50, v=intrinsics.cy)
        err = compute_angular_error_from_pixel(pixel, intrinsics)
        assert err.horizontal > 0
        assert abs(err.vertical) < 1e-10
        assert err.total > 0

    def test_left_pixel_negative_horizontal(self, intrinsics: CameraIntrinsics) -> None:
        """Pixel to the left → negative horizontal error."""
        pixel = PixelCoord(u=intrinsics.cx - 50, v=intrinsics.cy)
        err = compute_angular_error_from_pixel(pixel, intrinsics)
        assert err.horizontal < 0

    def test_below_pixel_positive_vertical(self, intrinsics: CameraIntrinsics) -> None:
        """Pixel below center → positive vertical error (Z-down convention)."""
        pixel = PixelCoord(u=intrinsics.cx, v=intrinsics.cy + 50)
        err = compute_angular_error_from_pixel(pixel, intrinsics)
        assert abs(err.horizontal) < 1e-10
        assert err.vertical > 0

    def test_symmetry(self, intrinsics: CameraIntrinsics) -> None:
        """Symmetric pixel offsets → same total error."""
        px_right = PixelCoord(u=intrinsics.cx + 50, v=intrinsics.cy)
        px_left = PixelCoord(u=intrinsics.cx - 50, v=intrinsics.cy)
        err_r = compute_angular_error_from_pixel(px_right, intrinsics)
        err_l = compute_angular_error_from_pixel(px_left, intrinsics)
        assert err_r.total == pytest.approx(err_l.total, abs=1e-10)

    def test_total_error_non_negative(self, intrinsics: CameraIntrinsics) -> None:
        """Total angular error is always >= 0."""
        for u in [0, 100, 320, 500, 639]:
            for v in [0, 100, 240, 400, 479]:
                err = compute_angular_error_from_pixel(
                    PixelCoord(u, v), intrinsics,
                )
                assert err.total >= 0

    def test_edge_pixel_matches_half_fov(self, intrinsics: CameraIntrinsics) -> None:
        """A pixel at the horizontal edge should give ≈ hfov/2 error."""
        # Right edge
        pixel = PixelCoord(u=float(intrinsics.width), v=intrinsics.cy)
        err = compute_angular_error_from_pixel(pixel, intrinsics)
        # Should be approximately half the horizontal FOV
        expected = intrinsics.hfov_rad / 2.0
        assert err.horizontal == pytest.approx(expected, rel=0.01)


class TestAngularSeparation:
    """Tests for vector-based angular separation."""

    def test_parallel_vectors_zero(self) -> None:
        v = np.array([1.0, 0.0, 0.0])
        assert compute_angular_separation(v, v) == pytest.approx(0.0, abs=1e-12)

    def test_perpendicular_vectors_90deg(self) -> None:
        vx = np.array([1.0, 0.0, 0.0])
        vy = np.array([0.0, 1.0, 0.0])
        assert compute_angular_separation(vx, vy) == pytest.approx(
            math.pi / 2, abs=1e-10,
        )

    def test_antiparallel_vectors_180deg(self) -> None:
        vx = np.array([1.0, 0.0, 0.0])
        vmx = np.array([-1.0, 0.0, 0.0])
        assert compute_angular_separation(vx, vmx) == pytest.approx(
            math.pi, abs=1e-10,
        )

    def test_symmetry(self) -> None:
        va = np.array([1.0, 1.0, 0.0]) / math.sqrt(2)
        vb = np.array([1.0, 0.0, 1.0]) / math.sqrt(2)
        assert compute_angular_separation(va, vb) == pytest.approx(
            compute_angular_separation(vb, va), abs=1e-12,
        )
