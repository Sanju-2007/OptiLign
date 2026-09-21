"""Tests for 3D-to-pixel projection and pixel-to-ray inverse."""

import math

import numpy as np
import pytest

from fsoc_pat.core.state import CameraState
from fsoc_pat.core.types import AngularState, Orientation, PixelCoord, Position3D
from fsoc_pat.geometry.camera_model import CameraIntrinsics
from fsoc_pat.geometry.projection import (
    is_in_fov,
    is_point_visible,
    pixel_to_ray,
    project_point_to_pixel,
)


@pytest.fixture
def intrinsics() -> CameraIntrinsics:
    """Standard test camera: 640×480, 60° HFOV."""
    return CameraIntrinsics.from_fov(640, 480, math.radians(60.0))


@pytest.fixture
def camera_at_origin() -> CameraState:
    """Camera at origin, looking along +X (no rotation)."""
    return CameraState(
        position=Position3D(0.0, 0.0, 0.0),
        orientation=Orientation(0.0, 0.0, 0.0),
        angular_state=AngularState(0.0, 0.0, 0.0, 0.0),
    )


class TestProjection:
    """Tests for forward projection (3D → pixel)."""

    def test_target_on_boresight_projects_to_center(
        self, intrinsics: CameraIntrinsics, camera_at_origin: CameraState,
    ) -> None:
        """A target directly ahead on the optical axis → image center."""
        target = np.array([100.0, 0.0, 0.0])  # 100m ahead on +X
        pixel = project_point_to_pixel(target, camera_at_origin, intrinsics)
        assert pixel is not None
        assert pixel.u == pytest.approx(intrinsics.cx, abs=0.5)
        assert pixel.v == pytest.approx(intrinsics.cy, abs=0.5)

    def test_target_right_projects_right(
        self, intrinsics: CameraIntrinsics, camera_at_origin: CameraState,
    ) -> None:
        """A target to the right (+Y) should project to u > cx."""
        target = np.array([100.0, 10.0, 0.0])  # ahead and right
        pixel = project_point_to_pixel(target, camera_at_origin, intrinsics)
        assert pixel is not None
        assert pixel.u > intrinsics.cx

    def test_target_below_projects_down(
        self, intrinsics: CameraIntrinsics, camera_at_origin: CameraState,
    ) -> None:
        """A target below (+Z in our convention) should project to v > cy."""
        target = np.array([100.0, 0.0, 10.0])  # ahead and down
        pixel = project_point_to_pixel(target, camera_at_origin, intrinsics)
        assert pixel is not None
        assert pixel.v > intrinsics.cy

    def test_target_behind_camera_returns_none(
        self, intrinsics: CameraIntrinsics, camera_at_origin: CameraState,
    ) -> None:
        """A target behind the camera (negative X) → None."""
        target = np.array([-100.0, 0.0, 0.0])
        pixel = project_point_to_pixel(target, camera_at_origin, intrinsics)
        assert pixel is None

    def test_target_exactly_at_camera_returns_none(
        self, intrinsics: CameraIntrinsics, camera_at_origin: CameraState,
    ) -> None:
        """A target at camera position → None (z=0 in camera frame)."""
        target = np.array([0.0, 0.0, 0.0])
        pixel = project_point_to_pixel(target, camera_at_origin, intrinsics)
        assert pixel is None


class TestInversProjection:
    """Tests for pixel → ray."""

    def test_center_pixel_ray_is_boresight(
        self, intrinsics: CameraIntrinsics,
    ) -> None:
        """Center pixel should produce a ray along +X (boresight)."""
        center = PixelCoord(u=intrinsics.cx, v=intrinsics.cy)
        ray = pixel_to_ray(center, intrinsics)
        # Ray should be approximately [1, 0, 0]
        assert ray[0] == pytest.approx(1.0, abs=1e-6)
        assert abs(ray[1]) < 1e-6
        assert abs(ray[2]) < 1e-6

    def test_ray_is_unit_vector(self, intrinsics: CameraIntrinsics) -> None:
        """All rays should be unit vectors."""
        for u, v in [(0, 0), (320, 240), (639, 479), (100, 300)]:
            ray = pixel_to_ray(PixelCoord(u, v), intrinsics)
            assert np.linalg.norm(ray) == pytest.approx(1.0, abs=1e-10)

    def test_right_pixel_ray_points_right(self, intrinsics: CameraIntrinsics) -> None:
        """Pixel to the right of center → ray with positive Y."""
        pixel = PixelCoord(u=intrinsics.cx + 100, v=intrinsics.cy)
        ray = pixel_to_ray(pixel, intrinsics)
        assert ray[1] > 0  # Y = right


class TestFOVCheck:
    """Tests for FOV boundary checks."""

    def test_center_is_in_fov(self, intrinsics: CameraIntrinsics) -> None:
        assert is_in_fov(PixelCoord(320, 240), intrinsics)

    def test_corner_is_in_fov(self, intrinsics: CameraIntrinsics) -> None:
        assert is_in_fov(PixelCoord(0, 0), intrinsics)

    def test_outside_is_not_in_fov(self, intrinsics: CameraIntrinsics) -> None:
        assert not is_in_fov(PixelCoord(-1, 240), intrinsics)
        assert not is_in_fov(PixelCoord(640, 240), intrinsics)
        assert not is_in_fov(PixelCoord(320, 480), intrinsics)

    def test_point_visibility(
        self, intrinsics: CameraIntrinsics, camera_at_origin: CameraState,
    ) -> None:
        # On boresight, visible
        assert is_point_visible(
            np.array([100.0, 0.0, 0.0]), camera_at_origin, intrinsics,
        )
        # Behind camera, not visible
        assert not is_point_visible(
            np.array([-100.0, 0.0, 0.0]), camera_at_origin, intrinsics,
        )
