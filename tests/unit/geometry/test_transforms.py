"""Tests for coordinate frame transformations."""

import math

import numpy as np
import pytest

from fsoc_pat.core.state import CameraState
from fsoc_pat.core.types import AngularState, Orientation, Position3D
from fsoc_pat.geometry.transforms import (
    boresight_in_world,
    camera_to_world,
    world_to_camera,
    world_to_camera_from_state,
    camera_to_world_from_state,
)


class TestWorldCameraTransform:
    """Tests for world ↔ camera frame transformations."""

    def test_identity_transform(self) -> None:
        """No rotation, no offset → point unchanged."""
        pos = Position3D(0.0, 0.0, 0.0)
        ori = Orientation(0.0, 0.0, 0.0)
        ang = AngularState(0.0, 0.0, 0.0, 0.0)
        point = np.array([10.0, 5.0, 3.0])

        result = world_to_camera(point, pos, ori, ang)
        np.testing.assert_allclose(result, point, atol=1e-10)

    def test_pure_translation(self) -> None:
        """Camera offset, no rotation → shifted point."""
        pos = Position3D(10.0, 0.0, 0.0)
        ori = Orientation(0.0, 0.0, 0.0)
        ang = AngularState(0.0, 0.0, 0.0, 0.0)
        point = np.array([20.0, 0.0, 0.0])

        result = world_to_camera(point, pos, ori, ang)
        np.testing.assert_allclose(result, [10.0, 0.0, 0.0], atol=1e-10)

    def test_round_trip(self) -> None:
        """world → camera → world should recover the original point."""
        pos = Position3D(5.0, 3.0, 1.0)
        ori = Orientation(0.1, 0.2, 0.3)
        ang = AngularState(0.05, -0.1, 0.0, 0.0)
        point = np.array([100.0, 20.0, -5.0])

        cam_point = world_to_camera(point, pos, ori, ang)
        recovered = camera_to_world(cam_point, pos, ori, ang)
        np.testing.assert_allclose(recovered, point, atol=1e-8)

    def test_round_trip_with_state(self) -> None:
        """Same round-trip using CameraState convenience functions."""
        state = CameraState(
            position=Position3D(5.0, 3.0, 1.0),
            orientation=Orientation(0.1, 0.2, 0.3),
            angular_state=AngularState(0.05, -0.1, 0.0, 0.0),
        )
        point = np.array([50.0, -10.0, 7.0])

        cam_point = world_to_camera_from_state(point, state)
        recovered = camera_to_world_from_state(cam_point, state)
        np.testing.assert_allclose(recovered, point, atol=1e-8)

    def test_pan_rotates_boresight(self) -> None:
        """Panning should rotate the boresight in the horizontal plane."""
        ori = Orientation(0.0, 0.0, 0.0)

        # No pan → boresight along +X
        bs0 = boresight_in_world(ori, AngularState(0.0, 0.0))
        np.testing.assert_allclose(bs0, [1.0, 0.0, 0.0], atol=1e-10)

        # Pan right (positive yaw about Z) → boresight rotates toward +Y
        bs_pan = boresight_in_world(ori, AngularState(pan=math.pi / 4, tilt=0.0))
        assert bs_pan[1] > 0  # Y component positive (turned right)
        assert bs_pan[0] > 0  # Still has forward component

    def test_tilt_rotates_boresight(self) -> None:
        """Tilting should rotate the boresight in the vertical plane."""
        ori = Orientation(0.0, 0.0, 0.0)

        # Tilt down (positive pitch about Y) → boresight rotates toward +Z (down)
        bs_tilt = boresight_in_world(ori, AngularState(pan=0.0, tilt=math.pi / 6))
        # With tilt as pitch about Y, positive tilt → Z component changes
        # The exact direction depends on convention, but it should deviate from [1,0,0]
        assert abs(bs_tilt[0]) < 1.0  # No longer purely forward

    def test_boresight_is_unit_vector(self) -> None:
        """Boresight must always be a unit vector."""
        for pan in [0, 0.3, -0.5, 1.0]:
            for tilt in [0, 0.2, -0.3]:
                bs = boresight_in_world(
                    Orientation(), AngularState(pan=pan, tilt=tilt),
                )
                assert np.linalg.norm(bs) == pytest.approx(1.0, abs=1e-10)
