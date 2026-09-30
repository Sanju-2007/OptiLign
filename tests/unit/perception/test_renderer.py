"""Tests for the synthetic image renderer."""

import math

import numpy as np
import pytest

from fsoc_pat.core.state import CameraState
from fsoc_pat.core.types import AngularState, Orientation, Position3D
from fsoc_pat.geometry.camera_model import CameraIntrinsics
from fsoc_pat.perception.renderer import RenderConfig, SyntheticRenderer
from fsoc_pat.simulation.beacon import BeaconProperties, BeaconShape


@pytest.fixture
def intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics.from_fov(640, 480, math.radians(60))


@pytest.fixture
def camera_state() -> CameraState:
    return CameraState(
        position=Position3D(0, 0, 0),
        orientation=Orientation(),
        angular_state=AngularState(),
    )


@pytest.fixture
def renderer(intrinsics: CameraIntrinsics) -> SyntheticRenderer:
    config = RenderConfig(
        noise_std=5.0,
        shot_noise=False,
        num_stars=5,
        enable_bloom=False,
        seed=42,
    )
    return SyntheticRenderer(intrinsics, config)


class TestSyntheticRenderer:

    def test_output_shape(
        self, renderer: SyntheticRenderer, camera_state: CameraState,
    ) -> None:
        """Output should be (H, W) uint8."""
        frame = renderer.render(camera_state, [])
        assert frame.shape == (480, 640)
        assert frame.dtype == np.uint8

    def test_empty_scene_is_dark(
        self, renderer: SyntheticRenderer, camera_state: CameraState,
    ) -> None:
        """Frame with no targets should be mostly dark."""
        frame = renderer.render(camera_state, [])
        assert frame.mean() < 50  # Dark background

    def test_beacon_adds_brightness(
        self, renderer: SyntheticRenderer, camera_state: CameraState,
    ) -> None:
        """A bright beacon should raise the frame's max intensity."""
        beacon = BeaconProperties(intensity=220.0, size_px=8.0)
        target_pos = np.array([100.0, 0.0, 0.0])

        frame_empty = renderer.render(camera_state, [])
        frame_beacon = renderer.render(camera_state, [(target_pos, beacon)])

        # Frame with beacon should be brighter overall
        assert frame_beacon.max() > frame_empty.max() + 20

    def test_beacon_on_boresight(
        self, renderer: SyntheticRenderer, camera_state: CameraState,
    ) -> None:
        """Beacon on boresight should be brightest near image center."""
        beacon = BeaconProperties(intensity=250.0, size_px=10.0)
        target_pos = np.array([100.0, 0.0, 0.0])

        frame = renderer.render(camera_state, [(target_pos, beacon)])

        # Center region should be brighter than edges
        center = frame[220:260, 300:340]
        edge = frame[0:40, 0:40]
        assert center.mean() > edge.mean()

    def test_beacon_off_center(
        self, renderer: SyntheticRenderer, camera_state: CameraState,
    ) -> None:
        """Beacon offset right should produce brightness on right side."""
        beacon = BeaconProperties(intensity=250.0, size_px=8.0)
        # Target to the right (positive Y)
        target_pos = np.array([100.0, 20.0, 0.0])

        frame = renderer.render(camera_state, [(target_pos, beacon)])

        # Right half should be brighter than left
        left = frame[:, :320].mean()
        right = frame[:, 320:].mean()
        assert right > left

    def test_deterministic_with_same_seed(
        self, intrinsics: CameraIntrinsics, camera_state: CameraState,
    ) -> None:
        """Same seed → identical frames."""
        config = RenderConfig(seed=99, noise_std=5.0, shot_noise=False)
        r1 = SyntheticRenderer(intrinsics, config)
        r2 = SyntheticRenderer(intrinsics, config)

        beacon = BeaconProperties(intensity=200.0, size_px=6.0)
        target = np.array([100.0, 5.0, 0.0])

        f1 = r1.render(camera_state, [(target, beacon)])
        f2 = r2.render(camera_state, [(target, beacon)])

        np.testing.assert_array_equal(f1, f2)

    def test_behind_camera_no_beacon(
        self, renderer: SyntheticRenderer, camera_state: CameraState,
    ) -> None:
        """Target behind camera should not appear in frame."""
        beacon = BeaconProperties(intensity=250.0, size_px=10.0)
        target_behind = np.array([-100.0, 0.0, 0.0])

        frame_behind = renderer.render(camera_state, [(target_behind, beacon)])
        frame_empty = renderer.render(camera_state, [])

        # Should be similar brightness (no beacon rendered)
        assert abs(float(frame_behind.max()) - float(frame_empty.max())) < 30

    def test_multiple_targets(
        self, renderer: SyntheticRenderer, camera_state: CameraState,
    ) -> None:
        """Multiple beacons should all appear."""
        beacon = BeaconProperties(intensity=220.0, size_px=6.0)
        targets = [
            (np.array([100.0, 10.0, 0.0]), beacon),
            (np.array([100.0, -10.0, 0.0]), beacon),
        ]
        frame = renderer.render(camera_state, targets)

        # Both sides should show brightness
        left_quarter = frame[:, 200:300].max()
        right_quarter = frame[:, 340:440].max()
        assert left_quarter > 50
        assert right_quarter > 50

    def test_beacon_shapes(
        self, renderer: SyntheticRenderer, camera_state: CameraState,
    ) -> None:
        """All beacon shapes should render without error."""
        target = np.array([100.0, 0.0, 0.0])
        for shape in BeaconShape:
            beacon = BeaconProperties(intensity=200.0, size_px=6.0, shape=shape)
            frame = renderer.render(camera_state, [(target, beacon)])
            assert frame.shape == (480, 640)

    def test_salt_pepper_noise_injection(
        self, intrinsics: CameraIntrinsics, camera_state: CameraState,
    ) -> None:
        """Salt & Pepper noise should create hot (255) and dead (0) pixels around configured ratio."""
        config = RenderConfig(
            enable_gaussian_noise=False,
            noise_std=0.0,
            shot_noise=False,
            enable_poisson_noise=False,
            enable_salt_pepper_noise=True,
            salt_pepper_ratio=0.10,
            background_mean=128.0,
            background_std=0.0,
            num_stars=0,
            seed=42,
        )
        renderer = SyntheticRenderer(intrinsics, config)
        frame = renderer.render(camera_state, [])
        # In a constant 128 background, pixels with salt=255 and pepper=0 indicate impulse noise
        salt_pixels = np.count_nonzero(frame == 255)
        pepper_pixels = np.count_nonzero(frame == 0)
        total_pixels = frame.size
        sp_ratio = (salt_pixels + pepper_pixels) / total_pixels
        assert pytest.approx(sp_ratio, abs=0.02) == 0.10

    def test_multi_noise_matrix_independent_toggles(
        self, intrinsics: CameraIntrinsics, camera_state: CameraState,
    ) -> None:
        """Gaussian, Poisson, and Salt & Pepper can be toggled independently."""
        # 1. Only S&P
        cfg_sp = RenderConfig(
            enable_gaussian_noise=False, noise_std=0.0,
            enable_poisson_noise=False, shot_noise=False,
            enable_salt_pepper_noise=True, salt_pepper_ratio=0.05,
            seed=1,
        )
        r_sp = SyntheticRenderer(intrinsics, cfg_sp)
        f_sp = r_sp.render(camera_state, [])
        assert np.any(f_sp == 255)

        # 2. All 3 combined
        cfg_all = RenderConfig(
            enable_gaussian_noise=True, noise_std=15.0,
            enable_poisson_noise=True, shot_noise=True,
            enable_salt_pepper_noise=True, salt_pepper_ratio=0.10,
            seed=2,
        )
        r_all = SyntheticRenderer(intrinsics, cfg_all)
        f_all = r_all.render(camera_state, [])
        assert f_all.shape == (480, 640)

