"""Tests for the classical beacon detector."""

import math

import cv2
import numpy as np
import pytest

from fsoc_pat.core.state import CameraState
from fsoc_pat.core.types import AngularState, Orientation, Position3D
from fsoc_pat.geometry.camera_model import CameraIntrinsics
from fsoc_pat.perception.classical_detector import ClassicalDetector, ClassicalDetectorConfig
from fsoc_pat.perception.detection_types import DetectionStatus
from fsoc_pat.perception.renderer import RenderConfig, SyntheticRenderer
from fsoc_pat.simulation.beacon import BeaconProperties


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


def make_beacon_image(
    intrinsics: CameraIntrinsics,
    camera_state: CameraState,
    target_pos: np.ndarray,
    beacon: BeaconProperties | None = None,
) -> np.ndarray:
    """Helper: render a single-target image."""
    beacon = beacon or BeaconProperties(intensity=220.0, size_px=8.0)
    config = RenderConfig(noise_std=3.0, shot_noise=False, num_stars=0,
                          enable_bloom=False, seed=42)
    renderer = SyntheticRenderer(intrinsics, config)
    return renderer.render_single_target(camera_state, target_pos, beacon)


class TestClassicalDetector:

    def test_detects_bright_beacon(
        self, intrinsics: CameraIntrinsics, camera_state: CameraState,
    ) -> None:
        """A bright beacon on boresight should be detected."""
        image = make_beacon_image(
            intrinsics, camera_state,
            np.array([100.0, 0.0, 0.0]),
            BeaconProperties(intensity=240.0, size_px=10.0),
        )
        detector = ClassicalDetector()
        result = detector.detect(image)

        assert result.detected
        assert result.best is not None
        assert result.confidence > 0.3

    def test_centroid_near_center(
        self, intrinsics: CameraIntrinsics, camera_state: CameraState,
    ) -> None:
        """Beacon on boresight should detect near image center."""
        image = make_beacon_image(
            intrinsics, camera_state,
            np.array([100.0, 0.0, 0.0]),
            BeaconProperties(intensity=240.0, size_px=10.0),
        )
        detector = ClassicalDetector()
        result = detector.detect(image)

        assert result.detected
        assert result.best is not None
        # Should be within 20px of center
        assert abs(result.best.u - 320) < 20
        assert abs(result.best.v - 240) < 20

    def test_centroid_tracks_offset(
        self, intrinsics: CameraIntrinsics, camera_state: CameraState,
    ) -> None:
        """Beacon offset to the right should detect right of center."""
        image = make_beacon_image(
            intrinsics, camera_state,
            np.array([100.0, 15.0, 0.0]),
            BeaconProperties(intensity=240.0, size_px=10.0),
        )
        detector = ClassicalDetector()
        result = detector.detect(image)

        assert result.detected
        assert result.best is not None
        assert result.best.u > 320  # Right of center

    def test_no_detection_on_empty(
        self, intrinsics: CameraIntrinsics, camera_state: CameraState,
    ) -> None:
        """Empty scene should produce no detection."""
        config = RenderConfig(noise_std=2.0, shot_noise=False, num_stars=0,
                              enable_bloom=False, seed=42)
        renderer = SyntheticRenderer(intrinsics, config)
        image = renderer.render(camera_state, [])

        detector = ClassicalDetector()
        result = detector.detect(image)

        assert not result.detected
        assert result.status == DetectionStatus.NO_DETECTION

    def test_dim_beacon_low_confidence(
        self, intrinsics: CameraIntrinsics, camera_state: CameraState,
    ) -> None:
        """A very dim beacon should have low confidence or no detection."""
        image = make_beacon_image(
            intrinsics, camera_state,
            np.array([100.0, 0.0, 0.0]),
            BeaconProperties(intensity=30.0, size_px=4.0),
        )
        detector = ClassicalDetector()
        result = detector.detect(image)

        # Either not detected or low confidence
        if result.detected:
            assert result.confidence < 0.8

    def test_processing_time_recorded(
        self, intrinsics: CameraIntrinsics, camera_state: CameraState,
    ) -> None:
        """Detection should record processing time."""
        image = make_beacon_image(
            intrinsics, camera_state, np.array([100.0, 0.0, 0.0]),
        )
        detector = ClassicalDetector()
        result = detector.detect(image)

        assert result.processing_time_ms > 0
        assert result.processing_time_ms < 100  # Should be fast

    def test_frame_index_increments(
        self, intrinsics: CameraIntrinsics, camera_state: CameraState,
    ) -> None:
        """Frame index should increment on each call."""
        image = make_beacon_image(
            intrinsics, camera_state, np.array([100.0, 0.0, 0.0]),
        )
        detector = ClassicalDetector()

        r1 = detector.detect(image)
        r2 = detector.detect(image)
        r3 = detector.detect(image)

        assert r1.frame_index == 1
        assert r2.frame_index == 2
        assert r3.frame_index == 3

    def test_synthetic_blob_detection(self) -> None:
        """Manually placed white circle on black should be detected."""
        image = np.zeros((480, 640), dtype=np.uint8)
        cv2.circle(image, (320, 240), 8, 200, -1)

        detector = ClassicalDetector()
        result = detector.detect(image)

        assert result.detected
        assert result.best is not None
        assert abs(result.best.u - 320) < 5
        assert abs(result.best.v - 240) < 5

    def test_reset_clears_frame_index(
        self, intrinsics: CameraIntrinsics, camera_state: CameraState,
    ) -> None:
        """Reset should clear frame counter."""
        image = make_beacon_image(
            intrinsics, camera_state, np.array([100.0, 0.0, 0.0]),
        )
        detector = ClassicalDetector()
        detector.detect(image)
        detector.detect(image)

        detector.reset()
        result = detector.detect(image)
        assert result.frame_index == 1
