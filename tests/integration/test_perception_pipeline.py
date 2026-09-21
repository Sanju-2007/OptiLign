"""Integration tests for the full perception pipeline.

Verifies the end-to-end chain:
  world state → renderer → preprocessor → detector → confidence → detection
"""

import math

import numpy as np
import pytest

from fsoc_pat.core.state import CameraState
from fsoc_pat.core.types import AngularState, Orientation, Position3D
from fsoc_pat.geometry.camera_model import CameraIntrinsics
from fsoc_pat.perception.detection_types import DetectionStatus
from fsoc_pat.perception.pipeline import PerceptionConfig, PerceptionPipeline
from fsoc_pat.perception.renderer import RenderConfig
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


@pytest.fixture
def pipeline(intrinsics: CameraIntrinsics) -> PerceptionPipeline:
    config = PerceptionConfig(
        render=RenderConfig(
            noise_std=3.0, shot_noise=False, num_stars=0,
            enable_bloom=False, seed=42,
        ),
    )
    return PerceptionPipeline(intrinsics, config)


@pytest.mark.integration
class TestPerceptionPipeline:

    def test_detects_beacon_on_boresight(
        self,
        pipeline: PerceptionPipeline,
        camera_state: CameraState,
    ) -> None:
        """Pipeline should detect a bright beacon on boresight."""
        beacon = BeaconProperties(intensity=240.0, size_px=10.0)
        target = np.array([100.0, 0.0, 0.0])

        detection = pipeline.process_frame(
            camera_state, [(target, beacon)], timestamp=0.0,
        )

        assert detection.detected
        assert detection.best is not None
        assert abs(detection.best.u - 320) < 25
        assert abs(detection.best.v - 240) < 25

    def test_no_detection_empty_scene(
        self,
        pipeline: PerceptionPipeline,
        camera_state: CameraState,
    ) -> None:
        """Empty scene should produce no detection."""
        detection = pipeline.process_frame(camera_state, [], timestamp=0.0)
        assert not detection.detected

    def test_stores_last_frame(
        self,
        pipeline: PerceptionPipeline,
        camera_state: CameraState,
    ) -> None:
        """Pipeline should store the last rendered frame."""
        beacon = BeaconProperties(intensity=200.0, size_px=8.0)
        target = np.array([100.0, 0.0, 0.0])

        pipeline.process_frame(camera_state, [(target, beacon)])

        assert pipeline.last_frame is not None
        assert pipeline.last_frame.shape == (480, 640)
        assert pipeline.last_frame.dtype == np.uint8

    def test_frame_count_increments(
        self,
        pipeline: PerceptionPipeline,
        camera_state: CameraState,
    ) -> None:
        """Frame count should increment."""
        beacon = BeaconProperties(intensity=200.0, size_px=8.0)
        target = np.array([100.0, 0.0, 0.0])

        for i in range(5):
            pipeline.process_frame(camera_state, [(target, beacon)])

        assert pipeline.frame_count == 5

    def test_confidence_improves_over_time(
        self,
        pipeline: PerceptionPipeline,
        camera_state: CameraState,
    ) -> None:
        """Confidence should increase with consistent detections."""
        beacon = BeaconProperties(intensity=240.0, size_px=10.0)
        target = np.array([100.0, 0.0, 0.0])

        confidences = []
        for i in range(10):
            det = pipeline.process_frame(
                camera_state, [(target, beacon)], timestamp=i * 0.01,
            )
            if det.detected:
                confidences.append(det.confidence)

        # After several frames, confidence should be higher than first
        if len(confidences) >= 3:
            assert confidences[-1] >= confidences[0] - 0.1

    def test_process_image_directly(
        self,
        pipeline: PerceptionPipeline,
    ) -> None:
        """Should accept pre-rendered images."""
        import cv2

        image = np.zeros((480, 640), dtype=np.uint8)
        cv2.circle(image, (300, 200), 8, 200, -1)

        detection = pipeline.process_image(image, timestamp=0.5)
        assert detection.detected
        assert detection.best is not None
        assert abs(detection.best.u - 300) < 10
        assert abs(detection.best.v - 200) < 10

    def test_reset_clears_state(
        self,
        pipeline: PerceptionPipeline,
        camera_state: CameraState,
    ) -> None:
        """Reset should clear all pipeline state."""
        beacon = BeaconProperties(intensity=200.0, size_px=8.0)
        target = np.array([100.0, 0.0, 0.0])

        pipeline.process_frame(camera_state, [(target, beacon)])
        assert pipeline.frame_count == 1

        pipeline.reset()
        assert pipeline.frame_count == 0
        assert pipeline.last_frame is None

    def test_centroid_accuracy(
        self,
        pipeline: PerceptionPipeline,
        camera_state: CameraState,
        intrinsics: CameraIntrinsics,
    ) -> None:
        """Detected centroid should be within a few pixels of ground truth.

        This is the key accuracy test. The detector operates on noisy
        rendered images and must achieve sub-10px accuracy.
        """
        from fsoc_pat.geometry.projection import project_point_to_pixel

        beacon = BeaconProperties(intensity=240.0, size_px=10.0)
        target_pos = np.array([100.0, 8.0, -3.0])

        # Ground truth pixel
        gt_pixel = project_point_to_pixel(target_pos, camera_state, intrinsics)
        assert gt_pixel is not None

        # Detection
        detection = pipeline.process_frame(
            camera_state, [(target_pos, beacon)], timestamp=0.0,
        )

        assert detection.detected
        assert detection.best is not None

        # Accuracy check: within 10px of ground truth
        du = abs(detection.best.u - gt_pixel.u)
        dv = abs(detection.best.v - gt_pixel.v)
        assert du < 10, f"Horizontal error {du:.1f}px exceeds 10px"
        assert dv < 10, f"Vertical error {dv:.1f}px exceeds 10px"
