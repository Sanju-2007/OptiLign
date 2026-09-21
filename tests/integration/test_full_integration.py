"""Full system integration test — Phases 1 + 2 + 3.

Verifies the complete chain:
  Phase 1: trajectory → target state → camera → projection → angular error
  Phase 2: renderer → preprocessor → classical detector → confidence
  Phase 3: AI model → hybrid detector → fusion

This test runs a simulated scenario and validates that:
  1. The simulation engine produces correct ground truth
  2. The perception pipeline detects the beacon
  3. Detection accuracy matches ground truth within tolerance
  4. AI model architecture is valid and can run inference
  5. Hybrid fusion produces results
"""

import math

import cv2
import numpy as np
import pytest
import torch

from fsoc_pat.ai.ai_detector import AIDetector
from fsoc_pat.ai.dataset import DatasetConfig, DatasetGenerator
from fsoc_pat.ai.hybrid_detector import HybridDetector, HybridConfig
from fsoc_pat.ai.model import BeaconNet
from fsoc_pat.config.models import CameraConfig, ExperimentConfig, ScenarioConfig, TargetConfig
from fsoc_pat.core.state import CameraState
from fsoc_pat.core.types import AngularState, Orientation, Position3D
from fsoc_pat.geometry.camera_model import CameraIntrinsics
from fsoc_pat.geometry.projection import project_point_to_pixel
from fsoc_pat.perception.classical_detector import ClassicalDetector, ClassicalDetectorConfig
from fsoc_pat.perception.pipeline import PerceptionConfig, PerceptionPipeline
from fsoc_pat.perception.preprocessing import PreprocessConfig
from fsoc_pat.perception.renderer import RenderConfig, SyntheticRenderer
from fsoc_pat.simulation.beacon import BeaconProperties
from fsoc_pat.simulation.engine import SimulationEngine
from fsoc_pat.simulation.trajectories import CircularTrajectory


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


@pytest.mark.integration
class TestFullSystemIntegration:
    """End-to-end integration tests across all three phases."""

    def test_engine_to_perception(
        self, intrinsics: CameraIntrinsics, camera_state: CameraState,
    ) -> None:
        """Phase 1 engine feeds Phase 2 perception — detection matches GT."""
        # Phase 1: Run simulation
        config = ExperimentConfig(
            scenario=ScenarioConfig(duration_s=0.5, timestep_s=0.05, seed=42),
            target=TargetConfig(
                trajectory_type="circular",
                position=(100.0, 0.0, 0.0),
                radius=10.0,
                angular_rate=1.0,
            ),
        )
        engine = SimulationEngine(config)
        engine.initialize()
        results = engine.run()

        # Phase 2: Perception pipeline
        pipeline = PerceptionPipeline(
            intrinsics,
            PerceptionConfig(
                render=RenderConfig(noise_std=5.0, shot_noise=False,
                                    num_stars=0, enable_bloom=False, seed=42),
                detector=ClassicalDetectorConfig(
                    min_brightness=50.0,
                    min_area_px=2.0,
                    min_circularity=0.1,
                    preprocess=PreprocessConfig(
                        threshold_method="global",
                        threshold_value=50,
                        morph_open_kernel=2,
                        morph_close_kernel=2,
                    ),
                ),
            ),
        )

        beacon = BeaconProperties(intensity=230.0, size_px=8.0)
        detections = 0
        total_error = 0.0

        for snap in results.ground_truth.snapshots[::2]:  # Every other frame
            target_pos = snap.target.position.as_array()

            det = pipeline.process_frame(
                camera_state, [(target_pos, beacon)], timestamp=snap.time.time_s,
            )

            gt_pixel = snap.true_pixel
            if det.detected and gt_pixel is not None:
                detections += 1
                err = math.sqrt(
                    (det.best.u - gt_pixel.u) ** 2
                    + (det.best.v - gt_pixel.v) ** 2
                )
                total_error += err

        # Should detect most frames
        assert detections > 3, f"Only {detections} detections"

        # Average error should be reasonable
        avg_error = total_error / detections
        assert avg_error < 15.0, f"Mean error {avg_error:.1f}px too high"

    def test_ai_model_inference_on_rendered_frame(
        self, intrinsics: CameraIntrinsics, camera_state: CameraState,
    ) -> None:
        """AI model should run inference on a rendered frame without error."""
        # Render a frame
        renderer = SyntheticRenderer(
            intrinsics,
            RenderConfig(noise_std=5.0, shot_noise=False, num_stars=5, seed=42),
        )
        beacon = BeaconProperties(intensity=220.0, size_px=8.0)
        target = np.array([100.0, 0.0, 0.0])
        frame = renderer.render_single_target(camera_state, target, beacon)

        # Run through AI detector (untrained)
        detector = AIDetector(confidence_threshold=0.0)
        result = detector.detect(frame)

        assert result.frame_index == 1
        assert result.processing_time_ms > 0
        # Untrained model may or may not detect — just verify no crash

    def test_hybrid_on_rendered_frame(
        self, intrinsics: CameraIntrinsics, camera_state: CameraState,
    ) -> None:
        """Hybrid detector should work on rendered frames."""
        renderer = SyntheticRenderer(
            intrinsics,
            RenderConfig(noise_std=3.0, shot_noise=False, num_stars=0,
                         enable_bloom=False, seed=42),
        )
        beacon = BeaconProperties(intensity=240.0, size_px=10.0)
        target = np.array([100.0, 0.0, 0.0])
        frame = renderer.render_single_target(camera_state, target, beacon)

        hybrid = HybridDetector(HybridConfig(
            classical_config=ClassicalDetectorConfig(
                min_brightness=50.0,
                min_area_px=2.0,
                preprocess=PreprocessConfig(
                    threshold_method="global",
                    threshold_value=50,
                ),
            ),
        ))
        result = hybrid.detect(frame)

        # Classical should detect it; hybrid result should exist
        assert result.best is not None
        assert abs(result.best.u - 320) < 20
        assert abs(result.best.v - 240) < 20

    def test_dataset_samples_are_valid(
        self, intrinsics: CameraIntrinsics,
    ) -> None:
        """Generated dataset samples should have valid images and labels."""
        gen = DatasetGenerator(intrinsics, DatasetConfig(num_samples=20, seed=42))
        samples = gen.generate()

        positive = [s for s in samples if s.present]
        negative = [s for s in samples if not s.present]

        assert len(positive) > 0
        assert len(negative) > 0

        for s in positive:
            assert s.image.shape == (480, 640)
            assert not math.isnan(s.u)
            assert not math.isnan(s.v)
            assert 0 <= s.u <= 640
            assert 0 <= s.v <= 480

        for s in negative:
            assert math.isnan(s.u)
            assert math.isnan(s.v)

    def test_beaconnet_architecture(self) -> None:
        """BeaconNet should satisfy design constraints."""
        model = BeaconNet()

        # < 100K parameters
        params = model.count_parameters()
        assert params < 100_000

        # Forward pass works
        x = torch.randn(1, 1, 120, 160)
        out = model(x)
        assert "presence_logit" in out
        assert "u_norm" in out
        assert "v_norm" in out

        # Outputs are finite
        assert torch.isfinite(out["presence_logit"]).all()
        assert torch.isfinite(out["u_norm"]).all()

    def test_ground_truth_never_fed_to_detector(
        self, intrinsics: CameraIntrinsics, camera_state: CameraState,
    ) -> None:
        """Architectural invariant: detectors never see ground truth.

        The classical and AI detectors receive ONLY pixel images.
        We verify this by checking that the detect() methods accept
        only numpy arrays, not TargetState or Position3D.
        """
        import inspect

        # Classical detector
        sig = inspect.signature(ClassicalDetector.detect)
        param_names = list(sig.parameters.keys())
        assert "image" in param_names
        assert "target" not in param_names
        assert "position" not in param_names
        assert "ground_truth" not in param_names

        # AI detector
        sig_ai = inspect.signature(AIDetector.detect)
        param_names_ai = list(sig_ai.parameters.keys())
        assert "image" in param_names_ai
        assert "target" not in param_names_ai
        assert "ground_truth" not in param_names_ai
