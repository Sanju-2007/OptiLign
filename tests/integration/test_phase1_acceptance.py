"""Phase 1 Acceptance Test — End-to-end mathematical correctness.

Verifies that a target moving on a circular trajectory projects to
the expected pixel path, and that angular errors match analytical
ground truth within tolerance.

This test validates the entire Phase 1 chain:
    trajectory → target state → camera state → projection → angular error
"""

import math

import numpy as np
import pytest

from fsoc_pat.config.models import CameraConfig, ExperimentConfig, ScenarioConfig, TargetConfig
from fsoc_pat.core.state import CameraState
from fsoc_pat.core.types import AngularState, Orientation, PixelCoord, Position3D
from fsoc_pat.geometry.angular_error import compute_angular_error_from_pixel
from fsoc_pat.geometry.camera_model import CameraIntrinsics
from fsoc_pat.geometry.projection import project_point_to_pixel
from fsoc_pat.simulation.engine import SimulationEngine
from fsoc_pat.simulation.trajectories import CircularTrajectory


@pytest.mark.integration
class TestPhase1Acceptance:
    """Integration tests verifying mathematical correctness end-to-end."""

    def test_circular_target_pixel_trace(self) -> None:
        """A circular trajectory should produce an elliptical pixel trace.

        The target orbits at radius R around a center point ahead of
        the camera. The projected pixel path should be an approximate
        ellipse on the image plane.
        """
        intrinsics = CameraIntrinsics.from_fov(640, 480, math.radians(60.0))
        camera_state = CameraState(
            position=Position3D(0.0, 0.0, 0.0),
            orientation=Orientation(),
            angular_state=AngularState(),
        )

        center = Position3D(100.0, 0.0, 0.0)
        radius = 10.0
        traj = CircularTrajectory(
            center=center, radius=radius, angular_rate=1.0,
        )

        pixels_u = []
        pixels_v = []

        # Sample one full orbit
        for i in range(100):
            t = i * (2 * math.pi) / 100  # one full period
            pos = traj.position(t)
            pixel = project_point_to_pixel(
                pos.as_array(), camera_state, intrinsics,
            )
            assert pixel is not None, f"Target should be visible at t={t:.3f}"
            pixels_u.append(pixel.u)
            pixels_v.append(pixel.v)

        # The trace should be centered near the image center
        mean_u = np.mean(pixels_u)
        mean_v = np.mean(pixels_v)
        assert mean_u == pytest.approx(intrinsics.cx, abs=5.0)
        assert mean_v == pytest.approx(intrinsics.cy, abs=5.0)

        # The trace should have nonzero spread in U (horizontal)
        assert max(pixels_u) - min(pixels_u) > 10.0

    def test_angular_error_matches_geometry(self) -> None:
        """Angular error from pixel should match analytical geometry.

        For a target at (100, Y, 0) with camera at origin:
            tan(angle) = Y / 100
            angle = atan(Y / 100)
        """
        intrinsics = CameraIntrinsics.from_fov(640, 480, math.radians(60.0))
        camera_state = CameraState(
            position=Position3D(0.0, 0.0, 0.0),
            orientation=Orientation(),
            angular_state=AngularState(),
        )

        for y_offset in [0.0, 5.0, 10.0, -5.0, -10.0]:
            target = np.array([100.0, y_offset, 0.0])
            pixel = project_point_to_pixel(target, camera_state, intrinsics)

            if pixel is None:
                continue

            err = compute_angular_error_from_pixel(pixel, intrinsics)

            # Analytical horizontal angle
            expected_h = math.atan(y_offset / 100.0)
            assert err.horizontal == pytest.approx(expected_h, abs=1e-6), (
                f"y_offset={y_offset}: expected {expected_h:.6f}, got {err.horizontal:.6f}"
            )

    def test_engine_end_to_end(self) -> None:
        """Full engine run with circular target — verify results are physical."""
        config = ExperimentConfig(
            scenario=ScenarioConfig(
                name="acceptance_test",
                duration_s=2.0,
                timestep_s=0.01,
                seed=42,
            ),
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

        # Should complete successfully
        assert results.total_steps == 200
        assert results.duration_s == pytest.approx(2.0)

        # Ground truth should have reasonable values
        for snap in results.ground_truth.snapshots:
            # Target position should stay on circle
            tp = snap.target.position
            dist_from_center = math.sqrt(
                (tp.x - 100.0) ** 2 + tp.y ** 2 + tp.z ** 2
            )
            assert dist_from_center == pytest.approx(10.0, abs=0.01)

            # Angular error should be defined (target in front of camera)
            assert snap.true_pixel is not None
            assert snap.true_angular_error is not None
            assert snap.true_angular_error.total >= 0

        # With no control, camera stays pointed at origin.
        # Angular error should correspond to the target's angular offset
        last = results.ground_truth[-1]
        assert last.true_angular_error is not None
        # Target is at ~10m offset at 100m range → ~0.1 rad = ~5.7°
        assert last.true_angular_error.total < 0.2  # sanity bound

    def test_target_velocity_recorded(self) -> None:
        """Ground truth should record correct velocity for circular trajectory."""
        config = ExperimentConfig(
            scenario=ScenarioConfig(duration_s=0.5, timestep_s=0.01, seed=1),
            target=TargetConfig(
                trajectory_type="circular",
                position=(100.0, 0.0, 0.0),
                radius=10.0,
                angular_rate=2.0,
            ),
        )
        engine = SimulationEngine(config)
        engine.initialize()
        results = engine.run()

        # Speed should be |ω| × r = 2 × 10 = 20 m/s
        for snap in results.ground_truth.snapshots:
            speed = snap.target.velocity.magnitude
            assert speed == pytest.approx(20.0, abs=0.1)

    def test_csv_export(self, tmp_path: object) -> None:
        """Results should export to CSV without error."""
        import os

        config = ExperimentConfig(
            scenario=ScenarioConfig(duration_s=0.1, timestep_s=0.01, seed=1),
            target=TargetConfig(trajectory_type="stationary", position=(100, 0, 0)),
        )
        engine = SimulationEngine(config)
        engine.initialize()
        results = engine.run()

        output_dir = str(tmp_path)
        results.export_csv(output_dir)
        assert os.path.exists(os.path.join(output_dir, "ground_truth.csv"))
