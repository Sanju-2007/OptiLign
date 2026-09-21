"""Simulation engine — the main headless simulation loop.

Orchestrates the simulation clock, world, ground-truth recording,
and (in later phases) perception, estimation, and control subsystems.

Phase 1 scope: Truth simulation only — no perception or control.
The engine advances the world, records ground truth, and measures
the geometric relationship between camera and target.

The engine runs completely independently of any GUI.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from fsoc_pat.core.clock import ClockMode, SimulationClock
from fsoc_pat.core.events import Event, EventBus, EventType
from fsoc_pat.core.state import CameraState, GroundTruthLog, GroundTruthSnapshot, TargetState
from fsoc_pat.core.types import AngularError, PixelCoord, SimulationTime
from fsoc_pat.geometry.angular_error import (
    compute_angular_error_from_pixel,
    compute_angular_separation,
)
from fsoc_pat.geometry.los import compute_boresight, compute_los_vector
from fsoc_pat.geometry.projection import is_in_fov, project_point_to_pixel

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SimulationResults:
    """Results from a completed simulation run.

    Attributes:
        total_steps: Total simulation steps executed.
        duration_s: Simulated duration (s).
        wall_time_s: Wall-clock time elapsed (s).
        ground_truth: Complete ground-truth log.
        final_angular_error_rad: Final angular error (rad).
    """

    total_steps: int
    duration_s: float
    wall_time_s: float
    ground_truth: GroundTruthLog
    final_angular_error_rad: float

    def export_csv(self, output_dir: str) -> None:
        """Export ground-truth log to CSV.

        Args:
            output_dir: Directory to write CSV file.
        """
        import csv
        import os

        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, "ground_truth.csv")

        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time_s", "step",
                "target_x", "target_y", "target_z",
                "target_vx", "target_vy", "target_vz",
                "camera_pan", "camera_tilt",
                "pixel_u", "pixel_v",
                "angular_error_h_rad", "angular_error_v_rad", "angular_error_total_rad",
                "in_fov",
            ])

            for snap in self.ground_truth.snapshots:
                tp = snap.target.position
                tv = snap.target.velocity
                cs = snap.camera.angular_state
                px_u = snap.true_pixel.u if snap.true_pixel else ""
                px_v = snap.true_pixel.v if snap.true_pixel else ""
                ae_h = snap.true_angular_error.horizontal if snap.true_angular_error else ""
                ae_v = snap.true_angular_error.vertical if snap.true_angular_error else ""
                ae_t = snap.true_angular_error.total if snap.true_angular_error else ""

                writer.writerow([
                    f"{snap.time.time_s:.6f}", snap.time.step,
                    f"{tp.x:.6f}", f"{tp.y:.6f}", f"{tp.z:.6f}",
                    f"{tv.x:.6f}", f"{tv.y:.6f}", f"{tv.z:.6f}",
                    f"{cs.pan:.6f}", f"{cs.tilt:.6f}",
                    px_u, px_v,
                    ae_h, ae_v, ae_t,
                    snap.target_in_fov,
                ])

        logger.info("Ground truth exported to %s", filepath)


class SimulationEngine:
    """Headless simulation engine.

    Runs the simulation loop without any GUI dependency. Manages the
    clock, world, and ground-truth recording.

    Usage:
        engine = SimulationEngine(config)
        engine.initialize()
        results = engine.run()
    """

    def __init__(self, config: object) -> None:
        """Initialize with a configuration object.

        Args:
            config: ExperimentConfig from the config package.
        """
        from fsoc_pat.config.models import ExperimentConfig

        if not isinstance(config, ExperimentConfig):
            raise TypeError(f"Expected ExperimentConfig, got {type(config).__name__}")

        self._config: ExperimentConfig = config
        self._clock: SimulationClock | None = None
        self._world = None
        self._ground_truth = GroundTruthLog()
        self._event_bus = EventBus()
        self._initialized = False

    def initialize(self) -> None:
        """Set up the simulation from configuration.

        Creates the clock, world, targets, and camera.
        Must be called before run().
        """
        from fsoc_pat.config.models import ExperimentConfig
        from fsoc_pat.geometry.camera_model import CameraIntrinsics
        from fsoc_pat.simulation.beacon import BeaconProperties
        from fsoc_pat.simulation.camera import VirtualCamera
        from fsoc_pat.simulation.dynamics import GimbalDynamics, VirtualGimbal
        from fsoc_pat.simulation.target import Target
        from fsoc_pat.simulation.trajectories import (
            CircularTrajectory,
            LinearTrajectory,
            SinusoidalTrajectory,
            StationaryTrajectory,
        )
        from fsoc_pat.core.types import Orientation, Position3D, Velocity3D

        cfg = self._config
        sc = cfg.scenario

        # Set random seed
        np.random.seed(sc.seed)

        # Create clock
        self._clock = SimulationClock(dt=sc.timestep_s, mode=ClockMode.OFFLINE)

        # Create camera intrinsics
        cam_cfg = cfg.camera
        intrinsics = CameraIntrinsics.from_fov(
            width=cam_cfg.width,
            height=cam_cfg.height,
            hfov_rad=cam_cfg.hfov_rad,
            vfov_rad=cam_cfg.vfov_rad if cam_cfg.vfov_rad > 0 else None,
        )

        # Create gimbal
        gimbal_dynamics = GimbalDynamics(
            pan_rate_limit=cam_cfg.pan_rate_limit,
            tilt_rate_limit=cam_cfg.tilt_rate_limit,
            angular_accel_limit=cam_cfg.angular_accel_limit,
            actuator_time_constant=cam_cfg.actuator_time_constant,
        )
        gimbal = VirtualGimbal(dynamics=gimbal_dynamics)
        gimbal.set_seed(sc.seed + 1)

        # Create camera
        camera = VirtualCamera(
            position=Position3D(*cam_cfg.position),
            platform_orientation=Orientation(),
            intrinsics=intrinsics,
            gimbal=gimbal,
            frame_rate_hz=cam_cfg.fps,
            latency_s=cam_cfg.latency_s,
        )

        # Create trajectory
        tgt_cfg = cfg.target
        trajectory = self._build_trajectory(tgt_cfg)

        # Create target
        target = Target(
            target_id=1,
            trajectory=trajectory,
            beacon=BeaconProperties(
                intensity=tgt_cfg.beacon_intensity,
                size_px=tgt_cfg.beacon_size_px,
            ),
            is_designated=True,
            label="Primary Beacon",
        )

        # Build world
        from fsoc_pat.simulation.world import SimulationWorld
        self._world = SimulationWorld(camera=camera, targets=[target])

        self._ground_truth.clear()
        self._initialized = True

        logger.info(
            "Simulation initialized: duration=%.1fs, dt=%.4fs, seed=%d",
            sc.duration_s, sc.timestep_s, sc.seed,
        )

    def _build_trajectory(self, tgt_cfg: object) -> object:
        """Build a trajectory from target configuration."""
        from fsoc_pat.core.types import Acceleration3D, Position3D, Velocity3D
        from fsoc_pat.simulation.trajectories import (
            CircularTrajectory,
            ConstantAccelerationTrajectory,
            LinearTrajectory,
            SinusoidalTrajectory,
            StationaryTrajectory,
        )

        p0 = Position3D(*tgt_cfg.position)
        ttype = tgt_cfg.trajectory_type.lower()

        if ttype == "stationary":
            return StationaryTrajectory(pos=p0)

        elif ttype == "linear":
            v = Velocity3D(*tgt_cfg.velocity)
            return LinearTrajectory(p0=p0, vel=v)

        elif ttype == "constant_acceleration":
            v = Velocity3D(*tgt_cfg.velocity)
            a = Acceleration3D(*tgt_cfg.acceleration)
            return ConstantAccelerationTrajectory(p0=p0, v0=v, accel=a)

        elif ttype == "circular":
            return CircularTrajectory(
                center=p0,
                radius=tgt_cfg.radius,
                angular_rate=tgt_cfg.angular_rate,
            )

        elif ttype == "sinusoidal":
            return SinusoidalTrajectory(
                center=p0,
                amplitude=Position3D(*tgt_cfg.amplitude),
                frequency=Position3D(*tgt_cfg.frequency),
            )

        else:
            raise ValueError(f"Unknown trajectory type: {ttype}")

    def run(self) -> SimulationResults:
        """Execute the simulation loop.

        Returns:
            SimulationResults with ground-truth data and summary metrics.

        Raises:
            RuntimeError: If engine was not initialized.
        """
        if not self._initialized or self._clock is None or self._world is None:
            raise RuntimeError("Engine must be initialized before running")

        wall_start = time.monotonic()
        self._clock.start()

        num_steps = int(self._config.scenario.duration_s / self._config.scenario.timestep_s)

        self._event_bus.publish(Event(
            event_type=EventType.SIMULATION_STARTED,
            timestamp=0.0,
            source="engine",
        ))

        # Record initial state
        self._record_ground_truth(self._clock.now)

        for _ in range(num_steps):
            sim_time = self._clock.tick()
            self._step(sim_time)

        wall_elapsed = time.monotonic() - wall_start

        # Compute final angular error
        final_error = 0.0
        if self._ground_truth.snapshots:
            last = self._ground_truth.snapshots[-1]
            if last.true_angular_error is not None:
                final_error = last.true_angular_error.total

        self._event_bus.publish(Event(
            event_type=EventType.SIMULATION_STOPPED,
            timestamp=self._clock.time_s,
            source="engine",
        ))

        logger.info(
            "Simulation complete: %d steps, %.3fs wall time",
            self._clock.step, wall_elapsed,
        )

        return SimulationResults(
            total_steps=self._clock.step,
            duration_s=self._clock.time_s,
            wall_time_s=wall_elapsed,
            ground_truth=self._ground_truth,
            final_angular_error_rad=final_error,
        )

    def _step(self, sim_time: SimulationTime) -> None:
        """Execute one simulation step.

        Phase 1: Record ground truth only (no control loop).
        """
        self._record_ground_truth(sim_time)

    def _record_ground_truth(self, sim_time: SimulationTime) -> None:
        """Record the ground-truth state at the current time.

        Computes the true geometric relationship between camera and target.
        """
        assert self._world is not None

        target = self._world.designated_target
        if target is None:
            return

        t = sim_time.time_s
        target_state = target.state_at(t)
        camera_state = self._world.camera.get_state_at(t)

        # Project target to image
        true_pixel = project_point_to_pixel(
            target_state.position.as_array(),
            camera_state,
            self._world.camera.intrinsics,
        )

        # Compute angular error
        true_angular_error: AngularError | None = None
        in_fov = False

        if true_pixel is not None:
            in_fov = is_in_fov(true_pixel, self._world.camera.intrinsics)
            true_angular_error = compute_angular_error_from_pixel(
                true_pixel, self._world.camera.intrinsics,
            )

        # Compute LOS vector
        try:
            los = compute_los_vector(camera_state.position, target_state.position)
            true_los = (float(los[0]), float(los[1]), float(los[2]))
        except ValueError:
            true_los = None

        snapshot = GroundTruthSnapshot(
            time=sim_time,
            target=target_state,
            camera=camera_state,
            true_pixel=true_pixel,
            true_angular_error=true_angular_error,
            true_los=true_los,
            target_in_fov=in_fov,
        )

        self._ground_truth.record(snapshot)
