"""Composite state objects used across the simulation.

These objects aggregate domain types into meaningful system snapshots.
They are used for ground-truth recording, performance evaluation,
and inter-subsystem communication.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fsoc_pat.core.types import (
    Acceleration3D,
    AngularError,
    AngularState,
    Orientation,
    PixelCoord,
    Position3D,
    SimulationTime,
    Velocity3D,
)


@dataclass(frozen=True, slots=True)
class TargetState:
    """Complete target state at a given instant.

    Attributes:
        position: Target position in world frame (m).
        velocity: Target velocity in world frame (m/s).
        acceleration: Target acceleration in world frame (m/s²).
        visible: Whether the target is theoretically visible to the camera.
        timestamp: Simulation time of this state.
    """

    position: Position3D
    velocity: Velocity3D
    acceleration: Acceleration3D
    visible: bool = True
    timestamp: float = 0.0


@dataclass(frozen=True, slots=True)
class CameraState:
    """Complete camera/gimbal state at a given instant.

    Attributes:
        position: Camera position in world frame (m).
        orientation: Camera orientation (Euler angles, rad).
        angular_state: Pan-tilt gimbal state.
        timestamp: Simulation time of this state.
    """

    position: Position3D
    orientation: Orientation
    angular_state: AngularState
    timestamp: float = 0.0


@dataclass(frozen=True, slots=True)
class GroundTruthSnapshot:
    """Single-timestep ground-truth record.

    Contains everything needed to evaluate system performance at one instant.
    This data is ONLY used for validation and metrics — never fed to the
    tracking/control loop.

    Attributes:
        time: Simulation time.
        target: True target state.
        camera: True camera state.
        true_pixel: True target pixel location (None if not in FOV).
        true_angular_error: True angular error between boresight and target LOS.
        true_los: True line-of-sight unit vector from camera to target (world frame).
        target_in_fov: Whether the target is within the camera FOV.
    """

    time: SimulationTime
    target: TargetState
    camera: CameraState
    true_pixel: PixelCoord | None = None
    true_angular_error: AngularError | None = None
    true_los: tuple[float, float, float] | None = None
    target_in_fov: bool = False


class GroundTruthLog:
    """Append-only container for ground-truth snapshots.

    Designed for efficient sequential access during post-simulation analysis.
    """

    def __init__(self) -> None:
        self._snapshots: list[GroundTruthSnapshot] = []

    def record(self, snapshot: GroundTruthSnapshot) -> None:
        """Append a ground-truth snapshot."""
        self._snapshots.append(snapshot)

    @property
    def snapshots(self) -> list[GroundTruthSnapshot]:
        """All recorded snapshots (read-only reference)."""
        return self._snapshots

    def __len__(self) -> int:
        return len(self._snapshots)

    def __getitem__(self, index: int) -> GroundTruthSnapshot:
        return self._snapshots[index]

    def clear(self) -> None:
        """Clear all recorded snapshots."""
        self._snapshots.clear()

    @property
    def timestamps(self) -> list[float]:
        """List of all recorded timestamps (s)."""
        return [s.time.time_s for s in self._snapshots]
