"""Virtual camera model.

Wraps the gimbal, intrinsics, and camera state into a unified sensor
model. In Phase 1, the camera provides state and ground truth — image
generation is added in Phase 2.

The camera does NOT know the estimator's internal state. It is a
passive sensor that generates measurements (images) from the world.

Units:
    Position: meters
    Orientation: radians
    Time: seconds
    Frame rate: Hz
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fsoc_pat.core.state import CameraState
from fsoc_pat.core.types import AngularState, Orientation, Position3D
from fsoc_pat.geometry.camera_model import CameraIntrinsics
from fsoc_pat.simulation.dynamics import GimbalDynamics, VirtualGimbal


@dataclass
class VirtualCamera:
    """Virtual camera sensor with gimbal and intrinsics.

    Attributes:
        position: Camera position in world frame (m). Fixed in Phase 1.
        platform_orientation: Platform body orientation (rad).
        intrinsics: Camera intrinsic parameters.
        gimbal: Virtual gimbal for pan-tilt control.
        frame_rate_hz: Nominal frame rate (Hz). Design assumption.
        latency_s: Measurement latency (s). Design assumption.
    """

    position: Position3D
    platform_orientation: Orientation
    intrinsics: CameraIntrinsics
    gimbal: VirtualGimbal = field(default_factory=VirtualGimbal)
    frame_rate_hz: float = 30.0
    latency_s: float = 0.0

    # Internal counters
    _frame_count: int = field(default=0, init=False, repr=False)

    @property
    def state(self) -> CameraState:
        """Current camera state snapshot."""
        return CameraState(
            position=self.position,
            orientation=self.platform_orientation,
            angular_state=self.gimbal.state,
            timestamp=0.0,  # Updated by engine
        )

    def get_state_at(self, t: float) -> CameraState:
        """Get camera state with timestamp.

        Args:
            t: Current simulation time (s).

        Returns:
            CameraState with the given timestamp.
        """
        return CameraState(
            position=self.position,
            orientation=self.platform_orientation,
            angular_state=self.gimbal.state,
            timestamp=t,
        )

    def command_gimbal(
        self,
        pan_rate: float,
        tilt_rate: float,
        dt: float,
    ) -> AngularState:
        """Send a rate command to the gimbal and advance dynamics.

        Args:
            pan_rate: Commanded pan rate (rad/s).
            tilt_rate: Commanded tilt rate (rad/s).
            dt: Timestep (s).

        Returns:
            New actual gimbal angular state.
        """
        return self.gimbal.step(pan_rate, tilt_rate, dt)

    def reset(self) -> None:
        """Reset camera and gimbal to initial state."""
        self.gimbal.reset()
        self._frame_count = 0

    @property
    def frame_count(self) -> int:
        """Number of frames captured."""
        return self._frame_count

    def increment_frame(self) -> int:
        """Increment and return the frame counter."""
        self._frame_count += 1
        return self._frame_count
