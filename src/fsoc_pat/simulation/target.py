"""Target model — combines trajectory with beacon properties.

A Target is a physical entity in the simulation world with a trajectory
(motion model) and beacon properties (optical signature). Multiple
targets can exist in the world.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fsoc_pat.core.interfaces import TrajectoryProtocol
from fsoc_pat.core.state import TargetState
from fsoc_pat.simulation.beacon import BeaconProperties


@dataclass
class Target:
    """A trackable target in the simulation world.

    Attributes:
        target_id: Unique identifier for this target.
        trajectory: Motion model implementing TrajectoryProtocol.
        beacon: Optical beacon properties.
        is_designated: Whether this is the designated tracking beacon.
            Only one target should be designated in a standard scenario.
        label: Human-readable label for display.
    """

    target_id: int
    trajectory: TrajectoryProtocol
    beacon: BeaconProperties = field(default_factory=BeaconProperties)
    is_designated: bool = True
    label: str = "Target"

    def state_at(self, t: float) -> TargetState:
        """Get the complete target state at time t.

        Queries the trajectory for position, velocity, and acceleration.
        Beacon visibility is determined by beacon blinking properties.

        Args:
            t: Simulation time (s).

        Returns:
            TargetState snapshot.
        """
        return TargetState(
            position=self.trajectory.position(t),
            velocity=self.trajectory.velocity(t),
            acceleration=self.trajectory.acceleration(t),
            visible=self.beacon.is_visible_at_time(t),
            timestamp=t,
        )
