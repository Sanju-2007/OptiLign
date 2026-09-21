"""Simulation world — contains all entities and advances their state.

The world is the container for targets, camera, and environment.
It does not perform perception, estimation, or control — those
are handled by higher-level subsystems.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from fsoc_pat.simulation.camera import VirtualCamera
from fsoc_pat.simulation.target import Target


@dataclass
class SimulationWorld:
    """Container for all simulation entities.

    Attributes:
        camera: The virtual camera/sensor.
        targets: List of targets in the world.
    """

    camera: VirtualCamera
    targets: list[Target] = field(default_factory=list)

    @property
    def designated_target(self) -> Target | None:
        """The designated tracking target, or None if no target is designated."""
        for target in self.targets:
            if target.is_designated:
                return target
        return None

    def add_target(self, target: Target) -> None:
        """Add a target to the world."""
        self.targets.append(target)

    def reset(self) -> None:
        """Reset the world state."""
        self.camera.reset()
