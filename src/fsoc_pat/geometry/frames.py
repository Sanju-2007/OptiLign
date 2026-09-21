"""Coordinate frame definitions and conventions.

OptiLign uses a consistent right-handed coordinate convention
throughout the system. This module documents and provides utilities
for the frame system.

Chosen Convention (camera/aerospace standard):
    X = Forward (optical axis / boresight direction)
    Y = Right
    Z = Down

Frame Hierarchy:
    WORLD   — Fixed inertial frame. Origin at scenario-defined location.
    PLATFORM — Attached to the camera platform (e.g., satellite body).
               May have its own motion relative to WORLD.
    GIMBAL   — Attached to the pan-tilt mechanism. Rotates relative to
               PLATFORM by (pan, tilt) angles.
    CAMERA   — Attached to the camera sensor. Typically coincident with
               GIMBAL unless there is a fixed boresight offset.
    TARGET   — Attached to the target (for reference only).

All rotations are expressed as ZYX intrinsic Euler angles
(yaw → pitch → roll) unless otherwise noted. Internally all angles
are in radians.
"""

from __future__ import annotations

from enum import Enum, auto


class Frame(Enum):
    """Coordinate reference frames used in the simulation."""

    WORLD = auto()
    PLATFORM = auto()
    GIMBAL = auto()
    CAMERA = auto()
    TARGET = auto()


# Frame axis convention documentation
AXIS_CONVENTION = {
    "X": "Forward (optical axis / boresight)",
    "Y": "Right",
    "Z": "Down",
    "handedness": "Right-handed",
    "rotation_convention": "ZYX intrinsic (yaw → pitch → roll)",
    "angle_unit": "radians (internal)",
}
