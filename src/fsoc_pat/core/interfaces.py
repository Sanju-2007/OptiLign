"""Protocol interfaces for pluggable algorithm components.

These protocols define the contracts that all detectors, estimators,
controllers, trajectories, and disturbances must satisfy. This enables
adding new algorithms without modifying the simulation engine, GUI,
metrics engine, or experiment engine.

Usage:
    A new detector must implement DetectorProtocol.
    A new controller must implement ControllerProtocol.
    Registration is done through configuration — no code changes needed
    in consuming modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from fsoc_pat.core.types import (
    Acceleration3D,
    AngularError,
    ControlCommand,
    Position3D,
    Velocity3D,
)

if TYPE_CHECKING:
    from fsoc_pat.core.state import CameraState, TargetState


# ---------------------------------------------------------------------------
# Trajectory Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class TrajectoryProtocol(Protocol):
    """Interface for target trajectory generators.

    Every trajectory must provide analytical position, velocity, and
    acceleration at any time t >= 0. Numerical differentiation of noisy
    positions is prohibited when analytical derivatives are available.
    """

    def position(self, t: float) -> Position3D:
        """Target position at time t (m).

        Args:
            t: Simulation time (s). Must be >= 0.
        """
        ...

    def velocity(self, t: float) -> Velocity3D:
        """Target velocity at time t (m/s).

        Args:
            t: Simulation time (s). Must be >= 0.
        """
        ...

    def acceleration(self, t: float) -> Acceleration3D:
        """Target acceleration at time t (m/s²).

        Args:
            t: Simulation time (s). Must be >= 0.
        """
        ...


# ---------------------------------------------------------------------------
# Detector Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class DetectorProtocol(Protocol):
    """Interface for beacon detection modules.

    A detector processes a camera image and returns structured detection
    results. It must NOT access ground-truth target position.
    """

    def detect(self, image: NDArray[np.uint8]) -> DetectionResult:
        """Detect beacon candidates in an image.

        Args:
            image: Camera frame (H, W) or (H, W, C) uint8 array.

        Returns:
            Structured detection result with candidates and confidence.
        """
        ...


@runtime_checkable
class DetectionResult(Protocol):
    """Result from a beacon detector."""

    @property
    def detected(self) -> bool:
        """Whether a valid beacon candidate was found."""
        ...

    @property
    def pixel_position(self) -> tuple[float, float] | None:
        """Detected beacon centroid (u, v) in pixels, or None."""
        ...

    @property
    def confidence(self) -> float:
        """Detection confidence in [0, 1]."""
        ...


# ---------------------------------------------------------------------------
# Estimator Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class EstimatorProtocol(Protocol):
    """Interface for state estimation filters (KF, EKF, UKF).

    The estimator maintains an internal state estimate and covariance.
    It must operate ONLY on measurements — never on ground truth.
    """

    def predict(self, dt: float) -> NDArray[np.float64]:
        """Predict state forward by dt seconds.

        Args:
            dt: Time step (s).

        Returns:
            Predicted state vector.
        """
        ...

    def update(self, measurement: NDArray[np.float64]) -> NDArray[np.float64]:
        """Update state estimate with a new measurement.

        Args:
            measurement: Measurement vector.

        Returns:
            Updated state vector.
        """
        ...

    @property
    def state(self) -> NDArray[np.float64]:
        """Current state estimate vector."""
        ...

    @property
    def covariance(self) -> NDArray[np.float64]:
        """Current state covariance matrix."""
        ...


# ---------------------------------------------------------------------------
# Controller Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class ControllerProtocol(Protocol):
    """Interface for control algorithms (PID, LQR, MPC).

    The controller computes gimbal commands from angular error and
    system state. It must NOT access ground-truth target position.
    """

    def compute(
        self,
        error: AngularError,
        dt: float,
        **kwargs: Any,
    ) -> ControlCommand:
        """Compute control command from pointing error.

        Args:
            error: Current angular pointing error (rad).
            dt: Control timestep (s).
            **kwargs: Additional state info (velocity estimate, confidence, etc.)

        Returns:
            Gimbal pan/tilt rate command.
        """
        ...

    def reset(self) -> None:
        """Reset controller internal state (integrators, etc.)."""
        ...


# ---------------------------------------------------------------------------
# Disturbance Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class DisturbanceProtocol(Protocol):
    """Interface for disturbance models.

    Each disturbance is independently configurable and controllable.
    Disturbances modify system state or measurements, never the
    ground truth.
    """

    def apply(self, value: float, t: float) -> float:
        """Apply disturbance to a scalar value.

        Args:
            value: Clean input value.
            t: Current simulation time (s).

        Returns:
            Disturbed value.
        """
        ...

    def reset(self) -> None:
        """Reset disturbance internal state."""
        ...
