"""Linear Kalman Filter.

Standard discrete-time Kalman filter for linear systems.
Implements the EstimatorProtocol interface.

State vector is in pixel space: [u, (vu), (au), v, (vv), (av)]
depending on the motion model.

Features:
  - Mahalanobis gating for outlier rejection
  - Innovation monitoring (NIS chi-squared test)
  - Covariance symmetry enforcement
  - Configurable measurement noise

This filter NEVER accesses ground truth. It operates solely on
pixel measurements from the detection pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fsoc_pat.estimation.motion_models import (
    MotionModel,
    MotionModelType,
    get_motion_model,
)


@dataclass
class KalmanConfig:
    """Kalman filter configuration.

    Attributes:
        motion_model: Motion model type (CV, CA, etc.).
        r_std: Measurement noise std (px).
        q_process: Process noise parameter (model-dependent).
        gate_threshold: Mahalanobis distance threshold for gating.
        dt: Default timestep (s).
    """

    motion_model: MotionModelType = MotionModelType.CONSTANT_VELOCITY
    r_std: float = 2.0
    q_process: float = 10.0
    gate_threshold: float = 9.21  # chi2(2, 0.99)
    dt: float = 0.05


class KalmanFilter:
    """Linear Kalman Filter for beacon tracking.

    Maintains a state estimate and covariance. Predict step
    propagates the state; update step incorporates a pixel
    measurement.

    Usage:
        kf = KalmanFilter(config)
        kf.initialize(measurement)  # First detection
        for each frame:
            kf.predict(dt)
            kf.update(measurement)
            pos = kf.estimated_position
    """

    def __init__(self, config: KalmanConfig | None = None) -> None:
        self._config = config or KalmanConfig()
        self._model: MotionModel | None = None
        self._x: NDArray[np.float64] | None = None  # State
        self._P: NDArray[np.float64] | None = None  # Covariance
        self._R: NDArray[np.float64] | None = None  # Measurement noise
        self._initialized = False
        self._last_innovation: NDArray[np.float64] | None = None
        self._last_nis: float = 0.0  # Normalized Innovation Squared
        self._gated_count: int = 0
        self._consecutive_gated: int = 0

    def initialize(
        self,
        measurement: NDArray[np.float64],
        dt: float | None = None,
    ) -> None:
        """Initialize the filter from a first measurement.

        Args:
            measurement: First pixel measurement [u, v].
            dt: Timestep (s). Uses config default if None.
        """
        cfg = self._config
        dt = dt or cfg.dt

        # Build motion model
        if cfg.motion_model == MotionModelType.CONSTANT_POSITION:
            self._model = get_motion_model(cfg.motion_model, dt, q_pos=cfg.q_process)
        elif cfg.motion_model == MotionModelType.CONSTANT_VELOCITY:
            self._model = get_motion_model(cfg.motion_model, dt, q_vel=cfg.q_process)
        elif cfg.motion_model == MotionModelType.CONSTANT_ACCELERATION:
            self._model = get_motion_model(cfg.motion_model, dt, q_acc=cfg.q_process)

        n = self._model.state_dim

        # Initialize state: position from measurement, velocities/accels = 0
        self._x = np.zeros(n, dtype=np.float64)
        H = self._model.H
        # Set position components from measurement using pseudo-inverse
        self._x = H.T @ measurement

        # Initial covariance: large uncertainty
        self._P = np.eye(n, dtype=np.float64) * 100.0

        # Measurement noise
        self._R = np.eye(2, dtype=np.float64) * (cfg.r_std ** 2)

        self._initialized = True

    def predict(self, dt: float | None = None) -> NDArray[np.float64]:
        """Predict state forward.

        Args:
            dt: Timestep. If different from config, rebuilds F and Q.

        Returns:
            Predicted state vector.
        """
        if not self._initialized:
            raise RuntimeError("Filter not initialized. Call initialize() first.")

        cfg = self._config
        dt = dt or cfg.dt

        # Rebuild model if dt changed
        if cfg.motion_model == MotionModelType.CONSTANT_POSITION:
            model = get_motion_model(cfg.motion_model, dt, q_pos=cfg.q_process)
        elif cfg.motion_model == MotionModelType.CONSTANT_VELOCITY:
            model = get_motion_model(cfg.motion_model, dt, q_vel=cfg.q_process)
        else:
            model = get_motion_model(cfg.motion_model, dt, q_acc=cfg.q_process)

        self._model = model
        F = model.F
        Q = model.Q

        # Predict
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q

        # Enforce symmetry
        self._P = (self._P + self._P.T) / 2

        return self._x.copy()

    def update(self, measurement: NDArray[np.float64]) -> NDArray[np.float64]:
        """Update state with a pixel measurement.

        Applies Mahalanobis gating. If the measurement is
        rejected, the state is not updated.

        Args:
            measurement: Pixel measurement [u, v].

        Returns:
            Updated (or unchanged if gated) state vector.
        """
        if not self._initialized:
            raise RuntimeError("Filter not initialized. Call initialize() first.")

        H = self._model.H

        # Innovation
        z_pred = H @ self._x
        y = measurement - z_pred  # Innovation
        self._last_innovation = y

        # Innovation covariance
        S = H @ self._P @ H.T + self._R

        # Mahalanobis distance (gating)
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Singular innovation covariance — skip update
            self._gated_count += 1
            return self._x.copy()

        nis = float(y.T @ S_inv @ y)
        self._last_nis = nis

        if nis > self._config.gate_threshold:
            # Outlier — reject measurement
            self._gated_count += 1
            self._consecutive_gated += 1
            if self._consecutive_gated >= 4:
                # Target maneuvered or relocked — recover state position
                self._x[0] = measurement[0]
                if self._model.state_dim >= 4:
                    self._x[2] = measurement[1]
                self._consecutive_gated = 0
            return self._x.copy()

        self._consecutive_gated = 0

        # Kalman gain
        K = self._P @ H.T @ S_inv

        # State update
        self._x = self._x + K @ y

        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self._model.state_dim) - K @ H
        self._P = I_KH @ self._P @ I_KH.T + K @ self._R @ K.T

        # Enforce symmetry
        self._P = (self._P + self._P.T) / 2

        return self._x.copy()

    @property
    def state(self) -> NDArray[np.float64]:
        """Current state estimate."""
        if self._x is None:
            return np.zeros(4, dtype=np.float64)
        return self._x.copy()

    @property
    def covariance(self) -> NDArray[np.float64]:
        """Current covariance matrix."""
        if self._P is None:
            return np.eye(4, dtype=np.float64) * 1e6
        return self._P.copy()

    @property
    def estimated_position(self) -> tuple[float, float]:
        """Estimated pixel position (u, v)."""
        if self._x is None:
            return (0.0, 0.0)
        H = self._model.H
        pos = H @ self._x
        return (float(pos[0]), float(pos[1]))

    @property
    def estimated_velocity(self) -> tuple[float, float]:
        """Estimated pixel velocity (vu, vv) px/s. Zero if model has no velocity."""
        if self._x is None or self._model.state_dim < 4:
            return (0.0, 0.0)
        return (float(self._x[1]), float(self._x[3]))

    @property
    def initialized(self) -> bool:
        """Whether the filter has been initialized."""
        return self._initialized

    @property
    def last_nis(self) -> float:
        """Normalized Innovation Squared from last update."""
        return self._last_nis

    @property
    def gated_count(self) -> int:
        """Number of measurements rejected by gating."""
        return self._gated_count

    def predict_ahead(self, steps: int, dt: float | None = None) -> NDArray[np.float64]:
        """Predict position N steps ahead without modifying filter state.

        Args:
            steps: Number of steps to predict.
            dt: Timestep per step.

        Returns:
            Predicted position [u, v] after N steps.
        """
        if not self._initialized:
            return np.array([0.0, 0.0])

        dt = dt or self._config.dt
        cfg = self._config

        if cfg.motion_model == MotionModelType.CONSTANT_POSITION:
            model = get_motion_model(cfg.motion_model, dt, q_pos=cfg.q_process)
        elif cfg.motion_model == MotionModelType.CONSTANT_VELOCITY:
            model = get_motion_model(cfg.motion_model, dt, q_vel=cfg.q_process)
        else:
            model = get_motion_model(cfg.motion_model, dt, q_acc=cfg.q_process)

        x = self._x.copy()
        for _ in range(steps):
            x = model.F @ x

        return model.H @ x

    def reset(self) -> None:
        """Reset filter to uninitialized state."""
        self._x = None
        self._P = None
        self._model = None
        self._initialized = False
        self._last_innovation = None
        self._last_nis = 0.0
        self._gated_count = 0
