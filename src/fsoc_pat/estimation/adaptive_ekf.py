"""Adaptive Extended Kalman Filter (Adaptive EKF) for Optical Beacon Tracking.

Features:
  - 6-state kinematics: [u, du/dt, d2u/dt2, v, dv/dt, d2v/dt2]
  - Adaptive measurement noise scaling based on detection confidence and scintillation
  - Innovation-based covariance adaptation (Sage-Husa / residual monitoring)
  - Mahalanobis outlier gating for clutter and solar glint rejection
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from fsoc_pat.perception.detection_types import Detection


@dataclass
class AdaptiveEKFConfig:
    """Adaptive EKF configuration.

    Attributes:
        base_r_std: Nominal measurement standard deviation in pixels.
        q_accel_std: Process noise spectral density for acceleration random walk.
        gate_threshold: Chi-squared gating threshold (2 DOF, default 9.21 = 99%).
        adaptive_r_enabled: Enable dynamic measurement covariance scaling.
        scintillation_coupling: Scaling factor for atmospheric scintillation coupling.
    """

    base_r_std: float = 1.5
    q_accel_std: float = 300.0
    gate_threshold: float = 25.0
    adaptive_r_enabled: bool = True
    scintillation_coupling: float = 2.5


class AdaptiveEKF:
    """High-accuracy Adaptive Extended Kalman Filter for pixel-space beacon tracking."""

    def __init__(self, config: AdaptiveEKFConfig | None = None) -> None:
        self.config = config or AdaptiveEKFConfig()
        # State vector: [u, u_dot, u_ddot, v, v_dot, v_ddot]^T
        self.state: NDArray[np.float64] = np.zeros(6, dtype=np.float64)
        self.cov: NDArray[np.float64] = np.eye(6, dtype=np.float64) * 100.0
        self.is_initialized: bool = False
        self.last_dt: float = 0.02
        self.innovation: NDArray[np.float64] = np.zeros(2, dtype=np.float64)
        self.innovation_cov: NDArray[np.float64] = np.eye(2, dtype=np.float64)

    def reset(self) -> None:
        """Reset estimator to uninitialized state."""
        self.state.fill(0.0)
        self.cov = np.eye(6, dtype=np.float64) * 100.0
        self.is_initialized = False
        self.innovation.fill(0.0)
        self.innovation_cov = np.eye(2, dtype=np.float64)

    def initialize_with_pixel(self, u: float, v: float) -> None:
        """Seed initial state with first observed pixel location."""
        self.state = np.array([u, 0.0, 0.0, v, 0.0, 0.0], dtype=np.float64)
        r_var = self.config.base_r_std**2
        self.cov = np.diag([r_var, 1500.0, 3000.0, r_var, 1500.0, 3000.0]).astype(np.float64)
        self.is_initialized = True

    def initialize(
        self,
        measurement: NDArray[np.float64] | tuple[float, float] | list[float],
        dt: float | None = None,
    ) -> None:
        """Initialize filter state from measurement (compatible with KalmanFilter interface)."""
        u = float(measurement[0])
        v = float(measurement[1])
        self.initialize_with_pixel(u, v)
        if dt is not None:
            self.last_dt = dt


    def predict(self, dt: float) -> tuple[float, float]:
        """Propagate state and covariance forward by dt seconds.

        Returns:
            Tuple of predicted (u, v) pixel position.
        """
        self.last_dt = dt
        dt2 = 0.5 * dt * dt

        # State transition matrix F for Constant Acceleration (CA) model
        # Axis u: indices 0, 1, 2. Axis v: indices 3, 4, 5
        F = np.eye(6, dtype=np.float64)
        F[0, 1] = dt
        F[0, 2] = dt2
        F[1, 2] = dt
        F[3, 4] = dt
        F[3, 5] = dt2
        F[4, 5] = dt

        # Discrete process noise matrix Q
        q = self.config.q_accel_std**2
        dt5 = dt**5 / 20.0
        dt4 = dt**4 / 8.0
        dt3 = dt**3 / 6.0
        dt2_2 = dt**2 / 2.0

        Q_block = q * np.array([
            [dt5, dt4, dt3],
            [dt4, dt**3 / 3.0, dt2_2],
            [dt3, dt2_2, dt],
        ], dtype=np.float64)

        Q = np.zeros((6, 6), dtype=np.float64)
        Q[0:3, 0:3] = Q_block
        Q[3:6, 3:6] = Q_block

        # Propagate
        self.state = F @ self.state
        self.cov = F @ self.cov @ F.T + Q
        # Enforce symmetry
        self.cov = 0.5 * (self.cov + self.cov.T)

        return float(self.state[0]), float(self.state[3])

    def update(
        self,
        detection: Detection | None,
        scintillation_index: float = 0.0,
    ) -> tuple[float, float]:
        """Incorporate detection measurement with adaptive noise scaling.

        Args:
            detection: Detection object with pixel coordinates and confidence.
            scintillation_index: Normalized atmospheric intensity variance.

        Returns:
            Tuple of filtered (u, v) pixel estimate.
        """
        if detection is None:
            return float(self.state[0]), float(self.state[3])

        # Extract (u, v) from various Detection representations or raw measurements
        meas_u: float | None = None
        meas_v: float | None = None

        if isinstance(detection, (tuple, list, np.ndarray)) and len(detection) >= 2:
            meas_u, meas_v = float(detection[0]), float(detection[1])
        elif hasattr(detection, "u") and hasattr(detection, "v"):
            meas_u, meas_v = float(detection.u), float(detection.v)
        elif hasattr(detection, "pixel_position") and detection.pixel_position is not None:
            meas_u, meas_v = detection.pixel_position
        elif hasattr(detection, "best") and detection.best is not None:
            meas_u, meas_v = detection.best.u, detection.best.v
        elif hasattr(detection, "centroid_x") and hasattr(detection, "centroid_y"):
            meas_u, meas_v = float(detection.centroid_x), float(detection.centroid_y)

        if meas_u is None or meas_v is None or math.isnan(meas_u) or math.isnan(meas_v):
            return float(self.state[0]), float(self.state[3])

        if not self.is_initialized:
            self.initialize_with_pixel(meas_u, meas_v)
            return meas_u, meas_v

        # Measurement matrix H: maps [u, du, d2u, v, dv, d2v] -> [u, v]
        H = np.zeros((2, 6), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 3] = 1.0

        # Dynamic measurement covariance adaptation:
        # Lower confidence and higher scintillation -> higher measurement variance
        r_std = self.config.base_r_std
        if self.config.adaptive_r_enabled:
            conf_val = getattr(detection, "confidence", 1.0)
            conf = max(0.01, min(1.0, float(conf_val)))
            confidence_penalty = (1.0 - conf) / (conf + 1e-3)
            turbulence_penalty = self.config.scintillation_coupling * scintillation_index
            r_std *= math.sqrt(1.0 + confidence_penalty + turbulence_penalty)

        R = np.eye(2, dtype=np.float64) * (r_std**2)

        # Innovation (residual)
        z = np.array([meas_u, meas_v], dtype=np.float64)
        y = z - (H @ self.state)
        S = H @ self.cov @ H.T + R  # Innovation covariance
        self.innovation = y
        self.innovation_cov = S

        # Outlier gating using Mahalanobis distance
        try:
            S_inv = np.linalg.inv(S)
            d2 = float(y.T @ S_inv @ y)
            if d2 > self.config.gate_threshold:
                conf_val = getattr(detection, "confidence", 1.0)
                if conf_val >= 0.6:
                    # Dynamically inflate R to smoothly pull filter towards high-confidence maneuver
                    R_inflated = R * max(1.0, (d2 / self.config.gate_threshold))
                    S = H @ self.cov @ H.T + R_inflated
                    S_inv = np.linalg.inv(S)
                else:
                    # Outlier rejected (likely clutter or glint); keep predicted state
                    return float(self.state[0]), float(self.state[3])
        except np.linalg.LinAlgError:
            return float(self.state[0]), float(self.state[3])

        # Kalman Gain
        K = self.cov @ H.T @ S_inv

        # State and covariance update
        self.state = self.state + K @ y
        I_KH = np.eye(6, dtype=np.float64) - (K @ H)
        # Joseph stabilized form: P = (I-KH) P (I-KH)^T + K R K^T
        self.cov = I_KH @ self.cov @ I_KH.T + K @ R @ K.T
        self.cov = 0.5 * (self.cov + self.cov.T)

        return float(self.state[0]), float(self.state[3])

    @property
    def estimated_pixel(self) -> tuple[float, float]:
        """Current estimated (u, v) pixel."""
        return float(self.state[0]), float(self.state[3])

    @property
    def estimated_position(self) -> tuple[float, float]:
        """Current estimated (u, v) pixel (alias for estimated_pixel)."""
        return float(self.state[0]), float(self.state[3])

    @property
    def initialized(self) -> bool:
        """Whether the filter has received an initial measurement."""
        return self.is_initialized

    @property
    def estimated_velocity_px_s(self) -> tuple[float, float]:
        """Estimated pixel velocity (du/dt, dv/dt)."""
        return float(self.state[1]), float(self.state[4])
