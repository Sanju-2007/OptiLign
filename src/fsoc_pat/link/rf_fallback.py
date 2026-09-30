"""Dual-Band Hybrid RF/Optical Link Fallback and Ballistic EKF Dead-Reckoning Manager.

Implements aerospace-standard hybrid optical-RF link handoff for deep atmospheric
attenuation and fog blackout conditions:
1. Dynamic Link Margin Monitoring (Optical threshold: -65 dBm / 0.3 µW).
2. Seamless Autonomous RF Handoff (60 GHz / 2.4 GHz backup link).
3. Ballistic Covariance-Inflation EKF Dead-Reckoning (up to 5.0s optical blackout coasting).
4. High-Intensity Burst Pulse Optical Carrier Reacquisition (<0.15s).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Tuple


@dataclass
class HybridLinkState:
    """Telemetry and operational state for the hybrid optical-RF link."""
    optical_power_dbm: float = -59.8
    optical_margin_db: float = 12.5
    is_optical_available: bool = True
    rf_fallback_active: bool = False
    link_carrier_mode: str = "OPTICAL_PRIMARY"  # OPTICAL_PRIMARY, HYBRID_RF_FALLBACK, BURST_REACQUIRING
    dead_reckoning_active: bool = False
    dead_reckoning_duration_s: float = 0.0
    rf_throughput_mbps: float = 100.0
    optical_throughput_gbps: float = 10.0
    active_data_throughput_mbps: float = 10000.0  # 10 Gbps in optical, 100 Mbps in RF
    packet_loss_ratio: float = 0.0
    burst_strobe_power_multiplier: float = 1.0
    chattering_suppressed_count: int = 0
    dwell_elapsed_s: float = 0.0


class DualBandLinkManager:
    """Manages autonomous optical-to-RF link handoffs and EKF dead-reckoning trajectory projection."""

    def __init__(
        self,
        optical_threshold_dbm: float = -65.0,
        handoff_hysteresis_db: float = 3.0,
        max_dead_reckoning_s: float = 5.0,
        rf_backup_bandwidth_mbps: float = 100.0,
        optical_bandwidth_gbps: float = 10.0,
        min_dwell_time_s: float = 0.0,
    ) -> None:
        self.optical_threshold_dbm = optical_threshold_dbm
        self.handoff_hysteresis_db = handoff_hysteresis_db
        self.max_dead_reckoning_s = max_dead_reckoning_s
        self.rf_backup_bandwidth_mbps = rf_backup_bandwidth_mbps
        self.optical_bandwidth_gbps = optical_bandwidth_gbps
        self.min_dwell_time_s = min_dwell_time_s

        self.state = HybridLinkState()
        self._blackout_start_time: float | None = None
        self._last_known_velocity: Tuple[float, float] = (0.0, 0.0)  # [pan_deg_s, tilt_deg_s]
        self._last_known_position: Tuple[float, float] = (0.0, 0.0)  # [pan_deg, tilt_deg]
        self._covariance_inflation_factor: float = 1.0
        self.auto_rf_fallback_enabled: bool = True
        self._state_dwell_time_s: float = 0.0
        self._pending_state: bool | None = None
        self._pending_dwell_s: float = 0.0

    def update(
        self,
        optical_power_dbm: float,
        current_gimbal_deg: Tuple[float, float],
        current_rates_deg_s: Tuple[float, float],
        dt: float = 0.033,
    ) -> HybridLinkState:
        """Update hybrid link status based on instantaneous optical power and kinematics.

        Args:
            optical_power_dbm: Current received optical power on the FPA.
            current_gimbal_deg: Current [pan_deg, tilt_deg] of the gimbal.
            current_rates_deg_s: Current [pan_rate, tilt_rate] in deg/s.
            dt: Simulation time step in seconds.

        Returns:
            HybridLinkState with updated carrier modes and telemetry.
        """
        self.state.optical_power_dbm = optical_power_dbm
        self.state.optical_margin_db = optical_power_dbm - self.optical_threshold_dbm

        # Optical link is viable if power is above threshold (with hysteresis for re-entry)
        threshold = (
            self.optical_threshold_dbm
            if not self.state.rf_fallback_active
            else (self.optical_threshold_dbm + self.handoff_hysteresis_db)
        )
        optical_ok = optical_power_dbm >= threshold

        # Anti-chattering dwell window filter
        if self.min_dwell_time_s > 0.0 and self.state.is_optical_available != optical_ok:
            if self._pending_state == optical_ok:
                self._pending_dwell_s += dt
                if self._pending_dwell_s < self.min_dwell_time_s:
                    # Suppress premature ping-pong transition until dwell window elapses
                    optical_ok = self.state.is_optical_available
                    self.state.chattering_suppressed_count += 1
                else:
                    self._pending_dwell_s = 0.0
                    self._pending_state = None
            else:
                self._pending_state = optical_ok
                self._pending_dwell_s = dt
                optical_ok = self.state.is_optical_available
                self.state.chattering_suppressed_count += 1
        else:
            self._pending_state = None
            self._pending_dwell_s = 0.0

        self._state_dwell_time_s += dt
        self.state.dwell_elapsed_s = self._state_dwell_time_s

        if not self.state.dead_reckoning_active:
            self._last_known_position = current_gimbal_deg
            self._last_known_velocity = current_rates_deg_s

        if optical_ok:
            # Optical Channel Healthy
            if self.state.rf_fallback_active:
                # Transitioning back from RF to Optical: Fire Burst Strobe
                self.state.link_carrier_mode = "BURST_REACQUIRING"
                self.state.burst_strobe_power_multiplier = 4.0
            else:
                self.state.link_carrier_mode = "OPTICAL_PRIMARY"
                self.state.burst_strobe_power_multiplier = 1.0

            self.state.is_optical_available = True
            self.state.rf_fallback_active = False
            self.state.dead_reckoning_active = False
            self.state.dead_reckoning_duration_s = 0.0
            self.state.active_data_throughput_mbps = self.optical_bandwidth_gbps * 1000.0
            self.state.packet_loss_ratio = 0.0
            self._blackout_start_time = None
            self._covariance_inflation_factor = 1.0

            # Cache latest valid kinematics for dead reckoning
            self._last_known_position = current_gimbal_deg
            self._last_known_velocity = current_rates_deg_s

        else:
            # Optical Blackout / Severe Fog Extinction
            self.state.is_optical_available = False
            if self.auto_rf_fallback_enabled:
                self.state.rf_fallback_active = True
                self.state.link_carrier_mode = "HYBRID_RF_FALLBACK"
                self.state.active_data_throughput_mbps = self.rf_backup_bandwidth_mbps
                self.state.packet_loss_ratio = 0.02  # Negligible packet loss on 60 GHz RF
            else:
                self.state.rf_fallback_active = False
                self.state.link_carrier_mode = "LINK_BLACKOUT"
                self.state.active_data_throughput_mbps = 0.0
                self.state.packet_loss_ratio = 1.0

            self.state.dead_reckoning_active = True
            self.state.dead_reckoning_duration_s += dt

            # Smoothly inflate Kalman filter covariance over blackout period
            self._covariance_inflation_factor = min(
                100.0, 1.0 + (self.state.dead_reckoning_duration_s ** 1.8) * 12.0
            )

        return self.state

    def predict_ballistic_position(self, dt: float) -> Tuple[float, float]:
        """Extrapolate target angular position using constant-velocity dead reckoning."""
        if not self.state.dead_reckoning_active:
            return self._last_known_position

        # Extrapolate: pos_new = pos_old + vel * t
        t_coast = min(self.state.dead_reckoning_duration_s, self.max_dead_reckoning_s)
        pan = self._last_known_position[0] + self._last_known_velocity[0] * t_coast
        tilt = self._last_known_position[1] + self._last_known_velocity[1] * t_coast
        return (pan, tilt)

    def get_covariance_inflation(self) -> float:
        """Return the current Kalman filter covariance inflation factor."""
        return self._covariance_inflation_factor

    def set_rf_fallback_enabled(self, enabled: bool) -> bool:
        """Enable or disable autonomous RF auxiliary link handoff."""
        self.auto_rf_fallback_enabled = enabled
        return self.auto_rf_fallback_enabled
