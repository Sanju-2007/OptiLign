"""Asymmetric Cooperative Master-Slave Handshake Protocol (CCSDS Standard).

Resolves the blind search paradox where two moving optical terminals are mutually unaligned:
1. Phase 1: Terminal A (Master) expands beacon divergence to 15 mrad (Floodlight)
   and executes a fast deterministic spiral sweep over the uncertainty cone.
2. Phase 2: Terminal B (Slave) stares in wide-field CMOS mode, detecting the beacon in <0.4s.
3. Phase 3: Terminal B fires a collimated 1 mrad reply pulse along the reciprocal vector.
4. Phase 4: Mutual Fine Lock — both terminals contract to 50 µrad divergence (diffraction-limited)
   and engage piezo Fast Steering Mirrors (FSM). Total handshake time: < 0.8 seconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass
class HandshakeState:
    """Telemetry representation of the reciprocal optical handshake."""
    active: bool = False
    phase: int = 0  # 0: Idle, 1: Master Floodlight, 2: Slave Detect, 3: Slave Reply, 4: Mutual Lock
    phase_name: str = "IDLE"
    elapsed_time_s: float = 0.0
    terminal_a_divergence_urad: float = 50.0  # Default communication divergence (50 µrad)
    terminal_b_divergence_urad: float = 50.0
    master_scan_angle_deg: float = 0.0
    master_spiral_radius_deg: float = 0.0
    mutual_lock_achieved: bool = False
    handshake_duration_s: float = 0.0
    link_margin_boost_db: float = 0.0


class CooperativeHandshakeManager:
    """Executes the automated 4-phase master-slave mutual acquisition handshake."""

    PHASE_NAMES = {
        0: "IDLE / MUTUAL LOCK",
        1: "PHASE 1: MASTER FLOODLIGHT (15 mrad Spiral)",
        2: "PHASE 2: SLAVE STARE DETECT (<0.4s)",
        3: "PHASE 3: RETRO-PENCIL REPLY (1 mrad)",
        4: "PHASE 4: MUTUAL FINE LOCK (50 µrad FSM Closed)",
    }

    def __init__(self, spiral_rate_hz: float = 4.0, max_cone_deg: float = 1.0) -> None:
        self.spiral_rate_hz = spiral_rate_hz
        self.max_cone_deg = max_cone_deg
        self.state = HandshakeState()
        self._phase_duration = {
            1: 0.35,  # Phase 1: 0.35s floodlight spiral sweep
            2: 0.15,  # Phase 2: 0.15s slave CMOS wavefront detection
            3: 0.18,  # Phase 3: 0.18s slave pencil-beam return response
            4: 0.12,  # Phase 4: 0.12s synchronous beam squeeze to 50 µrad
        }

    def trigger_handshake(self) -> HandshakeState:
        """Initiate the cooperative master-slave acquisition sequence."""
        self.state.active = True
        self.state.phase = 1
        self.state.phase_name = self.PHASE_NAMES[1]
        self.state.elapsed_time_s = 0.0
        self.state.mutual_lock_achieved = False
        self.state.handshake_duration_s = 0.0
        # Phase 1: Expand Master to 15 mrad (15,000 µrad) floodlight
        self.state.terminal_a_divergence_urad = 15000.0
        self.state.terminal_b_divergence_urad = 5000.0  # Slave in wide stare
        self.state.link_margin_boost_db = 6.0
        return self.state

    def step(self, dt: float = 0.033) -> HandshakeState:
        """Step the handshake state machine."""
        if not self.state.active:
            return self.state

        self.state.elapsed_time_s += dt

        # Phase 1: Master Floodlight Spiral Sweep (0.0s -> 0.35s)
        if self.state.phase == 1:
            t = self.state.elapsed_time_s
            progress = min(1.0, t / self._phase_duration[1])
            self.state.master_spiral_radius_deg = progress * self.max_cone_deg
            self.state.master_scan_angle_deg = (t * self.spiral_rate_hz * 360.0) % 360.0

            if t >= self._phase_duration[1]:
                self.state.phase = 2
                self.state.phase_name = self.PHASE_NAMES[2]

        # Phase 2: Slave Stare Detect (0.35s -> 0.50s)
        elif self.state.phase == 2:
            t_phase = self.state.elapsed_time_s - self._phase_duration[1]
            if t_phase >= self._phase_duration[2]:
                self.state.phase = 3
                self.state.phase_name = self.PHASE_NAMES[3]
                # Slave fires 1 mrad (1000 µrad) collimated response
                self.state.terminal_b_divergence_urad = 1000.0

        # Phase 3: Retro-Pencil Reply (0.50s -> 0.68s)
        elif self.state.phase == 3:
            t_phase = self.state.elapsed_time_s - (self._phase_duration[1] + self._phase_duration[2])
            if t_phase >= self._phase_duration[3]:
                self.state.phase = 4
                self.state.phase_name = self.PHASE_NAMES[4]
                # Mutual beam contraction to 50 µrad
                self.state.terminal_a_divergence_urad = 50.0
                self.state.terminal_b_divergence_urad = 50.0

        # Phase 4: Synchronous Fine Tracking Lock Closed (0.68s -> 0.80s)
        elif self.state.phase == 4:
            t_total = self._phase_duration[1] + self._phase_duration[2] + self._phase_duration[3] + self._phase_duration[4]
            if self.state.elapsed_time_s >= t_total:
                self.state.active = False
                self.state.mutual_lock_achieved = True
                self.state.handshake_duration_s = round(self.state.elapsed_time_s, 3)
                self.state.phase_name = f"MUTUAL LOCK ACHIEVED ({self.state.handshake_duration_s:.2f}s)"

        return self.state

    def get_telemetry_dict(self) -> Dict[str, Any]:
        """Return serialized telemetry dictionary."""
        return {
            "handshake_active": self.state.active,
            "handshake_phase": self.state.phase,
            "handshake_phase_name": self.state.phase_name,
            "handshake_elapsed_s": round(self.state.elapsed_time_s, 3),
            "handshake_duration_s": self.state.handshake_duration_s,
            "mutual_lock_achieved": self.state.mutual_lock_achieved,
            "terminal_a_divergence_urad": self.state.terminal_a_divergence_urad,
            "terminal_b_divergence_urad": self.state.terminal_b_divergence_urad,
            "master_spiral_radius_deg": round(self.state.master_spiral_radius_deg, 3),
        }

    get_telemetry = get_telemetry_dict
