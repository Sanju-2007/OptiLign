"""Unit tests for Cooperative Master-Slave Handshake Protocol."""

import pytest
from fsoc_pat.pat.cooperative_handshake import CooperativeHandshakeManager


def test_handshake_initialization():
    mgr = CooperativeHandshakeManager()
    assert mgr.state.active is False
    assert mgr.state.phase == 0
    assert mgr.state.mutual_lock_achieved is False


def test_handshake_trigger_and_phases():
    mgr = CooperativeHandshakeManager()
    state = mgr.trigger_handshake()
    assert state.active is True
    assert state.phase == 1
    # Check beam divergence expansion to 15 mrad
    assert state.terminal_a_divergence_urad == 15000.0

    # Step through Phase 1 into Phase 2
    mgr.step(dt=0.36)
    assert mgr.state.phase == 2

    # Step into Phase 3 (Slave reply: 1 mrad)
    mgr.step(dt=0.16)
    assert mgr.state.phase == 3
    assert mgr.state.terminal_b_divergence_urad == 1000.0

    # Step into Phase 4 (Synchronous beam squeeze to 50 µrad)
    mgr.step(dt=0.20)
    assert mgr.state.phase == 4
    assert mgr.state.terminal_a_divergence_urad == 50.0
    assert mgr.state.terminal_b_divergence_urad == 50.0

    # Complete handshake sequence (<0.85s total)
    mgr.step(dt=0.15)
    assert mgr.state.active is False
    assert mgr.state.mutual_lock_achieved is True
    assert mgr.state.handshake_duration_s < 1.0


def test_telemetry_dict():
    mgr = CooperativeHandshakeManager()
    mgr.trigger_handshake()
    mgr.step(dt=0.1)
    telem = mgr.get_telemetry_dict()
    assert telem["handshake_active"] is True
    assert telem["handshake_phase"] == 1
    assert "master_spiral_radius_deg" in telem
