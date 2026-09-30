"""Unit tests for Dual-Band Hybrid RF/Optical Link Fallback and Ballistic EKF Dead-Reckoning."""

import pytest
from fsoc_pat.link.rf_fallback import DualBandLinkManager, HybridLinkState


def test_dual_band_link_manager_initialization():
    mgr = DualBandLinkManager()
    assert mgr.optical_threshold_dbm == -65.0
    assert mgr.state.is_optical_available is True
    assert mgr.state.rf_fallback_active is False
    assert mgr.state.link_carrier_mode == "OPTICAL_PRIMARY"


def test_optical_healthy_operation():
    mgr = DualBandLinkManager()
    state = mgr.update(
        optical_power_dbm=-55.0,
        current_gimbal_deg=(10.0, 5.0),
        current_rates_deg_s=(1.0, 0.5),
        dt=0.033,
    )
    assert state.is_optical_available is True
    assert state.rf_fallback_active is False
    assert state.link_carrier_mode == "OPTICAL_PRIMARY"
    assert state.active_data_throughput_mbps == 10000.0  # 10 Gbps
    assert state.dead_reckoning_active is False


def test_fog_blackout_triggers_rf_fallback():
    mgr = DualBandLinkManager()
    # Optical drops below -65 dBm (e.g. -72 dBm in severe fog)
    state = mgr.update(
        optical_power_dbm=-72.0,
        current_gimbal_deg=(12.0, 6.0),
        current_rates_deg_s=(2.0, -1.0),
        dt=0.1,
    )
    assert state.is_optical_available is False
    assert state.rf_fallback_active is True
    assert state.link_carrier_mode == "HYBRID_RF_FALLBACK"
    assert state.active_data_throughput_mbps == 100.0  # RF 100 Mbps backup link
    assert state.dead_reckoning_active is True
    assert state.dead_reckoning_duration_s == pytest.approx(0.1, 0.01)

    # Check dead-reckoning extrapolation
    pred_pan, pred_tilt = mgr.predict_ballistic_position(dt=0.1)
    # pred_pan = 12.0 + 2.0 * 0.1 = 12.2
    # pred_tilt = 6.0 + (-1.0) * 0.1 = 5.9
    assert pred_pan == pytest.approx(12.2, 0.05)
    assert pred_tilt == pytest.approx(5.9, 0.05)


def test_burst_strobe_reacquisition_on_clear():
    mgr = DualBandLinkManager()
    # First drop into fog
    mgr.update(-75.0, (10.0, 5.0), (0.0, 0.0), dt=0.5)
    assert mgr.state.rf_fallback_active is True

    # Now clear channel restored (-50.0 dBm)
    state = mgr.update(-50.0, (10.0, 5.0), (0.0, 0.0), dt=0.033)
    assert state.is_optical_available is True
    assert state.rf_fallback_active is False
    assert state.link_carrier_mode == "BURST_REACQUIRING"
    assert state.burst_strobe_power_multiplier == 4.0


def test_rf_fallback_toggle():
    mgr = DualBandLinkManager()
    mgr.set_rf_fallback_enabled(False)
    state = mgr.update(-75.0, (0.0, 0.0), (0.0, 0.0), dt=0.1)
    assert state.rf_fallback_active is False
    assert state.link_carrier_mode == "LINK_BLACKOUT"
    assert state.active_data_throughput_mbps == 0.0
