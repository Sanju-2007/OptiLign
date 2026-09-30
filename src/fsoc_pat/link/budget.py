"""Complete FSOC optical link budget and communication performance calculator."""

from __future__ import annotations

import math

from fsoc_pat.link.beam import GaussianBeam
from fsoc_pat.link.coupling import ApertureCoupling
from fsoc_pat.link.link_types import (
    AtmospherePathConfig,
    LaserConfig,
    LinkBudgetResult,
    LinkState,
    OpticalReceiverConfig,
)


class FSOCLinkBudget:
    """Full-featured Free-Space Optical Communication Link Budget Calculator."""

    def __init__(
        self,
        laser: LaserConfig | None = None,
        receiver: OpticalReceiverConfig | None = None,
        atmosphere: AtmospherePathConfig | None = None,
    ) -> None:
        self.laser = laser or LaserConfig()
        self.receiver = receiver or OpticalReceiverConfig()
        self.atmosphere = atmosphere or AtmospherePathConfig()

        self.beam = GaussianBeam(self.laser)
        self.coupling = ApertureCoupling(self.beam, self.receiver)

    def evaluate(
        self,
        distance_m: float,
        pointing_error_rad: float,
        strehl_ratio: float = 1.0,
        smf_coupling_efficiency: float | None = None,
    ) -> LinkBudgetResult:
        """Calculate complete link budget for given distance, pointing error, and wavefront quality.

        Args:
            distance_m: Range to target (m).
            pointing_error_rad: Total instantaneous pointing error (rad).
            strehl_ratio: Optical Strehl ratio [0, 1] (1.0 = diffraction-limited).
            smf_coupling_efficiency: Optional Single-Mode Fiber coupling [0, 0.815].

        Returns:
            LinkBudgetResult with all communication metrics, loss breakdown, and AO gain.
        """
        dist = max(1.0, float(distance_m))
        theta_err = max(0.0, float(pointing_error_rad))

        # 1. Atmospheric transmission loss (Beer-Lambert law)
        km = dist / 1000.0
        atm_loss_db = self.atmosphere.attenuation_db_per_km * km
        t_atm = 10.0 ** (-atm_loss_db / 10.0)

        # 2. Incident power arriving at range z before aperture
        p_tx_eff = self.laser.power_w * self.laser.tx_efficiency
        p_incident = p_tx_eff * t_atm

        # 3. Aperture / Fiber collected power with Strehl / SMF wavefront coupling
        p_rx = self.coupling.collected_power_w(
            dist, theta_err, p_incident, strehl_ratio, smf_coupling_efficiency
        )
        p_rx = max(1e-18, p_rx)
        p_rx_dbm = 10.0 * math.log10(p_rx * 1000.0)

        # 4. Geometric and Pointing Loss breakdown
        w_z = self.beam.beam_radius_at(dist)
        i_0 = self.beam.peak_irradiance(dist, p_incident)
        pointing_loss_db = self.coupling.pointing_loss_db(theta_err)

        # Free space geometric expansion loss: 20 * log10(w(z) / w0)
        w0 = self.beam.config.beam_waist_w0_m
        free_space_loss_db = 20.0 * math.log10(max(1.0, w_z / w0))
        geom_ratio = min(1.0, max(1e-16, p_rx / max(1e-18, p_incident * self.receiver.rx_efficiency)))

        # 5. Electrical Signal-to-Noise Ratio (SNR)
        # Noise power P_noise = NEP * sqrt(B)
        noise_pwr_w = self.receiver.noise_equivalent_power_w_rt_hz * math.sqrt(self.receiver.data_rate_bps)
        snr_linear = (p_rx / max(1e-19, noise_pwr_w)) ** 2
        snr_db = 10.0 * math.log10(max(1e-6, snr_linear))

        # 6. Theoretical Bit Error Rate (BER) for On-Off Keying (OOK)
        # Q = sqrt(SNR) / 2; BER = 0.5 * erfc(Q / sqrt(2))
        q_factor = math.sqrt(max(0.0, snr_linear)) / 2.0
        try:
            ber = 0.5 * math.erfc(q_factor / math.sqrt(2.0))
        except (OverflowError, ValueError):
            ber = 1e-15 if q_factor > 8.0 else 0.5
        ber = max(1e-15, min(0.5, ber))

        # 7. Optical Link Margin
        link_margin_db = p_rx_dbm - self.receiver.detector_sensitivity_dbm

        # 8. Operational Link State
        if link_margin_db >= 3.0 and ber <= 1e-6:
            state = LinkState.NOMINAL
        elif link_margin_db >= 0.0:
            state = LinkState.DEGRADED
        else:
            state = LinkState.OUTAGE

        eff_coupling = smf_coupling_efficiency if smf_coupling_efficiency is not None else (0.8145 * strehl_ratio)
        ao_gain_db = 10.0 * math.log10(max(1e-3, eff_coupling) / 0.05) if strehl_ratio > 0.0 else 0.0

        return LinkBudgetResult(
            distance_m=dist,
            pointing_error_rad=theta_err,
            beam_radius_at_target_m=w_z,
            peak_irradiance_w_m2=i_0,
            rx_power_w=p_rx,
            rx_power_dbm=p_rx_dbm,
            free_space_loss_db=free_space_loss_db,
            atmospheric_loss_db=atm_loss_db,
            pointing_loss_db=pointing_loss_db,
            coupling_efficiency=geom_ratio,
            snr_db=snr_db,
            ber=ber,
            link_margin_db=link_margin_db,
            state=state,
            strehl_ratio=strehl_ratio,
            smf_coupling_efficiency=eff_coupling,
            ao_gain_db=ao_gain_db,
        )

