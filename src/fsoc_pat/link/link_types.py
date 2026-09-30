"""Data types and configurations for FSOC optical link budget modeling."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto


class LinkState(Enum):
    """Operational communication link status."""

    NOMINAL = auto()   # Margin >= 3 dB; BER <= 1e-9 (clean 1Gbps comms)
    DEGRADED = auto()  # 0 <= Margin < 3 dB (packet loss / retransmission)
    OUTAGE = auto()    # Margin < 0 dB (signal below receiver threshold)


@dataclass
class LaserConfig:
    """Optical laser transmitter configuration.

    Attributes:
        wavelength_m: Laser optical wavelength in meters (default 1550 nm).
        power_w: Average optical output power (W), default 0.5 W (27 dBm).
        beam_waist_w0_m: Initial 1/e^2 beam waist radius at collimator exit (m).
        m2_factor: Beam quality factor M^2 (1.0 = ideal diffraction-limited TEM00).
        tx_efficiency: Optical transmission efficiency of the launch optics.
    """

    wavelength_m: float = 1550e-9
    power_w: float = 0.5
    beam_waist_w0_m: float = 0.0005  # 0.5 mm waist radius (divergence ~ 1.1 mrad)
    m2_factor: float = 1.1
    tx_efficiency: float = 0.85

    @property
    def power_dbm(self) -> float:
        """Transmit power in dBm."""
        return 10.0 * math.log10(max(1e-12, self.power_w) * 1000.0)


@dataclass
class OpticalReceiverConfig:
    """Optical receiver telescope and detector configuration.

    Attributes:
        aperture_diameter_m: Telescope clear aperture diameter (m).
        rx_efficiency: Optical transmission of receiver filters & lenses.
        detector_sensitivity_dbm: Minimum received optical power for target BER (dBm).
        responsivity_a_w: Photodetector responsivity (A/W) at laser wavelength.
        noise_equivalent_power_w_rt_hz: Detector NEP in W/sqrt(Hz).
        data_rate_bps: Communication channel data rate (bits/second).
    """

    aperture_diameter_m: float = 0.10  # 10 cm collector
    rx_efficiency: float = 0.70
    detector_sensitivity_dbm: float = -38.0  # -38 dBm at 1 Gbps
    responsivity_a_w: float = 0.90
    noise_equivalent_power_w_rt_hz: float = 5e-12
    data_rate_bps: float = 1e9  # 1 Gbps

    @property
    def aperture_area_m2(self) -> float:
        """Receiver aperture collecting area."""
        r = self.aperture_diameter_m / 2.0
        return math.pi * (r ** 2)


@dataclass
class AtmospherePathConfig:
    """Atmospheric propagation channel configuration.

    Attributes:
        attenuation_db_per_km: Specific atmospheric absorption & scattering loss.
            Typical values: Clear air = 0.5 dB/km, Haze = 2.0 dB/km, Fog = 20+ dB/km.
    """

    attenuation_db_per_km: float = 0.5


@dataclass
class LinkBudgetResult:
    """Comprehensive optical link budget breakdown at a single timestep.

    Attributes:
        distance_m: Distance between transmitter and receiver (m).
        pointing_error_rad: Total instantaneous angular pointing error (rad).
        beam_radius_at_target_m: 1/e^2 beam spot radius at receiver plane (m).
        peak_irradiance_w_m2: On-axis optical irradiance at target range (W/m^2).
        rx_power_w: Total optical power collected by receiver aperture (W).
        rx_power_dbm: Received optical power in dBm.
        free_space_loss_db: Geometric path expansion loss (dB).
        atmospheric_loss_db: Atmospheric Beer-Lambert attenuation (dB).
        pointing_loss_db: Power loss due to off-axis pointing error (dB).
        coupling_efficiency: Fraction of beam intercepted by aperture.
        snr_db: Electrical Signal-to-Noise Ratio (dB).
        ber: Theoretical Bit Error Rate for On-Off Keying (OOK).
        link_margin_db: Margin relative to detector sensitivity (dB).
        state: Operational link state (NOMINAL, DEGRADED, OUTAGE).
    """

    distance_m: float
    pointing_error_rad: float
    beam_radius_at_target_m: float
    peak_irradiance_w_m2: float
    rx_power_w: float
    rx_power_dbm: float
    free_space_loss_db: float
    atmospheric_loss_db: float
    pointing_loss_db: float
    coupling_efficiency: float
    snr_db: float
    ber: float
    link_margin_db: float
    state: LinkState
    strehl_ratio: float = 1.0
    smf_coupling_efficiency: float = 0.8145
    ao_gain_db: float = 0.0

