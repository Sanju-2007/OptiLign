"""Aperture collection, coupling efficiency, and pointing loss models."""

from __future__ import annotations

import math

from fsoc_pat.link.beam import GaussianBeam
from fsoc_pat.link.link_types import OpticalReceiverConfig


class ApertureCoupling:
    """Computes aperture power collection and pointing-induced misalignment loss."""

    def __init__(
        self,
        beam: GaussianBeam,
        receiver: OpticalReceiverConfig | None = None,
    ) -> None:
        self.beam = beam
        self.receiver = receiver or OpticalReceiverConfig()

    def pointing_loss_linear(self, pointing_error_rad: float) -> float:
        """Compute fraction of peak on-axis power received due to pointing offset.

        L_p = exp(-2 * (theta / theta_div)^2)

        Args:
            pointing_error_rad: Angular pointing error between laser beam and receiver (rad).

        Returns:
            Linear loss factor in (0, 1]. 1.0 = perfect alignment.
        """
        theta_div = self.beam.divergence_half_angle_rad
        if theta_div <= 0.0:
            return 1.0
        return math.exp(-2.0 * ((pointing_error_rad / theta_div) ** 2))

    def pointing_loss_db(self, pointing_error_rad: float) -> float:
        """Compute pointing loss in decibels (positive value).

        L_p (dB) = 8.6859 * (theta / theta_div)^2

        Args:
            pointing_error_rad: Angular pointing error (rad).

        Returns:
            Pointing loss in dB (e.g. 3.0 dB).
        """
        theta_div = self.beam.divergence_half_angle_rad
        if theta_div <= 0.0:
            return 0.0
        return 8.6858896 * ((pointing_error_rad / theta_div) ** 2)

    def collected_power_w(
        self,
        distance_m: float,
        pointing_error_rad: float,
        incident_power_w: float,
    ) -> float:
        """Compute total optical power collected by the receiver aperture (W).

        Exact enclosed power scaled by off-axis pointing offset and receiver optics efficiency:
            P_rx = P_inc * (1 - exp(-2*(a_rx/w)^2)) * exp(-2*(r_off/w)^2) * eta_rx

        Args:
            distance_m: Propagation distance (m).
            pointing_error_rad: Angular pointing error (rad).
            incident_power_w: Power arriving at distance z before aperture (W).

        Returns:
            Collected optical power in Watts.
        """
        w_z = self.beam.beam_radius_at(distance_m)
        a_rx = self.receiver.aperture_diameter_m / 2.0
        r_offset = distance_m * pointing_error_rad

        # On-axis power fraction captured by aperture
        eta_aperture = 1.0 - math.exp(-2.0 * ((a_rx / w_z) ** 2))
        p_on_axis = incident_power_w * eta_aperture

        # Off-axis pointing loss factor
        eta_pointing = math.exp(-2.0 * ((r_offset / w_z) ** 2))

        # Collected optical power delivered to detector
        p_collected = p_on_axis * eta_pointing * self.receiver.rx_efficiency
        return max(1e-18, p_collected)
