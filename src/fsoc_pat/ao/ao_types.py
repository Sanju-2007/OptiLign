"""Adaptive Optics (AO) data structures, configurations, and enums.

Defines the mathematical and physical parameters for:
  - Zernike polynomial expansion (Noll / OSA indices)
  - Shack-Hartmann Wavefront Sensor (SH-WFS)
  - Continuous face-sheet Deformable Mirror (DM)
  - Atmospheric turbulence phase screens and closed-loop telemetry
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np
from numpy.typing import NDArray


class ZernikeIndex(int, Enum):
    """Standard Noll indexing for the first 22 Zernike polynomial modes.

    Ref: Noll, R. J. (1976). "Zernike polynomials and atmospheric turbulence."
    J. Opt. Soc. Am. 66, 207-211.
    """

    PISTON = 1               # n=0, m=0
    TIP_X = 2                # n=1, m=1
    TILT_Y = 3               # n=1, m=-1
    DEFOCUS = 4              # n=2, m=0
    ASTIGMATISM_OBLIQUE = 5  # n=2, m=-2 (45 deg)
    ASTIGMATISM_VERTICAL = 6 # n=2, m=2  (0 deg)
    COMA_Y = 7               # n=3, m=-1
    COMA_X = 8               # n=3, m=1
    TREFOIL_Y = 9            # n=3, m=-3
    TREFOIL_X = 10           # n=3, m=3
    SPHERICAL_PRIMARY = 11   # n=4, m=0
    SEC_ASTIGMATISM_OBL = 12 # n=4, m=2
    SEC_ASTIGMATISM_VERT= 13 # n=4, m=-2
    QUADRAFOIL_Y = 14        # n=4, m=-4
    QUADRAFOIL_X = 15        # n=4, m=4
    SEC_COMA_X = 16          # n=5, m=1
    SEC_COMA_Y = 17          # n=5, m=-1
    SEC_TREFOIL_Y = 18       # n=5, m=-3
    SEC_TREFOIL_X = 19       # n=5, m=3
    PENTAFOIL_Y = 20         # n=5, m=-5
    PENTAFOIL_X = 21         # n=5, m=5
    TERTIARY_SPHERICAL = 22  # n=6, m=0


# Mapping of Noll index j -> (radial degree n, azimuthal frequency m, name)
NOLL_TABLE: dict[int, tuple[int, int, str]] = {
    1: (0, 0, "Piston"),
    2: (1, 1, "Tip X"),
    3: (1, -1, "Tilt Y"),
    4: (2, 0, "Defocus"),
    5: (2, -2, "Astigmatism 45°"),
    6: (2, 2, "Astigmatism 0°"),
    7: (3, -1, "Coma Y"),
    8: (3, 1, "Coma X"),
    9: (3, -3, "Trefoil Y"),
    10: (3, 3, "Trefoil X"),
    11: (4, 0, "Primary Spherical"),
    12: (4, 2, "Secondary Astigmatism Y"),
    13: (4, -2, "Secondary Astigmatism X"),
    14: (4, -4, "Quadrafoil Y"),
    15: (4, 4, "Quadrafoil X"),
    16: (5, 1, "Secondary Coma X"),
    17: (5, -1, "Secondary Coma Y"),
    18: (5, -3, "Secondary Trefoil Y"),
    19: (5, 3, "Secondary Trefoil X"),
    20: (5, -5, "Pentafoil Y"),
    21: (5, 5, "Pentafoil X"),
    22: (6, 0, "Secondary Spherical"),
}


@dataclass
class WavefrontMap:
    """Two-dimensional optical phase map defined over a circular pupil aperture.

    Attributes:
        phase: 2D array of phase values in radians. Shape: (grid_size, grid_size).
        pupil_mask: Boolean 2D array indicating pixels inside circular optical aperture.
        diameter_m: Physical diameter of receiver telescope pupil (m).
        wavelength_m: Laser carrier wavelength (m). Default 1550 nm (C-band).
        rms_wfe_rad: Root-Mean-Square wavefront error in radians across the pupil.
        pv_wfe_rad: Peak-to-Valley wavefront error in radians across the pupil.
    """

    phase: NDArray[np.float64]
    pupil_mask: NDArray[np.bool_]
    diameter_m: float = 0.15
    wavelength_m: float = 1550e-9

    @property
    def grid_size(self) -> int:
        return int(self.phase.shape[0])

    @property
    def rms_wfe_rad(self) -> float:
        valid = self.phase[self.pupil_mask]
        if valid.size == 0:
            return 0.0
        # Zero-mean WFE RMS
        return float(np.std(valid))

    @property
    def rms_wfe_nm(self) -> float:
        """RMS wavefront error in nanometers (optical path difference OPD)."""
        # OPD = (phase * lambda) / (2 * pi)
        return self.rms_wfe_rad * (self.wavelength_m / (2.0 * np.pi)) * 1e9

    @property
    def pv_wfe_rad(self) -> float:
        valid = self.phase[self.pupil_mask]
        if valid.size == 0:
            return 0.0
        return float(np.ptp(valid))

    @property
    def strehl_ratio(self) -> float:
        """Maréchal approximation of Strehl ratio: S ≈ exp(-sigma_phi^2).

        Accurate for moderate aberrations (sigma_phi <= 1.0 rad), with smooth
        exponential roll-off for large aberrations.
        """
        sigma2 = self.rms_wfe_rad ** 2
        return float(np.exp(-sigma2))


@dataclass
class SHWFSConfig:
    """Shack-Hartmann Wavefront Sensor (SH-WFS) configuration.

    Attributes:
        grid_dim: Number of lenslets across pupil diameter (e.g. 8x8 or 16x16 array).
        lenslet_focal_length_m: Focal length of micro-lenslet array (m).
        pixel_pitch_m: Sensor detector pixel size (m).
        subaperture_threshold: Minimum area fraction of subaperture inside pupil to be active.
        measurement_noise_std_rad: Centroid measurement noise standard deviation in radians of tilt.
        wavelength_m: Optical wavelength (m).
    """

    grid_dim: int = 8  # 8x8 lenslet grid
    lenslet_focal_length_m: float = 0.015  # 15 mm
    pixel_pitch_m: float = 5.5e-6  # 5.5 um
    subaperture_threshold: float = 0.50
    measurement_noise_std_rad: float = 0.02
    wavelength_m: float = 1550e-9


@dataclass
class DeformableMirrorConfig:
    """Continuous face-sheet piezo Deformable Mirror (DM) configuration.

    Attributes:
        num_actuators: Total active actuators (default 37 arranged in concentric rings).
        max_stroke_um: Mechanical actuator stroke limit in micrometers (+/- 3.5 um).
        influence_coupling: Gaussian influence function coupling factor between adjacent actuators.
        damping_ratio: Actuator dynamics damping ratio.
        loop_gain: Closed-loop integrator gain (Ki).
        leak_factor: Leaky integrator forgetting factor (0.95 - 0.999) to prevent actuator drift.
        bandwidth_hz: Closed-loop correction bandwidth in Hertz.
    """

    num_actuators: int = 37
    max_stroke_um: float = 3.5  # +/- 3.5 um stroke (~ 4.5 waves @ 1550nm)
    influence_coupling: float = 0.15
    damping_ratio: float = 0.707
    loop_gain: float = 0.45
    leak_factor: float = 0.995
    bandwidth_hz: float = 800.0


@dataclass
class AOConfig:
    """Master Adaptive Optics subsystem configuration.

    Attributes:
        enabled: Whether AO correction is actively applied.
        pupil_diameter_m: Telescope entrance pupil diameter (m).
        wavelength_m: Communication laser carrier wavelength (m).
        grid_resolution: Resolution of simulation phase screen grid (e.g. 64x64).
        max_zernike_modes: Number of Zernike modes to correct (default 22: up to 5th radial order).
        shwfs: Shack-Hartmann sensor configuration.
        dm: Deformable mirror configuration.
        fiber_core_radius_um: Target Single-Mode Fiber (SMF) core radius (um). Default 4.5 um (SMF-28).
        fiber_mfd_um: Mode Field Diameter of fiber (um). Default 10.4 um @ 1550nm.
    """

    enabled: bool = True
    pupil_diameter_m: float = 0.15
    wavelength_m: float = 1550e-9
    grid_resolution: int = 64
    max_zernike_modes: int = 22
    shwfs: SHWFSConfig = field(default_factory=SHWFSConfig)
    dm: DeformableMirrorConfig = field(default_factory=DeformableMirrorConfig)
    fiber_core_radius_um: float = 4.5
    fiber_mfd_um: float = 10.4


@dataclass
class AOTelemetry:
    """Instantaneous telemetry packet for Adaptive Optics correction.

    Attributes:
        timestamp_s: Simulation timestamp (s).
        raw_wfe_rms_rad: Uncorrected turbulent wavefront error (rad RMS).
        raw_wfe_rms_nm: Uncorrected wavefront error (nm RMS).
        corrected_wfe_rms_rad: Residual wavefront error after DM correction (rad RMS).
        corrected_wfe_rms_nm: Residual wavefront error after DM correction (nm RMS).
        strehl_ratio_pre: Strehl ratio before AO (0 to 1).
        strehl_ratio_post: Strehl ratio after AO (0 to 1).
        smf_coupling_pre: Single-mode fiber coupling efficiency before AO (0 to 1).
        smf_coupling_post: Single-mode fiber coupling efficiency after AO (0 to 1).
        ao_optical_gain_db: Net optical power coupling gain provided by AO (dB).
        residual_variance_rad2: Residual wavefront variance (rad^2).
        zernike_coefficients_pre: List of uncorrected Zernike amplitudes (rad).
        zernike_coefficients_post: List of residual Zernike amplitudes (rad).
        dm_stroke_used_um: Maximum actuator stroke currently engaged (um).
        ao_loop_closed: Whether closed-loop correction is running.
    """

    timestamp_s: float
    raw_wfe_rms_rad: float
    raw_wfe_rms_nm: float
    corrected_wfe_rms_rad: float
    corrected_wfe_rms_nm: float
    strehl_ratio_pre: float
    strehl_ratio_post: float
    smf_coupling_pre: float
    smf_coupling_post: float
    ao_optical_gain_db: float
    residual_variance_rad2: float
    zernike_coefficients_pre: list[float]
    zernike_coefficients_post: list[float]
    dm_stroke_used_um: float
    ao_loop_closed: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert telemetry to serializable dictionary."""
        return {
            "timestamp_s": round(self.timestamp_s, 4),
            "raw_wfe_rms_rad": round(self.raw_wfe_rms_rad, 4),
            "raw_wfe_rms_nm": round(self.raw_wfe_rms_nm, 2),
            "corrected_wfe_rms_rad": round(self.corrected_wfe_rms_rad, 4),
            "corrected_wfe_rms_nm": round(self.corrected_wfe_rms_nm, 2),
            "strehl_ratio_pre": round(self.strehl_ratio_pre, 4),
            "strehl_ratio_post": round(self.strehl_ratio_post, 4),
            "smf_coupling_pre": round(self.smf_coupling_pre, 4),
            "smf_coupling_post": round(self.smf_coupling_post, 4),
            "ao_optical_gain_db": round(self.ao_optical_gain_db, 2),
            "residual_variance_rad2": round(self.residual_variance_rad2, 4),
            "dm_stroke_used_um": round(self.dm_stroke_used_um, 3),
            "ao_loop_closed": self.ao_loop_closed,
        }
