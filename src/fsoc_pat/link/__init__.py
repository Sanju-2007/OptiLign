"""OptiLign FSOC Optical Link & Communication Budget package.

Provides:
  - Fundamental Gaussian beam propagation (Rayleigh range, divergence, irradiance profile)
  - Aperture geometric coupling & pointing loss calculation
  - Full FSOC communication link budget (received power, SNR, BER, margin)
"""

from fsoc_pat.link.beam import GaussianBeam
from fsoc_pat.link.budget import FSOCLinkBudget
from fsoc_pat.link.coupling import ApertureCoupling
from fsoc_pat.link.link_types import (
    AtmospherePathConfig,
    LaserConfig,
    LinkBudgetResult,
    LinkState,
    OpticalReceiverConfig,
)

__all__ = [
    "ApertureCoupling",
    "AtmospherePathConfig",
    "FSOCLinkBudget",
    "GaussianBeam",
    "LaserConfig",
    "LinkBudgetResult",
    "LinkState",
    "OpticalReceiverConfig",
]
