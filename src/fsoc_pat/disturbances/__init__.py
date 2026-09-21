"""OptiLign Disturbances & Environmental Noise package.

Provides:
  - Platform jitter (vibrations, rotor harmonics, colored PSD noise)
  - Atmospheric turbulence (beam wander & scintillation fading)
  - Occlusions & line-of-sight dropouts (cloud crossings)
"""

from fsoc_pat.disturbances.atmospheric import AtmosphericConfig, AtmosphericTurbulence
from fsoc_pat.disturbances.occlusion import OcclusionConfig, OcclusionModel
from fsoc_pat.disturbances.platform_jitter import JitterConfig, PlatformJitter

__all__ = [
    "AtmosphericConfig",
    "AtmosphericTurbulence",
    "JitterConfig",
    "OcclusionConfig",
    "OcclusionModel",
    "PlatformJitter",
]
