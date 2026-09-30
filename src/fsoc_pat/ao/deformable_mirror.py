"""Continuous face-sheet piezoelectric Deformable Mirror (DM) actuator model.

Simulates:
  - 37-actuator zonal arrangement in concentric hexagonal/circular rings
  - Gaussian influence functions mapping actuator commands to 2D mirror surface
  - Double-pass reflective optical phase shift (Delta_phi = 2 * k * Delta_z)
  - Actuator stroke limits (+/- 3.5 um), saturation, and leaky integrator dynamics
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from fsoc_pat.ao.ao_types import DeformableMirrorConfig


@dataclass
class DMActuator:
    """A single piezoelectric actuator behind the DM face-sheet."""

    index: int
    x: float  # Normalized x-coordinate in [-1, 1]
    y: float  # Normalized y-coordinate in [-1, 1]
    ring: int  # Ring number (0 = center, 1, 2, 3)


class DeformableMirror:
    """Continuous face-sheet Deformable Mirror model."""

    def __init__(
        self,
        config: DeformableMirrorConfig | None = None,
        resolution: int = 64,
        wavelength_m: float = 1550e-9,
    ) -> None:
        self.config = config or DeformableMirrorConfig()
        self.resolution = resolution
        self.wavelength_m = wavelength_m

        # Build actuator grid
        self.actuators: list[DMActuator] = []
        self._build_actuator_layout()

        self.num_actuators = len(self.actuators)
        self.commands_um = np.zeros(self.num_actuators, dtype=np.float64)
        self.voltages = np.zeros(self.num_actuators, dtype=np.float64)

        # Precompute 2D Gaussian influence function matrix: shape (num_actuators, res, res)
        self._precompute_influence_functions()

    def _build_actuator_layout(self) -> None:
        """Place actuators in concentric circular rings (1 + 6 + 12 + 18 = 37)."""
        idx = 0
        # Ring 0: Center actuator
        self.actuators.append(DMActuator(index=idx, x=0.0, y=0.0, ring=0))
        idx += 1

        # Concentric rings
        rings = [
            (6, 0.32),   # Ring 1: 6 actuators at r = 0.32
            (12, 0.64),  # Ring 2: 12 actuators at r = 0.64
            (18, 0.94),  # Ring 3: 18 actuators at r = 0.94
        ]

        for n_pts, radius in rings:
            d_theta = 2.0 * math.pi / n_pts
            for k in range(n_pts):
                angle = k * d_theta
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                self.actuators.append(DMActuator(index=idx, x=x, y=y, ring=len(rings)))
                idx += 1

    def _precompute_influence_functions(self) -> None:
        """Precompute the 2D influence surface for each actuator."""
        lin = np.linspace(-1.0, 1.0, self.resolution)
        x_grid, y_grid = np.meshgrid(lin, lin)

        # Inter-actuator spacing is approx 0.32 in normalized coords
        # Gaussian width sigma chosen so coupling to neighbor is ~ 15%
        # exp(-0.32^2 / (2 * sigma^2)) = 0.15 => 2 * sigma^2 = 0.32^2 / ln(1/0.15) ≈ 0.054
        c = max(0.05, min(0.5, self.config.influence_coupling))
        # sigma^2 = -d^2 / (2 * ln(c))
        d_spacing = 0.32
        sigma2 = -(d_spacing ** 2) / (2.0 * math.log(c))
        sigma = math.sqrt(sigma2)

        self.influence_matrix = np.zeros(
            (self.num_actuators, self.resolution, self.resolution), dtype=np.float64
        )

        for act in self.actuators:
            r2 = (x_grid - act.x) ** 2 + (y_grid - act.y) ** 2
            inf_map = np.exp(-r2 / (2.0 * (sigma ** 2)))
            self.influence_matrix[act.index] = inf_map

    def reset(self) -> None:
        """Reset all actuators to flat zero stroke."""
        self.commands_um.fill(0.0)
        self.voltages.fill(0.0)

    def apply_delta_commands(self, delta_u_um: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply integrator increment to actuator strokes with leak and saturation limits.

        Leaky integrator equation:
          u_{t+1} = leak_factor * u_t + loop_gain * delta_u

        Args:
            delta_u_um: Increment array of actuator commands in micrometers.

        Returns:
            Current actuator stroke array in micrometers.
        """
        leak = self.config.leak_factor
        gain = self.config.loop_gain
        max_stroke = self.config.max_stroke_um

        # Update integrator
        self.commands_um = leak * self.commands_um + gain * delta_u_um

        # Clamp to physical stroke limits (+/- max_stroke)
        self.commands_um = np.clip(self.commands_um, -max_stroke, max_stroke)
        return self.commands_um.copy()

    def set_commands(self, commands_um: Sequence[float]) -> None:
        """Directly set actuator commands clamped to maximum stroke."""
        max_stroke = self.config.max_stroke_um
        n = min(len(commands_um), self.num_actuators)
        self.commands_um[:n] = np.clip(commands_um[:n], -max_stroke, max_stroke)

    def get_surface_displacement_um(self) -> NDArray[np.float64]:
        """Calculate the 2D physical surface displacement of the DM face-sheet (um).

        Returns:
            2D array of shape (resolution, resolution) with surface height in um.
        """
        # Linear superposition of actuator influence functions: H(x, y) = sum u_k * I_k(x, y)
        surface = np.tensordot(self.commands_um, self.influence_matrix, axes=(0, 0))
        return surface

    def get_optical_phase_radians(self) -> NDArray[np.float64]:
        """Calculate the 2D optical phase shift introduced by reflection off the DM.

        A reflective mirror surface with physical elevation delta_z introduces
        an Optical Path Difference OPD = 2 * delta_z.
        Phase shift: phi = 2 * (2 * pi / lambda) * delta_z

        Returns:
            2D array of phase shift in radians.
        """
        surface_um = self.get_surface_displacement_um()
        surface_m = surface_um * 1e-6
        k = 2.0 * math.pi / self.wavelength_m
        phase_rad = 2.0 * k * surface_m
        return phase_rad
