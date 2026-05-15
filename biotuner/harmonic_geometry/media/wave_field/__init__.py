"""
``wave_field`` family — open-medium coherent superpositions.

Each medium in this family sums travelling / standing waves in free
space (no boundary eigenproblem) and returns the resulting field.

Members
-------
- :class:`Interference` — five paradigms (harmonic, quasicrystal,
  standing_lattice, vortex, sources). Idealized dimensionless wave
  equation.
- :class:`Acoustic` — physically-grounded 2-D bulk acoustic field
  with explicit wave speed and per-source frequency assignment.
  Exposes pressure / intensity / Schlieren / phase observables.
"""

from biotuner.harmonic_geometry.media.wave_field.acoustic import Acoustic
from biotuner.harmonic_geometry.media.wave_field.interference import (
    Interference,
    harmonic_interference_field_2d,
    interference_field_2d,
    quasicrystal_field_2d,
    standing_wave_lattice_2d,
    vortex_field_2d,
)

__all__ = [
    "Acoustic",
    "Interference",
    "harmonic_interference_field_2d",
    "interference_field_2d",
    "quasicrystal_field_2d",
    "standing_wave_lattice_2d",
    "vortex_field_2d",
]
