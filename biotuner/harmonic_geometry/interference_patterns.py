"""
Compatibility shim — the real implementation lives in
:mod:`biotuner.harmonic_geometry.media.wave_field.interference`.

This module re-exports the historical interference paradigm functions so
existing imports keep working after the migration to the family-based
:mod:`biotuner.harmonic_geometry.media` layout. Prefer the new path in
new code.
"""

from biotuner.harmonic_geometry.media.wave_field.interference import (  # noqa: F401
    _emitter_positions,
    harmonic_interference_field_2d,
    interference_field_2d,
    quasicrystal_field_2d,
    standing_wave_lattice_2d,
    vortex_field_2d,
)

__all__ = [
    "harmonic_interference_field_2d",
    "interference_field_2d",
    "quasicrystal_field_2d",
    "standing_wave_lattice_2d",
    "vortex_field_2d",
]
