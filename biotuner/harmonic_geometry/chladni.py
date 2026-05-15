"""
Compatibility shim — the real implementation lives in
:mod:`biotuner.harmonic_geometry.media.eigenmode.rigid_plate`.

This module re-exports the historical ``chladni_*`` functional API so
that existing imports like ``from biotuner.harmonic_geometry.chladni
import chladni_field_rectangular`` keep working after the migration to
the family-based :mod:`biotuner.harmonic_geometry.media` layout. Prefer
the new path in new code.
"""

from biotuner.harmonic_geometry.media.eigenmode.rigid_plate import (  # noqa: F401
    chladni_field_3d_box,
    chladni_field_circular,
    chladni_field_polygon,
    chladni_field_rectangular,
    chladni_from_input,
    chladni_nodal_lines,
    chladni_nodal_surfaces,
    chladni_temporal,
    ratios_to_modes,
)

__all__ = [
    "chladni_field_3d_box",
    "chladni_field_circular",
    "chladni_field_polygon",
    "chladni_field_rectangular",
    "chladni_from_input",
    "chladni_nodal_lines",
    "chladni_nodal_surfaces",
    "chladni_temporal",
    "ratios_to_modes",
]
