"""
Compatibility shim — the real implementation lives in
:mod:`biotuner.harmonic_geometry.media.eigenmode.closed_surface`.

This module re-exports the historical ``spherical_harmonic_*`` functional
API so existing imports keep working after the migration to the
family-based :mod:`biotuner.harmonic_geometry.media` layout. Prefer the
new path in new code.
"""

from biotuner.harmonic_geometry.media.eigenmode.closed_surface import (  # noqa: F401
    _real_ylm,
    ratios_to_modes_lm,
    single_spherical_harmonic,
    spherical_harmonic_field,
    spherical_harmonic_from_input,
    spherical_harmonic_mesh,
    spherical_harmonic_temporal,
)

__all__ = [
    "ratios_to_modes_lm",
    "single_spherical_harmonic",
    "spherical_harmonic_field",
    "spherical_harmonic_from_input",
    "spherical_harmonic_mesh",
    "spherical_harmonic_temporal",
]
