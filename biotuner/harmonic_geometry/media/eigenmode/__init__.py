"""
``eigenmode`` family — bounded standing-wave eigenproblems.

Each medium in this family solves ``−L ψ = λ ψ`` on a bounded domain
(``L`` = Laplacian or biharmonic operator, depending on boundary
conditions) and returns the chord's projection onto the resulting
eigenmodes.

Members
-------
- :class:`RigidPlate`    — Chladni-style plate (rect / disk / polygon / 3-D box)
- :class:`ClosedSurface` — spherical harmonics on the unit sphere
- :class:`Elastic`       — anisotropic / variable-coefficient rectangular plate
- :class:`PlasmaLattice` — discrete Coulomb crystal in a chord-shaped
  standing-wave potential (point-cloud output)
"""

from biotuner.harmonic_geometry.media.eigenmode.rigid_plate import (
    RigidPlate,
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
from biotuner.harmonic_geometry.media.eigenmode.closed_surface import (
    ClosedSurface,
    ratios_to_modes_lm,
    single_spherical_harmonic,
    spherical_harmonic_field,
    spherical_harmonic_from_input,
    spherical_harmonic_mesh,
    spherical_harmonic_temporal,
)
from biotuner.harmonic_geometry.media.eigenmode.elastic import Elastic
from biotuner.harmonic_geometry.media.eigenmode.plasma_lattice import (
    PlasmaLattice,
)

__all__ = [
    # RigidPlate
    "RigidPlate",
    "chladni_field_3d_box",
    "chladni_field_circular",
    "chladni_field_polygon",
    "chladni_field_rectangular",
    "chladni_from_input",
    "chladni_nodal_lines",
    "chladni_nodal_surfaces",
    "chladni_temporal",
    "ratios_to_modes",
    # ClosedSurface
    "ClosedSurface",
    "ratios_to_modes_lm",
    "single_spherical_harmonic",
    "spherical_harmonic_field",
    "spherical_harmonic_from_input",
    "spherical_harmonic_mesh",
    "spherical_harmonic_temporal",
    # Elastic / Plasma
    "Elastic",
    "PlasmaLattice",
]
