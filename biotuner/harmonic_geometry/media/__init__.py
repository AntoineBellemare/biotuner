"""
biotuner.harmonic_geometry.media
================================

Module type: Subpackage.

Material-medium operators that turn a harmonic input into a response
field. See :mod:`.base` for the :class:`Medium` protocol and pipeline
composition; see the family subpackages for concrete media:

- :mod:`.eigenmode`     bounded standing-wave eigenfields
- :mod:`.wave_field`    open-medium travelling-wave superpositions
- :mod:`.parametric`    parametric-instability surfaces (cymatics / Faraday)
- :mod:`.transport`     passive redistribution on a wave field
- :mod:`.morphogenetic` pattern growth (crystallization, reaction-diffusion)
"""

from biotuner.harmonic_geometry.media.base import (
    Box3D,
    Circular,
    Domain,
    Family,
    Forcing,
    Medium,
    Pipeline,
    PolygonDomain,
    Rectangular,
    Sphere,
)

# Family subpackages — re-export the concrete media at the package level.
from biotuner.harmonic_geometry.media.eigenmode import (
    ClosedSurface,
    Elastic,
    PlasmaLattice,
    RigidPlate,
)
from biotuner.harmonic_geometry.media.wave_field import Acoustic, Interference
from biotuner.harmonic_geometry.media.parametric import Faraday
from biotuner.harmonic_geometry.media.transport import (
    Granular,
    Streaming,
    Tracer,
)
from biotuner.harmonic_geometry.media.morphogenetic import (
    Crystallization,
    ReactionDiffusion,
)

__all__ = [
    # Base
    "Medium",
    "Pipeline",
    "Family",
    "Forcing",
    # Domains
    "Domain",
    "Rectangular",
    "Circular",
    "PolygonDomain",
    "Box3D",
    "Sphere",
    # Eigenmode
    "RigidPlate",
    "ClosedSurface",
    "Elastic",
    "PlasmaLattice",
    # Wave field
    "Interference",
    "Acoustic",
    # Parametric
    "Faraday",
    # Transport
    "Granular",
    "Tracer",
    "Streaming",
    # Morphogenetic
    "Crystallization",
    "ReactionDiffusion",
]
