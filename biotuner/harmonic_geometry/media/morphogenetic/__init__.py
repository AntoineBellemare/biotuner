"""
``morphogenetic`` family — pattern growth via nonlinear PDEs / CAs.

Each medium in this family produces a pattern that *grows* through
nonlinear interaction with a forcing field, rather than being read off
as a snapshot of a wave. Examples include crystallization fronts and
reaction-diffusion systems.

Members
-------
- :class:`Crystallization` — Reiter cellular-automaton snowflake on a
  hexagonal grid, with chord-driven humidity / diffusion / anisotropy
  / Tonnetz-polygon seed pattern.
- :class:`ReactionDiffusion` — Gray-Scott chemical PDE; spots /
  stripes / labyrinths / mitosis as a function of chord-driven
  ``feed`` and ``kill`` rates.
"""

from biotuner.harmonic_geometry.media.morphogenetic.crystallization import (
    Crystallization,
)
from biotuner.harmonic_geometry.media.morphogenetic.reaction_diffusion import (
    ReactionDiffusion,
)

__all__ = ["Crystallization", "ReactionDiffusion"]
