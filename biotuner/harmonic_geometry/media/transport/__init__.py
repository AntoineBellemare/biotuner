"""
``transport`` family — passive redistribution on a wave field.

Each medium in this family consumes a pre-computed wave field
(:class:`GeometryData`) and computes the steady-state or transient
distribution of a passive quantity (particles, density, tracer, flow)
driven by that field. Unlike eigenmode / wave_field / parametric, these
media do not solve a wave problem — they layer redistribution on top
of one.

Members
-------
- :class:`Granular` — Boltzmann-equilibrium grain redistribution
  (the classical sand-on-Chladni-plate figure).
- :class:`Tracer` — passive scalar advection on a wave field;
  produces the velocity field of streamlines around the field's
  features (or the steady-state tracer density).
- :class:`Streaming` — Rayleigh acoustic streaming; produces the
  slow vortex pairs that appear between antinodes of a finite-
  amplitude standing wave.
"""

from biotuner.harmonic_geometry.media.transport.granular import Granular
from biotuner.harmonic_geometry.media.transport.streaming import Streaming
from biotuner.harmonic_geometry.media.transport.tracer import Tracer

__all__ = ["Granular", "Streaming", "Tracer"]
