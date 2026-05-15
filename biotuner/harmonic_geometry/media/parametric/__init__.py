"""
``parametric`` family — instability-driven response to a 2f drive.

Each medium in this family models a system that responds to vertical
periodic forcing through a parametric (Mathieu / Floquet) instability:
above a critical drive amplitude, certain modes grow exponentially while
others decay. The selected pattern is the visible response.

Members
-------
- :class:`Faraday` — capillary-gravity surface waves (true cymatics)
"""

from biotuner.harmonic_geometry.media.parametric.faraday import Faraday

__all__ = ["Faraday"]
