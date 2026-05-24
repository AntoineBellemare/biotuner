"""Backward-compatible re-export of the HarmonicInput descriptor.

``HarmonicInput`` originally lived in this submodule because
:mod:`biotuner.harmonic_geometry` was its only consumer. As of the
cross-module HarmonicInput promotion, the canonical home is
:mod:`biotuner.harmonic_input` at the top level — so unrelated modules
(``harmonic_timbre``, the engine backend, exporters) can consume the
descriptor without pulling in geometry as a transitive dependency.

This file remains so that the ~24 internal call sites and any external
code still doing ``from biotuner.harmonic_geometry.inputs import
HarmonicInput`` keep working unchanged. New code should prefer::

    from biotuner.harmonic_input import HarmonicInput, HarmonicSequence
"""

from biotuner.harmonic_input import (  # noqa: F401
    HarmonicInput,
    HarmonicSequence,
    RatioLike,
    SCALE_ATTRS,
    SCALE_KEYS,
    _SCALE_KEY_TO_ATTR,  # internal-but-exposed: a few tests inspect it
    _get_scale_values,
)

__all__ = [
    "HarmonicInput",
    "HarmonicSequence",
    "RatioLike",
    "SCALE_ATTRS",
    "SCALE_KEYS",
]
