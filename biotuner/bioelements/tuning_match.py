"""Tuning matching — a biosignal's *ratio structure* vs an element's line ratios.

The position matcher (:mod:`biotuner.bioelements.matching`) octave-folds each peak
into the optical band and asks *"does this peak land ON a spectral line?"* — an
**absolute**, anchored question. This module asks the **relative** one, the way a
musician compares scales: *"does the biosignal's tuning — the set of intervals
among its peaks — resemble the intervals among an element's emission lines?"*

Because it compares *ratios*, tuning matching is **transposition- and octave-
invariant**: shifting every peak by the same factor leaves the score unchanged
(the position matcher's score moves). It needs no octave-folding at all — ratios
are already scale-free.

A tuning is represented as a *tuning vector*: the octave-reduced pairwise ratios
of a frequency set, each reduced to a small-integer fraction
(``Fraction.limit_denominator(maxdenom)``, biotuner's convention) and weighted by
the product of the two amplitudes/intensities × the ratio's harmonicity
(:func:`~biotuner.metrics.dyad_similarity`). Two tunings are compared by the
**cosine** of their vectors — density-normalised, so a line-rich element does not
win by sheer count (naive "does it contain a 3:2 somewhere?" always would).
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd

from biotuner.bioelements import units, tables
from biotuner.bioelements.spectrum import Spectrum, element_spectrum
from biotuner.metrics import dyad_similarity


def tuning_vector(freqs, amps=None, *, octave: float = 2.0, maxdenom: int = 50,
                  harm_weight: bool = True) -> dict:
    """The octave-reduced tuning of a frequency set as a weighted ratio vector.

    Every pair of frequencies contributes its interval, octave-reduced into
    ``[1, octave)`` and snapped to ``Fraction(...).limit_denominator(maxdenom)``.
    Weight = product of the two amplitudes, times the interval's harmonicity
    (:func:`dyad_similarity`) when ``harm_weight`` — so a shared 3:2 counts far
    more than a shared 127:100. Returns ``{(p, q): weight}``; transposition- and
    octave-invariant.
    """
    freqs = np.asarray(freqs, float)
    if amps is None:
        amps = np.ones(len(freqs))
    amps = np.asarray(amps, float)
    out: dict = {}
    for i in range(len(freqs)):
        for j in range(len(freqs)):
            if i == j or freqs[i] <= 0 or freqs[j] <= 0:
                continue
            r = freqs[j] / freqs[i]
            if r < 1.0:
                continue                       # keep numerator ≥ denominator
            while r >= octave:                 # octave-reduce into [1, octave)
                r /= octave
            if r <= 1.0001:
                continue                       # unison / exact octave: no interval
            fr = Fraction(float(r)).limit_denominator(maxdenom)
            if fr.numerator == fr.denominator:
                continue
            key = (fr.numerator, fr.denominator)
            hw = dyad_similarity(fr.numerator / fr.denominator) if harm_weight else 1.0
            out[key] = out.get(key, 0.0) + float(amps[i] * amps[j]) * hw
    return out


def spectrum_tuning(spectrum: Spectrum, *, octave: float = 2.0, maxdenom: int = 50,
                    harm_weight: bool = True) -> dict:
    """The tuning vector of a spectrum's **raw** lines (folding would destroy the
    true line ratios). Line intensities are the amplitudes."""
    if len(spectrum) == 0:
        return {}
    freqs = units.angstrom_to_hertz(spectrum.wavelength)
    return tuning_vector(freqs, spectrum.intensity, octave=octave,
                         maxdenom=maxdenom, harm_weight=harm_weight)


def element_tuning(element: str, *, table: str = "air", top: int = 40,
                   octave: float = 2.0, maxdenom: int = 50,
                   harm_weight: bool = True) -> dict:
    """The tuning vector of an element's NIST emission lines."""
    sp = element_spectrum(element, table=table, top=top, normalise=True)
    return spectrum_tuning(sp, octave=octave, maxdenom=maxdenom, harm_weight=harm_weight)


_MATERIAL_TUNING_CACHE: dict = {}


def material_tuning_vector(material, *, table: str = "air", top: int = 40, basis: str = "atom",
                           octave: float = 2.0, maxdenom: int = 50, harm_weight: bool = True) -> dict:
    """The tuning vector of a material's composite spectrum (cached — materials
    are static, so the O(top²) build runs once per (material, params))."""
    key = (material.name, table, top, basis, octave, maxdenom, harm_weight)
    tv = _MATERIAL_TUNING_CACHE.get(key)
    if tv is None:
        sp = material.spectrum(table=table, top=top, basis=basis)
        tv = spectrum_tuning(sp, octave=octave, maxdenom=maxdenom, harm_weight=harm_weight)
        _MATERIAL_TUNING_CACHE[key] = tv
    return tv


def tuning_cosine(a: dict, b: dict) -> float:
    """Cosine similarity of two tuning vectors, in ``[0, 1]`` (density-normalised)."""
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    va = np.array([a.get(k, 0.0) for k in keys])
    vb = np.array([b.get(k, 0.0) for k in keys])
    na, nb = np.linalg.norm(va), np.linalg.norm(vb)
    return float(va @ vb / (na * nb)) if na > 0 and nb > 0 else 0.0


def match_tuning(peaks_hz, target, *, amps=None, table: str = "air", top: int = 40,
                 octave: float = 2.0, maxdenom: int = 50, harm_weight: bool = True) -> float:
    """How much a biosignal's tuning resembles a target's line-ratio structure.

    ``target`` may be an element name, a :class:`Spectrum`, a
    :class:`~biotuner.bioelements.composition.Composition` (uses its composite
    spectrum), or a pre-built tuning vector (dict). Returns the cosine similarity
    in ``[0, 1]`` — transposition- and octave-invariant.
    """
    sig = tuning_vector(peaks_hz, amps, octave=octave, maxdenom=maxdenom,
                        harm_weight=harm_weight)
    kw = dict(octave=octave, maxdenom=maxdenom, harm_weight=harm_weight)
    if isinstance(target, dict):
        tgt = target
    elif isinstance(target, Spectrum):
        tgt = spectrum_tuning(target, **kw)
    elif isinstance(target, str):
        tgt = element_tuning(target, table=table, top=top, **kw)
    elif hasattr(target, "spectrum"):                       # Composition / material
        tgt = spectrum_tuning(target.spectrum(table=table, top=top), **kw)
    else:
        raise TypeError(f"unsupported target {type(target)!r}")
    return tuning_cosine(sig, tgt)


def match_elements_by_tuning(peaks_hz, *, amps=None, table: str = "air", top: int = 40,
                             octave: float = 2.0, maxdenom: int = 50,
                             harm_weight: bool = True, min_score: float = 0.0) -> pd.DataFrame:
    """Rank every element by how well its line-ratio structure matches a
    biosignal's tuning — the ratio-structure counterpart of
    :func:`~biotuner.bioelements.matching.match_elements`.

    Returns a DataFrame (element, tuning_score, category) sorted descending.
    """
    sig = tuning_vector(peaks_hz, amps, octave=octave, maxdenom=maxdenom,
                        harm_weight=harm_weight)
    rows = []
    for elem in tables.available_elements(table):
        tgt = element_tuning(elem, table=table, top=top, octave=octave,
                             maxdenom=maxdenom, harm_weight=harm_weight)
        score = tuning_cosine(sig, tgt)
        if score >= min_score:
            rows.append({"element": elem, "tuning_score": score,
                         "category": tables.element_category(elem, table)})
    return (pd.DataFrame(rows).sort_values("tuning_score", ascending=False)
            .reset_index(drop=True))


def match_materials_by_tuning(peaks_hz, materials=None, *, amps=None, table: str = "air",
                              top: int = 40, basis: str = "atom", octave: float = 2.0,
                              maxdenom: int = 50, harm_weight: bool = True,
                              include_elements: bool = False) -> pd.DataFrame:
    """Rank the material dictionary by how well each material's line-ratio
    structure matches a biosignal's tuning — the ratio-structure counterpart of
    :func:`~biotuner.bioelements.affinity.match_materials`.

    Like ``match_materials``, bare elements are excluded by default
    (``include_elements=False``). Returns a DataFrame (material, tuning_score,
    kind, archetype) sorted descending — transposition- and octave-invariant.
    """
    from biotuner.bioelements.materials import MATERIALS
    mats = MATERIALS if materials is None else materials
    sig = tuning_vector(peaks_hz, amps, octave=octave, maxdenom=maxdenom,
                        harm_weight=harm_weight)
    rows = []
    for name, m in mats.items():
        if not include_elements and m.kind == "element":
            continue
        tgt = material_tuning_vector(m, table=table, top=top, basis=basis, octave=octave,
                                     maxdenom=maxdenom, harm_weight=harm_weight)
        rows.append({"material": name, "tuning_score": tuning_cosine(sig, tgt),
                     "kind": m.kind, "archetype": m.archetype or ""})
    return (pd.DataFrame(rows).sort_values("tuning_score", ascending=False)
            .reset_index(drop=True))
