"""
Chord-shapes-the-medium couplings.

This module exposes scalar reductions of a :class:`HarmonicInput` that
are intended to be wired into medium parameters — so the same chord
that *drives* a medium can also *shape* the medium's material properties.
Each function takes a :class:`HarmonicInput` and returns a single
``float`` in a documented range, suitable for parameter mapping.

The mappings here are deliberately simple and unitless. Each concrete
medium is responsible for scaling the unit-interval / strictly-positive
output of these helpers to its own parameter range.

Available reductions
--------------------
- :func:`consonance` — Tenney-like dyadic harmonic distance (lower = more consonant)
- :func:`ratio_complexity` — average denominator size of best rational fits
- :func:`spectral_spread` — log-frequency standard deviation of the peaks
- :func:`amplitude_entropy` — normalized Shannon entropy of amplitudes

Suggested medium-parameter mappings (each medium picks one or none):

==========================  ====================================
Reduction                   Suggested parameter
==========================  ====================================
:func:`consonance`          ``viscosity`` (high consonance → low viscosity)
:func:`ratio_complexity`    ``anisotropy_order`` (high complexity → higher-order anisotropy)
:func:`spectral_spread`     ``diffusion`` (wide chord → broader diffusion)
:func:`amplitude_entropy`   ``noise_temperature`` (flat amplitudes → high temperature)
==========================  ====================================
"""

from __future__ import annotations

from fractions import Fraction
from math import log2

import numpy as np

from biotuner.harmonic_geometry.inputs import HarmonicInput


# ============================================================== consonance


def consonance(input: HarmonicInput, max_denominator: int = 64) -> float:
    """Mean dyadic Tenney harmonic distance across pairs of peaks.

    Each ratio is approximated as a rational ``p / q`` (with
    ``max_denominator``); its harmonic distance is ``log2(p * q)``. The
    return value is the *negated, normalized* mean — high values
    correspond to **consonant** chords (low harmonic distance), suitable
    for direct use as a "clarity" knob.

    Returns
    -------
    float
        In ``[0.0, 1.0]``. ``1.0`` = pure unison; ``0.0`` = highly complex.
    """
    ratios = input.to_ratios()
    if len(ratios) < 2:
        return 1.0
    distances = []
    for r in ratios:
        f = Fraction(float(r)).limit_denominator(max_denominator)
        p, q = max(1, f.numerator), max(1, f.denominator)
        distances.append(log2(p * q))
    mean_d = float(np.mean(distances))
    # Cap at log2(max_denominator²) for the normalization.
    cap = 2.0 * log2(max_denominator)
    return float(max(0.0, 1.0 - mean_d / cap))


# ========================================================= ratio_complexity


def ratio_complexity(input: HarmonicInput, max_denominator: int = 64) -> float:
    """Mean ``max(p, q)`` of rational approximations of the chord's ratios.

    A simple integer-complexity proxy: ``2`` for ``2/1``, ``5`` for
    ``5/4``, ``11`` for ``11/8``, etc. Useful as a knob that biases
    higher-order anisotropy / lobe counts for less-rational chords.

    Returns
    -------
    float
        ``>= 1.0``. Unbounded above; consumers should cap as needed.
    """
    ratios = input.to_ratios()
    if not ratios:
        return 1.0
    cplx = []
    for r in ratios:
        f = Fraction(float(r)).limit_denominator(max_denominator)
        cplx.append(max(abs(f.numerator), abs(f.denominator)))
    return float(np.mean(cplx))


# ============================================================ spectral_spread


def spectral_spread(input: HarmonicInput) -> float:
    """Log-frequency standard deviation of the chord's peaks (octaves).

    Returns ``0.0`` for a single-component input.
    """
    peaks = input.to_peaks()
    if peaks.size < 2:
        return 0.0
    log_peaks = np.log2(peaks)
    return float(np.std(log_peaks))


# ========================================================== amplitude_entropy


def amplitude_entropy(input: HarmonicInput) -> float:
    """Normalized Shannon entropy of the chord's amplitudes.

    Returns
    -------
    float
        In ``[0.0, 1.0]``. ``1.0`` = perfectly uniform amplitudes
        (maximum entropy); ``0.0`` = all energy in one component.
    """
    amps = input.normalized_amplitudes()
    if amps.size <= 1:
        return 0.0
    p = amps / amps.sum() if amps.sum() > 0 else amps
    nz = p[p > 0]
    h = float(-np.sum(nz * np.log(nz)))
    return float(h / np.log(amps.size))
