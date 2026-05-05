"""
Harmonic-input extensions: enrich a chord with derived peaks / ratios.

This module bridges :mod:`biotuner.peaks_extension` and
:mod:`biotuner.scale_construction` (which operate on plain peak / ratio
arrays) with :class:`HarmonicInput` (the geometry-side adapter). Each
function takes a :class:`HarmonicInput` and returns a *new*
:class:`HarmonicInput` carrying an extended component set, ready to feed
back into any geometry function (Chladni, spherical, Talbot, etc.).

Why this matters for interference patterns
------------------------------------------
A bare chord has only a handful of components (3–4 for a triad, 4–5 for
a 7th chord). Many of the open-medium interference visualisations want
*dozens* of modes to read as visually rich. By extending the chord with
multiplicative harmonics, sub-harmonics, harmonic-fit common tones, or a
generated harmonic tuning, the resulting field carries the chord's
modal structure but renders with the textural richness of a real
diffractive carpet.

Functions
---------
- :func:`extend_harmonics` — add multiplicative harmonics for each peak
- :func:`extend_subharmonics` — add divisive sub-harmonics
- :func:`extend_harmonic_fit` — common harmonics across pairs (consonance lattice)
- :func:`extend_harmonic_tuning` — replace ratios with a generated harmonic tuning
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from biotuner.harmonic_geometry.inputs import HarmonicInput


# ============================================================ extend_harmonics


def extend_harmonics(
    input: HarmonicInput,
    n_harmonics: int = 4,
    *,
    include_fundamental: bool = True,
    decay: float = 1.0,
) -> HarmonicInput:
    """Add multiplicative harmonics ``2f, 3f, …, nf`` for every peak.

    Wraps :func:`biotuner.peaks_extension.EEG_harmonics_mult` and rebuilds
    a :class:`HarmonicInput` whose peaks are the union of the originals
    and their harmonics. Per-harmonic amplitude follows ``a_n = a_0 / n^decay``
    where ``a_0`` is the fundamental's amplitude.

    Parameters
    ----------
    input : HarmonicInput
    n_harmonics : int, default=4
        Number of harmonics to compute per peak (excludes the fundamental
        when ``include_fundamental=True``; the fundamental is always
        included when ``True``).
    include_fundamental : bool, default=True
        If False, drop the fundamental frequencies and keep only the
        harmonics. Useful for showing the *harmonic content* alone.
    decay : float, default=1.0
        Amplitude falloff exponent: harmonic ``n`` of a peak is given
        amplitude ``a_0 / n^decay``. ``decay=0`` keeps all harmonics at
        the same amplitude; ``decay=1`` gives the natural ``1/n`` law.

    Returns
    -------
    HarmonicInput
        A new input carrying the extended peak set and per-peak
        amplitudes, with ``metadata['extension'] = 'harmonics'``.
    """
    if n_harmonics < 1:
        raise ValueError(f"n_harmonics must be >= 1, got {n_harmonics!r}.")
    peaks0 = input.to_peaks()
    amps0 = input.normalized_amplitudes()
    phases0 = (
        np.asarray(input.phases, dtype=np.float64)
        if input.phases is not None
        else np.zeros_like(peaks0)
    )

    new_peaks: list[float] = []
    new_amps: list[float] = []
    new_phases: list[float] = []
    n_start = 1 if include_fundamental else 2
    for p, a, phi in zip(peaks0, amps0, phases0):
        for n in range(n_start, n_harmonics + 2):
            new_peaks.append(float(p * n))
            new_amps.append(float(a / (n ** float(decay))))
            new_phases.append(float(phi))

    return HarmonicInput(
        peaks=new_peaks,
        amplitudes=new_amps,
        phases=new_phases,
        base_freq=input.base_freq,
        equave=input.equave,
        metadata={**input.metadata, "extension": "harmonics"},
    )


# ========================================================= extend_subharmonics


def extend_subharmonics(
    input: HarmonicInput,
    n_harmonics: int = 4,
    *,
    include_fundamental: bool = True,
    decay: float = 1.0,
) -> HarmonicInput:
    """Add divisive sub-harmonics ``f/2, f/3, …, f/n`` for every peak.

    Wraps :func:`biotuner.peaks_extension.EEG_harmonics_div` semantics
    but stays peak-array native to avoid lifting a non-trivial dependency.
    The peak set is the union of originals and their sub-harmonics.

    Parameters
    ----------
    input : HarmonicInput
    n_harmonics : int, default=4
    include_fundamental : bool, default=True
    decay : float, default=1.0

    Returns
    -------
    HarmonicInput
        ``metadata['extension'] = 'subharmonics'``.
    """
    if n_harmonics < 1:
        raise ValueError(f"n_harmonics must be >= 1, got {n_harmonics!r}.")
    peaks0 = input.to_peaks()
    amps0 = input.normalized_amplitudes()
    phases0 = (
        np.asarray(input.phases, dtype=np.float64)
        if input.phases is not None
        else np.zeros_like(peaks0)
    )

    new_peaks: list[float] = []
    new_amps: list[float] = []
    new_phases: list[float] = []
    n_start = 1 if include_fundamental else 2
    for p, a, phi in zip(peaks0, amps0, phases0):
        for n in range(n_start, n_harmonics + 2):
            new_peaks.append(float(p / n))
            new_amps.append(float(a / (n ** float(decay))))
            new_phases.append(float(phi))

    return HarmonicInput(
        peaks=new_peaks,
        amplitudes=new_amps,
        phases=new_phases,
        base_freq=input.base_freq,
        equave=input.equave,
        metadata={**input.metadata, "extension": "subharmonics"},
    )


# ======================================================= extend_harmonic_fit


def extend_harmonic_fit(
    input: HarmonicInput,
    n_harm: int = 10,
    bounds: float = 0.1,
) -> HarmonicInput:
    """Add the *common harmonics* of the chord — its consonance lattice.

    Wraps :func:`biotuner.peaks_extension.harmonic_fit`. For every pair of
    peaks, this finds the harmonic positions where their multiplicative
    series coincide (within ``bounds`` Hz). The returned chord is
    augmented with these matched harmonics — peaks that *resolve* the
    chord because they sit at a common harmonic of two or more
    components.

    Parameters
    ----------
    input : HarmonicInput
    n_harm : int, default=10
        Number of harmonics computed per peak before pairwise matching.
    bounds : float, default=0.1
        Hz tolerance for considering two harmonics as matching.

    Returns
    -------
    HarmonicInput
        ``metadata['extension'] = 'harmonic_fit'``.
    """
    if n_harm < 2:
        raise ValueError(f"n_harm must be >= 2, got {n_harm!r}.")
    from biotuner.peaks_extension import harmonic_fit  # lazy: heavy deps

    peaks0 = input.to_peaks().tolist()
    amps0 = input.normalized_amplitudes().tolist()
    harm_fit, _, _, _ = harmonic_fit(
        peaks0, n_harm=n_harm, bounds=bounds, function="mult"
    )
    # harm_fit is a flat list of frequencies that match between pairs.
    extra = sorted({float(h) for h in (harm_fit or []) if h > 0})
    new_peaks = peaks0 + extra
    # Give the matched harmonics half the average original amplitude so
    # they're audible without dominating.
    base_amp = float(np.mean(amps0)) if amps0 else 1.0
    new_amps = amps0 + [0.5 * base_amp] * len(extra)

    return HarmonicInput(
        peaks=new_peaks,
        amplitudes=new_amps,
        base_freq=input.base_freq,
        equave=input.equave,
        metadata={**input.metadata, "extension": "harmonic_fit"},
    )


# ==================================================== extend_harmonic_tuning


def extend_harmonic_tuning(
    input: HarmonicInput,
    n_harmonics: int = 10,
    *,
    octave: float = 2.0,
    min_ratio: float = 1.0,
    max_ratio: float = 2.0,
) -> HarmonicInput:
    """Replace the input's ratios with a generated harmonic-series tuning.

    Wraps :func:`biotuner.scale_construction.harmonic_tuning`. The chord's
    peaks are first turned into integer harmonic indices, then the
    `harmonic_tuning` builder generates a tuning whose ratios are
    derived from those harmonics within ``[min_ratio, max_ratio]``.

    Useful for the report: the same chord rendered with `n_harmonics=4`
    looks coarse, with `n_harmonics=10` shows mid-detail, with
    `n_harmonics=20+` shows full intricate carpets.

    Parameters
    ----------
    input : HarmonicInput
    n_harmonics : int, default=10
        Number of harmonics fed into ``harmonic_tuning``. Higher = more
        ratios, more visual richness.
    octave, min_ratio, max_ratio
        Forwarded to ``harmonic_tuning``.

    Returns
    -------
    HarmonicInput
        ``metadata['extension'] = 'harmonic_tuning'``.
    """
    if n_harmonics < 2:
        raise ValueError(f"n_harmonics must be >= 2, got {n_harmonics!r}.")
    from biotuner.scale_construction import harmonic_tuning  # lazy

    # Use the input's peaks (or ratios * base_freq) as the harmonic seeds.
    peaks0 = input.to_peaks().tolist()
    # harmonic_tuning expects a list of harmonic indices; project peaks
    # to integer multiples of the smallest peak.
    if not peaks0:
        raise ValueError("Cannot extend an empty HarmonicInput.")
    f0 = min(peaks0)
    seed_indices = sorted({int(round(p / f0)) for p in peaks0 if p > 0})
    # Augment with low harmonics 1..n_harmonics for richness.
    seed_indices = sorted(set(seed_indices + list(range(1, n_harmonics + 1))))

    # harmonic_tuning may return tuples (ratio, ...) or plain floats; we
    # only need the ratio values.
    raw = harmonic_tuning(
        seed_indices,
        octave=float(octave),
        min_ratio=float(min_ratio),
        max_ratio=float(max_ratio),
    )
    ratios = []
    for r in raw:
        if isinstance(r, (tuple, list)):
            ratios.append(float(r[0]))
        else:
            ratios.append(float(r))
    ratios = sorted({round(x, 8) for x in ratios if x > 0})

    return HarmonicInput(
        ratios=ratios,
        base_freq=input.base_freq,
        equave=input.equave,
        metadata={**input.metadata, "extension": "harmonic_tuning"},
    )
