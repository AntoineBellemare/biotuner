"""Dynamic Tonality: spectra bent to fit a well-formed scale.

Everything else in :mod:`biotuner.mos` moves the *scale*.  This module moves
the *timbre*, and it is the move that makes the rest of the labyrinth playable.

The problem it solves is the one that has always limited microtonality.  Slide
the generator of a well-formed scale away from the familiar equal temperaments
and the scale's intervals stop landing near small-integer frequency ratios.
A harmonic tone -- partials at 1, 2, 3, 4, … times the fundamental -- then has
nothing to lock onto: partial 3 of one note sits a few dozen cents from partial
2 of the note a fifth above, which is exactly the spacing that maximises
Plomp-Levelt roughness.  The scale sounds out of tune not because it is
mistuned but because the *timbre* is tuned to a different lattice.

Milne et al. (2011) §6 close the paper by inverting the fix: keep the scale
wherever the musician put it and retune the partials instead, so that "the
pitch (relative to the fundamental) of each partial is mapped to a linear
combination of the pitch heights of the period and generator of the underlying
scale".  Concretely, harmonic ``h`` moves from ``h`` to

.. math::  P^{\\alpha} \\, G^{\\beta}

with ``P`` the period, ``G`` the generator, and ``(\\alpha, \\beta)`` the
integer pair that best approximates it.  Because every scale degree is *also*
a point on that ``(P, G)`` lattice, partials of different tones now coincide
exactly wherever scale intervals do -- and they keep coinciding as the
generator slides, since both move together.  Sensory dissonance
(Sethares, 2005) is minimised continuously across the whole tuning range
rather than at a handful of privileged temperaments.

A caution about ``max_beta`` and ``beta_penalty``
-------------------------------------------------
The mapping is only musically useful when ``|beta|`` is small.  A generator
chain of length 4 is something a listener tracks (four fifths *is* the major
third); a chain of length 21 is an accident of number theory.  Worse, a large
``beta`` budget defeats the whole exercise: given ~24 generators to spend, the
optimiser can approximate any just harmonic to within a cent or so, at which
point the "matched" spectrum is the harmonic series again in all but name, and
no partials coincide with anything.

Measured here on 26 MOS signatures at their noble tunings, sounding every
degree with 12 partials, the dissonance reduction over a plain harmonic timbre
of the same size and loudness is:

===========================  ==================  ================
setting                      reduction (median)  signatures worse
===========================  ==================  ================
``max_beta=5``               13.9 % (6.2--30.2)  0 / 26
``max_beta=8``                9.3 % (3.2--17.1)  0 / 26
``beta_penalty=3.0``          8.4 %              1 / 26
``max_beta=24`` (default)     0.7 %              6 / 26
===========================  ==================  ================

So the defaults -- which are the permissive ones the API contract specifies --
are the setting at which Dynamic Tonality barely works, and for six signatures
(``1L3s``, ``1L4s``, ``4L1s``, ``2L5s``, ``2L7s``, ``4L7s``) it comes out a
fraction of a percent *worse* than a harmonic timbre.  That is not a defect in
the theory, it is the beta budget swamping it.  Bounding the chain rescues
every one of them: the same six gain 6--24 % at ``max_beta=5``.  Bounding is
also more reliable than penalising -- ``beta_penalty=3.0`` still leaves
``3L2s`` 1.0 % worse.  :func:`dissonance_advantage` reports which side of this
you landed on for any given scale.

References
----------
Milne, A.J., Carlé, M., Sethares, W.A., Noll, T., Holland, S. (2011).
Scratching the Scale Labyrinth. In *Mathematics and Computation in Music*,
LNAI 6726, 180--195. §6 "Dynamic Tonality".

Sethares, W.A. (2005). *Tuning, Timbre, Spectrum, Scale* (2nd ed.). Springer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from biotuner.mos.scale import MOSScale

__all__ = [
    "PartialMap",
    "SimpleTimbre",
    "map_harmonic",
    "matched_partials",
    "matched_ratios",
    "matched_spectrum",
    "dynamic_timbre",
    "spectral_dissonance",
    "scale_dissonance",
    "dissonance_advantage",
]


# --------------------------------------------------------------------------- #
# One partial
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PartialMap:
    """Where one harmonic lands once it is pulled onto the scale's lattice.

    Attributes
    ----------
    harmonic : int
        The harmonic number ``h`` that was mapped, ``1`` being the fundamental.
    alpha : int
        Periods in the retuned partial.  May be negative: a partial can need
        pulling *down* by an octave once a long generator chain has overshot.
    beta : int
        Generators in the retuned partial.  Its magnitude is the interesting
        number -- see the module docstring.
    ratio : float
        The retuned partial, ``period ** alpha * generator_ratio ** beta``.
        This is what actually sounds.
    just_ratio : float
        The plain harmonic ``float(h)``, i.e. where the partial would sit in an
        ordinary harmonic tone.
    error_cents : float
        ``ratio`` minus ``just_ratio`` in cents, signed.  Zero means the scale's
        lattice happens to contain that harmonic exactly, which for an octave
        period is true of every power of two.
    """

    harmonic: int
    alpha: int
    beta: int
    ratio: float
    just_ratio: float
    error_cents: float

    @property
    def cents(self) -> float:
        """The retuned partial in cents above the fundamental."""
        return 1200.0 * math.log2(self.ratio)


@dataclass
class SimpleTimbre:
    """Fallback spectrum container for when ``harmonic_timbre`` is unavailable.

    :func:`dynamic_timbre` prefers :class:`biotuner.harmonic_timbre.Timbre`,
    which carries phases, decay times, modulators and exporters.  This stands
    in when that subpackage cannot be imported, and mirrors the field names of
    the fields it does carry so that downstream code can duck-type across the
    two.
    """

    partials_hz: np.ndarray
    amplitudes: np.ndarray
    base_freq: float = 1.0
    matched_tuning: Optional[list] = None
    matching_method: str = ""
    metadata: dict = field(default_factory=dict)

    def n_partials(self) -> int:
        return int(np.asarray(self.partials_hz).shape[0])


# --------------------------------------------------------------------------- #
# The mapping itself
# --------------------------------------------------------------------------- #
def _beta_order(max_beta: int) -> List[int]:
    """Candidate ``beta`` values, nearest to zero first, positive before negative.

    Search order *is* the tie-break rule.  Ties are not hypothetical: at a
    rational generator ``beta`` and ``beta ± denominator`` give bit-identical
    error, and the shorter chain is always the one to keep.  Preferring the
    positive member of a ``±beta`` tie matches the convention that generators
    stack upward from the root.
    """
    order = [0]
    for b in range(1, max_beta + 1):
        order.extend((b, -b))
    return order


def map_harmonic(
    h: int,
    scale: MOSScale,
    max_beta: int = 24,
    beta_penalty: float = 0.0,
) -> PartialMap:
    """Pull harmonic ``h`` onto the ``(period, generator)`` lattice of ``scale``.

    Finds the integer pair minimising

    ``|alpha * log(period) + beta * log(generator) - log(h)|``

    over ``|beta| <= max_beta``.  Only ``beta`` is searched: once it is fixed
    the best ``alpha`` is forced, because moving by whole periods is the
    coarsest possible adjustment and rounding to the nearest one is optimal by
    construction.  That turns a two-dimensional lattice search into a scan of
    ``2 * max_beta + 1`` candidates.

    Parameters
    ----------
    h : int
        Harmonic number, ``>= 1``.
    scale : MOSScale
        Supplies ``period`` and ``generator_ratio``.  Nothing else about the
        scale enters -- two scales sharing a generator share a timbre, which is
        why a whole MOS family can be played with one spectrum.
    max_beta : int, default 24
        Longest generator chain allowed.  See the module docstring: this
        default is permissive enough to hide the effect it is meant to produce.
    beta_penalty : float, default 0.0
        Cents of penalty per generator in the chain.  Non-zero trades tuning
        accuracy for a shorter, more audible chain; ``3.0`` is a good starting
        point for an octave period.

    Returns
    -------
    PartialMap

    Examples
    --------
    The fundamental is always the origin of the lattice, at no error:

    >>> from biotuner.mos.scale import MOSScale
    >>> m = MOSScale.from_signature(5, 2, tuning=31)
    >>> p = map_harmonic(1, m)
    >>> p.alpha, p.beta, p.error_cents
    (0, 0, 0.0)

    With an octave period the octave *is* the period, so harmonic 2 needs no
    generators at all:

    >>> p = map_harmonic(2, m)
    >>> p.alpha, p.beta, p.error_cents
    (1, 0, 0.0)

    Harmonic 3 is one period plus one generator -- a twelfth is an octave plus
    a fifth -- and in 31-EDO meantone that fifth is 5.18 cents flat:

    >>> p = map_harmonic(3, m)
    >>> p.alpha, p.beta, round(p.error_cents, 3)
    (1, 1, -5.181)

    Harmonic 5 is four generators and *no* periods, which is the definition of
    meantone: four fifths, octave-reduced, are the major third:

    >>> p = map_harmonic(5, m)
    >>> p.alpha, p.beta, round(p.error_cents, 3)
    (0, 4, 0.783)

    A penalty buys a shorter chain at the cost of accuracy.  At the noble
    tuning harmonic 5 defaults to a 21-generator chain that is essentially just
    the plain harmonic; three cents per generator collapses it to meantone's
    four, 30 cents sharp:

    At a rational generator the lattice closes on itself, so ``beta`` and
    ``beta ± denominator`` name the very same pitch.  The shorter chain is the
    one that comes back -- in 7-EDO the third harmonic is one fifth, not the
    bit-identical twenty-generator spelling:

    >>> seven = MOSScale.from_signature(5, 2, tuning='equalized')
    >>> map_harmonic(3, seven).alpha, map_harmonic(3, seven).beta
    (1, 1)
    >>> max(abs(map_harmonic(h, seven).beta) for h in range(1, 13))
    3

    >>> n = MOSScale.from_signature(5, 2)
    >>> map_harmonic(5, n).beta
    21
    >>> p = map_harmonic(5, n, beta_penalty=3.0)
    >>> p.beta, round(p.error_cents, 2)
    (4, 30.07)
    """
    if not isinstance(h, (int, np.integer)) or h < 1:
        raise ValueError(f"harmonic number must be an integer >= 1, got {h!r}")
    if max_beta < 0:
        raise ValueError(f"max_beta must be >= 0, got {max_beta!r}")
    if beta_penalty < 0:
        raise ValueError(
            f"beta_penalty is a cents cost per generator and must be >= 0, "
            f"got {beta_penalty!r}"
        )

    h = int(h)
    period = scale.period
    g = scale.generator  # generator as a fraction of the period
    # Work in period units throughout, then convert once: a period is
    # 1200*log2(period) cents, so an error of e periods is e * that.
    cents_per_period = 1200.0 * math.log2(period)
    target = math.log(h) / math.log(period)

    # How much better a longer chain must be before it displaces a shorter one.
    # This cannot be a fixed epsilon.  At a rational generator p/q the chains
    # `beta` and `beta ± q` are mathematically identical, but `beta * g` is
    # evaluated in binary floating point, so their computed costs drift apart
    # by roughly (|alpha| + |beta|) * eps * cents_per_period -- about 2e-12
    # cents for a 21-generator chain under an octave.  A 1e-12 threshold reads
    # that rounding noise as a genuine improvement and hands the win to the
    # *longest* chain in the tie, which is precisely backwards: at 7-EDO the
    # third harmonic came out as a 20-generator chain instead of the single
    # fifth that is bit-for-bit as accurate.
    #
    # So the tolerance tracks the error term itself: machine epsilon times the
    # magnitudes in play, times a healthy safety factor.  That is ~3e-8 cents
    # at the defaults -- four orders of magnitude above the observed drift and
    # eight below anything musically distinguishable -- and it stays sane as
    # ``max_beta`` grows, which a fixed relative constant would not.
    tol = (
        4096.0
        * 2.0**-52
        * (1.0 + max_beta)
        * max(1.0, cents_per_period, beta_penalty)
    )

    best_cost = math.inf
    best_alpha = 0
    best_beta = 0
    for beta in _beta_order(max_beta):
        # floor(x + 0.5) rather than round(): banker's rounding would make the
        # result depend on which side of a half-period tie the float lands on.
        alpha = int(math.floor(target - beta * g + 0.5))
        err = (alpha + beta * g - target) * cents_per_period
        cost = abs(err) + beta_penalty * abs(beta)
        if cost < best_cost - tol:
            best_cost, best_alpha, best_beta = cost, alpha, beta

    ratio = period**best_alpha * scale.generator_ratio**best_beta
    error_cents = 1200.0 * math.log2(ratio / h)
    return PartialMap(
        harmonic=h,
        alpha=best_alpha,
        beta=best_beta,
        ratio=ratio,
        just_ratio=float(h),
        error_cents=error_cents,
    )


def matched_partials(
    scale: MOSScale,
    n_partials: int = 12,
    max_beta: int = 24,
    beta_penalty: float = 0.0,
) -> List[PartialMap]:
    """Map harmonics ``1 .. n_partials`` onto ``scale``'s lattice.

    Parameters
    ----------
    scale : MOSScale
    n_partials : int, default 12
        How far up the harmonic series to go.  Twelve reaches the point where
        Plomp-Levelt roughness between neighbouring partials of a single tone
        starts to dominate, which is where a real instrument's spectrum matters.
    max_beta, beta_penalty
        Passed to :func:`map_harmonic`.

    Returns
    -------
    list of PartialMap
        Always begins with the fundamental at ``(0, 0)``.

    Examples
    --------
    >>> from biotuner.mos.scale import MOSScale
    >>> m = MOSScale.from_signature(5, 2, tuning=31)
    >>> [(p.alpha, p.beta) for p in matched_partials(m, n_partials=8)]
    [(0, 0), (1, 0), (1, 1), (2, 0), (0, 4), (2, 1), (-3, 10), (3, 0)]

    Bound the chain and the seventh partial gives up on 7/4, settling for the
    two-generator approximation instead:

    >>> [(p.alpha, p.beta) for p in matched_partials(m, 8, max_beta=5)]
    [(0, 0), (1, 0), (1, 1), (2, 0), (0, 4), (2, 1), (4, -2), (3, 0)]
    """
    if n_partials < 1:
        raise ValueError(f"n_partials must be >= 1, got {n_partials!r}")
    return [
        map_harmonic(h, scale, max_beta=max_beta, beta_penalty=beta_penalty)
        for h in range(1, int(n_partials) + 1)
    ]


def matched_ratios(scale: MOSScale, n_partials: int = 12, **kwargs: Any) -> List[float]:
    """The retuned partials of :func:`matched_partials` as bare frequency ratios.

    Examples
    --------
    >>> from biotuner.mos.scale import MOSScale
    >>> m = MOSScale.from_signature(5, 2, tuning=31)
    >>> [round(r, 4) for r in matched_ratios(m, n_partials=6)]
    [1.0, 2.0, 2.991, 4.0, 5.0023, 5.9821]
    """
    return [p.ratio for p in matched_partials(scale, n_partials, **kwargs)]


# --------------------------------------------------------------------------- #
# Spectra
# --------------------------------------------------------------------------- #
def _amplitude_vector(n_partials: int, amplitudes: Optional[Sequence[float]]) -> np.ndarray:
    """``1/h`` roll-off by default, normalised so the loudest partial is 1."""
    if amplitudes is None:
        amps = np.array([1.0 / h for h in range(1, n_partials + 1)], dtype=float)
    else:
        amps = np.asarray(amplitudes, dtype=float).ravel()
        if amps.shape[0] != n_partials:
            raise ValueError(
                f"amplitudes has length {amps.shape[0]} but n_partials is "
                f"{n_partials}; pass one amplitude per partial or None for 1/h"
            )
    peak = float(np.max(np.abs(amps)))
    if peak <= 0.0:
        raise ValueError("amplitudes must contain at least one non-zero value")
    return amps / peak


def matched_spectrum(
    scale: MOSScale,
    fundamental: float = 250.0,
    n_partials: int = 12,
    amplitudes: Optional[Sequence[float]] = None,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """A sounding Dynamic Tonality spectrum: frequencies in Hz and amplitudes.

    Parameters
    ----------
    scale : MOSScale
    fundamental : float, default 250.0
        Hz.  Roughness is not scale-invariant -- the Plomp-Levelt critical band
        widens with frequency -- so the choice of fundamental changes every
        dissonance number downstream.  250 Hz sits in the middle of the range
        where the curve is best characterised.
    n_partials : int, default 12
    amplitudes : sequence of float, optional
        One value per partial.  ``None`` gives the ``1/h`` roll-off of an
        idealised sawtooth.  Whatever is passed is rescaled to peak at 1.
    **kwargs
        ``max_beta`` and ``beta_penalty``, forwarded to :func:`map_harmonic`.

    Returns
    -------
    (frequencies, amplitudes) : tuple of ndarray

    Examples
    --------
    >>> from biotuner.mos.scale import MOSScale
    >>> m = MOSScale.from_signature(5, 2, tuning=31)
    >>> f, a = matched_spectrum(m, fundamental=100.0, n_partials=5)
    >>> [round(float(x), 3) for x in f]
    [100.0, 200.0, 299.104, 400.0, 500.226]
    >>> [round(float(x), 4) for x in a]
    [1.0, 0.5, 0.3333, 0.25, 0.2]
    """
    if fundamental <= 0:
        raise ValueError(f"fundamental must be a positive frequency in Hz, got {fundamental!r}")
    ratios = matched_ratios(scale, n_partials, **kwargs)
    amps = _amplitude_vector(len(ratios), amplitudes)
    freqs = np.asarray(ratios, dtype=float) * float(fundamental)
    return freqs, amps


def dynamic_timbre(
    scale: MOSScale,
    n_partials: int = 12,
    fundamental: float = 250.0,
    amplitudes: Optional[Sequence[float]] = None,
    **kwargs: Any,
):
    """Package :func:`matched_spectrum` as a timbre object ready for synthesis.

    Returns a :class:`biotuner.harmonic_timbre.Timbre` when that subpackage
    imports, so the result drops straight into ``render_additive``, the
    exporters and the cross-modal sidecar.  If the import fails -- the
    subpackage is not a hard requirement of :mod:`biotuner.mos` -- a
    :class:`SimpleTimbre` with the same core fields is returned instead, and
    the substitution is recorded in ``metadata['timbre_class']``.

    Parameters
    ----------
    scale, n_partials, fundamental, amplitudes, **kwargs
        As :func:`matched_spectrum`.

    Returns
    -------
    Timbre or SimpleTimbre
        Carrying ``partials_hz``, ``amplitudes``, ``base_freq``,
        ``matched_tuning`` (the scale's ratios), ``matching_method``
        (``'dynamic_tonality'``) and a ``metadata`` dict recording the
        signature, generator and per-partial ``(alpha, beta)`` mapping.

    Examples
    --------
    >>> from biotuner.mos.scale import MOSScale
    >>> t = dynamic_timbre(MOSScale.from_signature(5, 2, tuning=31),
    ...                    n_partials=4, fundamental=100.0)
    >>> [round(float(f), 3) for f in t.partials_hz]
    [100.0, 200.0, 299.104, 400.0]
    >>> t.matching_method
    'dynamic_tonality'
    >>> t.metadata['signature'], t.metadata['lattice'][2]
    ('5L2s', (1, 1))
    """
    maps = matched_partials(scale, n_partials, **kwargs)
    amps = _amplitude_vector(len(maps), amplitudes)
    freqs = np.array([p.ratio for p in maps], dtype=float) * float(fundamental)

    metadata = {
        "signature": scale.signature,
        "generator_cents": scale.generator_cents,
        "period": scale.period,
        "lattice": [(p.alpha, p.beta) for p in maps],
        "error_cents": [p.error_cents for p in maps],
        "max_abs_beta": max(abs(p.beta) for p in maps),
    }
    common = dict(
        partials_hz=freqs,
        amplitudes=amps,
        base_freq=float(fundamental),
        matched_tuning=list(scale.ratios),
        matching_method="dynamic_tonality",
        metadata=metadata,
    )
    try:
        from biotuner.harmonic_timbre import Timbre
    except Exception as exc:  # pragma: no cover - depends on optional install
        metadata["timbre_class"] = f"SimpleTimbre (harmonic_timbre unavailable: {exc})"
        return SimpleTimbre(**common)
    metadata["timbre_class"] = "Timbre"
    return Timbre(**common)


# --------------------------------------------------------------------------- #
# Sensory dissonance
# --------------------------------------------------------------------------- #
def spectral_dissonance(freqs: Sequence[float], amps: Sequence[float]) -> float:
    """Total pairwise Plomp-Levelt roughness of a spectrum.

    Thin wrapper over :func:`biotuner.scale_construction.dissmeasure`, which
    sums the roughness of every pair of sinusoids using the minimum of the two
    amplitudes (the beat amplitude) rather than their product.  Kept here so
    that the whole Dynamic Tonality argument can be checked without leaving
    this module, and so the lazy import stays in one place.

    Parameters
    ----------
    freqs : sequence of float
        Frequencies in Hz.  Order does not matter; ``dissmeasure`` sorts.
    amps : sequence of float
        One amplitude per frequency.

    Returns
    -------
    float
        Unnormalised roughness.  Only differences between spectra of the same
        size and loudness are meaningful; the absolute value is not a scale.

    Examples
    --------
    An octave is nearly smooth, a semitone is not:

    >>> round(spectral_dissonance([250.0, 500.0], [1.0, 1.0]), 6)
    0.000809
    >>> round(spectral_dissonance([250.0, 265.0], [1.0, 1.0]), 4)
    0.8413
    """
    f = np.asarray(freqs, dtype=float).ravel()
    a = np.asarray(amps, dtype=float).ravel()
    if f.shape[0] != a.shape[0]:
        raise ValueError(
            f"freqs and amps must be the same length, got {f.shape[0]} "
            f"frequencies and {a.shape[0]} amplitudes"
        )
    if f.shape[0] < 2:
        return 0.0
    if np.any(f <= 0):
        raise ValueError("all frequencies must be positive, got a non-positive value")

    from biotuner.scale_construction import dissmeasure

    return float(dissmeasure(f, a, model="min"))


def scale_dissonance(
    scale: MOSScale,
    n_partials: int = 12,
    matched: bool = True,
    fundamental: float = 250.0,
    amplitudes: Optional[Sequence[float]] = None,
    **kwargs: Any,
) -> float:
    """Roughness of the whole scale sounded at once, every degree, every partial.

    This is the quantity Dynamic Tonality is trying to minimise.  Sounding the
    scale as a single simultaneity is a blunt instrument -- no real music plays
    all seven notes together -- but it is the right blunt instrument here,
    because it counts every partial-against-partial collision the tuning
    affords, with none of the arbitrariness of picking a chord progression.

    Parameters
    ----------
    scale : MOSScale
    n_partials : int, default 12
    matched : bool, default True
        ``True`` sounds each degree with the lattice-matched partials of
        :func:`matched_partials`; ``False`` with a plain harmonic series.  The
        two spectra have the same number of partials and the same amplitude
        envelope, so the comparison isolates partial *placement*.
    fundamental : float, default 250.0
        Hz of the scale's root.
    amplitudes : sequence of float, optional
    **kwargs
        ``max_beta`` and ``beta_penalty``; ignored when ``matched=False``.

    Returns
    -------
    float

    Examples
    --------
    >>> from biotuner.mos.scale import MOSScale
    >>> m = MOSScale.from_signature(5, 2, tuning=31)
    >>> round(scale_dissonance(m, matched=False), 4)
    29.8699
    >>> round(scale_dissonance(m, matched=True), 4)
    28.327
    """
    if n_partials < 1:
        raise ValueError(f"n_partials must be >= 1, got {n_partials!r}")
    if fundamental <= 0:
        raise ValueError(f"fundamental must be a positive frequency in Hz, got {fundamental!r}")

    if matched:
        partial_ratios = matched_ratios(scale, n_partials, **kwargs)
    else:
        partial_ratios = [float(h) for h in range(1, int(n_partials) + 1)]
    amps = _amplitude_vector(len(partial_ratios), amplitudes)

    degrees = np.asarray(scale.ratios, dtype=float)
    # Outer product: every degree carries the full spectrum.
    freqs = (
        float(fundamental)
        * degrees[:, None]
        * np.asarray(partial_ratios, dtype=float)[None, :]
    ).ravel()
    all_amps = np.tile(amps, degrees.shape[0])
    return spectral_dissonance(freqs, all_amps)


def dissonance_advantage(scale: MOSScale, **kwargs: Any) -> Dict[str, float]:
    """How much roughness the matched timbre saves over a harmonic one.

    Parameters
    ----------
    scale : MOSScale
    **kwargs
        Forwarded to :func:`scale_dissonance` (``n_partials``, ``fundamental``,
        ``amplitudes``, ``max_beta``, ``beta_penalty``).  ``matched`` is not
        accepted -- both settings are computed, that is the point.

    Returns
    -------
    dict
        ``'harmonic'`` and ``'matched'`` total dissonances, their difference
        ``'reduction'`` (positive means the matched timbre won), and
        ``'reduction_pct'`` as a percentage of the harmonic total.

    Examples
    --------
    31-EDO meantone is close enough to 12-EDO that a harmonic timbre already
    half works, and the matched one still takes 5 % off:

    >>> from biotuner.mos.scale import MOSScale
    >>> adv = dissonance_advantage(MOSScale.from_signature(5, 2, tuning=31))
    >>> round(adv['reduction'], 4), round(adv['reduction_pct'], 2)
    (1.5429, 5.17)

    Push the generator out to 714 cents, far from anything 12-EDO can spell,
    and bound the generator chain to lengths a listener can follow:

    >>> far = MOSScale.from_signature(5, 2, tuning=0.5952)
    >>> adv = dissonance_advantage(far, max_beta=5)
    >>> round(adv['reduction_pct'], 2)
    19.72
    """
    if "matched" in kwargs:
        raise TypeError(
            "dissonance_advantage() computes both matched and harmonic "
            "spectra; drop the 'matched' argument or call scale_dissonance() "
            "directly"
        )
    d_h = scale_dissonance(scale, matched=False, **kwargs)
    d_m = scale_dissonance(scale, matched=True, **kwargs)
    reduction = d_h - d_m
    return {
        "harmonic": d_h,
        "matched": d_m,
        "reduction": reduction,
        "reduction_pct": 100.0 * reduction / d_h if d_h else 0.0,
    }
