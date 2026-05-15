"""
Structural descriptors of a chord's ratio set.

The :mod:`biotuner.harmonic_geometry.media.coupling` module provides
*scalar* reductions of a chord (one float per call). This module
provides *structural* descriptors — ints, polygons, matrices — that
expose the intrinsic geometric / number-theoretic content of a ratio
set. They are designed to drive media parameters that scalar coupling
cannot reach:

- ``prime_limit`` (int) — highest prime in any p/q. Microtonal regime.
- ``tonnetz_polygon`` (list of 2-tuples) — each ratio projected onto
  the just-intonation lattice using per-prime direction vectors.
  Different chord = different polygon shape. Major triad and minor
  triad are mirror-flipped polygons in this projection.
- ``pairwise_harmonic_distance`` ((n, n) array) — Tenney-style
  log(p·q) of every interval in the chord. Diagonal is zero.
- ``cf_depths`` (list of int) — continued-fraction depth of each ratio.
  Higher = "more irrational", more dendritic when used as a depth knob.
- ``max_common_int`` (int) — largest integer in the chord's common-
  denominator integer form. Wider distinguishing range than
  ``coupling.ratio_complexity`` (which averages and loses the spread).

These descriptors are used in concert by morphogenetic media (e.g.
:class:`~biotuner.harmonic_geometry.media.morphogenetic.crystallization.Crystallization`)
to give each chord a unique structural signature rather than relying
on a single 1-D scalar coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from typing import Sequence

import numpy as np

from biotuner.harmonic_geometry.inputs import HarmonicInput


# Each prime > 2 gets a 2-D direction vector. Primes 3 and 5 use the
# canonical 5-limit Tonnetz layout (fifth axis east, third axis 60°).
# Higher primes get directions spaced around the circle so 7-, 11-,
# 13-limit chords still produce distinct polygons.
PRIME_DIRECTIONS: dict[int, tuple[float, float]] = {
    3:  (1.0, 0.0),                              # east
    5:  (0.5, 0.86602540378),                    # 60°
    7:  (-0.5, 0.86602540378),                   # 120°
    11: (-1.0, 0.0),                             # 180°
    13: (-0.5, -0.86602540378),                  # 240°
    17: (0.5, -0.86602540378),                   # 300°
    19: (0.86602540378, 0.5),                    # 30°
    23: (-0.86602540378, 0.5),                   # 150°
}

_MAX_DENOM = 10000


# ============================================================== helpers


def _factor(n: int) -> dict[int, int]:
    """Prime factorization ``{prime: exponent}`` of a positive integer."""
    n = abs(int(n))
    out: dict[int, int] = {}
    if n <= 1:
        return out
    d = 2
    while d * d <= n:
        while n % d == 0:
            out[d] = out.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        out[n] = out.get(n, 0) + 1
    return out


def _as_fraction(x) -> Fraction:
    return Fraction(float(x)).limit_denominator(_MAX_DENOM)


def _ratios_as_fractions(input: HarmonicInput) -> list[Fraction]:
    return [_as_fraction(r) for r in input.to_ratios()]


# ============================================================ prime_limit


def prime_limit(input: HarmonicInput) -> int:
    """Highest prime appearing in any numerator or denominator of the chord.

    Returns ``2`` for chords expressible with octaves only.

    Examples
    --------
    >>> # Major triad 1 : 5/4 : 3/2 → prime limit 5
    >>> # Sus4 1 : 4/3 : 3/2 → prime limit 3 (Pythagorean)
    >>> # 1 : 6/5 : 7/5 → prime limit 7
    """
    fracs = _ratios_as_fractions(input)
    max_p = 2
    for f in fracs:
        for n in (f.numerator, f.denominator):
            for p in _factor(n):
                if p > max_p:
                    max_p = p
    return int(max_p)


# ========================================================= tonnetz_polygon


def _tonnetz_position(frac: Fraction) -> tuple[float, float]:
    """Project a single ratio to 2-D via per-prime direction summation.

    Octave (prime 2) is folded out — it contributes no spatial offset.
    Primes outside :data:`PRIME_DIRECTIONS` are ignored (rare; the
    table covers everything up to 23-limit).
    """
    x = 0.0
    y = 0.0
    num_factors = _factor(frac.numerator)
    den_factors = _factor(frac.denominator)
    for p, e in num_factors.items():
        if p == 2 or p not in PRIME_DIRECTIONS:
            continue
        dx, dy = PRIME_DIRECTIONS[p]
        x += e * dx
        y += e * dy
    for p, e in den_factors.items():
        if p == 2 or p not in PRIME_DIRECTIONS:
            continue
        dx, dy = PRIME_DIRECTIONS[p]
        x -= e * dx
        y -= e * dy
    return x, y


def tonnetz_polygon(input: HarmonicInput) -> list[tuple[float, float]]:
    """Project each chord ratio onto the just-intonation lattice (2-D).

    The polygon (one vertex per ratio) is a structural fingerprint of
    the chord: major and minor triads, for instance, are mirror-flipped
    triangles; sus4 is a line; 4-note chords are quadrilaterals.

    Returns
    -------
    list of (x, y)
        One point per ratio in the chord, in input order. The first
        vertex is always near the origin (the unison) when the chord
        is rooted at ``1/1``; non-rooted chords get the polygon
        shifted by their first ratio's projection.
    """
    return [_tonnetz_position(f) for f in _ratios_as_fractions(input)]


# =============================================== pairwise_harmonic_distance


def pairwise_harmonic_distance(input: HarmonicInput) -> np.ndarray:
    """Tenney harmonic distance ``log2(p·q)`` between every pair of ratios.

    Returns
    -------
    np.ndarray of shape (n, n)
        Symmetric matrix; ``out[i, j]`` is the harmonic distance of the
        interval ``r_j / r_i`` reduced to lowest terms. Diagonal is 0.
    """
    fracs = _ratios_as_fractions(input)
    n = len(fracs)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            interval = fracs[j] / fracs[i]
            p = max(1, abs(interval.numerator))
            q = max(1, abs(interval.denominator))
            out[i, j] = float(np.log2(p * q))
    return out


# ================================================================ cf_depths


def _cf_depth(frac: Fraction, max_terms: int = 32) -> int:
    """Number of terms in the continued-fraction expansion of ``frac``."""
    n, d = frac.numerator, frac.denominator
    depth = 0
    while d != 0 and depth < max_terms:
        n, d = d, n % d
        depth += 1
    return int(depth)


def cf_depths(input: HarmonicInput) -> list[int]:
    """Continued-fraction depth of each ratio in the chord.

    A measure of how "complex" each ratio is as a rational; deep CF
    expansions mean the ratio is close to irrational. Useful as a
    per-arm branching-frequency knob.
    """
    return [_cf_depth(f) for f in _ratios_as_fractions(input)]


# =========================================================== max_common_int


def max_common_int(input: HarmonicInput) -> int:
    """Largest integer in the chord's common-denominator integer form.

    Re-expresses every ratio over a common denominator (the LCM of all
    denominators) and returns the maximum integer appearing in any
    numerator. This is a *single* descriptor that varies sharply
    between chords with similar :func:`coupling.ratio_complexity`:

    - ``1 : 5/4 : 3/2`` → ``4:5:6``  → max = ``6``
    - ``1 : 6/5 : 3/2`` → ``10:12:15`` → max = ``15``
    - ``1 : 4/3 : 3/2`` → ``6:8:9``  → max = ``9``
    - ``1 : 6/5 : 7/5`` → ``5:6:7``  → max = ``7``
    """
    fracs = _ratios_as_fractions(input)
    if not fracs:
        return 1
    # Compute LCM of denominators.
    lcm_denom = 1
    for f in fracs:
        lcm_denom = lcm_denom * f.denominator // gcd(lcm_denom, f.denominator)
    # Convert each ratio to its integer form over the common denominator.
    nums = [int(f.numerator * (lcm_denom // f.denominator)) for f in fracs]
    # Reduce by their own GCD so 4:5:6 (not 8:10:12).
    g = nums[0]
    for n in nums[1:]:
        g = gcd(g, n)
    if g > 1:
        nums = [n // g for n in nums]
    return int(max(nums))


# ============================================================ pc_rotation


def _pc_set_12tet(input: HarmonicInput) -> tuple[int, ...]:
    """Pitch-class set of the chord in 12-TET (semitones modulo 12)."""
    ratios = input.to_ratios()
    if not ratios:
        return ()
    pcs = set()
    for r in ratios:
        # Map to nearest semitone modulo 12.
        semitones = round(12.0 * np.log2(float(r))) % 12
        pcs.add(int(semitones))
    return tuple(sorted(pcs))


def pc_rotation_order(input: HarmonicInput) -> int:
    """Rotational symmetry order of the chord in 12-TET pitch-class space.

    Returns the number of distinct rotations of the pitch-class set
    that map it to itself; ranges over divisors of 12. Most chords
    are 1-fold (no rotational symmetry); the exceptions are the
    musically symmetric ones:

    - Diminished 7th ``{0,3,6,9}`` → 4-fold
    - Augmented triad ``{0,4,8}``  → 3-fold
    - Tritone dyad   ``{0,6}``     → 2-fold
    - Whole-tone     ``{0,2,4,6,8,10}`` → 6-fold
    """
    pcs = _pc_set_12tet(input)
    if not pcs:
        return 1
    pcs_set = set(pcs)
    order = 0
    for k in range(12):
        rotated = {(p + k) % 12 for p in pcs_set}
        if rotated == pcs_set:
            order += 1
    return int(order) or 1


# ============================================================== signature


@dataclass(frozen=True)
class ChordStructure:
    """Bundle of all structural descriptors of a chord.

    Returned by :func:`chord_signature`. Use this when you need
    several descriptors at once; otherwise call the individual
    functions directly.
    """

    prime_limit: int
    tonnetz_polygon: list[tuple[float, float]]
    pairwise_harmonic_distance: np.ndarray
    cf_depths: list[int]
    max_common_int: int
    pc_rotation_order: int

    @property
    def n_ratios(self) -> int:
        return len(self.tonnetz_polygon)


def chord_signature(input: HarmonicInput) -> ChordStructure:
    """Compute the full structural signature of a chord."""
    return ChordStructure(
        prime_limit=prime_limit(input),
        tonnetz_polygon=tonnetz_polygon(input),
        pairwise_harmonic_distance=pairwise_harmonic_distance(input),
        cf_depths=cf_depths(input),
        max_common_int=max_common_int(input),
        pc_rotation_order=pc_rotation_order(input),
    )


# ===================================================== per-ratio weighting


def per_ratio_consonance_weights(input: HarmonicInput) -> np.ndarray:
    """Per-ratio weighting derived from pairwise harmonic distance.

    Each ratio receives a weight reflecting its mean amplitude-weighted
    harmonic distance to the rest of the chord, then inverted so
    *consonant* ratios get *higher* weight. Useful for per-arm growth
    rates in morphogenetic media: arms in directions toward consonant
    chord components grow faster.

    Returns
    -------
    np.ndarray of length n
        Strictly positive, with mean ≈ 1.0.
    """
    hd = pairwise_harmonic_distance(input)
    amps = input.normalized_amplitudes()
    n = hd.shape[0]
    if n < 2:
        return np.ones(n)
    if amps.size != n:
        amps = np.ones(n) / n
    weights = np.zeros(n)
    for i in range(n):
        total_amp = 0.0
        total_d = 0.0
        for j in range(n):
            if i == j:
                continue
            total_amp += amps[j]
            total_d += amps[j] * hd[i, j]
        weights[i] = total_d / max(total_amp, 1e-12)
    # Invert: low HD = consonant = high weight. Center mean at 1.
    max_d = float(weights.max())
    if max_d <= 0:
        return np.ones(n)
    inverted = 1.0 + (max_d - weights) / max_d
    return inverted / inverted.mean()
