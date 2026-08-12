"""Scales with three step sizes, and the simplex that replaces the labyrinth.

A moment-of-symmetry scale has two step sizes, so once the counts ``(nL, ns)``
are fixed the tuning space is one-dimensional -- a single generator -- and that
line segment is exactly an arc of the scale labyrinth
(:func:`biotuner.mos.plotting.plot_labyrinth`).  Allow a **third** step size and
the tuning space becomes two-dimensional: the picture is no longer a circle of
arcs but a triangle.

The geometry
------------
A ternary scale has a large step ``L``, a medium step ``M`` and a small step
``s``, with counts ``(a, b, c)`` and ``a + b + c = N`` notes.  They must fill
the period::

    a*L + b*M + c*s = P

Normalising ``P = 1`` and substituting ``u = a*L``, ``v = b*M``, ``w = c*s``
gives ``u + v + w = 1`` with ``u, v, w > 0``: the open standard 2-simplex.  So
``(u, v, w)`` -- the *fraction of the period each step class occupies in total*
-- are natural barycentric coordinates for the tuning space, and every ternary
scale of a given word is one point of one triangle.

What makes the triangle worth drawing:

- The three **edges** are where one step size vanishes, and the scale
  degenerates to a binary (MOS-like) scale: ``w = 0`` leaves ``a*L + b*M = 1``
  with ``a + b`` notes, and so on.  The boundary of the ternary simplex *is* the
  two-step world, so the simplex is glued onto the ordinary labyrinth along its
  edges.  :meth:`TernaryScale.degenerate_to` names the binary scale on each
  edge, and the tests verify numerically both that the degrees converge to it
  and that -- when the surviving counts are co-prime -- the resulting pattern is
  a rotation of the Christoffel word, i.e. a genuine MOS and therefore literally
  an arc of the labyrinth.
- The three **vertices** are equal divisions: two step sizes vanish, leaving
  ``a``, ``b`` or ``c`` equal steps, so the ``u`` vertex is ``a``-EDO.
- The **centre** ``u = v = w = 1/3`` is *not* the equal-step scale.  The
  equal-step point is ``L = M = s = 1/N``, i.e. ``(u, v, w) = (a/N, b/N, c/N)``.
  The two coincide only when ``a = b = c``.
- Requiring ``L > M > s`` -- which is what makes the letter names mean anything
  -- selects one of the six orderings, so the canonical region is a
  sub-triangle: one of six, with vertices at the ``u`` vertex, the point
  ``(a, b, 0)/(a+b)`` on the ``w = 0`` edge, and the equal-step point.  The six
  regions are the cevian subdivision through the equal-step point, so they are
  equal in *number* but not in area: the canonical one covers
  ``b*c / ((a+b)*N)`` of the triangle, which is ``1/6`` only when
  ``a = b = c``.

One consequence deserves flagging because it defeats an obvious conjecture: the
equal-step point lies in the *interior* of every word's simplex and is simply
``N``-EDO, where each generic interval class has one size and the class
orderings are strictly separated.  So **no ternary word is improper everywhere**
-- propriety always holds on an open neighbourhood of the equal-step tuning.
What varies between words is how far that neighbourhood spreads: over the
thirty rotation classes of ``3L2M2s`` the proper share of the triangle runs
from 4.2% to 11.2%, and the two MV3 words are at the top of that ordering.
See :func:`proper_fraction`.

Variety
-------
For a ternary scale the step class already contains three distinct sizes, so
``max_variety >= 3`` always, and ``max_variety == 3`` -- every generic interval
class showing at most three specific sizes -- is the ternary analogue of
Myhill's property.  It is a strong filter: of the 30 rotation classes of
``3L2M2s``, exactly two survive it.

Unlike the two-step case, variety can in principle depend on *where in the
simplex* the scale is tuned, so :func:`ternary_words` samples several generic
interior points and requires the bound at all of them.  That is a sampled
criterion, not a proof -- though a cheap argument says the sample is not
fragile: every entry of the interval matrix is a linear form in ``(u, v, w)``
with non-negative rational coefficients (``p*u/a + q*v/b + r*w/c`` for integer
step counts ``p, q, r``), so two entries are either identically equal or equal
only on a line.  Variety is therefore constant off a finite
union of lines, and the sampled value is the generic one with probability 1.
Measured on a 90-step grid, eight assorted seven-note words came out at a
single variety across the whole interior, with one exception of six pixels
where ``L = M`` exactly and the scale is really binary.

References
----------
Milne, A.J., Carlé, M., Sethares, W.A., Noll, T., Holland, S. (2011).
Scratching the Scale Labyrinth. In *Mathematics and Computation in Music*,
LNAI 6726, 180--195.  The two-step case this module extends is implemented in
:mod:`biotuner.mos.theory` and :mod:`biotuner.mos.scale`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from biotuner.mos import metrics as MT
from biotuner.mos.plotting import (
    INK,
    JI_5_LIMIT,
    LARGE_COLOR,
    SIGNAL_COLOR,
    SMALL_COLOR,
)
from biotuner.mos.theory import PERIOD_CENTS

__all__ = [
    "TernaryScale",
    "ternary_words",
    "sampled_max_variety",
    "variety_sample_points",
    "ternary_atlas",
    "mos_substitution",
    "proper_fraction",
    "barycentric_to_xy",
    "plot_ternary_simplex",
    "plot_ternary_atlas",
    "LETTERS",
    "MEDIUM_COLOR",
    "VARIETY_SAMPLE_SEED",
]

#: Letters, in decreasing nominal step size.
LETTERS: Tuple[str, str, str] = ("L", "M", "s")

#: Colour for the medium step class, completing the two-step palette of
#: :mod:`biotuner.mos.plotting` (``LARGE_COLOR`` blue, ``SMALL_COLOR`` orange).
MEDIUM_COLOR = "#7B3FA0"

#: Seed behind :func:`variety_sample_points`.  Fixed so that
#: :func:`ternary_words` is reproducible run to run.
VARIETY_SAMPLE_SEED = 20110726

#: Closing tolerance on ``a*L + b*M + c*s == 1``.
_SUM_TOL = 1e-9

#: How far inside the simplex a sampled or plotted point must stay.  Right on an
#: edge a step size is zero, two degrees coincide, and the metrics layer
#: (rightly) refuses the scale.
_EDGE_EPS = 1e-4

_SQRT3_2 = math.sqrt(3.0) / 2.0


# --------------------------------------------------------------------------- #
# Word helpers
# --------------------------------------------------------------------------- #
def _check_word(word: str) -> Tuple[int, int, int]:
    """Validate a ternary word and return its ``(a, b, c)`` counts."""
    if not isinstance(word, str) or not word:
        raise ValueError(f"word must be a non-empty string over 'L','M','s', got {word!r}")
    bad = sorted(set(word) - set(LETTERS))
    if bad:
        raise ValueError(
            f"word may only contain 'L', 'M' and 's', got {word!r} "
            f"with unexpected letter(s) {bad}"
        )
    counts = (word.count("L"), word.count("M"), word.count("s"))
    if 0 in counts:
        missing = [ell for ell, n in zip(LETTERS, counts) if n == 0]
        raise ValueError(
            f"a ternary word needs all three step classes; {word!r} is missing "
            f"{missing} (counts L={counts[0]}, M={counts[1]}, s={counts[2]}). "
            "A two-step word belongs to biotuner.mos.scale.MOSScale."
        )
    return counts


def _canonical_rotation(word: str) -> str:
    """The lexicographically least rotation -- one representative per class."""
    return min(word[k:] + word[:k] for k in range(len(word)))


def _multiset_permutations(counts: Dict[str, int], length: int) -> Iterator[str]:
    """Every distinct arrangement of the multiset, in lexicographic order."""
    stack: List[str] = []
    remaining = dict(counts)

    def walk() -> Iterator[str]:
        if len(stack) == length:
            yield "".join(stack)
            return
        for letter in LETTERS:
            if remaining[letter]:
                remaining[letter] -= 1
                stack.append(letter)
                yield from walk()
                stack.pop()
                remaining[letter] += 1

    yield from walk()


# --------------------------------------------------------------------------- #
# Barycentric geometry
# --------------------------------------------------------------------------- #
def barycentric_to_xy(
    u: float, v: float, w: float
) -> Tuple[float, float]:
    """Barycentric ``(u, v, w)`` to Cartesian, on the unit equilateral triangle.

    The ``u`` vertex sits at the origin, the ``v`` vertex at ``(1, 0)`` and the
    ``w`` vertex at ``(1/2, sqrt(3)/2)``, so the picture is a real equilateral
    triangle rather than a right-angled shear of one.

    Examples
    --------
    >>> [round(t, 6) for t in barycentric_to_xy(1.0, 0.0, 0.0)]
    [0.0, 0.0]
    >>> [round(t, 6) for t in barycentric_to_xy(0.0, 0.0, 1.0)]
    [0.5, 0.866025]
    >>> [round(t, 6) for t in barycentric_to_xy(1 / 3, 1 / 3, 1 / 3)]
    [0.5, 0.288675]
    """
    total = float(u) + float(v) + float(w)
    if total <= 0.0:
        raise ValueError(
            f"barycentric coordinates must have a positive sum, got "
            f"u={u!r}, v={v!r}, w={w!r}"
        )
    u, v, w = float(u) / total, float(v) / total, float(w) / total
    return (v + 0.5 * w, _SQRT3_2 * w)


def _xy_to_barycentric(x: np.ndarray, y: np.ndarray):
    """Inverse of :func:`barycentric_to_xy`, vectorised over arrays."""
    w = y / _SQRT3_2
    v = x - 0.5 * w
    u = 1.0 - v - w
    return u, v, w


def variety_sample_points(
    n_points: int = 7, seed: int = VARIETY_SAMPLE_SEED, floor: float = 0.08
) -> Tuple[Tuple[float, float, float], ...]:
    """``n_points`` fixed pseudo-random interior points of the simplex.

    Uniform on the simplex (a flat Dirichlet), rejecting anything with a
    coordinate below ``floor`` so that no sampled tuning is nearly degenerate --
    near an edge two interval sizes can fall within the metrics layer's
    clustering tolerance and variety would be *under*-counted.

    Deterministic given ``seed``, which is what makes :func:`ternary_words`
    reproducible.

    Examples
    --------
    >>> pts = variety_sample_points()
    >>> len(pts), len(pts[0])
    (7, 3)
    >>> round(sum(pts[0]), 12), round(min(min(p) for p in pts), 3) >= 0.08
    (1.0, True)
    >>> variety_sample_points() == variety_sample_points()
    True
    """
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1, got {n_points}")
    if not 0.0 <= floor < 1.0 / 3.0:
        raise ValueError(f"floor must lie in [0, 1/3), got {floor!r}")
    rng = np.random.default_rng(seed)
    out: List[Tuple[float, float, float]] = []
    for _ in range(10000):
        if len(out) == n_points:
            break
        p = rng.dirichlet((1.0, 1.0, 1.0))
        if float(p.min()) >= floor:
            out.append((float(p[0]), float(p[1]), float(p[2])))
    if len(out) < n_points:  # pragma: no cover - unreachable for sane floors
        raise RuntimeError(
            f"could not draw {n_points} points with all coordinates >= {floor}"
        )
    return tuple(out)


# --------------------------------------------------------------------------- #
# The scale object
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TernaryScale:
    """A scale with three step sizes, one point of one ternary simplex.

    Parameters
    ----------
    word : str
        Step pattern over ``'L'``, ``'M'`` and ``'s'``.  All three letters must
        appear -- a two-letter word is an MOS and belongs to
        :class:`~biotuner.mos.scale.MOSScale`.
    large, medium, small : float
        The three step sizes as **fractions of the period**.  They must satisfy
        ``a*L + b*M + c*s == 1`` for the word's counts, i.e. the steps must fill
        the period exactly.  Their names are nominal: nothing forces
        ``L > M > s`` (see :attr:`is_canonical`), because the six orderings are
        six regions of the same triangle.
    period : float, default 2.0
        Period as a frequency ratio.

    Examples
    --------
    ``3L2M2s`` in the word ``LMLsLMs``, tuned so that the three classes take
    equal shares of the octave:

    >>> t = TernaryScale.from_barycentric('LMLsLMs', 1 / 3, 1 / 3, 1 / 3)
    >>> t.signature, t.cardinality, t.counts
    ('3L2M2s', 7, (3, 2, 2))
    >>> [round(c, 2) for c in t.step_cents]
    [133.33, 200.0, 200.0]

    Equal shares are not equal steps: with three ``L``'s against two ``M``'s the
    large step comes out *smaller* than the medium one, so this point is not in
    the ``L > M > s`` region at all.

    >>> t.is_canonical
    False

    The tuning that does make the letters honest lies toward the ``u`` vertex:

    >>> t = TernaryScale.from_barycentric('LMLsLMs', 0.52, 0.30, 0.18)
    >>> [round(c, 2) for c in t.step_cents]
    [208.0, 180.0, 108.0]
    >>> t.is_canonical, t.is_proper, t.max_variety
    (True, True, 3)
    >>> [round(c, 1) for c in t.cents]
    [0.0, 208.0, 388.0, 596.0, 704.0, 912.0, 1092.0]
    """

    word: str
    large: float
    medium: float
    small: float
    period: float = 2.0

    # ---------------------------------------------------------------- #
    # Validation
    # ---------------------------------------------------------------- #
    def __post_init__(self) -> None:
        counts = _check_word(self.word)
        sizes = (self.large, self.medium, self.small)
        for name, size in zip(("large", "medium", "small"), sizes):
            if not isinstance(size, (int, float)) or isinstance(size, bool):
                raise TypeError(f"{name} step must be a number, got {size!r}")
            if not math.isfinite(float(size)) or float(size) <= 0.0:
                raise ValueError(
                    f"every step size must be finite and > 0 (a vanishing step "
                    f"is an edge of the simplex, where the scale is binary); "
                    f"got {name} = {size!r}"
                )
        if not isinstance(self.period, (int, float)) or isinstance(self.period, bool):
            raise TypeError(f"period must be a number, got {self.period!r}")
        if not math.isfinite(float(self.period)) or float(self.period) <= 1.0:
            raise ValueError(
                f"period ratio must exceed 1 and be finite, got {self.period!r}"
            )
        total = sum(n * float(s) for n, s in zip(counts, sizes))
        if abs(total - 1.0) > _SUM_TOL:
            raise ValueError(
                f"the steps must fill the period exactly: "
                f"{counts[0]}*{self.large!r} + {counts[1]}*{self.medium!r} + "
                f"{counts[2]}*{self.small!r} = {total!r}, expected 1.0 "
                f"(off by {total - 1.0:.3e}, tolerance {_SUM_TOL:g}). "
                "Use TernaryScale.from_barycentric() to build a scale from "
                "period shares that need not be normalised."
            )

    # ---------------------------------------------------------------- #
    # Constructors
    # ---------------------------------------------------------------- #
    @classmethod
    def from_barycentric(
        cls, word: str, u: float, v: float, w: float, period: float = 2.0
    ) -> "TernaryScale":
        """Build from the period shares of the three step classes.

        ``u`` is the fraction of the period taken by *all* the large steps
        together, so the individual large step is ``u / a``.  The coordinates
        are normalised to sum to 1, since barycentric coordinates are only
        defined up to scale.

        Examples
        --------
        >>> t = TernaryScale.from_barycentric('LMLsLMs', 5, 3, 2)
        >>> [round(x, 6) for x in t.barycentric]
        [0.5, 0.3, 0.2]
        >>> [round(c, 3) for c in t.step_cents]
        [200.0, 180.0, 120.0]
        """
        a, b, c = _check_word(word)
        u, v, w = float(u), float(v), float(w)
        if min(u, v, w) <= 0.0:
            raise ValueError(
                f"barycentric coordinates must all be > 0 (the boundary of the "
                f"simplex is where a step vanishes and the scale turns binary); "
                f"got u={u!r}, v={v!r}, w={w!r}"
            )
        total = u + v + w
        return cls(word, u / (a * total), v / (b * total), w / (c * total), period)

    @classmethod
    def equal_step(cls, word: str, period: float = 2.0) -> "TernaryScale":
        """The tuning ``L = M = s = 1/N``: the word's cardinality as an EDO.

        This point lies in the interior of the simplex at
        ``(a/N, b/N, c/N)`` -- *not* at the centroid ``(1/3, 1/3, 1/3)`` unless
        ``a = b = c`` -- and every ternary scale is trivially proper and of
        variety 1 there.

        Examples
        --------
        >>> t = TernaryScale.equal_step('LMLsLMs')
        >>> [round(x, 6) for x in t.barycentric]
        [0.428571, 0.285714, 0.285714]
        >>> t.max_variety, t.is_proper
        (1, True)
        >>> [round(c, 3) for c in t.cents[:3]]
        [0.0, 171.429, 342.857]
        """
        counts = _check_word(word)
        n = sum(counts)
        return cls(word, 1.0 / n, 1.0 / n, 1.0 / n, period)

    # ---------------------------------------------------------------- #
    # Identity
    # ---------------------------------------------------------------- #
    @property
    def cardinality(self) -> int:
        """Total notes per period."""
        return len(self.word)

    @property
    def counts(self) -> Tuple[int, int, int]:
        """``(a, b, c)`` -- how many large, medium and small steps."""
        return (self.word.count("L"), self.word.count("M"), self.word.count("s"))

    @property
    def signature(self) -> str:
        """Compact signature, e.g. ``'3L2M2s'``."""
        a, b, c = self.counts
        return f"{a}L{b}M{c}s"

    @property
    def is_canonical(self) -> bool:
        """True when ``L > M > s``, i.e. the letters mean what they say.

        Only one of the simplex's six ordering regions satisfies this; the
        others are the same scale with the letters permuted.
        """
        return self.large > self.medium > self.small

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        per = "octave" if math.isclose(self.period, 2.0) else f"{self.period:.4g}"
        lc, mc, sc = self.step_cents
        return (
            f"TernaryScale({self.signature} {self.word}, "
            f"L={lc:.1f}c M={mc:.1f}c s={sc:.1f}c, period={per}, "
            f"MV{self.max_variety}{', proper' if self.is_proper else ', improper'})"
        )

    # ---------------------------------------------------------------- #
    # Tuning
    # ---------------------------------------------------------------- #
    @property
    def period_cents(self) -> float:
        """Size of the period in cents (1200 for an octave)."""
        return PERIOD_CENTS * math.log2(self.period)

    @property
    def step_fractions(self) -> Dict[str, float]:
        """``{'L': large, 'M': medium, 's': small}`` as period fractions."""
        return {"L": self.large, "M": self.medium, "s": self.small}

    @property
    def degrees(self) -> List[float]:
        """Scale degrees as period fractions, ascending from ``0``.

        Strictly increasing, starting at 0 and all below 1, because every step
        is positive and the steps sum to exactly one period.
        """
        sizes = self.step_fractions
        out: List[float] = []
        acc = 0.0
        for letter in self.word:
            out.append(acc)
            acc += sizes[letter]
        return out

    @property
    def ratios(self) -> List[float]:
        """Scale degrees as frequency ratios in ``[1, period)``."""
        return [self.period**d for d in self.degrees]

    @property
    def cents(self) -> List[float]:
        """Scale degrees in cents, ascending, starting at 0."""
        pc = self.period_cents
        return [d * pc for d in self.degrees]

    @property
    def step_cents(self) -> Tuple[float, float, float]:
        """``(large, medium, small)`` step sizes in cents."""
        pc = self.period_cents
        return (self.large * pc, self.medium * pc, self.small * pc)

    @property
    def barycentric(self) -> Tuple[float, float, float]:
        """``(u, v, w)``: the share of the period taken by each step class.

        Examples
        --------
        >>> [round(x, 6) for x in TernaryScale.equal_step('LMLsLMs').barycentric]
        [0.428571, 0.285714, 0.285714]
        """
        a, b, c = self.counts
        return (a * self.large, b * self.medium, c * self.small)

    @property
    def xy(self) -> Tuple[float, float]:
        """Position in the drawn triangle, via :func:`barycentric_to_xy`."""
        return barycentric_to_xy(*self.barycentric)

    # ---------------------------------------------------------------- #
    # Structure -- measured through biotuner.mos.metrics
    # ---------------------------------------------------------------- #
    @property
    def _raw(self) -> Tuple[List[float], float]:
        """The ``(cents, period_cents)`` pair the metrics layer accepts.

        A ternary scale is not an MOS, so it goes through
        :mod:`biotuner.mos.metrics` by the raw-pair route rather than as a
        scale object.  Nothing about propriety or variety is reimplemented here.
        """
        return (self.cents, self.period_cents)

    @property
    def interval_matrix(self) -> np.ndarray:
        """``(N, N-1)`` specific sizes of every generic interval, in cents."""
        return MT.interval_matrix(self._raw)

    @property
    def generic_interval_sizes(self) -> Dict[int, List[float]]:
        """``{k: distinct specific sizes of the k-step interval}``, in cents."""
        return MT.generic_interval_sizes(self._raw)

    @property
    def max_variety(self) -> int:
        """Largest number of distinct specific sizes over the generic classes.

        ``3`` is the ternary analogue of Myhill's property; a generic ternary
        tuning cannot do better, since the step class alone already holds three
        sizes.

        Examples
        --------
        >>> TernaryScale.from_barycentric('LMLsLMs', 0.52, 0.3, 0.18).max_variety
        3
        >>> TernaryScale.from_barycentric('LLLMMss', 0.52, 0.3, 0.18).max_variety
        7
        """
        sizes = self.generic_interval_sizes
        return max(len(v) for v in sizes.values())

    @property
    def is_proper(self) -> bool:
        """Rothenberg propriety, from :func:`biotuner.mos.metrics.is_proper`.

        For strict propriety call that function directly on
        ``(scale.cents, scale.period_cents)``.
        """
        return bool(MT.is_proper(self._raw))

    @property
    def propriety_margin(self) -> float:
        """Cents by which the generic classes clear each other; ``< 0`` if not.

        ``min_k [ min(class k+1) - max(class k) ]``.  Positive exactly when the
        scale is proper, and it says *how* proper -- which is what the propriety
        field of :func:`plot_ternary_simplex` shades.

        Examples
        --------
        At the equal-step tuning the classes are separated by a whole step:

        >>> round(TernaryScale.equal_step('LMLsLMs').propriety_margin, 3)
        171.429
        """
        return _propriety_margin(self.interval_matrix)

    # ---------------------------------------------------------------- #
    # Relatives
    # ---------------------------------------------------------------- #
    def rotations(self) -> List["TernaryScale"]:
        """The ``N`` modes: the same step sizes, the word rotated.

        Examples
        --------
        >>> [t.word for t in TernaryScale.equal_step('LMs').rotations()]
        ['LMs', 'MsL', 'sLM']
        """
        return [
            TernaryScale(
                self.word[k:] + self.word[:k],
                self.large,
                self.medium,
                self.small,
                self.period,
            )
            for k in range(self.cardinality)
        ]

    def degenerate_to(self, letter: str) -> Tuple[str, Tuple[int, int]]:
        """The binary scale on the simplex edge where ``letter``'s step vanishes.

        Deleting one letter from the word leaves a two-step pattern; the two
        survivors are relabelled ``'L'``/``'s'`` by their nominal order, so the
        result is directly comparable with
        :func:`biotuner.mos.theory.christoffel_word` and with
        :class:`~biotuner.mos.scale.MOSScale`.

        Returns
        -------
        (pattern, (n_large, n_small))
            The pattern need not be a genuine MOS: when the two surviving counts
            share a factor it is a periodic repetition rather than a well-formed
            scale.

        Examples
        --------
        Kill the small steps of ``LMLsLMs`` and the pentatonic drops out -- and
        it really is the pentatonic, a rotation of the Christoffel word, so this
        edge of the triangle is an arc of the labyrinth:

        >>> t = TernaryScale.equal_step('LMLsLMs')
        >>> t.degenerate_to('s')
        ('LsLLs', (3, 2))
        >>> from biotuner.mos.theory import christoffel_word
        >>> chris = christoffel_word(3, 2)
        >>> chris, 'LsLLs' in [chris[k:] + chris[:k] for k in range(5)]
        ('sLsLL', True)

        Killing the large steps leaves ``2L2s``, which is *not* well formed --
        co-primality fails and the pattern simply repeats:

        >>> t.degenerate_to('L')
        ('LsLs', (2, 2))
        """
        if letter not in LETTERS:
            raise ValueError(
                f"letter must be one of {list(LETTERS)}, got {letter!r}"
            )
        survivors = [ell for ell in LETTERS if ell != letter]
        relabel = {survivors[0]: "L", survivors[1]: "s"}
        pattern = "".join(relabel[ch] for ch in self.word if ch != letter)
        return pattern, (pattern.count("L"), pattern.count("s"))

    # ---------------------------------------------------------------- #
    # Interop
    # ---------------------------------------------------------------- #
    def to_dict(self) -> Dict[str, object]:
        """Flat, JSON-friendly summary -- one row of a table.

        Examples
        --------
        >>> d = TernaryScale.equal_step('LMLsLMs').to_dict()
        >>> d['signature'], d['cardinality'], d['max_variety']
        ('3L2M2s', 7, 1)
        """
        a, b, c = self.counts
        lc, mc, sc = self.step_cents
        u, v, w = self.barycentric
        return {
            "signature": self.signature,
            "word": self.word,
            "cardinality": self.cardinality,
            "n_large": a,
            "n_medium": b,
            "n_small": c,
            "period": self.period,
            "period_cents": self.period_cents,
            "step_large_cents": lc,
            "step_medium_cents": mc,
            "step_small_cents": sc,
            "u": u,
            "v": v,
            "w": w,
            "is_canonical": self.is_canonical,
            "is_proper": self.is_proper,
            "propriety_margin_cents": self.propriety_margin,
            "max_variety": self.max_variety,
            "cents": list(self.cents),
        }

    def summary(self) -> str:
        """Multi-line human-readable description.

        Examples
        --------
        >>> print(TernaryScale.from_barycentric('LMLsLMs', .52, .3, .18).summary())
        3L2M2s  (7 notes)   LMLsLMs
          steps          L = 208.000 c,  M = 180.000 c,  s = 108.000 c   (L > M > s)
          period         1200.000 c  (ratio 2.000000)
          barycentric    u = 0.520000,  v = 0.300000,  w = 0.180000
          equal step at  u = 0.428571,  v = 0.285714,  w = 0.285714   (7-EDO)
          degrees        0.0, 208.0, 388.0, 596.0, 704.0, 912.0, 1092.0
          structure      max variety 3,  proper (margin +8.000 c)
          edges          L -> 0 : 2L2s (4 notes),  M -> 0 : 3L2s (5 notes),  s -> 0 : 3L2s (5 notes)
        """
        a, b, c = self.counts
        n = self.cardinality
        lc, mc, sc = self.step_cents
        u, v, w = self.barycentric
        order = "L > M > s" if self.is_canonical else "letters not in size order"
        edges = []
        for letter in LETTERS:
            pattern, (nl, ns) = self.degenerate_to(letter)
            edges.append(f"{letter} -> 0 : {nl}L{ns}s ({len(pattern)} notes)")
        margin = self.propriety_margin
        verdict = "proper" if self.is_proper else "IMPROPER"
        lines = [
            f"{self.signature}  ({n} notes)   {self.word}",
            f"  steps          L = {lc:.3f} c,  M = {mc:.3f} c,  s = {sc:.3f} c"
            f"   ({order})",
            f"  period         {self.period_cents:.3f} c  (ratio {self.period:.6f})",
            f"  barycentric    u = {u:.6f},  v = {v:.6f},  w = {w:.6f}",
            f"  equal step at  u = {a / n:.6f},  v = {b / n:.6f},  "
            f"w = {c / n:.6f}   ({n}-EDO)",
            "  degrees        " + ", ".join(f"{x:.1f}" for x in self.cents),
            f"  structure      max variety {self.max_variety},  {verdict} "
            f"(margin {margin:+.3f} c)",
            "  edges          " + ",  ".join(edges),
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Structure of a word, sampled across its simplex
# --------------------------------------------------------------------------- #
def _propriety_margin(matrix: np.ndarray) -> float:
    """``min_k [ min(class k+1) - max(class k) ]`` in cents, from the matrix."""
    n_classes = matrix.shape[1]
    if n_classes < 2:  # pragma: no cover - a 2-note scale has one class
        return float("inf")
    highs = matrix.max(axis=0)[:-1]
    lows = matrix.min(axis=0)[1:]
    return float(np.min(lows - highs))


def sampled_max_variety(
    word: str,
    *,
    points: Optional[Sequence[Tuple[float, float, float]]] = None,
    cap: Optional[int] = None,
) -> int:
    """Largest max-variety of ``word`` over several interior tunings.

    Variety is a property of a *tuning*, not only of a word: two interval sizes
    that differ generically can coincide on a lower-dimensional subset of the
    simplex -- on the line ``L = M`` a ternary scale is really binary, and at
    the equal-step point it is an EDO.  Since the interval sizes are linear
    forms in ``(u, v, w)``, those coincidences are confined to a finite union of
    lines, so the sampled value is the generic one with probability 1.  It is
    still a sampled criterion, not a proof.

    Parameters
    ----------
    word : str
    points : sequence of (u, v, w), optional
        Tunings to test.  :func:`variety_sample_points` by default.
    cap : int, optional
        Stop as soon as the running maximum exceeds ``cap``.  The returned value
        is then a lower bound above ``cap``, which is all a filter needs.

    Examples
    --------
    >>> sampled_max_variety('LMLsLMs')
    3
    >>> sampled_max_variety('LLLMMss')
    7
    """
    _check_word(word)
    pts = variety_sample_points() if points is None else points
    best = 0
    for u, v, w in pts:
        scale = TernaryScale.from_barycentric(word, u, v, w)
        best = max(best, scale.max_variety)
        if cap is not None and best > cap:
            return best
    return best


def ternary_words(
    a: int,
    b: int,
    c: int,
    *,
    max_variety: int = 3,
    unique_up_to_rotation: bool = True,
    max_words: Optional[int] = None,
) -> List[str]:
    """Arrangements of ``a`` L's, ``b`` M's and ``c`` s's worth keeping.

    Every arrangement is generated in lexicographic order; by default only the
    lexicographically least rotation of each rotation class is kept (the modes
    of a scale share its variety, so filtering before or after rotation gives
    the same answer).  A word survives only if its max variety stays at or below
    ``max_variety`` at *all* of :func:`variety_sample_points` -- see
    :func:`sampled_max_variety` for why this is sampled rather than proved.

    Parameters
    ----------
    a, b, c : int
        Counts of the three step classes, each at least 1.
    max_variety : int, default 3
        3 is the ternary Myhill condition and the reason to bother: it typically
        admits a couple of words per signature out of dozens.
    unique_up_to_rotation : bool, default True
        Return one representative per rotation class rather than every mode.
    max_words : int, optional
        Stop after enumerating this many arrangements.  Required above 14 notes,
        where ``N! / (a! b! c!)`` is large enough to matter.

    Examples
    --------
    Two of the thirty rotation classes of ``3L2M2s`` are MV3:

    >>> ternary_words(3, 2, 2)
    ['LMLsLMs', 'LMLsMLs']

    Relaxing the bound lets more through:

    >>> len(ternary_words(3, 2, 2, max_variety=4))
    14

    All the modes, rather than one representative each -- two words times seven
    rotations, which is the check that the rotation filter is not losing any:

    >>> len(ternary_words(3, 2, 2, unique_up_to_rotation=False))
    14
    """
    for name, n in (("a", a), ("b", b), ("c", c)):
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError(
                f"{name} must be an int >= 1 (a ternary scale has all three "
                f"step classes), got {n!r}"
            )
    if max_variety < 3:
        raise ValueError(
            f"max_variety must be at least 3: the step class of a ternary scale "
            f"already holds three distinct sizes, so no ternary word can meet "
            f"{max_variety}"
        )
    n = a + b + c
    total = math.factorial(n) // (
        math.factorial(a) * math.factorial(b) * math.factorial(c)
    )
    if n > 14 and max_words is None:
        raise ValueError(
            f"{a}L{b}M{c}s has {n} notes and {total} arrangements; enumerating "
            "them all is slow. Pass max_words=<int> to enumerate a prefix "
            "anyway, or work signature by signature at 14 notes or fewer."
        )
    if max_words is not None and max_words < 1:
        raise ValueError(f"max_words must be >= 1, got {max_words}")

    points = variety_sample_points()
    out: List[str] = []
    counts = {"L": a, "M": b, "s": c}
    for seen, word in enumerate(_multiset_permutations(counts, n), start=1):
        if max_words is not None and seen > max_words:
            break
        if unique_up_to_rotation and _canonical_rotation(word) != word:
            continue
        if sampled_max_variety(word, points=points, cap=max_variety) <= max_variety:
            out.append(word)
    return out


def proper_fraction(
    word: str,
    *,
    resolution: int = 60,
    canonical_only: bool = False,
    period: float = 2.0,
) -> float:
    """Fraction of the word's simplex on which the scale is proper.

    Estimated by counting a regular Cartesian grid of interior points, so it is
    an unbiased estimate of the area fraction to within the grid spacing.

    Every ternary word is proper on *some* open set -- the equal-step point is
    an interior point of the simplex and is just ``N``-EDO, where the generic
    classes are strictly separated -- so this is never exactly zero, however
    clumsily the word is arranged.  What it does do is rank words: across the
    thirty rotation classes of ``3L2M2s`` it runs from 0.042 to 0.112, and the
    top of that ordering is exactly the MV3 pair.

    Parameters
    ----------
    word : str
    resolution : int, default 60
        Grid samples across the base of the triangle.
    canonical_only : bool, default False
        Restrict to the ``L > M > s`` sub-region.  There the equal-step point is
        a *corner*, not an interior point, so a word can be improper on
        (almost) all of it.
    period : float, default 2.0

    Examples
    --------
    Propriety is demanding once three step sizes are free to separate: the MV3
    word holds it on a ninth of its triangle, the clumped word on a
    twenty-third of it.

    >>> round(proper_fraction('LMLsLMs', resolution=48), 3)
    0.11
    >>> round(proper_fraction('LLLMMss', resolution=48), 3)
    0.043
    """
    _check_word(word)
    values, mask = _field_grid(
        word, "propriety", resolution=resolution, period=period,
        canonical_only=canonical_only,
    )
    inside = np.count_nonzero(mask)
    if inside == 0:
        raise ValueError(
            f"resolution {resolution} is too coarse to place any sample point "
            f"inside the region; use a larger resolution"
        )
    return float(np.count_nonzero(values[mask] >= 0.0) / inside)


def mos_substitution(binary_word: str, sub_large: str, sub_small: str) -> str:
    """Build a ternary word by substituting into a two-letter MOS word.

    Each ``'L'`` of the MOS word becomes ``sub_large`` and each ``'s'`` becomes
    ``sub_small``.  This is the constructive route to musically sensible ternary
    scales: the parent MOS supplies the maximal evenness, and the substitution
    splits one or both of its step classes.

    The result is *not* guaranteed to be ternary (both substitutions could avoid
    ``'M'``) nor MV3; it is a string, and :class:`TernaryScale` or
    :func:`sampled_max_variety` is what judges it.  It does stack the odds:
    substituting every part of length 1--3 into every Christoffel word up to 7
    notes and keeping the ternary products of at most 12 notes gives 1840
    rotation classes, of which 89 are MV3 -- a rate of 4.8%, against 0.27% for
    the 67395 arbitrary rotation classes over the same signatures.  An 18-fold
    enrichment, re-measured by the test suite rather than taken on trust.

    Examples
    --------
    Splitting the large step of the 3-note MOS ``sLL`` into ``LM``, and its
    small step into ``LMs``:

    >>> from biotuner.mos.theory import christoffel_word
    >>> word = mos_substitution(christoffel_word(2, 1), 'LM', 'LMs')
    >>> word
    'LMsLMLM'
    >>> sampled_max_variety(word)
    3

    Splitting only the large step of the diatonic gives a 12-note MV3 word:

    >>> mos_substitution(christoffel_word(5, 2), 'LM', 's')
    'sLMLMsLMLMLM'
    """
    if not isinstance(binary_word, str) or not binary_word:
        raise ValueError(
            f"binary_word must be a non-empty string over 'L','s', got "
            f"{binary_word!r}"
        )
    bad = sorted(set(binary_word) - {"L", "s"})
    if bad:
        raise ValueError(
            f"binary_word may only contain 'L' and 's', got {binary_word!r} "
            f"with unexpected letter(s) {bad}"
        )
    for name, sub in (("sub_large", sub_large), ("sub_small", sub_small)):
        if not isinstance(sub, str) or not sub:
            raise ValueError(f"{name} must be a non-empty string, got {sub!r}")
        bad = sorted(set(sub) - set(LETTERS))
        if bad:
            raise ValueError(
                f"{name} may only contain 'L', 'M' and 's', got {sub!r} with "
                f"unexpected letter(s) {bad}"
            )
    return "".join(sub_large if ch == "L" else sub_small for ch in binary_word)


# --------------------------------------------------------------------------- #
# The atlas
# --------------------------------------------------------------------------- #
def ternary_atlas(
    cardinality: int,
    *,
    max_variety: int = 3,
    proper_resolution: int = 24,
) -> pd.DataFrame:
    """Every ternary signature at one cardinality that admits an MV word.

    One row per ``(a, b, c)`` with at least one admissible word, sorted by how
    many words it admits.

    Parameters
    ----------
    cardinality : int
        Notes per period; at least 3 (one of each step class).
    max_variety : int, default 3
        Passed to :func:`ternary_words`.
    proper_resolution : int, default 24
        Grid for the ``example_proper_fraction`` column.  Set small: this is a
        survey column, and the cost is quadratic.

    Returns
    -------
    pandas.DataFrame
        Columns ``n_large``, ``n_medium``, ``n_small``, ``cardinality``,
        ``signature``, ``n_words``, ``example_word``, ``equal_step_proper``,
        ``example_proper_fraction``.  ``equal_step_proper`` is a smoke test, not
        a discriminator: it is ``True`` for every scale, because the equal-step
        tuning is that cardinality's EDO.

    Examples
    --------
    Every one of the fifteen signatures at seven notes admits an MV3 word, and
    nine of them admit exactly two -- a word and its reversal:

    >>> atlas = ternary_atlas(7, proper_resolution=16)
    >>> list(atlas.columns)[:6]
    ['n_large', 'n_medium', 'n_small', 'cardinality', 'signature', 'n_words']
    >>> len(atlas), int(atlas['n_words'].sum())
    (15, 24)
    >>> row = atlas.iloc[0]
    >>> row['signature'], int(row['n_words']), row['example_word']
    ('1L1M5s', 2, 'LssMsss')
    >>> bool(atlas['equal_step_proper'].all())
    True
    """
    if cardinality < 3:
        raise ValueError(
            f"a ternary scale needs at least 3 notes, one per step class, got "
            f"{cardinality}"
        )
    rows = []
    for a in range(1, cardinality - 1):
        for b in range(1, cardinality - a):
            c = cardinality - a - b
            words = ternary_words(a, b, c, max_variety=max_variety)
            if not words:
                continue
            example = words[0]
            rows.append(
                {
                    "n_large": a,
                    "n_medium": b,
                    "n_small": c,
                    "cardinality": cardinality,
                    "signature": f"{a}L{b}M{c}s",
                    "n_words": len(words),
                    "example_word": example,
                    "equal_step_proper": TernaryScale.equal_step(example).is_proper,
                    "example_proper_fraction": proper_fraction(
                        example, resolution=proper_resolution
                    ),
                }
            )
    frame = pd.DataFrame(
        rows,
        columns=[
            "n_large", "n_medium", "n_small", "cardinality", "signature",
            "n_words", "example_word", "equal_step_proper",
            "example_proper_fraction",
        ],
    )
    if not frame.empty:
        frame = frame.sort_values(
            ["n_words", "signature"], ascending=[False, True]
        ).reset_index(drop=True)
    return frame


# --------------------------------------------------------------------------- #
# Fields over the simplex
# --------------------------------------------------------------------------- #
def _evaluate(
    word: str,
    u: float,
    v: float,
    w: float,
    field: str,
    targets: Sequence[float],
    period: float,
) -> float:
    """One field value at one interior point."""
    scale = TernaryScale.from_barycentric(word, u, v, w, period)
    if field == "propriety":
        return scale.propriety_margin
    if field == "variety":
        return float(scale.max_variety)
    if field == "ji_error":
        return float(MT.ji_error(scale._raw, targets)["mean_abs"])
    raise ValueError(  # pragma: no cover - guarded by the caller
        f"field must be 'propriety', 'variety', 'ji_error' or None, got {field!r}"
    )


def _canonical_vertices(counts: Tuple[int, int, int]) -> np.ndarray:
    """The ``L > M > s`` sub-triangle, as three barycentric rows.

    The three lines ``u/a = v/b``, ``v/b = w/c``, ``u/a = w/c`` all pass through
    the equal-step point, so they are cevians and cut the simplex into six
    ordering regions.  The canonical one has the ``u`` vertex, the point where
    ``u/a = v/b`` meets the ``w = 0`` edge, and the equal-step point as corners.
    """
    a, b, c = counts
    n = a + b + c
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [a / (a + b), b / (a + b), 0.0],
            [a / n, b / n, c / n],
        ]
    )


def _cell_centres(shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """The ``(x, y)`` a field of this shape actually sits at under ``imshow``.

    ``imshow`` with ``extent=(0, 1, 0, sqrt(3)/2)`` centres cell ``[i, j]`` on
    ``((j + 0.5) / nx, (i + 0.5) / ny * sqrt(3)/2)`` -- which is exactly where
    :func:`_field_grid` sampled it.  ``contour`` has to be given those same
    coordinates: handing it ``linspace(0, 1, nx)`` instead would draw the zero
    level half a cell off, and stretched by ``nx / (nx - 1)``, so the propriety
    boundary would not sit on the colour change it is supposed to mark.
    """
    ny, nx = shape
    return (
        (np.arange(nx) + 0.5) / nx,
        (np.arange(ny) + 0.5) / ny * _SQRT3_2,
    )


def _field_grid(
    word: str,
    field: str,
    *,
    resolution: int,
    targets: Sequence[float] = JI_5_LIMIT,
    period: float = 2.0,
    canonical_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Field values on a Cartesian grid over the triangle, plus an inside mask.

    Rows run bottom-to-top, so the arrays drop straight into ``imshow`` with
    ``origin='lower'`` and ``extent=(0, 1, 0, sqrt(3)/2)``.
    """
    if resolution < 4:
        raise ValueError(f"resolution must be at least 4, got {resolution}")
    counts = _check_word(word)
    nx = int(resolution)
    ny = max(4, int(round(resolution * _SQRT3_2)))
    xs, ys = _cell_centres((ny, nx))
    X, Y = np.meshgrid(xs, ys)
    U, V, W = _xy_to_barycentric(X, Y)
    mask = (U > _EDGE_EPS) & (V > _EDGE_EPS) & (W > _EDGE_EPS)
    if canonical_only:
        a, b, c = counts
        mask &= (U / a > V / b) & (V / b > W / c)
    values = np.full(X.shape, np.nan)
    for i, j in zip(*np.nonzero(mask)):
        values[i, j] = _evaluate(
            word, U[i, j], V[i, j], W[i, j], field, targets, period
        )
    return values, mask


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _margin_norm(values: np.ndarray) -> TwoSlopeNorm:
    """Diverging norm pinned at zero, each side scaled to its own extreme.

    The propriety margin is wildly asymmetric -- a few hundred cents improper
    against a few tens proper -- so a symmetric ramp would wash the proper
    region out to white.
    """
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    return TwoSlopeNorm(vcenter=0.0, vmin=min(lo, -1e-9), vmax=max(hi, 1e-9))


def _triangle_frame(ax, counts: Tuple[int, int, int], compact: bool) -> None:
    """Outline, vertices, edge labels and the canonical sub-region."""
    a, b, c = counts
    n = a + b + c
    corners = {
        "u": barycentric_to_xy(1, 0, 0),
        "v": barycentric_to_xy(0, 1, 0),
        "w": barycentric_to_xy(0, 0, 1),
    }
    tri = np.array([corners["u"], corners["v"], corners["w"], corners["u"]])
    # zorder 8: the canonical wedge below carries a 3pt white understroke, and
    # its first edge runs along the w = 0 base, so a lower outline would be
    # wiped out along exactly that stretch.
    ax.plot(tri[:, 0], tri[:, 1], color=INK, lw=1.4, zorder=8)

    # Vertices: two step classes have vanished, leaving an equal division.
    for key, count, colour in (
        ("u", a, LARGE_COLOR), ("v", b, MEDIUM_COLOR), ("w", c, SMALL_COLOR)
    ):
        x, y = corners[key]
        ax.plot([x], [y], "o", ms=7 if not compact else 4, color=colour,
                mec="white", mew=1.0, zorder=8, clip_on=False)
    if not compact:
        ax.text(corners["u"][0] - 0.045, corners["u"][1] - 0.045,
                f"{a}-EDO\nM, s → 0", fontsize=8.5, color=LARGE_COLOR,
                ha="right", va="top")
        ax.text(corners["v"][0] + 0.045, corners["v"][1] - 0.045,
                f"{b}-EDO\nL, s → 0", fontsize=8.5, color=MEDIUM_COLOR,
                ha="left", va="top")
        ax.text(corners["w"][0], corners["w"][1] + 0.05, f"{c}-EDO   L, M → 0",
                fontsize=8.5, color=SMALL_COLOR, ha="center", va="bottom")

        # Edges: one step class vanishes and the scale turns binary.
        stub = TernaryScale.equal_step("".join(LETTERS[i] * k
                                               for i, k in enumerate(counts)))
        for letter, key0, key1, rot, offset in (
            ("s", "u", "v", 0.0, (0.0, -0.035)),
            ("M", "u", "w", 60.0, (-0.035, 0.018)),
            ("L", "v", "w", -60.0, (0.035, 0.018)),
        ):
            pattern, (nl, ns) = stub.degenerate_to(letter)
            x0, y0 = corners[key0]
            x1, y1 = corners[key1]
            ax.text(
                (x0 + x1) / 2.0 + offset[0], (y0 + y1) / 2.0 + offset[1],
                f"{letter} → 0 :  {nl}L{ns}s,  {len(pattern)} notes",
                fontsize=8.5, color=INK, ha="center", va="center",
                rotation=rot, rotation_mode="anchor", zorder=9,
                # These sit half on the field, which for a top-of-scale variety
                # plot is near-black: dark ink on it is unreadable without a
                # halo.  Same reason the wedge gets an understroke.
                path_effects=[pe.withStroke(linewidth=2.6, foreground="white")],
            )

    # The L > M > s region: one of six, but not one sixth by area.
    region = np.array([barycentric_to_xy(*row) for row in _canonical_vertices(counts)])
    # White understroke: the field underneath can be anything from pale yellow
    # to navy, and a bare dark dashed line disappears into the dark end.
    ax.fill(region[:, 0], region[:, 1], facecolor="none", edgecolor=INK,
            lw=1.3, ls="--", zorder=7,
            path_effects=[pe.withStroke(linewidth=3.0, foreground="white")])
    area = b * c / ((a + b) * n)  # exact share of the triangle, see the docstring
    if not compact and area > 0.05:
        centre = region.mean(axis=0)
        ax.text(centre[0], centre[1], "L > M > s", fontsize=8.5, color=INK,
                ha="center", va="center", zorder=9,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                          alpha=0.7))

    # The equal-step tuning: N-EDO, and the meeting point of all six orderings.
    #
    # Marker only, deliberately: every word is proper on an open neighbourhood
    # of *this exact point*, so the proper region is always a blob centred on
    # it and any boxed callout at a fixed offset lands on top of the one
    # feature a propriety plot exists to show.  The legend carries the
    # "equal step (N-EDO)" wording instead.
    ex, ey = barycentric_to_xy(a / n, b / n, c / n)
    ax.plot([ex], [ey], marker="s", ms=8 if not compact else 4, color="white",
            mec=INK, mew=1.5, zorder=9)


def _mark_xy(mark) -> Tuple[float, float]:
    """Accept a TernaryScale or a barycentric triple and return xy."""
    if isinstance(mark, TernaryScale):
        return mark.xy
    seq = tuple(float(x) for x in mark)
    if len(seq) != 3:
        raise ValueError(
            f"mark must be a TernaryScale or a (u, v, w) triple, got {mark!r}"
        )
    return barycentric_to_xy(*seq)


def plot_ternary_simplex(
    word: str,
    *,
    field: Optional[str] = "propriety",
    resolution: int = 240,
    targets: Sequence[float] = JI_5_LIMIT,
    period: float = 2.0,
    mark: Union[None, "TernaryScale", Sequence[float]] = None,
    ax=None,
    figsize: Tuple[float, float] = (8.0, 7.2),
):
    """The tuning simplex of one ternary word.

    The two-dimensional replacement for a labyrinth arc.  Every interior point
    is one tuning of ``word``; the edges are the binary scales it degenerates
    to, the vertices are equal divisions, and the dashed sub-triangle is the
    region where ``L > M > s``.

    Parameters
    ----------
    word : str
        Step pattern over ``'L'``, ``'M'``, ``'s'``.
    field : {'propriety', 'variety', 'ji_error', None}, default 'propriety'
        What to shade.  ``'propriety'`` shades
        :attr:`TernaryScale.propriety_margin` on a diverging scale with the
        zero contour drawn, so proper and improper regions are separated by a
        line rather than by a colour judgement.  ``'variety'`` shades the max
        variety as integers.  ``'ji_error'`` shades the mean cents error from
        ``targets`` to the nearest degree.  ``None`` draws the geometry alone.
    resolution : int, default 240
        Grid samples across the base of the triangle.
    targets : sequence of float, default 5-limit consonances
    period : float, default 2.0
    mark : TernaryScale or (u, v, w), optional
        A tuning to mark.
    ax : matplotlib axes, optional
    figsize : tuple

    Returns
    -------
    (fig, ax)

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> fig, ax = plot_ternary_simplex('LMLsLMs', resolution=40)
    >>> round(ax.get_aspect() if isinstance(ax.get_aspect(), float) else 1.0, 3)
    1.0
    >>> plt.close(fig)
    """
    if field not in ("propriety", "variety", "ji_error", None):
        raise ValueError(
            f"field must be 'propriety', 'variety', 'ji_error' or None, got "
            f"{field!r}"
        )
    counts = _check_word(word)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    extent = (0.0, 1.0, 0.0, _SQRT3_2)
    if field is not None:
        values, mask = _field_grid(
            word, field, resolution=resolution, targets=targets, period=period
        )
        shown = np.ma.masked_where(~mask, values)
        if field == "propriety":
            # Two slopes, not one symmetric range: the improper side runs several
            # times further from zero than the proper side, and a symmetric ramp
            # would render the whole proper region as almost-white.
            im = ax.imshow(shown, origin="lower", extent=extent, cmap="RdBu",
                           norm=_margin_norm(values[mask]),
                           interpolation="nearest", zorder=1)
            label = "propriety margin (cents) — positive is proper"
            # The zero level is the actual boundary; draw it rather than trust
            # the eye to read it off a colour ramp.
            cx, cy = _cell_centres(values.shape)
            ax.contour(
                cx, cy, np.where(mask, values, np.nan), levels=[0.0],
                colors=[INK], linewidths=1.1, zorder=5,
            )
        elif field == "variety":
            # Fixed floor at 3 (no ternary scale can do better) and a ceiling of
            # at least 6, so the colour of MV3 means the same thing from one word
            # to the next -- and so an all-MV3 word does not come out uniformly
            # dark with the annotations unreadable on top of it.
            lo = 3
            hi = max(6, int(np.nanmax(values[mask])))
            cmap = plt.get_cmap("YlGnBu", hi - lo + 1)
            im = ax.imshow(shown, origin="lower", extent=extent, cmap=cmap,
                           vmin=lo - 0.5, vmax=hi + 0.5,
                           interpolation="nearest", zorder=1)
            seen = np.unique(values[mask])
            # Each interval-matrix entry is a linear form in (u, v, w), so two
            # entries either coincide identically or only on a line: variety is
            # constant off a finite union of lines, and a flat field is the
            # expected outcome rather than a failed plot.
            label = (
                f"max variety = {int(seen[0])} everywhere in the interior"
                if len(seen) == 1 else "max variety"
            )
        else:
            im = ax.imshow(shown, origin="lower", extent=extent, cmap="magma_r",
                           interpolation="nearest", zorder=1)
            label = f"mean cents error to {len(targets)} just intervals"
        cbar = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.03)
        cbar.set_label(label, fontsize=9, color=INK)
        if field == "variety":
            cbar.set_ticks(list(range(lo, hi + 1)))

    _triangle_frame(ax, counts, compact=False)

    if mark is not None:
        mx, my = _mark_xy(mark)
        ax.plot([mx], [my], marker="*", ms=17, color=SIGNAL_COLOR, mec="white",
                mew=0.9, zorder=10)

    a, b, c = counts
    n = a + b + c
    ax.set_xlim(-0.20, 1.20)
    ax.set_ylim(-0.17, 1.02)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        f"{a}L{b}M{c}s   {word}   —   the tuning simplex\n"
        f"u, v, w = share of the period taken by all L, all M, all s",
        fontsize=12, color=INK,
    )
    handles = [
        Line2D([], [], color=INK, ls="--", lw=1.3, label="L > M > s region"),
        Line2D([], [], color=INK, marker="s", ls="none", mfc="white", mew=1.5,
               ms=8, label=f"equal step ({n}-EDO)"),
    ]
    if field == "propriety":
        handles.append(Line2D([], [], color=INK, lw=1.1, label="propriety boundary"))
    if mark is not None:
        handles.append(
            Line2D([], [], color=SIGNAL_COLOR, marker="*", ls="none", ms=13,
                   label="marked tuning")
        )
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.02, 1.0),
              fontsize=8.5, frameon=False)
    return fig, ax


def plot_ternary_atlas(
    cardinality: int,
    *,
    max_variety: int = 3,
    max_panels: int = 12,
    resolution: int = 70,
    figsize: Optional[Tuple[float, float]] = None,
):
    """A grid of simplices, one per admissible signature at one cardinality.

    Each panel is the propriety field of that signature's first admissible word,
    drawn on the same colour scale so the panels are comparable: blue is proper,
    red improper, and the dashed wedge is the ``L > M > s`` region.

    Parameters
    ----------
    cardinality : int
    max_variety : int, default 3
    max_panels : int, default 12
        Signatures are ordered by how many words they admit; the rest are
        dropped and the title says so.
    resolution : int, default 70
        Per-panel grid.  Small on purpose -- this is a survey.
    figsize : tuple, optional

    Returns
    -------
    (fig, axes)

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> fig, axes = plot_ternary_atlas(7, max_panels=4, resolution=24)
    >>> len(axes)
    4
    >>> plt.close(fig)
    """
    atlas = ternary_atlas(cardinality, max_variety=max_variety,
                          proper_resolution=8)
    if atlas.empty:
        raise ValueError(
            f"no signature at {cardinality} notes admits a word of variety "
            f"<= {max_variety}"
        )
    shown = atlas.head(max_panels)
    n_panels = len(shown)
    ncols = min(4, n_panels)
    nrows = int(math.ceil(n_panels / ncols))
    if figsize is None:
        figsize = (3.3 * ncols, 3.4 * nrows + 0.9)
    # ``constrained_layout=True`` rather than ``layout='constrained'``: the
    # latter needs matplotlib 3.6 and the package supports 3.5.3.
    fig, grid = plt.subplots(nrows, ncols, figsize=figsize,
                             constrained_layout=True)
    axes = list(np.atleast_1d(grid).ravel())

    fields = []
    for _, row in shown.iterrows():
        values, mask = _field_grid(
            row["example_word"], "propriety", resolution=resolution
        )
        fields.append((values, mask))
    # One norm for every panel, so the colours are comparable across the atlas.
    norm = _margin_norm(np.concatenate([v[m] for v, m in fields]))

    extent = (0.0, 1.0, 0.0, _SQRT3_2)
    im = None
    for ax, (_, row), (values, mask) in zip(axes, shown.iterrows(), fields):
        im = ax.imshow(np.ma.masked_where(~mask, values), origin="lower",
                       extent=extent, cmap="RdBu", norm=norm,
                       interpolation="nearest", zorder=1)
        cx, cy = _cell_centres(values.shape)
        ax.contour(
            cx, cy, np.where(mask, values, np.nan), levels=[0.0], colors=[INK],
            linewidths=0.9, zorder=5,
        )
        _triangle_frame(ax, _check_word(row["example_word"]), compact=True)
        ax.set_xlim(-0.06, 1.06)
        ax.set_ylim(-0.06, _SQRT3_2 + 0.06)
        ax.set_aspect("equal")
        ax.axis("off")
        # Read the proper share off the panel's own field rather than the
        # atlas column: that column is sampled far more coarsely and rounds a
        # small proper region down to nothing.
        share = float(np.count_nonzero(values[mask] >= 0.0) / np.count_nonzero(mask))
        ax.set_title(
            f"{row['signature']}   {row['example_word']}\n"
            f"{row['n_words']} MV{max_variety} word"
            f"{'s' if row['n_words'] != 1 else ''},  "
            f"{share:.0%} of the simplex proper",
            fontsize=9, color=INK,
        )
    for ax in axes[n_panels:]:
        ax.axis("off")

    cbar = fig.colorbar(im, ax=axes[:n_panels], shrink=0.55, pad=0.01)
    cbar.set_label("propriety margin (cents)", fontsize=9, color=INK)
    fig.legend(
        handles=[
            Line2D([], [], color=LARGE_COLOR, marker="o", ls="none", ms=6,
                   label="corner: only L survives"),
            Line2D([], [], color=MEDIUM_COLOR, marker="o", ls="none", ms=6,
                   label="only M"),
            Line2D([], [], color=SMALL_COLOR, marker="o", ls="none", ms=6,
                   label="only s"),
            Line2D([], [], color=INK, ls="--", lw=1.2, label="L > M > s region"),
            Line2D([], [], color=INK, marker="s", ls="none", mfc="white", mew=1.4,
                   ms=7, label="equal step"),
        ],
        loc="lower center", ncol=5, fontsize=9, frameon=False,
    )

    dropped = len(atlas) - n_panels
    tail = f"  ({dropped} more not shown)" if dropped else ""
    fig.suptitle(
        f"Ternary simplices at {cardinality} notes — "
        f"{len(atlas)} signatures admit an MV{max_variety} word{tail}\n"
        f"blue proper, red improper, black line = propriety boundary",
        fontsize=12.5, color=INK,
    )
    return fig, axes
