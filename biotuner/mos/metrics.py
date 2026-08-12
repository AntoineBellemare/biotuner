"""Structural and harmonic measurements on a scale.

Everything in this module is read off one object: the **interval matrix**.
Row ``i``, column ``k-1`` of that matrix holds the specific size, in cents, of
the generic ``k``-step interval that starts on degree ``i`` and wraps through
the period.  Myhill's property, Rothenberg propriety, Blackwood's ``R`` and the
per-degree interval signatures are all statements about how few distinct values
appear in each of its columns, so computing them from a common source keeps
them mutually consistent by construction.

That matters because :class:`~biotuner.mos.scale.MOSScale` already answers some
of the same questions from the *combinatorics* of the generator -- propriety,
for instance, is available there as ``hardness <= 2``.  The functions here take
the other route: build the scale, measure every interval it actually contains,
and decide from the numbers.  Two independent derivations that must agree is a
much stronger test than either alone, and the test suite asserts the agreement
across many signatures and tunings.  It also found where the agreement breaks:
the ``hardness <= 2`` shortcut misclassifies every MOS with a single small
step, which :func:`is_proper` documents and the tests pin down.

Milne et al. (2011) §2 state the two structural facts a well-formed scale is
expected to satisfy: "every scale span (generic interval size) occurs in
exactly two interval sizes (Myhill's property)", and "every scale degree has a
unique pattern of intervals surrounding it".  :func:`myhill_property` and
:func:`has_unique_degree_signatures` check them empirically rather than
assuming them.

Every function takes a *scale-like* argument, which may be

- an :class:`~biotuner.mos.scale.MOSScale` (or a
  :class:`~biotuner.mos.modes.Mode`, or anything else exposing ``cents``),
- or a raw ``(cents_list, period_cents)`` pair.

The second form exists so that non-MOS scales -- a hand-written subset of an
EDO, a measured tuning, a scale from a Scala file -- can be put through exactly
the same measurements, which is the only way to show that these predicates
discriminate.  A hand-built scale that fails :func:`myhill_property` is what
gives the passing MOS cases their meaning.

Measuring a *signal* instead of a scale
---------------------------------------
Everything above takes a scale as given.  :func:`mos_ness` takes a set of
observed frequency ratios and asks the weaker, answerable question underneath
"which MOS is this?": **how much of this signal's structure needs a generator
at all?**  It fits three families of increasing freedom at one fixed
cardinality -- an equal division, a well-formed scale, a three-step scale --
through the same scoring code and the same transposition search, and reports
how much error each extra free parameter removes.  See :class:`MOSness`.

References
----------
Milne, A.J., Carlé, M., Sethares, W.A., Noll, T., Holland, S. (2011).
Scratching the Scale Labyrinth. In *Mathematics and Computation in Music*,
LNAI 6726, 180--195.

Rothenberg, D. (1978). A model for pattern perception with musical
applications. *Mathematical Systems Theory* 11, 199--234.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from biotuner.mos.theory import PERIOD_CENTS

if TYPE_CHECKING:  # pragma: no cover
    from biotuner.mos.scale import MOSScale

__all__ = [
    "interval_matrix",
    "generic_interval_sizes",
    "myhill_property",
    "is_proper",
    "blackwood_r",
    "degree_signatures",
    "has_unique_degree_signatures",
    "evenness",
    "ji_error",
    "harmonicity",
    "mos_report",
    "MODEL_PARAMETERS",
    "MOSness",
    "mos_ness",
]

#: What every function here accepts.  Duck-typed at runtime, so a
#: :class:`~biotuner.mos.modes.Mode` -- or anything else carrying ``cents`` --
#: works too even though it is not named here.
ScaleLike = Union["MOSScale", Tuple[Sequence[float], float]]


# --------------------------------------------------------------------------- #
# Input normalisation
# --------------------------------------------------------------------------- #
def _as_cents(scale: ScaleLike) -> Tuple[List[float], float]:
    """Reduce any accepted input to ``(sorted_cents, period_cents)``.

    Degrees are sorted rather than required sorted, because a hand-written
    cents list is the common case for the raw form and the interval matrix is
    defined on the ordered set.  The root is *not* forced to 0: the matrix only
    ever uses differences, so a scale whose lowest degree is nonzero is
    perfectly well defined.
    """
    cents_attr = getattr(scale, "cents", None)
    if cents_attr is not None:
        period = getattr(scale, "period_cents", None)
        if period is None:
            # A Mode carries its period on the parent scale.
            parent = getattr(scale, "scale", None)
            period = getattr(parent, "period_cents", None)
        if period is None:
            raise TypeError(
                f"{type(scale).__name__} exposes .cents but no .period_cents; "
                "pass a (cents_list, period_cents) pair instead"
            )
        raw: Sequence[float] = cents_attr
    elif isinstance(scale, (tuple, list)) and len(scale) == 2:
        raw, period = scale[0], scale[1]
        if isinstance(raw, (int, float)) or not hasattr(raw, "__len__"):
            raise TypeError(
                "a raw scale must be given as (cents_list, period_cents); got "
                f"first element {raw!r} of type {type(raw).__name__}"
            )
    else:
        raise TypeError(
            "expected an MOSScale (or any object with .cents and .period_cents) "
            f"or a (cents_list, period_cents) pair, got {type(scale).__name__}"
        )

    period = float(period)
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError(f"period_cents must be finite and > 0, got {period!r}")

    values = [float(c) for c in raw]
    if len(values) < 2:
        raise ValueError(
            f"a scale needs at least 2 degrees to have intervals, got {len(values)}"
        )
    if not all(math.isfinite(c) for c in values):
        raise ValueError(f"scale degrees must all be finite, got {values!r}")
    lo, hi = min(values), max(values)
    if lo < 0.0 or hi >= period:
        raise ValueError(
            f"scale degrees must lie in [0, {period}) cents, got "
            f"min {lo} and max {hi}; reduce them into the period first"
        )
    values.sort()
    for a, b in zip(values, values[1:]):
        if a == b:
            raise ValueError(
                f"scale degrees must be distinct, but {a} cents appears twice"
            )
    return values, period


def _as_ratios(scale: ScaleLike) -> List[float]:
    """Frequency ratios of the degrees, from the object or from its cents."""
    ratios = getattr(scale, "ratios", None)
    if ratios is not None:
        return [float(r) for r in ratios]
    cents, _ = _as_cents(scale)
    return [2.0 ** (c / 1200.0) for c in cents]


def _cluster(values: Sequence[float], tol: float) -> List[float]:
    """Sorted representatives of the distinct values, merging within ``tol``.

    Greedy single-linkage on a sorted list: a value joins the running cluster
    while it stays within ``tol`` of the previous one.  Interval sizes in a
    real scale are separated by a chroma (tens of cents), so the chaining
    behaviour that would bite on dense data never triggers here -- ``tol`` is
    only absorbing floating-point noise from the generator stack.
    """
    out: List[float] = []
    for v in sorted(values):
        if not out or (v - out[-1]) > tol:
            out.append(float(v))
    return out


def _rank(value: float, representatives: Sequence[float], tol: float) -> int:
    """Index of ``value``'s cluster among ascending ``representatives``."""
    idx = 0
    for j, r in enumerate(representatives):
        if value >= r - tol:
            idx = j
    return idx


# --------------------------------------------------------------------------- #
# The interval matrix
# --------------------------------------------------------------------------- #
def interval_matrix(scale: ScaleLike) -> np.ndarray:
    """Specific sizes of every generic interval, in cents.

    Entry ``[i, k-1]`` is the size of the ``k``-step interval rising from
    degree ``i``, wrapping through the period when it runs off the top of the
    scale.  Columns are therefore the *generic* interval classes (seconds,
    thirds, …) and the spread within a column is what all the structural
    predicates in this module look at.

    Parameters
    ----------
    scale : MOSScale, Mode, or (cents_list, period_cents)
        Either a scale object exposing ``cents`` (and ``period_cents``, taken
        from the parent scale for a :class:`~biotuner.mos.modes.Mode`), or a
        raw pair of a cents list and the period in cents.  The raw form is how
        non-MOS scales get measured -- see the module docstring.

    Returns
    -------
    numpy.ndarray
        Shape ``(N, N - 1)`` for an ``N``-note scale.  There is no column for
        ``k = N`` because that interval is the period for every degree.

    Examples
    --------
    The diatonic scale in 12-EDO.  Row 0 is the Lydian scale measured from its
    root; column 0 is the step pattern ``LLLsLLs``:

    >>> from biotuner.mos.scale import MOSScale
    >>> m = interval_matrix(MOSScale.from_signature(5, 2, tuning=12))
    >>> m.shape
    (7, 6)
    >>> [round(float(x)) for x in m[0]]
    [200, 400, 600, 700, 900, 1100]
    >>> [round(float(x)) for x in m[:, 0]]
    [200, 200, 200, 100, 200, 200, 100]

    A raw scale needs no MOS structure at all:

    >>> interval_matrix(([0.0, 100.0, 700.0], 1200.0))
    array([[ 100.,  700.],
           [ 600., 1100.],
           [ 500.,  600.]])
    """
    cents, period = _as_cents(scale)
    n = len(cents)
    out = np.empty((n, n - 1), dtype=float)
    for i in range(n):
        base = cents[i]
        for k in range(1, n):
            j = i + k
            top = cents[j] if j < n else cents[j - n] + period
            out[i, k - 1] = top - base
    return out


def generic_interval_sizes(
    scale: ScaleLike, tol: float = 1e-6
) -> Dict[int, List[float]]:
    """Distinct specific sizes found in each generic interval class.

    Parameters
    ----------
    scale : MOSScale, Mode, or (cents_list, period_cents)
    tol : float, default 1e-6
        Cents below which two sizes count as the same.  Needed because degrees
        come from repeatedly folding a float generator into the period, so
        nominally identical intervals differ in the last few bits.

    Returns
    -------
    dict
        ``{k: sorted distinct sizes in cents}`` for ``k`` in ``1 .. N-1``.

    Examples
    --------
    Two sizes per class -- the diatonic's major/minor seconds, thirds, and so
    on up to its major/minor sevenths:

    >>> from biotuner.mos.scale import MOSScale
    >>> sizes = generic_interval_sizes(MOSScale.from_signature(5, 2, tuning=12))
    >>> {k: [round(v) for v in vs] for k, vs in sizes.items()}
    {1: [100, 200], 2: [300, 400], 3: [500, 600], 4: [600, 700], 5: [800, 900], 6: [1000, 1100]}
    """
    matrix = interval_matrix(scale)
    return {
        k: _cluster(matrix[:, k - 1], tol) for k in range(1, matrix.shape[1] + 1)
    }


# --------------------------------------------------------------------------- #
# Structural predicates
# --------------------------------------------------------------------------- #
def myhill_property(scale: ScaleLike, tol: float = 1e-6) -> bool:
    """True when every generic interval class has exactly two specific sizes.

    Milne et al. §2 give this as the defining feature of the scales the
    labyrinth generates: "every scale span (generic interval size) occurs in
    exactly two interval sizes (Myhill's property)".  A non-degenerate MOS
    satisfies it for every class ``1 .. N-1``, which makes this the single
    strongest structural check available on the output of
    :class:`~biotuner.mos.scale.MOSScale`.

    A **degenerate** tuning -- one sitting exactly on a landmark equal
    temperament, where the large and small steps have collapsed onto each
    other -- has *one* size per class, so it returns ``False`` here even though
    it is the limit of a family of MOS.  Guard with
    :attr:`~biotuner.mos.scale.MOSScale.is_degenerate` if that distinction
    matters; ``myhill_property`` reports the structure of the scale as tuned,
    not the family it came from.

    Examples
    --------
    >>> from biotuner.mos.scale import MOSScale
    >>> myhill_property(MOSScale.from_signature(5, 2, tuning=12))
    True

    7-EDO is where the diatonic's two step sizes meet, so its intervals
    collapse to one size per class:

    >>> myhill_property(MOSScale.from_signature(5, 2, tuning='equalized'))
    False

    The harmonic minor is not well formed -- it already fails at the steps:

    >>> harmonic_minor = ([0, 200, 300, 500, 700, 800, 1100], 1200.0)
    >>> myhill_property(harmonic_minor)
    False
    """
    sizes = generic_interval_sizes(scale, tol=tol)
    return all(len(v) == 2 for v in sizes.values())


def is_proper(scale: ScaleLike, strict: bool = False, tol: float = 1e-6) -> bool:
    """Rothenberg propriety, decided from the interval matrix.

    A scale is proper when the specific sizes never cross the generic ordering:
    no ``k``-step interval is larger than any ``(k+1)``-step interval.  Strict
    propriety additionally forbids ties, so a scale with an ambiguous interval
    -- the diatonic tritone, which is both an augmented fourth and a diminished
    fifth -- is proper but not strictly so.

    This deliberately duplicates :attr:`~biotuner.mos.scale.MOSScale.is_proper`,
    which reaches its verdict from Blackwood's ``R <= 2`` (Milne et al. §2).
    Measuring the intervals is the independent check on that shortcut, so do not
    reimplement this in terms of hardness.  Sweeping every co-prime signature up
    to 13 notes at several tunings, the two agree everywhere **except when the
    scale has a single small step**, where the shortcut is simply wrong:

        max(class k) <= min(class k+1) reduces to ``(1 - d)L <= (2 - d)s`` with
        ``d = m_{k+1} - m_k`` in ``{0, 1}``, ``m_k`` being the number of large
        steps in the small variant of class ``k``.  A ``d = 1`` transition costs
        nothing; only ``d = 0`` transitions demand ``L <= 2s``.  Classes
        ``1 .. N-2`` contain exactly ``n_small - 1`` of them.

    So ``n_small == 1`` leaves *no* propriety constraint and such a scale is
    (strictly) proper at any hardness: 2L1s tuned to L = 500 c, s = 200 c has
    ``R = 2.5`` yet its largest second, 500 c, is well below its smallest third,
    700 c.  ``MOSScale.is_proper`` calls that improper.  Away from
    ``n_small == 1`` the two agree, up to ``tol`` at the boundary tuning
    ``R = 2`` where the conventions can round opposite ways.

    Parameters
    ----------
    scale : MOSScale, Mode, or (cents_list, period_cents)
    strict : bool, default False
        Require ``max(class k) < min(class k+1)`` rather than ``<=``.
    tol : float, default 1e-6
        Cents of slack, absorbing float noise in the degrees.

    Examples
    --------
    Pythagorean tuning stretches the diatonic past the propriety boundary --
    its major third (408 c) exceeds its diminished fourth (384 c):

    >>> from biotuner.mos.scale import MOSScale
    >>> is_proper(MOSScale.from_generator(3 / 2, 7))
    False

    12-EDO pulls it back in, but only just: the tritone is a tie, so it is
    proper without being strictly proper.

    >>> is_proper(MOSScale.from_signature(5, 2, tuning=12))
    True
    >>> is_proper(MOSScale.from_signature(5, 2, tuning=12), strict=True)
    False

    31-EDO meantone breaks the tie and is strictly proper:

    >>> is_proper(MOSScale.from_signature(5, 2, tuning=31), strict=True)
    True

    And the counterexample to the hardness shortcut:

    >>> hard = MOSScale.from_signature(2, 1, tuning='middle', bright=False)
    >>> [round(c) for c in hard.cents], round(hard.hardness, 3)
    ([0, 500, 1000], 2.5)
    >>> is_proper(hard), hard.is_proper
    (True, False)
    """
    matrix = interval_matrix(scale)
    n_classes = matrix.shape[1]
    for k in range(n_classes - 1):
        hi = float(matrix[:, k].max())
        lo = float(matrix[:, k + 1].min())
        if strict:
            if not hi < lo - tol:
                return False
        elif not hi <= lo + tol:
            return False
    return True


def blackwood_r(scale: ScaleLike) -> float:
    """Blackwood's ``R``: the ratio of the largest step to the smallest.

    Read off column 0 of the interval matrix, so it applies to any scale, not
    only to two-step-size ones -- for a scale with three or more step sizes it
    reports the extremes and says nothing about what lies between.  For an MOS
    it reproduces :attr:`~biotuner.mos.scale.MOSScale.hardness`.

    ``1`` means equal steps; ``2`` is the propriety boundary for a well-formed
    scale; ``inf`` when a step has vanished.

    Examples
    --------
    >>> from biotuner.mos.scale import MOSScale
    >>> round(blackwood_r(MOSScale.from_generator(3 / 2, 7)), 3)
    2.26
    >>> round(blackwood_r(MOSScale.from_signature(5, 2, tuning=12)), 6)
    2.0
    """
    steps = interval_matrix(scale)[:, 0]
    small = float(steps.min())
    if small <= 0.0:
        return float("inf")
    return float(steps.max()) / small


def degree_signatures(scale: ScaleLike, tol: float = 1e-6) -> List[Tuple[int, ...]]:
    """Which variant of each generic interval sits above each degree.

    Entry ``i`` is a tuple of length ``N-1``: position ``k-1`` holds the rank of
    degree ``i``'s ``k``-step interval among the distinct sizes of that class --
    ``0`` for the small variant, ``1`` for the large.  (Ranks above ``1`` only
    appear for scales that fail :func:`myhill_property`, where a class has more
    than two sizes.)

    This is the concrete form of Milne et al. §2's second structural claim,
    that in these scales "every scale degree has a unique pattern of intervals
    surrounding it" -- see :func:`has_unique_degree_signatures`.

    Examples
    --------
    The diatonic in 12-EDO, rooted on its brightest mode (C Lydian, degrees
    C D E F♯ G A B).  Degree 0 is the bottom of the chain of fifths, so every
    interval above it is the large variant; degree 3 -- the F♯ that closes the
    chain -- gets the small variant of all six:

    >>> from biotuner.mos.scale import MOSScale
    >>> for sig in degree_signatures(MOSScale.from_signature(5, 2, tuning=12)):
    ...     print(sig)
    (1, 1, 1, 1, 1, 1)
    (1, 1, 0, 1, 1, 0)
    (1, 0, 0, 1, 0, 0)
    (0, 0, 0, 0, 0, 0)
    (1, 1, 0, 1, 1, 1)
    (1, 0, 0, 1, 1, 0)
    (0, 0, 0, 1, 0, 0)
    """
    matrix = interval_matrix(scale)
    n, n_classes = matrix.shape
    reps = [_cluster(matrix[:, k], tol) for k in range(n_classes)]
    return [
        tuple(_rank(float(matrix[i, k]), reps[k], tol) for k in range(n_classes))
        for i in range(n)
    ]


def has_unique_degree_signatures(scale: ScaleLike, tol: float = 1e-6) -> bool:
    """True when no two degrees share the same interval pattern.

    Milne et al. §2: in a well-formed scale "every scale degree has a unique
    pattern of intervals surrounding it", which is what lets a listener locate
    themselves in the scale from its intervals alone.  Degenerate tunings fail
    this -- every degree of an equal division looks identical.

    Examples
    --------
    >>> from biotuner.mos.scale import MOSScale
    >>> has_unique_degree_signatures(MOSScale.from_signature(5, 2, tuning=12))
    True
    >>> has_unique_degree_signatures(MOSScale.from_signature(5, 2, tuning='equalized'))
    False
    """
    sigs = degree_signatures(scale, tol=tol)
    return len(set(sigs)) == len(sigs)


def evenness(scale: ScaleLike) -> float:
    """Largest departure of any degree from the equal division, in period fractions.

    Degree ``i`` of an ``N``-note equal division sits at ``i / N`` of the
    period; this returns ``max |degree_i - i/N|``, measured from the scale's
    lowest degree.  ``0`` is the equal division itself, and the value grows as
    the two step sizes separate -- a scalar companion to
    :func:`blackwood_r` that weighs *where* the unevenness accumulates rather
    than only how extreme the steps get.

    Examples
    --------
    The 12-EDO diatonic's worst-placed degree is its tritone, a full
    half-step (``1/14`` of the octave) above 7-EDO's:

    >>> from biotuner.mos.scale import MOSScale
    >>> round(evenness(MOSScale.from_signature(5, 2, tuning=12)), 6)
    0.071429
    >>> round(evenness(MOSScale.from_signature(5, 2, tuning='equalized')), 12)
    0.0
    """
    cents, period = _as_cents(scale)
    n = len(cents)
    base = cents[0]
    return max(abs((c - base) / period - i / n) for i, c in enumerate(cents))


# --------------------------------------------------------------------------- #
# Harmonic measurements
# --------------------------------------------------------------------------- #
def ji_error(
    scale: ScaleLike,
    targets: Sequence[float],
    weights: Optional[Sequence[float]] = None,
    period_reduce: bool = True,
) -> Dict[str, object]:
    """How well the scale approximates a set of just intervals.

    Each target is a *frequency ratio*; its distance to the nearest scale degree
    is reported in cents, signed positive when the target is sharp of the degree
    it lands on.  This is the quantity the labyrinth trades against structure:
    moving the generator inside a signature's valid range changes nothing about
    the scale's identity but everything about how close it gets to the ratios
    you care about.

    Parameters
    ----------
    scale : MOSScale, Mode, or (cents_list, period_cents)
    targets : sequence of float
        Frequency ratios, e.g. ``[3/2, 5/4, 6/5]``.  Must be finite and
        positive.  ``nan`` and ``inf`` are rejected rather than tolerated: they
        propagate through every summary statistic and would silently turn a
        whole :func:`mos_report` row into ``nan``.
    weights : sequence of float, optional
        Relative importance of each target, normalised internally to sum to 1.
        Uniform by default.  Must be finite, non-negative and not all zero.
    period_reduce : bool, default True
        Fold each target into the period before matching, and let the match
        wrap around it -- so a target just under the period matches the root.
        With ``False`` the target's raw cents (which for a target above the
        period exceed every degree) are compared to the degrees as they stand,
        so the error reports the absolute distance rather than the pitch-class
        distance.

    Returns
    -------
    dict
        ``errors`` (signed, per target, in cents), plus ``mean_abs``,
        ``max_abs``, ``rms`` and ``weighted_mean``, all computed on the
        absolute errors.

    Examples
    --------
    12-EDO's fifth is famously 2 cents flat of just, its major third 14 cents
    sharp, and its minor third badly off:

    >>> from biotuner.mos.scale import MOSScale
    >>> e = ji_error(MOSScale.from_signature(5, 2, tuning=12), [3 / 2, 5 / 4, 6 / 5])
    >>> [round(x, 2) for x in e['errors']]
    [1.96, -13.69, -84.36]
    >>> round(e['mean_abs'], 3), round(e['max_abs'], 3)
    (33.333, 84.359)

    Weighting the fifth heavily reflects how little its error matters:

    >>> w = ji_error(MOSScale.from_signature(5, 2, tuning=12),
    ...              [3 / 2, 5 / 4, 6 / 5], weights=[10, 1, 1])
    >>> round(w['weighted_mean'], 3)
    9.8
    """
    cents, period = _as_cents(scale)
    targets = [float(t) for t in targets]
    if not targets:
        raise ValueError("targets must contain at least one frequency ratio")
    if any(not math.isfinite(t) or t <= 0.0 for t in targets):
        # nan fails every ``<= 0`` test, so checking positivity alone lets it
        # through and poisons errors/mean_abs/rms with nan.
        raise ValueError(
            f"targets must be finite, positive frequency ratios, got {targets!r}"
        )

    if weights is None:
        w = [1.0 / len(targets)] * len(targets)
    else:
        w = [float(x) for x in weights]
        if len(w) != len(targets):
            raise ValueError(
                f"weights must match targets in length: {len(w)} weights for "
                f"{len(targets)} targets"
            )
        if any(not math.isfinite(x) or x < 0.0 for x in w):
            raise ValueError(
                f"weights must be finite and non-negative, got {weights!r}"
            )
        total = sum(w)
        if total <= 0.0:
            raise ValueError(f"weights must not sum to zero, got {weights!r}")
        w = [x / total for x in w]

    errors: List[float] = []
    for t in targets:
        target_cents = 1200.0 * math.log2(t)
        if period_reduce:
            target_cents %= period
            # Signed, wrapped difference to whichever degree is nearest.
            best = min(
                (((target_cents - c + period / 2.0) % period) - period / 2.0
                 for c in cents),
                key=abs,
            )
        else:
            best = min((target_cents - c for c in cents), key=abs)
        errors.append(float(best))

    abs_err = [abs(e) for e in errors]
    return {
        "errors": errors,
        "mean_abs": float(sum(abs_err) / len(abs_err)),
        "max_abs": float(max(abs_err)),
        "rms": float(math.sqrt(sum(e * e for e in abs_err) / len(abs_err))),
        "weighted_mean": float(sum(x * e for x, e in zip(w, abs_err))),
    }


def harmonicity(scale: ScaleLike, maxdenom: int = 1000) -> Dict[str, float]:
    """Biotuner's tuning-wide consonance metrics for this scale.

    Delegates to :func:`biotuner.metrics.tuning_to_metrics`, so an MOS is scored
    on exactly the same footing as any other tuning the toolbox handles -- that
    is the point of routing through it rather than reimplementing the measures.
    The import is deferred because :mod:`biotuner.metrics` pulls in the whole
    package, which is slow enough to notice on a module import.

    The underlying metrics rationalise every ratio with
    ``limit_denominator(maxdenom)``, so they are sensitive to that bound: a
    tuning whose degrees are irrational (any non-EDO generator) gets whatever
    fraction the bound admits, and the p/q-based scores move with it.

    Returns
    -------
    dict
        Whatever :func:`~biotuner.metrics.tuning_to_metrics` produced, with
        numpy scalars converted to floats.  **An empty dict** if that call
        raised -- the metrics assume octave-ish, well-conditioned ratio lists
        and can fail on degenerate or pseudo-octave scales, and a report that
        loses one column is more useful than one that cannot be produced.

    Examples
    --------
    >>> from biotuner.mos.scale import MOSScale
    >>> h = harmonicity(MOSScale.from_signature(5, 2, tuning=12))
    >>> round(h['harm_sim'], 2)
    14.46
    """
    from biotuner.metrics import tuning_to_metrics  # deferred: heavy import

    ratios = _as_ratios(scale)
    try:
        raw = tuning_to_metrics(ratios, maxdenom=maxdenom)
    except Exception:
        return {}
    if not isinstance(raw, dict):  # pragma: no cover - defensive
        return {}
    return {str(k): float(v) for k, v in raw.items()}


# --------------------------------------------------------------------------- #
# One-stop report
# --------------------------------------------------------------------------- #
def mos_report(
    scale: ScaleLike,
    targets: Optional[Sequence[float]] = None,
    weights: Optional[Sequence[float]] = None,
    maxdenom: int = 1000,
    harmonic: bool = True,
) -> Dict[str, object]:
    """Everything this module measures, in one flat dict.

    Merges :meth:`~biotuner.mos.scale.MOSScale.to_dict` with the structural
    predicates and, optionally, the harmonic metrics.  Values are plain
    ints/floats/bools/strings and lists thereof, so the result drops straight
    into a :class:`pandas.DataFrame` row or a JSON file -- which is how the
    plotting and derivation layers consume it.

    Parameters
    ----------
    scale : MOSScale
        Anything with a ``to_dict`` method; a raw ``(cents, period)`` pair has
        no identity to report and is rejected.
    targets : sequence of float, optional
        Just ratios to score against.  When given, :func:`ji_error`'s summary
        statistics are added under ``ji_*`` keys.
    weights : sequence of float, optional
        Passed through to :func:`ji_error`.
    maxdenom : int, default 1000
        Passed through to :func:`harmonicity`.
    harmonic : bool, default True
        Include :func:`harmonicity`'s keys.  Turn off for bulk scans, where the
        rationalisation of every ratio dominates the runtime.

    Examples
    --------
    >>> from biotuner.mos.scale import MOSScale
    >>> r = mos_report(MOSScale.from_signature(5, 2, tuning=12),
    ...                targets=[3 / 2, 5 / 4], harmonic=False)
    >>> r['signature'], r['myhill'], r['proper_from_matrix'], r['strictly_proper']
    ('5L2s', True, True, False)
    >>> round(r['blackwood_r'], 6), r['is_proper'] == r['proper_from_matrix']
    (2.0, True)
    >>> [round(x, 2) for x in r['ji_errors']]
    [1.96, -13.69]
    """
    to_dict = getattr(scale, "to_dict", None)
    if to_dict is None:
        raise TypeError(
            "mos_report needs a scale object with .to_dict() (an MOSScale or a "
            f"Mode), got {type(scale).__name__}; use the individual metric "
            "functions for a raw (cents_list, period_cents) pair"
        )

    sizes = generic_interval_sizes(scale)
    out: Dict[str, object] = dict(to_dict())
    out.update(
        {
            "myhill": bool(myhill_property(scale)),
            "proper_from_matrix": bool(is_proper(scale)),
            "strictly_proper": bool(is_proper(scale, strict=True)),
            "blackwood_r": float(blackwood_r(scale)),
            "unique_degree_signatures": bool(has_unique_degree_signatures(scale)),
            "evenness": float(evenness(scale)),
            "n_interval_classes": len(sizes),
            "interval_class_sizes": [sizes[k] for k in sorted(sizes)],
            "n_sizes_per_class": [len(sizes[k]) for k in sorted(sizes)],
            "degree_signatures": [list(s) for s in degree_signatures(scale)],
        }
    )
    if targets is not None:
        err = ji_error(scale, targets, weights=weights)
        out.update(
            {
                "ji_errors": err["errors"],
                "ji_mean_abs": err["mean_abs"],
                "ji_max_abs": err["max_abs"],
                "ji_rms": err["rms"],
                "ji_weighted_mean": err["weighted_mean"],
            }
        )
    if harmonic:
        out.update(harmonicity(scale, maxdenom=maxdenom))
    return out


# --------------------------------------------------------------------------- #
# How much of a signal needs a generator?
# --------------------------------------------------------------------------- #
#: Free parameters each scale family fits to the data, transposition included.
#:
#: At a *fixed* cardinality ``N`` the three families differ only in how much
#: freedom they have to place their degrees:
#:
#: ``edo``
#:     ``N`` equal steps.  Nothing about the shape is free; only where the
#:     whole thing sits against the data.
#: ``mos``
#:     One generator, stacked ``N`` times.  Two step sizes, distributed with
#:     maximal evenness -- the shape is a one-parameter family.
#: ``ternary``
#:     Three step sizes filling the period, i.e. two free shape parameters
#:     (:mod:`biotuner.mos.ternary`'s simplex is two-dimensional).
#:
#: Transposition is counted for all three because it really is fitted --
#: :func:`biotuner.mos.derive._evaluate` searches rotations -- even though a
#: scale and its transpositions are the same scale.  It is common to all three,
#: so it cancels out of any *difference* between them; it is here so that the
#: degrees-of-freedom correction below counts every quantity read off the data.
MODEL_PARAMETERS: Dict[str, int] = {"edo": 1, "mos": 2, "ternary": 3}

#: Below this, an error is zero: an exactly recovered scale lands at 1e-13
#: rather than at 0, and dividing one such number by another reports arithmetic
#: noise as if it were a finding.  Same threshold, same reason, as
#: :attr:`biotuner.mos.derive.MOSFit.improvement`.
_ZERO_ERROR_CENTS = 1e-9

#: How far inside the ternary simplex the search is allowed to go.  On an edge
#: a step size is zero, two degrees coincide, and the scale is really binary --
#: which is the MOS case, already measured separately.  Matches
#: :data:`biotuner.mos.ternary._EDGE_EPS`.
_SIMPLEX_MARGIN = 1e-4


def _share(error: float, baseline: float) -> float:
    """Fraction of ``baseline`` that ``error`` improves on, clamped to ``[0, 1]``.

    ``0`` when there is nothing to improve on -- a baseline already at zero
    leaves the richer model no error to remove, so its extra parameter is worth
    nothing however the arithmetic would otherwise come out.  That is the case
    an exactly equal-tempered input lands in, and zero is the right answer for
    it: the generator did not help.

    Examples
    --------
    >>> round(_share(5.0, 20.0), 3), _share(20.0, 20.0), _share(30.0, 20.0)
    (0.75, 0.0, 0.0)
    >>> _share(0.0, 0.0), _share(0.0, float('inf'))
    (0.0, 0.0)
    """
    if not math.isfinite(baseline) or baseline <= _ZERO_ERROR_CENTS:
        return 0.0
    if not math.isfinite(error):
        return 0.0
    return max(0.0, min(1.0, 1.0 - error / baseline))


def _wrap(text: str, width: int) -> List[str]:
    """Greedy word wrap, so :meth:`MOSness.summary` stays inside a terminal."""
    out: List[str] = [""]
    for word in text.split():
        if not out[-1]:
            out[-1] = word
        elif len(out[-1]) + 1 + len(word) <= width:
            out[-1] = f"{out[-1]} {word}"
        else:
            out.append(word)
    return out


def _derive():
    """The fitting layer, imported on demand.

    :mod:`biotuner.mos.derive` sits *above* this module in the package's
    layering and imports nothing from it.  A module-level import here would be
    a backwards edge waiting to become a genuine cycle the first time the
    fitting layer wants a metric, so it is taken per call instead; the cost is
    one dict lookup.  :mod:`biotuner.mos.ternary` is worse -- it already
    imports this module -- so that one *has* to be deferred.
    """
    from biotuner.mos import derive

    return derive


def _adjusted_error(error_cents: float, n_targets: int, model: str) -> float:
    """``error_cents`` inflated for the parameters that were fitted to reach it.

    A model with more free parameters always fits better, so comparing raw
    errors between the three families would hand the richest one a win it did
    not earn.  The correction is the familiar residual-degrees-of-freedom
    shrinkage: ``p`` of the ``n`` residuals were spent pinning the parameters
    down, so the surviving mean is optimistically small by roughly
    ``(n - p) / n`` and is divided back up::

        adjusted = error * n_targets / (n_targets - MODEL_PARAMETERS[model])

    This is exact for a mean *squared* residual under a linear model and only a
    first-order correction here, where the objective is a weighted mean
    *absolute* distance to the nearest of several degrees.  It is a stated rule
    rather than a derivation -- but it is applied identically to all three
    families, its direction is right (the penalty grows with the parameter
    count), and both the raw errors and the counts stay visible on
    :class:`MOSness` so a reader can redo the comparison another way.

    ``inf`` when the model has at least as many parameters as there are
    targets, where the fit is free and the error means nothing.
    """
    p = MODEL_PARAMETERS[model]
    if n_targets <= p:
        return float("inf")
    return float(error_cents) * n_targets / (n_targets - p)


def _equal_division(cardinality: int, period: float) -> "MOSScale":
    """``cardinality``-EDO, wearing the type the fitting layer scores.

    The null model has to run through
    :func:`biotuner.mos.derive._evaluate` alongside the others or the
    comparison is between implementations rather than between hypotheses, and
    that function reads ``degrees``, ``cardinality`` and ``period_cents`` off
    an :class:`~biotuner.mos.scale.MOSScale`.  An equal division *is* one: it
    is the degenerate tuning where the two step sizes coincide, and stacking
    ``1/N`` of the period ``N`` times lands on exactly ``0, 1/N, ..., (N-1)/N``.

    The ``(N-1)L1s`` label is a formality.  Every co-prime ``k/N`` produces the
    same degree set, so the signature says which MOS family this equal division
    is the degenerate limit *of*, not anything about the scale itself.

    Examples
    --------
    >>> e = _equal_division(5, 2.0)
    >>> e.signature, [round(c, 3) for c in e.cents]
    ('4L1s', [0.0, 240.0, 480.0, 720.0, 960.0])
    >>> e.is_degenerate, round(e.hardness, 12)
    (True, 1.0)
    """
    from biotuner.mos.scale import MOSScale

    return MOSScale(cardinality - 1, 1, 1.0 / cardinality, period, validate=False)


@lru_cache(maxsize=64)
def _ternary_words_at(cardinality: int, max_variety: int = 3) -> Tuple[str, ...]:
    """Every admissible ternary step pattern with ``cardinality`` notes.

    "Admissible" means max variety at most ``max_variety``, which at 3 is the
    ternary analogue of Myhill's property (:mod:`biotuner.mos.ternary`), and one
    representative per rotation class -- rotating a word transposes the scale,
    and the fit searches transpositions anyway.

    The filter is what makes the ternary family a fair third rung rather than a
    free pass.  A well-formed scale at ``N`` notes has exactly *one* step
    pattern, the Christoffel word; letting a three-step scale pick from all
    ``N! / (a! b! c!)`` arrangements would give it an enormous amount of
    discrete freedom that no parameter count reflects.  Requiring the ternary
    Myhill condition cuts that back to a handful -- 24 words at seven notes out
    of 258 rotation classes, 26 at twelve out of tens of thousands -- so what is
    being compared is one structured family against another.

    Cached because the enumeration is the expensive part of
    :func:`mos_ness` and depends on nothing but its two arguments.

    Examples
    --------
    >>> len(_ternary_words_at(5)), len(_ternary_words_at(7))
    (18, 24)
    >>> _ternary_words_at(4)
    ('LsMs', 'LMsM', 'LMLs')
    """
    from biotuner.mos.ternary import ternary_words

    out: List[str] = []
    for a in range(1, cardinality - 1):
        for b in range(1, cardinality - a):
            out.extend(
                ternary_words(a, b, cardinality - a - b, max_variety=max_variety)
            )
    return tuple(out)


def _fit_ternary_word(
    word: str,
    positions: np.ndarray,
    weights: np.ndarray,
    period: float,
    tolerance_cents: float,
    coarse: int = 10,
):
    """Best tuning of one ternary word against the targets, as a ``MOSFit``.

    The tuning space of a ternary word is the open 2-simplex of period shares
    ``(u, v, w)``, so this is a two-dimensional minimisation where
    :func:`biotuner.mos.derive._refine_generator` does a one-dimensional one.
    A coarse barycentric grid locates the basin -- the objective is piecewise
    linear in the shares with a kink wherever a target changes which degree it
    matches, so it has many local minima and a bare local search from a fixed
    start would find the wrong one -- and Nelder-Mead polishes it.

    Points outside the simplex are penalised rather than rejected, so the
    simplex search still gets a gradient pointing back inside instead of a wall
    of ``inf`` that Nelder-Mead cannot climb down from.
    """
    from biotuner.mos.ternary import TernaryScale

    D = _derive()

    def build(u: float, v: float) -> Optional["TernaryScale"]:
        w = 1.0 - u - v
        if min(u, v, w) < _SIMPLEX_MARGIN:
            return None
        return TernaryScale.from_barycentric(word, u, v, w, period)

    def objective(uv: Sequence[float]) -> float:
        u, v = float(uv[0]), float(uv[1])
        scale = build(u, v)
        if scale is None:
            # A cone rising out of the simplex: finite, so Nelder-Mead can walk
            # back down it, and steep enough that it never wins.
            outside = _SIMPLEX_MARGIN - min(u, v, 1.0 - u - v)
            return 1e6 * (1.0 + outside)
        return D._evaluate(
            scale, positions, weights, tolerance_cents, 0.0,
            align=True, n_anchors=None,
        ).error_cents

    # Every grid point is at least ``1 / coarse`` of the period from each edge,
    # so the incumbent is always a scale that can actually be built.
    best_uv = (1.0 / 3.0, 1.0 / 3.0)
    best_value = objective(best_uv)
    for i in range(1, coarse):
        for j in range(1, coarse - i):
            uv = (i / coarse, j / coarse)
            value = objective(uv)
            if value < best_value:
                best_value, best_uv = value, uv

    try:
        from scipy.optimize import minimize

        res = minimize(
            objective, x0=np.asarray(best_uv, dtype=float),
            method="Nelder-Mead",
            options={"xatol": 1e-7, "fatol": 1e-10, "maxiter": 600},
        )
        polished = (float(res.x[0]), float(res.x[1]))
        # The feasibility check is belt and braces -- the penalty cone starts at
        # 1e6 and no real error comes near it, so an outside point can never
        # beat an inside one -- but accepting an infeasible optimum here would
        # silently discard the whole search for this word.
        if float(res.fun) < best_value and build(*polished) is not None:
            best_uv = polished
    except Exception:  # pragma: no cover - scipy is a hard dep, but stay safe
        pass

    return D._evaluate(
        build(*best_uv), positions, weights, tolerance_cents, 0.0,
        align=True, n_anchors=None,
    )


def _best_ternary(
    positions: np.ndarray,
    weights: np.ndarray,
    period: float,
    cardinality: int,
    tolerance_cents: float,
):
    """Lowest-error admissible ternary scale at one cardinality, or ``None``."""
    words = _ternary_words_at(cardinality)
    best = None
    for word in words:
        fit = _fit_ternary_word(word, positions, weights, period, tolerance_cents)
        if best is None or fit.error_cents < best.error_cents:
            best = fit
    return best


@dataclass(frozen=True)
class MOSness:
    """How much of a signal's structure a *generator* accounts for.

    :attr:`~biotuner.mos.derive.MOSFit.evidence` answers a weaker question than
    it looks like it answers.  It compares a fitted scale against ratios
    scattered *uniformly*, so beating it establishes only that the signal is not
    noise -- any ``N``-note scale with somewhere to slide would beat it too.
    The alternative that matters is not noise, it is **an equally spaced scale
    of the same size**, which has all the same coverage and none of the
    structure.  This class is that comparison, run at one fixed cardinality
    across three families of increasing freedom:

    ============  ==========================  =====================
    family        shape                       free parameters
    ============  ==========================  =====================
    ``edo``       ``N`` equal steps           1 (transposition)
    ``mos``       one generator, two steps    2 (+ generator)
    ``ternary``   three step sizes            3 (+ a second shape)
    ============  ==========================  =====================

    All three are scored by :func:`biotuner.mos.derive._evaluate` with the same
    weights, the same targets and the same exhaustive transposition search, so
    the numbers differ because the hypotheses differ and for no other reason.
    Fixing the cardinality is what makes them comparable at all: a five-note
    equal division against a nine-note MOS would be measuring the note count.

    Attributes
    ----------
    cardinality : int
        Notes per period, shared by all three families.
    cardinality_rule : str
        How :attr:`cardinality` was chosen -- ``'edo-evidence'`` (the note count
        at which the equal division carried the most evidence, which does not
        select on the hypothesis under test), ``'mos-evidence'`` (likewise for
        the MOS fit, which does), ``'explicit'`` or ``'max'``.
    n_targets, n_merged : int
        Distinct pitch classes fitted, and how many input ratios were absorbed
        into one already present.
    period : float
    edo_error_cents, mos_error_cents : float
        Weighted mean absolute cents error of the best fit in each family.
    ternary_error_cents : float or None
        ``None`` when the ternary rung was skipped -- see :attr:`notes`.
    mos_ness : float
        **The headline.**  The share of the equal division's error that the
        generator removes, after both errors are corrected for the parameters
        that produced them (:func:`_adjusted_error`)::

            mos_ness = max(0, 1 - adjusted_mos_error / adjusted_edo_error)

        ``0`` means the generator bought nothing: the signal's ratios are
        spread as evenly as an equal division and a well-formed scale describes
        them no better.  ``1`` means the MOS fits exactly.  In between is the
        fraction of the null's error that well-formedness explains away.

        It is a ratio between two fits to the same data at the same
        cardinality, which is why it survives what the fitted generator does
        not: change the peak extraction and the generator moves hundreds of
        cents, but both fits move with it.

        Zero is the *definition* of no improvement, not the empirical null.
        Uniformly random ratios score around 0.09 on average and can reach 0.33
        -- see :func:`mos_ness`'s notes for the measured band and why the
        parameter correction cannot shrink it further.

        Two conventions, both deliberate.  It is **clamped at 0**: the equal
        division is the degenerate limit of every MOS at this cardinality --
        and its generator is planted in the candidate list precisely so the
        search cannot miss it -- so a negative value can only come from the
        parameter correction, which is what it is for.  And it is **0 when the
        equal division already fits exactly** (both errors below a nanocent),
        because there is then nothing left for a generator to remove; an
        equal-tempered input scores 0, which is the right answer for it.
    ternary_ness : float or None
        The same quantity for the three-step family.
    two_step_sufficiency : float or None
        ``mos_ness / ternary_ness``, clamped to ``[0, 1]``: of everything a
        third step size manages to explain beyond the equal division, how much
        two step sizes already had.  ``1`` says the third step is redundant and
        the signal is well formed rather than merely non-uniform; a low value
        says two step sizes were the wrong description.

        Clamped above because the admissible ternary words
        (:func:`_ternary_words_at`) are not *guaranteed* to contain the fitted
        MOS: their step patterns have three letters, and while the interior
        lines where two of the three sizes coincide do reduce a ternary word to
        a two-step scale, the particular MOS the signal wants need not be one of
        them.  Where it is not, a genuinely well-formed signal comes out fitting
        the ternary family worse, which reads as full sufficiency.
    ternary_step_cents : tuple of float or None
        ``(large, medium, small)`` of the winning ternary scale.  Worth
        reading: full two-step sufficiency usually shows up here as two of the
        three sizes collapsing onto each other, which is the third step size
        visibly declining to exist.  See :attr:`ternary_collapsed`.
    signature, generator, generator_cents, hardness : str, float, float, float
        The winning MOS.  Report these as *what was compared*, not as an
        identification: the fitted generator is famously unstable across peak
        extraction settings even when :attr:`mos_ness` is not.
    ternary_word, ternary_signature : str or None
    is_identifiable : bool
        ``False`` when :attr:`cardinality` is not strictly below
        :attr:`n_targets`.  Same idea as
        :attr:`biotuner.mos.derive.MOSFit.is_underdetermined`, one notch
        stricter: that flag fires when the scale has *more* degrees than there
        were targets, and this one already fires when it has as many, because a
        scale with a degree per target is fitting one number per observation.
        The result is still computed -- an ``N``-note equal division has exactly
        ``N`` pitch classes, so refusing would make the most informative test
        case unmeasurable -- but it is not evidence.
    notes : tuple of str
        Plain-language caveats raised during the computation: the
        identifiability warning, why the ternary rung was skipped, and so on.
        Empty is the clean case.
    by_cardinality : tuple of dict
        One row per candidate cardinality, each with ``cardinality``,
        ``edo_error_cents``, ``mos_error_cents``, ``mos_ness``, ``signature``,
        ``generator_cents``, ``mos_evidence``, ``edo_evidence`` and
        ``identifiable``.  The selected row is one of these.  Read the whole
        table before quoting the headline: the cardinality was chosen from it,
        and this is what shows whether the answer depended on that choice.  It
        is also the honest form of the result -- MOS-ness is a curve over note
        counts, and a single number is one point of it.

    Examples
    --------
    See :func:`mos_ness`.
    """

    cardinality: int
    cardinality_rule: str
    n_targets: int
    n_merged: int
    period: float
    edo_error_cents: float
    mos_error_cents: float
    ternary_error_cents: Optional[float]
    mos_ness: float
    ternary_ness: Optional[float]
    two_step_sufficiency: Optional[float]
    signature: str
    generator: float
    generator_cents: float
    hardness: float
    ternary_word: Optional[str]
    ternary_signature: Optional[str]
    is_identifiable: bool
    ternary_step_cents: Optional[Tuple[float, float, float]] = None
    notes: Tuple[str, ...] = ()
    by_cardinality: Tuple[Dict[str, object], ...] = ()

    @property
    def ternary_collapsed(self) -> Optional[bool]:
        """True when two of the fitted ternary steps agree to within a cent.

        The ternary simplex has interior lines on which two step sizes coincide
        and the scale is really binary.  Landing on one is not a failure of the
        search: it is the three-step family saying, from inside itself, that the
        third size was not wanted.

        ``None`` when the ternary rung was not computed.
        """
        if self.ternary_step_cents is None:
            return None
        low, mid, high = sorted(self.ternary_step_cents)
        return bool(min(mid - low, high - mid) <= 1.0)

    @property
    def adjusted_edo_error_cents(self) -> float:
        """:attr:`edo_error_cents` after :func:`_adjusted_error`."""
        return _adjusted_error(self.edo_error_cents, self.n_targets, "edo")

    @property
    def adjusted_mos_error_cents(self) -> float:
        """:attr:`mos_error_cents` after :func:`_adjusted_error`."""
        return _adjusted_error(self.mos_error_cents, self.n_targets, "mos")

    @property
    def adjusted_ternary_error_cents(self) -> Optional[float]:
        """:attr:`ternary_error_cents` after :func:`_adjusted_error`."""
        if self.ternary_error_cents is None:
            return None
        return _adjusted_error(self.ternary_error_cents, self.n_targets, "ternary")

    @property
    def raw_mos_ness(self) -> float:
        """:attr:`mos_ness` computed on the uncorrected errors.

        The same ratio without the parameter penalty, so the size of the
        penalty is visible rather than baked in.  Always at least
        :attr:`mos_ness`.
        """
        return _share(self.mos_error_cents, self.edo_error_cents)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        tern = (
            "n/a" if self.two_step_sufficiency is None
            else f"{self.two_step_sufficiency:.2f}"
        )
        return (
            f"MOSness(N={self.cardinality}, mos_ness={self.mos_ness:.3f}, "
            f"two_step={tern}, {self.signature} @ "
            f"{self.generator_cents:.1f}c"
            f"{'' if self.is_identifiable else ', UNDERDETERMINED'})"
        )

    def to_dict(self) -> Dict[str, object]:
        """Flat, JSON-friendly summary -- one row of a table.

        ``by_cardinality`` is kept as a list of dicts rather than flattened, so
        the row this result was read off stays attached to it.
        """
        return {
            "cardinality": self.cardinality,
            "cardinality_rule": self.cardinality_rule,
            "n_targets": self.n_targets,
            "n_merged": self.n_merged,
            "period": self.period,
            "edo_error_cents": self.edo_error_cents,
            "mos_error_cents": self.mos_error_cents,
            "ternary_error_cents": self.ternary_error_cents,
            "adjusted_edo_error_cents": self.adjusted_edo_error_cents,
            "adjusted_mos_error_cents": self.adjusted_mos_error_cents,
            "adjusted_ternary_error_cents": self.adjusted_ternary_error_cents,
            "mos_ness": self.mos_ness,
            "raw_mos_ness": self.raw_mos_ness,
            "ternary_ness": self.ternary_ness,
            "two_step_sufficiency": self.two_step_sufficiency,
            "signature": self.signature,
            "generator": self.generator,
            "generator_cents": self.generator_cents,
            "hardness": self.hardness,
            "ternary_word": self.ternary_word,
            "ternary_signature": self.ternary_signature,
            "ternary_step_cents": (
                None if self.ternary_step_cents is None
                else list(self.ternary_step_cents)
            ),
            "ternary_collapsed": self.ternary_collapsed,
            "is_identifiable": self.is_identifiable,
            "n_parameters": dict(MODEL_PARAMETERS),
            "notes": list(self.notes),
            "by_cardinality": [dict(row) for row in self.by_cardinality],
        }

    def summary(self) -> str:
        """Multi-line human-readable description, ASCII only.

        Examples
        --------
        >>> from biotuner.mos.scale import MOSScale
        >>> r = mos_ness(MOSScale.from_signature(5, 2, tuning=31).ratios,
        ...              ternary=False)
        >>> print(r.summary())
        MOS-ness 1.000   at 7 notes   (cardinality chosen by: edo-evidence)
          targets        7 pitch classes, 0 merged, period 1200.000 c
          equal div.     7-EDO                  error   18.960 c   (1 parameter)
          well formed    5L2s @ 696.774 c       error    0.000 c   (2 parameters)
          three steps    not computed
          two-step suff. n/a
          note           cardinality 7 is not below the 7 targets: the fit has a degree
                         per observation and is not evidence
          note           ternary rung skipped: ternary=False
        """
        def row(label: str, middle: str, error: float, n_params: int) -> str:
            unit = "parameter" if n_params == 1 else "parameters"
            return (
                f"  {label:<14} {middle:<22} error {error:8.3f} c   "
                f"({n_params} {unit})"
            )

        lines = [
            f"MOS-ness {self.mos_ness:.3f}   at {self.cardinality} notes   "
            f"(cardinality chosen by: {self.cardinality_rule})",
            f"  targets        {self.n_targets} pitch classes, "
            f"{self.n_merged} merged, period "
            f"{PERIOD_CENTS * math.log2(self.period):.3f} c",
            row(
                "equal div.", f"{self.cardinality}-EDO",
                self.edo_error_cents, MODEL_PARAMETERS["edo"],
            ),
            row(
                "well formed", f"{self.signature} @ {self.generator_cents:.3f} c",
                self.mos_error_cents, MODEL_PARAMETERS["mos"],
            ),
        ]
        if self.ternary_error_cents is None:
            lines.append("  three steps    not computed")
            lines.append("  two-step suff. n/a")
        else:
            lines.append(
                row(
                    "three steps",
                    f"{self.ternary_signature} {self.ternary_word}",
                    self.ternary_error_cents, MODEL_PARAMETERS["ternary"],
                )
            )
            steps = ", ".join(f"{c:.1f}" for c in self.ternary_step_cents)
            collapsed = "  (two of them equal)" if self.ternary_collapsed else ""
            lines.append(f"  step sizes     {steps} c{collapsed}")
            lines.append(
                f"  two-step suff. {self.two_step_sufficiency:.3f}"
                f"                  ternary-ness {self.ternary_ness:.3f}"
            )
        for note in self.notes:
            wrapped = _wrap(note, 62)
            lines.append(f"  {'note':<14} {wrapped[0]}")
            lines.extend(" " * 17 + line for line in wrapped[1:])
        return "\n".join(lines)


def mos_ness(
    ratios: Sequence[float],
    weights: Optional[Sequence[float]] = None,
    *,
    period: float = 2.0,
    cardinality: Union[int, str] = "edo",
    min_cardinality: int = 4,
    max_cardinality: Optional[int] = None,
    tolerance_cents: float = 15.0,
    ternary: bool = True,
    ternary_max_cardinality: int = 10,
    grid: int = 720,
    include_intervals: bool = True,
    fold: bool = True,
    allow_underdetermined: bool = False,
) -> MOSness:
    """How much of a signal's structure needs a *generator*.

    Not "which moment-of-symmetry scale is this signal in" -- that question is
    not answerable from spectral peaks, because the fitted generator moves by
    hundreds of cents when the peak extraction changes -- but "is this signal
    well formed **at all**, more so than an equally spaced scale of the same
    size would be".  The second question survives the instability of the first,
    because it is a ratio between two fits to the same data at the same
    cardinality and most of what moves moves in both.

    Three families are fitted at one shared cardinality and scored through the
    same code path with the same transposition search:

    1. the equal division -- maximally even, no shape parameters;
    2. the best well-formed scale -- one generator;
    3. the best admissible three-step scale -- two shape parameters.

    :attr:`MOSness.mos_ness` is how much of (1)'s error the generator removes;
    :attr:`MOSness.two_step_sufficiency` is how little (3) adds on top.  Both
    are computed on errors corrected for the number of parameters that produced
    them -- see :func:`_adjusted_error` for the rule and its limits.

    Parameters
    ----------
    ratios : sequence of float
        Frequency ratios to explain -- typically ``bt.peaks_ratios``.
    weights : sequence of float, optional
        Per-ratio importance, e.g. peak amplitudes.  Normalised internally;
        uniform when omitted.
    period : float, default 2.0
        Period as a frequency ratio.  ``2.0`` is the octave.
    cardinality : int or {'edo', 'mos', 'max'}, default 'edo'
        Where to make the comparison.

        - ``'edo'`` (default) -- the cardinality at which the *equal division*
          carries the most :attr:`~biotuner.mos.derive.MOSFit.evidence`, i.e.
          the note count at which the signal's pitch classes are most evenly
          spread.  The generator plays no part in choosing it, so the choice
          cannot inflate the answer.  That matters: over sixty draws of twenty
          uniform ratios the null level of :attr:`MOSness.mos_ness` averages
          0.09 under this rule and 0.18 under ``'mos'``, while a real MOS
          reaches 1.0 under either.  Recorded as ``'edo-evidence'``.
        - ``'mos'`` -- the cardinality at which the *MOS* fit carries the most
          evidence, i.e. the note count that best supports the well-formed
          reading.  A selection made on the hypothesis under test, so it biases
          :attr:`MOSness.mos_ness` upward; use it to ask "at how many notes is
          this signal well formed", not to compare conditions.  Recorded as
          ``'mos-evidence'``.
        - an ``int`` -- use it.  Recorded as ``'explicit'``.
        - ``'max'`` -- the largest cardinality searched.

        :attr:`MOSness.by_cardinality` carries every candidate either way, so
        the effect of the rule is always visible after the fact.
    min_cardinality : int, default 4
        Smallest note count to consider.  Below 4 the equal division has so few
        degrees that everything fits it badly and the ratio is noisy.
    max_cardinality : int, optional
        Largest note count to consider.  Defaults to the number of distinct
        pitch classes, which is the largest cardinality at which the scale has
        no spare degrees.  Larger values need ``allow_underdetermined``.
    tolerance_cents : float, default 15.0
        What counts as a hit, passed straight through to the scoring function
        for parity with :func:`~biotuner.mos.derive.fit_mos`.  It feeds only
        that function's coverage figure, which nothing on :class:`MOSness`
        reads, so changing it cannot change any number reported here.
    ternary : bool, default True
        Fit the three-step rung.  Turn off to halve the runtime when only
        :attr:`MOSness.mos_ness` is wanted.
    ternary_max_cardinality : int, default 10
        Skip the ternary rung above this note count, recording why in
        :attr:`MOSness.notes`.  The word enumeration behind
        :func:`_ternary_words_at` grows like ``3**N / N``: about 0.03 s at 7
        notes, 0.7 s at 10, 7 s at 12 and 25 s at 13.  Raise it if you need it
        and can wait; the result is cached per cardinality.
    grid : int, default 720
        Background generator-grid resolution for the MOS search.  ``0`` uses
        only signal-derived candidates.
    include_intervals : bool, default True
        Also try every interval *between* observed ratios as a generator.
    fold : bool, default True
        Reduce the ratios to distinct pitch classes first, as
        :func:`~biotuner.mos.derive.fit_mos` does.
    allow_underdetermined : bool, default False
        Permit a cardinality above the number of targets.  Refused by default:
        a scale with spare degrees can be rotated until everything lands
        somewhere, and its error stops being a measurement.

    Returns
    -------
    MOSness

    Raises
    ------
    ValueError
        If there are fewer than four distinct pitch classes (three fitted
        parameters need more than three observations to mean anything); if an
        explicit ``cardinality`` sits below ``min_cardinality``; or if a
        cardinality above the number of targets is requested without
        ``allow_underdetermined``.

    Notes
    -----
    The equal division's own generator -- ``(N-1)/N`` of the period, whose
    degenerate MOS is exactly ``N``-EDO -- is planted in the candidate list at
    every cardinality searched.  The MOS family therefore *contains* the null,
    so the best MOS can never do worse than the equal division except by the
    parameter correction, and :attr:`MOSness.mos_ness` is a genuine improvement
    rather than an artefact of two searches missing each other.

    **The null is not zero, the parameter correction does not make it zero, and
    the null band below does not transfer to your data.**  Sixty draws of
    *twenty* uniformly random ratios, searched over four to nine notes, give a
    mean MOS-ness of 0.088, a median of 0.082, a 90th percentile of 0.177 and a
    maximum of 0.330 (0.183 / 0.173 / 0.332 / 0.565 under
    ``cardinality='mos'``).  ``test_random_uniform_ratios_score_near_zero``
    re-measures that band rather than taking it on trust.

    Those numbers are for twenty targets and are **much too low for a typical
    spectral-peak set**.  The null rises steeply as targets get scarcer: at the
    seven or eight targets a peak finder actually returns, matched noise scores
    a median of 0.25 to 0.42, so more than half of pure noise clears the 0.330
    quoted above as a maximum.  Reading a real measurement against the wrong
    band is how you conclude a biosignal is well formed when it is not.
    **Generate your own null, matched to your own target count.**

    Matched on *what*, specifically, matters as much as matching at all.  For
    spectral peaks the surrogate must copy the real set's smallest separation
    between peaks, not just its count and frequency span.  A peak finder cannot
    return two peaks on top of each other, so a real set carries a minimum
    spacing that a log-uniform draw does not; two surrogate peaks landing close
    together fold to adjacent pitch classes and force a hole elsewhere, and the
    equal division in the denominator of this measure is precisely an evenness
    statistic.  The consequence is not subtle -- on one EEG dataset the same
    epochs at the same cardinality gave Cliff's delta -0.12 against a
    log-uniform surrogate and +0.04 against a spacing-matched one.  The sign of
    the answer was decided by the null, not by the data.

    The reason is worth stating, because it bounds what
    :func:`_adjusted_error` can do.  A generator is one number, but it is a
    number with enormous leverage: moving it slides *every* degree of the scale
    at once, in a correlated way.  Counting it as one parameter out of ``n``
    residuals -- which is what a degrees-of-freedom correction does -- charges
    it far less than it is worth on random data, where it removes about a tenth
    of the error rather than the twentieth the count would predict.  So read
    :attr:`MOSness.mos_ness` against the band above, not against zero: below
    roughly 0.2 is "no more well formed than noise", and the separation from a
    genuine MOS (1.0) is the whole usable range.

    Examples
    --------
    A scale that really is a moment of symmetry -- 31-EDO meantone -- is
    explained by its generator and not at all by an equal division of the same
    size:

    >>> from biotuner.mos.scale import MOSScale
    >>> meantone = MOSScale.from_signature(5, 2, tuning=31)
    >>> r = mos_ness(meantone.ratios, ternary=False)
    >>> r.cardinality, r.signature, round(r.mos_ness, 6)
    (7, '5L2s', 1.0)
    >>> round(r.edo_error_cents, 3), round(r.mos_error_cents, 9)
    (18.96, 0.0)

    Seven notes from seven pitch classes is a degree per observation, and the
    result says so rather than quietly pretending otherwise:

    >>> r.is_identifiable
    False
    >>> r.notes[0].startswith('cardinality 7 is not below the 7 targets')
    True

    An equal division is the case where the generator buys nothing.  7-EDO
    scores exactly zero -- both families fit it perfectly, so there is no error
    left for the extra parameter to remove:

    >>> edo7 = [2 ** (k / 7) for k in range(7)]
    >>> r = mos_ness(edo7, cardinality=7, ternary=False)
    >>> round(r.edo_error_cents, 9), r.mos_ness
    (0.0, 0.0)

    A genuinely three-step scale is *partly* well formed -- a MOS gets most of
    the way there, because three step sizes are a perturbation of two -- and
    the third step is what finishes the job.  ``two_step_sufficiency`` is what
    says so:

    >>> from biotuner.mos.ternary import TernaryScale
    >>> t = TernaryScale.from_barycentric('LMLsLMs', 0.52, 0.30, 0.18)
    >>> r = mos_ness(t.ratios, cardinality=7)
    >>> round(r.mos_error_cents, 3), round(r.ternary_error_cents, 9)
    (5.6, 0.0)
    >>> round(r.two_step_sufficiency, 2)
    0.68

    On the well-formed scale above the same rung finds nothing to add: it lands
    on an interior line of the ternary simplex where two of the three step
    sizes coincide, which is the three-step family declining the third step.

    >>> r = mos_ness(meantone.ratios, cardinality=7)
    >>> r.two_step_sufficiency, r.ternary_collapsed
    (1.0, True)
    >>> [round(c, 3) for c in r.ternary_step_cents]
    [193.548, 116.129, 193.548]
    """
    D = _derive()

    if min_cardinality < 3:
        raise ValueError(
            f"min_cardinality must be at least 3 (a 2-note scale has one "
            f"interval and no shape to measure), got {min_cardinality}"
        )
    if isinstance(cardinality, str) and cardinality not in ("mos", "edo", "max"):
        raise ValueError(
            f"cardinality must be an int or one of 'mos', 'edo', 'max'; got "
            f"{cardinality!r}"
        )

    # The trailing permutation maps the canonical target order back onto the
    # caller's; nothing here is reported per target, so it is not needed.
    positions, w, n_merged, targets, _ = D._prepare_targets(
        ratios, weights, period, fold=fold
    )
    n_targets = int(len(positions))
    if n_targets <= max(MODEL_PARAMETERS.values()):
        raise ValueError(
            f"mos_ness needs more than {max(MODEL_PARAMETERS.values())} distinct "
            f"pitch classes -- the richest family fits that many parameters, so "
            f"at or below it every family reaches zero error and the comparison "
            f"is vacuous; got {n_targets} "
            f"({n_merged} further ratios merged into them)"
        )

    top = n_targets if max_cardinality is None else int(max_cardinality)
    if isinstance(cardinality, int) and not isinstance(cardinality, bool):
        if cardinality < min_cardinality:
            raise ValueError(
                f"cardinality {cardinality} is below min_cardinality "
                f"{min_cardinality}; lower min_cardinality to compare there"
            )
        top = max(top, cardinality)
    if top > n_targets and not allow_underdetermined:
        raise ValueError(
            f"cardinality {top} exceeds the {n_targets} targets: a scale with "
            f"spare degrees can be transposed until every target lands on one, "
            f"so its error is not a measurement. Pass "
            f"allow_underdetermined=True to do it anyway."
        )
    low = min(int(min_cardinality), top)
    cards = list(range(low, top + 1))
    if not cards:  # pragma: no cover - guarded by the clamp above
        raise ValueError(
            f"empty cardinality range: {min_cardinality} .. {top}"
        )

    # Signal-derived generators, plus one per cardinality that *is* the equal
    # division.  Planting the null inside the alternative's search space is what
    # makes "the generator removed this much error" a nesting statement rather
    # than a race between two independent searches.
    candidates = list(
        D.generator_candidates(
            targets, period, include_intervals=include_intervals, grid=grid
        )
    )
    candidates.extend((card - 1) / card for card in cards)

    edo_fits = {
        card: D._evaluate(
            _equal_division(card, period), positions, w, tolerance_cents, 0.0,
            align=True, n_anchors=None,
        )
        for card in cards
    }
    mos_fits = _mos_by_cardinality(
        positions, w, period, cards, candidates, tolerance_cents
    )

    rows: List[Dict[str, object]] = []
    for card in cards:
        mos_fit = mos_fits.get(card)
        if mos_fit is None:
            # No generator in the candidate set admits a well-formed scale of
            # this size.  Cannot happen once the (N-1)/N generator is planted,
            # but a caller-supplied grid of 0 with pathological ratios could.
            continue
        edo_fit = edo_fits[card]
        rows.append(
            {
                "cardinality": card,
                "edo_error_cents": float(edo_fit.error_cents),
                "mos_error_cents": float(mos_fit.error_cents),
                "mos_ness": _share(
                    _adjusted_error(mos_fit.error_cents, n_targets, "mos"),
                    _adjusted_error(edo_fit.error_cents, n_targets, "edo"),
                ),
                "signature": mos_fit.scale.signature,
                "generator_cents": float(mos_fit.scale.generator_cents),
                "mos_evidence": float(mos_fit.evidence),
                "edo_evidence": float(edo_fit.evidence),
                "identifiable": bool(card < n_targets),
            }
        )
    if not rows:  # pragma: no cover - defensive
        raise ValueError(
            "no well-formed scale could be fitted at any cardinality in "
            f"{low} .. {top}; widen the range or raise `grid`"
        )

    chosen, rule = _choose_cardinality(rows, cardinality)
    mos_fit = mos_fits[chosen]
    edo_fit = edo_fits[chosen]

    notes: List[str] = []
    identifiable = chosen < n_targets
    if not identifiable:
        note = (
            f"cardinality {chosen} is not below the {n_targets} targets: the "
            "fit has a degree per observation and is not evidence"
        )
        # derive's own test, asked rather than re-derived: it fires only on
        # *spare* degrees, which is the worse half of the same problem.
        if mos_fit.is_underdetermined:
            note += (
                "; the scale also has spare degrees, so MOSFit"
                ".is_underdetermined agrees"
            )
        notes.append(note)

    ternary_fit = None
    if not ternary:
        notes.append("ternary rung skipped: ternary=False")
    elif chosen > ternary_max_cardinality:
        notes.append(
            f"ternary rung skipped: {chosen} notes is above "
            f"ternary_max_cardinality={ternary_max_cardinality}, and enumerating "
            "the admissible three-step words grows like 3**N / N"
        )
    elif chosen < 3:  # pragma: no cover - min_cardinality guards this
        notes.append("ternary rung skipped: a three-step scale needs 3 notes")
    else:
        ternary_fit = _best_ternary(
            positions, w, period, chosen, tolerance_cents
        )
        if ternary_fit is None:  # pragma: no cover - every N >= 3 has words
            notes.append(
                f"ternary rung skipped: no admissible three-step word at "
                f"{chosen} notes"
            )

    adj_edo = _adjusted_error(edo_fit.error_cents, n_targets, "edo")
    adj_mos = _adjusted_error(mos_fit.error_cents, n_targets, "mos")
    value = _share(adj_mos, adj_edo)

    ternary_error = ternary_ness = sufficiency = None
    ternary_word = ternary_signature = ternary_steps = None
    if ternary_fit is not None:
        ternary_error = float(ternary_fit.error_cents)
        ternary_ness = _share(
            _adjusted_error(ternary_error, n_targets, "ternary"), adj_edo
        )
        # A third step size that explains nothing beyond the equal division
        # leaves two step sizes with nothing to fall short of.
        sufficiency = (
            1.0 if ternary_ness <= 0.0
            else max(0.0, min(1.0, value / ternary_ness))
        )
        ternary_word = str(ternary_fit.scale.word)
        ternary_signature = str(ternary_fit.scale.signature)
        ternary_steps = tuple(float(c) for c in ternary_fit.scale.step_cents)

    return MOSness(
        cardinality=chosen,
        cardinality_rule=rule,
        n_targets=n_targets,
        n_merged=int(n_merged),
        period=float(period),
        edo_error_cents=float(edo_fit.error_cents),
        mos_error_cents=float(mos_fit.error_cents),
        ternary_error_cents=ternary_error,
        mos_ness=value,
        ternary_ness=ternary_ness,
        two_step_sufficiency=sufficiency,
        signature=mos_fit.scale.signature,
        generator=float(mos_fit.scale.generator),
        generator_cents=float(mos_fit.scale.generator_cents),
        hardness=float(mos_fit.scale.hardness),
        ternary_word=ternary_word,
        ternary_signature=ternary_signature,
        ternary_step_cents=ternary_steps,
        is_identifiable=bool(identifiable),
        notes=tuple(notes),
        by_cardinality=tuple(rows),
    )


def _choose_cardinality(
    rows: Sequence[Dict[str, object]], spec: Union[int, str]
) -> Tuple[int, str]:
    """Pick the row to report, and say which rule picked it."""
    available = [int(row["cardinality"]) for row in rows]
    if isinstance(spec, int) and not isinstance(spec, bool):
        if spec not in available:
            raise ValueError(
                f"no well-formed scale was fitted at {spec} notes; the "
                f"cardinalities that were fitted are {available}"
            )
        return spec, "explicit"
    if spec == "max":
        return max(available), "max"
    key = "mos_evidence" if spec == "mos" else "edo_evidence"
    # Ties broken toward the smaller scale: fewer degrees is the more
    # committal claim, and the evidence measure already prefers it when the
    # errors are equal.
    best = min(rows, key=lambda row: (-float(row[key]), int(row["cardinality"])))
    return int(best["cardinality"]), f"{spec}-evidence"


def _mos_by_cardinality(
    positions: np.ndarray,
    weights: np.ndarray,
    period: float,
    cards: Sequence[int],
    candidates: Sequence[float],
    tolerance_cents: float,
    n_refine: int = 3,
) -> Dict[int, object]:
    """Best well-formed scale at each cardinality, over one candidate sweep.

    This is :func:`~biotuner.mos.derive.fit_mos` restricted to a fixed note
    count and stripped of the surplus-note penalty (there is no surplus: every
    family has the same cardinality), reorganised so that one pass over the
    candidate generators serves every cardinality at once instead of one pass
    each.

    The coarse sweep uses the three-anchor transposition shortlist for speed,
    exactly as :func:`~biotuner.mos.derive.fit_mos` does; every fit that is
    *returned* has been re-scored with the exhaustive rotation search, so the
    numbers the comparison is made on all come from the same setting.
    """
    D = _derive()
    from biotuner.mos import theory as T
    from biotuner.mos.scale import MOSScale

    wanted = set(int(c) for c in cards)
    top = max(wanted)
    coarse: Dict[Tuple[int, int, int], object] = {}
    for g in candidates:
        if not 0.0 < g < 1.0:
            continue
        for card, n_large, n_small in T.mos_series(
            float(g), max_cardinality=top, include_trivial=True
        ):
            if card not in wanted:
                continue
            scale = MOSScale(n_large, n_small, float(g), period, validate=False)
            if D._is_degenerate_scale(scale):
                continue
            fit = D._evaluate(
                scale, positions, weights, tolerance_cents, 0.0,
                align=True, n_anchors=3,
            )
            key = (card, n_large, n_small)
            prev = coarse.get(key)
            if prev is None or fit.error_cents < prev.error_cents:
                coarse[key] = fit

    out: Dict[int, object] = {}
    for card in sorted(wanted):
        here = [fit for (c, _, _), fit in coarse.items() if c == card]
        if not here:
            continue
        here.sort(key=lambda f: f.error_cents)
        best = None
        for fit in here[:n_refine]:
            # `_refine_generator` scores with the exhaustive rotation search
            # throughout and returns the coarse scale unchanged when sliding the
            # generator does not help, so its result is both exactly scored and
            # never worse than what went in.  A shortlist of three signatures
            # rather than one, because the coarse sweep ranked them on the
            # approximate search and the order can change under the exact one.
            refined = D._refine_generator(
                fit.scale, positions, weights, tolerance_cents, 0.0, align=True
            )
            if best is None or refined.error_cents < best.error_cents:
                best = refined
        out[card] = best
    return out
