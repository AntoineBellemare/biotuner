"""Deriving moment-of-symmetry scales from biosignals.

The scale labyrinth is a *space*: every point is a (period, generator) pair, and
every ring a cardinality.  Milne et al. (2011) treat it as an instrument -- a
surface a performer navigates by hand.  This module treats it as a search
space instead, and asks the question the paper does not: given a signal's
spectral peaks, **which well-formed scale best explains them?**

The fit has three coordinates, matching the three choices the labyrinth
affords:

``generator``
    Where around the circle.  Candidates come from the signal itself -- every
    observed ratio, and every ratio between ratios, is a generator worth
    trying -- plus a background grid so nothing is missed.
``cardinality``
    Which ring.  Only the generator's own MOS cardinalities are considered,
    since those are the only note-counts at which a generated scale is
    well-formed at all.
``period``
    How big the circle is.  Usually the octave, but the paper is explicit that
    the period is a free parameter (a "pseudo-octave"), and a biosignal has no
    particular reason to prefer 2/1.  Set ``optimize_period=True`` to fit it.

Plus one nuisance coordinate the labyrinth does not show, because it is not a
property of the scale at all:

``transposition``
    Where the scale sits relative to the signal.  A scale and its
    transpositions are the same scale, so each candidate is free to slide onto
    the data.  Without that, a stack of fifths does not read as the pentatonic
    -- the pentatonic matches only in one of its five modes, and rooting every
    candidate at 1/1 picks the wrong one.  :attr:`MOSFit.offset` reports the
    fitted transposition and :attr:`MOSFit.mode` says which mode it lands in.

The objective is amplitude-weighted mean absolute cents error from each peak
ratio to its nearest scale degree, plus a penalty for surplus notes -- without
which a 53-note MOS would always "win" by brute coverage.

Two directions
--------------
That search is the **inverse** direction, and it is the default: the generator
is a *latent* parameter, recovered from the peaks without ever having to appear
between two of them.  Delete 3/2 from a stack of fifths and :func:`fit_mos`
still reports 701.96 cents, because 27/16 over 9/8 is a fifth.

:func:`forward_scales` asks the other question.  It takes an interval the
signal *actually states* -- the quotient of two peaks, or a peak ratio itself
-- declares it the generator, stacks it, and reads off the MOS that comes out.
Not "which latent generator explains these peaks" but "if this observed
interval were the generator, what scale would the signal be playing".  Nothing
is optimised there, so the result is a consequence rather than a fit.

The two are made comparable by scoring them the same way: every forward
reading is measured against the whole target set with the same objective and
the same transposition freedom the inverse search uses, so its ``error_cents``,
``coverage`` and ``evidence`` mean what they mean on a :class:`MOSFit`.  Both
also fold their generators into the bright half of the period, so the two land
on the same axis of the same plot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from biotuner.mos import theory as T
from biotuner.mos.scale import MOSScale

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

__all__ = [
    "FOLD_TOLERANCE_CENTS",
    "GENERATOR_EPSILON",
    "MIN_REFINED_STEP",
    "MIN_AUDIBLE_STEP_CENTS",
    "MOSFit",
    "FitField",
    "ForwardScale",
    "fit_field",
    "generator_candidates",
    "labyrinth_positions",
    "fit_mos",
    "forward_scales",
    "best_mos",
    "mos_tuning",
    "mos_from_biotuner",
    "compare_sources",
    "trajectory_from_windows",
    "mos_trajectory",
    "trajectory_dataframe",
    "explain_fit",
]

#: Cents within which two ratios count as the same pitch class when
#: :func:`fit_mos` folds its targets into the period.
#:
#: One cent is the same threshold :func:`generator_candidates` already uses to
#: thin its generator list, and the choice is bounded from both sides.  It has
#: to be large enough to absorb the near-duplicates a real derivation emits --
#: two peak pairs giving 1.1250 and 1.1249 are one pitch class measured twice,
#: 0.15 cents apart -- and small enough that it can never merge two degrees the
#: fit would otherwise distinguish, which at a 15-cent default hit tolerance
#: leaves more than an order of magnitude of headroom.
FOLD_TOLERANCE_CENTS = 1.0

#: Period fractions closer together than this are the same number, not two
#: candidate generators -- and a generator within this of a boundary of the
#: bright half is *on* that boundary.
#:
#: The unit is a fraction of the period, so the constant is period-independent;
#: at the octave it is 1.2e-6 cents.  It is bounded from both sides, and the
#: gap between the bounds is enormous, which is why a round number in the
#: middle is safe rather than tuned:
#:
#: *From below.*  Every generator here arrives through ``log(a) / log(b)``, and
#: the identities that ought to hold exactly do not.  ``log(2 ** 0.5) /
#: log(2)`` is ``0.5000000000000001``, not ``0.5``; a difference of positions
#: accumulates a few more ulps.  Empirically that noise stays under 1e-14, and
#: 1e-9 is five orders of magnitude above it.
#:
#: *From above.*  The narrowest thing this must not swallow is a genuine
#: generator a few cents from half the period -- 3 cents is 2.5e-3 of an octave
#: -- and 1e-9 is six orders of magnitude below that.  Even the sub-cent
#: distinctions :func:`_refine_generator` exists to preserve are 1e-6 of a
#: period, a thousand times coarser.
GENERATOR_EPSILON = 1e-9

#: Smallest step, as a fraction of the period, a *refined* scale may have.
#:
#: :data:`GENERATOR_EPSILON` asks whether two degrees are the same number.
#: This asks the different and larger question of whether they are the same
#: *note*, and it exists because :func:`_refine_generator` will happily answer
#: the first one correctly and the second one wrongly: sliding the generator to
#: the edge of its tuning range collapses the scale onto a smaller one, which
#: fits any data at least as well as the scale it claims to be, so the
#: optimiser goes there whenever the data lets it.  On five-tone equal input
#: that returned a ``5L6s`` whose eleven degrees were five pitch classes 0.0002
#: cents apart -- arithmetically distinct, musically one note each.
#:
#: 1e-5 of the period is 0.012 cents at the octave, and the choice is bounded
#: from both sides.  *From below*: four orders of magnitude above
#: :data:`GENERATOR_EPSILON`, so it can never fire on arithmetic noise, and an
#: order above the 1e-6 of a period that is the finest generator distinction
#: this module claims to resolve at all -- a step below the resolution of the
#: generator that produced it is not a step.  *From above*: across the fitted
#: corpus in ``tests/mos`` the narrowest step any scale legitimately wanted was
#: 0.81 cents, seventy times wider, and the collapsed ones sat at 0.0002 cents,
#: sixty times narrower.  The gap between the two populations is four orders of
#: magnitude wide, which is why a round number in the middle of it is safe
#: rather than tuned.
MIN_REFINED_STEP = 1e-5

#: Smallest step, in cents, that makes two degrees separate *notes*.
#:
#: :data:`GENERATOR_EPSILON` asks whether two degrees are the same number and
#: :data:`MIN_REFINED_STEP` bounds what the refiner may do; this asks the
#: musical question, and it is the one that decides whether a scale gets
#: returned at all.
#:
#: The two populations it separates were measured, not guessed.  Over a corpus
#: of forward and inverse fits the scales that had collapsed onto a smaller one
#: carried a smallest step of 0.012 to 0.030 cents and a hardness (``L / s``)
#: between 10158 and 24810 -- a "two step size" scale whose second step size is
#: four orders too small to be one.  Every scale that had not collapsed had
#: steps tens to hundreds of cents wide and a hardness of 1 to 6.  Nothing lands
#: in between.
#:
#: One cent sits 33x above the worst collapsed case, an order of magnitude below
#: the smallest step a well-formed scale in this cardinality range legitimately
#: has -- 53-EDO, far beyond the default ceiling of 24, still has 22-cent steps
#: -- and five times below the melodic just-noticeable difference, so nothing it
#: rejects could have been heard as two notes anyway.
MIN_AUDIBLE_STEP_CENTS = 1.0


# --------------------------------------------------------------------------- #
# Preparation
# --------------------------------------------------------------------------- #
def _as_positions(ratios: Sequence[float], period: float) -> np.ndarray:
    """Reduce frequency ratios to positions in ``[0, 1)`` of the period."""
    r = np.asarray(ratios, dtype=float)
    r = r[np.isfinite(r) & (r > 0)]
    if r.size == 0:
        raise ValueError(
            "no usable ratios: every value was non-finite or non-positive"
        )
    return np.mod(np.log(r) / np.log(period), 1.0)


def _clean_weights(
    weights: Optional[Sequence[float]], n: int
) -> np.ndarray:
    if weights is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise ValueError(
            f"weights must have one entry per ratio: got {w.shape} for {n} ratios"
        )
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    total = w.sum()
    if total <= 0:
        return np.full(n, 1.0 / n)
    return w / total


def _circular_distance(positions: np.ndarray, degrees: np.ndarray) -> np.ndarray:
    """Shortest distance from each position to any degree, around the period."""
    d = np.abs(positions[:, None] - degrees[None, :])
    d = np.minimum(d, 1.0 - d)
    return d


def _merge_pitch_classes(
    positions: np.ndarray,
    weights: np.ndarray,
    period: float,
    tolerance_cents: float,
) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Collapse positions that name the same pitch class into one target.

    A scale has no way to tell ``1/1`` from ``2/1``: they are one degree, and
    counting them as two targets both inflates ``n_targets`` and averages the
    unison's error in twice.  Merging is done on the circle, so ``0.999`` and
    ``0.001`` of a period meet as well.

    The weights of merged entries are *summed*, not discarded.  Two peaks an
    octave apart are two independent pieces of evidence for the same pitch
    class, and a fit that heard both should be pulled there harder than one
    that heard only one.

    ``positions`` **must arrive sorted ascending** -- :func:`_prepare_targets`
    is the only caller and sorts first.  That is not an optimisation, it is what
    makes the grouping a function of the multiset.  Greedy first-match grouping
    is order-sensitive by nature: three positions a tolerance apart in a chain
    group as ``{a, b}, {c}`` read left to right and as ``{c, b}, {a}`` read
    right to left.  Feeding the algorithm in ascending order fixes one answer
    for one multiset, and makes the surviving representative the *lowest*
    member of its group rather than whichever the caller happened to list
    first.

    Returns
    -------
    positions, weights, n_merged, group
        ``positions`` holds one representative per pitch class, ascending;
        ``n_merged`` is how many entries were absorbed; ``group[k]`` is the
        index of the representative that input entry ``k`` ended up in, so a
        caller can map its own rows onto the merged ones.
    """
    tol = tolerance_cents / (T.PERIOD_CENTS * math.log2(period))
    kept_pos: List[float] = []
    kept_w: List[float] = []
    group = np.empty(len(positions), dtype=np.intp)
    for i, (p, w) in enumerate(zip(positions, weights)):
        for k, q in enumerate(kept_pos):
            d = abs(float(p) - q)
            if min(d, 1.0 - d) <= tol:
                kept_w[k] += float(w)
                group[i] = k
                break
        else:
            group[i] = len(kept_pos)
            kept_pos.append(float(p))
            kept_w.append(float(w))
    n_merged = int(len(positions) - len(kept_pos))
    return (
        np.asarray(kept_pos, dtype=float),
        np.asarray(kept_w, dtype=float),
        n_merged,
        group,
    )


def _prepare_targets(
    ratios: Sequence[float],
    weights: Optional[Sequence[float]],
    period: float,
    fold: bool = True,
    fold_tolerance_cents: float = FOLD_TOLERANCE_CENTS,
) -> Tuple[np.ndarray, np.ndarray, int, Tuple[float, ...], np.ndarray]:
    """Everything :func:`fit_mos` needs from its inputs, in one canonical pass.

    Drops unusable ratios (carrying their weights with them), reduces the rest
    to positions in ``[0, 1)`` of the period, **sorts them ascending**,
    optionally folds octave-equivalent ratios together, and normalises the
    weights *after* folding so merged entries keep their combined share.

    The sort is the load-bearing step.  A set of ratios is a *set*: the answer
    must be a function of the multiset and of nothing else.  It was not,
    because three separate things downstream read the targets in the order they
    arrived -- :func:`_merge_pitch_classes` kept whichever member of a group
    came first, :func:`_offset_candidates` broke weight ties by position in the
    array (and the weights are uniform by default, so *every* tie was broken
    that way), and :func:`_evaluate` summed the weighted errors in that order,
    which decides the last bits of a float and therefore decides exact ties.
    Canonicalising once here fixes all three at their source, rather than
    patching three call sites that would have to stay patched.

    Sorting would break the one property the caller is promised -- that
    :attr:`MOSFit.targets`, :attr:`MOSFit.assignments` and
    :attr:`MOSFit.residuals` run parallel to the ratios handed in -- so the
    permutation is returned rather than discarded, and
    :func:`_in_caller_order` puts the per-target vectors back before any of
    them is seen from outside.

    Returns
    -------
    positions, weights, n_merged, targets, order
        ``positions``, ``weights`` and ``targets`` are in canonical
        (ascending-position) order; ``targets`` is the fitted ratios folded
        back into ``[1, period)``.  ``order`` maps canonical slots onto the
        caller's: ``order[j]`` is the canonical index of the ``j``-th surviving
        ratio in input order, so ``[targets[i] for i in order]`` is what the
        caller sees.  Where several ratios merged into one pitch class, the
        group takes the place of its earliest member and is represented by its
        lowest position.

    Examples
    --------
    Two ratios given high-then-low come back canonically ordered, with
    ``order`` restoring what the caller asked for:

    >>> pos, w, n, tg, order = _prepare_targets([1.5, 1.25], None, 2.0)
    >>> [round(t, 4) for t in tg]
    [1.25, 1.5]
    >>> [int(i) for i in order]
    [1, 0]
    >>> [round(tg[i], 4) for i in order]
    [1.5, 1.25]
    """
    r = np.asarray(ratios, dtype=float).ravel()
    w_raw: Optional[np.ndarray] = None
    if weights is not None:
        w_raw = np.asarray(weights, dtype=float).ravel()
        if w_raw.shape != r.shape:
            raise ValueError(
                f"weights must have one entry per ratio: got {w_raw.shape} "
                f"for {r.size} ratios"
            )
    # The mask is applied to the weights too, so dropping a NaN ratio does not
    # silently shift every remaining weight onto the wrong target.
    usable = np.isfinite(r) & (r > 0)
    r = r[usable]
    if r.size == 0:
        raise ValueError(
            "no usable ratios: every value was non-finite or non-positive"
        )
    if w_raw is None:
        w_raw = np.ones(r.size, dtype=float)
    else:
        w_raw = w_raw[usable]
        w_raw = np.where(np.isfinite(w_raw) & (w_raw > 0), w_raw, 0.0)

    positions = np.mod(np.log(r) / np.log(period), 1.0)
    # Canonical order, and the map back.  ``rank[k]`` is the input row that
    # sits in canonical slot ``k``; ``slot[i]`` is where input row ``i`` went.
    rank = np.argsort(positions, kind="stable")
    positions = positions[rank]
    w_raw = w_raw[rank]
    slot = np.empty(rank.size, dtype=np.intp)
    slot[rank] = np.arange(rank.size, dtype=np.intp)

    n_merged = 0
    if fold:
        positions, w_raw, n_merged, group = _merge_pitch_classes(
            positions, w_raw, period, fold_tolerance_cents
        )
    else:
        group = np.arange(positions.size, dtype=np.intp)

    # Each surviving target takes the place of the *earliest* input row that
    # landed in it, so the caller's per-target vectors still read in the order
    # the caller's own list would suggest.
    first_row = np.full(positions.size, rank.size, dtype=np.intp)
    for row in range(rank.size):
        g = int(group[slot[row]])
        if row < first_row[g]:
            first_row[g] = row
    order = np.argsort(first_row, kind="stable")

    total = float(w_raw.sum())
    if total > 0:
        w = w_raw / total
    else:
        w = np.full(positions.size, 1.0 / positions.size)
    targets = tuple(float(period**p) for p in positions)
    return positions, w, n_merged, targets, order


def _in_caller_order(
    fit: MOSFit,
    order: np.ndarray,
    n_merged: int,
    targets: Tuple[float, ...],
) -> MOSFit:
    """Re-express a fit's per-target vectors in the order the caller used.

    Everything inside the search runs on the canonical target order
    :func:`_prepare_targets` imposes.  That order is an implementation detail:
    :attr:`MOSFit.targets`, :attr:`MOSFit.assignments` and
    :attr:`MOSFit.residuals` are documented to run parallel to the ratios that
    were handed in, and they still do -- this is where the permutation is
    undone, on the way out.
    """
    idx = [int(i) for i in order]
    return replace(
        fit,
        assignments=tuple(fit.assignments[i] for i in idx),
        residuals=tuple(fit.residuals[i] for i in idx),
        targets=tuple(targets[i] for i in idx),
        n_merged=n_merged,
    )


# --------------------------------------------------------------------------- #
# Candidate generators
# --------------------------------------------------------------------------- #
def _fold_bright(value: float) -> Optional[float]:
    """Fold a period fraction into the bright half ``(0.5, 1)``, or ``None``.

    A generator and its complement within the period build the same scale
    (Milne et al. §4) -- the two pitch-class sets are mirror images, and for a
    well-formed scale the mirror is always a *mode* of the original.  Since
    every fit here is rotation-invariant, ``g`` and ``1 - g`` are one solution
    and only the bright spelling is ever reported.

    ``None`` comes back for the fractions that generate nothing: ``0`` (the
    unison and the bare period, which never leave the root) and ``1/2`` (which
    closes after two notes).  Both are refused *with a tolerance* of
    :data:`GENERATOR_EPSILON` rather than by exact comparison, because neither
    value survives the arithmetic that produces it.  A half-period interval
    arrives as ``log(2 ** 0.5) / log(2)``, which is ``0.5000000000000001``:
    exact-equality rejection lets it straight through, and the scales built on
    it then claim a cardinality they do not have -- five degrees at
    ``[0, 0, 0, 600, 600]`` cents is two pitch classes wearing a five-note
    label.  Anything within an epsilon of a boundary *is* that boundary.

    Both directions of the search share this function precisely so that a
    forward reading and an inverse fit land on the same axis and can be drawn
    on one plot.

    Examples
    --------
    The two degenerate fractions, and the floating-point neighbour that used to
    slip past them:

    >>> _fold_bright(0.5) is None, _fold_bright(0.0) is None
    (True, True)
    >>> _fold_bright(math.log(2 ** 0.5) / math.log(2)) is None
    True

    A generator three cents from half the period is a real generator, and is
    not touched:

    >>> round(_fold_bright(0.5 + 3 / 1200), 6)
    0.5025
    """
    g = float(value) % 1.0
    if g < 0.5:
        g = 1.0 - g
    if g <= 0.5 + GENERATOR_EPSILON or g >= 1.0 - GENERATOR_EPSILON:
        return None
    return g


def _is_degenerate_scale(scale: MOSScale) -> bool:
    """True when a scale has fewer distinct pitch classes than it claims degrees.

    A belt-and-braces invariant, checked on every scale either direction is
    about to score.  :func:`_fold_bright` already refuses the generators known
    to collapse, but "the pitch classes are fewer than the notes" is a property
    of the *scale*, provable from its own degrees and independent of whatever
    produced it -- so it is worth asserting where the scale is built rather
    than only where the generator is chosen.  A cardinality that overstates the
    note count corrupts everything downstream: the surplus-note penalty, the
    chance error, ``n_unmatched_degrees``, and the signature itself.

    Distinctness is judged at :data:`MIN_AUDIBLE_STEP_CENTS`, not at an
    arithmetic epsilon, because the scales that escape are not numerically
    identical -- they are separated by hundredths of a cent, distinct as floats
    and indistinguishable as music.  A guard set at ``GENERATOR_EPSILON`` is
    four orders too fine to catch them and never fires.

    That threshold cannot reject a legitimate equal division.  Twelve-tone
    equal temperament read as ``7L5s`` has twelve pitch classes a full 100
    cents apart, and even 53-EDO -- twice the default cardinality ceiling --
    has 22-cent steps.

    Examples
    --------
    An equal temperament is *not* degenerate in this sense -- its steps are all
    the same size, but its pitch classes are all different:

    >>> edo12 = MOSScale(7, 5, 7 / 12, validate=False)
    >>> edo12.is_degenerate, _is_degenerate_scale(edo12)
    (True, False)

    A generator at half the period is:

    >>> _is_degenerate_scale(MOSScale(2, 3, 0.5000000000000001, validate=False))
    True

    So is a scale whose second step size has shrunk to a hundredth of a cent,
    which is what the arithmetic epsilon used to wave through:

    >>> collapsed = MOSScale(4, 17, 0.7499999999, validate=False)
    >>> round(_min_step(collapsed) * collapsed.period_cents, 4)
    0.0
    >>> _is_degenerate_scale(collapsed)
    True
    """
    return _min_step(scale) * scale.period_cents <= MIN_AUDIBLE_STEP_CENTS


def _min_step(scale: MOSScale) -> float:
    """The smallest gap between two of a scale's pitch classes, as a fraction.

    Circular: the step from the last degree back up to the period counts as
    much as the steps between neighbours.  ``1.0`` for a scale with fewer than
    two degrees, which has no step to measure.

    Examples
    --------
    >>> round(_min_step(MOSScale(7, 5, 7 / 12, validate=False)), 6)
    0.083333
    """
    degrees = sorted(scale.degrees)
    if len(degrees) < 2:
        return 1.0
    gaps = [b - a for a, b in zip(degrees, degrees[1:])]
    gaps.append(1.0 - degrees[-1] + degrees[0])
    return min(gaps)


def generator_candidates(
    ratios: Sequence[float],
    period: float = 2.0,
    include_intervals: bool = True,
    grid: int = 720,
    dedupe_cents: float = 1.0,
) -> List[float]:
    """Generator fractions worth trying for a set of observed ratios.

    Three sources, merged and de-duplicated:

    1. Each observed ratio, read as a generator in its own right.  If a
       signal's peaks really are a stack of some interval, that interval is
       among them.
    2. Every ratio *between* two observed ratios, when ``include_intervals``.
       A generator need not appear as a peak -- the diatonic scale's fifth is a
       relation among its notes, not one of them.
    3. A uniform background grid, so a generator the signal only implies is
       still reachable.

    Everything is folded into the bright half ``(0.5, 1)``: a generator and its
    complement within the period build the same scale (Milne et al. §4), so
    searching one half covers the whole labyrinth.

    Parameters
    ----------
    ratios : sequence of float
        Frequency ratios, e.g. ``bt.peaks_ratios``.
    period : float, default 2.0
    include_intervals : bool, default True
    grid : int, default 720
        Background grid resolution across the full period.  ``0`` disables it.
    dedupe_cents : float, default 1.0
        How close a *grid* point may come to a candidate already kept before it
        is dropped, in cents of the period.  It does not apply between two
        signal-derived candidates -- see the notes.

    Returns
    -------
    list of float
        Sorted generator fractions, all in ``(0.5, 1)``.

    Notes
    -----
    De-duplication is priority-aware, and the priority runs one way only: the
    background grid is thinned against the signal, never the signal against
    itself.  A grid point half a cent from the exact generator would otherwise
    shadow it and the search would recover it only approximately -- but the
    same rule turned on two signal-derived candidates is strictly worse, since
    it discards a real proposal in favour of another real proposal chosen by
    nothing but sorted order.  On one recording (S004, eyes closed) that lost
    the generator at 810.302 cents to a neighbour 0.909 cents away that fits
    the peaks measurably worse, and no information available at this stage
    could have told them apart: which of two candidates is better is a question
    about the *fit*, and this function does not score anything.  Sub-cent
    precision is what :func:`_refine_generator` exists to protect, so it is not
    thrown away here.

    Signal-derived candidates are still collapsed at
    :data:`GENERATOR_EPSILON`, which removes arithmetic duplicates -- a stack
    of fifths proposes 3/2 three times over -- without ever removing a distinct
    proposal.  There are at most ``n (n + 1) / 2`` of them against a grid of
    hundreds, so keeping them all costs almost nothing.

    Examples
    --------
    A stack of fifths proposes the fifth itself, and keeps proposing it exactly
    even with a dense grid running alongside:

    >>> stack = [1.0, 1.125, 1.265625, 1.5]
    >>> fifth = math.log2(3 / 2)
    >>> any(abs(c - fifth) < 1e-12 for c in generator_candidates(stack, grid=0))
    True
    >>> any(abs(c - fifth) < 1e-12 for c in generator_candidates(stack, grid=720))
    True

    Two candidates the signal genuinely states less than a cent apart both
    survive, and the grid still does not crowd in between them:

    >>> peaks = [10.71, 10.47, 21.17, 13.12, 8.07, 17.18, 25.55]
    >>> cands = generator_candidates(peaks, grid=720)
    >>> [round(c * 1200, 5) for c in cands
    ...  if 809.0 < c * 1200 < 811.0]
    [809.39247, 810.30166]
    """
    if period <= 1:
        raise ValueError(f"period ratio must exceed 1, got {period!r}")
    pos = _as_positions(ratios, period)

    from_signal: List[float] = []
    for p in pos:
        f = _fold_bright(p)
        if f is not None:
            from_signal.append(f)
    if include_intervals:
        # Only the lower triangle: ``pos[i] - pos[j]`` and ``pos[j] - pos[i]``
        # are complements, and folding sends complements to the same bright
        # value, so the upper triangle proposes nothing new.
        for i in range(len(pos)):
            for j in range(i):
                f = _fold_bright(pos[i] - pos[j])
                if f is not None:
                    from_signal.append(f)

    tol = dedupe_cents / (T.PERIOD_CENTS * math.log2(period))

    def thin(
        values: List[float],
        spacing: float,
        against: Optional[List[float]] = None,
    ) -> List[float]:
        kept: List[float] = []
        for g in sorted(values):
            if kept and g - kept[-1] <= spacing:
                continue
            if against is not None:
                # np.searchsorted keeps this linear rather than quadratic.
                pos_i = int(np.searchsorted(against, g))
                near = against[max(0, pos_i - 1) : pos_i + 1]
                if any(abs(g - a) <= spacing for a in near):
                    continue
            kept.append(g)
        return kept

    # An epsilon, not ``dedupe_cents``: two proposals the signal really makes
    # are two candidates however close they are, and only arithmetic duplicates
    # of one proposal are collapsed.
    out = thin(from_signal, GENERATOR_EPSILON)
    if grid and grid > 0:
        from_grid = [f for f in (_fold_bright(k / grid) for k in range(1, grid)) if f]
        out = sorted(out + thin(from_grid, tol, against=out))
    return out


def labyrinth_positions(
    ratios: Sequence[float], period: float = 2.0, fold: bool = False
) -> List[float]:
    """Where a set of ratios sits on the labyrinth's circumference.

    The angle of each ratio, as a fraction of the period -- what
    :func:`~biotuner.mos.plotting.plot_labyrinth` needs to draw a signal's
    peaks on top of the scale universe.

    Examples
    --------
    >>> [round(p, 4) for p in labyrinth_positions([1.0, 1.5, 2.0])]
    [0.0, 0.585, 0.0]
    """
    pos = _as_positions(ratios, period)
    if fold:
        return [T.fold_generator(p) for p in pos]
    return [float(p) for p in pos]


# --------------------------------------------------------------------------- #
# The fit
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MOSFit:
    """One candidate explanation of a signal as a well-formed scale.

    Attributes
    ----------
    scale : MOSScale
    error_cents : float
        Amplitude-weighted mean absolute distance from each target ratio to its
        nearest scale degree.  The headline number.
    max_error_cents : float
    rms_error_cents : float
    coverage : float
        Weighted fraction of targets landing within ``tolerance_cents``.
    score : float
        ``error_cents`` plus the surplus-note penalty.  Fits are ranked by this,
        not by raw error, so a big scale cannot win by covering everything.
    assignments : tuple of int
        Which scale degree each target matched, parallel to :attr:`targets`.
    residuals : tuple of float
        Signed cents error per target (target minus degree, wrapped), parallel
        to :attr:`targets`.
    n_targets : int
        How many targets were actually fitted.  With folding on -- the default
        -- that is the number of distinct pitch classes in the input, not the
        number of ratios handed in; see :attr:`n_merged`.
    offset : float
        The transposition, as a period fraction, that put the scale where the
        data is.  A scale and its transpositions are the same scale, so this is
        a fitted nuisance parameter, not a property of the structure -- but it
        says which *mode* the signal is sitting in, which is musical
        information.  See :attr:`aligned_ratios`.
    n_merged : int
        How many input ratios were absorbed into a pitch class already present.
        Four of biotuner's eight working tuning derivations emit a ratio at
        exactly 1/1 or 2/1, so this is routinely non-zero.
    targets : tuple of float
        The ratios actually fitted, folded into ``[1, period)``.  This -- not
        the caller's original list -- is what :attr:`assignments` and
        :attr:`residuals` run parallel to once anything has been merged.

        The order is the caller's, not the search's.  Everything inside the fit
        runs on targets sorted ascending, because a set of ratios has to be
        fitted as a set; these three vectors are permuted back on the way out,
        so target ``i`` is the ``i``-th ratio the caller passed, minus any that
        were unusable and with each merged group standing in the place of its
        earliest member.
    """

    scale: MOSScale
    error_cents: float
    max_error_cents: float
    rms_error_cents: float
    coverage: float
    score: float
    assignments: Tuple[int, ...]
    residuals: Tuple[float, ...]
    n_targets: int
    offset: float = 0.0
    n_merged: int = 0
    targets: Tuple[float, ...] = ()

    @property
    def signature(self) -> str:
        return self.scale.signature

    @property
    def n_unmatched_degrees(self) -> int:
        """Scale degrees that no target landed on -- notes the signal never used."""
        return self.scale.cardinality - len(set(self.assignments))

    @property
    def aligned_degrees(self) -> List[float]:
        """The scale's degrees where the fit actually put them, rooted at the data.

        :attr:`scale` is rooted on its generator chain's origin, which is an
        arbitrary choice; the fit is free to transpose it. These are the fitted
        degrees rotated back so the tone the signal's own reference landed on
        comes first -- i.e. the *mode* the signal occupies, as a tuning.
        """
        degrees = sorted((d + self.offset) % 1.0 for d in self.scale.degrees)
        # The degree the data's reference (position 0) sits on.
        root = min(degrees, key=lambda d: min(d, 1.0 - d))
        return sorted((d - root) % 1.0 for d in degrees)

    @property
    def aligned_ratios(self) -> List[float]:
        """:attr:`aligned_degrees` as frequency ratios, starting at 1/1."""
        return [self.scale.period**d for d in self.aligned_degrees]

    @property
    def aligned_cents(self) -> List[float]:
        pc = self.scale.period_cents
        return [d * pc for d in self.aligned_degrees]

    @property
    def mode(self):
        """Which mode of :attr:`scale` the signal occupies, or ``None``.

        ``None`` only when the alignment does not land on a scale tone, which a
        degenerate tuning can produce.
        """
        target = self.aligned_cents
        for m in self.scale.modes():
            if all(abs(a - b) < 1e-6 for a, b in zip(m.cents, target)):
                return m
        return None

    @property
    def chance_error_cents(self) -> float:
        """Error a *random* set of ratios would get against a scale this size.

        Points scattered uniformly around the period sit, on average, a quarter
        of a step away from the nearest of ``N`` equally spaced degrees.  This
        is the baseline any fit has to beat, and it shrinks as the scale grows
        -- which is why a large MOS covering everything is not, by itself,
        evidence of anything.
        """
        return self.scale.period_cents / (4.0 * self.scale.cardinality)

    @property
    def improvement(self) -> float:
        """How many times better than chance the fit is.

        ``1.0`` means no better than a scale of this size would do on random
        input; large values mean the signal really does sit on these degrees.
        Infinite for an exact fit -- where "exact" means below a nanocent,
        since an exactly-recovered scale lands at 1e-13 rather than 0 and a
        ratio like 9e13 is arithmetic noise reported as if it were a finding.
        """
        if self.error_cents <= 1e-9:
            return float("inf")
        return self.chance_error_cents / self.error_cents

    @property
    def is_underdetermined(self) -> bool:
        """True when the scale has more degrees than the data had targets.

        A scale with spare notes can be rotated so that every target lands on
        some degree, so its error is not a measurement of anything:
        ``best_mos([1.5])`` reports a four-note scale at 0.000 cents from a
        single data point.  Such a fit is not returned any differently -- it may
        still be the right structure -- but it must not be read as evidence, and
        :func:`explain_fit` says so out loud.
        """
        return self.n_targets < self.scale.cardinality

    @property
    def evidence(self) -> float:
        """How many standard errors below chance the fit's mean error sits.

        :attr:`error_cents` on its own cannot be compared across fits, because
        a small target set can reach zero by luck and a large scale can reach
        it by having somewhere for everything to go.  This folds in how much
        data there was.

        Targets scattered uniformly around the period land uniformly in
        ``[0, 2 * chance]`` from the nearest of ``N`` evenly spaced degrees, a
        distribution with mean :attr:`chance_error_cents` and standard
        deviation ``chance / sqrt(3)``.  The mean of ``n`` such draws therefore
        has standard error ``chance / sqrt(3n)``, and

        ``evidence = sqrt(3 * n_targets) * (1 - error_cents / chance)``

        is how far below chance the observed mean falls in those units.  Zero
        is chance; larger is better supported.  It is a rule of thumb, not a
        test statistic -- the degrees are not exactly evenly spaced, the
        weights are not uniform, and the generator was fitted to the same data
        -- but it ranks fits in the order a reader would defend.

        Examples
        --------
        Two ratios and seven ratios can both be fitted exactly.  Only one of
        them is a finding:

        >>> two = best_mos([1.0, 1.5], max_cardinality=8)
        >>> seven = best_mos(MOSScale.from_signature(5, 2, tuning=31).ratios,
        ...                  max_cardinality=12)
        >>> round(two.error_cents, 4), round(seven.error_cents, 4)
        (0.0, 0.0)
        >>> round(two.evidence, 2), round(seven.evidence, 2)
        (2.45, 4.58)
        """
        chance = self.chance_error_cents
        if chance <= 0 or self.n_targets <= 0:
            return float("nan")
        return math.sqrt(3.0 * self.n_targets) * (1.0 - self.error_cents / chance)

    @property
    def _rank_key(self) -> Tuple[float, int, int, int, int, float]:
        """Deterministic ordering: score, then structural informativeness.

        Ties are real -- an exactly equal-tempered input is a *degenerate*
        well-formed scale (Milne et al. §2 footnote 6) and several signatures
        describe it equally well.  Among equal scores prefer, in order: the
        smaller scale; the more balanced signature (a ``1L11s`` is barely
        distinguishable from an equal division and says almost nothing); then a
        fixed lexicographic order so the answer never depends on dict ordering.

        The score is quantised to 1e-6 cents first: an exact fit lands at
        1e-13 or so rather than at zero, and comparing raw floats would let
        arithmetic noise decide the ranking instead of the tie-break.
        """
        return (
            round(self.score, 6),
            self.scale.cardinality,
            -min(self.scale.n_large, self.scale.n_small),
            self.scale.n_large,
            self.scale.n_small,
            self.scale.generator,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"MOSFit({self.scale.signature} @ "
            f"{self.scale.generator_cents:.1f}c, "
            f"err={self.error_cents:.2f}c, score={self.score:.2f})"
        )

    def to_dict(self) -> Dict[str, object]:
        d = self.scale.to_dict()
        d.update(
            error_cents=self.error_cents,
            max_error_cents=self.max_error_cents,
            rms_error_cents=self.rms_error_cents,
            coverage=self.coverage,
            score=self.score,
            n_targets=self.n_targets,
            n_merged=self.n_merged,
            n_unmatched_degrees=self.n_unmatched_degrees,
            offset=self.offset,
            chance_error_cents=self.chance_error_cents,
            improvement=self.improvement,
            evidence=self.evidence,
            is_underdetermined=self.is_underdetermined,
        )
        return d


def _offset_candidates(
    positions: np.ndarray,
    weights: np.ndarray,
    degrees: np.ndarray,
    n_anchors: Optional[int],
) -> np.ndarray:
    """Rotations of the scale worth trying against the data.

    A scale and its transpositions are the same scale, so comparing the data
    to degrees rooted at zero answers the wrong question: a stack of fifths
    reads as the pentatonic only once the pentatonic is allowed to sit where
    the data actually put it.

    Under the absolute-error objective the optimal rotation always places at
    least one target exactly on a degree, so the exact candidate set is
    ``{t_i - d_j}``.  That is ``len(targets) x len(degrees)`` rotations, which
    gets expensive for large scales, so ``n_anchors`` restricts ``t_i`` to the
    heaviest few targets -- the ones that dominate the objective anyway.
    ``None`` uses every target and is exact.

    The shortlist is cut on *weight alone*, and it is widened rather than split
    when the cut lands inside a run of equally heavy targets.  Both halves of
    that rule are load-bearing, and the second one is why this function is
    longer than its one line of intent.

    The old cut was ``np.argsort(weights)[::-1][:n]``, which looks like "the
    heaviest few" and is not: argsort is stable, so reversing it hands tied
    targets back in *reverse array order*, and the weights are uniform by
    default -- :func:`_clean_weights` returns ``1/n`` everywhere when none are
    supplied -- so under the default *every* tie was broken by where the caller
    happened to put the ratio.  Which rotations got tried, and therefore which
    scale won, was a function of the input order.

    Breaking those ties by the target's own position would fix that and break
    something else.  The origin of the position axis is arbitrary -- transpose
    the whole signal and every position slides -- so a rule that reads
    positions makes the fit depend on where the data sits rather than on its
    shape, and :func:`~biotuner.mos.metrics.mos_ness` really does report a
    different number for a signal shifted by 137 cents.  There is no third
    option: a tie between equal weights carries no information, so no function
    of the data can break it, and any rule that appears to is reading something
    that is not data.  The tie is therefore not broken at all -- every target
    in the tied run becomes an anchor.  With uniform weights that makes the
    scan exact, which is the honest reading of "restrict to the heaviest few"
    when none of them is heavier than another; the shortlist still does its job
    wherever the amplitudes genuinely single some peaks out.
    """
    if n_anchors is None or n_anchors >= len(positions):
        anchors = positions
    else:
        order = np.argsort(-weights, kind="stable")
        k = max(1, min(int(n_anchors), len(positions)))
        # ``-weights[order]`` ascends, so ``searchsorted`` finds where the run
        # the cut landed in ends.
        descending = -weights[order]
        k = int(np.searchsorted(descending, descending[k - 1], side="right"))
        anchors = positions[order[:k]]
    offsets = (anchors[:, None] - degrees[None, :]).ravel()
    # Zero is always included, so aligning can never score worse than not.
    offsets = np.concatenate([offsets, [0.0]])
    # The modulus first, then the rounding, then the modulus again: rounding an
    # offset a hair under the period up to exactly 1.0 would report the same
    # rotation as 0.0 under a different number and scan it twice.
    return np.unique(np.round(offsets % 1.0, 12) % 1.0)


def _evaluate(
    scale: MOSScale,
    positions: np.ndarray,
    weights: np.ndarray,
    tolerance_cents: float,
    complexity_penalty: float,
    align: bool = True,
    n_anchors: Optional[int] = 3,
) -> MOSFit:
    """Score one scale against one set of target positions."""
    degrees = np.asarray(scale.degrees, dtype=float)
    period_cents = scale.period_cents

    if align:
        offsets = _offset_candidates(positions, weights, degrees, n_anchors)
    else:
        offsets = np.zeros(1)

    # (offsets, targets, degrees), signed and wrapped into [-0.5, 0.5].
    diff = positions[None, :, None] - offsets[:, None, None] - degrees[None, None, :]
    diff = diff - np.round(diff)
    dist = np.abs(diff)
    nearest = np.argmin(dist, axis=2)
    picked = np.take_along_axis(dist, nearest[..., None], axis=2)[..., 0]
    per_offset = (weights[None, :] * picked).sum(axis=1)
    o = int(np.argmin(per_offset))

    offset = float(offsets[o])
    idx = nearest[o]
    rows = np.arange(len(positions))
    signed = diff[o][rows, idx] * period_cents
    abs_err = np.abs(signed)

    error = float(np.sum(weights * abs_err))
    rms = float(np.sqrt(np.sum(weights * abs_err**2)))
    coverage = float(np.sum(weights[abs_err <= tolerance_cents]))
    surplus = max(0, scale.cardinality - len(positions))
    score = error + complexity_penalty * surplus

    return MOSFit(
        scale=scale,
        error_cents=error,
        max_error_cents=float(abs_err.max()) if abs_err.size else 0.0,
        rms_error_cents=rms,
        coverage=coverage,
        score=score,
        assignments=tuple(int(i) for i in idx),
        residuals=tuple(float(s) for s in signed),
        n_targets=len(positions),
        offset=offset,
    )


def _refinement_bounds(scale: MOSScale) -> Optional[Tuple[float, float]]:
    """Generator fractions :func:`_refine_generator` may slide between.

    The signature's own tuning range, pulled in far enough at each end that the
    scale there still has :attr:`~MOSScale.cardinality` *arithmetically*
    distinct pitch classes -- its smallest step clears
    :data:`MIN_REFINED_STEP`, a hundredth of a cent at the octave.  ``None``
    when no such interval exists.

    That is deliberately not the audibility floor.  Whether a returned scale is
    musically real is :data:`MIN_AUDIBLE_STEP_CENTS`' job, enforced by
    :func:`_is_degenerate_scale` at every site a scale is built; this bound only
    has to stop the optimiser sliding onto the endpoint, and keeping it tight
    leaves the refiner as much room as it can safely have.

    A tuning range runs between two rationals, and exactly one of them -- the
    one whose denominator is smaller than the cardinality -- is a generator at
    which some multiples coincide, so the scale there has fewer pitch classes
    than degrees.  (The other endpoint's denominator *is* the cardinality: that
    is the scale's own equal division, where the steps are equal but the
    pitches are all different.  Checked over every signature to 24 notes: one
    collapsing end each, never two, never none.)

    How fast the collapse reopens on the way in is the endpoint's business, not
    a constant.  The smallest step grows as its denominator times the distance
    travelled, and that denominator runs from 1 to one less than the
    cardinality across the signatures, so a margin fixed at a millionth of the
    range -- which is what this used to be -- buys a step of anywhere between a
    millionth and a thousandth of the range.  For a wide range that is nothing:
    ``5L6s`` on five-tone equal input came back as an eleven-note scale whose
    pitch classes were 0.0002 cents apart, five of them wearing eleven names.
    The refinement had slid the generator onto ``4/5`` and parked there,
    because that is genuinely where the weighted error is smallest -- a scale
    that has collapsed onto the data fits it perfectly.

    So the bound is set from the thing it protects.  The margin starts at a
    millionth of the range (never below :data:`GENERATOR_EPSILON`, the point at
    which :func:`_fold_bright` and :func:`generator_candidates` stop telling two
    generators apart), and each end is then pushed in until the scale there
    clears :data:`MIN_REFINED_STEP`.  The rescale is a single step, not a search,
    because the step is exactly proportional to the distance from the endpoint
    and one measurement therefore fixes the constant; the passes after it only
    verify.  The equal-division end always clears the floor on the first probe
    and is left exactly where it was, so an equal-tempered input can still be
    fitted to the last decimal -- which is the case the bound had to not break.

    Examples
    --------
    ``5L2s`` may be tuned anywhere between 4/7 and 3/5 of the period.  Only one
    end of a range ever needs trimming -- the other is the equal division of
    the scale's own cardinality, where the steps are equal but the pitch
    classes are all different -- so seven-tone equal at 685.714 cents stays
    exactly reachable and only the 720-cent end gives up two thousandths of a
    cent:

    >>> lo, hi = _refinement_bounds(MOSScale(5, 2, 18 / 31, validate=False))
    >>> round(lo * 1200, 4), round(hi * 1200, 4)
    (685.7143, 719.9976)
    >>> bool(min(_min_step(MOSScale(5, 2, g, validate=False))
    ...          for g in (lo, hi)) >= MIN_REFINED_STEP)
    True

    The trimming is what stops a scale collapsing onto a smaller one.  At
    exactly 4/5 a ``5L6s`` is five pitch classes wearing eleven names, and a
    millionth of the range above it -- the old bound -- it still is, to any
    tolerance that means anything:

    >>> old_bound = 4 / 5 + (5 / 6 - 4 / 5) * 1e-6
    >>> round(_min_step(MOSScale(5, 6, old_bound, validate=False)) * 1200, 6)
    0.0002
    >>> lo, _ = _refinement_bounds(MOSScale(5, 6, 9 / 11, validate=False))
    >>> bool(lo > old_bound)
    True
    >>> round(_min_step(MOSScale(5, 6, lo, validate=False)) * 1200, 6)
    0.012
    """
    lo, hi = scale.tuning_range
    lo_f, hi_f = float(lo), float(hi)
    # Never below an epsilon: a bound the generator can stand on top of is not
    # a bound, and 1e-6 of a narrow range is smaller than the arithmetic noise
    # every fraction in this module carries.
    base = max((hi_f - lo_f) * 1e-6, GENERATOR_EPSILON)

    # The rescale below is exact in exact arithmetic and lands slightly short
    # in floating point: the step being measured is a difference between two
    # numbers of order one, so at 1e-7 of a period it carries several parts per
    # billion of cancellation error, and the extrapolation inherits it.  A
    # relative slack of a millionth swallows that and still pins the floor to
    # six figures.
    floor = MIN_REFINED_STEP * (1.0 - 1e-6)

    edges: List[float] = []
    for endpoint, inward in ((lo_f, 1.0), (hi_f, -1.0)):
        margin = base
        for _ in range(3):
            probe = MOSScale(
                scale.n_large, scale.n_small,
                endpoint + inward * margin, scale.period, validate=False,
            )
            step = _min_step(probe)
            if step >= floor:
                break
            if step <= 0.0:  # pragma: no cover - defensive; degrees coincide
                return None
            margin *= MIN_REFINED_STEP / step
        else:
            return None
        edges.append(endpoint + inward * margin)

    a, b = edges
    return (a, b) if a < b else None


def _refine_generator(
    scale: MOSScale,
    positions: np.ndarray,
    weights: np.ndarray,
    tolerance_cents: float,
    complexity_penalty: float,
    align: bool = True,
) -> MOSFit:
    """Slide the generator inside its valid range to minimise the error.

    Bounded so the signature cannot change: the scale that comes out is the
    same well-formed structure, optimally tuned to the signal.  This is exactly
    the freedom Milne et al. §2 call the scale's "valid tuning range", minus the
    neighbourhood of whichever end of it collapses the scale onto a smaller one
    -- see :func:`_refinement_bounds`, which is where that has to be measured
    rather than assumed.
    """
    original = _evaluate(scale, positions, weights, tolerance_cents,
                         complexity_penalty, align=align, n_anchors=None)
    bounds = _refinement_bounds(scale)
    if bounds is None:
        # No interval of this signature's range keeps every pitch class, so
        # there is nowhere safe to slide to.  A scale with fewer pitch classes
        # than degrees is never an improvement, however well it scores.
        return original
    lo_f, hi_f = bounds

    def objective(g: float) -> float:
        candidate = MOSScale(
            scale.n_large, scale.n_small, g, scale.period, validate=False
        )
        return _evaluate(
            candidate, positions, weights, tolerance_cents, complexity_penalty,
            align=align, n_anchors=None,
        ).score

    try:
        from scipy.optimize import minimize_scalar

        res = minimize_scalar(
            objective, bounds=(lo_f, hi_f), method="bounded",
            options={"xatol": 1e-9},
        )
        best_g = float(res.x)
    except Exception:  # pragma: no cover - scipy is a hard dep, but stay safe
        grid = np.linspace(lo_f, hi_f, 512)
        best_g = float(grid[int(np.argmin([objective(g) for g in grid]))])

    refined = MOSScale(
        scale.n_large, scale.n_small, best_g, scale.period, validate=False
    )
    if _is_degenerate_scale(refined):
        # The optimiser found the endpoint. A collapsed scale scores at least as
        # well as the scale it collapsed from -- it has spare degrees to place
        # wherever the data is -- so the objective alone will always prefer it,
        # and _refinement_bounds narrows the window but cannot guarantee the
        # minimiser stays off the edge to within its own tolerance. Whatever the
        # score says, a 4L3s whose small step is a hundredth of a cent is not a
        # 4L3s, so the unrefined scale stands.
        return original
    best = _evaluate(refined, positions, weights, tolerance_cents,
                     complexity_penalty, align=align, n_anchors=None)
    return best if best.score <= original.score else original


def fit_mos(
    ratios: Sequence[float],
    weights: Optional[Sequence[float]] = None,
    period: float = 2.0,
    min_cardinality: int = 4,
    max_cardinality: int = 24,
    tolerance_cents: float = 15.0,
    complexity_penalty: float = 1.0,
    grid: int = 720,
    include_intervals: bool = True,
    refine: bool = True,
    n_refine: int = 12,
    align: bool = True,
    n_anchors: int = 3,
    top_n: int = 5,
    candidates: Optional[Sequence[float]] = None,
    optimize_period: bool = False,
    period_bounds: Tuple[float, float] = (1.8, 2.2),
    period_steps: int = 21,
    fold: bool = True,
) -> List[MOSFit]:
    """Rank the well-formed scales that best explain a set of ratios.

    Parameters
    ----------
    ratios : sequence of float
        Frequency ratios to explain -- typically ``bt.peaks_ratios``.
    weights : sequence of float, optional
        Per-ratio importance, e.g. peak amplitudes.  Normalised internally.
        Uniform when omitted.
    period : float, default 2.0
        Period as a frequency ratio.  Ignored when ``optimize_period``.
    min_cardinality, max_cardinality : int
        Ring range to search.  The default upper bound of 24 keeps scales in
        playable territory; raise it to explore microtonal ones.
    tolerance_cents : float, default 15.0
        What counts as a hit, for :attr:`MOSFit.coverage`.  Does not affect the
        error or the ranking.
    complexity_penalty : float, default 1.0
        Cents of penalty per scale degree beyond the number of targets.  A
        larger scale can always cover more ratios; without this, the search
        just returns the biggest one allowed.  Set to ``0`` for pure error
        ranking.

        The default is calibrated, not guessed.  Fitting fourteen known MOS
        scales -- first exactly, then from five jittered peaks at 8 cents SD --
        recovers the true signature in 14/14 and 12/14 cases at ``1.0``, versus
        8/14 and 2/14 at ``0.0`` (where the search overfits to a median of 21
        notes).  Raising it to 3.0 costs recovery (10/14) but returns smaller
        scales on unstructured input, which is the trade to make if you care
        more about parsimony than about identification.
    grid : int, default 720
        Background generator-grid resolution.  ``0`` uses only signal-derived
        candidates.
    include_intervals : bool, default True
        Also try every interval *between* observed ratios as a generator.
    refine : bool, default True
        Slide the top ``n_refine`` fits' generators inside their valid ranges to
        minimise the error.  The signature never changes, only the tuning.
    n_refine, top_n : int
    align : bool, default True
        Let each candidate scale transpose itself onto the data.  A scale and
        its transpositions are the same scale, so this is the right comparison:
        without it a stack of fifths does not read as the pentatonic, because
        the pentatonic is only rooted correctly in one of its five modes.  Set
        ``False`` to pin every candidate to a root of 1/1.
    n_anchors : int, default 3
        How many of the heaviest targets seed candidate transpositions during
        the coarse scan.  The exact set uses every target, which is
        ``len(targets) x cardinality`` rotations per candidate and gets slow
        for large scales; the shortlist is always re-scored exactly during
        refinement.

        A cut landing inside a run of equally heavy targets takes the whole
        run, so with ``weights=None`` -- where every target weighs the same --
        the coarse scan is exact whatever this is set to.  A tie between equal
        weights says nothing about which target matters more, and any rule that
        broke it would be reading the input's order or its transposition rather
        than the input; see :func:`_offset_candidates`.
    candidates : sequence of float, optional
        Explicit generator fractions to try instead of deriving them.
    optimize_period : bool, default False
        Also fit the pseudo-octave, over ``period_steps`` values spanning
        ``period_bounds``.  Slower by that factor.
    period_bounds, period_steps
        Only used when ``optimize_period``.
    fold : bool, default True
        Reduce the ratios to distinct pitch classes before fitting: fold each
        into ``[1, period)`` and merge anything landing within
        :data:`FOLD_TOLERANCE_CENTS` of a pitch class already seen, summing the
        weights rather than dropping them.

        A scale cannot tell ``1/1`` from ``2/1``, so counting both as targets
        double-counts the unison, inflates ``n_targets``, and drags the error
        toward whatever the unison happens to do.  Most of biotuner's tuning
        derivations emit a ratio at exactly 1/1 or 2/1, so this is the common
        case rather than an edge case.  Set ``False`` to score every ratio as
        handed in.

    Returns
    -------
    list of MOSFit
        Best first, by :attr:`MOSFit.score`.  At most one fit per
        (signature, period) so the list is not filled with near-duplicates.

    Notes
    -----
    ``ratios`` is read as a *multiset*.  Permuting it -- carrying ``weights``
    along -- returns the same ranked list, in the same order, with bit-identical
    signatures, generators, scores, errors, coverages and offsets.  The only
    thing that moves is the three per-target vectors
    (:attr:`~MOSFit.targets`, :attr:`~MOSFit.assignments`,
    :attr:`~MOSFit.residuals`), which are defined to run parallel to the input
    and are permuted with it.  See :func:`_prepare_targets` for why that took
    canonicalising the targets rather than fixing the three places that read
    them in order.

    Examples
    --------
    A scale that *is* an MOS is recovered exactly -- generator, signature and
    all:

    >>> d = MOSScale.from_signature(5, 2, tuning=31)
    >>> fit = fit_mos(d.ratios, max_cardinality=12)[0]
    >>> fit.signature
    '5L2s'
    >>> round(fit.error_cents, 6)
    0.0
    >>> round(fit.scale.generator_cents, 2)
    696.77

    Twelve-tone equal temperament is recognised as the chromatic MOS at a
    700-cent generator.  It is a *degenerate* well-formed scale -- its two step
    sizes are identical (Milne et al. §2, footnote 6) -- so ``7L5s`` and its
    inverse ``5L7s`` describe it equally well, and the tie-break picks one:

    >>> edo12 = [2 ** (k / 12) for k in range(12)]
    >>> fit = fit_mos(edo12, max_cardinality=12)[0]
    >>> fit.signature, round(fit.scale.generator_cents, 3)
    ('7L5s', 700.0)
    >>> fit.scale.is_degenerate
    True

    A tuning that runs from the unison to the octave states the same pitch
    class at both ends.  Folding counts it once:

    >>> ladder = [1.0, 1.125, 1.25, 1.5, 2.0]
    >>> fit = fit_mos(ladder, max_cardinality=12)[0]
    >>> fit.n_targets, fit.n_merged
    (4, 1)
    >>> fit_mos(ladder, fold=False, max_cardinality=12)[0].n_targets
    5
    """
    if min_cardinality < 3:
        raise ValueError(
            f"min_cardinality must be at least 3 (a 2-note MOS is a bare "
            f"generator), got {min_cardinality}"
        )
    if max_cardinality < min_cardinality:
        raise ValueError(
            f"max_cardinality ({max_cardinality}) is below min_cardinality "
            f"({min_cardinality})"
        )

    periods: Sequence[float]
    if optimize_period:
        lo, hi = period_bounds
        if not 1.0 < lo <= hi:
            raise ValueError(
                f"period_bounds must satisfy 1 < lo <= hi, got {period_bounds!r}"
            )
        periods = np.linspace(lo, hi, period_steps)
    else:
        periods = [period]

    best_by_key: Dict[Tuple[int, int, int], MOSFit] = {}
    # Folding is period-relative, so the prepared targets are cached per period
    # rather than recomputed: the refinement pass has to score against exactly
    # the targets the coarse scan used, or the two rankings are not comparable.
    prepared: Dict[
        int, Tuple[np.ndarray, np.ndarray, int, Tuple[float, ...], np.ndarray]
    ] = {}

    for per in periods:
        per = float(per)
        pkey = int(round(per * 1e6))
        positions, w, n_merged, targets, order = _prepare_targets(
            ratios, weights, per, fold=fold
        )
        prepared[pkey] = (positions, w, n_merged, targets, order)

        cands = (
            list(candidates)
            if candidates is not None
            # Derived from the folded targets: a duplicated pitch class adds no
            # generator the original does not already propose, and the interval
            # between a ratio and its own octave is not a generator at all.
            else generator_candidates(
                targets, per, include_intervals=include_intervals, grid=grid
            )
        )
        for g in cands:
            if not 0.0 < g < 1.0:
                continue
            for card, n_large, n_small in T.mos_series(
                g, max_cardinality=max_cardinality, include_trivial=True
            ):
                if card < min_cardinality:
                    continue
                scale = MOSScale(n_large, n_small, g, per, validate=False)
                if _is_degenerate_scale(scale):
                    # Fewer pitch classes than degrees: whatever proposed it,
                    # the note count is a fiction and every quantity derived
                    # from the cardinality would be too.
                    continue
                fit = _evaluate(
                    scale, positions, w, tolerance_cents, complexity_penalty,
                    align=align, n_anchors=n_anchors,
                )
                key = (n_large, n_small, pkey)
                prev = best_by_key.get(key)
                if prev is None or fit._rank_key < prev._rank_key:
                    best_by_key[key] = fit

    ranked = sorted(best_by_key.values(), key=lambda f: f._rank_key)
    if refine and ranked:
        head = ranked[:n_refine]
        tail = ranked[n_refine:]
        refined = []
        for fit in head:
            positions, w, _, _, _ = prepared[int(round(fit.scale.period * 1e6))]
            refined.append(
                _refine_generator(
                    fit.scale, positions, w, tolerance_cents, complexity_penalty,
                    align=align,
                )
            )
        ranked = sorted(refined + tail, key=lambda f: f._rank_key)

    out = []
    for fit in ranked[:top_n]:
        _, _, n_merged, targets, order = prepared[
            int(round(fit.scale.period * 1e6))
        ]
        out.append(_in_caller_order(fit, order, n_merged, targets))
    return out


# --------------------------------------------------------------------------- #
# The whole labyrinth, scored
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FitField:
    """Every point of the labyrinth scored against one set of ratios.

    :func:`fit_mos` answers "which scale is this?" with a ranked shortlist,
    which hides how the answer sits among its neighbours. A signal is often
    compatible with several *disconnected* regions of the labyrinth, and a
    winner reported without that context reads as more decisive than it is.
    This is the same objective evaluated everywhere instead.

    Attributes
    ----------
    errors : np.ndarray, shape (max_cardinality + 1, len(generators))
        Weighted mean cents error per (cardinality, generator) cell. ``NaN``
        where the generator admits no MOS at that cardinality -- most of the
        plane, since well-formedness is rare.
    generators : np.ndarray
        Generator fractions, one per column.
    cardinalities : np.ndarray
        Row index, ``0 .. max_cardinality``. Rows 0 and 1 are always empty.
    period : float
    n_targets : int
    ratios : tuple of float
        The ratios this field was scored against.  Carried along so a figure
        drawn from a precomputed field can still mark them; without it the
        overlay silently disappears exactly when you reuse a field, which is
        the case it exists for.
    """

    errors: np.ndarray
    generators: np.ndarray
    cardinalities: np.ndarray
    period: float
    n_targets: int
    ratios: Tuple[float, ...] = ()

    @property
    def period_cents(self) -> float:
        return T.PERIOD_CENTS * math.log2(self.period)

    @property
    def coverage(self) -> float:
        """Fraction of cells that contain a well-formed scale at all."""
        return float(np.isfinite(self.errors).mean())

    def chance_error(self, cardinality: int) -> float:
        """Error a random set of ratios would get against a scale this size."""
        return self.period_cents / (4.0 * max(1, cardinality))

    def best(self) -> Dict[str, float]:
        """The single lowest-error cell, with its coordinates.

        No parsimony penalty is applied, so this is the raw best fit and will
        usually name a larger scale than :func:`fit_mos` does. That difference
        is the penalty doing its job, not a disagreement.
        """
        if not np.isfinite(self.errors).any():
            raise ValueError("no cell in this field contains a scale")
        flat = int(np.nanargmin(self.errors))
        row, col = np.unravel_index(flat, self.errors.shape)
        return {
            "cardinality": int(self.cardinalities[row]),
            "generator": float(self.generators[col]),
            "generator_cents": float(self.generators[col]) * self.period_cents,
            "error_cents": float(self.errors[row, col]),
        }

    def islands(self, threshold_cents: float = 3.0) -> int:
        """How many separate low-error regions there are.

        Counts connected components of cells under ``threshold_cents``, with
        the generator axis treated as circular because the labyrinth is. More
        than one means the signal genuinely fits in several unrelated places,
        which a single best-fit answer cannot tell you.
        """
        mask = np.isfinite(self.errors) & (self.errors <= threshold_cents)
        if not mask.any():
            return 0
        seen = np.zeros_like(mask, dtype=bool)
        n_rows, n_cols = mask.shape
        count = 0
        for start in zip(*np.nonzero(mask)):
            if seen[start]:
                continue
            count += 1
            stack = [start]
            seen[start] = True
            while stack:
                r, c = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, (c + dc) % n_cols  # wrap: the axis is a circle
                    if 0 <= nr < n_rows and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
        return count


def fit_field(
    ratios: Sequence[float],
    weights: Optional[Sequence[float]] = None,
    *,
    period: float = 2.0,
    max_cardinality: int = 24,
    resolution: int = 720,
    min_cardinality: int = 3,
    align: bool = True,
    n_anchors: int = 3,
) -> FitField:
    """Score every (generator, cardinality) in the labyrinth against some ratios.

    Parameters
    ----------
    ratios : sequence of float
        Frequency ratios to explain, e.g. ``bt.peaks_ratios``.
    weights : sequence of float, optional
        Per-ratio importance, normalised internally.
    period : float, default 2.0
    max_cardinality : int, default 24
        Outermost ring to score.
    resolution : int, default 720
        Generator samples across the full period. Cost is roughly linear in
        this and in the number of MOS cardinalities each generator admits.
    min_cardinality : int, default 3
    align : bool, default True
        Let each candidate transpose onto the data, as :func:`fit_mos` does. A
        scale and its transpositions are the same scale, so leaving this off
        answers a different and less useful question.
    n_anchors : int, default 3
        Transpositions are seeded from this many of the heaviest targets, and
        from all of them where the weights tie -- which, unweighted, is always.
        See :func:`_offset_candidates`.

    Returns
    -------
    FitField

    Notes
    -----
    The generator axis is *sampled*, not optimised. A rational generator such
    as ``9/19`` will not sit exactly on a uniform grid, so even a scale scored
    against itself lands slightly off zero here. :func:`fit_mos` refines within
    the valid range and does reach zero; this function draws the landscape that
    refinement happens inside, and the two should not be expected to agree to
    the last cent.

    Examples
    --------
    >>> ref = MOSScale.from_signature(4, 3, tuning=19)
    >>> field = fit_field(ref.ratios, max_cardinality=12, resolution=360)
    >>> field.errors.shape
    (13, 359)

    Most of the plane holds no well-formed scale at all:

    >>> bool(field.coverage < 0.6)
    True

    The best cell is close, but not exact -- see the note above:

    >>> bool(0.0 < field.best()["error_cents"] < 3.0)
    True
    """
    if resolution < 8:
        raise ValueError(f"resolution must be at least 8, got {resolution}")
    if max_cardinality < min_cardinality:
        raise ValueError(
            f"max_cardinality ({max_cardinality}) is below min_cardinality "
            f"({min_cardinality})"
        )
    positions = _as_positions(ratios, period)
    w = _clean_weights(weights, len(positions))
    # Canonical order, for the same reason :func:`_prepare_targets` imposes one:
    # the field is scored against a *set* of ratios, and both which rotations
    # get tried and the last bits of the weighted sum read the array in order.
    # Nothing here is parallel to the caller's list, so no permutation has to
    # be carried back.
    canonical = np.argsort(positions, kind="stable")
    positions, w = positions[canonical], w[canonical]

    grid = np.linspace(0.0, 1.0, resolution, endpoint=False)[1:]
    errors = np.full((max_cardinality + 1, len(grid)), np.nan)

    for col, g in enumerate(grid):
        for card, n_large, n_small in T.mos_series(
            float(g), max_cardinality=max_cardinality, include_trivial=True
        ):
            if card < min_cardinality:
                continue
            scale = MOSScale(n_large, n_small, float(g), period, validate=False)
            fit = _evaluate(scale, positions, w, 15.0, 0.0,
                            align=align, n_anchors=n_anchors)
            errors[card, col] = fit.error_cents

    return FitField(
        errors=errors,
        generators=grid,
        cardinalities=np.arange(max_cardinality + 1),
        period=period,
        n_targets=len(positions),
        ratios=tuple(float(r) for r in ratios),
    )


def best_mos(ratios: Sequence[float], **kwargs) -> MOSFit:
    """The single best-fitting well-formed scale.

    Raises
    ------
    ValueError
        If no MOS could be fitted at all -- which happens only when the search
        range excludes every cardinality.

    Examples
    --------
    >>> best_mos(MOSScale.from_signature(4, 3, tuning=19).ratios).signature
    '4L3s'
    """
    kwargs.setdefault("top_n", 1)
    fits = fit_mos(ratios, **kwargs)
    if not fits:
        raise ValueError(
            "no MOS scale could be fitted; widen "
            "min_cardinality/max_cardinality or supply candidates"
        )
    return fits[0]


def mos_tuning(ratios: Sequence[float], **kwargs) -> List[float]:
    """The best-fitting MOS as a plain list of frequency ratios.

    Drop-in for anywhere biotuner expects a tuning.

    Examples
    --------
    >>> ref = MOSScale.from_signature(4, 3, tuning=19)
    >>> tuning = mos_tuning(ref.ratios, max_cardinality=12)
    >>> len(tuning)
    7
    >>> max(abs(a - b) for a, b in zip(tuning, ref.ratios)) < 1e-9
    True
    """
    return list(best_mos(ratios, **kwargs).aligned_ratios)


# --------------------------------------------------------------------------- #
# The forward direction: an observed interval, taken as the generator
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ForwardScale:
    """A scale read *forward* from an interval the signal actually states.

    :func:`fit_mos` treats the generator as latent: it searches for whichever
    value best explains the peaks, and that value need not sit between any two
    of them.  This is the opposite reading.  An interval that is present in the
    data -- the quotient of two peaks, or a peak ratio itself -- is *declared*
    the generator, stacked, and folded into the period.  Nothing is optimised,
    so this is not a best fit to anything; it is the consequence of taking one
    audible interval seriously.

    What makes the two directions comparable is :attr:`fit`.  The scale that
    comes out is scored against the *whole* target set by the same machinery
    the inverse search uses, with the same objective and the same freedom to
    transpose onto the data, so :attr:`error_cents`, :attr:`coverage` and
    :attr:`evidence` mean exactly what they mean on a :class:`MOSFit`.  A
    forward reading that beats the inverse fit is a genuine finding -- the
    latent generator turned out to be audible after all.  One that loses is the
    ordinary case, and the gap is the price of insisting the generator be an
    interval you can point at.

    Attributes
    ----------
    fit : MOSFit
        The scale scored against every target.  ``fit.scale.generator`` is
        exactly the observed interval folded into the bright half, never a
        refined value: refining it would destroy the one property that defines
        a forward reading.
    sources : tuple of (float, float)
        Every observed ``(numerator, denominator)`` pair that proposed a
        generator inside this reading's de-duplication window.  More than one
        means several independent peak pairs state the same interval, which is
        stronger evidence for it than a single coincidence -- see
        :attr:`n_sources`.  A raw ratio taken as a generator in its own right
        is recorded as ``(ratio, 1.0)``, since that is the interval it forms
        with the reference.

        The first entry is the *representative*: the pair whose folded quotient
        is the generator actually used.  The rest follow in ascending order.
        Neither the choice nor the order depends on how the caller happened to
        order the input -- see :func:`forward_scales`.

    Notes
    -----
    Every public readout of :attr:`fit` is re-exported here under the same
    name, so a caller holding a ``ForwardScale`` never has to reach through to
    the fit to get at :attr:`residuals`, :attr:`offset` or :attr:`improvement`.
    The facade is complete on purpose: a partial one would imply the missing
    quantities mean something different in this direction, and they do not.
    """

    fit: MOSFit
    sources: Tuple[Tuple[float, float], ...]

    # -- which interval this came from ------------------------------------- #
    @property
    def interval_pair(self) -> Tuple[float, float]:
        """The ``(numerator, denominator)`` this reading is built on.

        One of the observed pairs, not an average of the group: the whole point
        is that the generator is an interval the signal states, and the mean of
        several near-identical quotients is not one of them.  It is the pair
        :func:`forward_scales` elected to represent the window -- the one whose
        scale scores best -- so :attr:`generator` is this quotient folded, to
        the last bit.
        """
        return self.sources[0]

    @property
    def interval(self) -> float:
        """The observed quotient, as a frequency ratio, before folding.

        Reported unfolded, so a peak pair spanning more than a period still
        reads as what it is: ``22.91 / 10.07`` is 2.275, which the generator
        (976.9 cents) no longer shows.
        """
        num, den = self.interval_pair
        return num / den

    @property
    def n_sources(self) -> int:
        """How many observed intervals proposed this generator."""
        return len(self.sources)

    # -- the scale --------------------------------------------------------- #
    @property
    def scale(self) -> MOSScale:
        return self.fit.scale

    @property
    def signature(self) -> str:
        return self.fit.scale.signature

    @property
    def generator(self) -> float:
        """The generator as a period fraction, always in ``(0.5, 1)``."""
        return self.fit.scale.generator

    @property
    def generator_cents(self) -> float:
        return self.fit.scale.generator_cents

    @property
    def generator_ratio(self) -> float:
        """The generator as a frequency ratio, always above ``sqrt(period)``.

        Not simply :attr:`interval` reduced into the period.  Folding keeps the
        bright spelling, so an observed quotient landing in the *dark* half
        comes back **inverted**: 19.31/15.64 is 1.2347, which reduces to 1.2347
        at 365 cents, and the number reported here is its complement
        ``2 / 1.2347 = 1.6199`` at 835 cents.  The two build the same scale
        (Milne et al. §4), which is why only one is quoted -- but the one
        quoted is then an interval the signal states upside down, and reading
        it as the audible interval would be wrong.  For that, use
        :attr:`interval`.

        Examples
        --------
        >>> reading = next(
        ...     r for r in forward_scales([10.07, 15.64, 19.31, 22.91],
        ...                               include_ratios=False,
        ...                               min_cardinality=7, max_cardinality=7)
        ...     if r.interval_pair == (19.31, 15.64)
        ... )
        >>> round(reading.interval, 4), round(reading.generator_ratio, 4)
        (1.2347, 1.6199)
        """
        return self.fit.scale.generator_ratio

    # -- how well it explains the input ------------------------------------ #
    #
    # The facade over :attr:`fit` is deliberately *complete*: every public
    # readout a :class:`MOSFit` offers is reachable from a ``ForwardScale``
    # under the same name.  A partial facade is the worse option of the two --
    # it reads as a statement that the missing quantities do not apply to a
    # forward reading, when in fact they are computed by exactly the same
    # evaluation and mean exactly the same thing.  ``.fit`` remains public for
    # anyone who wants the object itself.
    @property
    def error_cents(self) -> float:
        return self.fit.error_cents

    @property
    def max_error_cents(self) -> float:
        return self.fit.max_error_cents

    @property
    def rms_error_cents(self) -> float:
        return self.fit.rms_error_cents

    @property
    def coverage(self) -> float:
        return self.fit.coverage

    @property
    def score(self) -> float:
        return self.fit.score

    @property
    def assignments(self) -> Tuple[int, ...]:
        return self.fit.assignments

    @property
    def residuals(self) -> Tuple[float, ...]:
        return self.fit.residuals

    @property
    def n_targets(self) -> int:
        return self.fit.n_targets

    @property
    def offset(self) -> float:
        return self.fit.offset

    @property
    def n_merged(self) -> int:
        return self.fit.n_merged

    @property
    def targets(self) -> Tuple[float, ...]:
        return self.fit.targets

    @property
    def n_unmatched_degrees(self) -> int:
        return self.fit.n_unmatched_degrees

    @property
    def aligned_degrees(self) -> List[float]:
        return self.fit.aligned_degrees

    @property
    def aligned_ratios(self) -> List[float]:
        return self.fit.aligned_ratios

    @property
    def aligned_cents(self) -> List[float]:
        return self.fit.aligned_cents

    @property
    def mode(self):
        return self.fit.mode

    @property
    def chance_error_cents(self) -> float:
        return self.fit.chance_error_cents

    @property
    def improvement(self) -> float:
        return self.fit.improvement

    @property
    def evidence(self) -> float:
        return self.fit.evidence

    @property
    def is_underdetermined(self) -> bool:
        return self.fit.is_underdetermined

    @property
    def _rank_key(self) -> Tuple[float, ...]:
        """Score first, then agreement, then the structural tie-break.

        The leading element is :attr:`MOSFit._rank_key`'s quantised score, so a
        forward reading and an inverse fit are ordered on the same quantity.
        :attr:`n_sources` enters only as a tie-break: an interval three peak
        pairs agree on is better corroborated than one seen once, but
        corroboration is not explanatory power, and letting it outrank the
        score would promote a well-attested interval that fits the data badly.
        """
        base = self.fit._rank_key
        return (base[0], -self.n_sources) + base[1:]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        num, den = self.interval_pair
        return (
            f"ForwardScale({num:.6g}/{den:.6g} = {self.interval:.4f} -> "
            f"{self.generator_cents:.1f}c, {self.signature}, "
            f"err={self.error_cents:.2f}c, n={self.n_sources})"
        )

    def to_dict(self) -> Dict[str, object]:
        """Flat summary -- the fit's own fields plus where the generator came from."""
        num, den = self.interval_pair
        d = self.fit.to_dict()
        d.update(
            interval=self.interval,
            interval_numerator=num,
            interval_denominator=den,
            n_sources=self.n_sources,
        )
        return d


def _observed_intervals(
    ratios: Sequence[float],
    include_ratios: bool,
    include_intervals: bool,
) -> List[Tuple[float, float]]:
    """Every interval a list of ratios actually states, as numerator/denominator.

    Two sources, in this order:

    1. Each ratio read against the reference, ``(r, 1.0)`` -- the interval a
       peak forms with the fundamental the ratios were measured from.
    2. Each unordered *pair*, larger over smaller, so the quotient exceeds 1
       and reads as an interval rather than as its own inversion.  Both
       orderings fold to the same bright generator anyway, so enumerating one
       of them is not a loss.

    Duplicates are dropped by identity of the pair, which matters because
    biotuner's tunings routinely contain 1/1: without it every raw ratio would
    be counted a second time as its pair against the unison, and the
    ``n_sources`` tally would say "two peak pairs agree" about one observation.

    Identity is judged on the pair rounded to twelve decimals, so two ratios
    that differ by an ulp collide.  When they do, the survivor is the
    canonically smallest of them rather than whichever the caller happened to
    list first -- otherwise the retained pair, and therefore the generator it
    proposes, would depend on the order of a set.
    """
    r = np.asarray(ratios, dtype=float).ravel()
    r = r[np.isfinite(r) & (r > 0)]
    out: List[Tuple[float, float]] = []
    seen: set = set()
    index: Dict[Tuple[float, float], int] = {}

    def add(num: float, den: float) -> None:
        key = (round(num, 12), round(den, 12))
        if key in seen:
            # Same pair to twelve decimals. Keep the canonically smaller of the
            # two so the choice cannot depend on arrival order.
            at = index[key]
            if (num, den) < out[at]:
                out[at] = (float(num), float(den))
            return
        seen.add(key)
        index[key] = len(out)
        out.append((float(num), float(den)))

    if include_ratios:
        for value in r:
            add(float(value), 1.0)
    if include_intervals:
        for i in range(r.size):
            for j in range(i):
                a, b = float(r[i]), float(r[j])
                add(max(a, b), min(a, b))
    return out


def forward_scales(
    ratios: Sequence[float],
    weights: Optional[Sequence[float]] = None,
    period: float = 2.0,
    min_cardinality: int = 4,
    max_cardinality: int = 24,
    tolerance_cents: float = 15.0,
    complexity_penalty: float = 1.0,
    include_ratios: bool = True,
    include_intervals: bool = True,
    dedupe_cents: float = 1.0,
    align: bool = True,
    n_anchors: Optional[int] = None,
    fold: bool = True,
    top_n: Optional[int] = None,
) -> List[ForwardScale]:
    """Stack each interval the signal states, and see what scale it builds.

    The forward direction.  Where :func:`fit_mos` searches for a latent
    generator, this one refuses to invent anything: every generator it tries is
    an interval already present in ``ratios``.  For each it enumerates the
    cardinalities at which stacking is well-formed at all, builds the scale
    there, and scores it against the *whole* input with the same objective the
    inverse search uses -- so the two answers can be laid side by side.

    Parameters
    ----------
    ratios : sequence of float
        The observed values.  Frequency ratios such as ``bt.peaks_ratios``, or
        raw peak frequencies -- quotients of frequencies are ratios, so the
        pairwise part works either way.  With raw frequencies set
        ``include_ratios=False``; ``19.31`` is a frequency, not an interval,
        and reading it as a generator means nothing.
    weights : sequence of float, optional
        Per-ratio importance for the *scoring*, e.g. peak amplitudes.  It does
        not influence which generators are proposed: an interval is either
        stated by the signal or it is not.
    period : float, default 2.0
    min_cardinality, max_cardinality : int
        Ring range to enumerate, as in :func:`fit_mos`.  A generator supports
        only its own MOS cardinalities, so most rings produce nothing.
    tolerance_cents : float, default 15.0
        What counts as a hit, for :attr:`ForwardScale.coverage`.
    complexity_penalty : float, default 1.0
        Cents charged per degree beyond the number of targets.  Needed for the
        same reason as in :func:`fit_mos`, and needed *identically* if the two
        directions' scores are to be compared -- stacking an observed interval
        far enough will eventually cover everything by brute force.
    include_ratios : bool, default True
        Also read each ratio itself as a generator, not only the quotients
        between pairs.
    include_intervals : bool, default True
        Use the quotient of every pair of ratios.  Turning this off leaves only
        the ratios themselves.
    dedupe_cents : float, default 1.0
        Width of the window inside which several proposed generators are
        treated as one reading.  The window is represented by whichever of its
        proposals scores best, and every pair that proposed into it survives in
        :attr:`ForwardScale.sources`, so the corroboration count
        :attr:`ForwardScale.n_sources` is unaffected by the choice.  Matches
        :func:`generator_candidates`'s own threshold, so the two directions
        resolve the generator axis equally finely.
    align : bool, default True
        Let each scale transpose onto the data before scoring, exactly as
        :func:`fit_mos` does.  A scale and its transpositions are one scale.
    n_anchors : int, optional
        Targets seeding candidate transpositions.  ``None`` -- the default --
        uses every target, which is the exact optimum.  :func:`fit_mos` scans
        with a shortlist because it evaluates thousands of candidates and
        re-scores the survivors exactly; there are far fewer readings here, so
        the exact rotation is affordable from the start and the numbers need no
        caveat when set beside a refined inverse fit.
    fold : bool, default True
        Reduce the input to distinct pitch classes before scoring, as
        :func:`fit_mos` does.  Applies to the *targets* only; the proposed
        intervals are read off the ratios as given, so a peak pair spanning two
        octaves still proposes the interval it spans.
    top_n : int, optional
        Truncate the ranking.  ``None`` returns every reading.

    Returns
    -------
    list of ForwardScale
        Best first.  Empty when the signal states no interval capable of
        generating a scale -- a list of pure octaves, for instance, whose only
        quotients fold to nothing.  That is an answer, not a failure, so it is
        returned rather than raised.

    Notes
    -----
    Ranked by :attr:`ForwardScale.score` -- the same weighted error plus
    surplus-note penalty that ranks :func:`fit_mos` -- because the question a
    reader asks of this list is "which observed interval, used as a generator,
    accounts for the signal best?", and only a quantity the inverse fit also
    reports lets that be answered against the alternative.  Ties break on
    :attr:`ForwardScale.n_sources` (more corroboration first), then on the
    structural order :attr:`MOSFit._rank_key` uses.

    The generator is never refined.  :func:`fit_mos` slides its winners inside
    their valid tuning ranges to shave off cents; doing that here would replace
    the observed interval with a nearby unobserved one and quietly turn a
    forward reading back into an inverse fit.

    ``n_sources`` counts proposals, not distinct pitch classes.  Two input
    ratios an octave apart name one pitch class but are two proposals, so they
    corroborate a generator twice.  With ``fold=True`` the *targets* have
    already been merged, so this affects the tie-break only, never the score.

    The *answer* depends on the input only as a multiset: permuting ``ratios``
    (carrying ``weights`` along with it) returns the same readings, in the same
    order, with bit-identical generators, signatures, scores, errors and
    ``sources``.  Only the per-target vectors follow the caller's list, because
    they are defined to -- :attr:`~MOSFit.targets`,
    :attr:`~MOSFit.assignments` and :attr:`~MOSFit.residuals` run parallel to
    the input and are permuted with it.

    That invariance is not free, and it is not local.  Two separate order
    dependencies had to go.  *Proposals* arrive in whatever order the caller's
    list dictates, and grouping them greedily in that order lets the first
    arrival define its window and speak for it, so reversing four ratios could
    swap a ``7L5s`` at 699.75 cents for a ``5L7s`` at 700.25 -- a different
    generator, a different error, and a signature flipped to its inverse, from
    the same numbers.  Proposals are therefore sorted before the windows are
    cut, and each window is represented by its best-scoring member rather than
    by its first.  *Targets* had the same problem one level down, in machinery
    both directions share: which pitch class survived a merge, which targets
    seeded the rotation search, and the order the weighted errors were summed
    in all read the array as the caller filled it.
    :func:`_prepare_targets` sorts once, so all three are answered by the data.

    Examples
    --------
    A stack of fifths states the fifth outright, so the forward reading finds
    it without searching, and the pentatonic falls out exactly:

    >>> stack = [1.0, 1.125, 1.265625, 1.5]
    >>> top = forward_scales(stack, max_cardinality=12)[0]
    >>> top.signature, round(top.generator_cents, 3)
    ('2L3s', 701.955)
    >>> round(top.error_cents, 9), top.interval_pair
    (0.0, (1.5, 1.0))

    Two intervals proposed that generator -- 3/2 against the root, and 3/2
    against 9/8, which is 4/3 and folds to the same bright half:

    >>> top.n_sources
    2

    Real EEG (S001, eyes closed), four alpha-band peaks in Hz.  Each pair is
    taken as a generator in turn; printed here at the smallest scale each one
    supports, which is not the ranking but is the readable way to see six
    different answers at once:

    >>> peaks = [10.07, 15.64, 19.31, 22.91]
    >>> readings = forward_scales(peaks, include_ratios=False,
    ...                           min_cardinality=5, max_cardinality=7)
    >>> smallest = {}
    >>> for r in readings:
    ...     key = r.interval_pair
    ...     if (key not in smallest
    ...             or r.scale.cardinality < smallest[key].scale.cardinality):
    ...         smallest[key] = r
    >>> for r in sorted(smallest.values(), key=lambda r: r.generator_cents):
    ...     print(f"{r.interval_pair[0]:6.2f}/{r.interval_pair[1]:5.2f} = "
    ...           f"{r.interval:.3f} -> generator {r.generator_cents:6.1f} c"
    ...           f" -> {r.signature} ({r.scale.cardinality} notes)")
     22.91/15.64 = 1.465 -> generator  660.9 c -> 2L3s (5 notes)
     15.64/10.07 = 1.553 -> generator  762.2 c -> 3L2s (5 notes)
     19.31/15.64 = 1.235 -> generator  835.1 c -> 3L4s (7 notes)
     22.91/19.31 = 1.186 -> generator  904.0 c -> 4L1s (5 notes)
     22.91/10.07 = 2.275 -> generator  976.9 c -> 1L4s (5 notes)
     19.31/10.07 = 1.918 -> generator 1127.1 c -> 1L4s (5 notes)

    None of those is the generator the inverse search settles on, which is the
    finding: this signal's best latent explanation is an interval no pair of
    its peaks states.

    >>> inverse = fit_mos(peaks, max_cardinality=6)[0]
    >>> inverse.signature, round(inverse.scale.generator_cents, 2)
    ('1L3s', 930.44)
    """
    if min_cardinality < 3:
        raise ValueError(
            f"min_cardinality must be at least 3 (a 2-note MOS is a bare "
            f"generator), got {min_cardinality}"
        )
    if max_cardinality < min_cardinality:
        raise ValueError(
            f"max_cardinality ({max_cardinality}) is below min_cardinality "
            f"({min_cardinality})"
        )
    if period <= 1:
        raise ValueError(f"period ratio must exceed 1, got {period!r}")
    if dedupe_cents < 0:
        raise ValueError(f"dedupe_cents must be non-negative, got {dedupe_cents!r}")

    positions, w, n_merged, targets, order = _prepare_targets(
        ratios, weights, period, fold=fold
    )

    # Every interval the signal states, as a folded generator, in a canonical
    # order.  Sorting here is what makes the whole function a function of the
    # multiset of ratios rather than of the caller's list: the windows below
    # are cut from this sequence, and a sequence built in arrival order would
    # cut them differently for a different permutation of the same numbers.
    tol = dedupe_cents / (T.PERIOD_CENTS * math.log2(period))
    proposals: List[Tuple[float, float, float]] = []
    for num, den in _observed_intervals(ratios, include_ratios, include_intervals):
        g = _fold_bright(math.log(num / den) / math.log(period))
        if g is not None:
            proposals.append((g, num, den))
    proposals.sort()

    # The bright half is an open interval, so a plain linear comparison is
    # enough -- folding has already collapsed each generator onto its
    # complement, and nothing wraps around.  A window is a run of sorted
    # proposals within ``tol`` of wherever it opened.
    windows: List[List[Tuple[float, float, float]]] = []
    for item in proposals:
        if windows and item[0] - windows[-1][0][0] <= tol:
            windows[-1].append(item)
        else:
            windows.append([item])

    out: List[ForwardScale] = []
    for members in windows:
        # Which proposal represents the window is decided on merit, not on
        # arrival.  Two peak pairs three-quarters of a cent apart are one
        # reading by construction, but they are not equally good readings, and
        # the one that explains the targets better is the defensible thing to
        # report.  Merit is the best rank key the proposal achieves over its
        # own MOS series, with the generator dropped from the key: inside a
        # window the generators differ only in the sub-cent digits, and letting
        # that decide would be arbitration by rounding.  Ties fall to the
        # smallest ``(numerator, denominator)``, which is a property of the
        # data and not of its order.
        best_key: Optional[Tuple] = None
        best_fits: List[MOSFit] = []
        best_pair: Tuple[float, float] = (members[0][1], members[0][2])
        for g, num, den in members:
            fits: List[MOSFit] = []
            for card, n_large, n_small in T.mos_series(
                g, max_cardinality=max_cardinality, include_trivial=True
            ):
                if card < min_cardinality:
                    continue
                scale = MOSScale(n_large, n_small, g, period, validate=False)
                if _is_degenerate_scale(scale):
                    continue
                fits.append(
                    _evaluate(
                        scale, positions, w, tolerance_cents, complexity_penalty,
                        align=align, n_anchors=n_anchors,
                    )
                )
            merit = min((f._rank_key[:-1] for f in fits), default=None)
            # A proposal supporting no scale in range still corroborates the
            # window -- it counts in ``n_sources`` -- but it cannot speak for
            # it, so it sorts behind every proposal that does.
            key = (merit is None, merit if merit is not None else (), num, den)
            if best_key is None or key < best_key:
                best_key, best_fits, best_pair = key, fits, (num, den)

        others = sorted(
            pair for pair in ((m[1], m[2]) for m in members) if pair != best_pair
        )
        sources = (best_pair,) + tuple(others)
        for fit in best_fits:
            out.append(
                ForwardScale(
                    fit=_in_caller_order(fit, order, n_merged, targets),
                    sources=sources,
                )
            )

    # Stable, and everything feeding it is already in a canonical order, so
    # readings that tie on the rank key still come out the same way round.
    out.sort(key=lambda r: r._rank_key)
    return out if top_n is None else out[:top_n]


# --------------------------------------------------------------------------- #
# Biotuner object bridge
# --------------------------------------------------------------------------- #
#: Tuning source -> the ``compute_biotuner`` attribute that weights it.
#:
#: A ``compute_biotuner`` holds exactly two amplitude vectors: ``amps``, set
#: beside ``peaks`` by ``peaks_extraction``, and ``extended_amps``, set beside
#: ``extended_peaks`` by ``peaks_extension``.  Nothing else on the object is a
#: per-ratio weight.  ``cons_ratios`` looks like a counter-example, since
#: ``consonant_ratios`` returns a consonance value per ratio -- but the call
#: site discards it, and it is computed before the ``set()`` that de-duplicates
#: the ratios, so it would not line up even if it were kept.  The remaining
#: sources come out of a curve's minima (``diss_curve``, ``HE``) or are
#: constructed rather than measured (``euler_fokker``, ``harm_tuning``,
#: ``harm_fit_tuning``), and carry no weights at all.
_SOURCE_WEIGHTS: Dict[str, str] = {
    "peaks_ratios": "amps",
    "ratios": "amps",
    "extended_ratios": "extended_amps",
}


def _reject_circular_source(source: str) -> None:
    """Refuse to fit an MOS to an MOS."""
    if source == "mos":
        raise ValueError(
            "source='mos' would fit a moment-of-symmetry scale to a "
            "moment-of-symmetry scale: get_tuning('mos') returns the ratios of "
            "an earlier fit, so the answer is guaranteed in advance and the "
            "near-zero error it reports measures nothing. Choose a derivation "
            "the signal actually produced -- 'peaks_ratios', 'cons_ratios', "
            "'diss_curve', 'HE', 'euler_fokker', 'harm_fit_tuning'."
        )


#: Names :meth:`compute_biotuner.get_tuning` dispatches on beyond
#: :data:`~biotuner.biotuner_object.TUNING_SOURCES` -- the spellings it accepts
#: as synonyms for a canonical source.
_SOURCE_ALIASES: Tuple[str, ...] = (
    "ratios", "peaks_ratios_cons", "harmonic_entropy", "harmonic_tuning",
    "harmonic_fit",
)


def _reject_unknown_source(source) -> None:
    """Refuse a name no derivation answers to, before any window is analysed.

    :func:`mos_trajectory` treats a window it cannot derive the source from as
    a gap in the path rather than a failure, which is right for a source that
    is simply absent from one epoch -- but a *misspelt* source is absent from
    every epoch, and the gap-per-window rule turns it into an all-``None``
    trajectory that reads exactly like "this recording has no structure".  The
    name is therefore checked once, up front, where the answer cannot depend on
    the signal.
    """
    from biotuner.biotuner_object import TUNING_SOURCES

    if source in TUNING_SOURCES or source in _SOURCE_ALIASES:
        return
    raise ValueError(
        f"source must be one of {TUNING_SOURCES}, got {source!r}. A name no "
        f"derivation answers to cannot be a per-window gap: it would fail in "
        f"every window and come back as an all-None trajectory."
    )


def _source_weights(bt, source: str, ratios: Sequence[float]) -> Optional[np.ndarray]:
    """The amplitude vector belonging to a tuning source, if one genuinely does.

    Two guards, both deliberately unforgiving: nothing is padded, truncated,
    resampled or clipped to make a vector fit.

    *Length.*  The peak-ratio sources hold one ratio per *pair* of peaks, so
    the counts usually disagree and the vector is refused.  They do not always
    disagree -- an exact harmonic stack of five peaks de-duplicates to exactly
    five ratios -- so a match is not on its own proof of alignment; see the
    note below.

    *Sign.*  ``compute_biotuner`` stores peak amplitudes in decibels for
    several ``peaks_function`` settings, and a dB level is routinely negative
    (``extended_amps`` on a plain harmonic stack measures
    ``[5.24, 2.10, -0.81, -3.89, -6.86]``).  A negative entry is not a weight:
    passed through, :func:`_prepare_targets` clamps it to zero, which removes
    that target from ``error_cents``, ``rms_error_cents`` and ``coverage``
    while ``n_targets`` still counts it.  The fit then reports 0.000 cents
    beside residuals of 126 cents, and :attr:`MOSFit.evidence` -- which
    multiplies by ``sqrt(3 * n_targets)`` -- rewards it for the targets it
    just stopped measuring.  Refusing the whole vector is the same call the
    length check makes: no weighting beats wrong weighting.

    Notes
    -----
    A length match is necessary, not sufficient.  ``peaks_ratios`` is sorted by
    ratio value while ``amps`` is in peak order, so when the two do line up the
    weighting is positional coincidence rather than provenance.  Fixing that
    means re-deriving which pair of peaks produced each ratio and weighting by
    the pair, which is a feature rather than a guard, and is not done here.
    """
    attr = _SOURCE_WEIGHTS.get(source)
    if attr is None:
        return None
    amps = getattr(bt, attr, None)
    if amps is None:
        return None
    w = np.asarray(amps, dtype=float).ravel()
    if w.size != len(ratios):
        return None
    if not np.all(np.isfinite(w)) or np.any(w <= 0.0):
        return None
    return w


def mos_from_biotuner(
    bt,
    source: str = "peaks_ratios",
    use_amplitudes: bool = True,
    mode: str = "inverse",
    **kwargs,
) -> Union[List[MOSFit], List[ForwardScale]]:
    """Read MOS scales off a :class:`~biotuner.biotuner_object.compute_biotuner`.

    Every way the object has of deriving ratios can feed the fit, so the
    question "which well-formed scale is this signal in?" can be asked of the
    peak ratios, the dissonance-curve minima, the harmonic-entropy minima, an
    Euler-Fokker genus, or the common-harmonic tuning, and the answers compared
    -- see :func:`compare_sources`.

    Parameters
    ----------
    bt : compute_biotuner
        Must already have run ``peaks_extraction``; sources beyond the peak
        ratios may need their own precursor (``peaks_extension`` for
        ``'extended_ratios'``, for instance).
    source : str, default 'peaks_ratios'
        Any name :meth:`compute_biotuner.get_tuning` accepts, except ``'mos'``.
    use_amplitudes : bool, default True
        Weight each ratio by its peak amplitude where an amplitude vector
        genuinely lines up with the source.  A strong peak should pull the fit
        harder than a weak one.
    mode : {'inverse', 'forward'}, default 'inverse'
        Which question to ask.

        ``'inverse'`` runs :func:`fit_mos`: the generator is latent, searched
        for, and need not be an interval the signal contains.

        ``'forward'`` runs :func:`forward_scales`: every generator tried is an
        interval the signal states, and the result says what scale that
        interval builds.  The two are scored identically, so their
        ``error_cents`` and ``coverage`` can be compared directly.
    **kwargs
        Passed to :func:`fit_mos` or :func:`forward_scales`, per ``mode``.

    Returns
    -------
    list of MOSFit or list of ForwardScale
        Depending on ``mode``.

    Raises
    ------
    ValueError
        If ``source='mos'``, which would fit a moment-of-symmetry scale to a
        moment-of-symmetry scale; or if ``mode`` is neither ``'inverse'`` nor
        ``'forward'``.

    Notes
    -----
    Only ``'peaks_ratios'`` and ``'extended_ratios'`` have a candidate weight
    vector at all (``bt.amps`` and ``bt.extended_amps``), and each is used only
    when its length matches the derived ratios exactly -- see
    :data:`_SOURCE_WEIGHTS`.  Everything else is fitted unweighted, which is
    the honest default: an invented weighting would move the answer without
    being derived from anything.

    Examples
    --------
    >>> mos_from_biotuner(bt, mode='sideways')      # doctest: +SKIP
    Traceback (most recent call last):
    ValueError: mode must be 'inverse' or 'forward', got 'sideways'
    """
    if mode not in ("inverse", "forward"):
        raise ValueError(
            f"mode must be 'inverse' or 'forward', got {mode!r}. 'inverse' "
            f"searches for the latent generator that best explains the ratios; "
            f"'forward' takes an interval the signal actually states and "
            f"reports the scale it generates."
        )
    _reject_circular_source(source)
    ratios = bt.get_tuning(source)
    weights = _source_weights(bt, source, ratios) if use_amplitudes else None
    if mode == "forward":
        return forward_scales(ratios, weights=weights, **kwargs)
    return fit_mos(ratios, weights=weights, **kwargs)


def compare_sources(
    bt,
    sources: Optional[Sequence[str]] = None,
    use_amplitudes: bool = True,
    **kwargs,
) -> "pd.DataFrame":
    """Fit an MOS from every tuning derivation, and rank them.

    Biotuner derives a scale from a signal in eight different ways, and they do
    not agree.  This runs the same fit through all of them and puts the results
    side by side, so the question stops being "which scale is this?" and becomes
    "which *way of asking* produces a well-formed answer at all?".

    Parameters
    ----------
    bt : compute_biotuner
        Must already have run ``peaks_extraction``.
    sources : sequence of str, optional
        Which derivations to try.  Defaults to every name in
        :data:`~biotuner.biotuner_object.TUNING_SOURCES` except ``'mos'``,
        which would be circular.
    use_amplitudes : bool, default True
        As :func:`mos_from_biotuner`.
    **kwargs
        Passed to :func:`fit_mos`.

    Returns
    -------
    pandas.DataFrame
        One row per source, best first.  Columns: ``source``, ``n_ratios`` (as
        derived), ``n_targets`` and ``n_merged`` (as fitted, after folding),
        ``signature``, ``cardinality``, ``generator_cents``, ``error_cents``,
        ``chance_error_cents``, ``improvement``, ``evidence``, ``coverage``,
        ``score``, ``underdetermined``, ``reason``.

        A source that raises still gets a row, with ``reason`` holding the
        exception and everything else ``NaN``.  Silently dropping it would hide
        a real breakage: ``harm_tuning`` currently fails outright for every
        ``peaks_function`` other than ``'harmonic_recurrence'``, and a shorter
        table is not a report of that.

    Notes
    -----
    Rows are ordered by :attr:`MOSFit.evidence`, descending, failures last.

    "Most convincing" cannot mean lowest ``error_cents``.  A fit with two
    targets reports 0.00 cents against a four-note scale, because four degrees
    can be rotated onto any two points; ranking by error puts the derivation
    that produced the *least* data on top.  ``evidence`` measures the same error
    in units of the standard error a chance fit would have, so it grows with
    both the margin below chance and the number of targets that margin was
    measured over.  ``underdetermined`` marks the rows where the scale had more
    degrees than the data had points, which is the extreme case of the same
    problem.

    Examples
    --------
    >>> df = compare_sources(bt)                            # doctest: +SKIP
    >>> df[["source", "signature", "error_cents", "evidence"]]  # doctest: +SKIP
    """
    import pandas as pd

    if sources is None:
        from biotuner.biotuner_object import TUNING_SOURCES

        sources = [s for s in TUNING_SOURCES if s != "mos"]
    kwargs.setdefault("top_n", 1)

    rows: List[Dict[str, object]] = []
    for src in sources:
        row: Dict[str, object] = {
            "source": src,
            "n_ratios": np.nan,
            "n_targets": np.nan,
            "n_merged": np.nan,
            "signature": None,
            "cardinality": np.nan,
            "generator_cents": np.nan,
            "error_cents": np.nan,
            "chance_error_cents": np.nan,
            "improvement": np.nan,
            "evidence": np.nan,
            "coverage": np.nan,
            "score": np.nan,
            "underdetermined": None,
            "reason": None,
        }
        try:
            _reject_circular_source(src)
            ratios = bt.get_tuning(src)
            row["n_ratios"] = int(len(ratios))
            weights = _source_weights(bt, src, ratios) if use_amplitudes else None
            fits = fit_mos(ratios, weights=weights, **kwargs)
            if not fits:
                raise ValueError(
                    "no MOS scale could be fitted; widen the cardinality range"
                )
        except Exception as exc:
            # Reported, not swallowed: a source that cannot be computed is a
            # finding about this signal (or about biotuner), not a row to drop.
            row["reason"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
            continue

        fit = fits[0]
        row.update(
            n_targets=fit.n_targets,
            n_merged=fit.n_merged,
            signature=fit.signature,
            cardinality=fit.scale.cardinality,
            generator_cents=fit.scale.generator_cents,
            error_cents=fit.error_cents,
            chance_error_cents=fit.chance_error_cents,
            improvement=fit.improvement,
            evidence=fit.evidence,
            coverage=fit.coverage,
            score=fit.score,
            underdetermined=fit.is_underdetermined,
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    # mergesort is stable, so sources that tie keep the order they were asked in.
    return df.sort_values(
        "evidence", ascending=False, na_position="last", kind="mergesort"
    ).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Time-resolved: a path through the labyrinth
# --------------------------------------------------------------------------- #
def trajectory_from_windows(
    windows: Sequence[Sequence[float]],
    weights: Optional[Sequence[Optional[Sequence[float]]]] = None,
    **kwargs,
) -> List[Optional[MOSFit]]:
    """Best-fitting MOS for each window of ratios.

    Windows that yield no usable ratios become ``None`` rather than raising, so
    one bad epoch does not sink a whole recording.

    Examples
    --------
    >>> a = MOSScale.from_signature(5, 2, tuning=12).ratios
    >>> b = MOSScale.from_signature(4, 3, tuning=19).ratios
    >>> [f.signature for f in trajectory_from_windows([a, b], max_cardinality=12)]
    ['5L2s', '4L3s']
    """
    kwargs.setdefault("top_n", 1)
    out: List[Optional[MOSFit]] = []
    for i, win in enumerate(windows):
        w = None if weights is None else weights[i]
        try:
            fits = fit_mos(win, weights=w, **kwargs)
        except ValueError:
            out.append(None)
            continue
        out.append(fits[0] if fits else None)
    return out


def mos_trajectory(
    data: Sequence[float],
    sf: float,
    window_sec: float = 4.0,
    step_sec: Optional[float] = None,
    peaks_function: str = "EMD",
    n_peaks: int = 5,
    precision: float = 0.5,
    bt_kwargs: Optional[Dict[str, object]] = None,
    source: str = "peaks_ratios",
    use_amplitudes: bool = True,
    **kwargs,
) -> List[Optional[MOSFit]]:
    """Track which well-formed scale a signal occupies, window by window.

    A path through the labyrinth.  Each window's peaks are extracted with
    :class:`~biotuner.biotuner_object.compute_biotuner` and fitted
    independently, so the returned sequence shows the scale structure drifting
    (or holding) as the signal evolves --
    :func:`~biotuner.mos.plotting.plot_mos_trajectory` draws it.

    Parameters
    ----------
    data : sequence of float
        A single-channel time series.
    sf : float
        Sampling frequency, Hz.
    window_sec : float, default 4.0
    step_sec : float, optional
        Hop between windows; defaults to ``window_sec / 2`` (50 % overlap).
    peaks_function, n_peaks, precision
        Passed to peak extraction.
    bt_kwargs : dict, optional
        Extra keyword arguments for the ``compute_biotuner`` constructor.
    source : str, default 'peaks_ratios'
        Which derivation to fit in each window -- any name
        :meth:`compute_biotuner.get_tuning` accepts, except ``'mos'``.  A
        trajectory over ``'diss_curve'`` and one over ``'peaks_ratios'`` are
        different measurements of the same recording, and there is no reason to
        be able to make only the second.
    use_amplitudes : bool, default True
        Weight each window's ratios by peak amplitude where the source has a
        matching amplitude vector; see :func:`mos_from_biotuner`.
    **kwargs
        Passed to :func:`fit_mos`.

    Returns
    -------
    list of MOSFit or None
        One entry per window; ``None`` where peak extraction found nothing.

    Raises
    ------
    ValueError
        If ``source='mos'``, which would fit an MOS to an MOS in every window;
        or if ``source`` is a name no derivation answers to.  A misspelt source
        fails in every window, and the per-window gap rule would otherwise turn
        it into an all-``None`` path indistinguishable from a structureless
        recording, so the name is validated before any window is analysed.

    Notes
    -----
    This runs a full peak extraction per window, so it is the slow entry point
    in this module.  For a signal you have already windowed and analysed, call
    :func:`trajectory_from_windows` with the ratios directly.

    A source that cannot be derived in a given window yields ``None`` for that
    window rather than raising, on the same principle as an empty window: one
    bad epoch is a gap in the path, not a failed recording.  Sources needing a
    precursor the per-window object never runs -- ``'extended_ratios'`` wants a
    ``peaks_extension`` -- therefore come back as an all-``None`` trajectory,
    which is a truthful answer rather than a crash.
    """
    _reject_circular_source(source)
    _reject_unknown_source(source)
    from biotuner.biotuner_object import compute_biotuner

    x = np.asarray(data, dtype=float).squeeze()
    if x.ndim != 1:
        raise ValueError(f"data must be one-dimensional, got shape {x.shape}")
    step_sec = step_sec if step_sec is not None else window_sec / 2.0
    win = int(round(window_sec * sf))
    hop = int(round(step_sec * sf))
    if win < 2 or hop < 1:
        raise ValueError(
            f"window_sec={window_sec} and step_sec={step_sec} give {win} and "
            f"{hop} samples at sf={sf}; both must be at least 1 sample"
        )
    if win > x.size:
        raise ValueError(
            f"window of {win} samples exceeds the {x.size}-sample signal"
        )

    bt_kwargs = dict(bt_kwargs or {})
    windows: List[Sequence[float]] = []
    weights: List[Optional[Sequence[float]]] = []
    for start in range(0, x.size - win + 1, hop):
        seg = x[start : start + win]
        try:
            bt = compute_biotuner(
                sf, peaks_function=peaks_function, precision=precision, **bt_kwargs
            )
            bt.peaks_extraction(seg, n_peaks=n_peaks)
            ratios = bt.get_tuning(source)
            w = _source_weights(bt, source, ratios) if use_amplitudes else None
            windows.append(list(ratios))
            weights.append(None if w is None else list(w))
        except Exception:
            # A window with no extractable peaks -- or none this source can be
            # derived from -- is a gap, not a failure.
            windows.append([])
            weights.append(None)
    return trajectory_from_windows(windows, weights, **kwargs)


def trajectory_dataframe(
    trajectory: Sequence[Optional[MOSFit]],
    times: Optional[Sequence[float]] = None,
) -> "pd.DataFrame":
    """Tabulate a trajectory: one row per window, ``NaN`` where the fit failed.

    Examples
    --------
    >>> a = MOSScale.from_signature(5, 2, tuning=12).ratios
    >>> traj = trajectory_from_windows([a], max_cardinality=12)
    >>> df = trajectory_dataframe(traj)
    >>> list(df["signature"])
    ['5L2s']
    """
    import pandas as pd

    rows = []
    for i, fit in enumerate(trajectory):
        t = float(times[i]) if times is not None else float(i)
        if fit is None:
            rows.append({"window": i, "time": t, "signature": None})
            continue
        row = {"window": i, "time": t}
        row.update(fit.to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def explain_fit(fit: MOSFit, ratios: Optional[Sequence[float]] = None) -> str:
    """A readable account of what a fit claims, and how well it holds up.

    Alongside the error it prints the error a *random* set of ratios would get
    against a scale this size, because the raw number is unreadable without it:
    0.00 cents from two data points is not better than 5.19 cents from eight,
    it is a smaller scale having somewhere to put everything.  A fit with more
    degrees than targets is labelled ``UNDERDETERMINED`` outright.

    Examples
    --------
    >>> print(explain_fit(best_mos(MOSScale.from_signature(5, 2, tuning=12).ratios)))
    ... # doctest: +ELLIPSIS
    5L2s  (7 notes)   LLLsLLs
    ...
      fit            error 0.000 c (weighted mean), max 0.000 c, rms 0.000 c
    ...

    One ratio is not evidence for a four-note scale, however well it fits:

    >>> print(explain_fit(best_mos([1.5])))          # doctest: +ELLIPSIS
    1L3s...
      UNDERDETERMINED  1 target for 4 degrees...
    """
    lines = [fit.scale.summary()]
    lines.append(
        f"  fit            error {fit.error_cents:.3f} c (weighted mean), "
        f"max {fit.max_error_cents:.3f} c, rms {fit.rms_error_cents:.3f} c"
    )
    improvement = (
        "unbounded" if math.isinf(fit.improvement) else f"{fit.improvement:.2f}x"
    )
    lines.append(
        f"  chance         {fit.chance_error_cents:.3f} c for "
        f"{fit.scale.cardinality} degrees;  improvement {improvement};  "
        f"evidence {fit.evidence:.2f} SE below chance"
    )
    lines.append(
        f"  coverage       {fit.coverage:.1%} of weight within tolerance;  "
        f"{fit.n_targets} targets, {fit.n_unmatched_degrees} unused degrees;  "
        f"score {fit.score:.3f}"
    )
    if fit.n_merged:
        lines.append(
            f"  folded         {fit.n_merged} ratio(s) merged into a pitch "
            f"class already present (an octave is not a second target)"
        )
    if fit.is_underdetermined:
        plural = "" if fit.n_targets == 1 else "s"
        lines.append(
            f"  UNDERDETERMINED  {fit.n_targets} target{plural} for "
            f"{fit.scale.cardinality} degrees: a scale with spare notes can be "
            f"rotated onto any data, so this error is not evidence"
        )
    if ratios is not None:
        shown = list(ratios)
        if len(shown) != fit.n_targets and fit.targets:
            # Folding merged octave-equivalents, so the caller's list no longer
            # runs parallel to the residuals; report what was actually fitted.
            shown = list(fit.targets)
        cents = fit.scale.cents
        lines.append("  targets        each ratio and where it landed")
        for r, deg, res in zip(shown, fit.assignments, fit.residuals):
            lines.append(
                f"      {float(r):9.6f} -> degree {deg:2d} "
                f"({cents[deg]:8.2f} c, {res:+7.2f} c)"
            )
    return "\n".join(lines)
