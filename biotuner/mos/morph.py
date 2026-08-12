"""Morphing between well-formed scales -- three ways of navigating the labyrinth.

Milne et al. (2011) built the labyrinth as a surface to *move across*: their §1
describes choosing a structure and a tuning at once, and their §6 keeps the
timbre consonant while you do. This module makes the movement itself the
object, and offers three strategies that are genuinely different journeys
rather than three spellings of one.

``tuning``
    Hold the structure, slide the generator. The path runs **along a single
    arc** of the labyrinth. This is the Dynamic Tonality knob: the scale keeps
    its identity while its two step sizes co-vary, and where the path crosses
    an equalized landmark the large and small steps trade places and the scale
    becomes its own inverse. See :func:`tuning_morph`.

``tree``
    Change the structure, discretely. The path **hops between rings**, walking
    the labyrinth's own connectivity: a signature's children are
    ``(nL, nL+ns)`` and ``(nL+ns, ns)`` -- the Stern-Brocot mediant, which is
    why ``5L2s``'s child ``5L7s`` has exactly the twelve notes
    :func:`~biotuner.mos.theory.embedding` predicts -- its parent is the
    subtractive Euclidean step, and one further edge swaps ``(nL, ns)`` for
    ``(ns, nL)`` by crossing a landmark. Pentatonic to chromatic is a walk up
    this tree. See :func:`tree_morph`.

``voice``
    Move the notes and ignore the structure. Each tone glides to its nearest
    counterpart in the target; where the two scales have different note counts,
    tones split or merge. See :func:`voice_morph`.

The contrast is the point. The first two paths never leave the set of
well-formed scales, because every frame is one by construction. The third
does: its intermediate pitch sets are generally *not* well-formed, and
:attr:`MorphStep.wellformedness` measures how far outside it strays. Sliding a
generator and gliding the notes are different journeys between the same two
places, and they sound different.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from biotuner.mos import theory as T
from biotuner.mos.scale import MOSScale

__all__ = [
    "STRATEGIES",
    "MorphStep",
    "Morph",
    "morph",
    "tuning_morph",
    "tree_morph",
    "voice_morph",
    "signature_children",
    "signature_parent",
    "signature_route",
    "plot_morph_path",
    "plot_morph_trajectory",
    "plot_morph_filmstrip",
    "plot_morph_comparison",
    "animate_morph",
    "morph_audio",
]

#: The three journeys.
STRATEGIES: Tuple[str, ...] = ("tuning", "tree", "voice")


# --------------------------------------------------------------------------- #
# One frame of a morph
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MorphStep:
    """One frame: a set of pitches, and whatever structure it happens to have.

    Attributes
    ----------
    t : float
        Position along the morph, ``0`` at the start scale and ``1`` at the end.
    degrees : tuple of float
        Pitch classes as period fractions in ``[0, 1)``, ascending. Always
        present -- this is the only thing every strategy guarantees.
    scale : MOSScale or None
        The well-formed scale this frame *is*, when it is one. ``None`` for a
        voice-leading frame that has left the space.
    period : float
    label : str
        Short human-readable name for the frame.
    event : str or None
        Something worth hearing happened here: a landmark crossed, a step count
        changed, tones split or merged.
    wellformedness : float
        Cents by which this frame misses being a well-formed scale -- ``0.0``
        when it is one. Computed for voice frames, where it is the interesting
        quantity; the other strategies are exact by construction.
    """

    t: float
    degrees: Tuple[float, ...]
    scale: Optional[MOSScale] = None
    period: float = 2.0
    label: str = ""
    event: Optional[str] = None
    wellformedness: float = 0.0

    @property
    def cardinality(self) -> int:
        return len(self.degrees)

    @property
    def cents(self) -> List[float]:
        pc = T.PERIOD_CENTS * math.log2(self.period)
        return [d * pc for d in self.degrees]

    @property
    def ratios(self) -> List[float]:
        return [self.period**d for d in self.degrees]

    @property
    def is_well_formed(self) -> bool:
        return self.scale is not None


@dataclass(frozen=True)
class Morph:
    """A journey between two well-formed scales.

    Attributes
    ----------
    steps : tuple of MorphStep
    strategy : str
    start, end : MOSScale
    voices : tuple of tuple of int
        Per frame, which voice each degree belongs to, so a trajectory can be
        drawn as continuous lines rather than a scatter. Empty for strategies
        where the note count changes and voices are not tracked.
    """

    steps: Tuple[MorphStep, ...]
    strategy: str
    start: MOSScale
    end: MOSScale
    voices: Tuple[Tuple[int, ...], ...] = ()

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    def __getitem__(self, i):
        return self.steps[i]

    @property
    def period_cents(self) -> float:
        return T.PERIOD_CENTS * math.log2(self.start.period)

    def events(self) -> List[Tuple[float, str]]:
        """Every ``(t, description)`` where something audible happened."""
        return [(s.t, s.event) for s in self.steps if s.event]

    def signatures(self) -> List[str]:
        """The distinct signatures passed through, in order."""
        out: List[str] = []
        for s in self.steps:
            name = s.scale.signature if s.scale else "(off-scale)"
            if not out or out[-1] != name:
                out.append(name)
        return out

    def labyrinth_path(self) -> List[Optional[Tuple[float, int]]]:
        """``(generator, cardinality)`` per frame; ``None`` where off the map."""
        return [
            (s.scale.generator, s.scale.cardinality) if s.scale else None
            for s in self.steps
        ]

    def trajectory(self) -> np.ndarray:
        """Degrees as a ``(frames, max_cardinality)`` array, NaN-padded.

        Rows are frames. Columns are *voices* where the strategy tracked them,
        and sorted degrees where it did not.

        The difference matters wherever two tones cross. Read in ascending
        order, a crossing hands each tone to the other's column, which looks
        like two large jumps instead of two lines passing; summed up it charges
        motion no voice performs. :attr:`voices` records which tone is which,
        so a voice morph is unscrambled here and every consumer -- the
        trajectory plot, :func:`morph_audio`, :meth:`voice_leading_distance` --
        gets continuous lines for free.
        """
        width = max(s.cardinality for s in self.steps)
        out = np.full((len(self.steps), width), np.nan)
        tracked = len(self.voices) == len(self.steps)
        for i, s in enumerate(self.steps):
            order = self.voices[i] if tracked else ()
            if len(order) == s.cardinality and max(order, default=-1) < width:
                for slot, voice in enumerate(order):
                    out[i, voice] = s.degrees[slot]
            else:
                out[i, : s.cardinality] = s.degrees
        return out

    def voice_leading_distance(self) -> float:
        """Total pitch motion over the whole journey, in cents.

        Summed frame to frame over :meth:`trajectory`'s columns -- matched
        voices where the strategy tracked them -- so it measures how far the
        tones actually travel rather than how far apart the endpoints are. Each
        hop takes the shorter way round the period, and a column that is absent
        from either frame contributes nothing, which is how a change of note
        count is handled without inventing motion for the tones that are not
        there yet.

        The three strategies give genuinely different totals for the same pair;
        that difference is what makes them different journeys rather than
        different spellings of one.
        """
        gaps = np.abs(np.diff(self.trajectory(), axis=0))
        return float(np.nansum(np.minimum(gaps, 1.0 - gaps)) * self.period_cents)

    def summary(self) -> str:
        """Multi-line description of the journey."""
        lines = [
            f"{self.strategy} morph:  {self.start.signature} "
            f"({self.start.generator_cents:.1f} c) -> {self.end.signature} "
            f"({self.end.generator_cents:.1f} c)",
            f"  frames         {len(self.steps)}",
            f"  route          {' -> '.join(self.signatures())}",
            f"  voice motion   {self.voice_leading_distance():.1f} c total",
        ]
        off = [s for s in self.steps if not s.is_well_formed]
        if off:
            worst = max(s.wellformedness for s in off)
            lines.append(
                f"  leaves the labyrinth for {len(off)} of {len(self.steps)} "
                f"frames, by up to {worst:.1f} c"
            )
        else:
            lines.append("  every frame is a well-formed scale")
        for t, ev in self.events():
            lines.append(f"    t={t:.3f}  {ev}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# The signature graph -- the labyrinth's own connectivity
# --------------------------------------------------------------------------- #
def signature_children(n_large: int, n_small: int) -> List[Tuple[int, int]]:
    """The two signatures a scale is directly embedded in.

    Taking the Stern-Brocot mediant of the step counts, which is why the
    diatonic's child has exactly the twelve notes
    :func:`~biotuner.mos.theory.embedding` predicts.

    Examples
    --------
    >>> signature_children(5, 2)
    [(5, 7), (7, 2)]
    """
    return [(n_large, n_large + n_small), (n_large + n_small, n_small)]


def signature_parent(n_large: int, n_small: int) -> Optional[Tuple[int, int]]:
    """The signature this one is embedded in, or ``None`` at the root ``1L1s``.

    The subtractive Euclidean step, run backwards up the tree.

    Examples
    --------
    >>> signature_parent(5, 2)
    (3, 2)
    >>> signature_parent(1, 1) is None
    True
    """
    if n_large == n_small:
        return None
    if n_large > n_small:
        return (n_large - n_small, n_small)
    return (n_large, n_small - n_large)


def signature_route(
    start: Tuple[int, int],
    end: Tuple[int, int],
    max_cardinality: int = 40,
    allow_inverse: bool = True,
) -> List[Tuple[int, int]]:
    """Shortest route between two signatures through the labyrinth.

    Three legal moves: down to a child, up to the parent, and -- when
    ``allow_inverse`` -- across to ``(ns, nL)``, which is a single continuous
    move because the two meet at their shared equalized landmark.

    Shortest routes are usually not unique, and the tie-break is musical rather
    than arbitrary: among equally short routes, take the one whose sequence of
    note counts is lexicographically smallest, which is to say the one that
    stays small longest and adds notes only when it must. Pentatonic to
    chromatic then reads ``2L3s → 3L2s → 5L2s → 5L7s`` -- five, five, seven,
    twelve -- rather than an equally short detour that reaches twelve notes a
    step early and doubles back.

    Parameters
    ----------
    start, end : (int, int)
        Co-prime ``(n_large, n_small)`` pairs.
    max_cardinality : int, default 40
        Ceiling on ``nL + ns`` anywhere along the route. Too low and there may
        be no route at all.
    allow_inverse : bool, default True

    Returns
    -------
    list of (int, int)
        Including both endpoints. A single-element list when they coincide.

    Raises
    ------
    ValueError
        If either signature is not co-prime, or no route exists under the
        cardinality ceiling.

    Examples
    --------
    Pentatonic to diatonic is one step -- the diatonic is the pentatonic's child:

    >>> signature_route((2, 3), (5, 2))
    [(2, 3), (3, 2), (5, 2)]

    >>> signature_route((5, 2), (4, 3))
    [(5, 2), (3, 2), (1, 2), (1, 3), (4, 3)]
    """
    for sig in (start, end):
        if len(sig) != 2 or min(sig) < 1:
            raise ValueError(f"a signature must be two positive counts, got {sig!r}")
        if math.gcd(*sig) != 1:
            raise ValueError(
                f"a MOS signature must be co-prime, got {sig[0]}L{sig[1]}s "
                f"with gcd {math.gcd(*sig)}"
            )
    start, end = tuple(start), tuple(end)
    if start == end:
        return [start]

    def neighbours(sig):
        nL, ns = sig
        out = []
        for child in signature_children(nL, ns):
            if sum(child) <= max_cardinality:
                out.append(child)
        parent = signature_parent(nL, ns)
        if parent is not None:
            out.append(parent)
        if allow_inverse and nL != ns:
            out.append((ns, nL))
        return out

    # Best-first on (hops, note-count sequence). Both parts only grow as a path
    # is extended, so the first time the target is popped its route is optimal
    # under the whole key -- shortest first, then musically simplest.
    import heapq

    counter = 0
    seed = ((0, (sum(start),)), counter, [start])
    heap = [seed]
    best: Dict[Tuple[int, int], Tuple[int, Tuple[int, ...]]] = {start: seed[0]}
    while heap:
        key, _, path = heapq.heappop(heap)
        node = path[-1]
        if node == end:
            return path
        if key > best.get(node, key):
            continue
        for nxt in neighbours(node):
            nkey = (key[0] + 1, key[1] + (sum(nxt),))
            if nxt in best and best[nxt] <= nkey:
                continue
            best[nxt] = nkey
            counter += 1
            heapq.heappush(heap, (nkey, counter, path + [nxt]))
    raise ValueError(
        f"no route from {start[0]}L{start[1]}s to {end[0]}L{end[1]}s with "
        f"cardinality at most {max_cardinality}; raise max_cardinality"
    )


# --------------------------------------------------------------------------- #
# 1. Tuning morph -- along one arc
# --------------------------------------------------------------------------- #
def _landmarks_between(a: float, b: float, cardinality: int) -> List[Tuple[float, str]]:
    """Equal temperaments the generator passes on its way from ``a`` to ``b``."""
    lo, hi = (a, b) if a <= b else (b, a)
    found = []
    for q in range(2, cardinality * 3 + 1):
        for p in range(1, q):
            if math.gcd(p, q) != 1:
                continue
            v = p / q
            if lo < v < hi:
                found.append((v, f"{q}-EDO ({p}/{q})"))
    found.sort()
    # Only the simplest few are audible landmarks rather than arithmetic noise.
    found.sort(key=lambda pv: (len(str(pv[1])), pv[0]))
    return sorted(found[:6])


def tuning_morph(
    start: MOSScale,
    end: MOSScale,
    steps: int = 64,
) -> Morph:
    """Hold the structure, slide the generator: a path along one arc.

    The scale keeps its note count throughout and its two step sizes co-vary,
    exactly as Milne et al. §2 describe. If the path crosses the equalized
    landmark the signature flips -- the scale meets and becomes its own
    inverse -- which is reported as an event rather than hidden.

    Parameters
    ----------
    start, end : MOSScale
        Must have the same cardinality. Different signatures are fine and
        interesting: ``5L2s`` to ``2L5s`` crosses 7-EDO on the way.
    steps : int, default 64
        Frames, endpoints included.

    Returns
    -------
    Morph

    Raises
    ------
    ValueError
        If the cardinalities differ -- there is no way to hold a structure
        fixed between scales that do not have one in common. Use
        :func:`tree_morph` for that.

    Examples
    --------
    Meantone to Pythagorean, seven notes throughout:

    >>> a = MOSScale.from_signature(5, 2, tuning=31)
    >>> b = MOSScale.from_generator(3 / 2, 7)
    >>> m = tuning_morph(a, b, steps=9)
    >>> len(m), m.signatures()
    (9, ['5L2s'])
    >>> all(s.is_well_formed for s in m)
    True

    Crossing a landmark flips the signature:

    >>> m = tuning_morph(a, a.inverse, steps=33)
    >>> m.signatures()
    ['5L2s', '2L5s']
    """
    if steps < 2:
        raise ValueError(f"steps must be at least 2, got {steps}")
    if start.cardinality != end.cardinality:
        raise ValueError(
            f"a tuning morph holds the structure fixed, so both scales need the "
            f"same note count; got {start.cardinality} and {end.cardinality}. "
            "Use tree_morph() to change cardinality."
        )

    ts = np.linspace(0.0, 1.0, steps)
    card = start.cardinality
    crossings = _landmarks_between(start.generator, end.generator, card)
    seen_events = set()
    out: List[MorphStep] = []
    previous_sig: Optional[str] = None

    for t in ts:
        g = (1.0 - t) * start.generator + t * end.generator
        # Geometric, so a pseudo-octave interpolates in pitch rather than ratio.
        period = start.period ** (1.0 - t) * end.period**t
        try:
            n_large, n_small = T.mos_signature(g, card)
        except ValueError:
            n_large, n_small = start.n_large, start.n_small
        scale = MOSScale(n_large, n_small, g, period, validate=False)

        event = None
        if previous_sig is not None and scale.signature != previous_sig:
            event = (f"{previous_sig} becomes {scale.signature}: the large and "
                     "small steps trade places")
        else:
            for value, name in crossings:
                lo, hi = sorted((start.generator, end.generator))
                if abs(g - value) < abs(hi - lo) / (2 * (steps - 1)) + 1e-12:
                    if name not in seen_events:
                        seen_events.add(name)
                        event = f"passes {name}"
                    break
        previous_sig = scale.signature

        out.append(
            MorphStep(
                t=float(t),
                degrees=tuple(scale.degrees),
                scale=scale,
                period=period,
                label=f"{scale.signature} {scale.generator_cents:.1f}c",
                event=event,
            )
        )
    return Morph(tuple(out), "tuning", start, end)


# --------------------------------------------------------------------------- #
# 2. Tree morph -- hopping between rings
# --------------------------------------------------------------------------- #
def tree_morph(
    start: MOSScale,
    end: MOSScale,
    max_cardinality: int = 40,
    steps_per_edge: int = 1,
    allow_inverse: bool = True,
) -> Morph:
    """Change the structure, one legal move at a time: a path between rings.

    Walks the shortest route through the signature graph -- see
    :func:`signature_route` -- and gives every signature on the way a tuning,
    chosen as close to the straight line between the two generators as its own
    valid range allows. Pentatonic to chromatic is a walk up this tree.

    Parameters
    ----------
    start, end : MOSScale
    max_cardinality : int, default 40
        Ceiling on note count anywhere along the route.
    steps_per_edge : int, default 1
        Extra frames interpolated *within* each signature, which glides the
        tuning between hops instead of jumping. The note count still changes
        abruptly at each hop, because it must. A route of ``n`` signatures
        yields ``(n - 1) * steps_per_edge + 1`` frames: the destination is a
        single frame, having no edge to glide along. Two tunings of the same
        signature count as one edge, not zero.
    allow_inverse : bool, default True
        Permit the landmark-crossing move between a signature and its inverse.

    Returns
    -------
    Morph

    Examples
    --------
    >>> a = MOSScale.from_signature(2, 3, tuning=12)
    >>> b = MOSScale.from_signature(5, 7, tuning=12)
    >>> m = tree_morph(a, b)
    >>> m.signatures()
    ['2L3s', '3L2s', '5L2s', '5L7s']
    >>> [s.cardinality for s in m]
    [5, 5, 7, 12]
    >>> [e for _, e in m.events()]
    ['start at 2L3s', '5 notes -> 7: 5L2s', '7 notes -> 12: 5L7s']
    >>> all(s.is_well_formed for s in m)
    True
    """
    if steps_per_edge < 1:
        raise ValueError(f"steps_per_edge must be at least 1, got {steps_per_edge}")
    route = signature_route(
        (start.n_large, start.n_small), (end.n_large, end.n_small),
        max_cardinality=max_cardinality, allow_inverse=allow_inverse,
    )

    out: List[MorphStep] = []
    # Two tunings of one signature still make a journey -- the structure holds
    # and the tuning slides -- so a single-node route is walked as an edge from
    # the signature to itself. Without this the lone node is overwritten by the
    # final substitution and the morph reports one frame and no motion, having
    # quietly teleported from its start to its end.
    nodes = route if len(route) > 1 else [route[0], route[0]]
    n_nodes = len(nodes)
    edges = n_nodes - 1
    previous_card: Optional[int] = None
    for i, (n_large, n_small) in enumerate(nodes):
        # The last node has no edge leading away from it, so nothing to
        # interpolate along: it gets a single frame rather than
        # ``steps_per_edge`` copies of the destination.
        for k in range(1 if i == n_nodes - 1 else steps_per_edge):
            frac = min(1.0, (i + k / steps_per_edge) / edges)
            target_g = (1.0 - frac) * start.generator + frac * end.generator
            period = start.period ** (1.0 - frac) * end.period**frac

            # Take whichever mirror range sits nearer the straight line, then
            # clamp inside it: the signature dictates where its tunings live.
            best = None
            for lo, hi in T.signature_ranges(n_large, n_small):
                inset = (float(hi) - float(lo)) * 1e-6
                g = min(max(target_g, float(lo) + inset), float(hi) - inset)
                d = abs(g - target_g)
                if best is None or d < best[0]:
                    best = (d, g)
            scale = MOSScale(n_large, n_small, best[1], period, validate=False)

            event = None
            if previous_card is not None and scale.cardinality != previous_card:
                event = (f"{previous_card} notes -> {scale.cardinality}: "
                         f"{scale.signature}")
            elif previous_card is None:
                event = f"start at {scale.signature}"
            previous_card = scale.cardinality

            out.append(
                MorphStep(
                    t=float(frac),
                    degrees=tuple(scale.degrees),
                    scale=scale,
                    period=period,
                    label=f"{scale.signature} ({scale.cardinality})",
                    event=event,
                )
            )
    # Land exactly on the target rather than on a clamped approximation of it.
    out[-1] = MorphStep(
        t=1.0, degrees=tuple(end.degrees), scale=end, period=end.period,
        label=f"{end.signature} ({end.cardinality})",
        event=out[-1].event,
    )
    return Morph(tuple(out), "tree", start, end)


# --------------------------------------------------------------------------- #
# 3. Voice morph -- move the notes
# --------------------------------------------------------------------------- #
def _best_rotation(source: np.ndarray, target: np.ndarray) -> float:
    """Offset that brings two pitch-class sets closest together.

    The optimum always places some target tone exactly on some source tone, so
    the candidate set is finite and small and the search is exact rather than
    a numerical minimisation.

    Used only to decide *which* tone goes where. The morph itself travels from
    the unrotated source, since applying the rotation to the path would
    transpose the whole scale in the first frame -- an audible jump that has
    nothing to do with the journey being drawn.
    """
    candidates = np.unique(
        np.round(np.concatenate([(target[:, None] - source[None, :]).ravel(),
                                 [0.0]]) % 1.0, 12)
    )
    best_phi, best_cost = 0.0, np.inf
    for phi in candidates:
        d = np.abs(target[:, None] - (source[None, :] + phi))
        d = np.minimum(d % 1.0, 1.0 - (d % 1.0))
        cost = float(d.min(axis=1).sum())
        if cost < best_cost:
            best_phi, best_cost = float(phi), cost
    return best_phi


def _signed_gap(a: float, b: float) -> float:
    """Shortest signed distance from ``a`` to ``b`` around the circle."""
    d = (b - a) % 1.0
    return d if d <= 0.5 else d - 1.0


def voice_morph(
    start: MOSScale,
    end: MOSScale,
    steps: int = 64,
    locate: bool = True,
) -> Morph:
    """Move the notes, not the structure: the path you would actually hear.

    Each tone glides along the shorter way round the circle to its counterpart.
    When the two scales have equal note counts the correspondence is a
    bijection under the best rotation, which is optimal on a circle. When they
    differ, tones of the larger set share a source, so notes split apart on the
    way out or merge on the way in -- the same thing that happens at a landmark
    when a step size shrinks to nothing.

    Unlike the other two strategies this path **leaves the space of well-formed
    scales**: its intermediate pitch sets are generally not well-formed at all.
    That is the interesting part, and ``locate`` measures it.

    Parameters
    ----------
    start, end : MOSScale
    steps : int, default 64
    locate : bool, default True
        Fit each frame back onto the labyrinth to record how far outside it
        strays, in :attr:`MorphStep.wellformedness`. Costs a scale fit per
        frame; turn it off for long morphs.

    Returns
    -------
    Morph
        With :attr:`Morph.voices` populated, so a trajectory draws as
        continuous lines. The voice count is constant across the whole morph,
        even when the two scales have different note counts: the extra voices
        start (or finish) coincident with the tone they split from, which is
        both what splitting sounds like and what keeps the trajectory
        rectangular.

    Examples
    --------
    >>> a = MOSScale.from_signature(5, 2, tuning=12)
    >>> b = MOSScale.from_signature(4, 3, tuning=19)
    >>> m = voice_morph(a, b, steps=9, locate=False)
    >>> len(m), m[0].cardinality, m[-1].cardinality
    (9, 7, 7)

    The endpoints are the scales themselves:

    >>> np.allclose(m[0].degrees, a.degrees) and np.allclose(m[-1].degrees, b.degrees)
    True
    """
    if steps < 2:
        raise ValueError(f"steps must be at least 2, got {steps}")

    src = np.asarray(start.degrees, dtype=float)
    dst = np.asarray(end.degrees, dtype=float)
    phi = _best_rotation(src, dst)
    shifted = (src + phi) % 1.0

    # The pairing is worked out on the rotated copy; the path is then built
    # from the original tones, by index, so no frame is transposed.
    n_src, n_dst = len(shifted), len(dst)
    if n_src == n_dst:
        # Bijection: on a circle the optimal one is a cyclic rotation of the
        # sorted orders, so only n offsets need testing.
        order = np.argsort(shifted)
        s_sorted = shifted[order]
        best_r, best_cost = 0, np.inf
        for r in range(n_src):
            rolled = np.roll(s_sorted, r)
            cost = np.abs([_signed_gap(a, b) for a, b in zip(rolled, dst)]).sum()
            if cost < best_cost:
                best_r, best_cost = r, cost
        src_index = np.roll(order, best_r)
        dst_index = np.arange(n_dst)
        split_or_merge = 0
    elif n_dst > n_src:
        # More tones arriving than leaving: each target tracks its nearest
        # source, so a source claimed by several is one tone splitting in two.
        d = np.abs(dst[:, None] - shifted[None, :])
        d = np.minimum(d, 1.0 - d)
        src_index = d.argmin(axis=1)
        dst_index = np.arange(n_dst)
        split_or_merge = n_dst - len(set(src_index.tolist()))
    else:
        d = np.abs(shifted[:, None] - dst[None, :])
        d = np.minimum(d, 1.0 - d)
        dst_index = d.argmin(axis=1)
        src_index = np.arange(n_src)
        split_or_merge = n_src - len(set(dst_index.tolist()))

    from_, to_ = src[src_index], dst[dst_index]
    gaps = np.array([_signed_gap(a, b) for a, b in zip(from_, to_)])

    out: List[MorphStep] = []
    voices: List[Tuple[int, ...]] = []
    for t in np.linspace(0.0, 1.0, steps):
        # from_ + gaps is to_ by construction, so the endpoints are exact up to
        # rounding; they are snapped anyway so that `m[0].degrees` compares
        # equal to the start scale's rather than merely close to it.
        here = (from_ + t * gaps) % 1.0
        if t == 0.0:
            here = from_.copy()
        elif t == 1.0:
            here = to_.copy()
        order = np.argsort(here)
        degrees = tuple(float(x) for x in here[order])
        period = start.period ** (1.0 - t) * end.period**t

        scale, wf = None, 0.0
        if t == 0.0:
            scale = start
        elif t == 1.0:
            scale = end
        elif locate:
            from biotuner.mos.derive import fit_mos

            try:
                fits = fit_mos(
                    [period**d for d in degrees], period=period,
                    max_cardinality=max(len(degrees) + 2, 8),
                    min_cardinality=max(3, len(degrees)),
                    grid=180, refine=False, complexity_penalty=0.0, top_n=1,
                )
                if fits:
                    wf = fits[0].error_cents
                    if wf < 1e-6:
                        scale = fits[0].scale
            except ValueError:
                wf = float("nan")

        event = None
        if t == 0.0 and split_or_merge:
            event = (f"{split_or_merge} tone(s) "
                     f"{'split' if n_dst > n_src else 'merge'} on the way")
        out.append(
            MorphStep(t=float(t), degrees=degrees, scale=scale, period=period,
                      label=f"{len(degrees)} tones", event=event,
                      wellformedness=wf)
        )
        voices.append(tuple(int(i) for i in order))

    return Morph(tuple(out), "voice", start, end, tuple(voices))


# --------------------------------------------------------------------------- #
# Dispatcher
# --------------------------------------------------------------------------- #
def morph(start: MOSScale, end: MOSScale, strategy: str = "tuning", **kwargs) -> Morph:
    """Journey from one well-formed scale to another.

    Parameters
    ----------
    start, end : MOSScale
    strategy : {'tuning', 'tree', 'voice'}, default 'tuning'
        See the module docstring; they are different journeys, not different
        spellings of one.
    **kwargs
        Passed to the chosen strategy.

    Examples
    --------
    >>> a = MOSScale.from_signature(2, 3, tuning=12)
    >>> b = MOSScale.from_signature(5, 7, tuning=12)
    >>> morph(a, b, "tree").signatures()
    ['2L3s', '3L2s', '5L2s', '5L7s']
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"strategy must be one of {STRATEGIES}, got {strategy!r}"
        )
    return {"tuning": tuning_morph, "tree": tree_morph, "voice": voice_morph}[
        strategy
    ](start, end, **kwargs)


# --------------------------------------------------------------------------- #
# Seeing the journey
# --------------------------------------------------------------------------- #
_PALETTE = {
    "light": dict(bg="#ffffff", fg="#22252b", muted="#6a6f78", grid="#dfe1e4",
                  scenery="#c2c6cc", path="#C73E1D", cmap="viridis"),
    "noir": dict(bg="#0b0b0d", fg="#f2f2f0", muted="#8a8f96", grid="#1c1c20",
                 scenery="#3d434c", path="#ff7a55", cmap="magma"),
}


def _pal(name: str) -> Dict[str, str]:
    if name not in _PALETTE:
        raise ValueError(
            f"palette must be one of {tuple(_PALETTE)}, got {name!r}"
        )
    return _PALETTE[name]



def _dim_labyrinth(ax, pal: Dict[str, str], alpha: float = 0.55) -> None:
    """Push the labyrinth back so the journey reads on top of it.

    ``plot_labyrinth`` draws for its own sake on a light ground, colouring each
    ring differently. Here it is scenery, and that per-ring colour is a rival
    to the path rather than information -- the ring is already legible from the
    radius. So the arcs are flattened to one neutral tone and only their
    *weight* survives, which is the part that still means something: a thick
    arc is a coherent tuning range, a thin one is not.
    """
    ax.set_facecolor(pal["bg"])
    for artist in list(ax.lines) + list(ax.collections):
        artist.set_alpha((artist.get_alpha() or 1.0) * alpha)
        if hasattr(artist, "set_color"):
            artist.set_color(pal["scenery"])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(pal["muted"])
        label.set_fontsize(7)
    ax.spines["polar"].set_color(pal["grid"])
    ax.grid(color=pal["grid"], alpha=0.45, lw=0.4)


def plot_morph_path(
    m: Morph,
    *,
    max_cardinality: Optional[int] = None,
    palette: str = "light",
    show_events: bool = True,
    column: bool = True,
    colorbar: bool = True,
    ax=None,
    figsize: Tuple[float, float] = (8.6, 8.6),
):
    """The journey drawn on the labyrinth itself.

    Angle is the generator and radius the note count, so the shape of the path
    says which kind of journey it was at a glance: a ``tuning`` morph slides
    along one ring, a ``tree`` morph climbs between rings, and a ``voice``
    morph breaks wherever it has left the space of well-formed scales
    altogether.

    A morph's own scale sits on a single ring, which on the full labyrinth is a
    short arc and easy to miss. With ``column`` the figure also draws the other
    cardinalities the generator is well-formed at -- the radial column of
    :func:`~biotuner.mos.theory.mos_cardinalities` -- so the journey reads as a
    wedge of the labyrinth rather than a speck on it. The wedge is not
    decoration: its strands start and stop at the landmarks, so you can watch
    the column *branch*. Sliding a fifth flat past 7-EDO, the outer strand
    leaves ring 12 and picks up rings 9 and 16 -- the same event that flips
    ``5L2s`` into ``2L5s``, seen from outside the scale.

    Parameters
    ----------
    m : Morph
    max_cardinality : int, optional
        Outermost ring. Defaults to 18, or further out if the path needs it.
    palette : {'light', 'noir'}, default 'light'
    show_events : bool, default True
        Label the structural events -- a signature flipping, the note count
        changing. Landmark crossings get a tick on the path but no text: on a
        tuning morph three or four of them land within a couple of frames of
        each other and the labels would sit on top of one another.
        :meth:`Morph.summary` lists all of them with their exact ``t``.
    column : bool, default True
        Draw the generator's whole column of well-formed cardinalities behind
        the path.
    colorbar : bool, default True
        Show the ``t`` scale. Turn it off when the panel sits next to a
        trajectory plot that already has ``t`` on its x-axis.
    ax : matplotlib polar axes, optional
    figsize : tuple

    Returns
    -------
    (fig, ax)

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> a = MOSScale.from_signature(5, 2, tuning=31)
    >>> fig, ax = plot_morph_path(tuning_morph(a, a.inverse, steps=17))
    >>> ax.name
    'polar'
    >>> plt.close(fig)
    """
    from biotuner.mos.plotting import plot_labyrinth
    from biotuner.mos.theory import mos_cardinalities

    pal = _pal(palette)
    path = m.labyrinth_path()
    placed = [q for q in path if q is not None]
    if not placed:
        raise ValueError(
            "this morph never lands on a well-formed scale, so there is no "
            "path to draw on the labyrinth; use plot_morph_trajectory instead"
        )
    if max_cardinality is None:
        max_cardinality = max(18, max(c for _, c in placed) + 2)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize,
                               subplot_kw={"projection": "polar"})
    else:
        fig = ax.figure
    fig.patch.set_facecolor(pal["bg"])
    plot_labyrinth(max_cardinality, ax=ax, label="cents")
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    ax.set_title("")
    _dim_labyrinth(ax, pal)

    def _runs(points):
        """Split at the gaps, so a break reads as a break."""
        out, run = [], []
        for q in points:
            if q is None:
                if len(run) > 1:
                    out.append(run)
                run = []
            else:
                run.append(q)
        if len(run) > 1:
            out.append(run)
        return out

    if column:
        # A soft wedge over the generator range the journey covers, so the eye
        # finds the slice of the labyrinth in question before reading detail.
        lo = min(g for g, _ in placed)
        hi = max(g for g, _ in placed)
        edge = np.linspace(2 * math.pi * lo, 2 * math.pi * hi,
                           max(2, int(240 * (hi - lo)) + 2))
        ax.fill_between(edge, 0, max_cardinality, color=pal["path"],
                        alpha=0.06, lw=0, zorder=2)

        # One strand per cardinality the generator is well-formed at. A strand
        # stops where the morph crosses the landmark that dissolves it.
        strands: Dict[int, List[Optional[Tuple[float, int]]]] = {}
        for q in path:
            here = set() if q is None else set(
                mos_cardinalities(q[0], max_cardinality)
            )
            for c in here:
                strands.setdefault(c, [])
            for c, pts in strands.items():
                pts.append((q[0], c) if (q is not None and c in here) else None)
        for c, pts in strands.items():
            for r in _runs(pts):
                ax.plot([2 * math.pi * g for g, _ in r], [c for _, c in r],
                        "-", color=pal["path"], lw=2.0, alpha=0.55, zorder=7,
                        solid_capstyle="round")

    for r in _runs(path):
        theta = [2 * math.pi * g for g, _ in r]
        radius = [c for _, c in r]
        ax.plot(theta, radius, "-", color=pal["path"], lw=7.0, alpha=0.20,
                zorder=8, solid_capstyle="round")
        ax.plot(theta, radius, "-", color=pal["path"], lw=2.4, alpha=0.95,
                zorder=9, solid_capstyle="round")

    # Where the journey left the labyrinth, join the two banks with a dotted
    # line. Without it an off-scale morph draws as two lonely markers and says
    # nothing about what happened in between; with it, the gap is the point.
    gaps = []
    last = None
    for k, q in enumerate(path):
        if q is not None:
            if last is not None and k - last > 1:
                gaps.append((path[last], q, k - last - 1))
            last = k
    for a, b, width in gaps:
        ax.plot([2 * math.pi * a[0], 2 * math.pi * b[0]], [a[1], b[1]],
                ":", color=pal["path"], lw=1.6, alpha=0.6, zorder=8)
    if gaps:
        a, b, width = max(gaps, key=lambda g: g[2])
        ax.annotate(
            f"off the labyrinth for {width} frames",
            (math.pi * (a[0] + b[0]), 0.5 * (a[1] + b[1])),
            textcoords="offset points", xytext=(0, 10), fontsize=8.0,
            ha="center", color=pal["path"], alpha=0.85, zorder=13,
            path_effects=[pe.withStroke(linewidth=2.4, foreground=pal["bg"])],
        )

    ts = [step.t for step, q in zip(m.steps, path) if q is not None]
    sc = ax.scatter(
        [2 * math.pi * g for g, _ in placed], [c for _, c in placed],
        c=ts, cmap=pal["cmap"], s=52, zorder=10,
        edgecolor=pal["bg"], linewidth=0.8, vmin=0.0, vmax=1.0,
    )
    for scale, marker, size in ((m.start, "o", 165), (m.end, "s", 145)):
        ax.scatter([2 * math.pi * scale.generator], [scale.cardinality],
                   marker=marker, s=size, facecolor="none",
                   edgecolor=pal["path"], linewidth=2.0, zorder=11)

    if show_events:
        marked = 0
        for step, q in zip(m.steps, path):
            if not (step.event and q is not None):
                continue
            structural = any(k in step.event for k in
                             ("becomes", "notes", "start", "arrive"))
            if not structural:
                # A landmark crossing. Tick it on the path and leave the
                # wording to the summary rather than stacking labels.
                ax.plot([2 * math.pi * q[0]], [q[1]], marker="|", ms=14,
                        color=pal["path"], alpha=0.85, zorder=12)
                continue
            ax.annotate(
                step.event.split(":")[0].strip(),
                (2 * math.pi * q[0], q[1]),
                textcoords="offset points",
                xytext=(14, 14 if marked % 2 == 0 else -20),
                fontsize=8.0, color=pal["path"], zorder=13,
                path_effects=[pe.withStroke(linewidth=2.4,
                                            foreground=pal["bg"])],
            )
            marked += 1

    if colorbar:
        cbar = fig.colorbar(sc, ax=ax, pad=0.10, shrink=0.68)
        cbar.set_label("t  (0 = start, 1 = end)", color=pal["fg"], fontsize=9)
        cbar.ax.tick_params(colors=pal["muted"], labelsize=8)
        cbar.outline.set_edgecolor(pal["grid"])
    route = "  \u2192  ".join(m.signatures())
    ax.set_title(
        m.strategy + " morph\n" + route,
        fontsize=12, color=pal["fg"], pad=20,
    )
    return fig, ax

def plot_morph_trajectory(
    m: Morph,
    *,
    palette: str = "light",
    show_events: bool = True,
    show_wellformedness: bool = True,
    ax=None,
    figsize: Tuple[float, float] = (11.0, 6.4),
):
    """Every tone's pitch across the journey -- the voice-leading picture.

    One line per voice, cents against ``t``. This is where the three strategies
    look most different: a ``tuning`` morph fans its lines smoothly, a ``tree``
    morph shows lines appearing and vanishing as the note count changes, and a
    ``voice`` morph runs each line straight to its target.

    When the morph ever leaves the space of well-formed scales, a lower panel
    tracks how far outside it is.

    Parameters
    ----------
    m : Morph
    palette : {'light', 'noir'}, default 'light'
    show_events : bool, default True
    show_wellformedness : bool, default True
        Add the lower panel when there is anything to show in it.
    ax : matplotlib axes, optional
        Supplying one suppresses the lower panel.
    figsize : tuple

    Returns
    -------
    (fig, axes)

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> a = MOSScale.from_signature(5, 2, tuning=12)
    >>> b = MOSScale.from_signature(4, 3, tuning=19)
    >>> fig, axes = plot_morph_trajectory(voice_morph(a, b, steps=17, locate=False))
    >>> plt.close(fig)
    """
    pal = _pal(palette)
    strays = [s.wellformedness for s in m.steps if not s.is_well_formed]
    want_lower = show_wellformedness and bool(strays) and ax is None

    if ax is None:
        if want_lower:
            fig, axes = plt.subplots(
                2, 1, figsize=figsize, sharex=True,
                gridspec_kw={"height_ratios": [2.6, 1.0], "hspace": 0.12},
            )
        else:
            fig, single = plt.subplots(figsize=figsize)
            axes = np.array([single])
    else:
        fig, axes = ax.figure, np.atleast_1d(ax)
    top = axes[0]
    fig.patch.set_facecolor(pal["bg"])
    for a in axes:
        a.set_facecolor(pal["bg"])

    traj = m.trajectory()
    pc = m.period_cents
    ts = np.array([s.t for s in m.steps])
    cmap = plt.get_cmap(pal["cmap"])
    n_voices = traj.shape[1]
    for v in range(n_voices):
        y = traj[:, v] * pc
        good = ~np.isnan(y)
        if good.sum() < 2:
            continue
        # A voice that wraps the period would otherwise draw a spurious
        # vertical line straight across the plot.
        yy = y.copy()
        jump = np.abs(np.diff(yy)) > pc / 2
        yy[1:][jump] = np.nan
        top.plot(ts, yy, "-", color=cmap(0.15 + 0.7 * v / max(1, n_voices - 1)),
                 lw=1.9, alpha=0.95)

    for scale, x in ((m.start, 0.0), (m.end, 1.0)):
        top.scatter([x] * scale.cardinality, [d * pc for d in scale.degrees],
                    s=26, color=pal["fg"], zorder=5)

    if show_events:
        # Events cluster -- a signature flip and the EDO it passes through
        # can be one frame apart -- so drop each label to a lower shelf than
        # its neighbour whenever the two would otherwise be printed on top of
        # each other.
        shelf, prev_t = 0, -1.0
        for step in m.steps:
            if not step.event:
                continue
            top.axvline(step.t, color=pal["path"], lw=1.0, ls="--", alpha=0.7)
            shelf = (shelf + 1) % 3 if step.t - prev_t < 0.08 else 0
            prev_t = step.t
            top.annotate(step.event.split(":")[0].strip(),
                         (step.t, pc * (1.0 - 0.10 * shelf)),
                         rotation=90, fontsize=7, color=pal["path"],
                         va="top", ha="right",
                         textcoords="offset points", xytext=(-3, -4))

    top.set_ylabel("cents", fontsize=10, color=pal["fg"])
    top.set_ylim(0, pc)
    top.set_xlim(0, 1)
    top.spines[["top", "right"]].set_visible(False)
    top.set_title(
        f"{m.strategy} morph:  {m.start.signature} → {m.end.signature}   —   "
        f"{m.voice_leading_distance():.0f} c of total motion",
        fontsize=12, color=pal["fg"],
    )

    if want_lower:
        bot = axes[-1]
        y = [s.wellformedness for s in m.steps]
        bot.fill_between(ts, 0, y, color=pal["path"], alpha=0.25)
        bot.plot(ts, y, color=pal["path"], lw=1.6)
        bot.set_ylabel("off-scale (c)", fontsize=9.5, color=pal["fg"])
        bot.set_xlabel("t", fontsize=10, color=pal["fg"])
        bot.spines[["top", "right"]].set_visible(False)
        bot.annotate(
            "how far this frame is from being a well-formed scale",
            (0.5, max(y) if max(y) else 1.0), fontsize=8, color=pal["muted"],
            ha="center", va="top",
        )
    else:
        axes[-1].set_xlabel("t", fontsize=10, color=pal["fg"])
    return fig, axes


def plot_morph_filmstrip(
    m: Morph,
    *,
    n_frames: int = 9,
    style: str = "ring",
    palette: str = "noir",
    figsize: Optional[Tuple[float, float]] = None,
):
    """The scale itself, sampled along the journey.

    Uses :mod:`biotuner.mos.design`, so each frame carries its structure rather
    than only its pitches. Frames that have left the space of well-formed
    scales are drawn as bare polygons, since they have no signature to encode.

    Parameters
    ----------
    m : Morph
    n_frames : int, default 9
    style : str, default 'ring'
        Any :data:`biotuner.mos.design.STYLES` value. ``'ring'`` shows the step
        pattern; ``'chain'`` shows the generator structure, but only exists for
        frames that are genuinely well-formed.
    palette : {'light', 'noir'}, default 'noir'
    figsize : tuple, optional

    Returns
    -------
    (fig, axes)

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> a = MOSScale.from_signature(5, 2, tuning=31)
    >>> fig, axes = plot_morph_filmstrip(tuning_morph(a, a.inverse, steps=33),
    ...                                  n_frames=5)
    >>> len(axes)
    5
    >>> plt.close(fig)
    """
    from biotuner.mos.design import PALETTES, plot_scale_web

    if n_frames < 1:
        raise ValueError(f"n_frames must be at least 1, got {n_frames}")
    pal = _pal(palette)
    design_palette = palette if palette in PALETTES else "light"
    picks = np.unique(np.linspace(0, len(m) - 1, n_frames).round().astype(int))
    figsize = figsize or (2.5 * len(picks), 3.0)
    fig, grid = plt.subplots(1, len(picks), figsize=figsize)
    fig.patch.set_facecolor(pal["bg"])
    axes = list(np.atleast_1d(grid).ravel())

    for idx, ax in zip(picks, axes):
        step = m.steps[idx]
        if step.scale is not None:
            plot_scale_web(step.scale, style, palette=design_palette, ax=ax,
                           title="")
        else:
            # No signature to encode, so draw the bare pitch polygon.
            ang = np.array(step.degrees) * 2 * math.pi
            pts = np.stack([np.sin(ang), np.cos(ang)], axis=1)
            closed = np.vstack([pts, pts[:1]])
            ax.set_facecolor(pal["bg"])
            ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color=pal["grid"],
                                    lw=1.0))
            ax.plot(closed[:, 0], closed[:, 1], "-o", color=pal["path"], lw=1.6,
                    ms=4)
            ax.set_xlim(-1.22, 1.22)
            ax.set_ylim(-1.22, 1.22)
            ax.set_aspect("equal")
            ax.axis("off")
        name = step.scale.signature if step.scale else "off-scale"
        ax.set_title(f"t={step.t:.2f}\n{name}", fontsize=9, color=pal["fg"],
                     pad=8)
    fig.suptitle(
        f"{m.strategy} morph:  {m.start.signature} → {m.end.signature}",
        fontsize=12, color=pal["fg"], y=1.04,
    )
    fig.tight_layout()
    return fig, axes


def plot_morph_comparison(
    start: MOSScale,
    end: MOSScale,
    *,
    steps: int = 64,
    palette: str = "light",
    figsize: Tuple[float, float] = (16.0, 9.5),
    **kwargs,
):
    """All three journeys between the same pair, side by side.

    The top row is each path on the labyrinth, the bottom row the same journey
    as voice leading. Read down a column to see one strategy; read across to
    see how differently the same two scales can be connected.

    Parameters
    ----------
    start, end : MOSScale
    steps : int, default 64
        Frames for the continuous strategies. Ignored by ``tree``, whose length
        is set by the route.
    palette : {'light', 'noir'}, default 'light'
    figsize : tuple
    **kwargs
        Passed to every strategy that accepts them.

    Returns
    -------
    (fig, dict)
        The figure, and the three :class:`Morph` objects by strategy name, so
        their numbers can be reported alongside.

    Notes
    -----
    ``tuning`` needs both scales to have the same note count. When they do not,
    its column is left empty with a note rather than the whole figure failing --
    that limitation is a fact about the strategy and worth seeing.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> a = MOSScale.from_signature(5, 2, tuning=31)
    >>> fig, morphs = plot_morph_comparison(a, a.inverse, steps=25)
    >>> sorted(morphs)
    ['tree', 'tuning', 'voice']
    >>> plt.close(fig)
    """
    pal = _pal(palette)
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(pal["bg"])
    morphs: Dict[str, Morph] = {}
    failures: Dict[str, str] = {}

    for col, name in enumerate(STRATEGIES):
        try:
            if name == "tuning":
                morphs[name] = tuning_morph(start, end, steps=steps)
            elif name == "tree":
                morphs[name] = tree_morph(start, end)
            else:
                morphs[name] = voice_morph(start, end, steps=steps)
        except ValueError as exc:
            failures[name] = str(exc)

    for col, name in enumerate(STRATEGIES):
        top = fig.add_subplot(2, 3, col + 1, projection="polar")
        bot = fig.add_subplot(2, 3, col + 4)
        if name in failures:
            for a in (top, bot):
                a.axis("off")
                a.set_facecolor(pal["bg"])
            top.text(0.5, 0.5, f"{name}\nnot available here",
                     transform=top.transAxes, ha="center", va="center",
                     fontsize=11, color=pal["muted"])
            bot.text(0.5, 0.5, failures[name].split(";")[0], wrap=True,
                     transform=bot.transAxes, ha="center", va="center",
                     fontsize=8, color=pal["muted"])
            continue
        mm = morphs[name]
        try:
            # No colour bar: the panel below already has t on its x-axis, and
            # three identical bars would only eat the width the maps need.
            plot_morph_path(mm, ax=top, palette=palette, show_events=False,
                            colorbar=False)
        except ValueError:
            top.axis("off")
        plot_morph_trajectory(mm, ax=bot, palette=palette,
                              show_wellformedness=False)
        top.set_title(f"{name}", fontsize=13, color=pal["fg"], pad=18)
        bot.set_title(
            f"{mm.voice_leading_distance():.0f} c motion · "
            f"{len(mm)} frames · "
            f"{sum(1 for s in mm if not s.is_well_formed)} off-scale",
            fontsize=9.5, color=pal["muted"],
        )
    fig.suptitle(
        f"Three ways from {start.signature} to {end.signature}",
        fontsize=15, color=pal["fg"], y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig, morphs


def animate_morph(
    m: Morph,
    *,
    style: str = "ring",
    palette: str = "noir",
    interval: int = 80,
    figsize: Tuple[float, float] = (6.0, 6.4),
):
    """The journey as an animation.

    Returns a :class:`matplotlib.animation.FuncAnimation`. Save it with
    ``anim.save('morph.gif', writer='pillow')``, or display it in a notebook
    with ``HTML(anim.to_jshtml())``.

    Parameters
    ----------
    m : Morph
    style : str, default 'ring'
    palette : {'light', 'noir'}, default 'noir'
    interval : int, default 80
        Milliseconds per frame.
    figsize : tuple

    Returns
    -------
    matplotlib.animation.FuncAnimation

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> a = MOSScale.from_signature(5, 2, tuning=31)
    >>> anim = animate_morph(tuning_morph(a, a.inverse, steps=9))
    >>> anim.save.__name__
    'save'
    >>> plt.close("all")
    """
    from matplotlib.animation import FuncAnimation

    pal = _pal(palette)
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(pal["bg"])
    ax.set_facecolor(pal["bg"])
    ax.set_xlim(-1.25, 1.25)
    ax.set_ylim(-1.25, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(plt.Circle((0, 0), 1.0, fill=False, color=pal["grid"], lw=1.0))
    line, = ax.plot([], [], "-", color=pal["path"], lw=2.0)
    dots = ax.scatter([], [], s=34, color=pal["fg"], zorder=3)
    title = ax.set_title("", fontsize=11, color=pal["fg"], pad=10)

    def update(i):
        step = m.steps[i]
        ang = np.array(step.degrees) * 2 * math.pi
        pts = np.stack([np.sin(ang), np.cos(ang)], axis=1)
        closed = np.vstack([pts, pts[:1]])
        line.set_data(closed[:, 0], closed[:, 1])
        dots.set_offsets(pts)
        name = step.scale.signature if step.scale else "off-scale"
        title.set_text(f"{m.strategy}   t={step.t:.2f}   {name}")
        return line, dots, title

    return FuncAnimation(fig, update, frames=len(m), interval=interval,
                         blit=False, repeat=True)


# --------------------------------------------------------------------------- #
# Sound
# --------------------------------------------------------------------------- #
def morph_audio(
    m: Morph,
    fundamental: float = 220.0,
    seconds: float = 10.0,
    sample_rate: int = 44100,
    matched_timbre: bool = False,
    n_partials: int = 6,
) -> np.ndarray:
    """Render a morph as audio: every tone of the scale, sounding throughout.

    Each voice is one continuous sine (or matched-timbre stack) whose frequency
    glides with the morph, so what you hear is the movement itself rather than
    a sequence of chords. Frames are interpolated, so a 64-frame morph does not
    step audibly.

    Parameters
    ----------
    m : Morph
    fundamental : float, default 220.0
    seconds : float, default 10.0
    sample_rate : int, default 44100
    matched_timbre : bool, default False
        Give each tone Dynamic Tonality partials matched to the frame's scale,
        so the timbre tracks the tuning (Milne et al. §6). Costs a partial map
        per frame; audibly smoother on wildly detuned frames.
    n_partials : int, default 6
        Only used when ``matched_timbre``.

    Returns
    -------
    np.ndarray
        Mono float32 in ``[-1, 1]``.

    Examples
    --------
    >>> a = MOSScale.from_signature(5, 2, tuning=12)
    >>> audio = morph_audio(tuning_morph(a, a.inverse, steps=8), seconds=0.5)
    >>> audio.dtype, bool(abs(audio).max() <= 1.0)
    (dtype('float32'), True)
    """
    if seconds <= 0:
        raise ValueError(f"seconds must be positive, got {seconds}")
    n_samples = int(round(seconds * sample_rate))
    traj = m.trajectory()
    n_frames, n_voices = traj.shape
    frame_at = np.linspace(0, n_frames - 1, n_samples)
    lo, hi = np.floor(frame_at).astype(int), np.ceil(frame_at).astype(int)
    blend = frame_at - lo

    periods = np.array([s.period for s in m.steps])
    period_at = periods[lo] * (1 - blend) + periods[hi] * blend

    out = np.zeros(n_samples, dtype=np.float64)
    live = 0
    for v in range(n_voices):
        a, b = traj[lo, v], traj[hi, v]
        if np.isnan(a).all() or np.isnan(b).all():
            continue
        a = np.nan_to_num(a, nan=0.0)
        b = np.nan_to_num(b, nan=0.0)
        deg = a * (1 - blend) + b * blend
        freq = fundamental * period_at**deg
        # Integrate frequency so a glide has no phase discontinuities.
        phase = 2 * np.pi * np.cumsum(freq) / sample_rate
        partials = [(1.0, 1.0)]
        if matched_timbre:
            partials = [(float(h), 1.0 / h) for h in range(1, n_partials + 1)]
        for ratio, amp in partials:
            out += amp * np.sin(phase * ratio)
        live += 1

    if live:
        out /= live * (1.6 if matched_timbre else 1.0)
    # Short fades, so the file does not start or stop with a click.
    edge = min(int(0.02 * sample_rate), n_samples // 4)
    if edge > 1:
        ramp = np.linspace(0.0, 1.0, edge)
        out[:edge] *= ramp
        out[-edge:] *= ramp[::-1]
    peak = np.abs(out).max()
    if peak > 0:
        out = out / peak * 0.92
    return out.astype(np.float32)
