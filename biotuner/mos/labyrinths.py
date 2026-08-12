"""Alternative labyrinths -- other coordinates, other trees, other geometries.

:func:`~biotuner.mos.plotting.plot_labyrinth` draws Milne et al. (2011) Fig. 1,
and in doing so makes three choices that look like necessities but are only
conventions:

1. **angle** is the generator fraction ``g``;
2. **radius** is the node's denominator, i.e. the MOS cardinality;
3. the tree is built by the **mediant**, ``(a+c)/(b+d)``.

Each is replaceable, and replacing them separates two things the ordinary
picture conflates: the *combinatorics* of the Stern-Brocot tree (who brackets
whom, at what depth) and the *arithmetic* of the rationals it contains
(denominators, and where they sit on the line).

- Part A, :func:`minkowski_q`, replaces the angular coordinate with Minkowski's
  question-mark function ``?(x)``, the canonical map that flattens the
  Stern-Brocot tree onto the dyadic rationals.
- Part B, :class:`TreeRule` and :func:`variant_tree`, replaces the mediant with
  any weighted mediant ``(a + wc)/(b + wd)``.  At ``w = φ`` this is Wilson's
  noble mediant, whose nodes are the *least* rational points of their brackets
  -- generators whose MOS series never terminates.
- Part C, :func:`plot_labyrinth_variant` and :func:`plot_farey_tessellation`,
  makes angle and radius swappable functions and adds the hyperbolic picture
  the Stern-Brocot tree is dual to.

The headline finding, drawn out in the docstrings below: with angle ``?(g)``
and radius = tree depth the labyrinth degenerates into perfectly regular
concentric rings.  All the visual structure of the ordinary labyrinth -- the
spiral arms, the dense diatonic neighbourhood, the empty regions near the
period -- comes from the **arithmetic** coordinate (the denominator), not the
combinatorial one.

Relation to :func:`biotuner.harmonic_geometry.fractal.stern_brocot_tree`
------------------------------------------------------------------------
That function also offers a ``layout='hyperbolic'``, but it is a *tree drawing*
placed on a disk: nodes are positioned by in-order traversal index (angle) and
depth (radius), and the edges are straight parent-child links.  Nothing about
it is hyperbolic beyond the disk-shaped canvas, and it enumerates all positive
rationals from the bounds ``0/1`` and ``1/0``.

:func:`plot_farey_tessellation` here draws the actual Farey tessellation of the
hyperbolic plane -- the ideal triangulation the Stern-Brocot tree is the dual
of.  Its vertices are the rationals themselves, sitting *on* the boundary
circle at infinity, and its edges are true geodesics (circular arcs orthogonal
to that circle), computed rather than approximated by chords.

References
----------
Milne, A.J., Carlé, M., Sethares, W.A., Noll, T., Holland, S. (2011).
Scratching the Scale Labyrinth. In *Mathematics and Computation in Music*,
LNAI 6726, 180--195.

Minkowski, H. (1904). Zur Geometrie der Zahlen. *Verhandlungen des III.
Internationalen Mathematiker-Kongresses*, 164--173.

Wilson, E. (1975). *Letter to Chalmers pertaining to Moments of Symmetry /
Tanabe Cycle*.  (The noble mediant and the "metallic" generators.)

Series, C. (1985). The modular surface and continued fractions.
*Journal of the London Mathematical Society* 31, 69--80.  (Farey tessellation
as the geometric home of continued fractions.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from biotuner.mos import theory as T
from biotuner.mos.plotting import (
    GRIDC,
    HIGHLIGHT_COLORS,
    INK,
    MUTED,
    RING_CMAP,
    SIGNAL_COLOR,
)

__all__ = [
    # Part A
    "minkowski_q",
    # Part B
    "PHI",
    "TreeRule",
    "MEDIANT",
    "NOBLE",
    "metallic",
    "weighted",
    "METALLIC_MEANS",
    "METALLIC_NAMES",
    "VariantNode",
    "variant_tree",
    "variant_walk",
    # Part C
    "ANGLE_RULES",
    "RADIUS_RULES",
    "DISK_CENTRES",
    "plot_labyrinth_variant",
    "plot_farey_tessellation",
]

#: The golden ratio, the weight of the noble mediant.
PHI = (1.0 + 5.0**0.5) / 2.0

#: Stop the ``?`` series once the partial-denominator sum passes this: the
#: remaining tail is bounded by ``2**-_Q_MAX_SUM``, already below float epsilon.
_Q_MAX_SUM = 64

#: Below this a float remainder is treated as an exact continued-fraction
#: terminus rather than the rounding noise it almost always is.
_Q_TOL = 1e-13


# --------------------------------------------------------------------------- #
# Part A -- Minkowski's question mark
# --------------------------------------------------------------------------- #
def minkowski_q(x: Union[Fraction, float, int]) -> float:
    """Minkowski's question-mark function ``?(x)`` on ``[0, 1]``.

    For ``x = [0; a1, a2, …]`` in continued-fraction form,

    .. math:: ?(x) = 2 \\sum_{n \\ge 1} (-1)^{n+1} 2^{-(a_1 + \\cdots + a_n)}

    ``?`` is the unique increasing homeomorphism of ``[0, 1]`` that sends the
    Stern-Brocot tree onto the dyadic tree: the node at depth ``d`` goes to an
    odd multiple of ``2**-(d+1)``.  Equivalently it replaces *arithmetic*
    position (where a rational sits on the line) with *combinatorial* position
    (where it sits in the tree).

    Parameters
    ----------
    x : Fraction, float or int
        A point of ``[0, 1]``.  A :class:`~fractions.Fraction` takes the exact
        path -- the continued fraction comes from the Euclidean algorithm and
        no float ever enters the expansion.  A float is expanded by the Gauss
        map, which is only as good as the input's own precision.

    Returns
    -------
    float

    Notes
    -----
    **What ``?`` equalises.**  A Stern-Brocot bracket at depth ``d`` has
    ``?``-width *exactly* ``2**-d``, whatever its denominators are.  That is
    the whole content of the function for present purposes, and it is what
    :func:`plot_labyrinth_variant` turns into perfectly regular rings.

    **``?`` does not decompress the crowded diatonic region.**  Worth stating
    because it is the obvious thing to hope for and it is false.  Measured on
    the whole Stern-Brocot tree to denominator 18 (101 nodes):

    ====================  ==============  =================
    window                nodes by ``g``  nodes by ``?(g)``
    ====================  ==============  =================
    0.55 -- 0.62          8               3
    0.95 -- 1.00          0               18
    0.45 -- 0.55          9               13
    ====================  ==============  =================

    The diatonic neighbourhood ``0.55 -- 0.62`` holds ``3/5``, ``4/7``,
    ``5/9``, ``7/12``, ``8/13``, ``9/16``, ``10/17`` and ``11/18`` by
    generator, and only ``4/7``, ``7/12`` and ``10/17`` by ``?``.  It gets
    *less* room, not more.

    What fills the gained window ``0.95 -- 1.00`` is the tree's right spiral
    arm.  ``?`` sends ``k/(k+1)`` to ``1 - 2**-k`` where the generator
    coordinate puts it at ``1 - 1/(k+1)``: the arm converges on the period
    geometrically instead of harmonically, so from ``5/6`` onward every one of
    its nodes is past 0.95, and 18 nodes land in a window that was empty.
    Equalising by depth is not the same as equalising by denominator, and
    denominator is what a musician is choosing.

    Examples
    --------
    >>> minkowski_q(Fraction(1, 2)), minkowski_q(Fraction(1, 3))
    (0.5, 0.25)
    >>> minkowski_q(Fraction(2, 3)), minkowski_q(Fraction(7, 12))
    (0.75, 0.59375)
    >>> minkowski_q(0.0), minkowski_q(1.0)
    (0.0, 1.0)

    The tree node at depth ``d`` lands on a dyadic rational of denominator
    ``2**(d+1)`` -- ``3/8`` is at depth 2, and ``?`` sends it to ``3/8``'s
    dyadic slot three levels down:

    >>> Fraction(minkowski_q(Fraction(3, 8))).limit_denominator(1024)
    Fraction(5, 16)

    Symmetric about ``1/2``, like the labyrinth itself:

    >>> minkowski_q(Fraction(4, 7)) + minkowski_q(Fraction(3, 7))
    1.0
    """
    if isinstance(x, Fraction):
        p, q = x.numerator, x.denominator
        if not 0 <= p <= q:
            raise ValueError(
                f"minkowski_q is defined on [0, 1]; got {x} = {float(x)!r}"
            )
        total, sign, out = 0, 1.0, 0.0
        while p:
            a = q // p
            total += a
            if total > _Q_MAX_SUM:
                break
            out += sign * 2.0**-total
            sign = -sign
            p, q = q - a * p, p
        return 2.0 * out

    v = float(x)
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"minkowski_q is defined on [0, 1]; got {v!r}")
    total, sign, out = 0, 1.0, 0.0
    while v > _Q_TOL:
        v = 1.0 / v
        a = math.floor(v)
        v -= a
        total += int(a)
        if total > _Q_MAX_SUM:
            break
        out += sign * 2.0**-total
        sign = -sign
    return 2.0 * out


# --------------------------------------------------------------------------- #
# Part B -- alternative trees
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TreeRule:
    """A weighted-mediant rule: ``(a + w·c) / (b + w·d)`` for ``a/b`` and ``c/d``.

    Replacing the mediant replaces the tree, but *not* the bracketing
    combinatorics: :func:`variant_tree` always subdivides a bracket at its
    ordinary mediant, because that is the only split point that keeps both
    halves Farey-neighbour pairs.  The rule decides where inside the bracket
    the node's *value* is reported, and so which generator the node names.

    Attributes
    ----------
    name : str
        Used in plot titles and reprs.
    weight : float
        Must be positive.  ``1`` is the mediant; ``φ`` the noble mediant;
        ``w → 0`` walks the value onto the left endpoint, ``w → ∞`` onto the
        right one.

    Examples
    --------
    >>> MEDIANT(Fraction(1, 2), Fraction(3, 5))
    0.5714285714285714
    >>> round(NOBLE(Fraction(1, 2), Fraction(3, 5)), 7)
    0.5801787
    >>> float(T.mediant(Fraction(1, 2), Fraction(3, 5)))
    0.5714285714285714
    """

    name: str
    weight: float

    def __post_init__(self) -> None:
        if not self.weight > 0.0 or not math.isfinite(self.weight):
            raise ValueError(
                f"a tree rule's weight must be finite and positive, got "
                f"{self.weight!r} for rule {self.name!r}"
            )

    def __call__(self, left: Fraction, right: Fraction) -> float:
        w = self.weight
        num = left.numerator + w * right.numerator
        den = left.denominator + w * right.denominator
        return num / den


#: The ordinary mediant.  Reproduces the Stern-Brocot tree exactly.
MEDIANT = TreeRule("mediant", 1.0)

#: Wilson's noble mediant.  Its nodes are the points of each bracket furthest
#: from every simple rational -- the limits of the infinite ``LRLRLR…`` paths --
#: so they are generators whose MOS series never terminates.
NOBLE = TreeRule("noble", PHI)

#: The metallic means ``(k + √(k² + 4)) / 2`` for ``k = 1…4``.  Each is the
#: limit of a Fibonacci-like recurrence ``x_{n+1} = k·x_n + x_{n-1}`` and has
#: the all-``k`` continued fraction ``[k; k, k, …]``.
METALLIC_MEANS: Dict[int, float] = {
    k: (k + math.sqrt(k * k + 4.0)) / 2.0 for k in range(1, 5)
}

#: Conventional names of the metallic means.  ``golden``, ``silver`` and
#: ``bronze`` are standard; ``copper`` for ``k = 4`` is not, and is offered only
#: so the dictionary is total over :data:`METALLIC_MEANS`.
METALLIC_NAMES: Dict[int, str] = {1: "golden", 2: "silver", 3: "bronze", 4: "copper"}


def metallic(k: int) -> TreeRule:
    """Tree rule weighted by the ``k``-th metallic mean ``(k + √(k²+4)) / 2``.

    ``k = 1`` returns :data:`NOBLE` itself (the golden mean); ``k = 2`` is the
    silver mean ``1 + √2``; larger ``k`` pull the node value toward the
    bracket's right endpoint.  Every metallic mean is a quadratic irrational
    with a constant continued fraction, so every node of the tree is irrational
    and every node value generates an unterminating MOS series.

    Notes
    -----
    Seen through :func:`minkowski_q` the ``k``-th metallic tree is a *constant
    ratio* subdivision: it cuts every bracket's dyadic interval at exactly
    ``2**k / (2**k + 1)`` of the way across -- ``2/3`` golden, ``4/5`` silver,
    ``8/9`` bronze -- against the mediant's ``1/2``.  The all-``k`` continued
    fraction tail contributes a geometric series to ``?``, and its sum does not
    depend on where in the tree the bracket sits.

    Examples
    --------
    >>> metallic(1) is NOBLE
    True
    >>> round(metallic(2).weight, 9), metallic(2).name
    (2.414213562, 'silver')

    The constant split ratio, on the bracket ``(1/2, 3/5)`` at depth 3 -- whose
    ``?``-interval is ``(0.5, 0.625)``, of width ``2**-3``:

    >>> lo, hi = Fraction(1, 2), Fraction(3, 5)
    >>> for k in (1, 2, 3):
    ...     v = minkowski_q(metallic(k)(lo, hi))
    ...     print(k, round((v - 0.5) / 0.125, 9))
    1 0.666666667
    2 0.8
    3 0.888888889
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError(f"metallic index k must be an integer >= 1, got {k!r}")
    if k == 1:
        return NOBLE
    name = METALLIC_NAMES.get(k, f"metallic-{k}")
    return TreeRule(name, (k + math.sqrt(k * k + 4.0)) / 2.0)


def weighted(w: float) -> TreeRule:
    """Tree rule with an arbitrary positive weight.

    Examples
    --------
    >>> r = weighted(2.0)
    >>> r.name, round(r(Fraction(0, 1), Fraction(1, 1)), 6)
    ('w=2', 0.666667)
    """
    return TreeRule(f"w={float(w):g}", float(w))


@dataclass(frozen=True)
class VariantNode:
    """One node of a weighted-mediant tree over ``(0, 1)``.

    Attributes
    ----------
    left, right : Fraction
        The bracketing Farey pair, ``left < right``, with ``|ad - bc| == 1``.
        This pair is the same under every rule -- see :func:`variant_tree`.
    value : float
        Where ``rule`` places the node inside the bracket.  Under
        :data:`MEDIANT` this is ``float(mediant(left, right))``; under any other
        rule it is irrational.
    depth : int
        Tree levels descended; ``0`` for the root bracket ``(0/1, 1/1)``.
    turn : str
        ``'L'`` or ``'R'``, the branch taken to reach this node; ``''`` at the
        root.
    rule : TreeRule
    """

    left: Fraction
    right: Fraction
    value: float
    depth: int
    turn: str
    rule: TreeRule

    @property
    def mediant(self) -> Fraction:
        """The ordinary mediant of the bracket -- the point the tree splits at."""
        return T.mediant(self.left, self.right)

    @property
    def cardinality(self) -> Optional[int]:
        """``left.denominator + right.denominator``: the MOS cardinality here.

        This is the denominator of the *ordinary* mediant, and it is the subtle
        point of the whole module.  When ``rule`` is not :data:`MEDIANT` the
        node's own ``value`` is irrational and has no denominator at all, so it
        is not "an equal division into ``N`` steps" the way a Stern-Brocot node
        is.  What survives is the bracket: ``(left, right)`` is still the pair
        of equal temperaments between which one MOS pair keeps its identity,
        and that MOS still has ``b + d`` notes (Milne et al. §3 -- the
        sub-range ``(left, mediant)`` hosts ``b`` large and ``d`` small steps,
        the sub-range ``(mediant, right)`` the reverse).

        So ``cardinality`` should be read as *the cardinality of the MOS whose
        tuning range this node splits*, not as *the size of the equal division
        this node is*.  Under :data:`MEDIANT` the two readings coincide, which
        is exactly why the distinction is invisible in the ordinary labyrinth.

        ``None`` when the bracket is not a Farey-neighbour pair, in which case
        the sum has no MOS meaning.  Nodes produced by :func:`variant_tree` and
        :func:`variant_walk` never hit this case; a hand-built
        :class:`VariantNode` can.

        Examples
        --------
        >>> n = VariantNode(Fraction(1, 2), Fraction(3, 5), 0.58, 3, 'R', NOBLE)
        >>> n.cardinality, n.mediant
        (7, Fraction(4, 7))
        >>> VariantNode(Fraction(1, 3), Fraction(3, 5), 0.5, 0, '', NOBLE).cardinality \
is None
        True
        """
        if not T.is_farey_neighbor(self.left, self.right):
            return None
        return self.left.denominator + self.right.denominator

    def signature(self) -> Optional[Tuple[int, int]]:
        """``(b, d)`` -- the step counts of the MOS in the lower sub-range.

        The upper sub-range hosts the inverse, ``(d, b)``.  ``None`` when
        :attr:`cardinality` is.
        """
        if self.cardinality is None:
            return None
        return (self.left.denominator, self.right.denominator)


#: Refusing to build more than this many nodes when only ``max_depth`` bounds
#: the tree.  ``2**(d+1) - 1`` nodes at depth ``d``; 20 is already a million.
_MAX_UNBOUNDED_DEPTH = 20


def variant_tree(
    rule: TreeRule = MEDIANT,
    max_depth: Optional[int] = 8,
    max_cardinality: Optional[int] = None,
) -> List[VariantNode]:
    """Every node of ``rule``'s tree over ``(0, 1)``, breadth-first.

    The recursion is on *brackets*, not on values: the children of
    ``(left, right)`` are always ``(left, mediant)`` and ``(mediant, right)``
    with ``mediant`` the ordinary one.  That is forced -- the ordinary mediant
    is the only interior point whose two halves are again Farey-neighbour pairs
    -- and it is what makes the alternative trees comparable: they share their
    combinatorics and differ only in where each bracket's node value sits.

    Parameters
    ----------
    rule : TreeRule, default :data:`MEDIANT`
    max_depth : int or None, default 8
        Deepest level to expand, ``0`` being the root.  ``None`` means
        unbounded, which requires ``max_cardinality`` to keep the tree finite.
    max_cardinality : int, optional
        Drop (and do not expand) any bracket whose cardinality exceeds this.
        With ``max_depth=None`` this alone bounds the tree, exactly as in
        :func:`~biotuner.mos.theory.sb_tree_nodes`.

    Returns
    -------
    list of VariantNode

    Examples
    --------
    Bounded by depth the tree is complete: ``2**(d+1) - 1`` nodes.

    >>> len(variant_tree(NOBLE, max_depth=8))
    511

    Bounded by cardinality and with the mediant rule, it *is* the Stern-Brocot
    tree -- same nodes, same brackets, same depths:

    >>> mine = variant_tree(MEDIANT, max_depth=None, max_cardinality=12)
    >>> theirs = T.sb_tree_nodes(12)
    >>> [(n.left, n.right, n.depth) for n in mine] == \
[(n.left, n.right, n.depth) for n in theirs]
    True

    The noble tree has the same brackets and different values:

    >>> noble = variant_tree(NOBLE, max_depth=None, max_cardinality=12)
    >>> [(n.left, n.right) for n in noble] == [(n.left, n.right) for n in mine]
    True
    >>> round(mine[0].value, 6), round(noble[0].value, 6)
    (0.5, 0.618034)
    """
    if max_depth is None and max_cardinality is None:
        raise ValueError(
            "variant_tree needs a bound: pass max_depth, max_cardinality, or "
            "both; got neither"
        )
    if max_depth is None:
        max_depth = 1 << 30
    elif max_depth < 0:
        raise ValueError(f"max_depth must be >= 0 or None, got {max_depth!r}")
    elif max_depth > _MAX_UNBOUNDED_DEPTH and max_cardinality is None:
        raise ValueError(
            f"max_depth={max_depth} with no max_cardinality would build "
            f"2**{max_depth + 1} - 1 nodes; bound the tree by cardinality or "
            f"keep max_depth <= {_MAX_UNBOUNDED_DEPTH}"
        )
    if max_cardinality is not None and max_cardinality < 2:
        return []

    out: List[VariantNode] = []
    frontier: List[Tuple[Fraction, Fraction, int, str]] = [
        (Fraction(0, 1), Fraction(1, 1), 0, "")
    ]
    while frontier:
        nxt: List[Tuple[Fraction, Fraction, int, str]] = []
        for lo, hi, depth, turn in frontier:
            card = lo.denominator + hi.denominator
            if max_cardinality is not None and card > max_cardinality:
                continue
            out.append(
                VariantNode(
                    left=lo,
                    right=hi,
                    value=rule(lo, hi),
                    depth=depth,
                    turn=turn,
                    rule=rule,
                )
            )
            if depth < max_depth:
                med = T.mediant(lo, hi)
                nxt.append((lo, med, depth + 1, "L"))
                nxt.append((med, hi, depth + 1, "R"))
        frontier = nxt
    return out


def variant_walk(
    x: float, rule: TreeRule = MEDIANT, max_depth: int = 32
) -> Iterator[VariantNode]:
    """Walk ``rule``'s tree toward ``x``, as :func:`~biotuner.mos.theory.sb_walk` does.

    Branching is on the *ordinary mediant*, not on the rule's value.  It has to
    be: the bracket must keep containing ``x``, and only the mediant splits it
    into two brackets whose union is the original.  So the sequence of brackets
    is ``x``'s ordinary Stern-Brocot path -- its MOS cardinalities are
    unchanged -- and what the rule supplies is a different *representative
    generator* inside each bracket along the way.

    Parameters
    ----------
    x : float
        Target generator fraction in ``(0, 1)``.
    rule : TreeRule, default :data:`MEDIANT`
    max_depth : int, default 32

    Yields
    ------
    VariantNode

    Examples
    --------
    The fifth's Pythagorean series, whichever rule labels it:

    >>> g = math.log2(3 / 2)
    >>> [n.cardinality for n in variant_walk(g, max_depth=8)]
    [2, 3, 5, 7, 12, 17, 29, 41, 53]
    >>> [n.cardinality for n in variant_walk(g, NOBLE, max_depth=8)]
    [2, 3, 5, 7, 12, 17, 29, 41, 53]

    The values differ, though -- the noble walk names the least-rational
    generator of each bracket rather than the equal temperament that bounds it:

    >>> [round(n.value, 4) for n in variant_walk(g, max_depth=4)]
    [0.5, 0.6667, 0.6, 0.5714, 0.5833]
    >>> [round(n.value, 4) for n in variant_walk(g, NOBLE, max_depth=4)]
    [0.618, 0.7236, 0.618, 0.5802, 0.5867]
    """
    if not 0.0 < x < 1.0:
        raise ValueError(
            f"target must lie strictly in (0, 1), got {x!r}; use "
            "theory.fold_generator() or theory.generator_fraction() first"
        )
    lo, hi = Fraction(0, 1), Fraction(1, 1)
    turn = ""
    for depth in range(max_depth + 1):
        yield VariantNode(
            left=lo, right=hi, value=rule(lo, hi), depth=depth, turn=turn, rule=rule
        )
        med = T.mediant(lo, hi)
        m = float(med)
        if x == m:
            return
        if x < m:
            hi, turn = med, "L"
        else:
            lo, turn = med, "R"


# --------------------------------------------------------------------------- #
# Part C -- layouts
# --------------------------------------------------------------------------- #
#: Angular coordinate, as a function of a generator value in ``[0, 1]``.
#:
#: ``'generator'`` is the ordinary labyrinth's angle; ``'minkowski'`` replaces
#: arithmetic position with combinatorial position (see :func:`minkowski_q`).
ANGLE_RULES: Dict[str, Callable[[Union[Fraction, float]], float]] = {
    "generator": lambda v: 2.0 * math.pi * float(v),
    "minkowski": lambda v: 2.0 * math.pi * minkowski_q(v),
}

#: Radial coordinate, as a function of a :class:`VariantNode`.
#:
#: ``'cardinality'`` is the ordinary labyrinth's radius -- an arithmetic
#: quantity, the bracket's denominator sum.  ``'depth'`` is the purely
#: combinatorial alternative, ``depth + 1`` so the root sits on ring 1.
RADIUS_RULES: Dict[str, Callable[[VariantNode], float]] = {
    "cardinality": lambda n: float(n.cardinality),
    "depth": lambda n: float(n.depth + 1),
}


def _radial(r, r_max: float, scale: str):
    """Map ring numbers to plotted radii under ``'linear'`` or ``'log'``."""
    if scale == "linear":
        return r
    if scale == "log":
        # log1p keeps 0 -> 0 and is monotone; normalised so the rim is unmoved.
        return r_max * np.log1p(r) / math.log1p(r_max)
    raise ValueError(f"radial_scale must be 'linear' or 'log', got {scale!r}")


def _dyadic_marks(levels: int) -> List[Fraction]:
    """Rationals whose ``?``-images are the ``2**levels`` multiples of ``2**-levels``.

    ``?`` sends the Stern-Brocot node at depth ``d`` to an odd multiple of
    ``2**-(d+1)``, so the preimages of an evenly spaced dyadic grid are just
    ``0`` together with every node down to depth ``levels - 1``.  These are the
    only angular tick positions that come out evenly spaced under
    ``angle='minkowski'`` -- a uniform grid in ``g`` collapses almost entirely
    onto the top of the circle, since ``?(1/12)`` is already 0.00049.
    """
    nodes = variant_tree(MEDIANT, max_depth=max(0, levels - 1))
    return [Fraction(0, 1)] + sorted(n.mediant for n in nodes)


def plot_labyrinth_variant(
    *,
    rule: TreeRule = MEDIANT,
    angle: str = "generator",
    radius: str = "cardinality",
    max_depth: Optional[int] = 8,
    max_cardinality: Optional[int] = 24,
    radial_scale: str = "linear",
    show_spokes: bool = True,
    highlight: Union[None, float, Sequence[float]] = None,
    period: float = 2.0,
    label: str = "cents",
    n_labels: int = 12,
    ax=None,
    figsize: Tuple[float, float] = (9.0, 9.0),
):
    """The labyrinth with its angular, radial and tree coordinates swapped out.

    Same furniture as :func:`~biotuner.mos.plotting.plot_labyrinth` -- an arc
    per node spanning its bracket, spokes running inward from the rim, viridis
    by ring -- but every coordinate is a choice.

    Parameters
    ----------
    rule : TreeRule, default :data:`MEDIANT`
        Where inside each bracket the node's value (and so its spoke) sits.
    angle : {'generator', 'minkowski'}, default 'generator'
        Key of :data:`ANGLE_RULES`.
    radius : {'cardinality', 'depth'}, default 'cardinality'
        Key of :data:`RADIUS_RULES`.
    max_depth : int or None, default 8
        Deepest level drawn.  With ``radius='cardinality'`` this *also* clips
        the picture: the spiral arms ``1/k`` and ``k/(k+1)`` reach depth
        ``k - 1``, so reproducing Milne et al. Fig. 1 at 18 rings needs
        ``max_depth=None`` (or ``>= 16``), not the default 8.
    max_cardinality : int, optional, default 24
        Outermost ring when ``radius='cardinality'``.  Ignored (forced to
        ``None``) when ``radius='depth'``, so that the depth rings come out
        complete: the tree at depth ``d`` partitions the circle into exactly
        ``2**d`` arcs, and clipping by denominator would punch holes in them.
    radial_scale : {'linear', 'log'}, default 'linear'
        See the note below before reaching for ``'log'``.
    show_spokes : bool, default True
        Radial line from the rim inward to each node's ring, at the node's
        value.  Under a non-mediant rule these are no longer equal
        temperaments; they are the rule's chosen generator for each bracket.
    highlight : float or sequence of float, optional
        Generator fractions to trace, with a marker on every ring their walk
        visits.  A *fraction* of the period in ``(0, 1)``, not a cents value --
        use :func:`~biotuner.mos.theory.generator_fraction` to convert.
    period : float, default 2.0
        Equivalence interval, used only to size the ``label='cents'`` ticks:
        one full turn is ``PERIOD_CENTS * log2(period)`` cents, 1200 for the
        octave and 1901.955 for the tritave.  The geometry does not depend on
        it -- the angular coordinate is the generator *fraction* either way.
    label : {'cents', 'fraction', 'none'}, default 'cents'
    n_labels : int, default 12
        Angular tick count, at least 1.  Under ``angle='generator'`` these are
        the ``k / n_labels`` marks.  Under ``angle='minkowski'`` a uniform grid
        in ``g`` would collapse onto the top of the circle
        (``?(1/12) = 0.00049``), so the marks are instead the rationals whose
        ``?``-images are uniform: ``0`` plus every tree node down to depth
        ``round(log2(n_labels)) - 1``, clamped to between 2 and 5 levels (4 to
        32 marks).  Pass ``label='none'`` to suppress the ticks entirely.
    ax : matplotlib polar axes, optional
    figsize : tuple

    Returns
    -------
    (fig, ax)

    Notes
    -----
    **The regular-rings finding.**  With ``angle='minkowski'`` and
    ``radius='depth'`` the picture collapses into perfectly regular concentric
    rings: ring ``d`` is cut into ``2**d`` arcs of exactly equal angular width,
    every spoke lands on a dyadic angle, and nothing distinguishes one sector
    from another.  That is the point of drawing it.  Both of those coordinates
    are purely combinatorial, and the combinatorics of the Stern-Brocot tree
    are just those of the infinite binary tree -- featureless.  Every piece of
    visible structure in the ordinary labyrinth (the spiral arms, the dense
    diatonic neighbourhood around ``7/12``, the voids near the period) is
    contributed by the *arithmetic* coordinate: by the denominator, in the
    radius, and by where the rationals actually fall, in the angle.

    **The log radial scale does not do what you would want.**  It was added
    here to decompress the crowded outer rings and it does the opposite:
    ``log1p`` is concave, so it stretches the small radii and squeezes the
    large ones.  Ring 2 of a 24-ring plot moves out from 2.0 to 8.6 while rings
    20--24 are pressed into the width that rings 20--21 used to have.  It is
    genuinely useful -- the musically interesting low-cardinality rings are the
    ones the linear plot starves, jammed into a centre with no circumference --
    but it is a fix for inner crowding, not outer.  Outer crowding in this
    picture is *angular* (ring ``N`` carries ``φ(N)`` arcs) and no radial
    transform can touch it.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> fig, ax = plot_labyrinth_variant(max_cardinality=12, max_depth=None)
    >>> bool(ax.get_ylim()[1] > 12)
    True
    >>> plt.close(fig)

    The degenerate, perfectly regular case:

    >>> fig, ax = plot_labyrinth_variant(angle="minkowski", radius="depth",
    ...                                  max_depth=5)
    >>> plt.close(fig)
    """
    if angle not in ANGLE_RULES:
        raise ValueError(
            f"angle must be one of {sorted(ANGLE_RULES)}, got {angle!r}"
        )
    if radius not in RADIUS_RULES:
        raise ValueError(
            f"radius must be one of {sorted(RADIUS_RULES)}, got {radius!r}"
        )
    if label not in ("cents", "fraction", "none"):
        raise ValueError(f"label must be 'cents', 'fraction' or 'none', got {label!r}")
    if radial_scale not in ("linear", "log"):
        raise ValueError(
            f"radial_scale must be 'linear' or 'log', got {radial_scale!r}"
        )
    if label != "none" and n_labels < 1:
        raise ValueError(
            f"n_labels must be at least 1, got {n_labels!r}; use label='none' "
            "for no angular ticks"
        )
    if not period > 1.0 or not math.isfinite(period):
        raise ValueError(f"period must be a finite number > 1, got {period!r}")
    period_cents = T.PERIOD_CENTS * math.log2(period)

    angle_fn = ANGLE_RULES[angle]
    radius_fn = RADIUS_RULES[radius]

    if radius == "depth":
        # Complete rings, so no denominator clipping.
        if max_depth is None:
            raise ValueError(
                "radius='depth' needs a finite max_depth; got None"
            )
        nodes = variant_tree(rule, max_depth=max_depth, max_cardinality=None)
    else:
        nodes = variant_tree(
            rule, max_depth=max_depth, max_cardinality=max_cardinality
        )
    if not nodes:
        raise ValueError(
            f"no nodes to draw for max_depth={max_depth!r}, "
            f"max_cardinality={max_cardinality!r}"
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    else:
        fig = ax.figure
        if ax.name != "polar":
            raise ValueError(
                "this plot needs a polar axes; create it with "
                "subplot_kw={'projection': 'polar'}"
            )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    rings = sorted({int(radius_fn(n)) for n in nodes})
    r_max = float(rings[-1])
    rim = r_max + 0.8
    rim_p = _radial(rim, rim, radial_scale)
    cmap = plt.get_cmap(RING_CMAP)

    for node in nodes:
        r = radius_fn(node)
        r_p = _radial(r, rim, radial_scale)
        colour = cmap((r - 1.0) / max(1.0, r_max - 1.0))
        arc = np.linspace(angle_fn(node.left), angle_fn(node.right), 96)
        ax.plot(arc, np.full_like(arc, r_p), color=colour, lw=2.0, alpha=0.6,
                solid_capstyle="butt", zorder=2)
        t_node = angle_fn(node.value)
        ax.plot([t_node], [r_p], marker="|", ms=7, color=INK, alpha=0.7, zorder=4)
        if show_spokes:
            ax.plot([t_node, t_node], [rim_p, r_p], color=MUTED, lw=0.6,
                    alpha=0.45, zorder=1)

    if highlight is not None:
        items = highlight if isinstance(highlight, (list, tuple)) else [highlight]
        for i, g in enumerate(items):
            g = float(g)
            colour = HIGHLIGHT_COLORS[i % len(HIGHLIGHT_COLORS)]
            t = angle_fn(g)
            ax.plot([t, t], [0, rim_p], color=colour, lw=1.8, alpha=0.9, zorder=6)
            # Markers sit on the generator's own radial line, as in
            # plotting.plot_labyrinth: one per ring its walk reaches.
            for node in variant_walk(g, rule, max_depth=8 * len(rings) + 32):
                r = radius_fn(node)
                if r > r_max:
                    break
                ax.plot([t], [_radial(r, rim, radial_scale)], "o", ms=6,
                        color=colour, mec=INK, mew=0.5, zorder=7)

    ax.set_rlim(0, rim_p + 0.6)
    # Thin the ring labels until they stop overprinting each other; the log
    # scale squeezes the outer rings hard enough that most of them must go.
    ticks: List[float] = []
    shown: List[int] = []
    for n in rings:
        r_p = _radial(float(n), rim, radial_scale)
        if ticks and r_p - ticks[-1] < 0.055 * rim_p:
            continue
        ticks.append(r_p)
        shown.append(n)
    ax.set_rticks(ticks)
    ax.set_yticklabels([str(n) for n in shown], fontsize=6.5, color=MUTED)
    # Angle 0 is where the deep 1/k and k/(k+1) arms pile up under ?; park the
    # radial labels off it so they stay legible.
    ax.set_rlabel_position(0.0 if angle == "generator" else 22.5)
    ax.grid(color=GRIDC, lw=0.4, alpha=0.5)

    if label == "none":
        ax.set_xticks([])
    else:
        if angle == "minkowski":
            marks = _dyadic_marks(max(2, min(5, round(math.log2(n_labels)))))
        else:
            marks = [Fraction(k, n_labels) for k in range(n_labels)]
        ax.set_xticks([angle_fn(m) for m in marks])
        if label == "cents":
            ax.set_xticklabels(
                [f"{float(m) * period_cents:.0f}" for m in marks],
                fontsize=7.0, color=MUTED,
            )
        else:
            ax.set_xticklabels([str(m) for m in marks], fontsize=7.0, color=MUTED)

    r_label = "cardinality" if radius == "cardinality" else "tree depth"
    a_label = "generator g" if angle == "generator" else "?(g)"
    ax.set_title(
        f"{rule.name} tree   —   angle {a_label},  radius {r_label}"
        f"{'  (log)' if radial_scale == 'log' else ''}",
        fontsize=12.5, pad=18, color=INK,
    )
    ax.legend(
        handles=[
            Line2D([], [], color="#4c8f7d", lw=2.0, alpha=0.6,
                   label="bracket (valid tuning range)"),
            Line2D([], [], color=MUTED, lw=0.6,
                   label=f"{rule.name} node (spoke)"),
        ],
        loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, frameon=False,
    )
    return fig, ax


# --------------------------------------------------------------------------- #
# The Farey tessellation of the hyperbolic plane
# --------------------------------------------------------------------------- #
#: Where the upper half-plane's "centre" is taken to be when mapping to the
#: disk.  ``'i'`` is the literal Cayley transform ``w = (z - i)/(z + i)``;
#: ``'triangle'`` uses the incentre of the ideal triangle ``(0, 1, ∞)``, which
#: is the same tessellation viewed from a different point (a hyperbolic
#: isometry, not a distortion) and puts ``0``, ``1`` and ``∞`` at 120° apart.
DISK_CENTRES: Dict[str, complex] = {
    "i": 1j,
    "triangle": complex(0.5, math.sqrt(3.0) / 2.0),
}


def _to_disk(z, centre: complex = 1j):
    """``w = (z - c) / (z - c̄)`` -- upper half-plane to Poincaré disk.

    Any ``c`` with positive imaginary part works and sends ``c`` to the origin;
    ``c = i`` is the Cayley transform ``(z - i) / (z + i)``.  Every choice sends
    ``R ∪ {∞}`` onto the unit circle, which is where the ideal vertices of the
    Farey tessellation belong, and differs from every other by a hyperbolic
    isometry -- so the geodesics stay geodesics.
    """
    return (z - centre) / (z - centre.conjugate())


def _cayley(z):
    """The Cayley transform ``w = (z - i) / (z + i)``.

    Sends ``0 → -1``, ``1 → -i``, ``∞ → 1``.
    """
    return _to_disk(z, 1j)


def _ideal_point(f: Optional[Fraction], centre: complex = 1j) -> complex:
    """Disk position of an ideal vertex; ``None`` means the cusp at infinity."""
    if f is None:
        # lim_{z→∞} (z - c)/(z - c̄) = 1.
        return complex(1.0, 0.0)
    return _to_disk(complex(float(f), 0.0), centre)


def _geodesic(
    a: Optional[Fraction],
    b: Optional[Fraction],
    centre: complex = 1j,
    n: int = 192,
) -> np.ndarray:
    """Disk points along the true geodesic joining two ideal vertices.

    In the upper half-plane the geodesic between reals ``p`` and ``r`` is the
    semicircle centred at ``(p+r)/2`` orthogonal to the real axis; with one
    endpoint at ``∞`` it is the vertical line through the other.  Both are
    sampled in the half-plane and then pushed through :func:`_to_disk`, so what
    comes back is the exact image of the geodesic, not a chord across it.
    """
    if a is None or b is None:
        p = float(b if a is None else a)
        # z = p + i·tan(s): s → π/2 walks the vertical line up to the cusp.
        s = np.linspace(0.0, math.pi / 2.0, n, endpoint=False)
        w = _to_disk(p + 1j * np.tan(s), centre)
        return np.append(w, 1.0 + 0.0j)
    p, r = float(a), float(b)
    c, radius = (p + r) / 2.0, abs(r - p) / 2.0
    t = np.linspace(0.0, math.pi, n)
    return _to_disk(c + radius * np.exp(1j * t), centre)


def _farey_edges(max_denominator: int) -> List[Tuple[Fraction, Fraction]]:
    """Every Farey-neighbour pair in ``[0, 1]`` with both denominators bounded."""
    verts = T.farey_sequence(max_denominator)
    return [
        (x, y)
        for i, x in enumerate(verts)
        for y in verts[i + 1:]
        if T.is_farey_neighbor(x, y)
    ]


def plot_farey_tessellation(
    max_denominator: int = 12,
    *,
    highlight_generator: Optional[float] = None,
    show_infinity: bool = True,
    annotate_max_denominator: int = 5,
    center: str = "triangle",
    period: float = 2.0,
    ax=None,
    figsize: Tuple[float, float] = (8.0, 8.0),
):
    """The Farey tessellation in the Poincaré disk -- the labyrinth's native home.

    The Stern-Brocot tree is the dual graph of the Farey tessellation: an ideal
    triangulation of the hyperbolic plane whose vertices are the rationals, on
    the circle at infinity, and whose edges join exactly those pairs ``p/q``,
    ``r/s`` with ``|ps - qr| = 1`` -- the bracketing pairs of Milne et al. §3.
    Each triangle ``(left, mediant, right)`` is one tree node; descending the
    tree is crossing an edge into the next triangle.  Everything the labyrinth
    encodes by bending the tree into a circle is here as plain geometry.

    Vertices are drawn for the Farey sequence of order ``max_denominator``
    inside ``[0, 1]`` (the labyrinth's own domain), plus the cusp at infinity,
    which closes the root triangle ``(0, 1, ∞)``.

    Parameters
    ----------
    max_denominator : int, default 12
        Order of the Farey sequence whose vertices are drawn.
    highlight_generator : float, optional
        Trace this generator's Stern-Brocot path through the tessellation: the
        triangles it passes through, and the dual polyline joining them.
    show_infinity : bool, default True
        Draw the cusp at ``∞`` and the two vertical geodesics ``(0, ∞)`` and
        ``(1, ∞)`` that close the root triangle.  Without it the picture is
        confined to the quarter-disk that ``[0, 1]`` maps to.
    annotate_max_denominator : int, default 5
        Label ideal vertices up to this denominator.  Labels beyond about 7
        collide on the boundary circle.
    center : {'triangle', 'i'}, default 'triangle'
        Which point of the half-plane goes to the middle of the disk; see the
        note below.  ``'i'`` is the plain Cayley transform.
    period : float, default 2.0
        Equivalence interval, used only to render ``highlight_generator`` in
        cents in the title.  The tessellation itself is period-independent:
        its vertices are the rationals, not tunings.
    ax : matplotlib axes, optional
    figsize : tuple

    Returns
    -------
    (fig, ax)

    Notes
    -----
    **Choosing the viewpoint.**  The half-plane maps to the disk by
    ``w = (z - c) / (z - c̄)`` for any ``c`` above the real axis, and different
    ``c`` differ by a hyperbolic isometry -- the tessellation is the same, seen
    from a different place.  The textbook choice ``c = i`` is the Cayley
    transform, and it is a poor viewpoint for this particular figure: it sends
    ``[0, 1]`` to the arc from ``-1`` to ``-i``, one quarter of the boundary,
    leaving the whole upper half of the disk empty while the labyrinth's entire
    domain is squeezed into a corner.  ``'triangle'`` uses the incentre of the
    ideal triangle ``(0, 1, ∞)``, ``c = 1/2 + i√3/2``, which places ``0``,
    ``1`` and ``∞`` at 120° apart and gives ``[0, 1]`` a full third of the
    boundary -- the most a Möbius map can give it.

    **The crowding near the boundary is not an artefact.**  Whatever the
    viewpoint, arcs bunch up as they approach the circle at infinity.  That is
    what a hyperbolic tessellation is: infinitely many *congruent* triangles,
    looking smaller and smaller only because the metric blows up at the
    boundary.  The visual pile-up is the Stern-Brocot tree's exponential
    growth, drawn to scale.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> fig, ax = plot_farey_tessellation(8, highlight_generator=math.log2(3 / 2))
    >>> plt.close(fig)

    Ideal vertices really are on the boundary circle:

    >>> abs(abs(_ideal_point(Fraction(3, 5))) - 1.0) < 1e-12
    True

    The geodesic joining ``0/1`` and ``1/1`` is sampled from the vertex ``1``
    (which lands on ``-i``) to the vertex ``0`` (on ``-1``), passing through
    ``-0.2 - 0.4i``, the image of the semicircle's apex ``0.5 + 0.5i``.  It
    never leaves the closed disk:

    >>> arc = _geodesic(Fraction(0, 1), Fraction(1, 1))
    >>> bool(abs(arc[0] - (-1j)) < 1e-12), bool(abs(arc[-1] - (-1 + 0j)) < 1e-12)
    (True, True)
    >>> bool(abs(_cayley(0.5 + 0.5j) - (-0.2 - 0.4j)) < 1e-12)
    True
    >>> bool(np.abs(arc).max() <= 1.0 + 1e-12)
    True
    """
    if max_denominator < 1:
        raise ValueError(f"max_denominator must be >= 1, got {max_denominator}")
    if center not in DISK_CENTRES:
        raise ValueError(
            f"center must be one of {sorted(DISK_CENTRES)}, got {center!r}"
        )
    if not period > 1.0 or not math.isfinite(period):
        raise ValueError(f"period must be a finite number > 1, got {period!r}")
    period_cents = T.PERIOD_CENTS * math.log2(period)
    c = DISK_CENTRES[center]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # The circle at infinity.
    circle = np.exp(1j * np.linspace(0.0, 2.0 * math.pi, 720))
    ax.plot(circle.real, circle.imag, color=INK, lw=1.2, zorder=1)

    edges: List[Tuple[Optional[Fraction], Optional[Fraction]]] = list(
        _farey_edges(max_denominator)
    )
    if show_infinity:
        edges += [(Fraction(0, 1), None), (Fraction(1, 1), None)]

    # An edge's "cardinality" is the denominator of the mediant it will be cut
    # at -- the same quantity the labyrinth uses for its rings, so the same
    # colormap makes the two figures readable against each other.  The cusp at
    # infinity counts as 1/0, giving the root edges (0, ∞) and (1, ∞) their
    # rightful cardinality of 1.
    cmap = plt.get_cmap(RING_CMAP)
    span = max(2.0, 2.0 * max_denominator - 1.0)
    for a, b in edges:
        w = _geodesic(a, b, c)
        card = (a.denominator if a is not None else 0) + (
            b.denominator if b is not None else 0
        )
        frac = min(1.0, math.log(card + 1.0) / math.log(span + 1.0))
        ax.plot(w.real, w.imag, color=cmap(frac), lw=max(0.5, 1.9 - 1.6 * frac),
                alpha=max(0.35, 0.95 - 0.6 * frac), zorder=2,
                solid_capstyle="round")

    # --- the highlighted path ------------------------------------------- #
    if highlight_generator is not None:
        g = float(highlight_generator) % 1.0
        if not 0.0 < g < 1.0:
            raise ValueError(
                f"highlight_generator must reduce into (0, 1), got "
                f"{highlight_generator!r}"
            )
        centres: List[complex] = []
        for node in T.sb_walk(g, max_cardinality=max_denominator):
            trio = (node.left, node.node, node.right)
            for a, b in ((trio[0], trio[1]), (trio[1], trio[2]), (trio[0], trio[2])):
                w = _geodesic(a, b, c)
                ax.plot(w.real, w.imag, color=SIGNAL_COLOR, lw=1.8, alpha=0.85,
                        zorder=4)
            # Euclidean centroid of the three ideal vertices: a marker for the
            # dual path, not a hyperbolic centre (an ideal triangle has none).
            centres.append(sum(_ideal_point(f, c) for f in trio) / 3.0)
        if centres:
            cs = np.asarray(centres)
            ax.plot(cs.real, cs.imag, "--o", color=SIGNAL_COLOR, lw=1.0, ms=4.5,
                    mec="white", mew=0.6, alpha=0.95, zorder=6)
        tip = _ideal_point(Fraction(g).limit_denominator(10**6), c)
        ax.plot([tip.real], [tip.imag], "*", ms=15, color=SIGNAL_COLOR,
                mec="white", mew=0.7, zorder=7)

    # --- ideal vertices --------------------------------------------------- #
    verts: List[Optional[Fraction]] = list(T.farey_sequence(max_denominator))
    if show_infinity:
        verts.append(None)
    for f in verts:
        w = _ideal_point(f, c)
        big = f is None or f.denominator <= annotate_max_denominator
        ax.plot([w.real], [w.imag], "o", ms=5.0 if big else 2.6,
                color=INK if big else MUTED, zorder=5)
        if big:
            ang = math.atan2(w.imag, w.real)
            txt = "∞" if f is None else f"{f.numerator}/{f.denominator}"
            deg = math.degrees(ang)
            ax.text(1.04 * math.cos(ang), 1.04 * math.sin(ang), txt, fontsize=8.5,
                    ha="left" if -90 < deg <= 90 else "right", va="center",
                    rotation=deg if -90 < deg <= 90 else deg - 180.0,
                    rotation_mode="anchor", color=INK)

    ax.set_aspect("equal")
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.axis("off")
    title = f"Farey tessellation to denominator {max_denominator}"
    if highlight_generator is not None:
        title += (
            f"   —   path of g = {float(highlight_generator):.6f} "
            f"({float(highlight_generator) * period_cents:.1f} c)"
        )
    ax.set_title(title, fontsize=12, color=INK)
    return fig, ax
