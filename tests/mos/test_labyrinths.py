"""Tests for :mod:`biotuner.mos.labyrinths`.

The module's whole claim is that three conventions of the ordinary labyrinth --
the angular coordinate, the radial coordinate and the mediant -- are separable.
So the tests come in two kinds: *anchors*, which pin the alternatives back onto
:mod:`biotuner.mos.theory` (with the mediant rule, ``variant_tree`` must be the
Stern-Brocot tree, node for node), and *invariants*, which check the things
that must survive every substitution (brackets stay Farey pairs, values stay
inside their brackets).

Numbers quoted in the module docstrings are re-derived here rather than
restated, so a change of behaviour fails a test instead of quietly making the
prose wrong.
"""

import math
from fractions import Fraction

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biotuner.mos import labyrinths as L
from biotuner.mos import theory as T

FIFTH = math.log2(3 / 2)
PHI = (1 + 5**0.5) / 2


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# --------------------------------------------------------------------------- #
# Part A -- the question mark
# --------------------------------------------------------------------------- #
def test_question_mark_reference_values():
    assert L.minkowski_q(Fraction(0, 1)) == 0.0
    assert L.minkowski_q(Fraction(1, 1)) == 1.0
    assert L.minkowski_q(Fraction(1, 2)) == 0.5
    assert L.minkowski_q(Fraction(1, 3)) == 0.25
    assert L.minkowski_q(Fraction(2, 3)) == 0.75
    assert L.minkowski_q(Fraction(7, 12)) == 0.59375


def test_question_mark_is_strictly_increasing():
    xs = sorted({Fraction(p, q) for q in range(1, 30) for p in range(0, q + 1)})
    qs = [L.minkowski_q(x) for x in xs]
    assert len(xs) == 271
    assert all(a < b for a, b in zip(qs, qs[1:]))


def test_question_mark_is_symmetric_about_a_half():
    for node in T.sb_tree_nodes(20):
        x = node.node
        assert L.minkowski_q(1 - x) == pytest.approx(
            1.0 - L.minkowski_q(x), abs=1e-15
        )


def test_depth_d_node_maps_to_a_dyadic_of_denominator_2_pow_d_plus_1():
    for node in T.sb_tree_nodes(24):
        f = Fraction(L.minkowski_q(node.node)).limit_denominator(2**30)
        assert f.denominator == 2 ** (node.depth + 1), (
            f"{node.node} at depth {node.depth} -> {f}"
        )


def test_bracket_q_width_is_exactly_two_to_the_minus_depth():
    """The structural fact behind the perfectly regular ?-by-depth rings."""
    for node in L.variant_tree(L.MEDIANT, max_depth=9):
        width = L.minkowski_q(node.right) - L.minkowski_q(node.left)
        assert width == pytest.approx(2.0**-node.depth, abs=1e-15)


def test_float_path_agrees_with_the_exact_path():
    for x in [Fraction(7, 12), Fraction(3, 5), Fraction(1, 7), Fraction(17, 18),
              Fraction(11, 29)]:
        assert L.minkowski_q(float(x)) == pytest.approx(
            L.minkowski_q(x), abs=1e-12
        )


def test_question_mark_rejects_points_outside_the_unit_interval():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        L.minkowski_q(1.5)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        L.minkowski_q(Fraction(-1, 3))


def test_question_mark_does_not_decompress_the_diatonic_region():
    """The documented finding, re-derived rather than restated.

    ``?`` uniformises by tree depth, not by denominator: the diatonic window
    loses nodes and the empty window against the period gains eighteen.
    """
    nodes = [n.node for n in T.sb_tree_nodes(18)]
    assert len(nodes) == 101

    def count(lo, hi, key):
        return sum(1 for x in nodes if lo <= key(x) < hi)

    assert count(0.55, 0.62, float) == 8
    assert count(0.55, 0.62, L.minkowski_q) == 3
    assert count(0.95, 1.0, float) == 0
    assert count(0.95, 1.0, L.minkowski_q) == 18

    # The mechanism: the right spiral arm converges on the period
    # geometrically under ?, harmonically by generator.
    for k in range(1, 18):
        arm = Fraction(k, k + 1)
        assert L.minkowski_q(arm) == pytest.approx(1.0 - 2.0**-k, abs=1e-15)
        assert float(arm) == pytest.approx(1.0 - 1.0 / (k + 1))


# --------------------------------------------------------------------------- #
# Part B -- alternative trees
# --------------------------------------------------------------------------- #
def test_mediant_rule_reproduces_the_stern_brocot_tree_exactly():
    """The correctness anchor: same nodes, same brackets, same depths."""
    for bound in (6, 12, 18):
        mine = L.variant_tree(L.MEDIANT, max_depth=None, max_cardinality=bound)
        theirs = T.sb_tree_nodes(bound)
        assert len(mine) == len(theirs)
        for a, b in zip(mine, theirs):
            assert (a.left, a.right, a.depth, a.turn) == (
                b.left, b.right, b.depth, b.turn
            )
            assert a.mediant == b.node
            assert a.cardinality == b.cardinality
            assert a.value == float(b.node)


def test_every_rule_keeps_the_same_bracket_combinatorics():
    rules = [L.MEDIANT, L.NOBLE, L.metallic(2), L.metallic(4), L.weighted(0.3)]
    reference = [
        (n.left, n.right, n.depth, n.turn)
        for n in L.variant_tree(L.MEDIANT, max_depth=7)
    ]
    assert len(reference) == 2**8 - 1
    for rule in rules:
        got = [
            (n.left, n.right, n.depth, n.turn)
            for n in L.variant_tree(rule, max_depth=7)
        ]
        assert got == reference


def test_every_bracket_is_a_farey_neighbour_pair_under_every_rule():
    for rule in (L.MEDIANT, L.NOBLE, L.metallic(3), L.weighted(0.5)):
        for node in L.variant_tree(rule, max_depth=8):
            assert T.is_farey_neighbor(node.left, node.right)
            a, b = node.left, node.right
            assert abs(
                a.numerator * b.denominator - a.denominator * b.numerator
            ) == 1


def test_every_value_lies_strictly_inside_its_bracket():
    for rule in (L.MEDIANT, L.NOBLE, L.metallic(2), L.metallic(4),
                 L.weighted(0.1), L.weighted(25.0)):
        for node in L.variant_tree(rule, max_depth=8):
            assert float(node.left) < node.value < float(node.right), (
                f"{rule.name}: {node.value} outside ({node.left}, {node.right})"
            )


def test_noble_node_between_a_half_and_three_fifths():
    v = L.NOBLE(Fraction(1, 2), Fraction(3, 5))
    assert v == pytest.approx((1 + 3 * PHI) / (2 + 5 * PHI), abs=1e-15)
    assert v == pytest.approx(0.5801787282954641, abs=1e-15)
    # theory.noble_mediant is the same object by another route.
    assert v == T.noble_mediant(Fraction(1, 2), Fraction(3, 5))


def test_noble_node_values_are_irrational():
    """No noble node is a rational with denominator <= 10000."""
    closest = 1.0
    for node in L.variant_tree(L.NOBLE, max_depth=8):
        f = Fraction(node.value).limit_denominator(10_000)
        closest = min(closest, abs(float(f) - node.value))
    # A noble number is the worst-approximable kind; with q <= 10**4 the
    # error cannot fall below about 1/(sqrt(5) q**2) ~ 4e-9.
    assert closest > 1e-10
    assert closest == pytest.approx(4.497292427352306e-09, rel=1e-6)


def test_mediant_series_terminates_and_noble_series_does_not():
    """A rational generator's MOS series stops dead; a noble one never does."""
    rational = 7 / 12
    assert T.mos_cardinalities(rational, 200, include_trivial=True) == [
        2, 3, 5, 7, 12
    ]
    # Raising the bound a hundredfold adds nothing: 7/12 is hit exactly.
    assert T.mos_cardinalities(rational, 20_000, include_trivial=True) == [
        2, 3, 5, 7, 12
    ]

    noble = L.NOBLE(Fraction(1, 2), Fraction(3, 5))
    got = T.mos_cardinalities(noble, 200, include_trivial=True)
    assert got == [2, 3, 5, 7, 12, 19, 31, 50, 81, 131]
    # Still going at a hundred times the bound: twenty cardinalities, the
    # largest 16114, with consecutive ones settling onto the ratio phi.
    bigger = T.mos_cardinalities(noble, 20_000, include_trivial=True)
    assert len(bigger) == 20
    assert bigger[:10] == got
    assert bigger[-1] == 16114
    ratios = [b / a for a, b in zip(bigger, bigger[1:])]
    assert all(abs(r - PHI) < 1e-3 for r in ratios[-8:]), ratios


def test_metallic_means_and_names():
    assert L.metallic(1) is L.NOBLE
    assert L.METALLIC_MEANS[1] == pytest.approx(PHI)
    assert L.METALLIC_MEANS[2] == pytest.approx(1 + math.sqrt(2))
    assert L.METALLIC_MEANS[3] == pytest.approx((3 + math.sqrt(13)) / 2)
    assert L.METALLIC_MEANS[4] == pytest.approx(2 + math.sqrt(5))
    assert set(L.METALLIC_MEANS) == set(L.METALLIC_NAMES) == {1, 2, 3, 4}
    assert [L.METALLIC_NAMES[k] for k in (1, 2, 3)] == [
        "golden", "silver", "bronze"
    ]
    for k in range(1, 5):
        w = L.METALLIC_MEANS[k]
        assert w * w == pytest.approx(k * w + 1.0)  # x^2 = kx + 1


def test_metallic_trees_split_the_dyadic_interval_at_a_constant_ratio():
    """Under ``?`` the k-th metallic tree cuts at 2**k / (2**k + 1), always."""
    for k in (1, 2, 3):
        expected = 2**k / (2**k + 1.0)
        for node in L.variant_tree(L.metallic(k), max_depth=6):
            lo, hi = L.minkowski_q(node.left), L.minkowski_q(node.right)
            assert (L.minkowski_q(node.value) - lo) / (hi - lo) == pytest.approx(
                expected, abs=1e-9
            )
    # The mediant, by contrast, cuts every one of them in half.
    for node in L.variant_tree(L.MEDIANT, max_depth=6):
        lo, hi = L.minkowski_q(node.left), L.minkowski_q(node.right)
        assert (L.minkowski_q(node.value) - lo) / (hi - lo) == pytest.approx(0.5)


def test_tree_bounded_by_depth_is_complete():
    for d in range(0, 9):
        assert len(L.variant_tree(L.NOBLE, max_depth=d)) == 2 ** (d + 1) - 1


def test_tree_rejects_an_unbounded_request():
    with pytest.raises(ValueError, match="needs a bound"):
        L.variant_tree(L.MEDIANT, max_depth=None, max_cardinality=None)
    with pytest.raises(ValueError, match="2\\*\\*31"):
        L.variant_tree(L.MEDIANT, max_depth=30)


def test_tree_rule_rejects_a_non_positive_weight():
    with pytest.raises(ValueError, match="finite and positive"):
        L.TreeRule("bad", 0.0)
    with pytest.raises(ValueError, match="finite and positive"):
        L.TreeRule("bad", -1.0)
    with pytest.raises(ValueError, match="finite and positive"):
        L.TreeRule("bad", float("inf"))


def test_cardinality_is_none_when_the_bracket_is_not_a_farey_pair():
    ok = L.VariantNode(Fraction(1, 2), Fraction(3, 5), 0.58, 3, "R", L.NOBLE)
    assert ok.cardinality == 7
    assert ok.signature() == (2, 5)
    bad = L.VariantNode(Fraction(1, 3), Fraction(3, 5), 0.5, 0, "", L.NOBLE)
    assert bad.cardinality is None
    assert bad.signature() is None


def test_cardinality_matches_the_mos_the_bracket_hosts():
    """The subtle point: cardinality is the bracket's, not the value's."""
    for node in L.variant_tree(L.NOBLE, max_depth=7):
        assert node.cardinality == node.mediant.denominator
        b, d = node.signature()
        assert b + d == node.cardinality
        assert math.gcd(b, d) == 1
        if node.cardinality > 2:
            # A real MOS pair lives in this bracket, whatever the rule.
            assert T.signature_ranges(b, d)


def test_walk_brackets_match_sb_walk_under_every_rule():
    for rule in (L.MEDIANT, L.NOBLE, L.metallic(3)):
        mine = list(L.variant_walk(FIFTH, rule, max_depth=8))
        theirs = list(T.sb_walk(FIFTH, max_cardinality=53))
        assert [n.cardinality for n in mine] == [n.cardinality for n in theirs]
        assert [(n.left, n.right) for n in mine] == [
            (n.left, n.right) for n in theirs
        ]
    assert [n.cardinality for n in L.variant_walk(FIFTH, max_depth=8)] == [
        2, 3, 5, 7, 12, 17, 29, 41, 53
    ]


def test_walk_keeps_the_target_inside_the_bracket():
    for rule in (L.MEDIANT, L.NOBLE, L.weighted(0.2)):
        for node in L.variant_walk(FIFTH, rule, max_depth=20):
            assert float(node.left) < FIFTH < float(node.right)


def test_walk_stops_on_an_exact_rational_target():
    nodes = list(L.variant_walk(7 / 12, L.MEDIANT, max_depth=40))
    assert nodes[-1].mediant == Fraction(7, 12)
    assert [n.cardinality for n in nodes] == [2, 3, 5, 7, 12]


def test_walk_rejects_a_target_outside_the_open_unit_interval():
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        list(L.variant_walk(0.0))
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        list(L.variant_walk(1.0))


# --------------------------------------------------------------------------- #
# Part C -- layouts
# --------------------------------------------------------------------------- #
def test_angle_and_radius_rules_are_the_documented_functions():
    assert set(L.ANGLE_RULES) == {"generator", "minkowski"}
    assert set(L.RADIUS_RULES) == {"cardinality", "depth"}
    assert L.ANGLE_RULES["generator"](0.25) == pytest.approx(math.pi / 2)
    assert L.ANGLE_RULES["minkowski"](Fraction(1, 3)) == pytest.approx(
        2 * math.pi * 0.25
    )
    node = next(iter(L.variant_tree(L.MEDIANT, max_depth=0)))
    assert L.RADIUS_RULES["cardinality"](node) == 2.0
    assert L.RADIUS_RULES["depth"](node) == 1.0


@pytest.mark.parametrize("angle", ["generator", "minkowski"])
@pytest.mark.parametrize("radius", ["cardinality", "depth"])
def test_all_four_coordinate_combinations_draw(angle, radius):
    fig, ax = L.plot_labyrinth_variant(
        angle=angle, radius=radius, max_depth=6, max_cardinality=14
    )
    assert ax.name == "polar"
    assert len(ax.get_lines()) > 20
    radii = set()
    for line in ax.get_lines():
        radii.update(np.round(line.get_ydata(), 6))
    assert max(radii) > 1.0


def test_depth_rings_are_complete_and_carry_two_to_the_d_arcs():
    """At depth d the brackets tile the circle into exactly 2**d arcs."""
    for d in range(0, 6):
        nodes = [n for n in L.variant_tree(L.MEDIANT, max_depth=d)
                 if n.depth == d]
        assert len(nodes) == 2**d
        spans = sorted(
            (L.minkowski_q(n.left), L.minkowski_q(n.right)) for n in nodes
        )
        # Contiguous, non-overlapping, covering [0, 1], all the same width.
        assert spans[0][0] == pytest.approx(0.0)
        assert spans[-1][1] == pytest.approx(1.0)
        for (_, hi), (lo, _) in zip(spans, spans[1:]):
            assert hi == pytest.approx(lo, abs=1e-15)
        widths = [hi - lo for lo, hi in spans]
        assert all(w == pytest.approx(2.0**-d, abs=1e-15) for w in widths)


def test_minkowski_by_depth_puts_every_spoke_on_a_dyadic_angle():
    """The degenerate case: nothing distinguishes one sector from another."""
    nodes = L.variant_tree(L.MEDIANT, max_depth=5)
    angle = L.ANGLE_RULES["minkowski"]
    for n in nodes:
        share = angle(n.value) / (2 * math.pi)
        assert share * 2 ** (n.depth + 1) == pytest.approx(
            round(share * 2 ** (n.depth + 1)), abs=1e-12
        )


def test_log_radial_scale_stretches_the_inside_and_squeezes_the_outside():
    """Documented honestly: log fixes inner crowding, not outer."""
    rim = 24.8
    lin = [L._radial(float(r), rim, "linear") for r in (2, 3, 12, 23, 24)]
    log = [L._radial(float(r), rim, "log") for r in (2, 3, 12, 23, 24)]
    assert lin == [2.0, 3.0, 12.0, 23.0, 24.0]
    # Inner rings move outward, gaining room...
    assert log[0] > 8.0
    assert log[1] - log[0] > lin[1] - lin[0]
    # ...at the cost of the outer ones, which are pressed together.
    assert log[4] - log[3] < 0.4 * (lin[4] - lin[3])
    # Monotone and rim-preserving, so the picture stays a picture.
    assert all(a < b for a, b in zip(log, log[1:]))
    assert L._radial(rim, rim, "log") == pytest.approx(rim)


def test_minkowski_angular_ticks_are_the_dyadic_preimages():
    """The tick marks under ?, which must come out *evenly spaced*.

    Pins ``_dyadic_marks`` to its documented definition -- 0 plus every tree
    node down to depth ``levels - 1`` -- rather than to whatever it returns.
    """
    assert L._dyadic_marks(1) == [Fraction(0, 1), Fraction(1, 2)]
    assert L._dyadic_marks(2) == [
        Fraction(0, 1), Fraction(1, 3), Fraction(1, 2), Fraction(2, 3)
    ]
    assert L._dyadic_marks(3) == [
        Fraction(0, 1), Fraction(1, 4), Fraction(1, 3), Fraction(2, 5),
        Fraction(1, 2), Fraction(3, 5), Fraction(2, 3), Fraction(3, 4),
    ]
    for levels in range(1, 6):
        marks = L._dyadic_marks(levels)
        assert len(marks) == 2**levels
        images = [L.minkowski_q(m) for m in marks]
        assert images == sorted(images)
        for k, im in enumerate(images):
            assert im == pytest.approx(k / 2**levels, abs=1e-15)

    # And the plot really uses them: 12 requested -> round(log2 12) = 4 levels.
    fig, ax = L.plot_labyrinth_variant(
        angle="minkowski", max_cardinality=8, n_labels=12, label="fraction"
    )
    ticks = np.asarray(ax.get_xticks())
    assert len(ticks) == 16
    assert np.allclose(np.diff(ticks), 2 * math.pi / 16, atol=1e-12)
    assert [t.get_text() for t in ax.get_xticklabels()][:3] == ["0", "1/5", "1/4"]


def test_depth_radius_ignores_max_cardinality_so_the_rings_stay_complete():
    """Documented promise: clipping by denominator would punch holes in rings.

    ``max_cardinality`` must be *ignored* under ``radius='depth'``; a depth-5
    tree reaches cardinality 6 on its spiral arms, so a bound of 4 would drop
    arcs if it were honoured.
    """
    kw = dict(radius="depth", max_depth=5, show_spokes=False, label="none")
    tight, ax_t = L.plot_labyrinth_variant(max_cardinality=4, **kw)
    loose, ax_l = L.plot_labyrinth_variant(max_cardinality=None, **kw)
    arcs_t = [l for l in ax_t.get_lines() if len(np.atleast_1d(l.get_xdata())) == 96]
    arcs_l = [l for l in ax_l.get_lines() if len(np.atleast_1d(l.get_xdata())) == 96]
    assert len(arcs_t) == len(arcs_l) == 2**6 - 1

    # Every ring is fully covered: the depth-d arcs tile [0, 2*pi] end to end.
    by_r = {}
    for line in arcs_l:
        r = round(float(np.atleast_1d(line.get_ydata())[0]), 9)
        x = np.atleast_1d(line.get_xdata())
        by_r.setdefault(r, []).append((x[0], x[-1]))
    assert sorted(by_r) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    for d, r in enumerate(sorted(by_r)):
        spans = sorted(by_r[r])
        assert len(spans) == 2**d
        assert spans[0][0] == pytest.approx(0.0, abs=1e-12)
        assert spans[-1][1] == pytest.approx(2 * math.pi, abs=1e-12)
        for (_, hi), (lo, _) in zip(spans, spans[1:]):
            assert hi == pytest.approx(lo, abs=1e-12)


def test_cents_labels_follow_the_period_not_a_hardcoded_octave():
    """A tritave labyrinth must label its ticks to 1901.955 c, not 1200."""
    _, oct_ax = L.plot_labyrinth_variant(max_cardinality=8, n_labels=12)
    assert [t.get_text() for t in oct_ax.get_xticklabels()] == [
        str(k * 100) for k in range(12)
    ]
    _, tri_ax = L.plot_labyrinth_variant(max_cardinality=8, n_labels=12, period=3.0)
    pc = T.PERIOD_CENTS * math.log2(3.0)
    assert pc == pytest.approx(1901.955000865, abs=1e-6)
    assert [t.get_text() for t in tri_ax.get_xticklabels()] == [
        f"{k / 12 * pc:.0f}" for k in range(12)
    ]
    # The geometry is period-independent: the ticks sit at the same angles.
    assert np.allclose(oct_ax.get_xticks(), tri_ax.get_xticks())

    # Same for the Farey title.
    _, fax = L.plot_farey_tessellation(8, highlight_generator=FIFTH, period=3.0)
    assert f"{FIFTH * pc:.1f} c" in fax.get_title()
    _, fax2 = L.plot_farey_tessellation(8, highlight_generator=FIFTH)
    assert "702.0 c" in fax2.get_title()


@pytest.mark.parametrize("bad", [1.0, 0.5, 0.0, -2.0, float("inf")])
def test_a_period_that_is_not_an_interval_is_rejected(bad):
    with pytest.raises(ValueError, match="period must be"):
        L.plot_labyrinth_variant(max_cardinality=8, period=bad)
    with pytest.raises(ValueError, match="period must be"):
        L.plot_farey_tessellation(6, period=bad)


def test_n_labels_below_one_is_a_clear_error_not_a_math_domain_error():
    """It used to reach math.log2(0) under angle='minkowski'."""
    for angle in ("generator", "minkowski"):
        with pytest.raises(ValueError, match="n_labels must be at least 1"):
            L.plot_labyrinth_variant(angle=angle, max_cardinality=8, n_labels=0)
        with pytest.raises(ValueError, match="n_labels must be at least 1"):
            L.plot_labyrinth_variant(angle=angle, max_cardinality=8, n_labels=-3)
    # label='none' wants no ticks at all, so n_labels is irrelevant there.
    fig, ax = L.plot_labyrinth_variant(max_cardinality=8, n_labels=0, label="none")
    assert list(ax.get_xticks()) == []


def test_labyrinth_variant_rejects_bad_coordinate_names():
    with pytest.raises(ValueError, match="angle must be one of"):
        L.plot_labyrinth_variant(angle="nope")
    with pytest.raises(ValueError, match="radius must be one of"):
        L.plot_labyrinth_variant(radius="nope")
    with pytest.raises(ValueError, match="radial_scale must be"):
        L.plot_labyrinth_variant(radial_scale="nope")
    with pytest.raises(ValueError, match="max_depth"):
        L.plot_labyrinth_variant(radius="depth", max_depth=None)


def test_labyrinth_variant_highlight_adds_ink():
    plain, ax0 = L.plot_labyrinth_variant(max_depth=None, max_cardinality=12)
    n_plain = len(ax0.get_lines())
    fig, ax = L.plot_labyrinth_variant(
        max_depth=None, max_cardinality=12, highlight=FIFTH
    )
    # One radial line plus a marker on every MOS ring the fifth reaches.
    assert len(ax.get_lines()) == n_plain + 1 + 5


def test_noble_labyrinth_has_the_same_arcs_and_different_spokes():
    med = L.variant_tree(L.MEDIANT, max_depth=None, max_cardinality=14)
    nob = L.variant_tree(L.NOBLE, max_depth=None, max_cardinality=14)
    assert [(n.left, n.right) for n in med] == [(n.left, n.right) for n in nob]
    moved = sum(1 for a, b in zip(med, nob) if abs(a.value - b.value) > 1e-9)
    assert moved == len(med)


# --------------------------------------------------------------------------- #
# The Farey tessellation
# --------------------------------------------------------------------------- #
def test_cayley_transform_sends_the_real_line_to_the_unit_circle():
    assert L._cayley(0j) == pytest.approx(-1 + 0j)
    assert L._cayley(1 + 0j) == pytest.approx(-1j)
    assert L._cayley(0.5 + 0.5j) == pytest.approx(-0.2 - 0.4j)
    assert L._to_disk(1j, 1j) == 0j  # the centre goes to the origin


@pytest.mark.parametrize("centre", ["i", "triangle"])
def test_every_ideal_vertex_has_modulus_one(centre):
    c = L.DISK_CENTRES[centre]
    for f in T.farey_sequence(16):
        assert abs(abs(L._ideal_point(f, c)) - 1.0) < 1e-9
    assert abs(abs(L._ideal_point(None, c)) - 1.0) < 1e-9


@pytest.mark.parametrize("centre", ["i", "triangle"])
def test_every_geodesic_stays_inside_the_closed_disk(centre):
    c = L.DISK_CENTRES[centre]
    pairs = L._farey_edges(12) + [(Fraction(0, 1), None), (Fraction(1, 1), None)]
    assert len(pairs) > 60
    for a, b in pairs:
        w = L._geodesic(a, b, c)
        assert np.abs(w).max() <= 1.0 + 1e-9
        # Endpoints are the two ideal vertices, in some order.
        ends = {complex(np.round(w[0], 9)), complex(np.round(w[-1], 9))}
        want = {
            complex(np.round(L._ideal_point(a, c), 9)),
            complex(np.round(L._ideal_point(b, c), 9)),
        }
        assert ends == want


def test_the_zero_one_geodesic_is_a_specific_arc():
    """Under the plain Cayley transform: -i to -1, through -0.2 - 0.4i."""
    w = L._geodesic(Fraction(0, 1), Fraction(1, 1))
    assert w[0] == pytest.approx(-1j, abs=1e-12)
    assert w[-1] == pytest.approx(-1 + 0j, abs=1e-12)
    mid = w[len(w) // 2]
    assert mid == pytest.approx(-0.2 - 0.4j, abs=1e-2)
    # A geodesic is not the chord: the arc bulges measurably away from it.
    chord = np.linspace(w[0], w[-1], len(w))
    assert np.abs(w - chord).max() > 0.1


def test_the_triangle_centre_puts_zero_one_and_infinity_at_120_degrees():
    c = L.DISK_CENTRES["triangle"]
    angles = {
        name: math.degrees(
            math.atan2(L._ideal_point(f, c).imag, L._ideal_point(f, c).real)
        ) % 360.0
        for name, f in (("0", Fraction(0, 1)), ("1", Fraction(1, 1)),
                        ("inf", None))
    }
    assert angles["inf"] == pytest.approx(0.0, abs=1e-9)
    assert angles["0"] == pytest.approx(120.0, abs=1e-9)
    assert angles["1"] == pytest.approx(240.0, abs=1e-9)


def test_farey_edges_are_exactly_the_bracketing_pairs():
    edges = L._farey_edges(9)
    for a, b in edges:
        assert T.is_farey_neighbor(a, b)
        assert a < b
    # Every tree node's bracket with both denominators <= 9 must be an edge.
    as_set = set(edges)
    for node in T.sb_tree_nodes(9):
        if max(node.left.denominator, node.right.denominator) <= 9:
            assert (node.left, node.right) in as_set


def test_farey_tessellation_draws_with_and_without_a_path():
    plain, ax0 = L.plot_farey_tessellation(10)
    n_plain = len(ax0.get_lines())
    fig, ax = L.plot_farey_tessellation(10, highlight_generator=FIFTH)
    assert len(ax.get_lines()) > n_plain
    assert ax.get_aspect() == 1.0
    # Nothing drawn strays outside the disk (bar the boundary labels).
    for line in ax.get_lines():
        r = np.hypot(line.get_xdata(), line.get_ydata())
        assert r.max() <= 1.0 + 1e-6


def test_farey_tessellation_rejects_a_bad_viewpoint():
    with pytest.raises(ValueError, match="center must be one of"):
        L.plot_farey_tessellation(8, center="nope")
    with pytest.raises(ValueError, match="max_denominator"):
        L.plot_farey_tessellation(0)


def test_public_names_are_all_exported():
    for name in L.__all__:
        assert hasattr(L, name), name
