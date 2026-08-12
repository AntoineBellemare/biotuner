"""Tests for :mod:`biotuner.mos.morph` -- three journeys through the labyrinth.

The module's claim is that ``tuning``, ``tree`` and ``voice`` are genuinely
different journeys rather than three spellings of one, and that each of them
keeps a promise the others do not.  The tests are organised around those
promises:

*The signature graph* really is the labyrinth's own connectivity -- children are
the Stern-Brocot mediant and agree with :func:`~biotuner.mos.theory.embedding`,
the parent inverts them, and every hop a route takes is one of the three legal
moves.  The route's tie-break is musical, so it is pinned to a concrete list.

*A tuning morph* never leaves its arc: the note count is constant, every frame
is well-formed, the generator interpolates linearly and the period
geometrically, and a signature flip is reported once rather than silently.

*A tree morph* never leaves the set of well-formed scales either, and the
structural claim -- that each frame's generator lies inside its own signature's
valid range -- is checked directly against :func:`theory.signature_ranges`
rather than trusted.

*A voice morph* does leave the space, and three of its properties were bugs
once: the path must not be transposed between the first two frames, the voice
count must not change under it, and :meth:`Morph.trajectory` must follow voices
rather than sorted pitch order so that a crossing is not charged as motion.
Each of those has an explicit regression test here, on a pair where the buggy
behaviour and the correct one actually differ.
"""

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biotuner.mos import morph as MO
from biotuner.mos import theory as T
from biotuner.mos.scale import MOSScale

SIGS = [(5, 2), (2, 5), (4, 3), (3, 4), (2, 3), (3, 2), (5, 7), (7, 5), (1, 1),
        (1, 2), (2, 1)]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _meantone():
    return MOSScale.from_signature(5, 2, tuning=31)


def _legal_move(a, b, allow_inverse=True):
    """The three moves :func:`signature_route` is allowed to make."""
    if b in MO.signature_children(*a):
        return True
    if b == MO.signature_parent(*a):
        return True
    return allow_inverse and b == (a[1], a[0])


def _hops(traj):
    """Per-frame total motion over a trajectory, shorter way round the period."""
    d = np.abs(np.diff(traj, axis=0))
    return np.nansum(np.minimum(d, 1.0 - d), axis=1)


# --------------------------------------------------------------------------- #
# The signature graph
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_large,n_small", SIGS)
def test_children_are_the_stern_brocot_mediant_of_the_tuning_range(
    n_large, n_small
):
    """A child's note count is the denominator of the bracket's mediant.

    The signature's valid generator range is a Farey pair, so its mediant is
    the next Stern-Brocot node; that node's denominator is where the child
    scale lives.  This is the claim ``signature_children`` rests on, checked
    against :mod:`biotuner.mos.theory` rather than against itself.
    """
    lo, hi = T.signature_ranges(n_large, n_small)[1]
    assert T.is_farey_neighbor(lo, hi)
    med = T.mediant(lo, hi)
    first, second = MO.signature_children(n_large, n_small)
    assert first == (n_large, n_large + n_small)
    assert second == (n_large + n_small, n_small)
    assert sum(first) == med.denominator == 2 * n_large + n_small


@pytest.mark.parametrize("n_large,n_small", SIGS)
def test_children_are_the_embedding_scales_of_the_signature_and_its_inverse(
    n_large, n_small
):
    """The diatonic's first child is the twelve notes ``embedding`` predicts.

    And its second child is the *inverse's* embedding -- 5L2s embeds in 12,
    2L5s in 9, and 5L2s's two children have exactly 12 and 9 notes.  Each child
    is also equally tuned precisely at its parent's embedding tuning, which is
    what makes the hop a single continuous move rather than a jump.
    """
    first, second = MO.signature_children(n_large, n_small)
    for child, host in ((first, (n_large, n_small)), (second, (n_small, n_large))):
        card, tuning = T.embedding(*host)
        assert sum(child) == card
        assert T.mos_landmarks(*child).equalized == tuning


@pytest.mark.parametrize("n_large,n_small", SIGS)
def test_parent_inverts_children(n_large, n_small):
    for child in MO.signature_children(n_large, n_small):
        assert MO.signature_parent(*child) == (n_large, n_small)


def test_the_root_has_no_parent():
    """1L1s is where the subtractive Euclidean step runs out."""
    assert MO.signature_parent(1, 1) is None
    assert MO.signature_parent(5, 2) == (3, 2)
    assert MO.signature_parent(2, 5) == (2, 3)


@pytest.mark.parametrize(
    "start,end",
    [((2, 3), (5, 7)), ((5, 2), (4, 3)), ((1, 1), (7, 5)), ((5, 7), (2, 5)),
     ((3, 4), (5, 2)), ((7, 2), (2, 7))],
)
def test_every_hop_of_a_route_is_a_legal_move(start, end):
    route = MO.signature_route(start, end)
    assert route[0] == start
    assert route[-1] == end
    assert len(set(route)) == len(route), f"route revisits a node: {route}"
    for a, b in zip(route, route[1:]):
        assert _legal_move(a, b), f"{a} -> {b} is not a child, parent or inverse"


def test_a_route_to_itself_is_a_single_node():
    assert MO.signature_route((5, 2), (5, 2)) == [(5, 2)]
    assert MO.signature_route((1, 1), (1, 1)) == [(1, 1)]


def test_the_tie_break_stays_small_for_as_long_as_it_can():
    """Pentatonic to chromatic: five, five, seven, twelve.

    Several routes are three hops long; the documented tie-break takes the one
    whose note counts are lexicographically smallest, so it must not reach
    twelve notes a step early and double back.
    """
    route = MO.signature_route((2, 3), (5, 7))
    assert route == [(2, 3), (3, 2), (5, 2), (5, 7)]
    assert [sum(s) for s in route] == [5, 5, 7, 12]
    # And the documented example from the other direction.
    assert MO.signature_route((5, 2), (4, 3)) == [
        (5, 2), (3, 2), (1, 2), (1, 3), (4, 3)
    ]


def test_the_inverse_move_is_the_short_way_round():
    """5L2s and 2L5s meet at their shared equalized landmark: one hop.

    Forbidding that move must genuinely change the answer -- if it did not,
    ``allow_inverse`` would be a no-op flag.
    """
    with_inv = MO.signature_route((5, 2), (2, 5), allow_inverse=True)
    without = MO.signature_route((5, 2), (2, 5), allow_inverse=False)
    assert with_inv == [(5, 2), (2, 5)]
    assert without == [(5, 2), (3, 2), (1, 2), (1, 1), (2, 1), (2, 3), (2, 5)]
    for a, b in zip(without, without[1:]):
        assert _legal_move(a, b, allow_inverse=False)


def test_route_rejects_a_signature_that_is_not_a_signature():
    with pytest.raises(ValueError, match="co-prime"):
        MO.signature_route((2, 4), (5, 2))
    with pytest.raises(ValueError, match="co-prime"):
        MO.signature_route((5, 2), (6, 3))
    with pytest.raises(ValueError, match="two positive counts"):
        MO.signature_route((0, 3), (5, 2))


def test_route_reports_an_impossible_cardinality_ceiling():
    """5L7s has twelve notes; under a ceiling of six there is nowhere to go."""
    with pytest.raises(ValueError, match="raise max_cardinality"):
        MO.signature_route((2, 3), (5, 7), max_cardinality=6)
    # ... and lifting the ceiling makes the very same request succeed.
    assert MO.signature_route((2, 3), (5, 7), max_cardinality=12)[-1] == (5, 7)


# --------------------------------------------------------------------------- #
# Tuning morph -- along one arc
# --------------------------------------------------------------------------- #
def test_a_tuning_morph_holds_the_note_count_and_stays_well_formed():
    a = _meantone()
    m = MO.tuning_morph(a, a.inverse, steps=33)
    assert m.strategy == "tuning"
    assert {s.cardinality for s in m} == {7}
    assert all(s.is_well_formed for s in m)
    assert all(s.wellformedness == 0.0 for s in m)


def test_the_generator_interpolates_linearly_and_hits_both_endpoints():
    a = _meantone()
    b = MOSScale.from_generator(3 / 2, 7)
    m = MO.tuning_morph(a, b, steps=9)
    gs = [s.scale.generator for s in m]
    # Exactly the endpoints, not merely close: t=0 and t=1 are exact in float.
    assert gs[0] == a.generator
    assert gs[-1] == b.generator
    expected = [(1 - s.t) * a.generator + s.t * b.generator for s in m]
    assert gs == pytest.approx(expected, abs=1e-15)
    # Linear means constant spacing, which is the part a bug would break.
    assert np.allclose(np.diff(gs), np.diff(gs)[0])


def test_the_period_interpolates_geometrically_not_arithmetically():
    """A pseudo-octave has to glide in pitch, so log(period) is what is linear.

    Arithmetic interpolation between 2 and 3 would put 2.5 at the half-way
    frame; geometric puts sqrt(6) = 2.449, which is audibly a different place.
    """
    a = MOSScale.from_signature(5, 2, tuning=31, period=2.0)
    b = MOSScale.from_signature(5, 2, tuning=31, period=3.0)
    m = MO.tuning_morph(a, b, steps=5)
    periods = [s.period for s in m]
    assert periods[0] == pytest.approx(2.0)
    assert periods[-1] == pytest.approx(3.0)
    assert periods[2] == pytest.approx(math.sqrt(6.0))
    assert periods[2] != pytest.approx(2.5, abs=1e-3)
    logs = np.log(periods)
    assert np.allclose(np.diff(logs), np.diff(logs)[0])


def test_a_tritave_frame_reports_its_cents_against_the_tritave():
    a = MOSScale.from_signature(5, 2, tuning=31, period=3.0)
    m = MO.tuning_morph(a, a.retune(19), steps=3)
    pc = T.PERIOD_CENTS * math.log2(3.0)
    assert pc == pytest.approx(1901.955, abs=1e-3)
    assert m[0].cents == pytest.approx([d * pc for d in a.degrees], abs=1e-9)
    assert m[0].ratios == pytest.approx([3.0**d for d in a.degrees], abs=1e-12)
    assert m.period_cents == pytest.approx(pc)


def test_crossing_the_equalized_landmark_flips_the_signature_exactly_once():
    """5L2s to its own inverse passes through 7-EDO and the steps trade places."""
    a = _meantone()
    m = MO.tuning_morph(a, a.inverse, steps=33)
    assert m.signatures() == ["5L2s", "2L5s"]
    becomes = [e for _, e in m.events() if "becomes" in e]
    assert len(becomes) == 1
    assert becomes[0].startswith("5L2s becomes 2L5s")
    # The flip is at the landmark, which is the middle of a symmetric morph.
    t_flip = [t for t, e in m.events() if "becomes" in e][0]
    assert t_flip == pytest.approx(0.5, abs=1e-9)


def test_a_morph_that_stays_on_one_side_never_flips():
    """Meantone to Pythagorean keeps 5L2s all the way, so nothing 'becomes'."""
    a = _meantone()
    b = MOSScale.from_generator(3 / 2, 7)
    m = MO.tuning_morph(a, b, steps=9)
    assert m.signatures() == ["5L2s"]
    assert [e for _, e in m.events() if "becomes" in e] == []


def test_a_landmark_is_announced_once_however_fine_the_sampling():
    """The crossing test has a frame-width tolerance, so it can fire twice."""
    a = _meantone()
    for steps in (33, 129, 257):
        events = [e for _, e in MO.tuning_morph(a, a.inverse, steps=steps).events()]
        assert len(events) == len(set(events)), events
        assert sum(1 for e in events if e.startswith("passes")) >= 1


def test_a_tuning_morph_refuses_to_change_the_note_count():
    a = _meantone()
    b = MOSScale.from_signature(5, 7, tuning=12)
    with pytest.raises(ValueError, match="same note count"):
        MO.tuning_morph(a, b)


@pytest.mark.parametrize("steps", [1, 0, -4])
def test_a_morph_needs_two_frames(steps):
    a = _meantone()
    with pytest.raises(ValueError, match="steps must be at least 2"):
        MO.tuning_morph(a, a.inverse, steps=steps)
    with pytest.raises(ValueError, match="steps must be at least 2"):
        MO.voice_morph(a, a.inverse, steps=steps)


# --------------------------------------------------------------------------- #
# Tree morph -- hopping between rings
# --------------------------------------------------------------------------- #
def test_a_tree_morph_walks_the_route_it_was_built_from():
    a = MOSScale.from_signature(2, 3, tuning=12)
    b = MOSScale.from_signature(5, 7, tuning=12)
    route = MO.signature_route((2, 3), (5, 7))
    m = MO.tree_morph(a, b)
    assert m.strategy == "tree"
    assert m.signatures() == [f"{nL}L{ns}s" for nL, ns in route]
    assert [s.cardinality for s in m] == [sum(sig) for sig in route]
    assert [e for _, e in m.events()] == [
        "start at 2L3s", "5 notes -> 7: 5L2s", "7 notes -> 12: 5L7s"
    ]


def test_a_tree_morph_lands_on_the_target_itself():
    """The last frame must *be* the end scale, not a clamped approximation.

    Every intermediate generator is pushed inside its signature's range with a
    small inset, so without the explicit final substitution the journey would
    stop a hair short of where it was asked to go.
    """
    a = MOSScale.from_signature(2, 3, tuning=12)
    b = MOSScale.from_signature(5, 7, tuning=12)
    m = MO.tree_morph(a, b)
    assert m[-1].scale is b
    assert m[-1].degrees == tuple(b.degrees)
    assert m[-1].period == b.period
    assert m[-1].t == 1.0


@pytest.mark.parametrize(
    "start,end",
    [((2, 3), (5, 7)), ((5, 2), (4, 3)), ((5, 2), (2, 5)), ((3, 4), (5, 7))],
)
def test_every_tree_frame_is_tuned_inside_its_own_signature_range(start, end):
    """The real structural claim: a hop lands on a legal tuning of the new ring.

    A generator that fell outside ``signature_ranges`` would not produce the
    signature the frame claims, so the frame would be mislabelled rather than
    merely oddly tuned.
    """
    a = MOSScale.from_signature(*start, tuning="central")
    b = MOSScale.from_signature(*end, tuning="central")
    m = MO.tree_morph(a, b, steps_per_edge=3)
    assert all(s.is_well_formed for s in m)
    for s in m:
        sc = s.scale
        ranges = T.signature_ranges(sc.n_large, sc.n_small)
        assert any(float(lo) < sc.generator < float(hi) for lo, hi in ranges), (
            f"{sc.signature} generator {sc.generator} outside {ranges}"
        )
        # The label is not decoration: the tuning really produces the signature.
        assert T.mos_signature(sc.generator, sc.cardinality) == (
            sc.n_large, sc.n_small
        )


def test_steps_per_edge_multiplies_the_frame_count_without_changing_the_route():
    """One frame per edge subdivision, plus the destination.

    The destination is deliberately *not* multiplied: it has no edge leading
    away from it, so ``steps_per_edge`` copies of it would be a quarter of the
    morph standing still -- a doubled final panel in the filmstrip and a t axis
    that stops advancing.
    """
    a = MOSScale.from_signature(2, 3, tuning=12)
    b = MOSScale.from_signature(5, 7, tuning=12)
    base = MO.tree_morph(a, b)
    n_edges = len(MO.signature_route((2, 3), (5, 7))) - 1
    assert len(base) == n_edges + 1 == 4
    for k in (2, 3, 5):
        m = MO.tree_morph(a, b, steps_per_edge=k)
        assert len(m) == n_edges * k + 1
        assert m.signatures() == base.signatures()
        # Evenly spaced, and arriving exactly once.
        ts = [s.t for s in m]
        assert ts == sorted(ts)
        assert ts.count(1.0) == 1


def test_a_tree_morph_departs_from_the_scale_it_was_given():
    """The mirror of landing on the target, and the missing half of the pair.

    Frame 0's straight-line target *is* the start generator, and a scale's own
    generator is always interior to its own range, so the clamp must reproduce
    it. Without this, a tree morph could begin on the far mirror of the start
    signature -- a legal tuning of the right signature, but not the scale that
    was asked for.
    """
    for start, end in (((5, 2, 31), (4, 3, 19)), ((2, 3, 12), (5, 7, 12)),
                       ((3, 4, "central"), (5, 2, 31))):
        a = MOSScale.from_signature(start[0], start[1], tuning=start[2])
        b = MOSScale.from_signature(end[0], end[1], tuning=end[2])
        m = MO.tree_morph(a, b, steps_per_edge=2)
        assert m[0].scale.generator == pytest.approx(a.generator, abs=1e-5), (
            f"{a.signature} morph starts at {m[0].scale.generator_cents:.2f} c, "
            f"not the {a.generator_cents:.2f} c it was given"
        )
        assert m[0].degrees == pytest.approx(list(a.degrees), abs=1e-5)


@pytest.mark.parametrize(
    "start,end",
    [((5, 2, 31), (4, 3, 19)), ((2, 3, 12), (5, 7, 12)),
     ((5, 2, 12), (2, 5, "central")), ((3, 4, "central"), (5, 7, 12))],
)
def test_every_tree_frame_takes_the_clamp_nearest_the_straight_line(start, end):
    """The documented contract, recomputed rather than trusted.

    A signature has two mirror tuning ranges and both are legal, so checking
    legality cannot distinguish "nearest the straight line" from "whichever one
    came first". This recomputes the intended generator independently: the
    straight-line target, clamped into each mirror, nearest wins.
    """
    a = MOSScale.from_signature(start[0], start[1], tuning=start[2])
    b = MOSScale.from_signature(end[0], end[1], tuning=end[2])
    m = MO.tree_morph(a, b, steps_per_edge=2)
    # The last frame is substituted with the target itself, by design.
    for s in list(m)[:-1]:
        sc = s.scale
        target = (1.0 - s.t) * a.generator + s.t * b.generator
        clamps = [
            min(max(target, float(lo)), float(hi))
            for lo, hi in T.signature_ranges(sc.n_large, sc.n_small)
        ]
        nearest = min(clamps, key=lambda g: abs(g - target))
        assert sc.generator == pytest.approx(nearest, abs=1e-5), (
            f"t={s.t:.3f} {sc.signature}: took {sc.generator:.6f}, "
            f"nearest clamp to {target:.6f} is {nearest:.6f}"
        )


def test_pentatonic_to_chromatic_keeps_its_fifth_the_whole_way():
    """Both endpoints are 12-EDO fifths, so no frame has cause to leave one.

    A value pin on the module's flagship walk. 3L2s cannot hold 700 c -- its
    range stops at 720 c -- so that frame sits on its boundary and every other
    frame keeps the fifth exactly.
    """
    a = MOSScale.from_signature(2, 3, tuning=12)
    b = MOSScale.from_signature(5, 7, tuning=12)
    m = MO.tree_morph(a, b)
    assert [round(s.scale.generator_cents, 2) for s in m] == [
        700.0, 720.0, 700.0, 700.0
    ]


def test_a_tree_morph_between_two_tunings_of_one_signature_still_travels():
    """A single-node route is one edge, not zero.

    signature_route returns one node when the signatures agree.  The frame loop
    then ran once and the final substitution overwrote that frame with the
    destination, so the morph reported a single frame and zero voice motion
    while having silently teleported from meantone to Pythagorean.
    """
    a = MOSScale.from_signature(5, 2, tuning=31)
    b = MOSScale.from_signature(5, 2, tuning=12)
    assert a.signature == b.signature and a.generator != b.generator

    m = MO.tree_morph(a, b)
    assert len(m) == 2
    assert m[0].scale.generator == pytest.approx(a.generator, abs=1e-5)
    assert m[-1].scale is b
    # The same journey a tuning morph makes, and it must cost the same.
    assert m.voice_leading_distance() == pytest.approx(
        MO.tuning_morph(a, b, steps=2).voice_leading_distance(), rel=1e-9
    )
    assert m.voice_leading_distance() > 0.0

    fine = MO.tree_morph(a, b, steps_per_edge=4)
    assert len(fine) == 5
    assert [s.t for s in fine] == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])


def test_steps_per_edge_must_be_at_least_one():
    a = MOSScale.from_signature(2, 3, tuning=12)
    b = MOSScale.from_signature(5, 7, tuning=12)
    with pytest.raises(ValueError, match="steps_per_edge must be at least 1"):
        MO.tree_morph(a, b, steps_per_edge=0)


# --------------------------------------------------------------------------- #
# Voice morph -- move the notes
# --------------------------------------------------------------------------- #
def test_an_equal_cardinality_voice_morph_has_exact_endpoints():
    a = MOSScale.from_signature(5, 2, tuning=31)
    b = MOSScale.from_signature(4, 3, tuning=19)
    m = MO.voice_morph(a, b, steps=17, locate=False)
    # Equal, not merely close: the endpoints are snapped rather than computed.
    assert m[0].degrees == tuple(a.degrees)
    assert m[-1].degrees == tuple(b.degrees)
    assert m[0].scale is a
    assert m[-1].scale is b


def test_a_splitting_voice_morph_starts_with_coincident_tones():
    """Unequal note counts: the endpoints are pitch *sets*, with doublings.

    Frame 0 has as many voices as the target, so two of them start sitting on
    top of a tone of the source; the distinct pitches are still exactly the
    start scale's.
    """
    a = MOSScale.from_signature(2, 3, tuning=12)
    b = MOSScale.from_signature(5, 2, tuning=12)
    m = MO.voice_morph(a, b, steps=17, locate=False)
    assert sorted(set(m[0].degrees)) == pytest.approx(list(a.degrees))
    assert len(m[0].degrees) == b.cardinality
    assert len(set(m[0].degrees)) == a.cardinality
    assert m[-1].degrees == tuple(b.degrees)


@pytest.mark.parametrize(
    "start,end",
    [((5, 2, 31), (4, 3, 19)), ((2, 3, 12), (5, 2, 12)),
     ((5, 7, 12), (2, 3, 12)), ((5, 2, 12), (4, 3, 19))],
)
def test_the_voice_count_never_changes_mid_morph(start, end):
    """Regression: the count used to jump between frame 0 and frame 1.

    A trajectory whose width changes under it is not a set of voices, and the
    extra tone would arrive out of nowhere rather than splitting off something.
    """
    a = MOSScale.from_signature(start[0], start[1], tuning=start[2])
    b = MOSScale.from_signature(end[0], end[1], tuning=end[2])
    m = MO.voice_morph(a, b, steps=13, locate=False)
    expected = max(a.cardinality, b.cardinality)
    assert {s.cardinality for s in m} == {expected}
    assert len(m.voices) == len(m)
    assert {len(v) for v in m.voices} == {expected}
    # Every frame is a permutation of the same voice labels.
    assert all(sorted(v) == list(range(expected)) for v in m.voices)
    assert m.trajectory().shape == (13, expected)


@pytest.mark.parametrize(
    "start,end",
    [((5, 2, 31), (4, 3, "central")), ((5, 2, 12), (4, 3, 19)),
     ((3, 4, "central"), (5, 2, 31))],
)
def test_the_first_hop_is_no_bigger_than_the_others(start, end):
    """Regression: the best rotation must decide the *pairing*, not move the path.

    Applying it to the path transposed the whole scale between frame 0 and
    frame 1, an audible lurch that has nothing to do with the journey.  All
    three pairs here have a non-zero best rotation, so the buggy version's
    opening hop is 6x, 99x and 237x the interior one respectively; sampling
    finely shrinks the interior hops and leaves the discontinuity nowhere to
    hide.  An equal-tuning pair, where the rotation is zero, would pass either
    way and prove nothing.
    """
    a = MOSScale.from_signature(start[0], start[1], tuning=start[2])
    b = MOSScale.from_signature(end[0], end[1], tuning=end[2])
    phi = MO._best_rotation(np.array(a.degrees), np.array(b.degrees))
    assert phi != 0.0, "this pair would not exercise the rotation at all"

    m = MO.voice_morph(a, b, steps=65, locate=False)
    hops = _hops(m.trajectory())
    # Compared against the median rather than the maximum: a bug that lurches
    # at *both* ends would leave a first-versus-rest comparison satisfied.
    typical = float(np.median(hops))
    assert hops.max() <= 1.05 * typical, (
        f"hops {np.round(hops, 5)} are not uniform; median {typical:.5f}"
    )
    assert hops[0] <= 1.05 * typical
    assert hops[-1] <= 1.05 * typical


def test_a_split_costs_only_the_motion_the_tones_actually_make():
    """2L3s to 5L2s in 12-EDO: two tones move 100 c each and five stay put.

    Read in sorted pitch order the same morph looks like a general reshuffle
    and charges several times as much -- motion no voice performs.
    """
    a = MOSScale.from_signature(2, 3, tuning=12)
    b = MOSScale.from_signature(5, 2, tuning=12)
    m = MO.voice_morph(a, b, steps=17, locate=False)
    assert {s.cardinality for s in m} == {7}
    assert m.voice_leading_distance() == pytest.approx(200.0, abs=1e-6)

    sorted_view = np.sort(m.trajectory(), axis=1)
    assert _hops(sorted_view).sum() * 1200.0 > 1000.0
    assert [e for _, e in m.events()] == ["2 tone(s) split on the way"]


def test_locate_off_leaves_wellformedness_at_zero():
    a = MOSScale.from_signature(5, 2, tuning=12)
    b = MOSScale.from_signature(4, 3, tuning=19)
    m = MO.voice_morph(a, b, steps=9, locate=False)
    assert [s.wellformedness for s in m] == [0.0] * 9
    # The frames are still known to be off-scale; only the measurement is off.
    assert [s.is_well_formed for s in m] == [True] + [False] * 7 + [True]


def test_locate_on_measures_how_far_outside_the_labyrinth_the_path_strays():
    a = MOSScale.from_signature(5, 2, tuning=12)
    b = MOSScale.from_signature(4, 3, tuning=19)
    m = MO.voice_morph(a, b, steps=9, locate=True)
    interior = [s.wellformedness for s in m.steps[1:-1]]
    assert max(interior) > 1.0, interior
    assert all(w >= 0.0 for w in interior)
    # Endpoints are the scales themselves, so they are exactly on the map.
    assert m[0].wellformedness == 0.0 and m[-1].wellformedness == 0.0
    assert m.signatures() == ["5L2s", "(off-scale)", "4L3s"]


# --------------------------------------------------------------------------- #
# The Morph container
# --------------------------------------------------------------------------- #
def test_len_iter_and_getitem_agree():
    a = _meantone()
    m = MO.tuning_morph(a, a.inverse, steps=11)
    assert len(m) == 11 == len(m.steps)
    assert list(m) == list(m.steps)
    assert m[0] is m.steps[0]
    assert m[-1] is m.steps[-1]
    assert [s.t for s in m[2:5]] == [s.t for s in m.steps[2:5]]


def test_trajectory_is_nan_padded_to_the_widest_frame():
    a = MOSScale.from_signature(2, 3, tuning=12)
    b = MOSScale.from_signature(5, 7, tuning=12)
    m = MO.tree_morph(a, b)
    traj = m.trajectory()
    assert traj.shape == (len(m), 12)
    # One real value per note of the frame, the rest NaN.
    assert list((~np.isnan(traj)).sum(axis=1)) == [s.cardinality for s in m]
    for row, step in zip(traj, m.steps):
        assert row[: step.cardinality] == pytest.approx(list(step.degrees))
        assert np.isnan(row[step.cardinality:]).all()


def test_trajectory_follows_voices_rather_than_sorted_pitch_order():
    """Where two tones cross, the sorted view hands each to the other's column.

    5L2s in 12-EDO to 4L3s in 19-EDO crosses at nearly every frame, so the two
    readings differ starkly.  Both are measured the shorter way round the
    period, so a voice that simply wraps past the period boundary is not
    counted as a leap; what is left in the sorted view is the crossing itself.
    """
    a = MOSScale.from_signature(5, 2, tuning=12)
    b = MOSScale.from_signature(4, 3, tuning=19)
    m = MO.voice_morph(a, b, steps=25, locate=False)
    assert len(set(m.voices)) > 1, "no crossing, so nothing to unscramble"

    traj = m.trajectory()
    sorted_view = np.sort(traj, axis=1)

    def biggest_step(arr):
        d = np.abs(np.diff(arr, axis=0))
        return np.minimum(d, 1.0 - d).max()

    voice_jump = biggest_step(traj)
    sorted_jump = biggest_step(sorted_view)
    assert voice_jump < 0.05, "a tracked voice glides; it never leaps"
    assert sorted_jump > 10 * voice_jump
    # And the total charged differs by more than the whole journey is worth.
    assert m.voice_leading_distance() == pytest.approx(3815.789, abs=0.01)
    assert _hops(sorted_view).sum() * 1200.0 > 6000.0


def test_voice_leading_distance_takes_the_shorter_way_round():
    """Hand-computed on two frames: 0.95 -> 0.05 is 0.1 of a period, not 0.9.

    The second voice moves 0.1 -> 0.2.  With voices tracked the answer is
    (0.1 + 0.1) * 1200 = 240 c; read in sorted order the wrapping voice is
    mistaken for a leap and the total comes out at 360 c instead.
    """
    a = MOSScale.from_signature(5, 2, tuning=12)
    b = MOSScale.from_signature(5, 2, tuning=19)
    steps = (
        MO.MorphStep(t=0.0, degrees=(0.1, 0.95)),
        MO.MorphStep(t=1.0, degrees=(0.05, 0.2)),
    )
    tracked = MO.Morph(steps, "voice", a, b, voices=((1, 0), (0, 1)))
    assert tracked.trajectory().tolist() == [[0.95, 0.1], [0.05, 0.2]]
    assert tracked.voice_leading_distance() == pytest.approx(240.0)

    untracked = MO.Morph(steps, "voice", a, b)
    assert untracked.trajectory().tolist() == [[0.1, 0.95], [0.05, 0.2]]
    assert untracked.voice_leading_distance() == pytest.approx(360.0)


def test_voice_leading_distance_scales_with_the_period():
    """A tritave morph moves the same fraction but more cents."""
    steps = (
        MO.MorphStep(t=0.0, degrees=(0.0,), period=3.0),
        MO.MorphStep(t=1.0, degrees=(0.25,), period=3.0),
    )
    a = MOSScale.from_signature(5, 2, tuning=12, period=3.0)
    m = MO.Morph(steps, "voice", a, a)
    assert m.period_cents == pytest.approx(T.PERIOD_CENTS * math.log2(3.0))
    assert m.voice_leading_distance() == pytest.approx(0.25 * m.period_cents)


def test_the_three_strategies_give_three_different_totals():
    """If they agreed they would be spellings of one journey, not three."""
    a = MOSScale.from_signature(5, 2, tuning=31)
    b = MOSScale.from_signature(4, 3, tuning=19)
    totals = {
        name: MO.morph(a, b, name).voice_leading_distance()
        for name in MO.STRATEGIES
    }
    assert len(set(round(v, 3) for v in totals.values())) == 3, totals


def test_events_signatures_and_labyrinth_path_line_up_with_the_frames():
    a = MOSScale.from_signature(5, 2, tuning=12)
    b = MOSScale.from_signature(4, 3, tuning=19)
    m = MO.voice_morph(a, b, steps=9, locate=True)

    path = m.labyrinth_path()
    assert len(path) == len(m)
    assert [q is None for q in path] == [not s.is_well_formed for s in m]
    assert path[0] == (a.generator, a.cardinality)
    assert path[-1] == (b.generator, b.cardinality)

    # events() drops the frames with nothing to report, and keeps their t.
    assert m.events() == [(s.t, s.event) for s in m if s.event]
    assert len(m.events()) <= len(m)

    # signatures() collapses runs rather than repeating one per frame.
    long = MO.tuning_morph(a, a.inverse, steps=41)
    assert long.signatures() == ["5L2s", "2L5s"]
    assert len(long) == 41


def test_a_morph_with_no_events_reports_none():
    a = MOSScale.from_signature(5, 2, tuning=12)
    steps = (MO.MorphStep(t=0.0, degrees=(0.0,)), MO.MorphStep(t=1.0, degrees=(0.5,)))
    m = MO.Morph(steps, "voice", a, a)
    assert m.events() == []
    assert m.labyrinth_path() == [None, None]
    assert m.signatures() == ["(off-scale)"]


def test_summary_survives_a_cp1252_console():
    """Regression: it once used a unicode arrow and crashed on Windows.

    Printing a summary must not depend on the terminal's code page, so the
    whole string has to round-trip through cp1252.
    """
    a = _meantone()
    for m in (MO.tuning_morph(a, a.inverse, steps=17),
              MO.tree_morph(MOSScale.from_signature(2, 3, tuning=12),
                            MOSScale.from_signature(5, 7, tuning=12)),
              MO.voice_morph(a, a.inverse, steps=9, locate=True)):
        text = m.summary()
        text.encode("cp1252")          # the failure mode, exactly
        assert text == text.encode("ascii", "strict").decode("ascii")


def test_summary_reports_the_route_and_the_frame_count():
    a = MOSScale.from_signature(2, 3, tuning=12)
    b = MOSScale.from_signature(5, 7, tuning=12)
    text = MO.tree_morph(a, b, steps_per_edge=2).summary()
    assert "2L3s -> 3L2s -> 5L2s -> 5L7s" in text
    # (n_nodes - 1) * steps_per_edge + 1 = 3 * 2 + 1
    assert "frames         7" in text
    assert "every frame is a well-formed scale" in text
    assert "tree morph" in text


def test_summary_owns_up_to_leaving_the_labyrinth():
    a = MOSScale.from_signature(5, 2, tuning=12)
    b = MOSScale.from_signature(4, 3, tuning=19)
    m = MO.voice_morph(a, b, steps=9, locate=True)
    off = sum(1 for s in m if not s.is_well_formed)
    text = m.summary()
    assert f"leaves the labyrinth for {off} of 9 frames" in text
    assert "(off-scale)" in text


# --------------------------------------------------------------------------- #
# The dispatcher
# --------------------------------------------------------------------------- #
def test_the_dispatcher_reaches_all_three_strategies():
    a = MOSScale.from_signature(2, 3, tuning=12)
    b = MOSScale.from_signature(5, 7, tuning=12)
    assert MO.morph(a, b, "tree").signatures() == ["2L3s", "3L2s", "5L2s", "5L7s"]
    assert MO.morph(a, b, "voice", steps=7).strategy == "voice"
    same = MOSScale.from_signature(2, 3, tuning=19)
    assert MO.morph(a, same, "tuning", steps=6).strategy == "tuning"
    # The default is the tuning morph.
    assert MO.morph(a, same).strategy == "tuning"


def test_the_dispatcher_forwards_keyword_arguments():
    a = MOSScale.from_signature(2, 3, tuning=12)
    b = MOSScale.from_signature(5, 7, tuning=12)
    assert len(MO.morph(a, b, "voice", steps=23, locate=False)) == 23
    assert len(MO.morph(a, b, "tree", steps_per_edge=3)) == 10   # 3 edges * 3 + 1
    same = MOSScale.from_signature(2, 3, tuning=19)
    assert len(MO.morph(a, same, "tuning", steps=5)) == 5
    # A kwarg the strategy does not take must not be silently swallowed.
    with pytest.raises(TypeError):
        MO.morph(a, same, "tuning", steps_per_edge=2)


def test_an_unknown_strategy_names_the_valid_ones():
    a = MOSScale.from_signature(2, 3, tuning=12)
    with pytest.raises(ValueError, match="strategy must be one of") as exc:
        MO.morph(a, a, "warp")
    for name in MO.STRATEGIES:
        assert name in str(exc.value)


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #
def test_audio_is_bounded_float32_of_the_requested_length():
    a = MOSScale.from_signature(5, 2, tuning=12)
    m = MO.tuning_morph(a, a.inverse, steps=8)
    audio = MO.morph_audio(m, seconds=0.37, sample_rate=8000)
    assert audio.dtype == np.float32
    assert audio.shape == (round(0.37 * 8000),)
    assert np.abs(audio).max() <= 1.0
    assert np.isfinite(audio).all()
    # Not silence: every voice sounding throughout should fill the range.
    assert np.abs(audio).max() > 0.5
    assert audio.std() > 0.05


def test_audio_is_deterministic():
    a = MOSScale.from_signature(5, 2, tuning=12)
    m = MO.tuning_morph(a, a.inverse, steps=8)
    first = MO.morph_audio(m, seconds=0.25, sample_rate=8000)
    second = MO.morph_audio(m, seconds=0.25, sample_rate=8000)
    assert np.array_equal(first, second)


def test_a_matched_timbre_renders_and_sounds_different():
    a = MOSScale.from_signature(5, 2, tuning=12)
    m = MO.tuning_morph(a, a.inverse, steps=8)
    plain = MO.morph_audio(m, seconds=0.25, sample_rate=8000)
    matched = MO.morph_audio(m, seconds=0.25, sample_rate=8000,
                             matched_timbre=True, n_partials=5)
    assert matched.shape == plain.shape
    assert matched.dtype == np.float32
    assert np.abs(matched).max() <= 1.0
    assert not np.allclose(plain, matched)


def test_a_tree_morph_renders_despite_its_changing_note_count():
    """NaN columns must contribute nothing rather than poisoning the mix."""
    a = MOSScale.from_signature(2, 3, tuning=12)
    b = MOSScale.from_signature(5, 7, tuning=12)
    audio = MO.morph_audio(MO.tree_morph(a, b), seconds=0.25, sample_rate=8000)
    assert np.isfinite(audio).all()
    assert np.abs(audio).max() == pytest.approx(0.92, abs=1e-5)


@pytest.mark.parametrize("seconds", [0.0, -1.0])
def test_audio_needs_a_positive_duration(seconds):
    a = MOSScale.from_signature(5, 2, tuning=12)
    m = MO.tuning_morph(a, a.inverse, steps=4)
    with pytest.raises(ValueError, match="seconds must be positive"):
        MO.morph_audio(m, seconds=seconds)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def test_the_path_is_drawn_on_a_polar_labyrinth():
    a = _meantone()
    fig, ax = MO.plot_morph_path(MO.tuning_morph(a, a.inverse, steps=17))
    assert ax.name == "polar"
    assert ax.figure is fig
    assert "tuning morph" in ax.get_title()
    assert "5L2s" in ax.get_title() and "2L5s" in ax.get_title()


def test_the_generator_column_draws_strictly_more_than_the_bare_path():
    """``column`` is a feature, not a decoration: it must add real strands."""
    a = _meantone()
    m = MO.tuning_morph(a, a.inverse, steps=17)
    fig, bare = MO.plot_morph_path(m, column=False, colorbar=False)
    n_bare = len(bare.get_lines())
    plt.close(fig)
    fig, wedge = MO.plot_morph_path(m, column=True, colorbar=False)
    assert len(wedge.get_lines()) > n_bare
    # The soft wedge over the generator range is a filled collection, too.
    assert len(wedge.collections) > 0


def test_the_colorbar_is_the_difference_of_one_axes():
    a = _meantone()
    m = MO.tuning_morph(a, a.inverse, steps=17)
    fig_off, _ = MO.plot_morph_path(m, colorbar=False)
    n_off = len(fig_off.axes)
    plt.close(fig_off)
    fig_on, _ = MO.plot_morph_path(m, colorbar=True)
    assert len(fig_on.axes) == n_off + 1
    assert "t" in fig_on.axes[-1].get_ylabel()


def test_a_path_that_never_lands_on_a_scale_cannot_be_drawn_there():
    """Nothing to plot on the labyrinth, so say so instead of drawing nothing."""
    a = MOSScale.from_signature(5, 2, tuning=12)
    steps = tuple(
        MO.MorphStep(t=t, degrees=(0.0, 0.31, 0.62), scale=None, period=2.0)
        for t in (0.0, 0.5, 1.0)
    )
    nowhere = MO.Morph(steps, "voice", a, a)
    with pytest.raises(ValueError, match="never lands on a well-formed scale"):
        MO.plot_morph_path(nowhere)


def test_an_off_scale_stretch_is_drawn_as_a_gap():
    """A voice morph leaves the map, and the break has to read as a break."""
    a = MOSScale.from_signature(5, 2, tuning=12)
    b = MOSScale.from_signature(4, 3, tuning=19)
    m = MO.voice_morph(a, b, steps=9, locate=True)
    fig, ax = MO.plot_morph_path(m)
    dotted = [ln for ln in ax.get_lines() if ln.get_linestyle() == ":"]
    assert dotted, "the two banks of the gap should be joined by a dotted line"
    texts = [t.get_text() for t in ax.texts]
    assert any("off the labyrinth for" in t for t in texts), texts


@pytest.mark.parametrize("palette", sorted(MO._PALETTE))
def test_every_palette_draws(palette):
    a = _meantone()
    m = MO.tuning_morph(a, a.inverse, steps=13)
    fig, ax = MO.plot_morph_path(m, palette=palette, colorbar=False)
    assert ax.get_facecolor() is not None
    fig2, axes = MO.plot_morph_trajectory(m, palette=palette)
    assert len(axes) >= 1


def test_an_unknown_palette_lists_the_valid_names():
    a = _meantone()
    m = MO.tuning_morph(a, a.inverse, steps=5)
    for call in (
        lambda: MO.plot_morph_path(m, palette="neon"),
        lambda: MO.plot_morph_trajectory(m, palette="neon"),
        lambda: MO.plot_morph_filmstrip(m, palette="neon"),
        lambda: MO.animate_morph(m, palette="neon"),
    ):
        with pytest.raises(ValueError, match="palette must be one of") as exc:
            call()
        assert "light" in str(exc.value) and "noir" in str(exc.value)


def test_the_trajectory_plot_draws_one_line_per_voice():
    a = MOSScale.from_signature(5, 2, tuning=31)
    b = MOSScale.from_signature(4, 3, tuning=19)
    m = MO.voice_morph(a, b, steps=17, locate=False)
    fig, axes = MO.plot_morph_trajectory(m, show_wellformedness=False)
    top = axes[0]
    assert len(top.get_lines()) == m.trajectory().shape[1]
    assert top.get_ylim() == (0.0, m.period_cents)
    assert f"{m.voice_leading_distance():.0f} c" in top.get_title()


def test_the_wellformedness_panel_appears_only_when_it_has_something_to_say():
    a = MOSScale.from_signature(5, 2, tuning=31)
    on_map = MO.tuning_morph(a, a.inverse, steps=13)
    fig, axes = MO.plot_morph_trajectory(on_map)
    assert len(axes) == 1, "no frame strays, so there is no lower panel to draw"
    plt.close(fig)

    b = MOSScale.from_signature(4, 3, tuning=19)
    strays = MO.voice_morph(a, b, steps=9, locate=True)
    fig, axes = MO.plot_morph_trajectory(strays)
    assert len(axes) == 2
    assert "off-scale" in axes[-1].get_ylabel()


def test_the_filmstrip_has_one_panel_per_sampled_frame():
    a = _meantone()
    m = MO.tuning_morph(a, a.inverse, steps=33)
    for n_frames in (3, 5, 9):
        fig, axes = MO.plot_morph_filmstrip(m, n_frames=n_frames)
        assert len(axes) == n_frames
        assert all("t=" in ax.get_title() for ax in axes)
        plt.close(fig)
    with pytest.raises(ValueError, match="n_frames must be at least 1"):
        MO.plot_morph_filmstrip(m, n_frames=0)


def test_the_filmstrip_labels_off_scale_frames_as_such():
    a = MOSScale.from_signature(5, 2, tuning=12)
    b = MOSScale.from_signature(4, 3, tuning=19)
    m = MO.voice_morph(a, b, steps=9, locate=True)
    fig, axes = MO.plot_morph_filmstrip(m, n_frames=5)
    titles = [ax.get_title() for ax in axes]
    assert any("off-scale" in t for t in titles), titles
    assert "5L2s" in titles[0] and "4L3s" in titles[-1]


def test_the_comparison_builds_all_three_columns():
    a = _meantone()
    fig, morphs = MO.plot_morph_comparison(a, a.inverse, steps=13)
    assert sorted(morphs) == ["tree", "tuning", "voice"]
    assert all(isinstance(m, MO.Morph) for m in morphs.values())
    assert {m.strategy for m in morphs.values()} == set(MO.STRATEGIES)
    # Two rows of three: a labyrinth and a trajectory per strategy.
    assert len(fig.axes) == 6
    assert sum(1 for ax in fig.axes if ax.name == "polar") == 3


def test_the_comparison_drops_the_tuning_column_rather_than_failing():
    """Equal note counts are a fact about the strategy, not about the figure."""
    a = MOSScale.from_signature(2, 3, tuning=12)
    b = MOSScale.from_signature(5, 7, tuning=12)
    fig, morphs = MO.plot_morph_comparison(a, b, steps=9)
    assert sorted(morphs) == ["tree", "voice"]
    assert len(fig.axes) == 6
    said = " ".join(t.get_text() for ax in fig.axes for t in ax.texts)
    assert "not available here" in said


@pytest.mark.filterwarnings("ignore:Animation was deleted")
def test_the_animation_has_one_frame_per_morph_step():
    a = _meantone()
    for steps in (5, 9, 17):
        m = MO.tuning_morph(a, a.inverse, steps=steps)
        anim = MO.animate_morph(m, interval=40)
        assert len(list(anim.new_frame_seq())) == len(m) == steps
        # Drawing a frame must place exactly the tones of that frame.
        line, dots, title = anim._func(steps - 1)
        assert dots.get_offsets().shape == (m[-1].cardinality, 2)
        assert f"t={m[-1].t:.2f}" in title.get_text()
        assert m[-1].scale.signature in title.get_text()
        plt.close("all")


# --------------------------------------------------------------------------- #
# Exports
# --------------------------------------------------------------------------- #
def test_public_names_are_all_exported():
    for name in MO.__all__:
        assert hasattr(MO, name), name
    assert MO.STRATEGIES == ("tuning", "tree", "voice")
