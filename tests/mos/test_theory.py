"""Tests for :mod:`biotuner.mos.theory`.

Most of these check a claim made in Milne, Carlé, Sethares, Noll & Holland
(2011), *Scratching the Scale Labyrinth*, against the implementation.  Where a
number appears in the paper it is quoted in the test name or a comment, so a
failure points at either the code or the reading, not at a mystery constant.
"""

import math
from fractions import Fraction

import pytest

from biotuner.mos import theory as T

FIFTH = math.log2(3 / 2)
MINOR_THIRD_316 = 316 / 1200

# Every co-prime signature up to 13L13s, both mirror placements.
COPRIME_SIGS = [
    (a, b)
    for a in range(1, 14)
    for b in range(1, 14)
    if math.gcd(a, b) == 1
]


# --------------------------------------------------------------------------- #
# Farey / mediant primitives
# --------------------------------------------------------------------------- #
def test_mediant_is_not_the_average():
    assert T.mediant(Fraction(1, 2), Fraction(3, 5)) == Fraction(4, 7)
    assert T.mediant(Fraction(0, 1), Fraction(1, 1)) == Fraction(1, 2)


def test_mediant_lies_strictly_between():
    for lo, hi in [(Fraction(1, 3), Fraction(1, 2)), (Fraction(4, 7), Fraction(3, 5))]:
        assert lo < T.mediant(lo, hi) < hi


def test_noble_mediant_is_irrational_and_inside():
    lo, hi = Fraction(1, 2), Fraction(3, 5)
    nm = T.noble_mediant(lo, hi)
    assert float(lo) < nm < float(hi)
    # Far from every low-denominator rational in the interval.
    assert all(abs(nm - float(Fraction(p, q))) > 1e-4
               for q in range(2, 40) for p in range(1, q))


def test_farey_sequence_lengths_and_order():
    f5 = T.farey_sequence(5)
    assert f5 == sorted(f5)
    assert f5[0] == Fraction(0, 1) and f5[-1] == Fraction(1, 1)
    assert all(x.denominator <= 5 for x in f5)
    # |F_n| = 1 + sum_{k=1..n} phi(k); for n = 5 that is 1+1+1+2+2+4 = 11.
    assert len(f5) == 11


def test_farey_neighbours_have_unit_determinant():
    f7 = T.farey_sequence(7)
    assert all(T.is_farey_neighbor(a, b) for a, b in zip(f7, f7[1:]))


def test_farey_sequence_rejects_zero():
    with pytest.raises(ValueError, match="n must be >= 1"):
        T.farey_sequence(0)


# --------------------------------------------------------------------------- #
# Continued fractions
# --------------------------------------------------------------------------- #
def test_convergents_of_the_fifth():
    cf = T.continued_fraction(FIFTH, max_terms=8)
    convs = T.convergents_from_cf(cf)
    # 7/12 is the convergent behind 12-tone equal temperament.
    assert Fraction(7, 12) in convs
    assert Fraction(1, 2) in convs


def test_semiconvergent_denominators_are_the_mos_cardinalities():
    denoms = [f.denominator for f in T.semiconvergents(FIFTH, 53)]
    assert denoms == T.mos_cardinalities(FIFTH, 53, include_trivial=True)


# --------------------------------------------------------------------------- #
# The Stern-Brocot walk
# --------------------------------------------------------------------------- #
def test_fifth_generates_the_pythagorean_series():
    """Milne et al. section 2: 2, 3, 5, 7, 12, 17, 29, 41, 53."""
    assert T.mos_cardinalities(FIFTH, 53, include_trivial=True) == [
        2, 3, 5, 7, 12, 17, 29, 41, 53
    ]


def test_316_cent_generator_series():
    """Milne et al. section 2: 'when the generator is 316 cents ... 2, 3, 4, 7,
    11, 15, 19'."""
    assert T.mos_cardinalities(MINOR_THIRD_316, 19, include_trivial=True) == [
        2, 3, 4, 7, 11, 15, 19
    ]


def test_walk_brackets_are_always_farey_neighbours():
    for g in (FIFTH, MINOR_THIRD_316, 0.7071, 0.61803):
        for node in T.sb_walk(g, max_cardinality=60):
            assert T.is_farey_neighbor(node.left, node.right)
            assert node.node == T.mediant(node.left, node.right)
            assert node.left < node.node < node.right


def test_walk_brackets_the_target_ever_more_tightly():
    prev = None
    for node in T.sb_walk(FIFTH, max_cardinality=60):
        assert float(node.left) < FIFTH < float(node.right)
        width = float(node.right - node.left)
        if prev is not None:
            assert width < prev
        prev = width


def test_walk_terminates_exactly_on_a_rational_generator():
    nodes = list(T.sb_walk(7 / 12, max_cardinality=100))
    assert nodes[-1].node == Fraction(7, 12)
    assert nodes[-1].exact
    # Nothing beyond: a rational generator is an equal temperament and admits
    # no further moments of symmetry.
    assert all(not n.exact for n in nodes[:-1])


@pytest.mark.parametrize("g", [0.0, 1.0, -0.1, 1.5])
def test_walk_rejects_generators_outside_the_open_unit_interval(g):
    with pytest.raises(ValueError, match=r"strictly in \(0, 1\)"):
        list(T.sb_walk(g))


def test_sb_path_is_one_shorter_than_the_node_count():
    nodes = list(T.sb_walk(FIFTH, max_cardinality=12))
    assert T.sb_path(FIFTH, 12) == "".join(n.turn for n in nodes)
    assert len(T.sb_path(FIFTH, 12)) == len(nodes) - 1


def test_tree_enumerates_every_reduced_fraction_exactly_once():
    """The Stern-Brocot tree is a bijection onto the rationals."""
    for limit in (7, 12, 20):
        nodes = T.sb_tree_nodes(limit)
        got = sorted(n.node for n in nodes)
        want = sorted(
            Fraction(p, q)
            for q in range(2, limit + 1)
            for p in range(1, q)
            if math.gcd(p, q) == 1
        )
        assert got == want
        assert len(got) == len(set(got))


def test_tree_nodes_carry_valid_brackets():
    for node in T.sb_tree_nodes(16):
        assert T.is_farey_neighbor(node.left, node.right)
        assert node.node == T.mediant(node.left, node.right)


def test_sb_node_at_returns_none_off_the_path():
    assert T.sb_node_at(FIFTH, 7) is not None
    assert T.sb_node_at(FIFTH, 6) is None  # 6 is not an MOS cardinality here


# --------------------------------------------------------------------------- #
# Signatures
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "cardinality,expected",
    [(2, (1, 1)), (3, (2, 1)), (5, (2, 3)), (7, (5, 2)), (12, (5, 7)), (17, (12, 5))],
)
def test_fifth_signatures(cardinality, expected):
    """Milne et al. section 2: 'the diatonic scale ... WF(7, 2)', the pentatonic
    is 2L 3s, the diatonic 5L 2s."""
    assert T.mos_signature(FIFTH, cardinality) == expected


def test_signature_of_a_non_mos_cardinality_raises_helpfully():
    with pytest.raises(ValueError, match="not an MOS cardinality"):
        T.mos_signature(FIFTH, 6)


def test_step_counts_are_always_coprime():
    """Milne et al. section 2, 'Co-prime step numbers'."""
    for g in (FIFTH, MINOR_THIRD_316, 0.5417, 0.61803, 0.8891):
        for _, n_large, n_small in T.mos_series(g, 60, include_trivial=True):
            assert math.gcd(n_large, n_small) == 1


def test_signature_counts_sum_to_the_cardinality():
    for g in (FIFTH, MINOR_THIRD_316, 0.61803):
        for card, n_large, n_small in T.mos_series(g, 60, include_trivial=True):
            assert n_large + n_small == card


def test_mos_series_agrees_with_signature_lookups():
    for card, n_large, n_small in T.mos_series(FIFTH, 41, include_trivial=True):
        assert T.mos_signature(FIFTH, card) == (n_large, n_small)


# --------------------------------------------------------------------------- #
# Landmarks, ranges, embeddings, coherence
# --------------------------------------------------------------------------- #
def test_diatonic_landmarks_match_the_paper():
    """Milne et al. section 2: the diatonic 5L2s meets its inverse at 7-tet
    (685.714 c), is bounded at 5-tet (720 c) where its two small steps shrink to
    zero, and the anti-diatonic is bounded at 2-tet (600 c)."""
    lm = T.mos_landmarks(5, 2, bright=True)
    assert (lm.equalized, lm.small_vanishes, lm.large_vanishes) == (
        Fraction(4, 7), Fraction(3, 5), Fraction(1, 2)
    )
    cents = lm.as_cents()
    assert cents["equalized"] == pytest.approx(685.714, abs=1e-3)
    assert cents["small_vanishes"] == pytest.approx(720.0)
    assert cents["large_vanishes"] == pytest.approx(600.0)


def test_antidiatonic_landmarks_match_the_paper():
    lm = T.mos_landmarks(2, 5, bright=True)
    assert (lm.equalized, lm.small_vanishes, lm.large_vanishes) == (
        Fraction(4, 7), Fraction(1, 2), Fraction(3, 5)
    )


@pytest.mark.parametrize("n_large,n_small", COPRIME_SIGS)
@pytest.mark.parametrize("bright", [True, False])
def test_landmark_cardinalities_are_structural(n_large, n_small, bright):
    """Each landmark is the equal temperament left when a step size vanishes."""
    lm = T.mos_landmarks(n_large, n_small, bright=bright)
    assert lm.equalized_edo == n_large + n_small
    assert lm.small_vanishes_edo == n_large
    assert lm.large_vanishes_edo == n_small


def test_diatonic_tuning_ranges_match_the_paper():
    """Milne et al. section 3: 1/2 to 4/7 is the anti-diatonic, 4/7 to 3/5 the
    diatonic."""
    assert T.signature_ranges(5, 2)[1] == (Fraction(4, 7), Fraction(3, 5))
    assert T.signature_ranges(2, 5)[1] == (Fraction(1, 2), Fraction(4, 7))


@pytest.mark.parametrize("n_large,n_small", COPRIME_SIGS)
def test_the_two_ranges_are_mirror_images(n_large, n_small):
    """A generator and its complement build the same scale (Milne et al. section 4)."""
    dark, bright = T.signature_ranges(n_large, n_small)
    assert (1 - bright[1], 1 - bright[0]) == dark


@pytest.mark.parametrize("n_large,n_small", COPRIME_SIGS)
@pytest.mark.parametrize("bright", [True, False])
def test_a_generator_inside_the_range_reproduces_the_signature(
    n_large, n_small, bright
):
    lo, hi = T.signature_ranges(n_large, n_small)[1 if bright else 0]
    g = (float(lo) + float(hi)) / 2.0
    assert T.mos_signature(g, n_large + n_small) == (n_large, n_small)


def test_embedding_matches_the_paper():
    """Milne et al. section 3: the diatonic is embedded within 7/12 (12 tones),
    the anti-diatonic within 5/9 (9 tones)."""
    assert T.embedding(5, 2) == (12, Fraction(7, 12))
    assert T.embedding(2, 5) == (9, Fraction(5, 9))


@pytest.mark.parametrize("n_large,n_small", COPRIME_SIGS)
@pytest.mark.parametrize("bright", [True, False])
def test_embedding_cardinality_is_2p_plus_q(n_large, n_small, bright):
    """Milne et al. section 2: 'the lowest cardinality embedding scale has
    2p + q steps'."""
    card, tuning = T.embedding(n_large, n_small, bright=bright)
    assert card == 2 * n_large + n_small
    assert tuning.denominator == card


def test_coherence_ranges_match_the_paper():
    """Milne et al. section 3: 'the diatonic scale is coherent between 4/7 and
    7/12 ... the anti-diatonic 5/9 to 4/7'."""
    assert T.coherence_range(5, 2) == (Fraction(4, 7), Fraction(7, 12))
    assert T.coherence_range(2, 5) == (Fraction(5, 9), Fraction(4, 7))


@pytest.mark.parametrize("n_large,n_small", COPRIME_SIGS)
@pytest.mark.parametrize("bright", [True, False])
def test_coherence_range_sits_inside_the_valid_range(n_large, n_small, bright):
    lo, hi = T.coherence_range(n_large, n_small, bright=bright)
    r0, r1 = T.signature_ranges(n_large, n_small)[1 if bright else 0]
    assert min(r0, r1) <= lo < hi <= max(r0, r1)


@pytest.mark.parametrize("n_large,n_small", COPRIME_SIGS)
def test_coherence_boundary_is_exactly_hardness_two(n_large, n_small):
    """Well-formed scales are coherent while Blackwood's R < 2, and the
    embedding-EDO end of the coherent range is where R hits 2 exactly."""
    lo, hi = T.coherence_range(n_large, n_small, bright=True)
    card = n_large + n_small
    lm = T.mos_landmarks(n_large, n_small, bright=True)
    boundary = hi if lm.equalized == lo else lo
    large, small = T.step_sizes(float(boundary), card)
    if small > 0:
        assert large / small == pytest.approx(2.0, abs=1e-6)


def test_non_coprime_signature_is_rejected():
    with pytest.raises(ValueError, match="co-prime"):
        T.signature_brackets(4, 2)
    with pytest.raises(ValueError, match=">= 1"):
        T.signature_brackets(0, 3)


# --------------------------------------------------------------------------- #
# Words and step sizes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_large,n_small", COPRIME_SIGS)
def test_christoffel_word_has_the_right_letter_counts(n_large, n_small):
    w = T.christoffel_word(n_large, n_small)
    assert (w.count("L"), w.count("s")) == (n_large, n_small)
    assert len(w) == n_large + n_small


def test_christoffel_upper_word_is_the_reverse():
    assert T.christoffel_word(5, 2, lower=False) == T.christoffel_word(5, 2)[::-1]


@pytest.mark.parametrize("n_large,n_small", COPRIME_SIGS)
def test_stacked_word_matches_the_abstract_word_up_to_rotation(n_large, n_small):
    """The empirical step pattern is a rotation of the Christoffel word."""
    lo, hi = T.signature_ranges(n_large, n_small)[1]
    g = (float(lo) + float(hi)) / 2.0
    card = n_large + n_small
    empirical = T.word_from_generator(g, card)
    abstract = T.christoffel_word(n_large, n_small)
    rotations = {abstract[k:] + abstract[:k] for k in range(card)}
    assert empirical in rotations


def test_mos_word_rotations_cover_every_mode():
    word = T.christoffel_word(5, 2)
    assert {T.mos_word(5, 2, mode=k) for k in range(7)} == {
        word[k:] + word[:k] for k in range(7)
    }


@pytest.mark.parametrize("n_large,n_small", COPRIME_SIGS)
def test_steps_tile_the_period(n_large, n_small):
    """nL * L + ns * s == 1 period, exactly -- Milne et al. section 2."""
    lo, hi = T.signature_ranges(n_large, n_small)[1]
    g = (float(lo) + float(hi)) / 2.0
    card = n_large + n_small
    large, small = T.step_sizes(g, card)
    assert n_large * large + n_small * small == pytest.approx(1.0, abs=1e-9)


def test_degrees_are_sorted_rooted_and_inside_the_period():
    degs = T.degrees_from_generator(FIFTH, 7)
    assert degs == sorted(degs)
    assert degs[0] == 0.0
    assert all(0.0 <= d < 1.0 for d in degs)
    assert len(set(degs)) == 7


def test_degenerate_tuning_has_one_step_size():
    large, small = T.step_sizes(7 / 12, 12)
    assert large == pytest.approx(small)
    assert T.word_from_generator(7 / 12, 12) == "L" * 12


# --------------------------------------------------------------------------- #
# Common tones between generators
# --------------------------------------------------------------------------- #
def test_generators_of_one_edo_share_all_of_it():
    assert len(T.common_tones([7 / 12, 5 / 12], max_cardinality=12)) == 12
    assert len(T.common_tones([Fraction(3, 19), Fraction(11, 19)],
                              max_cardinality=19)) == 19


def test_a_generator_and_its_complement_share_only_the_root():
    """They build the same scale, mirrored -- which is not the same tones."""
    got = T.common_tones([FIFTH, 1 - FIFTH], max_cardinality=12)
    assert got == pytest.approx([0.0])


def test_unrelated_generators_share_only_the_root():
    assert T.common_tones([FIFTH, 0.7071], max_cardinality=12) == pytest.approx([0.0])


def test_shared_tones_grow_monotonically_with_tolerance():
    counts = [
        len(T.common_tones([FIFTH, 18 / 31], max_cardinality=12, tol_cents=t))
        for t in (1, 5, 12, 30, 80)
    ]
    assert counts == sorted(counts)
    assert counts[0] == 1        # always the root
    assert counts[-1] > counts[0]


def test_every_shared_tone_really_is_shared():
    gens = [FIFTH, 18 / 31, 0.62]
    tol = 8.0
    shared = T.common_tones(gens, max_cardinality=12, tol_cents=tol)
    pools = []
    for g in gens:
        cards = [c for c in T.mos_cardinalities(g, 12, include_trivial=True) if c >= 3]
        pools.append(T.degrees_from_generator(g, cards[-1]))
    for s in shared:
        near = sum(
            any(min(abs(s - d), 1 - abs(s - d)) * 1200 <= tol for d in pool)
            for pool in pools
        )
        assert near >= 2, f"{s} is not within tolerance of two different generators"


def test_fewer_than_two_generators_gives_nothing():
    assert T.common_tones([FIFTH]) == []
    assert T.common_tones([]) == []
    # Values outside (0, 1) are skipped, not an error.
    assert T.common_tones([FIFTH, 0.0, 1.0]) == []


def test_common_tones_validates_its_arguments():
    with pytest.raises(ValueError, match="tol_cents must be non-negative"):
        T.common_tones([FIFTH, 0.7], tol_cents=-1)
    with pytest.raises(ValueError, match="must exceed 1"):
        T.common_tones([FIFTH, 0.7], period=1.0)


def test_a_pseudo_octave_changes_what_counts_as_shared():
    """tol_cents is measured against the real period, not a hardcoded 1200."""
    a = T.common_tones([FIFTH, 18 / 31], max_cardinality=12, tol_cents=6.0)
    b = T.common_tones([FIFTH, 18 / 31], max_cardinality=12, tol_cents=6.0, period=3.0)
    # A tritave is wider, so the same cents tolerance is a smaller fraction of
    # it and cannot admit more tones.
    assert len(b) <= len(a)


# --------------------------------------------------------------------------- #
# Ratio conversions
# --------------------------------------------------------------------------- #
def test_generator_fraction_round_trip():
    for ratio, period in [(3 / 2, 2.0), (5 / 4, 2.0), (3 / 2, 3.0), (7 / 4, 2.05)]:
        g = T.generator_fraction(ratio, period)
        assert 0.0 < g < 1.0
        assert T.fraction_to_generator(g, period) == pytest.approx(ratio)


def test_generator_fraction_reduces_into_the_period():
    assert T.generator_fraction(3.0) == pytest.approx(T.generator_fraction(1.5))


def test_fold_generator_uses_the_mirror_symmetry():
    assert T.fold_generator(FIFTH) == pytest.approx(1 - FIFTH)
    assert T.fold_generator(0.3) == pytest.approx(0.3)
    assert 0.0 <= T.fold_generator(0.9999) <= 0.5


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_generator_fraction_rejects_non_positive_ratios(bad):
    with pytest.raises(ValueError, match="must be positive"):
        T.generator_fraction(bad)


@pytest.mark.parametrize("bad", [1.0, 0.5, 0.0])
def test_period_must_exceed_one(bad):
    with pytest.raises(ValueError, match="must exceed 1"):
        T.generator_fraction(1.5, period=bad)
    with pytest.raises(ValueError, match="must exceed 1"):
        T.fraction_to_generator(0.5, period=bad)
