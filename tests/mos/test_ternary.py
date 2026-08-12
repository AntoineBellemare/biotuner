"""Tests for :mod:`biotuner.mos.ternary`.

The claims these check are the ones the module is built on, and two of them came
back different from what was expected:

- **No ternary word is improper everywhere in its simplex.** The equal-step
  tuning is an interior point and is simply ``N``-EDO, where the generic
  interval classes are separated by a whole step, so propriety holds on an open
  neighbourhood of it for *every* word. Over the thirty rotation classes of
  ``3L2M2s`` the proper share of the triangle runs 4.2%--11.2% and never
  reaches zero. See ``test_no_ternary_word_is_improper_everywhere``.
- **The canonical ``L > M > s`` region is one of six, not one sixth.** The six
  ordering regions are the cevian subdivision through the equal-step point, so
  they have equal count but unequal area; the canonical one covers
  ``b*c / ((a+b)*N)``, which is ``1/6`` only when ``a == b == c``. Verified by
  Monte Carlo in ``test_canonical_region_area``.

The gluing claim -- that the boundary of the ternary simplex is the two-step
world -- is checked two ways: the degrees converge to the binary scale's as a
coordinate goes to zero, and for MV3 words every co-prime edge turns out to be a
*rotation of the Christoffel word*, i.e. a genuine MOS and therefore literally
an arc of the labyrinth (111 of 111 edges at 7, 8 and 9 notes; only 83% for
arbitrary words).
"""

import math

import matplotlib

matplotlib.use("Agg")

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from biotuner.mos import metrics as MT
from biotuner.mos import theory as T
from biotuner.mos.ternary import (
    LETTERS,
    TernaryScale,
    _canonical_rotation,
    _canonical_vertices,
    _multiset_permutations,
    _xy_to_barycentric,
    barycentric_to_xy,
    mos_substitution,
    plot_ternary_atlas,
    plot_ternary_simplex,
    proper_fraction,
    sampled_max_variety,
    ternary_atlas,
    ternary_words,
    variety_sample_points,
)

PERIOD = 1200.0

#: One MV3 word per size, plus two deliberately badly-arranged ones.
CORPUS = [
    "LMs",
    "LMLsLMs",
    "LMLsMLs",
    "LMLMLMs",
    "LLLMLLs",
    "LLLMMss",
    "LLLMsMs",
    "LMLsLMsLs",
    "LMsLMLM",
]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _necklaces(a, b, c):
    return [
        w
        for w in _multiset_permutations({"L": a, "M": b, "s": c}, a + b + c)
        if _canonical_rotation(w) == w
    ]


def _rotations(word):
    return {word[k:] + word[:k] for k in range(len(word))}


def _sample_scales(word, n=40, seed=7):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        u, v, w = rng.dirichlet((1.0, 1.0, 1.0))
        if min(u, v, w) < 1e-3:
            continue
        out.append(TernaryScale.from_barycentric(word, u, v, w))
    return out


# --------------------------------------------------------------------------- #
# The defining constraint
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("word", CORPUS)
def test_steps_fill_the_period_exactly(word):
    for scale in _sample_scales(word, n=12):
        a, b, c = scale.counts
        lc, mc, sc = scale.step_cents
        assert a * lc + b * mc + c * sc == pytest.approx(PERIOD, abs=1e-9)


@pytest.mark.parametrize("word", CORPUS)
def test_degrees_start_at_zero_increase_and_stay_inside_the_period(word):
    for scale in _sample_scales(word, n=12):
        degrees = scale.degrees
        assert len(degrees) == scale.cardinality == len(word)
        assert degrees[0] == 0.0
        assert all(b > a for a, b in zip(degrees, degrees[1:]))
        assert degrees[-1] < 1.0
        # The gap from the last degree back to the period is the last step.
        assert 1.0 - degrees[-1] == pytest.approx(
            scale.step_fractions[word[-1]], abs=1e-12
        )


def test_barycentric_is_the_share_of_the_period_per_class():
    scale = TernaryScale.from_barycentric("LMLsLMs", 0.52, 0.30, 0.18)
    u, v, w = scale.barycentric
    assert (u, v, w) == pytest.approx((0.52, 0.30, 0.18), abs=1e-12)
    assert u + v + w == pytest.approx(1.0, abs=1e-12)
    a, b, c = scale.counts
    assert u == pytest.approx(a * scale.large)
    assert v == pytest.approx(b * scale.medium)
    assert w == pytest.approx(c * scale.small)


def test_from_barycentric_normalises_its_arguments():
    a = TernaryScale.from_barycentric("LMLsLMs", 5, 3, 2)
    b = TernaryScale.from_barycentric("LMLsLMs", 0.5, 0.3, 0.2)
    assert a.step_cents == pytest.approx(b.step_cents, abs=1e-12)


def test_period_other_than_the_octave():
    scale = TernaryScale.from_barycentric("LMLsLMs", 1, 1, 1, period=3.0)
    assert scale.period_cents == pytest.approx(1200.0 * math.log2(3.0))
    a, b, c = scale.counts
    lc, mc, sc = scale.step_cents
    assert a * lc + b * mc + c * sc == pytest.approx(scale.period_cents, abs=1e-9)
    assert scale.ratios[0] == 1.0
    assert scale.ratios[-1] < 3.0


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_word_must_use_all_three_letters():
    with pytest.raises(ValueError, match="all three step classes"):
        TernaryScale("LLsLLs", 0.2, 0.2, 0.1)


def test_word_rejects_foreign_letters():
    with pytest.raises(ValueError, match=r"only contain 'L', 'M' and 's'"):
        TernaryScale("LMxs", 0.25, 0.25, 0.25)


def test_steps_must_be_positive():
    with pytest.raises(ValueError, match="finite and > 0"):
        TernaryScale("LMs", 0.5, 0.5, 0.0)


def test_steps_must_close_the_period_and_the_error_states_both_values():
    with pytest.raises(ValueError) as excinfo:
        TernaryScale("LMs", 0.3, 0.3, 0.3)
    message = str(excinfo.value)
    assert "0.8999999999999999" in message or "0.9" in message
    assert "expected 1.0" in message


def test_period_must_exceed_one():
    with pytest.raises(ValueError, match="period ratio must exceed 1"):
        TernaryScale("LMs", 1 / 3, 1 / 3, 1 / 3, period=1.0)


@pytest.mark.parametrize("period", [float("nan"), float("inf")])
def test_period_must_be_finite(period):
    """A non-finite period used to build silently and poison every cents value."""
    with pytest.raises(ValueError, match="period ratio must exceed 1"):
        TernaryScale("LMs", 1 / 3, 1 / 3, 1 / 3, period=period)


def test_period_must_be_a_number():
    with pytest.raises(TypeError, match="period must be a number"):
        TernaryScale("LMs", 1 / 3, 1 / 3, 1 / 3, period="octave")


def test_scale_is_frozen():
    scale = TernaryScale.equal_step("LMs")
    with pytest.raises(Exception):
        scale.large = 0.9


# --------------------------------------------------------------------------- #
# The equal-step point
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("word", CORPUS)
def test_equal_step_is_that_cardinality_as_an_edo(word):
    scale = TernaryScale.equal_step(word)
    n = scale.cardinality
    assert scale.cents == pytest.approx([i * PERIOD / n for i in range(n)], abs=1e-9)
    # Trivially proper, variety 1: the smoke test every ternary word must pass.
    assert scale.max_variety == 1
    assert scale.is_proper
    assert scale.propriety_margin == pytest.approx(PERIOD / n, abs=1e-9)


def test_equal_step_is_not_the_centroid():
    scale = TernaryScale.equal_step("LMLsLMs")
    assert scale.barycentric == pytest.approx((3 / 7, 2 / 7, 2 / 7), abs=1e-12)
    assert scale.barycentric != pytest.approx((1 / 3, 1 / 3, 1 / 3), abs=1e-3)
    # ...but it is the centroid exactly when the three counts agree.
    even = TernaryScale.equal_step("LMsLMsLMs")
    assert even.barycentric == pytest.approx((1 / 3, 1 / 3, 1 / 3), abs=1e-12)


def test_centroid_tuning_is_not_the_equal_step_scale():
    centroid = TernaryScale.from_barycentric("LMLsLMs", 1 / 3, 1 / 3, 1 / 3)
    assert centroid.step_cents == pytest.approx([400 / 3, 200.0, 200.0], abs=1e-9)
    assert not centroid.is_canonical  # three L's against two M's: L comes out smaller


# --------------------------------------------------------------------------- #
# Structure, measured through biotuner.mos.metrics
# --------------------------------------------------------------------------- #
def test_propriety_margin_sign_matches_the_metrics_verdict():
    disagreements = 0
    for word in CORPUS:
        for scale in _sample_scales(word, n=30):
            from_metrics = bool(MT.is_proper((scale.cents, scale.period_cents)))
            if from_metrics != (scale.propriety_margin >= -1e-9):
                disagreements += 1
    assert disagreements == 0


def test_variety_and_propriety_are_rotation_invariant():
    scale = TernaryScale.from_barycentric("LMLsLMs", 0.52, 0.30, 0.18)
    rotated = scale.rotations()
    assert len(rotated) == 7
    assert len({r.word for r in rotated}) == 7
    assert all(r.max_variety == scale.max_variety for r in rotated)
    assert all(r.is_proper == scale.is_proper for r in rotated)
    assert all(
        r.step_cents == pytest.approx(scale.step_cents, abs=1e-12) for r in rotated
    )


def test_max_variety_is_at_least_three_for_a_generic_tuning():
    for word in CORPUS:
        for scale in _sample_scales(word, n=5):
            assert scale.max_variety >= 3


def test_generic_interval_sizes_match_the_metrics_layer():
    scale = TernaryScale.from_barycentric("LMLsLMs", 0.52, 0.30, 0.18)
    direct = MT.generic_interval_sizes((scale.cents, scale.period_cents))
    assert scale.generic_interval_sizes == direct
    assert scale.interval_matrix.shape == (7, 6)
    # The step class holds exactly the three step sizes.
    assert direct[1] == pytest.approx(sorted(scale.step_cents), abs=1e-9)


# --------------------------------------------------------------------------- #
# The boundary: the ternary simplex is glued onto the labyrinth
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("word", ["LMLsLMs", "LMLMLMs", "LLLMLLs", "LMsLMLM"])
@pytest.mark.parametrize("letter", list(LETTERS))
def test_degrees_converge_to_the_binary_scale_on_each_edge(word, letter):
    pattern, (n_large, n_small) = TernaryScale.equal_step(word).degenerate_to(letter)
    assert set(pattern) <= {"L", "s"}
    assert n_large + n_small == len(word) - word.count(letter)

    index = LETTERS.index(letter)
    previous = None
    for eps in (1e-2, 1e-3, 1e-4):
        # Push the vanishing class to eps of the period, split the rest evenly.
        coords = [(1.0 - eps) / 2.0] * 3
        coords[index] = eps
        scale = TernaryScale.from_barycentric(word, *coords)
        survivors = [
            scale.step_fractions[ell] for ell in LETTERS if ell != letter
        ]
        binary_sizes = {"L": survivors[0], "s": survivors[1]}
        # The binary scale on the edge, built from the surviving step sizes.
        acc, expected = 0.0, []
        for ch in pattern:
            expected.append(acc)
            acc += binary_sizes[ch]
        # Its steps must fill the period once the vanishing class is dropped.
        assert acc == pytest.approx(1.0 - eps, abs=1e-12)
        gap = max(
            abs(d - e)
            for d, e in zip(
                [x for x, ch in zip(scale.degrees, word) if ch != letter], expected
            )
        )
        # Each surviving degree is displaced by at most the whole period share
        # of the vanishing class, so the edge is approached at rate eps.
        assert gap <= eps * (1.0 + 1e-9)
        if previous is not None:
            assert gap <= previous  # monotone convergence onto the edge
        previous = gap
    assert previous < 1e-3


@pytest.mark.parametrize("cardinality", [7, 8, 9])
def test_every_coprime_edge_of_an_mv3_word_is_a_genuine_mos(cardinality):
    """111 of 111 co-prime edges, over 7, 8 and 9 notes."""
    total = christoffel = 0
    for a in range(1, cardinality - 1):
        for b in range(1, cardinality - a):
            c = cardinality - a - b
            for word in ternary_words(a, b, c):
                stub = TernaryScale.equal_step(word)
                for letter in LETTERS:
                    pattern, (n_large, n_small) = stub.degenerate_to(letter)
                    if math.gcd(n_large, n_small) != 1:
                        continue  # not well formed: the pattern simply repeats
                    total += 1
                    christoffel += pattern in _rotations(
                        T.christoffel_word(n_large, n_small)
                    )
    assert total > 0
    assert christoffel == total


def test_arbitrary_words_do_not_all_have_mos_edges():
    """The MV3 result above is not vacuous: only 83% for arbitrary words."""
    total = christoffel = 0
    for a in range(1, 6):
        for b in range(1, 7 - a):
            c = 7 - a - b
            for word in _necklaces(a, b, c):
                stub = TernaryScale.equal_step(word)
                for letter in LETTERS:
                    pattern, (n_large, n_small) = stub.degenerate_to(letter)
                    if math.gcd(n_large, n_small) != 1:
                        continue
                    total += 1
                    christoffel += pattern in _rotations(
                        T.christoffel_word(n_large, n_small)
                    )
    rate = christoffel / total
    assert (total, christoffel) == (534, 444)
    assert 0.80 < rate < 0.86


def test_a_pentatonic_edge_of_the_diatonic_like_word():
    scale = TernaryScale.equal_step("LMLsLMs")
    assert scale.degenerate_to("s") == ("LsLLs", (3, 2))
    assert scale.degenerate_to("M") == ("LLsLs", (3, 2))
    # 2L2s is not co-prime, so that edge is a repetition rather than an MOS:
    # the pattern is a doubled 'Ls' and the scale's real period is half an
    # octave, which is outside what MOSScale accepts.
    assert scale.degenerate_to("L") == ("LsLs", (2, 2))
    assert "LsLs" == "Ls" * 2
    with pytest.raises(ValueError, match="co-prime"):
        T.mos_landmarks(2, 2)


def test_degenerate_to_rejects_an_unknown_letter():
    with pytest.raises(ValueError, match="letter must be one of"):
        TernaryScale.equal_step("LMs").degenerate_to("x")


def test_an_edge_is_one_dimensional():
    """Fixing the counts on an edge leaves a single free parameter."""
    word = "LMLsLMs"
    a, b, _ = TernaryScale.equal_step(word).counts
    for large in (0.10, 0.15, 0.22):
        # s -> 0 : a*L + b*M = 1 pins M down from L, so the edge is a line.
        medium = (1.0 - a * large) / b
        eps = 1e-9
        scale = TernaryScale.from_barycentric(
            word, a * large, b * medium, eps
        )
        assert scale.large / scale.medium == pytest.approx(large / medium, rel=1e-6)


# --------------------------------------------------------------------------- #
# Simplex geometry
# --------------------------------------------------------------------------- #
def test_barycentric_to_xy_round_trip():
    rng = np.random.default_rng(3)
    for _ in range(200):
        u, v, w = rng.dirichlet((1.0, 1.0, 1.0))
        x, y = barycentric_to_xy(u, v, w)
        back = _xy_to_barycentric(np.array(x), np.array(y))
        assert [float(t) for t in back] == pytest.approx([u, v, w], abs=1e-12)


def test_the_triangle_is_equilateral():
    corners = [barycentric_to_xy(*b) for b in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    sides = [
        math.dist(corners[i], corners[j]) for i, j in ((0, 1), (1, 2), (2, 0))
    ]
    assert sides == pytest.approx([1.0, 1.0, 1.0], abs=1e-12)


@pytest.mark.parametrize(
    "word, counts", [("LMLsLMs", (3, 2, 2)), ("LMsLMsLMs", (3, 3, 3)),
                     ("LLLMLLs", (5, 1, 1))]
)
def test_canonical_region_area(word, counts):
    """One of six orderings, but ``b*c/((a+b)*N)`` of the area -- not a sixth."""
    a, b, c = counts
    n = a + b + c
    predicted = b * c / ((a + b) * n)
    rng = np.random.default_rng(11)
    pts = rng.dirichlet((1.0, 1.0, 1.0), size=60000)
    hits = np.count_nonzero(
        (pts[:, 0] / a > pts[:, 1] / b) & (pts[:, 1] / b > pts[:, 2] / c)
    )
    assert hits / len(pts) == pytest.approx(predicted, abs=0.006)
    if a == b == c:
        assert predicted == pytest.approx(1 / 6)
    else:
        assert abs(predicted - 1 / 6) > 0.02


def test_the_six_orderings_partition_the_simplex():
    a, b, c = 3, 2, 2
    rng = np.random.default_rng(5)
    pts = rng.dirichlet((1.0, 1.0, 1.0), size=20000)
    scaled = pts / np.array([a, b, c], float)
    total = 0
    for order in itertools.permutations(range(3)):
        i, j, k = order
        total += np.count_nonzero(
            (scaled[:, i] > scaled[:, j]) & (scaled[:, j] > scaled[:, k])
        )
    assert total == len(pts)  # ties have probability zero


def test_canonical_region_corners_are_where_they_should_be():
    rows = _canonical_vertices((3, 2, 2))
    assert rows[0] == pytest.approx([1.0, 0.0, 0.0])       # the u vertex
    assert rows[1] == pytest.approx([0.6, 0.4, 0.0])       # L = M on the w = 0 edge
    assert rows[2] == pytest.approx([3 / 7, 2 / 7, 2 / 7])  # the equal-step point
    # The corner where two of the letters tie is on the region's boundary, so
    # the interior of the region is strictly ordered.
    inside = rows.mean(axis=0)
    scale = TernaryScale.from_barycentric("LMLsLMs", *inside)
    assert scale.is_canonical


def test_vertices_are_equal_divisions():
    """Push two coordinates to nothing and the scale becomes an EDO."""
    word = "LMLsLMs"
    a, b, c = TernaryScale.equal_step(word).counts
    eps = 1e-7
    scale = TernaryScale.from_barycentric(word, 1.0 - 2 * eps, eps, eps)
    kept = [d for d, ch in zip(scale.degrees, word) if ch == "L"]
    assert len(kept) == a
    assert kept == pytest.approx([i / a for i in range(a)], abs=1e-5)


# --------------------------------------------------------------------------- #
# Words
# --------------------------------------------------------------------------- #
def test_ternary_words_of_3l2m2s():
    words = ternary_words(3, 2, 2)
    assert words == ["LMLsLMs", "LMLsMLs"]
    assert len(_necklaces(3, 2, 2)) == 30  # MV3 keeps 2 of 30
    for word in words:
        assert sampled_max_variety(word) == 3
    # The two are each other's reversal, which is why they share every metric.
    assert words[1] in _rotations(words[0][::-1])


def test_rotation_filter_keeps_one_representative_per_class():
    reps = ternary_words(3, 2, 2)
    everything = ternary_words(3, 2, 2, unique_up_to_rotation=False)
    assert len(everything) == 7 * len(reps)
    assert {_canonical_rotation(w) for w in everything} == set(reps)


def test_ternary_words_validation():
    with pytest.raises(ValueError, match="must be an int >= 1"):
        ternary_words(0, 2, 2)
    with pytest.raises(ValueError, match="already holds three distinct sizes"):
        ternary_words(3, 2, 2, max_variety=2)


def test_ternary_words_guards_the_combinatorics_above_14_notes():
    with pytest.raises(ValueError, match="max_words"):
        ternary_words(5, 5, 5)
    prefix = ternary_words(5, 5, 5, max_words=500)
    assert isinstance(prefix, list)
    assert all(len(w) == 15 for w in prefix)


def test_variety_sample_points_are_reproducible_and_interior():
    a = variety_sample_points()
    b = variety_sample_points()
    assert a == b
    assert len(a) == 7
    for u, v, w in a:
        assert min(u, v, w) >= 0.08
        assert u + v + w == pytest.approx(1.0, abs=1e-12)


def test_sampled_max_variety_cap_short_circuits():
    assert sampled_max_variety("LLLMMss") == 7
    capped = sampled_max_variety("LLLMMss", cap=3)
    assert capped > 3  # a lower bound is all a filter needs


def test_sampled_max_variety_keeps_sampling_until_the_cap_is_exceeded():
    """The cap may stop at ``> cap``, never at ``== cap``.

    Otherwise the multi-point sample degenerates into "whatever the first point
    said", which is the one thing this function exists not to do. On the line
    ``L = M`` the word ``LMLsLMs`` is not ternary at all -- it collapses to the
    diatonic ``5L2s``, which has Myhill's property and variety 2 -- so a first
    point there must not be allowed to answer for the whole simplex.
    """
    degenerate = (0.3, 0.2, 0.5)  # u/3 == v/2, i.e. L == M
    generic = (0.52, 0.30, 0.18)
    collapsed = TernaryScale.from_barycentric("LMLsLMs", *degenerate)
    assert collapsed.large == pytest.approx(collapsed.medium, abs=1e-12)
    assert collapsed.max_variety == 2
    assert TernaryScale.from_barycentric("LMLsLMs", *generic).max_variety == 3
    assert sampled_max_variety(
        "LMLsLMs", points=[degenerate, generic], cap=2
    ) == 3


# --------------------------------------------------------------------------- #
# Propriety over the simplex
# --------------------------------------------------------------------------- #
def test_no_ternary_word_is_improper_everywhere():
    """The brief's conjecture, refuted: the equal-step point saves every word."""
    fractions = {}
    for word in _necklaces(3, 2, 2):
        assert TernaryScale.equal_step(word).is_proper
        fractions[word] = proper_fraction(word, resolution=48)
    assert min(fractions.values()) > 0.0
    assert min(fractions.values()) == pytest.approx(0.043, abs=0.005)
    assert max(fractions.values()) == pytest.approx(0.112, abs=0.005)
    # The MV3 words sit at the top of the ordering.
    best = max(fractions, key=fractions.get)
    assert fractions["LMLsLMs"] == pytest.approx(fractions[best], abs=1e-9)


def test_proper_fraction_prefers_the_even_word():
    assert proper_fraction("LMLsLMs", resolution=48) > 2 * proper_fraction(
        "LLLMMss", resolution=48
    )


def test_proper_fraction_restricted_to_the_canonical_region():
    """In the ``L > M > s`` wedge the equal-step point is a corner, not a centre."""
    full = proper_fraction("LLLMLLs", resolution=72)
    canonical = proper_fraction("LLLMLLs", resolution=72, canonical_only=True)
    assert full == pytest.approx(0.070, abs=0.01)
    assert canonical == pytest.approx(0.42, abs=0.03)


def test_substitution_products_from_a_proper_mos_stay_proper_near_equal_step():
    word = mos_substitution(T.christoffel_word(2, 1), "LM", "LMs")
    assert word == "LMsLMLM"
    assert TernaryScale.equal_step(word).is_proper
    assert proper_fraction(word, resolution=48) > 0.05


# --------------------------------------------------------------------------- #
# Substitution
# --------------------------------------------------------------------------- #
def test_mos_substitution_builds_the_expected_string():
    assert mos_substitution("sLL", "LM", "LMs") == "LMsLMLM"
    assert mos_substitution(T.christoffel_word(5, 2), "LM", "s") == "sLMLMsLMLMLM"
    assert mos_substitution("Ls", "L", "s") == "Ls"  # not ternary; caller's problem


def test_mos_substitution_validation():
    with pytest.raises(ValueError, match="only contain 'L' and 's'"):
        mos_substitution("LMs", "L", "s")
    with pytest.raises(ValueError, match="non-empty"):
        mos_substitution("Ls", "", "s")
    with pytest.raises(ValueError, match="sub_small may only contain"):
        mos_substitution("Ls", "L", "x")


def test_substitution_beats_arbitrary_arrangement_at_reaching_mv3():
    """Measured, not assumed: 20.4% against 6.4% on this corpus, a 3.2x lift."""
    subs = [s for k in (1, 2) for s in map("".join, itertools.product("LMs", repeat=k))]
    products = set()
    for n_large in range(1, 6):
        for n_small in range(1, 6):
            if math.gcd(n_large, n_small) != 1 or not 2 <= n_large + n_small <= 6:
                continue
            base = T.christoffel_word(n_large, n_small)
            for sub_large in subs:
                for sub_small in subs:
                    word = mos_substitution(base, sub_large, sub_small)
                    if len(word) > 9 or len(set(word)) != 3:
                        continue
                    products.add(_canonical_rotation(word))

    sub_hits = sum(1 for w in products if sampled_max_variety(w, cap=3) <= 3)
    sub_rate = sub_hits / len(products)

    signatures = {(w.count("L"), w.count("M"), w.count("s")) for w in products}
    total = hits = 0
    for a, b, c in signatures:
        pool = _necklaces(a, b, c)
        total += len(pool)
        hits += sum(1 for w in pool if sampled_max_variety(w, cap=3) <= 3)
    base_rate = hits / total

    assert (len(products), sub_hits) == (113, 23)
    assert (total, hits) == (951, 61)
    assert sub_rate == pytest.approx(0.2035, abs=0.001)
    assert base_rate == pytest.approx(0.0641, abs=0.001)
    assert sub_rate / base_rate > 2.5


# --------------------------------------------------------------------------- #
# Atlas
# --------------------------------------------------------------------------- #
def test_ternary_atlas_at_seven_notes():
    atlas = ternary_atlas(7, proper_resolution=12)
    assert isinstance(atlas, pd.DataFrame)
    assert len(atlas) == 15  # every 7-note signature admits an MV3 word
    assert int(atlas["n_words"].sum()) == 24
    assert set(atlas["cardinality"]) == {7}
    assert (atlas["n_large"] + atlas["n_medium"] + atlas["n_small"] == 7).all()
    # Sorted by how many words each signature admits.
    assert list(atlas["n_words"]) == sorted(atlas["n_words"], reverse=True)
    # The smoke-test column: true for every scale, since equal step is N-EDO.
    assert bool(atlas["equal_step_proper"].all())
    for _, row in atlas.iterrows():
        assert row["example_word"] in ternary_words(
            row["n_large"], row["n_medium"], row["n_small"]
        )
        assert row["signature"] == (
            f"{row['n_large']}L{row['n_medium']}M{row['n_small']}s"
        )


def test_ternary_atlas_thins_out_with_cardinality():
    """MV3 gets much harder to satisfy as the scale grows."""
    counts = {n: len(ternary_atlas(n, proper_resolution=8)) for n in (5, 7, 9)}
    assert counts == {5: 6, 7: 15, 9: 10}
    # 28 signatures exist at 9 notes but only 10 admit an MV3 word.
    assert counts[9] < (9 - 1) * (9 - 2) // 2


def test_ternary_atlas_rejects_tiny_cardinalities():
    with pytest.raises(ValueError, match="at least 3 notes"):
        ternary_atlas(2)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("field", ["propriety", "variety", "ji_error", None])
def test_plot_ternary_simplex_runs_for_every_field(field):
    fig, ax = plot_ternary_simplex("LMLsLMs", field=field, resolution=30)
    assert ax.get_aspect() == 1.0
    assert len(ax.images) == (0 if field is None else 1)
    # The triangle must not be squashed: xlim spans more than ylim.
    assert ax.get_xlim()[1] - ax.get_xlim()[0] > ax.get_ylim()[1] - ax.get_ylim()[0]


def test_plot_ternary_simplex_marks_a_tuning():
    scale = TernaryScale.from_barycentric("LMLsLMs", 0.52, 0.30, 0.18)
    fig, ax = plot_ternary_simplex("LMLsLMs", field=None, mark=scale)
    starred = [ln for ln in ax.lines if ln.get_marker() == "*"]
    assert len(starred) == 1
    assert starred[0].get_xdata()[0] == pytest.approx(scale.xy[0])
    fig, ax = plot_ternary_simplex("LMLsLMs", field=None, mark=(1, 1, 1))
    assert [ln for ln in ax.lines if ln.get_marker() == "*"]


def test_the_propriety_contour_is_registered_on_the_field_it_bounds():
    """The drawn boundary must sit where the field changes sign, not half a
    cell away.

    ``1L1M1s`` makes this exact rather than approximate. For a three-note scale
    ``max(class 1)`` is the largest step and ``min(class 2)`` is the period
    minus it, so the margin is ``P - 2*max(L, M, s)``: a piecewise-linear
    function whose zero set is precisely the medial triangle of the simplex,
    with corners at the three edge midpoints. Marching squares on a linear
    field reproduces a straight zero level exactly, so every contour vertex
    must land on that triangle to within float noise -- and the proper area is
    1/4, which is what ``proper_fraction('LMs')`` reports.
    """
    resolution = 24
    fig, ax = plot_ternary_simplex("LMs", field="propriety", resolution=resolution)
    verts = np.vstack(
        [
            path.vertices
            for coll in ax.collections
            for path in coll.get_paths()
            if len(path.vertices) > 3
        ]
    )
    assert len(verts) > 10
    half_height = math.sqrt(3.0) / 4.0
    midpoints = np.array([[0.5, 0.0], [0.25, half_height], [0.75, half_height]])

    def distance_to_segment(points, start, end):
        span = end - start
        t = np.clip((points - start) @ span / (span @ span), 0.0, 1.0)
        return np.linalg.norm(points - (start + t[:, None] * span), axis=1)

    gaps = np.min(
        np.stack(
            [
                distance_to_segment(verts, midpoints[i], midpoints[(i + 1) % 3])
                for i in range(3)
            ]
        ),
        axis=0,
    )
    # Half a grid cell is 1/48 here; the old full-extent coordinates missed by
    # exactly that, so this tolerance is an order of magnitude inside the bug.
    assert gaps.max() < 0.5 / resolution / 10.0
    assert proper_fraction("LMs", resolution=96) == pytest.approx(0.25, abs=0.01)


def test_plot_ternary_simplex_rejects_an_unknown_field():
    with pytest.raises(ValueError, match="field must be"):
        plot_ternary_simplex("LMLsLMs", field="hardness")


def test_plot_ternary_atlas_panels():
    fig, axes = plot_ternary_atlas(7, max_panels=6, resolution=20)
    # The whole 4x2 grid comes back; the two unused cells are blanked.
    assert len(axes) == 8
    drawn = [ax for ax in axes if ax.images]
    assert len(drawn) == 6
    assert all(ax.get_title() for ax in drawn)
    assert not any(ax.get_title() for ax in axes if not ax.images)
    # One shared colour normalisation across the panels.
    assert len({id(ax.images[0].norm) for ax in drawn}) == 1


def test_plot_ternary_atlas_handles_a_short_grid():
    fig, axes = plot_ternary_atlas(5, max_panels=12, resolution=16)
    assert len(axes) >= 6
