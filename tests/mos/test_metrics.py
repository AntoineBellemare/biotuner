"""Tests for biotuner.mos.metrics.

The structural claims of Milne et al. (2011) §2 -- Myhill's property, unique
degree signatures, and the tie between propriety and Blackwood's R -- are
treated here as hypotheses to be measured, not as facts to be assumed.  Each is
swept across every co-prime signature up to 13 notes at several tunings, and
paired with hand-built non-MOS scales that must fail it.

One claim did not survive the sweep and is recorded as such:
``MOSScale.is_proper`` short-cuts propriety as ``R <= 2``, which is wrong for
every signature with a single small step.  See
``test_propriety_shortcut_is_wrong_when_there_is_one_small_step``.
"""

import json
import math

import numpy as np
import pytest

from biotuner.mos.metrics import (
    MODEL_PARAMETERS,
    MOSness,
    _adjusted_error,
    _equal_division,
    _share,
    _ternary_words_at,
    blackwood_r,
    degree_signatures,
    evenness,
    generic_interval_sizes,
    harmonicity,
    has_unique_degree_signatures,
    interval_matrix,
    is_proper,
    ji_error,
    mos_ness,
    mos_report,
    myhill_property,
)
from biotuner.mos.scale import MOSScale
from biotuner.mos.ternary import TernaryScale

PERIOD = 1200.0

#: 12-EDO diatonic as built by stacking -- the brightest mode, Lydian.
LYDIAN_12 = [0.0, 200.0, 400.0, 600.0, 700.0, 900.0, 1100.0]


# --------------------------------------------------------------------------- #
# Scale corpus
# --------------------------------------------------------------------------- #
def _signatures(max_cardinality=13):
    return [
        (n_large, n_small)
        for n_large in range(1, max_cardinality)
        for n_small in range(1, max_cardinality)
        if math.gcd(n_large, n_small) == 1
        and 3 <= n_large + n_small <= max_cardinality
    ]


def _corpus(max_cardinality=13, tunings=(None, "central", "middle", 12, 19, 31)):
    """Every non-degenerate MOS we can build, over both mirror ranges.

    Skips two kinds of collapsed scale.  ``is_degenerate`` catches the equalized
    landmark, where L and s coincide.  The infinite-hardness check catches the
    *other* endpoint: ``MOSScale.from_signature(2, 5, tuning=12)`` resolves to
    g = 1/2 exactly, the tuning at which the large steps vanish, leaving a
    "7-note" scale with two distinct pitches.  Neither has intervals to measure.
    """
    out = []
    for n_large, n_small in _signatures(max_cardinality):
        for bright in (True, False):
            for tuning in tunings:
                try:
                    scale = MOSScale.from_signature(
                        n_large, n_small, tuning=tuning, bright=bright
                    )
                except ValueError:
                    continue
                if scale.is_degenerate or not math.isfinite(scale.hardness):
                    continue
                out.append(scale)
    return out


CORPUS = _corpus()

#: Hand-built scales that are *not* MOS, each failing a different way.
NON_MOS = {
    # Three step sizes (200/100/300) -- fails at the very first class.
    "harmonic_minor": [0.0, 200.0, 300.0, 500.0, 700.0, 800.0, 1100.0],
    # Exactly two step sizes, but the two small ones are adjacent instead of
    # maximally even, so the failure only shows up from class 2 onward.
    "LLLLLss": [0.0, 200.0, 400.0, 600.0, 800.0, 1000.0, 1100.0],
    # Transpositionally symmetric: repeats every 300 cents.
    "octatonic": [0.0, 100.0, 300.0, 400.0, 600.0, 700.0, 900.0, 1000.0],
    "blues": [0.0, 300.0, 500.0, 600.0, 700.0, 1000.0],
}


def test_corpus_is_big_enough_to_mean_something():
    assert len(CORPUS) > 300
    assert len({s.signature for s in CORPUS}) > 30


# --------------------------------------------------------------------------- #
# interval_matrix
# --------------------------------------------------------------------------- #
def test_interval_matrix_12edo_diatonic_by_hand():
    m = interval_matrix(MOSScale.from_signature(5, 2, tuning=12))
    assert m.shape == (7, 6)
    # Row 0 is Lydian measured from its root.
    np.testing.assert_allclose(m[0], [200, 400, 600, 700, 900, 1100], atol=1e-9)
    # Column 0 is the step pattern LLLsLLs.
    np.testing.assert_allclose(m[:, 0], [200, 200, 200, 100, 200, 200, 100], atol=1e-9)
    # The row rooted on the tritone wraps through the period.
    np.testing.assert_allclose(m[3], [100, 300, 500, 600, 800, 1000], atol=1e-9)


def test_interval_matrix_raw_pair_by_hand():
    m = interval_matrix(([0.0, 100.0, 700.0], 1200.0))
    np.testing.assert_allclose(
        m, [[100.0, 700.0], [600.0, 1100.0], [500.0, 600.0]], atol=1e-9
    )


def test_interval_matrix_accepts_scale_and_equivalent_raw_pair():
    scale = MOSScale.from_generator(3 / 2, 12)
    from_object = interval_matrix(scale)
    from_pair = interval_matrix((scale.cents, scale.period_cents))
    np.testing.assert_allclose(from_object, from_pair, atol=1e-12)


def test_interval_matrix_accepts_a_mode():
    scale = MOSScale.from_signature(5, 2, tuning=12)
    ionian = interval_matrix(scale.mode(1))
    np.testing.assert_allclose(ionian[0], [200, 400, 500, 700, 900, 1100], atol=1e-9)


def test_interval_matrix_sorts_an_unsorted_raw_list():
    a = interval_matrix(([700.0, 0.0, 100.0], 1200.0))
    b = interval_matrix(([0.0, 100.0, 700.0], 1200.0))
    np.testing.assert_allclose(a, b, atol=1e-12)


@pytest.mark.parametrize("scale", CORPUS[::17], ids=lambda s: repr(s))
def test_interval_matrix_column_sums_to_k_periods(scale):
    """Summing class k over all degrees walks the scale k times round."""
    m = interval_matrix(scale)
    for k in range(1, m.shape[1] + 1):
        assert m[:, k - 1].sum() == pytest.approx(k * scale.period_cents, abs=1e-6)


@pytest.mark.parametrize("scale", CORPUS[::17], ids=lambda s: repr(s))
def test_interval_matrix_complements_add_to_the_period(scale):
    """A k-step and the (N-k)-step above it must close the period."""
    m = interval_matrix(scale)
    n = scale.cardinality
    for i in range(n):
        for k in range(1, n):
            complement = m[(i + k) % n, n - k - 1]
            assert m[i, k - 1] + complement == pytest.approx(
                scale.period_cents, abs=1e-6
            )


@pytest.mark.parametrize("scale", CORPUS[::23], ids=lambda s: repr(s))
def test_interval_matrix_matches_a_reconstruction_from_the_word(scale):
    """Independent build: sum runs of L/s straight off the step pattern."""
    large, small = scale.step_cents
    steps = [large if c == "L" else small for c in scale.word]
    n = len(steps)
    expected = np.array(
        [[sum(steps[(i + j) % n] for j in range(k)) for k in range(1, n)]
         for i in range(n)]
    )
    np.testing.assert_allclose(interval_matrix(scale), expected, atol=1e-6)


#: A fifth stacked inside a tritave -- the standard trap for a hardcoded 1200.
TRITAVE = MOSScale.from_generator(3 / 2, 8, period=3.0)


def test_pseudo_octave_period_is_honoured():
    """A fifth inside a tritave: 8 notes, 1902-cent period, still measurable."""
    scale = TRITAVE
    m = interval_matrix(scale)
    assert scale.signature == "3L5s"
    assert scale.period_cents == pytest.approx(1200.0 * math.log2(3.0))
    assert m.shape == (8, 7)
    assert m[:, -1].max() < scale.period_cents
    assert m[:, 0].sum() == pytest.approx(scale.period_cents, abs=1e-6)
    assert myhill_property(scale) is True
    assert has_unique_degree_signatures(scale) is True


def test_pseudo_octave_evenness_divides_the_tritave_not_the_octave():
    """``evenness`` must normalise by ``period_cents``, never by 1200.

    Hand-derived: the tritave scale's degrees are 0, 203.910, 407.820, 701.955,
    905.865, 1109.775, 1403.910, 1607.820 cents against a 1901.955-cent period,
    so ``max |c_i / 1901.955 - i / 8|`` is 0.0415083.  Dividing by 1200 instead
    would give 0.4648 -- an order of magnitude out, and larger than 1/8, which
    is impossible for a genuine deviation from an equal division.
    """
    cents = TRITAVE.cents
    period = 1200.0 * math.log2(3.0)
    by_hand = max(abs(c / period - i / 8) for i, c in enumerate(cents))
    assert by_hand == pytest.approx(0.0415083, abs=1e-6)
    assert evenness(TRITAVE) == pytest.approx(by_hand, abs=1e-12)
    # The failure mode this pins down, spelled out so it cannot creep back.
    assert max(abs(c / 1200.0 - i / 8) for i, c in enumerate(cents)) > 0.46


def test_pseudo_octave_ji_error_wraps_at_the_tritave():
    """A target is folded into the *period*, so 2/1 is an interior interval.

    Inside a tritave the octave is not an equivalence -- it is just another
    ratio, 1200 cents up, and the nearest degree to it is the 1109.775-cent
    one, 90.225 cents below.  Folding at a hardcoded 1200 would map 2/1 onto
    the root and report a perfect 0.
    """
    err = ji_error(TRITAVE, [2.0])
    assert err["errors"][0] == pytest.approx(90.225, abs=1e-3)
    # 3/1 is the period itself and must wrap to the root exactly.
    assert ji_error(TRITAVE, [3.0])["errors"][0] == pytest.approx(0.0, abs=1e-9)
    # 3/2 is the generator, present exactly.
    assert ji_error(TRITAVE, [3 / 2])["errors"][0] == pytest.approx(0.0, abs=1e-9)
    # And the wrap really is at the period, not the octave: a target just under
    # the tritave is a small negative error against the root.
    just_under = 3.0 * 2.0 ** (-5.0 / 1200.0)
    assert ji_error(TRITAVE, [just_under])["errors"][0] == pytest.approx(
        -5.0, abs=1e-6
    )


def test_pseudo_octave_report_agrees_with_the_scale_object():
    report = mos_report(TRITAVE, harmonic=False)
    assert report["period_cents"] == pytest.approx(1200.0 * math.log2(3.0))
    assert report["blackwood_r"] == pytest.approx(TRITAVE.hardness, rel=1e-9)
    assert report["myhill"] is True
    assert report["evenness"] == pytest.approx(0.0415083, abs=1e-6)


@pytest.mark.parametrize(
    "bad, exc, message",
    [
        (([0.0], 1200.0), ValueError, "at least 2 degrees"),
        (([0.0, 0.0, 700.0], 1200.0), ValueError, "distinct"),
        (([0.0, 1300.0], 1200.0), ValueError, "must lie in"),
        # The period itself is the *next* octave of the root, not a degree.
        (([0.0, 700.0, 1200.0], 1200.0), ValueError, "must lie in"),
        (([0.0, -50.0], 1200.0), ValueError, "must lie in"),
        (([0.0, 700.0], 0.0), ValueError, "period_cents"),
        (([0.0, float("nan")], 1200.0), ValueError, "finite"),
        ("5L2s", TypeError, "expected an MOSScale"),
        ((1200.0, 1200.0), TypeError, "cents_list"),
    ],
)
def test_interval_matrix_rejects_bad_input(bad, exc, message):
    with pytest.raises(exc) as info:
        interval_matrix(bad)
    assert message in str(info.value)


# --------------------------------------------------------------------------- #
# generic_interval_sizes
# --------------------------------------------------------------------------- #
def test_generic_interval_sizes_12edo_diatonic():
    sizes = generic_interval_sizes(MOSScale.from_signature(5, 2, tuning=12))
    assert {k: [round(v) for v in vs] for k, vs in sizes.items()} == {
        1: [100, 200],
        2: [300, 400],
        3: [500, 600],
        4: [600, 700],
        5: [800, 900],
        6: [1000, 1100],
    }


@pytest.mark.parametrize("scale", CORPUS[::11], ids=lambda s: repr(s))
def test_the_two_sizes_of_a_class_differ_by_exactly_the_chroma(scale):
    """L - s is the *only* difference any generic class can show.

    Class k holds m or m+1 large steps; swapping one s for one L is the whole
    story, so every class's spread equals the augmented prime.
    """
    large, small = scale.step_cents
    for sizes in generic_interval_sizes(scale).values():
        assert len(sizes) == 2
        assert sizes[1] - sizes[0] == pytest.approx(large - small, abs=1e-6)


def test_degenerate_tuning_has_one_size_per_class():
    equal = MOSScale.from_signature(5, 2, tuning="equalized")
    sizes = generic_interval_sizes(equal)
    assert [len(v) for v in sizes.values()] == [1] * 6
    assert [round(v[0]) for v in sizes.values()] == [171, 343, 514, 686, 857, 1029]


# --------------------------------------------------------------------------- #
# Myhill's property
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scale", CORPUS, ids=lambda s: repr(s))
def test_every_non_degenerate_mos_has_myhills_property(scale):
    """Milne et al. §2's first structural claim, over the whole corpus."""
    assert myhill_property(scale) is True


def test_myhill_is_false_for_a_degenerate_tuning():
    assert myhill_property(MOSScale.from_signature(5, 2, tuning="equalized")) is False


@pytest.mark.parametrize("name", sorted(NON_MOS))
def test_non_mos_scales_fail_myhill(name):
    assert myhill_property((NON_MOS[name], PERIOD)) is False


def test_the_subtle_non_mos_passes_class_one_and_fails_class_two():
    """Two step sizes is not enough -- they have to be spread evenly.

    LLLLLss has the diatonic's step counts with the small steps bunched
    together.  Class 1 looks fine; class 2 gives it away with three sizes.
    """
    sizes = generic_interval_sizes((NON_MOS["LLLLLss"], PERIOD))
    assert len(sizes[1]) == 2
    assert sizes[2] == [200.0, 300.0, 400.0]
    assert myhill_property((NON_MOS["LLLLLss"], PERIOD)) is False


def test_myhill_failure_is_not_the_same_as_impropriety():
    """The harmonic minor is proper, well-behaved, and still not well-formed."""
    harmonic_minor = (NON_MOS["harmonic_minor"], PERIOD)
    assert is_proper(harmonic_minor) is True
    assert has_unique_degree_signatures(harmonic_minor) is True
    assert myhill_property(harmonic_minor) is False


# --------------------------------------------------------------------------- #
# Propriety
# --------------------------------------------------------------------------- #
def test_pythagorean_diatonic_is_improper():
    pythagorean = MOSScale.from_generator(3 / 2, 7)
    assert round(pythagorean.hardness, 3) == 2.26
    assert is_proper(pythagorean) is False
    assert pythagorean.is_proper is False


def test_12edo_diatonic_is_proper_but_not_strictly():
    twelve = MOSScale.from_signature(5, 2, tuning=12)
    assert twelve.hardness == pytest.approx(2.0)
    assert is_proper(twelve) is True
    assert is_proper(twelve, strict=True) is False
    # The tie is the tritone: an augmented fourth equal to a diminished fifth.
    sizes = generic_interval_sizes(twelve)
    assert max(sizes[3]) == pytest.approx(min(sizes[4]))


def test_31edo_meantone_diatonic_is_strictly_proper():
    meantone = MOSScale.from_signature(5, 2, tuning=31)
    assert meantone.generator_cents == pytest.approx(696.774, abs=1e-3)
    assert is_proper(meantone, strict=True) is True


@pytest.mark.parametrize(
    "scale", [s for s in CORPUS if s.n_small >= 2], ids=lambda s: repr(s)
)
def test_matrix_propriety_agrees_with_the_hardness_shortcut(scale):
    """The independent check on ``MOSScale.is_proper``, where it is valid."""
    assert is_proper(scale) == scale.is_proper
    assert is_proper(scale, strict=True) == (scale.hardness < 2.0 - 1e-9)


@pytest.mark.parametrize(
    "scale", [s for s in CORPUS if s.n_small == 1], ids=lambda s: repr(s)
)
def test_propriety_shortcut_is_wrong_when_there_is_one_small_step(scale):
    """``R <= 2`` is not equivalent to propriety for ``nL L 1s``.

    Propriety asks ``max(class k) <= min(class k+1)``.  Writing ``m_k`` for the
    number of large steps in the small variant of class k, the comparison
    reduces to ``(1 - d)L <= (2 - d)s`` with ``d = m_{k+1} - m_k`` in {0, 1}.
    A ``d = 1`` transition is free; only ``d = 0`` transitions impose
    ``L <= 2s``.  Over classes 1..N-2 there are exactly ``n_small - 1`` of the
    latter, so a scale with a single small step has *no* propriety constraint
    at all and stays proper however hard it gets.

    ``MOSScale.is_proper`` (``hardness <= 2``) therefore reports false
    negatives for this family -- e.g. 2L1s at L=500 c, s=200 c is plainly
    proper (its largest second, 500 c, is well under its smallest third,
    700 c) while R = 2.5.
    """
    assert is_proper(scale) is True
    assert is_proper(scale, strict=True) is True
    if scale.hardness > 2.0 + 1e-9:
        assert scale.is_proper is False, "expected the shortcut to disagree here"


def test_the_one_small_step_counterexample_in_full():
    scale = MOSScale.from_signature(2, 1, tuning="middle", bright=False)
    assert [round(c) for c in scale.cents] == [0, 500, 1000]
    assert scale.word == "LLs"
    assert scale.hardness == pytest.approx(2.5)
    sizes = generic_interval_sizes(scale)
    assert [round(v) for v in sizes[1]] == [200, 500]
    assert [round(v) for v in sizes[2]] == [700, 1000]
    # Largest second (500 c) comfortably under smallest third (700 c).
    assert max(sizes[1]) < min(sizes[2])
    assert is_proper(scale) is True
    assert is_proper(scale, strict=True) is True
    assert scale.is_proper is False


def test_strictness_is_stronger_than_propriety():
    for scale in CORPUS[::13]:
        if is_proper(scale, strict=True):
            assert is_proper(scale)


def test_improper_non_mos():
    assert is_proper((NON_MOS["blues"], PERIOD)) is False


# --------------------------------------------------------------------------- #
# Blackwood R
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scale", CORPUS[::7], ids=lambda s: repr(s))
def test_blackwood_r_reproduces_scale_hardness(scale):
    assert blackwood_r(scale) == pytest.approx(scale.hardness, rel=1e-9)


def test_blackwood_r_known_values():
    assert blackwood_r(MOSScale.from_generator(3 / 2, 7)) == pytest.approx(
        2.26, abs=5e-3
    )
    assert blackwood_r(MOSScale.from_signature(5, 2, tuning=12)) == pytest.approx(2.0)
    assert blackwood_r(MOSScale.from_signature(5, 2, tuning=31)) == pytest.approx(
        5 / 3
    )
    assert blackwood_r(MOSScale.from_signature(5, 2, tuning="equalized")) == (
        pytest.approx(1.0)
    )


def test_blackwood_r_grows_without_bound_as_a_step_shrinks():
    # Steps 0.5 / 699.5 / 500 cents.
    assert blackwood_r(([0.0, 0.5, 700.0], 1200.0)) == pytest.approx(1399.0, rel=1e-9)
    assert blackwood_r(([0.0, 0.05, 700.0], 1200.0)) == pytest.approx(13999.0, rel=1e-6)
    # A step of exactly zero is unreachable: duplicate degrees are rejected.
    with pytest.raises(ValueError):
        blackwood_r(([0.0, 700.0, 700.0], 1200.0))


# --------------------------------------------------------------------------- #
# Degree signatures
# --------------------------------------------------------------------------- #
def test_degree_signatures_of_the_12edo_diatonic():
    sigs = degree_signatures(MOSScale.from_signature(5, 2, tuning=12))
    assert sigs == [
        (1, 1, 1, 1, 1, 1),
        (1, 1, 0, 1, 1, 0),
        (1, 0, 0, 1, 0, 0),
        (0, 0, 0, 0, 0, 0),
        (1, 1, 0, 1, 1, 1),
        (1, 0, 0, 1, 1, 0),
        (0, 0, 0, 1, 0, 0),
    ]
    # Degree 0 opens the chain of fifths in Lydian, degree 3 (the sharp fourth)
    # closes it, so those two are the all-large and all-small extremes.
    assert sigs[0] == (1,) * 6
    assert sigs[3] == (0,) * 6


@pytest.mark.parametrize("scale", CORPUS, ids=lambda s: repr(s))
def test_every_non_degenerate_mos_has_unique_degree_signatures(scale):
    """Milne et al. §2's second structural claim, over the whole corpus."""
    assert has_unique_degree_signatures(scale) is True
    sigs = degree_signatures(scale)
    assert len(sigs) == scale.cardinality
    assert all(set(s) <= {0, 1} for s in sigs)


def test_degenerate_tuning_has_indistinguishable_degrees():
    equal = MOSScale.from_signature(5, 2, tuning="equalized")
    assert has_unique_degree_signatures(equal) is False
    assert set(degree_signatures(equal)) == {(0,) * 6}


def test_a_symmetric_non_mos_repeats_its_degree_signatures():
    """The octatonic maps onto itself every minor third, so degrees pair up."""
    octatonic = (NON_MOS["octatonic"], PERIOD)
    sigs = degree_signatures(octatonic)
    assert has_unique_degree_signatures(octatonic) is False
    assert len(set(sigs)) == 2


@pytest.mark.parametrize("scale", CORPUS[::19], ids=lambda s: repr(s))
def test_large_signature_entries_count_the_large_intervals(scale):
    """Column k of the signature table holds exactly the scales' large variants.

    For class k the large variant occurs ``k * n_large mod N`` times when that
    is nonzero -- the three-distance count.  Checking the tally rather than the
    pattern catches a mis-ranked cluster.
    """
    n = scale.cardinality
    sigs = degree_signatures(scale)
    for k in range(1, n):
        n_large_here = sum(s[k - 1] for s in sigs)
        assert n_large_here == (k * scale.n_large) % n


# --------------------------------------------------------------------------- #
# Evenness
# --------------------------------------------------------------------------- #
def test_evenness_is_zero_for_an_equal_division():
    assert evenness(MOSScale.from_signature(5, 2, tuning="equalized")) == (
        pytest.approx(0.0, abs=1e-12)
    )
    assert evenness((NON_MOS["octatonic"], PERIOD)) > 0


def test_evenness_of_the_12edo_diatonic_is_one_fourteenth():
    """The tritone sits a half-step above 7-EDO's fourth degree."""
    assert evenness(MOSScale.from_signature(5, 2, tuning=12)) == pytest.approx(1 / 14)


def test_evenness_grows_with_hardness():
    tunings = ["equalized", 19, 31, 12]
    scales = [MOSScale.from_signature(5, 2, tuning=t) for t in tunings]
    hardness = [s.hardness for s in scales]
    values = [evenness(s) for s in scales]
    assert hardness == sorted(hardness)
    assert values == sorted(values)
    assert values[0] < values[-1]


@pytest.mark.parametrize("scale", CORPUS[::13], ids=lambda s: repr(s))
def test_evenness_is_bounded_by_the_largest_step(scale):
    large, _ = scale.step_cents
    assert 0.0 <= evenness(scale) < large / scale.period_cents


# --------------------------------------------------------------------------- #
# JI error
# --------------------------------------------------------------------------- #
def test_ji_error_known_12edo_deviations():
    twelve = MOSScale.from_signature(5, 2, tuning=12)
    err = ji_error(twelve, [3 / 2, 5 / 4, 6 / 5])
    assert [round(e, 2) for e in err["errors"]] == [1.96, -13.69, -84.36]
    assert err["mean_abs"] == pytest.approx(33.3333, abs=1e-3)
    assert err["max_abs"] == pytest.approx(84.3587, abs=1e-3)
    assert err["rms"] == pytest.approx(49.3543, abs=1e-3)
    assert err["rms"] >= err["mean_abs"]


def test_ji_error_default_weights_are_uniform():
    twelve = MOSScale.from_signature(5, 2, tuning=12)
    targets = [3 / 2, 5 / 4, 6 / 5]
    plain = ji_error(twelve, targets)
    weighted = ji_error(twelve, targets, weights=[7.0, 7.0, 7.0])
    assert weighted["weighted_mean"] == pytest.approx(plain["mean_abs"])
    assert plain["weighted_mean"] == pytest.approx(plain["mean_abs"])


def test_ji_error_weights_are_normalised_not_summed():
    twelve = MOSScale.from_signature(5, 2, tuning=12)
    targets = [3 / 2, 5 / 4]
    a = ji_error(twelve, targets, weights=[1.0, 3.0])
    b = ji_error(twelve, targets, weights=[10.0, 30.0])
    assert a["weighted_mean"] == pytest.approx(b["weighted_mean"])
    assert a["weighted_mean"] == pytest.approx(
        0.25 * abs(a["errors"][0]) + 0.75 * abs(a["errors"][1])
    )


def test_ji_error_period_reduction_changes_the_match():
    twelve = MOSScale.from_signature(5, 2, tuning=12)
    wrapped = ji_error(twelve, [2.0], period_reduce=True)
    raw = ji_error(twelve, [2.0], period_reduce=False)
    assert wrapped["errors"][0] == pytest.approx(0.0)
    assert raw["errors"][0] == pytest.approx(100.0)


def test_ji_error_matches_a_target_across_the_period_boundary():
    """A target 5 cents under the octave is 5 cents from the root, not 95 from B."""
    twelve = MOSScale.from_signature(5, 2, tuning=12)
    almost_octave = 2.0 ** (1195.0 / 1200.0)
    err = ji_error(twelve, [almost_octave])
    assert err["errors"][0] == pytest.approx(-5.0, abs=1e-6)


@pytest.mark.parametrize("scale", CORPUS[::29], ids=lambda s: repr(s))
def test_ji_error_never_exceeds_half_the_largest_step(scale):
    large, _ = scale.step_cents
    targets = [1.1, 1.25, 4 / 3, 1.5, 1.75, 1.9]
    err = ji_error(scale, targets)
    assert err["max_abs"] <= large / 2.0 + 1e-9


def test_ji_error_shrinks_as_the_fifth_is_tuned_toward_just():
    just_fifth = [3 / 2]
    pythagorean = ji_error(MOSScale.from_generator(3 / 2, 7), just_fifth)
    twelve = ji_error(MOSScale.from_signature(5, 2, tuning=12), just_fifth)
    meantone = ji_error(MOSScale.from_signature(5, 2, tuning=31), just_fifth)
    assert pythagorean["max_abs"] < twelve["max_abs"] < meantone["max_abs"]
    assert pythagorean["max_abs"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    "targets, weights, message",
    [
        ([], None, "at least one"),
        ([0.0], None, "positive"),
        ([3 / 2], [1.0, 1.0], "match targets"),
        ([3 / 2], [-1.0], "non-negative"),
        ([3 / 2, 5 / 4], [0.0, 0.0], "sum to zero"),
        # nan is not caught by a `t <= 0` test -- every comparison against it is
        # False -- so it has to be rejected explicitly or it silently turns the
        # errors, mean_abs, max_abs and rms of a whole report into nan.
        ([float("nan")], None, "finite"),
        ([3 / 2, float("inf")], None, "finite"),
        ([3 / 2, 5 / 4], [1.0, float("nan")], "finite"),
        ([3 / 2, 5 / 4], [1.0, float("inf")], "finite"),
    ],
)
def test_ji_error_rejects_bad_targets_and_weights(targets, weights, message):
    twelve = MOSScale.from_signature(5, 2, tuning=12)
    with pytest.raises(ValueError) as info:
        ji_error(twelve, targets, weights=weights)
    assert message in str(info.value)


# --------------------------------------------------------------------------- #
# Harmonicity
# --------------------------------------------------------------------------- #
def test_harmonicity_returns_biotuner_tuning_metrics():
    metrics = harmonicity(MOSScale.from_signature(5, 2, tuning=12))
    assert set(metrics) >= {
        "sum_p_q",
        "sum_distinct_intervals",
        "metric_3",
        "harm_sim",
        "matrix_harm_sim",
        "matrix_cons",
        "matrix_denom",
    }
    assert all(isinstance(v, float) for v in metrics.values())
    assert all(math.isfinite(v) for v in metrics.values())
    assert metrics["harm_sim"] == pytest.approx(14.46, abs=1e-6)


def test_harmonicity_ranks_a_just_scale_above_an_equal_one():
    """A rational tuning must score better than an irrational one on harm_sim."""
    just = harmonicity(([0.0, 386.31, 701.96], 1200.0))
    equal = harmonicity(MOSScale.from_signature(5, 2, tuning="equalized"))
    assert just["harm_sim"] > equal["harm_sim"]


def test_harmonicity_swallows_a_failure_and_returns_an_empty_dict(monkeypatch):
    import biotuner.metrics as bm

    def boom(*args, **kwargs):
        raise RuntimeError("rationalisation blew up")

    monkeypatch.setattr(bm, "tuning_to_metrics", boom)
    assert harmonicity(MOSScale.from_signature(5, 2, tuning=12)) == {}


def test_harmonicity_works_on_a_raw_scale():
    metrics = harmonicity(([0.0, 400.0, 700.0], 1200.0))
    assert isinstance(metrics, dict)
    assert metrics["harm_sim"] == pytest.approx(33.51, abs=1e-6)
    assert metrics["sum_distinct_intervals"] == pytest.approx(5.0)


def test_harmonicity_of_a_raw_pair_matches_the_equivalent_scale_object():
    """The cents -> ratio fallback must reproduce the object's own ratios.

    ``harmonicity`` reads ``.ratios`` straight off a scale object and only
    falls back to ``2 ** (cents / 1200)`` for a raw pair, so the two paths are
    otherwise never compared.  Several of the underlying metrics are
    order-sensitive (reversing the ratio list moves ``sum_distinct_intervals``
    from 32 to 42), which is what makes this equality a real check on the
    fallback rather than a restatement of it.
    """
    twelve = MOSScale.from_signature(5, 2, tuning=12)
    from_object = harmonicity(twelve)
    from_pair = harmonicity((list(twelve.cents), twelve.period_cents))
    assert from_pair == from_object
    assert from_object["sum_distinct_intervals"] == pytest.approx(32.0)
    # The ratios really are the ascending 12-EDO ones, not some permutation.
    ratios = [2.0 ** (c / 1200.0) for c in twelve.cents]
    assert ratios == sorted(ratios)
    np.testing.assert_allclose(ratios, twelve.ratios, atol=1e-12)


# --------------------------------------------------------------------------- #
# mos_report
# --------------------------------------------------------------------------- #
def test_mos_report_merges_identity_and_metrics():
    report = mos_report(
        MOSScale.from_signature(5, 2, tuning=12), targets=[3 / 2, 5 / 4]
    )
    assert report["signature"] == "5L2s"
    assert report["cardinality"] == 7
    assert report["myhill"] is True
    assert report["proper_from_matrix"] is True
    assert report["strictly_proper"] is False
    assert report["unique_degree_signatures"] is True
    assert report["blackwood_r"] == pytest.approx(2.0)
    assert report["evenness"] == pytest.approx(1 / 14)
    assert report["n_interval_classes"] == 6
    assert report["n_sizes_per_class"] == [2] * 6
    assert [round(e, 2) for e in report["ji_errors"]] == [1.96, -13.69]
    assert report["harm_sim"] == pytest.approx(14.46, abs=1e-6)


def test_mos_report_is_json_serialisable():
    report = mos_report(
        MOSScale.from_generator(3 / 2, 12), targets=[3 / 2], harmonic=True
    )
    round_tripped = json.loads(json.dumps(report))
    assert round_tripped["signature"] == "5L7s"
    assert round_tripped["degree_signatures"][0] == list(
        degree_signatures(MOSScale.from_generator(3 / 2, 12))[0]
    )


def test_mos_report_can_skip_the_harmonic_metrics():
    report = mos_report(MOSScale.from_signature(5, 2, tuning=12), harmonic=False)
    assert "harm_sim" not in report
    assert "ji_errors" not in report
    assert report["myhill"] is True


def test_mos_report_rejects_a_raw_pair():
    with pytest.raises(TypeError) as info:
        mos_report((LYDIAN_12, PERIOD))
    assert "to_dict" in str(info.value)


@pytest.mark.parametrize("scale", CORPUS[::37], ids=lambda s: repr(s))
def test_mos_report_is_internally_consistent(scale):
    report = mos_report(scale, harmonic=False)
    assert report["blackwood_r"] == pytest.approx(report["hardness"], rel=1e-9)
    assert report["myhill"] is True
    assert report["unique_degree_signatures"] is True
    if scale.n_small >= 2:
        assert report["proper_from_matrix"] == report["is_proper"]


# --------------------------------------------------------------------------- #
# mos_ness
# --------------------------------------------------------------------------- #
# The three reference signals.  A well-formed scale, the equal division of the
# same size (which is that scale's own equalized landmark, i.e. the case where
# the generator has nothing left to do), and a genuinely three-step scale.
MEANTONE = MOSScale.from_signature(5, 2, tuning=31)
EDO_7 = [2.0 ** (k / 7) for k in range(7)]
TERNARY_7 = TernaryScale.from_barycentric("LMLsLMs", 0.52, 0.30, 0.18)


@pytest.fixture(scope="module")
def meantone():
    """The 31-EDO diatonic, all three rungs.  Roughly a second to fit."""
    return mos_ness(MEANTONE.ratios, cardinality=7)


@pytest.fixture(scope="module")
def edo7():
    return mos_ness(EDO_7, cardinality=7, min_cardinality=7)


@pytest.fixture(scope="module")
def ternary7():
    return mos_ness(TERNARY_7.ratios, cardinality=7)


# --- the pieces the comparison is built out of ---------------------------- #
def test_model_parameters_are_ordered_by_freedom():
    """Each rung adds exactly one parameter, and transposition is counted."""
    assert MODEL_PARAMETERS == {"edo": 1, "mos": 2, "ternary": 3}
    assert list(MODEL_PARAMETERS.values()) == sorted(MODEL_PARAMETERS.values())


@pytest.mark.parametrize("cardinality", [3, 5, 7, 12, 19])
def test_equal_division_really_is_the_equal_division(cardinality):
    """The null model's degrees are ``k / N``, exactly, and its steps are equal."""
    scale = _equal_division(cardinality, 2.0)
    assert scale.cardinality == cardinality
    np.testing.assert_allclose(
        scale.degrees, [k / cardinality for k in range(cardinality)], atol=1e-12
    )
    assert scale.is_degenerate is True
    assert blackwood_r(scale) == pytest.approx(1.0)
    assert evenness(scale) == pytest.approx(0.0, abs=1e-12)


def test_the_null_is_the_same_scale_a_degenerate_mos_would_give():
    """``_equal_division`` must be a relabelling, not a second implementation.

    7-EDO is reachable as the equalized tuning of *any* seven-note signature,
    and the choice of ``(N-1)L1s`` in ``_equal_division`` is only a label.  If
    the two disagreed, the null being scored would not be the equal division.
    """
    from_helper = _equal_division(7, 2.0)
    for n_large, n_small in ((5, 2), (4, 3), (6, 1)):
        equalized = MOSScale.from_signature(n_large, n_small, tuning="equalized")
        np.testing.assert_allclose(
            sorted(equalized.degrees), from_helper.degrees, atol=1e-12
        )


def test_adjusted_error_is_the_stated_degrees_of_freedom_rule():
    assert _adjusted_error(10.0, 20, "edo") == pytest.approx(10.0 * 20 / 19)
    assert _adjusted_error(10.0, 20, "mos") == pytest.approx(10.0 * 20 / 18)
    assert _adjusted_error(10.0, 20, "ternary") == pytest.approx(10.0 * 20 / 17)
    # Richer models are charged more for the same raw error.
    ladder = [_adjusted_error(10.0, 20, m) for m in ("edo", "mos", "ternary")]
    assert ladder == sorted(ladder)
    # No parameters left over means no measurement.
    assert _adjusted_error(10.0, 3, "ternary") == float("inf")
    assert _adjusted_error(10.0, 2, "mos") == float("inf")


def test_share_clamps_at_both_ends_and_at_a_zero_baseline():
    assert _share(5.0, 20.0) == pytest.approx(0.75)
    assert _share(25.0, 20.0) == 0.0          # worse than the baseline
    assert _share(0.0, 20.0) == 1.0
    # A baseline already at zero leaves nothing for the extra parameter to buy.
    assert _share(0.0, 0.0) == 0.0
    assert _share(0.0, 1e-12) == 0.0
    assert _share(1.0, float("inf")) == 0.0


def test_admissible_ternary_words_are_a_small_structured_family():
    """The MV3 filter is what keeps the third rung from being a free pass.

    A well-formed scale at N notes has exactly one step pattern.  Letting the
    ternary rung choose freely among all arrangements would hand it discrete
    freedom no parameter count reflects, so only the ternary-Myhill words are
    admitted -- a couple of dozen against thousands of rotation classes.
    """
    words = _ternary_words_at(7)
    assert len(words) == 24
    assert all(len(w) == 7 for w in words)
    assert all(set(w) == {"L", "M", "s"} for w in words)
    assert len(set(words)) == len(words)
    # Every admissible word really is MV3 when measured, not just when filtered.
    for word in words[:6]:
        scale = TernaryScale.from_barycentric(word, 0.44, 0.33, 0.23)
        assert scale.max_variety == 3


# --- the five claims ------------------------------------------------------ #
def test_a_real_mos_scores_high_and_its_own_equal_division_scores_zero(
    meantone, edo7
):
    """The headline discrimination, on the cleanest possible pair.

    31-EDO meantone and 7-EDO are the same signature at two tunings: the first
    somewhere inside 5L2s's valid range, the second exactly at its equalized
    landmark where the two step sizes coincide.  One needs a generator, the
    other is what you get when the generator stops mattering.
    """
    assert meantone.signature == "5L2s"
    assert meantone.mos_error_cents == pytest.approx(0.0, abs=1e-6)
    assert meantone.edo_error_cents == pytest.approx(18.96, abs=1e-2)
    assert meantone.mos_ness == pytest.approx(1.0, abs=1e-9)

    assert edo7.edo_error_cents == pytest.approx(0.0, abs=1e-9)
    assert edo7.mos_error_cents == pytest.approx(0.0, abs=1e-9)
    assert edo7.mos_ness == 0.0

    assert meantone.mos_ness - edo7.mos_ness > 0.9


@pytest.mark.parametrize("cardinality", [5, 7, 9, 12])
def test_an_equal_division_scores_zero_mos_ness(cardinality):
    """The generator buys nothing on maximally even input, at any size.

    Both families fit an equal division exactly, so there is no error for the
    extra parameter to remove and the answer is zero by definition rather than
    by luck -- which is why ``_share`` treats a zero baseline as zero
    improvement instead of dividing by it.
    """
    ratios = [2.0 ** (k / cardinality) for k in range(cardinality)]
    result = mos_ness(
        ratios, cardinality=cardinality, min_cardinality=cardinality, ternary=False
    )
    assert result.cardinality == cardinality
    assert result.edo_error_cents == pytest.approx(0.0, abs=1e-9)
    assert result.mos_error_cents == pytest.approx(0.0, abs=1e-9)
    assert result.mos_ness == 0.0
    assert result.raw_mos_ness == 0.0
    # An equal division has exactly N pitch classes, so this is the one case
    # that cannot be measured identifiably -- and it says so.
    assert result.is_identifiable is False


def test_random_uniform_ratios_score_near_zero():
    """The empirical null, re-measured rather than assumed.

    Zero is where "the generator bought nothing" sits by definition, but it is
    not where unstructured data sits: a free generator slides every degree at
    once, so it removes some error from anything, and a degrees-of-freedom
    correction that charges it as one parameter out of n cannot fully undo
    that.  Over sixty draws at these settings the measure averages 0.088 with a
    maximum of 0.330; this pins the seeded twelve well inside that band, and
    well below a genuine MOS.
    """
    rng = np.random.default_rng(20110726)
    values = [
        mos_ness(
            np.sort(2.0 ** rng.random(18)), ternary=False, max_cardinality=8
        ).mos_ness
        for _ in range(12)
    ]
    assert max(values) < 0.35
    assert float(np.mean(values)) < 0.20
    assert float(np.median(values)) < 0.20
    # The point of the measure: noise and a well-formed scale are far apart.
    assert mos_ness(MEANTONE.ratios, ternary=False).mos_ness - max(values) > 0.6


def test_a_three_step_scale_needs_its_third_step(ternary7, meantone):
    """Low two-step sufficiency is the ternary rung earning its place.

    ``LMLsLMs`` at 208 / 180 / 108 cents is a perturbation of the diatonic, so
    a MOS still explains most of it -- the interesting number is not that
    ``mos_ness`` is high but that the third step size finishes the job the
    generator could not.
    """
    assert ternary7.ternary_error_cents == pytest.approx(0.0, abs=1e-6)
    assert ternary7.mos_error_cents > 3.0
    assert ternary7.ternary_ness == pytest.approx(1.0, abs=1e-9)
    assert ternary7.two_step_sufficiency < 0.8
    assert ternary7.ternary_collapsed is False
    assert sorted(round(c, 3) for c in ternary7.ternary_step_cents) == [
        108.0, 180.0, 208.0
    ]

    # And the control: on a scale that really is well formed the same rung
    # finds nothing to add, by walking onto an interior line of the simplex
    # where two of the three step sizes coincide.
    assert meantone.two_step_sufficiency == pytest.approx(1.0)
    assert meantone.ternary_collapsed is True
    assert meantone.two_step_sufficiency > ternary7.two_step_sufficiency


#: Exact and jittered versions of the same scale.  The jittered one matters:
#: on an exact fit every quantity is 0 or 1 and an invariance test could pass
#: on a measure that ignored its input entirely.
_JITTER = np.random.default_rng(4).normal(0.0, 6.0, 7)
INVARIANCE_CASES = {
    "exact": list(MOSScale.from_signature(4, 3, tuning=19).ratios),
    "jittered": [
        r * 2.0 ** (c / 1200.0)
        for r, c in zip(MOSScale.from_signature(4, 3, tuning=19).ratios, _JITTER)
    ],
}


def _fold(ratios):
    return [r / 2.0 if r >= 2.0 else r for r in ratios]


@pytest.mark.parametrize("name", sorted(INVARIANCE_CASES))
def test_mos_ness_is_invariant_to_transposition(name):
    """A scale and its transpositions are the same scale.

    ``_evaluate`` searches rotations for every candidate, so shifting the whole
    signal must leave every reported number untouched -- not merely close.
    """
    ratios = INVARIANCE_CASES[name]
    plain = mos_ness(ratios, cardinality=7, ternary=False)
    for shift_cents in (137.0, 700.0, -311.0):
        moved = _fold([r * 2.0 ** (shift_cents / 1200.0) for r in ratios])
        moved = mos_ness(moved, cardinality=7, ternary=False)
        assert moved.mos_ness == pytest.approx(plain.mos_ness, abs=1e-9)
        assert moved.edo_error_cents == pytest.approx(
            plain.edo_error_cents, abs=1e-9
        )
        assert moved.mos_error_cents == pytest.approx(
            plain.mos_error_cents, abs=1e-9
        )
        assert moved.signature == plain.signature


@pytest.mark.parametrize("name", sorted(INVARIANCE_CASES))
def test_mos_ness_is_invariant_to_octave_folding(name):
    """A scale cannot tell 1/1 from 2/1, so neither may the measure.

    Every other ratio is moved up an octave -- *moved*, not duplicated, so the
    pitch-class count is unchanged and this tests the folding rather than the
    merging.
    """
    ratios = INVARIANCE_CASES[name]
    plain = mos_ness(ratios, cardinality=7, ternary=False)
    lifted = [r * 2.0 if i % 2 else r for i, r in enumerate(ratios)]
    lifted = mos_ness(lifted, cardinality=7, ternary=False)
    assert lifted.n_targets == plain.n_targets
    assert lifted.n_merged == plain.n_merged == 0
    assert lifted.mos_ness == pytest.approx(plain.mos_ness, abs=1e-9)
    assert lifted.mos_error_cents == pytest.approx(plain.mos_error_cents, abs=1e-9)
    assert lifted.signature == plain.signature


def test_octave_duplicates_are_merged_rather_than_double_counted():
    """The other half of folding: a repeated pitch class is one target."""
    ratios = list(MOSScale.from_signature(4, 3, tuning=19).ratios) + [2.0, 4.0]
    result = mos_ness(ratios, cardinality=7, ternary=False)
    assert result.n_targets == 7
    assert result.n_merged == 2


# --- the comparison is fair ------------------------------------------------ #
def test_the_mos_family_contains_the_null_at_every_cardinality(meantone):
    """Planting ``(N-1)/N`` in the candidate list makes this a nesting claim.

    An equal division is the degenerate tuning of any signature of the same
    size, so the best MOS can never be *worse* than it.  If the two searches
    were independent this would only hold by luck, and a negative raw
    improvement would occasionally show up as a clamp instead of as a number.
    """
    for row in meantone.by_cardinality:
        assert row["mos_error_cents"] <= row["edo_error_cents"] + 1e-9


def test_the_parameter_penalty_is_applied_and_visible(meantone):
    n = meantone.n_targets
    assert meantone.adjusted_edo_error_cents == pytest.approx(
        meantone.edo_error_cents * n / (n - MODEL_PARAMETERS["edo"])
    )
    assert meantone.adjusted_mos_error_cents == pytest.approx(
        meantone.mos_error_cents * n / (n - MODEL_PARAMETERS["mos"])
    )
    assert meantone.adjusted_ternary_error_cents == pytest.approx(
        meantone.ternary_error_cents * n / (n - MODEL_PARAMETERS["ternary"])
    )
    # The penalty can only lower the score, never raise it.
    assert meantone.mos_ness <= meantone.raw_mos_ness + 1e-12


def test_the_parameter_penalty_actually_bites():
    """A raw improvement can be entirely accounted for by the extra parameter.

    Meantone read at five notes: the generator drops the error from 39.8 to
    33.2 cents, a 17% raw improvement, which is exactly what one more parameter
    buys on seven targets.  The corrected score is zero, and reporting the raw
    17% as evidence of well-formedness at five notes would be wrong.
    """
    row = next(
        r for r in mos_ness(MEANTONE.ratios, ternary=False).by_cardinality
        if r["cardinality"] == 5
    )
    raw = 1.0 - row["mos_error_cents"] / row["edo_error_cents"]
    assert raw > 0.15
    assert row["mos_ness"] == pytest.approx(0.0, abs=1e-9)


def test_every_alternative_is_compared_at_one_cardinality(ternary7):
    """The whole point of fixing N: otherwise this measures the note count.

    A five-note equal division against a nine-note MOS would be a comparison of
    note counts wearing the clothes of a comparison of structures, so all three
    winners are checked back against the reported cardinality.
    """
    n = ternary7.cardinality
    assert n == 7
    assert _equal_division(n, ternary7.period).cardinality == n
    assert MOSScale.from_signature(
        *(int(part) for part in ternary7.signature.rstrip("s").split("L"))
    ).cardinality == n
    assert len(ternary7.ternary_word) == n
    assert len(ternary7.ternary_step_cents) == 3


# --- identifiability ------------------------------------------------------- #
def test_a_degree_per_observation_is_flagged_not_hidden(meantone):
    assert meantone.n_targets == 7
    assert meantone.cardinality == 7
    assert meantone.is_identifiable is False
    assert any("not below the 7 targets" in note for note in meantone.notes)
    assert "UNDERDETERMINED" in repr(meantone)


def test_more_targets_than_degrees_is_not_flagged():
    ratios = [2.0 ** (k / 12) for k in range(12)]
    result = mos_ness(ratios, cardinality=7, min_cardinality=7, ternary=False)
    assert result.is_identifiable is True
    assert result.notes == () or all(
        "not below" not in note for note in result.notes
    )


def test_a_cardinality_above_the_target_count_is_refused_by_default():
    ratios = [1.0, 1.2, 1.4, 1.6, 1.8]
    with pytest.raises(ValueError) as info:
        mos_ness(ratios, cardinality=9)
    assert "spare degrees" in str(info.value)
    # Explicitly asked for, it is allowed -- and flagged.
    loose = mos_ness(
        ratios, cardinality=9, ternary=False, allow_underdetermined=True
    )
    assert loose.cardinality == 9
    assert loose.is_identifiable is False
    # Nine degrees for five targets is spare degrees, not merely a tight fit,
    # so derive's own flag fires too and the note says which one is speaking.
    assert any("MOSFit.is_underdetermined" in note for note in loose.notes)


def test_too_few_pitch_classes_is_refused():
    with pytest.raises(ValueError) as info:
        mos_ness([1.0, 1.5, 1.75], ternary=False)
    assert "distinct" in str(info.value)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(min_cardinality=2), "at least 3"),
        (dict(cardinality="best"), "must be an int"),
        (dict(cardinality=3, min_cardinality=5), "below min_cardinality"),
    ],
)
def test_mos_ness_rejects_bad_arguments(kwargs, message):
    kwargs.setdefault("ternary", False)
    with pytest.raises(ValueError) as info:
        mos_ness(MEANTONE.ratios, **kwargs)
    assert message in str(info.value)


# --- the cardinality table ------------------------------------------------- #
def test_by_cardinality_covers_the_whole_searched_range(meantone):
    cards = [row["cardinality"] for row in meantone.by_cardinality]
    assert cards == list(range(4, 8))
    assert meantone.cardinality in cards
    chosen = next(
        r for r in meantone.by_cardinality if r["cardinality"] == meantone.cardinality
    )
    assert chosen["mos_error_cents"] == pytest.approx(meantone.mos_error_cents)
    assert chosen["edo_error_cents"] == pytest.approx(meantone.edo_error_cents)
    assert chosen["mos_ness"] == pytest.approx(meantone.mos_ness)
    assert chosen["signature"] == meantone.signature


@pytest.mark.parametrize(
    "spec, rule", [("edo", "edo-evidence"), ("mos", "mos-evidence"), ("max", "max")]
)
def test_the_cardinality_rule_is_recorded(spec, rule):
    result = mos_ness(MEANTONE.ratios, cardinality=spec, ternary=False)
    assert result.cardinality_rule == rule
    # Every rule finds the diatonic here; they differ on ambiguous input.
    assert result.cardinality == 7
    assert result.mos_ness == pytest.approx(1.0, abs=1e-9)


def test_an_explicit_cardinality_overrides_the_rules():
    result = mos_ness(MEANTONE.ratios, cardinality=5, ternary=False)
    assert result.cardinality == 5
    assert result.cardinality_rule == "explicit"


def test_choosing_by_mos_evidence_scores_at_least_as_high_as_by_edo():
    """The documented bias, made visible rather than argued about.

    Selecting the note count on the MOS fit's own evidence is a selection made
    on the hypothesis under test.  On random input it roughly doubles the null
    level, which is why ``'edo'`` is the default.
    """
    rng = np.random.default_rng(20110726)
    gaps = []
    for _ in range(6):
        ratios = np.sort(2.0 ** rng.random(18))
        common = dict(ternary=False, max_cardinality=8)
        by_mos = mos_ness(ratios, cardinality="mos", **common).mos_ness
        by_edo = mos_ness(ratios, cardinality="edo", **common).mos_ness
        gaps.append(by_mos - by_edo)
    assert float(np.mean(gaps)) > 0.0


# --- housekeeping ---------------------------------------------------------- #
def test_the_ternary_rung_is_skipped_loudly_when_it_would_be_slow():
    ratios = [2.0 ** (k / 13) * (1.0 + 0.001 * k) for k in range(13)]
    result = mos_ness(
        ratios, cardinality=11, min_cardinality=11, ternary_max_cardinality=10
    )
    assert result.ternary_error_cents is None
    assert result.two_step_sufficiency is None
    assert result.ternary_collapsed is None
    assert any("ternary_max_cardinality=10" in note for note in result.notes)


def test_mos_ness_honours_a_pseudo_octave():
    """A fifth stacked inside a tritave is well formed in the tritave.

    Every cents figure has to be measured against the period, so a hardcoded
    1200 anywhere in the chain would show up here as a broken null.
    """
    scale = MOSScale.from_generator(3 / 2, 8, period=3.0)
    result = mos_ness(scale.ratios, period=3.0, cardinality=8, ternary=False)
    assert result.period == 3.0
    assert result.signature == "3L5s"
    assert result.mos_error_cents == pytest.approx(0.0, abs=1e-6)
    assert result.mos_ness == pytest.approx(1.0, abs=1e-9)
    # The null is the tritave's eighth-tone division, not the octave's.
    assert result.edo_error_cents == pytest.approx(
        _reference_edo_error(scale.ratios, 8, 3.0), abs=1e-6
    )


def _reference_edo_error(ratios, cardinality, period):
    """Mean distance to the nearest ``N``-EDO degree, brute-forced.

    Deliberately written from scratch rather than through ``_evaluate``: this
    is the independent check that the null really is an equal division scored
    with full transposition freedom, and not something the fitting layer has
    quietly redefined.
    """
    period_cents = 1200.0 * math.log2(period)
    positions = np.mod(np.log(np.asarray(ratios)) / np.log(period), 1.0)
    best = float("inf")
    for offset in np.linspace(0.0, 1.0 / cardinality, 20001):
        d = np.abs((positions - offset) * cardinality)
        d = np.abs(d - np.round(d)) / cardinality
        best = min(best, float(d.mean()) * period_cents)
    return best


def test_mos_ness_weights_shift_the_answer():
    """Weights must reach the objective, not be silently dropped."""
    ratios = list(MOSScale.from_signature(4, 3, tuning=19).ratios)
    ratios[3] *= 2.0 ** (60.0 / 1200.0)     # drag one degree badly out of tune
    plain = mos_ness(ratios, cardinality=7, ternary=False)
    downweighted = mos_ness(
        ratios, weights=[1, 1, 1, 1e-6, 1, 1, 1], cardinality=7, ternary=False
    )
    assert downweighted.mos_error_cents < plain.mos_error_cents
    assert downweighted.mos_ness > plain.mos_ness


def test_mos_ness_result_is_json_serialisable(ternary7):
    payload = json.loads(json.dumps(ternary7.to_dict()))
    assert payload["cardinality"] == 7
    assert payload["n_parameters"] == {"edo": 1, "mos": 2, "ternary": 3}
    assert payload["ternary_collapsed"] is False
    assert len(payload["by_cardinality"]) == len(ternary7.by_cardinality)
    assert payload["mos_ness"] == pytest.approx(ternary7.mos_ness)


def test_mos_ness_summary_is_ascii_and_says_what_was_compared(meantone):
    text = meantone.summary()
    assert text.isascii()
    assert "MOS-ness" in text
    assert "equal div." in text and "well formed" in text and "three steps" in text
    assert "(1 parameter)" in text and "(2 parameters)" in text
    assert "not below the 7 targets" in text
    assert isinstance(MOSness.summary(meantone), str)
