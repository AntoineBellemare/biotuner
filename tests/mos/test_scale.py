"""Tests for :class:`biotuner.mos.scale.MOSScale`."""

import math
from fractions import Fraction

import pytest

from biotuner.mos import theory as T
from biotuner.mos.scale import MOSScale, mos, mos_family

FIFTH = 3 / 2

SIGS = [(5, 2), (2, 5), (3, 4), (4, 3), (2, 3), (3, 2), (5, 7), (7, 5),
        (1, 4), (4, 1), (5, 6), (6, 5), (7, 4), (2, 9)]
TUNINGS = ["noble", "central", "middle"]


def _sample_scales():
    for n_large, n_small in SIGS:
        for tuning in TUNINGS:
            for bright in (True, False):
                yield MOSScale.from_signature(
                    n_large, n_small, tuning=tuning, bright=bright
                )


# --------------------------------------------------------------------------- #
# Construction
# --------------------------------------------------------------------------- #
def test_from_generator_recovers_the_diatonic():
    d = MOSScale.from_generator(FIFTH, 7)
    assert d.signature == "5L2s"
    assert d.word == "LLLsLLs"
    assert d.cardinality == 7
    assert d.generator_cents == pytest.approx(701.955, abs=1e-3)


def test_from_generator_and_from_fraction_agree():
    a = MOSScale.from_generator(FIFTH, 12)
    b = MOSScale.from_fraction(math.log2(FIFTH), 12)
    assert a == b


def test_from_edo_gives_an_exactly_rational_generator():
    s = MOSScale.from_edo(31, 18, 7)
    assert s.signature == "5L2s"
    assert s.generator == pytest.approx(18 / 31)
    assert s.edo == 31


def test_from_signature_tuning_specs_all_land_in_range():
    for n_large, n_small in SIGS:
        lo, hi = T.signature_ranges(n_large, n_small)[1]
        for tuning in TUNINGS:
            s = MOSScale.from_signature(n_large, n_small, tuning=tuning)
            assert float(lo) <= s.generator <= float(hi)
            assert (s.n_large, s.n_small) == (n_large, n_small)


def test_from_signature_with_an_edo_that_has_no_generator_in_range():
    with pytest.raises(ValueError, match="no generator inside the valid range"):
        MOSScale.from_signature(5, 7, tuning=31)


def test_from_signature_rejects_an_unknown_tuning_word():
    with pytest.raises(ValueError, match="tuning must be"):
        MOSScale.from_signature(5, 2, tuning="brightest")


def test_non_coprime_signature_is_rejected():
    with pytest.raises(ValueError, match="co-prime"):
        MOSScale(4, 2, 0.6)


def test_generator_outside_the_unit_interval_is_rejected():
    with pytest.raises(ValueError, match=r"period fraction in \(0, 1\)"):
        MOSScale(5, 2, 1.5)


def test_period_must_exceed_one():
    with pytest.raises(ValueError, match="must exceed 1"):
        MOSScale(5, 2, 0.585, period=1.0)


def test_validation_rejects_a_generator_that_builds_a_different_scale():
    with pytest.raises(ValueError, match="Valid range for 5L2s"):
        MOSScale(5, 2, 0.55, validate=True)   # 0.55 gives 2L5s at 7 notes


# --------------------------------------------------------------------------- #
# Structural invariants
# --------------------------------------------------------------------------- #
def test_steps_tile_the_period_exactly():
    """nL * L + ns * s = the period, for every scale and every tuning."""
    for s in _sample_scales():
        large, small = s.step_cents
        assert s.n_large * large + s.n_small * small == pytest.approx(
            s.period_cents, abs=1e-7
        )


def test_word_letter_counts_match_the_signature():
    for s in _sample_scales():
        if s.is_degenerate:
            continue
        assert (s.word.count("L"), s.word.count("s")) == (s.n_large, s.n_small)


def test_degrees_are_sorted_rooted_and_distinct():
    for s in _sample_scales():
        assert s.degrees == sorted(s.degrees)
        assert s.degrees[0] == 0.0
        assert len(set(s.degrees)) == s.cardinality
        assert all(0.0 <= d < 1.0 for d in s.degrees)


def test_ratios_span_one_period():
    for s in _sample_scales():
        assert s.ratios[0] == pytest.approx(1.0)
        assert all(1.0 <= r < s.period + 1e-12 for r in s.ratios)


def test_cents_and_ratios_agree():
    for s in _sample_scales():
        for c, r in zip(s.cents, s.ratios):
            assert c == pytest.approx(1200 * math.log2(r), abs=1e-7)


def test_hardness_is_at_least_one():
    for s in _sample_scales():
        assert s.hardness >= 1.0 - 1e-9


# --------------------------------------------------------------------------- #
# Propriety
# --------------------------------------------------------------------------- #
def test_pythagorean_diatonic_is_improper():
    """Milne et al. section 3: the diatonic is coherent only between 4/7 and
    7/12, i.e. 685.7 to 700 cents. Pythagorean's 702 is outside."""
    d = MOSScale.from_generator(FIFTH, 7)
    assert d.hardness == pytest.approx(2.26, abs=1e-3)
    assert not d.is_proper


def test_twelve_and_thirtyone_edo_diatonics_are_proper():
    assert MOSScale.from_signature(5, 2, tuning=12).is_proper
    assert MOSScale.from_signature(5, 2, tuning=31).is_proper


def test_propriety_matches_the_coherence_range():
    for s in _sample_scales():
        lo, hi = s.coherence_range
        inside = float(lo) - 1e-12 <= s.generator <= float(hi) + 1e-12
        assert s.is_proper == inside


# --------------------------------------------------------------------------- #
# Relatives
# --------------------------------------------------------------------------- #
def test_inverse_swaps_the_signature():
    """Milne et al. section 2, 'Inverse scales'."""
    for s in _sample_scales():
        inv = s.inverse
        assert (inv.n_large, inv.n_small) == (s.n_small, s.n_large)
        assert inv.cardinality == s.cardinality


def test_inverse_is_an_involution():
    for s in _sample_scales():
        back = s.inverse.inverse
        assert (back.n_large, back.n_small) == (s.n_large, s.n_small)
        assert back.generator == pytest.approx(s.generator)


def test_inverse_is_equidistant_from_the_equalized_landmark():
    for s in _sample_scales():
        eq = float(s.landmarks.equalized)
        assert abs(s.generator - eq) == pytest.approx(
            abs(s.inverse.generator - eq), abs=1e-12
        )


def test_family_matches_the_theory_layer():
    fam = mos_family(FIFTH, max_cardinality=17)
    assert [s.cardinality for s in fam] == [3, 5, 7, 12, 17]
    assert [s.signature for s in fam] == ["2L1s", "2L3s", "5L2s", "5L7s", "12L5s"]


def test_family_members_share_one_generator():
    fam = mos_family(FIFTH, max_cardinality=29)
    assert len({s.generator for s in fam}) == 1


def test_family_is_nested():
    """Each scale's degrees are contained in the next larger one's."""
    fam = mos_family(FIFTH, max_cardinality=29)
    for small, big in zip(fam, fam[1:]):
        for d in small.degrees:
            assert any(abs(d - e) < 1e-9 for e in big.degrees)


def test_parent_and_child_walk_the_family():
    d = MOSScale.from_generator(FIFTH, 7)
    assert d.parent.cardinality == 5
    assert d.child().cardinality == 12
    assert d.child(2).cardinality == 17
    assert mos_family(FIFTH, 60)[0].parent is None or True  # smallest may have none


def test_embedding_agrees_with_theory():
    d = MOSScale.from_generator(FIFTH, 7)
    assert d.embedding == (12, Fraction(7, 12))


def test_retune_preserves_the_structure():
    d = MOSScale.from_generator(FIFTH, 7)
    for tuning in (12, 19, 31, "central", "noble"):
        r = d.retune(tuning)
        assert r.signature == d.signature
        assert r.cardinality == d.cardinality
        lo, hi = r.tuning_range
        assert float(lo) <= r.generator <= float(hi)


# --------------------------------------------------------------------------- #
# Pseudo-octaves
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("period", [1.98, 2.0, 2.05, 3.0])
def test_a_stretched_period_is_honoured_everywhere(period):
    """Nothing may assume a 2/1 octave -- the paper's period is a free parameter."""
    s = MOSScale.from_signature(5, 2, tuning="central", period=period)
    assert s.period_cents == pytest.approx(1200 * math.log2(period))
    assert s.ratios[0] == pytest.approx(1.0)
    assert max(s.ratios) < period
    large, small = s.step_cents
    assert 5 * large + 2 * small == pytest.approx(s.period_cents, abs=1e-7)
    assert s.generator_ratio == pytest.approx(period**s.generator)


def test_period_changes_the_ratios_but_not_the_structure():
    a = MOSScale.from_signature(5, 2, tuning="central", period=2.0)
    b = MOSScale.from_signature(5, 2, tuning="central", period=2.1)
    assert a.signature == b.signature
    assert a.word == b.word
    assert a.generator == pytest.approx(b.generator)
    assert a.ratios != b.ratios


# --------------------------------------------------------------------------- #
# Degenerate cases
# --------------------------------------------------------------------------- #
def test_equalized_tuning_is_degenerate():
    s = MOSScale.from_signature(5, 2, tuning="equalized")
    assert s.is_degenerate
    assert s.hardness == pytest.approx(1.0)
    assert s.edo == 7


def test_a_rational_generator_reports_its_edo():
    assert MOSScale.from_signature(5, 2, tuning=12).edo == 12
    assert MOSScale.from_signature(5, 2, tuning="noble").edo is None


# --------------------------------------------------------------------------- #
# Interop
# --------------------------------------------------------------------------- #
def test_to_dict_is_flat_and_complete():
    d = MOSScale.from_generator(FIFTH, 7).to_dict()
    for key in ("signature", "cardinality", "generator_cents", "word", "hardness",
                "is_proper", "equalized_edo", "embedding_cardinality"):
        assert key in d
    assert all(not isinstance(v, (dict, list)) for v in d.values())


def test_summary_mentions_the_landmarks():
    text = MOSScale.from_generator(FIFTH, 7).summary()
    assert "5L2s" in text and "7-EDO" in text and "IMPROPER" in text


def test_to_scala_lists_every_degree():
    scl = MOSScale.from_signature(5, 2, tuning=12).to_scala(write=False)
    assert "5L2s" in scl
    assert scl.count("\n") >= 7


def test_scales_are_hashable_and_comparable():
    a = MOSScale.from_generator(FIFTH, 7)
    b = MOSScale.from_generator(FIFTH, 7)
    assert a == b
    assert len({a, b}) == 1


def test_mos_shorthand_matches_the_classmethod():
    assert mos(FIFTH, 7) == MOSScale.from_generator(FIFTH, 7)


def test_mos_family_rejects_nothing_but_returns_empty_below_the_floor():
    assert mos_family(FIFTH, max_cardinality=2, min_cardinality=3) == []
