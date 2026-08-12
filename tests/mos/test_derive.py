"""Tests for :mod:`biotuner.mos.derive` -- fitting well-formed scales to data.

The demanding tests here are *recovery* tests: build a scale whose answer is
known, hand the fitter only its ratios (sometimes jittered, sometimes a
subset), and check the original comes back.  A fit routine that cannot recover
a scale it was literally given has nothing to say about a biosignal.
"""

import itertools
import math
import random

import numpy as np
import pytest

from biotuner.mos import derive
from biotuner.mos import theory as T
from biotuner.mos.derive import (
    GENERATOR_EPSILON,
    MIN_REFINED_STEP,
    ForwardScale,
    MOSFit,
    _offset_candidates,
    _refine_generator,
    _refinement_bounds,
    best_mos,
    explain_fit,
    fit_mos,
    forward_scales,
    generator_candidates,
    labyrinth_positions,
    mos_from_biotuner,
    mos_tuning,
    trajectory_dataframe,
    trajectory_from_windows,
)
from biotuner.mos.scale import MOSScale

FIFTH = math.log2(3 / 2)

RECOVERABLE = [(5, 2), (2, 5), (4, 3), (3, 4), (2, 3), (3, 2), (5, 6), (6, 5),
               (7, 4), (4, 7)]


# --------------------------------------------------------------------------- #
# Candidates
# --------------------------------------------------------------------------- #
def test_candidates_include_an_observed_generator_exactly():
    stack = [1.0, 1.125, 1.265625, 1.5]  # a stack of fifths, octave-reduced
    cands = generator_candidates(stack, grid=0)
    assert any(abs(c - FIFTH) < 1e-12 for c in cands)


def test_a_dense_grid_does_not_shadow_the_exact_generator():
    """Priority-aware de-duplication: signal beats grid."""
    stack = [1.0, 1.125, 1.265625, 1.5]
    cands = generator_candidates(stack, grid=1200)
    assert any(abs(c - FIFTH) < 1e-12 for c in cands)


#: Seven peaks of S004 eyes-closed, in Hz.  Two of the generators they state
#: sit 0.909 cents apart, which is inside the one-cent de-duplication window --
#: the input that used to have a real candidate thinned away.
S004_PEAKS = [10.71, 10.47, 21.17, 13.12, 8.07, 17.18, 25.55]


def test_two_signal_candidates_under_a_cent_apart_both_survive():
    """Thinning the signal against itself throws away information nothing at
    the candidate stage can replace.

    These peaks state a generator at 810.302 cents and another at 809.392, less
    than a cent apart.  Collapsing them keeps whichever comes first in sorted
    order, which is the lower one -- and it fits the peaks measurably worse.
    Which of two proposals is better is a question about the fit, and
    :func:`generator_candidates` scores nothing, so the only honest answer is
    to keep both and let the search decide.
    """
    cents = [c * 1200 for c in generator_candidates(S004_PEAKS, grid=0)]
    assert any(abs(c - 810.30166) < 1e-3 for c in cents)
    assert any(abs(c - 809.39247) < 1e-3 for c in cents)

    # ... and the one that used to be dropped is the better of the two.
    kept = fit_mos(S004_PEAKS, candidates=[809.39247 / 1200], refine=False,
                   top_n=1)[0]
    dropped = fit_mos(S004_PEAKS, candidates=[810.30166 / 1200], refine=False,
                      top_n=1)[0]
    assert dropped.score < kept.score


def test_a_dense_grid_is_still_thinned_against_the_signal():
    """The priority-aware half of the rule is the half worth keeping: a grid
    point may not crowd a candidate the signal actually stated."""
    signal = generator_candidates(S004_PEAKS, grid=0)
    everything = generator_candidates(S004_PEAKS, grid=720)
    from_grid = [c for c in everything
                 if all(abs(c - s) > 1e-12 for s in signal)]
    assert from_grid, "the grid contributed nothing to thin"
    tol = 1.0 / 1200
    for g in from_grid:
        assert min(abs(g - s) for s in signal) > tol - 1e-12


def test_duplicate_proposals_of_one_generator_are_still_collapsed():
    """A stack of fifths states the fifth three times over -- as 3/2, as
    3/2 over 9/8, and again over 81/64.  That is one candidate, not three."""
    stack = [1.0, 1.125, 1.265625, 1.5]
    cands = generator_candidates(stack, grid=0)
    assert len(cands) == len(set(cands))
    assert sum(abs(c - FIFTH) < 1e-9 for c in cands) == 1


def test_candidates_are_all_bright_sorted_and_thinned():
    cands = generator_candidates([1, 1.2, 1.4, 1.6], grid=360, dedupe_cents=1.0)
    assert cands == sorted(cands)
    assert all(0.5 < c < 1.0 for c in cands)
    assert all(b - a > 1.0 / 1200 - 1e-12 for a, b in zip(cands, cands[1:]))


def test_interval_candidates_can_be_switched_off():
    with_i = generator_candidates([1, 1.2, 1.5], grid=0, include_intervals=True)
    without = generator_candidates([1, 1.2, 1.5], grid=0, include_intervals=False)
    assert len(with_i) >= len(without)


def test_candidates_reject_a_bad_period():
    with pytest.raises(ValueError, match="must exceed 1"):
        generator_candidates([1, 1.5], period=1.0)


def test_labyrinth_positions_are_period_fractions():
    assert labyrinth_positions([1.0, 1.5, 2.0]) == pytest.approx(
        [0.0, FIFTH, 0.0], abs=1e-9
    )
    assert labyrinth_positions([1.5], period=3.0) == pytest.approx(
        [math.log(1.5) / math.log(3.0)], abs=1e-9
    )


# --------------------------------------------------------------------------- #
# Exact recovery
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_large,n_small", RECOVERABLE)
def test_a_scale_is_recovered_from_its_own_ratios(n_large, n_small):
    ref = MOSScale.from_signature(n_large, n_small, tuning="central")
    fit = fit_mos(ref.ratios, max_cardinality=max(14, ref.cardinality + 2))[0]
    assert fit.signature == ref.signature
    assert fit.error_cents < 1e-4
    assert fit.scale.generator == pytest.approx(ref.generator, abs=1e-6)


def test_recovered_degrees_match_the_original():
    ref = MOSScale.from_signature(4, 3, tuning=19)
    fit = best_mos(ref.ratios, max_cardinality=12)
    assert fit.scale.cents == pytest.approx(ref.cents, abs=1e-4)


def test_exact_fit_reports_infinite_improvement():
    ref = MOSScale.from_signature(5, 2, tuning=31)
    fit = best_mos(ref.ratios, max_cardinality=12)
    assert fit.error_cents == pytest.approx(0.0, abs=1e-6)
    assert math.isinf(fit.improvement)


def test_ranking_is_deterministic_on_a_degenerate_input():
    """12-EDO is a degenerate MOS; several signatures fit it perfectly, so the
    tie-break -- not float noise -- must decide."""
    edo12 = [2 ** (k / 12) for k in range(12)]
    picks = {fit_mos(edo12, max_cardinality=12)[0].signature for _ in range(3)}
    assert len(picks) == 1
    assert fit_mos(edo12, max_cardinality=12)[0].scale.is_degenerate


# --------------------------------------------------------------------------- #
# Recovery under noise
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("jitter_cents", [0.0, 3.0, 8.0])
def test_recovery_survives_peak_jitter(jitter_cents):
    rng = np.random.default_rng(11)
    hits = 0
    for n_large, n_small in RECOVERABLE:
        ref = MOSScale.from_signature(n_large, n_small, tuning="central")
        noisy = [
            r * 2 ** (rng.normal(0.0, jitter_cents) / 1200.0) for r in ref.ratios
        ]
        fit = fit_mos(noisy, max_cardinality=max(14, ref.cardinality + 2))[0]
        hits += fit.signature == ref.signature
    # Exact input must be perfect; jittered input mostly right.
    threshold = len(RECOVERABLE) if jitter_cents == 0 else int(0.7 * len(RECOVERABLE))
    assert hits >= threshold, f"{hits}/{len(RECOVERABLE)} at {jitter_cents} c"


def test_a_no_penalty_search_overfits():
    """Without a parsimony penalty the search takes the biggest scale allowed --
    which is why complexity_penalty defaults above zero."""
    rng = np.random.default_rng(3)
    arbitrary = sorted(2 ** rng.uniform(0, 1, 5))
    greedy = fit_mos(arbitrary, complexity_penalty=0.0, max_cardinality=24)[0]
    thrifty = fit_mos(arbitrary, complexity_penalty=5.0, max_cardinality=24)[0]
    assert greedy.scale.cardinality > thrifty.scale.cardinality


# --------------------------------------------------------------------------- #
# Rotation invariance
# --------------------------------------------------------------------------- #
def test_a_stack_of_fifths_reads_as_the_pentatonic():
    """Five consecutive fifths are the pentatonic -- but only in one of its modes.

    F-C-G-D-A as pitch classes against C is 0, 204, 498, 702, 906 cents. The
    2L3s scale rooted on its own generator-chain origin is 0, 204, 408, 702,
    906, which is a *different* mode; pinning every candidate to a 1/1 root
    therefore misses the answer entirely.
    """
    fifths = [2 ** ((k * 701.955 % 1200) / 1200) for k in range(-1, 4)]
    rotated = best_mos(fifths, max_cardinality=12)
    pinned = best_mos(fifths, max_cardinality=12, align=False)
    assert rotated.signature == "2L3s"
    assert rotated.error_cents < 0.01
    assert pinned.error_cents > rotated.error_cents


def test_alignment_never_scores_worse_than_no_alignment():
    """Offset zero is always among the candidates, so aligning cannot hurt."""
    rng = np.random.default_rng(5)
    for _ in range(6):
        ratios = sorted(2 ** rng.uniform(0, 1, 6))
        a = best_mos(ratios, max_cardinality=14, align=True)
        b = best_mos(ratios, max_cardinality=14, align=False)
        assert a.score <= b.score + 1e-9


def test_a_transposed_scale_is_recognised_in_every_mode():
    ref = MOSScale.from_signature(5, 2, tuning=31)
    for index in range(ref.cardinality):
        mode = ref.mode(index)
        fit = best_mos(mode.ratios, max_cardinality=12)
        assert fit.signature == "5L2s", f"mode {index} ({mode.name})"
        assert fit.error_cents < 1e-4


def test_aligned_ratios_start_at_unison_and_match_a_mode():
    ref = MOSScale.from_signature(5, 2, tuning=31)
    fit = best_mos(ref.mode(3).ratios, max_cardinality=12)
    assert fit.aligned_ratios[0] == pytest.approx(1.0)
    assert len(fit.aligned_ratios) == fit.scale.cardinality
    assert fit.mode is not None
    assert fit.aligned_cents == pytest.approx(fit.mode.cents, abs=1e-6)


def test_aligned_ratios_reproduce_the_targets():
    ref = MOSScale.from_signature(4, 3, tuning=19)
    mode = ref.mode(2)
    fit = best_mos(mode.ratios, max_cardinality=12)
    assert sorted(fit.aligned_ratios) == pytest.approx(sorted(mode.ratios), abs=1e-6)


def test_offset_is_reported_and_within_the_period():
    ref = MOSScale.from_signature(5, 2, tuning=31)
    fit = best_mos(ref.mode(4).ratios, max_cardinality=12)
    assert 0.0 <= fit.offset < 1.0
    assert "offset" in fit.to_dict()


def test_anchor_count_trades_speed_for_exactness_not_correctness():
    """Few anchors is an approximation in the coarse scan only; the shortlist is
    always re-scored with every target, so the answer should not move."""
    ref = MOSScale.from_signature(5, 2, tuning=31)
    ratios = list(ref.mode(5).ratios)
    for n_anchors in (1, 2, 3, 7):
        fit = best_mos(ratios, max_cardinality=12, n_anchors=n_anchors)
        assert fit.signature == "5L2s"


def test_mos_tuning_returns_the_aligned_scale():
    ref = MOSScale.from_signature(5, 2, tuning=31)
    mode = ref.mode(3)
    tuning = mos_tuning(mode.ratios, max_cardinality=12)
    assert tuning[0] == pytest.approx(1.0)
    assert sorted(tuning) == pytest.approx(sorted(mode.ratios), abs=1e-6)


# --------------------------------------------------------------------------- #
# Weights
# --------------------------------------------------------------------------- #
def test_weights_pull_the_fit_toward_the_loud_peaks():
    ref = MOSScale.from_signature(5, 2, tuning=12)
    ratios = list(ref.ratios)
    ratios[3] *= 2 ** (40 / 1200)          # one badly mistuned peak
    heavy = [1.0] * 7
    heavy[3] = 50.0                        # ... which is also the loudest
    light = [1.0] * 7
    light[3] = 0.001                       # ... versus barely there
    fit_heavy = best_mos(ratios, weights=heavy, max_cardinality=12)
    fit_light = best_mos(ratios, weights=light, max_cardinality=12)
    assert fit_heavy.residuals[3] < fit_light.residuals[3] or (
        abs(fit_heavy.residuals[3]) < abs(fit_light.residuals[3])
    )


def test_mismatched_weight_length_is_rejected():
    with pytest.raises(ValueError, match="one entry per ratio"):
        fit_mos([1.0, 1.5, 1.75], weights=[1.0, 2.0])


def test_a_dropped_ratio_takes_its_weight_with_it():
    """Weights are validated against the input, then masked alongside it.

    Filtering the ratios without filtering the weights would shift every
    remaining weight onto the wrong target.
    """
    fit = fit_mos([1.0, 1.5, float("nan")], weights=[1.0, 2.0, 5.0],
                  max_cardinality=8)
    assert fit[0].n_targets == 2


def test_degenerate_weights_fall_back_to_uniform():
    ref = MOSScale.from_signature(5, 2, tuning=12)
    fit = best_mos(ref.ratios, weights=[0.0] * 7, max_cardinality=12)
    assert fit.signature == "5L2s"


# --------------------------------------------------------------------------- #
# Folding: octave-equivalent ratios are one pitch class
# --------------------------------------------------------------------------- #
LADDER = [1.0, 1.125, 1.25, 1.5, 2.0]   # a tuning stated from unison to octave


def test_the_unison_and_the_octave_are_one_target():
    fit = fit_mos(LADDER, max_cardinality=12)[0]
    assert fit.n_targets == 4
    assert fit.n_merged == 1
    assert len(fit.residuals) == 4
    assert len(fit.targets) == 4
    assert all(1.0 <= t < 2.0 for t in fit.targets)


def test_fold_off_keeps_every_ratio_as_a_target():
    fit = fit_mos(LADDER, fold=False, max_cardinality=12)[0]
    assert fit.n_targets == 5
    assert fit.n_merged == 0


def test_folding_pays_the_parsimony_penalty_it_owes():
    """The error is unchanged -- summing the merged weights preserves the
    weighted mean exactly -- but the scale now has a spare degree to answer
    for, and the double-counted unison no longer hides it."""
    folded = fit_mos(LADDER, max_cardinality=12)[0]
    raw = fit_mos(LADDER, fold=False, max_cardinality=12)[0]
    assert folded.signature == raw.signature
    assert folded.error_cents == pytest.approx(raw.error_cents, abs=1e-9)
    assert folded.score > raw.score
    assert folded.is_underdetermined and not raw.is_underdetermined


def test_merged_ratios_keep_their_vote():
    """Two peaks an octave apart both vote for the pitch class, so folding
    adds their weights rather than dropping one."""
    merged = fit_mos([1.0, 1.25, 1.5, 2.0], weights=[1.0, 1.0, 1.0, 3.0],
                     max_cardinality=12)[0]
    by_hand = fit_mos([1.0, 1.25, 1.5], weights=[4.0, 1.0, 1.0],
                      max_cardinality=12)[0]
    assert merged.signature == by_hand.signature
    assert merged.error_cents == pytest.approx(by_hand.error_cents, abs=1e-9)


def test_near_duplicates_within_the_tolerance_also_merge():
    """Two peak pairs giving 1.1250 and 1.1249 are one pitch class measured
    twice, 0.15 cents apart."""
    fit = fit_mos([1.0, 1.1250, 1.1249, 1.5], max_cardinality=12)[0]
    assert fit.n_targets == 3
    assert fit.n_merged == 1


def test_a_pseudo_octave_folds_at_its_own_period():
    fit = fit_mos([1.0, 1.3, 1.6, 2.05], period=2.05, max_cardinality=10)[0]
    assert fit.n_targets == 3
    assert fit.n_merged == 1


def test_the_octave_duplicate_adds_no_target():
    """Stating the octave is not stating a fifth target.

    The ladder and the same ladder with its closing 2/1 removed name the same
    four pitch classes, and folding must make the fitter see exactly those.
    """
    with_octave = fit_mos(LADDER, max_cardinality=12)[0]
    without = fit_mos(LADDER[:-1], max_cardinality=12)[0]
    assert with_octave.n_targets == without.n_targets == 4
    assert with_octave.targets == pytest.approx(without.targets, abs=1e-12)


def test_the_octave_duplicate_is_a_second_vote_though():
    """It adds no target, but it is not free either -- and the difference is
    the whole point of summing merged weights.

    Hearing a pitch class twice is two pieces of evidence for it, so the
    unison's share of the weight goes from 1/4 to 2/4 and the fit moves onto
    it: measured, the unison's residual drops from 7.17 c to 0.00 c while
    1.125 -- heard once -- absorbs the slack and goes from 7.17 c to 10.75 c.
    The two fits are therefore *not* interchangeable, and a test asserting
    they are would be asserting that merged weights get dropped.
    """
    with_octave = fit_mos(LADDER, max_cardinality=12)[0]
    without = fit_mos(LADDER[:-1], max_cardinality=12)[0]
    doubled = fit_mos(LADDER[:-1], weights=[2.0, 1.0, 1.0, 1.0],
                      max_cardinality=12)[0]
    # Summed, not dropped: the folded ladder *is* the doubled-unison fit.
    assert with_octave.signature == doubled.signature
    assert with_octave.scale.generator == pytest.approx(doubled.scale.generator,
                                                        abs=1e-12)
    assert with_octave.error_cents == pytest.approx(doubled.error_cents, abs=1e-9)
    # ... and that is a different answer from one vote per class.
    assert with_octave.error_cents != pytest.approx(without.error_cents, abs=1e-6)
    assert abs(with_octave.residuals[0]) < abs(without.residuals[0])


def test_an_octave_pair_outvotes_a_single_peak():
    """A class reached by two peaks an octave apart pulls harder than a class
    reached by one.

    Nothing here is an MOS, so the fit cannot satisfy everyone: 1.42 is heard
    at 1.42 and 2.84, and the fit lands on it exactly.

    What the *other* three targets then pay is not a testable quantity, and an
    earlier version of this test asserted it was.  On these ratios the weighted
    absolute error has a flat optimum: ``once`` reaches 2.6046729299402105
    cents at a whole family of (generator, offset) pairs, which spread that
    same total over the four residuals in visibly different ways -- ``[0.0,
    -4.37, -3.98, 2.07]`` at one point of the plateau and ``[3.26, -0.75, 0.0,
    6.41]`` at another, bit-identical in the total.  Which point the optimiser
    stops at is a property of its search path, so asserting that 1.70's
    residual grows was asserting a coincidence.  The error, the coverage and
    the twice-heard class landing on a degree are what the data determines.
    """
    ratios = [1.0, 1.19, 1.42, 1.70]
    once = fit_mos(ratios, max_cardinality=9, top_n=1)[0]
    twice = fit_mos(ratios + [2.84], max_cardinality=9, top_n=1)[0]
    # Same targets on both sides -- only the weighting differs.
    assert once.n_targets == twice.n_targets == 4
    assert twice.n_merged == 1
    assert twice.targets == pytest.approx(once.targets, abs=1e-12)
    # The twice-heard class is where the extra weight went, and the fit puts a
    # degree on it exactly.
    assert twice.residuals[2] == pytest.approx(0.0, abs=1e-6)
    # Doubling one target's weight is a different measurement, not a rescaling
    # of the same one: it buys a lower weighted error than one vote per class.
    assert twice.error_cents < once.error_cents
    # Exactly as if its weight had been handed in already summed.
    by_hand = fit_mos(ratios, weights=[1.0, 1.0, 2.0, 1.0],
                      max_cardinality=9, top_n=1)[0]
    assert twice.error_cents == pytest.approx(by_hand.error_cents, abs=1e-12)
    assert twice.residuals == pytest.approx(by_hand.residuals, abs=1e-9)


def test_two_ratios_an_octave_apart_are_one_pitch_class():
    fit = fit_mos([1.5, 3.0], max_cardinality=8)[0]
    assert fit.n_targets == 1
    assert fit.n_merged == 1
    assert fit.targets == pytest.approx((1.5,))
    assert fit.is_underdetermined          # one pitch class, four degrees
    assert fit_mos([1.5, 3.0], fold=False, max_cardinality=8)[0].n_targets == 2


def test_fold_off_reproduces_the_unfolded_error_exactly():
    """The pre-folding behaviour, pinned so nobody can change it by accident.

    ``fold=False`` counts 1/1 and 2/1 as two targets; a tuning that simply
    omits the unison is a different, three-target problem with a different
    answer.  Both numbers are measured, not aspirational.
    """
    raw = fit_mos(LADDER, fold=False, max_cardinality=12)[0]
    assert raw.n_targets == 5
    assert raw.error_cents == pytest.approx(3.2259, abs=1e-3)
    three = fit_mos([1.125, 1.25, 1.5], fold=False, max_cardinality=12)[0]
    assert three.n_targets == 3
    assert three.error_cents == pytest.approx(2.3896, abs=1e-3)


def test_folding_uses_the_period_it_was_given_not_the_octave():
    """Under a tritave the octave is an ordinary interval, and 3/1 is the
    duplicate.  Folding everything at 2/1 regardless would merge the wrong
    pair and lose a real target."""
    tritave = fit_mos([1.0, 1.5, 2.0, 3.0], period=3.0, max_cardinality=10)[0]
    assert tritave.n_targets == 3
    assert tritave.n_merged == 1
    # 2/1 is kept -- under a tritave it is not a duplicate of anything.
    assert any(t == pytest.approx(2.0) for t in tritave.targets)
    octave = fit_mos([1.0, 1.5, 2.0, 3.0], period=2.0, max_cardinality=10)[0]
    assert octave.n_targets == 2                   # 2/1 -> 1/1 and 3/1 -> 3/2
    assert octave.n_merged == 2


# --------------------------------------------------------------------------- #
# Underdetermined fits
# --------------------------------------------------------------------------- #
def test_one_ratio_fits_a_four_note_scale_perfectly_and_says_so():
    """A scale with spare degrees can be rotated onto anything, so its error
    is not a measurement.  The fit is returned, but flagged."""
    fit = best_mos([1.5])
    assert fit.error_cents < 1e-6
    assert fit.is_underdetermined
    assert fit.n_targets < fit.scale.cardinality
    assert "UNDERDETERMINED" in explain_fit(fit)


def test_a_scale_fitted_to_its_own_ratios_is_determined():
    ref = MOSScale.from_signature(5, 2, tuning=31)
    fit = best_mos(ref.ratios, max_cardinality=12)
    assert not fit.is_underdetermined
    assert "UNDERDETERMINED" not in explain_fit(fit)


def test_evidence_prefers_the_fit_that_had_more_to_explain():
    """Both fit exactly; only one of them is evidence of anything."""
    two = best_mos([1.0, 1.5], max_cardinality=8)
    seven = best_mos(MOSScale.from_signature(5, 2, tuning=31).ratios,
                     max_cardinality=12)
    assert two.error_cents == pytest.approx(seven.error_cents, abs=1e-4)
    assert two.evidence < seven.evidence
    assert two.evidence == pytest.approx(math.sqrt(3 * 2), abs=1e-3)


def test_explain_fit_reports_the_chance_level_error():
    fit = best_mos(MOSScale.from_signature(4, 3, tuning=19).ratios,
                   max_cardinality=12)
    text = explain_fit(fit)
    assert "chance" in text
    assert f"{fit.chance_error_cents:.3f}" in text


def test_explain_fit_lists_the_targets_that_were_actually_fitted():
    """The caller's five ratios became four pitch classes, so five rows would
    silently mislabel the residuals."""
    fit = fit_mos(LADDER, max_cardinality=12)[0]
    assert explain_fit(fit, LADDER).count("-> degree") == 4


def test_the_flag_turns_over_exactly_at_one_target_per_degree():
    """Seven targets for seven degrees is determined; take one target away and
    the same scale is not.  The boundary is where the flag has to be, so it is
    tested against a scale held fixed on both sides rather than against two
    different winners."""
    ref = MOSScale.from_signature(5, 2, tuning=31)
    seven = fit_mos(ref.ratios, min_cardinality=7, max_cardinality=7, top_n=1)[0]
    six = fit_mos(list(ref.ratios)[:6], min_cardinality=7, max_cardinality=7,
                  top_n=1)[0]
    assert seven.n_targets == seven.scale.cardinality == 7
    assert not seven.is_underdetermined
    assert "UNDERDETERMINED" not in explain_fit(seven)
    assert six.n_targets == 6 and six.scale.cardinality == 7
    assert six.is_underdetermined
    assert "6 targets for 7 degrees" in explain_fit(six)


def test_chance_error_is_a_quarter_step_of_the_fitted_period():
    """A quarter of a step of *this* period, not of a hardcoded octave.  A fit
    against a stretched period has more room, so chance is more forgiving, and
    reporting the octave figure would flatter every pseudo-octave fit."""
    ref = MOSScale.from_signature(5, 2, tuning="central", period=2.08)
    fit = best_mos(ref.ratios, period=2.08, max_cardinality=10)
    expected = fit.scale.period_cents / (4 * fit.scale.cardinality)
    assert fit.chance_error_cents == pytest.approx(expected)
    assert fit.chance_error_cents != pytest.approx(1200.0 / (4 * 7), abs=1e-3)
    text = explain_fit(fit)
    assert f"chance         {expected:.3f} c for 7 degrees" in text


def test_to_dict_carries_the_new_fields():
    fit = best_mos([1.5])
    d = fit.to_dict()
    assert d["n_merged"] == 0
    assert d["is_underdetermined"] is True
    assert "evidence" in d


# --------------------------------------------------------------------------- #
# Pseudo-octaves
# --------------------------------------------------------------------------- #
def test_a_stretched_period_is_recovered_when_it_is_fitted():
    ref = MOSScale.from_signature(5, 2, tuning="central", period=2.08)
    fits = fit_mos(
        ref.ratios, max_cardinality=10, optimize_period=True,
        period_bounds=(2.0, 2.16), period_steps=17,
    )
    best = fits[0]
    assert best.scale.period == pytest.approx(2.08, abs=0.011)
    assert best.error_cents < 1.0


def test_a_fixed_wrong_period_fits_worse_than_the_right_one():
    ref = MOSScale.from_signature(5, 2, tuning="central", period=2.08)
    right = best_mos(ref.ratios, period=2.08, max_cardinality=10)
    wrong = best_mos(ref.ratios, period=2.0, max_cardinality=10)
    assert right.error_cents < wrong.error_cents


def test_bad_period_bounds_are_rejected():
    with pytest.raises(ValueError, match="period_bounds"):
        fit_mos([1, 1.5], optimize_period=True, period_bounds=(2.2, 1.9))


# --------------------------------------------------------------------------- #
# Validation and edge cases
# --------------------------------------------------------------------------- #
def test_cardinality_bounds_are_checked():
    with pytest.raises(ValueError, match="min_cardinality must be at least 3"):
        fit_mos([1, 1.5], min_cardinality=2)
    with pytest.raises(ValueError, match="below min_cardinality"):
        fit_mos([1, 1.5], min_cardinality=10, max_cardinality=5)


def test_unusable_ratios_are_rejected():
    with pytest.raises(ValueError, match="no usable ratios"):
        fit_mos([0.0, -1.0, float("nan")])


def test_non_finite_ratios_are_dropped_not_fatal():
    ref = MOSScale.from_signature(5, 2, tuning=12)
    fit = best_mos(list(ref.ratios) + [float("inf"), 0.0], max_cardinality=12)
    assert fit.n_targets == 7


def test_top_n_is_honoured_and_signatures_are_distinct():
    ref = MOSScale.from_signature(5, 2, tuning=12)
    fits = fit_mos(ref.ratios, max_cardinality=16, top_n=5)
    assert len(fits) <= 5
    assert len({f.signature for f in fits}) == len(fits)


def test_fits_are_ranked_by_score():
    fits = fit_mos([1, 1.2, 1.35, 1.5, 1.7], max_cardinality=16, top_n=5)
    scores = [f.score for f in fits]
    assert scores == sorted(scores)


def test_chance_error_shrinks_with_cardinality():
    small = MOSFit(MOSScale.from_signature(2, 3, tuning="central"),
                   1.0, 1.0, 1.0, 1.0, 1.0, (0,), (0.0,), 1)
    big = MOSFit(MOSScale.from_signature(12, 5, tuning="central"),
                 1.0, 1.0, 1.0, 1.0, 1.0, (0,), (0.0,), 1)
    assert small.chance_error_cents > big.chance_error_cents
    assert small.chance_error_cents == pytest.approx(1200 / (4 * 5))


def test_mos_tuning_returns_plain_ratios():
    ref = MOSScale.from_signature(4, 3, tuning=19)
    tuning = mos_tuning(ref.ratios, max_cardinality=12)
    assert isinstance(tuning, list)
    assert tuning == pytest.approx(list(ref.ratios), abs=1e-6)


def test_explain_fit_reports_every_target():
    ref = MOSScale.from_signature(4, 3, tuning=19)
    text = explain_fit(best_mos(ref.ratios, max_cardinality=12), ref.ratios)
    assert "4L3s" in text
    assert text.count("-> degree") == 7


# --------------------------------------------------------------------------- #
# Trajectories
# --------------------------------------------------------------------------- #
def test_trajectory_tracks_a_change_of_scale():
    a = MOSScale.from_signature(5, 2, tuning=12).ratios
    b = MOSScale.from_signature(4, 3, tuning=19).ratios
    traj = trajectory_from_windows([a, a, b, b, a], max_cardinality=12)
    assert [f.signature for f in traj] == ["5L2s", "5L2s", "4L3s", "4L3s", "5L2s"]


def test_an_empty_window_becomes_none_not_an_error():
    a = MOSScale.from_signature(5, 2, tuning=12).ratios
    traj = trajectory_from_windows([a, [], a], max_cardinality=12)
    assert traj[1] is None
    assert traj[0] is not None and traj[2] is not None


def test_trajectory_dataframe_keeps_a_row_per_window():
    a = MOSScale.from_signature(5, 2, tuning=12).ratios
    traj = trajectory_from_windows([a, [], a], max_cardinality=12)
    df = trajectory_dataframe(traj, times=[0.0, 1.0, 2.0])
    assert len(df) == 3
    assert list(df["time"]) == [0.0, 1.0, 2.0]
    assert df["signature"].tolist() == ["5L2s", None, "5L2s"]


def test_trajectory_weights_are_per_window():
    a = list(MOSScale.from_signature(5, 2, tuning=12).ratios)
    traj = trajectory_from_windows(
        [a, a], weights=[[1.0] * 7, None], max_cardinality=12
    )
    assert all(f is not None for f in traj)


# --------------------------------------------------------------------------- #
# The forward direction: an observed interval, declared the generator
# --------------------------------------------------------------------------- #
#: Four alpha-band peaks of S001 eyes-closed, in Hz.  Real data, and the one
#: input in this file where the two directions demonstrably disagree.
EEG_PEAKS = [10.07, 15.64, 19.31, 22.91]

#: The published reading of those peaks: for each pair, taken as a generator,
#: the smallest scale it supports.  ``(num, den) -> (interval, generator cents,
#: signature, cardinality)``.
EEG_FORWARD = {
    (22.91, 15.64): (1.465, 660.9, "2L3s", 5),
    (15.64, 10.07): (1.553, 762.2, "3L2s", 5),
    (19.31, 15.64): (1.235, 835.1, "3L4s", 7),
    (22.91, 19.31): (1.186, 904.0, "4L1s", 5),
    (22.91, 10.07): (2.275, 976.9, "1L4s", 5),
    (19.31, 10.07): (1.918, 1127.1, "1L4s", 5),
}

#: What the inverse search makes of the same four peaks, capped at six notes.
#:
#: 930.44 rather than the 930.00 recorded before the target set was
#: canonicalised: with four targets and a three-anchor shortlist the coarse
#: scan used to drop whichever target the caller listed first, and 930.00 was
#: the answer that dropping produced.  930.44 is what the exhaustive rotation
#: search returns -- and returned before the change too -- at 0.22 cents less
#: error, for the same signature.
EEG_INVERSE = ("1L3s", 930.44, 18.21, 0.5)


def _bright(fraction):
    """Fold a period fraction into the bright half, independently of derive.py.

    Re-implemented here on purpose: a test that called the library's own
    ``_fold_bright`` could not catch the two directions drifting onto different
    halves of the labyrinth, which is the property that lets them share a plot.
    """
    g = fraction % 1.0
    return 1.0 - g if g < 0.5 else g


class _StubBiotuner:
    """The two attributes :func:`mos_from_biotuner` actually reaches for.

    A real ``compute_biotuner`` is exercised in ``tests/mos/test_integration.py``.
    What is under test here is the ``mode`` switch, and a stub is what makes it
    possible to assert that the switch is checked *before* a tuning is derived.
    """

    def __init__(self, ratios, amps=None):
        self.ratios = list(ratios)
        self.amps = amps
        self.calls = []

    def get_tuning(self, source):
        self.calls.append(source)
        return self.ratios


# -- the bright-half convention, shared with the inverse search -------------- #
@pytest.mark.parametrize(
    "ratios,period",
    [
        ([1.0, 1.125, 1.265625, 1.5], 2.0),          # a stack of fifths
        (EEG_PEAKS, 2.0),                             # raw frequencies
        ([1.0, 1.3, 1.6, 2.05], 2.05),                # a pseudo-octave
        ([1.0, 1.6, 2.2], 3.0),                       # a tritave
        ([1.0, 2 ** 0.5, 1.7], 2.0),                  # a half-period interval
    ],
)
def test_every_forward_generator_is_bright(ratios, period):
    """``g`` and ``period - g`` build the same scale, so only one spelling is
    ever reported -- and it has to be the same one the inverse search uses or
    the two directions cannot be drawn on one axis."""
    readings = forward_scales(ratios, period=period, max_cardinality=12)
    assert readings, "nothing to check"
    assert all(0.5 < r.generator < 1.0 for r in readings)
    assert all(r.scale.period == pytest.approx(period) for r in readings)


def test_forward_and_inverse_fold_generators_the_same_way():
    ratios = [1.0, 1.125, 1.265625, 1.5]
    forward = {round(r.generator, 9) for r in
               forward_scales(ratios, max_cardinality=12)}
    inverse = {round(c, 9) for c in generator_candidates(ratios, grid=0)}
    assert forward <= inverse


# -- the generator is observed, and never refined ---------------------------- #
def test_a_forward_generator_is_always_an_interval_the_input_states():
    """The defining constraint.  Nothing here is optimised: every generator is
    the folded quotient of two inputs, to the last bit."""
    ratios = [1.0, 1.07, 1.31, 1.79, 2.6]
    readings = forward_scales(ratios, max_cardinality=14)
    assert readings
    for r in readings:
        stated = [_bright(math.log2(num / den)) for num, den in r.sources]
        assert min(abs(r.generator - s) for s in stated) < 1e-15


def test_a_pure_stack_proposes_its_own_interval_and_fits_it_exactly():
    """Four consecutive fifths state the fifth outright, so the forward reading
    does not have to search for it, and the scale it builds owes the data
    nothing."""
    stack = [1.0, 1.125, 1.265625, 1.5]
    top = forward_scales(stack, max_cardinality=12)[0]
    assert top.interval_pair == (1.5, 1.0)
    assert top.generator == pytest.approx(FIFTH, abs=1e-15)
    assert top.generator_cents == pytest.approx(701.955, abs=1e-3)
    assert top.signature == "2L3s"
    assert top.error_cents == pytest.approx(0.0, abs=1e-9)
    assert top.coverage == pytest.approx(1.0)


def test_a_stacked_generator_is_never_slid_off_the_observed_value():
    """:func:`fit_mos` refines its winners inside the valid tuning range.  Doing
    that here would swap the audible interval for a nearby inaudible one, so a
    forward reading has to keep the worse number.

    A stack of fifths with a few cents of jitter: every observed quotient is
    now slightly wrong, and the refined generator that splits the difference
    is not one of them.
    """
    jitter = [0.0, 5.0, -4.0, 6.0, -3.0]
    stack = [2 ** (((k * 701.955 + j) % 1200) / 1200)
             for k, j in enumerate(jitter)]
    readings = forward_scales(stack, max_cardinality=12)
    top = readings[0]
    inverse = fit_mos(stack, max_cardinality=12)[0]

    assert top.signature == inverse.signature == "2L3s"
    assert top.generator_cents == pytest.approx(706.955, abs=1e-6)
    assert top.interval_pair[1] == 1.0
    assert inverse.scale.generator_cents == pytest.approx(701.205, abs=1e-3)
    # Refining would have found it; forward mode is not allowed to.
    assert inverse.error_cents < top.error_cents
    assert all(
        abs(r.generator_cents - inverse.scale.generator_cents) > 1.0
        for r in readings
    )


# -- the six worked rows ----------------------------------------------------- #
def test_the_eeg_peaks_reproduce_the_six_published_readings():
    """Pinned against a hand-computed table: each peak pair as a generator, at
    the smallest scale that generator supports."""
    readings = forward_scales(
        EEG_PEAKS, include_ratios=False, min_cardinality=5, max_cardinality=7
    )
    smallest = {}
    for r in readings:
        seen = smallest.get(r.interval_pair)
        if seen is None or r.scale.cardinality < seen.scale.cardinality:
            smallest[r.interval_pair] = r

    assert set(smallest) == set(EEG_FORWARD)
    for pair, (interval, cents, signature, cardinality) in EEG_FORWARD.items():
        got = smallest[pair]
        assert got.interval == pytest.approx(interval, abs=5e-4), pair
        assert got.generator_cents == pytest.approx(cents, abs=0.05), pair
        assert got.signature == signature, pair
        assert got.scale.cardinality == cardinality, pair


def test_a_pair_spanning_more_than_an_octave_reports_the_interval_it_spans():
    """22.91/10.07 is 2.275 -- more than a period.  The generator folds; the
    interval must not, or the reading no longer names anything audible."""
    reading = next(
        r for r in forward_scales(EEG_PEAKS, include_ratios=False,
                                  min_cardinality=5, max_cardinality=7)
        if r.interval_pair == (22.91, 10.07)
    )
    assert reading.interval == pytest.approx(2.275, abs=5e-4)
    assert reading.generator_ratio < 2.0
    assert reading.generator_cents == pytest.approx(976.9, abs=0.05)


def test_including_the_ratios_themselves_is_optional():
    """With raw frequencies, 19.31 is a frequency and not an interval, so
    reading it as a generator would invent a reading the signal never made."""
    pairs_only = forward_scales(EEG_PEAKS, include_ratios=False,
                                max_cardinality=8)
    with_ratios = forward_scales(EEG_PEAKS, include_ratios=True,
                                 max_cardinality=8)
    assert {r.interval_pair for r in pairs_only} < {
        r.interval_pair for r in with_ratios
    }
    assert all(den != 1.0 for r in pairs_only for _, den in r.sources)
    assert any(pair == (19.31, 1.0) for r in with_ratios
               for pair in r.sources)


# -- de-duplication ---------------------------------------------------------- #
def test_two_pairs_stating_the_same_interval_are_one_reading_counted_twice():
    """1.2/1.0 and 1.44/1.2 are the same interval measured twice, not two
    different generators -- and the corroboration is what ``n_sources`` is for."""
    readings = forward_scales([1.0, 1.2, 1.44], max_cardinality=10)
    group = [r for r in readings
             if r.generator_cents == pytest.approx(884.359, abs=1e-2)]
    assert group
    assert len({r.sources for r in group}) == 1
    assert group[0].sources == ((1.2, 1.0), (1.44, 1.2))
    assert group[0].n_sources == 2
    # 1.44/1.0 is a genuinely different interval and keeps its own reading.
    other = {r.generator_cents for r in readings} - {
        r.generator_cents for r in group
    }
    assert len(other) == 1


def test_the_unison_does_not_corroborate_a_ratio_with_itself():
    """Biotuner tunings routinely contain 1/1.  Reading 1.5 as a generator and
    reading the pair 1.5/1.0 as a generator are one observation, not two."""
    readings = forward_scales([1.0, 1.5], max_cardinality=8)
    assert readings
    assert all(r.sources == ((1.5, 1.0),) for r in readings)
    assert all(r.n_sources == 1 for r in readings)


#: Four ratios whose pairwise quotients state two generators half a cent
#: apart, at 699.75 and 700.25 cents.  They land in one de-duplication window,
#: which is the situation in which "which proposal represents the window?" has
#: to be answered by something other than arrival order.
CROWDED_RATIOS = [1.0, 1.498090728472, 1.164733586468, 1.745380599926]


def _forward_summary(readings):
    """Everything about a ranked list that must not move under a permutation.

    Deliberately not the whole object.  ``targets``, ``assignments`` and
    ``residuals`` run parallel to the caller's ratios and are *defined* to
    follow their order; the structure, the numbers and the provenance are not.
    """
    return [
        (r.signature, round(r.generator_cents, 9), round(r.score, 9),
         round(r.error_cents, 9), r.sources)
        for r in readings
    ]


def test_forward_readings_do_not_depend_on_the_order_of_the_input():
    """The input is conceptually a *set*, so the whole ranked list has to be a
    function of the multiset and of nothing else.

    It was not.  Grouping proposals greedily in arrival order let the first
    arrival define its window and speak for it, so simply reversing these four
    ratios returned ``5L7s`` at 700.250 cents with 17.500 cents of error where
    the original order returned ``7L5s`` at 699.750 cents with 16.750 -- a
    different generator, a different error, and a signature flipped to its
    inverse, from the same numbers.
    """
    expected = _forward_summary(
        forward_scales(CROWDED_RATIOS, include_ratios=False, max_cardinality=14)
    )
    assert expected, "nothing to compare"

    shuffled = list(CROWDED_RATIOS)
    random.Random(20240607).shuffle(shuffled)
    orders = [
        list(reversed(CROWDED_RATIOS)),
        CROWDED_RATIOS[1:] + CROWDED_RATIOS[:1],
        sorted(CROWDED_RATIOS),
        sorted(CROWDED_RATIOS, reverse=True),
        shuffled,
    ]
    for order in orders:
        got = forward_scales(order, include_ratios=False, max_cardinality=14)
        assert _forward_summary(got) == expected, order


def test_order_independence_survives_weights_and_the_ratios_themselves():
    """Permuting the weights alongside the ratios is the same measurement, and
    the ratios read as generators in their own right must not reintroduce an
    order dependence of their own."""
    weights = [1.0, 3.0, 0.5, 2.0]
    expected = _forward_summary(
        forward_scales(CROWDED_RATIOS, weights=weights, max_cardinality=14)
    )
    assert expected

    rng = random.Random(4181)
    for _ in range(6):
        order = list(range(len(CROWDED_RATIOS)))
        rng.shuffle(order)
        got = forward_scales(
            [CROWDED_RATIOS[i] for i in order],
            weights=[weights[i] for i in order],
            max_cardinality=14,
        )
        assert _forward_summary(got) == expected, order


def test_a_crowded_window_is_represented_by_its_best_scoring_proposal():
    """Two proposals half a cent apart are one reading, but not equally good
    ones, and the one that explains the targets better is what gets reported.

    The corroboration is not lost in the process: both pairs stay in
    ``sources``, so ``n_sources`` still counts how many intervals landed in the
    window rather than how many happened to be elected.
    """
    top = forward_scales(CROWDED_RATIOS, include_ratios=False,
                         min_cardinality=12, max_cardinality=12)[0]
    assert top.signature == "7L5s"
    assert top.generator_cents == pytest.approx(699.75, abs=1e-3)
    assert top.error_cents == pytest.approx(16.75, abs=1e-3)
    assert top.n_sources == 2
    # sources[0] is the representative, and the generator is its quotient.
    assert top.interval_pair == (1.498090728472, 1.0)
    assert top.generator == pytest.approx(_bright(math.log2(top.interval)),
                                          abs=1e-15)
    # The loser of the election is still on the record as corroboration.
    assert (1.745380599926, 1.164733586468) in top.sources


def test_weights_change_the_score_but_not_which_intervals_are_proposed():
    """An interval is either stated by the signal or it is not; amplitudes have
    no vote in that."""
    ratios = [1.0, 1.19, 1.42, 1.73]
    flat = forward_scales(ratios, max_cardinality=10)
    loud = forward_scales(ratios, weights=[1.0, 40.0, 1.0, 1.0],
                          max_cardinality=10)
    assert {r.interval_pair for r in flat} == {r.interval_pair for r in loud}
    by_pair = {(r.interval_pair, r.signature): r.error_cents for r in flat}
    assert any(
        by_pair[(r.interval_pair, r.signature)] != pytest.approx(r.error_cents)
        for r in loud
    )


# -- the two directions genuinely differ ------------------------------------- #
def test_the_inverse_generator_need_not_be_an_interval_the_signal_states():
    """The whole reason both modes exist.

    On these four peaks the inverse search settles on 930.4 cents, which no
    pair of them states: the nearest observed interval is 26 cents away, far
    outside the one-cent grid on which either direction resolves a generator.
    A forward mode that quietly delegated to the inverse search could not
    produce this gap.
    """
    inverse = fit_mos(EEG_PEAKS, max_cardinality=6)[0]
    signature, cents, error, coverage = EEG_INVERSE
    assert inverse.signature == signature
    assert inverse.scale.generator_cents == pytest.approx(cents, abs=0.05)
    assert inverse.error_cents == pytest.approx(error, abs=0.01)
    assert inverse.coverage == pytest.approx(coverage)

    forward = sorted({r.generator_cents for r in
                      forward_scales(EEG_PEAKS, include_ratios=False,
                                     max_cardinality=12)})
    assert len(forward) == len(EEG_FORWARD)
    assert min(abs(g - inverse.scale.generator_cents) for g in forward) > 20.0


def test_the_inverse_search_reaches_generators_forward_mode_cannot():
    """Structurally, not just on one dataset: forward mode's candidates are a
    subset of the observed quotients, while the inverse search also carries a
    background grid."""
    observed = {_bright(math.log2(a / b))
                for a in EEG_PEAKS for b in EEG_PEAKS if a != b}
    forward = {r.generator for r in
               forward_scales(EEG_PEAKS, include_ratios=False,
                              max_cardinality=12)}
    assert all(min(abs(g - o) for o in observed) < 1e-9 for g in forward)

    searched = generator_candidates(EEG_PEAKS, grid=720)
    assert len(searched) > len(forward)
    unreachable = [c for c in searched
                   if all(abs(c - g) > 1e-3 for g in forward)]
    assert unreachable


# -- scored on the same footing as the inverse fit --------------------------- #
def test_forward_and_inverse_fit_the_same_targets():
    """Same input, same folding, same number of things to explain -- otherwise
    the two error figures are not comparable numbers."""
    for ratios in ([1.0, 1.125, 1.25, 1.5, 2.0], EEG_PEAKS,
                   list(MOSScale.from_signature(5, 2, tuning=31).ratios)):
        forward = forward_scales(ratios, max_cardinality=12)[0]
        inverse = fit_mos(ratios, max_cardinality=12)[0]
        assert forward.n_targets == inverse.n_targets
        assert forward.fit.n_merged == inverse.n_merged
        assert forward.fit.targets == pytest.approx(inverse.targets)


def test_forward_error_comes_out_of_the_inverse_scoring_code():
    """Handing :func:`fit_mos` the forward generator as its only candidate, with
    refinement off, must reproduce the forward numbers exactly -- bit for bit,
    because it is literally the same evaluation."""
    ratios = [1.0, 1.21, 1.47, 1.83]
    forward = forward_scales(ratios, max_cardinality=9)[0]
    same = fit_mos(
        ratios, candidates=[forward.generator], refine=False, n_anchors=None,
        max_cardinality=9, top_n=None,
    )
    twin = next(f for f in same if f.signature == forward.signature)
    assert twin.scale.generator == forward.generator
    assert twin.error_cents == forward.error_cents
    assert twin.rms_error_cents == forward.fit.rms_error_cents
    assert twin.coverage == forward.coverage
    assert twin.score == forward.score
    assert twin.residuals == forward.fit.residuals


def test_a_forward_reading_reports_a_weighted_mean_like_any_other_fit():
    for r in forward_scales([1.0, 1.21, 1.47, 1.83], max_cardinality=9)[:5]:
        assert len(r.fit.residuals) == r.n_targets == len(r.fit.targets)
        assert r.error_cents == pytest.approx(
            float(np.mean(np.abs(r.fit.residuals))), abs=1e-9
        )


def test_a_scale_reads_the_same_forward_and_backward_from_its_own_ratios():
    """When the generator *is* audible the two directions must agree, down to
    the reported error -- a disagreement there would mean one of them is
    measuring something else."""
    ref = MOSScale.from_signature(5, 2, tuning=31)
    forward = forward_scales(ref.ratios, max_cardinality=12)[0]
    inverse = best_mos(ref.ratios, max_cardinality=12)
    assert forward.signature == inverse.signature == "5L2s"
    assert forward.generator == pytest.approx(inverse.scale.generator, abs=1e-9)
    assert forward.error_cents == pytest.approx(inverse.error_cents, abs=1e-9)
    assert forward.coverage == pytest.approx(inverse.coverage)
    assert forward.evidence == pytest.approx(inverse.evidence, abs=1e-9)


def test_forward_readings_are_ranked_by_score():
    readings = forward_scales(EEG_PEAKS, include_ratios=False,
                              max_cardinality=12)
    assert [r.score for r in readings] == sorted(r.score for r in readings)
    assert len(forward_scales(EEG_PEAKS, include_ratios=False,
                              max_cardinality=12, top_n=3)) == 3


def test_a_forward_reading_summarises_like_a_fit_plus_its_provenance():
    top = forward_scales([1.0, 1.125, 1.265625, 1.5], max_cardinality=12)[0]
    d = top.to_dict()
    assert d["signature"] == "2L3s"
    assert d["error_cents"] == pytest.approx(top.error_cents)
    assert d["interval"] == pytest.approx(1.5)
    assert (d["interval_numerator"], d["interval_denominator"]) == (1.5, 1.0)
    assert d["n_sources"] == 2


# -- the mode switch on the biotuner bridge ---------------------------------- #
def test_mode_forward_reads_scales_off_the_observed_intervals():
    bt = _StubBiotuner([1.0, 1.125, 1.265625, 1.5])
    out = mos_from_biotuner(bt, mode="forward", max_cardinality=12)
    assert out and all(isinstance(r, ForwardScale) for r in out)
    assert out[0].signature == "2L3s"
    assert out[0].interval_pair == (1.5, 1.0)
    assert bt.calls == ["peaks_ratios"]


def test_amplitudes_reach_the_forward_scoring_too():
    """A loud peak has to pull a forward reading exactly as hard as it pulls an
    inverse fit, or the two error figures stop being the same measurement."""
    ratios = [1.0, 1.19, 1.42, 1.73]
    amps = [1.0, 40.0, 1.0, 1.0]
    weighted = mos_from_biotuner(_StubBiotuner(ratios, amps), mode="forward",
                                 max_cardinality=10)
    flat = mos_from_biotuner(_StubBiotuner(ratios), mode="forward",
                             max_cardinality=10)
    assert weighted
    assert weighted == forward_scales(ratios, weights=amps, max_cardinality=10)
    assert weighted != flat


def test_mode_inverse_is_the_default_and_is_left_alone():
    ratios = [1.0, 1.125, 1.265625, 1.5]
    default = mos_from_biotuner(_StubBiotuner(ratios), max_cardinality=12)
    explicit = mos_from_biotuner(_StubBiotuner(ratios), mode="inverse",
                                 max_cardinality=12)
    assert all(isinstance(f, MOSFit) for f in default)
    assert default == explicit
    assert default == fit_mos(ratios, max_cardinality=12)


def test_an_unknown_mode_is_refused_before_any_work_is_done():
    bt = _StubBiotuner([1.0, 1.5])
    with pytest.raises(ValueError, match="mode must be 'inverse' or 'forward'"):
        mos_from_biotuner(bt, mode="sideways")
    assert bt.calls == [], "the tuning was derived before the mode was checked"


def test_the_circular_source_is_refused_in_forward_mode_too():
    bt = _StubBiotuner([1.0, 1.5])
    with pytest.raises(ValueError, match="would fit a moment-of-symmetry"):
        mos_from_biotuner(bt, source="mos", mode="forward")
    assert bt.calls == []


# -- degenerate inputs -------------------------------------------------------- #
def test_a_single_ratio_still_states_one_interval_and_says_it_is_thin():
    readings = forward_scales([1.5], max_cardinality=8)
    assert readings
    assert all(r.n_targets == 1 for r in readings)
    assert all(r.generator == pytest.approx(FIFTH, abs=1e-15) for r in readings)
    assert all(r.is_underdetermined for r in readings)
    assert all(r.error_cents == pytest.approx(0.0, abs=1e-9) for r in readings)


def test_two_identical_ratios_are_one_observation_not_two():
    """Their quotient is a unison, which generates nothing, and they fold to a
    single pitch class -- so this must read exactly like the single ratio."""
    twice = forward_scales([1.5, 1.5], max_cardinality=8)
    once = forward_scales([1.5], max_cardinality=8)
    assert [r.signature for r in twice] == [r.signature for r in once]
    assert all(r.n_sources == 1 for r in twice)
    assert all(r.n_targets == 1 for r in twice)
    assert all(r.fit.n_merged == 1 for r in twice)


def test_an_input_of_pure_octaves_states_no_generator_at_all():
    """Every quotient is a period, which folds to nothing.  That is an answer
    about the signal, so it comes back empty rather than raising."""
    assert forward_scales([1.0, 2.0, 4.0], max_cardinality=12) == []


def test_a_unison_in_the_input_is_not_a_generator():
    readings = forward_scales([1.0, 1.0, 1.25, 1.5], max_cardinality=10)
    assert readings
    assert all(r.generator_ratio != pytest.approx(1.0) for r in readings)
    assert all(num != den for r in readings for num, den in r.sources)


def test_a_non_octave_period_folds_the_generator_at_its_own_base():
    """The bright half is half of *this* period.  Folding 1.6 at base 2 would
    put the generator somewhere the tritave scale never goes."""
    readings = forward_scales([1.0, 1.6], period=3.0, max_cardinality=10)
    assert readings
    expected = 1.0 - math.log(1.6) / math.log(3.0)
    for r in readings:
        assert r.generator == pytest.approx(expected, abs=1e-12)
        assert r.scale.period == pytest.approx(3.0)
        assert r.generator_cents == pytest.approx(
            expected * 1200 * math.log2(3.0), abs=1e-6
        )
    octave_reading = forward_scales([1.0, 1.6], max_cardinality=10)[0]
    assert octave_reading.generator != pytest.approx(expected, abs=1e-6)


def test_forward_mode_validates_its_arguments_like_the_inverse_search():
    with pytest.raises(ValueError, match="min_cardinality must be at least 3"):
        forward_scales([1.0, 1.5], min_cardinality=2)
    with pytest.raises(ValueError, match="below min_cardinality"):
        forward_scales([1.0, 1.5], min_cardinality=10, max_cardinality=5)
    with pytest.raises(ValueError, match="must exceed 1"):
        forward_scales([1.0, 1.5], period=1.0)
    with pytest.raises(ValueError, match="dedupe_cents"):
        forward_scales([1.0, 1.5], dedupe_cents=-1.0)
    with pytest.raises(ValueError, match="no usable ratios"):
        forward_scales([0.0, -1.0, float("nan")])


# -- the whole facade, not eight of nineteen readouts ------------------------ #
def test_a_forward_reading_exposes_everything_its_fit_does():
    """Half a facade reads as a claim that the other half does not apply to a
    forward reading, and it does: the two directions run the same evaluation."""
    top = forward_scales([1.0, 1.19, 1.42, 1.73], max_cardinality=10)[0]
    for name in ("scale", "signature", "error_cents", "max_error_cents",
                 "rms_error_cents", "coverage", "score", "assignments",
                 "residuals", "n_targets", "offset", "n_merged", "targets",
                 "n_unmatched_degrees", "aligned_degrees", "aligned_ratios",
                 "aligned_cents", "mode", "chance_error_cents", "improvement",
                 "evidence", "is_underdetermined"):
        assert hasattr(top, name), name
        assert getattr(top, name) == getattr(top.fit, name), name


def test_a_dark_half_quotient_is_reported_as_its_inversion():
    """``generator_ratio`` is not the observed interval reduced into the period.

    19.31/15.64 is 1.235, which sits at 365 cents -- the dark half.  Its bright
    spelling is the complement, so the generator is 1.620 at 835 cents: an
    interval the signal states upside down.  ``interval`` keeps the audible
    one, and the two must not be confused.
    """
    reading = next(
        r for r in forward_scales(EEG_PEAKS, include_ratios=False,
                                  min_cardinality=7, max_cardinality=7)
        if r.interval_pair == (19.31, 15.64)
    )
    assert reading.interval == pytest.approx(1.2347, abs=5e-4)
    assert reading.generator_ratio == pytest.approx(2.0 / reading.interval,
                                                    rel=1e-12)
    assert reading.generator_ratio > math.sqrt(2.0)
    assert reading.generator_cents == pytest.approx(835.1, abs=0.05)


# --------------------------------------------------------------------------- #
# The boundary of the bright half, and the degeneracy it used to admit
# --------------------------------------------------------------------------- #
#: How far apart two degrees have to be before they are two notes, in cents.
#:
#: Derived from the problem, not from the implementation.  The previous
#: version of this file measured distinctness at 1e-9 of a period -- which is
#: :data:`~biotuner.mos.derive.GENERATOR_EPSILON`, the very constant the guard
#: under test is written in -- so a scale could leak a collapsed pitch class by
#: any margin the guard itself tolerated and this file would call it fine.  It
#: did: ``fit_mos`` on five-tone equal input returned an eleven-note ``5L6s``
#: whose degrees sat 0.0002 cents apart, five pitch classes wearing eleven
#: names, and every assertion here passed.
#:
#: A thousandth of a cent is 60,000 times finer than the finest interval
#: anybody has claimed to hear and 4,000 times finer than the narrowest step
#: any scale in this file legitimately wants (0.81 cents), so it cannot fail on
#: a real scale; it is 1,000 times coarser than the noise floor, so it cannot
#: fire on arithmetic.
DISTINCT_PITCH_CLASS_CENTS = 1e-3


def _pitch_classes_are_all_distinct(scale, tolerance_cents=DISTINCT_PITCH_CLASS_CENTS):
    """Do a scale's degrees really name ``cardinality`` different pitches?

    Re-implemented here rather than imported for the same reason as
    :func:`_bright`: a test that called the library's own guard could not catch
    the guard itself being wrong.  The *threshold* is re-derived for the same
    reason -- see :data:`DISTINCT_PITCH_CLASS_CENTS`.
    """
    degrees = sorted(scale.degrees)
    gaps = [b - a for a, b in zip(degrees, degrees[1:])]
    gaps.append(1.0 - degrees[-1] + degrees[0])   # the wrap back to the period
    return min(gaps) * scale.period_cents > tolerance_cents


def test_a_half_period_interval_arrives_off_by_one_ulp():
    """The premise of the bug, pinned: the value that has to be refused is not
    the value the guard was comparing against."""
    assert math.log(2 ** 0.5) / math.log(2) != 0.5
    assert math.log(2 ** 0.5) / math.log(2) == pytest.approx(0.5, abs=1e-15)


def test_a_half_period_interval_generates_nothing_in_either_direction():
    """A generator at half the period closes after two notes, so every scale
    built on it overstates its own note count.

    Forward mode used to answer ``2L3s`` at 600 cents with degrees at
    ``[0, 0, 0, 600, 600]`` cents: two pitch classes wearing a five-note label.
    The honest answer is that the input states no generator at all.

    The inverse search can still *reach* the neighbourhood of 600 cents from
    its background grid, and refinement will slide a ``2L3s`` down to the edge
    of that signature's tuning range.  What it may not do is return a scale
    whose pitch classes have collapsed, and it does not.
    """
    assert generator_candidates([1.0, 2 ** 0.5], grid=0) == []
    assert forward_scales([1.0, 2 ** 0.5], max_cardinality=12) == []
    for fit in fit_mos([1.0, 2 ** 0.5], max_cardinality=12, top_n=None):
        assert _pitch_classes_are_all_distinct(fit.scale), fit


def test_a_generator_a_few_cents_off_half_the_period_still_gets_through():
    """The guard excludes a neighbourhood of the boundary, and the
    neighbourhood has to be small enough to leave real generators alone."""
    ratio = 2 ** ((600 + 3) / 1200)
    readings = forward_scales([1.0, ratio], max_cardinality=12)
    assert readings
    assert all(r.generator_cents == pytest.approx(603.0, abs=1e-6)
               for r in readings)


@pytest.mark.parametrize(
    "ratios",
    [
        [1.0, 2 ** 0.5],
        [1.0, 2 ** 0.5, 1.7],
        [1.0, 1.125, 1.265625, 1.5],
        EEG_PEAKS,
        S004_PEAKS,
        [2 ** (k / 12) for k in range(12)],
        [1.0, 1.19, 1.42, 1.70, 2.84],
        # Equal divisions are where the refinement used to collapse: the
        # optimiser slides the generator onto the rational at the end of the
        # signature's tuning range, because a scale that has fallen onto a
        # smaller one fits equal-tempered data perfectly.  Five- and seven-tone
        # equal each produced a scale with five pitch classes under eleven
        # names; they are here so that cannot come back.
        [2 ** (k / 5) for k in range(5)],
        [2 ** (k / 7) for k in range(7)],
        [2 ** (k / 19) for k in range(19)],
    ],
)
def test_no_returned_scale_repeats_a_pitch_class(ratios):
    """A scale with fewer distinct pitches than degrees is degenerate whatever
    produced it, and the cardinality it reports poisons everything downstream
    -- the surplus-note penalty, the chance error, the unused-degree count."""
    for fit in fit_mos(ratios, max_cardinality=24, top_n=None):
        assert _pitch_classes_are_all_distinct(fit.scale), fit
    for reading in forward_scales(ratios, max_cardinality=24):
        assert _pitch_classes_are_all_distinct(reading.scale), reading


def test_the_degeneracy_guard_leaves_equal_temperaments_alone():
    """An equal division is a *degenerate well-formed scale* in Milne et al.'s
    sense -- its two step sizes are equal -- but its pitch classes are all
    different, and refusing it would throw away the answer to the commonest
    input there is."""
    for n in (5, 7, 12, 19):
        edo = [2 ** (k / n) for k in range(n)]
        fit = fit_mos(edo, max_cardinality=max(12, n))[0]
        assert fit.scale.cardinality == n
        assert fit.scale.is_degenerate            # equal steps
        assert _pitch_classes_are_all_distinct(fit.scale)   # distinct pitches
        assert fit.error_cents == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# Where refinement is allowed to slide the generator
# --------------------------------------------------------------------------- #
def _signatures_up_to(max_cardinality):
    """Every ``(n_large, n_small)`` a scale of this size or smaller can have.

    Co-prime only: a signature whose counts share a factor is not well-formed
    (Milne et al. §2) and has no tuning range to bound.
    """
    for card in range(3, max_cardinality + 1):
        for n_large in range(1, card):
            n_small = card - n_large
            if math.gcd(n_large, n_small) == 1:
                yield n_large, n_small


def _centre_of_range(n_large, n_small, period=2.0):
    """A scale sitting in the middle of its signature's bright tuning range."""
    lo, hi = T.signature_ranges(n_large, n_small)[1]
    return MOSScale(n_large, n_small, (float(lo) + float(hi)) / 2.0, period,
                    validate=False)


def test_no_refinement_bound_stands_on_a_collapsed_tuning():
    """The postcondition the parking bug needed.

    Refinement may put the generator anywhere between the bounds, endpoints
    included, so every scale the bounds name has to be a scale of its stated
    cardinality.  It was not: the margin used to be a millionth of the tuning
    range, which for a range a twelfth of a period wide leaves the generator
    close enough to the collapsing end that whole pitch classes coincide to
    within a ten-thousandth of a cent.

    Checked over every signature up to 40 notes, well past the default ceiling
    of 24, at the threshold this file derives rather than the one the module
    uses.
    """
    checked = 0
    for n_large, n_small in _signatures_up_to(40):
        bounds = _refinement_bounds(_centre_of_range(n_large, n_small))
        if bounds is None:
            continue
        for g in bounds:
            edge = MOSScale(n_large, n_small, g, 2.0, validate=False)
            assert _pitch_classes_are_all_distinct(edge), (
                f"{n_large}L{n_small}s at {g!r}"
            )
            checked += 1
    assert checked > 500, checked


def test_no_refinement_bound_sits_within_an_epsilon_of_its_endpoint():
    """The narrower invariant underneath the audible one.

    A margin of a millionth of the range is a millionth of *whatever the range
    happens to be*, and for a narrow signature that is smaller than the
    arithmetic noise the module tolerates everywhere else: ``39L1s`` spans
    6.4e-4 of a period, so the old bound stood 6.4e-10 from its endpoint, two
    thirds of an epsilon.  A bound the generator can be on top of is not a
    bound, whatever the pitch classes are doing.

    One ulp of slack is allowed, and no more.  ``endpoint + 1e-9`` is a float
    sum, and near 0.5151 the nearest representable answer sits a fraction of a
    femto below the exact one; that is the arithmetic declining to be more
    precise, not the bound being loose.
    """
    for n_large, n_small in _signatures_up_to(60):
        scale = _centre_of_range(n_large, n_small)
        bounds = _refinement_bounds(scale)
        if bounds is None:
            continue
        lo_exact, hi_exact = (float(x) for x in scale.tuning_range)
        lo, hi = bounds
        assert lo - lo_exact >= GENERATOR_EPSILON - np.spacing(lo_exact), (
            n_large, n_small, lo - lo_exact,
        )
        assert hi_exact - hi >= GENERATOR_EPSILON - np.spacing(hi_exact), (
            n_large, n_small, hi_exact - hi,
        )


def test_the_equal_division_end_of_a_range_keeps_its_full_precision():
    """Only one end of a tuning range collapses; the other is the scale's own
    equal division, where the steps are equal but the pitch classes are not.

    Trimming both ends by the same amount would cost the fits that matter
    most -- an equal-tempered input wants the generator *exactly* at that
    endpoint -- so the bound is measured, not assumed, and leaves the safe end
    where it found it.
    """
    scale = MOSScale(5, 2, 18 / 31, validate=False)
    lo_exact, hi_exact = (float(x) for x in scale.tuning_range)
    lo, hi = _refinement_bounds(scale)
    # 4/7 is seven-tone equal: reachable to a millionth of the range.
    assert lo - lo_exact < (hi_exact - lo_exact) * 2e-6
    # 3/5 collapses seven degrees onto five: held off by much more than that.
    assert hi_exact - hi > (hi_exact - lo_exact) * 1e-5


def test_refinement_still_reaches_an_equal_temperament_exactly():
    """The bound must not cost the answers it was introduced to protect."""
    for n in (5, 7, 12, 19, 22):
        edo = [2 ** (k / n) for k in range(n)]
        fit = fit_mos(edo, max_cardinality=max(12, n))[0]
        assert fit.error_cents == pytest.approx(0.0, abs=1e-9)
        assert fit.scale.generator_cents == pytest.approx(
            round(fit.scale.generator * n) * 1200.0 / n, abs=1e-6
        )


def test_every_signature_worth_fitting_keeps_a_refinement_window():
    """The bound trims, it does not close.

    Checked exhaustively to 200 notes -- eight times the default ceiling -- so
    that raising ``max_cardinality`` cannot silently switch refinement off for
    some narrow signature.
    """
    closed = [
        (n_large, n_small)
        for n_large, n_small in _signatures_up_to(200)
        if _refinement_bounds(_centre_of_range(n_large, n_small)) is None
    ]
    assert closed == []


def test_a_range_too_narrow_to_stay_distinct_has_no_window_at_all():
    """Where the branch that gives up is reachable, and why it is so far away.

    A window closes only when the whole tuning range is narrower than twice the
    separation its ends need, and the range of a signature is ``1 / (q1 q2)``
    for endpoint denominators that sum to roughly the cardinality -- so the
    first signature with nowhere safe to slide to has tens of thousands of
    notes.  That is far outside anything this module fits, which is the point:
    the branch is a real one and it is not on the path of any real fit.
    """
    scale = _centre_of_range(39999, 1)
    lo, hi = (float(x) for x in scale.tuning_range)
    assert hi - lo < 2 * MIN_REFINED_STEP
    assert _refinement_bounds(scale) is None


def test_a_closed_window_leaves_the_fit_exactly_as_it_found_it(monkeypatch):
    """And when it does close, nothing slides.

    The scale that would be refined has forty thousand degrees, which is not a
    fit anyone can afford to score, so the closure is injected rather than
    provoked -- the branch under test is what ``_refine_generator`` does with
    the answer, not how the answer was reached.
    """
    monkeypatch.setattr(derive, "_refinement_bounds", lambda scale: None)
    scale = MOSScale(5, 2, 18 / 31, validate=False)
    positions = np.array([0.0, 0.31, 0.62])
    weights = np.full(3, 1 / 3)
    refined = _refine_generator(scale, positions, weights, 15.0, 1.0)
    assert refined.scale.generator == scale.generator
    assert refined.scale.signature == "5L2s"


# --------------------------------------------------------------------------- #
# A set of ratios is a set: nothing may depend on the order it arrives in
# --------------------------------------------------------------------------- #
def _fit_fingerprint(fits):
    """Everything about a ranked list that a permutation must not move.

    Exact values, not approximations.  "Almost the same answer" is not the
    claim -- the claim is that the search never saw the order, so every float
    it produced has to come back bit for bit.
    """
    return [
        (f.signature, f.scale.generator, f.scale.period, f.error_cents,
         f.max_error_cents, f.rms_error_cents, f.coverage, f.score,
         f.offset, f.n_targets, f.n_merged)
        for f in fits
    ]


def _forward_fingerprint(readings):
    return [
        (r.signature, r.scale.generator, r.error_cents, r.max_error_cents,
         r.rms_error_cents, r.coverage, r.score, r.offset, r.n_targets,
         r.n_merged, r.sources)
        for r in readings
    ]


def _per_target(fit):
    """The three vectors that *do* follow the caller, as one zipped list."""
    return list(zip(fit.targets, fit.assignments, fit.residuals))


#: Inputs the invariance has to hold for, and why each one is here.
PERMUTATION_CASES = {
    # An exact MOS: the answer is known, and every permutation must find it.
    "exact_mos": (list(MOSScale.from_signature(5, 2, tuning=31).ratios), None),
    "exact_mos_small": (list(MOSScale.from_signature(1, 3, tuning=9).ratios), None),
    # Jittered: the objective no longer has a clean winner, so ties are close
    # and the tie-break is doing real work.
    "jittered": ([1.0, 1.1247, 1.2662, 1.4991, 1.6871], None),
    # Real biotuner-derived material.
    "eeg_peaks": (EEG_PEAKS, None),
    "s004_peaks": (S004_PEAKS, None),
    "crowded": (CROWDED_RATIOS, None),
    # Duplicates and octave-equivalents, so the fold has groups to merge and a
    # representative to choose.
    "duplicates": ([1.0, 1.5, 1.5, 2.0, 3.0, 1.25], None),
    "near_duplicates": ([1.0, 1.1250, 1.1249, 1.4999, 1.5001, 1.7], None),
    # Weights, carried along with the ratios.
    "weighted": ([1.0, 1.19, 1.42, 1.70, 2.84],
                 [3.0, 1.0, 7.5, 0.25, 2.0]),
    "weighted_ties": ([1.0, 1.19, 1.42, 1.70], [2.0, 2.0, 1.0, 1.0]),
    # An equal division: every signature ties, so the ranking rests entirely on
    # the tie-break.
    "edo12": ([2 ** (k / 12) for k in range(12)], None),
    # Unusable entries, which are dropped -- with their weights.
    "with_junk": ([1.0, float("nan"), 1.5, -2.0, 1.25, 0.0],
                  [1.0, 5.0, 2.0, 9.0, 1.0, 4.0]),
}

#: The same, but fitted against a pseudo-octave rather than 2/1.
NON_OCTAVE_CASES = {
    "tritave": ([1.0, 1.31, 1.72, 2.11, 2.63], 3.0),
    "stretched": ([1.0, 1.19, 1.42, 1.70, 2.02], 2.05),
}


def _seed(name):
    """A per-case seed that does not change between interpreter runs.

    ``hash`` of a string is salted, so a shuffle seeded from it would try
    different orders every run and a failure would not reproduce.
    """
    return sum(i * ord(c) for i, c in enumerate(name, 1))


def _permutations_to_try(n, seed, exhaustive_up_to=4, n_random=8):
    """Every order for a small set, a random sample of them for a larger one.

    Exhaustive where it is affordable: a fit costs about half a second, so 24
    orders of a four-ratio set is the most that can be spent per case and still
    leave a suite anyone runs.  Above that the sample is random but seeded, so
    a failure is reproducible and the union across cases still covers a wide
    variety of orders.
    """
    if n <= exhaustive_up_to:
        return list(itertools.permutations(range(n)))
    rng = random.Random(seed)
    # The identity and the reversal by hand: reversal is the order that broke
    # the forward direction, and a random sample of 8 out of n! rarely hits it.
    out = [tuple(range(n)), tuple(reversed(range(n)))]
    for _ in range(n_random):
        order = list(range(n))
        rng.shuffle(order)
        out.append(tuple(order))
    return out


@pytest.mark.parametrize("name", sorted(PERMUTATION_CASES))
def test_fit_mos_reads_its_ratios_as_a_set(name):
    """``fit_mos`` must be a function of the multiset of ratios, exactly.

    Three separate things used to read the targets in arrival order, all of
    them below both public entry points: the merge kept whichever member of a
    pitch class came first, the rotation shortlist broke its (always tied, by
    default) weights by array position, and the weighted error was summed in
    that order, which decides the last bits and therefore decides exact ties.

    Every permutation is tried for a set small enough to enumerate, and a
    seeded sample of them otherwise.  The scalars are compared for exact
    equality; the three per-target vectors are compared as multisets, because
    those are defined to follow the caller's list and so are *expected* to move
    -- what must not move is which target got which degree and which residual.
    """
    ratios, weights = PERMUTATION_CASES[name]
    reference = fit_mos(ratios, weights=weights, max_cardinality=10, top_n=None)
    assert reference, "nothing to compare"
    expected = _fit_fingerprint(reference)
    expected_rows = [_per_target(f) for f in reference]

    for order in _permutations_to_try(len(ratios), seed=_seed(name)):
        got = fit_mos(
            [ratios[i] for i in order],
            weights=None if weights is None else [weights[i] for i in order],
            max_cardinality=10, top_n=None,
        )
        assert _fit_fingerprint(got) == expected, (name, order)
        # The per-target vectors moved with the input, and only with it.
        for fit, rows in zip(got, expected_rows):
            assert sorted(_per_target(fit)) == sorted(rows), (name, order)


@pytest.mark.parametrize("name", sorted(PERMUTATION_CASES))
def test_forward_scales_reads_its_ratios_as_a_set(name):
    """The same claim, for the forward direction -- which its docstring makes
    out loud, so it had better be true.

    A forward reading costs a fiftieth of what an inverse fit does (nothing is
    optimised), so this one can afford to enumerate every order of a six-ratio
    set rather than sample them.
    """
    ratios, weights = PERMUTATION_CASES[name]
    reference = forward_scales(ratios, weights=weights, max_cardinality=12)
    assert reference, "nothing to compare"
    expected = _forward_fingerprint(reference)
    expected_rows = [_per_target(r.fit) for r in reference]

    for order in _permutations_to_try(len(ratios), seed=_seed(name),
                                      exhaustive_up_to=6, n_random=24):
        got = forward_scales(
            [ratios[i] for i in order],
            weights=None if weights is None else [weights[i] for i in order],
            max_cardinality=12,
        )
        assert _forward_fingerprint(got) == expected, (name, order)
        for reading, rows in zip(got, expected_rows):
            assert sorted(_per_target(reading.fit)) == sorted(rows), (name, order)


@pytest.mark.parametrize("name", sorted(NON_OCTAVE_CASES))
def test_order_independence_holds_for_a_pseudo_octave(name):
    """The period is a free parameter, and every threshold in the fold is
    expressed relative to it, so the invariance has to be checked off the
    octave as well."""
    ratios, period = NON_OCTAVE_CASES[name]
    expected = _fit_fingerprint(
        fit_mos(ratios, period=period, max_cardinality=10, top_n=None)
    )
    assert expected
    for order in _permutations_to_try(len(ratios), seed=99, n_random=5):
        got = fit_mos([ratios[i] for i in order], period=period,
                      max_cardinality=10, top_n=None)
        assert _fit_fingerprint(got) == expected, (name, order)


def test_order_independence_holds_for_random_sets():
    """Hand-picked inputs can hide a bias.  These are not picked at all."""
    rng = np.random.default_rng(20240607)
    for trial in range(6):
        n = int(rng.integers(4, 9))
        ratios = list(np.round(rng.uniform(1.0, 2.0, size=n), 6))
        weights = list(rng.uniform(0.1, 5.0, size=n)) if trial % 2 else None
        expected = _fit_fingerprint(
            fit_mos(ratios, weights=weights, max_cardinality=10, top_n=None)
        )
        assert expected
        for order in _permutations_to_try(n, seed=trial, exhaustive_up_to=0,
                                          n_random=3):
            got = fit_mos(
                [ratios[i] for i in order],
                weights=None if weights is None else [weights[i] for i in order],
                max_cardinality=10, top_n=None,
            )
            assert _fit_fingerprint(got) == expected, (trial, order)


def test_the_per_target_vectors_follow_the_callers_list():
    """The other half of the contract.  Canonicalising the targets would be a
    silent breaking change if it left ``targets``, ``assignments`` and
    ``residuals`` sorted -- ``explain_fit`` and ``plot_fit`` both zip them
    against the ratios the caller handed in."""
    ratios = [1.7, 1.0, 1.42, 1.19]        # deliberately unsorted
    fit = fit_mos(ratios, max_cardinality=9, top_n=1)[0]
    assert fit.n_merged == 0
    assert list(fit.targets) == pytest.approx(ratios, abs=1e-12)

    shuffled = [ratios[i] for i in (2, 0, 3, 1)]
    moved = fit_mos(shuffled, max_cardinality=9, top_n=1)[0]
    assert list(moved.targets) == pytest.approx(shuffled, abs=1e-12)
    for j, i in enumerate((2, 0, 3, 1)):
        assert moved.assignments[j] == fit.assignments[i]
        assert moved.residuals[j] == fit.residuals[i]


def test_a_merged_group_stands_where_its_earliest_member_did():
    """Folding shortens the per-target vectors, and the surviving entries keep
    the caller's order: the pitch class 2.84 shares with 1.42 reports in 1.42's
    slot, because that is where the caller first mentioned it."""
    fit = fit_mos([1.0, 1.19, 1.42, 1.70, 2.84], max_cardinality=9, top_n=1)[0]
    assert fit.n_merged == 1
    assert list(fit.targets) == pytest.approx([1.0, 1.19, 1.42, 1.70], abs=1e-9)

    moved = fit_mos([2.84, 1.70, 1.19, 1.0, 1.42], max_cardinality=9, top_n=1)[0]
    # 2.84 is mentioned first this time, so the shared class reports first --
    # folded into the period, which is what `targets` always holds.
    assert list(moved.targets) == pytest.approx([1.42, 1.70, 1.19, 1.0], abs=1e-9)
    assert moved.error_cents == fit.error_cents
    assert moved.scale.generator == fit.scale.generator


def test_a_tie_between_equal_weights_is_not_broken_at_all():
    """The mechanism, at the level it lives on.

    A shortlist of the heaviest targets has to come from the weights and
    nothing else.  Reading array position instead -- which the old
    ``argsort(weights)[::-1]`` did, and which sorting the targets would have
    gone on doing in a tidier way -- reads either the caller's order or, once
    the targets are sorted, the arbitrary origin the positions are measured
    from: transpose the signal and the three lowest positions are three
    different targets.  So a tie is not broken; every equally heavy target
    anchors, and with uniform weights that makes the shortlist the exact set.
    """
    positions = np.array([0.0, 0.2503, 0.4471, 0.6118, 0.8894])
    degrees = np.array(MOSScale(5, 2, 18 / 31, validate=False).degrees)

    uniform = np.full(positions.size, 1.0 / positions.size)
    exact = _offset_candidates(positions, uniform, degrees, None)
    assert np.array_equal(_offset_candidates(positions, uniform, degrees, 3),
                          exact)
    assert np.array_equal(_offset_candidates(positions, uniform, degrees, 1),
                          exact)

    # Weights that genuinely single a peak out still shorten the list, and the
    # cut still runs to the end of whatever run it lands in.
    graded = np.array([0.6, 0.2, 0.1, 0.05, 0.05])
    assert len(_offset_candidates(positions, graded, degrees, 1)) < len(exact)
    assert np.array_equal(
        _offset_candidates(positions, graded, degrees, 4),
        _offset_candidates(positions, graded, degrees, 5),
    )


def test_transposing_the_signal_does_not_move_the_fit():
    """The same claim one level up, with the generator candidates pinned.

    ``fit_mos`` derives its candidates from the ratios themselves, and a
    transposed ratio is a different candidate, so the *search space* moves
    under transposition whatever the rotation code does.  Handing in a fixed
    candidate list isolates what this change is responsible for: given the same
    generators to try, the scoring must not care where the signal sits.
    """
    ratios = [1.0, 1.19, 1.42, 1.70, 1.93]
    cands = [0.51, 0.5833, 0.6, 0.635, 0.7, 0.75, 0.8333, 0.9]
    base = fit_mos(ratios, candidates=cands, max_cardinality=12,
                   refine=False, top_n=None)
    assert base
    for shift_cents in (137.0, 700.0, -311.0):
        factor = 2.0 ** (shift_cents / 1200.0)
        moved = fit_mos([r * factor for r in ratios], candidates=cands,
                        max_cardinality=12, refine=False, top_n=None)
        assert [f.signature for f in moved] == [f.signature for f in base]
        for a, b in zip(base, moved):
            assert a.error_cents == pytest.approx(b.error_cents, abs=1e-8)
            assert a.scale.generator == b.scale.generator

# --------------------------------------------------------------------------- #
# No collapsed scale escapes, in either direction
# --------------------------------------------------------------------------- #
# Deliberately NOT derived from the implementation. Two earlier attempts at this
# regression used the guard's own threshold to check distinctness, so neither
# could detect the leak it was written for. A fifth of a cent is a musical
# quantity: it is 25x below the melodic just-noticeable difference and 100x
# below the smallest step any well-formed scale in this cardinality range has,
# so a scale that fails it has collapsed whatever the module's constants say.
AUDIBLY_DISTINCT_CENTS = 0.2


def _smallest_step_cents(scale):
    """Smallest gap between two pitch classes, going round the period."""
    degrees = sorted(scale.degrees)
    n = len(degrees)
    gaps = [(degrees[(i + 1) % n] - degrees[i]) % 1.0 for i in range(n)]
    return min(gaps) * scale.period_cents


@pytest.mark.parametrize(
    "ratios",
    [
        [1.0, 2 ** 0.25, 2 ** 0.5, 2 ** 0.75],   # 4-EDO: every interval rational
        [1.0, 1.1892, 1.4142, 1.6818],           # the same, rounded -- near-rational
        [1.0, 2 ** 0.5],                          # exactly half the period
        [1.0, 1.5, 1.25, 1.75],
        [1.0, 1.333, 1.5, 1.777],
        [10.07, 15.64, 19.31, 22.91],            # real EEG peaks
    ],
)
def test_no_direction_returns_a_scale_that_has_collapsed(ratios):
    """A cardinality is a claim about how many notes there are.

    Both directions build scales at every MOS cardinality a generator supports,
    and near a rational generator the higher ones collapse: the degrees stay
    arithmetically distinct while the small step shrinks to hundredths of a
    cent, so the scale wears more names than it has notes. That corrupts the
    surplus-note penalty, the chance error and the signature itself.
    """
    for direction in (
        MO_forward := forward_scales(ratios),
        fit_mos(ratios, top_n=None),
    ):
        for item in direction:
            scale = item.scale
            step = _smallest_step_cents(scale)
            assert step >= AUDIBLY_DISTINCT_CENTS, (
                f"{scale.signature} claims {scale.cardinality} notes but its "
                f"smallest step is {step:.5f} cents (generator "
                f"{scale.generator_cents:.4f} c)"
            )
            distinct = len({round(d * scale.period_cents / AUDIBLY_DISTINCT_CENTS)
                            for d in scale.degrees})
            assert distinct == scale.cardinality, (
                f"{scale.signature} has {distinct} audibly distinct pitch "
                f"classes for {scale.cardinality} degrees"
            )


def test_refinement_never_slides_onto_the_collapsing_endpoint():
    """The optimiser prefers the endpoint; the guard has to overrule it.

    A collapsed scale always scores at least as well as the scale it collapsed
    from, because it has spare degrees to put wherever the data happens to be.
    So the objective cannot be trusted to stay off the edge and the refined
    result has to be checked on its own terms.
    """
    ratios = [1.0, 2 ** 0.25, 2 ** 0.5, 2 ** 0.75]
    for fit in fit_mos(ratios, top_n=None):
        assert _smallest_step_cents(fit.scale) >= AUDIBLY_DISTINCT_CENTS
        # The 4-EDO landmark of 4L3s sits at exactly 900 cents.
        if fit.scale.signature == "4L3s":
            assert abs(fit.scale.generator_cents - 900.0) > 0.5

def test_a_missing_cell_is_none_whatever_pandas_decides_a_string_column_is():
    """CI failed on this while the same code passed locally, on pandas alone.

    pandas 3 infers a dedicated string dtype for a column of strings and writes
    ``nan`` into its gaps; pandas 2 leaves the column as ``object`` and keeps
    the ``None`` it was handed. A caller writing ``is None`` would therefore
    work on one machine and quietly stop working on another, so the library
    pins the answer rather than inheriting it.

    ``future.infer_string`` is how pandas 2 opts into the pandas 3 behaviour,
    which makes the future testable today.
    """
    import pandas as pd

    from biotuner.mos.derive import trajectory_dataframe

    scale = MOSScale.from_signature(5, 2, tuning=31)
    fit = fit_mos(scale.ratios, max_cardinality=9, top_n=1)[0]

    previous = pd.get_option("future.infer_string")
    try:
        for infer in (False, True):
            pd.set_option("future.infer_string", infer)
            df = trajectory_dataframe([fit, None, fit])
            assert df["signature"].iloc[1] is None, (
                f"future.infer_string={infer} gave "
                f"{df['signature'].iloc[1]!r}, not None"
            )
            # The pandas-idiomatic test has to keep working too.
            assert bool(df["signature"].isna().iloc[1])
            assert df["signature"].iloc[0] == scale.signature
    finally:
        pd.set_option("future.infer_string", previous)
