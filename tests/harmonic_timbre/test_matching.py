"""Tests for biotuner.harmonic_timbre.matching."""

from __future__ import annotations

import numpy as np
import pytest

from biotuner.harmonic_timbre import (
    match_timbre,
    match_direct,
    match_consonance_weighted,
    match_sethares,
    match_harmonic_entropy,
    match_hybrid,
)
from biotuner.harmonic_timbre.matching import _dissonance_over_scale
from biotuner.harmonic_timbre._utils import amplitude_falloff


# ---------------------------------------------------------------------------
# match_direct
# ---------------------------------------------------------------------------

def test_match_direct_partials_at_input_ratios(simple_just_tuning):
    t = match_direct(simple_just_tuning, n_partials=len(simple_just_tuning), base_freq=100.0)
    assert t.matching_method == "direct"
    expected = np.asarray(sorted(set(simple_just_tuning))) * 100.0
    np.testing.assert_allclose(np.sort(t.partials_hz), np.sort(expected), rtol=1e-9)


def test_match_direct_extends_via_equave(simple_just_tuning):
    t = match_direct(simple_just_tuning, n_partials=12, base_freq=100.0, equave=2.0)
    assert t.n_partials() == 12
    # the 9th partial should be ratio[0] * equave = 1 * 2 = 2 (Hz: 200)
    assert any(np.isclose(t.partials_hz, 200.0, rtol=1e-3))


def test_match_direct_empty_ratios_raises():
    with pytest.raises(ValueError, match="empty"):
        match_direct([])


def test_match_direct_amplitude_falloff_kinds():
    for kind in ("1_over_n", "1_over_n_squared", "flat", "exponential"):
        t = match_direct([1.0, 1.5, 2.0], n_partials=4, amplitude_falloff_kind=kind)
        assert t.amplitudes.shape == (4,)


# ---------------------------------------------------------------------------
# match_consonance_weighted
# ---------------------------------------------------------------------------

def test_match_consonance_weighted_assigns_higher_amp_to_simpler_ratios():
    # With dyad similarity, 3:2 (consonant) should get higher weight than 7:5 (less consonant)
    ratios = [1.0, 3 / 2, 7 / 5, 2.0]
    t = match_consonance_weighted(ratios, n_partials=len(ratios), base_freq=1.0)
    # find indices of those ratios in t.partials_hz
    idx_3_2 = int(np.argmin(np.abs(t.partials_hz - 1.5)))
    idx_7_5 = int(np.argmin(np.abs(t.partials_hz - 1.4)))
    assert t.amplitudes[idx_3_2] > t.amplitudes[idx_7_5]


def test_match_consonance_weighted_metric_options():
    ratios = [1.0, 3 / 2, 2.0]
    for metric in ("dyad_similarity", "consonance", "tenney", "euler"):
        t = match_consonance_weighted(ratios, metric=metric)
        assert t.metadata["metric"] == metric
        assert t.matching_method == "consonance_weighted"


# ---------------------------------------------------------------------------
# match_sethares — property: dissonance non-increasing
# ---------------------------------------------------------------------------

def test_match_sethares_dissonance_monotone(simple_just_tuning):
    n = 8
    t = match_sethares(simple_just_tuning, n_partials=n, base_freq=100.0,
                       max_iter=50, perturbation=0.05)
    # baseline: dissonance with seed (harmonic series)
    seed = 100.0 * np.arange(1, n + 1, dtype=np.float64)
    amps = amplitude_falloff(n, "1_over_n")
    base_diss = _dissonance_over_scale(seed, amps, np.asarray(simple_just_tuning))
    final_diss = _dissonance_over_scale(t.partials_hz, t.amplitudes, np.asarray(simple_just_tuning))
    assert final_diss <= base_diss + 1e-9
    assert "objective_initial" in t.metadata
    assert t.metadata["objective_final"] <= t.metadata["objective_initial"] + 1e-9


# ---------------------------------------------------------------------------
# match_harmonic_entropy
# ---------------------------------------------------------------------------

def test_match_harmonic_entropy_runs_without_error(simple_just_tuning):
    t = match_harmonic_entropy(simple_just_tuning, base_freq=100.0)
    assert t.matching_method == "harmonic_entropy"
    assert t.amplitudes.shape == t.partials_hz.shape
    assert np.all(np.isfinite(t.amplitudes))


# ---------------------------------------------------------------------------
# match_hybrid
# ---------------------------------------------------------------------------

def test_match_hybrid_default_weights(simple_just_tuning):
    t = match_hybrid(simple_just_tuning, n_partials=8, base_freq=100.0)
    assert t.matching_method == "hybrid"
    assert "weights" in t.metadata
    # weights normalized
    assert abs(sum(t.metadata["weights"].values()) - 1.0) < 1e-9


def test_match_hybrid_zero_weights_raises():
    with pytest.raises(ValueError):
        match_hybrid([1.0, 2.0], weights={"direct": 0.0, "sethares": 0.0})


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def test_match_timbre_dispatch():
    ratios = [1.0, 3 / 2, 2.0]
    for method in ("direct", "consonance_weighted"):
        t = match_timbre(ratios, method=method)
        assert t.matching_method == method


def test_match_timbre_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown"):
        match_timbre([1.0, 2.0], method="not-a-real-method")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_match_direct_with_fractions():
    from fractions import Fraction
    rs = [Fraction(1), Fraction(3, 2), Fraction(2)]
    t = match_direct(rs)
    assert t.n_partials() == 3
