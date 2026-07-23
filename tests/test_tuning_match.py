"""Tests for biotuner.bioelements.tuning_match — ratio-structure matching."""
import numpy as np
import pytest

from biotuner.bioelements import (
    tuning_vector, tuning_cosine, match_tuning, match_elements_by_tuning,
    match_materials_by_tuning, element_tuning,
)
from biotuner.bioelements.matching import match_elements

# a signal with a rich, simple tuning (harmonic stack): ratios 3:2, 4:3, 5:4, ...
PEAKS = np.array([200.0, 300.0, 400.0, 500.0, 600.0, 700.0])


def test_tuning_vector_reduces_to_simple_ratios():
    tv = tuning_vector(PEAKS, maxdenom=24)
    assert (3, 2) in tv and (4, 3) in tv and (5, 4) in tv       # the harmonic scale
    assert all(p > q for (p, q) in tv)                          # numerator ≥ denominator
    assert all(v > 0 for v in tv.values())


def test_octave_invariance():
    """Octave-reducing means a peak moved by an octave (no frequency collision)
    leaves the tuning unchanged."""
    a = tuning_vector(PEAKS)
    shifted = PEAKS.copy(); shifted[-1] *= 2.0                  # 700 -> 1400, no collision
    b = tuning_vector(shifted)
    assert tuning_cosine(a, b) == pytest.approx(1.0, abs=1e-9)


def test_transposition_invariance_is_the_defining_property():
    """The whole point: transposing the signal leaves the TUNING score identical,
    while the POSITION score moves. Ratio structure vs absolute placement."""
    factor = 1.37                                              # arbitrary transposition
    tun_a = match_elements_by_tuning(PEAKS, top=30)
    tun_b = match_elements_by_tuning(PEAKS * factor, top=30)
    # same ranking + same scores under transposition
    a = tun_a.set_index("element")["tuning_score"]
    b = tun_b.set_index("element")["tuning_score"]
    assert np.allclose(a.values, b.reindex(a.index).values, atol=1e-9)

    # the position matcher, by contrast, is NOT transposition-invariant
    pos_a = match_elements(PEAKS, top=30).set_index("element")["score"]
    pos_b = match_elements(PEAKS * factor, top=30).set_index("element")["score"]
    assert not np.allclose(pos_a.values, pos_b.reindex(pos_a.index).values, atol=1e-6)


def test_tuning_and_position_answer_different_questions():
    """Ratio-structure and line-position rankings should genuinely differ."""
    tun_top = list(match_elements_by_tuning(PEAKS, top=40)["element"].head(5))
    pos_top = list(match_elements(PEAKS, top=40)["element"].head(5))
    assert tun_top != pos_top                                  # not the same top-5


def test_density_normalised_not_dominated_by_line_count():
    """Cosine must not simply rank line-rich elements first (naive containment does)."""
    df = match_elements_by_tuning(PEAKS, top=40)
    assert df["tuning_score"].std() > 0.1                      # discriminative spread
    assert 0.0 <= df["tuning_score"].min() and df["tuning_score"].max() <= 1.0


def test_match_tuning_accepts_element_spectrum_and_vector():
    name_score = match_tuning(PEAKS, "Iron")
    vec_score = match_tuning(PEAKS, element_tuning("Iron"))     # prebuilt vector
    assert name_score == pytest.approx(vec_score, abs=1e-9)
    assert 0.0 <= name_score <= 1.0


def test_match_tuning_accepts_material():
    from biotuner.bioelements.materials import MATERIALS
    s = match_tuning(PEAKS, MATERIALS["Water"])                # Composition target
    assert 0.0 <= s <= 1.0


def test_match_materials_by_tuning_ranks_dictionary():
    df = match_materials_by_tuning(PEAKS, top=40)
    assert "tuning_score" in df.columns and len(df) > 50       # 68 non-element materials
    assert (df["kind"] != "element").all()                     # elements excluded by default
    assert list(df["tuning_score"]) == sorted(df["tuning_score"], reverse=True)
    assert 0.0 <= df["tuning_score"].min() and df["tuning_score"].max() <= 1.0


def test_match_materials_by_tuning_is_transposition_invariant():
    a = match_materials_by_tuning(PEAKS, top=30).set_index("material")["tuning_score"]
    b = match_materials_by_tuning(PEAKS * 1.37, top=30).set_index("material")["tuning_score"]
    assert np.allclose(a.values, b.reindex(a.index).values, atol=1e-9)


def test_empty_or_single_peak_is_safe():
    assert tuning_vector(np.array([10.0])) == {}               # no pairs → empty tuning
    assert match_tuning(np.array([10.0]), "Iron") == 0.0
