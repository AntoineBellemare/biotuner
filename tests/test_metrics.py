import pytest
import numpy as np
from fractions import Fraction
from biotuner.metrics import (
    compute_consonance,
    euler,
    tenneyHeight,
    metric_denom,
    dyad_similarity,
    compute_subharmonics,
    compute_subharmonic_tension,
    compute_subharmonics_2lists,
)

def test_compute_consonance_properties():
    # Test basic properties of consonance
    ratio = 3 / 2
    cons = compute_consonance(ratio)
    assert cons > 0, "Consonance should be positive"
    assert cons < 2, "Consonance for any interval should not exceed 2"

    # Test symmetry
    ratio_inverse = 2 / 3
    cons_inverse = compute_consonance(ratio_inverse)
    assert cons == cons_inverse, "Consonance should be symmetric for inverse ratios"

    # Test edge cases
    cons_unison = compute_consonance(1)
    assert cons_unison == 2, "Consonance of unison (1:1) must be 2"

def test_euler_consistency():
    # Test Euler's "gradus suavitatis" for consistency
    peaks = [3, 5, 7]
    gs = euler(*peaks)
    assert isinstance(gs, int), "Euler's gradus suavitatis must return an integer"
    assert gs > 0, "Euler's value should be positive"

def test_tenneyHeight_increasing_dissonance():
    # Test increasing dissonance with larger numbers
    simple_peaks = [3, 5]
    complex_peaks = [3, 11]
    simple_th = tenneyHeight(simple_peaks)
    complex_th = tenneyHeight(complex_peaks)
    assert simple_th < complex_th, "Simpler intervals should have lower Tenney Height"

def test_metric_denom_properties():
    # Denominator metric should reflect complexity
    simple_ratio = 3 / 2
    complex_ratio = 8 / 7
    assert metric_denom(simple_ratio) < metric_denom(complex_ratio), \
        "Simpler ratios should have smaller denominators"

def test_dyad_similarity_monotonicity():
    # Test similarity decreases with increasing complexity
    simple_ratio = 3 / 2
    complex_ratio = 8 / 7
    assert dyad_similarity(simple_ratio) > dyad_similarity(complex_ratio), \
        "Simpler ratios should have higher harmonic similarity"

@pytest.mark.parametrize("peaks", [
    [3, 5, 7],
    [2, 4, 6],
    [11, 13, 17],
])
def test_tenneyHeight_consistency(peaks):
    # Check Tenney Height symmetry and consistency
    th = tenneyHeight(peaks)
    peaks_reversed = peaks[::-1]
    th_reversed = tenneyHeight(peaks_reversed)
    assert th == pytest.approx(th_reversed), "Tenney Height should be symmetric"

def test_random_ratios_consonance():
    # Generate random ratios and check their consonance distribution
    np.random.seed(42)
    random_ratios = np.random.uniform(1.1, 2, 100)
    consonances = [compute_consonance(r) for r in random_ratios]
    assert all(c > 0 for c in consonances), "Consonance values should be positive"
    assert np.mean(consonances) < 1, "Average consonance should be below the unison threshold"

def test_compute_subharmonics():
    # Basic test for computing subharmonics
    chord = [3, 5, 7]
    n_harmonics = 5
    delta_lim = 20
    subharms, common_subharms, delta_t = compute_subharmonics(chord, n_harmonics, delta_lim)

    assert len(subharms) == len(chord), "Subharmonics list length should match chord length"
    assert all(len(s) == n_harmonics for s in subharms), "Each frequency should have n_harmonics subharmonics"
    assert isinstance(common_subharms, list), "Common subharmonics should be a list"
    assert isinstance(delta_t, list), "Delta_t should be a list"
    if common_subharms:
        assert all(np.abs(s1 - s2) < delta_lim for s1, s2 in common_subharms), \
            "Common subharmonics should be within delta_lim"

def test_compute_subharmonic_tension():
    # Basic test for subharmonic tension
    chord = [31, 51, 71]
    n_harmonics = 5
    delta_lim = 20
    common_subs, delta_t, subharm_tension, harm_temp = compute_subharmonic_tension(chord, n_harmonics, delta_lim)

    assert isinstance(subharm_tension, list), "Subharmonic tension should be a list"
    if subharm_tension != "NaN":
        assert len(subharm_tension) == 1, "Subharmonic tension list should have one entry"
        assert subharm_tension[0] >= 0, "Subharmonic tension should be non-negative"
    assert isinstance(harm_temp, list), "Harmonic tension should be a list"

def test_compute_subharmonics_2lists():
    # Test subharmonics for pairs from two lists
    list1 = [5, 9]
    list2 = [13, 20, 7]
    n_harmonics = 5
    delta_lim = 30

    common_subs, delta_t, sub_tension, harm_temp, pair_melody = compute_subharmonics_2lists(
        list1, list2, n_harmonics, delta_lim
    )

    assert isinstance(common_subs, list), "Common subharmonics should be a list"
    assert isinstance(delta_t, list), "Delta_t should be a list"
    assert isinstance(sub_tension, float), "Subharmonic tension should be a float"
    if harm_temp:
        assert all(h > 0 for h in harm_temp), "Harmonic tension values should be positive"
    assert isinstance(pair_melody, tuple), "Pair melody should be a tuple"

def test_compute_subharmonic_tension_edge_cases():
    # Edge case: single frequency
    chord = [50]
    n_harmonics = 5
    delta_lim = 20
    _, _, subharm_tension, _ = compute_subharmonic_tension(chord, n_harmonics, delta_lim)
    assert subharm_tension == "NaN", "Subharmonic tension for single frequency should be NaN"
