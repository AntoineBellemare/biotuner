import numpy as np
import pytest
from biotuner.peaks_extension import (
    EEG_harmonics_mult,
    EEG_harmonics_div,
    harmonic_fit,
    multi_consonance
)

def test_harmonic_multiplication():
    """
    Test the EEG_harmonics_mult function.
    """
    peaks = [10.0, 20.0, 30.0]
    n_harmonics = 3
    mult_harmonics = EEG_harmonics_mult(peaks, n_harmonics, n_oct_up=1)
    
    assert mult_harmonics.shape == (len(peaks), n_harmonics + 1), \
        "EEG_harmonics_mult returned incorrect dimensions"
    assert np.all(mult_harmonics > 0), \
        "EEG_harmonics_mult returned non-positive values"

def test_harmonic_division():
    """
    Test the EEG_harmonics_div function.
    """
    peaks = [10.0, 20.0, 30.0]
    n_harmonics = 3
    div_harmonics, div_harm_bound = EEG_harmonics_div(peaks, n_harmonics, n_oct_up=1, mode="div")
    
    assert div_harmonics.shape == (len(peaks), n_harmonics + 1), \
        "EEG_harmonics_div returned incorrect dimensions"
    assert np.all(div_harm_bound >= 1) and np.all(div_harm_bound <= 2), \
        "EEG_harmonics_div bounded harmonics are not within [1, 2]"

def test_harmonic_fit():
    """
    Test the harmonic_fit function.
    """
    peaks = [10.0, 20.0, 30.0]
    n_harmonics = 10
    bounds = 0.1
    harm_fit, harmonics_pos, most_common_harmonics, matching_positions = harmonic_fit(
        peaks, n_harm=n_harmonics, bounds=bounds, function="mult"
    )
    
    assert len(harm_fit) > 0, \
        "harmonic_fit returned no common harmonics"
    assert all(isinstance(h, float) for h in harm_fit), \
        "harmonic_fit returned non-float harmonics"

def test_multi_consonance():
    """
    Test the multi_consonance function.
    """
    cons_pairs = [[10.0, 20.0], [20.0, 30.0], [30.0, 10.0]]
    top_consonant_freqs = multi_consonance(cons_pairs, n_freqs=2)
    
    assert len(top_consonant_freqs) == 2, \
        "multi_consonance did not return the correct number of consonant frequencies"
    assert all(f in [10.0, 20.0, 30.0] for f in top_consonant_freqs), \
        "multi_consonance returned frequencies outside the original set"
