import pytest
import numpy as np
from fractions import Fraction
from biotuner.scale_construction import (
    oct_subdiv,
    harmonic_tuning,
    generator_interval_tuning,
    tuning_MOS_info,
    measure_symmetry
)
@pytest.mark.parametrize(
    "ratio, n, expected_length",
    [
        (3/2, 3, 3),  # Should return exactly `n` subdivisions
        (4/3, 5, 5),  # Should return exactly `n` subdivisions
    ]
)
def test_oct_subdiv(ratio, n, expected_length):
    subdivisions, oct_values = oct_subdiv(ratio, n=n)
    
    # Check output length matches `n`
    assert len(subdivisions) == expected_length
    
    # Check subdivisions are integers
    assert all(isinstance(s, int) for s in subdivisions)
    
    # Check oct_values are floats and within reasonable bounds
    assert all(isinstance(v, float) for v in oct_values)
    assert all(1.0 <= v <= 2.0 for v in oct_values)


@pytest.mark.parametrize(
    "list_harmonics, octave, expected_range",
    [
        ([3, 5, 7], 2, (1.0, 2.0)),  # Ratios must lie between 1.0 and 2.0
        ([2, 4], 2, (1.0, 2.0)),     # Ratios must lie between 1.0 and 2.0
    ]
)
def test_harmonic_tuning(list_harmonics, octave, expected_range):
    tuning = harmonic_tuning(list_harmonics, octave=octave)
    
    # Check tuning is a list of floats
    assert all(isinstance(t, float) for t in tuning)
    
    # Check tuning values lie within the expected range
    assert all(expected_range[0] <= t <= expected_range[1] for t in tuning)

def test_generator_interval_tuning():
    interval, steps = 3/2, 12
    tuning, _ = generator_interval_tuning(interval, steps)
    
    # Check tuning length matches steps
    assert len(tuning) == steps
    
    # Check tuning is sorted and monotonic
    assert tuning == sorted(tuning)
    
    # Check tuning values lie within [1.0, 2.0)
    assert all(1.0 <= t < 2.0 for t in tuning)

@pytest.mark.parametrize(
    "interval, steps",
    [
        (3/2, 12),  
        (4/3, 10),  
    ]
)
def test_tuning_MOS_info(interval, steps):
    n_gaps, large, small, tuning, distances = tuning_MOS_info(interval, steps)

    
    # Check tuning is sorted and unique
    assert list(tuning) == sorted(set(tuning))

@pytest.mark.parametrize(
    "generator_interval, max_steps, expected_max_deviation_range",
    [
        (3/2, 20, (0.0, 1.0)),  # Deviation should be between 0.0 and 1.0
        (4/3, 15, (0.0, 1.0)),  # Deviation should be between 0.0 and 1.0
    ]
)
def test_measure_symmetry(generator_interval, max_steps, expected_max_deviation_range):
    max_deviation = measure_symmetry(generator_interval, max_steps=max_steps)
    
    # Check max deviation is within the expected range
    assert expected_max_deviation_range[0] <= max_deviation <= expected_max_deviation_range[1]
