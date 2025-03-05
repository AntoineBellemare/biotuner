import pytest
import numpy as np
from fractions import Fraction
import sympy as sp
from biotuner.scale_construction import (
    oct_subdiv,
    harmonic_tuning,
    generator_interval_tuning,
    tuning_MOS_info,
    measure_symmetry,
    create_mode,
    harmonic_entropy,
    tuning_reduction,
    euler_fokker_scale,
    create_mode,
    pac_mode,
    diss_curve,
    dissmeasure,
    horogram_tree_steps,
    horogram_tree,
    phi_convergent_point,
    Stern_Brocot,
    interval_exponents,
    interval_to_radian,
    tuning_to_radians,
    compare_oct_div,
    multi_oct_subdiv,
    convergents,
    MOS_metric_harmonic_mean,
    generator_to_stern_brocot_fractions,
)
from biotuner.metrics import dyad_similarity, metric_denom


@pytest.mark.parametrize(
    "ratio, n, expected_length",
    [
        (3 / 2, 3, 3),  # Should return exactly `n` subdivisions
        (4 / 3, 5, 5),  # Should return exactly `n` subdivisions
    ],
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


def test_oct_subdiv_precision():
    result = oct_subdiv(3 / 2, n=3)
    expected = (
        [12, 53, 106],
        [
            pytest.approx(1.0136, rel=1e-3),
            pytest.approx(1.0020, rel=1e-3),
            pytest.approx(1.0041, rel=1e-3),
        ],
    )
    assert result == expected


@pytest.mark.parametrize(
    "list_harmonics, octave, expected_range",
    [
        ([3, 5, 7], 2, (1.0, 2.0)),  # Ratios must lie between 1.0 and 2.0
        ([2, 4], 2, (1.0, 2.0)),  # Ratios must lie between 1.0 and 2.0
    ],
)
def test_harmonic_tuning(list_harmonics, octave, expected_range):
    tuning = harmonic_tuning(list_harmonics, octave=octave)

    # Check tuning is a list of floats
    assert all(isinstance(t, float) for t in tuning)

    # Check tuning values lie within the expected range
    assert all(expected_range[0] <= t <= expected_range[1] for t in tuning)


def test_euler_fokker_scale_correct_rebounding():
    """
    Test whether normalization correctly rebounds values within octave bounds.
    """
    intervals = [3, 5]
    scale = euler_fokker_scale(intervals, n=1, octave=2, normalize=True)

    for val in scale:
        assert 1 <= val <= 2  # Ensure all values are within the octave


def test_euler_fokker_scale_large_input():
    """
    Ensure function runs correctly for larger prime factors.
    """
    intervals = [2, 3, 5, 7, 11]
    scale = euler_fokker_scale(intervals, n=1, octave=2, normalize=True)

    assert len(scale) > 5  # Should produce multiple steps
    assert sp.Integer(1) in scale  # Must include fundamental
    assert sp.Integer(2) in scale  # Must include octave


def test_euler_fokker_scale_basic():
    """
    Test the basic functionality of the euler_fokker_scale function
    with a known example.
    """
    intervals = [3, 5, 7]
    expected_scale = [
        sp.Integer(1),
        sp.Rational(35, 32),
        sp.Rational(5, 4),
        sp.Rational(21, 16),
        sp.Rational(3, 2),
        sp.Rational(105, 64),
        sp.Rational(7, 4),
        sp.Rational(15, 8),
        sp.Integer(2),
    ]

    scale = euler_fokker_scale(intervals, n=1, octave=2, normalize=True)

    assert set(scale) == set(expected_scale)  # Order doesn't matter


def test_euler_fokker_scale_multiple_multiplicities():
    """
    Test if using different multiplicities yields the same result as
    repeating factors.
    """
    intervals_1 = [3, 5, 7]
    intervals_2 = [3, 3, 5, 5, 7, 7]  # Equivalent to multiplicity=2 for 3 and 5

    scale_1 = euler_fokker_scale(intervals_1, n=2, octave=2, normalize=True)
    scale_2 = euler_fokker_scale(intervals_2, n=1, octave=2, normalize=True)

    assert set(scale_1) == set(scale_2)  # Both should be equivalent


def test_generator_interval_tuning():
    """
    Tests generator_interval_tuning to ensure:
    - Correct number of tuning steps
    - Monotonic order of generated tuning
    - Values lie within the expected octave range
    - Edge cases for small and large step values
    """
    interval, steps = 3 / 2, 12
    tuning, _ = generator_interval_tuning(interval, steps)

    # Check correct number of steps
    assert len(tuning) == steps, f"Expected {steps} tuning steps, got {len(tuning)}"

    # Check tuning is sorted and monotonic
    assert tuning == sorted(tuning), "Tuning values are not sorted in ascending order"

    # Check all values are within the correct octave
    assert all(
        1.0 <= t < 2.0 for t in tuning
    ), "Some tuning values fall outside the [1.0, 2.0) range"


def test_generator_interval_tuning_edge_cases():
    """
    Tests edge cases for generator_interval_tuning:
    - Minimum number of steps
    - Large step count for performance
    - Exact octave match
    """
    # Single step should return just the fundamental frequency
    tuning, _ = generator_interval_tuning(3 / 2, 1)
    assert tuning == [1.0], f"Expected [1.0], got {tuning}"

    # Large step count should remain within computational limits
    large_steps = 100
    tuning, _ = generator_interval_tuning(3 / 2, large_steps)
    assert len(tuning) == large_steps, "Large step count should not fail"

    # Exact octave match test (should include 2.0 at the end)
    tuning, _ = generator_interval_tuning(2.0, 12)
    assert pytest.approx(tuning[-1]) == 2.0, "Last step should be the octave (2.0)"


def test_create_mode():
    tuning = [1, 1.2, 1.3, 1.5, 1.6, 1.8]
    mode = create_mode(tuning, n_steps=3, function=dyad_similarity)
    assert len(mode) == 3  # Reduced mode should have exactly 3 steps


def test_tuning_reduction():
    """
    Test if tuning reduction properly reduces the scale while maintaining harmonicity.
    """
    tuning = [1, 1.21, 1.31, 1.45, 1.5, 1.7, 1.875]
    mode_n_steps = 5
    function = dyad_similarity
    tuning_consonance, mode, mode_consonance = tuning_reduction(
        tuning, mode_n_steps, function
    )

    # Ensure mode has the correct number of steps
    assert len(mode) == mode_n_steps

    # Check that consonance of reduced mode is within a reasonable range
    assert mode_consonance >= tuning_consonance


def test_tuning_reduction_metric_denom():
    """
    Test tuning reduction with metric_denom as the function.
    """
    tuning = [1, 1.21, 1.31, 1.45, 1.5, 1.7, 1.875]
    mode_n_steps = 4
    function = metric_denom
    tuning_consonance, mode, mode_consonance = tuning_reduction(
        tuning, mode_n_steps, function
    )

    assert len(mode) == mode_n_steps
    assert mode_consonance <= tuning_consonance


def test_dissmeasure():
    """
    Tests dissmeasure to ensure:
    - It computes a positive dissonance value
    - The value changes with different amplitude models ('min' vs. 'product')
    - It handles invalid model names properly
    """
    freqs = [200, 300, 400, 500]
    amps = [1.0, 0.8, 0.6, 0.5]

    # Compute dissonance with both models
    diss_min = dissmeasure(freqs, amps, model="min")
    diss_product = dissmeasure(freqs, amps, model="product")

    # Ensure dissonance is positive
    assert diss_min > 0, "Dissonance (min model) should be positive"
    assert diss_product > 0, "Dissonance (product model) should be positive"

    # Ensure model differences produce distinct values
    assert (
        diss_min != diss_product
    ), "Different models should yield different dissonance values"

    # Test invalid model name
    with pytest.raises(ValueError, match='model should be "min" or "product"'):
        dissmeasure(freqs, amps, model="invalid")


def test_diss_curve():
    """
    Tests diss_curve to ensure:
    - The dissonance curve is computed properly
    - Expected values are within a reasonable range
    - The function handles empty and extreme inputs correctly
    """
    freqs = [200, 300, 400, 500]
    amps = [1.0, 0.8, 0.6, 0.5]

    # Compute the dissonance curve
    diss, intervals, ratios, euler_score, avg_diss, dyad_sims = diss_curve(
        freqs, amps, denom=1000, max_ratio=2, euler_comp=True, method="min", plot=False
    )

    # Check expected output lengths
    assert len(diss) == 1000, "Dissonance curve should have 1000 points"
    assert len(intervals) > 0, "Intervals list should not be empty"
    assert len(ratios) > 0, "Ratios list should not be empty"

    # Check values are in reasonable ranges
    assert avg_diss > 0, "Average dissonance should be positive"
    assert (
        euler_score is None or euler_score > 0
    ), "Euler consonance score should be positive"

    # Edge case: Empty input
    empty_test, _, _, _, _, _ = diss_curve(
        [], [], denom=1000, max_ratio=2, euler_comp=True, method="min", plot=False
    )
    assert all(d == 0 for d in empty_test), "Empty input should yield all zeros"

    # Edge case: Extreme frequencies
    extreme_freqs = [20, 20000]  # Low and high frequencies
    extreme_test = diss_curve(
        extreme_freqs,
        [1.0, 0.8],
        denom=1000,
        max_ratio=2,
        euler_comp=False,
        method="min",
        plot=False,
    )
    assert (
        len(extreme_test[0]) > 0
    ), "Extreme frequency input should still generate a curve"


def test_harmonic_entropy():
    """
    Tests harmonic_entropy function to ensure:
    - Correct computation of entropy values
    - Expected number of local minima
    - Proper handling of edge cases (empty input, single ratio)
    """
    # Test with a simple harmonic series
    ratios = [1.0, 1.25, 1.333, 1.5, 1.666, 1.75, 2.0]
    HE_minima, HE_avg, HE = harmonic_entropy(
        ratios, res=0.001, spread=0.01, plot_entropy=False
    )

    # Check that entropy values are computed
    assert len(HE) > 0, "Harmonic entropy array should not be empty"
    assert HE_avg > 0, "Average harmonic entropy should be positive"

    # Ensure entropy minima are within valid range
    assert len(HE_minima[0]) > 0, "Expected at least one minimum in the entropy curve"
    assert all(
        1.0 <= x <= 2.0 for x in HE_minima[0]
    ), "Minima should be within the expected octave range"

    # Edge case: Empty input should return empty results
    empty_test = harmonic_entropy([], plot_entropy=False)
    assert len(empty_test[0][0]) == 0, "Entropy minimum should be empty for empty input"

    assert np.all(np.isfinite(HE)) and np.all(
        HE >= 0
    ), "Entropy values should be finite and non-negative"


def test_pac_mode():
    """
    Test if pac_mode extracts a subset of the scale based on phase-amplitude coupling.
    """
    pac_freqs = [(1, 1.5), (1.5, 1.875), (1, 1.7), (1.45, 1.875)]
    mode_n_steps = 4

    mode = pac_mode(pac_freqs, mode_n_steps, dyad_similarity, method="subset")

    # Ensure mode has the correct number of steps
    assert len(mode) == mode_n_steps

    # Ensure mode contains only valid frequency values
    extracted_values = set(mode)
    # expected_values = set([1, 1.5, 1.7, 1.45, 1.875])
    # assert extracted_values.issubset(expected_values)


def test_pac_mode_pairwise():
    """
    Test pac_mode with method 'pairwise' to ensure correct subset selection.
    """
    pac_freqs = [(1, 1.5), (1.5, 1.875), (1, 1.7), (1.45, 1.875)]
    mode_n_steps = 4

    mode = pac_mode(pac_freqs, mode_n_steps, dyad_similarity, method="pairwise")
    print("mode:", mode)
    # Ensure mode has the correct number of steps
    assert len(mode) == mode_n_steps

    # Ensure mode contains only valid frequency values
    # extracted_values = set(mode)
    # expected_values = set([1, 1.5, 1.7, 1.45, 1.875])
    # assert extracted_values.issubset(expected_values)


@pytest.mark.parametrize(
    "interval, steps",
    [
        (3 / 2, 12),
        (4 / 3, 10),
    ],
)
def test_tuning_MOS_info(interval, steps):
    n_gaps, large, small, tuning, distances = tuning_MOS_info(interval, steps)

    # Check tuning is sorted and unique
    assert list(tuning) == sorted(set(tuning))


@pytest.mark.parametrize(
    "generator_interval, max_steps, expected_max_deviation_range",
    [
        (3 / 2, 20, (0.0, 1.0)),  # Deviation should be between 0.0 and 1.0
        (4 / 3, 15, (0.0, 1.0)),  # Deviation should be between 0.0 and 1.0
    ],
)
def test_measure_symmetry(generator_interval, max_steps, expected_max_deviation_range):
    max_deviation = measure_symmetry(generator_interval, max_steps=max_steps)

    # Check max deviation is within the expected range
    assert (
        expected_max_deviation_range[0]
        <= max_deviation
        <= expected_max_deviation_range[1]
    )

def test_horogram_tree_steps():
    ratio1 = 1/2
    ratio2 = 2/3
    steps = 7
    fractions, ratios = horogram_tree_steps(ratio1, ratio2, steps=steps)

    assert len(fractions) == steps + 2  # Includes initial two ratios
    assert len(ratios) == steps + 2
    assert all([0 < r < 1 for r in ratios])

def test_horogram_tree():
    ratio1 = 1/2
    ratio2 = 2/3
    limit = 1000
    result = horogram_tree(ratio1, ratio2, limit)
    
    assert isinstance(result, float)
    assert 0 < result < 1  # Should remain within reasonable bounds

def test_phi_convergent_point():
    ratio1 = 1/2
    ratio2 = 2/3
    result = phi_convergent_point(ratio1, ratio2)
    
    assert isinstance(result, float)
    assert 0 < result < 1  # Should remain within a logical range

def test_Stern_Brocot():
    depths = [5, 10, 15]
    for depth in depths:
        result = Stern_Brocot(depth, a=0, b=1, c=1, d=1)
        
        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)
        assert len(result) > 0  # Should return some values
    
    # Test case where depth is lower than a + b + c + d
    assert Stern_Brocot(2, a=0, b=1, c=1, d=1) == 0

def test_interval_exponents():
    interval = 3/2
    n_steps = 5
    result = interval_exponents(interval, n_steps)
    
    assert len(result) == n_steps
    assert all(isinstance(r, float) for r in result)

def test_interval_to_radian():
    interval = 3/2
    rad, deg = interval_to_radian(interval)
    
    assert isinstance(rad, float)
    assert isinstance(deg, float)
    assert 0 <= deg <= 360  # Should be within a valid degree range

def test_tuning_to_radians():
    interval = 3/2
    n_steps = 12
    radians, degrees = tuning_to_radians(interval, n_steps)
    
    assert len(radians) == len(degrees) == n_steps
    assert all(isinstance(r, float) for r in radians)
    assert all(isinstance(d, float) for d in degrees)

def test_compare_oct_div():
    avg_ratios, shared_steps = compare_oct_div(Octdiv=12, Octdiv2=53, bounds=0.005, octave=2)

    assert isinstance(avg_ratios, list)
    assert isinstance(shared_steps, list)
    assert all(isinstance(r, float) for r in avg_ratios)
    assert all(isinstance(s, tuple) and len(s) == 2 for s in shared_steps)

def test_multi_oct_subdiv():
    peaks = [11, 24, 32, 44]
    oct_divs, ratios = multi_oct_subdiv(peaks, max_sub=100)

    assert isinstance(oct_divs, list)
    assert isinstance(ratios, np.ndarray)
    assert all(isinstance(o, (int, np.integer)) for o in oct_divs)
    assert all(isinstance(r, float) for r in ratios)

def test_convergents():
    ratio = 3 / 2
    result = convergents(ratio)

    assert isinstance(result, list)
    assert all(isinstance(c, tuple) and len(c) == 2 for c in result)

def test_MOS_metric_harmonic_mean():
    MOS_dict = {
        "steps": [12, 24, 36],
        "harmsim": [0.8, 0.6, 0.7]
    }
    result = MOS_metric_harmonic_mean(MOS_dict, metric="harmsim")

    assert isinstance(result, float)
    assert 0 <= result <= 1  # Harmonic similarity should be within a valid range

def test_generator_to_stern_brocot_fractions():
    gen = 3 / 2
    limit = 20
    result = generator_to_stern_brocot_fractions(gen, limit)

    assert isinstance(result, list)
    assert all(isinstance(f, tuple) and len(f) == 2 for f in result)
    assert all(isinstance(n, int) and isinstance(d, int) for f in result for n, d in [f])
