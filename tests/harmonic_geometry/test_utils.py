"""Tests for biotuner.harmonic_geometry._utils."""

from fractions import Fraction

import numpy as np
import pytest

from biotuner.harmonic_geometry._utils import (
    coerce_ratio,
    coerce_ratios,
    coprime_pair,
    equave_reduce,
    is_coprime,
    log_ratio,
    normalize_amplitudes,
    ratios_to_floats,
)


class TestCoerceRatio:
    def test_fraction_is_returned_unchanged(self):
        f = Fraction(3, 2)
        assert coerce_ratio(f) is f

    def test_int(self):
        assert coerce_ratio(5) == Fraction(5, 1)

    def test_tuple(self):
        assert coerce_ratio((3, 2)) == Fraction(3, 2)

    def test_tuple_wrong_length_raises(self):
        with pytest.raises(ValueError):
            coerce_ratio((1, 2, 3))

    def test_float_rational(self):
        assert coerce_ratio(1.5) == Fraction(3, 2)

    def test_float_with_max_denominator(self):
        # pi has no exact rational form; should be approximated.
        approx = coerce_ratio(np.pi, max_denominator=100)
        assert isinstance(approx, Fraction)
        assert approx.denominator <= 100
        assert abs(float(approx) - np.pi) < 0.05

    def test_non_finite_raises(self):
        with pytest.raises(ValueError):
            coerce_ratio(float("inf"))
        with pytest.raises(ValueError):
            coerce_ratio(float("nan"))

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            coerce_ratio("3/2")

    def test_coerce_ratios_list(self):
        out = coerce_ratios([1, 1.5, (5, 4)])
        assert out == [Fraction(1, 1), Fraction(3, 2), Fraction(5, 4)]


class TestRatiosToFloats:
    def test_mixed(self):
        arr = ratios_to_floats([Fraction(3, 2), 1.25, 2])
        assert arr.dtype == np.float64
        np.testing.assert_allclose(arr, [1.5, 1.25, 2.0])


class TestCoprime:
    def test_is_coprime(self):
        assert is_coprime(3, 2)
        assert not is_coprime(4, 2)
        assert is_coprime(7, 5)

    def test_zero_is_not_coprime(self):
        assert not is_coprime(0, 5)
        assert not is_coprime(5, 0)

    def test_coprime_pair(self):
        assert coprime_pair(Fraction(6, 4)) == (3, 2)
        assert coprime_pair(2) == (2, 1)
        assert coprime_pair(0.75) == (3, 4)


class TestLogRatio:
    def test_octave(self):
        assert log_ratio(2.0, equave=2.0) == pytest.approx(1.0)

    def test_unison(self):
        assert log_ratio(1.0, equave=2.0) == pytest.approx(0.0)

    def test_tritave(self):
        assert log_ratio(3.0, equave=3.0) == pytest.approx(1.0)

    def test_invalid_equave(self):
        with pytest.raises(ValueError):
            log_ratio(2.0, equave=1.0)

    def test_non_positive_ratio(self):
        with pytest.raises(ValueError):
            log_ratio(-1.0)


class TestEquaveReduce:
    def test_already_in_range(self):
        assert equave_reduce(Fraction(3, 2)) == Fraction(3, 2)

    def test_above_octave(self):
        assert equave_reduce(Fraction(3, 1)) == Fraction(3, 2)

    def test_below_unison(self):
        # 1/3 reduces upward to 4/3 (multiply by 2 twice would overshoot;
        # multiply once -> 2/3, still below unison; twice -> 4/3, in range).
        assert equave_reduce(Fraction(1, 3)) == Fraction(4, 3)

    def test_float_path(self):
        out = equave_reduce(7.0, equave=2.0)
        assert 1.0 <= out < 2.0
        # 7 = 2^2 * 1.75, so reduced should be 1.75.
        assert out == pytest.approx(1.75)

    def test_irrational_equave_uses_float_path(self):
        # phi-equave: 7/4 should still land in [1, phi).
        phi = (1 + np.sqrt(5)) / 2
        out = equave_reduce(7.0 / 4.0, equave=phi)
        assert 1.0 <= out < phi


class TestNormalizeAmplitudes:
    def test_sums_to_one(self):
        out = normalize_amplitudes([1.0, 2.0, 3.0])
        assert out.sum() == pytest.approx(1.0)
        np.testing.assert_allclose(out, [1 / 6, 2 / 6, 3 / 6])

    def test_zero_sum_returns_uniform(self):
        out = normalize_amplitudes([0.0, 0.0, 0.0])
        np.testing.assert_allclose(out, [1 / 3, 1 / 3, 1 / 3])

    def test_empty(self):
        out = normalize_amplitudes([])
        assert out.shape == (0,)
