"""Tests for biotuner.harmonic_geometry.media.coupling — chord-shape reductions."""

from fractions import Fraction

import pytest

from biotuner.harmonic_geometry import HarmonicInput
from biotuner.harmonic_geometry.media import coupling


@pytest.fixture
def major():
    return HarmonicInput(
        ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)],
        amplitudes=[1.0, 0.7, 0.8],
    )


@pytest.fixture
def dissonant():
    return HarmonicInput(
        ratios=[Fraction(1), Fraction(11, 8), Fraction(17, 13)],
        amplitudes=[1.0, 1.0, 1.0],
    )


@pytest.fixture
def unison():
    return HarmonicInput(ratios=[Fraction(1)])


class TestConsonance:
    def test_in_unit_interval(self, major):
        c = coupling.consonance(major)
        assert 0.0 <= c <= 1.0

    def test_major_more_consonant_than_dissonant(self, major, dissonant):
        assert coupling.consonance(major) > coupling.consonance(dissonant)

    def test_unison_is_max(self, unison):
        assert coupling.consonance(unison) == 1.0


class TestRatioComplexity:
    def test_lower_bound(self, unison):
        assert coupling.ratio_complexity(unison) >= 1.0

    def test_dissonant_higher_than_simple(self, major, dissonant):
        assert coupling.ratio_complexity(dissonant) > coupling.ratio_complexity(major)


class TestSpectralSpread:
    def test_single_peak_is_zero(self, unison):
        assert coupling.spectral_spread(unison) == 0.0

    def test_nonnegative(self, major):
        assert coupling.spectral_spread(major) >= 0.0


class TestAmplitudeEntropy:
    def test_in_unit_interval(self, major):
        e = coupling.amplitude_entropy(major)
        assert 0.0 <= e <= 1.0

    def test_uniform_amplitudes_max(self, dissonant):
        # dissonant fixture has uniform amplitudes [1, 1, 1].
        assert coupling.amplitude_entropy(dissonant) == pytest.approx(1.0)

    def test_spike_low(self):
        chord = HarmonicInput(
            ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)],
            amplitudes=[1.0, 1e-6, 1e-6],
        )
        assert coupling.amplitude_entropy(chord) < 0.2
