"""Fixtures for harmonic_timbre tests."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def simple_just_tuning() -> list[float]:
    """Just-intonation diatonic — 8 ratios with octave."""
    return [1.0, 9 / 8, 5 / 4, 4 / 3, 3 / 2, 5 / 3, 15 / 8, 2.0]


@pytest.fixture
def bohlen_pierce_tuning() -> list[float]:
    """Tritave-based 13-tone scale (uses equave=3.0)."""
    return [1.0, 27 / 25, 25 / 21, 9 / 7, 7 / 5, 75 / 49, 5 / 3, 9 / 5,
            49 / 25, 15 / 7, 7 / 3, 63 / 25, 25 / 9, 3.0]


@pytest.fixture
def slendro_tuning() -> list[float]:
    """Approximate slendro pentatonic (gamelan)."""
    return [1.0, 1.146, 1.318, 1.516, 1.741, 2.0]


@pytest.fixture
def eeg_like_peaks() -> tuple[list[float], list[float]]:
    """EEG-like peak set: alpha/beta/gamma frequencies and amplitudes."""
    peaks = [2.5, 8.0, 12.5, 25.0, 40.0]
    amps = [0.6, 1.0, 0.7, 0.4, 0.2]
    return peaks, amps


class _MockBiotunerV1:
    """A mock compute_biotuner with v1 fields populated."""

    def __init__(
        self,
        peaks=(2.5, 8.0, 12.5, 25.0, 40.0),
        amps=(0.6, 1.0, 0.7, 0.4, 0.2),
        with_phases=True,
        with_fooof=True,
    ):
        self.peaks = list(peaks)
        self.amps = list(amps)
        # peaks_ratios = peaks / min(peaks), kept inside [1, 2)
        f0 = float(min(peaks))
        ratios: list[float] = []
        for p in peaks:
            r = p / f0
            while r >= 2.0:
                r /= 2.0
            ratios.append(r)
        self.peaks_ratios = sorted(set(ratios))
        # peaks_ratios_cons: subset of peaks_ratios that are most consonant
        self.peaks_ratios_cons = self.peaks_ratios[: max(1, len(self.peaks_ratios) - 1)]
        self.extended_peaks_ratios = self.peaks_ratios + [r * 2.0 for r in self.peaks_ratios]
        self.extended_peaks_ratios_cons = self.peaks_ratios_cons
        if with_phases:
            self.phases = [0.0, np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2][: len(peaks)]
        if with_fooof:
            self.linewidth = [0.5, 1.0, 1.2, 1.5, 1.8][: len(peaks)]
            self.aperiodic_exponent = 1.5
            self.spectral_flatness = 0.2


@pytest.fixture
def mock_biotuner_v1() -> _MockBiotunerV1:
    return _MockBiotunerV1()


@pytest.fixture
def mock_biotuner_minimal() -> _MockBiotunerV1:
    """Mock with only peaks/amps/ratios, no FOOOF, no phases."""
    return _MockBiotunerV1(with_phases=False, with_fooof=False)
