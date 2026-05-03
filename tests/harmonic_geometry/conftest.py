"""Shared fixtures for the harmonic_geometry test suite."""

from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def simple_ratios():
    """A basic just-intonation tetrad: 1, 3/2, 5/4, 7/4."""
    return [Fraction(1, 1), Fraction(3, 2), Fraction(5, 4), Fraction(7, 4)]


@pytest.fixture
def eeg_like_input_kwargs():
    """Constructor kwargs for an EEG-like HarmonicInput.

    Returned as a dict so individual tests can splice in / override fields
    without sharing mutable state.
    """
    return {
        "peaks": [2.5, 8.0, 12.5, 25.0],
        "amplitudes": [0.4, 1.0, 0.7, 0.3],
        "phases": [0.0, np.pi / 4, np.pi / 2, np.pi],
        "base_freq": 2.5,
    }


@pytest.fixture
def mock_biotuner_obj():
    """Minimal duck-typed substitute for a fitted ``compute_biotuner``.

    Carries ``peaks``, ``amps``, and ``peaks_ratios`` arrays sized to match
    each other, mirroring what biotuner's ``peaks_extraction`` produces.
    """
    peaks = np.array([4.0, 6.0, 8.0, 10.0])
    return SimpleNamespace(
        peaks=peaks,
        amps=np.array([1.0, 0.8, 0.6, 0.4]),
        peaks_ratios=peaks / peaks.min(),
    )


@pytest.fixture
def mock_biotuner_obj_no_ratios():
    """A biotuner-like object missing peaks_ratios (older / partial fits)."""
    return SimpleNamespace(
        peaks=np.array([3.0, 5.0, 7.0]),
        amps=np.array([1.0, 0.5, 0.25]),
    )


@pytest.fixture
def mock_biotuner_group(mock_biotuner_obj, mock_biotuner_obj_no_ratios):
    """A duck-typed BiotunerGroup with two stored objects."""
    return SimpleNamespace(
        objects=[mock_biotuner_obj, mock_biotuner_obj_no_ratios],
    )
