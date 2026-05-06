"""Fixtures for the harmonic_timbre exporter tests."""

from __future__ import annotations

import pytest

from biotuner.harmonic_timbre import match_consonance_weighted


@pytest.fixture
def matched_timbre():
    """A matched timbre on a JI scale, ready for any exporter."""
    ratios = [1.0, 9 / 8, 5 / 4, 4 / 3, 3 / 2, 5 / 3, 15 / 8, 2.0]
    return match_consonance_weighted(ratios, n_partials=8, base_freq=220.0)


@pytest.fixture
def short_pitches():
    """Three reference pitches — keeps WAV pack tests fast."""
    return [48, 60, 72]
