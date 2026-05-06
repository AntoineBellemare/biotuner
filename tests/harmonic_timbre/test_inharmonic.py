"""Tests for biotuner.harmonic_timbre.inharmonic."""

from __future__ import annotations

import numpy as np
import pytest

from biotuner.harmonic_timbre import (
    stretched_partials,
    compressed_partials,
    inharmonic_string,
    gamelan_partials,
    custom_partial_series,
    inharmonic_timbre,
)


def test_stretched_partials_shape():
    p = stretched_partials(8, stretch=1.05, base_freq=100.0)
    assert p.shape == (8,)
    assert p[0] == 100.0
    # monotonically increasing
    assert np.all(np.diff(p) > 0)


def test_compressed_partials_grows_more_slowly_than_harmonic():
    n = 8
    base = 100.0
    p = compressed_partials(n, compress=0.95, base_freq=base)
    harmonic = base * np.arange(1, n + 1)
    # compressed: by partial 8, should be lower than the 8th harmonic
    assert p[-1] < harmonic[-1]


def test_inharmonic_string_lorentzian_shape():
    # B = 1e-4 is piano-like; partial k frequencies = k*f0*sqrt(1+B*k^2)
    p = inharmonic_string(10, B=1e-4, base_freq=100.0)
    expected = np.arange(1, 11) * 100.0 * np.sqrt(1.0 + 1e-4 * np.arange(1, 11) ** 2)
    np.testing.assert_allclose(p, expected, rtol=1e-9)


def test_inharmonic_string_zero_B_recovers_harmonic():
    p = inharmonic_string(8, B=0.0, base_freq=100.0)
    np.testing.assert_allclose(p, np.arange(1, 9) * 100.0)


def test_gamelan_partials_known_profiles():
    p = gamelan_partials(instrument="saron", base_freq=200.0)
    assert p.shape == (5,)
    # ratios encoded in the profile: [1.00, 2.76, 5.40, 5.91, 6.99]
    expected = np.asarray([1.00, 2.76, 5.40, 5.91, 6.99]) * 200.0
    np.testing.assert_allclose(p, expected)


def test_gamelan_partials_unknown_raises():
    with pytest.raises(ValueError, match="unknown instrument"):
        gamelan_partials(instrument="cowbell")


def test_custom_partial_series():
    p = custom_partial_series([1.0, 1.5, 2.5], base_freq=100.0)
    np.testing.assert_allclose(p, [100.0, 150.0, 250.0])


def test_custom_partial_series_rejects_negative():
    with pytest.raises(ValueError):
        custom_partial_series([1.0, -2.0])


def test_inharmonic_timbre_via_string():
    t = inharmonic_timbre(
        inharmonic_string,
        n=6,
        base_freq=100.0,
        fn_kwargs={"B": 1e-4},
    )
    assert t.n_partials() == 6
    assert t.matching_method == "inharmonic"
    assert t.metadata["partial_series_fn"] == "inharmonic_string"


def test_inharmonic_timbre_via_gamelan():
    t = inharmonic_timbre(
        gamelan_partials,
        base_freq=200.0,
        fn_kwargs={"instrument": "bonang"},
    )
    # bonang profile has 4 partials
    assert t.n_partials() == 4


def test_railsback_piano_stretch_within_tolerance():
    """Empirical check: a 1.05-stretched series matches piano measurements
    in the rough monotonic-trend sense — partials get progressively
    sharper than equal temperament."""
    # 88-key piano: ~7 octaves. Take 30 stretched partials.
    p = stretched_partials(30, stretch=1.0005, base_freq=27.5)  # A0 ≈ 27.5 Hz
    # Stretched partials grow slightly faster than integer harmonics.
    harmonic = np.arange(1, 31) * 27.5
    deviation = (p - harmonic) / harmonic
    # deviation is monotonically increasing (more stretch on higher partials)
    assert np.all(np.diff(deviation) >= 0)
