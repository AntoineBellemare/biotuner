"""Tests for biotuner.harmonic_spectrum (slim, H-only module).

After the resonance refactor this module retains only the harmonic-spectrum
machinery: H(f) computation, the kernel similarity matrix, peak detection,
and the 3-spectrum complexity DataFrame helper.

Phase-coupling and resonance-spectrum tests moved to ``tests/resonance/``.

Sections:
  1. Matrix builder            — harmonicity_matrices
  2. Per-frequency H spectrum  — compute_harmonic_power, compute_harmonic_spectrum
  3. Peak detection            — find_spectral_peaks
  4. Complexity helpers        — harmonic_entropy, spectrum_complexity
  5. Ratio + bandwidth utils   — get_harmonic_ratio, count_theoretical_harmonic_partners
"""
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use("Agg")

from biotuner.harmonic_spectrum import (
    compute_harmonic_power,
    compute_harmonic_spectrum,
    count_theoretical_harmonic_partners,
    find_spectral_peaks,
    get_harmonic_ratio,
    harmonic_entropy,
    harmonicity_matrices,
)
from biotuner.metrics import spectrum_complexity


# ─── shared synthetic signal ────────────────────────────────────────────────


@pytest.fixture(scope="module")
def synthetic_signal():
    """4-second harmonic sine-wave bundle at 1000 Hz sampling."""
    sf = 1000
    duration = 4.0
    t = np.linspace(0, duration, int(sf * duration), endpoint=False)
    rng = np.random.default_rng(0)
    sig = sum(
        (1.0 / (i + 1)) * np.sin(2 * np.pi * f * t)
        for i, f in enumerate([5, 10, 20, 40])
    )
    sig += 0.02 * rng.standard_normal(len(t))
    return sig.astype(np.float64), sf


# ─── 1. Matrix builder ──────────────────────────────────────────────────────


def test_harmonicity_matrices_shape():
    freqs = np.linspace(1, 10, 8)
    M = harmonicity_matrices(freqs, metric="harmsim", n_harms=5)
    assert M.shape == (8, 8)


def test_harmonicity_matrices_diagonal_is_self_similarity():
    from biotuner.metrics import dyad_similarity
    freqs = np.array([2.0, 3.0, 5.0])
    M = harmonicity_matrices(freqs, metric="harmsim")
    expected = dyad_similarity(1.0)
    for i in range(len(freqs)):
        assert np.isclose(M[i, i], expected)


def test_harmonicity_matrices_handles_full_positive_freqs():
    freqs = np.linspace(1.0, 30.0, 10)
    M = harmonicity_matrices(freqs)
    assert M.shape == (10, 10)
    assert np.all(np.isfinite(M))


def test_harmonicity_matrices_deterministic():
    freqs = np.linspace(1, 30, 12)
    M1 = harmonicity_matrices(freqs)
    M2 = harmonicity_matrices(freqs)
    assert np.array_equal(M1, M2)


# ─── 2. Per-frequency H spectrum ────────────────────────────────────────────


def test_compute_harmonic_power_shapes():
    rng = np.random.default_rng(0)
    F = 6
    freqs = np.linspace(1, 30, F)
    dyad = rng.uniform(0, 1, size=(F, F))
    psd = rng.uniform(0.1, 1.0, size=F)
    H_vec, H_mat = compute_harmonic_power(freqs, dyad, psd, normalize=True)
    assert H_vec.shape == (F,)
    assert H_mat.shape == (F, F)


def test_compute_harmonic_power_diagonal_zero():
    F = 5
    freqs = np.linspace(1, 10, F)
    dyad = np.ones((F, F))
    psd = np.ones(F)
    _, H_mat = compute_harmonic_power(freqs, dyad, psd, normalize=True)
    assert np.allclose(np.diag(H_mat), 0.0)


def test_compute_harmonic_power_uses_probability_weighting():
    F = 4
    freqs = np.arange(1.0, F + 1)
    rng = np.random.default_rng(0)
    dyad = rng.uniform(0, 1, size=(F, F))
    psd = np.array([1.0, 2.0, 3.0, 4.0])
    H_a, _ = compute_harmonic_power(freqs, dyad, psd, normalize=True)
    H_b, _ = compute_harmonic_power(freqs, dyad, psd * 10.0, normalize=True)
    assert np.allclose(H_a, H_b)


def test_compute_harmonic_power_vectorised_equivalence():
    rng = np.random.default_rng(7)
    F = 6
    freqs = np.linspace(1, 20, F)
    dyad = rng.uniform(0, 1, size=(F, F))
    psd = rng.uniform(0.1, 1, size=F)
    H_vec, H_mat = compute_harmonic_power(freqs, dyad, psd, normalize=True)

    p = psd / psd.sum()
    PP = np.outer(p, p)
    np.fill_diagonal(PP, 0.0)
    expected_mat = dyad * PP
    expected_vec = p * (dyad @ p) - np.diag(dyad) * (p ** 2)
    assert np.allclose(H_mat, expected_mat)
    assert np.allclose(H_vec, expected_vec)


def test_compute_harmonic_spectrum_returns_full_tuple(synthetic_signal):
    """New H-only entry point returns (freqs, H, matrix, summary)."""
    sig, sf = synthetic_signal
    freqs, H, M, summary = compute_harmonic_spectrum(
        sig, precision_hz=0.5, fmin=2, fmax=30, fs=sf, n_peaks=3,
    )
    assert freqs.ndim == 1
    assert H.shape == freqs.shape
    assert M.shape == (freqs.size, freqs.size)
    assert isinstance(summary, dict)
    for key in ("flatness", "entropy", "spread", "higuchi", "peaks", "avg", "max"):
        assert key in summary


def test_compute_harmonic_spectrum_summary_finite(synthetic_signal):
    sig, sf = synthetic_signal
    _, _, _, summary = compute_harmonic_spectrum(
        sig, precision_hz=1.0, fmin=2, fmax=20, fs=sf, n_peaks=3,
    )
    for key in ("flatness", "entropy", "spread", "higuchi", "avg", "max"):
        assert np.isfinite(summary[key])


def test_compute_harmonic_spectrum_dispatches_kernel(synthetic_signal):
    """Unknown kernel name raises ValueError listing registered options."""
    sig, sf = synthetic_signal
    with pytest.raises(ValueError, match="harmonic_kernel"):
        compute_harmonic_spectrum(
            sig, precision_hz=1.0, fmin=2, fmax=20, fs=sf, harmonic_kernel="not_a_real_kernel",
        )


# ─── 3. Peak detection ──────────────────────────────────────────────────────


def test_find_spectral_peaks_returns_at_most_n_peaks():
    values = np.array([0.0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0])
    freqs = np.arange(10, 121, 10)
    pf, idx = find_spectral_peaks(values, freqs, n_peaks=3, prominence_threshold=0.5)
    assert len(pf) <= 3
    assert len(idx) == len(pf)


def test_find_spectral_peaks_picks_correct_locations():
    values = np.array([0.0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0])
    freqs = np.arange(10, 121, 10)
    pf, _ = find_spectral_peaks(values, freqs, n_peaks=3, prominence_threshold=0.5)
    assert 60 in pf
    assert 40 in pf or 80 in pf


def test_find_spectral_peaks_prominence_threshold_filters():
    values = np.array([0.0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0])
    freqs = np.arange(10, 121, 10)
    pf_low, _ = find_spectral_peaks(values, freqs, n_peaks=10, prominence_threshold=0.5)
    pf_high, _ = find_spectral_peaks(values, freqs, n_peaks=10, prominence_threshold=2.5)
    assert len(pf_high) <= len(pf_low)


def test_find_spectral_peaks_no_peaks_above_threshold():
    values = np.zeros(10)
    freqs = np.arange(10)
    pf, idx = find_spectral_peaks(values, freqs, n_peaks=5, prominence_threshold=0.5)
    assert len(pf) == 0
    assert len(idx) == 0


# ─── 4. Complexity helpers ──────────────────────────────────────────────────


def test_harmonic_entropy_returns_dataframe_shape():
    freqs = np.linspace(1, 30, 60)
    rng = np.random.default_rng(0)
    h = rng.uniform(0, 1, size=60)
    p = rng.uniform(0, 1, size=60)
    r = h * p
    df = harmonic_entropy(freqs, h, p, r)
    assert isinstance(df, pd.DataFrame)
    assert list(df.index) == ["Harmonicity", "Phase Coupling", "Resonance"]
    expected_cols = {"Spectral Flatness", "Spectral Entropy",
                     "Spectral Spread", "Higuchi Fractal Dimension"}
    assert expected_cols.issubset(set(df.columns))


def test_spectrum_complexity_keys():
    rng = np.random.default_rng(0)
    values = rng.uniform(0.1, 1, size=40)
    freqs = np.linspace(1, 30, 40)
    summary = spectrum_complexity(values, freqs, n_peaks=3)
    expected_keys = {
        "flatness", "entropy", "spread", "higuchi",
        "peaks", "peak_indices", "avg", "max", "peaks_avg",
        "peak_harmsim", "peak_harmsim_avg", "peak_harmsim_max",
    }
    assert expected_keys.issubset(set(summary.keys()))
    for k in ("flatness", "entropy", "spread", "higuchi", "avg", "max"):
        assert np.isfinite(summary[k])


# ─── 5. Ratio + bandwidth utils ─────────────────────────────────────────────


def test_get_harmonic_ratio_exact_match():
    """An exact 1:2 ratio is detected within 5% tolerance."""
    out = get_harmonic_ratio(10.0, 20.0, max_n=3, max_m=3, tolerance=0.05)
    assert out == (1, 2)


def test_get_harmonic_ratio_no_match_returns_none():
    """A non-integer ratio (e.g. √2 = 1.414) doesn't match any low n:m within 5%."""
    out = get_harmonic_ratio(10.0, 14.142, max_n=3, max_m=3, tolerance=0.001)
    assert out is None


def test_count_theoretical_harmonic_partners_returns_nonneg_int():
    """The partner count is a non-negative integer for any positive freq."""
    for f in (1.0, 5.0, 20.0):
        n = count_theoretical_harmonic_partners(f, fmin=1, fmax=30, max_ratio=5)
        assert isinstance(n, int)
        assert n >= 0
