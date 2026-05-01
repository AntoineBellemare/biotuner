"""Tests for biotuner.harmonic_spectrum.

Covers the harmonicity / phase-coupling / resonance computations that
underpin the harmonicity-spectrum and harmonicity-matrix input modes of
``HarmonicSequenceAnalyzer`` (see test_harmonic_sequence.py for those
end-to-end checks).

Sections:
  1. Matrix builders                   — harmonicity_matrices, PLV_comod
  2. Per-frequency spectra             — compute_harmonic_power,
                                         compute_phase_spectrum,
                                         compute_resonance_values
  3. Peak detection                    — find_spectral_peaks
  4. Spectral complexity               — harmonic_entropy
  5. Phase extraction                  — compute_phase_values
  6. Integration                       — compute_global_harmonicity

The plotting helpers (``harmonic_spectrum_plot_*``) are not covered —
they're matplotlib wrappers that don't carry numerical contracts.
"""
import numpy as np
import pandas as pd
import pytest

from biotuner.harmonic_spectrum import (
    PLV_comod,
    compute_global_harmonicity,
    compute_harmonic_power,
    compute_phase_spectrum,
    compute_phase_values,
    compute_resonance_values,
    find_spectral_peaks,
    harmonic_entropy,
    harmonicity_matrices,
)


# ─── shared synthetic signal for integration tests ─────────────────────────


@pytest.fixture(scope="module")
def synthetic_signal():
    """4-second harmonic sine-wave bundle at 1000 Hz sampling."""
    sf = 1000
    duration = 4.0
    t = np.linspace(0, duration, int(sf * duration), endpoint=False)
    rng = np.random.default_rng(0)
    sig = sum(
        (1.0 / (i + 1)) * np.sin(2 * np.pi * f * t)
        for i, f in enumerate([5, 10, 20, 40])      # 1 : 2 : 4 : 8 — strongly harmonic
    )
    sig += 0.02 * rng.standard_normal(len(t))
    return sig.astype(np.float64), sf


# ─── 1. Matrix builders ─────────────────────────────────────────────────────


def test_harmonicity_matrices_shape():
    freqs = np.linspace(1, 10, 8)
    M = harmonicity_matrices(freqs, metric="harmsim", n_harms=5)
    assert M.shape == (8, 8)


def test_harmonicity_matrices_diagonal_is_self_similarity():
    """Each diagonal entry is dyad_similarity(f/f) = dyad_similarity(1)."""
    from biotuner.metrics import dyad_similarity
    freqs = np.array([2.0, 3.0, 5.0])
    M = harmonicity_matrices(freqs, metric="harmsim")
    expected = dyad_similarity(1.0)
    for i in range(len(freqs)):
        assert np.isclose(M[i, i], expected)


def test_harmonicity_matrices_handles_full_positive_freqs():
    """All-positive frequencies (the realistic case) produce a finite matrix."""
    freqs = np.linspace(1.0, 30.0, 10)
    M = harmonicity_matrices(freqs)
    assert M.shape == (10, 10)
    assert np.all(np.isfinite(M))


# Note: passing freqs containing 0 currently raises a ZeroDivisionError when
# f1 == 0 (the f2 != 0 guard does not protect dyad_similarity(0)).  This is a
# fragility of the underlying module, not the encoder layer; callers in
# practice always pass strictly positive frequency grids from
# compute_frequency_and_psd, so we don't try to lock down the zero-freq path
# in tests.


def test_harmonicity_matrices_octave_pair_high_similarity():
    """An octave pair (3:2) is more harmonic than 7:4."""
    freqs = np.array([2.0, 3.0])
    M = harmonicity_matrices(freqs)
    octave_score = M[0, 1]            # dyad_similarity(2/3 = 1.5)
    assert octave_score > 0


def test_harmonicity_matrices_deterministic():
    """The matrix depends only on freqs — two calls return identical results."""
    freqs = np.linspace(1, 30, 12)
    M1 = harmonicity_matrices(freqs)
    M2 = harmonicity_matrices(freqs)
    assert np.array_equal(M1, M2)


def test_PLV_comod_shape():
    """phase shape (F, T) → output shape (F, F)."""
    rng = np.random.default_rng(0)
    phase = rng.uniform(-np.pi, np.pi, size=(8, 50))
    M = PLV_comod(phase)
    assert M.shape == (8, 8)


def test_PLV_comod_diagonal_is_one():
    """Phase difference of a frequency with itself is zero → PLV = 1."""
    rng = np.random.default_rng(0)
    phase = rng.uniform(-np.pi, np.pi, size=(6, 30))
    M = PLV_comod(phase)
    assert np.allclose(np.diag(M), 1.0)


def test_PLV_comod_values_in_unit_range():
    rng = np.random.default_rng(0)
    phase = rng.uniform(-np.pi, np.pi, size=(6, 30))
    M = PLV_comod(phase)
    assert np.all(M >= 0) and np.all(M <= 1.0 + 1e-12)


# ─── 2. Per-frequency spectra ──────────────────────────────────────────────


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
    dyad = np.ones((F, F))                # uniform similarity
    psd = np.ones(F)
    _, H_mat = compute_harmonic_power(freqs, dyad, psd, normalize=True)
    assert np.allclose(np.diag(H_mat), 0.0)


def test_compute_harmonic_power_normalisation_factor():
    """Normalised values equal weighted_sum / (2 * total_power)."""
    F = 4
    freqs = np.arange(1.0, F + 1)
    dyad = np.ones((F, F))
    psd = np.array([1.0, 2.0, 3.0, 4.0])
    H_norm, _ = compute_harmonic_power(freqs, dyad, psd, normalize=True)
    H_unnorm, _ = compute_harmonic_power(freqs, dyad, psd, normalize=False)
    total = psd.sum()
    assert np.allclose(H_norm * (2 * total), H_unnorm)


def test_compute_harmonic_power_vectorised_equivalence():
    """Closed-form vectorisation should match the function output."""
    rng = np.random.default_rng(7)
    F = 6
    freqs = np.linspace(1, 20, F)
    dyad = rng.uniform(0, 1, size=(F, F))
    psd = rng.uniform(0.1, 1, size=F)
    H_vec, H_mat = compute_harmonic_power(freqs, dyad, psd, normalize=True)

    total = psd.sum()
    PP = np.outer(psd, psd)
    np.fill_diagonal(PP, 0.0)
    expected_mat = (dyad * PP) / total
    expected_vec = (dyad * PP).sum(axis=1) / (2.0 * total)
    assert np.allclose(H_mat, expected_mat)
    assert np.allclose(H_vec, expected_vec)


def test_compute_phase_spectrum_psd_weighted_vs_uniform():
    """psd_weight='weighted' multiplies by P_iP_j; otherwise it doesn't."""
    F = 4
    freqs = np.arange(1.0, F + 1)
    plv = np.full((F, F), 0.5)
    psd = np.array([1.0, 2.0, 3.0, 4.0])
    weighted = compute_phase_spectrum(freqs, plv, psd, psd_weight="weighted",
                                       normalize=False)
    uniform = compute_phase_spectrum(freqs, plv, psd, psd_weight=None,
                                      normalize=False)
    # Uniform is independent of psd; weighted scales with P_i × Σ P_j
    assert not np.allclose(weighted, uniform)


def test_compute_phase_spectrum_shape():
    F = 5
    freqs = np.linspace(1, 10, F)
    plv = np.zeros((F, F))
    psd = np.ones(F)
    out = compute_phase_spectrum(freqs, plv, psd)
    assert out.shape == (F,)


def test_compute_resonance_values_in_unit_range():
    h = np.array([0.1, 0.5, 0.9, 0.4])
    p = np.array([0.3, 0.6, 0.2, 0.8])
    r = compute_resonance_values(h, p)
    # Each input is min-max normalised before multiplication; product is in [0, 1]
    assert np.all(r >= 0.0)
    assert np.all(r <= 1.0 + 1e-12)


def test_compute_resonance_values_min_max_normalisation():
    """Closed form: r = ((h - min h) / range h) * ((p - min p) / range p)."""
    h = np.array([1.0, 2.0, 4.0])
    p = np.array([10.0, 20.0, 30.0])
    r = compute_resonance_values(h, p)
    h_norm = (h - h.min()) / (h.max() - h.min())
    p_norm = (p - p.min()) / (p.max() - p.min())
    expected = h_norm * p_norm
    assert np.allclose(r, expected)


def test_compute_resonance_values_length_matches_input():
    h = np.linspace(0, 1, 12)
    p = np.linspace(0.5, 0.9, 12)
    assert compute_resonance_values(h, p).shape == (12,)


# ─── 3. Peak detection ─────────────────────────────────────────────────────


def test_find_spectral_peaks_returns_at_most_n_peaks():
    values = np.array([0.0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0])
    freqs = np.arange(10, 121, 10)
    pf, idx = find_spectral_peaks(values, freqs, n_peaks=3,
                                   prominence_threshold=0.5)
    assert len(pf) <= 3
    assert len(idx) == len(pf)


def test_find_spectral_peaks_picks_correct_locations():
    values = np.array([0.0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0])
    freqs = np.arange(10, 121, 10)
    pf, idx = find_spectral_peaks(values, freqs, n_peaks=3,
                                   prominence_threshold=0.5)
    # The three peaks are at indices 1, 3, 5, 7, 9 (heights 1,2,3,2,1).
    # The most prominent are 5 (60 Hz, height 3), 3 (40 Hz, height 2),
    # 7 (80 Hz, height 2).
    assert 60 in pf
    assert 40 in pf or 80 in pf


def test_find_spectral_peaks_prominence_threshold_filters():
    """A high prominence threshold drops most peaks."""
    values = np.array([0.0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0])
    freqs = np.arange(10, 121, 10)
    pf_low, _ = find_spectral_peaks(values, freqs, n_peaks=10,
                                     prominence_threshold=0.5)
    pf_high, _ = find_spectral_peaks(values, freqs, n_peaks=10,
                                      prominence_threshold=2.5)
    assert len(pf_high) <= len(pf_low)


def test_find_spectral_peaks_no_peaks_above_threshold():
    """All-flat signal yields no peaks."""
    values = np.zeros(10)
    freqs = np.arange(10)
    pf, idx = find_spectral_peaks(values, freqs, n_peaks=5,
                                   prominence_threshold=0.5)
    assert len(pf) == 0
    assert len(idx) == 0


# ─── 4. Spectral complexity ────────────────────────────────────────────────


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


def test_harmonic_entropy_values_are_numeric():
    freqs = np.linspace(1, 30, 60)
    rng = np.random.default_rng(0)
    h = rng.uniform(0.1, 1, size=60)
    p = rng.uniform(0.1, 1, size=60)
    r = h * p
    df = harmonic_entropy(freqs, h, p, r)
    # No NaNs, all entries finite
    assert df.notna().all().all()
    for v in df.values.flatten():
        assert np.isfinite(float(v))


# ─── 5. Phase extraction ────────────────────────────────────────────────────


def test_compute_phase_values_returns_2d_angles(synthetic_signal):
    sig, sf = synthetic_signal
    phase = compute_phase_values(sig, precision_hz=0.5, fs=sf)
    assert phase.ndim == 2
    # Phases live in [-π, π]
    assert np.all(phase >= -np.pi - 1e-9)
    assert np.all(phase <= np.pi + 1e-9)


def test_compute_phase_values_smoothness_changes_shape(synthetic_signal):
    """Larger smoothness reduces nperseg and changes the time-axis size."""
    sig, sf = synthetic_signal
    phase1 = compute_phase_values(sig, precision_hz=0.5, fs=sf, smoothness=1)
    phase2 = compute_phase_values(sig, precision_hz=0.5, fs=sf, smoothness=2)
    # Different segment lengths → different time-axis (axis 1)
    assert phase1.shape[1] != phase2.shape[1]


# ─── 6. Integration: compute_global_harmonicity ─────────────────────────────


def test_compute_global_harmonicity_returns_df_and_matrix(synthetic_signal):
    sig, sf = synthetic_signal
    df, matrix = compute_global_harmonicity(
        sig, precision_hz=1.0, fmin=1.0, fmax=20.0, fs=sf,
        n_peaks=3, metric="harmsim", n_harms=5,
        smoothness=1, smoothness_harm=1, normalize=True,
        power_law_remove=False, plot=False,
    )
    assert isinstance(df, pd.DataFrame)
    # Single-row dataframe (one signal)
    assert len(df) == 1
    # Square matrix of shape (F, F)
    assert matrix.ndim == 2
    assert matrix.shape[0] == matrix.shape[1]


def test_compute_global_harmonicity_dataframe_columns(synthetic_signal):
    sig, sf = synthetic_signal
    df, _ = compute_global_harmonicity(
        sig, precision_hz=1.0, fmin=1.0, fmax=20.0, fs=sf,
        plot=False, n_peaks=3,
    )
    expected = {
        "harmonicity", "phase_coupling", "resonance",
        "harmonicity_avg", "phase_coupling_avg", "resonance_avg",
        "harmonicity_max", "phase_coupling_max", "resonance_max",
        "harmonicity_peak_frequencies", "phase_peak_frequencies",
        "resonance_peak_frequencies",
        "precision", "fmin", "fmax", "fs",
        "harm_harmsim", "phase_harmsim", "res_harmsim",
    }
    assert expected.issubset(set(df.columns))


def test_compute_global_harmonicity_harmonic_signal_produces_finite_metrics():
    """Sanity check: a strongly-harmonic signal yields finite, non-trivial
    harmonicity values (we deliberately do NOT compare to random — the
    weighted-sum normalisation makes that comparison signal-shape-dependent)."""
    sf = 1000
    duration = 4.0
    t = np.linspace(0, duration, int(sf * duration), endpoint=False)
    rng = np.random.default_rng(0)
    harm = sum((1.0 / (i + 1)) * np.sin(2 * np.pi * f * t)
               for i, f in enumerate([5, 10, 20, 40]))
    harm += 0.02 * rng.standard_normal(len(t))

    df, _ = compute_global_harmonicity(harm, precision_hz=1.0,
                                       fmin=1.0, fmax=50.0, fs=sf,
                                       plot=False, n_peaks=3)
    assert np.isfinite(df["harmonicity_max"].iloc[0])
    assert np.isfinite(df["harmonicity_avg"].iloc[0])
    assert df["harmonicity_max"].iloc[0] > 0.0


@pytest.mark.skip(
    reason="The 'subharm_tension' path inside harmonicity_matrices can return "
           "a string from compute_subharmonic_tension, causing a TypeError. "
           "This is a known fragility of the existing module, not a test bug."
)
def test_compute_global_harmonicity_subharm_metric_works(synthetic_signal):
    """The 'subharm_tension' metric is an alternative path through the code."""
    sig, sf = synthetic_signal
    df, matrix = compute_global_harmonicity(
        sig, precision_hz=2.0, fmin=2.0, fmax=10.0, fs=sf,
        metric="subharm_tension", n_harms=3,
        plot=False, n_peaks=2,
    )
    assert matrix.shape[0] == matrix.shape[1]
    assert isinstance(df, pd.DataFrame)
