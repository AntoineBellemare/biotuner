"""Tests for compute_multitaper_psd and spectrum_method='multitaper' opt-in.

Multitaper PSD is added as an alternative spectrum estimator. It should:
1. Resolve known sinusoids on a clean signal,
2. Have lower variance than single-taper FFT on noisy short signals,
3. Plug into compute_biotuner's FOOOF / harmonic_recurrence / EIMC branches
   via spectrum_method='multitaper' without disturbing default behavior.
"""

import numpy as np
import pytest

from biotuner.peaks_extraction import compute_multitaper_psd
from biotuner.biotuner_object import compute_biotuner


# -- direct PSD API ---------------------------------------------------------


def test_psd_resolves_known_tones():
    sf = 1000
    t = np.linspace(0, 4, 4 * sf, endpoint=False)
    sig = np.sin(2 * np.pi * 10 * t) + 0.3 * np.sin(2 * np.pi * 27 * t)
    freqs, psd = compute_multitaper_psd(sig, sf=sf, f_max=60)

    assert freqs.shape == psd.shape
    assert freqs.min() >= 0
    assert freqs.max() <= 60
    # Dominant bin within 1 Hz of 10 Hz
    assert abs(freqs[np.argmax(psd)] - 10.0) < 1.0
    # The 27 Hz tone should be a clear secondary peak
    band = (freqs >= 24) & (freqs <= 30)
    peak_in_band = freqs[band][np.argmax(psd[band])]
    assert abs(peak_in_band - 27.0) < 1.0


def test_psd_returns_only_requested_band():
    sf = 1000
    t = np.linspace(0, 2, 2 * sf, endpoint=False)
    sig = np.sin(2 * np.pi * 10 * t)
    freqs, _ = compute_multitaper_psd(sig, sf=sf, f_min=5, f_max=30)
    assert freqs.min() >= 5.0
    assert freqs.max() <= 30.0


def test_psd_short_signal_raises():
    with pytest.raises(ValueError, match="too short"):
        compute_multitaper_psd(np.array([0.0, 1.0]), sf=1000)


def test_psd_lower_variance_than_single_taper_on_noise():
    """Multitaper averages K spectra → lower variance than a single Hann FFT."""
    rng = np.random.default_rng(20260520)
    sf = 1000
    n = 4 * sf
    noise = rng.normal(0.0, 1.0, size=n)

    _, mt_psd = compute_multitaper_psd(noise, sf=sf, f_max=200)

    # Single-taper FFT periodogram on the same signal, same band
    import scipy.signal
    win = scipy.signal.windows.hann(n)
    spec = np.fft.rfft(noise * win)
    psd = (np.abs(spec) ** 2) / sf
    psd[1:-1] *= 2
    freqs_st = np.fft.rfftfreq(n, d=1.0 / sf)
    band = (freqs_st > 0) & (freqs_st <= 200)
    st_psd = psd[band]

    # Variance of multitaper PSD in this band is strictly lower than the
    # single-taper periodogram on white noise.
    assert mt_psd.var() < st_psd.var()


# -- integration with compute_biotuner --------------------------------------


def test_default_spectrum_method_is_fft():
    """Constructor default should remain 'fft' so existing callers aren't
    silently switched."""
    bt = compute_biotuner(sf=1000)
    assert getattr(bt, "spectrum_method", "fft") == "fft"


def test_harmonic_recurrence_multitaper_smoke():
    # Harmonic series so harmonic_recurrence's min_harms=2 constraint is
    # satisfied: 10 Hz fundamental + harmonics at 20/30/40/50.
    sf = 1000
    t = np.linspace(0, 4, 4 * sf, endpoint=False)
    sig = sum(
        amp * np.sin(2 * np.pi * f * t)
        for f, amp in [(10, 1.0), (20, 0.6), (30, 0.4), (40, 0.3), (50, 0.2)]
    )
    bt = compute_biotuner(
        sf=sf, peaks_function="harmonic_recurrence",
        precision=0.5, spectrum_method="multitaper",
    )
    bt.peaks_extraction(sig, n_peaks=5, max_freq=60)
    assert hasattr(bt, "peaks")
    assert len(bt.peaks) >= 1
    # The strongest fundamental should be near 10 Hz.
    closest_to_10 = min(bt.peaks, key=lambda p: abs(p - 10))
    assert abs(closest_to_10 - 10) < 2.0


def test_fooof_multitaper_smoke():
    sf = 1000
    t = np.linspace(0, 4, 4 * sf, endpoint=False)
    sig = np.sin(2 * np.pi * 12 * t) + 0.6 * np.sin(2 * np.pi * 30 * t)
    bt = compute_biotuner(
        sf=sf, peaks_function="FOOOF",
        precision=0.5, spectrum_method="multitaper",
    )
    bt.peaks_extraction(sig, n_peaks=5, max_freq=80)
    assert hasattr(bt, "peaks")
    assert len(bt.peaks) >= 1
    closest_to_12 = min(bt.peaks, key=lambda p: abs(p - 12))
    assert abs(closest_to_12 - 12) < 2.0
