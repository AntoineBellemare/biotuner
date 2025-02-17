import numpy as np
import pytest
from biotuner.biotuner_utils import smooth, top_n_indexes, __get_norm, compute_IMs
from biotuner.biotuner_utils import __product_other_freqs, __freq_ind
from biotuner.vizs import plot_polycoherence
from biotuner.peaks_extraction import (
    EMD_eeg,
    extract_welch_peaks,
    compute_FOOOF,
    HilbertHuang1D,
    cepstrum,
    cepstral_peaks,
    harmonic_recurrence,
    endogenous_intermodulations,
    polyspectrum_frequencies,
)


# EMD tests
def test_EMD_basic():
    """Test EMD mode with a simple sine wave."""
    data = np.sin(2 * np.pi * np.linspace(0, 1, 1000) * 5)  # 5 Hz sine wave
    IMFs = EMD_eeg(data, method="EMD", graph=False)

    assert IMFs.shape[1] == len(data), "Number of time samples in IMFs should match input data"
    assert IMFs.shape[0] > 1, "EMD should produce more than one IMF"
    assert np.all(np.isfinite(IMFs)), "IMF output should contain finite values only"


def test_EEMD_basic():
    """Test EEMD mode with a simple sine wave."""
    data = np.sin(2 * np.pi * np.linspace(0, 1, 1000) * 5)
    IMFs = EMD_eeg(data, method="EEMD", graph=False)

    assert IMFs.shape[1] == len(data), "EEMD IMFs should match input data length"
    assert IMFs.shape[0] > 1, "EEMD should produce multiple IMFs"


def test_invalid_method():
    """Test handling of an invalid method name."""
    data = np.sin(2 * np.pi * np.linspace(0, 1, 1000) * 5)
    with pytest.raises(ValueError, match="method should be 'EMD', 'EEMD', or 'CEEMDAN'"):
        EMD_eeg(data, method="INVALID_METHOD", graph=False)


def test_empty_signal():
    """Test that an empty signal raises a ValueError."""
    data = np.array([])  # Empty array
    with pytest.raises(ValueError, match="Data is empty"):
        EMD_eeg(data, method="EMD", graph=False)


def test_constant_signal():
    """Test that a constant signal raises a ValueError."""
    data = np.ones(1000)  # Constant signal
    with pytest.raises(ValueError, match="Data is constant"):
        EMD_eeg(data, method="EMD", graph=False)


def test_short_signal():
    """Test with a very short signal (should gracefully handle)."""
    data = np.sin(2 * np.pi * np.linspace(0, 1, 10) * 5)  # Only 10 samples
    IMFs = EMD_eeg(data, method="EMD", graph=False)

    assert IMFs.shape[1] == len(data), "Short signal IMFs should match input length"
    assert IMFs.shape[0] > 0, "Short signal should still return at least one IMF"


def test_random_noise():
    """Test with a random noise signal."""
    np.random.seed(42)
    data = np.random.randn(1000)  # White noise
    IMFs = EMD_eeg(data, method="EMD", graph=False)

    assert IMFs.shape[1] == len(data), "Noise signal IMFs should match input data length"
    assert IMFs.shape[0] > 1, "Random noise should produce multiple IMFs"


def test_extract_welch_peaks_basic():
    """Test peak extraction on a simple sine wave."""
    sf = 2000  # Sampling frequency
    freq = 10  # Hz
    data = np.sin(2 * np.pi * np.linspace(0, 1, sf) * freq)  # 1s sine wave at 10 Hz

    peaks, amps, freqs, psd = extract_welch_peaks(data, sf, precision=0.5, min_freq=1, max_freq=50, extended_returns=True)

    assert len(peaks) > 0, "No peaks were detected"
    assert all(1 <= p <= 50 for p in peaks), "Detected peaks are outside expected range"
    assert np.isclose(peaks[0], freq, atol=1.0), f"Expected peak at {freq} Hz, got {peaks[0]}"


def test_extract_welch_peaks_out_type_all():
    """Test 'all' mode to extract multiple peaks."""
    sf = 2000
    data = np.sin(2 * np.pi * np.linspace(0, 1, sf) * 10) + np.sin(2 * np.pi * np.linspace(0, 1, sf) * 20)

    peaks, amps, freqs, psd = extract_welch_peaks(data, sf, out_type="all", extended_returns=True)

    assert len(peaks) >= 2, "Expected multiple peaks in 'all' mode"
    assert 9 <= peaks[0] <= 11, "First peak should be around 10 Hz"
    assert 19 <= peaks[1] <= 21, "Second peak should be around 20 Hz"


def test_extract_welch_peaks_out_type_single():
    """Test 'single' mode to extract the strongest peak."""
    sf = 2000
    data = np.sin(2 * np.pi * np.linspace(0, 1, sf) * 15)  # Strongest component at 15 Hz

    peak, amp, *_ = extract_welch_peaks(data, sf, out_type="single", extended_returns=True)

    assert isinstance(peak, float), "Single peak mode should return a float"
    assert 14 <= peak <= 16, f"Expected dominant peak at 15 Hz, got {peak}"


def test_extract_welch_peaks_out_type_bands():
    """Test 'bands' mode with predefined frequency bands."""
    sf = 2000
    data = np.sin(2 * np.pi * np.linspace(0, 1, sf) * 8) + np.sin(2 * np.pi * np.linspace(0, 1, sf) * 20)
    FREQ_BANDS = [[5, 10], [10, 30]]

    peaks, amps, *_ = extract_welch_peaks(data, sf, FREQ_BANDS=FREQ_BANDS, out_type="bands", extended_returns=True)

    assert len(peaks) == len(FREQ_BANDS), "Number of extracted peaks should match number of frequency bands"
    assert 7 <= peaks[0] <= 9, f"Expected first band peak around 8 Hz, got {peaks[0]}"
    assert 19 <= peaks[1] <= 21, f"Expected second band peak around 20 Hz, got {peaks[1]}"


def test_extract_welch_peaks_invalid_precision():
    """Test that an invalid precision (larger than band range) raises an error."""
    sf = 2000
    data = np.sin(2 * np.pi * np.linspace(0, 1, sf) * 10)
    FREQ_BANDS = [[5, 6]]  # Very small band range

    with pytest.raises(ValueError, match="Precision is larger than a band range"):
        extract_welch_peaks(data, sf, precision=2.0, FREQ_BANDS=FREQ_BANDS, out_type="bands")


def test_extract_welch_peaks_constant_signal():
    """Test that a constant signal results in no peaks detected."""
    sf = 2000
    data = np.ones(sf)  # Flat signal

    peaks, amps, *_ = extract_welch_peaks(data, sf, extended_returns=True)

    assert len(peaks) == 0, "No peaks should be detected in a constant signal"


def test_extract_welch_peaks_empty_signal():
    """Test that an empty signal raises an error."""
    sf = 2000
    data = np.array([])

    with pytest.raises(ValueError, match="cannot be empty"):
        extract_welch_peaks(data, sf, extended_returns=True)


def test_extract_welch_peaks_low_prominence():
    """Test detection of weak peaks using a low prominence threshold."""
    sf = 2000
    data = np.sin(2 * np.pi * np.linspace(0, 1, sf) * 10) * 0.01  # Very low amplitude

    peaks, amps, *_ = extract_welch_peaks(data, sf, prominence=0.01, extended_returns=True)

    assert len(peaks) > 0, "Expected to detect weak peaks with low prominence"


def test_compute_FOOOF():
    data = np.sin(2 * np.pi * np.linspace(0, 1, 1000) * 15)  # Sine wave at 15 Hz
    sf = 1000
    peaks, amps = compute_FOOOF(data, sf, max_freq=50, n_peaks=1, graph=False)
    assert len(peaks) == 1, "Number of peaks does not match n_peaks"
    assert 14 <= peaks[0] <= 16, "Detected peak frequency is outside the expected range"


def test_HilbertHuang1D():
    data = np.sin(2 * np.pi * np.linspace(0, 1, 1000) * 20)  # Sine wave at 20 Hz
    sf = 1000
    _, peaks, amps, _, _ = HilbertHuang1D(data, sf, nIMFs=5, min_freq=1, max_freq=50)
    assert len(peaks) > 0, "No peaks detected in Hilbert-Huang Transform"
    assert all(1 <= p <= 50 for p in peaks), "Peaks are outside the specified frequency range"


def test_cepstrum_and_cepstral_peaks():
    data = np.sin(2 * np.pi * np.linspace(0, 1, 1000) * 30)  # Sine wave at 30 Hz
    sf = 1000
    cepst, quef = cepstrum(data, sf, plot_cepstrum=False)
    peaks, amps = cepstral_peaks(cepst, quef, max_time=0.05, min_time=0.01)
    assert len(peaks) > 0, "No cepstral peaks detected"
    assert all(10 <= p <= 50 for p in peaks), "Cepstral peaks are outside the expected range"


def test_harmonic_recurrence():
    peaks = [5, 10, 20, 40]
    amps = [1, 0.8, 0.5, 0.2]
    max_n, max_peaks, max_amps, harmonics, _, _ = harmonic_recurrence(peaks, amps, min_harms=2)
    assert len(max_peaks) > 0, "No harmonics detected"
    assert all(5 <= p <= 40 for p in max_peaks), "Harmonics are outside the expected range"


def test_endogenous_intermodulations():
    peaks = [10, 20, 30]
    amps = [1, 0.8, 0.6]
    EIMs, IMCs_all, n_IM_peaks = endogenous_intermodulations(peaks, amps, order=3)
    assert len(EIMs) > 0, "No endogenous intermodulations detected"
    assert n_IM_peaks > 0, "No IM peaks detected"


# def test_polyspectrum_frequencies():
#     data = np.sin(2 * np.pi * np.linspace(0, 1, 1000) * 5) + np.sin(
#         2 * np.pi * np.linspace(0, 1, 1000) * 15
#     )
#     sf = 1000
#     poly_freqs, poly_amps = polyspectrum_frequencies(
#         data, sf, precision=0.5, n_values=5, method="bicoherence", graph=False
#     )
#     assert len(poly_freqs) == 5, "Number of polyspectrum frequencies does not match n_values"
#     assert len(poly_amps) == 5, "Number of polyspectrum amplitudes does not match n_values"


if __name__ == "__main__":
    pytest.main()
