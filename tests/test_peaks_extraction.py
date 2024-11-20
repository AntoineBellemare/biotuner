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
    pac_frequencies,
    harmonic_recurrence,
    endogenous_intermodulations,
    polyspectrum_frequencies,
)

# High-level unit tests for the provided signal processing module

def test_EMD_eeg():
    data = np.sin(2 * np.pi * np.linspace(0, 1, 1000) * 5)  # Simulated sine wave
    IMFs = EMD_eeg(data, method="EMD", graph=False)
    assert IMFs.shape[1] == len(data), "Number of IMFs does not match input data length"
    assert IMFs.shape[0] > 1, "EMD should produce more than one IMF"


def test_extract_welch_peaks():
    data = np.sin(2 * np.pi * np.linspace(0, 1, 1000) * 10)  # Sine wave at 10 Hz
    sf = 1000
    peaks, amps, freqs, psd = extract_welch_peaks(
        data, sf, precision=0.5, min_freq=1, max_freq=50, extended_returns=True
    )
    assert len(peaks) > 0, "No peaks were detected"
    assert all(1 <= p <= 50 for p in peaks), "Detected peaks are outside the expected frequency range"


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


def test_pac_frequencies():
    data = np.sin(2 * np.pi * np.linspace(0, 1, 1000) * 5) + np.sin(
        2 * np.pi * np.linspace(0, 1, 1000) * 40
    )
    sf = 1000
    pac_freqs, pac_coupling = pac_frequencies(
        data, sf, method="duprelatour", n_values=5, plot=False
    )
    assert len(pac_freqs) == 5, "Number of PAC frequencies does not match n_values"
    assert len(pac_coupling) == 5, "Number of PAC couplings does not match n_values"


def test_harmonic_recurrence():
    peaks = [5, 10, 20, 40]
    amps = [1, 0.8, 0.5, 0.2]
    max_n, max_peaks, max_amps, harmonics, _, _ = harmonic_recurrence(
        peaks, amps, min_harms=2
    )
    assert len(max_peaks) > 0, "No harmonics detected"
    assert all(5 <= p <= 40 for p in max_peaks), "Harmonics are outside the expected range"


def test_endogenous_intermodulations():
    peaks = [10, 20, 30]
    amps = [1, 0.8, 0.6]
    EIMs, IMCs_all, n_IM_peaks = endogenous_intermodulations(peaks, amps, order=3)
    assert len(EIMs) > 0, "No endogenous intermodulations detected"
    assert n_IM_peaks > 0, "No IM peaks detected"


def test_polyspectrum_frequencies():
    data = np.sin(2 * np.pi * np.linspace(0, 1, 1000) * 5) + np.sin(
        2 * np.pi * np.linspace(0, 1, 1000) * 15
    )
    sf = 1000
    poly_freqs, poly_amps = polyspectrum_frequencies(
        data, sf, precision=0.5, n_values=5, method="bicoherence", graph=False
    )
    assert len(poly_freqs) == 5, "Number of polyspectrum frequencies does not match n_values"
    assert len(poly_amps) == 5, "Number of polyspectrum amplitudes does not match n_values"


if __name__ == "__main__":
    pytest.main()
