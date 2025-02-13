import pytest
import numpy as np
from biotuner.biotuner_object import compute_biotuner


def create_test_signal(
    sampling_frequency, duration, peak_frequencies, amplitudes=None, noise_level=0.0
):
    t = np.linspace(0, duration, int(sampling_frequency * duration), endpoint=False)
    if amplitudes is None:
        amplitudes = [1.0] * len(peak_frequencies)

    # Create the signal by summing sinusoidal components
    signal = sum(
        amplitudes[i] * np.sin(2 * np.pi * f * t)
        for i, f in enumerate(peak_frequencies)
    )

    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, size=len(t))
    return signal + noise


# Use higher amplitudes and lower noise
AMPLITUDES = [1.5, 1.0, 1.2]  # Increase amplitudes
NOISE_LEVEL = 0.05  # Reduce noise

# Constants for the test signal
SAMPLING_FREQUENCY = 1000  # Hz
DURATION = 10.0  # seconds
PEAK_FREQUENCIES = [3, 10, 50]  # Hz
AMPLITUDES = [1.0, 0.5, 0.8]  # relative amplitudes
NOISE_LEVEL = 0.05  # Gaussian noise level


@pytest.fixture
def test_signal():
    return create_test_signal(
        SAMPLING_FREQUENCY, DURATION, PEAK_FREQUENCIES, AMPLITUDES, NOISE_LEVEL
    )


@pytest.fixture
def biotuner_instance():
    return compute_biotuner(sf=SAMPLING_FREQUENCY)


def test_initialization(biotuner_instance):
    bt = biotuner_instance
    assert bt.sf == SAMPLING_FREQUENCY
    assert bt.peaks_function == "EMD"
    assert bt.precision == 0.1


def test_peaks_extraction(biotuner_instance, test_signal):
    bt = biotuner_instance

    # Call peaks_extraction with the adjusted `nperseg`
    bt.peaks_extraction(test_signal, precision=1)

    print("Expected peaks:", PEAK_FREQUENCIES)
    print("Detected peaks:", bt.peaks)

    # Assert that detected peaks include the known peaks within tolerance
    for freq in PEAK_FREQUENCIES:
        assert any(
            abs(freq - detected) < 1 for detected in bt.peaks
        ), f"Peak {freq} not found"


def test_peaks_metrics(biotuner_instance, test_signal):
    bt = biotuner_instance
    bt.peaks_extraction(test_signal)
    bt.compute_peaks_metrics()
    assert "cons" in bt.peaks_metrics
    assert bt.peaks_metrics["cons"] >= 0


def test_peaks_extension(biotuner_instance, test_signal):
    bt = biotuner_instance
    bt.peaks_extraction(test_signal)
    bt.peaks_extension()
    assert len(bt.extended_peaks) >= len(bt.peaks)


# def test_resonance_computation(biotuner_instance, test_signal):
#     bt = biotuner_instance
#     bt.peaks_extraction(test_signal)
#     bt.compute_resonance()
#     assert bt.resonance is not None
#     assert isinstance(bt.resonant_freqs, list)


def test_harmonic_entropy(biotuner_instance, test_signal):
    bt = biotuner_instance
    bt.peaks_extraction(test_signal, precision=1)
    bt.peaks_extension()  # Generate extended peaks
    bt.compute_harmonic_entropy(plot_entropy=False, input_type="extended_peaks")
    assert len(bt.HE_scale) > 0
    assert bt.scale_metrics["HE"] is not None


def test_dissonance_curve(biotuner_instance, test_signal):
    bt = biotuner_instance
    bt.peaks_extraction(test_signal, precision=1)
    bt.compute_diss_curve()
    assert len(bt.diss_scale) > 0
    assert bt.scale_metrics["dissonance"] is not None


def test_invalid_input(biotuner_instance):
    bt = biotuner_instance
    with pytest.raises(ValueError):
        bt.peaks_extraction(data=[])


def test_known_output(biotuner_instance):
    bt = biotuner_instance
    test_signal = create_test_signal(
        SAMPLING_FREQUENCY, DURATION, PEAK_FREQUENCIES, AMPLITUDES, NOISE_LEVEL
    )
    bt.peaks_extraction(test_signal, precision=1)

    print("Expected Peaks:", PEAK_FREQUENCIES)
    print("Detected Peaks:", bt.peaks)

    for freq in PEAK_FREQUENCIES:
        matches = [detected for detected in bt.peaks if abs(freq - detected) < 1]
        print(f"Checking for peak near {freq}: {matches}")
        assert matches, f"Peak {freq} not found"
