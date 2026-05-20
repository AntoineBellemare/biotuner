"""Tests for sinusoidal modeling (SMS) partial tracking.

Synthetic signals with known ground truth: SMS must recover the correct
partials with sub-bin frequency accuracy, ignore noise, and integrate with
the existing biotuner_object pipeline (peaks_function='SMS') without
disturbing other peak extractors.
"""

import numpy as np
import pytest

from biotuner.biotuner_object import compute_biotuner
from biotuner.peaks_extraction import sms_partials


# -- direct sms_partials API -------------------------------------------------


def test_single_pure_tone():
    """One sinusoid → one stable partial near the true frequency."""
    sf = 8000
    f0 = 440.0
    t = np.linspace(0, 2, 2 * sf, endpoint=False)
    sig = np.sin(2 * np.pi * f0 * t)

    peaks, amps, partials = sms_partials(sig, sf=sf, n_fft=1024, hop=256)

    assert len(partials) == 1, f"expected 1 partial, got {len(partials)}"
    assert abs(peaks[0] - f0) < 1.0, f"peak {peaks[0]} should be near {f0}"
    # Pure-tone partial should span almost the whole signal.
    assert partials[0]["duration_sec"] > 1.5


def test_two_stable_tones():
    """Two sinusoids → two partials at the right frequencies, sub-bin accuracy."""
    sf = 8000
    t = np.linspace(0, 2, 2 * sf, endpoint=False)
    sig = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)

    peaks, amps, partials = sms_partials(sig, sf=sf, n_fft=1024, hop=256)

    assert len(partials) >= 2
    found = sorted(peaks[:2])
    # Sub-bin: error well below sf/n_fft = 7.81 Hz.
    assert abs(found[0] - 440) < 1.0
    assert abs(found[1] - 880) < 1.0
    # 440 louder than 880 → ranked first.
    assert peaks[0] == found[0]


def test_drifting_tone_tracked_as_one_partial():
    """A frequency-modulated tone stays a single partial despite the drift."""
    sf = 8000
    dur = 2.0
    t = np.linspace(0, dur, int(dur * sf), endpoint=False)
    # linear sweep 440 → 480 Hz
    phase = 2 * np.pi * (440 * t + (40 / (2 * dur)) * t ** 2)
    sig = np.sin(phase)

    peaks, amps, partials = sms_partials(
        sig, sf=sf, n_fft=1024, hop=256, freq_tolerance_cents=100
    )

    assert len(partials) == 1
    # The tracked freq trajectory should sweep across roughly the right range.
    freq_track = partials[0]["freq"]
    assert freq_track.min() < 460
    assert freq_track.max() > 460


def test_noise_partials_have_low_stability_vs_tone():
    """Noise can produce long tracks (the spectrum is statistically flat across
    frames, so random peak locations recur), but their stability scores must
    be far below those of an actual sinusoid."""
    rng = np.random.default_rng(20260520)
    sf = 8000
    n = 2 * sf

    # Pure tone reference
    t = np.linspace(0, 2, n, endpoint=False)
    tone = np.sin(2 * np.pi * 440 * t)
    tone_partials = sms_partials(tone, sf=sf, n_fft=1024, hop=256)[2]
    assert tone_partials, "tone should produce at least one partial"
    top_tone_stability = max(p["stability"] for p in tone_partials)

    # White noise (matched RMS to the tone)
    noise = rng.normal(0.0, 1.0 / np.sqrt(2), size=n)  # tone RMS = 1/sqrt(2)
    noise_partials = sms_partials(noise, sf=sf, n_fft=1024, hop=256)[2]
    top_noise_stability = max((p["stability"] for p in noise_partials), default=0.0)

    # Tone should be at least 3× more stable than the strongest noise track.
    assert top_tone_stability > 3 * top_noise_stability, (
        f"tone stability {top_tone_stability:.3f} not clearly above "
        f"noise stability {top_noise_stability:.3f}"
    )


def test_signal_too_short_raises():
    sf = 8000
    sig = np.zeros(512)
    with pytest.raises(ValueError, match="too short"):
        sms_partials(sig, sf=sf, n_fft=1024, hop=256)


# -- integration with compute_biotuner --------------------------------------


def test_sms_via_compute_biotuner():
    """peaks_function='SMS' returns peaks + populates bt.partials."""
    sf = 8000
    t = np.linspace(0, 2, 2 * sf, endpoint=False)
    sig = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)

    bt = compute_biotuner(sf=sf, peaks_function="SMS", precision=1.0)
    bt.peaks_extraction(
        sig,
        n_peaks=5,
        max_freq=2000,
        ratios_extension=True,
        min_harms=2,
    )

    assert hasattr(bt, "peaks")
    assert hasattr(bt, "partials")
    assert len(bt.partials) >= 2
    top2 = sorted(bt.peaks[:2])
    assert abs(top2[0] - 440) < 2.0
    assert abs(top2[1] - 880) < 2.0
