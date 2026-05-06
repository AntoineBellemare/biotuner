"""Tests for biotuner.harmonic_timbre.direct_synth."""

from __future__ import annotations

import numpy as np
import pytest

from biotuner.harmonic_timbre import (
    fm_patch_from_tuning,
    hilbert_instrument,
    render_modulated,
)


# ---------------------------------------------------------------------------
# hilbert_instrument
# ---------------------------------------------------------------------------

def test_hilbert_instrument_renders_a_pure_tone_for_a_pure_tone():
    """For a single-frequency biosignal, the output should have most of
    its energy concentrated near base_freq + (input_freq * pitch_factor)."""
    sf = 1000.0
    f_in = 5.0
    t = np.arange(2 * sf) / sf
    sig = np.sin(2 * np.pi * f_in * t)

    out = hilbert_instrument(
        sig, sf=sf,
        samplerate=8000, duration=2.0,
        base_freq=220.0, pitch_factor=20.0,
        smooth_amp=0.0,
    )
    # expected dominant audio freq ≈ 220 + 5*20 = 320 Hz
    spec = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(out.size, d=1.0 / 8000)
    peak_freq = float(freqs[int(np.argmax(spec))])
    assert abs(peak_freq - 320.0) < 5.0


def test_hilbert_instrument_short_signal_raises():
    with pytest.raises(ValueError, match="at least 4 samples"):
        hilbert_instrument([0.0, 1.0, 0.0], sf=1000.0)


def test_hilbert_instrument_amplitude_envelope_propagates():
    """An amplitude-modulated input should produce an amplitude-modulated output."""
    sf = 1000.0
    t = np.arange(2 * sf) / sf
    # carrier at 5 Hz with envelope at 0.5 Hz
    sig = (1.0 + 0.8 * np.sin(2 * np.pi * 0.5 * t)) * np.sin(2 * np.pi * 5.0 * t)
    out = hilbert_instrument(
        sig, sf=sf,
        samplerate=4000, duration=2.0,
        base_freq=200.0, pitch_factor=10.0,
        smooth_amp=0.05,
    )
    # the output envelope (smoothed) should not be flat
    env = np.abs(out)
    # high vs low percentile span
    assert np.percentile(env, 95) - np.percentile(env, 5) > 0.1


# ---------------------------------------------------------------------------
# fm_patch_from_tuning
# ---------------------------------------------------------------------------

def test_fm_patch_populates_modulators():
    ratios = [1.0, 5/4, 3/2, 2.0]
    timbre = fm_patch_from_tuning(ratios, n_carriers=4, fm_index=1.5)
    assert timbre.matching_method == "fm_patch_from_tuning"
    assert len(timbre.fm_modulators) == 4
    assert all(m.mod_type == "FM" for m in timbre.fm_modulators)


def test_fm_patch_renders_with_sidebands():
    """Rendering an FM-patch Timbre should produce audible sidebands per carrier."""
    ratios = [1.0, 3/2]
    timbre = fm_patch_from_tuning(ratios, n_carriers=2, fm_index=2.0, base_freq=200.0)
    out = render_modulated(timbre, samplerate=8000, duration=1.0, normalize=False)
    spec = np.abs(np.fft.rfft(out))
    spec_n = spec / spec.max()
    # at least 6 strong-ish FFT peaks (2 carriers × Bessel sidebands)
    n_strong = int((spec_n > 0.08).sum())
    assert n_strong >= 6


def test_fm_patch_empty_ratios_raises():
    with pytest.raises(ValueError, match="empty ratios"):
        fm_patch_from_tuning([])


def test_fm_patch_unknown_falloff_raises():
    with pytest.raises(ValueError, match="unknown falloff"):
        fm_patch_from_tuning([1.0, 1.5], falloff="rainbow")


def test_fm_patch_carrier_count_minimum():
    with pytest.raises(ValueError, match="n_carriers"):
        fm_patch_from_tuning([1.0, 1.5], n_carriers=0)
