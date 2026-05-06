"""Tests for biotuner.harmonic_timbre.synthesis."""

from __future__ import annotations

import numpy as np
import pytest

from biotuner.harmonic_timbre import (
    Timbre,
    render_additive,
    render_with_envelope,
    render_band_limited,
    render_wavetable_cycle,
)


# ---------------------------------------------------------------------------
# Shape & dtype invariants
# ---------------------------------------------------------------------------

def _basic_timbre() -> Timbre:
    return Timbre(
        partials_hz=[110.0, 220.0, 440.0],
        amplitudes=[1.0, 0.5, 0.25],
    )


def test_render_additive_shape_and_dtype():
    t = _basic_timbre()
    out = render_additive(t, samplerate=22050, duration=0.5)
    assert out.shape == (11025,)
    assert out.dtype == np.float32


def test_render_additive_zero_duration_returns_empty():
    out = render_additive(_basic_timbre(), samplerate=22050, duration=0.0)
    assert out.shape == (0,)


def test_render_normalized_buffer_within_unit():
    out = render_additive(_basic_timbre(), samplerate=22050, duration=0.25, normalize=True)
    assert np.max(np.abs(out)) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Spectral correctness — rendered partials show up in the FFT
# ---------------------------------------------------------------------------

def test_rendered_partials_appear_in_fft():
    sr = 22050
    duration = 1.0
    f1, f2, f3 = 110.0, 220.0, 440.0
    t = Timbre(
        partials_hz=[f1, f2, f3],
        amplitudes=[1.0, 1.0, 1.0],
    )
    out = render_additive(t, samplerate=sr, duration=duration, normalize=False)
    spec = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(out.size, d=1.0 / sr)
    for f in (f1, f2, f3):
        idx = int(np.argmin(np.abs(freqs - f)))
        # peak at the partial frequency
        assert spec[idx] > spec.mean() * 50  # well above mean — strong peak


def test_envelope_decays_per_partial():
    sr = 22050
    duration = 0.5
    t = Timbre(
        partials_hz=[110.0],
        amplitudes=[1.0],
        decay_times=[0.05],  # τ=50 ms
    )
    out = render_with_envelope(t, samplerate=sr, duration=duration, normalize=False)
    # envelope at t=0.4 s ≈ exp(-0.4 / 0.05) ≈ exp(-8) ≈ 3.3e-4
    front_window = out[:1000]
    tail_window = out[-1000:]
    assert np.max(np.abs(tail_window)) < np.max(np.abs(front_window)) * 0.05


def test_spectral_tilt_attenuates_high_partials():
    sr = 22050
    duration = 0.5
    t_no_tilt = Timbre(partials_hz=[110.0, 1100.0], amplitudes=[1.0, 1.0])
    t_with_tilt = Timbre(partials_hz=[110.0, 1100.0], amplitudes=[1.0, 1.0],
                         spectral_tilt=2.0)  # 1/f² falloff
    out_no = render_with_envelope(t_no_tilt, samplerate=sr, duration=duration, normalize=False)
    out_yes = render_with_envelope(t_with_tilt, samplerate=sr, duration=duration, normalize=False)
    # FFT magnitude at 1100 Hz
    spec_no = np.abs(np.fft.rfft(out_no))
    spec_yes = np.abs(np.fft.rfft(out_yes))
    freqs = np.fft.rfftfreq(out_no.size, d=1.0 / sr)
    idx_high = int(np.argmin(np.abs(freqs - 1100.0)))
    assert spec_yes[idx_high] < spec_no[idx_high] * 0.5


def test_noise_floor_adds_broadband_energy():
    sr = 22050
    duration = 0.5
    t_clean = Timbre(partials_hz=[110.0], amplitudes=[1.0])
    t_noisy = Timbre(partials_hz=[110.0], amplitudes=[1.0], noise_floor=0.5)
    out_clean = render_with_envelope(t_clean, samplerate=sr, duration=duration, normalize=False)
    out_noisy = render_with_envelope(t_noisy, samplerate=sr, duration=duration, normalize=False)
    spec_clean = np.abs(np.fft.rfft(out_clean))
    spec_noisy = np.abs(np.fft.rfft(out_noisy))
    # noisy spectrum has higher mean energy (broadband floor)
    assert spec_noisy.mean() > spec_clean.mean() * 2


# ---------------------------------------------------------------------------
# render_band_limited
# ---------------------------------------------------------------------------

def test_band_limited_falls_back_when_no_bandwidths():
    t = _basic_timbre()
    out = render_band_limited(t, samplerate=22050, duration=0.25)
    assert out.shape == (5512,)


def test_band_limited_runs_with_bandwidths():
    sr = 22050
    t = Timbre(
        partials_hz=[110.0, 220.0, 440.0],
        amplitudes=[1.0, 1.0, 1.0],
        bandwidths=[2.0, 5.0, 10.0],
    )
    out = render_band_limited(t, samplerate=sr, duration=0.5)
    assert out.shape == (11025,)
    assert out.dtype == np.float32
    assert np.any(out != 0)


# ---------------------------------------------------------------------------
# render_wavetable_cycle
# ---------------------------------------------------------------------------

def test_wavetable_cycle_size():
    t = Timbre(partials_hz=[100.0, 200.0], amplitudes=[1.0, 0.5], base_freq=100.0)
    cycle = render_wavetable_cycle(t, table_size=2048)
    assert cycle.shape == (2048,)
    assert cycle.dtype == np.float32


def test_wavetable_cycle_recovers_partials_from_fft():
    """One cycle of partials [1, 2] should show energy at FFT bins 1 and 2."""
    t = Timbre(partials_hz=[100.0, 200.0], amplitudes=[1.0, 1.0], base_freq=100.0)
    cycle = render_wavetable_cycle(t, table_size=2048)
    spec = np.abs(np.fft.rfft(cycle))
    # bin 1 = fundamental, bin 2 = 2nd harmonic; both should be peaks
    assert spec[1] > spec.mean() * 50
    assert spec[2] > spec.mean() * 50


# ---------------------------------------------------------------------------
# render_modulated — AM and FM sidebands
# ---------------------------------------------------------------------------

def test_render_modulated_falls_back_when_no_modulators():
    """Without any AM/FM modulators, render_modulated == render_with_envelope."""
    from biotuner.harmonic_timbre import render_modulated
    t = Timbre(partials_hz=[110.0], amplitudes=[1.0])
    out = render_modulated(t, samplerate=8000, duration=0.25)
    assert out.dtype == np.float32
    assert out.shape == (2000,)


def test_render_modulated_am_produces_sidebands():
    """AM at carrier f_c with modulator f_m should produce peaks at f_c ± f_m."""
    from biotuner.harmonic_timbre import Modulator, render_modulated
    f_c, f_m = 200.0, 20.0
    t = Timbre(
        partials_hz=[f_c],
        amplitudes=[1.0],
        am_modulators=[Modulator(carrier_idx=0, mod_freq=f_m, depth=0.5, mod_type="AM")],
    )
    sr = 8000
    out = render_modulated(t, samplerate=sr, duration=2.0, normalize=False)
    spec = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(out.size, d=1.0 / sr)

    def _at(f):
        return spec[int(np.argmin(np.abs(freqs - f)))]

    # carrier and both sidebands should be strong; carrier strongest
    assert _at(f_c) > _at(f_c - f_m) * 0.8
    assert _at(f_c - f_m) > spec.mean() * 30
    assert _at(f_c + f_m) > spec.mean() * 30


def test_render_modulated_fm_produces_bessel_spread():
    """FM with mod_index β = depth/mod_freq ≈ 4 should produce ≥ 5 strong
    sidebands (the central peak + at least the first 2 pairs of Bessel terms)."""
    from biotuner.harmonic_timbre import Modulator, render_modulated
    f_c, f_m, dev = 200.0, 20.0, 80.0  # β = 4
    t = Timbre(
        partials_hz=[f_c],
        amplitudes=[1.0],
        fm_modulators=[Modulator(carrier_idx=0, mod_freq=f_m, depth=dev, mod_type="FM")],
    )
    sr = 8000
    out = render_modulated(t, samplerate=sr, duration=2.0, normalize=False)
    spec = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(out.size, d=1.0 / sr)
    spec_n = spec / spec.max()
    # count strong peaks within the carrier's FM bandwidth (Carson's rule
    # bandwidth ~ 2(dev + mod_freq) = 200; window 100..300 Hz)
    band = (freqs >= f_c - dev) & (freqs <= f_c + dev)
    n_strong = int((spec_n[band] > 0.1).sum())
    assert n_strong >= 5


def test_timbre_synthesize_dispatches_to_modulated():
    """Timbre.synthesize() should route through render_modulated when
    am/fm modulators are present (i.e. produce a non-flat spectrum)."""
    from biotuner.harmonic_timbre import Modulator
    t = Timbre(
        partials_hz=[200.0],
        amplitudes=[1.0],
        am_modulators=[Modulator(carrier_idx=0, mod_freq=20.0, depth=0.5, mod_type="AM")],
    )
    sr = 8000
    out = t.synthesize(samplerate=sr, duration=1.0, normalize=False)
    spec = np.abs(np.fft.rfft(out))
    freqs = np.fft.rfftfreq(out.size, d=1.0 / sr)
    # Sidebands at 180 and 220 Hz
    bin_180 = spec[int(np.argmin(np.abs(freqs - 180.0)))]
    bin_220 = spec[int(np.argmin(np.abs(freqs - 220.0)))]
    assert bin_180 > spec.mean() * 30
    assert bin_220 > spec.mean() * 30
