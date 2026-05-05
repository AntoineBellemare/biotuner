"""Tests for beat_envelope / beat_spectrogram in rhythm_construction.

Focused on the new temporal-interference helpers; the rest of
rhythm_construction is tested in its own demos / notebooks.
"""

import numpy as np
import pytest

from biotuner.rhythm_construction import beat_envelope, beat_spectrogram


# ============================================================ beat_envelope


class TestBeatEnvelope:
    def test_shape(self):
        t, env = beat_envelope([100.0, 102.0], duration_s=1.0, sr=2000)
        assert t.shape == (2000,)
        assert env.shape == (2000,)

    def test_envelope_is_nonneg(self):
        _, env = beat_envelope([100.0, 102.0], duration_s=1.0, sr=2000)
        assert (env >= 0).all()

    def test_two_tone_beat_period(self):
        # Two tones at 100 and 102 Hz produce beats at |102 - 100| = 2 Hz,
        # so the envelope's beat period is 0.5 s. Across a 2-second window
        # we should see ~4 envelope minima and ~4 maxima.
        df = 2.0  # beat frequency
        t, env = beat_envelope(
            [100.0, 100.0 + df], amplitudes=[1.0, 1.0],
            duration_s=2.0, sr=4000,
        )
        # Find the dominant Fourier peak in the envelope (skip DC).
        env_centred = env - env.mean()
        spectrum = np.abs(np.fft.rfft(env_centred))
        freqs = np.fft.rfftfreq(env.size, d=1.0 / 4000)
        peak_freq = float(freqs[1 + int(np.argmax(spectrum[1:]))])
        # The envelope's dominant period is 1/(2·df) = 0.25 s ⇒ 4 Hz.
        # (Beats: |x| has period 0.5 s, but x² and analytic envelope at
        # equal amplitudes oscillate twice per beat → 4 Hz in our data.)
        # Allow 2 Hz or 4 Hz given different envelope conventions.
        assert peak_freq in (
            pytest.approx(df, abs=0.5),
            pytest.approx(2 * df, abs=0.5),
        )

    def test_return_signal(self):
        t, env, sig = beat_envelope(
            [100.0, 102.0], duration_s=0.5, sr=2000,
            return_signal=True,
        )
        assert sig.shape == env.shape
        # Signal can go negative; envelope cannot.
        assert sig.min() < 0
        assert env.min() >= 0

    def test_single_peak_constant_envelope(self):
        # Single peak: envelope of cos(2πft) is constant (Hilbert
        # analytic gives unit amplitude).
        _, env = beat_envelope(
            [100.0], amplitudes=[1.0],
            duration_s=0.5, sr=4000,
        )
        # Drop edge samples (Hilbert has edge artefacts).
        core = env[200:-200]
        np.testing.assert_allclose(core, core[0], atol=5e-3)

    def test_invalid_duration(self):
        with pytest.raises(ValueError):
            beat_envelope([100.0], duration_s=0.0)

    def test_invalid_sr(self):
        with pytest.raises(ValueError):
            beat_envelope([100.0], sr=0)

    def test_empty_peaks(self):
        with pytest.raises(ValueError):
            beat_envelope([])

    def test_amps_length_mismatch(self):
        with pytest.raises(ValueError):
            beat_envelope(
                [100.0, 150.0], amplitudes=[1.0],
                duration_s=0.5, sr=1000,
            )

    def test_too_short_duration(self):
        # duration_s * sr < 2 samples is rejected.
        with pytest.raises(ValueError):
            beat_envelope([100.0], duration_s=0.0001, sr=10)


# ============================================================ beat_spectrogram


class TestBeatSpectrogram:
    def test_shape(self):
        t, f, S = beat_spectrogram(
            [100.0, 150.0], duration_s=1.0, sr=2000,
            n_fft=256, hop=64,
        )
        # Expected: n_freq = 256/2 + 1 = 129
        assert f.shape == (129,)
        # Frames: 1 + (2000 - 256) // 64 = 1 + 27 = 28
        assert t.shape == (28,)
        assert S.shape == (129, 28)

    def test_peaks_appear_in_spectrum(self):
        t, f, S = beat_spectrogram(
            [200.0, 350.0], amplitudes=[1.0, 1.0],
            duration_s=1.0, sr=2000,
            n_fft=512, hop=128,
        )
        # The total spectrum (mean over time) should peak near the
        # input frequencies.
        total = S.mean(axis=1)
        # Find the two largest peaks.
        n_peaks = 2
        top_idx = np.argsort(total)[-n_peaks:]
        top_freqs = sorted(f[top_idx].tolist())
        # Allow ~10 Hz tolerance (FFT bin width ≈ 2000/512 ≈ 4 Hz).
        assert top_freqs[0] == pytest.approx(200.0, abs=10.0)
        assert top_freqs[1] == pytest.approx(350.0, abs=10.0)

    def test_log_power_vs_linear(self):
        # Both should have the same shape; values differ by log1p(·²) vs |·|.
        kw = dict(duration_s=1.0, sr=2000, n_fft=256, hop=64)
        _, _, S_log = beat_spectrogram([100.0, 150.0], log_power=True, **kw)
        _, _, S_lin = beat_spectrogram([100.0, 150.0], log_power=False, **kw)
        assert S_log.shape == S_lin.shape

    def test_invalid_n_fft(self):
        with pytest.raises(ValueError):
            beat_spectrogram([100.0, 150.0], n_fft=4)

    def test_invalid_hop(self):
        with pytest.raises(ValueError):
            beat_spectrogram([100.0, 150.0], hop=0)

    def test_signal_too_short(self):
        # n_fft larger than the signal → error.
        with pytest.raises(ValueError):
            beat_spectrogram(
                [100.0, 150.0], duration_s=0.05, sr=1000, n_fft=512,
            )


# ============================================================ composability


class TestComposability:
    def test_works_with_harmonic_geometry_input(self):
        """Smoke: take a HarmonicInput's peaks/amplitudes and render
        the temporal beat envelope of the same chord."""
        from biotuner.harmonic_geometry import HarmonicInput
        chord = HarmonicInput(peaks=[100.0, 125.0, 150.0])
        t, env = beat_envelope(
            chord.to_peaks(),
            chord.normalized_amplitudes(),
            duration_s=0.5, sr=2000,
        )
        assert env.shape == (1000,)
