"""Tests for the ``noise_to_structure`` wavetable evolution.

The FOOOF-decomposition-as-wavetable: frame 0 is a 1/f^k noise cycle
that loops cleanly; frame N is the standard additive-render of the
clean Timbre; intermediates are waveform-domain crossfades.

Coverage:
  - alpha=1 reproduces the standard render (within normalisation)
  - alpha=0 produces noise with the right spectral shape (1/f^k)
  - Same seed → identical noise frames (deterministic)
  - Different exponents produce different noise spectra
  - Integration through export_wavetable produces a valid multi-frame WAV
"""

import os
import json

import numpy as np
import pytest

from biotuner.harmonic_timbre import Timbre
from biotuner.harmonic_timbre.synthesis import render_wavetable_cycle
from biotuner.harmonic_timbre.exporters.to_wavetable import (
    _frame_with_noise_to_structure,
    _EVOLUTIONS,
    export_wavetable,
)


@pytest.fixture
def basic_timbre():
    return Timbre(
        partials_hz=np.array([220.0, 440.0, 660.0]),
        amplitudes=np.array([1.0, 0.5, 0.25]),
    )


class TestNoiseToStructureFrame:
    def test_alpha_one_matches_clean_render(self, basic_timbre):
        # At alpha=1 the noise contribution is zero; the output is a
        # renormalised version of the standard render. Verify high
        # correlation (waveform shape preserved).
        clean = render_wavetable_cycle(basic_timbre, table_size=512)
        out = _frame_with_noise_to_structure(
            basic_timbre, 1.0, table_size=512,
        )
        corr = float(np.corrcoef(clean, out)[0, 1])
        assert corr > 0.999, f"alpha=1 should be the clean render, corr={corr}"

    def test_alpha_zero_is_pure_noise(self, basic_timbre):
        # At alpha=0 there's no Timbre contribution. The output should
        # be uncorrelated with the clean render (random phases per bin).
        clean = render_wavetable_cycle(basic_timbre, table_size=1024)
        noise = _frame_with_noise_to_structure(
            basic_timbre, 0.0, table_size=1024, seed=42,
        )
        corr = abs(float(np.corrcoef(clean, noise)[0, 1]))
        assert corr < 0.3, f"alpha=0 should be noise (uncorrelated), got corr={corr}"

    def test_alpha_zero_has_powerlaw_psd(self, basic_timbre):
        # Verify the 1/f^k spectrum shape of the noise cycle. For k=1
        # (pink-ish), the PSD slope on a log-log plot should be ≈ -1.
        # We check that the high-freq bins have substantially less
        # energy than the low-freq bins.
        noise = _frame_with_noise_to_structure(
            basic_timbre, 0.0, table_size=2048, exponent=1.0, seed=7,
        )
        spec = np.abs(np.fft.rfft(noise))
        # Low-freq band (bins 1-10) vs high-freq band (bins 200-300).
        low_band  = float(np.mean(spec[1:10]))
        high_band = float(np.mean(spec[200:300]))
        assert low_band > high_band * 5, (
            f"1/f noise should have far more energy in low bins; "
            f"low_band={low_band}, high_band={high_band}"
        )

    def test_seed_determinism(self, basic_timbre):
        a = _frame_with_noise_to_structure(basic_timbre, 0.0, table_size=512, seed=99)
        b = _frame_with_noise_to_structure(basic_timbre, 0.0, table_size=512, seed=99)
        np.testing.assert_allclose(a, b)
        c = _frame_with_noise_to_structure(basic_timbre, 0.0, table_size=512, seed=100)
        # Different seed → different noise.
        assert not np.allclose(a, c)

    def test_exponent_changes_spectrum(self, basic_timbre):
        # k=0.5 (lighter slope) vs k=2.0 (steeper slope) should produce
        # measurably different ratios of low- to high-frequency content.
        flat   = _frame_with_noise_to_structure(
            basic_timbre, 0.0, table_size=2048, exponent=0.5, seed=1,
        )
        steep  = _frame_with_noise_to_structure(
            basic_timbre, 0.0, table_size=2048, exponent=2.0, seed=1,
        )
        # Steeper slope = more energy concentrated at the low end relative
        # to the high end.
        def low_high_ratio(buf):
            spec = np.abs(np.fft.rfft(buf))
            low  = float(np.mean(spec[1:20]))
            high = float(np.mean(spec[200:400]))
            return low / max(high, 1e-9)
        assert low_high_ratio(steep) > low_high_ratio(flat) * 1.5

    def test_exponent_falls_back_to_spectral_tilt(self):
        # When exponent= is None, the helper should pull
        # spectral_tilt off the Timbre.
        t = Timbre(
            partials_hz=np.array([220.0, 440.0]),
            amplitudes=np.array([1.0, 0.5]),
            spectral_tilt=1.5,
        )
        # Should not raise — and the noise spectrum should follow k=1.5.
        out = _frame_with_noise_to_structure(t, 0.0, table_size=1024, seed=0)
        assert out.shape == (1024,)
        assert out.dtype == np.float32

    def test_output_invariants(self, basic_timbre):
        for ts in (256, 1024, 2048):
            for alpha in (0.0, 0.5, 1.0):
                out = _frame_with_noise_to_structure(
                    basic_timbre, alpha, table_size=ts,
                )
                assert out.shape == (ts,)
                assert out.dtype == np.float32
                assert float(np.max(np.abs(out))) <= 0.991


class TestExportIntegration:
    def test_appears_in_evolutions(self):
        assert "noise_to_structure" in _EVOLUTIONS

    def test_export_noise_to_structure(self, basic_timbre, tmp_path):
        out = tmp_path / "n2s.wav"
        result = export_wavetable(
            basic_timbre, str(out),
            n_frames=8, evolution="noise_to_structure",
            noise_exponent=1.2, include_sidecar=False,
        )
        assert os.path.exists(result["wavetable"])
        import soundfile as sf
        data, sr = sf.read(result["wavetable"])
        assert data.shape[0] == 8 * 2048
        manifest = json.load(open(result["manifest"]))
        assert manifest["evolution"] == "noise_to_structure"
        ep = manifest["evolution_params"]
        assert ep["noise_exponent"] == pytest.approx(1.2)
        assert "seed" in ep
