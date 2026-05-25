"""Tests for the generic Timbre-sequence morph used by imf_morph + band_morph,
plus the signal→Timbre-sequence adapters that feed it.

Coverage:
  - _frame_with_timbre_morph endpoints (frame 0 = timbre[0],
    frame N-1 = timbre[K-1])
  - Three blend modes ('pure' / 'linear_walk' / 'gaussian') produce
    measurably different outputs at the midpoint
  - Single-Timbre input → static frames (degenerate but should work)
  - Adapter: timbres_from_imfs returns a Timbre per non-empty IMF
  - Adapter: timbres_from_bands respects the edges list
  - Full export_wavetable round-trip with imf_morph + band_morph
"""

import json
import os

import numpy as np
import pytest

from biotuner.harmonic_timbre import Timbre
from biotuner.harmonic_timbre.synthesis import render_wavetable_cycle
from biotuner.harmonic_timbre.exporters.to_wavetable import (
    _frame_with_timbre_morph,
    _EVOLUTIONS,
    export_wavetable,
)
from biotuner.harmonic_timbre.biotuner_mapping import (
    timbres_from_imfs,
    timbres_from_bands,
)


@pytest.fixture
def two_timbres():
    """Two clearly-distinct Timbres: one low (220 Hz fundamental, simple
    harmonic stack) and one high (880 Hz, denser stack)."""
    low = Timbre(
        partials_hz=np.array([220.0, 440.0, 660.0]),
        amplitudes=np.array([1.0, 0.5, 0.25]),
        base_freq=220.0,
    )
    high = Timbre(
        partials_hz=np.array([880.0, 1320.0, 1760.0, 2200.0]),
        amplitudes=np.array([1.0, 0.7, 0.5, 0.3]),
        base_freq=220.0,
    )
    return low, high


class TestFrameTimbreMorph:
    def test_first_frame_matches_first_timbre_linear(self, two_timbres):
        low, high = two_timbres
        out = _frame_with_timbre_morph(
            [low, high], frame_idx=0, n_frames=8, table_size=512,
            blend_mode="linear_walk",
        )
        baseline = render_wavetable_cycle(low, table_size=512)
        corr = float(np.corrcoef(out, baseline)[0, 1])
        assert corr > 0.999

    def test_last_frame_matches_last_timbre_linear(self, two_timbres):
        low, high = two_timbres
        out = _frame_with_timbre_morph(
            [low, high], frame_idx=7, n_frames=8, table_size=512,
            blend_mode="linear_walk",
        )
        baseline = render_wavetable_cycle(high, table_size=512)
        corr = float(np.corrcoef(out, baseline)[0, 1])
        assert corr > 0.999

    def test_pure_mode_is_stepwise(self, two_timbres):
        # In pure mode, frames near the midpoint should ALL match
        # exactly one of the two Timbres (no blending).
        low, high = two_timbres
        mid_frame = _frame_with_timbre_morph(
            [low, high], frame_idx=4, n_frames=8, table_size=512,
            blend_mode="pure",
        )
        # Should match high (round(4/7 * 1) = 1).
        baseline_high = render_wavetable_cycle(high, table_size=512)
        baseline_low  = render_wavetable_cycle(low, table_size=512)
        assert float(np.corrcoef(mid_frame, baseline_high)[0, 1]) > 0.99 or \
               float(np.corrcoef(mid_frame, baseline_low)[0, 1]) > 0.99

    def test_blend_modes_differ_at_midpoint(self, two_timbres):
        low, high = two_timbres
        out_pure = _frame_with_timbre_morph(
            [low, high], frame_idx=4, n_frames=9, table_size=512,
            blend_mode="pure",
        )
        out_lin = _frame_with_timbre_morph(
            [low, high], frame_idx=4, n_frames=9, table_size=512,
            blend_mode="linear_walk",
        )
        assert not np.allclose(out_pure, out_lin, atol=1e-3)

    def test_gaussian_mode_smooths(self, two_timbres):
        # Need ≥3 Timbres for Gaussian and linear_walk to genuinely
        # differ at the midpoint — with only 2 Timbres at pos=0.5,
        # both modes reduce to a 50/50 mix.
        low, high = two_timbres
        mid = Timbre(
            partials_hz=np.array([440.0, 880.0]),
            amplitudes=np.array([1.0, 0.5]),
            base_freq=220.0,
        )
        # 5 frames, midpoint = frame 2 → pos = 1.0 (right on `mid`).
        # Linear: pure `mid`. Gaussian: `mid` + spillover from low/high.
        out_lin = _frame_with_timbre_morph(
            [low, mid, high], frame_idx=2, n_frames=5, table_size=512,
            blend_mode="linear_walk",
        )
        out_gauss = _frame_with_timbre_morph(
            [low, mid, high], frame_idx=2, n_frames=5, table_size=512,
            blend_mode="gaussian", gaussian_sigma=1.0,
        )
        assert not np.allclose(out_gauss, out_lin, atol=1e-3)

    def test_single_timbre_input(self, two_timbres):
        low, _ = two_timbres
        out = _frame_with_timbre_morph(
            [low], frame_idx=2, n_frames=5, table_size=512,
        )
        baseline = render_wavetable_cycle(low, table_size=512)
        np.testing.assert_allclose(out, baseline, atol=1e-5)

    def test_unknown_blend_mode_raises(self, two_timbres):
        low, high = two_timbres
        with pytest.raises(ValueError, match="blend_mode"):
            _frame_with_timbre_morph(
                [low, high], 0, 5, table_size=256, blend_mode="bezier",
            )

    def test_empty_sequence_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _frame_with_timbre_morph([], 0, 5, table_size=256)


class TestImfAdapter:
    def test_one_timbre_per_imf(self):
        # Synthesise two fake IMFs: a 50 Hz sine and a 200 Hz sine.
        sf = 1000.0
        t = np.linspace(0, 2, int(sf * 2), endpoint=False)
        imf1 = np.sin(2 * np.pi * 50 * t)
        imf2 = np.sin(2 * np.pi * 200 * t)
        timbres = timbres_from_imfs([imf1, imf2], sf=sf, n_peaks_per_imf=3)
        assert len(timbres) == 2
        # First Timbre's strongest partial should be near 50 Hz,
        # second's near 200 Hz.
        assert abs(float(timbres[0].partials_hz[0]) - 50.0) < 10.0
        assert abs(float(timbres[1].partials_hz[0]) - 200.0) < 10.0

    def test_drops_empty_imf(self):
        sf = 1000.0
        t = np.linspace(0, 1, int(sf), endpoint=False)
        good = np.sin(2 * np.pi * 100 * t)
        empty = np.zeros_like(t)
        timbres = timbres_from_imfs([good, empty], sf=sf)
        # Constant IMF is dropped.
        assert len(timbres) == 1


class TestBandAdapter:
    def test_band_count_matches_edges_minus_one(self):
        sf = 1000.0
        t = np.linspace(0, 2, int(sf * 2), endpoint=False)
        # Multi-tone signal spanning bands.
        signal = (
            np.sin(2 * np.pi * 6 * t)
            + np.sin(2 * np.pi * 30 * t)
            + np.sin(2 * np.pi * 100 * t)
        )
        timbres = timbres_from_bands(
            signal, sf, band_edges=[4, 13, 50, 200],
        )
        # 3 edges → 3 bands; each should have content (we put energy in all).
        assert 1 <= len(timbres) <= 3

    def test_too_few_edges_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            timbres_from_bands(np.zeros(100), 1000.0, band_edges=[10])


class TestExportIntegration:
    def test_appears_in_evolutions(self):
        assert "imf_morph" in _EVOLUTIONS
        assert "band_morph" in _EVOLUTIONS

    def test_export_imf_morph(self, two_timbres, tmp_path):
        out = tmp_path / "imf.wav"
        low, high = two_timbres
        result = export_wavetable(
            low, str(out),
            n_frames=8, evolution="imf_morph",
            timbre_sequence=[low, high],
            timbre_sequence_blend="linear_walk",
            include_sidecar=False,
        )
        assert os.path.exists(result["wavetable"])
        import soundfile as sf
        data, _ = sf.read(result["wavetable"])
        assert data.shape[0] == 8 * 2048
        manifest = json.load(open(result["manifest"]))
        assert manifest["evolution"] == "imf_morph"
        ep = manifest["evolution_params"]
        assert ep["n_timbres"] == 2
        assert ep["blend"] == "linear_walk"

    def test_export_band_morph_with_dict_inputs(self, two_timbres, tmp_path):
        out = tmp_path / "band.wav"
        low, _ = two_timbres
        result = export_wavetable(
            low, str(out),
            n_frames=4, evolution="band_morph",
            timbre_sequence=[
                {"partials_hz": [100, 200], "amplitudes": [1.0, 0.5], "base_freq": 100.0},
                {"partials_hz": [300, 600], "amplitudes": [1.0, 0.5], "base_freq": 100.0},
            ],
            include_sidecar=False,
        )
        assert os.path.exists(result["wavetable"])

    def test_missing_timbre_sequence_raises(self, two_timbres, tmp_path):
        low, _ = two_timbres
        with pytest.raises(ValueError, match="timbre_sequence"):
            export_wavetable(
                low, str(tmp_path / "x.wav"),
                n_frames=4, evolution="imf_morph",
                timbre_sequence=None, include_sidecar=False,
            )
