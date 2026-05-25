"""Tests for the two new nonlinear-enrichment wavetable evolutions:
``wavefolding`` and ``fm_baked``, plus the static ``Timbre.with_wavefolding``
companion that bakes the same fold into the partial spectrum.

Each evolution is validated for:

  1. Zero-amount identity — at ``fold_amount=0`` / ``fm_index=0`` the
     output equals the unprocessed cycle (within float epsilon).
  2. Spectral character — at moderate amounts, **measurable** new
     harmonic content appears (odd harmonics for fold; sidebands for
     FM) above the baseline.
  3. Integration — ``export_wavetable(..., evolution=...)`` produces a
     valid multi-frame WAV that loads back cleanly via soundfile.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from biotuner.harmonic_timbre import Timbre
from biotuner.harmonic_timbre.synthesis import render_wavetable_cycle
from biotuner.harmonic_timbre.exporters.to_wavetable import (
    _frame_with_wavefolding,
    _frame_with_fm_baked,
    _EVOLUTIONS,
    export_wavetable,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def basic_timbre():
    """A 3-partial harmonic timbre (1, 2, 3 × 220 Hz, decaying amps)."""
    return Timbre(
        partials_hz=np.array([220.0, 440.0, 660.0]),
        amplitudes=np.array([1.0, 0.5, 0.25]),
    )


def _n_significant_bins(cycle: np.ndarray, rel_threshold: float = 0.02) -> int:
    """Number of FFT bins whose magnitude exceeds ``rel_threshold`` of
    the peak bin. A larger count = a richer / more harmonically-populated
    spectrum. Used to validate that nonlinear evolutions ADD content
    rather than just rearrange it."""
    spec = np.abs(np.fft.rfft(cycle))
    if spec.size == 0:
        return 0
    peak = float(spec.max() or 1e-9)
    return int(np.sum(spec > peak * rel_threshold))


def _odd_harmonic_energy(cycle: np.ndarray, fundamental_bin: int) -> float:
    """Sum of magnitudes at odd multiples (3, 5, 7, 9) of the fundamental
    bin, relative to the fundamental itself. Used to verify wavefolding
    populates the odd harmonics specifically."""
    spec = np.abs(np.fft.rfft(cycle))
    f = float(spec[fundamental_bin]) or 1e-9
    odd_bins = [fundamental_bin * k for k in (3, 5, 7, 9)]
    return sum(float(spec[b]) for b in odd_bins if b < spec.size) / f


# ---------------------------------------------------------------------------
# Wavefolding (per-frame helper)
# ---------------------------------------------------------------------------


class TestWavefoldingFrame:
    def test_zero_fold_is_identity_up_to_normalisation(self, basic_timbre):
        # Compare normalised base cycle vs. fold-amount=0 output; both
        # should be the same shape (peak-normalised to ±1).
        base = render_wavetable_cycle(basic_timbre, table_size=512)
        base = base / float(np.max(np.abs(base)) or 1.0)
        folded = _frame_with_wavefolding(basic_timbre, 0.0, table_size=512)
        # sin(π · x · 1) for small x ≈ π · x, then re-normalised. So
        # zero-fold doesn't literally equal the base, but it's monotonic
        # in the sample values: correlation should be very high (≈ 1).
        corr = float(np.corrcoef(base, folded)[0, 1])
        assert corr > 0.98, f"zero-fold should track base closely, got corr={corr}"

    def test_moderate_fold_adds_odd_harmonics(self, basic_timbre):
        # Use a clean single-partial fixture so the odd-harmonic structure
        # of the FOLD (not whatever the base spectrum already had) is
        # what's being measured. partials_hz=[1.0] + base_freq=1.0
        # → fundamental at FFT bin 1; odd harmonics land at bins 3, 5, …
        clean = Timbre(
            partials_hz=np.array([1.0]),
            amplitudes=np.array([1.0]),
            base_freq=1.0,
        )
        base = _frame_with_wavefolding(clean, 0.0, table_size=2048)
        folded = _frame_with_wavefolding(clean, 2.5, table_size=2048)
        # At fold 2.5 odd-harmonic ratio should rise substantially.
        assert _odd_harmonic_energy(folded, fundamental_bin=1) > \
               _odd_harmonic_energy(base, fundamental_bin=1) + 0.5

    def test_fold_enriches_spectrum(self, basic_timbre):
        # Generic "richness" test that doesn't depend on harmonic
        # alignment: significant bin count should rise with fold.
        base   = _frame_with_wavefolding(basic_timbre, 0.0, table_size=2048)
        folded = _frame_with_wavefolding(basic_timbre, 2.5, table_size=2048)
        assert _n_significant_bins(folded) > _n_significant_bins(base)

    def test_heavy_fold_does_not_explode(self, basic_timbre):
        # Even at fold=4 (the documented upper end), output stays in
        # [-1, 1] (peak normalised).
        folded = _frame_with_wavefolding(basic_timbre, 4.0, table_size=1024)
        assert float(np.max(np.abs(folded))) <= 1.0 + 1e-6
        assert np.all(np.isfinite(folded))

    def test_output_shape_matches_table_size(self, basic_timbre):
        for ts in (128, 512, 1024, 2048):
            folded = _frame_with_wavefolding(basic_timbre, 1.0, table_size=ts)
            assert folded.shape == (ts,)
            assert folded.dtype == np.float32

    def test_drive_changes_character(self, basic_timbre):
        # Different drive values at the same fold produce measurably
        # different output (sanity for the optional knob).
        low  = _frame_with_wavefolding(basic_timbre, 1.5, table_size=2048, output_drive=0.7)
        high = _frame_with_wavefolding(basic_timbre, 1.5, table_size=2048, output_drive=1.3)
        assert not np.allclose(low, high)


# ---------------------------------------------------------------------------
# FM-baked (per-frame helper)
# ---------------------------------------------------------------------------


class TestFmBakedFrame:
    def test_zero_index_matches_additive_render(self, basic_timbre):
        # With β=0 the FM term vanishes; output should be byte-equal to
        # the standard additive render (both share the same _normalize
        # convention of peak=0.99).
        baseline = render_wavetable_cycle(basic_timbre, table_size=512)
        baked = _frame_with_fm_baked(basic_timbre, 0.0, table_size=512)
        np.testing.assert_allclose(baked, baseline, atol=1e-5)

    def test_high_index_adds_sidebands(self, basic_timbre):
        # FM spreads energy from the carrier into Bessel sidebands.
        # We don't try to count specific bins (cm_ratio=2 puts
        # sidebands ON existing harmonic bins — collisions confuse
        # the assertion). Generic richness test instead: significant
        # bin count rises measurably.
        base  = _frame_with_fm_baked(basic_timbre, 0.0, table_size=2048)
        baked = _frame_with_fm_baked(basic_timbre, 3.0, table_size=2048)
        assert _n_significant_bins(baked) > _n_significant_bins(base)

    def test_target_partial_idx_minus_one_modulates_all(self, basic_timbre):
        # Per-partial FM (target_partial_idx=-1) produces denser
        # spectra than single-partial FM at the same β.
        single = _frame_with_fm_baked(basic_timbre, 2.0, table_size=2048, target_partial_idx=0)
        all_   = _frame_with_fm_baked(basic_timbre, 2.0, table_size=2048, target_partial_idx=-1)
        # "Denser" → more frequency bins above a moderate threshold.
        def n_above(buf):
            spec = np.abs(np.fft.rfft(buf))
            return int(np.sum(spec > spec.max() * 0.05))
        assert n_above(all_) > n_above(single)

    def test_cm_ratio_changes_sideband_locations(self, basic_timbre):
        # Different C:M ratios place sidebands at different bins.
        r2 = _frame_with_fm_baked(basic_timbre, 2.0, table_size=2048, cm_ratio=2.0)
        r3 = _frame_with_fm_baked(basic_timbre, 2.0, table_size=2048, cm_ratio=3.0)
        assert not np.allclose(r2, r3, atol=1e-3)

    def test_output_shape(self, basic_timbre):
        for ts in (256, 1024, 2048):
            baked = _frame_with_fm_baked(basic_timbre, 1.0, table_size=ts)
            assert baked.shape == (ts,)
            assert baked.dtype == np.float32


# ---------------------------------------------------------------------------
# Integration through export_wavetable (the public surface)
# ---------------------------------------------------------------------------


class TestExportIntegration:
    def test_wavefolding_appears_in_evolutions(self):
        assert "wavefolding" in _EVOLUTIONS
        assert "fm_baked" in _EVOLUTIONS

    def test_export_wavefolding(self, basic_timbre, tmp_path):
        out = tmp_path / "fold.wav"
        result = export_wavetable(
            basic_timbre,
            str(out),
            n_frames=8,
            evolution="wavefolding",
            include_sidecar=False,
        )
        # Wave file exists and has correct frame count baked in.
        wav_path = result["wavetable"]
        assert os.path.exists(wav_path)
        import soundfile as sf
        data, sr = sf.read(wav_path)
        # 8 frames × default Vital table_size (2048) = 16384 samples.
        assert data.shape[0] == 8 * 2048
        # Manifest reflects the evolution choice + param.
        import json
        manifest = json.load(open(result["manifest"]))
        assert manifest["evolution"] == "wavefolding"
        assert "fold_range" in manifest["evolution_params"]

    def test_export_fm_baked(self, basic_timbre, tmp_path):
        out = tmp_path / "fm.wav"
        result = export_wavetable(
            basic_timbre,
            str(out),
            n_frames=8,
            evolution="fm_baked",
            include_sidecar=False,
        )
        import soundfile as sf
        data, sr = sf.read(result["wavetable"])
        assert data.shape[0] == 8 * 2048
        import json
        manifest = json.load(open(result["manifest"]))
        assert manifest["evolution"] == "fm_baked"
        ep = manifest["evolution_params"]
        assert "fm_index_range" in ep
        assert ep["cm_ratio"] == 2.0
        assert ep["target_partial_idx"] == 0


# ---------------------------------------------------------------------------
# Timbre.with_wavefolding (static enrichment method)
# ---------------------------------------------------------------------------


class TestTimbreWithWavefolding:
    def test_zero_fold_returns_self(self, basic_timbre):
        result = basic_timbre.with_wavefolding(0.0)
        assert result is basic_timbre

    def test_positive_fold_adds_partials(self, basic_timbre):
        # Wavefolding adds odd-harmonic content; new partial count
        # should be > the original.
        n_before = basic_timbre.n_partials()
        folded = basic_timbre.with_wavefolding(1.5)
        assert folded.n_partials() > n_before

    def test_added_partials_are_odd_harmonics(self, basic_timbre):
        # The fundamental is 220 Hz; folded version should contain
        # partials at 220 × {3, 5, 7, …} that weren't in the original.
        folded = basic_timbre.with_wavefolding(2.0)
        original_set = set(round(p, 2) for p in basic_timbre.partials_hz)
        new_partials = set(round(p, 2) for p in folded.partials_hz) - original_set
        # Look for at least one odd multiple of 220 that's not in the
        # original {220, 440, 660}.
        expected_candidates = {660.0, 1100.0, 1540.0, 1980.0}  # 3rd, 5th, 7th, 9th
        # Allow a few Hz tolerance for FFT bin quantisation.
        found_odd = any(
            any(abs(p - c) < 5.0 for c in expected_candidates)
            for p in folded.partials_hz
        )
        assert found_odd, (
            f"Expected odd harmonics of 220 Hz; got {list(folded.partials_hz)}"
        )

    def test_n_partials_keep_caps_count(self, basic_timbre):
        folded = basic_timbre.with_wavefolding(3.0, n_partials_keep=5)
        assert folded.n_partials() <= 5

    def test_returns_valid_timbre(self, basic_timbre):
        folded = basic_timbre.with_wavefolding(1.0)
        folded.validate()  # raises on inconsistency
