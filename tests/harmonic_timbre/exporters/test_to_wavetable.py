"""Tests for biotuner.harmonic_timbre.exporters.to_wavetable."""

from __future__ import annotations

import os

import numpy as np
import pytest
import soundfile as sf

from biotuner.harmonic_timbre.exporters import (
    export_wavetable,
    export_wavetable_from_imfs,
    export_wavetable_morph,
)


def test_single_cycle_wavetable_size(tmp_path, matched_timbre):
    out = export_wavetable(
        matched_timbre, str(tmp_path / "wt"),
        n_frames=1, synth_target="vital",
    )
    audio, sr = sf.read(out["wavetable"])
    assert audio.size == 2048


def test_multi_frame_wavetable_size(tmp_path, matched_timbre):
    out = export_wavetable(
        matched_timbre, str(tmp_path / "wt"),
        n_frames=4, synth_target="vital",
    )
    audio, sr = sf.read(out["wavetable"])
    assert audio.size == 4 * 2048


def test_synth_targets(tmp_path, matched_timbre):
    for target in ("vital", "serum", "surge", "generic"):
        out = export_wavetable(
            matched_timbre, str(tmp_path / target),
            n_frames=1, synth_target=target,
        )
        assert os.path.exists(out["wavetable"])


def test_wavetable_recovers_partials_via_fft(tmp_path):
    """One cycle of a 100/200 Hz timbre (base_freq=100) → FFT bins 1, 2."""
    from biotuner.harmonic_timbre import Timbre
    t = Timbre(
        partials_hz=[100.0, 200.0, 300.0],
        amplitudes=[1.0, 0.7, 0.4],
        base_freq=100.0,
    )
    out = export_wavetable(t, str(tmp_path / "rec"), n_frames=1, synth_target="generic")
    audio, sr = sf.read(out["wavetable"])
    spec = np.abs(np.fft.rfft(audio[:2048]))
    for k in (1, 2, 3):
        assert spec[k] > spec.mean() * 50


def test_unknown_synth_target_raises(tmp_path, matched_timbre):
    with pytest.raises(ValueError, match="unknown synth_target"):
        export_wavetable(matched_timbre, str(tmp_path / "x"), synth_target="cowbell")


# ---------------------------------------------------------------------------
# Multi-frame evolution modes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("evolution", ["tilt", "harmonic_buildup", "amp_morph", "phase_sweep"])
def test_evolution_modes_produce_correct_length(tmp_path, matched_timbre, evolution):
    n_frames = 8
    out = export_wavetable(
        matched_timbre, str(tmp_path / f"wt_{evolution}"),
        n_frames=n_frames, synth_target="vital",
        evolution=evolution,
    )
    audio, _ = sf.read(out["wavetable"])
    assert audio.size == n_frames * 2048


def test_unknown_evolution_raises(tmp_path, matched_timbre):
    with pytest.raises(ValueError, match="unknown evolution"):
        export_wavetable(
            matched_timbre, str(tmp_path / "x"),
            n_frames=4, evolution="not-real",
        )


def test_harmonic_buildup_increases_spectral_richness(tmp_path, matched_timbre):
    """Frame 0 should have only the fundamental; the last frame should have
    every partial. Each frame is peak-normalized, so we check spectral
    richness via the count of FFT bins above a threshold, not loudness."""
    out = export_wavetable(
        matched_timbre, str(tmp_path / "buildup"),
        n_frames=8, evolution="harmonic_buildup",
    )
    audio, _ = sf.read(out["wavetable"])

    def _strong_bins(frame: np.ndarray) -> int:
        spec = np.abs(np.fft.rfft(frame))
        spec_n = spec / max(spec.max(), 1e-9)
        return int((spec_n > 0.05).sum())

    bins_first = _strong_bins(audio[:2048])
    bins_last = _strong_bins(audio[-2048:])
    assert bins_first < bins_last


# ---------------------------------------------------------------------------
# IMF-derived wavetable
# ---------------------------------------------------------------------------

def test_wavetable_from_imfs_creates_one_frame_per_imf(tmp_path):
    """3 IMFs → 3-frame wavetable (3 * 2048 samples)."""
    sr_imf = 1000
    t = np.arange(2 * sr_imf) / sr_imf
    imfs = [
        np.sin(2 * np.pi * 8 * t),    # 8 Hz alpha-like
        np.sin(2 * np.pi * 25 * t),   # 25 Hz beta-like
        np.sin(2 * np.pi * 50 * t),   # 50 Hz gamma-like
    ]
    out = export_wavetable_from_imfs(imfs, str(tmp_path / "imfs"))
    assert out["n_frames"] == 3
    audio, _ = sf.read(out["wavetable"])
    assert audio.size == 3 * 2048


def test_wavetable_from_imfs_strategies(tmp_path):
    sr_imf = 1000
    t = np.arange(sr_imf) / sr_imf
    imfs = [np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 20 * t)]
    for strategy in ("first_cycle", "avg_cycles", "whole_resampled"):
        out = export_wavetable_from_imfs(
            imfs, str(tmp_path / f"imfs_{strategy}"),
            cycle_strategy=strategy,
        )
        assert os.path.exists(out["wavetable"])


def test_wavetable_from_imfs_empty_raises(tmp_path):
    with pytest.raises(ValueError, match="empty IMFs"):
        export_wavetable_from_imfs([], str(tmp_path / "x"))


# ---------------------------------------------------------------------------
# Morph
# ---------------------------------------------------------------------------

def test_wavetable_morph_endpoints_match_inputs(tmp_path):
    """First and last frames should approximate the two source timbres."""
    from biotuner.harmonic_timbre import Timbre
    a = Timbre(partials_hz=[100.0, 200.0], amplitudes=[1.0, 0.5], base_freq=100.0)
    b = Timbre(partials_hz=[100.0, 300.0], amplitudes=[1.0, 0.5], base_freq=100.0)
    out = export_wavetable_morph(a, b, str(tmp_path / "morph"), n_frames=16)
    audio, _ = sf.read(out["wavetable"])
    assert audio.size == 16 * 2048
    # Frame 0 should have a strong harmonic 2; frame -1 should have a strong harmonic 3
    f0 = audio[:2048]
    flast = audio[-2048:]
    spec0 = np.abs(np.fft.rfft(f0))
    speclast = np.abs(np.fft.rfft(flast))
    assert spec0[2] > spec0[3]      # frame 0: 200/100 -> bin 2 dominates
    assert speclast[3] > speclast[2]  # frame -1: 300/100 -> bin 3 dominates


def test_wavetable_morph_n_frames_minimum(tmp_path, matched_timbre):
    with pytest.raises(ValueError, match="n_frames must be"):
        export_wavetable_morph(
            matched_timbre, matched_timbre, str(tmp_path / "x"), n_frames=1,
        )
