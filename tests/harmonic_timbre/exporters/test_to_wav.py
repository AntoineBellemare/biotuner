"""Tests for biotuner.harmonic_timbre.exporters.to_wav."""

from __future__ import annotations

import json
import os

import numpy as np
import soundfile as sf

from biotuner.harmonic_timbre.exporters import export_wav_pack


def test_wav_pack_creates_one_wav_per_pitch(tmp_path, matched_timbre, short_pitches):
    out = export_wav_pack(
        matched_timbre, str(tmp_path),
        bundle_name="b",
        reference_pitches_midi=short_pitches,
        duration=0.25, samplerate=22050,
    )
    assert len(out["wavs"]) == len(short_pitches)
    for w in out["wavs"]:
        assert os.path.exists(w)


def test_wav_pack_manifest_records_pitch_table(tmp_path, matched_timbre, short_pitches):
    out = export_wav_pack(
        matched_timbre, str(tmp_path),
        bundle_name="b",
        reference_pitches_midi=short_pitches,
        duration=0.2, samplerate=22050,
    )
    manifest = json.load(open(out["manifest"]))
    assert manifest["format"] == "biotuner_wav_pack"
    assert len(manifest["samples"]) == len(short_pitches)
    notes_in_manifest = [s["midi_note"] for s in manifest["samples"]]
    assert sorted(notes_in_manifest) == sorted(short_pitches)


def test_wav_pack_audio_has_correct_duration(tmp_path, matched_timbre, short_pitches):
    sr = 22050
    dur = 0.5
    out = export_wav_pack(
        matched_timbre, str(tmp_path),
        bundle_name="b",
        reference_pitches_midi=short_pitches,
        duration=dur, samplerate=sr,
    )
    for w in out["wavs"]:
        audio, sr_read = sf.read(w)
        assert sr_read == sr
        assert audio.shape[0] == int(dur * sr)


def test_wav_pack_includes_sidecar_by_default(tmp_path, matched_timbre, short_pitches):
    out = export_wav_pack(
        matched_timbre, str(tmp_path),
        bundle_name="b",
        reference_pitches_midi=short_pitches,
        duration=0.2, samplerate=22050,
    )
    assert "sidecar" in out
    assert "metadata" in out["sidecar"]
    assert os.path.exists(out["sidecar"]["metadata"])
