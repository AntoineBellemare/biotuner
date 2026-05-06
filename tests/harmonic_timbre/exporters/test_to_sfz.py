"""Tests for biotuner.harmonic_timbre.exporters.to_sfz."""

from __future__ import annotations

import os
import re

from biotuner.harmonic_timbre.exporters import export_sfz


def test_sfz_writes_instrument_and_samples(tmp_path, matched_timbre, short_pitches):
    out = export_sfz(
        matched_timbre, str(tmp_path),
        bundle_name="bun",
        reference_pitches_midi=short_pitches,
        duration=0.25, samplerate=22050,
    )
    # Files exist
    assert os.path.exists(out["sfz"])
    for w in out["wavs"]:
        assert os.path.exists(w)
    # Manifest exists
    assert os.path.exists(out["manifest"])


def test_sfz_contains_required_opcodes(tmp_path, matched_timbre, short_pitches):
    out = export_sfz(
        matched_timbre, str(tmp_path),
        bundle_name="bun",
        reference_pitches_midi=short_pitches,
        duration=0.2, samplerate=22050,
    )
    text = open(out["sfz"]).read()
    # required SFZ opcodes for a region-mapped instrument
    for opcode in ("<global>", "<region>", "sample=", "lokey=", "hikey=", "pitch_keycenter="):
        assert opcode in text, f"missing opcode {opcode!r} in SFZ"


def test_sfz_pitch_keycenters_match_input_pitches(tmp_path, matched_timbre, short_pitches):
    out = export_sfz(
        matched_timbre, str(tmp_path),
        bundle_name="bun",
        reference_pitches_midi=short_pitches,
        duration=0.2, samplerate=22050,
    )
    text = open(out["sfz"]).read()
    found = sorted(int(m) for m in re.findall(r"pitch_keycenter=(\d+)", text))
    assert found == sorted(short_pitches)


def test_sfz_writes_companion_tuning_files(tmp_path, matched_timbre, short_pitches):
    out = export_sfz(
        matched_timbre, str(tmp_path),
        bundle_name="bun",
        reference_pitches_midi=short_pitches,
        duration=0.2, samplerate=22050,
    )
    assert "scl" in out and os.path.exists(out["scl"])
    assert "kbm" in out and os.path.exists(out["kbm"])


def test_sfz_skips_tuning_when_disabled(tmp_path, matched_timbre, short_pitches):
    out = export_sfz(
        matched_timbre, str(tmp_path),
        bundle_name="bun",
        reference_pitches_midi=short_pitches,
        duration=0.2, samplerate=22050,
        write_tuning_files=False,
    )
    assert "scl" not in out or not out.get("scl")
