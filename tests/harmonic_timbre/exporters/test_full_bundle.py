"""Tests for export_full_bundle convenience wrapper."""

from __future__ import annotations

import json
import os

import pytest

from biotuner.harmonic_timbre.exporters import export_full_bundle


def test_full_bundle_creates_all_subdirs(tmp_path, matched_timbre):
    out = export_full_bundle(
        matched_timbre, str(tmp_path),
        bundle_name="kit",
        reference_pitches_midi=[48, 60, 72],
        duration=0.25, samplerate=22050,
        n_wavetable_frames=2,
    )
    assert "wav" in out
    assert "sfz" in out
    assert "wavetable" in out
    assert "surge" in out
    assert "csound" in out
    assert "supercollider" in out
    assert "tuning" in out
    assert "sidecar" in out
    assert "manifest" in out
    # subdirs are present
    for sub in ("wav", "sfz", "wavetable", "surge", "csound", "supercollider", "tuning", "sidecar"):
        assert os.path.isdir(os.path.join(str(tmp_path), sub))


def test_full_bundle_manifest_lists_formats(tmp_path, matched_timbre):
    out = export_full_bundle(
        matched_timbre, str(tmp_path),
        bundle_name="kit",
        reference_pitches_midi=[48, 60, 72],
        duration=0.25, samplerate=22050,
        n_wavetable_frames=2,
    )
    manifest = json.load(open(out["manifest"]))
    assert manifest["format"] == "biotuner_full_bundle"
    assert sorted(manifest["formats"]) == sorted([
        "wav", "sfz", "wavetable", "surge", "csound", "supercollider", "tuning",
    ])
    assert manifest["timbre"]["matched_tuning"] is not None


def test_full_bundle_subset_of_formats(tmp_path, matched_timbre):
    """Specifying a subset only writes those subfolders."""
    out = export_full_bundle(
        matched_timbre, str(tmp_path),
        bundle_name="kit",
        reference_pitches_midi=[60],
        duration=0.2, samplerate=22050,
        formats=["sfz", "csound"],
    )
    # only those + sidecar/manifest
    assert "sfz" in out and "csound" in out
    assert "wavetable" not in out
    assert "supercollider" not in out
    assert os.path.isdir(os.path.join(str(tmp_path), "sfz"))
    assert os.path.isdir(os.path.join(str(tmp_path), "csound"))
    assert not os.path.isdir(os.path.join(str(tmp_path), "wavetable"))


def test_full_bundle_unknown_format_raises(tmp_path, matched_timbre):
    with pytest.raises(ValueError, match="unknown format"):
        export_full_bundle(
            matched_timbre, str(tmp_path),
            formats=["wav", "not-a-real-format"],
        )


# ---------------------------------------------------------------------------
# export_full_bundle_sequence
# ---------------------------------------------------------------------------

def test_full_bundle_sequence_writes_morph_wavetable_and_manifest(tmp_path):
    from biotuner.harmonic_timbre import timbre_sequence_from_ratio_frames
    from biotuner.harmonic_timbre.exporters import export_full_bundle_sequence

    seq = timbre_sequence_from_ratio_frames([
        [1.0, 5/4, 3/2, 2.0],
        [1.0, 6/5, 3/2, 2.0],
        [1.0, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8, 2.0],
    ])
    out = export_full_bundle_sequence(
        seq, str(tmp_path / "seq_bundle"),
        bundle_name="testseq",
        per_frame_formats=[],   # skip per-frame to keep test fast
        sequence_audio=True,
        sequence_audio_frame_duration=0.1,
    )
    assert os.path.exists(out["morph_wavetable"])
    assert os.path.exists(out["manifest"])
    assert os.path.exists(out["sequence_audio"])
    # morph wavetable = 3 frames * 2048 samples * 4 bytes (float32)
    import soundfile as sf
    audio, sr = sf.read(out["morph_wavetable"])
    assert audio.size == 3 * 2048


def test_full_bundle_sequence_per_frame_subdirs(tmp_path):
    from biotuner.harmonic_timbre import timbre_sequence_from_ratio_frames
    from biotuner.harmonic_timbre.exporters import export_full_bundle_sequence

    seq = timbre_sequence_from_ratio_frames([
        [1.0, 3/2, 2.0],
        [1.0, 4/3, 2.0],
    ])
    out = export_full_bundle_sequence(
        seq, str(tmp_path / "seq_bundle"),
        bundle_name="testseq",
        per_frame_formats=["tuning"],
        sequence_audio=False,
    )
    # one frame_NN dir per frame
    frames_dir = os.path.join(str(tmp_path / "seq_bundle"), "frames")
    assert os.path.isdir(frames_dir)
    subdirs = sorted(os.listdir(frames_dir))
    assert len(subdirs) == 2
    assert subdirs[0].startswith("frame_")
    # each frame's tuning subdir has .scl + .kbm
    for sub in subdirs:
        tuning_dir = os.path.join(frames_dir, sub, "tuning")
        assert os.path.isdir(tuning_dir)
        files = os.listdir(tuning_dir)
        assert any(f.endswith(".scl") for f in files)
        assert any(f.endswith(".kbm") for f in files)


def test_full_bundle_sequence_manifest_records_frame_metadata(tmp_path):
    from biotuner.harmonic_timbre import timbre_sequence_from_ratio_frames
    from biotuner.harmonic_timbre.exporters import export_full_bundle_sequence

    frames = [
        [1.0, 5/4, 3/2, 2.0],
        [1.0, 6/5, 3/2, 2.0],
    ]
    seq = timbre_sequence_from_ratio_frames(frames)
    out = export_full_bundle_sequence(
        seq, str(tmp_path / "x"),
        bundle_name="m",
        per_frame_formats=[],
        sequence_audio=False,
    )
    m = json.load(open(out["manifest"]))
    assert m["format"] == "biotuner_full_bundle_sequence"
    assert m["n_frames"] == 2
    for i, frame_entry in enumerate(m["frames"]):
        assert frame_entry["index"] == i
        assert frame_entry["matched_tuning"] == frames[i]


def test_full_bundle_sequence_works_with_markov_walk(tmp_path):
    """End-to-end: a Markov walk → TimbreSequence → bundle."""
    import numpy as np
    from biotuner.harmonic_timbre import timbre_sequence_from_markov_walk
    from biotuner.harmonic_timbre.exporters import export_full_bundle_sequence

    P = np.array([[0.5, 0.5], [0.5, 0.5]])
    state_ratios = {
        0: [1.0, 5/4, 3/2, 2.0],
        1: [1.0, 6/5, 3/2, 2.0],
    }
    seq = timbre_sequence_from_markov_walk(P, state_ratios, n_steps=4, start=0, seed=0)
    out = export_full_bundle_sequence(
        seq, str(tmp_path / "markov_bundle"),
        bundle_name="mw",
        per_frame_formats=[],
        sequence_audio=False,
    )
    m = json.load(open(out["manifest"]))
    assert m["n_frames"] == 4
    # Each visited state's matched_tuning should match one of the two
    # state_ratios entries.
    for entry in m["frames"]:
        assert entry["matched_tuning"] in [state_ratios[0], state_ratios[1]]


def test_full_bundle_sequence_rejects_non_sequence(tmp_path, matched_timbre):
    from biotuner.harmonic_timbre.exporters import export_full_bundle_sequence
    with pytest.raises(TypeError, match="TimbreSequence"):
        export_full_bundle_sequence(matched_timbre, str(tmp_path / "x"))


def test_full_bundle_sequence_rejects_unknown_per_frame_format(tmp_path):
    from biotuner.harmonic_timbre import timbre_sequence_from_ratio_frames
    from biotuner.harmonic_timbre.exporters import export_full_bundle_sequence
    seq = timbre_sequence_from_ratio_frames([[1.0, 3/2, 2.0]])
    with pytest.raises(ValueError, match="unknown per_frame_formats"):
        export_full_bundle_sequence(
            seq, str(tmp_path / "x"),
            per_frame_formats=["sfz", "rainbow"],
        )
