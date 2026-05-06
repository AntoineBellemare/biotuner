"""Tests for biotuner.harmonic_timbre.exporters.to_surge."""

from __future__ import annotations

import json
import os

import pytest

from biotuner.harmonic_timbre.exporters import export_surge_bundle


def test_surge_bundle_writes_all_files(tmp_path, matched_timbre):
    out = export_surge_bundle(
        matched_timbre, str(tmp_path),
        bundle_name="surge_bun",
        n_wavetable_frames=4,
    )
    for key in ("wavetable", "scl", "kbm", "readme", "manifest"):
        assert key in out
        assert os.path.exists(out[key]), f"missing {key}"


def test_surge_bundle_readme_mentions_loading_steps(tmp_path, matched_timbre):
    out = export_surge_bundle(matched_timbre, str(tmp_path), bundle_name="b")
    readme = open(out["readme"]).read()
    assert "Surge XT" in readme
    assert "Wavetable" in readme
    assert ".scl" in readme
    assert ".kbm" in readme


def test_surge_bundle_manifest_lists_files(tmp_path, matched_timbre):
    out = export_surge_bundle(matched_timbre, str(tmp_path), bundle_name="b")
    manifest = json.load(open(out["manifest"]))
    assert manifest["format"] == "biotuner_surge_bundle"
    files = manifest["files"]
    assert "wavetable" in files
    assert "scl" in files
    assert "kbm" in files
    assert "readme" in files


def test_surge_bundle_requires_matched_tuning(tmp_path):
    from biotuner.harmonic_timbre import Timbre
    t = Timbre(partials_hz=[100.0, 200.0], amplitudes=[1.0, 0.5])  # no matched_tuning
    with pytest.raises(ValueError, match="matched_tuning"):
        export_surge_bundle(t, str(tmp_path), bundle_name="x")
