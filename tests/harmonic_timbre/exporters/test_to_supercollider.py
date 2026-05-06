"""Tests for biotuner.harmonic_timbre.exporters.to_supercollider."""

from __future__ import annotations

import os

from biotuner.harmonic_timbre.exporters import export_supercollider


def test_sc_creates_scd(tmp_path, matched_timbre):
    out = export_supercollider(
        matched_timbre, str(tmp_path / "patch"),
        synthdef_name="biotunerTest",
    )
    assert os.path.exists(out["scd"])


def test_sc_contains_synthdef_and_tuning(tmp_path, matched_timbre):
    out = export_supercollider(matched_timbre, str(tmp_path / "patch"))
    text = open(out["scd"]).read()
    assert "SynthDef" in text
    assert ".add" in text
    assert "Tuning.new" in text
    assert "Scale.new" in text


def test_sc_pbind_demo_when_enabled(tmp_path, matched_timbre):
    out = export_supercollider(
        matched_timbre, str(tmp_path / "patch"),
        include_demo=True,
    )
    text = open(out["scd"]).read()
    assert "Pbind" in text


def test_sc_no_pbind_when_disabled(tmp_path, matched_timbre):
    out = export_supercollider(
        matched_timbre, str(tmp_path / "patch"),
        include_demo=False,
    )
    text = open(out["scd"]).read()
    # No active Pbind() call. The skipped-demo placeholder comment is the
    # only place 'Pbind' appears (and it's behind '//').
    code_only = "\n".join(
        line for line in text.splitlines() if not line.strip().startswith("//")
    )
    assert "Pbind(" not in code_only


def test_sc_balanced_braces(tmp_path, matched_timbre):
    out = export_supercollider(matched_timbre, str(tmp_path / "patch"))
    text = open(out["scd"]).read()
    # strip comment lines first
    code_lines = [l for l in text.splitlines() if not l.strip().startswith("//")]
    code = "\n".join(code_lines)
    assert code.count("(") == code.count(")"), "unbalanced parentheses in .scd"
    assert code.count("{") == code.count("}"), "unbalanced curly braces in .scd"
    assert code.count("[") == code.count("]"), "unbalanced square brackets in .scd"


def test_sc_emits_fm_when_modulators_present(tmp_path, matched_timbre):
    """An FM-modulator-bearing Timbre should produce a SynthDef where the
    carrier frequency expression contains an embedded SinOsc."""
    from biotuner.harmonic_timbre import Modulator
    t = matched_timbre.with_partials(
        fm_modulators=[Modulator(carrier_idx=0, mod_freq=8.0, depth=20.0, mod_type="FM")],
    )
    out = export_supercollider(t, str(tmp_path / "fm"), include_demo=False)
    text = open(out["scd"]).read()
    assert "SinOsc.ar(8" in text  # FM modulator at 8 Hz


def test_sc_emits_am_when_am_modulators_present(tmp_path, matched_timbre):
    from biotuner.harmonic_timbre import Modulator
    t = matched_timbre.with_partials(
        am_modulators=[Modulator(carrier_idx=0, mod_freq=4.0, depth=0.5, mod_type="AM")],
    )
    out = export_supercollider(t, str(tmp_path / "am"), include_demo=False)
    text = open(out["scd"]).read()
    assert "(1 + SinOsc.ar(4" in text
