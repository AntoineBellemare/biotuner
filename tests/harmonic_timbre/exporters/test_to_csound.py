"""Tests for biotuner.harmonic_timbre.exporters.to_csound."""

from __future__ import annotations

import os

import pytest

from biotuner.harmonic_timbre.exporters import export_csound


def test_csound_creates_csd(tmp_path, matched_timbre):
    out = export_csound(matched_timbre, str(tmp_path / "song"),
                        base_freq=220.0, demo_pattern="scale")
    assert os.path.exists(out["csd"])


def test_csound_contains_required_sections(tmp_path, matched_timbre):
    out = export_csound(matched_timbre, str(tmp_path / "song"))
    text = open(out["csd"]).read()
    for section in ("<CsoundSynthesizer>", "<CsOptions>", "<CsInstruments>",
                    "<CsScore>", "instr 1", "endin"):
        assert section in text, f"missing section {section!r}"


def test_csound_has_one_oscili_per_partial(tmp_path, matched_timbre):
    out = export_csound(matched_timbre, str(tmp_path / "song"))
    text = open(out["csd"]).read()
    n_partials = matched_timbre.n_partials()
    # one oscili per partial
    n_oscili = text.count("oscili")
    assert n_oscili >= n_partials


def test_csound_demo_pattern_options(tmp_path, matched_timbre):
    for pattern in ("scale", "chord", "arpeggio", "none"):
        out = export_csound(
            matched_timbre, str(tmp_path / pattern),
            demo_pattern=pattern,
        )
        assert os.path.exists(out["csd"])


def test_csound_unknown_pattern_raises(tmp_path, matched_timbre):
    with pytest.raises(ValueError):
        export_csound(matched_timbre, str(tmp_path / "x"), demo_pattern="not-real")


def test_csound_emits_fm_oscillators_when_modulators_present(tmp_path, matched_timbre):
    """A Timbre with FM modulators should emit per-modulator oscili lines
    in the Csound instr body."""
    from biotuner.harmonic_timbre import Modulator
    t = matched_timbre.with_partials(
        fm_modulators=[
            Modulator(carrier_idx=0, mod_freq=5.0, depth=20.0, mod_type="FM"),
            Modulator(carrier_idx=1, mod_freq=7.0, depth=15.0, mod_type="FM"),
        ],
    )
    out = export_csound(t, str(tmp_path / "fm"), demo_pattern="none")
    text = open(out["csd"]).read()
    # The instrument body should contain auxiliary fm oscillators
    assert "fm0" in text or "a0fm" in text
    assert "fm1" in text or "a1fm" in text


def test_csound_emits_am_oscillators_when_am_modulators_present(tmp_path, matched_timbre):
    from biotuner.harmonic_timbre import Modulator
    t = matched_timbre.with_partials(
        am_modulators=[Modulator(carrier_idx=0, mod_freq=3.0, depth=0.5, mod_type="AM")],
    )
    out = export_csound(t, str(tmp_path / "am"), demo_pattern="none")
    text = open(out["csd"]).read()
    assert "a0am" in text   # AM oscillator for partial 0
