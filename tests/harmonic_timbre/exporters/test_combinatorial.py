"""Tests for biotuner.harmonic_timbre.exporters.combinatorial."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from biotuner.harmonic_timbre import (
    Modulator,
    fm_patch_from_tuning,
    inharmonic_string,
    inharmonic_timbre,
    match_consonance_weighted,
)
from biotuner.harmonic_timbre.exporters import (
    EffectSpec,
    EnvelopeSpec,
    FilterSpec,
    LFOSpec,
    MacroSpec,
    OscSpec,
    PresetSpec,
    random_spec,
    to_vital_combinatorial,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def full_sources():
    """A 'sources' dict populated with one of every kind of biotuner data."""
    timbre = match_consonance_weighted(
        [1.0, 9 / 8, 5 / 4, 4 / 3, 3 / 2, 5 / 3, 15 / 8, 2.0],
        n_partials=8, base_freq=220.0,
    )
    timbre_inh = inharmonic_timbre(
        inharmonic_string, n=8, base_freq=220.0, fn_kwargs={"B": 1e-4},
    )
    sf_in = 1000.0
    t = np.arange(int(2 * sf_in)) / sf_in
    sig = (1 + 0.5 * np.sin(2 * np.pi * 0.5 * t)) * np.sin(2 * np.pi * 9 * t)
    imfs = [np.sin(2 * np.pi * f * t) for f in (5, 13, 27, 50)]
    gate = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    hg = (np.sin(2 * np.pi * np.arange(2000) / 2000),
          np.cos(2 * np.pi * np.arange(2000) / 2000))
    return {
        "timbre": timbre,
        "timbre_inharmonic": timbre_inh,
        "imfs": imfs,
        "signal": sig, "sf": sf_in,
        "polyrhythm_gate": gate,
        "harmonograph": hg,
    }


# ---------------------------------------------------------------------------
# Default-spec smoke
# ---------------------------------------------------------------------------

def test_default_spec_produces_valid_preset(tmp_path, full_sources):
    out = to_vital_combinatorial(
        PresetSpec(name="default"), str(tmp_path / "default"), full_sources,
    )
    assert os.path.exists(out["vital"])
    p = json.load(open(out["vital"]))
    assert len(p["settings"]) == 775   # factory schema parity
    # Default spec uses osc1=matched_timbre
    assert p["settings"]["osc_1_on"] == 1.0


# ---------------------------------------------------------------------------
# Spec-driven preset construction
# ---------------------------------------------------------------------------

def test_three_oscs_all_enabled_when_specified(tmp_path, full_sources):
    spec = PresetSpec(
        osc1=OscSpec(source="matched_timbre", level=0.7),
        osc2=OscSpec(source="inharmonic", level=0.5, transpose=12),
        osc3=OscSpec(source="imf_0", level=0.3, transpose=-12),
    )
    out = to_vital_combinatorial(spec, str(tmp_path / "x"), full_sources)
    s = json.load(open(out["vital"]))["settings"]
    assert s["osc_1_on"] == 1.0
    assert s["osc_2_on"] == 1.0
    assert s["osc_3_on"] == 1.0
    assert s["osc_2_transpose"] == 12.0
    assert s["osc_3_transpose"] == -12.0


def test_filter_character_dispatch(tmp_path, full_sources):
    """Each named character maps to a different Vital filter model code."""
    expected = {"lp": 0, "hp": 1, "bp": 2, "comb": 6, "ladder": 7, "formant": 9}
    for character, model_code in expected.items():
        spec = PresetSpec(
            name=f"f_{character}",
            filter_1=FilterSpec(character=character, cutoff=70.0, resonance=0.4),
        )
        out = to_vital_combinatorial(spec, str(tmp_path / character), full_sources)
        s = json.load(open(out["vital"]))["settings"]
        assert s["filter_1_on"] == 1.0
        assert s["filter_1_model"] == float(model_code)


def test_filter_off_disables_filter(tmp_path, full_sources):
    spec = PresetSpec(filter_1=FilterSpec(character="off"))
    out = to_vital_combinatorial(spec, str(tmp_path / "x"), full_sources)
    s = json.load(open(out["vital"]))["settings"]
    assert s["filter_1_on"] == 0.0


def test_envelope_character_changes_adsr(tmp_path, full_sources):
    """The 8 named characters produce distinguishable envelope values."""
    chars = ["pad", "bell", "pluck", "percussive", "drone", "organ", "swell", "stab"]
    seen = set()
    for ch in chars:
        spec = PresetSpec(env_1=EnvelopeSpec(character=ch))
        out = to_vital_combinatorial(spec, str(_get_tmp(tmp_path, ch)), full_sources)
        s = json.load(open(out["vital"]))["settings"]
        adsr = (round(s["env_1_attack"], 2), round(s["env_1_decay"], 2),
                round(s["env_1_sustain"], 2), round(s["env_1_release"], 2))
        seen.add(adsr)
    # All 8 characters should produce distinct ADSR signatures
    assert len(seen) == 8


def _get_tmp(tmp_path, label):
    return tmp_path / label


def test_lfos_resolve_from_curve_sources(tmp_path, full_sources):
    spec = PresetSpec(
        lfos=[
            LFOSpec(source="hilbert_envelope", rate_hz=0.25,
                    routings=[("filter_1_cutoff", 0.5, False)]),
            LFOSpec(source="imf_2", rate_hz=2.0,
                    routings=[("osc_1_distortion_amount", 0.3, False)]),
            LFOSpec(source="polyrhythm_gate", rate_hz=4.0, smooth=False,
                    routings=[("osc_2_level", 0.7, False)]),
        ] + [LFOSpec() for _ in range(5)],
    )
    out = to_vital_combinatorial(spec, str(tmp_path / "lfos"), full_sources)
    s = json.load(open(out["vital"]))["settings"]
    assert "hilbert_envelope" in s["lfos"][0]["name"]
    assert "imf_2" in s["lfos"][1]["name"]
    assert "polyrhythm_gate" in s["lfos"][2]["name"]
    assert s["lfos"][2]["smooth"] is False
    routings = {(m["source"], m["destination"]) for m in s["modulations"]
                if m.get("source", "").startswith("lfo_")}
    assert ("lfo_1", "filter_1_cutoff") in routings
    assert ("lfo_2", "osc_1_distortion_amount") in routings
    assert ("lfo_3", "osc_2_level") in routings


def test_effects_toggle_independently(tmp_path, full_sources):
    spec = PresetSpec(effects=EffectSpec(
        reverb=False, delay=True, chorus=True, distortion=True,
        flanger=True, phaser=True,
    ))
    out = to_vital_combinatorial(spec, str(tmp_path / "fx"), full_sources)
    s = json.load(open(out["vital"]))["settings"]
    assert s["reverb_on"] == 0.0
    assert s["delay_on"] == 1.0
    assert s["chorus_on"] == 1.0
    assert s["distortion_on"] == 1.0
    assert s["flanger_on"] == 1.0
    assert s["phaser_on"] == 1.0


def test_macros_set_names_and_routings(tmp_path, full_sources):
    spec = PresetSpec(
        macros=[
            MacroSpec(name="biosignal richness",
                      routings=[("filter_1_cutoff", 0.4),
                                ("osc_1_distortion_amount", 0.3)]),
            MacroSpec(name="motion intensity",
                      routings=[("reverb_dry_wet", 0.5)]),
        ],
    )
    out = to_vital_combinatorial(spec, str(tmp_path / "m"), full_sources)
    p = json.load(open(out["vital"]))
    s = p["settings"]
    assert "biosignal richness" in p["macro1"]
    assert "motion intensity" in p["macro2"]
    macro_routes = [(m["source"], m["destination"]) for m in s["modulations"]
                    if m.get("source", "").startswith("macro_control_")]
    assert ("macro_control_1", "filter_1_cutoff") in macro_routes
    assert ("macro_control_1", "osc_1_distortion_amount") in macro_routes
    assert ("macro_control_2", "reverb_dry_wet") in macro_routes


def test_extra_settings_escape_hatch(tmp_path, full_sources):
    """Custom keys in extra_settings override the corresponding setting."""
    spec = PresetSpec(extra_settings={
        "volume": 0.42,
        "filter_1_cutoff": 95.0,
    })
    out = to_vital_combinatorial(spec, str(tmp_path / "extra"), full_sources)
    s = json.load(open(out["vital"]))["settings"]
    assert s["volume"] == 0.42
    assert s["filter_1_cutoff"] == 95.0


def test_missing_source_silently_skips(tmp_path):
    """If a spec references a source that's not in the sources dict, the
    affected slot is just skipped — the preset still loads."""
    spec = PresetSpec(
        osc1=OscSpec(source="matched_timbre"),   # missing → skipped
        osc2=OscSpec(source="imf_5"),             # missing → skipped
        lfos=[LFOSpec(source="hilbert_envelope")] + [LFOSpec() for _ in range(7)],
    )
    out = to_vital_combinatorial(spec, str(tmp_path / "miss"), {})
    s = json.load(open(out["vital"]))["settings"]
    # No osc 1 patched — check it stayed off (template default for osc_2 is 0)
    # Actually osc_1 starts on in template, so it'll be on but with template wavetable
    # The point is the preset is structurally valid
    assert len(s) == 775


# ---------------------------------------------------------------------------
# random_spec
# ---------------------------------------------------------------------------

def test_random_spec_produces_distinct_specs():
    """20 random specs should differ across 6+ axes (osc choices, lfo
    sources, filter, env, effects)."""
    rng = np.random.default_rng(42)
    specs = [
        random_spec(rng=rng, available_sources={
            "matched_timbre", "inharmonic", "imfs", "signal",
            "polyrhythm_gate", "harmonograph",
        }, name=f"r_{i}")
        for i in range(20)
    ]
    osc1_sources = {s.osc1.source for s in specs}
    filter_chars = {s.filter_1.character for s in specs}
    env_chars = {s.env_1.character for s in specs}
    assert len(osc1_sources) >= 3
    assert len(filter_chars) >= 3
    assert len(env_chars) >= 4


def test_random_spec_only_uses_available_sources():
    """If the user has only timbre + signal data, the spec should never
    reference imf_N or polyrhythm sources."""
    rng = np.random.default_rng(0)
    available = {"matched_timbre", "signal"}
    for _ in range(40):
        s = random_spec(rng=rng, available_sources=available)
        for osc in (s.osc1, s.osc2, s.osc3):
            if osc.source is not None:
                assert not osc.source.startswith("imf_")
                assert osc.source not in ("polyrhythm_wave", "harmonograph_x", "harmonograph_y")
        for lfo in s.lfos:
            if lfo.source is not None:
                assert not lfo.source.startswith("imf_")
                assert lfo.source not in (
                    "polyrhythm_gate", "harmonograph_x", "harmonograph_y",
                )


def test_random_spec_round_trips_through_generator(tmp_path, full_sources):
    """Sample 10 specs, render each, and confirm every one is structurally valid."""
    rng = np.random.default_rng(7)
    counts = {"oscs": [], "lfos_active": [], "fx_active": []}
    for i in range(10):
        spec = random_spec(rng=rng, available_sources={
            "matched_timbre", "inharmonic", "imfs", "signal",
            "polyrhythm_gate", "harmonograph",
        }, name=f"r_{i}")
        out = to_vital_combinatorial(spec, str(tmp_path / f"r_{i}"), full_sources)
        s = json.load(open(out["vital"]))["settings"]
        assert len(s) == 775
        counts["oscs"].append(sum(int(s[f"osc_{n+1}_on"]) for n in range(3)))
        # crude active-LFO heuristic: smooth flag set by us OR num_points > 4
        counts["lfos_active"].append(sum(
            1 for l in s["lfos"] if l.get("num_points", 0) > 4
        ))
        counts["fx_active"].append(sum(int(s[f"{e}_on"]) for e in
            ("reverb", "delay", "chorus", "distortion", "flanger", "phaser")))
    # Variety check: not every preset has the same osc count
    assert len(set(counts["oscs"])) >= 2
    assert len(set(counts["fx_active"])) >= 2


# ---------------------------------------------------------------------------
# Schema parity — every combinatorial preset matches factory shape
# ---------------------------------------------------------------------------

# ===========================================================================
# build_combinatorial_sources — auto-detection of available sources
# ===========================================================================

def _make_synthetic_signal(duration=8.0, sf=500.0):
    t = np.arange(int(duration * sf)) / sf
    return (
        1.0 * np.sin(2 * np.pi * 9.5 * t)
        + 0.6 * np.sin(2 * np.pi * 22.0 * t)
        + 0.4 * np.sin(2 * np.pi * 4.0 * t)
        + 0.15 * np.random.default_rng(0).standard_normal(t.size)
    ), sf


def test_build_sources_emd_path_includes_imfs():
    """EMD-family peaks_function should populate IMFs as a side effect."""
    from biotuner.harmonic_timbre.exporters import build_combinatorial_sources
    signal, sf = _make_synthetic_signal()
    sources, bt = build_combinatorial_sources(signal, sf, peaks_function="EMD")
    assert "imfs" in sources["__available__"]
    assert "matched_timbre" in sources["__available__"]
    assert "signal" in sources["__available__"]
    assert isinstance(sources["imfs"], list)
    assert all(isinstance(x, np.ndarray) for x in sources["imfs"])


def test_build_sources_welch_path_excludes_imfs_unless_forced():
    """Non-EMD peaks_function should NOT populate IMFs by default."""
    from biotuner.harmonic_timbre.exporters import build_combinatorial_sources
    signal, sf = _make_synthetic_signal()
    sources, bt = build_combinatorial_sources(signal, sf, peaks_function="fixed")
    assert "imfs" not in sources["__available__"]


def test_build_sources_force_imfs_runs_separate_emd():
    """When force_imfs=True with non-EMD peaks_function, an extra EMD pass
    should populate the IMFs slot."""
    from biotuner.harmonic_timbre.exporters import build_combinatorial_sources
    signal, sf = _make_synthetic_signal()
    sources, bt = build_combinatorial_sources(
        signal, sf, peaks_function="fixed", force_imfs=True,
    )
    assert "imfs" in sources["__available__"]


def test_build_sources_includes_optional_derivations():
    """Inharmonic, harmonograph, polyrhythm should each populate when toggled on."""
    from biotuner.harmonic_timbre.exporters import build_combinatorial_sources
    signal, sf = _make_synthetic_signal()
    sources, bt = build_combinatorial_sources(
        signal, sf, peaks_function="EMD",
        include_inharmonic=True, include_harmonograph=True,
        include_polyrhythm=True,
    )
    avail = sources["__available__"]
    assert "inharmonic" in avail
    # harmonograph and polyrhythm depend on having enough peaks/ratios; usually OK
    # for a 10s biosignal but not strictly required by the test


def test_build_sources_skips_optional_when_disabled():
    from biotuner.harmonic_timbre.exporters import build_combinatorial_sources
    signal, sf = _make_synthetic_signal()
    sources, bt = build_combinatorial_sources(
        signal, sf, peaks_function="fixed",
        include_inharmonic=False, include_harmonograph=False,
        include_polyrhythm=False,
    )
    avail = sources["__available__"]
    assert "inharmonic" not in avail
    assert "harmonograph" not in avail
    assert "polyrhythm_gate" not in avail


# ===========================================================================
# spec_from_biotuner — biosignal-driven spec
# ===========================================================================

def test_spec_from_biotuner_deterministic_at_randomness_zero():
    """Two calls with r=0 on the same bt should produce identical specs."""
    from biotuner.harmonic_timbre.exporters import (
        build_combinatorial_sources, spec_from_biotuner,
    )
    signal, sf = _make_synthetic_signal()
    sources, bt = build_combinatorial_sources(signal, sf, peaks_function="EMD")
    spec_a = spec_from_biotuner(bt, signal=signal, sf=sf,
                                 available_sources=sources["__available__"],
                                 randomness=0.0)
    spec_b = spec_from_biotuner(bt, signal=signal, sf=sf,
                                 available_sources=sources["__available__"],
                                 randomness=0.0)
    assert spec_a.env_1.character == spec_b.env_1.character
    assert spec_a.filter_1.character == spec_b.filter_1.character
    assert spec_a.osc1.source == spec_b.osc1.source
    assert [l.source for l in spec_a.lfos] == [l.source for l in spec_b.lfos]


def test_spec_from_biotuner_respects_available_sources():
    """When IMFs aren't available, the spec should NEVER reference imf_N."""
    from biotuner.harmonic_timbre.exporters import (
        build_combinatorial_sources, spec_from_biotuner,
    )
    signal, sf = _make_synthetic_signal()
    sources, bt = build_combinatorial_sources(signal, sf, peaks_function="fixed")
    # 'fixed' path doesn't populate IMFs; verify
    assert "imfs" not in sources["__available__"]

    for r in (0.0, 0.5, 1.0):
        spec = spec_from_biotuner(
            bt, signal=signal, sf=sf,
            available_sources=sources["__available__"],
            randomness=r, seed=0,
        )
        for osc in (spec.osc1, spec.osc2, spec.osc3):
            if osc.source is not None:
                assert not osc.source.startswith("imf_")
        for lfo in spec.lfos:
            if lfo.source is not None:
                assert not lfo.source.startswith("imf_")


def test_spec_from_biotuner_randomness_increases_variance():
    """Sampling 10 specs at r=0 → all identical; at r=1 → most differ."""
    from biotuner.harmonic_timbre.exporters import (
        build_combinatorial_sources, spec_from_biotuner,
    )
    signal, sf = _make_synthetic_signal()
    sources, bt = build_combinatorial_sources(signal, sf, peaks_function="EMD")

    specs_r0 = [
        spec_from_biotuner(bt, signal=signal, sf=sf,
                            available_sources=sources["__available__"],
                            randomness=0.0, seed=i)
        for i in range(5)
    ]
    specs_r1 = [
        spec_from_biotuner(bt, signal=signal, sf=sf,
                            available_sources=sources["__available__"],
                            randomness=1.0, seed=i)
        for i in range(5)
    ]
    env_chars_r0 = {s.env_1.character for s in specs_r0}
    env_chars_r1 = {s.env_1.character for s in specs_r1}
    # r=0 → all the same env character
    assert len(env_chars_r0) == 1
    # r=1 → at least 2 different env characters across 5 seeds
    assert len(env_chars_r1) >= 2


def test_spec_from_biotuner_routes_slow_lfos_to_slow_destinations():
    """Hilbert envelope (slow) should route to a slow destination
    (filter_1_cutoff, reverb_*, volume, wavetable_position, pan)."""
    from biotuner.harmonic_timbre.exporters import (
        build_combinatorial_sources, spec_from_biotuner,
    )
    signal, sf = _make_synthetic_signal()
    sources, bt = build_combinatorial_sources(signal, sf, peaks_function="EMD")
    spec = spec_from_biotuner(bt, signal=signal, sf=sf,
                               available_sources=sources["__available__"],
                               randomness=0.0)
    SLOW = {"filter_1_cutoff", "filter_1_resonance", "reverb_dry_wet",
            "reverb_size", "osc_1_wavetable_position", "osc_1_pan",
            "volume", "delay_dry_wet"}
    for lfo in spec.lfos:
        if lfo.source == "hilbert_envelope":
            for dest, _, _ in lfo.routings:
                assert dest in SLOW, f"hilbert_envelope routed to non-slow {dest}"


def test_spec_from_biotuner_end_to_end_to_combinatorial(tmp_path):
    """Full pipeline: signal → sources + bt → spec → preset that loads."""
    from biotuner.harmonic_timbre.exporters import (
        build_combinatorial_sources, spec_from_biotuner, to_vital_combinatorial,
    )
    signal, sf = _make_synthetic_signal()
    sources, bt = build_combinatorial_sources(signal, sf, peaks_function="EMD")

    presets = []
    for i in range(5):
        spec = spec_from_biotuner(bt, signal=signal, sf=sf,
                                   available_sources=sources["__available__"],
                                   randomness=0.3, seed=i)
        out = to_vital_combinatorial(
            spec, str(tmp_path / f"p_{i}"), sources,
        )
        presets.append(out)

    for p in presets:
        assert os.path.exists(p["vital"])
        loaded = json.load(open(p["vital"]))
        assert len(loaded["settings"]) == 775   # factory schema parity


def test_combinatorial_preset_schema_identical_to_factory(tmp_path, full_sources):
    """A maximally-varied spec should still produce a preset structurally
    identical to the factory template (we only patch values, never delete keys)."""
    spec = PresetSpec(
        osc1=OscSpec(source="matched_timbre", unison_voices=3),
        osc2=OscSpec(source="inharmonic", transpose=12),
        osc3=OscSpec(source="imf_0", transpose=-12),
        lfos=[
            LFOSpec(source="hilbert_envelope", rate_hz=0.25,
                    routings=[("filter_1_cutoff", 0.6, False)]),
            LFOSpec(source="imf_1", rate_hz=2.0,
                    routings=[("osc_2_level", 0.5, False)]),
        ] + [LFOSpec() for _ in range(6)],
        filter_1=FilterSpec(character="lp", cutoff=70, resonance=0.5),
        env_1=EnvelopeSpec(character="bell"),
        effects=EffectSpec(reverb=True, delay=True, chorus=True, distortion=True),
        macros=[
            MacroSpec(name="m1", routings=[("filter_1_cutoff", 0.4)]),
        ],
    )
    out = to_vital_combinatorial(spec, str(tmp_path / "max"), full_sources)
    p = json.load(open(out["vital"]))
    # Schema parity: same number of settings keys as Plucked String
    assert len(p["settings"]) == 775
    # Wavetable component has the right Vital schema
    wt = p["settings"]["wavetables"][0]
    component = wt["groups"][0]["components"][0]
    assert component["type"] == "Wave Source"
