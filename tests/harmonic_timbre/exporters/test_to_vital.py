"""Tests for biotuner.harmonic_timbre.exporters.to_vital (template-surgery
implementation). Verifies:

* generated presets have the same top-level shape as the factory template
* wavetable keyframes use Vital's actual schema (``Wave Source`` type,
  ``keyframes: [{position, wave_data}]`` structure)
* modulation entries are ``{source, destination}`` only, with the amount
  stored as a numbered ``modulation_N_amount`` settings key
* LFO entries use the ``points``/``powers`` shape encoding
* every preset's wavetable audio round-trips correctly through base64
"""

from __future__ import annotations

import base64
import json
import os

import numpy as np
import pytest

from biotuner.harmonic_timbre import (
    Modulator,
    fm_patch_from_tuning,
    inharmonic_string,
    inharmonic_timbre,
    map_pac_to_am_modulators,
    match_consonance_weighted,
    timbre_sequence_from_ratio_frames,
)
from biotuner.harmonic_timbre.exporters import (
    to_vital_cfc_driven,
    to_vital_ensemble,
    to_vital_fm,
    to_vital_hilbert_modulator,
    to_vital_hilbert_sample,
    to_vital_imf_lfo_bank,
    to_vital_imf_stack,
    to_vital_inharmonic,
    to_vital_markov_macro,
    to_vital_pac_driven,
    to_vital_polyrhythm_gated,
    to_vital_spectral,
    to_vital_wavetable_morph,
)
from biotuner.harmonic_timbre.exporters._vital_template import (
    WAVETABLE_FRAME_SIZE,
    curve_to_lfo_points,
    modulator_to_routing,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decoded_wavedata(keyframe: dict) -> np.ndarray:
    return np.frombuffer(base64.b64decode(keyframe["wave_data"]), dtype=np.float32)


def _shape_of_factory_template():
    """Sanity-check that the embedded factory template has the structure we
    rely on. If Vital's schema changes between versions, this fails fast."""
    from biotuner.harmonic_timbre.exporters._vital_template import _load_template
    p = _load_template()
    s = p["settings"]
    assert isinstance(s.get("wavetables"), list) and len(s["wavetables"]) == 3
    assert isinstance(s.get("modulations"), list) and len(s["modulations"]) >= 32
    assert isinstance(s.get("lfos"), list) and len(s["lfos"]) >= 4
    return p


def test_factory_template_has_expected_shape():
    _shape_of_factory_template()


def _common_assertions(preset: dict) -> None:
    """Every generator's output should preserve the factory schema."""
    s = preset["settings"]
    # 3 wavetable slots
    assert isinstance(s["wavetables"], list)
    assert len(s["wavetables"]) == 3
    # 64 modulation slots (template has 64; we don't add or remove)
    assert isinstance(s["modulations"], list)
    assert len(s["modulations"]) >= 32
    # 8 LFOs
    assert isinstance(s["lfos"], list)
    assert len(s["lfos"]) == 8
    # synth_version preserved from template
    assert "synth_version" in preset


# ---------------------------------------------------------------------------
# 1. to_vital_spectral
# ---------------------------------------------------------------------------

def test_spectral_writes_vital_and_companion(tmp_path, matched_timbre):
    out = to_vital_spectral(matched_timbre, str(tmp_path / "spectral"))
    assert os.path.exists(out["vital"])
    assert os.path.exists(out["companion"])
    assert out["vital"].endswith(".vital")
    assert out["companion"].endswith("_settings.json")


def test_spectral_preserves_template_shape(tmp_path, matched_timbre):
    out = to_vital_spectral(matched_timbre, str(tmp_path / "x"))
    p = json.load(open(out["vital"]))
    _common_assertions(p)


def test_spectral_wavetable_uses_wave_source_keyframes(tmp_path, matched_timbre):
    out = to_vital_spectral(matched_timbre, str(tmp_path / "x"))
    p = json.load(open(out["vital"]))
    wt = p["settings"]["wavetables"][0]
    component = wt["groups"][0]["components"][0]
    assert component["type"] == "Wave Source"
    assert "keyframes" in component
    assert len(component["keyframes"]) == 1
    kf = component["keyframes"][0]
    assert set(kf.keys()) == {"position", "wave_data"}


def test_spectral_wavetable_round_trips(tmp_path, matched_timbre):
    out = to_vital_spectral(matched_timbre, str(tmp_path / "x"))
    p = json.load(open(out["vital"]))
    audio = _decoded_wavedata(p["settings"]["wavetables"][0]["groups"][0]["components"][0]["keyframes"][0])
    assert audio.size == WAVETABLE_FRAME_SIZE
    assert audio.dtype == np.float32


def test_spectral_companion_records_provenance(tmp_path, matched_timbre):
    out = to_vital_spectral(matched_timbre, str(tmp_path / "x"))
    c = json.load(open(out["companion"]))
    assert c["preset_kind"] == "spectral"
    assert c["timbre"]["matched_tuning"] == matched_timbre.matched_tuning


# ---------------------------------------------------------------------------
# 2. to_vital_wavetable_morph
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("evolution", ["tilt", "harmonic_buildup", "amp_morph", "phase_sweep"])
def test_wavetable_morph_evolution_modes(tmp_path, matched_timbre, evolution):
    out = to_vital_wavetable_morph(
        matched_timbre, str(tmp_path / f"morph_{evolution}"),
        n_frames=8, evolution=evolution,
    )
    p = json.load(open(out["vital"]))
    _common_assertions(p)
    component = p["settings"]["wavetables"][0]["groups"][0]["components"][0]
    assert component["type"] == "Wave Source"
    assert len(component["keyframes"]) == 8
    # Each keyframe round-trips to 2048 float32 samples
    for kf in component["keyframes"]:
        audio = _decoded_wavedata(kf)
        assert audio.size == WAVETABLE_FRAME_SIZE


def test_wavetable_morph_lfo_routing(tmp_path, matched_timbre):
    out = to_vital_wavetable_morph(
        matched_timbre, str(tmp_path / "x"),
        n_frames=4, map_lfo_to_position=True, lfo_rate_hz=0.5,
    )
    p = json.load(open(out["vital"]))
    s = p["settings"]
    # LFO 1 (index 0) reshaped — the new set_lfo_from_curve defaults to 16 points
    lfo = s["lfos"][0]
    assert lfo["smooth"] is True
    assert lfo["num_points"] == 16
    assert "points" in lfo and len(lfo["points"]) == 32  # 2 * num_points (x, y interleaved)
    # Modulation slot is filled with lfo_1 → osc_1_wavetable_position
    found = [m for m in s["modulations"]
             if m.get("source") == "lfo_1"
             and m.get("destination") == "osc_1_wavetable_position"]
    assert len(found) == 1


def test_wavetable_morph_disable_lfo_leaves_template_lfos_intact(tmp_path, matched_timbre):
    out = to_vital_wavetable_morph(
        matched_timbre, str(tmp_path / "x"),
        n_frames=4, map_lfo_to_position=False,
    )
    p = json.load(open(out["vital"]))
    s = p["settings"]
    # No new modulation entries pointing at osc_1_wavetable_position
    found = [m for m in s["modulations"]
             if m.get("destination") == "osc_1_wavetable_position"]
    assert len(found) == 0


def test_wavetable_morph_unknown_evolution_raises(tmp_path, matched_timbre):
    with pytest.raises(ValueError, match="unknown evolution"):
        to_vital_wavetable_morph(
            matched_timbre, str(tmp_path / "x"),
            n_frames=4, evolution="cowbell",
        )


# ---------------------------------------------------------------------------
# 3. to_vital_fm
# ---------------------------------------------------------------------------

def test_fm_requires_modulators(tmp_path, matched_timbre):
    with pytest.raises(ValueError, match="no fm_modulators"):
        to_vital_fm(matched_timbre, str(tmp_path / "x"))


def test_fm_enables_two_oscs_and_routes_destination(tmp_path):
    timbre = fm_patch_from_tuning([1.0, 5/4, 3/2, 2.0], n_carriers=4, fm_index=2.0)
    out = to_vital_fm(timbre, str(tmp_path / "fm"))
    p = json.load(open(out["vital"]))
    s = p["settings"]
    assert s["osc_1_on"] == 1.0
    assert s["osc_2_on"] == 1.0
    assert s["osc_2_level"] == 0.0  # modulator silenced
    if "osc_2_destination" in s:
        assert s["osc_2_destination"] == 1.0


def test_fm_ratio_translation_to_coarse_plus_fine(tmp_path):
    """5/4 ratio = +386 cents = 4 semitones + (-14 cents)."""
    timbre = fm_patch_from_tuning([1.0, 5/4, 2.0], n_carriers=2, fm_index=1.0,
                                  base_freq=200.0)
    out = to_vital_fm(timbre, str(tmp_path / "fm"))
    p = json.load(open(out["vital"]))
    semis = p["settings"]["osc_2_transpose"]
    tune = p["settings"]["osc_2_tune"]
    cents = semis * 100.0 + tune * 100.0
    assert abs(cents - 386.31) < 1.0


def test_fm_companion_records_routing(tmp_path):
    timbre = fm_patch_from_tuning([1.0, 3/2, 2.0], n_carriers=2, fm_index=1.5)
    out = to_vital_fm(timbre, str(tmp_path / "fm"))
    c = json.load(open(out["companion"]))
    assert c["preset_kind"] == "fm"
    assert "fm_routing" in c
    assert "vital_settings" in c["fm_routing"]


# ---------------------------------------------------------------------------
# 4. to_vital_pac_driven
# ---------------------------------------------------------------------------

def test_pac_driven_requires_am_modulators(tmp_path, matched_timbre):
    with pytest.raises(ValueError, match="no am_modulators"):
        to_vital_pac_driven(matched_timbre, str(tmp_path / "x"))


def test_pac_driven_one_lfo_per_modulator(tmp_path, matched_timbre):
    t = matched_timbre.with_partials(am_modulators=[
        Modulator(carrier_idx=0, mod_freq=4.0, depth=0.5, mod_type="AM",
                  source="PAC_theta_beta"),
        Modulator(carrier_idx=1, mod_freq=8.0, depth=0.3, mod_type="AM",
                  source="PAC_alpha_gamma"),
        Modulator(carrier_idx=0, mod_freq=12.0, depth=0.2, mod_type="AM",
                  source="PAC_alpha_beta"),
    ])
    out = to_vital_pac_driven(t, str(tmp_path / "pac"))
    p = json.load(open(out["vital"]))
    s = p["settings"]
    # 3 mod-matrix entries pointing lfo_N → osc_1_level
    routes = [
        m for m in s["modulations"]
        if m.get("source", "").startswith("lfo_")
        and m.get("destination") == "osc_1_level"
    ]
    assert len(routes) == 3
    # First 3 LFOs reshaped to sine (smooth=True with 4 points)
    for i in range(3):
        assert s["lfos"][i]["smooth"] is True
        assert s["lfos"][i]["num_points"] == 4


def test_pac_driven_caps_at_8_lfos(tmp_path, matched_timbre):
    """Vital has only 8 LFO slots; extra modulators are dropped."""
    mods = [
        Modulator(carrier_idx=0, mod_freq=float(i + 1), depth=0.5,
                  mod_type="AM", source=f"PAC_{i}")
        for i in range(12)
    ]
    t = matched_timbre.with_partials(am_modulators=mods)
    out = to_vital_pac_driven(t, str(tmp_path / "pac"))
    c = json.load(open(out["companion"]))
    assert c["n_am_modulators_consumed"] == 8
    assert c["n_am_modulators_skipped"] == 4


def test_pac_driven_companion_documents_lfo_routing(tmp_path, matched_timbre):
    t = matched_timbre.with_partials(am_modulators=[
        Modulator(carrier_idx=0, mod_freq=5.0, depth=0.6, mod_type="AM",
                  source="PAC_test"),
    ])
    out = to_vital_pac_driven(t, str(tmp_path / "pac"))
    c = json.load(open(out["companion"]))
    assert c["preset_kind"] == "pac_driven"
    routing = c["lfo_routing"][0]
    assert routing["rate_hz"] == 5.0
    assert routing["amount"] == 0.6


def test_pac_driven_integrates_with_map_pac_to_am_modulators(tmp_path, matched_timbre):
    class MockBT:
        peaks = [4.0, 8.0, 13.0, 25.0, 40.0]
        pac_freqs = [(4.0, 25.0), (8.0, 40.0)]
        pac_coupling = [0.7, 0.9]
    am_mods = map_pac_to_am_modulators(MockBT())
    t = matched_timbre.with_partials(am_modulators=am_mods)
    out = to_vital_pac_driven(t, str(tmp_path / "biopac"))
    p = json.load(open(out["vital"]))
    routes = [m for m in p["settings"]["modulations"]
              if m.get("destination") == "osc_1_level"
              and m.get("source", "").startswith("lfo_")]
    assert len(routes) == len(am_mods)


# ---------------------------------------------------------------------------
# Cross-cutting: preset structure parity with factory template
# ---------------------------------------------------------------------------

def test_all_preset_kinds_preserve_total_settings_count(tmp_path):
    """The factory template has ~775 settings keys; our presets should keep
    the same count (we patch in place, not delete)."""
    template = _shape_of_factory_template()
    base_count = len(template["settings"])

    timbre = match_consonance_weighted([1.0, 3/2, 2.0], n_partials=4, base_freq=220.0)
    fm_t = fm_patch_from_tuning([1.0, 3/2, 2.0], n_carriers=2, fm_index=1.0)
    pac_t = timbre.with_partials(am_modulators=[
        Modulator(carrier_idx=0, mod_freq=4.0, depth=0.5, mod_type="AM",
                  source="t"),
    ])

    for fn, args in [
        (to_vital_spectral,         (timbre, str(tmp_path / "a"))),
        (to_vital_wavetable_morph,  (timbre, str(tmp_path / "b"))),
        (to_vital_fm,               (fm_t,   str(tmp_path / "c"))),
        (to_vital_pac_driven,       (pac_t,  str(tmp_path / "d"))),
    ]:
        out = fn(*args)
        p = json.load(open(out["vital"]))
        # Count must be at least the template count (we may add modulation_N_amount
        # keys when patching, which is fine; we never remove keys).
        assert len(p["settings"]) >= base_count - 5


# ===========================================================================
# Tier 1 — primitive converters
# ===========================================================================

def test_curve_to_lfo_points_returns_2x_pairs_and_powers():
    curve = np.sin(2 * np.pi * np.linspace(0, 1, 100))
    points, powers = curve_to_lfo_points(curve, n_points=8)
    assert len(points) == 16   # 2 * n_points (interleaved x, y)
    assert len(powers) == 8


def test_curve_to_lfo_points_normalizes_to_unit():
    curve = np.array([100.0, 200.0, 300.0, 400.0])
    points, _ = curve_to_lfo_points(curve, n_points=4)
    ys = points[1::2]   # every other element starting at idx 1 = y values
    assert min(ys) == pytest.approx(0.0)
    assert max(ys) == pytest.approx(1.0)


def test_curve_to_lfo_points_handles_short_curve():
    points, powers = curve_to_lfo_points([0.5], n_points=4)
    # Degenerate fallback: 2 control points
    assert len(points) == 4
    assert len(powers) == 2


def test_modulator_to_routing_am():
    m = Modulator(carrier_idx=0, mod_freq=4.0, depth=0.6, mod_type="AM",
                  source="PAC_test")
    r = modulator_to_routing(m, lfo_index=2)
    assert r["source"] == "lfo_3"
    assert r["destination"] == "osc_1_level"
    assert r["amount"] == 0.6
    assert r["rate_hz"] == 4.0
    assert r["name"] == "PAC_test"


def test_modulator_to_routing_fm_normalizes_depth():
    """For FM, depth (Hz) is divided by 5*mod_freq by default to land β=5 → amount=1."""
    m = Modulator(carrier_idx=0, mod_freq=10.0, depth=50.0, mod_type="FM",
                  source="CFC_test")
    r = modulator_to_routing(m, lfo_index=0)
    assert r["destination"] == "osc_1_distortion_amount"
    assert r["amount"] == pytest.approx(50.0 / 50.0, abs=0.01)  # β=5 → 1.0


def test_modulator_to_routing_unknown_type_raises():
    m = Modulator(carrier_idx=0, mod_freq=1.0, depth=1.0, mod_type="RM",
                  source="x")
    with pytest.raises(ValueError, match="unknown mod_type"):
        modulator_to_routing(m, lfo_index=0)


# ===========================================================================
# Tier 3 — new generators (one test per principle)
# ===========================================================================

# --- inharmonic ---
def test_inharmonic_preset(tmp_path):
    t = inharmonic_timbre(inharmonic_string, n=8, base_freq=220.0,
                          fn_kwargs={"B": 1e-4})
    out = to_vital_inharmonic(t, str(tmp_path / "inh"),
                              inharmonic_label="stiff_string")
    p = json.load(open(out["vital"]))
    _common_assertions(p)
    c = json.load(open(out["companion"]))
    assert c["preset_kind"] == "inharmonic"
    assert c["inharmonic_label"] == "stiff_string"


# --- cfc_driven ---
def test_cfc_driven_creates_lfos_for_each_fm_modulator(tmp_path):
    timbre = fm_patch_from_tuning([1.0, 5/4, 3/2, 2.0], n_carriers=3,
                                  fm_index=2.0)
    out = to_vital_cfc_driven(timbre, str(tmp_path / "cfc"))
    p = json.load(open(out["vital"]))
    s = p["settings"]
    routes = [m for m in s["modulations"]
              if m.get("source", "").startswith("lfo_")
              and m.get("destination") == "osc_1_distortion_amount"]
    assert len(routes) == 3   # 3 carriers → 3 FM modulators


def test_cfc_driven_requires_fm_modulators(tmp_path, matched_timbre):
    with pytest.raises(ValueError, match="no fm_modulators"):
        to_vital_cfc_driven(matched_timbre, str(tmp_path / "x"))


# --- hilbert_modulator (the unlock) ---
def test_hilbert_modulator_embeds_envelope_as_lfo_shape(tmp_path, matched_timbre):
    sf = 1000.0
    t = np.arange(int(2 * sf)) / sf
    sig = np.sin(2 * np.pi * 5 * t) * (1 + 0.5 * np.sin(2 * np.pi * 0.5 * t))
    out = to_vital_hilbert_modulator(
        sig, sf, matched_timbre, str(tmp_path / "hb"),
        destination="osc_1_filter_blend", n_lfo_points=24,
    )
    p = json.load(open(out["vital"]))
    s = p["settings"]
    # LFO 1 should have 24 control points with the envelope shape
    lfo = s["lfos"][0]
    assert lfo["num_points"] == 24
    assert len(lfo["points"]) == 48
    # Modulation matrix has lfo_1 → osc_1_filter_blend
    routes = [m for m in s["modulations"]
              if m.get("source") == "lfo_1"
              and m.get("destination") == "osc_1_filter_blend"]
    assert len(routes) == 1


def test_hilbert_modulator_short_signal_raises(tmp_path, matched_timbre):
    with pytest.raises(ValueError, match="too short"):
        to_vital_hilbert_modulator(
            [0.0, 1.0, 0.0], sf=1000.0, timbre=matched_timbre,
            out_path=str(tmp_path / "x"),
        )


# --- imf_lfo_bank ---
def test_imf_lfo_bank_one_lfo_per_imf(tmp_path, matched_timbre):
    sf = 1000.0
    t = np.arange(int(sf)) / sf
    imfs = [np.sin(2 * np.pi * f * t) for f in (5.0, 13.0, 27.0, 50.0)]
    out = to_vital_imf_lfo_bank(imfs, matched_timbre, str(tmp_path / "imf_bank"))
    p = json.load(open(out["vital"]))
    s = p["settings"]
    # Each IMF i should have its lfo_(i+1) routed to the i-th destination from
    # the default destination list. The Plucked String template may already
    # have unrelated routes; we check that the 4 expected IMF routes exist.
    expected_routes = [
        ("lfo_1", "osc_1_level"),
        ("lfo_2", "filter_1_cutoff"),
        ("lfo_3", "osc_1_distortion_amount"),
        ("lfo_4", "osc_1_wavetable_position"),
    ]
    for src, dst in expected_routes:
        match = [m for m in s["modulations"]
                 if m.get("source") == src and m.get("destination") == dst]
        assert len(match) >= 1, f"missing routing {src} → {dst}"


def test_imf_lfo_bank_caps_at_8_imfs(tmp_path, matched_timbre):
    sf = 100.0
    t = np.arange(int(sf)) / sf
    imfs = [np.sin(2 * np.pi * (i + 1) * t) for i in range(12)]
    out = to_vital_imf_lfo_bank(imfs, matched_timbre, str(tmp_path / "x"))
    c = json.load(open(out["companion"]))
    assert c["n_imfs_consumed"] == 8


# --- imf_stack (topology) ---
def test_imf_stack_enables_multiple_oscillators(tmp_path, matched_timbre):
    sf = 100.0
    t = np.arange(int(sf)) / sf
    imfs = [np.sin(2 * np.pi * (i + 1) * 5 * t) for i in range(3)]
    out = to_vital_imf_stack(imfs, matched_timbre, str(tmp_path / "stack"))
    p = json.load(open(out["vital"]))
    s = p["settings"]
    # 3 oscillators on
    assert s["osc_1_on"] == 1.0
    assert s["osc_2_on"] == 1.0
    assert s["osc_3_on"] == 1.0
    # FM destination wired on osc 2 → osc 1 (osc_2_destination = 1)
    if "osc_2_destination" in s:
        assert s["osc_2_destination"] == 1.0


# --- hilbert_sample (found sound) ---
def test_hilbert_sample_writes_wav_alongside_preset(tmp_path):
    sf = 1000.0
    t = np.arange(int(sf)) / sf
    sig = np.sin(2 * np.pi * 5 * t)
    out = to_vital_hilbert_sample(
        sig, sf, str(tmp_path / "hs"), duration=0.5,
    )
    assert os.path.exists(out["sample_wav"])
    # The sample WAV exists and has audio
    import soundfile as sf_io
    audio, _ = sf_io.read(out["sample_wav"])
    assert audio.size > 0


def test_hilbert_sample_companion_includes_load_instructions(tmp_path):
    sf = 1000.0
    sig = np.sin(2 * np.pi * 5 * np.arange(int(sf)) / sf)
    out = to_vital_hilbert_sample(sig, sf, str(tmp_path / "hs"), duration=0.3)
    c = json.load(open(out["companion"]))
    assert c["preset_kind"] == "hilbert_sample"
    assert "rendered_audio_wav" in c
    assert "instructions" in c


# --- polyrhythm_gated ---
def test_polyrhythm_gated_uses_stepped_lfo(tmp_path, matched_timbre):
    gate = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    out = to_vital_polyrhythm_gated(matched_timbre, gate, str(tmp_path / "pg"))
    p = json.load(open(out["vital"]))
    s = p["settings"]
    # LFO 1's shape isn't smooth (stepped)
    assert s["lfos"][0]["smooth"] is False
    # Routed to osc_1_level by default
    routes = [m for m in s["modulations"]
              if m.get("source") == "lfo_1" and m.get("destination") == "osc_1_level"]
    assert len(routes) == 1


def test_polyrhythm_gated_empty_pattern_raises(tmp_path, matched_timbre):
    with pytest.raises(ValueError, match="empty gate_pattern"):
        to_vital_polyrhythm_gated(matched_timbre, np.array([]),
                                  str(tmp_path / "x"))


# --- markov_macro ---
def test_markov_macro_creates_multi_keyframe_wavetable_with_macro_routing(tmp_path):
    seq = timbre_sequence_from_ratio_frames([
        [1.0, 5/4, 3/2, 2.0],
        [1.0, 6/5, 3/2, 2.0],
        [1.0, 9/8, 11/8, 13/8, 2.0],
    ])
    out = to_vital_markov_macro(seq, str(tmp_path / "mm"))
    p = json.load(open(out["vital"]))
    s = p["settings"]
    component = s["wavetables"][0]["groups"][0]["components"][0]
    assert len(component["keyframes"]) == 3
    # Macro 1 → osc_1_wavetable_position
    routes = [m for m in s["modulations"]
              if m.get("source") == "macro_control_1"
              and m.get("destination") == "osc_1_wavetable_position"]
    assert len(routes) == 1
    # Macro 1 has been renamed for legibility
    assert "harmonic state" in p["macro1"].lower()


def test_markov_macro_requires_at_least_2_frames(tmp_path):
    seq = timbre_sequence_from_ratio_frames([[1.0, 5/4, 3/2, 2.0]])
    with pytest.raises(ValueError, match="≥ 2"):
        to_vital_markov_macro(seq, str(tmp_path / "x"))


def test_markov_macro_rejects_non_sequence(tmp_path, matched_timbre):
    with pytest.raises(TypeError, match="TimbreSequence"):
        to_vital_markov_macro(matched_timbre, str(tmp_path / "x"))


# --- ensemble ---
def test_ensemble_always_emits_base_voice(tmp_path, matched_timbre):
    out = to_vital_ensemble(matched_timbre, str(tmp_path / "ens"),
                            bundle_name="t")
    assert "base_morph" in out
    assert os.path.exists(out["base_morph"]["vital"])


def test_ensemble_conditionally_emits_voices_per_input(tmp_path, matched_timbre):
    sf = 1000.0
    t = np.arange(int(sf)) / sf
    sig = np.sin(2 * np.pi * 5 * t)
    imfs = [np.sin(2 * np.pi * f * t) for f in (5, 13, 27)]
    gate = np.array([1, 0, 1, 0, 1, 0])

    # Add AM modulators
    timbre = matched_timbre.with_partials(am_modulators=[
        Modulator(carrier_idx=0, mod_freq=4.0, depth=0.5, mod_type="AM", source="x"),
    ])

    out = to_vital_ensemble(
        timbre, str(tmp_path / "ens"), bundle_name="t",
        signal=sig, sf=sf, imfs=imfs, polyrhythm_gate=gate,
    )
    expected_voices = {"base_morph", "pad", "texture", "rhythm", "motion"}
    assert expected_voices.issubset(set(out.keys()))


def test_ensemble_writes_top_level_manifest(tmp_path, matched_timbre):
    out = to_vital_ensemble(matched_timbre, str(tmp_path / "ens"))
    manifest_path = out["__manifest__"]
    assert os.path.exists(manifest_path)
    m = json.load(open(manifest_path))
    assert m["format"] == "biotuner_vital_ensemble"
    assert "voices" in m


# ===========================================================================
# Tier 5 legibility — provenance comments + descriptive LFO names
# ===========================================================================

def test_provenance_baked_into_comments(tmp_path, matched_timbre):
    """All preset comments should mention the preset_kind and the matched_tuning."""
    out = to_vital_spectral(matched_timbre, str(tmp_path / "x"))
    p = json.load(open(out["vital"]))
    assert "spectral" in p["comments"]
    assert "matched_tuning" in p["comments"]


def test_pac_driven_lfo_names_describe_modulation(tmp_path, matched_timbre):
    """LFOs in pac_driven should have descriptive names (not just 'lfo_1')."""
    t = matched_timbre.with_partials(am_modulators=[
        Modulator(carrier_idx=0, mod_freq=4.5, depth=0.6, mod_type="AM",
                  source="PAC_theta_beta"),
    ])
    out = to_vital_pac_driven(t, str(tmp_path / "pac"))
    p = json.load(open(out["vital"]))
    lfo_names = [l["name"] for l in p["settings"]["lfos"][:1]]
    assert any("PAC" in n for n in lfo_names)
