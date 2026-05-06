"""Vital preset template-surgery primitives (private).

Module type: Functions

Loads Vital's "Plucked String" factory preset (saved as
``_vital_template_base.json``) as a known-good base, and exposes a small,
typed primitive set that surgically patches the parts of the preset that
should change for a given biotuner-derived patch.

Primitive surface (3 converters + 4 placers + 2 convenience helpers)
--------------------------------------------------------------------

    Converters — bridge biotuner-canonical shapes to Vital-native shapes:
        curve_to_lfo_points(curve_1d, n_points)        1D array → (points, powers)
        modulator_to_routing(modulator, lfo_index)     Modulator dataclass → routing dict
        # (partials → wavetable cycle is render_wavetable_cycle in synthesis.py)

    Placers — surgical edits on the loaded template:
        replace_wavetable(preset, osc_idx, audio, name)
        set_lfo_from_curve(preset, lfo_index, curve_1d, *, rate_hz, name)
        set_modulation(preset, *, slot, source, destination, amount, bipolar)
        populate_osc(preset, osc_idx, *, transpose, fine_cents, level, …)

    Convenience:
        route_lfo(preset, lfo_index, destination, amount)
        find_free_modulation_slot(preset)
        set_lfo_sine(preset, lfo_index, rate_hz, name)   # convenience wrapping set_lfo_from_curve

Persistence:
    write_vital_preset(preset, out_path)
    write_settings_companion(preset_kind, out_path, ...)

Schema notes (from Vital 1.5.x factory presets)
----------------------------------------------
* Wavetable component ``type`` is ``'Wave Source'`` (with space). Each frame
  is a keyframe with ``{position: int, wave_data: base64-of-2048-float32s}``.
* Modulation entry: ``{source, destination}`` only. Amount lives in
  ``settings.modulation_N_amount`` (1-indexed N).
* LFO entry: ``{name, num_points, points, powers, smooth}``. Rate is in
  ``settings.lfo_N_frequency`` as ``log2(seconds)``.
* FM in Vital is routed via ``osc_X_destination`` + ``osc_X_distortion_type``
  set to FM (enum value 9 = "FM From Sample").
"""

from __future__ import annotations

import base64
import copy
import json
import math
import os
from typing import Any

import numpy as np


_TEMPLATE_PATH = os.path.join(
    os.path.dirname(__file__), "_vital_template_base.json"
)
WAVETABLE_FRAME_SIZE = 2048


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

def _load_template() -> dict:
    """Load the embedded base preset."""
    with open(_TEMPLATE_PATH, encoding="utf-8") as fp:
        return json.load(fp)


def make_blank_preset(
    *,
    name: str = "biotuner_preset",
    author: str = "biotuner.harmonic_timbre",
    comments: str = "",
) -> dict:
    """Deep-copy the factory base preset and overwrite identifying fields."""
    p = copy.deepcopy(_load_template())
    p["author"] = author
    p["comments"] = comments
    p["preset_name"] = name
    return p


# ---------------------------------------------------------------------------
# Converters — biotuner-canonical → Vital-native
# ---------------------------------------------------------------------------

def curve_to_lfo_points(
    curve_1d,
    *,
    n_points: int = 16,
    normalize_to_unit: bool = True,
) -> tuple[list[float], list[float]]:
    """Resample any 1D curve into Vital's points/powers LFO format.

    Vital LFO control points are stored as a flat list of alternating x, y
    values in [0, 1], plus one ``power`` per point controlling segment
    curvature (0 = linear). The curve direction is left-to-right in time;
    Vital loops at the right edge.

    Parameters
    ----------
    curve_1d : array-like
        Any 1D signal — Hilbert envelope, IMF, polyrhythm coincidence pattern,
        harmonograph axis projection, Markov-walk amplitude trace, etc.
    n_points : int, default=16
        Number of control points to sample. More points = higher-fidelity
        reproduction but heavier preset file.
    normalize_to_unit : bool, default=True
        Rescale the curve's y-values to [0, 1]. Required by Vital's UI.

    Returns
    -------
    (points, powers) : (list[float], list[float])
        ``points`` is length 2 * n_points (interleaved x, y); ``powers`` is
        length n_points.
    """
    arr = np.asarray(curve_1d, dtype=np.float64).flatten()
    if arr.size < 2:
        return [0.0, 0.5, 1.0, 0.5], [0.0, 0.0]

    src_x = np.linspace(0.0, 1.0, arr.size)
    dst_x = np.linspace(0.0, 1.0, n_points)
    resampled = np.interp(dst_x, src_x, arr)

    if normalize_to_unit:
        lo, hi = float(resampled.min()), float(resampled.max())
        if hi > lo:
            resampled = (resampled - lo) / (hi - lo)
        else:
            resampled = np.full_like(resampled, 0.5)

    points: list[float] = []
    for x, y in zip(dst_x, resampled):
        points.extend([float(x), float(y)])
    powers = [0.0] * n_points
    return points, powers


def modulator_to_routing(
    mod,
    *,
    lfo_index: int,
    destination_for_am: str = "osc_1_level",
    destination_for_fm: str = "osc_1_distortion_amount",
    fm_normalization_hz: float | None = None,
) -> dict:
    """Convert a :class:`Modulator` dataclass into a routing dict ready
    for ``set_lfo_from_curve`` + ``set_modulation``.

    Parameters
    ----------
    mod : Modulator
    lfo_index : int (0..7)
    destination_for_am : str, default='osc_1_level'
    destination_for_fm : str, default='osc_1_distortion_amount'
    fm_normalization_hz : float, optional
        For FM modulators, ``mod.depth`` is in Hz; Vital's amount is in
        [0, 1]. Divides by this value (default = ``5 * mod.mod_freq``,
        a heuristic that makes β=5 land at amount=1).

    Returns
    -------
    dict
        ``{'lfo_index', 'source', 'destination', 'amount', 'rate_hz', 'name'}``.
    """
    rate_hz = float(mod.mod_freq)
    if mod.mod_type == "AM":
        destination = destination_for_am
        amount = float(np.clip(mod.depth, 0.0, 1.0))
    elif mod.mod_type == "FM":
        destination = destination_for_fm
        norm = fm_normalization_hz if fm_normalization_hz is not None else max(rate_hz * 5.0, 1e-3)
        amount = float(np.clip(mod.depth / norm, 0.0, 1.0))
    else:
        raise ValueError(f"unknown mod_type {mod.mod_type!r}")

    return {
        "lfo_index": lfo_index,
        "source": f"lfo_{lfo_index + 1}",
        "destination": destination,
        "amount": amount,
        "rate_hz": rate_hz,
        "name": (mod.source or f"{mod.mod_type}_lfo_{lfo_index}")[:40],
    }


# ---------------------------------------------------------------------------
# Placer: wavetable
# ---------------------------------------------------------------------------

def _encode_cycle(cycle: np.ndarray) -> str:
    """Base64-encode one 2048-sample float32 cycle for a Wave Source keyframe."""
    a = np.asarray(cycle, dtype=np.float32, order="C").flatten()
    if a.size != WAVETABLE_FRAME_SIZE:
        raise ValueError(
            f"_encode_cycle: cycle must have {WAVETABLE_FRAME_SIZE} samples; got {a.size}"
        )
    return base64.b64encode(a.tobytes()).decode("ascii")


def replace_wavetable(
    preset: dict,
    osc_index: int,
    audio: np.ndarray,
    *,
    name: str = "biotuner_wt",
) -> None:
    """Replace one wavetable slot with a Wave Source built from concatenated cycles.

    Each chunk of ``WAVETABLE_FRAME_SIZE`` (2048) samples becomes one keyframe.
    Keyframe positions are spread evenly from 0 to 255.
    """
    if osc_index not in (0, 1, 2):
        raise ValueError(f"osc_index must be 0/1/2; got {osc_index}")
    a = np.asarray(audio, dtype=np.float32).flatten()
    if a.size % WAVETABLE_FRAME_SIZE != 0:
        raise ValueError(
            f"audio length {a.size} not a multiple of {WAVETABLE_FRAME_SIZE}"
        )
    n_frames = a.size // WAVETABLE_FRAME_SIZE
    if n_frames == 1:
        positions = [0]
    else:
        positions = [int(round(i * (255 / max(n_frames - 1, 1)))) for i in range(n_frames)]
    keyframes = [
        {"position": int(pos), "wave_data": _encode_cycle(a[i * WAVETABLE_FRAME_SIZE:(i + 1) * WAVETABLE_FRAME_SIZE])}
        for i, pos in enumerate(positions)
    ]
    preset["settings"]["wavetables"][osc_index] = {
        "name": name,
        "author": "",
        "version": "1.0.0",
        "full_normalize": False,
        "remove_all_dc": False,
        "groups": [{
            "components": [{
                "type": "Wave Source",
                "interpolation": 1,
                "interpolation_style": 1,
                "keyframes": keyframes,
            }],
        }],
    }


# ---------------------------------------------------------------------------
# Placer: oscillator parameters
# ---------------------------------------------------------------------------

def populate_osc(
    preset: dict,
    osc_index: int,
    *,
    wavetable_audio: np.ndarray | None = None,
    wavetable_name: str = "biotuner_wt",
    transpose: float = 0.0,
    fine_cents: float = 0.0,
    level: float = 0.7,
    pan: float = 0.0,
    on: bool = True,
    destination: float | None = None,
    distortion_type: float | None = None,
    distortion_amount: float | None = None,
) -> None:
    """One-call population of all relevant settings keys for one oscillator.

    Sets enable/level/pan/transpose/tune in one call; optionally also
    replaces the oscillator's wavetable, sets routing destination, and
    enables a distortion type+amount (FM, RM, AM, sync, etc.).
    """
    if osc_index not in (0, 1, 2):
        raise ValueError(f"osc_index must be 0/1/2; got {osc_index}")

    if wavetable_audio is not None:
        replace_wavetable(preset, osc_index, wavetable_audio, name=wavetable_name)

    s = preset["settings"]
    n = osc_index + 1
    if f"osc_{n}_on" in s:
        s[f"osc_{n}_on"] = 1.0 if on else 0.0
    if f"osc_{n}_level" in s:
        s[f"osc_{n}_level"] = float(level)
    if f"osc_{n}_pan" in s:
        s[f"osc_{n}_pan"] = float(pan)
    if f"osc_{n}_transpose" in s:
        s[f"osc_{n}_transpose"] = float(transpose)
    if f"osc_{n}_tune" in s:
        s[f"osc_{n}_tune"] = float(fine_cents) / 100.0

    if destination is not None and f"osc_{n}_destination" in s:
        s[f"osc_{n}_destination"] = float(destination)
    if distortion_type is not None and f"osc_{n}_distortion_type" in s:
        s[f"osc_{n}_distortion_type"] = float(distortion_type)
    if distortion_amount is not None and f"osc_{n}_distortion_amount" in s:
        s[f"osc_{n}_distortion_amount"] = float(distortion_amount)


# ---------------------------------------------------------------------------
# Placer: LFO
# ---------------------------------------------------------------------------

def set_lfo_from_curve(
    preset: dict,
    *,
    lfo_index: int,
    curve_1d,
    rate_hz: float = 1.0,
    name: str = "biotuner_curve",
    n_points: int = 16,
    smooth: bool = True,
) -> None:
    """Replace LFO ``lfo_index`` (0..7) with a custom shape from any 1D curve.

    The curve is resampled to ``n_points`` control points and normalized to
    [0, 1] for Vital's points/powers format. ``rate_hz`` is set on the
    matching ``lfo_N_frequency`` settings key (Vital uses ``log2(seconds)``).
    """
    if not 0 <= lfo_index < 8:
        raise ValueError(f"lfo_index must be 0..7; got {lfo_index}")

    points, powers = curve_to_lfo_points(curve_1d, n_points=n_points)
    preset["settings"]["lfos"][lfo_index] = {
        "name": name,
        "num_points": n_points,
        "points": points,
        "powers": powers,
        "smooth": smooth,
    }

    n = lfo_index + 1
    s = preset["settings"]
    if f"lfo_{n}_frequency" in s:
        s[f"lfo_{n}_frequency"] = float(math.log2(max(rate_hz, 1e-3)))
    if f"lfo_{n}_sync" in s:
        s[f"lfo_{n}_sync"] = 0.0


def set_lfo_sine(
    preset: dict,
    *,
    lfo_index: int,
    rate_hz: float,
    name: str = "biotuner_sine",
) -> None:
    """Convenience wrapper: set an LFO to a sine shape at ``rate_hz``."""
    sine_curve = np.sin(2.0 * np.pi * np.linspace(0.0, 1.0, 32, endpoint=False))
    set_lfo_from_curve(
        preset,
        lfo_index=lfo_index,
        curve_1d=sine_curve,
        rate_hz=rate_hz,
        name=name,
        n_points=4,
        smooth=True,
    )


# ---------------------------------------------------------------------------
# Placer: modulation matrix entry
# ---------------------------------------------------------------------------

def find_free_modulation_slot(preset: dict) -> int:
    """Return the index (0..63) of the first unused modulation slot."""
    mods = preset["settings"]["modulations"]
    for i, m in enumerate(mods):
        if not m.get("source") and not m.get("destination"):
            return i
    raise RuntimeError("no free modulation slots in template (all 64 used)")


def set_modulation(
    preset: dict,
    *,
    slot: int,
    source: str,
    destination: str,
    amount: float = 0.5,
    bipolar: bool = False,
) -> None:
    """Patch a specific modulation slot AND set its companion settings keys
    (``modulation_N_amount``, ``modulation_N_bipolar``, etc.)."""
    mods = preset["settings"]["modulations"]
    if not 0 <= slot < len(mods):
        raise IndexError(f"slot {slot} out of range 0..{len(mods)-1}")
    mods[slot] = {"source": source, "destination": destination}
    n = slot + 1
    s = preset["settings"]
    s[f"modulation_{n}_amount"] = float(amount)
    if f"modulation_{n}_bipolar" in s:
        s[f"modulation_{n}_bipolar"] = 1.0 if bipolar else 0.0
    if f"modulation_{n}_bypass" in s:
        s[f"modulation_{n}_bypass"] = 0.0
    if f"modulation_{n}_stereo" in s:
        s[f"modulation_{n}_stereo"] = 0.0


# ---------------------------------------------------------------------------
# Convenience: combined LFO routing
# ---------------------------------------------------------------------------

def route_lfo(
    preset: dict,
    *,
    lfo_index: int,
    destination: str,
    amount: float = 0.5,
    bipolar: bool = False,
) -> int:
    """Find a free modulation slot, route ``lfo_(lfo_index+1)`` to ``destination``.

    Returns the slot index used.
    """
    slot = find_free_modulation_slot(preset)
    set_modulation(
        preset,
        slot=slot,
        source=f"lfo_{lfo_index + 1}",
        destination=destination,
        amount=amount,
        bipolar=bipolar,
    )
    return slot


# ---------------------------------------------------------------------------
# Character placers — filter / sample / envelope / unison / effects
# ---------------------------------------------------------------------------

def set_filter_1(
    preset: dict,
    *,
    on: bool = True,
    model: int = 0,
    cutoff: float = 80.0,
    resonance: float = 0.3,
    mix: float = 1.0,
    drive: float = 0.0,
) -> None:
    """Configure filter 1.

    ``model`` codes (matches Vital's enum, may differ across versions; tested
    against 1.5.x):
        0 = Analog low-pass
        1 = Analog high-pass
        2 = Analog band-pass
        3 = Analog notch
        4 = Digital low-pass
        5 = Digital high-pass
        6 = Comb (default in Plucked String — heavy resonance)
        7 = Ladder
        8 = Diode
        9 = Formant

    ``cutoff`` is in Vital's MIDI-note-like range (0..127, where 60 ≈ middle C).
    """
    s = preset["settings"]
    if "filter_1_on" in s:
        s["filter_1_on"] = 1.0 if on else 0.0
    if "filter_1_model" in s:
        s["filter_1_model"] = float(model)
    if "filter_1_cutoff" in s:
        s["filter_1_cutoff"] = float(cutoff)
    if "filter_1_resonance" in s:
        s["filter_1_resonance"] = float(np.clip(resonance, 0.0, 1.0))
    if "filter_1_mix" in s:
        s["filter_1_mix"] = float(np.clip(mix, 0.0, 1.0))
    if "filter_1_drive" in s:
        s["filter_1_drive"] = float(drive)


def set_sample_osc(
    preset: dict,
    *,
    on: bool = False,
    level: float = 0.0,
) -> None:
    """Toggle the sample oscillator and set its level.

    Plucked String ships with the HVAC Unit sample loaded; we leave it loaded
    for compatibility but disable its level (= silent) by default. Pass
    ``on=True, level=0.5`` to use the sample slot in a preset like
    ``hilbert_sample`` where the user will load their own audio.
    """
    s = preset["settings"]
    if "sample_on" in s:
        s["sample_on"] = 1.0 if on else 0.0
    if "sample_level" in s:
        s["sample_level"] = float(np.clip(level, 0.0, 1.0))


def set_envelope(
    preset: dict,
    env_idx: int,
    *,
    attack: float | None = None,
    decay: float | None = None,
    sustain: float | None = None,
    release: float | None = None,
) -> None:
    """Set envelope ADSR. Each value is in Vital's normalized [0, 1] range
    (where ~0.0 = instant, ~1.0 = many seconds). Pass None to leave a
    parameter at its template value.
    """
    if env_idx not in (1, 2, 3, 4):
        raise ValueError(f"env_idx must be 1..4; got {env_idx}")
    s = preset["settings"]
    if attack is not None and f"env_{env_idx}_attack" in s:
        s[f"env_{env_idx}_attack"] = float(np.clip(attack, 0.0, 1.0))
    if decay is not None and f"env_{env_idx}_decay" in s:
        s[f"env_{env_idx}_decay"] = float(np.clip(decay, 0.0, 1.0))
    if sustain is not None and f"env_{env_idx}_sustain" in s:
        s[f"env_{env_idx}_sustain"] = float(np.clip(sustain, 0.0, 1.0))
    if release is not None and f"env_{env_idx}_release" in s:
        s[f"env_{env_idx}_release"] = float(np.clip(release, 0.0, 1.0))


def set_unison(
    preset: dict,
    osc_index: int,
    *,
    voices: int = 1,
    detune: float = 4.5,
) -> None:
    """Set unison voices and detune for an oscillator (fattening)."""
    if osc_index not in (0, 1, 2):
        raise ValueError(f"osc_index must be 0/1/2; got {osc_index}")
    n = osc_index + 1
    s = preset["settings"]
    if f"osc_{n}_unison_voices" in s:
        s[f"osc_{n}_unison_voices"] = float(max(1, voices))
    if f"osc_{n}_unison_detune" in s:
        s[f"osc_{n}_unison_detune"] = float(detune)


_EFFECT_NAMES = (
    "reverb", "delay", "chorus", "distortion", "compressor",
    "eq", "flanger", "phaser",
)


def set_effect(
    preset: dict,
    name: str,
    *,
    on: bool | None = None,
    **kwargs,
) -> None:
    """Toggle / configure one of Vital's effects.

    ``name`` ∈ ``_EFFECT_NAMES``. Extra kwargs map to ``<name>_<key>``
    settings keys (e.g. ``set_effect(preset, 'reverb', dry_wet=0.4, size=0.7)``
    sets ``reverb_dry_wet`` and ``reverb_size``).
    """
    if name not in _EFFECT_NAMES:
        raise ValueError(f"unknown effect {name!r}. Known: {_EFFECT_NAMES}")
    s = preset["settings"]
    if on is not None:
        key = f"{name}_on"
        if key in s:
            s[key] = 1.0 if on else 0.0
    for k, v in kwargs.items():
        full = f"{name}_{k}"
        if full in s:
            s[full] = float(v)


# Filter model enum (Vital 1.5.x; may shift in future versions)
FILTER_MODELS = {
    "lp":      0,    # Analog low-pass
    "hp":      1,    # Analog high-pass
    "bp":      2,    # Analog band-pass
    "notch":   3,    # Analog notch
    "lp_d":    4,    # Digital low-pass
    "hp_d":    5,    # Digital high-pass
    "comb":    6,    # Comb (dominant character of "Plucked String")
    "ladder":  7,    # Moog-style ladder
    "diode":   8,    # Diode ladder
    "formant": 9,    # Formant
}


# Envelope ADSR characters (in Vital's normalized 0..1 range — power-curved)
ENVELOPE_CHARACTERS = {
    # name           attack   decay  sustain  release
    "pad":         {"attack": 0.35, "decay": 0.5,  "sustain": 1.0, "release": 0.55},
    "bell":        {"attack": 0.0,  "decay": 0.7,  "sustain": 0.0, "release": 0.3},
    "pluck":       {"attack": 0.0,  "decay": 0.4,  "sustain": 0.0, "release": 0.2},
    "percussive":  {"attack": 0.0,  "decay": 0.15, "sustain": 0.0, "release": 0.1},
    "drone":       {"attack": 0.6,  "decay": 0.7,  "sustain": 1.0, "release": 0.7},
    "organ":       {"attack": 0.05, "decay": 0.2,  "sustain": 1.0, "release": 0.15},
    "swell":       {"attack": 0.55, "decay": 0.4,  "sustain": 0.7, "release": 0.45},
    "stab":        {"attack": 0.0,  "decay": 0.25, "sustain": 0.5, "release": 0.15},
}


def set_envelope_character(
    preset: dict, env_idx: int, character: str | None,
) -> None:
    """Apply a named ADSR character to envelope env_idx.

    ``None`` (or the literal string ``'None'``) leaves the envelope at the
    template's default. Unknown characters raise.
    """
    if character is None or character == "None":
        return
    if character not in ENVELOPE_CHARACTERS:
        raise ValueError(
            f"unknown envelope character {character!r}. "
            f"Known: {sorted(ENVELOPE_CHARACTERS)}"
        )
    set_envelope(preset, env_idx, **ENVELOPE_CHARACTERS[character])


def set_filter_1_character(
    preset: dict,
    *,
    character: str,
    cutoff: float = 80.0,
    resonance: float = 0.3,
    drive: float = 0.0,
) -> None:
    """Apply a named filter character ('lp', 'hp', 'comb', 'formant', ...)."""
    if character == "off":
        set_filter_1(preset, on=False)
        return
    if character not in FILTER_MODELS:
        raise ValueError(
            f"unknown filter character {character!r}. "
            f"Known: 'off' or {sorted(FILTER_MODELS)}"
        )
    set_filter_1(
        preset, on=True,
        model=FILTER_MODELS[character],
        cutoff=cutoff, resonance=resonance, drive=drive,
    )


def apply_minimal_baseline(preset: dict) -> None:
    """Reset the Plucked-String template to a neutral baseline:
    sample silenced, filter 1 off, distortion off, reverb modest.

    Each preset generator can call this first then opt in to specific
    character settings. Without this baseline, the template's heavy comb
    filter + distortion dominate every output regardless of other patching.
    """
    set_sample_osc(preset, on=False, level=0.0)
    set_filter_1(preset, on=False)
    set_effect(preset, "distortion", on=False)
    set_effect(preset, "chorus", on=False)
    set_effect(preset, "delay", on=False)
    set_effect(preset, "flanger", on=False)
    set_effect(preset, "phaser", on=False)
    # Keep reverb on but moderate
    set_effect(preset, "reverb", on=True, dry_wet=0.25, size=0.6)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def write_vital_preset(preset: dict, out_path: str) -> str:
    """Write a Vital preset to disk. Adds ``.vital`` if missing."""
    if not out_path.endswith(".vital"):
        out_path = out_path + ".vital"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(preset, fp, default=_json_default)
    return out_path


def write_settings_companion(
    preset_kind: str,
    *,
    out_path: str,
    base_freq: float,
    timbre_summary: dict,
    extras: dict | None = None,
) -> str:
    """Write a stable, version-agnostic JSON companion describing the patch."""
    if not out_path.endswith(".json"):
        out_path = out_path + ".json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    payload = {
        "format": "biotuner_vital_companion",
        "format_version": 3,
        "preset_kind": preset_kind,
        "vital_template_source": "Plucked String (Vital factory preset)",
        "base_freq": float(base_freq),
        "timbre": dict(timbre_summary),
    }
    if extras:
        payload.update({k: v for k, v in extras.items() if k not in payload})
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, default=_json_default)
    return out_path


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
