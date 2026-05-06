"""Combinatorial Vital preset generator.

Module type: Functions

One generator: :func:`to_vital_combinatorial`. Takes a :class:`PresetSpec`
that declares which biotuner sources fill which Vital slots and how they're
routed, plus a ``sources`` dict carrying the actual data.

The full surface — every dimension is independently settable:

    osc1 / osc2 / osc3       OscSpec       — wavetable source, transpose, level,
                                              unison, distortion type/amount,
                                              cross-osc routing destination
    sample                   OscSpec       — sample-osc on/off, level (uses
                                              'hilbert_audio' source to bake
                                              biosignal-as-audio into the slot)
    lfos[0..7]               list[LFOSpec] — per-LFO source, rate, smooth flag,
                                              and a list of routings
    filter_1 / filter_2      FilterSpec    — character + cutoff + resonance + drive
    env_1 / env_2 / env_3 / env_4   EnvelopeSpec  — named character (pad / bell / pluck / …)
    effects                  EffectSpec    — per-effect on/off + dry_wet + …
    macros[0..3]             list[MacroSpec] — macro names + their matrix routings
    free_modulations         list          — arbitrary (source, dest, amount) tuples
    extra_settings           dict          — escape hatch for any of the 775 settings
                                              keys not exposed above

Sample sources:
    'matched_timbre', 'inharmonic', 'imf_0..N', 'sine', 'square',
    'harmonograph_x', 'harmonograph_y', 'polyrhythm_wave',
    'lissajous_x', 'lissajous_y'

LFO sources (curve-typed):
    'hilbert_envelope', 'imf_0..N', 'polyrhythm_gate',
    'harmonograph_x', 'harmonograph_y', 'markov_walk',
    'sine', 'square', 'triangle', 'random_noise'

A free helper :func:`random_spec` samples a PresetSpec from the available
sources — drop into a ``for`` loop to generate N distinct preset variants
from one biotuner state.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from biotuner.harmonic_timbre.exporters._vital_template import (
    WAVETABLE_FRAME_SIZE,
    apply_minimal_baseline,
    find_free_modulation_slot,
    make_blank_preset,
    populate_osc,
    replace_wavetable,
    route_lfo,
    set_effect,
    set_envelope_character,
    set_filter_1_character,
    set_lfo_from_curve,
    set_modulation,
    set_sample_osc,
    set_unison,
    write_settings_companion,
    write_vital_preset,
)
from biotuner.harmonic_timbre.synthesis import render_wavetable_cycle
from biotuner.harmonic_timbre.timbre import Timbre


# ---------------------------------------------------------------------------
# Spec dataclasses
# ---------------------------------------------------------------------------

@dataclass
class OscSpec:
    """One oscillator slot specification."""
    source: str | None = None
    """Wavetable source ID. None = osc disabled. See module docstring for the
    list of legal sources."""
    level: float = 0.7
    pan: float = 0.0
    transpose: float = 0.0          # semitones
    fine_cents: float = 0.0
    unison_voices: int = 1
    unison_detune: float = 4.5
    distortion_type: int | None = None     # None = no distortion; 9 = FM
    distortion_amount: float = 0.0
    destination: int | None = None         # None = bypass; 1 → osc 1, 2 → osc 2


@dataclass
class LFOSpec:
    """One LFO slot specification."""
    source: str | None = None
    rate_hz: float = 1.0
    smooth: bool = True
    n_points: int = 16
    routings: list = field(default_factory=list)
    """List of ``(destination_str, amount, bipolar)`` triplets. Multiple
    destinations per LFO are allowed (each consumes one mod-matrix slot)."""


@dataclass
class FilterSpec:
    character: str = "off"   # 'off' | 'lp' | 'hp' | 'bp' | 'notch' | 'comb' | 'ladder' | 'diode' | 'formant'
    cutoff: float = 80.0
    resonance: float = 0.3
    drive: float = 0.0


@dataclass
class EnvelopeSpec:
    character: str | None = None   # 'pad' | 'bell' | 'pluck' | 'percussive' | 'drone' | 'organ' | 'swell' | 'stab' | None (leave at template default)


@dataclass
class EffectSpec:
    reverb: bool = True
    reverb_size: float = 0.6
    reverb_dry_wet: float = 0.3
    delay: bool = False
    delay_dry_wet: float = 0.3
    delay_sync: bool = True
    delay_frequency: float = 0.0   # log2 seconds
    chorus: bool = False
    chorus_dry_wet: float = 0.3
    distortion: bool = False
    distortion_type: int = 0
    distortion_drive: float = 0.5
    compressor: bool = False
    eq: bool = False
    flanger: bool = False
    phaser: bool = False


@dataclass
class MacroSpec:
    name: str = "Macro"
    initial: float = 0.5
    routings: list = field(default_factory=list)
    """List of ``(destination_str, amount)`` tuples mapping this macro."""


@dataclass
class PresetSpec:
    """The full combinatorial spec — every dimension is independently settable.

    Use :func:`random_spec` to sample a random one for stochastic exploration,
    or build manually for deliberate preset design. See module docstring.
    """
    name: str = "biotuner_combinatorial"
    osc1: OscSpec = field(default_factory=lambda: OscSpec(source="matched_timbre"))
    osc2: OscSpec = field(default_factory=OscSpec)
    osc3: OscSpec = field(default_factory=OscSpec)
    sample: OscSpec = field(default_factory=OscSpec)
    lfos: list[LFOSpec] = field(default_factory=lambda: [LFOSpec() for _ in range(8)])
    filter_1: FilterSpec = field(default_factory=FilterSpec)
    filter_2: FilterSpec = field(default_factory=FilterSpec)
    env_1: EnvelopeSpec = field(default_factory=EnvelopeSpec)   # amp envelope
    env_2: EnvelopeSpec = field(default_factory=EnvelopeSpec)   # mod env A
    env_3: EnvelopeSpec = field(default_factory=EnvelopeSpec)
    env_4: EnvelopeSpec = field(default_factory=EnvelopeSpec)
    effects: EffectSpec = field(default_factory=EffectSpec)
    macros: list[MacroSpec] = field(default_factory=list)
    free_modulations: list = field(default_factory=list)        # list of (source, dest, amount, bipolar)
    extra_settings: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Source resolvers
# ---------------------------------------------------------------------------

OSC_SOURCES = (
    "matched_timbre", "inharmonic", "sine", "square",
    "harmonograph_x", "harmonograph_y", "polyrhythm_wave",
    "lissajous_x", "lissajous_y",
    # plus dynamic 'imf_0', 'imf_1', ...
)

LFO_SOURCES = (
    "hilbert_envelope", "polyrhythm_gate",
    "harmonograph_x", "harmonograph_y", "markov_walk",
    "sine", "square", "triangle", "random_noise",
    # plus dynamic 'imf_0', 'imf_1', ...
)


def _resolve_osc_source(source: str, sources: dict, *, table_size: int = WAVETABLE_FRAME_SIZE) -> np.ndarray | None:
    """Resolve an osc source string to a 2048-sample wavetable cycle."""
    if source is None:
        return None
    if source == "matched_timbre":
        timbre = sources.get("timbre")
        if timbre is None:
            return None
        return render_wavetable_cycle(timbre, table_size=table_size)
    if source == "inharmonic":
        timbre_inh = sources.get("timbre_inharmonic") or sources.get("timbre")
        if timbre_inh is None:
            return None
        return render_wavetable_cycle(timbre_inh, table_size=table_size)
    if source == "sine":
        return np.sin(2 * np.pi * np.arange(table_size) / table_size).astype(np.float32)
    if source == "square":
        return np.sign(np.sin(2 * np.pi * np.arange(table_size) / table_size)).astype(np.float32)
    if source.startswith("imf_"):
        idx = int(source.split("_", 1)[1])
        imfs = sources.get("imfs") or []
        if idx >= len(imfs):
            return None
        return _resample_to(imfs[idx], table_size).astype(np.float32)
    if source in ("harmonograph_x", "harmonograph_y"):
        hg = sources.get("harmonograph")
        if hg is None:
            return None
        axis_idx = 0 if source.endswith("x") else 1
        return _resample_to(hg[axis_idx], table_size).astype(np.float32)
    if source == "polyrhythm_wave":
        gate = sources.get("polyrhythm_gate")
        if gate is None:
            return None
        return _resample_to(np.asarray(gate, dtype=np.float64), table_size).astype(np.float32)
    if source in ("lissajous_x", "lissajous_y"):
        liss = sources.get("lissajous")
        if liss is None:
            return None
        axis_idx = 0 if source.endswith("x") else 1
        return _resample_to(liss[axis_idx], table_size).astype(np.float32)
    raise ValueError(f"unknown osc source {source!r}")


def _resolve_lfo_curve(source: str, sources: dict) -> np.ndarray | None:
    """Resolve an LFO source string to a 1D curve (any length)."""
    if source is None:
        return None
    if source == "hilbert_envelope":
        from scipy.signal import hilbert
        sig = sources.get("signal")
        if sig is None:
            return None
        return np.abs(hilbert(np.asarray(sig, dtype=np.float64).flatten()))
    if source.startswith("imf_"):
        idx = int(source.split("_", 1)[1])
        imfs = sources.get("imfs") or []
        if idx >= len(imfs):
            return None
        return np.asarray(imfs[idx], dtype=np.float64)
    if source == "polyrhythm_gate":
        gate = sources.get("polyrhythm_gate")
        return np.asarray(gate, dtype=np.float64) if gate is not None else None
    if source in ("harmonograph_x", "harmonograph_y"):
        hg = sources.get("harmonograph")
        if hg is None:
            return None
        return np.asarray(hg[0 if source.endswith("x") else 1], dtype=np.float64)
    if source == "markov_walk":
        walk = sources.get("markov_walk")
        return np.asarray(walk, dtype=np.float64) if walk is not None else None
    if source == "sine":
        return np.sin(2 * np.pi * np.linspace(0, 1, 64))
    if source == "square":
        return np.sign(np.sin(2 * np.pi * np.linspace(0, 1, 64)))
    if source == "triangle":
        return 2 * np.abs(2 * (np.linspace(0, 1, 64) % 1) - 1) - 1
    if source == "random_noise":
        rng = np.random.default_rng(int(sources.get("seed", 0)))
        return rng.uniform(-1, 1, 64)
    raise ValueError(f"unknown lfo source {source!r}")


def _resample_to(curve, n: int) -> np.ndarray:
    arr = np.asarray(curve, dtype=np.float64).flatten()
    if arr.size == n:
        return arr
    if arr.size < 2:
        return np.zeros(n, dtype=np.float64)
    src_x = np.linspace(0, 1, arr.size)
    dst_x = np.linspace(0, 1, n)
    out = np.interp(dst_x, src_x, arr)
    m = float(np.max(np.abs(out)))
    if m > 0:
        out = out / m * 0.99
    return out


# ---------------------------------------------------------------------------
# Combinatorial generator
# ---------------------------------------------------------------------------

def to_vital_combinatorial(
    spec: PresetSpec,
    out_path: str,
    sources: dict[str, Any],
) -> dict:
    """Generate a Vital preset from a :class:`PresetSpec` + a ``sources`` dict.

    ``sources`` carries the actual data referenced by source-string IDs in
    the spec. Standard keys (all optional — only the ones referenced by the
    spec's source IDs need to be present):

        timbre              : Timbre              for 'matched_timbre' osc source
        timbre_inharmonic   : Timbre              for 'inharmonic' osc source
        imfs                : list[np.ndarray]    for 'imf_N' osc/lfo sources
        signal              : np.ndarray          for 'hilbert_envelope' lfo source
        sf                  : float               sample rate of signal
        polyrhythm_gate     : np.ndarray          for 'polyrhythm_*' osc/lfo sources
        harmonograph        : (np.ndarray, np.ndarray)   for 'harmonograph_x/y' sources
        lissajous           : (np.ndarray, np.ndarray)   for 'lissajous_x/y' sources
        markov_walk         : np.ndarray          for 'markov_walk' lfo source
        seed                : int                 for deterministic 'random_noise'

    Returns ``{'vital', 'companion', 'sample_wav': <only for sample.source='hilbert_audio'>}``.
    """
    out_path = out_path if out_path.endswith(".vital") else out_path + ".vital"
    name = spec.name or os.path.splitext(os.path.basename(out_path))[0]

    preset = make_blank_preset(name=name, comments=_render_comments(spec, sources))
    apply_minimal_baseline(preset)

    # ---- Oscillators 1, 2, 3 ----
    for osc_idx, osc_spec in enumerate([spec.osc1, spec.osc2, spec.osc3]):
        if osc_spec is None or osc_spec.source is None:
            continue
        cycle = _resolve_osc_source(osc_spec.source, sources)
        if cycle is None:
            continue
        populate_osc(
            preset, osc_idx,
            wavetable_audio=cycle,
            wavetable_name=osc_spec.source[:24],
            transpose=osc_spec.transpose,
            fine_cents=osc_spec.fine_cents,
            level=osc_spec.level, pan=osc_spec.pan,
            on=True,
            destination=float(osc_spec.destination) if osc_spec.destination is not None else None,
            distortion_type=float(osc_spec.distortion_type) if osc_spec.distortion_type is not None else None,
            distortion_amount=osc_spec.distortion_amount,
        )
        if osc_spec.unison_voices > 1 or osc_spec.unison_detune != 4.5:
            set_unison(preset, osc_idx,
                       voices=osc_spec.unison_voices, detune=osc_spec.unison_detune)

    # ---- Sample osc ----
    sample_wav_path: str | None = None
    if spec.sample is not None and spec.sample.source is not None:
        if spec.sample.source == "hilbert_audio":
            from biotuner.harmonic_timbre import hilbert_instrument
            import soundfile as sf_io
            sig = sources.get("signal")
            sr_in = sources.get("sf", 1000.0)
            if sig is not None:
                audio = hilbert_instrument(
                    sig, sf=float(sr_in), samplerate=48000,
                    duration=2.0, base_freq=220.0, pitch_factor=50.0,
                )
                sample_wav_path = out_path.replace(".vital", "_sample.wav")
                sf_io.write(sample_wav_path, audio, 48000)
                set_sample_osc(preset, on=True, level=spec.sample.level)
                # Note: Vital cannot pre-attach a sample file from a preset;
                # the user has to drag the WAV onto the Sample slot manually.
                preset["comments"] += (
                    f"\nDrag '{os.path.basename(sample_wav_path)}' onto Vital's Sample osc."
                )
        else:
            set_sample_osc(preset, on=True, level=spec.sample.level)

    # ---- LFOs 1..8 ----
    lfo_records: list[dict] = []
    for lfo_idx, lfo_spec in enumerate(spec.lfos[:8]):
        if lfo_spec.source is None:
            continue
        curve = _resolve_lfo_curve(lfo_spec.source, sources)
        if curve is None:
            continue
        set_lfo_from_curve(
            preset, lfo_index=lfo_idx, curve_1d=curve,
            rate_hz=lfo_spec.rate_hz,
            name=f"{lfo_spec.source} ({lfo_spec.rate_hz:.2f} Hz)"[:40],
            n_points=lfo_spec.n_points, smooth=lfo_spec.smooth,
        )
        for routing in lfo_spec.routings:
            if isinstance(routing, tuple) and len(routing) == 3:
                dest, amount, bipolar = routing
            elif isinstance(routing, tuple) and len(routing) == 2:
                dest, amount = routing; bipolar = False
            else:
                continue
            try:
                route_lfo(preset, lfo_index=lfo_idx,
                          destination=str(dest), amount=float(amount),
                          bipolar=bool(bipolar))
            except RuntimeError:
                # Out of mod slots; bail gracefully
                break
        lfo_records.append({
            "lfo_index": lfo_idx,
            "source": lfo_spec.source,
            "rate_hz": lfo_spec.rate_hz,
            "n_points": lfo_spec.n_points,
            "smooth": lfo_spec.smooth,
            "n_routings": len(lfo_spec.routings),
        })

    # ---- Filters ----
    set_filter_1_character(
        preset, character=spec.filter_1.character,
        cutoff=spec.filter_1.cutoff,
        resonance=spec.filter_1.resonance,
        drive=spec.filter_1.drive,
    )
    if spec.filter_2.character != "off":
        s = preset["settings"]
        if "filter_2_on" in s: s["filter_2_on"] = 1.0
        from biotuner.harmonic_timbre.exporters._vital_template import FILTER_MODELS
        if "filter_2_model" in s and spec.filter_2.character in FILTER_MODELS:
            s["filter_2_model"] = float(FILTER_MODELS[spec.filter_2.character])
        if "filter_2_cutoff" in s: s["filter_2_cutoff"] = float(spec.filter_2.cutoff)
        if "filter_2_resonance" in s: s["filter_2_resonance"] = float(spec.filter_2.resonance)

    # ---- Envelopes ----
    for env_idx, env_spec in enumerate([spec.env_1, spec.env_2, spec.env_3, spec.env_4], start=1):
        if env_spec.character is not None:
            set_envelope_character(preset, env_idx, env_spec.character)

    # ---- Effects ----
    e = spec.effects
    set_effect(preset, "reverb", on=e.reverb, dry_wet=e.reverb_dry_wet, size=e.reverb_size)
    set_effect(preset, "delay", on=e.delay, dry_wet=e.delay_dry_wet,
               sync=1.0 if e.delay_sync else 0.0, frequency=e.delay_frequency)
    set_effect(preset, "chorus", on=e.chorus, dry_wet=e.chorus_dry_wet)
    set_effect(preset, "distortion", on=e.distortion,
               type=e.distortion_type, drive=e.distortion_drive)
    set_effect(preset, "compressor", on=e.compressor)
    set_effect(preset, "eq", on=e.eq)
    set_effect(preset, "flanger", on=e.flanger)
    set_effect(preset, "phaser", on=e.phaser)

    # ---- Macros ----
    for macro_idx, macro_spec in enumerate(spec.macros[:4]):
        preset[f"macro{macro_idx + 1}"] = macro_spec.name[:40]
        for routing in macro_spec.routings:
            if not isinstance(routing, tuple) or len(routing) != 2:
                continue
            dest, amount = routing
            try:
                slot = find_free_modulation_slot(preset)
            except RuntimeError:
                break
            set_modulation(
                preset, slot=slot,
                source=f"macro_control_{macro_idx + 1}",
                destination=str(dest), amount=float(amount),
            )

    # ---- Free modulations ----
    for routing in spec.free_modulations:
        try:
            if len(routing) == 4:
                src, dst, amt, bp = routing
            else:
                src, dst, amt = routing[:3]; bp = False
        except (TypeError, ValueError):
            continue
        try:
            slot = find_free_modulation_slot(preset)
        except RuntimeError:
            break
        set_modulation(preset, slot=slot,
                       source=str(src), destination=str(dst),
                       amount=float(amt), bipolar=bool(bp))

    # ---- Escape hatch: any extra settings keys ----
    for key, value in spec.extra_settings.items():
        if key in preset["settings"]:
            preset["settings"][key] = value

    # ---- Persist ----
    out_v = write_vital_preset(preset, out_path)
    out_c = write_settings_companion(
        "combinatorial",
        out_path=out_v.replace(".vital", "_settings.json"),
        base_freq=float(sources.get("base_freq", 220.0)),
        timbre_summary=_summarize_sources(sources),
        extras={
            "spec": _spec_to_dict(spec),
            "lfo_records": lfo_records,
            "sample_wav": os.path.basename(sample_wav_path) if sample_wav_path else None,
        },
    )
    result = {"vital": out_v, "companion": out_c}
    if sample_wav_path is not None:
        result["sample_wav"] = sample_wav_path
    return result


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

def _render_comments(spec: PresetSpec, sources: dict) -> str:
    lines = ["biotuner.harmonic_timbre — combinatorial preset"]
    osc_summary = []
    for i, o in enumerate([spec.osc1, spec.osc2, spec.osc3], start=1):
        if o.source:
            osc_summary.append(f"osc{i}={o.source}@{o.transpose:+.0f}st x{o.unison_voices}")
    if osc_summary:
        lines.append("Oscillators: " + ", ".join(osc_summary))
    lfo_summary = [f"lfo{i+1}={l.source}" for i, l in enumerate(spec.lfos[:8]) if l.source]
    if lfo_summary:
        lines.append("LFOs: " + ", ".join(lfo_summary))
    lines.append(
        f"Filter1: {spec.filter_1.character}@{spec.filter_1.cutoff:.0f}/{spec.filter_1.resonance:.2f}"
    )
    lines.append(
        f"Env1: {spec.env_1.character or 'default'} | Effects: "
        + " ".join(name for name, on in [
            ("reverb", spec.effects.reverb), ("delay", spec.effects.delay),
            ("chorus", spec.effects.chorus), ("distortion", spec.effects.distortion),
            ("flanger", spec.effects.flanger), ("phaser", spec.effects.phaser),
        ] if on) or "(none)"
    )
    timbre = sources.get("timbre")
    if timbre is not None and timbre.matched_tuning is not None:
        rs = ", ".join(f"{r:.4f}" for r in timbre.matched_tuning[:6])
        lines.append(
            f"matched_tuning = [{rs}{'…' if len(timbre.matched_tuning) > 6 else ''}]"
        )
    return "\n".join(lines)


def _summarize_sources(sources: dict) -> dict:
    summary = {}
    timbre = sources.get("timbre")
    if timbre is not None:
        summary["timbre"] = {
            "n_partials": timbre.n_partials(),
            "matched_tuning": list(timbre.matched_tuning) if timbre.matched_tuning is not None else None,
            "matching_method": timbre.matching_method,
            "base_freq": float(timbre.base_freq),
        }
    if sources.get("imfs"):
        summary["imfs"] = {"n_imfs": len(sources["imfs"]),
                           "lengths": [int(np.asarray(x).size) for x in sources["imfs"]]}
    if sources.get("signal") is not None:
        summary["signal"] = {"n_samples": int(np.asarray(sources["signal"]).size),
                             "samplerate": float(sources.get("sf", 0.0))}
    return summary


def _spec_to_dict(spec: PresetSpec) -> dict:
    """Render a PresetSpec to a JSON-friendly dict for the companion file."""
    from dataclasses import asdict
    d = asdict(spec)
    # Ensure tuples become lists for JSON safety
    return d


# ---------------------------------------------------------------------------
# Random-spec sampler
# ---------------------------------------------------------------------------

def random_spec(
    *,
    rng: np.random.Generator | None = None,
    available_sources: set[str] | None = None,
    n_imfs: int = 4,
    name: str = "random_preset",
) -> PresetSpec:
    """Sample a random PresetSpec from the available source pool.

    ``available_sources`` is the set of *kinds* of source data that's actually
    populated in the user's biotuner state — e.g. ``{'matched_timbre',
    'imfs', 'signal', 'polyrhythm_gate'}``. Sources not in the set won't be
    referenced by the sampled spec, so the resulting preset is guaranteed
    to render with whatever data the user actually has.
    """
    rng = rng or np.random.default_rng()
    avail = available_sources or {"matched_timbre"}

    # Build osc-source pool from what's available
    osc_pool: list[str | None] = [None]
    if "matched_timbre" in avail:
        osc_pool.append("matched_timbre")
    if "inharmonic" in avail:
        osc_pool.append("inharmonic")
    if "imfs" in avail:
        osc_pool.extend([f"imf_{i}" for i in range(n_imfs)])
    osc_pool.append("sine")
    if "harmonograph" in avail:
        osc_pool.extend(["harmonograph_x", "harmonograph_y"])
    if "polyrhythm_gate" in avail:
        osc_pool.append("polyrhythm_wave")

    # LFO pool
    lfo_pool: list[str | None] = [None, None, None]   # bias toward "off"
    if "signal" in avail:
        lfo_pool.append("hilbert_envelope")
    if "imfs" in avail:
        lfo_pool.extend([f"imf_{i}" for i in range(n_imfs)])
    if "polyrhythm_gate" in avail:
        lfo_pool.append("polyrhythm_gate")
    if "harmonograph" in avail:
        lfo_pool.extend(["harmonograph_x", "harmonograph_y"])
    if "markov_walk" in avail:
        lfo_pool.append("markov_walk")
    lfo_pool.extend(["sine", "triangle"])

    # Routable destinations Vital understands
    lfo_dests = [
        "osc_1_level", "osc_1_pan", "osc_1_wavetable_position",
        "osc_2_level", "osc_2_transpose",
        "osc_3_level",
        "osc_1_distortion_amount", "osc_2_distortion_amount",
        "filter_1_cutoff", "filter_1_resonance", "filter_1_drive",
        "volume", "reverb_dry_wet", "reverb_size",
        "delay_dry_wet", "delay_frequency",
        "chorus_dry_wet", "distortion_drive",
    ]

    # Sample envelope and filter character
    env_chars = ["pad", "bell", "pluck", "percussive", "drone", "organ", "swell", "stab"]
    filter_chars = ["off", "off", "lp", "lp", "hp", "bp", "comb", "ladder"]

    spec = PresetSpec(
        name=name,
        osc1=OscSpec(
            source=str(rng.choice([s for s in osc_pool if s is not None])),
            level=float(rng.uniform(0.5, 0.85)),
            unison_voices=int(rng.choice([1, 1, 1, 3, 5])),
            unison_detune=float(rng.uniform(2.0, 8.0)),
            transpose=float(rng.choice([0, 0, 0, -12, 12])),
        ),
        osc2=OscSpec(
            source=_to_str_or_none(rng.choice(osc_pool)),
            level=float(rng.uniform(0.0, 0.5)),
            transpose=float(rng.choice([-12, -7, -5, 0, 7, 12])),
        ),
        osc3=OscSpec(
            source=_to_str_or_none(rng.choice([None, None] + osc_pool)),  # bias off
            level=float(rng.uniform(0.0, 0.4)),
            transpose=float(rng.choice([-24, -12, 0, 12, 24])),
        ),
        sample=OscSpec(),  # off by default
        lfos=[
            LFOSpec(
                source=_to_str_or_none(rng.choice(lfo_pool)),
                rate_hz=float(np.exp(rng.uniform(np.log(0.05), np.log(8.0)))),
                smooth=bool(rng.choice([True, True, False])),
                routings=[
                    (str(rng.choice(lfo_dests)),
                     float(rng.uniform(0.2, 1.0)),
                     bool(rng.choice([False, False, True]))),
                ] if rng.uniform() < 0.7 else [],
            )
            for _ in range(8)
        ],
        filter_1=FilterSpec(
            character=str(rng.choice(filter_chars)),
            cutoff=float(rng.uniform(40.0, 110.0)),
            resonance=float(rng.uniform(0.1, 0.85)),
        ),
        env_1=EnvelopeSpec(character=str(rng.choice(env_chars))),
        env_2=EnvelopeSpec(
            character=str(rng.choice(env_chars)) if rng.uniform() < 0.5 else None,
        ),
        effects=EffectSpec(
            reverb=bool(rng.choice([True, True, False])),
            reverb_dry_wet=float(rng.uniform(0.1, 0.5)),
            reverb_size=float(rng.uniform(0.3, 0.9)),
            delay=bool(rng.choice([False, False, True])),
            delay_sync=bool(rng.choice([True, False])),
            chorus=bool(rng.choice([False, False, True])),
            distortion=bool(rng.choice([False, False, True])),
            distortion_type=int(rng.choice([0, 1, 2, 9])),
            distortion_drive=float(rng.uniform(0.1, 0.6)),
            flanger=bool(rng.choice([False, False, False, True])),
            phaser=bool(rng.choice([False, False, False, True])),
        ),
        macros=[
            MacroSpec(
                name="biosignal richness",
                routings=[
                    ("filter_1_cutoff", float(rng.uniform(0.2, 0.6))),
                    ("osc_1_distortion_amount", float(rng.uniform(0.1, 0.4))),
                ],
            ),
        ],
    )
    return spec


def _to_str_or_none(v) -> str | None:
    if v is None:
        return None
    s = str(v)
    return None if s == "None" else s


# ===========================================================================
# build_combinatorial_sources — one-call setup
# ===========================================================================

_INHARMONIC_GENERATORS = {
    "stiff_string": ("inharmonic_string", {"B": 1e-4}),
    "stretched":    ("stretched_partials", {"stretch": 1.05}),
    "compressed":   ("compressed_partials", {"compress": 0.95}),
    "saron":        ("gamelan_partials",   {"instrument": "saron"}),
    "bonang":       ("gamelan_partials",   {"instrument": "bonang"}),
    "gender":       ("gamelan_partials",   {"instrument": "gender"}),
}


def build_combinatorial_sources(
    signal,
    sf: float,
    *,
    peaks_function: str = "EMD",
    matching_method: str = "consonance_weighted",
    n_peaks: int = 5,
    base_freq_hint: float = 220.0,
    include_inharmonic: bool = True,
    inharmonic_kind: str = "stiff_string",
    include_harmonograph: bool = True,
    include_polyrhythm: bool = True,
    include_lissajous: bool = False,
    force_imfs: bool = False,
) -> tuple[dict, Any]:
    """Run the biotuner pipeline once and bundle every derived source the
    combinatorial generator can consume into a ready-to-use ``sources`` dict.

    Auto-detects which sources are populated based on ``peaks_function``:

        * ``'EMD'`` / ``'EEMD'`` / ``'CEEMDAN'`` / ``'HilbertHuang'``  → IMFs come for free
        * ``'welch'`` / ``'fft'``                                       → no IMFs (set
          ``force_imfs=True`` to run a separate EMD pass; adds 1–5 s)

    The returned dict has a special ``__available__`` key — a set of source
    *kinds* that were actually populated. Pass it into
    :func:`spec_from_biotuner` or :func:`random_spec` to constrain the
    sampling to what's actually present.

    Parameters
    ----------
    signal : array-like
        1D biosignal samples.
    sf : float
        Source samplerate (Hz).
    peaks_function : str, default='EMD'
        Forwarded to ``compute_biotuner``. EMD-family methods populate IMFs
        as a side-effect; FFT/Welch don't.
    matching_method : str, default='consonance_weighted'
        Forwarded to ``timbre_from_biotuner``.
    n_peaks, base_freq_hint
        Standard Biotuner / Timbre parameters.
    include_inharmonic, include_harmonograph, include_polyrhythm,
    include_lissajous : bool
        Toggle whether to run each derivation. All cheap (<200 ms each)
        except on huge signals.
    inharmonic_kind : str
        ``'stiff_string'`` | ``'stretched'`` | ``'compressed'`` | ``'saron'``
        | ``'bonang'`` | ``'gender'``.
    force_imfs : bool, default=False
        If ``peaks_function`` doesn't itself produce IMFs, run a separate
        EMD pass to make them available. Costs an extra 1–5 s for typical
        EEG-length signals.

    Returns
    -------
    (sources, bt) : (dict, compute_biotuner)
        ``sources`` is the dict to pass to ``to_vital_combinatorial``.
        ``bt`` is the fitted Biotuner instance for further use.
    """
    from biotuner import compute_biotuner
    from biotuner.harmonic_timbre import (
        compressed_partials,
        gamelan_partials,
        inharmonic_string,
        inharmonic_timbre,
        stretched_partials,
        timbre_from_biotuner,
    )

    sig = np.asarray(signal, dtype=np.float64).flatten()
    sf = float(sf)

    # 1. Run biotuner. peaks_function is a peaks_extraction kwarg in the
    # current biotuner API; n_peaks is too. The constructor takes only
    # global config (sf, precision, harm settings).
    bt = compute_biotuner(sf=sf, peaks_function=peaks_function)
    bt.peaks_extraction(sig, n_peaks=n_peaks)

    # 2. Build the matched timbre
    timbre = timbre_from_biotuner(
        bt, matching_method=matching_method, base_freq=base_freq_hint,
    )

    sources: dict = {
        "signal": sig,
        "sf": sf,
        "timbre": timbre,
        "base_freq": float(timbre.base_freq) if timbre.base_freq > 0 else float(base_freq_hint),
    }
    available: set[str] = {"signal", "matched_timbre"}

    # 3. IMFs — auto-detect from peaks_function output, optionally force
    imfs = getattr(bt, "IMFs", None)
    if (imfs is None or (hasattr(imfs, "size") and imfs.size == 0)) and force_imfs:
        try:
            from biotuner.peaks_extraction import EMD_eeg
            imfs = EMD_eeg(sig, method="EMD", graph=False)
        except Exception:
            imfs = None
    if imfs is not None:
        try:
            imf_list = [np.asarray(x, dtype=np.float64).flatten() for x in imfs]
            imf_list = [x for x in imf_list if x.size >= 4]
            if imf_list:
                sources["imfs"] = imf_list
                available.add("imfs")
        except Exception:
            pass

    # 4. Inharmonic alternative timbre
    if include_inharmonic:
        gen_lookup = {
            "inharmonic_string":  (inharmonic_string,  _INHARMONIC_GENERATORS["stiff_string"][1]),
            "stretched_partials": (stretched_partials, _INHARMONIC_GENERATORS["stretched"][1]),
            "compressed_partials": (compressed_partials, _INHARMONIC_GENERATORS["compressed"][1]),
            "gamelan_partials":   (gamelan_partials,   None),  # filled per-kind below
        }
        if inharmonic_kind in _INHARMONIC_GENERATORS:
            fn_name, kw = _INHARMONIC_GENERATORS[inharmonic_kind]
            fn = gen_lookup[fn_name][0]
            try:
                sources["timbre_inharmonic"] = inharmonic_timbre(
                    fn, n=8, base_freq=sources["base_freq"], fn_kwargs=kw,
                )
                available.add("inharmonic")
            except Exception:
                pass

    # 5. Harmonograph
    if include_harmonograph:
        try:
            from biotuner.harmonic_geometry.harmonograph import (
                harmonograph_from_peaks,
            )
            peaks = list(getattr(bt, "peaks", []) or [])
            amps = list(getattr(bt, "amps", []) or [])
            phases = getattr(bt, "phases", None)
            if phases is not None:
                phases = list(phases)
            if peaks and amps:
                geom = harmonograph_from_peaks(
                    peaks=peaks, amps=amps, phases=phases,
                )
                coords = np.asarray(geom.coordinates, dtype=np.float64)
                if coords.ndim == 2 and coords.shape[1] >= 2:
                    sources["harmonograph"] = (coords[:, 0], coords[:, 1])
                    available.add("harmonograph")
        except Exception:
            pass

    # 6. Polyrhythm gate
    if include_polyrhythm:
        try:
            from biotuner.rhythm_construction import scale2polyrhythm
            ratios = (
                list(getattr(bt, "peaks_ratios_cons", []) or [])
                or list(getattr(bt, "peaks_ratios", []) or [])
            )
            if len(ratios) >= 2:
                result = scale2polyrhythm(ratios, max_denom=16)
                # result shape can vary; coincidence array is index 1 if tuple
                coincidences = result[1] if isinstance(result, tuple) and len(result) >= 2 else result
                gate = np.asarray(coincidences, dtype=np.float64).flatten()
                if gate.size > 0:
                    sources["polyrhythm_gate"] = gate
                    available.add("polyrhythm_gate")
        except Exception:
            pass

    # 7. Lissajous
    if include_lissajous:
        try:
            from biotuner.harmonic_geometry.lissajous import lissajous_2d
            peaks = list(getattr(bt, "peaks", []) or [])
            if len(peaks) >= 2:
                geom = lissajous_2d(peaks[:2])
                coords = np.asarray(geom.coordinates, dtype=np.float64)
                if coords.ndim == 2 and coords.shape[1] >= 2:
                    sources["lissajous"] = (coords[:, 0], coords[:, 1])
                    available.add("lissajous")
        except Exception:
            pass

    sources["__available__"] = available
    return sources, bt


# ===========================================================================
# spec_from_biotuner — biosignal-driven spec
# ===========================================================================

# Destination pools, ordered by "natural rate" — slow signals fit slow
# destinations, fast ones fit fast destinations.
_SLOW_DESTINATIONS = [
    "filter_1_cutoff", "filter_1_resonance",
    "reverb_dry_wet", "reverb_size",
    "osc_1_wavetable_position", "osc_1_pan",
    "volume", "delay_dry_wet",
]
_FAST_DESTINATIONS = [
    "osc_1_distortion_amount", "osc_2_distortion_amount",
    "osc_1_level", "osc_2_level", "osc_3_level",
    "osc_2_transpose", "filter_1_drive",
    "chorus_dry_wet",
]


def _safe_get(bt: Any, attr: str, default: Any = None) -> Any:
    """Read an attribute defensively. Returns ``default`` if missing,
    None, empty array, or empty list. Always returns a Python-truthy
    value safe to ``len()`` on (numpy arrays not used directly in ``or``)."""
    v = getattr(bt, attr, None)
    if v is None:
        return default
    try:
        if hasattr(v, "size") and v.size == 0:
            return default
    except Exception:
        pass
    try:
        if hasattr(v, "__len__") and len(v) == 0:
            return default
    except Exception:
        pass
    return v


def _safe_len(v) -> int:
    """Length of an attribute, safe for ``None`` / empty-array."""
    if v is None:
        return 0
    try:
        return int(len(v))
    except (TypeError, ValueError):
        return 0


def _safe_list(v) -> list:
    """Coerce an attribute to a list (defensive against ``None`` / numpy arrays)."""
    if v is None:
        return []
    try:
        return list(v)
    except TypeError:
        return []


def _compute_higuchi_fd(signal) -> float:
    try:
        from biotuner.metrics import higuchi_fd
        v = float(higuchi_fd(np.asarray(signal, dtype=np.float64).flatten()))
        if not np.isfinite(v):
            return 1.5
        return float(np.clip(v, 1.0, 2.0))
    except Exception:
        return 1.5


def _compute_consonance_score(ratios) -> float:
    if not ratios:
        return 0.5
    try:
        from biotuner.metrics import dyad_similarity
        vals = [float(dyad_similarity(float(r))) for r in ratios if float(r) > 0]
        if not vals:
            return 0.5
        # dyad_similarity returns a 0..100 percentage; normalize.
        return float(np.clip(np.mean(vals) / 100.0, 0.0, 1.0))
    except Exception:
        return 0.5


def spec_from_biotuner(
    bt: Any,
    *,
    signal=None,
    sf: float | None = None,
    available_sources: set[str] | None = None,
    randomness: float = 0.0,
    seed: int = 0,
    name: str = "biotuner_spec",
) -> PresetSpec:
    """Derive a PresetSpec from biotuner state, biased by biosignal properties.

    The CONTENT of every populated slot is biosignal-derived (always —
    that's just how the source IDs resolve in :func:`to_vital_combinatorial`).
    What this function adds is **biosignal-driven choice of routing topology**:
    which oscillators are enabled, which LFO sources go where, what filter
    character, what envelope, which effects on.

    Mappings used:

        higuchi_fd (fractal dimension)    → number of active LFOs
                                            (more chaotic biosignal → more
                                             modulation routings)
        spectral_tilt (FOOOF aperiodic)   → envelope character
                                            (steep 1/f → bell/pluck;
                                             flat → pad/drone)
        spectral_flatness / noise_floor   → distortion drive amount
        # of strong PAC pairs              → which LFOs use sine vs IMF curves
        # of detected peaks                → osc 2 / osc 3 enabled or not
        # of IMFs available                → osc 3 routing depth
        consonance metric (dyad sim mean)  → filter resonance + reverb dry/wet
                                            (more consonant → cleaner tone)

    Parameters
    ----------
    bt : compute_biotuner
        Fitted biotuner instance.
    signal : array-like, optional
        The raw biosignal (used for higuchi_fd if not present on bt).
    sf : float, optional
        Sample rate of signal.
    available_sources : set[str], optional
        Which source kinds are present. If None, derived from ``bt`` state.
    randomness : float, default=0.0
        ``0.0`` → fully deterministic from biotuner state.
        ``1.0`` → equivalent to :func:`random_spec` (no biosignal-driven
        routing; pure random sampling).
        In between → random sampling biased by biosignal properties; the
        exploration stays within the biosignal's "neighbourhood."
    seed : int
        Seed for the rng used when ``randomness > 0``.
    name : str

    Returns
    -------
    PresetSpec
    """
    rng = np.random.default_rng(seed)
    randomness = float(np.clip(randomness, 0.0, 1.0))

    # ------- Property extraction (all defensive against missing attrs) -------
    avail = available_sources or _infer_available_from_bt(bt, signal=signal)
    n_peaks = _safe_len(_safe_get(bt, "peaks"))
    n_imfs = _safe_len(_safe_get(bt, "IMFs"))

    pac_freqs = _safe_list(_safe_get(bt, "pac_freqs"))
    pac_coupling = _safe_list(_safe_get(bt, "pac_coupling"))
    n_pac = len(pac_freqs)
    strongest_pac_freq = 1.0
    if n_pac > 0:
        try:
            i_max = int(np.argmax(np.asarray(pac_coupling, dtype=np.float64)))
            strongest_pac_freq = float(pac_freqs[i_max][0])
        except Exception:
            pass

    hfd = _compute_higuchi_fd(signal) if signal is not None else 1.5
    chaos = float(np.clip(hfd - 1.0, 0.0, 1.0))   # 0..1, scaled

    tilt_raw = _safe_get(bt, "aperiodic_exponent", 1.0)
    tilt = float(tilt_raw) if tilt_raw is not None else 1.0
    noise_raw = _safe_get(bt, "spectral_flatness", 0.1)
    noise_floor = float(noise_raw) if noise_raw is not None else 0.1
    ratios = _safe_list(_safe_get(bt, "peaks_ratios_cons"))
    if not ratios:
        ratios = _safe_list(_safe_get(bt, "peaks_ratios"))
    cons_norm = _compute_consonance_score(ratios)

    # Mix-in helper: with randomness=0, return deterministic; with r=1, return random.
    def _maybe_random_choice(deterministic, options):
        if randomness < 1e-6:
            return deterministic
        if rng.uniform() < randomness:
            return options[int(rng.integers(0, len(options)))]
        return deterministic

    def _maybe_jitter(value, *, span):
        if randomness < 1e-6:
            return value
        return value + float(rng.uniform(-span, span)) * randomness

    # ------- Envelope character (driven by spectral tilt) -------
    if tilt >= 1.7:
        env_char_det = "pluck"
    elif tilt >= 1.3:
        env_char_det = "bell" if cons_norm > 0.5 else "stab"
    elif tilt >= 0.9:
        env_char_det = "pad"
    elif tilt >= 0.5:
        env_char_det = "swell"
    else:
        env_char_det = "drone"
    env_char = _maybe_random_choice(
        env_char_det,
        ["pad", "bell", "pluck", "percussive", "drone", "organ", "swell", "stab"],
    )

    # ------- Filter (driven by chaos + consonance) -------
    # chaos < 0.3 → off / lp; mid → lp / hp; high → comb / ladder / bp
    if chaos < 0.3:
        filter_char_det = "off" if cons_norm > 0.6 else "lp"
    elif chaos < 0.6:
        filter_char_det = "lp"
    else:
        filter_char_det = "comb" if cons_norm < 0.3 else "ladder"
    filter_char = _maybe_random_choice(
        filter_char_det,
        ["off", "lp", "hp", "bp", "notch", "comb", "ladder", "diode", "formant"],
    )
    filter_cutoff = _maybe_jitter(70.0 + 30.0 * (1.0 - cons_norm), span=15.0)
    filter_resonance = _maybe_jitter(0.7 - 0.5 * cons_norm, span=0.2)
    filter_resonance = float(np.clip(filter_resonance, 0.0, 0.95))

    # ------- Number of active LFOs (driven by chaos) -------
    target_n_lfos_det = int(round(2 + 6 * chaos))
    if randomness < 1e-6:
        target_n_lfos = target_n_lfos_det
    else:
        # Gaussian jitter scaled by randomness, rounded — gives proper variance
        # at moderate randomness values (the previous int(-2..3 * 0.3) collapse
        # always rounded to 0, leaving target_n_lfos invariant).
        jitter = float(rng.normal(0.0, 2.0 * randomness))
        target_n_lfos = int(np.clip(round(target_n_lfos_det + jitter), 1, 8))

    # ------- Build the LFO source priority queue (biosignal-derived) -------
    candidates: list[tuple[str, float, str]] = []   # (source_id, rate_hz, default_dest)

    # 1. PAC pairs at their natural rates → sine LFOs
    if n_pac > 0:
        for i in range(min(n_pac, 4)):
            try:
                low_f = float(pac_freqs[i][0])
                candidates.append(("sine", low_f,
                                   "osc_1_level" if i == 0 else "filter_1_cutoff"))
            except Exception:
                pass
    # 2. IMF curves (if available) at progressive rates
    if "imfs" in avail:
        for i in range(min(n_imfs, 5)):
            rate = 0.5 * (i + 1)   # 0.5, 1.0, 1.5, 2.0, 2.5 Hz
            candidates.append((f"imf_{i}", rate, ""))   # default-dest filled by rate-aware logic
    # 3. Hilbert envelope at low rate
    if "signal" in avail:
        candidates.append(("hilbert_envelope", 0.25, "filter_1_cutoff"))
    # 4. Polyrhythm gate at fast rate
    if "polyrhythm_gate" in avail:
        candidates.append(("polyrhythm_gate", 4.0, "osc_1_level"))
    # 5. Harmonograph axes
    if "harmonograph" in avail:
        candidates.append(("harmonograph_x", 0.5, "osc_1_pan"))
        candidates.append(("harmonograph_y", 0.7, "filter_1_cutoff"))
    # Fallback if nothing else
    if not candidates:
        candidates.append(("sine", 1.0, "filter_1_cutoff"))

    # Trim/shuffle to target_n_lfos
    if randomness > 0:
        rng.shuffle(candidates)
    candidates = candidates[: max(target_n_lfos, 1)]

    # Build LFO specs, choosing destinations based on the source's natural rate
    lfo_specs = [LFOSpec() for _ in range(8)]
    for slot, (src, rate_hz, dest_hint) in enumerate(candidates):
        if dest_hint:
            dest = dest_hint
        else:
            pool = _SLOW_DESTINATIONS if rate_hz < 1.0 else _FAST_DESTINATIONS
            if randomness < 1e-6:
                dest = pool[slot % len(pool)]
            else:
                dest = str(rng.choice(pool))
        amount = _maybe_jitter(0.6, span=0.3)
        amount = float(np.clip(amount, 0.1, 1.0))
        smooth = src not in ("polyrhythm_gate", "square")
        lfo_specs[slot] = LFOSpec(
            source=src,
            rate_hz=float(rate_hz),
            smooth=smooth,
            n_points=24 if smooth else 16,
            routings=[(dest, amount, False)],
        )

    # ------- Oscillators (driven by # peaks + IMFs) -------
    osc1 = OscSpec(
        source="matched_timbre",
        level=0.7,
        unison_voices=3 if chaos > 0.5 else 1,
        unison_detune=4.0 + 4.0 * chaos,
    )
    if n_peaks >= 4:
        osc2_source_det = "inharmonic" if "inharmonic" in avail else "matched_timbre"
        osc2 = OscSpec(
            source=_maybe_random_choice(osc2_source_det,
                                        [s for s in ("inharmonic", "matched_timbre",
                                                     "sine", "harmonograph_x")
                                         if s == "matched_timbre" or s.replace("_x", "")
                                         in (avail | {"sine"})]),
            level=0.3,
            transpose=12.0 if randomness < 1e-6 else float(rng.choice([12, -12, 7, -7])),
        )
    else:
        osc2 = OscSpec()

    if n_imfs >= 3 and "imfs" in avail:
        osc3 = OscSpec(
            source="imf_2",
            level=0.2,
            transpose=-12.0,
        )
    else:
        osc3 = OscSpec()

    # ------- Effects (driven by noise_floor, consonance, chaos) -------
    effects = EffectSpec(
        reverb=True,
        reverb_dry_wet=float(np.clip(0.2 + 0.4 * (1.0 - cons_norm), 0.1, 0.7)),
        reverb_size=float(np.clip(0.4 + 0.4 * chaos, 0.2, 0.95)),
        chorus=chaos > 0.6,
        chorus_dry_wet=0.3,
        distortion=noise_floor > 0.15,
        distortion_drive=float(np.clip(noise_floor * 2.0, 0.1, 0.7)),
        distortion_type=9 if "imfs" in avail else 0,
        delay=False,
    )
    if randomness > 0.5:
        effects.delay = bool(rng.choice([False, True]))
        effects.flanger = bool(rng.choice([False, False, True]))

    # ------- Macro that summarises "biosignal richness" -------
    macros = [
        MacroSpec(
            name=f"biosignal richness (hfd={hfd:.2f})",
            routings=[
                ("filter_1_cutoff", 0.4),
                ("osc_1_distortion_amount", 0.3),
                ("reverb_dry_wet", 0.3),
            ],
        ),
    ]

    return PresetSpec(
        name=name,
        osc1=osc1, osc2=osc2, osc3=osc3,
        lfos=lfo_specs,
        filter_1=FilterSpec(
            character=filter_char,
            cutoff=float(np.clip(filter_cutoff, 30.0, 120.0)),
            resonance=filter_resonance,
        ),
        env_1=EnvelopeSpec(character=env_char),
        effects=effects,
        macros=macros,
    )


def _infer_available_from_bt(bt: Any, *, signal=None) -> set[str]:
    """Determine which source kinds are populated from a fitted biotuner instance."""
    avail = {"matched_timbre"}
    if signal is not None:
        avail.add("signal")
    if _safe_len(_safe_get(bt, "IMFs")) > 0:
        avail.add("imfs")
    if _safe_len(_safe_get(bt, "peaks_ratios_cons")) > 0 or _safe_len(_safe_get(bt, "peaks_ratios")) > 0:
        avail.add("polyrhythm_gate")
    if _safe_len(_safe_get(bt, "peaks")) > 0 and _safe_len(_safe_get(bt, "amps")) > 0:
        avail.add("harmonograph")
    return avail
