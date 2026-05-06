"""Vital ``.vital`` preset exporters — modular, primitive-composed.

Module type: Functions

All generators below are composed from the small primitive set in
:mod:`_vital_template`. Each generator embodies a different *modular
principle* — biosignal as character / modulation curve / topology / found
sound / 2D motion / rhythm / time-evolution.

Catalog (by principle)
----------------------

biosignal as static character:
    to_vital_spectral(timbre)                    matched-timbre as one wavetable cycle
    to_vital_inharmonic(partial_series_fn)       gamelan/stiff-string/stretched as wavetable

biosignal as evolving character:
    to_vital_wavetable_morph(timbre, evolution)  64-frame brightness sweep / harmonic-buildup / amp-morph / phase-sweep

biosignal as modulation curve:
    to_vital_hilbert_modulator(signal, sf)       Hilbert envelope of the biosignal → custom LFO shape
    to_vital_imf_lfo_bank(imfs)                  N IMFs → N independently routable LFO shapes
    to_vital_pac_driven(timbre)                  PAC pairs → sine LFOs → osc level (AM)
    to_vital_cfc_driven(timbre)                  CFC pairs → sine LFOs → distortion amount (FM-flavored)

biosignal as synthesis topology:
    to_vital_imf_stack(imfs, base_freq)          IMFs as 3 oscillators with cross-FM derived from Hilbert-Huang hierarchy
    to_vital_fm(timbre)                          osc 2 → osc 1 FM at biotuner-derived ratio

biosignal as found sound:
    to_vital_hilbert_sample(signal, sf)          biosignal rendered to audio + (intent: load into Vital's sample osc)

biosignal as rhythm:
    to_vital_polyrhythm_gated(timbre, gate)      polyrhythm coincidence pattern → stepped LFO → amp gate

biosignal as time-evolution:
    to_vital_markov_macro(seq)                   TimbreSequence frames as multi-frame wavetable + Macro 1 for hand-walk

biosignal as compositional ensemble:
    to_vital_ensemble(timbre, *, signal/imfs/...)   coordinated set of ≤4 presets sharing tuning, varying modulation

Every generator
---------------
* writes ``.vital`` (preset surgically patched on top of factory template)
* writes ``_settings.json`` companion (stable, version-agnostic)
* embeds biotuner provenance in ``preset.comments``
* names LFOs and modulations descriptively for legibility in Vital's UI
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import numpy as np

from biotuner.harmonic_timbre.exporters._vital_template import (
    WAVETABLE_FRAME_SIZE,
    apply_minimal_baseline,
    curve_to_lfo_points,
    find_free_modulation_slot,
    make_blank_preset,
    modulator_to_routing,
    populate_osc,
    replace_wavetable,
    route_lfo,
    set_effect,
    set_envelope,
    set_filter_1,
    set_lfo_from_curve,
    set_lfo_sine,
    set_modulation,
    set_sample_osc,
    set_unison,
    write_settings_companion,
    write_vital_preset,
)
from biotuner.harmonic_timbre.synthesis import render_wavetable_cycle
from biotuner.harmonic_timbre.timbre import Timbre, TimbreSequence


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _summarize_timbre(timbre: Timbre) -> dict:
    return {
        "n_partials": timbre.n_partials(),
        "matched_tuning": list(timbre.matched_tuning) if timbre.matched_tuning is not None else None,
        "matching_method": timbre.matching_method,
        "base_freq": float(timbre.base_freq),
        "metadata": dict(timbre.metadata),
        "n_am_modulators": len(timbre.am_modulators),
        "n_fm_modulators": len(timbre.fm_modulators),
    }


def _provenance_comment(timbre: Timbre, *, preset_kind: str, extra: str = "") -> str:
    """Build a multi-line comment string for Vital's `comments` field."""
    parts = [f"biotuner.harmonic_timbre — preset_kind = {preset_kind}"]
    if timbre.matched_tuning is not None:
        rs = ", ".join(f"{r:.4f}" for r in timbre.matched_tuning[:8])
        parts.append(f"matched_tuning = [{rs}{'…' if len(timbre.matched_tuning) > 8 else ''}]")
    if timbre.matching_method:
        parts.append(f"matching_method = {timbre.matching_method}")
    if timbre.metadata:
        used = timbre.metadata.get("biotuner_fields_used")
        if used:
            parts.append(f"biotuner_fields_used = {used}")
        ss = timbre.metadata.get("scale_source")
        if ss:
            parts.append(f"scale_source = {ss}")
    if extra:
        parts.append(extra)
    return "\n".join(parts)


def _ratio_to_coarse_fine(ratio_to_carrier: float) -> tuple[float, float]:
    """ratio → (coarse_semis, fine_cents).  e.g. 5/4 → (4, -14)."""
    semis = float(np.log2(max(ratio_to_carrier, 1e-9)) * 12.0)
    coarse = float(np.round(semis))
    fine_cents = float(np.round((semis - coarse) * 100.0))
    return coarse, fine_cents


# ---------------------------------------------------------------------------
# 1. Static character — to_vital_spectral
# ---------------------------------------------------------------------------

def to_vital_spectral(
    timbre: Timbre,
    out_path: str,
    *,
    preset_name: str | None = None,
) -> dict:
    """Single-keyframe wavetable preset = the timbre's matched spectrum frozen as one cycle."""
    timbre.validate()
    cycle = render_wavetable_cycle(timbre, table_size=WAVETABLE_FRAME_SIZE)
    name = preset_name or os.path.splitext(os.path.basename(out_path))[0]

    preset = make_blank_preset(
        name=name,
        comments=_provenance_comment(timbre, preset_kind="spectral",
                                     extra="osc 1 = matched-timbre wavetable. PAD envelope, "
                                           "no filter, unison voices for width."),
    )
    apply_minimal_baseline(preset)
    populate_osc(preset, 0,
                 wavetable_audio=cycle, wavetable_name="matched_timbre",
                 level=0.7, on=True)
    set_unison(preset, 0, voices=3, detune=6.0)
    set_envelope(preset, 1, attack=0.35, decay=0.5, sustain=1.0, release=0.55)
    set_effect(preset, "reverb", on=True, dry_wet=0.3, size=0.65)

    out_v = write_vital_preset(preset, out_path)
    out_c = write_settings_companion(
        "spectral",
        out_path=out_v.replace(".vital", "_settings.json"),
        base_freq=float(timbre.base_freq),
        timbre_summary=_summarize_timbre(timbre),
        extras={"oscillator": {"index": 0, "n_frames": 1}},
    )
    return {"vital": out_v, "companion": out_c}


# ---------------------------------------------------------------------------
# 2. Inharmonic character — to_vital_inharmonic
# ---------------------------------------------------------------------------

def to_vital_inharmonic(
    timbre: Timbre,
    out_path: str,
    *,
    preset_name: str | None = None,
    inharmonic_label: str = "",
) -> dict:
    """Same as spectral, but the Timbre's partials come from an inharmonic series
    (gamelan/stiff-string/stretched/custom). Distinguished only by provenance.

    Build the timbre with :func:`biotuner.harmonic_timbre.inharmonic_timbre` and
    pass it here. The wavetable carries the inharmonic spectrum's character —
    bell-like, metallic, glassy depending on the source series.
    """
    timbre.validate()
    cycle = render_wavetable_cycle(timbre, table_size=WAVETABLE_FRAME_SIZE)
    name = preset_name or os.path.splitext(os.path.basename(out_path))[0]
    label = inharmonic_label or timbre.metadata.get("partial_series_fn", "inharmonic")

    preset = make_blank_preset(
        name=name,
        comments=_provenance_comment(
            timbre, preset_kind="inharmonic",
            extra=f"inharmonic series: {label}. BELL envelope, light HP filter, "
                  f"long reverb — bells need space.",
        ),
    )
    apply_minimal_baseline(preset)
    populate_osc(preset, 0,
                 wavetable_audio=cycle, wavetable_name=str(label)[:24],
                 level=0.7, on=True)
    # Bell envelope: instant attack, long decay, no sustain, short release
    set_envelope(preset, 1, attack=0.0, decay=0.75, sustain=0.0, release=0.3)
    # Light high-pass to bring out the bright inharmonic partials
    set_filter_1(preset, on=True, model=1, cutoff=40.0, resonance=0.2)
    set_effect(preset, "reverb", on=True, dry_wet=0.5, size=0.85)

    out_v = write_vital_preset(preset, out_path)
    out_c = write_settings_companion(
        "inharmonic",
        out_path=out_v.replace(".vital", "_settings.json"),
        base_freq=float(timbre.base_freq),
        timbre_summary=_summarize_timbre(timbre),
        extras={"inharmonic_label": str(label)},
    )
    return {"vital": out_v, "companion": out_c}


# ---------------------------------------------------------------------------
# 3. Evolving character — to_vital_wavetable_morph
# ---------------------------------------------------------------------------

def to_vital_wavetable_morph(
    timbre: Timbre,
    out_path: str,
    *,
    n_frames: int = 64,
    evolution: str = "tilt",
    tilt_range: tuple[float, float] = (0.0, 2.5),
    preset_name: str | None = None,
    map_lfo_to_position: bool = True,
    lfo_rate_hz: float = 0.25,
) -> dict:
    """Multi-keyframe wavetable + optional LFO 1 → osc 1 wavetable position."""
    timbre.validate()
    if n_frames < 1:
        raise ValueError("n_frames must be ≥ 1")

    name = preset_name or os.path.splitext(os.path.basename(out_path))[0]
    audio = _build_wavetable_audio(timbre, n_frames, evolution, tilt_range)

    preset = make_blank_preset(
        name=name,
        comments=_provenance_comment(
            timbre, preset_kind="wavetable_morph",
            extra=f"n_frames = {n_frames}, evolution = {evolution!r}",
        ),
    )
    populate_osc(preset, 0,
                 wavetable_audio=audio, wavetable_name=f"morph_{evolution}",
                 level=0.7, on=True)

    if map_lfo_to_position and n_frames > 1:
        sine_curve = np.sin(2.0 * np.pi * np.linspace(0, 1, 32, endpoint=False))
        set_lfo_from_curve(
            preset, lfo_index=0, curve_1d=sine_curve,
            rate_hz=lfo_rate_hz, name=f"position_LFO ({lfo_rate_hz:.2f} Hz)",
        )
        route_lfo(preset, lfo_index=0,
                  destination="osc_1_wavetable_position", amount=1.0)

    out_v = write_vital_preset(preset, out_path)
    out_c = write_settings_companion(
        "wavetable_morph",
        out_path=out_v.replace(".vital", "_settings.json"),
        base_freq=float(timbre.base_freq),
        timbre_summary=_summarize_timbre(timbre),
        extras={
            "oscillator": {
                "index": 0, "n_frames": int(n_frames),
                "evolution": evolution,
                "tilt_range": list(tilt_range) if evolution == "tilt" else None,
            },
            "lfo_routing": (
                {"lfo_index": 0, "rate_hz": float(lfo_rate_hz),
                 "destination": "osc_1_wavetable_position"}
                if map_lfo_to_position and n_frames > 1 else None
            ),
        },
    )
    return {"vital": out_v, "companion": out_c}


def _build_wavetable_audio(
    timbre: Timbre,
    n_frames: int,
    evolution: str,
    tilt_range: tuple[float, float],
) -> np.ndarray:
    if n_frames == 1:
        return render_wavetable_cycle(timbre, table_size=WAVETABLE_FRAME_SIZE)
    if evolution == "tilt":
        tilts = np.linspace(tilt_range[0], tilt_range[1], n_frames)
        frames = [
            render_wavetable_cycle(
                timbre.with_partials(spectral_tilt=float(t)),
                table_size=WAVETABLE_FRAME_SIZE,
            ) for t in tilts
        ]
    elif evolution == "harmonic_buildup":
        n = timbre.n_partials()
        active = np.linspace(1, n, n_frames).astype(int)
        frames = []
        for k in active:
            mask = np.zeros(n, dtype=np.float64); mask[: int(k)] = 1.0
            frames.append(render_wavetable_cycle(
                timbre.with_partials(amplitudes=timbre.amplitudes * mask),
                table_size=WAVETABLE_FRAME_SIZE,
            ))
    elif evolution == "amp_morph":
        rng = np.random.default_rng(0)
        random_amps = rng.uniform(0.0, 1.0, timbre.n_partials())
        alphas = np.linspace(0.0, 1.0, n_frames)
        frames = []
        for a in alphas:
            morphed = (1.0 - a) * random_amps + a * timbre.amplitudes
            morphed = morphed / max(np.max(np.abs(morphed)), 1e-9)
            frames.append(render_wavetable_cycle(
                timbre.with_partials(amplitudes=morphed),
                table_size=WAVETABLE_FRAME_SIZE,
            ))
    elif evolution == "phase_sweep":
        offsets = np.linspace(0.0, 2.0 * np.pi, n_frames)
        n = timbre.n_partials()
        frames = []
        for p in offsets:
            phases = (np.arange(1, n + 1) * p) % (2.0 * np.pi)
            frames.append(render_wavetable_cycle(
                timbre.with_partials(phases=phases),
                table_size=WAVETABLE_FRAME_SIZE,
            ))
    else:
        raise ValueError(f"unknown evolution {evolution!r}")
    return np.concatenate(frames).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# 4. FM topology — to_vital_fm
# ---------------------------------------------------------------------------

def to_vital_fm(
    timbre: Timbre,
    out_path: str,
    *,
    preset_name: str | None = None,
    fm_amount: float = 0.5,
) -> dict:
    """FM preset: osc 2 → osc 1, mod ratio from biotuner-derived ratio.

    Vital's FM is implemented as ``osc_2_destination = osc 1`` plus
    ``osc_1_distortion_type = FM`` (enum 9) with a non-zero distortion
    amount. The osc 2 transpose+tune encodes the FM modulator/carrier ratio.
    """
    timbre.validate()
    if not timbre.fm_modulators:
        raise ValueError("to_vital_fm: timbre has no fm_modulators")

    fm = next((m for m in timbre.fm_modulators if m.carrier_idx == 0), timbre.fm_modulators[0])
    name = preset_name or os.path.splitext(os.path.basename(out_path))[0]

    base = float(timbre.base_freq) if timbre.base_freq > 0 else float(min(timbre.partials_hz))
    carrier_hz = float(timbre.partials_hz[fm.carrier_idx])
    mod_hz = float(fm.mod_freq)
    ratio_to_carrier = mod_hz / max(carrier_hz, 1e-9)
    coarse, fine_cents = _ratio_to_coarse_fine(ratio_to_carrier)

    preset = make_blank_preset(
        name=name,
        comments=_provenance_comment(
            timbre, preset_kind="fm",
            extra=f"osc 2 → osc 1 FM, ratio = {ratio_to_carrier:.4f} "
                  f"(coarse {coarse:+.0f} semis, fine {fine_cents:+.0f}c), "
                  f"depth = {fm.depth:.2f} Hz, β = {fm.depth / max(mod_hz, 1e-9):.2f}",
        ),
    )
    sine_cycle = np.sin(2.0 * np.pi * np.arange(WAVETABLE_FRAME_SIZE) / WAVETABLE_FRAME_SIZE)
    populate_osc(preset, 0,
                 wavetable_audio=sine_cycle, wavetable_name="carrier",
                 level=0.7, on=True,
                 distortion_type=9.0, distortion_amount=float(np.clip(fm_amount, 0.0, 1.0)))
    populate_osc(preset, 1,
                 wavetable_audio=sine_cycle, wavetable_name="modulator",
                 level=0.0, on=True,
                 transpose=coarse, fine_cents=fine_cents,
                 destination=1.0)  # → osc 1

    out_v = write_vital_preset(preset, out_path)
    out_c = write_settings_companion(
        "fm",
        out_path=out_v.replace(".vital", "_settings.json"),
        base_freq=float(timbre.base_freq),
        timbre_summary=_summarize_timbre(timbre),
        extras={
            "fm_routing": {
                "carrier_osc": 1, "modulator_osc": 2,
                "carrier_freq_hz": carrier_hz, "modulator_freq_hz": mod_hz,
                "modulator_ratio_to_carrier": ratio_to_carrier,
                "depth_hz": float(fm.depth),
                "fm_index_beta": float(fm.depth / max(mod_hz, 1e-9)),
                "vital_settings": {
                    "osc_2_transpose": coarse, "osc_2_tune_semis": fine_cents / 100.0,
                    "osc_2_destination": 1.0, "osc_1_distortion_type": 9.0,
                    "osc_1_distortion_amount": float(fm_amount),
                },
                "source_tag": fm.source,
            },
        },
    )
    return {"vital": out_v, "companion": out_c}


# ---------------------------------------------------------------------------
# 5. PAC AM — to_vital_pac_driven
# ---------------------------------------------------------------------------

def to_vital_pac_driven(
    timbre: Timbre,
    out_path: str,
    *,
    preset_name: str | None = None,
    base_amplitude: float = 0.7,
) -> dict:
    """PAC-driven preset: each AM modulator → one Vital LFO → osc 1 level."""
    timbre.validate()
    if not timbre.am_modulators:
        raise ValueError("to_vital_pac_driven: timbre has no am_modulators")

    name = preset_name or os.path.splitext(os.path.basename(out_path))[0]
    cycle = render_wavetable_cycle(timbre, table_size=WAVETABLE_FRAME_SIZE)

    pac_summary = ", ".join(
        f"{m.mod_freq:.1f}Hz×{m.depth:.2f}" for m in timbre.am_modulators[:4]
    )
    preset = make_blank_preset(
        name=name,
        comments=_provenance_comment(
            timbre, preset_kind="pac_driven",
            extra=f"PAC pairs ({len(timbre.am_modulators)}): {pac_summary}"
                  f"{'…' if len(timbre.am_modulators) > 4 else ''}",
        ),
    )
    populate_osc(preset, 0,
                 wavetable_audio=cycle, wavetable_name="matched_timbre",
                 level=base_amplitude, on=True)

    n_lfos = min(len(timbre.am_modulators), 8)
    routing_records: list[dict] = []
    for i in range(n_lfos):
        mod = timbre.am_modulators[i]
        routing = modulator_to_routing(mod, lfo_index=i)
        set_lfo_sine(
            preset, lfo_index=i, rate_hz=routing["rate_hz"],
            name=routing["name"],
        )
        route_lfo(preset, lfo_index=i,
                  destination=routing["destination"], amount=routing["amount"])
        routing_records.append(routing)

    out_v = write_vital_preset(preset, out_path)
    out_c = write_settings_companion(
        "pac_driven",
        out_path=out_v.replace(".vital", "_settings.json"),
        base_freq=float(timbre.base_freq),
        timbre_summary=_summarize_timbre(timbre),
        extras={
            "n_am_modulators_consumed": n_lfos,
            "n_am_modulators_skipped": max(0, len(timbre.am_modulators) - 8),
            "lfo_routing": routing_records,
        },
    )
    return {"vital": out_v, "companion": out_c}


# ---------------------------------------------------------------------------
# 6. CFC FM — to_vital_cfc_driven
# ---------------------------------------------------------------------------

def to_vital_cfc_driven(
    timbre: Timbre,
    out_path: str,
    *,
    preset_name: str | None = None,
    distortion_type: float = 9.0,    # FM From Sample
) -> dict:
    """CFC-driven preset: each FM modulator → one Vital LFO → distortion amount.

    Vital can't do per-LFO audio-rate FM, but it can route LFOs to the
    distortion amount knob. The result is a slow tremolo-of-FM-intensity —
    the FM character itself breathes at the CFC rate.
    """
    timbre.validate()
    if not timbre.fm_modulators:
        raise ValueError("to_vital_cfc_driven: timbre has no fm_modulators")

    name = preset_name or os.path.splitext(os.path.basename(out_path))[0]
    cycle = render_wavetable_cycle(timbre, table_size=WAVETABLE_FRAME_SIZE)

    cfc_summary = ", ".join(
        f"{m.mod_freq:.1f}Hz×{m.depth:.2f}" for m in timbre.fm_modulators[:4]
    )
    preset = make_blank_preset(
        name=name,
        comments=_provenance_comment(
            timbre, preset_kind="cfc_driven",
            extra=f"CFC pairs ({len(timbre.fm_modulators)}): {cfc_summary}"
                  f"{'…' if len(timbre.fm_modulators) > 4 else ''}; "
                  f"each LFO modulates osc 1 distortion amount.",
        ),
    )
    populate_osc(preset, 0,
                 wavetable_audio=cycle, wavetable_name="matched_timbre",
                 level=0.7, on=True,
                 distortion_type=distortion_type, distortion_amount=0.4)

    n_lfos = min(len(timbre.fm_modulators), 8)
    routing_records: list[dict] = []
    for i in range(n_lfos):
        mod = timbre.fm_modulators[i]
        routing = modulator_to_routing(
            mod, lfo_index=i,
            destination_for_fm="osc_1_distortion_amount",
        )
        set_lfo_sine(
            preset, lfo_index=i, rate_hz=routing["rate_hz"],
            name=routing["name"],
        )
        route_lfo(preset, lfo_index=i,
                  destination=routing["destination"], amount=routing["amount"])
        routing_records.append(routing)

    out_v = write_vital_preset(preset, out_path)
    out_c = write_settings_companion(
        "cfc_driven",
        out_path=out_v.replace(".vital", "_settings.json"),
        base_freq=float(timbre.base_freq),
        timbre_summary=_summarize_timbre(timbre),
        extras={
            "n_fm_modulators_consumed": n_lfos,
            "n_fm_modulators_skipped": max(0, len(timbre.fm_modulators) - 8),
            "lfo_routing": routing_records,
            "distortion_type": float(distortion_type),
        },
    )
    return {"vital": out_v, "companion": out_c}


# ---------------------------------------------------------------------------
# 7. Hilbert envelope as LFO — the unlock
# ---------------------------------------------------------------------------

def to_vital_hilbert_modulator(
    signal,
    sf: float,
    timbre: Timbre,
    out_path: str,
    *,
    preset_name: str | None = None,
    destination: str = "osc_1_filter_blend",
    lfo_loop_seconds: float = 4.0,
    n_lfo_points: int = 32,
) -> dict:
    """Hilbert envelope of the biosignal → custom-shape LFO routed anywhere.

    This is the single most flexible biosignal-→-Vital mapping. We extract
    the analytic-signal envelope of the input signal, resample it to a
    Vital LFO's points/powers format, and route LFO 1 to ``destination``.
    Default destination is ``osc_1_filter_blend`` — the biosignal's
    envelope opens and closes the timbre over time.

    Try other destinations:
        ``osc_1_level``                     — biosignal-driven AM
        ``osc_1_wavetable_position``        — biosignal walks through a wavetable morph
        ``filter_1_cutoff``                 — biosignal opens the filter
        ``osc_1_distortion_amount``         — biosignal-driven distortion
        ``volume``                          — biosignal-driven master volume
    """
    from scipy.signal import hilbert
    sig = np.asarray(signal, dtype=np.float64).flatten()
    if sig.size < 4:
        raise ValueError("to_vital_hilbert_modulator: signal too short")

    timbre.validate()
    name = preset_name or os.path.splitext(os.path.basename(out_path))[0]

    envelope = np.abs(hilbert(sig))
    duration_s = sig.size / float(sf)
    rate_hz = 1.0 / max(lfo_loop_seconds, 0.01)
    cycle = render_wavetable_cycle(timbre, table_size=WAVETABLE_FRAME_SIZE)

    preset = make_blank_preset(
        name=name,
        comments=_provenance_comment(
            timbre, preset_kind="hilbert_modulator",
            extra=f"signal duration = {duration_s:.2f} s @ {sf:.0f} Hz; "
                  f"Hilbert envelope embedded as LFO 1 ({n_lfo_points} pts), "
                  f"loop = {lfo_loop_seconds} s, → {destination}.",
        ),
    )
    populate_osc(preset, 0,
                 wavetable_audio=cycle, wavetable_name="matched_timbre",
                 level=0.7, on=True)
    set_lfo_from_curve(
        preset, lfo_index=0, curve_1d=envelope,
        rate_hz=rate_hz, name=f"Hilbert env ({duration_s:.1f}s)",
        n_points=n_lfo_points, smooth=True,
    )
    route_lfo(preset, lfo_index=0, destination=destination, amount=1.0)

    out_v = write_vital_preset(preset, out_path)
    out_c = write_settings_companion(
        "hilbert_modulator",
        out_path=out_v.replace(".vital", "_settings.json"),
        base_freq=float(timbre.base_freq),
        timbre_summary=_summarize_timbre(timbre),
        extras={
            "signal": {
                "duration_s": duration_s,
                "samplerate": float(sf),
                "n_samples": int(sig.size),
            },
            "lfo_routing": {
                "lfo_index": 0,
                "rate_hz": rate_hz,
                "destination": destination,
                "n_points": int(n_lfo_points),
                "loop_seconds": float(lfo_loop_seconds),
            },
        },
    )
    return {"vital": out_v, "companion": out_c}


# ---------------------------------------------------------------------------
# 8. Multi-IMF LFO bank — to_vital_imf_lfo_bank
# ---------------------------------------------------------------------------

_IMF_DEFAULT_DESTINATIONS = [
    "osc_1_level",
    "filter_1_cutoff",
    "osc_1_distortion_amount",
    "osc_1_wavetable_position",
    "osc_1_pan",
    "filter_1_resonance",
    "volume",
    "reverb_dry_wet",
]


def to_vital_imf_lfo_bank(
    imfs: Sequence[np.ndarray],
    timbre: Timbre,
    out_path: str,
    *,
    preset_name: str | None = None,
    destinations: list[str] | None = None,
    lfo_loop_seconds: float = 4.0,
) -> dict:
    """Each IMF (Hilbert-Huang mode) becomes one Vital LFO — N independently
    routable biosignal-derived modulators. Vital has 8 LFO slots; up to 8
    IMFs are consumed.
    """
    if not imfs:
        raise ValueError("to_vital_imf_lfo_bank: empty IMFs")
    timbre.validate()
    n = min(len(imfs), 8)
    destinations = destinations or _IMF_DEFAULT_DESTINATIONS[:n]
    if len(destinations) < n:
        destinations = list(destinations) + _IMF_DEFAULT_DESTINATIONS[len(destinations):n]

    name = preset_name or os.path.splitext(os.path.basename(out_path))[0]
    cycle = render_wavetable_cycle(timbre, table_size=WAVETABLE_FRAME_SIZE)
    rate_hz = 1.0 / max(lfo_loop_seconds, 0.01)

    preset = make_blank_preset(
        name=name,
        comments=_provenance_comment(
            timbre, preset_kind="imf_lfo_bank",
            extra=f"{n} IMFs → 8 LFOs, each routed to a different destination "
                  f"({', '.join(destinations[:n])}).",
        ),
    )
    populate_osc(preset, 0,
                 wavetable_audio=cycle, wavetable_name="matched_timbre",
                 level=0.7, on=True)

    routings: list[dict] = []
    for i in range(n):
        imf_arr = np.asarray(imfs[i], dtype=np.float64).flatten()
        set_lfo_from_curve(
            preset, lfo_index=i, curve_1d=imf_arr,
            rate_hz=rate_hz,
            name=f"IMF {i}",
            n_points=32, smooth=True,
        )
        route_lfo(preset, lfo_index=i, destination=destinations[i], amount=1.0)
        routings.append({
            "lfo_index": i,
            "imf_length": int(imf_arr.size),
            "destination": destinations[i],
            "rate_hz": rate_hz,
        })

    out_v = write_vital_preset(preset, out_path)
    out_c = write_settings_companion(
        "imf_lfo_bank",
        out_path=out_v.replace(".vital", "_settings.json"),
        base_freq=float(timbre.base_freq),
        timbre_summary=_summarize_timbre(timbre),
        extras={"n_imfs_consumed": n,
                "lfo_routing": routings},
    )
    return {"vital": out_v, "companion": out_c}


# ---------------------------------------------------------------------------
# 9. IMF stack as synthesis topology
# ---------------------------------------------------------------------------

def to_vital_imf_stack(
    imfs: Sequence[np.ndarray],
    timbre: Timbre,
    out_path: str,
    *,
    preset_name: str | None = None,
    base_freq: float = 220.0,
    fm_amount: float = 0.4,
    cycle_strategy: str = "whole_resampled",
) -> dict:
    """Three IMFs → three oscillators with cross-FM. Hilbert-Huang
    decomposition realized as the synth's *architecture*, not just content.

    Topology: IMF 0 (slowest) → osc 3, IMF 1 → osc 2, IMF 2 (fastest) → osc 1.
    Each oscillator's wavetable is built from the corresponding IMF (whole
    resampled to one cycle so its multi-cycle structure becomes the wave
    shape). osc 3 → osc 2 → osc 1 cross-FM is wired automatically.
    """
    if not imfs:
        raise ValueError("to_vital_imf_stack: empty IMFs")
    timbre.validate()
    n = min(len(imfs), 3)
    name = preset_name or os.path.splitext(os.path.basename(out_path))[0]

    # Ordered slowest → fastest by mean instantaneous frequency proxy
    # (heuristic: higher index of the IMF list = faster mode in EMD output).
    # We assume the user passes them in order [fastest, ..., slowest] which
    # matches biotuner's EMD_eeg convention; but reverse is also fine.
    imf_arrays = [np.asarray(x, dtype=np.float64).flatten() for x in imfs[:n]]

    preset = make_blank_preset(
        name=name,
        comments=_provenance_comment(
            timbre, preset_kind="imf_stack",
            extra=f"{n} IMFs → {n} oscillators, slow IMF FMs faster IMF "
                  f"(Hilbert-Huang as FM topology).",
        ),
    )
    osc_indices = list(range(n))[::-1]   # [n-1, ..., 0]
    for osc_idx, imf_arr in zip(osc_indices, imf_arrays):
        cycle = _imf_to_cycle(imf_arr, cycle_strategy)
        # Each subsequent oscillator is one octave higher to spread them
        transpose = float(12 * (n - 1 - osc_idx))
        # Cross-FM routing: osc N+1 → osc N (so osc 3 → osc 2, osc 2 → osc 1)
        destination = float(osc_idx) if osc_idx < n - 1 else None
        populate_osc(
            preset, osc_idx,
            wavetable_audio=cycle, wavetable_name=f"IMF_{osc_idx}",
            level=0.7 if osc_idx == 0 else 0.0,   # only osc 1 audible; others FM
            on=True,
            transpose=transpose,
            destination=destination,
            distortion_type=9.0 if osc_idx == 0 else None,
            distortion_amount=float(fm_amount) if osc_idx == 0 else None,
        )

    out_v = write_vital_preset(preset, out_path)
    out_c = write_settings_companion(
        "imf_stack",
        out_path=out_v.replace(".vital", "_settings.json"),
        base_freq=float(base_freq),
        timbre_summary=_summarize_timbre(timbre),
        extras={"n_imfs_consumed": n,
                "fm_amount": float(fm_amount),
                "cycle_strategy": cycle_strategy},
    )
    return {"vital": out_v, "companion": out_c}


def _imf_to_cycle(imf: np.ndarray, strategy: str) -> np.ndarray:
    """Extract one wavetable cycle from an IMF."""
    if imf.size < 2:
        return np.zeros(WAVETABLE_FRAME_SIZE, dtype=np.float32)
    if strategy == "whole_resampled":
        src_x = np.linspace(0, 1, imf.size)
        dst_x = np.linspace(0, 1, WAVETABLE_FRAME_SIZE)
        cycle = np.interp(dst_x, src_x, imf)
        m = float(np.max(np.abs(cycle)))
        if m > 0:
            cycle = cycle * (0.99 / m)
        return cycle.astype(np.float32, copy=False)
    if strategy == "first_cycle":
        sign = np.sign(imf)
        crossings = np.where(np.diff(sign) > 0)[0]
        if crossings.size < 2:
            return _imf_to_cycle(imf, "whole_resampled")
        seg = imf[crossings[0]:crossings[1]]
        return _imf_to_cycle(seg, "whole_resampled")
    raise ValueError(f"unknown cycle_strategy {strategy!r}")


# ---------------------------------------------------------------------------
# 10. Hilbert as found sound
# ---------------------------------------------------------------------------

def to_vital_hilbert_sample(
    signal,
    sf: float,
    out_path: str,
    *,
    preset_name: str | None = None,
    base_freq: float = 220.0,
    pitch_factor: float = 50.0,
    duration: float = 2.0,
    samplerate_out: int = 48000,
) -> dict:
    """Render the biosignal as audio via :func:`hilbert_instrument`, save
    the WAV alongside, and produce a Vital preset whose comments instruct
    the user to load that WAV into Vital's sample oscillator.

    Vital has a sample oscillator independent of the wavetable osc but its
    sample-loading is GUI-driven; we cannot pre-attach the file. The
    output bundle is therefore (preset.vital + sample.wav + settings.json),
    and the preset's ``comments`` tells the musician exactly which file to
    drag onto Vital's "Sample" slot.
    """
    from biotuner.harmonic_timbre import hilbert_instrument

    sig = np.asarray(signal, dtype=np.float64).flatten()
    if sig.size < 4:
        raise ValueError("to_vital_hilbert_sample: signal too short")

    audio = hilbert_instrument(
        sig, sf=float(sf),
        samplerate=samplerate_out, duration=duration,
        base_freq=base_freq, pitch_factor=pitch_factor,
    )
    name = preset_name or os.path.splitext(os.path.basename(out_path))[0]

    # Save the audio file next to the preset
    if not out_path.endswith(".vital"):
        out_path = out_path + ".vital"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    wav_path = out_path.replace(".vital", "_signal.wav")
    import soundfile as sf_io
    sf_io.write(wav_path, audio, samplerate_out)

    # Create a stub timbre for provenance
    timbre = Timbre(
        partials_hz=[base_freq], amplitudes=[1.0], base_freq=base_freq,
        matching_method="hilbert_instrument",
        metadata={"signal_duration_s": sig.size / float(sf),
                  "pitch_factor": float(pitch_factor)},
    )
    preset = make_blank_preset(
        name=name,
        comments=_provenance_comment(
            timbre, preset_kind="hilbert_sample",
            extra=f"To finish setup: drag '{os.path.basename(wav_path)}' onto "
                  f"Vital's Sample oscillator. Then enable Sample on the right "
                  f"side of the Synth tab. Signal: {sig.size} samples @ {sf} Hz, "
                  f"audio: {duration}s @ {samplerate_out} Hz.",
        ),
    )

    out_v = write_vital_preset(preset, out_path)
    out_c = write_settings_companion(
        "hilbert_sample",
        out_path=out_v.replace(".vital", "_settings.json"),
        base_freq=float(base_freq),
        timbre_summary=_summarize_timbre(timbre),
        extras={
            "signal": {"duration_s": sig.size / float(sf), "samplerate": float(sf)},
            "rendered_audio_wav": os.path.basename(wav_path),
            "rendered_duration_s": float(duration),
            "instructions": "Drag the wav onto Vital's Sample oscillator slot.",
        },
    )
    return {"vital": out_v, "companion": out_c, "sample_wav": wav_path}


# ---------------------------------------------------------------------------
# 11. Polyrhythm-gated
# ---------------------------------------------------------------------------

def to_vital_polyrhythm_gated(
    timbre: Timbre,
    gate_pattern: np.ndarray,
    out_path: str,
    *,
    preset_name: str | None = None,
    cycle_seconds: float = 1.0,
    destination: str = "osc_1_level",
) -> dict:
    """Polyrhythm coincidence pattern → stepped LFO → osc level (rhythmic gating).

    ``gate_pattern`` is a 1D array of 0s and 1s (or any binary-ish curve).
    It's embedded as LFO 1's shape with ``smooth=False`` so Vital steps
    cleanly between values rather than interpolating. Routes LFO 1 to
    ``destination`` (osc level by default — the matched timbre clicks on
    and off rhythmically).
    """
    timbre.validate()
    arr = np.asarray(gate_pattern, dtype=np.float64).flatten()
    if arr.size < 1:
        raise ValueError("to_vital_polyrhythm_gated: empty gate_pattern")

    name = preset_name or os.path.splitext(os.path.basename(out_path))[0]
    cycle = render_wavetable_cycle(timbre, table_size=WAVETABLE_FRAME_SIZE)
    rate_hz = 1.0 / max(cycle_seconds, 0.001)
    n_points = min(max(arr.size, 4), 32)

    preset = make_blank_preset(
        name=name,
        comments=_provenance_comment(
            timbre, preset_kind="polyrhythm_gated",
            extra=f"polyrhythm coincidence pattern ({arr.size} steps) → "
                  f"stepped LFO at {rate_hz:.2f} Hz → {destination}.",
        ),
    )
    populate_osc(preset, 0,
                 wavetable_audio=cycle, wavetable_name="matched_timbre",
                 level=0.7, on=True)
    set_lfo_from_curve(
        preset, lfo_index=0, curve_1d=arr,
        rate_hz=rate_hz, name=f"polyrhythm gate ({arr.size} steps)",
        n_points=n_points, smooth=False,    # stepped, not bezier
    )
    route_lfo(preset, lfo_index=0, destination=destination, amount=1.0,
              bipolar=False)

    out_v = write_vital_preset(preset, out_path)
    out_c = write_settings_companion(
        "polyrhythm_gated",
        out_path=out_v.replace(".vital", "_settings.json"),
        base_freq=float(timbre.base_freq),
        timbre_summary=_summarize_timbre(timbre),
        extras={
            "polyrhythm": {
                "n_steps": int(arr.size),
                "cycle_seconds": float(cycle_seconds),
                "rate_hz": rate_hz,
            },
            "lfo_routing": {"lfo_index": 0, "destination": destination,
                            "smooth": False},
        },
    )
    return {"vital": out_v, "companion": out_c}


# ---------------------------------------------------------------------------
# 12. Markov macro — TimbreSequence as Macro 1 walk
# ---------------------------------------------------------------------------

def to_vital_markov_macro(
    seq: TimbreSequence,
    out_path: str,
    *,
    preset_name: str | None = None,
) -> dict:
    """TimbreSequence frames as a multi-frame wavetable + Macro 1 mapped to
    osc 1 wavetable position.

    The performer turns Macro 1 to walk through the sequence's harmonic
    states, frame by frame. Combined with auto-LFO on Macro 1 you get
    automatic state walking.
    """
    if not isinstance(seq, TimbreSequence):
        raise TypeError(f"expected TimbreSequence, got {type(seq).__name__}")
    if seq.n_frames() < 2:
        raise ValueError("to_vital_markov_macro: need ≥ 2 frames")

    name = preset_name or os.path.splitext(os.path.basename(out_path))[0]
    cycles = [
        render_wavetable_cycle(t, table_size=WAVETABLE_FRAME_SIZE)
        for t in seq.frames
    ]
    audio = np.concatenate(cycles).astype(np.float32, copy=False)

    # Use the first frame's metadata as the preset's reference timbre
    ref = seq.frames[0]
    preset = make_blank_preset(
        name=name,
        comments=_provenance_comment(
            ref, preset_kind="markov_macro",
            extra=f"TimbreSequence with {seq.n_frames()} frames → wavetable. "
                  f"Macro 1 walks osc 1 wavetable position 0 → {seq.n_frames() - 1}.",
        ),
    )
    populate_osc(preset, 0,
                 wavetable_audio=audio, wavetable_name="markov_walk",
                 level=0.7, on=True)

    # Map Macro 1 → osc_1_wavetable_position
    slot = find_free_modulation_slot(preset)
    set_modulation(
        preset, slot=slot,
        source="macro_control_1",
        destination="osc_1_wavetable_position",
        amount=1.0, bipolar=False,
    )
    preset["macro1"] = "harmonic state (frame 0..N-1)"

    out_v = write_vital_preset(preset, out_path)
    out_c = write_settings_companion(
        "markov_macro",
        out_path=out_v.replace(".vital", "_settings.json"),
        base_freq=float(ref.base_freq),
        timbre_summary=_summarize_timbre(ref),
        extras={
            "sequence": {
                "n_frames": int(seq.n_frames()),
                "matched_tunings_per_frame": [
                    list(t.matched_tuning) if t.matched_tuning is not None else None
                    for t in seq.frames
                ],
            },
            "macro_routing": {
                "macro": "macro_control_1", "name": "harmonic state",
                "destination": "osc_1_wavetable_position",
            },
        },
    )
    return {"vital": out_v, "companion": out_c}


# ---------------------------------------------------------------------------
# 13. Ensemble — N coordinated presets sharing a tuning
# ---------------------------------------------------------------------------

def to_vital_ensemble(
    timbre: Timbre,
    out_dir: str,
    *,
    bundle_name: str = "ensemble",
    signal: np.ndarray | None = None,
    sf: float | None = None,
    imfs: Sequence[np.ndarray] | None = None,
    polyrhythm_gate: np.ndarray | None = None,
) -> dict:
    """Generate a coordinated ensemble of presets sharing the matched tuning
    but emphasizing different biotuner facets.

    Always emits at least one preset (spectral). Conditionally emits more
    based on which optional inputs are provided:

        signal + sf provided     → adds a hilbert_modulator preset (pad voice)
        imfs provided            → adds an imf_lfo_bank preset (texture voice)
        polyrhythm_gate provided → adds a polyrhythm_gated preset (rhythm voice)
        AM modulators on timbre  → adds a pac_driven preset (motion voice)
        FM modulators on timbre  → adds a fm preset (lead voice)

    Each output preset has a descriptive name (e.g. "ensemble_pad",
    "ensemble_lead") so they're easy to identify in Vital's browser.

    Returns
    -------
    dict
        ``{voice_label: result_dict}`` with one entry per generated preset.
    """
    timbre.validate()
    os.makedirs(out_dir, exist_ok=True)
    results: dict[str, dict] = {}

    # Always: one wavetable_morph preset as the "base voice"
    results["base_morph"] = to_vital_wavetable_morph(
        timbre, os.path.join(out_dir, f"{bundle_name}_base"),
        n_frames=64, evolution="tilt", lfo_rate_hz=0.25,
        preset_name=f"{bundle_name}_base",
    )

    if signal is not None and sf is not None:
        results["pad"] = to_vital_hilbert_modulator(
            signal, sf, timbre,
            os.path.join(out_dir, f"{bundle_name}_pad"),
            destination="osc_1_level",
            preset_name=f"{bundle_name}_pad",
        )

    if imfs:
        results["texture"] = to_vital_imf_lfo_bank(
            imfs, timbre,
            os.path.join(out_dir, f"{bundle_name}_texture"),
            preset_name=f"{bundle_name}_texture",
        )

    if polyrhythm_gate is not None:
        results["rhythm"] = to_vital_polyrhythm_gated(
            timbre, polyrhythm_gate,
            os.path.join(out_dir, f"{bundle_name}_rhythm"),
            preset_name=f"{bundle_name}_rhythm",
        )

    if timbre.am_modulators:
        results["motion"] = to_vital_pac_driven(
            timbre, os.path.join(out_dir, f"{bundle_name}_motion"),
            preset_name=f"{bundle_name}_motion",
        )

    if timbre.fm_modulators:
        results["lead"] = to_vital_fm(
            timbre, os.path.join(out_dir, f"{bundle_name}_lead"),
            preset_name=f"{bundle_name}_lead",
        )

    # Top-level manifest
    import json
    manifest = {
        "format": "biotuner_vital_ensemble",
        "format_version": 1,
        "bundle_name": bundle_name,
        "voices": list(results.keys()),
        "shared_timbre": _summarize_timbre(timbre),
    }
    manifest_path = os.path.join(out_dir, f"{bundle_name}.ensemble.manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2, default=str)
    results["__manifest__"] = manifest_path

    return results
