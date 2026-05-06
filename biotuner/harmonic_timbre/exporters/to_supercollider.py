"""SuperCollider exporter — generate a .scd with a SynthDef + Pbind demo.

Module type: Functions

The generated ``.scd`` file is loadable in SuperCollider by selecting
the contents and pressing Cmd+Return / Ctrl+Return on each block. It
contains:

    1. A ``SynthDef`` — additive synth using ``SinOsc`` per partial,
       parameterized by relative partial frequency, amplitude, and
       optional decay time. Plays through ``EnvGen.kr(Env.linen, ...)``.
    2. ``Tuning.new(<ratios>)`` and ``Scale.new(...)`` definitions for
       the matched tuning, so a Pbind can address the scale by degree.
    3. An optional ``Pbind`` demo that walks the scale.

A composer pastes the file into the SC IDE; the SynthDef boots, the
demo plays.
"""

from __future__ import annotations

import os
from textwrap import dedent

from biotuner.harmonic_timbre.cross_modal import write_sidecar
from biotuner.harmonic_timbre.exporters._common import write_manifest
from biotuner.harmonic_timbre.timbre import Timbre


def _synthdef(timbre: Timbre, *, name: str) -> str:
    """SynthDef body: additive synth with per-partial controls.

    When ``timbre.am_modulators`` / ``timbre.fm_modulators`` are populated,
    each modulator becomes a ``SinOsc.ar`` that adds into the carrier's
    instantaneous frequency (FM) or multiplies its amplitude (AM). The
    output is a sum of ``n_partials`` carrier oscillators each with its
    own modulation graph.
    """
    n = timbre.n_partials()
    base = float(timbre.base_freq) if timbre.base_freq > 0 else float(min(timbre.partials_hz))
    rel = [float(f) / base for f in timbre.partials_hz]
    amps = list(timbre.amplitudes / max(timbre.amplitudes.max(), 1e-9))
    decays = (
        list(timbre.decay_times) if timbre.decay_times is not None
        else [None] * n
    )

    am_per_carrier: dict[int, list] = {}
    fm_per_carrier: dict[int, list] = {}
    for m in timbre.am_modulators:
        if 0 <= m.carrier_idx < n and m.mod_type == "AM":
            am_per_carrier.setdefault(m.carrier_idx, []).append(m)
    for m in timbre.fm_modulators:
        if 0 <= m.carrier_idx < n and m.mod_type == "FM":
            fm_per_carrier.setdefault(m.carrier_idx, []).append(m)

    partial_lines: list[str] = []
    for i, (r, a, d) in enumerate(zip(rel, amps, decays)):
        # FM: sum of SinOsc.ar terms summed into the carrier frequency
        if i in fm_per_carrier:
            fm_terms = " + ".join(
                f"SinOsc.ar({float(m.mod_freq):.6f}, {float(m.phase):.6f}) * {float(m.depth):.6f}"
                for m in fm_per_carrier[i]
            )
            freq_expr = f"(freq * {r:.6f} + {fm_terms})"
        else:
            freq_expr = f"freq * {r:.6f}"

        # AM: product of (1 + depth*SinOsc.ar(mod_freq))
        if i in am_per_carrier:
            am_terms = " * ".join(
                f"(1 + SinOsc.ar({float(m.mod_freq):.6f}, {float(m.phase):.6f}) * {float(m.depth):.6f})"
                for m in am_per_carrier[i]
            )
            am_expr = f" * {am_terms}"
        else:
            am_expr = ""

        # Decay envelope (per-partial)
        if d is not None and d > 0:
            decay_expr = f" * EnvGen.kr(Env.new([1, exp(-1 * dur / {d:.6f})], [dur]))"
        else:
            decay_expr = ""

        partial_lines.append(
            f"        SinOsc.ar({freq_expr}) * {a:.6f}{am_expr}{decay_expr} * 0.5,"
        )

    partials_block = "\n".join(partial_lines)

    return dedent(f"""
        SynthDef(\\{name}, {{ |freq=220, amp=0.5, dur=1, gate=1|
            var sig, env;
            env = EnvGen.kr(Env.linen(0.005, dur, 0.4), gate, doneAction: 2);
            sig = Mix.new([
        {partials_block}
            ]);
            Out.ar(0, (sig * env * amp).dup);
        }}).add;
    """).strip()


def _tuning_block(timbre: Timbre, *, var_name: str = "biotunerTuning") -> str:
    """Tuning.new + Scale.new for the matched tuning."""
    if timbre.matched_tuning is None:
        return "// no matched_tuning available; tuning block skipped"

    ratios = list(timbre.matched_tuning)
    # Tuning.new takes log2 of each ratio and an octave ratio.
    tuning_steps = [f"{1200.0 * (r ** 0).bit_length() if False else 0:.5f}"]  # placeholder unused
    # SC convention: pass cents per step, with octave as last value
    cents = [1200.0 * (float(r).__pow__(1)) for r in ratios]  # not what we want
    # Correct formula: cents = 1200 * log2(ratio)
    import math
    cents_list = [1200.0 * math.log2(float(r)) for r in ratios]
    cents_str = ", ".join(f"{c:.5f}" for c in cents_list[:-1])
    octave = cents_list[-1]
    return dedent(f"""
        ~{var_name} = Tuning.new([{cents_str}], {octave:.5f}, "biotuner-matched");
        ~biotunerScale = Scale.new((0..{len(ratios) - 2}), {len(ratios) - 1}, ~{var_name}, "biotuner");
    """).strip()


def _pbind_block(*, synthdef_name: str, base_freq: float, note_dur: float) -> str:
    return dedent(f"""
        // Run the demo:
        ~biotunerPattern = Pbind(
            \\instrument, \\{synthdef_name},
            \\scale, ~biotunerScale,
            \\degree, Pseq([0, 1, 2, 3, 4, 5, 6, 7, 0], 1),
            \\dur, {note_dur:.3f},
            \\amp, 0.5,
            \\octave, 5,
        ).play;
    """).strip()


_SCD_TEMPLATE = """\
// SuperCollider patch generated by biotuner.harmonic_timbre.
// Boot the server first:
//     s.boot;
// Then evaluate each block (Cmd/Ctrl+Return).
// matched_tuning = {matched_tuning}
// matching_method = {matching_method}

(
{synthdef}
)

(
{tuning_block}
)

(
{pbind_block}
)
"""


def export_supercollider(
    timbre: Timbre,
    out_path: str,
    *,
    synthdef_name: str = "biotunerTimbre",
    base_freq: float = 220.0,
    note_dur: float = 0.5,
    include_demo: bool = True,
    include_sidecar: bool = True,
) -> dict:
    """Write a SuperCollider ``.scd`` file with SynthDef + Tuning + Pbind demo.

    Returns
    -------
    dict
        ``{'scd', 'sidecar', 'manifest'}``.
    """
    timbre.validate()

    if not out_path.endswith(".scd"):
        out_path = out_path + ".scd"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    synthdef = _synthdef(timbre, name=synthdef_name)
    tuning_block = _tuning_block(timbre)
    pbind_block = (
        _pbind_block(synthdef_name=synthdef_name, base_freq=base_freq, note_dur=note_dur)
        if include_demo and timbre.matched_tuning is not None
        else "// (demo Pbind not included)"
    )

    body = _SCD_TEMPLATE.format(
        matched_tuning=timbre.matched_tuning,
        matching_method=timbre.matching_method,
        synthdef=synthdef,
        tuning_block=tuning_block,
        pbind_block=pbind_block,
    )
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write(body)

    result = {"scd": out_path}

    if include_sidecar:
        sidecar_dir = out_path.replace(".scd", "_sidecar")
        sidecar = write_sidecar(timbre, sidecar_dir, stem=os.path.basename(out_path).replace(".scd", ""))
        result["sidecar"] = sidecar

    manifest_path = out_path.replace(".scd", ".manifest.json")
    manifest = {
        "format": "biotuner_supercollider",
        "format_version": 1,
        "synthdef_name": synthdef_name,
        "base_freq": float(base_freq),
        "timbre": {
            "matched_tuning": list(timbre.matched_tuning) if timbre.matched_tuning is not None else None,
            "matching_method": timbre.matching_method,
            "n_partials": timbre.n_partials(),
        },
    }
    write_manifest(manifest_path, manifest)
    result["manifest"] = manifest_path

    return result
