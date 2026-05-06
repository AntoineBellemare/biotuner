"""Csound exporter — generate a self-contained .csd file.

Module type: Functions

A ``.csd`` (Csound document) bundles ``<CsOptions>``, ``<CsInstruments>``,
and ``<CsScore>`` in one file. The instrument body is hand-written
additive synthesis with one ``oscili`` per partial, parameterized by
the Timbre's partial frequencies and amplitudes. Optional spectral tilt
and per-partial decay are baked in.

A simple demo score (matched-tuning scale or chord) gives composers a
runnable starting point.
"""

from __future__ import annotations

import os
from textwrap import dedent

from biotuner.harmonic_timbre.cross_modal import write_sidecar
from biotuner.harmonic_timbre.exporters._common import write_manifest
from biotuner.harmonic_timbre.timbre import Timbre


def _instrument_body(timbre: Timbre, *, instr_num: int = 1) -> str:
    """Produce the Csound instrument body using the timbre's partials.

    When ``timbre.am_modulators`` and/or ``timbre.fm_modulators`` are
    populated, the instrument emits an FM/AM oscillator graph instead of
    plain additive synthesis. Each carrier partial gets:

        * one ``oscili`` per FM modulator that targets it (sums into the
          carrier's instantaneous frequency);
        * one ``oscili``-driven multiplier per AM modulator that targets it
          (chained as ``(1 + depth · sin(...))`` factors).
    """
    n = timbre.n_partials()
    base = float(timbre.base_freq) if timbre.base_freq > 0 else float(min(timbre.partials_hz))
    rel_freqs = [float(f) / base for f in timbre.partials_hz]
    amps = list(timbre.amplitudes / max(timbre.amplitudes.max(), 1e-9))
    decays = (
        list(timbre.decay_times) if timbre.decay_times is not None
        else [None] * n
    )

    # Bucket modulators per carrier
    am_per_carrier: dict[int, list] = {}
    fm_per_carrier: dict[int, list] = {}
    for m in timbre.am_modulators:
        if 0 <= m.carrier_idx < n and m.mod_type == "AM":
            am_per_carrier.setdefault(m.carrier_idx, []).append(m)
    for m in timbre.fm_modulators:
        if 0 <= m.carrier_idx < n and m.mod_type == "FM":
            fm_per_carrier.setdefault(m.carrier_idx, []).append(m)

    lines = [
        f"instr {instr_num}",
        "  ifreq = p4",
        "  iamp  = p5",
        "  idur  = p3",
        "  asig init 0",
    ]
    for i in range(n):
        rel = rel_freqs[i]
        amp = amps[i]
        decay = decays[i]
        carrier_freq_expr = f"ifreq * {rel:.6f}"

        # FM: build a per-partial deviation expression
        if i in fm_per_carrier:
            for j, m in enumerate(fm_per_carrier[i]):
                lines.append(
                    f"  a{i}fm{j} oscili {float(m.depth):.6f}, "
                    f"{float(m.mod_freq):.6f}, 1, {float(m.phase):.6f}"
                )
            fm_sum = " + ".join(f"a{i}fm{j}" for j in range(len(fm_per_carrier[i])))
            carrier_freq_expr = f"({carrier_freq_expr}) + {fm_sum}"

        # AM: build a per-partial amplitude-envelope multiplier
        am_factor = "1"
        if i in am_per_carrier:
            for j, m in enumerate(am_per_carrier[i]):
                lines.append(
                    f"  a{i}am{j} oscili {float(m.depth):.6f}, "
                    f"{float(m.mod_freq):.6f}, 1, {float(m.phase):.6f}"
                )
            am_factor = " * ".join(
                f"(1 + a{i}am{j})" for j in range(len(am_per_carrier[i]))
            )

        # Decay envelope (per-partial)
        if decay is not None and decay > 0:
            lines.append(
                f"  k{i}env expon  iamp * {amp:.6f}, idur, "
                f"iamp * {amp:.6f} * exp(-idur/{decay:.6f})"
            )
            amp_expr = f"k{i}env * {am_factor}"
        else:
            amp_expr = f"iamp * {amp:.6f} * {am_factor}"

        # The carrier oscillator. With FM, use a phasor + table_kr_freq pattern.
        if i in fm_per_carrier:
            # Csound: oscili can take an a-rate frequency. Use 'oscili amp, kfreq, 1'
            # with the freq-deviation already added.
            lines.append(f"  a{i}    oscili {amp_expr}, {carrier_freq_expr}, 1")
        else:
            lines.append(f"  a{i}    oscili {amp_expr}, {carrier_freq_expr}, 1")
        lines.append(f"  asig = asig + a{i}")

    lines += [
        "  outs asig, asig",
        "endin",
    ]
    return "\n".join(lines)


def _score_for_tuning(
    timbre: Timbre,
    *,
    base_freq: float,
    pattern: str,
    note_dur: float,
    note_amp: float,
    instr_num: int,
) -> str:
    """Produce a ``<CsScore>`` body that exercises the matched tuning."""
    if timbre.matched_tuning is None or pattern == "none":
        return f"i{instr_num} 0 {note_dur:.3f} {base_freq:.3f} {note_amp:.3f}"

    ratios = list(timbre.matched_tuning)
    lines: list[str] = []
    if pattern == "scale":
        t = 0.0
        for r in ratios:
            f = base_freq * float(r)
            lines.append(f"i{instr_num} {t:.3f} {note_dur:.3f} {f:.3f} {note_amp:.3f}")
            t += note_dur
    elif pattern == "chord":
        for r in ratios:
            f = base_freq * float(r)
            lines.append(f"i{instr_num} 0 {note_dur:.3f} {f:.3f} {note_amp:.3f}")
    elif pattern == "arpeggio":
        # Up then down
        seq = ratios + ratios[-2::-1]
        t = 0.0
        for r in seq:
            f = base_freq * float(r)
            lines.append(f"i{instr_num} {t:.3f} {note_dur / 2:.3f} {f:.3f} {note_amp:.3f}")
            t += note_dur / 2
    else:
        raise ValueError(f"_score_for_tuning: unknown pattern {pattern!r}")
    return "\n".join(lines)


_CSD_TEMPLATE = """\
<CsoundSynthesizer>
<CsOptions>
-odac
</CsOptions>

<CsInstruments>
sr     = 48000
ksmps  = 32
nchnls = 2
0dbfs  = 1

; Sine table (1024 points)
gisine ftgen 1, 0, 1024, 10, 1

{instrument}

</CsInstruments>

<CsScore>
{score}
e
</CsScore>
</CsoundSynthesizer>
"""


def export_csound(
    timbre: Timbre,
    out_path: str,
    *,
    instrument_num: int = 1,
    base_freq: float = 220.0,
    note_dur: float = 1.0,
    note_amp: float = 0.5,
    demo_pattern: str = "scale",
    include_sidecar: bool = True,
) -> dict:
    """Write a self-contained Csound document.

    Parameters
    ----------
    timbre : Timbre
    out_path : str
        Output ``.csd`` path. ``.csd`` appended if missing.
    instrument_num : int, default=1
        Csound instrument number.
    base_freq : float, default=220.0
        Hz at which the score plays the unison (1/1).
    note_dur : float, default=1.0
        Seconds per note in the demo score.
    note_amp : float, default=0.5
        ``p5`` amplitude in the score.
    demo_pattern : str, default='scale'
        ``'scale'`` | ``'chord'`` | ``'arpeggio'`` | ``'none'``.
    include_sidecar : bool, default=True

    Returns
    -------
    dict
        ``{'csd', 'sidecar', 'manifest'}``.
    """
    timbre.validate()

    if not out_path.endswith(".csd"):
        out_path = out_path + ".csd"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    instr_body = _instrument_body(timbre, instr_num=instrument_num)
    score = _score_for_tuning(
        timbre,
        base_freq=base_freq,
        pattern=demo_pattern,
        note_dur=note_dur,
        note_amp=note_amp,
        instr_num=instrument_num,
    )

    body = _CSD_TEMPLATE.format(instrument=instr_body, score=score)
    with open(out_path, "w", encoding="utf-8") as fp:
        fp.write(body)

    result = {"csd": out_path}

    if include_sidecar:
        sidecar_dir = out_path.replace(".csd", "_sidecar")
        sidecar = write_sidecar(timbre, sidecar_dir, stem=os.path.basename(out_path).replace(".csd", ""))
        result["sidecar"] = sidecar

    manifest_path = out_path.replace(".csd", ".manifest.json")
    manifest = {
        "format": "biotuner_csound",
        "format_version": 1,
        "instrument_num": int(instrument_num),
        "demo_pattern": demo_pattern,
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
