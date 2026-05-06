"""``export_full_bundle`` — one call → multi-format kit ready for DAW use.

Module type: Functions
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import soundfile as sf

from biotuner.harmonic_timbre.cross_modal import write_sidecar
from biotuner.harmonic_timbre.exporters._common import write_manifest
from biotuner.harmonic_timbre.exporters.to_csound import export_csound
from biotuner.harmonic_timbre.exporters.to_sfz import export_sfz
from biotuner.harmonic_timbre.exporters.to_supercollider import export_supercollider
from biotuner.harmonic_timbre.exporters.to_surge import export_surge_bundle
from biotuner.harmonic_timbre.exporters.to_wav import export_wav_pack
from biotuner.harmonic_timbre.exporters.to_wavetable import export_wavetable
from biotuner.harmonic_timbre.exporters.tuning_files import export_tuning_files
from biotuner.harmonic_timbre.synthesis import render_wavetable_cycle
from biotuner.harmonic_timbre.timbre import Timbre, TimbreSequence


_DEFAULT_FORMATS = ("wav", "sfz", "wavetable", "surge", "csound", "supercollider", "tuning")


def export_full_bundle(
    timbre: Timbre,
    out_dir: str,
    *,
    bundle_name: str = "biotuner_timbre",
    formats: Iterable[str] = _DEFAULT_FORMATS,
    reference_pitches_midi=(36, 48, 60, 72, 84),
    a4: float = 440.0,
    duration: float = 4.0,
    samplerate: int = 48000,
    bit_depth: int = 24,
    n_wavetable_frames: int = 8,
    csound_pattern: str = "scale",
    sc_base_freq: float = 220.0,
) -> dict:
    """One call → produce a multi-format kit for DAW / live use.

    The output directory contains the sub-bundles for every requested
    format, plus a top-level ``<bundle_name>.bundle.manifest.json``
    cataloguing what was written.

    Parameters
    ----------
    timbre : Timbre
    out_dir : str
        Output directory (created if missing). Each format writes into
        a subfolder so they don't collide.
    bundle_name : str
    formats : iterable of str, default=('wav','sfz','wavetable','surge','csound','supercollider','tuning')
        Which exporters to run. Unknown values raise.
    reference_pitches_midi, a4, duration, samplerate, bit_depth
        Forwarded to ``to_wav`` / ``to_sfz``.
    n_wavetable_frames
        Forwarded to ``to_wavetable`` and ``to_surge``.
    csound_pattern
        Forwarded to ``to_csound`` (``'scale'`` | ``'chord'`` | ``'arpeggio'`` | ``'none'``).
    sc_base_freq
        Forwarded to ``to_supercollider``.

    Returns
    -------
    dict
        Mapping of format name → result dict (the dict each exporter
        returns), plus ``'manifest'`` and ``'sidecar'`` at the top level.
    """
    timbre.validate()
    fmts = list(formats)
    known = {"wav", "sfz", "wavetable", "surge", "csound", "supercollider", "tuning"}
    bad = set(fmts) - known
    if bad:
        raise ValueError(
            f"export_full_bundle: unknown format(s) {sorted(bad)}. Known: {sorted(known)}"
        )

    os.makedirs(out_dir, exist_ok=True)
    results: dict = {}

    if "wav" in fmts:
        sub = os.path.join(out_dir, "wav")
        results["wav"] = export_wav_pack(
            timbre, sub,
            bundle_name=bundle_name,
            reference_pitches_midi=reference_pitches_midi,
            duration=duration, samplerate=samplerate, bit_depth=bit_depth, a4=a4,
            include_sidecar=False,  # one shared sidecar at the top level
        )

    if "sfz" in fmts:
        sub = os.path.join(out_dir, "sfz")
        results["sfz"] = export_sfz(
            timbre, sub,
            bundle_name=bundle_name,
            reference_pitches_midi=reference_pitches_midi,
            duration=duration, samplerate=samplerate, bit_depth=bit_depth, a4=a4,
            include_sidecar=False,
        )

    if "wavetable" in fmts:
        sub = os.path.join(out_dir, "wavetable")
        os.makedirs(sub, exist_ok=True)
        wt_path = os.path.join(sub, f"{bundle_name}.wav")
        results["wavetable"] = export_wavetable(
            timbre, wt_path,
            n_frames=n_wavetable_frames,
            synth_target="vital",
            include_sidecar=False,
        )

    if "surge" in fmts:
        sub = os.path.join(out_dir, "surge")
        results["surge"] = export_surge_bundle(
            timbre, sub,
            bundle_name=bundle_name,
            n_wavetable_frames=n_wavetable_frames,
            a4=a4,
        )

    if "tuning" in fmts and timbre.matched_tuning is not None:
        sub = os.path.join(out_dir, "tuning")
        os.makedirs(sub, exist_ok=True)
        results["tuning"] = export_tuning_files(
            timbre, sub, bundle_name=bundle_name, reference_freq=a4,
        )

    if "csound" in fmts:
        sub = os.path.join(out_dir, "csound")
        os.makedirs(sub, exist_ok=True)
        out_path = os.path.join(sub, f"{bundle_name}.csd")
        results["csound"] = export_csound(
            timbre, out_path,
            base_freq=sc_base_freq,
            demo_pattern=csound_pattern,
            include_sidecar=False,
        )

    if "supercollider" in fmts:
        sub = os.path.join(out_dir, "supercollider")
        os.makedirs(sub, exist_ok=True)
        out_path = os.path.join(sub, f"{bundle_name}.scd")
        results["supercollider"] = export_supercollider(
            timbre, out_path,
            base_freq=sc_base_freq,
            include_sidecar=False,
        )

    # Single shared sidecar at the bundle root
    sidecar_dir = os.path.join(out_dir, "sidecar")
    results["sidecar"] = write_sidecar(timbre, sidecar_dir, stem=bundle_name)

    # Top-level manifest
    manifest = {
        "format": "biotuner_full_bundle",
        "format_version": 1,
        "bundle_name": bundle_name,
        "formats": list(fmts),
        "timbre": {
            "matched_tuning": list(timbre.matched_tuning) if timbre.matched_tuning is not None else None,
            "matching_method": timbre.matching_method,
            "n_partials": timbre.n_partials(),
            "metadata": dict(timbre.metadata),
        },
        "subdirectories": {
            f: f for f in fmts
        },
    }
    manifest_path = os.path.join(out_dir, f"{bundle_name}.bundle.manifest.json")
    write_manifest(manifest_path, manifest)
    results["manifest"] = manifest_path

    return results


# ---------------------------------------------------------------------------
# export_full_bundle_sequence
# ---------------------------------------------------------------------------

def export_full_bundle_sequence(
    seq: TimbreSequence,
    out_dir: str,
    *,
    bundle_name: str = "biotuner_sequence",
    per_frame_formats: Iterable[str] = ("sfz", "tuning", "csound"),
    a4: float = 440.0,
    duration: float = 4.0,
    samplerate: int = 48000,
    bit_depth: int = 24,
    reference_pitches_midi: Iterable[int] = (36, 48, 60, 72, 84),
    morph_wavetable_synth_target: str = "vital",
    morph_wavetable_table_size: int = 2048,
    sequence_audio: bool = True,
    sequence_audio_frame_duration: float = 0.5,
    sequence_audio_crossfade: float = 0.05,
) -> dict:
    """Export a TimbreSequence as a multi-frame musician kit.

    A TimbreSequence is time-resolved — successive Timbres come from either
    signal-window chunking (``timbre_sequence_from_signal``,
    ``timbre_sequence_from_transitional_harmony``) or generative-model walks
    (``timbre_sequence_from_markov_walk``) or hand-authored ratio frames
    (``timbre_sequence_from_ratio_frames``). This function is source-agnostic.

    The output is a folder structured like:

        out_dir/
            <name>.morph.wav                cross-frame wavetable (one cycle per
                                            frame, concatenated → load as a Vital
                                            / Serum / Surge wavetable; the
                                            table-position knob walks through the
                                            sequence)
            <name>.sequence_audio.wav       continuous rendered audio of the whole
                                            sequence with crossfades (optional)
            <name>.sequence.manifest.json   top-level catalogue: frame timestamps,
                                            per-frame matched_tuning, file paths
            sidecar/                        single shared visual fingerprint
                                            using the FIRST frame's Timbre
            frames/
                frame_00/                   per-frame sub-bundle (only the formats
                frame_01/                   listed in per_frame_formats — typical
                ...                         choice: sfz + tuning + csound)
                frame_NN/

    Parameters
    ----------
    seq : TimbreSequence
        Source sequence.
    out_dir : str
        Output directory (created if missing).
    bundle_name : str, default='biotuner_sequence'
    per_frame_formats : iterable of str, default=('sfz', 'tuning', 'csound')
        Which sub-bundle formats to write *for each frame*. Use a small subset
        (default 3) — running the full 7-format bundle per frame on a 64-frame
        sequence produces hundreds of files. Set to ``[]`` to skip per-frame
        bundles entirely.
    a4, duration, samplerate, bit_depth, reference_pitches_midi
        Forwarded to per-frame ``export_full_bundle`` calls.
    morph_wavetable_synth_target : str, default='vital'
    morph_wavetable_table_size : int, default=2048
    sequence_audio : bool, default=True
        If True, also write ``<name>.sequence_audio.wav`` — a continuous
        rendering of the whole sequence with crossfades between frames.
    sequence_audio_frame_duration : float, default=0.5
    sequence_audio_crossfade : float, default=0.05

    Returns
    -------
    dict
        ``{
            'morph_wavetable': '<.wav path>',
            'sequence_audio':  '<.wav path>' or None,
            'manifest':        '<.json path>',
            'sidecar':         {...},
            'frames':          [list of per-frame export_full_bundle result dicts],
        }``
    """
    if not isinstance(seq, TimbreSequence):
        raise TypeError(
            f"export_full_bundle_sequence: expected TimbreSequence, got {type(seq).__name__}"
        )
    if seq.n_frames() < 1:
        raise ValueError("export_full_bundle_sequence: empty sequence")

    fmts = list(per_frame_formats)
    known = {"wav", "sfz", "wavetable", "surge", "csound", "supercollider", "tuning"}
    bad = set(fmts) - known
    if bad:
        raise ValueError(
            f"export_full_bundle_sequence: unknown per_frame_formats {sorted(bad)}. "
            f"Known: {sorted(known)}"
        )

    os.makedirs(out_dir, exist_ok=True)
    results: dict = {"frames": []}

    # 1. Cross-frame morph wavetable — one single-cycle per frame, concatenated.
    morph_wt_path = os.path.join(out_dir, f"{bundle_name}.morph.wav")
    morph_audio = _wavetable_audio_from_sequence(seq, morph_wavetable_table_size)
    subtype = {2048: "FLOAT"}.get(morph_wavetable_table_size, "FLOAT")
    if morph_wavetable_synth_target in ("serum", "surge"):
        subtype = "PCM_16"
    sf.write(morph_wt_path, morph_audio, 48000, subtype=subtype)
    results["morph_wavetable"] = morph_wt_path

    # 2. Continuous sequence audio (optional)
    if sequence_audio:
        seq_audio_path = os.path.join(out_dir, f"{bundle_name}.sequence_audio.wav")
        audio = seq.synthesize(
            samplerate=samplerate,
            frame_duration=sequence_audio_frame_duration,
            crossfade=sequence_audio_crossfade,
        )
        sf.write(seq_audio_path, audio, samplerate)
        results["sequence_audio"] = seq_audio_path
    else:
        results["sequence_audio"] = None

    # 3. Per-frame sub-bundles
    if fmts:
        frames_dir = os.path.join(out_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        for i, frame_timbre in enumerate(seq.frames):
            frame_dir = os.path.join(frames_dir, f"frame_{i:03d}")
            try:
                frame_result = export_full_bundle(
                    frame_timbre, frame_dir,
                    bundle_name=f"{bundle_name}_frame_{i:03d}",
                    formats=fmts,
                    reference_pitches_midi=reference_pitches_midi,
                    a4=a4, duration=duration,
                    samplerate=samplerate, bit_depth=bit_depth,
                    n_wavetable_frames=1,    # per-frame is single-cycle
                )
            except Exception as exc:
                # A single bad frame shouldn't kill the whole sequence export.
                frame_result = {"error": str(exc)}
            results["frames"].append(frame_result)

    # 4. Single shared sidecar (using the first frame's timbre as the reference)
    sidecar_dir = os.path.join(out_dir, "sidecar")
    results["sidecar"] = write_sidecar(seq.frames[0], sidecar_dir, stem=bundle_name)

    # 5. Top-level manifest
    times_list = (
        seq.times.tolist() if seq.times is not None
        else [float(i) for i in range(seq.n_frames())]
    )
    manifest = {
        "format": "biotuner_full_bundle_sequence",
        "format_version": 1,
        "bundle_name": bundle_name,
        "n_frames": int(seq.n_frames()),
        "per_frame_formats": list(fmts),
        "morph_wavetable": os.path.relpath(morph_wt_path, out_dir).replace(os.sep, "/"),
        "sequence_audio": (
            os.path.relpath(results["sequence_audio"], out_dir).replace(os.sep, "/")
            if results["sequence_audio"] else None
        ),
        "frames": [
            {
                "index": i,
                "time_seconds": times_list[i] if i < len(times_list) else None,
                "matched_tuning": (
                    list(t.matched_tuning) if t.matched_tuning is not None else None
                ),
                "matching_method": t.matching_method,
                "metadata": dict(t.metadata),
                "sub_bundle": (
                    f"frames/frame_{i:03d}" if fmts else None
                ),
            }
            for i, t in enumerate(seq.frames)
        ],
    }
    manifest_path = os.path.join(out_dir, f"{bundle_name}.sequence.manifest.json")
    write_manifest(manifest_path, manifest)
    results["manifest"] = manifest_path

    return results


def _wavetable_audio_from_sequence(seq: TimbreSequence, table_size: int) -> np.ndarray:
    """Build a (n_frames * table_size,)-shape audio array where each slot is
    one Timbre's single-cycle waveform. The result is the canonical "morph
    wavetable" representation of the sequence."""
    cycles: list[np.ndarray] = []
    for timbre in seq.frames:
        cycles.append(render_wavetable_cycle(timbre, table_size=table_size))
    return np.concatenate(cycles).astype(np.float32, copy=False)
