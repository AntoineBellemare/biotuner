"""WAV sample-pack exporter.

Module type: Functions

Renders a Timbre at multiple reference MIDI pitches and writes:
    <out_dir>/<bundle_name>_samples/<midi_note>_<note_name>.wav
    <out_dir>/<bundle_name>.manifest.json

The manifest records full Timbre provenance plus a pitch -> file table
that downstream samplers (SFZ, Kontakt, soundfont generators) can use
without re-running the synthesis pipeline.
"""

from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import soundfile as sf

from biotuner.harmonic_timbre.cross_modal import write_sidecar
from biotuner.harmonic_timbre.exporters._common import (
    bundle_paths,
    midi_note_name,
    midi_to_hz,
    safe_filename,
    write_manifest,
)
from biotuner.harmonic_timbre.timbre import Timbre


_DEFAULT_PITCHES = (36, 48, 60, 72, 84)  # C2..C6


def export_wav_pack(
    timbre: Timbre,
    out_dir: str,
    *,
    bundle_name: str = "biotuner_timbre",
    reference_pitches_midi: Iterable[int] = _DEFAULT_PITCHES,
    duration: float = 4.0,
    samplerate: int = 48000,
    bit_depth: int = 24,
    a4: float = 440.0,
    include_sidecar: bool = True,
    progress_callback=None,
) -> dict:
    """Render the Timbre at each reference MIDI pitch and write a sample pack.

    Parameters
    ----------
    timbre : Timbre
        The source timbre. Each rendered sample uses the timbre's
        spectrum at the corresponding fundamental.
    out_dir : str
        Output directory (created if missing).
    bundle_name : str
        Stem used for the samples folder, manifest, and sidecar.
    reference_pitches_midi : iterable of int, default=(36,48,60,72,84)
        MIDI note numbers to render. The 12-TET frequency of each MIDI
        note (per ``a4``) is used as the rendered fundamental.
    duration : float, default=4.0
        Seconds per sample.
    samplerate : int, default=48000
    bit_depth : int, default=24
        ``16``, ``24``, or ``32``. Maps to soundfile subtypes
        ``PCM_16``/``PCM_24``/``FLOAT``.
    a4 : float, default=440.0
        Reference pitch in Hz for MIDI 69.
    include_sidecar : bool, default=True
        If True, writes a sidecar bundle (signature.png + metadata.json)
        next to the manifest.
    progress_callback : callable, optional
        Called once per rendered sample as ``cb(i, n, midi_note)``.

    Returns
    -------
    dict
        ``{
            'manifest': '<...>.manifest.json',
            'samples_dir': '<...>',
            'wavs': [list of paths],
            'sidecar': {...},   # only if include_sidecar
        }``
    """
    timbre.validate()

    pitches = list(reference_pitches_midi)
    if not pitches:
        raise ValueError("export_wav_pack: reference_pitches_midi is empty")

    paths = bundle_paths(out_dir, bundle_name)
    os.makedirs(paths["samples"], exist_ok=True)

    subtype = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}.get(bit_depth)
    if subtype is None:
        raise ValueError(f"export_wav_pack: unsupported bit_depth {bit_depth}")

    wav_paths: list[str] = []
    pitch_table: list[dict] = []
    n = len(pitches)
    for i, midi_note in enumerate(pitches):
        f0 = midi_to_hz(midi_note, a4=a4)
        audio = timbre.synthesize(
            samplerate=samplerate, duration=duration, base_freq=f0,
            normalize=True,
        )
        name = safe_filename(f"{midi_note:03d}_{midi_note_name(midi_note)}.wav")
        wav_path = os.path.join(paths["samples"], name)
        sf.write(wav_path, audio, samplerate, subtype=subtype)
        wav_paths.append(wav_path)
        pitch_table.append({
            "midi_note": int(midi_note),
            "note_name": midi_note_name(midi_note),
            "freq_hz": f0,
            "file": os.path.relpath(wav_path, out_dir).replace(os.sep, "/"),
        })
        if progress_callback is not None:
            progress_callback(i, n, midi_note)

    # Manifest
    manifest = {
        "format": "biotuner_wav_pack",
        "format_version": 1,
        "bundle_name": bundle_name,
        "samplerate": samplerate,
        "bit_depth": bit_depth,
        "duration_seconds": duration,
        "a4_hz": a4,
        "samples": pitch_table,
        "timbre": {
            "matched_tuning": list(timbre.matched_tuning) if timbre.matched_tuning is not None else None,
            "matching_method": timbre.matching_method,
            "n_partials": timbre.n_partials(),
            "base_freq": timbre.base_freq,
            "spectral_tilt": timbre.spectral_tilt,
            "noise_floor": timbre.noise_floor,
            "metadata": dict(timbre.metadata),
        },
    }
    manifest_path = write_manifest(paths["manifest"], manifest)

    result: dict = {
        "manifest": manifest_path,
        "samples_dir": paths["samples"],
        "wavs": wav_paths,
    }

    if include_sidecar:
        sidecar = write_sidecar(timbre, paths["sidecar"], stem=bundle_name)
        result["sidecar"] = sidecar

    return result
