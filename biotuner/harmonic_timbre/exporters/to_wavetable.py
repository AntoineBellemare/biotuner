"""Wavetable exporters for Vital, Serum, Surge XT, and generic synths.

Module type: Functions

A *wavetable* is one or more single-cycle waveforms (typically 2048 samples
each) stored back-to-back in a WAV file. Wavetable synths (Vital, Serum,
Surge XT, Ableton Wavetable, etc.) read the file as ``n_frames`` cycles
and let the user morph between them with a "table position" knob.

This module exposes three generators, each producing a wavetable WAV:

    export_wavetable(timbre, ..., evolution=...)
        Frames derived from a single Timbre. The ``evolution`` parameter
        controls how successive frames differ:
            'tilt'             — spectral_tilt sweeps from 0 (flat) to high (dark)
            'harmonic_buildup' — partials added one by one across frames
            'amp_morph'        — amplitudes morph from random → matched
            'phase_sweep'      — partial phases sweep around the unit circle

    export_wavetable_from_imfs(imfs, ...)
        One frame per intrinsic mode function (from EMD on a biosignal).
        Each IMF cycle becomes a wavetable frame. The synth then morphs
        through the biosignal's natural oscillatory modes.

    export_wavetable_morph(timbre_a, timbre_b, ...)
        Frames interpolate between two Timbres in log-frequency and
        amplitude space. Useful for "morph between tunings" wavetables —
        e.g. JI 5-limit → Bohlen-Pierce, octave → tritave.

All three call :func:`~biotuner.harmonic_timbre.cross_modal.write_sidecar`
when ``include_sidecar=True``, so every output bundle carries the same
provenance + visual fingerprint as the rest of the pipeline.
"""

from __future__ import annotations

import math
import os
from typing import Any, Iterable, Sequence

import numpy as np

# soundfile is a heavy/optional dependency (libsndfile binding). Import lazily
# so ``import biotuner`` succeeds in environments without libsndfile.
try:
    import soundfile as sf
except ImportError:  # pragma: no cover - environment-dependent
    sf = None


def _require_sf():
    if sf is None:
        raise ImportError(
            "Wavetable export requires the 'soundfile' package. "
            "Install with: pip install soundfile"
        )

from biotuner.harmonic_timbre.cross_modal import write_sidecar
from biotuner.harmonic_timbre.exporters._common import write_manifest
from biotuner.harmonic_timbre.synthesis import render_wavetable_cycle
from biotuner.harmonic_timbre.timbre import Timbre


_TARGET_PROFILES = {
    # synth_target -> (table_size, subtype)
    "vital":   (2048, "FLOAT"),
    "serum":   (2048, "PCM_16"),
    "surge":   (2048, "PCM_16"),
    "generic": (2048, "FLOAT"),
}

_EVOLUTIONS = (
    "tilt", "harmonic_buildup", "amp_morph", "phase_sweep",
    # Spectral enrichment buildups (use Timbre transforms under the hood):
    "intermod_buildup", "harmonic_stack", "formant_sweep",
)


# ---------------------------------------------------------------------------
# Internal helpers — per-evolution frame generators
# ---------------------------------------------------------------------------

def _frame_with_tilt(timbre: Timbre, tilt: float, *, table_size: int) -> np.ndarray:
    return render_wavetable_cycle(
        timbre.with_partials(spectral_tilt=tilt),
        table_size=table_size,
    )


def _frame_with_active_partials(
    timbre: Timbre, n_active: int, *, table_size: int
) -> np.ndarray:
    """Frame with only the first ``n_active`` partials at full amplitude."""
    n = timbre.n_partials()
    n_active = max(1, min(n, n_active))
    mask = np.zeros(n, dtype=np.float64)
    mask[:n_active] = 1.0
    new_amps = timbre.amplitudes * mask
    return render_wavetable_cycle(
        timbre.with_partials(amplitudes=new_amps),
        table_size=table_size,
    )


def _frame_with_amp_morph(
    timbre: Timbre, alpha: float, *, table_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Interpolate amps between a random pattern (alpha=0) and the
    matched pattern (alpha=1)."""
    n = timbre.n_partials()
    random_amps = rng.uniform(0.0, 1.0, n)
    morphed = (1.0 - alpha) * random_amps + alpha * timbre.amplitudes
    morphed = morphed / max(np.max(np.abs(morphed)), 1e-9)
    return render_wavetable_cycle(
        timbre.with_partials(amplitudes=morphed),
        table_size=table_size,
    )


def _frame_with_phase_sweep(
    timbre: Timbre, phase_offset: float, *, table_size: int
) -> np.ndarray:
    """Frame whose partial phases are sweep-offset by an arithmetic progression.

    Each partial *k* gets phase ``k * phase_offset`` (mod 2π); as
    ``phase_offset`` advances across frames the partials' phase relations
    rotate, producing a comb-filter-like timbral evolution.
    """
    n = timbre.n_partials()
    phases = (np.arange(1, n + 1) * phase_offset) % (2.0 * np.pi)
    return render_wavetable_cycle(
        timbre.with_partials(phases=phases),
        table_size=table_size,
    )


def _frame_with_intermod_sidebands(
    timbre: Timbre, bt, depth: float, *, table_size: int
) -> np.ndarray:
    """Frame whose intermodulation sidebands ramp up to ``depth``."""
    if depth <= 0 or bt is None:
        return render_wavetable_cycle(timbre, table_size=table_size)
    enriched = timbre.with_intermod_sidebands(bt, depth=float(depth))
    return render_wavetable_cycle(enriched, table_size=table_size)


def _frame_with_harmonic_stack(
    timbre: Timbre, n_overtones: int, rolloff: float, *, table_size: int
) -> np.ndarray:
    """Frame with ``n_overtones`` harmonic overtones stacked on each partial."""
    if n_overtones <= 0:
        return render_wavetable_cycle(timbre, table_size=table_size)
    enriched = timbre.with_harmonic_stack(n=int(n_overtones), rolloff=float(rolloff))
    return render_wavetable_cycle(enriched, table_size=table_size)


def _frame_with_formant(
    timbre: Timbre, center_hz: float, width_hz: float, gain_db: float,
    *, table_size: int,
) -> np.ndarray:
    """Frame with a formant bump at ``center_hz``."""
    enriched = timbre.with_formant(
        center_hz=float(center_hz), width_hz=float(width_hz), gain_db=float(gain_db),
    )
    return render_wavetable_cycle(enriched, table_size=table_size)


def _normalize_cycle(buf: np.ndarray, peak: float = 0.99) -> np.ndarray:
    m = float(np.max(np.abs(buf))) if buf.size else 0.0
    if m <= 0:
        return buf
    return buf * (peak / m)


def _resample_to_table(cycle: np.ndarray, table_size: int) -> np.ndarray:
    """Linearly resample ``cycle`` to ``table_size`` points."""
    cycle = np.asarray(cycle, dtype=np.float64)
    if cycle.size == 0:
        return np.zeros(table_size, dtype=np.float32)
    if cycle.size == table_size:
        return cycle.astype(np.float32, copy=False)
    src_x = np.linspace(0.0, 1.0, cycle.size, endpoint=False)
    dst_x = np.linspace(0.0, 1.0, table_size, endpoint=False)
    return np.interp(dst_x, src_x, cycle).astype(np.float32, copy=False)


def _resolve_target(synth_target: str, table_size: int | None):
    if synth_target not in _TARGET_PROFILES:
        raise ValueError(
            f"unknown synth_target {synth_target!r}. Known: {sorted(_TARGET_PROFILES)}"
        )
    default_size, subtype = _TARGET_PROFILES[synth_target]
    return (table_size or default_size), subtype


def _ensure_wav_path(out_path: str) -> str:
    if not out_path.endswith(".wav"):
        out_path = out_path + ".wav"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    return out_path


# ---------------------------------------------------------------------------
# export_wavetable (extended with evolution modes)
# ---------------------------------------------------------------------------

def export_wavetable(
    timbre: Timbre,
    out_path: str,
    *,
    n_frames: int = 1,
    synth_target: str = "vital",
    evolution: str = "tilt",
    tilt_range: tuple[float, float] = (0.0, 2.5),
    phase_range: tuple[float, float] = (0.0, 2.0 * math.pi),
    table_size: int | None = None,
    seed: int = 0,
    include_sidecar: bool = True,
    # Spectral enrichment params (used by the new evolution modes):
    bt: Any = None,
    intermod_depth_range: tuple[float, float] = (0.0, 0.6),
    harmonic_stack_range: tuple[int, int] = (0, 4),
    harmonic_stack_rolloff: float = 0.9,
    formant_center_range: tuple[float, float] = (1000.0, 3000.0),
    formant_width_hz: float = 800.0,
    formant_gain_db: float = 4.0,
) -> dict:
    """Write a single- or multi-frame wavetable WAV from a single Timbre.

    Parameters
    ----------
    timbre : Timbre
    out_path : str
        Output ``.wav`` path. ``.wav`` is appended if missing.
    n_frames : int, default=1
        Number of wavetable frames. ``1`` = single-cycle WAV.
    synth_target : str, default='vital'
        ``'vital'``, ``'serum'``, ``'surge'``, or ``'generic'``. Selects
        table size + bit depth conventions.
    evolution : str, default='tilt'
        How frames differ when ``n_frames > 1``. Ignored when ``n_frames == 1``.

        * ``'tilt'``             — spectral_tilt sweeps ``tilt_range[0] → [1]``.
        * ``'harmonic_buildup'`` — partial 1 only in frame 0; all partials
          in frame N-1.
        * ``'amp_morph'``        — amplitudes morph from random (frame 0) to
          the timbre's matched amplitudes (frame N-1).
        * ``'phase_sweep'``      — partial phases offset by
          ``phase_range[0] → [1]`` across frames.
        * ``'intermod_buildup'`` — intermodulation sidebands ``f1 ± f2``
          ramp in across frames. Requires ``bt=`` (a biotuner-like object
          exposing ``endogenous_intermodulations``).
        * ``'harmonic_stack'``   — harmonic overtones (``2f, 3f, …``) of
          each partial fade in across frames; rolloff controlled by
          ``harmonic_stack_rolloff`` (sweet spot 0.7-1.2).
        * ``'formant_sweep'``    — multiplicative formant bump sweeps
          ``formant_center_range[0] → [1]`` Hz (vowel-like "ah → ee").
    tilt_range : (float, float), default=(0.0, 2.5)
        Used when ``evolution='tilt'``.
    phase_range : (float, float), default=(0, 2π)
        Used when ``evolution='phase_sweep'``.
    table_size : int, optional
        Per-frame samples. Default is target-specific (2048).
    seed : int, default=0
        RNG seed for ``evolution='amp_morph'``.
    bt : object, optional
        biotuner-like object with ``endogenous_intermodulations``.
        Required for ``evolution='intermod_buildup'``.
    intermod_depth_range : (float, float), default=(0.0, 0.6)
        Sideband depth ramp for ``'intermod_buildup'``.
    harmonic_stack_range : (int, int), default=(0, 4)
        Number of overtones added per source partial across frames.
    harmonic_stack_rolloff : float, default=0.9
        Amplitude rolloff exponent for stacked harmonics.
    formant_center_range : (float, float), default=(1000.0, 3000.0)
        Hz range for the formant center sweep.
    formant_width_hz : float, default=800.0
        Gaussian standard deviation of the formant bump.
    formant_gain_db : float, default=4.0
        Peak gain at the formant center.

    Returns
    -------
    dict
        ``{'wavetable': <path>, 'manifest': <path>, 'sidecar': {...}}``.
    """
    timbre.validate()
    if n_frames < 1:
        raise ValueError("export_wavetable: n_frames must be ≥ 1")
    if n_frames > 1 and evolution not in _EVOLUTIONS:
        raise ValueError(
            f"export_wavetable: unknown evolution {evolution!r}. "
            f"Known: {sorted(_EVOLUTIONS)}"
        )

    table_size, subtype = _resolve_target(synth_target, table_size)
    out_path = _ensure_wav_path(out_path)

    if n_frames == 1:
        full = render_wavetable_cycle(timbre, table_size=table_size)
    elif evolution == "tilt":
        tilts = np.linspace(tilt_range[0], tilt_range[1], n_frames)
        frames = [_frame_with_tilt(timbre, float(t), table_size=table_size) for t in tilts]
        full = np.concatenate(frames).astype(np.float32, copy=False)
    elif evolution == "harmonic_buildup":
        n = timbre.n_partials()
        active_per_frame = np.linspace(1, n, n_frames).astype(int)
        frames = [_frame_with_active_partials(timbre, int(k), table_size=table_size) for k in active_per_frame]
        full = np.concatenate(frames).astype(np.float32, copy=False)
    elif evolution == "amp_morph":
        rng = np.random.default_rng(seed)
        alphas = np.linspace(0.0, 1.0, n_frames)
        frames = [_frame_with_amp_morph(timbre, float(a), table_size=table_size, rng=rng) for a in alphas]
        full = np.concatenate(frames).astype(np.float32, copy=False)
    elif evolution == "phase_sweep":
        offsets = np.linspace(phase_range[0], phase_range[1], n_frames)
        frames = [_frame_with_phase_sweep(timbre, float(p), table_size=table_size) for p in offsets]
        full = np.concatenate(frames).astype(np.float32, copy=False)
    elif evolution == "intermod_buildup":
        if bt is None:
            raise ValueError(
                "evolution='intermod_buildup' requires bt= (a biotuner-like "
                "object exposing endogenous_intermodulations)"
            )
        depths = np.linspace(intermod_depth_range[0], intermod_depth_range[1], n_frames)
        frames = [
            _frame_with_intermod_sidebands(timbre, bt, float(d), table_size=table_size)
            for d in depths
        ]
        full = np.concatenate(frames).astype(np.float32, copy=False)
    elif evolution == "harmonic_stack":
        n_overtones = np.linspace(
            harmonic_stack_range[0], harmonic_stack_range[1], n_frames
        ).round().astype(int)
        frames = [
            _frame_with_harmonic_stack(
                timbre, int(k), harmonic_stack_rolloff, table_size=table_size,
            )
            for k in n_overtones
        ]
        full = np.concatenate(frames).astype(np.float32, copy=False)
    elif evolution == "formant_sweep":
        centers = np.linspace(formant_center_range[0], formant_center_range[1], n_frames)
        frames = [
            _frame_with_formant(
                timbre, float(c), formant_width_hz, formant_gain_db,
                table_size=table_size,
            )
            for c in centers
        ]
        full = np.concatenate(frames).astype(np.float32, copy=False)

    sr = 48000
    _require_sf()
    sf.write(out_path, full, sr, subtype=subtype)

    manifest = {
        "format": "biotuner_wavetable",
        "format_version": 2,
        "source": "single_timbre",
        "synth_target": synth_target,
        "table_size": int(table_size),
        "n_frames": int(n_frames),
        "evolution": evolution if n_frames > 1 else "single_cycle",
        "evolution_params": _evolution_params(
            evolution, n_frames, tilt_range, phase_range, seed,
            intermod_depth_range=intermod_depth_range,
            harmonic_stack_range=harmonic_stack_range,
            harmonic_stack_rolloff=harmonic_stack_rolloff,
            formant_center_range=formant_center_range,
            formant_width_hz=formant_width_hz,
            formant_gain_db=formant_gain_db,
        ),
        "subtype": subtype,
        "samplerate": sr,
        "timbre": _summarize_timbre(timbre),
    }
    manifest_path = write_manifest(out_path.replace(".wav", ".manifest.json"), manifest)

    result = {"wavetable": out_path, "manifest": manifest_path}
    if include_sidecar:
        sidecar_dir = out_path.replace(".wav", "_sidecar")
        sidecar = write_sidecar(timbre, sidecar_dir, stem=os.path.basename(out_path).replace(".wav", ""))
        result["sidecar"] = sidecar
    return result


def _evolution_params(
    evolution, n_frames, tilt_range, phase_range, seed,
    *,
    intermod_depth_range=None,
    harmonic_stack_range=None,
    harmonic_stack_rolloff=None,
    formant_center_range=None,
    formant_width_hz=None,
    formant_gain_db=None,
):
    if n_frames == 1:
        return {}
    if evolution == "tilt":
        return {"tilt_range": list(tilt_range)}
    if evolution == "harmonic_buildup":
        return {}
    if evolution == "amp_morph":
        return {"seed": seed}
    if evolution == "phase_sweep":
        return {"phase_range": list(phase_range)}
    if evolution == "intermod_buildup":
        return {"intermod_depth_range": list(intermod_depth_range or (0.0, 0.6))}
    if evolution == "harmonic_stack":
        return {
            "harmonic_stack_range": list(harmonic_stack_range or (0, 4)),
            "rolloff": float(harmonic_stack_rolloff if harmonic_stack_rolloff is not None else 0.9),
        }
    if evolution == "formant_sweep":
        return {
            "formant_center_range": list(formant_center_range or (1000.0, 3000.0)),
            "width_hz": float(formant_width_hz if formant_width_hz is not None else 800.0),
            "gain_db": float(formant_gain_db if formant_gain_db is not None else 4.0),
        }
    return {}


def _summarize_timbre(timbre: Timbre) -> dict:
    return {
        "matched_tuning": list(timbre.matched_tuning) if timbre.matched_tuning is not None else None,
        "matching_method": timbre.matching_method,
        "n_partials": timbre.n_partials(),
        "base_freq": timbre.base_freq,
        "metadata": dict(timbre.metadata),
    }


# ---------------------------------------------------------------------------
# export_wavetable_from_imfs
# ---------------------------------------------------------------------------

def export_wavetable_from_imfs(
    imfs: Sequence[np.ndarray],
    out_path: str,
    *,
    synth_target: str = "vital",
    table_size: int | None = None,
    cycle_strategy: str = "first_cycle",
    n_avg_cycles: int = 4,
    include_sidecar_for: Timbre | None = None,
) -> dict:
    """Build a wavetable from EMD intrinsic mode functions.

    Each IMF is a near-monochromatic oscillatory mode of the source
    biosignal (Empirical Mode Decomposition). One IMF becomes one
    wavetable frame; the synth then morphs across the biosignal's
    natural modes when the user sweeps the table-position knob.

    This is the time-scale projection principle in disguise: the slowest
    IMF is a low-frequency rhythm component; the fastest IMF is a
    high-frequency carrier. Stuffed into a wavetable, the table sweeps
    through them as if they were timbre.

    Parameters
    ----------
    imfs : sequence of 1D arrays
        IMFs as returned by :func:`biotuner.peaks_extraction.EMD_eeg`.
        The first IMF is conventionally the highest-frequency mode.
        Frames are written in input order.
    out_path : str
    synth_target : str, default='vital'
    table_size : int, optional
        Per-frame samples. Default is 2048 (Vital/Serum/Surge convention).
    cycle_strategy : str, default='first_cycle'
        How to extract a single cycle from each IMF:

        * ``'first_cycle'`` — find the first zero-crossing-to-zero-crossing
          pair after one full oscillation; resample to ``table_size``.
        * ``'avg_cycles'`` — segment the IMF into ``n_avg_cycles`` cycles
          and average them; resample.
        * ``'whole_resampled'`` — resample the whole IMF to ``table_size``
          (treats the entire IMF as the cycle).
    n_avg_cycles : int, default=4
        Used when ``cycle_strategy='avg_cycles'``.
    include_sidecar_for : Timbre, optional
        If provided, write_sidecar is called using this Timbre as the
        provenance source. Useful when the IMFs were extracted from a
        biosignal that also produced a tuned Timbre.

    Returns
    -------
    dict
        ``{'wavetable': <path>, 'manifest': <path>, 'n_frames': N, ...}``.
    """
    imf_list = [np.asarray(x, dtype=np.float64).flatten() for x in imfs]
    if not imf_list:
        raise ValueError("export_wavetable_from_imfs: empty IMFs")
    table_size, subtype = _resolve_target(synth_target, table_size)
    out_path = _ensure_wav_path(out_path)

    frames: list[np.ndarray] = []
    for imf in imf_list:
        if imf.size == 0:
            frames.append(np.zeros(table_size, dtype=np.float32))
            continue
        cycle = _extract_imf_cycle(imf, cycle_strategy, n_avg_cycles)
        cycle = _resample_to_table(cycle, table_size)
        cycle = _normalize_cycle(cycle)
        frames.append(cycle.astype(np.float32, copy=False))

    full = np.concatenate(frames)
    sr = 48000
    _require_sf()
    sf.write(out_path, full, sr, subtype=subtype)

    manifest = {
        "format": "biotuner_wavetable",
        "format_version": 2,
        "source": "imfs",
        "synth_target": synth_target,
        "table_size": int(table_size),
        "n_frames": len(frames),
        "cycle_strategy": cycle_strategy,
        "n_avg_cycles": int(n_avg_cycles) if cycle_strategy == "avg_cycles" else None,
        "subtype": subtype,
        "samplerate": sr,
        "imf_lengths": [int(x.size) for x in imf_list],
    }
    manifest_path = write_manifest(out_path.replace(".wav", ".manifest.json"), manifest)

    result = {
        "wavetable": out_path,
        "manifest": manifest_path,
        "n_frames": len(frames),
    }
    if include_sidecar_for is not None:
        sidecar_dir = out_path.replace(".wav", "_sidecar")
        sidecar = write_sidecar(
            include_sidecar_for, sidecar_dir,
            stem=os.path.basename(out_path).replace(".wav", ""),
        )
        result["sidecar"] = sidecar
    return result


def _extract_imf_cycle(imf: np.ndarray, strategy: str, n_avg: int) -> np.ndarray:
    """Extract one representative cycle from an IMF."""
    if strategy == "whole_resampled":
        return imf

    # Find zero crossings (positive-going only)
    sign = np.sign(imf)
    crossings = np.where(np.diff(sign) > 0)[0]

    if crossings.size < 2:
        # Fallback: not enough crossings — resample the whole thing
        return imf

    if strategy == "first_cycle":
        a, b = int(crossings[0]), int(crossings[1])
        return imf[a:b] if b > a else imf

    if strategy == "avg_cycles":
        n_avail = crossings.size - 1
        n_use = min(int(n_avg), n_avail)
        cycles: list[np.ndarray] = []
        target_len = None
        for i in range(n_use):
            a, b = int(crossings[i]), int(crossings[i + 1])
            if b > a:
                seg = imf[a:b]
                cycles.append(seg)
                target_len = target_len or seg.size
        if not cycles:
            return imf
        # resample each to a common length and average
        common = max(target_len, 64)
        stack = np.stack([_resample_to_table(c, common) for c in cycles])
        return stack.mean(axis=0)

    raise ValueError(f"_extract_imf_cycle: unknown strategy {strategy!r}")


# ---------------------------------------------------------------------------
# export_wavetable_morph
# ---------------------------------------------------------------------------

def export_wavetable_morph(
    timbre_a: Timbre,
    timbre_b: Timbre,
    out_path: str,
    *,
    n_frames: int = 64,
    synth_target: str = "vital",
    table_size: int | None = None,
    include_sidecar: bool = True,
) -> dict:
    """Render a wavetable that morphs between two timbres across frames.

    Both timbres must have the same ``n_partials`` (extra partials are
    padded with zero amplitude). The morph is linear in
    ``log2(partials_hz)`` and in amplitudes, so even tunings with very
    different equaves (octave → tritave) interpolate smoothly.

    Parameters
    ----------
    timbre_a, timbre_b : Timbre
    out_path : str
    n_frames : int, default=64
        Frame count. ``frame[0] == timbre_a``, ``frame[-1] == timbre_b``.

    Returns
    -------
    dict
    """
    timbre_a.validate()
    timbre_b.validate()
    if n_frames < 2:
        raise ValueError("export_wavetable_morph: n_frames must be ≥ 2")
    table_size, subtype = _resolve_target(synth_target, table_size)
    out_path = _ensure_wav_path(out_path)

    n = max(timbre_a.n_partials(), timbre_b.n_partials())
    pa = _pad_partials(timbre_a, n)
    pb = _pad_partials(timbre_b, n)
    aa = _pad_amps(timbre_a, n)
    ab = _pad_amps(timbre_b, n)

    # log-frequency interpolation (handles non-octave equaves cleanly)
    log_pa = np.log2(np.maximum(pa, 1e-9))
    log_pb = np.log2(np.maximum(pb, 1e-9))

    base = float(timbre_a.base_freq) if timbre_a.base_freq > 0 else 1.0

    frames: list[np.ndarray] = []
    for i in range(n_frames):
        alpha = i / (n_frames - 1)
        log_p = (1.0 - alpha) * log_pa + alpha * log_pb
        partials = np.power(2.0, log_p)
        amps = (1.0 - alpha) * aa + alpha * ab
        amps = amps / max(np.max(np.abs(amps)), 1e-9)
        frame_t = Timbre(
            partials_hz=partials,
            amplitudes=amps,
            base_freq=base,
            matching_method="morph",
        )
        frames.append(render_wavetable_cycle(frame_t, table_size=table_size))

    full = np.concatenate(frames).astype(np.float32, copy=False)
    sr = 48000
    _require_sf()
    sf.write(out_path, full, sr, subtype=subtype)

    manifest = {
        "format": "biotuner_wavetable",
        "format_version": 2,
        "source": "morph",
        "synth_target": synth_target,
        "table_size": int(table_size),
        "n_frames": int(n_frames),
        "subtype": subtype,
        "samplerate": sr,
        "timbre_a": _summarize_timbre(timbre_a),
        "timbre_b": _summarize_timbre(timbre_b),
    }
    manifest_path = write_manifest(out_path.replace(".wav", ".manifest.json"), manifest)

    result = {"wavetable": out_path, "manifest": manifest_path, "n_frames": int(n_frames)}
    if include_sidecar:
        sidecar_dir = out_path.replace(".wav", "_sidecar")
        sidecar = write_sidecar(timbre_a, sidecar_dir, stem=os.path.basename(out_path).replace(".wav", ""))
        result["sidecar"] = sidecar
    return result


def _pad_partials(timbre: Timbre, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float64)
    p = np.asarray(timbre.partials_hz, dtype=np.float64)
    out[: p.size] = p
    if p.size < n:
        # repeat last partial so log-interpolation doesn't blow up
        out[p.size :] = p[-1] if p.size else 1.0
    return out


def _pad_amps(timbre: Timbre, n: int) -> np.ndarray:
    out = np.zeros(n, dtype=np.float64)
    a = np.asarray(timbre.amplitudes, dtype=np.float64)
    out[: a.size] = a
    return out
