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
from dataclasses import dataclass, field
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
    # Nonlinear enrichments — introduce new partials via maths the additive
    # synth can't produce on its own. Bake the result into the per-frame
    # cycle so it survives export to wavetable-target formats.
    "wavefolding", "fm_baked",
    # Composite — combine 2+ of the above with per-axis weight curves.
    "composite",
    # Biosignal-structure evolutions — exploit aspects of the bt that
    # synthetic sources can't access.
    "noise_to_structure",
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


def _frame_with_wavefolding(
    timbre: Timbre, fold_amount: float, *, table_size: int,
    output_drive: float = 1.0,
) -> np.ndarray:
    """Apply sin-based wavefolding to the rendered cycle.

    Mathematical model (Buchla / Make Noise style)::

        y = sin( π · output_drive · base(t) · (1 + fold_amount) / 2 )

    The ``/ 2`` is the bit that makes ``fold_amount = 0`` behave like
    a near-identity transform: at fold=0 the argument range is
    ``[-π/2, +π/2]`` over the cycle's ``[-1, +1]`` range, so sin
    monotonically maps both endpoints to themselves and the curve
    sits close to ``y = x`` with a mild saturating S-shape. Rising
    fold_amount widens the argument range past ``±π/2`` and the wave
    starts folding back on itself, adding primarily **odd-order
    harmonics** with predictable amplitude rolloff.

    This is the most musical of the cheap nonlinearities (vs tanh,
    clipper, cubic) because the energy lands in well-defined harmonic
    positions rather than smearing into broadband distortion.

    Parameters
    ----------
    fold_amount : float
        ``0.0`` → near-unfolded (mild S-curve saturation, very close
        to the base cycle). ``~3.0`` → heavy fold with dense odd-
        harmonic enrichment. Beyond ``~4.0`` the spectrum spills
        above Nyquist; callers are expected to cap accordingly.
    output_drive : float, default=1.0
        Pre-fold gain. Stay in ``[0.7, 1.3]``; outside that range
        the fold becomes either inaudible or alias-dominated.

    Returns
    -------
    ndarray of float32, length ``table_size``
        The folded cycle, peak-normalised to ±1.
    """
    base = render_wavetable_cycle(timbre, table_size=table_size)
    pre  = base * float(output_drive)
    folded = np.sin(np.pi * pre * (1.0 + float(fold_amount)) / 2.0)
    peak = float(np.max(np.abs(folded))) or 1.0
    return (folded / peak).astype(np.float32, copy=False)


def _frame_with_fm_baked(
    timbre: Timbre, fm_index: float, *, table_size: int,
    cm_ratio: float = 2.0, target_partial_idx: int = 0,
) -> np.ndarray:
    """Render the cycle with audio-rate FM baked into the partial set.

    Different from ``Timbre.fm_modulators``: those apply at synthesis
    time (a separate LFO modulating the partial frequency). This bakes
    FM **into the single-cycle wavetable** so an exported .vital /
    .surge / .serum file carries the FM character without needing the
    host synth to recreate the modulation.

    Math: for each partial ``f`` at amplitude ``a`` selected by
    ``target_partial_idx``, synthesise::

        a · sin(2π · f · t + fm_index · sin(2π · (f · cm_ratio) · t))

    Non-target partials render normally (additive sines). The result
    is a single cycle of the fundamental whose spectrum contains the
    FM Bessel sidebands ``f ± k·(f·cm_ratio)``.

    Parameters
    ----------
    fm_index : float
        FM modulation index β. ``0`` → identical to additive render;
        ``≈ 1`` produces vibrato-rich sidebands; ``≈ 3`` produces full
        FM-EP / bell character; ``> 5`` starts aliasing badly at
        moderate table sizes.
    cm_ratio : float, default=2.0
        Carrier-to-modulator ratio. ``2.0`` = octave-mod = classic
        bell character; integer ratios stay periodic and "musical";
        non-integer ratios produce metallic / clangorous textures.
    target_partial_idx : int, default=0
        Index of the partial to FM. ``-1`` applies FM to every partial
        (heavier sound, denser sidebands per partial).

    Returns
    -------
    ndarray of float32, length ``table_size``
        The FM-bearing cycle, peak-normalised to ±1.
    """
    timbre.validate()
    if timbre.base_freq <= 0:
        raise ValueError("_frame_with_fm_baked: timbre.base_freq must be > 0")
    # Match render_wavetable_cycle's semantics: partials are interpreted
    # as harmonic indices (partial_hz / base_freq) and the table holds
    # one cycle of the fundamental sampled at table_size points.
    partials = np.asarray(timbre.partials_hz, dtype=np.float64) / float(timbre.base_freq)
    amps = np.asarray(timbre.amplitudes, dtype=np.float64)
    phases = (
        np.asarray(timbre.phases, dtype=np.float64)
        if timbre.phases is not None
        else np.zeros(partials.shape[0])
    )
    idx = np.arange(table_size, dtype=np.float64)
    theta = 2.0 * np.pi * idx / float(table_size)   # one cycle of fundamental

    fm_set = set(range(partials.size)) if target_partial_idx < 0 else {
        int(target_partial_idx) % max(1, partials.size)
    }
    beta = float(fm_index)
    out = np.zeros(table_size, dtype=np.float64)
    for k in range(partials.size):
        n = float(partials[k]); a = float(amps[k]); ph = float(phases[k])
        if k in fm_set and beta > 0:
            # Modulator at carrier × cm_ratio in the same normalised
            # harmonic space — produces FM Bessel sidebands at
            # n ± m·(n·cm_ratio) in the FFT.
            n_mod = n * float(cm_ratio)
            modulator = beta * np.sin(n_mod * theta)
            out += a * np.sin(n * theta + ph + modulator)
        else:
            out += a * np.sin(n * theta + ph)

    # Match render_wavetable_cycle's normalisation: scale so peak = 0.99
    # (the existing _normalize helper uses 0.99, not 1.0, to leave a
    # tiny bit of headroom for downstream DAW input gain). We replicate
    # the constant here to keep _frame_with_fm_baked self-contained.
    peak = float(np.max(np.abs(out))) or 1.0
    return (out * (0.99 / peak)).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Biosignal-structure evolutions — leverage what's UNIQUELY biosignal
# (vs. synthetic) about the timbre's source data.
# ---------------------------------------------------------------------------


def _frame_with_noise_to_structure(
    timbre: Timbre,
    alpha: float,
    *,
    table_size: int,
    exponent: float | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Cross-fade a 1/f^k noise cycle into the clean Timbre cycle.

    The "brain noise → brain structure" mode. Mirrors the FOOOF
    decomposition philosophy: the spectrum of a biosignal splits into
    an aperiodic (1/f^k) component plus periodic peaks on top. This
    mode makes that decomposition AUDIBLE by sweeping from "pure
    aperiodic" → "pure periodic" across wavetable frames.

    Frame 0 (alpha=0) is a single-period waveform whose magnitude
    spectrum is ``bin^(-k/2)`` and whose phases are random. Because
    the spectrum is defined on integer FFT bins, the resulting cycle
    LOOPS cleanly when played as a wavetable — it's "loopable
    coloured noise" rather than a random buffer slice.

    Frame N (alpha=1) is the standard additive render of the Timbre.

    Intermediates are linear blends in waveform domain.

    Parameters
    ----------
    alpha : float in [0, 1]
        0.0 = pure noise; 1.0 = pure structured timbre.
    exponent : float, optional
        Power-law slope (k) of the noise's PSD shape. When None, uses
        ``timbre.spectral_tilt`` if set (that's the bt's measured
        aperiodic_exponent), else defaults to 1.0 (pink noise). Larger
        k = darker / brown-noise-ish; smaller = brighter / white-ish.
    seed : int, default=0
        Deterministic phase randomisation. Same seed → same noise
        cycle, so the morph is reproducible across re-exports.
    """
    if exponent is None:
        exponent = (
            float(timbre.spectral_tilt) if timbre.spectral_tilt is not None else 1.0
        )
    alpha = float(np.clip(alpha, 0.0, 1.0))

    # Loopable 1/f^k noise cycle: random phases per harmonic bin,
    # magnitudes following bin^(-k/2). DC bin set to 0 to avoid
    # offset; phases at bin 0 don't matter.
    rng = np.random.default_rng(int(seed))
    n_bins = table_size // 2 + 1
    bins = np.arange(n_bins, dtype=np.float64)
    safe_bins = np.where(bins == 0, 1.0, bins)
    mags = safe_bins ** (-float(exponent) / 2.0)
    mags[0] = 0.0
    phases = rng.uniform(0.0, 2.0 * np.pi, n_bins)
    spec = mags * np.exp(1j * phases)
    noise_cycle = np.fft.irfft(spec, n=table_size)
    npk = float(np.max(np.abs(noise_cycle))) or 1.0
    noise_cycle = noise_cycle / npk

    # Structured cycle from the timbre.
    structured = render_wavetable_cycle(timbre, table_size=table_size)
    spk = float(np.max(np.abs(structured))) or 1.0
    structured = structured.astype(np.float64) / spk

    out = (1.0 - alpha) * noise_cycle + alpha * structured
    peak = float(np.max(np.abs(out))) or 1.0
    return (out * (0.99 / peak)).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Composite evolution — chain multiple effects with per-axis weight curves
# ---------------------------------------------------------------------------

_COMPOSITE_WEIGHT_CURVES = ("linear", "ease_in", "ease_out", "sine", "constant")
_COMPOSITE_ALLOWED = tuple(e for e in _EVOLUTIONS if e != "composite")


@dataclass
class WavetableLayer:
    """One axis of a composite wavetable evolution.

    A layer specifies (a) which single-axis evolution to apply, (b)
    the weight range that evolution sweeps across the wavetable, and
    (c) the curve shape the weight follows (linear, eased, etc.).
    Composite mode evaluates every layer's weight at each frame and
    applies the layers **in order** — spectral edits accumulate into
    a running Timbre; waveform edits apply to the rendered cycle;
    fm_baked terminates the spectral chain with a fresh FM-rendered
    cycle. See :func:`_frame_composite` for the dispatch table.

    Parameters
    ----------
    evolution : str
        One of :data:`_COMPOSITE_ALLOWED`. ``"composite"`` itself is
        not allowed (no recursion).
    weight_curve : str, default="linear"
        Shape of the weight as ``frame_idx`` goes 0 → N-1. Options:
        ``"linear"``, ``"ease_in"`` (squared), ``"ease_out"`` (1−(1−t)²),
        ``"sine"`` (smooth half-cycle), ``"constant"`` (always the
        midpoint of the range).
    weight_min, weight_max : float
        Start and end of the swept range. Semantic meaning depends on
        the evolution (tilt exponent, fold amount, formant Hz, etc.);
        see the per-mode docstrings.
    params : dict
        Extra mode-specific knobs the swept weight doesn't cover —
        e.g. ``{"width_hz": 800, "gain_db": 4}`` for formant_sweep,
        ``{"output_drive": 1.0}`` for wavefolding, ``{"cm_ratio": 2.0,
        "target_partial_idx": 0}`` for fm_baked.
    """

    evolution: str
    weight_curve: str = "linear"
    weight_min: float = 0.0
    weight_max: float = 1.0
    params: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.evolution == "composite":
            raise ValueError(
                "WavetableLayer.evolution='composite' would recurse; "
                f"pick one of {_COMPOSITE_ALLOWED}"
            )
        if self.evolution not in _COMPOSITE_ALLOWED:
            raise ValueError(
                f"WavetableLayer.evolution={self.evolution!r} not in "
                f"{_COMPOSITE_ALLOWED}"
            )
        if self.weight_curve not in _COMPOSITE_WEIGHT_CURVES:
            raise ValueError(
                f"WavetableLayer.weight_curve={self.weight_curve!r} not in "
                f"{_COMPOSITE_WEIGHT_CURVES}"
            )

    def weight_at(self, frame_idx: int, n_frames: int) -> float:
        """Evaluate the layer's weight at the given frame index.

        ``frame_idx`` ∈ [0, n_frames-1]. Returns a float in
        ``[weight_min, weight_max]`` after curve shaping.
        """
        if n_frames <= 1:
            t = 1.0
        else:
            t = float(frame_idx) / float(n_frames - 1)
        if self.weight_curve == "constant":
            t = 0.5
        elif self.weight_curve == "ease_in":
            t = t * t
        elif self.weight_curve == "ease_out":
            t = 1.0 - (1.0 - t) ** 2
        elif self.weight_curve == "sine":
            t = 0.5 - 0.5 * math.cos(math.pi * t)
        # 'linear' → unchanged.
        return float(self.weight_min) + t * float(self.weight_max - self.weight_min)


def _frame_composite(
    base_timbre: Timbre,
    layers: Sequence[WavetableLayer],
    frame_idx: int,
    n_frames: int,
    *,
    table_size: int,
    bt: Any = None,
    seed: int = 0,
) -> np.ndarray:
    """Render one wavetable frame by applying multiple layers in order.

    Three categories of layer:

    * **Spectral** (tilt, harmonic_buildup, intermod_buildup,
      harmonic_stack, formant_sweep, phase_sweep, amp_morph) — these
      mutate a running Timbre; their changes accumulate.
    * **Waveform** (wavefolding) — applied to the rendered cycle;
      requires a render before they can run.
    * **Terminal-render** (fm_baked) — replaces the rendered cycle
      with a fresh FM-injected render from the current Timbre.

    Layers run in the user-supplied order. The recommended pipeline is
    ``[spectral enrichments] → [shape] → [nonlinear post]`` but the
    order is not enforced — composing in different orders is the whole
    point of having a composite mode.

    Returns the rendered cycle as float32, length ``table_size``,
    peak-normalised to 0.99 (matching the convention used by every
    other helper in this module).
    """
    timbre = base_timbre
    cycle: np.ndarray | None = None
    rng = np.random.default_rng(int(seed))

    for layer in layers:
        w = layer.weight_at(frame_idx, n_frames)
        ev = layer.evolution

        if ev == "tilt":
            timbre = timbre.with_partials(spectral_tilt=float(w))

        elif ev == "harmonic_buildup":
            n = max(1, min(timbre.n_partials(), int(round(w))))
            mask = np.zeros(timbre.n_partials(), dtype=np.float64)
            mask[:n] = 1.0
            timbre = timbre.with_partials(
                amplitudes=np.asarray(timbre.amplitudes, dtype=np.float64) * mask
            )

        elif ev == "intermod_buildup":
            if bt is not None and float(w) > 0:
                timbre = timbre.with_intermod_sidebands(bt, depth=float(w))

        elif ev == "harmonic_stack":
            n = max(0, int(round(w)))
            if n > 0:
                rolloff = float(layer.params.get("rolloff", 0.9))
                timbre = timbre.with_harmonic_stack(n=n, rolloff=rolloff)

        elif ev == "formant_sweep":
            width = float(layer.params.get("width_hz", 800.0))
            gain  = float(layer.params.get("gain_db", 4.0))
            timbre = timbre.with_formant(
                center_hz=float(w), width_hz=width, gain_db=gain,
            )

        elif ev == "phase_sweep":
            n = timbre.n_partials()
            new_phases = (np.arange(1, n + 1) * float(w)) % (2.0 * np.pi)
            timbre = timbre.with_partials(phases=new_phases)

        elif ev == "amp_morph":
            alpha = float(w)
            n = timbre.n_partials()
            random_amps = rng.uniform(0.0, 1.0, n)
            morphed = (1 - alpha) * random_amps + alpha * np.asarray(
                timbre.amplitudes, dtype=np.float64
            )
            mx = float(np.max(np.abs(morphed))) or 1e-9
            timbre = timbre.with_partials(amplitudes=morphed / mx)

        elif ev == "wavefolding":
            # Needs the rendered cycle. Force a render the first time
            # we see a waveform layer.
            if cycle is None:
                cycle = render_wavetable_cycle(timbre, table_size=table_size)
            drive = float(layer.params.get("output_drive", 1.0))
            pre = cycle * drive
            folded = np.sin(np.pi * pre * (1.0 + float(w)) / 2.0)
            peak = float(np.max(np.abs(folded))) or 1.0
            cycle = (folded * (0.99 / peak)).astype(np.float32, copy=False)

        elif ev == "fm_baked":
            cm = float(layer.params.get("cm_ratio", 2.0))
            target = int(layer.params.get("target_partial_idx", 0))
            cycle = _frame_with_fm_baked(
                timbre, float(w), table_size=table_size,
                cm_ratio=cm, target_partial_idx=target,
            )

        else:  # pragma: no cover — guarded by _COMPOSITE_ALLOWED check above
            raise ValueError(f"Unknown composite layer evolution: {ev!r}")

    # Final render if no waveform / terminal layer touched the cycle.
    if cycle is None:
        cycle = render_wavetable_cycle(timbre, table_size=table_size)
    return cycle.astype(np.float32, copy=False)


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
    # Nonlinear-enrichment evolutions:
    fold_range: tuple[float, float] = (0.0, 4.0),
    fm_index_range: tuple[float, float] = (0.0, 3.0),
    fm_cm_ratio: float = 2.0,
    fm_target_partial_idx: int = 0,
    # Composite evolution: list of layers (or dicts coerced to layers).
    composite_layers: Sequence[Any] | None = None,
    # Noise-to-structure evolution:
    noise_exponent: float | None = None,    # 1/f^k slope; None → use timbre.spectral_tilt or 1.0
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
    elif evolution == "wavefolding":
        folds = np.linspace(fold_range[0], fold_range[1], n_frames)
        frames = [
            _frame_with_wavefolding(timbre, float(f), table_size=table_size)
            for f in folds
        ]
        full = np.concatenate(frames).astype(np.float32, copy=False)
    elif evolution == "fm_baked":
        indices = np.linspace(fm_index_range[0], fm_index_range[1], n_frames)
        frames = [
            _frame_with_fm_baked(
                timbre, float(b), table_size=table_size,
                cm_ratio=fm_cm_ratio, target_partial_idx=fm_target_partial_idx,
            )
            for b in indices
        ]
        full = np.concatenate(frames).astype(np.float32, copy=False)
    elif evolution == "noise_to_structure":
        alphas = np.linspace(0.0, 1.0, n_frames)
        frames = [
            _frame_with_noise_to_structure(
                timbre, float(a), table_size=table_size,
                exponent=noise_exponent, seed=seed,
            )
            for a in alphas
        ]
        full = np.concatenate(frames).astype(np.float32, copy=False)
    elif evolution == "composite":
        if not composite_layers:
            raise ValueError(
                "evolution='composite' requires composite_layers= "
                "(a non-empty list of WavetableLayer or dict configs)"
            )
        # Coerce dict entries to WavetableLayer for the helper.
        layers = [
            l if isinstance(l, WavetableLayer) else WavetableLayer(**l)
            for l in composite_layers
        ]
        frames = [
            _frame_composite(
                timbre, layers, i, n_frames,
                table_size=table_size, bt=bt, seed=seed,
            )
            for i in range(n_frames)
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
            fold_range=fold_range,
            fm_index_range=fm_index_range,
            fm_cm_ratio=fm_cm_ratio,
            fm_target_partial_idx=fm_target_partial_idx,
            composite_layers=composite_layers,
            noise_exponent=noise_exponent,
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
    fold_range=None,
    fm_index_range=None,
    fm_cm_ratio=None,
    fm_target_partial_idx=None,
    composite_layers=None,
    noise_exponent=None,
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
    if evolution == "wavefolding":
        return {"fold_range": list(fold_range or (0.0, 4.0))}
    if evolution == "fm_baked":
        return {
            "fm_index_range":         list(fm_index_range or (0.0, 3.0)),
            "cm_ratio":               float(fm_cm_ratio if fm_cm_ratio is not None else 2.0),
            "target_partial_idx":     int(fm_target_partial_idx
                                          if fm_target_partial_idx is not None else 0),
        }
    if evolution == "noise_to_structure":
        return {
            "noise_exponent": (
                float(noise_exponent) if noise_exponent is not None else None
            ),
            "seed": int(seed),
        }
    if evolution == "composite":
        # Coerce any WavetableLayer dataclasses in the list to plain
        # dicts so the manifest JSON can serialise them.
        serialised = []
        for l in (composite_layers or []):
            if isinstance(l, WavetableLayer):
                serialised.append({
                    "evolution":    l.evolution,
                    "weight_curve": l.weight_curve,
                    "weight_min":   float(l.weight_min),
                    "weight_max":   float(l.weight_max),
                    "params":       dict(l.params),
                })
            else:
                serialised.append(dict(l))
        return {"composite_layers": serialised}
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
