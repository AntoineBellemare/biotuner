"""biotuner.harmonic_timbre.direct_synth — instrument constructors that
do not go through the standard match → Timbre pipeline.

Module type: Functions

These are "direct" mappings from biotuner outputs (or raw signals) to
audio or to a modulator-bearing Timbre. They complement
:func:`~biotuner.harmonic_timbre.match_timbre`:

    hilbert_instrument(signal, sf, ...)
        Treat the signal's analytic-signal envelope and instantaneous
        frequency as drivers of a single oscillator. Plays the biosignal
        as audio without any tuning step in between.

    fm_patch_from_tuning(ratios, ...)
        Use a tuning's ratios as FM operator-to-carrier ratios.
        Produces a Timbre whose ``fm_modulators`` are populated; render
        via :func:`render_modulated` to hear it.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from biotuner.harmonic_timbre.timbre import Modulator, Timbre


# ---------------------------------------------------------------------------
# hilbert_instrument
# ---------------------------------------------------------------------------

def hilbert_instrument(
    signal,
    sf: float,
    *,
    samplerate: int = 48000,
    duration: float | None = None,
    base_freq: float = 220.0,
    pitch_factor: float = 50.0,
    smooth_amp: float = 0.01,
    normalize: bool = True,
) -> np.ndarray:
    """Render a biosignal as audio by treating its analytic signal as
    instantaneous frequency + instantaneous amplitude.

    Pipeline:
        1. ``analytic = scipy.signal.hilbert(signal)``
        2. inst. amplitude = ``|analytic|``
        3. inst. phase     = ``unwrap(angle(analytic))``
        4. inst. frequency = derivative of inst. phase × ``sf / 2π``
        5. shift instantaneous frequency up to audible range:
              audible_inst_freq = base_freq + inst_freq * pitch_factor
        6. integrate inst. freq → audio phase, oscillate, modulate by amp

    The result is a single-oscillator instrument that "plays" the
    biosignal — every change in the signal's amplitude or phase shows
    up in the audio. This is the most literal possible mapping and
    skips the tuning/timbre-matching pipeline entirely.

    Parameters
    ----------
    signal : array-like
        1D biosignal samples.
    sf : float
        Source samplerate (Hz) of the biosignal.
    samplerate : int, default=48000
        Output audio samplerate.
    duration : float, optional
        Output duration in seconds. If None, uses ``len(signal) / sf``.
    base_freq : float, default=220.0
        Audio carrier frequency that the biosignal's instantaneous
        frequency is added to.
    pitch_factor : float, default=50.0
        Multiplier on the biosignal's instantaneous frequency before
        adding to ``base_freq``. ``pitch_factor=50`` turns a 4 Hz
        modulation into a 200 Hz audio swing.
    smooth_amp : float, default=0.01
        Time-constant (s) for a single-pole low-pass on the amplitude
        envelope. Smooths out per-sample jitter; set to 0 for raw envelope.
    normalize : bool, default=True
        Peak-normalize the output to 0.99.

    Returns
    -------
    np.ndarray
        ``float32`` mono audio buffer.
    """
    from scipy.signal import hilbert

    sig = np.asarray(signal, dtype=np.float64).flatten()
    if sig.size < 4:
        raise ValueError("hilbert_instrument: signal must have at least 4 samples")

    sf = float(sf)
    if duration is None:
        duration = sig.size / sf
    n_samples = int(round(duration * samplerate))
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float32)

    analytic = hilbert(sig)
    inst_amp = np.abs(analytic)
    inst_phase = np.unwrap(np.angle(analytic))
    inst_freq_src = np.gradient(inst_phase) * sf / (2.0 * np.pi)

    # Resample inst_amp and inst_freq_src to the audio samplerate
    src_t = np.linspace(0.0, duration, sig.size, endpoint=False)
    dst_t = np.arange(n_samples) / float(samplerate)
    inst_amp_audio = np.interp(dst_t, src_t, inst_amp)
    inst_freq_audio = np.interp(dst_t, src_t, inst_freq_src)

    # Optional amplitude smoothing
    if smooth_amp and smooth_amp > 0:
        alpha = float(np.exp(-1.0 / (smooth_amp * samplerate)))
        env = np.empty_like(inst_amp_audio)
        prev = inst_amp_audio[0]
        for n, v in enumerate(inst_amp_audio):
            prev = alpha * prev + (1.0 - alpha) * v
            env[n] = prev
        inst_amp_audio = env

    # Map inst-freq into the audible range
    inst_freq_audio_audible = base_freq + inst_freq_audio * pitch_factor
    # accumulate phase
    audio_phase = 2.0 * np.pi * np.cumsum(inst_freq_audio_audible) / float(samplerate)

    out = inst_amp_audio * np.sin(audio_phase)

    if normalize:
        m = float(np.max(np.abs(out))) if out.size else 0.0
        if m > 0:
            out = out * (0.99 / m)
    return out.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# fm_patch_from_tuning
# ---------------------------------------------------------------------------

def fm_patch_from_tuning(
    ratios,
    *,
    base_freq: float = 220.0,
    n_carriers: int = 4,
    fm_index: float = 2.0,
    falloff: str = "1_over_n",
) -> Timbre:
    """Use a tuning's ratios as FM modulator-to-carrier frequency ratios.

    This is the classic Yamaha-DX7 approach generalized to arbitrary
    tunings. For each pair (carrier, ratio) the patch installs an FM
    operator with:

        carrier_freq = base_freq * (1, 2, 3, ...)         (one per carrier)
        mod_freq     = carrier_freq * ratio               (the tuning's ratio)
        depth_hz     = fm_index * mod_freq                (FM index β = fm_index)

    Different tunings produce different sideband patterns under FM —
    JI fifths give bright bell-like tones, Bohlen-Pierce ratios give
    metallic overtones, gamelan ratios give clangorous textures.
    Useful as a fast way to hear a tuning's "FM character."

    Parameters
    ----------
    ratios : iterable of float
        The tuning's ratios. Ratios > 1 are wrapped into the unit
        equave window before use.
    base_freq : float, default=220.0
        Frequency of the lowest carrier.
    n_carriers : int, default=4
        Number of integer-multiple carriers placed at base, 2*base,
        3*base, … . Each carrier gets its own FM modulator using one
        of the tuning's ratios (cycled if there are more carriers than
        ratios).
    fm_index : float, default=2.0
        FM modulation index (β). 0.5 is subtle, 2.0 is moderate, 5+ is
        very harsh.
    falloff : str, default='1_over_n'
        Carrier amplitude law (``'1_over_n'``, ``'1_over_n_squared'``,
        ``'flat'``).

    Returns
    -------
    Timbre
        With ``fm_modulators`` populated; render via
        :func:`~biotuner.harmonic_timbre.render_modulated`.
    """
    rs = [float(r) for r in ratios]
    if not rs:
        raise ValueError("fm_patch_from_tuning: empty ratios")
    if n_carriers < 1:
        raise ValueError("fm_patch_from_tuning: n_carriers must be ≥ 1")

    partials = base_freq * np.arange(1, n_carriers + 1, dtype=np.float64)

    if falloff == "1_over_n":
        amps = 1.0 / np.arange(1, n_carriers + 1, dtype=np.float64)
    elif falloff == "1_over_n_squared":
        amps = 1.0 / (np.arange(1, n_carriers + 1, dtype=np.float64) ** 2)
    elif falloff == "flat":
        amps = np.ones(n_carriers, dtype=np.float64)
    else:
        raise ValueError(f"fm_patch_from_tuning: unknown falloff {falloff!r}")
    amps = amps / amps.max()

    fm_modulators: list[Modulator] = []
    for i in range(n_carriers):
        # cycle through the tuning's ratios, skipping the unison if present
        ratio = rs[(i + 1) % len(rs)] if abs(rs[0] - 1.0) < 1e-9 else rs[i % len(rs)]
        car_freq = float(partials[i])
        mod_freq = car_freq * ratio
        depth_hz = float(fm_index) * mod_freq
        fm_modulators.append(Modulator(
            carrier_idx=i,
            mod_freq=mod_freq,
            depth=depth_hz,
            mod_type="FM",
            source=f"tuning_ratio[{ratio:.4f}]",
        ))

    return Timbre(
        partials_hz=partials,
        amplitudes=amps,
        base_freq=base_freq,
        matched_tuning=list(rs),
        matching_method="fm_patch_from_tuning",
        fm_modulators=fm_modulators,
        metadata={
            "n_carriers": int(n_carriers),
            "fm_index": float(fm_index),
            "falloff": falloff,
        },
    )
