"""biotuner.harmonic_timbre.synthesis — render Timbres to numpy audio.

Module type: Functions

Pure offline rendering. Not realtime, not streaming. The Goofy Pipe
realtime layer calls into ``Timbre.synthesize(duration=buf/sr)`` per
audio buffer; rendering must be efficient enough that 32 partials at
~512 samples completes well under 5 ms on reference hardware.

Phase 1 surface
---------------
render_additive
    Plain additive synthesis using a vectorized sum of sines. Fastest path.
render_with_envelope
    Adds per-partial decay envelopes (from ``timbre.decay_times``),
    spectral tilt (1/f), broadband noise floor.
render_band_limited
    Adds filtered-noise partials when ``timbre.bandwidths`` is set
    (Lorentzian-shaped per partial).
render_wavetable_cycle
    Single-cycle waveform at unit frequency for wavetable exporters.

The Phase 3 surface (modulators, layers, sequence morphing) is not
wired in this file yet but the dispatcher in ``Timbre.synthesize``
selects between these functions based on which optional fields are set.
"""

from __future__ import annotations

import numpy as np

from biotuner.harmonic_timbre.timbre import Timbre


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_partials(timbre: Timbre, base_freq: float | None) -> np.ndarray:
    """Return absolute partial frequencies, optionally rescaled to a new base.

    If ``base_freq`` is provided and ``timbre.base_freq != base_freq``,
    partials are scaled by ``base_freq / timbre.base_freq`` (treating
    ``timbre.partials_hz`` as relative to ``timbre.base_freq``).
    """
    p = np.asarray(timbre.partials_hz, dtype=np.float64)
    if base_freq is None or timbre.base_freq == 0:
        return p
    return p * (float(base_freq) / float(timbre.base_freq))


def _resolve_phases(timbre: Timbre) -> np.ndarray:
    if timbre.phases is None:
        return np.zeros(timbre.n_partials(), dtype=np.float64)
    return np.asarray(timbre.phases, dtype=np.float64)


def _apply_spectral_tilt(amps: np.ndarray, partials: np.ndarray, exponent: float) -> np.ndarray:
    """Multiply amplitudes by ``(f0 / f)**exponent`` (1/f-like falloff)."""
    if exponent == 0.0:
        return amps
    f0 = float(partials.min()) if partials.size else 1.0
    if f0 <= 0:
        return amps
    factor = np.power(f0 / np.maximum(partials, 1e-12), exponent)
    return amps * factor


def _normalize(buf: np.ndarray, peak: float = 0.99) -> np.ndarray:
    if buf.size == 0:
        return buf
    m = float(np.max(np.abs(buf)))
    if m <= 0:
        return buf
    return buf * (peak / m)


# ---------------------------------------------------------------------------
# render_additive
# ---------------------------------------------------------------------------

def render_additive(
    timbre: Timbre,
    *,
    samplerate: int = 48000,
    duration: float = 1.0,
    base_freq: float | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Vectorized additive synthesis. Returns float32 mono."""
    timbre.validate()
    partials = _resolve_partials(timbre, base_freq)
    amps = np.asarray(timbre.amplitudes, dtype=np.float64)
    phases = _resolve_phases(timbre)

    n_samples = int(round(duration * samplerate))
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(n_samples, dtype=np.float64) / float(samplerate)

    # outer product:  shape (n_partials, n_samples)
    arg = 2.0 * np.pi * partials[:, None] * t[None, :] + phases[:, None]
    sines = np.sin(arg)
    out = (amps[:, None] * sines).sum(axis=0)

    if normalize:
        out = _normalize(out)
    return out.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# render_with_envelope
# ---------------------------------------------------------------------------

def render_with_envelope(
    timbre: Timbre,
    *,
    samplerate: int = 48000,
    duration: float = 1.0,
    base_freq: float | None = None,
    attack: float = 0.005,
    release: float | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Additive synthesis with per-partial decay, optional global attack,
    spectral tilt, and broadband noise floor.

    ``timbre.decay_times`` is interpreted as the exponential decay time
    constant per partial (in seconds): the partial amplitude is multiplied
    by ``exp(-t / tau)``. ``None`` for any partial means "no decay" (the
    overall ``release`` argument can still apply).
    """
    timbre.validate()
    partials = _resolve_partials(timbre, base_freq)
    amps = np.asarray(timbre.amplitudes, dtype=np.float64).copy()
    phases = _resolve_phases(timbre)
    n_partials = partials.size

    # spectral tilt: 1/f exponent
    if timbre.spectral_tilt is not None:
        amps = _apply_spectral_tilt(amps, partials, float(timbre.spectral_tilt))

    n_samples = int(round(duration * samplerate))
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(n_samples, dtype=np.float64) / float(samplerate)

    # base sines, shape (n_partials, n_samples)
    arg = 2.0 * np.pi * partials[:, None] * t[None, :] + phases[:, None]
    sines = np.sin(arg)

    # per-partial decay envelope
    if timbre.decay_times is not None:
        tau = np.asarray(timbre.decay_times, dtype=np.float64)
        # treat non-finite or non-positive entries as "no decay" (large tau)
        tau = np.where(np.isfinite(tau) & (tau > 0), tau, 1e9)
        env = np.exp(-t[None, :] / tau[:, None])
        sines = sines * env

    out = (amps[:, None] * sines).sum(axis=0)

    # optional global attack (linear ramp, prevents click at t=0)
    if attack and attack > 0:
        n_atk = int(round(min(attack, duration) * samplerate))
        if n_atk > 0:
            ramp = np.linspace(0.0, 1.0, n_atk, dtype=np.float64)
            out[:n_atk] *= ramp

    # optional global release ramp
    if release and release > 0:
        n_rel = int(round(min(release, duration) * samplerate))
        if n_rel > 0:
            ramp = np.linspace(1.0, 0.0, n_rel, dtype=np.float64)
            out[-n_rel:] *= ramp

    # broadband noise floor
    if timbre.noise_floor is not None and timbre.noise_floor > 0:
        rng = np.random.default_rng(seed=0)  # deterministic; tests rely on this
        noise = rng.standard_normal(n_samples) * float(timbre.noise_floor)
        out = out + noise

    if normalize:
        out = _normalize(out)
    return out.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# render_band_limited
# ---------------------------------------------------------------------------

def render_modulated(
    timbre: Timbre,
    *,
    samplerate: int = 48000,
    duration: float = 1.0,
    base_freq: float | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Additive synthesis with AM and FM modulators applied per-partial.

    Reads ``timbre.am_modulators`` and ``timbre.fm_modulators``. Each
    :class:`Modulator` targets a partial via ``carrier_idx`` and shapes
    that partial's amplitude (AM) or instantaneous frequency (FM):

        AM:  a_k(t) = base_amp * (1 + depth * sin(2π·mod_freq·t + phase))
        FM:  f_k(t) = base_freq + depth * sin(2π·mod_freq·t + phase)
              (depth in Hz; FM index β = depth / mod_freq)

    Multiple modulators on the same carrier are *summed*: their
    contributions add into the same instantaneous-amplitude or
    instantaneous-frequency curve. Modulators with an out-of-range
    ``carrier_idx`` are silently skipped.

    Falls back to :func:`render_with_envelope` when both modulator lists
    are empty.
    """
    timbre.validate()
    if not timbre.am_modulators and not timbre.fm_modulators:
        return render_with_envelope(
            timbre,
            samplerate=samplerate,
            duration=duration,
            base_freq=base_freq,
            normalize=normalize,
        )

    partials = _resolve_partials(timbre, base_freq)
    amps = np.asarray(timbre.amplitudes, dtype=np.float64).copy()
    phases = _resolve_phases(timbre)
    n_partials = partials.size

    if timbre.spectral_tilt is not None:
        amps = _apply_spectral_tilt(amps, partials, float(timbre.spectral_tilt))

    n_samples = int(round(duration * samplerate))
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(n_samples, dtype=np.float64) / float(samplerate)

    # Bucket modulators per carrier
    am_per_carrier: dict[int, list] = {}
    fm_per_carrier: dict[int, list] = {}
    for m in timbre.am_modulators:
        if 0 <= m.carrier_idx < n_partials and m.mod_type == "AM":
            am_per_carrier.setdefault(m.carrier_idx, []).append(m)
    for m in timbre.fm_modulators:
        if 0 <= m.carrier_idx < n_partials and m.mod_type == "FM":
            fm_per_carrier.setdefault(m.carrier_idx, []).append(m)

    out = np.zeros(n_samples, dtype=np.float64)

    for k in range(n_partials):
        f_k = float(partials[k])
        a_k = float(amps[k])
        ph_k = float(phases[k])

        # Instantaneous frequency: base + sum of FM modulators
        if k in fm_per_carrier:
            df = np.zeros(n_samples, dtype=np.float64)
            for m in fm_per_carrier[k]:
                df += float(m.depth) * np.sin(2.0 * np.pi * float(m.mod_freq) * t + float(m.phase))
            inst_phase = 2.0 * np.pi * np.cumsum(f_k + df) / float(samplerate) + ph_k
        else:
            inst_phase = 2.0 * np.pi * f_k * t + ph_k

        # Instantaneous amplitude: base * Π(1 + depth * sin(...))
        # Multiple AMs on the same carrier multiply (each is an envelope).
        if k in am_per_carrier:
            am_env = np.ones(n_samples, dtype=np.float64)
            for m in am_per_carrier[k]:
                am_env *= 1.0 + float(m.depth) * np.sin(2.0 * np.pi * float(m.mod_freq) * t + float(m.phase))
        else:
            am_env = 1.0

        # Per-partial decay (if specified)
        if timbre.decay_times is not None:
            tau = float(timbre.decay_times[k])
            if np.isfinite(tau) and tau > 0:
                am_env = am_env * np.exp(-t / tau)

        out += a_k * am_env * np.sin(inst_phase)

    # Optional broadband noise
    if timbre.noise_floor is not None and timbre.noise_floor > 0:
        rng = np.random.default_rng(seed=0)
        out = out + rng.standard_normal(n_samples) * float(timbre.noise_floor)

    if normalize:
        out = _normalize(out)
    return out.astype(np.float32, copy=False)


def render_band_limited(
    timbre: Timbre,
    *,
    samplerate: int = 48000,
    duration: float = 1.0,
    base_freq: float | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Per-partial Lorentzian-shaped narrowband noise + sinusoid mix.

    For each partial with bandwidth ``Δf > 0``: synthesize a narrowband
    noise centered at the partial frequency, with width ``Δf`` (Hz).
    Implemented in the time domain by convolving a sinusoid with a
    short exponentially-decaying noise impulse — equivalent to driving
    a damped oscillator at that frequency. Cheap and stable.

    Falls back to :func:`render_with_envelope` when ``timbre.bandwidths``
    is None or all zeros.
    """
    timbre.validate()
    if timbre.bandwidths is None or float(np.max(np.abs(timbre.bandwidths))) <= 0:
        return render_with_envelope(
            timbre,
            samplerate=samplerate,
            duration=duration,
            base_freq=base_freq,
            normalize=normalize,
        )

    partials = _resolve_partials(timbre, base_freq)
    amps = np.asarray(timbre.amplitudes, dtype=np.float64).copy()
    bws = np.asarray(timbre.bandwidths, dtype=np.float64)
    phases = _resolve_phases(timbre)

    if timbre.spectral_tilt is not None:
        amps = _apply_spectral_tilt(amps, partials, float(timbre.spectral_tilt))

    n_samples = int(round(duration * samplerate))
    if n_samples <= 0:
        return np.zeros(0, dtype=np.float32)
    t = np.arange(n_samples, dtype=np.float64) / float(samplerate)

    rng = np.random.default_rng(seed=0)  # deterministic for tests
    out = np.zeros(n_samples, dtype=np.float64)

    for i, (f, a, bw, ph) in enumerate(zip(partials, amps, bws, phases)):
        if not np.isfinite(bw) or bw <= 0:
            # plain sine
            out += a * np.sin(2.0 * np.pi * f * t + ph)
            continue
        # narrowband noise: damped-oscillator filter at frequency f, decay π·bw
        # h[n] = exp(-π·bw·t) · cos(2π f t)
        # Generate noise, modulate by carrier, lowpass via simple exponential.
        decay = np.exp(-np.pi * bw * t)
        noise = rng.standard_normal(n_samples)
        # smooth noise by an exponential to give it bandwidth bw
        # one-pole filter: y[n] = α y[n-1] + (1-α) x[n], with α=exp(-π·bw/sr)
        alpha = float(np.exp(-np.pi * bw / samplerate))
        smoothed = np.empty(n_samples)
        prev = 0.0
        for n in range(n_samples):
            prev = alpha * prev + (1.0 - alpha) * noise[n]
            smoothed[n] = prev
        carrier = np.sin(2.0 * np.pi * f * t + ph)
        out += a * decay * carrier * (1.0 + 0.5 * smoothed)

    # noise floor (independent broadband)
    if timbre.noise_floor is not None and timbre.noise_floor > 0:
        out = out + rng.standard_normal(n_samples) * float(timbre.noise_floor)

    if normalize:
        out = _normalize(out)
    return out.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# render_wavetable_cycle
# ---------------------------------------------------------------------------

def render_wavetable_cycle(
    timbre: Timbre,
    *,
    table_size: int = 2048,
) -> np.ndarray:
    """Single-cycle waveform at unit frequency.

    Treats :attr:`Timbre.partials_hz` as harmonic indices when divided by
    :attr:`Timbre.base_freq` — partial *k* contributes a sinusoid at the
    appropriate multiple of the wavetable's fundamental. The result is
    the first cycle of an additive synthesis with these partials, sampled
    at ``table_size`` points.
    """
    timbre.validate()
    if timbre.base_freq <= 0:
        raise ValueError("render_wavetable_cycle: timbre.base_freq must be > 0")
    partials = np.asarray(timbre.partials_hz, dtype=np.float64) / float(timbre.base_freq)
    amps = np.asarray(timbre.amplitudes, dtype=np.float64)
    phases = _resolve_phases(timbre)

    # one cycle of the fundamental: theta = 2π * idx / table_size
    idx = np.arange(table_size, dtype=np.float64)
    theta = 2.0 * np.pi * idx / float(table_size)
    arg = partials[:, None] * theta[None, :] + phases[:, None]
    out = (amps[:, None] * np.sin(arg)).sum(axis=0)
    return _normalize(out).astype(np.float32, copy=False)
