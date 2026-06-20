"""biotuner.resonance.phase_estimators — instantaneous phase extraction.

Phase estimators produce a ``(n_freqs, n_times)`` phase matrix from a 1-D signal,
with **row ``i`` aligned to ``freqs[i]``** (the analysis frequency grid passed by
the orchestrator). Two estimators are registered:

  stft (default)
    STFT-bin phase. Bit-equivalent to legacy ``compute_phase_values`` for the raw
    transform, but now the returned rows are selected to match the analysis
    ``freqs`` grid. (Historically the full 0..Nyquist STFT grid was indexed with
    the [fmin, fmax]-clipped ``freqs``, so phase[i] was off by ``fmin`` — that
    bug corrupted every phase-coupling entry. Passing ``freqs`` here fixes it.)
    STFT-bin phase is still leakage-limited and is *not* true oscillation phase.

  hilbert
    Hilbert analytic phase of the signal band-pass-filtered around each analysis
    frequency. This is true instantaneous phase and recovers n:m phase locking
    that the STFT-bin estimator misses (validated in resonance_paper Study 5).
    Aligned to ``freqs`` by construction.

Both accept and ignore extra kwargs so the orchestrator can pass a common set.
"""

import numpy as np
from scipy.signal import stft, butter, sosfiltfilt, hilbert

from biotuner.resonance.registry import register_phase_estimator


def stft_phase(
    signal: np.ndarray,
    sf: float,
    *,
    precision_hz: float,
    noverlap: int = 10,
    smoothness: float = 1,
    freqs: np.ndarray = None,
    **_unused,
) -> np.ndarray:
    """STFT-bin phase, with rows aligned to ``freqs`` when provided.

    Parameters
    ----------
    signal : 1-D array
    sf : sampling frequency (Hz)
    precision_hz : frequency precision (Hz); ``nperseg = int(sf / precision_hz)``.
    noverlap : STFT overlap (samples)
    smoothness : divides nperseg; ``smoothness=1`` means nperseg as computed.
    freqs : analysis frequency grid. When given, the returned phase rows are the
        STFT bins nearest each ``freqs`` value, so ``phase[i]`` is the phase at
        ``freqs[i]``. When None, the full 0..Nyquist grid is returned (legacy).

    Returns
    -------
    ndarray (n_freqs, n_times) of ``np.angle(Zxx)``.
    """
    nperseg = int(sf / precision_hz)
    f_stft, _, Zxx = stft(signal, sf, nperseg=int(nperseg / smoothness), noverlap=noverlap)
    phase = np.angle(Zxx)
    if freqs is not None:
        # Select the STFT bin nearest each analysis frequency (alignment fix).
        idx = np.abs(f_stft[:, None] - np.asarray(freqs)[None, :]).argmin(axis=0)
        phase = phase[idx, :]
    return phase


def hilbert_bandpass_phase(
    signal: np.ndarray,
    sf: float,
    *,
    freqs: np.ndarray,
    bandwidth: float = 2.0,
    filter_order: int = 4,
    **_unused,
) -> np.ndarray:
    """Hilbert analytic phase of the signal band-passed around each ``freqs[i]``.

    Returns true instantaneous phase at sample resolution, aligned to ``freqs``
    by construction. For a band centered at ``f`` the passband is
    ``[f - bandwidth/2, f + bandwidth/2]`` (clamped to ``(0, Nyquist)``).

    Returns
    -------
    ndarray (n_freqs, n_times) of unwrapped-then-wrapped instantaneous phase
    angles (``np.angle`` of the analytic signal).
    """
    sig = np.asarray(signal, dtype=np.float64)
    nyq = sf / 2.0
    freqs = np.asarray(freqs, dtype=np.float64)
    out = np.empty((freqs.size, sig.size), dtype=np.float64)
    for i, f in enumerate(freqs):
        lo = max(f - bandwidth / 2.0, 1e-6)
        hi = min(f + bandwidth / 2.0, nyq - 1e-6)
        if hi <= lo:
            out[i] = 0.0
            continue
        sos = butter(filter_order, [lo / nyq, hi / nyq], btype="band", output="sos")
        band = sosfiltfilt(sos, sig)
        out[i] = np.angle(hilbert(band))
    return out


register_phase_estimator("stft", stft_phase)
register_phase_estimator("hilbert", hilbert_bandpass_phase)
