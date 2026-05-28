"""biotuner.resonance.phase_estimators — instantaneous phase extraction.

Phase estimators produce a ``(n_freqs, n_times)`` phase matrix from a 1-D signal.
Phase 1 ships ``stft_phase`` (the legacy STFT-bin phase). Phase 2 adds
``hilbert_bandpass_phase`` and ``morlet_wavelet_phase`` from plan §5.3.

Note: STFT-bin phase is *not* true oscillation phase (it's affected by leakage).
New code should prefer Hilbert-bandpass once available; STFT is retained for
backward compatibility only.
"""

import numpy as np
from scipy.signal import stft

from biotuner.resonance.registry import register_phase_estimator


def stft_phase(
    signal: np.ndarray,
    sf: float,
    *,
    precision_hz: float,
    noverlap: int = 10,
    smoothness: float = 1,
    **_unused,
) -> np.ndarray:
    """STFT-bin phase. Bit-equivalent to legacy ``compute_phase_values``.

    Parameters
    ----------
    signal : 1-D array
    sf : sampling frequency (Hz)
    precision_hz : frequency precision (Hz); ``nperseg = int(sf / precision_hz)``.
    noverlap : STFT overlap (samples)
    smoothness : divides nperseg; ``smoothness=1`` means nperseg as computed.

    Returns
    -------
    ndarray (n_freqs, n_times)
        ``np.angle(Zxx)`` from ``scipy.signal.stft``.
    """
    nperseg = int(sf / precision_hz)
    _, _, Zxx = stft(signal, sf, nperseg=int(nperseg / smoothness), noverlap=noverlap)
    return np.angle(Zxx)


register_phase_estimator("stft", stft_phase)
