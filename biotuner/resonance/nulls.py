"""biotuner.resonance.nulls — surrogate-based null normalization.

Wraps :func:`biotuner.surrogates.generate_surrogate` to z-score (and/or p-value)
the resonance spectrum and scalar summaries against a surrogate distribution.
Off by default; opt in via ``ResonanceConfig.null_model``.

For cross-channel analyses, three null generators are exposed for use with
:func:`biotuner.harmonic_connectivity.compute_cross_resonance`:

  phase_randomize_surrogate(signal)
    Fourier phase randomization — preserves PSD exactly, destroys phase
    structure. Permissive null (any signal that has the same PSD passes).

  iaaft_surrogate(signal)
    Iterated Amplitude-Adjusted Fourier Transform (Schreiber & Schmitz 1996).
    Preserves BOTH the PSD and the amplitude distribution. Tighter than plain
    phase randomization — rejects more spurious cross-channel coupling.

  time_shuffle_surrogate(signal)
    Block-shuffle the time-domain signal. Destroys temporal phase relationships
    while preserving local PSD structure. Strict null for cross-channel
    coherence testing.

Plan §5.6 and Appendix A.5.
"""

import numpy as np
from typing import Optional, Literal


def with_surrogate_null(
    signal: np.ndarray,
    sf: float,
    config,
    *,
    surr_type: str = "IAAFT",
    n: int = 200,
    correction: Literal["zscore", "pvalue", "both"] = "zscore",
    parallel: bool = True,
    rng_seed: Optional[int] = None,
):
    """Compute resonance on signal and on ``n`` surrogates; z-score the spectrum.

    Returns a :class:`ResonanceResult` with ``resonance_spectrum_z``,
    ``surrogate_mean``, ``surrogate_std`` populated (and ``summaries['p_value_spectrum']``
    if ``correction`` includes pvalue).

    Parameters
    ----------
    signal : 1-D ndarray
    sf : sampling frequency (Hz)
    config : ResonanceConfig (the null_model field is ignored to avoid recursion)
    surr_type : surrogate type understood by ``biotuner.surrogates.generate_surrogate``
    n : number of surrogates
    correction : 'zscore' | 'pvalue' | 'both'
    parallel : if True and joblib is importable, parallelize across surrogates
    rng_seed : optional seed for reproducibility
    """
    # Lazy import to avoid circular dep at package load
    from biotuner.resonance.orchestrator import compute_resonance
    from biotuner.surrogates import generate_surrogate

    # Strip null_model from the config we pass to the per-surrogate call
    import dataclasses
    cfg_no_null = dataclasses.replace(config, null_model=None)

    real = compute_resonance(signal, sf, config=cfg_no_null)
    n_freqs = real.resonance_spectrum.size

    def _one(seed):
        s = generate_surrogate(signal, surr_type=surr_type, sf=sf)
        r = compute_resonance(s, sf, config=cfg_no_null)
        return r.resonance_spectrum

    if parallel:
        try:
            from joblib import Parallel, delayed
            rng = np.random.default_rng(rng_seed)
            seeds = rng.integers(0, 2**31 - 1, size=n)
            surr_spectra = np.asarray(
                Parallel(n_jobs=-1)(delayed(_one)(int(s)) for s in seeds),
                dtype=np.float64,
            )
        except ImportError:
            surr_spectra = np.empty((n, n_freqs), dtype=np.float64)
            for k in range(n):
                surr_spectra[k] = _one(k)
    else:
        surr_spectra = np.empty((n, n_freqs), dtype=np.float64)
        for k in range(n):
            surr_spectra[k] = _one(k)

    mu = surr_spectra.mean(axis=0)
    sd = surr_spectra.std(axis=0) + 1e-12
    real.resonance_spectrum_z = (real.resonance_spectrum - mu) / sd
    real.surrogate_mean = mu
    real.surrogate_std = sd
    if correction in ("pvalue", "both"):
        p = (np.sum(surr_spectra >= real.resonance_spectrum[None, :], axis=0) + 1) / (n + 1)
        real.summaries["p_value_spectrum"] = p
    return real


# ---------------------------------------------------------------------------
# Cross-channel surrogate generators
# ---------------------------------------------------------------------------


def phase_randomize_surrogate(signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Fourier phase randomization. Preserves PSD exactly, destroys phase
    structure. Most permissive null for cross-channel coupling tests.
    """
    N = len(signal)
    X = np.fft.rfft(signal)
    phases = np.exp(1j * rng.uniform(0, 2 * np.pi, size=X.shape))
    phases[0] = 1.0  # DC must stay real
    if N % 2 == 0:
        phases[-1] = 1.0  # Nyquist must stay real for even N
    return np.fft.irfft(np.abs(X) * phases, n=N)


def iaaft_surrogate(
    signal: np.ndarray,
    rng: np.random.Generator,
    n_iter: int = 100,
    tol: float = 1e-6,
) -> np.ndarray:
    """Iterated Amplitude-Adjusted Fourier Transform (Schreiber & Schmitz 1996,
    *PRL* 77:635). Preserves BOTH the power spectrum AND the empirical
    amplitude distribution of the original signal — a tighter null than
    plain phase randomization.

    Algorithm:
      1. Start from a random shuffle of the signal.
      2. Iterate: (a) replace the FFT magnitudes with the original PSD's,
                  (b) rank-match back to the original amplitude distribution.
      3. Stop when amplitude distribution converges or n_iter reached.
    """
    N = len(signal)
    target_amp_sorted = np.sort(signal)
    target_psd_amp = np.abs(np.fft.rfft(signal))

    # Initial: random shuffle
    surr = rng.permutation(signal)
    prev_diff = np.inf
    for _ in range(n_iter):
        # Step 1: enforce target PSD
        X = np.fft.rfft(surr)
        phases = np.angle(X)
        X_new = target_psd_amp * np.exp(1j * phases)
        surr = np.fft.irfft(X_new, n=N)
        # Step 2: enforce target amplitude distribution by rank-matching
        ranks = np.argsort(np.argsort(surr))
        surr_new = target_amp_sorted[ranks]
        diff = float(np.mean((surr_new - surr) ** 2))
        surr = surr_new
        if abs(prev_diff - diff) < tol:
            break
        prev_diff = diff
    return surr


def time_shuffle_surrogate(
    signal: np.ndarray,
    rng: np.random.Generator,
    block_size: Optional[int] = None,
) -> np.ndarray:
    """Block-shuffle surrogate. Cuts the signal into blocks of length
    ``block_size`` (default len(signal)//20) and randomly reorders them.
    Destroys long-range temporal phase relationships while preserving local
    PSD structure within blocks. Strictest cross-channel null.
    """
    N = len(signal)
    if block_size is None:
        block_size = max(1, N // 20)
    n_blocks = N // block_size
    if n_blocks < 2:
        # Fall back to phase randomization if signal too short for blocking
        return phase_randomize_surrogate(signal, rng)
    trimmed_N = n_blocks * block_size
    blocks = signal[:trimmed_N].reshape(n_blocks, block_size)
    perm = rng.permutation(n_blocks)
    shuffled = blocks[perm].reshape(-1)
    # Pad back to original length with the tail of the original signal
    if trimmed_N < N:
        shuffled = np.concatenate([shuffled, signal[trimmed_N:]])
    return shuffled


CROSS_SURROGATE_GENERATORS = {
    "phase_randomize": phase_randomize_surrogate,
    "iaaft": iaaft_surrogate,
    "time_shuffle": time_shuffle_surrogate,
}
