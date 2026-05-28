"""biotuner.resonance.nulls — surrogate-based null normalization.

Wraps :func:`biotuner.surrogates.generate_surrogate` to z-score (and/or p-value)
the resonance spectrum and scalar summaries against a surrogate distribution.
Off by default; opt in via ``ResonanceConfig.null_model``.

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
