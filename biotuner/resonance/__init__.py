"""biotuner.resonance — modular, strategy-registry-based resonance framework.

This package builds the **per-frequency resonance spectrum** R(f) = H(f) · PC(f)
from a single signal via swappable harmonic kernels, ratio kernels, phase
estimators, pairwise coupling metrics, persistence (Q) measures, and combine
rules — with optional surrogate normalization.

Quick start
-----------

The recommended path uses default config (joint-probability PC + n:m ratio
gating + IAAFT-friendly cross-channel hooks already wired):

::

    from biotuner.resonance import compute_resonance, ResonanceConfig

    # Default config — recommended for new analyses
    result = compute_resonance(signal, sf=1000)
    # result.factors["H"], result.factors["PC"]
    # result.resonance_spectrum  — H · PC
    # result.summaries["H"/"PC"/"R"]  — complexity dict per spectrum
    # result.peaks  — prominence-detected peak freqs

    # To reproduce legacy compute_global_harmonicity bit-exactly:
    cfg = ResonanceConfig(psd_normalization="minmax_prob", ...)
    result = compute_resonance(signal, sf=1000, config=cfg)

Sister modules
--------------
- :mod:`biotuner.harmonic_spectrum` — narrow H-only entry point
  (``compute_harmonic_spectrum``).
- :mod:`biotuner.harmonic_connectivity` — cross-channel API:
  ``compute_cross_resonance`` for two signals,
  ``harmonic_connectivity(...).compute_cross_resonance_connectivity()`` for
  N-channel matrices, ``compute_cross_resonance_connectivity_zscore()`` for
  surrogate-normalized statistical inference.

Public API
----------
- :func:`compute_resonance` — main entry point
- :class:`ResonanceConfig` — all swappable knobs
- :class:`ResonanceResult` — output dataclass
- :class:`HigherOrderResult` — for Phase 3 higher-order metrics
- :func:`with_surrogate_null` — surrogate-z-scored variant of compute_resonance
- :func:`results_to_dataframe` — pack multiple results into a DataFrame for the plotters
- ``harmonic_spectrum_plot_*`` — comparison plots over collections of results

Internals
---------
Strategy catalogs live in ``biotuner.resonance.registry``. To add a new kernel,
metric, or combine rule, call the appropriate ``register_*`` function from your
own module; it'll be discoverable by name via :class:`ResonanceConfig`.
"""

# Importing kernel modules registers them in the registry as a side effect.
from biotuner.resonance import kernels_harmonic  # noqa: F401

from biotuner.resonance.registry import list_strategies

__all__ = ["list_strategies"]

try:
    from biotuner.resonance.orchestrator import (
        ResonanceConfig,
        ResonanceResult,
        HigherOrderResult,
        compute_resonance,
    )
    __all__.extend(["ResonanceConfig", "ResonanceResult", "HigherOrderResult", "compute_resonance"])
except ImportError:
    pass

try:
    from biotuner.resonance.nulls import with_surrogate_null
    __all__.append("with_surrogate_null")
except ImportError:
    pass

try:
    from biotuner.resonance.coupling import nm_intertrial_plv
    __all__.append("nm_intertrial_plv")
except ImportError:
    pass

try:
    from biotuner.resonance.plots import (
        harmonic_spectrum_plot_trial_corr,
        harmonic_spectrum_plot_freq_corr,
        harmonic_spectrum_plot_avg_corr,
        results_to_dataframe,
    )
    __all__.extend([
        "harmonic_spectrum_plot_trial_corr",
        "harmonic_spectrum_plot_freq_corr",
        "harmonic_spectrum_plot_avg_corr",
        "results_to_dataframe",
    ])
except ImportError:
    pass
