"""biotuner.resonance — modular, strategy-registry-based resonance framework.

The resonance package builds per-frequency resonance spectra from a signal via
swappable harmonic kernels, ratio kernels, phase estimators, pairwise coupling
metrics, persistence (Q) measures, and combine rules — with optional surrogate
normalization and a separate path for higher-order (triplet / N-ary / survey /
state-level) coupling metrics.

Public entry points (populated as each module lands):
    compute_resonance(signal, sf, config=None, freqs=None) -> ResonanceResult
    ResonanceConfig
    ResonanceResult
    HigherOrderResult
    with_surrogate_null(signal, sf, config, ...)

See ``biotuner/resonance/orchestrator.py`` for the per-step pipeline and
``biotuner/resonance/registry.py`` for the strategy catalog.
"""

# Importing kernel modules registers them in the registry as a side effect.
from biotuner.resonance import kernels_harmonic  # noqa: F401

__all__ = []

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
