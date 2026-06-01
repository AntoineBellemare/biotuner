"""biotuner.resonance.registry — name → callable lookup for resonance axes.

Each axis (harmonic kernel, ratio kernel, phase estimator, coupling metric,
persistence, combine rule, surrogate type) is a dict-based registry mapping
short string names to callables. The orchestrator dispatches by name; users
configure pipelines by setting strings in :class:`ResonanceConfig`.

Coupling metrics carry an explicit arity tag (``pairwise_symmetric``,
``pairwise_asymmetric``, ``triplet``, ``nary``, ``survey``, ``state``) that the
orchestrator enforces — only pairwise metrics feed the per-bin resonance
spectrum. Higher-order metrics run on a separate code path and are not valid
values for ``ResonanceConfig.coupling_metric``.
"""

from typing import Callable, Dict


HARMONIC_KERNELS: Dict[str, Callable] = {}
RATIO_KERNELS: Dict[str, Callable] = {}
PHASE_ESTIMATORS: Dict[str, Callable] = {}

PAIRWISE_COUPLING_METRICS: Dict[str, Callable] = {}
HIGHER_ORDER_COUPLING_METHODS: Dict[str, Callable] = {}
COUPLING_ARITY: Dict[str, str] = {}

# Input-type tag per metric: 'phase' = takes (phase_i, phase_j, n, m) where
# phase_* are real-valued instantaneous phase time series; 'analytic' = takes
# (analytic_i, analytic_j, n, m) where analytic_* are complex-valued analytic
# signals (e.g., from Hilbert transform or STFT) carrying both amplitude and
# phase information. The orchestrator and peak-based connectivity wrapper use
# this tag to decide whether to pass phase angles or full complex coefficients.
COUPLING_INPUT_TYPE: Dict[str, str] = {}

PERSISTENCE_METHODS: Dict[str, Callable] = {}
COMBINE_RULES: Dict[str, Callable] = {}
SURROGATE_TYPES: Dict[str, Callable] = {}


def register_harmonic_kernel(name: str, fn: Callable) -> None:
    HARMONIC_KERNELS[name] = fn


def register_ratio_kernel(name: str, fn: Callable) -> None:
    RATIO_KERNELS[name] = fn


def register_phase_estimator(name: str, fn: Callable) -> None:
    PHASE_ESTIMATORS[name] = fn


def register_coupling_metric(name: str, fn: Callable, arity: str, input_type: str = "phase") -> None:
    """Register a coupling metric with its arity tag and input-type tag.

    arity must be one of:
        'pairwise_symmetric', 'pairwise_asymmetric' (valid for coupling_metric)
        'triplet', 'nary', 'survey', 'state' (valid for higher_order_coupling)

    input_type must be one of:
        'phase'    — fn(phase_i, phase_j, n, m) on real-valued phase angles
        'analytic' — fn(analytic_i, analytic_j, n, m) on complex analytic signals
    """
    valid_arity = {
        "pairwise_symmetric",
        "pairwise_asymmetric",
        "triplet",
        "nary",
        "survey",
        "state",
    }
    if arity not in valid_arity:
        raise ValueError(f"arity must be one of {valid_arity}, got {arity!r}")
    valid_input = {"phase", "analytic"}
    if input_type not in valid_input:
        raise ValueError(f"input_type must be one of {valid_input}, got {input_type!r}")
    COUPLING_ARITY[name] = arity
    COUPLING_INPUT_TYPE[name] = input_type
    if arity.startswith("pairwise"):
        PAIRWISE_COUPLING_METRICS[name] = fn
    else:
        HIGHER_ORDER_COUPLING_METHODS[name] = fn


def register_persistence(name: str, fn: Callable) -> None:
    PERSISTENCE_METHODS[name] = fn


def register_combine_rule(name: str, fn: Callable) -> None:
    COMBINE_RULES[name] = fn


def register_surrogate_type(name: str, fn: Callable) -> None:
    SURROGATE_TYPES[name] = fn


def is_pairwise(metric_name: str) -> bool:
    return COUPLING_ARITY.get(metric_name, "").startswith("pairwise")


def list_strategies(verbose: bool = True) -> Dict[str, Dict[str, Callable]]:
    """Discover the strategies currently registered in the resonance package.

    Returns a dict keyed by registry name (``'HARMONIC_KERNELS'``,
    ``'PAIRWISE_COUPLING_METRICS'``, etc.). When ``verbose=True`` (the
    default), also prints a friendly summary.

    Use this as the first stop when starting a new analysis::

        >>> from biotuner.resonance import list_strategies
        >>> list_strategies()
        HARMONIC_KERNELS (2)
          - harmsim
          - subharm_tension
        RATIO_KERNELS (2)
          - binary
          - fraction
        ...

    All names returned here are valid for the corresponding
    :class:`ResonanceConfig` field (e.g. ``ResonanceConfig(harmonic_kernel=...)``).
    """
    registries = {
        "HARMONIC_KERNELS": HARMONIC_KERNELS,
        "RATIO_KERNELS": RATIO_KERNELS,
        "PHASE_ESTIMATORS": PHASE_ESTIMATORS,
        "PAIRWISE_COUPLING_METRICS": PAIRWISE_COUPLING_METRICS,
        "HIGHER_ORDER_COUPLING_METHODS": HIGHER_ORDER_COUPLING_METHODS,
        "PERSISTENCE_METHODS": PERSISTENCE_METHODS,
        "COMBINE_RULES": COMBINE_RULES,
        "SURROGATE_TYPES": SURROGATE_TYPES,
    }
    if verbose:
        for reg_name, reg in registries.items():
            names = sorted(reg)
            print(f"{reg_name} ({len(names)})")
            for n in names:
                # For coupling metrics, also show the input-type tag
                if reg_name == "PAIRWISE_COUPLING_METRICS" and n in COUPLING_INPUT_TYPE:
                    print(f"  - {n}  [{COUPLING_INPUT_TYPE[n]}]")
                else:
                    print(f"  - {n}")
            if not names:
                print("  (none — see plan §5 for the Phase 2/3 additions)")
            print()
    return registries
