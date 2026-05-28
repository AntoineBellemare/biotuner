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

PERSISTENCE_METHODS: Dict[str, Callable] = {}
COMBINE_RULES: Dict[str, Callable] = {}
SURROGATE_TYPES: Dict[str, Callable] = {}


def register_harmonic_kernel(name: str, fn: Callable) -> None:
    HARMONIC_KERNELS[name] = fn


def register_ratio_kernel(name: str, fn: Callable) -> None:
    RATIO_KERNELS[name] = fn


def register_phase_estimator(name: str, fn: Callable) -> None:
    PHASE_ESTIMATORS[name] = fn


def register_coupling_metric(name: str, fn: Callable, arity: str) -> None:
    """Register a coupling metric with its arity tag.

    arity must be one of:
        'pairwise_symmetric', 'pairwise_asymmetric' (valid for coupling_metric)
        'triplet', 'nary', 'survey', 'state' (valid for higher_order_coupling)
    """
    valid = {
        "pairwise_symmetric",
        "pairwise_asymmetric",
        "triplet",
        "nary",
        "survey",
        "state",
    }
    if arity not in valid:
        raise ValueError(f"arity must be one of {valid}, got {arity!r}")
    COUPLING_ARITY[name] = arity
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
