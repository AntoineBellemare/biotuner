"""biotuner.resonance.combine — combine rules for stacking factors into resonance.

Each rule takes a list of length-N arrays (the factors: H, PC, optionally Q) and
returns a single length-N resonance spectrum. The legacy default is ``product``;
plan §5.7 adds geomean, harmmean, min, weighted_log for downstream experimentation.
"""

import numpy as np

from biotuner.resonance.registry import register_combine_rule


def product(factors, **_unused):
    """np.prod(factors, axis=0). Legacy default."""
    return np.prod(np.asarray(factors, dtype=np.float64), axis=0)


def geomean(factors, **_unused):
    """Geometric mean: same zeros as product, but linear-scale interpretation."""
    arr = np.asarray(factors, dtype=np.float64)
    return np.prod(arr, axis=0) ** (1.0 / arr.shape[0])


def harmmean(factors, **_unused):
    """Harmonic mean: penalizes asymmetry between factors."""
    arr = np.asarray(factors, dtype=np.float64)
    safe = np.where(arr > 0, arr, np.inf)
    return arr.shape[0] / np.sum(1.0 / safe, axis=0)


def min_combine(factors, **_unused):
    """np.min(factors, axis=0). 'Bottleneck' reading."""
    return np.min(np.asarray(factors, dtype=np.float64), axis=0)


def weighted_log(factors, weights=None, **_unused):
    """Generalized geometric mean: exp(Σ w_i * log(f_i))."""
    arr = np.asarray(factors, dtype=np.float64)
    if weights is None:
        weights = np.ones(arr.shape[0]) / arr.shape[0]
    weights = np.asarray(weights, dtype=np.float64).reshape(-1, 1)
    log_arr = np.log(np.where(arr > 0, arr, 1e-300))
    return np.exp(np.sum(weights * log_arr, axis=0))


register_combine_rule("product", product)
register_combine_rule("geomean", geomean)
register_combine_rule("harmmean", harmmean)
register_combine_rule("min", min_combine)
register_combine_rule("weighted_log", weighted_log)
