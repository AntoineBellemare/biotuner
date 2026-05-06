"""biotuner.harmonic_timbre.matching — tuning -> timbre algorithms.

Module type: Functions

Thin wrappers over biotuner.scale_construction (Sethares dissonance,
harmonic entropy) and biotuner.metrics (Tenney height, Euler gradus,
dyad similarity). All matching functions accept ratios from any source
(peaks_ratios, peaks_ratios_cons, diss_scale, harm_tuning_scale,
euler_fokker, harmonic-entropy minima, hand-authored lists) — no
preference for a particular biotuner pipeline.

Algorithms
----------
match_direct
    Place partials at exact scale ratios, extended via equave repetition.
match_consonance_weighted
    Place partials at the ratios; weight amplitudes by Biotuner consonance
    metrics. Cheap, deterministic, no optimizer; the default.
match_sethares
    Minimize the integral of biotuner.scale_construction.diss_curve over
    the scale by perturbing partial frequencies. Iterative.
match_harmonic_entropy
    Use biotuner.scale_construction.harmonic_entropy as the objective
    when placing partials.
match_hybrid
    Convex combination of the above objectives.

References
----------
Sethares, W. (2005). Tuning, Timbre, Spectrum, Scale.
Erlich, P. (2002). The forms of tonality (harmonic entropy).
Plomp, R. & Levelt, W. (1965). Tonal consonance and critical bandwidth.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

from biotuner.harmonic_timbre._utils import (
    _coerce_ratios,
    amplitude_falloff,
    normalize_amplitudes,
    ratios_to_partials,
)
from biotuner.harmonic_timbre.timbre import Timbre

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# match_direct
# ---------------------------------------------------------------------------

def match_direct(
    ratios,
    *,
    n_partials: int | None = None,
    base_freq: float = 1.0,
    equave: float = 2.0,
    amplitude_falloff_kind: str = "1_over_n",
    amplitude_falloff_rate: float = 0.5,
) -> Timbre:
    """Place partials at exact scale ratios.

    If ``n_partials`` exceeds the number of unique ratios, the set is
    extended by equave repetition.

    Parameters
    ----------
    ratios : iterable of float
        Ratio set from any source.
    n_partials : int, optional
        Number of partials to produce. Default: ``len(ratios)``.
    base_freq : float, default=1.0
        Reference frequency (Hz) — partial 0 lands at ``base_freq * ratios[0]``.
    equave : float, default=2.0
        Repetition interval used to extend the partial set when
        ``n_partials > len(ratios)``. ``2.0`` for octave; ``3.0`` for
        Bohlen-Pierce-style tritave.
    amplitude_falloff_kind : str, default='1_over_n'
        Amplitude law: ``'1_over_n'``, ``'1_over_n_squared'``, ``'flat'``,
        or ``'exponential'``.
    amplitude_falloff_rate : float, default=0.5
        Decay rate for ``'exponential'``.

    Returns
    -------
    Timbre
        With ``matching_method='direct'`` and ``matched_tuning`` set to the
        input ratios.
    """
    rs = _coerce_ratios(ratios)
    if not rs:
        raise ValueError("match_direct: ratios is empty")
    n = n_partials or len(rs)
    partials = ratios_to_partials(rs, base_freq=base_freq, n_partials=n, equave=equave)
    amps = amplitude_falloff(n, amplitude_falloff_kind, rate=amplitude_falloff_rate)
    timbre = Timbre(
        partials_hz=partials,
        amplitudes=amps,
        base_freq=base_freq,
        matched_tuning=list(rs),
        matching_method="direct",
        metadata={
            "equave": equave,
            "amplitude_falloff": amplitude_falloff_kind,
        },
    )
    timbre.validate()
    return timbre


# ---------------------------------------------------------------------------
# match_consonance_weighted
# ---------------------------------------------------------------------------

def match_consonance_weighted(
    ratios,
    *,
    n_partials: int | None = None,
    base_freq: float = 1.0,
    equave: float = 2.0,
    metric: str = "dyad_similarity",
) -> Timbre:
    """Place partials at ratios; weight amplitudes by a Biotuner consonance metric.

    Default and recommended matching method. Deterministic, fast, no
    optimizer required. Uses Biotuner's existing harmonicity metrics —
    the ratios have usually already been filtered for consonance during
    the Biotuner pipeline, so this is well-aligned aesthetically.

    Parameters
    ----------
    metric : str, default='dyad_similarity'
        ``'dyad_similarity'`` (Gill & Purves) — higher = more consonant.
        ``'consonance'`` — ``compute_consonance(r) = (a+b)/(a*b)``.
        ``'tenney'`` — inverse Tenney height (higher = simpler ratio).
        ``'euler'`` — inverse Euler gradus suavitatis.
    """
    from biotuner.metrics import (
        dyad_similarity,
        compute_consonance,
        tenneyHeight,
        euler,
    )
    from fractions import Fraction

    rs = _coerce_ratios(ratios)
    if not rs:
        raise ValueError("match_consonance_weighted: ratios is empty")
    n = n_partials or len(rs)
    partials = ratios_to_partials(rs, base_freq=base_freq, n_partials=n, equave=equave)

    raw_weights: list[float] = []
    for p in partials:
        r = p / base_freq
        # bring back into the unit equave window for metric stability
        x = r
        while x >= equave:
            x /= equave
        while x < 1.0:
            x *= equave
        if metric == "dyad_similarity":
            w = dyad_similarity(x)
        elif metric == "consonance":
            w = compute_consonance(x)
        elif metric == "tenney":
            # tenneyHeight expects ≥ 2 frequencies; build a (1, x) pair
            try:
                w = 1.0 / max(tenneyHeight([1.0, x], avg=True), 1e-6)
            except Exception:
                w = 1.0
        elif metric == "euler":
            try:
                frac = Fraction(float(x)).limit_denominator(1000)
                w = 1.0 / max(euler(frac.numerator, frac.denominator), 1)
            except Exception:
                w = 1.0
        else:
            raise ValueError(f"match_consonance_weighted: unknown metric {metric!r}")
        raw_weights.append(float(w))

    weights = np.asarray(raw_weights, dtype=np.float64)
    if not np.all(np.isfinite(weights)) or weights.max() <= 0:
        # fall back to flat weights if metric blew up
        weights = np.ones_like(weights)
    amps = normalize_amplitudes(weights, peak=1.0)

    timbre = Timbre(
        partials_hz=partials,
        amplitudes=amps,
        base_freq=base_freq,
        matched_tuning=list(rs),
        matching_method="consonance_weighted",
        metadata={"equave": equave, "metric": metric},
    )
    timbre.validate()
    return timbre


# ---------------------------------------------------------------------------
# match_sethares — wrap biotuner.scale_construction.diss_curve / dissmeasure
# ---------------------------------------------------------------------------

def _dissonance_over_scale(
    partials: np.ndarray,
    amps: np.ndarray,
    ratios: np.ndarray,
) -> float:
    """Sum of dissmeasure over each (1, r) interval, treating ``partials``
    as the source spectrum. Lower is more consonant."""
    from biotuner.scale_construction import dissmeasure

    f = np.asarray(partials, dtype=np.float64)
    a = np.asarray(amps, dtype=np.float64)
    a_pair = np.concatenate([a, a])
    total = 0.0
    for r in ratios:
        if r <= 0:
            continue
        f_pair = np.concatenate([f, r * f])
        total += float(dissmeasure(f_pair, a_pair, "min"))
    return total


def match_sethares(
    ratios,
    *,
    n_partials: int | None = None,
    base_freq: float = 1.0,
    equave: float = 2.0,
    initial_partials: np.ndarray | None = None,
    max_iter: int = 200,
    optimizer: str = "lbfgs",
    perturbation: float = 0.05,
) -> Timbre:
    """Minimize total dissonance over the scale by perturbing partial freqs.

    Starts from the harmonic series (or ``initial_partials``) and shifts
    each partial in log-frequency to minimize the integral of
    :func:`biotuner.scale_construction.dissmeasure` summed over each
    scale interval. Uses :mod:`scipy.optimize.minimize`.

    Parameters
    ----------
    perturbation : float, default=0.05
        Maximum allowed log-frequency shift per partial (in octave units).
        Keeps the optimizer near the harmonic seed; prevents partials
        from sliding into pathological positions.
    optimizer : str, default='lbfgs'
        Currently passed straight to :func:`scipy.optimize.minimize`. Use
        ``'L-BFGS-B'`` (default), ``'Powell'``, ``'Nelder-Mead'``, etc.
    """
    from scipy.optimize import minimize

    rs = _coerce_ratios(ratios)
    if not rs:
        raise ValueError("match_sethares: ratios is empty")
    n = n_partials or max(len(rs), 6)
    if initial_partials is None:
        # harmonic seed: f0 * (1, 2, 3, ...)
        seed = base_freq * np.arange(1, n + 1, dtype=np.float64)
    else:
        seed = np.asarray(initial_partials, dtype=np.float64)
        if seed.shape[0] != n:
            raise ValueError(
                f"initial_partials length {seed.shape[0]} != n_partials {n}"
            )
    amps = amplitude_falloff(n, "1_over_n")

    # parameterize as log2 perturbations bounded by ``perturbation``
    log_seed = np.log2(seed)

    def objective(delta: np.ndarray) -> float:
        partials = np.power(2.0, log_seed + delta)
        return _dissonance_over_scale(partials, amps, np.asarray(rs))

    method = "L-BFGS-B" if optimizer.lower() == "lbfgs" else optimizer
    bounds = [(-perturbation, perturbation)] * n
    result = minimize(
        objective,
        x0=np.zeros(n),
        method=method,
        bounds=bounds if method == "L-BFGS-B" else None,
        options={"maxiter": max_iter},
    )
    final = np.power(2.0, log_seed + result.x)
    timbre = Timbre(
        partials_hz=final,
        amplitudes=amps,
        base_freq=base_freq,
        matched_tuning=list(rs),
        matching_method="sethares",
        metadata={
            "equave": equave,
            "objective_initial": float(objective(np.zeros(n))),
            "objective_final": float(result.fun),
            "n_iter": int(getattr(result, "nit", 0)),
            "perturbation": perturbation,
            "optimizer": method,
        },
    )
    timbre.validate()
    return timbre


# ---------------------------------------------------------------------------
# match_harmonic_entropy — use biotuner.scale_construction.harmonic_entropy
# ---------------------------------------------------------------------------

def match_harmonic_entropy(
    ratios,
    *,
    n_partials: int | None = None,
    base_freq: float = 1.0,
    equave: float = 2.0,
    res: float = 0.001,
    spread: float = 0.01,
) -> Timbre:
    """Place partials and weight by harmonic-entropy minima.

    Uses :func:`biotuner.scale_construction.harmonic_entropy` to assess
    the input ratios; partials are placed at the ratios and amplitudes
    are weighted *inversely* with the per-ratio harmonic entropy values
    (lower entropy -> louder partial).
    """
    from biotuner.scale_construction import harmonic_entropy

    rs = _coerce_ratios(ratios)
    if not rs:
        raise ValueError("match_harmonic_entropy: ratios is empty")
    n = n_partials or len(rs)
    partials = ratios_to_partials(rs, base_freq=base_freq, n_partials=n, equave=equave)

    # harmonic_entropy returns (HE_minima, HE_avg, HE) in Biotuner's contract
    try:
        _, he_avg, he_values = harmonic_entropy(
            list(rs),
            res=res,
            spread=spread,
            plot_entropy=False,
            plot_tenney=False,
            octave=equave,
        )
    except Exception as exc:  # pragma: no cover — depends on input pathology
        logger.warning("harmonic_entropy failed (%s); falling back to flat amps", exc)
        amps = amplitude_falloff(n, "flat")
        he_avg = float("nan")
    else:
        # he_values is one value per fine-grained ratio bin in Biotuner's contract;
        # we want one weight per partial. Use the per-ratio inverse mean entropy.
        # Bin per partial by nearest ratio.
        he_arr = np.asarray(he_values, dtype=np.float64)
        # weight: 1 / (he per partial, taken at the ratio); fall back to falloff
        per_partial: list[float] = []
        for p in partials:
            r = float(p) / float(base_freq)
            x = r
            while x >= equave:
                x /= equave
            while x < 1.0:
                x *= equave
            # map x in [1, equave] onto [0, len(he_arr))
            if he_arr.size == 0:
                per_partial.append(1.0)
                continue
            idx = int(np.clip(round((x - 1.0) / max(equave - 1.0, 1e-9) * (he_arr.size - 1)), 0, he_arr.size - 1))
            v = float(he_arr[idx])
            per_partial.append(1.0 / max(v, 1e-6))
        amps = normalize_amplitudes(np.asarray(per_partial), peak=1.0)

    timbre = Timbre(
        partials_hz=partials,
        amplitudes=amps,
        base_freq=base_freq,
        matched_tuning=list(rs),
        matching_method="harmonic_entropy",
        metadata={"equave": equave, "he_avg": float(he_avg), "res": res, "spread": spread},
    )
    timbre.validate()
    return timbre


# ---------------------------------------------------------------------------
# match_hybrid — convex combination
# ---------------------------------------------------------------------------

def match_hybrid(
    ratios,
    *,
    weights: dict[str, float] | None = None,
    n_partials: int | None = None,
    base_freq: float = 1.0,
    equave: float = 2.0,
) -> Timbre:
    """Convex combination of direct + sethares + harmonic_entropy + consonance_weighted.

    Parameters
    ----------
    weights : dict, optional
        Mapping from method name (``'direct'``, ``'sethares'``,
        ``'harmonic_entropy'``, ``'consonance_weighted'``) to weight in
        ``[0, 1]``. Will be normalized to sum to 1. Default:
        ``{'consonance_weighted': 0.5, 'sethares': 0.3, 'direct': 0.2}``.
    """
    if weights is None:
        weights = {"consonance_weighted": 0.5, "sethares": 0.3, "direct": 0.2}
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("match_hybrid: weights sum to zero")
    w = {k: v / total for k, v in weights.items()}

    timbres: list[tuple[float, Timbre]] = []
    if w.get("direct", 0) > 0:
        timbres.append((w["direct"], match_direct(
            ratios, n_partials=n_partials, base_freq=base_freq, equave=equave,
        )))
    if w.get("consonance_weighted", 0) > 0:
        timbres.append((w["consonance_weighted"], match_consonance_weighted(
            ratios, n_partials=n_partials, base_freq=base_freq, equave=equave,
        )))
    if w.get("sethares", 0) > 0:
        timbres.append((w["sethares"], match_sethares(
            ratios, n_partials=n_partials, base_freq=base_freq, equave=equave,
        )))
    if w.get("harmonic_entropy", 0) > 0:
        timbres.append((w["harmonic_entropy"], match_harmonic_entropy(
            ratios, n_partials=n_partials, base_freq=base_freq, equave=equave,
        )))
    if not timbres:
        raise ValueError("match_hybrid: no positive weights")

    # Combine: weighted average of partials_hz (in log freq) and amplitudes.
    # All component timbres share the same n_partials by construction.
    n = timbres[0][1].n_partials()
    log_partials = np.zeros(n, dtype=np.float64)
    amps = np.zeros(n, dtype=np.float64)
    for weight, t in timbres:
        if t.n_partials() != n:
            raise ValueError(
                "match_hybrid: component timbres have mismatched n_partials"
            )
        log_partials += weight * np.log2(t.partials_hz)
        amps += weight * t.amplitudes
    partials = np.power(2.0, log_partials)

    timbre = Timbre(
        partials_hz=partials,
        amplitudes=normalize_amplitudes(amps),
        base_freq=base_freq,
        matched_tuning=list(_coerce_ratios(ratios)),
        matching_method="hybrid",
        metadata={"equave": equave, "weights": w},
    )
    timbre.validate()
    return timbre


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------

_METHOD_TO_FUNC = {
    "direct": match_direct,
    "consonance_weighted": match_consonance_weighted,
    "sethares": match_sethares,
    "harmonic_entropy": match_harmonic_entropy,
    "hybrid": match_hybrid,
}


def match_timbre(
    ratios,
    method: str = "consonance_weighted",
    **kwargs,
) -> Timbre:
    """Single entry point — dispatch on ``method``.

    method
        ``'direct'``, ``'consonance_weighted'`` (default), ``'sethares'``,
        ``'harmonic_entropy'``, or ``'hybrid'``.
    **kwargs
        Forwarded to the matching function.
    """
    fn = _METHOD_TO_FUNC.get(method)
    if fn is None:
        raise ValueError(
            f"Unknown matching method {method!r}. Known: {sorted(_METHOD_TO_FUNC)}"
        )
    return fn(ratios, **kwargs)
