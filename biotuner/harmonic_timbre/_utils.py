"""Shared internal helpers for ``biotuner.harmonic_timbre``.

This module owns the *scale source abstraction* — the layer that lets every
public entry point in ``harmonic_timbre`` accept ratios from any Biotuner
ratio attribute, any constructed scale (dissonance minima, harmonic tuning,
Euler-Fokker, harmonic entropy), or raw user-supplied lists.

Identical ``scale`` selection across ``timbre_from_biotuner`` and
``rhythm_from_biotuner`` (Phase 3) is what guarantees the time-scale
projection invariant: timbre and rhythm anchored to the same ratios.
"""

from __future__ import annotations

import logging
from fractions import Fraction
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scale source registry
# ---------------------------------------------------------------------------

# Each value is a (kind, target) pair where:
#   kind == 'attr'         : direct attribute read on the biotuner instance
#   kind == 'attr_or_call' : prefer attribute; fall back to method call
#   kind == 'callable'     : always call the named method
#   kind == 'sequence'     : sequence-valued source (transitional / Markov)
#   kind == 'raw'          : caller supplies ratios directly
SCALE_SOURCES: dict[str, tuple[str, str | None]] = {
    # peaks-derived
    "peaks_ratios":               ("attr", "peaks_ratios"),
    "peaks_ratios_cons":          ("attr", "peaks_ratios_cons"),
    "peaks_ratios_harm":          ("attr", "peaks_ratios_harm"),
    "peaks_ratios_inc":           ("attr", "peaks_ratios_inc"),
    "peaks_ratios_inc_fit":       ("attr", "peaks_ratios_inc_fit"),
    "extended_peaks_ratios":      ("attr", "extended_peaks_ratios"),
    "extended_peaks_ratios_cons": ("attr", "extended_peaks_ratios_cons"),
    # constructed scales
    "diss_scale":                 ("attr_or_call", "compute_diss_curve"),
    "diss_scale_cons":            ("attr_or_call", "compute_diss_curve"),
    "harm_tuning_scale":          ("attr_or_call", "harmonic_tuning"),
    "harm_fit_tuning_scale":      ("attr_or_call", "harmonic_fit_tuning"),
    "euler_fokker":               ("attr_or_call", "euler_fokker_scale"),
    "harmonic_entropy":           ("callable", "compute_harmonic_entropy_scale"),
    # sequence sources (handled in Phase 3)
    "transitional_harmony":       ("sequence", "transitional_harmony"),
    "harmonic_sequence_markov":   ("sequence", "harmonic_sequence"),
    # raw / user-supplied
    "raw":                        ("raw", None),
}

# Default fallback chain when a ``scale`` arg is None
DEFAULT_SCALE_FALLBACK_CHAIN: tuple[str, ...] = (
    "peaks_ratios_cons",
    "peaks_ratios",
    "extended_peaks_ratios_cons",
    "extended_peaks_ratios",
    "diss_scale_cons",
    "diss_scale",
    "harm_tuning_scale",
    "euler_fokker",
)


def _coerce_ratios(ratios) -> list[float]:
    """Coerce an iterable of ratio-like values to a list of floats.

    Accepts ``Fraction``, ``sympy.Rational``, numpy scalars, and ints/floats.
    """
    if ratios is None:
        return []
    out: list[float] = []
    for r in ratios:
        if isinstance(r, Fraction):
            out.append(float(r))
        else:
            try:
                out.append(float(r))
            except (TypeError, ValueError):
                # sympy.Rational and similar
                out.append(float(r.evalf()) if hasattr(r, "evalf") else float(r))
    return out


def resolve_scale(
    source,
    *,
    bt=None,
    ratios=None,
    call_kwargs: dict | None = None,
    strict: bool = False,
) -> tuple[list[float], str]:
    """Resolve a scale-source selector to a concrete list of ratios.

    Parameters
    ----------
    source : str | Iterable[float] | None
        One of the keys in ``SCALE_SOURCES`` (e.g. ``'peaks_ratios_cons'``,
        ``'diss_scale'``, ``'euler_fokker'``); or an iterable of ratio-like
        values (treated as raw); or ``None`` to walk
        ``DEFAULT_SCALE_FALLBACK_CHAIN``.
    bt : object | None
        A fitted ``compute_biotuner`` instance. Required when ``source`` is a
        SCALE_SOURCES key (other than ``'raw'``).
    ratios : Iterable[float] | None
        Required when ``source == 'raw'`` (used together with the string form
        for symmetric APIs).
    call_kwargs : dict | None
        Extra kwargs forwarded to method-backed backends.
    strict : bool, default=False
        When False, attribute-or-call backends will lazily invoke the
        producing method on ``bt`` if the attribute is unset. When True,
        a missing attribute raises.

    Returns
    -------
    ratios : list of float
        The resolved ratio set.
    provenance : str
        A label suitable for ``Timbre.metadata['scale_source']``.

    Raises
    ------
    KeyError
        ``source`` is a string but not in ``SCALE_SOURCES``.
    ValueError
        Required arguments are missing or the resolved scale is empty.
    """
    call_kwargs = dict(call_kwargs or {})

    # Iterable directly => raw ratios
    if source is None:
        if bt is None:
            raise ValueError(
                "resolve_scale: source=None requires a biotuner instance to walk the fallback chain"
            )
        for candidate in DEFAULT_SCALE_FALLBACK_CHAIN:
            try:
                resolved, prov = resolve_scale(
                    candidate, bt=bt, call_kwargs=call_kwargs, strict=True
                )
                if len(resolved) > 0:
                    return resolved, prov
            except (AttributeError, ValueError, KeyError):
                continue
        raise ValueError(
            "resolve_scale: none of the fallback ratio sources were available on bt"
        )

    if not isinstance(source, str):
        # treat as raw iterable
        rs = _coerce_ratios(source)
        if not rs:
            raise ValueError("resolve_scale: empty raw ratios")
        return rs, "raw"

    if source not in SCALE_SOURCES:
        raise KeyError(
            f"Unknown scale source {source!r}. Known: {sorted(SCALE_SOURCES)}"
        )

    kind, target = SCALE_SOURCES[source]

    if kind == "raw":
        if ratios is None:
            raise ValueError(
                "resolve_scale: source='raw' requires ratios=... to be supplied"
            )
        rs = _coerce_ratios(ratios)
        if not rs:
            raise ValueError("resolve_scale: empty raw ratios")
        return rs, "raw"

    if bt is None:
        raise ValueError(
            f"resolve_scale: scale source {source!r} requires a biotuner instance"
        )

    if kind == "attr":
        return _read_attr_strict(bt, source, label=source)

    if kind == "attr_or_call":
        # try direct attribute first, then call producer
        if hasattr(bt, source) and getattr(bt, source) is not None:
            try:
                return _read_attr_strict(bt, source, label=source)
            except ValueError:
                pass  # fall through to lazy call
        if strict:
            raise AttributeError(
                f"resolve_scale: attribute {source!r} not populated on bt and strict=True"
            )
        method = getattr(bt, target, None)
        if method is None:
            raise AttributeError(
                f"resolve_scale: producer method {target!r} missing on bt"
            )
        logger.debug(
            "resolve_scale: lazily invoking %s.%s to populate %s", type(bt).__name__,
            target, source,
        )
        try:
            method(**call_kwargs)
        except TypeError:
            method()  # producer has no kwargs in some branches
        return _read_attr_strict(bt, source, label=source)

    if kind == "callable":
        method = getattr(bt, target, None)
        if method is None:
            raise AttributeError(
                f"resolve_scale: callable backend {target!r} missing on bt"
            )
        result = method(**call_kwargs)
        rs = _coerce_ratios(result)
        if not rs:
            raise ValueError(f"resolve_scale: callable {target!r} returned empty scale")
        return rs, source

    if kind == "sequence":
        # Phase 1 deliberately does not implement sequence resolution; defer to Phase 3.
        raise NotImplementedError(
            f"Scale source {source!r} is sequence-valued; use the dedicated "
            "TimbreSequence constructor (planned for v1.1)."
        )

    raise RuntimeError(f"resolve_scale: unhandled kind {kind!r}")


def _read_attr_strict(bt, attr: str, *, label: str) -> tuple[list[float], str]:
    """Read ``attr`` off ``bt`` and coerce to a ratio list. Raise if empty/missing."""
    if not hasattr(bt, attr):
        raise AttributeError(f"biotuner instance has no attribute {attr!r}")
    raw = getattr(bt, attr)
    if raw is None:
        raise ValueError(f"biotuner attribute {attr!r} is None")
    rs = _coerce_ratios(raw)
    if not rs:
        raise ValueError(f"biotuner attribute {attr!r} is empty")
    return rs, label


# ---------------------------------------------------------------------------
# Frequency/ratio math helpers
# ---------------------------------------------------------------------------

def normalize_ratios(ratios, *, equave: float = 2.0) -> list[float]:
    """Reduce ratios into the unit equave window ``[1, equave)`` and sort.

    Useful when stitching together ratios from heterogeneous sources.
    """
    out: list[float] = []
    for r in _coerce_ratios(ratios):
        if r <= 0:
            continue
        x = r
        while x >= equave:
            x /= equave
        while x < 1.0:
            x *= equave
        out.append(x)
    return sorted(set(out))


def ratios_to_partials(
    ratios,
    *,
    base_freq: float = 1.0,
    n_partials: int | None = None,
    equave: float = 2.0,
) -> np.ndarray:
    """Project ratios onto absolute partial frequencies.

    If ``n_partials`` exceeds the number of unique ratios, the set is
    extended by equave repetition (``r``, ``r·equave``, ``r·equave²``, …).
    """
    rs = _coerce_ratios(ratios)
    if not rs:
        raise ValueError("ratios_to_partials: empty ratios")
    n_partials = n_partials or len(rs)
    out: list[float] = []
    k = 0
    base_set = sorted(rs)
    while len(out) < n_partials:
        for r in base_set:
            out.append(r * (equave ** k) * base_freq)
            if len(out) >= n_partials:
                break
        k += 1
    return np.asarray(out[:n_partials], dtype=np.float64)


def amplitude_falloff(
    n: int,
    kind: str = "1_over_n",
    *,
    rate: float = 0.5,
) -> np.ndarray:
    """Return an amplitude vector of length ``n`` per a falloff law.

    Parameters
    ----------
    kind : str
        ``'1_over_n'`` (default), ``'1_over_n_squared'``, ``'flat'``,
        ``'exponential'``.
    rate : float
        Decay rate for ``'exponential'`` (per partial, applied as
        ``exp(-rate · k)``). Ignored otherwise.
    """
    idx = np.arange(1, n + 1, dtype=np.float64)
    if kind == "1_over_n":
        return 1.0 / idx
    if kind == "1_over_n_squared":
        return 1.0 / (idx ** 2)
    if kind == "flat":
        return np.ones(n, dtype=np.float64)
    if kind == "exponential":
        return np.exp(-rate * (idx - 1))
    raise ValueError(f"amplitude_falloff: unknown kind {kind!r}")


def normalize_amplitudes(amps, *, peak: float = 1.0) -> np.ndarray:
    a = np.asarray(amps, dtype=np.float64)
    if a.size == 0:
        return a
    m = float(np.max(np.abs(a)))
    if m <= 0:
        return a
    return a * (peak / m)
