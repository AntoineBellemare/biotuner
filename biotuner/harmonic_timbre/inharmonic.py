"""biotuner.harmonic_timbre.inharmonic — inharmonic partial series.

Module type: Functions

Source partial sets (stretched / compressed / stiff string / gamelan /
custom) for use as the seed of an inharmonic timbre. Combine with
:func:`biotuner.harmonic_timbre.matching.match_sethares` for full
Sethares-style matched-timbre construction against arbitrary tunings.

References
----------
Railsback, O. L. (1938). Scale temperament as applied to piano tuning.
Sethares, W. (2005). Tuning, Timbre, Spectrum, Scale.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from biotuner.harmonic_timbre._utils import amplitude_falloff, normalize_amplitudes
from biotuner.harmonic_timbre.timbre import Timbre


# ---------------------------------------------------------------------------
# Stretched / compressed series
# ---------------------------------------------------------------------------

def stretched_partials(
    n: int,
    *,
    stretch: float = 1.05,
    base_freq: float = 1.0,
) -> np.ndarray:
    """Stretched harmonic series.

    ``f_k = f_0 * k * stretch**(k-1)``  (multiplicative stretch per partial).

    The default ``stretch=1.05`` corresponds roughly to the Railsback
    piano stretch — partials get progressively sharper than the strict
    integer harmonics.
    """
    if n < 1:
        raise ValueError("stretched_partials: n must be ≥ 1")
    k = np.arange(1, n + 1, dtype=np.float64)
    return base_freq * k * np.power(stretch, k - 1)


def compressed_partials(
    n: int,
    *,
    compress: float = 0.95,
    base_freq: float = 1.0,
) -> np.ndarray:
    """Compressed harmonic series — partials grow more slowly than integer."""
    if compress <= 0:
        raise ValueError("compressed_partials: compress must be > 0")
    return stretched_partials(n, stretch=compress, base_freq=base_freq)


# ---------------------------------------------------------------------------
# Stiff-string inharmonicity
# ---------------------------------------------------------------------------

def inharmonic_string(
    n: int,
    *,
    B: float = 1e-4,
    base_freq: float = 1.0,
) -> np.ndarray:
    """Stiff-string inharmonicity: ``f_k = k * f_0 * sqrt(1 + B * k**2)``.

    ``B`` is the inharmonicity coefficient. Piano strings: ~1e-4.
    """
    if n < 1:
        raise ValueError("inharmonic_string: n must be ≥ 1")
    k = np.arange(1, n + 1, dtype=np.float64)
    return k * base_freq * np.sqrt(1.0 + B * (k ** 2))


# ---------------------------------------------------------------------------
# Gamelan instruments — empirical partial ratios
# ---------------------------------------------------------------------------

# Each profile lists ratios of the upper partials to the fundamental,
# truncated to the first few characteristic ones. Sources: averaged
# field measurements reported in Sethares (2005), Schneider (2001).
_GAMELAN_PROFILES: dict[str, tuple[float, ...]] = {
    "saron":   (1.00, 2.76, 5.40, 5.91, 6.99),
    "bonang":  (1.00, 1.52, 3.46, 3.92),
    "gender":  (1.00, 2.07, 4.13, 5.35),
    "kempul":  (1.00, 1.49, 2.13, 3.05, 4.18),
}


def gamelan_partials(
    *,
    instrument: str = "saron",
    base_freq: float = 1.0,
    n: int | None = None,
) -> np.ndarray:
    """Empirically-derived gamelan instrument partial ratios.

    Built-in profiles: ``'saron'``, ``'bonang'``, ``'gender'``, ``'kempul'``.
    """
    if instrument not in _GAMELAN_PROFILES:
        raise ValueError(
            f"gamelan_partials: unknown instrument {instrument!r}. "
            f"Known: {sorted(_GAMELAN_PROFILES)}"
        )
    ratios = np.asarray(_GAMELAN_PROFILES[instrument], dtype=np.float64)
    if n is not None:
        ratios = ratios[:n]
        if ratios.size < n:
            raise ValueError(
                f"gamelan_partials: only {ratios.size} partials available for {instrument!r}"
            )
    return base_freq * ratios


# ---------------------------------------------------------------------------
# Custom partial series
# ---------------------------------------------------------------------------

def custom_partial_series(
    ratios_to_fundamental: Iterable[float],
    *,
    base_freq: float = 1.0,
) -> np.ndarray:
    """User-defined inharmonic partial set.

    Accepts an iterable of ratios relative to the fundamental and returns
    absolute partial frequencies.
    """
    ratios = np.asarray(list(ratios_to_fundamental), dtype=np.float64)
    if ratios.size == 0:
        raise ValueError("custom_partial_series: empty ratios")
    if not np.all(np.isfinite(ratios)) or not np.all(ratios > 0):
        raise ValueError("custom_partial_series: ratios must be finite and > 0")
    return base_freq * ratios


# ---------------------------------------------------------------------------
# Inharmonic timbre constructor
# ---------------------------------------------------------------------------

def inharmonic_timbre(
    partial_series_fn,
    *,
    n: int = 8,
    base_freq: float = 1.0,
    amplitude_falloff_kind: str = "1_over_n",
    fn_kwargs: dict | None = None,
) -> Timbre:
    """Build a Timbre from an inharmonic partial series generator.

    Parameters
    ----------
    partial_series_fn : callable
        One of :func:`stretched_partials`, :func:`compressed_partials`,
        :func:`inharmonic_string`, :func:`gamelan_partials`,
        :func:`custom_partial_series`, or any callable that returns a
        numpy array of partial frequencies given the ``base_freq`` kwarg.
    n : int, default=8
        Number of partials (ignored if ``partial_series_fn`` doesn't take ``n``).
    base_freq : float, default=1.0
    fn_kwargs : dict, optional
        Forwarded to ``partial_series_fn``.
    """
    fn_kwargs = dict(fn_kwargs or {})
    fn_kwargs.setdefault("base_freq", base_freq)
    # Try to pass n if the function accepts it
    try:
        partials = partial_series_fn(n, **fn_kwargs)
    except TypeError:
        partials = partial_series_fn(**fn_kwargs)
    partials = np.asarray(partials, dtype=np.float64)
    amps = amplitude_falloff(partials.size, amplitude_falloff_kind)
    timbre = Timbre(
        partials_hz=partials,
        amplitudes=normalize_amplitudes(amps),
        base_freq=base_freq,
        matching_method="inharmonic",
        metadata={
            "partial_series_fn": getattr(partial_series_fn, "__name__", str(partial_series_fn)),
            "fn_kwargs": {k: v for k, v in fn_kwargs.items() if k != "base_freq"},
        },
    )
    timbre.validate()
    return timbre
