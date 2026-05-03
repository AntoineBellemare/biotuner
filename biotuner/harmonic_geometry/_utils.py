"""
Internal helpers for :mod:`biotuner.harmonic_geometry`.

These utilities are not part of the public API. They handle ratio
normalization, coprime checks, and log-space conversions used across the
geometry submodules.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np

# A "ratio-like" input may be a Fraction, an int, a float, or an (a, b) pair.
RatioLike = Union[Fraction, int, float, Tuple[int, int]]

# Maximum denominator used when converting a float to a Fraction. Large enough
# to capture musical ratios with several decimal digits of precision but small
# enough to keep coprime / Stern-Brocot computations cheap.
DEFAULT_MAX_DENOMINATOR = 10_000


def coerce_ratio(
    r: RatioLike,
    max_denominator: int = DEFAULT_MAX_DENOMINATOR,
) -> Union[Fraction, float]:
    """Coerce a ratio-like value to a :class:`~fractions.Fraction` when rational.

    Integers and ``(num, den)`` pairs become exact Fractions. Floats become
    Fractions if a close rational approximation exists within
    ``max_denominator``; otherwise the float is returned unchanged.

    Parameters
    ----------
    r : Fraction, int, float, or (int, int)
        Value to coerce.
    max_denominator : int, default=10000
        Cap on the denominator used when approximating a float.

    Returns
    -------
    Fraction or float
        The canonicalized value.
    """
    if isinstance(r, Fraction):
        return r
    if isinstance(r, tuple):
        if len(r) != 2:
            raise ValueError(f"Tuple ratio must have exactly two elements, got {r!r}.")
        num, den = r
        return Fraction(int(num), int(den))
    if isinstance(r, (int, np.integer)):
        return Fraction(int(r), 1)
    if isinstance(r, (float, np.floating)):
        if not np.isfinite(r):
            raise ValueError(f"Ratio must be finite, got {r!r}.")
        return Fraction(float(r)).limit_denominator(max_denominator)
    raise TypeError(f"Unsupported ratio type: {type(r).__name__}")


def coerce_ratios(
    ratios: Iterable[RatioLike],
    max_denominator: int = DEFAULT_MAX_DENOMINATOR,
) -> List[Union[Fraction, float]]:
    """Apply :func:`coerce_ratio` element-wise."""
    return [coerce_ratio(r, max_denominator=max_denominator) for r in ratios]


def ratios_to_floats(ratios: Sequence[Union[Fraction, float]]) -> np.ndarray:
    """Convert a list of ratios to a 1-D ``float64`` array."""
    return np.asarray([float(r) for r in ratios], dtype=np.float64)


def is_coprime(a: int, b: int) -> bool:
    """Return ``True`` if ``gcd(a, b) == 1``. Both arguments must be non-zero."""
    if a == 0 or b == 0:
        return False
    return math.gcd(int(abs(a)), int(abs(b))) == 1


def coprime_pair(r: Union[Fraction, float], max_denominator: int = DEFAULT_MAX_DENOMINATOR) -> Tuple[int, int]:
    """Return a coprime ``(numerator, denominator)`` pair for a ratio.

    Floats are first converted to a Fraction via :func:`coerce_ratio`. The
    returned tuple satisfies ``gcd(num, den) == 1`` and ``den > 0``.
    """
    frac = coerce_ratio(r, max_denominator=max_denominator)
    if isinstance(frac, float):
        frac = Fraction(frac).limit_denominator(max_denominator)
    return frac.numerator, frac.denominator


def log_ratio(r: Union[Fraction, float], equave: float = 2.0) -> float:
    """Logarithm of a ratio in the given equave (octave by default).

    Returns ``log_equave(r) = log(r) / log(equave)``. Pitch-class is the
    fractional part of this value.
    """
    if equave <= 1.0:
        raise ValueError(f"equave must be > 1, got {equave!r}.")
    val = float(r)
    if val <= 0.0:
        raise ValueError(f"Ratio must be > 0 to take a log, got {val!r}.")
    return math.log(val) / math.log(equave)


def equave_reduce(r: Union[Fraction, float], equave: float = 2.0) -> Union[Fraction, float]:
    """Reduce a positive ratio into the fundamental equave ``[1, equave)``.

    Equivalent to repeated multiplication or division by ``equave`` until the
    result lies in the half-open interval ``[1, equave)``.
    """
    if equave <= 1.0:
        raise ValueError(f"equave must be > 1, got {equave!r}.")
    val = float(r)
    if val <= 0.0:
        raise ValueError(f"Ratio must be > 0, got {val!r}.")

    if isinstance(r, Fraction):
        # Keep exact arithmetic when possible. equave may be irrational; if so
        # we fall back to float reduction.
        if equave == int(equave):
            eq = Fraction(int(equave), 1)
            cur = r
            while cur >= eq:
                cur = cur / eq
            while cur < 1:
                cur = cur * eq
            return cur

    # Float path.
    log_e = math.log(equave)
    n = math.floor(math.log(val) / log_e)
    return val / (equave ** n)


def normalize_amplitudes(amps: Sequence[float]) -> np.ndarray:
    """Return a copy of ``amps`` scaled to sum to 1.

    If the sum is zero, returns a uniform distribution.
    """
    arr = np.asarray(amps, dtype=np.float64)
    total = float(arr.sum())
    if total <= 0.0 or not np.isfinite(total):
        if arr.size == 0:
            return arr
        return np.full_like(arr, 1.0 / arr.size)
    return arr / total


def check_lengths_match(n_components: int, **named_seqs) -> None:
    """Raise ``ValueError`` if any provided sequence's length differs from ``n_components``.

    ``None`` values are skipped.
    """
    for name, seq in named_seqs.items():
        if seq is None:
            continue
        if len(seq) != n_components:
            raise ValueError(
                f"{name!r} has length {len(seq)} but expected {n_components} "
                "(must match n_components)."
            )
