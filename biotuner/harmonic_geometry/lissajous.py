"""
Lissajous-curve geometry.

Standard 2-D form: ``x(t) = A_x · sin(a · t + δ)``, ``y(t) = A_y · sin(b · t)``.

For coprime integer frequency ratios ``(a, b)`` and arbitrary phase ``δ``, the
curve closes after ``t ∈ [0, 2π]``. A pairwise-coprime triple ``(a, b, c)``
applied to the three Cartesian axes yields a Lissajous knot.

References
----------
.. [1] Bowditch, N. (1815). On the motion of a pendulum suspended from two
       points.
.. [2] Lissajous, J. A. (1857). Mémoire sur l'étude optique des mouvements
       vibratoires.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from biotuner.harmonic_geometry._utils import coerce_ratio, coprime_pair, is_coprime
from biotuner.harmonic_geometry.geometry_data import GeometryData
from biotuner.harmonic_geometry.inputs import HarmonicInput

RatioLike = Union[Fraction, int, float, Tuple[int, int]]


# ---------------------------------------------------------------- 2-D Lissajous


def lissajous_2d(
    ratio: RatioLike,
    phase: float = math.pi / 2,
    amps: Tuple[float, float] = (1.0, 1.0),
    n_points: int = 1000,
    n_periods: int = 1,
) -> GeometryData:
    """A single 2-D Lissajous curve.

    Samples ``x(t) = A_x · sin(a · t + phase)`` and ``y(t) = A_y · sin(b · t)``
    over ``t ∈ [0, 2π · n_periods]``, where ``(a, b)`` is a coprime
    representation of ``ratio``.

    Parameters
    ----------
    ratio : Fraction, int, float, or (int, int)
        Frequency ratio ``a / b`` of the x-component to the y-component.
    phase : float, default=π/2
        Phase shift ``δ`` in radians applied to the x-component.
    amps : tuple of float, default=(1.0, 1.0)
        Amplitudes ``(A_x, A_y)``.
    n_points : int, default=1000
        Number of samples along the curve.
    n_periods : int, default=1
        Number of fundamental periods to sample. For coprime ``(a, b)`` the
        curve closes after one period.

    Returns
    -------
    GeometryData
        ``geom_type='curve_2d'`` with shape ``(n_points, 2)``. Metadata
        includes the coprime ``(a, b)`` pair, closure flag, and phase.
    """
    a, b = coprime_pair(ratio)
    A_x, A_y = float(amps[0]), float(amps[1])

    t = np.linspace(0.0, 2.0 * np.pi * n_periods, n_points)
    x = A_x * np.sin(a * t + phase)
    y = A_y * np.sin(b * t)
    coords = np.stack([x, y], axis=1)

    return GeometryData(
        geom_type="curve_2d",
        coordinates=coords,
        parameters={
            "ratio": Fraction(a, b),
            "phase": float(phase),
            "amps": (A_x, A_y),
            "n_points": int(n_points),
            "n_periods": int(n_periods),
        },
        metadata={
            "kind": "lissajous_2d",
            "a": a,
            "b": b,
            "closed": is_coprime(a, b),
            "lobes_x": a,
            "lobes_y": b,
        },
    )


# ---------------------------------------------------------------- 3-D Lissajous


def lissajous_3d(
    ratios: Sequence[RatioLike],
    phases: Sequence[float] = (0.0, 0.0, 0.0),
    amps: Sequence[float] = (1.0, 1.0, 1.0),
    n_points: int = 2000,
) -> GeometryData:
    """A 3-D Lissajous curve.

    Samples ``x_i(t) = A_i · sin(f_i · t + φ_i)`` for ``i ∈ {0, 1, 2}``,
    where ``f_i`` is a coprime integer derived from ``ratios[i]``.

    When all three ``f_i`` are pairwise coprime the resulting curve is a
    *Lissajous knot*; this is flagged in ``metadata['knot']``.

    Parameters
    ----------
    ratios : sequence of length 3
        Frequencies for x, y, z.
    phases : sequence of length 3, default=(0, 0, 0)
        Phase per axis in radians.
    amps : sequence of length 3, default=(1, 1, 1)
        Amplitude per axis.
    n_points : int, default=2000

    Returns
    -------
    GeometryData
        ``geom_type='curve_3d'`` with shape ``(n_points, 3)``.
    """
    if len(ratios) != 3 or len(phases) != 3 or len(amps) != 3:
        raise ValueError("lissajous_3d requires exactly 3 ratios, phases, and amps.")

    freqs: List[int] = []
    dens: List[int] = []
    for r in ratios:
        num, den = coprime_pair(r)
        freqs.append(num)
        dens.append(den)
    # Common period requires using a common denominator. We sample over enough
    # time that each component completes its own periodicity at least once.
    lcm_den = 1
    for d in dens:
        lcm_den = lcm_den * d // math.gcd(lcm_den, d)
    t = np.linspace(0.0, 2.0 * np.pi * lcm_den, n_points)

    coords = np.empty((n_points, 3), dtype=np.float64)
    for i, (num, den) in enumerate(zip(freqs, dens)):
        # Effective frequency along axis i is num/den; sampling over
        # 2π * lcm_den ensures closure when components are commensurate.
        coords[:, i] = float(amps[i]) * np.sin(
            (num / den) * t + float(phases[i])
        )

    pairwise_coprime = all(
        is_coprime(freqs[i], freqs[j])
        for i in range(3)
        for j in range(i + 1, 3)
    )

    return GeometryData(
        geom_type="curve_3d",
        coordinates=coords,
        parameters={
            "ratios": [Fraction(n, d) for n, d in zip(freqs, dens)],
            "phases": tuple(float(p) for p in phases),
            "amps": tuple(float(a) for a in amps),
            "n_points": int(n_points),
        },
        metadata={
            "kind": "lissajous_3d",
            "freqs": tuple(freqs),
            "denominators": tuple(dens),
            "knot": pairwise_coprime,
        },
    )


# -------------------------------------------------------------- input adapters


def lissajous_pairwise_grid(
    input: HarmonicInput,
    n_points: int = 500,
    phase: float = math.pi / 2,
) -> List[List[GeometryData]]:
    """Build a 2-D grid of pairwise Lissajous curves from a HarmonicInput.

    For an input with N components, returns an N×N nested list where entry
    ``[i][j]`` is the 2-D Lissajous of component i (x-axis) against
    component j (y-axis). The diagonal contains 1:1 unison curves.

    Parameters
    ----------
    input : HarmonicInput
    n_points : int, default=500
    phase : float, default=π/2

    Returns
    -------
    list of list of GeometryData
        ``N × N`` matrix of ``curve_2d`` geometries.
    """
    peaks = input.to_peaks()
    amps = input.normalized_amplitudes()
    n = len(peaks)
    grid: List[List[GeometryData]] = []
    for i in range(n):
        row: List[GeometryData] = []
        for j in range(n):
            ratio = peaks[i] / peaks[j] if peaks[j] != 0 else 1.0
            row.append(
                lissajous_2d(
                    ratio,
                    phase=phase,
                    amps=(float(amps[i]) + 1e-12, float(amps[j]) + 1e-12),
                    n_points=n_points,
                )
            )
        grid.append(row)
    return grid


def lissajous_compound(
    input: HarmonicInput,
    n_points: int = 2000,
    n_periods: int = 1,
) -> GeometryData:
    """Sum-of-sinusoids Lissajous from a HarmonicInput.

    Treats the input components as a chord. The x-coordinate is the sum of
    all components phase-shifted by ``π/2``; the y-coordinate is the sum
    without the phase shift. This collapses an N-component HarmonicInput
    into a single 2-D Lissajous-like curve.

    Parameters
    ----------
    input : HarmonicInput
    n_points : int, default=2000
    n_periods : int, default=1

    Returns
    -------
    GeometryData
        ``geom_type='curve_2d'``.
    """
    peaks = input.to_peaks()
    amps = input.normalized_amplitudes()
    phases = (
        np.asarray(input.phases, dtype=np.float64)
        if input.phases is not None
        else np.zeros(len(peaks))
    )
    base = float(np.min(peaks)) if len(peaks) else 1.0
    norm_freqs = peaks / base  # frequency ratios relative to fundamental

    t = np.linspace(0.0, 2.0 * np.pi * n_periods, n_points)
    x = np.zeros(n_points)
    y = np.zeros(n_points)
    for f, A, ph in zip(norm_freqs, amps, phases):
        x += A * np.sin(f * t + ph + np.pi / 2)
        y += A * np.sin(f * t + ph)

    coords = np.stack([x, y], axis=1)
    return GeometryData(
        geom_type="curve_2d",
        coordinates=coords,
        parameters={
            "n_components": int(len(peaks)),
            "n_points": int(n_points),
            "n_periods": int(n_periods),
        },
        metadata={"kind": "lissajous_compound", "compound": True,
                  "n_components": int(len(peaks))},
    )


def lissajous_phase_drift(
    ratio: RatioLike,
    drift_rate: float,
    duration: float,
    sr: int = 1000,
    amps: Tuple[float, float] = (1.0, 1.0),
) -> GeometryData:
    """Lissajous with a linearly-drifting phase.

    The phase ``δ(t) = drift_rate · t`` evolves linearly with time, producing
    the classic "spinning" Lissajous animation when rendered.

    Parameters
    ----------
    ratio : Fraction, int, float, or (int, int)
    drift_rate : float
        Phase drift in radians per second.
    duration : float
        Total duration in seconds.
    sr : int, default=1000
        Sample rate (samples per second).
    amps : tuple of float, default=(1.0, 1.0)

    Returns
    -------
    GeometryData
        ``geom_type='curve_2d'`` with shape ``(int(sr * duration), 2)``.
    """
    if duration <= 0:
        raise ValueError(f"duration must be > 0, got {duration!r}.")
    if sr <= 0:
        raise ValueError(f"sr must be > 0, got {sr!r}.")

    a, b = coprime_pair(ratio)
    A_x, A_y = float(amps[0]), float(amps[1])

    n = int(sr * duration)
    t = np.linspace(0.0, float(duration), n)
    delta = drift_rate * t
    # Use the time variable directly (not a unitless 0..2π) so drift_rate
    # is in physical rad/s.
    x = A_x * np.sin(2 * np.pi * a * t + delta)
    y = A_y * np.sin(2 * np.pi * b * t)
    coords = np.stack([x, y], axis=1)

    return GeometryData(
        geom_type="curve_2d",
        coordinates=coords,
        parameters={
            "ratio": Fraction(a, b),
            "drift_rate": float(drift_rate),
            "duration": float(duration),
            "sr": int(sr),
            "amps": (A_x, A_y),
        },
        metadata={"kind": "lissajous_phase_drift", "a": a, "b": b,
                  "phase_drift": True},
    )


# ----------------------------------------------------------------- topology


def _count_self_intersections(coords: np.ndarray) -> int:
    """Count segment-segment intersections in a polyline.

    Brute force ``O(N²)``: for ``N`` points (``N-1`` segments) this is
    ``~N²/2`` orientation tests. Endpoints shared by adjacent segments are
    excluded.
    """
    pts = np.asarray(coords, dtype=np.float64)
    n = pts.shape[0]
    if n < 4:
        return 0

    p1 = pts[:-1]
    p2 = pts[1:]
    n_seg = n - 1

    def _ccw(ax, ay, bx, by, cx, cy):
        return (cy - ay) * (bx - ax) - (by - ay) * (cx - ax)

    count = 0
    for i in range(n_seg - 1):
        ax, ay = p1[i]
        bx, by = p2[i]
        # Compare against non-adjacent segments only.
        js = np.arange(i + 2, n_seg)
        if len(js) == 0:
            continue
        cx = p1[js, 0]
        cy = p1[js, 1]
        dx = p2[js, 0]
        dy = p2[js, 1]
        d1 = _ccw(cx, cy, dx, dy, ax, ay)
        d2 = _ccw(cx, cy, dx, dy, bx, by)
        d3 = _ccw(ax, ay, bx, by, cx, cy)
        d4 = _ccw(ax, ay, bx, by, dx, dy)
        intersects = ((d1 * d2) < 0) & ((d3 * d4) < 0)
        count += int(np.count_nonzero(intersects))
    return count


def lissajous_topology(geom: GeometryData) -> dict:
    """Inspect a Lissajous-style ``curve_2d`` and return topological summary.

    Parameters
    ----------
    geom : GeometryData
        Must have ``geom_type='curve_2d'``.

    Returns
    -------
    dict
        Keys:

        * ``'lobes_x'``, ``'lobes_y'`` — integer lobe counts (read from
          metadata when available; otherwise estimated by counting zero
          crossings on each axis).
        * ``'closed'`` — whether the first and last points coincide within
          a small tolerance.
        * ``'self_intersections'`` — count of polyline self-intersections
          (brute-force, ``O(N²)``).
        * ``'period_ratio'`` — :class:`~fractions.Fraction` representing
          the ``a / b`` ratio when known, else ``None``.
    """
    if geom.geom_type != "curve_2d":
        raise ValueError(
            f"lissajous_topology expects geom_type='curve_2d', got {geom.geom_type!r}."
        )
    coords = np.asarray(geom.coordinates, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coordinates must have shape (N, 2).")

    lobes_x = geom.metadata.get("lobes_x")
    lobes_y = geom.metadata.get("lobes_y")
    if lobes_x is None or lobes_y is None:
        lobes_x = _count_zero_crossings(coords[:, 0])
        lobes_y = _count_zero_crossings(coords[:, 1])

    scale = float(np.max(np.abs(coords))) + 1e-12
    closed = bool(np.linalg.norm(coords[0] - coords[-1]) < 1e-3 * scale)

    ratio = geom.parameters.get("ratio")
    if ratio is None and "a" in geom.metadata and "b" in geom.metadata:
        ratio = Fraction(int(geom.metadata["a"]), int(geom.metadata["b"]))

    return {
        "lobes_x": int(lobes_x),
        "lobes_y": int(lobes_y),
        "closed": closed,
        "self_intersections": _count_self_intersections(coords),
        "period_ratio": ratio if isinstance(ratio, Fraction) else (
            coerce_ratio(ratio) if ratio is not None else None
        ),
    }


def _count_zero_crossings(signal: np.ndarray) -> int:
    """Count sign changes in a 1-D signal (a rough lobe-count proxy)."""
    s = np.asarray(signal)
    if s.size < 2:
        return 0
    signs = np.sign(s)
    # Treat exact zeros as the sign of the next non-zero value to avoid
    # double-counting.
    nz = signs != 0
    if not np.any(nz):
        return 0
    signs_nz = signs[nz]
    return int(np.count_nonzero(np.diff(signs_nz) != 0))
