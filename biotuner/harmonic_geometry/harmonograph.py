"""
Harmonograph geometry — sums of damped sinusoids in 2-D and 3-D.

Per axis: ``x(t) = Σ_i A_i · sin(ω_i · t + φ_i) · exp(-d_i · t)``.

A classic two-pendulum harmonograph (lateral) traces a single 2-D path. A
rotary harmonograph adds a slow rotation around the origin. The 3-D variant
extends the same idea to three axes.

References
----------
.. [1] Whitaker, R. (2001). The Harmonograph: A Visual Guide to the
       Mathematics of Music.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from biotuner.harmonic_geometry.geometry_data import GeometryData
from biotuner.harmonic_geometry.inputs import HarmonicInput

# Default damping when the input has none. Small enough to leave the curve
# clearly visible over a 30 s exposure but non-zero so the trace converges.
DEFAULT_DAMPING = 0.01


# --------------------------------------------------------------------- helpers


def _resolve_damping(input: HarmonicInput, n: int) -> np.ndarray:
    if input.damping is not None:
        return np.asarray(input.damping, dtype=np.float64)
    return np.full(n, DEFAULT_DAMPING, dtype=np.float64)


def _resolve_phases(input: HarmonicInput, n: int) -> np.ndarray:
    if input.phases is not None:
        return np.asarray(input.phases, dtype=np.float64)
    return np.zeros(n, dtype=np.float64)


def _split_xy(n: int, x_components, y_components):
    """Resolve x/y component index lists, defaulting to alternation."""
    if x_components is None and y_components is None:
        x_components = list(range(0, n, 2))
        y_components = list(range(1, n, 2))
        # Ensure both axes have at least one component when n is small.
        if not y_components:
            y_components = x_components[-1:]
        if not x_components:
            x_components = y_components[:1]
    elif x_components is None:
        x_components = [i for i in range(n) if i not in set(y_components)]
        if not x_components:
            x_components = list(y_components)
    elif y_components is None:
        y_components = [i for i in range(n) if i not in set(x_components)]
        if not y_components:
            y_components = list(x_components)
    return list(x_components), list(y_components)


def _accumulate(
    t: np.ndarray,
    omegas: np.ndarray,
    amps: np.ndarray,
    phases: np.ndarray,
    damping: np.ndarray,
    indices: Sequence[int],
    extra_phase_offset: float = 0.0,
) -> np.ndarray:
    """Compute Σ_i A_i sin(ω_i t + φ_i) e^(-d_i t) for ``i in indices``."""
    out = np.zeros_like(t)
    for i in indices:
        out += (
            float(amps[i])
            * np.sin(2 * np.pi * omegas[i] * t + float(phases[i]) + extra_phase_offset)
            * np.exp(-float(damping[i]) * t)
        )
    return out


# ----------------------------------------------------------------- harmonograph


def harmonograph_lateral(
    input: HarmonicInput,
    duration: float = 30.0,
    sr: int = 200,
    x_components: Optional[Sequence[int]] = None,
    y_components: Optional[Sequence[int]] = None,
) -> GeometryData:
    """A two-pendulum lateral harmonograph trace.

    Parameters
    ----------
    input : HarmonicInput
        Provides peak frequencies, amplitudes, phases, and (optionally)
        damping per component.
    duration : float, default=30.0
        Duration of the trace in seconds.
    sr : int, default=200
        Sample rate in samples per second. Downstream renderers may
        resample.
    x_components, y_components : sequence of int, optional
        Indices of components assigned to each axis. If both are ``None``,
        components alternate (even indices → x, odd → y).

    Returns
    -------
    GeometryData
        ``geom_type='curve_2d'`` with shape ``(int(sr * duration), 2)``.
    """
    if duration <= 0:
        raise ValueError(f"duration must be > 0, got {duration!r}.")
    if sr <= 0:
        raise ValueError(f"sr must be > 0, got {sr!r}.")

    peaks = input.to_peaks()
    amps = input.normalized_amplitudes()
    phases = _resolve_phases(input, len(peaks))
    damping = _resolve_damping(input, len(peaks))

    n_components = len(peaks)
    x_idx, y_idx = _split_xy(n_components, x_components, y_components)

    n_samples = int(sr * duration)
    t = np.linspace(0.0, float(duration), n_samples)
    x = _accumulate(t, peaks, amps, phases, damping, x_idx)
    y = _accumulate(t, peaks, amps, phases, damping, y_idx, extra_phase_offset=np.pi / 2)

    coords = np.stack([x, y], axis=1)
    return GeometryData(
        geom_type="curve_2d",
        coordinates=coords,
        parameters={
            "duration": float(duration),
            "sr": int(sr),
            "x_components": x_idx,
            "y_components": y_idx,
            "n_components": n_components,
        },
        metadata={
            "kind": "harmonograph_lateral",
            "damping_default_used": input.damping is None,
        },
    )


def harmonograph_rotary(
    input: HarmonicInput,
    duration: float = 30.0,
    sr: int = 200,
    rotation_freq: float = 0.1,
) -> GeometryData:
    """Lateral harmonograph with an additional slow rotation about the origin.

    The lateral trace is rotated by an angle ``θ(t) = 2π · rotation_freq · t``,
    producing the rosette-like rotary patterns of a circular harmonograph.

    Parameters
    ----------
    input : HarmonicInput
    duration : float, default=30.0
    sr : int, default=200
    rotation_freq : float, default=0.1
        Angular drift in Hz applied to the entire trace.

    Returns
    -------
    GeometryData
        ``geom_type='curve_2d'``.
    """
    base = harmonograph_lateral(input, duration=duration, sr=sr)
    coords = np.asarray(base.coordinates, dtype=np.float64)
    n = coords.shape[0]
    t = np.linspace(0.0, float(duration), n)
    theta = 2 * np.pi * rotation_freq * t
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x_rot = coords[:, 0] * cos_t - coords[:, 1] * sin_t
    y_rot = coords[:, 0] * sin_t + coords[:, 1] * cos_t
    rotated = np.stack([x_rot, y_rot], axis=1)

    params = dict(base.parameters)
    params["rotation_freq"] = float(rotation_freq)
    meta = dict(base.metadata)
    meta["kind"] = "harmonograph_rotary"
    return GeometryData(
        geom_type="curve_2d",
        coordinates=rotated,
        parameters=params,
        metadata=meta,
    )


def harmonograph_3d(
    input: HarmonicInput,
    duration: float = 30.0,
    sr: int = 200,
    axis_assignment: str = "cyclic",
) -> GeometryData:
    """3-D harmonograph trace.

    Parameters
    ----------
    input : HarmonicInput
    duration : float, default=30.0
    sr : int, default=200
    axis_assignment : {'cyclic', 'split'}, default='cyclic'
        - ``'cyclic'``: component i is assigned to axis ``i % 3``.
        - ``'split'``: components are split contiguously into three
          near-equal blocks for x, y, z.

    Returns
    -------
    GeometryData
        ``geom_type='curve_3d'``.
    """
    if axis_assignment not in {"cyclic", "split"}:
        raise ValueError(
            f"axis_assignment must be 'cyclic' or 'split'; got {axis_assignment!r}."
        )
    if duration <= 0 or sr <= 0:
        raise ValueError("duration and sr must be > 0.")

    peaks = input.to_peaks()
    amps = input.normalized_amplitudes()
    phases = _resolve_phases(input, len(peaks))
    damping = _resolve_damping(input, len(peaks))
    n_components = len(peaks)

    if axis_assignment == "cyclic":
        idx_x = [i for i in range(n_components) if i % 3 == 0]
        idx_y = [i for i in range(n_components) if i % 3 == 1]
        idx_z = [i for i in range(n_components) if i % 3 == 2]
    else:
        third = max(1, n_components // 3)
        idx_x = list(range(0, third))
        idx_y = list(range(third, 2 * third))
        idx_z = list(range(2 * third, n_components))

    # Ensure each axis has at least one contributor.
    fallback = list(range(n_components))
    if not idx_x:
        idx_x = fallback[:1]
    if not idx_y:
        idx_y = fallback[:1]
    if not idx_z:
        idx_z = fallback[:1]

    n_samples = int(sr * duration)
    t = np.linspace(0.0, float(duration), n_samples)
    x = _accumulate(t, peaks, amps, phases, damping, idx_x)
    y = _accumulate(t, peaks, amps, phases, damping, idx_y, extra_phase_offset=np.pi / 2)
    z = _accumulate(t, peaks, amps, phases, damping, idx_z, extra_phase_offset=np.pi / 4)

    coords = np.stack([x, y, z], axis=1)
    return GeometryData(
        geom_type="curve_3d",
        coordinates=coords,
        parameters={
            "duration": float(duration),
            "sr": int(sr),
            "axis_assignment": axis_assignment,
            "x_components": idx_x,
            "y_components": idx_y,
            "z_components": idx_z,
            "n_components": n_components,
        },
        metadata={
            "kind": "harmonograph_3d",
            "damping_default_used": input.damping is None,
        },
    )


def harmonograph_from_peaks(
    peaks: Sequence[float],
    amps: Optional[Sequence[float]] = None,
    phases: Optional[Sequence[float]] = None,
    damping: Optional[Sequence[float]] = None,
    duration: float = 30.0,
    sr: int = 200,
) -> GeometryData:
    """Convenience: build a lateral harmonograph directly from peak Hz values.

    Internally constructs a :class:`HarmonicInput` and delegates to
    :func:`harmonograph_lateral`. If ``damping`` is ``None``, a uniform
    default of ``DEFAULT_DAMPING`` is used.
    """
    n = len(peaks)
    if damping is None:
        damping = [DEFAULT_DAMPING] * n
    inp = HarmonicInput.from_peaks(
        peaks,
        amplitudes=amps,
        phases=phases,
        damping=damping,
    )
    return harmonograph_lateral(inp, duration=duration, sr=sr)


def derive_damping_from_linewidth(
    linewidths: Sequence[float],
    default: float = DEFAULT_DAMPING,
) -> np.ndarray:
    """Convert spectral linewidths (FWHM, Hz) to damping coefficients (1/s).

    For a Lorentzian peak with full width at half maximum ``Δf``, the
    underlying decay rate is ``π · Δf`` (since the Lorentzian is the FT of
    a decaying exponential ``e^(-π Δf t)``).

    Non-positive or non-finite linewidths fall back to ``default``.
    """
    arr = np.asarray(linewidths, dtype=np.float64)
    out = np.where(np.isfinite(arr) & (arr > 0), np.pi * arr, default)
    return out
