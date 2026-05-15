"""
Bulk acoustic pressure field — coherent superposition of point sources.

Belongs to the ``wave_field`` family of
:mod:`biotuner.harmonic_geometry.media`. Where :class:`Interference`
solves an idealized dimensionless wave equation, :class:`Acoustic`
models a physically-grounded 2-D bulk medium with:

- explicit wave speed ``c`` (so wavelengths derive from chord
  frequencies),
- distance-dependent geometric spreading (``1/sqrt(r)`` in 2-D far
  field) plus optional exponential absorption,
- and four observables that the bare wave-field cannot expose:
  instantaneous pressure ``p``, time-averaged intensity ``⟨p²⟩``,
  Schlieren-style ``|∇²p|`` (shadowgraph), and the complex phase.

Each chord ratio defines one frequency; each of ``n_sources`` point
emitters radiates the entire frequency comb (or just one component,
selectable via ``source_assignment``). The geometry of the source
array is configurable: by default sources are placed on a small ring
around the domain center, equally spaced; the chord's pitch-class
angles can be used instead.

Output modes
------------
``"pressure"``  — signed snapshot ``p(x, y)`` (``field_2d``).
``"intensity"`` — time-averaged ``⟨p²⟩``; positive (``field_2d``).
``"schlieren"`` — ``|∇²p|`` normalized to ``[0, 1]``; produces the
                  shadowgraph / transparent-medium look (``field_2d``).
``"phase"``     — argument of complex amplitude in ``[−π, π]``
                  (``field_2d``).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

import numpy as np

from biotuner.harmonic_geometry.geometry_data import GeometryData
from biotuner.harmonic_geometry.inputs import HarmonicInput
from biotuner.harmonic_geometry.media.base import Medium


_VALID_OUTPUT_MODES = ("pressure", "intensity", "schlieren", "phase")
_VALID_SOURCE_LAYOUTS = ("ring", "chord_angles", "linear", "custom")
_VALID_ASSIGNMENT = ("shared", "per_ratio")


def _ring_positions(
    n: int, radius: float, phase_offset: float = 0.0
) -> np.ndarray:
    """N points equally spaced on a circle of given radius."""
    if n <= 0:
        return np.zeros((0, 2))
    angles = phase_offset + 2.0 * np.pi * np.arange(n) / max(n, 1)
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)


def _chord_angle_positions(
    chord: HarmonicInput, radius: float
) -> np.ndarray:
    """Source positions on a ring at the chord's pitch-class angles."""
    ratios = np.asarray(
        [float(r) for r in chord.to_ratios()], dtype=np.float64
    )
    angles = 2.0 * np.pi * np.mod(
        np.log2(np.maximum(ratios, 1e-12)), 1.0
    )
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)


def _linear_positions(n: int, extent: float) -> np.ndarray:
    """N points equally spaced along the x-axis around the center."""
    if n <= 1:
        return np.zeros((max(n, 0), 2))
    xs = np.linspace(-extent * 0.4, +extent * 0.4, n)
    return np.stack([xs, np.zeros_like(xs)], axis=1)


def _xy_grid(extent: float, resolution: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.linspace(-extent, extent, resolution)
    y = np.linspace(-extent, extent, resolution)
    return np.meshgrid(x, y, indexing="xy")


def _complex_pressure(
    sources: np.ndarray,
    source_freqs: list[np.ndarray],     # per-source array of frequencies
    source_amps: list[np.ndarray],      # per-source array of amplitudes
    X: np.ndarray,
    Y: np.ndarray,
    wave_speed: float,
    attenuation: float,
    near_field_eps: float,
) -> np.ndarray:
    """Sum over sources × frequencies; return the complex field P(x, y).

    ``p(x, y, t) = Re[ P(x, y) · exp(−i·ω·t) ]`` so the snapshot at
    ``t=0`` is ``Re[P]`` and the time-average of ``p²`` is ``|P|²/2``.
    """
    P = np.zeros_like(X, dtype=np.complex128)
    for (sx, sy), freqs, amps in zip(sources, source_freqs, source_amps):
        r = np.hypot(X - sx, Y - sy) + near_field_eps
        decay = np.exp(-attenuation * r) / np.sqrt(r)
        for f, a in zip(freqs, amps):
            k = 2.0 * np.pi * float(f) / wave_speed
            P = P + a * decay * np.exp(1j * k * r)
    return P


def _laplacian(
    field: np.ndarray, dx: float, dy: float
) -> np.ndarray:
    """5-point Laplacian with edge replication (no boundary artefacts)."""
    f = np.pad(field, 1, mode="edge")
    return (
        (f[1:-1, 2:] - 2.0 * f[1:-1, 1:-1] + f[1:-1, :-2]) / (dx * dx)
        + (f[2:, 1:-1] - 2.0 * f[1:-1, 1:-1] + f[:-2, 1:-1]) / (dy * dy)
    )


class Acoustic(Medium):
    """Wave-field-family medium: bulk 2-D acoustic pressure field.

    Parameters
    ----------
    n_sources : int, default 3
        Number of point emitters. Defaults to a small number so the
        interference pattern stays readable.
    source_layout : {"ring", "chord_angles", "linear", "custom"}, default "ring"
        How to place the sources in the domain.
    source_positions : (N, 2) array-like, optional
        Custom source coordinates, used only with ``source_layout="custom"``.
    source_radius : float, default 0.25
        Radius of the source ring (in units of ``extent``) when
        ``source_layout`` is ``"ring"`` or ``"chord_angles"``.
    source_assignment : {"shared", "per_ratio"}, default "shared"
        ``"shared"`` — every source radiates the full chord (one
        coherent emitter set per ratio per source).
        ``"per_ratio"`` — source ``j`` radiates only ratio ``j`` of
        the chord (cycling if ``n_sources != n_ratios``).
    wave_speed : float, default 1.0
        Propagation speed ``c``; wavelengths follow ``λ = c / f``.
    base_frequency : float, default 6.0
        Frequency of the chord's root ratio. Chord ratios scale this
        (a ratio of 3/2 emits at ``1.5 · base_frequency``).
    attenuation : float, default 0.0
        Exponential absorption rate (per unit distance). ``0`` =
        lossless propagation.
    extent : float, default 1.0
        Half-side of the square rendering domain.
    resolution : int, default 256
        Grid resolution in each direction.
    near_field_eps : float, default 1e-3
        Small distance offset to avoid singular 1/sqrt(r) at source
        points.
    output_mode : {"pressure", "intensity", "schlieren", "phase"}, default "pressure"
        Which observable to render. See module docstring.
    """

    family = "wave_field"

    def __init__(
        self,
        *,
        n_sources: int = 3,
        source_layout: str = "ring",
        source_positions: Optional[Sequence[Sequence[float]]] = None,
        source_radius: float = 0.25,
        source_assignment: str = "shared",
        wave_speed: float = 1.0,
        base_frequency: float = 6.0,
        attenuation: float = 0.0,
        extent: float = 1.0,
        resolution: int = 256,
        near_field_eps: float = 1e-3,
        output_mode: str = "pressure",
    ) -> None:
        if source_layout not in _VALID_SOURCE_LAYOUTS:
            raise ValueError(
                f"source_layout must be one of {_VALID_SOURCE_LAYOUTS}; "
                f"got {source_layout!r}."
            )
        if source_assignment not in _VALID_ASSIGNMENT:
            raise ValueError(
                f"source_assignment must be one of {_VALID_ASSIGNMENT}; "
                f"got {source_assignment!r}."
            )
        if output_mode not in _VALID_OUTPUT_MODES:
            raise ValueError(
                f"output_mode must be one of {_VALID_OUTPUT_MODES}; "
                f"got {output_mode!r}."
            )
        if n_sources < 1:
            raise ValueError("n_sources must be >= 1.")
        if wave_speed <= 0:
            raise ValueError("wave_speed must be > 0.")
        if base_frequency <= 0:
            raise ValueError("base_frequency must be > 0.")
        if attenuation < 0:
            raise ValueError("attenuation must be >= 0.")
        if extent <= 0:
            raise ValueError("extent must be > 0.")
        if resolution < 16:
            raise ValueError("resolution must be >= 16.")
        if not (0.0 < source_radius < 1.0):
            raise ValueError("source_radius must be in (0, 1).")
        if near_field_eps <= 0:
            raise ValueError("near_field_eps must be > 0.")
        if source_layout == "custom" and source_positions is None:
            raise ValueError(
                "source_layout='custom' requires source_positions."
            )

        self.n_sources = int(n_sources)
        self.source_layout = source_layout
        self.source_positions = (
            np.asarray(source_positions, dtype=np.float64)
            if source_positions is not None else None
        )
        self.source_radius = float(source_radius)
        self.source_assignment = source_assignment
        self.wave_speed = float(wave_speed)
        self.base_frequency = float(base_frequency)
        self.attenuation = float(attenuation)
        self.extent = float(extent)
        self.resolution = int(resolution)
        self.near_field_eps = float(near_field_eps)
        self.output_mode = output_mode

    # ----------------------------------------------------------- contract

    def default_source(self) -> None:
        return None

    def respond(
        self,
        forcing: HarmonicInput,
        **overrides: Any,
    ) -> GeometryData:
        if not isinstance(forcing, HarmonicInput):
            raise TypeError(
                "Acoustic.respond requires a HarmonicInput; got "
                f"{type(forcing).__name__}."
            )

        n_sources = int(overrides.pop("n_sources", self.n_sources))
        source_layout = overrides.pop("source_layout", self.source_layout)
        source_radius = float(overrides.pop(
            "source_radius", self.source_radius))
        source_assignment = overrides.pop(
            "source_assignment", self.source_assignment)
        wave_speed = float(overrides.pop("wave_speed", self.wave_speed))
        base_frequency = float(overrides.pop(
            "base_frequency", self.base_frequency))
        attenuation = float(overrides.pop("attenuation", self.attenuation))
        extent = float(overrides.pop("extent", self.extent))
        resolution = int(overrides.pop("resolution", self.resolution))
        near_field_eps = float(overrides.pop(
            "near_field_eps", self.near_field_eps))
        output_mode = overrides.pop("output_mode", self.output_mode)
        source_positions = overrides.pop(
            "source_positions", self.source_positions)
        if source_positions is not None:
            source_positions = np.asarray(source_positions, dtype=np.float64)

        if overrides:
            raise TypeError(
                f"Unexpected override keys: {sorted(overrides)}."
            )
        if output_mode not in _VALID_OUTPUT_MODES:
            raise ValueError(
                f"output_mode must be one of {_VALID_OUTPUT_MODES}."
            )

        # ------ resolve source positions
        if source_layout == "ring":
            sources = _ring_positions(
                n_sources, source_radius * extent
            )
        elif source_layout == "chord_angles":
            sources = _chord_angle_positions(
                forcing, source_radius * extent
            )
        elif source_layout == "linear":
            sources = _linear_positions(n_sources, extent)
        else:
            if source_positions is None:
                raise ValueError(
                    "source_layout='custom' requires source_positions."
                )
            sources = source_positions
        n_sources_actual = len(sources)

        # ------ resolve per-source frequencies / amplitudes
        ratios = np.asarray(
            [float(r) for r in forcing.to_ratios()], dtype=np.float64
        )
        amps = forcing.normalized_amplitudes()
        if amps.size != ratios.size:
            amps = np.ones_like(ratios) / max(ratios.size, 1)
        all_freqs = ratios * base_frequency
        if source_assignment == "shared":
            source_freqs = [all_freqs for _ in range(n_sources_actual)]
            source_amps = [amps for _ in range(n_sources_actual)]
        else:  # per_ratio
            source_freqs = []
            source_amps = []
            for j in range(n_sources_actual):
                k = j % ratios.size
                source_freqs.append(np.array([all_freqs[k]]))
                source_amps.append(np.array([amps[k]]))

        # ------ field
        X, Y = _xy_grid(extent, resolution)
        P = _complex_pressure(
            sources, source_freqs, source_amps,
            X, Y, wave_speed, attenuation, near_field_eps,
        )

        parameters = {
            "n_sources": n_sources_actual,
            "source_layout": source_layout,
            "source_radius": source_radius,
            "source_assignment": source_assignment,
            "wave_speed": wave_speed,
            "base_frequency": base_frequency,
            "attenuation": attenuation,
            "extent": extent,
            "resolution": resolution,
            "near_field_eps": near_field_eps,
            "output_mode": output_mode,
        }
        metadata = {
            "kind": f"acoustic_{output_mode}",
            "family": "wave_field",
            "source_positions": sources,
        }

        # ------ output selection
        if output_mode == "pressure":
            field = np.real(P).astype(np.float64)
        elif output_mode == "intensity":
            field = (np.abs(P) ** 2).astype(np.float64) * 0.5
        elif output_mode == "schlieren":
            p_real = np.real(P).astype(np.float64)
            dx = float(X[0, 1] - X[0, 0])
            dy = float(Y[1, 0] - Y[0, 0])
            lap = _laplacian(p_real, dx, dy)
            mag = np.abs(lap)
            mx = float(mag.max())
            field = (mag / mx if mx > 0 else mag).astype(np.float64)
        elif output_mode == "phase":
            field = np.angle(P).astype(np.float64)
        else:
            raise AssertionError("unreachable")  # pragma: no cover

        return GeometryData(
            geom_type="field_2d",
            coordinates=field,
            field_grid=(X, Y),
            parameters=parameters,
            metadata=metadata,
        )

    def __repr__(self) -> str:
        return (
            f"Acoustic(n_sources={self.n_sources}, "
            f"layout={self.source_layout!r}, "
            f"output_mode={self.output_mode!r})"
        )
