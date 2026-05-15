"""
Anisotropic / variable-coefficient eigenmode plate.

Belongs to the ``eigenmode`` family of
:mod:`biotuner.harmonic_geometry.media`. Where :class:`RigidPlate`
solves the *isotropic* wave equation ``-grad^2 psi = lambda * psi`` on
a uniform plate, :class:`Elastic` solves an anisotropic / inhomogeneous
variant:

    -(alpha(x, y) * d^2_x + beta(x, y) * d^2_y) psi = lambda * psi

The simplest physically-meaningful subset of this family — and what
this implementation provides — is the **constant anisotropy** case:

    alpha and beta are constants, with alpha != beta,

which is the wave equation of an elastic plate whose stiffness differs
along the ``x`` and ``y`` axes. The eigenmodes remain analytically
separable on a rectangle:

    psi_mn(x, y) = sin(m * pi * x / Lx) * sin(n * pi * y / Ly),
    lambda_mn    = alpha * (m * pi / Lx)^2 + beta * (n * pi / Ly)^2.

Different eigenfrequencies -> different chord-to-mode projection -> a
visually distinct nodal pattern from the isotropic plate.

Anisotropy axis
---------------
``anisotropy_axis`` rotates the (alpha, beta) axes by an angle theta.
The solver applies an internal change of coordinates
``(x', y') = R_theta * (x, y)`` so the user can apply the
directionally-stretched plate at any angle.

Chord coupling
--------------
With ``anisotropy_ratio`` set to ``None`` (default), it is derived
from ``coupling.consonance(chord)`` and mapped into ``[1.0, 4.0]``:
consonant chords get a near-isotropic plate (ratio ~ 1), dissonant
chords get a strongly anisotropic plate (ratio ~ 4), producing
characteristic elongated nodal patterns.

With ``anisotropy_axis`` set to ``None``, the axis is the chord's
mean pitch-class angle on the unit circle.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

from biotuner.harmonic_geometry.geometry_data import GeometryData
from biotuner.harmonic_geometry.inputs import HarmonicInput
from biotuner.harmonic_geometry.media import coupling as _coupling
from biotuner.harmonic_geometry.media.base import Medium, Rectangular
from biotuner.harmonic_geometry.media.eigenmode.rigid_plate import (
    ratios_to_modes,
)


def _anisotropic_mode_pairs(
    ratios: list[float],
    Lx: float,
    Ly: float,
    alpha: float,
    beta: float,
    max_mode: int,
    n_per_ratio: int,
    mode_scale: float = 8.0,
) -> tuple[list[tuple[int, int]], list[float]]:
    """For each ratio, find the ``n_per_ratio`` (m, n) pairs whose
    anisotropy-weighted eigenvalue is closest to ``(r * mode_scale)^2 *
    lambda_(1,1)``.

    On an *isotropic* plate the iso-eigenvalue contours are circles, so
    a chord ratio ``r`` selects modes on a circle of radius
    ``r * mode_scale``. Anisotropy stretches the contours into ellipses
    aligned with the (alpha, beta) axes — the *same* chord then
    projects onto a different mode set, which is what makes the visible
    field shape change with anisotropy.

    ``mode_scale`` sets where the chord's fundamental (``r = 1``) lands
    in mode space. With ``mode_scale = 8`` and chord ratios in ``[1,
    1.5]``, modes near ``m^2 + n^2 ~ 64..144`` are picked, well into the
    interior of a 20x20 mode grid where the elliptical vs circular
    distinction matters.

    Returns
    -------
    (modes, weights)
        Flattened lists of length ``len(ratios) * n_per_ratio``.
        Weights ``∝ 1/(eigenvalue_distance + eps)`` so the closest mode
        dominates.
    """
    mm = np.arange(1, max_mode + 1)
    nn = np.arange(1, max_mode + 1)
    Mg, Ng = np.meshgrid(mm, nn, indexing="ij")
    lam = alpha * (Mg * np.pi / Lx) ** 2 + beta * (Ng * np.pi / Ly) ** 2

    lam_unit = alpha * (np.pi / Lx) ** 2 + beta * (np.pi / Ly) ** 2

    modes: list[tuple[int, int]] = []
    weights: list[float] = []
    for r in ratios:
        target = ((float(r) * mode_scale) ** 2) * lam_unit
        dist = np.abs(lam - target).ravel()
        order = np.argsort(dist)[:max(int(n_per_ratio), 1)]
        for idx in order:
            mi, ni = Mg.ravel()[idx], Ng.ravel()[idx]
            modes.append((int(mi), int(ni)))
            weights.append(1.0 / (float(dist[idx]) + 1e-6))
    return modes, weights


def _rotate(
    X: np.ndarray, Y: np.ndarray, theta: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate the coordinate grid by ``theta`` radians."""
    c, s = np.cos(theta), np.sin(theta)
    return X * c + Y * s, -X * s + Y * c


def _elastic_field_rectangular(
    modes: list[tuple[int, int]],
    amplitudes: list[float],
    Lx: float,
    Ly: float,
    alpha: float,
    beta: float,
    theta: float,
    resolution: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the eigenmode field on an anisotropic rectangular plate.

    Returns ``(field, X, Y)`` where ``(X, Y)`` is the meshgrid in the
    original (un-rotated) coordinate system.
    """
    x = np.linspace(0.0, Lx, resolution)
    y = np.linspace(0.0, Ly, resolution)
    X, Y = np.meshgrid(x, y, indexing="xy")

    # Rotate into the anisotropy frame (where alpha aligns with x' and
    # beta with y'). After rotation, the modes are products of sines in
    # the rotated coordinates — use the larger of (Lx, Ly) as the
    # rotated bounding-box length so modes stay shape-consistent.
    Xr, Yr = _rotate(X - 0.5 * Lx, Y - 0.5 * Ly, theta)
    Lxp = max(Lx, Ly)
    Lyp = max(Lx, Ly)
    Xr = Xr + 0.5 * Lxp
    Yr = Yr + 0.5 * Lyp

    field = np.zeros_like(X, dtype=np.float64)
    for (m, n), a in zip(modes, amplitudes):
        mode = (
            np.sin(m * np.pi * Xr / Lxp)
            * np.sin(n * np.pi * Yr / Lyp)
        )
        # Anisotropy enters the eigenfrequency weighting.
        lam = alpha * (m * np.pi / Lxp) ** 2 + beta * (n * np.pi / Lyp) ** 2
        weight = a / np.sqrt(max(lam, 1e-9))
        field = field + weight * mode

    peak = float(np.nanmax(np.abs(field)))
    if peak > 0:
        field = field / peak
    return field, X, Y


def _chord_mean_pitch_class_angle(chord: HarmonicInput) -> float:
    """Mean pitch-class angle of the chord on the unit circle (radians)."""
    ratios = np.asarray(
        [float(r) for r in chord.to_ratios()], dtype=np.float64
    )
    amps = chord.normalized_amplitudes()
    if amps.size != ratios.size:
        amps = np.ones_like(ratios) / max(ratios.size, 1)
    angles = 2.0 * np.pi * np.mod(np.log2(np.maximum(ratios, 1e-12)), 1.0)
    cx = float(np.sum(amps * np.cos(angles)))
    cy = float(np.sum(amps * np.sin(angles)))
    return float(np.arctan2(cy, cx))


class Elastic(Medium):
    """Eigenmode-family medium: anisotropic rectangular plate.

    Parameters
    ----------
    domain : Rectangular, optional
        Plate domain. Defaults to a unit square.
    anisotropy_ratio : float, optional
        ``alpha / beta`` of the wave equation. ``1.0`` reduces to the
        isotropic :class:`RigidPlate`; larger values stretch the
        effective wavelengths along the anisotropy axis. If ``None``
        (default), derived from
        :func:`coupling.consonance` in ``[1.0, 4.0]``.
    anisotropy_axis : float, optional
        Rotation angle (radians) of the anisotropy axes relative to
        ``+x``. If ``None`` (default), derived from the chord's mean
        pitch-class angle.
    n_modes : int, default 32
        Number of (m, n) modes used in the projection.
    resolution : int, default 256
        Spatial resolution of the rendered field.
    """

    family = "eigenmode"

    def __init__(
        self,
        *,
        domain: Optional[Rectangular] = None,
        anisotropy_ratio: Optional[float] = None,
        anisotropy_axis: Optional[float] = None,
        n_modes: int = 32,
        resolution: int = 256,
    ) -> None:
        if domain is None:
            domain = Rectangular(1.0, 1.0)
        if not isinstance(domain, Rectangular):
            raise TypeError(
                "Elastic only supports Rectangular domains in this "
                "implementation."
            )
        if anisotropy_ratio is not None and anisotropy_ratio <= 0:
            raise ValueError("anisotropy_ratio must be > 0 when given.")
        if n_modes < 1:
            raise ValueError("n_modes must be >= 1.")
        if resolution < 16:
            raise ValueError("resolution must be >= 16.")

        self.domain = domain
        self.anisotropy_ratio = anisotropy_ratio
        self.anisotropy_axis = anisotropy_axis
        self.n_modes = int(n_modes)
        self.resolution = int(resolution)

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
                "Elastic.respond requires a HarmonicInput; got "
                f"{type(forcing).__name__}."
            )

        ratio_arg = overrides.pop("anisotropy_ratio", self.anisotropy_ratio)
        axis_arg = overrides.pop("anisotropy_axis", self.anisotropy_axis)
        n_modes = int(overrides.pop("n_modes", self.n_modes))
        resolution = int(overrides.pop("resolution", self.resolution))

        if overrides:
            raise TypeError(
                f"Unexpected override keys: {sorted(overrides)}."
            )
        if n_modes < 1:
            raise ValueError("n_modes must be >= 1.")

        # ----- chord-driven defaults
        if ratio_arg is None:
            cons = _coupling.consonance(forcing)
            ratio_arg = 1.0 + (1.0 - cons) * (4.0 - 1.0)
        anisotropy_ratio = float(ratio_arg)

        if axis_arg is None:
            axis_arg = _chord_mean_pitch_class_angle(forcing)
        anisotropy_axis = float(axis_arg)

        # alpha : beta = anisotropy_ratio : 1, with alpha + beta ~ 2.
        alpha = 2.0 * anisotropy_ratio / (anisotropy_ratio + 1.0)
        beta = 2.0 / (anisotropy_ratio + 1.0)

        # ----- mode projection: anisotropy-aware
        ratios = [float(r) for r in forcing.to_ratios()]
        amps_arr = forcing.normalized_amplitudes()
        if amps_arr.size != len(ratios):
            amps_arr = np.ones(len(ratios)) / max(len(ratios), 1)
        ratios_used = ratios[: n_modes]
        amps_used = amps_arr[: n_modes]
        # For each chord ratio, pick the K best-matching (m, n) under
        # the anisotropic dispersion — this is what makes the visible
        # field shape genuinely change with the plate's stiffness
        # anisotropy (rather than the modes being shape-invariant and
        # only re-weighted).
        n_per_ratio = 4
        ani_modes, ani_weights = _anisotropic_mode_pairs(
            ratios_used,
            self.domain.Lx, self.domain.Ly,
            alpha, beta,
            max_mode=20, n_per_ratio=n_per_ratio,
        )
        modes_list = ani_modes
        # Distribute chord amplitudes across the K modes per ratio.
        amps_list: list[float] = []
        for j, a in enumerate(amps_used):
            w_slice = ani_weights[j * n_per_ratio: (j + 1) * n_per_ratio]
            w_sum = sum(w_slice) or 1.0
            for w in w_slice:
                amps_list.append(float(a) * float(w) / w_sum)

        field, X, Y = _elastic_field_rectangular(
            modes_list, amps_list,
            self.domain.Lx, self.domain.Ly,
            alpha, beta, anisotropy_axis, resolution,
        )

        parameters = {
            "domain_kind": "rectangular",
            "Lx": self.domain.Lx,
            "Ly": self.domain.Ly,
            "anisotropy_ratio": anisotropy_ratio,
            "anisotropy_axis": anisotropy_axis,
            "alpha": alpha,
            "beta": beta,
            "n_modes": n_modes,
            "resolution": resolution,
        }
        metadata = {
            "kind": "elastic_eigenmode",
            "family": "eigenmode",
            "n_modes_used": len(modes_list),
        }
        return GeometryData(
            geom_type="field_2d",
            coordinates=field,
            field_grid=(X, Y),
            parameters=parameters,
            metadata=metadata,
        )

    def __repr__(self) -> str:
        return (
            f"Elastic(anisotropy_ratio={self.anisotropy_ratio}, "
            f"anisotropy_axis={self.anisotropy_axis})"
        )
