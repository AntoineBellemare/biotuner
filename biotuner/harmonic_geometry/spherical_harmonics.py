"""
Spherical-harmonic eigenfields on the unit sphere.

This module is the 3-D rotational analogue of :mod:`.chladni`. Where
Chladni produces eigenmodes of the Laplacian on a *bounded* plate, here
we produce eigenmodes of the Laplacian on the *closed* unit sphere — the
spherical harmonics ``Y_l^m(θ, φ)``.

A chord's frequency ratios are mapped to ``(l, m)`` mode pairs and the
weighted superposition

::

    Ψ(θ, φ) = Σ_k  a_k · cos(φ_k) · Y_{l_k}^{m_k}(θ, φ)

is evaluated on a θ × φ grid (polar × azimuthal). The resulting scalar
field has the same character as a Chladni plate field — bright resonance
bands and dark nodal lines — but on a closed surface that can be rotated.

Two views of the same field
---------------------------
- :func:`spherical_harmonic_field` returns the field as a 2-D
  ``(n_theta, n_phi)`` array (``geom_type='field_2d'``) with ``field_grid``
  the polar/azimuthal meshgrid. This is the "Chladni-on-a-sphere" view,
  ready for renderers that map (θ, φ) onto a 3-D sphere.
- :func:`spherical_harmonic_mesh` returns a wobbled mesh (``geom_type=
  'mesh_3d'``) where each vertex is the unit-sphere position radially
  displaced by the local field amplitude — a chord-shaped "blob" you can
  rotate.

Mode rules
----------
The chord-to-(l, m) mapping is parameterised because there is no single
canonical choice. Three presets are provided:

- ``'zonal'``      — ``m = 0``: banded patterns, parallel to the equator.
- ``'sectoral'``   — ``|m| = l``: vertical "orange-segment" lobes.
- ``'chord_balanced'`` — cycles through {0, ±l/2, ±l} per harmonic so
  the chord excites a mix of zonal, tesseral, and sectoral modes.

Connection to ambisonics
------------------------
Spherical harmonics are the basis used to encode 3-D spatial audio
(B-format / higher-order ambisonics). The functions here are the same
``Y_l^m`` used to define ambisonic channels — so a chord rendered as a
spherical-harmonic superposition is, in a literal sense, the chord
projected into the spatial-audio basis.

References
----------
.. [1] Arfken, G., Weber, H. (2005). Mathematical Methods for Physicists.
       Chapter on Legendre functions and spherical harmonics.
.. [2] Zotter, F., Frank, M. (2019). Ambisonics: A Practical 3D Audio
       Theory for Recording, Studio Production, Sound Reinforcement,
       and Virtual Reality. Springer.
.. [3] Williams, E. G. (1999). Fourier Acoustics. Chapter 6 (spherical
       wave functions).
"""

from __future__ import annotations

from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import sph_harm

from biotuner.harmonic_geometry.geometry_data import GeometryData
from biotuner.harmonic_geometry.inputs import HarmonicInput

# A single (l, m) mode index. l >= 0, -l <= m <= l.
ModePairLM = Tuple[int, int]

_MODE_RULES = {"zonal", "sectoral", "chord_balanced", "rounded"}
_L_RULES = {"numerator", "rounded"}


# ============================================================ ratios_to_modes_lm


def ratios_to_modes_lm(
    ratios: Sequence[float],
    mode_rule: str = "zonal",
    max_l: int = 10,
    l_rule: str = "numerator",
) -> List[ModePairLM]:
    """Map a sequence of frequency ratios to spherical-harmonic ``(l, m)`` pairs.

    Each ratio is converted to an integer degree ``l`` (via ``l_rule``)
    and then assigned an order ``m ∈ [-l, l]`` (via ``mode_rule``).

    Parameters
    ----------
    ratios : sequence of float
        Frequency ratios. Negative or zero values raise ``ValueError``.
    mode_rule : {'zonal', 'sectoral', 'chord_balanced', 'rounded'}, default='zonal'
        How to pick the order ``m`` for each component:

        * ``'zonal'``     — ``m = 0`` for every harmonic. Banded patterns
          parallel to the equator. Simplest, most legible.
        * ``'sectoral'``  — ``|m| = l``. Vertical orange-segment lobes;
          alternating sign per harmonic so the chord doesn't reduce to a
          single mode.
        * ``'chord_balanced'`` — cycles ``m`` through ``{0, ±l, ±l/2}``
          across the harmonics. Mixes zonal / tesseral / sectoral modes
          for visual variety.
        * ``'rounded'`` — alias of ``'zonal'``; kept for API parity with
          :func:`.chladni.ratios_to_modes`.
    max_l : int, default=10
        Cap on the degree ``l``.
    l_rule : {'numerator', 'rounded'}, default='numerator'
        How to convert each ratio ``r`` to a degree ``l``:

        * ``'numerator'`` — rationalise ``r`` with
          :meth:`Fraction.limit_denominator(max_l)` and use the
          numerator. Maps musical chords cleanly: ``4 : 5 : 6`` (i.e.
          ratios ``1, 5/4, 3/2``) yields ``l = 4, 5, 6``. This is the
          natural "harmonic-index" mapping and the default.
        * ``'rounded'`` — ``l = int(round(r))``, clamped to ``[0, max_l]``.
          Cheap and coarse; collapses ratios in ``[1, 2]`` to ``l ∈ {1, 2}``.

    Returns
    -------
    list of (int, int)
        One ``(l, m)`` tuple per input ratio.

    Notes
    -----
    Unlike Chladni's ``(m, n)`` pair, where both indices are positive and
    near-symmetric, here ``l`` is degree and ``m`` is order with
    ``m ∈ [-l, l]``. The two indices have very different geometric
    meanings: ``l`` controls how many nodal lines the mode has; ``m``
    controls how those lines are split between latitude and longitude.
    """
    if mode_rule not in _MODE_RULES:
        raise ValueError(
            f"mode_rule must be one of {sorted(_MODE_RULES)}, got {mode_rule!r}."
        )
    if l_rule not in _L_RULES:
        raise ValueError(
            f"l_rule must be one of {sorted(_L_RULES)}, got {l_rule!r}."
        )
    if max_l < 0:
        raise ValueError(f"max_l must be >= 0, got {max_l!r}.")

    out: List[ModePairLM] = []
    n_components = len(ratios)
    for idx, r in enumerate(ratios):
        rf = float(r)
        if rf <= 0:
            raise ValueError(f"Ratios must be > 0; got {rf!r}.")
        if l_rule == "numerator":
            # Limit the denominator first so numerators of common chord
            # ratios (1, 5/4, 3/2) come out as small integers (1, 5, 3 — or
            # 4, 5, 6 once the chord is expressed over a common denominator).
            l = Fraction(rf).limit_denominator(max(1, max_l)).numerator
        else:  # rounded
            l = int(round(rf))
        l = max(0, min(max_l, l))
        m = _pick_order(l, idx, n_components, mode_rule)
        out.append((l, m))
    return out


def _pick_order(l: int, idx: int, n_total: int, rule: str) -> int:
    """Choose ``m`` for the idx-th harmonic given its degree ``l``."""
    if l == 0:
        return 0
    if rule in {"zonal", "rounded"}:
        return 0
    if rule == "sectoral":
        return l if (idx % 2 == 0) else -l
    # chord_balanced
    pool: List[int]
    if l == 1:
        pool = [0, 1, -1]
    elif l >= 2:
        pool = [0, l, -l, l // 2, -(l // 2)]
    else:
        pool = [0]
    return pool[idx % len(pool)]


# ============================================================== Y_l^m primitives


def _real_ylm(
    l: int, m: int, theta: np.ndarray, phi: np.ndarray
) -> np.ndarray:
    """Real-valued spherical harmonic ``Y_l^m(θ, φ)``.

    Uses the standard physics convention:

    * ``θ`` is the polar (colatitudinal) angle in ``[0, π]``,
    * ``φ`` is the azimuthal angle in ``[0, 2π]``.

    The real spherical harmonics are defined as

    ::

        Y_l^m^real = √2 · (-1)^m · Re(Y_l^m)         (m > 0)
                   = Y_l^0                            (m = 0)
                   = √2 · (-1)^m · Im(Y_l^|m|)        (m < 0)

    which gives a real-valued orthonormal basis for ``L²(S²)``. Returns a
    real ndarray with the same shape as ``θ`` / ``φ``.
    """
    if l < 0:
        raise ValueError(f"l must be >= 0, got {l!r}.")
    if abs(m) > l:
        raise ValueError(f"|m| must be <= l; got l={l}, m={m}.")
    # scipy.special.sph_harm signature is (m, n, azimuth, polar). Our
    # convention is (theta=polar, phi=azimuth), so we swap when calling.
    if m == 0:
        return sph_harm(0, l, phi, theta).real
    if m > 0:
        Y = sph_harm(m, l, phi, theta)
        return float(np.sqrt(2)) * ((-1) ** m) * Y.real
    Y = sph_harm(abs(m), l, phi, theta)
    return float(np.sqrt(2)) * ((-1) ** m) * Y.imag


def _complex_ylm(
    l: int, m: int, theta: np.ndarray, phi: np.ndarray
) -> np.ndarray:
    """Complex-valued spherical harmonic, scipy passthrough with our angle convention."""
    if l < 0:
        raise ValueError(f"l must be >= 0, got {l!r}.")
    if abs(m) > l:
        raise ValueError(f"|m| must be <= l; got l={l}, m={m}.")
    return sph_harm(m, l, phi, theta)


# ============================================================== single-mode field


def single_spherical_harmonic(
    l: int,
    m: int,
    n_theta: int = 128,
    n_phi: int = 256,
    real: bool = True,
) -> GeometryData:
    """One spherical-harmonic mode ``Y_l^m`` evaluated on a (θ, φ) grid.

    Parameters
    ----------
    l : int
        Degree, ``l >= 0``.
    m : int
        Order, ``-l <= m <= l``.
    n_theta : int, default=128
        Polar samples in ``[0, π]``.
    n_phi : int, default=256
        Azimuthal samples in ``[0, 2π]``.
    real : bool, default=True
        If True, return the real spherical harmonic; otherwise the
        complex form (the field is then complex-valued).

    Returns
    -------
    GeometryData
        ``geom_type='field_2d'`` with ``coordinates`` shape
        ``(n_theta, n_phi)`` and ``field_grid=(THETA, PHI)`` meshgrids.
    """
    if n_theta < 4 or n_phi < 4:
        raise ValueError(
            f"n_theta and n_phi must each be >= 4, got "
            f"n_theta={n_theta}, n_phi={n_phi}."
        )
    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta, phi, indexing="ij")
    field = (
        _real_ylm(l, m, THETA, PHI) if real else _complex_ylm(l, m, THETA, PHI)
    )
    return GeometryData(
        geom_type="field_2d",
        coordinates=field,
        field_grid=(THETA, PHI),
        parameters={
            "l": int(l),
            "m": int(m),
            "n_theta": int(n_theta),
            "n_phi": int(n_phi),
            "real": bool(real),
        },
        metadata={
            "kind": "spherical_harmonic_single",
            "domain": "sphere",
            "angle_convention": "physics (theta=polar, phi=azimuth)",
        },
    )


# ============================================================ superposition field


def _resolve_amps_phases(
    n_modes: int,
    amps: Optional[Sequence[float]],
    phases: Optional[Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray]:
    if amps is None:
        a = np.full(n_modes, 1.0 / n_modes, dtype=np.float64)
    else:
        a = np.asarray(amps, dtype=np.float64)
        if a.shape[0] != n_modes:
            raise ValueError(
                f"amps has length {a.shape[0]} but {n_modes} modes were given."
            )
    if phases is None:
        p = np.zeros(n_modes, dtype=np.float64)
    else:
        p = np.asarray(phases, dtype=np.float64)
        if p.shape[0] != n_modes:
            raise ValueError(
                f"phases has length {p.shape[0]} but {n_modes} modes were given."
            )
    return a, p


def spherical_harmonic_field(
    modes_lm: Sequence[ModePairLM],
    amps: Optional[Sequence[float]] = None,
    phases: Optional[Sequence[float]] = None,
    n_theta: int = 128,
    n_phi: int = 256,
    real: bool = True,
) -> GeometryData:
    """Superposition of spherical-harmonic modes on a (θ, φ) grid.

    ``Ψ(θ, φ) = Σ_k  A_k · cos(φ_k) · Y_{l_k}^{m_k}(θ, φ)``

    Parameters
    ----------
    modes_lm : sequence of (int, int)
        ``(l, m)`` mode pairs, with ``l >= 0`` and ``|m| <= l``.
    amps : sequence of float, optional
        Per-mode amplitude. Defaults to uniform ``1 / n_modes``.
    phases : sequence of float, optional
        Per-mode phase in radians. Defaults to zeros (so ``cos(φ_k) = 1``).
    n_theta, n_phi : int
        Grid resolution in polar / azimuthal axes.
    real : bool, default=True
        Use real spherical harmonics (the orthonormal real basis).
        Set ``False`` to keep the complex-valued ``Y_l^m`` and let the
        field be complex.

    Returns
    -------
    GeometryData
        ``geom_type='field_2d'``. ``coordinates`` is the
        ``(n_theta, n_phi)`` field; ``field_grid`` is the meshgrid of the
        polar and azimuthal angles.
    """
    if not modes_lm:
        raise ValueError("modes_lm must be non-empty.")
    if n_theta < 4 or n_phi < 4:
        raise ValueError(
            f"n_theta and n_phi must each be >= 4, got "
            f"n_theta={n_theta}, n_phi={n_phi}."
        )
    n_modes = len(modes_lm)
    a, p = _resolve_amps_phases(n_modes, amps, phases)

    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta, phi, indexing="ij")

    if real:
        field = np.zeros_like(THETA, dtype=np.float64)
    else:
        field = np.zeros_like(THETA, dtype=np.complex128)

    for (l, m), Ai, phi_k in zip(modes_lm, a, p):
        ylm = (
            _real_ylm(int(l), int(m), THETA, PHI)
            if real
            else _complex_ylm(int(l), int(m), THETA, PHI)
        )
        field = field + Ai * np.cos(phi_k) * ylm

    return GeometryData(
        geom_type="field_2d",
        coordinates=field,
        field_grid=(THETA, PHI),
        parameters={
            "modes_lm": [(int(l), int(m)) for l, m in modes_lm],
            "amps": a.tolist(),
            "phases": p.tolist(),
            "n_theta": int(n_theta),
            "n_phi": int(n_phi),
            "real": bool(real),
        },
        metadata={
            "kind": "spherical_harmonic_field",
            "domain": "sphere",
            "n_modes": n_modes,
            "angle_convention": "physics (theta=polar, phi=azimuth)",
        },
    )


# =============================================================== HarmonicInput


def spherical_harmonic_temporal(
    input: HarmonicInput,
    t: float,
    mode_rule: str = "zonal",
    max_l: int = 10,
    l_rule: str = "numerator",
    n_theta: int = 128,
    n_phi: int = 256,
    real: bool = True,
) -> GeometryData:
    """Spherical-harmonic field at a specific time ``t``.

    Each component's phase is shifted by ``2π · f_k · t`` (with ``f_k``
    drawn from ``input.to_peaks()``), so iterating over a sequence of
    times produces a beating standing-wave evolution suitable for an
    animation. Mirrors the API of :func:`.chladni.chladni_temporal`.

    Parameters
    ----------
    input : HarmonicInput
    t : float
        Time in seconds.
    mode_rule, max_l, n_theta, n_phi, real
        Forwarded to :func:`spherical_harmonic_field`. See
        :func:`spherical_harmonic_from_input` for descriptions.

    Returns
    -------
    GeometryData
        ``geom_type='field_2d'``. The ``parameters['phases']`` field
        carries the time-shifted phase vector (so a renderer can label
        which frame it received without re-running the math).
    """
    peaks = input.to_peaks()
    drift = 2.0 * np.pi * np.asarray(peaks, dtype=np.float64) * float(t)
    base_phases = (
        np.asarray(input.phases, dtype=np.float64)
        if input.phases is not None
        else np.zeros(len(peaks), dtype=np.float64)
    )
    phases = (base_phases + drift).tolist()

    ratios = [float(r) for r in input.to_ratios()]
    amps = input.normalized_amplitudes().tolist()
    modes_lm = ratios_to_modes_lm(
        ratios, mode_rule=mode_rule, max_l=max_l, l_rule=l_rule
    )

    g = spherical_harmonic_field(
        modes_lm=modes_lm,
        amps=amps,
        phases=phases,
        n_theta=n_theta,
        n_phi=n_phi,
        real=real,
    )
    g.parameters["t"] = float(t)
    g.metadata["kind"] = "spherical_harmonic_temporal"
    return g


def spherical_harmonic_from_input(
    input: HarmonicInput,
    mode_rule: str = "zonal",
    max_l: int = 10,
    l_rule: str = "numerator",
    n_theta: int = 128,
    n_phi: int = 256,
    real: bool = True,
) -> GeometryData:
    """Build a spherical-harmonic field from a :class:`HarmonicInput`.

    Equivalent to :func:`.chladni.chladni_from_input` but rendered onto a
    sphere instead of a bounded plate.

    Ratios are mapped to ``(l, m)`` modes via :func:`ratios_to_modes_lm`,
    amplitudes are taken from ``input.normalized_amplitudes()``, phases
    default to ``input.phases`` or zeros.

    Parameters
    ----------
    input : HarmonicInput
    mode_rule : {'zonal', 'sectoral', 'chord_balanced', 'rounded'}
    max_l : int, default=10
    l_rule : {'numerator', 'rounded'}, default='numerator'
        See :func:`ratios_to_modes_lm`.
    n_theta, n_phi : int
        Grid resolution.
    real : bool, default=True
        Use real spherical harmonics.
    """
    ratios = [float(r) for r in input.to_ratios()]
    amps = input.normalized_amplitudes().tolist()
    phases = (
        list(input.phases) if input.phases is not None else [0.0] * len(ratios)
    )

    modes_lm = ratios_to_modes_lm(
        ratios, mode_rule=mode_rule, max_l=max_l, l_rule=l_rule
    )

    return spherical_harmonic_field(
        modes_lm=modes_lm,
        amps=amps,
        phases=phases,
        n_theta=n_theta,
        n_phi=n_phi,
        real=real,
    )


# ===================================================================== mesh


def spherical_harmonic_mesh(
    input: HarmonicInput,
    epsilon: float = 0.18,
    mode_rule: str = "zonal",
    max_l: int = 10,
    l_rule: str = "numerator",
    n_theta: int = 96,
    n_phi: int = 192,
) -> GeometryData:
    """Wobbled-radius mesh of a chord's spherical-harmonic superposition.

    Each vertex sits on the unit sphere displaced radially by

    ::

        r(θ, φ) = 1 + ε · Ψ̂(θ, φ)

    where ``Ψ̂`` is the chord's real-valued spherical-harmonic field
    rescaled so its peak absolute value is 1. The resulting mesh is the
    chord-shaped "blob" — concave at nodal lines, convex at antinodes.

    Parameters
    ----------
    input : HarmonicInput
    epsilon : float, default=0.18
        Maximum radial displacement (fraction of the unit radius).
        Smaller values keep the mesh close to a sphere; larger values
        emphasise the modal structure.
    mode_rule : str, default='zonal'
    max_l : int, default=10
    n_theta : int, default=96
    n_phi : int, default=192

    Returns
    -------
    GeometryData
        ``geom_type='mesh_3d'`` with

        * ``coordinates`` shape ``(V, 3)``: vertex positions in ``R³``.
          ``V = n_theta * n_phi``.
        * ``faces`` shape ``(F, 3)``: triangle indices into
          ``coordinates``. The mesh is a UV-sphere triangulation of the
          ``(n_theta - 1) × (n_phi - 1)`` quad grid (two triangles per
          quad).
        * ``weights`` shape ``(V,)``: the field amplitude at each vertex,
          useful for renderer colouring.
    """
    if epsilon < 0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon!r}.")

    field_data = spherical_harmonic_from_input(
        input,
        mode_rule=mode_rule,
        max_l=max_l,
        l_rule=l_rule,
        n_theta=n_theta,
        n_phi=n_phi,
        real=True,
    )
    field = np.asarray(field_data.coordinates, dtype=np.float64)
    THETA, PHI = field_data.field_grid  # (n_theta, n_phi) each

    peak = float(np.max(np.abs(field)))
    if peak < 1e-12:
        # Degenerate (e.g. all-zero superposition); fall back to unit sphere.
        normalized = np.zeros_like(field)
    else:
        normalized = field / peak

    r = 1.0 + epsilon * normalized
    X = r * np.sin(THETA) * np.cos(PHI)
    Y = r * np.sin(THETA) * np.sin(PHI)
    Z = r * np.cos(THETA)

    vertices = np.stack(
        [X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], axis=1
    )
    weights = normalized.reshape(-1)

    # Triangulate the quad grid. Vertex (i, j) is at index i * n_phi + j.
    nt, nph = field.shape
    faces: List[List[int]] = []
    for i in range(nt - 1):
        for j in range(nph - 1):
            v0 = i * nph + j
            v1 = i * nph + (j + 1)
            v2 = (i + 1) * nph + (j + 1)
            v3 = (i + 1) * nph + j
            faces.append([v0, v1, v2])
            faces.append([v0, v2, v3])
    faces_arr = np.asarray(faces, dtype=np.int64)

    return GeometryData(
        geom_type="mesh_3d",
        coordinates=vertices,
        faces=faces_arr,
        weights=weights,
        parameters={
            "epsilon": float(epsilon),
            "mode_rule": str(mode_rule),
            "max_l": int(max_l),
            "n_theta": int(n_theta),
            "n_phi": int(n_phi),
            "modes_lm": field_data.parameters["modes_lm"],
        },
        metadata={
            "kind": "spherical_harmonic_mesh",
            "domain": "sphere",
            "n_vertices": int(vertices.shape[0]),
            "n_faces": int(faces_arr.shape[0]),
        },
    )
