"""
Rigid-plate eigenmode response (Chladni nodal fields and surfaces).

Belongs to the ``eigenmode`` family of :mod:`biotuner.harmonic_geometry.media`:
the operator here solves a bounded standing-wave eigenproblem
``−∇²ψ = λψ`` on a finite domain (rectangle, disk, polygon, 3-D box) and
returns the chord's projection onto the resulting eigenmodes. Sand on a
Chladni plate is the classical illustration of this regime.

This module produces scalar displacement fields for vibrating plates and
volumes, plus utilities for extracting their nodal sets (the curves /
surfaces where the displacement is zero — the lines along which sand
collects in a Chladni experiment).

The class wrapper :class:`RigidPlate` plugs into the pipeline contract
defined in :mod:`biotuner.harmonic_geometry.media.base`; the
``chladni_*`` functions remain the underlying functional API.

Plate kinds supported
---------------------
- **Rectangular** (free-edge superposition): closed-form Neumann modes.
- **Circular** (clamped membrane): Bessel × angular modes.
- **Polygon** (clamped membrane): numerical eigenmode solver via finite
  differences on a rasterized polygon mask.
- **3-D box** (Dirichlet): closed-form standing-wave modes in a volume.

Nodal extraction
----------------
- 2-D: marching squares via :func:`skimage.measure.find_contours`.
- 3-D: marching cubes via :func:`skimage.measure.marching_cubes`.

Both extraction functions lazy-import ``scikit-image``. If the dependency
is missing they raise :class:`ImportError` with installation instructions.

References
----------
.. [1] Chladni, E. F. F. (1787). Entdeckungen über die Theorie des Klanges.
.. [2] Lord Rayleigh (1894). The Theory of Sound.
.. [3] Leissa, A. W. (1969). Vibration of Plates. NASA SP-160.
"""

from __future__ import annotations

import math
from fractions import Fraction
from functools import lru_cache
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.special import jn, jn_zeros

from biotuner.harmonic_geometry.geometry_data import GeometryData
from biotuner.harmonic_geometry.media.base import Medium as _Medium
from biotuner.harmonic_geometry.inputs import HarmonicInput

ModePair = Tuple[int, int]
ModeTriple = Tuple[int, int, int]
RatioLike = Union[Fraction, int, float, Tuple[int, int]]


# ============================================================ ratios_to_modes


def ratios_to_modes(
    ratios: Iterable[RatioLike],
    strategy: str = "stern_brocot",
    max_mode: int = 20,
) -> List[ModePair]:
    """Map a list of ratios to Chladni mode pairs ``(m, n)``.

    Each ratio ``r`` is approximated by a coprime pair ``(m, n)`` with
    ``m, n ≤ max_mode`` (when feasible).

    Parameters
    ----------
    ratios : iterable of Fraction, int, float, or (int, int)
    strategy : {'stern_brocot', 'continued_fraction', 'rounded', 'best_simple'}
        Algorithm used to pick ``(m, n)``. ``'stern_brocot'`` uses
        :meth:`Fraction.limit_denominator` (Stern-Brocot mediant search);
        ``'continued_fraction'`` walks the continued-fraction convergents
        and stops at the last one with both terms ``≤ max_mode``;
        ``'rounded'`` returns ``(round(r), 1)`` for each ratio (cheap and
        coarse); ``'best_simple'`` brute-forces the closest ``(m, n)``
        pair in ``[1, max_mode]^2``.
    max_mode : int, default=20

    Returns
    -------
    list of (int, int)
    """
    if strategy not in {"stern_brocot", "continued_fraction", "rounded", "best_simple"}:
        raise ValueError(
            f"Unknown strategy {strategy!r}; expected one of "
            "'stern_brocot', 'continued_fraction', 'rounded', 'best_simple'."
        )
    if max_mode < 1:
        raise ValueError(f"max_mode must be >= 1, got {max_mode!r}.")

    out: List[ModePair] = []
    for r in ratios:
        rf = float(r)
        if rf <= 0:
            raise ValueError(f"Ratios must be > 0; got {rf!r}.")
        if strategy == "stern_brocot":
            out.append(_sb_pair(rf, max_mode))
        elif strategy == "continued_fraction":
            out.append(_cf_pair(rf, max_mode))
        elif strategy == "rounded":
            m = max(1, min(max_mode, int(round(rf))))
            out.append((m, 1))
        else:  # best_simple
            out.append(_brute_force_pair(rf, max_mode))
    return out


def _sb_pair(r: float, max_mode: int) -> ModePair:
    f = Fraction(r).limit_denominator(max_mode)
    m, n = f.numerator, f.denominator
    if m > max_mode:
        # Clamp the numerator while preserving the ratio as best we can.
        m = max_mode
        n = max(1, int(round(m / r)))
    return (max(1, m), max(1, n))


def _cf_pair(r: float, max_mode: int) -> ModePair:
    """Best continued-fraction convergent with both terms <= max_mode."""
    a = []
    x = float(r)
    for _ in range(64):
        ai = int(math.floor(x))
        a.append(ai)
        frac = x - ai
        if frac < 1e-15:
            break
        x = 1.0 / frac

    h_prev, h_curr = 1, a[0]
    k_prev, k_curr = 0, 1
    best = (max(1, h_curr), 1)
    for ai in a[1:]:
        h_next = ai * h_curr + h_prev
        k_next = ai * k_curr + k_prev
        if h_next > max_mode or k_next > max_mode:
            break
        best = (h_next, k_next)
        h_prev, h_curr = h_curr, h_next
        k_prev, k_curr = k_curr, k_next
    return best


def _brute_force_pair(r: float, max_mode: int) -> ModePair:
    best_pair: ModePair = (1, 1)
    best_err = float("inf")
    for m in range(1, max_mode + 1):
        for n in range(1, max_mode + 1):
            err = abs(m / n - r)
            if err < best_err:
                best_err = err
                best_pair = (m, n)
                if err == 0.0:
                    return best_pair
    return best_pair


# ============================================================== rectangular


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


def chladni_field_rectangular(
    modes: Sequence[ModePair],
    amps: Optional[Sequence[float]] = None,
    phases: Optional[Sequence[float]] = None,
    Lx: float = 1.0,
    Ly: float = 1.0,
    resolution: int = 256,
) -> GeometryData:
    """Free-edge rectangular plate displacement.

    ``u(x, y) = Σ_k A_k · cos(m_k π x / Lx) · cos(n_k π y / Ly) · cos(φ_k)``

    Parameters
    ----------
    modes : sequence of (int, int)
        Mode pairs ``(m, n)``.
    amps : sequence of float, optional
        Amplitudes per mode. Defaults to uniform ``1 / n_modes``.
    phases : sequence of float, optional
        Phase per mode in radians. Defaults to zeros.
    Lx, Ly : float, default=1.0
        Plate dimensions.
    resolution : int, default=256
        Grid resolution along each axis. The output field is
        ``(resolution, resolution)``.

    Returns
    -------
    GeometryData
        ``geom_type='field_2d'`` with ``coordinates`` the ``(R, R)`` field
        and ``field_grid=(X, Y)`` meshgrid arrays.
    """
    if Lx <= 0 or Ly <= 0:
        raise ValueError("Lx and Ly must be > 0.")
    if resolution < 4:
        raise ValueError(f"resolution must be >= 4, got {resolution!r}.")
    if not modes:
        raise ValueError("modes must be non-empty.")

    n_modes = len(modes)
    a, p = _resolve_amps_phases(n_modes, amps, phases)

    x = np.linspace(0.0, Lx, resolution)
    y = np.linspace(0.0, Ly, resolution)
    X, Y = np.meshgrid(x, y, indexing="xy")

    field = np.zeros_like(X)
    for (m, n), Ai, phi in zip(modes, a, p):
        field += Ai * np.cos(m * np.pi * X / Lx) * np.cos(n * np.pi * Y / Ly) * np.cos(phi)

    return GeometryData(
        geom_type="field_2d",
        coordinates=field,
        field_grid=(X, Y),
        parameters={
            "modes": [tuple(map(int, m)) for m in modes],
            "amps": a.tolist(),
            "phases": p.tolist(),
            "Lx": float(Lx),
            "Ly": float(Ly),
            "resolution": int(resolution),
        },
        metadata={"kind": "chladni_field_rectangular",
                  "plate": "rectangular", "n_modes": n_modes},
    )


# ================================================================== circular


@lru_cache(maxsize=128)
def _bessel_zero(n: int, m: int) -> float:
    """Cached lookup of the m-th positive zero of ``J_n``. m is 1-indexed."""
    return float(jn_zeros(n, m)[-1])


def chladni_field_circular(
    modes_radial: Sequence[int],
    modes_angular: Sequence[int],
    amps: Optional[Sequence[float]] = None,
    phases: Optional[Sequence[float]] = None,
    R: float = 1.0,
    resolution: int = 256,
) -> GeometryData:
    """Clamped circular membrane displacement.

    For each mode ``k``: ``u_k(r, θ) = J_{n_k}(α_{n_k, m_k} · r / R) · cos(n_k · θ)``,
    where ``α_{n, m}`` is the m-th positive zero of ``J_n`` (m is 1-indexed).
    Outside the disk the field is set to ``NaN``.

    Parameters
    ----------
    modes_radial : sequence of int
        Radial mode index (``m``, 1-indexed). Same length as ``modes_angular``.
    modes_angular : sequence of int
        Angular mode index (``n``, ``≥ 0``). Same length as ``modes_radial``.
    amps, phases : sequences of float, optional
    R : float, default=1.0
        Disk radius.
    resolution : int, default=256
        Square-grid resolution covering ``[-R, R]^2``.

    Returns
    -------
    GeometryData
        ``geom_type='field_2d'``. Values outside the disk are ``NaN``.
    """
    if len(modes_radial) != len(modes_angular):
        raise ValueError(
            "modes_radial and modes_angular must have the same length."
        )
    if R <= 0:
        raise ValueError("R must be > 0.")
    if resolution < 4:
        raise ValueError(f"resolution must be >= 4, got {resolution!r}.")
    if not modes_radial:
        raise ValueError("modes_radial / modes_angular must be non-empty.")

    n_modes = len(modes_radial)
    a, p = _resolve_amps_phases(n_modes, amps, phases)

    x = np.linspace(-R, R, resolution)
    y = np.linspace(-R, R, resolution)
    X, Y = np.meshgrid(x, y, indexing="xy")
    r = np.hypot(X, Y)
    theta = np.arctan2(Y, X)
    inside = r <= R

    field = np.zeros_like(X)
    for (m, n), Ai, phi in zip(zip(modes_radial, modes_angular), a, p):
        if m < 1:
            raise ValueError(f"Radial mode index must be >= 1, got {m!r}.")
        if n < 0:
            raise ValueError(f"Angular mode index must be >= 0, got {n!r}.")
        alpha = _bessel_zero(int(n), int(m))
        radial = jn(int(n), alpha * r / R)
        angular = np.cos(int(n) * theta)
        field += Ai * radial * angular * np.cos(phi)

    field = np.where(inside, field, np.nan)

    return GeometryData(
        geom_type="field_2d",
        coordinates=field,
        field_grid=(X, Y),
        parameters={
            "modes_radial": list(map(int, modes_radial)),
            "modes_angular": list(map(int, modes_angular)),
            "amps": a.tolist(),
            "phases": p.tolist(),
            "R": float(R),
            "resolution": int(resolution),
        },
        metadata={"kind": "chladni_field_circular",
                  "plate": "circular", "n_modes": n_modes},
    )


# =================================================================== polygon


def _polygon_vertices(n_sides: int, radius: float) -> np.ndarray:
    """Vertices of a regular polygon, oriented counterclockwise."""
    angles = 2 * np.pi * np.arange(n_sides) / n_sides + np.pi / 2
    return np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=1)


def _rasterize_polygon(
    vertices: np.ndarray, resolution: int, radius: float
) -> np.ndarray:
    """Boolean mask of the polygon interior on a square grid covering [-radius, radius]^2."""
    x = np.linspace(-radius, radius, resolution)
    y = np.linspace(-radius, radius, resolution)
    X, Y = np.meshgrid(x, y, indexing="xy")
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)

    # Ray-casting / winding test using the standard "point in polygon" sign
    # check against each edge. For convex polygons this is exact.
    n = len(vertices)
    inside = np.ones(pts.shape[0], dtype=bool)
    for i in range(n):
        a = vertices[i]
        b = vertices[(i + 1) % n]
        edge = b - a
        normal = np.array([-edge[1], edge[0]])  # interior-pointing normal for CCW.
        rel = pts - a
        # Strictly inside / on edge: dot >= 0
        inside &= (rel @ normal) >= -1e-12
    return inside.reshape((resolution, resolution))


def _polygon_eigenmodes_fdm(
    mask: np.ndarray, k_modes: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Lowest-k Dirichlet eigenmodes of -∇² on the masked grid (FDM).

    Returns ``(eigvals, eigvecs)`` where ``eigvecs`` has shape
    ``(n_interior, k_modes)``. Each column is one eigenmode value at the
    interior grid points (in row-major order, i.e. ``mask.flatten()`` order).
    """
    Ny, Nx = mask.shape
    flat_mask = mask.ravel()
    interior = np.where(flat_mask)[0]
    if interior.size == 0:
        raise ValueError("Polygon mask has no interior cells; increase resolution.")
    if interior.size <= k_modes:
        raise ValueError(
            f"Need at least {k_modes + 1} interior cells; got {interior.size}. "
            "Increase resolution."
        )

    # Map full-grid index -> compressed interior index.
    idx_map = -np.ones(flat_mask.size, dtype=np.int64)
    idx_map[interior] = np.arange(interior.size)

    rows: List[int] = []
    cols: List[int] = []
    vals: List[float] = []
    # 5-point Laplacian with Dirichlet BC: missing neighbors contribute zero
    # (i.e. value 0 outside, multiplied by their off-diagonal coefficient).
    for k in interior:
        ci = idx_map[k]
        rows.append(ci)
        cols.append(ci)
        vals.append(4.0)
        for d in (-1, 1, -Nx, Nx):
            j = k + d
            if j < 0 or j >= flat_mask.size:
                continue
            # Avoid wraparound across rows for the ±1 neighbours.
            if d == -1 and (k % Nx) == 0:
                continue
            if d == 1 and ((k + 1) % Nx) == 0:
                continue
            if flat_mask[j]:
                rows.append(ci)
                cols.append(int(idx_map[j]))
                vals.append(-1.0)

    n = interior.size
    L = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    # eigsh requires a symmetric operator and k < n. Use shift-invert for the
    # smallest eigenvalues; sigma=0 with "LM" gives the smallest by magnitude.
    eigvals, eigvecs = eigsh(L, k=k_modes, sigma=0.0, which="LM")
    order = np.argsort(eigvals)
    return eigvals[order], eigvecs[:, order]


def chladni_field_polygon(
    modes: Sequence[int],
    n_sides: int,
    amps: Optional[Sequence[float]] = None,
    phases: Optional[Sequence[float]] = None,
    radius: float = 1.0,
    resolution: int = 128,
    solver: str = "fdm",
) -> GeometryData:
    """Clamped regular polygon membrane displacement.

    Solves the Dirichlet eigenproblem ``-∇²ψ = λψ`` numerically on a
    rasterized polygon mask, then sums a chosen subset of eigenmodes.

    Parameters
    ----------
    modes : sequence of int
        0-indexed eigenmode indices (``0`` is the fundamental).
    n_sides : int
        Polygon side count, ``>= 3``.
    amps, phases : sequences of float, optional
    radius : float, default=1.0
        Circumradius of the polygon.
    resolution : int, default=128
        Bounding-square grid resolution. Lower than the rectangular default
        because the FDM eigenproblem is ``O(n²)`` to ``O(n³)`` in the
        interior cell count.
    solver : {'fdm', 'fem'}, default='fdm'
        ``'fem'`` requires the optional ``scikit-fem`` dependency and is
        not yet wired through; selecting it raises :class:`NotImplementedError`.

    Returns
    -------
    GeometryData
        ``geom_type='field_2d'``. Cells outside the polygon are ``NaN``.
    """
    if solver == "fem":
        raise NotImplementedError(
            "FEM solver is not implemented yet. Use solver='fdm' (default)."
        )
    if solver != "fdm":
        raise ValueError(f"Unknown solver {solver!r}; expected 'fdm' or 'fem'.")
    if n_sides < 3:
        raise ValueError(f"n_sides must be >= 3, got {n_sides!r}.")
    if radius <= 0:
        raise ValueError("radius must be > 0.")
    if resolution < 16:
        raise ValueError(f"resolution must be >= 16, got {resolution!r}.")
    if not modes:
        raise ValueError("modes must be non-empty.")
    modes_idx = [int(m) for m in modes]
    if any(m < 0 for m in modes_idx):
        raise ValueError("Mode indices must be >= 0.")

    n_modes = len(modes_idx)
    a, p = _resolve_amps_phases(n_modes, amps, phases)

    vertices = _polygon_vertices(n_sides, radius)
    mask = _rasterize_polygon(vertices, resolution, radius)

    k_needed = max(modes_idx) + 1
    eigvals, eigvecs = _polygon_eigenmodes_fdm(mask, k_needed)

    x = np.linspace(-radius, radius, resolution)
    y = np.linspace(-radius, radius, resolution)
    X, Y = np.meshgrid(x, y, indexing="xy")

    interior_idx = np.where(mask.ravel())[0]
    full_field = np.full(mask.size, np.nan)
    accum = np.zeros(interior_idx.size, dtype=np.float64)
    for i, mi in enumerate(modes_idx):
        # Normalize each eigenmode to peak |value| = 1 so amplitudes are comparable.
        vec = eigvecs[:, mi]
        peak = np.max(np.abs(vec))
        if peak > 0:
            vec = vec / peak
        accum += a[i] * vec * np.cos(p[i])
    full_field[interior_idx] = accum
    field = full_field.reshape(mask.shape)

    return GeometryData(
        geom_type="field_2d",
        coordinates=field,
        field_grid=(X, Y),
        parameters={
            "modes": modes_idx,
            "n_sides": int(n_sides),
            "amps": a.tolist(),
            "phases": p.tolist(),
            "radius": float(radius),
            "resolution": int(resolution),
            "solver": solver,
        },
        metadata={
            "kind": "chladni_field_polygon",
            "plate": "polygon",
            "n_modes": n_modes,
            "eigenvalues": eigvals[modes_idx].tolist(),
        },
    )


# ==================================================================== 3D box


def chladni_field_3d_box(
    modes_3d: Sequence[ModeTriple],
    amps: Optional[Sequence[float]] = None,
    phases: Optional[Sequence[float]] = None,
    dimensions: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    resolution: int = 48,
) -> GeometryData:
    """Standing-wave displacement field in a 3-D box (Dirichlet BC).

    ``u(x, y, z) = Σ_k A_k · sin(l_k π x / Lx) · sin(m_k π y / Ly)
    · sin(n_k π z / Lz) · cos(φ_k)``

    Parameters
    ----------
    modes_3d : sequence of (int, int, int)
        Mode triples ``(l, m, n)`` with all entries ``>= 1``.
    amps, phases : sequences of float, optional
    dimensions : (float, float, float), default=(1, 1, 1)
        ``(Lx, Ly, Lz)`` box dimensions.
    resolution : int, default=48
        Grid resolution per axis. Total memory scales as ``O(R³)``; the
        default 48³ ≈ 110k cells is a balance for a quick eigen-sum.

    Returns
    -------
    GeometryData
        ``geom_type='field_3d'`` with ``coordinates`` the ``(R, R, R)``
        scalar field and ``field_grid=(X, Y, Z)``.
    """
    if not modes_3d:
        raise ValueError("modes_3d must be non-empty.")
    if resolution < 4:
        raise ValueError(f"resolution must be >= 4, got {resolution!r}.")
    Lx, Ly, Lz = (float(d) for d in dimensions)
    if Lx <= 0 or Ly <= 0 or Lz <= 0:
        raise ValueError("All box dimensions must be > 0.")
    for triple in modes_3d:
        if len(triple) != 3 or any(int(c) < 1 for c in triple):
            raise ValueError(
                f"Each mode must be a triple of positive integers; got {triple!r}."
            )

    n_modes = len(modes_3d)
    a, p = _resolve_amps_phases(n_modes, amps, phases)

    x = np.linspace(0.0, Lx, resolution)
    y = np.linspace(0.0, Ly, resolution)
    z = np.linspace(0.0, Lz, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    field = np.zeros_like(X)
    for (l, m, n), Ai, phi in zip(modes_3d, a, p):
        field += (
            Ai
            * np.sin(l * np.pi * X / Lx)
            * np.sin(m * np.pi * Y / Ly)
            * np.sin(n * np.pi * Z / Lz)
            * np.cos(phi)
        )

    return GeometryData(
        geom_type="field_3d",
        coordinates=field,
        field_grid=(X, Y, Z),
        parameters={
            "modes_3d": [tuple(map(int, m)) for m in modes_3d],
            "amps": a.tolist(),
            "phases": p.tolist(),
            "dimensions": (Lx, Ly, Lz),
            "resolution": int(resolution),
        },
        metadata={"kind": "chladni_field_3d_box",
                  "plate": "box_3d", "n_modes": n_modes},
    )


# ============================================================ nodal extraction


def _require_skimage():
    try:
        import skimage.measure  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Nodal extraction requires `scikit-image`. Install with "
            "`pip install scikit-image` and retry."
        ) from exc


def chladni_nodal_lines(
    field_data: GeometryData,
    threshold: float = 1e-3,
) -> GeometryData:
    """Extract nodal lines from a 2-D Chladni field via marching squares.

    Parameters
    ----------
    field_data : GeometryData
        Must have ``geom_type='field_2d'``.
    threshold : float, default=1e-3
        Iso-value at which to extract contours. Cells where the field is
        ``NaN`` (e.g. outside a circular plate) are treated as the boundary
        and skipped during extraction.

    Returns
    -------
    GeometryData
        ``geom_type='curve_set_2d'`` — one ``(N_i, 2)`` array per
        contour, with coordinates in the same units as the input
        ``field_grid``.
    """
    if field_data.geom_type != "field_2d":
        raise ValueError(
            f"chladni_nodal_lines expects geom_type='field_2d', got "
            f"{field_data.geom_type!r}."
        )
    _require_skimage()
    from skimage.measure import find_contours  # type: ignore

    field = np.asarray(field_data.coordinates, dtype=np.float64)
    safe = np.where(np.isfinite(field), field, threshold + 1.0)

    contours_idx = find_contours(safe, level=threshold)

    if field_data.field_grid is not None:
        X, Y = field_data.field_grid
        x_axis = X[0, :]
        y_axis = Y[:, 0]
    else:
        x_axis = np.arange(field.shape[1])
        y_axis = np.arange(field.shape[0])

    curves: List[np.ndarray] = []
    for c in contours_idx:
        # find_contours returns (row, col) -> (y_idx, x_idx).
        rows = c[:, 0]
        cols = c[:, 1]
        ys = np.interp(rows, np.arange(len(y_axis)), y_axis)
        xs = np.interp(cols, np.arange(len(x_axis)), x_axis)
        curves.append(np.stack([xs, ys], axis=1))

    return GeometryData(
        geom_type="curve_set_2d",
        coordinates=curves,
        parameters={"threshold": float(threshold)},
        metadata={
            "kind": "nodal_lines",
            "n_contours": len(curves),
            "source_plate": field_data.metadata.get("plate"),
        },
    )


def chladni_nodal_surfaces(
    field_3d: GeometryData,
    threshold: float = 1e-3,
) -> GeometryData:
    """Extract a 2-D nodal surface mesh from a 3-D Chladni field.

    Parameters
    ----------
    field_3d : GeometryData
        Must have ``geom_type='field_3d'``.
    threshold : float, default=1e-3
        Iso-value of the marching-cubes extraction.

    Returns
    -------
    GeometryData
        ``geom_type='mesh_3d'`` with vertex coordinates in the same units
        as the input ``field_grid`` and triangle ``faces``.
    """
    if field_3d.geom_type != "field_3d":
        raise ValueError(
            f"chladni_nodal_surfaces expects geom_type='field_3d', got "
            f"{field_3d.geom_type!r}."
        )
    _require_skimage()
    from skimage.measure import marching_cubes  # type: ignore

    field = np.asarray(field_3d.coordinates, dtype=np.float64)
    safe = np.where(np.isfinite(field), field, threshold + 1.0)

    if field_3d.field_grid is not None:
        X, Y, Z = field_3d.field_grid
        spacing = (
            float(X[1, 0, 0] - X[0, 0, 0]) if X.shape[0] > 1 else 1.0,
            float(Y[0, 1, 0] - Y[0, 0, 0]) if Y.shape[1] > 1 else 1.0,
            float(Z[0, 0, 1] - Z[0, 0, 0]) if Z.shape[2] > 1 else 1.0,
        )
        origin = (float(X[0, 0, 0]), float(Y[0, 0, 0]), float(Z[0, 0, 0]))
    else:
        spacing = (1.0, 1.0, 1.0)
        origin = (0.0, 0.0, 0.0)

    verts, faces, _, _ = marching_cubes(safe, level=threshold, spacing=spacing)
    verts = verts + np.asarray(origin)

    return GeometryData(
        geom_type="mesh_3d",
        coordinates=verts,
        faces=faces.astype(np.int64),
        parameters={"threshold": float(threshold)},
        metadata={
            "kind": "nodal_surface",
            "n_vertices": int(verts.shape[0]),
            "n_faces": int(faces.shape[0]),
            "source_plate": field_3d.metadata.get("plate"),
        },
    )


# ================================================================== adapters


_PLATE_KINDS = {"rectangular", "circular", "polygon", "box_3d"}


def chladni_temporal(
    input: HarmonicInput,
    t: float,
    plate: str = "rectangular",
    mode_strategy: str = "stern_brocot",
    max_mode: int = 20,
    **plate_kwargs: Any,
) -> GeometryData:
    """Chladni field at a specific time ``t``, using ``input`` as the mode source.

    The phases used in the field are shifted by ``2π · f_k · t`` for each
    component (with ``f_k`` taken from ``input.peaks``), implementing the
    standing-wave time evolution. Suitable for assembling a time sequence
    of frames.

    Parameters
    ----------
    input : HarmonicInput
    t : float
        Time in seconds.
    plate : {'rectangular', 'circular', 'polygon', 'box_3d'}, default='rectangular'
    mode_strategy : str, default='stern_brocot'
        Forwarded to :func:`ratios_to_modes`.
    max_mode : int, default=20
    **plate_kwargs
        Forwarded to the underlying field builder.

    Returns
    -------
    GeometryData
    """
    peaks = input.to_peaks()
    drift = 2.0 * np.pi * peaks * float(t)
    base_phases = (
        np.asarray(input.phases, dtype=np.float64)
        if input.phases is not None
        else np.zeros(len(peaks))
    )
    phases = base_phases + drift
    return chladni_from_input(
        input,
        plate=plate,
        plate_kwargs={"phases": phases.tolist(), **plate_kwargs},
        mode_strategy=mode_strategy,
        max_mode=max_mode,
    )


def chladni_from_input(
    input: HarmonicInput,
    plate: str = "rectangular",
    plate_kwargs: Optional[dict] = None,
    mode_strategy: str = "stern_brocot",
    max_mode: int = 20,
) -> GeometryData:
    """Build a Chladni field from a :class:`HarmonicInput`.

    The input's ratios are mapped to mode pairs (or triples, for ``box_3d``)
    via :func:`ratios_to_modes`; amplitudes are taken from
    ``input.normalized_amplitudes()``; phases default to the input's
    ``phases`` (or zero).

    Parameters
    ----------
    input : HarmonicInput
    plate : {'rectangular', 'circular', 'polygon', 'box_3d'}, default='rectangular'
    plate_kwargs : dict, optional
        Extra keyword arguments forwarded to the underlying field builder
        (e.g. ``Lx``, ``Ly``, ``resolution``, ``n_sides``).
    mode_strategy : str, default='stern_brocot'
    max_mode : int, default=20
    """
    if plate not in _PLATE_KINDS:
        raise ValueError(
            f"plate must be one of {sorted(_PLATE_KINDS)}, got {plate!r}."
        )
    plate_kwargs = dict(plate_kwargs) if plate_kwargs else {}

    ratios = input.to_ratios()
    amps = input.normalized_amplitudes().tolist()
    pairs = ratios_to_modes(ratios, strategy=mode_strategy, max_mode=max_mode)

    user_phases = plate_kwargs.pop("phases", None)
    phases = (
        list(user_phases)
        if user_phases is not None
        else (
            list(input.phases)
            if input.phases is not None
            else [0.0] * len(pairs)
        )
    )

    if plate == "rectangular":
        return chladni_field_rectangular(
            modes=pairs, amps=amps, phases=phases, **plate_kwargs
        )
    if plate == "circular":
        modes_radial = [m for m, _ in pairs]
        modes_angular = [n for _, n in pairs]
        return chladni_field_circular(
            modes_radial=modes_radial,
            modes_angular=modes_angular,
            amps=amps,
            phases=phases,
            **plate_kwargs,
        )
    if plate == "polygon":
        # For polygons, the (m, n) pair is collapsed to a single mode index
        # m * max_mode + (n - 1); this keeps the ordering deterministic
        # without re-running the ratio→mode strategy.
        modes_idx = [int(m) - 1 for m, _ in pairs]
        return chladni_field_polygon(
            modes=modes_idx,
            amps=amps,
            phases=phases,
            **plate_kwargs,
        )
    # box_3d
    triples: List[ModeTriple] = []
    for (m, n) in pairs:
        triples.append((int(m), int(n), max(1, int(round((m + n) / 2)))))
    return chladni_field_3d_box(
        modes_3d=triples, amps=amps, phases=phases, **plate_kwargs
    )


# ================================================================ Medium API


class RigidPlate(_Medium):
    """Eigenmode-family medium: chord projected onto a clamped/free-edge plate.

    Wraps :func:`chladni_from_input` in the pipeline contract defined by
    :class:`biotuner.harmonic_geometry.media.base.Medium`. Constructor
    arguments configure the domain, mode-mapping strategy, and any
    domain-specific knobs; per-call overrides are forwarded to the
    underlying ``chladni_field_*`` builder.

    Parameters
    ----------
    domain : Domain, optional
        One of :class:`Rectangular`, :class:`Circular`,
        :class:`PolygonDomain`, :class:`Box3D`. Defaults to
        ``Rectangular(1.0, 1.0)``.
    mode_strategy : str, default='stern_brocot'
        Forwarded to :func:`ratios_to_modes`.
    max_mode : int, default=20
        Forwarded to :func:`ratios_to_modes`.
    resolution : int, optional
        Grid resolution. Defaults to the domain-appropriate value
        (256 for 2-D, 48 for 3-D).
    """

    family = "eigenmode"

    def __init__(
        self,
        *,
        domain: Optional[Any] = None,
        mode_strategy: str = "stern_brocot",
        max_mode: int = 20,
        resolution: Optional[int] = None,
    ) -> None:
        # Lazy import to keep base / domain types optional from this module.
        from biotuner.harmonic_geometry.media.base import (
            Box3D,
            Circular,
            Domain,
            PolygonDomain,
            Rectangular,
        )

        if domain is None:
            domain = Rectangular()
        if not isinstance(domain, Domain):
            raise TypeError(
                f"domain must be a Domain instance; got {type(domain).__name__}."
            )
        if not isinstance(domain, (Rectangular, Circular, PolygonDomain, Box3D)):
            raise TypeError(
                f"{type(domain).__name__} is not a supported RigidPlate domain. "
                "Use Rectangular, Circular, PolygonDomain, or Box3D."
            )
        self.domain = domain
        self.mode_strategy = mode_strategy
        self.max_mode = int(max_mode)
        self.resolution = resolution

    def respond(
        self,
        forcing: HarmonicInput,
        **overrides: Any,
    ) -> GeometryData:
        """Project ``forcing`` onto the plate's eigenmodes.

        Parameters
        ----------
        forcing : HarmonicInput
            Chord to project.
        **overrides
            Per-call overrides forwarded to the underlying
            ``chladni_field_*`` builder (e.g. ``resolution``, ``Lx``,
            ``Ly``, ``R``, ``n_sides``, ``dimensions``).
        """
        if not isinstance(forcing, HarmonicInput):
            raise TypeError(
                "RigidPlate.respond requires a HarmonicInput; got "
                f"{type(forcing).__name__}. Use an eigenmode/wave_field/"
                "parametric medium as the upstream stage."
            )

        from biotuner.harmonic_geometry.media.base import (
            Box3D,
            Circular,
            PolygonDomain,
            Rectangular,
        )

        plate_kwargs: dict = {}
        if isinstance(self.domain, Rectangular):
            plate = "rectangular"
            plate_kwargs["Lx"] = self.domain.Lx
            plate_kwargs["Ly"] = self.domain.Ly
            if self.resolution is not None:
                plate_kwargs["resolution"] = self.resolution
        elif isinstance(self.domain, Circular):
            plate = "circular"
            plate_kwargs["R"] = self.domain.R
            if self.resolution is not None:
                plate_kwargs["resolution"] = self.resolution
        elif isinstance(self.domain, PolygonDomain):
            plate = "polygon"
            plate_kwargs["n_sides"] = self.domain.n_sides
            plate_kwargs["radius"] = self.domain.radius
            if self.resolution is not None:
                plate_kwargs["resolution"] = self.resolution
        elif isinstance(self.domain, Box3D):
            plate = "box_3d"
            plate_kwargs["dimensions"] = (
                self.domain.Lx,
                self.domain.Ly,
                self.domain.Lz,
            )
            if self.resolution is not None:
                plate_kwargs["resolution"] = self.resolution
        else:
            # Already validated in __init__; defensive.
            raise TypeError(f"Unsupported domain {type(self.domain).__name__}.")

        plate_kwargs.update(overrides)

        return chladni_from_input(
            forcing,
            plate=plate,
            plate_kwargs=plate_kwargs,
            mode_strategy=self.mode_strategy,
            max_mode=self.max_mode,
        )

    def default_source(self) -> None:
        return None

    def __call__(self, forcing: HarmonicInput, **overrides: Any) -> GeometryData:
        return self.respond(forcing, **overrides)

    def __repr__(self) -> str:
        return (
            f"RigidPlate(domain={self.domain!r}, "
            f"mode_strategy={self.mode_strategy!r}, max_mode={self.max_mode})"
        )
