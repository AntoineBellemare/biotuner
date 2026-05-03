"""
biotuner.harmonic_geometry.geometry_3d
=======================================
Phase 7: 3-D geometry structures derived from harmonic inputs.

Functions
---------
lissajous_tube       -- tube mesh extruded around a 3-D Lissajous curve
harmonic_knot        -- ratio-derived torus knot extruded into a tube mesh
harmonic_surface     -- deformed parametric surface (torus / sphere / cylinder)
lsystem_3d           -- 3-D turtle L-system branching tree
recursive_polyhedron -- Koch-style recursively stellated Platonic solid
harmonic_point_cloud -- point cloud on sphere / torus with harmonic-phase modulation
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import numpy as np

from biotuner.harmonic_geometry._utils import coerce_ratio
from biotuner.harmonic_geometry.geometry_data import GeometryData
from biotuner.harmonic_geometry.inputs import HarmonicInput

# ═══════════════════════════════════════════ internal helpers ══════════════════


def _ratio_to_fraction(r) -> Fraction:
    raw = coerce_ratio(r)
    if isinstance(raw, Fraction):
        return raw.limit_denominator(12)
    return Fraction(float(raw)).limit_denominator(12)


def _ratio_to_angle_deg(r) -> float:
    """360 / (p + q) for ratio p/q — same formula as Phase-5 lsystem."""
    f = _ratio_to_fraction(r)
    return 360.0 / max(f.numerator + f.denominator, 3)


# ── Parallel-transport frame ───────────────────────────────────────────────────


def _parallel_transport(
    curve: np.ndarray,
    closed: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(tangents, normals, binormals)`` for a ``(N, 3)`` polyline.

    Uses Bishop's parallel-transport frame (no Frenet flip when curvature
    vanishes). When ``closed=True``, the residual twist between the first
    and last frames is distributed evenly across the curve so that the
    seam closes without a discontinuity (twist correction).
    """
    N = len(curve)
    tangents = np.zeros((N, 3), dtype=np.float64)
    if closed:
        tangents[:-1] = curve[1:] - curve[:-1]
        tangents[-1]  = curve[0] - curve[-1]
    else:
        tangents[1:-1] = curve[2:] - curve[:-2]
        tangents[0]    = curve[1]  - curve[0]
        tangents[-1]   = curve[-1] - curve[-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1e-12, norms)
    tangents /= norms

    # Seed normal perpendicular to tangents[0]
    t0 = tangents[0]
    ref = np.array([1.0, 0.0, 0.0]) if abs(t0[1]) > 0.1 or abs(t0[2]) > 0.1 \
        else np.array([0.0, 1.0, 0.0])
    n0 = ref - np.dot(ref, t0) * t0
    n_len = np.linalg.norm(n0)
    n0 = n0 / (n_len if n_len > 1e-12 else 1.0)

    normals = np.zeros((N, 3), dtype=np.float64)
    normals[0] = n0
    for i in range(1, N):
        tp, tc = tangents[i - 1], tangents[i]
        b = np.cross(tp, tc)
        b_len = np.linalg.norm(b)
        if b_len < 1e-12:
            normals[i] = normals[i - 1]
        else:
            b /= b_len
            cos_a = float(np.clip(np.dot(tp, tc), -1.0, 1.0))
            sin_a = math.sqrt(max(0.0, 1.0 - cos_a ** 2))
            n_prev = normals[i - 1]
            n_new = (
                n_prev * cos_a
                + np.cross(b, n_prev) * sin_a
                + b * np.dot(b, n_prev) * (1.0 - cos_a)
            )
            n_len = np.linalg.norm(n_new)
            normals[i] = n_new / (n_len if n_len > 1e-12 else 1.0)

    # Closed-curve twist correction: redistribute residual rotation between
    # normals[N-1] and the parallel transport of normals[0] back to tangents[N-1].
    if closed and N > 2:
        # Transport normals[0] from tangents[0] one step further to tangents[-1]
        # The closing step is tangents[-1] -> tangents[0]; the "ideal" final
        # normal (no twist) should match normals[0] when transported back.
        # Compute the angle between normals[-1] and where it *should* be.
        t_last = tangents[-1]
        t_first = tangents[0]
        b = np.cross(t_last, t_first)
        b_len = np.linalg.norm(b)
        if b_len > 1e-12:
            b /= b_len
            cos_a = float(np.clip(np.dot(t_last, t_first), -1.0, 1.0))
            sin_a = math.sqrt(max(0.0, 1.0 - cos_a ** 2))
            n_target = (normals[-1] * cos_a
                        + np.cross(b, normals[-1]) * sin_a
                        + b * np.dot(b, normals[-1]) * (1.0 - cos_a))
            n_target /= max(np.linalg.norm(n_target), 1e-12)
        else:
            n_target = normals[-1]
        # angular twist about t_first between n_target and normals[0]
        cos_t = float(np.clip(np.dot(n_target, normals[0]), -1.0, 1.0))
        cross = np.cross(n_target, normals[0])
        sign  = 1.0 if np.dot(cross, t_first) >= 0.0 else -1.0
        twist = sign * math.acos(cos_t)
        # Apply linearly distributed counter-rotation to all frames (Rodrigues)
        ks = np.arange(N) / N
        thetas = twist * ks
        c = np.cos(thetas)[:, None]
        s = np.sin(thetas)[:, None]
        # Rotate each normal[i] about tangents[i] by thetas[i]
        dots = np.einsum("ij,ij->i", tangents, normals)[:, None]
        normals = (normals * c
                   + np.cross(tangents, normals) * s
                   + tangents * dots * (1.0 - c))
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)

    binormals = np.cross(tangents, normals)
    bn = np.linalg.norm(binormals, axis=1, keepdims=True)
    bn = np.where(bn < 1e-12, 1e-12, bn)
    binormals /= bn
    return tangents, normals, binormals


def _tube_mesh(
    curve: np.ndarray,
    radius: float,
    n_sides: int,
    closed: bool = False,
    radii: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Triangulated tube mesh around a 3-D polyline using parallel transport.

    Fully vectorised: vertex generation uses broadcasting, face indices are
    built with :func:`numpy.indices` rather than nested Python loops.
    """
    curve = np.asarray(curve, dtype=np.float64)
    N = len(curve)
    _, normals, binormals = _parallel_transport(curve, closed=closed)

    if radii is None:
        r_arr = np.full(N, float(radius), dtype=np.float64)
    else:
        r_arr = np.asarray(radii, dtype=np.float64)

    angles = np.linspace(0.0, 2.0 * math.pi, n_sides, endpoint=False)
    cos_a = np.cos(angles)[None, :, None]                # (1, n_sides, 1)
    sin_a = np.sin(angles)[None, :, None]                # (1, n_sides, 1)
    nrm_b = normals[:, None, :]                          # (N, 1, 3)
    bnm_b = binormals[:, None, :]                        # (N, 1, 3)
    rad_b = r_arr[:, None, None]                         # (N, 1, 1)
    offsets = rad_b * (cos_a * nrm_b + sin_a * bnm_b)    # (N, n_sides, 3)
    verts = (curve[:, None, :] + offsets).reshape(N * n_sides, 3)

    n_rings = N if closed else N - 1
    # Build face indices via meshgrid (n_rings × n_sides → 2 triangles each)
    ii, jj = np.meshgrid(np.arange(n_rings), np.arange(n_sides), indexing="ij")
    if closed:
        ii1 = (ii + 1) % N
    else:
        ii1 = ii + 1
    jj1 = (jj + 1) % n_sides
    v00 = ii  * n_sides + jj
    v01 = ii  * n_sides + jj1
    v10 = ii1 * n_sides + jj
    v11 = ii1 * n_sides + jj1
    tri_a = np.stack([v00, v10, v01], axis=-1)           # (n_rings, n_sides, 3)
    tri_b = np.stack([v01, v10, v11], axis=-1)
    faces = np.stack([tri_a, tri_b], axis=2).reshape(-1, 3).astype(np.int64)
    return verts, faces


def _grid_to_mesh(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    closed_u: bool = False,
    closed_v: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a parametric (Nu, Nv) grid to a triangulated mesh.

    Vectorised face-index construction via :func:`numpy.meshgrid`.
    """
    Nu, Nv = X.shape
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float64)
    nu_iter = Nu if closed_u else Nu - 1
    nv_iter = Nv if closed_v else Nv - 1
    if nu_iter <= 0 or nv_iter <= 0:
        return verts, np.empty((0, 3), dtype=np.int64)

    ii, jj = np.meshgrid(np.arange(nu_iter), np.arange(nv_iter), indexing="ij")
    ii1 = (ii + 1) % Nu if closed_u else ii + 1
    jj1 = (jj + 1) % Nv if closed_v else jj + 1
    v00 = ii  * Nv + jj
    v01 = ii  * Nv + jj1
    v10 = ii1 * Nv + jj
    v11 = ii1 * Nv + jj1
    tri_a = np.stack([v00, v10, v01], axis=-1)        # (nu_iter, nv_iter, 3)
    tri_b = np.stack([v01, v10, v11], axis=-1)
    faces = np.stack([tri_a, tri_b], axis=2).reshape(-1, 3).astype(np.int64)
    return verts, faces


# ── Platonic solids ────────────────────────────────────────────────────────────

def _platonic_solid(solid: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (vertices, faces) for a unit-sphere normalised Platonic solid."""
    if solid == "tetrahedron":
        v = np.array([
            [ 1.0,  1.0,  1.0],
            [ 1.0, -1.0, -1.0],
            [-1.0,  1.0, -1.0],
            [-1.0, -1.0,  1.0],
        ], dtype=np.float64)
        f = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int64)
    elif solid == "cube":
        c = 1.0 / math.sqrt(3.0)
        v = np.array([
            [-c, -c, -c], [ c, -c, -c], [ c,  c, -c], [-c,  c, -c],
            [-c, -c,  c], [ c, -c,  c], [ c,  c,  c], [-c,  c,  c],
        ], dtype=np.float64)
        f = np.array([
            [0, 2, 1], [0, 3, 2],   # bottom
            [4, 5, 6], [4, 6, 7],   # top
            [0, 1, 5], [0, 5, 4],   # front
            [2, 3, 7], [2, 7, 6],   # back
            [0, 4, 7], [0, 7, 3],   # left
            [1, 2, 6], [1, 6, 5],   # right
        ], dtype=np.int64)
    elif solid == "icosahedron":
        phi = (1.0 + math.sqrt(5.0)) / 2.0
        raw = np.array([
            [ 0,  1,  phi], [ 0, -1,  phi], [ 0,  1, -phi], [ 0, -1, -phi],
            [ 1,  phi,  0], [-1,  phi,  0], [ 1, -phi,  0], [-1, -phi,  0],
            [ phi,  0,  1], [-phi,  0,  1], [ phi,  0, -1], [-phi,  0, -1],
        ], dtype=np.float64)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        v = raw / norms
        f = np.array([
            [0, 1, 8], [0, 8, 4], [0, 4, 5], [0, 5, 9], [0, 9, 1],
            [1, 6, 8], [8, 6, 10], [8, 10, 4], [4, 10, 2], [4, 2, 5],
            [5, 2, 11], [5, 11, 9], [9, 11, 7], [9, 7, 1], [1, 7, 6],
            [3, 6, 7], [3, 10, 6], [3, 2, 10], [3, 11, 2], [3, 7, 11],
        ], dtype=np.int64)
    else:
        raise ValueError(f"solid must be 'tetrahedron', 'cube', or 'icosahedron', got {solid!r}.")
    v /= np.linalg.norm(v, axis=1, keepdims=True)  # project to unit sphere
    return v, f


# ── 3-D L-system helpers ───────────────────────────────────────────────────────

def _rot_U(H: np.ndarray, L: np.ndarray, U: np.ndarray, theta: float):
    """Rotate H and L around U by theta (yaw)."""
    c, s = math.cos(theta), math.sin(theta)
    return c * H - s * L, s * H + c * L, U


def _rot_L(H: np.ndarray, L: np.ndarray, U: np.ndarray, theta: float):
    """Rotate H and U around L by theta (pitch)."""
    c, s = math.cos(theta), math.sin(theta)
    return c * H + s * U, L, -s * H + c * U


def _rot_H(H: np.ndarray, L: np.ndarray, U: np.ndarray, theta: float):
    """Rotate L and U around H by theta (roll)."""
    c, s = math.cos(theta), math.sin(theta)
    return H, c * L - s * U, s * L + c * U


def _default_lsystem_3d_rule(n_extra: int) -> Dict[str, str]:
    """3-D branching grammar. n_extra = n_components - 2."""
    base_yaw   = "[+F][-F]"
    base_pitch = "[^F][&F]"
    if n_extra <= 0:
        return {"F": f"F{base_yaw}F"}
    if n_extra == 1:
        return {"F": f"F{base_yaw}F{base_pitch}"}
    roll_branch = "".join(f"[{'<' if k%2==0 else '>'}F]" for k in range(min(n_extra - 1, 2)))
    return {"F": f"F{base_yaw}{base_pitch}{roll_branch}F"}


def _apply_rules(axiom: str, rules: Dict[str, str], depth: int) -> str:
    s = axiom
    for _ in range(depth):
        s = "".join(rules.get(c, c) for c in s)
    return s


# ═══════════════════════════════════════════ public API ════════════════════════


def lissajous_tube(
    input: HarmonicInput,
    n_points: int = 800,
    n_periods: int = 6,
    tube_radius: float = 0.05,
    n_sides: int = 12,
) -> GeometryData:
    """Tube mesh extruded around a 3-D Lissajous curve.

    The first three ratio components drive the x, y, z frequencies.  Amplitude
    is mapped to tube-radius variation so louder components swell the tube.

    Parameters
    ----------
    input : HarmonicInput
    n_points : int, default=800
        Number of sample points along the curve.
    n_periods : int, default=6
        Number of full periods of the base (lowest-ratio) oscillation.
    tube_radius : float, default=0.05
        Base tube radius.
    n_sides : int, default=12
        Polygon sides for the tube cross-section.

    Returns
    -------
    GeometryData
        ``geom_type='mesh_3d'``.
    """
    if n_points < 8:
        raise ValueError(f"n_points must be >= 8, got {n_points!r}.")
    if tube_radius <= 0.0:
        raise ValueError(f"tube_radius must be > 0, got {tube_radius!r}.")
    if n_sides < 3:
        raise ValueError(f"n_sides must be >= 3, got {n_sides!r}.")

    ratios = [float(r) for r in input.to_ratios()]
    amps   = input.normalized_amplitudes().tolist()
    n = len(ratios)

    rx = ratios[0]           if n > 0 else 1.0
    ry = ratios[1]           if n > 1 else 2.0
    rz = ratios[2]           if n > 2 else 3.0
    ax = float(amps[0])      if n > 0 else 1.0
    ay = float(amps[1])      if n > 1 else 1.0
    az = float(amps[2])      if n > 2 else 1.0

    t = np.linspace(0.0, 2.0 * math.pi * n_periods, n_points)
    x = ax * np.sin(rx * t)
    y = ay * np.sin(ry * t + math.pi / 4.0)
    z = az * np.sin(rz * t + math.pi / 2.0)
    curve = np.stack([x, y, z], axis=1)

    # Amplitude-modulated radius: beats with ratio-frequency interference
    r_mod = np.full(n_points, tube_radius)
    if n > 3:
        for k in range(3, min(n, 6)):
            r_mod += tube_radius * 0.3 * float(amps[k]) * np.abs(np.sin(ratios[k] * t))

    verts, faces = _tube_mesh(curve, tube_radius, n_sides, radii=r_mod)
    return GeometryData(
        geom_type="mesh_3d",
        coordinates=verts,
        faces=faces,
        parameters={
            "n_points": n_points,
            "n_periods": n_periods,
            "tube_radius": tube_radius,
            "n_sides": n_sides,
        },
        metadata={
            "kind": "lissajous_tube",
            "rx": rx, "ry": ry, "rz": rz,
            "n_vertices": len(verts),
            "n_faces": len(faces),
        },
    )


def harmonic_knot(
    input: HarmonicInput,
    n_points: int = 600,
    tube_radius: float = 0.06,
    n_sides: int = 16,
    major_radius: float = 2.0,
    minor_radius: float = 0.7,
) -> GeometryData:
    """Torus knot T(p, q) derived from the dominant harmonic ratio.

    The simplest ratio ``p/q`` in the input determines the winding numbers.
    A 3/2 input (perfect fifth) gives a trefoil knot T(3, 2); 5/4 gives T(5, 4).
    Amplitude modulates the tube radius so louder harmonics thicken the knot.

    Parameters
    ----------
    input : HarmonicInput
    n_points : int, default=600
        Sample points along the knot curve.
    tube_radius : float, default=0.06
        Base tube radius.
    n_sides : int, default=16
        Cross-section polygon sides.
    major_radius : float, default=2.0
        Torus major radius (distance from torus centre to tube centre).
    minor_radius : float, default=0.7
        Torus minor radius (tube radius of the underlying torus).

    Returns
    -------
    GeometryData
        ``geom_type='mesh_3d'``.
    """
    if n_points < 8:
        raise ValueError(f"n_points must be >= 8, got {n_points!r}.")
    if tube_radius <= 0.0:
        raise ValueError(f"tube_radius must be > 0, got {tube_radius!r}.")
    if major_radius <= 0.0 or minor_radius <= 0.0:
        raise ValueError("major_radius and minor_radius must be > 0.")

    ratios = input.to_ratios()
    amps   = input.normalized_amplitudes()

    # Pick the most prominent non-trivial ratio as p/q
    order  = list(np.argsort(amps)[::-1])
    p, q   = 3, 2   # trefoil fallback
    for idx in order:
        f = _ratio_to_fraction(ratios[idx])
        if f.numerator != f.denominator and f.numerator > 0 and f.denominator > 0:
            p, q = f.numerator, f.denominator
            break

    t = np.linspace(0.0, 2.0 * math.pi, n_points, endpoint=False)
    x = (major_radius + minor_radius * np.cos(q * t)) * np.cos(p * t)
    y = (major_radius + minor_radius * np.cos(q * t)) * np.sin(p * t)
    z = minor_radius * np.sin(q * t)
    curve = np.stack([x, y, z], axis=1)

    # Amplitude-modulated radius along the knot
    r_mod = np.full(n_points, tube_radius)
    for k, (r_ratio, amp) in enumerate(zip(ratios, amps)):
        freq = float(_ratio_to_fraction(r_ratio).numerator)
        r_mod += tube_radius * 0.25 * float(amp) * np.abs(np.sin(freq * t))

    verts, faces = _tube_mesh(curve, tube_radius, n_sides, closed=True, radii=r_mod)
    return GeometryData(
        geom_type="mesh_3d",
        coordinates=verts,
        faces=faces,
        parameters={
            "p": p, "q": q,
            "n_points": n_points,
            "tube_radius": tube_radius,
            "n_sides": n_sides,
            "major_radius": major_radius,
            "minor_radius": minor_radius,
        },
        metadata={
            "kind": "harmonic_knot",
            "knot_type": f"T({p},{q})",
            "winding_p": int(p),
            "winding_q": int(q),
            "n_vertices": len(verts),
            "n_faces": len(faces),
        },
    )


def harmonic_surface(
    input: HarmonicInput,
    mode: str = "torus",
    resolution: int = 64,
) -> GeometryData:
    """Deformed parametric surface driven by harmonic ratio frequencies.

    Each ratio ``p/q`` contributes a standing-wave ripple whose angular
    frequency is (p, q) in the two surface parameters.  Amplitude controls
    the ripple depth.

    Parameters
    ----------
    input : HarmonicInput
    mode : {'torus', 'sphere', 'cylinder'}, default='torus'
        Base surface geometry.
    resolution : int, default=64
        Grid resolution per parameter axis (total vertices ≈ resolution²).

    Returns
    -------
    GeometryData
        ``geom_type='mesh_3d'``.
    """
    _MODES = {"torus", "sphere", "cylinder"}
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {_MODES}, got {mode!r}.")
    if resolution < 8:
        raise ValueError(f"resolution must be >= 8, got {resolution!r}.")

    ratios = [float(r) for r in input.to_ratios()]
    amps   = input.normalized_amplitudes().tolist()

    if mode == "torus":
        R, r_base = 2.0, 0.8
        u = np.linspace(0.0, 2.0 * math.pi, resolution, endpoint=False)
        v = np.linspace(0.0, 2.0 * math.pi, resolution, endpoint=False)
        U, V = np.meshgrid(u, v, indexing="ij")
        deform = np.zeros_like(U)
        for ratio, amp in zip(ratios, amps):
            frac = _ratio_to_fraction(ratio)
            m, n_ = frac.numerator, frac.denominator
            deform += float(amp) * 0.25 * r_base * np.cos(m * U + n_ * V)
        r_eff = r_base + deform
        X = (R + r_eff * np.cos(V)) * np.cos(U)
        Y = (R + r_eff * np.cos(V)) * np.sin(U)
        Z = r_eff * np.sin(V)
        verts, faces = _grid_to_mesh(X, Y, Z, closed_u=True, closed_v=True)

    elif mode == "sphere":
        theta = np.linspace(0.0, math.pi,       resolution)
        phi   = np.linspace(0.0, 2.0 * math.pi, resolution, endpoint=False)
        TH, PH = np.meshgrid(theta, phi, indexing="ij")
        deform = np.zeros_like(TH)
        for ratio, amp in zip(ratios, amps):
            frac = _ratio_to_fraction(ratio)
            l_ = min(frac.numerator,   8)
            m  = min(frac.denominator, max(l_, 1))
            deform += float(amp) * 0.35 * np.cos(l_ * TH) * np.cos(m * PH)
        r_eff = 1.0 + deform
        X = r_eff * np.sin(TH) * np.cos(PH)
        Y = r_eff * np.sin(TH) * np.sin(PH)
        Z = r_eff * np.cos(TH)
        verts, faces = _grid_to_mesh(X, Y, Z, closed_u=False, closed_v=True)

    else:  # cylinder
        u  = np.linspace(0.0, 2.0 * math.pi, resolution, endpoint=False)
        v  = np.linspace(-1.0, 1.0, resolution)
        U, V = np.meshgrid(u, v, indexing="ij")
        deform_r = np.zeros_like(U)
        deform_z = np.zeros_like(U)
        for ratio, amp in zip(ratios, amps):
            frac = _ratio_to_fraction(ratio)
            m, n_ = frac.numerator, frac.denominator
            deform_r += float(amp) * 0.2  * np.cos(m * U) * np.cos(n_ * math.pi * V)
            deform_z += float(amp) * 0.08 * np.sin(m * U + n_ * math.pi * V)
        r_eff = 1.0 + deform_r
        X = r_eff * np.cos(U)
        Y = r_eff * np.sin(U)
        Z = V + deform_z
        verts, faces = _grid_to_mesh(X, Y, Z, closed_u=True, closed_v=False)

    return GeometryData(
        geom_type="mesh_3d",
        coordinates=verts,
        faces=faces,
        parameters={"mode": mode, "resolution": resolution},
        metadata={"kind": "harmonic_surface", "mode": mode,
                  "n_vertices": len(verts), "n_faces": len(faces)},
    )


def lsystem_3d(
    input: HarmonicInput,
    depth: int = 3,
    step_length: float = 1.0,
    rules: Optional[Dict[str, str]] = None,
    axiom: str = "F",
) -> GeometryData:
    """3-D turtle L-system branching tree driven by harmonic ratios.

    The branch angle θ is derived from the dominant ratio ``360 / (p + q)``.
    The turtle supports the full six-degree-of-freedom rotation set:
    ``+/-`` yaw around U, ``^/&`` pitch around L, ``</>`` roll around H,
    ``|`` for a 180-degree U-turn, and ``[/]`` for state push/pop.

    Parameters
    ----------
    input : HarmonicInput
    depth : int, default=3
        L-system rewriting depth (keep ≤ 5 to avoid memory issues).
    step_length : float, default=1.0
        Length of each forward (F) segment.
    rules : dict, optional
        Custom symbol→replacement rules; defaults to a ratio-derived 3-D grammar.
    axiom : str, default='F'
        Starting string.

    Returns
    -------
    GeometryData
        ``geom_type='tree'`` with 3-D node coordinates ``(N, 3)``.
    """
    if depth < 1 or depth > 6:
        raise ValueError(f"depth must be in [1, 6], got {depth!r}.")
    if step_length <= 0.0:
        raise ValueError(f"step_length must be > 0, got {step_length!r}.")

    ratios = input.to_ratios()
    amps   = input.normalized_amplitudes()
    n = len(ratios)

    # Angle from dominant non-trivial ratio
    order = list(np.argsort(amps)[::-1])
    theta_deg = 25.0
    for idx in order:
        f = _ratio_to_fraction(ratios[idx])
        if abs(float(ratios[idx]) - 1.0) > 1e-6:
            theta_deg = _ratio_to_angle_deg(ratios[idx])
            break
    theta = math.radians(theta_deg)

    if rules is None:
        rules = _default_lsystem_3d_rule(n - 2)

    lstring = _apply_rules(axiom, rules, depth)

    # Turtle state
    pos  = np.zeros(3, dtype=np.float64)
    H    = np.array([0.0, 1.0, 0.0])   # heading: grow upward
    L    = np.array([1.0, 0.0, 0.0])   # left
    U    = np.array([0.0, 0.0, 1.0])   # up

    stack: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    pts:   List[np.ndarray] = []
    edges: List[List[int]]  = []

    for c in lstring:
        if c == "F":
            i_from = len(pts)
            pts.append(pos.copy())
            pos = pos + step_length * H
            pts.append(pos.copy())
            edges.append([i_from, i_from + 1])
        elif c == "+":
            H, L, U = _rot_U(H, L, U,  theta)
        elif c == "-":
            H, L, U = _rot_U(H, L, U, -theta)
        elif c == "^":
            H, L, U = _rot_L(H, L, U,  theta)
        elif c == "&":
            H, L, U = _rot_L(H, L, U, -theta)
        elif c == "<":
            H, L, U = _rot_H(H, L, U,  theta)
        elif c == ">":
            H, L, U = _rot_H(H, L, U, -theta)
        elif c == "|":
            H, L, U = _rot_U(H, L, U,  math.pi)
        elif c == "[":
            stack.append((pos.copy(), H.copy(), L.copy(), U.copy()))
        elif c == "]":
            if stack:
                pos, H, L, U = stack.pop()

    if not pts:
        coords = np.zeros((2, 3), dtype=np.float64)
        edge_arr = np.array([[0, 1]], dtype=np.int64)
    else:
        coords   = np.array(pts,   dtype=np.float64)
        edge_arr = np.array(edges, dtype=np.int64)

    return GeometryData(
        geom_type="tree",
        coordinates=coords,
        edges=edge_arr,
        parameters={
            "depth": depth,
            "step_length": step_length,
            "theta_deg": round(theta_deg, 4),
        },
        metadata={
            "kind": "lsystem_3d",
            "n_segments": len(edges),
            "lstring_len": len(lstring),
            "lstring_preview": lstring[:120],
        },
    )


def recursive_polyhedron(
    input: HarmonicInput,
    depth: int = 2,
    solid: Optional[str] = None,
    per_face_bump: bool = True,
    apex_twist: bool = True,
) -> GeometryData:
    """Koch-style recursively stellated Platonic solid, ratio-differentiated.

    At each depth level every triangular face is subdivided into four
    smaller triangles and a tetrahedral bump is raised at the centre face.
    By default each face's bump scale is keyed to the *closest harmonic
    ratio* (the ratio whose log2 value best matches the face's normal-vector
    polar angle), so chord-tones literally sculpt their own surface region.

    Parameters
    ----------
    input : HarmonicInput
    depth : int, default=2
        Recursion levels.  Faces quadruple each level; keep ≤ 4.
    solid : {'tetrahedron', 'cube', 'icosahedron'} or ``None``
        Base solid. When ``None`` (default), picks based on ``n_components``:
        ≤3 → tetrahedron, 4 → cube, ≥5 → icosahedron, so chord size also
        differentiates the silhouette.
    per_face_bump : bool, default=True
        If ``True``, each face's bump scale interpolates between component
        amplitudes weighted by alignment of the face normal with each
        ratio's pitch-class direction. If ``False``, all faces share a
        single global ``scale`` (the legacy behaviour).
    apex_twist : bool, default=True
        If ``True``, each bump apex is *shifted laterally* in the face's
        tangent plane by an amount proportional to ``input.phases[k]``,
        where ``k`` is the nearest-ratio index.  The lateral offset is
        scaled by edge length and clamped so the apex stays above the
        face.  This makes the bump visibly tilted (rather than just
        rotating about the normal, which would be invisible).

    Returns
    -------
    GeometryData
        ``geom_type='mesh_3d'``. ``metadata['face_ratio_index']`` is a
        per-face int array mapping each face to its nearest-ratio
        index — renderers can use it to colour the surface by chord-tone.
    """
    if depth < 0:
        raise ValueError(f"depth must be >= 0, got {depth!r}.")
    if depth > 4:
        raise ValueError(f"depth must be <= 4, got {depth!r}.")

    n = input.n_components()
    amps = input.normalized_amplitudes()

    if solid is None:
        if   n <= 3: solid = "tetrahedron"
        elif n == 4: solid = "cube"
        else:        solid = "icosahedron"

    # Pitch-class angle for each ratio (theta in [0, 2π))
    ratios = [float(r) for r in input.to_ratios()]
    pcs = np.array(
        [(math.log2(max(r, 1e-12)) % 1.0) * 2.0 * math.pi for r in ratios],
        dtype=np.float64,
    )
    phases = (
        np.asarray(input.phases, dtype=np.float64)
        if input.phases is not None
        else np.zeros(len(ratios), dtype=np.float64)
    )

    # Global scale fallback (legacy behaviour for per_face_bump=False)
    scale_global = float(np.max(amps)) / (n + 1)
    scale_global = min(max(scale_global, 0.05), 0.4)

    verts, faces = _platonic_solid(solid)
    v_arr = np.asarray(verts, dtype=np.float64)
    f_arr = np.asarray(faces, dtype=np.int64)
    # Face -> nearest-ratio index, propagated through subdivision so we can
    # colour and sculpt each face by which chord-tone "owns" it.
    face_ratio_idx = np.zeros(len(f_arr), dtype=np.int64)

    def _face_pcs(v: np.ndarray, f: np.ndarray) -> np.ndarray:
        """Return per-face pitch-class angle (azimuth of face centroid)."""
        centroids = v[f].mean(axis=1)
        # azimuth = atan2(y, x), folded to [0, 2π)
        az = np.arctan2(centroids[:, 1], centroids[:, 0])
        return np.mod(az, 2.0 * math.pi)

    def _nearest_ratio(face_pc_arr: np.ndarray) -> np.ndarray:
        """For each face's pitch-class angle, find nearest ratio index."""
        # circular distance on [0, 2π)
        d = np.abs(face_pc_arr[:, None] - pcs[None, :])
        d = np.minimum(d, 2.0 * math.pi - d)
        return np.argmin(d, axis=1).astype(np.int64)

    # Initial face-to-ratio assignment
    init_pcs = _face_pcs(v_arr, f_arr)
    face_ratio_idx = _nearest_ratio(init_pcs)

    def _subdivide(v_arr: np.ndarray, f_arr: np.ndarray,
                    face_ratio: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorised one-level Koch stellation with per-face bump + twist."""
        n_face = f_arr.shape[0]
        n_v_old = v_arr.shape[0]
        A = v_arr[f_arr[:, 0]]
        B = v_arr[f_arr[:, 1]]
        C = v_arr[f_arr[:, 2]]
        MAB = (A + B) * 0.5
        MBC = (B + C) * 0.5
        MCA = (C + A) * 0.5
        centroid = (MAB + MBC + MCA) / 3.0
        edge_len = np.linalg.norm(B - A, axis=1, keepdims=True)
        normals  = np.cross(B - A, C - A)
        n_norm   = np.linalg.norm(normals, axis=1, keepdims=True)
        normals  = np.where(n_norm > 1e-12, normals / np.maximum(n_norm, 1e-12), normals)

        # Per-face scale: amplitude of the owning ratio (with widened range
        # so chord-tone differences are visible). Uses √amp instead of /(n+1)
        # so the spread between strong and weak ratios is more dramatic.
        if per_face_bump:
            face_amps = np.asarray(amps)[face_ratio]
            face_scale = (0.10 + 0.55 * np.sqrt(face_amps)).clip(0.05, 0.65)
            face_scale = face_scale.reshape(-1, 1)
        else:
            face_scale = np.full((n_face, 1), scale_global)

        # Apex point along the face normal
        peaks = centroid + normals * (face_scale * edge_len)

        # Apex twist: shift apex laterally in the tangent plane by phase[k].
        # Rotating about the normal would be invisible (apex sits on the axis);
        # instead we offset the apex along an in-plane direction whose angle
        # is set by the phase.
        if apex_twist:
            face_phase = phases[face_ratio]
            # Build an in-plane orthonormal basis (e1, e2) per face.
            # e1 = (B - A) / ||B - A||  (one of the triangle edges)
            BA = (B - A)
            BA_norm = np.linalg.norm(BA, axis=1, keepdims=True)
            e1 = BA / np.maximum(BA_norm, 1e-12)
            # e2 = normal × e1   (in-plane, orthogonal to e1)
            e2 = np.cross(normals, e1)
            cos_p = np.cos(face_phase).reshape(-1, 1)
            sin_p = np.sin(face_phase).reshape(-1, 1)
            tangent_dir = e1 * cos_p + e2 * sin_p
            # Lateral shift: amplitude × edge_len × 0.3 (subtle but visible)
            face_amps = np.asarray(amps)[face_ratio].reshape(-1, 1)
            lateral = tangent_dir * (face_amps * edge_len * 0.35)
            peaks = peaks + lateral

        new_v = np.concatenate([MAB, MBC, MCA, peaks], axis=0)
        v_new = np.concatenate([v_arr, new_v], axis=0)

        base = n_v_old
        i_mab  = base + 0 * n_face + np.arange(n_face)
        i_mbc  = base + 1 * n_face + np.arange(n_face)
        i_mca  = base + 2 * n_face + np.arange(n_face)
        i_peak = base + 3 * n_face + np.arange(n_face)
        a_idx, b_idx, c_idx = f_arr[:, 0], f_arr[:, 1], f_arr[:, 2]

        outer = np.stack([
            np.column_stack([a_idx,  i_mab,  i_mca]),
            np.column_stack([i_mab,  b_idx,  i_mbc]),
            np.column_stack([i_mca,  i_mbc,  c_idx]),
        ], axis=1)
        bumps = np.stack([
            np.column_stack([i_mab,  i_peak, i_mbc]),
            np.column_stack([i_mbc,  i_peak, i_mca]),
            np.column_stack([i_mca,  i_peak, i_mab]),
        ], axis=1)
        f_new = np.concatenate([outer, bumps], axis=1).reshape(-1, 3).astype(np.int64)
        # Each face spawns 6 children that inherit the parent's ratio index
        face_ratio_new = np.repeat(face_ratio, 6)
        return v_new, f_new, face_ratio_new

    for _ in range(depth):
        v_arr, f_arr, face_ratio_idx = _subdivide(v_arr, f_arr, face_ratio_idx)

    return GeometryData(
        geom_type="mesh_3d",
        coordinates=v_arr,
        faces=f_arr,
        parameters={
            "depth": depth, "solid": solid,
            "bump_scale": round(scale_global, 6),
            "per_face_bump": bool(per_face_bump),
            "apex_twist":    bool(apex_twist),
        },
        metadata={
            "kind": "recursive_polyhedron",
            "n_vertices": len(v_arr),
            "n_faces": len(f_arr),
            "face_ratio_index": face_ratio_idx,
            "n_ratios": int(n),
        },
    )


def harmonic_point_cloud(
    input: HarmonicInput,
    n_points: int = 2000,
    surface: str = "sphere",
) -> GeometryData:
    """Point cloud on a sphere or torus with harmonic-phase density modulation.

    Base points are distributed via Fibonacci spiral (golden-angle method)
    for uniform coverage, then the field value at each point is computed as a
    superposition of ratio-frequency waves.  Points where the field exceeds its
    median are retained; the remainder are discarded (up to ``n_points``
    survivors).

    Parameters
    ----------
    input : HarmonicInput
    n_points : int, default=2000
        Number of points in the output cloud.
    surface : {'sphere', 'torus', 'klein', 'hyperbolic', 'mos'}, default='sphere'

        * ``'sphere'`` / ``'torus'`` — classic surfaces, see Phase 7.
        * ``'klein'`` — Klein bottle (immersion in R³ via Lawson form).
        * ``'hyperbolic'`` — Poincaré disk lifted to a saddle / pseudosphere.
        * ``'mos'`` — moment-of-symmetry helical curve at log-equave height.
    oversample : int, default=3
        Multiplier for the candidate-point pool before field-based selection.
        Higher values give finer-grained density at the cost of more compute.

    Returns
    -------
    GeometryData
        ``geom_type='point_cloud_3d'``. ``weights`` carries the field value
        at each point (useful for colouring); ``metadata['surface']`` and
        ``metadata['field_range']`` describe the underlying scalar field.
    """
    _SURFACES = {"sphere", "torus", "klein", "hyperbolic", "mos"}
    if surface not in _SURFACES:
        raise ValueError(
            f"surface must be one of {sorted(_SURFACES)}, got {surface!r}."
        )
    if n_points < 4:
        raise ValueError(f"n_points must be >= 4, got {n_points!r}.")

    ratios = [float(r) for r in input.to_ratios()]
    amps   = input.normalized_amplitudes().tolist()
    golden = (1.0 + math.sqrt(5.0)) / 2.0

    N_cand = n_points * 3
    indices = np.arange(N_cand)

    if surface == "sphere":
        theta_c = np.arccos(1.0 - 2.0 * (indices + 0.5) / N_cand)
        phi_c   = 2.0 * math.pi * indices / golden
        x_c = np.sin(theta_c) * np.cos(phi_c)
        y_c = np.sin(theta_c) * np.sin(phi_c)
        z_c = np.cos(theta_c)
        field = np.zeros(N_cand)
        for ratio, amp in zip(ratios, amps):
            frac = _ratio_to_fraction(ratio)
            l_ = min(frac.numerator, 8)
            m  = min(frac.denominator, max(l_, 1))
            field += float(amp) * np.abs(np.cos(l_ * theta_c) * np.cos(m * phi_c))

    elif surface == "torus":
        R, r = 2.0, 0.8
        u_c = 2.0 * math.pi * indices / golden
        v_c = 2.0 * math.pi * np.sqrt(2.0) * indices / N_cand
        x_c = (R + r * np.cos(v_c)) * np.cos(u_c)
        y_c = (R + r * np.cos(v_c)) * np.sin(u_c)
        z_c = r * np.sin(v_c)
        field = np.zeros(N_cand)
        for ratio, amp in zip(ratios, amps):
            frac = _ratio_to_fraction(ratio)
            m, n_ = frac.numerator, frac.denominator
            field += float(amp) * np.abs(np.cos(m * u_c + n_ * v_c))

    elif surface == "klein":
        # Lawson's immersion of the Klein bottle in R³.
        u = 2.0 * math.pi * indices / golden
        v = 2.0 * math.pi * np.sqrt(2.0) * indices / N_cand
        cu, su = np.cos(u), np.sin(u)
        cv, sv = np.cos(v / 2.0), np.sin(v / 2.0)
        x_c = (2.0 + cu * cv - sv * np.sin(v)) * np.cos(u / 2.0) * 0.5
        y_c = (2.0 + cu * cv - sv * np.sin(v)) * np.sin(u / 2.0) * 0.5
        z_c = su * cv + np.sin(v) * sv * 0.5
        field = np.zeros(N_cand)
        for ratio, amp in zip(ratios, amps):
            frac = _ratio_to_fraction(ratio)
            m, n_ = frac.numerator, frac.denominator
            field += float(amp) * np.abs(np.cos(m * u + n_ * v / 2.0))

    elif surface == "hyperbolic":
        # Poincaré-disk samples lifted onto a pseudosphere of revolution.
        # r in [0, r_max), theta uniform; height = -log(1 - r²) (saddle).
        r_disk = np.sqrt((indices + 0.5) / N_cand) * 0.93
        theta_d = 2.0 * math.pi * indices / golden
        x_disk = r_disk * np.cos(theta_d)
        y_disk = r_disk * np.sin(theta_d)
        z_lift = -np.log(np.maximum(1.0 - r_disk ** 2, 1e-3)) * 0.4
        x_c, y_c, z_c = x_disk, y_disk, z_lift
        field = np.zeros(N_cand)
        for ratio, amp in zip(ratios, amps):
            frac = _ratio_to_fraction(ratio)
            m, n_ = frac.numerator, frac.denominator
            field += float(amp) * np.abs(
                np.cos(m * theta_d) * (1.0 - r_disk ** n_)
            )

    else:  # mos — log-equave helix
        # Each ratio contributes a helical wind; height = log_equave(pitch).
        equave = float(input.equave)
        log_e = math.log(equave)
        # Distribute candidates along an equave-stack of generators
        n_layers = max(int(math.sqrt(N_cand)), 4)
        n_per = N_cand // n_layers
        layer = (indices // n_per).clip(0, n_layers - 1)
        within = indices % n_per
        # Aggregate phase: each ratio contributes 2π·log_e(ratio)·layer
        theta_c = 2.0 * math.pi * within / max(n_per, 1)
        for ratio in ratios:
            theta_c = theta_c + 2.0 * math.pi * (math.log(max(ratio, 1e-9)) / log_e) * layer / max(len(ratios), 1)
        h = layer.astype(np.float64) / max(n_layers - 1, 1) * 2.0 - 1.0
        radius = 1.0 + 0.05 * np.sin(theta_c * 3.0)
        x_c = radius * np.cos(theta_c)
        y_c = radius * np.sin(theta_c)
        z_c = h
        field = np.zeros(N_cand)
        for ratio, amp in zip(ratios, amps):
            field += float(amp) * np.abs(np.cos(theta_c * float(ratio)))

    keep = np.argsort(field)[-n_points:]
    coords = np.stack([x_c[keep], y_c[keep], z_c[keep]], axis=1)
    weights = field[keep].astype(np.float64)

    return GeometryData(
        geom_type="point_cloud_3d",
        coordinates=coords.astype(np.float64),
        weights=weights,
        parameters={"n_points": n_points, "surface": surface},
        metadata={
            "kind": "harmonic_point_cloud",
            "actual_n_points": len(coords),
            "surface": surface,
            "field_range": [float(weights.min()), float(weights.max())],
        },
    )
