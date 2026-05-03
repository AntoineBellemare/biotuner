"""
biotuner.harmonic_geometry.generative
======================================
Phase 5: generative fractal structures derived from harmonic inputs.

Functions
---------
lsystem_from_ratios   -- L-system branching plant driven by ratio-derived angles
recursive_polygon     -- Koch-like self-similar polygon boundary
self_similar_tuning   -- pitch lattice at multiple equave levels (harmonic spiral)
geometry_sequence     -- map a geometry function over every frame of a HarmonicSequence
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from biotuner.harmonic_geometry._utils import coerce_ratio
from biotuner.harmonic_geometry.geometry_data import GeometryData
from biotuner.harmonic_geometry.inputs import HarmonicInput, HarmonicSequence

# ════════════════════════════════════════════ internal helpers ════════════════


def _ratio_to_fraction(r) -> Fraction:
    raw = coerce_ratio(r)
    if isinstance(raw, Fraction):
        return raw.limit_denominator(20)
    return Fraction(float(raw)).limit_denominator(20)


def _ratio_to_angle(r) -> float:
    """360 / (p + q) for ratio p/q — maps small integer ratios to nice angles."""
    f = _ratio_to_fraction(r)
    return 360.0 / max(f.numerator + f.denominator, 3)


# ── L-system helpers ──────────────────────────────────────────────────────────

# Each character maps to a signed multiple of the base angle θ.
_ANGLE_MAP: Dict[str, int] = {
    "+": 1,  "-": -1,
    "{": 2,  "}": -2,
    "<": 3,  ">": -3,
    "~": 4,  "|": -4,
}
_L_CHARS = ["+", "{", "<", "~"]
_R_CHARS = ["-", "}", ">", "|"]


def _default_lsystem_rule(n_extra: int) -> Dict[str, str]:
    """Branching plant rule for `n_extra` side-branches."""
    if n_extra <= 0:
        return {"F": "F+F--F+F"}          # Koch-like for degenerate single-ratio
    branches = "".join(f"[{_L_CHARS[k % 4]}F]" for k in range(n_extra))
    return {"F": f"F{branches}"}


def _apply_lsystem(axiom: str, rules: Dict[str, str], depth: int) -> str:
    s = axiom
    for _ in range(depth):
        s = "".join(rules.get(c, c) for c in s)
    return s


def _turtle_segments(
    lstring: str,
    step: float,
    theta_deg: float,
    start_heading: float = 90.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Execute turtle-graphics; return (N×2 coords, M×2 edge indices)."""
    x, y, heading = 0.0, 0.0, start_heading
    stack: List[Tuple[float, float, float]] = []
    pts: List[List[float]] = []
    eds: List[List[int]] = []

    for c in lstring:
        if c == "F":
            rad = math.radians(heading)
            nx = x + step * math.cos(rad)
            ny = y + step * math.sin(rad)
            i = len(pts)
            pts.append([x, y])
            pts.append([nx, ny])
            eds.append([i, i + 1])
            x, y = nx, ny
        elif c in _ANGLE_MAP:
            heading += _ANGLE_MAP[c] * theta_deg
        elif c == "[":
            stack.append((x, y, heading))
        elif c == "]":
            x, y, heading = stack.pop()

    if not pts:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.int64)
    return (
        np.array(pts, dtype=np.float64),
        np.array(eds, dtype=np.int64),
    )


# ── recursive polygon helpers ─────────────────────────────────────────────────


def _rotate_2d(v: np.ndarray, angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _subdivide_edges(
    verts: np.ndarray,
    scale: float,
    bump_angle: float,
) -> np.ndarray:
    """One Koch-like subdivision step.  Each edge A→B becomes [A, C1, bump, C2]."""
    n = len(verts)
    new_verts: List[np.ndarray] = []
    for i in range(n):
        A = verts[i]
        B = verts[(i + 1) % n]
        C1 = A + scale * (B - A)
        C2 = A + (1.0 - scale) * (B - A)
        bump = C1 + _rotate_2d(C2 - C1, bump_angle)
        new_verts.extend([A, C1, bump, C2])
    return np.array(new_verts, dtype=np.float64)


# ── pitch-set helpers for self_similar_tuning ─────────────────────────────────


def _reduce_to_equave(pitch: float, equave: float) -> float:
    """Fold pitch into [1, equave)."""
    while pitch >= equave:
        pitch /= equave
    while pitch < 1.0:
        pitch *= equave
    return pitch


def _deduplicate_pitches(
    pitches: List[Tuple[float, float]],
    cents_tol: float = 5.0,
    equave: float = 2.0,
) -> List[Tuple[float, float]]:
    """Merge pitches within `cents_tol` of each other, summing amplitudes."""
    if not pitches:
        return []
    sorted_p = sorted(pitches, key=lambda x: x[0])
    merged: List[List[float]] = [list(sorted_p[0])]
    log_eq = math.log2(equave)
    for pitch, amp in sorted_p[1:]:
        last = merged[-1][0]
        cents = abs(1200.0 * math.log2(pitch / last) / log_eq)
        if cents < cents_tol:
            merged[-1][1] += amp
        else:
            merged.append([pitch, amp])
    return [(p, a) for p, a in merged]


# ════════════════════════════════════════════ public API ═════════════════════


def lsystem_from_ratios(
    input: HarmonicInput,
    depth: int = 4,
    axiom: str = "F",
    rules: Optional[Dict[str, str]] = None,
    step_size: float = 1.0,
) -> GeometryData:
    """L-system branching plant parameterised by harmonic ratios.

    The base turning angle θ is derived from the first non-unison ratio as
    ``360 / (p + q)`` for ratio ``p/q``.  The number of side-branches equals
    ``n_components - 1``, mapping each interval of the chord to a branch
    direction.  Override both with explicit ``rules``.

    Parameters
    ----------
    input : HarmonicInput
    depth : int, default=4
        Rewriting depth.  Keep ≤ 6 to avoid very large strings.
    axiom : str, default='F'
        Starting L-system string.
    rules : dict, optional
        Symbol → replacement string.  Defaults to a ratio-derived plant rule.
    step_size : float, default=1.0
        Forward step length per ``F`` symbol.

    Returns
    -------
    GeometryData
        ``geom_type='graph'`` with all turtle segments as edges.
        ``metadata['lstring_preview']`` holds the first 120 characters of the
        rewritten string.
    """
    if depth < 1 or depth > 7:
        raise ValueError(f"depth must be in [1, 7], got {depth!r}.")
    if not axiom:
        raise ValueError("axiom must be non-empty.")

    ratios = input.to_ratios()
    non_trivial = [r for r in ratios if abs(float(r) - 1.0) > 1e-9]
    theta = _ratio_to_angle(non_trivial[0]) if non_trivial else 60.0

    if rules is None:
        rules = _default_lsystem_rule(input.n_components() - 1)

    lstring = _apply_lsystem(axiom, rules, depth)
    coords, edges = _turtle_segments(lstring, step=step_size, theta_deg=theta)

    return GeometryData(
        geom_type="graph",
        coordinates=coords,
        edges=edges,
        parameters={
            "axiom": axiom,
            "rules": rules,
            "depth": depth,
            "angle_deg": theta,
            "step_size": step_size,
        },
        metadata={
            "kind": "lsystem_from_ratios",
            "lstring_length": len(lstring),
            "lstring_preview": lstring[:120],
            "n_segments": int(len(edges)),
            "angle_deg": theta,
        },
    )


def recursive_polygon(
    input: HarmonicInput,
    depth: int = 4,
    n_sides: Optional[int] = None,
    scale_factor: Optional[float] = None,
) -> GeometryData:
    """Koch-like self-similar polygon boundary driven by harmonic ratios.

    Each edge is recursively replaced by four sub-edges forming an outward
    triangular bump.  The bump scale and rotation angle are derived from the
    first non-unison ratio.

    Parameters
    ----------
    input : HarmonicInput
    depth : int, default=4
        Subdivision steps.  Edge count grows as ``4^depth * n_sides``.
    n_sides : int, optional
        Number of polygon sides.  Defaults to ``n_components`` (min 3).
    scale_factor : float, optional
        Fraction of the edge length occupied by each outer sub-edge.
        Defaults to ``1 / (p + 1)`` for the first non-trivial ratio ``p/q``.

    Returns
    -------
    GeometryData
        ``geom_type='curve_2d'`` with the closed fractal boundary.
    """
    if depth < 1 or depth > 6:
        raise ValueError(f"depth must be in [1, 6], got {depth!r}.")

    ratios = input.to_ratios()
    non_trivial = [r for r in ratios if abs(float(r) - 1.0) > 1e-9]

    if non_trivial:
        f = _ratio_to_fraction(non_trivial[0])
        p, q = f.numerator, f.denominator
    else:
        p, q = 1, 3

    sides = max(3, input.n_components() if n_sides is None else n_sides)
    if n_sides is not None and n_sides < 3:
        raise ValueError(f"n_sides must be >= 3, got {n_sides!r}.")

    scale = scale_factor if scale_factor is not None else 1.0 / (p + 1)
    if not (0.0 < scale < 0.5):
        raise ValueError(f"scale_factor must be in (0, 0.5), got {scale!r}.")

    # bump angle: π/q gives an isoceles bump; for q=1 cap at π/3 (equilateral)
    bump_angle = math.pi / max(q, 3)

    # Initial regular polygon vertices.
    angles = [2.0 * math.pi * k / sides for k in range(sides)]
    verts = np.array([[math.cos(a), math.sin(a)] for a in angles], dtype=np.float64)

    for _ in range(depth):
        verts = _subdivide_edges(verts, scale, bump_angle)

    # Close the boundary.
    closed = np.vstack([verts, verts[:1]])

    return GeometryData(
        geom_type="curve_2d",
        coordinates=closed,
        parameters={
            "n_sides": sides,
            "scale_factor": scale,
            "bump_angle_rad": bump_angle,
            "depth": depth,
        },
        metadata={
            "kind": "recursive_polygon",
            "n_vertices": len(closed),
            "source_ratio": f"{p}/{q}",
        },
    )


def self_similar_tuning(
    input: HarmonicInput,
    n_levels: int = 4,
    equave: float = 2.0,
) -> GeometryData:
    """Self-similar pitch lattice at multiple equave levels.

    Starting from the input ratios as generators, each subsequent level is
    formed by multiplying every pitch in the previous level by every generator
    and reducing back to ``[1, equave)``.  The result is the free abelian group
    generated by the ratios, truncated to ``n_levels`` generations.

    Pitches are arranged on concentric circles — the k-th circle has radius
    ``(k + 1) / n_levels`` — at angular positions proportional to
    ``log_equave(pitch)``.  Edges connect each pitch to its parent (the
    closest pitch at the previous level that generated it).

    Parameters
    ----------
    input : HarmonicInput
    n_levels : int, default=4
        Number of generative levels (generations).  Level 0 is the seed.
    equave : float, default=2.0
        Interval of equivalence (2.0 = octave).

    Returns
    -------
    GeometryData
        ``geom_type='graph'`` with nodes on concentric circles and edges
        tracing the generative lineage.
    """
    if n_levels < 1:
        raise ValueError(f"n_levels must be >= 1, got {n_levels!r}.")
    if equave <= 1.0:
        raise ValueError(f"equave must be > 1, got {equave!r}.")

    ratios = input.to_ratios()
    amps = input.normalized_amplitudes().tolist()
    log_eq = math.log2(float(equave))

    # Build pitch sets level by level.
    PitchList = List[Tuple[float, float]]  # (pitch, amplitude)
    levels: List[PitchList] = []
    seed: PitchList = [
        (_reduce_to_equave(float(r), float(equave)), float(a))
        for r, a in zip(ratios, amps)
    ]
    levels.append(_deduplicate_pitches(seed, equave=float(equave)))

    for lv in range(1, n_levels):
        prev = levels[lv - 1]
        candidates: PitchList = []
        for p_pitch, p_amp in prev:
            for r, r_amp in zip(ratios, amps):
                new_p = _reduce_to_equave(p_pitch * float(r), float(equave))
                candidates.append((new_p, p_amp * r_amp))
        levels.append(_deduplicate_pitches(candidates, equave=float(equave)))

    # Build node positions and edge connectivity.
    all_coords: List[List[float]] = []
    all_weights: List[float] = []
    all_edges: List[List[int]] = []
    level_offsets = [0]
    pitch_log_per_level: List[List[float]] = []

    for lv, pitches in enumerate(levels):
        radius = (lv + 1.0) / n_levels
        lp: List[float] = []
        for pitch, amp in pitches:
            angle = 2.0 * math.pi * math.log2(pitch) / log_eq
            all_coords.append([radius * math.cos(angle), radius * math.sin(angle)])
            all_weights.append(amp)
            lp.append(math.log2(pitch))
        level_offsets.append(level_offsets[-1] + len(pitches))
        pitch_log_per_level.append(lp)

    # Connect each node to its nearest ancestor at the previous level.
    for lv in range(1, n_levels):
        parent_logs = pitch_log_per_level[lv - 1]
        parent_offset = level_offsets[lv - 1]
        child_offset = level_offsets[lv]
        for ci, child_log in enumerate(pitch_log_per_level[lv]):
            best_j = int(np.argmin([abs(child_log - pl) for pl in parent_logs]))
            all_edges.append([parent_offset + best_j, child_offset + ci])

    coords = np.array(all_coords, dtype=np.float64) if all_coords else np.empty((0, 2))
    edges = (
        np.array(all_edges, dtype=np.int64)
        if all_edges
        else np.empty((0, 2), dtype=np.int64)
    )
    weights = np.array(all_weights, dtype=np.float64)

    return GeometryData(
        geom_type="graph",
        coordinates=coords,
        edges=edges,
        weights=weights,
        parameters={
            "n_levels": n_levels,
            "equave": float(equave),
        },
        metadata={
            "kind": "self_similar_tuning",
            "n_nodes_per_level": [len(lv) for lv in levels],
            "total_nodes": int(len(all_coords)),
            "total_edges": int(len(all_edges)),
            "pitches_level_0": [round(p, 6) for p, _ in levels[0]],
        },
    )


def geometry_sequence(
    input_seq: Union[HarmonicSequence, Iterable[HarmonicInput]],
    fn: Callable[..., GeometryData],
    **kwargs,
) -> List[GeometryData]:
    """Map a geometry function over every frame of a :class:`HarmonicSequence`.

    Parameters
    ----------
    input_seq : HarmonicSequence or Iterable[HarmonicInput]
    fn : callable taking a HarmonicInput as first argument
    **kwargs : forwarded to ``fn`` for every frame

    Returns
    -------
    list of GeometryData

    Raises
    ------
    ValueError
        If ``input_seq`` contains no frames.
    """
    if isinstance(input_seq, HarmonicSequence):
        frames: List[HarmonicInput] = input_seq.frames
    else:
        frames = list(input_seq)
    if not frames:
        raise ValueError("geometry_sequence requires at least one frame.")
    return [fn(frame, **kwargs) for frame in frames]
