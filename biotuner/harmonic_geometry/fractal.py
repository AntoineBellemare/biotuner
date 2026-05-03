"""
Fractal-flavoured harmonic geometries.

Phase 4 ships the deterministic group:

- :func:`stern_brocot_tree` — the canonical mediant tree of all positive
  rationals, annotated with biotuner harmonicity scores.
- :func:`continued_fraction_rectangles` — the "Euclid-algorithm" recursive
  square / rectangle subdivision of a ratio.
- :func:`farey_sequence_layout` — the Farey sequence of order ``n`` placed
  on a circle or line.
- :func:`subharmonic_tree` — recursive subharmonic expansion of an input
  using ``biotuner.metrics.compute_subharmonics``.
- :func:`ifs_harmonic` — chaos-game iterated-function-system attractor
  whose contractions are derived from the input ratios.

The generative group (L-systems, harmonic Julia sets, recursive polygons,
Cantor rhythms, self-similar tunings) lands in Phase 5.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from biotuner.harmonic_geometry._utils import coprime_pair, log_ratio
from biotuner.harmonic_geometry.geometry_data import GeometryData
from biotuner.harmonic_geometry.inputs import HarmonicInput

RatioLike = Union[Fraction, int, float, Tuple[int, int]]


# ============================================================ Stern-Brocot tree


def _stern_brocot_recurse(
    left: Tuple[int, int],
    right: Tuple[int, int],
    depth: int,
    max_depth: int,
    nodes: List[Tuple[int, int]],
    edges: List[Tuple[int, int]],
    parent_idx: int,
) -> None:
    """Append mediants of (left, right) to nodes; record edge to parent.

    The bounds are represented as ``(num, den)`` tuples since the Stern-Brocot
    tree canonically uses ``0/1`` and ``1/0``; ``Fraction`` rejects ``1/0``.
    """
    if depth >= max_depth:
        return
    ln, ld = left
    rn, rd = right
    mediant = (ln + rn, ld + rd)
    idx = len(nodes)
    nodes.append(mediant)
    if parent_idx >= 0:
        edges.append((parent_idx, idx))
    _stern_brocot_recurse(left, mediant, depth + 1, max_depth, nodes, edges, idx)
    _stern_brocot_recurse(mediant, right, depth + 1, max_depth, nodes, edges, idx)


def stern_brocot_tree(
    input: Optional[HarmonicInput] = None,
    max_depth: int = 6,
    layout: str = "hyperbolic",
) -> GeometryData:
    """Stern-Brocot mediant tree to ``max_depth`` levels.

    Starts from the canonical bounds ``0/1`` and ``1/0``; each node is the
    mediant of its bracketing pair. The tree at depth ``d`` has exactly
    ``2^d - 1`` interior nodes (the bounds are excluded from the output).

    Each node is annotated with a harmonicity score in
    ``metadata['harmonicity']`` — by default ``dyad_similarity(p/q)``.
    If ``input`` is provided, an additional
    ``metadata['nearest_input_dist_cents']`` array records the cents
    distance from each tree node to the closest ratio in ``input``,
    helpful for highlighting where the chord lives in the rational lattice.

    Parameters
    ----------
    input : HarmonicInput, optional
        If given, used only for the nearest-input-distance annotation.
    max_depth : int, default=6
        Tree depth. The number of nodes grows as ``2^depth - 1``.
    layout : {'hyperbolic', 'tree'}, default='hyperbolic'
        ``'hyperbolic'`` places nodes on the Poincaré disk by traversal
        position and depth; ``'tree'`` uses a flat dendrogram layout.

    Returns
    -------
    GeometryData
        ``geom_type='tree'``.
    """
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth!r}.")
    if layout not in {"hyperbolic", "tree"}:
        raise ValueError(f"layout must be 'hyperbolic' or 'tree', got {layout!r}.")

    nodes: List[Tuple[int, int]] = []
    edges: List[Tuple[int, int]] = []
    _stern_brocot_recurse(
        (0, 1), (1, 0), 0, max_depth, nodes, edges, parent_idx=-1
    )
    fraction_nodes: List[Fraction] = [Fraction(p, q) for p, q in nodes]

    n = len(fraction_nodes)
    # Determine each node's depth via BFS from root (index 0).
    depths = np.zeros(n, dtype=np.int32)
    children: List[List[int]] = [[] for _ in range(n)]
    for u, v in edges:
        children[u].append(v)
        depths[v] = depths[u] + 1

    # In-order traversal to assign x-positions.
    x_pos = np.zeros(n, dtype=np.float64)
    counter = [0]

    def in_order(idx: int) -> None:
        if children[idx]:
            in_order(children[idx][0])
            x_pos[idx] = float(counter[0])
            counter[0] += 1
            if len(children[idx]) > 1:
                in_order(children[idx][1])
        else:
            x_pos[idx] = float(counter[0])
            counter[0] += 1

    in_order(0)
    # Normalize x to [-1, 1].
    if n > 1:
        x_pos = 2.0 * (x_pos - x_pos.min()) / (x_pos.max() - x_pos.min()) - 1.0

    if layout == "tree":
        coords = np.stack([x_pos, -depths.astype(np.float64) / max(1, max_depth)], axis=1)
    else:  # hyperbolic
        # Place by (angle, radius); radius grows with depth, capped < 1.
        angles = np.pi * x_pos  # in [-π, π]
        radii = depths.astype(np.float64) / max(1, max_depth)
        radii = 0.95 * radii  # keep inside disk
        coords = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)

    # Lazy import biotuner.metrics so test environments without scipy.special.* still load.
    from biotuner.metrics import dyad_similarity

    harmonicity = np.array(
        [float(dyad_similarity(float(r))) for r in fraction_nodes], dtype=np.float64
    )

    metadata: dict = {
        "kind": "stern_brocot_tree",
        "max_depth": int(max_depth),
        "layout": layout,
        "depth_per_node": depths.tolist(),
        "ratios": [str(r) for r in fraction_nodes],
        "harmonicity": harmonicity.tolist(),
    }
    if input is not None:
        equave = input.equave
        in_pcs = np.array(
            [log_ratio(r, equave=equave) % 1.0 for r in input.to_ratios()],
            dtype=np.float64,
        )
        node_pcs = np.array(
            [log_ratio(float(r), equave=equave) % 1.0 for r in fraction_nodes],
            dtype=np.float64,
        )
        # Cents distance to nearest input pitch class on the equave circle.
        equave_cents = 1200.0 * math.log2(equave)
        diffs = np.abs(node_pcs[:, None] - in_pcs[None, :])
        wrapped = np.minimum(diffs, 1.0 - diffs) * equave_cents
        metadata["nearest_input_dist_cents"] = wrapped.min(axis=1).tolist()

    edges_arr = (
        np.asarray(edges, dtype=np.int64)
        if edges
        else np.empty((0, 2), dtype=np.int64)
    )
    return GeometryData(
        geom_type="tree",
        coordinates=coords,
        edges=edges_arr,
        weights=harmonicity,
        parameters={"max_depth": int(max_depth), "layout": layout},
        metadata=metadata,
    )


# ===================================================== continued-fraction tiling


def continued_fraction_rectangles(ratio: RatioLike, depth: int = 10) -> GeometryData:
    """Recursive Euclid-algorithm square / rectangle decomposition of a ratio.

    Visualizes the continued-fraction expansion of ``p/q`` (assumed
    ``> 1``; smaller values are inverted internally and the output is
    flagged in metadata). The starting rectangle is ``p × q``; the
    largest possible squares of side ``min(p, q)`` are stripped off
    repeatedly, each time rotating the residual strip by 90°. The
    sequence of squares is the continued-fraction expansion of ``p/q``.

    Parameters
    ----------
    ratio : Fraction, int, float, or (int, int)
    depth : int, default=10
        Maximum number of squares to record. The full expansion
        terminates earlier if ``p/q`` is rational.

    Returns
    -------
    GeometryData
        ``geom_type='polygon_set'`` — one rectangular polygon per square,
        in original-rectangle units (the bounding rectangle is the unit-area
        rectangle ``[0, 1] × [0, q/p]``).
    """
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth!r}.")

    p, q = coprime_pair(ratio)
    inverted = False
    if p < q:
        p, q = q, p
        inverted = True
    if p == 0 or q == 0:
        raise ValueError("Ratio must be a positive non-zero rational.")

    # Work in continuous floats but track exact integer remainders for closure.
    # Place the bounding rectangle as [0, p_norm] x [0, q_norm] where p_norm = 1.
    # We'll record squares in original units (p, q) and then normalize at the end.
    squares: List[Tuple[float, float, float, float]] = []  # (x, y, side, side)
    # Iterative Euclid with geometric strip tracking.
    x0, y0 = 0.0, 0.0
    width, height = float(p), float(q)
    horizontal = True  # True: strip is horizontal, squares stacked to the right
    pi, qi = int(p), int(q)
    iterations = 0
    while qi > 0 and iterations < depth:
        a = pi // qi
        for _ in range(a):
            side = min(width, height) if horizontal else min(width, height)
            if horizontal:
                squares.append((x0, y0, side, side))
                x0 += side
                width -= side
            else:
                squares.append((x0, y0, side, side))
                y0 += side
                height -= side
            if min(width, height) <= 1e-15:
                break
        # Rotate strip 90°.
        horizontal = not horizontal
        pi, qi = qi, pi - a * qi
        iterations += 1

    # Normalize so the bounding rectangle has width 1 (for inverted ratios this
    # keeps the output centered around the unit interval on the long axis).
    polygons: List[np.ndarray] = []
    for x, y, w, h in squares:
        x_n = x / float(p)
        y_n = y / float(p)
        w_n = w / float(p)
        h_n = h / float(p)
        rect = np.array(
            [
                [x_n, y_n],
                [x_n + w_n, y_n],
                [x_n + w_n, y_n + h_n],
                [x_n, y_n + h_n],
            ],
            dtype=np.float64,
        )
        polygons.append(rect)

    return GeometryData(
        geom_type="polygon_set",
        coordinates=polygons,
        parameters={
            "ratio": Fraction(p, q),
            "depth": int(depth),
        },
        metadata={
            "kind": "continued_fraction_rectangles",
            "inverted": inverted,
            "n_squares": len(polygons),
            "iterations": iterations,
        },
    )


# ===================================================== Farey sequence layout


def _farey_sequence(order: int) -> List[Fraction]:
    """Farey sequence F_n: sorted reduced fractions p/q with 0 <= q <= n in [0, 1]."""
    a, b, c, d = 0, 1, 1, order
    out = [Fraction(a, b)]
    while c <= order:
        out.append(Fraction(c, d))
        k = (order + b) // d
        a, b, c, d = c, d, k * c - a, k * d - b
    return out


def farey_sequence_layout(order: int, layout: str = "circle") -> GeometryData:
    """Farey sequence ``F_n`` placed on a circle, line, or as Ford circles.

    Parameters
    ----------
    order : int
        Sequence order, ``>= 1``.
    layout : {'circle', 'line', 'ford'}, default='circle'

        * ``'circle'`` / ``'line'`` — points only; weight encodes
          ``1 / denominator``.
        * ``'ford'`` — each fraction ``p/q ∈ F_n`` becomes a circle of
          radius ``1 / (2 q²)`` tangent to the x-axis at ``x = p/q``.
          Adjacent Farey fractions correspond to tangent Ford circles —
          the classic visual of the Farey structure.

    Returns
    -------
    GeometryData
        For ``'circle'`` / ``'line'``: ``geom_type='point_cloud_2d'``.
        For ``'ford'``: ``geom_type='polygon_set'`` — each entry is a
        polyline approximation of one Ford circle.  ``metadata['radii']``
        and ``metadata['centers']`` carry the analytic geometry.
    """
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order!r}.")
    if layout not in {"circle", "line", "ford"}:
        raise ValueError(
            f"layout must be 'circle', 'line', or 'ford', got {layout!r}."
        )

    seq = _farey_sequence(order)
    values = np.array([float(f) for f in seq], dtype=np.float64)
    denoms = np.array([f.denominator for f in seq], dtype=np.float64)

    if layout == "ford":
        # Each fraction p/q ∈ F_n ⊂ [0, 1] becomes a circle:
        #   centre = (p/q, 1/(2 q²))  radius = 1/(2 q²)
        # tangent to the x-axis (y = 0). Adjacent F_n fractions yield
        # tangent circles, which is the classic Farey/Ford property.
        centres = np.stack([values, 1.0 / (2.0 * denoms ** 2)], axis=1)
        radii   = 1.0 / (2.0 * denoms ** 2)
        # Polyline approximation per circle (32 vertices each)
        n_v = 32
        thetas = np.linspace(0.0, 2.0 * np.pi, n_v, endpoint=False)
        cos_t = np.cos(thetas); sin_t = np.sin(thetas)
        polys: List[np.ndarray] = []
        for (cx, cy), r in zip(centres, radii):
            xs = cx + r * cos_t
            ys = cy + r * sin_t
            polys.append(np.stack([xs, ys], axis=1))
        coords_obj = np.empty(len(polys), dtype=object)
        for i, p in enumerate(polys):
            coords_obj[i] = p
        return GeometryData(
            geom_type="polygon_set",
            coordinates=coords_obj,
            weights=1.0 / denoms,
            parameters={"order": int(order), "layout": layout},
            metadata={
                "kind": "farey_sequence_layout",
                "n_terms": int(len(seq)),
                "fractions": [str(f) for f in seq],
                "centres": centres.tolist(),
                "radii":   radii.tolist(),
            },
        )

    if layout == "line":
        coords = np.stack([values * 2.0 - 1.0, np.zeros_like(values)], axis=1)
    else:  # circle
        angles = 2.0 * np.pi * values
        coords = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    return GeometryData(
        geom_type="point_cloud_2d",
        coordinates=coords,
        weights=1.0 / denoms,  # higher weight for simpler fractions
        parameters={"order": int(order), "layout": layout},
        metadata={
            "kind": "farey_sequence_layout",
            "n_terms": int(len(seq)),
            "fractions": [str(f) for f in seq],
        },
    )


# =========================================================== subharmonic tree


def subharmonic_tree(
    input: HarmonicInput,
    depth: int = 4,
    n_harmonics: int = 5,
    min_freq: float = 0.1,
    layout: str = "depth",
) -> GeometryData:
    """Recursive subharmonic expansion as a tree.

    Each input peak ``f`` is the root of a sub-tree whose children are
    its first ``n_harmonics`` subharmonics ``f / 2, f / 3, ..., f / (k + 1)``.
    Each child is expanded the same way to ``depth`` levels. Nodes with
    frequency below ``min_freq`` are pruned.

    Notes
    -----
    The plan originally suggested using
    ``biotuner.metrics.compute_subharmonics``, which finds *common*
    subharmonics across a chord. That's a different operation from the
    per-peak subharmonic series this tree visualizes; we use the
    classical ``f / k`` definition directly.

    Parameters
    ----------
    input : HarmonicInput
        Source peaks for the root level.
    depth : int, default=4
        Number of expansion levels below the root.
    n_harmonics : int, default=5
        Number of subharmonics per node.
    min_freq : float, default=0.1
        Frequencies below this are not expanded further.
    layout : {'depth', 'polar'}, default='depth'
        ``'depth'`` — original dendrogram layout (depth on Y, sorted on X).
        ``'polar'`` — each input peak gets its own angular sector; depth
        becomes radial distance, so different chords produce visibly
        different fan-out shapes instead of identical depth-stacks.

    Returns
    -------
    GeometryData
        ``geom_type='tree'``. ``metadata['root_index_per_node']`` tags every
        node with its originating root-peak index, useful for colour-coding.
    """
    if depth < 1:
        raise ValueError(f"depth must be >= 1, got {depth!r}.")
    if n_harmonics < 1:
        raise ValueError(f"n_harmonics must be >= 1, got {n_harmonics!r}.")
    if min_freq <= 0:
        raise ValueError(f"min_freq must be > 0, got {min_freq!r}.")
    if layout not in {"depth", "polar"}:
        raise ValueError(f"layout must be 'depth' or 'polar', got {layout!r}.")

    peaks = input.to_peaks().tolist()
    nodes: List[float] = list(peaks)
    parent_of: List[int] = [-1] * len(peaks)
    depth_of: List[int] = [0] * len(peaks)
    root_of: List[int] = list(range(len(peaks)))   # which root-peak owns each node

    frontier = list(range(len(peaks)))
    for d in range(1, depth + 1):
        next_frontier: List[int] = []
        for parent_idx in frontier:
            f = nodes[parent_idx]
            for k in range(2, n_harmonics + 2):
                sf = f / float(k)
                if sf < min_freq or not math.isfinite(sf):
                    continue
                idx = len(nodes)
                nodes.append(float(sf))
                parent_of.append(parent_idx)
                depth_of.append(d)
                root_of.append(root_of[parent_idx])
                next_frontier.append(idx)
        frontier = next_frontier
        if not frontier:
            break

    n = len(nodes)
    if n == 0:
        raise ValueError("subharmonic_tree produced no nodes.")
    edges = np.asarray(
        [(parent_of[i], i) for i in range(n) if parent_of[i] >= 0],
        dtype=np.int64,
    )
    if edges.size == 0:
        edges = np.empty((0, 2), dtype=np.int64)

    depths_arr = np.asarray(depth_of, dtype=np.float64)
    coords = np.zeros((n, 2), dtype=np.float64)

    if layout == "depth":
        # Original layout: depth on y; spread on x by log-frequency rank.
        for d in range(int(depths_arr.max()) + 1):
            mask = depths_arr == d
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                continue
            freqs = np.asarray([nodes[i] for i in idxs], dtype=np.float64)
            order = np.argsort(freqs)
            sorted_idxs = idxs[order]
            if len(sorted_idxs) == 1:
                xs = np.array([0.0])
            else:
                xs = np.linspace(-1.0, 1.0, len(sorted_idxs))
            for x, i in zip(xs, sorted_idxs):
                coords[i, 0] = float(x)
                coords[i, 1] = -float(d) / max(1.0, depths_arr.max())
    else:
        # Polar: one angular sector per root-peak; depth → radius.
        n_roots = len(peaks)
        sector = 2.0 * math.pi / max(n_roots, 1)
        # For each root, count children at each depth and lay them out
        # within that root's sector at angular positions proportional to
        # rank within depth.
        max_d = max(int(depths_arr.max()), 1)
        for root_idx in range(n_roots):
            root_mask = np.asarray(root_of) == root_idx
            for d in range(max_d + 1):
                mask = root_mask & (depths_arr == d)
                idxs = np.where(mask)[0]
                if len(idxs) == 0:
                    continue
                freqs = np.asarray([nodes[i] for i in idxs], dtype=np.float64)
                order = np.argsort(freqs)
                sorted_idxs = idxs[order]
                # Angular position: root centre ± half-sector, evenly spaced
                centre_angle = root_idx * sector
                spread = sector * 0.45   # leave a small gap between sectors
                if len(sorted_idxs) == 1:
                    angles = np.array([centre_angle])
                else:
                    angles = centre_angle + np.linspace(
                        -spread, spread, len(sorted_idxs)
                    )
                radius = (d + 1) / float(max_d + 1)
                for ang, i in zip(angles, sorted_idxs):
                    coords[i, 0] = float(radius * np.cos(ang))
                    coords[i, 1] = float(radius * np.sin(ang))

    return GeometryData(
        geom_type="tree",
        coordinates=coords,
        edges=edges,
        weights=np.asarray(nodes, dtype=np.float64),
        parameters={
            "depth": int(depth),
            "n_harmonics": int(n_harmonics),
            "min_freq": float(min_freq),
            "layout": layout,
        },
        metadata={
            "kind": "subharmonic_tree",
            "n_nodes": int(n),
            "frequencies_hz": list(nodes),
            "depth_per_node": [int(x) for x in depth_of],
            "root_index_per_node": [int(x) for x in root_of],
        },
    )


# =================================================================== IFS chaos


def ifs_harmonic(
    input: HarmonicInput,
    n_points: int = 50_000,
    contraction: str = "ratio_inverse",
    transient: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> GeometryData:
    """Iterated-function-system attractor driven by harmonic ratios.

    Each input ratio defines an affine contraction
    ``z -> z · s_i + v_i``, where ``s_i`` is the contraction factor
    (derived from the ratio per ``contraction``) and ``v_i`` is the i-th
    vertex of an N-gon scaled to the unit disk. The classic chaos
    game then samples the attractor.

    Parameters
    ----------
    input : HarmonicInput
        Provides the N ratios.
    n_points : int, default=50_000
    contraction : {'ratio_inverse', 'log_ratio', 'fixed_half'}, default='ratio_inverse'
        - ``'ratio_inverse'``: ``s_i = 1 / r_i`` (rebound to ``< 1``).
        - ``'log_ratio'``: ``s_i = 1 / (1 + log(r_i))``.
        - ``'fixed_half'``: ``s_i = 0.5`` for all i (Sierpinski-like).
    transient : int, default=200
        Number of warm-up iterations to discard before recording points.
    rng : np.random.Generator, optional
        Source of randomness. Default: ``np.random.default_rng()``.

    Returns
    -------
    GeometryData
        ``geom_type='point_cloud_2d'`` with ``(n_points, 2)`` coordinates.
    """
    if n_points < 100:
        raise ValueError(f"n_points must be >= 100, got {n_points!r}.")
    if transient < 0:
        raise ValueError(f"transient must be >= 0, got {transient!r}.")
    if rng is None:
        rng = np.random.default_rng()

    ratios = [float(r) for r in input.to_ratios()]
    n = len(ratios)
    if n < 2:
        raise ValueError("ifs_harmonic needs at least 2 components.")

    # Vertex positions: regular N-gon on the unit circle.
    angles = 2.0 * np.pi * np.arange(n) / n
    vertices = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    # Contraction factors per map.
    if contraction == "ratio_inverse":
        scales = np.array(
            [1.0 / r if r > 1.0 else float(r) for r in ratios],
            dtype=np.float64,
        )
        # Floor in (0, 1) to keep the IFS strictly contractive.
        scales = np.clip(scales, 0.05, 0.95)
    elif contraction == "log_ratio":
        scales = np.array(
            [1.0 / (1.0 + abs(math.log(max(float(r), 1e-9)))) for r in ratios],
            dtype=np.float64,
        )
        scales = np.clip(scales, 0.05, 0.95)
    elif contraction == "fixed_half":
        scales = np.full(n, 0.5)
    else:
        raise ValueError(
            f"Unknown contraction {contraction!r}. Use 'ratio_inverse', "
            "'log_ratio', or 'fixed_half'."
        )

    # Probability per map: weight by amplitude so louder components recruit
    # more points in the attractor.
    probs = input.normalized_amplitudes()
    if probs.sum() <= 0:
        probs = np.full(n, 1.0 / n)

    z = np.zeros(2, dtype=np.float64)
    for _ in range(transient):
        i = rng.choice(n, p=probs)
        z = z * scales[i] + (1.0 - scales[i]) * vertices[i]

    out = np.empty((n_points, 2), dtype=np.float64)
    for k in range(n_points):
        i = rng.choice(n, p=probs)
        z = z * scales[i] + (1.0 - scales[i]) * vertices[i]
        out[k] = z

    # Approximate bounding extent for downstream renderers.
    span = float(np.max(np.abs(out)))
    return GeometryData(
        geom_type="point_cloud_2d",
        coordinates=out,
        weights=np.ones(n_points, dtype=np.float64),
        parameters={
            "n_points": int(n_points),
            "contraction": contraction,
            "transient": int(transient),
        },
        metadata={
            "kind": "ifs_harmonic",
            "n_maps": int(n),
            "scales": scales.tolist(),
            "probabilities": probs.tolist(),
            "span": span,
        },
    )
