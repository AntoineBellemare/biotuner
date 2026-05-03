"""
Polygon and circular geometric structures.

Phase 2 implements the basic ratio-driven shapes. Phase 4 adds the
biotuner-metric-driven variants: ``interval_vector_diagram``,
``polygon_chord_pattern``, and ``consonance_polygon``.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from biotuner.harmonic_geometry._utils import coprime_pair, log_ratio
from biotuner.harmonic_geometry.geometry_data import GeometryData
from biotuner.harmonic_geometry.inputs import HarmonicInput

RatioLike = Union[Fraction, int, float, Tuple[int, int]]


# ----------------------------------------------------------------- star polygon


def star_polygon(n: int, k: int, radius: float = 1.0) -> GeometryData:
    """Schläfli star polygon ``{n/k}``.

    Vertices are placed on a circle of given ``radius``. Edges connect
    vertex ``i`` to vertex ``(i + k) mod n``. When ``gcd(n, k) > 1`` the
    figure decomposes into ``gcd(n, k)`` disjoint compound polygons; the
    return type is then ``polygon_set``.

    Parameters
    ----------
    n : int
        Number of vertices on the circle, ``n >= 3``.
    k : int
        Step size, ``1 <= k < n``.
    radius : float, default=1.0

    Returns
    -------
    GeometryData
        - ``geom_type='polygon'`` when ``gcd(n, k) == 1``
        - ``geom_type='polygon_set'`` when ``gcd(n, k) > 1``
    """
    if n < 3:
        raise ValueError(f"n must be >= 3, got {n!r}.")
    if not (1 <= k < n):
        raise ValueError(f"k must satisfy 1 <= k < n, got n={n}, k={k}.")
    if radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius!r}.")

    angles = 2 * np.pi * np.arange(n) / n
    base_vertices = np.stack(
        [radius * np.cos(angles), radius * np.sin(angles)], axis=1
    )

    g = math.gcd(n, k)
    if g == 1:
        # Single closed compound traversal: i, (i+k), (i+2k), ... mod n.
        order = (np.arange(n) * k) % n
        coords = base_vertices[order]
        return GeometryData(
            geom_type="polygon",
            coordinates=coords,
            parameters={"n": int(n), "k": int(k), "radius": float(radius)},
            metadata={
                "kind": "star_polygon",
                "schlafli": f"{{{n}/{k}}}",
                "compound": False,
                "n_vertices": int(n),
            },
        )

    # Compound: g disjoint polygons of length n // g.
    polygons = []
    seen = np.zeros(n, dtype=bool)
    for start in range(n):
        if seen[start]:
            continue
        path = []
        i = start
        while not seen[i]:
            seen[i] = True
            path.append(i)
            i = (i + k) % n
        polygons.append(base_vertices[np.asarray(path)])
    return GeometryData(
        geom_type="polygon_set",
        coordinates=polygons,
        parameters={"n": int(n), "k": int(k), "radius": float(radius)},
        metadata={
            "kind": "star_polygon",
            "schlafli": f"{{{n}/{k}}}",
            "compound": True,
            "n_components": int(g),
            "vertices_per_component": int(n // g),
        },
    )


# ---------------------------------------------------------- times-table circle


def times_table_circle(
    n_points: int,
    multiplier: float,
    radius: float = 1.0,
) -> GeometryData:
    """Modular-multiplication "times-table" pattern on a circle.

    ``n_points`` points are placed evenly on a circle of given ``radius``.
    For each ``i ∈ [0, n_points)``, an edge is drawn from ``i`` to
    ``int(round(i * multiplier)) mod n_points``. Self-loops (the i==j case)
    are dropped.

    Parameters
    ----------
    n_points : int, must be >= 2
    multiplier : float
        Modular multiplier. Integer multipliers produce the classic
        Mardi-Gras patterns; non-integer values produce richer textures.
    radius : float, default=1.0

    Returns
    -------
    GeometryData
        ``geom_type='graph'``.
    """
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points!r}.")
    if radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius!r}.")

    angles = 2 * np.pi * np.arange(n_points) / n_points
    coords = np.stack(
        [radius * np.cos(angles), radius * np.sin(angles)], axis=1
    )

    targets = np.mod(np.round(np.arange(n_points) * float(multiplier)).astype(int), n_points)
    sources = np.arange(n_points)
    mask = sources != targets
    edges = np.stack([sources[mask], targets[mask]], axis=1)

    return GeometryData(
        geom_type="graph",
        coordinates=coords,
        edges=edges,
        parameters={
            "n_points": int(n_points),
            "multiplier": float(multiplier),
            "radius": float(radius),
        },
        metadata={"kind": "times_table_circle", "n_edges": int(edges.shape[0])},
    )


def times_table_from_input(
    input: HarmonicInput,
    n_points: int = 360,
    mode: str = "ratio",
    radius: float = 1.0,
) -> GeometryData:
    """Chord-driven times-table: one edge family per harmonic ratio.

    Each component of ``input`` contributes its own multiplier; all edge
    families share the same ``n_points``-vertex circle and are returned in
    a single :class:`GeometryData` so they can be drawn as overlaid colour
    layers.

    Parameters
    ----------
    input : HarmonicInput
    n_points : int, default=360
        Number of points on the circle.  360 is a convenient default
        because most rational chord ratios give clean modular periods.
    mode : {'ratio', 'pitch_class', 'integer'}, default='ratio'
        How each ratio is converted into a multiplier:

        * ``'ratio'`` — multiplier = ``ratio`` (float).
          ``i → int(round(i * ratio)) mod n_points``.
        * ``'pitch_class'`` — multiplier = ``round(n_points * log2(ratio))``.
          One "octave" wraps the circle once.
        * ``'integer'`` — multiplier = ``Fraction(ratio).limit_denominator(32).numerator``.
    radius : float, default=1.0

    Returns
    -------
    GeometryData
        ``geom_type='graph'``.  ``edges`` carries every edge across all
        ratio families; ``metadata['ratio_index']`` is an int array
        aligned with ``edges`` mapping each edge to the originating ratio
        index (so a renderer can colour each family separately).
        ``metadata['multipliers']`` lists the resolved multiplier per
        ratio.
    """
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points!r}.")
    if radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius!r}.")
    if mode not in {"ratio", "pitch_class", "integer"}:
        raise ValueError(
            f"mode must be 'ratio', 'pitch_class' or 'integer', got {mode!r}."
        )

    ratios = input.to_ratios()
    if not ratios:
        raise ValueError("times_table_from_input requires at least one ratio.")

    angles = 2 * np.pi * np.arange(n_points) / n_points
    coords = np.stack(
        [radius * np.cos(angles), radius * np.sin(angles)], axis=1
    )

    edges_all: List[List[int]] = []
    ratio_idx: List[int] = []
    multipliers: List[float] = []

    for k, r in enumerate(ratios):
        rf = float(r)
        if mode == "ratio":
            mult = rf
        elif mode == "pitch_class":
            mult = float(round(n_points * math.log2(max(rf, 1e-9))))
        else:  # integer
            from fractions import Fraction as _F
            mult = float(_F(rf).limit_denominator(32).numerator)
        multipliers.append(mult)

        targets = np.mod(
            np.round(np.arange(n_points) * mult).astype(int), n_points
        )
        sources = np.arange(n_points)
        mask = sources != targets
        for s, t in zip(sources[mask].tolist(), targets[mask].tolist()):
            edges_all.append([s, t])
            ratio_idx.append(k)

    edges = (np.asarray(edges_all, dtype=np.int64)
             if edges_all else np.empty((0, 2), dtype=np.int64))

    return GeometryData(
        geom_type="graph",
        coordinates=coords,
        edges=edges,
        parameters={
            "n_points": int(n_points),
            "mode": mode,
            "radius": float(radius),
        },
        metadata={
            "kind": "times_table_from_input",
            "n_ratios": len(ratios),
            "multipliers": multipliers,
            "ratio_index": np.asarray(ratio_idx, dtype=np.int64),
        },
    )


# --------------------------------------------------------------- tuning circle


def tuning_circle(input: HarmonicInput, radius: float = 1.0) -> GeometryData:
    """Place input components on a circle by their log-equave pitch class.

    For each ratio ``r``, the angle is ``2π · log_equave(r)``, wrapped into
    ``[0, 2π)``. Amplitudes are exposed as per-point weights.

    Parameters
    ----------
    input : HarmonicInput
    radius : float, default=1.0

    Returns
    -------
    GeometryData
        ``geom_type='point_cloud_2d'`` with shape ``(n_components, 2)`` and
        ``weights`` of length ``n_components``.
    """
    if radius <= 0:
        raise ValueError(f"radius must be > 0, got {radius!r}.")

    ratios = input.to_ratios()
    n = len(ratios)
    if n == 0:
        raise ValueError("HarmonicInput must have at least one component.")

    pitch_classes = np.array(
        [log_ratio(r, equave=input.equave) % 1.0 for r in ratios],
        dtype=np.float64,
    )
    angles = 2 * np.pi * pitch_classes
    coords = np.stack(
        [radius * np.cos(angles), radius * np.sin(angles)], axis=1
    )
    weights = input.normalized_amplitudes()

    return GeometryData(
        geom_type="point_cloud_2d",
        coordinates=coords,
        weights=weights,
        parameters={"radius": float(radius), "equave": float(input.equave)},
        metadata={
            "kind": "tuning_circle",
            "pitch_classes": pitch_classes.tolist(),
            "n_components": int(n),
        },
    )


# ------------------------------------------------------------------ rose curve


def rose_curve(
    ratio: RatioLike,
    n_points: int = 2000,
    n_periods: Optional[int] = None,
    radius: float = 1.0,
) -> GeometryData:
    """Polar rose: ``r(θ) = radius · cos((p/q) · θ)``.

    For coprime ``(p, q)``, the curve closes after:
    - ``θ ∈ [0, q · π]`` if ``p + q`` is even,
    - ``θ ∈ [0, 2 · q · π]`` if ``p + q`` is odd.

    If ``n_periods`` is given, the curve is sampled over ``[0, n_periods · π]``
    explicitly; otherwise the closure-aware default above is used.

    Parameters
    ----------
    ratio : Fraction, int, float, or (int, int)
    n_points : int, default=2000
    n_periods : int, optional
        Override the auto-computed sampling range (in units of π).
    radius : float, default=1.0

    Returns
    -------
    GeometryData
        ``geom_type='curve_2d'``.
    """
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points!r}.")

    p, q = coprime_pair(ratio)
    if n_periods is None:
        n_periods = q if (p + q) % 2 == 0 else 2 * q

    theta = np.linspace(0.0, float(n_periods) * np.pi, n_points)
    r = float(radius) * np.cos((p / q) * theta)
    coords = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)

    return GeometryData(
        geom_type="curve_2d",
        coordinates=coords,
        parameters={
            "ratio": Fraction(p, q),
            "n_points": int(n_points),
            "n_periods": int(n_periods),
            "radius": float(radius),
        },
        metadata={
            "kind": "rose",
            "p": p,
            "q": q,
            "petals": p if (p + q) % 2 == 0 else 2 * p,
        },
    )


# ----------------------------------------------------------- (epi|hypo)cycloid


def epicycloid(
    ratio: RatioLike,
    R: float = 1.0,
    n_points: int = 2000,
) -> GeometryData:
    """Epicycloid traced by a point on a small circle rolling outside a large one.

    With ``ratio = R / r = p / q`` (coprime), the curve has ``p`` cusps and
    closes after the small circle completes ``q`` revolutions.

    Parameters
    ----------
    ratio : Fraction, int, float, or (int, int)
        Ratio of fixed circle radius to rolling circle radius, ``R / r``.
    R : float, default=1.0
        Radius of the fixed (large) circle.
    n_points : int, default=2000

    Returns
    -------
    GeometryData
        ``geom_type='curve_2d'``.
    """
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points!r}.")
    if R <= 0:
        raise ValueError(f"R must be > 0, got {R!r}.")

    p, q = coprime_pair(ratio)
    if p <= 0 or q <= 0:
        raise ValueError(f"epicycloid requires positive p/q; got {p}/{q}.")
    r_small = R * q / p

    # Closure: t goes through 2π · q so the small circle returns home.
    t = np.linspace(0.0, 2.0 * np.pi * q, n_points)
    k = (R + r_small) / r_small
    x = (R + r_small) * np.cos(t) - r_small * np.cos(k * t)
    y = (R + r_small) * np.sin(t) - r_small * np.sin(k * t)
    coords = np.stack([x, y], axis=1)

    return GeometryData(
        geom_type="curve_2d",
        coordinates=coords,
        parameters={
            "ratio": Fraction(p, q),
            "R": float(R),
            "r": float(r_small),
            "n_points": int(n_points),
        },
        metadata={"kind": "epicycloid", "cusps": p},
    )


def hypocycloid(
    ratio: RatioLike,
    R: float = 1.0,
    n_points: int = 2000,
) -> GeometryData:
    """Hypocycloid traced by a point on a small circle rolling inside a large one.

    With ``ratio = R / r = p / q`` (coprime, ``p > q``), the curve has
    ``p - q`` cusps. For ``p = q`` the trace is a degenerate point.

    Parameters
    ----------
    ratio : Fraction, int, float, or (int, int)
        ``R / r``. Must satisfy ``R > r > 0`` for a non-degenerate curve;
        i.e., the coprime form must have ``p > q``.
    R : float, default=1.0
    n_points : int, default=2000

    Returns
    -------
    GeometryData
        ``geom_type='curve_2d'``.
    """
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points!r}.")
    if R <= 0:
        raise ValueError(f"R must be > 0, got {R!r}.")

    p, q = coprime_pair(ratio)
    if p <= q:
        raise ValueError(
            f"hypocycloid requires ratio > 1 (p > q); got {p}/{q}."
        )
    r_small = R * q / p

    t = np.linspace(0.0, 2.0 * np.pi * q, n_points)
    k = (R - r_small) / r_small
    x = (R - r_small) * np.cos(t) + r_small * np.cos(k * t)
    y = (R - r_small) * np.sin(t) - r_small * np.sin(k * t)
    coords = np.stack([x, y], axis=1)

    return GeometryData(
        geom_type="curve_2d",
        coordinates=coords,
        parameters={
            "ratio": Fraction(p, q),
            "R": float(R),
            "r": float(r_small),
            "n_points": int(n_points),
        },
        metadata={"kind": "hypocycloid", "cusps": p - q},
    )


# ============================================ Phase 4: metric-driven variants


def _tuning_circle_positions(input: HarmonicInput, radius: float) -> np.ndarray:
    """Place input components on the equave-circle; angles in radians."""
    ratios = input.to_ratios()
    pcs = np.array(
        [log_ratio(r, equave=input.equave) % 1.0 for r in ratios],
        dtype=np.float64,
    )
    angles = 2.0 * np.pi * pcs
    return np.stack(
        [radius * np.cos(angles), radius * np.sin(angles)], axis=1
    )


def _resolve_pair_metric(
    metric: Union[str, Callable[[float], float]],
) -> Callable[[float, float], float]:
    """Return a pair-wise metric function ``f(r_i, r_j) -> score``.

    String options are resolved against ``biotuner.metrics``. Most biotuner
    metrics take a single ratio (e.g. ``dyad_similarity(r)``); for pairs we
    apply them to the quotient ``r_j / r_i`` (rebound to ``> 1``).
    """
    from biotuner import metrics as bt_metrics  # lazy

    def _rebound_quotient(a: float, b: float) -> float:
        q = b / a if a != 0 else 1.0
        if q < 1.0 and q > 0.0:
            q = 1.0 / q
        return q

    if callable(metric):
        single = metric

        def pair(a: float, b: float) -> float:
            return float(single(_rebound_quotient(a, b)))

        return pair

    name = str(metric)
    if name == "dyad_similarity":
        fn = bt_metrics.dyad_similarity
        return lambda a, b: float(fn(_rebound_quotient(a, b)))
    if name == "compute_consonance":
        fn = bt_metrics.compute_consonance
        return lambda a, b: float(fn(_rebound_quotient(a, b)))
    if name == "tenneyHeight":
        fn = bt_metrics.tenneyHeight
        # Lower = more consonant. Invert so higher = more consonant for edge weights.
        return lambda a, b: -float(fn([float(a), float(b)]))
    if name == "log_distance":
        # Falls back to a trivial log-distance metric (no biotuner needed).
        return lambda a, b: -abs(math.log(_rebound_quotient(a, b)))
    raise ValueError(
        f"Unknown metric {metric!r}. Use a callable or one of "
        "'dyad_similarity', 'compute_consonance', 'tenneyHeight', 'log_distance'."
    )


def interval_vector_diagram(
    input: HarmonicInput,
    radius: float = 1.0,
    bin_cents: float = 50.0,
) -> GeometryData:
    """Graph of pairwise intervals binned into cents-classes.

    Nodes are placed on the tuning circle. For each pair ``(i, j)``,
    the interval in cents is computed (modulo equave). Intervals are
    bucketed in ``bin_cents``-wide classes; an edge's weight is the
    *interval-class count* — how many other pairs share that bucket.
    Edges thus highlight the chord's interval-vector multiplicities.

    Parameters
    ----------
    input : HarmonicInput
    radius : float, default=1.0
    bin_cents : float, default=50.0
        Bucket width for grouping intervals into classes.

    Returns
    -------
    GeometryData
        ``geom_type='graph'`` with ``edges`` of shape ``(E, 2)`` and
        per-edge ``weights``.
    """
    if bin_cents <= 0:
        raise ValueError(f"bin_cents must be > 0, got {bin_cents!r}.")
    coords = _tuning_circle_positions(input, radius)
    ratios = np.array([float(r) for r in input.to_ratios()], dtype=np.float64)
    n = len(ratios)
    if n < 2:
        raise ValueError("interval_vector_diagram needs at least 2 components.")

    equave_cents = 1200.0 * math.log2(input.equave)
    edges = []
    raw_intervals = []
    for i in range(n):
        for j in range(i + 1, n):
            cents = 1200.0 * math.log2(ratios[j] / ratios[i])
            cents = cents % equave_cents
            cents = min(cents, equave_cents - cents)  # interval class folding
            edges.append((i, j))
            raw_intervals.append(cents)
    raw_intervals_arr = np.asarray(raw_intervals)

    # Bucket and count.
    buckets = np.round(raw_intervals_arr / bin_cents).astype(int)
    weights = np.array(
        [int(np.sum(buckets == b)) for b in buckets],
        dtype=np.float64,
    )

    return GeometryData(
        geom_type="graph",
        coordinates=coords,
        edges=np.asarray(edges, dtype=np.int64),
        weights=weights,
        parameters={
            "radius": float(radius),
            "bin_cents": float(bin_cents),
            "equave": float(input.equave),
        },
        metadata={
            "kind": "interval_vector_diagram",
            "n_nodes": int(n),
            "n_edges": int(len(edges)),
            "intervals_cents": raw_intervals_arr.tolist(),
        },
    )


def polygon_chord_pattern(
    input: HarmonicInput,
    metric: Union[str, Callable[[float], float]] = "dyad_similarity",
    threshold: Optional[float] = None,
    radius: float = 1.0,
) -> GeometryData:
    """Chord-pattern graph weighted by a biotuner harmonicity metric.

    Nodes are placed on the tuning circle. Every pair of distinct
    components has an edge weighted by ``metric(ratio_j / ratio_i)``
    (after rebounding the quotient into ``[1, ∞)``). Optionally
    threshold to keep only the strongest pairs — yields the polygonal
    "chord skeleton" of the most consonant relationships.

    Parameters
    ----------
    input : HarmonicInput
    metric : str or callable, default='dyad_similarity'
        Either a single-ratio biotuner metric name (one of
        ``'dyad_similarity'``, ``'compute_consonance'``,
        ``'tenneyHeight'``, ``'log_distance'``) or a callable
        ``ratio -> score``. ``tenneyHeight`` is sign-flipped so higher
        always means "more consonant" downstream.
    threshold : float, optional
        Drop edges with weight below ``threshold``. ``None`` keeps all.
    radius : float, default=1.0

    Returns
    -------
    GeometryData
        ``geom_type='graph'``.
    """
    coords = _tuning_circle_positions(input, radius)
    ratios = [float(r) for r in input.to_ratios()]
    n = len(ratios)
    if n < 2:
        raise ValueError("polygon_chord_pattern needs at least 2 components.")
    pair_metric = _resolve_pair_metric(metric)

    edges_list = []
    weights_list = []
    for i in range(n):
        for j in range(i + 1, n):
            w = pair_metric(ratios[i], ratios[j])
            if threshold is not None and w < threshold:
                continue
            edges_list.append((i, j))
            weights_list.append(float(w))

    edges = np.asarray(edges_list, dtype=np.int64) if edges_list else np.empty((0, 2), dtype=np.int64)
    weights = np.asarray(weights_list, dtype=np.float64) if weights_list else np.empty((0,), dtype=np.float64)

    return GeometryData(
        geom_type="graph",
        coordinates=coords,
        edges=edges,
        weights=weights,
        parameters={
            "metric": metric if isinstance(metric, str) else "<callable>",
            "threshold": threshold,
            "radius": float(radius),
        },
        metadata={
            "kind": "polygon_chord_pattern",
            "n_nodes": int(n),
            "n_edges": int(edges.shape[0]),
        },
    )


def consonance_polygon(
    input: HarmonicInput,
    metric: Union[str, Callable[[float], float]] = "dyad_similarity",
    radius: float = 1.0,
) -> GeometryData:
    """Convex polygon whose vertex angles encode each ratio's consonance share.

    Each component's "consonance share" is the sum of pairwise
    ``metric`` scores against all other components. Vertices are then
    placed at cumulative-angle positions: the polygon's angular density
    spikes around the most-connected ratios and thins around outliers.

    Parameters
    ----------
    input : HarmonicInput
    metric : str or callable, default='dyad_similarity'
    radius : float, default=1.0

    Returns
    -------
    GeometryData
        ``geom_type='polygon'`` with ``weights`` carrying the per-vertex
        consonance share. The first vertex is placed at angle 0.
    """
    ratios = [float(r) for r in input.to_ratios()]
    n = len(ratios)
    if n < 3:
        raise ValueError("consonance_polygon needs at least 3 components.")
    pair_metric = _resolve_pair_metric(metric)

    # Per-vertex consonance share = sum of pairwise scores.
    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            scores[i] += pair_metric(ratios[i], ratios[j])

    # Shift to non-negative so cumulative angles are monotonic, even if the
    # metric is negative (e.g. tenneyHeight after sign-flip can still be
    # arbitrary). Add a small epsilon so all-equal scores still produce a
    # valid polygon.
    shifted = scores - np.min(scores) + 1e-9
    total = float(np.sum(shifted))
    angles = 2.0 * np.pi * np.cumsum(shifted) / total
    # Anchor first vertex at angle 0 by rotating everything.
    angles = angles - angles[0]
    coords = np.stack(
        [radius * np.cos(angles), radius * np.sin(angles)], axis=1
    )

    return GeometryData(
        geom_type="polygon",
        coordinates=coords,
        weights=scores,
        parameters={
            "metric": metric if isinstance(metric, str) else "<callable>",
            "radius": float(radius),
        },
        metadata={
            "kind": "consonance_polygon",
            "n_vertices": int(n),
            "consonance_share": scores.tolist(),
        },
    )
