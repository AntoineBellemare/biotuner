"""Scales as figures: star polygons, interval webs, nested families.

The old ``vizs`` MOS plots made attractive patterns almost by accident. The
static ``plot_labyrinth`` collapsed every scale onto radius 1 or 2, which
destroyed it as a labyrinth but turned it into a dense web of overlapping
polygons; the interactive one drew stacked generators as spirals. Both are
kept in :mod:`biotuner.mos.legacy`.

This module does that deliberately, and makes each pattern *mean* something.
Nothing here is decoration for its own sake:

``'chain'``
    Connect the degrees in generator-chain order rather than pitch order. For a
    well-formed scale that closes into a **star polygon**: the diatonic circle
    of fifths is the heptagram ``{7/3}``, the chromatic one ``{12/5}``. The
    shape *is* the scale's generator structure. Its density is the modular
    inverse of Carey's ``WF(N, g)``, not that number itself -- see
    :func:`star_hop`.
``'ring'``
    Consecutive degrees in pitch order: a polygon whose edge lengths are the
    step sizes, so the large/small distribution is the silhouette.
``'web'``
    Every pair of degrees, coloured and weighted by the harmonicity of the
    interval between them. A scale's whole interval content in one figure.
``'nested'``
    One ring per member of the generator's family, so the embedding structure
    -- each scale a subset of the next -- is visible as concentric shapes.
``'spiral'``
    The original stacked-generator spiral, kept for continuity.

Every style takes a musical ``mode``, and the geometry is available separately
from the drawing through :func:`web_geometry`, so the shapes can be exported or
tested without a figure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from biotuner.mos import theory as T
from biotuner.mos.scale import MOSScale, mos_family

__all__ = [
    "STYLES",
    "PALETTES",
    "WebGeometry",
    "web_geometry",
    "star_density",
    "star_hop",
    "plot_scale_web",
    "plot_web_gallery",
]

#: Drawing styles, each encoding a different fact about the scale.
STYLES: Tuple[str, ...] = ("chain", "ring", "web", "nested", "spiral")

#: Ground / ink / accent triples. ``'ink'`` and ``'noir'`` are for print-like
#: output where the figure is the whole point rather than an inset in a page.
PALETTES: Dict[str, Dict[str, object]] = {
    "light": dict(bg="#ffffff", fg="#22252b", grid="#e4e7e9", cmap="viridis"),
    "dark": dict(bg="#12161a", fg="#e6ebec", grid="#232b30", cmap="viridis"),
    "ink": dict(bg="#f6f4ee", fg="#1a1a18", grid="#ddd9cd", cmap="copper"),
    "noir": dict(bg="#0b0b0d", fg="#f2f2f0", grid="#1c1c20", cmap="magma"),
}


@dataclass(frozen=True)
class WebGeometry:
    """The shape a style produces, before anything is drawn.

    Attributes
    ----------
    points : np.ndarray, shape (n, 2)
        Vertex positions in Cartesian coordinates on the unit circle (or, for
        ``'spiral'`` and ``'nested'``, on nested circles).
    segments : np.ndarray, shape (m, 2, 2)
        Line segments as ``[[x0, y0], [x1, y1]]`` pairs.
    weights : np.ndarray, shape (m,)
        One value per segment in ``[0, 1]``, raw: harmonicity for ``'web'``,
        depth for the others. :func:`plot_scale_web` stretches these to the
        colour ramp at draw time rather than storing them pre-scaled, so the
        numbers here stay comparable between scales.
    labels : tuple of str
        One per point, for annotation.
    style : str
    """

    points: np.ndarray
    segments: np.ndarray
    weights: np.ndarray
    labels: Tuple[str, ...]
    style: str

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def n_segments(self) -> int:
        return len(self.segments)


def star_hop(scale: MOSScale) -> int:
    """How many *pitch* steps one generator step advances.

    Carey's ``WF(N, g)`` says how many places along the generator chain one
    *scale step* advances. Drawing the chain asks the reverse question, and the
    two are modular inverses, not the same number: the diatonic's ``WF(7, 2)``
    means one scale step is two fifths, so one fifth is ``2⁻¹ mod 7 = 4`` scale
    steps.

    Examples
    --------
    >>> from biotuner.mos.scale import MOSScale
    >>> star_hop(MOSScale.from_generator(3 / 2, 7))
    4
    >>> star_hop(MOSScale.from_generator(3 / 2, 12))
    7
    """
    from biotuner.mos.modes import wf_number

    n = scale.cardinality
    wf = wf_number(scale.generator, n) % n
    return pow(wf, -1, n)


def star_density(scale: MOSScale) -> int:
    """The Schläfli density of the star polygon the generator chain traces.

    Connecting the degrees in the order the generator produces them, rather
    than in pitch order, closes into a star polygon ``{N/d}``. Stepping by
    :func:`star_hop` and stepping by ``N - star_hop`` trace the same figure in
    opposite directions, so the density is conventionally the smaller of the
    two.

    Examples
    --------
    The circle of fifths on the seven diatonic notes is the familiar heptagram
    ``{7/3}``, and on the twelve chromatic ones the dodecagram ``{12/5}``:

    >>> from biotuner.mos.scale import MOSScale
    >>> star_density(MOSScale.from_generator(3 / 2, 7))
    3
    >>> star_density(MOSScale.from_generator(3 / 2, 12))
    5
    """
    n = scale.cardinality
    hop = star_hop(scale)
    return min(hop, n - hop)


def _unit(degrees: Sequence[float], radius: float = 1.0) -> np.ndarray:
    """Period fractions to points on a circle, zero at the top, clockwise."""
    ang = np.asarray(degrees, dtype=float) * 2.0 * math.pi
    return np.stack([radius * np.sin(ang), radius * np.cos(ang)], axis=1)


def _stretch(weights: np.ndarray, floor: float = 0.12) -> np.ndarray:
    """Rescale weights to use the whole colour ramp.

    Harmonicity is the reason this is needed. ``dyad_similarity`` scores a
    tempered interval against low integer ratios, and a scale built from an
    irrational generator has none, so a 31-EDO diatonic's twenty-one intervals
    all land between 0.002 and 0.019 -- indistinguishable at the dark end of
    any colormap. The absolute value carries no information here; the ordering
    within one scale does, so the drawing stretches to it.

    A near-constant input is left flat rather than amplified into false
    contrast.
    """
    if weights.size == 0:
        return weights
    lo, hi = float(weights.min()), float(weights.max())
    if hi - lo < 1e-9:
        return np.full_like(weights, 0.6)
    return floor + (1.0 - floor) * (weights - lo) / (hi - lo)


def _harmonicity(ratios: Sequence[float]) -> np.ndarray:
    """Dyad similarity of each ratio, normalised to ``[0, 1]``.

    Falls back to a flat mid value if the metric refuses an input, so a single
    awkward ratio cannot blank an entire figure.
    """
    from biotuner.metrics import dyad_similarity

    out = []
    for r in ratios:
        try:
            out.append(float(dyad_similarity(float(r))))
        except Exception:
            out.append(50.0)
    arr = np.asarray(out, dtype=float)
    arr = np.clip(arr, 0.0, 100.0) / 100.0
    return arr


def web_geometry(
    scale: MOSScale,
    style: str = "chain",
    *,
    mode: int = 0,
    max_cardinality: int = 24,
    min_harmonicity: float = 0.0,
) -> WebGeometry:
    """The geometry one style produces for one scale, without drawing it.

    Parameters
    ----------
    scale : MOSScale
    style : {'chain', 'ring', 'web', 'nested', 'spiral'}, default 'chain'
        See the module docstring for what each one encodes.
    mode : int, default 0
        Rotate to this mode first. Changes which tone sits at the top of the
        circle, and for ``'chain'`` which vertex the star starts from.
    max_cardinality : int, default 24
        Only used by ``'nested'`` and ``'spiral'``.
    min_harmonicity : float, default 0.0
        ``'web'`` only: drop segments below this normalised harmonicity, which
        turns the full web into a consonance skeleton.

    Returns
    -------
    WebGeometry

    Examples
    --------
    The diatonic's chain closes into a seven-pointed star:

    >>> from biotuner.mos.scale import MOSScale
    >>> d = MOSScale.from_generator(3 / 2, 7)
    >>> g = web_geometry(d, "chain")
    >>> g.n_points, g.n_segments
    (7, 7)

    Its full web has one segment per unordered pair:

    >>> web_geometry(d, "web").n_segments
    21
    """
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}, got {style!r}")
    if not 0.0 <= min_harmonicity <= 1.0:
        raise ValueError(
            f"min_harmonicity must lie in [0, 1], got {min_harmonicity!r}"
        )

    m = scale.mode(mode % scale.cardinality)
    degrees = list(m.degrees)
    n = len(degrees)
    labels = tuple(f"{c:.0f}" for c in m.cents)

    if style == "ring":
        pts = _unit(degrees)
        idx = [(i, (i + 1) % n) for i in range(n)]
        segs = np.array([[pts[a], pts[b]] for a, b in idx])
        # Edge length is the step size; weight it so large steps read heavier.
        steps = [(degrees[(i + 1) % n] - degrees[i]) % 1.0 for i in range(n)]
        w = np.asarray(steps) / max(steps)
        return WebGeometry(pts, segs, w, labels, style)

    if style == "chain":
        # Degrees in the order the generator produces them, not pitch order.
        chain = [(k * scale.generator) % 1.0 for k in range(n)]
        pts = _unit(chain)
        idx = [(i, (i + 1) % n) for i in range(n)]
        segs = np.array([[pts[a], pts[b]] for a, b in idx])
        w = np.linspace(1.0, 0.35, n)  # fade along the chain, so it reads as a path
        lab = tuple(f"{(c % 1.0) * scale.period_cents:.0f}" for c in chain)
        return WebGeometry(pts, segs, w, lab, style)

    if style == "web":
        pts = _unit(degrees)
        pairs, ratios = [], []
        for i in range(n):
            for j in range(i + 1, n):
                gap = abs(degrees[i] - degrees[j])
                gap = min(gap, 1.0 - gap)
                pairs.append((i, j))
                ratios.append(scale.period**gap)
        w = _harmonicity(ratios)
        keep = w >= min_harmonicity
        pairs = [p for p, k in zip(pairs, keep) if k]
        w = w[keep]
        segs = (np.array([[pts[a], pts[b]] for a, b in pairs])
                if pairs else np.empty((0, 2, 2)))
        return WebGeometry(pts, segs, w, labels, style)

    family = mos_family(scale.generator_ratio, max_cardinality, scale.period)
    if not family:
        family = [scale]

    if style == "nested":
        all_pts, all_segs, all_w = [], [], []
        for k, member in enumerate(family):
            r = 0.28 + 0.72 * (k + 1) / len(family)
            degs = member.mode(mode % member.cardinality).degrees
            pts = _unit(degs, radius=r)
            all_pts.append(pts)
            c = member.cardinality
            for i in range(c):
                all_segs.append([pts[i], pts[(i + 1) % c]])
                all_w.append((k + 1) / len(family))
        pts = np.concatenate(all_pts)
        return WebGeometry(pts, np.array(all_segs), np.asarray(all_w),
                           tuple(f"{m2.signature}" for m2 in family), style)

    # spiral: angle is the stacked degree, radius its index in the stack --
    # the original figure, rebuilt.
    all_pts, all_segs, all_w = [], [], []
    for k, member in enumerate(family):
        c = member.cardinality
        chain = [((i + 1) * scale.generator) % 1.0 for i in range(c)]
        radii = np.linspace(0.25, 1.0, c)
        pts = np.stack(
            [radii * np.sin(np.asarray(chain) * 2 * math.pi),
             radii * np.cos(np.asarray(chain) * 2 * math.pi)], axis=1
        )
        all_pts.append(pts)
        for i in range(c - 1):
            all_segs.append([pts[i], pts[i + 1]])
            all_w.append((k + 1) / len(family))
    pts = np.concatenate(all_pts) if all_pts else np.empty((0, 2))
    segs = np.array(all_segs) if all_segs else np.empty((0, 2, 2))
    return WebGeometry(pts, segs, np.asarray(all_w),
                       tuple(m2.signature for m2 in family), style)


def plot_scale_web(
    scale: MOSScale,
    style: str = "chain",
    *,
    mode: int = 0,
    palette: str = "light",
    max_cardinality: int = 24,
    min_harmonicity: float = 0.0,
    show_points: bool = True,
    annotate: bool = False,
    linewidth: Tuple[float, float] = (0.5, 3.2),
    title: Optional[str] = None,
    ax=None,
    figsize: Tuple[float, float] = (7.0, 7.0),
):
    """Draw one scale as a figure.

    Parameters
    ----------
    scale : MOSScale
    style : {'chain', 'ring', 'web', 'nested', 'spiral'}, default 'chain'
    mode : int, default 0
        Which mode to rotate to, brightest first.
    palette : {'light', 'dark', 'ink', 'noir'}, default 'light'
    max_cardinality : int, default 24
        ``'nested'`` and ``'spiral'`` only.
    min_harmonicity : float, default 0.0
        ``'web'`` only: keep the consonance skeleton, drop the rest.
    show_points, annotate : bool
    linewidth : (float, float), default (0.5, 3.2)
        Segment width at the lowest and highest weight.
    title : str, optional
        ``None`` writes a short automatic one; pass ``''`` for none at all.
    ax : matplotlib axes, optional
        A plain Cartesian axes, not polar -- the geometry is precomputed.
    figsize : tuple

    Returns
    -------
    (fig, ax)

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> from biotuner.mos.scale import MOSScale
    >>> fig, ax = plot_scale_web(MOSScale.from_generator(3 / 2, 7), "chain")
    >>> ax.get_aspect()
    1.0
    >>> plt.close(fig)
    """
    if palette not in PALETTES:
        raise ValueError(
            f"palette must be one of {tuple(PALETTES)}, got {palette!r}"
        )
    pal = PALETTES[palette]
    geo = web_geometry(scale, style, mode=mode, max_cardinality=max_cardinality,
                       min_harmonicity=min_harmonicity)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    fig.patch.set_facecolor(pal["bg"])
    ax.set_facecolor(pal["bg"])

    circle = plt.Circle((0, 0), 1.0, fill=False, color=pal["grid"], lw=1.0,
                        zorder=0)
    ax.add_patch(circle)

    if geo.n_segments:
        cmap = plt.get_cmap(pal["cmap"])
        lo, hi = linewidth
        # Geometry keeps raw weights; the stretch is a drawing decision.
        shown = _stretch(geo.weights)
        widths = lo + (hi - lo) * shown
        colours = cmap(0.15 + 0.8 * shown)
        ax.add_collection(
            LineCollection(geo.segments, colors=colours, linewidths=widths,
                           alpha=0.85, capstyle="round", zorder=2)
        )

    if show_points and geo.n_points:
        ax.scatter(geo.points[:, 0], geo.points[:, 1], s=26,
                   color=pal["fg"], zorder=3, edgecolor=pal["bg"], linewidth=0.8)

    if annotate:
        for (x, y), lab in zip(geo.points, geo.labels * 8):
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=(0, 8),
                        fontsize=7, color=pal["fg"], ha="center")

    lim = 1.22
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axis("off")

    if title is None:
        bits = [scale.signature, style]
        if style == "chain":
            bits.append(f"star {{{scale.cardinality}/{star_density(scale)}}}")
        if mode:
            bits.append(f"mode {mode}")
        title = "  ·  ".join(bits)
    if title:
        ax.set_title(title, fontsize=11, color=pal["fg"], pad=14)
    return fig, ax


def plot_web_gallery(
    scales: Sequence[MOSScale],
    style: str = "chain",
    *,
    mode: int = 0,
    palette: str = "light",
    n_cols: int = 4,
    panel: float = 3.0,
    **kwargs,
):
    """A grid of scale figures -- one generator's family, or many scales.

    Parameters
    ----------
    scales : sequence of MOSScale
    style : str, default 'chain'
    mode : int, default 0
    palette : str, default 'light'
    n_cols : int, default 4
    panel : float, default 3.0
        Side length of each panel, in inches.
    **kwargs
        Forwarded to :func:`plot_scale_web`.

    Returns
    -------
    (fig, axes)

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> from biotuner.mos.scale import mos_family
    >>> fig, axes = plot_web_gallery(mos_family(3 / 2, 17), n_cols=3)
    >>> len(axes)
    5
    >>> plt.close(fig)
    """
    scales = list(scales)
    if not scales:
        raise ValueError("nothing to draw: scales is empty")
    n_cols = max(1, min(n_cols, len(scales)))
    n_rows = math.ceil(len(scales) / n_cols)
    pal = PALETTES.get(palette, PALETTES["light"])

    fig, grid = plt.subplots(n_rows, n_cols,
                             figsize=(panel * n_cols, panel * n_rows))
    fig.patch.set_facecolor(pal["bg"])
    flat = np.atleast_1d(np.asarray(grid)).ravel()
    for ax in flat[len(scales):]:
        ax.set_facecolor(pal["bg"])
        ax.axis("off")
    used = []
    for scale, ax in zip(scales, flat):
        plot_scale_web(scale, style, mode=mode, palette=palette, ax=ax, **kwargs)
        used.append(ax)
    fig.tight_layout()
    return fig, used
