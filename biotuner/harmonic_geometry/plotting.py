"""
biotuner.harmonic_geometry.plotting
====================================
Visualisation for :class:`GeometryData` objects.

This module is the single home for matplotlib-based rendering of every
``geom_type`` produced by the rest of :mod:`biotuner.harmonic_geometry`
(curve/polygon/graph/tree/field/mesh/point-cloud, in 2-D and 3-D), plus
the higher-level helpers used by the report generator scripts (galleries,
sweep strips, rotation gifs, GA fitness curves, grid-relaxation animations).

Matplotlib is imported lazily inside each function so that simply importing
``biotuner.harmonic_geometry`` does not pull in the matplotlib stack.

Sections
--------
1. Style helpers           — palette, axis cleanup, figure save
2. 2-D primitive renderers — curve/polygon/graph/field/cloud/tree/rectangles
3. 3-D primitive renderers — mesh/cloud/tree, plus 3-D axis helpers
4. Dispatcher              — :func:`plot_geometry`
5. Layout helpers          — galleries and parameter sweeps
6. Animation helpers       — rotation, sequence, grid relax, evolution
7. Resonance helpers       — coupling matrices, GA curves, attractor graphs

Public API is exported through ``biotuner.harmonic_geometry.__init__`` so
external code can simply ``from biotuner.harmonic_geometry import plotting``
or import individual functions.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

from biotuner.harmonic_geometry.geometry_data import GeometryData

# ═══════════════════════════════════════════════════ 1. STYLE & HELPERS ════════

# Default 8-colour palette shared by every report.  Roughly matches Solarized
# accents — chosen for colour-blind-friendly contrast on white background.
PALETTE: Dict[str, str] = {
    "blue":    "#1f3b73",
    "red":     "#a23e2c",
    "green":   "#3a7a4d",
    "gold":    "#7a5d24",
    "purple":  "#5c2e7a",
    "teal":    "#1f6b6b",
    "orange":  "#b05f1a",
    "slate":   "#3d4f60",
}
CHORD_COLORS: List[str] = list(PALETTE.values())

# Default cosmetic constants (overridable as kwargs throughout this module).
_DEFAULT_DPI         = 150
_DEFAULT_FIG_WIDTH   = 6.5      # inches; column width
_DEFAULT_FACECOLOR   = "white"
_DEFAULT_BG          = "#F8F8F8"


def axis_clean(ax, equal: bool = True, grid: bool = False,
               spine_color: str = "#cccccc", spine_lw: float = 0.5) -> None:
    """Strip ticks, harmonise spines, optionally enforce equal aspect / grid."""
    if equal:
        try:
            ax.set_aspect("equal", adjustable="box")
        except Exception:
            pass
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(spine_color)
        sp.set_linewidth(spine_lw)
    if grid:
        ax.grid(alpha=0.25, linestyle=":")


def title_ax(ax, text: str, sub: str = "", fontsize: int = 8) -> None:
    """Two-line axis title; the second line is rendered smaller and dimmer."""
    if sub:
        ax.set_title(f"{text}\n{sub}", fontsize=fontsize, pad=4, color="#222222")
    else:
        ax.set_title(text, fontsize=fontsize, pad=4, color="#222222")


def make_axis_3d(fig, pos, title: str = "",
                 elev: int = 25, azim: int = 45):
    """Create a 3-D matplotlib axis with cleaned spines and equal aspect."""
    ax = fig.add_subplot(pos, projection="3d")
    if title:
        ax.set_title(title, fontsize=8, pad=3)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass
    return ax


def save_figure(fig, path: Union[str, Path],
                dpi: int = _DEFAULT_DPI, close: bool = True) -> Path:
    """Save ``fig`` to ``path`` (creating parent dirs) and optionally close it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=_DEFAULT_FACECOLOR)
    if close:
        import matplotlib.pyplot as plt
        plt.close(fig)
    return path


# ═══════════════════════════════════════════ 2. 2-D PRIMITIVE RENDERERS ═══════


def draw_curve_2d(geom: GeometryData, ax,
                  color: str = PALETTE["blue"], lw: float = 0.9,
                  marker: Optional[str] = None) -> None:
    """Polyline in 2-D."""
    coords = np.asarray(geom.coordinates)
    ax.plot(coords[:, 0], coords[:, 1], color=color, lw=lw,
            marker=marker, ms=3 if marker else None)
    axis_clean(ax)


def draw_polygon(geom: GeometryData, ax,
                 color: str = PALETTE["red"], lw: float = 1.4,
                 fill: bool = False, alpha: float = 0.08,
                 marker: Optional[str] = "o", ms: float = 4) -> None:
    """Closed polygon (auto-closes by appending the first vertex)."""
    coords = np.asarray(geom.coordinates)
    closed = np.vstack([coords, coords[:1]])
    ax.plot(closed[:, 0], closed[:, 1], color=color, lw=lw,
            marker=marker, ms=ms)
    if fill:
        ax.fill(closed[:, 0], closed[:, 1], color=color, alpha=alpha)
    axis_clean(ax)


def draw_polygon_set(geom: GeometryData, ax,
                     palette: Optional[Sequence[str]] = None,
                     lw: float = 1.4, ms: float = 3) -> None:
    """List/sequence of polygons (geom.coordinates is iterable of (k, 2) arrays)."""
    pal = list(palette) if palette is not None else CHORD_COLORS
    for i, poly in enumerate(geom.coordinates):
        c = pal[i % len(pal)]
        closed = np.vstack([poly, poly[:1]])
        ax.plot(closed[:, 0], closed[:, 1], color=c, lw=lw, marker="o", ms=ms)
    axis_clean(ax)


def draw_graph_2d(geom: GeometryData, ax,
                  edge_color: str = PALETTE["blue"], edge_lw: float = 0.4,
                  edge_alpha: float = 0.55,
                  node_color: str = PALETTE["red"], node_size: float = 4,
                  use_weights: bool = False) -> None:
    """Edges as a LineCollection plus node scatter."""
    from matplotlib.collections import LineCollection
    coords = np.asarray(geom.coordinates)
    edges  = np.asarray(geom.edges)
    if edges.size == 0:
        ax.scatter(coords[:, 0], coords[:, 1], s=node_size, color=node_color, zorder=3)
    else:
        segments = coords[edges]
        if use_weights and geom.weights is not None and len(geom.weights) == len(edges):
            import matplotlib.cm as cm
            wts = np.asarray(geom.weights)
            if wts.max() > wts.min():
                norm_wts = (wts - wts.min()) / (wts.max() - wts.min())
            else:
                norm_wts = np.full_like(wts, 0.5)
            colors = cm.get_cmap("cool")(norm_wts)
            lc = LineCollection(segments, colors=colors, linewidths=edge_lw, alpha=edge_alpha)
        else:
            lc = LineCollection(segments, colors=edge_color, linewidths=edge_lw, alpha=edge_alpha)
        ax.add_collection(lc)
        ax.scatter(coords[:, 0], coords[:, 1], s=node_size, color=node_color, zorder=3)
    pad = 0.05 * (np.max(coords) - np.min(coords) + 1e-9)
    ax.set_xlim(coords[:, 0].min() - pad, coords[:, 0].max() + pad)
    ax.set_ylim(coords[:, 1].min() - pad, coords[:, 1].max() + pad)
    axis_clean(ax)


def draw_hyperbolic_graph(geom: GeometryData, ax,
                          color_by: str = "harmonicity",
                          cmap: str = "plasma",
                          highlight_mask: Optional[np.ndarray] = None,
                          highlight_color: str = PALETTE["red"],
                          show_disk: bool = True) -> Any:
    """Stern-Brocot hyperbolic-disk layout. Returns the scatter (for colorbar)."""
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Circle
    coords = np.asarray(geom.coordinates)
    edges  = np.asarray(geom.edges)
    if show_disk:
        ax.add_patch(Circle((0, 0), 0.95, fill=False, edgecolor="#dddddd", lw=0.6))
    if edges.size:
        ax.add_collection(LineCollection(coords[edges], colors="#dddddd",
                                          linewidths=0.4, zorder=1))
    color_array = None
    if color_by and color_by in (geom.metadata or {}):
        color_array = np.asarray(geom.metadata[color_by])
    if color_array is not None and len(color_array) == len(coords):
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=color_array,
                        cmap=cmap, s=8, zorder=2,
                        vmin=0, vmax=1, edgecolors="none")
    else:
        sc = ax.scatter(coords[:, 0], coords[:, 1], s=8, zorder=2,
                        color=PALETTE["blue"], edgecolors="none")
    if highlight_mask is not None and highlight_mask.any():
        ax.scatter(coords[highlight_mask, 0], coords[highlight_mask, 1],
                   color=highlight_color, s=22, zorder=3,
                   edgecolors="white", lw=0.4)
    ax.set_xlim(-1.08, 1.08); ax.set_ylim(-1.08, 1.08)
    axis_clean(ax)
    return sc


def draw_point_cloud_2d(geom: GeometryData, ax,
                        color: str = PALETTE["red"],
                        edge_color: str = PALETTE["blue"],
                        size: float = 30, use_weights: bool = True,
                        ref_circle: bool = False) -> None:
    """2-D scatter; sizes scaled by geom.weights when present."""
    coords = np.asarray(geom.coordinates)
    if use_weights and geom.weights is not None:
        w = np.asarray(geom.weights)
        if w.max() > 0:
            sizes = (size * w / w.max()) * 6.0
        else:
            sizes = np.full(coords.shape[0], size)
    else:
        sizes = np.full(coords.shape[0], size)
    ax.scatter(coords[:, 0], coords[:, 1], s=sizes,
               color=color, edgecolor=edge_color, lw=0.8)
    if ref_circle:
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(th), np.sin(th), color="#888888", lw=0.5, ls="--")
    axis_clean(ax)


def draw_field_2d(geom: GeometryData, ax,
                  cmap: str = "RdBu_r",
                  show_nodal: bool = True,
                  signed: bool = True,
                  vmin: Optional[float] = None,
                  vmax: Optional[float] = None) -> Any:
    """Pcolormesh of a 2-D scalar field stored in geom.coordinates.

    For signed fields (e.g. Chladni) draws a zero-level contour. Returns the
    QuadMesh object (so callers can attach a colorbar).
    """
    field = np.asarray(geom.coordinates)
    grid = geom.field_grid
    if grid is None:
        ny, nx = field.shape
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    else:
        X, Y = grid
    if signed:
        if vmax is None:
            vmax = float(np.nanmax(np.abs(field))) or 1.0
        if vmin is None:
            vmin = -vmax
    else:
        if vmin is None:
            vmin = float(np.nanmin(field))
        if vmax is None:
            vmax = float(np.nanmax(field))
    mesh = ax.pcolormesh(X, Y, field, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    if show_nodal and signed:
        try:
            ax.contour(X, Y, field, levels=[0.0], colors="#222222", linewidths=0.6)
        except Exception:
            pass
    axis_clean(ax)
    return mesh


def draw_vector_field_2d(geom: GeometryData, ax,
                         style: str = "streamlines",
                         cmap: str = "viridis",
                         density: float = 1.2,
                         scale: Optional[float] = None,
                         step: int = 10,
                         color_by_magnitude: bool = True,
                         background: Optional[np.ndarray] = None,
                         background_cmap: str = "Greys",
                         background_alpha: float = 0.6,
                         linewidth: float = 0.8) -> Any:
    """Render a 2-D vector field stored as ``(H, W, 2)`` in ``coordinates``.

    Parameters
    ----------
    geom : GeometryData with ``geom_type='vector_field_2d'``
        ``coordinates`` is ``(H, W, 2)`` giving ``(u, v)`` per cell;
        ``field_grid`` is ``(X, Y)`` meshgrids.
    style : {"streamlines", "quiver"}
        ``"streamlines"`` (default) draws integral curves of the field;
        ``"quiver"`` draws arrows at a coarsened grid (one arrow per
        ``step`` cells).
    background : ndarray, optional
        Scalar field of the same shape as ``coordinates[..., 0]`` rendered
        underneath the vectors (e.g. the parent wave field).
    """
    field = np.asarray(geom.coordinates)
    if field.ndim != 3 or field.shape[-1] != 2:
        raise ValueError(
            f"draw_vector_field_2d expects (H, W, 2); got {field.shape}."
        )
    u = field[..., 0]
    v = field[..., 1]
    grid = geom.field_grid
    if grid is None:
        ny, nx = u.shape
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny))
    else:
        X, Y = grid

    if background is not None:
        bg = np.asarray(background)
        bv = float(np.nanmax(np.abs(bg))) or 1.0
        ax.pcolormesh(X, Y, bg, cmap=background_cmap,
                      vmin=-bv, vmax=bv, shading="auto",
                      alpha=background_alpha, zorder=0)

    if style == "quiver":
        s = max(int(step), 1)
        Xc = X[::s, ::s]
        Yc = Y[::s, ::s]
        Uc = u[::s, ::s]
        Vc = v[::s, ::s]
        mag = np.hypot(Uc, Vc) if color_by_magnitude else None
        kw: Dict[str, Any] = {"angles": "xy", "scale_units": "xy",
                              "pivot": "mid", "width": 0.003}
        if scale is not None:
            kw["scale"] = scale
        if mag is not None:
            quiv = ax.quiver(Xc, Yc, Uc, Vc, mag, cmap=cmap, **kw)
        else:
            quiv = ax.quiver(Xc, Yc, Uc, Vc, **kw)
        axis_clean(ax)
        return quiv

    # streamlines (default)
    mag = np.hypot(u, v)
    color = mag if color_by_magnitude else None
    sl = ax.streamplot(
        X, Y, u, v,
        density=density,
        color=color if color is not None else "0.2",
        cmap=cmap if color is not None else None,
        linewidth=linewidth,
        arrowsize=0.7,
    )
    axis_clean(ax)
    return sl


def draw_vector_field_3d(ax, geom: GeometryData,
                         step: int = 4,
                         length: float = 0.05,
                         color: str = "0.25",
                         linewidth: float = 0.6) -> Any:
    """Quiver-style renderer for a 3-D vector field ``(D, H, W, 3)``.

    Only a sparse subgrid is drawn (every ``step``-th cell along each
    axis) — otherwise the plot is unreadable. Intended for diagnostic
    visualization rather than production rendering.
    """
    field = np.asarray(geom.coordinates)
    if field.ndim != 4 or field.shape[-1] != 3:
        raise ValueError(
            f"draw_vector_field_3d expects (D, H, W, 3); got {field.shape}."
        )
    grid = geom.field_grid
    if grid is None:
        nz, ny, nx, _ = field.shape
        X, Y, Z = np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
        )
    else:
        X, Y, Z = grid

    s = max(int(step), 1)
    return ax.quiver(
        X[::s, ::s, ::s], Y[::s, ::s, ::s], Z[::s, ::s, ::s],
        field[::s, ::s, ::s, 0],
        field[::s, ::s, ::s, 1],
        field[::s, ::s, ::s, 2],
        length=length, color=color, linewidth=linewidth, normalize=True,
    )


def draw_image(geom: GeometryData, ax,
               cmap: str = "inferno", origin: str = "lower",
               extent: Optional[Tuple[float, float, float, float]] = None,
               vmin: float = 0.0, vmax: float = 1.0) -> Any:
    """Imshow renderer for fields stored as a (H, W) array (Julia, Cantor, ...)."""
    field = np.asarray(geom.coordinates)
    img = ax.imshow(field, origin=origin, cmap=cmap, vmin=vmin, vmax=vmax,
                    extent=extent)
    axis_clean(ax, equal=False)
    return img


def draw_tree_2d(geom: GeometryData, ax,
                 color: str = PALETTE["blue"], lw: float = 0.5,
                 alpha: float = 0.75) -> None:
    """L-system / fractal tree edges drawn as a LineCollection."""
    from matplotlib.collections import LineCollection
    coords = np.asarray(geom.coordinates)
    edges  = np.asarray(geom.edges)
    if edges.size:
        segs = coords[edges]
        ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, alpha=alpha))
    pad = 0.05 * (coords.ptp(axis=0).max() + 1e-3)
    ax.set_xlim(coords[:, 0].min() - pad, coords[:, 0].max() + pad)
    ax.set_ylim(coords[:, 1].min() - pad, coords[:, 1].max() + pad)
    axis_clean(ax)


def draw_rectangles(geom: GeometryData, ax,
                    cmap: str = "tab10",
                    edge_color: str = "white", edge_lw: float = 0.8) -> None:
    """Continued-fraction rectangle tilings — a sequence of (4, 2) polygons."""
    import matplotlib
    from matplotlib.collections import PolyCollection
    rects = list(geom.coordinates)
    cm = matplotlib.colormaps[cmap]
    n = max(len(rects), 1)
    colors = [cm(i / max(n - 1, 1)) for i in range(n)]
    polys = [r.tolist() if hasattr(r, "tolist") else list(r) for r in rects]
    ax.add_collection(PolyCollection(polys, facecolors=colors,
                                       edgecolors=edge_color, linewidths=edge_lw))
    if rects:
        all_pts = np.vstack(rects)
        ax.set_xlim(all_pts[:, 0].min() - 0.01, all_pts[:, 0].max() + 0.01)
        ax.set_ylim(all_pts[:, 1].min() - 0.01, all_pts[:, 1].max() + 0.01)
    axis_clean(ax, equal=False)


# ═══════════════════════════════════════════ 3. 3-D PRIMITIVE RENDERERS ═══════


def draw_mesh_3d(ax, geom: GeometryData,
                 color: str = PALETTE["blue"], alpha: float = 0.75,
                 lw: float = 0.2, edge_color: str = "white") -> None:
    """3-D triangle mesh via Poly3DCollection."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    V = np.asarray(geom.coordinates)
    F = np.asarray(geom.faces)
    polys = [V[tri] for tri in F]
    coll = Poly3DCollection(polys, alpha=alpha, facecolor=color,
                            edgecolor=edge_color, linewidth=lw)
    ax.add_collection3d(coll)
    ax.auto_scale_xyz(V[:, 0], V[:, 1], V[:, 2])


def draw_tree_3d(ax, geom: GeometryData,
                 color: str = PALETTE["blue"], lw: float = 0.5,
                 alpha: float = 0.7) -> None:
    """3-D L-system tree edges via Line3DCollection."""
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    V = np.asarray(geom.coordinates)
    E = np.asarray(geom.edges)
    if E.size == 0:
        return
    segs = [(V[e[0]], V[e[1]]) for e in E]
    ax.add_collection3d(Line3DCollection(segs, colors=color, linewidths=lw, alpha=alpha))
    ax.auto_scale_xyz(V[:, 0], V[:, 1], V[:, 2])


def draw_point_cloud_3d(ax, geom: GeometryData,
                        color: str = PALETTE["red"],
                        size: float = 2, alpha: float = 0.6) -> None:
    """3-D scatter."""
    V = np.asarray(geom.coordinates)
    ax.scatter(V[:, 0], V[:, 1], V[:, 2], c=color, s=size, alpha=alpha)


def draw_spherical_harmonic_field(
    ax,
    geom: GeometryData,
    cmap: str = "RdBu_r",
    radius: float = 1.0,
    alpha: float = 1.0,
    vmax: Optional[float] = None,
) -> Any:
    """Render a spherical-harmonic ``(θ, φ)`` field on the unit sphere.

    The ``coordinates`` array (shape ``(n_theta, n_phi)``) is mapped
    through ``cmap`` and painted as ``facecolors`` on a 3-D sphere
    surface plot. Designed for the output of
    :func:`.spherical_harmonics.spherical_harmonic_field`,
    :func:`.spherical_harmonics.spherical_harmonic_from_input`, and
    :func:`.spherical_harmonics.single_spherical_harmonic`.

    Parameters
    ----------
    ax : matplotlib 3-D axis
    geom : GeometryData with ``geom_type='field_2d'`` and ``field_grid``
        the ``(THETA, PHI)`` meshgrid (physics convention: θ polar,
        φ azimuthal).
    cmap : str, default='RdBu_r'
        Diverging colormap; signed fields look natural with this default.
    radius : float, default=1.0
        Sphere radius.
    alpha : float, default=1.0
    vmax : float, optional
        Override the symmetric colour limit. Defaults to the field's
        peak absolute value.
    """
    import matplotlib.pyplot as plt
    field = np.asarray(geom.coordinates)
    if np.iscomplexobj(field):
        field = field.real
    grid = geom.field_grid
    if grid is None:
        raise ValueError(
            "draw_spherical_harmonic_field expects field_grid=(THETA, PHI)."
        )
    THETA, PHI = grid

    X = radius * np.sin(THETA) * np.cos(PHI)
    Y = radius * np.sin(THETA) * np.sin(PHI)
    Z = radius * np.cos(THETA)

    if vmax is None:
        vmax = float(np.nanmax(np.abs(field))) or 1.0
    norm = np.clip((field + vmax) / (2.0 * vmax), 0.0, 1.0)
    cm = plt.get_cmap(cmap)
    facecolors = cm(norm)

    surf = ax.plot_surface(
        X, Y, Z,
        facecolors=facecolors,
        rstride=1, cstride=1,
        antialiased=False,
        shade=False,
        linewidth=0,
        alpha=alpha,
    )
    lim = radius * 1.05
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass
    return surf


def draw_spherical_harmonic_mesh(
    ax,
    geom: GeometryData,
    cmap: str = "RdBu_r",
    edge_color: Optional[str] = None,
    alpha: float = 1.0,
    lw: float = 0.0,
) -> Any:
    """Wobbled-radius spherical-harmonic mesh with face colours from weights.

    The ``geom.weights`` array is the per-vertex normalised field
    amplitude (in ``[-1, 1]``). Each triangle is coloured by the mean of
    its three vertex weights mapped through ``cmap``.

    Parameters
    ----------
    ax : matplotlib 3-D axis
    geom : GeometryData with ``geom_type='mesh_3d'``, populated ``faces``
        and ``weights`` (see
        :func:`.spherical_harmonics.spherical_harmonic_mesh`).
    cmap : str, default='RdBu_r'
    edge_color : str, optional
        Triangle outline colour. Default ``None`` (no edge outlines).
    alpha : float, default=1.0
    lw : float, default=0.0
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.pyplot as plt

    V = np.asarray(geom.coordinates)
    F = np.asarray(geom.faces)
    polys = [V[tri] for tri in F]
    if geom.weights is not None and len(geom.weights) == len(V):
        W = np.asarray(geom.weights)
        face_w = W[F].mean(axis=1)
        norm = np.clip((face_w + 1.0) / 2.0, 0.0, 1.0)
        cm = plt.get_cmap(cmap)
        facecolors = cm(norm)
        coll = Poly3DCollection(
            polys,
            alpha=alpha,
            facecolors=facecolors,
            edgecolor=edge_color or "none",
            linewidth=lw,
        )
    else:
        coll = Poly3DCollection(
            polys,
            alpha=alpha,
            facecolor=PALETTE["blue"],
            edgecolor=edge_color or "none",
            linewidth=lw,
        )
    ax.add_collection3d(coll)
    rmax = float(np.linalg.norm(V, axis=1).max())
    pad = 0.05
    ax.set_xlim(-rmax - pad, rmax + pad)
    ax.set_ylim(-rmax - pad, rmax + pad)
    ax.set_zlim(-rmax - pad, rmax + pad)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass
    return coll


def draw_interference_field_2d(
    geom: GeometryData,
    ax,
    cmap: str = "inferno",
    show_sources: bool = False,
    source_color: str = "#ff66cc",
    source_size: float = 14.0,
) -> Any:
    """Multi-source interference renderer with optional source-position dots.

    Uses a perceptual colormap on intensity / amplitude fields (no
    negative values) and a diverging cmap when the field is signed
    (``output='real'``).

    Parameters
    ----------
    geom : GeometryData with ``metadata.kind='interference_field_2d'``.
    ax : matplotlib 2-D axis
    cmap : str, default='inferno'
    show_sources : bool, default=False
        If True, plot the emitter positions as a magenta scatter on top
        of the field. Reads geometry parameters to recompute positions.
    """
    field = np.asarray(geom.coordinates, dtype=np.float64)
    grid = geom.field_grid
    if grid is None:
        raise ValueError(
            "draw_interference_field_2d expects field_grid=(X, Y)."
        )
    X, Y = grid

    has_negative = bool(field.min() < 0)
    used_cmap = "RdBu_r" if has_negative else cmap
    if has_negative:
        vmax = float(np.nanmax(np.abs(field))) or 1.0
        vmin = -vmax
    else:
        vmin = float(np.nanmin(field))
        vmax = float(np.nanmax(field)) or 1.0

    mesh = ax.pcolormesh(
        X, Y, field, cmap=used_cmap, vmin=vmin, vmax=vmax, shading="auto"
    )

    if show_sources:
        params = geom.parameters or {}
        layout = params.get("layout", "line")
        spacing = params.get("spacing", 1.0)
        n_sources = (geom.metadata or {}).get("n_sources", 0)
        if n_sources > 0:
            try:
                from biotuner.harmonic_geometry.interference_patterns import (
                    _emitter_positions,
                )
                pos = _emitter_positions(n_sources, layout, spacing)
                ax.scatter(
                    pos[:, 0], pos[:, 1],
                    c=source_color, s=source_size,
                    edgecolors="white", linewidths=0.5, zorder=3,
                )
            except Exception:
                pass

    ax.set_aspect("equal")
    return mesh


def draw_curve_3d(ax, geom: GeometryData,
                  color: str = PALETTE["blue"], lw: float = 0.6) -> None:
    """3-D polyline."""
    V = np.asarray(geom.coordinates)
    ax.plot(V[:, 0], V[:, 1], V[:, 2], color=color, lw=lw)
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass


# ═══════════════════════════════════════════════════════ 4. DISPATCHER ═════════


# Mapping from the geom_type strings used across the package to (renderer, dim).
# Each entry is `(callable, ndim)` where ndim ∈ {2, 3} indicates the axis kind.
_DISPATCH: Dict[str, Tuple[Callable[..., Any], int]] = {
    "curve_2d":         (draw_curve_2d,        2),
    "curve_3d":         (draw_curve_3d,        3),
    "polygon":          (draw_polygon,         2),
    "polygon_2d":       (draw_polygon,         2),
    "polygon_set":      (draw_polygon_set,     2),
    "graph":            (draw_graph_2d,        2),
    "tree":             (draw_tree_2d,         2),       # fallback; may be 3-D
    "rectangles":       (draw_rectangles,      2),
    "field_2d":         (draw_field_2d,        2),
    "image":            (draw_image,           2),
    "point_cloud":      (draw_point_cloud_2d,  2),
    "point_cloud_2d":   (draw_point_cloud_2d,  2),
    "point_cloud_3d":   (draw_point_cloud_3d,  3),
    "mesh_3d":          (draw_mesh_3d,         3),
    "tree_3d":          (draw_tree_3d,         3),
    "vector_field_2d":  (draw_vector_field_2d, 2),
    "vector_field_3d":  (draw_vector_field_3d, 3),
}


def _is_3d_tree(geom: GeometryData) -> bool:
    coords = np.asarray(geom.coordinates)
    return coords.ndim == 2 and coords.shape[1] == 3


def _accepts_kwarg(fn: Callable[..., Any], name: str) -> bool:
    """Cheap predicate: does ``fn`` accept ``name`` as a keyword arg?

    Used by :func:`gallery` to avoid injecting ``color=`` into drawers
    that paint via a colormap (e.g. :func:`draw_field_2d`,
    :func:`draw_polygon_set`, :func:`draw_spherical_harmonic_field`)
    and would otherwise raise ``TypeError: got unexpected keyword
    argument 'color'``.
    """
    import inspect
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return True  # be permissive for C-implemented callables
    for p in sig.parameters.values():
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if p.name == name:
            return True
    return False


def _resolve_dispatch(geom: GeometryData) -> Tuple[Callable[..., Any], int]:
    """Return the ``(drawer, ndim)`` pair that :func:`plot_geometry` will
    end up using for ``geom``. Centralised so :func:`gallery` and
    :func:`plot_geometry` always agree on the axis projection and on
    whether the drawer accepts the standard ``color`` keyword."""
    gt = geom.geom_type
    fn, ndim = _DISPATCH.get(gt, (draw_curve_2d, 2))
    if gt == "tree" and _is_3d_tree(geom):
        fn, ndim = draw_tree_3d, 3
    elif gt == "graph" and _is_3d_tree(geom):
        fn, ndim = draw_tree_3d, 3
    kind = (geom.metadata or {}).get("kind", "")
    if gt == "field_2d" and kind in (
        "spherical_harmonic_field",
        "spherical_harmonic_single",
        "spherical_harmonic_temporal",
    ):
        fn, ndim = draw_spherical_harmonic_field, 3
    elif gt == "mesh_3d" and kind == "spherical_harmonic_mesh":
        fn, ndim = draw_spherical_harmonic_mesh, 3
    elif gt == "field_2d" and kind == "interference_field_2d":
        fn, ndim = draw_interference_field_2d, 2
    return fn, ndim


def plot_geometry(geom: GeometryData, ax=None, **kwargs):
    """Auto-route a :class:`GeometryData` to the matching renderer.

    Parameters
    ----------
    geom : GeometryData
    ax : matplotlib axis, optional
        If ``None``, a new figure is created with the appropriate projection.
    **kwargs
        Forwarded to the underlying drawer.

    Returns
    -------
    (fig, ax) : tuple
    """
    import matplotlib.pyplot as plt
    gt = geom.geom_type
    if gt not in _DISPATCH:
        raise ValueError(
            f"plot_geometry: no renderer for geom_type {gt!r}. "
            f"Known types: {sorted(_DISPATCH)}."
        )
    fn, ndim = _DISPATCH[gt]
    # Special-case: 'tree' may be 3-D
    if gt == "tree" and _is_3d_tree(geom):
        fn, ndim = draw_tree_3d, 3
    if gt == "graph" and _is_3d_tree(geom):
        # graphs in 3-D fall back to a 3-D line collection
        fn, ndim = draw_tree_3d, 3
    # Spherical-harmonic outputs share geom_types with generic field/mesh
    # data but want a domain-specific renderer (sphere surface or weight-
    # coloured wobbled mesh). Detect via metadata.kind.
    kind = (geom.metadata or {}).get("kind", "")
    if (
        gt == "field_2d"
        and kind
        in (
            "spherical_harmonic_field",
            "spherical_harmonic_single",
            "spherical_harmonic_temporal",
        )
    ):
        fn, ndim = draw_spherical_harmonic_field, 3
    elif gt == "mesh_3d" and kind == "spherical_harmonic_mesh":
        fn, ndim = draw_spherical_harmonic_mesh, 3
    # Open-medium multi-source interference field — perceptual cmap +
    # optional emitter dots.
    elif gt == "field_2d" and kind == "interference_field_2d":
        fn, ndim = draw_interference_field_2d, 2
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        if ndim == 3:
            ax = fig.add_subplot(111, projection="3d")
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            try:
                ax.set_box_aspect([1, 1, 1])
            except Exception:
                pass
        else:
            ax = fig.add_subplot(111)
    else:
        fig = ax.figure
    if ndim == 3:
        fn(ax, geom, **kwargs)
    else:
        fn(geom, ax, **kwargs)
    return fig, ax


# ═══════════════════════════════════════════════════ 5. LAYOUT HELPERS ════════


def gallery(geometries: Sequence[GeometryData],
            titles: Optional[Sequence[str]] = None,
            n_cols: int = 3,
            fig_width: float = _DEFAULT_FIG_WIDTH,
            row_height: Optional[float] = None,
            color: Optional[Union[str, Sequence[str]]] = None,
            suptitle: Optional[str] = None,
            draw_kwargs: Optional[Dict[str, Any]] = None,
            renderer: Optional[Callable] = None):
    """Render an N × n_cols grid of geometries via :func:`plot_geometry`.

    Use ``renderer`` to force a specific drawer (otherwise dispatched per-geom).
    ``color`` may be a single string (applied to all) or a list.
    Returns ``(fig, axes)``.
    """
    import matplotlib.pyplot as plt
    n = len(geometries)
    if n == 0:
        raise ValueError("gallery: at least one geometry required.")
    n_rows = math.ceil(n / n_cols)
    rh = row_height if row_height is not None else fig_width / n_cols
    # Determine projection mix
    needs_3d = any(_DISPATCH.get(g.geom_type, (None, 2))[1] == 3 or
                   ((g.geom_type in ("tree", "graph")) and _is_3d_tree(g))
                   for g in geometries)
    fig = plt.figure(figsize=(fig_width, rh * n_rows))
    axes: List[Any] = []
    colors_arr = ([color] * n if isinstance(color, str)
                  else list(color) if color is not None else CHORD_COLORS)
    draw_kwargs = dict(draw_kwargs or {})
    for k, g in enumerate(geometries):
        # Use the unified dispatch so the axis projection matches the
        # drawer plot_geometry will actually call (e.g. spherical-harmonic
        # fields render onto a sphere and need a 3-D axis even though the
        # underlying geom_type is 'field_2d').
        chosen_drawer, ndim = _resolve_dispatch(g)
        is3 = ndim == 3
        if is3:
            ax = fig.add_subplot(n_rows, n_cols, k + 1, projection="3d")
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            try:
                ax.set_box_aspect([1, 1, 1])
            except Exception:
                pass
        else:
            ax = fig.add_subplot(n_rows, n_cols, k + 1)
        kw = dict(draw_kwargs)
        # Only inject `color=` if the drawer accepts it; field-style
        # drawers (chladni, interference, spherical-harmonic, polygon_set)
        # paint via a colormap and would raise on color=.
        target_drawer = renderer if renderer is not None else chosen_drawer
        if "color" not in kw and _accepts_kwarg(target_drawer, "color"):
            kw["color"] = colors_arr[k % len(colors_arr)]
        if renderer is not None:
            if is3:
                renderer(ax, g, **kw)
            else:
                renderer(g, ax, **kw)
        else:
            plot_geometry(g, ax=ax, **kw)
        if titles and k < len(titles):
            title_ax(ax, titles[k])
        axes.append(ax)
    if suptitle:
        fig.suptitle(suptitle, fontsize=10, y=1.01)
    fig.tight_layout()
    return fig, axes


def sweep_strip(geometries: Sequence[GeometryData],
                labels: Optional[Sequence[str]] = None,
                fig_width: float = _DEFAULT_FIG_WIDTH,
                row_height: Optional[float] = None,
                color: str = PALETTE["blue"],
                suptitle: Optional[str] = None,
                draw_kwargs: Optional[Dict[str, Any]] = None,
                renderer: Optional[Callable] = None):
    """1×N strip — convenience wrapper over :func:`gallery` with n_cols=N."""
    return gallery(
        geometries, titles=labels, n_cols=len(geometries),
        fig_width=fig_width, row_height=row_height,
        color=color, suptitle=suptitle, draw_kwargs=draw_kwargs,
        renderer=renderer,
    )


# ═══════════════════════════════════════════════════ 6. ANIMATION HELPERS ═════


def rotation_strip(geom: GeometryData,
                   n_strip: int = 6,
                   fig_width: float = _DEFAULT_FIG_WIDTH,
                   elev: float = 20.0,
                   color: str = PALETTE["blue"],
                   draw_kwargs: Optional[Dict[str, Any]] = None,
                   renderer: Optional[Callable] = None,
                   suptitle: Optional[str] = None):
    """1×N strip of the same 3-D geometry at evenly spaced azimuths.

    Returns ``(fig, axes)``. Useful for showing a tube/knot/mesh from
    several viewing angles in a single static figure.
    """
    import matplotlib.pyplot as plt
    draw_kwargs = dict(draw_kwargs or {})
    if "color" not in draw_kwargs:
        draw_kwargs["color"] = color
    if renderer is None:
        renderer = _DISPATCH.get(geom.geom_type, (draw_mesh_3d, 3))[0]

    fig, axes = plt.subplots(
        1, n_strip,
        figsize=(fig_width, fig_width / n_strip * 1.1),
        subplot_kw={"projection": "3d"},
    )
    if n_strip == 1:
        axes = [axes]
    for k, ax in enumerate(axes):
        azim = k * (360 // n_strip)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass
        renderer(ax, geom, **draw_kwargs)
        ax.set_title(f"{azim}°", fontsize=6)
    if suptitle:
        fig.suptitle(suptitle, fontsize=10, y=1.04)
    fig.tight_layout()
    return fig, axes


def animate_rotation(geom: GeometryData,
                     out_path: Union[str, Path],
                     n_frames: int = 36,
                     fps: int = 12,
                     elev: float = 20.0,
                     fig_size: Tuple[float, float] = (3.5, 3.5),
                     dpi: int = 100,
                     color: str = PALETTE["blue"],
                     draw_kwargs: Optional[Dict[str, Any]] = None,
                     renderer: Optional[Callable] = None) -> Path:
    """Save a rotating GIF of a 3-D :class:`GeometryData` (mesh / tree / cloud).

    Frames sweep azimuth from 0° to (n_frames−1) × (360 / n_frames).
    ``renderer`` defaults to :func:`plot_geometry`'s 3-D dispatch.
    """
    import matplotlib.pyplot as plt
    from matplotlib import animation
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    draw_kwargs = dict(draw_kwargs or {})
    if "color" not in draw_kwargs:
        draw_kwargs["color"] = color

    fig = plt.figure(figsize=fig_size, facecolor=_DEFAULT_FACECOLOR)
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    try:
        ax.set_box_aspect([1, 1, 1])
    except Exception:
        pass

    fn = renderer
    if fn is None:
        gt = geom.geom_type
        fn = _DISPATCH.get(gt, (draw_mesh_3d, 3))[0]

    azim_step = 360.0 / max(n_frames, 1)

    def update(frame):
        ax.clear()
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass
        ax.view_init(elev=elev, azim=frame * azim_step)
        fn(ax, geom, **draw_kwargs)
        return []

    ani = animation.FuncAnimation(fig, update, frames=n_frames, blit=False)
    ani.save(str(out_path), writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return out_path


def animate_geometry_sequence(geoms: Sequence[GeometryData],
                              out_path: Union[str, Path],
                              fps: int = 8,
                              fig_size: Tuple[float, float] = (4.5, 4.5),
                              dpi: int = 100,
                              color: str = PALETTE["blue"],
                              draw_kwargs: Optional[Dict[str, Any]] = None,
                              elev: float = 25.0,
                              azim: float = 35.0) -> Path:
    """Save a GIF cycling through a list of geometries (2-D or 3-D)."""
    import matplotlib.pyplot as plt
    from matplotlib import animation
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not geoms:
        raise ValueError("animate_geometry_sequence: empty geoms list.")
    draw_kwargs = dict(draw_kwargs or {})
    if "color" not in draw_kwargs:
        draw_kwargs["color"] = color

    is3 = _DISPATCH.get(geoms[0].geom_type, (None, 2))[1] == 3 or _is_3d_tree(geoms[0])
    fig = plt.figure(figsize=fig_size, facecolor=_DEFAULT_FACECOLOR)
    if is3:
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = fig.add_subplot(111)

    def update(k):
        ax.clear()
        ax.set_xticks([]); ax.set_yticks([])
        if is3:
            ax.set_zticks([])
            try:
                ax.set_box_aspect([1, 1, 1])
            except Exception:
                pass
            ax.view_init(elev=elev, azim=azim)
        plot_geometry(geoms[k], ax=ax, **draw_kwargs)
        ax.set_title(f"frame {k}", fontsize=8)
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(geoms), blit=False)
    ani.save(str(out_path), writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return out_path


# ═══════════════════════════════════════════ 7. METRICS PLOTTING ══════════════


def plot_metric_radar(rows,
                      labels: Optional[Sequence[str]] = None,
                      metrics: Optional[Sequence[str]] = None,
                      ax=None,
                      colors: Optional[Sequence[str]] = None,
                      fill_alpha: float = 0.20,
                      line_lw: float = 1.4,
                      title: Optional[str] = None):
    """Radar chart of normalised metrics for one or several rows.

    Parameters
    ----------
    rows : dict or list of dicts
        Either a single ``{metric: value}`` dict (one polygon) or a list of
        such dicts (multi-overlay). Values are auto-normalised to [0, 1] via
        :func:`biotuner.harmonic_geometry.metrics.normalize_metrics`.
    labels : sequence of str, optional
        One label per row. Defaults to ``row_0, row_1, …``.
    metrics : sequence of str, optional
        Subset of metric keys to include. Defaults to the keys of the first
        row (with ``n_components`` excluded — it's a count, not a metric).
    ax : polar matplotlib axis, optional
    colors : sequence of str, optional
    fill_alpha, line_lw : floats
    title : str, optional

    Returns
    -------
    (fig, ax) : tuple
    """
    import matplotlib.pyplot as plt
    from biotuner.harmonic_geometry.metrics import normalize_metrics

    if isinstance(rows, dict):
        rows_list = [rows]
    else:
        rows_list = list(rows)
    if not rows_list:
        raise ValueError("plot_metric_radar: no rows provided.")

    keys = list(metrics) if metrics else [
        k for k in rows_list[0].keys() if k != "n_components"
    ]
    if len(keys) < 3:
        raise ValueError(
            f"plot_metric_radar: need at least 3 metrics for a meaningful "
            f"radar, got {keys}."
        )

    norm_rows = normalize_metrics(rows_list, metrics=keys)

    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax  = fig.add_subplot(111, projection="polar")
    else:
        fig = ax.figure

    n = len(keys)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])

    palette = list(colors) if colors else CHORD_COLORS
    labels = list(labels) if labels else [f"row_{i}" for i in range(len(norm_rows))]

    for i, (row, lbl) in enumerate(zip(norm_rows, labels)):
        vals = np.array([
            row.get(k, 0.0) if math.isfinite(row.get(k, float("nan"))) else 0.0
            for k in keys
        ])
        vals_closed = np.concatenate([vals, vals[:1]])
        c = palette[i % len(palette)]
        ax.plot(angles_closed, vals_closed, color=c, lw=line_lw, label=lbl)
        ax.fill(angles_closed, vals_closed, color=c, alpha=fill_alpha)

    ax.set_xticks(angles)
    ax.set_xticklabels([k.replace("_", "\n") for k in keys], fontsize=7)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75])
    ax.set_yticklabels(["0.25", "0.50", "0.75"], fontsize=6, color="#888")
    ax.grid(alpha=0.3, lw=0.5)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=10, pad=14)
    if len(norm_rows) > 1:
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10),
                   fontsize=7, frameon=False)
    return fig, ax


def plot_metric_trajectory(metrics_dict_or_seq,
                           generator: Optional[Any] = None,
                           generator_kwargs: Optional[Dict[str, Any]] = None,
                           metrics: Optional[Sequence[str]] = None,
                           times: Optional[np.ndarray] = None,
                           ax=None,
                           normalize: bool = False,
                           colors: Optional[Sequence[str]] = None,
                           title: Optional[str] = None):
    """Line plot of geometry-metric trajectories over time.

    Two ways to pass data:

    * **Pre-computed**: ``metrics_dict_or_seq`` is a dict
      ``{name: 1-D ndarray of length T}`` — typically the output of
      :func:`sequence_metrics`. ``generator`` should be ``None``.
    * **From a HarmonicSequence**: pass the sequence as the first argument
      and supply ``generator`` (e.g. ``harmonic_knot``). The function
      computes :func:`sequence_metrics(seq, generator, **kwargs)` internally.

    Parameters
    ----------
    metrics_dict_or_seq : dict[str, ndarray] or HarmonicSequence
    generator : callable, optional
        Required when the first argument is a :class:`HarmonicSequence`.
    generator_kwargs : dict, optional
        Forwarded to ``generator`` for every frame.
    metrics : sequence of str, optional
        Subset of metric trajectories to plot. Defaults to all keys.
    times : ndarray, optional
        Explicit time axis. Inferred from the sequence if available.
    ax : matplotlib axis, optional
    normalize : bool, default False
        Per-metric min-max scaling to [0, 1].

    Returns
    -------
    (fig, ax) : tuple
    """
    import matplotlib.pyplot as plt

    if hasattr(metrics_dict_or_seq, "frames"):
        if generator is None:
            raise ValueError(
                "plot_metric_trajectory: a HarmonicSequence requires a "
                "generator argument (e.g. harmonic_knot). For a "
                "pre-computed metric dict, pass it directly without "
                "a generator."
            )
        from biotuner.harmonic_geometry.metrics import sequence_metrics
        seq = metrics_dict_or_seq
        m = sequence_metrics(seq, generator, **(generator_kwargs or {}))
        if times is None and seq.times is not None:
            times = np.asarray(seq.times, dtype=np.float64)
    else:
        m = dict(metrics_dict_or_seq)

    keys = list(metrics) if metrics else list(m.keys())
    keys = [k for k in keys if k in m]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    palette = list(colors) if colors else CHORD_COLORS
    if times is None and keys:
        times = np.arange(len(m[keys[0]]), dtype=np.float64)

    for i, k in enumerate(keys):
        y = np.asarray(m[k], dtype=np.float64)
        label = k
        if normalize:
            finite = y[np.isfinite(y)]
            if finite.size == 0:
                continue
            if finite.ptp() > 0:
                y = (y - finite.min()) / finite.ptp()
            else:
                # Zero-variance metric under normalize=True → render at 0.5
                # (matches normalize_metrics' "consensus" semantics) and tag
                # the legend so the reader knows the trajectory was constant.
                y = np.full_like(y, 0.5)
                label = f"{k}  (constant)"
        ax.plot(times, y, color=palette[i % len(palette)], lw=1.4, label=label)
    if normalize:
        ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("frame" if normalize else "time", fontsize=8)
    ax.set_ylabel("normalised value" if normalize else "metric value",
                  fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(loc="best", fontsize=7, frameon=False)
    ax.grid(alpha=0.25, linestyle=":")
    return fig, ax


# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # palette + helpers
    "PALETTE", "CHORD_COLORS",
    "axis_clean", "title_ax", "make_axis_3d", "save_figure",
    # 2-D primitives
    "draw_curve_2d", "draw_polygon", "draw_polygon_set",
    "draw_graph_2d", "draw_hyperbolic_graph",
    "draw_point_cloud_2d", "draw_field_2d", "draw_image",
    "draw_tree_2d", "draw_rectangles",
    # 3-D primitives
    "draw_mesh_3d", "draw_tree_3d", "draw_point_cloud_3d", "draw_curve_3d",
    # dispatch
    "plot_geometry",
    # layouts
    "gallery", "sweep_strip", "rotation_strip",
    # animations
    "animate_rotation", "animate_geometry_sequence",
    # metrics
    "plot_metric_radar", "plot_metric_trajectory",
]
