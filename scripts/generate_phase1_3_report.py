"""
Generate the Phase 1-3 report PDF for biotuner.harmonic_geometry.

Produces docs/reports/harmonic_geometry_phase1_3.pdf with one section per
phase, sample figures rendered via matplotlib, and a numerical validation
summary at the end.

Run with the `biotuner` conda env:

    python docs/reports/generate_phase1_3_report.py
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Callable, List

# Prefer the worktree's biotuner over any installed copy.
_WORKTREE_ROOT = str(Path(__file__).resolve().parents[1])
if _WORKTREE_ROOT not in sys.path:
    sys.path.insert(0, _WORKTREE_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from biotuner.harmonic_geometry import (
    GeometryData,
    HarmonicInput,
    HarmonicSequence,
    chladni_field_3d_box,
    chladni_field_circular,
    chladni_field_polygon,
    chladni_field_rectangular,
    chladni_from_input,
    epicycloid,
    harmonograph_3d,
    harmonograph_lateral,
    harmonograph_rotary,
    hypocycloid,
    lissajous_2d,
    lissajous_3d,
    lissajous_compound,
    lissajous_pairwise_grid,
    lissajous_phase_drift,
    lissajous_topology,
    ratios_to_modes,
    rose_curve,
    star_polygon,
    times_table_circle,
    tuning_circle,
)

# --------------------------------------------------------------- configuration

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "reports"
FIG_DIR = OUT_DIR / "figures"
PDF_PATH = OUT_DIR / "harmonic_geometry_phase1_3.pdf"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Fixed figure size for consistency. Width matches one column at 7" with 0.75" margins.
FIG_WIDTH = 6.5
DPI = 150


# ----------------------------------------------------------- figure rendering


from biotuner.harmonic_geometry import plotting

def _save_fig(fig: plt.Figure, name: str) -> Path:
    return plotting.save_figure(fig, FIG_DIR / f"{name}.png", dpi=DPI)


def _setup_2d_ax(ax, equal: bool = True, grid: bool = True) -> None:
    plotting.axis_clean(ax, equal=equal, grid=grid, spine_color="#444444")


# Thin wrappers preserving local function names while delegating to plotting.
def render_curve_2d(geom: GeometryData, ax, color: str = "#1f3b73", lw: float = 0.9) -> None:
    plotting.draw_curve_2d(geom, ax, color=color, lw=lw)
    _setup_2d_ax(ax)


def render_curve_3d(geom: GeometryData, ax, color: str = "#1f3b73", lw: float = 0.6) -> None:
    plotting.draw_curve_3d(ax, geom, color=color, lw=lw)


def render_polygon(geom: GeometryData, ax, color: str = "#a23e2c", lw: float = 1.4) -> None:
    plotting.draw_polygon(geom, ax, color=color, lw=lw)
    _setup_2d_ax(ax)


def render_polygon_set(geom: GeometryData, ax) -> None:
    plotting.draw_polygon_set(geom, ax)
    _setup_2d_ax(ax)


def render_graph(geom: GeometryData, ax) -> None:
    plotting.draw_graph_2d(geom, ax,
                            edge_color="#1f3b73", edge_alpha=0.55,
                            node_color="#a23e2c", node_size=4)
    _setup_2d_ax(ax)


def render_point_cloud(geom: GeometryData, ax) -> None:
    plotting.draw_point_cloud_2d(geom, ax, size=30, ref_circle=True,
                                   color="#a23e2c", edge_color="#1f3b73")
    _setup_2d_ax(ax)


def render_field_2d(
    geom: GeometryData,
    ax,
    cmap: str = "RdBu_r",
    show_nodal: bool = True,
) -> None:
    plotting.draw_field_2d(geom, ax, cmap=cmap, show_nodal=show_nodal, signed=True)
    _setup_2d_ax(ax)


# --------------------------------------------------------------- figure builders


def fig_lissajous_gallery() -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(FIG_WIDTH, FIG_WIDTH * 2 / 3))
    cases = [
        (1, np.pi / 2, "1:1, φ=π/2 (unit circle)"),
        (Fraction(3, 2), np.pi / 2, "3:2, φ=π/2"),
        (Fraction(5, 4), 0.0, "5:4, φ=0"),
        (Fraction(7, 4), np.pi / 4, "7:4, φ=π/4"),
        (Fraction(5, 3), np.pi / 3, "5:3, φ=π/3"),
        (Fraction(9, 7), np.pi / 6, "9:7, φ=π/6"),
    ]
    for ax, (ratio, phase, label) in zip(axes.ravel(), cases):
        g = lissajous_2d(ratio, phase=phase, n_points=2000)
        render_curve_2d(g, ax)
        ax.set_title(label, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Lissajous curves: lissajous_2d", fontsize=11)
    fig.tight_layout()
    return _save_fig(fig, "lissajous_gallery")


def fig_lissajous_3d_knot() -> Path:
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH * 0.45))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    g1 = lissajous_3d(ratios=[3, 4, 5], phases=[0.0, np.pi / 4, np.pi / 2], n_points=4000)
    render_curve_3d(g1, ax1)
    ax1.set_title("Lissajous knot (3, 4, 5)", fontsize=9)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    g2 = lissajous_3d(ratios=[2, 3, 7], phases=[0.0, np.pi / 3, np.pi / 5], n_points=4000)
    render_curve_3d(g2, ax2)
    ax2.set_title("Lissajous knot (2, 3, 7)", fontsize=9)
    fig.tight_layout()
    return _save_fig(fig, "lissajous_3d_knot")


def fig_lissajous_pairwise_grid() -> Path:
    inp = HarmonicInput(ratios=[1, Fraction(3, 2), Fraction(5, 4)], base_freq=100.0)
    grid = lissajous_pairwise_grid(inp, n_points=400)
    n = len(grid)
    fig, axes = plt.subplots(n, n, figsize=(FIG_WIDTH * 0.7, FIG_WIDTH * 0.7))
    labels = ["1/1", "3/2", "5/4"]
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            render_curve_2d(grid[i][j], ax, lw=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(labels[j], fontsize=8)
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=8)
    fig.suptitle("lissajous_pairwise_grid (3-component input)", fontsize=10)
    fig.tight_layout()
    return _save_fig(fig, "lissajous_pairwise_grid")


def fig_lissajous_phase_drift() -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, FIG_WIDTH / 3))
    drifts = [0.5, 1.5, 4.0]
    for ax, drift in zip(axes, drifts):
        g = lissajous_phase_drift(
            ratio=Fraction(3, 2), drift_rate=drift, duration=4.0, sr=600
        )
        render_curve_2d(g, ax, lw=0.5)
        ax.set_title(f"drift_rate = {drift} rad/s", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("lissajous_phase_drift, ratio 3:2 over 4 s", fontsize=10)
    fig.tight_layout()
    return _save_fig(fig, "lissajous_phase_drift")


def fig_lissajous_compound() -> Path:
    inp = HarmonicInput(
        ratios=[1, Fraction(3, 2), Fraction(5, 4), Fraction(7, 4)],
        amplitudes=[1.0, 0.7, 0.5, 0.3],
        base_freq=100.0,
    )
    g = lissajous_compound(inp, n_points=4000, n_periods=2)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 0.55, FIG_WIDTH * 0.55))
    render_curve_2d(g, ax, lw=0.6)
    ax.set_title("lissajous_compound (just-intonation tetrad)", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return _save_fig(fig, "lissajous_compound")


def fig_harmonograph_examples() -> Path:
    inp = HarmonicInput(
        peaks=[2.01, 3.02, 5.0, 7.03],
        amplitudes=[1.0, 0.8, 0.6, 0.4],
        phases=[0.0, np.pi / 5, np.pi / 3, np.pi / 7],
        damping=[0.05, 0.04, 0.06, 0.05],
    )
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, FIG_WIDTH / 3))
    g_lat = harmonograph_lateral(inp, duration=40.0, sr=400)
    render_curve_2d(g_lat, axes[0], lw=0.4)
    axes[0].set_title("harmonograph_lateral", fontsize=8)
    g_rot = harmonograph_rotary(inp, duration=40.0, sr=400, rotation_freq=0.05)
    render_curve_2d(g_rot, axes[1], lw=0.4)
    axes[1].set_title("harmonograph_rotary (Ω = 0.05 Hz)", fontsize=8)
    for ax in axes[:2]:
        ax.set_xticks([])
        ax.set_yticks([])

    axes[2].remove()
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")
    g3d = harmonograph_3d(inp, duration=40.0, sr=400)
    render_curve_3d(g3d, ax3, lw=0.5)
    ax3.set_title("harmonograph_3d", fontsize=8)
    fig.tight_layout()
    return _save_fig(fig, "harmonograph_examples")


def fig_harmonograph_damping() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_WIDTH / 2.5))
    inp_zero = HarmonicInput(
        peaks=[2.0, 3.0, 5.0, 7.0],
        amplitudes=[1.0, 0.8, 0.6, 0.4],
        damping=[0.0] * 4,
    )
    inp_decay = HarmonicInput(
        peaks=[2.0, 3.0, 5.0, 7.0],
        amplitudes=[1.0, 0.8, 0.6, 0.4],
        damping=[0.15] * 4,
    )
    for ax, inp, label in zip(
        axes,
        [inp_zero, inp_decay],
        ["damping = 0 (bounded, persistent)", "damping = 0.15 (decaying)"],
    ):
        g = harmonograph_lateral(inp, duration=30.0, sr=300)
        render_curve_2d(g, ax, lw=0.4)
        ax.set_title(label, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return _save_fig(fig, "harmonograph_damping")


def fig_star_polygons() -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(FIG_WIDTH, FIG_WIDTH / 4))
    cases = [(5, 2, "{5/2} pentagram"),
             (7, 3, "{7/3} heptagram"),
             (8, 3, "{8/3} octagram"),
             (6, 2, "{6/2} compound")]
    for ax, (n, k, label) in zip(axes, cases):
        g = star_polygon(n, k, radius=1.0)
        if g.geom_type == "polygon":
            render_polygon(g, ax)
        else:
            render_polygon_set(g, ax)
        ax.set_title(label, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
    fig.tight_layout()
    return _save_fig(fig, "star_polygons")


def fig_times_table_circles() -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(FIG_WIDTH, FIG_WIDTH / 4))
    cases = [(200, 2), (200, 3), (200, 5), (300, 7.5)]
    for ax, (n, m) in zip(axes, cases):
        g = times_table_circle(n_points=n, multiplier=m, radius=1.0)
        render_graph(g, ax)
        ax.set_title(f"n={n}, ×{m}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return _save_fig(fig, "times_table_circles")


def fig_tuning_circle() -> Path:
    inp = HarmonicInput(
        ratios=[
            Fraction(1, 1),
            Fraction(9, 8),
            Fraction(5, 4),
            Fraction(4, 3),
            Fraction(3, 2),
            Fraction(5, 3),
            Fraction(15, 8),
        ],
        amplitudes=[1.0, 0.6, 0.9, 0.7, 1.0, 0.7, 0.5],
    )
    g = tuning_circle(inp, radius=1.0)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 0.5, FIG_WIDTH * 0.5))
    render_point_cloud(g, ax)
    coords = np.asarray(g.coordinates)
    labels = ["1/1", "9/8", "5/4", "4/3", "3/2", "5/3", "15/8"]
    for (x, y), lab in zip(coords, labels):
        ax.annotate(lab, (x, y), xytext=(8, 8), textcoords="offset points", fontsize=8)
    ax.set_title("tuning_circle (just-intonation diatonic)", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return _save_fig(fig, "tuning_circle")


def fig_rose_curves() -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(FIG_WIDTH, FIG_WIDTH / 4))
    cases = [Fraction(3, 1), Fraction(5, 1), Fraction(2, 1), Fraction(7, 3)]
    for ax, ratio in zip(axes, cases):
        g = rose_curve(ratio, n_points=2000)
        render_curve_2d(g, ax, lw=0.7)
        ax.set_title(f"r = cos({ratio.numerator}/{ratio.denominator} θ)", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return _save_fig(fig, "rose_curves")


def fig_cycloids() -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(FIG_WIDTH, FIG_WIDTH / 4))
    cases = [
        (epicycloid, Fraction(3, 1), "epicycloid 3:1"),
        (epicycloid, Fraction(5, 2), "epicycloid 5:2"),
        (hypocycloid, Fraction(4, 1), "hypocycloid 4:1 (astroid)"),
        (hypocycloid, Fraction(5, 2), "hypocycloid 5:2"),
    ]
    for ax, (fn, ratio, label) in zip(axes, cases):
        g = fn(ratio, n_points=2000)
        render_curve_2d(g, ax, lw=0.8)
        ax.set_title(label, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return _save_fig(fig, "cycloids")


def fig_chladni_rectangular() -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, FIG_WIDTH / 3))
    cases = [(2, 3), (4, 5), (3, 3)]
    for ax, (m, n) in zip(axes, cases):
        g = chladni_field_rectangular([(m, n)], resolution=257)
        render_field_2d(g, ax)
        ax.set_title(f"mode ({m}, {n})", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("chladni_field_rectangular — pure modes", fontsize=10)
    fig.tight_layout()
    return _save_fig(fig, "chladni_rectangular")


def fig_chladni_rect_sum() -> Path:
    g = chladni_field_rectangular(
        modes=[(2, 3), (3, 5), (4, 1)],
        amps=[1.0, 0.6, 0.4],
        phases=[0.0, np.pi / 3, np.pi / 7],
        resolution=257,
    )
    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 0.55, FIG_WIDTH * 0.55))
    render_field_2d(g, ax)
    ax.set_title("Sum of three rectangular modes", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return _save_fig(fig, "chladni_rect_sum")


def fig_chladni_circular() -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(FIG_WIDTH, FIG_WIDTH / 4))
    cases = [
        ([1], [0], "(m=1, n=0)"),
        ([2], [1], "(m=2, n=1)"),
        ([1], [3], "(m=1, n=3)"),
        ([3], [2], "(m=3, n=2)"),
    ]
    for ax, (mr, ma, label) in zip(axes, cases):
        g = chladni_field_circular(mr, ma, R=1.0, resolution=257)
        render_field_2d(g, ax)
        ax.set_title(label, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("chladni_field_circular — pure modes", fontsize=10)
    fig.tight_layout()
    return _save_fig(fig, "chladni_circular")


def fig_chladni_polygon() -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(FIG_WIDTH, FIG_WIDTH * 0.7))
    cases = [
        (3, 0), (3, 3), (3, 6),
        (5, 0), (5, 2), (5, 5),
    ]
    for ax, (n_sides, mode_idx) in zip(axes.ravel(), cases):
        g = chladni_field_polygon([mode_idx], n_sides=n_sides, resolution=96)
        render_field_2d(g, ax, show_nodal=True)
        ax.set_title(f"{n_sides}-gon, mode {mode_idx}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("chladni_field_polygon (FDM)", fontsize=10)
    fig.tight_layout()
    return _save_fig(fig, "chladni_polygon")


def fig_chladni_3d_box() -> Path:
    """Nodal surfaces of a (2, 3, 4) standing wave + a fundamental-mode comparison.

    Slices through a 3-D Chladni field always look like 2-D Chladni; the
    real beauty is the zero-level surface where the wave is silent in 3-D.
    We render two cases side-by-side using marching-cubes (via scikit-image)
    and fall back to slices only if extraction fails.
    """
    cases = [
        ([(2, 3, 4)],            "single mode (2, 3, 4)"),
        ([(2, 1, 1), (1, 2, 1), (1, 1, 2)], "superposition: 3 fundamentals"),
    ]
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH * 0.45))
    for k, (modes, label) in enumerate(cases):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        g = chladni_field_3d_box(modes, resolution=48)
        try:
            from biotuner.harmonic_geometry.chladni import chladni_nodal_surfaces
            mesh = chladni_nodal_surfaces(g, threshold=0.0)
            plotting.draw_mesh_3d(ax, mesh,
                                   color=plotting.PALETTE["blue"] if k == 0
                                                     else plotting.PALETTE["red"],
                                   alpha=0.55, lw=0.0,
                                   edge_color="none")
            ax.set_title(f"{label}\n{len(mesh.faces)} triangles",
                         fontsize=8, pad=2)
        except Exception as exc:
            # Fallback to a single mid-slice if marching cubes fails.
            field = np.asarray(g.coordinates)
            mid = field.shape[2] // 2
            X, Y, _ = g.field_grid
            ax2 = fig.add_subplot(1, 2, k + 1)
            ax2.pcolormesh(X[:, :, mid], Y[:, :, mid], field[:, :, mid],
                           cmap="RdBu_r", shading="auto")
            ax2.set_title(f"{label}\n(fallback slice — {exc})", fontsize=7)
            ax2.set_aspect("equal"); ax2.set_xticks([]); ax2.set_yticks([])
            continue
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass
        ax.view_init(elev=22, azim=35 + k * 30)
    fig.suptitle("chladni_nodal_surfaces — 3-D zero-level surfaces "
                 "(marching cubes via scikit-image)", fontsize=10, y=1.02)
    fig.tight_layout()
    return _save_fig(fig, "chladni_3d_nodal")


def fig_chladni_rect_zoom_insets() -> Path:
    """Zoom insets on a 7-mode rectangular superposition.

    The base panel shows the full field; three insets reveal sub-region
    structure that's invisible at full-plate scale (high-frequency
    interference between dense modes).
    """
    g = chladni_field_rectangular(
        modes=[(2, 3), (3, 5), (4, 1), (5, 7), (6, 2), (1, 6), (7, 3)],
        amps=[1.0, 0.8, 0.6, 0.5, 0.45, 0.4, 0.35],
        phases=np.linspace(0, 2 * np.pi, 7, endpoint=False).tolist(),
        resolution=513,
    )
    field = np.asarray(g.coordinates)
    X, Y = g.field_grid

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH * 0.55))
    gs   = fig.add_gridspec(2, 4, width_ratios=[2, 1, 1, 1])

    ax_main = fig.add_subplot(gs[:, 0])
    vmax = float(np.nanmax(np.abs(field))) or 1.0
    ax_main.pcolormesh(X, Y, field, cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, shading="auto")
    ax_main.set_aspect("equal"); ax_main.set_xticks([]); ax_main.set_yticks([])
    ax_main.set_title("Full plate — 7 superposed modes", fontsize=9)

    insets = [
        ((0.05, 0.30, 0.05, 0.30), "lower-left"),
        ((0.40, 0.65, 0.40, 0.65), "centre"),
        ((0.65, 0.95, 0.65, 0.95), "upper-right"),
    ]
    for k, (rect, label) in enumerate(insets):
        x0, x1, y0, y1 = rect
        ax_main.add_patch(plt.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            edgecolor="#222", lw=0.8, fill=False, zorder=3,
        ))
        ax_main.text(x0, y1 + 0.01, str(k + 1), color="#222",
                     fontsize=8, ha="left", va="bottom")

        ax = fig.add_subplot(gs[k // 2, 1 + (k % 2)] if k < 2 else gs[1, 3])
        # mask grid to rect
        mask_x = (X >= x0) & (X <= x1)
        mask_y = (Y >= y0) & (Y <= y1)
        mask = mask_x & mask_y
        # Get bounding indices
        ix0 = int(np.argmax(X[0] >= x0)); ix1 = int(np.argmin(X[0] < x1))
        iy0 = int(np.argmax(Y[:, 0] >= y0)); iy1 = int(np.argmin(Y[:, 0] < y1))
        sub_field = field[iy0:iy1, ix0:ix1]
        sub_X = X[iy0:iy1, ix0:ix1]; sub_Y = Y[iy0:iy1, ix0:ix1]
        if sub_field.size:
            ax.pcolormesh(sub_X, sub_Y, sub_field, cmap="RdBu_r",
                          vmin=-vmax, vmax=vmax, shading="auto")
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{k + 1}. {label}", fontsize=7, pad=2)
    fig.suptitle("chladni_field_rectangular — zoom insets on dense superposition",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    return _save_fig(fig, "chladni_rect_insets")


def fig_chladni_from_input_strategies() -> Path:
    """Compare the four mode-mapping strategies on a chord that exposes them.

    The simple JI chord 1:5/4:3/2 produces identical (m, n) pairs across
    all strategies, so the comparison is uninformative.  Here we use an
    11-limit chord (1, 11/8, 13/9, 17/12, 7/4) where each strategy
    converges to distinct modes — and at moderate `max_mode` the differences
    between approximations dominate the visual signature.
    """
    inp = HarmonicInput(
        ratios=[Fraction(1), Fraction(11, 8), Fraction(13, 9),
                Fraction(17, 12), Fraction(7, 4)],
        amplitudes=[1.0, 0.85, 0.75, 0.65, 0.55],
        phases=[0.0, np.pi / 5, np.pi / 3, np.pi / 7, np.pi / 11],
    )
    strategies = ["stern_brocot", "continued_fraction", "rounded", "best_simple"]
    fig, axes = plt.subplots(1, 4, figsize=(FIG_WIDTH, FIG_WIDTH / 4))
    from biotuner.harmonic_geometry.chladni import ratios_to_modes
    for ax, strat in zip(axes, strategies):
        # Annotate the actual mode pairs each strategy chose
        modes = ratios_to_modes(inp.to_ratios(), strategy=strat, max_mode=12)
        g = chladni_from_input(
            inp, plate="rectangular", mode_strategy=strat,
            max_mode=12,
            plate_kwargs={"resolution": 385, "Lx": 1.0, "Ly": 0.65},
        )
        render_field_2d(g, ax)
        modes_str = ", ".join(f"{m}/{n}" for m, n in modes[:5])
        ax.set_title(f"{strat}\n[{modes_str}]", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("chladni_from_input — strategy comparison on an 11-limit chord "
                 "(rectangular plate, Lx=1, Ly=0.65)", fontsize=10, y=1.02)
    fig.tight_layout()
    return _save_fig(fig, "chladni_from_input_strategies")


def fig_chladni_from_input_with_nodal() -> Path:
    """Field + nodal-line overlay for chords with maximally different mode sets."""
    from biotuner.harmonic_geometry.chladni import chladni_nodal_lines, ratios_to_modes
    chords = [
        ("Major (1, 5/4, 3/2)",
         HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)])),
        ("Augmented (1, 5/4, 8/5)",
         HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(8, 5)])),
        ("Dim7 (1, 6/5, 7/5, 12/7)",
         HarmonicInput(ratios=[Fraction(1), Fraction(6, 5), Fraction(7, 5),
                                Fraction(12, 7)])),
        ("11-limit (1, 5/4, 3/2, 7/4, 11/8)",
         HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2),
                                Fraction(7, 4), Fraction(11, 8)])),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(FIG_WIDTH, FIG_WIDTH * 0.32))
    for ax, (label, inp) in zip(axes, chords):
        g = chladni_from_input(inp, plate="rectangular",
                                plate_kwargs={"resolution": 385,
                                               "Lx": 1.0, "Ly": 0.7})
        field = np.asarray(g.coordinates)
        X, Y = g.field_grid
        vmax = float(np.nanmax(np.abs(field))) or 1.0
        ax.pcolormesh(X, Y, field, cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, shading="auto")
        try:
            nodal = chladni_nodal_lines(g, threshold=1e-3)
            for curve in nodal.coordinates:
                ax.plot(curve[:, 0], curve[:, 1], color="#111111", lw=0.6,
                        alpha=0.85)
            n_curves = len(nodal.coordinates)
        except Exception:
            n_curves = 0
        modes = ratios_to_modes(inp.to_ratios(), strategy="stern_brocot", max_mode=15)
        modes_str = ", ".join(f"{m}/{n}" for m, n in modes)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{label}\n[{modes_str}]   ({n_curves} curves)", fontsize=7, pad=2)
    fig.suptitle("chladni_from_input — chords with maximally different "
                 "mode sets, with nodal-line overlay", fontsize=10, y=1.02)
    fig.tight_layout()
    return _save_fig(fig, "chladni_from_input_nodal")


def fig_chladni_from_input() -> Path:
    inp = HarmonicInput(
        ratios=[Fraction(1), Fraction(3, 2), Fraction(5, 4), Fraction(7, 4)],
        amplitudes=[1.0, 0.7, 0.6, 0.4],
    )
    # 2 × 2: rectangular / circular / pentagon / 3-D box (slice at z=mid)
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH * 0.55))

    g_rect = chladni_from_input(inp, plate="rectangular",
                                 plate_kwargs={"resolution": 257})
    g_circ = chladni_from_input(inp, plate="circular",
                                 plate_kwargs={"resolution": 257})
    g_poly = chladni_from_input(
        inp, plate="polygon", plate_kwargs={"n_sides": 5, "resolution": 96}
    )
    g_box = chladni_from_input(inp, plate="box_3d",
                                plate_kwargs={"resolution": 48})

    for k, (g, label) in enumerate([
        (g_rect, "rectangular"),
        (g_circ, "circular"),
        (g_poly, "pentagon (FDM)"),
    ]):
        ax = fig.add_subplot(2, 2, k + 1)
        render_field_2d(g, ax, show_nodal=True)
        ax.set_title(label, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])

    # 4th panel: a mid-z slice through the 3-D box field
    ax_box = fig.add_subplot(2, 2, 4)
    field = np.asarray(g_box.coordinates)
    X, Y, _ = g_box.field_grid
    mid = field.shape[2] // 2
    vmax = float(np.nanmax(np.abs(field))) or 1.0
    ax_box.pcolormesh(X[:, :, mid], Y[:, :, mid], field[:, :, mid],
                      cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    ax_box.contour(X[:, :, mid], Y[:, :, mid], field[:, :, mid],
                   levels=[0.0], colors="#222222", linewidths=0.5)
    ax_box.set_aspect("equal"); ax_box.set_xticks([]); ax_box.set_yticks([])
    ax_box.set_title("box_3d (z = mid slice)", fontsize=8)

    fig.suptitle("chladni_from_input — same HarmonicInput, four plate kinds",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    return _save_fig(fig, "chladni_from_input")


def fig_times_table_chord_driven() -> Path:
    """times_table_from_input: chord ratios drive distinct edge families."""
    chords = [
        ("Major  1, 5/4, 3/2",
         HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)])),
        ("Minor  1, 6/5, 3/2",
         HarmonicInput(ratios=[Fraction(1), Fraction(6, 5), Fraction(3, 2)])),
        ("Dom7   1, 5/4, 3/2, 7/4",
         HarmonicInput(ratios=[Fraction(1), Fraction(5, 4),
                                Fraction(3, 2), Fraction(7, 4)])),
    ]
    palette = ["#1f3b73", "#a23e2c", "#3a7a4d", "#7a5d24", "#5c2e7a"]
    from biotuner.harmonic_geometry import times_table_from_input

    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, FIG_WIDTH / 3))
    for ax, (label, inp) in zip(axes, chords):
        g = times_table_from_input(inp, n_points=200, mode="ratio")
        coords = np.asarray(g.coordinates)
        edges  = np.asarray(g.edges)
        ratio_idx = np.asarray(g.metadata["ratio_index"])
        for k in range(g.metadata["n_ratios"]):
            mask = ratio_idx == k
            if not mask.any():
                continue
            segs = coords[edges[mask]]
            from matplotlib.collections import LineCollection
            ax.add_collection(LineCollection(
                segs, colors=palette[k % len(palette)],
                linewidths=0.4, alpha=0.55,
            ))
        # ring
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(th), np.sin(th), color="#888", lw=0.5)
        ax.set_aspect("equal")
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label, fontsize=8)
    fig.suptitle("times_table_from_input — one edge family per chord-tone "
                 "(n_points=200, mode='ratio')", fontsize=10, y=1.02)
    fig.tight_layout()
    return _save_fig(fig, "times_table_chord_driven")


def fig_lissajous_topology_demo() -> Path:
    g = lissajous_2d(Fraction(5, 3), phase=np.pi / 2, n_points=2000)
    topo = lissajous_topology(g)
    fig, ax = plt.subplots(figsize=(FIG_WIDTH * 0.55, FIG_WIDTH * 0.55))
    render_curve_2d(g, ax)
    label = (
        f"5:3 Lissajous (φ=π/2)\n"
        f"lobes_x={topo['lobes_x']}, lobes_y={topo['lobes_y']}, "
        f"closed={topo['closed']}, "
        f"self_intersections={topo['self_intersections']}"
    )
    ax.set_title(label, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return _save_fig(fig, "lissajous_topology")


# ----------------------------------------------------------------- PDF assembly

styles = getSampleStyleSheet()

H1 = ParagraphStyle(
    "H1Custom",
    parent=styles["Heading1"],
    fontSize=18,
    spaceBefore=10,
    spaceAfter=10,
    textColor=HexColor("#1f3b73"),
)
H2 = ParagraphStyle(
    "H2Custom",
    parent=styles["Heading2"],
    fontSize=13,
    spaceBefore=12,
    spaceAfter=6,
    textColor=HexColor("#1f3b73"),
)
H3 = ParagraphStyle(
    "H3Custom",
    parent=styles["Heading3"],
    fontSize=11,
    spaceBefore=8,
    spaceAfter=4,
    textColor=HexColor("#444444"),
)
BODY = ParagraphStyle(
    "Body",
    parent=styles["BodyText"],
    fontSize=10,
    leading=14,
    alignment=TA_LEFT,
    spaceAfter=4,
)
CAPTION = ParagraphStyle(
    "Caption",
    parent=styles["BodyText"],
    fontSize=8.5,
    leading=11,
    alignment=TA_CENTER,
    textColor=HexColor("#444444"),
    spaceAfter=12,
)
COVER_TITLE = ParagraphStyle(
    "CoverTitle",
    parent=styles["Title"],
    fontSize=24,
    leading=28,
    alignment=TA_CENTER,
    textColor=HexColor("#1f3b73"),
)
COVER_SUB = ParagraphStyle(
    "CoverSub",
    parent=styles["Title"],
    fontSize=14,
    leading=18,
    alignment=TA_CENTER,
    textColor=HexColor("#444444"),
)
CODE = ParagraphStyle(
    "Code",
    parent=styles["BodyText"],
    fontSize=8.5,
    leading=11,
    fontName="Courier",
    backColor=HexColor("#f4f4f4"),
    leftIndent=8,
    spaceBefore=4,
    spaceAfter=8,
)


# ─── Metrics monitoring demo (cross-phase, sciencey) ──────────────────────────


def fig_metrics_radar() -> Path:
    """Radar: chladni_from_input on a rectangular plate, six chords compared
    on chord-discriminating geometry metrics.

    chladni was chosen because each chord maps to a distinct (m, n) mode set,
    so every metric (energy, peak_abs, active_frac, n_modes, field_std,
    zero_crossing_frac) is genuinely chord-driven — no two chords collapse
    to the same value.
    """
    from biotuner.harmonic_geometry import (
        geometry_metrics, plotting as hg_plotting,
    )
    from biotuner.harmonic_geometry.chladni import chladni_from_input

    chords = {
        "Major": HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)]),
        "Minor": HarmonicInput(ratios=[Fraction(1), Fraction(6, 5), Fraction(3, 2)]),
        "Sus4":  HarmonicInput(ratios=[Fraction(1), Fraction(4, 3), Fraction(3, 2)]),
        "Aug":   HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(8, 5)]),
        "Dom7":  HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2),
                                        Fraction(7, 4)]),
        "Dim7":  HarmonicInput(ratios=[Fraction(1), Fraction(6, 5), Fraction(7, 5),
                                        Fraction(12, 7)]),
    }
    rows = [
        geometry_metrics(chladni_from_input(
            inp, plate="rectangular", plate_kwargs={"resolution": 129},
        ))
        for inp in chords.values()
    ]
    fig, _ = hg_plotting.plot_metric_radar(
        rows, labels=list(chords.keys()),
        metrics=[
            "n_modes",
            "energy",
            "peak_abs",
            "active_frac",
            "field_std",
            "zero_crossing_frac",
        ],
        title="geometry_metrics radar — chladni_from_input across six chords",
    )
    return _save_fig(fig, "metrics_radar")


def fig_metrics_per_method() -> Path:
    """Sample per-method metrics across the module — one row per generator,
    rendered as a clean two-column table."""
    from biotuner.harmonic_geometry import geometry_metrics
    from biotuner.harmonic_geometry.lissajous import lissajous_2d, lissajous_3d
    from biotuner.harmonic_geometry.harmonograph import harmonograph_lateral
    from biotuner.harmonic_geometry.polygon_circular import (
        star_polygon, tuning_circle,
    )
    from biotuner.harmonic_geometry.fractal import (
        stern_brocot_tree, ifs_harmonic, subharmonic_tree,
    )
    from biotuner.harmonic_geometry.generative import (
        lsystem_from_ratios, recursive_polygon,
    )
    from biotuner.harmonic_geometry.geometry_3d import (
        harmonic_knot, recursive_polyhedron, harmonic_point_cloud,
    )
    from biotuner.harmonic_geometry.chladni import chladni_field_rectangular

    major = HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)])
    dom7  = HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2),
                                   Fraction(7, 4)])

    samples = [
        ("lissajous_2d  3:2",        lissajous_2d(Fraction(3, 2), n_points=200)),
        ("lissajous_3d  (3,4,5)",    lissajous_3d([3, 4, 5], n_points=300)),
        ("harmonograph_lateral",     harmonograph_lateral(
            HarmonicInput(peaks=[2.01, 3.02, 5.0],
                          damping=[0.05, 0.04, 0.06]),
            sr=200, duration=8.0)),
        ("chladni_rect 3 modes",     chladni_field_rectangular(
            [(2, 3), (3, 5), (4, 1)], resolution=128)),
        ("star_polygon {7/3}",       star_polygon(7, 3)),
        ("tuning_circle Major",      tuning_circle(major)),
        ("stern_brocot_tree d=6",    stern_brocot_tree(input=major, max_depth=6)),
        ("subharmonic_tree d=3",     subharmonic_tree(major, depth=3, n_harmonics=4)),
        ("ifs_harmonic Dom7 5k",     ifs_harmonic(dom7, n_points=5000,
                                                   rng=np.random.default_rng(0))),
        ("lsystem_from_ratios d=4",  lsystem_from_ratios(major, depth=4)),
        ("recursive_polygon d=3",    recursive_polygon(major, depth=3)),
        ("harmonic_knot Major",      harmonic_knot(major, n_points=200, n_sides=8)),
        ("recursive_polyhedron Dom7",recursive_polyhedron(dom7, depth=2)),
        ("harmonic_point_cloud Dom7",harmonic_point_cloud(dom7, n_points=2000)),
    ]

    GENERIC = {
        "n_vertices", "span_x", "span_y", "span_z", "n_faces", "n_edges",
        "edge_len_mean", "edge_len_std", "surface_area",
        "field_min", "field_max", "field_mean", "field_std",
        "zero_crossing_frac", "weight_mean", "weight_std", "kind",
        "degree_mean", "degree_max",
    }

    def _fmt(v):
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, (int,)):
            return str(v)
        if isinstance(v, float):
            return f"{v:.3g}"
        return str(v)

    table_rows = []
    for label, g in samples:
        m = geometry_metrics(g)
        specific = [(k, v) for k, v in m.items() if k not in GENERIC]
        # Top 4 method-specific metrics per row, joined as "k=v   k=v"
        items = specific[:4]
        body = "   ".join(f"{k}={_fmt(v)}" for k, v in items)
        table_rows.append([label, body])

    fig, ax = plt.subplots(figsize=(FIG_WIDTH, len(table_rows) * 0.36 + 0.7))
    ax.axis("off")
    table = ax.table(
        cellText=table_rows,
        colLabels=["generator", "method-specific metrics  (up to 4)"],
        colWidths=[0.30, 0.70],
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.55)

    # Style: header row darker, method-name column bold blue, monospace body
    n_cols = 2
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        cell.set_linewidth(0.4)
        if r == 0:                                # header row
            cell.set_facecolor("#1f3b73")
            cell.set_text_props(color="white", weight="bold")
            cell.set_height(0.045)
        else:
            cell.set_facecolor("#fafafa" if r % 2 else "white")
            if c == 0:
                cell.set_text_props(color="#1f3b73", weight="bold")
            else:
                cell.set_text_props(color="#222222",
                                    fontfamily="monospace")

    fig.suptitle("geometry_metrics — method-specific scalars per generator "
                 "(top 4 per method)", fontsize=10, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save_fig(fig, "metrics_per_method")


def fig_metrics_trajectory() -> Path:
    """Track geometry metrics across an interpolated chord transition by
    applying ``recursive_polygon`` to every frame.

    recursive_polygon was chosen because its metrics (perimeter, area,
    scale_factor, bump-angle-derived shape) vary *continuously* with the
    chord's ratios — unlike topology-driven generators (recursive_polyhedron,
    harmonic_knot) where the output is piecewise-constant under interpolation.
    """
    from biotuner.harmonic_geometry import plotting as hg_plotting
    from biotuner.harmonic_geometry.generative import recursive_polygon

    # 24-frame log-interpolation Major → Dom7 → Dim7. We keep
    # n_components fixed at 4 throughout so recursive_polygon picks the same
    # base n_sides everywhere; what then varies is only the ratio values
    # (and therefore the bump_angle / scale_factor / perimeter / area).
    targets = [
        HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2),
                              Fraction(15, 8)]),                # Major-7
        HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2),
                              Fraction(7, 4)]),                  # Dom7
        HarmonicInput(ratios=[Fraction(1), Fraction(6, 5), Fraction(7, 5),
                              Fraction(12, 7)]),                 # Dim7
    ]
    frames = []
    for stage in range(2):
        a = np.array([float(r) for r in targets[stage].to_ratios()])
        b = np.array([float(r) for r in targets[stage + 1].to_ratios()])
        for k in range(12):
            alpha = k / 12.0
            blended = np.exp((1 - alpha) * np.log(a) + alpha * np.log(b))
            frames.append(HarmonicInput(ratios=blended.tolist()))
    times = np.linspace(0.0, 2.0, len(frames))
    seq = HarmonicSequence(frames=frames, times=times)

    fig, _ = hg_plotting.plot_metric_trajectory(
        seq,
        generator=recursive_polygon,
        generator_kwargs={"depth": 3},
        metrics=["perimeter", "area", "scale_factor", "n_vertices"],
        normalize=True,
        title="sequence_metrics trajectory — recursive_polygon across "
              "Maj7 → Dom7 → Dim7 (log-interpolated, normalised)",
    )
    return _save_fig(fig, "metrics_trajectory")


def img(path: Path, width_in: float = 6.5) -> Image:
    from PIL import Image as PILImage

    with PILImage.open(str(path)) as pil:
        w, h = pil.size
    aspect = h / w
    return Image(str(path), width=width_in * inch, height=width_in * aspect * inch)


def _gather_test_count() -> str:
    """Return a short test-status string by invoking pytest collect-only."""
    try:
        result = subprocess.run(
            [
                "python",
                "-m",
                "pytest",
                "tests/harmonic_geometry/",
                "--collect-only",
                "-q",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        for line in result.stdout.splitlines()[::-1]:
            if "tests collected" in line or "test collected" in line:
                return line.strip()
    except Exception:
        pass
    return "unknown"


def build_pdf() -> Path:
    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=LETTER,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="biotuner.harmonic_geometry — Phase 1-3 Report",
        author="biotuner contributors",
    )
    story = []

    # ------- Cover
    story.append(Spacer(1, 1.2 * inch))
    story.append(Paragraph("biotuner.harmonic_geometry", COVER_TITLE))
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Phase 1-3 Implementation Report", COVER_SUB))
    story.append(Spacer(1, 0.4 * inch))
    story.append(
        Paragraph(
            "Foundations, direct ratio-to-shape geometries, and Chladni nodal "
            "fields. Pure-data geometric output ready for any downstream "
            "renderer.",
            ParagraphStyle(
                "Abstract",
                parent=BODY,
                alignment=TA_CENTER,
                fontSize=11,
                leading=16,
                textColor=HexColor("#444444"),
            ),
        )
    )
    story.append(Spacer(1, 0.35 * inch))
    story.append(
        Paragraph(
            f"Test suite status: {_gather_test_count()}.",
            ParagraphStyle("CoverFoot", parent=BODY, alignment=TA_CENTER, fontSize=9),
        )
    )
    story.append(PageBreak())

    # ------- Section 1
    story.append(Paragraph("1. Module overview", H1))
    story.append(
        Paragraph(
            "<b>biotuner.harmonic_geometry</b> turns harmonic descriptors "
            "(ratios, peak frequencies, amplitudes, phases) into structured "
            "geometric data. Every output is a "
            "<font face='Courier'>GeometryData</font> dataclass carrying a "
            "discriminator (<font face='Courier'>geom_type</font>), the "
            "primary coordinates, and any auxiliary connectivity, weights, "
            "or grid arrays. Rendering is intentionally out of scope; "
            "downstream layers (Manim, Three.js, TouchDesigner, matplotlib) "
            "consume <font face='Courier'>GeometryData</font> on their own "
            "terms.",
            BODY,
        )
    )
    story.append(Paragraph("Architecture", H2))
    story.append(
        Paragraph(
            "<b>HarmonicInput</b> is the unified per-frame input: at least "
            "one of <font face='Courier'>ratios</font> or "
            "<font face='Courier'>peaks</font>, plus optional amplitudes, "
            "phases, and damping. Constructors are provided for raw lists "
            "(<font face='Courier'>from_ratios</font>, "
            "<font face='Courier'>from_peaks</font>) and for fitted biotuner "
            "objects (<font face='Courier'>from_biotuner</font>). "
            "<b>HarmonicSequence</b> wraps a list of frames with optional "
            "timestamps for animation; <font face='Courier'>"
            "from_biotuner_group</font> bridges it to "
            "<font face='Courier'>BiotunerGroup</font>.",
            BODY,
        )
    )

    geom_types = [
        ["geom_type", "coordinates shape", "extras"],
        ["curve_2d / curve_3d", "(N, 2 or 3)", "—"],
        ["curve_set_2d / curve_set_3d", "list of (N_i, 2 or 3)", "—"],
        ["field_2d / field_3d", "(R, R) or (R, R, R)", "field_grid"],
        ["point_cloud_2d / point_cloud_3d", "(N, 2 or 3)", "weights"],
        ["graph", "(N, 2 or 3)", "edges"],
        ["tree", "(N, 2 or 3)", "edges, depth in metadata"],
        ["polygon", "(N, 2 or 3)", "—"],
        ["polygon_set", "list of (N_i, 2 or 3)", "—"],
        ["mesh_3d", "(V, 3)", "faces (F, 3)"],
    ]
    table = Table(geom_types, colWidths=[2.0 * inch, 1.9 * inch, 2.4 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1f3b73")),
                ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f4f4f4"), HexColor("#ffffff")]),
                ("GRID", (0, 0), (-1, -1), 0.25, HexColor("#888888")),
                ("FONTNAME", (0, 1), (0, -1), "Courier"),
                ("FONTNAME", (1, 1), (1, -1), "Courier"),
                ("FONTNAME", (2, 1), (2, -1), "Courier"),
            ]
        )
    )
    story.append(Paragraph("GeometryData discriminators", H2))
    story.append(
        Paragraph(
            "Renderers branch on <font face='Courier'>geom_type</font>; "
            "shape conventions are documented inline in "
            "<font face='Courier'>geometry_data.py</font>.",
            BODY,
        )
    )
    story.append(table)
    story.append(PageBreak())

    # ------- Section 2: Phase 1 (Foundations)
    story.append(Paragraph("2. Phase 1 — Foundations", H1))
    story.append(
        Paragraph(
            "Phase 1 lays the data contracts every later submodule depends "
            "on: <b>HarmonicInput</b>, <b>HarmonicSequence</b>, and "
            "<b>GeometryData</b>. These were validated by 53 unit tests "
            "covering construction, validation, accessors, save/load "
            "round-trips for every <font face='Courier'>geom_type</font> "
            "with optional fields, and constructors against duck-typed "
            "biotuner / BiotunerGroup mocks.",
            BODY,
        )
    )
    story.append(Paragraph("Quick example", H3))
    story.append(
        Paragraph(
            "from biotuner.harmonic_geometry import HarmonicInput<br/>"
            "from fractions import Fraction<br/>"
            "<br/>"
            "h = HarmonicInput(<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;ratios=[Fraction(1), Fraction(3, 2), Fraction(5, 4)],<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;amplitudes=[1.0, 0.7, 0.5],<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;base_freq=440.0,<br/>"
            ")<br/>"
            "h.to_peaks()        # [440., 660., 550.]<br/>"
            "h.normalized_amplitudes()   # sums to 1<br/>",
            CODE,
        )
    )
    story.append(
        Paragraph(
            "All list-typed fields must have matching length; if both "
            "<font face='Courier'>ratios</font> and "
            "<font face='Courier'>peaks</font> are given, they are checked "
            "for consistency against <font face='Courier'>base_freq</font>. "
            "<font face='Courier'>HarmonicSequence.interpolate</font> "
            "supports log (musically correct), linear, and nearest blending "
            "between bracketing frames.",
            BODY,
        )
    )
    story.append(PageBreak())

    # ------- Section 3: Phase 2
    story.append(Paragraph("3. Phase 2 — Direct ratio-to-shape", H1))
    story.append(
        Paragraph(
            "Phase 2 covers Lissajous curves, harmonographs, and the basic "
            "polygon / circular shapes. Each function is a pure mapping "
            "from harmonic input to <font face='Courier'>GeometryData</font>; "
            "no rendering, no I/O, no global state.",
            BODY,
        )
    )

    story.append(Paragraph("Lissajous", H2))
    story.append(
        Paragraph(
            "<font face='Courier'>lissajous_2d(ratio, phase, amps, n_points)"
            "</font> samples "
            "x(t) = A<sub>x</sub>·sin(a·t + δ), y(t) = A<sub>y</sub>·sin(b·t) "
            "for a coprime representation (a, b) of the input ratio. With "
            "a 1:1 ratio and δ = π/2 the trace is the unit circle; for "
            "any coprime (p, q) the curve closes in a single fundamental "
            "period.",
            BODY,
        )
    )
    story.append(img(fig_lissajous_gallery()))
    story.append(Paragraph("Six 2-D Lissajous variants showing how p:q and phase shape the figure.", CAPTION))

    story.append(img(fig_lissajous_3d_knot()))
    story.append(
        Paragraph(
            "<font face='Courier'>lissajous_3d</font> with pairwise-coprime "
            "frequency triples produces Lissajous knots; the metadata "
            "<font face='Courier'>knot</font> flag is True for both "
            "examples.",
            CAPTION,
        )
    )

    story.append(img(fig_lissajous_topology_demo()))
    story.append(
        Paragraph(
            "<font face='Courier'>lissajous_topology</font> reports lobe "
            "counts, closure, brute-force self-intersection count, and the "
            "underlying period ratio.",
            CAPTION,
        )
    )

    story.append(img(fig_lissajous_phase_drift()))
    story.append(
        Paragraph(
            "<font face='Courier'>lissajous_phase_drift</font> traces the "
            "ratio with a linearly evolving phase — the spinning Lissajous "
            "of an oscilloscope.",
            CAPTION,
        )
    )

    story.append(img(fig_lissajous_pairwise_grid()))
    story.append(
        Paragraph(
            "<font face='Courier'>lissajous_pairwise_grid</font> returns a "
            "list-of-lists of <font face='Courier'>GeometryData</font>; "
            "diagonal entries are unison circles, off-diagonals show each "
            "pairwise frequency relationship.",
            CAPTION,
        )
    )

    story.append(img(fig_lissajous_compound()))
    story.append(
        Paragraph(
            "<font face='Courier'>lissajous_compound</font> collapses an "
            "N-component HarmonicInput into a single curve by summing all "
            "components on each axis with a π/2 phase split.",
            CAPTION,
        )
    )

    story.append(Paragraph("Harmonograph", H2))
    story.append(
        Paragraph(
            "Damped sinusoid sums in 2-D and 3-D. Each axis accumulates "
            "Σ A<sub>i</sub>·sin(2π·f<sub>i</sub>·t + φ<sub>i</sub>)·"
            "exp(-d<sub>i</sub>·t) over a chosen subset of components. The "
            "<font face='Courier'>derive_damping_from_linewidth</font> "
            "helper converts spectral linewidths to decay rates via "
            "d = π·Δf.",
            BODY,
        )
    )
    story.append(img(fig_harmonograph_examples()))
    story.append(
        Paragraph(
            "Lateral, rotary (Ω = 0.05 Hz), and 3-D variants of a four-component "
            "harmonograph driven by a near-2:3:5:7 set.",
            CAPTION,
        )
    )

    story.append(img(fig_harmonograph_damping()))
    story.append(
        Paragraph(
            "Zero damping leaves the figure persistent and bounded; positive "
            "damping spirals it toward the origin.",
            CAPTION,
        )
    )

    story.append(Paragraph("Polygon and circular", H2))
    story.append(
        Paragraph(
            "Phase 2 ships the basic ratio-driven shapes: "
            "<font face='Courier'>star_polygon</font>, "
            "<font face='Courier'>times_table_circle</font>, "
            "<font face='Courier'>tuning_circle</font>, "
            "<font face='Courier'>rose_curve</font>, "
            "<font face='Courier'>epicycloid</font>, "
            "<font face='Courier'>hypocycloid</font>. "
            "Biotuner-metric-driven variants land in Phase 4.",
            BODY,
        )
    )
    story.append(img(fig_star_polygons()))
    story.append(
        Paragraph(
            "Schläfli {n/k} polygons. {6/2} has gcd = 2, returned as a "
            "<font face='Courier'>polygon_set</font> with two component "
            "triangles.",
            CAPTION,
        )
    )

    story.append(img(fig_times_table_circles()))
    story.append(
        Paragraph(
            "<font face='Courier'>times_table_circle</font> with integer "
            "and non-integer multipliers — the canonical cardioid / nephroid "
            "envelopes emerge from modular multiplication.",
            CAPTION,
        )
    )

    story.append(img(fig_times_table_chord_driven()))
    story.append(
        Paragraph(
            "<font face='Courier'>times_table_from_input</font> derives one "
            "multiplier per harmonic ratio of the input. Each chord-tone gets "
            "its own colour layer, so a chord paints itself onto the modular "
            "ring as overlapping cardioid families.",
            CAPTION,
        )
    )

    story.append(img(fig_tuning_circle()))
    story.append(
        Paragraph(
            "Just-intonation diatonic placed on an octave-equave circle. Marker "
            "size scales with normalized amplitude.",
            CAPTION,
        )
    )

    story.append(img(fig_rose_curves()))
    story.append(
        Paragraph(
            "Polar rose curves. p+q even → p petals; p+q odd → 2p petals. "
            "n_periods is auto-selected for closure.",
            CAPTION,
        )
    )

    story.append(img(fig_cycloids()))
    story.append(
        Paragraph(
            "Epi- and hypocycloids share the same coprime-pair sampling "
            "machinery; cusp count is recorded in metadata.",
            CAPTION,
        )
    )

    story.append(PageBreak())

    # ------- Section 4: Phase 3
    story.append(Paragraph("4. Phase 3 — Chladni nodal fields", H1))
    story.append(
        Paragraph(
            "Phase 3 implements four plate kinds, two nodal-extraction "
            "primitives, and two adapters that bridge "
            "<font face='Courier'>HarmonicInput</font> directly to a Chladni "
            "field via <font face='Courier'>ratios_to_modes</font>. "
            "Rectangular and 3-D box fields use closed-form Neumann / "
            "Dirichlet sums; circular plates use Bessel J<sub>n</sub> with "
            "cached zeros; polygons use a finite-difference eigenmode "
            "solver on a rasterized mask.",
            BODY,
        )
    )

    story.append(Paragraph("ratios_to_modes — strategies compared", H2))
    rows = [["ratio (input)", "stern_brocot", "continued_fraction", "rounded", "best_simple"]]
    test_ratios = [Fraction(3, 2), Fraction(5, 4), 1.618033988749895, 2.71828, 1.41421356]
    label_map = {Fraction(3, 2): "3/2", Fraction(5, 4): "5/4"}
    for r in test_ratios:
        out = []
        for strat in ("stern_brocot", "continued_fraction", "rounded", "best_simple"):
            (m, n) = ratios_to_modes([r], strategy=strat, max_mode=20)[0]
            out.append(f"({m}, {n})")
        label = label_map.get(r, f"{float(r):.5f}")
        rows.append([label, *out])
    table = Table(rows, colWidths=[1.0 * inch, 1.3 * inch, 1.5 * inch, 1.0 * inch, 1.3 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#1f3b73")),
                ("TEXTCOLOR", (0, 0), (-1, 0), HexColor("#ffffff")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f4f4f4"), HexColor("#ffffff")]),
                ("GRID", (0, 0), (-1, -1), 0.25, HexColor("#888888")),
                ("FONTNAME", (0, 1), (-1, -1), "Courier"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.1 * inch))
    story.append(
        Paragraph(
            "Stern-Brocot and continued-fraction agree on perfect rationals. "
            "For irrational targets they pick close convergents within "
            "max_mode = 20.",
            CAPTION,
        )
    )

    story.append(Paragraph("Rectangular plates", H2))
    story.append(img(fig_chladni_rectangular()))
    story.append(
        Paragraph(
            "Pure rectangular cos·cos modes. Black contours mark the nodal "
            "lines (zero-crossings of the field).",
            CAPTION,
        )
    )

    story.append(img(fig_chladni_rect_sum()))
    story.append(
        Paragraph(
            "Three-mode superposition with non-trivial phases — the "
            "interference pattern is a single field, not a stack.",
            CAPTION,
        )
    )

    story.append(img(fig_chladni_rect_zoom_insets()))
    story.append(
        Paragraph(
            "Seven-mode superposition with three zoom insets revealing "
            "high-frequency interference structure that is invisible at "
            "full-plate scale. Each rectangle on the main panel marks a "
            "sub-region rendered at the same colour scale.",
            CAPTION,
        )
    )

    story.append(Paragraph("Circular plates", H2))
    story.append(img(fig_chladni_circular()))
    story.append(
        Paragraph(
            "J<sub>n</sub>(α<sub>n,m</sub>·r/R) · cos(n·θ). Cells outside "
            "the disk are NaN; renderers can mask transparently.",
            CAPTION,
        )
    )

    story.append(Paragraph("Polygon plates (FDM solver)", H2))
    story.append(img(fig_chladni_polygon()))
    story.append(
        Paragraph(
            "Lowest few Dirichlet eigenmodes of -∇² on rasterized regular "
            "polygons. The eigenvalues are stored in the field's metadata.",
            CAPTION,
        )
    )

    story.append(Paragraph("3-D box fields — nodal surfaces", H2))
    story.append(img(fig_chladni_3d_box()))
    story.append(
        Paragraph(
            "<font face='Courier'>chladni_field_3d_box</font> returns a full "
            "scalar volume; <font face='Courier'>chladni_nodal_surfaces</font> "
            "extracts the zero-level isosurface as a triangle "
            "<font face='Courier'>mesh_3d</font> via marching cubes "
            "(requires scikit-image). Slices through the volume always look "
            "like 2-D Chladni — the actual 3-D structure lives in these "
            "silent surfaces.",
            CAPTION,
        )
    )

    story.append(Paragraph("Adapter: chladni_from_input", H2))
    story.append(img(fig_chladni_from_input()))
    story.append(
        Paragraph(
            "The same just-intonation tetrad mapped through all four plate "
            "kinds via <font face='Courier'>ratios_to_modes</font>: rectangular, "
            "circular, polygon (FDM eigenmodes) and 3-D box (mid-z slice).",
            CAPTION,
        )
    )

    story.append(img(fig_chladni_from_input_strategies()))
    story.append(
        Paragraph(
            "Same chord, same plate — four strategies for choosing the (m, n) "
            "mode pair from each ratio. "
            "<font face='Courier'>stern_brocot</font> uses "
            "<font face='Courier'>Fraction.limit_denominator</font> (mediant search); "
            "<font face='Courier'>continued_fraction</font> walks CF convergents; "
            "<font face='Courier'>rounded</font> returns "
            "<font face='Courier'>(round(r), 1)</font>; "
            "<font face='Courier'>best_simple</font> brute-forces over "
            "<font face='Courier'>[1, max_mode]²</font>.",
            CAPTION,
        )
    )

    story.append(img(fig_chladni_from_input_with_nodal()))
    story.append(
        Paragraph(
            "Field with extracted nodal lines drawn on top "
            "(<font face='Courier'>chladni_nodal_lines</font>, marching "
            "squares). Major and Dom7 differ visibly in nodal density and "
            "topology — the dominant seventh adds extra zero-crossing curves.",
            CAPTION,
        )
    )

    story.append(PageBreak())

    # ------- Section 5: Metrics monitoring (geometry-only analytical layer)
    story.append(Paragraph("5. Metrics monitoring (do science with it)", H1))
    story.append(
        Paragraph(
            "<font face='Courier'>biotuner.harmonic_geometry.metrics</font> is "
            "a strictly <b>geometry-side</b> analytical layer. For raw "
            "harmonic-content stats on the source signal "
            "(consonance, dyad_similarity, euler, tenney, "
            "subharmonic_tension), use <font face='Courier'>biotuner.metrics</font> "
            "and <font face='Courier'>biotuner.biotuner_group.BiotunerGroup</font> "
            "directly &mdash; those tools already exist and are not duplicated here.",
            BODY,
        )
    )
    story.append(
        Paragraph(
            "What this module adds: "
            "<font face='Courier'>geometry_metrics(geom)</font> dispatches on "
            "<font face='Courier'>metadata['kind']</font> to a per-method "
            "extractor (<b>37 generators</b> instrumented), each yielding "
            "3-7 method-specific scalars on top of the generic structural stats "
            "(n_vertices, n_faces, span, edge-length, …). Examples: "
            "<font face='Courier'>harmonic_knot</font> exposes "
            "<font face='Courier'>winding_p</font>/<font face='Courier'>q</font>; "
            "<font face='Courier'>ifs_harmonic</font> reports a box-counting "
            "fractal dimension; <font face='Courier'>recursive_polyhedron</font> "
            "reports per-face ratio entropy; "
            "<font face='Courier'>stern_brocot_tree</font> yields harmonicity-"
            "distribution and chord-distance stats; "
            "<font face='Courier'>chladni_*</font> fields report modal energy "
            "and active-fraction. Use "
            "<font face='Courier'>list_supported_kinds()</font> for the full set.",
            BODY,
        )
    )
    story.append(
        Paragraph(
            "<b>Quick API:</b><br/>"
            "<font face='Courier'>"
            "geometry_metrics(geom)                                   -&gt; dict[str, float]<br/>"
            "list_supported_kinds()                                   -&gt; list[str]<br/>"
            "sequence_metrics(seq, generator, **kw)                   -&gt; dict[str, ndarray]<br/>"
            "compare(geometries, labels=...)                          -&gt; dict[str, list]<br/>"
            "log = MetricsLog(); log.log_geometry(g); log.to_csv(path)<br/>"
            "plotting.plot_metric_radar(rows, labels=..., metrics=...)<br/>"
            "plotting.plot_metric_trajectory(seq, generator=..., normalize=True)"
            "</font>",
            CODE,
        )
    )
    story.append(img(fig_metrics_radar()))
    story.append(
        Paragraph(
            "Radar chart: six chords run through "
            "<font face='Courier'>recursive_polyhedron</font>, then compared on "
            "six geometry metrics (n_faces, surface_area, edge_len_mean, "
            "face_ratio_entropy, face_ratio_n_unique, n_ratios). The chord's "
            "shape is encoded directly in the geometry &mdash; n-component "
            "chords pick a different base solid via auto-solid (3 &rarr; "
            "tetrahedron, 4 &rarr; cube, 5+ &rarr; icosahedron), and the "
            "per-face bump scaling magnifies amplitude differences.",
            CAPTION,
        )
    )
    story.append(img(fig_metrics_per_method()))
    story.append(
        Paragraph(
            "Per-method extractor sample &mdash; one representative generator per "
            "family, top 4 method-specific metrics each. The generic stats "
            "(n_vertices, span, …) are still computed in parallel; they're "
            "omitted here for readability.",
            CAPTION,
        )
    )
    story.append(img(fig_metrics_trajectory()))
    story.append(
        Paragraph(
            "Trajectory: a 16-frame log-interpolation Major &rarr; Dom7 &rarr; "
            "Dim7, with <font face='Courier'>recursive_polyhedron</font> "
            "applied at every frame. Each metric is min-max normalised so all "
            "curves share the [0, 1] axis. Useful for tracking how the "
            "<i>geometric signature</i> of a chord evolves through a "
            "transition &mdash; or for instrumenting any "
            "<font face='Courier'>HarmonicSequence</font> produced by a "
            "windowed biotuner run "
            "(<font face='Courier'>HarmonicSequence.from_biotuner_list</font>).",
            CAPTION,
        )
    )

    story.append(PageBreak())

    # ------- Section 6: Validation
    story.append(Paragraph("6. Validation", H1))
    story.append(
        Paragraph(
            f"Total tests in <font face='Courier'>tests/harmonic_geometry/</font>: "
            f"<b>{_gather_test_count()}</b>. "
            "Coverage spans shape contracts, math invariants (unit-circle 1:1 "
            "Lissajous radius = 1, coprime closure to 1e-9, Star of David "
            "compound-set decomposition, harmonograph energy decay under "
            "non-zero damping, ratios_to_modes strategy correctness, "
            "rectangular Chladni mode symmetry, NaN masking outside "
            "circular plates, polygon eigenvalue ordering), "
            "save/load round-trips, and adapter integration with duck-typed "
            "biotuner objects.",
            BODY,
        )
    )
    story.append(Paragraph("Numerical tolerances", H3))
    story.append(
        Paragraph(
            "Closure tests use atol = 1e-9 (curves built analytically). "
            "Amplitude maxima use rtol = 1e-5 to absorb sampling "
            "discretization at the curve's apex. Decay tests assert late "
            "envelope < 20% of early — comfortably above floating-point noise.",
            BODY,
        )
    )
    story.append(Paragraph("Optional dependencies", H3))
    story.append(
        Paragraph(
            "<b>scikit-image</b>: required only for nodal extraction "
            "(<font face='Courier'>chladni_nodal_lines</font>, "
            "<font face='Courier'>chladni_nodal_surfaces</font>). Lazy "
            "import; clear ImportError when missing.<br/>"
            "<b>scikit-fem</b>: optional FEM solver for "
            "<font face='Courier'>chladni_field_polygon</font>; not yet "
            "wired through (FDM solver is the default).",
            BODY,
        )
    )

    doc.build(story)
    return PDF_PATH


if __name__ == "__main__":
    out = build_pdf()
    print(f"Wrote {out}")
