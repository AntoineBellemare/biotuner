"""
Generate the Phase 4 report PDF for biotuner.harmonic_geometry.fractal.

Produces docs/reports/harmonic_geometry_phase4.pdf demonstrating all five
deterministic fractal functions with varied inputs and parameter sweeps.

Run with the `biotuner` conda env:

    python docs/reports/generate_phase4_report.py
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

_WORKTREE_ROOT = str(Path(__file__).resolve().parents[1])
if _WORKTREE_ROOT not in sys.path:
    sys.path.insert(0, _WORKTREE_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Circle

from reportlab.lib.colors import HexColor, black, white
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
    HarmonicInput,
    continued_fraction_rectangles,
    farey_sequence_layout,
    ifs_harmonic,
    stern_brocot_tree,
    subharmonic_tree,
)

# ─────────────────────────────────────────────────── paths / constants ─────────

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR   = REPO_ROOT / "docs" / "reports"
FIG_DIR   = OUT_DIR / "figures"
PDF_PATH  = OUT_DIR / "harmonic_geometry_phase4.pdf"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FIG_WIDTH = 6.5
DPI       = 150

# ─────────────────────────────────────────────────── chord presets ─────────────

CHORDS: dict[str, HarmonicInput] = {
    "Major": HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)]),
    "Minor": HarmonicInput(ratios=[Fraction(1), Fraction(6, 5), Fraction(3, 2)]),
    "Dom7":  HarmonicInput(
        ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2), Fraction(7, 4)]
    ),
    "Maj7":  HarmonicInput(
        ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2), Fraction(15, 8)]
    ),
    "Sus4":  HarmonicInput(ratios=[Fraction(1), Fraction(4, 3), Fraction(3, 2)]),
    "Dim7":  HarmonicInput(
        ratios=[Fraction(1), Fraction(6, 5), Fraction(7, 5), Fraction(12, 7)]
    ),
}

PALETTE = [
    "#1f3b73", "#a23e2c", "#3a7a4d", "#7a5d24",
    "#5c2e7a", "#1f6b6b", "#b05f1a", "#3d4f60",
]

# ─────────────────────────────────────────────────── helpers ───────────────────


from biotuner.harmonic_geometry import plotting


def _save_fig(fig: plt.Figure, name: str) -> Path:
    return plotting.save_figure(fig, FIG_DIR / f"{name}.png", dpi=DPI)


def _ax_clean(ax, equal: bool = True) -> None:
    plotting.axis_clean(ax, equal=equal)


def _title_ax(ax, text: str, sub: str = "") -> None:
    plotting.title_ax(ax, text, sub)


# ══════════════════════════════════════════════════ figure builders ══════════════

# ──────────────────────────────────── Fig 1: Stern-Brocot layout comparison ────

def fig_stern_brocot_layouts() -> Path:
    """Hyperbolic vs. tree layout for Major chord, max_depth=6."""
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_WIDTH * 0.46))

    for ax, layout in zip(axes, ["hyperbolic", "tree"]):
        g = stern_brocot_tree(CHORDS["Major"], max_depth=6, layout=layout)
        coords = np.asarray(g.coordinates)
        edges  = np.asarray(g.edges)
        harm   = np.asarray(g.metadata["harmonicity"])

        # Background unit disk for hyperbolic layout
        if layout == "hyperbolic":
            disk = Circle((0, 0), 0.95, fill=False, edgecolor="#dddddd", lw=0.8)
            ax.add_patch(disk)

        # Edges in grey
        segs = coords[edges]
        lc = LineCollection(segs, colors="#cccccc", linewidths=0.45, zorder=1)
        ax.add_collection(lc)

        # Nodes coloured by harmonicity
        sc = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=harm, cmap="plasma", s=12, zorder=2,
            vmin=0, vmax=1, edgecolors="none",
        )

        if layout == "hyperbolic":
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
        else:
            pad = 0.05
            ax.set_xlim(coords[:, 0].min() - pad, coords[:, 0].max() + pad)
            ax.set_ylim(coords[:, 1].min() - 0.05, coords[:, 1].max() + 0.05)

        _ax_clean(ax)
        n_nodes = len(coords)
        _title_ax(ax, f"layout='{layout}'", f"{n_nodes} nodes")

    fig.colorbar(sc, ax=axes[1], fraction=0.04, pad=0.04, label="harmonicity")
    fig.suptitle("Stern-Brocot Tree — layout comparison (Major chord, depth=6)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    return _save_fig(fig, "p4_sbt_layouts")


# ──────────────────────────────────── Fig 2: Stern-Brocot depth sweep ──────────

def fig_stern_brocot_depth_sweep() -> Path:
    """Hyperbolic layout, depth 3–6; no chord input."""
    depths = [3, 4, 5, 6]
    fig, axes = plt.subplots(1, 4, figsize=(FIG_WIDTH, FIG_WIDTH * 0.27))
    cmap = matplotlib.colormaps["plasma"]

    for ax, d in zip(axes, depths):
        g = stern_brocot_tree(max_depth=d, layout="hyperbolic")
        coords = np.asarray(g.coordinates)
        edges  = np.asarray(g.edges)
        harm   = np.asarray(g.metadata["harmonicity"])

        disk = Circle((0, 0), 0.95, fill=False, edgecolor="#eeeeee", lw=0.6)
        ax.add_patch(disk)
        segs = coords[edges]
        lc = LineCollection(segs, colors="#dddddd", linewidths=0.4, zorder=1)
        ax.add_collection(lc)
        ax.scatter(coords[:, 0], coords[:, 1],
                   c=harm, cmap=cmap, s=max(2, 10 - d), zorder=2,
                   vmin=0, vmax=1, edgecolors="none")
        ax.set_xlim(-1.08, 1.08); ax.set_ylim(-1.08, 1.08)
        _ax_clean(ax)
        _title_ax(ax, f"depth={d}", f"{len(coords)} nodes")

    fig.suptitle("Stern-Brocot Tree — depth sweep (hyperbolic layout)",
                 fontsize=10, y=1.03)
    fig.tight_layout()
    return _save_fig(fig, "p4_sbt_depth")


# ──────────────────────────────────── Fig 3: SBT chord highlighting ────────────

def fig_stern_brocot_chord_highlight() -> Path:
    """Hyperbolic layout, max_depth=7; highlight chord tones from 6 different chords."""
    names = ["Major", "Minor", "Dom7", "Maj7", "Sus4", "Dim7"]
    fig, axes = plt.subplots(2, 3, figsize=(FIG_WIDTH, FIG_WIDTH * 0.72))
    cmap_base = matplotlib.colormaps["Greys"]
    highlights = PALETTE

    for ax, name, hi_color in zip(axes.ravel(), names, highlights):
        g = stern_brocot_tree(CHORDS[name], max_depth=7, layout="hyperbolic")
        coords = np.asarray(g.coordinates)
        edges  = np.asarray(g.edges)
        harm   = np.asarray(g.metadata["harmonicity"])
        dist   = np.asarray(g.metadata.get("nearest_input_dist_cents",
                                           [100.0] * len(coords)))

        disk = Circle((0, 0), 0.95, fill=False, edgecolor="#eeeeee", lw=0.5)
        ax.add_patch(disk)
        segs = coords[edges]
        lc = LineCollection(segs, colors="#dddddd", linewidths=0.3, zorder=1)
        ax.add_collection(lc)

        # Background nodes in grey, foreground chord-tone nodes highlighted
        close_mask = dist < 15.0   # within 15 cents of a chord tone
        ax.scatter(coords[~close_mask, 0], coords[~close_mask, 1],
                   c=harm[~close_mask], cmap=cmap_base, s=3, zorder=2,
                   vmin=0, vmax=1, edgecolors="none", alpha=0.5)
        if close_mask.any():
            ax.scatter(coords[close_mask, 0], coords[close_mask, 1],
                       color=hi_color, s=20, zorder=3, edgecolors="white", lw=0.4)

        ax.set_xlim(-1.08, 1.08); ax.set_ylim(-1.08, 1.08)
        _ax_clean(ax)
        n_close = int(close_mask.sum())
        _title_ax(ax, name, f"{n_close} chord tones highlighted")

    fig.suptitle("Stern-Brocot Tree — chord-tone highlighting (depth=7, ≤15 cents)",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    return _save_fig(fig, "p4_sbt_chords")


# ──────────────────────────────────── Fig 4: Continued-fraction gallery ────────

def fig_cf_ratio_gallery() -> Path:
    """Six classic JI ratios, each shown as a CF rectangle tiling."""
    ratios_labels = [
        (Fraction(3, 2),  "3/2 — perfect fifth"),
        (Fraction(5, 4),  "5/4 — major third"),
        (Fraction(7, 4),  "7/4 — harmonic seventh"),
        (Fraction(9, 8),  "9/8 — major tone"),
        (Fraction(16, 9), "16/9 — minor seventh"),
        (Fraction(5, 3),  "5/3 — major sixth"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(FIG_WIDTH, FIG_WIDTH * 0.7))
    cmap = matplotlib.colormaps["tab10"]

    for ax, (ratio, label) in zip(axes.ravel(), ratios_labels):
        g = continued_fraction_rectangles(ratio)
        rects = g.coordinates   # list of (4,2) polygons
        colors = [cmap(i / max(len(rects) - 1, 1)) for i in range(len(rects))]
        polys = [r.tolist() for r in rects]
        pc = PolyCollection(polys, facecolors=colors, edgecolors="white", linewidths=0.8)
        ax.add_collection(pc)
        all_pts = np.vstack(rects)
        ax.set_xlim(all_pts[:, 0].min() - 0.01, all_pts[:, 0].max() + 0.01)
        ax.set_ylim(all_pts[:, 1].min() - 0.01, all_pts[:, 1].max() + 0.01)
        _ax_clean(ax, equal=False)
        n = g.metadata["n_squares"]
        _title_ax(ax, label, f"{n} squares in CF expansion")

    fig.suptitle("Continued-Fraction Rectangles — JI ratio gallery",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    return _save_fig(fig, "p4_cf_gallery")


# ──────────────────────────────────── Fig 5: CF depth sweep ────────────────────

def fig_cf_depth_sweep() -> Path:
    """Ratio 7/4 at depth 2, 4, 6, full."""
    depths = [2, 4, 6, 20]
    labels = ["depth=2", "depth=4", "depth=6", "full (depth=20)"]
    fig, axes = plt.subplots(1, 4, figsize=(FIG_WIDTH, FIG_WIDTH * 0.28))
    ratio = Fraction(7, 4)
    cmap  = matplotlib.colormaps["tab10"]

    for ax, d, lbl in zip(axes, depths, labels):
        g = continued_fraction_rectangles(ratio, depth=d)
        rects = g.coordinates
        colors = [cmap(i / max(len(rects) - 1, 1)) for i in range(len(rects))]
        pc = PolyCollection([r.tolist() for r in rects],
                            facecolors=colors, edgecolors="white", linewidths=0.8)
        ax.add_collection(pc)
        all_pts = np.vstack(rects)
        ax.set_xlim(all_pts[:, 0].min() - 0.01, all_pts[:, 0].max() + 0.01)
        ax.set_ylim(all_pts[:, 1].min() - 0.01, all_pts[:, 1].max() + 0.01)
        _ax_clean(ax, equal=False)
        _title_ax(ax, lbl, f"{g.metadata['n_squares']} squares")

    fig.suptitle("Continued-Fraction Rectangles — depth sweep (ratio 7/4)",
                 fontsize=10, y=1.03)
    fig.tight_layout()
    return _save_fig(fig, "p4_cf_depth")


# ──────────────────────────────────── Fig 6: Farey circle gallery ──────────────

def fig_farey_circle_gallery() -> Path:
    """Farey circle layout for orders 3, 5, 7, 10, 15, 20."""
    orders = [3, 5, 7, 10, 15, 20]
    fig, axes = plt.subplots(2, 3, figsize=(FIG_WIDTH, FIG_WIDTH * 0.72))

    for ax, order, color in zip(axes.ravel(), orders, PALETTE):
        g = farey_sequence_layout(order, layout="circle")
        coords  = np.asarray(g.coordinates)
        weights = np.asarray(g.weights)   # 1/denominator

        # Draw unit circle
        ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor="#eeeeee", lw=0.7))

        # Size and alpha by weight (simpler fractions are larger/brighter)
        s    = weights * 120 + 4
        alph = weights * 0.8 + 0.15
        ax.scatter(coords[:, 0], coords[:, 1],
                   s=s, c=color, alpha=np.clip(alph, 0, 1), edgecolors="none", zorder=2)

        ax.set_xlim(-1.18, 1.18); ax.set_ylim(-1.18, 1.18)
        _ax_clean(ax)
        _title_ax(ax, f"F_{order}", f"{g.metadata['n_terms']} terms")

    fig.suptitle("Farey Sequence Layout — circle layout, orders 3–20",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    return _save_fig(fig, "p4_farey_circle")


# ──────────────────────────────────── Fig 7: Farey line layout ─────────────────

def fig_farey_line_gallery() -> Path:
    """Farey line layout for orders 5, 10, 20, 30; stacked."""
    orders = [5, 10, 20, 30]
    fig, axes = plt.subplots(4, 1, figsize=(FIG_WIDTH, FIG_WIDTH * 0.55),
                             gridspec_kw={"hspace": 0.55})

    for ax, order, color in zip(axes, orders, PALETTE):
        g = farey_sequence_layout(order, layout="line")
        coords  = np.asarray(g.coordinates)
        weights = np.asarray(g.weights)

        s    = weights * 100 + 5
        alph = weights * 0.8 + 0.15
        ax.scatter(coords[:, 0], coords[:, 1],
                   s=s, c=color, alpha=np.clip(alph, 0, 1), edgecolors="none", zorder=2)
        ax.axhline(0, color="#eeeeee", lw=0.5)
        ax.set_xlim(-1.08, 1.08)
        ax.set_ylim(-0.5, 0.5)
        _ax_clean(ax, equal=False)
        ax.set_ylabel(f"F_{order}\n({g.metadata['n_terms']})", fontsize=7,
                      rotation=0, ha="right", va="center", labelpad=8)

    fig.suptitle("Farey Sequence — line layout, orders 5–30 (dot size ∝ 1/denominator)",
                 fontsize=10, y=1.01)
    return _save_fig(fig, "p4_farey_line")


# ──────────────── Fig 8: Subharmonic — polar layout (chord-differentiated) ─────

def fig_subharmonic_polar() -> Path:
    """Polar layout: each input peak gets its own angular sector.

    With the depth layout every chord produced an identical-looking dendrogram;
    the polar layout fans each root-peak into its own sector and colours all
    descendants by their root, so different chords produce visibly different
    shapes.
    """
    names = ["Major", "Minor", "Dom7", "Sus4"]
    fig, axes = plt.subplots(1, 4, figsize=(FIG_WIDTH, FIG_WIDTH * 0.28))
    palette_cmap = matplotlib.colormaps["tab10"]
    for ax, name in zip(axes, names):
        g = subharmonic_tree(CHORDS[name], depth=4, n_harmonics=4, layout="polar")
        coords = np.asarray(g.coordinates)
        edges  = np.asarray(g.edges)
        freqs  = np.asarray(g.weights)
        roots  = np.asarray(g.metadata["root_index_per_node"])
        n_roots = int(roots.max()) + 1

        # Draw a faint reference disk
        ax.add_patch(Circle((0, 0), 1.0, fill=False,
                             edgecolor="#eeeeee", lw=0.4))

        # Edges coloured by descendant's root
        edge_colors = [palette_cmap((roots[v] / max(n_roots - 1, 1))) for _, v in edges]
        if edges.size:
            ax.add_collection(LineCollection(
                coords[edges], colors=edge_colors, linewidths=0.7, alpha=0.65,
                zorder=1,
            ))

        # Node sizes scale with log-frequency; colours by root
        log_f = np.log1p(freqs)
        sizes = log_f / (log_f.max() + 1e-9) * 50 + 6
        node_colors = palette_cmap(roots / max(n_roots - 1, 1))
        ax.scatter(coords[:, 0], coords[:, 1], s=sizes,
                    c=node_colors, edgecolors="white", linewidths=0.3,
                    zorder=2)
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
        _ax_clean(ax)
        _title_ax(ax, name,
                   f"{g.metadata['n_nodes']} nodes / {n_roots} sectors")
    fig.suptitle("Subharmonic Tree — polar layout "
                 "(one angular sector per chord-tone; colour by root)",
                 fontsize=10, y=1.04)
    fig.tight_layout()
    return _save_fig(fig, "p4_sub_polar")


# ──────────────────────────────────── Fig 10: IFS chord gallery ────────────────

def fig_ifs_chord_gallery() -> Path:
    """6 chords with 'ratio_inverse' contraction; 30k points each."""
    names  = ["Major", "Minor", "Dom7", "Maj7", "Sus4", "Dim7"]
    rng    = np.random.default_rng(42)
    fig, axes = plt.subplots(2, 3, figsize=(FIG_WIDTH, FIG_WIDTH * 0.72))

    for ax, name, color in zip(axes.ravel(), names, PALETTE):
        g = ifs_harmonic(CHORDS[name], n_points=30_000, contraction="ratio_inverse", rng=rng)
        pts = np.asarray(g.coordinates)
        ax.scatter(pts[:, 0], pts[:, 1], s=0.3, c=color, alpha=0.25,
                   edgecolors="none", rasterized=True)
        span = g.metadata["span"] + 0.05
        ax.set_xlim(-span, span); ax.set_ylim(-span, span)
        _ax_clean(ax)
        n_maps = g.metadata["n_maps"]
        scales = [f"{s:.2f}" for s in g.metadata["scales"]]
        _title_ax(ax, name, f"{n_maps} maps · scales: {', '.join(scales)}")

    fig.suptitle("IFS Harmonic Attractor — chord gallery (ratio_inverse contraction, 30k pts)",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    return _save_fig(fig, "p4_ifs_chords")


# ──────────────────────────────────── Fig 11: IFS contraction sweep ────────────

def fig_ifs_contraction_sweep() -> Path:
    """Dom7 with all three contraction modes; 50k points."""
    contractions = ["ratio_inverse", "log_ratio", "fixed_half"]
    labels       = ["ratio_inverse\n(s = 1/r)", "log_ratio\n(s = 1/(1+ln r))", "fixed_half\n(s = 0.5)"]
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH, FIG_WIDTH * 0.38))
    for ax, mode, lbl, color in zip(axes, contractions, labels, PALETTE):
        g = ifs_harmonic(CHORDS["Dom7"], n_points=50_000, contraction=mode, rng=rng)
        pts  = np.asarray(g.coordinates)
        span = g.metadata["span"] + 0.05
        ax.scatter(pts[:, 0], pts[:, 1], s=0.3, c=color, alpha=0.25,
                   edgecolors="none", rasterized=True)
        ax.set_xlim(-span, span); ax.set_ylim(-span, span)
        _ax_clean(ax)
        _title_ax(ax, lbl)

    fig.suptitle("IFS Harmonic — contraction mode comparison (Dom7, 50k pts)",
                 fontsize=10, y=1.03)
    fig.tight_layout()
    return _save_fig(fig, "p4_ifs_contractions")


# ──────────── Fig 11b: IFS density heatmap with zoom insets ───────────────────

def fig_ifs_density_zoom() -> Path:
    """High-density IFS attractor rendered as a 2-D density heatmap with insets."""
    rng = np.random.default_rng(11)
    g = ifs_harmonic(CHORDS["Dom7"], n_points=200_000,
                     contraction="ratio_inverse", rng=rng)
    pts  = np.asarray(g.coordinates)
    span = float(g.metadata["span"]) + 0.05

    # 2-D histogram of point visit counts → log density
    n_bins = 512
    H, x_edges, y_edges = np.histogram2d(
        pts[:, 0], pts[:, 1],
        bins=n_bins, range=[[-span, span], [-span, span]],
    )
    H_log = np.log1p(H.T)   # transpose so y indexes rows for imshow

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH * 0.5))
    gs   = fig.add_gridspec(2, 4, width_ratios=[2, 1, 1, 1])

    ax_main = fig.add_subplot(gs[:, 0])
    ax_main.imshow(H_log, origin="lower", cmap="inferno",
                    extent=[-span, span, -span, span], aspect="equal")
    ax_main.set_xticks([]); ax_main.set_yticks([])
    _title_ax(ax_main, "Dom7 attractor", "200k pts → log-density heatmap")

    insets = [
        ((-span * 0.55, -span * 0.10, -span * 0.15,  span * 0.30), "1"),
        (( span * 0.10,  span * 0.55, -span * 0.30,  span * 0.15), "2"),
        ((-span * 0.10,  span * 0.35,  span * 0.30,  span * 0.75), "3"),
    ]
    cmap = matplotlib.colormaps["inferno"]
    for k, (rect, label) in enumerate(insets):
        x0, x1, y0, y1 = rect
        ax_main.add_patch(plt.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            edgecolor="#fff", lw=0.7, fill=False, zorder=3,
        ))
        ax_main.text(x0, y1 + span * 0.01, label, color="#fff",
                     fontsize=8, ha="left", va="bottom")
        # sub-bin region
        ix0 = max(int((x0 + span) / (2 * span) * n_bins), 0)
        ix1 = min(int((x1 + span) / (2 * span) * n_bins), n_bins)
        iy0 = max(int((y0 + span) / (2 * span) * n_bins), 0)
        iy1 = min(int((y1 + span) / (2 * span) * n_bins), n_bins)
        sub = H_log[iy0:iy1, ix0:ix1]
        if k < 2:
            ax = fig.add_subplot(gs[k // 2, 1 + (k % 2)])
        else:
            ax = fig.add_subplot(gs[1, 3])
        if sub.size:
            ax.imshow(sub, origin="lower", cmap="inferno",
                      extent=[x0, x1, y0, y1], aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        _title_ax(ax, f"inset {label}", f"{int(H[ix0:ix1, iy0:iy1].sum())} pts")
    fig.suptitle("ifs_harmonic — log-density rendering with zoom insets",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    return _save_fig(fig, "p4_ifs_density")


# ──────────────────────────────────── Fig 12: combined showcase ────────────────

def fig_combined_showcase() -> Path:
    """One-line snapshot of all 5 Phase 4 functions for the Major chord."""
    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(1, 5, figsize=(FIG_WIDTH, FIG_WIDTH * 0.26))

    # 1 Stern-Brocot
    ax = axes[0]
    g  = stern_brocot_tree(CHORDS["Major"], max_depth=5, layout="hyperbolic")
    coords = np.asarray(g.coordinates); edges = np.asarray(g.edges)
    harm   = np.asarray(g.metadata["harmonicity"])
    ax.add_patch(Circle((0, 0), 0.95, fill=False, edgecolor="#eeeeee", lw=0.6))
    lc = LineCollection(coords[edges], colors="#dddddd", linewidths=0.4, zorder=1)
    ax.add_collection(lc)
    ax.scatter(coords[:, 0], coords[:, 1], c=harm, cmap="plasma", s=5, zorder=2,
               vmin=0, vmax=1, edgecolors="none")
    ax.set_xlim(-1.08, 1.08); ax.set_ylim(-1.08, 1.08)
    _ax_clean(ax); ax.set_title("Stern-Brocot", fontsize=7, pad=2)

    # 2 CF rectangles
    ax = axes[1]
    g  = continued_fraction_rectangles(Fraction(3, 2))
    rects = g.coordinates
    cmap  = matplotlib.colormaps["tab10"]
    colors = [cmap(i / max(len(rects) - 1, 1)) for i in range(len(rects))]
    pc = PolyCollection([r.tolist() for r in rects],
                        facecolors=colors, edgecolors="white", linewidths=0.8)
    ax.add_collection(pc)
    all_pts = np.vstack(rects)
    ax.set_xlim(all_pts[:, 0].min() - 0.01, all_pts[:, 0].max() + 0.01)
    ax.set_ylim(all_pts[:, 1].min() - 0.01, all_pts[:, 1].max() + 0.01)
    _ax_clean(ax, equal=False)
    ax.set_title("CF Rectangles", fontsize=7, pad=2)

    # 3 Farey
    ax = axes[2]
    g  = farey_sequence_layout(10, layout="circle")
    coords  = np.asarray(g.coordinates)
    weights = np.asarray(g.weights)
    ax.add_patch(Circle((0, 0), 1.0, fill=False, edgecolor="#eeeeee", lw=0.7))
    ax.scatter(coords[:, 0], coords[:, 1],
               s=weights * 80 + 3, c=PALETTE[4], alpha=0.7, edgecolors="none")
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
    _ax_clean(ax); ax.set_title("Farey (F10)", fontsize=7, pad=2)

    # 4 Subharmonic tree
    ax = axes[3]
    g  = subharmonic_tree(CHORDS["Major"], depth=3, n_harmonics=4)
    coords = np.asarray(g.coordinates); edges = np.asarray(g.edges)
    lc = LineCollection(coords[edges], colors="#cccccc", linewidths=0.5, zorder=1)
    ax.add_collection(lc)
    ax.scatter(coords[:, 0], coords[:, 1], s=12, c=PALETTE[2], edgecolors="none", zorder=2)
    ax.set_xlim(coords[:, 0].min() - 0.08, coords[:, 0].max() + 0.08)
    ax.set_ylim(coords[:, 1].min() - 0.05, coords[:, 1].max() + 0.05)
    _ax_clean(ax, equal=False); ax.set_title("Subharmonic Tree", fontsize=7, pad=2)

    # 5 IFS
    ax = axes[4]
    g  = ifs_harmonic(CHORDS["Major"], n_points=25_000, rng=rng)
    pts  = np.asarray(g.coordinates)
    span = g.metadata["span"] + 0.05
    ax.scatter(pts[:, 0], pts[:, 1], s=0.3, c=PALETTE[0], alpha=0.25,
               edgecolors="none", rasterized=True)
    ax.set_xlim(-span, span); ax.set_ylim(-span, span)
    _ax_clean(ax); ax.set_title("IFS Attractor", fontsize=7, pad=2)

    fig.suptitle("Phase 4 — all five deterministic fractal functions (Major chord)",
                 fontsize=10, y=1.04)
    fig.tight_layout()
    return _save_fig(fig, "p4_combined_showcase")


# ══════════════════════════════════════════════════ PDF assembly ════════════════


def _styles():
    base = getSampleStyleSheet()
    H1 = ParagraphStyle(
        "H1", parent=base["Heading1"], fontSize=16, leading=20,
        textColor=HexColor("#1f3b73"), spaceBefore=18, spaceAfter=8,
    )
    H2 = ParagraphStyle(
        "H2", parent=base["Heading2"], fontSize=12, leading=16,
        textColor=HexColor("#a23e2c"), spaceBefore=14, spaceAfter=6,
    )
    BODY = ParagraphStyle(
        "Body", parent=base["Normal"], fontSize=9.5, leading=14,
        spaceAfter=6, textColor=HexColor("#333333"),
    )
    CAPTION = ParagraphStyle(
        "Caption", parent=base["Normal"], fontSize=8, leading=11,
        alignment=TA_CENTER, textColor=HexColor("#666666"), spaceAfter=10,
    )
    CODE = ParagraphStyle(
        "Code", parent=base["Code"], fontSize=7.5, leading=11,
        leftIndent=18, spaceAfter=8,
        textColor=HexColor("#1a1a2e"), backColor=HexColor("#f2f2f5"),
    )
    return H1, H2, BODY, CAPTION, CODE


def _img(path: Path, width_in: float = FIG_WIDTH) -> Image:
    return Image(str(path), width=width_in * inch, height=9 * inch, kind="proportional")


def build_pdf(figures: dict[str, Path]) -> None:
    H1, H2, BODY, CAPTION, CODE = _styles()

    doc = SimpleDocTemplate(
        str(PDF_PATH), pagesize=LETTER,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
    )
    story = []

    # ── Cover ──────────────────────────────────────────────────────────────────
    story += [
        Spacer(1, 0.4 * inch),
        Paragraph("biotuner.harmonic_geometry", ParagraphStyle(
            "sub_cover", parent=H2, fontSize=11, textColor=HexColor("#888888"),
        )),
        Paragraph("Phase 4 Report: Deterministic Fractal Structures", ParagraphStyle(
            "cover_title", parent=H1, fontSize=22, leading=28, spaceBefore=4,
        )),
        Paragraph(
            "Five functions that construct classical number-theoretic and IFS structures "
            "driven by harmonic ratios.  All outputs are <b>GeometryData</b> objects "
            "ready for downstream rendering or analysis.",
            BODY,
        ),
        Spacer(1, 0.15 * inch),
        _img(figures["combined"], 6.0),
        Paragraph(
            "Figure 1. All five Phase 4 generators applied to the Just-Intonation Major chord "
            "(1, 5/4, 3/2).  Left to right: Stern-Brocot hyperbolic tree (nodes coloured by "
            "harmonicity) · continued-fraction rectangle tiling of 3/2 · Farey circle F₁₀ · "
            "subharmonic expansion tree · IFS chaos-game attractor.",
            CAPTION,
        ),
        PageBreak(),
    ]

    # ── Section 1: stern_brocot_tree ───────────────────────────────────────────
    story += [
        Paragraph("1. stern_brocot_tree", H1),
        Paragraph(
            "The Stern-Brocot tree is the canonical enumeration of all positive rationals "
            "via successive mediants.  At each node, the mediant of its left and right "
            "bounding fractions is computed: mediant((a,b),(c,d)) = (a+c, b+d).  The tree "
            "thus encodes every reduced fraction exactly once, organized by complexity "
            "(denominator size grows with depth).",
            BODY,
        ),
        Paragraph(
            "Each node is annotated with a <b>harmonicity score</b> from "
            "<i>biotuner.metrics.dyad_similarity</i>.  When a <b>HarmonicInput</b> is "
            "provided, the cents distance from each tree node to the nearest chord tone "
            "is also recorded, enabling chord highlighting.",
            BODY,
        ),
        Paragraph(
            "<b>Signature:</b>  "
            "stern_brocot_tree(input=None, max_depth=6, layout='hyperbolic')",
            CODE,
        ),
        _img(figures["sbt_layouts"]),
        Paragraph(
            "Figure 2. Hyperbolic (Poincaré disk) vs. flat dendrogram layout for the Major "
            "chord at depth=6.  Node colour encodes harmonicity (plasma scale, 0=low→1=high).",
            CAPTION,
        ),
        _img(figures["sbt_depth"]),
        Paragraph(
            "Figure 3. Depth sweep without chord input (depth 3–6, hyperbolic layout).  "
            "Node count grows as 2^depth − 1.",
            CAPTION,
        ),
        Paragraph(
            "Providing a chord highlights where the chord's intervals land within the "
            "rational lattice:",
            BODY,
        ),
        _img(figures["sbt_chords"]),
        Paragraph(
            "Figure 4. Six JI chords highlighted on the Stern-Brocot tree (depth=7).  "
            "Coloured dots mark tree nodes within 15 cents of a chord tone; grey dots are "
            "all other rationals.",
            CAPTION,
        ),
        PageBreak(),
    ]

    # ── Section 2: continued_fraction_rectangles ───────────────────────────────
    story += [
        Paragraph("2. continued_fraction_rectangles", H1),
        Paragraph(
            "Visualizes the continued-fraction expansion of a ratio p/q as a recursive "
            "Euclidean decomposition: the p×q rectangle is tiled by the largest possible "
            "squares of side min(p,q), and the residual strip is rotated 90° and tiled "
            "again.  The sequence of square sizes encodes the continued-fraction coefficients "
            "[a₀; a₁, a₂, …] of p/q.",
            BODY,
        ),
        Paragraph(
            "Ratios with small CF coefficients (like the perfect fifth 3/2 = [1; 2]) "
            "produce simple tilings with few large squares.  Complex ratios (like 9/8 "
            "= [1; 8]) produce many small squares.",
            BODY,
        ),
        Paragraph(
            "<b>Signature:</b>  continued_fraction_rectangles(ratio, depth=10)",
            CODE,
        ),
        _img(figures["cf_gallery"]),
        Paragraph(
            "Figure 5. CF rectangle tilings for six JI ratios.  Each colour represents "
            "one step (square) of the continued-fraction expansion.  The bounding "
            "rectangle is normalized to width=1.",
            CAPTION,
        ),
        _img(figures["cf_depth"]),
        Paragraph(
            "Figure 6. Depth sweep for 7/4 = [1; 1, 3].  The expansion terminates "
            "after 3 steps; higher depth limits have no effect for exact rationals.",
            CAPTION,
        ),
        PageBreak(),
    ]

    # ── Section 3: farey_sequence_layout ──────────────────────────────────────
    story += [
        Paragraph("3. farey_sequence_layout", H1),
        Paragraph(
            "The Farey sequence F_n is the sorted set of all reduced fractions p/q in "
            "[0, 1] with denominator ≤ n.  It has deep connections to the Stern-Brocot "
            "tree (F_n is an in-order traversal of the tree down to depth n) and to "
            "modular arithmetic.",
            BODY,
        ),
        Paragraph(
            "Two layouts are supported: <b>circle</b> maps each fraction to an angle "
            "2π·p/q on the unit circle, revealing clustering patterns and periodicity; "
            "<b>line</b> places fractions on the interval [−1, 1], useful for comparing "
            "density across orders.  Node size encodes 1/denominator — simpler fractions "
            "appear larger.",
            BODY,
        ),
        Paragraph(
            "<b>Signature:</b>  farey_sequence_layout(order, layout='circle')",
            CODE,
        ),
        _img(figures["farey_circle"]),
        Paragraph(
            "Figure 7. Farey circle layouts for orders 3–20.  Dot size ∝ 1/denominator; "
            "unit fractions 1/n cluster near 2π/n on the circle.",
            CAPTION,
        ),
        _img(figures["farey_line"]),
        Paragraph(
            "Figure 8. Farey line layouts stacked for orders 5–30.  The increasing "
            "density of the sequence with order is clearly visible.",
            CAPTION,
        ),
        PageBreak(),
    ]

    # ── Section 4: subharmonic_tree ────────────────────────────────────────────
    story += [
        Paragraph("4. subharmonic_tree", H1),
        Paragraph(
            "Recursively expands each input peak frequency f into its first k subharmonics "
            "f/2, f/3, …, f/(k+1), then expands each child the same way up to a given depth.  "
            "The result is a multi-level harmonic descent tree rooted at the input peaks.  "
            "Branches with frequency below <i>min_freq</i> are pruned.",
            BODY,
        ),
        Paragraph(
            "Nodes are laid out with depth on the y-axis and log-frequency rank on the "
            "x-axis within each level.  Node size scales with the logarithm of the node "
            "frequency so root (higher-frequency) nodes appear larger.",
            BODY,
        ),
        Paragraph(
            "<b>Signature:</b>  subharmonic_tree(input, depth=4, n_harmonics=5, "
            "min_freq=0.1, layout='depth' | 'polar')",
            CODE,
        ),
        Paragraph(
            "The polar layout fans each root-peak into its own angular sector and "
            "colours descendants by root. Depth becomes radial distance, so the "
            "chord's interval structure shapes the silhouette directly:",
            BODY,
        ),
        _img(figures["sub_polar"]),
        Paragraph(
            "Figure 9. Subharmonic trees with <font face='Courier'>layout='polar'</font>. "
            "Each root-peak owns a 2π/N angular sector; depth becomes radial distance. "
            "Different chords produce visibly different fan-out signatures.",
            CAPTION,
        ),
        PageBreak(),
    ]

    # ── Section 5: ifs_harmonic ────────────────────────────────────────────────
    story += [
        Paragraph("5. ifs_harmonic", H1),
        Paragraph(
            "Implements the chaos game for an iterated-function system (IFS) whose "
            "contractions are derived from the input ratios.  Each ratio r_i defines an "
            "affine map:   z → s_i · z + (1 − s_i) · v_i,  where v_i is the i-th vertex "
            "of a regular N-gon on the unit disk.  The contraction factor s_i is derived "
            "from the ratio according to the chosen <i>contraction</i> mode.",
            BODY,
        ),
        Paragraph(
            "The chaos-game orbit converges to a strange attractor whose geometry "
            "depends on both the vertex positions (N-gon, determined by n_components) "
            "and the per-map contraction scales (determined by the ratios).",
            BODY,
        ),
        Paragraph(
            "<b>Signature:</b>  "
            "ifs_harmonic(input, n_points=50_000, contraction='ratio_inverse', "
            "transient=200, rng=None)",
            CODE,
        ),
        _img(figures["ifs_chords"]),
        Paragraph(
            "Figure 11. IFS attractors for six JI chords (ratio_inverse contraction, 30k pts).  "
            "Each chord's N-gon layout and per-ratio contraction scales produce a "
            "distinct attractor geometry.",
            CAPTION,
        ),
        Paragraph(
            "Three contraction modes provide qualitatively different attractors.  "
            "ratio_inverse scales inversely with ratio size; log_ratio is more uniform; "
            "fixed_half gives a classical Sierpinski-like attractor:",
            BODY,
        ),
        _img(figures["ifs_contractions"]),
        Paragraph(
            "Figure 12. Contraction mode comparison for the Dom7 chord (50k pts).  "
            "ratio_inverse (left) preserves interval-content in the contraction geometry; "
            "fixed_half (right) is equivalent to the classical chaos game.",
            CAPTION,
        ),
        Paragraph(
            "At low point counts the attractor looks like noise. Cranking density "
            "200 000+ points and rendering as a log-density heatmap reveals the "
            "inner structure; zoom insets at the densest regions surface fine "
            "self-similar detail:",
            BODY,
        ),
        _img(figures["ifs_density"]),
        Paragraph(
            "Figure 12b. <font face='Courier'>ifs_harmonic</font> Dom7 attractor "
            "rendered as a 200k-point 2-D histogram (log scale, inferno cmap). "
            "Three rectangles on the main panel mark sub-regions shown at higher "
            "magnification.",
            CAPTION,
        ),
        PageBreak(),
    ]

    # ── Parameter reference table ──────────────────────────────────────────────
    story += [
        Paragraph("Parameter Reference", H1),
        Spacer(1, 0.1 * inch),
    ]

    table_data = [
        ["Function",                    "Parameter",    "Default",       "Range / Notes"],
        ["stern_brocot_tree",           "max_depth",    "6",             "int ≥ 1; nodes grow as 2^depth − 1"],
        ["",                            "layout",       "'hyperbolic'",  "'hyperbolic' or 'tree'"],
        ["",                            "input",        "None",          "HarmonicInput — enables chord highlighting"],
        ["continued_fraction_rectangles","depth",       "10",            "int ≥ 1; terminates early for rationals"],
        ["",                            "ratio",        "—",             "Fraction / int / float / (int,int)"],
        ["farey_sequence_layout",       "order",        "—",             "int ≥ 1"],
        ["",                            "layout",       "'circle'",      "'circle' or 'line'"],
        ["subharmonic_tree",            "depth",        "4",             "int ≥ 1"],
        ["",                            "n_harmonics",  "5",             "int ≥ 1"],
        ["",                            "min_freq",     "0.1",           "float > 0; prunes low-frequency branches"],
        ["ifs_harmonic",                "n_points",     "50 000",        "int ≥ 100"],
        ["",                            "contraction",  "'ratio_inverse'","'ratio_inverse', 'log_ratio', 'fixed_half'"],
        ["",                            "transient",    "200",           "int ≥ 0; warm-up iterations discarded"],
        ["",                            "rng",          "None",          "np.random.Generator for reproducibility"],
    ]

    col_w = [1.75 * inch, 1.45 * inch, 1.05 * inch, 2.25 * inch]
    tbl = Table(table_data, colWidths=col_w)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  HexColor("#1f3b73")),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [HexColor("#f8f8fb"), white]),
        ("FONTSIZE",      (0, 1), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID",          (0, 0), (-1, -1), 0.25, HexColor("#cccccc")),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.2 * inch))

    story += [Paragraph("Return Type Summary", H2)]
    ret_data = [
        ["Function",                      "geom_type",       "coordinates",            "Notable metadata"],
        ["stern_brocot_tree",             "tree",            "(N×2) 2-D layout",       "harmonicity, ratios, depth_per_node"],
        ["continued_fraction_rectangles", "polygon_set",     "list of (4×2) rects",    "n_squares, inverted"],
        ["farey_sequence_layout",         "point_cloud_2d",  "(M×2) circle/line pts",  "n_terms, fractions"],
        ["subharmonic_tree",              "tree",            "(K×2) depth layout",     "n_nodes, frequencies_hz"],
        ["ifs_harmonic",                  "point_cloud_2d",  "(n_points×2) attractor", "n_maps, scales, probabilities"],
    ]
    col_w2 = [1.75 * inch, 0.95 * inch, 1.55 * inch, 2.25 * inch]
    tbl2 = Table(ret_data, colWidths=col_w2)
    tbl2.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  HexColor("#a23e2c")),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0),  8),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [HexColor("#fdf8f6"), white]),
        ("FONTSIZE",      (0, 1), (-1, -1), 8),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID",          (0, 0), (-1, -1), 0.25, HexColor("#cccccc")),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl2)

    doc.build(story)
    print(f"[✓] PDF written → {PDF_PATH}")


# ══════════════════════════════════════════════════════════ main ════════════════

def main():
    print("Rendering figures…")
    figures = {
        "combined":         fig_combined_showcase(),
        "sbt_layouts":      fig_stern_brocot_layouts(),
        "sbt_depth":        fig_stern_brocot_depth_sweep(),
        "sbt_chords":       fig_stern_brocot_chord_highlight(),
        "cf_gallery":       fig_cf_ratio_gallery(),
        "cf_depth":         fig_cf_depth_sweep(),
        "farey_circle":     fig_farey_circle_gallery(),
        "farey_line":       fig_farey_line_gallery(),
        "sub_polar":        fig_subharmonic_polar(),
        "ifs_chords":       fig_ifs_chord_gallery(),
        "ifs_contractions": fig_ifs_contraction_sweep(),
        "ifs_density":      fig_ifs_density_zoom(),
    }
    print("Building PDF…")
    build_pdf(figures)


if __name__ == "__main__":
    main()
