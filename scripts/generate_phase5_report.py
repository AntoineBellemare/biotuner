"""
Generate the Phase 5 report PDF for biotuner.harmonic_geometry.generative.

Produces docs/reports/harmonic_geometry_phase5.pdf demonstrating all five
generative fractal functions with varied inputs and parameter sweeps.

Run with the `biotuner` conda env:

    python docs/reports/generate_phase5_report.py
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
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

from reportlab.lib.colors import HexColor, black, white
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from biotuner.harmonic_geometry import (
    HarmonicInput,
    lsystem_from_ratios,
    recursive_polygon,
    self_similar_tuning,
)

# ──────────────────────────────────────────────────── paths / constants ────────

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "docs" / "reports"
FIG_DIR = OUT_DIR / "figures"
PDF_PATH = OUT_DIR / "harmonic_geometry_phase5.pdf"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FIG_WIDTH = 6.5
DPI = 150

# ──────────────────────────────────────────────────── chord presets ────────────

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
    "Aug":   HarmonicInput(
        ratios=[Fraction(1), Fraction(5, 4), Fraction(8, 5)]
    ),
    "Dim7":  HarmonicInput(
        ratios=[Fraction(1), Fraction(6, 5), Fraction(7, 5), Fraction(12, 7)]
    ),
}

# ──────────────────────────────────────────────────── helpers ──────────────────

PALETTE = {
    "dark_blue":   "#1f3b73",
    "red":         "#a23e2c",
    "green":       "#3a7a4d",
    "gold":        "#7a5d24",
    "purple":      "#5c2e7a",
    "teal":        "#1f6b6b",
    "orange":      "#b05f1a",
    "slate":       "#3d4f60",
}
CHORD_COLORS = list(PALETTE.values())


from biotuner.harmonic_geometry import plotting


def _save_fig(fig: plt.Figure, name: str) -> Path:
    return plotting.save_figure(fig, FIG_DIR / f"{name}.png", dpi=DPI)


def _ax_clean(ax, equal: bool = True) -> None:
    plotting.axis_clean(ax, equal=equal)


def _title_ax(ax, text: str, sub: str = "") -> None:
    plotting.title_ax(ax, text, sub)


# ══════════════════════════════════════════════════════════ figure builders ═════

# ─────────────────────────────── Fig 1: L-system chord gallery ────────────────

def fig_lsystem_chord_gallery() -> Path:
    """6 chords at fixed depth=4; one panel per chord."""
    names = ["Major", "Minor", "Dom7", "Maj7", "Sus4", "Dim7"]
    fig, axes = plt.subplots(2, 3, figsize=(FIG_WIDTH, FIG_WIDTH * 0.7))
    for ax, name, color in zip(axes.ravel(), names, CHORD_COLORS):
        g = lsystem_from_ratios(CHORDS[name], depth=4)
        coords = g.coordinates
        edges  = g.edges
        segs   = coords[edges]
        lc = LineCollection(segs, colors=color, linewidths=0.5, alpha=0.75)
        ax.add_collection(lc)
        pad = 0.05 * (coords.ptp(axis=0).max() + 1e-3)
        ax.set_xlim(coords[:, 0].min() - pad, coords[:, 0].max() + pad)
        ax.set_ylim(coords[:, 1].min() - pad, coords[:, 1].max() + pad)
        _ax_clean(ax)
        θ = g.metadata["angle_deg"]
        n_seg = g.metadata["n_segments"]
        _title_ax(ax, name, f"θ={θ:.1f}°  |  {n_seg} segs")
    fig.suptitle("L-system from Ratios — chord gallery (depth=4)", fontsize=10, y=1.01)
    fig.tight_layout()
    return _save_fig(fig, "p5_lsystem_chords")


# ─────────────────────────────── Fig 2: L-system depth sweep ──────────────────

def fig_lsystem_depth_sweep() -> Path:
    """Major chord across depth 2–5, showing segment growth."""
    depths = [2, 3, 4, 5]
    fig, axes = plt.subplots(1, 4, figsize=(FIG_WIDTH, FIG_WIDTH * 0.32))
    for ax, d in zip(axes, depths):
        g = lsystem_from_ratios(CHORDS["Major"], depth=d)
        coords = g.coordinates
        edges  = g.edges
        segs   = coords[edges]
        lc = LineCollection(segs, colors=PALETTE["dark_blue"], linewidths=max(0.2, 0.8 - d * 0.1), alpha=0.7)
        ax.add_collection(lc)
        pad = 0.04 * (coords.ptp(axis=0).max() + 1e-3)
        ax.set_xlim(coords[:, 0].min() - pad, coords[:, 0].max() + pad)
        ax.set_ylim(coords[:, 1].min() - pad, coords[:, 1].max() + pad)
        _ax_clean(ax)
        _title_ax(ax, f"depth={d}", f"{g.metadata['n_segments']} segs")
    fig.suptitle("L-system — depth sweep (Major chord)", fontsize=10, y=1.03)
    fig.tight_layout()
    return _save_fig(fig, "p5_lsystem_depth")


# ─────────────────────────────── Fig 3: Recursive polygon gallery ─────────────

def fig_polygon_chord_gallery() -> Path:
    """6 chords → recursive polygon at depth=3."""
    names = ["Major", "Minor", "Dom7", "Maj7", "Sus4", "Dim7"]
    fig, axes = plt.subplots(2, 3, figsize=(FIG_WIDTH, FIG_WIDTH * 0.7))
    for ax, name, color in zip(axes.ravel(), names, CHORD_COLORS):
        g = recursive_polygon(CHORDS[name], depth=3)
        coords = np.asarray(g.coordinates)
        ax.plot(coords[:, 0], coords[:, 1], color=color, lw=0.7)
        ax.fill(coords[:, 0], coords[:, 1], color=color, alpha=0.08)
        _ax_clean(ax)
        sf = g.parameters["scale_factor"]
        _title_ax(ax, name, f"scale={sf:.3f}  |  {g.metadata['n_vertices']} verts")
    fig.suptitle("Recursive Polygon — chord gallery (depth=3)", fontsize=10, y=1.01)
    fig.tight_layout()
    return _save_fig(fig, "p5_polygon_chords")


# ─────────────────────────────── Fig 4: Recursive polygon depth sweep ─────────

def fig_polygon_depth_sweep() -> Path:
    """Major chord across depth 1–4."""
    fig, axes = plt.subplots(1, 4, figsize=(FIG_WIDTH, FIG_WIDTH * 0.28))
    for ax, d in enumerate(range(1, 5), start=1):
        g = recursive_polygon(CHORDS["Major"], depth=d)
        coords = np.asarray(g.coordinates)
        axes[d - 1].plot(coords[:, 0], coords[:, 1], color=PALETTE["red"], lw=0.7)
        axes[d - 1].fill(coords[:, 0], coords[:, 1], color=PALETTE["red"], alpha=0.08)
        _ax_clean(axes[d - 1])
        _title_ax(axes[d - 1], f"depth={d}", f"{g.metadata['n_vertices'] - 1} edges")
    fig.suptitle("Recursive Polygon — depth sweep (Major chord)", fontsize=10, y=1.03)
    fig.tight_layout()
    return _save_fig(fig, "p5_polygon_depth")


# ─────────────────────────────── Fig 5: Polygon n_sides sweep ─────────────────

def fig_polygon_sides_sweep() -> Path:
    """Fix chord=Major, sweep n_sides=3,4,5,6,7,8 at depth=3."""
    sides_list = [3, 4, 5, 6, 7, 8]
    fig, axes = plt.subplots(2, 3, figsize=(FIG_WIDTH, FIG_WIDTH * 0.7))
    for ax, n_sides in zip(axes.ravel(), sides_list):
        g = recursive_polygon(CHORDS["Major"], depth=3, n_sides=n_sides)
        coords = np.asarray(g.coordinates)
        ax.plot(coords[:, 0], coords[:, 1], color=PALETTE["purple"], lw=0.7)
        ax.fill(coords[:, 0], coords[:, 1], color=PALETTE["purple"], alpha=0.08)
        _ax_clean(ax)
        _title_ax(ax, f"{n_sides}-gon", f"{g.metadata['n_vertices'] - 1} edges")
    fig.suptitle("Recursive Polygon — n_sides sweep (Major, depth=3)", fontsize=10, y=1.01)
    fig.tight_layout()
    return _save_fig(fig, "p5_polygon_sides")


# ─────────────────────────────── Fig 11: Self-similar tuning chord gallery ────

def fig_tuning_chord_gallery() -> Path:
    """6 chords at n_levels=4, concentric layout."""
    names = ["Major", "Minor", "Dom7", "Maj7", "Sus4", "Dim7"]
    fig, axes = plt.subplots(2, 3, figsize=(FIG_WIDTH, FIG_WIDTH * 0.75))
    for ax, name, color in zip(axes.ravel(), names, CHORD_COLORS):
        g = self_similar_tuning(CHORDS[name], n_levels=4)
        coords  = np.asarray(g.coordinates)
        edges   = np.asarray(g.edges)
        weights = np.asarray(g.weights)

        # Background circles
        for r in [(k + 1) / 4 for k in range(4)]:
            circ = Circle((0, 0), r, fill=False, edgecolor="#dddddd", lw=0.6)
            ax.add_patch(circ)

        # Edges
        segs = coords[edges]
        lc = LineCollection(segs, colors="#aaaaaa", linewidths=0.6, alpha=0.6, zorder=2)
        ax.add_collection(lc)

        # Nodes — size proportional to amplitude
        wn = weights / max(weights.max(), 1e-9)
        ax.scatter(coords[:, 0], coords[:, 1], s=wn * 60 + 6,
                   c=color, edgecolors="white", linewidths=0.4, zorder=3)

        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        _ax_clean(ax)
        n_nodes = g.metadata["total_nodes"]
        _title_ax(ax, name, f"{n_nodes} pitches, 4 levels")
    fig.suptitle("Self-Similar Tuning — chord gallery (n_levels=4)", fontsize=10, y=1.01)
    fig.tight_layout()
    return _save_fig(fig, "p5_tuning_chords")


# ─────────────────────────────── Fig 12: Tuning n_levels sweep ────────────────

def fig_tuning_levels_sweep() -> Path:
    """Major chord at n_levels = 2, 3, 4, 5."""
    levels_list = [2, 3, 4, 5]
    fig, axes = plt.subplots(1, 4, figsize=(FIG_WIDTH, FIG_WIDTH * 0.3))
    for ax, nl in zip(axes, levels_list):
        g = self_similar_tuning(CHORDS["Major"], n_levels=nl)
        coords  = np.asarray(g.coordinates)
        edges   = np.asarray(g.edges)
        weights = np.asarray(g.weights)

        for r in [(k + 1) / nl for k in range(nl)]:
            circ = Circle((0, 0), r, fill=False, edgecolor="#dddddd", lw=0.5)
            ax.add_patch(circ)

        if len(edges):
            segs = coords[edges]
            lc = LineCollection(segs, colors="#aaaaaa", linewidths=0.5, alpha=0.5, zorder=2)
            ax.add_collection(lc)

        wn = weights / max(weights.max(), 1e-9)
        ax.scatter(coords[:, 0], coords[:, 1], s=wn * 50 + 5,
                   c=PALETTE["dark_blue"], edgecolors="white", linewidths=0.3, zorder=3)

        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        _ax_clean(ax)
        _title_ax(ax, f"n_levels={nl}", f"{g.metadata['total_nodes']} nodes")
    fig.suptitle("Self-Similar Tuning — level sweep (Major chord)", fontsize=10, y=1.04)
    fig.tight_layout()
    return _save_fig(fig, "p5_tuning_levels")


# ─────────────────────────────── Fig 13: Tuning equave sweep ─────────────────

def fig_tuning_equave_sweep() -> Path:
    """Dom7, equave = 2.0, 3.0, 4.0, 5.0."""
    equaves = [2.0, 3.0, 4.0, 5.0]
    labels  = ["Octave (2)", "Tritave (3)", "Quadrave (4)", "Quintave (5)"]
    fig, axes = plt.subplots(1, 4, figsize=(FIG_WIDTH, FIG_WIDTH * 0.3))
    for ax, eq, lbl in zip(axes, equaves, labels):
        g = self_similar_tuning(CHORDS["Dom7"], n_levels=3, equave=eq)
        coords  = np.asarray(g.coordinates)
        edges   = np.asarray(g.edges)
        weights = np.asarray(g.weights)

        for r in [(k + 1) / 3 for k in range(3)]:
            circ = Circle((0, 0), r, fill=False, edgecolor="#dddddd", lw=0.5)
            ax.add_patch(circ)

        if len(edges):
            segs = coords[edges]
            lc = LineCollection(segs, colors="#aaaaaa", linewidths=0.5, alpha=0.5, zorder=2)
            ax.add_collection(lc)

        wn = weights / max(weights.max(), 1e-9)
        ax.scatter(coords[:, 0], coords[:, 1], s=wn * 50 + 5,
                   c=PALETTE["green"], edgecolors="white", linewidths=0.3, zorder=3)

        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        _ax_clean(ax)
        _title_ax(ax, lbl, f"{g.metadata['total_nodes']} nodes")
    fig.suptitle("Self-Similar Tuning — equave sweep (Dom7 chord, n_levels=3)",
                 fontsize=10, y=1.04)
    fig.tight_layout()
    return _save_fig(fig, "p5_tuning_equave")


# ─────────────────────────────── Fig 14: Combined showcase ───────────────────

def fig_combined_showcase() -> Path:
    """One large figure: all 3 generative functions for the Major chord side-by-side."""
    fig = plt.figure(figsize=(FIG_WIDTH, FIG_WIDTH * 0.4))
    axes = fig.subplots(1, 3)

    inp = CHORDS["Major"]

    # ── L-system
    ax = axes[0]
    g = lsystem_from_ratios(inp, depth=4)
    coords = g.coordinates; edges = g.edges
    lc = LineCollection(coords[edges], colors=PALETTE["dark_blue"], linewidths=0.5, alpha=0.75)
    ax.add_collection(lc)
    pad = 0.04 * (coords.ptp(axis=0).max() + 1e-3)
    ax.set_xlim(coords[:, 0].min() - pad, coords[:, 0].max() + pad)
    ax.set_ylim(coords[:, 1].min() - pad, coords[:, 1].max() + pad)
    _ax_clean(ax); ax.set_title("L-system", fontsize=8, pad=3)

    # ── Recursive polygon
    ax = axes[1]
    g = recursive_polygon(inp, depth=4)
    coords = np.asarray(g.coordinates)
    ax.plot(coords[:, 0], coords[:, 1], color=PALETTE["red"], lw=0.6)
    ax.fill(coords[:, 0], coords[:, 1], color=PALETTE["red"], alpha=0.07)
    _ax_clean(ax); ax.set_title("Recursive Polygon", fontsize=8, pad=3)

    # ── Self-similar tuning
    ax = axes[2]
    g = self_similar_tuning(inp, n_levels=4)
    coords = np.asarray(g.coordinates); edges = np.asarray(g.edges)
    weights = np.asarray(g.weights)
    for r in [0.25, 0.5, 0.75, 1.0]:
        ax.add_patch(Circle((0, 0), r, fill=False, edgecolor="#dddddd", lw=0.5))
    if len(edges):
        lc = LineCollection(coords[edges], colors="#aaaaaa", linewidths=0.5, alpha=0.5, zorder=2)
        ax.add_collection(lc)
    wn = weights / max(weights.max(), 1e-9)
    ax.scatter(coords[:, 0], coords[:, 1], s=wn * 50 + 5, c=PALETTE["gold"],
               edgecolors="white", linewidths=0.3, zorder=3)
    ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.15)
    _ax_clean(ax); ax.set_title("Self-Similar Tuning", fontsize=8, pad=3)

    fig.suptitle("Phase 5 — generative functions (Major chord)", fontsize=10, y=1.03)
    fig.tight_layout()
    return _save_fig(fig, "p5_combined_showcase")


# ══════════════════════════════════════════════════════════ PDF assembly ════════

def _styles():
    base = getSampleStyleSheet()
    H1 = ParagraphStyle(
        "H1",
        parent=base["Heading1"],
        fontSize=16,
        leading=20,
        textColor=HexColor("#1f3b73"),
        spaceBefore=18,
        spaceAfter=8,
    )
    H2 = ParagraphStyle(
        "H2",
        parent=base["Heading2"],
        fontSize=12,
        leading=16,
        textColor=HexColor("#a23e2c"),
        spaceBefore=14,
        spaceAfter=6,
    )
    BODY = ParagraphStyle(
        "Body",
        parent=base["Normal"],
        fontSize=9.5,
        leading=14,
        spaceAfter=6,
        textColor=HexColor("#333333"),
    )
    CAPTION = ParagraphStyle(
        "Caption",
        parent=base["Normal"],
        fontSize=8,
        leading=11,
        alignment=TA_CENTER,
        textColor=HexColor("#666666"),
        spaceAfter=10,
    )
    CODE = ParagraphStyle(
        "Code",
        parent=base["Code"],
        fontSize=7.5,
        leading=11,
        leftIndent=18,
        spaceAfter=8,
        textColor=HexColor("#1a1a2e"),
        backColor=HexColor("#f2f2f5"),
    )
    return H1, H2, BODY, CAPTION, CODE


def _img(path: Path, width_in: float = FIG_WIDTH) -> Image:
    return Image(str(path), width=width_in * inch, height=9 * inch, kind="proportional")


def build_pdf(figures: dict[str, Path]) -> None:
    H1, H2, BODY, CAPTION, CODE = _styles()

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=LETTER,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    story = []

    # ── Cover
    story += [
        Spacer(1, 0.4 * inch),
        Paragraph("biotuner.harmonic_geometry", ParagraphStyle(
            "sub_cover", parent=H2, fontSize=11, textColor=HexColor("#888888")
        )),
        Paragraph("Phase 5 Report: Generative Fractal Structures", ParagraphStyle(
            "cover_title", parent=H1, fontSize=22, leading=28, spaceBefore=4
        )),
        Paragraph(
            "Five functions that derive fractal and self-similar geometry directly "
            "from harmonic inputs (ratios, peaks, amplitudes).  All outputs are "
            "<b>GeometryData</b> objects—coordinates, edges, fields—ready for "
            "downstream rendering or analysis.",
            BODY,
        ),
        Spacer(1, 0.15 * inch),
        _img(figures["combined"], 6.0),
        Paragraph(
            "Figure 1. The four Phase 5 generators applied to the Just-Intonation Major chord "
            "(1, 5/4, 3/2).  Left to right: L-system branching plant · Koch-like recursive "
            "polygon · escape-time Julia set · self-similar tuning lattice. "
            "(Cantor-rhythm has moved to a future rhythm module — see "
            "docs/notes/rhythm_module_todos.md.)",
            CAPTION,
        ),
        PageBreak(),
    ]

    # ── Section 1: lsystem_from_ratios
    story += [
        Paragraph("1. lsystem_from_ratios", H1),
        Paragraph(
            "Generates an L-system branching plant whose turning angle θ and branching "
            "topology are derived directly from a <b>HarmonicInput</b>.  The base angle "
            "comes from the first non-unison ratio as <i>θ = 360 / (p + q)</i>.  "
            "The number of side-branches equals <i>n_components − 1</i>, so each "
            "interval of the chord maps to a distinct branch direction.  Custom rewriting "
            "rules can override both.  Returns a <b>graph</b> GeometryData.",
            BODY,
        ),
        Paragraph(
            "<b>Signature:</b>  "
            "lsystem_from_ratios(input, depth=4, axiom='F', rules=None, step_size=1.0)",
            CODE,
        ),
        Paragraph(
            "The chord gallery below shows how the plant morphology changes across six "
            "JI chord classes at a fixed depth of 4.  Each chord produces a distinct "
            "branching angle and a different number of side-branches, yielding visually "
            "distinguishable fractal plants.",
            BODY,
        ),
        _img(figures["lsystem_chords"]),
        Paragraph(
            "Figure 2. L-system plants for six JI chords at depth=4.  "
            "θ is the turning angle; 'segs' is the total number of turtle segments "
            "(edges in the returned graph).",
            CAPTION,
        ),
        Paragraph(
            "Increasing depth exponentially multiplies the number of segments: each "
            "rewriting step replaces every 'F' token with a branch string, growing "
            "the total segment count by a factor equal to the number of F symbols in "
            "the rule right-hand side.",
            BODY,
        ),
        _img(figures["lsystem_depth"]),
        Paragraph(
            "Figure 3. Depth sweep for the Major chord (depth 2–5).  "
            "The turning angle θ ≈ 30° (derived from the 3:2 perfect fifth) "
            "remains fixed; depth controls the iteration count.",
            CAPTION,
        ),
        PageBreak(),
    ]

    # ── Section 2: recursive_polygon
    story += [
        Paragraph("2. recursive_polygon", H1),
        Paragraph(
            "Constructs a Koch-like self-similar boundary by iteratively subdividing "
            "each edge of a regular polygon.  Each subdivision replaces a straight edge "
            "with four sub-edges forming an outward triangular bump.  The <i>scale</i> "
            "and <i>bump angle</i> are read from the first non-unison ratio: "
            "scale = 1/(p+1) and bump angle = π/q for ratio p/q.  The number of polygon "
            "sides defaults to n_components.  Returns a closed <b>curve_2d</b> GeometryData.",
            BODY,
        ),
        Paragraph(
            "<b>Signature:</b>  "
            "recursive_polygon(input, depth=4, n_sides=None, scale_factor=None)",
            CODE,
        ),
        _img(figures["polygon_chords"]),
        Paragraph(
            "Figure 4. Recursive polygon boundaries at depth=3 for six JI chords.  "
            "'scale' is the bump sub-edge fraction; 'verts' is the total vertex count.",
            CAPTION,
        ),
        _img(figures["polygon_depth"]),
        Paragraph(
            "Figure 5. Depth sweep for the Major chord (depth 1–4).  "
            "Vertex count grows as 4^depth × n_sides (here n_sides=3).",
            CAPTION,
        ),
        Paragraph(
            "The n_sides parameter is independent of the harmonic content and lets you "
            "explore how the chord's subdivision geometry looks on different base polygons:",
            BODY,
        ),
        _img(figures["polygon_sides"]),
        Paragraph(
            "Figure 6. n_sides sweep (3–8) for the Major chord at depth=3.  "
            "The scale and bump angle remain fixed; only the base shape changes.",
            CAPTION,
        ),
        PageBreak(),
    ]

    # ── Section 3: self_similar_tuning (Julia removed — produced visually
    # similar fractals across chord ratios; deferred to a future module if
    # a more chord-discriminative formulation is found.)
    story += [
        Paragraph("3. self_similar_tuning", H1),
        Paragraph(
            "Generates a multi-level harmonic spiral lattice.  Starting from the "
            "input ratios as generators (level 0), each subsequent generation is "
            "formed by multiplying every pitch at the previous level by every "
            "generator and reducing to [1, equave) — the free abelian group generated "
            "by the ratios, truncated at n_levels generations.  Close pitches (within "
            "5 cents) are merged.",
            BODY,
        ),
        Paragraph(
            "Pitches are placed on concentric rings: level k sits at radius (k+1)/n_levels.  "
            "Angular position is log_equave(pitch), so the octave maps to a full "
            "revolution.  Edges connect each node to its closest ancestor at the "
            "previous level.  Node sizes encode normalised amplitudes.  Returns a "
            "<b>graph</b> GeometryData.",
            BODY,
        ),
        Paragraph(
            "<b>Signature:</b>  "
            "self_similar_tuning(input, n_levels=4, equave=2.0)",
            CODE,
        ),
        _img(figures["tuning_chords"]),
        Paragraph(
            "Figure 12. Self-similar tuning lattices for six JI chords (n_levels=4, "
            "equave=2.0).  Concentric rings mark generational levels; edges trace "
            "the generative lineage; node size encodes amplitude.",
            CAPTION,
        ),
        _img(figures["tuning_levels"]),
        Paragraph(
            "Figure 13. Level sweep for the Major chord (n_levels 2–5).  "
            "Each additional level adds a new ring of derived pitches.",
            CAPTION,
        ),
        Paragraph(
            "The equave parameter shifts the logarithmic metric used for pitch "
            "placement, producing different rotational symmetries when non-octave "
            "equivalences (tritave, etc.) are used:",
            BODY,
        ),
        _img(figures["tuning_equave"]),
        Paragraph(
            "Figure 14. Equave sweep for the Dom7 chord (n_levels=3).  "
            "The octave (2), tritave (3), quadrave (4), and quintave (5) each "
            "yield a distinct rotational grouping on the spiral.",
            CAPTION,
        ),
        PageBreak(),
    ]

    # ── Parameter table
    story += [
        Paragraph("Parameter Reference", H1),
        Paragraph(
            "Summary of all public parameters across the five Phase 5 functions.",
            BODY,
        ),
        Spacer(1, 0.1 * inch),
    ]

    table_data = [
        ["Function", "Parameter", "Default", "Range / Notes"],
        ["lsystem_from_ratios", "depth",      "4",    "int ∈ [1, 7]"],
        ["",                    "axiom",      "'F'",  "non-empty string"],
        ["",                    "rules",      "None", "dict → auto-derived from ratios"],
        ["",                    "step_size",  "1.0",  "float > 0"],
        ["recursive_polygon",   "depth",      "4",    "int ∈ [1, 6]"],
        ["",                    "n_sides",    "None", "int ≥ 3 (default = n_components)"],
        ["",                    "scale_factor","None","float ∈ (0, 0.5)"],
        ["self_similar_tuning", "n_levels",   "4",    "int ≥ 1"],
        ["",                    "equave",     "2.0",  "float > 1.0"],
    ]

    col_w = [1.65 * inch, 1.45 * inch, 0.85 * inch, 2.55 * inch]
    tbl = Table(table_data, colWidths=col_w)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0),  HexColor("#1f3b73")),
        ("TEXTCOLOR",    (0, 0), (-1, 0),  white),
        ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, 0),  8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#f8f8fb"), white]),
        ("FONTSIZE",     (0, 1), (-1, -1), 8),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("GRID",         (0, 0), (-1, -1), 0.25, HexColor("#cccccc")),
        ("LEFTPADDING",  (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.2 * inch))

    # ── Return types table
    story += [
        Paragraph("Return Type Summary", H2),
    ]
    ret_data = [
        ["Function",            "geom_type", "coordinates shape",   "Notable metadata"],
        ["lsystem_from_ratios", "graph",     "(N×2) turtle pts",    "lstring_preview, n_segments, angle_deg"],
        ["recursive_polygon",   "curve_2d",  "(M×2) closed boundary","n_vertices, source_ratio"],
        ["self_similar_tuning", "graph",     "(K×2) node positions","n_nodes_per_level, pitches_level_0"],
    ]
    col_w2 = [1.65 * inch, 0.75 * inch, 1.55 * inch, 2.55 * inch]
    tbl2 = Table(ret_data, colWidths=col_w2)
    tbl2.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  HexColor("#a23e2c")),
        ("TEXTCOLOR",      (0, 0), (-1, 0),  white),
        ("FONTNAME",       (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0, 0), (-1, 0),  8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [HexColor("#fdf8f6"), white]),
        ("FONTSIZE",       (0, 1), (-1, -1), 8),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",     (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",  (0, 0), (-1, -1), 4),
        ("GRID",           (0, 0), (-1, -1), 0.25, HexColor("#cccccc")),
        ("LEFTPADDING",    (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",   (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl2)

    doc.build(story)
    print(f"[✓] PDF written → {PDF_PATH}")


# ══════════════════════════════════════════════════════════ main ════════════════

def main():
    print("Rendering figures…")
    figures = {
        "combined":       fig_combined_showcase(),
        "lsystem_chords": fig_lsystem_chord_gallery(),
        "lsystem_depth":  fig_lsystem_depth_sweep(),
        "polygon_chords": fig_polygon_chord_gallery(),
        "polygon_depth":  fig_polygon_depth_sweep(),
        "polygon_sides":  fig_polygon_sides_sweep(),
        "tuning_chords":  fig_tuning_chord_gallery(),
        "tuning_levels":  fig_tuning_levels_sweep(),
        "tuning_equave":  fig_tuning_equave_sweep(),
    }
    print("Building PDF…")
    build_pdf(figures)


if __name__ == "__main__":
    main()
