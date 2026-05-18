"""
Generate the Phase 7 report PDF + rotating GIF animations for
biotuner.harmonic_geometry.geometry_3d.

Produces:
  docs/reports/harmonic_geometry_phase7.pdf
  docs/reports/figures/anim_*.gif  (one per geometry type)

Run with the `biotuner` conda env:

    python docs/reports/generate_phase7_report.py
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
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import numpy as np

from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)

from biotuner.harmonic_geometry import (
    HarmonicInput,
    harmonic_knot,
    harmonic_point_cloud,
    harmonic_surface,
    lissajous_tube,
    lsystem_3d,
    plotting,
    recursive_polyhedron,
)

# ─── paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR   = REPO_ROOT / "docs" / "reports"
FIG_DIR   = OUT_DIR / "figures"
ANIM_DIR  = FIG_DIR / "anim_phase7"
PDF_PATH  = OUT_DIR / "harmonic_geometry_phase7.pdf"
FIG_DIR.mkdir(parents=True, exist_ok=True)
ANIM_DIR.mkdir(parents=True, exist_ok=True)

FIG_W = 6.5
DPI   = 120
ANIM_DPI  = 80
ANIM_FRAMES = 36   # 36 × 10° = full 360°

# ─── chord presets ────────────────────────────────────────────────────────────

CHORDS: dict[str, HarmonicInput] = {
    "Major":  HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)]),
    "Minor":  HarmonicInput(ratios=[Fraction(1), Fraction(6, 5), Fraction(3, 2)]),
    "Dom7":   HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2), Fraction(7, 4)]),
    "Maj7":   HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2), Fraction(15, 8)]),
    "Sus4":   HarmonicInput(ratios=[Fraction(1), Fraction(4, 3), Fraction(3, 2)]),
    "Aug":    HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(8, 5)]),
    "Dim7":   HarmonicInput(ratios=[Fraction(1), Fraction(6, 5), Fraction(7, 5), Fraction(12, 7)]),
    "Perfect5": HarmonicInput(ratios=[Fraction(3, 2)]),
}

PALETTE = {
    "dark_blue": "#1f3b73",
    "red":       "#a23e2c",
    "green":     "#3a7a4d",
    "gold":      "#7a5d24",
    "purple":    "#5c2e7a",
    "teal":      "#1f6b6b",
    "orange":    "#b05f1a",
    "slate":     "#3d4f60",
}
CHORD_COLORS = list(PALETTE.values())

# ─── helpers (thin wrappers over biotuner.harmonic_geometry.plotting) ────────

def _save(fig, name: str) -> Path:
    return plotting.save_figure(fig, FIG_DIR / f"{name}.png", dpi=DPI)


def _ax3(fig, pos, title="", elev=25, azim=45):
    return plotting.make_axis_3d(fig, pos, title=title, elev=elev, azim=azim)


# Direct aliases — same signature as the plotting primitives.
_draw_mesh   = plotting.draw_mesh_3d
_draw_tree   = plotting.draw_tree_3d


def _draw_points(ax, geom, color, s=2, alpha=0.6):
    plotting.draw_point_cloud_3d(ax, geom, color=color, size=s, alpha=alpha)


# ─── rotation animation ───────────────────────────────────────────────────────

def _make_rotation_gif(draw_fn, geom, gif_name: str, color,
                       elev: int = 20, draw_kwargs: dict | None = None) -> Path:
    """Save a 36-frame rotating GIF for a single GeometryData object."""
    return plotting.animate_rotation(
        geom, ANIM_DIR / f"{gif_name}.gif",
        n_frames=ANIM_FRAMES, fps=12, elev=elev,
        fig_size=(3, 3), dpi=ANIM_DPI, color=color,
        draw_kwargs=draw_kwargs, renderer=draw_fn,
    )


def _animation_strip(geom, draw_fn, color, name: str,
                     n_strip: int = 6, elev: int = 20,
                     draw_kwargs: dict | None = None) -> Path:
    """Save n_strip evenly-spaced rotation frames as a single PNG strip."""
    fig, _ = plotting.rotation_strip(
        geom, n_strip=n_strip, fig_width=FIG_W, elev=elev, color=color,
        draw_kwargs=draw_kwargs, renderer=draw_fn,
    )
    return _save(fig, name)


# ════════════════════════════════════════════ figure builders ══════════════════

# ── Fig 1: lissajous_tube chord gallery ───────────────────────────────────────

def fig_tube_chord_gallery() -> Path:
    names = ["Major", "Minor", "Dom7", "Maj7", "Sus4", "Dim7"]
    fig = plt.figure(figsize=(FIG_W, 2.4))
    for k, (name, color) in enumerate(zip(names, CHORD_COLORS)):
        ax = _ax3(fig, int(f"16{k+1}"), title=name, azim=35 + k * 12)
        g  = lissajous_tube(CHORDS[name], n_points=400, n_sides=10)
        _draw_mesh(ax, g, color=color, alpha=0.8)
    fig.suptitle("lissajous_tube — 6 chords", fontsize=10, y=1.02)
    fig.tight_layout()
    return _save(fig, "p7_tube_gallery")


def fig_tube_rotation_strip() -> Path:
    g = lissajous_tube(CHORDS["Dom7"], n_points=600, n_sides=12)
    return _animation_strip(g, _draw_mesh, PALETTE["purple"],
                            "p7_tube_strip", n_strip=6, elev=20,
                            draw_kwargs={"alpha": 0.8})


# ── Fig 2: harmonic_knot gallery ──────────────────────────────────────────────

def fig_knot_gallery() -> Path:
    ratios_knots = [
        (Fraction(2, 1), "T(2,1) — 2/1"),
        (Fraction(3, 2), "T(3,2) — trefoil"),
        (Fraction(4, 3), "T(4,3) — 4/3"),
        (Fraction(5, 3), "T(5,3) — 5/3"),
        (Fraction(5, 4), "T(5,4) — 5/4"),
        (Fraction(7, 4), "T(7,4) — 7/4"),
    ]
    fig = plt.figure(figsize=(FIG_W, 2.4))
    cmap = matplotlib.colormaps["plasma"]
    for k, (ratio, label) in enumerate(ratios_knots):
        inp = HarmonicInput(ratios=[ratio])
        ax  = _ax3(fig, int(f"16{k+1}"), title=label, azim=30 + k * 15)
        g   = harmonic_knot(inp, n_points=300, n_sides=10)
        _draw_mesh(ax, g, color=cmap(k / 6), alpha=0.85)
    fig.suptitle("harmonic_knot — T(p,q) for 6 ratios", fontsize=10, y=1.02)
    fig.tight_layout()
    return _save(fig, "p7_knot_gallery")


def fig_knot_rotation_strip() -> Path:
    g = harmonic_knot(CHORDS["Perfect5"], n_points=400, n_sides=14)
    return _animation_strip(g, _draw_mesh, PALETTE["red"],
                            "p7_knot_strip", n_strip=6, elev=25,
                            draw_kwargs={"alpha": 0.85})


# ── Fig 3: harmonic_surface modes ────────────────────────────────────────────

def fig_surface_modes() -> Path:
    modes = ["torus", "sphere", "cylinder"]
    chords = ["Major", "Dom7", "Dim7"]
    cmap_names = ["Blues", "Reds", "Greens"]
    fig = plt.figure(figsize=(FIG_W, 4.5))
    idx = 1
    for row, chord_name in enumerate(chords):
        for col, (mode, cname) in enumerate(zip(modes, cmap_names)):
            ax = _ax3(fig, int(f"33{idx}"), title=f"{mode}\n{chord_name}", azim=40 + col*15)
            g  = harmonic_surface(CHORDS[chord_name], mode=mode, resolution=40)
            c  = matplotlib.colormaps[cname](0.55)
            _draw_mesh(ax, g, color=c, alpha=0.7, lw=0.1)
            idx += 1
    fig.suptitle("harmonic_surface — 3 modes × 3 chords", fontsize=10)
    fig.tight_layout()
    return _save(fig, "p7_surface_modes")


def fig_surface_chord_sweep() -> Path:
    """Torus surface for 6 chords at fixed resolution."""
    names  = ["Major", "Minor", "Dom7", "Maj7", "Sus4", "Dim7"]
    cmap_m = matplotlib.colormaps["viridis"]
    fig    = plt.figure(figsize=(FIG_W, 2.4))
    for k, (name, color) in enumerate(zip(names, CHORD_COLORS)):
        ax = _ax3(fig, int(f"16{k+1}"), title=name, azim=40 + k*10)
        g  = harmonic_surface(CHORDS[name], mode="torus", resolution=32)
        _draw_mesh(ax, g, color=color, alpha=0.75, lw=0.1)
    fig.suptitle("harmonic_surface (torus) — 6 chords", fontsize=10, y=1.02)
    fig.tight_layout()
    return _save(fig, "p7_surface_chords")


def fig_surface_rotation_strip() -> Path:
    g = harmonic_surface(CHORDS["Dom7"], mode="sphere", resolution=48)
    return _animation_strip(g, _draw_mesh, PALETTE["teal"],
                            "p7_surface_strip", n_strip=6, elev=30,
                            draw_kwargs={"alpha": 0.75, "lw": 0.1})


# ── Fig 4: lsystem_3d ────────────────────────────────────────────────────────

def fig_lsystem3d_gallery() -> Path:
    names = ["Major", "Minor", "Dom7", "Maj7", "Sus4", "Dim7"]
    fig   = plt.figure(figsize=(FIG_W, 2.4))
    for k, (name, color) in enumerate(zip(names, CHORD_COLORS)):
        ax = _ax3(fig, int(f"16{k+1}"), title=name, azim=30 + k*12, elev=15)
        g  = lsystem_3d(CHORDS[name], depth=3)
        _draw_tree(ax, g, color=color, lw=0.4)
    fig.suptitle("lsystem_3d — 6 chords (depth=3)", fontsize=10, y=1.02)
    fig.tight_layout()
    return _save(fig, "p7_lsystem3d_gallery")


def fig_lsystem3d_depth_sweep() -> Path:
    fig = plt.figure(figsize=(FIG_W, 2.2))
    cmap = matplotlib.colormaps["magma"]
    for k, depth in enumerate([1, 2, 3, 4]):
        ax = _ax3(fig, int(f"14{k+1}"), title=f"depth={depth}", azim=40, elev=20)
        g  = lsystem_3d(CHORDS["Dom7"], depth=depth)
        n  = g.metadata["n_segments"]
        _draw_tree(ax, g, color=cmap(0.3 + 0.15 * k), lw=max(0.6 - 0.1 * k, 0.15))
        ax.set_title(f"depth={depth}\n{n} segs", fontsize=7, pad=3)
    fig.suptitle("lsystem_3d depth sweep (Dom7)", fontsize=10, y=1.02)
    fig.tight_layout()
    return _save(fig, "p7_lsystem3d_depth")


def fig_lsystem3d_rotation_strip() -> Path:
    g = lsystem_3d(CHORDS["Dom7"], depth=3)
    return _animation_strip(g, _draw_tree, PALETTE["dark_blue"],
                            "p7_lsystem3d_strip", n_strip=6, elev=20,
                            draw_kwargs={"lw": 0.5})


# ── Fig 5: recursive_polyhedron ──────────────────────────────────────────────

def fig_polyhedron_solids() -> Path:
    solids = ["tetrahedron", "cube", "icosahedron"]
    depths = [0, 1, 2]
    cmap   = matplotlib.colormaps["cool"]
    fig    = plt.figure(figsize=(FIG_W, 4.5))
    idx    = 1
    for row, solid in enumerate(solids):
        for col, depth in enumerate(depths):
            ax = _ax3(fig, int(f"33{idx}"), title=f"{solid}\ndepth={depth}", azim=35)
            g  = recursive_polyhedron(CHORDS["Major"], depth=depth, solid=solid)
            c  = cmap(row / 3 + col * 0.05)
            _draw_mesh(ax, g, color=c, alpha=0.8, lw=0.15)
            idx += 1
    fig.suptitle("recursive_polyhedron — 3 solids × 3 depths (Major)", fontsize=10)
    fig.tight_layout()
    return _save(fig, "p7_polyhedron_solids")


def fig_polyhedron_chord_gallery() -> Path:
    names = ["Major", "Minor", "Dom7", "Maj7", "Sus4", "Dim7"]
    fig   = plt.figure(figsize=(FIG_W, 2.4))
    for k, (name, color) in enumerate(zip(names, CHORD_COLORS)):
        ax = _ax3(fig, int(f"16{k+1}"), title=name, azim=40 + k*8)
        g  = recursive_polyhedron(CHORDS[name], depth=2, solid="icosahedron")
        _draw_mesh(ax, g, color=color, alpha=0.8, lw=0.1)
    fig.suptitle("recursive_polyhedron (icosahedron, depth=2) — 6 chords", fontsize=10, y=1.02)
    fig.tight_layout()
    return _save(fig, "p7_polyhedron_chords")


def fig_polyhedron_rotation_strip() -> Path:
    g = recursive_polyhedron(CHORDS["Dom7"], depth=2, solid="icosahedron")
    return _animation_strip(g, _draw_mesh, PALETTE["gold"],
                            "p7_polyhedron_strip", n_strip=6, elev=30,
                            draw_kwargs={"alpha": 0.82, "lw": 0.1})


def fig_polyhedron_differentiation() -> Path:
    """Show how per-face bump scaling + auto-solid + apex twist differentiate chords.

    Top row: same chord (Dom7), four toggle combinations.
    Bottom row: four different chords with auto-picked solids and per-face bumps.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    cmap = matplotlib.colormaps["tab10"]

    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.55))

    # Top row — toggle comparison on Dom7
    toggles = [
        (False, False, "no per-face\nno twist"),
        (True,  False, "per-face\nno twist"),
        (False, True,  "no per-face\ntwist"),
        (True,  True,  "per-face\ntwist (default)"),
    ]
    for k, (pf, tw, label) in enumerate(toggles):
        ax = _ax3(fig, int(f"24{k+1}"), title=label, azim=35 + k * 6, elev=22)
        g = recursive_polyhedron(CHORDS["Dom7"], depth=2, solid="icosahedron",
                                   per_face_bump=pf, apex_twist=tw)
        # Per-face colour by ratio index
        ratio_idx = np.asarray(g.metadata["face_ratio_index"])
        n_ratios  = int(g.metadata["n_ratios"])
        verts     = np.asarray(g.coordinates)
        faces     = np.asarray(g.faces)
        face_colors = cmap(ratio_idx / max(n_ratios - 1, 1))
        polys = [verts[tri] for tri in faces]
        coll = Poly3DCollection(polys, facecolor=face_colors,
                                  edgecolor="white", linewidth=0.05, alpha=0.85)
        ax.add_collection3d(coll)
        ax.auto_scale_xyz(verts[:, 0], verts[:, 1], verts[:, 2])

    # Bottom row — four chords, all with new defaults (auto-solid, per-face, twist)
    chords = ["Major", "Minor", "Dom7", "Maj7"]
    for k, name in enumerate(chords):
        ax = _ax3(fig, int(f"24{k+5}"), title=f"{name}\n(auto-solid)",
                   azim=30 + k * 12, elev=22)
        g = recursive_polyhedron(CHORDS[name], depth=2, solid=None)
        ratio_idx = np.asarray(g.metadata["face_ratio_index"])
        n_ratios  = int(g.metadata["n_ratios"])
        verts     = np.asarray(g.coordinates)
        faces     = np.asarray(g.faces)
        face_colors = cmap(ratio_idx / max(n_ratios - 1, 1))
        polys = [verts[tri] for tri in faces]
        coll = Poly3DCollection(polys, facecolor=face_colors,
                                  edgecolor="white", linewidth=0.05, alpha=0.85)
        ax.add_collection3d(coll)
        ax.auto_scale_xyz(verts[:, 0], verts[:, 1], verts[:, 2])

    fig.suptitle("recursive_polyhedron — chord differentiation "
                 "(per-face bump + apex twist + auto-solid; faces coloured by ratio)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    return _save(fig, "p7_polyhedron_differentiation")


# ── Fig 6: harmonic_point_cloud ───────────────────────────────────────────────

def fig_point_cloud_gallery() -> Path:
    names    = ["Major", "Minor", "Dom7", "Maj7", "Sus4", "Dim7"]
    surfaces = ["sphere", "sphere", "torus", "torus", "sphere", "torus"]
    fig      = plt.figure(figsize=(FIG_W, 2.4))
    for k, (name, surf, color) in enumerate(zip(names, surfaces, CHORD_COLORS)):
        ax = _ax3(fig, int(f"16{k+1}"), title=f"{name}\n{surf}", azim=30 + k*12, elev=20)
        g  = harmonic_point_cloud(CHORDS[name], n_points=800, surface=surf)
        _draw_points(ax, g, color=color, s=1, alpha=0.5)
    fig.suptitle("harmonic_point_cloud — 6 chords (sphere / torus)", fontsize=10, y=1.02)
    fig.tight_layout()
    return _save(fig, "p7_cloud_gallery")


def fig_point_cloud_rotation_strip() -> Path:
    g = harmonic_point_cloud(CHORDS["Dom7"], n_points=1500, surface="sphere")
    return _animation_strip(g, _draw_points, PALETTE["green"],
                            "p7_cloud_strip", n_strip=6, elev=20,
                            draw_kwargs={"s": 1, "alpha": 0.5})


def fig_point_cloud_surfaces() -> Path:
    """All five supported surfaces on Dom7, high density, coloured by field."""
    surfaces = ["sphere", "torus", "klein", "hyperbolic", "mos"]
    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.30))
    cmap = matplotlib.colormaps["plasma"]
    for k, surf in enumerate(surfaces):
        ax = _ax3(fig, int(f"15{k+1}"), title=surf, azim=35 + k * 18, elev=22)
        g = harmonic_point_cloud(CHORDS["Dom7"], n_points=4000, surface=surf)
        V = np.asarray(g.coordinates)
        w = np.asarray(g.weights)
        # normalise field to [0,1]
        wn = (w - w.min()) / (w.max() - w.min() + 1e-9)
        ax.scatter(V[:, 0], V[:, 1], V[:, 2], c=cmap(wn),
                   s=2.0, alpha=0.6, edgecolors="none")
    fig.suptitle("harmonic_point_cloud — five surfaces (Dom7, 4 000 pts/each, "
                 "colour = field amplitude)", fontsize=9, y=1.04)
    fig.tight_layout()
    return _save(fig, "p7_cloud_surfaces")


def fig_point_cloud_density() -> Path:
    """High-density coloured-by-field clouds for four chords on a sphere."""
    chords = ["Major", "Minor", "Dom7", "Maj7"]
    fig = plt.figure(figsize=(FIG_W, FIG_W * 0.28))
    cmap = matplotlib.colormaps["viridis"]
    for k, name in enumerate(chords):
        ax = _ax3(fig, int(f"14{k+1}"), title=name, azim=35 + k * 18, elev=22)
        g = harmonic_point_cloud(CHORDS[name], n_points=8000, surface="sphere")
        V = np.asarray(g.coordinates)
        w = np.asarray(g.weights)
        wn = (w - w.min()) / (w.max() - w.min() + 1e-9)
        ax.scatter(V[:, 0], V[:, 1], V[:, 2], c=cmap(wn),
                   s=1.0, alpha=0.5, edgecolors="none")
    fig.suptitle("harmonic_point_cloud — high-density spheres, coloured by field "
                 "(8 000 pts / chord)", fontsize=9, y=1.04)
    fig.tight_layout()
    return _save(fig, "p7_cloud_density")


# ─── GIF animations ────────────────────────────────────────────────────────────

def build_all_gifs() -> dict[str, Path]:
    print("  Generating GIF animations …")
    gifs = {}

    specs = [
        ("tube",       lissajous_tube,      CHORDS["Dom7"],      PALETTE["purple"],  _draw_mesh,   {"n_points": 400, "n_sides": 12},  {}),
        ("knot",       harmonic_knot,       CHORDS["Perfect5"],  PALETTE["red"],     _draw_mesh,   {"n_points": 400, "n_sides": 14},  {}),
        ("surface_t",  harmonic_surface,    CHORDS["Major"],     PALETTE["teal"],    _draw_mesh,   {"mode": "torus",    "resolution": 40}, {"lw": 0.1}),
        ("surface_s",  harmonic_surface,    CHORDS["Dom7"],      PALETTE["dark_blue"],_draw_mesh,  {"mode": "sphere",   "resolution": 40}, {"lw": 0.1}),
        ("surface_c",  harmonic_surface,    CHORDS["Dim7"],      PALETTE["green"],   _draw_mesh,   {"mode": "cylinder", "resolution": 40}, {"lw": 0.1}),
        ("lsystem3d",  lsystem_3d,          CHORDS["Dom7"],      PALETTE["gold"],    _draw_tree,   {"depth": 3},                      {"lw": 0.5}),
        ("polyhedron", recursive_polyhedron,CHORDS["Major"],     PALETTE["orange"],  _draw_mesh,   {"depth": 2, "solid": "icosahedron"},  {"lw": 0.1}),
        ("cloud_sphere",harmonic_point_cloud,CHORDS["Dom7"],     PALETTE["slate"],   _draw_points, {"n_points": 1200, "surface": "sphere"}, {"s":1,"alpha":0.5}),
        ("cloud_torus", harmonic_point_cloud,CHORDS["Maj7"],     PALETTE["green"],   _draw_points, {"n_points": 1200, "surface": "torus"},  {"s":1,"alpha":0.5}),
    ]

    for key, fn, inp, color, draw_fn, fn_kwargs, draw_kwargs in specs:
        print(f"    anim_{key} …", end=" ", flush=True)
        try:
            g = fn(inp, **fn_kwargs)
            path = _make_rotation_gif(draw_fn, g, f"anim_{key}", color,
                                      draw_kwargs=draw_kwargs)
            gifs[key] = path
            print("done")
        except Exception as exc:
            print(f"FAILED: {exc}")
            raise

    return gifs


# ════════════════════════════════════════════════════════ PDF assembly ════════

def build_pdf(figure_paths: dict[str, Path], gif_paths: dict[str, Path]) -> None:
    styles = getSampleStyleSheet()

    H1 = ParagraphStyle("H1", parent=styles["Heading1"],
                        fontSize=16, spaceAfter=6, textColor=HexColor("#1f3b73"))
    H2 = ParagraphStyle("H2", parent=styles["Heading2"],
                        fontSize=12, spaceAfter=4, textColor=HexColor("#3a7a4d"))
    BODY = ParagraphStyle("Body", parent=styles["Normal"],
                          fontSize=9, leading=13, spaceAfter=6)
    MONO = ParagraphStyle("Mono", parent=styles["Code"],
                          fontSize=8, leading=11, leftIndent=12,
                          backColor=HexColor("#f4f4f4"))
    CAP = ParagraphStyle("Cap", parent=styles["Normal"],
                         fontSize=8, textColor=HexColor("#555555"),
                         alignment=TA_CENTER, spaceAfter=8)

    def img(name: str, w: float = 6.0) -> Image:
        return Image(str(figure_paths[name]), width=w * inch,
                     height=9 * inch, kind="proportional")

    def api_table(rows):
        tbl = Table(rows, colWidths=[1.7 * inch, 4.9 * inch])
        tbl.setStyle(TableStyle([
            ("FONT",          (0, 0), (0, -1), "Helvetica-Bold", 8),
            ("FONT",          (1, 0), (1, -1), "Helvetica",      8),
            ("TEXTCOLOR",     (0, 0), (0, -1), HexColor("#1f3b73")),
            ("VALIGN",        (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("ROWBACKGROUNDS",(0, 0), (-1, -1), [white, HexColor("#f8f8fc")]),
            ("GRID",          (0, 0), (-1, -1), 0.3, HexColor("#cccccc")),
        ]))
        return tbl

    doc = SimpleDocTemplate(
        str(PDF_PATH), pagesize=LETTER,
        leftMargin=0.85*inch, rightMargin=0.85*inch,
        topMargin=0.85*inch,  bottomMargin=0.85*inch,
    )
    story = []

    # ── cover ─────────────────────────────────────────────────────────────────
    story += [
        Spacer(1, 0.4*inch),
        Paragraph("biotuner.harmonic_geometry", H2),
        Paragraph("Phase 7 — 3-D Geometry Structures", H1),
        Spacer(1, 0.1*inch),
        Paragraph(
            "Phase 7 adds six functions that produce <b>mesh_3d</b>, "
            "<b>point_cloud_3d</b>, and <b>tree</b> (3-D) geometry from "
            "<b>HarmonicInput</b> objects. These fill the remaining gaps in "
            "the GeometryData type system and complement the existing "
            "lissajous_3d, harmonograph_3d, and chladni_field_3d_box functions.",
            BODY),
        Spacer(1, 0.08*inch),
        api_table([
            ["lissajous_tube",     "Tube mesh extruded around a 3-D Lissajous curve  →  mesh_3d"],
            ["harmonic_knot",      "Torus knot T(p,q) from dominant ratio, with tube mesh  →  mesh_3d"],
            ["harmonic_surface",   "Deformed parametric surface (torus/sphere/cylinder)  →  mesh_3d"],
            ["lsystem_3d",         "Full 6-DOF turtle L-system with yaw/pitch/roll  →  tree (3-D)"],
            ["recursive_polyhedron","Koch-style stellated Platonic solid  →  mesh_3d"],
            ["harmonic_point_cloud","Harmonic-density point cloud on sphere/torus  →  point_cloud_3d"],
        ]),
        Spacer(1, 0.1*inch),
        Paragraph(
            "<b>Rotating GIF animations</b> for each geometry type are saved alongside "
            "this PDF in <i>docs/reports/figures/anim_phase7/</i>.",
            BODY),
    ]

    # ── Section 1: lissajous_tube ─────────────────────────────────────────────
    story += [PageBreak(),
        Paragraph("1  lissajous_tube", H1),
        Paragraph(
            "Extrudes a 3-D Lissajous curve (x, y, z driven by the first three "
            "ratio components) into a tube mesh using parallel-transport frames. "
            "The tube radius is amplitude-modulated so louder harmonics create "
            "rhythmic swells along the surface.",
            BODY),
        Paragraph("<b>Signature:</b>  lissajous_tube(input, n_points=800, n_periods=6, tube_radius=0.05, n_sides=12) → mesh_3d", MONO),
        Spacer(1, 0.08*inch),
        img("p7_tube_gallery"),
        Paragraph("Fig 1. lissajous_tube for 6 chords (n_points=400, n_sides=10). Each chord produces a distinct knot topology.", CAP),
        Spacer(1, 0.08*inch),
        img("p7_tube_strip"),
        Paragraph("Fig 2. Dom7 lissajous_tube at 6 azimuth angles (0°–300°). Animation: anim_tube.gif", CAP),
    ]

    # ── Section 2: harmonic_knot ──────────────────────────────────────────────
    story += [PageBreak(),
        Paragraph("2  harmonic_knot", H1),
        Paragraph(
            "The dominant ratio p/q determines the winding numbers of a torus knot "
            "T(p, q). The knot curve is extruded into a tube; amplitude modulates "
            "the tube radius along the knot arc. The trefoil T(3,2) — produced by "
            "any 3/2 input (perfect fifth) — is the simplest non-trivial knot.",
            BODY),
        Paragraph("<b>Signature:</b>  harmonic_knot(input, n_points=600, tube_radius=0.06, n_sides=16, major_radius=2.0, minor_radius=0.7) → mesh_3d", MONO),
        Spacer(1, 0.08*inch),
        img("p7_knot_gallery"),
        Paragraph("Fig 3. T(p,q) knots for 6 ratios: 2/1, 3/2 (trefoil), 4/3, 5/3, 5/4, 7/4.", CAP),
        Spacer(1, 0.08*inch),
        img("p7_knot_strip"),
        Paragraph("Fig 4. Trefoil T(3,2) at 6 azimuth angles. Animation: anim_knot.gif", CAP),
    ]

    # ── Section 3: harmonic_surface ───────────────────────────────────────────
    story += [PageBreak(),
        Paragraph("3  harmonic_surface", H1),
        Paragraph(
            "A deformed parametric surface where each ratio p/q contributes a "
            "standing-wave ripple with angular frequencies (p, q) in the two "
            "surface parameters. Three base geometries are available:",
            BODY),
        Paragraph(
            "• <b>torus</b> — ripples deform the tube radius in (u, v) space.<br/>"
            "• <b>sphere</b> — spherical-harmonic-like deformation of the unit sphere.<br/>"
            "• <b>cylinder</b> — radial and axial deformation of an open cylinder.",
            BODY),
        Paragraph("<b>Signature:</b>  harmonic_surface(input, mode='torus', resolution=64) → mesh_3d", MONO),
        Spacer(1, 0.08*inch),
        img("p7_surface_modes", w=6.0),
        Paragraph("Fig 5. Three modes × three chords (resolution=40). Deformation depth scales with amplitude.", CAP),
        Spacer(1, 0.08*inch),
        img("p7_surface_chords"),
        Paragraph("Fig 6. Torus mode for 6 chords. Animation: anim_surface_t.gif / anim_surface_s.gif / anim_surface_c.gif", CAP),
        Spacer(1, 0.08*inch),
        img("p7_surface_strip"),
        Paragraph("Fig 7. Dom7 sphere at 6 azimuth angles.", CAP),
    ]

    # ── Section 4: lsystem_3d ─────────────────────────────────────────────────
    story += [PageBreak(),
        Paragraph("4  lsystem_3d", H1),
        Paragraph(
            "Extends the Phase-5 L-system to full 3-D turtle graphics. The turtle "
            "maintains a (H, L, U) orientation matrix and supports six rotation "
            "axes: <b>+/-</b> yaw around U, <b>^/&amp;</b> pitch around L, "
            "<b>&lt;/&gt;</b> roll around H, and <b>|</b> for a U-turn. "
            "The branch angle θ = 360/(p+q) is derived from the dominant ratio, "
            "identical to the Phase-5 formula.",
            BODY),
        Paragraph("<b>Signature:</b>  lsystem_3d(input, depth=3, step_length=1.0, rules=None, axiom='F') → tree", MONO),
        Spacer(1, 0.08*inch),
        img("p7_lsystem3d_gallery"),
        Paragraph("Fig 8. 3-D L-system for 6 chords at depth=3.", CAP),
        Spacer(1, 0.08*inch),
        img("p7_lsystem3d_depth"),
        Paragraph("Fig 9. Dom7 depth sweep (1→4). Segment count grows exponentially with depth.", CAP),
        Spacer(1, 0.08*inch),
        img("p7_lsystem3d_strip"),
        Paragraph("Fig 10. Dom7 3-D tree at 6 azimuth angles. Animation: anim_lsystem3d.gif", CAP),
    ]

    # ── Section 5: recursive_polyhedron ───────────────────────────────────────
    story += [PageBreak(),
        Paragraph("5  recursive_polyhedron", H1),
        Paragraph(
            "Koch-style recursive stellation of a Platonic solid. At each depth "
            "level every triangular face is replaced by three outer triangles plus "
            "a tetrahedral bump (3 new faces), giving 6× face count per level. "
            "The bump height is controlled by the ratio-amplitude-derived scale "
            "factor (same formula as Phase-5's recursive_polygon). Three Platonic "
            "solids are supported: tetrahedron (4 base faces), cube (12 triangulated "
            "faces), and icosahedron (20 base faces).",
            BODY),
        Paragraph("<b>Signature:</b>  recursive_polyhedron(input, depth=2, solid='icosahedron') → mesh_3d", MONO),
        Spacer(1, 0.08*inch),
        img("p7_polyhedron_solids", w=6.0),
        Paragraph("Fig 11. Three solids × three depths for the Major chord.", CAP),
        Spacer(1, 0.08*inch),
        img("p7_polyhedron_chords"),
        Paragraph("Fig 12. Icosahedron depth=2 for 6 chords (per_face_bump + apex_twist defaults). "
                  "Animation: anim_polyhedron.gif", CAP),
        Spacer(1, 0.08*inch),
        img("p7_polyhedron_diff"),
        Paragraph("Fig 12b. <b>Per-face differentiation</b>. Top row: Dom7 with the four "
                  "per_face_bump × apex_twist toggle combinations. Bottom row: four chords "
                  "rendered with auto-selected base solid (3 components → tetrahedron, 4 → cube, "
                  "5+ → icosahedron) and faces coloured by which input ratio drove their bump.", CAP),
        Spacer(1, 0.08*inch),
        img("p7_polyhedron_strip"),
        Paragraph("Fig 13. Dom7 icosahedron depth=2 at 6 azimuth angles.", CAP),
    ]

    # ── Section 6: harmonic_point_cloud ───────────────────────────────────────
    story += [PageBreak(),
        Paragraph("6  harmonic_point_cloud", H1),
        Paragraph(
            "A 3-D point cloud on a sphere or torus surface where point density "
            "is modulated by a harmonic standing-wave field. Candidate points are "
            "generated via the Fibonacci golden-angle method (maximally uniform "
            "base distribution), then filtered to keep only the top-N by field "
            "value — regions of constructive interference in ratio-frequency space "
            "become denser.",
            BODY),
        Paragraph("<b>Signature:</b>  harmonic_point_cloud(input, n_points=2000, "
                  "surface='sphere' | 'torus' | 'klein' | 'hyperbolic' | 'mos') → point_cloud_3d "
                  "with field-amplitude weights for colouring", MONO),
        Spacer(1, 0.08*inch),
        img("p7_cloud_gallery"),
        Paragraph("Fig 14. Point clouds for 6 chords alternating sphere/torus surface (n_points=800).", CAP),
        Spacer(1, 0.08*inch),
        img("p7_cloud_surfaces"),
        Paragraph("Fig 14b. The five supported surfaces for Dom7 (4 000 points each, "
                  "coloured by harmonic field amplitude). Klein bottle uses Lawson's "
                  "immersion in R³; hyperbolic lifts a Poincaré-disk sample onto a "
                  "saddle; mos arranges generators along a log-equave helix.", CAP),
        Spacer(1, 0.08*inch),
        img("p7_cloud_density"),
        Paragraph("Fig 14c. High-density spheres (8 000 points / chord) coloured by "
                  "field amplitude — the chord's interference pattern becomes visible "
                  "as bright bands of constructive resonance.", CAP),
        Spacer(1, 0.08*inch),
        img("p7_cloud_strip"),
        Paragraph("Fig 15. Dom7 sphere cloud at 6 azimuth angles. Animation: anim_cloud_sphere.gif / anim_cloud_torus.gif", CAP),
    ]

    doc.build(story)
    print(f"[ok] saved {PDF_PATH}")


# ════════════════════════════════════════════════════════════════════ main ════

def main():
    print("Generating Phase 7 figures …")
    figure_paths: dict[str, Path] = {}

    static_builders = [
        ("p7_tube_gallery",      fig_tube_chord_gallery),
        ("p7_tube_strip",        fig_tube_rotation_strip),
        ("p7_knot_gallery",      fig_knot_gallery),
        ("p7_knot_strip",        fig_knot_rotation_strip),
        ("p7_surface_modes",     fig_surface_modes),
        ("p7_surface_chords",    fig_surface_chord_sweep),
        ("p7_surface_strip",     fig_surface_rotation_strip),
        ("p7_lsystem3d_gallery", fig_lsystem3d_gallery),
        ("p7_lsystem3d_depth",   fig_lsystem3d_depth_sweep),
        ("p7_lsystem3d_strip",   fig_lsystem3d_rotation_strip),
        ("p7_polyhedron_solids", fig_polyhedron_solids),
        ("p7_polyhedron_chords", fig_polyhedron_chord_gallery),
        ("p7_polyhedron_diff",   fig_polyhedron_differentiation),
        ("p7_polyhedron_strip",  fig_polyhedron_rotation_strip),
        ("p7_cloud_gallery",     fig_point_cloud_gallery),
        ("p7_cloud_surfaces",    fig_point_cloud_surfaces),
        ("p7_cloud_density",     fig_point_cloud_density),
        ("p7_cloud_strip",       fig_point_cloud_rotation_strip),
    ]

    for key, fn in static_builders:
        print(f"  {key} …", end=" ", flush=True)
        try:
            figure_paths[key] = fn()
            print("done")
        except Exception as exc:
            print(f"FAILED: {exc}")
            raise

    gif_paths = build_all_gifs()

    print("Assembling PDF …")
    build_pdf(figure_paths, gif_paths)


if __name__ == "__main__":
    main()
