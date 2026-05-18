"""
Unified ``biotuner.harmonic_geometry`` report.

Stitches every figure produced by the per-phase report scripts (phase 1-3, 4,
5, 7) into a single PDF, plus the metrics-monitoring section.  Per-phase
figures are reused from the existing scripts — they're imported as
modules so we don't duplicate any figure code.

Run:
    python docs/reports/generate_full_report.py

Output:
    docs/reports/harmonic_geometry_full.pdf
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

# ── project root on path ──────────────────────────────────────────────────────
# This script lives at scripts/generate_full_report.py — the repo root is
# parents[1], not parents[2] like the per-phase scripts used when they lived
# under docs/reports/.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")

from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph,
    SimpleDocTemplate, Spacer, Table, TableStyle,
)

# ── output ────────────────────────────────────────────────────────────────────
# Per-phase report modules sit next to this script, but the rendered PDF
# is written under docs/reports/ alongside the figures the per-phase
# scripts emit.
SCRIPT_DIR = Path(__file__).parent
OUT_DIR    = ROOT / "docs" / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PDF_PATH   = OUT_DIR / "harmonic_geometry_full.pdf"


# ─── load existing per-phase report modules ───────────────────────────────────

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_all_modules() -> Dict[str, Any]:
    print("Loading per-phase report modules ...")
    return {
        "p13": _load_module("p13", SCRIPT_DIR / "generate_phase1_3_report.py"),
        "p4":  _load_module("p4",  SCRIPT_DIR / "generate_phase4_report.py"),
        "p5":  _load_module("p5",  SCRIPT_DIR / "generate_phase5_report.py"),
        "p7":  _load_module("p7",  SCRIPT_DIR / "generate_phase7_report.py"),
    }


# ─── styles ───────────────────────────────────────────────────────────────────

def _styles():
    base = getSampleStyleSheet()
    H0 = ParagraphStyle("H0", parent=base["Heading1"], fontSize=24,
                        leading=30, textColor=HexColor("#1f3b73"),
                        spaceBefore=20, spaceAfter=12, alignment=TA_CENTER)
    H1 = ParagraphStyle("H1", parent=base["Heading1"], fontSize=18,
                        leading=22, textColor=HexColor("#1f3b73"),
                        spaceBefore=18, spaceAfter=10)
    H2 = ParagraphStyle("H2", parent=base["Heading2"], fontSize=13,
                        leading=17, textColor=HexColor("#a23e2c"),
                        spaceBefore=12, spaceAfter=6)
    BODY = ParagraphStyle("Body", parent=base["Normal"], fontSize=10,
                          leading=14, spaceAfter=6,
                          textColor=HexColor("#222222"))
    CAP = ParagraphStyle("Cap", parent=base["Normal"], fontSize=8,
                         leading=11, alignment=TA_CENTER,
                         textColor=HexColor("#666666"), spaceAfter=10)
    CODE = ParagraphStyle("Code", parent=base["Code"], fontSize=8,
                          leading=11, fontName="Courier",
                          backColor=HexColor("#f4f4f4"),
                          leftIndent=8, spaceBefore=4, spaceAfter=8)
    return {"H0": H0, "H1": H1, "H2": H2, "BODY": BODY,
            "CAP": CAP, "CODE": CODE}


def _img(path: Path, width_in: float = 6.4) -> Image:
    """Embed an existing PNG, preserving aspect ratio."""
    from PIL import Image as PILImage
    with PILImage.open(str(path)) as pil:
        w, h = pil.size
    aspect = h / w
    return Image(str(path), width=width_in * inch,
                 height=width_in * aspect * inch)


# ─── narrative builders ───────────────────────────────────────────────────────


def _title_page(S):
    return [
        Spacer(1, 1.4 * inch),
        Paragraph("biotuner . harmonic_geometry", S["H0"]),
        Spacer(1, 0.2 * inch),
        Paragraph("Full Module Report",
                  ParagraphStyle("Sub", parent=S["H0"],
                                  fontSize=16, leading=20,
                                  textColor=HexColor("#a23e2c"),
                                  spaceBefore=0, spaceAfter=10,
                                  alignment=TA_CENTER)),
        Spacer(1, 0.5 * inch),
        Paragraph(
            "A unified visual + analytical reference for every generator in "
            "the harmonic-geometry submodule. Figures are reused verbatim "
            "from the per-phase report scripts; the ordering is "
            "<b>2-D curves &rarr; acoustic plates &rarr; fractals &rarr; "
            "generative &rarr; 3-D geometry &rarr; metrics monitoring</b>. "
            "Every generator carries a "
            "<font face='Courier'>metadata['kind']</font> tag so that "
            "<font face='Courier'>geometry_metrics()</font> can dispatch to "
            "a method-specific extractor (37 kinds supported).",
            ParagraphStyle("desc", parent=S["BODY"], fontSize=11,
                            leading=16, alignment=TA_CENTER,
                            textColor=HexColor("#444"))),
        Spacer(1, 0.7 * inch),
        Paragraph(
            "<b>Sections:</b><br/>"
            "1. 2-D curves &mdash; Lissajous, harmonograph<br/>"
            "2. Polygons &amp; circular forms<br/>"
            "3. Chladni acoustic plates<br/>"
            "4. Fractal &amp; number-theoretic structures<br/>"
            "5. Generative self-similar geometry<br/>"
            "6. 3-D geometry &mdash; tubes, knots, surfaces, polyhedra, clouds<br/>"
            "7. Metrics monitoring &mdash; do science with it<br/>"
            "8. Validation &amp; references",
            ParagraphStyle("toc", parent=S["BODY"], fontSize=11,
                            leading=18, alignment=TA_CENTER,
                            textColor=HexColor("#1f3b73"))),
        PageBreak(),
    ]


# ── Section 1+2: Lissajous + harmonograph (phase 1-3) ─────────────────────────

def _section_curves(S, p13):
    print("  Section 1: 2-D curves & harmonograph")
    return [
        Paragraph("1. 2-D curves &mdash; Lissajous &amp; harmonograph", S["H1"]),
        Paragraph(
            "Two ratio-driven curve families. Lissajous traces sine motion "
            "on orthogonal axes (closed when the frequency ratio is rational). "
            "The harmonograph adds exponential damping per axis so the trajectory "
            "decays into the origin &mdash; the time-evolved Lissajous of an "
            "oscilloscope or pendulum drawing machine.",
            S["BODY"]),
        Paragraph("Lissajous family", S["H2"]),
        _img(p13.fig_lissajous_gallery()),
        Paragraph("Six 2-D Lissajous variants showing how p:q and phase shape the figure.",
                   S["CAP"]),
        _img(p13.fig_lissajous_3d_knot()),
        Paragraph("3-D Lissajous knots from pairwise-coprime frequency triples.",
                   S["CAP"]),
        _img(p13.fig_lissajous_topology_demo()),
        Paragraph("lissajous_topology reports lobe counts, closure, intersections, and period.",
                   S["CAP"]),
        _img(p13.fig_lissajous_phase_drift()),
        Paragraph("lissajous_phase_drift &mdash; rotating ratio with linearly-evolving phase.",
                   S["CAP"]),
        _img(p13.fig_lissajous_pairwise_grid()),
        Paragraph("lissajous_pairwise_grid &mdash; the matrix of all pairwise ratios.",
                   S["CAP"]),
        _img(p13.fig_lissajous_compound()),
        Paragraph("lissajous_compound &mdash; an N-component HarmonicInput collapsed to one curve.",
                   S["CAP"]),
        Paragraph("Harmonograph", S["H2"]),
        _img(p13.fig_harmonograph_examples()),
        Paragraph("Lateral, rotary (&Omega;=0.05 Hz), and 3-D harmonographs.",
                   S["CAP"]),
        _img(p13.fig_harmonograph_damping()),
        Paragraph("Damping-rate sweep showing energy-decay envelope.",
                   S["CAP"]),
        PageBreak(),
    ]


# ── Section 2: polygons & circular ────────────────────────────────────────────

def _section_polygons(S, p13):
    print("  Section 2: polygons & circular")
    return [
        Paragraph("2. Polygons &amp; circular forms", S["H1"]),
        Paragraph(
            "Schl&auml;fli star polygons {n/k}, modular times-table circles, "
            "ratio-driven tuning circles, rose curves, epicycloids and "
            "hypocycloids. The chord-driven <i>times_table_from_input</i> "
            "overlays one edge family per harmonic ratio so a chord paints "
            "itself as colour-coded cardioid envelopes on a modular ring.",
            S["BODY"]),
        _img(p13.fig_star_polygons()),
        Paragraph("Schl&auml;fli {n/k} polygons. {6/2} has gcd=2, returned as a polygon_set.",
                   S["CAP"]),
        _img(p13.fig_times_table_circles()),
        Paragraph("times_table_circle &mdash; integer / non-integer multipliers; "
                   "cardioid envelopes from modular multiplication.",
                   S["CAP"]),
        _img(p13.fig_times_table_chord_driven()),
        Paragraph("times_table_from_input &mdash; one edge family per chord-tone, "
                   "overlaid as colour layers.",
                   S["CAP"]),
        _img(p13.fig_tuning_circle()),
        Paragraph("Just-intonation diatonic on an octave-equave circle.",
                   S["CAP"]),
        _img(p13.fig_rose_curves()),
        Paragraph("rose_curve &mdash; r=cos(k&theta;) with rational k.",
                   S["CAP"]),
        _img(p13.fig_cycloids()),
        Paragraph("epicycloid &amp; hypocycloid &mdash; ratios determine cusp count.",
                   S["CAP"]),
        PageBreak(),
    ]


# ── Section 3: chladni ────────────────────────────────────────────────────────

def _section_chladni(S, p13):
    print("  Section 3: chladni")
    return [
        Paragraph("3. Chladni acoustic plates", S["H1"]),
        Paragraph(
            "Standing-wave displacement fields on rectangular, circular, "
            "polygon (FDM eigenmodes), and 3-D box plates. A "
            "<font face='Courier'>HarmonicInput</font> can be lifted to a "
            "plate via <font face='Courier'>chladni_from_input</font>, which "
            "maps each ratio to an (m, n) mode pair via a chosen strategy "
            "(stern_brocot / continued_fraction / rounded / best_simple). "
            "Nodal lines and surfaces are extracted with marching squares / "
            "marching cubes (scikit-image).",
            S["BODY"]),
        _img(p13.fig_chladni_rectangular()),
        Paragraph("Pure rectangular cos&middot;cos modes with nodal contours.",
                   S["CAP"]),
        _img(p13.fig_chladni_rect_sum()),
        Paragraph("Three-mode superposition with non-trivial phases.",
                   S["CAP"]),
        _img(p13.fig_chladni_rect_zoom_insets()),
        Paragraph("Seven-mode superposition with three zoom insets.",
                   S["CAP"]),
        _img(p13.fig_chladni_circular()),
        Paragraph("Circular plates &mdash; J<sub>n</sub>(&alpha;<sub>n,m</sub>r/R)cos(n&theta;).",
                   S["CAP"]),
        _img(p13.fig_chladni_polygon()),
        Paragraph("Polygon plates &mdash; lowest Dirichlet eigenmodes via FDM.",
                   S["CAP"]),
        _img(p13.fig_chladni_3d_box()),
        Paragraph("3-D nodal surfaces extracted with marching cubes.",
                   S["CAP"]),
        _img(p13.fig_chladni_from_input()),
        Paragraph("chladni_from_input &mdash; same Dom7 chord across all four plate kinds.",
                   S["CAP"]),
        _img(p13.fig_chladni_from_input_strategies()),
        Paragraph("Strategy comparison on an 11-limit chord (chosen modes printed).",
                   S["CAP"]),
        _img(p13.fig_chladni_from_input_with_nodal()),
        Paragraph("Chord set with maximally different mode sets, with extracted nodal lines.",
                   S["CAP"]),
        PageBreak(),
    ]


# ── Section 4: fractal (phase 4) ──────────────────────────────────────────────

def _section_fractal(S, p4):
    print("  Section 4: fractal")
    print("    rendering p4 figures ...")
    figs = {
        "sbt_layouts": p4.fig_stern_brocot_layouts(),
        "sbt_depth":   p4.fig_stern_brocot_depth_sweep(),
        "sbt_chords":  p4.fig_stern_brocot_chord_highlight(),
        "cf_gallery":  p4.fig_cf_ratio_gallery(),
        "cf_depth":    p4.fig_cf_depth_sweep(),
        "farey_circle": p4.fig_farey_circle_gallery(),
        "farey_line":   p4.fig_farey_line_gallery(),
        "sub_polar":    p4.fig_subharmonic_polar(),
        "ifs_chords":      p4.fig_ifs_chord_gallery(),
        "ifs_contractions": p4.fig_ifs_contraction_sweep(),
        "ifs_density":      p4.fig_ifs_density_zoom(),
    }
    return [
        Paragraph("4. Fractal &amp; number-theoretic structures", S["H1"]),
        Paragraph(
            "The Stern-Brocot tree of all positive rationals (annotated by "
            "harmonicity); continued-fraction rectangle tilings; the Farey "
            "sequence; per-peak subharmonic descent trees; and chaos-game "
            "iterated function systems. Each function emits a structured "
            "<font face='Courier'>GeometryData</font> with method-specific "
            "metadata.",
            S["BODY"]),
        Paragraph("Stern-Brocot tree", S["H2"]),
        _img(figs["sbt_layouts"]),
        Paragraph("Hyperbolic vs. flat-tree layouts.", S["CAP"]),
        _img(figs["sbt_depth"]),
        Paragraph("Depth sweep (3-6) on the hyperbolic layout.", S["CAP"]),
        _img(figs["sbt_chords"]),
        Paragraph("Chord-tone highlighting at depth=7 (within 15 cents).", S["CAP"]),
        Paragraph("Continued-fraction rectangles", S["H2"]),
        _img(figs["cf_gallery"]),
        Paragraph("CF rectangle tilings for six JI ratios.", S["CAP"]),
        _img(figs["cf_depth"]),
        Paragraph("Depth sweep for 7/4 = [1; 1, 3].", S["CAP"]),
        Paragraph("Farey sequence", S["H2"]),
        _img(figs["farey_circle"]),
        Paragraph("Circle layouts for orders 3-20 (dot size &prop; 1/denominator).",
                   S["CAP"]),
        _img(figs["farey_line"]),
        Paragraph("Line layouts stacked for orders 5-30.", S["CAP"]),
        Paragraph("Subharmonic tree", S["H2"]),
        _img(figs["sub_polar"]),
        Paragraph("Polar layout: each root-peak gets its own angular sector.",
                   S["CAP"]),
        Paragraph("IFS attractor", S["H2"]),
        _img(figs["ifs_chords"]),
        Paragraph("IFS attractors for six chords (ratio_inverse, 30k pts).",
                   S["CAP"]),
        _img(figs["ifs_contractions"]),
        Paragraph("Contraction-mode comparison for Dom7.", S["CAP"]),
        _img(figs["ifs_density"]),
        Paragraph("200k-point log-density heatmap with zoom insets.", S["CAP"]),
        PageBreak(),
    ]


# ── Section 5: generative (phase 5) ───────────────────────────────────────────

def _section_generative(S, p5):
    print("  Section 5: generative")
    print("    rendering p5 figures ...")
    figs = {
        "ls_chords":  p5.fig_lsystem_chord_gallery(),
        "ls_depth":   p5.fig_lsystem_depth_sweep(),
        "poly_chords": p5.fig_polygon_chord_gallery(),
        "poly_depth":  p5.fig_polygon_depth_sweep(),
        "poly_sides":  p5.fig_polygon_sides_sweep(),
        "tuning_chords": p5.fig_tuning_chord_gallery(),
        "tuning_levels": p5.fig_tuning_levels_sweep(),
        "tuning_equave": p5.fig_tuning_equave_sweep(),
    }
    return [
        Paragraph("5. Generative self-similar geometry", S["H1"]),
        Paragraph(
            "L-system branching plants whose turning angles come from chord "
            "ratios; Koch-like recursive polygons; multi-level harmonic-"
            "spiral lattices. Cantor-rhythm and Julia-set generators have "
            "been moved to a future rhythm/fractal module respectively (see "
            "<font face='Courier'>docs/notes/rhythm_module_todos.md</font>).",
            S["BODY"]),
        Paragraph("L-system from ratios", S["H2"]),
        _img(figs["ls_chords"]),
        Paragraph("Six chords at fixed depth=4.", S["CAP"]),
        _img(figs["ls_depth"]),
        Paragraph("Depth sweep (Major chord).", S["CAP"]),
        Paragraph("Recursive polygon", S["H2"]),
        _img(figs["poly_chords"]),
        Paragraph("Six chords at depth=3.", S["CAP"]),
        _img(figs["poly_depth"]),
        Paragraph("Depth sweep.", S["CAP"]),
        _img(figs["poly_sides"]),
        Paragraph("n_sides sweep at depth=3.", S["CAP"]),
        Paragraph("Self-similar tuning lattice", S["H2"]),
        _img(figs["tuning_chords"]),
        Paragraph("Six chords on concentric generation rings.", S["CAP"]),
        _img(figs["tuning_levels"]),
        Paragraph("Generation-level sweep (Major).", S["CAP"]),
        _img(figs["tuning_equave"]),
        Paragraph("Equave sweep (Dom7) &mdash; octave / tritave / quadrave / quintave.",
                   S["CAP"]),
        PageBreak(),
    ]


# ── Section 6: 3-D geometry (phase 7) ─────────────────────────────────────────

def _section_3d(S, p7):
    print("  Section 6: 3-D geometry")
    print("    rendering p7 static figures (no GIF re-render) ...")
    static_builders = [
        ("p7_tube_gallery",      p7.fig_tube_chord_gallery,
         "lissajous_tube &mdash; 6 chords."),
        ("p7_tube_strip",        p7.fig_tube_rotation_strip,
         "Dom7 tube at 6 azimuth angles."),
        ("p7_knot_gallery",      p7.fig_knot_gallery,
         "harmonic_knot &mdash; T(p,q) for 6 ratios."),
        ("p7_knot_strip",        p7.fig_knot_rotation_strip,
         "Perfect-fifth knot rotation."),
        ("p7_surface_modes",     p7.fig_surface_modes,
         "harmonic_surface &mdash; 3 modes &times; 3 chords."),
        ("p7_surface_chords",    p7.fig_surface_chord_sweep,
         "Torus surface for 6 chords."),
        ("p7_surface_strip",     p7.fig_surface_rotation_strip,
         "Sphere surface rotation."),
        ("p7_lsystem3d_gallery", p7.fig_lsystem3d_gallery,
         "lsystem_3d for 6 chords (depth=3)."),
        ("p7_lsystem3d_depth",   p7.fig_lsystem3d_depth_sweep,
         "Depth sweep (Major)."),
        ("p7_lsystem3d_strip",   p7.fig_lsystem3d_rotation_strip,
         "Dom7 lsystem rotation."),
        ("p7_polyhedron_solids", p7.fig_polyhedron_solids,
         "3 base solids &times; 3 depths."),
        ("p7_polyhedron_chords", p7.fig_polyhedron_chord_gallery,
         "Icosahedron depth=2 for 6 chords."),
        ("p7_polyhedron_diff",   p7.fig_polyhedron_differentiation,
         "Per-face differentiation: per_face_bump &times; apex_twist toggles + "
         "auto-solid chord row."),
        ("p7_polyhedron_strip",  p7.fig_polyhedron_rotation_strip,
         "Dom7 polyhedron rotation."),
        ("p7_cloud_gallery",     p7.fig_point_cloud_gallery,
         "Point clouds &mdash; 6 chords alternating sphere / torus."),
        ("p7_cloud_surfaces",    p7.fig_point_cloud_surfaces,
         "5 surfaces (sphere / torus / klein / hyperbolic / mos)."),
        ("p7_cloud_density",     p7.fig_point_cloud_density,
         "8 000-point spheres coloured by field amplitude."),
        ("p7_cloud_strip",       p7.fig_point_cloud_rotation_strip,
         "Dom7 sphere cloud rotation."),
    ]
    out = [
        Paragraph("6. 3-D geometry", S["H1"]),
        Paragraph(
            "Tubes, torus knots, deformed surfaces, 3-D L-system trees, "
            "stellated polyhedra, and harmonic-field point clouds. All of "
            "the static figures from the Phase 7 report are reproduced here "
            "in the same order. The corresponding rotating GIFs ship "
            "alongside the original Phase 7 PDF "
            "(<font face='Courier'>docs/reports/figures/anim_phase7/</font>).",
            S["BODY"]),
    ]
    for label, builder, caption in static_builders:
        path = builder()
        out.append(_img(path))
        out.append(Paragraph(caption, S["CAP"]))
    out.append(PageBreak())
    return out


# ── Section 7: metrics monitoring (in-process) ────────────────────────────────

def _section_metrics(S, p13):
    print("  Section 7: metrics monitoring")
    return [
        Paragraph("7. Metrics monitoring &mdash; do science with it", S["H1"]),
        Paragraph(
            "<font face='Courier'>biotuner.harmonic_geometry.metrics</font> is "
            "a strictly <b>geometry-side</b> analytical layer. For raw "
            "harmonic-content stats on the underlying signal, use "
            "<font face='Courier'>biotuner.metrics</font> and "
            "<font face='Courier'>BiotunerGroup</font> directly &mdash; not "
            "duplicated here. What this module adds is per-method extractors "
            "for every recognised generator, plus an append-only "
            "<font face='Courier'>MetricsLog</font> with CSV / JSON export.",
            S["BODY"]),
        Paragraph(
            "<b>Quick API</b><br/>"
            "<font face='Courier'>"
            "geometry_metrics(geom)                                # dispatches "
            "on metadata['kind'] (37 supported)<br/>"
            "list_supported_kinds()                                # full list<br/>"
            "sequence_metrics(seq, generator, **kw)                # dict[str, ndarray]<br/>"
            "compare(geometries, labels=...)                       # column-oriented<br/>"
            "log = MetricsLog(); log.log_geometry(g); log.to_csv(path)<br/>"
            "plotting.plot_metric_radar(rows, labels=..., metrics=...)<br/>"
            "plotting.plot_metric_trajectory(seq, generator=..., normalize=True)"
            "</font>",
            S["CODE"]),
        _img(p13.fig_metrics_radar()),
        Paragraph(
            "Six chords compared on six harmonic metrics. Dissonance metrics "
            "are auto-inverted by <font face='Courier'>normalize_metrics</font>. "
            "When a metric has zero variance across the input rows (e.g. "
            "subharmonic_tension = 0 for every clean JI chord because the "
            "metric finds a perfect common subharmonic), the radar shows it "
            "at 0.5 = consensus / no information.",
            S["CAP"]),
        _img(p13.fig_metrics_per_method()),
        Paragraph(
            "Per-method extractor sample &mdash; one representative generator "
            "per family, top 4 method-specific metrics each.",
            S["CAP"]),
        _img(p13.fig_metrics_trajectory()),
        Paragraph(
            "sequence_metrics trajectory: a 16-frame log-interpolation "
            "Major &rarr; Dom7 &rarr; Dim7. Useful for tracking how harmonic "
            "identity evolves through a chord transition (or the output of "
            "<font face='Courier'>biotuner.transitional_harmony</font>).",
            S["CAP"]),
        PageBreak(),
    ]


def _section_validation(S):
    return [
        Paragraph("8. Validation &amp; references", S["H1"]),
        Paragraph(
            "All generators are covered by deterministic shape / math / metadata "
            "tests. The current full <font face='Courier'>tests/harmonic_geometry/"
            "</font> suite is <b>>400 tests</b> across 9 modules "
            "(2-D, 3-D, fractal, generative, chladni, polygon-circular, plotting, "
            "metrics, inputs).",
            S["BODY"]),
        Paragraph(
            "Run with <font face='Courier'>pytest tests/harmonic_geometry/ -q</font>. "
            "All tests are deterministic (seeded RNG where needed); no flaky "
            "tests are expected.",
            S["BODY"]),
        Paragraph("Optional dependencies", S["H2"]),
        Paragraph(
            "<b>scikit-image</b> &mdash; required for nodal-line / nodal-surface "
            "extraction (<font face='Courier'>chladni_nodal_lines</font>, "
            "<font face='Courier'>chladni_nodal_surfaces</font>). "
            "<b>biotuner.metrics</b> &mdash; required for the harmonic-metrics "
            "summary; the geometry generators themselves do not depend on it.",
            S["BODY"]),
        Paragraph("Per-phase report scripts", S["H2"]),
        Paragraph(
            "The figure-builder functions still live in the original "
            "<font face='Courier'>generate_phase*_report.py</font> scripts "
            "&mdash; this unified report imports them as modules and "
            "orchestrates the layout. The 3-D animations referenced in "
            "section 6 ship as <font face='Courier'>.gif</font> files in "
            "<font face='Courier'>docs/reports/figures/anim_phase7/</font>.",
            S["BODY"]),
    ]


# ─── PDF assembly ─────────────────────────────────────────────────────────────


def build_pdf() -> Path:
    mods = load_all_modules()
    S = _styles()

    print("Building unified PDF ...")
    doc = SimpleDocTemplate(
        str(PDF_PATH), pagesize=LETTER,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        title="biotuner.harmonic_geometry — Full Report",
    )
    story = []
    story += _title_page(S)
    story += _section_curves(S, mods["p13"])
    story += _section_polygons(S, mods["p13"])
    story += _section_chladni(S, mods["p13"])
    story += _section_fractal(S, mods["p4"])
    story += _section_generative(S, mods["p5"])
    story += _section_3d(S, mods["p7"])
    story += _section_metrics(S, mods["p13"])
    story += _section_validation(S)

    doc.build(story)
    print(f"[ok] saved {PDF_PATH}")
    return PDF_PATH


if __name__ == "__main__":
    build_pdf()
