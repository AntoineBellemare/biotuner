"""
Build the docs notebooks for :mod:`biotuner.harmonic_geometry`.

Each notebook is generated from a list of ``(kind, source)`` blocks so the
content stays version-controlled in this single file. The generated
``.ipynb`` files sit next to this script and are committed to the repo so
Jupyter Book / Sphinx can pick them up without needing a build step.

By default, each notebook is also *executed in-process* — every code cell
is run with matplotlib in Agg mode, and the resulting figures are baked
into the notebook as base-64 PNG outputs. That matches the rest of the
docs (e.g. ``scale_construction/scale_construction.ipynb``) which ship
with pre-rendered cell outputs so the HTML build doesn't depend on a
running kernel. Skip execution with ``--no-execute`` for a quick rebuild.

Run:

    python docs/examples/harmonic_geometry/_build_notebooks.py
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
KERNELSPEC = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
LANG_INFO = {
    "name": "python",
    "mimetype": "text/x-python",
    "file_extension": ".py",
    "pygments_lexer": "ipython3",
    "codemirror_mode": {"name": "ipython", "version": 3},
    "nbconvert_exporter": "python",
}


def _cell(kind: str, src: str) -> dict:
    lines = src.splitlines(keepends=True)
    if kind == "md":
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": lines,
        }
    if kind == "code":
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines,
        }
    raise ValueError(kind)


def _capture_figures_as_outputs(exec_count: int) -> list[dict]:
    """Render every currently-open matplotlib figure as a base-64 PNG
    cell output, then close them. The returned list matches the
    nbformat-4 ``outputs`` schema and is ready to be assigned to a
    code cell."""
    outputs: list[dict] = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        try:
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
        except Exception:
            plt.close(fig)
            continue
        png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        outputs.append({
            "output_type": "display_data",
            "data": {
                "image/png": png_b64,
                "text/plain": ["<Figure>"],
            },
            "metadata": {},
        })
        plt.close(fig)
    return outputs


def _execute_cells(cells: list[dict]) -> None:
    """Run the code cells in-place, populating ``execution_count`` and
    ``outputs``. Errors are recorded as ``stream`` outputs so the build
    doesn't abort silently — the smoke test catches real failures."""
    ns: dict = {"__name__": "__main__"}
    counter = 0
    for cell in cells:
        if cell["cell_type"] != "code":
            continue
        counter += 1
        src = "".join(cell["source"])
        try:
            exec(compile(src, "<notebook-cell>", "exec"), ns)
        except Exception as exc:  # noqa: BLE001
            cell["execution_count"] = counter
            cell["outputs"] = [{
                "output_type": "stream",
                "name": "stderr",
                "text": [f"{type(exc).__name__}: {exc}\n"],
            }]
            # Close any half-rendered figures
            for num in plt.get_fignums():
                plt.close(num)
            continue
        cell["execution_count"] = counter
        cell["outputs"] = _capture_figures_as_outputs(counter)


def write_notebook(name: str, cells: List[Tuple[str, str]],
                   *, execute: bool = True) -> Path:
    cell_dicts = [_cell(k, s) for k, s in cells]
    if execute:
        _execute_cells(cell_dicts)
    nb = {
        "cells": cell_dicts,
        "metadata": {
            "kernelspec": KERNELSPEC,
            "language_info": LANG_INFO,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = HERE / f"{name}.ipynb"
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return path


# ----------------------------------------------------------------------
# Notebook content
# ----------------------------------------------------------------------

COMMON_SETUP = '''\
import warnings
from fractions import Fraction

import numpy as np
import matplotlib.pyplot as plt

from biotuner.harmonic_geometry import HarmonicInput, plotting

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 110
'''

# ── 1. Lissajous + Harmonograph ───────────────────────────────────────
NB1 = [
    ("md", """# Lissajous and harmonograph curves

`biotuner.harmonic_geometry` ships closed-form Lissajous figures and damped
double-pendulum harmonographs. Both turn a ratio-set into a 2-D or 3-D
trajectory whose visual complexity is a direct expression of the input
harmonics — coprime ratios produce closed knots, near-rational ratios drift
through dense rosettes, and a small amount of damping makes the trace decay
inward like a real harmonograph drawing.

This notebook reproduces the Lissajous and harmonograph figures from the
`harmonic_geometry` report.
"""),
    ("code", COMMON_SETUP),
    ("md", "## Lissajous gallery — 2-D curves at different ratios"),
    ("code", """from biotuner.harmonic_geometry import lissajous_2d

cases = [
    (Fraction(1, 1),  np.pi/2, "1:1, φ=π/2 (circle)"),
    (Fraction(3, 2),  np.pi/2, "3:2, φ=π/2"),
    (Fraction(5, 4),  0.0,     "5:4, φ=0"),
    (Fraction(7, 4),  np.pi/4, "7:4, φ=π/4"),
    (Fraction(5, 3),  np.pi/3, "5:3, φ=π/3"),
    (Fraction(9, 7),  np.pi/6, "9:7, φ=π/6"),
]
geoms  = [lissajous_2d(r, phase=p, n_points=2000) for r, p, _ in cases]
titles = [lab for _, _, lab in cases]
plotting.gallery(geoms, titles=titles, n_cols=3,
                 suptitle="lissajous_2d — six ratios");
"""),
    ("md", """## 3-D Lissajous knots

When three coprime integer frequencies share a single trajectory the curve
closes into a knot — the spatial counterpart of a chord."""),
    ("code", """from biotuner.harmonic_geometry import lissajous_3d

geoms = [
    lissajous_3d(ratios=[3, 4, 5], phases=[0.0, np.pi/4, np.pi/2], n_points=4000),
    lissajous_3d(ratios=[2, 3, 7], phases=[0.0, np.pi/3, np.pi/5], n_points=4000),
]
plotting.gallery(geoms, titles=["(3, 4, 5)", "(2, 3, 7)"], n_cols=2,
                 suptitle="lissajous_3d — knotted trajectories");
"""),
    ("md", """## Pairwise grid and compound curves

`lissajous_pairwise_grid` traces every component pair of a chord, so the
diagonal contains 1:1 circles and off-diagonals encode interval structure.
`lissajous_compound` sums every component on each axis, giving a single
amplitude-weighted figure of the whole chord."""),
    ("code", """from biotuner.harmonic_geometry import lissajous_pairwise_grid, lissajous_compound

inp = HarmonicInput(ratios=[1, Fraction(3, 2), Fraction(5, 4)], base_freq=100.0)
grid = lissajous_pairwise_grid(inp, n_points=400)

labels = ["1/1", "3/2", "5/4"]
n = len(grid)
fig, axes = plt.subplots(n, n, figsize=(5.5, 5.5))
for i in range(n):
    for j in range(n):
        plotting.draw_curve_2d(grid[i][j], axes[i, j], lw=0.6)
        plotting.axis_clean(axes[i, j])
        axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
        if i == 0: axes[i, j].set_title(labels[j], fontsize=9)
        if j == 0: axes[i, j].set_ylabel(labels[i], fontsize=9)
fig.suptitle("lissajous_pairwise_grid (3-component chord)")
fig.tight_layout();
"""),
    ("code", """inp = HarmonicInput(
    ratios=[1, Fraction(3, 2), Fraction(5, 4), Fraction(7, 4)],
    amplitudes=[1.0, 0.7, 0.5, 0.3], base_freq=100.0,
)
g = lissajous_compound(inp, n_points=4000, n_periods=2)
fig, ax = plotting.plot_geometry(g, lw=0.6)
ax.set_title("lissajous_compound — just-intonation tetrad");
"""),
    ("md", """## Phase drift

A slowly-changing phase between two components un-closes a Lissajous figure
and lets it precess through every member of its family."""),
    ("code", """from biotuner.harmonic_geometry import lissajous_phase_drift

geoms = [
    lissajous_phase_drift(ratio=Fraction(3, 2), drift_rate=d, duration=4.0, sr=600)
    for d in (0.5, 1.5, 4.0)
]
plotting.gallery(geoms, titles=[f"drift = {d} rad/s" for d in (0.5, 1.5, 4.0)],
                 n_cols=3, suptitle="lissajous_phase_drift, ratio 3:2 over 4 s",
                 draw_kwargs={"lw": 0.5});
"""),
    ("md", """## Harmonograph examples

A real harmonograph couples two damped pendulums; `harmonograph_lateral`,
`harmonograph_rotary`, and `harmonograph_3d` cover the three common rigs.
Pass a `HarmonicInput` with `peaks` and per-component `damping` and the
trace will decay exactly as a physical apparatus does."""),
    ("code", """from biotuner.harmonic_geometry import (
    harmonograph_3d, harmonograph_lateral, harmonograph_rotary,
)

inp = HarmonicInput(
    peaks=[2.01, 3.02, 5.0, 7.03],
    amplitudes=[1.0, 0.8, 0.6, 0.4],
    phases=[0.0, np.pi/5, np.pi/3, np.pi/7],
    damping=[0.05, 0.04, 0.06, 0.05],
)
g_lat = harmonograph_lateral(inp, duration=40.0, sr=400)
g_rot = harmonograph_rotary(inp, duration=40.0, sr=400, rotation_freq=0.05)
g_3d  = harmonograph_3d(inp, duration=40.0, sr=400)

plotting.gallery([g_lat, g_rot, g_3d],
                 titles=["harmonograph_lateral", "harmonograph_rotary", "harmonograph_3d"],
                 n_cols=3, draw_kwargs={"lw": 0.5},
                 suptitle="harmonograph family — same input, three rigs");
"""),
    ("md", """## Effect of damping

Zero damping gives a bounded but persistent trace; even mild damping makes
the figure spiral inward to a point."""),
    ("code", """inp_zero  = HarmonicInput(peaks=[2.0, 3.0, 5.0, 7.0],
                          amplitudes=[1.0, 0.8, 0.6, 0.4],
                          damping=[0.0]*4)
inp_decay = HarmonicInput(peaks=[2.0, 3.0, 5.0, 7.0],
                          amplitudes=[1.0, 0.8, 0.6, 0.4],
                          damping=[0.15]*4)
geoms = [harmonograph_lateral(inp_zero,  duration=30.0, sr=300),
         harmonograph_lateral(inp_decay, duration=30.0, sr=300)]
plotting.gallery(geoms,
                 titles=["damping = 0", "damping = 0.15"],
                 n_cols=2, draw_kwargs={"lw": 0.5},
                 suptitle="harmonograph — damping comparison");
"""),
]

# ── 2. Chladni + Spherical harmonics ─────────────────────────────────
NB2 = [
    ("md", """# Chladni plates and spherical harmonics

Eigenmodes of the wave equation on a bounded medium give standing-wave
patterns whose nodal lines/surfaces fall on the points where the field is
zero. `harmonic_geometry` exposes rectangular, circular, polygonal, and
3-D box plates, plus the spherical-harmonic basis on a sphere — the
closed-surface analogue of a Chladni plate.

This notebook reproduces the plate and sphere figures from the report.
"""),
    ("code", COMMON_SETUP),
    ("md", "## Rectangular plate — pure modes"),
    ("code", """from biotuner.harmonic_geometry import chladni_field_rectangular

modes = [(2, 3), (4, 5), (3, 3)]
geoms = [chladni_field_rectangular([m], resolution=257) for m in modes]
plotting.gallery(geoms, titles=[f"mode {m}" for m in modes], n_cols=3,
                 suptitle="chladni_field_rectangular — pure modes");
"""),
    ("md", """### Superposition of modes

Summing several modes with prescribed amplitudes and phases is what a real
plate does under a chord-shaped excitation."""),
    ("code", """g = chladni_field_rectangular(
    modes=[(2, 3), (3, 5), (4, 1)],
    amps=[1.0, 0.6, 0.4],
    phases=[0.0, np.pi/3, np.pi/7],
    resolution=257,
)
fig, ax = plotting.plot_geometry(g)
ax.set_title("Sum of three rectangular modes");
"""),
    ("md", "## Circular plate (Bessel modes)"),
    ("code", """from biotuner.harmonic_geometry import chladni_field_circular

cases = [([1], [0], "(m=1, n=0)"),
         ([2], [1], "(m=2, n=1)"),
         ([1], [3], "(m=1, n=3)"),
         ([3], [2], "(m=3, n=2)")]
geoms  = [chladni_field_circular(mr, ma, R=1.0, resolution=257) for mr, ma, _ in cases]
titles = [lab for _, _, lab in cases]
plotting.gallery(geoms, titles=titles, n_cols=4,
                 suptitle="chladni_field_circular — pure modes");
"""),
    ("md", """## Polygonal plate (finite-difference)

For arbitrary convex polygons the eigenproblem is solved on a finite
difference mesh."""),
    ("code", """from biotuner.harmonic_geometry import chladni_field_polygon

cases = [(3, 0), (3, 3), (3, 6), (5, 0), (5, 2), (5, 5)]
geoms = [chladni_field_polygon([m], n_sides=ns, resolution=96)
         for ns, m in cases]
titles = [f"{ns}-gon, mode {m}" for ns, m in cases]
plotting.gallery(geoms, titles=titles, n_cols=3,
                 draw_kwargs={"show_nodal": True},
                 suptitle="chladni_field_polygon (FDM)");
"""),
    ("md", """## From a chord input — `chladni_from_input`

`chladni_from_input` ties a `HarmonicInput` chord to a plate: each ratio
is mapped to a 2-D or 3-D mode index, the modes are summed, and the result
is returned as a `field_2d`/`field_3d` GeometryData."""),
    ("code", """from biotuner.harmonic_geometry import chladni_from_input

chords = {
    "Major": HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)]),
    "Sus4":  HarmonicInput(ratios=[Fraction(1), Fraction(4, 3), Fraction(3, 2)]),
    "Dom7":  HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2),
                                    Fraction(7, 4)]),
}
geoms = [chladni_from_input(c, plate="rectangular",
                             plate_kwargs={"resolution": 129})
         for c in chords.values()]
plotting.gallery(geoms, titles=list(chords.keys()), n_cols=3,
                 suptitle="chladni_from_input — three chords on a rectangular plate");
"""),
    ("md", """## Spherical harmonics — closed-surface eigenmodes

`single_spherical_harmonic` and `spherical_harmonic_field` produce the
`(l, m)` eigenmodes of the Laplacian on the unit sphere. The plot below
shows three pure modes rendered as colour on the sphere surface."""),
    ("code", """from biotuner.harmonic_geometry import single_spherical_harmonic

cases = [(2, 0), (3, 2), (5, 3)]
geoms  = [single_spherical_harmonic(l, m, n_theta=80, n_phi=160)
          for l, m in cases]
titles = [f"Y_{l}^{m}" for l, m in cases]
plotting.gallery(geoms, titles=titles, n_cols=3,
                 suptitle="single_spherical_harmonic — pure (l, m) modes");
"""),
    ("md", "### Chord on a sphere"),
    ("code", """from biotuner.harmonic_geometry import spherical_harmonic_from_input

dom7 = HarmonicInput(ratios=[Fraction(1), Fraction(5, 4),
                              Fraction(3, 2), Fraction(7, 4)])
g = spherical_harmonic_from_input(dom7, n_theta=96, n_phi=192)
fig, ax = plotting.plot_geometry(g)
ax.set_title("spherical_harmonic_from_input — Dom7");
"""),
]

# ── 3. Polygons, circles, cycloids ────────────────────────────────────
NB3 = [
    ("md", """# Tuning circles, polygons, and cycloids

A pitch-class or ratio set wraps onto a circle exactly once per equave, and
the resulting layout exposes interval structure at a glance. This notebook
reproduces the circular/polygonal figures from the report — tuning circles,
star polygons, times-table circles, and the rose / cycloid families.
"""),
    ("code", COMMON_SETUP),
    ("md", "## Star polygons"),
    ("code", """from biotuner.harmonic_geometry import star_polygon

cases = [(5, 2, "{5/2} pentagram"),
         (7, 3, "{7/3} heptagram"),
         (8, 3, "{8/3} octagram"),
         (6, 2, "{6/2} compound")]
geoms  = [star_polygon(n, k, radius=1.0) for n, k, _ in cases]
titles = [lab for _, _, lab in cases]
plotting.gallery(geoms, titles=titles, n_cols=4,
                 suptitle="star_polygon — Schläfli {n/k}");
"""),
    ("md", """## Times-table circles

Connect point `k` on a circle to point `k × m mod n`. Distinct integer
multipliers carve out cardioid / nephroid / epicycloid envelopes."""),
    ("code", """from biotuner.harmonic_geometry import times_table_circle

cases = [(200, 2), (200, 3), (200, 5), (300, 7.5)]
geoms  = [times_table_circle(n_points=n, multiplier=m, radius=1.0) for n, m in cases]
titles = [f"n={n}, ×{m}" for n, m in cases]
plotting.gallery(geoms, titles=titles, n_cols=4,
                 suptitle="times_table_circle — multipliers");
"""),
    ("md", """## Tuning circle

A 7-note just-intonation diatonic mapped onto a circle, with point size
encoding amplitude."""),
    ("code", """from biotuner.harmonic_geometry import tuning_circle

inp = HarmonicInput(
    ratios=[Fraction(1, 1), Fraction(9, 8), Fraction(5, 4), Fraction(4, 3),
            Fraction(3, 2), Fraction(5, 3), Fraction(15, 8)],
    amplitudes=[1.0, 0.6, 0.9, 0.7, 1.0, 0.7, 0.5],
)
g = tuning_circle(inp, radius=1.0)
fig, ax = plotting.plot_geometry(g)
coords = np.asarray(g.coordinates)
for (x, y), lab in zip(coords, ["1/1", "9/8", "5/4", "4/3",
                                 "3/2", "5/3", "15/8"]):
    ax.annotate(lab, (x, y), xytext=(8, 8), textcoords="offset points",
                fontsize=8)
ax.set_title("tuning_circle — just-intonation diatonic");
"""),
    ("md", "## Rose curves"),
    ("code", """from biotuner.harmonic_geometry import rose_curve

cases = [Fraction(3, 1), Fraction(5, 1), Fraction(2, 1), Fraction(7, 3)]
geoms  = [rose_curve(r, n_points=2000) for r in cases]
titles = [f"r = cos({r.numerator}/{r.denominator} θ)" for r in cases]
plotting.gallery(geoms, titles=titles, n_cols=4, draw_kwargs={"lw": 0.7},
                 suptitle="rose_curve — various ratios");
"""),
    ("md", "## Epicycloid / hypocycloid"),
    ("code", """from biotuner.harmonic_geometry import epicycloid, hypocycloid

cases = [
    (epicycloid, Fraction(3, 1), "epicycloid 3:1"),
    (epicycloid, Fraction(5, 2), "epicycloid 5:2"),
    (hypocycloid, Fraction(4, 1), "hypocycloid 4:1 (astroid)"),
    (hypocycloid, Fraction(5, 2), "hypocycloid 5:2"),
]
geoms = [fn(r, n_points=2000) for fn, r, _ in cases]
plotting.gallery(geoms, titles=[lab for _, _, lab in cases], n_cols=4,
                 draw_kwargs={"lw": 0.8},
                 suptitle="cycloid family");
"""),
]

# ── 4. Fractal / generative ───────────────────────────────────────────
NB4 = [
    ("md", """# Fractal layouts and generative L-systems

Phases 4–5 of `harmonic_geometry` expose deterministic fractal generators
that turn a chord into a self-similar layout: Stern-Brocot tree, continued
fractions, Farey sequences, iterated-function systems, L-systems, and
recursive polygons. Every generator returns a `GeometryData` that the
`plotting` submodule renders without further arguments.
"""),
    ("code", COMMON_SETUP),
    ("code", """from biotuner.harmonic_geometry import (
    HarmonicInput, plotting,
    stern_brocot_tree, continued_fraction_rectangles, farey_sequence_layout,
    subharmonic_tree, ifs_harmonic,
    lsystem_from_ratios, recursive_polygon, self_similar_tuning,
)

CHORDS = {
    "Major": HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)]),
    "Sus4":  HarmonicInput(ratios=[Fraction(1), Fraction(4, 3), Fraction(3, 2)]),
    "Dom7":  HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2),
                                    Fraction(7, 4)]),
    "Dim7":  HarmonicInput(ratios=[Fraction(1), Fraction(6, 5),
                                    Fraction(7, 5), Fraction(12, 7)]),
}
"""),
    ("md", """## Stern-Brocot tree — chord-driven layouts

`stern_brocot_tree` builds the binary tree of mediants up to a given depth
and lays it out either as a regular dyadic tree (``layout="tree"``) or in
the Poincaré disk (``layout="hyperbolic"``)."""),
    ("code", """geoms  = [stern_brocot_tree(CHORDS["Major"], max_depth=6, layout=lay)
          for lay in ("tree", "hyperbolic")]
plotting.gallery(geoms, titles=["tree (dyadic)", "hyperbolic (Poincaré disk)"],
                 n_cols=2,
                 suptitle="stern_brocot_tree — two layouts (Major)");
"""),
    ("md", "## Continued-fraction rectangles"),
    ("code", """ratios = [Fraction(22, 7), Fraction(355, 113), Fraction(7, 5)]
geoms = [continued_fraction_rectangles(r) for r in ratios]
plotting.gallery(geoms, titles=[str(r) for r in ratios], n_cols=3,
                 suptitle="continued_fraction_rectangles");
"""),
    ("md", "## Farey sequence layouts"),
    ("code", """orders = [5, 8, 12]
geoms_circle = [farey_sequence_layout(o, layout="circle") for o in orders]
geoms_line   = [farey_sequence_layout(o, layout="line")   for o in orders]
plotting.gallery(geoms_circle + geoms_line,
                 titles=[f"circle, order {o}" for o in orders]
                       + [f"line, order {o}"   for o in orders],
                 n_cols=3,
                 suptitle="farey_sequence_layout — circle vs line");
"""),
    ("md", "## Subharmonic tree — chord-driven polar layout"),
    ("code", """geoms  = [subharmonic_tree(CHORDS[n], depth=4, n_harmonics=4, layout="polar")
          for n in ("Major", "Sus4", "Dom7", "Dim7")]
plotting.gallery(geoms, titles=list(CHORDS.keys()), n_cols=4,
                 suptitle="subharmonic_tree (polar)");
"""),
    ("md", """## Iterated-function system from a chord

Each ratio defines a contractive affine map; their non-empty intersection
is the IFS attractor. Sampled via the chaos game."""),
    ("code", """rng = np.random.default_rng(0)
geoms = [ifs_harmonic(CHORDS[n], n_points=30_000,
                       contraction="ratio_inverse", rng=rng)
         for n in ("Major", "Sus4", "Dom7", "Dim7")]
plotting.gallery(geoms, titles=list(CHORDS.keys()), n_cols=4,
                 suptitle="ifs_harmonic — chord attractors");
"""),
    ("md", "## L-system curves and recursive polygons"),
    ("code", """geoms = [lsystem_from_ratios(CHORDS[n], depth=4) for n in CHORDS]
plotting.gallery(geoms, titles=list(CHORDS.keys()), n_cols=4,
                 suptitle="lsystem_from_ratios (depth 4)");
"""),
    ("code", """geoms = [recursive_polygon(CHORDS[n], depth=3) for n in CHORDS]
plotting.gallery(geoms, titles=list(CHORDS.keys()), n_cols=4,
                 suptitle="recursive_polygon (depth 3)");
"""),
    ("md", "## Self-similar tuning (nested ratio scaffolds)"),
    ("code", """geoms = [self_similar_tuning(CHORDS[n], n_levels=4) for n in CHORDS]
plotting.gallery(geoms, titles=list(CHORDS.keys()), n_cols=4,
                 suptitle="self_similar_tuning (4 levels)");
"""),
]

# ── 5. 3-D geometry ───────────────────────────────────────────────────
NB5 = [
    ("md", """# 3-D harmonic geometry

Phase 7 of `harmonic_geometry` adds 3-D generators: harmonic knots,
Lissajous tubes, harmonic surfaces (sphere, torus, Klein), point clouds,
3-D L-systems, and recursive polyhedra. All return a `GeometryData` whose
`plotting.plot_geometry()` does the right thing — wireframes, mesh, or
3-D scatter — with no extra arguments.
"""),
    ("code", COMMON_SETUP),
    ("code", """from biotuner.harmonic_geometry import (
    harmonic_knot, harmonic_point_cloud, harmonic_surface, lissajous_tube,
    lsystem_3d, recursive_polyhedron,
)

CHORDS = {
    "Major": HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)]),
    "Sus4":  HarmonicInput(ratios=[Fraction(1), Fraction(4, 3), Fraction(3, 2)]),
    "Dom7":  HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2),
                                    Fraction(7, 4)]),
    "Dim7":  HarmonicInput(ratios=[Fraction(1), Fraction(6, 5),
                                    Fraction(7, 5), Fraction(12, 7)]),
}
"""),
    ("md", "## Harmonic knots — 3 coprime frequencies weave a closed curve"),
    ("code", """geoms = [harmonic_knot(CHORDS[n], n_points=300, n_sides=10) for n in CHORDS]
plotting.gallery(geoms, titles=list(CHORDS.keys()), n_cols=4,
                 suptitle="harmonic_knot — 4 chords");
"""),
    ("md", "## Lissajous tubes — extruded 3-D Lissajous curves"),
    ("code", """geoms = [lissajous_tube(CHORDS[n], n_points=400, n_sides=10) for n in CHORDS]
plotting.gallery(geoms, titles=list(CHORDS.keys()), n_cols=4,
                 suptitle="lissajous_tube — 4 chords");
"""),
    ("md", """### Rotating a single knot

`rotation_strip` is a convenience that renders the same 3-D geometry at
several azimuths in a single horizontal strip — useful when a static
notebook can't capture a 3-D shape from one viewpoint alone."""),
    ("code", """g = harmonic_knot(CHORDS["Dom7"], n_points=400, n_sides=14)
plotting.rotation_strip(g, n_strip=5,
                        suptitle="harmonic_knot Dom7 — rotation strip");
"""),
    ("md", "## Harmonic surfaces — sphere, torus, cylinder"),
    ("code", """surfaces = ["sphere", "torus", "cylinder"]
geoms  = [harmonic_surface(CHORDS["Dom7"], mode=s, resolution=40)
          for s in surfaces]
plotting.gallery(geoms, titles=surfaces, n_cols=3,
                 suptitle="harmonic_surface (Dom7) — three modes");
"""),
    ("md", "## Harmonic point clouds"),
    ("code", """geoms = [harmonic_point_cloud(CHORDS[n], n_points=2000) for n in CHORDS]
plotting.gallery(geoms, titles=list(CHORDS.keys()), n_cols=4,
                 suptitle="harmonic_point_cloud — 4 chords");
"""),
    ("md", "## 3-D L-systems and recursive polyhedra"),
    ("code", """geoms = [lsystem_3d(CHORDS[n], depth=3) for n in CHORDS]
plotting.gallery(geoms, titles=list(CHORDS.keys()), n_cols=4,
                 suptitle="lsystem_3d (depth 3)");
"""),
    ("code", """geoms = [recursive_polyhedron(CHORDS[n], depth=2) for n in CHORDS]
plotting.gallery(geoms, titles=list(CHORDS.keys()), n_cols=4,
                 suptitle="recursive_polyhedron (depth 2)");
"""),
]

# ── 6. Metrics and transitions ────────────────────────────────────────
NB6 = [
    ("md", """# Geometry metrics and chord transitions

Every generator in `harmonic_geometry` returns a `GeometryData` that
`geometry_metrics()` summarises into a scalar dict — span, energy, active
fraction, edge statistics, mode counts, etc. The same scalars feed the
radar plot and the trajectory plot in this notebook, so you can compare
chords on the same plate or watch metrics evolve along a chord morph.

The `transitions` submodule provides `interpolate_chords`,
`fade_in_components`, and `blend_fields` to drive animation frames — used
in the GeometryV2 reel and reproducible here on still frames.
"""),
    ("code", COMMON_SETUP),
    ("md", "## Radar — six chords on a rectangular Chladni plate"),
    ("code", """from biotuner.harmonic_geometry import chladni_from_input, geometry_metrics

chords = {
    "Major": HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)]),
    "Minor": HarmonicInput(ratios=[Fraction(1), Fraction(6, 5), Fraction(3, 2)]),
    "Sus4":  HarmonicInput(ratios=[Fraction(1), Fraction(4, 3), Fraction(3, 2)]),
    "Aug":   HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(8, 5)]),
    "Dom7":  HarmonicInput(ratios=[Fraction(1), Fraction(5, 4),
                                    Fraction(3, 2), Fraction(7, 4)]),
    "Dim7":  HarmonicInput(ratios=[Fraction(1), Fraction(6, 5),
                                    Fraction(7, 5), Fraction(12, 7)]),
}
rows = [
    geometry_metrics(chladni_from_input(
        inp, plate="rectangular", plate_kwargs={"resolution": 129},
    ))
    for inp in chords.values()
]
fig, _ = plotting.plot_metric_radar(
    rows, labels=list(chords.keys()),
    metrics=["n_modes", "energy", "peak_abs",
             "active_frac", "field_std", "zero_crossing_frac"],
    title="geometry_metrics — chladni_from_input across six chords",
);
"""),
    ("md", """## Trajectory — recursive polygon along a chord morph

`recursive_polygon` was chosen because its scalar metrics (perimeter,
scale_factor, bump_angle, area) vary continuously with the chord — every
interpolation step changes the output. Topology-driven generators jump in
discrete steps and are less informative here."""),
    ("code", """from biotuner.harmonic_geometry import interpolate_chords, recursive_polygon

major7 = HarmonicInput(ratios=[Fraction(1), Fraction(5, 4),
                                Fraction(3, 2), Fraction(15, 8)])
dom7   = HarmonicInput(ratios=[Fraction(1), Fraction(5, 4),
                                Fraction(3, 2), Fraction(7, 4)])
dim7   = HarmonicInput(ratios=[Fraction(1), Fraction(6, 5),
                                Fraction(7, 5), Fraction(12, 7)])
def morph(a, b, n):
    return [interpolate_chords(a, b, i/(n-1)) for i in range(n)]

frames = morph(major7, dom7, 12) + morph(dom7, dim7, 12)
metrics_per_frame = [geometry_metrics(recursive_polygon(f, depth=3))
                     for f in frames]

# plot_metric_trajectory expects a dict {name: array(T)} — transpose the
# per-frame list of dicts to the column-oriented layout it wants.
keys_to_plot = ["perimeter", "scale_factor", "bump_angle", "area"]
metric_arrays = {
    k: np.array([m.get(k, np.nan) for m in metrics_per_frame], dtype=float)
    for k in keys_to_plot
}

fig, _ = plotting.plot_metric_trajectory(
    metric_arrays,
    metrics=keys_to_plot,
    title="recursive_polygon metrics across Major7 → Dom7 → Dim7",
);
"""),
    ("md", """## `interpolate_chords` — visualising the morph

Sampling the morph at fixed `t` and rendering with `recursive_polygon`
gives a visual preview of what the animation does between two chords."""),
    ("code", """ts     = np.linspace(0.0, 1.0, 6)
frames = [interpolate_chords(major7, dom7, float(t)) for t in ts]
geoms  = [recursive_polygon(f, depth=3) for f in frames]
plotting.gallery(geoms, titles=[f"t={t:.2f}" for t in ts], n_cols=6,
                 suptitle="interpolate_chords: Major7 → Dom7 (depth-3 polygon)");
"""),
    ("md", """## `fade_in_components` — growing a chord by extension

Useful for animations that build up a chord one component at a time —
ramp `t` from 0 to 1 and the extra components appear without disturbing
the ratio of the existing ones."""),
    ("code", """from biotuner.harmonic_geometry import fade_in_components

base = HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)],
                     amplitudes=[1.0, 0.8, 0.7])
ext  = HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2),
                              Fraction(7, 4), Fraction(15, 8)],
                     amplitudes=[1.0, 0.8, 0.7, 0.6, 0.5])
ts     = np.linspace(0.0, 1.0, 5)
frames = [fade_in_components(base, ext, float(t)) for t in ts]
geoms  = [recursive_polygon(f, depth=3) for f in frames]
plotting.gallery(geoms, titles=[f"t={t:.2f}" for t in ts], n_cols=5,
                 suptitle="fade_in_components: triad → extended pentad");
"""),
    ("md", """## `blend_fields` — pixel-space crossfade between two algorithms

Used in the reel to morph between two paradigms on a shared grid — for
instance a Chladni plate fading into a quasicrystal field. The two
geometries must share `field_grid` shape."""),
    ("code", """from biotuner.harmonic_geometry import (
    blend_fields, chladni_field_rectangular,
    harmonic_interference_field_2d,
)
a = chladni_field_rectangular([(2, 3), (3, 5), (4, 1)], resolution=129)
b = harmonic_interference_field_2d(
    HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)]),
    resolution=129, extent=1.0,
)
ts     = np.linspace(0.0, 1.0, 5)
frames = [blend_fields(a, b, float(t), require_same_grid=False) for t in ts]
plotting.gallery(frames, titles=[f"t={t:.2f}" for t in ts], n_cols=5,
                 suptitle="blend_fields: Chladni → interference (pixel crossfade)");
"""),
]


NOTEBOOKS = [
    ("01_lissajous_and_harmonograph",     NB1),
    ("02_chladni_and_spherical_harmonics", NB2),
    ("03_circular_patterns",               NB3),
    ("04_fractal_and_generative",          NB4),
    ("05_three_dimensional",               NB5),
    ("06_metrics_and_transitions",         NB6),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-execute", action="store_true",
        help="Skip in-process execution; emit bare notebooks without outputs.",
    )
    args = parser.parse_args()

    # Make the worktree's biotuner importable when we execute cells.
    worktree = HERE.parents[2]
    if str(worktree) not in sys.path:
        sys.path.insert(0, str(worktree))

    for name, cells in NOTEBOOKS:
        path = write_notebook(name, cells, execute=not args.no_execute)
        n_code = sum(1 for k, _ in cells if k == "code")
        suffix = "with outputs" if not args.no_execute else "no execution"
        print(f"wrote {path.relative_to(HERE.parents[1])}  "
              f"({len(cells)} cells, {n_code} code, {suffix})")


if __name__ == "__main__":
    main()
