"""
Build the docs notebooks for :mod:`biotuner.harmonic_geometry.media`.

Same authoring style as the harmonic_geometry notebooks: every notebook
is described as a list of ``(kind, source)`` blocks in this file, the
``.ipynb`` is written next to the script, and (by default) every code
cell is executed in-process so the committed notebooks ship with
pre-rendered matplotlib PNG outputs.

Run:

    python docs/examples/harmonic_geometry_media/_build_notebooks.py
    python docs/examples/harmonic_geometry_media/_build_notebooks.py --no-execute
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
KERNELSPEC = {"display_name": "Python 3", "language": "python", "name": "python3"}
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
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    if kind == "code":
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": lines,
        }
    raise ValueError(kind)


def _capture_figures_as_outputs() -> list[dict]:
    outputs: list[dict] = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        try:
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
        except Exception:
            plt.close(fig)
            continue
        outputs.append({
            "output_type": "display_data",
            "data": {
                "image/png": base64.b64encode(buf.getvalue()).decode("ascii"),
                "text/plain": ["<Figure>"],
            },
            "metadata": {},
        })
        plt.close(fig)
    return outputs


def _execute_cells(cells: list[dict]) -> None:
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
            for num in plt.get_fignums():
                plt.close(num)
            continue
        cell["execution_count"] = counter
        cell["outputs"] = _capture_figures_as_outputs()


def write_notebook(name: str, cells: List[Tuple[str, str]],
                   *, execute: bool = True) -> Path:
    cell_dicts = [_cell(k, s) for k, s in cells]
    if execute:
        _execute_cells(cell_dicts)
    nb = {
        "cells": cell_dicts,
        "metadata": {"kernelspec": KERNELSPEC, "language_info": LANG_INFO},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path = HERE / f"{name}.ipynb"
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return path


# ----------------------------------------------------------------------
# Shared setup string
# ----------------------------------------------------------------------
COMMON_SETUP = '''\
import warnings
from fractions import Fraction

import numpy as np
import matplotlib.pyplot as plt

from biotuner.harmonic_geometry import HarmonicInput, plotting

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 110

# A small reference set of chord inputs used across the notebook.
CHORDS = {
    "Major": HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)]),
    "Sus4":  HarmonicInput(ratios=[Fraction(1), Fraction(4, 3), Fraction(3, 2)]),
    "Dom7":  HarmonicInput(ratios=[Fraction(1), Fraction(5, 4),
                                    Fraction(3, 2), Fraction(7, 4)]),
    "Dim7":  HarmonicInput(ratios=[Fraction(1), Fraction(6, 5),
                                    Fraction(7, 5), Fraction(12, 7)]),
}
'''


# ── 1. The Medium protocol + Pipelines + Domains ──────────────────────
NB1 = [
    ("md", """# The `media` subpackage: medium protocol, domains, pipelines

`biotuner.harmonic_geometry.media` is a family of **chord-driven response
operators**. Every medium implements ``respond(forcing) → GeometryData``
where the forcing is either a :class:`HarmonicInput` (a chord) or a
pre-computed :class:`GeometryData` (a wave field), and the output is one
of the standard geometry types — typically a 2-D field, a point cloud,
or a vector field.

Media are organised into five families:

| family | what it does |
|---|---|
| `eigenmode`     | bounded standing-wave eigenproblem (plate / sphere / elastic / ion lattice) |
| `wave_field`    | open-medium coherent superposition (interference / acoustic) |
| `parametric`    | parametric-instability surface (Faraday, cymatics) |
| `transport`     | passive redistribution on an existing wave field (Granular, Tracer, Streaming) |
| `morphogenetic` | pattern growth shaped by a chord (Crystallization, Reaction-Diffusion) |

This notebook walks through the protocol, the available `Domain` types,
and how to compose media into pipelines.
"""),
    ("code", COMMON_SETUP),
    ("md", """## A single medium — `respond(chord)` returns a GeometryData"""),
    ("code", """from biotuner.harmonic_geometry.media import RigidPlate, Rectangular

plate = RigidPlate(domain=Rectangular(Lx=1.0, Ly=1.0), resolution=128)
g     = plate.respond(CHORDS["Major"])
print("output geom_type:", g.geom_type)
print("output kind:     ", (g.metadata or {}).get("kind"))
print("output shape:    ", np.asarray(g.coordinates).shape)

fig, ax = plotting.plot_geometry(g)
ax.set_title("RigidPlate (rectangular) — Major chord");
"""),
    ("md", """## Domains — change the boundary, keep the chord

`Rectangular`, `Circular`, `PolygonDomain`, `Box3D`, and `Sphere` swap in
without touching the medium's other parameters."""),
    ("code", """from biotuner.harmonic_geometry.media import Circular, PolygonDomain

domains = [
    ("Rectangular",         Rectangular(Lx=1.0, Ly=1.0)),
    ("Circular (R=1)",      Circular(R=1.0)),
    ("Polygon (5-gon)",     PolygonDomain(n_sides=5, radius=1.0)),
]
geoms = [RigidPlate(domain=d, resolution=128)(CHORDS["Dom7"])
         for _, d in domains]
plotting.gallery(geoms, titles=[name for name, _ in domains], n_cols=3,
                 suptitle="RigidPlate — three domains, same chord (Dom7)");
"""),
    ("md", """## Pipelines — chain media

`Pipeline(A, B, C)(chord)` is equivalent to `C(B(A(chord)))`, with the
correct forcing type negotiated at each stage. Below we feed a plate
field into the `Granular` transport medium (sand on a vibrating plate)
and into the `Tracer` flow medium (light particles advected by ∇²
streamlines of the field)."""),
    ("code", """from biotuner.harmonic_geometry.media import Pipeline, Granular, Tracer

plate    = RigidPlate(domain=Rectangular(), resolution=160)
sand     = Pipeline(plate, Granular(n_particles=3000, output_mode="density"))
flow     = Pipeline(plate, Tracer(output_mode="speed"))

geoms = [plate(CHORDS["Dom7"]),
         sand (CHORDS["Dom7"]),
         flow (CHORDS["Dom7"])]
plotting.gallery(
    geoms,
    titles=["RigidPlate field", "Granular density (sand)",
            "Tracer speed (∇² streamlines)"],
    n_cols=3,
    suptitle="Pipeline composition — same chord, three stages on the same plate",
);
"""),
    ("md", """## `default_source` — transport / morphogenetic media auto-wrap

`Granular`, `Tracer`, `Streaming`, `Crystallization`, and
`ReactionDiffusion` declare a `default_source()` so they accept a
:class:`HarmonicInput` directly: the medium silently wraps the chord
through its default upstream medium. The `Granular` default is a square
:class:`RigidPlate`, so the two snippets below are equivalent."""),
    ("code", """# Explicit two-stage pipeline:
explicit = Pipeline(RigidPlate(), Granular(n_particles=3000))(CHORDS["Sus4"])

# Implicit: pass a chord straight to Granular and let it auto-wrap.
implicit = Granular(n_particles=3000)(CHORDS["Sus4"])

plotting.gallery([explicit, implicit],
                 titles=["explicit Pipeline(RigidPlate, Granular)",
                         "implicit Granular(chord)  (auto-wrapped)"],
                 n_cols=2,
                 suptitle="default_source() auto-wraps the same chord");
"""),
]


# ── 2. Eigenmode + wave-field media ───────────────────────────────────
NB2 = [
    ("md", """# Eigenmode and wave-field media

The `eigenmode` family solves a bounded standing-wave eigenproblem and
returns the chord-weighted superposition of the eigenmodes; the
`wave_field` family superposes coherent travelling waves on an unbounded
medium.

Same chord set, two very different physics."""),
    ("code", COMMON_SETUP),
    ("md", """## Eigenmode — `RigidPlate` across three domains"""),
    ("code", """from biotuner.harmonic_geometry.media import (
    RigidPlate, ClosedSurface, Elastic, PlasmaLattice,
    Rectangular, Circular, PolygonDomain,
)

geoms = [
    RigidPlate(domain=Rectangular(),               resolution=128)(CHORDS["Dom7"]),
    RigidPlate(domain=Circular(R=1.0),             resolution=128)(CHORDS["Dom7"]),
    RigidPlate(domain=PolygonDomain(n_sides=5),    resolution=96)(CHORDS["Dom7"]),
]
plotting.gallery(geoms,
                 titles=["rectangular", "circular", "pentagon"],
                 n_cols=3,
                 suptitle="RigidPlate (Dom7) across three domains");
"""),
    ("md", """## Eigenmode — `ClosedSurface` on a sphere (chord-driven Y_lm modes)

The default ``mode_rule='zonal'`` uses only $(l, 0)$ modes, which have no
longitudinal variation — the sphere lights up symmetrically around the
poles. The other three rules expose richer chord-dependent geometry:

- ``sectoral``         — uses $|m| = l$ (banana-shaped equatorial lobes)
- ``chord_balanced``   — mixes $m \\in \\{0, ±1, …, ±l\\}$ across components
- ``rounded``          — rounds $m$ proportional to the ratio magnitude
"""),
    ("code", """# Side-by-side comparison: same chord, four mode_rules
sphere_rules = ["zonal", "sectoral", "chord_balanced", "rounded"]
geoms = [ClosedSurface(mode_rule=r, max_l=12,
                        n_theta=96, n_phi=192)(CHORDS["Dom7"])
         for r in sphere_rules]
plotting.gallery(geoms, titles=sphere_rules, n_cols=4,
                 suptitle="ClosedSurface — same chord (Dom7), four mode_rules");
"""),
    ("code", """# Four chords, rendered with the visually richest mode_rule.
sphere = ClosedSurface(mode_rule="chord_balanced", max_l=12,
                       n_theta=96, n_phi=192)
geoms  = [sphere(CHORDS[n]) for n in CHORDS]
plotting.gallery(geoms, titles=list(CHORDS.keys()), n_cols=4,
                 suptitle="ClosedSurface (chord_balanced) — four chords on a sphere");
"""),
    ("md", """## Eigenmode — `Elastic` (anisotropy in the wave operator)

`Elastic` solves the elastic-wave eigenproblem on a rectangular domain
with an optional anisotropy axis. The fundamental mode is then
chord-modulated."""),
    ("code", """elastic = Elastic(resolution=160, n_modes=24)
geoms   = [elastic(CHORDS[n]) for n in CHORDS]
plotting.gallery(geoms, titles=list(CHORDS.keys()), n_cols=4,
                 suptitle="Elastic plate (anisotropic) — four chords");
"""),
    ("md", """## Eigenmode — `PlasmaLattice` (chord-tuned ion crystal)

A relaxed Coulomb crystal where the trapping potential is modulated by
the chord ratios; the medium returns the equilibrium ion positions as a
2-D point cloud."""),
    ("code", """lattice = PlasmaLattice(n_ions=36, n_steps=200,
                         chord_resolution=128, rng_seed=0)
geoms   = [lattice(CHORDS[n]) for n in CHORDS]
plotting.gallery(geoms, titles=list(CHORDS.keys()), n_cols=4,
                 suptitle="PlasmaLattice — equilibrium ion positions");
"""),
    ("md", """## Wave-field — `Interference` (five paradigms)

`Interference` is a thin facade around the five travelling-wave
paradigms exposed in :mod:`harmonic_geometry.interference_patterns`.
The same chord becomes five visually distinct fields."""),
    ("code", """from biotuner.harmonic_geometry.media import Interference

paradigms = ["harmonic", "quasicrystal", "standing_lattice", "vortex", "sources"]
geoms = [Interference(paradigm=p, resolution=192)(CHORDS["Dom7"])
         for p in paradigms]
plotting.gallery(geoms, titles=paradigms, n_cols=3,
                 suptitle="Interference — five paradigms on the Dom7 chord");
"""),
    ("md", """## Wave-field — `Acoustic` (multi-source pressure field)

`Acoustic` places ``n_sources`` emitters on a configurable layout (a ring
by default), assigns one chord component per source, and superposes the
resulting outgoing pressure waves."""),
    ("code", """from biotuner.harmonic_geometry.media import Acoustic

acoustic = Acoustic(n_sources=5, source_layout="ring", resolution=224,
                    base_frequency=8.0)
geoms = [acoustic(CHORDS[n]) for n in CHORDS]
plotting.gallery(geoms, titles=list(CHORDS.keys()), n_cols=4,
                 suptitle="Acoustic — 5-source ring, pressure field");
"""),
]


# ── 3. Parametric + transport media ───────────────────────────────────
NB3 = [
    ("md", """# Parametric and transport media

`Faraday` is a parametric-instability surface: the chord ratios pick the
unstable wave-numbers via a Mathieu-style dispersion, and the resulting
pattern is a chord-tuned cymatics image. The `transport` family then
takes a wave field (or a chord, auto-wrapped) and redistributes mass or
flow on top of it — sand grains piling on the nodes (`Granular`), tracer
particles flowing along the gradient streamlines (`Tracer`), acoustic
streaming velocities (`Streaming`)."""),
    ("code", COMMON_SETUP),
    ("md", "## `Faraday` — chord-tuned cymatics patterns"),
    ("code", """from biotuner.harmonic_geometry.media import Faraday

patterns = ["hexagonal", "stripe", "square", "random"]
geoms = [Faraday(pattern=p, resolution=192, seed=0)(CHORDS["Dom7"])
         for p in patterns]
plotting.gallery(geoms, titles=patterns, n_cols=4,
                 suptitle="Faraday — four pattern symmetries (Dom7)");
"""),
    ("md", "## `Granular` — sand-on-plate density across chords"),
    ("code", """from biotuner.harmonic_geometry.media import Granular, RigidPlate, Pipeline

# Auto-wrap: Granular's default_source is a square RigidPlate.
geoms = [Granular(n_particles=3000, output_mode="density",
                   seed=0)(CHORDS[n]) for n in CHORDS]
plotting.gallery(geoms, titles=list(CHORDS.keys()), n_cols=4,
                 suptitle="Granular density (Chladni sand) — four chords");
"""),
    ("md", """### Same chord, two output modes

`output_mode="density"` returns a smoothed 2-D density field;
`output_mode="particles"` returns the raw scatter."""),
    ("code", """g_density   = Granular(n_particles=4000, output_mode="density",   seed=0)(CHORDS["Dom7"])
g_particles = Granular(n_particles=4000, output_mode="particles", seed=0)(CHORDS["Dom7"])
plotting.gallery([g_density, g_particles],
                 titles=["density (field_2d)", "particles (point_cloud_2d)"],
                 n_cols=2,
                 suptitle="Granular — two output modes (Dom7)");
"""),
    ("md", """## `Tracer` — flow streamlines on a wave field

`Tracer` returns either a speed map (``output_mode='speed'``), a tracer
density (``output_mode='tracer_density'``), or the raw 2-D vector field
itself (``output_mode='flow'``). The vector-field output is rendered
with the new ``draw_vector_field_2d`` helper in
:mod:`harmonic_geometry.plotting` — streamlines on a magma background
proportional to the local flow magnitude."""),
    ("code", """from biotuner.harmonic_geometry.media import Tracer

t_speed   = Tracer(output_mode="speed")(CHORDS["Dom7"])
t_density = Tracer(output_mode="tracer_density")(CHORDS["Dom7"])
t_flow    = Tracer(output_mode="flow")(CHORDS["Dom7"])

plotting.gallery([t_speed, t_density, t_flow],
                 titles=["speed (|∇²ψ|)", "tracer_density",
                         "flow (vector field, streamlines)"],
                 n_cols=3,
                 suptitle="Tracer — three output modes (Dom7)");
"""),
    ("md", "## `Streaming` — acoustic-streaming velocity magnitude"),
    ("code", """from biotuner.harmonic_geometry.media import Streaming

geoms = [Streaming(output_mode="speed")(CHORDS[n]) for n in CHORDS]
plotting.gallery(geoms, titles=list(CHORDS.keys()), n_cols=4,
                 suptitle="Streaming — chord-driven acoustic-streaming speed");
"""),
]


# ── 4. Morphogenetic media ────────────────────────────────────────────
NB4 = [
    ("md", """# Morphogenetic media — pattern growth shaped by chords

The `morphogenetic` family runs an iterative pattern-formation process
whose **parameters** depend on the chord. The result is a 2-D scalar
field whose final morphology is a fingerprint of the input chord —
crystal branches, Turing patterns, etc.
"""),
    ("code", COMMON_SETUP),
    ("md", """## `Crystallization` — snowflake-style chord growth

A reaction-diffusion-like process on a hex grid; the anisotropy strength
and number of sectors are derived from the chord ratios. Higher chord
complexity → more elaborate branching."""),
    ("code", """from biotuner.harmonic_geometry.media import Crystallization

# The polygon seed plants a small V-blob at every Tonnetz-polygon vertex
# of the chord. With ``seed_branch_length > 0`` each seed is extended
# along the polygon edges, so the resulting crystal silhouette carries
# the chord's polygonal signature — Major reads as a triangle with
# branches, Sus4 as an elongated triad, Dom7 / Dim7 as four-pointed
# motifs. ``anisotropy_strength=0`` turns off the angular kernel bias
# so the chord polygon, not the lattice, drives symmetry.
chord_crystal = Crystallization(
    n_steps=1500, grid_radius=130, output_resolution=256,
    seed_strategy="polygon", seed_branch_length=6,
    anisotropy_strength=0.0, rng_seed=0,
)
geoms = [chord_crystal(CHORDS[n]) for n in CHORDS]
plotting.gallery(geoms, titles=list(CHORDS.keys()), n_cols=4,
                 suptitle="Crystallization — chord polygon seed (chord shape preserved)");
"""),
    ("md", """### Sweep `diffusion` on a single central seed

With a single central seed and no chord-polygon to bias the shape, the
underlying triangular Reiter lattice produces the classic six-armed
snowflake. ``diffusion`` then controls how far latent water can spread
between freezing steps — low diffusion gives a small compact crystal,
high diffusion gives sparser, more ramified dendrites of the same
6-fold symmetry."""),
    ("code", """diffusions = [0.2, 0.5, 0.8]
geoms = [Crystallization(n_steps=4000, grid_radius=150,
                          output_resolution=256,
                          seed_strategy="single", diffusion=d,
                          rng_seed=0)(CHORDS["Major"])
         for d in diffusions]
plotting.gallery(geoms,
                 titles=[f"diffusion = {d}" for d in diffusions], n_cols=3,
                 suptitle="Crystallization — diffusion sweep (single seed, Major chord)");
"""),
    ("md", """## `ReactionDiffusion` — chord-driven Gray-Scott patterns

Standard Gray-Scott $(U, V)$ kinetics where ``feed`` and ``kill`` rates
are derived from the chord (or supplied explicitly). The output is the
``v``-species concentration after ``n_steps`` updates — spots, stripes,
mazes, or fingerprint depending on where the chord places you in
parameter space."""),
    ("code", """from biotuner.harmonic_geometry.media import ReactionDiffusion

# The chord-derived F/K range is broad — some chords land in regions
# of the Pearson plane where V dies out (uniform-red attractor). The
# four below all map into the *active* labyrinth band so the chord
# differences read cleanly:
#   Sus4 → fine maze with stripe segments
#   Aug  → coarser maze
#   Maj7 → asymmetric stripes
#   Dim7 → tight bubble field (chord with the densest seed polygon)
active = {
    "Sus4":  CHORDS["Sus4"],
    "Aug":   HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(8, 5)]),
    "Maj7":  HarmonicInput(ratios=[Fraction(1), Fraction(5, 4),
                                    Fraction(3, 2), Fraction(15, 8)]),
    "Dim7":  CHORDS["Dim7"],
}
rd    = ReactionDiffusion(n_steps=10000, resolution=192, rng_seed=0)
geoms = [rd(c) for c in active.values()]
plotting.gallery(geoms, titles=list(active.keys()), n_cols=4,
                 suptitle="ReactionDiffusion (Gray-Scott v-field) — four chords");
"""),
    ("md", """### Canonical Pearson regimes (chord fixed, $(F, K)$ swept)

Holding the chord constant and stepping through the canonical Pearson
$(F, K)$ coordinates visits four qualitatively different attractors:
β-foam, γ-spots, κ-labyrinth, and δ-replicating dots. The central-seed
trigger makes the regimes converge faster than the polygon seed."""),
    ("code", """pts = [
    (0.018, 0.045, "β — foam"),
    (0.022, 0.051, "γ — spots"),
    (0.030, 0.057, "κ — labyrinth"),
    (0.042, 0.059, "δ — replicating dots"),
]
geoms = [ReactionDiffusion(n_steps=10000, resolution=160,
                            feed=f, kill=k, seed_strategy="single",
                            rng_seed=0)(CHORDS["Major"])
         for f, k, _ in pts]
plotting.gallery(geoms,
                 titles=[f"{lbl}\\nF={f}, K={k}" for f, k, lbl in pts],
                 n_cols=4,
                 suptitle="ReactionDiffusion — Pearson regime sweep (Major chord)");
"""),
]


NOTEBOOKS = [
    ("01_media_protocol_and_pipelines", NB1),
    ("02_eigenmode_and_wave_field",     NB2),
    ("03_parametric_and_transport",     NB3),
    ("04_morphogenetic",                NB4),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-execute", action="store_true",
        help="Skip in-process execution; emit bare notebooks without outputs.",
    )
    args = parser.parse_args()

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
