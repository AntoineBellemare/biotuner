"""
Build the cymatics-style Chladni demo notebook.

Same authoring pattern as docs/examples/harmonic_geometry{,_media}: each
notebook is declared as a list of ``(kind, source)`` blocks below, the
``.ipynb`` is written next to this script, and every code cell is
executed in-process so the committed notebook ships with pre-rendered
matplotlib PNG outputs.

Run:

    python docs/examples/chladni_cymatics/_build_notebooks.py
    python docs/examples/chladni_cymatics/_build_notebooks.py --no-execute
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
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=110,
                        facecolor=fig.get_facecolor())
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


# ---------------------------------------------------------------------------
# Notebook 1 — full tour of the new mode schemes + symmetry + density + sand
# ---------------------------------------------------------------------------

NB = [
    ("md", """# Chladni cymatics — mode schemes, D4 symmetry, sand grains

`biotuner.harmonic_geometry.media.eigenmode.RigidPlate` now exposes four
chord→mode schemes, two D4 symmetry options, and a Gaussian-of-zero-
crossing density transform — together they reproduce the iconic
sand-on-square-plate Chladni patterns from integer-ratio chord inputs.
This notebook walks every option, then animates the morph between
chords.

The cymatics schemes treat each chord ratio as an **integer
wavenumber** `(m, n)` on the plate. Two routes to feed them:

- pass a small-integer chord list directly (e.g. ``Major = [4, 5, 6]``,
  ``Dom7 = [4, 5, 6, 7]``) — the natural just-intonation form,
- or pass a ``HarmonicInput`` of ``Fraction`` ratios and let
  ``chord_to_int_modes`` (LCM of denominators) recover an integer form.

The first route is preferred for "clean" patterns: ``Dim7`` as
``[5, 6, 7, 9]`` reads as a clean cross-lattice, while the same chord
expressed as ``[1, 6/5, 7/5, 12/7]`` blows up under LCM to
``[35, 42, 49, 60]`` and tiles the plate with sub-pixel detail. The
new ``max_mode`` cap (section 6) scales such cases back into a
visible range without losing the chord's ratio structure.
"""),
    ("code", """import warnings
import os
from fractions import Fraction

import numpy as np
import matplotlib.pyplot as plt

from biotuner.harmonic_geometry import HarmonicInput, plotting
from biotuner.harmonic_geometry.media import RigidPlate, Granular, Pipeline
from biotuner.harmonic_geometry.media.eigenmode.rigid_plate import (
    chord_to_int_modes,
    chladni_field_pairwise,
    chladni_field_triple_antisymmetric,
    chladni_nodal_density,
)

warnings.filterwarnings("ignore")
plt.rcParams["figure.dpi"] = 110

# Small-integer just-intoned chord representations. These are the natural
# wavenumber form for the cymatics schemes — pass them straight to
# chladni_field_pairwise / chladni_field_triple_antisymmetric.
CHORDS_INT = {
    "Major": [4, 5, 6],
    "Sus4":  [6, 8, 9],
    "Dom7":  [4, 5, 6, 7],
    "Dim7":  [5, 6, 7, 9],
}

# Same chords as HarmonicInput / Fraction ratios — for the LCM-conversion
# section below and for routing through the RigidPlate medium.
CHORDS_HI = {
    "Major": HarmonicInput(ratios=[Fraction(r, CHORDS_INT["Major"][0])
                                    for r in CHORDS_INT["Major"]]),
    "Sus4":  HarmonicInput(ratios=[Fraction(r, CHORDS_INT["Sus4"][0])
                                    for r in CHORDS_INT["Sus4"]]),
    "Dom7":  HarmonicInput(ratios=[Fraction(r, CHORDS_INT["Dom7"][0])
                                    for r in CHORDS_INT["Dom7"]]),
    "Dim7":  HarmonicInput(ratios=[Fraction(r, CHORDS_INT["Dim7"][0])
                                    for r in CHORDS_INT["Dim7"]]),
}
"""),
    ("md", """## 1. Integer chord ratios — lossless, and when to cap

For chords already in just-intonation small-integer form
(``Major = [4, 5, 6]``), ``chord_to_int_modes`` round-trips them
unchanged. For chords in 12-TET-flavoured ``Fraction`` form the
LCM-of-denominators conversion can blow up — that is the maths of the
rationals, not a bug — and the pattern becomes a fine-grained lattice
that reads as "dots". The new ``max_mode`` cap rescues that case by
scaling all wavenumbers proportionally down to a visible range
(ratios are preserved):
"""),
    ("code", """from biotuner.harmonic_geometry.media.eigenmode.rigid_plate import _cap_modes

for name, c in CHORDS_HI.items():
    int_form = chord_to_int_modes(c.to_ratios())
    capped   = _cap_modes(int_form, 12)
    print(f"  {name:6s}  Fraction = {[str(r) for r in c.to_ratios()]}")
    print(f"          → int_form = {int_form}    capped@12 = {capped}")
"""),
    ("md", """## 2. The four `mode_scheme` options

All four schemes are eigenmode-family operators on a rectangular plate.
The classical ``per_ratio`` scheme assigns one Stern-Brocot mode pair to
each ratio. The cymatics schemes sum modes over **pairs** (or
**triples**) of ratios — emphasising the harmonic *relationships* in
the chord rather than the individual partials."""),
    ("code", """dom7_int = CHORDS_INT["Dom7"]
dom7_hi  = CHORDS_HI["Dom7"]

# per_ratio routes through HarmonicInput (Stern-Brocot mapping per ratio);
# the cymatics routines accept the integer chord list directly.
g_pr = RigidPlate(mode_scheme="per_ratio", resolution=400)(dom7_hi)
g_pa = chladni_field_pairwise(dom7_int, antisymmetric=True,  resolution=400)
g_ps = chladni_field_pairwise(dom7_int, antisymmetric=False, resolution=400)
g_tr = chladni_field_triple_antisymmetric(dom7_int, resolution=400)

plotting.gallery(
    [g_pr, g_pa, g_ps, g_tr], n_cols=4,
    titles=["per_ratio (classical)",
            "pairwise antisymmetric",
            "pairwise symmetric",
            "triple antisymmetric"],
    suptitle="Four mode schemes — Dom7 [4, 5, 6, 7]",
    fig_width=14.0,
);
"""),
    ("md", """## 3. D4 symmetrisation — `none` / `d4_max` / `d4_sum`

The dihedral group D4 has 8 elements (4 rotations × 2 reflections).

- **``none``** — leave the field as-is.
- **``d4_max``** — element-wise *maximum* over the orbit. Non-linear
  symmetriser that preserves bright features → the crystalline lattice
  look (the original cymatics demo used this).
- **``d4_sum``** — orbit *average*. Linear, smoother; enforces strict
  D4 symmetry without amplifying any one orientation.
"""),
    ("code", """geoms = [
    chladni_field_pairwise(dom7_int, antisymmetric=True, symmetry=s, resolution=400)
    for s in ("none", "d4_sum", "d4_max")
]
plotting.gallery(
    geoms, n_cols=3,
    titles=["symmetry='none'", "symmetry='d4_sum' (avg)", "symmetry='d4_max'"],
    suptitle="D4 symmetrisation — pairwise antisymmetric (Dom7)",
    fig_width=13.0,
);
"""),
    ("md", """## 4. Nodal density — `exp(-w² / σ²)`

The classical Chladni experiment shows sand **collecting on the nodal
lines** (where the plate displacement is zero). Mathematically captured
by the Gaussian-of-zero-crossing transform:

$$d(x, y) = \\exp\\!\\left(-\\frac{w(x, y)^2}{\\sigma^2}\\right)$$

`σ` controls the stripe width — small `σ` gives razor-thin nodal lines,
large `σ` gives wider soft bands.
"""),
    ("code", """field = chladni_field_pairwise(
    dom7_int, antisymmetric=True, symmetry="d4_max", resolution=400,
)
sigmas = [0.02, 0.05, 0.10, 0.25]
geoms = [chladni_nodal_density(field, sigma=s) for s in sigmas]
plotting.gallery(
    geoms, n_cols=4,
    titles=[f"σ = {s}" for s in sigmas],
    suptitle="Nodal density (Dom7, pairwise antisym + D4 max) — σ sweep",
    fig_width=14.0,
);
"""),
    ("md", """### Antinodal density

``chladni_nodal_density(..., mode="antinodal")`` gives the complement —
sand on the antinodes instead of the nodes. A coral / fingerprint
texture rather than the iconic lattice."""),
    ("code", """plotting.gallery(
    [chladni_nodal_density(field, sigma=0.05, mode="nodal"),
     chladni_nodal_density(field, sigma=0.05, mode="antinodal")],
    n_cols=2,
    titles=["nodal_density", "antinodal_density"],
    suptitle="Nodal vs antinodal — pairwise antisym + D4 max (Dom7)",
    fig_width=10.0,
);
"""),
    ("md", """## 5. Chord-fingerprint gallery — pairwise antisym + D4 max + nodal density

The full cymatics stack gives a distinctive fingerprint per chord:"""),
    ("code", """sigma = 0.05
geoms = [
    chladni_nodal_density(
        chladni_field_pairwise(
            CHORDS_INT[name], antisymmetric=True, symmetry="d4_max",
            resolution=400,
        ),
        sigma=sigma,
    )
    for name in CHORDS_INT
]
plotting.gallery(
    geoms, n_cols=4, titles=list(CHORDS_INT.keys()),
    suptitle="Chladni-cymatics signature per chord (small-int forms)",
    fig_width=16.0,
);
"""),
    ("md", """## 6. Sand-grain particle rendering

The most photographic look: instead of an ``imshow`` of the density,
stochastically sample N particles from it and draw small white dots on
black. ``Granular(nodal_emphasis=True, ...)`` bypasses its Boltzmann
formulation and uses ``exp(-w²/σ²)`` directly on the incoming field.
Compose any field-producing medium with
``Granular(output_mode="particles", nodal_emphasis=True)``.

Visible-feature size depends on the chord's wavenumber range — small
integer chords (max ~10) give bold flowing curves; let high-LCM chords
through unscaled and they crumble into the lattice. With small-int
forms, no cap is needed and the patterns read clearly:
"""),
    ("code", """def sand_for(chord_int, n_particles=180_000, sigma=0.06):
    field = chladni_field_pairwise(
        chord_int, antisymmetric=True, symmetry="d4_max", resolution=400,
    )
    return Granular(
        output_mode="particles", nodal_emphasis=True, sigma=sigma,
        n_particles=n_particles, seed=0,
    ).respond(field)

fig, axes = plt.subplots(1, 4, figsize=(18, 4.7), facecolor="black")
for ax, (name, chord_int) in zip(axes, CHORDS_INT.items()):
    pts = sand_for(chord_int).coordinates
    ax.set_facecolor("black")
    ax.scatter(pts[:, 0], pts[:, 1], s=0.5, c="white", alpha=0.45,
               linewidths=0)
    ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(name, color="white", fontsize=14)
fig.suptitle("Cymatics sand-grain rendering (180 k particles per chord)",
             color="white", fontsize=13, y=1.02)
fig.tight_layout(pad=0.5);
"""),
    ("md", """### `max_mode` rescues high-LCM chords

If you pass ``Dim7`` as its 12-TET-flavoured ``Fraction`` form
(``[1, 6/5, 7/5, 12/7]``), ``chord_to_int_modes`` blows it up to
``[35, 42, 49, 60]`` — sub-pixel lattice. ``max_mode=...`` scales
proportionally:"""),
    ("code", """dim7_hi   = CHORDS_HI["Dim7"]  # Fraction form — round-trips to small ints here
dim7_high_lcm = [35, 42, 49, 60]  # the same chord re-expressed in 12-TET-flavoured form
print(f"  Dim7 (12-TET-style Fraction LCM): {dim7_high_lcm}")

plotting.gallery(
    [chladni_field_pairwise(dim7_high_lcm, antisymmetric=True, symmetry="d4_max",
                             resolution=400),  # no cap
     chladni_field_pairwise(dim7_high_lcm, antisymmetric=True, symmetry="d4_max",
                             resolution=400, max_mode=12)],
    n_cols=2,
    titles=[f"no cap (max wavenumber = {max(dim7_high_lcm)})",
            "max_mode=12 (scaled proportionally)"],
    suptitle="Dim7 blown-up wavenumbers — cap restores visible features",
    fig_width=10.0,
);
"""),
    ("md", """## 6b. `pair_subset` — keeping curves bold at any chord size

For a chord with ``n`` ratios, ``chladni_field_pairwise`` sums one
antisymmetric mode per ratio pair. The mathematics behave qualitatively
differently as ``n`` grows:

- ``n = 3`` (3 pairs): the sum's zero-set is a **curve** — the iconic
  Chladni form. This is what your favourite cymatics demos show.
- ``n = 4`` (6 pairs): the sum's zero-set becomes a much more restricted
  **set of isolated points** — the renderings goes "dotty / speckled".
  It's not a rendering bug; the field genuinely has near-zero amplitude
  only at scattered intersections, not along curves.

The ``pair_subset`` parameter exposes a fix: rather than summing all
``C(n, 2)`` pairs, sum a subset of ``n - 1`` pairs — keeping the field
qualitatively *like a 3-ratio chord* (continuous curves) regardless of
chord size:

- ``pair_subset="auto"`` (the default): ``'all'`` for ``n ≤ 3``,
  ``'root'`` for ``n ≥ 4``. Bold curves at every chord size.
- ``pair_subset="all"``: classical sum, all pairs. Use when you want
  the literal mathematical fingerprint (will dot for ``n ≥ 4``).
- ``pair_subset="adjacent"``: only consecutive pairs.
- ``pair_subset="root"``: pairs involving the first (root) ratio.
- A literal ``list`` of ``(m, n)`` tuples for full control.
"""),
    ("code", """from biotuner.harmonic_geometry.media.eigenmode.rigid_plate import (
    _auto_resolution_for_modes,
)

fig, axes = plt.subplots(2, 4, figsize=(18, 9.5), facecolor="black")

# Top row: pair_subset="all" — the canonical sum. 3-ratio chords look
# beautiful; 4-ratio chords dot.
# Bottom row: pair_subset="auto" — fixes 4-ratio chords automatically.
gallery_chords = [
    ("Major [4,5,6]",        [4, 5, 6]),
    ("[3,4,7] (demo ref)",   [3, 4, 7]),
    ("Dom7 [4,5,6,7]",       [4, 5, 6, 7]),
    ("11-limit\\n[9,11,13,17]", [9, 11, 13, 17]),
]
for ax, (name, chord) in zip(axes[0], gallery_chords):
    field = chladni_field_pairwise(
        chord, antisymmetric=True, symmetry="d4_max",
        resolution=_auto_resolution_for_modes(chord),
        pair_subset="all",
    )
    plotting.draw_chladni_sand(field, ax, n_particles=160_000,
                                point_size=0.5, point_alpha=0.5)
    n = field.parameters.get("n_pairs")
    ax.set_title(f"{name}\\npair_subset='all' ({n} pairs)",
                  color="white", fontsize=10)
for ax, (name, chord) in zip(axes[1], gallery_chords):
    field = chladni_field_pairwise(
        chord, antisymmetric=True, symmetry="d4_max",
        resolution=_auto_resolution_for_modes(chord),
        # default pair_subset='auto'
    )
    plotting.draw_chladni_sand(field, ax, n_particles=160_000,
                                point_size=0.5, point_alpha=0.5)
    actual = field.parameters.get("pair_subset")
    n = field.parameters.get("n_pairs")
    ax.set_title(f"{name}\\npair_subset='auto' → '{actual}' ({n} pairs)",
                  color="white", fontsize=10)
fig.suptitle("pair_subset — top: literal 'all', dotty for 4-ratio chords.   bottom: 'auto', bold curves everywhere.",
             color="white", y=0.995, fontsize=13)
fig.tight_layout(pad=0.6);
"""),
    ("md", """## 7. Painted rendering — robust for higher wavenumbers

The classical sand-particle rendering reads as "individual grains" when
the chord's peak wavenumber climbs above ~15 — each nodal patch then
shrinks below the visible particle-cluster scale. ``plotting.draw_chladni_painted``
is an imshow-based alternative that:

- auto-scales σ inversely with the chord's peak wavenumber (so stripe
  width tracks the local wavelength — high-WN chords automatically get
  thinner stripes),
- auto-scales grid resolution proportional to the peak wavenumber,
- renders with a perceptual luminance ramp (``afmhot`` by default) +
  gamma midtone brightening + a 1-pixel anti-alias blur.

Result: bold flowing curves at *every* wavenumber range, without having
to tune σ manually. Two styles:

- ``style='nodal'`` (default) — Gaussian-of-zero-crossing density.
- ``style='envelope'`` — heavy smoothing of ``|w|²`` to wash out the
  finest wavenumber detail and show the macroscopic energy distribution.
  Useful when the chord's mathematics genuinely produces a dense lattice
  (e.g. when wavenumbers can't be capped without losing the chord's
  identity).
"""),
    ("code", """from biotuner.harmonic_geometry.media.eigenmode.rigid_plate import (
    _auto_resolution_for_modes,
)

# A wider range of chords — small to mid wavenumbers — all rendered with
# the painted nodal style, auto-σ, auto-resolution.
GAMUT = [
    ("Major\\n[4, 5, 6]",        [4, 5, 6]),
    ("Sus4\\n[6, 8, 9]",         [6, 8, 9]),
    ("Dom7\\n[4, 5, 6, 7]",      [4, 5, 6, 7]),
    ("Dim7\\n[5, 6, 7, 9]",      [5, 6, 7, 9]),
    ("Tritone\\n[5, 7]",          [5, 7]),
    ("Just 11\\n[8, 11]",         [8, 11]),
    ("Quintal\\n[3, 7, 11]",      [3, 7, 11]),
    ("11-limit\\n[9, 11, 13, 17]", [9, 11, 13, 17]),
]

fig, axes = plt.subplots(2, 4, figsize=(18, 9.5), facecolor="black")
for ax, (name, chord) in zip(axes.flat, GAMUT):
    field = chladni_field_pairwise(
        chord, antisymmetric=True, symmetry="d4_max",
        resolution=_auto_resolution_for_modes(chord),
    )
    plotting.draw_chladni_painted(field, ax, style="nodal", gamma=0.85)
    ax.set_title(name, color="white", fontsize=11)
fig.suptitle(
    "Painted Chladni — auto-σ, auto-resolution, perceptual colormap (afmhot)",
    color="white", fontsize=13, y=0.995,
)
fig.tight_layout(pad=0.6);
"""),
    ("md", """### Grayscale variant — classic Chladni look without particles

The same painted renderer with ``cmap="gray"`` (and a slightly stronger
gamma boost) reproduces the canonical white-on-black Chladni aesthetic
without sampling individual particles. Smooth, anti-aliased, scales to
any chord."""),
    ("code", """fig, axes = plt.subplots(2, 4, figsize=(18, 9.5), facecolor="black")
for ax, (name, chord) in zip(axes.flat, GAMUT):
    field = chladni_field_pairwise(
        chord, antisymmetric=True, symmetry="d4_max",
        resolution=_auto_resolution_for_modes(chord),
    )
    plotting.draw_chladni_painted(field, ax, cmap="gray", gamma=0.55)
    ax.set_title(name, color="white", fontsize=11)
fig.suptitle("Painted Chladni — grayscale (classic black-and-white aesthetic)",
             color="white", fontsize=13, y=0.995)
fig.tight_layout(pad=0.6);
"""),
    ("md", """### True sand-grain particles — `draw_chladni_sand`

For the canonical *photographic* Chladni look — actual white grains
density-sampled and scattered on a black plate — use
``plotting.draw_chladni_sand``. It accepts a signed amplitude field
(auto-converts to a density with auto-σ from int_modes), inverse-CDF
samples ``n_particles`` positions, and renders them as a black-faced
scatter. Pairs cleanly with ``Granular(output_mode="particles",
nodal_emphasis=True)`` for pipeline composition, or stand-alone here
for one-shot rendering.
"""),
    ("code", """fig, axes = plt.subplots(2, 4, figsize=(18, 9.5), facecolor="black")
for ax, (name, chord) in zip(axes.flat, GAMUT):
    field = chladni_field_pairwise(
        chord, antisymmetric=True, symmetry="d4_max",
        resolution=_auto_resolution_for_modes(chord),
    )
    plotting.draw_chladni_sand(field, ax, n_particles=180_000,
                                point_size=0.45, point_alpha=0.5)
    ax.set_title(name, color="white", fontsize=11)
fig.suptitle("Sand-grain particles — white on black, the classic Chladni look",
             color="white", fontsize=13, y=0.995)
fig.tight_layout(pad=0.6);
"""),
    ("md", """### `envelope` mode — last-resort smoothing for blown-up chords

When the chord's `Fraction` form has high LCM (e.g. 12-TET Dim7 →
``[35, 42, 49, 60]``) **and** you don't want to cap the wavenumbers,
``style='envelope'`` shows the smoothed ``|w|²`` energy distribution
instead of trying to draw 60+ nodal lines per side. The chord-fingerprint
survives as a coarser pattern."""),
    ("code", """dim7_blown = [35, 42, 49, 60]
field_blown = chladni_field_pairwise(
    dim7_blown, antisymmetric=True, symmetry="d4_max",
    resolution=_auto_resolution_for_modes(dim7_blown),
)
field_capped = chladni_field_pairwise(
    dim7_blown, antisymmetric=True, symmetry="d4_max",
    resolution=_auto_resolution_for_modes(dim7_blown), max_mode=12,
)

fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), facecolor="black")
plotting.draw_chladni_painted(field_blown,  axes[0], style="nodal",    gamma=0.85)
axes[0].set_title("no cap, style='nodal'\\n(fine lattice)", color="white", fontsize=11)
plotting.draw_chladni_painted(field_blown,  axes[1], style="envelope", gamma=0.85)
axes[1].set_title("no cap, style='envelope'\\n(macroscopic envelope)", color="white", fontsize=11)
plotting.draw_chladni_painted(field_capped, axes[2], style="nodal",    gamma=0.85)
axes[2].set_title("max_mode=12, style='nodal'\\n(scaled-down + bold)", color="white", fontsize=11)
fig.suptitle("Dim7 [35, 42, 49, 60] — three ways to make it visible",
             color="white", fontsize=13, y=1.01)
fig.tight_layout();
"""),
    ("md", """## 8. Triple-antisymmetric — chords of ≥3 ratios

The triple-antisymmetric scheme sums the cyclic chain of antisymmetric
pair-modes over every distinct triple in the chord:

$$M(a,b) + M(b,c) + M(c,a)\\quad\\text{for each}\;(a,b,c)\\subset\\text{chord}$$

where $M(p,q) = \\cos(p\\pi X)\\cos(q\\pi Y) - \\cos(q\\pi X)\\cos(p\\pi Y)$.
This emphasises **triadic** interactions — a triadic chord contributes
one rich three-fold-flavoured mode rather than three independent
pair modes.
"""),
    ("code", """sigma = 0.05
geoms = [
    chladni_nodal_density(
        chladni_field_triple_antisymmetric(
            CHORDS_INT[name], symmetry="d4_max", resolution=400,
        ),
        sigma=sigma,
    )
    for name in CHORDS_INT
]
plotting.gallery(geoms, n_cols=4, titles=list(CHORDS_INT.keys()),
                 suptitle="Triple-antisymmetric + D4 max + nodal density",
                 fig_width=16.0);
"""),
    ("md", """## 9. Animation — chord-sequence morph

``plotting.animate_chord_sequence`` drives any chord→geometry builder
through a cosine-eased loop. Each keyframe is an integer-ratio list;
mid-segment frames receive interpolated floats (which the cymatics
builders accept — they don't have to be exact eigenmodes mid-morph).

The cell below renders a ~8-second MP4 at 24 fps × 48 frames-per-segment
through ``[Major, Sus4, "alt-Major", "alt-Sus4"]`` and back. The MP4
file is written to ``renders/`` next to the notebook — open it locally
to play.
"""),
    ("code", """# All keyframes must have the same length for component-wise interpolation.
CHORD_KEYFRAMES = [
    CHORDS_INT["Major"],            # [4, 5, 6]
    CHORDS_INT["Sus4"],             # [6, 8, 9]
    [2, 3, 5],                       # demo's first chord
    [3, 4, 7],                       # demo's third chord
]
print("keyframes:", CHORD_KEYFRAMES)

def cymatics_builder(chord):
    field = chladni_field_pairwise(
        list(chord), antisymmetric=True, symmetry="d4_max", resolution=320,
    )
    return Granular(
        output_mode="particles", nodal_emphasis=True, sigma=0.06,
        n_particles=80_000, seed=0,
    ).respond(field)

os.makedirs("renders", exist_ok=True)
anim = plotting.animate_chord_sequence(
    CHORD_KEYFRAMES, cymatics_builder,
    frames_per_segment=48, fps=24, loop=True,
    figsize=(6.5, 6.5),
    point_size=0.5, point_alpha=0.45,
    save_path="renders/chladni_morph_demo.mp4",
    dpi=110,
)
print("rendered:", os.path.getsize("renders/chladni_morph_demo.mp4"), "bytes")
"""),
    ("md", """## Recipe quick reference

| Want | Configuration |
|---|---|
| classical Chladni (default) | `RigidPlate()` |
| iconic square-plate Chladni lattice | `chladni_field_pairwise(chord_int, antisymmetric=True, symmetry="d4_max")` |
| sand-on-the-nodes density | wrap with `chladni_nodal_density(..., sigma=0.05)` |
| sand-grain photographic look | pipe field into `Granular(output_mode="particles", nodal_emphasis=True, sigma=0.06, n_particles=180_000)` |
| triadic mode flavour | `chladni_field_triple_antisymmetric(chord_int, symmetry="d4_max")` (≥3 ratios) |
| smooth D4 averaging | `symmetry="d4_sum"` instead of `"d4_max"` |
| cap high-LCM chords | pass `max_mode=12` (or any cap) to the pairwise / triple builders |
| bold curves for 4+ ratio chords | `pair_subset="auto"` (default) — uses `'all'` for n ≤ 3, `'root'` for n ≥ 4 |
| force literal mathematical sum | `pair_subset="all"` (dotty for 4+ ratio chords by design) |
| chord-morph MP4 | `plotting.animate_chord_sequence([chord1, chord2, ...], builder, save_path=...)` |
| painted aesthetic (afmhot), auto-σ | `plotting.draw_chladni_painted(field, ax, cmap="afmhot")` |
| classic grayscale aesthetic | `plotting.draw_chladni_painted(field, ax, cmap="gray", gamma=0.55)` |
| photographic sand-grain look | `plotting.draw_chladni_sand(field, ax, n_particles=180_000)` |
| painted aesthetic for blown-up chords | `plotting.draw_chladni_painted(field, ax, style="envelope")` |
"""),
]


NOTEBOOKS = [("01_chladni_cymatics_tour", NB)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-execute", action="store_true",
                        help="Skip in-process execution; emit bare notebooks.")
    args = parser.parse_args()

    worktree = HERE.parents[2]
    if str(worktree) not in sys.path:
        sys.path.insert(0, str(worktree))

    for name, cells in NOTEBOOKS:
        path = write_notebook(name, cells, execute=not args.no_execute)
        suffix = "with outputs" if not args.no_execute else "no execution"
        n_code = sum(1 for k, _ in cells if k == "code")
        print(f"wrote {path.relative_to(HERE.parents[1])}  "
              f"({len(cells)} cells, {n_code} code, {suffix})")


if __name__ == "__main__":
    main()
