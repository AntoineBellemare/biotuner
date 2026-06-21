"""
Prototype stills for four candidate reel aesthetics, rendered straight from
the real biotuner.harmonic_geometry generators (no reimplementation):

  1. Frozen Chords      — media.morphogenetic.crystallization.Crystallization
  2. Forbidden Symmetry — wave_field.interference quasicrystal + vortex
  3. Chord on a Sphere  — eigenmode.closed_surface.spherical_harmonic_mesh
  4. Sand & Current     — transport.granular + transport.streaming

Each idea → one PNG contact sheet (a row of chords/variants) under out/proto/.
Run from docs/reports/animation/:  python prototype_aesthetics.py
"""
from __future__ import annotations

import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))  # repo root → import biotuner

from biotuner.harmonic_input import HarmonicInput  # noqa: E402

OUT = HERE / "out" / "proto"
OUT.mkdir(parents=True, exist_ok=True)
BG = "#05070d"


def chord(ratios, amps=None):
    amps = amps or [1.0] + [0.75] * (len(ratios) - 1)
    return HarmonicInput(ratios=[F(r).limit_denominator(64) for r in ratios],
                         amplitudes=amps, base_freq=196.0)


# Custom muted/elegant colormaps
ICE = LinearSegmentedColormap.from_list(
    "ice", ["#05080f", "#0e2433", "#1d5e6b", "#7fb6c4", "#eaf6fb"])
EMBERMUTE = LinearSegmentedColormap.from_list(
    "embermute", ["#0a0810", "#3a2436", "#8a4a5a", "#d08a5a", "#f0dcc0"])
DUSKHUE = LinearSegmentedColormap.from_list(
    "duskhue", ["#080a12", "#283a52", "#5a7a8a", "#b48a9a", "#e8dcc8"])


def _panel(ax, title=None, color="#aeb9cc"):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_facecolor(BG)
    if title:
        ax.set_title(title, color=color, fontsize=13, pad=8,
                     fontfamily="DejaVu Sans")


def _sheet(name, big_title, n):
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 5.2), facecolor=BG)
    fig.suptitle(big_title, color="#e6ecf5", fontsize=22, y=0.98,
                 fontfamily="DejaVu Sans", fontweight="bold")
    if n == 1:
        axes = [axes]
    return fig, axes, name


def _save(fig, name):
    p = OUT / f"{name}.png"
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(p, dpi=120, facecolor=BG)
    plt.close(fig)
    print(f"  wrote {p.relative_to(HERE)}")


# ── 1. FROZEN CHORDS — crystallization ──────────────────────────────────
def frozen_chords():
    from biotuner.harmonic_geometry.media.morphogenetic.crystallization import (
        Crystallization,
    )
    specs = [
        ("Major  1:5/4:3/2", [1, F(5, 4), F(3, 2)]),
        ("Dom 7  +7/4 (7-limit)", [1, F(5, 4), F(3, 2), F(7, 4)]),
        ("Dissonant 11/8:13/8", [1, F(11, 8), F(13, 8), F(7, 4)]),
    ]
    fig, axes, name = _sheet("frozen_chords", "❄  FROZEN CHORDS — crystallization", 3)
    for ax, (lab, rr) in zip(axes, specs):
        cr = Crystallization(grid_radius=95, n_steps=2200, output_resolution=340,
                             output_mode="water", seed_strategy="polygon",
                             target_fill=0.22, rng_seed=7)
        g = cr(chord(rr))
        field = np.asarray(g.coordinates, dtype=float)
        cmap = ICE.copy(); cmap.set_bad(BG)
        ax.imshow(np.ma.masked_invalid(field), cmap=cmap, origin="lower",
                  interpolation="bilinear")
        _panel(ax, lab)
    _save(fig, name)


# ── 2. FORBIDDEN SYMMETRY — quasicrystal + vortex ───────────────────────
def forbidden_symmetry():
    from biotuner.harmonic_geometry.media.wave_field.interference import (
        quasicrystal_field_2d, vortex_field_2d,
    )
    c = chord([1, F(5, 4), F(3, 2), F(7, 4)])
    fig, axes, name = _sheet("forbidden_symmetry",
                             "✦  FORBIDDEN SYMMETRY — quasicrystals & vortices", 3)
    q5 = quasicrystal_field_2d(c, n_fold=5, resolution=460, output="amplitude_pow")
    q12 = quasicrystal_field_2d(c, n_fold=12, resolution=460, output="amplitude_pow")
    vx = vortex_field_2d(c, radial_kind="bessel", resolution=460,
                         output="amplitude_pow")
    for ax, (lab, g, cmap) in zip(axes, [
        ("5-fold quasicrystal", q5, "twilight"),
        ("12-fold quasicrystal", q12, EMBERMUTE),
        ("vortex spirals (bessel)", vx, "twilight_shifted"),
    ]):
        ax.imshow(np.asarray(g.coordinates, float), cmap=cmap, origin="lower",
                  interpolation="bilinear")
        _panel(ax, lab)
    _save(fig, name)


# ── 3. A CHORD ON A SPHERE — spherical harmonics (3D) ───────────────────
def chord_on_sphere():
    from biotuner.harmonic_geometry.media.eigenmode.closed_surface import (
        spherical_harmonic_mesh,
    )
    specs = [
        ("Major · zonal", [1, F(5, 4), F(3, 2)], "zonal", 20),
        ("Dom7 · sectoral", [1, F(5, 4), F(3, 2), F(7, 4)], "sectoral", 35),
        ("11-limit · balanced", [1, F(7, 4), F(11, 8), F(13, 8)], "chord_balanced", 45),
    ]
    fig = plt.figure(figsize=(4.6 * 3, 5.2), facecolor=BG)
    fig.suptitle("◉  A CHORD ON A SPHERE — spherical harmonics",
                 color="#e6ecf5", fontsize=22, y=0.98, fontweight="bold")
    for i, (lab, rr, rule, azim) in enumerate(specs, 1):
        g = spherical_harmonic_mesh(chord(rr), epsilon=0.34, mode_rule=rule,
                                    max_l=10, n_theta=120, n_phi=240)
        V = np.asarray(g.coordinates, float)
        Faces = np.asarray(g.faces, int)
        w = np.asarray(g.weights, float)
        fw = w[Faces].mean(axis=1)
        ax = fig.add_subplot(1, 3, i, projection="3d", facecolor=BG)
        tri = ax.plot_trisurf(V[:, 0], V[:, 1], V[:, 2], triangles=Faces,
                              cmap=DUSKHUE, linewidth=0, antialiased=True)
        tri.set_array(fw)
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()
        ax.view_init(elev=22, azim=azim)
        m = 1.25
        ax.set_xlim(-m, m); ax.set_ylim(-m, m); ax.set_zlim(-m, m)
        ax.set_title(lab, color="#aeb9cc", fontsize=13, pad=0)
    _save(fig, "chord_on_sphere")


# ── 4. SAND & CURRENT — granular + streaming ────────────────────────────
def sand_and_current():
    from biotuner.harmonic_geometry.media.transport.granular import Granular
    from biotuner.harmonic_geometry.media.transport.streaming import Streaming
    c = chord([1, F(5, 4), F(3, 2), F(7, 4)])

    fig, axes, name = _sheet("sand_and_current",
                             "≋  SAND & CURRENT — granular & streaming", 3)
    # sand to nodes
    g_node = Granular(output_mode="particles", n_particles=9000,
                      affinity=1.0, nodal_emphasis=True, temperature=0.06)(c)
    pts = np.asarray(g_node.coordinates, float)
    axes[0].scatter(pts[:, 0], pts[:, 1], s=0.7, c="#eadfc6", alpha=0.8,
                    linewidths=0)
    axes[0].set_aspect("equal"); _panel(axes[0], "sand → nodes (Chladni)")

    # powder to antinodes
    g_anti = Granular(output_mode="particles", n_particles=9000,
                      affinity=-1.0, temperature=0.12)(c)
    pa = np.asarray(g_anti.coordinates, float)
    axes[1].scatter(pa[:, 0], pa[:, 1], s=0.7, c="#bcd6d0", alpha=0.75,
                    linewidths=0)
    axes[1].set_aspect("equal"); _panel(axes[1], "powder → antinodes")

    # streaming flow
    s = Streaming(output_mode="flow")(c)
    vec = np.asarray(s.coordinates, float)
    X, Y = s.field_grid
    U, Vv = vec[..., 0], vec[..., 1]
    speed = np.hypot(U, Vv)
    axes[2].streamplot(X, Y, U, Vv, density=1.5, color=speed,
                       cmap="twilight", linewidth=0.8, arrowsize=0.6)
    axes[2].set_aspect("equal"); _panel(axes[2], "streaming (Rayleigh rolls)")
    _save(fig, name)


if __name__ == "__main__":
    jobs = [("frozen", frozen_chords), ("forbidden", forbidden_symmetry),
            ("sphere", chord_on_sphere), ("transport", sand_and_current)]
    for tag, fn in jobs:
        try:
            print(f"[{tag}]")
            fn()
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()
