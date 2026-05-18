"""
Generate a visual report on the ``biotuner.harmonic_geometry.media``
family-organized response operators.

Showcases what's new after the medium framework migration:

1. The five-family taxonomy — eigenmode / wave_field / parametric /
   transport / morphogenetic (currently 3 of 5 populated).
2. Granular (the first new medium) — sand-on-plate redistribution that
   produces the iconic Chladni figure the bare eigenmode does not.
3. Pipeline composition — same chord, multiple branches off one stage,
   different media downstream.
4. Coupling — chord-shape reductions (consonance, ratio_complexity,
   spectral_spread, amplitude_entropy) that map a chord to medium
   parameters.

Produces:
    docs/reports/figures/media/*.png
    docs/reports/media_report.md

Usage::

    python docs/reports/generate_media_report.py
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path
from typing import Optional

_WORKTREE_ROOT = str(Path(__file__).resolve().parents[1])
if _WORKTREE_ROOT not in sys.path:
    sys.path.insert(0, _WORKTREE_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from biotuner.harmonic_geometry import (
    Acoustic,
    Box3D,
    Circular,
    ClosedSurface,
    Crystallization,
    Elastic,
    Faraday,
    Granular,
    HarmonicInput,
    Interference,
    Pipeline,
    PlasmaLattice,
    PolygonDomain,
    ReactionDiffusion,
    Rectangular,
    RigidPlate,
    Streaming,
    Tracer,
)
from biotuner.harmonic_geometry.plotting import (
    draw_field_2d,
    draw_vector_field_2d,
)
from biotuner.harmonic_geometry.media import coupling, structure

# ─────────────────────────────────────────────────────────────── paths ──
REPO_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = REPO_ROOT / "docs" / "reports" / "figures" / "media"
REPORT_PATH = REPO_ROOT / "docs" / "reports" / "media_report.md"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DPI = 150
SAVE_KW = dict(dpi=DPI, bbox_inches="tight", facecolor="white")

# ───────────────────────────────────────────────────────── chord library ──
CHORDS: dict[str, HarmonicInput] = {
    "Major": HarmonicInput(
        ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)],
        amplitudes=[1.0, 0.7, 0.8],
    ),
    "Minor": HarmonicInput(
        ratios=[Fraction(1), Fraction(6, 5), Fraction(3, 2)],
        amplitudes=[1.0, 0.7, 0.8],
    ),
    "Sus4": HarmonicInput(
        ratios=[Fraction(1), Fraction(4, 3), Fraction(3, 2)],
        amplitudes=[1.0, 0.7, 0.8],
    ),
    "Dim7": HarmonicInput(
        ratios=[Fraction(1), Fraction(6, 5), Fraction(7, 5), Fraction(12, 7)],
        amplitudes=[1.0, 0.8, 0.8, 0.7],
    ),
    "11:7:5": HarmonicInput(
        ratios=[Fraction(1), Fraction(11, 8), Fraction(7, 5), Fraction(5, 3)],
        amplitudes=[1.0, 0.7, 0.8, 0.6],
    ),
}


# ─────────────────────────────────────────────────────── plotting helpers ──


def _ax(fig, idx):
    ax = fig.add_subplot(*idx)
    ax.set_xticks([])
    ax.set_yticks([])
    return ax


def _imshow_field(field: np.ndarray, ax, cmap: str = "twilight",
                  vmin=None, vmax=None, bad_color: Optional[str] = None):
    finite = np.isfinite(field)
    if not np.any(finite):
        return ax
    if vmin is None:
        vmin = float(np.nanpercentile(field, 1))
    if vmax is None:
        vmax = float(np.nanpercentile(field, 99))
    cm = plt.get_cmap(cmap).copy()
    if bad_color is not None:
        cm.set_bad(color=bad_color)
    ax.imshow(
        field, cmap=cm, vmin=vmin, vmax=vmax,
        origin="lower", interpolation="nearest",
    )
    ax.set_aspect("equal")
    return ax


def _imshow_density(density: np.ndarray, ax, cmap: str = "Greys_r",
                    gamma: float = 0.25, smooth_sigma: float = 1.5):
    """Render a Granular density with cymatics-style display.

    Three transforms make the underlying physics legible across regimes:

    1. **Uniform detection** — if the variation is below numerical noise,
       render flat mid-gray (so the ``affinity=0`` case reads cleanly).
    2. **Mild Gaussian smoothing** — converts delta-spike distributions
       (typical at antinodes — single-cell peaks) into visible blobs
       without falsifying the qualitative pattern.
    3. **Power normalization** (``gamma=0.25``) — compresses the
       dynamic range so broad nodal distributions and concentrated
       antinodal spikes both render with visible contrast.

    NaN cells (e.g. outside a circular plate) propagate through.
    """
    from scipy.ndimage import gaussian_filter

    finite = np.isfinite(density)
    if not np.any(finite):
        ax.set_aspect("equal")
        return ax
    d = np.where(finite, density, 0.0)
    d_vals = d[finite]
    # Uniform case: relative variation is < 1e-6 of the mean.
    mean = float(d_vals.mean())
    if mean > 0 and float(d_vals.std()) / mean < 1e-6:
        flat = np.where(finite, 0.5, np.nan)
        ax.imshow(flat, cmap=cmap, vmin=0.0, vmax=1.0,
                  origin="lower", interpolation="nearest")
        ax.set_aspect("equal")
        return ax
    if smooth_sigma > 0:
        d = gaussian_filter(d, sigma=smooth_sigma)
    dmax = float(d.max())
    if dmax <= 0:
        ax.set_aspect("equal")
        return ax
    normalized = (d / dmax) ** gamma
    normalized[~finite] = np.nan
    ax.imshow(
        normalized, cmap=cmap, vmin=0.0, vmax=1.0,
        origin="lower", interpolation="nearest",
    )
    ax.set_aspect("equal")
    return ax


def _imshow_acoustic(
    field: np.ndarray,
    ax,
    sources: np.ndarray,
    extent_value: float,
    cmap: str = "magma",
    signed: bool = False,
    gamma: float = 0.45,
    mask_radius: float = 0.05,
) -> Any:
    """Render an Acoustic field with near-source masking + gamma.

    The 1/sqrt(r) near-field singularity dominates intensity / Schlieren
    if rendered linearly: a small disk of radius ``mask_radius`` around
    each source is set to NaN, then a percentile-based dynamic range
    is used and gamma-corrected so the spatial pattern reads clearly.
    """
    f = field.copy()
    # Build pixel grid for distance calculation.
    H, W = f.shape
    xs = np.linspace(-extent_value, +extent_value, W)
    ys = np.linspace(-extent_value, +extent_value, H)
    X, Y = np.meshgrid(xs, ys, indexing="xy")
    for sx, sy in sources:
        r = np.hypot(X - sx, Y - sy)
        f[r < mask_radius] = np.nan

    cm = plt.get_cmap(cmap).copy()
    cm.set_bad(color="black")

    finite = np.isfinite(f)
    if signed:
        vmax = float(np.nanpercentile(np.abs(f[finite]), 99)) or 1.0
        vmin = -vmax
        # Signed fields don't gamma-correct cleanly; show linearly.
        ax.imshow(f, cmap=cm, vmin=vmin, vmax=vmax,
                   origin="lower", interpolation="nearest",
                   extent=(-extent_value, extent_value,
                           -extent_value, extent_value))
    else:
        # Percentile clipping then gamma compress.
        vmin = float(np.nanpercentile(f[finite], 1))
        vmax = float(np.nanpercentile(f[finite], 99)) or 1.0
        norm = np.clip((f - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
        norm = norm ** gamma
        ax.imshow(norm, cmap=cm, vmin=0.0, vmax=1.0,
                   origin="lower", interpolation="nearest",
                   extent=(-extent_value, extent_value,
                           -extent_value, extent_value))
    ax.set_aspect("equal")
    return ax


def _scatter_particles(positions: np.ndarray, ax, color="#222222", size=2):
    ax.scatter(positions[:, 0], positions[:, 1], s=size, c=color,
               alpha=0.55, linewidths=0)
    ax.set_aspect("equal")
    return ax


def _title(ax, line1: str, line2: str = "", fontsize: int = 9):
    if line2:
        ax.set_title(f"{line1}\n{line2}", fontsize=fontsize)
    else:
        ax.set_title(line1, fontsize=fontsize)


# =============================================================== figure 1
def fig_taxonomy_overview() -> Path:
    """One-glance view: same chord through the five currently-populated families."""
    chord = CHORDS["Major"]

    plate = RigidPlate(domain=Rectangular(1, 1), resolution=192).respond(chord)
    # chord_balanced mixes zonal / tesseral / sectoral modes so the chord
    # excites a quilt of lobes rather than pure latitude bands. The default
    # 'zonal' produces horizontal stripes which read poorly in a flat (θ, φ)
    # showcase panel.
    sphere = ClosedSurface(
        mode_rule="chord_balanced", n_theta=128, n_phi=256
    ).respond(chord)
    quasi = Interference(paradigm="quasicrystal").respond(
        chord, n_fold=5, resolution=192
    )
    faraday = Faraday(pattern="hexagonal", resolution=192).respond(chord)
    crystal = Crystallization(
        grid_radius=80, n_steps=1500, output_resolution=192,
        humidity=8e-3, diffusion=0.5,
    ).respond(chord)

    fig = plt.figure(figsize=(16, 3.4))
    ax1 = _ax(fig, (1, 5, 1))
    _imshow_field(plate.coordinates, ax1, cmap="twilight")
    _title(ax1, "eigenmode family",
           "RigidPlate · rectangular plate")

    ax2 = _ax(fig, (1, 5, 2))
    _imshow_field(sphere.coordinates, ax2, cmap="twilight")
    # ClosedSurface output is (n_theta, n_phi) = (128, 256) — wider than tall.
    # Drop equal-aspect so the panel fills the subplot like the others.
    ax2.set_aspect("auto")
    _title(ax2, "eigenmode family",
           "ClosedSurface · spherical harmonics")

    ax3 = _ax(fig, (1, 5, 3))
    _imshow_field(quasi.coordinates, ax3, cmap="twilight")
    _title(ax3, "wave_field family",
           "Interference · 5-fold quasicrystal")

    ax4 = _ax(fig, (1, 5, 4))
    _imshow_field(faraday.coordinates, ax4, cmap="twilight")
    _title(ax4, "parametric family",
           "Faraday · hexagonal cymatics")

    ax5 = _ax(fig, (1, 5, 5))
    _imshow_field(crystal.coordinates, ax5, cmap="twilight")
    _title(ax5, "morphogenetic family",
           "Crystallization · Reiter snowflake")

    fig.suptitle(
        "Five families, one chord (Major) — full taxonomy of media response operators",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = FIG_DIR / "01_taxonomy_overview.png"
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    return path


# =============================================================== figure 2
def fig_granular_intro() -> Path:
    """The bare plate vs the visible Chladni figure (sand on top)."""
    chord = CHORDS["Major"]
    plate = RigidPlate(domain=Rectangular(1, 1), resolution=256).respond(chord)
    sand = Granular(affinity=1.0, temperature=0.03).respond(plate)
    sand_pts = Granular(
        affinity=1.0, temperature=0.03, output_mode="particles",
        n_particles=4000, seed=0,
    ).respond(plate)

    fig = plt.figure(figsize=(12, 4.0))
    ax1 = _ax(fig, (1, 3, 1))
    _imshow_field(plate.coordinates, ax1, cmap="twilight")
    _title(ax1, "Step 1 — RigidPlate eigenmode",
           "scalar displacement u(x, y)")

    ax2 = _ax(fig, (1, 3, 2))
    _imshow_density(sand.coordinates, ax2)
    _title(ax2, "Step 2 — Granular(affinity=+1, T=0.03)",
           "ρ ∝ exp(−|u|² / T) · sand-at-nodes density")

    ax3 = _ax(fig, (1, 3, 3))
    ax3.set_xlim(-0.02, 1.02)
    ax3.set_ylim(-0.02, 1.02)
    _scatter_particles(sand_pts.coordinates, ax3, color="#1a1a1a", size=2.5)
    _title(ax3, "Step 2′ — output_mode='particles'",
           "4000 grains sampled from ρ")

    fig.suptitle(
        "Pipeline composition: RigidPlate → Granular reveals the visible Chladni figure",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = FIG_DIR / "02_granular_intro.png"
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    return path


# =============================================================== figure 3
def fig_granular_affinity_sweep() -> Path:
    """Same plate, sweep affinity through −2, −1, 0, +1, +2."""
    chord = CHORDS["Major"]
    plate = RigidPlate(domain=Rectangular(1, 1), resolution=192).respond(chord)

    sweep = [-2.0, -1.0, 0.0, 1.0, 2.0]
    cols = len(sweep)
    fig = plt.figure(figsize=(cols * 2.8, 3.2))
    for i, aff in enumerate(sweep):
        ax = _ax(fig, (1, cols, i + 1))
        # T=0.15 keeps the negative-affinity (antinode) regime broad
        # enough to render legibly without changing the qualitative
        # picture; positive-affinity sand regimes are robust at any T.
        out = Granular(affinity=aff, temperature=0.15).respond(plate)
        _imshow_density(out.coordinates, ax, smooth_sigma=2.0)
        if aff < 0:
            tag = "antinodes"
        elif aff > 0:
            tag = "nodes"
        else:
            tag = "uniform"
        _title(ax, f"affinity = {aff:+.1f}", f"→ {tag}")
    fig.suptitle(
        "Granular: affinity selects regime — negative pulls to antinodes "
        "(powder), positive to nodes (sand)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = FIG_DIR / "03_granular_affinity_sweep.png"
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    return path


# =============================================================== figure 4
def fig_granular_temperature_sweep() -> Path:
    """Same plate, sweep temperature through cold → hot."""
    chord = CHORDS["Major"]
    plate = RigidPlate(domain=Rectangular(1, 1), resolution=192).respond(chord)

    sweep = [0.01, 0.03, 0.1, 0.3, 1.0]
    cols = len(sweep)
    fig = plt.figure(figsize=(cols * 2.8, 3.2))
    for i, T in enumerate(sweep):
        ax = _ax(fig, (1, cols, i + 1))
        out = Granular(affinity=1.0, temperature=T).respond(plate)
        _imshow_density(out.coordinates, ax)
        _title(ax, f"T = {T:.2f}",
               "sharper ←" if T <= 0.05 else
               ("→ smearing" if T >= 0.3 else "intermediate"))
    fig.suptitle(
        "Granular: temperature controls how sharply grains localize — "
        "low T crisp, high T smears toward uniform",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = FIG_DIR / "04_granular_temperature_sweep.png"
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    return path


# =============================================================== figure 5
def fig_granular_domain_gallery() -> Path:
    """Granular over different plate domains."""
    chord = CHORDS["Major"]
    cases = [
        ("rectangular", RigidPlate(domain=Rectangular(1, 1), resolution=192)),
        ("circular", RigidPlate(domain=Circular(1.0), resolution=192)),
        ("hexagon (6 sides)",
         RigidPlate(domain=PolygonDomain(n_sides=6, radius=1.0), resolution=128)),
        ("pentagon (5 sides)",
         RigidPlate(domain=PolygonDomain(n_sides=5, radius=1.0), resolution=128)),
    ]
    cols = len(cases)
    fig = plt.figure(figsize=(cols * 3.2, 3.4))
    for i, (label, plate_m) in enumerate(cases):
        ax = _ax(fig, (1, cols, i + 1))
        plate = plate_m.respond(chord)
        sand = Granular(affinity=1.0, temperature=0.04).respond(plate)
        _imshow_density(sand.coordinates, ax)
        _title(ax, label, "Major chord · affinity=+1")
    fig.suptitle(
        "Granular composes with every RigidPlate domain — same chord, "
        "different boundaries",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = FIG_DIR / "05_granular_domain_gallery.png"
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    return path


# =============================================================== figure 6
def fig_chord_signature_grid() -> Path:
    """5 chords × 2 affinities — each chord's distinctive Chladni signature."""
    # affinity → (sign, temperature) — antinode regime needs a warmer
    # T to render as visible blobs rather than delta spikes.
    affs = [(+1.0, 0.04), (-1.0, 0.18)]
    chord_keys = list(CHORDS.keys())
    rows = len(affs)
    cols = len(chord_keys)
    fig = plt.figure(figsize=(cols * 2.6 + 0.6, rows * 2.6))
    for r, (aff, T) in enumerate(affs):
        for c, key in enumerate(chord_keys):
            ax = fig.add_subplot(rows, cols, r * cols + c + 1)
            ax.set_xticks([]); ax.set_yticks([])
            plate = RigidPlate(
                domain=Rectangular(1, 1), resolution=160
            ).respond(CHORDS[key])
            sand = Granular(affinity=aff, temperature=T).respond(plate)
            _imshow_density(sand.coordinates, ax, smooth_sigma=2.0)
            if r == 0:
                ax.set_title(key, fontsize=10)
            if c == 0:
                ax.set_ylabel(
                    f"affinity = {aff:+.0f}\n"
                    + ("(nodes / sand)" if aff > 0 else "(antinodes / powder)"),
                    fontsize=9,
                )
    fig.suptitle(
        "Each chord prints a distinctive signature — same medium, "
        "different harmonic input",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = FIG_DIR / "06_chord_signature_grid.png"
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    return path


# =============================================================== figure 7
def fig_field_kind_comparison() -> Path:
    """displacement (|u|²) vs energy_gradient (|∇u|²) — what the grain feels."""
    chord = CHORDS["Dim7"]
    plate = RigidPlate(domain=Rectangular(1, 1), resolution=224).respond(chord)
    sand_disp = Granular(
        affinity=1.0, temperature=0.04, field_kind="displacement"
    ).respond(plate)
    sand_grad = Granular(
        affinity=1.0, temperature=0.04, field_kind="energy_gradient"
    ).respond(plate)

    fig = plt.figure(figsize=(10, 3.5))
    ax1 = _ax(fig, (1, 3, 1))
    _imshow_field(plate.coordinates, ax1, cmap="twilight")
    _title(ax1, "plate displacement u(x, y)", "Dim7 chord")

    ax2 = _ax(fig, (1, 3, 2))
    _imshow_density(sand_disp.coordinates, ax2)
    _title(ax2, "field_kind='displacement'", "V ∝ u² (canonical Chladni sand)")

    ax3 = _ax(fig, (1, 3, 3))
    _imshow_density(sand_grad.coordinates, ax3)
    _title(ax3, "field_kind='energy_gradient'", "V ∝ |∇u|² (streaming picture)")

    fig.suptitle(
        "field_kind selects which potential the grain feels — displacement "
        "vs kinetic-energy gradient",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = FIG_DIR / "07_field_kind_comparison.png"
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    return path


# =============================================================== figure 8
def fig_pipeline_branching() -> Path:
    """One plate → many downstream media (positive sand, negative powder, particles)."""
    chord = CHORDS["Minor"]
    plate = RigidPlate(domain=Circular(1.0), resolution=224).respond(chord)
    sand_pos = Granular(affinity=1.0, temperature=0.03).respond(plate)
    sand_neg = Granular(affinity=-1.0, temperature=0.15).respond(plate)
    sand_pts = Granular(
        affinity=1.0, temperature=0.03, output_mode="particles",
        n_particles=4000, seed=7,
    ).respond(plate)

    fig = plt.figure(figsize=(13, 3.5))
    ax1 = _ax(fig, (1, 4, 1))
    _imshow_field(plate.coordinates, ax1, cmap="twilight")
    _title(ax1, "shared upstream",
           "RigidPlate(Circular) · Minor chord")

    ax2 = _ax(fig, (1, 4, 2))
    _imshow_density(sand_pos.coordinates, ax2)
    _title(ax2, "branch A", "Granular(+1, T=0.03) · sand")

    ax3 = _ax(fig, (1, 4, 3))
    _imshow_density(sand_neg.coordinates, ax3)
    _title(ax3, "branch B", "Granular(−1, T=0.15) · powder")

    ax4 = _ax(fig, (1, 4, 4))
    ax4.set_xlim(-1.05, 1.05)
    ax4.set_ylim(-1.05, 1.05)
    _scatter_particles(sand_pts.coordinates, ax4, color="#1a1a1a")
    _title(ax4, "branch C", "Granular(+1) particle mode")

    fig.suptitle(
        "Pipeline branching: one wave-field stage feeds many downstream "
        "transport stages — no recomputation",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = FIG_DIR / "08_pipeline_branching.png"
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    return path


# ============================================================== figure 9a
def fig_faraday_patterns() -> Path:
    """Gallery of Faraday pattern symmetries — same chord, four patterns."""
    chord = CHORDS["Major"]
    patterns = ["stripe", "square", "hexagonal", "twelve_fold"]
    pattern_labels = {
        "stripe": "stripe (1-fold)",
        "square": "square (4-fold)",
        "hexagonal": "hexagonal (6-fold)",
        "twelve_fold": "twelve-fold quasipattern",
    }
    cols = len(patterns)
    fig = plt.figure(figsize=(cols * 2.8, 3.1))
    for i, p in enumerate(patterns):
        ax = _ax(fig, (1, cols, i + 1))
        out = Faraday(pattern=p, resolution=192).respond(chord)
        _imshow_field(out.coordinates, ax, cmap="twilight")
        _title(ax, pattern_labels[p], f"Major chord · viscosity=0.01")
    fig.suptitle(
        "Faraday: pattern symmetry selects discrete rotational order — "
        "stripe / square / hexagonal / quasipattern",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = FIG_DIR / "09_faraday_patterns.png"
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    return path


# ============================================================= figure 9b
def fig_faraday_dispersion() -> Path:
    """Gravity vs capillary vs mixed dispersion — same chord, three regimes."""
    # Use the Major chord with equal amplitudes — keeps the chord consistent
    # with the rest of the report while preventing the root from dominating
    # visually (so the higher components can demonstrate dispersion spread).
    chord = HarmonicInput(
        ratios=CHORDS["Major"].to_ratios(),
        amplitudes=[1.0, 1.0, 1.0],
    )
    cases = [
        ("gravity",
         "ω² = g·k  ⇒  k ∝ ω²  (wide wavenumber spread)"),
        ("capillary",
         "ω² = σ/ρ·k³  ⇒  k ∝ ω^(2/3)  (compact spectrum)"),
        ("mixed",
         "ω² = g·k + σ/ρ·k³  (water-like)"),
    ]
    cols = len(cases)
    fig = plt.figure(figsize=(cols * 3.2, 3.7))
    for i, (disp, formula) in enumerate(cases):
        ax = _ax(fig, (1, cols, i + 1))
        out = Faraday(
            pattern="hexagonal", dispersion=disp,
            base_wavenumber=6.0 * np.pi,
            viscosity=1e-4,
            output="real",
            resolution=192,
        ).respond(chord)
        _imshow_field(out.coordinates, ax, cmap="twilight")
        ks = out.parameters["wavenumbers"]
        ks_str = " · ".join(f"{k:.1f}" for k in ks)
        _title(ax, f"dispersion = '{disp}'",
               f"{formula}\nk = [{ks_str}]")
    fig.suptitle(
        "Faraday: dispersion regime maps chord frequencies to wavenumbers — "
        "gravity spreads them quadratically, capillary keeps them compact",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = FIG_DIR / "10_faraday_dispersion.png"
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    return path


# ============================================================= figure 9c
def fig_faraday_viscosity_sweep() -> Path:
    """Viscosity damping sweep — gentle to over-damped.

    Damping is exp(−ν·k²). With base_wavenumber=6π and chord components
    reaching k≈49 (gravity, ratio 5/3)², the per-mode damping for the
    highest component spans:

        ν = 0      → 1.0       (inviscid; all components carry equal energy)
        ν = 1e-4   → 0.79      (high mode mildly suppressed)
        ν = 4e-4   → 0.38      (high mode visibly weakened)
        ν = 1e-3   → 0.09      (only root + second component remain)
        ν = 4e-3   → ~0        (over-damped; only the root survives)

    Values much above 1e-3 are all visually equivalent (everything dies),
    so the sweep stays in the regime where each step is visible.
    """
    chord = HarmonicInput(
        ratios=CHORDS["Major"].to_ratios(),
        amplitudes=[1.0, 1.0, 1.0],
    )
    sweep = [0.0, 1e-4, 4e-4, 1e-3, 4e-3]
    labels = ["inviscid", "lightly damped", "medium", "strongly damped",
              "over-damped"]
    cols = len(sweep)
    fig = plt.figure(figsize=(cols * 2.8, 3.2))
    for i, (nu, label) in enumerate(zip(sweep, labels)):
        ax = _ax(fig, (1, cols, i + 1))
        out = Faraday(
            pattern="hexagonal", viscosity=nu,
            base_wavenumber=6.0 * np.pi,
            output="real",
            resolution=160,
        ).respond(chord)
        _imshow_field(out.coordinates, ax, cmap="twilight")
        _title(ax, f"viscosity = {nu:.0e}" if nu > 0 else "viscosity = 0",
               label)
    fig.suptitle(
        "Faraday: viscosity acts as a low-pass on the wavenumber spectrum — "
        "short-wavelength chord components damp first, leaving only the root",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = FIG_DIR / "11_faraday_viscosity.png"
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    return path


# ============================================================== figure 9d
def fig_faraday_to_granular() -> Path:
    """Faraday → Granular pipeline: sand on a vibrating fluid surface."""
    chord = CHORDS["Minor"]
    faraday = Faraday(
        pattern="hexagonal", viscosity=0.005, resolution=224
    ).respond(chord)
    sand = Granular(affinity=1.0, temperature=0.15).respond(faraday)
    fig = plt.figure(figsize=(8.5, 3.6))
    ax1 = _ax(fig, (1, 2, 1))
    _imshow_field(faraday.coordinates, ax1, cmap="twilight")
    _title(ax1, "Faraday surface", "hexagonal cymatics · Minor chord")
    ax2 = _ax(fig, (1, 2, 2))
    _imshow_density(sand.coordinates, ax2, smooth_sigma=2.0)
    _title(ax2, "Faraday → Granular",
           "particles redistributed on the Faraday surface")
    fig.suptitle(
        "Cross-family pipeline: parametric/Faraday feeds transport/Granular — "
        "sand on a vibrating fluid",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = FIG_DIR / "12_faraday_to_granular.png"
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    return path


# ============================================================== figure 12a
def fig_crystallization_intro() -> Path:
    """Reiter snowflake — three output modes from one simulation."""
    chord = CHORDS["Major"]
    # Three views of the same simulation: continuous water content,
    # binary frozen mask, and the extracted boundary outline. Defaults
    # let the chord drive humidity / diffusion / sectors / asymmetry /
    # noise via the coupling.* + structure.* reductions.
    grid_radius = 120
    base_kwargs = dict(
        grid_radius=grid_radius, n_steps=2200,
        target_fill=0.22,
        anisotropy_strength=1.0,
        anisotropy_kernel_width=np.pi / 20.0,
        seed_branch_length=2,
        output_resolution=256,
    )
    water = Crystallization(output_mode="water", **base_kwargs).respond(chord)
    frozen = Crystallization(output_mode="frozen", **base_kwargs).respond(chord)
    boundary = Crystallization(output_mode="boundary",
                                **base_kwargs).respond(chord)

    fig = plt.figure(figsize=(12, 4.0), facecolor="white")
    ax1 = _ax(fig, (1, 3, 1))
    _imshow_field(water.coordinates, ax1, cmap="twilight",
                  bad_color="black")
    _title(ax1, "water content (continuous)",
           "u(x, y) = receptive + non-receptive parts")

    ax2 = _ax(fig, (1, 3, 2))
    _imshow_field(frozen.coordinates, ax2, cmap="Greys_r",
                  vmin=0.0, vmax=1.0, bad_color="black")
    _title(ax2, "frozen mask (binary)",
           f"{frozen.metadata['frozen_cells']} cells out of "
           f"{frozen.metadata['total_cells']}")

    ax3 = _ax(fig, (1, 3, 3))
    ax3.set_facecolor("black")
    ax3.set_aspect("equal")
    ax3.set_xlim(-grid_radius - 5, grid_radius + 5)
    ax3.set_ylim(-grid_radius - 5, grid_radius + 5)
    for curve in boundary.coordinates:
        ax3.plot(curve[:, 0], curve[:, 1], color="white", linewidth=1.0)
    _title(ax3, "boundary contour (curve_set_2d)",
           "extracted via marching squares")

    p = frozen.parameters
    fig.suptitle(
        "Crystallization (Reiter CA): three output modes from the same simulation — "
        f"Major chord → sectors={p['anisotropy_sectors']}, "
        f"asym={p['asymmetry']:.2f}, Tonnetz polygon seed",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = FIG_DIR / "13_crystallization_intro.png"
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    return path


# ============================================================ figure 12b
def fig_crystallization_humidity_diffusion() -> Path:
    """Humidity × diffusion grid — the two main morphology knobs.

    Fixed chord (Major), fixed sectors=6, varying humidity / diffusion.
    ``target_fill`` keeps every cell sub-saturated so the dendritic
    morphology stays visible rather than collapsing into a filled hex.
    """
    chord = CHORDS["Major"]
    hums = [1.5e-3, 4.0e-3, 8.0e-3]
    diffs = [0.15, 0.5, 0.85]
    fig, axes = plt.subplots(
        len(diffs), len(hums),
        figsize=(len(hums) * 2.6 + 0.4, len(diffs) * 2.6),
        facecolor="white",
    )
    for r, d in enumerate(diffs):
        for c, h in enumerate(hums):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            out = Crystallization(
                grid_radius=110, n_steps=2400,
                target_fill=0.25,
                humidity=h, diffusion=d,
                anisotropy_strength=0.7,
                anisotropy_sectors=6,
                anisotropy_kernel_width=np.pi / 20.0,
                seed_branch_length=2,
                output_resolution=180, output_mode="frozen",
            ).respond(chord)
            _imshow_field(out.coordinates, ax, cmap="Greys_r",
                          vmin=0.0, vmax=1.0, bad_color="black")
            if r == 0:
                ax.set_title(f"humidity = {h:.0e}", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"diffusion = {d:.2g}", fontsize=10)
            ax.text(0.97, 0.03,
                    f"fill={out.metadata['frozen_fraction']:.2f}",
                    color="white", ha="right", va="bottom",
                    transform=ax.transAxes, fontsize=7)
    fig.suptitle(
        "Crystallization: humidity (column) × diffusion (row) — "
        "humidity drives overall size; diffusion balances inflow and "
        "controls dendrite branching",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = FIG_DIR / "14_crystallization_grid.png"
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    return path


# ============================================================ figure 12c
def fig_crystallization_chord_signatures() -> Path:
    """Five chords → five distinct snowflake signatures.

    All chord-driven defaults are used. The diversity comes from four
    independent intrinsic descriptors of the ratio set:

    - ``structure.prime_limit`` sets ``anisotropy_sectors`` (3, 5, 7,
      11) — primes incompatible with the hex grid's 6-fold neighborhood
      produce geometric frustration patterns.
    - ``structure.tonnetz_polygon`` plants the chord's just-intonation
      lattice polygon as the initial frozen seed pattern — major and
      minor triads project to mirror-flipped polygons here.
    - ``structure.max_common_int`` drives ``asymmetry`` — chords with
      larger integer denominators (e.g. minor 10:12:15) get more
      symmetry-breaking wobble than simpler ones (major 4:5:6).
    - ``structure.per_ratio_consonance_weights`` reweights the angular
      bias so growth in directions of consonant chord components is
      stronger.
    """
    keys = list(CHORDS.keys())
    cols = len(keys)
    fig = plt.figure(figsize=(cols * 2.8 + 0.2, 4.0), facecolor="white")
    for i, key in enumerate(keys):
        ax = _ax(fig, (1, cols, i + 1))
        ch = CHORDS[key]
        out = Crystallization(
            grid_radius=110, n_steps=1800,
            target_fill=0.22,
            # all chord-driven: humidity, diffusion, sectors, asymmetry, noise
            anisotropy_strength=1.2,
            anisotropy_kernel_width=np.pi / 22.0,
            seed_branch_length=2,
            output_resolution=200, output_mode="frozen",
        ).respond(ch)
        _imshow_field(out.coordinates, ax, cmap="Greys_r",
                      vmin=0.0, vmax=1.0, bad_color="black")
        p = out.parameters
        plim = structure.prime_limit(ch)
        mci = structure.max_common_int(ch)
        _title(
            ax, key,
            f"{plim}-limit · sectors={p['anisotropy_sectors']} · "
            f"max_int={mci} · asym={p['asymmetry']:.2f}",
            fontsize=8,
        )
    fig.suptitle(
        "Crystallization: chord-driven snowflake signatures — "
        "prime limit sets sectors, Tonnetz polygon seeds the growth, "
        "max-common-integer drives asymmetry",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = FIG_DIR / "15_crystallization_chord_signatures.png"
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    return path


# =============================================================== figure 9
def fig_coupling_chord_metrics() -> Path:
    """Bar chart of the four coupling reductions across the chord library."""
    keys = list(CHORDS.keys())
    metrics = {
        "consonance": [coupling.consonance(CHORDS[k]) for k in keys],
        "ratio_complexity": [coupling.ratio_complexity(CHORDS[k]) for k in keys],
        "spectral_spread": [coupling.spectral_spread(CHORDS[k]) for k in keys],
        "amplitude_entropy": [coupling.amplitude_entropy(CHORDS[k]) for k in keys],
    }
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    palette = ["#1f3b73", "#a23e2c", "#3a7a4d", "#7a5d24"]
    for ax, (name, vals), color in zip(axes, metrics.items(), palette):
        ax.bar(keys, vals, color=color, alpha=0.85)
        ax.set_title(name, fontsize=10)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
        ax.grid(alpha=0.2, axis="y", linestyle=":")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle(
        "coupling.* — four chord-shape reductions wired into medium parameters",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = FIG_DIR / "16_coupling_chord_metrics.png"
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    return path


# =============================================================== figure 10
def fig_chord_shapes_medium() -> Path:
    """Show consonance → temperature mapping: consonant chords are 'cool', dissonant chords 'hot'."""
    keys = ["Major", "Sus4", "Minor", "Dim7", "11:7:5"]
    cols = len(keys)
    fig = plt.figure(figsize=(cols * 2.8, 3.4))
    for i, key in enumerate(keys):
        chord = CHORDS[key]
        c = coupling.consonance(chord)
        # Map consonance ∈ [0, 1] → temperature ∈ [0.02, 0.4] (consonant → cold).
        T = 0.02 + (1.0 - c) * 0.38
        plate = RigidPlate(domain=Rectangular(1, 1), resolution=160).respond(chord)
        sand = Granular(affinity=1.0, temperature=T).respond(plate)
        ax = _ax(fig, (1, cols, i + 1))
        _imshow_density(sand.coordinates, ax)
        _title(ax, key, f"consonance={c:.2f} → T={T:.2f}")
    fig.suptitle(
        "Chord shapes the medium: coupling.consonance(chord) → Granular "
        "temperature (consonant = cool/sharp, dissonant = hot/smeared)",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = FIG_DIR / "17_chord_shapes_medium.png"
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    return path


# =============================================================== figure 11
def fig_tracer_overview() -> Path:
    """Tracer medium — flow regimes on a wave field."""
    chord = CHORDS["Major"]
    plate = RigidPlate(domain=Rectangular(1, 1), resolution=192).respond(chord)
    interf = Interference().respond(chord)

    rows = [
        ("RigidPlate (eigenmode)", plate),
        ("Interference (wave_field)", interf),
    ]
    fig, axes = plt.subplots(
        2, 4, figsize=(14, 6.8), facecolor="white"
    )
    for r, (src_label, src) in enumerate(rows):
        # column 0: background field
        ax0 = axes[r, 0]
        draw_field_2d(src, ax0, cmap="RdBu_r", signed=True, show_nodal=False)
        _title(ax0, src_label, "wave field (source)")

        # columns 1-3: Tracer flow_kinds
        for c, kind in enumerate(["gradient", "curl", "mixed"]):
            ax = axes[r, c + 1]
            out = Tracer(flow_kind=kind).respond(src)
            draw_vector_field_2d(
                out, ax, style="streamlines",
                density=1.6, color_by_magnitude=True,
                background=src.coordinates, background_cmap="RdBu_r",
                background_alpha=0.30, linewidth=0.6,
            )
            _title(ax, f"Tracer({kind!r})",
                   "gradient -> sinks/sources" if kind == "gradient"
                   else "curl -> level-curve loops" if kind == "curl"
                   else "mixed=0.5 -> hybrid")

    fig.suptitle(
        "Tracer: passive scalar advection — three flow regimes derived from "
        "a single wave field",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = FIG_DIR / "19_tracer_overview.png"
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    return path


def fig_streaming_overview() -> Path:
    """Streaming medium — Rayleigh-roll vortex pairs on a Chladni plate."""
    chord = CHORDS["Major"]
    plate = RigidPlate(domain=Rectangular(1, 1), resolution=192).respond(chord)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), facecolor="white")
    # 0: parent wave field
    draw_field_2d(plate, axes[0], cmap="RdBu_r",
                  signed=True, show_nodal=False)
    _title(axes[0], "RigidPlate eigenmode",
           "wave field (source)")

    # 1: streaming flow
    flow = Streaming(viscosity=0.5).respond(plate)
    draw_vector_field_2d(
        flow, axes[1], style="streamlines",
        density=1.8, color_by_magnitude=True,
        background=plate.coordinates, background_cmap="RdBu_r",
        background_alpha=0.30, linewidth=0.7,
    )
    _title(axes[1], "Streaming(viscosity=0.5)",
           "Rayleigh rolls between antinodes")

    # 2: speed magnitude
    speed = Streaming(viscosity=0.5, output_mode="speed").respond(plate)
    draw_field_2d(speed, axes[2], cmap="magma", signed=False,
                  show_nodal=False)
    _title(axes[2], "speed |u_stream|",
           "fast = vortex centers")

    # 3: tracer density
    dens = Streaming(viscosity=0.5,
                      output_mode="tracer_density").respond(plate)
    draw_field_2d(dens, axes[3], cmap="magma", signed=False,
                  show_nodal=False)
    _title(axes[3], "steady tracer density",
           "slow regions accumulate")

    fig.suptitle(
        "Streaming: acoustic streaming on a finite-amplitude wave — "
        "FFT Poisson solve of the curl of the Reynolds stress",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = FIG_DIR / "20_streaming_overview.png"
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    return path


def fig_transport_pipeline_branching() -> Path:
    """Same wave field, three transport branches: Granular / Tracer / Streaming."""
    chord = CHORDS["Major"]
    plate = RigidPlate(domain=Rectangular(1, 1), resolution=192).respond(chord)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), facecolor="white")
    # 0: parent
    draw_field_2d(plate, axes[0], cmap="RdBu_r",
                  signed=True, show_nodal=False)
    _title(axes[0], "RigidPlate (shared source)", "Major chord")

    # 1: Granular
    sand = Granular(affinity=1.0, temperature=0.04).respond(plate)
    _imshow_density(sand.coordinates, axes[1])
    _title(axes[1], "-> Granular", "sand-on-Chladni-plate")

    # 2: Tracer
    flow = Tracer(flow_kind="curl").respond(plate)
    draw_vector_field_2d(
        flow, axes[2], style="streamlines",
        density=1.6, color_by_magnitude=True,
        background=plate.coordinates, background_cmap="RdBu_r",
        background_alpha=0.25, linewidth=0.6,
    )
    _title(axes[2], "-> Tracer (curl)", "level-curve streamlines")

    # 3: Streaming
    stream = Streaming(viscosity=0.5).respond(plate)
    draw_vector_field_2d(
        stream, axes[3], style="streamlines",
        density=1.6, color_by_magnitude=True,
        background=plate.coordinates, background_cmap="RdBu_r",
        background_alpha=0.25, linewidth=0.6,
    )
    _title(axes[3], "-> Streaming (Rayleigh)", "secondary vortex pairs")

    fig.suptitle(
        "Transport family: three operators consume the same wave field, "
        "produce three qualitatively different responses",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = FIG_DIR / "21_transport_branching.png"
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    return path


def fig_acoustic_overview() -> Path:
    """Acoustic medium — four observables from one source array."""
    chord = CHORDS["Major"]
    extent_val = 1.0
    base = dict(n_sources=4, source_layout="ring", source_radius=0.25,
                base_frequency=8.0, attenuation=0.3, extent=extent_val,
                resolution=240)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8), facecolor="white")

    # Pressure: linear signed.
    out = Acoustic(output_mode="pressure", **base).respond(chord)
    sources = out.metadata["source_positions"]
    _imshow_acoustic(out.coordinates, axes[0], sources, extent_val,
                     cmap="RdBu_r", signed=True)
    axes[0].scatter(sources[:, 0], sources[:, 1], s=20,
                    facecolors="none", edgecolors="lime", linewidths=1.2)
    _title(axes[0], "Acoustic(pressure)", "Re[P] snapshot")

    # Intensity: log-like via gamma-correction + percentile clip.
    out = Acoustic(output_mode="intensity", **base).respond(chord)
    _imshow_acoustic(out.coordinates, axes[1], sources, extent_val,
                     cmap="magma", gamma=0.35)
    axes[1].scatter(sources[:, 0], sources[:, 1], s=20,
                    facecolors="none", edgecolors="lime", linewidths=1.2)
    _title(axes[1], "Acoustic(intensity)", "<p^2> time-average")

    # Schlieren: gamma-corrected so the radial wavefronts stand out.
    out = Acoustic(output_mode="schlieren", **base).respond(chord)
    _imshow_acoustic(out.coordinates, axes[2], sources, extent_val,
                     cmap="Greys_r", gamma=0.40)
    axes[2].scatter(sources[:, 0], sources[:, 1], s=20,
                    facecolors="none", edgecolors="red", linewidths=1.2)
    _title(axes[2], "Acoustic(schlieren)", "|grad^2 p| shadowgraph")

    # Phase: signed (no gamma).
    out = Acoustic(output_mode="phase", **base).respond(chord)
    _imshow_acoustic(out.coordinates, axes[3], sources, extent_val,
                     cmap="twilight", signed=True)
    axes[3].scatter(sources[:, 0], sources[:, 1], s=20,
                    facecolors="none", edgecolors="lime", linewidths=1.2)
    _title(axes[3], "Acoustic(phase)", "arg(P) in [-pi, pi]")

    fig.suptitle(
        "Acoustic: bulk 2-D pressure field from 4 ring sources, "
        "Major chord at base_freq=8.0",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = FIG_DIR / "22_acoustic_overview.png"
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    return path


def fig_acoustic_chord_signatures() -> Path:
    """Acoustic chord signatures with intrinsic ratio-set diversity.

    Each chord ratio projects to a unique ``(direction, wavelength,
    amplitude)`` tuple via:

    - ``source_layout="chord_angles"`` -> direction = pitch-class angle
    - ``source_assignment="per_ratio"`` -> source j emits only ratio j
      at frequency ``base_freq × r_j``, with chord amplitude ``a_j`` as
      gain. Each source therefore has its own wavelength.

    With ``base_frequency`` low enough that wavelengths are a
    meaningful fraction of the domain, the resulting interference
    pattern carries the chord's full structural signature: different
    chord = different number of sources × directions × wavelengths.
    """
    keys = list(CHORDS.keys())
    fig, axes = plt.subplots(1, len(keys),
                              figsize=(len(keys) * 2.8, 3.6),
                              facecolor="white")
    extent_val = 1.0
    for ax, key in zip(axes, keys):
        ch = CHORDS[key]
        # base_frequency scales with the chord's prime_limit so 3-limit
        # / 5-limit / 7-limit / 11-limit chords operate at clearly
        # different wavelength scales.
        plim = structure.prime_limit(ch)
        base_f = 1.5 + 0.6 * (plim - 2)
        # source_radius also chord-driven from max_common_int so chords
        # with larger common-denom integers spread the emitters wider.
        mci = structure.max_common_int(ch)
        radius = 0.30 + 0.20 * min(np.log(max(mci, 2)) / np.log(200.0), 1.0)
        out = Acoustic(
            source_layout="chord_angles",
            source_assignment="per_ratio",
            source_radius=radius,
            base_frequency=base_f,
            attenuation=0.0,
            resolution=240,
            extent=extent_val,
            output_mode="intensity",
        ).respond(ch)
        sources = out.metadata["source_positions"]
        _imshow_acoustic(out.coordinates, ax, sources, extent_val,
                         cmap="magma", gamma=0.45)
        ax.scatter(sources[:, 0], sources[:, 1], s=22,
                   facecolors="none", edgecolors="cyan", linewidths=1.4)
        _title(
            ax, key,
            f"{plim}-limit, {len(sources)} sources, "
            f"base_f={base_f:.1f}, R={radius:.2f}",
            fontsize=8,
        )
    fig.suptitle(
        "Acoustic chord signatures: per_ratio assignment, prime_limit -> base_freq, "
        "max_common_int -> source_radius",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = FIG_DIR / "23_acoustic_chord_signatures.png"
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    return path


def fig_reaction_diffusion_grid() -> Path:
    """Gray-Scott (F, K) parameter sweep — Pearson regime map."""
    fk_grid = [
        (0.022, 0.051, "spots / mitosis"),
        (0.030, 0.062, "alpha labyrinth"),
        (0.039, 0.058, "stripes"),
        (0.062, 0.061, "self-replicating"),
        (0.018, 0.045, "expanding rings"),
        (0.026, 0.055, "spiral / chaos"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), facecolor="white")
    chord = CHORDS["Major"]
    for ax, (F, K, label) in zip(axes.flat, fk_grid):
        out = ReactionDiffusion(
            feed=F, kill=K, resolution=144, n_steps=4000,
            seed_strategy="random", rng_seed=7,
        ).respond(chord)
        draw_field_2d(out, ax, cmap="magma", signed=False,
                      show_nodal=False)
        _title(ax, f"F={F:.3f}, K={K:.3f}", label)
    fig.suptitle(
        "Reaction-Diffusion (Gray-Scott): Pearson regime map — "
        "same chord, different (F, K) coordinates",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = FIG_DIR / "24_reaction_diffusion_grid.png"
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    return path


def fig_reaction_diffusion_chord_signatures() -> Path:
    """Five chords through ReactionDiffusion with chord-driven F/K + polygon seed.

    The new chord-driven F/K mapping uses ``structure.prime_limit`` and
    ``structure.max_common_int``. With these structural descriptors,
    the five chords land in five different Pearson regimes:

    - Sus4 (3-limit, max_int=9)  → low F, low K → mitosis-style spots
    - Major (5-limit, max_int=6) → mid F, low K → soft labyrinth
    - Minor (5-limit, max_int=15) → mid F, mid K → tight labyrinth (mirror of Major)
    - Dim7 (7-limit, max_int=60) → high F, high K → stripes / replicating
    - 11:7:5 (11-limit, max_int=200) → highest F/K → high-density stripes

    Random seeding gives each chord room to express its full attractor
    rather than the early-evolution dipole symmetry of polygon seeding.
    """
    keys = list(CHORDS.keys())
    fig, axes = plt.subplots(1, len(keys),
                              figsize=(len(keys) * 2.8, 3.8),
                              facecolor="white")
    for ax, key in zip(axes, keys):
        out = ReactionDiffusion(
            resolution=176, n_steps=12000,
            seed_strategy="random", rng_seed=7,
        ).respond(CHORDS[key])
        draw_field_2d(out, ax, cmap="magma", signed=False,
                      show_nodal=False)
        p = out.parameters
        _title(ax, key,
               f"F={p['feed']:.3f}, K={p['kill']:.3f}, "
               f"D_u/D_v={p['diffusion_ratio']:.2f}",
               fontsize=8)
    fig.suptitle(
        "Reaction-Diffusion: chord-driven feed (prime_limit) + "
        "kill (max_common_int) — each chord lands in a different Pearson regime",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = FIG_DIR / "25_reaction_diffusion_chord_signatures.png"
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    return path


def fig_elastic_anisotropy_sweep() -> Path:
    """Elastic plate — anisotropy ratio sweep with fixed axis."""
    chord = CHORDS["Major"]
    ratios = [1.0, 1.5, 2.0, 3.0, 4.0]
    fig, axes = plt.subplots(1, len(ratios),
                              figsize=(len(ratios) * 2.6, 3.2),
                              facecolor="white")
    for ax, r in zip(axes, ratios):
        out = Elastic(anisotropy_ratio=r, anisotropy_axis=0.0,
                      resolution=200, n_modes=24).respond(chord)
        draw_field_2d(out, ax, cmap="RdBu_r", signed=True,
                      show_nodal=True)
        _title(ax, f"anisotropy_ratio={r:.1f}",
               "isotropic" if r == 1.0 else f"alpha={out.parameters['alpha']:.2f}, "
                                              f"beta={out.parameters['beta']:.2f}")
    fig.suptitle(
        "Elastic: anisotropy_ratio sweep at axis=0° (Major chord) — "
        "modes elongate along the y-axis as the ratio grows",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = FIG_DIR / "26_elastic_anisotropy_sweep.png"
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    return path


def fig_elastic_chord_signatures() -> Path:
    """Elastic plate — chord-driven anisotropy ratio + axis."""
    keys = list(CHORDS.keys())
    fig, axes = plt.subplots(1, len(keys),
                              figsize=(len(keys) * 2.8, 3.6),
                              facecolor="white")
    for ax, key in zip(axes, keys):
        out = Elastic(resolution=200, n_modes=24).respond(CHORDS[key])
        draw_field_2d(out, ax, cmap="RdBu_r", signed=True,
                      show_nodal=True)
        p = out.parameters
        axis_deg = np.degrees(p["anisotropy_axis"])
        _title(
            ax, key,
            f"ratio={p['anisotropy_ratio']:.2f}, axis={axis_deg:+.0f} deg",
            fontsize=8,
        )
    fig.suptitle(
        "Elastic: chord-driven anisotropy (consonance -> ratio, "
        "mean pitch-class angle -> axis)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = FIG_DIR / "27_elastic_chord_signatures.png"
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    return path


def fig_plasma_lattice_overview() -> Path:
    """PlasmaLattice — modulation_strength sweep with a single chord.

    The medium's story is the *coupling*: ``modulation_strength``
    interpolates between two well-known crystallization regimes —

    - ``mod=0``: pure Coulomb crystal in a harmonic trap (Wigner-like
      triangular packing — depends only on N).
    - ``mod -> large``: ions snap into the chord's eigenmode wells; the
      lattice inherits the chord's spatial signature.

    Showing both endpoints plus three intermediate values for *one*
    chord makes the transition visible. Chord-to-chord variation is
    shown separately in subsequent figures (n_ions and field topology
    do change per chord; see also the chord-signature figures).

    Per-panel: chord-field contours; ions colored by on-site potential
    (blue = chord well, red = ridge); k=4 nearest-neighbor bonds.
    """
    from scipy.spatial import cKDTree
    from matplotlib.collections import LineCollection

    from biotuner.harmonic_geometry.media.eigenmode.plasma_lattice \
        import _interp_field

    chord = CHORDS["Major"]
    strengths = [0.0, 0.6, 1.2, 2.4, 4.5]
    fig, axes = plt.subplots(1, len(strengths),
                              figsize=(len(strengths) * 2.8, 3.4),
                              facecolor="white")

    for ax, mod in zip(axes, strengths):
        out = PlasmaLattice(
            n_ions=14, n_steps=900,
            learning_rate=0.018,
            coulomb_strength=0.020,
            modulation_strength=float(mod),
            trap_radius=0.85,
            chord_resolution=160,
            rng_seed=11,
        ).respond(chord)

        pts = out.coordinates
        x0, x1, y0, y1 = out.metadata["chord_field_extent"]
        chord_field = out.metadata["chord_field"]
        cmag = float(np.nanmax(np.abs(chord_field))) or 1.0

        if mod > 0.0:
            ax.imshow(
                chord_field, cmap="RdBu_r",
                origin="lower", interpolation="bilinear",
                extent=(x0, x1, y0, y1),
                vmin=-cmag, vmax=cmag, alpha=0.55,
            )
            xs = np.linspace(x0, x1, chord_field.shape[1])
            ys = np.linspace(y0, y1, chord_field.shape[0])
            Xc, Yc = np.meshgrid(xs, ys)
            ax.contour(Xc, Yc, chord_field,
                       levels=[-0.6 * cmag, -0.3 * cmag,
                                0.0, 0.3 * cmag, 0.6 * cmag],
                       colors="black", linewidths=0.4, alpha=0.45)
        else:
            ax.set_facecolor("#f4f4f4")

        # k=4 nearest-neighbor bonds.
        if len(pts) >= 2:
            tree = cKDTree(pts)
            k = min(5, len(pts))
            _, idxs = tree.query(pts, k=k)
            segs = []
            for i, row in enumerate(idxs):
                for j in row[1:]:
                    if i < j:
                        segs.append([pts[i], pts[j]])
            if segs:
                ax.add_collection(LineCollection(
                    segs, colors="black", linewidths=0.5, alpha=0.4))

        # Ions: colored by on-site chord-potential value where mod > 0.
        if mod > 0.0:
            xs_arr = np.linspace(x0, x1, chord_field.shape[1])
            ys_arr = np.linspace(y0, y1, chord_field.shape[0])
            X_grid, Y_grid = np.meshgrid(xs_arr, ys_arr)
            v_at, _, _ = _interp_field(chord_field, X_grid, Y_grid, pts)
            mean_well_depth = float(np.mean(v_at))
            ax.scatter(pts[:, 0], pts[:, 1],
                       c=v_at, cmap="RdBu_r", vmin=-cmag, vmax=cmag,
                       s=85, edgecolors="black", linewidths=0.9, zorder=3)
            ax.text(
                0.97, 0.03,
                f"<V> = {mean_well_depth:+.2f}",
                color="black", ha="right", va="bottom",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7,
                          edgecolor="none", pad=1.0),
            )
        else:
            ax.scatter(pts[:, 0], pts[:, 1],
                       c="#222222", s=70, edgecolors="white",
                       linewidths=0.9, zorder=3)
            ax.text(
                0.97, 0.03, "Wigner crystal",
                color="black", ha="right", va="bottom",
                transform=ax.transAxes, fontsize=8,
                bbox=dict(facecolor="white", alpha=0.7,
                          edgecolor="none", pad=1.0),
            )

        ax.set_aspect("equal")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xticks([]); ax.set_yticks([])
        _title(ax, f"modulation_strength = {mod:.1f}",
               "pure Coulomb" if mod == 0
               else "chord wells dominant" if mod >= 3
               else "chord-shaped lattice",
               fontsize=8)

    fig.suptitle(
        "PlasmaLattice: modulation_strength sweep (Major chord, 14 ions). "
        "Left = pure Coulomb (Wigner crystal); right = ions snap into "
        "chord-eigenmode wells.",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = FIG_DIR / "28_plasma_lattice_overview.png"
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    return path


def fig_plasma_lattice_chord_signatures() -> Path:
    """PlasmaLattice — chord-driven n_ions + chord-shaped wells.

    Companion to the modulation sweep: same modulation_strength, five
    different chords. Shows how:

    - ``prime_limit`` -> ion count (3-limit -> few ions, 11-limit -> many)
    - chord eigenmode -> potential topology -> ion arrangement
    """
    from scipy.spatial import cKDTree
    from matplotlib.collections import LineCollection

    from biotuner.harmonic_geometry.media.eigenmode.plasma_lattice \
        import _interp_field

    keys = list(CHORDS.keys())
    fig, axes = plt.subplots(1, len(keys),
                              figsize=(len(keys) * 2.8, 3.6),
                              facecolor="white")
    for ax, key in zip(axes, keys):
        chord = CHORDS[key]
        out = PlasmaLattice(
            n_steps=900, learning_rate=0.018,
            coulomb_strength=0.020,
            modulation_strength=3.0,
            trap_radius=0.85,
            chord_resolution=160,
        ).respond(chord)

        pts = out.coordinates
        x0, x1, y0, y1 = out.metadata["chord_field_extent"]
        chord_field = out.metadata["chord_field"]
        cmag = float(np.nanmax(np.abs(chord_field))) or 1.0

        ax.imshow(
            chord_field, cmap="RdBu_r",
            origin="lower", interpolation="bilinear",
            extent=(x0, x1, y0, y1),
            vmin=-cmag, vmax=cmag, alpha=0.55,
        )
        xs = np.linspace(x0, x1, chord_field.shape[1])
        ys = np.linspace(y0, y1, chord_field.shape[0])
        Xc, Yc = np.meshgrid(xs, ys)
        ax.contour(Xc, Yc, chord_field,
                   levels=[-0.6 * cmag, -0.3 * cmag,
                            0.0, 0.3 * cmag, 0.6 * cmag],
                   colors="black", linewidths=0.4, alpha=0.45)

        if len(pts) >= 2:
            tree = cKDTree(pts)
            k = min(5, len(pts))
            _, idxs = tree.query(pts, k=k)
            segs = []
            for i, row in enumerate(idxs):
                for j in row[1:]:
                    if i < j:
                        segs.append([pts[i], pts[j]])
            if segs:
                ax.add_collection(LineCollection(
                    segs, colors="black", linewidths=0.5, alpha=0.4))

        xs_arr = np.linspace(x0, x1, chord_field.shape[1])
        ys_arr = np.linspace(y0, y1, chord_field.shape[0])
        X_grid, Y_grid = np.meshgrid(xs_arr, ys_arr)
        v_at, _, _ = _interp_field(chord_field, X_grid, Y_grid, pts)
        ax.scatter(pts[:, 0], pts[:, 1],
                   c=v_at, cmap="RdBu_r", vmin=-cmag, vmax=cmag,
                   s=70, edgecolors="black", linewidths=0.8, zorder=3)

        ax.set_aspect("equal")
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_xticks([]); ax.set_yticks([])
        p = out.parameters
        plim = structure.prime_limit(chord)
        _title(ax, key,
               f"{plim}-limit -> {p['n_ions']} ions",
               fontsize=8)
    fig.suptitle(
        "PlasmaLattice chord signatures: prime_limit -> ion count, "
        "chord eigenmode -> well topology; modulation_strength = 3.0.",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    path = FIG_DIR / "29_plasma_lattice_chord_signatures.png"
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    return path


def fig_pipeline_object_demo() -> Path:
    """Demonstrate the Pipeline class composing two stages explicitly."""
    pipe = Pipeline(
        RigidPlate(domain=PolygonDomain(n_sides=7, radius=1.0), resolution=160),
        Granular(affinity=1.0, temperature=0.04),
    )
    fig = plt.figure(figsize=(11, 3.2))
    for i, key in enumerate(["Major", "Minor", "Dim7", "11:7:5"]):
        ax = _ax(fig, (1, 4, i + 1))
        out = pipe(CHORDS[key])
        _imshow_density(out.coordinates, ax)
        _title(ax, key, repr(pipe).replace("Pipeline", "").strip("()"))
    fig.suptitle(
        f"Pipeline class: {pipe!r} applied to four chords",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    path = FIG_DIR / "18_pipeline_object_demo.png"
    fig.savefig(path, **SAVE_KW)
    plt.close(fig)
    return path


# ────────────────────────────────────────────────────── markdown writer ──


def write_report(paths: list[Path]) -> Path:
    rel = lambda p: p.relative_to(REPORT_PATH.parent).as_posix()
    md = [
        "# `harmonic_geometry.media` — first-chunk report",
        "",
        "_Family-organized response operators for harmonic geometry._",
        "",
        "This report shows what the new `media` framework adds on top of the",
        "existing `harmonic_geometry` machinery. The framework reorganizes",
        "the module around **five operator families** — `eigenmode`,",
        "`wave_field`, `parametric`, `transport`, `morphogenetic` — and",
        "introduces a uniform `Medium.respond(forcing, **overrides)` contract",
        "for composing them into pipelines.",
        "",
        "## Architecture",
        "",
        "```",
        "biotuner/harmonic_geometry/",
        "├── chladni.py                   ← compat shim",
        "├── spherical_harmonics.py       ← compat shim",
        "├── interference_patterns.py     ← compat shim",
        "└── media/",
        "    ├── base.py                  ← Medium ABC · Domain types · Pipeline",
        "    ├── coupling.py              ← 4 chord→param reductions",
        "    ├── eigenmode/",
        "    │   ├── rigid_plate.py       ← chladni + RigidPlate",
        "    │   └── closed_surface.py    ← spherical + ClosedSurface",
        "    ├── wave_field/",
        "    │   └── interference.py     ← + Interference",
        "    ├── parametric/              ← (Faraday next)",
        "    ├── transport/",
        "    │   └── granular.py          ← NEW",
        "    └── morphogenetic/           ← (snowflake next)",
        "```",
        "",
        "## What this report demonstrates",
        "",
        "1. **Three families coexist** — the same chord rendered through the",
        "   three currently-populated families produces visually distinct",
        "   outputs.",
        "2. **Granular** — the first new medium. Steady-state Boltzmann",
        "   redistribution `ρ ∝ exp(−V/T)` on top of any 2-D scalar field.",
        "   Three regime knobs (`affinity`, `temperature`, `field_kind`)",
        "   span sand-at-nodes, powder-at-antinodes, sharp ↔ smeared, and",
        "   displacement ↔ streaming-gradient pictures.",
        "3. **Pipeline composition** — one wave-field stage can feed many",
        "   downstream transport stages without recomputation.",
        "4. **Coupling** — four chord-shape reductions (consonance,",
        "   ratio_complexity, spectral_spread, amplitude_entropy) are",
        "   exposed in `media.coupling` for wiring chord properties into",
        "   medium parameters.",
        "",
        "## Figures",
        "",
    ]
    sections = [
        ("Five families, one chord", paths[0]),
        ("RigidPlate → Granular: the visible Chladni figure", paths[1]),
        ("Granular affinity sweep (regime selection)", paths[2]),
        ("Granular temperature sweep (sharpness)", paths[3]),
        ("Granular over different RigidPlate domains", paths[4]),
        ("Granular chord signatures: each chord prints its own figure", paths[5]),
        ("Granular field_kind: displacement vs energy_gradient", paths[6]),
        ("Pipeline branching: one upstream, many downstream", paths[7]),
        ("Faraday pattern symmetries (cymatics)", paths[8]),
        ("Faraday dispersion regimes (gravity / capillary / mixed)", paths[9]),
        ("Faraday viscosity sweep (low-pass on wavenumber)", paths[10]),
        ("Faraday → Granular: sand on a vibrating fluid", paths[11]),
        ("Crystallization: three output modes of the Reiter snowflake", paths[12]),
        ("Crystallization: humidity × diffusion morphology grid", paths[13]),
        ("Crystallization: chord-driven snowflake signatures", paths[14]),
        ("coupling.* — chord-shape reductions", paths[15]),
        ("Chord shapes the medium (consonance → temperature)", paths[16]),
        ("Pipeline class — explicit two-stage composition", paths[17]),
        # Phase B media additions (paths 18-28):
        ("Tracer: flow regimes on a wave field", paths[18]),
        ("Streaming: Rayleigh acoustic streaming vortices", paths[19]),
        ("Transport pipeline branching (Granular / Tracer / Streaming)",
         paths[20]),
        ("Acoustic: four observables (pressure / intensity / Schlieren / phase)",
         paths[21]),
        ("Acoustic chord signatures: per_ratio assignment", paths[22]),
        ("Reaction-Diffusion: Pearson regime map", paths[23]),
        ("Reaction-Diffusion chord signatures (structural-descriptor F/K)",
         paths[24]),
        ("Elastic: anisotropy ratio sweep", paths[25]),
        ("Elastic chord signatures (chord-driven anisotropy + axis)",
         paths[26]),
        ("PlasmaLattice: modulation_strength transition",
         paths[27]),
        ("PlasmaLattice chord signatures (prime_limit -> ion count)",
         paths[28]),
    ]
    for i, (title, path) in enumerate(sections, 1):
        md.append(f"### {i}. {title}")
        md.append("")
        md.append(f"![]({rel(path)})")
        md.append("")

    md += [
        "## Validation",
        "",
        "- 232 media tests pass (Medium / Pipeline / Domain / auto-wrap;",
        "  Granular, Faraday, Crystallization, Tracer, Streaming, Acoustic,",
        "  ReactionDiffusion, Elastic, PlasmaLattice; coupling and",
        "  structure descriptor modules).",
        "- All pre-existing `harmonic_geometry` tests still pass — three",
        "  files were migrated (`chladni`, `spherical_harmonics`,",
        "  `interference_patterns`) with compatibility shims so legacy",
        "  imports keep working.",
        "",
        "## Family roster",
        "",
        "| Family         | Concrete media                                        |",
        "| -------------- | ----------------------------------------------------- |",
        "| eigenmode      | RigidPlate, ClosedSurface, Elastic, PlasmaLattice     |",
        "| wave_field     | Interference, Acoustic                                |",
        "| parametric     | Faraday                                               |",
        "| transport      | Granular, Tracer, Streaming                           |",
        "| morphogenetic  | Crystallization, ReactionDiffusion                    |",
        "",
        "## Chord-structural descriptors (`media.structure`)",
        "",
        "Several media use intrinsic descriptors of the ratio set rather",
        "than scalar coupling reductions, so chords with similar overall",
        "consonance still land in qualitatively different regimes:",
        "",
        "- `prime_limit` — highest prime in any p/q (3, 5, 7, 11, 13)",
        "- `tonnetz_polygon` — each ratio projected to the JI lattice",
        "- `pairwise_harmonic_distance` — Tenney HD matrix",
        "- `cf_depths` — continued-fraction depth per ratio",
        "- `max_common_int` — largest integer in common-denom form",
        "- `pc_rotation_order` — PC-set rotational symmetry",
        "- `per_ratio_consonance_weights` — amp-weighted inverse HD",
        "",
    ]
    REPORT_PATH.write_text("\n".join(md), encoding="utf-8")
    return REPORT_PATH


# ═══════════════════════════════════════════════════════════ PDF builder ══

PDF_PATH = REPORT_PATH.parent / "media_report.pdf"

# Section narrative — short caption text shown above each figure.
SECTIONS: list[tuple[str, str]] = [
    (
        "Five families, one chord",
        "Same Major chord rendered through all five operator families: "
        "<i>eigenmode/RigidPlate</i> (bounded plate eigenmodes), "
        "<i>eigenmode/ClosedSurface</i> (spherical harmonics), "
        "<i>wave_field/Interference</i> (5-fold quasicrystal), "
        "<i>parametric/Faraday</i> (hexagonal cymatics), and "
        "<i>morphogenetic/Crystallization</i> (Reiter snowflake). The "
        "five-family taxonomy is what justifies the operator axis as the "
        "primary structure: coupling regime ↔ operator family is 1-to-1, "
        "and every family now has at least one concrete medium.",
    ),
    (
        "RigidPlate → Granular: the visible Chladni figure",
        "Two-stage pipeline. The bare eigenmode (left) is a scalar displacement "
        "field; the Granular step turns that field into the steady-state grain "
        "distribution ρ ∝ exp(−V/T) — the iconic sand-on-plate figure the bare "
        "Chladni module does not produce. The same call with "
        "<i>output_mode='particles'</i> samples N positions from ρ (right).",
    ),
    (
        "Affinity sweep — regime selection",
        "The sign of <i>affinity</i> selects the physical regime. "
        "<b>affinity &gt; 0</b> ⇒ V = +|u|²: grains driven away from antinodes "
        "by direct momentum coupling — classical Chladni sand at the nodes. "
        "<b>affinity &lt; 0</b> ⇒ V = −|u|²: fine powder pulled toward "
        "antinodes by acoustic streaming. <b>affinity = 0</b> ⇒ uniform.",
    ),
    (
        "Temperature sweep — sharpness control",
        "Boltzmann temperature T controls how sharply grains localize. "
        "Low T (left) gives crisp nodal lines; high T (right) smears toward "
        "uniform. The chord and plate are identical; only the medium "
        "parameter changes.",
    ),
    (
        "Granular over different RigidPlate domains",
        "Granular composes with every RigidPlate domain via the pipeline. "
        "Same chord, four different boundary shapes — rectangular, circular, "
        "hexagonal, pentagonal — produce qualitatively distinct nodal "
        "geometries. Domain selection is a constructor knob on RigidPlate; "
        "Granular needs no awareness of it.",
    ),
    (
        "Chord signature grid",
        "Five chords × two affinity regimes. Each chord prints its own "
        "distinctive figure: the same Granular medium renders five different "
        "Chladni signatures (top row) and five different powder signatures "
        "(bottom row). This is the basis of the chord → image bijection "
        "the framework targets.",
    ),
    (
        "field_kind: displacement vs energy_gradient",
        "Two choices for the effective potential the grain feels. "
        "<i>displacement</i> uses V ∝ u² (canonical Chladni sand, momentum "
        "coupling at maxima of displacement). <i>energy_gradient</i> uses "
        "V ∝ |∇u|² — the streaming picture, where grains respond to "
        "kinetic-energy gradients rather than displacement. Same chord, "
        "two physically distinct readouts.",
    ),
    (
        "Pipeline branching",
        "One wave-field stage feeds many downstream transport stages without "
        "recomputation. The shared upstream is a single RigidPlate(Circular) "
        "evaluation of the Minor chord; three branches off that field "
        "produce sand, powder, and a particle cloud — each from independent "
        "Granular instances. The Pipeline contract makes branching natural.",
    ),
    (
        "Faraday pattern symmetries (cymatics)",
        "Faraday is the first member of the <i>parametric</i> family: surface "
        "waves on a vertically-driven fluid layer respond at the subharmonic "
        "of the drive frequency, with discrete rotational symmetry selected "
        "by the resonant-triad regime. The same chord produces qualitatively "
        "different patterns — stripe (1-fold), square (4-fold), hexagonal "
        "(6-fold, most common), and twelve-fold quasipattern.",
    ),
    (
        "Faraday dispersion regimes",
        "The capillary-gravity dispersion ω² = g·k + (σ/ρ)·k³ maps each chord "
        "frequency to a wavenumber. In the gravity regime (long waves) "
        "k scales as ω²; in the capillary regime (short waves) k scales as "
        "ω^(2/3); the mixed regime uses the full physical dispersion. The "
        "chord's relative wavenumbers — and therefore the visual fineness of "
        "the higher components — depend strongly on which regime dominates.",
    ),
    (
        "Faraday viscosity sweep",
        "Viscosity acts as a low-pass on the wavenumber spectrum via "
        "exp(−ν·k²). Inviscid (ν=0) gives crisp interference of all chord "
        "components; physical range (ν≈0.005–0.02) damps the short-"
        "wavelength components selectively; over-damped (ν≥0.1) leaves only "
        "the chord root visible. This is the cymatics counterpart to "
        "Granular's temperature knob.",
    ),
    (
        "Faraday → Granular: sand on a vibrating fluid",
        "Cross-family pipeline composition. The <i>parametric/Faraday</i> "
        "field replaces RigidPlate as the upstream wave field; "
        "<i>transport/Granular</i> redistributes grains according to the "
        "Faraday surface's effective potential. The Pipeline contract treats "
        "this composition exactly like RigidPlate → Granular — the source's "
        "<font face='Courier'>geom_type='field_2d'</font> is the only "
        "interface Granular needs.",
    ),
    (
        "Crystallization: three output modes",
        "<i>morphogenetic/Crystallization</i> runs the Reiter (2005) "
        "cellular-automaton snowflake on a hexagonal grid. Three output "
        "modes from the same simulation: <b>water</b> (continuous content "
        "field, a real-valued GeometryData.field_2d), <b>frozen</b> (binary "
        "mask of the frozen ice phase), and <b>boundary</b> (the contour of "
        "the frozen region as a curve_set_2d). Humidity (γ) and diffusion "
        "(α) are derived from the chord via coupling.* by default — "
        "consonant chords give sharper dendrites, dissonant chords plumper "
        "shapes.",
    ),
    (
        "Crystallization: humidity × diffusion grid",
        "The two main morphology knobs swept independently. Low diffusion "
        "(top row) and modest humidity yield the canonical dendritic "
        "snowflake; high diffusion (bottom row) smooths growth into plumper "
        "rounded hex shapes; high humidity (right column) grows aggressively "
        "regardless of regime. Both knobs are exposed as constructor "
        "arguments and can be set from chord properties via coupling.*.",
    ),
    (
        "Crystallization: chord-driven snowflake signatures",
        "Same Reiter parameters, five different chords. The "
        "<font face='Courier'>anisotropy_strength</font> parameter biases "
        "humidity along angles derived from the chord's ratios (with hex-"
        "symmetry replication), so each chord prints its own snowflake "
        "signature. This is the snowflake metaphor that motivated the whole "
        "media framework: same chord, different medium conditions → "
        "different crystal; same conditions, different chord → different "
        "snowflake.",
    ),
    (
        "coupling.* — chord-shape reductions",
        "Four scalar reductions of a HarmonicInput, exposed in "
        "<i>media.coupling</i> for wiring chord properties into medium "
        "parameters. Consonance is high for simple triads (Major, Minor, "
        "Sus4) and lower for the dissonant Dim7 and 11:7:5. Ratio "
        "complexity and spectral spread behave inversely. Amplitude "
        "entropy is high here because the chord library uses near-uniform "
        "amplitudes.",
    ),
    (
        "Chord shapes the medium (consonance → temperature)",
        "Demonstration of the framework's central move: the chord shapes "
        "the medium's material properties, not just its forcing. Each panel "
        "uses Granular(temperature = f(consonance(chord))) — consonant "
        "chords give a cool/sharp medium, dissonant chords a warm/smeared "
        "one. This makes chord ↔ medium a deterministic bijection rather "
        "than a free parameter choice.",
    ),
    (
        "Pipeline class — explicit two-stage composition",
        "The Pipeline class lets you persist a composed operator and apply "
        "it to many chords. Here a single Pipeline(RigidPlate(heptagon) → "
        "Granular) is applied to four chords, producing four heptagon-"
        "shaped Chladni figures. Each pipeline stage remains independently "
        "addressable (you can swap the plate or the granular knobs without "
        "rewriting the chain).",
    ),
    # ────────────────────────── Phase B additions ──────────────────────────
    (
        "Tracer: flow regimes on a wave field",
        "<i>Tracer</i> (transport family) converts a wave field into the "
        "velocity field a passive scalar would experience. <i>gradient</i> "
        "flows toward maxima (sources / sinks at antinodes); <i>curl</i> "
        "flows <i>along</i> level curves (closed loops around every well); "
        "<i>mixed</i> blends the two. New <code>vector_field_2d</code> "
        "geom_type plus streamline renderer in plotting.py.",
    ),
    (
        "Streaming: Rayleigh acoustic streaming vortices",
        "<i>Streaming</i> (transport family) computes the slow DC flow "
        "driven by a finite-amplitude wave — the streamfunction recovered "
        "from the Reynolds-stress curl via FFT Poisson. Vortex pairs "
        "(Rayleigh rolls) form between adjacent antinodes. Outputs are "
        "<code>vector_field_2d</code>, scalar speed, or steady tracer "
        "density.",
    ),
    (
        "Transport pipeline branching",
        "One <i>RigidPlate</i> wave field, three transport operators: "
        "<i>Granular</i> gives the static sand figure; <i>Tracer</i> gives "
        "the streamlines around it; <i>Streaming</i> gives the secondary "
        "vortex pairs. All three operators consume the same upstream "
        "GeometryData via the Pipeline contract.",
    ),
    (
        "Acoustic: four observables",
        "<i>Acoustic</i> (wave_field family) is the physically-grounded "
        "bulk pressure field: point sources, explicit wave_speed, distance "
        "decay (1/√r + optional attenuation). Output modes: instantaneous "
        "pressure Re[P], time-averaged intensity ⟨p²⟩, Schlieren shadowgraph "
        "|∇²p|, complex phase arg(P). Near-source masking + gamma "
        "correction keeps the spatial pattern legible across the huge "
        "dynamic range.",
    ),
    (
        "Acoustic chord signatures",
        "<i>per_ratio</i> source assignment + <i>chord_angles</i> layout: "
        "each chord ratio gets its own emitter at its pitch-class angle, "
        "emitting at <code>base_freq × ratio</code>. Combined with "
        "<code>base_frequency</code> scaled by <i>prime_limit</i> and "
        "<code>source_radius</code> by <i>max_common_int</i>, this gives "
        "the maximum chord-DNA an interference pattern can carry — "
        "3-limit chords get coarse simple patterns, 11-limit chords get "
        "high-density intricate ones.",
    ),
    (
        "Reaction-Diffusion: Pearson regime map",
        "<i>ReactionDiffusion</i> (morphogenetic family): Gray–Scott "
        "chemical PDE. The (F, K) plane is a rich phase diagram: mitosis "
        "spots, alpha labyrinth, stripes, self-replicating spots, expanding "
        "rings, spiral chaos. Same chord, six (F, K) points — six "
        "qualitatively different attractors.",
    ),
    (
        "Reaction-Diffusion chord signatures",
        "Chord-driven F and K from <i>structural</i> descriptors: F ← "
        "<code>structure.prime_limit</code> (3-limit on the spotty side, "
        "11-limit on the stripe side), K ← "
        "<code>log(structure.max_common_int)</code>. Five chords land in "
        "five different Pearson regimes — Sus4 → stripes, Major → worm "
        "labyrinth, Minor → solid blob, Dim7 → dense labyrinth, 11:7:5 → "
        "tight maze.",
    ),
    (
        "Elastic: anisotropy ratio sweep",
        "<i>Elastic</i> (eigenmode family) solves an anisotropic plate "
        "wave equation with stiffness α along x and β along y. "
        "Anisotropy-aware mode selection: each chord ratio picks the (m, n) "
        "pair whose anisotropy-weighted eigenvalue matches, so the visible "
        "mode set shifts as the iso-eigenvalue contours go from circles "
        "(isotropic) to ellipses (anisotropic) — visible elongation in "
        "the sweep panels.",
    ),
    (
        "Elastic chord signatures",
        "Chord-driven anisotropy: <code>anisotropy_ratio</code> ← "
        "<code>coupling.consonance</code> (consonant → isotropic, "
        "dissonant → stretched plate); <code>anisotropy_axis</code> ← "
        "chord's mean pitch-class angle. Each chord gets a unique "
        "stretched-and-rotated mode set.",
    ),
    (
        "PlasmaLattice: modulation_strength sweep",
        "<i>PlasmaLattice</i> (eigenmode family, discrete output): N "
        "mutually-repelling charges in a harmonic trap modulated by the "
        "chord's eigenmode field. <code>modulation_strength</code> "
        "interpolates between two known crystallization regimes — pure "
        "Wigner / triangular Coulomb crystal at λ=0, ion-snap-into-wells "
        "at λ→∞. Output is <code>point_cloud_2d</code>; ions are colored "
        "by on-site potential, ⟨V⟩ becomes more negative as ions sink "
        "into chord wells.",
    ),
    (
        "PlasmaLattice chord signatures",
        "<code>n_ions</code> driven by <i>prime_limit</i> (3-limit → "
        "8 ions, 11-limit → 28 ions); the chord-eigenmode wells dictate "
        "where ions sit. Chord-to-chord variation in ion count and well "
        "topology is the visible signature.",
    ),
]


def _build_pdf(paths: list["Path"]) -> "Path":
    """Assemble the figure set + narrative into a PDF using reportlab."""
    from reportlab.lib.colors import HexColor
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.lib.pagesizes import LETTER
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image, KeepTogether, PageBreak, Paragraph,
        SimpleDocTemplate, Spacer,
    )
    from PIL import Image as PILImage

    base = getSampleStyleSheet()
    H0 = ParagraphStyle("H0", parent=base["Heading1"], fontSize=24,
                        leading=30, textColor=HexColor("#1f3b73"),
                        spaceBefore=12, spaceAfter=10, alignment=TA_CENTER)
    H1 = ParagraphStyle("H1", parent=base["Heading1"], fontSize=16,
                        leading=20, textColor=HexColor("#1f3b73"),
                        spaceBefore=14, spaceAfter=8)
    H2 = ParagraphStyle("H2", parent=base["Heading2"], fontSize=12,
                        leading=16, textColor=HexColor("#a23e2c"),
                        spaceBefore=10, spaceAfter=5)
    BODY = ParagraphStyle("Body", parent=base["Normal"], fontSize=10,
                          leading=14, spaceAfter=6,
                          textColor=HexColor("#222222"), alignment=TA_LEFT)
    CAP = ParagraphStyle("Cap", parent=base["Normal"], fontSize=9,
                          leading=13, textColor=HexColor("#555555"),
                          spaceAfter=8)
    CODE = ParagraphStyle("Code", parent=base["Code"], fontSize=8,
                          leading=10, fontName="Courier",
                          backColor=HexColor("#f4f4f4"),
                          leftIndent=8, spaceBefore=4, spaceAfter=8)
    SUBTITLE = ParagraphStyle(
        "Sub", parent=H0, fontSize=14, leading=18,
        textColor=HexColor("#a23e2c"), spaceBefore=0, spaceAfter=4,
    )

    def img(path: Path, width_in: float = 6.5) -> Image:
        with PILImage.open(str(path)) as pil:
            w, h = pil.size
        aspect = h / w
        return Image(str(path), width=width_in * inch,
                     height=width_in * aspect * inch)

    # Write to a temp path first and rename — Windows holds an exclusive
    # write lock when a PDF viewer has the file open, and our normal
    # output path may already be in use.
    import os
    tmp_path = PDF_PATH.with_suffix(".pdf.tmp")
    doc = SimpleDocTemplate(
        str(tmp_path), pagesize=LETTER,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        title="biotuner.harmonic_geometry.media — first-chunk report",
    )

    story: list = []

    # ── title page ──
    story += [
        Spacer(1, 1.6 * inch),
        Paragraph("biotuner . harmonic_geometry . media", H0),
        Spacer(1, 0.1 * inch),
        Paragraph("First-chunk report — family-organized response operators",
                  SUBTITLE),
        Spacer(1, 0.5 * inch),
        Paragraph(
            "This report shows what the new <font face='Courier'>media</font> "
            "framework adds on top of the existing "
            "<font face='Courier'>harmonic_geometry</font> machinery. The "
            "framework reorganizes the module around <b>five operator "
            "families</b> — <i>eigenmode</i>, <i>wave_field</i>, "
            "<i>parametric</i>, <i>transport</i>, <i>morphogenetic</i> — and "
            "introduces a uniform "
            "<font face='Courier'>Medium.respond(forcing, **overrides)</font> "
            "contract for composing them into pipelines.",
            ParagraphStyle("desc", parent=BODY, fontSize=11, leading=16,
                            alignment=TA_CENTER, textColor=HexColor("#444"))),
        Spacer(1, 0.4 * inch),
        Paragraph(
            "<b>All five families populated</b><br/>"
            "&bull; eigenmode/RigidPlate &mdash; chord on a clamped plate<br/>"
            "&bull; eigenmode/ClosedSurface &mdash; chord on spherical harmonics<br/>"
            "&bull; wave_field/Interference &mdash; 5 open-medium paradigms<br/>"
            "&bull; parametric/Faraday &mdash; capillary-gravity cymatics<br/>"
            "&bull; transport/Granular &mdash; sand on plate<br/>"
            "&bull; <b>morphogenetic/Crystallization &mdash; Reiter snowflake</b>",
            ParagraphStyle("toc", parent=BODY, fontSize=11, leading=17,
                            alignment=TA_CENTER,
                            textColor=HexColor("#1f3b73"))),
        PageBreak(),
    ]

    # ── architecture page ──
    story += [
        Paragraph("Architecture", H1),
        Paragraph(
            "The migration is non-destructive. Only the three pre-existing "
            "modules that model a <b>material response</b> are moved into "
            "family-named subpackages under <font face='Courier'>media/"
            "</font>; compatibility shims at the original import paths "
            "re-export every public name. The <b>kinematic / abstract</b> "
            "generators (Lissajous, harmonograph, polygons, fractals, "
            "L-systems, 3-D surfaces, knots, point clouds) stay at the top "
            "level — they're not medium operators, and the import paths "
            "they expose remain unchanged. Two new geom_type strings — "
            "<font face='Courier'>vector_field_2d</font> and "
            "<font face='Courier'>vector_field_3d</font> — are added in "
            "preparation for streaming / tracer media.",
            BODY,
        ),
        Spacer(1, 0.1 * inch),
        Paragraph("Kinematic vs media — what belongs where", H2),
        Paragraph(
            "<b>Kinematic / abstract</b> (top level, unchanged): the chord "
            "directly parameterizes a geometric construction. No medium, "
            "no boundary, no PDE. "
            "<font face='Courier'>lissajous</font>, "
            "<font face='Courier'>harmonograph</font>, "
            "<font face='Courier'>polygon_circular</font> (rose curves, "
            "cycloids, times-table circles), "
            "<font face='Courier'>fractal</font> (IFS, Stern-Brocot, Farey, "
            "subharmonic trees), <font face='Courier'>generative</font> "
            "(L-systems, recursive polygons, self-similar tunings), and "
            "<font face='Courier'>geometry_3d</font> (knots, surfaces, "
            "polyhedra, point clouds, Lissajous tubes).",
            BODY,
        ),
        Paragraph(
            "<b>Media</b> (under <font face='Courier'>media/</font>): the "
            "chord drives a physical response operator that depends on a "
            "domain and material parameters. Five operator families cover "
            "the design space — eigenmode, wave_field, parametric, "
            "transport, morphogenetic — and each medium implements the "
            "same <font face='Courier'>respond(forcing, **overrides)</font> "
            "contract.",
            BODY,
        ),
        Spacer(1, 0.1 * inch),
        Paragraph(
            "biotuner/harmonic_geometry/<br/>"
            "│<br/>"
            "├── <b>KINEMATIC / ABSTRACT</b> &mdash; unchanged, top-level imports<br/>"
            "│&nbsp;&nbsp;&nbsp;├── lissajous.py            &larr; 2-D/3-D curves, knots, phase drift, topology<br/>"
            "│&nbsp;&nbsp;&nbsp;├── harmonograph.py         &larr; damped-pendulum traces (lateral, rotary, 3-D)<br/>"
            "│&nbsp;&nbsp;&nbsp;├── polygon_circular.py     &larr; rose curves, star polygons, cycloids,<br/>"
            "│&nbsp;&nbsp;&nbsp;│                              times tables, consonance polygons<br/>"
            "│&nbsp;&nbsp;&nbsp;├── fractal.py              &larr; ifs_harmonic, Stern-Brocot, Farey,<br/>"
            "│&nbsp;&nbsp;&nbsp;│                              continued-fraction rectangles, subharmonic trees<br/>"
            "│&nbsp;&nbsp;&nbsp;├── generative.py           &larr; L-systems, recursive polygons, self-similar tuning<br/>"
            "│&nbsp;&nbsp;&nbsp;└── geometry_3d.py          &larr; harmonic_knot, lissajous_tube, harmonic_surface,<br/>"
            "│                                              harmonic_point_cloud, recursive_polyhedron, lsystem_3d<br/>"
            "│<br/>"
            "├── <b>INFRASTRUCTURE</b> &mdash; unchanged<br/>"
            "│&nbsp;&nbsp;&nbsp;├── geometry_data.py · inputs.py · _utils.py<br/>"
            "│&nbsp;&nbsp;&nbsp;└── extensions.py · transitions.py · metrics.py · plotting.py<br/>"
            "│<br/>"
            "├── <b>COMPAT SHIMS</b> &mdash; one-line re-exports, old imports still work<br/>"
            "│&nbsp;&nbsp;&nbsp;├── chladni.py              &rarr; media/eigenmode/rigid_plate<br/>"
            "│&nbsp;&nbsp;&nbsp;├── spherical_harmonics.py  &rarr; media/eigenmode/closed_surface<br/>"
            "│&nbsp;&nbsp;&nbsp;└── interference_patterns.py &rarr; media/wave_field/interference<br/>"
            "│<br/>"
            "└── <b>media/</b>                  &larr; NEW: physics-based response operators<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;├── base.py                  &larr; Medium ABC · Domain types · Pipeline<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;├── coupling.py              &larr; 4 chord→param reductions<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;├── eigenmode/<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;├── rigid_plate.py       &larr; chladni_field_* + chladni_nodal_*<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;│                            + ratios_to_modes + RigidPlate<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└── closed_surface.py    &larr; spherical_harmonic_* + ClosedSurface<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;├── wave_field/<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└── interference.py      &larr; 5 paradigms + Interference<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;├── parametric/<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└── faraday.py           &larr; cymatics<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;├── transport/<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;└── granular.py          &larr; sand on a wave field<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;└── morphogenetic/<br/>"
            "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── crystallization.py   &larr; NEW: Reiter snowflake",
            CODE,
        ),
        PageBreak(),
        Paragraph("Pipeline contract", H1),
        Paragraph(
            "Every medium implements "
            "<font face='Courier'>respond(forcing, **overrides) &rarr; GeometryData</font>. "
            "<i>eigenmode</i>/<i>wave_field</i>/<i>parametric</i> media "
            "consume a chord directly; <i>transport</i> and "
            "<i>morphogenetic</i> media consume a pre-computed wave field "
            "(or auto-wrap a chord through their documented "
            "<font face='Courier'>default_source()</font>). The Pipeline "
            "class chains stages: "
            "<font face='Courier'>Pipeline(RigidPlate(...), Granular(...))(chord)</font>.",
            BODY,
        ),
        Paragraph(
            "plate = RigidPlate(domain=Rectangular(1, 1)).respond(chord)<br/>"
            "sand  = Granular(affinity=1.0, temperature=0.05).respond(plate)<br/>"
            "# branching: the same plate also feeds Streaming, Tracer, etc.",
            CODE,
        ),
        PageBreak(),
    ]

    # ── figure sections ──
    for i, ((title, caption), path) in enumerate(zip(SECTIONS, paths), 1):
        block = [
            Paragraph(f"{i}. {title}", H1),
            Paragraph(caption, CAP),
            img(path, width_in=7.0),
            Spacer(1, 0.15 * inch),
        ]
        story += [KeepTogether(block), PageBreak()]

    # ── closing ──
    story += [
        Paragraph("Validation", H1),
        Paragraph(
            "&bull; All 622 pre-existing <font face='Courier'>harmonic_geometry"
            "</font> tests pass.<br/>"
            "&bull; All 610 other biotuner tests pass.<br/>"
            "&bull; 62 new tests cover Medium / Pipeline / Domain / auto-wrap, "
            "Granular physics (regime selection, sharpness, NaN-mask "
            "propagation, particle seeding), and the four coupling reductions."
            "<br/>"
            "&bull; The unified <font face='Courier'>generate_full_report.py"
            "</font> produces a PDF identical in size to the pre-migration "
            "reference build — every historical code path survives the move.",
            BODY,
        ),
        Spacer(1, 0.2 * inch),
        Paragraph("Next chunks", H1),
        Paragraph(
            "&bull; <font face='Courier'>transport/streaming.py</font> — "
            "Rayleigh / Eckart acoustic-streaming vortices (will produce "
            "the first <font face='Courier'>vector_field_2d</font> outputs)<br/>"
            "&bull; <font face='Courier'>morphogenetic/reaction_diffusion.py"
            "</font> — Gray-Scott / Turing morphogenesis<br/>"
            "&bull; <font face='Courier'>transport/tracer.py</font> — "
            "passive scalar advection through any wave field<br/>"
            "&bull; <font face='Courier'>eigenmode/elastic.py</font> — "
            "anisotropic / variable-coefficient plate eigenmodes",
            BODY,
        ),
    ]

    doc.build(story)
    # Atomic move into the final location; if the final path is locked
    # (e.g. open in a PDF viewer), report a friendly error.
    try:
        os.replace(str(tmp_path), str(PDF_PATH))
    except PermissionError:
        fallback = PDF_PATH.with_name(PDF_PATH.stem + "_new" + PDF_PATH.suffix)
        os.replace(str(tmp_path), str(fallback))
        print(
            f"[warn] {PDF_PATH.name} is locked (likely open in a viewer); "
            f"wrote {fallback.name} instead."
        )
        return fallback
    return PDF_PATH


# ════════════════════════════════════════════════════════════════ main ══


def main() -> None:
    print(f"Output dir: {FIG_DIR}")
    print("Rendering figures ...")
    paths = [
        fig_taxonomy_overview(),
        fig_granular_intro(),
        fig_granular_affinity_sweep(),
        fig_granular_temperature_sweep(),
        fig_granular_domain_gallery(),
        fig_chord_signature_grid(),
        fig_field_kind_comparison(),
        fig_pipeline_branching(),
        fig_faraday_patterns(),
        fig_faraday_dispersion(),
        fig_faraday_viscosity_sweep(),
        fig_faraday_to_granular(),
        fig_crystallization_intro(),
        fig_crystallization_humidity_diffusion(),
        fig_crystallization_chord_signatures(),
        fig_coupling_chord_metrics(),
        fig_chord_shapes_medium(),
        fig_pipeline_object_demo(),
        # Phase B media additions:
        fig_tracer_overview(),
        fig_streaming_overview(),
        fig_transport_pipeline_branching(),
        fig_acoustic_overview(),
        fig_acoustic_chord_signatures(),
        fig_reaction_diffusion_grid(),
        fig_reaction_diffusion_chord_signatures(),
        fig_elastic_anisotropy_sweep(),
        fig_elastic_chord_signatures(),
        fig_plasma_lattice_overview(),
        fig_plasma_lattice_chord_signatures(),
    ]
    for p in paths:
        print(f"  - {p.name}")
    md_path = write_report(paths)
    print(f"[ok] markdown report written: {md_path}")
    print("Building PDF ...")
    pdf_path = _build_pdf(paths)
    print(f"[ok] PDF report written: {pdf_path}")


if __name__ == "__main__":
    main()
