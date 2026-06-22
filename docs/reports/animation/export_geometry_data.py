"""
Export harmonic_geometry sample data as JSON for the Remotion animation.

Run with the biotuner conda env:

    python docs/reports/animation/export_geometry_data.py

Writes ``docs/reports/animation/public/geometry.json`` consumed by the
Remotion scenes. Coordinates are normalized to roughly fit a unit-radius
viewport (centered at 0, range ≈ ±1) so the React side can scale them
freely.
"""

from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

WORKTREE_ROOT = Path(__file__).resolve().parents[3]
if str(WORKTREE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKTREE_ROOT))

from biotuner.harmonic_geometry import (
    HarmonicInput,
    chladni_field_circular,
    chladni_field_rectangular,
    chladni_from_input,
    epicycloid,
    harmonic_point_cloud,
    harmonograph_lateral,
    hypocycloid,
    lissajous_2d,
    lissajous_3d,
    lissajous_compound,
    lsystem_3d,
    rose_curve,
    star_polygon,
    times_table_circle,
    tuning_circle,
)

OUT_DIR = Path(__file__).resolve().parent / "public"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_xy(arr: np.ndarray, target: float = 0.9) -> list[list[float]]:
    a = np.asarray(arr, dtype=np.float64)
    span = float(np.max(np.abs(a)))
    if span == 0:
        span = 1.0
    return (a / span * target).tolist()


# ── Musical chord presets in just intonation ─────────────────────────────────
CHORDS = [
    {
        "name": "Major",
        "ratios_str": "4 : 5 : 6",
        "ratios": [Fraction(1), Fraction(5, 4), Fraction(3, 2)],
        "peaks": [4.02, 5.0, 6.03],
        "amps": [1.0, 0.9, 0.8],
        "phases": [0.0, np.pi / 4, np.pi / 2],
        "damping": [0.018, 0.022, 0.020],
    },
    {
        "name": "Minor",
        "ratios_str": "10 : 12 : 15",
        "ratios": [Fraction(1), Fraction(6, 5), Fraction(3, 2)],
        "peaks": [10.02, 12.0, 15.03],
        "amps": [1.0, 0.85, 0.75],
        "phases": [0.0, np.pi / 3, 2 * np.pi / 3],
        "damping": [0.018, 0.020, 0.022],
    },
    {
        "name": "Dom 7th",
        "ratios_str": "4 : 5 : 6 : 7",
        "ratios": [Fraction(1), Fraction(5, 4), Fraction(3, 2), Fraction(7, 4)],
        "peaks": [4.01, 5.02, 6.0, 7.03],
        "amps": [1.0, 0.9, 0.8, 0.7],
        "phases": [0.0, np.pi / 5, np.pi / 3, np.pi / 7],
        "damping": [0.016, 0.020, 0.018, 0.022],
    },
    {
        "name": "Maj 7th",
        "ratios_str": "8 : 10 : 12 : 15",
        "ratios": [Fraction(1), Fraction(5, 4), Fraction(3, 2), Fraction(15, 8)],
        "peaks": [8.01, 10.0, 12.02, 15.03],
        "amps": [1.0, 0.85, 0.8, 0.7],
        "phases": [0.0, np.pi / 6, np.pi / 3, np.pi / 2],
        "damping": [0.015, 0.018, 0.020, 0.022],
    },
    {
        "name": "Sus 4",
        "ratios_str": "6 : 8 : 9",
        "ratios": [Fraction(1), Fraction(4, 3), Fraction(3, 2)],
        "peaks": [6.02, 8.0, 9.03],
        "amps": [1.0, 0.9, 0.85],
        "phases": [0.0, np.pi / 4, np.pi / 3],
        "damping": [0.018, 0.022, 0.020],
    },
    {
        "name": "Augmented",
        "ratios_str": "1 : 5/4 : 8/5",
        "ratios": [Fraction(1), Fraction(5, 4), Fraction(8, 5)],
        "peaks": [5.02, 6.25, 8.03],
        "amps": [1.0, 0.9, 0.8],
        "phases": [0.0, np.pi / 3, 2 * np.pi / 3],
        "damping": [0.018, 0.020, 0.022],
    },
    {
        "name": "Dim 7th",
        "ratios_str": "5 : 6 : 7 : 9",
        "ratios": [Fraction(1), Fraction(6, 5), Fraction(7, 5), Fraction(9, 5)],
        "peaks": [5.02, 6.0, 7.03, 9.01],
        "amps": [1.0, 0.85, 0.8, 0.7],
        "phases": [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        "damping": [0.015, 0.018, 0.020, 0.022],
    },
]


def _scene_chord_morph() -> dict:
    """Compound Lissajous + harmonograph traces for each musical chord."""
    items = []
    for chord in CHORDS:
        inp = HarmonicInput(
            ratios=chord["ratios"],
            amplitudes=chord["amps"],
        )
        lis = lissajous_compound(inp, n_points=1200)

        h_inp = HarmonicInput(
            peaks=chord["peaks"],
            amplitudes=chord["amps"],
            phases=chord["phases"],
            damping=chord["damping"],
        )
        harm = harmonograph_lateral(h_inp, duration=12.0, sr=100)

        items.append(
            {
                "name": chord["name"],
                "ratios_str": chord["ratios_str"],
                "lissajous": _normalize_xy(lis.coordinates, target=0.85),
                "harmonograph": _normalize_xy(harm.coordinates, target=0.85),
            }
        )
    return {"name": "chord_morph", "chords": items}


def _scene_lissajous_morph() -> dict:
    """Series of Lissajous traces with ratio labels for a smooth morph scene."""
    ratios = [
        Fraction(1, 1),
        Fraction(3, 2),
        Fraction(5, 4),
        Fraction(5, 3),
        Fraction(7, 4),
        Fraction(9, 7),
    ]
    frames = []
    for r in ratios:
        g = lissajous_2d(r, phase=np.pi / 2, n_points=600)
        frames.append(
            {
                "label": f"{r.numerator} : {r.denominator}",
                "ratio_p": r.numerator,
                "ratio_q": r.denominator,
                "points": _normalize_xy(g.coordinates, target=0.85),
                "metadata": {
                    "lobes_x": g.metadata["lobes_x"],
                    "lobes_y": g.metadata["lobes_y"],
                },
            }
        )
    return {"name": "lissajous_morph", "frames": frames}


def _scene_harmonograph_variants() -> dict:
    """Four harmonograph traces: general drift + three chord-based patterns."""
    specs = [
        {
            "label": "harmonic drift",
            "subtitle": "peaks near 2 : 3 : 5 : 7",
            "peaks": [2.01, 3.02, 5.0, 7.03],
            "amps": [1.0, 0.8, 0.6, 0.4],
            "phases": [0.0, np.pi / 5, np.pi / 3, np.pi / 7],
            "damping": [0.04, 0.035, 0.05, 0.045],
            "duration": 40.0,
        },
        {
            "label": "major chord",
            "subtitle": "peaks at 4 : 5 : 6",
            "peaks": [4.02, 5.0, 6.03],
            "amps": [1.0, 0.9, 0.8],
            "phases": [0.0, np.pi / 4, np.pi / 2],
            "damping": [0.012, 0.015, 0.014],
            "duration": 35.0,
        },
        {
            "label": "dom 7th",
            "subtitle": "peaks at 4 : 5 : 6 : 7",
            "peaks": [4.01, 5.02, 6.0, 7.03],
            "amps": [1.0, 0.9, 0.8, 0.65],
            "phases": [0.0, np.pi / 5, np.pi / 3, np.pi / 7],
            "damping": [0.018, 0.022, 0.020, 0.025],
            "duration": 38.0,
        },
        {
            "label": "minor spiral",
            "subtitle": "peaks at 10 : 12 : 15",
            # Lower frequencies + heavier damping + shorter duration so the
            # trace doesn't compact into a black smear of overlapping cycles.
            "peaks": [4.01, 4.81, 6.01],   # 10:12:15 ratio at lower fundamental
            "amps": [1.0, 0.85, 0.75],
            "phases": [0.0, np.pi / 3, 2 * np.pi / 3],
            "damping": [0.060, 0.066, 0.072],
            "duration": 22.0,
        },
    ]
    out = []
    for v in specs:
        h_inp = HarmonicInput(
            peaks=v["peaks"],
            amplitudes=v["amps"],
            phases=v["phases"],
            damping=v["damping"],
        )
        # sr=300: the fastest chord component (~7 Hz) needs ≥~40 samples per
        # cycle to read as a smooth curve — at sr=100 it was only ~14, which
        # showed as visible straight polyline segments ("breaking lines").
        g = harmonograph_lateral(h_inp, duration=v["duration"], sr=300)
        out.append(
            {
                "label": v["label"],
                "subtitle": v["subtitle"],
                "points": _normalize_xy(g.coordinates, target=0.85),
                "duration_s": v["duration"],
            }
        )
    return {"name": "harmonograph_variants", "variants": out}


def _scene_chladni_morph() -> dict:
    """Rectangular and circular Chladni fields – broader mode survey."""
    res_full = 257
    res_keep = 64

    rect_cases = [
        (2, 3), (4, 5), (3, 6), (5, 4),
        (6, 3), (7, 5), (5, 7), (8, 3),
    ]
    # circular: modes_radial=m (1-indexed Bessel zero), modes_angular=n (angular order ≥0)
    # formula: J_n(α_{n,m} · r/R) · cos(n·θ)
    circ_cases = [
        ([1], [0]),  # 0 nodal diameters, 1 ring  (n=0, m=1)
        ([1], [1]),  # 1 diameter                  (n=1, m=1)
        ([1], [2]),  # 2 diameters                 (n=2, m=1)
        ([2], [0]),  # 2 rings                     (n=0, m=2)
    ]

    out = []

    for m, n in rect_cases:
        g = chladni_field_rectangular([(m, n)], resolution=res_full)
        field = np.asarray(g.coordinates, dtype=np.float64)
        block = res_full // res_keep
        trimmed = field[: block * res_keep, : block * res_keep]
        small = trimmed.reshape(res_keep, block, res_keep, block).mean(axis=(1, 3))
        peak = float(np.max(np.abs(small))) or 1.0
        out.append(
            {
                "label": f"rect ({m},{n})",
                "plate": "rect",
                "resolution": res_keep,
                "field": (small / peak).flatten().tolist(),
            }
        )

    for mr, ma in circ_cases:
        g = chladni_field_circular(mr, ma, resolution=res_full)
        field = np.nan_to_num(np.asarray(g.coordinates, dtype=np.float64), nan=0.0)
        block = res_full // res_keep
        trimmed = field[: block * res_keep, : block * res_keep]
        small = trimmed.reshape(res_keep, block, res_keep, block).mean(axis=(1, 3))
        peak = float(np.max(np.abs(small))) or 1.0
        out.append(
            {
                "label": f"circ n={mr[0]},m={ma[0]}",
                "plate": "circ",
                "resolution": res_keep,
                "field": (small / peak).flatten().tolist(),
            }
        )

    return {"name": "chladni_morph", "items": out}


def _scene_star_polygons() -> dict:
    cases = [(5, 2), (7, 3), (8, 3), (9, 4)]
    out = []
    for n, k in cases:
        g = star_polygon(n, k, radius=1.0)
        out.append(
            {
                "label": f"{{{n}/{k}}}",
                "n": n,
                "k": k,
                "points": _normalize_xy(np.asarray(g.coordinates), target=0.78),
            }
        )
    return {"name": "star_polygons", "items": out}


def _scene_tuning_circle() -> dict:
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
    g = tuning_circle(inp, radius=0.85)
    points = np.asarray(g.coordinates).tolist()
    weights = list(g.weights)
    labels = ["1/1", "9/8", "5/4", "4/3", "3/2", "5/3", "15/8"]
    return {
        "name": "tuning_circle",
        "points": points,
        "weights": weights,
        "labels": labels,
    }


def _scene_times_table_sweep() -> dict:
    """Times-table circle for integer multipliers 2 through 12."""
    n_pts = 200
    steps = []
    for mult in range(2, 13):
        g = times_table_circle(n_points=n_pts, multiplier=mult, radius=1.0)
        steps.append(
            {
                "multiplier": mult,
                "points": _normalize_xy(np.asarray(g.coordinates), target=0.85),
                "edges": np.asarray(g.edges).tolist(),
            }
        )
    return {"name": "times_table_sweep", "steps": steps}


def _scene_rose_and_cycloids() -> dict:
    rose = rose_curve(Fraction(5, 1), n_points=900)
    epi = epicycloid(Fraction(5, 2), n_points=900)
    hypo = hypocycloid(Fraction(5, 1), n_points=900)
    return {
        "name": "rose_and_cycloids",
        "items": [
            {
                "label": "rose r = cos(5θ)",
                "points": _normalize_xy(rose.coordinates, target=0.78),
            },
            {
                "label": "epicycloid 5:2",
                "points": _normalize_xy(epi.coordinates, target=0.78),
            },
            {
                "label": "hypocycloid 5:1",
                "points": _normalize_xy(hypo.coordinates, target=0.78),
            },
        ],
    }


def _downsample_field(field: np.ndarray, res_keep: int = 64) -> list[float]:
    """Downsample a 2-D field to res_keep × res_keep via block-mean, normalize to [-1,1]."""
    f = np.nan_to_num(np.asarray(field, dtype=np.float64), nan=0.0)
    h, w = f.shape
    bh = h // res_keep
    bw = w // res_keep
    bh = max(bh, 1)
    bw = max(bw, 1)
    th, tw = bh * res_keep, bw * res_keep
    trimmed = f[:th, :tw]
    small = trimmed.reshape(res_keep, bh, res_keep, bw).mean(axis=(1, 3))
    peak = float(np.max(np.abs(small))) or 1.0
    return (small / peak).flatten().tolist()


def _scene_chladni_from_input() -> dict:
    """Same HarmonicInput through rectangular, circular, and pentagon plates.

    Each chord is shown as three simultaneous heatmaps — exactly the comparison
    in the Phase 1-3 report figure.
    """
    RES_RECT = 257   # rectangular / circular: high-res then downsample ×4
    RES_CIRC = 257
    RES_POLY = 65    # FDM polygon: lower res, trim to 64

    items = []
    for chord in CHORDS:
        inp = HarmonicInput(ratios=chord["ratios"])
        print(f"  {chord['name']} ...", flush=True)

        g_rect = chladni_from_input(inp, plate="rectangular",
                                    plate_kwargs={"resolution": RES_RECT})
        g_circ = chladni_from_input(inp, plate="circular",
                                    plate_kwargs={"resolution": RES_CIRC})
        g_poly = chladni_from_input(inp, plate="polygon",
                                    plate_kwargs={"n_sides": 5, "resolution": RES_POLY})

        items.append(
            {
                "chord_name": chord["name"],
                "ratios_str": chord["ratios_str"],
                "plates": [
                    {
                        "type": "rect",
                        "label": "rectangular",
                        "resolution": 64,
                        "field": _downsample_field(g_rect.coordinates),
                    },
                    {
                        "type": "circ",
                        "label": "circular",
                        "resolution": 64,
                        "field": _downsample_field(g_circ.coordinates),
                    },
                    {
                        "type": "poly5",
                        "label": "pentagon",
                        "resolution": 64,
                        "field": _downsample_field(g_poly.coordinates),
                    },
                ],
            }
        )

    return {"name": "chladni_from_input", "items": items}


def _normalize_xyz(arr: np.ndarray, target: float = 0.9) -> list[list[float]]:
    """Like _normalize_xy but for 3-D coordinate arrays."""
    a = np.asarray(arr, dtype=np.float64)
    span = float(np.max(np.abs(a)))
    if span == 0:
        span = 1.0
    return (a / span * target).tolist()


# ─────────────────────────── 3-D LISSAJOUS KNOTS ─────────────────────────────


def _scene_lissajous_3d_knots() -> dict:
    """3-D Lissajous knots (rotating in the TS scene).

    Each entry has a 3-D point cloud (n_points × 3) plus an explanatory
    pedagogical caption. Frequency triples are pairwise-coprime so the
    curve is a closed knot.
    """
    cases = [
        {"freqs": [3, 4, 5],   "phases": [0, np.pi / 4, np.pi / 2],
         "label": "3 : 4 : 5",
         "subtitle": "three coprime frequencies weave a closed knot"},
        {"freqs": [2, 3, 5],   "phases": [0, np.pi / 3, 2 * np.pi / 3],
         "label": "2 : 3 : 5",
         "subtitle": "low-order trefoil cousin"},
        {"freqs": [3, 5, 7],   "phases": [0, np.pi / 5, np.pi / 3],
         "label": "3 : 5 : 7",
         "subtitle": "three odd primes — tightly wound"},
        {"freqs": [4, 5, 7],   "phases": [0, np.pi / 4, np.pi / 2],
         "label": "4 : 5 : 7",
         "subtitle": "asymmetric weave from a dominant 7-like ratio"},
    ]
    out = []
    for c in cases:
        g = lissajous_3d(c["freqs"], phases=c["phases"], n_points=720)
        out.append({
            "label": c["label"],
            "subtitle": c["subtitle"],
            "vertices": _normalize_xyz(g.coordinates, target=0.85),
            "is_knot": bool(g.metadata.get("knot", False)),
        })
    return {"name": "lissajous_3d_knots", "items": out}


# ─────────────────────────────── 3-D L-SYSTEMS ───────────────────────────────


def _scene_lsystem_3d_variants() -> dict:
    """Chord-driven 3-D L-system trees with hand-tuned per-chord parameters.

    Default ``lsystem_3d`` derives the rule from ``n_components`` only and
    the branching angle from the largest-amplitude ratio.  With uniform
    amplitudes (the default) Major and Sus4 both pick 3/2 as dominant and
    therefore produce identical trees; the same is true of Dom7/Dim7.
    To make every chord read as a different *species* of tree we pass:

    * an explicit ``amplitudes`` vector that emphasises each chord's
      *characteristic* interval (the one that distinguishes it from its
      neighbours — major-3rd for Major, perfect-4th for Sus4, harmonic-7th
      for Dom7, harmonic-tritone for Dim7);
    * a custom 3-D L-system ``rules`` dict per chord, hand-designed so the
      resulting topology mirrors the chord's harmonic character (balanced,
      open, dense, spiralling).
    """
    cases = [
        {
            "name": "Major",
            "ratios":     [Fraction(1), Fraction(5, 4), Fraction(3, 2)],
            "amplitudes": [0.30, 1.00, 0.55],   # emphasise 5/4 → angle 40°
            "rules":      {"F": "F[+F][-F]F[^F][&F]"},  # symmetric 4-branch
            "step":       1.0,
        },
        {
            "name": "Sus4",
            "ratios":     [Fraction(1), Fraction(4, 3), Fraction(3, 2)],
            "amplitudes": [0.30, 1.00, 0.55],   # emphasise 4/3 → angle ≈51°
            "rules":      {"F": "F[^F][&F]F[+F][-F][^FF]"},  # vertical bias
            "step":       1.05,
        },
        {
            "name": "Dom7",
            "ratios":     [Fraction(1), Fraction(5, 4), Fraction(3, 2),
                           Fraction(7, 4)],
            "amplitudes": [0.30, 0.50, 0.55, 1.00],  # emphasise 7/4 → angle ≈33°
            "rules":      {"F": "F[+F][^F][-F][&F][<F][>F]"},  # all-6-axes dense
            "step":       0.85,
        },
        {
            "name": "Dim7",
            "ratios":     [Fraction(1), Fraction(6, 5), Fraction(7, 5),
                           Fraction(12, 7)],
            "amplitudes": [0.30, 0.45, 0.55, 1.00],  # emphasise 12/7 → angle ≈19°
            "rules":      {"F": "F[<F][>F]F[+F][&F][<<F]"},  # roll-heavy spiral
            "step":       0.90,
        },
    ]
    out = []
    for c in cases:
        inp = HarmonicInput(ratios=c["ratios"], amplitudes=c["amplitudes"])
        g = lsystem_3d(
            inp,
            depth=3,
            step_length=c["step"],
            rules=c["rules"],
        )
        coords = np.asarray(g.coordinates, dtype=np.float64)
        edges  = np.asarray(g.edges, dtype=np.int64)
        out.append({
            "name": c["name"],
            "vertices": _normalize_xyz(coords, target=0.85),
            "edges": edges.tolist(),
            "n_segments": int(g.metadata.get("n_segments", len(edges))),
        })
    return {"name": "lsystem_3d_variants", "items": out}


# ────────────────────────── HARMONIC POINT CLOUDS ────────────────────────────


def _scene_harmonic_point_clouds() -> dict:
    """Harmonic point clouds on three surfaces, rotating in the TS scene.

    n_points kept moderate (~1500) so the SVG renderer can handle each
    rotated frame without lag.
    """
    cases = [
        {"name": "Major / sphere",
         "ratios": [Fraction(1), Fraction(5, 4), Fraction(3, 2)],
         "surface": "sphere",
         "subtitle": "consonant chord on a sphere — bright bands of resonance"},
        {"name": "Dom7 / torus",
         "ratios": [Fraction(1), Fraction(5, 4), Fraction(3, 2), Fraction(7, 4)],
         "surface": "torus",
         "subtitle": "the seventh adds a double-period winding to the field"},
        {"name": "Dim7 / klein",
         "ratios": [Fraction(1), Fraction(6, 5), Fraction(7, 5), Fraction(12, 7)],
         "surface": "klein",
         "subtitle": "non-orientable surface — Klein bottle immersion"},
    ]
    out = []
    for c in cases:
        inp = HarmonicInput(ratios=c["ratios"])
        # 600 pts (was 1500) — keeps the field-band structure visible while
        # cutting per-frame SVG circle count by 2.5×, the main render
        # bottleneck on this scene.
        g = harmonic_point_cloud(inp, n_points=600, surface=c["surface"])
        coords = np.asarray(g.coordinates, dtype=np.float64)
        weights = np.asarray(g.weights, dtype=np.float64)
        # Normalise field weights to [0, 1] for colour mapping
        w_lo, w_hi = float(weights.min()), float(weights.max())
        wn = ((weights - w_lo) / max(w_hi - w_lo, 1e-9)).tolist()
        out.append({
            "name": c["name"],
            "surface": c["surface"],
            "subtitle": c["subtitle"],
            "vertices": _normalize_xyz(coords, target=0.85),
            "weights": wn,
        })
    return {"name": "harmonic_point_clouds", "items": out}


# ──────────────────────── EXPANDED CHLADNI (4 plates) ────────────────────────


def _scene_chladni_expanded() -> dict:
    """chladni_from_input rendered on FOUR plates per chord.

    Adds a ``box_3d`` mid-z slice to the existing rect / circ / pentagon
    triplet so each chord shows on four plate kinds simultaneously.
    """
    RES_RECT = 257
    RES_CIRC = 257
    RES_POLY = 65
    RES_BOX  = 49

    items = []
    for chord in CHORDS:
        inp = HarmonicInput(ratios=chord["ratios"])
        print(f"  expanded {chord['name']} ...", flush=True)

        g_rect = chladni_from_input(inp, plate="rectangular",
                                     plate_kwargs={"resolution": RES_RECT})
        g_circ = chladni_from_input(inp, plate="circular",
                                     plate_kwargs={"resolution": RES_CIRC})
        g_poly = chladni_from_input(inp, plate="polygon",
                                     plate_kwargs={"n_sides": 5,
                                                    "resolution": RES_POLY})
        g_box  = chladni_from_input(inp, plate="box_3d",
                                     plate_kwargs={"resolution": RES_BOX})
        # Take a mid-z slice of the 3-D box field for display
        box_field = np.asarray(g_box.coordinates, dtype=np.float64)
        mid = box_field.shape[2] // 2
        box_slice = box_field[:, :, mid]

        items.append({
            "chord_name": chord["name"],
            "ratios_str": chord["ratios_str"],
            "plates": [
                {"type": "rect",  "label": "rectangular",
                 "resolution": 64,
                 "field": _downsample_field(g_rect.coordinates)},
                {"type": "circ",  "label": "circular",
                 # 140 res: the GeometryV3 reel renders the circle as a smooth
                 # nodal-density "sand" Bessel plate and morphs the field itself.
                 "resolution": 140,
                 "field": _downsample_field(g_circ.coordinates, res_keep=140)},
                {"type": "poly5", "label": "pentagon",
                 "resolution": 64,
                 "field": _downsample_field(g_poly.coordinates)},
                {"type": "box3d", "label": "3-D box (mid-z)",
                 "resolution": 49,
                 "field": _downsample_field(box_slice, res_keep=49)},
            ],
        })
    return {"name": "chladni_expanded", "items": items}


def main() -> None:
    print("Exporting chord morph (Lissajous + harmonograph per chord)...")
    chord_morph = _scene_chord_morph()
    print("Exporting lissajous morph...")
    lissajous_morph = _scene_lissajous_morph()
    print("Exporting harmonograph variants...")
    harmonograph_variants = _scene_harmonograph_variants()
    print("Exporting star polygons...")
    star_polygons = _scene_star_polygons()
    print("Exporting times table sweep (multipliers 2-12)...")
    times_table_sweep = _scene_times_table_sweep()
    print("Exporting tuning circle...")
    tuning_circle_data = _scene_tuning_circle()
    print("Exporting rose and cycloids...")
    rose_and_cycloids = _scene_rose_and_cycloids()
    print("Exporting Chladni morph (8 rect + 4 circ)...")
    chladni_morph = _scene_chladni_morph()
    print("Exporting Chladni from input (rect + circ + pentagon per chord)...")
    chladni_from_input_data = _scene_chladni_from_input()

    # ─── new sections for the v2 Geometry composition ────────────────────────
    print("Exporting 3-D Lissajous knots...")
    lissajous_3d_knots = _scene_lissajous_3d_knots()
    print("Exporting 3-D L-system trees...")
    lsystem_3d_data = _scene_lsystem_3d_variants()
    print("Exporting harmonic point clouds...")
    point_clouds = _scene_harmonic_point_clouds()
    print("Exporting Chladni expanded (4 plates per chord)...")
    chladni_expanded = _scene_chladni_expanded()

    payload = {
        "title": "biotuner.harmonic_geometry",
        "subtitle": "Harmonic Geometry of Musical Chords",
        "scenes": {
            "chord_morph": chord_morph,
            "lissajous_morph": lissajous_morph,
            "harmonograph_variants": harmonograph_variants,
            "star_polygons": star_polygons,
            "times_table_sweep": times_table_sweep,
            "tuning_circle": tuning_circle_data,
            "rose_and_cycloids": rose_and_cycloids,
            "chladni_morph": chladni_morph,
            "chladni_from_input": chladni_from_input_data,
            # v2 additions
            "lissajous_3d_knots":    lissajous_3d_knots,
            "lsystem_3d_variants":   lsystem_3d_data,
            "harmonic_point_clouds": point_clouds,
            "chladni_expanded":      chladni_expanded,
        },
    }

    out_path = OUT_DIR / "geometry.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))
    size_kb = out_path.stat().st_size / 1024
    print(f"Wrote {out_path} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
