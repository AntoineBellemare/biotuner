"""Creative views with the *series* as the canvas; brain peaks dropped into it.

The companion script ``math_series_creative.py`` uses the octave as the canvas
and overlays series ratios. Here it is the other way round: each mathematical
series defines the coordinate system, and we show where the EEG's (extended)
peaks fall inside it.

Figures (biotuner-styled) -> ``docs/img/``:

1. ``math_series_comb.png``    — each series, repeated across octaves, is a
   microtonal frequency grid (the "canvas"); scaled to best fit the EEG, with
   the actual EEG extended peaks (Hz) snapped onto the nearest grid step.
   Realizes "take a microtonal tuning across octaves and find the steps of the
   series that match the brain peaks."
2. ``math_series_bubbles.png`` — each series' ratios as bubbles sized by
   simplicity (1/denominator, a Ford-circle idea); shows whether the brain
   peaks land on the simple rungs of each series or in its dense filler.

Run from the repo root::

    python scripts/math_series_series_canvas.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from biotuner.biotuner_object import compute_biotuner
from biotuner.biotuner_utils import ratio2frac
from biotuner.math_series import math_series, series_ratio_pairs
from biotuner.plot_config import get_color_palette, set_biotuner_style

warnings.filterwarnings("ignore")

EEG_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "examples", "data", "EEG_example.npy")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "img")

SF = 1000
SERIES = ["fibonacci", "lucas", "harmonics", "farey"]
ORDER = 9            # keep the across-octave comb legible
MAXDENOM = 40
PEAKS_FUNCTION = "FOOOF"
FMIN, FMAX = 2.0, 45.0
COLORS = dict(zip(SERIES, get_color_palette("biotuner_gradient", n_colors=len(SERIES))))


def cents(ratios):
    return 1200.0 * np.log2(np.asarray(ratios, dtype=float))


def series_ratio_set(name, order=ORDER):
    vals = {round(r, 5) for r, _ in series_ratio_pairs(name, order, octave=2.0)}
    return np.array(sorted(vals))


def fit_representative(data):
    for ch in range(data.shape[0]):
        try:
            bt = compute_biotuner(sf=SF, peaks_function=PEAKS_FUNCTION, precision=0.5)
            bt.peaks_extraction(data[ch], min_freq=2, max_freq=40, n_peaks=6)
            bt.peaks_extension(n_harm=5)
        except Exception:
            continue
        if len(bt.peaks_ratios) >= 4 and getattr(bt, "extended_peaks", None):
            return ch, bt
    raise RuntimeError("No usable channel found.")


# --------------------------------------------------------------------------- comb
def octave_grid(ratios, base):
    """Series ratios replicated across octaves and scaled by ``base`` (Hz)."""
    grid = []
    for k in range(-3, 7):
        for r in ratios:
            g = base * r * (2.0 ** k)
            if FMIN <= g <= FMAX:
                grid.append(g)
    return np.array(sorted(set(np.round(grid, 4))))


def nearest_dev_cents(freqs, grid):
    fc, gc = cents(freqs), cents(grid)
    return np.array([np.min(np.abs(f - gc)) for f in fc])


def best_base(ratios, peaks):
    """Pick the base (within one octave) that best aligns the comb to the peaks."""
    best_b, best_d = 1.0, np.inf
    for b in np.linspace(1.0, 2.0, 240, endpoint=False):
        grid = octave_grid(ratios, b)
        if len(grid) == 0:
            continue
        d = float(np.mean(nearest_dev_cents(peaks, grid)))
        if d < best_d:
            best_b, best_d = b, d
    return best_b, best_d


def plot_comb(series_sets, peaks_hz, path):
    fig, axes = plt.subplots(len(SERIES), 1, figsize=(13, 9), sharex=True)
    for ax, name in zip(axes, SERIES):
        ratios = series_sets[name]
        base, mean_d = best_base(ratios, peaks_hz)
        grid = octave_grid(ratios, base)
        gc = cents(grid)

        for g in grid:  # the series comb = the canvas
            ax.axvline(g, color=COLORS[name], alpha=0.35, lw=1.2, zorder=1)
        # EEG peaks snapped to the nearest comb step
        for f in peaks_hz:
            nearest = grid[np.argmin(np.abs(cents([f])[0] - gc))]
            ax.plot([f, nearest], [0.5, 0.5], color="0.3", lw=0.8, alpha=0.5, zorder=2)
        ax.scatter(peaks_hz, [0.5] * len(peaks_hz), marker="v", s=90,
                   color="#1F1F1F", zorder=3)
        ax.set_yticks([])
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=12)
        ax.text(0.995, 0.82, f"mean miss {mean_d:.0f} c", transform=ax.transAxes,
                ha="right", va="top", fontsize=10, color="#333")
    axes[-1].set_xscale("log")
    axes[-1].set_xlim(FMIN, FMAX)
    axes[-1].set_xlabel("Frequency (Hz)")
    axes[-1].set_xticks([2, 5, 10, 20, 40])
    axes[-1].set_xticklabels(["2", "5", "10", "20", "40"])
    fig.suptitle("Each series as an across-octave frequency comb (canvas)\n"
                 "triangles = EEG extended peaks snapped onto the nearest series step", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- bubbles
def plot_bubbles(series_sets, brain_ext_ratios, matched, path):
    fig, ax = plt.subplots(figsize=(13, 5))
    ext_c = cents(brain_ext_ratios)
    for c in ext_c:
        ax.axvline(c, color="#1F1F1F", lw=1.0, ls="--", alpha=0.25, zorder=1)

    for i, name in enumerate(SERIES):
        y = len(SERIES) - i
        for r in series_ratio_set(name):
            _, q = ratio2frac(r, MAXDENOM)
            size = 900.0 / q  # simpler ratio (small denominator) -> bigger bubble
            is_match = round(r, 5) in matched[name]
            ax.scatter(cents([r])[0], y, s=size, color=COLORS[name],
                       alpha=0.85 if is_match else 0.35,
                       edgecolor="k" if is_match else "none",
                       linewidth=1.3 if is_match else 0, zorder=3 if is_match else 2)
    ax.scatter(ext_c, [0] * len(ext_c), marker="v", s=110, color="#1F1F1F", zorder=4)

    ax.set_yticks([0] + list(range(1, len(SERIES) + 1)))
    ax.set_yticklabels(["EEG extended peaks"] + [SERIES[len(SERIES) - i] for i in range(1, len(SERIES) + 1)])
    ax.set_ylim(-0.7, len(SERIES) + 0.7)
    ax.set_xlim(0, 1200)
    ax.set_xlabel("Cents within the octave")
    ax.set_title("Series ratios as simplicity bubbles (bigger = simpler ratio)\n"
                 "do the EEG peaks land on the simple rungs?  (outlined = matched)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_biotuner_style()
    data = np.load(EEG_PATH)
    ch, bt = fit_representative(data)

    peaks_hz = np.array(sorted(f for f in bt.extended_peaks if FMIN <= f <= FMAX))
    brain_ext_ratios = np.array(sorted(set(np.round(bt.extended_peaks_ratios, 5))))
    series_sets = {name: series_ratio_set(name) for name in SERIES}
    series_sets_comb = {name: series_ratio_set(name, order=7) for name in SERIES}
    ms = math_series(bt, ratios_source="extended_peaks_ratios",
                     series_names=SERIES, maxdenom=MAXDENOM).analyze()
    matched = {name: {round(r, 5) for r, _ in ms.series_scores[name]["matched_series_pairs"]}
               for name in SERIES}

    print(f"Channel {ch}: {len(peaks_hz)} extended peaks in band, best={ms.best_series}")
    plot_comb(series_sets_comb, peaks_hz, os.path.join(OUT_DIR, "math_series_comb.png"))
    plot_bubbles(series_sets, brain_ext_ratios, matched, os.path.join(OUT_DIR, "math_series_bubbles.png"))
    print(f"Figures written to {os.path.normpath(OUT_DIR)}")


if __name__ == "__main__":
    main()
