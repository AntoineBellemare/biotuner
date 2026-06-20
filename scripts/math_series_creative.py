"""Creative views of where an EEG's (extended) peaks sit among the series.

All ratios are folded into one octave [1, 2), i.e. 0-1200 cents. Each
mathematical series defines a lattice of ratios in that octave; a brain
signal contributes a handful of (extended) peak ratios. These figures show
the series lattices and overlay where the brain's extended peaks fall.

Figures (biotuner-styled) -> ``docs/img/``:

1. ``math_series_wheel.png``  — octave wrapped to a circle (angle = cents).
   One ring per series (ticks = its ratios); brain extended peaks are spokes;
   a match lights up where a spoke crosses a ring.
2. ``math_series_ruler.png``  — each series as a lane of ratio-ticks on a
   0-1200 cents axis; brain extended peaks drawn as guide lines.
3. ``math_series_fit.png``    — for every extended peak, the cents distance to
   the nearest ratio of each series (how tightly each series hugs the spectrum).

Run from the repo root::

    python scripts/math_series_creative.py
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
from biotuner.math_series import math_series, series_ratio_pairs
from biotuner.plot_config import get_color_palette, set_biotuner_style

warnings.filterwarnings("ignore")

EEG_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "examples", "data", "EEG_example.npy")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "img")

SF = 1000
SERIES = ["fibonacci", "lucas", "harmonics", "farey"]
ORDER = 13            # moderate order keeps the lattices legible
MAXDENOM = 40
PEAKS_FUNCTION = "FOOOF"
COLORS = dict(zip(SERIES, get_color_palette("biotuner_gradient", n_colors=len(SERIES))))


def cents(ratios):
    return 1200.0 * np.log2(np.asarray(ratios, dtype=float))


def series_ratio_set(name):
    """Unique octave-folded ratios of a series."""
    vals = {round(r, 5) for r, _ in series_ratio_pairs(name, ORDER, octave=2.0)}
    return np.array(sorted(vals))


def fit_representative(data):
    """First channel with >= 4 peak ratios and non-empty extended ratios."""
    for ch in range(data.shape[0]):
        try:
            bt = compute_biotuner(sf=SF, peaks_function=PEAKS_FUNCTION, precision=0.5)
            bt.peaks_extraction(bt_data := data[ch], min_freq=2, max_freq=40, n_peaks=6)
            bt.peaks_extension(n_harm=5)
        except Exception:
            continue
        if len(bt.peaks_ratios) >= 4 and getattr(bt, "extended_peaks_ratios", None):
            return ch, bt
    raise RuntimeError("No usable channel found.")


def matched_series_ratios(bt):
    """Per series, the set of series ratios (rounded) that match an extended peak."""
    ms = math_series(bt, ratios_source="extended_peaks_ratios",
                     series_names=SERIES, maxdenom=MAXDENOM).analyze()
    out = {}
    for name in SERIES:
        out[name] = {round(r, 5) for r, _ in ms.series_scores[name]["matched_series_pairs"]}
    return ms, out


# --------------------------------------------------------------------------- wheel
def plot_wheel(series_sets, brain_ext, matched, path):
    fig = plt.figure(figsize=(8.5, 8.5))
    ax = fig.add_subplot(projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    radii = np.linspace(1.25, 2.0, len(SERIES))
    for name, r_ring in zip(SERIES, radii):
        ratios = series_sets[name]
        theta = 2 * np.pi * np.log2(ratios)
        ax.plot(np.linspace(0, 2 * np.pi, 360), [r_ring] * 360, color="0.85", lw=1, zorder=1)
        ax.scatter(theta, [r_ring] * len(theta), s=22, color=COLORS[name],
                   alpha=0.55, zorder=2, label=name)
        # light up matched ratios on this ring
        if matched[name]:
            mt = 2 * np.pi * np.log2(np.array(sorted(matched[name])))
            ax.scatter(mt, [r_ring] * len(mt), s=130, color=COLORS[name],
                       edgecolor="k", linewidth=1.2, zorder=4)

    # brain extended peaks as spokes
    for r in brain_ext:
        th = 2 * np.pi * np.log2(r)
        ax.plot([th, th], [0.95, 2.08], color="#1F1F1F", lw=1.6, alpha=0.55, zorder=3)
    ax.scatter(2 * np.pi * np.log2(brain_ext), [2.13] * len(brain_ext),
               marker="v", s=80, color="#1F1F1F", zorder=5, label="EEG extended peaks")

    ax.set_ylim(0, 2.3)
    ax.set_yticklabels([])
    ax.set_xticks(np.linspace(0, 2 * np.pi, 13)[:-1])
    ax.set_xticklabels([f"{c}" for c in range(0, 1200, 100)])
    ax.set_title("Octave wheel: series ratio-rings + EEG extended-peak spokes\n"
                 "(angle = cents; filled dot = matched)", pad=18)
    ax.legend(loc="center left", bbox_to_anchor=(1.04, 0.5), fontsize=10, frameon=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- ruler
def plot_ruler(series_sets, brain_ext, matched, path):
    fig, ax = plt.subplots(figsize=(13, 5))
    ext_cents = cents(brain_ext)

    # guide lines from each extended peak across all lanes
    for c in ext_cents:
        ax.axvline(c, color="#1F1F1F", lw=1.0, alpha=0.18, zorder=1)

    for i, name in enumerate(SERIES):
        y = len(SERIES) - i  # series stacked above the EEG lane
        cs = cents(series_sets[name])
        ax.vlines(cs, y - 0.32, y + 0.32, color=COLORS[name], lw=1.4, alpha=0.55, zorder=2)
        if matched[name]:
            mc = cents(np.array(sorted(matched[name])))
            ax.vlines(mc, y - 0.4, y + 0.4, color=COLORS[name], lw=3.0, zorder=3)

    # EEG extended-peak lane at the bottom
    ax.scatter(ext_cents, [0] * len(ext_cents), marker="v", s=110,
               color="#1F1F1F", zorder=4)

    ax.set_yticks([0] + list(range(1, len(SERIES) + 1)))
    ax.set_yticklabels(["EEG extended peaks"] + [SERIES[len(SERIES) - i] for i in range(1, len(SERIES) + 1)])
    ax.set_ylim(-0.7, len(SERIES) + 0.7)
    ax.set_xlim(0, 1200)
    ax.set_xlabel("Cents within the octave")
    ax.set_title("Where the EEG extended peaks land among each series' ratio lattice\n"
                 "(thin ticks = series ratios, bold = matched, triangles = EEG peaks)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- fit
def plot_fit(series_sets, brain_ext, path):
    ext_cents = cents(brain_ext)
    fig, ax = plt.subplots(figsize=(9, 5))
    rng = np.linspace(-0.18, 0.18, len(ext_cents))
    for i, name in enumerate(SERIES):
        scs = cents(series_sets[name])
        nearest = np.array([np.min(np.abs(scs - c)) for c in ext_cents])
        ax.scatter(np.full_like(nearest, i) + rng, nearest, s=70,
                   color=COLORS[name], edgecolor="k", linewidth=0.5, zorder=3, alpha=0.9)
        ax.hlines(nearest.mean(), i - 0.3, i + 0.3, color=COLORS[name], lw=3, zorder=2)
        ax.text(i, nearest.mean() + 4, f"mean {nearest.mean():.0f}c",
                ha="center", fontsize=10, color="#333")

    ax.set_xticks(range(len(SERIES)))
    ax.set_xticklabels(SERIES)
    ax.set_ylabel("Cents to nearest series ratio")
    ax.set_xlabel("Mathematical series")
    ax.set_title("How tightly each series hugs the EEG's extended-peak ratios\n"
                 "(each point = one extended peak; lower = closer)")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_biotuner_style()
    data = np.load(EEG_PATH)
    ch, bt = fit_representative(data)
    brain_ext = np.array(sorted(set(np.round(bt.extended_peaks_ratios, 5))))
    series_sets = {name: series_ratio_set(name) for name in SERIES}
    ms, matched = matched_series_ratios(bt)

    print(f"Representative channel {ch}: {len(bt.peaks)} peaks, "
          f"{len(brain_ext)} extended-peak ratios. best(extended)={ms.best_series}")
    print("Extended-peak ratios (cents):", [int(c) for c in cents(brain_ext)])

    plot_wheel(series_sets, brain_ext, matched, os.path.join(OUT_DIR, "math_series_wheel.png"))
    plot_ruler(series_sets, brain_ext, matched, os.path.join(OUT_DIR, "math_series_ruler.png"))
    plot_fit(series_sets, brain_ext, os.path.join(OUT_DIR, "math_series_fit.png"))
    print(f"Figures written to {os.path.normpath(OUT_DIR)}")


if __name__ == "__main__":
    main()
