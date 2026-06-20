"""Showcase for :mod:`biotuner.math_series` on real EEG.

Runs the actual workflow the module is meant for: take a biosignal, extract its
spectral peaks AND extended (harmonic-fit) peaks with ``compute_biotuner``,
derive the ratios, and ask which mathematical series (Fibonacci, Lucas,
harmonics, Farey) is most present. Nothing about the answer is known in advance.

Figures (styled with :func:`biotuner.plot_config.set_biotuner_style`) go to
``docs/img/``:

1. ``math_series_eeg_chunks.png``     — a grid of EEG chunks (channels); for each,
   the match proportion per series for the PEAK ratios vs the EXTENDED-peak ratios.
2. ``math_series_eeg_pipeline.png``   — one chunk end to end: spectrum + detected
   peaks, then its peak-ratio vs extended-ratio proportions.
3. ``math_series_eeg_population.png`` — across all channels: how often each series
   wins, and the mean match proportion per series.
4. ``math_series_eeg_scatter.png``    — the sequence-pairs scatter for one chunk.

Run from the repo root::

    python scripts/math_series_showcase.py

(A controlled "ground-truth" check — feed Fibonacci-numbered peaks, confirm the
matcher returns Fibonacci — lives in the test suite, not here; it validates the
logic but is circular by construction, so it is not a meaningful demo.)
"""

import os
import sys

# Prefer the local worktree package over any installed biotuner.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from biotuner.biotuner_object import compute_biotuner
from biotuner.math_series import math_series
from biotuner.plot_config import get_color_palette, set_biotuner_style

warnings.filterwarnings("ignore")

EEG_PATH = os.path.join(os.path.dirname(__file__), "..", "docs", "examples", "data", "EEG_example.npy")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "img")

SF = 1000
SERIES = ["fibonacci", "lucas", "harmonics", "farey"]
MAXDENOM = 40
PEAKS_FUNCTION = "FOOOF"
MIN_FREQ, MAX_FREQ, N_PEAKS = 2, 40, 6

# Two consistent colours for the two ratio sources, drawn from the biotuner palette.
_GRAD = get_color_palette("biotuner_gradient", n_colors=5)
PK_COLOR, EXT_COLOR = _GRAD[2], _GRAD[0]  # orange = peaks, blue = extended


def fit_channel(signal):
    """Extract peaks + extended peaks from one channel. Returns the bt object."""
    bt = compute_biotuner(sf=SF, peaks_function=PEAKS_FUNCTION, precision=0.5)
    bt.peaks_extraction(signal, min_freq=MIN_FREQ, max_freq=MAX_FREQ, n_peaks=N_PEAKS)
    try:
        bt.peaks_extension(n_harm=5)
    except Exception:
        pass  # extended peaks are optional
    return bt


def pick_chunks(data, n=6):
    """Return up to ``n`` (channel, bt) pairs that have both peak and extended ratios."""
    out = []
    for ch in range(data.shape[0]):
        try:
            bt = fit_channel(data[ch])
        except Exception:
            continue
        if len(bt.peaks_ratios) >= 4 and getattr(bt, "extended_peaks_ratios", None):
            out.append((ch, bt))
        if len(out) >= n:
            break
    return out


def _match(bt, source):
    return math_series(bt, ratios_source=source, series_names=SERIES, maxdenom=MAXDENOM).analyze()


def plot_chunks_grid(chunks, path):
    """Grid of EEG chunks: peak-ratio vs extended-ratio proportions per series."""
    ncols = 3
    nrows = (len(chunks) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.4 * nrows), squeeze=False)
    x = np.arange(len(SERIES))
    w = 0.38
    for idx, (ch, bt) in enumerate(chunks):
        ax = axes[idx // ncols][idx % ncols]
        ms_p, ms_e = _match(bt, "peaks_ratios"), _match(bt, "extended_peaks_ratios")
        ax.bar(x - w / 2, [ms_p.series_scores[s]["proportion"] for s in SERIES],
               width=w, color=PK_COLOR, label="peak ratios")
        ax.bar(x + w / 2, [ms_e.series_scores[s]["proportion"] for s in SERIES],
               width=w, color=EXT_COLOR, label="extended ratios")
        ax.set_xticks(x)
        ax.set_xticklabels(SERIES, rotation=20, ha="right", fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title(f"chunk {ch}   best (peaks): {ms_p.best_series}", fontsize=12)
        if idx % ncols == 0:
            ax.set_ylabel("proportion matched")
    for j in range(len(chunks), nrows * ncols):  # hide spare axes
        axes[j // ncols][j % ncols].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Which series is present in each EEG chunk — peak vs extended-peak ratios",
                 y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pipeline(bt, signal, path):
    """One chunk end to end: spectrum + peaks, and proportions per series."""
    from scipy.signal import welch

    ms_p = _match(bt, "peaks_ratios")
    has_ext = bool(getattr(bt, "extended_peaks_ratios", None))
    ms_e = _match(bt, "extended_peaks_ratios") if has_ext else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    f, pxx = welch(signal, fs=SF, nperseg=min(len(signal), 1024))
    mask = f <= MAX_FREQ + 5
    ax1.semilogy(f[mask], pxx[mask], color="0.35", lw=1.4, label="PSD (Welch)")
    for i, p in enumerate(bt.peaks):
        ax1.axvline(p, color=PK_COLOR, lw=1.8, alpha=0.95, label="peaks" if i == 0 else None)
    if has_ext and getattr(bt, "extended_peaks", None) is not None:
        for i, p in enumerate(bt.extended_peaks):
            ax1.axvline(p, color=EXT_COLOR, lw=1.1, ls="--", alpha=0.7,
                        label="extended peaks" if i == 0 else None)
    ax1.set_xlim(0, MAX_FREQ + 5)
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Power (log)")
    ax1.set_title("Step 1-2: extract peaks")
    ax1.legend(fontsize=9)

    x = np.arange(len(SERIES))
    w = 0.38
    ax2.bar(x - w / 2, [ms_p.series_scores[s]["proportion"] for s in SERIES],
            width=w, color=PK_COLOR, label="peak ratios")
    if ms_e is not None:
        ax2.bar(x + w / 2, [ms_e.series_scores[s]["proportion"] for s in SERIES],
                width=w, color=EXT_COLOR, label="extended ratios")
    ax2.set_xticks(x)
    ax2.set_xticklabels(SERIES)
    ax2.set_ylabel("Proportion matched")
    ax2.set_ylim(0, 1)
    ax2.set_title(f"Step 3-4: best = {ms_p.best_series}")
    ax2.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return ms_p


def analyze_population(data):
    best_counts = {s: 0 for s in SERIES}
    prop_lists = {s: [] for s in SERIES}
    n_ok = 0
    for ch in range(data.shape[0]):
        try:
            ms = _match(fit_channel(data[ch]), "peaks_ratios")
        except Exception:
            continue
        n_ok += 1
        best_counts[ms.best_series] += 1
        for s in SERIES:
            prop_lists[s].append(ms.series_scores[s]["proportion"])
    return best_counts, prop_lists, n_ok


def plot_population(best_counts, prop_lists, n_ok, path):
    colors = get_color_palette("biotuner_gradient", n_colors=len(SERIES))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    ax1.bar(SERIES, [best_counts[s] for s in SERIES], color=colors)
    ax1.set_ylabel("Channels where it wins")
    ax1.set_title(f"Best-matching series across {n_ok} EEG channels")
    means = [np.mean(prop_lists[s]) for s in SERIES]
    stds = [np.std(prop_lists[s]) for s in SERIES]
    ax2.bar(SERIES, means, yerr=stds, capsize=4, color=colors)
    ax2.set_ylabel("Match proportion (mean +/- sd)")
    ax2.set_ylim(0, 1)
    ax2.set_title("Mean presence of each series")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    set_biotuner_style()
    data = np.load(EEG_PATH)
    print(f"Loaded EEG {data.shape} @ {SF} Hz; peaks_function={PEAKS_FUNCTION}")

    chunks = pick_chunks(data, n=6)
    if not chunks:
        raise RuntimeError("No channel produced usable peak + extended ratios.")
    print("Chunks used (channels):", [c for c, _ in chunks])
    for ch, bt in chunks:
        p, e = _match(bt, "peaks_ratios"), _match(bt, "extended_peaks_ratios")
        print(f"  ch{ch:3d}  peaks->{p.best_series:10s} "
              f"{ {s: round(p.series_scores[s]['proportion'], 2) for s in SERIES} }  "
              f"| extended->{e.best_series}")

    plot_chunks_grid(chunks, os.path.join(OUT_DIR, "math_series_eeg_chunks.png"))

    rep_ch, rep_bt = chunks[0]
    ms = plot_pipeline(rep_bt, data[rep_ch], os.path.join(OUT_DIR, "math_series_eeg_pipeline.png"))
    ms.plot_ratio_pairs(plot=False, save=True,
                        savename=os.path.join(OUT_DIR, "math_series_eeg_scatter"))

    best_counts, prop_lists, n_ok = analyze_population(data)
    plot_population(best_counts, prop_lists, n_ok,
                    os.path.join(OUT_DIR, "math_series_eeg_population.png"))
    print(f"\nPopulation ({n_ok} channels): best wins = {best_counts}")
    print("  mean proportion:", {s: round(float(np.mean(prop_lists[s])), 2) for s in SERIES})
    print(f"\nFigures written to {os.path.normpath(OUT_DIR)}")


if __name__ == "__main__":
    main()
