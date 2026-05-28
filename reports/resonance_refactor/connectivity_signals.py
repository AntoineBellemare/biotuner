"""Connectivity validation figures for Layer A peak-based methods.

Multi-channel synthetic dataset with KNOWN ground-truth coupling structure:

  e1 — 10 Hz alpha oscillator (clean reference)
  e2 — 10 Hz alpha + π/4 phase offset    (PHASE-LOCKED to e1)
  e3 — 10 Hz alpha + Wiener phase drift  (same freq, NO stable phase locking)
  e4 — 20 Hz beta (2× alpha, locked phase)  (HARMONIC + phase-locked to e1)
  e5 — 17 Hz independent oscillator       (INDEPENDENT)
  e6 — 1/f pink noise                     (NO peaks)

Expected matrix structures:
  harmsim H:        e1-e2, e1-e3 high (same freq);   e1-e4 high (1:2 harmonic);   e5/e6 low
  nm_plv PC:        e1-e2, e1-e4 high (phase-locked); e1-e3 LOW (drifting phase); e5/e6 low
  nm_wpli_complex:  e1-e2 lower (0-lag suppression); e1-e4 still detectable;     e5/e6 ~0
  Resonance R:      Only e1-e2 and e1-e4 high — H AND PC must both score

This is the empirical proof that the registry-based metrics discriminate the
right way on a constructed ground truth.

Usage:
    python reports/resonance_refactor/connectivity_signals.py
"""
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from biotuner.harmonic_connectivity import harmonic_connectivity  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.8,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "legend.frameon": False, "lines.linewidth": 1.3,
})


def _pink_noise(n, sf, seed=0):
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    f = np.fft.rfftfreq(n, d=1.0 / sf)
    f[0] = f[1]
    spectrum = np.fft.rfft(white) / np.sqrt(f)
    pink = np.fft.irfft(spectrum, n=n)
    return pink / np.std(pink)


def build_dataset(sf=500, duration=10.0):
    """Six channels with known coupling structure."""
    rng = np.random.default_rng(42)
    n = int(sf * duration)
    t = np.arange(n) / sf
    dt = 1.0 / sf

    # e1 — clean 10 Hz alpha
    e1 = np.sin(2 * np.pi * 10 * t) + 0.25 * _pink_noise(n, sf, seed=1)

    # e2 — 10 Hz + π/4 phase, locked
    e2 = np.sin(2 * np.pi * 10 * t + np.pi / 4) + 0.25 * _pink_noise(n, sf, seed=2)

    # e3 — 10 Hz with Wiener phase drift (no stable lock)
    dphi = rng.standard_normal(n) * 1.5 * np.sqrt(dt)
    phi_drift = np.cumsum(dphi)
    e3 = np.sin(2 * np.pi * 10 * t + phi_drift) + 0.25 * _pink_noise(n, sf, seed=3)

    # e4 — 20 Hz (1:2 harmonic to e1), locked phase
    e4 = np.sin(2 * np.pi * 20 * t + np.pi / 8) + 0.25 * _pink_noise(n, sf, seed=4)

    # e5 — 17 Hz independent
    e5 = np.sin(2 * np.pi * 17 * t + 1.7) + 0.25 * _pink_noise(n, sf, seed=5)

    # e6 — pink noise only
    e6 = _pink_noise(n, sf, seed=6)

    return np.stack([e1, e2, e3, e4, e5, e6])


ELECTRODE_LABELS = ["e1\n10 Hz\n(ref)", "e2\n10 Hz\n(locked)", "e3\n10 Hz\n(drift)",
                     "e4\n20 Hz\n(harmonic)", "e5\n17 Hz\n(indep)", "e6\npink\n(noise)"]


def _annotate_matrix(ax, M, title, cmap="viridis", vmin=None, vmax=None, fmt=".2f"):
    M_for_range = np.where(np.isfinite(M), M, np.nan)
    eff_vmin = vmin if vmin is not None else float(np.nanmin(M_for_range))
    eff_vmax = vmax if vmax is not None else float(np.nanmax(M_for_range))
    if eff_vmax == eff_vmin:
        eff_vmax = eff_vmin + 1e-9
    im = ax.imshow(M, cmap=cmap, vmin=eff_vmin, vmax=eff_vmax, aspect="equal")
    ax.set_xticks(range(M.shape[0]))
    ax.set_yticks(range(M.shape[0]))
    ax.set_xticklabels(ELECTRODE_LABELS, fontsize=7, rotation=0)
    ax.set_yticklabels(ELECTRODE_LABELS, fontsize=7, rotation=0)
    ax.set_title(title, fontsize=10)
    mid = (eff_vmin + eff_vmax) / 2
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M[i, j]
            if np.isfinite(val):
                color = "white" if val < mid else "black"
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center",
                        fontsize=7, color=color)
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="grey")
    return im


def fig16_overview(hc, data, sf):
    """Multi-channel dataset overview: time series + PSDs."""
    print("Fig 16: connectivity dataset overview ...")
    n_elec = data.shape[0]
    fig, axes = plt.subplots(n_elec, 2, figsize=(12, 9), gridspec_kw={"width_ratios": [2, 1]})

    t = np.arange(int(sf * 2)) / sf  # 2-second window
    for i in range(n_elec):
        # Time series
        axes[i, 0].plot(t, data[i, :len(t)], color="#37474f", lw=0.6)
        axes[i, 0].set_ylabel(f"{ELECTRODE_LABELS[i].split(chr(10))[0]}", fontsize=10, rotation=0, labelpad=20, va="center")
        if i == n_elec - 1:
            axes[i, 0].set_xlabel("Time (s)")
        axes[i, 0].set_xlim(0, 2)

        # PSD
        from scipy.signal import welch
        f, p = welch(data[i], sf, nperseg=int(sf * 2))
        mask = (f >= 2) & (f <= 30)
        axes[i, 1].semilogy(f[mask], p[mask], color="#37474f", lw=1.0)
        axes[i, 1].set_xlim(2, 30)
        if i == n_elec - 1:
            axes[i, 1].set_xlabel("Frequency (Hz)")

    axes[0, 0].set_title("a) Time-domain signals (2-s window)", loc="left", fontsize=11)
    axes[0, 1].set_title("b) Welch PSD", loc="left", fontsize=11)

    fig.suptitle("Figure 16 — Multi-channel test dataset: 6 signals with known coupling structure",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG_DIR / "fig16_connectivity_overview.png")
    fig.savefig(FIG_DIR / "fig16_connectivity_overview.pdf")
    print(f"  wrote fig16_connectivity_overview.png + .pdf")
    plt.close(fig)


def _harmsim_with_max_denom(hc, max_denom=16, FREQ_BANDS=None):
    """Compute harmsim connectivity but round each ratio to a Fraction with
    bounded denominator before applying dyad_similarity. This corrects the
    legacy 10:20 → 1.998 → 999/500 behavior that arises when FOOOF peak
    locations have sub-Hz noise. NOT a change to the core: this rounding is
    applied LOCALLY in the analysis (per user request).
    """
    from fractions import Fraction
    from biotuner.metrics import dyad_similarity
    from biotuner.biotuner_utils import rebound_list
    import itertools

    data = hc.data
    n_elec = len(data)
    M = np.zeros((n_elec, n_elec))
    for i in range(n_elec):
        for j in range(n_elec):
            peaks1, peaks2 = hc._extract_peaks_for_pair(data[i], data[j], FREQ_BANDS=FREQ_BANDS)
            if not peaks1 or not peaks2:
                continue
            pairs = list(itertools.product(peaks1, peaks2))
            ratios = []
            for p in pairs:
                if p[0] > p[1]:
                    ratios.append(p[0] / p[1])
                elif p[1] > p[0]:
                    ratios.append(p[1] / p[0])
            if not ratios:
                continue
            ratios = rebound_list(ratios)
            # Apply max_denom-bounded rounding to each ratio before dyad_similarity
            sims = []
            for r in ratios:
                frac = Fraction(float(r)).limit_denominator(max_denom)
                sims.append(dyad_similarity(frac.numerator / frac.denominator))
            M[i, j] = float(np.mean(sims))
    return M


def fig17_harm_connectivity(hc):
    """Legacy H connectivity (harmsim — default and with bounded max_denom)
    vs harm_fit (now that the harm_fit bug is fixed)."""
    print("Fig 17: harmonicity connectivity ...")
    M_harmsim_default = hc.compute_harm_connectivity(metric="harmsim", graph=False)
    print("  computing harmsim with max_denom=16 (analysis-local rounding)...")
    M_harmsim_bounded = _harmsim_with_max_denom(hc, max_denom=16)
    M_harm_fit = hc.compute_harm_connectivity(metric="harm_fit", graph=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    _annotate_matrix(axes[0], np.nan_to_num(M_harmsim_default),
                      "a) harmsim — default (limit_denominator=1000)\n"
                      "  legacy quirk: FOOOF sub-Hz peak offset → complex Fraction",
                      cmap="viridis")
    _annotate_matrix(axes[1], np.nan_to_num(M_harmsim_bounded),
                      "b) harmsim — analysis-local max_denom=16\n"
                      "  rounds ratios to simple fractions before dyad_similarity",
                      cmap="viridis")
    _annotate_matrix(axes[2], np.nan_to_num(M_harm_fit),
                      "c) harm_fit — # shared harmonics (bug fixed)",
                      cmap="magma")

    fig.suptitle("Figure 17 — Peak-based harmonic connectivity (legacy compute_harm_connectivity)",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIG_DIR / "fig17_harm_connectivity.png")
    fig.savefig(FIG_DIR / "fig17_harm_connectivity.pdf")
    print(f"  wrote fig17_harm_connectivity.png + .pdf")
    plt.close(fig)
    return M_harmsim_default


def fig18_pc_metrics_comparison(hc):
    """All 6 pairwise PC metrics side-by-side."""
    print("Fig 18: PC metrics comparison ...")
    metrics = ["nm_plv", "nm_pli", "nm_wpli", "nm_rrci", "nm_plv_canonical", "nm_wpli_complex"]
    matrices = {}
    for m in metrics:
        print(f"  computing {m} ...")
        t0 = time.time()
        matrices[m] = hc.compute_peak_phase_coupling_connectivity(
            coupling_metric=m, graph=False,
        )
        print(f"    done in {time.time()-t0:.1f}s")

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    for ax, m in zip(axes.flat, metrics):
        _annotate_matrix(ax, matrices[m], m, cmap="viridis", vmin=0, vmax=1)

    fig.suptitle("Figure 18 — Peak-based phase-coupling connectivity: 6 registered pairwise metrics\n"
                 "(compute_peak_phase_coupling_connectivity, coupling_metric=...)",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / "fig18_pc_metrics_comparison.png")
    fig.savefig(FIG_DIR / "fig18_pc_metrics_comparison.pdf")
    print(f"  wrote fig18_pc_metrics_comparison.png + .pdf")
    plt.close(fig)
    return matrices


def fig19_resonance_combine_rules(hc):
    """Peak-based R = H × PC connectivity under different combine rules."""
    print("Fig 19: resonance combine rules ...")
    combines = ["product", "geomean", "harmmean", "min"]
    matrices = {}
    for c in combines:
        print(f"  computing combine={c} ...")
        matrices[c] = hc.compute_peak_resonance_connectivity(
            harm_metric="harmsim", coupling_metric="nm_plv", combine=c, graph=False,
        )

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, c in zip(axes.flat, combines):
        _annotate_matrix(ax, matrices[c], f"R = combine({c})(H, PC)", cmap="plasma", vmin=0)

    fig.suptitle("Figure 19 — Peak-based resonance connectivity: combine rules\n"
                 "(compute_peak_resonance_connectivity, combine=...)",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / "fig19_resonance_combine_rules.png")
    fig.savefig(FIG_DIR / "fig19_resonance_combine_rules.pdf")
    print(f"  wrote fig19_resonance_combine_rules.png + .pdf")
    plt.close(fig)
    return matrices


def fig20_ground_truth_comparison(hc, M_harmsim, M_pc_dict, M_r_dict):
    """Side-by-side: ground truth structure + best H, PC, R picks."""
    print("Fig 20: ground truth comparison ...")
    # Ground truth coupling matrix (qualitative)
    GT = np.array([
        # e1   e2   e3   e4   e5   e6
        [1.0, 1.0, 0.3, 0.8, 0.0, 0.0],   # e1: locked-to-e2, drift-with-e3, harmonic-w-e4, indep-from-e5/e6
        [1.0, 1.0, 0.3, 0.8, 0.0, 0.0],   # e2: same
        [0.3, 0.3, 1.0, 0.2, 0.0, 0.0],   # e3: only weakly to alpha (no PC), nothing to others
        [0.8, 0.8, 0.2, 1.0, 0.0, 0.0],   # e4: harmonic to e1/e2
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],   # e5: independent
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],   # e6: noise
    ])

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    _annotate_matrix(axes[0, 0], GT, "a) Ground truth (qualitative)", cmap="Greys", vmin=0, vmax=1)
    _annotate_matrix(axes[0, 1], np.nan_to_num(M_harmsim) / max(np.nanmax(M_harmsim), 1e-12),
                      "b) harmsim H (normalized)", cmap="viridis", vmin=0, vmax=1)
    _annotate_matrix(axes[1, 0], M_pc_dict["nm_plv"],
                      "c) nm_plv PC", cmap="viridis", vmin=0, vmax=1)
    _annotate_matrix(axes[1, 1], M_r_dict["product"] / max(M_r_dict["product"].max(), 1e-12),
                      "d) R = H · PC (normalized)", cmap="plasma", vmin=0, vmax=1)

    fig.suptitle("Figure 20 — Connectivity framework on ground truth:\n"
                 "R(H × PC) discriminates phase-locked from amplitude-only coupling",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "fig20_connectivity_ground_truth.png")
    fig.savefig(FIG_DIR / "fig20_connectivity_ground_truth.pdf")
    print(f"  wrote fig20_connectivity_ground_truth.png + .pdf")
    plt.close(fig)


def main():
    sf = 500
    data = build_dataset(sf=sf, duration=10.0)
    print(f"Dataset: {data.shape} (n_elec, n_samples), sf={sf} Hz")

    hc = harmonic_connectivity(
        sf=sf, data=data, peaks_function="FOOOF",
        precision=0.5, n_harm=5, min_freq=2, max_freq=30, n_peaks=4,
    )

    fig16_overview(hc, data, sf)
    M_harmsim = fig17_harm_connectivity(hc)
    M_pc = fig18_pc_metrics_comparison(hc)
    M_r = fig19_resonance_combine_rules(hc)
    fig20_ground_truth_comparison(hc, M_harmsim, M_pc, M_r)
    print(f"\nAll figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
