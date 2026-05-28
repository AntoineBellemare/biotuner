"""Layer C validation: compute_cross_resonance_connectivity on multi-channel data.

  Fig 31 — Cross-resonance connectivity matrix on the 6-channel synthetic
           dataset (same as Fig 16 for peak-based connectivity). Compare
           with the peak-based H/PC/R matrices to see how the spectrum-based
           connectivity adds value.
  Fig 32 — Surrogate-normalized z-scored connectivity matrix with all 3
           surrogate generators (phase_randomize, iaaft, time_shuffle).
           Highlights which connections are statistically significant under
           each null.
"""
import sys, time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from biotuner.harmonic_connectivity import harmonic_connectivity  # noqa: E402
from biotuner.resonance import ResonanceConfig  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 9, "legend.frameon": False, "lines.linewidth": 1.3,
})


def _pink_noise(n, sf, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n)
    f = np.fft.rfftfreq(n, 1 / sf); f[0] = f[1]
    return np.fft.irfft(np.fft.rfft(w) / np.sqrt(f), n=n)


def build_six_channel_dataset(sf=500, duration=20.0):
    """Same 6-channel dataset as Layer A Fig 16:
      e1: 10 Hz alpha (clean reference)
      e2: 10 Hz + π/4 phase offset (PHASE-LOCKED to e1)
      e3: 10 Hz + Wiener phase drift (same freq, no lock)
      e4: 20 Hz + π/8 phase offset (1:2 HARMONIC + phase-locked to e1)
      e5: 17 Hz independent
      e6: pink noise only
    """
    rng = np.random.default_rng(42)
    n = int(sf * duration)
    t = np.arange(n) / sf
    dt = 1.0 / sf
    noise = lambda seed: 0.25 * _pink_noise(n, sf, seed=seed)
    e1 = np.sin(2 * np.pi * 10 * t) + noise(1)
    e2 = np.sin(2 * np.pi * 10 * t + np.pi / 4) + noise(2)
    dphi = rng.standard_normal(n) * 1.5 * np.sqrt(dt)
    e3 = np.sin(2 * np.pi * 10 * t + np.cumsum(dphi)) + noise(3)
    e4 = np.sin(2 * np.pi * 20 * t + np.pi / 8) + noise(4)
    e5 = np.sin(2 * np.pi * 17 * t + 1.7) + noise(5)
    e6 = _pink_noise(n, sf, seed=6)
    return np.stack([e1, e2, e3, e4, e5, e6])


ELECTRODE_LABELS = ["e1\n10Hz\n(ref)", "e2\n10Hz\n(locked)", "e3\n10Hz\n(drift)",
                     "e4\n20Hz\n(harm)", "e5\n17Hz\n(indep)", "e6\npink\n(noise)"]


def _annotate_matrix(ax, M, title, cmap="viridis", vmin=None, vmax=None, fmt=".2g", center=None):
    M_for_range = np.where(np.isfinite(M), M, np.nan)
    if vmin is None:
        vmin = float(np.nanmin(M_for_range))
    if vmax is None:
        vmax = float(np.nanmax(M_for_range))
    if vmax == vmin:
        vmax = vmin + 1e-9
    if center is not None:
        bound = max(abs(vmin), abs(vmax))
        vmin, vmax = -bound, bound
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(M.shape[0]))
    ax.set_yticks(range(M.shape[0]))
    ax.set_xticklabels(ELECTRODE_LABELS, fontsize=7)
    ax.set_yticklabels(ELECTRODE_LABELS, fontsize=7)
    ax.set_title(title, fontsize=9)
    mid = (vmin + vmax) / 2
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            val = M[i, j]
            if np.isfinite(val):
                color = "white" if val < mid else "black"
                ax.text(j, i, f"{val:{fmt}}", ha="center", va="center", fontsize=6, color=color)
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=8, color="grey")
    return im


def _save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png")
    fig.savefig(FIG_DIR / f"{name}.pdf")
    print(f"  wrote {name}.png + .pdf")
    plt.close(fig)


def fig31_cross_resonance_matrix():
    """6-channel cross-resonance connectivity using the new defaults."""
    print("Fig 31: 6-channel cross-resonance connectivity ...")
    sf = 500
    data = build_six_channel_dataset(sf=sf)
    hc = harmonic_connectivity(
        sf=sf, data=data, peaks_function="FOOOF",
        precision=0.5, n_harm=5, min_freq=2, max_freq=30, n_peaks=4,
    )
    cfg = ResonanceConfig(
        precision_hz=0.5, fmin=2, fmax=30, noverlap=1, smoothness=1,
        remove_aperiodic=False, harmonic_kernel="harmsim",
        harmonic_kernel_params={"n_harms": 10, "delta_lim": 0.1, "min_notes": 2},
        phase_estimator="stft", coupling_metric="nm_wpli_complex",
        gaussian_smooth_sigma=1.0, combine="product",
        # Inherits cross_pc_reducer='joint', cross_use_ratio_kernel=True defaults
        ratio_kernel="binary",
        ratio_kernel_params={"max_nm": 3, "tolerance": 0.05, "fallback_to_1_1": True},
    )

    H_mat = hc.compute_cross_resonance_connectivity(config=cfg, factor="H", flavor="all", aggregate="max", graph=False)
    PC_mat = hc.compute_cross_resonance_connectivity(config=cfg, factor="PC", flavor="all", aggregate="max", graph=False)
    R_mat = hc.compute_cross_resonance_connectivity(config=cfg, factor="R", flavor="all", aggregate="max", graph=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    _annotate_matrix(axes[0], H_mat, "a) Cross H matrix (max H[all] per pair)", cmap="Blues")
    _annotate_matrix(axes[1], PC_mat, "b) Cross PC matrix (max PC[all] per pair)", cmap="Purples")
    _annotate_matrix(axes[2], R_mat, "c) Cross R matrix (max R[all] per pair)", cmap="Reds")

    fig.suptitle("Figure 31 — Cross-resonance connectivity on 6-channel synthetic data\n"
                 "(refined defaults: joint PC reducer + n:m kernel; aggregate=max)",
                 fontsize=12, fontweight="bold", y=1.005)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    _save(fig, "fig31_cross_resonance_matrix")


def fig32_cross_resonance_zscore_matrix():
    """Z-scored cross-resonance connectivity under 3 surrogate generators."""
    print("Fig 32: z-scored cross-resonance matrix under 3 surrogates ...")
    sf = 500
    data = build_six_channel_dataset(sf=sf)
    hc = harmonic_connectivity(
        sf=sf, data=data, peaks_function="FOOOF",
        precision=0.5, n_harm=5, min_freq=2, max_freq=30, n_peaks=4,
    )
    cfg = ResonanceConfig(
        precision_hz=0.5, fmin=2, fmax=30, noverlap=1, smoothness=1,
        remove_aperiodic=False, harmonic_kernel="harmsim",
        harmonic_kernel_params={"n_harms": 10, "delta_lim": 0.1, "min_notes": 2},
        phase_estimator="stft", coupling_metric="nm_wpli_complex",
        gaussian_smooth_sigma=1.0, combine="product",
        ratio_kernel="binary",
        ratio_kernel_params={"max_nm": 3, "tolerance": 0.05, "fallback_to_1_1": True},
    )

    surrogates = ["phase_randomize", "iaaft", "time_shuffle"]
    n_surr = 25
    results = {}
    for surr_kind in surrogates:
        print(f"  {surr_kind}: running {n_surr} surrogates ...")
        t0 = time.time()
        observed, z, p = hc.compute_cross_resonance_connectivity_zscore(
            config=cfg, factor="R", flavor="all", aggregate="max",
            surrogate_kind=surr_kind, n_surrogates=n_surr, graph=False,
        )
        results[surr_kind] = (observed, z, p)
        print(f"    done in {time.time()-t0:.1f}s")

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    for row, surr_kind in enumerate(surrogates):
        observed, z, p = results[surr_kind]
        _annotate_matrix(axes[row, 0], observed, f"{surr_kind}: observed R (max)", cmap="Reds")
        _annotate_matrix(axes[row, 1], z, f"{surr_kind}: z-score", cmap="coolwarm", center=0)
        _annotate_matrix(axes[row, 2], p, f"{surr_kind}: empirical p", cmap="viridis_r", vmin=0, vmax=1, fmt=".2f")

    fig.suptitle("Figure 32 — Cross-resonance connectivity z-scored against 3 surrogate nulls\n"
                 "(6-channel synthetic data; n_surr=25 per generator)",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "fig32_cross_resonance_zscore_matrix")


def main():
    fig31_cross_resonance_matrix()
    fig32_cross_resonance_zscore_matrix()
    print(f"\nAll figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
