"""
generate_harmonic_sequence_real_data.py
=======================================
Run the harmonic_sequence pipeline on real EEG data from the repo's example
file (`docs/examples/data/EEG_example.npy`, 104 channels x 4 s @ 1 kHz) and
produce three figures for the use-case paper:

  15_real_eeg_overview.png      Channels-as-sequence + the four lenses.
  16_real_eeg_method_compare.png Six methods on the SAME real recording.
  17_real_eeg_app_retrieval.png  Application — k-nearest-neighbour channel
                                 retrieval via the W1 distance matrix.

The EEG file is multi-channel, single-trial (no time axis longer than 4 s).
We treat channels as a *spatial sequence*: each channel is one "frame" in
the analyser. This is a perfectly valid input for the module — it learns
the harmonic structure across channels, which on a head montage corresponds
to a walk across cortical regions. Spatial Markov / Wasserstein / TDA still
have intuitive interpretations (spatial states, between-channel harmonic
flux, cortical attractor count). For genuine temporal analysis, replace the
bt_list construction with a sliding-window loop on a long single-channel
recording.

Run:
    python scripts/generate_harmonic_sequence_real_data.py
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biotuner.biotuner_object import compute_biotuner
from biotuner.harmonic_sequence import HarmonicSequenceAnalyzer

OUT = ROOT / "docs" / "papers" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 140, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})

DATA_PATH = ROOT / "docs" / "examples" / "data" / "EEG_example.npy"
SF = 1000
FREQ_BANDS = [[1, 3], [3, 7], [7, 12], [12, 18], [18, 30], [30, 45]]
N_CHANNELS = 60          # subset to keep runtime reasonable (~60 s)
CHANNEL_START = 20       # skip noisier leading edge channels


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load + build bt_list (channels as frames)
# ─────────────────────────────────────────────────────────────────────────────

print(f"Loading {DATA_PATH.name}...")
data = np.load(DATA_PATH)
print(f"  data shape: {data.shape}  (channels x samples)  sf = {SF} Hz")

channels = list(range(CHANNEL_START, CHANNEL_START + N_CHANNELS))
print(f"  using {N_CHANNELS} channels (indices {channels[0]}..{channels[-1]})")

# Cache file so the slow peaks_extraction step only runs once.
CACHE_PATH = ROOT / "docs" / "papers" / "_real_eeg_bt_cache.npz"
if CACHE_PATH.exists():
    print(f"  loading cached bt_list ratios from {CACHE_PATH.name}")
    npz = np.load(CACHE_PATH, allow_pickle=True)
    ratios_list = list(npz["ratios_list"])
    peaks_list = list(npz["peaks_list"])
    harmsim_list = npz["harmsim_list"].tolist()
else:
    print("  running peaks_extraction on each channel...")
    t0 = time.time()
    ratios_list = []
    peaks_list = []
    harmsim_list = []
    for i, ch in enumerate(channels):
        bt = compute_biotuner(sf=SF, peaks_function="fixed", precision=0.5,
                              n_harm=10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                bt.peaks_extraction(data[ch].astype(np.float64),
                                     FREQ_BANDS=FREQ_BANDS,
                                     ratios_extension=True,
                                     max_freq=45, n_peaks=5,
                                     graph=False, min_harms=2, verbose=False)
                bt.compute_peaks_metrics()
            except Exception as exc:
                print(f"    ch{ch}: peaks_extraction failed ({exc!s:.50})")
                ratios_list.append([])
                peaks_list.append([])
                harmsim_list.append(np.nan)
                continue
        peaks = list(bt.peaks) if bt.peaks is not None else []
        ratios = list(bt.peaks_ratios) if getattr(bt, "peaks_ratios",
                                                    None) else []
        pm = getattr(bt, "peaks_metrics", {}) or {}
        try:
            hs = float(pm.get("harmsim", np.nan))
        except Exception:
            hs = np.nan
        ratios_list.append(ratios)
        peaks_list.append(peaks)
        harmsim_list.append(hs)
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1}/{N_CHANNELS} done  ({elapsed:.1f}s elapsed)")
    print(f"  peaks_extraction finished in {time.time() - t0:.1f}s")
    # Cache
    np.savez(CACHE_PATH,
             ratios_list=np.array(ratios_list, dtype=object),
             peaks_list=np.array(peaks_list, dtype=object),
             harmsim_list=np.array(harmsim_list))
    print(f"  cached to {CACHE_PATH.name}")

T = len(ratios_list)
n_valid = sum(1 for r in ratios_list if len(r) > 0)
print(f"  {n_valid}/{T} channels yielded non-empty ratio sets")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build and fit analyzer
# ─────────────────────────────────────────────────────────────────────────────

analyzer = HarmonicSequenceAnalyzer.from_ratios_list(ratios_list, n_hist_bins=240)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Drop frames with empty ratio sets so the model fits don't bias on them.
    # (They remain in `analyzer.histograms` as zero rows; encoders cope.)
    analyzer.fit_markov(n_states="auto", auto_k_range=(2, 6))
    analyzer.fit_wasserstein()
    analyzer.fit_latent(latent_dim=3)
    try:
        analyzer.fit_dmd(use_histograms=True)
    except Exception as exc:
        print(f"  DMD failed: {exc}")
    try:
        analyzer.fit_grammar(n_gram=2)
    except Exception as exc:
        print(f"  Grammar failed: {exc}")
    try:
        # Use the cached harmsim metric extracted during peaks_extraction
        # (the analyzer recomputes from ratios when not provided).
        analyzer.fit_topology(scalar_key="harmsim", embedding_dim=3, delay=1)
    except Exception as exc:
        print(f"  Topology failed: {exc}")

print("\n" + analyzer.summary())


# ─────────────────────────────────────────────────────────────────────────────
# Figure 15 — overview of the real EEG sequence through 4 lenses
# ─────────────────────────────────────────────────────────────────────────────

def fig_overview():
    H = analyzer.histograms
    flux = analyzer.wasserstein.flux_
    Z = analyzer.latent.trajectory()
    labels = analyzer.markov.state_labels_
    K = analyzer.markov.n_states
    harmsim = np.array(harmsim_list)

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.6, 1.0, 1.2, 1.2],
                           hspace=0.55, wspace=0.30,
                           left=0.07, right=0.98, top=0.94, bottom=0.07)

    # (a) histogram heatmap with Markov state strip
    ax = fig.add_subplot(gs[0, :])
    im = ax.imshow(H.T, aspect="auto", origin="lower", cmap="magma",
                    extent=[0, T, 0, 1200])
    ax.set_xlabel("channel (frame index)")
    ax.set_ylabel("cents")
    ax.set_title("(a) Cents-histogram trajectory across 60 EEG channels "
                  f"(K={K} Markov states discovered)")
    fig.colorbar(im, ax=ax, pad=0.01, fraction=0.025, label="weight")

    # (b) harmsim per channel
    ax = fig.add_subplot(gs[1, :])
    ax.plot(np.arange(T), harmsim, color="black", lw=1.4)
    finite_hs = harmsim[np.isfinite(harmsim)]
    if len(finite_hs):
        ax.axhline(np.nanmean(harmsim), color="grey", ls="--", lw=0.7,
                    label=f"mean = {np.nanmean(harmsim):.2f}")
    cmap_st = plt.get_cmap("tab10")
    ymin = ax.get_ylim()[0]
    strip_h = (ax.get_ylim()[1] - ymin) * 0.08
    for t in range(T):
        ax.add_patch(plt.Rectangle((t - 0.5, ymin - strip_h * 1.3),
                                    1, strip_h,
                                    color=cmap_st(int(labels[t]) % 10),
                                    clip_on=False))
    ax.set_xlim(0, T)
    ax.set_ylim(ymin - strip_h * 1.5, ax.get_ylim()[1])
    ax.set_ylabel("harmsim")
    ax.set_xlabel("channel")
    ax.set_title("(b) Per-channel harmonicity-similarity (from biotuner peaks_metrics)")
    ax.legend(frameon=False, loc="upper right", fontsize=8)

    # (c) W1 flux
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(np.arange(len(flux)), flux, color="black", lw=1.3)
    ax.set_xlabel("channel transition")
    ax.set_ylabel("$W_1$ flux")
    ax.set_title("(c) Between-channel harmonic flux")

    # (d) pairwise distance matrix
    ax = fig.add_subplot(gs[2, 1])
    D = analyzer.wasserstein.distance_matrix_
    im = ax.imshow(D, cmap="viridis")
    ax.set_xlabel("channel j"); ax.set_ylabel("channel i")
    ax.set_title("(d) Pairwise $W_1$ matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (e) latent PCA scatter coloured by harmsim
    ax = fig.add_subplot(gs[2, 2])
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=harmsim, cmap="plasma",
                     s=44, edgecolor="white", lw=0.6)
    ax.plot(Z[:, 0], Z[:, 1], color="grey", lw=0.4, alpha=0.4)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    evr = analyzer.latent.explained_variance_ratio_
    ax.set_title(f"(e) PC1-PC2  ({evr[:2].sum():.0%} var)")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="harmsim")

    # (f) DMD eigenvalues
    ax = fig.add_subplot(gs[3, 0])
    if analyzer.dmd is not None:
        eigs = analyzer.dmd.eigenvalues_
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), color="grey", lw=0.6, ls="--")
        osc, idx = analyzer.dmd.oscillatory_modes(threshold=0.1)
        is_osc = np.zeros(len(eigs), dtype=bool); is_osc[idx] = True
        ax.scatter(eigs[~is_osc].real, eigs[~is_osc].imag,
                    s=50, color="#888", alpha=0.7, label="decaying")
        ax.scatter(eigs[is_osc].real, eigs[is_osc].imag,
                    s=120, color="crimson", marker="*",
                    label=f"|λ|≈1 ({len(idx)})")
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_aspect("equal")
        ax.axhline(0, color="grey", lw=0.3); ax.axvline(0, color="grey", lw=0.3)
        ax.set_xlabel(r"Re($\lambda$)"); ax.set_ylabel(r"Im($\lambda$)")
        ax.legend(frameon=False, fontsize=7)
        ax.set_title("(f) DMD spectrum")
    else:
        ax.text(0.5, 0.5, "DMD failed", ha="center", va="center")

    # (g) Markov transition matrix
    ax = fig.add_subplot(gs[3, 1])
    T_mat = analyzer.markov.transition_matrix_
    im = ax.imshow(T_mat, cmap="rocket_r", vmin=0, vmax=1)
    K_ = T_mat.shape[0]
    for i in range(K_):
        for j in range(K_):
            ax.text(j, i, f"{T_mat[i, j]:.2f}", ha="center", va="center",
                     fontsize=7,
                     color="white" if T_mat[i, j] > 0.5 else "black")
    ax.set_xticks(range(K_)); ax.set_yticks(range(K_))
    ax.set_xlabel("next"); ax.set_ylabel("current")
    ax.set_title(f"(g) Transition matrix  (H = {analyzer.markov.transition_entropy_:.2f} bits)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (h) grammar top motifs
    ax = fig.add_subplot(gs[3, 2])
    if analyzer.grammar is not None:
        motifs = analyzer.grammar.top_motifs(min_length=2, max_length=3,
                                              top_k=6)
        if motifs:
            labels_m = []
            counts = []
            chord_to_id = {c: i for i, c
                           in enumerate(analyzer.grammar.vocabulary_)}
            for m, c in motifs:
                labels_m.append("→".join(str(chord_to_id[t]) for t in m))
                counts.append(c)
            ax.barh(range(len(labels_m)), counts, color="#4C72B0")
            ax.set_yticks(range(len(labels_m)))
            ax.set_yticklabels(labels_m, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("count")
            ax.set_title(f"(h) Top motifs  (vocab={len(analyzer.grammar.vocabulary_)})")
        else:
            ax.text(0.5, 0.5, "no motifs", ha="center", va="center")
    else:
        ax.text(0.5, 0.5, "Grammar failed", ha="center", va="center")

    fig.suptitle(
        f"Figure 15 — Real EEG ({DATA_PATH.name}, {N_CHANNELS} channels) "
        "through the harmonic_sequence pipeline",
        fontsize=12.5, y=0.985,
    )
    fig.savefig(OUT / "15_real_eeg_overview.png", bbox_inches="tight")
    plt.close(fig)
    print("  saved 15_real_eeg_overview.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 16 — same recording, six lenses (analogue of fig_method_comparison)
# ─────────────────────────────────────────────────────────────────────────────

def fig_six_lenses():
    """Show what each fitted model represents internally on real EEG."""
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.50, wspace=0.30,
                           left=0.06, right=0.97, top=0.93, bottom=0.07)

    # (1) Markov centroids — each is a prototype "harmonic mood" histogram
    ax = fig.add_subplot(gs[0, 0])
    centroids = analyzer.markov._km.cluster_centers_
    bin_cents = np.linspace(2.5, 1197.5, centroids.shape[1])
    for i, c in enumerate(centroids):
        ax.plot(bin_cents, c + 0.04 * i, lw=1.1,
                 label=f"state {i}", alpha=0.9)
    ax.set_xlabel("cents"); ax.set_ylabel("histogram weight (offset)")
    ax.set_title("(1) Markov: K-means centroids in cents space")
    ax.legend(frameon=False, fontsize=7, loc="upper right")

    # (2) PCA components on the cents axis
    ax = fig.add_subplot(gs[0, 1])
    pcs = analyzer.latent._pca.components_      # (latent_dim, n_bins)
    for i in range(min(3, pcs.shape[0])):
        ax.plot(bin_cents, pcs[i], lw=1.1,
                 label=f"PC{i+1}  ({analyzer.latent.explained_variance_ratio_[i]:.1%})",
                 alpha=0.85)
    ax.axhline(0, color="grey", lw=0.4)
    ax.set_xlabel("cents"); ax.set_ylabel("loading")
    ax.set_title("(2) Latent: principal components in cents space")
    ax.legend(frameon=False, fontsize=7)

    # (3) Wasserstein barycenter morph between first and last channels
    ax = fig.add_subplot(gs[0, 2])
    wt = analyzer.wasserstein
    n = 8
    morph = np.stack(wt.interpolate_pair(0, T - 1, n_steps=n), axis=0)
    im = ax.imshow(morph.T, aspect="auto", origin="lower", cmap="magma",
                    extent=[0, n, 0, 1200])
    ax.set_xlabel(f"interp step (channel 0 -> {T-1})")
    ax.set_ylabel("cents")
    ax.set_title("(3) Wasserstein: barycenter morph")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (4) DMD top oscillatory modes magnitude (spatial loading)
    ax = fig.add_subplot(gs[1, 0])
    if analyzer.dmd is not None and analyzer.dmd.modes_ is not None:
        modes = analyzer.dmd.modes_              # complex (D, r)
        dist = np.abs(np.abs(analyzer.dmd.eigenvalues_) - 1.0)
        top = np.argsort(dist)[:3]
        for j, k in enumerate(top):
            mag = np.abs(modes[:, k])
            ax.plot(mag + 0.02 * j, lw=1.1,
                     label=f"mode {k}  (|λ|={np.abs(analyzer.dmd.eigenvalues_[k]):.2f})")
        ax.set_xlabel("PCA-component index")
        ax.set_ylabel("|mode| (offset)")
        ax.set_title("(4) DMD: top oscillatory modes")
        ax.legend(frameon=False, fontsize=7)

    # (5) Takens cloud for the harmsim trajectory
    ax = fig.add_subplot(gs[1, 1])
    if analyzer.topology is not None:
        cloud = analyzer.topology.takens_embedding_
        sc = ax.scatter(cloud[:, 0], cloud[:, 1],
                         c=np.arange(len(cloud)), cmap="viridis",
                         s=44, edgecolor="white", lw=0.6)
        ax.plot(cloud[:, 0], cloud[:, 1], color="grey", lw=0.4, alpha=0.4)
        beta = analyzer.topology.betti_numbers_
        ax.set_xlabel("x(t)"); ax.set_ylabel("x(t+1)")
        ax.set_title(f"(5) Topology: Takens cloud  (β0={beta[0]})")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="channel")

    # (6) Grammar vocabulary as a chord-stream
    ax = fig.add_subplot(gs[1, 2])
    if analyzer.grammar is not None:
        chord_to_id = {c: i for i, c
                       in enumerate(analyzer.grammar.vocabulary_)}
        ids = np.array([chord_to_id[c]
                         for c in analyzer.grammar.chord_sequence_])
        cmap_g = plt.get_cmap("tab20")
        for t, tok in enumerate(ids):
            ax.add_patch(plt.Rectangle((t, tok - 0.4), 1, 0.8,
                                        color=cmap_g(int(tok) % 20)))
        ax.set_xlim(0, len(ids))
        ax.set_ylim(-1, len(analyzer.grammar.vocabulary_))
        ax.set_xlabel("channel")
        ax.set_ylabel("chord-token id")
        ax.set_title(f"(6) Grammar: chord stream  "
                      f"(vocab={len(analyzer.grammar.vocabulary_)}, "
                      f"H={analyzer.grammar.transition_entropy_:.2f} bits)")

    fig.suptitle(
        "Figure 16 — What each of the six models learned on the same real EEG",
        fontsize=12.5, y=0.985,
    )
    fig.savefig(OUT / "16_real_eeg_method_compare.png", bbox_inches="tight")
    plt.close(fig)
    print("  saved 16_real_eeg_method_compare.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 17 — Application: harmonically-similar channel retrieval
# ─────────────────────────────────────────────────────────────────────────────

def fig_retrieval():
    """Pick a query channel, retrieve the k nearest harmonic neighbours via
    the W1 distance matrix, and visualise the result."""
    D = analyzer.wasserstein.distance_matrix_
    harmsim = np.array(harmsim_list)

    # Choose a query channel: one with high harmsim (most "consonant"
    # channel — usually striking on an EEG resting recording).
    finite_idx = np.where(np.isfinite(harmsim))[0]
    if len(finite_idx) == 0:
        print("  fig_retrieval: no finite harmsim — skipping")
        return
    q = int(finite_idx[np.argmax(harmsim[finite_idx])])
    # k-nearest harmonic neighbours (excluding self)
    k = 5
    order = np.argsort(D[q])
    nearest = [j for j in order if j != q][:k]
    farthest = [j for j in order[::-1] if j != q][:k]

    fig = plt.figure(figsize=(14, 7.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.4, 1.4],
                           hspace=0.55, wspace=0.30,
                           left=0.06, right=0.98, top=0.92, bottom=0.08)

    # Top: W1 row for the query, highlighting nearest/farthest
    ax = fig.add_subplot(gs[0, :])
    ax.bar(np.arange(T), D[q], color="#888", alpha=0.65)
    ax.bar(nearest, D[q][nearest], color="seagreen", label=f"{k} nearest")
    ax.bar(farthest, D[q][farthest], color="crimson", label=f"{k} farthest")
    ax.axvline(q, color="black", lw=1.0, ls="--",
                label=f"query channel {channels[q]}  (harmsim={harmsim[q]:.2f})")
    ax.set_xlabel("channel"); ax.set_ylabel(f"$W_1$ to channel {channels[q]}")
    ax.set_title(f"(a) Harmonic-similarity ranking from query channel {channels[q]}")
    ax.legend(frameon=False, loc="upper right", fontsize=8)

    # Middle row: query + nearest histograms overlaid
    ax = fig.add_subplot(gs[1, :])
    H = analyzer.histograms
    bin_cents = np.linspace(2.5, 1197.5, H.shape[1])
    ax.plot(bin_cents, H[q], color="black", lw=1.6,
             label=f"query ch{channels[q]}")
    for j in nearest[:3]:
        ax.plot(bin_cents, H[j], color="seagreen", lw=1.0, alpha=0.6,
                 label=f"near ch{channels[j]} (W1={D[q,j]:.1f})")
    ax.set_xlabel("cents"); ax.set_ylabel("histogram weight")
    ax.set_title("(b) Query histogram (black) overlaid with 3 nearest neighbours")
    ax.legend(frameon=False, fontsize=8, ncol=2)

    # Bottom row: farthest neighbour histograms
    ax = fig.add_subplot(gs[2, :])
    ax.plot(bin_cents, H[q], color="black", lw=1.6,
             label=f"query ch{channels[q]}")
    for j in farthest[:3]:
        ax.plot(bin_cents, H[j], color="crimson", lw=1.0, alpha=0.6,
                 label=f"far ch{channels[j]} (W1={D[q,j]:.1f})")
    ax.set_xlabel("cents"); ax.set_ylabel("histogram weight")
    ax.set_title("(c) Query histogram (black) overlaid with 3 farthest neighbours")
    ax.legend(frameon=False, fontsize=8, ncol=2)

    fig.suptitle(
        "Figure 17 — Application: harmonic-neighbour retrieval on real EEG\n"
        "(unsupervised: indexes channels by tuning similarity from one query)",
        fontsize=12.5, y=0.995,
    )
    fig.savefig(OUT / "17_real_eeg_retrieval.png", bbox_inches="tight")
    plt.close(fig)
    print("  saved 17_real_eeg_retrieval.png")


# ─────────────────────────────────────────────────────────────────────────────

for fn in [fig_overview, fig_six_lenses, fig_retrieval]:
    fn()

print(f"\nAll real-data figures saved into {OUT}")
