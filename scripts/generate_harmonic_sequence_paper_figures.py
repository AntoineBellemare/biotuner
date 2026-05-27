"""
generate_harmonic_sequence_paper_figures.py
===========================================
Produce the figures used in the ``harmonic_sequence`` use-case paper.

Builds a controlled 36-frame ratio sequence cycling through three harmonic
regimes (just-major, just-minor, harmonic-7) with intra-regime drift, then
plots one or two illustrative figures per approach into
``docs/papers/figures/``.

Run:
    python scripts/generate_harmonic_sequence_paper_figures.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biotuner.harmonic_sequence import (
    HarmonicSequenceAnalyzer,
    encode_histograms,
    histogram_to_ratios,
)

OUT = ROOT / "docs" / "papers" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 140,
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Three harmonic regimes
JUST_MAJOR = [1.0, 9 / 8, 5 / 4, 4 / 3, 3 / 2, 5 / 3, 15 / 8]
JUST_MINOR = [1.0, 9 / 8, 6 / 5, 4 / 3, 3 / 2, 8 / 5, 9 / 5]
HARM_7 = [1.0, 9 / 8, 5 / 4, 11 / 8, 3 / 2, 7 / 4, 15 / 8]
REGIMES = [JUST_MAJOR, JUST_MINOR, HARM_7]
REGIME_NAMES = ["major", "minor", "harm7"]
REGIME_COLOURS = ["#4C72B0", "#DD8452", "#55A868"]


def build_sequence(n_cycles: int = 4, frames_per_regime: int = 3, seed: int = 0):
    """Build T = n_cycles * 3 * frames_per_regime alternating ratio sequence."""
    rng = np.random.default_rng(seed)
    ratios_list = []
    regime_index = []
    # Two scales of variation:
    #   - tiny intra-regime drift (~0.5 cents, keeps grammar tokens stable)
    #   - regime-coherent micro-variation (~1.5 cents on every k>0 frame)
    for c in range(n_cycles):
        for r_idx, base in enumerate(REGIMES):
            for k in range(frames_per_regime):
                jitter = 1.0 + 0.0003 * rng.standard_normal(len(base))
                ratios_list.append([float(x * j) for x, j in zip(base, jitter)])
                regime_index.append(r_idx)
    return ratios_list, np.array(regime_index)


# ---------------------------------------------------------------------------
# Build the shared analyzer
# ---------------------------------------------------------------------------
print(f"Output directory: {OUT}")
ratios_list, regime_idx = build_sequence(n_cycles=4, frames_per_regime=3)
T = len(ratios_list)
print(f"Sequence length T = {T} frames over {T // 9} cycles of 3 regimes")

analyzer = HarmonicSequenceAnalyzer.from_ratios_list(ratios_list, n_hist_bins=240)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    analyzer.fit_all(n_states=3, latent_dim=3, n_gram=2,
                     topology_scalar="harmsim")
print(analyzer.summary())

H = analyzer.histograms                # (T, 240)


# ---------------------------------------------------------------------------
# Fig 0  — Pipeline schematic + the raw histogram-trajectory heatmap
# ---------------------------------------------------------------------------
def fig_overview():
    fig, ax = plt.subplots(figsize=(8.5, 2.4))
    im = ax.imshow(H.T, aspect="auto", origin="lower", cmap="magma",
                   extent=[0, T, 0, 1200])
    ax.set_xlabel("Frame (window index)")
    ax.set_ylabel("Cents (0 = unison, 1200 = octave)")
    ax.set_title("Encoded cents histogram across time  —  the input every model consumes")
    # Mark regime boundaries with thin lines
    for t in range(1, T):
        if regime_idx[t] != regime_idx[t - 1]:
            ax.axvline(t, color="white", lw=0.4, alpha=0.5)
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label("histogram weight")
    fig.tight_layout()
    fig.savefig(OUT / "00_input_histograms.png")
    plt.close(fig)
    print("  saved 00_input_histograms.png")


# ---------------------------------------------------------------------------
# Fig 1  — Markov: transition matrix + state timeline
# ---------------------------------------------------------------------------
def fig_markov():
    mk = analyzer.markov
    K = mk.n_states
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4),
                              gridspec_kw={"width_ratios": [1.2, 2.2]})

    # (a) transition matrix heatmap
    ax = axes[0]
    im = ax.imshow(mk.transition_matrix_, cmap="rocket_r", vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{mk.transition_matrix_[i, j]:.2f}",
                     ha="center", va="center", fontsize=8,
                     color="white" if mk.transition_matrix_[i, j] > 0.5 else "black")
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xlabel("next state"); ax.set_ylabel("current state")
    ax.set_title(f"(a) Transition matrix  (K={K})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (b) state timeline (top = Markov labels, bottom = ground-truth regime)
    ax = axes[1]
    cmap_states = plt.get_cmap("tab10")
    for t in range(T):
        ax.add_patch(plt.Rectangle((t, 0.55), 1, 0.4,
                                    color=cmap_states(mk.state_labels_[t] % 10)))
        ax.add_patch(plt.Rectangle((t, 0.05), 1, 0.4,
                                    color=REGIME_COLOURS[regime_idx[t]]))
    ax.set_xlim(0, T); ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.75]); ax.set_yticklabels(["regime (ground truth)",
                                                       "Markov state"])
    ax.set_xlabel("Frame")
    ax.set_title("(b) Discovered state timeline vs ground-truth regime")
    ax.set_xticks(np.arange(0, T + 1, 9))
    # Legend strip for regimes
    for i, name in enumerate(REGIME_NAMES):
        ax.scatter([], [], color=REGIME_COLOURS[i], s=60, label=name)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, -0.18), ncol=3,
              frameon=False, fontsize=8)

    fig.suptitle("Approach 1 — HarmonicMarkov", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "01_markov.png", bbox_inches="tight")
    plt.close(fig)
    print("  saved 01_markov.png")


# ---------------------------------------------------------------------------
# Fig 2  — Wasserstein: distance matrix + flux + MDS embedding
# ---------------------------------------------------------------------------
def fig_wasserstein():
    wt = analyzer.wasserstein
    D = wt.distance_matrix_
    flux = wt.flux_

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))

    # (a) pairwise W1 distance
    ax = axes[0]
    im = ax.imshow(D, cmap="viridis")
    ax.set_xlabel("frame j"); ax.set_ylabel("frame i")
    ax.set_title("(a) Pairwise $W_1$ distance")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (b) flux over time
    ax = axes[1]
    ax.plot(np.arange(len(flux)), flux, color="black", lw=1.4)
    # Shade regime boundaries
    for t in range(1, T):
        if regime_idx[t] != regime_idx[t - 1]:
            ax.axvline(t - 0.5, color="crimson", alpha=0.5, lw=0.8, ls="--")
    ax.set_xlabel("Frame transition (t → t+1)")
    ax.set_ylabel("$W_1(h_t, h_{t+1})$  (cents-units)")
    ax.set_title("(b) Harmonic flux  (dashes = regime change)")

    # (c) MDS embedding
    ax = axes[2]
    Z = wt.embed(n_components=2, method="mds")
    for r_idx, name in enumerate(REGIME_NAMES):
        mask = regime_idx == r_idx
        ax.scatter(Z[mask, 0], Z[mask, 1], color=REGIME_COLOURS[r_idx],
                    s=42, alpha=0.8, label=name, edgecolor="white", lw=0.8)
    # Connect consecutive frames with thin lines
    ax.plot(Z[:, 0], Z[:, 1], color="grey", lw=0.4, alpha=0.5, zorder=0)
    ax.set_xlabel("MDS-1"); ax.set_ylabel("MDS-2")
    ax.set_title("(c) MDS embedding")
    ax.legend(frameon=False, fontsize=8, loc="best")

    fig.suptitle("Approach 2 — WassersteinTrajectory", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "02_wasserstein.png", bbox_inches="tight")
    plt.close(fig)
    print("  saved 02_wasserstein.png")


# ---------------------------------------------------------------------------
# Fig 3  — DMD: eigenvalues on complex plane + reconstruction
# ---------------------------------------------------------------------------
def fig_dmd():
    # Rebuild analyzer with DMD on histogram-PCA so we have meaningful spectrum
    a2 = HarmonicSequenceAnalyzer.from_ratios_list(ratios_list, n_hist_bins=240)
    a2.fit_dmd(use_histograms=True)
    dmd = a2.dmd
    eigs = dmd.eigenvalues_

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    # (a) eigenvalues on the complex plane
    ax = axes[0]
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="grey", lw=0.8, ls="--")
    ax.add_patch(Circle((0, 0), 1, fill=False, color="grey", lw=0.8))
    ax.axhline(0, color="grey", lw=0.4); ax.axvline(0, color="grey", lw=0.4)
    osc, idx = dmd.oscillatory_modes(threshold=0.1)
    is_osc = np.zeros(len(eigs), dtype=bool); is_osc[idx] = True
    ax.scatter(eigs[~is_osc].real, eigs[~is_osc].imag,
                s=70, color="#888", alpha=0.8, label="decaying")
    ax.scatter(eigs[is_osc].real, eigs[is_osc].imag,
                s=110, color="crimson", marker="*", label="oscillatory")
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.set_xlabel(r"Re($\lambda$)"); ax.set_ylabel(r"Im($\lambda$)")
    ax.set_title("(a) DMD eigenvalues  (unit circle = pure oscillation)")
    ax.legend(frameon=False, fontsize=8)

    # (b) growth-rate vs frequency
    ax = axes[1]
    ax.scatter(dmd.frequencies_, dmd.growth_rates_,
                s=70, c=np.abs(eigs), cmap="plasma")
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.set_xlabel("frequency  Im[log $\\lambda$]")
    ax.set_ylabel("growth rate  Re[log $\\lambda$]")
    ax.set_title("(b) Growth vs. oscillation frequency per mode")

    fig.suptitle("Approach 3 — HarmonicDMD", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "03_dmd.png", bbox_inches="tight")
    plt.close(fig)
    print("  saved 03_dmd.png")


# ---------------------------------------------------------------------------
# Fig 4  — Latent: trajectory in 3D PCA space + scree
# ---------------------------------------------------------------------------
def fig_latent():
    ls = analyzer.latent
    Z = ls.trajectory()
    evr = ls.explained_variance_ratio_

    fig = plt.figure(figsize=(10.5, 3.6))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    for r_idx, name in enumerate(REGIME_NAMES):
        mask = regime_idx == r_idx
        ax1.scatter(Z[mask, 0], Z[mask, 1], Z[mask, 2],
                     color=REGIME_COLOURS[r_idx], s=42, alpha=0.9, label=name,
                     edgecolor="white", lw=0.6)
    ax1.plot(Z[:, 0], Z[:, 1], Z[:, 2], color="grey", lw=0.5, alpha=0.4)
    ax1.set_xlabel("PC1"); ax1.set_ylabel("PC2"); ax1.set_zlabel("PC3")
    ax1.set_title("(a) Trajectory in 3-D latent space")
    ax1.legend(frameon=False, fontsize=8)

    ax2 = fig.add_subplot(1, 2, 2)
    cum = np.cumsum(evr)
    ax2.bar(np.arange(1, len(evr) + 1), evr, color="#4C72B0", alpha=0.7,
             label="per-dim")
    ax2.plot(np.arange(1, len(evr) + 1), cum, "o-", color="crimson",
              label="cumulative")
    ax2.set_xlabel("latent dim"); ax2.set_ylabel("explained variance")
    ax2.set_title(f"(b) Scree  (3 dims explain {cum[-1]:.1%})")
    ax2.legend(frameon=False, fontsize=8)
    ax2.set_xticks(np.arange(1, len(evr) + 1))

    fig.suptitle("Approach 4 — HarmonicLatentSpace", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "04_latent.png", bbox_inches="tight")
    plt.close(fig)
    print("  saved 04_latent.png")


# ---------------------------------------------------------------------------
# Fig 5  — Topology: Takens embedding + persistence diagram
# ---------------------------------------------------------------------------
def fig_topology():
    top = analyzer.topology
    cloud = top.takens_embedding_     # (N, d=3)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))

    # (a) Takens cloud — colour by frame number
    ax = axes[0]
    sc = ax.scatter(cloud[:, 0], cloud[:, 1],
                     c=np.arange(len(cloud)), cmap="viridis", s=44,
                     edgecolor="white", lw=0.6)
    ax.plot(cloud[:, 0], cloud[:, 1], color="grey", lw=0.4, alpha=0.4)
    ax.set_xlabel("x(t)"); ax.set_ylabel(rf"x(t+{top.delay})")
    ax.set_title(f"(a) Takens delay-embedding  (d={top.embedding_dim}, "
                  rf"$\tau={top.delay}$)")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="frame")

    # (b) persistence diagram
    ax = axes[1]
    diagrams = top.persistence_diagram_
    max_finite = 0.0
    for dim, dgm in enumerate(diagrams):
        finite = dgm[np.isfinite(dgm[:, 1])]
        if len(finite):
            max_finite = max(max_finite, finite[:, 1].max())
            label = f"H{dim}"
            ax.scatter(finite[:, 0], finite[:, 1], s=40,
                        label=label, alpha=0.8)
    lims = [0, max_finite * 1.1 if max_finite > 0 else 1]
    ax.plot(lims, lims, color="grey", lw=0.6, ls="--")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("birth"); ax.set_ylabel("death")
    title = "(b) Persistence diagram"
    if len(diagrams) == 1:
        title += "  (H0 only — install ripser for H1)"
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Approach 5 — HarmonicTopology  (TDA)", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "05_topology.png", bbox_inches="tight")
    plt.close(fig)
    print("  saved 05_topology.png")


# ---------------------------------------------------------------------------
# Fig 6  — Grammar: token sequence + top motifs
# ---------------------------------------------------------------------------
def fig_grammar():
    gr = analyzer.grammar
    seq = gr.chord_sequence_
    vocab = gr.vocabulary_
    chord_to_id = {c: i for i, c in enumerate(vocab)}
    ids = np.array([chord_to_id[c] for c in seq])

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6),
                              gridspec_kw={"width_ratios": [2, 1]})

    # (a) chord-token stream as stem plot
    ax = axes[0]
    cmap_tok = plt.get_cmap("tab20")
    for t, tok in enumerate(ids):
        ax.add_patch(plt.Rectangle((t, tok - 0.4), 1, 0.8,
                                    color=cmap_tok(tok % 20)))
    ax.set_xlim(0, len(ids)); ax.set_ylim(-1, len(vocab))
    ax.set_xlabel("Frame")
    ax.set_ylabel("Chord-token id")
    ax.set_yticks(np.arange(len(vocab)))
    short = []
    for c in vocab:
        if len(c) == 0:
            short.append("∅")
        else:
            # Shorten frozenset labels for axis ticks
            names = sorted(c, key=len)[:1]
            label = next(iter(c))
            short.append((label[:18] + "…") if len(label) > 18 else label)
    ax.set_yticklabels(short, fontsize=7)
    ax.set_title(f"(a) Symbolic chord stream  (vocab={len(vocab)})")

    # (b) top motifs as bar chart
    ax = axes[1]
    motifs = gr.top_motifs(min_length=2, max_length=3, top_k=8)
    if motifs:
        labels = [f"len{len(m)}: " + "→".join(str(chord_to_id[t]) for t in m)
                  for m, _ in motifs]
        counts = [c for _, c in motifs]
        ax.barh(range(len(labels)), counts, color="#4C72B0")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("count")
        ax.set_title("(b) Top motifs (chord-id sequences)")
    else:
        ax.text(0.5, 0.5, "no motifs", ha="center", va="center")

    fig.suptitle(f"Approach 6 — HarmonicGrammar  "
                  f"(H_trans = {gr.transition_entropy_:.2f} bits)",
                  fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "06_grammar.png", bbox_inches="tight")
    plt.close(fig)
    print("  saved 06_grammar.png")


# ---------------------------------------------------------------------------
# Fig 7  — Render bridge: Wasserstein-interpolated glissando
# ---------------------------------------------------------------------------
def fig_bridge():
    n = 10
    glide = analyzer.get_histograms(source="wasserstein_interp",
                                     t1=0, t2=15, n_steps=n)
    fig, ax = plt.subplots(figsize=(9, 3.0))
    im = ax.imshow(glide.T, aspect="auto", origin="lower", cmap="magma",
                    extent=[0, n, 0, 1200])
    ax.set_xlabel("interpolation step  (0 → major frame, 9 → minor frame)")
    ax.set_ylabel("cents")
    ax.set_title("Render bridge — Wasserstein glissando between two frames")
    fig.colorbar(im, ax=ax, label="weight", pad=0.01)
    fig.tight_layout()
    fig.savefig(OUT / "07_bridge_glissando.png")
    plt.close(fig)
    print("  saved 07_bridge_glissando.png")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------
for fn in [fig_overview, fig_markov, fig_wasserstein, fig_dmd,
           fig_latent, fig_topology, fig_grammar, fig_bridge]:
    fn()

print("\nAll figures generated.")
