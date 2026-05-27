"""
generate_harmonic_sequence_classical_apps.py
============================================
For each of the six approaches in harmonic_sequence, implement one
"classical application" inspired by the canonical use of the underlying
mathematical tool in another field, then adapt it to biotuner:

  22. HarmonicMarkov  ->  Algorithmic composition from EEG
        (Hiller-Isaacson n-state composition + render bridge)
  23. Wasserstein     ->  Inter-group OT geodesic
        (Schiebinger-style developmental trajectory between two
         aggregate condition distributions)
  24. HarmonicDMD     ->  Short-term harmonic forecasting
        (Schmid/Brunton style train-then-forecast, with held-out RMSE)
  25. LatentSpace     ->  Word-embedding-style group separation
        (shared PCA + inter-group direction vector)
  26. Topology        ->  Chaotic vs structured regime detection
        (Takens fingerprint vs shuffle control, cardiac-arrhythmia style)
  27. Grammar         ->  Stylometric authorship attribution
        (n-gram log-likelihood classifier across conditions,
         Mosteller-Wallace style)

All six experiments reuse caches from the previous scripts:
  - _real_eeg_bt_cache.npz       (Part IV, 60-channel EEG, ratios only)
  - _tier1_A_cache.npz           (Part V.A, 234 sliding windows)
  - _tier1_B_cache.npz           (Part V.B, 15 aud + 15 vis MNE epochs)

Run:
    python scripts/generate_harmonic_sequence_classical_apps.py
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biotuner.harmonic_sequence import (
    HarmonicSequenceAnalyzer,
    histograms_to_midi,
    histogram_to_ratios,
)

CACHE_DIR = ROOT / "docs" / "papers"
OUT = ROOT / "docs" / "papers" / "figures"
MIDI_OUT = ROOT / "docs" / "papers" / "audio"
OUT.mkdir(parents=True, exist_ok=True)
MIDI_OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 140, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})


# ─────────────────────────────────────────────────────────────────────────────
# Cache loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_partIV():
    npz = np.load(CACHE_DIR / "_real_eeg_bt_cache.npz", allow_pickle=True)
    return list(npz["ratios_list"])


def load_A():
    npz = np.load(CACHE_DIR / "_tier1_A_cache.npz", allow_pickle=True)
    return (list(npz["ratios_list"]),
            list(npz["peaks_list"]),
            list(npz["metrics_list"]))


def load_B():
    npz = np.load(CACHE_DIR / "_tier1_B_cache.npz", allow_pickle=True)
    return (list(npz["ratios_aud"]),
            list(npz["ratios_vis"]),
            list(npz["metrics_aud"]),
            list(npz["metrics_vis"]))


# ═════════════════════════════════════════════════════════════════════════════
# App 22 — Markov: algorithmic composition from EEG
# ═════════════════════════════════════════════════════════════════════════════

def app_markov_composition():
    print("\n[22] Markov: algorithmic composition from EEG (Hiller-Isaacson style)",
          flush=True)
    ratios_list = load_partIV()
    az = HarmonicSequenceAnalyzer.from_ratios_list(ratios_list, n_hist_bins=240)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        az.fit_markov(n_states="auto", auto_k_range=(2, 6))
        az.fit_wasserstein()

    K = az.markov.n_states
    print(f"    K={K}, H_trans={az.markov.transition_entropy_:.2f} bits")

    # Sample a 48-step chord sequence from the trained model
    rng = np.random.default_rng(7)
    T_sample = 48
    sampled = az.get_histograms(source="markov_sample", n_steps=T_sample)
    observed = az.histograms

    # Render the sampled trajectory as MIDI
    midi_path = MIDI_OUT / "22_markov_composition"
    try:
        cwd = os.getcwd()
        os.chdir(MIDI_OUT)
        try:
            histograms_to_midi(
                sampled, filename="22_markov_composition",
                base_freq=220.0, duration_beats=0.4, n_peaks=4,
                microtonal=True, velocity=72,
            )
            wrote = (MIDI_OUT / "22_markov_composition.mid").exists()
        finally:
            os.chdir(cwd)
    except Exception as exc:
        print(f"    MIDI render failed: {exc}")
        wrote = False

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1.2, 1.0],
                           hspace=0.55, wspace=0.30,
                           left=0.06, right=0.97, top=0.93, bottom=0.07)

    # (a) Observed histogram trajectory
    ax = fig.add_subplot(gs[0, :])
    vmax = float(np.percentile(observed.ravel(), 99))
    im = ax.imshow(observed.T, aspect="auto", origin="lower", cmap="magma",
                    vmin=0, vmax=max(vmax, 1e-3),
                    extent=[0, observed.shape[0], 0, 1200])
    ax.set_xlabel("channel (training data)"); ax.set_ylabel("cents")
    ax.set_title("(a) Observed harmonic trajectory  "
                  f"(T={observed.shape[0]} channels) — input to Markov fit")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)

    # (b) Sampled trajectory
    ax = fig.add_subplot(gs[1, :])
    im = ax.imshow(sampled.T, aspect="auto", origin="lower", cmap="magma",
                    vmin=0, vmax=max(vmax, 1e-3),
                    extent=[0, T_sample, 0, 1200])
    ax.set_xlabel("composition step"); ax.set_ylabel("cents")
    midi_note = (" — exported to " + str(midi_path.name) + ".mid" if wrote
                 else "")
    ax.set_title(f"(b) Algorithmically composed trajectory "
                  f"(Markov-sample of length {T_sample})" + midi_note)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)

    # (c) State usage histograms (observed vs sampled)
    ax = fig.add_subplot(gs[2, 0])
    obs_lab = az.markov.state_labels_
    smp_lab = np.array([
        int(np.argmin(np.linalg.norm(
            az.markov._km.cluster_centers_ - row, axis=1)))
        for row in sampled
    ])
    obs_counts = np.bincount(obs_lab, minlength=K) / max(len(obs_lab), 1)
    smp_counts = np.bincount(smp_lab, minlength=K) / max(len(smp_lab), 1)
    x = np.arange(K)
    width = 0.4
    ax.bar(x - width/2, obs_counts, width, color="#4C72B0", label="observed")
    ax.bar(x + width/2, smp_counts, width, color="#DD8452", label="sampled")
    ax.set_xticks(x); ax.set_xticklabels([f"S{i}" for i in range(K)])
    ax.set_ylabel("fraction of frames")
    ax.set_title("(c) State usage: observed vs composed")
    ax.legend(frameon=False, fontsize=8)

    # (d) Transition matrix heatmap
    ax = fig.add_subplot(gs[2, 1])
    T_mat = az.markov.transition_matrix_
    im = ax.imshow(T_mat, cmap="rocket_r", vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{T_mat[i, j]:.2f}", ha="center", va="center",
                     fontsize=7,
                     color="white" if T_mat[i, j] > 0.5 else "black")
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels([f"S{i}" for i in range(K)])
    ax.set_yticklabels([f"S{i}" for i in range(K)])
    ax.set_title(f"(d) Learned transition matrix\n"
                  f"H={az.markov.transition_entropy_:.2f} bits")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # (e) Cluster centroids as cents-histogram lines
    ax = fig.add_subplot(gs[2, 2])
    bin_cents = np.linspace(2.5, 1197.5, 240)
    cmap_st = plt.get_cmap("tab10")
    for i, c in enumerate(az.markov._km.cluster_centers_):
        ax.plot(bin_cents, c + 0.025 * i, lw=1.0,
                 color=cmap_st(i % 10), label=f"S{i}")
    ax.set_xlabel("cents"); ax.set_ylabel("weight (offset)")
    ax.set_title("(e) Cluster centroids = prototype tunings")
    ax.legend(frameon=False, fontsize=7, ncol=2)

    fig.suptitle(
        "Figure 22 — Classical application: algorithmic composition "
        "from EEG (Hiller-Isaacson 1957, adapted)",
        fontsize=12.5, y=0.985)
    fig.savefig(OUT / "22_app_markov_composition.png", bbox_inches="tight")
    plt.close(fig)
    print("    saved 22_app_markov_composition.png "
          f"(MIDI={'yes' if wrote else 'no'})")


# ═════════════════════════════════════════════════════════════════════════════
# App 23 — Wasserstein: inter-group OT geodesic
# ═════════════════════════════════════════════════════════════════════════════

def app_wasserstein_geodesic():
    print("\n[23] Wasserstein: inter-group OT geodesic between aud and vis",
          flush=True)
    ratios_aud, ratios_vis, _, _ = load_B()
    az_aud = HarmonicSequenceAnalyzer.from_ratios_list(ratios_aud,
                                                       n_hist_bins=240)
    az_vis = HarmonicSequenceAnalyzer.from_ratios_list(ratios_vis,
                                                       n_hist_bins=240)
    H_aud = az_aud.histograms.mean(axis=0)
    H_vis = az_vis.histograms.mean(axis=0)
    # Normalise (mean of normalised hist is renormalised)
    H_aud = H_aud / max(H_aud.sum(), 1e-9)
    H_vis = H_vis / max(H_vis.sum(), 1e-9)

    # OT-geodesic between H_aud and H_vis via quantile-space interpolation
    from biotuner.harmonic_sequence import WassersteinTrajectory
    wt = WassersteinTrajectory(n_bins=240)
    alphas = np.linspace(0, 1, 9)
    geodesic = np.stack([wt.barycenter(H_aud, H_vis, a) for a in alphas])
    print(f"    geodesic shape: {geodesic.shape}, "
          f"steps {alphas[0]:.2f} -> {alphas[-1]:.2f}")

    # Render geodesic as MIDI (the brain-state morph audio)
    try:
        cwd = os.getcwd()
        os.chdir(MIDI_OUT)
        try:
            histograms_to_midi(
                geodesic, filename="23_wasserstein_geodesic",
                base_freq=220.0, duration_beats=0.8, n_peaks=4,
                microtonal=True, velocity=72,
            )
            wrote = (MIDI_OUT / "23_wasserstein_geodesic.mid").exists()
        finally:
            os.chdir(cwd)
    except Exception as exc:
        print(f"    MIDI render failed: {exc}")
        wrote = False

    # W1 distance between centroids
    from scipy.stats import wasserstein_distance
    bins = np.arange(240, dtype=float)
    W_aud_vis = wasserstein_distance(bins, bins, H_aud, H_vis)
    print(f"    W1(aud_mean, vis_mean) = {W_aud_vis:.2f} cents-bin units")

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.5, 1.2],
                           hspace=0.55, wspace=0.30,
                           left=0.06, right=0.97, top=0.93, bottom=0.07)

    bin_cents = np.linspace(2.5, 1197.5, 240)

    # (a) Two group means
    ax = fig.add_subplot(gs[0, :])
    ax.plot(bin_cents, H_aud, color="#4C72B0", lw=1.4,
             label="auditory mean")
    ax.plot(bin_cents, H_vis, color="#DD8452", lw=1.4,
             label="visual mean")
    ax.fill_between(bin_cents, 0, H_aud, color="#4C72B0", alpha=0.15)
    ax.fill_between(bin_cents, 0, H_vis, color="#DD8452", alpha=0.15)
    ax.set_xlabel("cents"); ax.set_ylabel("normalised weight")
    ax.set_title(f"(a) Group-mean cents histograms  "
                  f"($W_1 = {W_aud_vis:.2f}$ between them)")
    ax.legend(frameon=False, fontsize=9)

    # (b) Geodesic as a heatmap (interp step on x, cents on y)
    ax = fig.add_subplot(gs[1, :])
    vmax = float(np.percentile(geodesic.ravel(), 99))
    im = ax.imshow(geodesic.T, aspect="auto", origin="lower", cmap="magma",
                    vmin=0, vmax=max(vmax, 1e-3),
                    extent=[0, len(alphas) - 1, 0, 1200])
    ax.set_xlabel(r"interpolation step  ($\alpha=0$: aud  $\to$  $\alpha=1$: vis)")
    ax.set_ylabel("cents")
    midi_note = (" -> exported as " + "23_wasserstein_geodesic.mid"
                 if wrote else "")
    ax.set_title(f"(b) $W_1$-geodesic between the two group distributions"
                  + midi_note)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="weight")
    for i in range(1, len(alphas) - 1):
        ax.axvline(i, color="white", lw=0.3, alpha=0.4)

    # (c) Three slices: aud, midpoint, vis
    ax = fig.add_subplot(gs[2, 0])
    ax.plot(bin_cents, geodesic[0], color="#4C72B0", lw=1.3,
             label=r"$\alpha=0$ (aud)")
    ax.plot(bin_cents, geodesic[len(alphas) // 2],
             color="#888", lw=1.3, label=r"$\alpha=0.5$ (midpoint)")
    ax.plot(bin_cents, geodesic[-1], color="#DD8452", lw=1.3,
             label=r"$\alpha=1$ (vis)")
    ax.set_xlabel("cents"); ax.set_ylabel("weight")
    ax.set_title("(c) Three geodesic slices")
    ax.legend(frameon=False, fontsize=8)

    # (d) Per-step W1 distance from aud / from vis
    ax = fig.add_subplot(gs[2, 1])
    d_from_aud = [wasserstein_distance(bins, bins, g, H_aud)
                  for g in geodesic]
    d_from_vis = [wasserstein_distance(bins, bins, g, H_vis)
                  for g in geodesic]
    ax.plot(alphas, d_from_aud, "o-", color="#4C72B0", label="W1 to aud")
    ax.plot(alphas, d_from_vis, "s-", color="#DD8452", label="W1 to vis")
    ax.set_xlabel(r"$\alpha$"); ax.set_ylabel("W1")
    ax.set_title("(d) Linear interpolation in W1 distance")
    ax.legend(frameon=False, fontsize=8)

    # (e) Decoded ratios at each step (text)
    ax = fig.add_subplot(gs[2, 2])
    ax.axis("off")
    text_rows = []
    for i, a in enumerate(alphas):
        r = histogram_to_ratios(geodesic[i], n_peaks=3,
                                include_unison=False, include_octave=False)
        rstr = ", ".join(f"{x:.3f}" for x in r) if r else "—"
        text_rows.append(f"α={a:.2f}: {rstr}")
    ax.text(0.0, 1.0, "\n".join(text_rows), fontsize=8.5,
             family="monospace", va="top")
    ax.set_title("(e) Top-3 ratios at each geodesic step", fontsize=10)

    fig.suptitle("Figure 23 — Classical application: inter-group OT "
                  "geodesic (developmental-trajectory style)",
                  fontsize=12.5, y=0.985)
    fig.savefig(OUT / "23_app_wasserstein_geodesic.png", bbox_inches="tight")
    plt.close(fig)
    print("    saved 23_app_wasserstein_geodesic.png "
          f"(MIDI={'yes' if wrote else 'no'})")


# ═════════════════════════════════════════════════════════════════════════════
# App 24 — DMD: short-term harmonic forecasting
# ═════════════════════════════════════════════════════════════════════════════

def app_dmd_forecasting():
    print("\n[24] DMD: short-term harmonic forecasting on Tier 1 A (234 windows)",
          flush=True)
    ratios_list, _, _ = load_A()
    n_total = len(ratios_list)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train
    print(f"    train: {n_train}, test: {n_test}")

    # Fit cents histograms then PCA-reduce, then DMD on PCA
    az_train = HarmonicSequenceAnalyzer.from_ratios_list(
        ratios_list[:n_train], n_hist_bins=240,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        az_train.fit_dmd(use_histograms=True)
        az_train.fit_latent(latent_dim=5)

    # PCA basis from train, project both train and test histograms
    from sklearn.decomposition import PCA
    H_full = HarmonicSequenceAnalyzer.from_ratios_list(
        ratios_list, n_hist_bins=240,
    ).histograms
    pca = PCA(n_components=10).fit(H_full[:n_train])
    Z_train = pca.transform(H_full[:n_train])
    Z_test = pca.transform(H_full[n_train:])

    # DMD on Z_train, forecast n_test steps from end of train
    from biotuner.harmonic_sequence import HarmonicDMD
    dmd = HarmonicDMD(rank=8, center=True).fit(Z_train)
    Z_pred = dmd.reconstruct(n_steps=n_test)
    print(f"    Z_pred shape {Z_pred.shape}, Z_test shape {Z_test.shape}")

    # RMSE per PC + global RMSE
    err = Z_test - Z_pred
    rmse_per_pc = np.sqrt(np.mean(err ** 2, axis=0))
    rmse_global = float(np.sqrt(np.mean(err ** 2)))
    # Compare to naive last-value forecast
    naive = np.tile(Z_train[-1], (n_test, 1))
    rmse_naive = float(np.sqrt(np.mean((Z_test - naive) ** 2)))
    # And to a "mean-of-train" forecast
    rmse_mean = float(np.sqrt(np.mean(
        (Z_test - Z_train.mean(axis=0)) ** 2)))
    print(f"    RMSE: DMD={rmse_global:.4f}  naive_last={rmse_naive:.4f}  "
          f"naive_mean={rmse_mean:.4f}")

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1.2, 1.0],
                           hspace=0.55, wspace=0.30,
                           left=0.06, right=0.97, top=0.93, bottom=0.07)

    # (a) PC1 actual vs forecast over time
    ax = fig.add_subplot(gs[0, :])
    ax.plot(np.arange(n_train), Z_train[:, 0],
             color="black", lw=1.0, label="train")
    ax.plot(np.arange(n_train, n_total), Z_test[:, 0],
             color="#4C72B0", lw=1.4, label="test (actual)")
    ax.plot(np.arange(n_train, n_total), Z_pred[:, 0],
             color="crimson", lw=1.4, ls="--", label="DMD forecast")
    ax.axvline(n_train - 0.5, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("window"); ax.set_ylabel("PC1 coordinate")
    ax.set_title("(a) PC1 trajectory: train -> forecast vs actual")
    ax.legend(frameon=False, fontsize=8)

    # (b) PC2
    ax = fig.add_subplot(gs[1, :])
    ax.plot(np.arange(n_train), Z_train[:, 1],
             color="black", lw=1.0, label="train")
    ax.plot(np.arange(n_train, n_total), Z_test[:, 1],
             color="#4C72B0", lw=1.4, label="test")
    ax.plot(np.arange(n_train, n_total), Z_pred[:, 1],
             color="crimson", lw=1.4, ls="--", label="DMD forecast")
    ax.axvline(n_train - 0.5, color="grey", lw=0.6, ls=":")
    ax.set_xlabel("window"); ax.set_ylabel("PC2 coordinate")
    ax.set_title("(b) PC2 trajectory")
    ax.legend(frameon=False, fontsize=8)

    # (c) RMSE bar chart
    ax = fig.add_subplot(gs[2, 0])
    names = ["DMD", "naive\nlast-value", "naive\ntrain-mean"]
    vals = [rmse_global, rmse_naive, rmse_mean]
    colors = ["crimson", "#888", "#444"]
    bars = ax.bar(names, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005 * max(vals),
                 f"{v:.3f}", ha="center", fontsize=8)
    ax.set_ylabel("RMSE")
    ax.set_title("(c) Held-out forecast RMSE")

    # (d) Per-PC RMSE
    ax = fig.add_subplot(gs[2, 1])
    ax.bar(np.arange(len(rmse_per_pc)) + 1, rmse_per_pc, color="#4C72B0")
    ax.set_xlabel("PC index"); ax.set_ylabel("per-PC RMSE")
    ax.set_title("(d) Forecast error per PC dim")

    # (e) DMD eigenvalues
    ax = fig.add_subplot(gs[2, 2])
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="grey", lw=0.7, ls="--")
    eigs = dmd.eigenvalues_
    ax.scatter(eigs.real, eigs.imag, s=60, color="crimson", alpha=0.8)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_aspect("equal")
    ax.axhline(0, color="grey", lw=0.3); ax.axvline(0, color="grey", lw=0.3)
    ax.set_xlabel(r"Re($\lambda$)"); ax.set_ylabel(r"Im($\lambda$)")
    ax.set_title(f"(e) DMD spectrum (rank={len(eigs)})")

    fig.suptitle("Figure 24 — Classical application: short-term DMD "
                  "forecasting (Schmid-Brunton style)",
                  fontsize=12.5, y=0.985)
    fig.savefig(OUT / "24_app_dmd_forecasting.png", bbox_inches="tight")
    plt.close(fig)
    print("    saved 24_app_dmd_forecasting.png")


# ═════════════════════════════════════════════════════════════════════════════
# App 25 — Latent: word-embedding-style group separation
# ═════════════════════════════════════════════════════════════════════════════

def app_latent_embedding():
    print("\n[25] Latent: shared PCA + inter-group direction (aud vs vis)",
          flush=True)
    ratios_aud, ratios_vis, _, _ = load_B()
    az_aud = HarmonicSequenceAnalyzer.from_ratios_list(ratios_aud,
                                                       n_hist_bins=240)
    az_vis = HarmonicSequenceAnalyzer.from_ratios_list(ratios_vis,
                                                       n_hist_bins=240)
    H_aud = az_aud.histograms
    H_vis = az_vis.histograms

    # Joint PCA on combined histograms
    from sklearn.decomposition import PCA
    H_all = np.vstack([H_aud, H_vis])
    labels = np.array([0] * len(H_aud) + [1] * len(H_vis))
    pca = PCA(n_components=3).fit(H_all - H_all.mean(axis=0))
    Z = pca.transform(H_all - H_all.mean(axis=0))
    Z_aud = Z[labels == 0]
    Z_vis = Z[labels == 1]
    centroid_aud = Z_aud.mean(axis=0)
    centroid_vis = Z_vis.mean(axis=0)
    direction = centroid_vis - centroid_aud
    direction_norm = float(np.linalg.norm(direction))
    print(f"    inter-group direction norm: {direction_norm:.3f}")
    print(f"    explained var: {pca.explained_variance_ratio_.round(3)}")

    # Quick group-separability metric: silhouette score with the labels
    from sklearn.metrics import silhouette_score
    try:
        sil = silhouette_score(Z, labels)
    except Exception:
        sil = float("nan")

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.30,
                           left=0.06, right=0.97, top=0.92, bottom=0.07)

    # (a) Joint PC1-PC2 scatter with ellipses
    def _ellipse(points, ax, **kwargs):
        if len(points) < 2: return
        cov = np.cov(points.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        # 2-sigma ellipse for visualisation
        w, h = 2 * 2 * np.sqrt(np.maximum(eigvals, 0))
        e = Ellipse(xy=points.mean(axis=0), width=w, height=h, angle=angle,
                     **kwargs)
        ax.add_patch(e)

    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(Z_aud[:, 0], Z_aud[:, 1], s=42, color="#4C72B0",
                edgecolor="white", lw=0.6, label="auditory")
    ax.scatter(Z_vis[:, 0], Z_vis[:, 1], s=42, color="#DD8452",
                edgecolor="white", lw=0.6, label="visual")
    _ellipse(Z_aud[:, :2], ax, ec="#4C72B0", fc="none", lw=1.2, alpha=0.7)
    _ellipse(Z_vis[:, :2], ax, ec="#DD8452", fc="none", lw=1.2, alpha=0.7)
    ax.scatter(*centroid_aud[:2], s=160, marker="X", color="#4C72B0",
                edgecolor="black", lw=1.0, zorder=5)
    ax.scatter(*centroid_vis[:2], s=160, marker="X", color="#DD8452",
                edgecolor="black", lw=1.0, zorder=5)
    ax.annotate("", xy=centroid_vis[:2], xytext=centroid_aud[:2],
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.4))
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"(a) Joint embedding  (silhouette={sil:.2f})")
    ax.legend(frameon=False, fontsize=8)

    # (b) PC1 vs PC3
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(Z_aud[:, 0], Z_aud[:, 2], s=42, color="#4C72B0",
                edgecolor="white", lw=0.6, label="auditory")
    ax.scatter(Z_vis[:, 0], Z_vis[:, 2], s=42, color="#DD8452",
                edgecolor="white", lw=0.6, label="visual")
    ax.scatter(*[centroid_aud[0], centroid_aud[2]], s=160, marker="X",
                color="#4C72B0", edgecolor="black", lw=1.0, zorder=5)
    ax.scatter(*[centroid_vis[0], centroid_vis[2]], s=160, marker="X",
                color="#DD8452", edgecolor="black", lw=1.0, zorder=5)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC3")
    ax.set_title("(b) PC1 vs PC3")
    ax.legend(frameon=False, fontsize=8)

    # (c) Inter-group direction on the cents axis (the "aud-to-vis vector")
    ax = fig.add_subplot(gs[0, 2])
    # Project direction back to histogram space via inverse PCA
    delta_hist = pca.inverse_transform(direction[np.newaxis, :])[0]
    bin_cents = np.linspace(2.5, 1197.5, 240)
    ax.plot(bin_cents, delta_hist, color="black", lw=1.2)
    ax.axhline(0, color="grey", lw=0.4)
    ax.fill_between(bin_cents, 0, delta_hist,
                     where=delta_hist > 0, color="#DD8452", alpha=0.4,
                     label="visual+ direction")
    ax.fill_between(bin_cents, 0, delta_hist,
                     where=delta_hist <= 0, color="#4C72B0", alpha=0.4,
                     label="auditory+ direction")
    ax.set_xlabel("cents"); ax.set_ylabel("aud->vis loading")
    ax.set_title("(c) Inter-group direction on cents axis")
    ax.legend(frameon=False, fontsize=8)

    # (d) Per-PC group means with error bars
    ax = fig.add_subplot(gs[1, 0])
    pcs = np.arange(3)
    width = 0.36
    aud_mean = Z_aud.mean(axis=0); aud_std = Z_aud.std(axis=0)
    vis_mean = Z_vis.mean(axis=0); vis_std = Z_vis.std(axis=0)
    ax.bar(pcs - width/2, aud_mean, width, yerr=aud_std, capsize=4,
            color="#4C72B0", label="auditory", alpha=0.85)
    ax.bar(pcs + width/2, vis_mean, width, yerr=vis_std, capsize=4,
            color="#DD8452", label="visual", alpha=0.85)
    ax.set_xticks(pcs); ax.set_xticklabels(["PC1", "PC2", "PC3"])
    ax.set_ylabel("coordinate (mean ± std)")
    ax.set_title("(d) Per-PC group means")
    ax.legend(frameon=False, fontsize=8)

    # (e) Linear classifier accuracy
    ax = fig.add_subplot(gs[1, 1])
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    accs = []
    for n_pc in range(1, 4):
        scores = cross_val_score(
            LogisticRegression(max_iter=400),
            Z[:, :n_pc], labels, cv=5)
        accs.append(scores.mean())
    ax.plot([1, 2, 3], accs, "o-", color="black", lw=1.4)
    ax.axhline(0.5, color="grey", lw=0.5, ls="--", label="chance")
    ax.set_xticks([1, 2, 3])
    ax.set_xlabel("PCs used"); ax.set_ylabel("5-fold CV accuracy")
    ax.set_title("(e) Logistic-regression classifier accuracy")
    ax.set_ylim(0.3, 1.05)
    ax.legend(frameon=False, fontsize=8)

    # (f) Explained variance
    ax = fig.add_subplot(gs[1, 2])
    ax.bar(np.arange(3) + 1, pca.explained_variance_ratio_,
            color="#4C72B0")
    ax.set_xlabel("PC"); ax.set_ylabel("explained variance")
    ax.set_title(f"(f) Joint-PCA scree  "
                  f"({pca.explained_variance_ratio_.sum():.1%} cumul)")

    fig.suptitle("Figure 25 — Classical application: word-embedding-style "
                  "group separation in shared latent space",
                  fontsize=12.5, y=0.985)
    fig.savefig(OUT / "25_app_latent_embedding.png", bbox_inches="tight")
    plt.close(fig)
    print(f"    saved 25_app_latent_embedding.png  (silhouette={sil:.2f}, "
          f"CV-acc(3PC)={accs[-1]:.2f})")


# ═════════════════════════════════════════════════════════════════════════════
# App 26 — Topology: chaotic vs structured via shuffle control
# ═════════════════════════════════════════════════════════════════════════════

def app_topology_chaos():
    print("\n[26] Topology: chaotic vs structured (shuffle-control test)",
          flush=True)
    _, _, metrics_list = load_A()
    hs = np.array([float(m.get("harmsim", np.nan)) for m in metrics_list],
                  dtype=float)
    hs = hs[np.isfinite(hs)]
    print(f"    harmsim time series length: {len(hs)}")

    rng = np.random.default_rng(0)
    hs_shuffled = hs.copy(); rng.shuffle(hs_shuffled)

    # Takens embed + persistent homology on both
    from biotuner.harmonic_sequence import HarmonicTopology
    top_real = HarmonicTopology(embedding_dim=3, delay=2).fit(hs)
    top_rand = HarmonicTopology(embedding_dim=3, delay=2).fit(hs_shuffled)
    fp_real = top_real.session_fingerprint()
    fp_rand = top_rand.session_fingerprint()
    print(f"    real fingerprint:    {fp_real.round(3).tolist()}")
    print(f"    shuffled fingerprint:{fp_rand.round(3).tolist()}")

    # Multiple shuffles for null distribution
    n_shuffles = 30
    fps_null = []
    for s in range(n_shuffles):
        sh = hs.copy(); rng.shuffle(sh)
        try:
            fps_null.append(
                HarmonicTopology(embedding_dim=3, delay=2).fit(sh)
                                                          .session_fingerprint())
        except Exception:
            pass
    fps_null = np.array(fps_null)
    print(f"    {len(fps_null)} successful shuffles")

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.2, 1.0],
                           hspace=0.55, wspace=0.35,
                           left=0.06, right=0.97, top=0.93, bottom=0.07)

    # (a) Scalar time series (real)
    ax = fig.add_subplot(gs[0, :])
    ax.plot(hs, color="#4C72B0", lw=1.0, label="real harmsim")
    ax.plot(hs_shuffled, color="#DD8452", lw=0.6, alpha=0.5,
             label="shuffled")
    ax.set_xlabel("window"); ax.set_ylabel("harmsim")
    ax.set_title("(a) Scalar harmonicity time series — real signal "
                  "and one shuffled control")
    ax.legend(frameon=False, fontsize=8)

    # (b) Takens cloud REAL
    ax = fig.add_subplot(gs[1, 0])
    cloud_r = top_real.takens_embedding_
    ax.scatter(cloud_r[:, 0], cloud_r[:, 1],
                c=np.arange(len(cloud_r)), cmap="viridis", s=20,
                edgecolor="white", lw=0.4)
    ax.plot(cloud_r[:, 0], cloud_r[:, 1], color="grey", lw=0.3, alpha=0.4)
    ax.set_title("(b) Takens cloud — REAL")
    ax.set_xlabel("x(t)"); ax.set_ylabel(r"x(t+$\tau$)")

    # (c) Takens cloud SHUFFLED
    ax = fig.add_subplot(gs[1, 1])
    cloud_s = top_rand.takens_embedding_
    ax.scatter(cloud_s[:, 0], cloud_s[:, 1],
                c=np.arange(len(cloud_s)), cmap="plasma", s=20,
                edgecolor="white", lw=0.4)
    ax.set_title("(c) Takens cloud — SHUFFLED")
    ax.set_xlabel("x(t)"); ax.set_ylabel(r"x(t+$\tau$)")

    # (d) Persistence diagrams overlaid
    ax = fig.add_subplot(gs[1, 2])
    for top, name, color in [(top_real, "real", "#4C72B0"),
                              (top_rand, "shuffled", "#DD8452")]:
        diagrams = top.persistence_diagram_
        for dim, dgm in enumerate(diagrams[:1]):
            finite = dgm[np.isfinite(dgm[:, 1])]
            if len(finite):
                ax.scatter(finite[:, 0], finite[:, 1], s=24, alpha=0.6,
                            color=color, label=f"{name} H{dim}")
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1], 1)
    ax.plot([0, lim], [0, lim], color="grey", lw=0.4, ls="--")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("birth"); ax.set_ylabel("death")
    ax.set_title("(d) Persistence diagrams")
    ax.legend(frameon=False, fontsize=8)

    # (e) Fingerprint comparison with null distribution
    metric_names = ["mean_H0_pers", "max_H0_pers", "n_H0_bars",
                    "mean_H1_pers", "max_H1_pers", "n_H1_bars"]
    ax = fig.add_subplot(gs[2, :])
    if len(fps_null):
        bp = ax.boxplot(fps_null, positions=np.arange(len(metric_names)),
                         widths=0.55, patch_artist=True,
                         boxprops=dict(facecolor="#DD8452", alpha=0.4),
                         medianprops=dict(color="black"))
    ax.scatter(np.arange(len(metric_names)), fp_real,
                s=120, marker="*", color="crimson", zorder=5,
                label="real fingerprint")
    ax.scatter(np.arange(len(metric_names)),
                fps_null.mean(axis=0) if len(fps_null) else [],
                s=60, marker="o", color="black", zorder=4,
                label=f"shuffle-mean (n={len(fps_null)})")
    ax.set_xticks(np.arange(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("value")
    ax.set_title("(e) Real fingerprint vs shuffle null distribution — "
                  "where the star sits outside the box, structure is "
                  "stronger than chance")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Figure 26 — Classical application: topological chaos "
                  "vs structure (cardiac-arrhythmia style)",
                  fontsize=12.5, y=0.985)
    fig.savefig(OUT / "26_app_topology_chaos.png", bbox_inches="tight")
    plt.close(fig)
    print("    saved 26_app_topology_chaos.png")


# ═════════════════════════════════════════════════════════════════════════════
# App 27 — Grammar: stylometric condition attribution
# ═════════════════════════════════════════════════════════════════════════════

def app_grammar_stylometry():
    print("\n[27] Grammar: n-gram stylometry classifying aud vs vis epochs",
          flush=True)
    ratios_aud, ratios_vis, _, _ = load_B()
    # Build chord sequences via the grammar's labelling
    from biotuner.harmonic_sequence import HarmonicGrammar, _chord_label
    n = len(ratios_aud)
    seqs_aud = [_chord_label(r, tolerance_cents=30.0) for r in ratios_aud]
    seqs_vis = [_chord_label(r, tolerance_cents=30.0) for r in ratios_vis]
    # Each epoch is currently a single chord (a frozenset). For an n-gram
    # to be meaningful we group N adjacent epochs into one "phrase".
    PHRASE_LEN = 3
    def _phrases(seqs):
        phrases = []
        for i in range(0, len(seqs) - PHRASE_LEN + 1):
            phrases.append(tuple(seqs[i:i+PHRASE_LEN]))
        return phrases
    phr_aud = _phrases(seqs_aud)
    phr_vis = _phrases(seqs_vis)
    print(f"    aud phrases: {len(phr_aud)}  vis phrases: {len(phr_vis)}")

    # Build bigram models per condition via collections.Counter
    from collections import Counter
    def _bigram_model(phrases):
        counts = Counter()
        contexts = Counter()
        for phrase in phrases:
            for i in range(len(phrase) - 1):
                counts[(phrase[i], phrase[i + 1])] += 1
                contexts[(phrase[i],)] += 1
        return counts, contexts

    counts_aud, ctx_aud = _bigram_model(phr_aud)
    counts_vis, ctx_vis = _bigram_model(phr_vis)
    print(f"    aud bigrams: {len(counts_aud)}  vis bigrams: {len(counts_vis)}")

    # Score a phrase under a model: sum log P(token_{i+1} | token_i)
    # with Laplace smoothing
    VOCAB = set(
        [t for phrase in (phr_aud + phr_vis) for t in phrase] +
        list(set([t for c in (counts_aud, counts_vis) for t in [k[1] for k in c]]))
    )
    V = max(1, len(VOCAB))

    def _logp_phrase(phrase, counts, contexts):
        # P(t_{i+1}|t_i) = (count + 1) / (ctx + V)
        s = 0.0
        for i in range(len(phrase) - 1):
            c = counts.get((phrase[i], phrase[i+1]), 0)
            n = contexts.get((phrase[i],), 0)
            s += np.log((c + 1.0) / (n + V))
        return s

    # Test: leave-one-phrase-out cross-validation
    all_phrases = phr_aud + phr_vis
    labels = np.array([0] * len(phr_aud) + [1] * len(phr_vis))
    n_phrases = len(all_phrases)

    preds = []
    for i in range(n_phrases):
        # Remove phrase i from training, rebuild bigram counts
        train_phr_aud = [p for j, p in enumerate(phr_aud) if j != i]
        train_phr_vis = [p for j, p in enumerate(phr_vis)
                          if (j + len(phr_aud)) != i]
        c_a, x_a = _bigram_model(train_phr_aud)
        c_v, x_v = _bigram_model(train_phr_vis)
        lp_a = _logp_phrase(all_phrases[i], c_a, x_a)
        lp_v = _logp_phrase(all_phrases[i], c_v, x_v)
        preds.append(0 if lp_a > lp_v else 1)
    preds = np.array(preds)
    acc = float((preds == labels).mean())
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    print(f"    LOO accuracy: {acc:.2f}, confusion:\n{cm}")

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.50, wspace=0.30,
                           left=0.06, right=0.97, top=0.92, bottom=0.07)

    # (a) Chord-token streams (aud and vis side by side, coloured)
    ax = fig.add_subplot(gs[0, :])
    all_tokens = list({t for seqs in (seqs_aud, seqs_vis) for t in seqs})
    tok_id = {t: i for i, t in enumerate(all_tokens)}
    cmap = plt.get_cmap("tab20")
    for i, t in enumerate(seqs_aud):
        ax.add_patch(plt.Rectangle((i, 0), 1, 0.4,
                                    color=cmap(tok_id[t] % 20)))
    for i, t in enumerate(seqs_vis):
        ax.add_patch(plt.Rectangle((i + len(seqs_aud) + 2, 0), 1, 0.4,
                                    color=cmap(tok_id[t] % 20)))
    ax.set_xlim(0, len(seqs_aud) + len(seqs_vis) + 2)
    ax.set_ylim(-0.05, 0.55)
    ax.set_yticks([])
    ax.set_xlabel("epoch")
    ax.text(len(seqs_aud) / 2, 0.46, "AUDITORY", ha="center",
             fontsize=9, color="#4C72B0", fontweight="bold")
    ax.text(len(seqs_aud) + 2 + len(seqs_vis) / 2, 0.46, "VISUAL",
             ha="center", fontsize=9, color="#DD8452", fontweight="bold")
    ax.set_title(f"(a) Per-epoch chord tokens  "
                  f"(vocab={len(all_tokens)} distinct chords)")

    # (b) Confusion matrix
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(cm, cmap="rocket_r", vmin=0)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     fontsize=14, color="white" if cm[i, j] > cm.max()*0.5
                     else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["aud", "vis"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["aud", "vis"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"(b) LOO confusion matrix\n"
                  f"accuracy = {acc:.2f}  (chance = 0.5)")

    # (c) Per-phrase log-likelihood difference distribution
    ax = fig.add_subplot(gs[1, 1])
    diffs = []
    for i in range(n_phrases):
        lp_a = _logp_phrase(all_phrases[i], counts_aud, ctx_aud)
        lp_v = _logp_phrase(all_phrases[i], counts_vis, ctx_vis)
        diffs.append(lp_a - lp_v)
    diffs = np.array(diffs)
    ax.hist(diffs[labels == 0], bins=12, color="#4C72B0", alpha=0.6,
             label=f"true aud (n={(labels==0).sum()})")
    ax.hist(diffs[labels == 1], bins=12, color="#DD8452", alpha=0.6,
             label=f"true vis (n={(labels==1).sum()})")
    ax.axvline(0, color="black", lw=0.8, ls="--", label="decision = 0")
    ax.set_xlabel(r"$\log P(\mathrm{phrase}|\mathrm{aud}) - \log P(\mathrm{phrase}|\mathrm{vis})$",
                  fontsize=8)
    ax.set_ylabel("count")
    ax.set_title("(c) Per-phrase classifier score")
    ax.legend(frameon=False, fontsize=8)

    # (d) Vocabulary venn-ish: per-token frequency aud vs vis
    ax = fig.add_subplot(gs[1, 2])
    cnt_aud = Counter(seqs_aud); cnt_vis = Counter(seqs_vis)
    keys = sorted(set(cnt_aud) | set(cnt_vis), key=lambda k: -cnt_aud.get(k, 0) - cnt_vis.get(k, 0))[:12]
    a_freq = [cnt_aud.get(k, 0) / max(len(seqs_aud), 1) for k in keys]
    v_freq = [cnt_vis.get(k, 0) / max(len(seqs_vis), 1) for k in keys]
    x = np.arange(len(keys))
    width = 0.4
    ax.bar(x - width/2, a_freq, width, color="#4C72B0", label="aud")
    ax.bar(x + width/2, v_freq, width, color="#DD8452", label="vis")
    ax.set_xticks(x)
    ax.set_xticklabels([f"C{i}" for i in range(len(keys))], fontsize=8)
    ax.set_ylabel("token frequency")
    ax.set_title(f"(d) Top-12 chord-token frequencies\n"
                  f"(vocab_aud={len(cnt_aud)}, vocab_vis={len(cnt_vis)})")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Figure 27 — Classical application: stylometric "
                  "condition attribution (Mosteller-Wallace style)",
                  fontsize=12.5, y=0.985)
    fig.savefig(OUT / "27_app_grammar_stylometry.png", bbox_inches="tight")
    plt.close(fig)
    print(f"    saved 27_app_grammar_stylometry.png  (LOO acc={acc:.2f})")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    apps = [
        app_markov_composition,
        app_wasserstein_geodesic,
        app_dmd_forecasting,
        app_latent_embedding,
        app_topology_chaos,
        app_grammar_stylometry,
    ]
    for fn in apps:
        try:
            fn()
        except Exception as exc:
            import traceback
            print(f"    !! {fn.__name__} failed: {exc}")
            traceback.print_exc()
    print(f"\nAll classical-application figures in {OUT}")
