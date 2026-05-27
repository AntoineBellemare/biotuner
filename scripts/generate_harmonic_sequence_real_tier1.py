"""
generate_harmonic_sequence_real_tier1.py
========================================
Extended real-EEG explorations of the harmonic_sequence module, covering all
four "Tier 1" experiments from the proposal:

  A. Sliding-window temporal analysis on a single channel.
  B. Cross-condition fingerprinting on MNE-Sample auditory vs visual epochs.
  C. Scientific-pathway demo (harmonicity_spectrum) on the bundled EEG file.
  D. Frequency-band stratification (delta/theta/alpha/beta/gamma) on the
     bundled EEG file.

Each experiment is independent, has its own disk cache, and produces one
publication-quality figure into docs/papers/figures/.

Run:
    python scripts/generate_harmonic_sequence_real_tier1.py [A|B|C|D|all]
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from biotuner.biotuner_object import compute_biotuner
from biotuner.harmonic_sequence import HarmonicSequenceAnalyzer

OUT = ROOT / "docs" / "papers" / "figures"
CACHE_DIR = ROOT / "docs" / "papers"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 140, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})

EEG_PATH = ROOT / "docs" / "examples" / "data" / "EEG_example.npy"
SF = 1000
FREQ_BANDS_FULL = [[1, 3], [3, 7], [7, 12], [12, 18], [18, 30], [30, 45]]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class StubBt:
    """Lightweight bt-like object with just `.data, .sf, .peaks, .peaks_ratios,
    .peaks_metrics` — enough for HarmonicSequenceAnalyzer + the scientific
    pathway encoder (which needs `.data`)."""
    def __init__(self, data, sf, peaks, ratios, metrics):
        self.data = data
        self.sf = sf
        self.peaks = peaks
        self.peaks_ratios = ratios
        self.peaks_metrics = metrics
        self.scale_metrics = {}


def _safe_extract(sig, sf, freq_bands=None, max_freq=45, n_peaks=4,
                  precision=0.5, peaks_function="fixed"):
    """Run compute_biotuner.peaks_extraction with sensible failure handling.
    Returns (peaks, ratios, metrics_dict)."""
    bt = compute_biotuner(sf=sf, peaks_function=peaks_function,
                          precision=precision, n_harm=10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            kwargs = dict(max_freq=max_freq, n_peaks=n_peaks,
                          graph=False, min_harms=2, verbose=False)
            if freq_bands is not None:
                kwargs["FREQ_BANDS"] = freq_bands
            bt.peaks_extraction(sig.astype(np.float64), **kwargs)
            bt.compute_peaks_metrics()
        except Exception:
            return [], [], {}
    peaks = list(bt.peaks) if bt.peaks is not None else []
    ratios = list(bt.peaks_ratios) if getattr(bt, "peaks_ratios", None) else []
    metrics = dict(getattr(bt, "peaks_metrics", {}) or {})
    return peaks, ratios, metrics


def _bandpass(sig, low, high, sf, order=4):
    sos = butter(order, [low, high], btype="bandpass", fs=sf, output="sos")
    return sosfiltfilt(sos, sig)


# ═════════════════════════════════════════════════════════════════════════════
# A. Sliding-window temporal analysis on a single channel
# ═════════════════════════════════════════════════════════════════════════════

def run_A():
    print("\n[A] Sliding-window temporal analysis on a concatenated EEG epoch",
          flush=True)
    cache = CACHE_DIR / "_tier1_A_cache.npz"
    # 4 s per channel is too short for meaningful temporal dynamics.
    # We concatenate 30 channels with a tiny linear crossfade between each
    # pair to make a 120 s pseudo-continuous recording; then slide 2 s
    # windows with 0.5 s hop over it. The crossfade keeps the
    # sample-to-sample step small so peaks_extraction doesn't see a step
    # discontinuity at the join.
    raw = np.load(EEG_PATH)
    CHANNEL_FROM, CHANNEL_TO = 20, 50
    XFADE = 50         # samples (50 ms) of linear crossfade per join
    pieces = [raw[c].astype(np.float64) for c in range(CHANNEL_FROM, CHANNEL_TO)]
    sig_full = pieces[0].copy()
    for p in pieces[1:]:
        # crossfade tail of current with head of next
        tail = sig_full[-XFADE:].copy()
        head = p[:XFADE].copy()
        ramp = np.linspace(0, 1, XFADE)
        blended = tail * (1 - ramp) + head * ramp
        sig_full = np.concatenate([sig_full[:-XFADE], blended, p[XFADE:]])
    print(f"    concatenated channels {CHANNEL_FROM}..{CHANNEL_TO-1} "
          f"with {XFADE}-sample crossfade -> {len(sig_full)} samples "
          f"({len(sig_full)/SF:.1f} s)")
    win_len, hop = 2000, 500
    starts = list(range(0, len(sig_full) - win_len + 1, hop))
    T = len(starts)
    print(f"    {T} windows of {win_len/SF:.1f} s with "
          f"{hop/SF:.2f} s hop", flush=True)

    if cache.exists():
        npz = np.load(cache, allow_pickle=True)
        ratios_list = list(npz["ratios_list"])
        peaks_list = list(npz["peaks_list"])
        metrics_list = list(npz["metrics_list"])
        print(f"    loaded cache from {cache.name}", flush=True)
    else:
        ratios_list, peaks_list, metrics_list = [], [], []
        t0 = time.time()
        for i, s in enumerate(starts):
            p, r, m = _safe_extract(sig_full[s:s + win_len], SF,
                                     freq_bands=FREQ_BANDS_FULL,
                                     max_freq=45, n_peaks=5, precision=1.0)
            peaks_list.append(p); ratios_list.append(r); metrics_list.append(m)
            if (i + 1) % 10 == 0 or (i + 1) == T:
                print(f"      window {i+1}/{T} done  ({time.time()-t0:.1f}s)",
                      flush=True)
        np.savez(cache,
                 ratios_list=np.array(ratios_list, dtype=object),
                 peaks_list=np.array(peaks_list, dtype=object),
                 metrics_list=np.array(metrics_list, dtype=object))
        print(f"    cached to {cache.name}", flush=True)

    # Build bt-like stubs so harmsim is exposed (analyzer's _get_scalar_series
    # can pull from peaks_metrics)
    bt_list = []
    for i, s in enumerate(starts):
        bt_list.append(StubBt(sig_full[s:s+win_len], SF,
                              peaks_list[i], ratios_list[i], metrics_list[i]))

    az = HarmonicSequenceAnalyzer.from_biotuner_list(
        bt_list, tuning="peaks_ratios", n_hist_bins=240,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        az.fit_all(n_states="auto", auto_k_range=(2, 4))
    print(f"    " + az.summary().replace("\n", "\n    "))

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.0, 1.4, 1.2, 1.2],
                           hspace=0.55, wspace=0.32,
                           left=0.06, right=0.97, top=0.94, bottom=0.07)

    # (a) Raw signal with channel-boundary marks
    ax = fig.add_subplot(gs[0, :])
    t_axis = np.arange(len(sig_full)) / SF
    ax.plot(t_axis, sig_full * 1e6, color="black", lw=0.3)
    # Channel boundaries fall every 4 s in the concatenated signal (one
    # 4-second source channel per join). Subtract the cumulative XFADE
    # so the line lands on the actual join sample.
    SOURCE_LEN_S = 4.0
    n_channels_used = 30
    for k in range(1, n_channels_used):
        # each join consumed XFADE samples of overlap
        x = k * SOURCE_LEN_S - k * XFADE / SF / 2
        ax.axvline(x, color="steelblue", alpha=0.25, lw=0.6)
    ax.set_xlim(0, len(sig_full) / SF)
    ax.set_xlabel("time (s, pseudo-temporal)"); ax.set_ylabel("µV")
    ax.set_title(f"(a) Concatenated EEG signal "
                  f"({n_channels_used} channels x 4 s with crossfade)  "
                  f"— faint blue lines mark channel boundaries")

    # (b) Histogram trajectory + Markov state strip
    ax = fig.add_subplot(gs[1, :])
    H = az.histograms
    im = ax.imshow(H.T, aspect="auto", origin="lower", cmap="magma",
                    extent=[0, T, 0, 1200])
    labels = az.markov.state_labels_
    cmap_st = plt.get_cmap("tab10")
    ymin, ymax = 0, 1200
    strip = 70
    for t in range(T):
        ax.add_patch(plt.Rectangle((t - 0.5, ymax + 12), 1, strip,
                                    color=cmap_st(int(labels[t]) % 10),
                                    clip_on=False))
    ax.set_ylim(0, ymax + strip + 18)
    ax.set_xlabel("window index"); ax.set_ylabel("cents")
    ax.set_title("(b) Cents-histogram trajectory + Markov state strip "
                  f"(K={az.markov.n_states}, "
                  f"H={az.markov.transition_entropy_:.2f} bits)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="weight")

    # (c) Flux time series
    ax = fig.add_subplot(gs[2, 0])
    flux = az.wasserstein.flux_
    ax.plot(np.arange(len(flux)), flux, color="black", lw=1.4)
    z = (flux - flux.mean()) / (flux.std() + 1e-9)
    for d in np.where(z > 2.0)[0]:
        ax.scatter(d, flux[d], s=70, color="crimson", zorder=5)
    ax.set_xlabel("window transition"); ax.set_ylabel("$W_1$ flux")
    ax.set_title(f"(c) Temporal harmonic flux (mean={flux.mean():.2f})")

    # (d) Latent PC1-PC2 trajectory
    ax = fig.add_subplot(gs[2, 1])
    Z = az.latent.trajectory()
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=np.arange(T), cmap="viridis",
                     s=46, edgecolor="white", lw=0.6)
    ax.plot(Z[:, 0], Z[:, 1], color="grey", lw=0.5, alpha=0.5)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    evr = az.latent.explained_variance_ratio_
    ax.set_title(f"(d) Latent trajectory  ({evr[:2].sum():.0%} var)")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="window")

    # (e) DMD eigenvalues — now this can find genuine oscillation
    ax = fig.add_subplot(gs[2, 2])
    if az.dmd is not None and len(az.dmd.eigenvalues_) > 0:
        eigs = az.dmd.eigenvalues_
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), color="grey", lw=0.8, ls="--")
        osc, idx = az.dmd.oscillatory_modes(threshold=0.15)
        is_osc = np.zeros(len(eigs), dtype=bool); is_osc[idx] = True
        ax.scatter(eigs[~is_osc].real, eigs[~is_osc].imag,
                    s=60, color="#888", alpha=0.75, label="decaying")
        ax.scatter(eigs[is_osc].real, eigs[is_osc].imag,
                    s=130, color="crimson", marker="*",
                    label=f"|λ|≈1 ({len(idx)})")
        ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_aspect("equal")
        ax.axhline(0, color="grey", lw=0.3); ax.axvline(0, color="grey", lw=0.3)
        ax.set_xlabel(r"Re($\lambda$)"); ax.set_ylabel(r"Im($\lambda$)")
        ax.set_title("(e) DMD spectrum  (temporal: oscillation possible)")
        ax.legend(frameon=False, fontsize=7)

    # (f) Per-window harmsim curve
    ax = fig.add_subplot(gs[3, :])
    hs = np.array([m.get("harmsim", np.nan) for m in metrics_list],
                  dtype=float)
    cons = np.array([m.get("cons", np.nan) for m in metrics_list],
                    dtype=float)
    ax.plot(np.arange(T), hs, "o-", color="#4C72B0", label="harmsim",
             lw=1.3, markersize=5)
    ax2 = ax.twinx()
    ax2.plot(np.arange(T), cons, "s--", color="#DD8452", label="cons",
              lw=1.0, markersize=4, alpha=0.85)
    ax.set_xlabel("window"); ax.set_ylabel("harmsim", color="#4C72B0")
    ax2.set_ylabel("cons", color="#DD8452")
    ax.set_title("(f) Per-window scalar metrics from peaks_metrics — "
                  "the temporal harmonic-similarity trace")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right", fontsize=8)

    fig.suptitle(
        "Figure 18 — Experiment A: sliding-window temporal analysis "
        f"on EEG_example.npy channel 39  ({T} windows)",
        fontsize=12.5, y=0.985)
    fig.savefig(OUT / "18_real_eeg_sliding_window.png", bbox_inches="tight")
    plt.close(fig)
    print("    saved 18_real_eeg_sliding_window.png")


# ═════════════════════════════════════════════════════════════════════════════
# B. Cross-condition fingerprinting on MNE-Sample auditory vs visual epochs
# ═════════════════════════════════════════════════════════════════════════════

def run_B():
    print("\n[B] Cross-condition fingerprinting on MNE-Sample auditory vs visual",
          flush=True)
    cache = CACHE_DIR / "_tier1_B_cache.npz"
    N_EPOCHS = 15
    MAX_FREQ_B = 30        # tighter band for faster peaks_extraction

    if cache.exists():
        npz = np.load(cache, allow_pickle=True)
        ratios_aud = list(npz["ratios_aud"])
        ratios_vis = list(npz["ratios_vis"])
        metrics_aud = list(npz["metrics_aud"])
        metrics_vis = list(npz["metrics_vis"])
        sig_aud = npz["sig_aud"]
        sig_vis = npz["sig_vis"]
        sf_used = float(npz["sf_used"])
        print(f"    loaded cache from {cache.name}")
    else:
        import mne
        from mne.datasets import sample
        sample_dir = Path(sample.data_path()) / "MEG" / "sample"
        raw_fname = sample_dir / "sample_audvis_raw.fif"
        event_fname = sample_dir / "sample_audvis_raw-eve.fif"
        print(f"    loading {raw_fname.name} ...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)
            raw.pick(["eeg"]).filter(1, MAX_FREQ_B, verbose=False, n_jobs=1)
            events = mne.read_events(event_fname)
            # Event IDs: 1=aud-left, 2=aud-right, 3=vis-left, 4=vis-right
            event_id = {"auditory_left": 1, "auditory_right": 2,
                        "visual_left": 3, "visual_right": 4}
            epochs = mne.Epochs(raw, events, event_id=event_id,
                                tmin=-0.1, tmax=1.0, baseline=(None, 0),
                                preload=True, verbose=False)
        sf_used = float(epochs.info["sfreq"])
        # Average 5 adjacent EEG channels per epoch — this gives a much
        # cleaner signal than any single channel and yields more meaningful
        # peak extraction. We pick a posterior cluster which should
        # respond strongly to both conditions in different ways.
        ch_names = [f"EEG 0{n:02d}" for n in (54, 55, 56, 57, 58)]
        ch_idx = [epochs.ch_names.index(c) for c in ch_names]
        print(f"    averaging across {ch_names}, sf={sf_used:.0f} Hz, "
              f"{len(epochs)} total epochs", flush=True)

        aud = epochs["auditory_left", "auditory_right"].get_data()
        vis = epochs["visual_left", "visual_right"].get_data()
        # mean across the 5 channels -> (n_epochs, n_times)
        sig_aud = aud[:N_EPOCHS, ch_idx, :].mean(axis=1).astype(np.float64)
        sig_vis = vis[:N_EPOCHS, ch_idx, :].mean(axis=1).astype(np.float64)
        print(f"    aud epochs: {sig_aud.shape}, vis epochs: {sig_vis.shape}",
              flush=True)

        def _extract(sigs, label):
            rats, mets = [], []
            t0 = time.time()
            for i, s in enumerate(sigs):
                _, r, m = _safe_extract(s, sf_used,
                                         freq_bands=[[1,4],[4,8],[8,12],
                                                     [12,18],[18,30]],
                                         max_freq=MAX_FREQ_B, n_peaks=5,
                                         precision=1.0)
                rats.append(r); mets.append(m)
                print(f"      {label} {i+1}/{len(sigs)} "
                      f"({time.time()-t0:.1f}s)", flush=True)
            return rats, mets

        ratios_aud, metrics_aud = _extract(sig_aud, "aud")
        ratios_vis, metrics_vis = _extract(sig_vis, "vis")
        np.savez(cache,
                 ratios_aud=np.array(ratios_aud, dtype=object),
                 ratios_vis=np.array(ratios_vis, dtype=object),
                 metrics_aud=np.array(metrics_aud, dtype=object),
                 metrics_vis=np.array(metrics_vis, dtype=object),
                 sig_aud=sig_aud, sig_vis=sig_vis,
                 sf_used=np.array(sf_used))
        print(f"    cached to {cache.name}")

    # Build analyzers per condition
    def _make_az(ratios, metrics):
        # Use stub bts so harmsim flows in via peaks_metrics
        bts = [StubBt(np.array([]), sf_used, [], r, m)
               for r, m in zip(ratios, metrics)]
        az = HarmonicSequenceAnalyzer.from_biotuner_list(
            bts, tuning="peaks_ratios", n_hist_bins=240,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            az.fit_all(n_states="auto", auto_k_range=(2, 5))
        return az

    az_aud = _make_az(ratios_aud, metrics_aud)
    az_vis = _make_az(ratios_vis, metrics_vis)

    print(f"    AUD: {az_aud.summary().splitlines()[1]}")
    print(f"    VIS: {az_vis.summary().splitlines()[1]}")

    def _fingerprint(az):
        flux = az.wasserstein.flux_
        beta = az.topology.betti_numbers_ if az.topology else np.array([0, 0])
        evr = (az.latent.explained_variance_ratio_[:2].sum()
               if az.latent else 0.0)
        eigs = (np.abs(az.dmd.eigenvalues_) if az.dmd is not None
                else np.array([]))
        osc = float((np.abs(eigs - 1.0) < 0.1).mean()) if len(eigs) else 0.0
        return {
            "flux_mean": float(flux.mean()),
            "flux_std": float(flux.std()),
            "K_markov": float(az.markov.n_states),
            "H_markov": float(az.markov.transition_entropy_),
            "var_PC1+2": float(evr),
            "beta0": float(beta[0]),
            "osc_frac": osc,
            "H_grammar": float(az.grammar.transition_entropy_)
                         if az.grammar else 0.0,
            "vocab": float(len(az.grammar.vocabulary_)) if az.grammar else 0,
        }

    fp_aud = _fingerprint(az_aud)
    fp_vis = _fingerprint(az_vis)
    metrics = list(fp_aud.keys())
    aud_vals = np.array([fp_aud[m] for m in metrics])
    vis_vals = np.array([fp_vis[m] for m in metrics])

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 1.0],
                           hspace=0.55, wspace=0.30,
                           left=0.06, right=0.97, top=0.93, bottom=0.07)

    # (a-b) Histogram trajectories side-by-side
    # Sparse cents histograms with ~5 peaks per epoch leave most cells at
    # zero; we set vmax to the 95th percentile of the union so the
    # populated cells are clearly visible.
    H_a, H_v = az_aud.histograms, az_vis.histograms
    vmax = float(np.percentile(np.concatenate([H_a.ravel(), H_v.ravel()]),
                               99))
    vmax = max(vmax, 1e-3)
    ax = fig.add_subplot(gs[0, :2])
    im = ax.imshow(H_a.T, aspect="auto", origin="lower", cmap="magma",
                    vmin=0, vmax=vmax,
                    extent=[0, len(ratios_aud), 0, 1200])
    ax.set_xlabel("auditory epoch"); ax.set_ylabel("cents")
    ax.set_title(f"(a) Auditory cents-histogram trajectory "
                  f"({len(ratios_aud)} epochs, 5-channel avg)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)

    ax = fig.add_subplot(gs[1, :2])
    im = ax.imshow(H_v.T, aspect="auto", origin="lower", cmap="magma",
                    vmin=0, vmax=vmax,
                    extent=[0, len(ratios_vis), 0, 1200])
    ax.set_xlabel("visual epoch"); ax.set_ylabel("cents")
    ax.set_title(f"(b) Visual cents-histogram trajectory "
                  f"({len(ratios_vis)} epochs, 5-channel avg)")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)

    # (c) Harmsim distribution per condition (violin)
    ax = fig.add_subplot(gs[0, 2])
    hs_aud = np.array([m.get("harmsim", np.nan) for m in metrics_aud],
                      dtype=float)
    hs_vis = np.array([m.get("harmsim", np.nan) for m in metrics_vis],
                      dtype=float)
    hs_aud = hs_aud[np.isfinite(hs_aud)]
    hs_vis = hs_vis[np.isfinite(hs_vis)]
    parts = ax.violinplot([hs_aud, hs_vis], positions=[0, 1],
                           showmeans=True, widths=0.7)
    for pc, c in zip(parts["bodies"], ["#4C72B0", "#DD8452"]):
        pc.set_facecolor(c); pc.set_alpha(0.7)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["auditory", "visual"])
    ax.set_ylabel("harmsim")
    from scipy.stats import mannwhitneyu
    u, p = mannwhitneyu(hs_aud, hs_vis, alternative="two-sided")
    ax.set_title(f"(c) Harmsim per epoch\n"
                  f"Mann-Whitney p = {p:.3f}", fontsize=9.5)

    # (d) Flux time series both conditions
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(az_aud.wasserstein.flux_, color="#4C72B0", lw=1.3,
             label=f"auditory (μ={fp_aud['flux_mean']:.1f})")
    ax.plot(az_vis.wasserstein.flux_, color="#DD8452", lw=1.3,
             label=f"visual (μ={fp_vis['flux_mean']:.1f})")
    ax.set_xlabel("epoch transition"); ax.set_ylabel("$W_1$ flux")
    ax.set_title("(d) Between-epoch flux")
    ax.legend(frameon=False, fontsize=8)

    # (e) Side-by-side fingerprint bars
    ax = fig.add_subplot(gs[2, :2])
    x = np.arange(len(metrics))
    width = 0.36
    # Normalise per metric so they fit on one chart
    pair = np.stack([aud_vals, vis_vals])
    norm = pair / (np.maximum(pair.max(axis=0), 1e-9))
    ax.bar(x - width/2, norm[0], width, color="#4C72B0", label="auditory")
    ax.bar(x + width/2, norm[1], width, color="#DD8452", label="visual")
    for i, m in enumerate(metrics):
        ax.text(i - width/2, norm[0, i] + 0.02, f"{aud_vals[i]:.2g}",
                 ha="center", fontsize=7, color="#4C72B0")
        ax.text(i + width/2, norm[1, i] + 0.02, f"{vis_vals[i]:.2g}",
                 ha="center", fontsize=7, color="#DD8452")
    ax.set_xticks(x); ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.set_ylabel("value (per-metric normalised)")
    ax.set_title("(e) 9-metric fingerprint — values printed in absolute units")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_ylim(0, 1.18)

    # (f) Markov transition matrices side-by-side
    ax = fig.add_subplot(gs[2, 2])
    Tm_a = az_aud.markov.transition_matrix_
    Tm_v = az_vis.markov.transition_matrix_
    Ka, Kv = Tm_a.shape[0], Tm_v.shape[0]
    # Plot two heatmaps stacked
    combined = np.zeros((Ka, Ka + 1 + Kv))
    combined[:Ka, :Ka] = Tm_a
    if Kv == Ka:
        combined[:Kv, Ka+1:] = Tm_v
    im = ax.imshow(combined, cmap="rocket_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks([Ka/2-0.5, Ka + Kv/2 + 0.5])
    ax.set_xticklabels([f"AUD (K={Ka})", f"VIS (K={Kv})"], fontsize=9)
    ax.set_yticks([])
    ax.set_title(f"(f) Markov transition matrices\n"
                  f"H_aud={fp_aud['H_markov']:.2f}, "
                  f"H_vis={fp_vis['H_markov']:.2f} bits", fontsize=9.5)

    fig.suptitle(
        "Figure 19 — Experiment B: cross-condition fingerprinting on "
        "MNE-Sample auditory vs visual epochs", fontsize=12.5, y=0.985)
    fig.savefig(OUT / "19_real_eeg_cross_condition.png", bbox_inches="tight")
    plt.close(fig)
    print("    saved 19_real_eeg_cross_condition.png")


# ═════════════════════════════════════════════════════════════════════════════
# C. Scientific pathway (harmonicity_spectrum) on EEG_example.npy
# ═════════════════════════════════════════════════════════════════════════════

def run_C():
    print("\n[C] Scientific pathway: harmonicity_spectrum vs cents_histogram")
    # Reuse the cache built by generate_harmonic_sequence_real_data.py
    src_cache = CACHE_DIR / "_real_eeg_bt_cache.npz"
    if not src_cache.exists():
        print(f"    !! requires {src_cache.name} from "
              "generate_harmonic_sequence_real_data.py")
        return
    npz = np.load(src_cache, allow_pickle=True)
    ratios_list = list(npz["ratios_list"])
    peaks_list = list(npz["peaks_list"])
    harmsim_list = npz["harmsim_list"]

    # Need bt objects with `.data` for the harmonicity_spectrum encoder.
    eeg = np.load(EEG_PATH)
    CHANNEL_START = 20; N_CHANNELS = 60
    channels = list(range(CHANNEL_START, CHANNEL_START + N_CHANNELS))
    metrics_list = [{"harmsim": float(h)} for h in harmsim_list]

    bts = [StubBt(eeg[ch].astype(np.float64), SF,
                  peaks_list[i], ratios_list[i], metrics_list[i])
           for i, ch in enumerate(channels)]

    az_cents = HarmonicSequenceAnalyzer.from_biotuner_list(
        bts, tuning="peaks_ratios",
        representation="cents_histogram", n_hist_bins=240,
    )
    az_spec = HarmonicSequenceAnalyzer.from_biotuner_list(
        bts, tuning="peaks_ratios",
        representation="harmonicity_spectrum",
        representation_kwargs=dict(fmin=1.0, fmax=45.0, precision_hz=0.5,
                                    metric="harmsim", n_harms=10,
                                    normalize=True),
    )

    print("    fitting cents-pathway analyzer...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        az_cents.fit_markov(n_states="auto", auto_k_range=(2, 6))
        az_cents.fit_wasserstein()
        az_cents.fit_latent(latent_dim=3)
    print(f"    cents:  {az_cents.summary().splitlines()[1]}")

    print("    fitting spectrum-pathway analyzer (slow: encodes harmonicity)")
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        az_spec.fit_markov(n_states="auto", auto_k_range=(2, 6))
        az_spec.fit_wasserstein()
        az_spec.fit_latent(latent_dim=3)
    print(f"    spectrum: {az_spec.summary().splitlines()[1]} "
          f"({time.time()-t0:.1f}s)")

    H_cents = az_cents.histograms
    H_spec = az_spec.features                 # (T, F)
    spec_freqs = az_spec.features_freqs_      # (F,)

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14.5, 9))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1.2, 1.2],
                           hspace=0.55, wspace=0.30,
                           left=0.06, right=0.97, top=0.93, bottom=0.07)

    # (a) cents-histogram trajectory
    ax = fig.add_subplot(gs[0, :])
    im = ax.imshow(H_cents.T, aspect="auto", origin="lower", cmap="magma",
                    extent=[0, H_cents.shape[0], 0, 1200])
    ax.set_xlabel("channel"); ax.set_ylabel("cents")
    ax.set_title("(a) MUSICAL pathway — cents-histogram trajectory")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="weight")

    # (b) harmonicity-spectrum trajectory in Hz
    ax = fig.add_subplot(gs[1, :])
    im = ax.imshow(H_spec.T, aspect="auto", origin="lower", cmap="viridis",
                    extent=[0, H_spec.shape[0],
                             spec_freqs[0], spec_freqs[-1]])
    ax.set_xlabel("channel"); ax.set_ylabel("frequency (Hz)")
    ax.set_title("(b) SCIENTIFIC pathway — harmonicity-spectrum trajectory")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="harmonicity")

    # (c) flux comparison
    ax = fig.add_subplot(gs[2, 0])
    f_c = az_cents.wasserstein.flux_
    f_s = az_spec.wasserstein.flux_
    f_c_n = f_c / (f_c.max() + 1e-9)
    f_s_n = f_s / (f_s.max() + 1e-9)
    ax.plot(f_c_n, color="#4C72B0", lw=1.3, label="cents (norm)")
    ax.plot(f_s_n, color="#55A868", lw=1.3, label="spectrum (norm)")
    ax.set_xlabel("channel transition"); ax.set_ylabel("normalised flux")
    from scipy.stats import spearmanr
    rho, p = spearmanr(f_c, f_s)
    ax.set_title(f"(c) Flux comparison  "
                  f"(Spearman ρ = {rho:.2f}, p = {p:.2g})", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)

    # (d) Markov state agreement
    ax = fig.add_subplot(gs[2, 1])
    lab_c = az_cents.markov.state_labels_
    lab_s = az_spec.markov.state_labels_
    T = len(lab_c)
    cmap_st = plt.get_cmap("tab10")
    for t in range(T):
        ax.add_patch(plt.Rectangle((t, 0.55), 1, 0.4,
                                    color=cmap_st(int(lab_c[t]) % 10)))
        ax.add_patch(plt.Rectangle((t, 0.05), 1, 0.4,
                                    color=cmap_st(int(lab_s[t]) % 10)))
    from sklearn.metrics import adjusted_rand_score
    ari = adjusted_rand_score(lab_c, lab_s)
    ax.set_xlim(0, T); ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.75]); ax.set_yticklabels(["spectrum", "cents"])
    ax.set_xlabel("channel")
    ax.set_title(f"(d) Markov label agreement  "
                  f"(adjusted Rand index = {ari:.2f})", fontsize=9.5)

    # (e) PCA scatter for both
    ax = fig.add_subplot(gs[2, 2])
    Zc = az_cents.latent.trajectory()
    Zs = az_spec.latent.trajectory()
    Zc_n = (Zc - Zc.mean(axis=0)) / (Zc.std(axis=0) + 1e-9)
    Zs_n = (Zs - Zs.mean(axis=0)) / (Zs.std(axis=0) + 1e-9)
    ax.scatter(Zc_n[:, 0], Zc_n[:, 1], s=28, c="#4C72B0", alpha=0.6,
                label="cents path", edgecolor="none")
    ax.scatter(Zs_n[:, 0], Zs_n[:, 1], s=28, c="#55A868", alpha=0.6,
                label="spectrum path", edgecolor="none")
    ax.set_xlabel("PC1 (z-scored)"); ax.set_ylabel("PC2 (z-scored)")
    ax.set_title("(e) Latent overlay  "
                  f"(cents {az_cents.latent.explained_variance_ratio_[:2].sum():.0%}, "
                  f"spec {az_spec.latent.explained_variance_ratio_[:2].sum():.0%})",
                  fontsize=9.5)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        "Figure 20 — Experiment C: same EEG, two pathways "
        "(musical cents vs scientific harmonicity-spectrum)",
        fontsize=12.5, y=0.985)
    fig.savefig(OUT / "20_real_eeg_two_pathways.png", bbox_inches="tight")
    plt.close(fig)
    print("    saved 20_real_eeg_two_pathways.png")


# ═════════════════════════════════════════════════════════════════════════════
# D. Frequency-band stratification on EEG_example.npy
# ═════════════════════════════════════════════════════════════════════════════

def run_D():
    print("\n[D] Frequency-band stratification (delta/theta/alpha/beta/gamma)")
    cache = CACHE_DIR / "_tier1_D_cache.npz"
    eeg = np.load(EEG_PATH)
    N_CHANNELS = 30
    CHANNEL_START = 25
    channels = list(range(CHANNEL_START, CHANNEL_START + N_CHANNELS))

    BANDS = [
        ("delta", 1, 4,   [[1, 2], [2, 3], [3, 4]]),
        ("theta", 4, 8,   [[4, 5], [5, 6], [6, 7], [7, 8]]),
        ("alpha", 8, 12,  [[8, 9], [9, 10], [10, 11], [11, 12]]),
        ("beta",  12, 30, [[12, 16], [16, 20], [20, 24], [24, 30]]),
        ("gamma", 30, 45, [[30, 35], [35, 40], [40, 45]]),
    ]

    if cache.exists():
        npz = np.load(cache, allow_pickle=True)
        results = {b[0]: {"ratios": list(npz[f"{b[0]}_ratios"]),
                           "metrics": list(npz[f"{b[0]}_metrics"])}
                   for b in BANDS}
        print(f"    loaded cache from {cache.name}")
    else:
        results = {}
        for name, low, high, freq_bands in BANDS:
            print(f"    extracting {name}-band [{low}-{high} Hz]"
                  f" on {N_CHANNELS} channels...")
            t0 = time.time()
            rats, mets = [], []
            for ch in channels:
                filt = _bandpass(eeg[ch], low, high, SF)
                _, r, m = _safe_extract(filt, SF, freq_bands=freq_bands,
                                         max_freq=high + 2, n_peaks=4,
                                         precision=0.5)
                rats.append(r); mets.append(m)
            results[name] = {"ratios": rats, "metrics": mets}
            print(f"      {name} done ({time.time()-t0:.1f}s, "
                  f"{sum(1 for r in rats if r)}/{N_CHANNELS} valid)")

        np.savez(cache, **{f"{n}_ratios": np.array(results[n]["ratios"],
                                                      dtype=object)
                              for n in [b[0] for b in BANDS]},
                          **{f"{n}_metrics": np.array(results[n]["metrics"],
                                                      dtype=object)
                              for n in [b[0] for b in BANDS]})
        print(f"    cached to {cache.name}")

    # Fit one analyzer per band
    fingerprints = {}
    fitted = {}
    for name, _low, _high, _ in BANDS:
        ratios = results[name]["ratios"]
        metrics = results[name]["metrics"]
        bts = [StubBt(np.array([]), SF, [], r, m)
               for r, m in zip(ratios, metrics)]
        az = HarmonicSequenceAnalyzer.from_biotuner_list(
            bts, tuning="peaks_ratios", n_hist_bins=240,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            az.fit_all(n_states="auto", auto_k_range=(2, 5))
        fitted[name] = az
        flux = az.wasserstein.flux_
        beta = az.topology.betti_numbers_ if az.topology else np.array([0, 0])
        evr = (az.latent.explained_variance_ratio_[:2].sum()
               if az.latent else 0.0)
        hs = np.array([m.get("harmsim", np.nan) for m in metrics],
                      dtype=float)
        hs = hs[np.isfinite(hs)]
        fingerprints[name] = {
            "flux_mean": float(flux.mean()),
            "K_markov": float(az.markov.n_states),
            "H_markov": float(az.markov.transition_entropy_),
            "var_PC1+2": float(evr),
            "beta0": float(beta[0]),
            "vocab": float(len(az.grammar.vocabulary_)) if az.grammar else 0,
            "H_grammar": (float(az.grammar.transition_entropy_)
                          if az.grammar else 0.0),
            "harmsim_mean": float(hs.mean()) if len(hs) else 0.0,
            "harmsim_std": float(hs.std()) if len(hs) else 0.0,
        }
        print(f"    {name}: {fingerprints[name]}")

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(3, 5, height_ratios=[1.4, 1.1, 1.1],
                           hspace=0.55, wspace=0.35,
                           left=0.05, right=0.97, top=0.93, bottom=0.07)

    band_names = [b[0] for b in BANDS]
    band_colours = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    # Row 1: per-band cents-histogram heatmaps
    for col, name in enumerate(band_names):
        ax = fig.add_subplot(gs[0, col])
        H = fitted[name].histograms
        ax.imshow(H.T, aspect="auto", origin="lower", cmap="magma",
                   extent=[0, H.shape[0], 0, 1200])
        ax.set_title(f"{name}  [{BANDS[col][1]}-{BANDS[col][2]} Hz]")
        ax.set_xlabel("channel")
        if col == 0:
            ax.set_ylabel("cents")
        else:
            ax.set_yticklabels([])

    # Row 2: 4 fingerprint metrics across bands
    chosen_metrics = ["H_markov", "flux_mean", "beta0", "harmsim_mean"]
    titles_m = [
        r"(b) Markov transition entropy $H$ (bits)",
        "(c) Mean $W_1$ flux (cents-bin units)",
        r"(d) Topological basin count $\beta_0$",
        "(e) Mean harmsim across channels",
    ]
    for col, (mkey, title) in enumerate(zip(chosen_metrics, titles_m)):
        ax = fig.add_subplot(gs[1, col])
        vals = [fingerprints[n][mkey] for n in band_names]
        bars = ax.bar(band_names, vals, color=band_colours)
        ax.set_title(title, fontsize=9.5)
        ax.tick_params(axis="x", labelsize=8)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, v + 0.02 * max(vals),
                     f"{v:.2g}", ha="center", fontsize=8)

    # Empty last cell row 2 - put grammar entropy there
    ax = fig.add_subplot(gs[1, 4])
    vals = [fingerprints[n]["H_grammar"] for n in band_names]
    bars = ax.bar(band_names, vals, color=band_colours)
    ax.set_title(r"(f) Grammar entropy $H_g$ (bits)", fontsize=9.5)
    ax.tick_params(axis="x", labelsize=8)
    for b, v in zip(bars, vals):
        if max(vals) > 0:
            ax.text(b.get_x() + b.get_width()/2, v + 0.02 * max(vals),
                     f"{v:.2g}", ha="center", fontsize=8)

    # Row 3: heatmap of all fingerprints
    ax = fig.add_subplot(gs[2, :])
    full_metrics = list(fingerprints[band_names[0]].keys())
    M = np.array([[fingerprints[b][m] for m in full_metrics]
                  for b in band_names])
    Mn = (M - M.min(axis=0)) / (M.max(axis=0) - M.min(axis=0) + 1e-9)
    im = ax.imshow(Mn, aspect="auto", cmap="viridis")
    for i in range(Mn.shape[0]):
        for j in range(Mn.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2g}", ha="center", va="center",
                     fontsize=7.5,
                     color="white" if Mn[i, j] > 0.55 else "black")
    ax.set_xticks(range(len(full_metrics)))
    ax.set_xticklabels(full_metrics, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(band_names)))
    ax.set_yticklabels(band_names, fontsize=9)
    ax.set_title("(g) Per-band fingerprint matrix  "
                  "(raw values shown, colour = column-normalised)")

    fig.suptitle("Figure 21 — Experiment D: per-band stratification of "
                  f"{N_CHANNELS} EEG channels", fontsize=12.5, y=0.985)
    fig.savefig(OUT / "21_real_eeg_band_stratification.png",
                 bbox_inches="tight")
    plt.close(fig)
    print("    saved 21_real_eeg_band_stratification.png")


# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS = {"A": run_A, "B": run_B, "C": run_C, "D": run_D}


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "all":
        for k in ["A", "B", "C", "D"]:
            EXPERIMENTS[k]()
    elif which in EXPERIMENTS:
        EXPERIMENTS[which]()
    else:
        print(f"Usage: {sys.argv[0]} [A|B|C|D|all]")
        sys.exit(2)
    print(f"\nAll requested experiments done. Figures in {OUT}")
