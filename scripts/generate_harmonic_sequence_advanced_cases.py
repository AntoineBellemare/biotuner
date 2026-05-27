"""
generate_harmonic_sequence_advanced_cases.py
============================================
Companion script producing the "showcase & applications" figures used in the
second half of docs/papers/harmonic_sequence_use_cases.md.

Five scenarios capture qualitatively different harmonic dynamics:

    A. STABLE         — meditation baseline: same tuning + noise
    B. DRIFT          — slow continuous deformation (major -> minor)
    C. CYCLE          — categorical regime cycling (major <-> minor <-> harm7)
    D. OSCILLATION    — sinusoidal blend between two tunings (DMD goldmine)
    E. PERTURBED      — long stable epoch + abrupt event

Each scenario is run through the full HarmonicSequenceAnalyzer, and we draw
side-by-side comparisons highlighting what each modelling approach reveals.

Run:
    python scripts/generate_harmonic_sequence_advanced_cases.py
"""
from __future__ import annotations

import sys
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

from biotuner.harmonic_sequence import HarmonicSequenceAnalyzer

OUT = ROOT / "docs" / "papers" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 140, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
})

# Note: the unison 1.0 maps to cents=0, which sits on the lower histogram
# edge. Any negative-direction jitter (~half of all draws) would push it
# below 0 cents and get silently dropped by the histogram, distorting
# per-row normalisation and inflating Wasserstein flux. We drop the unison
# from all base scales — it is included automatically by histogram_to_ratios
# at decode time anyway.
JUST_MAJOR = [9/8, 5/4, 4/3, 3/2, 5/3, 15/8]
JUST_MINOR = [9/8, 6/5, 4/3, 3/2, 8/5, 9/5]
HARM_7     = [9/8, 5/4, 11/8, 3/2, 7/4, 15/8]


# ─────────────────────────────────────────────────────────────────────────────
# Scenario constructors
# ─────────────────────────────────────────────────────────────────────────────

def _jitter(rng, base, sigma=0.0003):
    return [float(x * (1.0 + sigma * rng.standard_normal())) for x in base]


def scenario_stable(T: int = 32, seed: int = 0):
    """A. Stable: same just-major tuning every frame, tiny noise."""
    rng = np.random.default_rng(seed)
    return [_jitter(rng, JUST_MAJOR) for _ in range(T)]


def scenario_drift(T: int = 32, seed: int = 0):
    """B. Drift: continuous linear interpolation JUST_MAJOR -> JUST_MINOR."""
    rng = np.random.default_rng(seed)
    out = []
    for t in range(T):
        a = t / (T - 1)
        base = [(1 - a) * x + a * y for x, y in zip(JUST_MAJOR, JUST_MINOR)]
        out.append(_jitter(rng, base))
    return out


def scenario_cycle(T: int = 36, seed: int = 0):
    """C. Categorical cycling: 3 frames each of major / minor / harm7, looped."""
    rng = np.random.default_rng(seed)
    regimes = [JUST_MAJOR, JUST_MINOR, HARM_7]
    out = []
    for c in range(T // 9):
        for base in regimes:
            for _ in range(3):
                out.append(_jitter(rng, base))
    return out


def scenario_oscillation(T: int = 48, period: int = 8, seed: int = 0):
    """D. Sinusoidal blend major <-> minor with given period (in frames)."""
    rng = np.random.default_rng(seed)
    out = []
    for t in range(T):
        a = 0.5 * (1 + np.sin(2 * np.pi * t / period))
        base = [(1 - a) * x + a * y for x, y in zip(JUST_MAJOR, JUST_MINOR)]
        out.append(_jitter(rng, base))
    return out


def scenario_perturbed(T: int = 40, event_at: int = 24, seed: int = 0):
    """E. Stable baseline then abrupt switch to a different tuning."""
    rng = np.random.default_rng(seed)
    out = []
    for t in range(T):
        base = JUST_MAJOR if t < event_at else HARM_7
        out.append(_jitter(rng, base))
    return out


SCENARIOS = {
    "Stable": scenario_stable(),
    "Drift": scenario_drift(),
    "Cycle": scenario_cycle(),
    "Oscillation": scenario_oscillation(),
    "Perturbed": scenario_perturbed(),
}
SCEN_COLOURS = {
    "Stable":      "#4C72B0",
    "Drift":       "#55A868",
    "Cycle":       "#C44E52",
    "Oscillation": "#8172B2",
    "Perturbed":   "#CCB974",
}


def fit_analyzer(ratios_list, n_states="auto", auto_k_range=(2, 6)):
    az = HarmonicSequenceAnalyzer.from_ratios_list(ratios_list, n_hist_bins=240)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Fit per approach with sensible per-scenario parameters
        az.fit_markov(n_states=n_states, auto_k_range=auto_k_range)
        az.fit_wasserstein()
        az.fit_latent(latent_dim=3)
        try:
            az.fit_topology(scalar_key="harmsim", embedding_dim=3, delay=1)
        except Exception:
            pass
        try:
            az.fit_dmd(use_histograms=True)
        except Exception:
            pass
        try:
            az.fit_grammar(n_gram=2)
        except Exception:
            pass
    return az


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8 — Scenarios overview (4 scenarios x 3 panels each)
# ─────────────────────────────────────────────────────────────────────────────

def fig_scenarios_overview():
    scenario_names = ["Stable", "Drift", "Cycle", "Oscillation"]
    # First pass: fit all analyzers with a FIXED K=3 (apples-to-apples
    # transition entropy across scenarios).
    fits = {}
    flux_max_all = 0.0
    for name in scenario_names:
        az = fit_analyzer(SCENARIOS[name], n_states=3)
        fits[name] = az
        flux_max_all = max(flux_max_all, float(az.wasserstein.flux_.max()))

    fig, axes = plt.subplots(
        nrows=len(scenario_names), ncols=3,
        figsize=(16, 13),
        gridspec_kw={"width_ratios": [2.4, 2.6, 1.8],
                     "hspace": 0.70, "wspace": 0.30},
    )
    for row, name in enumerate(scenario_names):
        ratios = SCENARIOS[name]
        az = fits[name]
        H = az.histograms
        T = H.shape[0]
        flux = az.wasserstein.flux_
        labels = az.markov.state_labels_

        # column 0: histogram heatmap
        ax = axes[row, 0]
        ax.imshow(H.T, aspect="auto", origin="lower", cmap="magma",
                   extent=[0, T, 0, 1200])
        ax.set_ylabel("cents")
        ax.set_title(f"{name}: encoded histograms", fontsize=10)
        if row == len(scenario_names) - 1:
            ax.set_xlabel("frame")

        # column 1: flux + Markov state strip (shared y-axis across rows)
        ax = axes[row, 1]
        ax.plot(np.arange(len(flux)), flux, color="black", lw=1.4)
        ax.set_ylabel("$W_1$ flux (bin units)")
        ax.set_title(f"{name}: flux + Markov states  "
                      f"(K={az.markov.n_states}, "
                      f"H={az.markov.transition_entropy_:.2f} bits)", fontsize=10)
        if row == len(scenario_names) - 1:
            ax.set_xlabel("frame")

        # Overlay Markov state strip below the flux line; share y-scale
        strip_h = max(flux_max_all * 0.10, 0.5)
        ymin = -strip_h * 1.4
        cmap_st = plt.get_cmap("tab10")
        for t in range(T):
            ax.add_patch(plt.Rectangle((t - 0.5, ymin), 1, strip_h,
                                        color=cmap_st(int(labels[t]) % 10)))
        ax.set_ylim(ymin - strip_h * 0.2, flux_max_all * 1.10)
        ax.set_xlim(0, T)

        # column 2: latent 2D scatter
        ax = axes[row, 2]
        Z = az.latent.trajectory()
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=np.arange(T), cmap="viridis",
                         s=32, edgecolor="white", lw=0.5)
        ax.plot(Z[:, 0], Z[:, 1], color="grey", lw=0.4, alpha=0.4)
        evr_sum = az.latent.explained_variance_ratio_[:2].sum()
        ax.set_title(f"{name}: PC1-PC2  ({evr_sum:.0%} var)", fontsize=10)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        if row == 0:
            cb = fig.colorbar(sc, ax=ax, fraction=0.06, pad=0.04)
            cb.set_label("frame", fontsize=8)

    fig.suptitle("Figure 8 — Four scenarios, four signatures",
                  fontsize=12, y=0.995)
    fig.savefig(OUT / "08_scenarios_overview.png", bbox_inches="tight")
    plt.close(fig)
    print("  saved 08_scenarios_overview.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 9 — DMD on an oscillatory signal
# ─────────────────────────────────────────────────────────────────────────────

def fig_dmd_oscillation():
    ratios = scenario_oscillation(T=64, period=8, seed=1)
    az = HarmonicSequenceAnalyzer.from_ratios_list(ratios)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        az.fit_dmd(use_histograms=True)
        az.fit_latent(latent_dim=3)
    dmd = az.dmd
    eigs = dmd.eigenvalues_

    fig = plt.figure(figsize=(15, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.5, 1.4], wspace=0.45,
                           left=0.07, right=0.97, bottom=0.16, top=0.85)

    # (a) eigenvalues on the unit circle
    ax = fig.add_subplot(gs[0, 0])
    theta = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(theta), np.sin(theta), color="grey", lw=0.8, ls="--")
    osc, idx = dmd.oscillatory_modes(threshold=0.1)
    is_osc = np.zeros(len(eigs), dtype=bool); is_osc[idx] = True
    ax.scatter(eigs[~is_osc].real, eigs[~is_osc].imag,
                s=70, color="#888", alpha=0.7, label="decaying")
    ax.scatter(eigs[is_osc].real, eigs[is_osc].imag,
                s=140, color="crimson", marker="*",
                label=f"on unit circle ({len(idx)})")
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
    ax.set_aspect("equal")
    ax.axhline(0, color="grey", lw=0.4); ax.axvline(0, color="grey", lw=0.4)
    ax.set_xlabel(r"Re($\lambda$)"); ax.set_ylabel(r"Im($\lambda$)")
    ax.set_title("(a) DMD spectrum")
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    # (b) reconstructed vs observed first PC over time
    ax = fig.add_subplot(gs[0, 1])
    # ground-truth: average cents (centroid of histogram), as a 1-d signal
    H = az.histograms
    bin_centers_cents = (np.arange(H.shape[1]) + 0.5) * (1200 / H.shape[1])
    centroid = (H * bin_centers_cents).sum(axis=1) / np.maximum(H.sum(axis=1), 1e-9)
    ax.plot(centroid, color="black", lw=1.3, label="observed centroid")
    # forecast 16 steps from the end and overlay
    n_fwd = 16
    X_pred = dmd.reconstruct(n_steps=n_fwd)             # (16, 10) in PCA space
    # Invert PCA via the stored PCA via a small helper:
    # we don't have direct access; reconstruct returns the same space DMD was fit in.
    # Build a faithful 1-D summary by projecting both observed PCA and predicted PCA
    # onto the leading PC and re-using the centroid-correlated direction.
    # Simpler: compute "predicted centroid" from the reconstruction's first column.
    pred_signal = X_pred[:, 0]
    # Align scales so they overlay visibly
    obs_pc1 = (H - H.mean(axis=0)).dot(
        np.linalg.svd(H - H.mean(axis=0), full_matrices=False)[2][0]
    )
    obs_pc1_norm = (obs_pc1 - obs_pc1.mean()) / (obs_pc1.std() + 1e-9)
    pred_norm = (pred_signal - pred_signal.mean()) / (pred_signal.std() + 1e-9)
    ax2 = ax.twinx()
    ax2.plot(obs_pc1_norm, color="#888", lw=1.0, ls="--", alpha=0.8,
              label="observed PC1 (scaled)")
    ax2.plot(np.arange(len(obs_pc1_norm),
                       len(obs_pc1_norm) + n_fwd),
              pred_norm, color="crimson", lw=1.6, label="DMD forecast")
    ax2.axvline(len(obs_pc1_norm) - 0.5, color="black", lw=0.7, ls=":")
    ax2.set_ylabel("PC1 (z-score)")
    ax.set_xlabel("frame")
    ax.set_ylabel("centroid (cents)")
    ax.set_title("(b) Observed centroid + 16-step DMD forecast on PC1")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper left")

    # (c) period estimate from dominant eigenvalue
    ax = fig.add_subplot(gs[0, 2])
    # Sort eigenvalues by closeness to unit circle
    dist = np.abs(np.abs(eigs) - 1.0)
    order = np.argsort(dist)
    top = order[:4]
    ax.scatter(np.abs(eigs[top]), np.abs(dmd.frequencies_[top]),
                s=80, c="crimson")
    # Annotate inferred period (frames per cycle)
    for i, k in enumerate(top):
        freq = np.abs(dmd.frequencies_[k])
        if freq > 1e-6:
            period_est = 2 * np.pi / freq
            ax.annotate(f"mode {k}: period ≈ {period_est:.1f} frames",
                         (np.abs(eigs[k]), np.abs(dmd.frequencies_[k])),
                         xytext=(8, 0), textcoords="offset points", fontsize=8)
    ax.set_xlabel(r"$|\lambda|$  (closer to 1 = persistent)")
    ax.set_ylabel(r"$|\mathrm{Im}\log\lambda|$  (oscillation rate)")
    ax.set_title("(c) Top oscillatory modes and inferred periods")

    fig.suptitle("Figure 9 — DMD finds the 8-frame oscillation in the data",
                  fontsize=12, y=0.96)
    fig.savefig(OUT / "09_dmd_oscillation.png", bbox_inches="tight")
    plt.close(fig)
    print("  saved 09_dmd_oscillation.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 10 — Topology: stable vs cycling vs oscillating
# ─────────────────────────────────────────────────────────────────────────────

def fig_topology_scenarios():
    scenario_pick = ["Stable", "Cycle", "Oscillation"]
    fig, axes = plt.subplots(2, len(scenario_pick), figsize=(15, 9),
                              gridspec_kw={"hspace": 0.40, "wspace": 0.35,
                                           "left": 0.07, "right": 0.97,
                                           "top": 0.92, "bottom": 0.08})
    for col, name in enumerate(scenario_pick):
        ratios = (scenario_oscillation(T=64, period=8) if name == "Oscillation"
                   else SCENARIOS[name])
        az = HarmonicSequenceAnalyzer.from_ratios_list(ratios)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            az.fit_topology(scalar_key="harmsim", embedding_dim=3, delay=1)

        top = az.topology
        cloud = top.takens_embedding_
        beta = top.betti_numbers_
        fp = top.session_fingerprint()

        ax = axes[0, col]
        sc = ax.scatter(cloud[:, 0], cloud[:, 1],
                         c=np.arange(len(cloud)), cmap="viridis",
                         s=42, edgecolor="white", lw=0.6)
        ax.plot(cloud[:, 0], cloud[:, 1], color="grey", lw=0.4, alpha=0.4)
        ax.set_title(f"{name}: Takens cloud  (β0={beta[0]})", fontsize=10)
        ax.set_xlabel("x(t)"); ax.set_ylabel("x(t+1)")
        if col == len(scenario_pick) - 1:
            cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label("frame", fontsize=8)

        ax = axes[1, col]
        diagrams = top.persistence_diagram_
        max_finite = 0.0
        for dim, dgm in enumerate(diagrams):
            finite = dgm[np.isfinite(dgm[:, 1])]
            if len(finite):
                max_finite = max(max_finite, finite[:, 1].max())
                ax.scatter(finite[:, 0], finite[:, 1], s=35, alpha=0.8,
                            label=f"H{dim}")
        lims = [0, max_finite * 1.1 if max_finite > 0 else 1]
        ax.plot(lims, lims, color="grey", lw=0.5, ls="--")
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel("birth"); ax.set_ylabel("death")
        ax.set_title(
            f"Persistence  (fp = {np.round(fp[:3], 2).tolist()})",
            fontsize=9,
        )
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Figure 10 — Topology distinguishes stable / cyclic / "
                  "oscillating dynamics",
                  fontsize=12, y=0.995)
    fig.savefig(OUT / "10_topology_scenarios.png", bbox_inches="tight")
    plt.close(fig)
    print("  saved 10_topology_scenarios.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 11 — Application: stimulus / event detection (PERTURBED scenario)
# ─────────────────────────────────────────────────────────────────────────────

def fig_application_event_detection():
    EVENT_AT = 24
    ratios = scenario_perturbed(T=40, event_at=EVENT_AT, seed=3)
    az = fit_analyzer(ratios, n_states=2)
    H = az.histograms
    T = H.shape[0]
    flux = az.wasserstein.flux_
    labels = az.markov.state_labels_

    # Build a simple z-score threshold detector
    z = (flux - flux.mean()) / (flux.std() + 1e-9)
    detected = np.where(z > 2.0)[0]

    fig, axes = plt.subplots(3, 1, figsize=(11, 7.0),
                              gridspec_kw={"height_ratios": [2.0, 1.4, 0.7],
                                           "hspace": 0.35})

    # (a) histogram heatmap with event marker
    ax = axes[0]
    im = ax.imshow(H.T, aspect="auto", origin="lower", cmap="magma",
                    extent=[0, T, 0, 1200])
    ax.axvline(EVENT_AT, color="cyan", lw=1.6, ls="--", alpha=0.9,
                label=f"ground-truth event @ t={EVENT_AT}")
    for d in detected:
        ax.axvline(d + 0.5, color="red", lw=1.0, ls=":", alpha=0.8)
    ax.set_ylabel("cents")
    ax.set_title("(a) Encoded histograms  (cyan = ground-truth event, "
                  "red dotted = z(flux)>2)")
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.colorbar(im, ax=ax, pad=0.01)

    # (b) flux time series with detection threshold
    ax = axes[1]
    ax.plot(np.arange(len(flux)), flux, color="black", lw=1.4)
    threshold = flux.mean() + 2.0 * flux.std()
    ax.axhline(threshold, color="red", lw=0.8, ls="--",
                label=f"μ + 2σ = {threshold:.2f}")
    ax.axvline(EVENT_AT - 0.5, color="cyan", lw=1.4, ls="--", alpha=0.9)
    for d in detected:
        ax.scatter(d, flux[d], s=80, color="red", zorder=5)
    ax.set_xlim(0, T)
    ax.set_ylabel("$W_1$ flux")
    ax.set_xlabel("")
    ax.set_title(f"(b) Wasserstein flux detects the event "
                  f"(detection latency: {detected[0] - EVENT_AT + 1 if len(detected) else 'N/A'} frames)")
    ax.legend(loc="upper left", frameon=False, fontsize=8)

    # (c) Markov state strip
    ax = axes[2]
    cmap_st = plt.get_cmap("tab10")
    for t in range(T):
        ax.add_patch(plt.Rectangle((t, 0), 1, 1,
                                    color=cmap_st(int(labels[t]) % 10)))
    ax.axvline(EVENT_AT, color="cyan", lw=1.6, ls="--", alpha=0.9)
    ax.set_xlim(0, T); ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_xlabel("frame"); ax.set_title("(c) Discovered Markov states")

    fig.suptitle("Figure 11 — Application: detecting a stimulus event "
                  "from harmonic flux", fontsize=12, y=0.995)
    fig.savefig(OUT / "11_application_event_detection.png", bbox_inches="tight")
    plt.close(fig)
    print("  saved 11_application_event_detection.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 12 — Application: session fingerprinting across scenarios
# ─────────────────────────────────────────────────────────────────────────────

def fig_application_fingerprinting():
    """Compute compact per-session fingerprints, then plot them as a grid."""
    rows = []
    for name in ["Stable", "Drift", "Cycle", "Oscillation", "Perturbed"]:
        ratios = SCENARIOS[name]
        az = fit_analyzer(ratios, n_states="auto", auto_k_range=(2, 5))
        flux = az.wasserstein.flux_
        beta = az.topology.betti_numbers_
        evr = az.latent.explained_variance_ratio_
        eigs_abs = np.abs(az.dmd.eigenvalues_) if az.dmd is not None else np.array([])
        osc_frac = float((np.abs(eigs_abs - 1.0) < 0.1).mean()) if len(eigs_abs) else 0.0
        rows.append([
            float(flux.mean()),
            float(flux.std()),
            float(az.markov.n_states),
            float(az.markov.transition_entropy_),
            float(evr[:2].sum()),
            float(beta[0]),
            float(osc_frac),
            float(az.grammar.transition_entropy_) if az.grammar is not None else 0.0,
        ])
    rows = np.array(rows)
    metric_names = [
        "flux_mean", "flux_std", "K_markov", "H_markov",
        "var_PC1+2", "beta0", "osc_frac", "H_grammar",
    ]
    scen_names = list(SCENARIOS.keys())

    # Normalise per column for fair colour scaling
    rows_n = (rows - rows.min(axis=0)) / (rows.max(axis=0) - rows.min(axis=0) + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2),
                              gridspec_kw={"width_ratios": [2.0, 1.2],
                                           "wspace": 0.35})
    ax = axes[0]
    im = ax.imshow(rows_n, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(scen_names)))
    ax.set_yticklabels(scen_names)
    # Print raw values on cells
    for i in range(rows.shape[0]):
        for j in range(rows.shape[1]):
            txt = f"{rows[i, j]:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                     color="white" if rows_n[i, j] > 0.55 else "black")
    ax.set_title("(a) Per-scenario fingerprints  (raw values shown, "
                  "colour = column-normalised)")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)

    # Radar plot of 4 chosen metrics for visual distinction
    ax = axes[1]
    chosen = ["flux_mean", "H_markov", "var_PC1+2", "beta0"]
    idx = [metric_names.index(c) for c in chosen]
    angles = np.linspace(0, 2 * np.pi, len(chosen) + 1)
    ax = fig.add_subplot(1, 2, 2, projection="polar")
    fig.delaxes(axes[1])
    for i, name in enumerate(scen_names):
        vals = rows_n[i, idx].tolist() + [rows_n[i, idx[0]]]
        ax.plot(angles, vals, color=SCEN_COLOURS[name], lw=1.4, label=name)
        ax.fill(angles, vals, color=SCEN_COLOURS[name], alpha=0.10)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(chosen, fontsize=8)
    ax.set_yticklabels([])
    ax.set_title("(b) Normalised fingerprint radar", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.05),
               fontsize=8, frameon=False)

    fig.suptitle("Figure 12 — Application: scenario fingerprints "
                  "distinguish recording types", fontsize=12, y=1.02)
    fig.savefig(OUT / "12_application_fingerprinting.png", bbox_inches="tight")
    plt.close(fig)
    print("  saved 12_application_fingerprinting.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 13 — Application: cross-session similarity (W1 + Levenshtein)
# ─────────────────────────────────────────────────────────────────────────────

def fig_application_similarity():
    """Build a 5x5 similarity matrix between sessions using two metrics."""
    scen_names = list(SCENARIOS.keys())
    n = len(scen_names)

    # Fit analyzers and align histogram lengths (truncate to min T)
    azs = {name: fit_analyzer(SCENARIOS[name], n_states="auto",
                                auto_k_range=(2, 5)) for name in scen_names}
    Ts = [az.histograms.shape[0] for az in azs.values()]
    T_min = min(Ts)

    # Mean Wasserstein between full histograms via centroid distance:
    # Use the mean of per-row Wasserstein already in distance_matrix_
    # Cross-session: build a pairwise mean-histogram W1.
    from scipy.stats import wasserstein_distance
    mean_hists = {name: azs[name].histograms[:T_min].mean(axis=0)
                  for name in scen_names}
    bin_centers = np.arange(240, dtype=float)
    W = np.zeros((n, n))
    for i, ni in enumerate(scen_names):
        for j, nj in enumerate(scen_names):
            W[i, j] = wasserstein_distance(bin_centers, bin_centers,
                                            mean_hists[ni], mean_hists[nj])

    # Levenshtein on grammar chord sequences (truncate to T_min)
    L = np.zeros((n, n))
    for i, ni in enumerate(scen_names):
        for j, nj in enumerate(scen_names):
            si = azs[ni].grammar.chord_sequence_[:T_min]
            sj = azs[nj].grammar.chord_sequence_[:T_min]
            L[i, j] = azs[ni].grammar.levenshtein(si, sj)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3),
                              gridspec_kw={"wspace": 0.4})
    for ax, M, title, cmap_name in [
        (axes[0], W, "Mean-histogram $W_1$ distance", "viridis"),
        (axes[1], L, "Grammar chord-sequence Levenshtein", "plasma"),
    ]:
        im = ax.imshow(M, cmap=cmap_name)
        ax.set_xticks(range(n)); ax.set_xticklabels(scen_names, rotation=35,
                                                      ha="right", fontsize=9)
        ax.set_yticks(range(n)); ax.set_yticklabels(scen_names, fontsize=9)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{M[i, j]:.1f}",
                         ha="center", va="center", fontsize=8,
                         color="white" if (M[i, j] > M.max() * 0.55) else "black")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Figure 13 — Application: pairwise session similarity "
                  "via two complementary metrics", fontsize=12, y=1.02)
    fig.savefig(OUT / "13_application_similarity.png", bbox_inches="tight")
    plt.close(fig)
    print("  saved 13_application_similarity.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 14 — Application: latent-space morphing between two recorded states
# ─────────────────────────────────────────────────────────────────────────────

def fig_application_morphing():
    """Pick two anchor frames in the Cycle scenario and morph between them
    using BOTH Wasserstein interpolation and latent-space interpolation."""
    ratios = SCENARIOS["Cycle"]
    az = fit_analyzer(ratios, n_states=3)
    T = az.histograms.shape[0]
    # The cycle is 9-periodic (3 frames major, 3 minor, 3 harm7).
    # Pick anchors from different regimes within the same cycle:
    t1, t2 = 1, 7    # major-frame -> harm7-frame
    n_steps = 12
    H_w = az.get_histograms(source="wasserstein_interp",
                             t1=t1, t2=t2, n_steps=n_steps)
    H_l = az.get_histograms(source="latent_interp",
                             t1=t1, t2=t2, n_steps=n_steps)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.2),
                              gridspec_kw={"hspace": 0.45})
    for ax, H, title in [
        (axes[0], H_w, "(a) Wasserstein-quantile interpolation"),
        (axes[1], H_l, "(b) Latent (PCA) interpolation"),
    ]:
        im = ax.imshow(H.T, aspect="auto", origin="lower", cmap="magma",
                        extent=[0, n_steps, 0, 1200])
        ax.set_ylabel("cents")
        ax.set_xlabel(f"interpolation step  ({t1} -> {t2})")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)

    fig.suptitle("Figure 14 — Application: morphing between two recorded "
                  "harmonic states  (two methods, same endpoints)",
                  fontsize=12, y=0.98)
    fig.savefig(OUT / "14_application_morphing.png", bbox_inches="tight")
    plt.close(fig)
    print("  saved 14_application_morphing.png")


# ─────────────────────────────────────────────────────────────────────────────

for fn in [
    fig_scenarios_overview,
    fig_dmd_oscillation,
    fig_topology_scenarios,
    fig_application_event_detection,
    fig_application_fingerprinting,
    fig_application_similarity,
    fig_application_morphing,
]:
    fn()

print("\nAll advanced figures generated into", OUT)
