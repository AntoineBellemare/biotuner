"""Generate a paper-ready evaluation report for the resonance-package refactor.

Five figures + a summary table — saved as both PNG (for quick view) and PDF
(for paper inclusion). All figures use a consistent biotuner color palette and
publication-grade typography settings.

Usage:
    python reports/resonance_refactor/generate_report.py
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "resonance"))

from _signals import SIGNALS, BASELINE_CONFIG, legacy_default_resonance_config_kwargs  # noqa: E402
from biotuner.resonance import compute_resonance, ResonanceConfig  # noqa: E402
from biotuner.resonance.combine import product, geomean, harmmean, min_combine  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)
SNAPSHOT_DIR = ROOT / "tests" / "resonance" / "snapshots"

# ---------------------------------------------------------------------------
# Publication-grade matplotlib settings
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.5,
})

# Biotuner palette — matches the legacy plotter colors
COLOR_H = "#1a237e"      # darkblue: harmonicity
COLOR_PC = "#6a1b9a"     # darkviolet: phase coupling
COLOR_R = "#b71c1c"      # darkred: resonance
COLOR_SIG = "#37474f"    # dark slate: input signal / PSD
COLOR_LEGACY = "#9e9e9e" # grey: legacy baseline (used in overlay diffs)
SIGNAL_LABELS = {
    "harmonic_5_10_20_40": "Harmonic (5/10/20/40 Hz)",
    "pink_noise": "Pink noise (1/f)",
    "inharmonic_7_11_18_23": "Inharmonic (7/11.3/17.9/23 Hz)",
}


def _save(fig, name):
    """Save a figure as both PNG and PDF."""
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    print(f"  wrote {png.name} + {pdf.name}")


def _compute_legacy_default(name):
    """Run the new orchestrator with the legacy-default config."""
    sig = SIGNALS[name](sf=BASELINE_CONFIG["fs"])
    cfg = ResonanceConfig(**legacy_default_resonance_config_kwargs())
    return compute_resonance(sig, sf=BASELINE_CONFIG["fs"], config=cfg), sig


# ---------------------------------------------------------------------------
# Figure 1 — bit-exact regression: legacy snapshots vs new orchestrator
# ---------------------------------------------------------------------------


def fig1_regression():
    print("Fig 1: regression overlay ...")
    names = list(SIGNALS.keys())
    fig, axes = plt.subplots(3, 3, figsize=(11, 7.5), sharex="col")

    for col, name in enumerate(names):
        result, _ = _compute_legacy_default(name)
        snap = np.load(SNAPSHOT_DIR / f"{name}.npz", allow_pickle=True)
        freqs = result.freqs

        for row, (factor_key, snap_key, color, label) in enumerate([
            ("H", "harmonicity", COLOR_H, "Harmonicity H(f)"),
            ("PC", "phase_coupling", COLOR_PC, "Phase coupling PC(f)"),
            (None, "resonance", COLOR_R, "Resonance R(f)"),
        ]):
            ax = axes[row, col]
            new_arr = result.resonance_spectrum if factor_key is None else result.factors[factor_key]
            old_arr = snap[snap_key]

            # Legacy as thick light line, new as thin dark line directly on top
            ax.plot(freqs, old_arr, color=COLOR_LEGACY, lw=3.0, alpha=0.55, label="Legacy")
            ax.plot(freqs, new_arr, color=color, lw=1.2, label="Refactor")

            diff = float(np.max(np.abs(new_arr - old_arr)))
            ax.text(0.97, 0.94, f"max |Δ| = {diff:.1e}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, family="monospace",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))

            if col == 0:
                ax.set_ylabel(label, color=color)
            if row == 0:
                ax.set_title(SIGNAL_LABELS[name], fontsize=11)
            if row == 2:
                ax.set_xlabel("Frequency (Hz)")
            if row == 0 and col == 0:
                ax.legend(loc="upper left", fontsize=8)

    fig.suptitle("Figure 1 — Bit-exact regression: refactored pipeline reproduces legacy output",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    _save(fig, "fig1_regression")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — H × PC = R decomposition with peak markers and complexity panel
# ---------------------------------------------------------------------------


def fig2_decomposition():
    print("Fig 2: H × PC = R decomposition ...")
    result, sig = _compute_legacy_default("harmonic_5_10_20_40")
    freqs = result.freqs

    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(4, 4, height_ratios=[1.2, 1, 1, 1], hspace=0.55, wspace=0.4)

    # Top row: raw signal (left half) + spectrogram-style PSD bars (right half)
    ax_sig = fig.add_subplot(gs[0, :2])
    t = np.arange(len(sig)) / BASELINE_CONFIG["fs"]
    show_t = t < 1.0
    ax_sig.plot(t[show_t], sig[show_t], color=COLOR_SIG, lw=0.8)
    ax_sig.set_xlabel("Time (s)")
    ax_sig.set_ylabel("Amplitude")
    ax_sig.set_title("a) Input signal (1 s window)")

    # PSD from the intermediates
    cfg = ResonanceConfig(return_intermediates=True, **legacy_default_resonance_config_kwargs())
    result_int = compute_resonance(sig, sf=BASELINE_CONFIG["fs"], config=cfg)
    psd = result_int.intermediates["psd_prob"]
    ax_psd = fig.add_subplot(gs[0, 2:])
    ax_psd.bar(freqs, psd, width=0.45, color=COLOR_SIG, alpha=0.85)
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("Probability mass")
    ax_psd.set_title("b) Normalized PSD (after 1/f removal)")

    # Rows 2-4: H, PC, R with peak markers
    for row, (key, color, label) in enumerate([
        ("H", COLOR_H, "Harmonicity H(f)"),
        ("PC", COLOR_PC, "Phase coupling PC(f)"),
        ("R", COLOR_R, "Resonance R(f) = H · PC"),
    ]):
        ax = fig.add_subplot(gs[row + 1, :])
        if key == "R":
            arr = result.resonance_spectrum
        else:
            arr = result.factors[key]
        ax.plot(freqs, arr, color=color, lw=1.6)
        ax.fill_between(freqs, 0, arr, color=color, alpha=0.12)

        # Mark peaks from result.peaks dict
        peak_freqs = result.peaks[key]
        for pf in peak_freqs:
            idx = np.argmin(np.abs(freqs - pf))
            ax.plot(pf, arr[idx], "o", color=color, ms=6,
                    markerfacecolor="white", markeredgewidth=1.6)
            ax.annotate(f"{pf:.1f} Hz", (pf, arr[idx]),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=8, color=color, fontweight="bold")

        # Summary stats in top-right corner
        s = result.summaries[key]
        stats = (
            f"avg={s['avg']:.3g}   max={s['max']:.3g}\n"
            f"flat={s['flatness']:.3g}   ent={s['entropy']:.3g}\n"
            f"hfd={s['higuchi']:.3g}   spread={s['spread']:.2f}"
        )
        ax.text(0.985, 0.95, stats, transform=ax.transAxes,
                ha="right", va="top", fontsize=8, family="monospace",
                bbox=dict(facecolor="white", edgecolor="lightgrey", alpha=0.9, pad=4))

        ax.set_ylabel(label, color=color)
        if row == 2:
            ax.set_xlabel("Frequency (Hz)")
        ax.set_title(f"{'cde'[row]}) {label}", loc="left")
        ax.set_xlim(freqs.min(), freqs.max())

    fig.suptitle("Figure 2 — H × PC = R decomposition on harmonic 1:2:4:8 signal",
                 fontsize=13, fontweight="bold", y=0.995)
    _save(fig, "fig2_decomposition")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — kernel comparison (harmsim vs subharm_tension on harmonic signal)
# ---------------------------------------------------------------------------


def fig3_kernel_comparison():
    print("Fig 3: kernel comparison ...")
    sig = SIGNALS["harmonic_5_10_20_40"](sf=BASELINE_CONFIG["fs"])

    cfg_kwargs = legacy_default_resonance_config_kwargs()
    # harmsim (legacy default)
    r_harmsim = compute_resonance(sig, sf=BASELINE_CONFIG["fs"],
                                   config=ResonanceConfig(**cfg_kwargs))
    # subharm_tension
    cfg_sub = dict(cfg_kwargs)
    cfg_sub["harmonic_kernel"] = "subharm_tension"
    r_sub = compute_resonance(sig, sf=BASELINE_CONFIG["fs"], config=ResonanceConfig(**cfg_sub))

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5))

    # Top row: H(f) for each kernel
    axes[0, 0].plot(r_harmsim.freqs, r_harmsim.factors["H"], color=COLOR_H, lw=1.6)
    axes[0, 0].fill_between(r_harmsim.freqs, 0, r_harmsim.factors["H"], color=COLOR_H, alpha=0.12)
    axes[0, 0].set_title("a) H(f) — harmsim kernel (dyad similarity)")
    axes[0, 0].set_ylabel("Harmonicity")

    axes[0, 1].plot(r_sub.freqs, r_sub.factors["H"], color=COLOR_H, lw=1.6)
    axes[0, 1].fill_between(r_sub.freqs, 0, r_sub.factors["H"], color=COLOR_H, alpha=0.12)
    axes[0, 1].set_title("b) H(f) — subharm_tension kernel")

    # Bottom row: R(f) for each kernel
    axes[1, 0].plot(r_harmsim.freqs, r_harmsim.resonance_spectrum, color=COLOR_R, lw=1.6)
    axes[1, 0].fill_between(r_harmsim.freqs, 0, r_harmsim.resonance_spectrum, color=COLOR_R, alpha=0.12)
    axes[1, 0].set_title("c) R(f) — harmsim kernel")
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("Resonance")

    axes[1, 1].plot(r_sub.freqs, r_sub.resonance_spectrum, color=COLOR_R, lw=1.6)
    axes[1, 1].fill_between(r_sub.freqs, 0, r_sub.resonance_spectrum, color=COLOR_R, alpha=0.12)
    axes[1, 1].set_title("d) R(f) — subharm_tension kernel")
    axes[1, 1].set_xlabel("Frequency (Hz)")

    for ax in axes.flat:
        ax.set_xlim(r_harmsim.freqs.min(), r_harmsim.freqs.max())

    fig.suptitle("Figure 3 — Kernel registry: swappable harmonic kernels via "
                 "ResonanceConfig(harmonic_kernel=...)",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "fig3_kernel_comparison")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — combine-rule comparison (same H, PC; different combines)
# ---------------------------------------------------------------------------


def fig4_combine_rules():
    print("Fig 4: combine-rule comparison ...")
    sig = SIGNALS["harmonic_5_10_20_40"](sf=BASELINE_CONFIG["fs"])
    cfg = ResonanceConfig(**legacy_default_resonance_config_kwargs())
    r = compute_resonance(sig, sf=BASELINE_CONFIG["fs"], config=cfg)
    H, PC = r.factors["H"], r.factors["PC"]
    freqs = r.freqs

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6))

    # Top: H and PC factors
    axes[0].plot(freqs, H / H.max(), color=COLOR_H, lw=1.6, label="H(f) (normalized)")
    axes[0].plot(freqs, PC / PC.max(), color=COLOR_PC, lw=1.6, label="PC(f) (normalized)")
    axes[0].set_ylabel("Normalized factor")
    axes[0].set_title("a) Input factors")
    axes[0].legend(loc="upper right")

    # Bottom: 4 combine rules
    combines = {
        "product (legacy)": (product, COLOR_R),
        "geomean": (geomean, "#00838f"),
        "harmmean": (harmmean, "#ef6c00"),
        "min": (min_combine, "#6d4c41"),
    }
    for name, (fn, color) in combines.items():
        # Normalize each factor before combining so the comparison is fair
        Hn = H / H.max()
        PCn = PC / PC.max()
        R = fn([Hn, PCn])
        axes[1].plot(freqs, R, color=color, lw=1.4, label=name)
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Combined resonance")
    axes[1].set_title("b) Combine-rule comparison (factors min-max normalized)")
    axes[1].legend(loc="upper right", ncol=2)

    for ax in axes:
        ax.set_xlim(freqs.min(), freqs.max())

    fig.suptitle("Figure 4 — Combine-rule registry: ResonanceConfig(combine=...)",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "fig4_combine_rules")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 — cross-signal characterization (complexity summaries side-by-side)
# ---------------------------------------------------------------------------


def fig5_cross_signal():
    print("Fig 5: cross-signal characterization ...")
    rows = []
    spectra = {}
    for name in SIGNALS:
        r, _ = _compute_legacy_default(name)
        spectra[name] = r
        for key in ("H", "PC", "R"):
            s = r.summaries[key]
            rows.append({
                "signal": SIGNAL_LABELS[name],
                "spectrum": key,
                "avg": s["avg"], "max": s["max"],
                "flatness": s["flatness"], "entropy": s["entropy"],
                "spread": s["spread"], "higuchi": s["higuchi"],
            })
    df = pd.DataFrame(rows)
    # Persist as CSV for the report
    df.to_csv(FIG_DIR / "fig5_summary_table.csv", index=False)

    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(3, 3, hspace=0.55, wspace=0.32)
    metric_panels = [("flatness", "Spectral flatness"),
                     ("entropy", "Spectral entropy"),
                     ("higuchi", "Higuchi FD")]
    spectrum_colors = {"H": COLOR_H, "PC": COLOR_PC, "R": COLOR_R}

    bar_axes = [fig.add_subplot(gs[0, col]) for col in range(3)]
    spec_axes = [[fig.add_subplot(gs[1, col]) for col in range(3)],
                 [fig.add_subplot(gs[2, col]) for col in range(3)]]

    for col, (metric_key, metric_label) in enumerate(metric_panels):
        x = np.arange(len(SIGNALS))
        width = 0.26
        ax = bar_axes[col]
        for i, spec_key in enumerate(("H", "PC", "R")):
            vals = [df[(df["signal"] == SIGNAL_LABELS[n]) & (df["spectrum"] == spec_key)][metric_key].iloc[0]
                    for n in SIGNALS]
            ax.bar(x + (i - 1) * width, vals, width, color=spectrum_colors[spec_key],
                   alpha=0.85, label=f"{spec_key}(f)" if col == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels([SIGNAL_LABELS[n].split(" (")[0] for n in SIGNALS], rotation=12, fontsize=8)
        ax.set_title(metric_label, fontsize=10)
        ax.set_ylabel(metric_label.split()[1] if " " in metric_label else metric_label, fontsize=9)
        if col == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=3, columnspacing=0.6)

    # Rows 2-3: actual spectra side by side
    for col, name in enumerate(SIGNALS):
        r = spectra[name]
        ax_h = spec_axes[0][col]
        ax_h.plot(r.freqs, r.factors["H"] / max(r.factors["H"].max(), 1e-12), color=COLOR_H, lw=1.3, label="H")
        ax_h.plot(r.freqs, r.factors["PC"] / max(r.factors["PC"].max(), 1e-12), color=COLOR_PC, lw=1.3, label="PC")
        ax_h.set_title(SIGNAL_LABELS[name].split(" (")[0], fontsize=10)
        ax_h.set_xlim(r.freqs.min(), r.freqs.max())
        if col == 0:
            ax_h.set_ylabel("H, PC (normalized)")
            ax_h.legend(loc="upper right", fontsize=8)

        ax_r = spec_axes[1][col]
        ax_r.plot(r.freqs, r.resonance_spectrum, color=COLOR_R, lw=1.4)
        ax_r.fill_between(r.freqs, 0, r.resonance_spectrum, color=COLOR_R, alpha=0.15)
        ax_r.set_xlim(r.freqs.min(), r.freqs.max())
        ax_r.set_xlabel("Frequency (Hz)")
        if col == 0:
            ax_r.set_ylabel("R(f)")

    fig.suptitle("Figure 5 — Cross-signal characterization: H/PC/R complexity discriminates "
                 "harmonic / pink / inharmonic regimes",
                 fontsize=13, fontweight="bold", y=0.995)
    _save(fig, "fig5_cross_signal")
    plt.close(fig)

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Generating evaluation report figures ...")
    fig1_regression()
    fig2_decomposition()
    fig3_kernel_comparison()
    fig4_combine_rules()
    summary_df = fig5_cross_signal()
    print("\nSummary table:")
    print(summary_df.pivot(index="signal", columns="spectrum",
                            values=["flatness", "entropy", "higuchi", "spread"]).to_string())
    print(f"\nAll figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
