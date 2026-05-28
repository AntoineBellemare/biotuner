"""Cross-channel resonance validation figures (Layer B).

Validates compute_cross_resonance on signal-pair regimes with known structure:

  Fig 21 — H / PC / R for 4 representative signal-pair regimes
  Fig 22 — Three-flavor reducer comparison ('1to2', '2to1', 'all')
  Fig 23 — Harmonic-kernel swap: harmsim vs subharm_tension on same pair
  Fig 24 — Coupling-metric swap: nm_wpli_complex vs phase-only metrics
  Fig 25 — Bit-exact equivalence to legacy compute_cross_spectrum_harmonicity

Usage:
    python reports/resonance_refactor/cross_channel_signals.py
"""
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from biotuner.harmonic_connectivity import (  # noqa: E402
    compute_cross_resonance,
    compute_cross_spectrum_harmonicity,
)
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
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "legend.frameon": False, "lines.linewidth": 1.3,
})

COLOR_H = "#1a237e"
COLOR_PC = "#6a1b9a"
COLOR_R = "#b71c1c"
COLOR_SIG = "#37474f"


def _pink_noise(n, sf, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n)
    f = np.fft.rfftfreq(n, 1 / sf); f[0] = f[1]
    return np.fft.irfft(np.fft.rfft(w) / np.sqrt(f), n=n)


def build_signal_pairs(sf=500, duration=10.0):
    """Four representative signal-pair regimes:

    (A) Same frequency + phase-locked:  10 Hz × 10 Hz (π/4 offset)
    (B) 1:2 harmonic + phase-locked:    10 Hz × 20 Hz
    (C) Same frequency, phase-drifting: 10 Hz × 10 Hz (Wiener drift)
    (D) Independent oscillators:         10 Hz × 17 Hz
    """
    n = int(sf * duration)
    t = np.arange(n) / sf
    rng = np.random.default_rng(0)
    dt = 1.0 / sf
    noise = lambda seed: 0.25 * _pink_noise(n, sf, seed=seed)

    # A: locked alpha pair
    a1 = np.sin(2 * np.pi * 10 * t) + noise(1)
    a2 = np.sin(2 * np.pi * 10 * t + np.pi / 4) + noise(2)

    # B: 1:2 harmonic locked
    b1 = np.sin(2 * np.pi * 10 * t) + noise(3)
    b2 = np.sin(2 * np.pi * 20 * t + np.pi / 8) + noise(4)

    # C: same freq but Wiener phase drift on signal 2
    c1 = np.sin(2 * np.pi * 10 * t) + noise(5)
    dphi = rng.standard_normal(n) * 1.5 * np.sqrt(dt)
    phi_drift = np.cumsum(dphi)
    c2 = np.sin(2 * np.pi * 10 * t + phi_drift) + noise(6)

    # D: independent (different frequencies, no coupling)
    d1 = np.sin(2 * np.pi * 10 * t) + noise(7)
    d2 = np.sin(2 * np.pi * 17 * t + 1.7) + noise(8)

    return {
        "A: locked alpha (10 × 10 Hz, π/4)": (a1, a2),
        "B: 1:2 harmonic (10 × 20 Hz, locked)": (b1, b2),
        "C: phase-drifting alpha (10 × 10 Hz, Wiener)": (c1, c2),
        "D: independent (10 × 17 Hz)": (d1, d2),
    }


def _default_cfg(precision_hz=0.5, fmin=2, fmax=30, **overrides):
    cfg_kwargs = dict(
        precision_hz=precision_hz, fmin=fmin, fmax=fmax, noverlap=1,
        smoothness=1, n_peaks=5, remove_aperiodic=False,
        harmonic_kernel="harmsim",
        harmonic_kernel_params={"n_harms": 10, "delta_lim": 0.1, "min_notes": 2},
        phase_estimator="stft", coupling_metric="nm_wpli_complex",
        gaussian_smooth_sigma=1.0, combine="product",
    )
    cfg_kwargs.update(overrides)
    return ResonanceConfig(**cfg_kwargs)


def _save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png")
    fig.savefig(FIG_DIR / f"{name}.pdf")
    print(f"  wrote {name}.png + .pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 21 — H / PC / R on 4 signal-pair regimes
# ---------------------------------------------------------------------------


def fig21_signal_pair_regimes():
    print("Fig 21: cross-resonance on 4 signal-pair regimes ...")
    sf = 500
    pairs = build_signal_pairs(sf=sf)
    cfg = _default_cfg()

    fig, axes = plt.subplots(4, 3, figsize=(14, 11))

    for row, (label, (s1, s2)) in enumerate(pairs.items()):
        result = compute_cross_resonance(s1, s2, sf=sf, config=cfg)
        f = result.freqs

        # Column 0: H(f) 'all' flavor with peaks
        axes[row, 0].plot(f, result.factors["H"]["all"], color=COLOR_H, lw=1.4)
        axes[row, 0].fill_between(f, 0, result.factors["H"]["all"], color=COLOR_H, alpha=0.13)
        for pf in result.peaks["H"]:
            axes[row, 0].axvline(pf, color=COLOR_H, ls=":", alpha=0.5)
        axes[row, 0].set_ylabel(label.split(":")[0], fontsize=11, rotation=0, labelpad=25, va="center")
        if row == 0:
            axes[row, 0].set_title("a) Cross-channel H(f)", fontsize=10)

        # Column 1: PC(f) 'all'
        axes[row, 1].plot(f, result.factors["PC"]["all"], color=COLOR_PC, lw=1.4)
        axes[row, 1].fill_between(f, 0, result.factors["PC"]["all"], color=COLOR_PC, alpha=0.13)
        for pf in result.peaks["PC"]:
            axes[row, 1].axvline(pf, color=COLOR_PC, ls=":", alpha=0.5)
        if row == 0:
            axes[row, 1].set_title("b) Cross-channel PC(f)", fontsize=10)

        # Column 2: R(f) 'all'
        axes[row, 2].plot(f, result.resonance_spectrum["all"], color=COLOR_R, lw=1.4)
        axes[row, 2].fill_between(f, 0, result.resonance_spectrum["all"], color=COLOR_R, alpha=0.13)
        for pf in result.peaks["R"]:
            axes[row, 2].axvline(pf, color=COLOR_R, ls=":", alpha=0.5)
        if row == 0:
            axes[row, 2].set_title("c) Cross-channel R(f) = H · PC", fontsize=10)

        # Show full pair label as subtitle on the H column
        axes[row, 0].text(0.5, 1.05, label, transform=axes[row, 0].transAxes,
                          ha="center", va="bottom", fontsize=9, color="grey", fontstyle="italic")

        if row == 3:
            for col in range(3):
                axes[row, col].set_xlabel("Frequency (Hz)")
        for col in range(3):
            axes[row, col].set_xlim(f.min(), f.max())

    fig.suptitle("Figure 21 — Cross-channel compute_cross_resonance on 4 signal-pair regimes",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "fig21_cross_resonance_regimes")


# ---------------------------------------------------------------------------
# Fig 22 — three reducer flavors on an asymmetric pair
# ---------------------------------------------------------------------------


def fig22_three_flavors():
    print("Fig 22: 3-flavor reducer comparison ...")
    sf = 500
    n = int(sf * 10)
    t = np.arange(n) / sf
    # Strongly asymmetric pair: signal1 has DOMINANT 10 Hz tone (narrowband),
    # signal2 has WEAK 10 Hz + STRONG 20 Hz (its peak power is at 20 Hz).
    # The asymmetry of psd1[i] × psd2[j] vs psd2[i] × psd1[j] should produce
    # visibly different '1to2' vs '2to1' profiles.
    sig1 = 2.5 * np.sin(2 * np.pi * 10 * t) + 0.2 * _pink_noise(n, sf, seed=10)
    sig2 = (0.3 * np.sin(2 * np.pi * 10 * t + np.pi / 4)
              + 2.5 * np.sin(2 * np.pi * 20 * t + np.pi / 8)
              + 0.2 * _pink_noise(n, sf, seed=11))
    cfg = _default_cfg()
    result = compute_cross_resonance(sig1, sig2, sf=sf, config=cfg)
    f = result.freqs

    fig, axes = plt.subplots(3, 2, figsize=(13, 9))

    # Row 0: signals
    t_window = np.arange(int(sf * 1.5)) / sf
    axes[0, 0].plot(t_window, sig1[:len(t_window)], color=COLOR_SIG, lw=0.7)
    axes[0, 0].set_title("a) Signal 1: dominant 10 Hz", fontsize=10)
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 1].plot(t_window, sig2[:len(t_window)], color=COLOR_SIG, lw=0.7)
    axes[0, 1].set_title("b) Signal 2: dominant 20 Hz + weak 10 Hz", fontsize=10)
    axes[0, 1].set_xlabel("Time (s)")

    # Row 1: H factor — three flavors
    axes[1, 0].plot(f, result.factors["H"]["1to2"], color=COLOR_H, lw=1.5, label="H[1→2]")
    axes[1, 0].plot(f, result.factors["H"]["2to1"], color=COLOR_H, lw=1.5, ls="--", alpha=0.7, label="H[2→1]")
    axes[1, 0].plot(f, result.factors["H"]["all"], color="black", lw=1.0, ls=":", label="H[all] (avg)")
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("Harmonicity")
    axes[1, 0].set_title("c) H — three reducer flavors", fontsize=10)
    axes[1, 0].legend(loc="upper right")
    axes[1, 0].set_xlim(f.min(), f.max())

    # Row 1 right: PC factor — three flavors
    axes[1, 1].plot(f, result.factors["PC"]["1to2"], color=COLOR_PC, lw=1.5, label="PC[1→2]")
    axes[1, 1].plot(f, result.factors["PC"]["2to1"], color=COLOR_PC, lw=1.5, ls="--", alpha=0.7, label="PC[2→1]")
    axes[1, 1].plot(f, result.factors["PC"]["all"], color="black", lw=1.0, ls=":", label="PC[all] (avg)")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("Phase coupling")
    axes[1, 1].set_title("d) PC — three reducer flavors", fontsize=10)
    axes[1, 1].legend(loc="upper right")
    axes[1, 1].set_xlim(f.min(), f.max())

    # Row 2: R factor — three flavors
    axes[2, 0].plot(f, result.resonance_spectrum["1to2"], color=COLOR_R, lw=1.5, label="R[1→2]")
    axes[2, 0].plot(f, result.resonance_spectrum["2to1"], color=COLOR_R, lw=1.5, ls="--", alpha=0.7, label="R[2→1]")
    axes[2, 0].plot(f, result.resonance_spectrum["all"], color="black", lw=1.0, ls=":", label="R[all] (avg)")
    axes[2, 0].fill_between(f, 0, result.resonance_spectrum["all"], color=COLOR_R, alpha=0.08)
    axes[2, 0].set_xlabel("Frequency (Hz)")
    axes[2, 0].set_ylabel("Resonance")
    axes[2, 0].set_title("e) R = H · PC — three reducer flavors", fontsize=10)
    axes[2, 0].legend(loc="upper right")
    axes[2, 0].set_xlim(f.min(), f.max())

    # Row 2 right: ratio R[1→2] / R[2→1] to highlight asymmetry
    ratio = result.resonance_spectrum["1to2"] / (result.resonance_spectrum["2to1"] + 1e-12)
    axes[2, 1].plot(f, ratio, color=COLOR_R, lw=1.4)
    axes[2, 1].axhline(1.0, color="grey", ls=":", lw=1)
    axes[2, 1].set_xlabel("Frequency (Hz)")
    axes[2, 1].set_ylabel("R[1→2] / R[2→1]")
    axes[2, 1].set_title("f) Directional asymmetry of resonance", fontsize=10)
    axes[2, 1].set_xlim(f.min(), f.max())

    fig.suptitle("Figure 22 — Three reducer flavors expose directional asymmetry in cross-channel coupling",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "fig22_three_flavors")


# ---------------------------------------------------------------------------
# Fig 23 — harmonic-kernel swap (harmsim vs subharm_tension)
# ---------------------------------------------------------------------------


def fig23_kernel_swap():
    print("Fig 23: kernel swap (harmsim vs subharm_tension) ...")
    sf = 500
    pairs = build_signal_pairs(sf=sf)
    # Use the 1:2 harmonic pair for the demo
    sig1, sig2 = pairs["B: 1:2 harmonic (10 × 20 Hz, locked)"]
    cfg_harmsim = _default_cfg(harmonic_kernel="harmsim")
    cfg_subharm = _default_cfg(harmonic_kernel="subharm_tension")

    r_harmsim = compute_cross_resonance(sig1, sig2, sf=sf, config=cfg_harmsim)
    r_subharm = compute_cross_resonance(sig1, sig2, sf=sf, config=cfg_subharm)
    f = r_harmsim.freqs

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))

    axes[0, 0].plot(f, r_harmsim.factors["H"]["all"], color=COLOR_H, lw=1.5)
    axes[0, 0].fill_between(f, 0, r_harmsim.factors["H"]["all"], color=COLOR_H, alpha=0.13)
    axes[0, 0].set_title("a) H(f) — harmsim kernel (Gill & Purves 2009)", fontsize=10)
    axes[0, 0].set_ylabel("Harmonicity")
    axes[0, 0].set_xlim(f.min(), f.max())

    axes[0, 1].plot(f, r_subharm.factors["H"]["all"], color=COLOR_H, lw=1.5)
    axes[0, 1].fill_between(f, 0, r_subharm.factors["H"]["all"], color=COLOR_H, alpha=0.13)
    axes[0, 1].set_title("b) H(f) — subharm_tension kernel", fontsize=10)
    axes[0, 1].set_xlim(f.min(), f.max())

    axes[1, 0].plot(f, r_harmsim.resonance_spectrum["all"], color=COLOR_R, lw=1.5)
    axes[1, 0].fill_between(f, 0, r_harmsim.resonance_spectrum["all"], color=COLOR_R, alpha=0.13)
    axes[1, 0].set_title("c) R(f) — harmsim kernel", fontsize=10)
    axes[1, 0].set_xlabel("Frequency (Hz)")
    axes[1, 0].set_ylabel("Resonance")
    axes[1, 0].set_xlim(f.min(), f.max())

    axes[1, 1].plot(f, r_subharm.resonance_spectrum["all"], color=COLOR_R, lw=1.5)
    axes[1, 1].fill_between(f, 0, r_subharm.resonance_spectrum["all"], color=COLOR_R, alpha=0.13)
    axes[1, 1].set_title("d) R(f) — subharm_tension kernel", fontsize=10)
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_xlim(f.min(), f.max())

    fig.suptitle("Figure 23 — Cross-channel kernel registry: ResonanceConfig(harmonic_kernel=...)\n"
                 "Same 1:2 harmonic stimulus through two kernels — peak locations match, weights differ",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "fig23_cross_kernel_swap")


# ---------------------------------------------------------------------------
# Fig 24 — bit-exact equivalence to legacy compute_cross_spectrum_harmonicity
# ---------------------------------------------------------------------------


def fig24_legacy_equivalence():
    print("Fig 24: bit-exact equivalence to legacy ...")
    sf = 1000
    # Use the same 3 pairs from the snapshot regression test
    sys.path.insert(0, str(ROOT / "tests" / "resonance"))
    from _signals import SIGNALS  # noqa
    pair_specs = [
        ("harmonic x pink", "harmonic_5_10_20_40", "pink_noise"),
        ("harmonic x inharmonic", "harmonic_5_10_20_40", "inharmonic_7_11_18_23"),
        ("pink x inharmonic", "pink_noise", "inharmonic_7_11_18_23"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 9))

    for row, (label, s1_name, s2_name) in enumerate(pair_specs):
        sig1 = SIGNALS[s1_name](sf=sf)
        sig2 = SIGNALS[s2_name](sf=sf)

        # Legacy path (now a shim)
        df = compute_cross_spectrum_harmonicity(
            sig1, sig2, precision_hz=0.5, fmin=2, fmax=30, fs=sf,
            noverlap=1, power_law_remove=False, n_peaks=5,
            metric="harmsim", n_harms=10, delta_lim=0.1, min_notes=2,
            plot=False, smoothness=1, smoothness_harm=1, phase_mode=None,
        )
        plt.close("all")
        H_leg = np.asarray(df["harmonicity"].iloc[0])
        PC_leg = np.asarray(df["phase_coupling"].iloc[0])
        R_leg = np.asarray(df["resonance"].iloc[0])

        # New direct API
        cfg = _default_cfg(precision_hz=0.5, fmin=2, fmax=30)
        result = compute_cross_resonance(sig1, sig2, sf=sf, config=cfg)
        f = result.freqs
        H_new = result.factors["H"]["all"]
        PC_new = result.factors["PC"]["all"]
        R_new = result.resonance_spectrum["all"]

        max_diffs = {
            "H": np.max(np.abs(H_new - H_leg)),
            "PC": np.max(np.abs(PC_new - PC_leg)),
            "R": np.max(np.abs(R_new - R_leg)),
        }

        for col, (key, color, leg, new) in enumerate([
            ("H", COLOR_H, H_leg, H_new),
            ("PC", COLOR_PC, PC_leg, PC_new),
            ("R", COLOR_R, R_leg, R_new),
        ]):
            axes[row, col].plot(f, leg, color="grey", lw=3.0, alpha=0.55, label="Legacy")
            axes[row, col].plot(f, new, color=color, lw=1.2, label="compute_cross_resonance")
            axes[row, col].text(0.97, 0.94, f"max |Δ| = {max_diffs[key]:.1e}",
                                transform=axes[row, col].transAxes, ha="right", va="top",
                                fontsize=8, family="monospace",
                                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2))
            axes[row, col].set_xlim(f.min(), f.max())
            if col == 0:
                axes[row, col].set_ylabel(label, fontsize=10)
            if row == 0:
                axes[row, col].set_title(f"{key}(f)", color=color, fontsize=11)
            if row == 0 and col == 0:
                axes[row, col].legend(loc="upper left", fontsize=8)
            if row == 2:
                axes[row, col].set_xlabel("Frequency (Hz)")

    fig.suptitle("Figure 24 — Bit-exact equivalence: compute_cross_resonance matches legacy compute_cross_spectrum_harmonicity\n"
                 "(max |Δ| ≲ 1e-15 — float-precision noise — across 3 signal pairs)",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "fig24_cross_legacy_equivalence")


# ---------------------------------------------------------------------------
# Fig 25 — Cross-channel Kuramoto sweep
# ---------------------------------------------------------------------------


def _two_coupled_kuramoto(K, sf=500, duration=20.0, omega1=10.0, omega2=10.5, seed=0):
    """Two Kuramoto oscillators with bidirectional coupling. Returns
    (signal1, signal2, mean_order_parameter).

      dθ_1/dt = ω_1 + K * sin(θ_2 - θ_1)
      dθ_2/dt = ω_2 + K * sin(θ_1 - θ_2)
    """
    rng = np.random.default_rng(seed)
    T = int(sf * duration)
    dt = 1.0 / sf
    w1 = 2 * np.pi * omega1
    w2 = 2 * np.pi * omega2
    theta1, theta2 = rng.uniform(0, 2 * np.pi), rng.uniform(0, 2 * np.pi)
    sig1 = np.empty(T); sig2 = np.empty(T)
    r_t = np.empty(T)
    for t in range(T):
        sig1[t] = np.sin(theta1)
        sig2[t] = np.sin(theta2)
        # Order parameter for the 2-oscillator system
        z = (np.exp(1j * theta1) + np.exp(1j * theta2)) / 2
        r_t[t] = np.abs(z)
        d1 = w1 + K * np.sin(theta2 - theta1)
        d2 = w2 + K * np.sin(theta1 - theta2)
        theta1 += d1 * dt
        theta2 += d2 * dt
    # Add small observation noise so STFT doesn't see a perfect tone
    sig1 += 0.05 * _pink_noise(T, sf, seed=seed + 100)
    sig2 += 0.05 * _pink_noise(T, sf, seed=seed + 101)
    # Mean order parameter after transient
    mean_r = float(np.mean(r_t[int(sf * 3):]))
    return sig1, sig2, mean_r


def fig25_kuramoto_pair_sweep():
    print("Fig 25: cross-channel Kuramoto pair sweep ...")
    sf = 500
    K_values = [0.0, 0.5, 1.5, 5.0, 15.0]
    cfg = _default_cfg(precision_hz=0.5, fmin=2, fmax=30)
    results = []
    print(f"  simulating {len(K_values)} Kuramoto pairs (T=20 s)")
    t0 = time.time()
    for K in K_values:
        sig1, sig2, mean_r = _two_coupled_kuramoto(K, sf=sf, duration=20.0)
        r = compute_cross_resonance(sig1, sig2, sf=sf, config=cfg)
        results.append({"K": K, "mean_r": mean_r, "result": r,
                         "sig1": sig1, "sig2": sig2})
    print(f"  done in {time.time() - t0:.1f}s")

    fig = plt.figure(figsize=(12, 8.5))
    gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.30,
                            height_ratios=[1, 1.3, 1.1], width_ratios=[2, 1])

    # Panel a — heatmap of R(f) vs K
    f = results[0]["result"].freqs
    R_mat = np.stack([r["result"].resonance_spectrum["all"] for r in results])
    R_norm = R_mat / (R_mat.max(axis=1, keepdims=True) + 1e-12)
    ax_heat = fig.add_subplot(gs[0, 0])
    im = ax_heat.imshow(R_norm, aspect="auto", origin="lower",
                        extent=[f.min(), f.max(), -0.5, len(K_values) - 0.5],
                        cmap="magma", interpolation="nearest")
    ax_heat.set_yticks(range(len(K_values)))
    ax_heat.set_yticklabels([f"K = {K}" for K in K_values])
    ax_heat.set_xlabel("Frequency (Hz)")
    ax_heat.set_title("a) Cross R(f) across coupling strengths (row-normalized)")
    plt.colorbar(im, ax=ax_heat, fraction=0.04, pad=0.02, label="R / max(R)")

    # Panel b — order parameter + max R vs K
    ax_b = fig.add_subplot(gs[0, 1])
    mean_rs = [r["mean_r"] for r in results]
    max_Rs = [r["result"].resonance_spectrum["all"].max() for r in results]
    ax_b.plot(K_values, mean_rs, "o-", color="#1a237e", lw=1.6, label="⟨|Z|⟩")
    ax_b2 = ax_b.twinx()
    ax_b2.plot(K_values, max_Rs, "s-", color="#b71c1c", lw=1.6, label="max R[all]")
    ax_b.set_xlabel("Coupling K")
    ax_b.set_ylabel("Order param ⟨|Z|⟩", color="#1a237e")
    ax_b2.set_ylabel("max cross R(f)", color="#b71c1c")
    ax_b.set_title("b) Synchronization signatures")
    lines1, labels1 = ax_b.get_legend_handles_labels()
    lines2, labels2 = ax_b2.get_legend_handles_labels()
    ax_b.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    # Panel c-d — time domain at K=0 vs K=max
    t_window = np.arange(int(sf * 3)) / sf
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.plot(t_window, results[0]["sig1"][:len(t_window)], color=COLOR_SIG, lw=0.7, label="sig1")
    ax_c.plot(t_window, results[0]["sig2"][:len(t_window)] - 2.5, color="#9c27b0", lw=0.7, label="sig2 (shifted)")
    ax_c.set_title(f"c) Time domain — K = {K_values[0]} (incoherent)", fontsize=10)
    ax_c.set_xlabel("Time (s)"); ax_c.legend(loc="upper right", fontsize=8)

    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.plot(t_window, results[-1]["sig1"][:len(t_window)], color=COLOR_SIG, lw=0.7, label="sig1")
    ax_d.plot(t_window, results[-1]["sig2"][:len(t_window)] - 2.5, color="#9c27b0", lw=0.7, label="sig2")
    ax_d.set_title(f"d) K = {K_values[-1]} (synchronized)", fontsize=10)
    ax_d.set_xlabel("Time (s)"); ax_d.legend(loc="upper right", fontsize=8)

    # Panel e — overlay R(f) at K=0 vs K=max
    ax_e = fig.add_subplot(gs[2, :])
    ax_e.plot(f, results[0]["result"].resonance_spectrum["all"],
                color=COLOR_R, lw=1.5, label=f"K = {K_values[0]}")
    ax_e.plot(f, results[-1]["result"].resonance_spectrum["all"],
                color=COLOR_R, lw=1.5, ls="--", label=f"K = {K_values[-1]}")
    ax_e.fill_between(f, 0, results[-1]["result"].resonance_spectrum["all"], color=COLOR_R, alpha=0.1)
    ax_e.set_xlabel("Frequency (Hz)"); ax_e.set_ylabel("Cross R(f)")
    ax_e.set_title("e) Cross R(f) — incoherent vs synchronized", fontsize=10)
    ax_e.legend(loc="upper right")
    ax_e.set_xlim(f.min(), f.max())

    fig.suptitle("Figure 25 — Cross-channel Kuramoto pair: max R[all](f) tracks the order parameter",
                 fontsize=13, fontweight="bold", y=0.995)
    _save(fig, "fig25_cross_kuramoto_pair")


# ---------------------------------------------------------------------------
# Fig 26 — Realistic multi-band EEG-pair
# ---------------------------------------------------------------------------


def _multiband_eeg_pair(sf=500, duration=20.0, seed=0):
    """Two-channel realistic synthetic EEG with PARTIAL band overlap:

      ch1: 1/f bg + sustained theta (5Hz) + alpha (10Hz) bursts +     beta (22Hz) bursts
      ch2: 1/f bg +                       alpha (10Hz) bursts + beta (22Hz) bursts + gamma (45Hz)

    Both share alpha and beta — the cross-channel pipeline should highlight
    these. Theta (ch1 only) and gamma (ch2 only) should NOT appear in cross-R.
    """
    rng = np.random.default_rng(seed)
    n = int(sf * duration)
    t = np.arange(n) / sf
    bg1 = 1.0 * _pink_noise(n, sf, seed=seed + 1)
    bg2 = 1.0 * _pink_noise(n, sf, seed=seed + 2)

    def _bursts(freq, times, dur, amp=1.0, phase_align=False):
        out = np.zeros(n)
        for bt in times:
            idx = (t >= bt) & (t <= bt + dur)
            local = t[idx] - bt
            w = np.sin(np.pi * local / dur) ** 2
            phi = 0.0 if phase_align else rng.uniform(0, 2 * np.pi)
            out[idx] += amp * w * np.sin(2 * np.pi * freq * local + phi)
        return out

    # Shared alpha bursts (phase-aligned across channels)
    alpha_times = np.linspace(1.5, duration - 1.5, 8) + rng.uniform(-0.1, 0.1, size=8)
    alpha_ch1 = _bursts(10.0, alpha_times, 0.7, amp=1.5, phase_align=True)
    alpha_ch2 = _bursts(10.0, alpha_times, 0.7, amp=1.5, phase_align=True)

    # Shared beta bursts (phase-aligned)
    beta_times = np.linspace(0.5, duration - 0.5, 12) + rng.uniform(-0.1, 0.1, size=12)
    beta_ch1 = _bursts(22.0, beta_times, 0.35, amp=0.8, phase_align=True)
    beta_ch2 = _bursts(22.0, beta_times, 0.35, amp=0.8, phase_align=True)

    # Unique theta (ch1 only) - sustained
    theta_ch1 = 0.5 * np.sin(2 * np.pi * 5.0 * t + rng.uniform(0, 2 * np.pi))

    # Unique gamma (ch2 only) - bursts
    gamma_times = np.linspace(2.0, duration - 2.0, 15) + rng.uniform(-0.1, 0.1, size=15)
    gamma_ch2 = _bursts(45.0, gamma_times, 0.3, amp=0.6)

    sig1 = bg1 + theta_ch1 + alpha_ch1 + beta_ch1
    sig2 = bg2 + alpha_ch2 + beta_ch2 + gamma_ch2
    return sig1, sig2


def fig26_multiband_eeg_pair():
    print("Fig 26: multi-band EEG-pair ...")
    sf = 500
    sig1, sig2 = _multiband_eeg_pair(sf=sf, duration=20.0)
    cfg = _default_cfg(precision_hz=0.5, fmin=2, fmax=60)
    result = compute_cross_resonance(sig1, sig2, sf=sf, config=cfg)
    f = result.freqs

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1, 1.1], hspace=0.55)

    # Time domain (4 s) for both signals
    t_window = np.arange(int(sf * 4)) / sf
    ax_t = fig.add_subplot(gs[0])
    ax_t.plot(t_window, sig1[:len(t_window)], color=COLOR_SIG, lw=0.6, label="ch1: 1/f + θ + α + β")
    ax_t.plot(t_window, sig2[:len(t_window)] - 5, color="#9c27b0", lw=0.6, label="ch2: 1/f + α + β + γ")
    ax_t.set_xlabel("Time (s)"); ax_t.set_yticks([])
    ax_t.set_title("a) Two-channel synthetic EEG — shared α (10 Hz) + β (22 Hz), unique θ (ch1) + γ (ch2)")
    ax_t.legend(loc="upper right", fontsize=8)

    # PSDs side by side (compact)
    from scipy.signal import welch
    ax_psd = fig.add_subplot(gs[1])
    f1, p1 = welch(sig1, sf, nperseg=int(sf * 2))
    f2, p2 = welch(sig2, sf, nperseg=int(sf * 2))
    mask = (f1 >= 2) & (f1 <= 60)
    ax_psd.semilogy(f1[mask], p1[mask], color=COLOR_SIG, lw=1.0, label="ch1 PSD")
    ax_psd.semilogy(f2[mask], p2[mask], color="#9c27b0", lw=1.0, label="ch2 PSD")
    for fc, label in [(5, "θ"), (10, "α"), (22, "β"), (45, "γ")]:
        ax_psd.axvline(fc, color="grey", ls=":", alpha=0.5)
        ax_psd.text(fc, ax_psd.get_ylim()[1] * 0.6, label, ha="center", fontsize=9, color="grey")
    ax_psd.set_xlabel("Frequency (Hz)"); ax_psd.set_ylabel("PSD (log)")
    ax_psd.set_title("b) Per-channel PSD (ground truth band content)", fontsize=10)
    ax_psd.legend(loc="upper right", fontsize=8)

    # H, PC, R cross-channel
    ax_h = fig.add_subplot(gs[2])
    ax_h.plot(f, result.factors["H"]["all"], color=COLOR_H, lw=1.4)
    ax_h.fill_between(f, 0, result.factors["H"]["all"], color=COLOR_H, alpha=0.13)
    ax_h.set_title("c) Cross H(f) — joint harmonic similarity weighted by both channels' PSDs", fontsize=10)
    ax_h.set_ylabel("H[all]", color=COLOR_H); ax_h.set_xlim(f.min(), f.max())

    ax_pc = fig.add_subplot(gs[3])
    ax_pc.plot(f, result.factors["PC"]["all"], color=COLOR_PC, lw=1.4)
    ax_pc.fill_between(f, 0, result.factors["PC"]["all"], color=COLOR_PC, alpha=0.13)
    ax_pc.set_title("d) Cross PC(f) — STFT cross-spectrum wPLI", fontsize=10)
    ax_pc.set_ylabel("PC[all]", color=COLOR_PC); ax_pc.set_xlim(f.min(), f.max())

    ax_r = fig.add_subplot(gs[4])
    ax_r.plot(f, result.resonance_spectrum["all"], color=COLOR_R, lw=1.5)
    ax_r.fill_between(f, 0, result.resonance_spectrum["all"], color=COLOR_R, alpha=0.15)
    for fc, label in [(5, "θ"), (10, "α"), (22, "β"), (45, "γ")]:
        ax_r.axvline(fc, color="grey", ls=":", alpha=0.5)
        ax_r.text(fc, ax_r.get_ylim()[1] * 0.92, label, ha="center", fontsize=9, color="grey")
    ax_r.set_xlabel("Frequency (Hz)"); ax_r.set_ylabel("R[all]", color=COLOR_R)
    ax_r.set_title("e) Cross R(f) = H · PC — SHARED frequencies should dominate", fontsize=10)
    ax_r.set_xlim(f.min(), f.max())

    fig.suptitle("Figure 26 — Realistic multi-band EEG-pair: cross R(f) isolates shared α + β carriers",
                 fontsize=12, fontweight="bold", y=0.995)
    _save(fig, "fig26_cross_multiband_eeg")


# ---------------------------------------------------------------------------
# Fig 27 — Cross-channel SNR robustness sweep
# ---------------------------------------------------------------------------


def fig27_cross_snr_sweep():
    print("Fig 27: cross-channel SNR robustness ...")
    sf = 500
    n = int(sf * 16)
    t = np.arange(n) / sf
    # Clean coupled signals: same alpha tone with π/4 phase offset
    clean1 = np.sin(2 * np.pi * 10 * t)
    clean2 = np.sin(2 * np.pi * 10 * t + np.pi / 4)
    clean1 /= np.std(clean1); clean2 /= np.std(clean2)

    snrs_db = [np.inf, 20, 10, 5, 0, -5, -10]
    results = []
    for snr in snrs_db:
        if snr == np.inf:
            sig1, sig2 = clean1.copy(), clean2.copy()
        else:
            noise_power = 1.0 / (10 ** (snr / 10))
            sig1 = clean1 + np.sqrt(noise_power) * _pink_noise(n, sf, seed=int(abs(snr) + 1))
            sig2 = clean2 + np.sqrt(noise_power) * _pink_noise(n, sf, seed=int(abs(snr) + 100))
        cfg = _default_cfg(precision_hz=0.5, fmin=2, fmax=30)
        results.append({"snr": snr, "result": compute_cross_resonance(sig1, sig2, sf=sf, config=cfg)})

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    f = results[0]["result"].freqs
    palette = plt.cm.viridis(np.linspace(0.05, 0.92, len(snrs_db)))

    # Panel a: R(f) at each SNR (peak-normalized for comparability)
    for r, c in zip(results, palette):
        R = r["result"].resonance_spectrum["all"]
        Rn = R / (R.max() + 1e-12)
        snr_label = "∞" if r["snr"] == np.inf else f"{r['snr']:+d} dB"
        axes[0].plot(f, Rn, color=c, lw=1.4, label=f"SNR = {snr_label}")
    axes[0].set_xlabel("Frequency (Hz)"); axes[0].set_ylabel("Cross R(f) / max")
    axes[0].set_title("a) Cross R(f) (peak-normalized) across SNRs", fontsize=10)
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[0].set_xlim(f.min(), f.max())

    # Panel b: peak detectability max/median vs SNR
    detect = []
    for r in results:
        R = r["result"].resonance_spectrum["all"]
        detect.append(R.max() / (np.median(R) + 1e-12))
    snr_finite = [s if s != np.inf else 35 for s in snrs_db]
    axes[1].plot(snr_finite, detect, "o-", color=COLOR_R, lw=1.6)
    axes[1].set_xlabel("SNR (dB) — ∞ shown as 35 dB")
    axes[1].set_ylabel("max R / median R")
    axes[1].set_title("b) Peak detectability vs SNR", fontsize=10)
    axes[1].set_yscale("log"); axes[1].grid(True, alpha=0.3, which="both")

    fig.suptitle("Figure 27 — Cross-channel SNR robustness: R(f) structure preserved down to ~0 dB",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "fig27_cross_snr_sweep")


# ---------------------------------------------------------------------------
# Fig 28 — Surrogate-null normalization for cross-channel
# ---------------------------------------------------------------------------


def fig28_cross_surrogate_null():
    print("Fig 28: cross-channel surrogate null ...")
    sf = 500
    # Realistic EEG-pair (shared alpha + beta) — the surrogate destroys
    # cross-channel phase coherence but preserves per-channel PSD.
    sig1, sig2 = _multiband_eeg_pair(sf=sf, duration=20.0)
    cfg = _default_cfg(precision_hz=0.5, fmin=2, fmax=30, n_peaks=5)
    real = compute_cross_resonance(sig1, sig2, sf=sf, config=cfg)
    n_freqs = real.freqs.size

    n_surr = 100
    print(f"  running {n_surr} phase-randomized surrogates ...")
    rng = np.random.default_rng(42)
    surr_R = np.empty((n_surr, n_freqs))
    surr_PC = np.empty((n_surr, n_freqs))

    def _phase_randomize(sig, rng):
        N = len(sig)
        X = np.fft.rfft(sig)
        phases = np.exp(1j * rng.uniform(0, 2 * np.pi, size=X.shape))
        phases[0] = 1.0
        if N % 2 == 0:
            phases[-1] = 1.0
        return np.fft.irfft(np.abs(X) * phases, n=N)

    t0 = time.time()
    for k in range(n_surr):
        s1_surr = _phase_randomize(sig1, rng)
        s2_surr = _phase_randomize(sig2, rng)
        r_s = compute_cross_resonance(s1_surr, s2_surr, sf=sf, config=cfg)
        surr_R[k] = r_s.resonance_spectrum["all"]
        surr_PC[k] = r_s.factors["PC"]["all"]
    print(f"  done in {time.time() - t0:.1f}s")

    mu_R, sd_R = surr_R.mean(axis=0), surr_R.std(axis=0) + 1e-12
    mu_PC, sd_PC = surr_PC.mean(axis=0), surr_PC.std(axis=0) + 1e-12
    z_R = (real.resonance_spectrum["all"] - mu_R) / sd_R
    z_PC = (real.factors["PC"]["all"] - mu_PC) / sd_PC

    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5))
    f = real.freqs

    p10, p90 = np.percentile(surr_PC, [5, 95], axis=0)
    axes[0].fill_between(f, p10, p90, color="grey", alpha=0.4, label="Surrogate 5-95%")
    axes[0].plot(f, mu_PC, color="grey", lw=1.0, label="Surrogate mean")
    axes[0].plot(f, real.factors["PC"]["all"], color=COLOR_PC, lw=1.6, label="Observed PC[all]")
    axes[0].set_ylabel("Phase coupling")
    axes[0].set_title("a) Cross PC: observed vs surrogate (per-channel phase-randomized)", fontsize=10)
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    axes[0].set_xlim(f.min(), f.max())

    p10R, p90R = np.percentile(surr_R, [5, 95], axis=0)
    axes[1].fill_between(f, p10R, p90R, color="grey", alpha=0.4, label="Surrogate 5-95%")
    axes[1].plot(f, mu_R, color="grey", lw=1.0, label="Surrogate mean")
    axes[1].plot(f, real.resonance_spectrum["all"], color=COLOR_R, lw=1.6, label="Observed R[all]")
    axes[1].set_ylabel("Resonance")
    axes[1].set_title("b) Cross R: observed vs surrogate", fontsize=10)
    axes[1].legend(loc="upper right", ncol=3, fontsize=8)
    axes[1].set_xlim(f.min(), f.max())

    axes[2].plot(f, z_R, color="#00838f", lw=1.6, label="z(R[all])")
    axes[2].plot(f, z_PC, color=COLOR_PC, lw=1.2, alpha=0.7, label="z(PC[all])")
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].axhline(2, color="k", lw=0.5, ls="--", label="z = ±2")
    axes[2].axhline(-2, color="k", lw=0.5, ls="--")
    sig_mask = z_R > 2
    if sig_mask.any():
        axes[2].fill_between(f, 0, z_R, where=sig_mask, color=COLOR_R, alpha=0.18,
                              label=f"z(R) > 2 ({sig_mask.sum()} bins)")
    axes[2].set_xlabel("Frequency (Hz)"); axes[2].set_ylabel("z-score")
    axes[2].set_title("c) Surrogate-normalized z(R) — shared α/β survive above the null", fontsize=10)
    axes[2].legend(loc="upper right", ncol=4, fontsize=8)
    axes[2].set_xlim(f.min(), f.max())

    fig.suptitle("Figure 28 — Cross-channel surrogate null: per-channel phase-randomization preserves "
                 "PSD, destroys coherence",
                 fontsize=12, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "fig28_cross_surrogate_null")


# ---------------------------------------------------------------------------
# Fig 29 — Refinements A, B, C side-by-side vs legacy default
# ---------------------------------------------------------------------------


def fig29_refinements_AB():
    """Compare default cross_pc_reducer='count' (legacy) vs 'joint' (Refinement A),
    and cross_use_ratio_kernel=False (legacy) vs True (Refinement B), on the
    realistic multi-band EEG-pair from Fig 26."""
    print("Fig 29: refinements A (joint PC) and B (n:m PC) ...")
    sf = 500
    sig1, sig2 = _multiband_eeg_pair(sf=sf, duration=20.0)

    cfg_default = _default_cfg(precision_hz=0.5, fmin=2, fmax=60)
    cfg_jointPC = _default_cfg(precision_hz=0.5, fmin=2, fmax=60,
                                 cross_pc_reducer="joint")
    cfg_nmPC = _default_cfg(precision_hz=0.5, fmin=2, fmax=60,
                              cross_use_ratio_kernel=True,
                              ratio_kernel="binary",
                              ratio_kernel_params={"max_nm": 3, "tolerance": 0.05, "fallback_to_1_1": True})
    cfg_both = _default_cfg(precision_hz=0.5, fmin=2, fmax=60,
                              cross_pc_reducer="joint",
                              cross_use_ratio_kernel=True,
                              ratio_kernel="binary",
                              ratio_kernel_params={"max_nm": 3, "tolerance": 0.05, "fallback_to_1_1": True})

    r_default = compute_cross_resonance(sig1, sig2, sf=sf, config=cfg_default)
    r_jointPC = compute_cross_resonance(sig1, sig2, sf=sf, config=cfg_jointPC)
    r_nmPC = compute_cross_resonance(sig1, sig2, sf=sf, config=cfg_nmPC)
    r_both = compute_cross_resonance(sig1, sig2, sf=sf, config=cfg_both)
    f = r_default.freqs

    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    cases = [
        (r_default, "a) DEFAULT\nPC='count', 1:1 cross-spec"),
        (r_jointPC, "b) Refinement A\nPC='joint' (joint p1·p2)"),
        (r_nmPC, "c) Refinement B\nn:m PC via binary_nm kernel"),
        (r_both, "d) A + B combined"),
    ]
    bands = [(5, "θ"), (10, "α"), (22, "β"), (45, "γ")]

    for col, (r, title) in enumerate(cases):
        # PC row
        axes[0, col].plot(f, r.factors["PC"]["all"], color=COLOR_PC, lw=1.4)
        axes[0, col].fill_between(f, 0, r.factors["PC"]["all"], color=COLOR_PC, alpha=0.13)
        axes[0, col].set_title(title, fontsize=9)
        axes[0, col].set_xlim(f.min(), f.max())
        for fc, lbl in bands:
            axes[0, col].axvline(fc, color="grey", ls=":", alpha=0.4)
        if col == 0:
            axes[0, col].set_ylabel("PC[all](f)", color=COLOR_PC)

        # R row
        axes[1, col].plot(f, r.resonance_spectrum["all"], color=COLOR_R, lw=1.4)
        axes[1, col].fill_between(f, 0, r.resonance_spectrum["all"], color=COLOR_R, alpha=0.13)
        axes[1, col].set_xlabel("Frequency (Hz)")
        axes[1, col].set_xlim(f.min(), f.max())
        for fc, lbl in bands:
            axes[1, col].axvline(fc, color="grey", ls=":", alpha=0.4)
            ymax = axes[1, col].get_ylim()[1]
            axes[1, col].text(fc, ymax * 0.93, lbl, ha="center", fontsize=9, color="grey")
        if col == 0:
            axes[1, col].set_ylabel("R[all](f)", color=COLOR_R)

    fig.suptitle("Figure 29 — Cross-channel PC refinements on multi-band EEG-pair (ch1 = 1/f + θ + α + β,  ch2 = 1/f + α + β + γ)\n"
                 "Joint-probability PC (b) frequency-localizes; n:m kernel (c) sharpens harmonic ratios; combined (d) does both",
                 fontsize=11, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "fig29_refinements_AB")


# ---------------------------------------------------------------------------
# Fig 30 — Refinement C: 3 surrogate types compared
# ---------------------------------------------------------------------------


def fig30_refinement_C_surrogates():
    """Compare the 3 surrogate generators (phase_randomize, iaaft, time_shuffle)
    in terms of cross-R(f) null distribution width and z-score power on the
    multi-band EEG-pair."""
    print("Fig 30: refinement C (surrogate null comparison) ...")
    from biotuner.resonance.nulls import (
        phase_randomize_surrogate, iaaft_surrogate, time_shuffle_surrogate,
    )

    sf = 500
    sig1, sig2 = _multiband_eeg_pair(sf=sf, duration=20.0)
    # Use the JOINT PC reducer for this comparison since it discriminates better
    cfg = _default_cfg(precision_hz=0.5, fmin=2, fmax=30, cross_pc_reducer="joint")
    real = compute_cross_resonance(sig1, sig2, sf=sf, config=cfg)
    n_freqs = real.freqs.size
    f = real.freqs

    SURR_GENS = {
        "phase_randomize": phase_randomize_surrogate,
        "iaaft": (lambda s, rng: iaaft_surrogate(s, rng, n_iter=80)),
        "time_shuffle": time_shuffle_surrogate,
    }
    n_surr = 80
    rng_master = np.random.default_rng(42)

    null_R = {}
    null_z = {}
    for name, gen in SURR_GENS.items():
        print(f"  {name}: running {n_surr} surrogates ...")
        t0 = time.time()
        surr_R = np.empty((n_surr, n_freqs))
        for k in range(n_surr):
            s1_s = gen(sig1, np.random.default_rng(rng_master.integers(0, 2**31)))
            s2_s = gen(sig2, np.random.default_rng(rng_master.integers(0, 2**31)))
            r = compute_cross_resonance(s1_s, s2_s, sf=sf, config=cfg)
            surr_R[k] = r.resonance_spectrum["all"]
        null_R[name] = surr_R
        mu, sd = surr_R.mean(axis=0), surr_R.std(axis=0) + 1e-12
        null_z[name] = (real.resonance_spectrum["all"] - mu) / sd
        print(f"    done in {time.time()-t0:.1f}s")

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
    palette = {"phase_randomize": "#37474f", "iaaft": "#1565c0", "time_shuffle": "#c62828"}

    for col, name in enumerate(SURR_GENS):
        surr_R = null_R[name]
        z_R = null_z[name]
        p10, p90 = np.percentile(surr_R, [5, 95], axis=0)
        mu = surr_R.mean(axis=0)
        # Top row: observed R vs surrogate band
        axes[0, col].fill_between(f, p10, p90, color="grey", alpha=0.4, label="5-95%")
        axes[0, col].plot(f, mu, color="grey", lw=1.0, label="mean")
        axes[0, col].plot(f, real.resonance_spectrum["all"], color=palette[name], lw=1.6, label="Observed R")
        axes[0, col].set_title(f"{name}", fontsize=10, color=palette[name])
        axes[0, col].legend(loc="upper right", fontsize=8)
        axes[0, col].set_xlim(f.min(), f.max())
        if col == 0:
            axes[0, col].set_ylabel("Cross R(f)")

        # Bottom row: z-scored R
        axes[1, col].plot(f, z_R, color=palette[name], lw=1.5, label="z(R)")
        axes[1, col].axhline(0, color="k", lw=0.4)
        axes[1, col].axhline(2, color="k", lw=0.4, ls="--", label="z=±2")
        axes[1, col].axhline(-2, color="k", lw=0.4, ls="--")
        sig_mask = z_R > 2
        if sig_mask.any():
            axes[1, col].fill_between(f, 0, z_R, where=sig_mask, color=palette[name], alpha=0.18,
                                       label=f"z>2 ({sig_mask.sum()} bins)")
        axes[1, col].set_xlabel("Frequency (Hz)")
        axes[1, col].legend(loc="upper right", fontsize=8)
        axes[1, col].set_xlim(f.min(), f.max())
        if col == 0:
            axes[1, col].set_ylabel("z-score")

    fig.suptitle("Figure 30 — Cross-channel surrogate-null comparison (multi-band EEG-pair, joint PC reducer)\n"
                 "Tighter nulls (IAAFT, time_shuffle) produce stronger z-scores at shared α/β carriers",
                 fontsize=11, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    _save(fig, "fig30_surrogate_null_comparison")


def main():
    print("Generating cross-channel validation figures ...")
    fig21_signal_pair_regimes()
    fig22_three_flavors()
    fig23_kernel_swap()
    fig24_legacy_equivalence()
    fig25_kuramoto_pair_sweep()
    fig26_multiband_eeg_pair()
    fig27_cross_snr_sweep()
    fig28_cross_surrogate_null()
    fig29_refinements_AB()
    fig30_refinement_C_surrogates()
    print(f"\nAll figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
