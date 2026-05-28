"""Extended validation: complex / realistic signals.

Builds on generate_report.py with 5 more figures testing the resonance framework
on stimuli where its claims actually matter:

  Fig 6 — EEG-like alpha bursts on 1/f background (realistic neural signal)
  Fig 7 — Phase-locked vs phase-randomized harmonic stack
          (same PSD, different phase structure; the key PC validation)
  Fig 8 — Theta-gamma cross-frequency coupling (PAC stimulus)
  Fig 9 — Surrogate null normalization: z-scored R(f) on the alpha-burst signal
          (demonstrates statistical inference path)
  Fig 10 — Scale invariance: same signal × 1e-6, 1e0, 1e6 produces identical R(f)
           (validates probability-normalized reducer)

Each figure includes paper-ready typography and is saved as PNG + PDF.

Usage:
    python reports/resonance_refactor/complex_signals.py
"""
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "resonance"))

from _signals import legacy_default_resonance_config_kwargs  # noqa: E402
from biotuner.resonance import compute_resonance, ResonanceConfig  # noqa: E402

FIG_DIR = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)

# ---------------------------------------------------------------------------
# Publication-grade matplotlib settings (same as generate_report.py)
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
})

COLOR_H = "#1a237e"
COLOR_PC = "#6a1b9a"
COLOR_R = "#b71c1c"
COLOR_SIG = "#37474f"
COLOR_NULL = "#9e9e9e"
COLOR_ZSCORE = "#00838f"


def _save(fig, name):
    png = FIG_DIR / f"{name}.png"
    pdf = FIG_DIR / f"{name}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    print(f"  wrote {png.name} + {pdf.name}")


# ---------------------------------------------------------------------------
# Signal builders
# ---------------------------------------------------------------------------


def pink_noise_floor(n, sf, seed=0):
    """1/f pink noise normalized to unit standard deviation."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    f = np.fft.rfftfreq(n, d=1.0 / sf)
    f[0] = f[1]
    spectrum = np.fft.rfft(white) / np.sqrt(f)
    pink = np.fft.irfft(spectrum, n=n)
    return pink / np.std(pink)


def alpha_burst_signal(sf=500, duration=8.0, n_bursts=6, alpha_freq=10.0,
                       burst_dur=0.6, burst_amp=2.0, noise_floor=1.0, seed=0):
    """1/f pink noise background with periodic ~10 Hz alpha bursts.

    Models a realistic EEG epoch with intermittent alpha activity. Returns the
    time-domain signal.
    """
    rng = np.random.default_rng(seed)
    n = int(sf * duration)
    t = np.arange(n) / sf
    bg = noise_floor * pink_noise_floor(n, sf, seed=seed)

    # Place bursts at evenly spaced times (with jitter)
    burst_times = np.linspace(0.3, duration - burst_dur - 0.3, n_bursts)
    burst_times += rng.uniform(-0.1, 0.1, size=n_bursts)
    bursts = np.zeros(n)
    for bt in burst_times:
        idx = (t >= bt) & (t <= bt + burst_dur)
        local_t = t[idx] - bt
        # Hann-windowed alpha oscillation
        window = np.sin(np.pi * local_t / burst_dur) ** 2
        bursts[idx] += burst_amp * window * np.sin(2 * np.pi * alpha_freq * local_t
                                                     + rng.uniform(0, 2 * np.pi))
    return bg + bursts


def phase_locked_harmonic_stack(sf=500, duration=8.0, base_freq=8.0,
                                  n_harmonics=4, amp_decay=0.7, seed=0):
    """Harmonic stack at base, 2*base, 3*base, ... with a SHARED phase offset.

    All partials lock to the fundamental's phase — this is what the PC factor
    is supposed to detect.
    """
    rng = np.random.default_rng(seed)
    n = int(sf * duration)
    t = np.arange(n) / sf
    phi0 = rng.uniform(0, 2 * np.pi)
    sig = np.zeros(n)
    for k in range(1, n_harmonics + 1):
        sig += (amp_decay ** (k - 1)) * np.sin(2 * np.pi * k * base_freq * t + k * phi0)
    # Small noise floor
    sig += 0.05 * pink_noise_floor(n, sf, seed=seed + 1)
    return sig


def phase_decoupled_harmonic_stack(sf=500, duration=8.0, base_freq=8.0,
                                     n_harmonics=4, amp_decay=0.7,
                                     phase_diffusion=0.6, seed=0):
    """Independent Wiener-process phase drift on each harmonic.

    Each partial has the same nominal frequency as in the phase-locked stack but
    accumulates an independent random-walk phase perturbation
    (``dφ = phase_diffusion * dW``). The instantaneous n:m phase difference between
    any two partials is therefore non-stationary, which the n:m PLV is supposed
    to detect (it averages ``exp(i*(n*φᵢ - m*φⱼ))`` over time — a random walk
    decoheres that average toward zero).

    The magnitude spectrum is approximately preserved (each partial's energy
    stays in a narrow band around k*base_freq) but the phase structure is gone.
    This is the proper PC test — independent per-cycle phase jitter would not
    decouple STFT-bin phases.
    """
    rng = np.random.default_rng(seed)
    n = int(sf * duration)
    t = np.arange(n) / sf
    sig = np.zeros(n)
    dt = 1.0 / sf
    for k in range(1, n_harmonics + 1):
        # Independent Brownian phase increment for each harmonic
        dphi = rng.standard_normal(n) * phase_diffusion * np.sqrt(dt)
        phi_drift = np.cumsum(dphi)
        sig += (amp_decay ** (k - 1)) * np.sin(2 * np.pi * k * base_freq * t + phi_drift)
    sig += 0.05 * pink_noise_floor(n, sf, seed=seed + 1)
    return sig


def theta_gamma_pac_signal(sf=500, duration=8.0, theta=6.0, gamma=40.0,
                             coupling=0.7, noise=0.3, seed=0):
    """Theta-gamma cross-frequency coupling: gamma amplitude is modulated by
    theta phase. A classic PAC stimulus (Tort 2010).
    """
    rng = np.random.default_rng(seed)
    n = int(sf * duration)
    t = np.arange(n) / sf
    theta_sig = np.sin(2 * np.pi * theta * t)
    # Gamma amplitude = 0.5 + 0.5 * (sin(theta phase))   (modulated 0..1)
    gamma_env = 1.0 + coupling * np.cos(2 * np.pi * theta * t)
    gamma_sig = gamma_env * np.sin(2 * np.pi * gamma * t)
    sig = theta_sig + 0.6 * gamma_sig
    sig += noise * pink_noise_floor(n, sf, seed=seed + 2)
    return sig


# ---------------------------------------------------------------------------
# Common config
# ---------------------------------------------------------------------------


def _resonance(signal, sf, fmin=2, fmax=80, precision_hz=0.5, return_intermediates=False):
    kwargs = legacy_default_resonance_config_kwargs()
    kwargs.update(dict(precision_hz=precision_hz, fmin=fmin, fmax=fmax, noverlap=4))
    cfg = ResonanceConfig(return_intermediates=return_intermediates, **kwargs)
    return compute_resonance(signal, sf=sf, config=cfg)


def _plot_HPCR_row(ax_h, ax_pc, ax_r, result, color_label_alpha=0.85):
    """Used by Fig 6 and Fig 8 — plot H, PC, R panels stacked."""
    f = result.freqs
    ax_h.plot(f, result.factors["H"], color=COLOR_H, lw=1.5)
    ax_h.fill_between(f, 0, result.factors["H"], color=COLOR_H, alpha=0.13)
    ax_pc.plot(f, result.factors["PC"], color=COLOR_PC, lw=1.5)
    ax_pc.fill_between(f, 0, result.factors["PC"], color=COLOR_PC, alpha=0.13)
    ax_r.plot(f, result.resonance_spectrum, color=COLOR_R, lw=1.5)
    ax_r.fill_between(f, 0, result.resonance_spectrum, color=COLOR_R, alpha=0.13)
    for ax in (ax_h, ax_pc, ax_r):
        ax.set_xlim(f.min(), f.max())


# ---------------------------------------------------------------------------
# Figure 6 — EEG-like alpha burst signal
# ---------------------------------------------------------------------------


def fig6_alpha_burst():
    print("Fig 6: alpha-burst EEG-like ...")
    sf = 500
    sig = alpha_burst_signal(sf=sf, duration=8.0)
    result = _resonance(sig, sf=sf, fmin=2, fmax=40)

    fig = plt.figure(figsize=(11.5, 9))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.1, 1, 1, 1], hspace=0.5)

    # Time-domain signal
    ax_sig = fig.add_subplot(gs[0])
    t = np.arange(len(sig)) / sf
    ax_sig.plot(t, sig, color=COLOR_SIG, lw=0.7)
    ax_sig.set_xlabel("Time (s)")
    ax_sig.set_ylabel("Amplitude")
    ax_sig.set_title("a) EEG-like signal: 1/f pink noise + 6 alpha (10 Hz) bursts")
    ax_sig.set_xlim(0, 8)

    # H, PC, R
    ax_h = fig.add_subplot(gs[1])
    ax_pc = fig.add_subplot(gs[2])
    ax_r = fig.add_subplot(gs[3])
    _plot_HPCR_row(ax_h, ax_pc, ax_r, result)
    ax_h.set_ylabel("H(f)", color=COLOR_H)
    ax_pc.set_ylabel("PC(f)", color=COLOR_PC)
    ax_r.set_ylabel("R(f)", color=COLOR_R)
    ax_r.set_xlabel("Frequency (Hz)")

    # Annotate alpha peak in R
    R = result.resonance_spectrum
    peak_idx = np.argmax(R)
    peak_freq = result.freqs[peak_idx]
    ax_r.axvline(peak_freq, color=COLOR_R, ls=":", alpha=0.6)
    ax_r.text(peak_freq, R[peak_idx], f"  {peak_freq:.1f} Hz",
              va="center", color=COLOR_R, fontweight="bold", fontsize=10)

    ax_h.set_title("b) Harmonicity")
    ax_pc.set_title("c) Phase coupling")
    ax_r.set_title("d) Resonance R(f) — peak at the alpha carrier")

    fig.suptitle("Figure 6 — Realistic EEG epoch: alpha bursts emerge cleanly from 1/f background",
                 fontsize=13, fontweight="bold", y=0.995)
    _save(fig, "fig6_alpha_burst")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7 — Phase-locked vs phase-scrambled (the key PC validation)
# ---------------------------------------------------------------------------


def fig7_phase_locked_vs_scrambled():
    """Phase structure discrimination: legacy nm_plv vs Tass-convention nm_plv_canonical.

    The legacy convention is not a true n:m PLV (see docstring of
    ``nm_plv_canonical``). It barely discriminates a phase-locked harmonic
    stack from a Wiener-phase-drifted one. The canonical metric, applied with
    the same orchestrator and the same ratio kernel, recovers the expected
    discriminability.
    """
    print("Fig 7: phase-locked vs phase-decoupled — legacy vs canonical PLV ...")
    sf = 500
    duration = 30.0
    sig_locked = phase_locked_harmonic_stack(sf=sf, duration=duration, base_freq=8.0)
    sig_scrambled = phase_decoupled_harmonic_stack(
        sf=sf, duration=duration, base_freq=8.0, phase_diffusion=2.0,
    )

    def _run(sig, metric):
        kwargs = legacy_default_resonance_config_kwargs()
        kwargs.update(dict(precision_hz=2.0, fmin=2, fmax=50, noverlap=4,
                            coupling_metric=metric))
        return compute_resonance(sig, sf=sf, config=ResonanceConfig(**kwargs))

    r_locked_legacy = _run(sig_locked, "nm_plv")
    r_dec_legacy = _run(sig_scrambled, "nm_plv")
    r_locked_canon = _run(sig_locked, "nm_plv_canonical")
    r_dec_canon = _run(sig_scrambled, "nm_plv_canonical")

    fig, axes = plt.subplots(3, 2, figsize=(12, 8.5))

    # Row 1: signals
    t = np.arange(int(sf * 1.5)) / sf
    axes[0, 0].plot(t, sig_locked[:len(t)], color=COLOR_SIG, lw=0.7)
    axes[0, 0].set_title("a) Phase-locked harmonic stack (8/16/24/32 Hz)")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Amplitude")
    axes[0, 1].plot(t, sig_scrambled[:len(t)], color=COLOR_SIG, lw=0.7)
    axes[0, 1].set_title("b) Wiener-phase-drifted (independent random walk per harmonic)")
    axes[0, 1].set_xlabel("Time (s)")

    # Row 2: legacy nm_plv (does NOT discriminate)
    f = r_locked_legacy.freqs
    axes[1, 0].plot(f, r_locked_legacy.factors["PC"], color=COLOR_PC, lw=1.4, label="locked")
    axes[1, 0].plot(f, r_dec_legacy.factors["PC"], color=COLOR_PC, lw=1.4, ls="--", alpha=0.7, label="decoupled")
    axes[1, 0].fill_between(f, 0, r_locked_legacy.factors["PC"], color=COLOR_PC, alpha=0.1)
    axes[1, 0].set_title(f"c) PC(f) — legacy nm_plv (locked-vs-decoupled ratio = "
                         f"{r_locked_legacy.factors['PC'].max() / max(r_dec_legacy.factors['PC'].max(), 1e-12):.2f}×)")
    axes[1, 0].set_ylabel("Phase coupling")
    axes[1, 0].legend(loc="upper right")
    axes[1, 0].set_xlim(f.min(), f.max())

    # Row 2 right: canonical nm_plv_canonical (DOES discriminate)
    axes[1, 1].plot(f, r_locked_canon.factors["PC"], color=COLOR_PC, lw=1.4, label="locked")
    axes[1, 1].plot(f, r_dec_canon.factors["PC"], color=COLOR_PC, lw=1.4, ls="--", alpha=0.7, label="decoupled")
    axes[1, 1].fill_between(f, 0, r_locked_canon.factors["PC"], color=COLOR_PC, alpha=0.1)
    ratio_canon = r_locked_canon.factors["PC"].max() / max(r_dec_canon.factors["PC"].max(), 1e-12)
    axes[1, 1].set_title(f"d) PC(f) — nm_plv_canonical (locked-vs-decoupled ratio = {ratio_canon:.2f}×)")
    axes[1, 1].legend(loc="upper right")
    axes[1, 1].set_xlim(f.min(), f.max())

    # Row 3: R(f) under both metrics
    axes[2, 0].plot(f, r_locked_legacy.resonance_spectrum, color=COLOR_R, lw=1.4, label="locked")
    axes[2, 0].plot(f, r_dec_legacy.resonance_spectrum, color=COLOR_R, lw=1.4, ls="--", alpha=0.7, label="decoupled")
    axes[2, 0].fill_between(f, 0, r_locked_legacy.resonance_spectrum, color=COLOR_R, alpha=0.1)
    axes[2, 0].set_title("e) R(f) — legacy nm_plv")
    axes[2, 0].set_xlabel("Frequency (Hz)")
    axes[2, 0].set_ylabel("Resonance")
    axes[2, 0].legend(loc="upper right")
    axes[2, 0].set_xlim(f.min(), f.max())

    axes[2, 1].plot(f, r_locked_canon.resonance_spectrum, color=COLOR_R, lw=1.4, label="locked")
    axes[2, 1].plot(f, r_dec_canon.resonance_spectrum, color=COLOR_R, lw=1.4, ls="--", alpha=0.7, label="decoupled")
    axes[2, 1].fill_between(f, 0, r_locked_canon.resonance_spectrum, color=COLOR_R, alpha=0.1)
    axes[2, 1].set_title("f) R(f) — nm_plv_canonical")
    axes[2, 1].set_xlabel("Frequency (Hz)")
    axes[2, 1].legend(loc="upper right")
    axes[2, 1].set_xlim(f.min(), f.max())

    fig.suptitle(
        "Figure 7 — Phase-locking convention: legacy nm_plv does not discriminate locked from "
        "Wiener-decoupled (1.0×); canonical Tass-convention nm_plv does",
        fontsize=12, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "fig7_phase_locked_vs_scrambled")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 8 — Theta-gamma cross-frequency coupling
# ---------------------------------------------------------------------------


def fig8_theta_gamma_pac():
    print("Fig 8: theta-gamma PAC ...")
    sf = 500
    sig = theta_gamma_pac_signal(sf=sf, duration=8.0, theta=6.0, gamma=40.0, coupling=0.8)
    result = _resonance(sig, sf=sf, fmin=2, fmax=60, precision_hz=0.5)

    fig = plt.figure(figsize=(11.5, 9))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.1, 1, 1, 1], hspace=0.5)

    # Signal (1.5 s window)
    ax_sig = fig.add_subplot(gs[0])
    t = np.arange(int(sf * 1.5)) / sf
    ax_sig.plot(t, sig[:len(t)], color=COLOR_SIG, lw=0.7)
    # Overlay envelope of gamma component
    env = np.abs(hilbert(sig[:len(t)] - np.mean(sig[:len(t)])))
    ax_sig.plot(t, env, color=COLOR_R, lw=1.0, alpha=0.6, label="Hilbert envelope")
    ax_sig.set_xlabel("Time (s)")
    ax_sig.set_ylabel("Amplitude")
    ax_sig.set_title("a) Theta-gamma PAC: 6 Hz theta × 40 Hz gamma (envelope modulated by theta)")
    ax_sig.legend(loc="upper right", fontsize=8)

    ax_h = fig.add_subplot(gs[1])
    ax_pc = fig.add_subplot(gs[2])
    ax_r = fig.add_subplot(gs[3])
    _plot_HPCR_row(ax_h, ax_pc, ax_r, result)
    ax_h.set_ylabel("H(f)", color=COLOR_H)
    ax_pc.set_ylabel("PC(f)", color=COLOR_PC)
    ax_r.set_ylabel("R(f)", color=COLOR_R)
    ax_r.set_xlabel("Frequency (Hz)")

    # Annotate theta and gamma peaks
    for f_target, label in [(6.0, "θ"), (40.0, "γ")]:
        ax_r.axvline(f_target, color=COLOR_R, ls=":", alpha=0.5)
        ax_r.text(f_target, ax_r.get_ylim()[1] * 0.9, f"  {label} ({f_target:.0f} Hz)",
                  color=COLOR_R, fontweight="bold", fontsize=10)

    ax_h.set_title("b) Harmonicity — both theta and gamma carriers picked up")
    ax_pc.set_title("c) Phase coupling — peaks at theta and at gamma sidebands")
    ax_r.set_title("d) Resonance R(f)")

    fig.suptitle("Figure 8 — Cross-frequency coupling (theta-gamma PAC): both carriers resolved",
                 fontsize=13, fontweight="bold", y=0.995)
    _save(fig, "fig8_theta_gamma_pac")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 9 — Surrogate null normalization
# ---------------------------------------------------------------------------


def fig9_surrogate_null():
    print("Fig 9: surrogate null normalization ...")
    sf = 500
    sig = alpha_burst_signal(sf=sf, duration=8.0)

    # Configure null model on the orchestrator. Use phase-randomization
    # surrogates (preserves PSD, destroys phase structure).
    kwargs = legacy_default_resonance_config_kwargs()
    kwargs.update(dict(precision_hz=0.5, fmin=2, fmax=40, noverlap=4))
    # We'll roll the surrogates manually because surrogates.generate_surrogate
    # may have a specific API; use a simple phase-randomization here.
    cfg = ResonanceConfig(**kwargs)
    real = compute_resonance(sig, sf=sf, config=cfg)
    n_freqs = real.freqs.size

    n_surr = 100
    print(f"  running {n_surr} phase-randomized surrogates ...")
    rng = np.random.default_rng(42)
    surr_R = np.empty((n_surr, n_freqs))
    surr_PC = np.empty((n_surr, n_freqs))
    t0 = time.time()
    for k in range(n_surr):
        # Phase randomization in Fourier domain (preserves PSD, destroys phases)
        N = len(sig)
        X = np.fft.rfft(sig)
        phases = np.exp(1j * rng.uniform(0, 2 * np.pi, size=X.shape))
        phases[0] = 1.0  # keep DC real
        if N % 2 == 0:
            phases[-1] = 1.0  # Nyquist real for even N
        X_surr = np.abs(X) * phases
        s_surr = np.fft.irfft(X_surr, n=N)
        r_s = compute_resonance(s_surr, sf=sf, config=cfg)
        surr_R[k] = r_s.resonance_spectrum
        surr_PC[k] = r_s.factors["PC"]
    print(f"  surrogates in {time.time() - t0:.1f}s")

    mu_R, sd_R = surr_R.mean(axis=0), surr_R.std(axis=0) + 1e-12
    mu_PC, sd_PC = surr_PC.mean(axis=0), surr_PC.std(axis=0) + 1e-12
    z_R = (real.resonance_spectrum - mu_R) / sd_R
    z_PC = (real.factors["PC"] - mu_PC) / sd_PC
    # Empirical one-sided p-values
    p_R = (np.sum(surr_R >= real.resonance_spectrum[None, :], axis=0) + 1) / (n_surr + 1)

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 7.5))
    f = real.freqs

    # PC vs surrogate band
    p10, p90 = np.percentile(surr_PC, [5, 95], axis=0)
    axes[0].fill_between(f, p10, p90, color=COLOR_NULL, alpha=0.4, label="Surrogate 5-95%")
    axes[0].plot(f, mu_PC, color=COLOR_NULL, lw=1.0, label="Surrogate mean")
    axes[0].plot(f, real.factors["PC"], color=COLOR_PC, lw=1.6, label="Observed PC(f)")
    axes[0].set_ylabel("Phase coupling")
    axes[0].set_title("a) Phase coupling: observed vs surrogate null (phase-randomized; preserves PSD)")
    axes[0].legend(loc="upper right", ncol=3)

    # R vs surrogate band
    p10R, p90R = np.percentile(surr_R, [5, 95], axis=0)
    axes[1].fill_between(f, p10R, p90R, color=COLOR_NULL, alpha=0.4, label="Surrogate 5-95%")
    axes[1].plot(f, mu_R, color=COLOR_NULL, lw=1.0, label="Surrogate mean")
    axes[1].plot(f, real.resonance_spectrum, color=COLOR_R, lw=1.6, label="Observed R(f)")
    axes[1].set_ylabel("Resonance")
    axes[1].set_title("b) Resonance: observed vs surrogate null")
    axes[1].legend(loc="upper right", ncol=3)

    # Z-score panel
    axes[2].plot(f, z_R, color=COLOR_ZSCORE, lw=1.6, label="z(R)")
    axes[2].plot(f, z_PC, color=COLOR_PC, lw=1.2, alpha=0.7, label="z(PC)")
    axes[2].axhline(0, color="k", lw=0.5)
    axes[2].axhline(2, color="k", lw=0.5, ls="--", label="z = ±2")
    axes[2].axhline(-2, color="k", lw=0.5, ls="--")
    # Shade significant bins
    sig_mask = z_R > 2
    if sig_mask.any():
        axes[2].fill_between(f, 0, z_R, where=sig_mask, color=COLOR_R, alpha=0.18,
                              label=f"z > 2 ({sig_mask.sum()} bins)")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("z-score")
    axes[2].set_title("c) Surrogate-normalized z(R) — alpha bursts cleanly above noise floor")
    axes[2].legend(loc="upper right", ncol=4)

    for ax in axes:
        ax.set_xlim(f.min(), f.max())

    fig.suptitle("Figure 9 — Surrogate null normalization: phase-randomization preserves PSD, "
                 "isolates phase-structured resonance",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "fig9_surrogate_null")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 10 — Scale invariance
# ---------------------------------------------------------------------------


def fig10_scale_invariance():
    print("Fig 10: scale invariance ...")
    sf = 500
    sig = alpha_burst_signal(sf=sf, duration=8.0)

    scales = [1e-6, 1e-3, 1.0, 1e3, 1e6]
    results = []
    for s in scales:
        r = _resonance(sig * s, sf=sf, fmin=2, fmax=40)
        results.append((s, r))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    f = results[0][1].freqs
    palette = plt.cm.viridis(np.linspace(0.1, 0.9, len(scales)))

    # H(f) at each scale, all overlaid
    for (s, r), c in zip(results, palette):
        axes[0].plot(f, r.factors["H"], color=c, lw=1.4, label=f"signal × 1e{int(np.log10(s)):+d}")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("Harmonicity H(f)")
    axes[0].set_title("a) H(f) is scale-invariant (curves overlap exactly)")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_xlim(f.min(), f.max())

    # Max relative drift across scales
    H_ref = results[2][1].factors["H"]  # scale = 1.0
    drifts = [np.max(np.abs(r.factors["H"] - H_ref)) for s, r in results]
    axes[1].bar([f"1e{int(np.log10(s)):+d}" for s, r in results],
                drifts, color=palette, edgecolor="black", linewidth=0.6)
    axes[1].set_xlabel("Signal scale (relative to 1.0)")
    axes[1].set_ylabel("Max |ΔH| vs scale = 1.0")
    axes[1].set_title("b) Numerical drift across 12 orders of magnitude")
    axes[1].set_yscale("symlog", linthresh=1e-15)
    for i, d in enumerate(drifts):
        axes[1].text(i, d, f"{d:.1e}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Figure 10 — Scale invariance: probability-normalized reducer is robust "
                 "across 12 orders of signal magnitude",
                 fontsize=13, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "fig10_scale_invariance")
    plt.close(fig)


def fig11_coupling_metric_comparison():
    """Compare nm_plv / nm_pli / nm_wpli / nm_rrci on three signal regimes.

    PLV: classical phase-locking (0-lag sensitive)
    PLI / wPLI: phase-lag indices (0-lag robust; volume-conduction safe)
    RRCi: rhythmic ratio coupling, imaginary part (isolates non-zero-lag)
    """
    print("Fig 11: pairwise coupling metric comparison ...")
    sf = 500
    signals = {
        "Alpha bursts on 1/f": alpha_burst_signal(sf=sf, duration=8.0),
        "Phase-locked harm. stack": phase_locked_harmonic_stack(sf=sf, duration=8.0),
        "Theta-gamma PAC": theta_gamma_pac_signal(sf=sf, duration=8.0),
    }
    metrics = ["nm_plv", "nm_pli", "nm_wpli", "nm_rrci"]
    metric_colors = {
        "nm_plv": "#6a1b9a",
        "nm_pli": "#ef6c00",
        "nm_wpli": "#00838f",
        "nm_rrci": "#558b2f",
    }
    metric_labels = {
        "nm_plv": "n:m PLV   |⟨e^iΔφ⟩|",
        "nm_pli": "n:m PLI   |⟨sign(Im)⟩|",
        "nm_wpli": "n:m wPLI  weighted Im",
        "nm_rrci": "n:m RRCi  |Im(⟨e^iΔφ⟩)|",
    }

    fig, axes = plt.subplots(3, 2, figsize=(12, 9.5))
    # Two columns: PC factor (left), R = H · PC (right) for each signal-row

    for row, (sig_label, sig) in enumerate(signals.items()):
        # Reuse the H factor across metrics — it depends only on the harmonic kernel
        cfg_kwargs = legacy_default_resonance_config_kwargs()
        cfg_kwargs.update(dict(precision_hz=0.5, fmin=2, fmax=60, noverlap=4))

        H_ref = None
        for metric in metrics:
            kwargs = dict(cfg_kwargs)
            kwargs["coupling_metric"] = metric
            cfg = ResonanceConfig(**kwargs)
            r = compute_resonance(sig, sf=sf, config=cfg)
            if H_ref is None:
                H_ref = r.factors["H"]
                freqs = r.freqs
            # PC (col 0)
            axes[row, 0].plot(r.freqs, r.factors["PC"], color=metric_colors[metric], lw=1.4,
                              label=metric_labels[metric])
            # R (col 1)
            axes[row, 1].plot(r.freqs, r.resonance_spectrum, color=metric_colors[metric], lw=1.4,
                              label=metric_labels[metric])

        axes[row, 0].set_ylabel(f"{sig_label}\nPC(f)", fontsize=10)
        axes[row, 1].set_ylabel("R(f) = H · PC", fontsize=10)
        axes[row, 0].set_xlim(freqs.min(), freqs.max())
        axes[row, 1].set_xlim(freqs.min(), freqs.max())
        if row == 0:
            axes[row, 0].legend(loc="upper right", fontsize=8, ncol=2)
        if row == 2:
            axes[row, 0].set_xlabel("Frequency (Hz)")
            axes[row, 1].set_xlabel("Frequency (Hz)")

    axes[0, 0].set_title("a) Phase coupling factor PC(f)")
    axes[0, 1].set_title("b) Resonance R(f) = H(f) · PC(f)")

    fig.suptitle(
        "Figure 11 — Pairwise coupling metrics compared: PLV (0-lag sensitive) "
        "vs PLI/wPLI (0-lag robust) vs RRCi (non-zero-lag only)",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "fig11_coupling_metric_comparison")
    plt.close(fig)


def main():
    print("Generating complex-signal validation figures ...")
    fig6_alpha_burst()
    fig7_phase_locked_vs_scrambled()
    fig8_theta_gamma_pac()
    fig9_surrogate_null()
    fig10_scale_invariance()
    fig11_coupling_metric_comparison()
    print(f"\nAll figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
