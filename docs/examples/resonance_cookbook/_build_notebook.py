"""Build the resonance_cookbook.ipynb from a list of (kind, source) cells.

Run this once to (re)generate the notebook. Keeping cells defined in Python
keeps the notebook content under git review-friendly diffs.

Usage:
    python docs/examples/resonance_cookbook/_build_notebook.py
"""
import json
from pathlib import Path

OUT = Path(__file__).parent / "resonance_cookbook.ipynb"

# Each cell: (kind, source). kind in {'md', 'code'}.
CELLS = []
md = lambda s: CELLS.append(("md", s))
code = lambda s: CELLS.append(("code", s))

# ---------------------------------------------------------------------------
md("""
# Resonance & Connectivity Cookbook

A practical tour of `biotuner.resonance`, `biotuner.harmonic_spectrum`, and
`biotuner.harmonic_connectivity` — the three modules that together cover
single-signal harmonic analysis and cross-channel connectivity.

This notebook walks through ten workflow categories:

1. **Setup** — synthetic signals + strategy discovery
2. **H-only spectrum** — `compute_harmonic_spectrum`
3. **Full single-signal R(f)** — `compute_resonance`
4. **Swapping kernels** — harmonic kernel
5. **Swapping coupling metrics** — including the n:m convention rule
6. **Swapping ratio kernels** — binary vs. fraction
7. **Swapping combine rules**
8. **Surrogate-null normalization**
9. **Cross-channel resonance** — `compute_cross_resonance`, 3 reducer flavors
10. **Connectivity matrices + statistical inference**

The cells are self-contained — each section can be run independently after the
"Setup" block has been executed.
""")

# ---------------------------------------------------------------------------
md("## 1. Setup\n\nImports + small helpers + synthetic-signal generators.")

code("""
import numpy as np
import matplotlib.pyplot as plt

# biotuner.resonance — single-signal H × PC = R machinery
from biotuner.resonance import (
    compute_resonance, ResonanceConfig, ResonanceResult,
    with_surrogate_null, list_strategies, results_to_dataframe,
)

# biotuner.harmonic_spectrum — narrow H-only entry point
from biotuner.harmonic_spectrum import compute_harmonic_spectrum

# biotuner.harmonic_connectivity — cross-channel APIs
from biotuner.harmonic_connectivity import (
    harmonic_connectivity, compute_cross_resonance, CrossResonanceResult,
)

%matplotlib inline
plt.rcParams.update({"figure.dpi": 110, "lines.linewidth": 1.4,
                      "axes.spines.top": False, "axes.spines.right": False})
""")

code("""
SF = 500   # sampling frequency for all examples

def pink_noise(n, sf, seed=0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(n)
    f = np.fft.rfftfreq(n, 1/sf); f[0] = f[1]
    return np.fft.irfft(np.fft.rfft(w) / np.sqrt(f), n=n)

def harmonic_signal(sf=SF, duration=8.0, freqs=(5, 10, 20, 40), amp_decay=0.7, seed=0):
    \"\"\"Harmonic stack: sum of sinusoids at the given freqs, decaying amplitude.\"\"\"
    t = np.arange(int(sf * duration)) / sf
    sig = sum((amp_decay ** i) * np.sin(2 * np.pi * f * t) for i, f in enumerate(freqs))
    sig += 0.05 * pink_noise(len(t), sf, seed=seed)
    return sig

def alpha_burst_signal(sf=SF, duration=8.0, alpha_freq=10.0, n_bursts=6, seed=0):
    \"\"\"EEG-like: 1/f background + Hann-windowed alpha bursts.\"\"\"
    rng = np.random.default_rng(seed)
    n = int(sf * duration)
    t = np.arange(n) / sf
    sig = 1.0 * pink_noise(n, sf, seed=seed)
    burst_times = np.linspace(0.5, duration - 1.0, n_bursts) + rng.uniform(-0.1, 0.1, size=n_bursts)
    for bt in burst_times:
        idx = (t >= bt) & (t <= bt + 0.6)
        local = t[idx] - bt
        win = np.sin(np.pi * local / 0.6) ** 2
        sig[idx] += 2.0 * win * np.sin(2 * np.pi * alpha_freq * local + rng.uniform(0, 2 * np.pi))
    return sig

# Display a sample signal
sig = harmonic_signal()
fig, ax = plt.subplots(figsize=(10, 2.5))
t = np.arange(len(sig))/SF
ax.plot(t[:int(SF*2)], sig[:int(SF*2)], color='#37474f', lw=0.7)
ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
ax.set_title("Sample synthetic signal: harmonic stack 5/10/20/40 Hz on pink noise")
plt.show()
""")

# ---------------------------------------------------------------------------
md("""
### 1b. Discovery: what strategies are available?

`list_strategies()` prints every registered kernel, coupling metric, combine
rule, etc. All names returned are valid for the corresponding
`ResonanceConfig` field.
""")

code("list_strategies()")

# ---------------------------------------------------------------------------
md("""
## 2. H-only spectrum (single signal)

For just the harmonicity factor H(f) — without the full PC and R computation —
use `compute_harmonic_spectrum`. Returns the spectrum plus a rich complexity
summary dict.
""")

code("""
sig = harmonic_signal()

freqs, H, S, summary = compute_harmonic_spectrum(
    sig, precision_hz=0.5, fmin=2, fmax=30, fs=SF,
)

print("H shape:", H.shape)
print("S (kernel matrix) shape:", S.shape)
print("summary keys:", sorted(summary.keys()))
print("flatness:", summary['flatness'])
print("entropy:", summary['entropy'])
print("peaks:", summary['peaks'])

fig, ax = plt.subplots(figsize=(10, 3))
ax.plot(freqs, H, color='#1a237e')
ax.fill_between(freqs, 0, H, color='#1a237e', alpha=0.15)
ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Harmonicity H(f)")
ax.set_title("compute_harmonic_spectrum on a 5/10/20/40 Hz harmonic stack")
plt.show()
""")

# ---------------------------------------------------------------------------
md("""
## 3. Full single-signal resonance R = H × PC

`compute_resonance` runs the full pipeline: harmonic kernel → ratio kernel →
phase estimator → coupling metric → reducers → combine.
""")

code("""
sig = harmonic_signal()
result = compute_resonance(sig, sf=SF)

print("type:", type(result).__name__)
print("freqs:", result.freqs.shape)
print("factors:", list(result.factors.keys()))
print("resonance_spectrum:", result.resonance_spectrum.shape)
print("peaks:", result.peaks)
print()
print("Summaries (complexity per spectrum):")
for k, s in result.summaries.items():
    print(f"  {k}: flatness={s['flatness']:.3f}  entropy={s['entropy']:.3f}  "
          f"avg={s['avg']:.3g}  peaks={s['peaks']}")
""")

code("""
freqs = result.freqs
H = result.factors['H']; PC = result.factors['PC']; R = result.resonance_spectrum

fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
for ax, (name, vals, color) in zip(axes, [
    ('H — harmonicity', H, '#1a237e'),
    ('PC — phase coupling', PC, '#6a1b9a'),
    ('R = H · PC — resonance', R, '#b71c1c'),
]):
    ax.plot(freqs, vals, color=color)
    ax.fill_between(freqs, 0, vals, color=color, alpha=0.15)
    ax.set_ylabel(name)
axes[-1].set_xlabel("Frequency (Hz)")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------
md("""
## 4. Swapping harmonic kernels

Two harmonic kernels are registered: `harmsim` (Gill-Purves dyad similarity)
and `subharm_tension` (Chan subharmonic tension, inverted). They give
qualitatively similar peak structure but differ in absolute magnitude and how
strongly they penalize complex ratios.
""")

code("""
sig = harmonic_signal()

results = {}
for kernel in ['harmsim', 'subharm_tension']:
    cfg = ResonanceConfig(harmonic_kernel=kernel)
    results[kernel] = compute_resonance(sig, sf=SF, config=cfg)

fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
for ax, (kernel, r) in zip(axes, results.items()):
    ax.plot(r.freqs, r.factors['H'], color='#1a237e', label='H(f)')
    ax.fill_between(r.freqs, 0, r.factors['H'], color='#1a237e', alpha=0.15)
    ax.set_title(f"harmonic_kernel='{kernel}'")
    ax.set_ylabel('H(f)')
axes[-1].set_xlabel("Frequency (Hz)")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------
md("""
## 5. Coupling metrics — and the n:m convention rule

Six pairwise metrics are registered, falling into two input-type categories:

- **`phase` inputs** — operate on real-valued phase angles (e.g. STFT-bin phase):
  `nm_plv`, `nm_pli`, `nm_wpli`, `nm_rrci`, `nm_plv_canonical`.
- **`analytic` inputs** — operate on complex analytic signals (carries amplitude
  AND phase): `nm_wpli_complex`.

### Important: convention rule

All ratio kernels in biotuner (`binary`, `fraction`, future Arnold-tongue)
return `(n, m)` with the convention `ratio = f_j / f_i = m / n`. The plain
`nm_plv` applies this as `n·φ_i − m·φ_j`, which is mathematically wrong for true
n:m phase locking under Tass 1998. **For correct n:m phase coupling tests, use
`coupling_metric='nm_plv_canonical'`** — it swaps `(n, m)` internally to apply
the Tass formula.

`nm_plv` is kept for **bit-exact reproduction** of legacy paper analyses only.
""")

code("""
sig = harmonic_signal()

# Compare the 5 phase-input metrics on the same signal
metrics = ['nm_plv', 'nm_plv_canonical', 'nm_pli', 'nm_wpli', 'nm_rrci']
fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 9), sharex=True)
for ax, metric in zip(axes, metrics):
    cfg = ResonanceConfig(coupling_metric=metric)
    r = compute_resonance(sig, sf=SF, config=cfg)
    ax.plot(r.freqs, r.factors['PC'], color='#6a1b9a')
    ax.fill_between(r.freqs, 0, r.factors['PC'], color='#6a1b9a', alpha=0.15)
    ax.set_ylabel(metric)
axes[-1].set_xlabel("Frequency (Hz)")
axes[0].set_title("Phase-coupling factor PC(f) across pairwise metrics")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------
md("""
## 6. Ratio kernels — `binary` (legacy) vs `fraction` (new)

The ratio kernel decides which (n, m) integer pair to test at each frequency
pair, and how heavily to weight the result.

- **`binary`** — tries (n, m) with 1 ≤ n, m ≤ max_nm (default 3) and picks the
  best match within 5% tolerance. Returns W=1 if matched, W=0 otherwise
  (or W=1 at 1:1 if `fallback_to_1_1=True`). Misses any ratio outside the
  small preset table.
- **`fraction`** — computes `Fraction(f_j / f_i).limit_denominator(max_denom)`
  to find the EXACT closest rational for ANY frequency pair. Weights via Tenney
  height: `W = exp(-β · log₂(n·m))` so simple ratios dominate.

Both share the same `(n, m)` convention, so pairing with `nm_plv_canonical`
gives correct n:m tests regardless of which ratio kernel you pick.
""")

code("""
sig = harmonic_signal()

for kernel_name, kernel_params in [
    ('binary',  {'max_nm': 3, 'tolerance': 0.05, 'fallback_to_1_1': True}),
    ('fraction', {'max_denom': 16, 'beta': 1.0}),
]:
    cfg = ResonanceConfig(
        ratio_kernel=kernel_name,
        ratio_kernel_params=kernel_params,
        coupling_metric='nm_plv_canonical',  # always pair with canonical
    )
    r = compute_resonance(sig, sf=SF, config=cfg)
    print(f"ratio_kernel='{kernel_name}': PC peaks = {r.peaks['PC']}, "
          f"max PC = {r.factors['PC'].max():.4g}")
""")

# ---------------------------------------------------------------------------
md("""
## 7. Combine rules — H × PC → R

Five combine rules. Default `product` (R = H · PC) is the legacy semantics.
""")

code("""
sig = harmonic_signal()

fig, axes = plt.subplots(5, 1, figsize=(10, 9), sharex=True)
for ax, combine in zip(axes, ['product', 'geomean', 'harmmean', 'min', 'weighted_log']):
    cfg = ResonanceConfig(combine=combine)
    r = compute_resonance(sig, sf=SF, config=cfg)
    ax.plot(r.freqs, r.resonance_spectrum, color='#b71c1c')
    ax.fill_between(r.freqs, 0, r.resonance_spectrum, color='#b71c1c', alpha=0.15)
    ax.set_ylabel(combine)
axes[-1].set_xlabel("Frequency (Hz)")
axes[0].set_title("Combine rules: R(f) shape under different aggregation")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------
md("""
## 8. Surrogate-null normalization

For statistical inference, `with_surrogate_null` runs the resonance pipeline
on the observed signal AND on `n` AAFT/phase-randomized surrogates, then
z-scores the observed against the surrogate distribution.

For a noisy alpha-burst signal, the surrogates destroy phase structure while
preserving PSD — so high z-scores at the alpha carrier are evidence of true
phase-locked alpha, not just elevated PSD.
""")

code("""
sig = alpha_burst_signal()

cfg = ResonanceConfig(precision_hz=0.5, fmin=2, fmax=30)
result_z = with_surrogate_null(
    sig, sf=SF, config=cfg, surr_type='AAFT', n=50, correction='both', parallel=False,
)

freqs = result_z.freqs
R = result_z.resonance_spectrum
z = result_z.resonance_spectrum_z
p = result_z.summaries.get('p_value_spectrum')

fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
axes[0].plot(freqs, R, color='#b71c1c')
axes[0].fill_between(freqs, 0, R, color='#b71c1c', alpha=0.15)
axes[0].set_ylabel("Observed R(f)")

axes[1].plot(freqs, z, color='#00838f')
axes[1].axhline(0, color='k', lw=0.5); axes[1].axhline(2, color='k', ls='--', lw=0.5)
sig_mask = z > 2
if sig_mask.any():
    axes[1].fill_between(freqs, 0, z, where=sig_mask, color='#b71c1c', alpha=0.2,
                          label=f"z > 2 ({sig_mask.sum()} bins)")
    axes[1].legend()
axes[1].set_ylabel("z(R) vs surrogate")
axes[1].set_xlabel("Frequency (Hz)")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------
md("""
## 9. Cross-channel resonance — `compute_cross_resonance`

The cross-channel analog of `compute_resonance`, for two signals. Returns a
`CrossResonanceResult` with **three reducer flavors** per factor:

- `'1to2'` — asymmetric (signal1 at i, signal2 at j)
- `'2to1'` — transposed
- `'all'` — symmetrized average

The three flavors expose directional information that the symmetric average
loses.
""")

code("""
# Two coupled alpha signals (signal2 = signal1 shifted by π/4)
t = np.arange(int(SF * 8)) / SF
sig1 = np.sin(2 * np.pi * 10 * t) + 0.25 * pink_noise(len(t), SF, seed=1)
sig2 = np.sin(2 * np.pi * 10 * t + np.pi/4) + 0.25 * pink_noise(len(t), SF, seed=2)

cross = compute_cross_resonance(sig1, sig2, sf=SF)

print("Flavors:", list(cross.resonance_spectrum.keys()))
print("Factors:", list(cross.factors.keys()))
print(f"R peak (all flavor): {cross.resonance_spectrum['all'].max():.4f}")
print(f"PC peak (all flavor): {cross.factors['PC']['all'].max():.4f}")
""")

code("""
freqs = cross.freqs
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
for ax, (name, color) in zip(axes, [
    ('H', '#1a237e'), ('PC', '#6a1b9a'), ('R', '#b71c1c'),
]):
    for flavor, ls in zip(['all', '1to2', '2to1'], ['-', '--', ':']):
        if name == 'R':
            vals = cross.resonance_spectrum[flavor]
        else:
            vals = cross.factors[name][flavor]
        ax.plot(freqs, vals, color=color, ls=ls, label=f"{flavor}")
    ax.set_ylabel(name); ax.legend(loc='upper right', fontsize=8)
axes[-1].set_xlabel("Frequency (Hz)")
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------
md("""
## 10. Connectivity matrices on multi-channel data

Use the `harmonic_connectivity` class for N-channel analyses. It exposes both
the legacy peak-based methods and the new spectrum-based ones.
""")

code("""
# Build a 4-channel synthetic dataset
def build_4ch_data(sf=SF, duration=8.0):
    t = np.arange(int(sf * duration)) / sf
    rng = np.random.default_rng(0)
    noise = lambda seed: 0.25 * pink_noise(len(t), sf, seed=seed)
    return np.stack([
        np.sin(2*np.pi*10*t)          + noise(1),  # e1: clean 10 Hz
        np.sin(2*np.pi*10*t + np.pi/4) + noise(2),  # e2: 10 Hz phase-locked to e1
        np.sin(2*np.pi*20*t + np.pi/8) + noise(3),  # e3: 1:2 harmonic of e1
        np.sin(2*np.pi*17*t + 1.7)     + noise(4),  # e4: independent
    ])

data = build_4ch_data()

hc = harmonic_connectivity(
    sf=SF, data=data, peaks_function='FOOOF',
    precision=0.5, n_harm=5, min_freq=2, max_freq=30, n_peaks=3,
)
print("data shape:", data.shape)
""")

md("### 10a. Peak-based connectivity (legacy + new)")

code("""
# Legacy peak-based H: harmsim
M_harmsim = hc.compute_harm_connectivity(metric='harmsim', graph=False)

# New: peak-based phase coupling (registry-dispatched)
M_pc = hc.compute_peak_phase_coupling_connectivity(
    coupling_metric='nm_plv', graph=False,
)

# New: peak-based H × PC = R
M_r = hc.compute_peak_resonance_connectivity(
    harm_metric='harmsim', coupling_metric='nm_plv', combine='product',
    graph=False,
)

import seaborn as sns
labels = ['e1\\n10Hz', 'e2\\n10Hz\\nlocked', 'e3\\n20Hz', 'e4\\n17Hz']
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, M, title in zip(axes, [M_harmsim, M_pc, M_r],
                         ['peak harmsim H', 'peak nm_plv PC', 'peak R = H · PC']):
    sns.heatmap(np.nan_to_num(M), ax=ax, annot=True, fmt='.2g',
                xticklabels=labels, yticklabels=labels, cbar=False)
    ax.set_title(title)
plt.tight_layout(); plt.show()
""")

md("### 10b. Spectrum-based cross-resonance matrix")

code("""
# Spectrum-based: loop compute_cross_resonance over all pairs
cfg = ResonanceConfig(precision_hz=0.5, fmin=2, fmax=30)

M_max = hc.compute_cross_resonance_connectivity(
    config=cfg, factor='R', flavor='all', aggregate='max', graph=False,
)
M_p2m = hc.compute_cross_resonance_connectivity(
    config=cfg, factor='R', flavor='all', aggregate='peak_to_median', graph=False,
)
M_pz = hc.compute_cross_resonance_connectivity(
    config=cfg, factor='R', flavor='all', aggregate='peak_z', graph=False,
)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, M, title in zip(axes, [M_max, M_p2m, M_pz],
                         ['aggregate=max\\n(broadband-biased)',
                          'aggregate=peak_to_median\\n(DEFAULT)',
                          'aggregate=peak_z']):
    sns.heatmap(np.nan_to_num(M), ax=ax, annot=True, fmt='.2g',
                xticklabels=labels, yticklabels=labels, cbar=False,
                cmap='viridis')
    ax.set_title(title)
plt.tight_layout(); plt.show()
""")

md("""
### 10c. Statistical inference: surrogate-z-scored connectivity matrix

For paper-quality discrimination between true cross-channel phase coupling and
broadband-power artifacts, run the connectivity matrix against an
**IAAFT** surrogate null (preserves per-channel PSD, destroys cross-channel
phase). High z-scores = real coupling above the broadband baseline.
""")

code("""
obs, z_matrix, p_matrix = hc.compute_cross_resonance_connectivity_zscore(
    config=cfg, factor='R', flavor='all', aggregate='peak_to_median',
    surrogate_kind='iaaft', n_surrogates=20, rng_seed=42,
    graph=False,
)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, M, title, cmap, kw in [
    (axes[0], obs, "Observed R", "viridis", {}),
    (axes[1], z_matrix, "z-score (IAAFT null)", "coolwarm", {"center": 0}),
    (axes[2], p_matrix, "Empirical p", "viridis_r", {"vmin": 0, "vmax": 1}),
]:
    sns.heatmap(np.nan_to_num(M), ax=ax, annot=True, fmt='.2g',
                xticklabels=labels, yticklabels=labels, cbar=False,
                cmap=cmap, **kw)
    ax.set_title(title)
plt.tight_layout(); plt.show()
""")

# ---------------------------------------------------------------------------
md("""
## 11. Reproducing legacy `compute_global_harmonicity`

Existing analyses that used the pre-refactor `compute_global_harmonicity`
reproduce bit-exactly by passing the **legacy preset** ResonanceConfig.
""")

code("""
# Build a legacy-equivalent ResonanceConfig
legacy_cfg = ResonanceConfig(
    psd_normalization='minmax_prob',           # legacy two-step PSD norm
    harmonic_kernel='harmsim',
    harmonic_kernel_params={'n_harms': 10, 'delta_lim': 20, 'min_notes': 2},
    ratio_kernel='binary',
    ratio_kernel_params={'max_nm': 3, 'tolerance': 0.05, 'fallback_to_1_1': True},
    phase_estimator='stft',
    coupling_metric='nm_plv',                  # legacy convention
    gaussian_smooth_sigma=1.0,
    legacy_self_pair_subtract=True,
    normalize=True, bandwidth_correction=False,
    combine='product',
)

sig = harmonic_signal()
result_legacy = compute_resonance(sig, sf=SF, config=legacy_cfg)
print("This reproduces the pre-refactor compute_global_harmonicity numerics "
      "within float-precision.")
print("Recommended for: reproducing published paper outputs only.")
""")

md("""
For **new analyses**, the recommended config is the default:

```python
result = compute_resonance(signal, sf=1000)
```

which uses the refined defaults (joint PC reducer, n:m ratio kernel where
applicable, `nm_plv_canonical`-compatible conventions).

---

## Further reading

- **Module docstrings** — `help(biotuner.resonance)`, `help(biotuner.harmonic_spectrum)`,
  `help(biotuner.harmonic_connectivity)` — each has a Quick Start.
- **`list_strategies()`** — discoverable inventory of every registered kernel,
  metric, and combine rule.
- **Sphinx API docs** — under `docs/api/resonance.rst`,
  `docs/api/harmonic_spectrum.rst`, `docs/api/harmonic_connectivity.rst`.
- **Plan** — `biotuner_resonance_plan.md` documents the full Phase 1 / 2 / 3
  roadmap and references for every algorithm.

### What's not in this notebook (Phase 2/3 additions)

The registry slots exist for these but aren't filled yet:

- More harmonic kernels: `sethares`, `stolzenburg`, `hopf`, `lorentzian`,
  `harmonic_entropy`, plus the existing-but-not-wired biotuner metrics
  `tenney_height`, `euler`, `compute_consonance`, `integral_tenney_height`.
- More ratio kernels: `arnold_tongue` (soft Gaussian), `stern_brocot`.
- More phase estimators: `hilbert_bandpass`, `morlet_wavelet`.
- Higher-order coupling: `bplv` (triplet), `mplv` (N-ary), `cf_plm`, `gpla`.
- Persistence (Q-factor) axis: `lagged_coherence`, `lhac`, `fooof_bandwidth`.

Each lands as a one-line `register_*` call against the existing registry —
contributions welcome.
""")


def build_notebook():
    cells_json = []
    for kind, src in CELLS:
        if kind == "md":
            cells_json.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [line + "\n" for line in src.strip().split("\n")],
            })
        else:
            cells_json.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in src.strip().split("\n")],
            })
    nb = {
        "cells": cells_json,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1))
    print(f"Wrote {OUT}")
    print(f"  {len(CELLS)} cells ({sum(1 for k, _ in CELLS if k == 'md')} md, "
          f"{sum(1 for k, _ in CELLS if k == 'code')} code)")


if __name__ == "__main__":
    build_notebook()
