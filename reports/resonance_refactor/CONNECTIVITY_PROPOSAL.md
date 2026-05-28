# Harmonic Connectivity — survey & extension proposal

## Question

Now that `biotuner.resonance` provides a strategy-registry of harmonic kernels,
phase estimators, pairwise coupling metrics, ratio gates, combine rules, and
reducers — can `biotuner.harmonic_connectivity` leverage it? Specifically:

1. **Keep** the existing peak-based harmonicity metric (current default behavior).
2. **Extend** with peak-based **phase coupling** and **resonance** metrics.
3. **Extend** with **harmonic + phase coupling + resonance cross-spectrum** metrics.

Short answer: yes, cleanly, via two new modules and one refactor — see "Proposal" below.

---

## Survey of current `harmonic_connectivity.py`

The 1774-line module already contains TWO parallel computation paths and
nine standalone coupling/MI utility functions. The new resonance package
unifies most of this. Here's the map:

### Path A — Peak-based connectivity (`compute_harm_connectivity`)

For each electrode pair, extract peaks from each, then collapse to a scalar
metric over the peak lists. Returns an `n_elec × n_elec` matrix.

| Metric name | Branch type | What it actually does today |
|---|---|---|
| `harmsim` | harmonic-only (H) | `np.mean(ratios2harmsim(pairwise ratios))` |
| `subharm_tension` | harmonic-only (H) | `compute_subharmonics_2lists(...)` |
| `harm_fit` | harmonic-only (H) | counts shared harmonics via `harmonic_fit` |
| `euler` | harmonic-only (H) | Euler's *gradus suavitatis* on concatenated peaks |
| `RRCi` | phase-coupling (PC) | for each peak pair: `cross_frequency_rrci` (bandpass + Hilbert + RRCi) |
| `wPLI_crossfreq` | phase-coupling (PC) | for each peak pair: `wPLI_crossfreq` (bandpass + Hilbert + wPLI) |
| `wPLI_multiband` | phase-coupling (PC) | wPLI on fixed band edges (not peak-based) |
| `MI` | phase-coupling (PC) | Hilbert phase + Mutual Information histogram |
| `MI_spectral` | phase-coupling (PC) | CWT phase + MI over peak pairs |

**Observations:**
- The H metrics use peak-list combinatorics.
- The PC metrics already do exactly the right thing: bandpass each signal at peak freq → Hilbert phase → pairwise n:m metric. *This is what the resonance package now centralizes via `nm_plv` / `nm_pli` / `nm_wpli` / `nm_rrci` / `nm_plv_canonical`.*
- **No `R` (peak-based resonance) metric exists.** It would be `H(peaks) × PC(peaks)` — trivial to add as a combine rule over the existing scalars.
- The current PC branches each carry their own bandpass + Hilbert plumbing (50+ lines repeated across `wPLI_crossfreq`, `cross_frequency_rrci`, `MI`). This is the obvious deduplication target.

### Path B — Cross-spectrum connectivity (`compute_harmonic_spectrum_connectivity` → `compute_cross_spectrum_harmonicity`)

For each electrode pair `(sig1, sig2)`, build:
- `dyad_similarities[i, j]` matrix (harmsim or subharm_tension on freq pairs)
- `phase_coupling_matrix[i, j]` from cross-spectrum STFT: `imag(X·Y*) / |imag(X·Y*)|` — this is **cross-channel wPLI** computed bin-by-bin
- Three reducer outputs:
  - `H1(f) = Σⱼ S[i,j] · p1[i] · p2[j]` (asymmetric, weighted by sig1 at i, sig2 at j)
  - `H2(f) = Σⱼ S[i,j] · p2[i] · p1[j]` (transposed)
  - `H_all(f) = (H1 + H2) / 2` (symmetrized)
- Same three flavors for `PC` and for `R = H · PC`
- A rich DataFrame of complexity metrics matching what the legacy `compute_global_harmonicity` produced (flatness, entropy, higuchi, spread, peaks, harmsim)

This is **the cross-channel analog of the single-channel resonance pipeline** — the math is the same, just with two PSDs and two phase arrays. The implementation duplicates a lot of what now lives in `biotuner.resonance` (kernels, ratio gates, reducers, complexity bundle).

### Utility functions (re-used by Path A)

- `wPLI_crossfreq`, `wPLI_multiband` — bandpass + Hilbert + wPLI
- `cross_frequency_rrci`, `rhythmic_ratio_coupling_imaginary`, `compute_rhythmic_ratio` — bandpass + Hilbert + n:m RRCi
- `n_m_phase_locking` — bandpass + Hilbert + nm_plv (on RAW signals; the resonance package's `nm_plv` operates on already-extracted phase)
- `compute_mutual_information`, `MI_spectral` — phase-MI variants
- `compute_cross_spectrum_harmonicity` — described above
- `HilbertHuang1D_nopeaks`, `EMD_time_resolved_harmonicity`, `compute_IMF_correlation` — IMF-based time-resolved variant (different beast; not a target of this refactor)
- `temporal_correlation_fdr` — generic FDR; stays as-is

---

## What the resonance package already provides

| Resonance API | What it gives harmonic_connectivity |
|---|---|
| `HARMONIC_KERNELS` registry (harmsim, subharm_tension; +5 in Phase 2) | The S matrix for cross-spectrum; the H scalar for peak-based |
| `PAIRWISE_COUPLING_METRICS` (nm_plv, nm_pli, nm_wpli, nm_rrci, nm_plv_canonical) | The PC matrix (cross-spectrum) and the PC scalar (peak-based) |
| `RATIO_KERNELS` (binary_nm; Arnold-tongue in Phase 2) | The (n, m) selection for both paths |
| `PHASE_ESTIMATORS` (stft; hilbert/morlet in Phase 2) | Phase extraction for PC |
| `COMBINE_RULES` (product, geomean, harmmean, min, weighted_log) | R from H × PC (peak or cross-spectrum), with options beyond legacy product |
| `reduce_matrix_to_spectrum(M, p, ...)` | Generic reducer; supports legacy self-pair-subtract AND clean off-diagonal modes |
| `spectrum_complexity(values, freqs, ...)` | The flatness/entropy/higuchi/spread/peaks bundle (shared with `compute_harmonic_spectrum` and `compute_resonance`) |
| `with_surrogate_null(...)` | Surrogate-normalized z(R) — applies one-channel for now; trivially extensible |
| `ResonanceConfig` dataclass | Single source of truth for all knobs |

---

## Proposal — three layers of extension

### Layer 1: Peak-based connectivity (extend existing path)

**Keep** `compute_harm_connectivity(metric='harmsim'|'subharm_tension'|...)` as-is. Then add **two new public methods**:

#### 1a. `compute_peak_phase_coupling_connectivity(coupling_metric='nm_plv', ratio_kernel='binary', ...)`

```python
def compute_peak_phase_coupling_connectivity(
    self,
    coupling_metric='nm_plv',          # any name in PAIRWISE_COUPLING_METRICS
    coupling_metric_params=None,
    ratio_kernel='binary',             # any name in RATIO_KERNELS
    ratio_kernel_params=None,
    bandwidth=1.0,                     # bandpass bandwidth around each peak (Hz)
    aggregate='mean',                  # 'mean'|'max' over peak pairs
    graph=True,
):
    """For each electrode pair:
       1. Extract peaks (uses self.peaks_function, self.precision, etc.)
       2. For each peak pair (f1, f2):
          - Look up (n, m) from the ratio kernel
          - Bandpass-filter each signal at its peak freq
          - Hilbert transform → instantaneous phase
          - Compute pairwise coupling metric (nm_plv, nm_pli, nm_wpli, nm_rrci, ...)
       3. Aggregate (mean/max) over peak pairs
    Returns the n_elec × n_elec scalar coupling matrix.
    """
```

This **replaces the ad-hoc `RRCi`, `wPLI_crossfreq`, `MI` branches** with a single registry-driven method. Adding new pairwise metrics in Phase 2/3 (gc-PAC, etc.) automatically becomes available here. The legacy branches stay as thin shims:
```python
# in compute_harm_connectivity:
if metric == 'RRCi':
    return self.compute_peak_phase_coupling_connectivity(coupling_metric='nm_rrci', ...)
if metric == 'wPLI_crossfreq':
    return self.compute_peak_phase_coupling_connectivity(coupling_metric='nm_wpli', ...)
```

#### 1b. `compute_peak_resonance_connectivity(harm_metric='harmsim', coupling_metric='nm_plv', combine='product')`

```python
def compute_peak_resonance_connectivity(
    self,
    harm_metric='harmsim',
    coupling_metric='nm_plv',
    combine='product',                 # any name in COMBINE_RULES
    coupling_metric_params=None,
    ratio_kernel='binary',
    ratio_kernel_params=None,
    bandwidth=1.0,
    graph=True,
):
    """Peak-based H × PC resonance per electrode pair.
    Returns the n_elec × n_elec resonance matrix.
    """
    H_matrix = self.compute_harm_connectivity(metric=harm_metric, graph=False)
    PC_matrix = self.compute_peak_phase_coupling_connectivity(
        coupling_metric=coupling_metric,
        coupling_metric_params=coupling_metric_params,
        ratio_kernel=ratio_kernel,
        ratio_kernel_params=ratio_kernel_params,
        bandwidth=bandwidth,
        graph=False,
    )
    combine_fn = COMBINE_RULES[combine]
    R_matrix = combine_fn([H_matrix, PC_matrix])
    if graph:
        sbn.heatmap(R_matrix)
        plt.title(f"Peak resonance connectivity ({harm_metric} × {coupling_metric})")
        plt.show()
    self.peak_resonance_matrix = R_matrix
    return R_matrix
```

This gives users a peak-based **R(electrode pair)** scalar — the connectivity analog of the per-frequency R(f) in the single-channel framework.

---

### Layer 2: Cross-spectrum (refactor existing path, add new clean API)

#### 2a. Add `biotuner/resonance/cross.py`

The cross-channel orchestrator. Mirrors `compute_resonance` but takes two signals:

```python
# biotuner/resonance/cross.py

@dataclass
class CrossResonanceResult:
    freqs: np.ndarray
    # Three flavors per spectrum (asymmetric × 2 + symmetrized)
    resonance_spectrum: dict       # {'1to2': R1, '2to1': R2, 'all': R_avg}
    factors: dict                  # {'H': {'1to2', '2to1', 'all'},
                                   #  'PC': {'1to2', '2to1', 'all'}}
    summaries: dict                # nested: factor → flavor → complexity dict
    peaks: dict                    # nested: factor → flavor → peak freqs
    config: ResonanceConfig
    # Optional null fields (same shape as factors / resonance_spectrum)
    resonance_spectrum_z: Optional[dict] = None
    intermediates: Optional[dict] = None


def compute_cross_resonance(
    signal1: np.ndarray,
    signal2: np.ndarray,
    sf: float,
    config: Optional[ResonanceConfig] = None,
) -> CrossResonanceResult:
    """Cross-channel resonance: H(f), PC(f), R(f) between two signals.

    Pipeline mirrors compute_resonance, with these differences:
      - Two PSDs (p1, p2)
      - Cross-spectrum phase coupling: Φ[i,j] = pairwise_metric(phase1[i], phase2[j], n, m)
        (instead of phase[i] vs phase[j] on the same signal)
      - Three reducer outputs (1to2, 2to1, all)
      - Optional null model destroys cross-channel coherence while preserving each PSD
    """
```

#### 2b. Refactor `compute_cross_spectrum_harmonicity` → thin shim

After `compute_cross_resonance` exists, the legacy `compute_cross_spectrum_harmonicity` becomes a one-shot wrapper that builds a `ResonanceConfig` with legacy defaults (`psd_normalization='minmax_prob'`, `phase_estimator='stft'`, `coupling_metric='nm_wpli'` to match the current cross-spectrum wPLI formula, `legacy_self_pair_subtract=True`, `combine='product'`), calls `compute_cross_resonance`, and shapes the output into the legacy DataFrame.

Bit-exact reproduction of current outputs is the snapshot regression target — same approach we used for the single-channel refactor.

#### 2c. Plug into the harmonic_connectivity class

```python
def compute_cross_resonance_connectivity(
    self,
    config: Optional[ResonanceConfig] = None,
    aggregate='max',                  # 'max'|'mean'|'sum' — how to reduce R(f) → scalar
    flavor='all',                     # which of the 3 reducer outputs to use
    graph=True,
):
    """For each electrode pair, run compute_cross_resonance and extract a
    scalar summary of R(f). Returns n_elec × n_elec matrix.

    Optionally returns the full list of CrossResonanceResult per pair via
    the attribute self.cross_resonance_results for downstream analysis
    (graph metrics, network analysis, etc.).
    """
```

And the existing `get_harm_spectrum_metric_matrix(metric)` keeps working — it just queries the DataFrame produced by the legacy shim (or, optionally, builds itself from the new `CrossResonanceResult` list).

---

### Layer 3: Surrogate normalization for connectivity

`with_surrogate_null` (currently single-channel) becomes multi-channel-aware. For cross-resonance the appropriate null is:

- **`iaaft_per_channel`** — IAAFT on each signal independently → preserves each PSD, destroys cross-channel phase
- **`phase_randomize_per_channel`** — Fourier phase randomization per channel → same property, faster

This is one extra function in `biotuner.resonance.nulls`. Connectivity gets z-scored R(f) per electrode pair for free.

---

## What this buys

| Before | After |
|---|---|
| 9 ad-hoc peak-based metrics, 1 cross-spectrum function | Same metrics + 2 new methods, all dispatching the resonance registry |
| RRCi/wPLI/MI each carry their own bandpass+Hilbert+formula plumbing | One implementation; 5 metrics swappable by name |
| No peak-based PC connectivity exposed cleanly (must go via specific metrics) | `compute_peak_phase_coupling_connectivity(coupling_metric=...)` |
| No peak-based R connectivity at all | `compute_peak_resonance_connectivity(combine='product'|'geomean'|...)` |
| Cross-spectrum tied to legacy STFT-wPLI convention | Cross-spectrum dispatches any registered pairwise metric (incl. canonical PLV) |
| No statistical inference on connectivity matrices | Surrogate-normalized z(R) per pair |
| Phase-2 kernels (sethares, stolzenburg, …) wouldn't reach connectivity | They land in connectivity for free once registered |

## What this doesn't break

- `compute_harm_connectivity` keeps the existing API, all 9 metric strings still work.
- `compute_harmonic_spectrum_connectivity` keeps the existing API + DataFrame columns.
- `get_harm_spectrum_metric_matrix(metric)` keeps working.
- IMF-based methods (`compute_IMF_correlation`, `EMD_time_resolved_harmonicity`) are untouched — they're a different abstraction layer (time-resolved instead of spectral) and the refactor isn't relevant to them.
- All existing tests continue to pass.

## Implementation effort estimate

| Phase | Scope | LOC | Risk |
|---|---|---|---|
| **A** | Add `compute_peak_phase_coupling_connectivity` and `compute_peak_resonance_connectivity`. Refactor the existing RRCi/wPLI/MI branches in `compute_harm_connectivity` to delegate. Add tests. | ~250 | Low — additive, existing branches preserved |
| **B** | Add `biotuner/resonance/cross.py` with `compute_cross_resonance` and `CrossResonanceResult`. Snapshot regression vs current `compute_cross_spectrum_harmonicity` output. Refactor the legacy function to call it. | ~450 | Medium — needs snapshots; bit-exact reproduction will reveal additional legacy quirks (the cross-spectrum wPLI uses an `imag(X·Y*) / |imag(X·Y*)|` formula that doesn't quite match any registered metric — likely closest to `nm_wpli`; will need verification) |
| **C** | Add `compute_cross_resonance_connectivity` method + multi-channel surrogate-null variants. Demo figures for the report. | ~200 | Low |

**Total:** ~900 LOC + tests + figures. Same shape as the original resonance refactor PR.

## Suggested PR breakdown

Either ship as one big PR (like the resonance refactor) or three sequential PRs:

1. **`feat(connectivity): peak-based phase coupling + resonance via resonance registry`** — Layer A. Smallest, lowest risk.
2. **`feat(resonance): cross-channel orchestrator (compute_cross_resonance)`** — Layer B in `biotuner/resonance/cross.py`. Snapshot regression locked in.
3. **`refactor(connectivity): cross-spectrum delegates to compute_cross_resonance`** — Layer C wires it through the existing `compute_harmonic_spectrum_connectivity` API.

---

## Questions before implementation

1. **Cross-spectrum wPLI formula** — the current `compute_cross_spectrum_harmonicity` builds the phase coupling matrix as `|⟨imag(X·Y*)⟩| / ⟨|imag(X·Y*)|⟩` where `X = Zxx1[i]` and `Y = Zxx2[j]`. This is the **complex-spectrum wPLI**, which differs subtly from the phase-only `nm_wpli` I registered (`|⟨sign(Im(exp(iΔφ)))⟩|`). Bit-exact reproduction requires the complex-spectrum formula. Want me to add it as a separate registered metric (e.g. `nm_wpli_complex` that takes STFT complex coefficients instead of phases), or keep the existing inline formula in the cross-spectrum path?

2. **`bandwidth` parameter for peak-based PC** — current `wPLI_crossfreq` hardcodes `bandwidth=1` Hz; `cross_frequency_rrci` exposes it as a param. Worth exposing on the new method too.

3. **Coupling-metric arity in cross context** — the new orchestrator currently enforces "pairwise" for `coupling_metric`. For cross-channel the same constraint applies, so the arity check transfers cleanly. Higher-order cross-channel metrics (e.g., bPLV across 3 electrodes) would land in Phase 3, on a separate path.

4. **Peak-based R combine rule** — the resonance combine rules expect length-N arrays. For peak-based scalars, the same rules apply trivially (treat each scalar as a length-1 array). Default `product` reproduces standard "H × PC" semantics.

Happy to proceed with any subset of the three phases on your go-ahead.
