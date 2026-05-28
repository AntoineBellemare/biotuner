# Resonance-package refactor — evaluation report

**Branch:** `claude/sweet-hamilton-6c0f0d`  ·  **PR:** [#14](https://github.com/AntoineBellemare/biotuner/pull/14)
**Test suite:** 1662 passed, 1 skipped (no regressions)
**Snapshots:** 3 reference signals × {H, PC, R} match pre-refactor numerics at `atol = 1e-6`

This report presents five paper-ready figures evaluating the refactor. All
figures are saved as both `.png` and `.pdf` in `figures/`. Source: `generate_report.py`.

---

## Figure 1 — Bit-exact regression

![Figure 1](figures/fig1_regression.png)

Each panel overlays the **legacy** `compute_global_harmonicity` output (thick
light grey) and the **refactored** `compute_resonance` output (thin colored)
across the three reference signals (columns) and three spectra (rows). The
`max |Δ|` annotation in each panel shows the maximum absolute difference
between the two curves — **0.0e+00 everywhere**, i.e. bit-exact reproduction.

**What this validates:** the legacy-default `ResonanceConfig` preserves every
quirk that affects numeric output (min-max PSD rescale → probability, the
legacy self-pair-subtract reducer, binary 5% n:m gate with 1:1 fallback, STFT
phase, Gaussian smoothing on H and PC, product combine). The refactor introduces
zero numerical drift.

---

## Figure 2 — Decomposition: how `R(f) = H(f) · PC(f)` is built

![Figure 2](figures/fig2_decomposition.png)

A single signal (the 1:2:4:8 harmonic test stimulus) is decomposed through the
full pipeline:

- **(a)** Input signal in the time domain (1-second window for legibility).
- **(b)** Normalized PSD after 1/f aperiodic removal — the probability weights `p_i` that feed the per-bin reducer.
- **(c)** Harmonicity factor `H(f)` — high at the fundamental and integer multiples.
- **(d)** Phase coupling factor `PC(f)` — high where bins phase-lock at low-order n:m ratios.
- **(e)** Resonance spectrum `R(f) = H · PC` — only frequencies satisfying both criteria survive.

Each panel reports the full complexity summary (`avg`, `max`, `flatness`,
`entropy`, `higuchi`, `spread`) that lives in `ResonanceResult.summaries`,
and marks the prominence-detected peaks from `ResonanceResult.peaks`. These
are exactly the per-spectrum metrics the legacy DataFrame contained —
preserved, now structured as a nested dict per factor.

---

## Figure 3 — Kernel registry: swappable harmonic kernels

![Figure 3](figures/fig3_kernel_comparison.png)

Same signal, same pipeline, **one configuration change**:
`ResonanceConfig(harmonic_kernel='harmsim')` vs `'subharm_tension'`. The kernels
produce different magnitudes (dyad similarity vs subharmonic tension) but
qualitatively similar peak structure — the orchestrator handles the swap
transparently via the `HARMONIC_KERNELS` registry.

**What this validates:** the strategy registry is real. Adding the Phase 2
kernels (sethares, stolzenburg, harmonic_entropy, hopf, lorentzian) means
writing one function and calling `register_harmonic_kernel(name, fn)` — every
downstream consumer (orchestrator, `compute_harmonic_spectrum`) picks them up
automatically.

---

## Figure 4 — Combine-rule registry

![Figure 4](figures/fig4_combine_rules.png)

- **(a)** The two factors H and PC (min-max normalized for comparable scales).
- **(b)** Four combine rules applied to the same factors: `product` (legacy default), `geomean`, `harmmean`, `min`.

The product rule sharpens the resonance peaks (because both H and PC must be
high simultaneously). Geomean, harmmean, and min are softer — they highlight
agreement without compressing dynamic range as aggressively. All ship in the
`COMBINE_RULES` registry; users select via `ResonanceConfig(combine=...)`.

**What this validates:** the combine axis is fully pluggable. Phase 2's
`weighted_log` (generalized geometric mean with per-factor weights) is already
registered and ready for the multi-factor compound consonance rule from
Harrison & Pearce 2020 (composite consonance: roughness + periodicity + familiarity).

---

## Figure 5 — Cross-signal characterization

![Figure 5](figures/fig5_cross_signal.png)

**Top row** — three complexity metrics across the three reference signals,
grouped by spectrum (H/PC/R bars):

- **Spectral flatness:** inharmonic ≪ harmonic < pink noise. Confirms the
  rank ordering: peaked spectra → low flatness; flat noise → high flatness.
- **Spectral entropy:** harmonic and pink ≈ similar (broad coverage),
  inharmonic distinctly lower (energy concentrated in narrow bins).
- **Higuchi FD:** R(f) has the highest fractal dimension across all three
  signals — the resonance spectrum is the most structurally rich of the three
  factors.

**Middle and bottom rows** — the actual H, PC, R spectra side by side.
Inharmonic clearly localizes peaks at 7 / 11.3 / 17.9 / 23 Hz (no integer
ratios → narrow peaks). Pink noise is broadband. Harmonic shows the 1:2:4
structure.

**What this validates:** the rich-metric output is preserved and remains a
discriminative fingerprint of signal regime. A downstream classifier built on
`ResonanceResult.summaries` would have the same input features as one built on
the legacy DataFrame.

---

## Cross-signal complexity table

Generated by the report and saved to `figures/fig5_summary_table.csv`. Reproduced here:

| Signal | Spectrum | Flatness | Entropy | Higuchi | Spread |
|---|---|---|---|---|---|
| Harmonic (5/10/20/40 Hz) | H  | 0.800 | 5.422 | 0.891 | 8.269 |
| Harmonic (5/10/20/40 Hz) | PC | 0.850 | 5.591 | 0.809 | 8.314 |
| Harmonic (5/10/20/40 Hz) | R  | 0.416 | 4.216 | 0.937 | 7.360 |
| Pink noise (1/f)         | H  | 0.946 | 5.739 | 0.911 | 8.679 |
| Pink noise (1/f)         | PC | 0.993 | 5.823 | 0.823 | 8.338 |
| Pink noise (1/f)         | R  | 0.926 | 5.691 | 0.960 | 8.881 |
| Inharmonic (7/11.3/17.9/23 Hz) | H  | 0.091 | 4.319 | 0.762 | 6.293 |
| Inharmonic (7/11.3/17.9/23 Hz) | PC | 0.098 | 4.328 | 0.745 | 6.253 |
| Inharmonic (7/11.3/17.9/23 Hz) | R  | 0.003 | 3.731 | 0.868 | 6.320 |

Observe the resonance row for inharmonic — flatness drops to 0.003 (essentially
zero), reflecting that R(f) has cleanly isolated the 4 inharmonic carriers
with virtually no off-peak energy.

---

## What's not yet in the figures

The figures above exercise everything Phase 1 ships. They do **not** yet
demonstrate:

- **Surrogate normalization** (`with_surrogate_null`) — wired but expensive to
  run inline; a Phase-2 figure with IAAFT-z-scored R(f) for a real EEG epoch
  would be the natural follow-up.
- **Phase 2 kernels** (Arnold-tongue, sethares, stolzenburg) — not yet implemented.
- **Phase 3 higher-order coupling** (bPLV, M-PLV, GPLA) — not yet implemented.
- **Persistence / Q-factor** (lagged coherence, LHaC) — not yet implemented.

Each lands as an additive registration against the existing registry — no
orchestrator changes required.

---

## How to regenerate

```bash
python reports/resonance_refactor/generate_report.py
```

Output: `reports/resonance_refactor/figures/{fig1..5}.{png,pdf}` and
`figures/fig5_summary_table.csv`.
