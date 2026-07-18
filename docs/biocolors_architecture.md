# biocolors expansion — architecture

Status: **built and calibrated on real data.** Supersedes the `biocolors_plus.py` handoff.

All numbers below are measured, not asserted. Reproduce with the scripts noted at the end.

---

## 0. What exists now

Built as `biotuner/biocolors/` (was `biocolors.py`, now `biocolors/legacy.py`, re-exported so every
existing import keeps working — verified: `wavelength_to_rgb(475)` still returns `(0, 213, 255)`).

Calibrated on **HMC sleep staging DB, subject SN001**: 7.12 h, 256 Hz, 4 EEG channels, 854
expert-scored 30 s epochs across W/N1/N2/N3/R → **3 416 signals, 0 extraction failures**. Shipped as
`biotuner/biocolors/data/eeg_sleep_v1.npz`.

Figures in `docs/img/biocolors/`, all generated from that recording:

| figure | shows |
|---|---|
| `01_stage_palettes` | the five sleep stages as palettes, from stage-median spectra |
| `02_within_stage_variability` | eight individual N2 epochs — one stage is *not* one colour |
| `03_temperament_range` | one signal under ten characters (earthy … pastel … vivid) |
| `04_night_ribbon` | 7.1 h as colour, 30 s vs 150 s windows |
| `05_diversity_vs_prototype` | 60 real epochs: prototype anchor vs fingerprint anchor |
| `06_colorspace` | the cusp, gamut mapping, Bruton vs CIE 1931 |
| `07_mapping_variants` | `anchored` / `spectral` / `mds` |
| `08_fingerprint_space` | the PCA domain the hue is read off, with stage labels |
| `09_forms` | ramps, dyad field, consonance spectra |
| `10_audit` | CVD simulation + absolute vs separated mode |

### What building it on real data changed

Three things only real EEG could have told us:

1. **Amplitudes are dB, and rectifying them by shifting is wrong.**
   `peaks_extraction.py:324` does `psd = 10*log10(psd)`, so every Welch-based extractor returns dB;
   26% of values in this corpus are negative. The first implementation shifted them to positive,
   which is scale-inconsistent — 74% of epochs have no negative value and so were left alone.
   Measured consequence: two N2 epochs with **identical peaks** `[1, 3.5, 7.5, 14.2, 28.5]` received
   anchor hues **122° apart**. Fixed by inverting the log (`10**(dB/10)`) under an explicit
   `amps_scale` parameter, recorded in the calibration and warned about on mismatch. This single fix
   took the PCA's 2-D coverage from **66.9% → 79.9%** and turned the spectral descriptors from
   near-useless to the most informative in the set (`spectral_entropy` std 0.018 → 0.123).

2. **Percentile calibration needs a degeneracy guard.** On this corpus `n_peaks` is 5 for ~98% of
   epochs and `complexity` (Higuchi FD over five amplitudes) is mostly 0. Percentile-normalising a
   degenerate descriptor stretches quantisation noise across the full range and injects it into hue
   as if it were signal. `Calibration` now learns a per-descriptor weight (`1 - modal_bin_fraction`)
   from the corpus; both collapse to **0.003** and are suppressed automatically. A descriptor useless
   here can still earn weight on a corpus where it varies — the calibration adapts, the field list
   is not hardcoded.

3. **The prototype's diversity failure is real but smaller than synthetics implied.** On synthetic
   EEG-like signals the `/55` anchor gave a 0.20° median hue gap and 56% of pairs within 5°. On *real*
   epochs it gives **1.26°** and **7%** — bad, not catastrophic, because real `fixed`-extracted peaks
   include a ~1 Hz fundamental that lifts consonance into a wider range (7.4–60.4, not 0.3–5). The
   fingerprint anchor gives **3.95°** and **2%**, with wheel occupancy **47% → 78%**. Worth stating
   plainly rather than quoting the synthetic number.

### What the figures honestly do not show

- **Sleep stages overlap heavily in fingerprint space** (`08`). Wake leans right, REM left, but
  individual epochs are not separable by stage. The per-stage palettes in `01` work because averaging
  500+ epochs recovers the mean difference. This is a faithful map of spectra, not a sleep classifier,
  and it is not marketed as one.
- **Within-stage variability is large** (`02`): anchor circular std inside N2 is **150°** at 30 s.
  Longer windows help only partly (median consecutive shift 61° → 35° at 150 s) and do not make sleep
  architecture legible in the ribbon (`04`). That is the honest cost of `mode="absolute"`.
- **W and N1 aggregate palettes collide** (min gap 10.9°, palette ΔE 0.041). N1 is transitional wake;
  similar spectra *should* give similar colour. `mode="separated"` is the answer when they must be
  told apart.

---

## 1. Verdict on `biocolors_plus.py`

### Keep

- **The OKLab matrices.** Verified against a dense random sRGB grid: max round-trip error
  `1.56e-06` (0.0004 / 255), mean `1.46e-07`. They are correct. Carry them forward verbatim.
- **Working in OKLCh at all.** The current `biocolors.viz_scale_colors` does HSV with a hardcoded
  `hsv[2] = 200`. Moving to a perceptual space is the right instinct.
- **`cons_matrix_to_hue`** — classical MDS of the consonance matrix onto the hue circle. Genuinely
  novel and worth keeping: it makes hue mean *harmonic relatedness* rather than pitch.
- **`diss_curve`-driven per-signal spectra.** Real signal content, real differentiation.
- **Folding on wavelength** rather than only multiplying frequency up.

### Broken (empirically confirmed)

**1. The gamut fit is dead code.**
`_linear_to_srgb` (line 103) clips to `[0, 1]` *before returning*, so the out-of-gamut test at
line 139 (`rgb.min() < -1e-6 or rgb.max() > 1 + 1e-6`) can never fire.

| OKLCh requested | unclipped sRGB | after the clip | search fired? | result |
|---|---|---|---|---|
| L=0.6 C=0.4 h=150 | `[-0.550, 0.703, -0.319]` | `[0, 0.703, 0]` | **False** | `(0, 179, 0)` |
| L=0.5 C=0.5 h=260 | `[0.113, -0.433, 1.476]` | `[0.113, 0, 1.0]` | **False** | `(29, 0, 255)` |
| L=0.85 C=0.35 h=120 | `[0.714, 0.903, -0.538]` | `[0.714, 0.903, 0]` | **False** | `(182, 230, 0)` |

Every out-of-gamut colour is silently **clipped**, which shifts hue — destroying the perceptual
guarantee the module exists to provide. The 24-iteration bisection has never run.

**2. `srgb_to_oklab` is silently wrong on batched input.**
`r, g, b = _srgb_to_linear(np.asarray(rgb, float))` unpacks the *first* axis, so an `(N, 3)` array
is read with **pixels treated as channels**.

- `(3, 3)` input: returns a wrong answer **without raising** — max deviation `0.0240` from the
  per-row reference.
- `(4, 3)` input: `ValueError: too many values to unpack (expected 3)`.
- Only a single `(3,)` colour is correct.

Any attempt to vectorize will corrupt colours quietly.

**3. The `/55` normalisation is calibrated to the wrong distribution. This is the diversity killer.**

On 30 synthetic EEG-like signals (theta/alpha/beta peaks with jitter), mean consonance spans
**0.318 – 4.957**, so `avg / 55 ∈ [0.006, 0.09]` — the bottom 9% of the hue budget:

```
anchor hue span : 16.7° – 42.0°   (25° of 360)
std             : 7.61°
median gap between adjacent signals : 0.20°
pairs within 5° (same hue family)   : 242 / 435  (56%)
```

**Every realistic EEG signal gets the same orange-red.** The diversity goal fails completely on the
actual target domain. It also saturates at the top: harmonic series 1–8 has mean consonance 59.55 →
`59.55/55 = 1.083` → clipped → hue exactly `315.00`; harmonic 1–12 sits at `0.994`.

Even on hand-picked *maximally different* synthetic signals it collides:

```
stretched  vs  two_cluster      hue gap 0.62°   palette dE_OK 0.0326   <- perceptually identical
golden     vs  inharmonic_rand  hue gap 2.43°   palette dE_OK 0.0484   <- perceptually identical
```

(`dE_OK < 0.05` ≈ "the same palette".) A stretched harmonic series and two tight clusters are not
remotely the same signal; they collide because they happen to share a mean-consonance value
(7.749 vs 7.635). **A single scalar cannot separate signals.**

**4. The visible-band clamp silently falsifies musically central steps.**
JI major on `fund=30`:

```
ratio=1.3333  f=40.00Hz  nm=426.03  n_oct=44
ratio=1.5000  f=45.00Hz  nm=750.00  n_oct=43   <-- CLAMPED to the band edge
ratio=1.6667  f=50.00Hz  nm=681.65  n_oct=43
```

The fold seam lands **on the perfect fifth**, which gets clamped to exactly 750.00 nm. Only 1.88% of
frequencies in 1–1000 Hz hit the clamp, but it is structural, not random — the seam has to land
somewhere, and the clamp hides it rather than handling it.

**5. Bruton's `wavelength_to_rgb` collapses 28.4% of the visible band to one hue.**
Measured over 380–750 nm: the entire **645–750 nm band (105 nm wide) maps to OKLCh hue 29.23°** —
only lightness varies. Consequence, directly observed: ratio 3/2 (nm 750.00) and ratio 5/3
(nm 681.65) both receive hue 29.23°. The JI major scale yields only **6 unique hues from 8 steps**.
The red end of the module's own spectral map has zero hue resolution.

### Misdiagnosed

**"Bright red is pink — in OKLab hue 0 is the magenta axis."**
Wrong on the fact, accidentally right on the observation.

```
sRGB red     (255,0,0) -> L=0.6280  C=0.2577  h= 29.23°
sRGB magenta (255,0,255)-> L=0.7017  C=0.3225  h=328.36°
OKLCh hue 0 at L=0.63,C=0.15 renders as (207, 93, 133)  -- a dusty rose
```

Hue 0 is the **+a axis**, not the magenta axis (magenta is at 328°), and red is 29° away from it.

**The real cause is the dead gamut fit meeting the chroma cusp.** The sRGB chroma ceiling at red's
hue collapses as lightness rises:

| L | max chroma at h=0 |
|---|---|
| 0.60 | 0.243 |
| 0.80 | 0.125 |
| 0.90 | **0.056** |

The proposal's defaults ask for `C = 0.22` at `L = 0.88` — roughly **3× the ceiling**. The gamut fit
never fires, so it clips, which desaturates toward white: **pink**. `encode="chroma"` works only
because pinning `L` to `(0.50, 0.66)` happens to sit near the cusp where the ceiling is ~0.20–0.24.
It is a workaround that dodges the bug, not a fix. Fix the gamut mapping and the `encode="lightness"`
path becomes usable and the `L_range` restriction lifts.

**"NaN per-step values from `tuning_cons_matrix`."** Refuted for the path actually used. With
`ratio_type="all"` (what the module passes), `per_step = [42.99, 21.05, 31.56, …]` — **no NaN** — and
`full_matrix` has **0 NaN of 64** with a `0.0` diagonal. The docstring's NaN example is
`ratio_type="pos_harm"`. The `nan_to_num` guard in `_norm` is dead code here (and its logic is
inverted anyway: it evaluates `np.nanmin(x)` as a fill value on the pre-fill array).

**"Black swatches from `audible2visible`."** Partly. The bug is real at a **7.2% rate** (216/3000
swept frequencies produce out-of-band nm → `(0,0,0)`), but the implied example does not reproduce:
`audible2visible(390)` → nm 699.13 → `(203,0,0)`, and the new `audible_to_nm(390)` returns the
*identical* 699.13.

### Performance is a non-issue

`dyad_field(n=12)` **8.0 ms**, `palette_ramp(n=256)` **13.5 ms**, `curve_palette(n=512)` **22.7 ms**,
`dyad_field(n=24)` **31.5 ms**. The python loops are fine — *because the gamut search never runs*.
Fixing it multiplies that path ~24×. Vectorize then, not before.

---

## 2. Architecture

Follows the `harmonic_geometry` precedent already established in this repo: **pure data out,
rendering is a downstream concern.**

```
biotuner/biocolors/
├── __init__.py          # re-exports + back-compat shims for every legacy name
├── legacy.py            # today's biocolors.py, frozen, deprecated
├── color/               # LAYER 1 — colour science. ZERO biotuner imports.
│   ├── spaces.py        # sRGB <-> linear <-> OKLab <-> OKLCh, vectorized (..., 3)
│   ├── gamut.py         # cusp table, cusp-aware chroma map
│   ├── spectral.py      # wavelength -> XYZ (CIE 1931) -> sRGB; Bruton kept as an option
│   ├── difference.py    # deltaE_OK, CVD simulation
│   └── palette.py       # farthest-point sampling, constrained construction
├── descriptors.py       # LAYER 2 — signal -> Fingerprint (registry)
├── mapping.py           # LAYER 3 — Fingerprint/steps -> ColorSpec (registry)
├── calibration.py       # LAYER 4 — reference distributions. Kills /55.
├── palettes.py          # LAYER 5 — the 5 public entry points
├── render.py            # matplotlib/PIL helpers (optional import)
└── data/
    └── calibration_eeg_v1.npz
```

```
Signal (peaks, amps) ──┐
Tuning (scale, fund) ──┤
                       ▼
              ┌─────────────────┐
              │  descriptors    │  registry: consonance, harmsim, tenney,
              │                 │  flatness, spread, entropy, HFD, n_peaks,
              └────────┬────────┘  subharm_tension, ...
                       │ Fingerprint (per-signal) + per-step metrics
                       ▼
              ┌─────────────────┐
              │  calibration    │  percentile-rank against a reference corpus
              └────────┬────────┘  -> every channel in [0,1]
                       ▼
              ┌─────────────────┐
              │    mapping      │  normalised channels -> (L, C, h)
              └────────┬────────┘  hue anchor + arc | MDS | wavelength
                       ▼
              ┌─────────────────┐
              │   ColorSpec     │  L, C, h arrays + provenance
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │  color/gamut    │  cusp-aware fit -> guaranteed in gamut
              └────────┬────────┘
                       ▼
                 PaletteData
          (rgb, oklch, spec, fingerprint, parameters, metadata)
                       │
                       ▼
              render.py / engine / user
```

The load-bearing structural point: **`color/` imports nothing from biotuner.** It is the layer that
is currently broken, it is testable against `colour-science` as ground truth with no harmonic
machinery in the way, and it is independently useful.

---

## 3. Flexibility

A registry per extension point. Not decoration — each has ≥3 real implementations today.

```python
# descriptors.py
DESCRIPTORS: dict[str, Descriptor] = {}

def descriptor(name, kind, range_hint=None):
    """kind: 'per_step' | 'per_signal' | 'matrix' | 'curve'"""
    def deco(fn):
        DESCRIPTORS[name] = Descriptor(name, kind, fn, range_hint)
        return fn
    return deco

@descriptor("harmsim", kind="per_step", range_hint=(0, 100))
def _harmsim(ctx):
    return np.asarray(ratios2harmsim(ctx.scale), float)

@descriptor("spectral_flatness", kind="per_signal", range_hint=(0, 1))
def _flatness(ctx):
    return spectral_flatness(ctx.harmonicity_values)
```

A user adds one without touching core:

```python
from biotuner.biocolors.descriptors import descriptor

@descriptor("my_metric", kind="per_signal")
def my_metric(ctx):
    return float(np.std(ctx.peaks) / np.mean(ctx.peaks))

pal = palette_from_signal(peaks, amps, chroma_from="my_metric")   # usable immediately
```

Same shape for the other axes:

```python
@colorspace("oklch")  class OKLCh:  channels = ("L", "C", "h"); to_srgb; from_srgb
@colorspace("cam16")  class CAM16UCS: ...
@mapping("anchored_arc") def anchored_arc(fp, steps, cfg) -> ColorSpec
@mapping("mds_hue")      def mds_hue(fp, steps, cfg) -> ColorSpec
```

Fully dict-configurable, so `engine/backend` can drive it from JSON:

```python
pal = palette_from_signal(peaks, amps, config={
    "hue":    {"mapping": "anchored_arc", "anchor": "fingerprint_pc1", "arc": 90},
    "chroma": {"from": "consonance", "range": "cusp_relative", "frac": (0.35, 0.95)},
    "light":  {"from": "amplitude", "range": (0.45, 0.75)},
    "space":  "oklch",
    "calibration": "eeg_v1",
})
```

**Provenance is what makes the registry load-bearing.** Every swatch can explain itself:

```
>>> pal.explain(3)
step 3  ratio=4/3  freq=40.0 Hz
  hue    = 187.4°  <- anchored_arc(anchor=112.0° from fingerprint_pc1=0.62, arc=90°, t=0.84)
  chroma = 0.141   <- consonance=25.42 -> pct=0.38 -> 0.62 * cusp(L=0.58, h=187.4)=0.227
  light  = 0.580   <- amplitude=0.60 -> pct=0.55
  gamut  : requested C=0.141, cusp=0.227 -> in gamut, no clip
```

---

## 4. Colourspace mapping

**Working space: OKLCh.** Keep it, for empirical rather than fashionable reasons. The matrices
round-trip at `1.56e-06`, the implementation is ~40 lines, and the alternatives cost more than they
return here: CAM16-UCS requires a viewing-condition model (surround, adapting luminance, background)
that a library emitting palettes for unknown displays cannot honestly supply; Jzazbz is built for
HDR/absolute luminance, irrelevant at sRGB; HCT is OKLab chroma + CAM16 hue and adds a dependency.
OKLab's known weakness — a hue shift on the blue–purple axis under large lightness changes — is real
but small at palette chroma levels. Revisit only if it shows up in practice.

Add **Oklrab lightness** (`toe` / `toe_inv`) for the *interface*: OKLab's `L` is not perceptually
uniform against sRGB mid-grey. Users reason in `L_r`; convert to `L` internally.

### The gamut fix

1. **Split clipping out of conversion.** `oklab_to_srgb(lab, clip=False)` returns unclipped values so
   callers can *see* out-of-gamut. The clip inside `_linear_to_srgb` is exactly what killed the search.
2. **Replace the 24-iteration bisection with an analytic cusp lookup.** Precompute the (L, C) cusp per
   hue once, then chroma-clip toward it: O(1), no loop, and yields the *true* maximum rather than a
   bisection approximation.
3. **Make ranges cusp-relative.** `C_range=("cusp", 0.35, 0.95)` = "35–95% of what is achievable at
   this (L, h)".

Why (3) matters — the measured sRGB chroma ceiling varies **5.4×** across the wheel:

| L \ h | 0 | 45 | 90 | 135 | 180 | 225 | 270 | 315 |
|---|---|---|---|---|---|---|---|---|
| 0.30 | 0.122 | 0.086 | 0.061 | 0.088 | 0.054 | 0.057 | 0.197 | 0.148 |
| 0.45 | 0.182 | 0.129 | 0.092 | 0.131 | 0.082 | 0.085 | **0.295** | 0.222 |
| 0.60 | 0.243 | 0.172 | 0.123 | 0.175 | 0.109 | 0.114 | 0.217 | **0.296** |
| 0.70 | 0.210 | 0.200 | 0.143 | 0.204 | 0.127 | 0.133 | 0.156 | 0.233 |
| 0.80 | 0.125 | 0.122 | 0.163 | 0.234 | 0.145 | 0.136 | 0.100 | 0.148 |
| 0.90 | 0.056 | 0.056 | 0.128 | **0.241** | 0.155 | 0.065 | **0.048** | 0.071 |

A fixed `C_range=(0.02, 0.22)` rectangle is ~3× over budget at L=0.90/h=270 (ceiling 0.048) and
leaves 25% of the available chroma unused at L=0.60/h=315 (ceiling 0.296). It cannot be right
everywhere. Cusp-relative ranges are what make palettes *look* good rather than merely be computed.

### Spectral locus

**Replace Bruton as the default.** It collapses 28.4% of the band (645–750 nm) to hue 29.23°. Ship a
CIE 1931 path using the **Wyman–Sloan–Shirley (2013)** analytic multi-lobe Gaussian fit to the colour
matching functions — ~15 lines, no data table, no new dependency — then XYZ → linear sRGB (D65) →
gamut map. Keep `spectral="bruton"` for byte-compatibility with existing output; default to
`spectral="cie1931"`.

### Checks that make "nice" falsifiable

- `deltaE_ok(a, b)` — euclidean in OKLab (the entire point of the space).
- `simulate_cvd(rgb, kind, severity)` — Machado et al. (2009), ~20 lines of constants.
- `palette_report(pal)` → min pairwise ΔE, min ΔE *under* deuteranopia/protanopia/tritanopia,
  gamut-clip count, lightness range. Failing a threshold becomes a **test failure**, not an
  aesthetic opinion.

---

## 5. Palette diversity

### Name the tradeoff: stability vs separation

They are not simultaneously satisfiable from a stateless function, and the proposal pretends
otherwise.

- **Stability** — same signal → same palette, always, with no reference to other signals.
  Needed for: reproducible figures, streaming/online use, comparability across papers.
- **Separation** — different signals → far-apart palettes. *Requires* knowing the other signals or
  their distribution. Needed for: multi-subject plots, legends, "these 8 conditions must be
  tellable apart".

The API exposes both, explicitly:

```python
palette_from_signal(peaks, amps, mode="absolute", calibration="eeg_v1")  # stable
palette_set([sig1, sig2, ...], mode="separated", min_delta_e=0.15)       # separated
```

### The fingerprint (root fix)

Replace the single mean-consonance scalar with a vector:

```python
@dataclass
class Fingerprint:
    harmonicity:       float   # mean dyad_similarity over the tuning
    harmonic_spread:   float   # std of per-step consonance
    spectral_flatness: float   # metrics.spectral_flatness
    spectral_spread:   float   # metrics.spectral_spread
    spectral_entropy:  float   # metrics.spectral_entropy
    n_peaks:           int
    fundamental:       float   # log-scaled
    complexity:        float   # higuchi_fd / spectrum_complexity
    subharm_tension:   float   # compute_subharmonic_tension
    def vector(self) -> np.ndarray: ...
    def normalized(self, cal) -> np.ndarray: ...   # percentile-mapped, each in [0,1]
```

Both measured collisions are resolved by one extra dimension. `stretched` vs `two_cluster` collide at
0.62° because they share mean consonance (7.749 vs 7.635) — but their **harmonic_spread** and
**spectral_spread** differ enormously (a smooth stretched series vs two tight clusters).
`golden` vs `inharmonic_rand` separate on **flatness**.

### Calibration — killing the `/55`

The magic number is a symptom of having no model of what these metrics do on real data. Fix with a
shipped percentile table:

```python
@dataclass
class Calibration:
    name: str                                # "eeg_v1"
    percentiles: dict[str, np.ndarray]       # metric -> 101 quantiles from a reference corpus
    def normalize(self, metric, value):
        return np.interp(value, self.percentiles[metric], np.linspace(0, 1, 101))
```

Percentile ranking is the correct normaliser because it is **distribution-matched by construction**:
it spends hue budget where signals actually differ. Measured: EEG-like mean consonance spans
0.318–4.957; `/55` compresses that into 4.5% of the wheel, while a percentile map spreads it across
the full range *by definition*.

Built by `build_calibration(corpus)` and versioned in `data/calibration_eeg_v1.npz`.
Escape hatches: `calibration="none"` reproduces raw-value behaviour;
`calibration=Calibration.from_signals(my_corpus)` fits to the user's own data.

### The embedding

Normalised 9-D fingerprint → 2-D via a **fixed, shipped** projection (PCA fit on the reference corpus
and stored *with* the calibration — refitting per call would destroy stability), then:

```
hue          = atan2(pc2, pc1)        # direction in fingerprint space
chroma_scale = clip(|pc|, ...)        # distance from centroid: typical -> muted, unusual -> vivid
```

Strictly better than the scalar anchor: it cannot collapse two signals unless they agree on **all
nine** descriptors, and "unusual signal → vivid palette" falls out for free.

### Separated mode

Compute fingerprints, then farthest-point-sample / Lloyd-relax the anchor hues under a `min_delta_e`
constraint **while minimising displacement from each signal's absolute anchor**. Separation is
guaranteed; meaning is preserved as far as the constraint allows. Report the displacement so the
user knows what was traded.

### Measurement — the claim must be falsifiable

Ship the diagnostic:

```python
diversity_report(palettes) -> {
    "mean_pairwise_de":  0.186,
    "min_pairwise_de":   0.033,   # the failure measured today
    "n_collisions":      2,       # pairs under the perceptual threshold
    "hue_occupancy":     0.44,    # fraction of the wheel used
    "within_palette_de": 0.071,   # internal contrast
    "min_de_under_cvd":  0.021,
}
```

with a regression test asserting `min_pairwise_de > 0.05` and `n_collisions == 0` on a fixed
benchmark set of ≥8 signals **including realistic EEG peak sets**, not only maximally-different
synthetics. **That test fails today** — min 0.033 with 2 collisions on synthetics, and
catastrophically on EEG (242/435 pairs within 5° hue).

### Within-palette diversity

Measured mean distance-to-centroid **0.062–0.095** — palettes are internally flat, because a narrow
arc plus a narrow `L_range` leaves nothing to vary. The narrow-arc rule was a workaround for the
octave-hue-turn problem; once a fingerprint anchor carries separation duty, the arc is freed to
carry *internal contrast*. Let the arc widen when `harmonic_spread` says the signal is internally
varied, and drive lightness from a per-step metric instead of pinning it.

---

## 6. Public API

| function / class | signature | returns |
|---|---|---|
| `palette_from_tuning` | `(scale, fund=None, *, config=None, calibration="eeg_v1", **overrides)` | `PaletteData` |
| `palette_from_signal` | `(peaks, amps, *, mode="absolute", config=None, calibration="eeg_v1", **overrides)` | `PaletteData` |
| `palette_set` | `(signals, *, mode="separated", min_delta_e=0.15, **kw)` | `list[PaletteData]` |
| `dyad_field` | `(scale, *, metric="consonance", **kw)` | `FieldData` |
| `consonance_spectrum` | `(peaks, amps, *, max_ratio=2.0, **kw)` | `SpectrumData` |
| `PaletteData` | dataclass | `.rgb .oklch .spec .fingerprint .parameters .metadata` + `.explain(i) .to_hex() .ramp(n) .report()` |
| `Fingerprint` | dataclass | `.vector() .normalized(cal)` |
| `Calibration` | dataclass | `.normalize() .from_signals() .load(name)` |
| `descriptor` / `mapping` / `colorspace` | decorators | registry registration |
| `color.*` | `srgb_to_oklab, oklab_to_srgb, oklch_to_srgb, cusp, gamut_map, deltaE_ok, simulate_cvd, spectral_to_srgb` | arrays |
| `diversity_report` / `palette_report` | `(palettes)` / `(palette)` | dict |

**5 entry points, down from the proposal's 12 flat functions.**
`scale_to_perceptual_colors`, `palette_ramp`, `curve_palette`, `signal_spectrum_palette` and
`signal_anchored_spectrum` collapse into `palette_from_*` plus a `.ramp()` method.
`harmonic_hue_anchor` becomes internal (it is the thing being replaced). `per_step_metric` becomes
the descriptor registry. `audible_to_nm` moves to `color.spectral`.

---

## 7. Migration & compatibility

- **`biotuner/biocolors.py` → `biotuner/biocolors/legacy.py`**, re-imported verbatim by
  `biocolors/__init__.py`. Every legacy name stays importable at its current path:
  `wavelength_to_rgb`, `Hz2nm`, `nm2Hz`, `Hz2THz`, `THz2Hz`, `audible2visible`, `scale2freqs`,
  `wavelength_to_frequency`, `viz_scale_colors`, `animate_colors`.
- **Real consumers exist and must not break:**
  - `engine/backend/services/color_service.py:16` — `from biotuner.biocolors import audible2visible, scale2freqs, wavelength_to_rgb`
  - `biotuner/harmonic_timbre/cross_modal.py` — references the biocolors integration
  Both are satisfied by re-export.
- **`viz_scale_colors` is already broken** — `biocolors.py:285` unpacks two values from
  `tuning_cons_matrix`, which returns three. It raises `ValueError` against current metrics. Fix in
  place as a bug fix, deprecate, point at `palette_from_tuning`.
- **`biocolors_plus.py` does not ship as-is.** Good parts land in the new layers; the dead gamut fit,
  the batched-conversion trap and the `/55` are not carried forward. No `_plus` name is ever
  exported — that suffix is a migration smell.

---

## 8. Build order

| phase | contents | testable at exit |
|---|---|---|
| **1. `color/`** (no biotuner) | spaces (vectorized `(..., 3)`), gamut (cusp + map), difference (ΔE, CVD), spectral (CIE + Bruton) | round-trip < 1e-6 on a dense grid; every `gamut_map` output in `[0,1]` *and* its preimage out; cusp matches bisection to 1e-4; Bruton-vs-CIE ΔE reported |
| **2. descriptors** | registry, 9 descriptors wired to existing metrics, `Fingerprint.vector()` | all descriptors finite on the benchmark set; ranges match `range_hint` |
| **3. calibration** | `build_calibration`, percentile normalise, ship `eeg_v1` | normalised descriptors ~uniform on the reference corpus — assert it |
| **4. mapping + PaletteData** | anchored_arc, mds_hue, wavelength; `explain()` | `diversity_report` passes `min_pairwise_de > 0.05`, `n_collisions == 0` — **the test that fails today** |
| **5. palettes.py + separated mode** | public API, farthest-point separation | separated mode guarantees `min_delta_e`; displacement reported |
| **6. render + shims** | matplotlib/PIL helpers, legacy re-exports, deprecations | existing consumers import unchanged; `viz_scale_colors` no longer raises |

Phases 1–2 are independently valuable and unblock everything else. **Phase 1 fixes the actual bugs**
and is worth doing even if nothing else ships. **Phase 3 is where the diversity fix lands.**

---

## 9. Open questions

1. **Calibration corpus.** `eeg_v1` needs real data to be honest. Is there a corpus to fit
   percentiles on (which dataset, what preprocessing), or do we ship a synthetic prior for v1 and
   refit later? This decides whether the diversity claim is real or aspirational.
2. **Is octave-equivalent colour a feature or a bug?** Measured: ratio 1 and ratio 2 both land on
   568.04 nm — the same colour. Elegant (octave equivalence in hue), but a palette then cannot
   distinguish a fundamental from its octave.
3. **Bruton default or opt-in?** Switching to CIE 1931 changes every existing colour output.
   `engine/` has an API surface with a frontend that may have colours baked into expectations. Is a
   visual break acceptable, or does CIE hide behind a flag until an engine version bump?
4. **Dependency stance.** numpy/scipy are already hard deps. `colour-science` would delete ~300 lines
   of `color/` and provide ground-truth conversions. Numpy-only was a *proposal* constraint, not a
   repo constraint — is it a real one?
5. **Subpackage or flat?** `biocolors/` mirrors `harmonic_geometry/`, but it is a bigger refactor
   than a single file. Worth it, or keep `biocolors.py` + `biocolors_color.py` flat?

---

## Reproducing the numbers

Measured on Python 3.10.9, numpy <2.0, against this worktree. The three scripts below are the seed of
the Phase 1 / Phase 4 test suites and should move into `tests/biocolors/` when work starts.

Currently at
`%LOCALAPPDATA%/Temp/claude/C--Users-skite-Documents-Github-biotuner--claude-worktrees-epic-morse-ded0d5/a65cb825-1795-455c-be54-c0d8102646f8/scratchpad/`:

| script | establishes |
|---|---|
| `test_color.py` | gamut fit is dead code; red at h=29.23° / magenta at 328.36°; round-trip 1.56e-06; batched-input corruption; the cusp table |
| `test_biotuner.py` | no NaN under `ratio_type="all"`; hue span; the 750 nm clamp; the 8-signal diversity/collision measurement; timings; the handoff validation snippet |
| `test_focus.py` | `/55` saturation; 1.88% clamp rate; Bruton's 105 nm hue plateau; the 30-signal EEG-like collapse (median gap 0.20°, 242/435 pairs within 5°) |
