# bioelements expansion — architecture plan

Status: proposal. Grounded in a working proof-of-concept; numbers below are measured
against `data/air_elements.csv` (NIST emission lines, 21 896 lines, 99 elements)
on pandas 2.3.3.

The goal: take the module from *"which atoms does a biosignal resonate with?"* to
*"which **materials** does it resonate with, and can I build material structures from
biosignals?"* — where a material is a composition of elements, and a structure is a
composition of materials.

---

## 1. The intuition, and the expansion

**What bioelements does today.** Every atom emits light at a characteristic set of
wavelengths — its emission spectrum. `bioelements` treats those lines as a chord: it
octave-folds a biosignal's peaks into the optical band and asks which elements' lines
they land on. The same octave-transpose that `biocolors` uses to send a frequency to a
*colour*, `bioelements` uses to send it to the *periodic table*. It already renders each
element as a 7-step tuning (`data/Elemental_tunings/*.scl`), built from that element's
most intense lines.

So the atom is already a first-class harmonic object: **element = a spectrum = a chord =
a tuning**.

**The expansion.** Matter is not made of bare atoms; it is made of *compositions* of
them. Water is H₂O, wood is mostly cellulose (C₆H₁₀O₅)ₙ, air is a mixture of N₂/O₂/Ar/CO₂,
tissue is a biomolecular blend, a cloud is water suspended in air, fire is a combustion
process. If an element has a spectrum, then a **composition of elements has a composite
spectrum** — the weighted superposition of its constituents — and therefore its own chord,
its own tuning, its own colour, its own geometry. A biosignal can then resonate not just
with oxygen but with *water*, not just with carbon but with *wood*.

The compositional hierarchy is the whole plan:

```
  Element     H, O, C, N, Fe …            leaf spectra (NIST lines)
     │  stoichiometry
  Compound    H₂O, NaCl, CO₂, cellulose   fixed atom ratios
     │  proportion
  Mixture     air, wood, tissue, cloud    blends of compounds/elements
     │  arrangement
  Structure   layered/spatial materials   compositions of materials (→ geometry)
```

Each level is a **weighted composition of the level below**, and one recursive operation —
"superpose the component spectra, weighted" — carries all four. That operation is the core
of the expansion.

---

## 2. Verdict on the current module

`biotuner/bioelements.py` is 145 lines of loose functions. The idea is sound; the
implementation needs consolidation before it can carry composites.

### Broken (confirmed)

- **`compute_ratios_df` does not run on pandas ≥ 2.0.** It calls `DataFrame.append`, removed
  in pandas 2.0; measured failure: `AttributeError: 'DataFrame' object has no attribute
  'append'`. Any element-internal ratio analysis is currently dead. (Rewrite with
  `pd.concat` / vectorised ratio computation.)
- **The matcher's tolerance model is unusable.** `find_matching_spectral_lines` takes an
  *absolute* wavelength tolerance and defaults to `1e-9`. Across a line table spanning
  56–46 525 Å, an absolute tolerance is meaningless — a 1 Å window is 0.002% at the red end
  and 1.8% at the blue end. Measured: a plausible EEG-like peak set returns **0 matches** at
  tolerance 1.0. Matching must be **relative** (cents, or a fractional/ppm window), the same
  discipline `biocolors` uses for ratios.

### Messy (blocks the expansion)

- **Data lives outside the package.** The line tables are in repo-root `data/air_elements.csv`
  and `data/vacuum_elements.csv`; scripts load them by hardcoded `../data/...` paths. Compare
  `biocolors`, which ships its calibrations inside `biotuner/biocolors/data/*.npz` and loads
  them by name. The element tables (and the material dictionary) should live in
  `biotuner/bioelements/data/`.
- **One file, many concerns.** Unit conversions (Å/Hz/nm/eV), EM-spectrum regions, the
  matcher, `compute_ratios_df`, and plotting are all in one module. The composite layer needs
  a clean separation: data access, the spectrum object, composition, and the biotuner bridges.
- **No object.** Everything is a bare function over a DataFrame. A composite material needs to
  be a *thing* you can pass around, tune, colour, and compose — an object, as `biocolors` has
  `Palette` and `ColorSpec`.

### Keep

- The **NIST line tables** themselves (element, wavelength Å, intensity, persistence, type,
  spectrum_region) — 99 elements, with **intensity**, which is the amplitude analogue and the
  key to weighting.
- The **octave-fold-to-optical** idea and the **element → tuning** reduction
  (`create_elemental_tunings.py`): the composite layer reuses both, unchanged in spirit.
- The unit conversions (`Angstrom_to_hertz`, `nm_to_hertz`, `hertz_to_volt`, …) — move them
  into a `units` submodule.

---

## 3. The compositional model (the heart)

A single dataclass and a single operation carry every level of the hierarchy.

```python
@dataclass
class Composition:
    """An element, or a weighted bag of sub-compositions.

    Every material carries its position on all four coverage axes (§4.5) as
    tags, so the dictionary is auditable: a missing category is a query, not a
    hunch. Element-category coverage (axis 4) is *derived* from the constituent
    elements' NIST type, not stored — you cannot tag it wrong.
    """
    name: str
    parts: dict[str | "Composition", float]    # component -> weight
    kind: str = "compound"                      # AXIS 1  element|compound|mixture|structure
    material_class: str = ""                    # AXIS 2  molecular|ionic|metallic|…
    domain: str = ""                            # AXIS 3  geosphere|hydrosphere|…
    archetype: str = ""                         # cross-check: earth|water|air|fire|wood|metal|ether|""
    tags: dict = field(default_factory=dict)    # free-form (state, formula, source, citation)
```

Every shipped material must set `kind`, `material_class`, and `domain`; `archetype`
is optional (only the classical/Wu-Xing set uses it). The registry refuses a
material missing a required tag, so the axes cannot silently rot as the dictionary
grows.

- An **element** is a leaf: `parts` empty, its spectrum read straight from the NIST table.
- A **compound** weights elements by stoichiometry: `water = Composition("Water",
  {"Hydrogen": 2, "Oxygen": 1})`.
- A **mixture** weights compounds/elements by proportion: `air = Composition("Air",
  {"Nitrogen": 78, "Oxygen": 21, "Argon": 1})`.
- A **structure** weights materials: `cloud = Composition("Cloud", {water: 0.05, air: 0.95})`
  — note the key is another `Composition`, so composition is **recursive**.

The spectrum of any composition is the weighted superposition of its parts' spectra:

```
  spectrum(comp) = Σ_i  w_i · normalise( spectrum(part_i) )
```

evaluated recursively down to element leaves. That one function produces, for water, air,
wood, tissue, salt — measured in the proof-of-concept — a composite line list and, folded,
a **material tuning**. Everything a material *affords* (§6) is a transform of this composite
spectrum.

---

## 4. The weighting problem (this module's "calibration")

The proof-of-concept exposed the one decision that governs whether composites are meaningful,
exactly as the `/55` constant did for `biocolors`.

Weighting each element's lines by `atom_count × line_intensity` **over-weights line-rich
elements**. Measured: water came out **81% Oxygen / 19% Hydrogen** by summed weighted
intensity — because oxygen contributes 136 NIST lines to hydrogen's 44, so O dominates even
though H₂O is 2:1 H:O by *atom count*. Air, by contrast, came out **N 61% / O 36% / Ar 3%**,
which is close to reality — but that is luck, because N and O have comparable line counts.

The fix is to make the per-element contribution **budget-normalised** before applying the
stoichiometric weight:

```
  spectrum(comp) = Σ_i  w_i · ( spectrum(elem_i) / Σ intensity(elem_i) )
```

so each element contributes a unit of "presence" scaled by its weight `w_i`, independent of
how many lines NIST happens to list for it. Under this rule water is 2/3 H, 1/3 O by design,
and the line-count artefact disappears. This normalisation is the analogue of percentile
calibration in `biocolors`: **a raw physical quantity (line intensity) must be normalised
against the population it lives in before it can be compared across elements.**

Two further weighting choices should be exposed, not hardcoded, because they change what a
"material" *is*:

- **stoichiometry basis** — atom count vs mass fraction vs mole fraction. Water is 2:1 by
  atoms but 1:8 by mass (O is 16×). Different bases answer different questions ("how many
  atoms resonate" vs "how much mass resonates"); default to atom count, expose the rest.
- **line selection** — top-N by intensity, or an intensity threshold, or persistence-filtered.
  The tuning reduction is sensitive to this (the PoC's material tunings clustered near 1.0–1.2
  because the strongest lines cluster in wavelength); the `biocolors` lesson applies —
  octave-rebound and reduce with care.

---

## 4.5 Coverage methodology — spanning the space of materials

Hand-picking materials misses whole classes of matter, and the gap is measurable.
The naïve set (water, wood, air, fire, tissue, cloud + salt) touches only **4 of the
10** NIST element categories — it never uses a Transition Metal (29 elements,
including iron with the richest spectra in the table), an Alkaline Earth Metal
(Ca, Mg — bone, chlorophyll), or a Metalloid (Si — rock, most of Earth's crust).
So the dictionary is not enumerated by taste; it is enumerated by **axes**, filled
into a grid, and cross-checked by independent taxonomies so a gap in one shows up
in another.

Four coverage axes, each catching what the others miss:

1. **Compositional level** (`kind`): element → compound → mixture → structure.
   *Rule:* exemplars at every level (the recursive spine of §3).
2. **Material class** (`material_class`): molecular · ionic-salt · covalent-network
   · metallic · organic-polymer · composite · amorphous/glass · gas-mixture ·
   colloid/suspension · plasma/energetic · ice/volatile · void. *Rule:* ≥1 exemplar
   per class. This axis is what surfaces the missing metals, glass, and plasma.
3. **Natural domain** (`domain`): geosphere · hydrosphere · atmosphere · biosphere ·
   cosmosphere · technosphere · energetic-process. *Rule:* every sphere represented.
4. **Element-category coverage** (derived): the union of NIST `type` across every
   material's elements. *Rule, testable:* the dictionary must exercise **≥ 8 of the
   10** element categories. This is the falsifiable metric — an empty category is a
   failing assertion, not an opinion.

Plus a **cross-cultural archetype checklist** as a completeness heuristic and the
poetic layer: classical four (earth/water/air/fire), Wu Xing (wood/fire/earth/**metal**/water),
Pañca Bhūta (earth/water/fire/air/**ether**). Their union — earth, water, air, fire,
wood, metal, ether/void — independently flags the same gaps the data check finds
(**metal**, earth-as-mineral, **ether/void**), and ether/void maps onto the
`vacuum_elements` table. Two independent taxonomies converging on the same missing
pieces is the cross-check working.

A `coverage_report(MATERIALS)` function makes this a build gate: it prints the grid
(which axis values are filled, which cells empty), the element-category tally, and
whether the ≥8/10 assertion passes. Phase 2 does not exit until it does.

## 5. Module layout

Mirror the `biocolors` package structure — the refactor that made that module extensible.

```
biotuner/bioelements/
├── __init__.py          # re-exports + back-compat shims for the current flat API
├── legacy.py            # today's bioelements.py, frozen, deprecated
├── units.py             # Å ↔ Hz ↔ nm ↔ eV, EM-spectrum regions
├── tables.py            # load/cache the NIST line tables from data/, by name
├── spectrum.py          # ElementSpectrum: lines, intensities, normalise, fold-to-optical
├── composition.py       # Composition dataclass + recursive weighted superposition
├── materials.py         # the material dictionary (compounds, mixtures) + registry
├── matching.py          # relative-tolerance (cents) biosignal ↔ element/material matcher
├── affinity.py          # graded biosignal → material affinity scoring
├── bridges.py           # material → tuning / palette / geometry (biotuner integrations)
└── data/
    ├── air_elements.npz         # repackaged NIST tables (compressed, in-package)
    ├── vacuum_elements.npz
    └── materials_v1.yaml        # the shipped material dictionary
```

`materials.py` uses a **registry** (`@material`) so a user can add "granite" or "myelin"
without editing core, exactly as `biocolors` registers descriptors and mappings.

---

## 6. What a material affords

Because a `Composition` produces a composite spectrum, it plugs straight into the rest of
biotuner. This is the payoff — the expansion is not a silo, it is a hub.

| affordance | via | result |
|---|---|---|
| **tuning** | fold composite lines → `compute_peak_ratios` → `tuning_reduction` | a material scale (`.scl`) |
| **colour** | composite lines → `biocolors.wavelength_to_srgb` / `palette_from_signal` | a material palette |
| **chord / sound** | composite lines → `harmonic_spectrum` | a material chord to sonify |
| **geometry** | material tuning/chord → `harmonic_geometry` | a material's form (Chladni, Lissajous) |
| **affinity** | signal peaks folded, scored vs composite lines | graded biosignal → material match |

Concretely: `water.palette()`, `wood.tuning()`, `fire.chord()`, `cloud.geometry()`, and
`material_affinity(eeg_peaks, tissue)` all become one-liners over the same composite spectrum.
A biosignal can be **rendered as the material it most resonates with** — its colour, its scale,
its shape.

---

## 7. The material dictionary

The first-class deliverable of the expansion: a curated, cited dictionary of materials as
`Composition`s. Three tiers, matching the hierarchy.

**Compounds (fixed stoichiometry).** water H₂O, carbon dioxide CO₂, table salt NaCl, quartz
SiO₂, cellulose C₆H₁₀O₅, calcium phosphate (bone) Ca₅(PO₄)₃OH, chlorophyll, haemoglobin
(Fe-centred).

**Mixtures (proportioned blends).**
- **air** — N₂ 78% / O₂ 21% / Ar 0.9% / CO₂ 0.04% (mole fractions).
- **wood** — cellulose ~50% / lignin ~25% / hemicellulose ~20% / water + minerals.
- **soft tissue** — by atomic %: H 63 / O 26 / C 9 / N 1.4 / trace P, S, Ca (ICRP-style).
- **seawater**, **soil**, **blood** — analogous blends.

**Process / archetype materials.** Two of the user's targets are not compounds; they are
processes or suspensions, and the model handles both:

- **fire** — combustion is a *process*, not a formula. Model it as a hot C/H/O plasma whose
  emission is dominated by specific radiators: the sodium D lines (yellow), the C₂ Swan bands
  and CH/OH bands, over a thermal continuum. Represent as a `Composition` of {C, H, O, Na}
  with an emission-weighting that favours the visible radiators, plus an optional blackbody
  continuum term (temperature → Planck curve). Fire is where the module meets thermal physics.
- **cloud** — a suspension: water droplets in air. `cloud = Composition({water: p, air: 1-p})`
  — a *structure* whose parts are themselves materials. This is the recursive case that proves
  the model: cloud is literally "water and air as building blocks," exactly as the brief asks.

There is a poetic bridge worth naming: **water, air, fire, wood** are also the classical and
Wu-Xing archetypal elements. The dictionary can ship these as named archetypes, giving the
module a foot in both modern chemistry (the periodic table) and the elemental-archetype
vocabulary that suits biotuner's world — without mysticism, since each archetype resolves to a
concrete, cited composite spectrum.

### The tagged starter dictionary

Each entry carries its tags across all four axes; the grid is the coverage audit. This set
lights up transition metals (Fe, Cu), alkaline earths (Ca, Mg) and metalloids (Si) — the
categories the naïve set missed — while keeping the biosphere/atmosphere core.

| material | `kind` | `material_class` | `domain` | archetype | key elements |
|---|---|---|---|---|---|
| water H₂O | compound | molecular | hydrosphere | water | H, O |
| carbon dioxide CO₂ | compound | molecular | atmosphere | — | C, O |
| halite NaCl | compound | ionic-salt | geosphere | — | Na, Cl |
| quartz SiO₂ | compound | covalent-network | geosphere | earth | **Si**, O |
| iron / bronze | element / mixture | metallic | geosphere/technosphere | metal | **Fe**, Cu, Sn |
| bone Ca₅(PO₄)₃OH | compound | biomineral-composite | biosphere | — | **Ca**, P, O, H |
| cellulose (wood) | mixture | organic-polymer/composite | biosphere | wood | C, H, O |
| chlorophyll | compound | organic-polymer | biosphere | — | **Mg**, C, H, N |
| haemoglobin / blood | mixture | organic-polymer/fluid | biosphere | — | **Fe**, C, H, O, N |
| soft tissue | mixture | composite/fluid | biosphere | — | H, O, C, N |
| air (dry) | mixture | gas-mixture | atmosphere | air | N, O, Ar |
| cloud | structure | colloid/suspension | atmosphere | — | (water + air) |
| fire | structure | plasma/energetic | energetic-process | fire | C, H, O, Na |
| lightning | structure | plasma/energetic | atmosphere/energetic | — | N, O |
| stellar plasma | mixture | plasma/energetic | cosmosphere | — | H, He |
| water ice / dry ice | compound | ice/volatile | hydrosphere/cosmosphere | — | H, O / C, O |
| silica glass | compound | amorphous/glass | technosphere | — | Si, O |
| vacuum | element | void | cosmosphere | ether | — (vacuum table) |

Fifteen-plus materials, every `material_class` filled, every domain represented, and — by the
`key elements` column — all ten NIST element categories reachable once the rare-earth/actinide
edge cases (a lanthanide phosphor, uranium glass) are added. `cloud`, `fire`, and `lightning`
sit at the **structure** level, exercising the recursion; `vacuum` is the `ether` archetype and
the one material that reads from the vacuum line table.

---

## 8. Public API

```python
from biotuner.bioelements import (
    Composition, material, MATERIALS,        # the model + registry
    load_elements, element_spectrum,         # data access
    composite_spectrum,                      # the core operation
    match_elements, match_materials,         # biosignal -> {element|material} affinity
    coverage_report,                         # audit the dictionary across the four axes
)

coverage_report(MATERIALS)   # grid of filled/empty cells + the >=8/10 element-category gate

water = MATERIALS["water"]
water.spectrum()            # composite line list (wavelength, intensity, source)
water.tuning(n_steps=7)     # material scale, reduced
water.palette(fund=200)     # via biocolors
water.affinity(eeg_peaks)   # how strongly this signal resonates with water

cloud = Composition("cloud", {MATERIALS["water"]: 0.05, MATERIALS["air"]: 0.95},
                    kind="structure")
cloud.tuning()              # recursion: water+air spectra superposed, folded, reduced

# rank a biosignal against the whole dictionary
ranked = match_materials(eeg_peaks, weighting="atom", normalise="budget")
```

---

## 9. Build order

| phase | contents | testable at exit |
|---|---|---|
| **0. Solidify** | package `bioelements/`; move data in-package (`.npz`); fix `compute_ratios_df` (pandas 2); relative-tolerance matcher; `ElementSpectrum` object; back-compat shims | element tunings reproduce; matcher returns sensible hits on a known line; legacy imports unchanged |
| **1. Composition** | `Composition` + recursive budget-normalised superposition; the weighting knobs (§4) | water = 2/3 H, 1/3 O by design; air ≈ real; composite spectrum finite, sorted, sourced |
| **2. Material dictionary** | the tagged grid (§7); the registry (rejects untagged materials); `materials_v1.yaml`; `coverage_report` | every material yields a finite tuning + palette; `match_materials` ranks a signal; **`coverage_report` passes ≥8/10 element categories and every class/domain filled** |
| **3. Bridges** | `material.tuning/palette/chord/geometry`; `affinity` | a material round-trips to a `.scl`, a `Palette`, a `harmonic_geometry` form |
| **4. Structures** | recursive material-of-materials; layered/spatial arrangements → geometry | cloud (water+air) and a layered structure render; docs notebook |

Phase 0 is worth doing on its own — it fixes a real bug and packages the data — even if the
composite layer never ships. Phase 1 is where the vision becomes real. Phases 3–4 are pure
integration, because the composite spectrum already speaks every biotuner dialect.

---

## 10. Open questions for you

1. **Default weighting.** Atom count is the natural default, but "how much *mass* resonates"
   (mass fraction) is a defensible alternative and gives very different materials (water becomes
   O-dominated by mass). Ship atom-count default with the others exposed, or pick a different
   default?
2. **Fire's continuum.** Do you want fire (and hot materials generally) to carry a blackbody
   *continuum* term (temperature → Planck), or stay purely line-based like the cold materials?
   The continuum is physically right for incandescence but adds a temperature parameter and a
   different kind of spectrum.
3. **Air vs vacuum tables.** There are two NIST tables (air- and vacuum-wavelength). Air is the
   right default for terrestrial materials; should vacuum be an option (for astrophysical
   signals), or dropped?
4. **How curated should the dictionary be?** A tight, cited set (~15 materials: the classical
   archetypes + common biomaterials) is defensible and maintainable. A large auto-generated set
   (every common compound) is more impressive but harder to keep honest. Which?
5. **Archetype framing.** Lean into water/air/fire/wood as named classical/Wu-Xing archetypes
   (evocative, fits biotuner), or keep the vocabulary strictly chemical (compounds/mixtures)?

---

## Reproducing the numbers

Measured on pandas 2.3.3 against `data/air_elements.csv`:

- Broken `compute_ratios_df`, 0-match tolerance, composite dominance (water 81% O, air N 61%/O
  36%/Ar 3%), material tunings, and the biosignal→material affinity proof-of-concept:
  `scripts/bioelements_composite_poc.py`.
