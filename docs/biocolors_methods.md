# Deriving colour from a biosignal: a perceptual, metric-driven approach

## Abstract

Turning a biosignal into colour is not a display problem but a cross-modal psychophysics problem, and it has a three-century history. Newton bent the linear spectrum into a closed circle and mapped it onto the diatonic scale; Castel's ocular harpsichord and Scriabin's *clavier à lumières* made the analogy audible and visible. What all of them lacked was a perceptual metric: a principled account of when two colours are the same, when they differ, and by how much. We supply that metric and build a biosignal-to-colour module around it. The engineering rests on three psychophysical commitments: a perceptually uniform working space (OKLCh) used as an explicit observer model, so that Euclidean ΔE approximates perceived difference; a critical-band consonance model in the Helmholtz–Plomp–Levelt–Sethares lineage, so that "consonance" as a colour axis names a property of the ear; and a preference for relational over absolute representation, following Shepard, so that the meaningful quantity is the distance between signals rather than any signal's absolute hue. We contribute a dependency-free perceptual colour layer, a mapping registry spanning six families and eight methods, a percentile-calibration layer, and an explicit account of three design tensions, demonstrated on the HMC sleep-EEG corpus (subject SN001).

## Introduction

### A three-century-old problem, still missing its metric

The idea that sound and colour belong to a single sensory continuum is old and recurrent. In the *Opticks* (1704), Newton did two things that still structure every attempt in this lineage. First, he bent the linear spectrum into a closed **colour circle**, joining red to violet — an act of perceptual closure with no physical basis, since red and violet are not spectral neighbours. Second, he mapped the visible bands onto the notes of the diatonic scale. Both moves are psychophysical claims dressed as observations: that colour experience is circular, and that pitch and hue can be placed in correspondence. The eighteenth and nineteenth centuries built instruments on the same intuition — Castel's *clavecin oculaire*, later Scriabin's *clavier à lumières* in *Prometheus* — machines that played light alongside sound.

These are the aesthetic ancestors of any biosignal-to-colour system, and they share a single deficiency. None of them had a **perceptual metric**. They could assert that a note corresponds to a colour, but not measure whether two colours are perceptually the same, nor how far apart two signals lie in the space of possible appearances. Analogy is not measurement. The thesis of this paper is that the missing ingredient has always been psychophysical, and that it is now available: a perceptually uniform colour space that functions as an **observer model**, a **critical-band** model of consonance, and a commitment to **relational rather than absolute** representation. Read this way, the engineering choices in our module are not ad hoc conveniences; they are instances of established psychophysical principles, and we frame them throughout as such.

### Why the naïve maps fail

The temptation is to reach for the obvious pipeline: take a spectral feature, run it through a colour formula, and read off a hue. This fails for reasons that are conceptual before they are technical, and each failure points to the principle that repairs it.

**The octave-of-light asymmetry.** Human hearing spans roughly ten octaves; human vision spans a frequency ratio near 2:1 — about *one* octave of light. Audition is a ten-octave analyser and vision a one-octave analyser of frequency. Any pitch-to-hue map therefore compresses a wide frequency range into a single visible octave, and one octave of frequency corresponds to one full traversal of the visible band. Because every EEG epoch carries a low-frequency delta peak that folds to green, every physically grounded palette opens on the same green. This is not an implementation defect to be patched away; it is the octave-of-light asymmetry made visible, and any honest map must own it.

**A single scalar cannot separate.** Mapping one descriptor to hue produces a readable axis but a crowded wheel. A biosignal's identity is multidimensional; collapsing it to a scalar guarantees that distinct signals collide. The declarative colourspaces — which name one descriptor for each of lightness, chroma, and hue precisely so the axis *means* something — separate least of the eight methods, while the separation-oriented methods spread signals nearly twice as far, a trade we quantify later. The scalar's interpretability is bought with separation, and no formula recovers what the projection discarded.

**HSV is not an observer model.** The deeper failure is choosing a colour space that encodes nothing about the human observer. Equal steps in HSV, or in CIE *xy* chromaticity, are perceptually unequal — MacAdam's discrimination ellipses showed this directly. A colour distance is meaningful only if it is a **psychological** distance, a model of the observer rather than a property of light. This is the tradition CIELAB opened and OKLab (Ottosson) refines; ΔE in such a space approximates perceived difference. Our colour layer works in OKLCh throughout and treats chroma as **cusp-relative** — a fraction of what the observer-plus-display can actually realize — because the realizable-chroma ceiling varies sharply around the wheel. Requesting saturated red at high lightness asks for a colour that does not exist; the space must know this, and HSV does not.

### Contributions

- A **dependency-free perceptual colour layer** (`biotuner/biocolors/color/`) that imports nothing from the rest of the package and is testable against reference colour science, with legacy behaviour held bit-exact.
- A **metric-driven mapping registry** of eight methods across six families — physical, relational, statistical, declarative, learned, and interval/tuning forms — each a different answer to what a biosignal's colour should *mean*, evaluated on separation, stability, and interpretability.
- A **percentile-calibration layer** that normalises each spectral descriptor against a corpus and learns per-descriptor weights, with per-level routing across three calibrations to keep signals spread across the wheel rather than pinned into a sliver of it.
- An explicit account of **three tensions** — stability versus separation, separation versus interpretability, and physical versus statistical grounding — that we argue are structural to the problem rather than defects of any one design.

We ground the exposition in the **HMC sleep-EEG corpus** (subject SN001: 7.12 h at 256 Hz, four channels, 854 expert-scored 30 s epochs across stages W/N1/N2/N3/R), using it not as a classification target — sleep stages overlap heavily in this feature space and are not separable epoch by epoch — but as a demanding, real-world stream of biosignals against which each mapping's promises can be measured.

## A psychophysical stance

Supplying a metric is what separates a measurement from an analogy, and it is the whole content of the engineering here. The claim of this section is that the module's design choices are not ad hoc software decisions but instances of established psychophysical principle, and that reading them that way is the only way to see which choices are forced and which are conventional. Each principle below recurs in a later section, doing different work with different facts.

### Fechner's programme, and why everything is a logarithm

Psychophysics, as Fechner defined it in the *Elemente*, is the study of the quantitative relation between a physical stimulus and the sensation it evokes. This module is a psychophysics artefact in exactly that sense: it ingests a physical spectrum and emits a perceptual response, a colour. Weber's law states that the just-noticeable difference is proportional to the magnitude of the stimulus; the Weber–Fechner integration makes sensation grow with the *logarithm* of the stimulus, and Stevens' power law refines this to a compressive power function for continua such as brightness. The consequences are not decorative. Pitch is heard in octaves and cents — log frequency. Loudness is measured in decibels — log power. Perceived lightness rises through a compressive nonlinearity. The module reasons about intervals in log-frequency, treats amplitudes as decibels, and places lightness in a log-like perceptual coordinate for exactly these reasons.

This is where the prototype's most instructive bug lived. Spectral amplitudes are decibels — `10*log10(PSD)` — and 26 % of them are negative. The prototype shifted these values additively to make them positive, a scale-inconsistent operation on a logarithmic quantity: it deforms the ratios that carry the perceptual information. The symptom was stark: two N2 epochs with *identical* peak structure landed 122° apart on the hue wheel. The fix was to respect the domain — invert the log with `10**(dB/10)` before combining. A Fechnerian encoding error is not a rounding nuisance; it silently destroys the very structure the map is supposed to preserve.

### Perceptual uniformity, trichromacy, and the closed circle

The colour layer rests entirely on a perceptually uniform space, and such a space is a *psychological* construct, not a physical one. MacAdam's discrimination ellipses showed that CIE *xy* chromaticity is badly non-uniform; CIELAB was an attempt to build a space in which Euclidean distance approximates perceived difference; OKLab is a modern refinement of the same aim. The module's reliance on ΔE_OK and on cusp-relative chroma is a commitment to this stance: distance in the space is a *model of the observer*, not a property of light. Trichromacy — the Young–Helmholtz insight that colour vision rests on three cone classes — is why colour experience is three-dimensional at all, and why the realizable colours form a bounded solid whose boundary (the sRGB gamut) bites at the cusp. Asking for chroma relative to what the observer-plus-display can realize is trichromacy taken seriously. And Newton's circle is itself a psychophysical act: joining red to violet is perceptual closure with no basis in the wavelength axis, inherited every time the module treats hue as an angle so that difference is angular distance.

### Critical-band consonance behind every "consonance" axis

Several of the module's colourspaces put "consonance" on an axis, and that word is not a folk quantity here. Helmholtz grounded consonance in the absence of beating and roughness between partials. Plomp and Levelt tied consonance and dissonance to *critical bandwidth*: roughness peaks when two partials fall within a critical band of the cochlea, modelled as a bank of overlapping auditory filters (the Fletcher and Zwicker tradition, and the Bark scale). Sethares turned this into a dissonance *curve* computed directly from a sound's measured partials — tuning-independent, timbre-dependent. The module's dissonance curve and consonance metrics implement exactly this critical-band roughness psychophysics, so when a colourspace sends consonance to hue, the axis is a model of the ear rather than an arbitrary index.

### One octave of light: the hinge that forces the fold

Two facts about the senses set the terms of the whole enterprise. Audition is a ten-octave analyser of frequency; vision is a one-octave analyser. Mapping sound onto colour is therefore an intrinsically lossy dimensional compression, and the compression ratio is fixed by physiology, not by any software decision. This is the hinge on which the physical mapping turns, and it guarantees that method's limitation as tightly as its virtue. The auditory side has its own circle, and it rhymes with the colour circle: octave equivalence — tones a 2:1 apart heard as the same pitch class — is a robust cross-cultural phenomenon, and Shepard modelled pitch as a helix, monotonic height plus a circular pitch-class dimension. The closed hue circle and the closed pitch-class circle are the same kind of object, which is why folding one onto the other is natural even though it discards height.

### Shepard's relational reading of the arbitrary-hue result

The module's most consequential finding sounds at first like a failure: in the statistical and learned methods, the absolute hue assigned to a signal is a convention. Shepard's universal law of generalization holds that perceived similarity falls off exponentially with distance in an internal psychological space, and — the deeper point — that the mind traffics in *relations and structure*, not absolute coordinates. A representation that fixes pairwise distances while leaving the absolute frame free is precisely a structural representation. The arbitrariness of the absolute anchor hue is the map declining to invent an absolute code it has no grounds to assert, while faithfully preserving the structure it can measure. We develop this in the Evaluation as the paper's central result.

### Cross-modal correspondence, and categorical perception as a constraint

If absolute hue is arbitrary in the statistical methods, why is it ever legitimate to *declare* that a hue means a particular quantity? The answer is cross-modal correspondence. As Spence's review and the earlier work of Marks document, there are systematic, largely involuntary associations between dimensions of different senses; the most robust here is that higher pitch is matched to brighter or lighter. Sound symbolism — Köhler's *takete* and *maluma*, later *bouba/kiki* — shows even shape-sound mappings are shared and non-arbitrary. A colourspace that routes pitch to lightness honours a measured bias rather than inventing one. One caution follows from the same literature. Both colour, in Berlin and Kay's basic-colour-term work, and musical pitch are perceived and named *categorically*; language warps the continuum. A viewer will categorize whatever palette we hand them, so a hue difference below a category boundary may be read as "the same." This is why the module's temperament machinery is careful about *where* on the wheel it spends its range: warm hues at mid lightness are inherently muted — at L = 0.60, hue 30° reads as rust, 60° as ochre, 90° as olive — and 62 % of extended-peak EEG anchors fall into that warm-green region. The auto temperament sets a coherent single-temperature arc (45° + 130° × normalised harmonic spread); the opt-in `aurora` option instead opens a 250° arc from warm through cool, keeping all eight test signals perceptually balanced at a measured separation cost of 0.124 mean palette ΔE_OK. Managing categorical perception is a live design constraint, not an afterthought.

## From signal to colour: the pipeline

A biosignal arrives as a time series, and the first act of the pipeline is to turn it into a spectrum: a set of peak frequencies and the amplitudes at those peaks, produced by a spectral peak-extraction step. Everything downstream is a transformation of that spectral description into channels of a perceptual colour space — a sequence of physical-to-perceptual transductions, each answerable to a psychophysical law rather than to convenience.

### Amplitudes are decibels, and decibels are already Fechnerian

The single most consequential detail at the front of the pipeline is that the amplitudes are decibels — `10*log10(PSD)` — not linear power. Spectral tooling has internalised the Weber–Fechner form so thoroughly that it hands you the log without announcing it; the amplitude column is a sensation scale, not an intensity scale. The hazard is that any stage which treats a dB value as though it were linear power applies arithmetic to a quantity that already lives in log space. The remedy is to invert the encoding — recover linear power as `10**(dB/10)` — before any operation that assumes proportional magnitudes, and the effect of getting this right is not cosmetic: it raised the PCA two-dimensional coverage of the descriptor space from 66.9 % to 79.9 % and lifted the standard deviation of spectral entropy from 0.018 to 0.123, unfreezing a descriptor that had been nearly constant. A quantity that is silently logarithmic and a step that silently assumes linearity is the general shape of the failure; naming the encoding is the general fix.

### Representation levels, and why palette width is not a knob

Between "a spectrum" and "the numbers we colour" sits a choice of representation level, and there are five. `peaks` is the raw spectral maxima; `extended` adds harmonic multiples of those peaks; `ratios` and `extended_ratios` reduce the spectrum to the frequency *ratios* between components; `cons_ratios` keeps the consonant subset of those ratios. Each level is a different claim about what in the spectrum is worth representing. A direct consequence, easy to miss, is that palette width follows the representation level rather than being a parameter the caller sets: you do not ask for an eight-swatch palette, you ask for `extended_ratios` and receive as many swatches as the signal has extended ratios. The level is the substantive decision and the width is its shadow.

The level interacts with the mapping in ways that are not free to ignore. The physical mapping folds frequency onto the single visible octave, so octave-related components land on the same wavelength; harmonically extended peaks — whose 2×/4×/8× multiples *are* octaves — are therefore actively destructive there, and the physical mapping must be fed `level=peaks`. The lesson generalises: a representation level encodes a commitment about equivalence, and that commitment has to agree with the equivalence the colour space already imposes.

### Per-step metrics versus per-signal descriptors

The pipeline computes at two granularities. *Per-step* metrics are computed for each element of the representation — each peak, each ratio. Sethares-style local consonance is the canonical example: a per-step consonance value, derived from critical-band roughness, that a declarative colourspace can send to hue so each swatch's colour *means* the consonance of its own component. *Per-signal* descriptors summarise the whole spectrum into the fixed-length fingerprint — for a spectrum, eleven descriptors: harmonicity, harmonic spread, spectral flatness, spectral centroid, spectral spread, spectral entropy, n_peaks, fundamental, octave span, complexity, and Tenney height. A tuning, being a set of designed intervals with no amplitudes, makes four of these constant (spectral flatness, spectral entropy, and complexity pin to 1; fundamental to 0), so the tuning fingerprint drops to seven fields. The width of the fingerprint is dictated by what the domain can actually vary, not by a schema.

### The data flow

**Signal → Descriptors → Channels → ColorSpec → Render.**

- **Signal → Descriptors.** The peak-extraction step yields peaks and dB amplitudes; the log encoding is inverted where linear magnitudes are needed; the level fixes what is described; per-step metrics and the per-signal fingerprint are computed. *Commitment:* the dB amplitudes are a Fechnerian sensation scale.
- **Descriptors → Channels.** Descriptors and per-step metrics are assigned to lightness, chroma, and hue — by a declarative colourspace, a calibrated PCA direction, or the physical fold. *Commitment:* the channels are the three dimensions of trichromatic colour experience, and mapping a critical-band consonance value or a pitch onto one honours an auditory psychophysics or a measured cross-modal correspondence.
- **Channels → ColorSpec.** Channel values become coordinates in OKLCh, with chroma requested relative to the local gamut cusp. *Commitment:* a perceptually uniform space is an observer model, so distances in it are psychological distances.
- **ColorSpec → Render.** Gamut mapping brings every coordinate into sRGB by reducing chroma toward the cusp while preserving hue exactly. *Commitment:* the render respects the physiological-and-device boundary rather than clipping through it, so hue — the dimension chosen to carry meaning — survives the trip to the display intact.

The 122° error is the standing reminder of the cost of forgetting this: it was not a colour-space failure at all, but an arithmetic operation performed in the wrong sensory scale, one step before colour ever entered the picture.

## The colour layer as an observer model

The colour layer makes a single foundational commitment: colour is treated not as a physical property of light but as a *response* of a specific viewer. All colour arithmetic happens in OKLCh — the cylindrical form of OKLab — and when the module computes ΔE_OK as a plain Euclidean distance it is not measuring light but querying a model of the observer. That distinction organises the entire layer.

### Why OKLCh, and not HSV, CAM16, or Jzazbz

HSV is convenient and it is not an observer model: its hue, saturation, and value are algebraic reshufflings of sRGB coordinates with no claim on perceived lightness or perceived difference. Using it to place a biosignal on a colour wheel would silently smuggle the display's gamma curve and primaries into what is supposed to be a perceptual statement. At the other extreme sit the appearance models — CAM16, and spaces such as Jzazbz — which encode viewing conditions and are more faithful in absolute terms. But the target here is sRGB output on ordinary displays, and within that regime OKLab is a deliberate trade: it captures the perceptual-uniformity properties that matter while remaining cheap, invertible, and free of viewing-condition parameters we cannot pin down for an unknown reader on an unknown screen. That fidelity is checkable: the sRGB → OKLab → sRGB round trip is exact to a maximum error of 4.45×10⁻¹⁴, against 1.56×10⁻⁶ in the prototype — a lossless re-encoding of the display gamut, not a lossy approximation of it.

### Lightness as a Fechnerian nonlinearity

Equal increments of physical luminance are not equal increments of seen lightness, and the mismatch is worst in the dark tones — the Weber–Fechner and Stevens reading of brightness as a compressive continuum. The colour layer honours this by carrying lightness in Oklrab's `L_r`, a perceptual toe applying exactly that compressive nonlinearity, and only then converting to OKLab's `L`. The toe is a genuine transform, exactly invertible: the round trip closes to 2.2×10⁻¹⁶. A linear lightness axis would place perceptually crowded dark colours too far apart and sparse bright ones too close — precisely the non-uniformity the observer model exists to correct.

### The cusp: where trichromacy and the display bite

Only a bounded volume of colours is physically realizable, and OKLCh does not respect that boundary for free. At each lightness and hue there is a maximum chroma the sRGB gamut can hold — the cusp — and the critical empirical fact is how violently that ceiling moves: across the wheel it varies **6.1×**, from `max_chroma = 0.048` at (L = 0.90, h = 270°) to `0.296` at (L = 0.60, h = 315°). No fixed chroma value can be safe: a naive fixed rectangle in (L, C) is simultaneously **3× over budget** at some hues and **wastes 25 %** of the available chroma at others. The module therefore requests chroma as a fraction of `max_chroma(L, h)` — trichromacy made operational. This also dissolves a category of impossible colours: **saturated red at high lightness does not exist**. At L ≈ 0.88 the red-hue ceiling is about 0.064, so a request for chroma 0.22 there is 3.4× over budget and can only resolve by desaturating toward white. The gamut has no room for a bright vivid red; the cusp is where that fact becomes visible in the numbers.

### Cusp-preserving gamut mapping, and the spectral locus

When a request falls outside the gamut, the module reduces chroma toward the cusp while **preserving hue exactly**, on the psychophysical judgement that a viewer tolerates a less saturated colour far more readily than a wrong hue. This moves in-gamut coverage from **28.4 % to 100 %**, hue held fixed, and it is cheap: `gamut_map` processes 20 000 colours in 362 ms (18 µs each), and `srgb_to_oklab` converts 100 000 colours in 37 ms. Where the pipeline needs a wavelength-to-colour map, the observer model reappears in the choice of spectral locus: the default uses the CIE 1931 colour-matching functions via the Wyman–Sloan–Shirley analytic fit, a direct model of the standard observer's cone-integrated response. The tempting shortcut, Bruton's approximation, is kept only as an option because it distorts where the observer is most sensitive: it collapses the entire 645–750 nm band (105 nm, 28.4 % of the visible span) onto a single hue at 29.23°, where CIE 1931's worst plateau is 73 nm (19.7 %). Preferring the colour-matching functions is the same principle as preferring OKLab over HSV: stay faithful to the measured observer, and refuse to let an approximation invent perceptual identities that are not there.

### ΔE_OK and its interpretable scale

Because distance in this space is a psychological quantity, it comes with an interpretable scale: ΔE_OK below 0.02 is imperceptible, below 0.05 reads as the same colour, around 0.10 is comfortably distinct, and above 0.20 obviously different. Every separation claim in this paper is denominated in these units, which is legitimate only because OKLab is an observer model. The same model extends to viewers whose observer differs from the standard: colour-vision deficiency is simulated via the Machado *et al.* (2009) model, and under deuteranopia red and green collapse to a ΔE_OK of roughly 0.22 — a pair obviously different to a typical viewer becoming barely distinct to a dichromat. That is not a rendering artefact; it is the observer model correctly reporting a different observer.

### The observer model is where naive implementations fail

Three prototype bugs are evidence for the paper's central methodological claim: audio-to-colour code goes silently wrong precisely where it disrespects the observer model. First, the prototype's gamut fit was **dead code** — colours were clipped to the display before the fit was consulted, so out-of-gamut requests were resolved by naive channel clipping, which shifts hue; the hue-preservation guarantee existed only on paper. Second, a long-standing complaint that "bright red comes out pink" had been **misdiagnosed** as a hue error: sRGB red sits at OKLCh hue 29.23° (magenta is near 328°), so the hue was never wrong. The real cause was the chroma request of 0.22 at L = 0.88 above, forcing a desaturation toward white; the bug was invisible until the cusp was modelled. Third, `srgb_to_oklab` was **silently wrong on batched input** — a 0.024 error at index (3,3), a `ValueError` on other shapes — the most dangerous kind of failure, because it corrupts perceptual distances in bulk without announcing itself; the rewrite is exact at any batch size. Legacy behaviour, by contrast, is held bit-exact: `wavelength_to_rgb(475)` still returns `(0, 213, 255)`. Each failure lived in the gap between what the display does and what the viewer sees, and closing that gap is what it means to treat the colour layer as an observer model rather than a coordinate convenience.

## Six ways to derive colour

A biosignal is a spectrum; a colour is a point in a three-dimensional perceptual space. Any map between them must decide *what about the spectrum the colour is supposed to mean*, and that decision is a psychophysical commitment. The module registers eight mapping methods across six families, distinguished not by their code but by the kind of meaning they assign to hue.

| Method | Family | What hue encodes | Psychophysical reading | Strength | Cost |
|---|---|---|---|---|---|
| `spectral` | Physical | folded wavelength of a peak frequency | octave-of-light transcription (Newton) | absolute, corpus-free, immune to calibration failure | weak separation; every palette opens green |
| `mds` | Relational | position in an MDS embedding of the pairwise consonance matrix | Shepard relational structure; Plomp–Levelt affinity | highest palette variety | absolute placement is a free convention |
| `anchored` | Statistical | direction in a calibrated multi-descriptor PCA | Shepard: relations, not absolute coordinates | strongest, widest-arc separation | absolute hue arbitrary; needs the right corpus |
| `tonotopic` | Declarative | a named pitch descriptor | cross-modal correspondence (higher pitch → lighter) | readable axis | separates weakly |
| `consonance` | Declarative | critical-band consonance | Helmholtz–Plomp–Levelt–Sethares ear model | readable; hue monotone in consonance | separates weakly |
| `harmonic` | Declarative | a named harmonic descriptor | harmonic-structure axis | readable, invertible | separates least |
| `tenney` | Declarative | Tenney height | interval-complexity axis | readable, invertible | separates weakly |
| `derived` | Learned | direction in the data's own top principal components | Shepard relational; data-derived structure | strong, self-scaling separation | absolute hue not interpretable |

### Physical: literal cross-modal transcription

The `spectral` method is Newton's descendant: it takes a peak frequency, folds it by octaves onto the visible band, reads the resulting wavelength through the CIE 1931 colour-matching functions, and converts to sRGB. Hue *is* folded wavelength; no fingerprint, calibration, or corpus enters, so a given frequency yields the same hue in any dataset, on any day. Its strength is absolute, corpus-free meaning; its cost is separation, and the cost is the octave-of-light asymmetry made concrete. Over the 1–45 Hz EEG range — 5.5 octaves — the visible band repeats five to six times, two peaks an octave apart share a colour, and the ubiquitous ~1 Hz delta peak folds to ~532 nm, so every spectral palette opens green: hue occupancy is only 6–8 % of the wheel against 25 % for the fingerprint. The fold is well-behaved locally — a +0.25 Hz shift moves hue about 8.6°, and the 8–12 Hz band wraps cleanly — and precision is the lever that recovers separation. Rounding peaks to 0.25 Hz leaves a minimum pairwise ΔE_OK of 0.010 and 69 % distinct hues; tightening to 0.10 Hz raises those to 0.038 and 87 %, and doubles the fraction of peaks landing on physically distinct wavelengths from 28 % to 56 %. One combination must be avoided: harmonically extended peaks, whose octave multiples fold onto one wavelength — on one N2 epoch, eleven extended peaks collapsed to six.

### Relational: a similarity map onto the hue circle

The `mds` method computes the pairwise consonance matrix over a signal's steps and embeds it by classical multidimensional scaling onto the hue circle, so consonant steps land near one another. This is Shepard's psychology made visible: an embedding that discards absolute position and keeps the geometry of similarity, driven by a consonance grounded in Plomp–Levelt critical-band roughness. Its strength is the highest palette variety of any method; its cost is that, as with every relational representation, the absolute placement is a convention and only the pattern of near-and-far carries meaning.

### Statistical: hue as a direction in a calibrated fingerprint

The `anchored` method summarises each spectrum as the eleven-descriptor fingerprint, percentile-normalises each descriptor against a reference corpus, weights them, and projects through a stored two-dimensional PCA; hue is the angle of that projection, `atan2(pc2, pc1)`, and chroma scales with the percentile-ranked magnitude. This is the method that separates best, and the honesty the rest of the paper turns on lives here: the absolute anchor hue is arbitrary, corpus-dependent, and the method failed twice before it worked (both failures are dissected under Calibration). Its strength is separation; its costs are corpus-dependence and an absolute hue that means nothing. `anchored` answers *are these signals different, and by how much?* — not *what colour is this signal?*

### Declarative colourspaces: hue that names a quantity

The declarative family refuses the trade the previous two make. Here the analyst *names* the axes: this descriptor drives lightness, that one chroma, this one hue, so a reader who knows the convention can invert the colour back to the measurement. The module ships four — `tonotopic`, `consonance`, `harmonic`, and `tenney`. The justification is cross-modal correspondence: routing pitch to lightness honours the robust *higher pitch → lighter* bias, and when `consonance` drives hue the axis is a critical-band model of the ear in the Helmholtz–Plomp–Levelt–Sethares lineage. That the mapping is genuinely readable can be shown directly: in `consonance`-space hue is monotone in the named quantity, sweeping from 250° to 30° as consonance rises from 4.8 to 42 — you can read the number off the colour. The cost is separation, and it is structural: because hue is pinned to a descriptor scale shared across all signals, signals that share a descriptor value share a hue whether or not they differ elsewhere. This is the interpretability tax, a defensible trade rather than a defect.

### Learned: axes discovered in the data

The `derived` method lets the data choose its axes, taking the top three principal components of a per-step feature matrix as L, C, and h. Psychophysically this is the same relational stance as `mds` — a data-driven internal space in which position encodes structure — and it inherits the same caveat: a rotation of the eigenbasis is as valid as any other, so absolute hue carries no fixed meaning. Its strength is a strong, self-scaling separation obtained without committing to a named descriptor; its cost is that, like every learned embedding, it tells you that two signals differ and roughly along what latent contrast, but not what the colour *is*.

### Interval and tuning forms: colouring a sound's own structure

The sixth family turns inward. `dyad_field` assigns hue by interval class, completing one full turn of the wheel per octave with lightness carrying consonance — a direct visual analogue of the pitch helix, where the circular dimension is pitch class and the 2:1 octave is read as sameness. `consonance_spectrum` colours a signal's own Sethares dissonance curve, so the valleys of consonance and peaks of roughness become visible bands. The psychophysical content here is the most concrete in the module, because the object being coloured *is* a psychophysical function. The strength of this family is fidelity to a single sound's own perceptual structure; its cost is precisely that it makes no cross-signal claim. These are forms for looking *into* a signal, not for telling two signals apart.

## Calibration

### Why a hand-tuned constant cannot be right

The prototype normalised the consonance descriptor by dividing by a fixed constant of 55. A single constant encodes an assumption about a descriptor's range, and that assumption was wrong for the data. On 30 EEG-like signals mean consonance spans 0.318 to 4.957; dividing by 55 maps every one into the bottom 9 % of the hue wheel, with a median adjacent gap of 0.20° and 242 of 435 pairs falling within 5° of each other. On 60 real EEG epochs the same constant yields a median hue gap of 1.26°, with only 7 % of pairs separated by more than 5°. The failure is not a poorly chosen constant; it is the idea of a constant. A just-noticeable difference proportional to stimulus magnitude — Weber's law — cannot be honoured by a linear rescaling that treats the whole range uniformly.

### Percentile normalisation as an ecology model

The fix replaces the constant with a **percentile normalisation against a reference corpus**: each descriptor is ranked against the distribution it actually takes across a large body of signals, and the hue and chroma scales derive from that rank. A perceptually uniform colour space is already a model of the observer; percentile normalisation extends the same stance one layer out, fitting the map to the population of stimuli the observer will encounter, so the available perceptual range is spent where signals actually differ. Against the EEG corpus, replacing the constant with the fingerprint normalisation raised the median hue gap from 1.26° to 3.95° and lifted wheel occupancy from 47 % to 78 %.

### Informativeness weighting

Not every descriptor carries information on every corpus. Calibration therefore learns a **per-descriptor weight equal to 1 minus the modal-bin fraction**: a descriptor whose values pile into a single histogram bin is nearly constant and is down-weighted. On the EEG spectra, `n_peaks` and `complexity` collapse to weights of 0.003–0.004 — degenerate — and are suppressed, so they cannot inject spurious hue rotation. This is the same discipline as the amplitude fix upstream: a quantity that does not vary should not be allowed to speak.

### Three calibrations and per-level routing

A calibration is a fitted object, valid only on the domain it was fitted to. The module ships three: `eeg_sleep_v1`, learned from spectra; `tuning_v1`, learned from 532 designed scales (n-TET, just intonation, harmonic and subharmonic, inharmonic, stretched); and `tuning_eeg_v1`, learned from 6821 EEG-derived tunings. These are not interchangeable. Scoring a designed tuning against the EEG spectrum corpus saturates 64–91 % of the descriptors and pins all four sleep stages to a single hue of 69° with a minimum gap of zero — even though the raw projection angles were 348°, 335°, 36°, and 1° apart. The remedy is **per-level routing**: each of the five representation levels is sent to the calibration fitted for its domain. The gap is stark — `extended_ratios` scored on `tuning_v1` spans 60° (median gap 19.6°, mean ΔE_OK 0.098); scored on `tuning_eeg_v1` it spans 270° (median gap 92.5°, mean ΔE_OK 0.205).

### TUNING_FIELDS

Because a tuning has no amplitudes, four of the eleven fingerprint descriptors are constant on any scale (`spectral_flatness` = 1, `spectral_entropy` = 1, `complexity` = 1, `fundamental` = 0), and feeding constants into a fit is the degeneracy problem in its purest form. Dropping them to a seven-field descriptor set (`TUNING_FIELDS`) moves the anchor by at most 0.11° — the constants carried no information to lose — while raising tuning PCA coverage to 86.5 % / 72.3 % and dropping descriptor saturation on designed scales from 18–45 % to 0 %. Calibration, in the end, is not a knob to be turned but a fit to be matched: to the right population, with the right descriptors, evaluated on the right domain.

## Three tensions

The eight methods do not form a quality ranking; they occupy a design space governed by three tensions, and every method is a particular resolution of them. Naming the tensions is the paper's organising claim: there is no free lunch to be found by tuning, because the trade-offs are structural.

### Stability versus separation

A stateless map gives the same signal the same palette every time — `spectral`, in particular, sends a given frequency to a given hue in any dataset, immune to every calibration failure above. That stability is exactly what precludes optimal separation, because knowing how far apart to push two palettes requires knowing about *the other signals*. The module exposes the choice directly: `mode=absolute` returns the stable, context-free colour; `palette_set(mode=separated)` spends context to pull a set of signals apart. A colour that means the same thing everywhere cannot also be the colour that best distinguishes this particular set, because the second property is a property of the set, not of the signal.

### Separation versus interpretability

The clearest reading of this tension comes from Shepard: the mind traffics in **relations and structure, not absolute coordinates**. The methods split precisely along this line, which the Evaluation quantifies. The separation-oriented methods (`mds`, `spectral`, `anchored`) top the variety table; the declarative colourspaces sit at the bottom — not from a defect but as the direct cost of interpretability. In a declarative colourspace hue *means* a named descriptor, so hue is pinned to a scale shared across all signals and cannot be spent freely to separate them. The statistical anchor does the reverse, and pays for maximal separation with absolute interpretability. Four equally valid conventions — an eigenvector sign flip, a PC1↔PC2 swap, a reference rotation — give different absolute anchor hues but *identical* pairwise angular distances (46°, 102°, 176° preserved across all four). Bootstrap resamples move the anchors about 2° of circular standard deviation, reproducibly — but reproducible is not meaningful, and only the angular distance between signals is corpus-independent. Read through Shepard, the arbitrary absolute hue is not a bug: it is the system correctly representing structure rather than inventing an absolute code the perception does not warrant. The declarative map makes the opposite bet, buying a nameable axis at the price of relational reach.

### Physical versus statistical grounding

The last tension is about where the map's authority comes from. `spectral` is physically grounded — Hz octave-folded on wavelength, through the CIE 1931 functions to sRGB — needing no corpus, and it is bulletproof for exactly that reason; its cost is the octave-of-light compression that confines it to one colour family. `anchored`, the statistical map, separates far better but draws its authority from a corpus, and failed twice when it lacked the right one. Physical grounding buys immunity at the cost of reach; statistical grounding buys reach at the cost of a dependency that can, and did, break. The three tensions do not resolve to a winner: a method is a stance on all three at once, and the right choice is a property not of the module but of the question.

## Evaluation

### The corpus and what "0 failures" does and does not mean

Every quantitative claim is measured on the HMC sleep-staging database, subject SN001: 7.12 h at 256 Hz across four EEG channels, segmented into 854 expert-scored 30 s epochs. Crossed with channels this yields 3416 signals, and the pipeline extracted a fingerprint and a colour for all 3416 with **0 failures**. That number is a robustness claim, not an accuracy claim: it says the extraction path is numerically total over a realistic range of neural spectra, not that the resulting colours are correct in any absolute sense. We evaluate colour assignments with falsifiable metrics — mean and minimum pairwise ΔE_OK, hue occupancy, and collisions below a perceptual threshold — which are meaningful only because the working space is an observer model. A claim like "these two epochs are the same colour" is therefore checkable against a number, not asserted.

### Variety by method

Across 10 sleep-EEG chunks (extended peaks, 0.10 Hz precision), mean pairwise palette ΔE_OK orders the methods:

| Method | Family | Mean pairwise ΔE_OK |
|---|---|---|
| `mds` | relational | 0.182 |
| `spectral` | physical | 0.179 |
| `anchored` | statistical | 0.153 |
| `derived` | learned | 0.147 |
| `tonotopic` | declarative | 0.121 |
| `consonance` | declarative | 0.118 |
| `tenney` | declarative | 0.094 |
| `harmonic` | declarative | 0.080 |

The separation-oriented methods score highest and the declarative colourspaces lowest — the interpretability tax quantified. When a colourspace sends "consonance" to hue, the axis is a psychophysical model of the ear in the Helmholtz–Plomp–Levelt–Sethares lineage: critical-band roughness computed from the signal's own partials, not an arbitrary index. The interpretability the low ΔE_OK appears to "cost" is exactly what that axis buys.

### Arbitrary absolute hue confirms a relational thesis

The `anchored` and `derived` methods raise a result that would be a scandal if colour were meant to be an absolute code: the absolute hue is a convention, fixed only up to a free rotation, while the pairwise angular distances are invariant and corpus-independent. We read this as confirmation, not caveat. Shepard's universal law of generalization holds that the mind represents relations and structure rather than absolute coordinates, and that perceived similarity is a function of distance in an internal space. A representation whose absolute origin is free while its pairwise distances are invariant is precisely a structural representation. The `anchored` method therefore answers a question about structure — *are these signals different, and by how much?* — and honestly declines to answer *what colour, in itself, is this signal?*, because that question presupposes an absolute code the data do not contain.

### Not a sleep classifier

A tempting misreading of a sleep-EEG corpus is that the colours should sort by sleep stage. They do not. In fingerprint space the five stages overlap heavily; individual 30 s epochs are not separable by stage, and within-stage variability is large — the anchor's circular standard deviation inside N2 alone is 150° at 30 s. Per-stage palettes are legible only after aggregating on the order of 500+ epochs, at which point the palette describes a stage's central tendency rather than classifying any epoch. The module produces a perceptual coordinate for a spectrum; the overlap is a property of the neural signal, not a failure of the colour map.

## Limitations

**Single subject, single extractor.** The calibrations rest on narrow ground: `eeg_sleep_v1` is built from one subject, each calibration fit with a single extraction configuration. Percentile normalisation and the stored PCA are corpus-statistics, so cross-subject and cross-montage transfer is unestablished. The 0-failure result speaks to numerical totality over one recording, not generality across people or hardware.

**Absolute hue is arbitrary for the statistical and learned methods.** `anchored` and `derived` carry no absolute colour meaning; their hue is fixed only up to a free rotation. We reframe this as principled rather than apologise for it, but a reader who wants a colour to *name* a signal in isolation must use a physical or declarative method instead, knowingly.

**Octave collapse in the physical map.** Hearing spans ~10 octaves, vision ~1; mapping the former onto the latter is lossy compression, so octave-equivalent peaks share a colour and every palette opens green. Precision partly offsets this, but harmonically extended peaks must never be combined with the fold. This is a fact about the two senses, not a bug to be patched.

**Calibration-domain sensitivity.** Using the wrong calibration degrades silently into meaninglessness rather than erroring — a designed tuning scored against the EEG corpus pinned all four sleep stages to a single hue — which is why per-level routing and the tuning-appropriate field set are load-bearing, not cosmetic. A statistical method needs the right corpus, and there is no corpus that is right for everything.

**Several strong results arrived only after a confident wrong guess.** The dead-code gamut fit, the "bright red is pink" misdiagnosis, the batched `srgb_to_oklab` error, and the dB scale error each improved a headline number once fixed, but each was preceded by a plausible, confident, wrong hypothesis. The honest reading is that the perceptual metric is what made the errors visible: without a uniform space and a ΔE_OK threshold, "close enough" hides them.

## Conclusion

The taxonomy is the deliverable, and it is a toolbox, not a ranking. Read through psychophysics, the three tensions are one question asked three ways, and the answer is that there is no single right colour for a biosignal — there is a right colour for a *question*. If the question is physical identity ("what frequency is this?"), the octave-fold gives an absolute, corpus-free hue in Newton's tradition of bending the spectrum into a circle. If the question is relational structure ("how do these signals differ?"), the statistical and learned methods give invariant pairwise distances, and their arbitrary absolute hue is Shepard's law made visible: perception traffics in relations, not absolute codes. If the question is a readable quantity ("how consonant is this?"), a declarative colourspace maps a named psychophysical axis — critical-band roughness, Tenney height, tonotopic pitch — directly to a perceptual dimension, honouring cross-modal correspondences rather than inventing them. The audio-to-colour project is three centuries old, from Newton's diatonic spectrum through Castel's and Scriabin's instruments; what those attempts lacked was a perceptual metric. Supplying one — a uniform colour space as an observer model, a critical-band consonance model as an ear, and Fechnerian log encoding throughout — is what turns the analogy into a measurement. The colour you should derive from a biosignal depends on which question you want it to answer, and the contribution here is a module that lets you choose, and tells you what each choice costs.

## References

- Berlin, B. & Kay, P. — *Basic Color Terms* (1969).
- Castel, L.-B. — *clavecin oculaire* (ocular harpsichord).
- CIE — CIELAB colour space (1976).
- Fechner, G. — *Elemente der Psychophysik* (1860).
- Fletcher, H. — critical bands of hearing.
- Helmholtz, H. von — *On the Sensations of Tone* (1863).
- Köhler, W. — *takete/maluma* (sound symbolism); later *bouba/kiki*.
- Krumhansl, C. — probe-tone studies of tonal hierarchy.
- MacAdam, D. — discrimination ellipses (1942).
- Marks, L. — cross-modal correspondence.
- Newton, I. — *Opticks* (1704).
- Ottosson, B. — OKLab colour space (2020).
- Plomp, R. & Levelt, W. — tonal consonance and critical bandwidth (1965).
- Scriabin, A. — *clavier à lumières* (*Prometheus*).
- Sethares, W. — *Local consonance* (1993); *Tuning, Timbre, Spectrum, Scale* (1998).
- Shepard, R. — circular/helical model of pitch (1964); universal law of generalization (*Science*, 1987).
- Spence, C. — cross-modal correspondences: a review (2011).
- Stevens, S. S. — the power law of sensation (1957).
- Weber, E. H. — Weber's law (the just-noticeable difference).
- Young, T. & Helmholtz, H. von — trichromatic theory of colour vision.
- Zwicker, E. — the Bark critical-band scale.

## Availability

The module is available as `biotuner.biocolors`, with the perceptual colour layer under `biotuner/biocolors/color/` importing nothing from the rest of the package and testable against reference colour science. All empirical results are computed on the HMC sleep-EEG corpus, subject SN001.