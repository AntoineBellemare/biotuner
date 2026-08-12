# `biotuner.mos` — moment-of-symmetry scales, end to end

## Why this package exists

Before this work, MOS support in biotuner was three fragments that never met:

| Where | What | Problem |
|---|---|---|
| `scale_construction.py` L967–1359 | `find_MOS`, `tuning_MOS_info`, `Stern_Brocot`, `tuning_range_to_MOS`, … | brute-force step counting; `octave` silently ignored in several places; bare `except: pass`; parallel-list dict output; no scale object |
| `vizs.py` `plot_labyrinth` | polar plot | not the labyrinth — radius was `sig.index(max(sig)) + 1`, i.e. always 1 or 2 |
| `vizs.py` `MOS_interactive` | ipywidgets | plotted generator stacks, not the scale universe; no landmarks, arcs, or spokes |

Nothing connected MOS to a biosignal. `compute_biotuner` had no way to ask
*"which well-formed scale does this brain live in?"*

`biotuner.mos` replaces the fragments with one coherent layer built on the exact
combinatorics of Milne, Carlé, Sethares, Noll & Holland (2011), *Scratching the
Scale Labyrinth*, LNAI 6726, 180–195.

## What the paper contributes, and where each idea lands

| Paper | § | Implemented in |
|---|---|---|
| WF/MOS = generated, two step sizes, maximally even (Christoffel word) | 2 | `theory.mos_word`, `scale.MOSScale` |
| Stern–Brocot enumerates every MOS of a generator | 3 | `theory.sb_walk`, `theory.mos_cardinalities` |
| Three landmark tunings per MOS pair | 2, 3 | `theory.mos_landmarks` |
| Valid tuning range = bracket-to-mediant interval | 3 | `theory.signature_ranges` |
| Embedding scale has `2p + q` tones, at mediant `(2a+c)/(2b+d)` | 3 | `theory.embedding` |
| Coherence ⇔ Blackwood `R = L/s < 2` ⇔ between equalized and embedding | 2, 3 | `theory.coherence_range`, `metrics.is_proper` |
| Co-prime `(nL, ns)`; inverse scale swaps them | 2 | enforced in `theory`, `MOSScale.inverse` |
| Myhill's property; unique interval signature per degree | 2 | `metrics.myhill_property`, `metrics.degree_signatures` |
| The labyrinth: rings, angles, spokes, arcs | 4 | `plotting.plot_labyrinth`, `interactive.labyrinth_plotly` |
| Named temperaments / optimal-tuning lines on the labyrinth | 4 | `temperaments.py` — computed from commas, not hardcoded |
| Modes: parsimony, height–width duality, free ℤ² of rank 2 | 4 | `modes.py` |
| Fourier Scratching: play states in ℂⁿ, DFT, partials | 5 | `fourier.py` |
| Dynamic Tonality: partial pitch = α·period + β·generator | 6 | `timbre.py` |
| The labyrinth as a surface to move *across* — structure and tuning chosen together, timbre following | 1–3, 6 | `morph.py` — `tuning_morph`, `tree_morph`, `voice_morph`, `morph_audio` |

## New: biosignals → MOS

The piece the paper does not have. `derive.py` treats the labyrinth as a
*search space* and asks which well-formed scale best explains a signal.

The fit has three coordinates matching the labyrinth's own choices —
**generator**, **cardinality**, **period** — plus a fourth *nuisance*
coordinate, **transposition**, because a scale and its transpositions are the
same scale. That last one is not optional: a stack of fifths is the pentatonic,
but only in one of the pentatonic's five modes. Pin every candidate to a 1/1
root and the answer is missed (10.6 ¢ error instead of 0.9 ¢ on the worked
example).

- `generator_candidates` — every observed ratio, every ratio *between* ratios,
  plus a background grid. De-duplication is priority-aware and runs one way
  only: the grid is thinned against the signal so a grid point never shadows an
  exact signal-derived generator, but two signal-derived candidates are never
  thinned against each other. Which of two real proposals is better is a
  question about the fit, and nothing at the candidate stage can answer it.
- `fit_mos` — amplitude-weighted mean cents error to the nearest degree, plus a
  penalty per surplus note. The default penalty is calibrated, not guessed:
  across fourteen known scales fitted from five jittered peaks it recovers the
  true signature 12/14 times, versus 2/14 with no penalty (which overfits to a
  median of 21 notes).
- `MOSFit.improvement` — error relative to what a scale of that size would get
  on *random* input. A large scale always sits close to any data, so the raw
  error means little without this.
- `MOSFit.evidence` — the same margin below chance, in units of the standard
  error it was measured with. `improvement` still rewards a tiny target set for
  fitting exactly; `evidence` is the one that survives both failure modes, and
  it is what `compare_sources` ranks on.
- `forward_scales` / `ForwardScale` — the other direction; see below.
- `mos_trajectory` — a path through the labyrinth over time.
- `compute_biotuner.fit_mos()` / `.compare_mos_sources()` / `.plot_labyrinth()`
  / `.mos_trajectory()`, `mos_from_biotuner(bt, mode=…)`, and `'mos'` as a
  `get_tuning` source.
- `plotting.plot_forward_vs_inverse` — both directions on one labyrinth.

### Two directions, two questions

There are two ways to put a generator and a signal in the same sentence, and
they are not variants of one method. They ask different questions and are
allowed to disagree.

| | **inverse** — `fit_mos`, the default | **forward** — `forward_scales` |
|---|---|---|
| The generator is | *latent*: solved for | *given*: an interval the signal states |
| Question | which well-formed scale best explains these peaks? | if this audible interval were the generator, what scale would the signal be playing? |
| Free parameters | generator, cardinality, rotation (period too, optionally) | rotation only |
| The answer is | a fit | a consequence |
| Fails by | overfitting | explaining the signal badly |

Both score against the same targets with the same objective, so `error_cents`,
`coverage`, `score` and `evidence` mean the same thing on a `ForwardScale` as on
a `MOSFit` and the two lists can be read against each other.
`mos_from_biotuner(bt, mode='forward')` is the switch; `'inverse'` is the
default and is unchanged.

**When each is the right question.** Ask the inverse when you want the scale and
do not care where the generator came from — identification, comparison across
conditions, feeding `get_tuning('mos')`. Ask the forward when the *provenance*
of the generator is the claim: to say "this signal's alpha-to-beta ratio,
stacked, gives a pentatonic" is only honest if the interval you stacked is one
the signal produced. The inverse will happily hand you a generator no pair of
peaks states — a feature when identifying a scale, a problem when making a claim
about what the signal contains.

#### What the inverse actually does

It is easy to state loosely — "find the generator whose scale contains the
peaks" — and every clause of that is wrong. Precisely, `fit_mos`:

1. **Folds the inputs to pitch classes first**, merging anything within
   `FOLD_TOLERANCE_CENTS` and summing the weights, so the targets it fits are
   distinct pitch classes rather than the ratios as handed in. See *Folding*
   below; the point here is only that it happens before anything else.
2. **Searches (generator, cardinality, rotation) jointly**, not in stages. For
   each candidate generator it enumerates that generator's own MOS
   cardinalities — the note counts at which stacking is well-formed at all —
   builds the scale at each, and inside every one of those evaluations chooses
   the best rotation. Rotation is enumerated rather than sampled: under an
   absolute-error objective the optimal transposition always lands some target
   exactly on a degree, so the candidate set is `{tᵢ − dⱼ}` — restricted during
   the coarse scan to the `n_anchors` heaviest targets, then re-scored over
   every target for the survivors. Only afterwards are the leading generators
   refined inside their valid tuning ranges, which sharpens the tuning and
   cannot change the signature. `forward_scales` skips the shortlist
   (`n_anchors=None`) because it has far fewer readings to score, so its
   rotations are exact from the start.
3. **Matches each ratio to its *nearest* degree.** Containment is never
   required, in either direction: no ratio has to equal a degree, and no degree
   has to be claimed by a ratio. Fitting `[1, 1.19, 1.34, 1.51, 1.68]` returns
   `5L2s` at 699.51 ¢ and 4.17 ¢ mean error — one target sits 13.95 ¢ off its
   degree and still counts as a hit at the 15 ¢ tolerance, and two of the seven
   degrees go unused (`MOSFit.n_unmatched_degrees`).
4. **Ranks by error *plus* a complexity penalty**, never by error alone.
   Nearest-degree matching means a bigger scale is always at least as close, so
   raw error is a race to the largest ring allowed. On that same five-ratio
   probe, `complexity_penalty=0` returns `12L7s` — nineteen notes, 1.66 ¢ —
   where the default `1.0` returns the seven-note `5L2s` at 4.17 ¢ (score
   4.17 + 2 surplus notes = 6.17).

**The inverse generator is latent, and need not be any interval you can point
at.** Two demonstrations, both reproducible:

- Take a stack of fifths, `[1, 9/8, 81/64, 3/2, 27/16]`, and *delete* `3/2`. The
  fit still returns `2L3s` at **701.9550 ¢** with zero error. The fifth is no
  longer one of the ratios, but 27/16 over 9/8 is a fifth, and the search has no
  reason to care about the difference.
- Stronger, because here the generator is not among the observed intervals at
  all: four alpha-band peaks from real EEG (S001, eyes closed — 10.07 / 15.64 /
  19.31 / 22.91 Hz), capped at six notes, fit `1L3s` at **930.00 ¢**, 18.44 ¢
  error, 50 % coverage. The six intervals those peaks state, folded, are 660.9,
  762.2, 835.1, 904.0, 976.9 and 1127.1 ¢. The nearest is **26.0 ¢ away**. The
  winning generator is not in the signal; it is the value that best organises
  it.

#### What forward mode does

`forward_scales` is the complement, and it refuses to invent anything. It
enumerates the intervals the signal *states* — each ratio against the reference
as `(r, 1.0)`, and each unordered pair as larger-over-smaller — declares each one
the generator, stacks it, folds it into the period, and reads off the MOS that
falls out. Then it scores that scale against the whole target set with the same
objective and the same transposition freedom, so the number is comparable to an
inverse fit rather than merely adjacent to it.

Nothing about the generator is optimised, and no code path could optimise it:
across all 39 readings the four EEG peaks produce at `max_cardinality=24`,
`scale.generator` equals the folded observed quotient to **0.0** — bit-identical,
not "to within a cent". Refining it would swap the observed interval for a nearby
unobserved one and quietly turn the forward reading back into an inverse fit.

The same four peaks, each pair taken as the generator, printed at the smallest
scale each one supports:

| interval | ratio | generator | scale |
|---|---|---|---|
| 22.91 / 15.64 | 1.465 | 660.9 ¢ | `2L3s` (5 notes) |
| 15.64 / 10.07 | 1.553 | 762.2 ¢ | `3L2s` (5 notes) |
| 19.31 / 15.64 | 1.235 | 835.1 ¢ | `3L4s` (7 notes) |
| 22.91 / 19.31 | 1.186 | 904.0 ¢ | `4L1s` (5 notes) |
| 22.91 / 10.07 | 2.275 | 976.9 ¢ | `1L4s` (5 notes) |
| 19.31 / 10.07 | 1.918 | 1127.1 ¢ | `1L4s` (5 notes) |

Two things that table hides. It is *smallest per generator*, not the ranking —
`forward_scales` returns every `(generator, cardinality)` pair ordered by score,
and one generator can occupy many rings at once: 1127.14 ¢ alone accounts for
fourteen of those 39 rows. And raw frequencies need `include_ratios=False`,
because 19.31 is a frequency rather than an interval and reading it as a
generator means nothing; peak *ratios* are intervals, and the default is right
for them.

An empty list is a legitimate answer rather than a failure —
`forward_scales([1, 2, 4])` returns `[]`, because pure octaves state no interval
that generates anything.

Proposals landing within `dedupe_cents` of each other are one reading, and the
grouping is done on the *sorted* proposals rather than in arrival order. The
input is conceptually a set of ratios, so permuting it must return the same
readings in the same order with the same numbers (only `targets`,
`assignments` and `residuals` follow the caller's list, because they are
defined to) — and a greedy walk in arrival order does not, because the first
arrival gets to define its window and speak for it. On four ratios whose
quotients state generators at 699.75 ¢ and 700.25 ¢, reversing the list used to
swap a `7L5s` at 699.75 ¢ (16.75 ¢ error) for a `5L7s` at 700.25 ¢ (17.50 ¢) —
same numbers, inverted signature. Each window is now represented by whichever of
its proposals *scores* best, and every pair that proposed into it stays in
`sources`, so `n_sources` keeps counting corroboration rather than election
results.

#### Where the two directions meet

They are not condemned to disagree. Raise the EEG fit's ceiling to twelve notes
and both land on `1L9s`: the inverse at 1127.0142 ¢ (1.03 ¢ error), the forward
at 1127.1414 ¢ (1.29 ¢). Turn the inverse's refinement off and it reports
1127.1414 ¢ exactly — the forward reading *is* the inverse search's own
candidate, before refinement slid it 0.13 ¢ off the observed value to buy a
quarter of a cent. Agreement like that is the interesting outcome: the latent
generator turned out to be audible after all. Disagreement, as at the six-note
cap above, is the ordinary one, and the size of the gap is the price of
insisting that the generator be an interval you can point at.

### The bright half — a convention, not a finding

A generator `g` and its complement `period − g` build **the same scale**. The two
pitch-class sets are mirror images, and for a well-formed scale the mirror is
always a *mode* of the original: `2L3s` built on 0.584963 of an octave (701.955 ¢,
the fifth) and on 0.415037 (498.045 ¢, the fourth) differ only by rotation — the
second is rotation #3 of the first. Since every fit here is rotation-invariant,
`g` and `period − g` are one solution, not two.

So `derive._fold_bright` folds every generator into the open bright half
`(0.5, 1)`, and both directions call that one function rather than
reimplementing the rule — which is what makes their generators directly
comparable on one axis. Two fractions generate nothing and come back as `None`:
`0` (the unison and the bare period, which never leave the root) and `1/2`
(which closes after two notes). Both are refused with a tolerance
(`derive.GENERATOR_EPSILON`, 1e-9 of a period) rather than by exact comparison,
because neither value survives the arithmetic that produces it: a half-period
interval arrives as `log(2**0.5) / log(2)`, which is `0.5000000000000001`, and
an exact test lets it through to build a "five-note" scale with two pitch
classes in it.

The consequence is worth stating out loud, because it looks like a result:
**`fit_mos` and `forward_scales` can never report a generator below half the
period.** Build a `2L3s` explicitly on a 498.045 ¢ generator, hand its ratios to
either function, and both answer 701.955 ¢ with zero error — the same scale, in
its bright spelling. A labyrinth carrying markers on one side only is showing
this convention, not a signal that avoided the other half.
`plotting.plot_forward_vs_inverse` therefore draws the bright half alone
(θ ∈ [180°, 360°]) rather than leaving an empty semicircle to be misread as
absence of evidence. `fit_field`, which samples the whole circle instead of
reporting a winner, keeps both halves — and they carry the same errors, which is
exactly why the labyrinth picture is left–right symmetric.

### Any derivation can feed the fit

`compute_biotuner` turns a signal into ratios eight different ways, and they do
not agree. Each is now a first-class input to the fit — `bt.fit_mos(source=…)`,
`mos_from_biotuner(bt, source=…)`, `bt.mos_trajectory(source=…)` — and
`compare_sources` (`bt.compare_mos_sources()`) runs the whole set and ranks it,
so the question stops being *"which scale is this signal in?"* and becomes
*"which way of asking produces a well-formed answer at all?"*.

Measured on the notebook's worked example — 30 s of a 5 Hz fundamental with a
stack of fifths above it under noise, FOOOF peaks at 5.00 / 7.50 / 9.99 / 11.25
/ 16.88 Hz, `peaks_extension` run so `extended_ratios` resolves,
`max_cardinality=16`:

| source | ratios → targets | best fit | error ¢ | improvement | **evidence** | coverage |
|---|---|---|---|---|---|---|
| `peaks_ratios` | 9 → 7 | `2L3s` @ 702.7 ¢ | 0.58 | 104× | **4.54** | 1.00 |
| `extended_ratios` | 18 → 17 | `12L1s` @ 1101.5 ¢ | 9.95 | 2.3× | **4.06** | 0.72 |
| `diss_curve` † | 8 → 8 | `7L3s` @ 848.6 ¢ | 5.19 | 5.8× | **4.05** | 0.88 |
| `euler_fokker` † | 9 → 8 | `4L7s` @ 881.3 ¢ | 6.93 | 3.9× | **3.65** | 0.89 |
| `HE` † | 5 → 5 | `1L8s` @ 1096.4 ¢ | 4.71 | 7.1× | **3.33** | 0.80 |
| `cons_ratios` † | 2 → 2 | `1L3s` @ 951.0 ¢ | 0.000002 | 3.4 × 10⁷ | **2.45** | 1.00 |
| `harm_fit_tuning` | 37 → 37 | `3L13s` @ 821.9 ¢ | 16.28 | 1.15× | **1.39** | 0.54 |
| `harm_tuning` | — | *`ValueError`, reported in the* `reason` *column* | — | — | — | — |

† underdetermined: the winning scale has more degrees than the data had targets.

**"Most convincing" cannot mean lowest error, and the table is the proof.** The
derivation with by far the smallest error is `cons_ratios` at two *millionths*
of a cent — and it ranks sixth of seven. It found two ratios; four degrees can
be rotated so that both land exactly, so the number measures the scale's spare
capacity rather than the signal. `improvement` does not rescue the ranking
either: dividing by chance error puts the same two-point fit on top by seven
orders of magnitude. `evidence` is the column that behaves, because it counts
how much data the margin below chance was measured over —
`sqrt(3·n_targets)·(1 − error/chance)`, the number of standard errors the
weighted mean falls below what a random ratio set would score against a scale
that size. It self-limits at `sqrt(3n)`, so two targets can never exceed 2.45
however exact they are, while seven exact targets reach 4.58. Rows are sorted by
it, failures last.

Read down the column and the table says something about the derivations
themselves. `peaks_ratios` wins by recovering the structure that was planted —
`2L3s` at 702.7 ¢ is a stack of fifths cut at five notes, the pentatonic.
`harm_fit_tuning` has the most data and the least structure: 37 ratios form a
ladder dense enough that the largest scale allowed still misses nearly half of
them (coverage 0.54, improvement 1.15× — barely distinguishable from chance).
And `harm_tuning` gets a row rather than silence. It raises here because
`self.all_harmonics` is only measured by `peaks_extraction(peaks_function=
'harmonic_recurrence')` and this object used FOOOF; the exception text lands in
the `reason` column verbatim. A shorter table is not a report of a broken
source.

Two refusals are deliberate. `source='mos'` raises rather than returning a
spectacular 0.00 ¢: `get_tuning('mos')` hands back the ratios of an earlier fit,
so the answer is guaranteed in advance. And amplitude weighting is applied only
where a weight vector genuinely lines up — `bt.amps` for `peaks_ratios`,
`bt.extended_amps` for `extended_ratios`, each on an exact length match, nothing
padded or resampled. On the signal above *no* source received weights, because
five peaks give nine pairwise ratios and twenty extended peaks give eighteen
de-duplicated ones. Rejecting is the right outcome: a vector that merely happens
to be the right length would scramble the weighting silently.

### Folding: a ratio list is not a target list

A scale has no way to tell `1/1` from `2/1`. They are one degree, and a
derivation that emits both has not supplied two independent facts. `fit_mos`
therefore folds its inputs into the period and merges pitch classes within
`FOLD_TOLERANCE_CENTS` (1.0 ¢) before fitting — one cent being large enough to
absorb the near-duplicates a real derivation emits and more than an order of
magnitude below the 15 ¢ default hit tolerance, so it can never merge two
degrees the fit would otherwise distinguish. `fold=False` restores the old
behaviour exactly.

It is not an accuracy fix, and saying so matters. Merged weights are *summed*,
not discarded, which preserves the weighted mean exactly: `[1, 9/8, 5/4, 3/2, 2]`
fits `2L3s` at 3.2259442404033445 ¢ with folding on and with it off, the same
float. What folding corrects is the **sample size** and everything computed from
it. `euler_fokker` above lists both 1.0 and 2.0, so its 9 ratios are 8 pitch
classes: `n_targets` 9 → 8, the surplus-note penalty 8.93 → 9.93, and evidence
3.88 → 3.65. A 12-EDO chromatic written out with both `1/1` and `2/1` is
thirteen numbers naming twelve pitch classes, and unfolded it scores 6.245
instead of 6.000 — a quarter of a standard error bought from a data point that
repeats one already counted. Three of the seven derivations that ran fold
something on this signal (`peaks_ratios` 9 → 7, `extended_ratios` 18 → 17,
`euler_fokker` 9 → 8), so this is routine, not a corner case.

Where the error *does* move is on near-duplicates, because merging keeps the
first occurrence's position rather than averaging: `[1.125, 1.12533, 1.332, 1.5,
1.50044, 1.688, 1.998]` fits `2L3s` at 0.257 ¢ folded and `5L2s` at 0.402 ¢
unfolded. Changing the winner is rare — across 100 trials (six signatures × four
tunings × three ways of duplicating the octave, plus forty random six-ratio sets
with `1/1` and `2/1` appended) the winning signature changed once — but it is
possible, which is why `MOSFit` carries `n_merged` and `targets`, the ratios
actually fitted, so `assignments` and `residuals` always have something to run
parallel to.

### Underdetermined fits are flagged, not dropped

`fit_mos([1.5])` returns a four-note `1L3s` at 0.000 ¢ with unbounded
improvement. One data point, a perfect fit, and nothing learned: a scale with
spare degrees can be rotated until every target lands on some degree.
`MOSFit.is_underdetermined` is exactly `n_targets < cardinality`.

Nothing is dropped, because dropping would be its own dishonesty. A five-peak
recording cannot produce twelve targets, and a `5L7s` that genuinely describes
the signal should still be named — the defect is in reading its error as
evidence, not in the structure. So the fit is returned unchanged and labelled at
every surface: `explain_fit` prints `UNDERDETERMINED  1 target for 4 degrees: a
scale with spare notes can be rotated onto any data, so this error is not
evidence`, alongside a `chance` line giving the baseline error and the evidence
in standard errors; `compare_sources` carries it as a column; `info()` appends
it to the MOS line.

Expect to see it. Four of the seven working derivations in the table above are
underdetermined at `max_cardinality=16`, and the default `complexity_penalty=1.0`
is cheap enough that a scale with spare degrees often still wins. That is the
honest reading of five peaks, not a bug — and if the ratio is unwelcome the lever
is `complexity_penalty` or a tighter `max_cardinality`, not the flag. `evidence`
already keeps such rows from dominating a comparison on their own.

## Moving between scales

The other piece the paper points at without formalising. Its §1 describes the
labyrinth as a surface a musician moves *across*, choosing structure and tuning
at once. `morph.py` makes the movement itself the object, and offers three
strategies that are genuinely different journeys rather than three spellings of
one.

| Strategy | What moves | Shape of the path | Every frame well-formed? |
|---|---|---|---|
| `tuning_morph` | the generator; the note count is held | along a single arc | yes |
| `tree_morph` | the structure, one legal move at a time | hops between rings | yes |
| `voice_morph` | the notes themselves | leaves the map | no — deliberately |

Each returns a `Morph`: a sequence of `MorphStep` frames carrying degrees, the
signature where the frame has one, and the events worth hearing (a landmark
crossed, a note count changed, tones split or merged). One pair settles that the
three are not the same journey drawn three ways — `5L2s` in 12-EDO to `4L3s` in
19-EDO costs **7229 ¢** of total voice motion as a tuning morph, **3816 ¢** as a
voice morph and **1858 ¢** as a tree morph, and only the tuning morph passes five
equal temperaments and flips its signature twice (`5L2s → 3L4s → 4L3s`) on the
way.

**The signature graph is the labyrinth's own connectivity.** A signature's
children are `(nL, nL+ns)` and `(nL+ns, ns)` — the Stern–Brocot mediant, which is
why `5L2s`'s child `5L7s` has exactly the twelve tones `theory.embedding`
predicts. Its parent is the subtractive Euclidean step run backwards. One further
edge swaps `(nL, ns)` for `(ns, nL)`, and that is a single *continuous* move
rather than a jump, because a scale and its inverse meet at their shared
equalized landmark. `signature_route` searches this graph best-first, and since
shortest routes are rarely unique the tie-break is musical rather than arbitrary:
among equally short routes take the one whose sequence of note counts is
lexicographically smallest — the one that stays small longest and adds notes only
when it must. Pentatonic to chromatic then reads `2L3s → 3L2s → 5L2s → 5L7s`,
five–five–seven–twelve, rather than the equally short `2L3s → 2L5s → 7L5s → 5L7s`,
which reaches twelve tones a step early and sits there.

**`voice_morph` leaves the space of well-formed scales on purpose.** The first
two strategies cannot leave it: every frame is a `MOSScale` by construction. The
third glides each tone the shorter way round the circle to its counterpart, and
the pitch sets in between are generally not well-formed at all — `5L2s` to `4L3s`
in 64 frames spends 62 of them off the map, straying up to 18.2 ¢ from the
nearest well-formed scale. That distance is the measurement, not the defect:
`MorphStep.wellformedness` records it per frame (fitted back with
`derive.fit_mos`), and it is exactly what makes gliding the notes audibly a
different journey from sliding a generator between the same two endpoints.

Two details keep those numbers honest. The best rotation between the two pitch
sets decides *which* tone goes where, but the path travels from the unrotated
source, so the first frame is the start scale rather than a transposition of it;
and the voice count is constant across the whole morph even when the two scales
differ in size, with split tones starting coincident. `Morph.voices` records
which degree belongs to which voice, so `trajectory()` gives every voice its own
column and `voice_leading_distance()` measures motion the tones actually perform
instead of charging a crossing to both of them. `morph_audio` renders the result
as one continuous glide per voice, optionally with `timbre`-matched partials (§6)
so the timbre tracks the tuning as it moves.

## Layout

```
biotuner/mos/
  theory.py         pure number theory (stdlib only, exact Fractions)
  scale.py          MOSScale — the central frozen object
  modes.py          Mode, mode lattice, ℤ² height–width duality
  metrics.py        propriety, Myhill, evenness, JI error, harmonicity
  temperaments.py   comma → saturated mapping → HNF → optimal generator
  derive.py         biosignal → MOS (fit, trajectory, candidates)
  fourier.py        Fourier Scratching play states
  timbre.py         Dynamic Tonality partial mapping
  plotting.py       matplotlib: labyrinth, tree, wheel, ranges, modes, fits
  morph.py          moving between scales: tuning / tree / voice journeys
  interactive.py    plotly + ipywidgets explorers
```

Dependency order is strictly downward: `theory` → `scale` → `modes`/`metrics` →
everything else. `theory.py` imports only the standard library, so its
correctness is testable in isolation.

## Two places the paper needed sharpening

Both were found by testing its claims rather than assuming them.

**Fig. 8's coverage claim is about a mode, not a scale.** The paper says a
coherent well-formed scale "will be played in generic scalar order by the first
partial play state" — *n* evenly spaced fingers on a keyboard whose keys are as
wide as the steps above their tones, each key struck once. That holds only in
the rotation whose step pattern is the Christoffel word, which is the
floor-quantisation of the equal division. In any other mode two fingers share a
key and another is missed. `modes.christoffel_mode` returns the right one, and
`tests/mos/test_fourier.py` verifies coverage across every proper signature,
swept end to end through the coherent range.

**POTE is not "constrain the period, then least-squares".** `temperaments.py`
exposes both: `generator_cents` holds the period pure from the outset (CTE) and
`pote_generator_cents` optimises freely then rescales to a pure octave, which is
what published tables quote. They agree to a fraction of a cent for accurate
temperaments and diverge by several for inaccurate ones — meantone is 697.21 ¢
under CTE and 696.24 ¢ under POTE. Labyrinth overlays default to POTE so they
can be cross-checked against the literature.

## Back-compat

`scale_construction`'s MOS functions keep working unchanged.
`vizs.plot_labyrinth` and `vizs.MOS_interactive` now delegate to the corrected
implementations, keeping their signatures.
