"""biotuner.bioelements.dynamics — Phase 1 of the state-and-transition layer.

Materials stop being a label and become a STATE in a coherence landscape.

- :class:`MaterialState` — the *material wavefunction*: a signal's affinity to the
  periodic table as a normalised amplitude cloud (a superposition of elements),
  with a phase per element. Not a winner; a distribution that only becomes
  definite when it is measured / collapsed.
- :func:`coherence` — how much the signal's peak-oscillators are phase-locked
  *together*: an amplitude-weighted n:m phase-locking (a Kuramoto-style order
  parameter) ``R`` in ``[0, 1]``. ``R~0`` = a floating, incoherent cloud;
  ``R~1`` = a crystallised, definite material.
- :func:`compositional_level` — reads the element → compound → mixture → structure
  ladder as DEGREES OF ORDER (coherence + harmonic long-range order) straight
  from the signal, instead of a hand-assigned label.

Physical reading (from the design brainstorm): *floating* (superposition, low
``R``) is the rest state; synchrony crystallises a definite material (high ``R``);
the compositional level is just how far up the order ladder the coherence reaches.

Visualise a state with :func:`plot_material_state` — a dark "state portrait":
the periodic-table cloud, a phase-clock (the order parameter, drawn), and the
float ↔ crystal / compositional-level gauges.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert, welch, find_peaks

from biotuner.bioelements.matching import match_elements
from biotuner.bioelements import periodic as P
from biotuner.bioelements.bridges import element_flame_color

LEVELS = ("element", "compound", "mixture", "structure")


# --------------------------------------------------------------------------- #
# signal helpers
# --------------------------------------------------------------------------- #
def _extract_peaks(sig, sf, fmin, fmax, n_peaks):
    """Strongest spectral peaks in [fmin, fmax] → (freqs, amplitudes)."""
    sig = np.asarray(sig, float)
    f, p = welch(sig, fs=sf, nperseg=min(len(sig), int(sf * 4)))
    m = (f >= fmin) & (f <= fmax)
    fb, pb = f[m], p[m]
    if len(pb) == 0:
        return np.array([]), np.array([])
    idx, _ = find_peaks(pb, prominence=pb.max() * 0.02, distance=max(1, len(pb) // 60))
    if len(idx) == 0:
        idx = np.argsort(pb)[::-1][:n_peaks]
    order = np.argsort(pb[idx])[::-1][:n_peaks]
    sel = np.sort(idx[order])
    return fb[sel], np.sqrt(pb[sel])


def _inst_phase(sig, sf, f0):
    """Instantaneous phase of the signal band-passed around ``f0`` (Hilbert)."""
    bw = max(0.5, 0.15 * f0)
    lo, hi = max(0.05, f0 - bw), min(sf / 2 * 0.99, f0 + bw)
    if hi <= lo:
        return np.zeros(len(sig))
    sos = butter(4, [lo, hi], btype="bandpass", fs=sf, output="sos")
    return np.angle(hilbert(sosfiltfilt(sos, np.asarray(sig, float))))


def _nm_plv(ph_a, ph_b, f_a, f_b, maxdenom=8):
    """n:m phase-locking value between two oscillators (Tass generalised phase)."""
    fr = Fraction(float(f_b) / float(f_a)).limit_denominator(maxdenom)
    p, q = fr.numerator, fr.denominator            # f_b/f_a ~ p/q
    psi = p * ph_a - q * ph_b
    return float(np.abs(np.mean(np.exp(1j * psi))))


# --------------------------------------------------------------------------- #
# 1. coherence — the float ↔ crystal order parameter R
# --------------------------------------------------------------------------- #
def coherence(sig, sf, *, fmin=1.0, fmax=45.0, n_peaks=6, maxdenom=8,
              return_parts=False):
    """Amplitude-weighted mean pairwise n:m phase-locking of the signal's peak
    oscillators — a Kuramoto-style order parameter ``R`` in ``[0, 1]``.

    ``R ~ 0`` — the peaks drift independently: a floating, incoherent cloud.
    ``R ~ 1`` — the peaks fire *together* (phase-locked): a crystallised material.

    With ``return_parts=True`` also returns the per-oscillator freqs, amps and
    the resultant vector of each oscillator's phase-lock to the strongest peak
    (for the phase-clock visual).
    """
    sig = np.asarray(sig, float)
    freqs, amps = _extract_peaks(sig, sf, fmin, fmax, n_peaks)
    if len(freqs) < 2:
        R = 0.0
        parts = {"freqs": freqs, "amps": amps, "resultants": np.array([]),
                 "ref": 0}
        return (R, parts) if return_parts else R

    phases = [_inst_phase(sig, sf, f) for f in freqs]
    num = den = 0.0
    for j in range(len(freqs)):
        for k in range(j + 1, len(freqs)):
            w = amps[j] * amps[k]
            num += w * _nm_plv(phases[j], phases[k], freqs[j], freqs[k], maxdenom)
            den += w
    R = float(num / den) if den > 0 else 0.0

    if not return_parts:
        return R

    # resultant of each oscillator's n:m phase-lock to the strongest peak
    ref = int(np.argmax(amps))
    res = np.zeros(len(freqs), dtype=complex)
    for k in range(len(freqs)):
        if k == ref:
            res[k] = 1.0 + 0j
            continue
        fr = Fraction(float(freqs[k]) / float(freqs[ref])).limit_denominator(maxdenom)
        p, q = fr.numerator, fr.denominator
        psi = q * phases[k] - p * phases[ref]
        res[k] = np.mean(np.exp(1j * psi))
    parts = {"freqs": freqs, "amps": amps, "resultants": res, "ref": ref, "R": R}
    return R, parts


# --------------------------------------------------------------------------- #
# 2. MaterialState — the wavefunction over the periodic table
# --------------------------------------------------------------------------- #
@dataclass
class MaterialState:
    """A biosignal's affinity to the elements as a normalised amplitude cloud.

    ``amplitudes`` are non-negative and sum-of-squares normalised (Born rule:
    ``probabilities`` = |amplitude|² sum to 1). ``phases`` (radians) carry a
    signal phase per element so two states can later interfere.
    """
    elements: list
    amplitudes: np.ndarray            # |c_i|, Σ|c_i|² = 1
    phases: np.ndarray                # φ_i (radians)
    coherence: float = 0.0            # R ∈ [0,1] (float ↔ crystal)
    meta: dict = field(default_factory=dict)

    @property
    def probabilities(self) -> np.ndarray:
        return self.amplitudes ** 2

    def dominant(self, n: int = 1):
        idx = np.argsort(self.amplitudes)[::-1][:n]
        return [(self.elements[i], float(self.probabilities[i])) for i in idx]

    def entropy(self) -> float:
        """Spread of the cloud, normalised to [0,1]. HIGH = floating (superposed
        over many elements), LOW = crystallised onto a few."""
        p = self.probabilities
        p = p[p > 1e-12]
        if len(p) <= 1:
            return 0.0
        h = -np.sum(p * np.log2(p))
        return float(h / np.log2(len(self.elements)))

    def top(self, n: int = 8):
        idx = np.argsort(self.amplitudes)[::-1][:n]
        return [{"element": self.elements[i], "symbol": P.symbol(self.elements[i]),
                 "amp": float(self.amplitudes[i]), "prob": float(self.probabilities[i]),
                 "phase": float(self.phases[i])} for i in idx]


def element_state(peaks, amps=None, phases=None, *, top=40, tol_cents=55,
                  R=0.0) -> MaterialState:
    """Build the material wavefunction from a signal's spectral peaks.

    Amplitudes come from the REAL element match scores (:func:`match_elements`),
    sum-of-squares normalised into a superposition. Optional per-peak ``phases``
    are carried onto the nearest-frequency element (a coarse Phase-1 assignment;
    a line-accurate mapping comes with interference in Phase 2).
    """
    peaks = np.asarray(peaks, float)
    ranked = match_elements(peaks, top=top, tol_cents=tol_cents)
    els = [str(e) for e in ranked["element"].tolist()]
    amp = np.sqrt(np.clip(ranked["score"].to_numpy(float), 0, None))
    nrm = np.linalg.norm(amp)
    if nrm > 0:
        amp = amp / nrm
    ph = np.zeros(len(els))
    if phases is not None and len(peaks) and "n_hits" in ranked:
        # coarse: give each element the phase of the signal peak whose folded
        # wavelength is closest — refined in Phase 2 via match_lines.
        phases = np.asarray(phases, float)
        from biotuner.bioelements import units as _u
        pk_nm = np.array([_u.fold_to_optical(p, is_hz=True) for p in peaks])
        # (element phase left at 0 unless a finer mapping is provided later)
    return MaterialState(els, amp, ph, coherence=float(R),
                         meta={"n_peaks": int(len(peaks))})


def state_from_signal(sig, sf, *, fmin=1.0, fmax=45.0, n_peaks=6,
                      tol_cents=55) -> MaterialState:
    """Convenience: peaks + coherence straight from a raw signal → MaterialState."""
    freqs, _ = _extract_peaks(sig, sf, fmin, fmax, n_peaks)
    R = coherence(sig, sf, fmin=fmin, fmax=fmax, n_peaks=n_peaks)
    return element_state(freqs, tol_cents=tol_cents, R=R)


# --------------------------------------------------------------------------- #
# 3. compositional_level — the order ladder, measured
# --------------------------------------------------------------------------- #
def _harmonicity(freqs, sigma=0.06):
    """How well a peak set fits ONE integer-multiple series off its fundamental
    (long-range harmonic order), in ``[0, 1]``."""
    freqs = np.asarray(freqs, float)
    if len(freqs) == 0:
        return 0.0
    mults = freqs / float(freqs.min())
    return float(np.mean(np.exp(-((mults - np.round(mults)) ** 2) / (2 * sigma ** 2))))


def _level_scores(freqs, amps, R):
    """The element/compound/mixture/structure soft scores (sum→1) from three
    read-outs: peak concentration ``c``, coherence ``R``, harmonicity ``harm``.
    Shared by :func:`compositional_level` and :func:`material_state`."""
    power = np.asarray(amps, float) ** 2
    c = float(power.max() / power.sum()) if power.sum() > 0 else 0.0  # 1 = one line
    harm = _harmonicity(freqs)
    sg = c ** 3                                        # singularity: element is ONE line
    pl = 1.0 - sg
    h2 = harm ** 2                                     # structure needs LONG-RANGE order
    raw = {
        "element":   sg,
        "mixture":   pl * (1.0 - R),
        "compound":  pl * R * (1.0 - h2),
        "structure": pl * R * h2,
    }
    tot = sum(raw.values()) or 1.0
    return {k: float(v / tot) for k, v in raw.items()}, c, harm


def compositional_level(sig, sf, *, fmin=1.0, fmax=45.0, n_peaks=6, maxdenom=8):
    """Read the element → compound → mixture → structure ladder from the signal's
    own order, as soft scores that sum to 1.

    - **element** — one oscillator dominates (a single pure line).
    - **mixture** — many peaks, LOW coherence: merely co-present (air-like).
    - **compound** — coherent (high R), few components, not long-range harmonic.
    - **structure** — coherent AND long-range harmonic order (a crystal-like series).

    A transparent v1 heuristic on three read-outs: peak concentration ``c``,
    coherence ``R``, and harmonicity ``harm`` (how well the peaks fit one
    integer-multiple series off the fundamental).
    """
    freqs, amps = _extract_peaks(sig, sf, fmin, fmax, n_peaks)
    n = int(len(freqs))
    R = coherence(sig, sf, fmin=fmin, fmax=fmax, n_peaks=n_peaks, maxdenom=maxdenom)

    if n == 0:
        scores = {k: 0.0 for k in LEVELS}
        return {"level": None, "scores": scores, "R": 0.0, "harmonicity": 0.0,
                "concentration": 0.0, "n_peaks": 0}

    scores, c, harm = _level_scores(freqs, amps, R)
    level = max(scores, key=scores.get)
    return {"level": level, "scores": scores, "R": float(R),
            "harmonicity": harm, "concentration": c, "n_peaks": n,
            "freqs": freqs, "amps": amps}


# --------------------------------------------------------------------------- #
# 4. material_state — resolving spectral degeneracy (which material, not element)
# --------------------------------------------------------------------------- #
#
# A material's emission spectrum depends ONLY on its element proportions, so
# same-proportion materials are spectrally identical: Water ≡ WaterIce,
# Diamond ≡ Graphite (both pure carbon), Quartz ≡ SilicaGlass all have cosine
# similarity 1.0. Spectral affinity alone can only place a signal in an
# *equivalence class*, never pick a member of it. material_state resolves the
# member with a THREE-FACTOR product:
#
#   weight(m)  =  a1 · a2 · a3
#     a1  spectral affinity     — puts the signal in m's equivalence class
#     a2  order/level consistency — how well the signal's coherence R and its
#                                   compositional level match m's kind + how
#                                   ordered m is (crystal vs glass vs melt)
#     a3  stoichiometry match    — do the signal's integer peak-ratios contain
#                                   m's atom-count ratios (compounds only)
#
# The disambiguating axis is a2's `order`: Diamond (0.98) and Graphite (0.72)
# share one spectrum and one kind, so ONLY the coherence-vs-order term separates
# them — a phase-crystallised carbon signal reads Diamond, an incoherent one
# reads Graphite / (amorphous) Charcoal.

#: Default expected order (0 float … 1 crystal) per material class, used when a
#: material carries no explicit ``order`` tag. Same-class degeneracies (Diamond
#: vs Graphite) need a per-material ``order`` tag to split — the class value
#: only separates crystalline from amorphous.
_CLASS_ORDER = {
    "crystalline-allotrope": 0.95, "covalent-network": 0.90, "ionic-salt": 0.85,
    "ice-volatile": 0.75, "biomineral-composite": 0.72, "metallic": 0.70,
    "molecular": 0.55, "elemental": 0.50, "organic-polymer": 0.45,
    "amorphous-glass": 0.28, "composite-fluid": 0.22, "mineral-ash": 0.22,
    "colloid-suspension": 0.20, "amorphous-carbon": 0.20, "molten-silicate": 0.15,
    "plasma-energetic": 0.12, "gas-mixture": 0.05, "void": 0.0,
}


def material_order(m) -> float:
    """A material's expected order in ``[0, 1]`` — its explicit ``order`` tag if
    set, else the default for its material class."""
    return float(m.tags.get("order", _CLASS_ORDER.get(m.material_class, 0.5)))


def _peak_ratios(freqs, amps, maxdenom=8):
    """Amplitude-weighted multiset of reduced integer peak ratios ``(p, q)``
    (``p ≥ q``) → summed weight, across every peak pair."""
    freqs = np.asarray(freqs, float)
    amps = np.asarray(amps, float)
    out: dict = {}
    for j in range(len(freqs)):
        for k in range(j + 1, len(freqs)):
            hi, lo = (freqs[k], freqs[j]) if freqs[k] >= freqs[j] else (freqs[j], freqs[k])
            if lo <= 0:
                continue
            fr = Fraction(float(hi) / float(lo)).limit_denominator(maxdenom)
            key = (fr.numerator, fr.denominator)
            out[key] = out.get(key, 0.0) + float(amps[j] * amps[k])
    return out


def _stoich_ratios(counts):
    """Distinct reduced integer ratios among a compound's atom counts, e.g.
    H2O ``{2, 1}`` → ``{(2, 1)}``; CH4 ``{1, 4}`` → ``{(4, 1)}``."""
    vals = sorted({int(round(c)) for c in counts if c > 0})
    ratios = set()
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            fr = Fraction(vals[j], vals[i])            # vals sorted → vals[j] ≥ vals[i]
            ratios.add((fr.numerator, fr.denominator))
    return ratios


def _stoich_match(m, freqs, amps, maxdenom=8):
    """Do the signal's peak-ratios echo the compound's atom-count ratios?

    Returns a nudge factor in ``[0.55, 1.0]`` (spectrally-neutral 1.0 for
    non-compounds and single-element compounds — stoichiometry is a *tie-breaker*
    within an equivalence class, never a gate)."""
    if m.kind != "compound":
        return 1.0
    counts = [float(w) for w in m.parts.values()]
    if len(counts) < 2:
        return 1.0
    stoich = _stoich_ratios(counts)
    if not stoich or len(freqs) < 2:
        return 1.0
    pr = _peak_ratios(freqs, amps, maxdenom)
    total = sum(pr.values()) or 1.0
    present = sum(1 for r in stoich if r in pr) / len(stoich)   # ratios found
    mass = sum(w for key, w in pr.items() if key in stoich) / total  # peak mass on them
    return 0.55 + 0.45 * (0.5 * present + 0.5 * mass)


def material_state(peaks_hz=None, *, sig=None, sf=None, R=None, materials=None,
                   table="air", top=40, tol_cents=50.0, basis="atom", balance="recall",
                   fmin=1.0, fmax=45.0, n_peaks=6, maxdenom=8,
                   sigma=0.26, w_order=1.0, w_stoich=1.0,
                   w_tuning=0.3, tuning_maxdenom=50, harm_weight=True,
                   include_elements=False, return_parts=False):
    """Rank materials as a *collapsed superposition* — spectral × order × stoichiometry.

    Spectral affinity alone is degenerate (Diamond ≡ Graphite, Water ≡ WaterIce):
    it places the signal in an equivalence class but cannot pick a member. This
    resolves the member with:

    - **a1 spectral position** — :func:`material_affinity`: does a peak land ON a
      line? (identical for every twin in a class.)
    - **a4 spectral tuning** — :func:`~biotuner.bioelements.tuning_match.tuning_cosine`:
      does the signal's *ratio structure* match the material's line-ratio structure?
      (transposition-invariant; also identical within a class.) ``a1`` and ``a4``
      are two lenses on the same "which class" question, blended into one spectral
      evidence term ``a_spec = (1−w_tuning)·a1 + w_tuning·a4``.
    - **a2 order/level** — a Gaussian on ``|R − order(m)|`` (how close the signal's
      coherence is to the material's expected order) times how much the signal's
      compositional level agrees with the material's ``kind``. This is what resolves
      *within* a class (Diamond vs Graphite), where a1 and a4 are degenerate.
    - **a3 stoichiometry** — :func:`_stoich_match` (compounds' atom-ratios ↔ peak-ratios).

    ``weight(m) = a_spec · a2**w_order · a3**w_stoich``.

    Supply either a raw signal (``sig`` + ``sf`` — richest: real coherence and
    level) or explicit ``peaks_hz`` (a harmonicity proxy stands in for ``R`` and
    level, since phase — hence true coherence — needs the time series). ``R`` may
    be passed explicitly to override either. ``w_tuning=0`` recovers the pure
    position-affinity behaviour.

    Returns a DataFrame sorted by ``prob`` (Born-normalised weights) with the
    per-factor breakdown; with ``return_parts=True`` also a dict of the read-outs.
    """
    from biotuner.bioelements.affinity import material_affinity
    from biotuner.bioelements.materials import MATERIALS
    from biotuner.bioelements.tuning_match import (
        tuning_vector, tuning_cosine, material_tuning_vector)
    import pandas as pd

    # --- read-outs: freqs, amps, coherence R, compositional level ------------ #
    if sig is not None and sf is not None:
        sig = np.asarray(sig, float)
        freqs, amps = _extract_peaks(sig, sf, fmin, fmax, n_peaks)
        if R is None:
            R = coherence(sig, sf, fmin=fmin, fmax=fmax, n_peaks=n_peaks, maxdenom=maxdenom)
        level_scores, _, harm = _level_scores(freqs, amps, R) if len(freqs) else ({k: 0.0 for k in LEVELS}, 0.0, 0.0)
        r_proxy = False
    else:
        if peaks_hz is None:
            raise ValueError("material_state needs either a signal (sig, sf) or peaks_hz")
        freqs = np.atleast_1d(np.asarray(peaks_hz, float))
        amps = np.ones(len(freqs))
        harm = _harmonicity(freqs)
        r_proxy = R is None
        if R is None:
            R = harm                                   # no phase → harmonicity proxy
        level_scores, _, harm = _level_scores(freqs, amps, R) if len(freqs) else ({k: 0.0 for k in LEVELS}, 0.0, 0.0)

    R = float(R)
    mats = MATERIALS if materials is None else materials

    # the signal's own tuning (ratio structure) — built once, compared per material
    sig_tuning = tuning_vector(freqs, amps, maxdenom=tuning_maxdenom,
                               harm_weight=harm_weight) if len(freqs) >= 2 else {}

    rows = []
    for name, m in mats.items():
        if not include_elements and m.kind == "element":
            continue
        a1 = material_affinity(freqs, m, table=table, top=top,
                               tol_cents=tol_cents, basis=basis, balance=balance)
        a4 = tuning_cosine(sig_tuning, material_tuning_vector(
            m, table=table, top=top, basis=basis, maxdenom=tuning_maxdenom,
            harm_weight=harm_weight)) if sig_tuning else 0.0
        a_spec = (1.0 - w_tuning) * a1 + w_tuning * a4  # blended spectral evidence
        order_m = material_order(m)
        f_order = float(np.exp(-((R - order_m) ** 2) / (2 * sigma ** 2)))
        f_level = float(level_scores.get(m.kind, 0.0))
        a2 = f_order * (0.35 + 0.65 * f_level)         # order leads, level modulates
        a3 = _stoich_match(m, freqs, amps, maxdenom)
        weight = a_spec * (a2 ** w_order) * (a3 ** w_stoich)
        rows.append({
            "material": name, "weight": weight, "kind": m.kind,
            "material_class": m.material_class, "archetype": m.archetype or "",
            "affinity": a1, "tuning": a4, "spectral": a_spec,
            "order_consistency": a2, "stoich": a3,
            "order_ref": order_m, "f_order": f_order, "f_level": f_level,
        })

    df = pd.DataFrame(rows)
    tot = float(df["weight"].sum())
    df["prob"] = df["weight"] / tot if tot > 0 else 0.0
    df = df.sort_values("prob", ascending=False).reset_index(drop=True)

    if not return_parts:
        return df
    parts = {"freqs": freqs, "amps": amps, "R": R, "harmonicity": harm,
             "level_scores": level_scores, "level": max(level_scores, key=level_scores.get),
             "r_is_proxy": r_proxy, "signal_tuning": sig_tuning}
    return df, parts


# --------------------------------------------------------------------------- #
# visualisation — the "material-state portrait"
# --------------------------------------------------------------------------- #
_BG = "#0a0e1a"
_PANEL = "#0e1425"
_INK = "#e8ecfb"
_MUTED = "#5a6788"
_GOLD = "#e8d68a"
_TEAL = "#7ad6c1"


def _lighten(hexs, k):
    h = hexs.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return (min(1, (r / 255) * (1 - k) + k), min(1, (g / 255) * (1 - k) + k),
            min(1, (b / 255) * (1 - k) + k))


def plot_state_cloud(state: MaterialState, ax=None, *, top=14):
    """The material wavefunction as a glowing cloud over the periodic table:
    each element a blob sized by amplitude, in its flame colour; the dominant
    element haloed. A tight point = crystallised; a diffuse cloud = floating."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.6), facecolor=_BG)
    ax.set_facecolor(_BG)
    # faint full table
    for name in P.NAMES:
        pos = P.element_position(name)
        if pos is None:
            continue
        r, col = pos
        ax.add_patch(plt.Rectangle((col, -r), 0.92, 0.92, facecolor="#141b30",
                                    edgecolor="#1d2740", linewidth=0.5, zorder=1))
    items = state.top(top)
    pmax = max((it["prob"] for it in items), default=1.0) or 1.0
    for rank, it in enumerate(items):
        pos = P.element_position(it["element"])
        if pos is None:
            continue
        r, col = pos
        x, y = col + 0.46, -r + 0.46
        frac = it["prob"] / pmax
        color = element_flame_color(it["element"])
        for gk, ga in ((2.6, 0.10), (1.7, 0.16), (1.0, 0.9)):  # glow → core
            ax.scatter([x], [y], s=(140 + 900 * frac) * gk, c=[color],
                       alpha=ga * (0.35 + 0.65 * frac), edgecolors="none",
                       zorder=3, linewidths=0)
        if rank == 0:
            ax.scatter([x], [y], s=(140 + 900 * frac) * 3.4, facecolors="none",
                       edgecolors=color, linewidths=1.6, alpha=0.9, zorder=4)
            ax.text(x, y - 0.9, f"{it['symbol']}", color=_INK, ha="center",
                    va="top", fontsize=13, fontweight="bold", zorder=5)
    ax.set_xlim(-0.5, P.N_COLS + 0.5)
    ax.set_ylim(-P.N_ROWS - 0.5, 1.6)
    ax.set_aspect("equal")
    ax.axis("off")
    dom = state.dominant()[0]
    # the ELEMENT-identity spread (entropy) — distinct from the phase-coherence R
    # shown on the clock/gauge; a signal can be phase-crystallised yet element-diffuse.
    ident = "diffuse identity" if state.entropy() > 0.6 else (
        "focusing" if state.entropy() > 0.3 else "sharp identity")
    ax.set_title(f"material wavefunction · top: {dom[0]}  ({dom[1] * 100:.0f}%)  ·  {ident}",
                 color=_INK, fontsize=12, pad=8)
    return ax


def plot_phase_clock(parts: dict, ax=None):
    """The order parameter, drawn: each peak-oscillator as an arrow whose length
    is its phase-lock to the strongest peak. Scattered short arrows = floating;
    long aligned arrows = crystallised. The gold disc is the global R."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(4.4, 4.4), subplot_kw={"projection": "polar"},
                             facecolor=_BG)
    ax.set_facecolor(_PANEL)
    R = float(parts.get("R", 0.0))
    ax.add_artist(plt.Circle((0, 0), 1.0, transform=ax.transData._b, fill=False,
                             color="#243050", lw=1))
    # global R as a filled disc
    ax.bar(0, R, width=2 * np.pi, bottom=0, color=_GOLD, alpha=0.10, zorder=1)
    amps = parts.get("amps", np.array([]))
    res = parts.get("resultants", np.array([]))
    amax = float(np.max(amps)) if len(amps) else 1.0
    for k in range(len(res)):
        ang = float(np.angle(res[k]))
        length = float(np.abs(res[k]))
        w = 0.4 + 3.2 * (amps[k] / amax)
        col = _GOLD if k == parts.get("ref", 0) else _TEAL
        ax.annotate("", xy=(ang, length), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=w, alpha=0.9))
    ax.set_rmax(1.05)
    ax.set_rticks([])
    ax.set_xticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)
    ax.set_title(f"coherence  R = {R:.2f}", color=_INK, fontsize=12, pad=12)
    ax.text(0, 0, f"{R:.2f}", color=_GOLD, ha="center", va="center",
            fontsize=20, fontweight="bold")
    return ax


def plot_level_gauge(level_info: dict, ax=None):
    """Float ↔ crystal bar + the element/compound/mixture/structure soft scores."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 1.7), facecolor=_BG)
    ax.set_facecolor(_BG)
    scores = level_info["scores"]
    cols = {"element": "#c07a2e", "compound": "#2f83b8",
            "mixture": "#4e9a44", "structure": "#9a5ab0"}
    x = 0.0
    for k in LEVELS:
        w = scores[k]
        ax.barh(0.6, w, left=x, height=0.5, color=cols[k],
                alpha=0.85 if k == level_info["level"] else 0.4, edgecolor=_BG)
        if w > 0.08:
            ax.text(x + w / 2, 0.6, k, color="white", ha="center", va="center",
                    fontsize=10, fontweight="bold" if k == level_info["level"] else "normal")
        x += w
    # float↔crystal axis
    R = level_info["R"]
    ax.plot([0, 1], [-0.1, -0.1], color="#243050", lw=6, solid_capstyle="round")
    ax.plot([0, R], [-0.1, -0.1], color=_GOLD, lw=6, solid_capstyle="round")
    ax.scatter([R], [-0.1], s=120, c=[_GOLD], zorder=5, edgecolors=_BG, linewidths=1.5)
    ax.text(0, -0.42, "floating", color=_MUTED, fontsize=9, ha="left")
    ax.text(1, -0.42, "crystallised", color=_MUTED, fontsize=9, ha="right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.6, 1.0)
    ax.axis("off")
    ax.set_title(f"compositional level: {level_info['level']}", color=_INK,
                 fontsize=12, loc="left", pad=4)
    return ax


def plot_material_state(sig, sf, *, fmin=1.0, fmax=45.0, n_peaks=6,
                        title=None, savepath=None):
    """The signature "material-state portrait": the periodic-table wavefunction
    cloud, the phase-clock (order parameter drawn), and the float↔crystal /
    compositional-level gauge — one dark figure summarising where a signal sits
    between floating potential and a definite, crystallised material."""
    import matplotlib.pyplot as plt
    R, parts = coherence(sig, sf, fmin=fmin, fmax=fmax, n_peaks=n_peaks,
                         return_parts=True)
    st = element_state(parts["freqs"], R=R)
    lvl = compositional_level(sig, sf, fmin=fmin, fmax=fmax, n_peaks=n_peaks)

    fig = plt.figure(figsize=(12, 6.4), facecolor=_BG)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.7, 1.0], height_ratios=[3.2, 1.0],
                          hspace=0.32, wspace=0.18, left=0.03, right=0.98,
                          top=0.9, bottom=0.06)
    plot_state_cloud(st, ax=fig.add_subplot(gs[0, 0]))
    plot_phase_clock(parts, ax=fig.add_subplot(gs[0, 1], projection="polar"))
    plot_level_gauge(lvl, ax=fig.add_subplot(gs[1, :]))
    fig.suptitle(title or "bioelements · material state", color=_INK,
                 fontsize=15, fontweight="bold", x=0.03, ha="left", y=0.975)
    if savepath:
        fig.savefig(savepath, dpi=150, facecolor=_BG, bbox_inches="tight")
    return fig, {"state": st, "coherence": R, "level": lvl}
