"""Signal -> descriptors -> :class:`Fingerprint`.

Module type: Functions + dataclasses

A *descriptor* is one number (or one number per scale step) extracted from a
signal. A :class:`Fingerprint` is the vector of per-signal descriptors that
decides what a palette looks like.

Why a vector and not a scalar
-----------------------------
The prototype anchored hue on mean consonance alone. Two signals with similar
mean consonance therefore receive the same colour no matter how different they
actually are. Measured on synthetic signals: a *stretched harmonic series* and a
*two-cluster spectrum* -- nothing alike -- landed 0.62 deg apart in hue
(palette dE_OK 0.033, i.e. perceptually the same palette) purely because their
mean consonance matched at 7.75 vs 7.64. Both collisions found in testing are
resolved by adding a single extra dimension: those two separate immediately on
``harmonic_spread`` and ``spectral_spread``.

A scalar cannot separate signals. A vector can, and it degrades gracefully: two
signals collide only if they agree on *every* descriptor.

Adding your own
---------------
>>> from biotuner.biocolors.descriptors import descriptor
>>> @descriptor("peak_skew", kind="per_signal", range_hint=(-3, 3))
... def _peak_skew(ctx):
...     return float(scipy.stats.skew(ctx.peaks))
>>> palette_from_signal(peaks, amps, chroma_from="peak_skew")   # usable at once

No core edit required; the registry is the extension point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from biotuner.biotuner_utils import compute_peak_ratios
from biotuner.metrics import (
    dyad_similarity,
    higuchi_fd,
    ratios2harmsim,
    tuning_cons_matrix,
)

__all__ = [
    "SignalContext",
    "Descriptor",
    "Fingerprint",
    "descriptor",
    "fingerprint",
    "DESCRIPTORS",
    "PER_SIGNAL",
    "PER_STEP",
    "FINGERPRINT_FIELDS",
]

KINDS = ("per_signal", "per_step", "matrix", "curve")


# --------------------------------------------------------------------------- #
# Context
# --------------------------------------------------------------------------- #
@dataclass
class SignalContext:
    """Everything a descriptor may look at, computed once and shared.

    Built by :func:`make_context` from peaks + amplitudes, or from a bare
    tuning. Lazily derives the scale, the consonance matrix and the PSD-ish
    quantities so that a palette needing three descriptors does not recompute
    the consonance matrix three times.
    """

    peaks: np.ndarray
    amps: np.ndarray
    scale: np.ndarray
    fund: float
    sf: Optional[float] = None
    psd_freqs: Optional[np.ndarray] = None
    psd: Optional[np.ndarray] = None
    function: Callable = dyad_similarity
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def cons_matrix(self):
        """Full pairwise consonance matrix over the scale, symmetrised."""
        if "cons" not in self._cache:
            per_step, avg, full = tuning_cons_matrix(
                list(self.scale), self.function, ratio_type="all"
            )
            M = np.array(full, dtype=float)
            M = np.nan_to_num(M, nan=0.0)
            self._cache["cons"] = (
                np.asarray(per_step, float),
                float(avg),
                0.5 * (M + M.T),
            )
        return self._cache["cons"]

    @property
    def per_step_consonance(self):
        return self.cons_matrix[0]

    @property
    def mean_consonance(self):
        return self.cons_matrix[1]


#: How amplitudes returned by a peak extractor are scaled.
#:
#: biotuner's Welch-based extractors (``fixed``, ``adapt``,
#: ``harmonic_recurrence``, ...) all take ``10 * log10(psd)`` before picking
#: peaks (see ``peaks_extraction.extract_welch_peaks``), so their amplitudes are
#: **dB** and are routinely negative -- measured on real sleep EEG,
#: ``[14.96, 5.16, 10.44, 2.29, -4.77]``, with 26% of all values below zero.
#: ``EMD`` / ``HilbertHuang1D`` instead return linear envelope amplitudes.
AMP_SCALES = ("db", "linear", "auto")

#: Which amplitude scale each ``peaks_function`` actually returns.
#:
#: This is not cosmetic. biotuner's extractors disagree, and passing the wrong
#: scale silently produces an arbitrary palette. Measured on one 2-minute N2
#: segment:
#:
#: ==================  ==========  ==========================================
#: peaks_function      scale       example amps
#: ==================  ==========  ==========================================
#: fixed, adapt        db          ``[16.4, 9.4, 7.0, 2.8, -10.7]``
#: EMD, EEMD           db          ``[12.5, 7.4, 4.4]``      (Welch per IMF)
#: EIMC                db          ``[7.0, 5.9, 5.1, -5.4, -8.1]``
#: FOOOF, EMD_FOOOF    linear      ``[1.19, 0.96, 0.80, 0.50]``  (peak params)
#: HH1D_max            linear      ``[60702, 34306, 44714]``  (Hilbert env.)
#: cepstrum            linear      ``[2058, 1129, 721, 576]``  (cepstral mag.)
#: ==================  ==========  ==========================================
#:
#: Anything Welch-based inherits ``10*log10`` from
#: ``extract_welch_peaks``; anything reporting a fitted or measured amplitude
#: does not. :func:`amps_scale_for` looks the answer up so callers do not have
#: to remember, and :func:`~biotuner.biocolors.palettes.palette_from_biotuner`
#: applies it automatically.
AMPS_SCALE_BY_PEAKS_FUNCTION = {
    "fixed": "db",
    "adapt": "db",
    "EMD": "db",
    "EEMD": "db",
    "CEEMDAN": "db",
    "EIMC": "db",
    "harmonic_recurrence": "db",
    "bicoherence": "db",
    "PAC": "db",
    "FOOOF": "linear",
    "EMD_FOOOF": "linear",
    "HH1D_max": "linear",
    "cepstrum": "linear",
    "SMS": "linear",
}


def amps_scale_for(peaks_function, default="db"):
    """Amplitude scale a given ``peaks_function`` returns. See
    :data:`AMPS_SCALE_BY_PEAKS_FUNCTION`."""
    return AMPS_SCALE_BY_PEAKS_FUNCTION.get(str(peaks_function), default)


def _normalize_amps(amps, scale="db"):
    """Coerce amplitudes to positive linear weights that sum to 1.

    Negative weights are meaningless to Sethares dissonance (which reads them as
    loudness) and to any amplitude-driven colour channel, so dB must be
    *inverted*, not shifted.

    ``scale='db'`` (default, matching biotuner's Welch extractors) applies
    ``10 ** (a / 10)``. ``'linear'`` passes values through. ``'auto'`` guesses
    dB when any value is negative.

    Why ``'auto'`` is **not** the default, despite being the obvious choice: it
    decides per signal, from the values themselves. On this corpus 26% of epochs
    contain a negative amplitude and 74% do not -- so auto would invert the log
    for roughly a quarter of them and pass the rest through, putting two
    otherwise-identical spectra on different scales. Measured consequence before
    this was fixed: two N2 epochs with *identical* peaks
    ``[1, 3.5, 7.5, 14.2, 28.5]`` received anchor hues 122 deg apart. A scale
    convention has to be a property of the extractor, not of the sample.
    """
    a = np.asarray(amps, float)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    if a.size == 0:
        return a
    if scale == "auto":
        scale = "db" if a.min() < 0 else "linear"
    if scale == "db":
        # Clip absurd dB before exponentiating so one bad bin cannot swamp the
        # palette; -120 dB is 1e-12 of unit power, far below any real peak.
        a = 10.0 ** (np.clip(a, -120.0, 120.0) / 10.0)
    elif scale != "linear":
        raise ValueError(f"amps_scale must be one of {AMP_SCALES}, got {scale!r}")
    a = np.maximum(a, 0.0)
    total = a.sum()
    return a / total if total > 0 else np.ones_like(a) / len(a)


def make_context(peaks, amps=None, sf=None, psd=None, psd_freqs=None,
                 function=dyad_similarity, octave=2, amps_scale="db"):
    """Build a :class:`SignalContext` from spectral peaks and amplitudes.

    ``amps_scale`` must match what produced ``amps``; see :func:`_normalize_amps`.
    Default ``'db'`` matches biotuner's Welch-based peak extractors.
    """
    pk_raw = np.asarray(peaks, float)
    keep = np.isfinite(pk_raw) & (pk_raw > 0)
    pk = pk_raw[keep]
    if pk.size == 0:
        raise ValueError("no positive, finite peaks")

    if amps is None:
        am_raw = np.ones(pk.size)
        scale_used = "linear"
    else:
        am_raw = np.asarray(amps, float)
        scale_used = amps_scale
        if am_raw.shape != pk_raw.shape:
            # Amplitudes we cannot align to peaks are worse than none at all.
            am_raw = np.ones(pk.size)
            scale_used = "linear"
        else:
            am_raw = am_raw[keep]

    # Rectify *before* reordering so the dB inversion sees the raw values, then
    # sort peaks and amplitudes together.
    am = _normalize_amps(am_raw, scale=scale_used)
    order = np.argsort(pk)
    pk, am = pk[order], am[order]

    scale = np.asarray(pk / pk.min(), float)
    return SignalContext(
        peaks=pk, amps=am, scale=scale, fund=float(pk.min()),
        sf=sf, psd=psd, psd_freqs=psd_freqs, function=function,
    )


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass
class Descriptor:
    name: str
    kind: str
    fn: Callable
    range_hint: Optional[Tuple[float, float]] = None
    doc: str = ""


DESCRIPTORS: Dict[str, Descriptor] = {}


def descriptor(name, kind="per_signal", range_hint=None):
    """Register a descriptor under ``name``.

    ``kind`` is one of ``per_signal``, ``per_step``, ``matrix``, ``curve``.
    ``range_hint`` documents the expected range; it is *not* used for
    normalisation (:mod:`~biotuner.biocolors.calibration` does that from real
    data) but it is asserted in tests.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")

    def deco(fn):
        DESCRIPTORS[name] = Descriptor(name, kind, fn, range_hint, fn.__doc__ or "")
        return fn

    return deco


def compute(name, ctx):
    """Evaluate a registered descriptor against a context."""
    if name not in DESCRIPTORS:
        raise KeyError(
            f"unknown descriptor {name!r}. Registered: {sorted(DESCRIPTORS)}"
        )
    return DESCRIPTORS[name].fn(ctx)


# --------------------------------------------------------------------------- #
# Per-signal descriptors -- these are what separate one signal from another
# --------------------------------------------------------------------------- #
@descriptor("harmonicity", kind="per_signal", range_hint=(0, 100))
def _harmonicity(ctx):
    """Mean pairwise dyad similarity across the tuning. High = harmonic."""
    return float(ctx.mean_consonance)


@descriptor("harmonic_spread", kind="per_signal", range_hint=(0, 50))
def _harmonic_spread(ctx):
    """Std of per-step consonance. Separates 'evenly harmonic' from 'mixed'."""
    v = ctx.per_step_consonance
    return float(np.std(v)) if v.size else 0.0


@descriptor("spectral_flatness", kind="per_signal", range_hint=(0, 1))
def _spectral_flatness(ctx):
    """Wiener entropy of the peak amplitudes: tonal (0) vs noise-like (1)."""
    a = np.asarray(ctx.amps, float)
    a = a[a > 0]
    if a.size < 2:
        return 0.0
    gm = np.exp(np.mean(np.log(a)))
    am = np.mean(a)
    return float(gm / am) if am > 0 else 0.0


@descriptor("spectral_centroid", kind="per_signal", range_hint=(0, 60))
def _spectral_centroid(ctx):
    """Amplitude-weighted mean peak frequency, in Hz."""
    a = np.asarray(ctx.amps, float)
    return float(np.sum(ctx.peaks * a) / np.sum(a)) if np.sum(a) > 0 else float(ctx.fund)


@descriptor("spectral_spread", kind="per_signal", range_hint=(0, 30))
def _spectral_spread(ctx):
    """Amplitude-weighted std of peak frequency, in Hz."""
    a = np.asarray(ctx.amps, float)
    if np.sum(a) <= 0:
        return 0.0
    c = np.sum(ctx.peaks * a) / np.sum(a)
    return float(np.sqrt(np.sum(a * (ctx.peaks - c) ** 2) / np.sum(a)))


@descriptor("spectral_entropy", kind="per_signal", range_hint=(0, 1))
def _spectral_entropy(ctx):
    """Normalised Shannon entropy of the amplitude distribution."""
    a = np.asarray(ctx.amps, float)
    a = a[a > 0]
    if a.size < 2:
        return 0.0
    p = a / a.sum()
    return float(-np.sum(p * np.log2(p)) / np.log2(len(p)))


@descriptor("ratio_centroid", kind="per_signal", range_hint=(1, 4))
def _ratio_centroid(ctx):
    """Mean of the scale's ratios. The tuning analogue of spectral centroid.

    Deliberately reads ``ctx.scale`` and never ``ctx.amps``. For a tuning,
    ``spectral_centroid`` degenerates into exactly this quantity anyway (uniform
    weights make the amplitude-weighted mean the plain mean) -- but under a name
    that claims a spectrum the caller does not have. This one says what it is.
    """
    return float(np.mean(ctx.scale))


@descriptor("ratio_spread", kind="per_signal", range_hint=(0, 2))
def _ratio_spread(ctx):
    """Std of the scale's ratios: how spread out the steps are. See
    :func:`_ratio_centroid`."""
    return float(np.std(ctx.scale))


@descriptor("n_peaks", kind="per_signal", range_hint=(1, 12))
def _n_peaks(ctx):
    """Number of spectral peaks (or scale steps)."""
    return float(len(ctx.peaks))


@descriptor("fundamental", kind="per_signal", range_hint=(0, 6))
def _fundamental(ctx):
    """log2 of the lowest peak frequency."""
    return float(np.log2(max(ctx.fund, 1e-6)))


@descriptor("octave_span", kind="per_signal", range_hint=(0, 6))
def _octave_span(ctx):
    """log2(highest peak / lowest peak): how many octaves the spectrum covers."""
    return float(np.log2(ctx.peaks.max() / max(ctx.peaks.min(), 1e-9)))


@descriptor("complexity", kind="per_signal", range_hint=(1, 2))
def _complexity(ctx):
    """Higuchi fractal dimension of the amplitude sequence.

    Guarded on both sides. ``higuchi_fd`` takes ``log`` of a curve length, so a
    **constant** amplitude series makes it evaluate ``log(0)`` and return NaN
    *without raising* — a bare ``try/except`` does not catch it. Constant amps
    are not an edge case: every tuning reaches here that way, since a scale of
    ratios carries no amplitudes and :func:`make_context` weights the steps
    equally. Unguarded, that NaN propagates through the fingerprint into the
    hue anchor and every swatch renders black.
    """
    a = np.asarray(ctx.amps, float)
    if a.size < 5 or np.ptp(a) < 1e-12:
        return 1.0
    try:
        v = float(higuchi_fd(a, kmax=min(4, a.size // 2)))
    except Exception:
        return 1.0
    return v if np.isfinite(v) else 1.0


@descriptor("tenney", kind="per_signal", range_hint=(0, 20))
def _tenney(ctx):
    """Mean Tenney height of the scale ratios. High = complex ratios."""
    from fractions import Fraction
    vals = []
    for r in ctx.scale:
        fr = Fraction(float(r)).limit_denominator(1000)
        vals.append(np.log2(fr.numerator * fr.denominator))
    return float(np.mean(vals)) if vals else 0.0


# --------------------------------------------------------------------------- #
# Per-step descriptors -- these vary *within* a palette
# --------------------------------------------------------------------------- #
@descriptor("consonance", kind="per_step", range_hint=(0, 100))
def _consonance(ctx):
    """Averaged consonance of each step against all others."""
    return np.asarray(ctx.per_step_consonance, float)


@descriptor("harmsim", kind="per_step", range_hint=(0, 100))
def _harmsim(ctx):
    """Harmonic similarity per step (Gill & Purves 2009)."""
    return np.asarray(ratios2harmsim(list(ctx.scale)), float)


@descriptor("amplitude", kind="per_step", range_hint=(0, 1))
def _amplitude(ctx):
    """Per-step amplitude (rectified and normalised; see _normalize_amps)."""
    return np.asarray(ctx.amps, float)


@descriptor("tenney_step", kind="per_step", range_hint=(0, 20))
def _tenney_step(ctx):
    """Per-step Tenney height, inverted so high = simple ratio."""
    from fractions import Fraction
    th = []
    for r in ctx.scale:
        fr = Fraction(float(r)).limit_denominator(1000)
        th.append(np.log2(fr.numerator * fr.denominator))
    th = np.asarray(th, float)
    return th.max() - th if th.size else th


@descriptor("pitch", kind="per_step", range_hint=(0, 6))
def _pitch(ctx):
    """log2 of each step's ratio: position within the octave."""
    return np.log2(np.asarray(ctx.scale, float))


@descriptor("uniform", kind="per_step", range_hint=(0, 1))
def _uniform(ctx):
    """Constant. Use to pin a channel."""
    return np.ones(len(ctx.scale))


PER_SIGNAL = tuple(n for n, d in DESCRIPTORS.items() if d.kind == "per_signal")
PER_STEP = tuple(n for n, d in DESCRIPTORS.items() if d.kind == "per_step")

#: The descriptors that make up a :class:`Fingerprint` for a **spectrum**, in a
#: fixed order. Fixed because the calibration's PCA projection is fit against
#: this order and stored; reordering would silently invalidate every shipped
#: calibration.
FINGERPRINT_FIELDS = (
    "harmonicity",
    "harmonic_spread",
    "spectral_flatness",
    "spectral_centroid",
    "spectral_spread",
    "spectral_entropy",
    "n_peaks",
    "fundamental",
    "octave_span",
    "complexity",
    "tenney",
)

#: The descriptors that make up a :class:`Fingerprint` for a **tuning**.
#:
#: A tuning is a set of ratios with no amplitudes, so :func:`make_context`
#: weights every step equally and every amplitude-derived descriptor collapses
#: to a constant: ``spectral_flatness`` and ``spectral_entropy`` are 1 for any
#: tuning ever, ``complexity`` is Higuchi FD of a flat series, and
#: ``fundamental`` is 0 for any scale starting at 1. Four of the eleven
#: :data:`FINGERPRINT_FIELDS` therefore carry no information here.
#:
#: They were harmless -- a constant is annihilated twice, once by the PCA's mean
#: subtraction and again by the informativeness weighting, and dropping all four
#: moves the anchor by <= 0.11 deg. But they were also actively misleading: a
#: tuning's fingerprint printed ``spectral_flatness=1, complexity=1`` as though
#: those meant something, and ``complexity`` reached ``higuchi_fd`` on a constant
#: series, which is the ``log(0)`` -> NaN path that once rendered every bare
#: tuning black.
#:
#: ``spectral_centroid`` / ``spectral_spread`` are replaced by
#: :func:`_ratio_centroid` / :func:`_ratio_spread`, which compute the same
#: numbers straight from the ratios without pretending a spectrum exists.
TUNING_FIELDS = (
    "harmonicity",
    "harmonic_spread",
    "ratio_centroid",
    "ratio_spread",
    "n_peaks",
    "octave_span",
    "tenney",
)


# --------------------------------------------------------------------------- #
# Fingerprint
# --------------------------------------------------------------------------- #
@dataclass
class Fingerprint:
    """The per-signal descriptor vector that decides a palette's identity.

    ``values`` maps descriptor name -> raw value. :meth:`vector` returns them in
    :data:`FINGERPRINT_FIELDS` order; :meth:`normalized` percentile-maps each to
    ``[0, 1]`` against a :class:`~biotuner.biocolors.calibration.Calibration`.
    """

    values: Dict[str, float]

    def vector(self, fields=FINGERPRINT_FIELDS):
        return np.array([float(self.values.get(f, 0.0)) for f in fields], float)

    def normalized(self, calibration, fields=FINGERPRINT_FIELDS):
        """Percentile-rank each descriptor against the calibration -> ``[0, 1]``."""
        return np.array(
            [calibration.normalize(f, self.values.get(f, 0.0)) for f in fields],
            float,
        )

    def __getitem__(self, k):
        return self.values[k]

    def __repr__(self):
        body = ", ".join(f"{k}={v:.3g}" for k, v in self.values.items())
        return f"Fingerprint({body})"


def fingerprint(ctx_or_peaks, amps=None, fields=FINGERPRINT_FIELDS, **kw):
    """Compute a :class:`Fingerprint` from a context, or from peaks + amps.

    Non-finite descriptor values are replaced with 0.0 rather than propagated.
    A single NaN anywhere in the vector otherwise poisons the PCA projection,
    which makes the hue anchor NaN, which renders every swatch black — a
    failure mode that looks like a colour bug but is a metric bug. Descriptors
    should guard themselves; this is the backstop.
    """
    ctx = (
        ctx_or_peaks
        if isinstance(ctx_or_peaks, SignalContext)
        else make_context(ctx_or_peaks, amps, **kw)
    )
    out = {}
    for f in fields:
        try:
            v = float(compute(f, ctx))
        except Exception:
            v = 0.0
        out[f] = v if np.isfinite(v) else 0.0
    return Fingerprint(out)
