"""Fingerprint + per-step metrics -> :class:`ColorSpec` (L, C, h).

Module type: Functions + dataclasses

This is the layer that decides what a palette *looks like*. Three ideas carry it.

1. Hue is anchored by a **direction in fingerprint space**, not a scalar
   ---------------------------------------------------------------------
   ``hue = atan2(pc2, pc1)`` of the calibrated fingerprint's PCA projection.
   Two signals collide only if they agree on every descriptor, instead of
   merely sharing one average. The distance from the corpus centroid drives
   chroma, so an unusual signal reads vivid and a typical one reads muted --
   for free, from the same projection.

2. Chroma is **cusp-relative**, never absolute
   -------------------------------------------
   A channel asks for a *fraction* of the chroma sRGB actually allows at that
   (L, h). The ceiling varies 5x around the wheel (0.048 to 0.296 measured), so
   a fixed range is simultaneously over budget at some hues and wasteful at
   others. Fractions keep palettes in gamut *and* as vivid as the hue permits.

3. Palette character is a **continuous 2-D temperament**, not a fixed rectangle
   ---------------------------------------------------------------------------
   ``luminosity`` (where lightness sits) and ``vividness`` (what fraction of the
   available chroma is used) span a plane. Named presets -- earthy, pastel,
   vivid, deep, balanced -- are just points in it, and ``temperament="auto"``
   lets the signal choose its own point. That is what makes one recording come
   out earthy and another pastel *from the same code path*, rather than every
   palette being a differently-hued version of the same look.

Lightness is specified in **Oklrab L_r** (perceptual lightness), then converted
to OKLab L. OKLab's raw L is not uniform against mid-grey, so an evenly spaced
L range produces unevenly spaced swatches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from biotuner.biocolors.color.gamut import max_chroma
from biotuner.biocolors.color.spaces import toe_inv
from biotuner.biocolors.descriptors import compute

__all__ = [
    "ColorSpec",
    "ColorAxes",
    "Temperament",
    "TEMPERAMENTS",
    "mapping",
    "MAPPINGS",
    "COLORSPACES",
    "register_colorspace",
    "auto_temperament",
]


# --------------------------------------------------------------------------- #
# Temperament: the character of a palette
# --------------------------------------------------------------------------- #
@dataclass
class Temperament:
    """Where a palette sits in (luminosity, vividness) and how far it spreads.

    Parameters
    ----------
    L_center, L_spread : float
        Perceptual lightness (Oklrab L_r) centre and half-range across steps.
    C_frac_center, C_frac_spread : float
        Fraction of the *locally available* chroma; 1.0 means "ride the gamut
        boundary". Spread lets consonant steps sit more saturated than dissonant
        ones (or the reverse).
    arc : float
        Degrees of hue swept across the scale. Small = a tight family; large =
        a broad spread. This is internal contrast, not signal separation --
        separation is the anchor's job.
    hue_bias : float
        Degrees added to the anchor. Presets use it to lean warm or cool.
    hue_pull, hue_pull_to : float, float
        Pull the anchor a fraction ``hue_pull`` of the way toward a fixed hue
        family. This deliberately **trades separation for character**: pigment
        words like "earthy" name a region of hue (ochre, olive, clay), not just
        a lightness and chroma regime, so ``earthy`` applied to a magenta-
        anchored signal without a pull yields dusty mauve rather than clay.

        Naming a temperament is a request for a *look*, and this is the knob
        that honours it. It is 0 for ``auto`` -- the default path keeps hue
        entirely signal-derived, so diversity is never silently spent.
    """

    name: str = "custom"
    L_center: float = 0.62
    L_spread: float = 0.14
    C_frac_center: float = 0.55
    C_frac_spread: float = 0.25
    arc: float = 90.0
    hue_bias: float = 0.0
    hue_pull: float = 0.0
    hue_pull_to: float = 60.0

    def blend(self, other, w):
        """Linear blend toward ``other`` by weight ``w``."""
        w = float(np.clip(w, 0.0, 1.0))
        return Temperament(
            name=f"{self.name}~{other.name}",
            L_center=(1 - w) * self.L_center + w * other.L_center,
            L_spread=(1 - w) * self.L_spread + w * other.L_spread,
            C_frac_center=(1 - w) * self.C_frac_center + w * other.C_frac_center,
            C_frac_spread=(1 - w) * self.C_frac_spread + w * other.C_frac_spread,
            arc=(1 - w) * self.arc + w * other.arc,
            hue_bias=(1 - w) * self.hue_bias + w * other.hue_bias,
            hue_pull=(1 - w) * self.hue_pull + w * other.hue_pull,
            hue_pull_to=_circ_blend(self.hue_pull_to, other.hue_pull_to, w),
        )


def _circ_blend(a, b, w):
    """Blend two angles the short way round the circle."""
    d = (b - a + 180.0) % 360.0 - 180.0
    return (a + w * d) % 360.0


def _pull_hue(anchor, target, amount):
    """Move ``anchor`` a fraction ``amount`` toward ``target`` along the short arc."""
    if amount <= 0:
        return anchor % 360.0
    return _circ_blend(anchor, target, float(np.clip(amount, 0.0, 1.0)))


#: Named palette characters. Points in the (luminosity, vividness) plane.
TEMPERAMENTS: Dict[str, Temperament] = {
    # Muted, mid-dark, pulled toward ochre/olive. Clay, umber, moss.
    "earthy": Temperament("earthy", L_center=0.55, L_spread=0.15,
                          C_frac_center=0.34, C_frac_spread=0.18,
                          arc=70.0, hue_bias=0.0,
                          hue_pull=0.55, hue_pull_to=62.0),
    # High lightness, low chroma. Chalky, airy.
    "pastel": Temperament("pastel", L_center=0.84, L_spread=0.10,
                          C_frac_center=0.46, C_frac_spread=0.20,
                          arc=120.0, hue_bias=0.0),
    # Rides the gamut boundary at mid lightness. Maximum punch.
    "vivid": Temperament("vivid", L_center=0.63, L_spread=0.11,
                         C_frac_center=0.93, C_frac_spread=0.10,
                         arc=90.0, hue_bias=0.0),
    # Dark and saturated. Jewel tones.
    "deep": Temperament("deep", L_center=0.43, L_spread=0.14,
                        C_frac_center=0.82, C_frac_spread=0.16,
                        arc=75.0, hue_bias=0.0),
    # Wide lightness range, moderate chroma. The safe all-rounder.
    "balanced": Temperament("balanced", L_center=0.63, L_spread=0.21,
                            C_frac_center=0.58, C_frac_spread=0.25,
                            arc=105.0, hue_bias=0.0),
    # Nearly achromatic; lets lightness carry the structure.
    "ashen": Temperament("ashen", L_center=0.60, L_spread=0.26,
                         C_frac_center=0.12, C_frac_spread=0.08,
                         arc=45.0, hue_bias=0.0),
    # Very wide hue arc: a full spectrum reading of the scale.
    "prismatic": Temperament("prismatic", L_center=0.66, L_spread=0.13,
                             C_frac_center=0.78, C_frac_spread=0.16,
                             arc=220.0, hue_bias=0.0),
    # Wide arc at moderate chroma: one palette deliberately spanning warm THROUGH
    # cool. Where 'auto' keeps a coherent single-temperature family (arc
    # 45-175deg from harmonic_spread), 'aurora' opts into the balanced spread the
    # user asks for -- red/orange on one side, teal/blue on the other, softer than
    # 'prismatic'. The signal still sets the anchor and the per-step ordering;
    # only the sweep is forced wide.
    "aurora": Temperament("aurora", L_center=0.64, L_spread=0.20,
                          C_frac_center=0.60, C_frac_spread=0.20,
                          arc=250.0, hue_bias=0.0),
    # Cool, low-chroma, high-lightness: the "clinical figure" look.
    "glacial": Temperament("glacial", L_center=0.78, L_spread=0.18,
                           C_frac_center=0.30, C_frac_spread=0.14,
                           arc=80.0, hue_pull=0.5, hue_pull_to=225.0),
    # Warm, dark, saturated: candle/amber.
    "ember": Temperament("ember", L_center=0.50, L_spread=0.17,
                         C_frac_center=0.85, C_frac_spread=0.14,
                         arc=60.0, hue_pull=0.6, hue_pull_to=35.0),
}


def auto_temperament(fp, cal):
    """Let the signal choose its own character.

    The mapping, and the reasoning behind each axis:

    - **vividness <- tonality.** ``spectral_flatness`` says how noise-like the
      spectrum is. A tonal signal (clear peaks, low flatness) earns saturated
      colour; a noisy one goes muted and earthy. This is the axis that makes
      broadband recordings look like clay and peaky ones look like glass.
    - **luminosity <- harmonicity.** Harmonic signals sit light and open;
      inharmonic ones sit dark and dense.
    - **arc <- harmonic_spread.** A signal whose steps differ a lot in
      consonance gets a wider hue sweep, i.e. more internal contrast. A
      uniform signal stays a tight family.

    Returns a :class:`Temperament` blended between the named presets, so
    ``auto`` output always lands somewhere on the same plane the presets live
    on -- never off in an unreachable corner.

    On a **tuning** fingerprint (:data:`~biotuner.biocolors.descriptors.TUNING_FIELDS`)
    there is no ``spectral_flatness``, because there are no amplitudes to be
    flat or peaky. ``tonality`` then falls back to 0.5 and the vividness axis
    sits mid-scale, which is the honest answer: a bare scale of ratios has no
    tonality to read. Luminosity and arc still respond, via ``harmonicity`` and
    ``harmonic_spread``.
    """
    n = {f: cal.normalize(f, fp.values.get(f, 0.0)) for f in fp.values}
    tonality = 1.0 - n.get("spectral_flatness", 0.5)   # 1 = tonal; 0.5 if absent
    harmonic = n.get("harmonicity", 0.5)
    spread = n.get("harmonic_spread", 0.5)

    # vividness axis: earthy (muted) <-> vivid
    base = TEMPERAMENTS["earthy"].blend(TEMPERAMENTS["vivid"], tonality)
    # luminosity axis: deep (dark) <-> pastel (light)
    lum = TEMPERAMENTS["deep"].blend(TEMPERAMENTS["pastel"], harmonic)
    t = base.blend(lum, 0.4)
    t.arc = 45.0 + 130.0 * spread
    # No hue pull on auto: hue must stay signal-derived or separation is spent
    # without the caller asking. Named temperaments may pull; auto never does.
    t.hue_pull = 0.0
    t.name = "auto"
    return t


# --------------------------------------------------------------------------- #
# ColorSpec
# --------------------------------------------------------------------------- #
@dataclass
class ColorSpec:
    """OKLCh channels for a palette, plus why each value is what it is.

    ``provenance`` records, per channel, the descriptor and the numbers behind
    it, so :meth:`~biotuner.biocolors.palettes.Palette.explain` can answer "why
    is step 3 that colour?" without re-deriving anything.
    """

    L: np.ndarray
    C: np.ndarray
    h: np.ndarray
    anchor_hue: float = 0.0
    temperament: Optional[Temperament] = None
    provenance: dict = field(default_factory=dict)

    @property
    def lch(self):
        return np.stack([self.L, self.C, self.h], axis=-1)

    def __len__(self):
        return len(self.L)


MAPPINGS: Dict[str, Callable] = {}


def mapping(name):
    """Register a fingerprint+steps -> :class:`ColorSpec` mapping."""

    def deco(fn):
        MAPPINGS[name] = fn
        return fn

    return deco


def _rank01(x):
    """Rank-transform to [0, 1]. Robust to outliers and to flat input.

    Ranking rather than min-max scaling means one extreme step cannot squash
    every other step into a corner of the range -- the usual cause of a palette
    where seven swatches look identical and one is an outlier.
    """
    x = np.asarray(x, float)
    x = np.nan_to_num(x, nan=float(np.nanmedian(x)) if np.isfinite(x).any() else 0.0)
    if len(x) == 1:
        return np.array([0.5])
    if np.ptp(x) < 1e-12:
        return np.full(len(x), 0.5)
    order = np.argsort(np.argsort(x))
    return order / (len(x) - 1)


#: Fallback chain per channel, used when the requested metric is constant.
#:
#: A constant channel is not a hypothetical. ``amplitude`` is the default for
#: lightness, but a *tuning* carries no amplitudes -- :func:`make_context`
#: weights its steps equally -- so ``_rank01`` returns all 0.5 and every swatch
#: gets the identical lightness. Measured before this existed: an 18-element
#: ``extended_peaks_ratios`` palette had lightness range 0.50-0.50 and a minimum
#: pairwise dE of **0.005**, i.e. eighteen swatches of the same brown.
#:
#: Silently rendering a dead channel is worse than substituting a live one, so
#: the mapping falls back to the first metric that actually varies and records
#: the substitution in the provenance.
_FALLBACKS = {
    "amplitude": ("tenney_step", "consonance", "pitch"),
    "consonance": ("harmsim", "tenney_step", "pitch"),
    "harmsim": ("consonance", "tenney_step", "pitch"),
    "tenney_step": ("pitch", "consonance"),
    "pitch": ("tenney_step",),
}


def _channel(ctx, name):
    """Rank-transformed per-step channel, substituting if ``name`` is constant.

    Returns ``(t, used_name)``.
    """
    v = np.asarray(compute(name, ctx), float)
    if len(v) > 1 and np.ptp(np.nan_to_num(v)) > 1e-12:
        return _rank01(v), name
    for alt in _FALLBACKS.get(name, ()):
        try:
            w = np.asarray(compute(alt, ctx), float)
        except Exception:
            continue
        if len(w) > 1 and np.ptp(np.nan_to_num(w)) > 1e-12:
            return _rank01(w), alt
    return _rank01(v), name


#: Minimum hue separation between adjacent swatches, in degrees.
#:
#: The arc a temperament asks for describes a *character* (a tight family vs a
#: broad sweep) and knows nothing about how many swatches must fit inside it.
#: That breaks down as palettes grow: 18 elements in a 45 deg arc land 2.5 deg
#: apart, which at palette chroma is far below the perceptual threshold. The arc
#: is therefore widened, when needed, so that adjacent steps stay this far apart
#: -- never narrowed, so a temperament's character is preserved wherever it is
#: already sufficient.
MIN_HUE_STEP = 11.0
MAX_ARC = 330.0


def _fit_arc(arc, n):
    """Widen ``arc`` so ``n`` swatches keep :data:`MIN_HUE_STEP` between them."""
    if n < 2:
        return arc
    return float(min(max(arc, MIN_HUE_STEP * (n - 1)), MAX_ARC))


# --------------------------------------------------------------------------- #
# The default mapping
# --------------------------------------------------------------------------- #
@mapping("anchored")
def anchored(ctx, fp, cal, temperament="auto",
             hue_from="pitch", light_from="amplitude", chroma_from="consonance",
             anchor="fingerprint", hue_rotate=0.0, reverse_arc=False, **kw):
    """Fingerprint-anchored palette. The default.

    Hue anchor comes from the fingerprint's PCA direction; each step is placed
    along an arc from that anchor by ``hue_from``. Chroma is a cusp-relative
    fraction driven by ``chroma_from``; lightness by ``light_from``. The
    temperament sets the regime all three live in.
    """
    t = (temperament if isinstance(temperament, Temperament)
         else auto_temperament(fp, cal) if temperament == "auto"
         else TEMPERAMENTS[temperament])

    n = len(ctx.scale)

    # -- anchor ------------------------------------------------------------ #
    if anchor == "fingerprint":
        pc = cal.project(fp)
        anchor_hue = float(np.degrees(np.arctan2(pc[1], pc[0])) % 360.0)
        # Percentile-rank the radius: raw distances are corpus-scale dependent,
        # so a bare magnitude would make "unusual = vivid" mean different things
        # for different calibrations.
        radius = cal.normalize("__radius__", float(np.hypot(pc[0], pc[1])))
    elif anchor == "harmonicity":
        anchor_hue = 360.0 * cal.normalize("harmonicity", fp.values["harmonicity"])
        radius = 0.5
    else:
        anchor_hue = float(anchor)
        radius = 0.5
    anchor_hue = _pull_hue(anchor_hue, t.hue_pull_to, t.hue_pull)
    anchor_hue = (anchor_hue + t.hue_bias + hue_rotate) % 360.0

    # -- hue: sweep the arc, widened if needed to fit n swatches ----------- #
    hue_t, hue_used = _channel(ctx, hue_from)
    if reverse_arc:
        hue_t = 1.0 - hue_t
    arc = _fit_arc(t.arc, n)
    h = (anchor_hue + (hue_t - 0.5) * arc) % 360.0

    # -- lightness (in Oklrab, then to OKLab) ------------------------------ #
    light_t, light_used = _channel(ctx, light_from)
    # Spread lightness further as the palette grows, for the same reason the arc
    # widens: hue alone cannot separate 20 swatches inside one family.
    L_spread = min(t.L_spread * max(1.0, (n / 6.0) ** 0.5), 0.34)
    Lr = t.L_center + (light_t - 0.5) * 2.0 * L_spread
    Lr = np.clip(Lr, 0.06, 0.98)
    L = toe_inv(Lr)

    # -- chroma: a fraction of what the gamut allows here ------------------ #
    chroma_t, chroma_used = _channel(ctx, chroma_from)
    frac = t.C_frac_center + (chroma_t - 0.5) * 2.0 * t.C_frac_spread
    # Unusual signals (far from the corpus centroid) ride closer to the cusp.
    frac = frac * (0.80 + 0.40 * np.clip(radius, 0.0, 1.0))
    frac = np.clip(frac, 0.03, 1.0)
    ceiling = max_chroma(L, h)
    C = frac * ceiling

    return ColorSpec(
        L=L, C=C, h=h,
        anchor_hue=anchor_hue,
        temperament=t,
        provenance={
            "anchor": {"mode": anchor, "hue": anchor_hue, "radius": radius,
                       "pca_explained": float(np.sum(cal.pca_explained))},
            "hue": {"from": hue_used, "requested": hue_from, "t": hue_t,
                    "arc": arc, "arc_requested": t.arc},
            "light": {"from": light_used, "requested": light_from, "t": light_t,
                      "Lr": Lr, "center": t.L_center, "spread": L_spread,
                      "spread_requested": t.L_spread},
            "chroma": {"from": chroma_used, "requested": chroma_from, "t": chroma_t,
                       "frac": frac, "ceiling": ceiling},
            "temperament": t.name,
            "calibration": cal.name,
            "n": n,
        },
    )


@mapping("spectral")
def spectral(ctx, fp, cal, temperament="balanced",
             light_from="amplitude", chroma_from="consonance",
             spectral_method="cie1931", fold_mode="fold", **kw):
    """Hue from each step's *physical* folded wavelength.

    The original biocolors idea, kept first-class: hue is the real spectral
    colour of the frequency, folded into 380-750 nm. Nothing here reads a
    fingerprint, a calibration or a corpus -- a given Hz yields the same hue in
    any dataset, forever, which makes this path immune to every calibration
    failure the ``anchored`` mapping can suffer.

    What it costs is separation. Measured over ten varied EEG chunks: mean
    pairwise palette dE 0.161 vs 0.178 for ``anchored`` -- comparable -- but hue
    occupancy 6% vs 25% and min pairwise dE 0.010. Every EEG epoch carries a
    ~1 Hz delta peak, which folds to 532 nm, so every spectral palette opens
    green and runs to coral: real variation inside one family rather than across
    the wheel. Raising peak variability does not help; across extractors the
    figure stays ~0.16 even when peak spread rises 10x (FOOOF, spread 7.03 Hz).

    **Do not combine this with harmonically extended peaks.** The fold maps an
    octave onto the same wavelength -- that is octave equivalence, and it is the
    point. But ``peaks_extension(method='harmonic_fit')`` multiplies the
    fundamental by integers, and 2x / 4x / 8x *are* octaves, so extension injects
    exact hue duplicates. Measured on one N2 epoch: 11 extended peaks collapsed
    onto **6 distinct wavelengths**, with 7.25 / 14.5 / 14.5 / 29 / 58 Hz all
    landing on 587.6 nm. Extension adds octaves; the fold removes them. Use
    ``level='peaks'`` with this mapping, or use ``anchored`` on the extended set.
    """
    from biotuner.biocolors.color.spaces import srgb_to_oklch
    from biotuner.biocolors.color.spectral import audible_to_nm, wavelength_to_srgb

    t = (temperament if isinstance(temperament, Temperament)
         else auto_temperament(fp, cal) if temperament == "auto"
         else TEMPERAMENTS[temperament])

    freqs = np.asarray(ctx.scale, float) * ctx.fund
    nm, _ = audible_to_nm(freqs, mode=fold_mode)
    rgb = wavelength_to_srgb(nm, method=spectral_method)
    h = srgb_to_oklch(rgb)[..., 2]

    n = len(ctx.scale)
    light_t, light_used = _channel(ctx, light_from)
    L_spread = min(t.L_spread * max(1.0, (n / 6.0) ** 0.5), 0.34)
    Lr = np.clip(t.L_center + (light_t - 0.5) * 2.0 * L_spread, 0.06, 0.98)
    L = toe_inv(Lr)

    chroma_t, chroma_used = _channel(ctx, chroma_from)
    frac = np.clip(t.C_frac_center + (chroma_t - 0.5) * 2.0 * t.C_frac_spread, 0.03, 1.0)
    C = frac * max_chroma(L, h)

    return ColorSpec(
        L=L, C=C, h=h, anchor_hue=float(h[0]) if len(h) else 0.0, temperament=t,
        provenance={
            "hue": {"from": "wavelength", "nm": nm, "method": spectral_method,
                    "fold": fold_mode},
            "light": {"from": light_used, "requested": light_from, "t": light_t},
            "chroma": {"from": chroma_used, "requested": chroma_from, "t": chroma_t,
                       "frac": frac},
            "temperament": t.name, "calibration": cal.name, "n": n,
        },
    )


@mapping("mds")
def mds(ctx, fp, cal, temperament="balanced",
        light_from="amplitude", chroma_from="consonance", hue_rotate=0.0, **kw):
    """Hue from a classical-MDS embedding of the consonance matrix.

    Consonant steps end up near each other on the hue circle, so hue encodes
    *harmonic relatedness* rather than pitch. Inherited from the prototype --
    the one idea in it that was both novel and sound.
    """
    t = (temperament if isinstance(temperament, Temperament)
         else auto_temperament(fp, cal) if temperament == "auto"
         else TEMPERAMENTS[temperament])

    _, _, M = ctx.cons_matrix
    D = M.max() - M
    np.fill_diagonal(D, 0.0)
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    vals, vecs = np.linalg.eigh(B)
    top = np.argsort(vals)[::-1][:2]
    coords = vecs[:, top] * np.sqrt(np.maximum(vals[top], 0.0))
    h = (np.degrees(np.arctan2(coords[:, 1], coords[:, 0])) + hue_rotate) % 360.0

    light_t, light_used = _channel(ctx, light_from)
    L_spread = min(t.L_spread * max(1.0, (n / 6.0) ** 0.5), 0.34)
    Lr = np.clip(t.L_center + (light_t - 0.5) * 2.0 * L_spread, 0.06, 0.98)
    L = toe_inv(Lr)

    chroma_t, chroma_used = _channel(ctx, chroma_from)
    frac = np.clip(t.C_frac_center + (chroma_t - 0.5) * 2.0 * t.C_frac_spread, 0.03, 1.0)
    C = frac * max_chroma(L, h)

    return ColorSpec(
        L=L, C=C, h=h, anchor_hue=float(np.mean(h)) if len(h) else 0.0,
        temperament=t,
        provenance={
            "hue": {"from": "consonance_mds", "coords": coords},
            "light": {"from": light_used, "requested": light_from, "t": light_t},
            "chroma": {"from": chroma_used, "requested": chroma_from, "t": chroma_t,
                       "frac": frac},
            "temperament": t.name, "calibration": cal.name, "n": n,
        },
    )


# --------------------------------------------------------------------------- #
# Declarative colourspaces
# --------------------------------------------------------------------------- #
# A colourspace is a mapping whose three axes are *declared*: you name one
# descriptor for lightness, one for chroma, one for hue. Where `anchored` derives
# hue from an arbitrary PCA rotation with no absolute meaning, a colourspace's
# hue IS the named descriptor -- "hue = consonance" is a statable fact, monotone
# and identical across every signal. The trade is separation power: one
# descriptor per axis carries less than the full 11-D fingerprint, so a
# colourspace reads well but tells signals apart less strongly than `anchored`.
#
# Registered exactly like any other method, so palette_from_signal(..., method=
# "consonance") works with no new concept. A per-step descriptor makes an axis
# vary WITHIN the palette; a per-signal descriptor makes it constant across the
# palette (varying between signals instead).
from biotuner.biocolors.descriptors import DESCRIPTORS as _DESCRIPTORS  # noqa: E402


@dataclass
class ColorAxes:
    """A declarative colourspace: (descriptor, range) for each of L, C, h.

    L / C ranges are Oklrab L_r and cusp-fraction; h is degrees. Registered as a
    method via :func:`register_colorspace`.
    """
    name: str
    L: tuple           # (descriptor_name, (lo, hi))  -- Oklrab lightness
    C: tuple           # (descriptor_name, (lo, hi))  -- fraction of local cusp
    h: tuple           # (descriptor_name, (lo, hi))  -- degrees
    doc: str = ""


def _axis(spec, ctx, cal):
    """Evaluate one declared axis to a per-step array in [lo, hi]."""
    name, (lo, hi) = spec
    v = np.asarray(compute(name, ctx), float)
    n = len(ctx.scale)
    kind = _DESCRIPTORS[name].kind if name in _DESCRIPTORS else "per_step"
    if kind == "per_signal":
        t = cal.normalize(name, float(v)) if cal is not None else 0.5
        t = np.full(n, float(t))
    else:
        t = _rank01(v)
    return lo + t * (hi - lo)


def register_colorspace(axes: ColorAxes):
    """Compile a :class:`ColorAxes` into a registered mapping named ``axes.name``.

    The temperament is ignored on purpose: a colourspace's ranges *are* its
    character, so its look is fixed by the axis declaration, not by 'auto'.
    """

    @mapping(axes.name)
    def _colorspace_mapping(ctx, fp, cal, temperament="auto", **kw):
        n = len(ctx.scale)
        Lr = np.clip(_axis(axes.L, ctx, cal), 0.05, 0.98)
        L = toe_inv(Lr)
        h = _axis(axes.h, ctx, cal) % 360.0
        frac = np.clip(_axis(axes.C, ctx, cal), 0.03, 1.0)
        ceiling = max_chroma(L, h)
        C = frac * ceiling
        return ColorSpec(
            L=L, C=C, h=h,
            anchor_hue=float(np.mean(h)) if len(h) else 0.0,
            temperament=None,
            provenance={
                "colorspace": axes.name, "doc": axes.doc,
                "hue": {"from": axes.h[0], "range": axes.h[1]},
                "light": {"from": axes.L[0], "range": axes.L[1]},
                "chroma": {"from": axes.C[0], "range": axes.C[1],
                           "frac": frac, "ceiling": ceiling},
                "temperament": axes.name, "calibration": cal.name, "n": n,
            },
        )

    return _colorspace_mapping


#: Built-in declarative colourspaces. Each names its three axes.
COLORSPACES: Dict[str, ColorAxes] = {
    "tonotopic": ColorAxes(
        "tonotopic",
        L=("consonance", (0.45, 0.82)), C=("amplitude", (0.30, 0.95)),
        h=("pitch", (0.0, 330.0)),
        doc="h = pitch (low->red, high->violet) | C = loudness | L = consonance"),
    "consonance": ColorAxes(
        "consonance",
        L=("amplitude", (0.42, 0.80)), C=("harmsim", (0.35, 0.95)),
        h=("consonance", (250.0, 30.0)),
        doc="h = consonance (dissonant->blue, consonant->red) | C = harmsim | L = loudness"),
    "harmonic": ColorAxes(
        "harmonic",
        L=("amplitude", (0.40, 0.82)), C=("consonance", (0.30, 0.95)),
        h=("harmsim", (140.0, 40.0)),
        doc="h = harmonic similarity | C = consonance | L = loudness"),
    "tenney": ColorAxes(
        "tenney",
        L=("amplitude", (0.45, 0.85)), C=("tenney_step", (0.30, 0.92)),
        h=("tenney_step", (30.0, 300.0)),
        doc="h = ratio simplicity | C = simplicity | L = loudness"),
}

for _cs in COLORSPACES.values():
    register_colorspace(_cs)


@mapping("derived")
def derived(ctx, fp, cal, temperament="auto",
            features=("consonance", "harmsim", "amplitude", "pitch", "tenney_step"),
            **kw):
    """A colourspace whose axes are LEARNED, not declared.

    Builds an (n_steps x n_features) matrix from the per-step descriptors, takes
    its principal components, and maps the top three to (lightness, hue, chroma).
    The axes are whatever directions of variation the signal's own steps
    exhibit -- a colourspace derived from the data rather than chosen. The
    absolute hue is therefore not interpretable (like ``anchored``), but the
    within-palette structure reflects the step covariance.
    """
    n = len(ctx.scale)
    X = np.stack([_rank01(np.asarray(compute(f, ctx), float)) for f in features], axis=-1)
    X = X - X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    pcs = X @ vecs[:, :3]

    def _nz(v):
        return (v - v.min()) / (np.ptp(v) + 1e-9)

    Lr = np.clip(0.40 + 0.48 * _nz(pcs[:, 0]), 0.05, 0.98)
    L = toe_inv(Lr)
    h = 360.0 * _nz(pcs[:, 1])
    frac = 0.30 + 0.60 * (_nz(pcs[:, 2]) if pcs.shape[1] > 2 else np.full(n, 0.5))
    ceiling = max_chroma(L, h)
    C = frac * ceiling
    ev = vals[:3] / (vals.sum() + 1e-12)
    return ColorSpec(
        L=L, C=C, h=h, anchor_hue=float(np.mean(h)) if len(h) else 0.0,
        temperament=None,
        provenance={
            "colorspace": "derived", "pca_explained": ev.tolist(),
            "hue": {"from": "learned_PC2"}, "light": {"from": "learned_PC1"},
            "chroma": {"from": "learned_PC3", "frac": frac, "ceiling": ceiling},
            "temperament": "derived", "calibration": cal.name, "n": n,
        },
    )
