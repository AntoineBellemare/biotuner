"""Public entry points: signals and tunings -> :class:`Palette`.

Module type: Functions + dataclass

Stability vs separation
-----------------------
These cannot both hold in a stateless function, and pretending otherwise is
what broke the prototype. They are exposed as separate modes:

- ``palette_from_signal(..., mode="absolute")`` -- **stable**. The same signal
  always yields the same palette, with no reference to any other signal. Use
  for reproducible figures, streaming, and cross-paper comparability.
- ``palette_set([...], mode="separated")`` -- **separated**. Anchors are pushed
  apart until every pair clears ``min_deltaE``, while staying as close as
  possible to where each signal's own content puts it. Use when N palettes must
  be told apart in one figure.

Absolute mode is honest about its limit: two genuinely similar signals *should*
look similar, and they will. Separated mode buys distinguishability by spending
fidelity, and reports exactly how much it spent (``anchor_shift``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from biotuner.biocolors.calibration import Calibration, load_calibration
from biotuner.biocolors.color.difference import (
    min_separation,
    pairwise_deltaE,
    simulate_cvd,
)
from biotuner.biocolors.color.gamut import clip_report, gamut_map
from biotuner.biocolors.color.spaces import (
    oklch_to_srgb,
    srgb_to_oklab,
    toe,
)
from biotuner.biocolors.descriptors import (
    Fingerprint,
    SignalContext,
    fingerprint,
    make_context,
)
from biotuner.biocolors.mapping import MAPPINGS, ColorSpec, TEMPERAMENTS

__all__ = [
    "Palette",
    "palette_from_signal",
    "palette_from_tuning",
    "palette_from_biotuner",
    "palette_from_raw",
    "palette_set",
    "dyad_field",
    "consonance_spectrum",
    "diversity_report",
    "palette_report",
]


def _hex(rgb):
    r, g, b = (np.clip(np.asarray(rgb, float), 0, 1) * 255).round().astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


@dataclass
class Palette:
    """A palette plus everything needed to explain and audit it."""

    rgb: np.ndarray                 # (n, 3) in 0-1, guaranteed in gamut
    spec: ColorSpec
    fingerprint: Fingerprint
    context: SignalContext
    calibration: Calibration
    parameters: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.rgb)

    @property
    def oklch(self):
        return self.spec.lch

    @property
    def lab(self):
        return srgb_to_oklab(self.rgb)

    def hex(self):
        """List of ``#rrggbb`` strings."""
        return [_hex(c) for c in self.rgb]

    def rgb255(self):
        return (np.clip(self.rgb, 0, 1) * 255).round().astype(int)

    def ramp(self, n=256, order="pitch"):
        """Smooth OKLab-interpolated gradient across the swatches.

        Interpolating in OKLab (not sRGB) keeps the ribbon perceptually even and
        routes hue transitions through lower chroma instead of through mud.
        """
        lab = self.lab
        if order == "pitch":
            idx = np.argsort(np.asarray(self.context.scale, float))
        elif order == "lightness":
            idx = np.argsort(self.spec.L)
        elif order == "hue":
            idx = np.argsort(self.spec.h)
        else:
            idx = np.arange(len(lab))
        lab = lab[idx]
        stops = np.linspace(0, 1, len(lab))
        ts = np.linspace(0, 1, n)
        out = np.stack([np.interp(ts, stops, lab[:, k]) for k in range(3)], axis=-1)
        from biotuner.biocolors.color.spaces import oklab_to_oklch
        return _finalize(oklab_to_oklch(out))

    def explain(self, i):
        """Why is swatch ``i`` that colour? Returns a human-readable string."""
        p = self.spec.provenance
        s = self.context.scale[i]
        f = s * self.context.fund
        lines = [
            f"step {i}  ratio={s:.4f}  freq={f:.2f} Hz  ->  {self.hex()[i]}",
        ]
        if "anchor" in p:
            a = p["anchor"]
            lines.append(
                f"  anchor  = {a['hue']:.1f}deg  <- {a['mode']} "
                f"(radius={a['radius']:.3f}, PCA covers {a['pca_explained']*100:.0f}% of corpus var)"
            )
        hu = p.get("hue", {})
        lines.append(
            f"  hue     = {self.spec.h[i]:7.2f}deg <- {hu.get('from','?')}"
            + (f", t={hu['t'][i]:.2f}, arc={hu.get('arc',0):.0f}deg" if "t" in hu else "")
        )
        li = p.get("light", {})
        lines.append(
            f"  light   = {self.spec.L[i]:7.4f}   <- {li.get('from','?')}"
            + (f", t={li['t'][i]:.2f} -> Lr={li['Lr'][i]:.3f}" if "Lr" in li else "")
        )
        ch = p.get("chroma", {})
        if "ceiling" in ch:
            lines.append(
                f"  chroma  = {self.spec.C[i]:7.4f}   <- {ch.get('from','?')}, "
                f"t={ch['t'][i]:.2f} -> {ch['frac'][i]*100:.0f}% of cusp "
                f"({ch['ceiling'][i]:.3f})"
            )
        else:
            lines.append(f"  chroma  = {self.spec.C[i]:7.4f}")
        lines.append(
            f"  style   = {p.get('temperament','?')}   calibration = {p.get('calibration','?')}"
        )
        return "\n".join(lines)

    def report(self):
        return palette_report(self)

    def __repr__(self):
        t = self.spec.provenance.get("temperament", "?")
        return (f"Palette(n={len(self)}, style={t!r}, "
                f"anchor={self.spec.anchor_hue:.0f}deg, {self.hex()[:3]}...)")


def _finalize(lch):
    """Gamut-map then render to sRGB. The only path to pixels."""
    lch = gamut_map(np.asarray(lch, float), mode="chroma")
    return np.clip(oklch_to_srgb(lch, clip=True), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def palette_from_signal(peaks, amps=None, *, mode="absolute",
                        method="anchored", calibration="eeg_sleep_v1",
                        temperament="auto", amps_scale="db", config=None,
                        **overrides):
    """Palette from a signal's spectral peaks.

    Parameters
    ----------
    peaks, amps : array-like
        Spectral peaks (Hz) and their amplitudes.
    mode : {'absolute'}
        Absolute (stateless, reproducible). For separation across several
        signals use :func:`palette_set`.
    method : str
        A registered mapping: ``anchored`` (default), ``spectral``, ``mds``.
    calibration : str or Calibration
        Percentile reference. ``'none'`` disables calibration.
    temperament : str or Temperament
        ``'auto'`` lets the signal pick its character, or name one of
        :data:`~biotuner.biocolors.mapping.TEMPERAMENTS`.
    amps_scale : {'db', 'linear', 'auto'}
        Scale of ``amps``. Default ``'db'`` matches biotuner's Welch-based
        extractors (``fixed``, ``adapt``, ...), which return ``10*log10(psd)``.
        Pass ``'linear'`` for ``EMD`` / ``HilbertHuang1D`` envelopes. This must
        match the calibration's ``amps_scale`` or fingerprints will not be
        comparable.

    Returns
    -------
    Palette
    """
    cfg = dict(config or {})
    cfg.update(overrides)
    if mode != "absolute":
        raise ValueError("palette_from_signal only does mode='absolute'; "
                         "use palette_set(mode='separated') for a group")

    ctx = make_context(peaks, amps, amps_scale=amps_scale)
    cal = load_calibration(calibration)
    cal_scale = cal.meta.get("amps_scale")
    # No amplitudes means no scale to disagree about: every step is weighted
    # equally either way, so the mismatch warning would be noise.
    if amps is not None and cal_scale is not None and cal_scale != amps_scale:
        import warnings as _w
        _w.warn(
            f"amps_scale={amps_scale!r} but calibration {cal.name!r} was fit with "
            f"{cal_scale!r}; fingerprints are not comparable and palettes will be "
            f"arbitrary. Pass amps_scale={cal_scale!r} or refit.",
            RuntimeWarning, stacklevel=2,
        )
    fp = fingerprint(ctx, fields=cal.fields)
    sat = cal.saturation(fp)
    if sat > 0.5:
        import warnings as _w
        _w.warn(
            f"{sat:.0%} of this signal's descriptors fall outside calibration "
            f"{cal.name!r} (fit on peaks_function="
            f"{cal.meta.get('peaks_function','?')!r}, n_peaks="
            f"{cal.meta.get('n_peaks','?')}). Those percentiles are pinned at 0/1, so the "
            f"hue anchor is largely arbitrary and unrelated signals may share it. "
            f"Refit with build_calibration() on your own extractor, or pass "
            f"calibration='none'. See Palette.report()['fingerprint_saturation'].",
            RuntimeWarning, stacklevel=2,
        )
    if method not in MAPPINGS:
        raise KeyError(f"unknown mapping {method!r}. Registered: {sorted(MAPPINGS)}")
    spec = MAPPINGS[method](ctx, fp, cal, temperament=temperament, **cfg)
    rgb = _finalize(spec.lch)
    return Palette(
        rgb=rgb, spec=spec, fingerprint=fp, context=ctx, calibration=cal,
        parameters={"mode": mode, "method": method, "temperament": temperament,
                    "calibration": cal.name, **cfg},
    )


def palette_from_tuning(scale, fund=1.0, *, method="anchored",
                        calibration="tuning_v1", temperament="auto",
                        amps=None, config=None, **overrides):
    """Palette from tuning ratios. ``scale`` is ratios (1..2), ``fund`` in Hz.

    Defaults to the ``tuning_v1`` calibration, not ``eeg_sleep_v1``: a scale of
    ratios is not an EEG spectrum, and against the EEG corpus a tuning saturates
    45-91% of its descriptors (measured), which leaves the hue anchor mostly
    arbitrary. Under ``tuning_v1`` the same scales saturate 18-45%. For tunings
    *derived from a measured signal*, pass ``calibration='tuning_eeg_v1'`` —
    that is a third population again.

    A tuning still goes through the same :class:`~biotuner.biocolors.descriptors.Fingerprint`
    machinery as a spectrum; there is no separate path. But it is worth knowing
    that **four of the eleven descriptors are structurally constant for any
    tuning**, because a tuning has no amplitudes and :func:`make_context` then
    weights every step equally:

    ========================  ==================================================
    dead for tunings          why
    ========================  ==================================================
    ``spectral_flatness``     Wiener entropy of uniform weights == 1, always
    ``spectral_entropy``      Shannon entropy of uniform weights == 1, always
    ``complexity``            Higuchi FD of a constant series (guarded to 1.0)
    ``fundamental``           ``log2(min ratio)`` == 0 for any scale starting at 1
    ========================  ==================================================

    They cost nothing: a constant is annihilated twice over -- ``project()``
    subtracts ``pca_mean`` so a constant contributes exactly 0, and
    :func:`~biotuner.biocolors.calibration._weights` independently drives three
    of them to 0.000 on any tuning corpus. Measured, dropping all four shifts
    the anchor by <= 0.11 deg. So the tuning anchor is effectively computed from
    seven descriptors: ``harmonicity``, ``harmonic_spread``, ``tenney``,
    ``n_peaks``, ``octave_span``, ``spectral_centroid``, ``spectral_spread``.

    Note the last two are misnomers here. Under uniform weights they reduce to
    ``mean(ratios)`` and ``std(ratios)`` -- still informative about a scale, but
    nothing to do with a spectrum. They are kept under the spectral names so one
    field order serves both populations and the shipped PCA projections stay
    valid.

    Lightness: any channel reading ``amplitude`` goes flat for the same reason,
    so it falls back automatically (see ``_FALLBACKS``) to ``tenney_step``.
    Pass ``light_from='pitch'`` if you want position-in-scale instead of ratio
    simplicity.

    A note on ``method='spectral'``: that mapping folds a *frequency* to a
    wavelength, so it needs real Hz. With the default ``fund=1.0`` the scale is
    dimensionless ratios in [1, 2], which all fold into a near-degenerate band
    (measured: ratios 1.007-1.042 all land within 3 nm, one green). The call is
    therefore guarded -- see below. Two legitimate uses remain: pass a real
    ``fund`` (e.g. 440) to colour an actual set of audible tones, or use
    :func:`dyad_field`, whose interval-class hue (one turn per octave) is the
    ratio-domain analogue of the wavelength fold.
    """
    # spectral maps frequency -> wavelength; dimensionless ratios (fund == 1.0)
    # have no frequency to fold and collapse into one hue band. Warn rather than
    # silently return a meaningless green wash. A real fund is fine, so only the
    # sentinel default trips the guard. Mirrors the calibration-domain warning.
    if str(method) == "spectral" and float(fund) == 1.0:
        import warnings as _w
        _w.warn(
            "method='spectral' folds frequency to wavelength, but these are "
            "dimensionless ratios (fund=1.0) with no absolute frequency: every "
            "ratio in [1, 2] folds into a near-degenerate hue band. For a "
            "ratio/interval representation use dyad_field (interval-class hue, "
            "one turn per octave); to colour real tones pass fund in Hz; to use "
            "spectral on a spectrum apply it to level='peaks' or 'extended'.",
            RuntimeWarning, stacklevel=2,
        )
    scale = np.asarray(scale, float)
    peaks = scale * float(fund)
    return palette_from_signal(
        peaks, amps, method=method, calibration=calibration,
        temperament=temperament, amps_scale="linear", config=config, **overrides,
    )


#: What a :class:`~biotuner.biotuner_object.compute_biotuner` exposes at each
#: level of representation, and what it means for a palette.
LEVELS = ("peaks", "extended", "ratios", "extended_ratios", "cons_ratios")


#: Default calibration per level, for tunings that came from a *measured signal*.
#:
#: Two separate lessons are baked in here, both learned the hard way.
#:
#: First, peak levels are spectra and ratio levels are tunings; scoring one
#: against the other's corpus saturates 64-91% of descriptors and pins every
#: signal to the same anchor. Measured: all four sleep stages collapsed to
#: 69 deg (min hue gap 0) at every ratio level, while their raw projection
#: angles were in fact 348, 335, 36 and 1 deg.
#:
#: Second -- and less obvious -- a tuning *derived from EEG* and a tuning
#: *designed by a theorist* are different populations, so ``tuning_v1``
#: (synthetic: n-TET, JI, harmonic series) is still the wrong corpus here.
#: Against it, every EEG-derived tuning lands at nearly the same percentile of
#: the strongest descriptors -- normalised ``harmonicity`` std 0.012, ``tenney``
#: 0.020 across ten varied signals -- so those axes carry no information no
#: matter how heavily they are weighted, and the anchors collapse into a ~60 deg
#: wedge (vs 240 deg at the peak level). ``tuning_eeg_v1`` percentile-ranks
#: against tunings actually derived from this recording.
#:
#: :func:`palette_from_tuning` keeps ``tuning_v1`` as its default, because a bare
#: scale handed in by a user is a designed object, not a measured one.
_LEVEL_CALIBRATION = {
    "peaks": "eeg_sleep_v1",
    "extended": "eeg_sleep_v1",
    "ratios": "tuning_eeg_v1",
    "extended_ratios": "tuning_eeg_v1",
    "cons_ratios": "tuning_eeg_v1",
}


def palette_from_biotuner(bt, level="peaks", *, method="anchored",
                          calibration="auto", temperament="auto",
                          **overrides):
    """Palette straight from a ``compute_biotuner`` object.

    ``calibration='auto'`` (the default) picks per level: ``eeg_sleep_v1`` for
    the peak levels, ``tuning_v1`` for the ratio levels. Pass an explicit name
    to override, but note that a tuning scored against the EEG corpus, or vice
    versa, is out of domain — check ``Palette.report()['fingerprint_saturation']``.

    Reads ``bt.peaks_function`` and applies the right amplitude scale
    automatically (see :data:`~biotuner.biocolors.descriptors.AMPS_SCALE_BY_PEAKS_FUNCTION`)
    -- biotuner's extractors disagree about whether amplitudes are dB or linear,
    and getting it wrong yields an arbitrary palette rather than an error.

    ``level`` selects the representation, which also sets the palette's size:

    - ``'peaks'``           ``bt.peaks`` — the measured spectral peaks.
    - ``'extended'``        ``bt.extended_peaks`` — peaks plus fitted harmonics
      (requires a prior ``bt.peaks_extension(...)``). Larger, and more harmonic
      by construction, so palettes shift toward the consonant end.
    - ``'ratios'``          ``bt.peaks_ratios`` — octave-rebounded pairwise
      ratios. A *tuning*, not a spectrum: no amplitudes, more elements.
    - ``'extended_ratios'`` ``bt.extended_peaks_ratios``.
    - ``'cons_ratios'``     ``bt.peaks_ratios_cons`` — consonance-filtered, so
      typically only a handful survive.

    Ratio levels have no amplitudes; see :func:`palette_from_tuning`.
    """
    from biotuner.biocolors.descriptors import amps_scale_for

    if level not in _LEVEL_CALIBRATION:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    if calibration == "auto":
        calibration = _LEVEL_CALIBRATION[level]

    pf = getattr(bt, "peaks_function", None)
    if level == "peaks":
        pk, am = getattr(bt, "peaks", None), getattr(bt, "amps", None)
        scale_ = amps_scale_for(pf)
    elif level == "extended":
        pk = getattr(bt, "extended_peaks", None)
        am = getattr(bt, "extended_amps", None)
        scale_ = amps_scale_for(pf)
    elif level in ("ratios", "extended_ratios", "cons_ratios"):
        attr = {"ratios": "peaks_ratios", "extended_ratios": "extended_peaks_ratios",
                "cons_ratios": "peaks_ratios_cons"}[level]
        r = getattr(bt, attr, None)
        if r is None:
            raise AttributeError(
                f"biotuner object has no {attr!r}; run peaks_extraction "
                f"(and peaks_extension for extended levels) first"
            )
        return palette_from_tuning(
            np.asarray(r, float), fund=1.0, method=method,
            calibration=calibration, temperament=temperament, **overrides,
        )
    else:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")

    if pk is None or len(np.asarray(pk, float)) == 0:
        raise AttributeError(
            f"biotuner object has no usable {level!r}; run peaks_extraction "
            f"(and peaks_extension for level='extended') first"
        )
    if am is None:
        am = np.ones(len(np.asarray(pk, float)))
        scale_ = "linear"
    return palette_from_signal(
        pk, am, method=method, calibration=calibration, temperament=temperament,
        amps_scale=scale_, **overrides,
    )


# The "signal -> tuning by method" dispatch lives in the tuning layer, not here.
# Use biotuner.biotuner_object.tuning_from_raw (raw -> ratios) or
# compute_biotuner.get_tuning(source) (from an extracted object); this module only
# *colours* the result. biotuner_object is imported lazily inside the functions
# below so that importing biocolors does not pull its heavier dependencies.


def palette_from_raw(data, sf, *, peaks_function="fixed", precision=0.1,
                     n_peaks=5, min_freq=1.0, max_freq=45.0, n_harm=10,
                     level="peaks", tuning=None, tuning_params=None,
                     method="anchored", calibration="auto",
                     temperament="auto", return_bt=False, extraction=None,
                     **overrides):
    """Palette from a **raw time series** — the front door for a recording.

    Runs biotuner peak extraction with the parameters you choose, then maps the
    result. This is the one call that goes signal -> peaks -> colour; use
    :func:`palette_from_signal` when you already have peaks, or
    :func:`palette_from_biotuner` when you already have an extracted object.

    Parameters
    ----------
    data : array-like
        The raw signal (a 1-D time series).
    sf : float
        Sampling frequency in Hz.
    peaks_function : str
        biotuner extractor: ``'fixed'``, ``'adapt'``, ``'FOOOF'``, ``'EMD'``,
        ``'cepstrum'``, ... The amplitude scale (dB vs linear) is inferred from
        this automatically, so you never set it by hand.
    precision : float
        Spectral resolution in Hz. Finer (e.g. 0.1) separates close peaks onto
        distinct colours; 0.25 is coarser and faster. Matters most for
        ``method='spectral'``.
    n_peaks, min_freq, max_freq, n_harm :
        Passed to biotuner extraction (and extension, for extended levels).
    level : str
        Spectrum representation to colour (see :func:`palette_from_biotuner`):
        ``peaks``, ``extended``, ``ratios``, ``extended_ratios``, ``cons_ratios``.
        Ignored when ``tuning`` is given.
    tuning : str, optional
        Colour a **tuning derived from the signal** instead of the spectrum. One
        of :data:`TUNING_SOURCES` — ``diss_curve``, ``harmonic_entropy``,
        ``euler_fokker``, ``harmonic_tuning``, ``harmonic_fit``, or the ratio
        sources. Overrides ``level``. Uses the ``tuning_eeg_v1`` calibration by
        default (a signal-derived tuning is a measured, not designed, scale).
    tuning_params : dict, optional
        Extra keyword args for the tuning derivation (e.g. ``list_harmonics`` for
        ``harmonic_tuning``, ``res`` for ``harmonic_entropy``).
    extraction : dict, optional
        Extra keyword arguments forwarded verbatim to ``bt.peaks_extraction`` for
        anything not surfaced above (prominence, nIMFs, ...).
    return_bt : bool
        If True, return ``(palette, bt)`` so you can inspect the peaks/tuning the
        colour came from.

    Returns
    -------
    Palette, or (Palette, compute_biotuner) if ``return_bt``.

    See also
    --------
    tuning_from_raw : the same signal -> tuning derivations, returning the ratios
        (for export, sonification, or scale construction) rather than a palette.
    """
    # Lazy import: compute_biotuner pulls heavier deps (fooof/specparam); keep
    # them out of the import path of anyone using only the peaks/tuning APIs.
    from biotuner.biotuner_object import compute_biotuner

    bt = compute_biotuner(sf=sf, peaks_function=peaks_function,
                          precision=precision, n_harm=n_harm)
    bt.peaks_extraction(
        np.asarray(data, float), min_freq=min_freq, max_freq=max_freq,
        n_peaks=n_peaks, graph=False, **(extraction or {}),
    )
    meta = {"peaks_function": peaks_function, "precision": precision,
            "n_peaks": n_peaks, "min_freq": min_freq, "max_freq": max_freq,
            "sf": sf}

    if tuning is not None:
        # Colour a signal-derived tuning. The derivation itself lives in the
        # tuning layer (bt.get_tuning); we only colour it. Route to the
        # measured-tuning corpus.
        if tuning == "extended_ratios":
            bt.peaks_extension(method="harmonic_fit", n_harm=n_harm)
        scale = bt.get_tuning(tuning, **(tuning_params or {}))
        cal = "tuning_eeg_v1" if calibration == "auto" else calibration
        pal = palette_from_tuning(
            scale, method=method, calibration=cal, temperament=temperament,
            **overrides,
        )
        pal.metadata["extraction"] = {**meta, "tuning_source": tuning}
        return (pal, bt) if return_bt else pal

    if level in ("extended", "extended_ratios"):
        bt.peaks_extension(method="harmonic_fit", n_harm=n_harm)
    pal = palette_from_biotuner(
        bt, level=level, method=method, calibration=calibration,
        temperament=temperament, **overrides,
    )
    pal.metadata["extraction"] = {**meta, "level": level}
    return (pal, bt) if return_bt else pal


def palette_set(signals, *, mode="separated", min_deltaE=0.15,
                method="anchored", calibration="eeg_sleep_v1",
                temperament="auto", max_iter=200, **kw):
    """Palettes for several signals at once.

    ``mode='separated'`` relaxes the anchor hues apart until every pair of
    palettes clears ``min_deltaE``, while pulling each anchor back toward where
    its own fingerprint put it. Reports the cost in
    ``palette.metadata['anchor_shift']`` (degrees moved) so you can see what was
    traded for legibility.

    ``mode='absolute'`` just maps each independently -- reproducible, but two
    similar signals will look similar.
    """
    pals = [
        palette_from_signal(p, a, method=method, calibration=calibration,
                            temperament=temperament, **kw)
        for p, a in signals
    ]
    if mode == "absolute":
        return pals
    if mode != "separated":
        raise ValueError(f"unknown mode {mode!r}")
    if len(pals) < 2:
        return pals

    home = np.array([p.spec.anchor_hue for p in pals], float)
    hue = home.copy()
    n = len(hue)
    # Repel on the hue circle until the minimum gap clears the target, pulling
    # each anchor back toward home so the mapping stays meaningful.
    target_gap = min(360.0 / n, 360.0 * min_deltaE)
    for _ in range(max_iter):
        force = np.zeros(n)
        worst = 360.0
        for i in range(n):
            for j in range(i + 1, n):
                d = (hue[j] - hue[i] + 180.0) % 360.0 - 180.0
                ad = abs(d)
                worst = min(worst, ad)
                if ad < target_gap and ad > 1e-9:
                    push = (target_gap - ad) * 0.25 * np.sign(d)
                    force[i] -= push
                    force[j] += push
                elif ad <= 1e-9:
                    force[i] -= target_gap * 0.25
                    force[j] += target_gap * 0.25
        if worst >= target_gap:
            break
        # spring back toward the signal's own anchor
        home_pull = ((home - hue + 180.0) % 360.0 - 180.0) * 0.05
        hue = (hue + force + home_pull) % 360.0

    out = []
    for p, h_new, h_old in zip(pals, hue, home):
        shift = float((h_new - h_old + 180.0) % 360.0 - 180.0)
        peaks, amps = p.context.peaks, p.context.amps
        q = palette_from_signal(
            peaks, amps, method=method, calibration=calibration,
            temperament=temperament, hue_rotate=shift, **kw,
        )
        q.metadata["anchor_shift"] = shift
        q.metadata["anchor_home"] = float(h_old)
        out.append(q)
    return out


# --------------------------------------------------------------------------- #
# Other palette forms
# --------------------------------------------------------------------------- #
def dyad_field(scale, function=None, temperament="balanced", chroma_frac=0.7):
    """Pairwise interval matrix as a colour field. Returns ``(img, cons)``.

    Cell (i, j) takes hue from the interval class of ``scale[i]/scale[j]`` (one
    hue turn per octave, so unison and octave share a hue and the tritone sits
    opposite) and lightness from that interval's consonance.
    """
    from biotuner.metrics import dyad_similarity
    from biotuner.biocolors.color.gamut import max_chroma

    function = function or dyad_similarity
    scale = np.asarray(scale, float)
    n = len(scale)
    cons = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            if i != j:
                try:
                    cons[i, j] = function(float(scale[i] / scale[j]))
                except Exception:
                    cons[i, j] = np.nan
    fin = cons[np.isfinite(cons)]
    lo, hi = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0)

    t = TEMPERAMENTS[temperament] if isinstance(temperament, str) else temperament
    ratio = scale[:, None] / scale[None, :]
    h = (np.log2(ratio) % 1.0) * 360.0
    cn = (cons - lo) / (hi - lo + 1e-12)
    Lr = np.where(np.isfinite(cn), t.L_center + (cn - 0.5) * 2 * t.L_spread, 0.95)
    from biotuner.biocolors.color.spaces import toe_inv
    L = toe_inv(np.clip(Lr, 0.06, 0.98))
    frac = np.where(np.isfinite(cn), chroma_frac, 0.05)
    C = frac * max_chroma(L, h)
    img = _finalize(np.stack([L, C, h], axis=-1))
    return img, cons


def consonance_spectrum(peaks, amps, *, max_ratio=2.0, n=512, smooth=17,
                        calibration="eeg_sleep_v1", temperament="auto",
                        arc=None):
    """Continuous consonance spectrum coloured from the signal's own dissonance curve.

    Slides the measured spectrum against itself (Sethares) across ``[1, max_ratio]``.
    Returns ``(ratios, consonance, rgb)``.
    """
    from biotuner.biocolors.color.gamut import max_chroma
    from biotuner.biocolors.color.spaces import toe_inv
    from biotuner.biocolors.mapping import auto_temperament
    from biotuner.scale_construction import diss_curve
    from biotuner.biocolors.descriptors import _normalize_amps

    ctx = make_context(peaks, amps)
    cal = load_calibration(calibration)
    fp = fingerprint(ctx, fields=cal.fields)
    t = (auto_temperament(fp, cal) if temperament == "auto"
         else TEMPERAMENTS[temperament] if isinstance(temperament, str)
         else temperament)
    if arc is not None:
        t.arc = float(arc)

    # diss_curve needs positive amplitudes; ctx.amps is already rectified.
    diss = np.asarray(
        diss_curve(np.asarray(ctx.peaks, float), np.asarray(ctx.amps, float),
                   denom=100, max_ratio=max_ratio, method="min", plot=False)[0],
        float,
    )
    rs = np.linspace(1.0, max_ratio, len(diss))
    cons = 1.0 - (diss - diss.min()) / (np.ptp(diss) + 1e-12)
    if smooth and smooth > 2 and len(cons) > smooth:
        k = np.hanning(smooth)
        k /= k.sum()
        cons = np.convolve(cons, k, mode="same")
        cons = (cons - cons.min()) / (np.ptp(cons) + 1e-12)

    pc = cal.project(fp)
    anchor = float(np.degrees(np.arctan2(pc[1], pc[0])) % 360.0) + t.hue_bias
    span = np.log2(max_ratio)
    h = (anchor + (np.log2(rs) / span - 0.5) * t.arc) % 360.0
    Lr = np.clip(t.L_center + (cons - 0.5) * 2 * t.L_spread, 0.06, 0.98)
    L = toe_inv(Lr)
    frac = np.clip(t.C_frac_center + (cons - 0.5) * 2 * t.C_frac_spread, 0.03, 1.0)
    C = frac * max_chroma(L, h)
    rgb = _finalize(np.stack([L, C, h], axis=-1))
    return rs, cons, rgb


# --------------------------------------------------------------------------- #
# Diagnostics -- these make "the palettes are diverse" a measured claim
# --------------------------------------------------------------------------- #
def circular_range(deg):
    """Angular extent of a set of hues, in degrees.

    ``np.ptp`` is wrong for angles: hues at 359 and 1 are 2 degrees apart, not
    358. This finds the smallest arc containing every hue by taking the largest
    gap on the circle and subtracting it from 360.
    """
    a = np.sort(np.asarray(deg, float) % 360.0)
    if len(a) < 2:
        return 0.0
    gaps = np.diff(np.concatenate([a, a[:1] + 360.0]))
    return float(360.0 - gaps.max())


def palette_report(palette):
    """Audit one palette: separation, CVD safety, gamut, lightness range."""
    rgb = np.asarray(palette.rgb, float)
    lab = srgb_to_oklab(rgb)
    d = pairwise_deltaE(lab)
    iu = np.triu_indices(len(rgb), k=1)
    sep = min_separation(rgb, under_cvd=True)
    frac = palette.spec.provenance.get("chroma", {}).get("frac", np.nan)
    # Adjacent separation is the right question for a large palette; all-pairs is
    # the right question for a legend. A 25-step scale read as a gradient SHOULD
    # have distant swatches that resemble each other -- demanding otherwise would
    # force a rainbow. What must never happen is neighbours you cannot tell apart.
    adj = (float(np.linalg.norm(np.diff(lab, axis=0), axis=-1).min())
           if len(rgb) > 1 else float("inf"))
    return {
        "n": len(rgb),
        "min_deltaE": float(d[iu].min()) if len(rgb) > 1 else float("inf"),
        "min_adjacent_deltaE": adj,
        "mean_deltaE": float(d[iu].mean()) if len(rgb) > 1 else 0.0,
        "min_deltaE_cvd": {k: v for k, v in sep.items() if k != "normal"},
        "lightness_range": (float(toe(palette.spec.L).min()),
                            float(toe(palette.spec.L).max())),
        "chroma_range": (float(palette.spec.C.min()), float(palette.spec.C.max())),
        "hue_arc": circular_range(palette.spec.h),
        "mean_chroma_frac": float(np.mean(frac)),
        "gamut": clip_report(palette.spec.lch),
        "temperament": palette.spec.provenance.get("temperament"),
        # >0.3 means the signal sits outside the calibration corpus and the
        # anchor is being driven by pinned percentiles. See Calibration.saturation.
        "fingerprint_saturation": palette.calibration.saturation(palette.fingerprint),
    }


def diversity_report(palettes, threshold=0.05):
    """Audit a *set* of palettes: do different signals look different?

    ``mean_palette_deltaE`` is the mean OKLab distance between palettes
    (swatch-wise). ``n_collisions`` counts pairs closer than ``threshold`` --
    i.e. pairs a reader would call the same palette. A non-zero count is the
    failure the fingerprint exists to prevent.
    """
    labs = [srgb_to_oklab(np.asarray(p.rgb, float)) for p in palettes]
    n = len(labs)
    if n < 2:
        return {"n": n}
    ds, pairs = [], []
    for i in range(n):
        for j in range(i + 1, n):
            m = min(len(labs[i]), len(labs[j]))
            d = float(np.linalg.norm(labs[i][:m] - labs[j][:m], axis=-1).mean())
            ds.append(d)
            pairs.append((d, i, j))
    ds = np.array(ds)
    pairs.sort()
    hues = np.array([p.spec.anchor_hue for p in palettes])
    hgaps = []
    for i in range(n):
        for j in range(i + 1, n):
            g = abs(hues[i] - hues[j])
            hgaps.append(min(g, 360 - g))
    within = [float(np.linalg.norm(l - l.mean(0), axis=-1).mean()) for l in labs]
    occ = np.zeros(36, dtype=bool)
    occ[(hues.astype(int) // 10) % 36] = True
    return {
        "n": n,
        "mean_palette_deltaE": float(ds.mean()),
        "min_palette_deltaE": float(ds.min()),
        "max_palette_deltaE": float(ds.max()),
        "n_collisions": int(np.sum(ds < threshold)),
        "collision_pairs": [(i, j, round(d, 4)) for d, i, j in pairs if d < threshold],
        "closest_pair": (pairs[0][1], pairs[0][2], round(pairs[0][0], 4)),
        "anchor_hues": np.round(hues, 1).tolist(),
        "min_hue_gap": float(np.min(hgaps)),
        "median_hue_gap": float(np.median(hgaps)),
        "hue_occupancy": float(occ.mean()),
        "mean_within_palette_deltaE": float(np.mean(within)),
    }
