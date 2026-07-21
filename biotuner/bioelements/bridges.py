"""Bridges from a material's composite spectrum into the rest of biotuner.

Because a :class:`~biotuner.bioelements.composition.Composition` yields a composite
spectrum, it plugs straight into biotuner: fold the strongest lines to audio and
you have a **material chord**; reduce the ratios and you have a **material tuning**;
send it through biocolors and you have a **material palette**.
"""
from __future__ import annotations

import numpy as np

from biotuner.bioelements import units

_AUDIBLE_BAND_HZ = (80.0, 1600.0)


def _fold_hz(freq: float, band=_AUDIBLE_BAND_HZ) -> float:
    """Octave-fold a frequency into an audible band [lo, hi] Hz."""
    lo, hi = band
    f = float(freq)
    if f <= 0:
        return lo
    while f > hi:
        f /= 2.0
    while f < lo:
        f *= 2.0
    return f


def material_chord(material, *, top: int = 8, table: str = "air",
                   basis: str = "atom", band=_AUDIBLE_BAND_HZ):
    """The material's strongest composite lines, folded to audio: ``(freqs_hz, amps)``.

    ``amps`` are the NIST intensities (linear), so the loudest partial is the
    material's brightest emission line — the amplitude analogue carries through.
    """
    spec = material.spectrum(table=table, top=top * 4, basis=basis).select(top=top)
    if len(spec) == 0:
        return np.array([]), np.array([])
    freqs = np.array([_fold_hz(f, band) for f in spec.to_hz()])
    amps = np.asarray(spec.intensity, float)
    # merge near-duplicate folded frequencies (within ~1 Hz), summing amplitude
    order = np.argsort(freqs)
    freqs, amps = freqs[order], amps[order]
    keep_f, keep_a = [], []
    for f, a in zip(freqs, amps):
        if keep_f and abs(f - keep_f[-1]) < 1.0:
            keep_a[-1] += a
        else:
            keep_f.append(f); keep_a.append(a)
    return np.array(keep_f), np.array(keep_a)


def material_tuning(material, *, n_steps: int = 7, top: int = 12, table: str = "air",
                    basis: str = "atom", max_ratio: float = 2.0) -> list[float]:
    """A material scale: octave-reduced ratios among its strongest composite lines."""
    from biotuner.biotuner_utils import compute_peak_ratios

    spec = material.spectrum(table=table, top=top * 3, basis=basis).select(top=top)
    if len(spec) < 2:
        return []
    hz = spec.to_hz()
    ratios = compute_peak_ratios(list(hz), rebound=True)
    ratios = sorted({round(float(r), 4) for r in ratios if 1.0 <= float(r) <= max_ratio})
    return ratios[:n_steps]


_VISIBLE_NM = (380.0, 720.0)   # clamp red edge to <=720 nm: the CIE1931 fit greens out beyond


def _fold_to_visible(wavelength_angstrom: float, band=_VISIBLE_NM) -> float:
    """Octave-fold a wavelength (Å) into the visible band [lo, hi] nm."""
    lo, hi = band
    nm = float(wavelength_angstrom) / 10.0
    if nm <= 0:
        return lo
    while nm > hi:
        nm /= 2.0
    while nm < lo:
        nm *= 2.0
    return nm


def material_palette(material, *, n: int = 6, table: str = "air", basis: str = "atom",
                     chroma_frac: float = 0.62):
    """The material's **emission-colour palette** as a list of ``#rrggbb`` strings.

    The composite lines are octave-folded into the visible band and binned across
    it; each occupied bin becomes one swatch whose hue is the bin's intensity-
    weighted mean wavelength (CIE-1931) and whose lightness tracks how strongly the
    material emits there. Chroma is softened in OKLCh (``chroma_frac``) so the result
    reads rich/earthy rather than neon. Materials differ by *where* they emit, so
    water (H/O) ≠ fire (Na/C) ≠ iron (dense transition-metal lines).
    """
    import numpy as np
    from biotuner.biocolors.color.spectral import wavelength_to_srgb
    from biotuner.biocolors.color.spaces import srgb_to_oklch, oklch_to_srgb
    from biotuner.biocolors.color.gamut import gamut_map

    spec = material.spectrum(table=table, top=240, basis=basis)
    if len(spec) == 0:
        return []
    nm = np.array([_fold_to_visible(w) for w in spec.wavelength])
    inten = np.asarray(spec.intensity, float)

    edges = np.linspace(_VISIBLE_NM[0], _VISIBLE_NM[1], n + 1)
    means, weights = [], []
    for i in range(n):
        hi = edges[i + 1] + (1e-6 if i == n - 1 else 0.0)
        m = (nm >= edges[i]) & (nm < hi)
        if m.any():
            means.append(float(np.average(nm[m], weights=inten[m])))
            weights.append(float(inten[m].sum()))
    if not means:
        return []
    wn = np.asarray(weights) / max(weights)
    base = np.clip(np.asarray(wavelength_to_srgb(np.asarray(means), method="cie1931"), float), 0, 1)
    base = np.atleast_2d(base)
    out = []
    for rgb, ln in zip(base, wn):
        L, C, h = srgb_to_oklch(rgb)
        lch = gamut_map(np.array([0.42 + 0.42 * float(ln), C * chroma_frac, h]), mode="chroma")
        srgb = np.clip(oklch_to_srgb(lch, clip=True), 0, 1)
        r, g, b = (srgb * 255).round().astype(int)
        out.append(f"#{r:02x}{g:02x}{b:02x}")
    return out


def element_flame_color(element, *, table: str = "air", top: int = 3) -> str:
    """The element's characteristic **flame / emission colour** as ``#rrggbb``.

    Uses the element's strongest lines *within the visible band* (the flame-test
    colour: Na→amber, Cu→green, K→violet, Li→crimson, Ca→orange, …); if it has no
    visible lines, the strongest lines are octave-folded into the visible band.
    """
    import numpy as np
    from biotuner.biocolors.color.spectral import wavelength_to_srgb
    from biotuner.bioelements.spectrum import element_spectrum
    from biotuner.bioelements.periodic import FLAME_COLORS

    # The iconic flame colour if known (canonical, not garish); else data-derived.
    if element in FLAME_COLORS:
        return FLAME_COLORS[element]
    try:
        sp = element_spectrum(element, table=table)
    except KeyError:
        return "#777777"
    if len(sp) == 0:
        return "#777777"
    wl_nm = np.asarray(sp.wavelength, float) / 10.0
    inten = np.asarray(sp.intensity, float)
    vis = (wl_nm >= 380) & (wl_nm <= 720)
    if vis.any():
        w, it = wl_nm[vis], inten[vis]
    else:
        w = np.array([_fold_to_visible(x) for x in sp.wavelength]); it = inten
    idx = np.argsort(it)[::-1][:top]
    wmean = float(np.average(w[idx], weights=it[idx]))
    rgb = np.clip(np.asarray(wavelength_to_srgb(np.array([wmean]), method="cie1931"), float).ravel(), 0, 1)
    r, g, b = (rgb * 255).round().astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


def material_flame_palette(material, *, n: int = 7, table: str = "air") -> list[str]:
    """A material's palette as the **flame colours of its dominant elements**.

    One swatch per dominant element (ordered by composition fraction), each the
    element's characteristic emission colour. Because different materials are made
    of different elements, the palettes are genuinely distinct — water (H/O) reads
    nothing like a copper salt (green) or sodium (amber)."""
    fr = material.elements()
    els = [e for e, _ in sorted(fr.items(), key=lambda kv: -kv[1]) if e != "Vacuum"][:n]
    return [element_flame_color(e, table=table) for e in els]


def material_biocolors_palette(material, *, top: int = 8, table: str = "air",
                               basis: str = "atom", temperament: str = "auto", **kw):
    """A biocolors :class:`~biotuner.biocolors.Palette` for the material (audio-fold path)."""
    from biotuner.biocolors import palette_from_signal
    freqs, amps = material_chord(material, top=top, table=table, basis=basis)
    if len(freqs) == 0:
        raise ValueError(f"material {material.name!r} has no lines to colour")
    return palette_from_signal(freqs, amps, amps_scale="linear",
                               calibration="none", temperament=temperament, **kw)


def material_geometry(material, *, top: int = 8, table: str = "air", basis: str = "atom"):
    """A :class:`~biotuner.harmonic_input.HarmonicInput` from the material's chord.

    This is the hand-off into :mod:`biotuner.harmonic_geometry`: pass the returned
    input to any generator (``lissajous``, a Chladni plate, a harmonograph, …) to
    render the material's *form*. Peaks are the folded composite lines; amplitudes
    are their NIST intensities.
    """
    from biotuner.harmonic_input import HarmonicInput
    freqs, amps = material_chord(material, top=top, table=table, basis=basis)
    if len(freqs) == 0:
        raise ValueError(f"material {material.name!r} has no lines for geometry")
    return HarmonicInput.from_peaks(
        list(map(float, freqs)), amplitudes=list(map(float, amps)),
        metadata={"material": material.name, "kind": material.kind,
                  "archetype": material.archetype},
    )
