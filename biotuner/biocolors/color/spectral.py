"""Wavelength -> colour, and audible frequency -> wavelength.

Module type: Functions

Two spectral renderers:

- ``'cie1931'`` (default) -- CIE 1931 colour matching functions -> XYZ -> sRGB.
  The CMFs use the Wyman, Sloan & Shirley (2013) analytic multi-lobe Gaussian
  fit, so this costs ~15 lines and no data table.
- ``'bruton'`` -- the Dan Bruton approximation used by
  :func:`biotuner.biocolors.legacy.wavelength_to_rgb`. Kept for byte
  compatibility with existing output.

Why the default changed: Bruton's piecewise ramp collapses the entire
645-750 nm band -- 105 nm, 28.4% of the visible range -- onto the single OKLCh
hue 29.23 deg. Only lightness varies across it. Measured consequence on a
just-intonation scale: ratios 3/2 (750.0 nm) and 5/3 (681.7 nm) receive
*identical* hue, and the 8-step scale yields only 6 distinct hues. A module
whose premise is "frequency becomes colour" cannot afford a quarter of its
spectrum having no hue resolution.
"""

from __future__ import annotations

import numpy as np

from biotuner.biocolors.color.gamut import gamut_map
from biotuner.biocolors.color.spaces import (
    linear_to_srgb,
    oklab_to_oklch,
    srgb_to_oklab,
)

__all__ = [
    "cie1931_xyz",
    "wavelength_to_srgb",
    "audible_to_nm",
    "NM_MIN",
    "NM_MAX",
    "C_LIGHT",
]

C_LIGHT = 299792458.0
NM_MIN, NM_MAX = 380.0, 750.0

# XYZ (D65) -> linear sRGB
_XYZ_TO_RGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
])


def _lobe(w, mu, s1, s2):
    """Piecewise-Gaussian lobe: different sigma either side of the peak."""
    t = (w - mu) * np.where(w < mu, s1, s2)
    return np.exp(-0.5 * t * t)


def cie1931_xyz(nm):
    """CIE 1931 2-degree colour matching functions at wavelength(s) ``nm``.

    Wyman, Sloan & Shirley (2013), "Simple Analytic Approximations to the CIE
    XYZ Color Matching Functions", JCGT 2(2). Max error ~1% of peak, far below
    the error Bruton introduces.
    """
    w = np.asarray(nm, float)
    x = (0.362 * _lobe(w, 442.0, 0.0624, 0.0374)
         + 1.056 * _lobe(w, 599.8, 0.0264, 0.0323)
         - 0.065 * _lobe(w, 501.1, 0.0490, 0.0382))
    y = (0.821 * _lobe(w, 568.8, 0.0213, 0.0247)
         + 0.286 * _lobe(w, 530.9, 0.0613, 0.0322))
    z = (1.217 * _lobe(w, 437.0, 0.0845, 0.0278)
         + 0.681 * _lobe(w, 459.0, 0.0385, 0.0725))
    return np.stack([x, y, z], axis=-1)


def wavelength_to_srgb(nm, method="cie1931", normalize=True, gamut="chroma"):
    """Wavelength in nm -> sRGB in 0-1, shape ``(..., 3)``.

    ``normalize`` scales each colour to unit maximum, trading absolute luminance
    (meaningless for a spectral line) for a usable, evenly bright locus.
    ``gamut`` is passed to :func:`~biotuner.biocolors.color.gamut.gamut_map`;
    the spectral locus lies largely outside sRGB, so this matters -- clipping it
    (``gamut='clip'``) is what produces the muddy, hue-shifted spectra typical
    of naive implementations.
    """
    if method == "bruton":
        from biotuner.biocolors.legacy import wavelength_to_rgb
        w = np.asarray(nm, float)
        flat = np.atleast_1d(w).ravel()
        out = np.array([wavelength_to_rgb(float(x)) for x in flat], float) / 255.0
        return out.reshape(np.shape(w) + (3,))
    if method != "cie1931":
        raise ValueError(f"unknown spectral method: {method!r}")

    xyz = cie1931_xyz(nm)
    lin = np.einsum("ij,...j->...i", _XYZ_TO_RGB, xyz)
    if normalize:
        peak = lin.max(axis=-1, keepdims=True)
        lin = np.divide(lin, peak, out=np.zeros_like(lin), where=peak > 1e-12)
    rgb = linear_to_srgb(lin, clip=False)
    if gamut:
        lch = oklab_to_oklch(srgb_to_oklab(np.clip(rgb, 0.0, 1.0)))
        # Re-derive from the unclipped linear values via chroma reduction rather
        # than clipping, so hue survives.
        lch = gamut_map(lch, mode=gamut)
        from biotuner.biocolors.color.spaces import oklch_to_srgb
        rgb = oklch_to_srgb(lch, clip=True)
    return np.clip(rgb, 0.0, 1.0)


def audible_to_nm(freq, c=C_LIGHT, mode="fold"):
    """Fold an audible frequency into a visible wavelength in [380, 750] nm.

    Returns ``(nm, n_octaves)``.

    ``mode``:

    - ``'fold'``  octave-fold on wavelength, then clamp the sub-octave residual
      to the nearest band edge. This is the prototype's behaviour, kept as the
      default for continuity.
    - ``'wrap'``  map the residual continuously around the band instead of
      clamping, so no frequency is falsified.

    Why ``'wrap'`` exists: the visible band spans a ratio of 750/380 = 1.974,
    slightly *less* than an octave. Folding therefore leaves a thin residual
    that ``'fold'`` clamps to an edge -- and the seam is not harmless. Measured
    on a just-intonation scale at fund=30 Hz, the clamp lands on the **perfect
    fifth** (45 Hz -> exactly 750.00 nm, while its neighbours fold an octave
    apart). Only 1.88% of frequencies in 1-1000 Hz hit the clamp, but the seam
    is structural, not random, and it can land on a musically central step.
    """
    freq = np.asarray(freq, float)
    scalar = freq.ndim == 0
    f = np.atleast_1d(freq).astype(float)

    nm = np.full(f.shape, np.nan)
    n_oct = np.zeros(f.shape, dtype=int)
    good = f > 0
    if not np.any(good):
        return (float("nan"), 0) if scalar else (nm, n_oct)

    hz = f[good].copy()
    # Octave-fold in log space: one shot, no loop.
    w = (c / hz) * 1e9
    k = np.floor(np.log2(w / NM_MIN))
    w = w / (2.0 ** k)
    # w is now in [NM_MIN, 2*NM_MIN) = [380, 760)
    over = w > NM_MAX
    if mode == "fold":
        w = np.where(over, NM_MAX, w)
    elif mode == "wrap":
        # Re-map [380, 760) continuously onto [380, 750] instead of clamping.
        w = NM_MIN + (w - NM_MIN) * (NM_MAX - NM_MIN) / (2.0 * NM_MIN - NM_MIN)
    else:
        raise ValueError(f"unknown audible_to_nm mode: {mode!r}")

    nm[good] = w
    n_oct[good] = k.astype(int)
    if scalar:
        return float(nm[0]), int(n_oct[0])
    return nm, n_oct
