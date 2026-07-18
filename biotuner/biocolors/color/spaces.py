"""sRGB <-> linear <-> OKLab <-> OKLCh conversions.

Module type: Functions

Every function here is vectorized over a trailing channel axis: input of shape
``(..., 3)`` returns shape ``(..., 3)``. A single colour is just ``(3,)``.

This module imports nothing from biotuner. It is pure colour science and is
tested against ``colour-science`` where available.

Two deliberate departures from the ``biocolors_plus`` prototype:

1. **Clipping is not fused into the conversion.** ``oklab_to_srgb`` returns
   unclipped values by default so that callers can *detect* out-of-gamut
   colours. The prototype clipped inside its linear->sRGB step, which made its
   own out-of-gamut test (``rgb.min() < 0 or rgb.max() > 1``) unreachable: the
   values it tested had already been clipped into range. Its gamut-fitting
   binary search therefore never executed and every out-of-gamut colour was
   silently clipped, shifting hue. See :mod:`biotuner.biocolors.color.gamut`.

2. **Channel unpacking is done on the last axis.** The prototype used
   ``r, g, b = arr``, which unpacks the *first* axis: for an ``(N, 3)`` array it
   read pixels as channels, returning silently wrong values for ``N == 3`` and
   raising ``ValueError`` for any other ``N``.

OKLab after Björn Ottosson (2020). The Oklrab toe is his 2023 addendum.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "srgb_to_linear",
    "linear_to_srgb",
    "srgb_to_oklab",
    "oklab_to_srgb",
    "oklab_to_oklch",
    "oklch_to_oklab",
    "srgb_to_oklch",
    "oklch_to_srgb",
    "toe",
    "toe_inv",
    "oklab_to_oklrab",
    "oklrab_to_oklab",
]

# Linear sRGB -> LMS (Ottosson). Applied along the last axis.
_M1 = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
])
# LMS' (cube-rooted) -> OKLab
_M2 = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
])
_M1_INV = np.linalg.inv(_M1)
_M2_INV = np.linalg.inv(_M2)


def _apply(matrix, arr):
    """Apply a 3x3 matrix along the last axis of ``arr``."""
    return np.einsum("ij,...j->...i", matrix, arr)


def srgb_to_linear(rgb):
    """sRGB (0-1) -> linear sRGB. Handles out-of-range input symmetrically."""
    c = np.asarray(rgb, float)
    sign = np.sign(c)
    a = np.abs(c)
    return sign * np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(lin, clip=False):
    """Linear sRGB -> sRGB (0-1).

    ``clip=False`` (the default) preserves out-of-range values so callers can
    detect out-of-gamut colours. Pass ``clip=True`` only at the very end of a
    pipeline, after gamut mapping.
    """
    c = np.asarray(lin, float)
    sign = np.sign(c)
    a = np.abs(c)
    out = sign * np.where(a <= 0.0031308, 12.92 * a, 1.055 * (a ** (1 / 2.4)) - 0.055)
    return np.clip(out, 0.0, 1.0) if clip else out


def srgb_to_oklab(rgb):
    """sRGB (0-1, shape ``(..., 3)``) -> OKLab (shape ``(..., 3)``)."""
    lin = srgb_to_linear(rgb)
    lms = _apply(_M1, lin)
    lms_ = np.cbrt(lms)
    return _apply(_M2, lms_)


def oklab_to_srgb(lab, clip=False):
    """OKLab (shape ``(..., 3)``) -> sRGB (0-1).

    Returns unclipped values by default; ``rgb.min() < 0`` or ``rgb.max() > 1``
    then correctly signals an out-of-gamut colour.
    """
    lms_ = _apply(_M2_INV, np.asarray(lab, float))
    lms = lms_ ** 3
    lin = _apply(_M1_INV, lms)
    return linear_to_srgb(lin, clip=clip)


def oklab_to_oklch(lab):
    """OKLab -> OKLCh. Hue in degrees, ``[0, 360)``."""
    lab = np.asarray(lab, float)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    C = np.hypot(a, b)
    h = np.degrees(np.arctan2(b, a)) % 360.0
    return np.stack([L, C, h], axis=-1)


def oklch_to_oklab(lch):
    """OKLCh (hue in degrees) -> OKLab."""
    lch = np.asarray(lch, float)
    L, C, h = lch[..., 0], lch[..., 1], lch[..., 2]
    hr = np.deg2rad(h)
    return np.stack([L, C * np.cos(hr), C * np.sin(hr)], axis=-1)


def srgb_to_oklch(rgb):
    """sRGB (0-1) -> OKLCh."""
    return oklab_to_oklch(srgb_to_oklab(rgb))


def oklch_to_srgb(lch, clip=False):
    """OKLCh -> sRGB (0-1). Unclipped by default; gamut-map first."""
    return oklab_to_srgb(oklch_to_oklab(lch), clip=clip)


# --------------------------------------------------------------------------- #
# Oklrab: a lightness estimate that matches CIE L* far better than OKLab's L.
# Use L_r wherever a human reasons about lightness (ranges, ramps); convert to
# L before doing colour maths. Ottosson (2023).
# --------------------------------------------------------------------------- #
_K1, _K2 = 0.206, 0.03
_K3 = (1.0 + _K1) / (1.0 + _K2)


def toe(L):
    """OKLab L -> Oklrab L_r (perceptual lightness)."""
    L = np.asarray(L, float)
    return 0.5 * (_K3 * L - _K1 + np.sqrt((_K3 * L - _K1) ** 2 + 4 * _K2 * _K3 * L))


def toe_inv(Lr):
    """Oklrab L_r -> OKLab L."""
    Lr = np.asarray(Lr, float)
    return (Lr ** 2 + _K1 * Lr) / (_K3 * (Lr + _K2))


def oklab_to_oklrab(lab):
    """OKLab -> Oklrab (only the lightness channel changes)."""
    lab = np.asarray(lab, float).copy()
    lab[..., 0] = toe(lab[..., 0])
    return lab


def oklrab_to_oklab(lrab):
    """Oklrab -> OKLab."""
    lrab = np.asarray(lrab, float).copy()
    lrab[..., 0] = toe_inv(lrab[..., 0])
    return lrab
