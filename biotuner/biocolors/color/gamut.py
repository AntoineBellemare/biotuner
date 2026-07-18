"""sRGB gamut boundary and gamut mapping in OKLCh.

Module type: Functions

The sRGB chroma ceiling varies by more than 5x across the OKLCh hue circle:
measured, it runs from 0.048 (L=0.90, h=270) to 0.296 (L=0.60, h=315). A fixed
absolute chroma range therefore cannot be right everywhere -- it is several
times over budget at some hues and leaves a quarter of the available chroma
unused at others. Requesting chroma as a *fraction of the local ceiling*
(:func:`max_chroma`) is what keeps palettes both in gamut and vivid.

The prototype's failure mode this module exists to prevent: it clipped RGB
inside its conversion before testing for out-of-gamut, so its chroma-reduction
search never ran and colours were silently clipped instead -- which shifts hue
and washes saturated colours toward white. (That, not "hue 0 is the magenta
axis", is why its bright reds came out pink: at L=0.88 the ceiling at red's hue
is ~0.06 while it was requesting 0.22.)

Gamut triangle approximation after Björn Ottosson (2020).
"""

from __future__ import annotations

import numpy as np

from biotuner.biocolors.color.spaces import (
    oklab_to_srgb,
    oklch_to_oklab,
    oklch_to_srgb,
)

__all__ = ["in_gamut", "max_chroma", "cusp", "gamut_map", "cusp_table"]

_EPS = 1e-9


def in_gamut(lch, tol=1e-6):
    """True where an OKLCh colour lies inside sRGB. Shape ``(...,)``."""
    rgb = oklch_to_srgb(lch, clip=False)
    return (rgb.min(axis=-1) >= -tol) & (rgb.max(axis=-1) <= 1.0 + tol)


def max_chroma(L, h, hi=0.5, iters=32):
    """Largest in-gamut chroma at lightness ``L`` and hue ``h`` (degrees).

    Vectorized bisection on the true sRGB boundary. Exact to ``hi / 2**iters``
    (~1e-10 by default). ``L`` and ``h`` broadcast against each other.

    This is the "cusp ceiling" that :func:`gamut_map` and cusp-relative chroma
    ranges are measured against.
    """
    L = np.asarray(L, float)
    h = np.asarray(h, float)
    L, h = np.broadcast_arrays(L, h)
    lo = np.zeros(L.shape)
    hi = np.full(L.shape, float(hi))
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        lch = np.stack([L, mid, h], axis=-1)
        ok = in_gamut(lch, tol=_EPS)
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)
    return lo


def cusp(h, n_L=256):
    """The (L, C) cusp: the most chromatic point sRGB allows at hue ``h``.

    Returns ``(L_cusp, C_cusp)``, each shaped like ``h``. Found by scanning
    lightness and taking the argmax of :func:`max_chroma`.
    """
    h = np.asarray(h, float)
    Ls = np.linspace(0.0, 1.0, n_L)
    # (n_L, ...) grid of ceilings
    grid = max_chroma(Ls.reshape((-1,) + (1,) * h.ndim), h[None, ...])
    idx = np.argmax(grid, axis=0)
    C_cusp = np.take_along_axis(grid, idx[None, ...], axis=0)[0]
    L_cusp = Ls[idx]
    return L_cusp, C_cusp


def cusp_table(n_hue=360, n_L=256):
    """Precomputed cusp per integer hue. Returns ``(hues, L_cusp, C_cusp)``.

    Cache this once if you are mapping many colours; the bisection is exact but
    not free.
    """
    hues = np.arange(n_hue, dtype=float)
    L_c, C_c = cusp(hues, n_L=n_L)
    return hues, L_c, C_c


def gamut_map(lch, mode="chroma", tol=1e-6):
    """Bring OKLCh colours into the sRGB gamut.

    ``mode``:

    - ``'chroma'``  reduce chroma, hold L and h. The perceptually honest
      default: hue is preserved exactly and lightness does not drift.
    - ``'clip'``    clip RGB directly. Fast, shifts hue. Provided only to
      reproduce naive behaviour for comparison.

    Out-of-gamut colours are pulled to the boundary; in-gamut colours are
    returned untouched. Shape-preserving.
    """
    lch = np.asarray(lch, float)
    if mode == "clip":
        from biotuner.biocolors.color.spaces import srgb_to_oklch
        return srgb_to_oklch(oklch_to_srgb(lch, clip=True))
    if mode != "chroma":
        raise ValueError(f"unknown gamut_map mode: {mode!r}")

    L, C, h = lch[..., 0], lch[..., 1], lch[..., 2]
    ceiling = max_chroma(L, h)
    return np.stack([L, np.minimum(C, ceiling), h], axis=-1)


def clip_report(lch):
    """How much chroma a gamut map would remove. Diagnostic, not a transform.

    Returns a dict with the fraction of colours out of gamut and the worst and
    mean chroma excess. Used by ``palette_report`` so that "this palette is in
    gamut" is a measured claim rather than an assumption.
    """
    lch = np.asarray(lch, float)
    L, C, h = lch[..., 0], lch[..., 1], lch[..., 2]
    ceiling = max_chroma(L, h)
    excess = np.maximum(C - ceiling, 0.0)
    return {
        "frac_out_of_gamut": float(np.mean(excess > 1e-6)),
        "max_excess": float(excess.max()) if excess.size else 0.0,
        "mean_excess": float(excess.mean()) if excess.size else 0.0,
        "mean_headroom": float(np.mean(ceiling - C)),
    }
