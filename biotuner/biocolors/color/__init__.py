"""
biotuner.biocolors.color
========================

Module type: Subpackage

Colour science with **no biotuner imports**. Conversions, gamut boundary and
mapping, spectral rendering, perceptual difference and CVD simulation.

Kept free of harmonic machinery so it can be tested directly against
``colour-science`` and reused by anything that needs correct sRGB/OKLab work.
Everything is vectorized over a trailing ``(..., 3)`` channel axis.

Submodules: spaces, gamut, spectral, difference, palette.
"""

from biotuner.biocolors.color.spaces import (
    srgb_to_linear,
    linear_to_srgb,
    srgb_to_oklab,
    oklab_to_srgb,
    oklab_to_oklch,
    oklch_to_oklab,
    srgb_to_oklch,
    oklch_to_srgb,
    toe,
    toe_inv,
    oklab_to_oklrab,
    oklrab_to_oklab,
)
from biotuner.biocolors.color.gamut import (
    in_gamut,
    max_chroma,
    cusp,
    cusp_table,
    gamut_map,
    clip_report,
)
from biotuner.biocolors.color.spectral import (
    cie1931_xyz,
    wavelength_to_srgb,
    audible_to_nm,
    NM_MIN,
    NM_MAX,
    C_LIGHT,
)
from biotuner.biocolors.color.difference import (
    deltaE_ok,
    pairwise_deltaE,
    simulate_cvd,
    min_separation,
    CVD_KINDS,
)

__all__ = [
    # spaces
    "srgb_to_linear", "linear_to_srgb",
    "srgb_to_oklab", "oklab_to_srgb",
    "oklab_to_oklch", "oklch_to_oklab",
    "srgb_to_oklch", "oklch_to_srgb",
    "toe", "toe_inv", "oklab_to_oklrab", "oklrab_to_oklab",
    # gamut
    "in_gamut", "max_chroma", "cusp", "cusp_table", "gamut_map", "clip_report",
    # spectral
    "cie1931_xyz", "wavelength_to_srgb", "audible_to_nm",
    "NM_MIN", "NM_MAX", "C_LIGHT",
    # difference
    "deltaE_ok", "pairwise_deltaE", "simulate_cvd", "min_separation", "CVD_KINDS",
]
