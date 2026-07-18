"""
biotuner.biocolors
==================

Module type: Subpackage

Perceptual, metric-driven colour mapping for biosignals and tunings.

Turns a signal's spectral peaks -- or a tuning's ratios -- into palettes whose
colour axes are driven by swappable harmonic metrics, in a perceptually uniform
space, guaranteed inside the sRGB gamut.

Quick start
-----------
>>> from biotuner.biocolors import palette_from_signal
>>> pal = palette_from_signal(peaks, amps)
>>> pal.hex()
['#5d7a8c', '#8fa9a0', ...]
>>> pal.report()["min_deltaE"]
0.118

Layers
------
- :mod:`~biotuner.biocolors.color`        colour science; no biotuner imports
- :mod:`~biotuner.biocolors.descriptors`  signal -> :class:`Fingerprint`
- :mod:`~biotuner.biocolors.calibration`  percentile normalisation against a corpus
- :mod:`~biotuner.biocolors.mapping`      fingerprint + steps -> ``ColorSpec``
- :mod:`~biotuner.biocolors.palettes`     the public entry points
- :mod:`~biotuner.biocolors.legacy`       the original module, frozen

Backwards compatibility
-----------------------
Every name that lived in the old ``biotuner/biocolors.py`` is re-exported here,
so ``from biotuner.biocolors import audible2visible, wavelength_to_rgb`` keeps
working unchanged. Those live in :mod:`~biotuner.biocolors.legacy` now and are
deprecated in favour of :mod:`~biotuner.biocolors.color`.
"""

# --- legacy surface, unchanged ------------------------------------------- #
from biotuner.biocolors.legacy import (
    wavelength_to_rgb,
    scale2freqs,
    nm2Hz,
    Hz2nm,
    Hz2THz,
    THz2Hz,
    audible2visible,
    wavelength_to_frequency,
    viz_scale_colors,
    animate_colors,
)

# --- new API -------------------------------------------------------------- #
from biotuner.biocolors import color
from biotuner.biocolors.color import (
    srgb_to_oklab,
    oklab_to_srgb,
    srgb_to_oklch,
    oklch_to_srgb,
    max_chroma,
    cusp,
    gamut_map,
    deltaE_ok,
    simulate_cvd,
    wavelength_to_srgb,
    audible_to_nm,
)
from biotuner.biocolors.descriptors import (
    Fingerprint,
    SignalContext,
    descriptor,
    fingerprint,
    amps_scale_for,
    DESCRIPTORS,
    AMPS_SCALE_BY_PEAKS_FUNCTION,
)
from biotuner.biocolors.calibration import (
    Calibration,
    build_calibration,
    load_calibration,
)
from biotuner.biocolors.mapping import (
    ColorSpec, ColorAxes, mapping, MAPPINGS, COLORSPACES, register_colorspace,
)
from biotuner.biocolors.palettes import (
    Palette,
    palette_from_signal,
    palette_from_tuning,
    palette_from_biotuner,
    palette_from_raw,
    palette_set,
    LEVELS,
    dyad_field,
    consonance_spectrum,
    diversity_report,
    palette_report,
)

__all__ = [
    # legacy
    "wavelength_to_rgb", "scale2freqs", "nm2Hz", "Hz2nm", "Hz2THz", "THz2Hz",
    "audible2visible", "wavelength_to_frequency", "viz_scale_colors",
    "animate_colors",
    # colour layer
    "color",
    "srgb_to_oklab", "oklab_to_srgb", "srgb_to_oklch", "oklch_to_srgb",
    "max_chroma", "cusp", "gamut_map", "deltaE_ok", "simulate_cvd",
    "wavelength_to_srgb", "audible_to_nm",
    # descriptors
    "Fingerprint", "SignalContext", "descriptor", "fingerprint", "DESCRIPTORS",
    "amps_scale_for", "AMPS_SCALE_BY_PEAKS_FUNCTION",
    # calibration
    "Calibration", "build_calibration", "load_calibration",
    # mapping
    "ColorSpec", "ColorAxes", "mapping", "MAPPINGS", "COLORSPACES", "register_colorspace",
    # palettes
    "Palette", "palette_from_signal", "palette_from_tuning", "palette_from_biotuner",
    "palette_from_raw", "palette_set", "LEVELS",
    "dyad_field", "consonance_spectrum", "diversity_report", "palette_report",
]
