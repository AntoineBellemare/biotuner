"""The :class:`Spectrum` object — a weighted line list an element or a material emits.

A spectrum is a set of ``(wavelength Å, intensity, source-label)`` lines. The same
object represents an element leaf (one element's NIST lines) and a composite
material (the weighted superposition of its constituents), so composition is
closed: superposing spectra yields a spectrum.

Budget-normalisation (:meth:`Spectrum.normalise`) is this module's calibration
step — it rescales each source's lines so its *total* intensity is one unit of
"presence", removing the artefact that line-rich elements (Fe, O) otherwise
dominate a composite regardless of stoichiometry (see the architecture doc §4).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from biotuner.bioelements import tables, units


@dataclass
class Spectrum:
    wavelength: np.ndarray   # angstrom
    intensity: np.ndarray    # arbitrary units (NIST relative intensity)
    label: np.ndarray        # per-line source (element name), for dominance/audit
    name: str = ""

    def __post_init__(self):
        self.wavelength = np.asarray(self.wavelength, float)
        self.intensity = np.asarray(self.intensity, float)
        self.label = np.asarray(self.label, dtype=object)

    def __len__(self):
        return len(self.wavelength)

    def __repr__(self):
        d = self.dominant()
        dom = ", ".join(f"{k} {v*100:.0f}%" for k, v in list(d.items())[:3])
        return f"Spectrum({self.name!r}, {len(self)} lines, dominant: {dom})"

    # --- transforms ------------------------------------------------------- #
    def normalise(self) -> "Spectrum":
        """Budget-normalise so total intensity == 1 (unit of presence)."""
        s = float(self.intensity.sum())
        it = self.intensity / s if s > 0 else self.intensity
        return Spectrum(self.wavelength, it, self.label, self.name)

    def scaled(self, factor: float) -> "Spectrum":
        """A copy with intensities multiplied by ``factor`` (for weighting)."""
        return Spectrum(self.wavelength, self.intensity * float(factor), self.label, self.name)

    def select(self, *, top: int | None = None, threshold: float | None = None) -> "Spectrum":
        """Keep the strongest lines: top-N by intensity and/or above a threshold."""
        mask = np.ones(len(self), bool)
        if threshold is not None:
            mask &= self.intensity >= threshold
        idx = np.where(mask)[0]
        if top is not None and len(idx) > top:
            order = idx[np.argsort(self.intensity[idx])[::-1][:top]]
            idx = np.sort(order)
        return Spectrum(self.wavelength[idx], self.intensity[idx], self.label[idx], self.name)

    # --- readouts --------------------------------------------------------- #
    def to_hz(self) -> np.ndarray:
        """Line frequencies (Hz)."""
        return units.angstrom_to_hertz(self.wavelength)

    def fold_to_optical(self, band=units.OPTICAL_BAND_ANGSTROM) -> np.ndarray:
        """Every line's wavelength octave-folded into the optical band (Å)."""
        return np.array([units.fold_to_optical(w, is_hz=False, band=band) for w in self.wavelength])

    def dominant(self) -> dict:
        """Fraction of total intensity contributed by each source label."""
        if len(self) == 0:
            return {}
        s = pd.Series(self.intensity, index=self.label).groupby(level=0).sum()
        s = s.sort_values(ascending=False)
        tot = float(s.sum()) or 1.0
        return {str(k): float(v / tot) for k, v in s.items()}

    def as_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "wavelength": self.wavelength,
            "intensity": self.intensity,
            "source": self.label,
            "region": [units.spectrum_region(w) for w in self.wavelength],
        }).sort_values("wavelength").reset_index(drop=True)


def element_spectrum(element: str, *, table: str = "air",
                     top: int | None = None, threshold: float | None = None,
                     normalise: bool = False) -> Spectrum:
    """Build a leaf :class:`Spectrum` for one element from its NIST lines."""
    sub = tables.element_table(element, table)
    spec = Spectrum(sub["wavelength"].values, sub["intensity"].values,
                    np.full(len(sub), element, dtype=object), name=element)
    if top is not None or threshold is not None:
        spec = spec.select(top=top, threshold=threshold)
    return spec.normalise() if normalise else spec


def superpose(spectra, weights=None) -> Spectrum:
    """Weighted superposition of several spectra into one composite spectrum.

    Each spectrum is expected to be budget-normalised already (so weights carry
    the stoichiometry, not the line count). Concatenates the weighted line lists;
    line identities (labels) are preserved for dominance auditing.
    """
    spectra = list(spectra)
    if weights is None:
        weights = [1.0] * len(spectra)
    wls, its, lbls = [], [], []
    for sp, w in zip(spectra, weights):
        if len(sp) == 0:
            continue
        wls.append(sp.wavelength)
        its.append(sp.intensity * float(w))
        lbls.append(sp.label)
    if not wls:
        return Spectrum(np.array([]), np.array([]), np.array([], dtype=object))
    return Spectrum(np.concatenate(wls), np.concatenate(its), np.concatenate(lbls))
