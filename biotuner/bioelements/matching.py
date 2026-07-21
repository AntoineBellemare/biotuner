"""Relative-tolerance matching of biosignal peaks to element/material lines.

The original matcher used an *absolute* wavelength tolerance, meaningless across a
table spanning 56–46 525 Å (a plausible peak set returned 0 matches). Matching here
is **relative** — a musical-cents window — the same discipline the rest of biotuner
uses for ratios. A peak matches a line when the two, folded into a common band, sit
within ``tol_cents`` of each other.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from biotuner.bioelements import units
from biotuner.bioelements.spectrum import Spectrum, element_spectrum
from biotuner.bioelements import tables


def cents(w_a, w_b) -> np.ndarray:
    """Signed interval, in cents, between two wavelengths (or arrays)."""
    return 1200.0 * np.log2(np.asarray(w_a, float) / np.asarray(w_b, float))


def match_lines(peaks_hz, spectrum: Spectrum, *, tol_cents: float = 50.0,
                band=units.OPTICAL_BAND_ANGSTROM) -> pd.DataFrame:
    """Match biosignal peaks (Hz) to a spectrum's lines within ``tol_cents``.

    Each peak is octave-folded into ``band``; a spectrum line matches if it lies
    within ``tol_cents`` of the folded peak (lines are folded the same way, so
    the comparison is octave-invariant). Returns one row per (peak, line) hit.
    """
    if len(spectrum) == 0:
        return pd.DataFrame(columns=["peak_hz", "folded_wl", "line_wl", "cents", "intensity", "source"])
    line_wl = spectrum.fold_to_optical(band)
    rows = []
    for pk in np.atleast_1d(np.asarray(peaks_hz, float)):
        w = units.fold_to_optical(pk, is_hz=True, band=band)
        dc = cents(line_wl, w)
        hit = np.abs(dc) <= tol_cents
        for k in np.where(hit)[0]:
            rows.append({
                "peak_hz": float(pk), "folded_wl": float(w),
                "line_wl": float(line_wl[k]), "cents": float(dc[k]),
                "intensity": float(spectrum.intensity[k]),
                "source": str(spectrum.label[k]),
            })
    return pd.DataFrame(rows)


def _match_score(peaks_hz, spectrum: Spectrum, *, tol_cents: float, band) -> float:
    """Fraction of a spectrum's (budget-normalised) intensity that a signal hits."""
    if len(spectrum) == 0:
        return 0.0
    line_wl = spectrum.fold_to_optical(band)
    folded = np.array([units.fold_to_optical(pk, is_hz=True, band=band)
                       for pk in np.atleast_1d(np.asarray(peaks_hz, float))])
    hit = np.zeros(len(spectrum), bool)
    for w in folded:
        hit |= np.abs(cents(line_wl, w)) <= tol_cents
    inten = spectrum.intensity
    return float(inten[hit].sum() / (inten.sum() + 1e-12))


def match_elements(peaks_hz, *, table: str = "air", top: int = 40,
                   tol_cents: float = 50.0, band=units.OPTICAL_BAND_ANGSTROM,
                   min_score: float = 0.0) -> pd.DataFrame:
    """Rank every element by how strongly a signal resonates with its lines.

    Score = fraction of the element's budget-normalised line intensity that falls
    within ``tol_cents`` of a folded signal peak. Returns a sorted DataFrame
    (element, score, category, n_hits).
    """
    peaks_hz = np.atleast_1d(np.asarray(peaks_hz, float))
    rows = []
    for elem in tables.available_elements(table):
        spec = element_spectrum(elem, table=table, top=top, normalise=True)
        score = _match_score(peaks_hz, spec, tol_cents=tol_cents, band=band)
        if score >= min_score:
            n_hits = len(match_lines(peaks_hz, spec, tol_cents=tol_cents, band=band))
            rows.append({"element": elem, "score": score,
                         "category": tables.element_category(elem, table),
                         "n_hits": n_hits})
    out = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    return out
