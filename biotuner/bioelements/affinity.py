"""Graded biosignal → material affinity: how strongly does a signal resonate with a material?"""
from __future__ import annotations

import numpy as np
import pandas as pd

from biotuner.bioelements import units
from biotuner.bioelements.matching import _match_score


def material_affinity(peaks_hz, material, *, table: str = "air", top: int = 40,
                      tol_cents: float = 50.0, basis: str = "atom",
                      band=units.OPTICAL_BAND_ANGSTROM, balance: str = "recall") -> float:
    """How strongly a signal resonates with a material's composite spectrum.

    ``balance`` (see :func:`~biotuner.bioelements.matching._match_score`):
    ``"recall"`` (default) is the fraction of the material's intensity hit;
    ``"f1"`` balances that against the fraction of the signal explained — less
    biased toward line-sparse materials. 0 = no resonance."""
    spec = material.spectrum(table=table, top=top, basis=basis).normalise()
    return _match_score(peaks_hz, spec, tol_cents=tol_cents, band=band, balance=balance)


def match_materials(peaks_hz, materials=None, *, table: str = "air", top: int = 40,
                    tol_cents: float = 50.0, basis: str = "atom",
                    include_elements: bool = False, balance: str = "recall") -> pd.DataFrame:
    """Rank a biosignal against the material dictionary.

    ``include_elements=False`` (default) ranks only compounds/mixtures/structures —
    the "which **material** do I resonate with?" question — because bare elements
    have their own dense spectra and their own ranker (:func:`match_elements`), and
    would otherwise dominate. Pass ``include_elements=True`` to rank everything.
    """
    from biotuner.bioelements.materials import MATERIALS
    mats = MATERIALS if materials is None else materials
    peaks_hz = np.atleast_1d(np.asarray(peaks_hz, float))
    rows = []
    for name, mat in mats.items():
        if not include_elements and mat.kind == "element":
            continue
        rows.append({
            "material": name,
            "affinity": material_affinity(peaks_hz, mat, table=table, top=top,
                                          tol_cents=tol_cents, basis=basis, balance=balance),
            "kind": mat.kind,
            "archetype": mat.archetype or "",
        })
    return pd.DataFrame(rows).sort_values("affinity", ascending=False).reset_index(drop=True)
