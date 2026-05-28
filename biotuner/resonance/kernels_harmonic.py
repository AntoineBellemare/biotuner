"""biotuner.resonance.kernels_harmonic — harmonic similarity kernels.

Each kernel takes two frequency arrays and returns an ``(len(freqs_i), len(freqs_j))``
similarity matrix. Phase 1 ships ``harmsim`` and ``subharm_tension`` (bit-equivalent
ports of the legacy :func:`biotuner.harmonic_spectrum.harmonicity_matrices` formulas).
Phase 2 adds ``sethares``, ``stolzenburg``, ``harmonic_entropy``; Phase 3 adds ``hopf``
and ``lorentzian``.

References:
    Dyad similarity / harmsim: biotuner native; see :func:`biotuner.metrics.dyad_similarity`.
    Subharmonic tension: see :func:`biotuner.metrics.compute_subharmonic_tension`.
"""

import numpy as np

from biotuner.metrics import dyad_similarity, compute_subharmonic_tension
from biotuner.resonance.registry import register_harmonic_kernel


def kernel_harmsim(
    freqs_i: np.ndarray,
    freqs_j: np.ndarray,
    *,
    diagonal: float = None,
    **_unused,
) -> np.ndarray:
    """Harmonic similarity matrix: ``dyad_similarity(f_i / f_j)``.

    Bit-equivalent to the legacy :func:`biotuner.harmonic_spectrum.harmonicity_matrices`
    branch for ``metric='harmsim'``. ``f_j == 0`` entries are zero; otherwise the raw
    (un-reduced) ratio ``f_i / f_j`` is passed to ``dyad_similarity`` (which performs
    its own reduction internally).
    """
    fi = np.asarray(freqs_i, dtype=np.float64).reshape(-1)
    fj = np.asarray(freqs_j, dtype=np.float64).reshape(-1)
    M = np.zeros((fi.size, fj.size), dtype=np.float64)
    for i, f1 in enumerate(fi):
        for j, f2 in enumerate(fj):
            if f2 != 0:
                M[i, j] = dyad_similarity(f1 / f2)
    if diagonal is not None and fi.size == fj.size and np.array_equal(fi, fj):
        np.fill_diagonal(M, diagonal)
    return M


def kernel_subharm_tension(
    freqs_i: np.ndarray,
    freqs_j: np.ndarray,
    *,
    n_harms: int = 10,
    delta_lim: float = 20,
    min_notes: int = 2,
    diagonal: float = None,
    **_unused,
) -> np.ndarray:
    """1 - subharmonic_tension for each (f_i, f_j) pair.

    Bit-equivalent to the legacy ``harmonicity_matrices`` branch for
    ``metric='subharm_tension'``.
    """
    fi = np.asarray(freqs_i, dtype=np.float64).reshape(-1)
    fj = np.asarray(freqs_j, dtype=np.float64).reshape(-1)
    M = np.zeros((fi.size, fj.size), dtype=np.float64)
    for i, f1 in enumerate(fi):
        for j, f2 in enumerate(fj):
            if f2 == 0:
                continue
            _, _, subharm, _ = compute_subharmonic_tension(
                [f1, f2], n_harmonics=n_harms, delta_lim=delta_lim, min_notes=min_notes
            )
            # subharm[0] is occasionally a sentinel string from
            # compute_subharmonic_tension when no valid tension can be
            # computed; treat as zero similarity (cell will be 1.0 since
            # 1 - 0 = 1, matching the "no information" baseline).
            try:
                M[i, j] = 1.0 - float(subharm[0])
            except (TypeError, ValueError):
                M[i, j] = 0.0
    if diagonal is not None and fi.size == fj.size and np.array_equal(fi, fj):
        np.fill_diagonal(M, diagonal)
    return M


register_harmonic_kernel("harmsim", kernel_harmsim)
register_harmonic_kernel("subharm_tension", kernel_subharm_tension)
