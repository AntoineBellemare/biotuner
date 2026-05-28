"""biotuner.resonance.kernels_ratio — soft (and binary) n:m ratio gates.

Ratio kernels return a weight in [0, 1] for each (f_i, f_j) pair that gates the
phase-coupling matrix. Phase 1 ships the binary 5% legacy gate (with 1:1 fallback);
Phase 2 adds the soft Arnold-tongue and Stern-Brocot variants from plan §5.2.

Each kernel returns ``(W, N, M)`` where:
    W : (Ni, Nj) weights
    N, M : (Ni, Nj) int arrays giving the best (n, m) pair per cell

Reference for soft Arnold-tongue (Phase 2): Pikovsky, Rosenblum, Kurths 2001
*Synchronization* Ch. 7.
"""

import numpy as np

from biotuner.resonance.registry import register_ratio_kernel


def binary_nm_kernel(
    freqs_i: np.ndarray,
    freqs_j: np.ndarray,
    *,
    max_nm: int = 3,
    tolerance: float = 0.05,
    fallback_to_1_1: bool = True,
    **_unused,
):
    """Binary n:m gate (legacy): returns 1 for the best n:m match within tolerance, 0 otherwise.

    Mirrors the behavior of legacy :func:`biotuner.harmonic_spectrum.get_harmonic_ratio`:
    iterate over (n, m) with 1 <= n, m <= max_nm, pick the one minimizing
    ``|ratio - m/n| / (m/n)`` if any falls below ``tolerance``. If no match is found
    and ``fallback_to_1_1`` is True, the pair gets ``(W=1, N=1, M=1)``; otherwise
    ``(W=0, N=0, M=0)``.

    Returns
    -------
    (W, N, M) : tuple of (Ni, Nj) ndarrays
    """
    fi = np.asarray(freqs_i, dtype=np.float64).reshape(-1)
    fj = np.asarray(freqs_j, dtype=np.float64).reshape(-1)
    Ni, Nj = fi.size, fj.size
    W = np.zeros((Ni, Nj), dtype=np.float64)
    N = np.zeros((Ni, Nj), dtype=np.int64)
    M = np.zeros((Ni, Nj), dtype=np.int64)

    for i in range(Ni):
        if fi[i] == 0:
            continue
        for j in range(Nj):
            ratio = fj[j] / fi[i]
            best_match = None
            min_error = float("inf")
            for n in range(1, max_nm + 1):
                for m in range(1, max_nm + 1):
                    expected = m / n
                    err = abs(ratio - expected) / expected
                    if err < tolerance and err < min_error:
                        min_error = err
                        best_match = (n, m)
            if best_match is not None:
                W[i, j] = 1.0
                N[i, j] = best_match[0]
                M[i, j] = best_match[1]
            elif fallback_to_1_1:
                W[i, j] = 1.0
                N[i, j] = 1
                M[i, j] = 1
    return W, N, M


register_ratio_kernel("binary", binary_nm_kernel)
