"""biotuner.resonance.kernels_ratio — n:m ratio gates for phase coupling.

Ratio kernels decide, for each frequency pair (f_i, f_j), which integer ratio
(n, m) to test for phase coupling and how much weight to give the result.
The orchestrator's PC pipeline calls these to drive ``build_pairwise_coupling_matrix``.

Each kernel returns ``(W, N, M)`` where:
    W : (Ni, Nj) weights in [0, 1]
    N, M : (Ni, Nj) int arrays giving the (n, m) pair to test per cell
           Convention: n * f_i ≈ m * f_j  (mode-lock condition)

Registered kernels:

  fraction (DEFAULT)
    For each pair, computes ``Fraction(f_j / f_i).limit_denominator(max_denom)``
    to get the EXACT closest rational ratio. Works for any frequency pair —
    e.g. for (10 Hz, 17 Hz) returns (n=10, m=17) at max_denom>=17, testing the
    actual 10:17 mode-lock. Weight W = exp(-beta * log2(n*m)) so high-order
    ratios get small weight while simple ones (like 1:2) get W close to 1.

  binary
    Legacy gate (preserves bit-exact reproduction of pre-refactor pipeline):
    tries (n, m) pairs with 1 ≤ n, m ≤ max_nm (default 3) and picks the best
    match within tolerance. Returns W=1 if ANY match found, W=0 otherwise
    (or W=1 at (1,1) if fallback_to_1_1=True). Misses coupling at any ratio
    outside the small preset table — use 'fraction' instead for new analyses.

Phase 2 will add 'arnold_tongue' (soft Gaussian membership of Arnold tongues,
Pikovsky-Rosenblum-Kurths 2001) and 'stern_brocot' (depth-weighted complexity).
"""

import numpy as np
from fractions import Fraction

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


def fraction_kernel(
    freqs_i: np.ndarray,
    freqs_j: np.ndarray,
    *,
    max_denom: int = 16,
    beta: float = 1.0,
    **_unused,
):
    """For each pair, picks (n, m) as the exact closest rational of f_j/f_i.

    For ANY frequency pair this returns a meaningful (n, m) that tests the
    actual mode-lock condition ``n * f_i ≈ m * f_j``. The weight W penalizes
    high-order ratios via Tenney height (``log2(n * m)``):

        W[i, j] = exp(-beta * log2(n * m))

    Examples (with max_denom=16):
        (10, 20) → (1, 2), W = 0.5             — simple octave, high weight
        (10, 15) → (2, 3), W = 0.167           — perfect fifth, moderate
        (10, 17) → (10, 17), W ≈ 0.0059        — complex ratio, low weight
        (10, 14.14) → (5, 7), W = 0.0286       — closest simple approximation

    Compare to the legacy ``binary_nm_kernel`` which only tests (n, m) ≤
    max_nm=3 and falls back to a meaningless 1:1 for any pair outside that
    preset table.

    Parameters
    ----------
    freqs_i, freqs_j : 1-D arrays
    max_denom : int, default=16
        Maximum denominator passed to ``Fraction.limit_denominator``. Larger
        values give more exact ratios; smaller values force simpler
        approximations. The legacy ``cross_frequency_rrci`` used 16 as well.
    beta : float, default=1.0
        Tenney-height complexity penalty exponent. beta=0 gives W=1 everywhere
        (no complexity penalty); beta=1 gives a moderate penalty matching
        typical musicological assumptions.

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
            if fj[j] == 0:
                continue
            frac = Fraction(float(fj[j] / fi[i])).limit_denominator(max_denom)
            # Convention: n * f_i ≈ m * f_j, so n / m = f_j / f_i = frac.
            # frac = numerator / denominator, so n = numerator, m = denominator.
            n = frac.numerator
            m = frac.denominator
            if n <= 0 or m <= 0:
                continue
            N[i, j] = n
            M[i, j] = m
            tenney = np.log2(n * m)
            W[i, j] = float(np.exp(-beta * tenney))
    return W, N, M


register_ratio_kernel("binary", binary_nm_kernel)
register_ratio_kernel("fraction", fraction_kernel)
