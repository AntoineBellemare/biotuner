"""biotuner.resonance.kernels_ratio — n:m ratio gates for phase coupling.

Ratio kernels decide, for each frequency pair (f_i, f_j), which integer ratio
(n, m) to test for phase coupling and how much weight to give the result.
The orchestrator's PC pipeline calls these to drive ``build_pairwise_coupling_matrix``.

Each kernel returns ``(W, N, M)`` where:
    W : (Ni, Nj) weights in [0, 1]
    N, M : (Ni, Nj) int arrays giving the (n, m) pair to test per cell

Convention (shared by all registered kernels)
---------------------------------------------
All ratio kernels return ``(n, m)`` such that
``ratio = f_j / f_i ≈ m / n``. This is the legacy biotuner convention,
preserved across all kernels for consistency. To get a mathematically
correct n:m phase-locking test (Tass 1998 form ``n·φ_i − m·φ_j = const``),
ALWAYS pair these kernels with the coupling metric ``nm_plv_canonical``
(which swaps internally). Pairing with the raw ``nm_plv`` keeps the legacy
behavior preserved for bit-exact paper reproduction.

Quick rule
----------
* New analyses: ``coupling_metric='nm_plv_canonical'`` with any ratio kernel.
* Paper reproduction: ``coupling_metric='nm_plv'`` with ``ratio_kernel='binary'``.

Registered kernels
------------------

  fraction
    For each pair, computes ``Fraction(f_j / f_i).limit_denominator(max_denom)``
    to get the EXACT closest rational. Works for any frequency pair —
    e.g. (10 Hz, 17 Hz) gives (n=10, m=17), testing the actual 10:17 mode-lock.
    Weight W = exp(-beta * log2(n*m)) penalizes high-order ratios.

  binary (DEFAULT)
    Legacy gate (preserves bit-exact reproduction): tries (n, m) pairs with
    1 ≤ n, m ≤ max_nm (default 3) and picks the best match within tolerance.
    Returns W=1 if any match found, W=0 otherwise (or W=1 at (1,1) if
    ``fallback_to_1_1=True``). Misses coupling at any ratio outside the small
    preset table — use 'fraction' for new analyses.

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

    For ANY frequency pair this returns a meaningful (n, m) — the legacy
    ``binary_nm_kernel`` could only handle ratios up to max_nm=3.

    Convention
    ----------
    Returns ``(n, m)`` such that ``ratio = f_j / f_i ≈ m / n`` — the SAME
    convention as :func:`binary_nm_kernel`. This is the legacy biotuner
    convention; to get a mathematically correct n:m phase-locking test,
    pair this kernel with ``coupling_metric='nm_plv_canonical'`` (which
    swaps internally to apply the Tass 1998 convention).

    The weight W penalizes high-order ratios via Tenney height:

        W[i, j] = exp(-beta * log2(n * m))

    Examples (max_denom=16, beta=1.0):
        (10, 20) → (n=1, m=2),  W = 0.368   octave
        (10, 15) → (n=2, m=3),  W = 0.075   perfect fifth
        (10, 17) → (n=10, m=17), W ≈ 6e-4   complex, but exact test
        (10, 14.14) → (n=12, m=17), W ≈ 5e-4  closest rational to √2

    The recommended pairing for new analyses is::

        ResonanceConfig(
            ratio_kernel='fraction',
            coupling_metric='nm_plv_canonical',   # not just 'nm_plv'!
        )

    Parameters
    ----------
    freqs_i, freqs_j : 1-D arrays
    max_denom : int, default=16
        Maximum denominator passed to ``Fraction.limit_denominator``. Larger
        values give more exact ratios; smaller values force simpler
        approximations. The legacy ``cross_frequency_rrci`` used 16 as well.
    beta : float, default=1.0
        Tenney-height complexity penalty exponent. beta=0 gives W=1 everywhere;
        beta=1 matches typical musicological assumptions.

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
            # Convention: ratio = f_j / f_i = m / n (matches binary_nm_kernel).
            # frac = numerator / denominator, so m = numerator, n = denominator.
            m = frac.numerator
            n = frac.denominator
            if n <= 0 or m <= 0:
                continue
            N[i, j] = n
            M[i, j] = m
            tenney = np.log2(n * m)
            W[i, j] = float(np.exp(-beta * tenney))
    return W, N, M


register_ratio_kernel("binary", binary_nm_kernel)
register_ratio_kernel("fraction", fraction_kernel)
