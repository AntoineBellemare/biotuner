"""biotuner.resonance.coupling — phase coupling metrics and per-bin reducers.

Pairwise coupling metrics (arity 2) build a ``Φ[i,j]`` matrix that feeds the
per-bin phase-coupling spectrum. Higher-order metrics (arity 3, K, survey, state)
go in plan Phase 3 and run on a separate code path.

Phase 1 ships:
    - ``nm_plv(phase_i, phase_j, n, m)``     scalar n:m PLV
    - ``build_nm_plv_matrix(phase, freqs, ratio_kernel_fn, ...)``  full Φ matrix
    - ``reduce_matrix_to_spectrum(M, p, ..., legacy_self_pair=True)``  Φ → PC(f)

References:
    n:m PLV: Tass et al. 1998 *PRL* 81:3291; Palva, Palva, Kaila 2005 *J Neurosci* 25:3962.
"""

import numpy as np

from biotuner.resonance.registry import register_coupling_metric


def nm_plv(phase_i: np.ndarray, phase_j: np.ndarray, n: int, m: int) -> float:
    """Scalar n:m phase-locking value: ``| <exp(i*(n*φᵢ - m*φⱼ))> |``."""
    phase_diff = n * phase_i - m * phase_j
    return float(np.abs(np.mean(np.exp(1j * phase_diff))))


def build_nm_plv_matrix(
    phase: np.ndarray,
    freqs: np.ndarray,
    ratio_kernel_fn,
    ratio_kernel_params: dict,
) -> np.ndarray:
    """Build the ``(n_freqs, n_freqs)`` symmetric n:m PLV matrix.

    For each upper-triangular pair (i, j), consults ``ratio_kernel_fn(freqs[i:i+1],
    freqs[j:j+1], **ratio_kernel_params)`` to determine the best (n, m); when the
    ratio kernel returns ``W=0``, the pair gets PLV=0. With the legacy binary kernel
    and ``fallback_to_1_1=True``, every pair gets a value (n:m if matched, else 1:1).
    """
    n_freqs = len(freqs)
    M = np.zeros((n_freqs, n_freqs), dtype=np.float64)

    # Pre-compute (W, N, M) for all pairs at once
    W, N_mat, M_mat = ratio_kernel_fn(freqs, freqs, **ratio_kernel_params)

    for i in range(n_freqs):
        for j in range(i + 1, n_freqs):
            if W[i, j] <= 0:
                continue
            n, m = int(N_mat[i, j]), int(M_mat[i, j])
            plv = nm_plv(phase[i], phase[j], n, m)
            # Weight by the ratio-kernel membership (binary kernel: W is 1.0; soft
            # kernels: PLV is scaled by Arnold-tongue membership).
            value = W[i, j] * plv
            M[i, j] = value
            M[j, i] = value
    return M


def reduce_matrix_to_spectrum(
    matrix: np.ndarray,
    psd_prob: np.ndarray,
    *,
    normalize: bool = True,
    legacy_self_pair_subtract: bool = True,
    alpha_self: float = 1.0,
    alpha_partner: float = 1.0,
) -> np.ndarray:
    """Reduce an N×N similarity/coupling matrix to a length-N per-bin spectrum.

    Two reduction modes:

    legacy_self_pair_subtract=True (DEFAULT, matches legacy compute_phase_spectrum
    and compute_harmonic_power):
        v[i] = p_i * Σ_j (M[i,j] * p_j)  -  M[i,i] * p_i^2

    legacy_self_pair_subtract=False (plan §A.4 — recommended for new code):
        v[i] = p_i^alpha_self * Σ_{j ≠ i} (M[i,j] * p_j^alpha_partner)
        Off-diagonal mask, no subtraction artifact.

    Parameters
    ----------
    matrix : (N, N) similarity/coupling matrix
    psd_prob : (N,) probability weights summing to 1
    normalize : if False, mirrors legacy ``normalize=False`` branch (compute_phase_spectrum
        uses an asymmetric formula; compute_harmonic_power uses a row sum of M*p_i*p_j)
    legacy_self_pair_subtract : reproduce legacy diagonal-subtraction quirk
    alpha_self, alpha_partner : only used in non-legacy mode
    """
    N = matrix.shape[0]
    p = np.asarray(psd_prob, dtype=np.float64)
    out = np.zeros(N, dtype=np.float64)

    if legacy_self_pair_subtract:
        for i in range(N):
            if normalize:
                out[i] = p[i] * np.sum(matrix[i, :] * p) - matrix[i, i] * p[i] ** 2
            else:
                # Legacy compute_phase_spectrum normalize=False branch:
                # uses M[i,:] * p[i] * p (asymmetric in p[i] vs partner p)
                out[i] = np.sum(matrix[i, :] * p[i] * p) - matrix[i, i] * p[i] ** 2
        return out

    # Clean off-diagonal sum (plan §A.4)
    mask = ~np.eye(N, dtype=bool)
    M_off = np.where(mask, matrix, 0.0)
    p_partner = p ** alpha_partner
    return (p ** alpha_self) * (M_off @ p_partner)


register_coupling_metric("nm_plv", nm_plv, arity="pairwise_symmetric")
