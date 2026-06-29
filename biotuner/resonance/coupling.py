"""biotuner.resonance.coupling — phase coupling metrics and per-bin reducers.

Pairwise coupling metrics (arity 2) build a ``Φ[i,j]`` matrix that feeds the
per-bin phase-coupling spectrum. Higher-order metrics (arity 3, K, survey, state)
go in plan Phase 3 and run on a separate code path.

Phase 1 ships four pairwise-symmetric variants on the n:m complex-exponential
mean, each emphasizing a different aspect of phase coherence:

    nm_plv  — |<exp(i*(n*φᵢ - m*φⱼ))>|
              Classical phase-locking value. Maximum when phase difference is
              constant across time. Sensitive to volume conduction (0-lag bias).
              Ref: Tass et al. 1998 *PRL* 81:3291.

    nm_pli  — |<sign(Im(exp(i*(n*φᵢ - m*φⱼ))))>|
              Phase-Lag Index. Counts only the SIGN of the imaginary part;
              zero when phases are exactly aligned or anti-aligned. Robust to
              instantaneous (0-lag) common reference but discards magnitude
              information.
              Ref: Stam, Nolte, Daffertshofer 2007 *Hum Brain Mapp* 28:1178.

    nm_wpli — |<|Im(X)| * sign(Im(X))>| / <|Im(X)|>
              Weighted Phase-Lag Index. Same 0-lag robustness as PLI but weights
              by magnitude of the imaginary part — less variance-biased and
              more sensitive than PLI on noisy data.
              Ref: Vinck et al. 2011 *NeuroImage* 55:1548.

    nm_rrci — |Im(<exp(i*(n*φᵢ - m*φⱼ))>)|
              Rhythmic Ratio Coupling, imaginary part. Like PLV but discards the
              real part of the mean exponential, isolating non-zero-lag coupling.
              Ref: Scheffer-Teixeira & Tort 2016 *eLife* 5:e20515.

Higher-order metrics (bplv triplet, mplv N-ary, cf_plm survey, gpla state) land
in Phase 3 and are NOT valid for ``ResonanceConfig.coupling_metric`` — see
plan §4.4 (arity contract).
"""

import numpy as np

from biotuner.resonance.registry import register_coupling_metric


# ---------------------------------------------------------------------------
# Scalar pairwise metrics on already-extracted phase time series
# ---------------------------------------------------------------------------


def nm_plv(phase_i: np.ndarray, phase_j: np.ndarray, n: int, m: int) -> float:
    """n:m phase-locking value (Tass 1998): ``| <exp(i*(n*φᵢ - m*φⱼ))> |``."""
    phase_diff = n * phase_i - m * phase_j
    return float(np.abs(np.mean(np.exp(1j * phase_diff))))


def nm_pli(phase_i: np.ndarray, phase_j: np.ndarray, n: int, m: int) -> float:
    """n:m phase-lag index (Stam 2007): ``|<sign(Im(exp(i*(n*φᵢ - m*φⱼ))))>|``.

    Volume-conduction robust: returns 0 for perfectly synchronous (0-lag) or
    anti-synchronous (π-lag) phase differences. Counts only sign of Im, so
    discards magnitude.
    """
    phase_diff = n * phase_i - m * phase_j
    im = np.sin(phase_diff)  # imaginary part of exp(i*phase_diff)
    return float(np.abs(np.mean(np.sign(im))))


def nm_wpli(phase_i: np.ndarray, phase_j: np.ndarray, n: int, m: int) -> float:
    """n:m weighted phase-lag index (Vinck 2011):
    ``|<|Im(X)| * sign(Im(X))>| / <|Im(X)|>`` where ``X = exp(i*(n*φᵢ - m*φⱼ))``.

    Same 0-lag robustness as PLI but uses Im magnitude as weight, reducing
    variance bias and improving noise sensitivity.
    """
    phase_diff = n * phase_i - m * phase_j
    im = np.sin(phase_diff)
    abs_im = np.abs(im)
    denom = np.mean(abs_im)
    if denom < 1e-15:
        return 0.0
    return float(np.abs(np.mean(im)) / denom)


def nm_rrci(phase_i: np.ndarray, phase_j: np.ndarray, n: int, m: int) -> float:
    """n:m Rhythmic Ratio Coupling, imaginary (Scheffer-Teixeira & Tort 2016):
    ``|Im(<exp(i*(n*φᵢ - m*φⱼ))>)|``.

    Like PLV but discards the real part of the mean exponential, isolating
    out-of-phase (non-zero-lag) coupling.
    """
    phase_diff = n * phase_i - m * phase_j
    mean_exp = np.mean(np.exp(1j * phase_diff))
    return float(np.abs(np.imag(mean_exp)))


def nm_wpli_complex(
    analytic_i: np.ndarray,
    analytic_j: np.ndarray,
    n: int = 1,
    m: int = 1,
    epsilon: float = 1e-10,
) -> float:
    """Amplitude-weighted n:m wPLI on complex analytic signals.

    Computes::

        |⟨Im(X * conj(Y))⟩| / ⟨|Im(X * conj(Y))|⟩

    where ``X = |a_i| · exp(i·n·φ_i)``, ``Y = |a_j| · exp(i·m·φ_j)``, and the
    analytic signals carry both amplitude and phase. For ``n = m = 1`` this is
    simply ``|⟨Im(a_i · conj(a_j))⟩| / ⟨|Im(a_i · conj(a_j))|⟩``.

    This matches the cross-spectrum formula used in legacy
    ``compute_cross_spectrum_harmonicity``, where ``analytic_*`` are STFT
    coefficients (``Zxx``) at the relevant frequency bins.

    Differs from :func:`nm_wpli` in that the latter discards amplitude
    information (uses only ``sin(Δφ)``), while this variant weights by the
    instantaneous magnitudes ``|a_i| · |a_j|`` — more sensitive when joint
    high-amplitude epochs carry the coupling signal.

    Reference: Vinck et al. 2011 NeuroImage 55:1548 (wPLI); applied to STFT
    cross-spectrum coefficients.
    """
    phase_i = np.angle(analytic_i)
    phase_j = np.angle(analytic_j)
    amp_i = np.abs(analytic_i)
    amp_j = np.abs(analytic_j)
    # n:m generalization of X · conj(Y): magnitude product × phase rotation
    cross = amp_i * amp_j * np.exp(1j * (n * phase_i - m * phase_j))
    im = np.imag(cross)
    # The ``+ epsilon`` in the denominator matches legacy
    # compute_cross_spectrum_harmonicity (epsilon=1e-10) and prevents
    # divide-by-zero for bins with negligible imaginary cross-spectrum.
    return float(np.abs(np.mean(im)) / (np.mean(np.abs(im)) + epsilon))


def nm_plv_canonical(phase_i: np.ndarray, phase_j: np.ndarray, n: int, m: int) -> float:
    """n:m PLV with the **Tass 1998** convention.

    Note on convention
    ------------------
    The legacy biotuner ratio kernel (``get_harmonic_ratio`` / ``binary_nm_kernel``)
    returns ``(n, m)`` such that ``freq_j / freq_i ≈ m / n``. The standard Tass et
    al. 1998 convention is ``freq_j / freq_i = n / m``, with PLV defined as
    ``|<exp(i(n*φᵢ - m*φⱼ))>|`` — so when ``n*f_i = m*f_j`` (a true n:m mode-lock),
    the phase difference is constant in time and PLV = 1.

    The legacy ``nm_plv`` applies ``n*φᵢ - m*φⱼ`` with the swapped ``(n, m)``,
    which yields a non-stationary phase difference even for perfectly locked
    harmonics — it measures STFT-phase-progression coherence rather than true
    n:m phase locking. Bit-exact reproduction of the legacy snapshot preserves
    this behavior.

    This ``nm_plv_canonical`` variant internally swaps (n, m) to recover the
    Tass convention, so it correctly returns 1.0 for perfectly locked harmonic
    pairs. Use this for new analyses where standard n:m PLV semantics matter.
    """
    n_tass, m_tass = m, n  # swap to match Tass convention
    phase_diff = n_tass * phase_i - m_tass * phase_j
    return float(np.abs(np.mean(np.exp(1j * phase_diff))))


def nm_intertrial_plv(
    phase_epochs_i: np.ndarray,
    phase_epochs_j: np.ndarray,
    n: int = 1,
    m: int = 1,
) -> float:
    """Inter-trial n:m phase-locking value across epochs (Tass convention).

    Inputs are 2-D **epoched** phase arrays ``(n_epochs, n_times)`` for the two
    frequencies. For each epoch the time-averaged n:m relative-phase resultant
    ``< exp(i*(m*φ_i - n*φ_j)) >_t`` is formed (note the m/n swap, matching
    :func:`nm_plv_canonical`); the inter-trial PLV is the magnitude of the mean
    of those per-epoch resultants across epochs::

        ITC = | (1/E) Σ_e  < exp(i*(m*φ_i[e] - n*φ_j[e])) >_t |

    This is the correct estimator when coupling is *consistent across trials* but
    the absolute phase resets between trials — the regime where a continuous
    single-trial PLV cannot distinguish genuine coupling from a stationary
    process (see resonance_paper Study 1B / Study 5). It is provided as a
    standalone utility because it requires an epoch dimension that the
    single-trial orchestrator does not model; it is therefore NOT registered as
    a ``PAIRWISE_COUPLING_METRIC`` (those receive 1-D phase from the orchestrator).

    Parameters
    ----------
    phase_epochs_i, phase_epochs_j : ndarray (n_epochs, n_times)
        Instantaneous phase per epoch at the two frequencies (e.g. from the
        ``hilbert`` phase estimator applied per epoch).
    n, m : int
        n:m ratio. For 1:1 this is the standard inter-trial coherence of the
        phase difference.

    Returns
    -------
    float in [0, 1] — 1 = perfectly trial-consistent n:m relative phase.
    """
    pi = np.atleast_2d(np.asarray(phase_epochs_i, dtype=np.float64))
    pj = np.atleast_2d(np.asarray(phase_epochs_j, dtype=np.float64))
    if pi.shape != pj.shape:
        raise ValueError(
            f"phase epoch arrays must have matching shape, got {pi.shape} vs {pj.shape}"
        )
    # m/n swap matches nm_plv_canonical's Tass convention.
    rel = m * pi - n * pj
    per_epoch = np.mean(np.exp(1j * rel), axis=1)   # (n_epochs,) resultant per trial
    return float(np.abs(np.mean(per_epoch)))


# ---------------------------------------------------------------------------
# n:m indices BEYOND the first circular moment (Tass 1998; Palus 1997)
#
# The PLV family above reads only the FIRST circular moment of the relative phase
# psi = n*phi_i - m*phi_j, so it is maximal only for a UNIMODAL psi and CANCELS on
# multimodal / antipodal / multistable n:m locks (resonance_paper Study 41: PLV
# anti-detects them, AUC<0.5). These indices read the whole psi distribution and
# recover those locks. Same (n, m) convention as the raw metrics — pair with the
# canonical wrapper (or correct multipliers) for genuine n:m.
# ---------------------------------------------------------------------------


def nm_rho_entropy(phase_i, phase_j, n, m, nbins: int = 18):
    """Tass 1998 n:m entropy synchronization index in [0, 1].

    ``rho = (Hmax - H)/Hmax`` where ``H`` is the Shannon entropy of the histogram
    of ``psi = n*phi_i - m*phi_j`` (wrapped) and ``Hmax = ln(nbins)``. Reads ANY
    departure of psi from uniformity (all moments), so it detects multimodal n:m
    locks the PLV family misses. Positively biased at small N — use a surrogate z.
    Ref: Tass et al. 1998 PRL 81:3291.
    """
    psi = np.angle(np.exp(1j * (n * np.asarray(phase_i, dtype=np.float64)
                                - m * np.asarray(phase_j, dtype=np.float64))))
    h, _ = np.histogram(psi, bins=nbins, range=(-np.pi, np.pi))
    p = h / h.sum()
    p = p[p > 0]
    H = -np.sum(p * np.log(p))
    return float((np.log(nbins) - H) / np.log(nbins))


def nm_conditional_prob(phase_i, phase_j, n, m, nbins: int = 18, min_count: int = 5):
    """Tass 1998 n:m conditional-probability index in [0, 1].

    Bins ``n*phi_i`` and takes the mean resultant of ``m*phi_j`` within each bin.
    Captures dependence of one phase given the other, but each bin's resultant is
    again a first moment — so it shares the PLV family's blindness to *within-bin*
    antipodal structure. Ref: Tass et al. 1998 PRL 81:3291.
    """
    x = np.angle(np.exp(1j * n * np.asarray(phase_i, dtype=np.float64)))
    y = m * np.asarray(phase_j, dtype=np.float64)
    edges = np.linspace(-np.pi, np.pi, nbins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, nbins - 1)
    rs = [np.abs(np.mean(np.exp(1j * y[idx == k]))) for k in range(nbins) if np.sum(idx == k) > min_count]
    return float(np.mean(rs)) if rs else 0.0


def nm_phase_mi(phase_i, phase_j, n, m, nbins: int = 16):
    """Normalized mutual information in [0, 1] between ``n*phi_i`` and ``m*phi_j``.

    Model-free, all-moment statistical dependence — the most general n:m detector;
    recovers any multimodal/nonlinear lock. Most data-hungry; use a surrogate z.
    Ref: Palus 1997 Phys Lett A 235:341.
    """
    x = np.angle(np.exp(1j * n * np.asarray(phase_i, dtype=np.float64)))
    y = np.angle(np.exp(1j * m * np.asarray(phase_j, dtype=np.float64)))
    c, _, _ = np.histogram2d(x, y, bins=nbins, range=[[-np.pi, np.pi], [-np.pi, np.pi]])
    pxy = c / c.sum()
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    outer = px * py
    nz = pxy > 0
    mi = float(np.sum(pxy[nz] * np.log(pxy[nz] / outer[nz])))
    Hx = -np.sum(px[px > 0] * np.log(px[px > 0]))
    Hy = -np.sum(py[py > 0] * np.log(py[py > 0]))
    return float(mi / (min(Hx, Hy) + 1e-12))


# ---------------------------------------------------------------------------
# Matrix builder — dispatches the pairwise metric over freq pairs
# ---------------------------------------------------------------------------


def build_pairwise_coupling_matrix(
    phase: np.ndarray,
    freqs: np.ndarray,
    ratio_kernel_fn,
    ratio_kernel_params: dict,
    metric_fn=None,
) -> np.ndarray:
    """Build the ``(n_freqs, n_freqs)`` symmetric coupling matrix.

    For each upper-triangular pair (i, j), consults ``ratio_kernel_fn`` to
    determine the best (n, m); when the ratio kernel returns ``W=0``, the pair
    gets coupling=0. With the legacy binary kernel and ``fallback_to_1_1=True``,
    every pair gets a value (n:m if matched, else 1:1).

    Parameters
    ----------
    phase : (n_freqs, n_times) phase time series
    freqs : (n_freqs,) frequency grid
    ratio_kernel_fn : callable returning (W, N, M) arrays
    ratio_kernel_params : dict passed to ratio_kernel_fn
    metric_fn : scalar pairwise metric ``f(phase_i, phase_j, n, m) -> float``.
        Default is :func:`nm_plv` for backward compatibility.
    """
    if metric_fn is None:
        metric_fn = nm_plv
    n_freqs = len(freqs)
    M = np.zeros((n_freqs, n_freqs), dtype=np.float64)

    W, N_mat, M_mat = ratio_kernel_fn(freqs, freqs, **ratio_kernel_params)

    for i in range(n_freqs):
        for j in range(i + 1, n_freqs):
            if W[i, j] <= 0:
                continue
            n, m = int(N_mat[i, j]), int(M_mat[i, j])
            value = W[i, j] * metric_fn(phase[i], phase[j], n, m)
            M[i, j] = value
            M[j, i] = value
    return M


# Backwards-compat alias — the orchestrator originally called this name
def build_nm_plv_matrix(
    phase: np.ndarray,
    freqs: np.ndarray,
    ratio_kernel_fn,
    ratio_kernel_params: dict,
) -> np.ndarray:
    """Backwards-compat alias for :func:`build_pairwise_coupling_matrix` with
    ``metric_fn=nm_plv``."""
    return build_pairwise_coupling_matrix(
        phase, freqs, ratio_kernel_fn, ratio_kernel_params, metric_fn=nm_plv,
    )


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


# ---------------------------------------------------------------------------
# Canonical (Tass-convention) variants — apply the (n, m) -> (m, n) swap for ALL
# metrics, not just PLV.
#
# Rationale: the ratio kernels (binary_nm_kernel, fraction_kernel) return (n, m)
# under the legacy convention ``ratio = f_j / f_i ≈ m / n``. The base metrics apply
# ``n*φ_i - m*φ_j`` literally, which is NON-stationary for a true n:m mode-lock
# (validated in resonance_paper Study 39: every raw metric anti-detects clean
# polyrhythmic ground truth — PLV AUC≈0.13). The mathematically correct n:m
# phase-locking test (Tass 1998) needs ``m*φ_i - n*φ_j`` for these kernels, i.e.
# the (n, m) -> (m, n) swap that ``nm_plv_canonical`` already applies. These
# wrappers extend that fix to pli / wpli / rrci / wpli_complex so the
# volume-conduction-robust metrics can be used at genuine n:m. Use the canonical
# variants for any new n:m analysis; the raw names are kept for bit-exact
# reproduction of prior papers.


def _canonical(metric_fn):
    """Wrap a pairwise metric to apply the Tass n:m convention (swap n<->m).

    The wrapped metric receives the ratio-kernel's ``(n, m)`` (ratio≈m/n) and
    internally calls ``metric_fn(arg_i, arg_j, m, n, ...)`` so the stationary
    combination ``m*φ_i - n*φ_j`` is tested. For 1:1 (n==m) it is a no-op.
    """
    def _wrapped(arg_i, arg_j, n, m, **kw):
        return metric_fn(arg_i, arg_j, m, n, **kw)
    _wrapped.__name__ = metric_fn.__name__ + "_canonical"
    _wrapped.__qualname__ = _wrapped.__name__
    _wrapped.__doc__ = (
        f"Canonical (Tass-convention) variant of :func:`{metric_fn.__name__}` — "
        "swaps (n, m) so the genuine n:m mode-lock is tested with the legacy ratio kernels."
    )
    return _wrapped


nm_pli_canonical = _canonical(nm_pli)
nm_wpli_canonical = _canonical(nm_wpli)
nm_rrci_canonical = _canonical(nm_rrci)
nm_wpli_complex_canonical = _canonical(nm_wpli_complex)
nm_rho_entropy_canonical = _canonical(nm_rho_entropy)
nm_conditional_prob_canonical = _canonical(nm_conditional_prob)
nm_phase_mi_canonical = _canonical(nm_phase_mi)


register_coupling_metric("nm_plv", nm_plv, arity="pairwise_symmetric", input_type="phase")
register_coupling_metric("nm_pli", nm_pli, arity="pairwise_symmetric", input_type="phase")
register_coupling_metric("nm_wpli", nm_wpli, arity="pairwise_symmetric", input_type="phase")
register_coupling_metric("nm_rrci", nm_rrci, arity="pairwise_symmetric", input_type="phase")
register_coupling_metric("nm_plv_canonical", nm_plv_canonical, arity="pairwise_symmetric", input_type="phase")
register_coupling_metric("nm_wpli_complex", nm_wpli_complex, arity="pairwise_symmetric", input_type="analytic")
# canonical variants for the 0-lag-robust / amplitude-weighted metrics (Study 39 fix)
register_coupling_metric("nm_pli_canonical", nm_pli_canonical, arity="pairwise_symmetric", input_type="phase")
register_coupling_metric("nm_wpli_canonical", nm_wpli_canonical, arity="pairwise_symmetric", input_type="phase")
register_coupling_metric("nm_rrci_canonical", nm_rrci_canonical, arity="pairwise_symmetric", input_type="phase")
register_coupling_metric("nm_wpli_complex_canonical", nm_wpli_complex_canonical, arity="pairwise_symmetric", input_type="analytic")
# all-moment n:m indices (Study 41: recover multimodal locks the PLV family misses)
register_coupling_metric("nm_rho_entropy", nm_rho_entropy, arity="pairwise_symmetric", input_type="phase")
register_coupling_metric("nm_conditional_prob", nm_conditional_prob, arity="pairwise_symmetric", input_type="phase")
register_coupling_metric("nm_phase_mi", nm_phase_mi, arity="pairwise_symmetric", input_type="phase")
register_coupling_metric("nm_rho_entropy_canonical", nm_rho_entropy_canonical, arity="pairwise_symmetric", input_type="phase")
register_coupling_metric("nm_conditional_prob_canonical", nm_conditional_prob_canonical, arity="pairwise_symmetric", input_type="phase")
register_coupling_metric("nm_phase_mi_canonical", nm_phase_mi_canonical, arity="pairwise_symmetric", input_type="phase")
