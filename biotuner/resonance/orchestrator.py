"""biotuner.resonance.orchestrator — main entry point for resonance computation.

The orchestrator dispatches strategies named in :class:`ResonanceConfig` against the
registries in :mod:`biotuner.resonance.registry`, runs the per-bin pipeline
(harmonic kernel → ratio kernel → phase coupling → combine), and returns a
:class:`ResonanceResult`.

Output-arity invariant (plan §4.4): the per-bin resonance spectrum is built ONLY
from pairwise-or-lower factors. Higher-order coupling metrics (bplv / mplv / cf_plm
/ gpla) run on a separate code path and attach as ``ResonanceResult.higher_order``.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import detrend as _detrend

# Side-effect imports register kernels / metrics / combine rules / etc.
from biotuner.resonance import kernels_harmonic  # noqa: F401
from biotuner.resonance import kernels_ratio  # noqa: F401
from biotuner.resonance import phase_estimators  # noqa: F401
from biotuner.resonance import coupling as _coupling
from biotuner.resonance import combine as _combine  # noqa: F401
from biotuner.resonance.registry import (
    HARMONIC_KERNELS,
    RATIO_KERNELS,
    PHASE_ESTIMATORS,
    PAIRWISE_COUPLING_METRICS,
    COMBINE_RULES,
    COUPLING_ARITY,
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ResonanceConfig:
    """Configuration for :func:`compute_resonance`. Plan §4.1.

    Default values reproduce the legacy ``compute_global_harmonicity`` pipeline so
    that the snapshot regression test (``tests/resonance/test_snapshot.py``) passes.
    New code should override these to opt in to cleaner numerics (e.g. switch
    ``psd_normalization`` to ``'prob'`` and ``legacy_self_pair_subtract`` to False).
    """

    # Spectral preprocessing
    psd_method: Literal["welch", "multitaper"] = "welch"
    remove_aperiodic: bool = True
    psd_normalization: Literal["prob", "minmax_prob", "none"] = "minmax_prob"

    # Harmonic kernel
    harmonic_kernel: str = "harmsim"
    harmonic_kernel_params: Dict[str, Any] = field(default_factory=dict)

    # Ratio kernel (n:m gate)
    ratio_kernel: str = "binary"
    ratio_kernel_params: Dict[str, Any] = field(
        default_factory=lambda: {"max_nm": 3, "tolerance": 0.05, "fallback_to_1_1": True}
    )

    # Phase estimation
    phase_estimator: str = "stft"
    phase_estimator_params: Dict[str, Any] = field(default_factory=dict)

    # Pairwise phase coupling metric (must be arity 'pairwise_*')
    coupling_metric: str = "nm_plv"
    coupling_metric_params: Dict[str, Any] = field(default_factory=dict)
    pac_convention: Literal["row", "col", "symmetrize"] = "row"

    # Higher-order coupling (separate path; plan §4.4)
    higher_order_coupling: Optional[str] = None
    higher_order_params: Dict[str, Any] = field(default_factory=dict)
    project_higher_order_to_bins: bool = False
    projection_op: Literal["max", "sum", "complexity_weighted_sum"] = "complexity_weighted_sum"

    # Persistence / Q (Phase 3)
    persistence: Optional[str] = None
    persistence_params: Dict[str, Any] = field(default_factory=dict)

    # Combine rule
    combine: str = "product"
    combine_params: Dict[str, Any] = field(default_factory=dict)

    # Power weighting exponents (clean off-diagonal mode only)
    alpha_self: float = 1.0
    alpha_partner: float = 1.0

    # Reducer mode: legacy keeps the bit-equivalent self-pair-subtract formula
    legacy_self_pair_subtract: bool = True

    # Smoothing & detrending (per plan; legacy defaults preserved)
    gaussian_smooth_sigma: float = 1.0
    detrend: bool = False
    rescale_factors_after_detrend: bool = True  # legacy behavior

    # PSD / phase computation knobs (legacy compatibility)
    precision_hz: float = 1.0
    fmin: float = 1.0
    fmax: float = 30.0
    noverlap: int = 1
    smoothness: float = 1.0  # divides STFT nperseg
    n_peaks: int = 5
    normalize: bool = True  # for the per-bin reducer
    bandwidth_correction: bool = False

    # Null model (off by default; opt in via dict)
    null_model: Optional[Dict[str, Any]] = None

    # --- Cross-channel-specific config (used only by
    #     biotuner.harmonic_connectivity.compute_cross_resonance) ---
    # PC reducer for cross-channel:
    #   'count'         — legacy uniform average over freq pairs (default; preserves
    #                     compute_cross_spectrum_harmonicity bit-exactness)
    #   'joint'         — same joint-probability weighting as H (uses
    #                     p1[i] * p2[j] gating; frequency-localizes PC)
    #   'joint_2T_count'— legacy 'weighted' phase_mode behavior
    cross_pc_reducer: Literal["count", "joint", "joint_2T_count"] = "count"
    # If True, use config.ratio_kernel + ratio_kernel_params to determine (n, m)
    # for each cross-channel freq pair, and pass them to the coupling metric
    # for n:m phase locking. Default False = always use (n=1, m=1) which is
    # what the legacy cross-spectrum wPLI does.
    cross_use_ratio_kernel: bool = False

    # Logging / debug
    return_intermediates: bool = False


@dataclass
class HigherOrderResult:
    """Plan §4.4. Populated only when ``config.higher_order_coupling`` is set."""

    method: str
    triplets: Optional[List[Tuple]] = None  # bplv
    polyrhythms: Optional[List[Tuple]] = None  # mplv
    coupled_pairs: Optional[List[Tuple]] = None  # cf_plm
    gplv: Optional[float] = None  # gpla
    singular_vectors: Optional[Tuple] = None  # gpla
    summaries: Dict[str, float] = field(default_factory=dict)


@dataclass
class ResonanceResult:
    """Plan §4.3. Output of :func:`compute_resonance`."""

    freqs: np.ndarray
    resonance_spectrum: np.ndarray
    resonance_spectrum_z: Optional[np.ndarray] = None
    surrogate_mean: Optional[np.ndarray] = None
    surrogate_std: Optional[np.ndarray] = None
    factors: Dict[str, np.ndarray] = field(default_factory=dict)
    summaries: Dict[str, Any] = field(default_factory=dict)
    config: Optional[ResonanceConfig] = None
    peaks: Optional[Dict[str, np.ndarray]] = None
    higher_order: Optional[HigherOrderResult] = None
    participation_spectrum: Optional[np.ndarray] = None
    intermediates: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _validate_pairwise(metric_name: str) -> None:
    arity = COUPLING_ARITY.get(metric_name)
    if arity is None:
        raise ValueError(
            f"Unknown coupling metric {metric_name!r}. "
            f"Registered pairwise metrics: {list(PAIRWISE_COUPLING_METRICS)}"
        )
    if not arity.startswith("pairwise"):
        raise ValueError(
            f"coupling_metric must be a pairwise metric (got {metric_name!r} with arity "
            f"{arity!r}). Higher-order methods belong in higher_order_coupling."
        )


def compute_resonance(
    signal: np.ndarray,
    sf: float,
    config: Optional[ResonanceConfig] = None,
    freqs: Optional[np.ndarray] = None,
) -> ResonanceResult:
    """Compute the resonance spectrum + factor breakdown for a 1-D signal.

    With default ``ResonanceConfig`` this reproduces the legacy
    ``compute_global_harmonicity`` numerics within ``atol=1e-6`` on the snapshot
    regression set (``tests/resonance/snapshots/``).

    Returns
    -------
    ResonanceResult
        freqs, resonance_spectrum, factors={'H', 'PC'}, summaries (added by callers),
        config, peaks={'H', 'PC', 'R'}, optional surrogate fields and higher_order.
    """
    cfg = config or ResonanceConfig()

    # Step 0 — arity check
    _validate_pairwise(cfg.coupling_metric)

    # Step 1-3 — PSD, optional aperiodic removal, normalization to probability
    from biotuner.biotuner_utils import compute_frequency_and_psd, apply_power_law_remove

    freqs_arr, psd = compute_frequency_and_psd(
        signal,
        cfg.precision_hz,
        smoothness=cfg.smoothness,
        fs=sf,
        noverlap=cfg.noverlap,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
    )
    psd_clean = apply_power_law_remove(freqs_arr, psd, cfg.remove_aperiodic)

    if cfg.psd_normalization == "minmax_prob":
        psd_min = np.min(psd_clean)
        psd_max = np.max(psd_clean)
        psd_clean = (psd_clean - psd_min) / (psd_max - psd_min)
        psd_prob_for_reducer = psd_clean / np.sum(psd_clean)
    elif cfg.psd_normalization == "prob":
        psd_prob_for_reducer = psd_clean / np.sum(psd_clean)
    else:  # 'none'
        psd_prob_for_reducer = psd_clean.copy()

    # Step 4 — phase extraction
    phase_fn = PHASE_ESTIMATORS[cfg.phase_estimator]
    phase = phase_fn(
        signal,
        sf=sf,
        precision_hz=cfg.precision_hz,
        noverlap=cfg.noverlap,
        smoothness=cfg.smoothness,
        **cfg.phase_estimator_params,
    )

    # Step 5 — harmonic similarity matrix S[i,j]
    kernel_fn = HARMONIC_KERNELS[cfg.harmonic_kernel]
    S = kernel_fn(freqs_arr, freqs_arr, **cfg.harmonic_kernel_params)

    # Step 6-7 — phase coupling matrix Φ[i,j] via the ratio kernel + pairwise metric
    ratio_fn = RATIO_KERNELS[cfg.ratio_kernel]
    metric_fn = PAIRWISE_COUPLING_METRICS[cfg.coupling_metric]
    Phi = _coupling.build_pairwise_coupling_matrix(
        phase, freqs_arr, ratio_fn, cfg.ratio_kernel_params, metric_fn=metric_fn,
    )

    # Step 9 — per-bin reduction
    PC_values = _coupling.reduce_matrix_to_spectrum(
        Phi,
        psd_prob_for_reducer,
        normalize=cfg.normalize,
        legacy_self_pair_subtract=cfg.legacy_self_pair_subtract,
        alpha_self=cfg.alpha_self,
        alpha_partner=cfg.alpha_partner,
    )
    H_values = _coupling.reduce_matrix_to_spectrum(
        S,
        psd_prob_for_reducer,
        normalize=cfg.normalize,
        legacy_self_pair_subtract=cfg.legacy_self_pair_subtract,
        alpha_self=cfg.alpha_self,
        alpha_partner=cfg.alpha_partner,
    )

    # Legacy: apply bandwidth correction to H only (compute_harmonic_power signature)
    if cfg.bandwidth_correction:
        max_possible_partners = int(cfg.fmax / cfg.fmin) - 1
        for i in range(len(freqs_arr)):
            n_partners = int(cfg.fmax / freqs_arr[i]) - 1
            if n_partners > 0:
                H_values[i] *= max_possible_partners / n_partners

    # Step 10 — Gaussian smoothing on H and PC (legacy: smoothness_harm)
    if cfg.gaussian_smooth_sigma > 0:
        H_values = gaussian_filter(H_values, cfg.gaussian_smooth_sigma)
        PC_values = gaussian_filter(PC_values, cfg.gaussian_smooth_sigma)

    if cfg.detrend:
        H_values = _detrend(H_values, type="linear")
        if cfg.rescale_factors_after_detrend and len(H_values) > 0:
            rng = np.max(H_values) - np.min(H_values)
            if rng > 0:
                H_values = (H_values - np.min(H_values)) / rng
        PC_values = _detrend(PC_values, type="linear")
        if cfg.rescale_factors_after_detrend and len(PC_values) > 0:
            rng = np.max(PC_values) - np.min(PC_values)
            if rng > 0:
                PC_values = (PC_values - np.min(PC_values)) / rng

    # Step 11 — combine factors
    combine_fn = COMBINE_RULES[cfg.combine]
    resonance_spectrum = combine_fn([H_values, PC_values], **cfg.combine_params)

    # Step 15 — peaks per factor + rich complexity summaries per spectrum
    from biotuner.harmonic_spectrum import find_spectral_peaks
    from biotuner.metrics import spectrum_complexity

    peaks = {
        "H": find_spectral_peaks(H_values, freqs_arr, cfg.n_peaks, prominence_threshold=0.5)[0],
        "PC": find_spectral_peaks(PC_values, freqs_arr, cfg.n_peaks, prominence_threshold=0.0001)[0],
        "R": find_spectral_peaks(resonance_spectrum, freqs_arr, cfg.n_peaks, prominence_threshold=0.00001)[0],
    }

    summaries = {
        "H": spectrum_complexity(H_values, freqs_arr, n_peaks=cfg.n_peaks, prominence_threshold=0.5),
        "PC": spectrum_complexity(PC_values, freqs_arr, n_peaks=cfg.n_peaks, prominence_threshold=0.0001),
        "R": spectrum_complexity(resonance_spectrum, freqs_arr, n_peaks=cfg.n_peaks, prominence_threshold=0.00001),
    }

    result = ResonanceResult(
        freqs=freqs_arr,
        resonance_spectrum=resonance_spectrum,
        factors={"H": H_values, "PC": PC_values},
        peaks=peaks,
        summaries=summaries,
        config=cfg,
    )

    if cfg.return_intermediates:
        result.intermediates = {
            "psd_clean": psd_clean,
            "psd_prob": psd_prob_for_reducer,
            "harmonicity_matrix": S,
            "phase_coupling_matrix": Phi,
            "phase": phase,
        }

    # Step 12 — null model
    if cfg.null_model is not None:
        from biotuner.resonance.nulls import with_surrogate_null
        params = dict(cfg.null_model)
        result = with_surrogate_null(signal, sf, cfg, **params)

    return result
