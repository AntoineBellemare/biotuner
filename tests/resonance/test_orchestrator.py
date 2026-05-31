"""Integration tests for biotuner.resonance.compute_resonance.

These tests replace the former ``compute_global_harmonicity`` integration tests
from ``tests/test_harmonic_spectrum.py``. They cover:
  - end-to-end pipeline produces sensible ResonanceResult
  - factor dict contains H and PC
  - summaries dict contains per-spectrum complexity
  - peaks dict contains H/PC/R peak frequencies
  - arity validation rejects higher-order metrics in the pairwise slot
  - alternative configs (different kernel, prob normalization) run without error
"""
import os
import sys

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")

# Make _signals discoverable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _signals import (  # noqa: E402
    SIGNALS,
    BASELINE_CONFIG,
    legacy_default_resonance_config_kwargs,
)

from biotuner.resonance import compute_resonance, ResonanceConfig, ResonanceResult  # noqa: E402


@pytest.fixture(scope="module")
def harmonic_sig():
    return SIGNALS["harmonic_5_10_20_40"](sf=1000), 1000


def test_compute_resonance_returns_result(harmonic_sig):
    sig, sf = harmonic_sig
    cfg = ResonanceConfig(**legacy_default_resonance_config_kwargs())
    result = compute_resonance(sig, sf=sf, config=cfg)
    assert isinstance(result, ResonanceResult)


def test_compute_resonance_result_shapes(harmonic_sig):
    sig, sf = harmonic_sig
    cfg = ResonanceConfig(**legacy_default_resonance_config_kwargs())
    result = compute_resonance(sig, sf=sf, config=cfg)
    n_freqs = result.freqs.size
    assert n_freqs > 0
    assert result.resonance_spectrum.shape == (n_freqs,)
    assert result.factors["H"].shape == (n_freqs,)
    assert result.factors["PC"].shape == (n_freqs,)


def test_compute_resonance_summaries_per_spectrum(harmonic_sig):
    sig, sf = harmonic_sig
    cfg = ResonanceConfig(**legacy_default_resonance_config_kwargs())
    result = compute_resonance(sig, sf=sf, config=cfg)
    for key in ("H", "PC", "R"):
        assert key in result.summaries
        s = result.summaries[key]
        for k in ("flatness", "entropy", "spread", "higuchi", "avg", "max", "peaks", "peak_harmsim"):
            assert k in s


def test_compute_resonance_peaks_dict(harmonic_sig):
    sig, sf = harmonic_sig
    cfg = ResonanceConfig(**legacy_default_resonance_config_kwargs())
    result = compute_resonance(sig, sf=sf, config=cfg)
    assert set(result.peaks.keys()) == {"H", "PC", "R"}
    for v in result.peaks.values():
        assert isinstance(v, np.ndarray)


def test_compute_resonance_default_config_works():
    """Calling compute_resonance with no config uses ResonanceConfig defaults."""
    sf = 1000
    sig = SIGNALS["harmonic_5_10_20_40"](sf=sf)
    result = compute_resonance(sig, sf=sf)
    assert result.freqs.size > 0


def test_arity_validation_rejects_higher_order():
    """Passing a triplet / nary metric in coupling_metric must raise."""
    # Manually register a fake higher-order metric so the validator has something to reject
    from biotuner.resonance.registry import register_coupling_metric

    def _fake_triplet(*args, **kwargs):
        return 0.0

    register_coupling_metric("fake_triplet", _fake_triplet, arity="triplet")

    cfg = ResonanceConfig(coupling_metric="fake_triplet")
    sf = 1000
    sig = SIGNALS["harmonic_5_10_20_40"](sf=sf)
    with pytest.raises(ValueError, match="pairwise"):
        compute_resonance(sig, sf=sf, config=cfg)


def test_arity_validation_rejects_unknown_metric():
    cfg = ResonanceConfig(coupling_metric="totally_made_up_name")
    sf = 1000
    sig = SIGNALS["harmonic_5_10_20_40"](sf=sf)
    with pytest.raises(ValueError, match="Unknown coupling metric"):
        compute_resonance(sig, sf=sf, config=cfg)


def test_alternative_psd_normalization_runs():
    """psd_normalization='prob' (the cleaner non-legacy mode) runs without error."""
    cfg_kwargs = legacy_default_resonance_config_kwargs()
    cfg_kwargs["psd_normalization"] = "prob"
    cfg = ResonanceConfig(**cfg_kwargs)
    sf = 1000
    sig = SIGNALS["harmonic_5_10_20_40"](sf=sf)
    result = compute_resonance(sig, sf=sf, config=cfg)
    assert np.all(np.isfinite(result.resonance_spectrum))


def test_alternative_combine_rule_geomean():
    cfg_kwargs = legacy_default_resonance_config_kwargs()
    cfg_kwargs["combine"] = "geomean"
    cfg = ResonanceConfig(**cfg_kwargs)
    sf = 1000
    sig = SIGNALS["harmonic_5_10_20_40"](sf=sf)
    result = compute_resonance(sig, sf=sf, config=cfg)
    assert np.all(np.isfinite(result.resonance_spectrum))


# ---------------------------------------------------------------------------
# fraction_kernel — Fraction.limit_denominator-based ratio kernel
# ---------------------------------------------------------------------------


def test_fraction_kernel_returns_exact_simple_ratios():
    """For pairs at exact simple ratios, fraction_kernel returns (n, m) = the
    obvious reduced fraction."""
    from biotuner.resonance.kernels_ratio import fraction_kernel

    freqs = np.array([10.0, 20.0, 30.0, 15.0])
    W, N, M = fraction_kernel(freqs, freqs, max_denom=16, beta=1.0)
    # f_j / f_i convention: n*f_i ≈ m*f_j → n/m = f_j/f_i

    # (10, 20): ratio = 2 → (n, m) = (2, 1)
    assert (N[0, 1], M[0, 1]) == (2, 1)
    # (10, 30): ratio = 3 → (3, 1)
    assert (N[0, 2], M[0, 2]) == (3, 1)
    # (10, 15): ratio = 1.5 → (3, 2)
    assert (N[0, 3], M[0, 3]) == (3, 2)
    # (20, 30): ratio = 1.5 → (3, 2)
    assert (N[1, 2], M[1, 2]) == (3, 2)


def test_fraction_kernel_returns_exact_complex_ratio():
    """For non-simple pairs like 10:17, fraction_kernel returns the actual
    integer ratio (n=17, m=10) when max_denom is large enough — unlike
    binary_nm_kernel which only goes up to max_nm=3 and falls back to 1:1."""
    from biotuner.resonance.kernels_ratio import fraction_kernel

    freqs = np.array([10.0, 17.0])
    W, N, M = fraction_kernel(freqs, freqs, max_denom=20, beta=1.0)
    # 17/10 = 17/10 exactly
    assert (N[0, 1], M[0, 1]) == (17, 10), (
        f"Expected (17, 10), got ({N[0, 1]}, {M[0, 1]})"
    )


def test_fraction_kernel_simpler_approximation_with_low_max_denom():
    """When max_denom is small, fraction_kernel returns the closest simpler
    approximation. 17/10 ≈ 5/3 (closest with denom ≤ 5)."""
    from biotuner.resonance.kernels_ratio import fraction_kernel

    freqs = np.array([10.0, 17.0])
    W, N, M = fraction_kernel(freqs, freqs, max_denom=5, beta=1.0)
    # Closest rational to 1.7 with denom <= 5: 5/3 ≈ 1.667
    assert (N[0, 1], M[0, 1]) == (5, 3), (
        f"Expected (5, 3) (closest to 1.7 with denom <= 5), got ({N[0, 1]}, {M[0, 1]})"
    )


def test_fraction_kernel_weight_decreases_with_complexity():
    """Tenney-height penalty: simpler ratios get higher weight than complex ones."""
    from biotuner.resonance.kernels_ratio import fraction_kernel

    freqs = np.array([10.0, 17.0, 20.0])
    W, N, M = fraction_kernel(freqs, freqs, max_denom=20, beta=1.0)
    # (10, 20): (2, 1), T=log2(2)=1 → W ≈ e^-1 ≈ 0.37
    # (10, 17): (17, 10), T=log2(170)≈7.4 → W ≈ e^-7.4 ≈ 6e-4
    w_simple = W[0, 2]   # 10:20 = 1:2
    w_complex = W[0, 1]  # 10:17
    assert w_simple > w_complex * 100, (
        f"Simple ratio weight {w_simple:.4f} should be >> complex {w_complex:.4f}"
    )


def test_fraction_kernel_beta_zero_gives_uniform_weight():
    """beta=0 disables the Tenney penalty — every found ratio gets W=1."""
    from biotuner.resonance.kernels_ratio import fraction_kernel

    freqs = np.array([10.0, 20.0, 17.0])
    W, N, M = fraction_kernel(freqs, freqs, max_denom=20, beta=0.0)
    # Off-diagonal entries should all be 1.0 (every freq pair finds a ratio)
    off_diag = W[~np.eye(3, dtype=bool)]
    np.testing.assert_allclose(off_diag, 1.0, atol=1e-12)


def test_fraction_kernel_handles_zero_freqs():
    """Zero frequencies produce W=0 (no division)."""
    from biotuner.resonance.kernels_ratio import fraction_kernel

    freqs = np.array([0.0, 10.0, 20.0])
    W, N, M = fraction_kernel(freqs, freqs, max_denom=16)
    # Row 0 and col 0 (involving the 0 Hz bin) should be zero
    assert np.all(W[0, :] == 0)
    assert np.all(W[:, 0] == 0)
    # Other entries should be nonzero
    assert W[1, 2] > 0


def test_fraction_kernel_registered():
    """fraction_kernel is discoverable via the registry."""
    from biotuner.resonance.registry import RATIO_KERNELS
    assert "fraction" in RATIO_KERNELS


def test_orchestrator_works_with_fraction_kernel():
    """End-to-end: compute_resonance with ratio_kernel='fraction' runs and
    produces finite output."""
    cfg_kwargs = legacy_default_resonance_config_kwargs()
    cfg_kwargs["ratio_kernel"] = "fraction"
    cfg_kwargs["ratio_kernel_params"] = {"max_denom": 16, "beta": 1.0}
    cfg = ResonanceConfig(**cfg_kwargs)
    sf = 1000
    sig = SIGNALS["harmonic_5_10_20_40"](sf=sf)
    result = compute_resonance(sig, sf=sf, config=cfg)
    assert result.resonance_spectrum.shape == result.factors["H"].shape


@pytest.mark.parametrize("metric", ["nm_plv", "nm_pli", "nm_wpli", "nm_rrci"])
def test_all_pairwise_coupling_metrics_run(metric):
    """Each registered pairwise coupling metric produces a finite per-bin spectrum."""
    cfg_kwargs = legacy_default_resonance_config_kwargs()
    cfg_kwargs["coupling_metric"] = metric
    cfg = ResonanceConfig(**cfg_kwargs)
    sf = 1000
    sig = SIGNALS["harmonic_5_10_20_40"](sf=sf)
    result = compute_resonance(sig, sf=sf, config=cfg)
    assert np.all(np.isfinite(result.factors["PC"])), f"PC contains non-finite for {metric}"
    assert np.all(np.isfinite(result.resonance_spectrum)), f"R non-finite for {metric}"


def test_coupling_metrics_produce_different_outputs():
    """The four pairwise metrics differ in how they aggregate complex coupling —
    they should produce different PC spectra on the same input."""
    sf = 1000
    sig = SIGNALS["harmonic_5_10_20_40"](sf=sf)
    pcs = {}
    for metric in ["nm_plv", "nm_pli", "nm_wpli", "nm_rrci"]:
        cfg_kwargs = legacy_default_resonance_config_kwargs()
        cfg_kwargs["coupling_metric"] = metric
        result = compute_resonance(sig, sf=sf, config=ResonanceConfig(**cfg_kwargs))
        pcs[metric] = result.factors["PC"]
    # No two metrics should produce identical outputs (within sane tolerance)
    metric_names = list(pcs.keys())
    for i in range(len(metric_names)):
        for j in range(i + 1, len(metric_names)):
            assert not np.allclose(pcs[metric_names[i]], pcs[metric_names[j]], atol=1e-8), \
                f"{metric_names[i]} and {metric_names[j]} produced identical PC"


def test_nm_pli_zero_for_perfect_synchrony():
    """PLI is volume-conduction robust: zero for 0-lag synchrony."""
    from biotuner.resonance.coupling import nm_pli
    rng = np.random.default_rng(0)
    # Two phase series that are perfectly aligned (0-lag): phi_j = phi_i
    phi = rng.uniform(-np.pi, np.pi, size=5000)
    # PLI of perfectly aligned phases: Im(exp(0)) = 0 everywhere → mean sign = 0
    assert nm_pli(phi, phi, 1, 1) < 1e-10


def test_nm_plv_high_for_perfect_synchrony():
    """PLV is sensitive to 0-lag synchrony (returns 1.0 for identical phases)."""
    from biotuner.resonance.coupling import nm_plv
    rng = np.random.default_rng(0)
    phi = rng.uniform(-np.pi, np.pi, size=5000)
    assert nm_plv(phi, phi, 1, 1) > 0.999


def test_return_intermediates_populates_field():
    cfg_kwargs = legacy_default_resonance_config_kwargs()
    cfg = ResonanceConfig(return_intermediates=True, **cfg_kwargs)
    sf = 1000
    sig = SIGNALS["harmonic_5_10_20_40"](sf=sf)
    result = compute_resonance(sig, sf=sf, config=cfg)
    assert result.intermediates is not None
    assert "harmonicity_matrix" in result.intermediates
    assert "phase_coupling_matrix" in result.intermediates
