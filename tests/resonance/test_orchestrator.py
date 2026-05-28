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
