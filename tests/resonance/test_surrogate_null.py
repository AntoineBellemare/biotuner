"""Tests for biotuner.resonance.with_surrogate_null.

Covers the two fixes:
  1. the default surr_type='IAAFT' actually runs (previously raised because the
     single-signal generator did not implement IAAFT);
  2. factor-level z-scores (H_z, PC_z, R_z) are exposed, not just R_z.
"""
import numpy as np
import pytest

from biotuner.resonance import compute_resonance, ResonanceConfig, with_surrogate_null


def _signal(sf=500, dur=12):
    t = np.arange(int(sf * dur)) / sf
    rng = np.random.default_rng(0)
    return (np.sin(2 * np.pi * 6 * t) + 0.7 * np.sin(2 * np.pi * 12 * t)
            + 0.3 * rng.standard_normal(len(t)))


CFG = ResonanceConfig(precision_hz=0.5, fmin=2, fmax=30)


def test_default_iaaft_runs():
    """The documented default surr_type must work out of the box."""
    res = with_surrogate_null(_signal(), sf=500, config=CFG, n=12, parallel=False, rng_seed=0)
    assert res.resonance_spectrum_z is not None
    assert res.resonance_spectrum_z.shape == res.resonance_spectrum.shape
    assert np.isfinite(res.resonance_spectrum_z).all()


def test_factor_level_z_exposed():
    res = with_surrogate_null(_signal(), sf=500, config=CFG, n=12, parallel=False, rng_seed=0)
    assert set(res.factor_z) == {"H", "PC", "R"}
    for k in ("H", "PC", "R"):
        assert res.factor_z[k].shape == res.freqs.shape
        assert res.factor_surrogate_mean[k].shape == res.freqs.shape
        assert res.factor_surrogate_std[k].shape == res.freqs.shape
    # R-spectrum back-compat fields mirror factor_z['R']
    assert np.allclose(res.resonance_spectrum_z, res.factor_z["R"])
    assert np.allclose(res.surrogate_mean, res.factor_surrogate_mean["R"])


@pytest.mark.parametrize("surr_type", ["IAAFT", "AAFT", "phase_randomize", "time_shuffle", "phase"])
def test_surrogate_types_route(surr_type):
    res = with_surrogate_null(_signal(), sf=500, config=CFG, surr_type=surr_type,
                              n=6, parallel=False, rng_seed=1)
    assert np.isfinite(res.factor_z["R"]).all()


def test_per_factor_pvalues():
    res = with_surrogate_null(_signal(), sf=500, config=CFG, n=12, correction="both",
                              parallel=False, rng_seed=0)
    for key in ("p_value_spectrum", "p_value_PC", "p_value_H"):
        assert key in res.summaries
        p = res.summaries[key]
        assert ((p >= 0) & (p <= 1)).all()


def test_unknown_surr_type_raises():
    with pytest.raises(ValueError):
        with_surrogate_null(_signal(), sf=500, config=CFG, surr_type="nope",
                            n=2, parallel=False)


def test_reproducible_with_seed():
    a = with_surrogate_null(_signal(), sf=500, config=CFG, n=10, parallel=False, rng_seed=42)
    b = with_surrogate_null(_signal(), sf=500, config=CFG, n=10, parallel=False, rng_seed=42)
    assert np.allclose(a.factor_z["PC"], b.factor_z["PC"])


def test_intertrial_plv_separates_coupled_from_uncoupled():
    """nm_intertrial_plv: trial-consistent relative phase -> ~1; random -> ~0."""
    from biotuner.resonance import nm_intertrial_plv
    rng = np.random.default_rng(0)
    n_ep, n_t = 40, 1000
    base = rng.uniform(0, 2 * np.pi, size=(n_ep, 1)) + np.linspace(0, 4 * np.pi, n_t)[None, :]
    # coupled: B = A + constant offset every trial
    pi = base
    pj_coupled = base + np.pi / 4
    pj_uncoupled = base + rng.uniform(0, 2 * np.pi, size=(n_ep, 1))  # random per-trial offset
    itc_c = nm_intertrial_plv(pi, pj_coupled, 1, 1)
    itc_u = nm_intertrial_plv(pi, pj_uncoupled, 1, 1)
    assert itc_c > 0.9
    assert itc_u < 0.5
    assert itc_c > itc_u


def test_intertrial_plv_shape_validation():
    from biotuner.resonance import nm_intertrial_plv
    with pytest.raises(ValueError):
        nm_intertrial_plv(np.zeros((3, 100)), np.zeros((3, 50)), 1, 1)
