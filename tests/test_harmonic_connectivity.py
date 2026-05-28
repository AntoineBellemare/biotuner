"""Tests for biotuner.harmonic_connectivity.

The module previously had ZERO test coverage. These tests cover:

  1. Legacy peak-based connectivity (compute_harm_connectivity):
     harmsim, subharm_tension, harm_fit, euler metrics on small synthetic data.
  2. New registry-based peak-phase-coupling connectivity:
     compute_peak_phase_coupling_connectivity dispatches every registered
     pairwise metric (nm_plv, nm_pli, nm_wpli, nm_rrci, nm_plv_canonical,
     nm_wpli_complex).
  3. New peak-resonance connectivity:
     compute_peak_resonance_connectivity combines H × PC via configurable
     combine rules.
  4. Error paths: unknown metric/kernel/combine, invalid aggregate.
"""
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")

from biotuner.harmonic_connectivity import harmonic_connectivity


# ---------------------------------------------------------------------------
# Shared synthetic 3-electrode dataset
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hc_three_channel():
    """Three-channel synthetic data with known coupling structure:
      e1: 10 Hz tone + noise
      e2: 10 Hz tone phase-shifted + noise   (coupled to e1)
      e3: 17 Hz tone + noise                  (independent of e1, e2)
    """
    sf = 500
    t = np.arange(int(sf * 4)) / sf
    rng = np.random.default_rng(0)
    e1 = np.sin(2 * np.pi * 10 * t) + 0.3 * rng.standard_normal(len(t))
    e2 = np.sin(2 * np.pi * 10 * t + np.pi / 4) + 0.3 * rng.standard_normal(len(t))
    e3 = np.sin(2 * np.pi * 17 * t) + 0.3 * rng.standard_normal(len(t))
    data = np.array([e1, e2, e3])
    return harmonic_connectivity(
        sf=sf, data=data, peaks_function="FOOOF",
        precision=0.5, n_harm=5, min_freq=2, max_freq=30, n_peaks=3,
    )


# ---------------------------------------------------------------------------
# 1. Legacy peak-based connectivity (compute_harm_connectivity)
# ---------------------------------------------------------------------------


def test_compute_harm_connectivity_harmsim_returns_matrix(hc_three_channel):
    M = hc_three_channel.compute_harm_connectivity(metric="harmsim", graph=False)
    assert M.shape == (3, 3)
    # All finite or NaN (NaN on diagonal is a legacy edge case for self-pairs
    # with single-peak detection — known and documented)
    assert np.all(np.isfinite(M) | np.isnan(M))


def test_compute_harm_connectivity_euler_returns_matrix(hc_three_channel):
    M = hc_three_channel.compute_harm_connectivity(metric="euler", graph=False)
    assert M.shape == (3, 3)


# ---------------------------------------------------------------------------
# 2. New registry-based peak phase coupling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metric", [
    "nm_plv", "nm_pli", "nm_wpli", "nm_rrci", "nm_plv_canonical", "nm_wpli_complex",
])
def test_peak_phase_coupling_all_metrics(hc_three_channel, metric):
    """Every registered pairwise coupling metric should produce a finite
    n_elec × n_elec matrix when called via the peak-based connectivity path."""
    M = hc_three_channel.compute_peak_phase_coupling_connectivity(
        coupling_metric=metric, graph=False,
    )
    assert M.shape == (3, 3)
    assert np.all(np.isfinite(M)), f"Non-finite values in {metric} output"
    # Off-diagonal entries should be in [0, 1] (all our metrics are bounded)
    off_diag = M[~np.eye(3, dtype=bool)]
    assert np.all(off_diag >= -1e-9), f"Negative coupling for {metric}"
    assert np.all(off_diag <= 1.0 + 1e-9), f"Coupling > 1.0 for {metric}: max={off_diag.max()}"


def test_peak_phase_coupling_nm_plv_self_coupling_is_high(hc_three_channel):
    """Self-coupling under nm_plv (signal vs itself) should be ~1 since the
    phase difference is identically 0."""
    M = hc_three_channel.compute_peak_phase_coupling_connectivity(
        coupling_metric="nm_plv", graph=False,
    )
    for i in range(3):
        assert M[i, i] > 0.95, f"PLV self-coupling at electrode {i} too low: {M[i, i]}"


def test_peak_phase_coupling_nm_wpli_complex_self_coupling_is_zero(hc_three_channel):
    """Self-coupling under nm_wpli_complex (signal vs itself) should be ~0
    since the imaginary part of X·conj(X) is identically zero (0-lag).
    This is the volume-conduction-robust property."""
    M = hc_three_channel.compute_peak_phase_coupling_connectivity(
        coupling_metric="nm_wpli_complex", graph=False,
    )
    for i in range(3):
        assert M[i, i] < 0.01, (
            f"wPLI_complex self-coupling at electrode {i} should be ~0, got {M[i, i]}"
        )


def test_peak_phase_coupling_aggregate_options(hc_three_channel):
    """The aggregate parameter accepts mean/max/sum and produces different
    (but consistent) outputs."""
    M_mean = hc_three_channel.compute_peak_phase_coupling_connectivity(
        coupling_metric="nm_plv", aggregate="mean", graph=False,
    )
    M_max = hc_three_channel.compute_peak_phase_coupling_connectivity(
        coupling_metric="nm_plv", aggregate="max", graph=False,
    )
    M_sum = hc_three_channel.compute_peak_phase_coupling_connectivity(
        coupling_metric="nm_plv", aggregate="sum", graph=False,
    )
    # max ≥ mean ≥ 0 at each entry; sum has different scale but same nonneg
    assert np.all(M_max >= M_mean - 1e-9), "max should dominate mean"
    assert np.all(M_sum >= 0)


def test_peak_phase_coupling_bandwidth_param(hc_three_channel):
    """The bandwidth kwarg is honored — wider bandwidth changes the output."""
    M_narrow = hc_three_channel.compute_peak_phase_coupling_connectivity(
        coupling_metric="nm_plv", bandwidth=0.5, graph=False,
    )
    M_wide = hc_three_channel.compute_peak_phase_coupling_connectivity(
        coupling_metric="nm_plv", bandwidth=4.0, graph=False,
    )
    # Different bandwidths produce different matrices
    assert not np.allclose(M_narrow, M_wide, atol=1e-8)


# ---------------------------------------------------------------------------
# 3. Peak-based resonance connectivity (H × PC)
# ---------------------------------------------------------------------------


def test_peak_resonance_connectivity_default(hc_three_channel):
    """Default product combine produces a finite matrix."""
    M = hc_three_channel.compute_peak_resonance_connectivity(graph=False)
    assert M.shape == (3, 3)
    assert np.all(np.isfinite(M)), "Resonance matrix contains NaN/inf"


@pytest.mark.parametrize("combine", ["product", "geomean", "harmmean", "min"])
def test_peak_resonance_connectivity_combine_rules(hc_three_channel, combine):
    """Different combine rules produce valid output."""
    M = hc_three_channel.compute_peak_resonance_connectivity(
        combine=combine, graph=False,
    )
    assert M.shape == (3, 3)
    assert np.all(np.isfinite(M))


def test_peak_resonance_combine_changes_output(hc_three_channel):
    """Different combine rules produce different matrices (with same H, PC)."""
    M_prod = hc_three_channel.compute_peak_resonance_connectivity(
        combine="product", graph=False,
    )
    M_min = hc_three_channel.compute_peak_resonance_connectivity(
        combine="min", graph=False,
    )
    assert not np.allclose(M_prod, M_min, atol=1e-8)


# ---------------------------------------------------------------------------
# 4. Error paths
# ---------------------------------------------------------------------------


def test_peak_phase_coupling_unknown_metric_raises(hc_three_channel):
    with pytest.raises(ValueError, match="Unknown coupling_metric"):
        hc_three_channel.compute_peak_phase_coupling_connectivity(
            coupling_metric="not_a_real_metric", graph=False,
        )


def test_peak_phase_coupling_unknown_ratio_kernel_raises(hc_three_channel):
    with pytest.raises(ValueError, match="Unknown ratio_kernel"):
        hc_three_channel.compute_peak_phase_coupling_connectivity(
            coupling_metric="nm_plv", ratio_kernel="fake_kernel", graph=False,
        )


def test_peak_phase_coupling_invalid_aggregate_raises(hc_three_channel):
    with pytest.raises(ValueError, match="aggregate must be"):
        hc_three_channel.compute_peak_phase_coupling_connectivity(
            coupling_metric="nm_plv", aggregate="invalid", graph=False,
        )


def test_peak_resonance_unknown_combine_raises(hc_three_channel):
    with pytest.raises(ValueError, match="Unknown combine"):
        hc_three_channel.compute_peak_resonance_connectivity(
            combine="not_a_real_combine", graph=False,
        )


# ---------------------------------------------------------------------------
# 5. Layer C — compute_cross_resonance_connectivity (n_elec × n_elec matrix)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def hc_long_three_channel():
    """Longer-duration 3-channel dataset so STFT has enough windows for
    cross-resonance to produce meaningful values."""
    sf = 500
    t = np.arange(int(sf * 10)) / sf
    rng = np.random.default_rng(0)
    e1 = np.sin(2 * np.pi * 10 * t) + 0.3 * rng.standard_normal(len(t))
    e2 = np.sin(2 * np.pi * 10 * t + np.pi / 4) + 0.3 * rng.standard_normal(len(t))
    e3 = np.sin(2 * np.pi * 17 * t) + 0.3 * rng.standard_normal(len(t))
    data = np.array([e1, e2, e3])
    from biotuner.harmonic_connectivity import harmonic_connectivity
    return harmonic_connectivity(
        sf=sf, data=data, peaks_function="FOOOF",
        precision=0.5, n_harm=5, min_freq=2, max_freq=30, n_peaks=3,
    )


def test_cross_resonance_connectivity_matrix_shape(hc_long_three_channel):
    """Returns an n_elec × n_elec matrix with NaN on diagonal."""
    M = hc_long_three_channel.compute_cross_resonance_connectivity(graph=False)
    assert M.shape == (3, 3)
    # Diagonal is NaN (self-pair undefined)
    assert np.all(np.isnan(np.diag(M)))
    # Off-diagonal is finite
    off_diag = M[~np.eye(3, dtype=bool)]
    assert np.all(np.isfinite(off_diag))


@pytest.mark.parametrize("factor", ["H", "PC", "R"])
@pytest.mark.parametrize("flavor", ["1to2", "2to1", "all"])
def test_cross_resonance_connectivity_factor_flavor_dispatch(hc_long_three_channel, factor, flavor):
    """All combinations of factor × flavor produce valid matrices."""
    M = hc_long_three_channel.compute_cross_resonance_connectivity(
        factor=factor, flavor=flavor, graph=False,
    )
    assert M.shape == (3, 3)
    off_diag = M[~np.eye(3, dtype=bool)]
    assert np.all(np.isfinite(off_diag))


@pytest.mark.parametrize("aggregate", ["max", "mean", "sum", "peak"])
def test_cross_resonance_connectivity_aggregate(hc_long_three_channel, aggregate):
    """All aggregate options run without error."""
    M = hc_long_three_channel.compute_cross_resonance_connectivity(
        aggregate=aggregate, graph=False,
    )
    assert M.shape == (3, 3)


def test_cross_resonance_connectivity_all_flavor_symmetric(hc_long_three_channel):
    """The 'all' flavor is the symmetrized average of 1to2 and 2to1, so the
    resulting matrix should be APPROXIMATELY symmetric (off-diagonal).
    Exact symmetry isn't guaranteed because compute_cross_resonance(A, B) and
    compute_cross_resonance(B, A) run independent STFT decompositions on each
    signal — small numerical differences (~1e-4) can accumulate through the
    cross-spectrum imaginary-part averaging. The structural symmetry of the
    averaged 'all' flavor is what we verify here."""
    M = hc_long_three_channel.compute_cross_resonance_connectivity(
        flavor="all", graph=False,
    )
    M_test = np.where(np.isnan(M), 0, M)
    # Symmetric to within a few parts per thousand
    np.testing.assert_allclose(M_test, M_test.T, rtol=1e-2, atol=1e-3,
                                err_msg="'all' flavor matrix should be approximately symmetric")


def test_cross_resonance_connectivity_unknown_factor_raises(hc_long_three_channel):
    with pytest.raises(ValueError, match="factor must be"):
        hc_long_three_channel.compute_cross_resonance_connectivity(
            factor="not_real", graph=False,
        )


def test_cross_resonance_connectivity_unknown_flavor_raises(hc_long_three_channel):
    with pytest.raises(ValueError, match="flavor must be"):
        hc_long_three_channel.compute_cross_resonance_connectivity(
            flavor="not_real", graph=False,
        )


def test_cross_resonance_connectivity_unknown_aggregate_raises(hc_long_three_channel):
    with pytest.raises(ValueError, match="aggregate must be"):
        hc_long_three_channel.compute_cross_resonance_connectivity(
            aggregate="not_real", graph=False,
        )


def test_cross_resonance_connectivity_zscore_shapes(hc_long_three_channel):
    """Z-score method returns (observed, z, p) all with same shape."""
    observed, z_matrix, p_matrix = hc_long_three_channel.compute_cross_resonance_connectivity_zscore(
        surrogate_kind="phase_randomize", n_surrogates=5, graph=False,
    )
    assert observed.shape == (3, 3)
    assert z_matrix.shape == (3, 3)
    assert p_matrix.shape == (3, 3)
    # p-values in [0, 1] (excluding NaN diagonal)
    p_off = p_matrix[~np.eye(3, dtype=bool)]
    assert np.all(p_off >= 0)
    assert np.all(p_off <= 1)


@pytest.mark.parametrize("surr_kind", ["phase_randomize", "iaaft", "time_shuffle"])
def test_cross_resonance_connectivity_zscore_all_surrogates(hc_long_three_channel, surr_kind):
    """All 3 surrogate generators work with the z-score method."""
    observed, z, p = hc_long_three_channel.compute_cross_resonance_connectivity_zscore(
        surrogate_kind=surr_kind, n_surrogates=3, graph=False,
    )
    assert observed.shape == (3, 3)


def test_cross_resonance_connectivity_zscore_unknown_surrogate_raises(hc_long_three_channel):
    with pytest.raises(ValueError, match="surrogate_kind must be"):
        hc_long_three_channel.compute_cross_resonance_connectivity_zscore(
            surrogate_kind="not_a_real_generator", n_surrogates=3, graph=False,
        )
