"""Tests for biotuner.biotuner_group.BiotunerGroup.

Covers the group-level orchestrator that wraps a batch of compute_biotuner
objects — the public-facing API for multi-trial / multi-channel analyses.

Sections:
  1. Construction & shape validation        — 2D, 3D, axis_labels, metadata
  2. Index DataFrame                        — flat & 3D-flattened layout
  3. compute_peaks                          — peaks populated, marker set,
                                              store_objects=False semantics,
                                              graceful no-peak handling
  4. Fluent API                             — fit methods return self
  5. compute_metrics / compute_diss_curve / compute_harmonic_entropy /
     compute_euler_fokker / compute_harmonic_tuning
                                            — ordering guards & happy paths
  6. get_attribute                          — list/array, missing_value
  7. summary                                — DataFrame shape & columns
  8. tuning_summary / get_tuning_scales     — convenience wrappers
  9. compare_groups                         — t-test, DataFrame output, attrs
 10. Dunder methods                         — __repr__, __len__

Plotting methods are intentionally NOT exercised for numerical assertions —
they rely on matplotlib state and don't carry stable contracts.  One smoke
check confirms ``plot_group_peaks`` runs without error on a small fixture.
"""
import os
import warnings

import matplotlib
matplotlib.use("Agg")               # avoid GUI in CI

import numpy as np
import pandas as pd
import pytest

from biotuner.biotuner_group import BiotunerGroup


SF = 1000
DURATION = 4.0
N_SAMPLES = int(SF * DURATION)
PEAKS_FUNCTION = "fixed"          # fast and deterministic
PRECISION = 0.5


# ─── shared fixtures ────────────────────────────────────────────────────────


def _signal(freqs, seed=0):
    """Synthetic biosignal with known spectral peaks."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)
    sig = sum((1.0 / (i + 1)) * np.sin(2 * np.pi * f * t)
              for i, f in enumerate(freqs))
    sig += 0.02 * rng.standard_normal(len(t))
    return sig.astype(np.float64)


@pytest.fixture(scope="module")
def data_2d():
    """4 trials × 4000 samples — single-channel batch."""
    return np.stack([
        _signal([5, 10, 20, 40], seed=0),
        _signal([5, 10, 20, 40], seed=1),
        _signal([5, 7, 13, 23], seed=2),     # different peak set
        _signal([5, 7, 13, 23], seed=3),
    ], axis=0)


@pytest.fixture(scope="module")
def metadata_2d():
    """Two-condition labels matching data_2d's 4 trials."""
    return {"condition": ["A", "A", "B", "B"]}


@pytest.fixture(scope="module")
def data_3d():
    """2 trials × 3 channels × 4000 samples."""
    rows = []
    for trial in range(2):
        chans = []
        for ch in range(3):
            chans.append(_signal([5, 10, 20, 40], seed=trial * 10 + ch))
        rows.append(np.stack(chans, axis=0))
    return np.stack(rows, axis=0)            # (2, 3, N_SAMPLES)


@pytest.fixture(scope="module")
def bt_group_with_peaks(data_2d, metadata_2d):
    """A BiotunerGroup with peaks extracted (no metrics yet) — module-scoped
    so the slow peaks_extraction runs once for the whole module."""
    btg = BiotunerGroup(
        data_2d, sf=SF,
        metadata=metadata_2d,
        peaks_function=PEAKS_FUNCTION,
        precision=PRECISION,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        btg.compute_peaks(min_freq=1.0, max_freq=50.0, n_peaks=4)
    return btg


@pytest.fixture(scope="module")
def bt_group_full(data_2d, metadata_2d):
    """A BiotunerGroup with peaks + metrics — for tests of summary, etc."""
    btg = BiotunerGroup(
        data_2d, sf=SF,
        metadata=metadata_2d,
        peaks_function=PEAKS_FUNCTION,
        precision=PRECISION,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        btg.compute_peaks(min_freq=1.0, max_freq=50.0, n_peaks=4)
        btg.compute_metrics()
    return btg


# ─── 1. Construction & shape validation ────────────────────────────────────


def test_init_2d_basic(data_2d):
    btg = BiotunerGroup(data_2d, sf=SF)
    assert btg.shape == (4, N_SAMPLES)
    assert btg.n_series == 4
    assert btg.is_3d is False
    assert btg.objects == []                   # store_objects=True default


def test_init_3d_flattens(data_3d):
    btg = BiotunerGroup(data_3d, sf=SF, axis_labels=["trial", "channel"])
    assert btg.shape == (2, 3, N_SAMPLES)
    assert btg.n_series == 6                   # 2 * 3
    assert btg.is_3d is True


def test_init_rejects_1d():
    with pytest.raises(ValueError, match="at least 2D"):
        BiotunerGroup(np.zeros(100), sf=SF)


def test_init_rejects_4d():
    with pytest.raises(ValueError, match="2D or 3D"):
        BiotunerGroup(np.zeros((2, 2, 2, 100)), sf=SF)


def test_init_axis_labels_length_validation(data_3d):
    """axis_labels must have ndim-1 elements."""
    with pytest.raises(ValueError, match="axis_labels"):
        BiotunerGroup(data_3d, sf=SF, axis_labels=["only_one"])


def test_init_metadata_length_validation(data_2d):
    """metadata column lengths must match the first data dimension."""
    bad_metadata = {"condition": ["A", "B"]}    # length 2, expected 4
    with pytest.raises(ValueError, match="length"):
        BiotunerGroup(data_2d, sf=SF, metadata=bad_metadata)


def test_init_store_objects_false(data_2d):
    """store_objects=False sets self.objects to None."""
    btg = BiotunerGroup(data_2d, sf=SF, store_objects=False)
    assert btg.objects is None
    assert btg.store_objects is False


def test_init_default_axis_labels(data_2d, data_3d):
    btg2 = BiotunerGroup(data_2d, sf=SF)
    btg3 = BiotunerGroup(data_3d, sf=SF)
    assert btg2.axis_labels == ["series"]
    assert btg3.axis_labels == ["dim0", "dim1"]


def test_init_stores_biotuner_kwargs(data_2d):
    btg = BiotunerGroup(data_2d, sf=SF, peaks_function="fixed", precision=0.5)
    assert btg.biotuner_kwargs["peaks_function"] == "fixed"
    assert btg.biotuner_kwargs["precision"] == 0.5


# ─── 2. Index DataFrame ────────────────────────────────────────────────────


def test_index_df_2d_shape_and_columns(data_2d, metadata_2d):
    btg = BiotunerGroup(data_2d, sf=SF, metadata=metadata_2d)
    idx = btg.index_df
    assert len(idx) == 4
    assert "series_idx" in idx.columns
    assert "condition" in idx.columns
    assert list(idx["condition"]) == ["A", "A", "B", "B"]


def test_index_df_3d_metadata_expanded(data_3d):
    """Trial-level metadata expands across all channels of that trial."""
    btg = BiotunerGroup(
        data_3d, sf=SF,
        axis_labels=["trial", "channel"],
        metadata={"condition": ["X", "Y"]},     # length matches first dim only
    )
    idx = btg.index_df
    assert len(idx) == 6
    # Trial 0 → "X" repeated 3× (channels), trial 1 → "Y" repeated 3×
    assert list(idx["condition"][:3]) == ["X", "X", "X"]
    assert list(idx["condition"][3:]) == ["Y", "Y", "Y"]


# ─── 3. compute_peaks ──────────────────────────────────────────────────────


def test_compute_peaks_populates_objects(bt_group_with_peaks):
    btg = bt_group_with_peaks
    assert len(btg.objects) == 4
    for bt in btg.objects:
        assert hasattr(bt, "peaks")
        assert hasattr(bt, "peaks_ratios")


def test_compute_peaks_marker_in_computed_methods(bt_group_with_peaks):
    assert "peaks_extraction" in bt_group_with_peaks._computed_methods


def test_compute_peaks_3d_runs(data_3d):
    btg = BiotunerGroup(
        data_3d, sf=SF, axis_labels=["trial", "channel"],
        peaks_function=PEAKS_FUNCTION, precision=PRECISION,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        btg.compute_peaks(min_freq=1.0, max_freq=50.0, n_peaks=4)
    assert len(btg.objects) == 6


def test_compute_peaks_returns_self(data_2d):
    btg = BiotunerGroup(data_2d, sf=SF, peaks_function=PEAKS_FUNCTION,
                        precision=PRECISION)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = btg.compute_peaks(min_freq=1.0, max_freq=50.0, n_peaks=4)
    assert out is btg


# ─── 4. Method-chaining & ordering guards ───────────────────────────────────


def test_compute_metrics_requires_peaks(data_2d):
    btg = BiotunerGroup(data_2d, sf=SF)
    with pytest.raises(RuntimeError, match="compute_peaks"):
        btg.compute_metrics()


def test_compute_diss_curve_requires_peaks(data_2d):
    btg = BiotunerGroup(data_2d, sf=SF)
    with pytest.raises(RuntimeError, match="compute_peaks"):
        btg.compute_diss_curve()


def test_compute_harmonic_entropy_requires_peaks(data_2d):
    btg = BiotunerGroup(data_2d, sf=SF)
    with pytest.raises(RuntimeError, match="compute_peaks"):
        btg.compute_harmonic_entropy()


def test_compute_euler_fokker_requires_peaks(data_2d):
    btg = BiotunerGroup(data_2d, sf=SF)
    with pytest.raises(RuntimeError, match="compute_peaks"):
        btg.compute_euler_fokker()


def test_compute_harmonic_tuning_requires_peaks(data_2d):
    btg = BiotunerGroup(data_2d, sf=SF)
    with pytest.raises(RuntimeError, match="compute_peaks"):
        btg.compute_harmonic_tuning()


def test_compute_metrics_marks_method(bt_group_full):
    assert "compute_peaks_metrics" in bt_group_full._computed_methods


def test_compute_metrics_populates_peaks_metrics(bt_group_full):
    for bt in bt_group_full.objects:
        assert hasattr(bt, "peaks_metrics")
        assert isinstance(bt.peaks_metrics, dict)
        assert len(bt.peaks_metrics) > 0


# ─── 5. store_objects=False enforcement ────────────────────────────────────


def test_compute_metrics_requires_store_objects(data_2d):
    btg = BiotunerGroup(data_2d, sf=SF, store_objects=False,
                        peaks_function=PEAKS_FUNCTION, precision=PRECISION)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        btg.compute_peaks(min_freq=1.0, max_freq=50.0, n_peaks=4)
    with pytest.raises(RuntimeError, match="store_objects"):
        btg.compute_metrics()


def test_get_attribute_requires_store_objects(data_2d):
    btg = BiotunerGroup(data_2d, sf=SF, store_objects=False)
    with pytest.raises(RuntimeError, match="store_objects"):
        btg.get_attribute("peaks")


def test_summary_requires_store_objects(data_2d):
    btg = BiotunerGroup(data_2d, sf=SF, store_objects=False)
    with pytest.raises(RuntimeError, match="store_objects"):
        btg.summary()


# ─── 6. get_attribute ───────────────────────────────────────────────────────


def test_get_attribute_returns_per_object_list(bt_group_with_peaks):
    btg = bt_group_with_peaks
    peaks_all = btg.get_attribute("peaks")
    assert len(peaks_all) == btg.n_series


def test_get_attribute_missing_returns_default(bt_group_with_peaks):
    out = bt_group_with_peaks.get_attribute(
        "this_attribute_does_not_exist", missing_value="MISSING",
    )
    assert all(v == "MISSING" for v in out)


def test_get_attribute_no_objects_yet(data_2d):
    btg = BiotunerGroup(data_2d, sf=SF)
    # store_objects=True but compute_peaks not run yet
    with pytest.raises(RuntimeError, match="No biotuner objects"):
        btg.get_attribute("peaks")


# ─── 7. summary ─────────────────────────────────────────────────────────────


def test_summary_returns_dataframe_with_one_row_per_series(bt_group_full):
    df = bt_group_full.summary()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == bt_group_full.n_series


def test_summary_includes_metadata_columns(bt_group_full):
    df = bt_group_full.summary()
    assert "condition" in df.columns
    assert "series_idx" in df.columns


def test_summary_n_peaks_column(bt_group_full):
    df = bt_group_full.summary()
    assert "n_peaks" in df.columns
    assert (df["n_peaks"] >= 0).all()


def test_summary_aggregation_subset(bt_group_full):
    """Custom aggregation list returns only those statistics."""
    df = bt_group_full.summary(aggregation=["mean", "std"])
    # Aggregations of peak_freq should be present for these keys
    has_mean = any(c.endswith("_mean") for c in df.columns)
    has_std = any(c.endswith("_std") for c in df.columns)
    has_median = any(c.endswith("_median") for c in df.columns)
    assert has_mean and has_std
    assert not has_median                       # excluded


def test_summary_results_attribute_is_set(bt_group_full):
    """summary() caches its output on self.results."""
    df = bt_group_full.summary()
    assert bt_group_full.results is df


def test_summary_without_index(bt_group_full):
    df = bt_group_full.summary(include_index=False)
    assert "series_idx" not in df.columns
    assert "condition" not in df.columns


# ─── 8. tuning_summary / get_tuning_scales ──────────────────────────────────


def test_tuning_summary_empty_when_no_scales(bt_group_full):
    """No scale_* metrics yet → empty DataFrame returned with a printed note."""
    out = bt_group_full.tuning_summary()
    assert isinstance(out, pd.DataFrame)
    # No scale columns yet, so result has no rows
    assert len(out) == 0


def test_get_tuning_scales_returns_list_per_object(bt_group_with_peaks):
    """Without compute_diss_curve, every object lacks diss_scale → empty lists."""
    scales = bt_group_with_peaks.get_tuning_scales("diss_scale")
    assert len(scales) == bt_group_with_peaks.n_series


def test_get_tuning_scales_unknown_attr(bt_group_with_peaks):
    """Unknown scale type returns an empty list per object."""
    scales = bt_group_with_peaks.get_tuning_scales("nonexistent_scale_attr")
    assert all(s == [] for s in scales)


# ─── 9. compare_groups ─────────────────────────────────────────────────────


def test_compare_groups_returns_dataframe(bt_group_full):
    btg = bt_group_full
    btg.summary()                          # ensure results is built
    res = btg.compare_groups("condition", metric="n_peaks", plot=False)
    assert isinstance(res, pd.DataFrame)
    # One row per group
    assert sorted(res["group"]) == ["A", "B"]
    # Stats columns present
    for col in ("n", "mean", "std", "median", "sem"):
        assert col in res.columns


def test_compare_groups_test_attrs(bt_group_full):
    btg = bt_group_full
    btg.summary()
    res = btg.compare_groups("condition", metric="n_peaks", plot=False)
    # Test results stored as attrs
    assert "test_name" in res.attrs
    assert "p-value" in res.attrs
    assert "significant" in res.attrs


def test_compare_groups_missing_metric_raises(bt_group_full):
    btg = bt_group_full
    btg.summary()
    with pytest.raises(ValueError, match="not found"):
        btg.compare_groups("condition", metric="nonexistent", plot=False)


def test_compare_groups_missing_groupby_raises(bt_group_full):
    btg = bt_group_full
    btg.summary()
    with pytest.raises(ValueError, match="not found"):
        btg.compare_groups("nonexistent_column", metric="n_peaks", plot=False)


def test_compare_groups_mannwhitney_test(bt_group_full):
    """Non-parametric test path."""
    btg = bt_group_full
    btg.summary()
    res = btg.compare_groups("condition", metric="n_peaks",
                              test="mannwhitneyu", plot=False)
    assert "Mann-Whitney" in res.attrs["test_name"]


# ─── 10. Dunder methods ────────────────────────────────────────────────────


def test_len_returns_n_series(bt_group_with_peaks):
    assert len(bt_group_with_peaks) == bt_group_with_peaks.n_series


def test_repr_mentions_n_series(bt_group_with_peaks):
    s = repr(bt_group_with_peaks)
    assert "BiotunerGroup" in s
    assert str(bt_group_with_peaks.n_series) in s


# ─── 11. Plot smoke check (no assertions on figure content) ────────────────


def test_plot_group_peaks_runs(bt_group_with_peaks):
    """plot_group_peaks should not crash on a small fixture."""
    import matplotlib.pyplot as plt
    plt.close("all")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            bt_group_with_peaks.plot_group_peaks(show_individual=False)
        finally:
            plt.close("all")
