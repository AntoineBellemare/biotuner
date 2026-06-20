import math
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # headless: no display needed for figure tests

from biotuner.harmonic_input import HarmonicInput
from biotuner.math_series import (
    DEFAULT_SERIES,
    SERIES_FUNCS,
    fibonacci,
    farey,
    harmonics,
    hofstadter_q,
    jacobsthal,
    lucas,
    math_series,
    mersenne,
    padovan,
    pell,
    series_ratio_pairs,
    triangular,
)


# --------------------------------------------------------------------------- generators
@pytest.mark.parametrize(
    "func, n, expected",
    [
        (fibonacci, 8, [0, 1, 1, 2, 3, 5, 8, 13]),
        (lucas, 7, [2, 1, 3, 4, 7, 11, 18]),
        (padovan, 7, [1, 1, 1, 2, 2, 3, 4]),
        (pell, 6, [0, 1, 2, 5, 12, 29]),
        (jacobsthal, 6, [0, 1, 1, 3, 5, 11]),
        (mersenne, 5, [0, 1, 3, 7, 15]),
        (hofstadter_q, 8, [1, 1, 2, 3, 3, 4, 5, 5]),
        (harmonics, 5, [1, 2, 3, 4, 5]),
        (triangular, 5, [1, 3, 6, 10, 15]),
    ],
)
def test_generators_known_values(func, n, expected):
    assert func(n) == expected


def test_lucas_custom_seed():
    assert lucas(5, seed=(5, 3)) == [5, 3, 8, 11, 19]


def test_farey_pairs():
    assert farey(4) == [(0, 1), (1, 4), (1, 3), (1, 2), (2, 3), (3, 4), (1, 1)]


def test_all_default_series_registered():
    for name in DEFAULT_SERIES:
        assert name in SERIES_FUNCS


# --------------------------------------------------------------------------- ratio pairs
@pytest.mark.parametrize("name", list(SERIES_FUNCS))
def test_series_ratio_pairs_folded_into_octave(name):
    pairs = series_ratio_pairs(name, 12, octave=2.0)
    assert len(pairs) > 0
    for ratio, ab in pairs:
        assert 1.0 <= ratio < 2.0
        assert isinstance(ab, tuple) and len(ab) == 2


def test_series_ratio_pairs_unknown_raises():
    with pytest.raises(ValueError):
        series_ratio_pairs("not_a_series", 10)


def test_which_filters_direction():
    both = series_ratio_pairs("harmonics", 6, which="both")
    high_low = series_ratio_pairs("harmonics", 6, which="high/low")
    assert all(ab[0] > ab[1] for _, ab in high_low)
    assert len(high_low) < len(both)


# --------------------------------------------------------------------------- analysis
@pytest.fixture
def target_ratios():
    rng = np.random.default_rng(42)
    return sorted({round(1 + rng.random(), 4) for _ in range(12)})


@pytest.fixture
def ms_instance(target_ratios):
    return math_series(ratios=target_ratios, maxdenom=24).analyze()


def test_analyze_populates_scores(ms_instance):
    assert set(ms_instance.series_scores) == set(DEFAULT_SERIES)
    for score in ms_instance.series_scores.values():
        assert 0.0 <= score["proportion"] <= 1.0
        assert score["n_matched"] <= score["n_target"]
    assert ms_instance.best_series in DEFAULT_SERIES


def test_normalized_proportions_sum_to_one(ms_instance):
    total = sum(s["proportion_normalized"] for s in ms_instance.series_scores.values())
    assert total == pytest.approx(1.0)


def test_best_series_has_max_proportion(ms_instance):
    best = ms_instance.best_series
    best_prop = ms_instance.series_scores[best]["proportion"]
    assert best_prop == max(s["proportion"] for s in ms_instance.series_scores.values())


def test_summary_dataframe(ms_instance):
    df = ms_instance.summary()
    assert list(df["series"]) == sorted(
        ms_instance.series_names,
        key=lambda n: ms_instance.series_scores[n]["proportion"],
        reverse=True,
    )
    assert df["proportion"].is_monotonic_decreasing


def test_maxdenom_controls_conservatism(target_ratios):
    lenient = math_series(ratios=target_ratios, maxdenom=12).analyze()
    strict = math_series(ratios=target_ratios, maxdenom=200).analyze()
    lenient_total = sum(s["n_matched"] for s in lenient.series_scores.values())
    strict_total = sum(s["n_matched"] for s in strict.series_scores.values())
    # A coarser grid (lower maxdenom) cannot match fewer ratios than a finer one.
    assert lenient_total >= strict_total


def test_lazy_analyze_via_methods(target_ratios):
    ms = math_series(ratios=target_ratios)
    assert ms.series_scores == {}  # not analyzed yet
    _ = ms.summary()  # triggers analysis
    assert ms.series_scores != {}


# --------------------------------------------------------------------------- input adapters
def test_extract_from_biotuner_mock():
    bt = SimpleNamespace(
        peaks_ratios=[1.5, 1.25, 1.333, 1.667],
        extended_peaks_ratios=[1.2, 1.4, 1.75, 1.6],
    )
    ms_peaks = math_series(bt, ratios_source="peaks_ratios").analyze()
    ms_ext = math_series(bt, ratios_source="extended_peaks_ratios").analyze()
    assert ms_peaks.ratios != ms_ext.ratios
    assert ms_peaks.best_series in DEFAULT_SERIES


def test_extract_from_harmonic_input():
    hi = HarmonicInput(ratios=[1.5, 1.25, 1.333, 1.667], ratios_source="peaks_ratios")
    ms = math_series(hi, ratios_source="peaks_ratios").analyze()
    assert len(ms.ratios) == 4
    assert all(1.0 <= r < 2.0 for r in ms.ratios)


def test_harmonic_input_alternate_scale():
    hi = HarmonicInput(
        ratios=[1.5, 1.25],
        ratios_source="peaks_ratios",
        ratios_alternates={"extended_peaks_ratios": [1.2, 1.4, 1.75, 1.6]},
    )
    ms = math_series(hi, ratios_source="extended_peaks_ratios").analyze()
    assert len(ms.ratios) == 4


def test_harmonic_input_missing_scale_warns():
    hi = HarmonicInput(ratios=[1.5, 1.25], ratios_source="peaks")
    with pytest.warns(UserWarning):
        math_series(hi, ratios_source="extended_peaks_ratios")


# --------------------------------------------------------------------------- musical output
def test_series_scale_sorted_with_unison(ms_instance):
    scale = ms_instance.series_scale()
    assert scale == sorted(scale)
    assert scale[0] == 1.0
    assert all(1.0 <= s < 2.0 for s in scale)


@pytest.mark.parametrize("method", ["subset", "pairwise"])
def test_series_mode(ms_instance, method):
    best = ms_instance.best_series
    scale = ms_instance.series_scale(best)
    if len(scale) < 5:
        pytest.skip("matched scale too small to reduce")
    mode = ms_instance.series_mode(best, n_steps=4, method=method)
    assert mode == sorted(mode)
    assert 0 < len(mode) <= len(scale)


def test_series_mode_string_function(ms_instance):
    mode = ms_instance.series_mode(n_steps=4, function="dyad_similarity")
    assert isinstance(mode, list)


def test_scale_cents(ms_instance):
    cents = ms_instance.scale_cents()
    assert cents[0] == pytest.approx(0.0)
    assert all(0.0 <= c < 1200.0 for c in cents)


# --------------------------------------------------------------------------- plotting
def test_plot_proportions_returns_fig(ms_instance):
    fig = ms_instance.plot_proportions(plot=False)
    assert fig is not None


def test_plot_ratio_pairs_returns_fig(ms_instance):
    fig = ms_instance.plot_ratio_pairs(plot=False)
    assert fig is not None


def test_plots_compose_onto_provided_ax(ms_instance):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2)
    f1 = ms_instance.plot_proportions(ax=ax1, plot=False)
    f2 = ms_instance.plot_ratio_pairs(ax=ax2, plot=False)
    # Both should have drawn onto the supplied axes' shared figure.
    assert f1 is fig and f2 is fig
    assert ax1.patches  # bars
    assert len(ax2.collections) > 0  # scatter points


# --------------------------------------------------------------------------- edge cases
def test_no_input_raises():
    with pytest.raises(ValueError):
        math_series()


def test_empty_ratios_raises():
    with pytest.raises(ValueError):
        math_series(ratios=[])


def test_unknown_series_name_raises():
    with pytest.raises(ValueError):
        math_series(ratios=[1.5], series_names=["fibonacci", "bogus"])


def test_unsupported_source_raises():
    with pytest.raises(TypeError):
        math_series(object(), ratios_source="peaks_ratios")
