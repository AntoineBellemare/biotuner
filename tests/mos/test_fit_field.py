"""Tests for the fit field -- the whole labyrinth scored against one signal.

The field's job is to show that a single best-fit answer hides structure, so
the tests check that it really does find the places ``fit_mos`` names, that it
finds *more* than one of them, and that it never invents a scale where none
exists.
"""

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biotuner.mos import theory as T
from biotuner.mos.derive import FitField, best_mos, fit_field
from biotuner.mos.plotting import plot_fit_field
from biotuner.mos.scale import MOSScale

PENT = [1.0, 1.125, 1.3333, 1.5, 1.6875]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def field():
    return fit_field(PENT, max_cardinality=18, resolution=360)


# --------------------------------------------------------------------------- #
# Shape and emptiness
# --------------------------------------------------------------------------- #
def test_shape_and_axes(field):
    assert field.errors.shape == (19, 359)
    assert len(field.generators) == 359
    assert list(field.cardinalities) == list(range(19))
    assert field.n_targets == len(PENT)
    assert field.ratios == tuple(PENT)


def test_most_of_the_plane_is_empty(field):
    """Well-formedness is rare: a generator admits only a handful of note counts."""
    assert 0.2 < field.coverage < 0.6


def test_a_cell_is_filled_exactly_when_a_mos_exists(field):
    """No scale may be scored where the theory says none exists, and none skipped."""
    rng = np.random.default_rng(0)
    cols = rng.choice(len(field.generators), size=40, replace=False)
    for col in cols:
        g = float(field.generators[col])
        expected = {
            c for c, _, _ in T.mos_series(g, max_cardinality=18, include_trivial=True)
            if c >= 3
        }
        got = {int(r) for r in np.nonzero(np.isfinite(field.errors[:, col]))[0]}
        assert got == expected, f"generator {g}"


def test_rows_zero_one_two_are_always_empty(field):
    assert not np.isfinite(field.errors[:3]).any()


# --------------------------------------------------------------------------- #
# It agrees with the fitter, where it should
# --------------------------------------------------------------------------- #
def test_the_field_is_dark_where_fit_mos_lands():
    """The winner must sit in a low-error neighbourhood, not on an isolated spike."""
    ref = MOSScale.from_signature(5, 2, tuning=31)
    f = fit_field(ref.ratios, max_cardinality=14, resolution=720)
    col = int(np.argmin(np.abs(f.generators - ref.generator)))
    err = f.errors[ref.cardinality, col]
    assert np.isfinite(err)
    assert err < 3.0, f"expected a low-error cell at the true generator, got {err}"


def test_the_field_never_beats_the_refined_fit_by_much():
    """The field samples; fit_mos refines. The field cannot be dramatically better."""
    ref = MOSScale.from_signature(4, 3, tuning=19)
    f = fit_field(ref.ratios, max_cardinality=12, resolution=360)
    refined = best_mos(ref.ratios, max_cardinality=12, complexity_penalty=0.0)
    assert f.best()["error_cents"] >= refined.error_cents - 1e-9


def test_best_reports_a_real_cell(field):
    best = field.best()
    col = int(np.argmin(np.abs(field.generators - best["generator"])))
    assert field.errors[best["cardinality"], col] == pytest.approx(
        best["error_cents"]
    )


def test_best_raises_on_an_empty_field():
    f = FitField(np.full((4, 5), np.nan), np.linspace(0.1, 0.9, 5),
                 np.arange(4), 2.0, 3)
    with pytest.raises(ValueError, match="no cell in this field contains a scale"):
        f.best()


# --------------------------------------------------------------------------- #
# Islands -- the point of the whole thing
# --------------------------------------------------------------------------- #
def test_a_signal_lives_in_more_than_one_place(field):
    """The finding the field exists to show: several disconnected good regions."""
    assert field.islands(3.0) > 1


def test_island_count_falls_as_the_threshold_tightens(field):
    counts = [field.islands(t) for t in (30.0, 10.0, 3.0, 1.0, 0.0)]
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] == 0


def test_islands_wrap_around_the_generator_axis():
    """The axis is a circle, so a region spanning 0 must count once, not twice."""
    errors = np.full((4, 10), np.nan)
    errors[3, 0] = 0.5
    errors[3, -1] = 0.5      # adjacent only if the axis wraps
    f = FitField(errors, np.linspace(0.0, 0.9, 10), np.arange(4), 2.0, 3)
    assert f.islands(1.0) == 1


def test_islands_counts_genuinely_separate_regions():
    errors = np.full((4, 12), np.nan)
    errors[3, 2] = 0.1
    errors[3, 7] = 0.1
    f = FitField(errors, np.linspace(0.0, 0.92, 12), np.arange(4), 2.0, 3)
    assert f.islands(1.0) == 2


# --------------------------------------------------------------------------- #
# Chance level
# --------------------------------------------------------------------------- #
def test_chance_error_shrinks_with_cardinality(field):
    assert field.chance_error(5) > field.chance_error(20)
    assert field.chance_error(5) == pytest.approx(1200 / 20)


def test_a_pseudo_octave_scales_the_chance_level():
    f = fit_field([1.0, 1.4, 1.9], period=3.0, max_cardinality=8, resolution=90)
    assert f.period_cents == pytest.approx(1200 * math.log2(3.0))
    assert f.chance_error(6) == pytest.approx(f.period_cents / 24)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_validation():
    with pytest.raises(ValueError, match="resolution must be at least 8"):
        fit_field(PENT, resolution=4)
    with pytest.raises(ValueError, match="below min_cardinality"):
        fit_field(PENT, max_cardinality=4, min_cardinality=9)
    with pytest.raises(ValueError, match="no usable ratios"):
        fit_field([0.0, -1.0])


def test_alignment_can_only_help(field):
    """Offset zero is among the candidates, so aligning never scores worse."""
    aligned = fit_field(PENT, max_cardinality=12, resolution=180, align=True)
    pinned = fit_field(PENT, max_cardinality=12, resolution=180, align=False)
    both = np.isfinite(aligned.errors) & np.isfinite(pinned.errors)
    assert (aligned.errors[both] <= pinned.errors[both] + 1e-9).all()


# --------------------------------------------------------------------------- #
# The figure
# --------------------------------------------------------------------------- #
def test_polar_and_cartesian_both_draw(field):
    fig, ax = plot_fit_field(field)
    assert ax.name == "polar"
    plt.close(fig)
    fig, ax = plot_fit_field(field, polar=False)
    assert ax.name == "rectilinear"


def test_peaks_survive_a_precomputed_field(field):
    """The overlay must not vanish exactly when a field is reused."""
    fig, ax = plot_fit_field(field, show_peaks=True)
    dashed = [ln for ln in ax.get_lines() if ln.get_linestyle() != "-"]
    assert len(dashed) == len(PENT)


def test_peaks_can_be_overridden(field):
    fig, ax = plot_fit_field(field, peaks=[1.0, 1.5])
    dashed = [ln for ln in ax.get_lines() if ln.get_linestyle() != "-"]
    assert len(dashed) == 2


def test_peaks_can_be_switched_off(field):
    fig, ax = plot_fit_field(field, show_peaks=False)
    dashed = [ln for ln in ax.get_lines() if ln.get_linestyle() != "-"]
    assert dashed == []


def test_mark_accepts_a_scale_and_a_list(field):
    s = MOSScale.from_signature(5, 2, tuning=31)
    fig, ax = plot_fit_field(field, mark=s)
    solid = [ln for ln in ax.get_lines() if ln.get_linestyle() == "-"]
    assert len(solid) == 1
    plt.close(fig)
    fig, ax = plot_fit_field(field, mark=[s, 0.7])
    solid = [ln for ln in ax.get_lines() if ln.get_linestyle() == "-"]
    assert len(solid) == 2


def test_the_colour_ceiling_is_applied(field):
    fig, ax = plot_fit_field(field, max_error_cents=12.0)
    assert ax.collections[0].get_clim() == (0.0, 12.0)


def test_ratios_can_be_passed_directly():
    fig, ax = plot_fit_field(PENT, max_cardinality=10, resolution=120)
    dashed = [ln for ln in ax.get_lines() if ln.get_linestyle() != "-"]
    assert len(dashed) == len(PENT)
