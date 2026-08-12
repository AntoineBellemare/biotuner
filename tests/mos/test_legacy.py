"""The frozen pre-replacement figures still run, and still misbehave as recorded.

The point of :mod:`biotuner.mos.legacy` is that the old and new can be compared,
which only works if the old one keeps working. These tests pin its behaviour --
including the defects, since a "fix" here would quietly destroy the comparison.
"""

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biotuner.mos import legacy
from biotuner.mos import plotting as P


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def test_legacy_exports():
    assert set(legacy.__all__) == {
        "plot_labyrinth", "plot_MOS_spiral", "MOS_interactive"
    }


# --------------------------------------------------------------------------- #
# The defects, pinned
# --------------------------------------------------------------------------- #
def test_the_old_labyrinth_still_collapses_to_radius_one_or_two():
    """Its radius is ``sig.index(max(sig)) + 1``, which cannot exceed 2.

    This is the bug the replacement exists to fix. If this test ever starts
    failing, someone has edited frozen code.
    """
    fig, ax = legacy.plot_labyrinth([4 / 3, 3 / 2, 9 / 5], max_steps=16)
    radii = set()
    for line in ax.get_lines():
        radii.update(np.round(line.get_ydata(), 6))
    drawn = {r for r in radii if r > 0}
    assert drawn <= {1.0, 2.0}, f"expected only radius 1 or 2, got {sorted(drawn)}"


def test_the_new_labyrinth_uses_the_whole_radial_range():
    """The same call on the replacement reaches every ring."""
    fig, ax = P.plot_labyrinth(16)
    radii = set()
    for line in ax.get_lines():
        radii.update(np.round(line.get_ydata(), 6))
    assert max(r for r in radii if r <= 16) == 16.0


def test_the_old_spiral_plots_step_index_as_radius():
    """Radius runs 1..steps, so it is the position in the stack, not a cardinality."""
    fig, ax = legacy.plot_MOS_spiral([1.5], max_steps=14)
    spirals = [ln for ln in ax.get_lines() if ln.get_marker() == "o"]
    assert spirals
    for line in spirals:
        y = np.asarray(line.get_ydata())
        assert np.allclose(y, np.arange(1, len(y) + 1))


# --------------------------------------------------------------------------- #
# It still runs
# --------------------------------------------------------------------------- #
def test_spiral_accepts_a_supplied_axes():
    """The one deliberate change: the closure was hoisted so it can be composed."""
    fig, axes = plt.subplots(1, 2, subplot_kw={"projection": "polar"})
    got_fig, got_ax = legacy.plot_MOS_spiral([1.25, 1.5], max_steps=12, ax=axes[0])
    assert got_ax is axes[0]
    assert got_fig is fig


@pytest.mark.parametrize(
    "generators",
    [(1.5, 4 / 3), (1.25, 1.5), (1.5, 1.5), (1.25, 1.5, 1.8), (1.5, 2 ** (18 / 31))],
)
def test_the_old_shared_angle_markers_never_actually_fired(generators):
    """The original's best idea was dead code.

    It collected stacked-degree angles into a dict keyed on the raw float and
    drew a dashed radial wherever one key was reached by two different
    generators -- an exact float-equality test between independently computed
    transcendental quantities, which never succeeds. Measured across every
    generator pair tried: zero dashed lines, always. Identical generators do
    not trigger it either, because the guard compares the *interval values* and
    finds only one distinct value.

    :func:`biotuner.mos.theory.common_tones` is the same idea with a cents
    tolerance, which is what makes it fire at all.
    """
    fig, ax = legacy.plot_MOS_spiral(list(generators), max_steps=14)
    dashed = [
        ln for ln in ax.get_lines()
        if ln.get_linestyle() not in ("-", "solid") and len(ln.get_xdata()) == 2
    ]
    assert dashed == [], "the original is not supposed to draw these"


@pytest.mark.parametrize(
    "generators,tol,expected",
    [((1.5, 4 / 3), 5.0, 1), ((1.25, 1.5), 5.0, 2), ((1.5, 1.5), 0.5, 17)],
)
def test_the_replacement_finds_the_tones_the_original_missed(generators, tol, expected):
    """Same question, answered with a tolerance instead of float equality."""
    from biotuner.mos import theory as T

    fractions = [T.generator_fraction(g) for g in generators]
    got = T.common_tones(fractions, max_cardinality=18, tol_cents=tol)
    assert len(got) == expected


def test_spiral_survives_a_single_generator():
    fig, ax = legacy.plot_MOS_spiral([1.5], max_steps=10)
    assert ax.name == "polar"


def test_mos_interactive_builds_without_a_kernel():
    pytest.importorskip("ipywidgets")
    pytest.importorskip("IPython")
    # The original displays rather than returning; success is not raising.
    assert legacy.MOS_interactive() is None


def test_vizs_shim_no_longer_runs_the_original():
    """`vizs.MOS_interactive` delegates now -- which is why legacy exists."""
    from biotuner import vizs

    assert "legacy" in (vizs.MOS_interactive.__doc__ or "")
    assert "legacy" in (vizs.plot_labyrinth.__doc__ or "")


def test_legacy_honours_the_octave_argument_it_always_did():
    fig, ax = legacy.plot_labyrinth([3 / 2], max_steps=12, octave=2)
    assert ax.name == "polar"
