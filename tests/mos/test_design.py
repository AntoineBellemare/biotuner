"""Tests for :mod:`biotuner.mos.design`.

These figures are meant to *encode* the scale, not decorate it, so the tests
check the encoding: that the chain style really traces the star polygon its
title claims, that the ring style's edges really are the step sizes, and that
the web really covers every interval.
"""

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biotuner.mos import design as D
from biotuner.mos.modes import wf_number
from biotuner.mos.scale import MOSScale, mos_family

SIGS = [(5, 2), (2, 5), (4, 3), (3, 4), (2, 3), (5, 7), (7, 5), (5, 6)]


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _scales():
    for n_large, n_small in SIGS:
        yield MOSScale.from_signature(n_large, n_small, tuning="central")


# --------------------------------------------------------------------------- #
# The star polygon
# --------------------------------------------------------------------------- #
def test_star_figures_of_the_familiar_scales():
    """The diatonic circle of fifths is the heptagram {7/3}, chromatic {12/5}."""
    assert D.star_density(MOSScale.from_generator(3 / 2, 7)) == 3
    assert D.star_density(MOSScale.from_generator(3 / 2, 12)) == 5


@pytest.mark.parametrize("n_large,n_small", SIGS)
def test_the_hop_is_the_modular_inverse_of_the_wf_number(n_large, n_small):
    """One scale step is WF generators, so one generator is WF^-1 scale steps.

    Conflating the two is easy and wrong: they coincide only when WF is
    self-inverse mod N, as it happens to be for the chromatic scale.
    """
    s = MOSScale.from_signature(n_large, n_small, tuning="central")
    n = s.cardinality
    wf = wf_number(s.generator, n)
    assert (D.star_hop(s) * wf) % n == 1
    assert D.star_density(s) == min(D.star_hop(s), n - D.star_hop(s))


def test_wf_and_hop_differ_for_the_diatonic():
    """The case that caught the error, kept as a regression."""
    d = MOSScale.from_generator(3 / 2, 7)
    assert wf_number(d.generator, 7) == 2
    assert D.star_hop(d) == 4


def test_the_chain_really_is_a_star_polygon():
    """Consecutive chain vertices must be a constant number of pitch steps apart.

    That constant is the star's density; if it were not constant the figure
    would be some arbitrary polygon and the title would be a lie.
    """
    for s in _scales():
        geo = D.web_geometry(s, "chain")
        n = s.cardinality
        # Recover each vertex's rank in pitch order.
        angles = np.arctan2(geo.points[:, 0], geo.points[:, 1]) % (2 * math.pi)
        order = np.argsort(angles)
        rank = np.empty(n, dtype=int)
        rank[order] = np.arange(n)
        hops = {(int(rank[(i + 1) % n]) - int(rank[i])) % n for i in range(n)}
        assert len(hops) == 1, f"{s.signature}: inconsistent hop {hops}"
        assert hops.pop() == D.star_hop(s)


def test_the_chain_closes():
    for s in _scales():
        geo = D.web_geometry(s, "chain")
        assert geo.n_segments == s.cardinality
        assert np.allclose(geo.segments[-1][1], geo.segments[0][0], atol=1e-9)


# --------------------------------------------------------------------------- #
# The other styles
# --------------------------------------------------------------------------- #
def test_ring_edges_are_the_step_sizes():
    for s in _scales():
        geo = D.web_geometry(s, "ring")
        assert geo.n_segments == s.cardinality
        # Two distinct edge lengths, in the counts the signature promises.
        lengths = np.linalg.norm(geo.segments[:, 1] - geo.segments[:, 0], axis=1)
        distinct = np.unique(np.round(lengths, 9))
        assert len(distinct) == 2, f"{s.signature}: {len(distinct)} edge lengths"
        counts = sorted(int((np.round(lengths, 9) == d).sum()) for d in distinct)
        assert counts == sorted([s.n_large, s.n_small])


def test_web_covers_every_unordered_pair():
    for s in _scales():
        n = s.cardinality
        assert D.web_geometry(s, "web").n_segments == n * (n - 1) // 2


def test_web_threshold_only_removes_segments():
    s = MOSScale.from_signature(5, 2, tuning=12)
    full = D.web_geometry(s, "web").n_segments
    counts = [
        D.web_geometry(s, "web", min_harmonicity=t).n_segments
        for t in (0.0, 0.002, 0.005, 1.0)
    ]
    assert counts[0] == full
    assert counts == sorted(counts, reverse=True)
    assert counts[-1] == 0


def test_nested_draws_the_whole_family():
    s = MOSScale.from_generator(3 / 2, 7)
    fam = mos_family(s.generator_ratio, 17, s.period)
    geo = D.web_geometry(s, "nested", max_cardinality=17)
    assert geo.n_segments == sum(m.cardinality for m in fam)
    assert geo.labels == tuple(m.signature for m in fam)


def test_every_style_produces_geometry():
    s = MOSScale.from_signature(5, 2, tuning=31)
    for style in D.STYLES:
        geo = D.web_geometry(s, style, max_cardinality=17)
        assert geo.style == style
        assert geo.n_points > 0
        assert geo.n_segments > 0
        assert len(geo.weights) == geo.n_segments


def test_points_lie_on_the_unit_circle_for_the_flat_styles():
    s = MOSScale.from_signature(5, 2, tuning=31)
    for style in ("chain", "ring", "web"):
        r = np.linalg.norm(D.web_geometry(s, style).points, axis=1)
        assert np.allclose(r, 1.0, atol=1e-9)


# --------------------------------------------------------------------------- #
# Modes
# --------------------------------------------------------------------------- #
def test_mode_rotates_the_ring_without_changing_its_edges():
    """A mode is a rotation, so the multiset of edge lengths is invariant."""
    s = MOSScale.from_signature(5, 2, tuning=31)
    base = np.sort(np.round(np.linalg.norm(
        np.diff(D.web_geometry(s, "ring", mode=0).segments, axis=1)[:, 0], axis=1), 9))
    for m in range(1, s.cardinality):
        got = np.sort(np.round(np.linalg.norm(
            np.diff(D.web_geometry(s, "ring", mode=m).segments, axis=1)[:, 0],
            axis=1), 9))
        assert np.allclose(base, got)


def test_mode_index_wraps():
    s = MOSScale.from_signature(5, 2, tuning=31)
    a = D.web_geometry(s, "ring", mode=0)
    b = D.web_geometry(s, "ring", mode=s.cardinality)
    assert np.allclose(a.points, b.points)


# --------------------------------------------------------------------------- #
# Weight stretching
# --------------------------------------------------------------------------- #
def test_stretch_uses_the_whole_range():
    out = D._stretch(np.array([0.002, 0.006, 0.019]))
    assert out.min() == pytest.approx(0.12)
    assert out.max() == pytest.approx(1.0)


def test_stretch_leaves_a_flat_input_flat():
    """Constant input must not be amplified into invented contrast."""
    out = D._stretch(np.full(5, 0.3))
    assert np.allclose(out, 0.6)


def test_stretch_handles_an_empty_input():
    assert D._stretch(np.array([])).size == 0


def test_geometry_keeps_raw_weights():
    """The stretch is a drawing decision, so the data must stay comparable."""
    s = MOSScale.from_signature(5, 2, tuning=31)
    w = D.web_geometry(s, "web").weights
    assert w.max() < 0.1, "raw harmonicity should be small, not pre-stretched"


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("style", D.STYLES)
@pytest.mark.parametrize("palette", list(D.PALETTES))
def test_every_style_and_palette_draws(style, palette):
    s = MOSScale.from_signature(5, 2, tuning=31)
    fig, ax = D.plot_scale_web(s, style, palette=palette, max_cardinality=17)
    assert ax.get_aspect() == 1.0
    assert ax.collections or ax.lines


def test_chain_title_names_the_star():
    s = MOSScale.from_generator(3 / 2, 7)
    fig, ax = D.plot_scale_web(s, "chain")
    assert "{7/3}" in ax.get_title()


def test_title_can_be_suppressed():
    fig, ax = D.plot_scale_web(MOSScale.from_generator(3 / 2, 7), title="")
    assert ax.get_title() == ""


def test_gallery_has_one_panel_per_scale():
    fam = mos_family(3 / 2, 29)
    fig, axes = D.plot_web_gallery(fam, "chain", n_cols=3)
    assert len(axes) == len(fam)


def test_validation():
    s = MOSScale.from_generator(3 / 2, 7)
    with pytest.raises(ValueError, match="style must be one of"):
        D.web_geometry(s, "sunburst")
    with pytest.raises(ValueError, match=r"min_harmonicity must lie in \[0, 1\]"):
        D.web_geometry(s, "web", min_harmonicity=1.5)
    with pytest.raises(ValueError, match="palette must be one of"):
        D.plot_scale_web(s, palette="neon")
    with pytest.raises(ValueError, match="scales is empty"):
        D.plot_web_gallery([])


def test_a_pseudo_octave_still_draws():
    s = MOSScale.from_signature(5, 2, tuning="central", period=3.0)
    fig, ax = D.plot_scale_web(s, "web")
    assert ax.collections
