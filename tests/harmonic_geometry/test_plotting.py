"""Smoke tests for biotuner.harmonic_geometry.plotting.

Each renderer is invoked once on a representative GeometryData object;
the test passes if the call returns without exception and produces a
matplotlib Figure (or, for animation helpers, a saved file path that
exists). The numerical content is not asserted — these tests guard the
public API surface, not pixel correctness.
"""
from __future__ import annotations

import os
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

# Force the headless backend before any pyplot import.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from biotuner.harmonic_geometry import HarmonicInput, plotting
from biotuner.harmonic_geometry.geometry_data import GeometryData
from biotuner.harmonic_geometry.lissajous import lissajous_2d, lissajous_3d
from biotuner.harmonic_geometry.polygon_circular import star_polygon
from biotuner.harmonic_geometry.chladni import chladni_field_rectangular
from biotuner.harmonic_geometry.fractal import (
    continued_fraction_rectangles, ifs_harmonic, stern_brocot_tree,
)
from biotuner.harmonic_geometry.generative import lsystem_from_ratios
from biotuner.harmonic_geometry.geometry_3d import (
    lissajous_tube, lsystem_3d, harmonic_point_cloud,
)


MAJOR = HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)])
DOM7  = HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2),
                              Fraction(7, 4)])


# ── helper ────────────────────────────────────────────────────────────────────

def _ax(projection: str = "rect"):
    fig = plt.figure(figsize=(3, 3))
    if projection == "3d":
        ax = fig.add_subplot(111, projection="3d")
    else:
        ax = fig.add_subplot(111)
    return fig, ax


# ── 2-D primitives ────────────────────────────────────────────────────────────

class Test2DPrimitives:
    def test_draw_curve_2d(self):
        g = lissajous_2d(Fraction(3, 2), n_points=200)
        fig, ax = _ax()
        plotting.draw_curve_2d(g, ax)
        plt.close(fig)

    def test_draw_polygon(self):
        g = star_polygon(7, 2)
        fig, ax = _ax()
        plotting.draw_polygon(g, ax, fill=True)
        plt.close(fig)

    def test_draw_polygon_set(self):
        # Build a synthetic GeometryData with a list of polygons
        polys = [
            np.array([[0, 0], [1, 0], [0.5, 1]]),
            np.array([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]),
        ]
        g = GeometryData(geom_type="polygon_set",
                          coordinates=np.array(polys, dtype=object))
        fig, ax = _ax()
        plotting.draw_polygon_set(g, ax)
        plt.close(fig)

    def test_draw_graph_2d(self):
        g = stern_brocot_tree(MAJOR, max_depth=4)
        fig, ax = _ax()
        plotting.draw_graph_2d(g, ax)
        plt.close(fig)

    def test_draw_hyperbolic_graph(self):
        g = stern_brocot_tree(MAJOR, max_depth=4, layout="hyperbolic")
        fig, ax = _ax()
        plotting.draw_hyperbolic_graph(g, ax)
        plt.close(fig)

    def test_draw_point_cloud_2d(self):
        g = ifs_harmonic(MAJOR, n_points=400)
        fig, ax = _ax()
        plotting.draw_point_cloud_2d(g, ax, ref_circle=True)
        plt.close(fig)

    def test_draw_field_2d(self):
        g = chladni_field_rectangular([(3, 2), (5, 1)], resolution=32)
        fig, ax = _ax()
        plotting.draw_field_2d(g, ax)
        plt.close(fig)

    def test_draw_image(self):
        # draw_image renders arbitrary 2D field arrays; reuse a chladni field.
        g = chladni_field_rectangular([(3, 2)], resolution=32)
        fig, ax = _ax()
        plotting.draw_image(g, ax, extent=(-1.5, 1.5, -1.5, 1.5))
        plt.close(fig)

    def test_draw_tree_2d(self):
        g = lsystem_from_ratios(MAJOR, depth=3)
        fig, ax = _ax()
        plotting.draw_tree_2d(g, ax)
        plt.close(fig)

    def test_draw_rectangles(self):
        g = continued_fraction_rectangles(Fraction(7, 4))
        fig, ax = _ax()
        plotting.draw_rectangles(g, ax)
        plt.close(fig)


# ── 3-D primitives ────────────────────────────────────────────────────────────

class Test3DPrimitives:
    def test_draw_mesh_3d(self):
        g = lissajous_tube(MAJOR, n_points=200, n_sides=8)
        fig, ax = _ax(projection="3d")
        plotting.draw_mesh_3d(ax, g)
        plt.close(fig)

    def test_draw_tree_3d(self):
        g = lsystem_3d(MAJOR, depth=2)
        fig, ax = _ax(projection="3d")
        plotting.draw_tree_3d(ax, g)
        plt.close(fig)

    def test_draw_point_cloud_3d(self):
        g = harmonic_point_cloud(MAJOR, n_points=200, surface="sphere")
        fig, ax = _ax(projection="3d")
        plotting.draw_point_cloud_3d(ax, g)
        plt.close(fig)

    def test_draw_curve_3d(self):
        g = lissajous_3d([3, 4, 5], n_points=200)
        fig, ax = _ax(projection="3d")
        plotting.draw_curve_3d(ax, g)
        plt.close(fig)


# ── dispatcher ────────────────────────────────────────────────────────────────

class TestPlotGeometryDispatch:
    def test_curve_2d(self):
        g = lissajous_2d(Fraction(3, 2), n_points=200)
        fig, ax = plotting.plot_geometry(g)
        assert ax is not None
        plt.close(fig)

    def test_mesh_3d(self):
        g = lissajous_tube(MAJOR, n_points=200, n_sides=8)
        fig, ax = plotting.plot_geometry(g)
        plt.close(fig)

    def test_field_2d(self):
        g = chladni_field_rectangular([(3, 2), (5, 1)], resolution=32)
        fig, ax = plotting.plot_geometry(g)
        plt.close(fig)

    def test_invalid_geom_type(self):
        # GeometryData ctor validates geom_type against an enum, so we pass
        # a valid one and then mutate it to drive plot_geometry's error path.
        bad = GeometryData(geom_type="curve_2d",
                            coordinates=np.zeros((4, 2)))
        bad.geom_type = "not_a_real_type"
        with pytest.raises(ValueError):
            plotting.plot_geometry(bad)


# ── layouts ───────────────────────────────────────────────────────────────────

class TestLayouts:
    def test_gallery_2d(self):
        geoms = [lissajous_2d(Fraction(3, 2), n_points=100),
                  star_polygon(7, 2)]
        fig, axes = plotting.gallery(geoms, n_cols=2, suptitle="t")
        assert len(axes) == 2
        plt.close(fig)

    def test_sweep_strip(self):
        geoms = [lissajous_2d(Fraction(p, q), n_points=100)
                  for p, q in [(3, 2), (5, 4), (7, 4)]]
        fig, axes = plotting.sweep_strip(geoms, labels=["3:2", "5:4", "7:4"])
        plt.close(fig)

    def test_gallery_3d(self):
        geoms = [lissajous_tube(MAJOR, n_points=100, n_sides=8),
                  lissajous_tube(DOM7,  n_points=100, n_sides=8)]
        fig, axes = plotting.gallery(geoms, n_cols=2)
        plt.close(fig)

    def test_rotation_strip(self):
        g = lissajous_tube(MAJOR, n_points=100, n_sides=8)
        fig, axes = plotting.rotation_strip(g, n_strip=4)
        assert len(axes) == 4
        plt.close(fig)


# ── animations (write to tmp dir) ────────────────────────────────────────────

class TestAnimations:
    def test_animate_rotation(self, tmp_path):
        g   = lissajous_tube(MAJOR, n_points=80, n_sides=6)
        out = tmp_path / "rot.gif"
        plotting.animate_rotation(g, out, n_frames=4, fps=4, dpi=60)
        assert out.exists() and out.stat().st_size > 0

    def test_animate_geometry_sequence(self, tmp_path):
        geoms = [lissajous_2d(Fraction(p, q), n_points=80) for (p, q) in
                  [(3, 2), (5, 4), (7, 4)]]
        out = tmp_path / "seq.gif"
        plotting.animate_geometry_sequence(geoms, out, fps=4, dpi=60)
        assert out.exists() and out.stat().st_size > 0

# ── style / save helpers ─────────────────────────────────────────────────────

class TestStyleHelpers:
    def test_axis_clean(self):
        fig, ax = _ax()
        plotting.axis_clean(ax)
        plt.close(fig)

    def test_title_ax(self):
        fig, ax = _ax()
        plotting.title_ax(ax, "main", "sub")
        plt.close(fig)

    def test_make_axis_3d(self):
        import matplotlib.pyplot as plt2
        fig = plt2.figure()
        ax = plotting.make_axis_3d(fig, 111, title="t")
        plt2.close(fig)

    def test_save_figure(self, tmp_path):
        fig, ax = _ax()
        ax.plot([0, 1], [0, 1])
        out = tmp_path / "x.png"
        path = plotting.save_figure(fig, out)
        assert path.exists() and path.stat().st_size > 0
