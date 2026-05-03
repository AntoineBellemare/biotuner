"""Tests for biotuner.harmonic_geometry.lissajous."""

from fractions import Fraction

import numpy as np
import pytest

from biotuner.harmonic_geometry import HarmonicInput
from biotuner.harmonic_geometry.lissajous import (
    lissajous_2d,
    lissajous_3d,
    lissajous_compound,
    lissajous_pairwise_grid,
    lissajous_phase_drift,
    lissajous_topology,
)


class TestLissajous2D:
    def test_unit_circle(self):
        """1:1 ratio with phase π/2 and amps (1,1) should trace the unit circle."""
        g = lissajous_2d(ratio=1, phase=np.pi / 2, amps=(1.0, 1.0), n_points=500)
        radii = np.linalg.norm(g.coordinates, axis=1)
        np.testing.assert_allclose(radii, 1.0, atol=1e-10)
        assert g.metadata["closed"] is True

    def test_shape(self):
        g = lissajous_2d(ratio=Fraction(3, 2), n_points=777)
        assert g.geom_type == "curve_2d"
        assert g.coordinates.shape == (777, 2)

    def test_coprime_closes(self):
        g = lissajous_2d(ratio=Fraction(5, 4), phase=0.3, n_points=2000)
        np.testing.assert_allclose(g.coordinates[0], g.coordinates[-1], atol=1e-9)

    def test_metadata_preserves_coprime_pair(self):
        # Non-canonical input (6/4) coerces to (3/2).
        g = lissajous_2d(ratio=Fraction(6, 4), n_points=200)
        assert g.metadata["a"] == 3
        assert g.metadata["b"] == 2
        assert g.parameters["ratio"] == Fraction(3, 2)

    def test_amplitudes_scale_axes(self):
        # Use fine sampling so the discrete max approaches the analytic max.
        g = lissajous_2d(ratio=1, phase=np.pi / 2, amps=(2.0, 0.5), n_points=10_000)
        np.testing.assert_allclose(np.max(np.abs(g.coordinates[:, 0])), 2.0, rtol=1e-5)
        np.testing.assert_allclose(np.max(np.abs(g.coordinates[:, 1])), 0.5, rtol=1e-5)


class TestLissajous3D:
    def test_shape_and_type(self):
        g = lissajous_3d(ratios=[1, 2, 3], n_points=400)
        assert g.geom_type == "curve_3d"
        assert g.coordinates.shape == (400, 3)

    def test_pairwise_coprime_flagged_as_knot(self):
        g = lissajous_3d(ratios=[3, 4, 5], n_points=200)
        assert g.metadata["knot"] is True

    def test_non_coprime_not_a_knot(self):
        g = lissajous_3d(ratios=[2, 4, 6], n_points=200)
        assert g.metadata["knot"] is False

    def test_wrong_number_of_axes(self):
        with pytest.raises(ValueError):
            lissajous_3d(ratios=[1, 2])


class TestLissajousAdapters:
    def test_pairwise_grid_dimensions(self, simple_ratios):
        inp = HarmonicInput(ratios=simple_ratios, base_freq=100.0)
        grid = lissajous_pairwise_grid(inp, n_points=80)
        n = inp.n_components()
        assert len(grid) == n
        assert all(len(row) == n for row in grid)
        for row in grid:
            for cell in row:
                assert cell.geom_type == "curve_2d"
                assert cell.coordinates.shape == (80, 2)

    def test_compound_returns_curve_2d(self, simple_ratios):
        inp = HarmonicInput(ratios=simple_ratios, base_freq=100.0)
        g = lissajous_compound(inp, n_points=512)
        assert g.geom_type == "curve_2d"
        assert g.coordinates.shape == (512, 2)
        assert g.metadata["compound"] is True

    def test_phase_drift_sample_count(self):
        g = lissajous_phase_drift(
            ratio=Fraction(3, 2), drift_rate=0.5, duration=2.0, sr=500
        )
        assert g.geom_type == "curve_2d"
        assert g.coordinates.shape == (1000, 2)

    def test_phase_drift_invalid_args(self):
        with pytest.raises(ValueError):
            lissajous_phase_drift(ratio=1, drift_rate=0.1, duration=0.0)
        with pytest.raises(ValueError):
            lissajous_phase_drift(ratio=1, drift_rate=0.1, duration=1.0, sr=0)


class TestLissajousTopology:
    def test_unit_circle_topology(self):
        g = lissajous_2d(ratio=1, phase=np.pi / 2, amps=(1.0, 1.0), n_points=400)
        topo = lissajous_topology(g)
        assert topo["closed"] is True
        assert topo["lobes_x"] == 1
        assert topo["lobes_y"] == 1
        assert topo["self_intersections"] == 0
        assert topo["period_ratio"] == Fraction(1, 1)

    def test_3_2_topology(self):
        g = lissajous_2d(ratio=Fraction(3, 2), phase=np.pi / 2, n_points=600)
        topo = lissajous_topology(g)
        assert topo["closed"] is True
        assert topo["lobes_x"] == 3
        assert topo["lobes_y"] == 2
        assert topo["period_ratio"] == Fraction(3, 2)
        # Coprime (3, 2) Lissajous with phase π/2 has more than 0 self-crossings.
        assert topo["self_intersections"] > 0

    def test_rejects_non_curve_2d(self):
        from biotuner.harmonic_geometry.geometry_data import GeometryData

        g = GeometryData(geom_type="curve_3d", coordinates=np.zeros((10, 3)))
        with pytest.raises(ValueError):
            lissajous_topology(g)
