"""Tests for biotuner.harmonic_geometry.polygon_circular (Phase 2 basics)."""

from fractions import Fraction

import numpy as np
import pytest

from biotuner.harmonic_geometry import HarmonicInput
from biotuner.harmonic_geometry.polygon_circular import (
    epicycloid,
    hypocycloid,
    rose_curve,
    star_polygon,
    times_table_circle,
    tuning_circle,
)


class TestStarPolygon:
    def test_n_1_is_regular_polygon(self):
        g = star_polygon(n=5, k=1, radius=1.0)
        assert g.geom_type == "polygon"
        assert g.coordinates.shape == (5, 2)
        assert g.metadata["compound"] is False
        # Every vertex should sit on the unit circle.
        radii = np.linalg.norm(g.coordinates, axis=1)
        np.testing.assert_allclose(radii, 1.0, atol=1e-12)

    def test_pentagram(self):
        g = star_polygon(n=5, k=2, radius=1.0)
        assert g.geom_type == "polygon"
        assert g.coordinates.shape == (5, 2)
        assert g.metadata["schlafli"] == "{5/2}"
        # gcd(5, 2) = 1, so it's a single closed traversal.
        assert g.metadata["compound"] is False

    def test_compound_hexagram(self):
        # {6/2} = two overlapping triangles (Star of David).
        g = star_polygon(n=6, k=2, radius=1.0)
        assert g.geom_type == "polygon_set"
        assert isinstance(g.coordinates, list)
        assert len(g.coordinates) == 2
        assert all(arr.shape == (3, 2) for arr in g.coordinates)
        assert g.metadata["n_components"] == 2
        assert g.metadata["vertices_per_component"] == 3

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            star_polygon(n=2, k=1)

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            star_polygon(n=5, k=5)
        with pytest.raises(ValueError):
            star_polygon(n=5, k=0)

    def test_radius_scales_vertices(self):
        g = star_polygon(n=8, k=3, radius=2.5)
        radii = np.linalg.norm(g.coordinates, axis=1)
        np.testing.assert_allclose(radii, 2.5, atol=1e-12)


class TestTimesTableCircle:
    def test_integer_multiplier(self):
        g = times_table_circle(n_points=20, multiplier=2)
        assert g.geom_type == "graph"
        assert g.coordinates.shape == (20, 2)
        # i=0 is a self-loop (0 * 2 = 0 mod 20), so 19 edges remain.
        assert g.edges.shape == (19, 2)

    def test_edges_index_into_coordinates(self):
        g = times_table_circle(n_points=12, multiplier=5)
        n = g.coordinates.shape[0]
        assert np.all(g.edges >= 0)
        assert np.all(g.edges < n)

    def test_vertex_radii(self):
        g = times_table_circle(n_points=30, multiplier=7, radius=3.0)
        radii = np.linalg.norm(g.coordinates, axis=1)
        np.testing.assert_allclose(radii, 3.0, atol=1e-12)

    def test_invalid_n(self):
        with pytest.raises(ValueError):
            times_table_circle(n_points=1, multiplier=2)


class TestTimesTableFromInput:
    def test_returns_graph(self):
        from biotuner.harmonic_geometry import times_table_from_input
        inp = HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)])
        g = times_table_from_input(inp, n_points=60, mode="ratio")
        assert g.geom_type == "graph"
        assert g.coordinates.shape == (60, 2)
        assert g.metadata["n_ratios"] == 3
        # ratio_index aligns with edges
        assert len(g.metadata["ratio_index"]) == g.edges.shape[0]

    @pytest.mark.parametrize("mode", ["ratio", "pitch_class", "integer"])
    def test_modes(self, mode):
        from biotuner.harmonic_geometry import times_table_from_input
        inp = HarmonicInput(ratios=[Fraction(3, 2), Fraction(5, 4)])
        g = times_table_from_input(inp, n_points=40, mode=mode)
        assert g.geom_type == "graph"

    def test_invalid_mode(self):
        from biotuner.harmonic_geometry import times_table_from_input
        inp = HarmonicInput(ratios=[Fraction(3, 2)])
        with pytest.raises(ValueError):
            times_table_from_input(inp, n_points=40, mode="bogus")


class TestTuningCircle:
    def test_unison_at_zero_angle(self):
        # Ratio 1 has log_ratio = 0, so it should land at angle 0 (point (R, 0)).
        inp = HarmonicInput(ratios=[1, Fraction(3, 2)])
        g = tuning_circle(inp, radius=1.0)
        assert g.geom_type == "point_cloud_2d"
        assert g.coordinates.shape == (2, 2)
        np.testing.assert_allclose(g.coordinates[0], [1.0, 0.0], atol=1e-12)

    def test_octave_wraps_to_zero(self):
        # Ratio 2 (one octave above) wraps back to angle 0 because
        # pitch_class = log_2(2) % 1 = 0.
        inp = HarmonicInput(ratios=[Fraction(2, 1)])
        g = tuning_circle(inp)
        np.testing.assert_allclose(g.coordinates[0], [1.0, 0.0], atol=1e-12)

    def test_weights_match_amplitudes(self):
        inp = HarmonicInput(
            ratios=[1, Fraction(3, 2), 2],
            amplitudes=[3.0, 1.0, 1.0],
        )
        g = tuning_circle(inp)
        # weights should be normalized amplitudes.
        np.testing.assert_allclose(g.weights, [3 / 5, 1 / 5, 1 / 5])

    def test_radius(self):
        inp = HarmonicInput(ratios=[Fraction(5, 4)])
        g = tuning_circle(inp, radius=2.0)
        radii = np.linalg.norm(g.coordinates, axis=1)
        np.testing.assert_allclose(radii, 2.0, atol=1e-12)


class TestRoseCurve:
    def test_shape_and_type(self):
        g = rose_curve(ratio=Fraction(3, 2), n_points=500)
        assert g.geom_type == "curve_2d"
        assert g.coordinates.shape == (500, 2)

    def test_p_q_even_petals(self):
        # p+q even (e.g. 3/1): closes after qπ, p petals.
        g = rose_curve(ratio=Fraction(3, 1), n_points=200)
        assert g.metadata["petals"] == 3

    def test_p_q_odd_petals(self):
        # p+q odd (e.g. 2/1): closes after 2qπ, 2p petals.
        g = rose_curve(ratio=Fraction(2, 1), n_points=200)
        assert g.metadata["petals"] == 4

    def test_radius_bounds_curve(self):
        g = rose_curve(ratio=Fraction(3, 1), n_points=400, radius=2.5)
        max_r = np.max(np.linalg.norm(g.coordinates, axis=1))
        assert max_r <= 2.5 + 1e-9

    def test_invalid_n_points(self):
        with pytest.raises(ValueError):
            rose_curve(ratio=Fraction(3, 2), n_points=1)


class TestEpiHypocycloid:
    def test_epicycloid_shape(self):
        g = epicycloid(ratio=Fraction(3, 1), n_points=500)
        assert g.geom_type == "curve_2d"
        assert g.coordinates.shape == (500, 2)
        assert g.metadata["cusps"] == 3

    def test_epicycloid_closes(self):
        g = epicycloid(ratio=Fraction(5, 2), n_points=2000)
        np.testing.assert_allclose(g.coordinates[0], g.coordinates[-1], atol=1e-9)

    def test_hypocycloid_astroid(self):
        # Ratio 4:1 produces an astroid with 3 cusps... actually 4-1=3 cusps.
        g = hypocycloid(ratio=Fraction(4, 1), n_points=500)
        assert g.metadata["cusps"] == 3

    def test_hypocycloid_rejects_ratio_le_1(self):
        with pytest.raises(ValueError):
            hypocycloid(ratio=Fraction(2, 3))
        with pytest.raises(ValueError):
            hypocycloid(ratio=1)
