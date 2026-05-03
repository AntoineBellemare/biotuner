"""Tests for biotuner.harmonic_geometry.fractal (Phase 4 deterministic group)."""

from fractions import Fraction

import numpy as np
import pytest

from biotuner.harmonic_geometry import HarmonicInput
from biotuner.harmonic_geometry.fractal import (
    continued_fraction_rectangles,
    farey_sequence_layout,
    ifs_harmonic,
    stern_brocot_tree,
    subharmonic_tree,
)


# ============================================================== Stern-Brocot


class TestSternBrocotTree:
    def test_node_count_matches_2d_minus_1(self):
        for d in range(1, 6):
            g = stern_brocot_tree(max_depth=d)
            assert g.coordinates.shape == (2 ** d - 1, 2)
            assert g.edges.shape == (2 ** d - 2, 2)

    def test_root_is_unison(self):
        g = stern_brocot_tree(max_depth=4)
        # Root is the first inserted mediant, ratio 1/1.
        ratios = g.metadata["ratios"]
        assert ratios[0] == "1"

    def test_harmonicity_attached(self):
        g = stern_brocot_tree(max_depth=3)
        assert g.weights.shape == (g.coordinates.shape[0],)
        assert np.all(np.isfinite(g.weights))

    def test_hyperbolic_layout_inside_disk(self):
        g = stern_brocot_tree(max_depth=5, layout="hyperbolic")
        radii = np.linalg.norm(g.coordinates, axis=1)
        assert np.all(radii < 1.0 + 1e-12)

    def test_tree_layout_depth_on_y(self):
        g = stern_brocot_tree(max_depth=4, layout="tree")
        ys = g.coordinates[:, 1]
        depths = np.array(g.metadata["depth_per_node"])
        # Larger depth -> more negative y (or equal).
        # Sort by depth and check monotonic non-increasing y.
        order = np.argsort(depths)
        ys_sorted = ys[order]
        diffs = np.diff(ys_sorted)
        assert np.all(diffs <= 1e-12)

    def test_with_input_attaches_distance_metadata(self):
        inp = HarmonicInput(ratios=[1, Fraction(3, 2), Fraction(5, 4)])
        g = stern_brocot_tree(input=inp, max_depth=4)
        assert "nearest_input_dist_cents" in g.metadata
        dists = np.asarray(g.metadata["nearest_input_dist_cents"])
        assert dists.shape == (g.coordinates.shape[0],)
        # Some node (the root 1/1) should be exactly 0 cents from input ratio 1.
        assert dists.min() < 1e-9

    def test_invalid_max_depth(self):
        with pytest.raises(ValueError):
            stern_brocot_tree(max_depth=0)

    def test_invalid_layout(self):
        with pytest.raises(ValueError):
            stern_brocot_tree(layout="cubist")


# ===================================================== continued_fraction_rectangles


class TestContinuedFractionRectangles:
    def test_returns_polygon_set(self):
        g = continued_fraction_rectangles(Fraction(7, 3), depth=10)
        assert g.geom_type == "polygon_set"
        assert isinstance(g.coordinates, list)
        for rect in g.coordinates:
            assert rect.shape == (4, 2)

    def test_inversion_handled_for_ratio_below_one(self):
        g = continued_fraction_rectangles(Fraction(3, 7), depth=10)
        assert g.metadata["inverted"] is True

    def test_squares_count_for_simple_ratio(self):
        # 7 / 3 = 2 + 1 / (3 / 1) -> CF [2; 3]. Total squares = 2 + 3 = 5.
        g = continued_fraction_rectangles(Fraction(7, 3), depth=10)
        assert g.metadata["n_squares"] == 5

    def test_invalid_depth(self):
        with pytest.raises(ValueError):
            continued_fraction_rectangles(Fraction(3, 2), depth=0)


# ============================================================ Farey sequence


class TestFareySequenceLayout:
    def test_known_lengths(self):
        # |F_n| = 1 + sum_{k=1..n} phi(k); we don't recompute phi here, just
        # assert known values: |F_1| = 2, |F_5| = 11, |F_8| = 23.
        for order, expected_len in [(1, 2), (5, 11), (8, 23)]:
            g = farey_sequence_layout(order)
            assert g.metadata["n_terms"] == expected_len

    def test_circle_layout_on_unit_circle(self):
        g = farey_sequence_layout(8, layout="circle")
        radii = np.linalg.norm(g.coordinates, axis=1)
        np.testing.assert_allclose(radii, 1.0, atol=1e-12)

    def test_line_layout_on_x_axis(self):
        g = farey_sequence_layout(7, layout="line")
        np.testing.assert_allclose(g.coordinates[:, 1], 0.0, atol=1e-12)
        # x ranges over [-1, 1].
        assert g.coordinates[:, 0].min() == pytest.approx(-1.0, abs=1e-12)
        assert g.coordinates[:, 0].max() == pytest.approx(1.0, abs=1e-12)

    def test_invalid_order(self):
        with pytest.raises(ValueError):
            farey_sequence_layout(0)

    def test_invalid_layout(self):
        with pytest.raises(ValueError):
            farey_sequence_layout(4, layout="hyperbolic")


# =========================================================== subharmonic_tree


class TestSubharmonicTree:
    def test_grows_geometrically(self):
        inp = HarmonicInput(peaks=[1000.0, 1500.0])
        g = subharmonic_tree(inp, depth=2, n_harmonics=3)
        # Root: 2 nodes. Level 1: 2 * 3 = 6. Level 2: 6 * 3 = 18. Total 26.
        assert g.metadata["n_nodes"] == 2 + 6 + 18

    def test_min_freq_prunes(self):
        # With min_freq close to root, no expansion happens.
        inp = HarmonicInput(peaks=[10.0])
        g = subharmonic_tree(inp, depth=3, n_harmonics=4, min_freq=6.0)
        # Root: 10 Hz. Subharmonics: 5, 3.33, 2.5, 2.0 -> all below 6.0. No expansion.
        assert g.metadata["n_nodes"] == 1
        assert g.edges.shape == (0, 2)

    def test_invalid_depth(self):
        inp = HarmonicInput(peaks=[100.0])
        with pytest.raises(ValueError):
            subharmonic_tree(inp, depth=0)

    def test_polar_layout(self):
        inp = HarmonicInput(peaks=[400.0, 500.0, 600.0])
        g = subharmonic_tree(inp, depth=3, n_harmonics=3, layout="polar")
        # Coordinates inside unit disk
        radii = np.linalg.norm(g.coordinates, axis=1)
        assert radii.max() <= 1.0 + 1e-9
        assert "root_index_per_node" in g.metadata
        assert len(g.metadata["root_index_per_node"]) == g.coordinates.shape[0]

    def test_invalid_layout(self):
        inp = HarmonicInput(peaks=[100.0])
        with pytest.raises(ValueError):
            subharmonic_tree(inp, depth=2, layout="cubist")


# =================================================================== IFS chaos


class TestIFSHarmonic:
    def test_shape_and_determinism(self, simple_ratios):
        inp = HarmonicInput(ratios=simple_ratios, base_freq=100.0)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        g1 = ifs_harmonic(inp, n_points=2000, rng=rng1)
        g2 = ifs_harmonic(inp, n_points=2000, rng=rng2)
        assert g1.coordinates.shape == (2000, 2)
        np.testing.assert_array_equal(g1.coordinates, g2.coordinates)

    def test_different_seeds_differ(self, simple_ratios):
        inp = HarmonicInput(ratios=simple_ratios, base_freq=100.0)
        a = ifs_harmonic(inp, n_points=1000, rng=np.random.default_rng(1))
        b = ifs_harmonic(inp, n_points=1000, rng=np.random.default_rng(2))
        assert not np.allclose(a.coordinates, b.coordinates)

    def test_attractor_bounded(self, simple_ratios):
        inp = HarmonicInput(ratios=simple_ratios)
        g = ifs_harmonic(inp, n_points=5000, contraction="fixed_half",
                         rng=np.random.default_rng(0))
        # Sierpinski-like attractor with vertices on unit circle stays within
        # the unit disk.
        radii = np.linalg.norm(g.coordinates, axis=1)
        assert radii.max() <= 1.0 + 1e-9

    def test_unknown_contraction(self, simple_ratios):
        inp = HarmonicInput(ratios=simple_ratios)
        with pytest.raises(ValueError):
            ifs_harmonic(inp, n_points=200, contraction="weird")

    def test_too_few_components(self):
        inp = HarmonicInput(ratios=[1])
        with pytest.raises(ValueError):
            ifs_harmonic(inp, n_points=200)

    def test_too_few_points(self, simple_ratios):
        inp = HarmonicInput(ratios=simple_ratios)
        with pytest.raises(ValueError):
            ifs_harmonic(inp, n_points=10)


# ============================================================ Farey Ford circles


class TestFareyFordCircles:
    def test_returns_polygon_set(self):
        g = farey_sequence_layout(order=5, layout="ford")
        assert g.geom_type == "polygon_set"
        # F_5 has 11 fractions in [0, 1]
        assert g.metadata["n_terms"] == 11
        assert "centres" in g.metadata
        assert "radii" in g.metadata

    def test_radii_match_denominators(self):
        g = farey_sequence_layout(order=5, layout="ford")
        radii   = np.asarray(g.metadata["radii"])
        centres = np.asarray(g.metadata["centres"])
        # circles tangent to x-axis: y_centre == radius
        np.testing.assert_allclose(centres[:, 1], radii, atol=1e-12)

    def test_invalid_layout(self):
        with pytest.raises(ValueError):
            farey_sequence_layout(order=5, layout="bogus")
