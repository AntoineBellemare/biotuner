"""Tests for biotuner.harmonic_geometry.generative (Phase 5)."""

from fractions import Fraction

import numpy as np
import pytest

from biotuner.harmonic_geometry import HarmonicInput
from biotuner.harmonic_geometry.generative import (
    lsystem_from_ratios,
    recursive_polygon,
    self_similar_tuning,
)


# ─────────────────────────────────────────── fixtures ────────────────────────


@pytest.fixture
def major():
    return HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)])


@pytest.fixture
def dom7():
    return HarmonicInput(
        ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2), Fraction(7, 4)]
    )


# ═══════════════════════════════════════ lsystem_from_ratios ═════════════════


class TestLsystemFromRatios:
    def test_returns_graph(self, major):
        g = lsystem_from_ratios(major, depth=3)
        assert g.geom_type == "graph"

    def test_coordinates_shape(self, major):
        g = lsystem_from_ratios(major, depth=3)
        assert g.coordinates.ndim == 2
        assert g.coordinates.shape[1] == 2

    def test_edges_reference_valid_indices(self, major):
        g = lsystem_from_ratios(major, depth=3)
        n = g.coordinates.shape[0]
        assert g.edges.shape[1] == 2
        assert int(g.edges.max()) < n

    def test_more_depth_more_segments(self, major):
        g3 = lsystem_from_ratios(major, depth=3)
        g4 = lsystem_from_ratios(major, depth=4)
        assert g4.metadata["n_segments"] > g3.metadata["n_segments"]

    def test_custom_rules_accepted(self, major):
        g = lsystem_from_ratios(major, depth=2, rules={"F": "F+F-F"})
        assert g.geom_type == "graph"

    def test_metadata_lstring_preview(self, major):
        g = lsystem_from_ratios(major, depth=2)
        assert "lstring_preview" in g.metadata
        assert isinstance(g.metadata["lstring_preview"], str)

    def test_invalid_depth_too_low(self, major):
        with pytest.raises(ValueError):
            lsystem_from_ratios(major, depth=0)

    def test_invalid_depth_too_high(self, major):
        with pytest.raises(ValueError):
            lsystem_from_ratios(major, depth=8)

    def test_empty_axiom_raises(self, major):
        with pytest.raises(ValueError):
            lsystem_from_ratios(major, axiom="")

    def test_four_component_more_branches(self, major, dom7):
        g3 = lsystem_from_ratios(major, depth=3)
        g4 = lsystem_from_ratios(dom7, depth=3)
        # dom7 has one more component → more branches → more segments at same depth
        assert g4.metadata["n_segments"] > g3.metadata["n_segments"]

    def test_peaks_input_works(self):
        inp = HarmonicInput(peaks=[200.0, 250.0, 300.0])
        g = lsystem_from_ratios(inp, depth=3)
        assert g.geom_type == "graph"


# ══════════════════════════════════════ recursive_polygon ════════════════════


class TestRecursivePolygon:
    def test_returns_curve_2d(self, major):
        g = recursive_polygon(major, depth=2)
        assert g.geom_type == "curve_2d"

    def test_curve_is_closed(self, major):
        g = recursive_polygon(major, depth=2)
        # First and last vertex should match (closed boundary).
        np.testing.assert_allclose(
            g.coordinates[0], g.coordinates[-1], atol=1e-10
        )

    def test_depth_multiplies_vertices(self, major):
        g1 = recursive_polygon(major, depth=1)
        g2 = recursive_polygon(major, depth=2)
        n1 = g1.coordinates.shape[0] - 1   # subtract closing vertex
        n2 = g2.coordinates.shape[0] - 1
        assert n2 == n1 * 4   # Koch rule: 4× per level

    def test_n_sides_respected(self, major):
        g = recursive_polygon(major, depth=1, n_sides=5)
        # After 1 subdivision: 5 * 4 = 20 edges + 1 closing vertex = 21 points
        assert g.coordinates.shape[0] == 21

    def test_explicit_scale_factor(self, major):
        g = recursive_polygon(major, depth=2, scale_factor=0.25)
        assert g.geom_type == "curve_2d"

    def test_invalid_depth(self, major):
        with pytest.raises(ValueError):
            recursive_polygon(major, depth=0)

    def test_invalid_n_sides(self, major):
        with pytest.raises(ValueError):
            recursive_polygon(major, depth=1, n_sides=2)

    def test_invalid_scale_factor(self, major):
        with pytest.raises(ValueError):
            recursive_polygon(major, depth=1, scale_factor=0.6)


# ═══════════════════════════════════════ self_similar_tuning ═════════════════


class TestSelfSimilarTuning:
    def test_returns_graph(self, major):
        g = self_similar_tuning(major, n_levels=3)
        assert g.geom_type == "graph"

    def test_level_0_node_count_matches_n_components(self, major):
        g = self_similar_tuning(major, n_levels=3)
        n0 = g.metadata["n_nodes_per_level"][0]
        assert n0 == major.n_components()

    def test_deeper_levels_have_more_nodes(self, major):
        g = self_similar_tuning(major, n_levels=4)
        counts = g.metadata["n_nodes_per_level"]
        # Each level should have >= as many pitches as the previous.
        for i in range(3):
            assert counts[i + 1] >= counts[i]

    def test_node_positions_inside_unit_disk(self, major):
        g = self_similar_tuning(major, n_levels=4)
        radii = np.linalg.norm(g.coordinates, axis=1)
        assert float(np.max(radii)) <= 1.0 + 1e-9

    def test_edge_count_equals_non_seed_nodes(self, major):
        g = self_similar_tuning(major, n_levels=3)
        counts = g.metadata["n_nodes_per_level"]
        expected_edges = sum(counts[1:])
        assert int(g.edges.shape[0]) == expected_edges

    def test_weights_are_positive(self, major):
        g = self_similar_tuning(major, n_levels=3)
        assert np.all(g.weights > 0)

    def test_custom_equave(self, major):
        g = self_similar_tuning(major, n_levels=2, equave=3.0)
        assert g.parameters["equave"] == 3.0

    def test_invalid_n_levels(self, major):
        with pytest.raises(ValueError):
            self_similar_tuning(major, n_levels=0)

    def test_invalid_equave(self, major):
        with pytest.raises(ValueError):
            self_similar_tuning(major, equave=0.5)

    def test_pitches_stay_in_equave(self, major):
        g = self_similar_tuning(major, n_levels=3, equave=2.0)
        pitches = g.metadata["pitches_level_0"]
        for p in pitches:
            assert 1.0 <= p < 2.0 + 1e-9
