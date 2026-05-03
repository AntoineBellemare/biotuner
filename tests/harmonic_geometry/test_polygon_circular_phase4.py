"""Tests for the Phase 4 metric-driven polygon_circular variants."""

from fractions import Fraction

import numpy as np
import pytest

from biotuner.harmonic_geometry import HarmonicInput
from biotuner.harmonic_geometry.polygon_circular import (
    consonance_polygon,
    interval_vector_diagram,
    polygon_chord_pattern,
)


# -------------------------------------------------------- interval_vector_diagram


class TestIntervalVectorDiagram:
    def test_basic_graph_shape(self, simple_ratios):
        inp = HarmonicInput(ratios=simple_ratios, base_freq=100.0)
        g = interval_vector_diagram(inp)
        n = inp.n_components()
        assert g.geom_type == "graph"
        assert g.coordinates.shape == (n, 2)
        # Complete graph: n*(n-1)/2 pairs.
        assert g.edges.shape == (n * (n - 1) // 2, 2)
        assert g.weights.shape == (g.edges.shape[0],)

    def test_weights_count_interval_class_multiplicities(self):
        # An evenly-spaced 4-component scale (2:3:4:5 ratios) creates
        # repeated intervals; the smallest weight should be >= 1.
        inp = HarmonicInput(ratios=[1, Fraction(3, 2), Fraction(4, 3), Fraction(5, 4)])
        g = interval_vector_diagram(inp, bin_cents=20.0)
        # All weights are interval-class counts and must be >= 1.
        assert np.all(g.weights >= 1)

    def test_invalid_bin_cents(self):
        inp = HarmonicInput(ratios=[1, 2])
        with pytest.raises(ValueError):
            interval_vector_diagram(inp, bin_cents=0.0)

    def test_single_component_rejected(self):
        inp = HarmonicInput(ratios=[1])
        with pytest.raises(ValueError):
            interval_vector_diagram(inp)


# ---------------------------------------------------------- polygon_chord_pattern


class TestPolygonChordPattern:
    def test_complete_graph_default(self, simple_ratios):
        inp = HarmonicInput(ratios=simple_ratios, base_freq=100.0)
        g = polygon_chord_pattern(inp)
        n = inp.n_components()
        assert g.edges.shape == (n * (n - 1) // 2, 2)

    def test_threshold_drops_weak_pairs(self):
        inp = HarmonicInput(
            ratios=[1, Fraction(3, 2), Fraction(5, 4), Fraction(11, 7)],
            base_freq=100.0,
        )
        g_all = polygon_chord_pattern(inp)
        # dyad_similarity max is 100 (unison). Threshold above any plausible
        # non-unison pair should drop everything.
        g_high = polygon_chord_pattern(inp, threshold=200.0)
        assert g_high.edges.shape[0] == 0
        # Threshold of 0 keeps the same pairs as no threshold.
        g_low = polygon_chord_pattern(inp, threshold=0.0)
        assert g_low.edges.shape[0] == g_all.edges.shape[0]

    def test_callable_metric(self):
        inp = HarmonicInput(ratios=[1, Fraction(3, 2), Fraction(5, 4)])
        # Identity-like metric: just returns the ratio quotient unchanged.
        g = polygon_chord_pattern(inp, metric=lambda r: float(r))
        assert g.edges.shape[0] == 3
        assert np.all(np.isfinite(g.weights))

    def test_unknown_metric_raises(self):
        inp = HarmonicInput(ratios=[1, 2])
        with pytest.raises(ValueError):
            polygon_chord_pattern(inp, metric="nonsense")

    @pytest.mark.parametrize(
        "metric",
        ["dyad_similarity", "compute_consonance", "tenneyHeight", "log_distance"],
    )
    def test_all_named_metrics_work(self, metric):
        inp = HarmonicInput(ratios=[1, Fraction(3, 2), Fraction(5, 4)])
        g = polygon_chord_pattern(inp, metric=metric)
        assert g.edges.shape[0] == 3
        assert np.all(np.isfinite(g.weights))


# ---------------------------------------------------------------- consonance_polygon


class TestConsonancePolygon:
    def test_polygon_shape(self, simple_ratios):
        inp = HarmonicInput(ratios=simple_ratios, base_freq=100.0)
        g = consonance_polygon(inp)
        n = inp.n_components()
        assert g.geom_type == "polygon"
        assert g.coordinates.shape == (n, 2)
        # Vertices live on the unit circle.
        radii = np.linalg.norm(g.coordinates, axis=1)
        np.testing.assert_allclose(radii, 1.0, atol=1e-12)

    def test_weights_carry_consonance_share(self, simple_ratios):
        inp = HarmonicInput(ratios=simple_ratios, base_freq=100.0)
        g = consonance_polygon(inp)
        assert g.weights.shape == (inp.n_components(),)
        assert np.all(np.isfinite(g.weights))

    def test_first_vertex_anchored(self):
        inp = HarmonicInput(
            ratios=[1, Fraction(3, 2), Fraction(5, 4), Fraction(7, 4)],
        )
        g = consonance_polygon(inp)
        # First vertex anchored at angle 0 -> (radius, 0).
        np.testing.assert_allclose(g.coordinates[0], [1.0, 0.0], atol=1e-12)

    def test_too_few_components(self):
        inp = HarmonicInput(ratios=[1, 2])
        with pytest.raises(ValueError):
            consonance_polygon(inp)
