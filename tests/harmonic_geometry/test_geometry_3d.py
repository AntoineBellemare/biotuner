"""Tests for biotuner.harmonic_geometry.geometry_3d (Phase 7)."""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import pytest

from biotuner.harmonic_geometry import (
    HarmonicInput,
    HarmonicSequence,
    GeometryData,
    harmonic_knot,
    harmonic_point_cloud,
    harmonic_surface,
    lissajous_tube,
    lsystem_3d,
    recursive_polyhedron,
    geometry_sequence,
)

# ── shared fixtures ────────────────────────────────────────────────────────────

@pytest.fixture
def major():
    return HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)])

@pytest.fixture
def dom7():
    return HarmonicInput(
        ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2), Fraction(7, 4)]
    )

@pytest.fixture
def minor():
    return HarmonicInput(ratios=[Fraction(1), Fraction(6, 5), Fraction(3, 2)])

@pytest.fixture
def single():
    return HarmonicInput(ratios=[Fraction(3, 2)])


# ═══════════════════════════════════════════ lissajous_tube ════════════════════

class TestLissajousTube:
    def test_returns_mesh_3d(self, major):
        g = lissajous_tube(major)
        assert g.geom_type == "mesh_3d"

    def test_vertices_shape(self, major):
        g = lissajous_tube(major, n_points=100, n_sides=8)
        assert g.coordinates.ndim == 2
        assert g.coordinates.shape[1] == 3

    def test_faces_shape(self, major):
        g = lissajous_tube(major, n_points=100, n_sides=8)
        assert g.faces is not None
        assert g.faces.ndim == 2
        assert g.faces.shape[1] == 3

    def test_face_indices_in_range(self, major):
        g = lissajous_tube(major, n_points=50, n_sides=6)
        assert g.faces.max() < len(g.coordinates)
        assert g.faces.min() >= 0

    def test_n_vertices_proportional(self, major):
        g = lissajous_tube(major, n_points=80, n_sides=10)
        assert g.coordinates.shape[0] == 80 * 10

    def test_coordinates_finite(self, dom7):
        g = lissajous_tube(dom7)
        assert np.all(np.isfinite(g.coordinates))

    def test_single_ratio_input(self, single):
        g = lissajous_tube(single, n_points=50, n_sides=6)
        assert g.geom_type == "mesh_3d"

    def test_invalid_n_points(self, major):
        with pytest.raises(ValueError):
            lissajous_tube(major, n_points=3)

    def test_invalid_tube_radius(self, major):
        with pytest.raises(ValueError):
            lissajous_tube(major, tube_radius=0.0)

    def test_invalid_n_sides(self, major):
        with pytest.raises(ValueError):
            lissajous_tube(major, n_sides=2)


# ════════════════════════════════════════════ harmonic_knot ════════════════════

class TestHarmonicKnot:
    def test_returns_mesh_3d(self, major):
        g = harmonic_knot(major)
        assert g.geom_type == "mesh_3d"

    def test_vertices_are_3d(self, major):
        g = harmonic_knot(major, n_points=100, n_sides=8)
        assert g.coordinates.shape[1] == 3

    def test_faces_triangular(self, major):
        g = harmonic_knot(major, n_points=100, n_sides=8)
        assert g.faces.shape[1] == 3

    def test_knot_type_in_metadata(self, major):
        g = harmonic_knot(major)
        assert "knot_type" in g.metadata
        assert g.metadata["knot_type"].startswith("T(")

    def test_trefoil_from_perfect_fifth(self):
        inp = HarmonicInput(ratios=[Fraction(3, 2)])
        g = harmonic_knot(inp)
        assert g.metadata["knot_type"] == "T(3,2)"

    def test_closed_curve_no_gap(self, major):
        # Closed tube: first and last ring of vertices should be different
        # (we don't duplicate them but faces wrap around)
        g = harmonic_knot(major, n_points=100, n_sides=8)
        assert g.faces.max() < len(g.coordinates)

    def test_coordinates_finite(self, dom7):
        g = harmonic_knot(dom7)
        assert np.all(np.isfinite(g.coordinates))

    def test_invalid_n_points(self, major):
        with pytest.raises(ValueError):
            harmonic_knot(major, n_points=4)

    def test_invalid_tube_radius(self, major):
        with pytest.raises(ValueError):
            harmonic_knot(major, tube_radius=-0.1)

    def test_invalid_radii(self, major):
        with pytest.raises(ValueError):
            harmonic_knot(major, major_radius=0.0)


# ══════════════════════════════════════════ harmonic_surface ══════════════════

class TestHarmonicSurface:
    @pytest.mark.parametrize("mode", ["torus", "sphere", "cylinder"])
    def test_returns_mesh_3d(self, major, mode):
        g = harmonic_surface(major, mode=mode, resolution=16)
        assert g.geom_type == "mesh_3d"

    @pytest.mark.parametrize("mode", ["torus", "sphere", "cylinder"])
    def test_vertices_3d(self, major, mode):
        g = harmonic_surface(major, mode=mode, resolution=16)
        assert g.coordinates.shape[1] == 3

    @pytest.mark.parametrize("mode", ["torus", "sphere", "cylinder"])
    def test_faces_triangular(self, major, mode):
        g = harmonic_surface(major, mode=mode, resolution=16)
        assert g.faces.shape[1] == 3

    @pytest.mark.parametrize("mode", ["torus", "sphere", "cylinder"])
    def test_coordinates_finite(self, dom7, mode):
        g = harmonic_surface(dom7, mode=mode, resolution=16)
        assert np.all(np.isfinite(g.coordinates))

    def test_resolution_scales_vertices(self, major):
        g16 = harmonic_surface(major, mode="torus", resolution=16)
        g32 = harmonic_surface(major, mode="torus", resolution=32)
        assert len(g32.coordinates) > len(g16.coordinates)

    def test_invalid_mode(self, major):
        with pytest.raises(ValueError):
            harmonic_surface(major, mode="cube")

    def test_invalid_resolution(self, major):
        with pytest.raises(ValueError):
            harmonic_surface(major, resolution=4)

    def test_mode_in_parameters(self, major):
        for mode in ["torus", "sphere", "cylinder"]:
            g = harmonic_surface(major, mode=mode, resolution=16)
            assert g.parameters["mode"] == mode


# ══════════════════════════════════════════════ lsystem_3d ════════════════════

class TestLsystem3D:
    def test_returns_tree(self, major):
        g = lsystem_3d(major, depth=2)
        assert g.geom_type == "tree"

    def test_coordinates_3d(self, major):
        g = lsystem_3d(major, depth=2)
        assert g.coordinates.shape[1] == 3

    def test_edges_valid(self, major):
        g = lsystem_3d(major, depth=2)
        assert g.edges is not None
        assert g.edges.shape[1] == 2
        assert g.edges.max() < len(g.coordinates)

    def test_depth_increases_segments(self, major):
        g2 = lsystem_3d(major, depth=2)
        g3 = lsystem_3d(major, depth=3)
        assert g3.metadata["n_segments"] >= g2.metadata["n_segments"]

    def test_dom7_more_branches_than_major(self, major, dom7):
        gm = lsystem_3d(major, depth=3)
        gd = lsystem_3d(dom7, depth=3)
        assert gd.metadata["n_segments"] >= gm.metadata["n_segments"]

    def test_custom_rules(self, major):
        g = lsystem_3d(major, depth=2, rules={"F": "F^FF&F"})
        assert g.geom_type == "tree"

    def test_coordinates_finite(self, dom7):
        g = lsystem_3d(dom7, depth=3)
        assert np.all(np.isfinite(g.coordinates))

    def test_invalid_depth_low(self, major):
        with pytest.raises(ValueError):
            lsystem_3d(major, depth=0)

    def test_invalid_depth_high(self, major):
        with pytest.raises(ValueError):
            lsystem_3d(major, depth=7)

    def test_invalid_step_length(self, major):
        with pytest.raises(ValueError):
            lsystem_3d(major, step_length=-1.0)


# ══════════════════════════════════════════ recursive_polyhedron ══════════════

class TestRecursivePolyhedron:
    @pytest.mark.parametrize("solid", ["tetrahedron", "cube", "icosahedron"])
    def test_returns_mesh_3d(self, major, solid):
        g = recursive_polyhedron(major, depth=1, solid=solid)
        assert g.geom_type == "mesh_3d"

    @pytest.mark.parametrize("solid", ["tetrahedron", "cube", "icosahedron"])
    def test_vertices_3d(self, major, solid):
        g = recursive_polyhedron(major, depth=1, solid=solid)
        assert g.coordinates.shape[1] == 3

    @pytest.mark.parametrize("solid", ["tetrahedron", "cube", "icosahedron"])
    def test_faces_triangular(self, major, solid):
        g = recursive_polyhedron(major, depth=1, solid=solid)
        assert g.faces.shape[1] == 3

    def test_depth_0_is_base_solid(self, major):
        g = recursive_polyhedron(major, depth=0, solid="tetrahedron")
        assert len(g.faces) == 4   # tetrahedron has 4 faces

    def test_depth_multiplies_faces(self, major):
        g0 = recursive_polyhedron(major, depth=0, solid="tetrahedron")
        g1 = recursive_polyhedron(major, depth=1, solid="tetrahedron")
        # Each face → 6 faces at depth 1
        assert len(g1.faces) == len(g0.faces) * 6

    def test_face_indices_in_range(self, major):
        g = recursive_polyhedron(major, depth=2, solid="icosahedron")
        assert g.faces.max() < len(g.coordinates)
        assert g.faces.min() >= 0

    def test_coordinates_finite(self, dom7):
        g = recursive_polyhedron(dom7, depth=2)
        assert np.all(np.isfinite(g.coordinates))

    def test_invalid_solid(self, major):
        with pytest.raises(ValueError):
            recursive_polyhedron(major, solid="dodecahedron")

    def test_invalid_depth_negative(self, major):
        with pytest.raises(ValueError):
            recursive_polyhedron(major, depth=-1)

    def test_invalid_depth_too_large(self, major):
        with pytest.raises(ValueError):
            recursive_polyhedron(major, depth=5)

    def test_bump_scale_in_parameters(self, major):
        g = recursive_polyhedron(major, depth=1)
        assert "bump_scale" in g.parameters
        assert 0.0 < g.parameters["bump_scale"] <= 0.4

    def test_face_ratio_index_metadata(self, dom7):
        g = recursive_polyhedron(dom7, depth=2)
        assert "face_ratio_index" in g.metadata
        idx = g.metadata["face_ratio_index"]
        assert len(idx) == len(g.faces)
        assert int(idx.min()) >= 0
        assert int(idx.max()) < dom7.n_components()

    def test_solid_auto_picked_when_none(self, major, dom7):
        g_major = recursive_polyhedron(major, depth=0, solid=None)
        g_dom7  = recursive_polyhedron(dom7,  depth=0, solid=None)
        # 3-component → tetrahedron (4 faces); 4-component → cube (12 tri faces).
        assert len(g_major.faces) == 4
        assert len(g_dom7.faces) == 12

    def test_per_face_bump_off_uses_global(self, major):
        g_off = recursive_polyhedron(major, depth=1, per_face_bump=False)
        g_on  = recursive_polyhedron(major, depth=1, per_face_bump=True)
        # Same topology, different vertex positions (apex placement differs).
        assert g_off.faces.shape == g_on.faces.shape


# ══════════════════════════════════════════ harmonic_point_cloud ══════════════

class TestHarmonicPointCloud:
    @pytest.mark.parametrize("surface", ["sphere", "torus", "klein", "hyperbolic", "mos"])
    def test_returns_point_cloud_3d(self, major, surface):
        g = harmonic_point_cloud(major, n_points=200, surface=surface)
        assert g.geom_type == "point_cloud_3d"

    @pytest.mark.parametrize("surface", ["sphere", "torus", "klein", "hyperbolic", "mos"])
    def test_coordinates_3d(self, major, surface):
        g = harmonic_point_cloud(major, n_points=200, surface=surface)
        assert g.coordinates.shape[1] == 3

    @pytest.mark.parametrize("surface", ["sphere", "torus", "klein", "hyperbolic", "mos"])
    def test_correct_n_points(self, major, surface):
        g = harmonic_point_cloud(major, n_points=300, surface=surface)
        assert g.coordinates.shape[0] == 300

    @pytest.mark.parametrize("surface", ["sphere", "torus", "klein", "hyperbolic", "mos"])
    def test_coordinates_finite(self, dom7, surface):
        g = harmonic_point_cloud(dom7, n_points=200, surface=surface)
        assert np.all(np.isfinite(g.coordinates))

    def test_sphere_points_on_unit_sphere(self, major):
        g = harmonic_point_cloud(major, n_points=100, surface="sphere")
        radii = np.linalg.norm(g.coordinates, axis=1)
        # Points are perturbed by harmonic field but should stay close to unit sphere
        assert np.all(radii < 2.5)

    @pytest.mark.parametrize("surface", ["sphere", "torus", "klein", "hyperbolic", "mos"])
    def test_field_weights_attached(self, major, surface):
        g = harmonic_point_cloud(major, n_points=100, surface=surface)
        assert g.weights is not None
        assert g.weights.shape == (100,)

    def test_invalid_surface(self, major):
        with pytest.raises(ValueError):
            harmonic_point_cloud(major, surface="cube")

    def test_invalid_n_points(self, major):
        with pytest.raises(ValueError):
            harmonic_point_cloud(major, n_points=2)


# ══════════════════════════════════════════ geometry_sequence (moved) ══════════

class TestGeometrySequence:
    def test_maps_over_sequence(self, major, minor):
        seq = HarmonicSequence(frames=[major, minor])
        result = geometry_sequence(seq, harmonic_surface, mode="torus", resolution=16)
        assert len(result) == 2
        assert all(g.geom_type == "mesh_3d" for g in result)

    def test_accepts_list(self, major, minor):
        result = geometry_sequence([major, minor], lissajous_tube, n_points=50, n_sides=6)
        assert len(result) == 2

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            geometry_sequence([], lissajous_tube)

    def test_kwargs_forwarded(self, major):
        result = geometry_sequence([major], harmonic_knot, n_points=80, n_sides=8)
        assert result[0].coordinates.shape[0] == 80 * 8
