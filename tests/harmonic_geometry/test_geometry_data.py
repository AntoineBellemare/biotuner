"""Tests for biotuner.harmonic_geometry.geometry_data."""

from fractions import Fraction

import numpy as np
import pytest

from biotuner.harmonic_geometry.geometry_data import GEOM_TYPES, GeometryData


class TestGeometryDataConstruction:
    def test_curve_2d(self):
        coords = np.zeros((10, 2))
        g = GeometryData(geom_type="curve_2d", coordinates=coords)
        assert g.geom_type == "curve_2d"
        assert g.edges is None
        assert g.faces is None
        assert g.parameters == {}
        assert g.metadata == {}

    def test_unknown_geom_type_raises(self):
        with pytest.raises(ValueError):
            GeometryData(geom_type="not_a_real_type", coordinates=np.zeros((5, 2)))

    def test_all_documented_types_accepted(self):
        for gt in GEOM_TYPES:
            # Construction with a placeholder array; we are only checking
            # that the discriminator is recognized.
            GeometryData(geom_type=gt, coordinates=np.zeros((1, 2)))

    def test_curve_set_2d_coordinates_can_be_list(self):
        curves = [np.zeros((10, 2)), np.zeros((20, 2))]
        g = GeometryData(geom_type="curve_set_2d", coordinates=curves)
        assert isinstance(g.coordinates, list)
        assert len(g.coordinates) == 2

    def test_mesh_3d_with_faces(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        g = GeometryData(geom_type="mesh_3d", coordinates=verts, faces=faces)
        assert g.faces is not None
        assert g.faces.shape == (4, 3)


class TestGeometryDataSaveLoad:
    def test_curve_2d_roundtrip(self, tmp_path):
        rng = np.random.default_rng(42)
        coords = rng.standard_normal((50, 2))
        g = GeometryData(
            geom_type="curve_2d",
            coordinates=coords,
            parameters={"ratio": Fraction(3, 2), "n_points": 50},
            metadata={"closed": True, "lobes": (3, 2)},
        )
        path = tmp_path / "geom.npz"
        g.save(str(path))
        loaded = GeometryData.load(str(path))
        assert loaded.geom_type == "curve_2d"
        np.testing.assert_array_equal(loaded.coordinates, coords)
        assert loaded.parameters["ratio"] == Fraction(3, 2)
        assert loaded.parameters["n_points"] == 50
        assert loaded.metadata["closed"] is True
        assert loaded.metadata["lobes"] == (3, 2)

    def test_curve_set_roundtrip_preserves_per_curve_shapes(self, tmp_path):
        curves = [
            np.zeros((10, 2)),
            np.ones((25, 2)),
            np.full((7, 2), 3.0),
        ]
        g = GeometryData(geom_type="curve_set_2d", coordinates=curves)
        path = tmp_path / "set.npz"
        g.save(str(path))
        loaded = GeometryData.load(str(path))
        assert isinstance(loaded.coordinates, list)
        assert len(loaded.coordinates) == 3
        for original, restored in zip(curves, loaded.coordinates):
            np.testing.assert_array_equal(original, restored)

    def test_graph_roundtrip_preserves_edges_and_weights(self, tmp_path):
        coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])
        weights = np.array([0.5, 1.0, 0.7, 0.3])
        g = GeometryData(
            geom_type="graph",
            coordinates=coords,
            edges=edges,
            weights=weights,
        )
        path = tmp_path / "graph.npz"
        g.save(str(path))
        loaded = GeometryData.load(str(path))
        np.testing.assert_array_equal(loaded.edges, edges)
        np.testing.assert_array_equal(loaded.weights, weights)

    def test_field_2d_roundtrip_preserves_field_grid(self, tmp_path):
        x = np.linspace(0, 1, 16)
        y = np.linspace(0, 1, 16)
        X, Y = np.meshgrid(x, y, indexing="xy")
        field = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        g = GeometryData(
            geom_type="field_2d",
            coordinates=field,
            field_grid=(X, Y),
        )
        path = tmp_path / "field.npz"
        g.save(str(path))
        loaded = GeometryData.load(str(path))
        assert loaded.field_grid is not None
        assert len(loaded.field_grid) == 2
        np.testing.assert_array_equal(loaded.field_grid[0], X)
        np.testing.assert_array_equal(loaded.field_grid[1], Y)
        np.testing.assert_array_equal(loaded.coordinates, field)

    def test_mesh_3d_roundtrip(self, tmp_path):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        g = GeometryData(geom_type="mesh_3d", coordinates=verts, faces=faces)
        path = tmp_path / "mesh.npz"
        g.save(str(path))
        loaded = GeometryData.load(str(path))
        np.testing.assert_array_equal(loaded.coordinates, verts)
        np.testing.assert_array_equal(loaded.faces, faces)
