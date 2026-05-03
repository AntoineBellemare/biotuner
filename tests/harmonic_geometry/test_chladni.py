"""Tests for biotuner.harmonic_geometry.chladni."""

from fractions import Fraction

import numpy as np
import pytest

from biotuner.harmonic_geometry import HarmonicInput
from biotuner.harmonic_geometry.chladni import (
    chladni_field_3d_box,
    chladni_field_circular,
    chladni_field_polygon,
    chladni_field_rectangular,
    chladni_from_input,
    chladni_nodal_lines,
    chladni_nodal_surfaces,
    chladni_temporal,
    ratios_to_modes,
)

_HAS_SKIMAGE = True
try:
    import skimage  # noqa: F401
except ImportError:
    _HAS_SKIMAGE = False


# ============================================================ ratios_to_modes


class TestRatiosToModes:
    def test_stern_brocot_simple(self):
        modes = ratios_to_modes([Fraction(3, 2), Fraction(5, 4), 2], strategy="stern_brocot")
        assert modes == [(3, 2), (5, 4), (2, 1)]

    def test_stern_brocot_irrational(self):
        # phi ≈ 1.618; with max_mode=20, the best approximation is 13/8 (= 1.625) or close.
        modes = ratios_to_modes([1.618033988749895], strategy="stern_brocot", max_mode=20)
        assert len(modes) == 1
        m, n = modes[0]
        assert 1 <= m <= 20 and 1 <= n <= 20
        assert abs(m / n - 1.618033988749895) < 0.05

    def test_continued_fraction(self):
        modes = ratios_to_modes([Fraction(7, 4)], strategy="continued_fraction")
        assert modes == [(7, 4)]

    def test_rounded(self):
        modes = ratios_to_modes([1.4, 2.7, 4.49], strategy="rounded")
        assert modes == [(1, 1), (3, 1), (4, 1)]

    def test_best_simple(self):
        modes = ratios_to_modes([1.5], strategy="best_simple", max_mode=10)
        # Many pairs hit 1.5 exactly (3/2, 6/4, 9/6); the loop returns the first.
        m, n = modes[0]
        assert m / n == 1.5

    def test_unknown_strategy(self):
        with pytest.raises(ValueError):
            ratios_to_modes([1.5], strategy="bogus")

    def test_zero_ratio_rejected(self):
        with pytest.raises(ValueError):
            ratios_to_modes([0.0])

    def test_invalid_max_mode(self):
        with pytest.raises(ValueError):
            ratios_to_modes([1.0], max_mode=0)


# ============================================================== rectangular


class TestRectangular:
    def test_field_shape(self):
        g = chladni_field_rectangular([(2, 3), (1, 4)], resolution=64)
        assert g.geom_type == "field_2d"
        assert g.coordinates.shape == (64, 64)
        assert g.field_grid is not None
        assert g.field_grid[0].shape == (64, 64)
        assert g.field_grid[1].shape == (64, 64)

    def test_pure_mode_symmetry(self):
        """Pure (m, n) cos·cos mode has m+1 nodal columns and n+1 nodal rows.

        For free-edge cosines on a unit square, the field
        ``cos(mπx) cos(nπy)`` vanishes when ``mπx = π/2 + kπ`` (m
        positions in [0, 1]) or the analogous condition on y. The mid-row
        and mid-column should each contain exactly the right number of
        sign changes.
        """
        m, n = 3, 2
        g = chladni_field_rectangular([(m, n)], resolution=257)  # odd to land on midpoints
        field = g.coordinates
        mid_row = field[field.shape[0] // 2, :]
        mid_col = field[:, field.shape[1] // 2]
        # cos(mπx) over [0, 1] has m sign changes (zeros).
        sign_changes_x = int(np.sum(np.diff(np.sign(mid_col)) != 0))
        sign_changes_y = int(np.sum(np.diff(np.sign(mid_row)) != 0))
        # Allow off-by-one due to boundary sampling.
        assert abs(sign_changes_x - m) <= 1
        assert abs(sign_changes_y - n) <= 1

    def test_amps_normalize_when_default(self):
        g = chladni_field_rectangular([(2, 3), (1, 1)], resolution=32)
        # Default amps are 1 / n_modes.
        np.testing.assert_allclose(g.parameters["amps"], [0.5, 0.5])

    def test_invalid_amps_length(self):
        with pytest.raises(ValueError):
            chladni_field_rectangular([(1, 1), (2, 2)], amps=[1.0], resolution=32)

    def test_empty_modes(self):
        with pytest.raises(ValueError):
            chladni_field_rectangular([], resolution=32)

    def test_invalid_dimensions(self):
        with pytest.raises(ValueError):
            chladni_field_rectangular([(1, 1)], Lx=0)


# ================================================================== circular


class TestCircular:
    def test_field_shape(self):
        g = chladni_field_circular([1, 2], [0, 1], R=1.0, resolution=64)
        assert g.geom_type == "field_2d"
        assert g.coordinates.shape == (64, 64)

    def test_outside_disk_is_nan(self):
        g = chladni_field_circular([1], [0], R=1.0, resolution=65)
        # The corner cells should be outside the disk.
        assert np.isnan(g.coordinates[0, 0])
        # The center should be finite.
        assert np.isfinite(g.coordinates[32, 32])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            chladni_field_circular([1, 2], [0], resolution=32)

    def test_invalid_modes(self):
        with pytest.raises(ValueError):
            chladni_field_circular([0], [0], resolution=32)
        with pytest.raises(ValueError):
            chladni_field_circular([1], [-1], resolution=32)


# ==================================================================== 3D box


class TestBox3D:
    def test_field_shape(self):
        g = chladni_field_3d_box([(1, 1, 1), (2, 1, 1)], resolution=24)
        assert g.geom_type == "field_3d"
        assert g.coordinates.shape == (24, 24, 24)
        assert len(g.field_grid) == 3

    def test_invalid_mode_triple(self):
        with pytest.raises(ValueError):
            chladni_field_3d_box([(1, 0, 1)], resolution=16)

    def test_invalid_dimensions(self):
        with pytest.raises(ValueError):
            chladni_field_3d_box([(1, 1, 1)], dimensions=(1, 0, 1), resolution=16)


# =================================================================== polygon


class TestPolygon:
    def test_basic_field(self):
        g = chladni_field_polygon([0, 1], n_sides=4, resolution=32)
        assert g.geom_type == "field_2d"
        assert g.coordinates.shape == (32, 32)
        # Outside the polygon is NaN; inside is finite.
        assert np.isnan(g.coordinates).any()
        assert np.isfinite(g.coordinates).any()

    def test_invalid_n_sides(self):
        with pytest.raises(ValueError):
            chladni_field_polygon([0], n_sides=2, resolution=32)

    def test_fem_not_implemented(self):
        with pytest.raises(NotImplementedError):
            chladni_field_polygon([0], n_sides=3, solver="fem", resolution=32)

    def test_unknown_solver(self):
        with pytest.raises(ValueError):
            chladni_field_polygon([0], n_sides=3, solver="bogus", resolution=32)

    def test_eigenvalues_recorded(self):
        g = chladni_field_polygon([0, 2], n_sides=3, resolution=32)
        eigs = g.metadata["eigenvalues"]
        assert len(eigs) == 2
        # Lowest mode has the smallest eigenvalue.
        assert eigs[0] <= eigs[1]


# ============================================================ nodal extraction


@pytest.mark.skipif(not _HAS_SKIMAGE, reason="scikit-image not installed")
class TestNodalLines:
    def test_extracts_curve_set_2d(self):
        g = chladni_field_rectangular([(3, 2)], resolution=129)
        nodal = chladni_nodal_lines(g, threshold=0.0)
        assert nodal.geom_type == "curve_set_2d"
        assert isinstance(nodal.coordinates, list)
        assert len(nodal.coordinates) > 0
        for c in nodal.coordinates:
            assert c.ndim == 2 and c.shape[1] == 2

    def test_rejects_non_field_2d(self):
        from biotuner.harmonic_geometry.geometry_data import GeometryData

        g = GeometryData(geom_type="curve_2d", coordinates=np.zeros((10, 2)))
        with pytest.raises(ValueError):
            chladni_nodal_lines(g)


@pytest.mark.skipif(not _HAS_SKIMAGE, reason="scikit-image not installed")
class TestNodalSurfaces:
    def test_extracts_mesh_3d(self):
        g = chladni_field_3d_box([(2, 2, 2)], resolution=24)
        mesh = chladni_nodal_surfaces(g, threshold=0.0)
        assert mesh.geom_type == "mesh_3d"
        assert mesh.coordinates.shape[1] == 3
        assert mesh.faces is not None
        assert mesh.faces.shape[1] == 3

    def test_rejects_non_field_3d(self):
        from biotuner.harmonic_geometry.geometry_data import GeometryData

        g = GeometryData(geom_type="field_2d", coordinates=np.zeros((10, 10)))
        with pytest.raises(ValueError):
            chladni_nodal_surfaces(g)


class TestNodalLinesMissingDep:
    def test_clear_error_when_skimage_unavailable(self, monkeypatch):
        if _HAS_SKIMAGE:
            pytest.skip("scikit-image is installed; can't test the error path.")
        g = chladni_field_rectangular([(1, 1)], resolution=32)
        with pytest.raises(ImportError, match="scikit-image"):
            chladni_nodal_lines(g)


# ================================================================== adapters


class TestChladniFromInput:
    def test_rectangular(self, simple_ratios):
        inp = HarmonicInput(ratios=simple_ratios, base_freq=100.0)
        g = chladni_from_input(inp, plate="rectangular", plate_kwargs={"resolution": 64})
        assert g.geom_type == "field_2d"
        assert g.coordinates.shape == (64, 64)
        assert g.metadata["plate"] == "rectangular"

    def test_circular(self, simple_ratios):
        inp = HarmonicInput(ratios=simple_ratios, base_freq=100.0)
        g = chladni_from_input(inp, plate="circular", plate_kwargs={"resolution": 48})
        assert g.geom_type == "field_2d"
        assert g.metadata["plate"] == "circular"

    def test_polygon(self, simple_ratios):
        inp = HarmonicInput(ratios=simple_ratios, base_freq=100.0)
        g = chladni_from_input(
            inp,
            plate="polygon",
            plate_kwargs={"n_sides": 5, "resolution": 32},
        )
        assert g.geom_type == "field_2d"
        assert g.metadata["plate"] == "polygon"

    def test_box_3d(self, simple_ratios):
        inp = HarmonicInput(ratios=simple_ratios, base_freq=100.0)
        g = chladni_from_input(inp, plate="box_3d", plate_kwargs={"resolution": 16})
        assert g.geom_type == "field_3d"
        assert g.metadata["plate"] == "box_3d"

    def test_unknown_plate(self):
        inp = HarmonicInput(ratios=[1, 2])
        with pytest.raises(ValueError):
            chladni_from_input(inp, plate="hyperdrive")


class TestChladniTemporal:
    def test_two_times_differ(self):
        inp = HarmonicInput(peaks=[2.0, 3.0])
        g0 = chladni_temporal(inp, t=0.0, plate="rectangular", resolution=32)
        g1 = chladni_temporal(inp, t=0.25, plate="rectangular", resolution=32)
        # Frames should differ once we step in time (phase drift).
        assert not np.allclose(g0.coordinates, g1.coordinates)
