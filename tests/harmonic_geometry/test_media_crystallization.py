"""Tests for biotuner.harmonic_geometry.media.morphogenetic.crystallization."""

from fractions import Fraction

import numpy as np
import pytest

from biotuner.harmonic_geometry import HarmonicInput
from biotuner.harmonic_geometry.media import Crystallization
from biotuner.harmonic_geometry.media.morphogenetic.crystallization import (
    _hex_mask,
    _hex_neighbor_or,
    _hex_neighbor_sum,
)


@pytest.fixture
def major():
    return HarmonicInput(
        ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)],
        amplitudes=[1.0, 0.7, 0.8],
    )


# ============================================================ hex helpers


class TestHexHelpers:
    def test_hex_mask_center_is_inside(self):
        mask = _hex_mask(5)
        assert mask.shape == (11, 11)
        assert mask[5, 5]

    def test_hex_mask_corners_outside(self):
        # Square-corner cells (Q=R, RR=R) lie outside the hex because
        # max(|Q|, |RR|, |Q+RR|) = 2R > R.
        mask = _hex_mask(5)
        assert not mask[0, 0]   # Q=-5, R=-5 → max=10
        assert not mask[10, 10]  # Q=+5, R=+5 → max=10

    def test_hex_mask_count(self):
        # Hexagonal grid of radius R has 3R² + 3R + 1 cells.
        for R in (3, 5, 8):
            mask = _hex_mask(R)
            assert int(mask.sum()) == 3 * R ** 2 + 3 * R + 1

    def test_neighbor_sum_zero_padded(self):
        # Single-1 cell at the corner: only neighbors inside the array
        # contribute, no wraparound from the opposite edge.
        arr = np.zeros((5, 5))
        arr[0, 0] = 1.0
        s = _hex_neighbor_sum(arr)
        # The (0, 0) cell has only 3 of its 6 hex neighbors inside the array.
        assert s[1, 0] == 1.0
        assert s[0, 1] == 1.0
        # Cells on the opposite edge must be 0 (no wraparound).
        assert s[4, 4] == 0.0

    def test_neighbor_or(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        out = _hex_neighbor_or(mask)
        # The 6 hex neighbors of (2, 2) should now be True.
        assert int(out.sum()) == 6


# ============================================================== basics


class TestCrystallizationBasic:
    def test_family(self):
        assert Crystallization.family == "morphogenetic"

    def test_default_source_is_none(self):
        assert Crystallization().default_source() is None

    def test_water_output(self, major):
        out = Crystallization(grid_radius=30, n_steps=200,
                              output_resolution=96).respond(major)
        assert out.geom_type == "field_2d"
        assert out.coordinates.shape == (96, 96)
        # NaN cells outside the hex.
        assert np.isnan(out.coordinates).any()
        assert np.isfinite(out.coordinates).any()

    def test_frozen_output(self, major):
        out = Crystallization(grid_radius=30, n_steps=200,
                              output_resolution=96,
                              output_mode="frozen").respond(major)
        assert out.geom_type == "field_2d"
        finite = out.coordinates[np.isfinite(out.coordinates)]
        # Frozen mask is binary {0, 1}.
        assert set(np.unique(finite)) <= {0.0, 1.0}

    def test_growth_happens(self, major):
        # With reasonable params, the center cell starts frozen (1) and
        # growth adds more frozen cells.
        out = Crystallization(grid_radius=30, n_steps=400, humidity=2e-4,
                              diffusion=0.5, anisotropy_strength=0.0,
                              output_resolution=64).respond(major)
        assert out.metadata["frozen_cells"] > 1

    def test_callable_shorthand(self, major):
        out = Crystallization(grid_radius=20, n_steps=80,
                              output_resolution=64)(major)
        assert out.geom_type == "field_2d"

    def test_rejects_non_harmonic_input(self):
        with pytest.raises(TypeError):
            Crystallization().respond(np.zeros((4, 4)))


# ============================================================ morphology


class TestMorphology:
    def test_isotropic_growth_is_six_fold_symmetric(self, major):
        # Without anisotropy, growth from a centered seed on a hex grid
        # should be 6-fold symmetric. The frozen-mask field has NaNs outside
        # the hex; rotate by 60° (in the cartesian rasterization) and check
        # that the binarized frozen pattern is consistent under 60° rotation
        # within rasterization tolerance.
        out = Crystallization(
            grid_radius=30, n_steps=300, humidity=2e-4, diffusion=0.5,
            anisotropy_strength=0.0, output_resolution=120,
            output_mode="frozen",
        ).respond(major)
        from scipy.ndimage import rotate

        field = np.nan_to_num(out.coordinates, nan=0.0).astype(np.float32)
        rotated = rotate(field, angle=60.0, reshape=False, order=0, mode="constant", cval=0.0)
        # Compare frozen-pixel counts; nearest-neighbor rotation will not be
        # pixel-perfect on a rectangular grid but should agree to within
        # a small relative tolerance.
        a = float((field >= 0.5).sum())
        b = float((rotated >= 0.5).sum())
        assert abs(a - b) / max(a, 1.0) < 0.10

    def test_higher_humidity_grows_more(self, major):
        # Use low diffusion so humidity is the dominant growth driver
        # (with high diffusion, neighbor-inflow swamps the γ effect),
        # and a generous grid so neither run hits the boundary early-stop.
        a = Crystallization(grid_radius=50, n_steps=400, humidity=2e-3,
                            diffusion=0.05, anisotropy_strength=0.0,
                            output_resolution=64).respond(major)
        b = Crystallization(grid_radius=50, n_steps=400, humidity=8e-3,
                            diffusion=0.05, anisotropy_strength=0.0,
                            output_resolution=64).respond(major)
        assert b.metadata["frozen_cells"] > a.metadata["frozen_cells"]

    def test_initial_water_bounds(self, major):
        with pytest.raises(ValueError):
            Crystallization(initial_water=0.0)
        with pytest.raises(ValueError):
            Crystallization(initial_water=1.0)
        # Just inside bounds should construct fine.
        Crystallization(initial_water=0.01)
        Crystallization(initial_water=0.99)


# =========================================================== chord coupling


class TestChordCoupling:
    def test_default_humidity_derived_from_chord(self, major):
        # Without an explicit humidity, Crystallization should use a value
        # derived from coupling.spectral_spread.
        out = Crystallization(grid_radius=15, n_steps=50,
                              output_resolution=32).respond(major)
        # Value is in the documented [1e-3, 8e-3] range.
        h = out.parameters["humidity"]
        assert 1.0e-3 <= h <= 8.0e-3 + 1e-9

    def test_default_diffusion_derived_from_chord(self, major):
        out = Crystallization(grid_radius=15, n_steps=50,
                              output_resolution=32).respond(major)
        d = out.parameters["diffusion"]
        assert 0.15 <= d <= 0.85 + 1e-9

    def test_default_sectors_derived_from_chord(self, major):
        out = Crystallization(grid_radius=15, n_steps=50,
                              output_resolution=32).respond(major)
        s = out.parameters["anisotropy_sectors"]
        assert s in {3, 4, 5, 6, 8, 12}

    def test_default_noise_derived_from_chord(self, major):
        out = Crystallization(grid_radius=15, n_steps=50,
                              output_resolution=32).respond(major)
        n = out.parameters["noise_temperature"]
        assert 0.0 <= n <= 3.0e-4 + 1e-9

    def test_explicit_humidity_overrides_default(self, major):
        out = Crystallization(grid_radius=15, n_steps=50,
                              humidity=1.5e-4,
                              output_resolution=32).respond(major)
        assert out.parameters["humidity"] == pytest.approx(1.5e-4)

    def test_anisotropy_breaks_symmetry_at_high_strength(self, major):
        # Compare frozen pixel counts in two 60° wedges around the seed.
        # With strong anisotropy, growth should be uneven across wedges.
        out_iso = Crystallization(
            grid_radius=25, n_steps=300, humidity=2e-4, diffusion=0.5,
            anisotropy_strength=0.0, output_resolution=96,
            output_mode="frozen",
        ).respond(major)
        out_aniso = Crystallization(
            grid_radius=25, n_steps=300, humidity=2e-4, diffusion=0.5,
            anisotropy_strength=8.0, output_resolution=96,
            output_mode="frozen",
        ).respond(major)
        # Anisotropy still keeps 6-fold replication, so this is a soft check:
        # the frozen-cell count or distribution should differ.
        assert out_iso.metadata["frozen_cells"] != out_aniso.metadata[
            "frozen_cells"]


# ============================================================== boundary


class TestBoundaryOutput:
    def test_boundary_returns_curve_set(self, major):
        # Boundary extraction uses skimage.measure.find_contours; skip on
        # environments where scikit-image is not installed (it's an optional
        # dependency surfaced only by this output_mode).
        pytest.importorskip("skimage")
        out = Crystallization(grid_radius=25, n_steps=200,
                              humidity=2e-4, output_resolution=96,
                              output_mode="boundary").respond(major)
        assert out.geom_type == "curve_set_2d"
        assert isinstance(out.coordinates, list)
        assert len(out.coordinates) >= 1
        for curve in out.coordinates:
            assert curve.ndim == 2
            assert curve.shape[1] == 2


# ============================================================ validation


class TestValidation:
    def test_rejects_bad_output_mode(self):
        with pytest.raises(ValueError):
            Crystallization(output_mode="bogus")

    def test_rejects_negative_humidity(self):
        with pytest.raises(ValueError):
            Crystallization(humidity=-1e-4)

    def test_rejects_diffusion_outside_unit_interval(self):
        with pytest.raises(ValueError):
            Crystallization(diffusion=1.5)
        with pytest.raises(ValueError):
            Crystallization(diffusion=-0.1)

    def test_rejects_zero_steps(self):
        with pytest.raises(ValueError):
            Crystallization(n_steps=0)

    def test_rejects_small_grid(self):
        with pytest.raises(ValueError):
            Crystallization(grid_radius=2)

    def test_rejects_low_output_resolution(self):
        with pytest.raises(ValueError):
            Crystallization(output_resolution=8)

    def test_rejects_negative_anisotropy(self):
        with pytest.raises(ValueError):
            Crystallization(anisotropy_strength=-0.1)

    def test_rejects_unexpected_override(self, major):
        with pytest.raises(TypeError, match="Unexpected override"):
            Crystallization(grid_radius=20, n_steps=50,
                            output_resolution=32).respond(major, bogus=5)
