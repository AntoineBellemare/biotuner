"""Tests for biotuner.harmonic_geometry.media.transport.granular."""

from fractions import Fraction

import numpy as np
import pytest

from biotuner.harmonic_geometry import HarmonicInput
from biotuner.harmonic_geometry.media import (
    Circular,
    Granular,
    Pipeline,
    Rectangular,
    RigidPlate,
)


@pytest.fixture
def major():
    return HarmonicInput(
        ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)],
        amplitudes=[1.0, 0.7, 0.8],
    )


@pytest.fixture
def plate_field(major):
    return RigidPlate(domain=Rectangular(1, 1), resolution=64).respond(major)


# =============================================================== Granular


class TestGranularBasic:
    def test_family(self):
        assert Granular.family == "transport"

    def test_default_source_is_rigid_plate(self):
        src = Granular().default_source()
        assert isinstance(src, RigidPlate)

    def test_density_output(self, plate_field):
        out = Granular(affinity=1.0, temperature=0.1).respond(plate_field)
        assert out.geom_type == "field_2d"
        assert out.coordinates.shape == plate_field.coordinates.shape
        # Density sums to 1 on the valid region.
        assert np.isclose(np.nansum(out.coordinates), 1.0)

    def test_particle_output(self, plate_field):
        out = Granular(
            affinity=1.0,
            temperature=0.1,
            output_mode="particles",
            n_particles=500,
            seed=0,
        ).respond(plate_field)
        assert out.geom_type == "point_cloud_2d"
        assert out.coordinates.shape == (500, 2)
        # Particles must lie within the plate extent.
        xs, ys = out.coordinates[:, 0], out.coordinates[:, 1]
        assert xs.min() >= -0.05 and xs.max() <= 1.05  # small jitter slack
        assert ys.min() >= -0.05 and ys.max() <= 1.05


class TestGranularPhysics:
    """Sanity checks that the regime knobs do what the physics says."""

    def test_positive_affinity_concentrates_at_low_field(self, plate_field):
        # affinity > 0 → V = +|u|² → density peaks where |u| is low (nodes).
        density = Granular(affinity=2.0, temperature=0.01).respond(plate_field)
        u2 = plate_field.coordinates ** 2
        finite = np.isfinite(plate_field.coordinates)
        # Pearson correlation between density and |u|² should be NEGATIVE.
        d = density.coordinates[finite]
        e = u2[finite]
        r = np.corrcoef(d, e)[0, 1]
        assert r < -0.3, f"expected negative correlation, got {r}"

    def test_negative_affinity_concentrates_at_high_field(self, plate_field):
        # affinity < 0 → V = -|u|² → density peaks at antinodes.
        # Antinodes are points (vs. nodes which are lines), so the
        # resulting density is delta-spike-like; use a top-quantile
        # check rather than Pearson correlation (which is diluted by
        # the large bulk of near-zero cells).
        density = Granular(affinity=-2.0, temperature=0.01).respond(plate_field)
        u2 = plate_field.coordinates ** 2
        finite = np.isfinite(plate_field.coordinates)
        d = density.coordinates[finite]
        e = u2[finite]
        # Top 5% of density cells should be located at high-|u|² cells.
        thresh = np.quantile(d, 0.95)
        u2_top = e[d >= thresh]
        u2_bottom = e[d < np.quantile(d, 0.50)]
        assert u2_top.mean() > 3 * u2_bottom.mean(), (
            f"top-density mean |u|²={u2_top.mean():.3f} should dominate "
            f"bottom-half mean={u2_bottom.mean():.3f}"
        )

    def test_zero_affinity_is_uniform(self, plate_field):
        density = Granular(affinity=0.0, temperature=0.1).respond(plate_field)
        finite = np.isfinite(plate_field.coordinates)
        d = density.coordinates[finite]
        # Uniform → all equal.
        assert np.allclose(d, d[0], rtol=1e-9, atol=1e-12)

    def test_high_temperature_flattens(self, plate_field):
        cold = Granular(affinity=2.0, temperature=0.01).respond(plate_field)
        hot = Granular(affinity=2.0, temperature=10.0).respond(plate_field)
        finite = np.isfinite(plate_field.coordinates)
        # Variance(hot) << Variance(cold).
        assert np.nanvar(hot.coordinates[finite]) < np.nanvar(
            cold.coordinates[finite]
        )

    def test_energy_gradient_mode_runs(self, plate_field):
        out = Granular(
            affinity=1.0, temperature=0.1, field_kind="energy_gradient"
        ).respond(plate_field)
        assert out.geom_type == "field_2d"
        assert np.isclose(np.nansum(out.coordinates), 1.0)


class TestGranularComposition:
    def test_auto_wrap_from_harmonic_input(self, major):
        out = Granular(affinity=1.0, temperature=0.1).respond(major)
        assert out.geom_type == "field_2d"

    def test_pipeline_composes(self, major):
        pipe = Pipeline(
            RigidPlate(resolution=64), Granular(affinity=1.0, temperature=0.1)
        )
        out = pipe(major)
        assert out.geom_type == "field_2d"
        assert np.isclose(np.nansum(out.coordinates), 1.0)

    def test_circular_domain_preserves_nan_mask(self, major):
        plate = RigidPlate(domain=Circular(1.0), resolution=64).respond(major)
        density = Granular(affinity=1.0, temperature=0.1).respond(plate)
        # NaNs in the input field must remain NaN in the output density.
        assert np.array_equal(
            np.isnan(plate.coordinates), np.isnan(density.coordinates)
        )

    def test_particle_seeding_is_deterministic(self, plate_field):
        a = Granular(output_mode="particles", n_particles=100, seed=42).respond(plate_field)
        b = Granular(output_mode="particles", n_particles=100, seed=42).respond(plate_field)
        assert np.array_equal(a.coordinates, b.coordinates)

    def test_particle_seeding_different_seeds_differ(self, plate_field):
        a = Granular(output_mode="particles", n_particles=100, seed=1).respond(plate_field)
        b = Granular(output_mode="particles", n_particles=100, seed=2).respond(plate_field)
        assert not np.array_equal(a.coordinates, b.coordinates)


class TestGranularValidation:
    def test_rejects_non_field_2d(self, major):
        # Pass a HarmonicInput — auto-wrapped to field_2d; build a fake 3D.
        plate_3d = RigidPlate(
            domain=__import__("biotuner").harmonic_geometry.media.Box3D(1, 1, 1),
            resolution=16,
        ).respond(major)
        with pytest.raises(TypeError, match="2-D scalar wave field"):
            Granular().respond(plate_3d)

    def test_rejects_bad_field_kind(self):
        with pytest.raises(ValueError):
            Granular(field_kind="bogus")

    def test_rejects_bad_output_mode(self):
        with pytest.raises(ValueError):
            Granular(output_mode="bogus")

    def test_rejects_bad_temperature(self):
        with pytest.raises(ValueError):
            Granular(temperature=0.0)

    def test_rejects_zero_particles(self):
        with pytest.raises(ValueError):
            Granular(n_particles=0)

    def test_rejects_unexpected_override(self, plate_field):
        with pytest.raises(TypeError, match="Unexpected override"):
            Granular().respond(plate_field, bogus_kwarg=5)
