"""Tests for biotuner.harmonic_geometry.media.transport.streaming."""

from fractions import Fraction

import numpy as np
import pytest

from biotuner.harmonic_geometry import (
    GeometryData,
    HarmonicInput,
    Rectangular,
    RigidPlate,
    Streaming,
)


@pytest.fixture
def major():
    return HarmonicInput(
        ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)],
        amplitudes=[1.0, 0.7, 0.8],
    )


@pytest.fixture
def plate(major):
    return RigidPlate(domain=Rectangular(1, 1), resolution=96).respond(major)


class TestBasic:
    def test_family(self):
        assert Streaming.family == "transport"

    def test_default_source_is_eigenmode(self):
        src = Streaming().default_source()
        assert src.family == "eigenmode"

    def test_flow_output(self, plate):
        out = Streaming(viscosity=0.5).respond(plate)
        assert out.geom_type == "vector_field_2d"
        assert out.coordinates.shape == plate.coordinates.shape + (2,)

    def test_speed_output(self, plate):
        out = Streaming(output_mode="speed").respond(plate)
        assert out.geom_type == "field_2d"
        finite = np.isfinite(out.coordinates)
        assert (out.coordinates[finite] >= 0).all()

    def test_tracer_density_output(self, plate):
        out = Streaming(output_mode="tracer_density").respond(plate)
        assert out.geom_type == "field_2d"
        finite = np.isfinite(out.coordinates)
        assert out.coordinates[finite].max() == pytest.approx(1.0, abs=1e-9)

    def test_auto_wraps_harmonic_input(self, major):
        out = Streaming().respond(major)
        assert out.geom_type == "vector_field_2d"


class TestPhysics:
    def test_lower_viscosity_gives_stronger_flow_unnormalized(self, plate):
        # With normalize=False, viscosity directly scales magnitude.
        out_hi = Streaming(viscosity=2.0, normalize=False).respond(plate)
        out_lo = Streaming(viscosity=0.5, normalize=False).respond(plate)
        u_hi = out_hi.coordinates[..., 0]
        u_lo = out_lo.coordinates[..., 0]
        finite = np.isfinite(u_hi) & np.isfinite(u_lo)
        # |u_lo| should exceed |u_hi| because viscosity divides magnitude.
        assert float(np.abs(u_lo[finite]).mean()) > \
               float(np.abs(u_hi[finite]).mean())

    def test_streamfunction_is_divergence_free(self, plate):
        # Flow derived from a streamfunction should have ∇·u ≈ 0.
        out = Streaming(viscosity=1.0, normalize=False).respond(plate)
        u = np.nan_to_num(out.coordinates[..., 0])
        v = np.nan_to_num(out.coordinates[..., 1])
        dvdy, dudx = np.gradient(u, axis=0), np.gradient(v, axis=1)
        # Use both partial derivatives in the divergence: du/dx + dv/dy.
        du_dx = np.gradient(u, axis=1)
        dv_dy = np.gradient(v, axis=0)
        div = du_dx + dv_dy
        # Pick the interior, away from the boundary where FFT periodicity
        # and finite-difference edge effects are strongest.
        H, W = u.shape
        interior = div[H // 4: 3 * H // 4, W // 4: 3 * W // 4]
        # Tolerance is generous: FFT Poisson + numpy.gradient round-trip
        # accumulates discretization error.
        mag = float(np.abs(u).mean()) + 1e-9
        assert float(np.abs(interior).mean()) / mag < 0.20


class TestValidation:
    def test_rejects_unknown_output_mode(self):
        with pytest.raises(ValueError):
            Streaming(output_mode="bogus")

    def test_rejects_nonpositive_viscosity(self):
        with pytest.raises(ValueError):
            Streaming(viscosity=0)
        with pytest.raises(ValueError):
            Streaming(viscosity=-1)

    def test_rejects_non_field_2d_input(self):
        bad = GeometryData(geom_type="point_cloud_2d",
                           coordinates=np.zeros((10, 2)))
        with pytest.raises(TypeError):
            Streaming().respond(bad)

    def test_rejects_unexpected_override(self, plate):
        with pytest.raises(TypeError):
            Streaming().respond(plate, bogus=5)
