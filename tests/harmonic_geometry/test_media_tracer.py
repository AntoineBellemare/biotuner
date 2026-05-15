"""Tests for biotuner.harmonic_geometry.media.transport.tracer."""

from fractions import Fraction

import numpy as np
import pytest

from biotuner.harmonic_geometry import (
    GeometryData,
    HarmonicInput,
    Rectangular,
    RigidPlate,
    Tracer,
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


# =============================================================== basics


class TestBasic:
    def test_family(self):
        assert Tracer.family == "transport"

    def test_default_source_is_interference(self):
        src = Tracer().default_source()
        assert src is not None
        assert src.family == "wave_field"

    def test_curl_flow_from_plate(self, plate):
        out = Tracer(flow_kind="curl").respond(plate)
        assert out.geom_type == "vector_field_2d"
        assert out.coordinates.shape == plate.coordinates.shape + (2,)

    def test_gradient_flow_from_plate(self, plate):
        out = Tracer(flow_kind="gradient").respond(plate)
        assert out.geom_type == "vector_field_2d"

    def test_speed_output(self, plate):
        out = Tracer(output_mode="speed").respond(plate)
        assert out.geom_type == "field_2d"
        finite = np.isfinite(out.coordinates)
        assert (out.coordinates[finite] >= 0).all()

    def test_tracer_density_output(self, plate):
        out = Tracer(output_mode="tracer_density").respond(plate)
        assert out.geom_type == "field_2d"
        finite = np.isfinite(out.coordinates)
        # Density normalized to [0, 1].
        assert 0.0 <= out.coordinates[finite].min() <= 1.0
        assert out.coordinates[finite].max() == pytest.approx(1.0, abs=1e-9)

    def test_auto_wraps_harmonic_input(self, major):
        # Without a pre-computed field, default_source kicks in.
        out = Tracer().respond(major)
        assert out.geom_type == "vector_field_2d"


# =============================================================== physics


class TestPhysics:
    def test_curl_flow_is_perpendicular_to_gradient(self, plate):
        g_out = Tracer(flow_kind="gradient", normalize=False).respond(plate)
        c_out = Tracer(flow_kind="curl", normalize=False).respond(plate)
        u_g = g_out.coordinates[..., 0]
        v_g = g_out.coordinates[..., 1]
        u_c = c_out.coordinates[..., 0]
        v_c = c_out.coordinates[..., 1]
        # Curl is a 90° rotation of gradient: u_curl ≈ -v_grad, v_curl ≈ u_grad.
        finite = np.isfinite(u_g) & np.isfinite(u_c)
        assert np.allclose(u_c[finite], -v_g[finite], atol=1e-10)
        assert np.allclose(v_c[finite], u_g[finite], atol=1e-10)

    def test_mixed_at_extremes_matches_pure(self, plate):
        c_out = Tracer(flow_kind="curl", normalize=False).respond(plate)
        g_out = Tracer(flow_kind="gradient", normalize=False).respond(plate)
        m0 = Tracer(flow_kind="mixed", mixing=0.0, normalize=False).respond(plate)
        m1 = Tracer(flow_kind="mixed", mixing=1.0, normalize=False).respond(plate)
        # mixing=0 → pure curl; mixing=1 → pure gradient.
        finite = np.isfinite(c_out.coordinates[..., 0])
        assert np.allclose(m0.coordinates[finite],
                            c_out.coordinates[finite], atol=1e-10)
        assert np.allclose(m1.coordinates[finite],
                            g_out.coordinates[finite], atol=1e-10)


# =============================================================== validation


class TestValidation:
    def test_rejects_unknown_flow_kind(self):
        with pytest.raises(ValueError):
            Tracer(flow_kind="bogus")

    def test_rejects_unknown_output_mode(self):
        with pytest.raises(ValueError):
            Tracer(output_mode="bogus")

    def test_rejects_mixing_outside_unit(self):
        with pytest.raises(ValueError):
            Tracer(mixing=-0.1)
        with pytest.raises(ValueError):
            Tracer(mixing=1.5)

    def test_rejects_nonpositive_epsilon(self):
        with pytest.raises(ValueError):
            Tracer(epsilon=0)

    def test_rejects_non_field_2d_input(self, plate, major):
        bad = GeometryData(geom_type="point_cloud_2d",
                           coordinates=np.zeros((10, 2)))
        with pytest.raises(TypeError):
            Tracer().respond(bad)

    def test_rejects_unexpected_override(self, plate):
        with pytest.raises(TypeError):
            Tracer().respond(plate, bogus=5)
