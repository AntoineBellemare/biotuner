"""Tests for biotuner.harmonic_geometry.media.eigenmode.elastic."""

from fractions import Fraction

import numpy as np
import pytest

from biotuner.harmonic_geometry import (
    Elastic,
    HarmonicInput,
    Rectangular,
)


@pytest.fixture
def major():
    return HarmonicInput(
        ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)],
        amplitudes=[1.0, 0.7, 0.8],
    )


class TestBasic:
    def test_family(self):
        assert Elastic.family == "eigenmode"

    def test_default_source_is_none(self):
        assert Elastic().default_source() is None

    def test_field_shape(self, major):
        out = Elastic(resolution=96).respond(major)
        assert out.geom_type == "field_2d"
        assert out.coordinates.shape == (96, 96)

    def test_chord_driven_ratio_in_range(self, major):
        out = Elastic(resolution=64).respond(major)
        r = out.parameters["anisotropy_ratio"]
        assert 1.0 <= r <= 4.0 + 1e-9

    def test_isotropic_reduces_to_rigid_plate_layout(self, major):
        out = Elastic(resolution=64, anisotropy_ratio=1.0,
                       anisotropy_axis=0.0).respond(major)
        # The field should be non-trivial.
        assert np.isfinite(out.coordinates).all()
        assert out.coordinates.std() > 1e-3


class TestValidation:
    def test_rejects_non_rectangular_domain(self):
        from biotuner.harmonic_geometry import Circular
        with pytest.raises(TypeError):
            Elastic(domain=Circular(R=1.0))

    def test_rejects_nonpositive_ratio(self):
        with pytest.raises(ValueError):
            Elastic(anisotropy_ratio=0)
        with pytest.raises(ValueError):
            Elastic(anisotropy_ratio=-1)

    def test_rejects_zero_modes(self):
        with pytest.raises(ValueError):
            Elastic(n_modes=0)

    def test_rejects_small_resolution(self):
        with pytest.raises(ValueError):
            Elastic(resolution=8)

    def test_rejects_unexpected_override(self, major):
        with pytest.raises(TypeError):
            Elastic(resolution=32).respond(major, bogus=5)
