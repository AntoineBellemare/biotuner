"""Tests for biotuner.harmonic_geometry.media.wave_field.acoustic."""

from fractions import Fraction

import numpy as np
import pytest

from biotuner.harmonic_geometry import Acoustic, HarmonicInput


@pytest.fixture
def major():
    return HarmonicInput(
        ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)],
        amplitudes=[1.0, 0.7, 0.8],
    )


class TestBasic:
    def test_family(self):
        assert Acoustic.family == "wave_field"

    def test_pressure_output(self, major):
        out = Acoustic(resolution=64).respond(major)
        assert out.geom_type == "field_2d"
        assert out.coordinates.shape == (64, 64)
        # Pressure is signed.
        assert out.coordinates.min() < 0
        assert out.coordinates.max() > 0

    def test_intensity_is_positive(self, major):
        out = Acoustic(resolution=64, output_mode="intensity").respond(major)
        assert (out.coordinates >= 0).all()

    def test_schlieren_in_unit_interval(self, major):
        out = Acoustic(resolution=64, output_mode="schlieren").respond(major)
        assert 0.0 <= out.coordinates.min() <= 1.0
        assert out.coordinates.max() == pytest.approx(1.0, abs=1e-9)

    def test_phase_in_pi(self, major):
        out = Acoustic(resolution=64, output_mode="phase").respond(major)
        assert out.coordinates.min() >= -np.pi - 1e-9
        assert out.coordinates.max() <= np.pi + 1e-9

    def test_source_positions_recorded(self, major):
        out = Acoustic(n_sources=4, resolution=64).respond(major)
        sources = out.metadata["source_positions"]
        assert sources.shape == (4, 2)


class TestLayouts:
    def test_ring_layout(self, major):
        out = Acoustic(n_sources=5, source_layout="ring",
                       source_radius=0.3, resolution=64).respond(major)
        sources = out.metadata["source_positions"]
        # All on a circle of radius 0.3 * extent = 0.3.
        radii = np.hypot(sources[:, 0], sources[:, 1])
        assert np.allclose(radii, 0.3, atol=1e-9)

    def test_chord_angles_layout(self, major):
        out = Acoustic(source_layout="chord_angles",
                       source_radius=0.25, resolution=64).respond(major)
        sources = out.metadata["source_positions"]
        # One source per chord ratio.
        assert sources.shape == (3, 2)

    def test_custom_layout_requires_positions(self):
        with pytest.raises(ValueError):
            Acoustic(source_layout="custom")

    def test_custom_layout(self, major):
        pts = [[0.0, 0.0], [0.2, 0.0], [-0.2, 0.0]]
        out = Acoustic(source_layout="custom", source_positions=pts,
                       resolution=64).respond(major)
        sources = out.metadata["source_positions"]
        assert np.allclose(sources, pts)


class TestPhysics:
    def test_higher_attenuation_reduces_far_field(self, major):
        lo = Acoustic(attenuation=0.0, resolution=64,
                      output_mode="intensity").respond(major)
        hi = Acoustic(attenuation=5.0, resolution=64,
                      output_mode="intensity").respond(major)
        # Average intensity drops with attenuation.
        assert hi.coordinates.mean() < lo.coordinates.mean()


class TestValidation:
    def test_rejects_unknown_output_mode(self):
        with pytest.raises(ValueError):
            Acoustic(output_mode="bogus")

    def test_rejects_unknown_layout(self):
        with pytest.raises(ValueError):
            Acoustic(source_layout="bogus")

    def test_rejects_bad_assignment(self):
        with pytest.raises(ValueError):
            Acoustic(source_assignment="bogus")

    def test_rejects_zero_sources(self):
        with pytest.raises(ValueError):
            Acoustic(n_sources=0)

    def test_rejects_nonpositive_wave_speed(self):
        with pytest.raises(ValueError):
            Acoustic(wave_speed=0)

    def test_rejects_unexpected_override(self, major):
        with pytest.raises(TypeError):
            Acoustic().respond(major, bogus=5)
