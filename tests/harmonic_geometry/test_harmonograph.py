"""Tests for biotuner.harmonic_geometry.harmonograph."""

import numpy as np
import pytest

from biotuner.harmonic_geometry import HarmonicInput
from biotuner.harmonic_geometry.harmonograph import (
    DEFAULT_DAMPING,
    derive_damping_from_linewidth,
    harmonograph_3d,
    harmonograph_from_peaks,
    harmonograph_lateral,
    harmonograph_rotary,
)


class TestHarmonographLateral:
    def test_shape(self):
        inp = HarmonicInput(peaks=[1.0, 1.5, 2.0, 3.0])
        g = harmonograph_lateral(inp, duration=5.0, sr=200)
        assert g.geom_type == "curve_2d"
        assert g.coordinates.shape == (1000, 2)

    def test_zero_damping_is_bounded(self):
        inp = HarmonicInput(
            peaks=[1.0, 2.0, 3.0, 5.0],
            damping=[0.0, 0.0, 0.0, 0.0],
        )
        g = harmonograph_lateral(inp, duration=20.0, sr=100)
        # All amplitudes are non-zero and bounded by sum of normalized amps (= 1).
        assert np.all(np.isfinite(g.coordinates))
        assert np.max(np.abs(g.coordinates)) <= 1.0 + 1e-9

    def test_positive_damping_decays(self):
        inp = HarmonicInput(
            peaks=[2.0, 3.0, 5.0],
            damping=[0.2, 0.2, 0.2],
        )
        g = harmonograph_lateral(inp, duration=15.0, sr=200)
        n = g.coordinates.shape[0]
        early = np.max(np.abs(g.coordinates[: n // 5]))
        late = np.max(np.abs(g.coordinates[-n // 5 :]))
        # Late envelope must be much smaller than the early envelope.
        assert late < 0.2 * early

    def test_default_damping_flag(self):
        inp = HarmonicInput(peaks=[2.0, 3.0])
        g = harmonograph_lateral(inp, duration=5.0, sr=100)
        assert g.metadata["damping_default_used"] is True

    def test_explicit_xy_components(self):
        inp = HarmonicInput(peaks=[2.0, 3.0, 4.0, 5.0])
        g = harmonograph_lateral(
            inp,
            duration=2.0,
            sr=100,
            x_components=[0, 2],
            y_components=[1, 3],
        )
        assert g.parameters["x_components"] == [0, 2]
        assert g.parameters["y_components"] == [1, 3]

    def test_invalid_duration(self):
        inp = HarmonicInput(peaks=[1.0])
        with pytest.raises(ValueError):
            harmonograph_lateral(inp, duration=0)


class TestHarmonographRotary:
    def test_rotary_differs_from_lateral(self):
        inp = HarmonicInput(peaks=[2.0, 3.0, 5.0])
        lat = harmonograph_lateral(inp, duration=4.0, sr=200)
        rot = harmonograph_rotary(inp, duration=4.0, sr=200, rotation_freq=0.5)
        assert not np.allclose(lat.coordinates, rot.coordinates)
        assert rot.parameters["rotation_freq"] == 0.5
        assert rot.metadata["kind"] == "harmonograph_rotary"

    def test_zero_rotation_equals_lateral(self):
        inp = HarmonicInput(peaks=[2.0, 3.0, 5.0])
        lat = harmonograph_lateral(inp, duration=2.0, sr=100)
        rot = harmonograph_rotary(inp, duration=2.0, sr=100, rotation_freq=0.0)
        np.testing.assert_allclose(lat.coordinates, rot.coordinates)


class TestHarmonograph3D:
    def test_shape_and_type(self):
        inp = HarmonicInput(peaks=[2.0, 3.0, 5.0, 7.0, 11.0])
        g = harmonograph_3d(inp, duration=4.0, sr=200)
        assert g.geom_type == "curve_3d"
        assert g.coordinates.shape == (800, 3)

    def test_invalid_axis_assignment(self):
        inp = HarmonicInput(peaks=[1.0, 2.0])
        with pytest.raises(ValueError):
            harmonograph_3d(inp, axis_assignment="diagonal")


class TestHarmonographFromPeaks:
    def test_default_damping_applied(self):
        g = harmonograph_from_peaks([2.0, 3.0, 5.0], duration=3.0, sr=100)
        # damping wasn't None at the HarmonicInput level (we filled it),
        # so the metadata flag should be False.
        assert g.metadata["damping_default_used"] is False

    def test_custom_phase_passes_through(self):
        g = harmonograph_from_peaks(
            [2.0, 3.0],
            phases=[0.0, np.pi / 3],
            duration=2.0,
            sr=50,
        )
        assert g.coordinates.shape == (100, 2)


class TestDeriveDamping:
    def test_lorentzian_formula(self):
        out = derive_damping_from_linewidth([1.0, 2.0, 4.0])
        np.testing.assert_allclose(out, [np.pi, 2 * np.pi, 4 * np.pi])

    def test_zero_or_negative_falls_back(self):
        out = derive_damping_from_linewidth([0.0, -1.0, np.nan, 1.0])
        np.testing.assert_allclose(
            out,
            [DEFAULT_DAMPING, DEFAULT_DAMPING, DEFAULT_DAMPING, np.pi],
        )
