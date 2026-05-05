"""Tests for biotuner.harmonic_geometry.transitions."""

from fractions import Fraction

import numpy as np
import pytest

from biotuner.harmonic_geometry import (
    HarmonicInput,
    blend_fields,
    extend_harmonics,
    fade_in_components,
    harmonic_interference_field_2d,
    interpolate_chords,
    quasicrystal_field_2d,
)


@pytest.fixture
def major_chord() -> HarmonicInput:
    return HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)])


@pytest.fixture
def minor_chord() -> HarmonicInput:
    return HarmonicInput(ratios=[Fraction(1), Fraction(6, 5), Fraction(3, 2)])


@pytest.fixture
def dom7_chord() -> HarmonicInput:
    return HarmonicInput(
        ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2), Fraction(7, 4)]
    )


# ============================================================ interpolate_chords


class TestInterpolateChords:
    def test_t_zero_recovers_a(self, major_chord, minor_chord):
        out = interpolate_chords(major_chord, minor_chord, 0.0)
        ratios = sorted([float(r) for r in out.to_ratios()])
        assert ratios == pytest.approx([1.0, 1.25, 1.5])

    def test_t_one_recovers_b(self, major_chord, minor_chord):
        out = interpolate_chords(major_chord, minor_chord, 1.0)
        ratios = sorted([float(r) for r in out.to_ratios()])
        assert ratios == pytest.approx([1.0, 1.2, 1.5])

    def test_midpoint_is_average(self, major_chord, minor_chord):
        out = interpolate_chords(major_chord, minor_chord, 0.5)
        ratios = sorted([float(r) for r in out.to_ratios()])
        # Major: 1, 1.25, 1.5 ; Minor: 1, 1.2, 1.5 → midpoint: 1, 1.225, 1.5
        assert ratios == pytest.approx([1.0, 1.225, 1.5])

    def test_clamps_t_outside_range(self, major_chord, minor_chord):
        out_neg = interpolate_chords(major_chord, minor_chord, -0.5)
        out_high = interpolate_chords(major_chord, minor_chord, 1.7)
        # Should clamp to a and b respectively.
        ra_neg = sorted([float(r) for r in out_neg.to_ratios()])
        ra_high = sorted([float(r) for r in out_high.to_ratios()])
        assert ra_neg == pytest.approx([1.0, 1.25, 1.5])
        assert ra_high == pytest.approx([1.0, 1.2, 1.5])

    def test_unequal_chord_sizes_extras_fade(self, major_chord, dom7_chord):
        # Major has 3 components, Dom7 has 4. At t=0 the extra (7/4)
        # should have amplitude 0; at t=1 it should have full amplitude.
        out0 = interpolate_chords(major_chord, dom7_chord, 0.0)
        out1 = interpolate_chords(major_chord, dom7_chord, 1.0)
        # At t=0: total amplitude should match major's normalized sum.
        amps0 = list(out0.amplitudes or [])
        # At t=0 the extra (7/4) component should be at amplitude 0.
        # Find the component with ratio closest to 7/4 = 1.75.
        ratios0 = [float(r) for r in out0.to_ratios()]
        idx = int(np.argmin(np.abs(np.array(ratios0) - 1.75)))
        assert amps0[idx] == pytest.approx(0.0, abs=1e-12)
        # At t=1, that same ratio should have nonzero amplitude.
        amps1 = list(out1.amplitudes or [])
        ratios1 = [float(r) for r in out1.to_ratios()]
        idx1 = int(np.argmin(np.abs(np.array(ratios1) - 1.75)))
        assert amps1[idx1] > 0.0

    def test_metadata_records_transition(self, major_chord, minor_chord):
        out = interpolate_chords(major_chord, minor_chord, 0.42)
        meta = out.metadata.get("transition", {})
        assert meta.get("kind") == "interpolate_chords"
        assert meta.get("t") == pytest.approx(0.42)

    def test_smooth_animation_no_jumps(self, major_chord, minor_chord):
        """Sample 21 frames at t = 0, 0.05, …, 1.0 and confirm the
        ratios change monotonically and continuously."""
        ts = np.linspace(0.0, 1.0, 21)
        # The middle ratio (5/4 → 6/5) should monotonically decrease.
        mid_ratios = []
        for t in ts:
            out = interpolate_chords(major_chord, minor_chord, float(t))
            ratios = sorted([float(r) for r in out.to_ratios()])
            mid_ratios.append(ratios[1])
        assert all(
            mid_ratios[i + 1] <= mid_ratios[i] + 1e-12
            for i in range(len(mid_ratios) - 1)
        )

    def test_phases_interpolate(self):
        a = HarmonicInput(ratios=[Fraction(1), Fraction(5, 4)],
                          phases=[0.0, 0.0])
        b = HarmonicInput(ratios=[Fraction(1), Fraction(5, 4)],
                          phases=[1.0, 2.0])
        mid = interpolate_chords(a, b, 0.5)
        assert sorted(mid.phases) == pytest.approx([0.5, 1.0])


# ============================================================ fade_in_components


class TestFadeInComponents:
    def test_t_zero_recovers_base(self, major_chord):
        ext = extend_harmonics(major_chord, n_harmonics=3)
        out = fade_in_components(major_chord, ext, 0.0)
        # Base components keep their amplitude, new ones are at 0.
        # Total active components should equal extended size.
        assert out.n_components() == ext.n_components()
        # Sum of amplitudes for new components ≈ 0.
        ext_ratios = [float(r) for r in ext.to_ratios()]
        base_ratios = [float(r) for r in major_chord.to_ratios()]
        amps = list(out.amplitudes or [])
        for r, a in zip(ext_ratios, amps):
            is_base = any(abs(r - br) < 1e-6 for br in base_ratios)
            if not is_base:
                assert a == pytest.approx(0.0, abs=1e-12)

    def test_t_one_recovers_extended(self, major_chord):
        # At t=1 the output should reproduce ``extended`` exactly in
        # both ratio order and (normalized) amplitudes — fade_in_components
        # preserves the extended iteration order, so a positional compare
        # is the cleanest invariant to assert.
        ext = extend_harmonics(major_chord, n_harmonics=3)
        out = fade_in_components(major_chord, ext, 1.0)
        np.testing.assert_allclose(
            [float(r) for r in out.to_ratios()],
            [float(r) for r in ext.to_ratios()],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            list(out.normalized_amplitudes()),
            list(ext.normalized_amplitudes()),
            atol=1e-9,
        )

    def test_shared_amps_interpolate_monotonically(self, major_chord):
        # Shared components linearly interpolate from base to extended
        # amplitude. For a chord at t = 0, 0.5, 1.0 the amplitude of the
        # ratio-1 component should change monotonically.
        ext = extend_harmonics(major_chord, n_harmonics=4)
        outs = [fade_in_components(major_chord, ext, t) for t in [0.0, 0.5, 1.0]]
        amps_at_one = []
        for out in outs:
            ratios = [float(r) for r in out.to_ratios()]
            amps = list(out.amplitudes or [])
            j = int(np.argmin(np.abs(np.array(ratios) - 1.0)))
            amps_at_one.append(amps[j])
        # Must be monotonic (either increasing or decreasing depending on
        # whether base or extended has higher amplitude at this ratio).
        diffs = np.diff(amps_at_one)
        assert (diffs >= -1e-12).all() or (diffs <= 1e-12).all()

    def test_metadata(self, major_chord):
        ext = extend_harmonics(major_chord, n_harmonics=2)
        out = fade_in_components(major_chord, ext, 0.3)
        meta = out.metadata.get("transition", {})
        assert meta["kind"] == "fade_in_components"
        assert meta["t"] == pytest.approx(0.3)


# ================================================================ blend_fields


class TestBlendFields:
    def _make_pair(self, major_chord):
        """Two field_2d geometries on the same grid (via same extent/resolution)."""
        kw = dict(extent=1.5, resolution=64)
        gA = harmonic_interference_field_2d(
            major_chord, n_directions=12, **kw,
        )
        gB = quasicrystal_field_2d(
            major_chord, n_fold=7, **kw,
        )
        return gA, gB

    def test_t_zero_recovers_a(self, major_chord):
        gA, gB = self._make_pair(major_chord)
        out = blend_fields(gA, gB, 0.0)
        np.testing.assert_allclose(out.coordinates, gA.coordinates, atol=1e-12)

    def test_t_one_recovers_b(self, major_chord):
        gA, gB = self._make_pair(major_chord)
        out = blend_fields(gA, gB, 1.0)
        np.testing.assert_allclose(out.coordinates, gB.coordinates, atol=1e-12)

    def test_midpoint_is_average(self, major_chord):
        gA, gB = self._make_pair(major_chord)
        out = blend_fields(gA, gB, 0.5)
        np.testing.assert_allclose(
            out.coordinates,
            0.5 * gA.coordinates + 0.5 * gB.coordinates,
            atol=1e-12,
        )

    def test_metadata_records_kinds(self, major_chord):
        gA, gB = self._make_pair(major_chord)
        out = blend_fields(gA, gB, 0.4)
        assert out.metadata["kind"] == "blended"
        meta = out.metadata.get("transition", {})
        assert meta["kind_a"] == "harmonic_interference_field_2d"
        assert meta["kind_b"] == "quasicrystal_field_2d"

    def test_shape_mismatch_rejected(self, major_chord):
        gA = harmonic_interference_field_2d(major_chord, resolution=64)
        gB = harmonic_interference_field_2d(major_chord, resolution=32)
        with pytest.raises(ValueError):
            blend_fields(gA, gB, 0.5)

    def test_grid_mismatch_rejected_by_default(self, major_chord):
        gA = harmonic_interference_field_2d(major_chord, extent=1.5, resolution=64)
        gB = harmonic_interference_field_2d(major_chord, extent=2.0, resolution=64)
        with pytest.raises(ValueError):
            blend_fields(gA, gB, 0.5)

    def test_grid_mismatch_allowed_with_override(self, major_chord):
        gA = harmonic_interference_field_2d(major_chord, extent=1.5, resolution=64)
        gB = harmonic_interference_field_2d(major_chord, extent=2.0, resolution=64)
        out = blend_fields(gA, gB, 0.5, require_same_grid=False)
        assert out.coordinates.shape == (64, 64)

    def test_non_field_2d_rejected(self):
        from biotuner.harmonic_geometry.geometry_data import GeometryData
        a = GeometryData(geom_type="curve_2d", coordinates=np.zeros((10, 2)))
        b = GeometryData(geom_type="curve_2d", coordinates=np.zeros((10, 2)))
        with pytest.raises(ValueError):
            blend_fields(a, b, 0.5)


# ============================================================== composability


class TestComposability:
    def test_chord_morph_through_paradigm(self, major_chord, minor_chord):
        """Smoke: render the same paradigm at several t values without error."""
        ts = np.linspace(0.0, 1.0, 5)
        for t in ts:
            inp = interpolate_chords(major_chord, minor_chord, float(t))
            g = quasicrystal_field_2d(inp, n_fold=7, resolution=32)
            assert g.coordinates.shape == (32, 32)

    def test_extension_morph_through_paradigm(self, major_chord):
        ext = extend_harmonics(major_chord, n_harmonics=4)
        ts = np.linspace(0.0, 1.0, 5)
        for t in ts:
            inp = fade_in_components(major_chord, ext, float(t))
            g = harmonic_interference_field_2d(
                inp, n_directions=8, resolution=32,
            )
            assert g.coordinates.shape == (32, 32)


# ============================================================== public API


class TestPublicAPI:
    def test_module_exports(self):
        from biotuner.harmonic_geometry import (
            blend_fields,
            fade_in_components,
            interpolate_chords,
        )

        assert callable(blend_fields)
        assert callable(fade_in_components)
        assert callable(interpolate_chords)
