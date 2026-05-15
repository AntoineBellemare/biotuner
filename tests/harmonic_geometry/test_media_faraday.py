"""Tests for biotuner.harmonic_geometry.media.parametric.faraday."""

from fractions import Fraction

import numpy as np
import pytest

from biotuner.harmonic_geometry import HarmonicInput
from biotuner.harmonic_geometry.media import Faraday, Granular, Pipeline


@pytest.fixture
def major():
    return HarmonicInput(
        ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)],
        amplitudes=[1.0, 0.7, 0.8],
    )


# =================================================================== basic


class TestFaradayBasic:
    def test_family(self):
        assert Faraday.family == "parametric"

    def test_default_source_is_none(self):
        assert Faraday().default_source() is None

    def test_basic_output(self, major):
        out = Faraday(resolution=64).respond(major)
        assert out.geom_type == "field_2d"
        assert out.coordinates.shape == (64, 64)
        assert np.all(np.isfinite(out.coordinates))
        assert out.metadata["family"] == "parametric"
        assert out.metadata["symmetry"] == "hexagonal"

    def test_callable_shorthand(self, major):
        out = Faraday(resolution=32)(major)
        assert out.geom_type == "field_2d"

    def test_rejects_non_harmonic_input(self):
        with pytest.raises(TypeError):
            Faraday().respond(np.zeros((4, 4)))


# ================================================================ patterns


class TestPatterns:
    @pytest.mark.parametrize(
        "pattern", ["stripe", "square", "hexagonal", "twelve_fold"]
    )
    def test_each_pattern_runs(self, pattern, major):
        out = Faraday(pattern=pattern, resolution=64).respond(major)
        assert out.geom_type == "field_2d"
        assert out.parameters["pattern"] == pattern
        assert out.metadata["symmetry"] == pattern

    def test_pattern_angle_counts(self, major):
        # Each pattern has a known angle count.
        expected = {
            "stripe": 1,
            "square": 2,
            "hexagonal": 3,
            "twelve_fold": 6,
        }
        for pattern, n in expected.items():
            out = Faraday(pattern=pattern, resolution=32).respond(major)
            assert len(out.parameters["angles"]) == n

    def test_random_pattern_with_seed_is_deterministic(self, major):
        a = Faraday(pattern="random", n_random=4, seed=42,
                    resolution=48).respond(major)
        b = Faraday(pattern="random", n_random=4, seed=42,
                    resolution=48).respond(major)
        assert np.allclose(a.coordinates, b.coordinates)
        assert a.parameters["angles"] == b.parameters["angles"]

    def test_random_pattern_different_seeds_differ(self, major):
        a = Faraday(pattern="random", n_random=4, seed=1,
                    resolution=48).respond(major)
        b = Faraday(pattern="random", n_random=4, seed=2,
                    resolution=48).respond(major)
        assert not np.allclose(a.coordinates, b.coordinates)

    def test_rejects_unknown_pattern(self):
        with pytest.raises(ValueError):
            Faraday(pattern="bogus")


# ============================================================ dispersion


class TestDispersion:
    @pytest.mark.parametrize("disp", ["gravity", "capillary", "mixed"])
    def test_each_dispersion_runs(self, disp, major):
        out = Faraday(dispersion=disp, resolution=48).respond(major)
        assert out.parameters["dispersion"] == disp
        ks = out.parameters["wavenumbers"]
        assert len(ks) == 3
        assert all(k > 0 for k in ks)

    def test_gravity_scales_quadratically(self, major):
        # k_i / k_0 = (ratio_i)² in pure-gravity regime.
        out = Faraday(dispersion="gravity", base_wavenumber=10.0,
                       resolution=32).respond(major)
        ks = out.parameters["wavenumbers"]
        ratios = [1.0, 5 / 4, 3 / 2]
        for k, r in zip(ks, ratios):
            assert k == pytest.approx(10.0 * r ** 2, rel=1e-6)

    def test_capillary_scales_two_thirds(self, major):
        out = Faraday(dispersion="capillary", base_wavenumber=10.0,
                       resolution=32).respond(major)
        ks = out.parameters["wavenumbers"]
        ratios = [1.0, 5 / 4, 3 / 2]
        for k, r in zip(ks, ratios):
            assert k == pytest.approx(10.0 * r ** (2.0 / 3.0), rel=1e-6)

    def test_mixed_between_gravity_and_capillary(self, major):
        ks_g = Faraday(dispersion="gravity", resolution=32).respond(major).parameters["wavenumbers"]
        ks_c = Faraday(dispersion="capillary", resolution=32).respond(major).parameters["wavenumbers"]
        ks_m = Faraday(dispersion="mixed", resolution=32).respond(major).parameters["wavenumbers"]
        # k_0 is anchored to base_wavenumber regardless of regime.
        assert ks_g[0] == pytest.approx(ks_m[0], rel=1e-6)
        assert ks_c[0] == pytest.approx(ks_m[0], rel=1e-6)
        # For higher ratios, mixed should lie between capillary and gravity
        # (gravity grows fastest, capillary slowest in k).
        for i in (1, 2):
            assert ks_c[i] <= ks_m[i] + 1e-9
            assert ks_m[i] <= ks_g[i] + 1e-9

    def test_rejects_unknown_dispersion(self):
        with pytest.raises(ValueError):
            Faraday(dispersion="bogus")


# ============================================================== knobs


class TestKnobs:
    def test_viscosity_damps_amplitude(self, major):
        low = Faraday(viscosity=0.0, resolution=64).respond(major)
        high = Faraday(viscosity=0.01, resolution=64).respond(major)
        # Higher viscosity damps the field magnitude.
        assert float(np.max(np.abs(high.coordinates))) < float(
            np.max(np.abs(low.coordinates))
        )

    def test_drive_amplitude_scales_linearly(self, major):
        low = Faraday(drive_amplitude=1.0, viscosity=0.0,
                       resolution=48).respond(major)
        high = Faraday(drive_amplitude=2.5, viscosity=0.0,
                        resolution=48).respond(major)
        # Output is |sum of plane waves|; scales linearly with drive_amplitude.
        ratio = float(np.max(np.abs(high.coordinates))) / float(
            np.max(np.abs(low.coordinates))
        )
        assert ratio == pytest.approx(2.5, rel=0.05)

    def test_base_wavenumber_controls_pattern_scale(self, major):
        # Larger k = more oscillations across the same domain.
        out = Faraday(base_wavenumber=20.0, resolution=128).respond(major)
        # Count zero-crossings along a horizontal cut as a coarse oscillation
        # proxy; more wavenumber should give more zero-crossings.
        field = out.coordinates.real if np.iscomplexobj(out.coordinates) else out.coordinates
        slice_low = Faraday(base_wavenumber=5.0, resolution=128).respond(major).coordinates
        n_low = np.sum(np.diff(np.sign(slice_low[64] - np.mean(slice_low[64]))) != 0)
        n_high = np.sum(np.diff(np.sign(field[64] - np.mean(field[64]))) != 0)
        assert n_high > n_low

    @pytest.mark.parametrize("output", ["amplitude", "intensity", "real"])
    def test_output_modes(self, output, major):
        out = Faraday(output=output, resolution=48).respond(major)
        assert out.geom_type == "field_2d"
        if output == "amplitude":
            assert np.all(out.coordinates >= 0)
        elif output == "intensity":
            assert np.all(out.coordinates >= 0)

    def test_rejects_bad_output(self):
        with pytest.raises(ValueError):
            Faraday(output="bogus")


# ============================================================ validation


class TestValidation:
    def test_rejects_zero_base_wavenumber(self):
        with pytest.raises(ValueError):
            Faraday(base_wavenumber=0.0)

    def test_rejects_negative_viscosity(self):
        with pytest.raises(ValueError):
            Faraday(viscosity=-0.1)

    def test_rejects_zero_extent(self):
        with pytest.raises(ValueError):
            Faraday(extent=0.0)

    def test_rejects_small_resolution(self):
        with pytest.raises(ValueError):
            Faraday(resolution=2)

    def test_rejects_unexpected_override(self, major):
        with pytest.raises(TypeError, match="Unexpected override"):
            Faraday().respond(major, bogus=5)


# =========================================================== composition


class TestComposition:
    def test_pipeline_faraday_to_granular(self, major):
        pipe = Pipeline(
            Faraday(pattern="hexagonal", resolution=64),
            Granular(affinity=1.0, temperature=0.1),
        )
        out = pipe(major)
        assert out.geom_type == "field_2d"
        assert out.metadata["family"] == "transport"
        assert out.metadata["source_kind"] == "faraday_field_2d"

    def test_faraday_into_granular_auto_wrap_fails(self, major):
        # Granular's default_source is RigidPlate (eigenmode), not Faraday.
        # Passing a HarmonicInput to Granular still auto-wraps with RigidPlate,
        # not Faraday — that's the documented contract.
        sand = Granular().respond(major)
        assert sand.metadata["source_kind"] == "chladni_field_rectangular"
