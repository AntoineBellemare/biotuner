"""Tests for biotuner.harmonic_geometry.interference_patterns."""

from fractions import Fraction

import numpy as np
import pytest

from biotuner.harmonic_geometry import HarmonicInput
from biotuner.harmonic_geometry.interference_patterns import (
    harmonic_interference_field_2d,
    interference_field_2d,
    quasicrystal_field_2d,
    standing_wave_lattice_2d,
    vortex_field_2d,
    _emitter_positions,
)


@pytest.fixture
def major_chord() -> HarmonicInput:
    return HarmonicInput(
        ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)]
    )


# ===================================== flagship: harmonic_interference_field_2d


class TestHarmonicInterferenceField2D:
    def test_shape_and_metadata(self, major_chord):
        g = harmonic_interference_field_2d(
            major_chord, n_directions=8, resolution=64, extent=1.5,
        )
        assert g.geom_type == "field_2d"
        assert g.coordinates.shape == (64, 64)
        assert g.metadata["kind"] == "harmonic_interference_field_2d"
        assert g.metadata["symmetry"] == "soft_rotational"

    def test_central_peak(self, major_chord):
        g = harmonic_interference_field_2d(
            major_chord, n_directions=12, resolution=129, extent=1.5,
            output="amplitude",
        )
        f = g.coordinates
        ny, nx = f.shape
        centre = f[ny // 2, nx // 2]
        assert centre >= f.max() - 1e-10

    def test_invalid_n_directions(self, major_chord):
        with pytest.raises(ValueError):
            harmonic_interference_field_2d(major_chord, n_directions=1)

    def test_invalid_resolution(self, major_chord):
        with pytest.raises(ValueError):
            harmonic_interference_field_2d(major_chord, resolution=2)

    def test_invalid_extent(self, major_chord):
        with pytest.raises(ValueError):
            harmonic_interference_field_2d(major_chord, extent=0.0)

    def test_invalid_base_period(self, major_chord):
        with pytest.raises(ValueError):
            harmonic_interference_field_2d(major_chord, base_period=0.0)

    def test_unknown_output(self, major_chord):
        with pytest.raises(ValueError):
            harmonic_interference_field_2d(
                major_chord, n_directions=8, resolution=32, output="bogus"
            )


# ============================================== quasicrystal_field_2d


class TestQuasicrystalField2D:
    def test_shape_and_metadata(self, major_chord):
        g = quasicrystal_field_2d(
            major_chord, n_fold=5, resolution=64, extent=1.5,
        )
        assert g.geom_type == "field_2d"
        assert g.coordinates.shape == (64, 64)
        assert g.metadata["kind"] == "quasicrystal_field_2d"
        assert g.metadata["symmetry"] == "discrete_nfold"

    def test_n_fold_rotational_symmetry(self):
        """For zero-phase input, the field should be invariant under
        rotation by 2π/n_fold (because that rotation permutes the
        plane-wave directions back into themselves)."""
        # Use a single component so the symmetry argument is exact.
        inp = HarmonicInput(ratios=[Fraction(1)])
        n_fold = 5
        g = quasicrystal_field_2d(
            inp, n_fold=n_fold, resolution=129, extent=1.5,
            output="amplitude",
        )
        f = g.coordinates
        # Sample two radii at angles 0 and 2π/n_fold and confirm match.
        ny, nx = f.shape
        cy, cx = ny // 2, nx // 2
        # Pick a radius (in pixels).
        r = 24
        # Angle 0: (cx + r, cy)
        v0 = f[cy, cx + r]
        # Angle 2π/n_fold:
        ang = 2.0 * np.pi / n_fold
        ix = int(round(cx + r * np.cos(ang)))
        iy = int(round(cy + r * np.sin(ang)))
        v1 = f[iy, ix]
        assert abs(v0 - v1) / max(abs(v0), 1e-9) < 0.05

    def test_invalid_n_fold(self, major_chord):
        with pytest.raises(ValueError):
            quasicrystal_field_2d(major_chord, n_fold=1)

    def test_invalid_base_period(self, major_chord):
        with pytest.raises(ValueError):
            quasicrystal_field_2d(major_chord, base_period=0.0)

    def test_invalid_resolution(self, major_chord):
        with pytest.raises(ValueError):
            quasicrystal_field_2d(major_chord, resolution=2)


# ============================================== standing_wave_lattice_2d


class TestStandingWaveLattice2D:
    def test_shape_and_metadata(self, major_chord):
        g = standing_wave_lattice_2d(
            major_chord, resolution=64, extent=2.0,
        )
        assert g.geom_type == "field_2d"
        assert g.coordinates.shape == (64, 64)
        assert g.metadata["kind"] == "standing_wave_lattice_2d"
        assert g.metadata["symmetry"] == "cartesian"
        assert g.metadata["n_modes"] == 9  # 3 components → 3² = 9 modes

    def test_xy_swap_symmetry_zero_phase(self, major_chord):
        """When all phases are 0 and cross_phase=False, the (i,j) and
        (j,i) mode pairs are equal so the field is symmetric under
        x↔y swap."""
        g = standing_wave_lattice_2d(
            major_chord, resolution=65, extent=2.0,
            output="amplitude",
        )
        f = g.coordinates
        # Field should equal its transpose under x↔y swap.
        np.testing.assert_allclose(f, f.T, atol=1e-9)

    def test_cross_phase_breaks_symmetry(self):
        """With cross_phase=True and non-zero per-component phases, the
        x↔y symmetry is broken."""
        inp = HarmonicInput(
            ratios=[Fraction(1), Fraction(5, 4)],
            phases=[0.3, 0.7],
        )
        g = standing_wave_lattice_2d(
            inp, resolution=65, extent=2.0,
            cross_phase=True, output="amplitude",
        )
        f = g.coordinates
        # Field should NOT equal its transpose.
        assert not np.allclose(f, f.T, atol=1e-3)

    def test_invalid_base_period(self, major_chord):
        with pytest.raises(ValueError):
            standing_wave_lattice_2d(major_chord, base_period=0.0)

    def test_invalid_resolution(self, major_chord):
        with pytest.raises(ValueError):
            standing_wave_lattice_2d(major_chord, resolution=2)


# ============================================== vortex_field_2d


class TestVortexField2D:
    def test_shape_and_metadata(self, major_chord):
        g = vortex_field_2d(
            major_chord, resolution=64, extent=2.0,
        )
        assert g.geom_type == "field_2d"
        assert g.coordinates.shape == (64, 64)
        assert g.metadata["kind"] == "vortex_field_2d"
        assert g.metadata["symmetry"] == "spiral"
        # Default radial kind is now bessel.
        assert g.parameters["radial_kind"] == "bessel"
        assert g.metadata["radial_kind"] == "bessel"

    def test_charges_from_numerator(self):
        inp = HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)])
        g = vortex_field_2d(inp, use_numerator_charges=True)
        assert g.parameters["charges"] == [1, 5, 3]

    def test_charges_from_rounded(self):
        inp = HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)])
        g = vortex_field_2d(inp, use_numerator_charges=False)
        for c in g.parameters["charges"]:
            assert isinstance(c, int)
            assert c >= 0

    def test_charge_scale(self, major_chord):
        g1 = vortex_field_2d(major_chord, charge_scale=1.0)
        g2 = vortex_field_2d(major_chord, charge_scale=2.0)
        c1 = np.array(g1.parameters["charges"])
        c2 = np.array(g2.parameters["charges"])
        assert (c2 == 2 * c1).all()

    # ── radial_kind: bessel (default) ─────────────────────────────────
    def test_bessel_zero_at_origin_for_nonzero_charge(self):
        """J_l(0) = 0 for l > 0 ⇒ Bessel-mode vortex vanishes at origin."""
        inp = HarmonicInput(ratios=[Fraction(2)])
        g = vortex_field_2d(
            inp, radial_kind="bessel", resolution=129, extent=2.0,
            output="amplitude",
        )
        f = g.coordinates
        ny, nx = f.shape
        cy, cx = ny // 2, nx // 2
        assert f[cy, cx] < f.max() * 0.05

    def test_bessel_has_more_radial_oscillations_than_gaussian(self):
        """The Bessel radial factor oscillates; the Gaussian one doesn't.
        Compare zero-crossings along a radial line for a single-component
        input."""
        inp = HarmonicInput(ratios=[Fraction(3)])
        g_b = vortex_field_2d(
            inp, radial_kind="bessel", resolution=129, extent=3.0,
            output="real",
        )
        g_g = vortex_field_2d(
            inp, radial_kind="gaussian", resolution=129, extent=3.0,
            output="real",
        )
        # Sample +x radial slice from centre.
        ny, nx = g_b.coordinates.shape
        cy, cx = ny // 2, nx // 2
        slice_b = g_b.coordinates[cy, cx:]
        slice_g = g_g.coordinates[cy, cx:]
        # Count sign changes
        zc_b = int(np.sum(np.diff(np.sign(slice_b)) != 0))
        zc_g = int(np.sum(np.diff(np.sign(slice_g)) != 0))
        assert zc_b > zc_g  # Bessel oscillates, Gaussian doesn't

    # ── radial_kind: laguerre_gauss ───────────────────────────────────
    def test_lg_zero_at_origin_for_nonzero_charge(self):
        inp = HarmonicInput(ratios=[Fraction(2)])
        g = vortex_field_2d(
            inp, radial_kind="laguerre_gauss",
            p_index_rule="zero",
            resolution=129, extent=2.0,
            output="amplitude",
        )
        f = g.coordinates
        ny, nx = f.shape
        cy, cx = ny // 2, nx // 2
        assert f[cy, cx] < f.max() * 0.05

    def test_lg_p_indices_from_denominator(self):
        # Major chord 1, 5/4, 3/2 → denominators 1, 4, 2 → p = 0, 3, 1.
        inp = HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)])
        g = vortex_field_2d(
            inp, radial_kind="laguerre_gauss", p_index_rule="denominator",
        )
        assert g.parameters["p_indices"] == [0, 3, 1]

    def test_lg_p_indices_from_index_rule(self):
        inp = HarmonicInput(ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)])
        g = vortex_field_2d(
            inp, radial_kind="laguerre_gauss", p_index_rule="index",
        )
        assert g.parameters["p_indices"] == [0, 1, 2]

    def test_lg_p_indices_override(self, major_chord):
        g = vortex_field_2d(
            major_chord, radial_kind="laguerre_gauss",
            radial_indices=[2, 3, 5],
        )
        assert g.parameters["p_indices"] == [2, 3, 5]

    def test_radial_indices_length_mismatch(self, major_chord):
        with pytest.raises(ValueError):
            vortex_field_2d(
                major_chord, radial_kind="laguerre_gauss",
                radial_indices=[1, 2],  # 3 components needed
            )

    # ── radial_kind: propagating ──────────────────────────────────────
    def test_propagating_oscillates(self):
        inp = HarmonicInput(ratios=[Fraction(2)])
        g = vortex_field_2d(
            inp, radial_kind="propagating", resolution=129, extent=3.0,
            output="real",
        )
        # Should have at least one zero crossing along a radial slice.
        ny, nx = g.coordinates.shape
        slice_ = g.coordinates[ny // 2, nx // 2:]
        zc = int(np.sum(np.diff(np.sign(slice_)) != 0))
        assert zc >= 1

    # ── radial_kind: gaussian (back-compat) ───────────────────────────
    def test_gaussian_kind_still_supported(self, major_chord):
        g = vortex_field_2d(major_chord, radial_kind="gaussian")
        assert g.parameters["radial_kind"] == "gaussian"

    # ── argument validation ───────────────────────────────────────────
    def test_invalid_radial_kind(self, major_chord):
        with pytest.raises(ValueError):
            vortex_field_2d(major_chord, radial_kind="bogus")

    def test_invalid_p_index_rule(self, major_chord):
        with pytest.raises(ValueError):
            vortex_field_2d(major_chord, p_index_rule="bogus")

    def test_invalid_beam_waist(self, major_chord):
        with pytest.raises(ValueError):
            vortex_field_2d(major_chord, beam_waist=0.0)

    def test_invalid_resolution(self, major_chord):
        with pytest.raises(ValueError):
            vortex_field_2d(major_chord, resolution=2)


# ============================================================ Young's-style


class TestEmitterPositions:
    def test_line_layout_centred(self):
        pos = _emitter_positions(4, "line", spacing=2.0)
        assert pos.shape == (4, 2)
        np.testing.assert_allclose(pos[:, 1], 0.0)
        np.testing.assert_allclose(pos[:, 0].mean(), 0.0)

    def test_circle_layout_radius(self):
        pos = _emitter_positions(8, "circle", spacing=3.0)
        radii = np.hypot(pos[:, 0], pos[:, 1])
        np.testing.assert_allclose(radii, 3.0, atol=1e-12)

    def test_pairwise_columns(self):
        pos = _emitter_positions(4, "pairwise", spacing=2.0)
        np.testing.assert_allclose(pos[:2, 0], -1.0)
        np.testing.assert_allclose(pos[2:, 0], +1.0)

    def test_unknown_layout(self):
        with pytest.raises(ValueError):
            _emitter_positions(2, "bogus", 1.0)


class TestInterferenceField2D:
    def test_shape_and_metadata(self, major_chord):
        g = interference_field_2d(
            major_chord, layout="line", spacing=1.0,
            extent=2.0, resolution=64,
        )
        assert g.geom_type == "field_2d"
        assert g.coordinates.shape == (64, 64)
        assert g.metadata["domain"] == "open_2d"
        assert g.metadata["n_sources"] == 3

    def test_field_nonneg_for_amplitude_pow(self, major_chord):
        g = interference_field_2d(
            major_chord, resolution=32, extent=2.0, output="amplitude_pow"
        )
        assert (g.coordinates >= 0).all()

    def test_real_can_be_signed(self, major_chord):
        g = interference_field_2d(
            major_chord, resolution=32, extent=2.0, output="real"
        )
        assert g.coordinates.min() < 0
        assert g.coordinates.max() > 0

    def test_resolution_too_low(self, major_chord):
        with pytest.raises(ValueError):
            interference_field_2d(major_chord, resolution=2)

    def test_extent_invalid(self, major_chord):
        with pytest.raises(ValueError):
            interference_field_2d(major_chord, extent=0.0)

    def test_wavelength_invalid(self, major_chord):
        with pytest.raises(ValueError):
            interference_field_2d(major_chord, base_wavelength=0.0)

    def test_unknown_output(self, major_chord):
        with pytest.raises(ValueError):
            interference_field_2d(major_chord, output="bogus")


# =============================================================== public API


class TestPublicAPI:
    def test_module_exports(self):
        from biotuner.harmonic_geometry import (
            harmonic_interference_field_2d,
            interference_field_2d,
            quasicrystal_field_2d,
            standing_wave_lattice_2d,
            vortex_field_2d,
        )

        assert callable(harmonic_interference_field_2d)
        assert callable(interference_field_2d)
        assert callable(quasicrystal_field_2d)
        assert callable(standing_wave_lattice_2d)
        assert callable(vortex_field_2d)
