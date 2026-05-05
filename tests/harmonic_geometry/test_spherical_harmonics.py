"""Tests for biotuner.harmonic_geometry.spherical_harmonics."""

from fractions import Fraction

import numpy as np
import pytest

from biotuner.harmonic_geometry import HarmonicInput
from biotuner.harmonic_geometry.spherical_harmonics import (
    ratios_to_modes_lm,
    single_spherical_harmonic,
    spherical_harmonic_field,
    spherical_harmonic_from_input,
    spherical_harmonic_mesh,
    spherical_harmonic_temporal,
    _real_ylm,
)


# ============================================================ ratios_to_modes_lm


class TestRatiosToModesLM:
    def test_zonal_integer_ratios(self):
        # Integer ratios behave the same under both l_rule values.
        modes = ratios_to_modes_lm([1.0, 2.0, 3.0], mode_rule="zonal")
        assert modes == [(1, 0), (2, 0), (3, 0)]

    def test_l_rule_numerator_spreads_chord(self):
        # Major chord 1, 5/4, 3/2 should produce l = 1, 5, 3 (numerators).
        modes = ratios_to_modes_lm(
            [1.0, 5.0 / 4.0, 3.0 / 2.0],
            mode_rule="zonal",
            l_rule="numerator",
            max_l=10,
        )
        assert modes == [(1, 0), (5, 0), (3, 0)]

    def test_l_rule_rounded_collapses_chord(self):
        # Same chord with the legacy 'rounded' rule collapses to l = 1, 1, 2.
        modes = ratios_to_modes_lm(
            [1.0, 5.0 / 4.0, 3.0 / 2.0],
            mode_rule="zonal",
            l_rule="rounded",
        )
        assert modes == [(1, 0), (1, 0), (2, 0)]

    def test_l_rule_rounded_irrational(self):
        # 1.6 → l=2, 2.4 → l=2, 4.51 → l=5
        modes = ratios_to_modes_lm(
            [1.6, 2.4, 4.51], mode_rule="zonal", l_rule="rounded"
        )
        assert modes == [(2, 0), (2, 0), (5, 0)]

    def test_sectoral_alternates_sign(self):
        modes = ratios_to_modes_lm(
            [3, 4, 5, 6], mode_rule="sectoral", l_rule="rounded"
        )
        # idx 0/2 → +l, idx 1/3 → -l
        assert modes == [(3, 3), (4, -4), (5, 5), (6, -6)]

    def test_chord_balanced_pool(self):
        # For l >= 2 the pool is [0, l, -l, l//2, -l//2]; cycles through.
        modes = ratios_to_modes_lm(
            [3, 3, 3, 3, 3], mode_rule="chord_balanced", l_rule="rounded"
        )
        ms = [m for _, m in modes]
        assert ms == [0, 3, -3, 1, -1]  # l=3 ⇒ pool [0, 3, -3, 1, -1]

    def test_clamp_max_l(self):
        modes = ratios_to_modes_lm(
            [15.0], mode_rule="zonal", l_rule="rounded", max_l=5
        )
        assert modes == [(5, 0)]

    def test_l_zero_only_m_zero(self):
        modes = ratios_to_modes_lm(
            [0.4, 0.4], mode_rule="sectoral", l_rule="rounded"
        )
        # round(0.4) = 0 ⇒ l=0, |m| must be 0
        for l, m in modes:
            assert l == 0 and m == 0

    def test_unknown_mode_rule(self):
        with pytest.raises(ValueError):
            ratios_to_modes_lm([1.0], mode_rule="bogus")

    def test_unknown_l_rule(self):
        with pytest.raises(ValueError):
            ratios_to_modes_lm([1.0], l_rule="bogus")

    def test_zero_ratio_rejected(self):
        with pytest.raises(ValueError):
            ratios_to_modes_lm([0.0])

    def test_negative_ratio_rejected(self):
        with pytest.raises(ValueError):
            ratios_to_modes_lm([-1.0])

    def test_negative_max_l_rejected(self):
        with pytest.raises(ValueError):
            ratios_to_modes_lm([1.0], max_l=-1)

    def test_rounded_mode_rule_alias(self):
        a = ratios_to_modes_lm([1.5, 2.5, 3.5], mode_rule="rounded")
        b = ratios_to_modes_lm([1.5, 2.5, 3.5], mode_rule="zonal")
        assert a == b


# ============================================================ Y_l^m primitives


class TestRealYLM:
    """Validate against known closed-form spherical harmonics."""

    def test_y00_constant(self):
        # Y_0^0 = 1 / (2 sqrt(pi))
        theta = np.linspace(0.01, np.pi - 0.01, 7)
        phi = np.linspace(0.0, 2 * np.pi, 11, endpoint=False)
        T, P = np.meshgrid(theta, phi, indexing="ij")
        Y = _real_ylm(0, 0, T, P)
        expected = 1.0 / (2.0 * np.sqrt(np.pi))
        assert np.allclose(Y, expected, atol=1e-10)

    def test_y10_dipole_along_z(self):
        # Y_1^0 = (1/2) * sqrt(3 / pi) * cos(theta) — depends only on theta.
        theta = np.linspace(0.01, np.pi - 0.01, 9)
        phi = np.linspace(0.0, 2 * np.pi, 13, endpoint=False)
        T, P = np.meshgrid(theta, phi, indexing="ij")
        Y = _real_ylm(1, 0, T, P)
        expected = 0.5 * np.sqrt(3.0 / np.pi) * np.cos(T)
        assert np.allclose(Y, expected, atol=1e-10)

    def test_y11_real_is_x_component(self):
        # Real Y_1^1 ∝ sin(theta) cos(phi) — the x/r angular pattern.
        theta = np.linspace(0.1, np.pi - 0.1, 6)
        phi = np.linspace(0.0, 2 * np.pi, 8, endpoint=False)
        T, P = np.meshgrid(theta, phi, indexing="ij")
        Y = _real_ylm(1, 1, T, P)
        # The pattern matches sin(theta)*cos(phi) up to a normalisation
        # constant. Just check the proportionality is exact.
        ref = np.sin(T) * np.cos(P)
        # Both vanish at the poles and at phi=±pi/2; ratio is constant
        # everywhere ref != 0.
        nonzero = np.abs(ref) > 1e-2
        ratios = Y[nonzero] / ref[nonzero]
        assert np.allclose(ratios, ratios[0], rtol=1e-6)

    def test_y1_minus1_real_is_y_component(self):
        # Real Y_1^{-1} ∝ sin(theta) sin(phi).
        theta = np.linspace(0.1, np.pi - 0.1, 6)
        phi = np.linspace(0.05, 2 * np.pi - 0.05, 8)
        T, P = np.meshgrid(theta, phi, indexing="ij")
        Y = _real_ylm(1, -1, T, P)
        ref = np.sin(T) * np.sin(P)
        nonzero = np.abs(ref) > 1e-2
        ratios = Y[nonzero] / ref[nonzero]
        assert np.allclose(ratios, ratios[0], rtol=1e-6)

    def test_real_ylm_orthonormality(self):
        # ∫ Y_l^m Y_l'^m' dΩ = δ_ll' δ_mm'.
        # Approximate the integral on a fine grid via Simpson-ish rule:
        # dΩ = sin(theta) dtheta dphi.
        n_t, n_p = 256, 512
        theta = np.linspace(0.0, np.pi, n_t)
        phi = np.linspace(0.0, 2 * np.pi, n_p, endpoint=False)
        T, P = np.meshgrid(theta, phi, indexing="ij")
        weight = np.sin(T)
        dtheta = np.pi / (n_t - 1)
        dphi = 2 * np.pi / n_p

        def inner(l1, m1, l2, m2):
            Y1 = _real_ylm(l1, m1, T, P)
            Y2 = _real_ylm(l2, m2, T, P)
            return np.sum(Y1 * Y2 * weight) * dtheta * dphi

        # Same mode: norm ≈ 1
        assert abs(inner(2, 1, 2, 1) - 1.0) < 5e-3
        assert abs(inner(3, -2, 3, -2) - 1.0) < 5e-3
        # Different (l, m): orthogonal
        assert abs(inner(1, 0, 2, 0)) < 5e-3
        assert abs(inner(2, 1, 2, -1)) < 5e-3
        assert abs(inner(2, 2, 3, 2)) < 5e-3

    def test_invalid_l(self):
        theta = np.array([1.0])
        phi = np.array([1.0])
        with pytest.raises(ValueError):
            _real_ylm(-1, 0, theta, phi)

    def test_invalid_m(self):
        theta = np.array([1.0])
        phi = np.array([1.0])
        with pytest.raises(ValueError):
            _real_ylm(2, 3, theta, phi)


# ============================================================== single mode


class TestSingleSphericalHarmonic:
    def test_shape(self):
        g = single_spherical_harmonic(2, 1, n_theta=32, n_phi=64)
        assert g.geom_type == "field_2d"
        assert g.coordinates.shape == (32, 64)
        T, P = g.field_grid
        assert T.shape == (32, 64)
        assert P.shape == (32, 64)

    def test_metadata(self):
        g = single_spherical_harmonic(3, -2)
        assert g.metadata["domain"] == "sphere"
        assert g.parameters["l"] == 3
        assert g.parameters["m"] == -2
        assert g.parameters["real"] is True

    def test_complex_form(self):
        g = single_spherical_harmonic(2, 1, n_theta=16, n_phi=32, real=False)
        assert np.iscomplexobj(g.coordinates)

    def test_low_resolution_rejected(self):
        with pytest.raises(ValueError):
            single_spherical_harmonic(1, 0, n_theta=2, n_phi=64)
        with pytest.raises(ValueError):
            single_spherical_harmonic(1, 0, n_theta=64, n_phi=2)


# =============================================================== field super


class TestSphericalHarmonicField:
    def test_shape(self):
        g = spherical_harmonic_field([(1, 0), (2, 0), (3, 0)], n_theta=32, n_phi=64)
        assert g.geom_type == "field_2d"
        assert g.coordinates.shape == (32, 64)

    def test_uniform_amplitudes_default(self):
        g = spherical_harmonic_field([(1, 0), (2, 0)])
        amps = g.parameters["amps"]
        assert len(amps) == 2
        assert amps[0] == pytest.approx(0.5)
        assert amps[1] == pytest.approx(0.5)

    def test_amps_length_mismatch(self):
        with pytest.raises(ValueError):
            spherical_harmonic_field([(1, 0), (2, 0)], amps=[1.0])

    def test_phases_length_mismatch(self):
        with pytest.raises(ValueError):
            spherical_harmonic_field([(1, 0), (2, 0)], phases=[0.0])

    def test_empty_modes_rejected(self):
        with pytest.raises(ValueError):
            spherical_harmonic_field([])

    def test_low_resolution_rejected(self):
        with pytest.raises(ValueError):
            spherical_harmonic_field([(1, 0)], n_theta=3, n_phi=64)

    def test_phase_scaling(self):
        # Y_1^0 alone with phase = π/2 should produce a (near-)zero field
        # because cos(π/2) = 0. With phase = 0, the field has nonzero
        # amplitude.
        g0 = spherical_harmonic_field([(1, 0)], phases=[0.0], n_theta=32, n_phi=64)
        gpi2 = spherical_harmonic_field(
            [(1, 0)], phases=[np.pi / 2], n_theta=32, n_phi=64
        )
        assert np.max(np.abs(g0.coordinates)) > 0.1
        assert np.max(np.abs(gpi2.coordinates)) < 1e-12

    def test_complex_field(self):
        g = spherical_harmonic_field(
            [(2, 1), (3, -1)], real=False, n_theta=16, n_phi=32
        )
        assert np.iscomplexobj(g.coordinates)


# =========================================================== HarmonicInput


class TestSphericalHarmonicFromInput:
    def test_major_chord_zonal_numerator(self):
        # Default l_rule='numerator' — Major chord 1, 5/4, 3/2 has
        # numerators 1, 5, 3 → l values 1, 5, 3.
        inp = HarmonicInput(ratios=[Fraction(4, 4), Fraction(5, 4), Fraction(6, 4)])
        g = spherical_harmonic_from_input(inp, mode_rule="zonal", n_theta=32, n_phi=64)
        assert g.parameters["modes_lm"] == [(1, 0), (5, 0), (3, 0)]
        assert g.coordinates.shape == (32, 64)

    def test_major_chord_zonal_rounded(self):
        inp = HarmonicInput(ratios=[Fraction(4, 4), Fraction(5, 4), Fraction(6, 4)])
        g = spherical_harmonic_from_input(
            inp, mode_rule="zonal", l_rule="rounded", n_theta=32, n_phi=64
        )
        # Ratios 1, 1.25, 1.5 → round to 1, 1, 2.
        assert g.parameters["modes_lm"] == [(1, 0), (1, 0), (2, 0)]

    def test_amps_from_input(self):
        inp = HarmonicInput(
            ratios=[1.0, 1.5, 2.0], amplitudes=[1.0, 0.5, 0.25]
        )
        g = spherical_harmonic_from_input(inp, n_theta=16, n_phi=32)
        amps = g.parameters["amps"]
        # amplitudes are sum-normalised (not max-normalised)
        assert sum(amps) == pytest.approx(1.0)
        assert amps[0] > amps[1] > amps[2]

    def test_phases_from_input(self):
        inp = HarmonicInput(ratios=[1.0, 2.0], phases=[0.1, 0.2])
        g = spherical_harmonic_from_input(inp, n_theta=16, n_phi=32)
        assert g.parameters["phases"] == pytest.approx([0.1, 0.2])

    def test_chord_balanced_produces_variety(self):
        inp = HarmonicInput(ratios=[2.0, 2.0, 2.0, 2.0])
        g = spherical_harmonic_from_input(
            inp, mode_rule="chord_balanced", l_rule="rounded"
        )
        ms = [m for _, m in g.parameters["modes_lm"]]
        # The mode_rule should distribute m across the chord components
        # rather than collapsing them all to m=0.
        assert len(set(ms)) > 1


# ===================================================================== mesh


class TestSphericalHarmonicMesh:
    def test_shape(self):
        inp = HarmonicInput(ratios=[1.0, 1.5, 2.0])
        g = spherical_harmonic_mesh(inp, n_theta=24, n_phi=48)
        assert g.geom_type == "mesh_3d"
        assert g.coordinates.shape == (24 * 48, 3)
        # 2 triangles per quad, (n_theta-1) × (n_phi-1) quads
        assert g.faces.shape == (2 * 23 * 47, 3)
        assert g.weights.shape == (24 * 48,)

    def test_zero_epsilon_unit_sphere(self):
        inp = HarmonicInput(ratios=[1.0, 2.0])
        g = spherical_harmonic_mesh(inp, epsilon=0.0, n_theta=16, n_phi=32)
        radii = np.linalg.norm(g.coordinates, axis=1)
        assert np.allclose(radii, 1.0, atol=1e-12)

    def test_radial_displacement_bounded(self):
        # With the field rescaled to peak |Ψ̂| = 1, vertex radii fall in
        # [1 - epsilon, 1 + epsilon].
        inp = HarmonicInput(ratios=[1.0, 1.5, 2.0])
        eps = 0.2
        g = spherical_harmonic_mesh(inp, epsilon=eps, n_theta=24, n_phi=48)
        radii = np.linalg.norm(g.coordinates, axis=1)
        assert radii.min() >= 1.0 - eps - 1e-9
        assert radii.max() <= 1.0 + eps + 1e-9

    def test_negative_epsilon_rejected(self):
        inp = HarmonicInput(ratios=[1.0])
        with pytest.raises(ValueError):
            spherical_harmonic_mesh(inp, epsilon=-0.1)

    def test_face_indices_valid(self):
        inp = HarmonicInput(ratios=[1.0, 2.0])
        g = spherical_harmonic_mesh(inp, n_theta=16, n_phi=32)
        n_v = g.coordinates.shape[0]
        assert g.faces.min() >= 0
        assert g.faces.max() < n_v

    def test_metadata(self):
        inp = HarmonicInput(ratios=[1.0, 2.0])
        g = spherical_harmonic_mesh(inp, n_theta=16, n_phi=32)
        assert g.metadata["kind"] == "spherical_harmonic_mesh"
        assert g.metadata["domain"] == "sphere"
        assert g.metadata["n_vertices"] == 16 * 32


# ============================================================ temporal


class TestSphericalHarmonicTemporal:
    def test_t_zero_matches_static(self):
        inp = HarmonicInput(peaks=[100.0, 150.0, 200.0])
        gt = spherical_harmonic_temporal(inp, t=0.0, n_theta=24, n_phi=48)
        gs = spherical_harmonic_from_input(inp, n_theta=24, n_phi=48)
        # phases differ by 2π·f·0 = 0 → fields should match exactly
        np.testing.assert_allclose(gt.coordinates, gs.coordinates, atol=1e-10)

    def test_metadata_kind(self):
        inp = HarmonicInput(peaks=[100.0, 200.0])
        g = spherical_harmonic_temporal(inp, t=0.5, n_theta=16, n_phi=32)
        assert g.metadata["kind"] == "spherical_harmonic_temporal"
        assert g.parameters["t"] == pytest.approx(0.5)

    def test_phase_drift_present(self):
        inp = HarmonicInput(peaks=[100.0, 200.0])
        g = spherical_harmonic_temporal(inp, t=0.001, n_theta=16, n_phi=32)
        # Drift = 2π · f · t for each component, so phases should be
        # proportional to peaks.
        phases = np.asarray(g.parameters["phases"])
        expected = 2.0 * np.pi * np.array([100.0, 200.0]) * 0.001
        np.testing.assert_allclose(phases, expected, atol=1e-10)

    def test_periodic_in_time(self):
        # cos((φ + 2π)) == cos(φ) ⇒ field at t and t + 1/f1 should differ
        # only by the contributions of OTHER peaks. With a single peak
        # the field is exactly periodic.
        inp = HarmonicInput(peaks=[100.0])
        g0 = spherical_harmonic_temporal(inp, t=0.0, n_theta=16, n_phi=32)
        gT = spherical_harmonic_temporal(inp, t=0.01, n_theta=16, n_phi=32)
        # period = 1/100 = 0.01 s
        np.testing.assert_allclose(g0.coordinates, gT.coordinates, atol=1e-10)


# ============================================================== public API


class TestPublicAPI:
    def test_module_exports(self):
        from biotuner.harmonic_geometry import (
            ratios_to_modes_lm,
            single_spherical_harmonic,
            spherical_harmonic_field,
            spherical_harmonic_from_input,
            spherical_harmonic_mesh,
            spherical_harmonic_temporal,
        )

        assert callable(ratios_to_modes_lm)
        assert callable(single_spherical_harmonic)
        assert callable(spherical_harmonic_field)
        assert callable(spherical_harmonic_from_input)
        assert callable(spherical_harmonic_mesh)
        assert callable(spherical_harmonic_temporal)
