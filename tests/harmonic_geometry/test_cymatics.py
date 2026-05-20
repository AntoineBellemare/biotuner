"""Tests for the cymatics-style additions to
:mod:`biotuner.harmonic_geometry.media.eigenmode.rigid_plate`,
:mod:`biotuner.harmonic_geometry.media.transport.granular`,
and :mod:`biotuner.harmonic_geometry.plotting`.
"""
from __future__ import annotations

from fractions import Fraction

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from biotuner.harmonic_geometry import HarmonicInput  # noqa: E402
from biotuner.harmonic_geometry.media import (  # noqa: E402
    Circular,
    Granular,
    Pipeline,
    Rectangular,
    RigidPlate,
)
from biotuner.harmonic_geometry.media.eigenmode.rigid_plate import (  # noqa: E402
    _auto_resolution_for_modes,
    _auto_sigma_for_modes,
    _cap_modes,
    _d4_symmetrize,
    chladni_field_pairwise,
    chladni_field_triple_antisymmetric,
    chladni_nodal_density,
    chord_to_int_modes,
)
from biotuner.harmonic_geometry.plotting import (  # noqa: E402
    animate_chord_sequence,
    draw_chladni_painted,
    draw_chladni_sand,
)


# ─────────────────────────────────────────────────────── chord_to_int_modes


class TestChordToIntModes:
    """Lossless integer-mode extraction from a chord's Fraction ratios."""

    def test_pure_integer_chord(self):
        assert chord_to_int_modes(
            [Fraction(4), Fraction(5), Fraction(6)]
        ) == [4, 5, 6]

    def test_major_just_intonation(self):
        # [1, 5/4, 3/2]: LCM of denominators (1, 4, 2) = 4 → [4, 5, 6].
        chord = [Fraction(1), Fraction(5, 4), Fraction(3, 2)]
        assert chord_to_int_modes(chord) == [4, 5, 6]

    def test_dom7_just_intonation(self):
        chord = [Fraction(1), Fraction(5, 4), Fraction(3, 2), Fraction(7, 4)]
        assert chord_to_int_modes(chord) == [4, 5, 6, 7]

    def test_dim7_12tet_flavoured_blows_up(self):
        # Denominators 5, 5, 7 → LCM = 35 → [35, 42, 49, 60].
        chord = [Fraction(1), Fraction(6, 5), Fraction(7, 5), Fraction(12, 7)]
        assert chord_to_int_modes(chord) == [35, 42, 49, 60]

    def test_empty_chord(self):
        assert chord_to_int_modes([]) == []

    def test_int_inputs(self):
        assert chord_to_int_modes([2, 3, 5]) == [2, 3, 5]

    def test_float_inputs_via_limit_denominator(self):
        assert chord_to_int_modes([1.0, 1.25, 1.5]) == [4, 5, 6]

    def test_tuple_inputs(self):
        assert chord_to_int_modes([(4, 1), (5, 1), (6, 1)]) == [4, 5, 6]

    def test_round_trip_via_harmonic_input(self):
        chord = HarmonicInput(
            ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)]
        )
        assert chord_to_int_modes(chord.to_ratios()) == [4, 5, 6]


# ───────────────────────────────────────────────────────── auto-σ / -res / cap


class TestAutoSigma:
    def test_scales_inverse_with_peak_wavenumber(self):
        s_low = _auto_sigma_for_modes([2, 3, 5])
        s_high = _auto_sigma_for_modes([20, 30, 50])
        assert s_low > s_high

    def test_clamped_to_lower_bound_for_huge_peaks(self):
        assert _auto_sigma_for_modes([1000]) == pytest.approx(0.005)

    def test_clamped_to_upper_bound_for_tiny_peaks(self):
        assert _auto_sigma_for_modes([1]) == pytest.approx(0.18)

    def test_pair_count_scales_sigma_up(self):
        s3 = _auto_sigma_for_modes([4, 5, 6], n_pairs=3)
        s6 = _auto_sigma_for_modes([4, 5, 6], n_pairs=6)
        assert s6 > s3

    def test_empty_modes_does_not_crash(self):
        assert _auto_sigma_for_modes([]) > 0


class TestAutoResolution:
    def test_floor_for_small_chords(self):
        assert _auto_resolution_for_modes([2, 3, 5]) == 400

    def test_scales_with_peak(self):
        res_low = _auto_resolution_for_modes([4, 5, 6])
        res_hi = _auto_resolution_for_modes([60, 80, 100])
        assert res_hi > res_low

    def test_oversample_factor(self):
        # peak=60, oversample=8 → 480.
        assert _auto_resolution_for_modes([10, 30, 60]) == 480

    def test_empty(self):
        assert _auto_resolution_for_modes([]) >= 400


class TestCapModes:
    def test_no_cap_when_already_under(self):
        assert _cap_modes([4, 5, 6], 12) == [4, 5, 6]

    def test_proportional_scaling_preserves_ratios(self):
        result = _cap_modes([35, 42, 49, 60], 12)
        assert max(result) == pytest.approx(12.0)
        for r, original in zip(result, [35, 42, 49, 60]):
            assert r == pytest.approx(original * 12.0 / 60.0)

    def test_none_cap_is_passthrough(self):
        assert _cap_modes([35, 42, 49, 60], None) == [35, 42, 49, 60]

    def test_empty_modes(self):
        assert _cap_modes([], 10) == []


class TestD4Symmetrize:
    def test_max_makes_d4_invariant(self):
        rng = np.random.default_rng(0)
        field = rng.random((16, 16))
        out = _d4_symmetrize(field, mode="max")
        for k in range(1, 4):
            np.testing.assert_allclose(np.rot90(out, k), out, atol=1e-10)
        np.testing.assert_allclose(out.T, out, atol=1e-10)

    def test_max_idempotent_on_already_symmetric_field(self):
        n = 16
        x = np.linspace(-1, 1, n)
        X, Y = np.meshgrid(x, x)
        sym = np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)
        out = _d4_symmetrize(sym, mode="max")
        np.testing.assert_allclose(out, sym, atol=1e-10)

    def test_sum_average_of_uniform(self):
        np.testing.assert_allclose(
            _d4_symmetrize(np.ones((8, 8)), mode="sum"),
            np.ones((8, 8)),
            atol=1e-10,
        )

    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            _d4_symmetrize(np.zeros((10, 12)), mode="max")

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            _d4_symmetrize(np.zeros((8, 8)), mode="invalid")


# ─────────────────────────────────────────────────────── chladni_field_pairwise


class TestChladniFieldPairwise:
    def test_basic_output_shape_and_kind(self):
        g = chladni_field_pairwise([2, 3, 5], resolution=64)
        assert g.geom_type == "field_2d"
        assert g.coordinates.shape == (64, 64)
        assert g.metadata["kind"] == "chladni_field_pairwise"
        assert g.metadata["scheme"] == "pairwise_antisymmetric"

    def test_int_modes_stored_in_params(self):
        g = chladni_field_pairwise([2, 3, 5], resolution=32)
        assert g.parameters["int_modes"] == [2, 3, 5]

    def test_pair_subset_all_three_ratios(self):
        g = chladni_field_pairwise(
            [2, 3, 5], pair_subset="all", resolution=32
        )
        assert g.parameters["n_pairs"] == 3
        assert g.parameters["pair_subset"] == "all"

    def test_pair_subset_all_four_ratios(self):
        g = chladni_field_pairwise(
            [4, 5, 6, 7], pair_subset="all", resolution=32
        )
        assert g.parameters["n_pairs"] == 6

    def test_pair_subset_adjacent(self):
        g = chladni_field_pairwise(
            [4, 5, 6, 7], pair_subset="adjacent", resolution=32
        )
        assert g.parameters["n_pairs"] == 3
        assert g.parameters["pairs"] == [(4.0, 5.0), (5.0, 6.0), (6.0, 7.0)]

    def test_pair_subset_root(self):
        g = chladni_field_pairwise(
            [4, 5, 6, 7], pair_subset="root", resolution=32
        )
        assert g.parameters["n_pairs"] == 3
        assert g.parameters["pairs"] == [(4.0, 5.0), (4.0, 6.0), (4.0, 7.0)]

    def test_pair_subset_auto_three_ratios_picks_all(self):
        g = chladni_field_pairwise(
            [4, 5, 6], pair_subset="auto", resolution=32
        )
        assert g.parameters["pair_subset"] == "all"
        assert g.parameters["n_pairs"] == 3

    def test_pair_subset_auto_four_ratios_picks_root(self):
        g = chladni_field_pairwise(
            [4, 5, 6, 7], pair_subset="auto", resolution=32
        )
        assert g.parameters["pair_subset"] == "root"
        assert g.parameters["n_pairs"] == 3

    def test_explicit_pair_list(self):
        g = chladni_field_pairwise(
            [4, 5, 6, 7], pair_subset=[(4, 7), (5, 6)], resolution=32
        )
        assert g.parameters["n_pairs"] == 2
        assert g.parameters["pair_subset"] == "explicit"

    def test_invalid_pair_subset_string(self):
        with pytest.raises(ValueError, match="pair_subset"):
            chladni_field_pairwise(
                [4, 5, 6], pair_subset="bogus", resolution=32
            )

    def test_invalid_pair_subset_non_iterable(self):
        with pytest.raises(ValueError, match="pair_subset"):
            chladni_field_pairwise([4, 5, 6], pair_subset=42, resolution=32)

    def test_too_few_modes_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            chladni_field_pairwise([5], resolution=32)

    def test_symmetric_vs_antisymmetric_differ(self):
        anti = chladni_field_pairwise(
            [4, 5, 6], antisymmetric=True, resolution=32
        )
        sym = chladni_field_pairwise(
            [4, 5, 6], antisymmetric=False, resolution=32
        )
        assert anti.metadata["scheme"] == "pairwise_antisymmetric"
        assert sym.metadata["scheme"] == "pairwise_symmetric"
        assert not np.allclose(anti.coordinates, sym.coordinates)

    def test_max_mode_caps_pair_components(self):
        g = chladni_field_pairwise(
            [20, 30, 40], max_mode=10, resolution=32
        )
        max_in_pairs = max(max(p) for p in g.parameters["pairs"])
        assert max_in_pairs == pytest.approx(10.0)

    def test_symmetry_stored_in_params(self):
        g = chladni_field_pairwise(
            [4, 5, 6], symmetry="d4_max", resolution=32
        )
        assert g.parameters["symmetry"] == "d4_max"

    def test_field_NOT_symmetrized_at_field_stage(self):
        # The signed field is left untouched; symmetry is applied
        # downstream in chladni_nodal_density.
        g = chladni_field_pairwise(
            [3, 4, 7], symmetry="d4_max", resolution=32
        )
        assert not np.allclose(
            np.rot90(g.coordinates, 1), g.coordinates, atol=1e-6
        )

    def test_float_modes_accepted_for_animation_interpolation(self):
        g = chladni_field_pairwise([2.4, 3.7, 5.1], resolution=32)
        assert np.isfinite(g.coordinates).all()


# ───────────────────────────────────────────────── chladni_field_triple_anti


class TestChladniFieldTripleAntisymmetric:
    def test_basic_output(self):
        g = chladni_field_triple_antisymmetric([2, 3, 5], resolution=32)
        assert g.geom_type == "field_2d"
        assert g.metadata["scheme"] == "triple_antisymmetric"

    def test_requires_three_modes(self):
        with pytest.raises(ValueError, match="at least 3"):
            chladni_field_triple_antisymmetric([2, 3], resolution=32)

    def test_four_ratios_gives_four_triples(self):
        # C(4, 3) = 4.
        g = chladni_field_triple_antisymmetric(
            [4, 5, 6, 7], resolution=32
        )
        assert len(g.parameters["triples"]) == 4
        assert g.parameters["triples"] == [
            (4.0, 5.0, 6.0), (4.0, 5.0, 7.0),
            (4.0, 6.0, 7.0), (5.0, 6.0, 7.0),
        ]

    def test_int_modes_stored(self):
        g = chladni_field_triple_antisymmetric(
            [4, 5, 6], resolution=32
        )
        assert g.parameters["int_modes"] == [4, 5, 6]

    def test_n_pairs_metadata_is_three_times_n_triples(self):
        g = chladni_field_triple_antisymmetric(
            [4, 5, 6, 7], resolution=32
        )
        assert g.parameters["n_pairs"] == 3 * 4


# ─────────────────────────────────────────────────────── chladni_nodal_density


class TestChladniNodalDensity:
    @staticmethod
    def _field(modes=(4, 5, 6), symmetry="none"):
        return chladni_field_pairwise(
            list(modes), symmetry=symmetry, resolution=64
        )

    def test_output_range_within_unit(self):
        d = chladni_nodal_density(self._field(), sigma=0.05)
        assert d.coordinates.min() >= 0
        assert d.coordinates.max() <= 1.0 + 1e-10

    def test_auto_sigma_from_metadata(self):
        d = chladni_nodal_density(self._field())
        assert d.metadata["nodal_sigma"] > 0

    def test_explicit_sigma_wins(self):
        d = chladni_nodal_density(self._field(), sigma=0.1)
        assert d.metadata["nodal_sigma"] == pytest.approx(0.1)

    def test_antinodal_is_complement(self):
        nodal = chladni_nodal_density(
            self._field(), sigma=0.05, mode="nodal"
        )
        anti = chladni_nodal_density(
            self._field(), sigma=0.05, mode="antinodal"
        )
        np.testing.assert_allclose(
            nodal.coordinates + anti.coordinates,
            np.ones_like(nodal.coordinates),
            atol=1e-10,
        )

    def test_d4_applied_when_metadata_requests(self):
        field = self._field(symmetry="d4_max")
        d = chladni_nodal_density(field, sigma=0.05)
        for k in range(1, 4):
            np.testing.assert_allclose(
                np.rot90(d.coordinates, k),
                d.coordinates,
                atol=1e-10,
            )

    def test_d4_not_applied_when_symmetry_none(self):
        field = self._field(symmetry="none")
        d = chladni_nodal_density(field, sigma=0.05)
        assert not np.allclose(
            np.rot90(d.coordinates, 1), d.coordinates, atol=1e-6
        )

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            chladni_nodal_density(self._field(), sigma=0.05, mode="bogus")

    def test_invalid_sigma(self):
        with pytest.raises(ValueError, match="sigma"):
            chladni_nodal_density(self._field(), sigma=-0.1)

    def test_wrong_geom_type_raises(self):
        from biotuner.harmonic_geometry.geometry_data import GeometryData
        bad = GeometryData(
            geom_type="point_cloud_2d", coordinates=np.zeros((10, 2))
        )
        with pytest.raises(ValueError, match="field_2d"):
            chladni_nodal_density(bad, sigma=0.05)


# ─────────────────────────────────────────────────────── RigidPlate extensions


class TestRigidPlateCymatics:
    def test_default_is_per_ratio_classical(self):
        rp = RigidPlate()
        assert rp.mode_scheme == "per_ratio"
        chord = HarmonicInput(
            ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)]
        )
        g = rp(chord)
        assert g.metadata["kind"] == "chladni_field_rectangular"

    def test_pairwise_antisymmetric_routes_to_new_path(self):
        rp = RigidPlate(mode_scheme="pairwise_antisymmetric", resolution=64)
        chord = HarmonicInput(
            ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)]
        )
        g = rp(chord)
        assert g.metadata["kind"] == "chladni_field_pairwise"
        assert g.metadata["scheme"] == "pairwise_antisymmetric"

    def test_pairwise_symmetric(self):
        rp = RigidPlate(mode_scheme="pairwise_symmetric", resolution=64)
        chord = HarmonicInput(
            ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)]
        )
        g = rp(chord)
        assert g.metadata["scheme"] == "pairwise_symmetric"

    def test_triple_antisymmetric(self):
        rp = RigidPlate(mode_scheme="triple_antisymmetric", resolution=64)
        chord = HarmonicInput(
            ratios=[
                Fraction(1), Fraction(5, 4),
                Fraction(3, 2), Fraction(7, 4),
            ]
        )
        g = rp(chord)
        assert g.metadata["scheme"] == "triple_antisymmetric"

    def test_triple_requires_three_ratios(self):
        rp = RigidPlate(mode_scheme="triple_antisymmetric", resolution=64)
        chord_short = HarmonicInput(
            ratios=[Fraction(1), Fraction(3, 2)]
        )
        with pytest.raises(ValueError, match="at least 3"):
            rp(chord_short)

    def test_cymatics_scheme_requires_rectangular_domain(self):
        with pytest.raises(ValueError, match="Rectangular"):
            RigidPlate(
                domain=Circular(R=1.0),
                mode_scheme="pairwise_antisymmetric",
            )

    def test_d4_requires_square_domain(self):
        with pytest.raises(ValueError, match="square"):
            RigidPlate(
                domain=Rectangular(Lx=2.0, Ly=1.0),
                mode_scheme="pairwise_antisymmetric",
                symmetry="d4_max",
            )

    def test_output_nodal_density_returns_density(self):
        rp = RigidPlate(
            mode_scheme="pairwise_antisymmetric",
            output="nodal_density",
            sigma=0.05,
            resolution=64,
        )
        chord = HarmonicInput(
            ratios=[Fraction(4), Fraction(5), Fraction(6)]
        )
        g = rp(chord)
        assert g.coordinates.min() >= 0
        assert g.coordinates.max() <= 1.0 + 1e-10

    def test_output_antinodal_density(self):
        rp = RigidPlate(
            mode_scheme="pairwise_antisymmetric",
            output="antinodal_density",
            sigma=0.05,
            resolution=64,
        )
        chord = HarmonicInput(
            ratios=[Fraction(4), Fraction(5), Fraction(6)]
        )
        g = rp(chord)
        assert g.coordinates.min() >= 0
        assert g.coordinates.max() <= 1.0 + 1e-10

    def test_invalid_mode_scheme(self):
        with pytest.raises(ValueError, match="mode_scheme"):
            RigidPlate(mode_scheme="bogus")

    def test_invalid_symmetry(self):
        with pytest.raises(ValueError, match="symmetry"):
            RigidPlate(symmetry="bogus")

    def test_invalid_output(self):
        with pytest.raises(ValueError, match="output"):
            RigidPlate(output="bogus")

    def test_invalid_sigma(self):
        with pytest.raises(ValueError, match="sigma"):
            RigidPlate(sigma=-0.1)


# ─────────────────────────────────────────────────────── Granular extensions


class TestGranularNodalEmphasis:
    def test_defaults_preserve_classical(self):
        gr = Granular()
        assert gr.nodal_emphasis is False
        assert gr.sigma is None

    def test_nodal_emphasis_density(self):
        field = chladni_field_pairwise([4, 5, 6], resolution=64)
        gr = Granular(
            output_mode="density", nodal_emphasis=True, sigma=0.05
        )
        d = gr(field)
        assert d.geom_type == "field_2d"
        s = d.coordinates[np.isfinite(d.coordinates)].sum()
        assert s == pytest.approx(1.0, abs=1e-6)

    def test_nodal_emphasis_auto_sigma_from_metadata(self):
        field = chladni_field_pairwise([4, 5, 6], resolution=64)
        gr = Granular(output_mode="density", nodal_emphasis=True)
        d = gr(field)
        assert d.parameters["sigma"] is not None
        assert d.parameters["sigma"] > 0

    def test_nodal_emphasis_d4_from_upstream_metadata(self):
        field = chladni_field_pairwise(
            [4, 5, 6], symmetry="d4_max", resolution=64
        )
        gr = Granular(
            output_mode="density", nodal_emphasis=True, sigma=0.05
        )
        d = gr(field)
        for k in range(1, 4):
            np.testing.assert_allclose(
                np.rot90(d.coordinates, k),
                d.coordinates,
                atol=1e-10,
            )

    def test_pipeline_composition_to_particles(self):
        plate = RigidPlate(
            mode_scheme="pairwise_antisymmetric",
            symmetry="d4_max",
            resolution=64,
        )
        sand = Granular(
            output_mode="particles",
            nodal_emphasis=True,
            sigma=0.05,
            n_particles=500,
            seed=0,
        )
        chord = HarmonicInput(
            ratios=[Fraction(4), Fraction(5), Fraction(6)]
        )
        result = Pipeline(plate, sand)(chord)
        assert result.geom_type == "point_cloud_2d"
        assert result.coordinates.shape == (500, 2)

    def test_invalid_sigma_at_construction(self):
        with pytest.raises(ValueError, match="sigma"):
            Granular(sigma=-0.1)


# ─────────────────────────────────────────────────────────────────── plotting


class TestPlottingCymatics:
    @staticmethod
    def _field():
        return chladni_field_pairwise(
            [4, 5, 6], symmetry="d4_max", resolution=64
        )

    def test_draw_chladni_sand_signed_field(self):
        fig, ax = plt.subplots()
        try:
            art = draw_chladni_sand(self._field(), ax, n_particles=500)
            assert art is not None
        finally:
            plt.close(fig)

    def test_draw_chladni_sand_density_input(self):
        d = chladni_nodal_density(self._field(), sigma=0.05)
        fig, ax = plt.subplots()
        try:
            draw_chladni_sand(d, ax, n_particles=500)
        finally:
            plt.close(fig)

    def test_draw_chladni_painted_nodal(self):
        fig, ax = plt.subplots()
        try:
            art = draw_chladni_painted(self._field(), ax, style="nodal")
            assert art is not None
        finally:
            plt.close(fig)

    def test_draw_chladni_painted_envelope(self):
        fig, ax = plt.subplots()
        try:
            draw_chladni_painted(self._field(), ax, style="envelope")
        finally:
            plt.close(fig)

    def test_draw_chladni_painted_invalid_style(self):
        fig, ax = plt.subplots()
        try:
            with pytest.raises(ValueError, match="style"):
                draw_chladni_painted(self._field(), ax, style="invalid")
        finally:
            plt.close(fig)

    def test_animate_chord_sequence_builds_without_save(self):
        chords = [[2, 3, 5], [3, 4, 7]]

        def builder(chord):
            return chladni_field_pairwise(list(chord), resolution=32)

        anim = animate_chord_sequence(
            chords, builder,
            frames_per_segment=2, fps=12,
            figsize=(2, 2),
        )
        assert anim is not None
        plt.close("all")

    def test_animate_chord_sequence_invalid_interp(self):
        with pytest.raises(ValueError, match="interp"):
            animate_chord_sequence(
                [[2, 3, 5], [3, 4, 7]],
                lambda c: chladni_field_pairwise(list(c), resolution=16),
                frames_per_segment=2,
                interp="bogus",
            )
        plt.close("all")

    def test_animate_chord_sequence_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            animate_chord_sequence(
                [[2, 3, 5], [3, 4, 7, 9]],
                lambda c: chladni_field_pairwise(list(c), resolution=16),
                frames_per_segment=2,
            )
        plt.close("all")


# ──────────────────────────────────────────────────── legacy backward-compat


class TestLegacyPerRatioPathPreserved:
    """The legacy classical-Chladni path must keep working exactly as before."""

    def test_default_rigid_plate_routes_to_chladni_from_input(self):
        chord = HarmonicInput(
            ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)]
        )
        g = RigidPlate()(chord)
        assert g.metadata["kind"] == "chladni_field_rectangular"

    def test_explicit_per_ratio_unchanged(self):
        chord = HarmonicInput(
            ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)]
        )
        g1 = RigidPlate()(chord)
        g2 = RigidPlate(mode_scheme="per_ratio")(chord)
        np.testing.assert_allclose(
            g1.coordinates, g2.coordinates, atol=1e-12
        )

    def test_per_ratio_circular_still_works(self):
        chord = HarmonicInput(
            ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)]
        )
        g = RigidPlate(domain=Circular(R=1.0))(chord)
        assert g.metadata["kind"] == "chladni_field_circular"
