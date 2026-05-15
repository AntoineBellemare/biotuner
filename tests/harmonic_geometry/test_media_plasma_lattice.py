"""Tests for biotuner.harmonic_geometry.media.eigenmode.plasma_lattice."""

from fractions import Fraction

import numpy as np
import pytest

from biotuner.harmonic_geometry import HarmonicInput, PlasmaLattice


@pytest.fixture
def major():
    return HarmonicInput(
        ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)],
        amplitudes=[1.0, 0.7, 0.8],
    )


class TestBasic:
    def test_family(self):
        assert PlasmaLattice.family == "eigenmode"

    def test_default_source_is_none(self):
        assert PlasmaLattice().default_source() is None

    def test_point_cloud_output(self, major):
        out = PlasmaLattice(n_ions=10, n_steps=80,
                             chord_resolution=64).respond(major)
        assert out.geom_type == "point_cloud_2d"
        assert out.coordinates.shape == (10, 2)
        assert np.isfinite(out.coordinates).all()

    def test_chord_driven_n_ions(self, major):
        out = PlasmaLattice(n_steps=60, chord_resolution=64).respond(major)
        # Major chord has prime_limit=5 → 6 + 3 * 2.5 = 13.5 → 13.
        n = out.parameters["n_ions"]
        assert 6 <= n <= 32

    def test_min_pair_distance_positive(self, major):
        out = PlasmaLattice(n_ions=8, n_steps=200,
                             chord_resolution=64).respond(major)
        d_min = out.metadata["min_pair_distance"]
        assert d_min > 0.0


class TestPhysics:
    def test_higher_coulomb_spreads_ions(self, major):
        out_lo = PlasmaLattice(n_ions=8, n_steps=300,
                                coulomb_strength=0.005,
                                chord_resolution=64,
                                rng_seed=7).respond(major)
        out_hi = PlasmaLattice(n_ions=8, n_steps=300,
                                coulomb_strength=0.05,
                                chord_resolution=64,
                                rng_seed=7).respond(major)
        # Higher repulsion -> larger mean pairwise distance.
        def mean_pair(pts):
            d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
            np.fill_diagonal(d, np.nan)
            return np.nanmean(d)
        assert mean_pair(out_hi.coordinates) > mean_pair(out_lo.coordinates)


class TestValidation:
    def test_rejects_small_n_ions(self):
        with pytest.raises(ValueError):
            PlasmaLattice(n_ions=1)

    def test_rejects_nonpositive_trap(self):
        with pytest.raises(ValueError):
            PlasmaLattice(trap_radius=0)

    def test_rejects_negative_modulation(self):
        with pytest.raises(ValueError):
            PlasmaLattice(modulation_strength=-0.1)

    def test_rejects_zero_steps(self):
        with pytest.raises(ValueError):
            PlasmaLattice(n_steps=0)

    def test_rejects_unexpected_override(self, major):
        with pytest.raises(TypeError):
            PlasmaLattice(n_ions=6, n_steps=40,
                          chord_resolution=64).respond(major, bogus=5)
