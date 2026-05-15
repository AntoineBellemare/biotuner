"""Tests for biotuner.harmonic_geometry.media.morphogenetic.reaction_diffusion."""

from fractions import Fraction

import numpy as np
import pytest

from biotuner.harmonic_geometry import HarmonicInput, ReactionDiffusion


@pytest.fixture
def major():
    return HarmonicInput(
        ratios=[Fraction(1), Fraction(5, 4), Fraction(3, 2)],
        amplitudes=[1.0, 0.7, 0.8],
    )


class TestBasic:
    def test_family(self):
        assert ReactionDiffusion.family == "morphogenetic"

    def test_v_output_shape(self, major):
        out = ReactionDiffusion(resolution=64, n_steps=300).respond(major)
        assert out.geom_type == "field_2d"
        assert out.coordinates.shape == (64, 64)

    def test_u_output(self, major):
        out = ReactionDiffusion(resolution=64, n_steps=300,
                                output_mode="u").respond(major)
        assert out.coordinates.shape == (64, 64)
        # U starts at ≈1, stays in ~(0, 1] under stable parameters.
        assert out.coordinates.max() <= 1.0 + 1e-3

    def test_difference_output(self, major):
        out = ReactionDiffusion(resolution=64, n_steps=300,
                                output_mode="difference").respond(major)
        assert out.coordinates.shape == (64, 64)


class TestSeeds:
    def test_single_seed(self, major):
        out = ReactionDiffusion(resolution=64, n_steps=300,
                                seed_strategy="single").respond(major)
        assert out.geom_type == "field_2d"

    def test_polygon_seed(self, major):
        out = ReactionDiffusion(resolution=64, n_steps=300,
                                seed_strategy="polygon").respond(major)
        assert out.geom_type == "field_2d"

    def test_random_seed(self, major):
        out = ReactionDiffusion(resolution=64, n_steps=300,
                                seed_strategy="random",
                                rng_seed=42).respond(major)
        assert out.geom_type == "field_2d"


class TestChordCoupling:
    def test_default_feed_in_range(self, major):
        out = ReactionDiffusion(resolution=64, n_steps=50).respond(major)
        F = out.parameters["feed"]
        # Driven by structure.prime_limit (3 → 0.024, 11 → 0.058).
        assert 0.020 <= F <= 0.062 + 1e-9

    def test_default_kill_in_range(self, major):
        out = ReactionDiffusion(resolution=64, n_steps=50).respond(major)
        K = out.parameters["kill"]
        assert 0.045 <= K <= 0.065 + 1e-9

    def test_diffusion_ratio_chord_driven(self, major):
        out = ReactionDiffusion(resolution=64, n_steps=50).respond(major)
        ratio = out.parameters["diffusion_ratio"]
        # Bounded between 1.5 and 3.0 (mci-driven).
        assert 1.5 <= ratio <= 3.0 + 1e-9


class TestValidation:
    def test_rejects_unknown_output_mode(self):
        with pytest.raises(ValueError):
            ReactionDiffusion(output_mode="bogus")

    def test_rejects_unknown_seed_strategy(self):
        with pytest.raises(ValueError):
            ReactionDiffusion(seed_strategy="bogus")

    def test_rejects_bad_feed(self):
        with pytest.raises(ValueError):
            ReactionDiffusion(feed=-0.1)
        with pytest.raises(ValueError):
            ReactionDiffusion(feed=0.5)

    def test_rejects_bad_kill(self):
        with pytest.raises(ValueError):
            ReactionDiffusion(kill=-0.1)

    def test_rejects_small_resolution(self):
        with pytest.raises(ValueError):
            ReactionDiffusion(resolution=16)

    def test_rejects_nonpositive_dt(self):
        with pytest.raises(ValueError):
            ReactionDiffusion(dt=0)

    def test_rejects_unexpected_override(self, major):
        with pytest.raises(TypeError):
            ReactionDiffusion(resolution=32,
                              n_steps=50).respond(major, bogus=5)
