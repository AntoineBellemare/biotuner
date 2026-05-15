"""Tests for biotuner.harmonic_geometry.media.structure."""

from fractions import Fraction

import numpy as np
import pytest

from biotuner.harmonic_geometry import HarmonicInput
from biotuner.harmonic_geometry.media import structure as st


def chord(*ratios, amps=None):
    fr = [Fraction(r).limit_denominator(10000) if not isinstance(r, Fraction)
          else r for r in ratios]
    if amps is None:
        amps = [1.0] * len(fr)
    return HarmonicInput(ratios=fr, amplitudes=amps)


# =========================================================== prime_limit


class TestPrimeLimit:
    def test_pythagorean_is_3_limit(self):
        # Sus4: 1 : 4/3 : 3/2 — primes 2, 3 only.
        assert st.prime_limit(chord(Fraction(1), Fraction(4, 3), Fraction(3, 2))) == 3

    def test_classical_is_5_limit(self):
        # Major triad 1 : 5/4 : 3/2 — uses prime 5.
        assert st.prime_limit(chord(Fraction(1), Fraction(5, 4), Fraction(3, 2))) == 5
        # Minor too.
        assert st.prime_limit(chord(Fraction(1), Fraction(6, 5), Fraction(3, 2))) == 5

    def test_septimal(self):
        # 1 : 6/5 : 7/5 — prime 7 from the 7/5.
        assert st.prime_limit(chord(Fraction(1), Fraction(6, 5), Fraction(7, 5))) == 7

    def test_undecimal(self):
        # 5:7:11 normalized → 1, 7/5, 11/5 — prime 11.
        assert st.prime_limit(chord(Fraction(1), Fraction(7, 5), Fraction(11, 5))) == 11

    def test_pure_unison_is_2_limit(self):
        # Single ratio 1/1 → no primes > 2.
        assert st.prime_limit(chord(Fraction(1))) == 2


# ======================================================= tonnetz_polygon


class TestTonnetzPolygon:
    def test_unison_at_origin(self):
        poly = st.tonnetz_polygon(chord(Fraction(1)))
        assert poly == [(0.0, 0.0)]

    def test_major_and_minor_are_mirrored(self):
        # Major 1, 5/4, 3/2 → 5/4 at +y; minor 1, 6/5, 3/2 → 6/5 at -y.
        major_poly = st.tonnetz_polygon(
            chord(Fraction(1), Fraction(5, 4), Fraction(3, 2)))
        minor_poly = st.tonnetz_polygon(
            chord(Fraction(1), Fraction(6, 5), Fraction(3, 2)))
        # Y component of the third ratio's projection should be opposite-signed.
        assert major_poly[1][1] > 0
        assert minor_poly[1][1] < 0
        # The fifth ratio (3/2) is on the x-axis in both.
        assert major_poly[2] == pytest.approx((1.0, 0.0), abs=1e-6)
        assert minor_poly[2] == pytest.approx((1.0, 0.0), abs=1e-6)

    def test_pythagorean_chord_is_collinear(self):
        # 1, 4/3, 3/2 — all on the x-axis (only prime 3 contributes).
        poly = st.tonnetz_polygon(
            chord(Fraction(1), Fraction(4, 3), Fraction(3, 2)))
        assert all(abs(y) < 1e-9 for _, y in poly)


# ============================================== pairwise_harmonic_distance


class TestPairwiseHD:
    def test_diagonal_zero(self):
        hd = st.pairwise_harmonic_distance(
            chord(Fraction(1), Fraction(5, 4), Fraction(3, 2)))
        for i in range(hd.shape[0]):
            assert hd[i, i] == 0.0

    def test_symmetry(self):
        # Tenney HD is undirected; matrix should be symmetric.
        hd = st.pairwise_harmonic_distance(
            chord(Fraction(1), Fraction(5, 4), Fraction(3, 2)))
        assert np.allclose(hd, hd.T)

    def test_known_intervals(self):
        # 3/2 between (1) and (3/2): HD = log2(3·2) = log2(6).
        hd = st.pairwise_harmonic_distance(chord(Fraction(1), Fraction(3, 2)))
        assert hd[0, 1] == pytest.approx(np.log2(6.0), abs=1e-9)


# ============================================================= cf_depths


class TestCFDepths:
    def test_simple_ratios(self):
        # 3/2 has CF [1; 2] → depth 2.
        depths = st.cf_depths(chord(Fraction(3, 2)))
        assert depths == [2]

    def test_unison_is_depth_one(self):
        # 1/1 has CF [1] → depth 1.
        depths = st.cf_depths(chord(Fraction(1)))
        assert depths == [1]


# ========================================================= max_common_int


class TestMaxCommonInt:
    def test_major_is_six(self):
        # 1, 5/4, 3/2 → 4:5:6, max=6.
        assert st.max_common_int(
            chord(Fraction(1), Fraction(5, 4), Fraction(3, 2))) == 6

    def test_minor_is_fifteen(self):
        # 1, 6/5, 3/2 → 10:12:15, max=15.
        assert st.max_common_int(
            chord(Fraction(1), Fraction(6, 5), Fraction(3, 2))) == 15

    def test_sus4_is_nine(self):
        # 1, 4/3, 3/2 → 6:8:9, max=9.
        assert st.max_common_int(
            chord(Fraction(1), Fraction(4, 3), Fraction(3, 2))) == 9

    def test_septimal(self):
        # 1, 6/5, 7/5 → 5:6:7, max=7.
        assert st.max_common_int(
            chord(Fraction(1), Fraction(6, 5), Fraction(7, 5))) == 7


# ==================================================== pc_rotation_order


class TestPCRotationOrder:
    def test_major_no_symmetry(self):
        # {0, 4, 7} has no rotational symmetry → 1-fold.
        assert st.pc_rotation_order(
            chord(Fraction(1), Fraction(5, 4), Fraction(3, 2))) == 1

    def test_diminished_seventh_is_four_fold(self):
        # {0, 3, 6, 9} repeats every minor third → 4-fold.
        # 12-TET ratios: 1, 2^(3/12), 2^(6/12), 2^(9/12).
        dim7 = chord(
            Fraction(1),
            Fraction(2 ** (3 / 12)).limit_denominator(1000),
            Fraction(2 ** (6 / 12)).limit_denominator(1000),
            Fraction(2 ** (9 / 12)).limit_denominator(1000),
        )
        assert st.pc_rotation_order(dim7) == 4

    def test_augmented_is_three_fold(self):
        # {0, 4, 8} → 3-fold.
        aug = chord(
            Fraction(1),
            Fraction(2 ** (4 / 12)).limit_denominator(1000),
            Fraction(2 ** (8 / 12)).limit_denominator(1000),
        )
        assert st.pc_rotation_order(aug) == 3


# ============================================== per_ratio_consonance_weights


class TestPerRatioWeights:
    def test_mean_one(self):
        w = st.per_ratio_consonance_weights(
            chord(Fraction(1), Fraction(5, 4), Fraction(3, 2)))
        assert w.mean() == pytest.approx(1.0, abs=1e-9)

    def test_single_ratio_returns_one(self):
        w = st.per_ratio_consonance_weights(chord(Fraction(1)))
        assert w.shape == (1,)
        assert w[0] == 1.0

    def test_consonant_ratio_has_higher_weight(self):
        # In (1, 5/4, 3/2), the 3/2 ratio is more consonant than 5/4
        # to either of the others, so should weight higher.
        w = st.per_ratio_consonance_weights(
            chord(Fraction(1), Fraction(5, 4), Fraction(3, 2)))
        # 1 (root) and 3/2 (fifth) form a 3:2; both are consonant to one
        # another. The 5/4 is the most "outlying" by Tenney HD.
        assert w[1] < w[0] or w[1] < w[2]


# ============================================================ signature


class TestChordSignature:
    def test_signature_bundles_all(self):
        sig = st.chord_signature(
            chord(Fraction(1), Fraction(5, 4), Fraction(3, 2)))
        assert sig.prime_limit == 5
        assert sig.n_ratios == 3
        assert len(sig.tonnetz_polygon) == 3
        assert sig.pairwise_harmonic_distance.shape == (3, 3)
        assert len(sig.cf_depths) == 3
        assert sig.max_common_int == 6
        assert sig.pc_rotation_order == 1
