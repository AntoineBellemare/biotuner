"""Tests for :mod:`biotuner.mos.timbre` -- Dynamic Tonality (Milne et al. §6).

The interesting tests here are the last two groups.  ``TestHeadlineClaim``
checks that a lattice-matched spectrum really is less rough than a harmonic one
against the same scale, and ``TestBetaBudgetFinding`` pins down the conditions
under which it is *not* -- which turned out to be the library's own defaults.
Those negative results are asserted rather than tolerated, so that if the
mapping ever changes the suite says which direction it moved.
"""

import dataclasses
import math

import numpy as np
import pytest

from biotuner.mos.scale import MOSScale
from biotuner.mos.timbre import (
    PartialMap,
    SimpleTimbre,
    dissonance_advantage,
    dynamic_timbre,
    map_harmonic,
    matched_partials,
    matched_ratios,
    matched_spectrum,
    scale_dissonance,
    spectral_dissonance,
)

# Every co-prime signature up to 7+5 notes, used for the population-level scans.
ALL_SIGNATURES = [
    (1, 3), (3, 1), (1, 4), (4, 1), (1, 5), (5, 1),
    (2, 3), (3, 2), (2, 5), (5, 2), (2, 7), (7, 2),
    (3, 4), (4, 3), (3, 5), (5, 3), (3, 7), (7, 3),
    (4, 5), (5, 4), (4, 7), (7, 4),
    (5, 6), (6, 5), (5, 7), (7, 5),
]

#: 31-EDO meantone diatonic -- the reference tuning for the mapping tests.
MEANTONE = MOSScale.from_signature(5, 2, tuning=31)


# --------------------------------------------------------------------------- #
# Claims 1 and 2: the lattice origin and the period
# --------------------------------------------------------------------------- #
class TestLatticeOrigin:
    @pytest.mark.parametrize(
        "scale",
        [
            MOSScale.from_signature(5, 2, tuning=31),
            MOSScale.from_signature(5, 2, tuning=12),  # rational: ties exist
            MOSScale.from_signature(5, 2),  # noble
            MOSScale.from_signature(7, 4),
            MOSScale.from_signature(2, 5),
            MOSScale.from_signature(5, 2, tuning=31, period=3.0),
        ],
        ids=["31edo", "12edo", "noble", "7L4s", "2L5s", "tritave"],
    )
    def test_harmonic_one_is_the_origin(self, scale):
        """Claim 1: h=1 maps to (0, 0) with zero error."""
        p = map_harmonic(1, scale)
        assert (p.alpha, p.beta) == (0, 0)
        assert p.error_cents == 0.0
        assert p.ratio == 1.0
        assert p.just_ratio == 1.0

    def test_origin_survives_an_exact_tie(self):
        """In 12-EDO, beta=12 also gives zero error; the short chain must win.

        g = 7/12, so 12 generators are exactly 7 periods and (alpha, beta) =
        (-7, 12) is bit-for-bit as accurate as (0, 0).  Ties like this are the
        only reason the search order matters.
        """
        s = MOSScale.from_signature(5, 2, tuning=12)
        tied = 0 + 12 * s.generator - 0.0
        assert tied == pytest.approx(7.0, abs=1e-15)  # the tie is real
        assert (map_harmonic(1, s).alpha, map_harmonic(1, s).beta) == (0, 0)
        assert (map_harmonic(2, s).alpha, map_harmonic(2, s).beta) == (1, 0)

    @pytest.mark.parametrize(
        "scale",
        [
            MOSScale.from_signature(5, 2, tuning=31),
            MOSScale.from_signature(5, 2, tuning=12),
            MOSScale.from_signature(5, 2),
            MOSScale.from_signature(7, 4),
            MOSScale.from_signature(3, 4),
            MOSScale.from_signature(1, 4),
        ],
    )
    def test_octave_is_the_period(self, scale):
        """Claim 2: with period == 2, harmonic 2 is (1, 0) exactly."""
        assert scale.period == 2.0
        p = map_harmonic(2, scale)
        assert (p.alpha, p.beta) == (1, 0)
        assert p.error_cents == 0.0
        assert p.ratio == pytest.approx(2.0)

    def test_period_not_octave(self):
        """The claim is about the *period*, not the octave.

        Under a tritave period, harmonic 3 is the free one and harmonic 2 has
        to be bought with generators.
        """
        s = MOSScale.from_signature(5, 2, tuning=31, period=3.0)
        p3 = map_harmonic(3, s)
        assert (p3.alpha, p3.beta) == (1, 0)
        assert p3.error_cents == 0.0
        assert map_harmonic(2, s).beta != 0

    @pytest.mark.parametrize("h", [2, 4, 8, 16])
    def test_powers_of_two_are_exact_under_an_octave_period(self, h):
        p = map_harmonic(h, MEANTONE)
        assert (p.alpha, p.beta) == (int(math.log2(h)), 0)
        assert p.error_cents == 0.0


# --------------------------------------------------------------------------- #
# Claim 3: meantone means five is four fifths
# --------------------------------------------------------------------------- #
class TestMeantoneMapping:
    def test_harmonic_three_is_period_plus_generator(self):
        """Claim 3a: (alpha, beta) == (1, 1) -- a twelfth is an octave + a fifth.

        The sign convention comes out positive: alpha counts periods up, beta
        counts generators up, no reflection anywhere.
        """
        p = map_harmonic(3, MEANTONE)
        assert (p.alpha, p.beta) == (1, 1)
        # 31-EDO's fifth is 5.18 cents flat of just, and that is the whole error.
        assert p.error_cents == pytest.approx(-5.1808, abs=1e-3)
        assert p.error_cents == pytest.approx(
            MEANTONE.generator_cents - 1200.0 * math.log2(1.5), abs=1e-9
        )

    def test_harmonic_five_is_four_generators(self):
        """Claim 3b: beta == 4 -- the definition of meantone."""
        p = map_harmonic(5, MEANTONE)
        assert p.beta == 4
        assert p.alpha == 0  # four fifths already exceed two octaves
        assert p.error_cents == pytest.approx(0.7831, abs=1e-3)
        # 4 generators, no octave reduction needed to reach the 5th harmonic.
        assert 4 * MEANTONE.generator_cents == pytest.approx(
            1200.0 * math.log2(5) + p.error_cents, abs=1e-9
        )

    def test_full_meantone_table(self):
        expected = [
            (0, 0), (1, 0), (1, 1), (2, 0), (0, 4), (2, 1),
            (-3, 10), (3, 0), (2, 2), (1, 4), (11, -13), (3, 1),
        ]
        assert [(p.alpha, p.beta) for p in matched_partials(MEANTONE, 12)] == expected

    def test_eleventh_partial_takes_the_short_side_of_an_exact_tie(self):
        """h=11 has two bit-identical spellings; the 13-chain must beat the 18.

        In 31-EDO a lattice point is ``alpha * 31 + beta * 18`` steps, so
        ``(-7, 18)`` and ``(11, -13)`` are both 107 steps -- they differ by
        exactly one full 31-generator circle.  Both fit inside the default
        ``max_beta=24``, so the tie is reachable and the shorter chain wins.
        """
        p = map_harmonic(11, MEANTONE)
        assert (p.alpha, p.beta) == (11, -13)
        assert -7 * 31 + 18 * 18 == 11 * 31 - 13 * 18 == 107  # the tie is exact
        assert p.error_cents == pytest.approx(
            107 * 1200 / 31 - 1200 * math.log2(11), abs=1e-9
        )

    @pytest.mark.parametrize(
        "tuning,max_expected",
        [("equalized", 3), (7, 3), (12, 6), (19, 9), (22, 9), (31, 13)],
    )
    def test_rational_generators_never_spell_the_long_side_of_a_tie(
        self, tuning, max_expected
    ):
        """Regression: float noise used to hand rational ties to the *longest* chain.

        At a generator of ``p / q`` the chains ``beta`` and ``beta +- q`` are
        mathematically identical, but ``beta * g`` is evaluated in binary
        floating point and drifts by ~1e-12 cents over a 21-generator chain.
        A fixed 1e-12 improvement threshold read that as a real gain, so 7-EDO
        spelled the third harmonic as a 20-generator chain instead of a single
        fifth.  Every one of these tunings regressed; none may again.
        """
        scale = MOSScale.from_signature(5, 2, tuning=tuning)
        maps = matched_partials(scale, 12)
        assert max(abs(p.beta) for p in maps) == max_expected

        # No partial may use a chain when an equally-accurate shorter one exists.
        for p in maps:
            for shorter in range(-abs(p.beta) + 1, abs(p.beta)):
                alt_alpha = int(
                    math.floor(math.log2(p.harmonic) - shorter * scale.generator + 0.5)
                )
                alt = 2.0**alt_alpha * scale.generator_ratio**shorter
                if abs(1200.0 * math.log2(alt / p.harmonic)) < abs(p.error_cents) - 1e-6:
                    raise AssertionError(
                        f"h={p.harmonic}: beta={p.beta} beaten by beta={shorter}"
                    )

    @pytest.mark.parametrize("max_beta", [24, 100, 1000])
    def test_short_chain_survives_an_absurd_beta_budget(self, max_beta):
        """A bigger budget offers more tied spellings; none may displace the fifth.

        At g = 4/7 every ``beta + 7k`` names the same pitch, so raising
        ``max_beta`` from 24 to 1000 adds ~140 more chains all bit-identical to
        ``beta = 1``.  The accumulated rounding noise grows with the budget, so
        this is where a too-tight tolerance shows up.
        """
        seven = MOSScale.from_signature(5, 2, tuning="equalized")
        assert map_harmonic(3, seven, max_beta=max_beta).beta == 1
        assert max(
            abs(map_harmonic(h, seven, max_beta=max_beta).beta) for h in range(1, 13)
        ) == 3

    def test_seven_edo_third_harmonic_is_one_fifth_not_twenty(self):
        """The concrete case that used to fail, spelled out."""
        seven = MOSScale.from_signature(5, 2, tuning="equalized")
        assert seven.generator == pytest.approx(4 / 7, abs=1e-15)
        p = map_harmonic(3, seven)
        assert (p.alpha, p.beta) == (1, 1)
        # (13, -20) is the bit-identical long spelling the bug used to return.
        assert 1 * 7 + 1 * 4 == 13 * 7 - 20 * 4 == 11  # both are 11 steps of 7-EDO
        assert p.error_cents == pytest.approx(
            11 * 1200 / 7 - 1200 * math.log2(3), abs=1e-9
        )

    def test_ratio_is_the_lattice_point(self):
        for p in matched_partials(MEANTONE, 12):
            assert p.ratio == pytest.approx(
                MEANTONE.period**p.alpha * MEANTONE.generator_ratio**p.beta
            )
            assert p.error_cents == pytest.approx(
                1200.0 * math.log2(p.ratio / p.just_ratio)
            )

    def test_bounding_the_chain_changes_the_seventh(self):
        """max_beta=5 cannot afford the 10-generator 7/4; it settles for -2."""
        got = [(p.alpha, p.beta) for p in matched_partials(MEANTONE, 8, max_beta=5)]
        assert got[6] == (4, -2)
        assert abs(map_harmonic(7, MEANTONE, max_beta=5).error_cents) > abs(
            map_harmonic(7, MEANTONE).error_cents
        )


class TestBetaPenalty:
    def test_penalty_shortens_the_chain(self):
        noble = MOSScale.from_signature(5, 2)
        assert map_harmonic(5, noble).beta == 21
        assert map_harmonic(5, noble, beta_penalty=3.0).beta == 4

    def test_penalty_trades_accuracy_for_simplicity(self):
        noble = MOSScale.from_signature(5, 2)
        loose = map_harmonic(5, noble)
        tight = map_harmonic(5, noble, beta_penalty=3.0)
        assert abs(tight.beta) < abs(loose.beta)
        assert abs(tight.error_cents) > abs(loose.error_cents)

    @pytest.mark.parametrize("penalty", [0.0, 1.0, 3.0, 10.0, 50.0])
    def test_penalty_is_monotone_in_chain_length(self, penalty):
        """A larger penalty never lengthens the chain of any partial."""
        noble = MOSScale.from_signature(5, 2)
        betas = [abs(p.beta) for p in matched_partials(noble, 12, beta_penalty=penalty)]
        base = [abs(p.beta) for p in matched_partials(noble, 12, beta_penalty=0.0)]
        assert sum(betas) <= sum(base)

    def test_huge_penalty_collapses_to_periods_only(self):
        noble = MOSScale.from_signature(5, 2)
        assert all(p.beta == 0 for p in matched_partials(noble, 12, beta_penalty=1e6))


# --------------------------------------------------------------------------- #
# Spectra
# --------------------------------------------------------------------------- #
class TestSpectra:
    def test_matched_ratios_values(self):
        got = [round(r, 4) for r in matched_ratios(MEANTONE, 6)]
        assert got == [1.0, 2.0, 2.991, 4.0, 5.0023, 5.9821]

    def test_matched_ratios_track_partial_maps(self):
        assert matched_ratios(MEANTONE, 9) == [
            p.ratio for p in matched_partials(MEANTONE, 9)
        ]

    def test_spectrum_shape_and_defaults(self):
        f, a = matched_spectrum(MEANTONE, fundamental=100.0, n_partials=5)
        assert f.shape == a.shape == (5,)
        assert f[0] == pytest.approx(100.0)
        assert a.max() == pytest.approx(1.0)
        assert list(a) == pytest.approx([1.0, 0.5, 1 / 3, 0.25, 0.2])

    def test_spectrum_scales_with_fundamental(self):
        f1, _ = matched_spectrum(MEANTONE, fundamental=100.0, n_partials=8)
        f2, _ = matched_spectrum(MEANTONE, fundamental=440.0, n_partials=8)
        assert np.allclose(f2 / f1, 4.4)

    def test_custom_amplitudes_are_normalised(self):
        _, a = matched_spectrum(
            MEANTONE, n_partials=4, amplitudes=[0.2, 0.1, 0.05, 0.025]
        )
        assert a.max() == pytest.approx(1.0)
        assert list(a) == pytest.approx([1.0, 0.5, 0.25, 0.125])

    def test_amplitude_length_mismatch_reports_both_numbers(self):
        with pytest.raises(ValueError, match=r"length 3.*n_partials is 5"):
            matched_spectrum(MEANTONE, n_partials=5, amplitudes=[1.0, 0.5, 0.25])

    @pytest.mark.parametrize("bad", [0, -1])
    def test_bad_n_partials(self, bad):
        with pytest.raises(ValueError, match="n_partials must be >= 1"):
            matched_partials(MEANTONE, bad)

    def test_bad_harmonic(self):
        with pytest.raises(ValueError, match="integer >= 1"):
            map_harmonic(0, MEANTONE)

    def test_bad_fundamental(self):
        with pytest.raises(ValueError, match="positive frequency"):
            matched_spectrum(MEANTONE, fundamental=0.0)

    def test_negative_beta_penalty(self):
        with pytest.raises(ValueError, match="cents cost per generator"):
            map_harmonic(3, MEANTONE, beta_penalty=-1.0)


class TestDynamicTimbre:
    def test_fields(self):
        t = dynamic_timbre(MEANTONE, n_partials=6, fundamental=100.0)
        assert [round(float(x), 3) for x in t.partials_hz] == [
            100.0, 200.0, 299.104, 400.0, 500.226, 598.207
        ]
        assert float(np.max(t.amplitudes)) == pytest.approx(1.0)
        assert t.base_freq == 100.0
        assert t.matching_method == "dynamic_tonality"
        assert t.matched_tuning == pytest.approx(list(MEANTONE.ratios))

    def test_metadata_records_the_lattice(self):
        t = dynamic_timbre(MEANTONE, n_partials=6)
        assert t.metadata["signature"] == "5L2s"
        assert t.metadata["lattice"] == [(0, 0), (1, 0), (1, 1), (2, 0), (0, 4), (2, 1)]
        assert t.metadata["max_abs_beta"] == 4
        assert t.metadata["generator_cents"] == pytest.approx(696.7742, abs=1e-3)

    def test_prefers_the_real_timbre_class(self):
        """The subpackage is a soft dependency, but here it should be present."""
        pytest.importorskip("biotuner.harmonic_timbre")
        from biotuner.harmonic_timbre import Timbre

        t = dynamic_timbre(MEANTONE, n_partials=4)
        assert isinstance(t, Timbre)
        assert t.metadata["timbre_class"] == "Timbre"
        t.validate()  # the real class knows how to check itself

    def test_fallback_class_has_the_same_core_fields(self):
        t = SimpleTimbre(
            partials_hz=np.array([100.0, 200.0]),
            amplitudes=np.array([1.0, 0.5]),
            base_freq=100.0,
        )
        assert t.n_partials() == 2
        real = dynamic_timbre(MEANTONE, n_partials=2, fundamental=100.0)
        for fname in ("partials_hz", "amplitudes", "base_freq",
                      "matched_tuning", "matching_method", "metadata"):
            assert hasattr(t, fname) and hasattr(real, fname)


# --------------------------------------------------------------------------- #
# Dissonance plumbing
# --------------------------------------------------------------------------- #
class TestDissonance:
    def test_octave_is_smoother_than_a_semitone(self):
        octave = spectral_dissonance([250.0, 500.0], [1.0, 1.0])
        semitone = spectral_dissonance([250.0, 265.0], [1.0, 1.0])
        assert octave == pytest.approx(0.000809, abs=1e-6)
        assert semitone == pytest.approx(0.8413, abs=1e-3)
        assert semitone > 100 * octave

    def test_single_partial_has_no_pairs(self):
        assert spectral_dissonance([250.0], [1.0]) == 0.0
        assert spectral_dissonance([], []) == 0.0

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            spectral_dissonance([1.0, 2.0], [1.0])

    def test_nonpositive_frequency(self):
        with pytest.raises(ValueError, match="positive"):
            spectral_dissonance([0.0, 100.0], [1.0, 1.0])

    def test_matches_dissmeasure_directly(self):
        from biotuner.scale_construction import dissmeasure

        f = np.array([250.0, 375.0, 500.0, 625.0])
        a = np.array([1.0, 0.6, 0.4, 0.3])
        assert spectral_dissonance(f, a) == pytest.approx(float(dissmeasure(f, a)))

    def test_scale_dissonance_reference_values(self):
        assert scale_dissonance(MEANTONE, matched=False) == pytest.approx(29.8699, abs=1e-3)
        assert scale_dissonance(MEANTONE, matched=True) == pytest.approx(28.327, abs=1e-3)

    def test_scale_dissonance_grows_with_partial_count(self):
        vals = [scale_dissonance(MEANTONE, n_partials=n) for n in (2, 4, 8, 12)]
        assert vals == sorted(vals)

    def test_advantage_dict_is_self_consistent(self):
        adv = dissonance_advantage(MEANTONE)
        assert adv["reduction"] == pytest.approx(adv["harmonic"] - adv["matched"])
        assert adv["reduction_pct"] == pytest.approx(
            100.0 * adv["reduction"] / adv["harmonic"]
        )
        assert adv["reduction"] == pytest.approx(1.5429, abs=1e-3)
        assert adv["reduction_pct"] == pytest.approx(5.165, abs=1e-2)

    def test_advantage_rejects_matched_kwarg(self):
        with pytest.raises(TypeError, match="computes both"):
            dissonance_advantage(MEANTONE, matched=True)

    def test_advantage_forwards_kwargs(self):
        adv = dissonance_advantage(MEANTONE, n_partials=6, fundamental=300.0)
        assert adv["harmonic"] == pytest.approx(
            scale_dissonance(MEANTONE, n_partials=6, matched=False, fundamental=300.0)
        )


# --------------------------------------------------------------------------- #
# Claim 4: the headline
# --------------------------------------------------------------------------- #
class TestHeadlineClaim:
    """Matched partials beat harmonic partials -- when the chain is bounded."""

    # (name, scale, minimum expected % reduction at the library defaults)
    HOLDS_AT_DEFAULTS = [
        ("5L2s @714c", MOSScale.from_signature(5, 2, tuning=0.5952), 4.0),
        ("5L2s @31edo", MOSScale.from_signature(5, 2, tuning=31), 5.0),
        ("5L2s noble", MOSScale.from_signature(5, 2), 0.6),
        ("7L4s noble", MOSScale.from_signature(7, 4), 0.4),
        ("2L3s noble", MOSScale.from_signature(2, 3), 6.0),
        ("3L4s noble", MOSScale.from_signature(3, 4), 2.5),
        ("4L3s noble", MOSScale.from_signature(4, 3), 1.2),
        ("5L3s noble", MOSScale.from_signature(5, 3), 1.0),
        ("3L5s noble", MOSScale.from_signature(3, 5), 0.8),
        ("7L2s noble", MOSScale.from_signature(7, 2), 2.8),
        ("7L5s noble", MOSScale.from_signature(7, 5), 3.0),
        ("4L5s noble", MOSScale.from_signature(4, 5), 2.0),
    ]

    @pytest.mark.parametrize(
        "scale,floor_pct",
        [(s, p) for _, s, p in HOLDS_AT_DEFAULTS],
        ids=[n for n, _, _ in HOLDS_AT_DEFAULTS],
    )
    def test_reduction_is_positive_at_defaults(self, scale, floor_pct):
        adv = dissonance_advantage(scale)
        assert adv["reduction"] > 0.0
        assert adv["reduction_pct"] >= floor_pct

    def test_far_from_12edo_the_advantage_is_large(self):
        """A 714-cent generator is 14 cents off anything 12-EDO can spell."""
        far = MOSScale.from_signature(5, 2, tuning=0.5952)
        assert far.generator_cents == pytest.approx(714.24, abs=0.01)
        assert dissonance_advantage(far)["reduction_pct"] > 4.0
        assert dissonance_advantage(far, max_beta=5)["reduction_pct"] > 19.0

    @pytest.mark.parametrize("n_large,n_small", ALL_SIGNATURES)
    def test_bounded_chain_always_wins(self, n_large, n_small):
        """Every one of 26 signatures improves, by at least 3 %, at max_beta=5."""
        scale = MOSScale.from_signature(n_large, n_small)
        adv = dissonance_advantage(scale, max_beta=5)
        assert adv["reduction"] > 0.0, f"{n_large}L{n_small}s got {adv}"
        assert adv["reduction_pct"] > 3.0

    @pytest.mark.parametrize("max_beta", [3, 5, 8])
    def test_bounded_chain_wins_across_the_5L2s_tuning_range(self, max_beta):
        """Not just at one tuning -- everywhere the diatonic can be tuned."""
        lo, hi = 0.5715, 0.5999
        for i in range(15):
            g = lo + (hi - lo) * i / 14
            scale = MOSScale.from_signature(5, 2, tuning=g)
            adv = dissonance_advantage(scale, max_beta=max_beta)
            assert adv["reduction"] > 0.0, f"g={g:.5f} max_beta={max_beta}: {adv}"


class TestBetaBudgetFinding:
    """The honest half of claim 4: where the effect disappears, and why.

    At ``max_beta=24, beta_penalty=0`` the optimiser has enough generators to
    approximate any just harmonic to within roughly a cent, so the "matched"
    spectrum is a harmonic spectrum in disguise, no partials coincide with
    scale intervals, and the residual difference is sign-indifferent noise.
    Six of 26 signatures land on the wrong side of zero.  These are asserted
    as measured facts, not tolerated as flakiness.
    """

    KNOWN_NEGATIVE_AT_DEFAULTS = {"1L3s", "1L4s", "4L1s", "2L5s", "2L7s", "4L7s"}

    def test_which_signatures_fail_at_the_defaults(self):
        negatives = {}
        for n_large, n_small in ALL_SIGNATURES:
            scale = MOSScale.from_signature(n_large, n_small)
            adv = dissonance_advantage(scale)
            if adv["reduction"] <= 0.0:
                negatives[scale.signature] = adv["reduction_pct"]
        assert set(negatives) == self.KNOWN_NEGATIVE_AT_DEFAULTS, negatives
        # All of them are small -- the matched timbre is not badly worse, it is
        # simply indistinguishable from the harmonic one.
        assert all(pct > -1.5 for pct in negatives.values()), negatives

    @pytest.mark.parametrize("signature", sorted(KNOWN_NEGATIVE_AT_DEFAULTS))
    def test_bounding_the_chain_rescues_every_failure(self, signature):
        n_large, n_small = (int(x) for x in signature.rstrip("s").split("L"))
        scale = MOSScale.from_signature(n_large, n_small)
        assert dissonance_advantage(scale)["reduction"] <= 0.0
        rescued = dissonance_advantage(scale, max_beta=5)
        assert rescued["reduction_pct"] > 6.0, (signature, rescued)

    def test_advantage_shrinks_as_the_budget_grows(self):
        """The whole finding in one scale: more beta, less Dynamic Tonality."""
        scale = MOSScale.from_signature(5, 2, tuning=0.5952)
        pcts = [
            dissonance_advantage(scale, max_beta=b)["reduction_pct"]
            for b in (3, 5, 8, 12, 24)
        ]
        assert pcts == sorted(pcts, reverse=True), pcts
        assert pcts[0] > 20.0 and pcts[-1] < 6.0

    def test_large_budget_makes_matched_partials_nearly_just(self):
        """Why it happens: at max_beta=24 the mapping is a harmonic series."""
        scale = MOSScale.from_signature(5, 2)
        loose = matched_partials(scale, 12, max_beta=24)
        tight = matched_partials(scale, 12, max_beta=5)
        assert max(abs(p.error_cents) for p in loose) < 8.0
        assert max(abs(p.error_cents) for p in tight) > 25.0
        assert max(abs(p.beta) for p in loose) >= 20

    def test_penalty_is_less_reliable_than_bounding(self):
        """beta_penalty=3.0 fixes five of the six but breaks 3L2s instead."""
        bad = {}
        for n_large, n_small in ALL_SIGNATURES:
            scale = MOSScale.from_signature(n_large, n_small)
            adv = dissonance_advantage(scale, beta_penalty=3.0)
            if adv["reduction"] <= 0.0:
                bad[scale.signature] = adv["reduction_pct"]
        assert set(bad) == {"3L2s"}, bad
        assert bad["3L2s"] == pytest.approx(-1.02, abs=0.1)


# --------------------------------------------------------------------------- #
# Claim 5: the mapping is stable as the generator slides
# --------------------------------------------------------------------------- #
class TestGeneratorSweep:
    LO, HI = 0.5715, 0.5999  # the 5L2s valid range, kept off the endpoints

    def _sweep(self, steps, **kw):
        for i in range(steps + 1):
            g = self.LO + (self.HI - self.LO) * i / steps
            yield g, MOSScale.from_signature(5, 2, tuning=g)

    def test_bounded_mapping_never_switches(self):
        """With max_beta=5 the integer mapping is constant over the whole range.

        That is the strong form of the Dynamic Tonality promise: one spectrum
        follows the generator continuously from 685 to 720 cents, with no
        re-spelling of any partial along the way.
        """
        seen = set()
        for _, scale in self._sweep(200):
            seen.add(tuple((p.alpha, p.beta) for p in matched_partials(scale, 8, max_beta=5)))
        assert len(seen) == 1, seen

    def test_ratios_move_continuously(self):
        """No partial jumps: adjacent sweep points differ by a fraction of a cent."""
        prev = None
        worst = 0.0
        for _, scale in self._sweep(400):
            cur = matched_ratios(scale, 8, max_beta=5)
            if prev is not None:
                worst = max(
                    worst,
                    max(abs(1200.0 * math.log2(a / b)) for a, b in zip(cur, prev)),
                )
            prev = cur
        # The generator itself moves 0.085 cents per step; a beta of at most 5
        # can amplify that by 5, so anything under a cent is continuous motion.
        assert worst < 1.0, worst

    def test_ratios_are_monotone_in_the_generator(self):
        """Each partial's ratio moves in one direction, set by the sign of beta."""
        gens, ratios = [], []
        for g, scale in self._sweep(60):
            gens.append(g)
            ratios.append(matched_ratios(scale, 8, max_beta=5))
        betas = [p.beta for p in matched_partials(
            MOSScale.from_signature(5, 2, tuning=(self.LO + self.HI) / 2), 8, max_beta=5
        )]
        for k, beta in enumerate(betas):
            series = [r[k] for r in ratios]
            if beta > 0:
                assert series == sorted(series), (k, beta)
            elif beta < 0:
                assert series == sorted(series, reverse=True), (k, beta)
            else:
                assert max(series) == pytest.approx(min(series))

    def test_unbounded_mapping_switches_often(self):
        """The contrast: at the defaults the spelling churns as the generator moves."""
        seen = set()
        for _, scale in self._sweep(200):
            seen.add(tuple((p.alpha, p.beta) for p in matched_partials(scale, 8)))
        assert len(seen) > 20, len(seen)

    def test_advantage_is_finite_and_bounded_across_the_sweep(self):
        for g, scale in self._sweep(20):
            adv = dissonance_advantage(scale, max_beta=5)
            assert np.isfinite(adv["harmonic"]) and np.isfinite(adv["matched"])
            assert 0.0 < adv["reduction_pct"] < 100.0, (g, adv)


def test_partial_map_is_frozen_and_has_a_cents_view():
    p = map_harmonic(3, MEANTONE)
    assert isinstance(p, PartialMap)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.alpha = 5
    assert p.cents == pytest.approx(1200.0 * math.log2(3) + p.error_cents)
