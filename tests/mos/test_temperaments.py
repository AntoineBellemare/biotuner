"""Tests for :mod:`biotuner.mos.temperaments`.

The point of the module is that its numbers are derived, so most of these tests
are structural identities that must hold in exact integer arithmetic, plus a
comparison of the computed generators against published tunings.
"""

import math
from fractions import Fraction
from itertools import combinations, product

import pytest

from biotuner.mos.temperaments import (
    PRIMES_5,
    PRIMES_7,
    Rank2Temperament,
    TEMPERAMENTS,
    all_temperaments,
    hermite_normal_form,
    monzo,
    nearest_temperaments,
    rank2_from_comma,
    saturated_annihilator,
    temperament,
)

ALL = all_temperaments()
NAMES = sorted(TEMPERAMENTS)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def _in_lattice(hnf_rows, v):
    """True when integer vector ``v`` lies in the lattice spanned by ``hnf_rows``.

    Greedy elimination is valid because Hermite rows are in echelon form: each
    pivot column is zero in every later row, so subtracting a multiple of one
    row never disturbs an already-cleared column.
    """
    v = list(v)
    for row in hnf_rows:
        pivot = next(i for i, x in enumerate(row) if x != 0)
        if v[pivot] % row[pivot] != 0:
            return False
        q = v[pivot] // row[pivot]
        v = [a - q * b for a, b in zip(v, row)]
    return not any(v)


def _minor_gcd(mapping):
    """gcd of the 2x2 minors -- 1 exactly when the row lattice is saturated."""
    a, b = mapping
    g = 0
    for i, j in combinations(range(len(a)), 2):
        g = math.gcd(g, abs(a[i] * b[j] - a[j] * b[i]))
    return g


# --------------------------------------------------------------------------- #
# monzo
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "ratio, primes, expected",
    [
        (Fraction(81, 80), PRIMES_5, (-4, 4, -1)),
        (Fraction(3, 2), PRIMES_5, (-1, 1, 0)),
        (2, PRIMES_5, (1, 0, 0)),
        (1, PRIMES_5, (0, 0, 0)),
        (Fraction(1, 2), PRIMES_5, (-1, 0, 0)),
        (Fraction(2048, 2025), PRIMES_5, (11, -4, -2)),
        (Fraction(531441, 524288), PRIMES_5, (-19, 12, 0)),
        (Fraction(64, 63), PRIMES_7, (6, -2, 0, -1)),
        (Fraction(126, 125), PRIMES_7, (1, 2, -3, 1)),
    ],
)
def test_monzo_values(ratio, primes, expected):
    assert monzo(ratio, primes) == expected


@pytest.mark.parametrize("ratio, primes", [(Fraction(81, 80), PRIMES_5),
                                           (Fraction(126, 125), PRIMES_7)])
def test_monzo_round_trip(ratio, primes):
    m = monzo(ratio, primes)
    back = Fraction(1)
    for p, e in zip(primes, m):
        back *= Fraction(p) ** e
    assert back == ratio


def test_monzo_rejects_out_of_limit():
    with pytest.raises(ValueError, match="not 5-limit"):
        monzo(Fraction(7, 5))


def test_monzo_rejects_float():
    with pytest.raises(TypeError, match="exact ratio"):
        monzo(81 / 80)


def test_monzo_rejects_non_positive():
    with pytest.raises(ValueError, match="must be positive"):
        monzo(Fraction(-3, 2))


@pytest.mark.parametrize(
    "primes, match",
    [
        ((2, 3, 3), "strictly increasing"),
        ((2, 5, 3), "strictly increasing"),
        ((2, 4), "not prime"),
        ((2, 3, 25), "not prime"),
        ((1, 2, 3), "not prime"),
        ((), "empty"),
    ],
)
def test_monzo_rejects_a_malformed_prime_basis(primes, match):
    """A repeated or composite basis makes the monzo non-unique, not merely odd.

    Over ``(2, 3, 3)`` the factorisation loop drains every factor of 3 into the
    first of the two slots and reports the second as untouched, so the mapping,
    the periods and every cent value downstream come out self-consistent and
    wrong.
    """
    with pytest.raises(ValueError, match=match):
        monzo(Fraction(9, 8), primes)


# --------------------------------------------------------------------------- #
# integer linear algebra
# --------------------------------------------------------------------------- #
def test_hermite_known_values():
    assert hermite_normal_form([[-1, -1, 0], [0, 1, 4]]) == [[1, 0, -4], [0, 1, 4]]
    assert hermite_normal_form([[2, 4], [3, 6]]) == [[1, 2]]
    assert hermite_normal_form([[0, 0], [0, 0]]) == []


def test_hermite_is_canonical_across_bases():
    """Unimodular recombinations of a basis share one Hermite normal form."""
    base = [[1, 0, -4], [0, 1, 4]]
    variants = [
        [[1, 0, -4], [0, 1, 4]],
        [[1, 1, 0], [0, 1, 4]],
        [[-1, -1, 0], [0, 1, 4]],
        [[3, 2, -4], [2, 1, -4]],  # det -1 recombination
        [[1, 3, 8], [1, 2, 4]],
    ]
    for v in variants:
        assert hermite_normal_form(v) == base


def test_hermite_is_idempotent():
    for name in NAMES:
        rows = [list(r) for r in ALL[name].mapping]
        once = hermite_normal_form(rows)
        assert hermite_normal_form(once) == once


def test_hermite_pivots_positive_and_reduced_above():
    for name in NAMES:
        rows = hermite_normal_form([list(r) for r in ALL[name].mapping])
        p0 = next(i for i, x in enumerate(rows[0]) if x != 0)
        p1 = next(i for i, x in enumerate(rows[1]) if x != 0)
        assert p0 < p1
        assert rows[0][p0] > 0 and rows[1][p1] > 0
        assert 0 <= rows[0][p1] < rows[1][p1]


@pytest.mark.parametrize("name", NAMES)
def test_annihilator_is_saturated_by_brute_force(name):
    """Every small val that tempers the comma out must lie in the basis lattice.

    This is the check a rational nullspace with cleared denominators fails: such
    a basis spans a finite-index sublattice, so some genuine vals are missing
    from it, and the missing ones are exactly the ones with the true (smaller)
    number of periods per octave.
    """
    m = monzo(TEMPERAMENTS[name], PRIMES_5)
    hnf = hermite_normal_form(saturated_annihilator([m]))
    found = 0
    for v in product(range(-8, 9), repeat=3):
        if _dot(v, m) != 0:
            continue
        found += 1
        assert _in_lattice(hnf, v), f"{name}: val {v} tempers out {m} but is not in {hnf}"
    assert found > 1


def test_annihilator_rejects_dependent_commas():
    with pytest.raises(ValueError, match="dependent"):
        saturated_annihilator([monzo(Fraction(81, 80)), monzo(Fraction(81, 80))])


def test_annihilator_rank():
    assert len(saturated_annihilator([monzo(Fraction(81, 80))])) == 2
    assert len(saturated_annihilator([monzo(Fraction(81, 80), PRIMES_7)])) == 3
    assert (
        len(
            saturated_annihilator(
                [monzo(Fraction(81, 80), PRIMES_7), monzo(Fraction(126, 125), PRIMES_7)]
            )
        )
        == 2
    )


# --------------------------------------------------------------------------- #
# structural invariants over the whole catalogue
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", NAMES)
def test_mapping_annihilates_comma_exactly(name):
    t = ALL[name]
    for comma in t.commas:
        m = monzo(comma, t.primes)
        assert tuple(_dot(row, m) for row in t.mapping) == (0, 0)


@pytest.mark.parametrize("name", NAMES)
def test_periods_read_off_the_mapping(name):
    t = ALL[name]
    assert t.periods_per_octave == t.mapping[0][0] >= 1
    # Prime 2 is periods only: that is what makes the period a pure 1/n octave.
    assert t.mapping[1][0] == 0
    assert t.period_cents == pytest.approx(1200.0 / t.periods_per_octave)


@pytest.mark.parametrize("name", NAMES)
def test_mapping_lattice_is_saturated(name):
    assert _minor_gcd(ALL[name].mapping) == 1


@pytest.mark.parametrize("name", NAMES)
def test_octave_is_pure_and_generator_reduced(name):
    t = ALL[name]
    assert t.prime_errors[0] == pytest.approx(0.0, abs=1e-9)
    assert 0.0 <= t.generator_cents < t.period_cents
    assert 0.0 <= t.generator_fraction() < 1.0
    assert t.complement_cents == pytest.approx(t.period_cents - t.generator_cents)


@pytest.mark.parametrize("name", NAMES)
def test_prime_errors_match_the_mapping(name):
    t = ALL[name]
    period = 1.0 / t.periods_per_octave
    gen = t.generator_octaves
    for (ai, bi, p, err) in zip(t.mapping[0], t.mapping[1], t.primes, t.prime_errors):
        assert err == pytest.approx(1200.0 * (ai * period + bi * gen - math.log2(p)))
    assert t.max_error == pytest.approx(max(abs(e) for e in t.prime_errors))
    assert t.rms_error == pytest.approx(
        math.sqrt(sum(e * e for e in t.prime_errors) / len(t.prime_errors))
    )
    assert t.rms_error <= t.max_error


@pytest.mark.parametrize("name", NAMES)
def test_comma_vanishes_in_the_tempered_tuning(name):
    """The defining comma must measure exactly zero cents once tempered."""
    t = ALL[name]
    for comma in t.commas:
        m = monzo(comma, t.primes)
        size = sum(e * err for e, err in zip(m, t.prime_errors))
        just = 1200.0 * math.log2(float(comma))
        assert size + just == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("name", NAMES)
def test_generator_is_the_least_squares_optimum(name):
    """Perturbing the generator must increase the Tenney-weighted error."""
    t = ALL[name]
    period = 1.0 / t.periods_per_octave

    def cost(gen):
        total = 0.0
        for ai, bi, p in zip(t.mapping[0], t.mapping[1], t.primes):
            l = math.log2(p)
            total += ((ai * period + bi * gen - l) / l) ** 2
        return total

    best = cost(t.generator_octaves)
    for step in (1e-4, -1e-4, 1e-3, -1e-3):
        assert cost(t.generator_octaves + step) > best


# --------------------------------------------------------------------------- #
# published mappings and periods
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name, periods",
    [
        ("srutal", 2),
        ("augmented", 3),
        ("diminished", 4),
        ("blackwood", 5),
        ("compton", 12),
        ("meantone", 1),
        ("magic", 1),
    ],
)
def test_periods_per_octave(name, periods):
    assert ALL[name].periods_per_octave == periods


@pytest.mark.parametrize(
    "name, mapping",
    [
        ("meantone", ((1, 1, 0), (0, 1, 4))),
        ("magic", ((1, 0, 2), (0, 5, 1))),
        ("hanson", ((1, 0, 1), (0, 6, 5))),
        ("tetracot", ((1, 1, 1), (0, 4, 9))),
        ("porcupine", ((1, 2, 3), (0, -3, -5))),
        ("negri", ((1, 2, 2), (0, -4, 3))),
        ("srutal", ((2, 3, 5), (0, 1, -2))),
        ("compton", ((12, 19, 27), (0, 0, 1))),
        ("blackwood", ((5, 8, 11), (0, 0, 1))),
    ],
)
def test_published_mappings(name, mapping):
    assert ALL[name].mapping == mapping


def test_meantone_hermite_form_is_the_familiar_one():
    """5 is four generators up; the syntonic comma is what makes that true."""
    assert ALL["meantone"].hermite_mapping == ((1, 0, -4), (0, 1, 4))
    assert ALL["meantone"].mapping == ((1, 1, 0), (0, 1, 4))


def test_septimal_meantone_needs_two_commas():
    sm = rank2_from_comma(
        Fraction(81, 80), PRIMES_7, "septimal meantone",
        extra_commas=[Fraction(126, 125)],
    )
    assert sm.mapping == ((1, 1, 0, -3), (0, 1, 4, 10))
    assert sm.periods_per_octave == 1
    assert sm.generator_cents == pytest.approx(696.952, abs=1e-3)
    for comma in (Fraction(81, 80), Fraction(126, 125)):
        m = monzo(comma, PRIMES_7)
        assert tuple(_dot(row, m) for row in sm.mapping) == (0, 0)
    # Adding prime 7 pulls the fifth flatter than the 5-limit optimum.
    assert sm.generator_cents < ALL["meantone"].generator_cents


# --------------------------------------------------------------------------- #
# generator values against published tunings
# --------------------------------------------------------------------------- #
#: Approximate published 5-limit generators.  A generator and its complement
#: build identical scales, so both are accepted.
PUBLISHED = {
    "meantone": 696.2,
    "schismatic": 701.7,
    "porcupine": 163.3,
    "magic": 380.7,
    "hanson": 317.1,
    "tetracot": 176.1,
    "negri": 126.2,
    "srutal": 104.7,
}


@pytest.mark.parametrize("name, published", sorted(PUBLISHED.items()))
def test_pote_generators_match_published(name, published):
    t = ALL[name]
    pote = t.pote_generator_cents
    distance = min(abs(pote - published), abs(t.period_cents - pote - published))
    assert distance < 1.0, f"{name}: POTE {pote:.3f} vs published {published}"


@pytest.mark.parametrize("name, published", sorted(PUBLISHED.items()))
def test_cte_generators_are_close_to_published_too(name, published):
    """CTE and POTE differ, but not by much for temperaments this accurate."""
    t = ALL[name]
    cte = t.generator_cents
    distance = min(abs(cte - published), abs(t.complement_cents - published))
    assert distance < 1.1, f"{name}: CTE {cte:.3f} vs published {published}"


def test_meantone_cte_and_pote_are_different_and_both_known():
    """696.239 is the published POTE fifth; 697.214 is the published CTE fifth."""
    mt = ALL["meantone"]
    assert mt.generator_cents == pytest.approx(697.214, abs=1e-3)
    assert mt.pote_generator_cents == pytest.approx(696.239, abs=1e-3)


@pytest.mark.parametrize(
    "name, cte, pote",
    [
        # Documented disagreements with the "published POTE" figures the module
        # was checked against.  These are pinned so a regression is visible.
        # mavila: CTE 677.145 sits on the quoted ~677; POTE does not.
        ("mavila", 677.145, 679.806),
        # dicot: max error 31.6 c, so the two optima straddle the quoted ~350.9.
        ("dicot", 354.664, 348.594),
        # father: max error 76.2 c; quoted ~447 is neither, and is the
        # complement of the direction this module's sign convention picks.
        ("father", 737.469, 743.986),
    ],
)
def test_inaccurate_temperaments_pinned(name, cte, pote):
    t = ALL[name]
    assert t.generator_cents == pytest.approx(cte, abs=1e-3)
    assert t.pote_generator_cents == pytest.approx(pote, abs=1e-3)


def test_error_magnitudes_rank_as_expected():
    """Schismatic is the most accurate 5-limit name here; father the worst."""
    worst = max(ALL.values(), key=lambda t: t.max_error)
    best = min(ALL.values(), key=lambda t: t.max_error)
    assert worst.name == "father"
    assert best.name == "schismatic"
    assert ALL["schismatic"].max_error < 0.3
    assert ALL["amity"].max_error < 0.5
    assert ALL["father"].max_error > 50.0


# --------------------------------------------------------------------------- #
# construction and canonicalisation
# --------------------------------------------------------------------------- #
def test_any_basis_of_the_lattice_canonicalises_the_same():
    a = Rank2Temperament("m", Fraction(81, 80), PRIMES_5, ((1, 0, -4), (0, 1, 4)))
    b = Rank2Temperament("m", Fraction(81, 80), PRIMES_5, ((1, 1, 0), (0, 1, 4)))
    c = Rank2Temperament("m", Fraction(81, 80), PRIMES_5, ((1, 3, 8), (1, 2, 4)))
    assert a.mapping == b.mapping == c.mapping == ((1, 1, 0), (0, 1, 4))
    assert a == b == c


def test_construction_rejects_a_mapping_that_misses_the_comma():
    with pytest.raises(ValueError, match="does not temper out"):
        Rank2Temperament("bad", Fraction(81, 80), PRIMES_5, ((1, 0, 0), (0, 1, 0)))


def test_construction_rejects_dependent_rows():
    with pytest.raises(ValueError, match="linearly dependent"):
        Rank2Temperament("bad", Fraction(81, 80), PRIMES_5, ((1, 0, -4), (2, 0, -8)))


def test_construction_rejects_wrong_row_length():
    with pytest.raises(ValueError, match="one entry per prime"):
        Rank2Temperament("bad", Fraction(81, 80), PRIMES_5, ((1, 0), (0, 1)))


def test_rank2_from_comma_rejects_non_primitive():
    with pytest.raises(ValueError, match="primitive root 81/80"):
        rank2_from_comma(Fraction(6561, 6400))


def test_rank2_from_comma_rejects_unison():
    with pytest.raises(ValueError, match="tempers nothing out"):
        rank2_from_comma(Fraction(1, 1))


def test_rank2_from_comma_needs_enough_commas_above_the_5_limit():
    with pytest.raises(ValueError, match="rank-3 temperament"):
        rank2_from_comma(Fraction(81, 80), PRIMES_7)


def test_rank2_from_comma_default_name():
    assert rank2_from_comma(Fraction(81, 80)).name == "81/80"


def test_a_no_twos_subgroup_is_refused_rather_than_mistuned():
    """The class is octave-locked; a 3-equave basis must raise, not mis-report.

    Before this guard, ``(3, 5, 7)`` produced ``period_cents == 1200.0`` for an
    equave of 3 (the truth is 1901.955), ``period_ratio == 2.0``,
    ``prime_errors[0] == -701.955`` against a docstring promising exactly 0.0,
    and a max error of 1504 cents -- all silently.
    """
    with pytest.raises(ValueError, match="must start at 2"):
        rank2_from_comma(Fraction(245, 243), (3, 5, 7))
    with pytest.raises(ValueError, match="must start at 2"):
        Rank2Temperament(
            "x", Fraction(245, 243), (3, 5, 7), ((1, 1, 2), (0, 2, -1))
        )


def test_a_reordered_prime_basis_is_refused():
    """``(5, 3, 2)`` used to yield periods_per_octave 4 and a 300-cent period."""
    with pytest.raises(ValueError, match="strictly increasing"):
        rank2_from_comma(Fraction(81, 80), (5, 3, 2))


@pytest.mark.parametrize("name", NAMES)
def test_every_catalogue_temperament_is_octave_based(name):
    """The invariant the octave-lock protects, asserted where it is relied on."""
    t = ALL[name]
    assert t.primes[0] == 2
    assert t.period_ratio == pytest.approx(2.0 ** (1.0 / t.periods_per_octave))
    assert t.period_cents * t.periods_per_octave == pytest.approx(1200.0)
    assert t.prime_errors[0] == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# catalogue access
# --------------------------------------------------------------------------- #
def test_temperament_is_cached():
    assert temperament("meantone") is temperament("meantone")
    assert temperament("MEANTONE") is temperament("meantone")


def test_temperament_unknown_name_suggests():
    with pytest.raises(ValueError, match="did you mean 'meantone'"):
        temperament("meantime")
    with pytest.raises(ValueError, match="known names are"):
        temperament("zzz")


def test_all_temperaments_covers_the_catalogue():
    assert set(all_temperaments()) == set(TEMPERAMENTS)
    assert len(TEMPERAMENTS) == 20
    assert all(isinstance(c, Fraction) and c > 1 for c in TEMPERAMENTS.values())


# --------------------------------------------------------------------------- #
# nearest_temperaments
# --------------------------------------------------------------------------- #
def test_nearest_finds_meantone():
    got = nearest_temperaments(696.6)
    assert [name for name, _, _ in got] == ["meantone", "schismatic", "mavila"]
    assert got[0][2] == pytest.approx(abs(ALL["meantone"].generator_cents - 696.6))
    assert all(got[i][2] <= got[i + 1][2] for i in range(len(got) - 1))


def test_nearest_respects_n_and_max_distance():
    assert len(nearest_temperaments(696.6, n=1)) == 1
    assert nearest_temperaments(696.6, max_distance_cents=0.1) == []
    assert nearest_temperaments(640.0) == []


def test_nearest_filters_by_period():
    """A half-octave period is a different disc; 1200-cent names must not leak."""
    got = nearest_temperaments(105.0, period_cents=600.0)
    assert [name for name, _, _ in got] == ["srutal"]
    assert all(t.periods_per_octave == 2 for _, t, _ in got)
    # The same query against a full-octave period must not find srutal, even
    # though srutal's generator is 105 cents -- ripple is what lives there.
    octave_side = nearest_temperaments(105.0, period_cents=1200.0)
    assert "srutal" not in [name for name, _, _ in octave_side]
    assert octave_side[0][0] == "ripple"


def test_nearest_matches_the_complement():
    """503.8 c is meantone's fourth: the same temperament, drawn mirrored."""
    got = nearest_temperaments(ALL["meantone"].complement_cents)
    assert got[0][0] == "meantone"
    assert got[0][2] == pytest.approx(0.0, abs=1e-9)


def test_nearest_reduces_the_query_into_the_period():
    """A twelfth is a fifth as far as the labyrinth is concerned."""
    got = nearest_temperaments(ALL["meantone"].generator_cents + 1200.0)
    assert got[0][0] == "meantone"
    assert got[0][2] == pytest.approx(0.0, abs=1e-9)


def test_nearest_rejects_bad_period():
    with pytest.raises(ValueError, match="period_cents must be positive"):
        nearest_temperaments(700.0, period_cents=0.0)


# --------------------------------------------------------------------------- #
# scales
# --------------------------------------------------------------------------- #
def test_meantone_family_contains_the_diatonic():
    fam = ALL["meantone"].mos_family(12)
    assert [s.signature for s in fam] == ["2L1s", "2L3s", "5L2s", "7L5s"]
    diatonic = next(s for s in fam if s.cardinality == 7)
    assert diatonic.signature == "5L2s"
    assert diatonic.is_proper
    assert diatonic.generator_cents == pytest.approx(ALL["meantone"].generator_cents)


def test_hanson_family_is_the_kleismic_series():
    assert [s.cardinality for s in ALL["hanson"].mos_family(19)] == [3, 4, 7, 11, 15, 19]


def test_porcupine_family_contains_7_and_8_notes():
    cards = [s.cardinality for s in ALL["porcupine"].mos_family(16)]
    assert 7 in cards and 8 in cards
    seven = next(s for s in ALL["porcupine"].mos_family(16) if s.cardinality == 7)
    assert seven.signature in ("6L1s", "1L6s")


@pytest.mark.parametrize("name", NAMES)
def test_family_scales_carry_the_temperament_period(name):
    t = ALL[name]
    fam = t.mos_family(24)
    assert fam, f"{name} produced no MOS at all"
    for s in fam:
        assert s.period_cents == pytest.approx(t.period_cents)
        assert s.generator_cents == pytest.approx(t.generator_cents, abs=1e-6)
        assert t.supports(s)


def test_supports_tolerance():
    from biotuner.mos.scale import MOSScale

    mt = ALL["meantone"]
    assert mt.supports(MOSScale.from_signature(5, 2, tuning=31))
    assert mt.supports(MOSScale.from_generator(3 / 2, 7))  # 4.7 c away
    assert not mt.supports(MOSScale.from_generator(3 / 2, 7), tol_cents=2.0)
    # 19-EDO's fifth is 694.74: inside 5 cents of meantone, outside 2.
    assert mt.supports(MOSScale.from_signature(5, 2, tuning=19))
    assert not mt.supports(MOSScale.from_signature(5, 2, tuning=19), tol_cents=2.0)


def test_supports_rejects_the_wrong_period():
    from biotuner.mos.scale import MOSScale

    half_octave = MOSScale.from_signature(5, 2, tuning=12, period=2.0 ** 0.5)
    assert not ALL["meantone"].supports(half_octave)
    assert not ALL["srutal"].supports(MOSScale.from_signature(5, 2, tuning=12))


def test_supports_period_check_is_not_masked_by_the_generator_check():
    """A scale with the *right* generator and the *wrong* period must fail.

    The two cases above are already rejected on their generators alone (152 c
    and 205 c away), so they pass even with the period comparison deleted.
    Srutal's 105.136-cent generator over a full octave instead of a half octave
    isolates the period branch: nothing but that comparison can reject it.
    """
    from biotuner.mos.scale import MOSScale

    srutal = ALL["srutal"]
    assert srutal.period_cents == pytest.approx(600.0)
    wrong_period = MOSScale.from_generator(
        2.0 ** (srutal.generator_cents / 1200.0), 7
    )
    assert wrong_period.generator_cents == pytest.approx(
        srutal.generator_cents, abs=1e-9
    )
    assert wrong_period.period_cents == pytest.approx(1200.0)
    assert not srutal.supports(wrong_period)
    # ... and it stays rejected however loose the generator tolerance gets,
    # right up to the point where 600 c and 1200 c are themselves "close".
    assert not srutal.supports(wrong_period, tol_cents=100.0)


# --------------------------------------------------------------------------- #
# interop
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", NAMES)
def test_to_dict_is_flat_and_complete(name):
    d = ALL[name].to_dict()
    assert d["name"] == name
    assert d["periods_per_octave"] == ALL[name].periods_per_octave
    assert Fraction(d["comma"]) == TEMPERAMENTS[name]
    assert len(d["prime_errors"]) == len(PRIMES_5)
    assert set(d) >= {
        "mapping", "hermite_mapping", "period_cents", "generator_cents",
        "complement_cents", "pote_generator_cents", "generator_fraction",
        "max_error", "rms_error",
    }


def test_summary_mentions_the_comma_and_the_generator():
    text = ALL["porcupine"].summary()
    assert "250/243" in text
    assert "<0 -3 -5|" in text
    assert "164.166" in text


def test_frozen():
    with pytest.raises(Exception):
        ALL["meantone"].name = "nope"
