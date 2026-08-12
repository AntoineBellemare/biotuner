"""Tests for :mod:`biotuner.mos.fourier` -- Fourier scratching (Milne et al. §5).

The interesting tests are the ones that check the paper's claims about pure
partials against the Fig. 8 keyboard (``test_fig8_*`` and ``test_coprime_*``).
They are *not* unconditionally true, and the conditions are asserted here rather
than assumed: see :func:`christoffel_mode` and the failure-mode tests.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from biotuner.mos import theory as T
from biotuner.mos.fourier import (
    TWO_PI,
    NoteEvent,
    PlayState,
    keyboard_sectors,
    partial,
    phase_to_degree,
    scratch_sequence,
    to_events,
    to_frequencies,
)
from biotuner.mos.modes import christoffel_mode
from biotuner.mos.scale import MOSScale

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def is_cyclic_rotation(seq):
    """True when ``seq`` is ``[0, 1, ..., n-1]`` rotated by some amount."""
    base = list(range(len(seq)))
    return any(list(seq) == base[r:] + base[:r] for r in range(len(base)))


def random_state(n, seed):
    rng = np.random.default_rng(seed)
    return PlayState(rng.normal(size=n) + 1j * rng.normal(size=n))


PROPER_SIGNATURES = [
    (2, 1), (1, 2), (3, 1), (1, 3), (3, 2), (2, 3), (4, 1), (1, 4),
    (5, 2), (2, 5), (4, 3), (3, 4), (5, 3), (3, 5), (5, 4), (4, 5),
    (7, 2), (2, 7), (5, 7), (7, 5),
]


# --------------------------------------------------------------------------- #
# PlayState construction and views
# --------------------------------------------------------------------------- #
def test_construction_coerces_and_copies():
    src = np.array([1.0, 2.0, 3.0])
    s = PlayState(src)
    assert s.f.dtype == np.complex128
    assert s.n == 3
    src[0] = 99.0
    assert s.f[0] == 1.0  # the state kept its own copy
    assert s.f.flags.writeable is False


@pytest.mark.parametrize(
    "bad", [np.zeros((2, 2)), np.zeros(0), np.zeros((1, 3)), np.zeros((2, 2, 2))]
)
def test_construction_rejects_non_1d_and_empty(bad):
    with pytest.raises(ValueError):
        PlayState(bad)


def test_from_polar_round_trip():
    mags = [1.0, 0.25, 3.0, 0.5]
    phs = [0.0, 1.0, 3.0, 5.5]
    s = PlayState.from_polar(mags, phs)
    assert np.allclose(s.magnitudes, mags)
    assert np.allclose(s.phases, phs)


def test_from_polar_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        PlayState.from_polar([1.0, 2.0], [0.0])


def test_phases_are_in_zero_two_pi():
    s = random_state(64, seed=1)
    ph = s.phases
    assert np.all(ph >= 0.0)
    assert np.all(ph < TWO_PI)


def test_phase_just_below_zero_snaps_to_root_not_top():
    # exp(-1e-15j) is the root to any musical precision; without the snap it
    # would reduce to ~2*pi and read as the top key of the scale.
    s = PlayState([np.exp(-1e-15j)])
    assert s.phases[0] == 0.0


def test_energy_matches_parseval():
    s = random_state(11, seed=2)
    assert s.energy == pytest.approx(s.n * float(np.sum(np.abs(s.spectrum) ** 2)))
    assert partial(9, 4).energy == pytest.approx(9.0)


def test_equality_and_unhashable():
    a = PlayState([1, 2j])
    assert a == PlayState([1, 2j])
    assert a != PlayState([1, 2j, 0])
    assert (a == "not a state") is False
    assert a.__eq__("not a state") is NotImplemented
    with pytest.raises(TypeError):
        hash(a)


# --------------------------------------------------------------------------- #
# Claim 1 -- the DFT round trip
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [1, 2, 3, 5, 7, 12, 31, 64])
def test_claim1_spectrum_round_trip(n):
    s = random_state(n, seed=100 + n)
    assert PlayState.from_spectrum(s.spectrum).allclose(s)


@pytest.mark.parametrize("n", [1, 2, 5, 12])
def test_claim1_round_trip_the_other_way(n):
    rng = np.random.default_rng(7 * n)
    a = rng.normal(size=n) + 1j * rng.normal(size=n)
    assert np.allclose(PlayState.from_spectrum(a).spectrum, a)


def test_from_spectrum_rejects_bad_shapes():
    with pytest.raises(ValueError):
        PlayState.from_spectrum(np.zeros((2, 2)))
    with pytest.raises(ValueError):
        PlayState.from_spectrum([])


# --------------------------------------------------------------------------- #
# Claim 2 -- a partial's spectrum is a unit impulse
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [1, 2, 3, 5, 7, 12, 19])
def test_claim2_partial_spectrum_is_unit_impulse(n):
    for k in range(n):
        a = partial(n, k).spectrum
        expected = np.zeros(n, dtype=complex)
        expected[k] = 1.0
        assert np.allclose(a, expected, atol=1e-12), (n, k, a)
        assert abs(a[k] - 1.0) < 1e-12
        others = np.delete(np.abs(a), k)
        assert np.max(others, initial=0.0) < 1e-12


@pytest.mark.parametrize("k", [-13, -7, -1, 0, 5, 7, 19])
def test_claim2_partial_index_wraps_modulo_n(k):
    assert partial(7, k) == partial(7, k % 7)


def test_partial_fingers_are_unit_loudness_and_evenly_spaced():
    p = partial(8, 3)
    assert np.allclose(p.magnitudes, 1.0)
    assert sorted(round(float(x), 9) for x in p.phases) == pytest.approx(
        [round(TWO_PI * j / 8, 9) for j in range(8)]
    )


def test_partial_rejects_zero_fingers():
    with pytest.raises(ValueError, match="at least one finger"):
        partial(0, 1)


# --------------------------------------------------------------------------- #
# Claim 5 -- the Fig. 8 keyboard
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_large,n_small", PROPER_SIGNATURES)
def test_claim5_sector_widths_are_proportional_to_steps(n_large, n_small):
    scale = MOSScale.from_signature(n_large, n_small, tuning="central")
    secs = keyboard_sectors(scale)
    degs = list(scale.degrees) + [1.0]
    steps = [degs[i + 1] - degs[i] for i in range(scale.cardinality)]
    widths = [hi - lo for lo, hi in secs]
    assert len(secs) == scale.cardinality
    for w, st in zip(widths, steps):
        assert w == pytest.approx(TWO_PI * st, abs=1e-12)


@pytest.mark.parametrize("n_large,n_small", PROPER_SIGNATURES)
def test_claim5_sectors_tile_the_circle(n_large, n_small):
    scale = MOSScale.from_signature(n_large, n_small, tuning="central")
    secs = keyboard_sectors(scale)
    assert secs[0][0] == 0.0
    assert secs[-1][1] == pytest.approx(TWO_PI, abs=1e-12)
    for (_, end), (start, _) in zip(secs, secs[1:]):
        assert end == start  # no gaps, no overlaps -- shared edge object
    assert sum(hi - lo for lo, hi in secs) == pytest.approx(TWO_PI, abs=1e-12)


def test_claim5_keys_are_not_uniform_and_track_hardness():
    scale = MOSScale.from_signature(5, 2, tuning=31)  # meantone, R = 5/3
    widths = [hi - lo for lo, hi in keyboard_sectors(scale)]
    wide = max(widths)
    narrow = min(widths)
    assert widths.count(pytest.approx(wide)) == 5
    assert widths.count(pytest.approx(narrow)) == 2
    assert wide / narrow == pytest.approx(scale.hardness)
    assert wide / narrow == pytest.approx(5 / 3)


def test_keyboard_sectors_exact_widths_in_12edo():
    secs = keyboard_sectors(MOSScale.from_signature(5, 2, tuning=12))
    widths = [round(hi - lo, 9) for lo, hi in secs]
    tone, semitone = round(TWO_PI / 6, 9), round(TWO_PI / 12, 9)
    assert widths == [tone, tone, tone, semitone, tone, tone, semitone]


@pytest.mark.parametrize("n_large,n_small", PROPER_SIGNATURES)
def test_phase_to_degree_agrees_with_sectors(n_large, n_small):
    scale = MOSScale.from_signature(n_large, n_small, tuning="central")
    secs = keyboard_sectors(scale)
    rng = np.random.default_rng(11)
    probes = list(rng.uniform(0.0, TWO_PI, 200))
    probes += [(lo + hi) / 2 for lo, hi in secs]  # sector midpoints
    for p in probes:
        i = phase_to_degree(p, scale)
        lo, hi = secs[i]
        assert lo <= p < hi, (p, i, lo, hi)


def test_phase_to_degree_reduces_modulo_two_pi():
    scale = MOSScale.from_signature(5, 2, tuning=12)
    for p in (0.9, 0.9 + TWO_PI, 0.9 - 3 * TWO_PI):
        assert phase_to_degree(p, scale) == phase_to_degree(0.9, scale)


def test_phase_to_degree_boundary_belongs_to_the_key_above():
    scale = MOSScale.from_signature(5, 2, tuning=12)
    secs = keyboard_sectors(scale)
    for i, (lo, _) in enumerate(secs):
        assert phase_to_degree(lo, scale) == i


def test_keyboard_rejects_non_scales():
    with pytest.raises(TypeError, match="MOSScale or a Mode"):
        keyboard_sectors(object())

    class NotRooted:
        degrees = [0.1, 0.5]

    with pytest.raises(ValueError, match="must start at 0"):
        keyboard_sectors(NotRooted())

    class NotAscending:
        degrees = [0.0, 0.5, 0.5]

    with pytest.raises(ValueError, match="strictly ascending"):
        keyboard_sectors(NotAscending())

    class OutOfRange:
        degrees = [0.0, 1.5]

    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        keyboard_sectors(OutOfRange())


# --------------------------------------------------------------------------- #
# Claim 3 -- Fig. 8: the first partial plays every tone exactly once
# --------------------------------------------------------------------------- #
def _degrees_of(state, mode):
    return [e.degree for e in to_events(state, mode)]


def test_fig8_seven_note_case():
    scale = MOSScale.from_signature(5, 2, tuning=12)
    mode = christoffel_mode(scale)
    assert mode.word == "sLLsLLL"  # Locrian
    assert mode.index == 6
    degrees = _degrees_of(partial(7, 1), mode)
    assert set(degrees) == set(range(7))
    assert len(degrees) == len(set(degrees))
    assert is_cyclic_rotation(degrees)
    assert degrees == [0, 1, 2, 3, 4, 5, 6]  # the rotation is the identity


def test_fig8_five_note_case():
    scale = MOSScale.from_signature(2, 3, tuning=12)  # anhemitonic pentatonic
    mode = christoffel_mode(scale)
    assert mode.word == "ssLsL"
    degrees = _degrees_of(partial(5, 1), mode)
    assert set(degrees) == set(range(5))
    assert is_cyclic_rotation(degrees)
    assert degrees == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("n_large,n_small", PROPER_SIGNATURES)
def test_fig8_holds_in_the_christoffel_mode(n_large, n_small):
    scale = MOSScale.from_signature(n_large, n_small, tuning="central")
    n = scale.cardinality
    degrees = _degrees_of(partial(n, 1), christoffel_mode(scale))
    assert set(degrees) == set(range(n))
    assert is_cyclic_rotation(degrees)


@pytest.mark.parametrize("n_large,n_small", PROPER_SIGNATURES)
def test_fig8_propriety_is_sufficient(n_large, n_small):
    """Every coherent tuning works -- swept across the whole coherent range.

    ``coherence_range`` is exactly the interval over which Blackwood's
    ``R = L/s`` runs from 1 to 2, so this sweeps propriety end to end.
    """
    lo, hi = T.coherence_range(n_large, n_small, bright=True)
    n = n_large + n_small
    for t in np.linspace(0.001, 0.999, 21):
        g = float(lo) + t * (float(hi) - float(lo))
        scale = MOSScale(n_large, n_small, g, validate=False)
        assert scale.hardness <= 2.0 + 1e-9
        degrees = _degrees_of(partial(n, 1), christoffel_mode(scale))
        assert set(degrees) == set(range(n)), (scale.signature, g, degrees)


def test_fig8_every_failure_is_an_improper_scale():
    """The contrapositive of sufficiency, on a batch that really does fail.

    ``tuning='middle'`` is the midpoint of the *full* valid generator range
    rather than of the coherent sub-range, and turns out to be improper for
    every signature -- a useful stress batch: half of these 20 miss tones, and
    every one that does has ``R`` well above 2.

    Within *this* batch the two groups happen to separate cleanly (worst
    survivor ``R = 2.8``, best casualty ``R = 3.25``), but that separation is an
    artefact of the batch, not a law -- see
    :func:`test_fig8_has_no_global_hardness_threshold`, where ``5L2s`` sails
    past ``R = 6``.  The cut-off is a property of the signature, not of hardness
    alone, which is why the guarantee this module documents is stated as
    propriety and nothing sharper.
    """
    worked, failed = [], []
    for n_large, n_small in PROPER_SIGNATURES:
        scale = MOSScale.from_signature(n_large, n_small, tuning="middle")
        assert scale.is_proper is False
        n = scale.cardinality
        degrees = _degrees_of(partial(n, 1), christoffel_mode(scale))
        (worked if set(degrees) == set(range(n)) else failed).append(
            (scale.signature, scale.hardness)
        )
    assert failed, "stress batch degenerated -- nothing failed, test is vacuous"
    assert worked, "stress batch is too hard -- nothing worked, test is vacuous"
    assert all(R > 2.0 for _, R in failed)
    assert max(R for _, R in worked) == pytest.approx(2.8)
    assert dict(failed)["4L5s"] == pytest.approx(3.25)
    assert dict(failed)["1L2s"] == pytest.approx(4.0)
    assert [sig for sig, _ in failed] == [
        "1L2s", "1L3s", "2L3s", "1L4s", "2L5s", "3L4s", "3L5s", "4L5s",
        "2L7s", "5L7s",
    ]


def test_fig8_has_no_global_hardness_threshold():
    """``5L2s`` survives an ``R`` at which ``4L5s`` has long since broken.

    This is the claim the module's docstring makes, isolated from the
    ``tuning='middle'`` batch so that it does not depend on that batch's
    accidental clean split.
    """
    diatonic = MOSScale.from_fraction(0.594, 7)
    assert diatonic.signature == "5L2s"
    assert diatonic.hardness > 6.0
    assert set(_degrees_of(partial(7, 1), christoffel_mode(diatonic))) == set(range(7))

    nine = MOSScale.from_signature(4, 5, tuning="middle")
    assert nine.hardness == pytest.approx(3.25)
    assert set(_degrees_of(partial(9, 1), christoffel_mode(nine))) != set(range(9))
    assert nine.hardness < diatonic.hardness


def test_fig8_1L2s_at_R4_lands_exactly_on_a_boundary_and_fails():
    """A regression guard for the boundary tolerance.

    ``1L2s`` at ``R = 4`` has Christoffel-mode degrees ``0, 1/6, 1/3`` and
    finger 1 of ``partial(3, 1)`` sits at exactly ``1/3`` of a turn -- exactly
    on degree 2's lower edge.  Keys are half-open, so that finger belongs to
    key 2 and degree 1 is missed.  Floating point puts the computed edge about
    9e-16 *above* the computed phase, so without the boundary tolerance in
    :func:`phase_to_degree` this reads as the (wrong) success ``[0, 1, 2]``.
    """
    scale = MOSScale.from_signature(1, 2, tuning="middle")
    assert scale.hardness == pytest.approx(4.0)
    mode = christoffel_mode(scale)
    assert mode.word == "ssL"
    assert [pytest.approx(x) for x in mode.degrees] == [0.0, 1 / 6, 1 / 3]
    # finger 1 is one third of a turn round, i.e. exactly degree 2's edge
    assert float(partial(3, 1).phases[1]) == pytest.approx(TWO_PI / 3, abs=1e-12)
    assert _degrees_of(partial(3, 1), mode) == [0, 2, 2]


def test_fig8_survives_well_past_propriety_for_the_diatonic():
    """Coherence is sufficient, not necessary: 5L2s keeps working out to R ~ 8."""
    pythagorean = MOSScale.from_generator(3 / 2, 7)
    assert pythagorean.is_proper is False
    assert pythagorean.hardness == pytest.approx(2.26, abs=0.01)
    assert _degrees_of(partial(7, 1), christoffel_mode(pythagorean)) == [
        0, 1, 2, 3, 4, 5, 6
    ]
    # R = 6 still fine
    hard = MOSScale.from_fraction(0.594, 7)
    assert hard.hardness > 6.0
    assert set(_degrees_of(partial(7, 1), christoffel_mode(hard))) == set(range(7))


def test_fig8_fails_for_a_grossly_improper_scale():
    """Past R ~ 8 the narrow keys of 5L2s start missing fingers entirely."""
    scale = MOSScale.from_fraction(0.598, 7)
    assert scale.signature == "5L2s"
    assert scale.hardness > 8.0
    degrees = _degrees_of(partial(7, 1), christoffel_mode(scale))
    assert degrees == [0, 1, 2, 4, 4, 5, 6]
    assert set(degrees) != set(range(7))
    assert 3 not in degrees  # a tone the first partial never reaches
    assert degrees.count(4) == 2
    assert not is_cyclic_rotation(degrees)


def test_fig8_fails_outside_the_christoffel_mode():
    """The claim is about a mode, not about a scale -- documented failure mode.

    Under the Fig. 8 layout a tone sits at the *bottom* edge of its key, so
    finger 0 (phase exactly 0) falls into key 0, and key 0 of the brightest mode
    is a whole tone wide while the fingers are only 1/7 of a turn apart.
    """
    scale = MOSScale.from_signature(5, 2, tuning=12)
    brightest = scale.mode(0)
    assert brightest.word == "LLLsLLs"
    degrees = _degrees_of(partial(7, 1), brightest)
    assert degrees == [0, 0, 1, 2, 3, 4, 5]
    assert degrees.count(0) == 2
    assert 6 not in degrees
    assert not is_cyclic_rotation(degrees)
    # and only the Christoffel mode gets it right
    working = [
        m.index
        for m in scale.modes()
        if set(_degrees_of(partial(7, 1), m)) == set(range(7))
    ]
    assert working == [christoffel_mode(scale).index] == [6]


# --------------------------------------------------------------------------- #
# Claim 4 -- higher partials and generic interval cycles
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_large,n_small,k", [(5, 2, 1), (5, 2, 2), (5, 2, 3),
                                               (5, 2, 4), (5, 2, 5), (5, 2, 6),
                                               (2, 5, 3), (4, 3, 5), (3, 4, 2)])
def test_claim4_coprime_partials_are_complete_cycles(n_large, n_small, k):
    scale = MOSScale.from_signature(n_large, n_small, tuning="central")
    n = scale.cardinality
    assert math.gcd(k, n) == 1
    degrees = _degrees_of(partial(n, k), christoffel_mode(scale))
    assert sorted(degrees) == list(range(n))
    # a complete generic interval cycle: consecutive events step by a constant
    # generic interval k
    assert all((degrees[(i + 1) % n] - degrees[i]) % n == k % n for i in range(n))


@pytest.mark.parametrize("k", list(range(13)))
def test_claim4_gcd_splits_the_cycle(k):
    """gcd(k, n) = d  =>  n/d distinct degrees, each struck d times."""
    scale = MOSScale.from_signature(5, 7, tuning="central")  # n = 12
    n = scale.cardinality
    assert n == 12
    degrees = _degrees_of(partial(n, k), christoffel_mode(scale))
    d = math.gcd(k, n)
    counts = {deg: degrees.count(deg) for deg in set(degrees)}
    assert len(counts) == n // d
    assert set(counts.values()) == {d}


def test_claim4_seven_note_branches():
    scale = MOSScale.from_signature(5, 2, tuning="central")
    mode = christoffel_mode(scale)
    for k in (1, 2, 3, 4, 5, 6):  # every k in 1..6 is coprime to 7
        assert sorted(_degrees_of(partial(7, k), mode)) == list(range(7))
    # k = 0 and k = 7 both collapse onto the root
    assert _degrees_of(partial(7, 0), mode) == [0] * 7
    assert _degrees_of(partial(7, 7), mode) == [0] * 7


def test_claim4_third_partial_is_a_cycle_of_thirds():
    mode = christoffel_mode(MOSScale.from_signature(5, 2, tuning=12))
    assert _degrees_of(partial(7, 3), mode) == [0, 3, 6, 2, 5, 1, 4]


# --------------------------------------------------------------------------- #
# Claim 6 -- zero_pad / prune / truncate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n,m", [(3, 3), (3, 5), (4, 16), (7, 12)])
def test_claim6_zero_pad_preserves_coefficients(n, m):
    s = random_state(n, seed=300 + n)
    padded = s.zero_pad(m)
    assert padded.n == m
    a, b = s.spectrum, padded.spectrum
    assert np.allclose(b[:n], a)
    assert np.allclose(b[n:], 0.0)


@pytest.mark.parametrize("n,m,k", [(4, 8, 1), (5, 15, 2), (7, 7, 3), (3, 9, 0)])
def test_claim6_zero_pad_maps_partial_to_partial(n, m, k):
    assert partial(n, k).zero_pad(m).allclose(partial(m, k))


def test_claim6_zero_pad_refuses_to_shrink():
    with pytest.raises(ValueError, match="only grows"):
        PlayState([1, 2, 3]).zero_pad(2)


def test_claim6_prune_keeps_the_loudest_coefficients():
    s = PlayState.from_spectrum([3.0, 0.1, 2.0, 0.2, 1.0])
    q = s.prune(2)
    assert q.n == 3
    assert np.allclose(np.abs(q.spectrum), [3.0, 2.0, 1.0])  # index order kept


def test_claim6_prune_drops_minimal_energy_and_keeps_the_rest():
    s = random_state(9, seed=42)
    a = s.spectrum
    energies = np.abs(a) ** 2
    for m in range(0, 9):
        q = s.prune(m)
        assert q.n == 9 - m
        keep = np.sort(np.argsort(energies, kind="stable")[m:])
        assert np.allclose(q.spectrum, a[keep])
        # the surviving energy is the largest achievable for that many keepers
        assert float(np.sum(np.abs(q.spectrum) ** 2)) == pytest.approx(
            float(np.sum(np.sort(energies)[m:]))
        )


@pytest.mark.parametrize("m", [1, 2, 5, 8, 9])
def test_claim6_truncate_is_the_complement_of_prune(m):
    s = random_state(9, seed=43)
    assert s.truncate(m).allclose(s.prune(9 - m))
    assert s.truncate(m).n == m


def test_claim6_prune_and_truncate_bounds():
    s = PlayState([1, 2, 3, 4])
    assert s.prune(0).allclose(s)
    assert s.truncate(4).allclose(s)
    with pytest.raises(ValueError):
        s.prune(4)
    with pytest.raises(ValueError):
        s.prune(-1)
    with pytest.raises(ValueError):
        s.truncate(0)
    with pytest.raises(ValueError):
        s.truncate(5)


def test_claim6_prune_then_zero_pad_is_lossy_but_dimension_restoring():
    s = random_state(8, seed=44)
    back = s.prune(3).zero_pad(8)
    assert back.n == 8
    assert not back.allclose(s)
    assert back.energy < s.energy  # energy was thrown away, never gained


# --------------------------------------------------------------------------- #
# Scratching
# --------------------------------------------------------------------------- #
def test_scratch_sets_magnitude_absolutely():
    s = random_state(6, seed=5)
    q = s.scratch(2, magnitude=4.0)
    assert abs(q.spectrum[2]) == pytest.approx(4.0)
    # phase is untouched
    assert np.angle(q.spectrum[2]) == pytest.approx(np.angle(s.spectrum[2]))


def test_scratch_scale_multiplies_magnitude():
    s = random_state(6, seed=6)
    before = abs(s.spectrum[3])
    q = s.scratch(3, scale=0.25)
    assert abs(q.spectrum[3]) == pytest.approx(0.25 * before)


def test_scratch_sets_and_rotates_phase():
    s = random_state(6, seed=7)
    q = s.scratch(1, phase=1.0)
    assert np.angle(q.spectrum[1]) == pytest.approx(1.0)
    r = s.scratch(1, rotate=0.5)
    assert np.angle(r.spectrum[1]) == pytest.approx(
        (np.angle(s.spectrum[1]) + 0.5 + math.pi) % TWO_PI - math.pi
    )


def test_scratch_touches_only_one_coefficient():
    s = random_state(8, seed=8)
    q = s.scratch(5, magnitude=2.0, rotate=0.3)
    a, b = s.spectrum, q.spectrum
    changed = [i for i in range(8) if abs(a[i] - b[i]) > 1e-12]
    assert changed == [5]
    # ...but every finger moved
    assert np.all(np.abs(s.f - q.f) > 1e-12)


def test_scratch_negative_index():
    s = random_state(6, seed=9)
    assert s.scratch(-1, magnitude=1.0).allclose(s.scratch(5, magnitude=1.0))


def test_scratch_rejects_conflicting_arguments():
    s = PlayState([1, 2, 3])
    with pytest.raises(ValueError, match="magnitude= sets"):
        s.scratch(0, magnitude=1.0, scale=2.0)
    with pytest.raises(ValueError, match="phase= sets"):
        s.scratch(0, phase=1.0, rotate=2.0)
    # the two *different* attributes combine fine
    s.scratch(0, magnitude=1.0, rotate=2.0)
    s.scratch(0, scale=1.0, phase=2.0)


def test_scratch_rejects_negative_magnitude_and_bad_index():
    s = PlayState([1, 2, 3])
    with pytest.raises(ValueError, match="non-negative"):
        s.scratch(0, magnitude=-1.0)
    with pytest.raises(IndexError, match="out of range"):
        s.scratch(3)
    with pytest.raises(IndexError):
        s.scratch(-4)
    with pytest.raises(TypeError, match="must be an integer"):
        s.scratch(1.5)


def test_scratch_zeroing_a_partials_own_coefficient_silences_it():
    p = partial(6, 2)
    assert p.scratch(2, magnitude=0.0).energy == pytest.approx(0.0, abs=1e-20)


def test_scratch_scale_minus_one_is_a_pi_rotation():
    s = random_state(5, seed=10)
    assert s.scratch(1, scale=-1.0).allclose(s.scratch(1, rotate=math.pi))


# --------------------------------------------------------------------------- #
# Whole-hand gestures
# --------------------------------------------------------------------------- #
def test_rotate_all_shifts_every_phase_equally():
    p = random_state(9, seed=11)
    delta = 0.7
    q = p.rotate_all(delta)
    assert np.allclose((q.phases - p.phases) % TWO_PI, delta)
    assert np.allclose(q.magnitudes, p.magnitudes)


def test_rotate_all_is_a_global_spectral_factor():
    p = random_state(7, seed=12)
    q = p.rotate_all(1.3)
    assert np.allclose(q.spectrum, p.spectrum * np.exp(1j * 1.3))
    assert q.rotate_all(-1.3).allclose(p)
    assert p.rotate_all(TWO_PI).allclose(p, atol=1e-9)


def test_interpolate_endpoints_and_midpoint():
    a, b = partial(6, 1), partial(6, 4)
    assert a.interpolate(b, 0.0).allclose(a)
    assert a.interpolate(b, 1.0).allclose(b)
    mid = a.interpolate(b, 0.5)
    assert np.allclose(np.abs(mid.spectrum), [0, 0.5, 0, 0, 0.5, 0])
    # linear in Fourier == linear in fingers, because the DFT is linear
    assert np.allclose(mid.f, 0.5 * (a.f + b.f))


def test_interpolate_requires_matching_dimensions():
    with pytest.raises(ValueError, match="matching finger counts"):
        partial(4, 1).interpolate(partial(5, 1), 0.5)


def test_scratch_sequence_is_absolute_not_cumulative():
    base = partial(4, 1)
    traj = scratch_sequence(base, 1, [0.0, math.pi / 2, math.pi], attr="rotate")
    assert len(traj) == 3
    assert traj[0].allclose(base)
    assert traj[2].allclose(base.rotate_all(math.pi))
    # every element is derived from `base`, not from its predecessor
    assert traj[1].allclose(base.scratch(1, rotate=math.pi / 2))


def test_scratch_sequence_sweeps_other_attributes():
    base = random_state(5, seed=13)
    mags = scratch_sequence(base, 2, [0.0, 1.0, 2.0], attr="magnitude")
    assert [round(float(abs(s.spectrum[2])), 9) for s in mags] == [0.0, 1.0, 2.0]
    with pytest.raises(ValueError, match="attr must be one of"):
        scratch_sequence(base, 0, [1.0], attr="loudness")


# --------------------------------------------------------------------------- #
# Events and frequencies
# --------------------------------------------------------------------------- #
def test_to_events_is_in_finger_order_and_carries_loudness():
    scale = MOSScale.from_signature(5, 2, tuning=12)
    state = PlayState.from_polar(
        [1.0, 0.5, 0.25], [0.0, TWO_PI * 0.3, TWO_PI * 0.8]
    )
    events = to_events(state, scale)
    assert [e.index for e in events] == [0, 1, 2]
    assert [round(e.loudness, 9) for e in events] == [1.0, 0.5, 0.25]
    assert all(isinstance(e, NoteEvent) for e in events)
    assert [e.degree for e in events] == [0, 1, 5]
    assert [round(e.cents, 6) for e in events] == [0.0, 200.0, 900.0]
    assert [round(e.phase, 9) for e in events] == [
        0.0, round(TWO_PI * 0.3, 9), round(TWO_PI * 0.8, 9)
    ]


def test_to_events_ratio_and_cents_come_from_the_scale():
    mode = MOSScale.from_signature(5, 2, tuning=12).mode(1)  # Ionian
    events = to_events(partial(7, 1), mode)
    for e in events:
        assert e.ratio == pytest.approx(mode.ratios[e.degree])
        assert e.cents == pytest.approx(mode.cents[e.degree])
        assert e.ratio == pytest.approx(2.0 ** (e.cents / 1200.0))


def test_to_events_allows_more_fingers_than_keys():
    scale = MOSScale.from_signature(2, 3, tuning=12)  # 5 keys
    events = to_events(partial(12, 1), scale)  # 12 fingers
    assert len(events) == 12
    assert set(e.degree for e in events) <= set(range(5))


def test_to_frequencies():
    scale = MOSScale.from_signature(2, 3, tuning=12)
    events = to_events(partial(5, 1), scale)
    freqs = to_frequencies(events, fund=200.0)
    assert freqs == pytest.approx(
        [200.0 * scale.ratios[e.degree] for e in events]
    )
    assert freqs[0] == pytest.approx(200.0)
    assert to_frequencies(events)[0] == pytest.approx(250.0)  # default fund
    with pytest.raises(ValueError, match="must be positive"):
        to_frequencies(events, fund=0.0)


def test_to_events_with_a_bare_degree_container_assumes_an_octave():
    class Bare:
        degrees = [0.0, 0.25, 0.5, 0.75]

    events = to_events(partial(4, 1), Bare())
    assert [e.degree for e in events] == [0, 1, 2, 3]
    assert [round(e.cents, 6) for e in events] == [0.0, 300.0, 600.0, 900.0]
    assert events[1].ratio == pytest.approx(2.0**0.25)


def test_to_events_bare_container_honours_a_declared_period():
    """The 2/1 is a *fallback*, not a hardcoding: a declared period wins."""

    class Tritave:
        degrees = [0.0, 0.25, 0.5, 0.75]
        period = 3.0

    events = to_events(partial(4, 1), Tritave())
    assert events[1].ratio == pytest.approx(3.0**0.25)
    assert events[1].cents == pytest.approx(0.25 * 1200.0 * math.log2(3.0))
    assert [e.cents for e in events] == pytest.approx(
        [0.0, 475.4888, 950.9775, 1426.4663], abs=1e-3
    )


@pytest.mark.parametrize("period", [2.0, 3.0, 1.5, 2.5])
def test_to_events_ratio_and_cents_never_disagree(period):
    """Whichever of ratios/cents is missing is derived from the other."""
    full = MOSScale.from_signature(5, 2, tuning=12, period=period).mode(6)

    class RatiosOnly:
        degrees = list(full.degrees)
        ratios = list(full.ratios)

    class CentsOnly:
        degrees = list(full.degrees)
        cents = list(full.cents)

    for obj in (full, RatiosOnly(), CentsOnly()):
        for e in to_events(partial(7, 1), obj):
            assert e.cents == pytest.approx(1200.0 * math.log2(e.ratio), abs=1e-9)
    # and both partial containers agree with the fully-specified scale
    assert [e.cents for e in to_events(partial(7, 1), RatiosOnly())] == pytest.approx(
        [e.cents for e in to_events(partial(7, 1), full)]
    )
    assert [e.ratio for e in to_events(partial(7, 1), CentsOnly())] == pytest.approx(
        [e.ratio for e in to_events(partial(7, 1), full)]
    )


# --------------------------------------------------------------------------- #
# Boundary tolerance -- rounding down makes one ulp cost a whole scale step
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", list(range(2, 41)))
def test_equal_divisions_quantise_to_the_identity(n):
    """Every finger of ``partial(n, 1)`` sits exactly on a tone of ``n``-EDO.

    ``exp``/``angle`` round-tripping the finger and multiplying the degree by
    ``2*pi`` are different computations, so for most ``n`` they disagree in the
    last ulp -- and since quantising rounds *down*, an unguarded comparison
    turns that ulp into a whole scale step (``n = 3`` gave ``[0, 1, 1]``).
    """

    class EqualDivision:
        degrees = [i / n for i in range(n)]

    assert _degrees_of(partial(n, 1), EqualDivision()) == list(range(n))


@pytest.mark.parametrize(
    "n_large,n_small,tuning",
    [(1, 2, 12), (2, 1, 12), (1, 3, 12), (5, 2, 7), (2, 3, 5), (1, 1, 2)],
)
def test_degenerate_mos_quantises_to_the_identity(n_large, n_small, tuning):
    """``L == s`` -- the degenerate edge of every MOS family -- is an EDO."""
    scale = MOSScale.from_signature(n_large, n_small, tuning=tuning)
    assert scale.hardness == pytest.approx(1.0)
    n = scale.cardinality
    assert _degrees_of(partial(n, 1), scale) == list(range(n))


def test_boundary_tolerance_is_ulp_scale_not_musical():
    """The tolerance must not swallow anything a listener could hear."""
    scale = MOSScale.from_signature(5, 2, tuning=12)
    # one ulp below degree 1's edge reads as degree 1 ...
    edge = TWO_PI * scale.degrees[1]
    assert phase_to_degree(math.nextafter(edge, 0.0), scale) == 1
    # ... but a hundredth of a cent below it is still degree 0
    hundredth_cent = TWO_PI * 0.01 / 1200.0
    assert phase_to_degree(edge - hundredth_cent, scale) == 0
    assert phase_to_degree(edge - 1e-9, scale) == 0
    # the same asymmetry at the wrap point: one ulp below a full turn is the
    # root, an audible hair below it is the top key
    assert phase_to_degree(math.nextafter(TWO_PI, 0.0), scale) == 0
    assert phase_to_degree(-1e-15, scale) == 0
    assert phase_to_degree(TWO_PI - hundredth_cent, scale) == len(scale.degrees) - 1
    assert phase_to_degree(-1e-6, scale) == len(scale.degrees) - 1


# --------------------------------------------------------------------------- #
# prune/truncate tie-breaking
# --------------------------------------------------------------------------- #
def test_prune_breaks_ties_toward_the_lower_index():
    """Equal energies: the lower-indexed coefficient survives.

    ``argsort(kind='stable')`` alone gets this backwards -- it puts the lower
    index first in the *ascending* order, i.e. first in line to be dropped.
    """
    s = PlayState.from_spectrum([1, 1j, 0, 0])
    assert s.truncate(1).spectrum[0] == pytest.approx(1 + 0j)
    assert s.prune(3).spectrum[0] == pytest.approx(1 + 0j)

    t = PlayState.from_spectrum([0, 2j, 2, 0])
    assert t.truncate(1).spectrum[0] == pytest.approx(2j)

    u = PlayState.from_spectrum([1, 1, 1, 1])
    assert np.allclose(u.truncate(2).spectrum, [1, 1])
    # all four tie: the two lowest indices survive, so the result is the DFT of
    # coefficients 0 and 1, not of 2 and 3 -- distinguishable via zero_pad
    assert u.truncate(2).allclose(PlayState.from_spectrum([1, 1]))
