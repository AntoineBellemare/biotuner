"""Tests for :mod:`biotuner.mos.modes`.

The headline claim is Milne et al. (2011) §4: the modes of a well-formed scale
are *parsimonious* -- neighbouring modes differ by a single tone, moved by the
augmented prime -- and the two mode transformations commute, generating a free
ℤ² of rank 2.  Both are checked here across many signatures, not just the
diatonic one the paper illustrates.
"""

import math

import pytest

from biotuner.mos.modes import (
    DIATONIC_MODE_NAMES,
    Mode,
    chain_order,
    mode_lattice,
    mode_names,
    parsimony_chain,
    wf_number,
)
from biotuner.mos.scale import MOSScale

FIFTH = math.log2(3 / 2)

SIGS = [(5, 2), (2, 5), (3, 4), (4, 3), (2, 3), (3, 2), (5, 7), (7, 5),
        (5, 6), (6, 5), (7, 4), (4, 7), (3, 8), (8, 3)]


def _sample_scales():
    for n_large, n_small in SIGS:
        for tuning in ("noble", "central"):
            yield MOSScale.from_signature(n_large, n_small, tuning=tuning)


# --------------------------------------------------------------------------- #
# Carey's WF(N, g)
# --------------------------------------------------------------------------- #
def test_wf_numbers_from_the_paper():
    """Milne et al. section 1: 'the diatonic scale and its inverse belong to
    WF(7, 2) while the chromatic scale and its inverse belong to WF(12, 7)'."""
    assert wf_number(FIFTH, 7) == 2
    assert wf_number(FIFTH, 12) == 7


def test_wf_number_is_coprime_to_the_cardinality():
    for s in _sample_scales():
        assert math.gcd(wf_number(s.generator, s.cardinality), s.cardinality) == 1


def test_chain_order_is_a_permutation():
    for s in _sample_scales():
        order = chain_order(s.generator, s.cardinality)
        assert sorted(order) == list(range(s.cardinality))


def test_chain_order_steps_by_the_wf_number():
    """One scale step up advances a fixed number of places along the generator
    chain -- that number is Carey's g."""
    for s in _sample_scales():
        n = s.cardinality
        wf = wf_number(s.generator, n)
        rank = chain_order(s.generator, n)
        inverse = [0] * n
        for chain_pos, sorted_pos in enumerate(rank):
            inverse[sorted_pos] = chain_pos
        for i in range(n):
            assert (inverse[(i + 1) % n] - inverse[i]) % n == wf


# --------------------------------------------------------------------------- #
# Mode basics
# --------------------------------------------------------------------------- #
def test_diatonic_mode_names_in_brightness_order():
    d = MOSScale.from_signature(5, 2, tuning=12)
    assert [m.name for m in d.modes()] == list(DIATONIC_MODE_NAMES)


def test_ionian_is_the_familiar_major_scale():
    d = MOSScale.from_signature(5, 2, tuning=12)
    ionian = d.mode(1)
    assert ionian.name == "Ionian"
    assert [round(c) for c in ionian.cents] == [0, 200, 400, 500, 700, 900, 1100]
    assert ionian.word == "LLsLLLs"


def test_unnamed_signatures_get_a_generic_label():
    s = MOSScale.from_signature(4, 3, tuning=19)
    assert s.mode(2).name == "mode 2 of 4L3s"
    assert mode_names(4, 3) is None


def test_mode_count_equals_cardinality():
    for s in _sample_scales():
        assert len(s.modes()) == s.cardinality


def test_mode_index_is_bounds_checked():
    d = MOSScale.from_signature(5, 2, tuning=12)
    with pytest.raises(ValueError, match=r"must lie in \[0, 7\)"):
        Mode(d, 7)
    with pytest.raises(ValueError, match=r"must lie in \[0, 7\)"):
        Mode(d, -1)


def test_the_stacked_scale_is_one_of_the_modes():
    """``MOSScale.word`` is the rotation rooted on the generator chain's origin.

    That is mode 0 only when stacking darkens; when it brightens, the chain
    origin is the *darkest* mode, so the two differ -- but the stacked scale is
    always some mode of the scale.
    """
    from biotuner.mos.modes import stacking_brightens

    for s in _sample_scales():
        words = {m.word for m in s.modes()}
        assert s.word in words
        expected = s.cardinality - 1 if stacking_brightens(s) else 0
        assert s.mode(expected).word == s.word
        assert s.mode(expected).cents == pytest.approx(s.cents)


def test_every_mode_is_a_rotation_of_the_same_word():
    for s in _sample_scales():
        base = s.word
        rotations = {base[k:] + base[:k] for k in range(s.cardinality)}
        assert {m.word for m in s.modes()} == rotations


def test_modes_are_rooted_and_ordered():
    for s in _sample_scales():
        for m in s.modes():
            assert m.degrees[0] == 0.0
            assert m.degrees == sorted(m.degrees)
            assert len(set(m.degrees)) == m.cardinality


# --------------------------------------------------------------------------- #
# Brightness
# --------------------------------------------------------------------------- #
def test_brightness_decreases_strictly_with_index():
    for s in _sample_scales():
        vals = [m.brightness for m in s.modes()]
        assert all(a > b for a, b in zip(vals, vals[1:]))


def test_index_zero_is_the_brightest_mode():
    for s in _sample_scales():
        modes = s.modes()
        assert modes[0].brightness == max(m.brightness for m in modes)


# --------------------------------------------------------------------------- #
# Parsimony -- Milne et al. section 4
# --------------------------------------------------------------------------- #
def test_adjacent_modes_differ_by_exactly_one_tone():
    """'one mode can be transformed into another by the replacement of a single
    tone ... by the augmented prime'."""
    for s in _sample_scales():
        for _, _, moved in parsimony_chain(s):
            assert len(moved) == 1


def test_the_moving_tone_moves_by_exactly_the_chroma():
    for s in _sample_scales():
        chroma = s.mode(0).chroma
        for _, _, moved in parsimony_chain(s):
            _, mine, theirs = moved[0]
            assert mine - theirs == pytest.approx(chroma, abs=1e-7)


def test_a_different_tone_moves_at_every_step():
    """Walking the brightness order touches each alterable degree once."""
    for s in _sample_scales():
        steps = [moved[0][0] for _, _, moved in parsimony_chain(s)]
        assert len(set(steps)) == len(steps)


def test_chroma_is_the_difference_of_the_step_sizes():
    for s in _sample_scales():
        large, small = s.step_cents
        assert s.mode(0).chroma == pytest.approx(large - small)


# --------------------------------------------------------------------------- #
# The two transformations
# --------------------------------------------------------------------------- #
def test_brighten_and_darken_are_inverses():
    for s in _sample_scales():
        for k in range(s.cardinality):
            assert Mode(s, k).brighten().darken().index == k


def test_brighten_by_the_cardinality_is_the_identity():
    for s in _sample_scales():
        m = Mode(s, 0)
        assert m.brighten(s.cardinality).index == 0
        assert m.rotate(s.cardinality).index == 0


def test_transformations_from_the_paper():
    """Milne et al. Fig. 7: C-Ionian -> D-Dorian (common origin, sigma) and
    C-Ionian -> C-Lydian (common finalis, tau)."""
    d = MOSScale.from_signature(5, 2, tuning=12)
    ionian = d.mode(1)
    assert ionian.rotate().name == "Dorian"
    assert ionian.brighten().name == "Lydian"


def test_sigma_and_tau_commute():
    """The modal universe is 'freely generated from two basic and commuting
    transformations' (Milne et al. section 4)."""
    for s in _sample_scales():
        for base in range(s.cardinality):
            m = Mode(s, base)
            for i in (1, 2, 3):
                for j in (1, 2, 3):
                    assert (
                        m.rotate(j).brighten(i).index
                        == m.brighten(i).rotate(j).index
                    )


def test_rotate_preserves_the_pitch_collection():
    """Diatonic transposition keeps the same notes, only the finalis moves."""
    for s in _sample_scales():
        base = Mode(s, 0)
        absolute = sorted((d + base.root_degree) % 1.0 for d in base.degrees)
        for k in range(1, s.cardinality):
            other = base.rotate(k)
            moved = sorted((d + other.root_degree) % 1.0 for d in other.degrees)
            assert moved == pytest.approx(absolute, abs=1e-9)


def test_brighten_changes_the_pitch_collection():
    """Chromatic transposition keeps the finalis and alters a note."""
    d = MOSScale.from_signature(5, 2, tuning=12)
    ionian, lydian = d.mode(1), d.mode(0)
    assert ionian.degrees[0] == lydian.degrees[0] == 0.0
    assert ionian.cents != lydian.cents


# --------------------------------------------------------------------------- #
# The lattice
# --------------------------------------------------------------------------- #
def test_lattice_coords_are_consecutive_generators_around_the_finalis():
    for s in _sample_scales():
        for index in range(s.cardinality):
            m = Mode(s, index)
            coords = m.lattice_coords()
            widths = [w for w, _ in coords]
            # Consecutive integers, containing 0 -- the finalis itself.
            assert widths == list(range(widths[0], widths[0] + s.cardinality))
            assert (0, 0) in coords
            # Height only ever drops, by at most one period per generator step.
            heights = [h for _, h in coords]
            assert all(0 >= a - b >= -1 for a, b in zip(heights[1:], heights))


def test_the_extreme_modes_sit_at_the_ends_of_the_chain():
    """The brightest and darkest modes are the chain's two endpoints.

    Their frames lie wholly on one side of the finalis, and on opposite sides
    from each other.  *Which* side is brightest depends on the direction the
    generator walks the modes, so the test asserts the pairing rather than a
    fixed sign.
    """
    for s in _sample_scales():
        brightest = [w for w, _ in Mode(s, 0).lattice_coords()]
        darkest = [w for w, _ in Mode(s, s.cardinality - 1).lattice_coords()]
        assert (min(brightest) == 0) != (min(darkest) == 0)
        assert (max(brightest) == 0) != (max(darkest) == 0)
        assert 0 in (min(brightest), max(brightest))
        assert 0 in (min(darkest), max(darkest))


def test_lattice_coords_reconstruct_the_mode():
    for s in _sample_scales():
        for index in range(s.cardinality):
            m = Mode(s, index)
            rebuilt = sorted(
                (w * s.generator + h) % 1.0 for w, h in m.lattice_coords()
            )
            assert rebuilt == pytest.approx(m.degrees, abs=1e-9)


def test_mode_lattice_is_a_commuting_grid():
    d = MOSScale.from_signature(5, 2, tuning=12)
    grid = mode_lattice(d, width=3, height=2)
    assert [[m.name for m in row] for row in grid] == [
        ["Lydian", "Ionian", "Mixolydian"],
        ["Mixolydian", "Dorian", "Aeolian"],
    ]


def test_mode_lattice_shape():
    d = MOSScale.from_signature(4, 3, tuning=19)
    grid = mode_lattice(d, width=5, height=4)
    assert len(grid) == 4 and all(len(r) == 5 for r in grid)


def test_mode_to_dict_round_trips_the_essentials():
    m = MOSScale.from_signature(5, 2, tuning=12).mode(1)
    d = m.to_dict()
    assert d["name"] == "Ionian"
    assert d["word"] == "LLsLLLs"
    assert d["signature"] == "5L2s"
    assert len(d["cents"]) == 7
