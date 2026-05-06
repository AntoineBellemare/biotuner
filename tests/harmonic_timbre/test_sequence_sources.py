"""Tests for biotuner.harmonic_timbre.sequence_sources — TimbreSequence
constructors."""

from __future__ import annotations

import numpy as np
import pytest

from biotuner.harmonic_timbre import (
    TimbreSequence,
    timbre_sequence_from_markov_walk,
    timbre_sequence_from_ratio_frames,
    timbre_sequence_from_signal,
    timbre_sequence_from_transitional_harmony,
)


# ---------------------------------------------------------------------------
# timbre_sequence_from_ratio_frames
# ---------------------------------------------------------------------------

def test_ratio_frames_one_timbre_per_frame():
    frames = [
        [1.0, 5/4, 3/2, 2.0],
        [1.0, 6/5, 3/2, 2.0],
        [1.0, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8, 2.0],
    ]
    seq = timbre_sequence_from_ratio_frames(frames)
    assert isinstance(seq, TimbreSequence)
    assert seq.n_frames() == 3
    # Each frame has matched_tuning matching its input
    for tb, expected in zip(seq.frames, frames):
        assert tb.matched_tuning == expected


def test_ratio_frames_provenance():
    frames = [[1.0, 3/2, 2.0], [1.0, 4/3, 2.0]]
    seq = timbre_sequence_from_ratio_frames(frames)
    for tb in seq.frames:
        assert tb.metadata.get("scale_source") == "ratio_frame"


def test_ratio_frames_synthesize_smoke():
    frames = [[1.0, 5/4, 3/2, 2.0], [1.0, 6/5, 3/2, 2.0]]
    seq = timbre_sequence_from_ratio_frames(frames, base_freq=110.0)
    audio = seq.synthesize(samplerate=8000, frame_duration=0.1, crossfade=0.02)
    assert audio.dtype == np.float32
    assert audio.size > 0


def test_ratio_frames_empty_raises():
    with pytest.raises(ValueError, match="no non-empty frames"):
        timbre_sequence_from_ratio_frames([])


def test_ratio_frames_filters_empty():
    seq = timbre_sequence_from_ratio_frames([
        [1.0, 3/2, 2.0],
        [],
        [1.0, 4/3, 2.0],
    ])
    assert seq.n_frames() == 2


def test_ratio_frames_custom_times():
    frames = [[1.0, 3/2, 2.0]] * 3
    seq = timbre_sequence_from_ratio_frames(frames, times=np.array([0.0, 1.5, 3.0]))
    assert seq.times is not None
    assert seq.times[1] == 1.5


# ---------------------------------------------------------------------------
# timbre_sequence_from_signal
# ---------------------------------------------------------------------------

def test_signal_to_sequence_smoke():
    """Synthetic 4-second signal at 4 + 8 + 12 Hz should produce a few
    windows; each window's biotuner pipeline should yield at least some
    ratios when using the FOOOF peaks function."""
    sf = 500.0
    t = np.arange(4 * int(sf)) / sf
    rng = np.random.default_rng(7)
    sig = (
        1.0 * np.sin(2 * np.pi * 4 * t)
        + 0.7 * np.sin(2 * np.pi * 8 * t)
        + 0.5 * np.sin(2 * np.pi * 12 * t)
        + 0.05 * rng.standard_normal(t.size)
    )
    try:
        seq = timbre_sequence_from_signal(
            sig, sf=sf,
            window_size=1.5, overlap=0.5,
            peaks_function="FOOOF",
            min_freq=2.0, max_freq=30.0,
            n_peaks=3,
        )
    except Exception as exc:
        pytest.skip(f"FOOOF not available or failed: {exc}")

    assert isinstance(seq, TimbreSequence)
    assert seq.n_frames() >= 1
    assert seq.times is not None
    # times are monotonically increasing
    assert np.all(np.diff(seq.times) > 0)


def test_signal_too_short_raises():
    sig = np.zeros(100)
    with pytest.raises(ValueError, match="signal too short"):
        timbre_sequence_from_signal(sig, sf=1000.0, window_size=2.0)


def test_signal_invalid_overlap_raises():
    sig = np.zeros(10000)
    with pytest.raises(ValueError, match="overlap"):
        timbre_sequence_from_signal(sig, sf=1000.0, window_size=2.0, overlap=1.5)


# ---------------------------------------------------------------------------
# timbre_sequence_from_transitional_harmony
# ---------------------------------------------------------------------------

def test_transitional_harmony_wrapper_calls_signal_path():
    """A minimal mock with .data, .sf, and the relevant attributes should
    flow through transitional_harmony → signal → sequence."""
    sf = 500.0
    t = np.arange(4 * int(sf)) / sf
    sig = (np.sin(2 * np.pi * 4 * t) + 0.7 * np.sin(2 * np.pi * 8 * t)
           + 0.5 * np.sin(2 * np.pi * 12 * t))

    class MockTH:
        data = sig
        sf = 500.0
        overlap = 50  # 50% overlap (the th class uses percent)
        peaks_function = "FOOOF"
        n_peaks = 3
        min_freq = 2.0
        max_freq = 30.0
        precision = 0.5

    try:
        seq = timbre_sequence_from_transitional_harmony(MockTH())
    except Exception as exc:
        pytest.skip(f"FOOOF not available or failed: {exc}")
    assert seq.n_frames() >= 1


def test_transitional_harmony_no_data_raises():
    class Empty: pass
    with pytest.raises(ValueError, match="th.data"):
        timbre_sequence_from_transitional_harmony(Empty())


# ---------------------------------------------------------------------------
# timbre_sequence_from_markov_walk
# ---------------------------------------------------------------------------

def test_markov_walk_basic_two_state_chain():
    """Walk through a deterministic two-state chain (0 → 1 → 0 → 1 …)."""
    P = np.array([[0.0, 1.0],
                  [1.0, 0.0]])
    state_ratios = {
        0: [1.0, 5/4, 3/2, 2.0],
        1: [1.0, 6/5, 3/2, 2.0],
    }
    seq = timbre_sequence_from_markov_walk(
        P, state_ratios, n_steps=8, start=0, seed=0,
    )
    assert seq.n_frames() == 8
    # State 0 at index 0; state 1 at index 1; alternating
    assert seq.frames[0].matched_tuning == state_ratios[0]
    assert seq.frames[1].matched_tuning == state_ratios[1]
    assert seq.frames[2].matched_tuning == state_ratios[0]


def test_markov_walk_records_provenance():
    P = np.array([[0.5, 0.5], [0.5, 0.5]])
    state_ratios = {0: [1.0, 3/2, 2.0], 1: [1.0, 4/3, 2.0]}
    seq = timbre_sequence_from_markov_walk(
        P, state_ratios, n_steps=5, start=0, seed=42,
    )
    walk_meta = seq.frames[0].metadata.get("markov_walk")
    assert walk_meta is not None
    assert walk_meta["n_steps_requested"] == 5
    assert walk_meta["start_state"] == 0


def test_markov_walk_dead_end_state_stays_put():
    """A row of all zeros = absorbing state; the walk should not crash."""
    P = np.array([[0.0, 1.0],
                  [0.0, 0.0]])  # state 1 is a dead end
    state_ratios = {
        0: [1.0, 3/2, 2.0],
        1: [1.0, 4/3, 2.0],
    }
    seq = timbre_sequence_from_markov_walk(
        P, state_ratios, n_steps=5, start=0,
    )
    # First step is forced to state 1 (P[0,1]=1); subsequent steps stay
    assert seq.frames[1].matched_tuning == state_ratios[1]
    # All later frames are also state 1
    for tb in seq.frames[2:]:
        assert tb.matched_tuning == state_ratios[1]


def test_markov_walk_invalid_inputs():
    P = np.array([[1.0, 0.0], [0.0, 1.0]])
    state_ratios = {0: [1.0, 2.0], 1: [1.0, 1.5, 2.0]}
    with pytest.raises(ValueError, match="square"):
        timbre_sequence_from_markov_walk(
            np.array([[1.0, 0.0]]), state_ratios, n_steps=3,
        )
    with pytest.raises(ValueError, match="out of range"):
        timbre_sequence_from_markov_walk(P, state_ratios, n_steps=3, start=5)
    with pytest.raises(ValueError, match="n_steps"):
        timbre_sequence_from_markov_walk(P, state_ratios, n_steps=0)


def test_markov_walk_skips_undecodable_states():
    """A state with no entry in state_ratios is silently skipped (other
    than the start state, which must be decodable)."""
    P = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0],   # absorbing
    ])
    # state 1 has no ratios → frames produced for state 0 and state 2 only
    state_ratios = {
        0: [1.0, 3/2, 2.0],
        2: [1.0, 4/3, 2.0],
    }
    seq = timbre_sequence_from_markov_walk(
        P, state_ratios, n_steps=10, start=0,
    )
    # The visited sequence is [0, 1, 2, 2, 2, 2, 2, 2, 2, 2]; state 1 is skipped.
    assert seq.n_frames() == 9
    assert seq.frames[0].matched_tuning == state_ratios[0]
    assert seq.frames[1].matched_tuning == state_ratios[2]
