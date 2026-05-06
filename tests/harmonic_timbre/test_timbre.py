"""Tests for biotuner.harmonic_timbre.timbre — Timbre/TimbreSequence/Modulator."""

from __future__ import annotations

import os

import numpy as np
import pytest

from biotuner.harmonic_timbre import Modulator, Timbre, TimbreSequence


# ---------------------------------------------------------------------------
# Modulator
# ---------------------------------------------------------------------------

def test_modulator_roundtrip():
    m = Modulator(carrier_idx=2, mod_freq=5.0, depth=0.3, mod_type="AM",
                  phase=0.1, source="PAC_theta_gamma")
    d = m.to_dict()
    m2 = Modulator.from_dict(d)
    assert m == m2


# ---------------------------------------------------------------------------
# Timbre construction & validation
# ---------------------------------------------------------------------------

def test_timbre_minimal_construction():
    t = Timbre(
        partials_hz=[100.0, 200.0, 300.0],
        amplitudes=[1.0, 0.5, 0.25],
    )
    t.validate()
    assert t.n_partials() == 3
    assert t.partials_hz.dtype == np.float64
    assert t.amplitudes.dtype == np.float64


def test_timbre_validate_length_mismatch():
    with pytest.raises(ValueError, match="amplitudes length"):
        Timbre(
            partials_hz=[100.0, 200.0, 300.0],
            amplitudes=[1.0, 0.5],
        ).validate()


def test_timbre_validate_negative_freq():
    with pytest.raises(ValueError, match="strictly positive"):
        Timbre(
            partials_hz=[100.0, -200.0, 300.0],
            amplitudes=[1.0, 1.0, 1.0],
        ).validate()


def test_timbre_validate_per_partial_field_lengths():
    with pytest.raises(ValueError, match="phases length"):
        Timbre(
            partials_hz=[100.0, 200.0, 300.0],
            amplitudes=[1.0, 0.5, 0.25],
            phases=[0.0, np.pi],  # wrong length
        ).validate()


def test_timbre_validate_noise_floor_range():
    with pytest.raises(ValueError, match="noise_floor"):
        Timbre(
            partials_hz=[100.0],
            amplitudes=[1.0],
            noise_floor=1.5,
        ).validate()


def test_timbre_with_partials_immutability():
    t = Timbre(
        partials_hz=[100.0, 200.0],
        amplitudes=[1.0, 0.5],
        metadata={"origin": "alpha"},
    )
    t2 = t.with_partials(amplitudes=[0.7, 0.7])
    # original not mutated
    assert np.allclose(t.amplitudes, [1.0, 0.5])
    # copy correct
    assert np.allclose(t2.amplitudes, [0.7, 0.7])
    # metadata is a deep-ish copy (not shared dict)
    t2.metadata["origin"] = "beta"
    assert t.metadata["origin"] == "alpha"


def test_timbre_normalized_amplitudes_peaks_at_1():
    t = Timbre(
        partials_hz=[100.0, 200.0],
        amplitudes=[3.0, 1.0],
    )
    n = t.normalized_amplitudes()
    assert np.isclose(np.max(np.abs(n)), 1.0)


# ---------------------------------------------------------------------------
# Save / load roundtrip
# ---------------------------------------------------------------------------

def test_timbre_save_load_roundtrip(tmp_path):
    t = Timbre(
        partials_hz=[100.0, 200.0, 400.0],
        amplitudes=[1.0, 0.5, 0.25],
        phases=[0.0, np.pi / 2, np.pi],
        decay_times=[1.0, 0.5, 0.25],
        spectral_tilt=1.5,
        noise_floor=0.1,
        am_modulators=[Modulator(0, 4.0, 0.5, "AM", source="test")],
        palette=[(255, 0, 0), (0, 255, 0)],
        elements=["H", "O"],
        base_freq=100.0,
        matched_tuning=[1.0, 2.0, 4.0],
        matching_method="direct",
        metadata={"k": "v"},
    )
    stem = str(tmp_path / "tb")
    t.save(stem)
    assert os.path.exists(stem + ".json")
    assert os.path.exists(stem + ".npz")
    t2 = Timbre.load(stem)
    np.testing.assert_allclose(t.partials_hz, t2.partials_hz)
    np.testing.assert_allclose(t.amplitudes, t2.amplitudes)
    np.testing.assert_allclose(t.phases, t2.phases)
    np.testing.assert_allclose(t.decay_times, t2.decay_times)
    assert t2.spectral_tilt == t.spectral_tilt
    assert t2.noise_floor == t.noise_floor
    assert len(t2.am_modulators) == 1 and t2.am_modulators[0].source == "test"
    assert t2.palette == [(255, 0, 0), (0, 255, 0)]
    assert t2.elements == ["H", "O"]
    assert t2.matched_tuning == [1.0, 2.0, 4.0]
    assert t2.matching_method == "direct"
    assert t2.metadata == {"k": "v"}


# ---------------------------------------------------------------------------
# Synthesize dispatch
# ---------------------------------------------------------------------------

def test_timbre_synthesize_dispatches_additive():
    t = Timbre(
        partials_hz=[110.0, 220.0],
        amplitudes=[1.0, 0.5],
    )
    audio = t.synthesize(samplerate=22050, duration=0.1)
    assert audio.dtype == np.float32
    assert audio.shape == (2205,)
    assert np.max(np.abs(audio)) <= 1.0


def test_timbre_synthesize_uses_envelope_when_decay_set():
    t = Timbre(
        partials_hz=[110.0, 220.0],
        amplitudes=[1.0, 1.0],
        decay_times=[0.05, 0.05],
    )
    audio = t.synthesize(samplerate=22050, duration=0.5, normalize=False)
    # with τ=0.05 s, after 0.4 s amplitude should decay to ~exp(-8) ≈ 3e-4
    front = np.max(np.abs(audio[:1000]))
    tail = np.max(np.abs(audio[-1000:]))
    assert tail < front * 0.05


# ---------------------------------------------------------------------------
# TimbreSequence
# ---------------------------------------------------------------------------

def test_timbre_sequence_construction_and_at():
    t1 = Timbre(partials_hz=[100.0], amplitudes=[1.0])
    t2 = Timbre(partials_hz=[200.0], amplitudes=[1.0])
    seq = TimbreSequence(frames=[t1, t2])
    assert seq.n_frames() == 2
    # at(0) -> t1; at(1) -> t2 (uniform mode)
    assert seq.at(0.0) is t1
    assert seq.at(1.0) is t2


def test_timbre_sequence_synthesize_continuous():
    t1 = Timbre(partials_hz=[110.0], amplitudes=[1.0])
    t2 = Timbre(partials_hz=[220.0], amplitudes=[1.0])
    seq = TimbreSequence(frames=[t1, t2])
    audio = seq.synthesize(samplerate=22050, frame_duration=0.1, crossfade=0.02)
    expected_len = int(0.1 * 22050) * 2 - int(0.02 * 22050)
    assert audio.shape == (expected_len,)
    assert audio.dtype == np.float32
