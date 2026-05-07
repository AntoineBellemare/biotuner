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


# ---------------------------------------------------------------------------
# Spectral enrichment transforms (with_intermod_sidebands, with_harmonic_stack,
# with_phase_mode, with_formant, with_slight_detune)
# ---------------------------------------------------------------------------

class _StubBt:
    """Minimal biotuner stand-in carrying only ``endogenous_intermodulations``."""

    def __init__(self, intermod):
        self.endogenous_intermodulations = intermod


def _base_timbre():
    return Timbre(
        partials_hz=np.array([100.0, 200.0, 300.0, 400.0]),
        amplitudes=np.array([1.0, 0.5, 0.3, 0.2]),
        base_freq=100.0,
    )


def test_with_intermod_sidebands_adds_sum_and_diff_partials():
    t = _base_timbre()
    bt = _StubBt([(100.0, 200.0)])  # 1:2 integer ratio
    t2 = t.with_intermod_sidebands(bt, depth=0.5, integer_ratio_only=True)
    # Expect partials at 100+200=300 (already present, but appended) and |100-200|=100
    expected_extras = {300.0, 100.0}
    new_partials = set(t2.partials_hz.tolist())
    assert expected_extras.issubset(new_partials)
    # Sideband amplitude is depth * min(amp_f1, amp_f2) = 0.5 * 0.5 = 0.25
    sidebands = t2.amplitudes[t.n_partials():]
    assert np.allclose(sidebands, 0.25)


def test_with_intermod_sidebands_drops_non_integer_ratio_pairs():
    t = _base_timbre()
    bt = _StubBt([(100.0, 173.0)])  # 1.73 ≠ near integer
    t2 = t.with_intermod_sidebands(bt, integer_ratio_only=True)
    # Pair rejected → unchanged
    assert t2.n_partials() == t.n_partials()


def test_with_intermod_sidebands_returns_self_when_attribute_missing():
    t = _base_timbre()
    class _NoBt: pass
    assert t.with_intermod_sidebands(_NoBt()) is t


def test_with_intermod_sidebands_bandlimits():
    # Force a sideband above max_bin*base_freq → it should be dropped.
    # base_freq=10, max_bin=5 → cap at 50 Hz. f1+f2=300 must be dropped.
    t = Timbre(
        partials_hz=np.array([100.0, 200.0]),
        amplitudes=np.array([1.0, 1.0]),
        base_freq=10.0,
    )
    bt = _StubBt([(100.0, 200.0)])
    t2 = t.with_intermod_sidebands(bt, max_bin=5)
    assert np.all(t2.partials_hz / 10.0 <= 5.0)


def test_with_harmonic_stack_adds_overtones():
    t = _base_timbre()
    t2 = t.with_harmonic_stack(n=3, rolloff=1.0)
    # Originals + 3 overtones per partial = 4 + 12 = 16
    assert t2.n_partials() == 16
    # Overtone amplitudes follow 1/h: amp[100]/2, amp[100]/3, amp[100]/4
    new_partials = t2.partials_hz.tolist()
    assert 200.0 in new_partials  # 2 * 100 (overtone)
    assert 300.0 in new_partials  # 3 * 100 (overtone)


def test_with_harmonic_stack_rolloff_attenuates_higher_overtones():
    t = Timbre(partials_hz=[100.0], amplitudes=[1.0], base_freq=100.0)
    t2 = t.with_harmonic_stack(n=3, rolloff=1.0)
    # amps for 200, 300, 400 should be 1/2, 1/3, 1/4
    assert t2.amplitudes[1] > t2.amplitudes[2] > t2.amplitudes[3]


def test_with_harmonic_stack_n_zero_is_noop():
    t = _base_timbre()
    assert t.with_harmonic_stack(n=0).n_partials() == t.n_partials()


def test_with_phase_mode_schroeder_zero_for_first_partial():
    t = _base_timbre()
    t2 = t.with_phase_mode("schroeder")
    # phi_1 = -π·1·0/N = 0
    assert t2.phases[0] == 0.0
    # Phases are wrapped into [0, 2π)
    assert np.all(t2.phases >= 0)
    assert np.all(t2.phases < 2 * np.pi)


def test_with_phase_mode_cosine_zeros_all():
    t = _base_timbre()
    t2 = t.with_phase_mode("cosine")
    assert np.all(t2.phases == 0.0)


def test_with_phase_mode_random_seeded_deterministic():
    t = _base_timbre()
    a = t.with_phase_mode("random", seed=42)
    b = t.with_phase_mode("random", seed=42)
    np.testing.assert_array_equal(a.phases, b.phases)


def test_with_phase_mode_biosignal_is_noop():
    t = _base_timbre().with_partials(phases=np.array([0.1, 0.2, 0.3, 0.4]))
    t2 = t.with_phase_mode("biosignal")
    np.testing.assert_array_equal(t2.phases, t.phases)


def test_with_phase_mode_invalid_raises():
    with pytest.raises(ValueError, match="phase mode"):
        _base_timbre().with_phase_mode("nope")


def test_with_formant_boosts_partials_near_center():
    t = _base_timbre()
    t2 = t.with_formant(center_hz=200.0, width_hz=50.0, gain_db=12.0)
    # Partial at 200 should be boosted; 100 and 400 less so
    ratio_at_center = t2.amplitudes[1] / t.amplitudes[1]
    ratio_far = t2.amplitudes[3] / t.amplitudes[3]
    assert ratio_at_center > ratio_far
    # 12 dB at center ≈ 4× linear gain
    assert ratio_at_center > 3.5


def test_with_formant_invalid_width_raises():
    with pytest.raises(ValueError):
        _base_timbre().with_formant(width_hz=0.0)


def test_with_slight_detune_only_perturbs_n_partials():
    t = _base_timbre()
    t2 = t.with_slight_detune(percent=1.0, n_partials=2, seed=0)
    # Exactly 2 partials should differ from the original
    diffs = np.sum(np.abs(t2.partials_hz - t.partials_hz) > 1e-9)
    assert diffs == 2


def test_with_slight_detune_magnitude_within_percent():
    t = _base_timbre()
    t2 = t.with_slight_detune(percent=2.0, n_partials=4, seed=0)
    rel = np.abs(t2.partials_hz - t.partials_hz) / t.partials_hz
    assert np.all(rel <= 0.02 + 1e-9)  # ≤ 2% tolerance


def test_with_slight_detune_seeded_deterministic():
    t = _base_timbre()
    a = t.with_slight_detune(percent=1.0, n_partials=3, seed=7)
    b = t.with_slight_detune(percent=1.0, n_partials=3, seed=7)
    np.testing.assert_array_equal(a.partials_hz, b.partials_hz)


def test_transforms_chain_cleanly():
    """The five transforms should compose without coercion errors."""
    t = _base_timbre()
    bt = _StubBt([(100.0, 200.0)])
    chained = (
        t.with_harmonic_stack(n=2, rolloff=0.9)
         .with_intermod_sidebands(bt, depth=0.3)
         .with_formant(center_hz=2000.0, width_hz=600.0, gain_db=4.0)
         .with_phase_mode("schroeder")
         .with_slight_detune(percent=0.5, n_partials=2, seed=1)
    )
    chained.validate()
    assert chained.n_partials() >= t.n_partials()
    # Phases set & well-formed
    assert chained.phases is not None
    assert chained.phases.shape == chained.partials_hz.shape
