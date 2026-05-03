"""Tests for biotuner.harmonic_geometry.inputs."""

from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from biotuner.harmonic_geometry.inputs import HarmonicInput, HarmonicSequence


# ---------------------------------------------------------------- HarmonicInput


class TestHarmonicInputConstruction:
    def test_from_ratios(self, simple_ratios):
        h = HarmonicInput(ratios=simple_ratios, base_freq=100.0)
        assert h.n_components() == 4
        np.testing.assert_allclose(h.to_peaks(), [100.0, 150.0, 125.0, 175.0])

    def test_from_peaks_fixture(self, eeg_like_input_kwargs):
        h = HarmonicInput(**eeg_like_input_kwargs)
        assert h.n_components() == 4
        np.testing.assert_allclose(h.to_peaks(), [2.5, 8.0, 12.5, 25.0])

    def test_requires_ratios_or_peaks(self):
        with pytest.raises(ValueError):
            HarmonicInput()

    def test_invalid_equave(self):
        with pytest.raises(ValueError):
            HarmonicInput(ratios=[1, 2], equave=1.0)

    def test_invalid_base_freq(self):
        with pytest.raises(ValueError):
            HarmonicInput(ratios=[1, 2], base_freq=0.0)

    def test_negative_amplitude(self):
        with pytest.raises(ValueError):
            HarmonicInput(ratios=[1, 2], amplitudes=[1.0, -0.5])

    def test_non_positive_peak(self):
        with pytest.raises(ValueError):
            HarmonicInput(peaks=[1.0, 0.0])

    def test_length_mismatch_amplitudes(self):
        with pytest.raises(ValueError):
            HarmonicInput(ratios=[1, 2, 3], amplitudes=[1.0, 1.0])

    def test_length_mismatch_phases(self):
        with pytest.raises(ValueError):
            HarmonicInput(peaks=[1.0, 2.0], phases=[0.0])

    def test_ratios_and_peaks_consistent(self):
        # base_freq=2 -> peaks should be [2, 3, 5/2, 4]
        h = HarmonicInput(
            ratios=[Fraction(1, 1), Fraction(3, 2), Fraction(5, 4), Fraction(2, 1)],
            peaks=[2.0, 3.0, 2.5, 4.0],
            base_freq=2.0,
        )
        assert h.n_components() == 4

    def test_ratios_and_peaks_inconsistent_raises(self):
        with pytest.raises(ValueError):
            HarmonicInput(
                ratios=[1, 2],
                peaks=[10.0, 99.0],  # would imply ratio 9.9, not 2
                base_freq=10.0,
            )


class TestHarmonicInputAccessors:
    def test_to_ratios_from_peaks(self):
        h = HarmonicInput(peaks=[2.0, 4.0, 8.0])
        # base_freq defaults to 1.0, so ratios = peaks
        ratios = h.to_ratios()
        assert ratios == [Fraction(2, 1), Fraction(4, 1), Fraction(8, 1)]

    def test_to_peaks_from_ratios_with_base_freq(self):
        h = HarmonicInput(ratios=[1, Fraction(3, 2)], base_freq=440.0)
        np.testing.assert_allclose(h.to_peaks(), [440.0, 660.0])

    def test_normalized_amplitudes_uniform_default(self):
        h = HarmonicInput(ratios=[1, 2, 3, 4])
        amps = h.normalized_amplitudes()
        np.testing.assert_allclose(amps, np.full(4, 0.25))

    def test_normalized_amplitudes_custom(self):
        h = HarmonicInput(ratios=[1, 2], amplitudes=[3.0, 1.0])
        amps = h.normalized_amplitudes()
        np.testing.assert_allclose(amps, [0.75, 0.25])


class TestHarmonicInputConvenienceConstructors:
    def test_from_ratios(self):
        h = HarmonicInput.from_ratios([1, 1.5, 2], base_freq=100.0)
        assert h.base_freq == 100.0
        assert isinstance(h.ratios[0], Fraction)

    def test_from_peaks_default_base_is_min(self):
        h = HarmonicInput.from_peaks([4.0, 8.0, 16.0])
        assert h.base_freq == 4.0
        np.testing.assert_allclose([float(r) for r in h.to_ratios()], [1.0, 2.0, 4.0])

    def test_from_peaks_explicit_base(self):
        h = HarmonicInput.from_peaks([10.0, 20.0], base_freq=5.0)
        assert h.base_freq == 5.0

    def test_from_peaks_empty_raises(self):
        with pytest.raises(ValueError):
            HarmonicInput.from_peaks([])


class TestHarmonicInputFromBiotuner:
    def test_full_object(self, mock_biotuner_obj):
        h = HarmonicInput.from_biotuner(mock_biotuner_obj)
        assert h.n_components() == 4
        np.testing.assert_allclose(h.to_peaks(), [4.0, 6.0, 8.0, 10.0])
        # base_freq should be the smallest peak.
        assert h.base_freq == 4.0
        # amplitudes carried over from .amps.
        np.testing.assert_allclose(
            h.amplitudes, [1.0, 0.8, 0.6, 0.4]
        )
        assert h.metadata.get("source") == "compute_biotuner"

    def test_object_without_ratios(self, mock_biotuner_obj_no_ratios):
        h = HarmonicInput.from_biotuner(mock_biotuner_obj_no_ratios)
        assert h.n_components() == 3
        np.testing.assert_allclose(h.to_peaks(), [3.0, 5.0, 7.0])

    def test_missing_peaks_attribute_raises(self):
        with pytest.raises(AttributeError):
            HarmonicInput.from_biotuner(SimpleNamespace())

    def test_empty_peaks_raises(self):
        bt = SimpleNamespace(peaks=np.array([]))
        with pytest.raises(ValueError):
            HarmonicInput.from_biotuner(bt)

    def test_mismatched_amps_are_dropped(self):
        # If amps length != peaks length we should not propagate them rather
        # than failing validation.
        bt = SimpleNamespace(
            peaks=np.array([2.0, 4.0]),
            amps=np.array([1.0, 0.5, 0.25]),  # wrong length
        )
        h = HarmonicInput.from_biotuner(bt)
        assert h.amplitudes is None


# -------------------------------------------------------------- HarmonicSequence


class TestHarmonicSequence:
    def test_construction(self):
        frames = [
            HarmonicInput(ratios=[1, 2]),
            HarmonicInput(ratios=[1, 3]),
        ]
        seq = HarmonicSequence(frames=frames)
        assert seq.n_frames() == 2

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            HarmonicSequence(frames=[])

    def test_times_length_mismatch_raises(self):
        frames = [HarmonicInput(ratios=[1, 2])]
        with pytest.raises(ValueError):
            HarmonicSequence(frames=frames, times=[0.0, 1.0])

    def test_times_must_be_non_decreasing(self):
        frames = [HarmonicInput(ratios=[1, 2])] * 2
        with pytest.raises(ValueError):
            HarmonicSequence(frames=frames, times=[1.0, 0.0])

    def test_at_returns_nearest(self):
        frames = [
            HarmonicInput(ratios=[1, 2]),
            HarmonicInput(ratios=[1, 3]),
            HarmonicInput(ratios=[1, 4]),
        ]
        seq = HarmonicSequence(frames=frames, times=[0.0, 1.0, 2.0])
        assert seq.at(0.0) is frames[0]
        assert seq.at(0.4) is frames[0]
        assert seq.at(0.6) is frames[1]
        assert seq.at(2.5) is frames[2]


class TestHarmonicSequenceInterpolate:
    def _two_frames(self):
        a = HarmonicInput(peaks=[100.0, 200.0])
        b = HarmonicInput(peaks=[200.0, 400.0])
        return HarmonicSequence(frames=[a, b], times=[0.0, 1.0])

    def test_endpoints(self):
        seq = self._two_frames()
        np.testing.assert_allclose(seq.interpolate(0.0).to_peaks(), [100.0, 200.0])
        np.testing.assert_allclose(seq.interpolate(1.0).to_peaks(), [200.0, 400.0])

    def test_log_midpoint_is_geometric_mean(self):
        seq = self._two_frames()
        mid = seq.interpolate(0.5, mode="log").to_peaks()
        # geometric mean of (100, 200) is sqrt(20000) ≈ 141.42; (200, 400) is ≈ 282.84.
        np.testing.assert_allclose(mid, [np.sqrt(100 * 200), np.sqrt(200 * 400)])

    def test_linear_midpoint(self):
        seq = self._two_frames()
        mid = seq.interpolate(0.5, mode="linear").to_peaks()
        np.testing.assert_allclose(mid, [150.0, 300.0])

    def test_nearest_mode(self):
        seq = self._two_frames()
        # 0.4 < 0.5, so nearest is frame 0
        result = seq.interpolate(0.4, mode="nearest")
        np.testing.assert_allclose(result.to_peaks(), [100.0, 200.0])

    def test_invalid_mode(self):
        seq = self._two_frames()
        with pytest.raises(ValueError):
            seq.interpolate(0.5, mode="cubic")

    def test_mismatched_components_raises(self):
        a = HarmonicInput(peaks=[100.0, 200.0])
        b = HarmonicInput(peaks=[100.0, 200.0, 300.0])
        seq = HarmonicSequence(frames=[a, b], times=[0.0, 1.0])
        with pytest.raises(ValueError):
            seq.interpolate(0.5)

    def test_clamps_outside_range(self):
        seq = self._two_frames()
        # Below first time -> first frame.
        np.testing.assert_allclose(
            seq.interpolate(-1.0).to_peaks(), [100.0, 200.0]
        )
        # Above last time -> last frame.
        np.testing.assert_allclose(
            seq.interpolate(99.0).to_peaks(), [200.0, 400.0]
        )


class TestHarmonicSequenceFromBiotuner:
    def test_from_biotuner_list(self, mock_biotuner_obj, mock_biotuner_obj_no_ratios):
        seq = HarmonicSequence.from_biotuner_list(
            [mock_biotuner_obj, mock_biotuner_obj_no_ratios]
        )
        assert seq.n_frames() == 2

    def test_skips_empty_objects(self, mock_biotuner_obj):
        empty = SimpleNamespace(peaks=np.array([]))
        seq = HarmonicSequence.from_biotuner_list([empty, mock_biotuner_obj, empty])
        assert seq.n_frames() == 1

    def test_all_empty_raises(self):
        empty = SimpleNamespace(peaks=np.array([]))
        with pytest.raises(ValueError):
            HarmonicSequence.from_biotuner_list([empty, empty])

    def test_times_length_mismatch_raises(self, mock_biotuner_obj):
        with pytest.raises(ValueError):
            HarmonicSequence.from_biotuner_list(
                [mock_biotuner_obj, mock_biotuner_obj], times=[0.0]
            )

    def test_from_biotuner_group(self, mock_biotuner_group):
        seq = HarmonicSequence.from_biotuner_group(mock_biotuner_group)
        assert seq.n_frames() == 2

    def test_from_biotuner_group_no_objects_raises(self):
        # store_objects=False would leave .objects=None
        btg = SimpleNamespace(objects=None)
        with pytest.raises(AttributeError):
            HarmonicSequence.from_biotuner_group(btg)

    def test_top_level_namespace(self):
        # The plan requires `from biotuner import harmonic_geometry` to work
        # in a clean install.
        import biotuner

        assert hasattr(biotuner, "harmonic_geometry")
        assert biotuner.harmonic_geometry.HarmonicInput is HarmonicInput
