"""Tests for ``Timbre.from_harmonic_input`` — the new HarmonicInput adapter.

Covers the field-by-field mapping (peaks→partials, amps→amplitudes,
linewidths→decay_times+bandwidths, aperiodic→tilt, flatness→noise_floor,
ratios_source→metadata), default behaviour for missing fields, override
precedence, and integration via the new ``bt.to_harmonic_input()``
convenience method.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from biotuner.harmonic_input import HarmonicInput
from biotuner.harmonic_timbre import Timbre


class TestFromHarmonicInputBasic:
    def test_peaks_become_partials(self):
        hi = HarmonicInput(peaks=[100.0, 200.0, 300.0])
        t = Timbre.from_harmonic_input(hi)
        np.testing.assert_allclose(t.partials_hz, [100.0, 200.0, 300.0])

    def test_default_uniform_amplitudes(self):
        # No amps supplied → uniform 1/n distribution.
        hi = HarmonicInput(peaks=[100.0, 200.0, 300.0])
        t = Timbre.from_harmonic_input(hi)
        assert t.amplitudes.shape == (3,)
        np.testing.assert_allclose(t.amplitudes, [1 / 3, 1 / 3, 1 / 3])

    def test_amplitudes_propagated(self):
        hi = HarmonicInput(
            peaks=[100.0, 200.0],
            amplitudes=[1.0, 0.5],
        )
        t = Timbre.from_harmonic_input(hi)
        np.testing.assert_allclose(t.amplitudes, [1.0, 0.5])

    def test_phases_propagated(self):
        hi = HarmonicInput(
            peaks=[100.0, 200.0],
            phases=[0.0, np.pi / 2],
        )
        t = Timbre.from_harmonic_input(hi)
        np.testing.assert_allclose(t.phases, [0.0, np.pi / 2])

    def test_matched_tuning_carries_ratios(self):
        # Bare constructor leaves base_freq=1.0; use from_peaks to get the
        # min-peak inference. The matched_tuning then reflects ratios
        # against that base.
        hi = HarmonicInput.from_peaks([100.0, 150.0, 200.0])
        t = Timbre.from_harmonic_input(hi)
        # Ratios derived from peaks/min_peak should be [1, 1.5, 2].
        assert t.matched_tuning == pytest.approx([1.0, 1.5, 2.0])

    def test_base_freq_carried(self):
        hi = HarmonicInput(peaks=[100.0, 200.0], base_freq=100.0)
        t = Timbre.from_harmonic_input(hi)
        assert t.base_freq == 100.0

    def test_ratios_only_input_resolves_partials(self):
        # No peaks; partials should be base_freq * ratios.
        hi = HarmonicInput(ratios=[1.0, 1.5, 2.0], base_freq=440.0)
        t = Timbre.from_harmonic_input(hi)
        np.testing.assert_allclose(t.partials_hz, [440.0, 660.0, 880.0])


class TestLinewidthsDerivation:
    def test_linewidths_become_decay_and_bandwidths(self):
        hi = HarmonicInput(
            peaks=[100.0, 200.0],
            linewidths=[2.0, 4.0],
        )
        t = Timbre.from_harmonic_input(hi)
        # decay = 1 / (π · linewidth)
        expected_decay = 1.0 / (np.pi * np.array([2.0, 4.0]))
        np.testing.assert_allclose(t.decay_times, expected_decay)
        # bandwidths preserved as raw linewidth values
        np.testing.assert_allclose(t.bandwidths, [2.0, 4.0])

    def test_zero_linewidth_yields_finite_decay(self):
        hi = HarmonicInput(
            peaks=[100.0, 200.0],
            linewidths=[2.0, 0.0],  # 0 means "very long decay"
        )
        t = Timbre.from_harmonic_input(hi)
        # First partial gets normal decay
        assert np.isfinite(t.decay_times[0])
        # Second gets clamped to a large finite value (1e6 sentinel)
        assert t.decay_times[1] == pytest.approx(1e6)

    def test_no_linewidths_means_no_decay_no_bandwidths(self):
        hi = HarmonicInput(peaks=[100.0, 200.0])
        t = Timbre.from_harmonic_input(hi)
        assert t.decay_times is None
        assert t.bandwidths is None


class TestScalarFieldsPropagation:
    def test_aperiodic_exponent_becomes_tilt(self):
        hi = HarmonicInput(peaks=[100.0], aperiodic_exponent=1.2)
        t = Timbre.from_harmonic_input(hi)
        assert t.spectral_tilt == 1.2

    def test_spectral_flatness_becomes_noise_floor(self):
        hi = HarmonicInput(peaks=[100.0], spectral_flatness=0.05)
        t = Timbre.from_harmonic_input(hi)
        assert t.noise_floor == 0.05

    def test_missing_scalars_stay_none(self):
        hi = HarmonicInput(peaks=[100.0])
        t = Timbre.from_harmonic_input(hi)
        assert t.spectral_tilt is None
        assert t.noise_floor is None


class TestMetadataPropagation:
    def test_scale_source_carried_to_metadata(self):
        hi = HarmonicInput(peaks=[100.0], ratios_source="HE")
        t = Timbre.from_harmonic_input(hi)
        assert t.metadata["scale_source"] == "HE"

    def test_from_harmonic_input_marker(self):
        hi = HarmonicInput(peaks=[100.0])
        t = Timbre.from_harmonic_input(hi)
        assert t.metadata.get("from_harmonic_input") is True

    def test_hi_metadata_merged_without_overwriting(self):
        hi = HarmonicInput(
            peaks=[100.0],
            metadata={"source": "compute_biotuner", "extra": "foo"},
        )
        t = Timbre.from_harmonic_input(hi)
        # scale_source is set by the adapter, NOT overwritten by hi metadata
        assert t.metadata["scale_source"] == "peaks"  # the default
        # Other HI metadata flows through
        assert t.metadata["extra"] == "foo"
        assert t.metadata["source"] == "compute_biotuner"

    def test_matching_method_marker(self):
        hi = HarmonicInput(peaks=[100.0])
        t = Timbre.from_harmonic_input(hi)
        assert t.matching_method == "harmonic_input"


class TestOverrides:
    def test_override_amplitudes(self):
        hi = HarmonicInput(peaks=[100.0, 200.0], amplitudes=[1.0, 0.5])
        t = Timbre.from_harmonic_input(
            hi, amplitudes=np.array([0.8, 0.3])
        )
        np.testing.assert_allclose(t.amplitudes, [0.8, 0.3])

    def test_override_spectral_tilt(self):
        hi = HarmonicInput(peaks=[100.0], aperiodic_exponent=1.2)
        t = Timbre.from_harmonic_input(hi, spectral_tilt=0.5)
        assert t.spectral_tilt == 0.5


class TestRoundTripFromBiotuner:
    """End-to-end: a duck-typed bt → HarmonicInput → Timbre."""

    def test_via_to_harmonic_input_convenience(self):
        # The compute_biotuner.to_harmonic_input() shim should produce a
        # HarmonicInput we can hand straight into Timbre.from_harmonic_input.
        from biotuner.biotuner_object import compute_biotuner

        # Build a minimal real compute_biotuner with peaks set manually
        # (avoids running the full peak extraction pipeline in this test).
        bt = compute_biotuner.__new__(compute_biotuner)
        bt.peaks = np.array([100.0, 200.0, 300.0])
        bt.amps = np.array([1.0, 0.6, 0.3])
        hi = bt.to_harmonic_input()
        assert isinstance(hi, HarmonicInput)
        t = Timbre.from_harmonic_input(hi)
        assert t.n_partials() == 3
        np.testing.assert_allclose(t.partials_hz, [100.0, 200.0, 300.0])
        np.testing.assert_allclose(t.amplitudes, [1.0, 0.6, 0.3])

    def test_to_harmonic_input_forwards_scale_priority(self):
        from biotuner.biotuner_object import compute_biotuner

        bt = compute_biotuner.__new__(compute_biotuner)
        bt.peaks = np.array([100.0, 200.0])
        bt.HE_scale = np.array([1.0, 1.5])
        hi = bt.to_harmonic_input(scale_priority=["HE", "peaks_ratios"])
        assert hi.ratios_source == "HE"
