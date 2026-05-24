"""Tests for the Tier-A HarmonicInput extension.

Covers the new fields (linewidths, freqs/psd, spectrum_method,
aperiodic_exponent, spectral_flatness, ratios_source, ratios_alternates),
their validation, the new ``scale_priority`` selection mode in
:meth:`HarmonicInput.from_biotuner`, and the canonical SCALE_ATTRS
vocabulary.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from biotuner.harmonic_geometry.inputs import (
    HarmonicInput,
    SCALE_ATTRS,
    SCALE_KEYS,
)


# ---------------------------------------------------------------------------
# Field-level construction + validation
# ---------------------------------------------------------------------------


class TestTierAFields:
    def test_linewidths_attach_to_components(self):
        h = HarmonicInput(
            peaks=[100.0, 200.0, 300.0],
            linewidths=[2.0, 4.0, 6.0],
        )
        assert h.linewidths == [2.0, 4.0, 6.0]

    def test_linewidths_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="linewidths"):
            HarmonicInput(peaks=[1.0, 2.0], linewidths=[1.0])

    def test_negative_linewidth_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            HarmonicInput(peaks=[1.0, 2.0], linewidths=[1.0, -0.5])

    def test_freqs_and_psd_pair(self):
        freqs = list(np.linspace(0, 100, 50))
        psd = list(np.random.RandomState(0).rand(50))
        h = HarmonicInput(peaks=[10.0], freqs=freqs, psd=psd)
        assert len(h.freqs) == 50 and len(h.psd) == 50

    def test_freqs_without_psd_raises(self):
        with pytest.raises(ValueError, match="freqs.*psd"):
            HarmonicInput(peaks=[1.0], freqs=[0.0, 1.0, 2.0])

    def test_psd_without_freqs_raises(self):
        with pytest.raises(ValueError, match="freqs.*psd"):
            HarmonicInput(peaks=[1.0], psd=[0.1, 0.2])

    def test_freqs_psd_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="matching length"):
            HarmonicInput(peaks=[1.0], freqs=[0.0, 1.0], psd=[0.1])

    def test_spectrum_method_is_str(self):
        h = HarmonicInput(peaks=[1.0], spectrum_method="multitaper")
        assert h.spectrum_method == "multitaper"

    def test_aperiodic_exponent_scalar(self):
        h = HarmonicInput(peaks=[1.0], aperiodic_exponent=1.2)
        assert h.aperiodic_exponent == 1.2

    def test_spectral_flatness_in_range(self):
        h = HarmonicInput(peaks=[1.0], spectral_flatness=0.3)
        assert h.spectral_flatness == 0.3

    def test_spectral_flatness_out_of_range_raises(self):
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            HarmonicInput(peaks=[1.0], spectral_flatness=1.5)

    def test_ratios_source_default(self):
        h = HarmonicInput(peaks=[1.0])
        assert h.ratios_source == "peaks"

    def test_ratios_alternates_default_empty_dict(self):
        h = HarmonicInput(peaks=[1.0])
        assert h.ratios_alternates == {}

    def test_ratios_alternates_drops_empty_entries(self):
        h = HarmonicInput(
            peaks=[1.0],
            ratios_alternates={"a": [1.0, 1.5], "b": [], "c": None},
        )
        assert "a" in h.ratios_alternates
        assert "b" not in h.ratios_alternates
        assert "c" not in h.ratios_alternates


# ---------------------------------------------------------------------------
# SCALE_ATTRS vocabulary
# ---------------------------------------------------------------------------


class TestScaleVocabulary:
    def test_scale_attrs_is_list_of_pairs(self):
        for entry in SCALE_ATTRS:
            assert len(entry) == 2
            assert isinstance(entry[0], str) and isinstance(entry[1], str)

    def test_scale_keys_matches_scale_attrs(self):
        assert SCALE_KEYS == [k for k, _ in SCALE_ATTRS]

    def test_canonical_keys_are_unique(self):
        assert len(SCALE_KEYS) == len(set(SCALE_KEYS))

    def test_peaks_ratios_cons_first(self):
        # The consonance-filtered ratios are the highest-confidence scale
        # source on a fitted biotuner. They should be the default top
        # priority — checked here so we don't silently lose that.
        assert SCALE_KEYS[0] == "peaks_ratios_cons"


# ---------------------------------------------------------------------------
# from_biotuner with scale_priority
# ---------------------------------------------------------------------------


class TestFromBiotunerScalePriority:
    @pytest.fixture
    def rich_bt(self):
        """A duck-typed bt with several scale variants populated."""
        peaks = np.array([4.0, 6.0, 8.0, 10.0])
        return SimpleNamespace(
            peaks=peaks,
            amps=np.array([1.0, 0.8, 0.6, 0.4]),
            peaks_ratios=peaks / peaks.min(),
            peaks_ratios_cons=np.array([1.5, 2.0, 2.5]),  # different length
            HE_scale=np.array([1.0, 1.25, 1.5, 2.0]),
            euler_fokker=np.array([1.0, 9 / 8, 5 / 4, 3 / 2]),
        )

    def test_legacy_call_unchanged(self, rich_bt):
        # No scale_priority → behaves exactly like before: uses peaks
        # (and peaks_ratios when length-aligned). source == "peaks".
        h = HarmonicInput.from_biotuner(rich_bt)
        assert h.ratios_source == "peaks"
        assert h.ratios_alternates == {}  # legacy path doesn't populate
        assert h.n_components() == 4

    def test_picks_first_non_empty(self, rich_bt):
        h = HarmonicInput.from_biotuner(
            rich_bt, scale_priority=["HE", "peaks_ratios_cons", "peaks_ratios"]
        )
        assert h.ratios_source == "HE"
        assert h.n_components() == 4

    def test_falls_through_to_second_when_first_missing(self):
        bt = SimpleNamespace(
            peaks=np.array([1.0, 2.0]),
            HE_scale=np.array([1.0, 1.5]),
            # no peaks_ratios_cons
        )
        h = HarmonicInput.from_biotuner(
            bt, scale_priority=["peaks_ratios_cons", "HE"]
        )
        assert h.ratios_source == "HE"

    def test_unknown_key_raises(self, rich_bt):
        with pytest.raises(ValueError, match="Unknown scale_priority"):
            HarmonicInput.from_biotuner(
                rich_bt, scale_priority=["bogus_scale_name"]
            )

    def test_alternates_populated(self, rich_bt):
        h = HarmonicInput.from_biotuner(rich_bt, scale_priority=["HE"])
        # HE is the canonical; the others should land in alternates.
        assert "HE" not in h.ratios_alternates  # not in alternates
        assert "peaks_ratios_cons" in h.ratios_alternates
        assert "peaks_ratios" in h.ratios_alternates
        assert "euler_fokker" in h.ratios_alternates
        assert h.ratios_alternates["euler_fokker"] == pytest.approx(
            [1.0, 9 / 8, 5 / 4, 3 / 2]
        )

    def test_alternates_off(self, rich_bt):
        h = HarmonicInput.from_biotuner(
            rich_bt, scale_priority=["HE"], include_alternates=False
        )
        assert h.ratios_alternates == {}

    def test_amps_dropped_when_chosen_scale_mismatched(self, rich_bt):
        # peaks_ratios_cons has length 3, amps has length 4 → drop amps.
        h = HarmonicInput.from_biotuner(
            rich_bt, scale_priority=["peaks_ratios_cons"]
        )
        assert h.amplitudes is None
        assert h.n_components() == 3


# ---------------------------------------------------------------------------
# from_biotuner Tier-A field extraction
# ---------------------------------------------------------------------------


class TestFromBiotunerTierAExtraction:
    def test_extracts_spectrum_when_present(self):
        freqs = np.linspace(0, 50, 100)
        psd = np.random.RandomState(0).rand(100)
        bt = SimpleNamespace(
            peaks=np.array([10.0, 20.0]),
            freqs=freqs,
            psd=psd,
            spectrum_method="welch",
        )
        h = HarmonicInput.from_biotuner(bt)
        assert h.freqs is not None and len(h.freqs) == 100
        assert h.psd is not None and len(h.psd) == 100
        assert h.spectrum_method == "welch"

    def test_skips_spectrum_when_only_one_present(self):
        bt = SimpleNamespace(
            peaks=np.array([10.0]),
            freqs=np.array([1.0, 2.0]),
            # no psd
        )
        h = HarmonicInput.from_biotuner(bt)
        assert h.freqs is None and h.psd is None

    def test_include_spectrum_false_skips(self):
        bt = SimpleNamespace(
            peaks=np.array([10.0]),
            freqs=np.array([1.0, 2.0]),
            psd=np.array([0.1, 0.2]),
        )
        h = HarmonicInput.from_biotuner(bt, include_spectrum=False)
        assert h.freqs is None and h.psd is None

    def test_extracts_fooof_aperiodic_exponent_direct(self):
        bt = SimpleNamespace(
            peaks=np.array([10.0]),
            aperiodic_exponent=1.7,
        )
        h = HarmonicInput.from_biotuner(bt)
        assert h.aperiodic_exponent == 1.7

    def test_extracts_fooof_aperiodic_exponent_from_params(self):
        bt = SimpleNamespace(
            peaks=np.array([10.0]),
            aperiodic_params=(0.5, 1.2),  # (offset, exponent)
        )
        h = HarmonicInput.from_biotuner(bt)
        assert h.aperiodic_exponent == 1.2

    def test_linewidths_attached_when_length_matches(self):
        bt = SimpleNamespace(
            peaks=np.array([10.0, 20.0]),
            linewidth=np.array([1.5, 2.5]),
        )
        h = HarmonicInput.from_biotuner(bt)
        assert h.linewidths == [1.5, 2.5]

    def test_linewidths_skipped_when_length_mismatch(self):
        bt = SimpleNamespace(
            peaks=np.array([10.0, 20.0]),
            linewidth=np.array([1.5, 2.5, 3.5]),
        )
        h = HarmonicInput.from_biotuner(bt)
        assert h.linewidths is None

    def test_extracts_spectral_flatness(self):
        bt = SimpleNamespace(
            peaks=np.array([10.0]),
            spectral_flatness=0.42,
        )
        h = HarmonicInput.from_biotuner(bt)
        assert h.spectral_flatness == 0.42

    def test_spectral_flatness_clipped(self):
        bt = SimpleNamespace(
            peaks=np.array([10.0]),
            spectral_flatness=1.7,  # out of range, should clip to 1.0
        )
        h = HarmonicInput.from_biotuner(bt)
        assert h.spectral_flatness == 1.0

    def test_falls_back_to_spectral_entropy(self):
        bt = SimpleNamespace(
            peaks=np.array([10.0]),
            spectral_entropy=0.6,
        )
        h = HarmonicInput.from_biotuner(bt)
        assert h.spectral_flatness == 0.6


# ---------------------------------------------------------------------------
# Top-level re-export
# ---------------------------------------------------------------------------


class TestTopLevelReExport:
    def test_top_level_import_works(self):
        from biotuner.harmonic_input import HarmonicInput as HI_top
        from biotuner.harmonic_geometry.inputs import HarmonicInput as HI_orig
        assert HI_top is HI_orig

    def test_scale_attrs_exposed_at_top_level(self):
        from biotuner.harmonic_input import SCALE_ATTRS as TOP
        from biotuner.harmonic_geometry.inputs import SCALE_ATTRS as ORIG
        assert TOP is ORIG

    def test_harmonic_sequence_at_top_level(self):
        from biotuner.harmonic_input import HarmonicSequence
        assert HarmonicSequence is not None
