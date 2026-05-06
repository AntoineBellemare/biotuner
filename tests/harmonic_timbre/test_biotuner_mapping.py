"""Tests for biotuner.harmonic_timbre.biotuner_mapping."""

from __future__ import annotations

import numpy as np
import pytest

from biotuner.harmonic_timbre import (
    DEFAULT_MAPPING_V1,
    ALL_MAPPINGS,
    SCALE_SOURCES,
    resolve_scale,
    timbre_from_biotuner,
    timbre_from_ratios,
)
from biotuner.harmonic_timbre.biotuner_mapping import (
    map_peaks_to_partials,
    map_amps_to_amplitudes,
    map_phases,
    map_linewidth_to_decay,
    map_aperiodic_to_tilt,
    map_flatness_to_noise,
    map_consonance_priors,
)


# ---------------------------------------------------------------------------
# Mapping sets
# ---------------------------------------------------------------------------

def test_default_mapping_v1_subset_of_all_mappings():
    assert DEFAULT_MAPPING_V1 <= ALL_MAPPINGS


def test_default_mapping_v1_keys():
    assert "partials" in DEFAULT_MAPPING_V1
    assert "amplitudes" in DEFAULT_MAPPING_V1
    assert "phases" in DEFAULT_MAPPING_V1
    assert "consonance_weighting" in DEFAULT_MAPPING_V1


# ---------------------------------------------------------------------------
# resolve_scale
# ---------------------------------------------------------------------------

def test_resolve_scale_known_source(mock_biotuner_v1):
    ratios, prov = resolve_scale("peaks_ratios_cons", bt=mock_biotuner_v1)
    assert prov == "peaks_ratios_cons"
    assert len(ratios) > 0
    assert ratios == list(mock_biotuner_v1.peaks_ratios_cons)


def test_resolve_scale_raw_iterable():
    ratios, prov = resolve_scale([1.0, 1.5, 2.0])
    assert prov == "raw"
    assert ratios == [1.0, 1.5, 2.0]


def test_resolve_scale_unknown_source_raises():
    with pytest.raises(KeyError):
        resolve_scale("not_a_source", bt=object())


def test_resolve_scale_none_walks_fallback(mock_biotuner_v1):
    # peaks_ratios_cons populated → falls back chain picks it first
    ratios, prov = resolve_scale(None, bt=mock_biotuner_v1)
    assert prov in ("peaks_ratios_cons", "peaks_ratios")


def test_resolve_scale_known_sources_count():
    # Sanity: at least the v1 set we documented
    assert "peaks_ratios" in SCALE_SOURCES
    assert "peaks_ratios_cons" in SCALE_SOURCES
    assert "diss_scale" in SCALE_SOURCES
    assert "harmonic_entropy" in SCALE_SOURCES
    assert "raw" in SCALE_SOURCES


# ---------------------------------------------------------------------------
# Per-mapping helpers on the v1 mock
# ---------------------------------------------------------------------------

def test_map_peaks_to_partials(mock_biotuner_v1):
    p = map_peaks_to_partials(mock_biotuner_v1)
    assert p is not None
    assert p.shape == (5,)
    np.testing.assert_allclose(p, mock_biotuner_v1.peaks)


def test_map_amps_to_amplitudes(mock_biotuner_v1):
    a = map_amps_to_amplitudes(mock_biotuner_v1)
    assert a is not None
    assert a.shape == (5,)


def test_map_phases_present_when_set(mock_biotuner_v1):
    ph = map_phases(mock_biotuner_v1)
    assert ph is not None
    assert ph.shape == (5,)


def test_map_phases_returns_none_when_absent(mock_biotuner_minimal):
    assert map_phases(mock_biotuner_minimal) is None


def test_map_linewidth_to_decay(mock_biotuner_v1):
    decay, bw = map_linewidth_to_decay(mock_biotuner_v1)
    assert decay is not None and bw is not None
    # decay = 1/(π·linewidth), so should be inversely related
    expected_decay = 1.0 / (np.pi * np.asarray(mock_biotuner_v1.linewidth))
    np.testing.assert_allclose(decay, expected_decay)


def test_map_linewidth_returns_none_when_absent(mock_biotuner_minimal):
    decay, bw = map_linewidth_to_decay(mock_biotuner_minimal)
    assert decay is None and bw is None


def test_map_aperiodic_to_tilt(mock_biotuner_v1):
    tilt = map_aperiodic_to_tilt(mock_biotuner_v1)
    assert tilt == 1.5


def test_map_aperiodic_returns_none_when_absent(mock_biotuner_minimal):
    assert map_aperiodic_to_tilt(mock_biotuner_minimal) is None


def test_map_flatness_to_noise(mock_biotuner_v1):
    nf = map_flatness_to_noise(mock_biotuner_v1)
    assert nf == 0.2


def test_map_consonance_priors_normalized(mock_biotuner_v1):
    w = map_consonance_priors(mock_biotuner_v1)
    assert w is not None
    assert np.isclose(w.max(), 1.0)
    assert np.all(w >= 0)


# ---------------------------------------------------------------------------
# timbre_from_biotuner — full path
# ---------------------------------------------------------------------------

def test_timbre_from_biotuner_default(mock_biotuner_v1):
    t = timbre_from_biotuner(mock_biotuner_v1, scale="peaks_ratios_cons")
    assert t.matching_method == "consonance_weighted"
    assert t.metadata["scale_source"] == "peaks_ratios_cons"
    assert "biotuner_fields_used" in t.metadata
    # spectral_tilt and noise_floor are populated from the mock
    assert t.spectral_tilt == 1.5
    assert t.noise_floor == 0.2
    # consonance_weighting is in the use set
    assert "consonance_weighting" in t.metadata["biotuner_fields_used"]


def test_timbre_from_biotuner_with_raw_scale(mock_biotuner_v1):
    # bypass biotuner ratios entirely — secondary fields still pulled
    t = timbre_from_biotuner(
        mock_biotuner_v1,
        scale=[1.0, 5 / 4, 3 / 2, 2.0],
    )
    assert t.metadata["scale_source"] == "raw"
    assert t.matched_tuning == [1.0, 5 / 4, 3 / 2, 2.0]


def test_timbre_from_biotuner_use_subset(mock_biotuner_v1):
    t = timbre_from_biotuner(
        mock_biotuner_v1,
        scale="peaks_ratios_cons",
        use=["partials", "amplitudes"],  # no spectral_tilt request
    )
    # spectral_tilt should NOT have been pulled
    assert t.spectral_tilt is None


def test_timbre_from_biotuner_minimal_skips_unavailable(mock_biotuner_minimal):
    """When only peaks/amps/ratios are present, spectral_tilt etc. are
    silently skipped — never raised."""
    t = timbre_from_biotuner(mock_biotuner_minimal, scale="peaks_ratios_cons")
    assert t.spectral_tilt is None
    assert t.noise_floor is None
    # but the timbre is still valid
    t.validate()


def test_timbre_from_biotuner_unknown_method_raises(mock_biotuner_v1):
    with pytest.raises(ValueError, match="unknown matching method"):
        timbre_from_biotuner(
            mock_biotuner_v1,
            scale="peaks_ratios_cons",
            matching_method="not-real",
        )


# ---------------------------------------------------------------------------
# timbre_from_ratios — no biotuner
# ---------------------------------------------------------------------------

def test_timbre_from_ratios_works_without_biotuner():
    t = timbre_from_ratios([1.0, 5 / 4, 3 / 2, 2.0])
    assert t.metadata.get("scale_source") == "raw"
    assert t.matching_method == "consonance_weighted"  # default


def test_timbre_from_ratios_method_override():
    t = timbre_from_ratios([1.0, 3 / 2, 2.0], matching_method="direct")
    assert t.matching_method == "direct"


# ---------------------------------------------------------------------------
# Provenance invariants
# ---------------------------------------------------------------------------

def test_scale_source_recorded_in_metadata(mock_biotuner_v1):
    for scale in ("peaks_ratios", "peaks_ratios_cons", "extended_peaks_ratios"):
        t = timbre_from_biotuner(mock_biotuner_v1, scale=scale)
        assert t.metadata["scale_source"] == scale


# ---------------------------------------------------------------------------
# v1.1 mappings — AM/FM modulators
# ---------------------------------------------------------------------------

class _MockBTPACCFC:
    """A bt-like object with PAC, CFC, and intermod data populated."""

    def __init__(self):
        self.peaks = [4.0, 8.0, 13.0, 25.0, 40.0]
        self.amps = [0.5, 1.0, 0.8, 0.4, 0.25]
        self.pac_freqs = [(4.0, 25.0), (8.0, 40.0), (4.0, 40.0)]
        self.pac_coupling = [0.6, 0.8, 0.3]
        self.cfc_freqs = [(8.0, 25.0), (4.0, 13.0)]
        self.cfc_coupling = [0.5, 0.4]
        self.endogenous_intermodulations = [(13.0, 8.0), (25.0, 4.0)]


def test_map_pac_returns_one_am_per_pac_pair():
    from biotuner.harmonic_timbre import map_pac_to_am_modulators
    bt = _MockBTPACCFC()
    mods = map_pac_to_am_modulators(bt)
    assert len(mods) == 3
    assert all(m.mod_type == "AM" for m in mods)
    # sorted strongest-first → first is the 0.8-coupling one (8 -> 40)
    assert mods[0].mod_freq == 8.0
    assert mods[0].depth == 0.8


def test_map_pac_carrier_idx_is_nearest_partial():
    from biotuner.harmonic_timbre import map_pac_to_am_modulators
    bt = _MockBTPACCFC()
    mods = map_pac_to_am_modulators(bt)
    # First PAC pair: high freq 40 Hz → bt.peaks index 4
    pair_8_40 = next(m for m in mods if m.mod_freq == 8.0)
    assert pair_8_40.carrier_idx == 4   # peaks[4] == 40.0
    pair_4_25 = next(m for m in mods if m.mod_freq == 4.0 and m.source.startswith("PAC[4.00->25"))
    assert pair_4_25.carrier_idx == 3   # peaks[3] == 25.0


def test_map_pac_threshold_filters_weak_couplings():
    from biotuner.harmonic_timbre import map_pac_to_am_modulators
    bt = _MockBTPACCFC()
    mods = map_pac_to_am_modulators(bt, coupling_threshold=0.5)
    # 0.3-coupling pair filtered out
    assert len(mods) == 2


def test_map_pac_returns_empty_when_no_data():
    from biotuner.harmonic_timbre import map_pac_to_am_modulators
    class Empty:
        peaks = [10.0]
    assert map_pac_to_am_modulators(Empty()) == []


def test_map_cfc_returns_fm_modulators():
    from biotuner.harmonic_timbre import map_cfc_to_fm_modulators
    bt = _MockBTPACCFC()
    mods = map_cfc_to_fm_modulators(bt)
    assert len(mods) == 2
    assert all(m.mod_type == "FM" for m in mods)
    # depth scales with high-freq target × coupling
    strongest = mods[0]
    assert strongest.depth > 0


def test_map_cfc_falls_back_to_pac_when_cfc_absent():
    """Pipelines that only populate PAC should still produce CFC modulators."""
    from biotuner.harmonic_timbre import map_cfc_to_fm_modulators
    class OnlyPAC:
        peaks = [4.0, 8.0, 13.0, 25.0]
        pac_freqs = [(4.0, 25.0)]
        pac_coupling = [0.7]
    mods = map_cfc_to_fm_modulators(OnlyPAC())
    assert len(mods) == 1
    assert mods[0].mod_type == "FM"


def test_map_intermod_returns_one_modulator_per_pair():
    from biotuner.harmonic_timbre import map_intermod_to_modulators
    bt = _MockBTPACCFC()
    mods_am = map_intermod_to_modulators(bt, mode="AM")
    assert len(mods_am) == 2
    assert all(m.mod_type == "AM" for m in mods_am)
    mods_fm = map_intermod_to_modulators(bt, mode="FM")
    assert len(mods_fm) == 2
    assert all(m.mod_type == "FM" for m in mods_fm)


def test_map_intermod_invalid_mode_raises():
    from biotuner.harmonic_timbre import map_intermod_to_modulators
    bt = _MockBTPACCFC()
    with pytest.raises(ValueError):
        map_intermod_to_modulators(bt, mode="ringmod")
