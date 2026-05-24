"""Tests for ``Timbre.attach_*`` methods — Phase-3 bt-only augmentations.

Covers PAC → AM modulators, CFC → FM modulators, intermodulation →
AM/FM modulators, the unified ``attach_all_from_biotuner``, no-op
behaviour when bt has nothing, and the chainable / immutable contract.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from biotuner.harmonic_input import HarmonicInput
from biotuner.harmonic_timbre import Timbre
from biotuner.harmonic_timbre.timbre import Modulator


@pytest.fixture
def basic_timbre():
    """A 3-partial Timbre to attach modulators onto."""
    hi = HarmonicInput.from_peaks([100.0, 200.0, 400.0])
    return Timbre.from_harmonic_input(hi)


# ---------------------------------------------------------------------------
# PAC → AM
# ---------------------------------------------------------------------------


class TestAttachPac:
    def test_pac_pairs_become_am_modulators(self, basic_timbre):
        bt = SimpleNamespace(
            pac_freqs=[(5.0, 100.0), (10.0, 400.0)],
            pac_coupling=[0.8, 0.6],
        )
        result = basic_timbre.attach_modulators_from_pac(bt)
        assert len(result.am_modulators) == 2
        # Highest-coupling pair first.
        assert result.am_modulators[0].mod_type == "AM"
        assert result.am_modulators[0].mod_freq == 5.0  # low freq
        assert 0 <= result.am_modulators[0].carrier_idx < basic_timbre.n_partials()

    def test_returns_self_when_no_pac_data(self, basic_timbre):
        bt = SimpleNamespace()
        result = basic_timbre.attach_modulators_from_pac(bt)
        assert result is basic_timbre

    def test_empty_pac_lists_no_op(self, basic_timbre):
        bt = SimpleNamespace(pac_freqs=[], pac_coupling=[])
        result = basic_timbre.attach_modulators_from_pac(bt)
        assert result is basic_timbre

    def test_threshold_drops_weak_pairs(self, basic_timbre):
        bt = SimpleNamespace(
            pac_freqs=[(5.0, 100.0), (10.0, 400.0)],
            pac_coupling=[0.8, 0.1],
        )
        result = basic_timbre.attach_modulators_from_pac(
            bt, coupling_threshold=0.5
        )
        assert len(result.am_modulators) == 1
        assert result.am_modulators[0].mod_freq == 5.0

    def test_immutable_original_unchanged(self, basic_timbre):
        # Sanity: ensure the original Timbre keeps its empty modulator list
        # even after attach returns a new one.
        bt = SimpleNamespace(
            pac_freqs=[(5.0, 100.0)],
            pac_coupling=[0.7],
        )
        original_count = len(basic_timbre.am_modulators)
        _ = basic_timbre.attach_modulators_from_pac(bt)
        assert len(basic_timbre.am_modulators) == original_count

    def test_existing_modulators_preserved(self, basic_timbre):
        existing = Modulator(
            carrier_idx=0, mod_freq=1.0, depth=0.5, mod_type="AM",
            source="manual",
        )
        seeded = basic_timbre.with_partials(am_modulators=[existing])
        bt = SimpleNamespace(
            pac_freqs=[(5.0, 100.0)],
            pac_coupling=[0.7],
        )
        result = seeded.attach_modulators_from_pac(bt)
        assert len(result.am_modulators) == 2
        # The pre-existing one stays at the front.
        assert result.am_modulators[0].source == "manual"


# ---------------------------------------------------------------------------
# CFC → FM
# ---------------------------------------------------------------------------


class TestAttachCfc:
    def test_cfc_pairs_become_fm_modulators(self, basic_timbre):
        bt = SimpleNamespace(
            cfc_freqs=[(8.0, 200.0)],
            cfc_coupling=[0.9],
        )
        result = basic_timbre.attach_modulators_from_cfc(bt)
        assert len(result.fm_modulators) == 1
        m = result.fm_modulators[0]
        assert m.mod_type == "FM"
        assert m.mod_freq == 8.0
        # Deviation is coupling * scale * f_high; depth in Hz.
        assert m.depth > 0

    def test_falls_back_to_pac(self, basic_timbre):
        # CFC missing → use PAC data as the source.
        bt = SimpleNamespace(
            pac_freqs=[(5.0, 100.0)],
            pac_coupling=[0.8],
        )
        result = basic_timbre.attach_modulators_from_cfc(bt)
        assert len(result.fm_modulators) == 1

    def test_returns_self_when_no_data(self, basic_timbre):
        bt = SimpleNamespace()
        result = basic_timbre.attach_modulators_from_cfc(bt)
        assert result is basic_timbre

    def test_deviation_scale(self, basic_timbre):
        bt = SimpleNamespace(
            cfc_freqs=[(8.0, 200.0)],
            cfc_coupling=[1.0],
        )
        weak = basic_timbre.attach_modulators_from_cfc(bt, deviation_scale=0.1)
        strong = basic_timbre.attach_modulators_from_cfc(bt, deviation_scale=1.0)
        assert weak.fm_modulators[0].depth < strong.fm_modulators[0].depth


# ---------------------------------------------------------------------------
# Intermodulation
# ---------------------------------------------------------------------------


class TestAttachIntermod:
    def test_am_mode_appends_am_modulators(self, basic_timbre):
        bt = SimpleNamespace(
            endogenous_intermodulations=[(100.0, 50.0), (200.0, 25.0)],
        )
        result = basic_timbre.attach_intermodulation_modulators(bt, mode="AM")
        assert len(result.am_modulators) == 2
        for m in result.am_modulators:
            assert m.mod_type == "AM"

    def test_fm_mode_appends_fm_modulators(self, basic_timbre):
        bt = SimpleNamespace(
            endogenous_intermodulations=[(100.0, 50.0)],
        )
        result = basic_timbre.attach_intermodulation_modulators(bt, mode="FM")
        assert len(result.fm_modulators) == 1
        assert result.fm_modulators[0].mod_type == "FM"

    def test_unknown_mode_raises(self, basic_timbre):
        bt = SimpleNamespace(endogenous_intermodulations=[(100.0, 50.0)])
        with pytest.raises(ValueError, match="must be 'AM' or 'FM'"):
            basic_timbre.attach_intermodulation_modulators(bt, mode="PM")

    def test_no_intermod_no_op(self, basic_timbre):
        bt = SimpleNamespace()
        result = basic_timbre.attach_intermodulation_modulators(bt)
        assert result is basic_timbre


# ---------------------------------------------------------------------------
# Unified attach_all_from_biotuner
# ---------------------------------------------------------------------------


class TestAttachAll:
    def test_attaches_pac_cfc_intermod_together(self, basic_timbre):
        bt = SimpleNamespace(
            pac_freqs=[(5.0, 100.0)],
            pac_coupling=[0.8],
            cfc_freqs=[(8.0, 200.0)],
            cfc_coupling=[0.6],
            endogenous_intermodulations=[(100.0, 50.0)],
        )
        result = basic_timbre.attach_all_from_biotuner(bt)
        # PAC → 1 AM, intermod AM → 1 AM, CFC → 1 FM
        assert len(result.am_modulators) == 2
        assert len(result.fm_modulators) == 1

    def test_can_disable_individual_attachments(self, basic_timbre):
        bt = SimpleNamespace(
            pac_freqs=[(5.0, 100.0)],
            pac_coupling=[0.8],
            cfc_freqs=[(8.0, 200.0)],
            cfc_coupling=[0.6],
            endogenous_intermodulations=[(100.0, 50.0)],
        )
        result = basic_timbre.attach_all_from_biotuner(
            bt, pac=False, intermod=False
        )
        assert len(result.am_modulators) == 0  # PAC + intermod skipped
        assert len(result.fm_modulators) == 1  # CFC kept

    def test_bare_bt_returns_self(self, basic_timbre):
        bt = SimpleNamespace()
        result = basic_timbre.attach_all_from_biotuner(bt)
        # Every sub-attach is a no-op → no changes; depending on
        # implementation we either get the same instance or an
        # equivalent one. Verify behaviourally.
        assert len(result.am_modulators) == 0
        assert len(result.fm_modulators) == 0

    def test_chainable_pattern(self, basic_timbre):
        # Demonstrates the documented usage pattern.
        bt = SimpleNamespace(
            pac_freqs=[(5.0, 100.0)],
            pac_coupling=[0.8],
            cfc_freqs=[(8.0, 200.0)],
            cfc_coupling=[0.6],
        )
        result = (
            basic_timbre
            .attach_modulators_from_pac(bt)
            .attach_modulators_from_cfc(bt)
        )
        assert len(result.am_modulators) == 1
        assert len(result.fm_modulators) == 1
