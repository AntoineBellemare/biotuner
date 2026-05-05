"""Tests for biotuner.harmonic_geometry.extensions."""

from fractions import Fraction

import numpy as np
import pytest

from biotuner.harmonic_geometry import HarmonicInput
from biotuner.harmonic_geometry.extensions import (
    extend_harmonic_fit,
    extend_harmonic_tuning,
    extend_harmonics,
    extend_subharmonics,
)


@pytest.fixture
def major_chord() -> HarmonicInput:
    return HarmonicInput(
        peaks=[100.0, 125.0, 150.0]  # 4 : 5 : 6 chord at base_freq=25 Hz
    )


# ============================================================ extend_harmonics


class TestExtendHarmonics:
    def test_n_components_grows(self, major_chord):
        ext = extend_harmonics(major_chord, n_harmonics=4)
        # 3 peaks × (1 + 4) harmonics = 15 entries (with the
        # n_start/n_harmonics+2 loop: 1..n+1 inclusive of fundamental
        # gives 5 per peak).
        assert ext.n_components() >= 3 * 4

    def test_metadata_tag(self, major_chord):
        ext = extend_harmonics(major_chord, n_harmonics=2)
        assert ext.metadata.get("extension") == "harmonics"

    def test_excluding_fundamental(self, major_chord):
        ext = extend_harmonics(
            major_chord, n_harmonics=3, include_fundamental=False
        )
        # When fundamental excluded, first harmonic of 100 Hz is 200 Hz.
        peaks = ext.to_peaks().tolist()
        assert 100.0 not in peaks  # fundamental dropped
        assert 200.0 in peaks       # 2nd harmonic kept

    def test_decay_zero_uniform_amps(self, major_chord):
        ext = extend_harmonics(major_chord, n_harmonics=3, decay=0.0)
        amps = ext.amplitudes
        # decay=0 means each peak's amplitude is the same as its
        # parent's (a / 1**0 = a). Per-parent amplitudes are uniform
        # since major_chord didn't pass amplitudes.
        # All amps should be equal.
        assert max(amps) - min(amps) < 1e-12

    def test_invalid_n_harmonics(self, major_chord):
        with pytest.raises(ValueError):
            extend_harmonics(major_chord, n_harmonics=0)


# ========================================================= extend_subharmonics


class TestExtendSubharmonics:
    def test_subharmonics_below_fundamental(self, major_chord):
        ext = extend_subharmonics(major_chord, n_harmonics=3)
        peaks = ext.to_peaks().tolist()
        # 100/2 = 50 should appear
        assert any(abs(p - 50.0) < 1e-9 for p in peaks)
        assert any(abs(p - 100.0 / 3) < 1e-9 for p in peaks)

    def test_metadata_tag(self, major_chord):
        ext = extend_subharmonics(major_chord, n_harmonics=2)
        assert ext.metadata.get("extension") == "subharmonics"


# ======================================================= extend_harmonic_fit


class TestExtendHarmonicFit:
    def test_returns_input(self, major_chord):
        # Even when no harmonics match (which can happen with the default
        # bounds for a clean rational chord), the result should at least
        # contain the originals.
        ext = extend_harmonic_fit(major_chord, n_harm=8, bounds=0.5)
        assert ext.n_components() >= major_chord.n_components()

    def test_metadata_tag(self, major_chord):
        ext = extend_harmonic_fit(major_chord, n_harm=8, bounds=0.5)
        assert ext.metadata.get("extension") == "harmonic_fit"

    def test_invalid_n_harm(self, major_chord):
        with pytest.raises(ValueError):
            extend_harmonic_fit(major_chord, n_harm=1)


# ==================================================== extend_harmonic_tuning


class TestExtendHarmonicTuning:
    def test_returns_harmonic_input(self, major_chord):
        ext = extend_harmonic_tuning(major_chord, n_harmonics=8)
        assert isinstance(ext, HarmonicInput)
        assert ext.metadata.get("extension") == "harmonic_tuning"

    def test_richer_with_more_harmonics(self, major_chord):
        coarse = extend_harmonic_tuning(major_chord, n_harmonics=4)
        rich = extend_harmonic_tuning(major_chord, n_harmonics=20)
        assert rich.n_components() >= coarse.n_components()

    def test_invalid_n_harmonics(self, major_chord):
        with pytest.raises(ValueError):
            extend_harmonic_tuning(major_chord, n_harmonics=1)


# ============================================================== composability


class TestComposability:
    def test_extended_input_works_with_geometry_function(self, major_chord):
        """Smoke test: an extended input feeds cleanly into a quasicrystal field."""
        from biotuner.harmonic_geometry import quasicrystal_field_2d
        ext = extend_harmonics(major_chord, n_harmonics=3)
        g = quasicrystal_field_2d(
            ext, n_fold=5, resolution=64, extent=1.5,
        )
        assert g.coordinates.shape == (64, 64)

    def test_extended_input_works_with_chladni(self, major_chord):
        """Smoke test: an extended input also works for Chladni
        (proves composability across the whole submodule)."""
        from biotuner.harmonic_geometry import chladni_from_input
        ext = extend_harmonics(major_chord, n_harmonics=2)
        g = chladni_from_input(
            ext, plate="rectangular",
            plate_kwargs={"resolution": 32},
        )
        assert g.coordinates.shape == (32, 32)


# ============================================================== public API


class TestPublicAPI:
    def test_module_exports(self):
        from biotuner.harmonic_geometry import (
            extend_harmonics,
            extend_subharmonics,
            extend_harmonic_fit,
            extend_harmonic_tuning,
        )

        assert callable(extend_harmonics)
        assert callable(extend_subharmonics)
        assert callable(extend_harmonic_fit)
        assert callable(extend_harmonic_tuning)
