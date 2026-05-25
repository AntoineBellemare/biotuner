"""Tests for the ``composite`` wavetable evolution.

Composite layers chain multiple single-axis evolutions with per-layer
weight curves. Coverage:

  - WavetableLayer construction + validation
  - weight_at math across all curve types
  - _frame_composite with a single layer = same as that mode directly
  - Layer ordering is preserved (different orders → different outputs)
  - Spectral + waveform layers compose (e.g. formant → wavefolding)
  - fm_baked terminates the spectral chain cleanly
  - Full export_wavetable round-trip via the composite_layers kwarg
"""

import os
import json
from types import SimpleNamespace

import numpy as np
import pytest

from biotuner.harmonic_timbre import Timbre
from biotuner.harmonic_timbre.synthesis import render_wavetable_cycle
from biotuner.harmonic_timbre.exporters.to_wavetable import (
    WavetableLayer,
    _frame_composite,
    _frame_with_tilt,
    _frame_with_wavefolding,
    _COMPOSITE_ALLOWED,
    _COMPOSITE_WEIGHT_CURVES,
    export_wavetable,
)


@pytest.fixture
def basic_timbre():
    return Timbre(
        partials_hz=np.array([220.0, 440.0, 660.0]),
        amplitudes=np.array([1.0, 0.5, 0.25]),
    )


# ---------------------------------------------------------------------------
# WavetableLayer
# ---------------------------------------------------------------------------


class TestWavetableLayer:
    def test_construct_minimal(self):
        layer = WavetableLayer(evolution="tilt")
        assert layer.evolution == "tilt"
        assert layer.weight_curve == "linear"
        assert layer.weight_min == 0.0
        assert layer.weight_max == 1.0
        assert layer.params == {}

    def test_construct_with_params(self):
        layer = WavetableLayer(
            evolution="formant_sweep",
            weight_curve="sine",
            weight_min=800.0,
            weight_max=2800.0,
            params={"width_hz": 600.0, "gain_db": 5.0},
        )
        assert layer.params["width_hz"] == 600.0

    def test_composite_recursion_rejected(self):
        with pytest.raises(ValueError, match="recurse"):
            WavetableLayer(evolution="composite")

    def test_unknown_evolution_rejected(self):
        with pytest.raises(ValueError, match="not in"):
            WavetableLayer(evolution="bogus_mode")

    def test_unknown_curve_rejected(self):
        with pytest.raises(ValueError, match="weight_curve"):
            WavetableLayer(evolution="tilt", weight_curve="spline")

    @pytest.mark.parametrize("curve", _COMPOSITE_WEIGHT_CURVES)
    def test_all_curves_accepted(self, curve):
        layer = WavetableLayer(evolution="tilt", weight_curve=curve)
        assert layer.weight_curve == curve


# ---------------------------------------------------------------------------
# weight_at — curve math
# ---------------------------------------------------------------------------


class TestWeightAt:
    def test_linear_endpoints(self):
        layer = WavetableLayer(evolution="tilt", weight_min=0.0, weight_max=10.0)
        assert layer.weight_at(0, 5) == pytest.approx(0.0)
        assert layer.weight_at(4, 5) == pytest.approx(10.0)
        assert layer.weight_at(2, 5) == pytest.approx(5.0)

    def test_constant_returns_midpoint(self):
        layer = WavetableLayer(
            evolution="tilt", weight_curve="constant",
            weight_min=2.0, weight_max=8.0,
        )
        for i in range(5):
            assert layer.weight_at(i, 5) == pytest.approx(5.0)

    def test_ease_in_starts_slow(self):
        layer = WavetableLayer(
            evolution="tilt", weight_curve="ease_in",
            weight_min=0.0, weight_max=1.0,
        )
        # At t=0.5, linear=0.5; ease_in=0.25 (quadratic).
        assert layer.weight_at(2, 5) == pytest.approx(0.25)

    def test_ease_out_starts_fast(self):
        layer = WavetableLayer(
            evolution="tilt", weight_curve="ease_out",
            weight_min=0.0, weight_max=1.0,
        )
        # At t=0.5, linear=0.5; ease_out=0.75 (1 − (1−0.5)²).
        assert layer.weight_at(2, 5) == pytest.approx(0.75)

    def test_sine_smooth_midpoint(self):
        layer = WavetableLayer(
            evolution="tilt", weight_curve="sine",
            weight_min=0.0, weight_max=1.0,
        )
        # sine curve hits 0.5 at the midpoint (where the half-cosine is at 0).
        assert layer.weight_at(2, 5) == pytest.approx(0.5)

    def test_single_frame_returns_max(self):
        # Edge case — n_frames=1 should return weight_max.
        layer = WavetableLayer(evolution="tilt", weight_min=0.2, weight_max=0.8)
        assert layer.weight_at(0, 1) == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# _frame_composite — dispatch + chaining
# ---------------------------------------------------------------------------


class TestFrameComposite:
    def test_single_layer_matches_direct_mode(self, basic_timbre):
        # Tilt layer alone @ tilt=1.5 should produce the same cycle as
        # _frame_with_tilt(timbre, 1.5).
        layer = WavetableLayer(
            evolution="tilt", weight_curve="constant",
            weight_min=1.5, weight_max=1.5,
        )
        composite = _frame_composite(
            basic_timbre, [layer], frame_idx=2, n_frames=5, table_size=512,
        )
        direct = _frame_with_tilt(basic_timbre, 1.5, table_size=512)
        np.testing.assert_allclose(composite, direct, atol=1e-5)

    def test_two_layers_chain_in_order(self, basic_timbre):
        # harmonic_stack ADDS partials at 2f, 3f, …; formant_sweep
        # then boosts amplitudes near a Hz center. Order matters:
        # stack → formant boosts BOTH original and new overtones;
        # formant → stack boosts only originals, then adds raw stack
        # overtones (no boost).
        stack = WavetableLayer(
            evolution="harmonic_stack", weight_curve="constant",
            weight_min=3.0, weight_max=3.0,
            params={"rolloff": 0.9},
        )
        formant = WavetableLayer(
            evolution="formant_sweep", weight_curve="constant",
            weight_min=900.0, weight_max=900.0,   # near 2nd–3rd partial range
            params={"width_hz": 300, "gain_db": 8},
        )
        a = _frame_composite(basic_timbre, [stack, formant], 0, 1, table_size=1024)
        b = _frame_composite(basic_timbre, [formant, stack], 0, 1, table_size=1024)
        # Should differ measurably — order matters.
        assert not np.allclose(a, b, atol=1e-4)

    def test_spectral_plus_waveform_layer(self, basic_timbre):
        # harmonic_stack adds overtones; then wavefolding folds the
        # result. Should NOT equal wavefolding alone.
        stack = WavetableLayer(
            evolution="harmonic_stack", weight_curve="constant",
            weight_min=3.0, weight_max=3.0,
            params={"rolloff": 0.9},
        )
        fold = WavetableLayer(
            evolution="wavefolding", weight_curve="constant",
            weight_min=2.0, weight_max=2.0,
        )
        combined = _frame_composite(
            basic_timbre, [stack, fold], 0, 1, table_size=1024,
        )
        fold_only = _frame_with_wavefolding(basic_timbre, 2.0, table_size=1024)
        # The stacked-then-folded cycle should differ from plain folded.
        assert not np.allclose(combined, fold_only, atol=1e-3)

    def test_fm_baked_terminates_chain(self, basic_timbre):
        # If fm_baked is in the chain, it OVERRIDES any previously
        # rendered cycle. Verify by chaining tilt → fm_baked: result
        # should be the fm-baked cycle off the tilted Timbre.
        tilt = WavetableLayer(
            evolution="tilt", weight_curve="constant",
            weight_min=0.8, weight_max=0.8,
        )
        fm = WavetableLayer(
            evolution="fm_baked", weight_curve="constant",
            weight_min=2.0, weight_max=2.0,
            params={"cm_ratio": 2.0, "target_partial_idx": 0},
        )
        out = _frame_composite(basic_timbre, [tilt, fm], 0, 1, table_size=512)
        # Output should be a valid float32 array, peak ≤ 0.99 (FM-baked
        # normalisation), not all zeros.
        assert out.dtype == np.float32
        assert out.shape == (512,)
        assert float(np.max(np.abs(out))) > 0.1
        assert float(np.max(np.abs(out))) <= 0.991

    def test_intermod_layer_needs_bt(self, basic_timbre):
        # intermod_buildup is a no-op when bt is missing (graceful).
        layer = WavetableLayer(
            evolution="intermod_buildup", weight_curve="constant",
            weight_min=0.5, weight_max=0.5,
        )
        # Without bt → graceful degradation, output is the unmodified
        # render of basic_timbre.
        out_nobt = _frame_composite(basic_timbre, [layer], 0, 1, table_size=512)
        baseline = render_wavetable_cycle(basic_timbre, table_size=512)
        np.testing.assert_allclose(out_nobt, baseline, atol=1e-5)

    def test_intermod_layer_uses_bt_when_present(self, basic_timbre):
        bt = SimpleNamespace(
            endogenous_intermodulations=[(220.0, 110.0), (440.0, 220.0)],
        )
        layer = WavetableLayer(
            evolution="intermod_buildup", weight_curve="constant",
            weight_min=0.5, weight_max=0.5,
        )
        out = _frame_composite(basic_timbre, [layer], 0, 1, table_size=512, bt=bt)
        baseline = render_wavetable_cycle(basic_timbre, table_size=512)
        # With bt, intermod adds sidebands → output should DIFFER from
        # plain render.
        assert not np.allclose(out, baseline, atol=1e-3)

    def test_amp_morph_is_seeded(self, basic_timbre):
        # Same seed → identical output regardless of how often we call.
        layer = WavetableLayer(
            evolution="amp_morph", weight_curve="constant",
            weight_min=0.3, weight_max=0.3,
        )
        a = _frame_composite(basic_timbre, [layer], 0, 1, table_size=512, seed=42)
        b = _frame_composite(basic_timbre, [layer], 0, 1, table_size=512, seed=42)
        np.testing.assert_allclose(a, b)
        # Different seed → different output.
        c = _frame_composite(basic_timbre, [layer], 0, 1, table_size=512, seed=43)
        assert not np.allclose(a, c)

    def test_output_shape_and_dtype(self, basic_timbre):
        layer = WavetableLayer(evolution="tilt", weight_min=0.5, weight_max=1.5)
        for ts in (256, 1024, 2048):
            out = _frame_composite(
                basic_timbre, [layer], 0, 3, table_size=ts,
            )
            assert out.shape == (ts,)
            assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# Full export_wavetable round-trip with composite
# ---------------------------------------------------------------------------


class TestExportComposite:
    def test_composite_export(self, basic_timbre, tmp_path):
        out = tmp_path / "composite.wav"
        result = export_wavetable(
            basic_timbre,
            str(out),
            n_frames=8,
            evolution="composite",
            composite_layers=[
                {"evolution": "harmonic_stack", "weight_curve": "linear",
                 "weight_min": 0, "weight_max": 4,
                 "params": {"rolloff": 0.9}},
                {"evolution": "wavefolding", "weight_curve": "ease_in",
                 "weight_min": 0.0, "weight_max": 2.5},
            ],
            include_sidecar=False,
        )
        assert os.path.exists(result["wavetable"])
        import soundfile as sf
        data, sr = sf.read(result["wavetable"])
        assert data.shape[0] == 8 * 2048
        manifest = json.load(open(result["manifest"]))
        assert manifest["evolution"] == "composite"
        assert len(manifest["evolution_params"]["composite_layers"]) == 2

    def test_composite_requires_layers(self, basic_timbre, tmp_path):
        with pytest.raises(ValueError, match="composite_layers"):
            export_wavetable(
                basic_timbre, str(tmp_path / "x.wav"),
                n_frames=8, evolution="composite",
                composite_layers=None, include_sidecar=False,
            )

    def test_composite_accepts_dict_or_dataclass(self, basic_timbre, tmp_path):
        # Mixed list — one dict, one dataclass — should both coerce.
        result = export_wavetable(
            basic_timbre,
            str(tmp_path / "mixed.wav"),
            n_frames=4,
            evolution="composite",
            composite_layers=[
                WavetableLayer(evolution="tilt", weight_min=0, weight_max=2),
                {"evolution": "phase_sweep", "weight_min": 0, "weight_max": 1.0},
            ],
            include_sidecar=False,
        )
        assert os.path.exists(result["wavetable"])
