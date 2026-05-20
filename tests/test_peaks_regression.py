"""Regression baseline for the peak-extraction pipeline.

This test runs each peaks_function the engine exposes against a fixed
synthetic signal and compares the outputs (peaks + tuning) against a
stored JSON baseline. It exists so future changes to peak-extraction
machinery (e.g. adding SMS or multitaper spectrum estimation) cannot
silently drift any existing method's behavior.

To regenerate the baseline after an intentional change:

    REGEN_BASELINE=1 pytest tests/test_peaks_regression.py

Without that environment variable, the test asserts exact reproducibility
within a tight tolerance.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from biotuner.biotuner_object import compute_biotuner

BASELINE_PATH = Path(__file__).parent / "regression" / "peaks_baseline.json"

# Methods exposed via the engine API. EEMD/CEEMDAN are intentionally omitted
# because they include a non-deterministic ensemble step.
METHODS = ["fixed", "harmonic_recurrence", "EIMC", "FOOOF", "cepstrum", "EMD"]

# Per-method tolerance for output drift. peaks are compared in Hz, tuning in
# ratio. Methods built on iterative solvers can drift slightly between numpy
# minor versions; tighter rtol where we can afford it.
TOLERANCES = {
    "fixed":               {"peaks_atol": 1e-3, "tuning_atol": 1e-4},
    "harmonic_recurrence": {"peaks_atol": 1e-3, "tuning_atol": 1e-4},
    "EIMC":                {"peaks_atol": 1e-3, "tuning_atol": 1e-4},
    "FOOOF":               {"peaks_atol": 5e-2, "tuning_atol": 1e-3},
    "cepstrum":            {"peaks_atol": 5e-2, "tuning_atol": 1e-3},
    "EMD":                 {"peaks_atol": 5e-1, "tuning_atol": 1e-2},
}


def _signal():
    """Deterministic synthetic signal: three sinusoids + low gaussian noise."""
    rng = np.random.default_rng(20260520)  # fixed seed
    sf = 1000.0
    duration = 10.0
    t = np.linspace(0, duration, int(sf * duration), endpoint=False)
    components = [(3.0, 1.0), (10.0, 0.5), (50.0, 0.8)]
    signal = sum(a * np.sin(2 * np.pi * f * t) for f, a in components)
    signal += rng.normal(0.0, 0.05, size=len(t))
    return signal, sf


def _run_method(method, signal, sf):
    """Run a single peaks_function and return JSON-friendly outputs."""
    bt = compute_biotuner(sf=sf, peaks_function=method, precision=0.5)
    bt.peaks_extraction(
        signal,
        n_peaks=5,
        max_freq=100,
        ratios_extension=True,
        min_harms=2,
    )
    peaks = np.asarray(getattr(bt, "peaks", [])).astype(float).tolist()
    tuning = np.asarray(getattr(bt, "peaks_ratios", [])).astype(float).tolist()
    return {"peaks": peaks, "tuning": tuning}


def _collect_all():
    signal, sf = _signal()
    out = {}
    for m in METHODS:
        try:
            out[m] = _run_method(m, signal, sf)
        except Exception as e:
            # Record the failure so a baseline regen captures the same error
            # rather than masking it as a missing key.
            out[m] = {"error": f"{type(e).__name__}: {e}"}
    return out


def _maybe_regen():
    """If REGEN_BASELINE=1, write the current outputs to the baseline file."""
    if os.environ.get("REGEN_BASELINE") != "1":
        return False
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = _collect_all()
    BASELINE_PATH.write_text(json.dumps(data, indent=2, sort_keys=True))
    return True


@pytest.fixture(scope="module")
def baseline():
    if _maybe_regen():
        pytest.skip("Regenerated baseline; rerun without REGEN_BASELINE to verify.")
    if not BASELINE_PATH.exists():
        pytest.skip(
            f"No baseline at {BASELINE_PATH}. "
            f"Run with REGEN_BASELINE=1 to generate it."
        )
    return json.loads(BASELINE_PATH.read_text())


@pytest.fixture(scope="module")
def current():
    return _collect_all()


@pytest.mark.parametrize("method", METHODS)
def test_method_output_matches_baseline(method, baseline, current):
    """Each peak extractor must produce the same output as the baseline."""
    assert method in baseline, f"Baseline missing method: {method}"
    assert method in current,  f"Current run missing method: {method}"

    base = baseline[method]
    cur  = current[method]

    # Surface the same errors on both sides so we don't pretend things work.
    if "error" in base or "error" in cur:
        assert base.get("error") == cur.get("error"), (
            f"{method}: error state diverged.\n"
            f"  baseline: {base.get('error')}\n"
            f"  current : {cur.get('error')}"
        )
        return

    tol = TOLERANCES[method]

    base_peaks = np.asarray(base["peaks"], dtype=float)
    cur_peaks  = np.asarray(cur["peaks"], dtype=float)
    assert base_peaks.shape == cur_peaks.shape, (
        f"{method}: peaks length changed. baseline={base_peaks.shape}, "
        f"current={cur_peaks.shape}"
    )
    np.testing.assert_allclose(
        cur_peaks, base_peaks,
        atol=tol["peaks_atol"], rtol=0,
        err_msg=f"{method}: peaks drifted beyond {tol['peaks_atol']} Hz tolerance",
    )

    base_tuning = np.asarray(base["tuning"], dtype=float)
    cur_tuning  = np.asarray(cur["tuning"], dtype=float)
    assert base_tuning.shape == cur_tuning.shape, (
        f"{method}: tuning length changed. baseline={base_tuning.shape}, "
        f"current={cur_tuning.shape}"
    )
    np.testing.assert_allclose(
        cur_tuning, base_tuning,
        atol=tol["tuning_atol"], rtol=0,
        err_msg=f"{method}: tuning drifted beyond {tol['tuning_atol']} tolerance",
    )
