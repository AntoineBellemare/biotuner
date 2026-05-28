"""Bit-exact snapshot regression for the cross-channel resonance pipeline.

Asserts that the refactored ``compute_cross_spectrum_harmonicity`` (now a
thin shim over :func:`biotuner.harmonic_connectivity.compute_cross_resonance`)
reproduces the pre-refactor numerics within ``atol=1e-5`` on 3 reference
signal pairs.

Snapshots in ``snapshots/cross_*.npz`` were captured from the pre-refactor
``compute_cross_spectrum_harmonicity`` via ``_generate_cross_baseline.py``.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _signals import SIGNALS  # noqa: E402

from biotuner.harmonic_connectivity import (  # noqa: E402
    compute_cross_resonance,
    compute_cross_spectrum_harmonicity,
    CrossResonanceResult,
)
from biotuner.resonance import ResonanceConfig  # noqa: E402


SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"
RTOL = 1e-3
ATOL = 1e-5

PAIRS = [
    ("harmonic_pink", "harmonic_5_10_20_40", "pink_noise"),
    ("harmonic_inharmonic", "harmonic_5_10_20_40", "inharmonic_7_11_18_23"),
    ("pink_inharmonic", "pink_noise", "inharmonic_7_11_18_23"),
]

BASE = dict(
    precision_hz=0.5, fmin=2, fmax=30, fs=1000, noverlap=1,
    power_law_remove=False, n_peaks=5, metric="harmsim",
    n_harms=10, delta_lim=0.1, min_notes=2, plot=False,
    smoothness=1, smoothness_harm=1, phase_mode=None,
)


@pytest.mark.parametrize("pair_name,sig1_name,sig2_name", PAIRS)
def test_cross_spectrum_shim_matches_snapshot(pair_name, sig1_name, sig2_name):
    """The refactored compute_cross_spectrum_harmonicity (now a thin shim
    over compute_cross_resonance) must match the pre-refactor snapshot."""
    snap_path = SNAPSHOT_DIR / f"cross_{pair_name}.npz"
    if not snap_path.exists():
        pytest.skip(f"Snapshot missing: {snap_path}")

    sig1 = SIGNALS[sig1_name](sf=BASE["fs"])
    sig2 = SIGNALS[sig2_name](sf=BASE["fs"])
    df = compute_cross_spectrum_harmonicity(sig1, sig2, **BASE)
    plt.close("all")

    snap = np.load(snap_path, allow_pickle=True)
    np.testing.assert_allclose(
        np.asarray(df["harmonicity"].iloc[0]), snap["harmonicity"],
        rtol=RTOL, atol=ATOL,
        err_msg=f"H mismatch on {pair_name}",
    )
    np.testing.assert_allclose(
        np.asarray(df["phase_coupling"].iloc[0]), snap["phase_coupling"],
        rtol=RTOL, atol=ATOL,
        err_msg=f"PC mismatch on {pair_name}",
    )
    np.testing.assert_allclose(
        np.asarray(df["resonance"].iloc[0]), snap["resonance"],
        rtol=RTOL, atol=ATOL,
        err_msg=f"R mismatch on {pair_name}",
    )


@pytest.mark.parametrize("pair_name,sig1_name,sig2_name", PAIRS)
def test_compute_cross_resonance_matches_snapshot(pair_name, sig1_name, sig2_name):
    """The new compute_cross_resonance API directly must also match (via the
    same legacy-default ResonanceConfig). This is the API users should call
    going forward."""
    snap_path = SNAPSHOT_DIR / f"cross_{pair_name}.npz"
    if not snap_path.exists():
        pytest.skip(f"Snapshot missing: {snap_path}")

    sig1 = SIGNALS[sig1_name](sf=BASE["fs"])
    sig2 = SIGNALS[sig2_name](sf=BASE["fs"])
    cfg = ResonanceConfig(
        precision_hz=BASE["precision_hz"],
        fmin=BASE["fmin"], fmax=BASE["fmax"],
        noverlap=BASE["noverlap"], smoothness=BASE["smoothness"],
        n_peaks=BASE["n_peaks"],
        remove_aperiodic=BASE["power_law_remove"],
        harmonic_kernel="harmsim",
        harmonic_kernel_params={"n_harms": BASE["n_harms"], "delta_lim": BASE["delta_lim"], "min_notes": BASE["min_notes"]},
        phase_estimator="stft",
        coupling_metric="nm_wpli_complex",
        gaussian_smooth_sigma=BASE["smoothness_harm"],
        combine="product",
    )
    result = compute_cross_resonance(sig1, sig2, sf=BASE["fs"], config=cfg)

    snap = np.load(snap_path, allow_pickle=True)
    np.testing.assert_allclose(
        result.factors["H"]["all"], snap["harmonicity"],
        rtol=RTOL, atol=ATOL,
        err_msg=f"H[all] mismatch on {pair_name}",
    )
    np.testing.assert_allclose(
        result.factors["PC"]["all"], snap["phase_coupling"],
        rtol=RTOL, atol=ATOL,
        err_msg=f"PC[all] mismatch on {pair_name}",
    )
    np.testing.assert_allclose(
        result.resonance_spectrum["all"], snap["resonance"],
        rtol=RTOL, atol=ATOL,
        err_msg=f"R[all] mismatch on {pair_name}",
    )


def test_cross_resonance_result_has_three_flavors():
    """CrossResonanceResult exposes the 3 reducer flavors per factor."""
    sig1 = SIGNALS["harmonic_5_10_20_40"](sf=1000)
    sig2 = SIGNALS["pink_noise"](sf=1000)
    result = compute_cross_resonance(sig1, sig2, sf=1000)

    assert isinstance(result, CrossResonanceResult)
    assert set(result.resonance_spectrum.keys()) == {"1to2", "2to1", "all"}
    assert set(result.factors["H"].keys()) == {"1to2", "2to1", "all"}
    assert set(result.factors["PC"].keys()) == {"1to2", "2to1", "all"}
    n = result.freqs.size
    for flavor in ("1to2", "2to1", "all"):
        assert result.factors["H"][flavor].shape == (n,)
        assert result.factors["PC"][flavor].shape == (n,)
        assert result.resonance_spectrum[flavor].shape == (n,)


def test_cross_resonance_default_config_works():
    """Calling compute_cross_resonance with no config uses sensible defaults
    (matches legacy compute_cross_spectrum_harmonicity behavior)."""
    sig1 = SIGNALS["harmonic_5_10_20_40"](sf=1000)
    sig2 = SIGNALS["inharmonic_7_11_18_23"](sf=1000)
    result = compute_cross_resonance(sig1, sig2, sf=1000)
    assert result.freqs.size > 0
    assert np.all(np.isfinite(result.resonance_spectrum["all"]))
