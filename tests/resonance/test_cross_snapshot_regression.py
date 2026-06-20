"""Snapshot regression for the cross-channel resonance pipeline.

``compute_cross_spectrum_harmonicity`` is now a thin shim over
:func:`biotuner.harmonic_connectivity.compute_cross_resonance`.

Contract (post phase-alignment fix)
-----------------------------------
* **H (harmonicity)** reproduces the FROZEN pre-refactor cross-spectrum snapshot
  bit-exactly (H is phase-independent — unchanged by the alignment fix).
* **PC / R** are compared to CORRECTED baselines (regenerated from the fixed
  pipeline). The legacy cross path indexed the full STFT grid with the
  fmin-clipped frequency array, offsetting every cross-frequency phase-coupling
  entry; this is fixed, so PC/R intentionally differ from the original legacy
  values.

Snapshots in ``snapshots/cross_*.npz``: ``harmonicity`` is the frozen legacy
value; ``phase_coupling`` / ``resonance`` are corrected baselines.
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
    # Explicitly opt out of the new joint+n:m defaults so snapshot
    # reproduction tests the LEGACY behavior of compute_cross_spectrum_harmonicity.
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
        cross_pc_reducer="count",          # legacy reducer
        cross_use_ratio_kernel=False,       # legacy 1:1 cross-spectrum
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
    """Calling compute_cross_resonance with no config produces sensible output.

    The defaults are the NEW recommended ones (joint PC + n:m PC), which
    differ from the legacy compute_cross_spectrum_harmonicity. Bit-exact
    legacy behavior is available via cross_pc_reducer='count' +
    cross_use_ratio_kernel=False (tested separately above).
    """
    sig1 = SIGNALS["harmonic_5_10_20_40"](sf=1000)
    sig2 = SIGNALS["inharmonic_7_11_18_23"](sf=1000)
    result = compute_cross_resonance(sig1, sig2, sf=1000)
    assert result.freqs.size > 0
    assert np.all(np.isfinite(result.resonance_spectrum["all"]))


def test_cross_resonance_new_defaults_differ_from_legacy():
    """Confirms the new defaults (cross_pc_reducer='joint' +
    cross_use_ratio_kernel=True) produce DIFFERENT output from the legacy
    config — i.e. flipping the defaults actually had effect.
    """
    sig1 = SIGNALS["harmonic_5_10_20_40"](sf=1000)
    sig2 = SIGNALS["inharmonic_7_11_18_23"](sf=1000)

    # No-args call → uses NEW defaults
    r_new = compute_cross_resonance(sig1, sig2, sf=1000)
    # Explicit legacy values
    cfg_legacy = ResonanceConfig(
        precision_hz=1.0, fmin=1.0, fmax=30.0, noverlap=1, smoothness=1.0,
        remove_aperiodic=False, harmonic_kernel="harmsim",
        harmonic_kernel_params={"n_harms": 10, "delta_lim": 0.1, "min_notes": 2},
        phase_estimator="stft", coupling_metric="nm_wpli_complex",
        gaussian_smooth_sigma=1.0, combine="product",
        cross_pc_reducer="count",
        cross_use_ratio_kernel=False,
    )
    r_legacy = compute_cross_resonance(sig1, sig2, sf=1000, config=cfg_legacy)

    # PC and R should differ substantially between new and legacy
    assert not np.allclose(
        r_new.factors["PC"]["all"], r_legacy.factors["PC"]["all"], atol=1e-4
    )
    assert not np.allclose(
        r_new.resonance_spectrum["all"], r_legacy.resonance_spectrum["all"], atol=1e-4
    )


def test_resonance_config_field_defaults_are_refined():
    """The ResonanceConfig field defaults must be the refined values."""
    cfg = ResonanceConfig()
    assert cfg.cross_pc_reducer == "joint", (
        f"Expected cross_pc_reducer='joint', got {cfg.cross_pc_reducer!r}"
    )
    assert cfg.cross_use_ratio_kernel is True, (
        f"Expected cross_use_ratio_kernel=True, got {cfg.cross_use_ratio_kernel}"
    )


# ---------------------------------------------------------------------------
# Layer B refinements A & B — opt-in config flags don't break legacy default
# ---------------------------------------------------------------------------


def test_refinement_A_joint_pc_reducer_changes_output():
    """cross_pc_reducer='joint' produces different PC(f) from default 'count'."""
    sig1 = SIGNALS["harmonic_5_10_20_40"](sf=1000)
    sig2 = SIGNALS["pink_noise"](sf=1000)
    cfg_legacy = ResonanceConfig(
        precision_hz=0.5, fmin=2, fmax=30, noverlap=1, smoothness=1,
        remove_aperiodic=False, harmonic_kernel="harmsim",
        harmonic_kernel_params={"n_harms": 10, "delta_lim": 0.1, "min_notes": 2},
        phase_estimator="stft", coupling_metric="nm_wpli_complex",
        gaussian_smooth_sigma=1.0, combine="product",
        cross_pc_reducer="count",
    )
    cfg_joint = ResonanceConfig(
        **{**cfg_legacy.__dict__, "cross_pc_reducer": "joint"}
    )
    r_legacy = compute_cross_resonance(sig1, sig2, sf=1000, config=cfg_legacy)
    r_joint = compute_cross_resonance(sig1, sig2, sf=1000, config=cfg_joint)

    # The joint reducer should produce a noticeably different PC spectrum
    assert not np.allclose(
        r_legacy.factors["PC"]["all"], r_joint.factors["PC"]["all"], atol=1e-6
    )
    # Both should be finite
    assert np.all(np.isfinite(r_joint.factors["PC"]["all"]))
    assert np.all(np.isfinite(r_joint.resonance_spectrum["all"]))


def test_refinement_B_ratio_kernel_enables_nm_pc():
    """cross_use_ratio_kernel=True dispatches the binary_nm kernel to determine
    (n, m) per freq pair, producing a different Phi[i,j] matrix than the
    default 1:1 cross-spectrum."""
    sig1 = SIGNALS["harmonic_5_10_20_40"](sf=1000)
    sig2 = SIGNALS["harmonic_5_10_20_40"](sf=1000)
    cfg_11 = ResonanceConfig(
        precision_hz=0.5, fmin=2, fmax=30, noverlap=1, smoothness=1,
        remove_aperiodic=False, harmonic_kernel="harmsim",
        harmonic_kernel_params={"n_harms": 10, "delta_lim": 0.1, "min_notes": 2},
        phase_estimator="stft", coupling_metric="nm_wpli_complex",
        gaussian_smooth_sigma=1.0, combine="product",
        cross_use_ratio_kernel=False,
    )
    cfg_nm = ResonanceConfig(
        **{**cfg_11.__dict__, "cross_use_ratio_kernel": True}
    )
    r_11 = compute_cross_resonance(sig1, sig2, sf=1000, config=cfg_11)
    r_nm = compute_cross_resonance(sig1, sig2, sf=1000, config=cfg_nm)
    # n:m gating should produce a sparser, qualitatively different PC matrix
    assert not np.allclose(
        r_11.factors["PC"]["all"], r_nm.factors["PC"]["all"], atol=1e-6
    )
    assert np.all(np.isfinite(r_nm.factors["PC"]["all"]))


def test_refinement_C_surrogate_generators():
    """The 3 surrogate generators all preserve PSD or amplitude distribution
    as documented."""
    from biotuner.resonance.nulls import (
        phase_randomize_surrogate, iaaft_surrogate, time_shuffle_surrogate,
    )
    rng_factory = lambda seed: np.random.default_rng(seed)
    sig = SIGNALS["harmonic_5_10_20_40"](sf=1000)

    # Phase randomization: PSD preserved exactly
    s_pr = phase_randomize_surrogate(sig, rng_factory(0))
    psd_orig = np.abs(np.fft.rfft(sig))
    psd_pr = np.abs(np.fft.rfft(s_pr))
    assert np.allclose(psd_orig, psd_pr, atol=1e-10), \
        f"phase_randomize PSD drift {np.max(np.abs(psd_orig - psd_pr)):.2e}"

    # IAAFT: amplitude distribution preserved exactly (rank-matched)
    s_iaaft = iaaft_surrogate(sig, rng_factory(0), n_iter=50)
    assert np.allclose(np.sort(sig), np.sort(s_iaaft), atol=1e-10), \
        "IAAFT did not preserve amplitude distribution"

    # Time shuffle: amplitude distribution preserved exactly (same samples)
    s_ts = time_shuffle_surrogate(sig, rng_factory(0))
    assert np.allclose(np.sort(sig), np.sort(s_ts), atol=1e-10), \
        "time_shuffle did not preserve amplitude distribution"

    # All produce different signals from the original
    assert not np.allclose(sig, s_pr)
    assert not np.allclose(sig, s_iaaft)
    assert not np.allclose(sig, s_ts)
