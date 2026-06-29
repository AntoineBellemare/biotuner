"""Regression tests for biotuner.resonance.nm_detect — the validated cross-signal n:m detector.

Encodes the soundness properties established in resonance_paper studies 38-43:
  - correct Tass multipliers for a frequency pair;
  - a GENUINE cross-signal lock is detected by every panel metric;
  - INDEPENDENT signals and the WRONG ratio stay null;
  - a BIMODAL (antipodal) lock is detected by the all-moment indices (rho_entropy, phase_mi) but NOT by
    the first-moment PLV (regime-dependence, Study 41);
  - the scope guard warns on within-signal / shared-source input.
"""
import numpy as np
import pytest

from biotuner.resonance.nm_detect import detect_nm_coupling, nm_multipliers

SF = 500.0
DUR = 16.0
N = int(SF * DUR)
T = np.arange(N) / SF


def _norm(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def _phi(seed, f0=10.0, diff=0.5):
    r = np.random.default_rng(seed)
    return 2 * np.pi * f0 * T + np.cumsum(diff * np.sqrt(1.0 / SF) * r.standard_normal(N))


def _genuine_23(seed):
    """A @10 Hz, B @15 Hz, B phase-locked 2:3 to A (single stable relative phase)."""
    r = np.random.default_rng(seed)
    phi = _phi(seed)
    A = _norm(np.sin(phi) + 0.3 * r.standard_normal(N))
    B = _norm(np.sin(1.5 * phi + np.pi / 4) + 0.3 * r.standard_normal(N))
    return A, B


def _independent(seed):
    r = np.random.default_rng(seed)
    A = _norm(np.sin(_phi(seed)) + 0.3 * r.standard_normal(N))
    B = _norm(np.sin(_phi(seed + 1000, f0=15.0)) + 0.3 * r.standard_normal(N))
    return A, B


def _bimodal_23(seed):
    """Genuine 2:3 dependence but the relative phase flips antipodally (theta in {0, pi/2})."""
    r = np.random.default_rng(seed)
    phi = _phi(seed)
    sw = np.zeros(N)
    blk = int(SF * 1.5)
    for i in range(0, N, blk):
        sw[i:i + blk] = r.integers(0, 2)
    theta = (np.pi / 2.0) * sw                       # a=2 -> antipodal psi
    A = _norm(np.sin(phi) + 0.3 * r.standard_normal(N))
    B = _norm(np.sin(1.5 * phi + theta) + 0.3 * r.standard_normal(N))
    return A, B


def test_nm_multipliers():
    assert nm_multipliers(10.0, 15.0) == (3, 2)      # 2:3 -> 3*phi_a - 2*phi_b stationary
    assert nm_multipliers(10.0, 20.0) == (2, 1)      # 1:2
    assert nm_multipliers(10.0, 10.0) == (1, 1)
    assert nm_multipliers(12.0, 20.0) == (5, 3)      # 3:5


def test_genuine_lock_detected_by_whole_panel():
    A, B = _genuine_23(0)
    out = detect_nm_coupling(A, B, SF, [(10.0, 15.0)], n_surrogates=49, seed=1)
    res = out["results"][0]
    assert res["ratio"] == "3:2"
    for name, mr in res["metrics"].items():
        assert mr["z"] > 3.0, f"{name} z={mr['z']:.2f} should detect genuine lock"
        assert mr["rank_p"] <= 0.05, f"{name} rank_p={mr['rank_p']:.3f}"
    assert out["warning"] is None


def test_independent_is_null():
    A, B = _independent(2)
    out = detect_nm_coupling(A, B, SF, [(10.0, 15.0)], n_surrogates=49, seed=3)
    assert out["results"][0]["metrics"]["nm_plv"]["rank_p"] > 0.05


def test_wrong_ratio_is_null():
    # genuine 2:3 data (A@10, B@15), but probe a well-separated frequency the coupling does not
    # live at (10 vs 25 = 2:5); B has no 25 Hz component, so the pair must stay null.
    A, B = _genuine_23(4)
    out = detect_nm_coupling(A, B, SF, [(10.0, 25.0)], n_surrogates=49, seed=5)
    assert out["results"][0]["metrics"]["nm_plv"]["rank_p"] > 0.05


def test_bimodal_lock_regime_dependence():
    """PLV (first moment) is blind to an antipodal lock; rho_entropy / phase_mi recover it."""
    A, B = _bimodal_23(6)
    out = detect_nm_coupling(A, B, SF, [(10.0, 15.0)],
                             metrics=("nm_plv", "nm_rho_entropy", "nm_phase_mi"),
                             n_surrogates=49, seed=7)
    m = out["results"][0]["metrics"]
    assert m["nm_plv"]["rank_p"] > 0.05, "PLV should NOT detect an antipodal bimodal lock"
    assert m["nm_rho_entropy"]["z"] > 3.0, "rho_entropy should detect it"
    assert m["nm_phase_mi"]["z"] > 3.0, "phase_mi should detect it"


def test_scope_guard_warns_on_within_signal():
    x, _ = _genuine_23(8)
    with pytest.warns(UserWarning, match="confounded with waveform shape"):
        detect_nm_coupling(x, x, SF, [(10.0, 20.0)], n_surrogates=10, seed=9)
