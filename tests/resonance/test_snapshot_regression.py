"""Regression test: the new compute_resonance pipeline with the legacy-default
ResonanceConfig reproduces the snapshots captured from the pre-refactor
``compute_global_harmonicity`` on all reference signals.

Tolerance
---------
The snapshots in ``snapshots/*.npz`` were captured on Windows. CI runs on
Linux. FFT / FOOOF / scipy math routines exhibit platform-dependent floating
point drift in the last 4-5 significant digits — typically O(1e-6) absolute
and O(1e-4) relative for the magnitudes involved here. This is BLAS/LAPACK /
libc-math noise, not an algorithmic regression.

We therefore use ``rtol=1e-3, atol=1e-5`` — orders of magnitude tighter than
any drift introduced by an actual refactor bug, while tolerant of cross-platform
float noise. The looser bound still flags any meaningful algorithmic change:
a one-bin shift, an off-by-one in the reducer, a wrong PSD normalization, or
a kernel formula change all produce diffs ≥ 1e-2 in our experience.

If this test ever fails by a margin larger than the tolerance, regenerate the
snapshots by running ``generate_baseline.py`` on the pre-change commit and
diff the outputs to identify the real source of drift.
"""
import os
import sys

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _signals import (  # noqa: E402
    SIGNALS,
    BASELINE_CONFIG,
    legacy_default_resonance_config_kwargs,
)

from biotuner.resonance import compute_resonance, ResonanceConfig  # noqa: E402


SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")

# Cross-platform tolerances. See module docstring for rationale.
RTOL = 1e-3
ATOL = 1e-5


@pytest.mark.parametrize("name", list(SIGNALS.keys()))
def test_snapshot_matches_legacy_output(name):
    snapshot_path = os.path.join(SNAPSHOT_DIR, f"{name}.npz")
    if not os.path.exists(snapshot_path):
        pytest.skip(f"Snapshot {snapshot_path} missing — run generate_baseline.py first")

    sig = SIGNALS[name](sf=BASELINE_CONFIG["fs"])
    cfg = ResonanceConfig(**legacy_default_resonance_config_kwargs())
    result = compute_resonance(sig, sf=BASELINE_CONFIG["fs"], config=cfg)

    snap = np.load(snapshot_path, allow_pickle=True)
    np.testing.assert_allclose(result.factors["H"], snap["harmonicity"],
                                rtol=RTOL, atol=ATOL,
                                err_msg=f"H mismatch on {name}")
    np.testing.assert_allclose(result.factors["PC"], snap["phase_coupling"],
                                rtol=RTOL, atol=ATOL,
                                err_msg=f"PC mismatch on {name}")
    np.testing.assert_allclose(result.resonance_spectrum, snap["resonance"],
                                rtol=RTOL, atol=ATOL,
                                err_msg=f"R mismatch on {name}")


# NOTE: The legacy ``compute_global_harmonicity`` returned a *weighted*
# harmonicity matrix (S[i,j] * p[i] * p[j], diagonal zeroed) from its
# ``compute_harmonic_power`` step, not the raw kernel similarity matrix. The
# new orchestrator exposes the raw kernel matrix via ``intermediates``, which
# is more reusable. The legacy weighted form can be reconstructed from
# intermediates if needed; we don't enforce a snapshot match for it.
