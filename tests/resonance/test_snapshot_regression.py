"""Bit-exact regression test: the new compute_resonance pipeline with the
legacy-default ResonanceConfig must reproduce the snapshots captured from the
pre-refactor compute_global_harmonicity within atol=1e-6 on all reference
signals.

If this test ever fails after an intentional numeric change, regenerate the
snapshots by running ``generate_baseline.py`` on the pre-change commit and
diff the outputs.
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
ATOL = 1e-6


@pytest.mark.parametrize("name", list(SIGNALS.keys()))
def test_snapshot_matches_legacy_output(name):
    snapshot_path = os.path.join(SNAPSHOT_DIR, f"{name}.npz")
    if not os.path.exists(snapshot_path):
        pytest.skip(f"Snapshot {snapshot_path} missing — run generate_baseline.py first")

    sig = SIGNALS[name](sf=BASELINE_CONFIG["fs"])
    cfg = ResonanceConfig(**legacy_default_resonance_config_kwargs())
    result = compute_resonance(sig, sf=BASELINE_CONFIG["fs"], config=cfg)

    snap = np.load(snapshot_path, allow_pickle=True)
    np.testing.assert_allclose(result.factors["H"], snap["harmonicity"], atol=ATOL,
                                err_msg=f"H mismatch on {name}")
    np.testing.assert_allclose(result.factors["PC"], snap["phase_coupling"], atol=ATOL,
                                err_msg=f"PC mismatch on {name}")
    np.testing.assert_allclose(result.resonance_spectrum, snap["resonance"], atol=ATOL,
                                err_msg=f"R mismatch on {name}")


# NOTE: The legacy ``compute_global_harmonicity`` returned a *weighted*
# harmonicity matrix (S[i,j] * p[i] * p[j], diagonal zeroed) from its
# ``compute_harmonic_power`` step, not the raw kernel similarity matrix. The
# new orchestrator exposes the raw kernel matrix via ``intermediates``, which
# is more reusable. The legacy weighted form can be reconstructed from
# intermediates if needed; we don't enforce a snapshot match for it.
