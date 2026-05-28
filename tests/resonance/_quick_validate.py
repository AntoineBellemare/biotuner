"""Quick snapshot validation harness — used during refactor development.

Compares `compute_resonance(legacy_default_config)` output to the .npz snapshots
captured before any code changes by `generate_baseline.py`. The proper pytest-based
regression test lives at `test_snapshot_regression.py`.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _signals import SIGNALS, BASELINE_CONFIG, legacy_default_resonance_config_kwargs  # noqa: E402
from biotuner.resonance import compute_resonance, ResonanceConfig  # noqa: E402

SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
ATOL = 1e-6

failures = 0
for name, sig_fn in SIGNALS.items():
    sig = sig_fn(sf=BASELINE_CONFIG["fs"])
    cfg = ResonanceConfig(**legacy_default_resonance_config_kwargs())
    result = compute_resonance(sig, sf=BASELINE_CONFIG["fs"], config=cfg)

    snap = np.load(os.path.join(SNAPSHOT_DIR, f"{name}.npz"), allow_pickle=True)
    h_diff = float(np.max(np.abs(result.factors["H"] - snap["harmonicity"])))
    pc_diff = float(np.max(np.abs(result.factors["PC"] - snap["phase_coupling"])))
    r_diff = float(np.max(np.abs(result.resonance_spectrum - snap["resonance"])))
    status = "PASS" if max(h_diff, pc_diff, r_diff) < ATOL else "FAIL"
    if status == "FAIL":
        failures += 1
    print(f"  [{status}] {name:32s}  H={h_diff:.2e}  PC={pc_diff:.2e}  R={r_diff:.2e}")

print(f"\n{failures} failure(s) across {len(SIGNALS)} signals at atol={ATOL}")
sys.exit(0 if failures == 0 else 1)
