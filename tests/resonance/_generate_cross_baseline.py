"""Generate baseline snapshots from the CURRENT compute_cross_spectrum_harmonicity.

Run BEFORE refactoring the cross-spectrum function to delegate to
compute_cross_resonance. The .npz files become the oracle that proves the
refactor preserves bit-exact numerics.

Usage:
    python tests/resonance/_generate_cross_baseline.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _signals import SIGNALS  # noqa: E402
from biotuner.harmonic_connectivity import compute_cross_spectrum_harmonicity  # noqa: E402

SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"
SNAPSHOT_DIR.mkdir(exist_ok=True, parents=True)


# Three signal-PAIRS representative of typical biotuner cross-channel usage.
# All built from the same SIGNALS dictionary used by the single-channel tests.
PAIRS = [
    ("harmonic_pink",      "harmonic_5_10_20_40", "pink_noise"),
    ("harmonic_inharmonic", "harmonic_5_10_20_40", "inharmonic_7_11_18_23"),
    ("pink_inharmonic",     "pink_noise",          "inharmonic_7_11_18_23"),
]


# Baseline config — the documented defaults of compute_cross_spectrum_harmonicity
# plus realistic values for precision_hz, fmin, fmax.
BASE = dict(
    precision_hz=0.5,
    fmin=2,
    fmax=30,
    noverlap=1,
    fs=1000,
    power_law_remove=False,
    n_peaks=5,
    metric="harmsim",
    n_harms=10,
    delta_lim=0.1,
    min_notes=2,
    plot=False,
    smoothness=1,
    smoothness_harm=1,
    phase_mode=None,
)


# Columns we'll persist for the regression test.
ARRAY_COLUMNS = [
    "harmonicity", "phase_coupling", "resonance",
    "harmonicity_peak_frequencies", "phase_peak_frequencies", "resonance_peak_frequencies",
]
SCALAR_COLUMNS = [
    "harm_spectral_flatness", "harm_spectral_entropy", "harm_higuchi", "harm_spectral_spread",
    "phase_spectral_flatness", "phase_spectral_entropy", "phase_higuchi", "phase_spectral_spread",
    "res_spectral_flatness", "res_spectral_entropy", "res_higuchi", "res_spectral_spread",
    "harmonicity_avg", "phase_coupling_avg", "resonance_avg",
    "harmonicity_peaks_avg", "phase_peaks_avg", "res_peaks_avg",
    "resonance_max",
    "harm_harmsim_avg", "phase_harmsim_avg", "res_harmsim_avg",
    "harm_harmsim_max", "phase_harmsim_max", "res_harmsim_max",
]


def _safe_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def main():
    for name, sig1_name, sig2_name in PAIRS:
        print(f"[cross-baseline] {name} ({sig1_name} x {sig2_name}) ...", flush=True)
        sig1 = SIGNALS[sig1_name](sf=BASE["fs"])
        sig2 = SIGNALS[sig2_name](sf=BASE["fs"])
        df = compute_cross_spectrum_harmonicity(sig1, sig2, **BASE)
        plt.close("all")

        out = {}
        for col in ARRAY_COLUMNS:
            val = df[col].iloc[0]
            out[col] = np.asarray(val, dtype=np.float64) if val is not None else np.array([], dtype=np.float64)
        out["_scalar_names"] = np.array(SCALAR_COLUMNS, dtype=object)
        out["_scalar_values"] = np.array([_safe_float(df[c].iloc[0]) for c in SCALAR_COLUMNS], dtype=np.float64)

        path = SNAPSHOT_DIR / f"cross_{name}.npz"
        np.savez_compressed(path, **out)
        print(f"  -> {path.name}  H shape={out['harmonicity'].shape}, "
              f"PC shape={out['phase_coupling'].shape}, R shape={out['resonance'].shape}")

    print("[cross-baseline] done")


if __name__ == "__main__":
    main()
