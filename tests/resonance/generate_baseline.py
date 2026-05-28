"""Generate regression snapshots from the CURRENT compute_global_harmonicity.

Run BEFORE any resonance-refactor changes. The .npz files written by this
script become the oracle for the snapshot regression test that validates
the new `compute_resonance` orchestrator reproduces legacy outputs within
atol=1e-6 when called with the legacy-default ResonanceConfig.

Re-running this AFTER refactor would defeat the purpose. To regenerate
intentionally (e.g. after an intentional numeric change), delete the .npz
files and re-run on the legacy commit.

Usage:
    python tests/resonance/generate_baseline.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# Ensure project root is on sys.path when run as a script
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from biotuner.harmonic_spectrum import compute_global_harmonicity
except ImportError as exc:
    raise RuntimeError(
        "compute_global_harmonicity has been removed in the resonance-package "
        "refactor. This baseline-generation script is a one-time tool to be "
        "run on the PRE-REFACTOR commit (before commit 478ccce). To regenerate "
        "snapshots from the current code, check out main first:\n"
        "    git checkout main\n"
        "    python tests/resonance/generate_baseline.py\n"
        "    git checkout <your-branch>\n"
    ) from exc

SNAPSHOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)


def _harmonic_signal(sf=1000, duration=4.0, seed=0):
    """Strongly harmonic 1:2:4:8 sine bundle (same as test_harmonic_spectrum fixture)."""
    t = np.linspace(0, duration, int(sf * duration), endpoint=False)
    rng = np.random.default_rng(seed)
    sig = sum(
        (1.0 / (i + 1)) * np.sin(2 * np.pi * f * t)
        for i, f in enumerate([5, 10, 20, 40])
    )
    sig += 0.02 * rng.standard_normal(len(t))
    return sig.astype(np.float64)


def _pink_noise(sf=1000, duration=4.0, seed=1):
    """1/f pink noise via FFT shaping."""
    rng = np.random.default_rng(seed)
    n = int(sf * duration)
    white = rng.standard_normal(n)
    f = np.fft.rfftfreq(n, d=1.0 / sf)
    f[0] = f[1]  # avoid div-by-zero
    spectrum = np.fft.rfft(white) / np.sqrt(f)
    pink = np.fft.irfft(spectrum, n=n)
    return (pink / np.std(pink)).astype(np.float64)


def _inharmonic_signal(sf=1000, duration=4.0, seed=2):
    """Inharmonic mixture: 7, 11.3, 17.9, 23 Hz — no simple integer ratios."""
    t = np.linspace(0, duration, int(sf * duration), endpoint=False)
    rng = np.random.default_rng(seed)
    sig = sum(
        np.sin(2 * np.pi * f * t)
        for f in [7.0, 11.3, 17.9, 23.0]
    )
    sig += 0.05 * rng.standard_normal(len(t))
    return sig.astype(np.float64)


SIGNALS = {
    "harmonic_5_10_20_40": _harmonic_signal,
    "pink_noise": _pink_noise,
    "inharmonic_7_11_18_23": _inharmonic_signal,
}

# Single canonical config — matches the defaults exercised by current tests
# and notebooks. If you need additional regression points, add them here.
BASELINE_CONFIG = dict(
    precision_hz=0.5,
    fmin=2,
    fmax=30,
    fs=1000,
    noverlap=1,
    power_law_remove=True,
    n_peaks=5,
    metric="harmsim",
    n_harms=10,
    delta_lim=20,
    min_notes=2,
    plot=False,
    smoothness=1,
    smoothness_harm=1,
    phase_mode=None,
    normalize=True,
    bandwidth_correction=False,
    detrend_harmonicity=False,
)


# Scalar df columns we'll round-trip — the array columns (harmonicity,
# phase_coupling, resonance, *_peak_frequencies) are saved separately as
# arrays for tolerance comparisons.
SCALAR_COLUMNS = [
    "harm_spectral_flatness", "harm_spectral_entropy", "harm_higuchi", "harm_spectral_spread",
    "phase_spectral_flatness", "phase_spectral_entropy", "phase_higuchi", "phase_spectral_spread",
    "res_spectral_flatness", "res_spectral_entropy", "res_higuchi", "res_spectral_spread",
    "harmonicity_avg", "phase_coupling_avg", "resonance_avg",
    "harmonicity_peaks_avg", "phase_peaks_avg", "res_peaks_avg",
    "resonance_max", "harmonicity_max", "phase_coupling_max",
    "harm_harmsim_avg", "phase_harmsim_avg", "res_harmsim_avg",
    "harm_harmsim_max", "phase_harmsim_max", "res_harmsim_max",
]

ARRAY_COLUMNS = [
    "harmonicity", "phase_coupling", "resonance",
    "harmonicity_peak_frequencies", "phase_peak_frequencies", "resonance_peak_frequencies",
]


def _safe_float(x):
    """Map NaN/None/non-numeric to np.nan so np.savez stores them."""
    try:
        v = float(x)
        return v
    except (TypeError, ValueError):
        return np.nan


def _row_to_arrays(row):
    """Convert a single-row pandas df into a dict suitable for np.savez."""
    out = {}
    for col in ARRAY_COLUMNS:
        val = row[col].iloc[0]
        out[col] = np.asarray(val, dtype=np.float64) if val is not None else np.array([], dtype=np.float64)
    scalars = np.array([_safe_float(row[col].iloc[0]) for col in SCALAR_COLUMNS], dtype=np.float64)
    out["_scalar_values"] = scalars
    out["_scalar_names"] = np.array(SCALAR_COLUMNS, dtype=object)
    return out


def main():
    for name, signal_fn in SIGNALS.items():
        print(f"[baseline] {name} ...", flush=True)
        sig = signal_fn(sf=BASELINE_CONFIG["fs"])
        df, matrix = compute_global_harmonicity(sig, **BASELINE_CONFIG)
        plt.close("all")
        arrays = _row_to_arrays(df)
        arrays["harmonicity_matrix"] = np.asarray(matrix, dtype=np.float64)
        out_path = os.path.join(SNAPSHOT_DIR, f"{name}.npz")
        np.savez_compressed(out_path, **arrays)
        print(f"  -> {out_path}  shapes: H={arrays['harmonicity'].shape}, "
              f"PC={arrays['phase_coupling'].shape}, R={arrays['resonance'].shape}, "
              f"M={arrays['harmonicity_matrix'].shape}", flush=True)

    # Also persist the exact config used, so the snapshot test can replay it.
    cfg_path = os.path.join(SNAPSHOT_DIR, "_baseline_config.npz")
    np.savez(cfg_path,
             keys=np.array(list(BASELINE_CONFIG.keys()), dtype=object),
             values=np.array([str(v) for v in BASELINE_CONFIG.values()], dtype=object))
    print(f"[baseline] wrote {cfg_path}")
    print("[baseline] done")


if __name__ == "__main__":
    main()
