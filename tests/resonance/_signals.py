"""Reference signal generators shared by ``generate_baseline.py`` and the
snapshot regression test (``test_snapshot_regression.py``). Kept separate so the
test can run without importing the deleted legacy
``compute_global_harmonicity`` symbol.
"""

import numpy as np


def harmonic_signal(sf=1000, duration=4.0, seed=0):
    """Strongly harmonic 1:2:4:8 sine bundle (same as test_harmonic_spectrum fixture)."""
    t = np.linspace(0, duration, int(sf * duration), endpoint=False)
    rng = np.random.default_rng(seed)
    sig = sum(
        (1.0 / (i + 1)) * np.sin(2 * np.pi * f * t)
        for i, f in enumerate([5, 10, 20, 40])
    )
    sig += 0.02 * rng.standard_normal(len(t))
    return sig.astype(np.float64)


def pink_noise(sf=1000, duration=4.0, seed=1):
    """1/f pink noise via FFT shaping."""
    rng = np.random.default_rng(seed)
    n = int(sf * duration)
    white = rng.standard_normal(n)
    f = np.fft.rfftfreq(n, d=1.0 / sf)
    f[0] = f[1]
    spectrum = np.fft.rfft(white) / np.sqrt(f)
    pink = np.fft.irfft(spectrum, n=n)
    return (pink / np.std(pink)).astype(np.float64)


def inharmonic_signal(sf=1000, duration=4.0, seed=2):
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
    "harmonic_5_10_20_40": harmonic_signal,
    "pink_noise": pink_noise,
    "inharmonic_7_11_18_23": inharmonic_signal,
}


# The legacy ``compute_global_harmonicity`` config used to generate the baseline.
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


def legacy_default_resonance_config_kwargs(baseline_config=None):
    """Return kwargs for :class:`biotuner.resonance.ResonanceConfig` that reproduce
    the legacy compute_global_harmonicity numerics for snapshot regression."""
    bc = baseline_config or BASELINE_CONFIG
    return dict(
        precision_hz=bc["precision_hz"],
        fmin=bc["fmin"],
        fmax=bc["fmax"],
        noverlap=bc["noverlap"],
        smoothness=bc["smoothness"],
        n_peaks=bc["n_peaks"],
        remove_aperiodic=bc["power_law_remove"],
        psd_normalization="minmax_prob",
        harmonic_kernel=bc["metric"],
        harmonic_kernel_params={
            "n_harms": bc["n_harms"],
            "delta_lim": bc["delta_lim"],
            "min_notes": bc["min_notes"],
        },
        ratio_kernel="binary",
        ratio_kernel_params={"max_nm": 3, "tolerance": 0.05, "fallback_to_1_1": True},
        phase_estimator="stft",
        coupling_metric="nm_plv",
        gaussian_smooth_sigma=bc["smoothness_harm"],
        detrend=bc["detrend_harmonicity"],
        rescale_factors_after_detrend=True,
        legacy_self_pair_subtract=True,
        normalize=bc["normalize"],
        bandwidth_correction=bc["bandwidth_correction"],
        combine="product",
    )
