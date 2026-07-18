"""Tests for biotuner.biocolors.

Grouped by layer: the colour foundation (spaces, gamut, spectral, difference),
the descriptor/fingerprint layer, calibration, the mapping methods, the public
palette API, and legacy back-compatibility. Nothing here depends on external
recordings -- signals are synthetic and calibrations ship with the package.
"""
import warnings

import numpy as np
import pytest

from biotuner.biocolors import color
from biotuner.biocolors.color import (
    srgb_to_oklab, oklab_to_srgb, srgb_to_oklch, oklch_to_srgb,
    max_chroma, cusp, gamut_map, in_gamut, deltaE_ok, simulate_cvd,
    wavelength_to_srgb, audible_to_nm, toe, toe_inv, CVD_KINDS,
)
from biotuner.biocolors import (
    palette_from_signal, palette_from_tuning, palette_from_biotuner,
    diversity_report, palette_report, load_calibration, MAPPINGS, LEVELS,
    Calibration, build_calibration,
)
from biotuner.biocolors.descriptors import (
    make_context, fingerprint, FINGERPRINT_FIELDS, TUNING_FIELDS,
    amps_scale_for,
)

RNG = np.random.default_rng(0)
JI = np.array([1, 9 / 8, 5 / 4, 4 / 3, 3 / 2, 5 / 3, 15 / 8, 2.0])
# A synthetic spectrum with dB-like amplitudes (some negative, as real extractors give).
PK = np.array([1.0, 4.0, 7.75, 14.5, 29.25])
AM = np.array([17.4, 10.7, 6.1, 2.8, -10.1])


# --------------------------------------------------------------------------- #
# Colour spaces
# --------------------------------------------------------------------------- #
def test_oklab_roundtrip_is_lossless():
    grid = RNG.random((4000, 3))
    back = oklab_to_srgb(srgb_to_oklab(grid))
    assert np.abs(back - grid).max() < 1e-6


def test_srgb_to_oklab_is_vectorized_over_last_axis():
    px = RNG.random((7, 3))
    batched = srgb_to_oklab(px)
    per_row = np.array([srgb_to_oklab(p) for p in px])
    assert batched.shape == (7, 3)
    assert np.allclose(batched, per_row)
    # higher rank
    img = RNG.random((5, 6, 3))
    assert srgb_to_oklab(img).shape == (5, 6, 3)


def test_toe_roundtrip():
    Ls = np.linspace(0, 1, 100)
    assert np.abs(toe_inv(toe(Ls)) - Ls).max() < 1e-9


def test_red_lands_near_29_degrees_not_zero():
    # The prototype misdiagnosis: hue 0 is not the magenta axis; red is ~29 deg.
    _, _, h = srgb_to_oklch(np.array([1.0, 0.0, 0.0]))
    assert 25.0 < h < 33.0


# --------------------------------------------------------------------------- #
# Gamut
# --------------------------------------------------------------------------- #
def test_gamut_map_lands_in_gamut_and_preserves_hue():
    lch = np.stack([
        RNG.uniform(0.15, 0.95, 3000),
        RNG.uniform(0.0, 0.45, 3000),
        RNG.uniform(0, 360, 3000),
    ], axis=-1)
    mapped = gamut_map(lch)
    assert np.mean(in_gamut(mapped, tol=1e-5)) > 0.999
    # hue is never changed
    assert np.allclose(mapped[..., 2], lch[..., 2])


def test_gamut_map_leaves_in_gamut_colours_untouched():
    lch = np.stack([
        RNG.uniform(0.3, 0.7, 500),
        np.full(500, 0.02),   # tiny chroma -> always in gamut
        RNG.uniform(0, 360, 500),
    ], axis=-1)
    assert np.all(in_gamut(lch))
    assert np.allclose(gamut_map(lch), lch)


def test_cusp_ceiling_varies_strongly_around_wheel():
    hues = np.arange(0, 360, 30.0)
    ceilings = np.array([max_chroma(0.6, h) for h in hues])
    assert ceilings.min() > 0
    # measured ~5-6x variation across the wheel
    assert ceilings.max() / ceilings.min() > 2.0


def test_saturated_red_at_high_lightness_does_not_exist():
    assert max_chroma(0.88, 29.0) < 0.10   # measured ~0.064


# --------------------------------------------------------------------------- #
# Spectral locus
# --------------------------------------------------------------------------- #
def test_wavelength_to_srgb_in_unit_cube():
    nm = np.linspace(380, 750, 200)
    for method in ("cie1931", "bruton"):
        rgb = wavelength_to_srgb(nm, method=method)
        assert rgb.shape == (200, 3)
        assert rgb.min() >= 0.0 and rgb.max() <= 1.0


def test_audible_to_nm_stays_in_band():
    freqs = np.linspace(1, 1000, 5000)
    nm, _ = audible_to_nm(freqs, mode="fold")
    assert nm.min() >= 380.0 - 1e-6 and nm.max() <= 750.0 + 1e-6
    nm_wrap, _ = audible_to_nm(freqs, mode="wrap")
    assert nm_wrap.min() >= 380.0 - 1e-6 and nm_wrap.max() <= 750.0 + 1e-6


def test_cie1931_has_finer_hue_resolution_than_bruton():
    # Bruton collapses the red end onto one hue; CIE spreads it.
    nm = np.linspace(380, 750, 371)
    def widest_plateau(method):
        h = srgb_to_oklch(wavelength_to_srgb(nm, method=method))[..., 2]
        widest, start = 0.0, 0
        for i in range(1, len(nm)):
            if abs(h[i] - h[start]) >= 0.5:
                widest = max(widest, nm[i - 1] - nm[start]); start = i
        return max(widest, nm[-1] - nm[start])
    assert widest_plateau("cie1931") < widest_plateau("bruton")


# --------------------------------------------------------------------------- #
# Difference and CVD
# --------------------------------------------------------------------------- #
def test_deltaE_zero_and_positive():
    lab = srgb_to_oklab(np.array([1.0, 0, 0]))
    assert deltaE_ok(lab, lab) == 0.0
    assert deltaE_ok(srgb_to_oklab(np.array([1.0, 0, 0])),
                     srgb_to_oklab(np.array([0.0, 1, 0]))) > 0.3


def test_cvd_collapses_red_green():
    red = np.array([1.0, 0, 0]); grn = np.array([0.0, 1, 0])
    d_normal = deltaE_ok(srgb_to_oklab(red), srgb_to_oklab(grn))
    d = simulate_cvd(np.stack([red, grn]), kind="deutan")
    d_deutan = deltaE_ok(srgb_to_oklab(d[0]), srgb_to_oklab(d[1]))
    assert d_deutan < d_normal


@pytest.mark.parametrize("kind", CVD_KINDS)
def test_cvd_output_in_unit_cube(kind):
    out = simulate_cvd(RNG.random((50, 3)), kind=kind)
    assert out.min() >= 0.0 and out.max() <= 1.0


# --------------------------------------------------------------------------- #
# Descriptors and fingerprint
# --------------------------------------------------------------------------- #
def test_fingerprint_has_11_finite_fields():
    fp = fingerprint(make_context(PK, AM, amps_scale="db"))
    assert len(fp.values) == len(FINGERPRINT_FIELDS) == 11
    assert np.all(np.isfinite(fp.vector()))


def test_tuning_fingerprint_has_7_fields():
    fp = fingerprint(make_context(JI, None, amps_scale="linear"), fields=TUNING_FIELDS)
    assert len(fp.values) == len(TUNING_FIELDS) == 7
    assert np.all(np.isfinite(fp.vector(TUNING_FIELDS)))


def test_amps_scale_lookup():
    assert amps_scale_for("fixed") == "db"
    assert amps_scale_for("FOOOF") == "linear"


def test_negative_db_amps_rectified_positive():
    ctx = make_context(PK, AM, amps_scale="db")
    assert np.all(ctx.amps >= 0)
    assert np.isclose(ctx.amps.sum(), 1.0)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ["eeg_sleep_v1", "tuning_v1", "tuning_eeg_v1"])
def test_shipped_calibrations_load(name):
    cal = load_calibration(name)
    assert isinstance(cal, Calibration)
    assert cal.pca_components.shape[0] == 2


def test_normalize_in_unit_interval():
    cal = load_calibration("eeg_sleep_v1")
    for f in cal.fields:
        for v in (-1e9, 0.0, 1e9):
            t = cal.normalize(f, v)
            assert 0.0 <= t <= 1.0


def test_identity_calibration():
    cal = load_calibration("none")
    assert cal.name == "identity"


def test_build_calibration_from_fingerprints():
    fps = [fingerprint(make_context(np.sort(RNG.uniform(1, 40, 5)),
                                    RNG.uniform(-10, 20, 5), amps_scale="db"))
           for _ in range(30)]
    cal = build_calibration(fps, name="test")
    assert 0.0 <= cal.meta["pca_explained_2d"] <= 1.0


# --------------------------------------------------------------------------- #
# Mapping methods
# --------------------------------------------------------------------------- #
ALL_METHODS = ["anchored", "spectral", "mds", "tonotopic", "consonance",
               "harmonic", "tenney", "derived"]


def test_all_methods_registered():
    assert set(ALL_METHODS) <= set(MAPPINGS)


@pytest.mark.parametrize("method", ALL_METHODS)
def test_method_produces_finite_in_gamut_palette(method):
    p = palette_from_signal(PK, AM, method=method)
    assert len(p) == len(PK)
    assert np.all(np.isfinite(p.rgb))
    assert p.rgb.min() >= 0.0 and p.rgb.max() <= 1.0


def test_consonance_space_hue_is_monotone_in_consonance():
    from biotuner.biocolors.descriptors import compute
    ctx = make_context(PK, AM, amps_scale="db")
    p = palette_from_signal(PK, AM, method="consonance")
    cons = np.asarray(compute("consonance", ctx), float)
    order = np.argsort(cons)
    h = p.spec.h[order]
    # monotone (hue range set so consonant -> low hue); allow either direction
    diffs = np.diff(h)
    assert np.all(diffs <= 1e-6) or np.all(diffs >= -1e-6)


# --------------------------------------------------------------------------- #
# Public palette API
# --------------------------------------------------------------------------- #
def test_palette_from_signal_basic():
    p = palette_from_signal(PK, AM)
    assert len(p.hex()) == len(PK)
    assert all(c.startswith("#") and len(c) == 7 for c in p.hex())


def test_palette_from_tuning_is_finite():
    p = palette_from_tuning(JI)
    assert np.all(np.isfinite(p.rgb))
    assert len(p) == len(JI)


def test_palette_report_keys():
    r = palette_from_signal(PK, AM).report()
    for k in ("min_deltaE", "min_adjacent_deltaE", "fingerprint_saturation", "gamut"):
        assert k in r
    assert 0.0 <= r["fingerprint_saturation"] <= 1.0


@pytest.mark.filterwarnings("ignore:.*fall outside calibration.*:RuntimeWarning")
def test_diversity_report_on_a_set():
    # Random uniform-amp signals are out of the sleep corpus domain; the domain
    # warning is expected here and silenced so it does not clutter the run.
    sigs = [np.sort(RNG.uniform(1, 40, 5)) for _ in range(6)]
    pals = [palette_from_signal(s, np.ones(5)) for s in sigs]
    rep = diversity_report(pals)
    assert rep["n"] == 6
    assert rep["mean_palette_deltaE"] >= 0.0


def test_temperament_changes_lightness_range():
    lo = palette_from_signal(PK, AM, temperament="deep").report()["lightness_range"]
    hi = palette_from_signal(PK, AM, temperament="pastel").report()["lightness_range"]
    assert hi[0] > lo[0]   # pastel sits lighter than deep


def test_aurora_spans_wider_arc_than_earthy():
    a = palette_from_signal(PK, AM, temperament="aurora").report()["hue_arc"]
    e = palette_from_signal(PK, AM, temperament="earthy").report()["hue_arc"]
    assert a > e


# --------------------------------------------------------------------------- #
# palette_from_biotuner with a minimal stand-in object
# --------------------------------------------------------------------------- #
class _FakeBT:
    """Minimal duck-type of compute_biotuner for the peaks level."""
    peaks_function = "fixed"
    peaks = PK
    amps = AM


def test_palette_from_biotuner_peaks_level():
    p = palette_from_biotuner(_FakeBT(), level="peaks")
    assert len(p) == len(PK)
    assert np.all(np.isfinite(p.rgb))


def test_levels_constant():
    assert LEVELS == ("peaks", "extended", "ratios", "extended_ratios", "cons_ratios")


def _synth_signal(sf=250, dur=8):
    t = np.arange(0, dur, 1 / sf)
    return t, (np.sin(2 * np.pi * 10 * t) + 0.6 * np.sin(2 * np.pi * 20 * t)
               + 0.8 * np.sin(2 * np.pi * 6 * t) + 0.4 * np.sin(2 * np.pi * 15 * t)
               + 0.5 * RNG.standard_normal(len(t)))


def test_palette_from_raw_end_to_end():
    # The raw-signal front door: extraction params are chosen here, amps are
    # auto-scaled, and the recovered peaks drive the palette.
    from biotuner.biocolors import palette_from_raw
    sf = 250
    _, sig = _synth_signal(sf)
    p, bt = palette_from_raw(sig, sf, peaks_function="fixed", precision=0.25,
                             n_peaks=5, return_bt=True)
    assert np.all(np.isfinite(p.rgb))
    assert p.metadata["extraction"]["precision"] == 0.25
    peaks = np.asarray(bt.peaks, float)
    assert np.any(np.abs(peaks - 10) < 1.0) and np.any(np.abs(peaks - 20) < 1.0)


def test_palette_from_raw_with_tuning_source():
    # biocolors colours a signal-derived tuning; the derivation itself lives in
    # biotuner_object (tested there). Here we check the colour integration:
    # palette_from_raw(tuning=...) delegates to bt.get_tuning and routes the
    # measured tuning to the tuning-EEG calibration.
    from biotuner.biocolors import palette_from_raw
    sf = 250
    _, sig = _synth_signal(sf)
    p = palette_from_raw(sig, sf, tuning="diss_curve", precision=0.5)
    assert np.all(np.isfinite(p.rgb))
    assert p.metadata["extraction"]["tuning_source"] == "diss_curve"
    assert p.calibration.name == "tuning_eeg_v1"   # signal-derived tuning routes here


# --------------------------------------------------------------------------- #
# The spectral-on-ratios guard
# --------------------------------------------------------------------------- #
def test_spectral_on_dimensionless_ratios_warns():
    with pytest.warns(RuntimeWarning, match="spectral"):
        palette_from_tuning(JI, method="spectral")


def test_spectral_on_real_fund_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # any spectral warning would raise
        palette_from_tuning(JI, fund=440.0, method="spectral")


def test_anchored_on_ratios_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        palette_from_tuning(JI, method="anchored")


# --------------------------------------------------------------------------- #
# Legacy back-compatibility
# --------------------------------------------------------------------------- #
def test_legacy_wavelength_to_rgb_unchanged():
    from biotuner.biocolors import wavelength_to_rgb
    assert wavelength_to_rgb(475) == (0, 213, 255)


def test_legacy_names_importable():
    from biotuner.biocolors import (
        wavelength_to_rgb, scale2freqs, nm2Hz, Hz2nm, Hz2THz, THz2Hz,
        audible2visible, wavelength_to_frequency,
    )
    assert callable(wavelength_to_rgb) and callable(audible2visible)
