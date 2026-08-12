"""End-to-end tests: MOS scales reached through the ordinary biotuner surfaces.

A signal goes in, a well-formed scale comes out, and every existing tuning
consumer keeps working on it.
"""

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biotuner.biotuner_object import TUNING_SOURCES, compute_biotuner
from biotuner.mos.scale import MOSScale

SF = 1000.0


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def harmonic_signal():
    """A signal whose peaks sit on a stack of fifths, plus noise."""
    rng = np.random.default_rng(0)
    t = np.arange(0, 20, 1 / SF)
    base = 5.0
    freqs = [base, base * 1.5, base * 2.25, base * 3.375, base * 2.0]
    x = sum(a * np.sin(2 * np.pi * f * t)
            for a, f in zip([1.0, 0.9, 0.7, 0.5, 0.6], freqs))
    return x + 0.15 * rng.standard_normal(t.size)


@pytest.fixture(scope="module")
def fitted(harmonic_signal):
    bt = compute_biotuner(SF, peaks_function="fixed", precision=0.1)
    bt.peaks_extraction(harmonic_signal, min_freq=2, max_freq=40, n_peaks=5)
    bt.fit_mos(max_cardinality=16)
    return bt


@pytest.fixture(scope="module")
def probe(harmonic_signal):
    """A spectrally parameterised object with every precursor a source needs.

    ``peaks_extension`` is what makes ``extended_ratios`` exist at all, so an
    object without it cannot answer the question this section asks.
    """
    bt = compute_biotuner(SF, peaks_function="FOOOF", precision=0.1)
    bt.peaks_extraction(harmonic_signal, min_freq=2, max_freq=40, n_peaks=5)
    bt.peaks_extension(method="harmonic_fit")
    return bt


@pytest.fixture(scope="module")
def recurrent(harmonic_signal):
    """The one extraction that measures harmonic positions.

    ``harm_tuning`` is built out of the harmonic numbers at which peaks were
    found to recur, and only ``peaks_function='harmonic_recurrence'`` measures
    them -- so it is the object that source has to be asked of.
    """
    bt = compute_biotuner(SF, peaks_function="harmonic_recurrence", precision=0.1)
    bt.peaks_extraction(harmonic_signal, min_freq=2, max_freq=40, n_peaks=5)
    return bt


# --------------------------------------------------------------------------- #
# The object surface
# --------------------------------------------------------------------------- #
def test_mos_is_a_registered_tuning_source():
    assert "mos" in TUNING_SOURCES


def test_fit_mos_sets_the_documented_attributes(fitted):
    assert hasattr(fitted, "mos_fit")
    assert hasattr(fitted, "mos_fits")
    assert hasattr(fitted, "mos_scale")
    assert fitted.mos_scale is fitted.mos_fit.scale
    assert isinstance(fitted.mos_scale, MOSScale)
    assert len(fitted.mos_fits) >= 1


def test_fits_are_ranked(fitted):
    scores = [f.score for f in fitted.mos_fits]
    assert scores == sorted(scores)
    assert fitted.mos_fit is fitted.mos_fits[0]


def test_the_fit_is_a_well_formed_scale(fitted):
    s = fitted.mos_scale
    assert math.gcd(s.n_large, s.n_small) == 1
    assert s.cardinality == s.n_large + s.n_small
    large, small = s.step_cents
    assert s.n_large * large + s.n_small * small == pytest.approx(1200.0, abs=1e-6)


def test_get_tuning_returns_the_fitted_scale(fitted):
    tuning = fitted.get_tuning("mos")
    assert len(tuning) == fitted.mos_scale.cardinality
    assert np.allclose(tuning, fitted.mos_scale.ratios)


def test_private_tuning_helper_also_knows_mos(fitted):
    assert np.allclose(
        fitted._get_tuning_data("mos"), list(fitted.mos_scale.ratios)
    )


def test_get_tuning_computes_a_fit_on_demand(harmonic_signal):
    bt = compute_biotuner(SF, peaks_function="fixed", precision=0.1)
    bt.peaks_extraction(harmonic_signal, min_freq=2, max_freq=40, n_peaks=5)
    assert not hasattr(bt, "mos_fit")
    tuning = bt.get_tuning("mos")
    assert len(tuning) > 0
    assert hasattr(bt, "mos_fit")


def test_an_unknown_source_still_lists_mos(fitted):
    with pytest.raises(ValueError, match="mos"):
        fitted._get_tuning_data("not_a_tuning")


def test_object_labyrinth_plot(fitted):
    fig, ax = fitted.plot_labyrinth(14)
    assert ax.name == "polar"


def test_object_labyrinth_plot_without_a_fit(harmonic_signal):
    bt = compute_biotuner(SF, peaks_function="fixed", precision=0.1)
    bt.peaks_extraction(harmonic_signal, min_freq=2, max_freq=40, n_peaks=5)
    fig, ax = bt.plot_labyrinth(12, highlight_fit=False)
    assert ax is not None


def test_object_trajectory(harmonic_signal):
    bt = compute_biotuner(SF, peaks_function="fixed", precision=0.5)
    traj = bt.mos_trajectory(
        harmonic_signal, SF, window_sec=5.0, step_sec=5.0, max_cardinality=12
    )
    assert len(traj) == 4
    assert any(f is not None for f in traj)


# --------------------------------------------------------------------------- #
# Every derivation can feed the fit
# --------------------------------------------------------------------------- #
def test_fitting_a_mos_to_a_mos_is_refused(fitted):
    """get_tuning('mos') returns an earlier fit, so the answer is guaranteed."""
    from biotuner.mos.derive import mos_from_biotuner

    with pytest.raises(ValueError, match="would fit a moment-of-symmetry"):
        mos_from_biotuner(fitted, source="mos")
    with pytest.raises(ValueError, match="would fit a moment-of-symmetry"):
        fitted.fit_mos(source="mos")


@pytest.mark.parametrize(
    "source", ["peaks_ratios", "cons_ratios", "diss_curve", "HE",
               "euler_fokker", "harm_fit_tuning"]
)
def test_each_derivation_can_be_fitted(fitted, source):
    from biotuner.mos.derive import mos_from_biotuner

    fits = mos_from_biotuner(fitted, source=source, max_cardinality=14, top_n=1)
    assert fits and fits[0].scale.cardinality >= 4


def test_compare_sources_ranks_every_derivation(fitted):
    from biotuner.mos.derive import compare_sources

    df = compare_sources(fitted, max_cardinality=14)
    assert set(df["source"]) == {s for s in TUNING_SOURCES if s != "mos"}
    assert "mos" not in set(df["source"])          # circular, never included
    for column in ("signature", "error_cents", "chance_error_cents",
                   "evidence", "coverage", "n_targets", "n_merged",
                   "underdetermined", "reason"):
        assert column in df.columns
    scored = df["evidence"].dropna()
    assert list(scored) == sorted(scored, reverse=True)


def test_compare_sources_reports_a_broken_source_rather_than_hiding_it(fitted):
    """``harm_tuning`` raises for every peaks_function but harmonic_recurrence.
    A shorter table would not be a report of that."""
    from biotuner.mos.derive import compare_sources

    df = compare_sources(fitted, sources=["peaks_ratios", "harm_tuning"],
                         max_cardinality=14)
    broken = df[df["source"] == "harm_tuning"].iloc[0]
    assert broken["reason"] is not None
    assert broken["signature"] is None
    assert df[df["source"] == "peaks_ratios"].iloc[0]["reason"] is None
    # Failures sort last, whatever they were asked in.
    assert df["source"].iloc[-1] == "harm_tuning"


def test_compare_sources_honours_an_explicit_source_list(fitted):
    from biotuner.mos.derive import compare_sources

    df = compare_sources(fitted, sources=["peaks_ratios", "diss_curve"],
                         max_cardinality=14)
    assert len(df) == 2


def test_trajectory_can_track_a_source_other_than_the_peak_ratios(harmonic_signal):
    bt = compute_biotuner(SF, peaks_function="fixed", precision=0.5)
    traj = bt.mos_trajectory(
        harmonic_signal, SF, window_sec=5.0, step_sec=5.0,
        source="diss_curve", max_cardinality=12,
    )
    assert len(traj) == 4
    assert any(f is not None for f in traj)


def test_a_source_a_window_cannot_derive_becomes_none(harmonic_signal):
    """Per-window objects never run peaks_extension, so extended_ratios is not
    available -- a gap in the path, not a crash."""
    bt = compute_biotuner(SF, peaks_function="fixed", precision=0.5)
    traj = bt.mos_trajectory(
        harmonic_signal, SF, window_sec=5.0, step_sec=5.0,
        source="extended_ratios", max_cardinality=12,
    )
    assert traj == [None, None, None, None]


def test_the_trajectory_refuses_the_circular_source(harmonic_signal):
    bt = compute_biotuner(SF, peaks_function="fixed", precision=0.5)
    with pytest.raises(ValueError, match="would fit a moment-of-symmetry"):
        bt.mos_trajectory(harmonic_signal, SF, source="mos")


# --------------------------------------------------------------------------- #
# The whole tuning surface, on one signal
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("source", [s for s in TUNING_SOURCES if s != "mos"])
def test_every_derivation_can_drive_a_fit(probe, recurrent, source):
    """Eight ways of turning a signal into ratios, eight well-formed scales.

    ``harm_tuning`` is asked of the harmonic-recurrence object because that is
    the only extraction that measures the harmonic positions it is built from;
    everything else comes off the spectrally parameterised probe.  A source
    that silently produced nothing would fail here, not be excused.
    """
    from biotuner.mos.derive import mos_from_biotuner

    bt = recurrent if source == "harm_tuning" else probe
    ratios = bt.get_tuning(source)
    assert len(ratios) >= 2

    fits = mos_from_biotuner(bt, source=source, max_cardinality=12, top_n=1)
    fit = fits[0]
    assert fit.scale.cardinality >= 4
    assert 0 < fit.n_targets <= len(ratios)
    assert fit.targets and all(1.0 <= t < 2.0 for t in fit.targets)
    assert math.isfinite(fit.error_cents)
    assert math.gcd(fit.scale.n_large, fit.scale.n_small) == 1


def test_harm_tuning_reports_a_missing_input_instead_of_a_typeerror(probe):
    """It used to print a remark, leave ``list_harmonics`` as ``None``, and
    then iterate it: ``TypeError: 'NoneType' object is not iterable``.  A
    derivation that cannot be made must say what is missing."""
    try:
        probe.get_tuning("harm_tuning")
    except TypeError as exc:                                  # the old failure
        pytest.fail(f"harm_tuning still raises TypeError: {exc}")
    except ValueError as exc:
        assert "list_harmonics" in str(exc)
        assert "harmonic_recurrence" in str(exc)
    else:
        pytest.fail("harm_tuning derived a tuning with no harmonic positions")


def test_folding_engages_on_the_real_derivations(probe):
    """Not a synthetic worry: the derivations really do state 1/1 and 2/1.

    ``euler_fokker`` spans unison to octave, so the fit it drives must count
    that pitch class once and report having done so.
    """
    from biotuner.mos.derive import mos_from_biotuner

    ratios = probe.get_tuning("euler_fokker")
    assert min(ratios) == pytest.approx(1.0) and max(ratios) == pytest.approx(2.0)
    fit = mos_from_biotuner(probe, source="euler_fokker", max_cardinality=12,
                            top_n=1)[0]
    assert fit.n_merged >= 1
    assert fit.n_targets == len(ratios) - fit.n_merged
    assert len(fit.residuals) == fit.n_targets
    assert all(1.0 <= t < 2.0 for t in fit.targets)


def test_compare_sources_keeps_a_row_for_everything_it_was_asked(probe):
    """One row per requested source, in the same count, whatever happened to
    it -- a table that dropped its failures would read as if the signal simply
    had fewer derivations than it has."""
    from biotuner.mos.derive import compare_sources

    asked = ["peaks_ratios", "diss_curve", "harm_tuning", "not_a_source", "mos"]
    df = compare_sources(probe, sources=asked, max_cardinality=12)
    assert len(df) == len(asked)
    assert set(df["source"]) == set(asked)

    worked = df[df["reason"].isna()]
    failed = df[df["reason"].notna()]
    assert set(worked["source"]) == {"peaks_ratios", "diss_curve"}
    assert set(failed["source"]) == {"harm_tuning", "not_a_source", "mos"}
    # Each failure says which one it was, in its own words.
    reasons = dict(zip(df["source"], df["reason"]))
    assert "list_harmonics" in reasons["harm_tuning"]
    assert "not_a_source" in reasons["not_a_source"]
    assert "moment-of-symmetry" in reasons["mos"]
    # A failed row carries no fit numbers to be mistaken for results.
    assert failed["signature"].isna().all()
    assert failed["evidence"].isna().all()
    # ... and sorts below every row that produced one.
    assert list(df["reason"].notna()) == [False, False, True, True, True]


def test_the_trajectory_derives_the_source_it_was_given(harmonic_signal, monkeypatch):
    """Two things, because either alone would pass a dropped parameter.

    The spy proves every window really asked for ``diss_curve`` -- forwarding
    the name into a dict nobody reads would not show up in the output -- and
    the comparison proves the answer depends on it, which a spy on its own
    cannot.
    """
    asked = []
    real_get_tuning = compute_biotuner.get_tuning

    def spy(self, source, *args, **kwargs):
        asked.append(source)
        return real_get_tuning(self, source, *args, **kwargs)

    monkeypatch.setattr(compute_biotuner, "get_tuning", spy)

    kwargs = dict(window_sec=5.0, step_sec=5.0, max_cardinality=12)
    bt = compute_biotuner(SF, peaks_function="fixed", precision=0.5)
    peaks_traj = bt.mos_trajectory(harmonic_signal, SF, **kwargs)
    assert asked == ["peaks_ratios"] * 4

    asked.clear()
    bt = compute_biotuner(SF, peaks_function="fixed", precision=0.5)
    diss_traj = bt.mos_trajectory(harmonic_signal, SF, source="diss_curve", **kwargs)
    assert asked == ["diss_curve"] * 4

    assert all(f is not None for f in peaks_traj + diss_traj)
    # The dissonance curve is not the peak ratios, and the path shows it.
    assert [f.signature for f in peaks_traj] != [f.signature for f in diss_traj]


# --------------------------------------------------------------------------- #
# The fitted scale flows through the rest of the toolbox
# --------------------------------------------------------------------------- #
def test_the_fitted_scale_exports_to_scala(fitted):
    scl = fitted.mos_scale.to_scala(write=False)
    assert fitted.mos_scale.signature in scl


def test_existing_tuning_plots_accept_the_mos_source(fitted):
    fitted.plot_tuning_scale(tuning="mos")
    assert plt.gcf() is not None


def test_metrics_run_on_the_fitted_scale(fitted):
    from biotuner.mos.metrics import mos_report

    report = mos_report(fitted.mos_scale)
    assert report["signature"] == fitted.mos_scale.signature
    assert "myhill" in report or "myhill_property" in report


def test_modes_of_the_fitted_scale(fitted):
    modes = fitted.mos_scale.modes()
    assert len(modes) == fitted.mos_scale.cardinality
    brightness = [m.brightness for m in modes]
    assert brightness == sorted(brightness, reverse=True)


# --------------------------------------------------------------------------- #
# The corrected vizs shims
# --------------------------------------------------------------------------- #
def test_vizs_plot_labyrinth_now_draws_real_rings():
    from biotuner import vizs

    fig, ax = vizs.plot_labyrinth([4 / 3, 3 / 2, 9 / 5], max_steps=14)
    assert ax.name == "polar"
    # The old implementation put everything at radius 1 or 2.
    radii = set()
    for line in ax.get_lines():
        radii.update(np.round(line.get_ydata(), 6))
    assert max(radii) >= 14


def test_vizs_shim_honours_a_pseudo_octave():
    from biotuner import vizs

    fig, ax = vizs.plot_labyrinth([3 / 2], max_steps=9, octave=2.05)
    assert ax is not None


def test_package_exports_are_all_importable():
    import biotuner.mos as M

    assert not [n for n in M.__all__ if not hasattr(M, n)]
