"""The compute_biotuner side of the MOS integration.

Everything here is about the *object*: how it reports a derivation it cannot
make, how it exposes the source comparison, and what it says about a fit when
asked to describe itself.  The scale mathematics lives in test_derive.py and
the end-to-end plumbing in test_integration.py.
"""

import inspect

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from biotuner.biotuner_object import TUNING_SOURCES, compute_biotuner

SF = 1000.0


def _signal(seconds=20.0):
    """Peaks on a stack of fifths over a 5 Hz fundamental, plus noise."""
    rng = np.random.default_rng(0)
    t = np.arange(0, seconds, 1 / SF)
    base = 5.0
    freqs = [base, base * 1.5, base * 2.25, base * 3.375, base * 2.0]
    x = sum(a * np.sin(2 * np.pi * f * t)
            for a, f in zip([1.0, 0.9, 0.7, 0.5, 0.6], freqs))
    return x + 0.15 * rng.standard_normal(t.size)


@pytest.fixture(scope="module")
def signal():
    return _signal()


@pytest.fixture
def extracted(signal):
    bt = compute_biotuner(SF, peaks_function="fixed", precision=0.1)
    bt.peaks_extraction(signal, min_freq=2, max_freq=40, n_peaks=5)
    return bt


# --------------------------------------------------------------------------- #
# harmonic_tuning: a missing input is an error, not a printed remark
# --------------------------------------------------------------------------- #
def test_harmonic_tuning_without_harmonics_raises_instead_of_printing(extracted, capsys):
    """It used to print 'No list of harmonics provided' and then crash with
    TypeError: 'NoneType' object is not iterable."""
    with pytest.raises(ValueError) as excinfo:
        extracted.harmonic_tuning()
    assert capsys.readouterr().out == ""
    message = str(excinfo.value)
    # Names what to pass, and the two other ways out.
    assert "list_harmonics" in message
    assert "harmonic_recurrence" in message
    assert "harm_fit_tuning" in message
    assert "'fixed'" in message          # the peaks_function actually in use


def test_the_message_is_ascii(extracted):
    with pytest.raises(ValueError) as excinfo:
        extracted.harmonic_tuning()
    str(excinfo.value).encode("ascii")   # raises UnicodeEncodeError if not


def test_get_tuning_surfaces_the_message_rather_than_a_typeerror(extracted):
    with pytest.raises(ValueError, match="harmonic_tuning"):
        extracted.get_tuning("harm_tuning")


def test_compare_mos_sources_reports_it_as_a_valueerror(extracted):
    df = extracted.compare_mos_sources(sources=["harm_tuning"], max_cardinality=12)
    reason = df.iloc[0]["reason"]
    assert reason.startswith("ValueError: harmonic_tuning()")
    assert "list_harmonics" in reason


def test_an_explicit_list_of_harmonics_still_builds_a_tuning(extracted):
    ratios = extracted.harmonic_tuning(list_harmonics=[1, 2, 3, 5, 7])
    assert ratios == pytest.approx([1.25, 1.5, 1.75, 2.0])
    # Cached under the non-conflicting name, and reachable by source name.
    assert extracted.get_tuning("harm_tuning") == pytest.approx(ratios)


def test_an_empty_explicit_list_says_so_specifically(extracted):
    with pytest.raises(ValueError, match="empty list_harmonics"):
        extracted.harmonic_tuning(list_harmonics=[])


def test_an_empty_measurement_blames_the_extraction_not_the_caller(extracted):
    extracted.all_harmonics = []
    with pytest.raises(ValueError, match="all_harmonics empty"):
        extracted.harmonic_tuning()


def test_the_harmonic_recurrence_default_path_is_untouched(signal):
    bt = compute_biotuner(SF, peaks_function="harmonic_recurrence", precision=0.1)
    bt.peaks_extraction(signal, min_freq=2, max_freq=40, n_peaks=5)
    ratios = bt.harmonic_tuning()
    assert len(ratios) > 1
    assert all(1.0 <= r <= 2.0 for r in ratios)
    assert bt.get_tuning("harm_tuning") == pytest.approx(ratios)


def test_harmonic_positions_may_be_an_array(extracted):
    """all_harmonics arrives as a numpy array, so the default path must take one."""
    extracted.all_harmonics = np.array([2.0, 3.0, 5.0])
    assert extracted.harmonic_tuning() == pytest.approx([1.25, 1.5, 2.0])


# --------------------------------------------------------------------------- #
# compare_mos_sources
# --------------------------------------------------------------------------- #
def test_compare_mos_sources_tries_every_derivation_but_the_circular_one(extracted):
    df = extracted.compare_mos_sources(max_cardinality=12)
    assert set(df["source"]) == {s for s in TUNING_SOURCES if s != "mos"}


def test_compare_mos_sources_stores_the_table_and_returns_it(extracted):
    df = extracted.compare_mos_sources(sources=["peaks_ratios", "diss_curve"],
                                       max_cardinality=12)
    assert extracted.mos_sources is df
    assert len(df) == 2


def test_compare_mos_sources_ranks_by_evidence(extracted):
    df = extracted.compare_mos_sources(max_cardinality=12)
    scored = df["evidence"].dropna()
    assert list(scored) == sorted(scored, reverse=True)
    # A source that could not be derived sorts last and keeps its reason.
    assert df["reason"].iloc[-1] is not None


def test_asking_for_the_circular_source_explicitly_is_reported(extracted):
    df = extracted.compare_mos_sources(sources=["mos"], max_cardinality=12)
    assert "moment-of-symmetry" in df.iloc[0]["reason"]


# --------------------------------------------------------------------------- #
# mos_trajectory takes a source
# --------------------------------------------------------------------------- #
def test_the_trajectory_source_is_an_explicit_parameter():
    sig = inspect.signature(compute_biotuner.mos_trajectory)
    assert sig.parameters["source"].default == "peaks_ratios"


def test_the_trajectory_source_reaches_the_derivation(signal, monkeypatch):
    import biotuner.mos.derive as derive

    seen = {}
    real = derive.mos_trajectory

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(derive, "mos_trajectory", spy)
    bt = compute_biotuner(SF, peaks_function="fixed", precision=0.5)
    bt.mos_trajectory(signal, SF, window_sec=10.0, step_sec=10.0,
                      source="diss_curve", max_cardinality=10)
    assert seen["source"] == "diss_curve"
    assert bt.mos_traj is not None


# --------------------------------------------------------------------------- #
# Discoverability: info() and fit_all()
# --------------------------------------------------------------------------- #
def test_info_reports_the_fit(extracted, capsys):
    extracted.fit_mos(max_cardinality=12)
    extracted.info()
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[-2] == "MOS"
    assert extracted.mos_fit.signature in lines[-1]
    assert "err" in lines[-1] and "evidence" in lines[-1]
    out.encode("ascii")


def test_info_says_a_fit_is_missing_rather_than_staying_silent(extracted, capsys):
    extracted.info()
    lines = capsys.readouterr().out.splitlines()
    assert lines[-2] == "MOS"
    assert "fit_mos()" in lines[-1]


def test_info_flags_an_underdetermined_fit(extracted, capsys):
    """A scale with more degrees than the data has targets is not evidence, and
    the summary line must not read like it is."""
    fit = extracted.fit_mos(source="cons_ratios", max_cardinality=12)
    assert fit.is_underdetermined      # a handful of ratios, a whole scale
    extracted.info()
    assert "UNDERDETERMINED" in capsys.readouterr().out.splitlines()[-1]


def test_info_does_not_flag_a_determined_fit(extracted, capsys):
    from biotuner.mos.derive import best_mos
    from biotuner.mos.scale import MOSScale

    extracted.mos_fit = best_mos(
        MOSScale.from_signature(5, 2, tuning=12).ratios, max_cardinality=12
    )
    assert not extracted.mos_fit.is_underdetermined
    extracted.info()
    assert "UNDERDETERMINED" not in capsys.readouterr().out.splitlines()[-1]


def test_fit_all_leaves_the_mos_search_out_by_default():
    bt = compute_biotuner(SF, peaks_function="fixed", precision=0.5)
    fitted = bt.fit_all(_signal(seconds=5.0), compute_diss=False, compute_HE=False,
                        compute_peaks_extension=False)
    assert not hasattr(fitted, "mos_fit")


def test_fit_all_computes_the_mos_when_asked():
    bt = compute_biotuner(SF, peaks_function="fixed", precision=0.5)
    fitted = bt.fit_all(_signal(seconds=5.0), compute_diss=False, compute_HE=False,
                        compute_peaks_extension=False, compute_mos=True)
    assert fitted.mos_fit is fitted.mos_fits[0]
    assert fitted.mos_scale is fitted.mos_fit.scale


# --------------------------------------------------------------------------- #
# Nothing else on the tuning surface moved
# --------------------------------------------------------------------------- #
def test_every_tuning_source_is_documented_in_get_tuning():
    doc = compute_biotuner.get_tuning.__doc__
    for source in TUNING_SOURCES:
        assert "``'{}'``".format(source) in doc


def test_the_labyrinth_plot_survives_the_folding_change(extracted):
    fig, ax = extracted.plot_labyrinth(12)
    assert ax.name == "polar"
    matplotlib.pyplot.close(fig)


def test_get_tuning_mos_still_returns_the_fitted_scale(extracted):
    tuning = extracted.get_tuning("mos")
    assert len(tuning) == extracted.mos_scale.cardinality
    assert np.allclose(tuning, extracted.mos_fit.aligned_ratios)
