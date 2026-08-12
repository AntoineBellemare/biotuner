"""Tests for the ipywidgets explorers added to :mod:`biotuner.mos.interactive`.

A widget test cannot see a figure, so the split here is deliberate: the pure
computations behind each explorer (the simplex clamp, the fit-target
reconstruction, the dissonance curve, the generator sweep and its verdict, the
class-overlap detection) are tested directly on their own numbers, and the
widgets are tested for the things that actually break at construction time --
missing controls, an initial state that is not a real scale, edge inputs that
should raise a readable error instead of a traceback from three layers down,
and observers that throw when a slider moves.

Every explorer is built headlessly.  ipywidgets constructs perfectly well
without a kernel; only the display side needs one, and ``Output`` captures the
figures either way.
"""

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from biotuner.mos import interactive as I
from biotuner.mos import metrics as MT
from biotuner.mos import theory as T
from biotuner.mos.derive import trajectory_from_windows
from biotuner.mos.scale import MOSScale
from biotuner.mos.ternary import TernaryScale

ipywidgets = pytest.importorskip("ipywidgets")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _controls(ui):
    """The control VBox of an explorer, which is always its first child."""
    return ui.children[0].children


# --------------------------------------------------------------------------- #
# Exports
# --------------------------------------------------------------------------- #
def test_the_four_are_exported():
    for name in ("simplex_explorer", "trajectory_explorer",
                 "dissonance_explorer", "matrix_explorer"):
        assert name in I.__all__
        assert callable(getattr(I, name))


def test_the_four_reach_the_package_namespace():
    import biotuner.mos as M

    for name in ("simplex_explorer", "trajectory_explorer",
                 "dissonance_explorer", "matrix_explorer"):
        assert name in M.__all__
        assert getattr(M, name) is getattr(I, name)


# --------------------------------------------------------------------------- #
# The simplex clamp -- pure
# --------------------------------------------------------------------------- #
def test_clamp_leaves_interior_points_alone():
    u, v, w, clamped = I._clamp_simplex(0.4, 0.35)
    assert (u, v) == (0.4, 0.35)
    assert w == pytest.approx(0.25)
    assert clamped is False


@pytest.mark.parametrize(
    "pair",
    [(0.0, 0.0), (1.0, 1.0), (0.999, 0.999), (1.0, 0.0), (0.0, 1.0),
     (-5.0, 0.5), (0.5, 1e9), (0.6, 0.6), (0.5, 0.5)],
)
def test_clamp_always_lands_inside_the_open_simplex(pair):
    """Whatever the pair, the three coordinates come back strictly positive.

    This is the property TernaryScale needs and the only one worth asserting:
    a non-positive step is an edge of the simplex, where the scale is binary.
    """
    u, v, w, _ = I._clamp_simplex(*pair)
    assert u > 0.0 and v > 0.0 and w > 0.0
    assert u + v < 1.0
    assert u + v + w == pytest.approx(1.0, abs=1e-12)


def test_clamp_output_always_builds_a_ternary_scale():
    for u0 in np.linspace(-0.2, 1.2, 15):
        for v0 in np.linspace(-0.2, 1.2, 15):
            u, v, w, _ = I._clamp_simplex(float(u0), float(v0))
            scale = TernaryScale.from_barycentric("LMLsLMs", u, v, w)
            assert min(scale.step_cents) > 0.0


def test_clamp_reports_when_it_moved_the_point():
    assert I._clamp_simplex(0.3, 0.3)[3] is False
    assert I._clamp_simplex(0.8, 0.8)[3] is True


def test_clamp_preserves_the_direction_it_was_dragged():
    eps = I._SIMPLEX_EPS
    u, v, _, _ = I._clamp_simplex(0.9, 0.45)
    assert (u - eps) / (v - eps) == pytest.approx((0.9 - eps) / (0.45 - eps))


def test_clamp_rejects_a_nonsense_epsilon():
    with pytest.raises(ValueError, match=r"eps must lie in \(0, 1/3\)"):
        I._clamp_simplex(0.3, 0.3, eps=0.5)


def test_simplex_word_list_keeps_the_requested_word():
    """ternary_words only returns MV3 arrangements; 'LLMsLMs' is MV4."""
    assert "LLMsLMs" not in __import__(
        "biotuner.mos.ternary", fromlist=["_"]
    ).ternary_words(3, 2, 2)
    options = I._simplex_words("LLMsLMs")
    assert options[0] == "LLMsLMs"
    assert "LMLsLMs" in options


# --------------------------------------------------------------------------- #
# simplex_explorer
# --------------------------------------------------------------------------- #
def test_simplex_explorer_builds_with_the_documented_controls():
    ui = I.simplex_explorer()
    assert isinstance(ui, ipywidgets.VBox)
    assert len(ui.children) == 3
    controls = _controls(ui)
    kinds = [type(c).__name__ for c in controls]
    assert kinds == ["Dropdown", "FloatSlider", "FloatSlider", "ToggleButtons",
                     "IntSlider", "Button"]
    assert [o[1] for o in controls[3].options] == [
        "propriety", "variety", "ji_error", None
    ]


def test_simplex_explorer_opens_on_the_equal_step_tuning():
    ui = I.simplex_explorer("LMLsLMs")
    _, u, v, _, _, _ = _controls(ui)
    expected = TernaryScale.equal_step("LMLsLMs").barycentric
    assert u.value == pytest.approx(expected[0])
    assert v.value == pytest.approx(expected[1])


def test_simplex_explorer_sliders_cannot_leave_the_simplex():
    """Both sliders to their maximum is the case that used to raise."""
    ui = I.simplex_explorer("LMLsLMs")
    _, u, v, _, _, _ = _controls(ui)
    u.value = u.max
    v.value = v.max
    assert u.value + v.value < 1.0
    assert 1.0 - u.value - v.value > 0.0
    # And the widget kept a real scale, not a half-built one.
    scale = TernaryScale.from_barycentric(
        "LMLsLMs", u.value, v.value, 1.0 - u.value - v.value
    )
    assert scale.cardinality == 7


def test_simplex_explorer_survives_every_field_and_a_word_change():
    ui = I.simplex_explorer("LMLsLMs")
    pick, _, _, field, res, _ = _controls(ui)
    for value in ("variety", "ji_error", None, "propriety"):
        field.value = value
    res.value = 20
    pick.value = "LMLsMLs"


def test_simplex_explorer_rejects_an_unknown_field():
    with pytest.raises(ValueError, match="field must be"):
        I.simplex_explorer("LMLsLMs", field="harmonicity")


def test_simplex_explorer_rejects_a_binary_word():
    with pytest.raises(ValueError):
        I.simplex_explorer("LLLsLLs")


# --------------------------------------------------------------------------- #
# Trajectory helpers -- pure
# --------------------------------------------------------------------------- #
def _demo_trajectory():
    a = MOSScale.from_signature(5, 2, tuning=12).ratios
    b = MOSScale.from_signature(4, 3, tuning=19).ratios
    return trajectory_from_windows([a, [], b, a], max_cardinality=12)


def test_fit_targets_round_trip_exactly():
    a = MOSScale.from_signature(5, 2, tuning=12).ratios
    fit = trajectory_from_windows([a], max_cardinality=12)[0]
    back = I._fit_targets(fit)
    assert sorted(back) == pytest.approx(sorted(a), abs=1e-9)


def test_fit_targets_round_trip_on_an_offset_fit():
    """A fit that had to transpose the scale must still invert."""
    ratios = [1.0, 1.07, 1.19, 1.34, 1.51, 1.7]
    fit = trajectory_from_windows([ratios], max_cardinality=12)[0]
    back = I._fit_targets(fit)
    period = fit.scale.period
    folded = sorted(period ** (math.log(r, period) % 1.0) for r in ratios)
    assert sorted(back) == pytest.approx(folded, abs=1e-9)


def test_draw_path_breaks_at_a_failed_window():
    traj = _demo_trajectory()
    assert traj[1] is None
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    n = I._draw_path(ax, traj, len(traj) - 1)
    assert n == 3
    # Two runs -- [w0] and [w2, w3] -- so one joining line, not two, and one
    # marker call per run.
    lines = ax.get_lines()
    joins = [ln for ln in lines if ln.get_linestyle() == "-"
             and ln.get_marker() == "None"]
    assert len(joins) == 1
    assert len(joins[0].get_xdata()) == 2


def test_draw_path_stops_at_the_requested_window():
    traj = _demo_trajectory()
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    assert I._draw_path(ax, traj, 0) == 1
    assert I._draw_path(ax, traj, 1) == 1  # window 1 failed, nothing is added
    assert I._draw_path(ax, traj, 2) == 2


# --------------------------------------------------------------------------- #
# trajectory_explorer
# --------------------------------------------------------------------------- #
def test_trajectory_explorer_builds_with_the_documented_controls():
    ui = I.trajectory_explorer(_demo_trajectory())
    assert len(ui.children) == 3
    window, path, residuals = _controls(ui)
    assert isinstance(window, ipywidgets.IntSlider)
    assert isinstance(path, ipywidgets.Checkbox)
    assert isinstance(residuals, ipywidgets.Checkbox)
    assert (window.min, window.max) == (0, 3)


def test_trajectory_explorer_opens_on_a_window_that_was_fitted():
    """Opening on a None window would show an empty labyrinth for no reason."""
    traj = _demo_trajectory()
    ui = I.trajectory_explorer([None] + list(traj))
    window = _controls(ui)[0]
    assert window.value == 1


def test_trajectory_explorer_scrubs_over_a_failed_window():
    traj = _demo_trajectory()
    ui = I.trajectory_explorer(traj)
    window, path, residuals = _controls(ui)
    for i in range(len(traj)):
        window.value = i          # window 1 is None and must not raise
    path.value = False
    residuals.value = False


def test_trajectory_explorer_rejects_an_empty_trajectory():
    with pytest.raises(ValueError, match="trajectory is empty"):
        I.trajectory_explorer([])


def test_trajectory_explorer_rejects_an_all_none_trajectory():
    with pytest.raises(ValueError, match="could be\n?\\s*fitted|could be fitted"):
        I.trajectory_explorer([None, None, None])


def test_trajectory_explorer_rejects_mismatched_times():
    traj = _demo_trajectory()
    with pytest.raises(ValueError, match="times has 2 entries"):
        I.trajectory_explorer(traj, times=[0.0, 1.0])


def test_trajectory_explorer_accepts_times():
    traj = _demo_trajectory()
    ui = I.trajectory_explorer(traj, times=[0.0, 0.5, 1.0, 1.5])
    assert _controls(ui)[0].max == 3


# --------------------------------------------------------------------------- #
# Dissonance -- pure
# --------------------------------------------------------------------------- #
def test_harmonic_spectrum_matches_the_matched_one_in_shape():
    from biotuner.mos.timbre import matched_spectrum

    scale = MOSScale.from_signature(5, 2, tuning=31)
    f_m, a_m = matched_spectrum(scale, 250.0, 8)
    f_h, a_h = I._harmonic_spectrum(250.0, 8)
    assert f_h.shape == f_m.shape
    assert a_h == pytest.approx(a_m)      # same envelope, so only placement differs
    assert f_h[0] == pytest.approx(f_m[0])
    assert not np.allclose(f_h, f_m)


def test_harmonic_spectrum_rejects_bad_arguments():
    with pytest.raises(ValueError, match="n_partials must be >= 1"):
        I._harmonic_spectrum(250.0, 0)
    with pytest.raises(ValueError, match="positive frequency"):
        I._harmonic_spectrum(0.0, 4)


def test_dissonance_curve_has_the_textbook_shape():
    scale = MOSScale.from_signature(5, 2, tuning=12)
    x, y = I._dissonance_curve(scale, matched=False, n_partials=6,
                               resolution=241)
    assert x[0] == 0.0 and x[-1] == pytest.approx(1200.0)
    assert y.shape == x.shape
    # Peak roughness within a semitone of the unison, near-silence at the octave.
    assert x[int(y.argmax())] < 120.0
    assert y[-1] < 0.05 * y.max()
    # The fifth is a local minimum: 700 c is index 140 on a 241-point grid.
    k = int(round(700.0 / 1200.0 * 240))
    assert y[k] < y[k - 8] and y[k] < y[k + 8]


def test_dissonance_curve_minima_move_onto_the_degrees_when_matched():
    """The whole Dynamic Tonality claim, on one deliberately odd tuning.

    At 714 cents the fifth is far from just, so the harmonic timbre's minimum
    stays near 702 c while the matched one follows the scale degree.
    """
    scale = MOSScale.from_signature(5, 2, tuning=0.5952)
    degree = scale.cents[4]
    x, y_h = I._dissonance_curve(scale, matched=False, n_partials=8,
                                 resolution=481)
    _, y_m = I._dissonance_curve(scale, matched=True, n_partials=8,
                                 resolution=481)
    window = (x > degree - 40.0) & (x < degree + 40.0)
    assert abs(x[window][y_m[window].argmin()] - degree) < abs(
        x[window][y_h[window].argmin()] - degree
    )


def test_dissonance_curve_rejects_a_degenerate_resolution():
    scale = MOSScale.from_signature(5, 2, tuning=12)
    with pytest.raises(ValueError, match="resolution must be at least 2"):
        I._dissonance_curve(scale, resolution=1)


def test_generator_sweep_stays_inside_the_valid_range():
    lo, hi = T.signature_ranges(5, 2)[1]
    g, h, m = I._generator_sweep(5, 2, n_partials=6, resolution=11)
    assert g.shape == h.shape == m.shape == (11,)
    assert g.min() > float(lo) * 1200.0
    assert g.max() < float(hi) * 1200.0
    assert np.all(np.diff(g) > 0)
    assert np.all(h > 0) and np.all(m > 0)


def test_generator_sweep_reproduces_scale_dissonance_pointwise():
    from biotuner.mos.timbre import scale_dissonance

    g, h, m = I._generator_sweep(5, 2, n_partials=6, resolution=5)
    for k in range(len(g)):
        scale = MOSScale(5, 2, float(g[k]) / 1200.0, 2.0, validate=False)
        assert h[k] == pytest.approx(
            scale_dissonance(scale, n_partials=6, matched=False)
        )
        assert m[k] == pytest.approx(
            scale_dissonance(scale, n_partials=6, matched=True)
        )


def test_generator_sweep_rejects_a_degenerate_resolution():
    with pytest.raises(ValueError, match="resolution must be at least 2"):
        I._generator_sweep(5, 2, resolution=1)


def test_sweep_verdict_counts_wins_and_finds_the_loss_bands():
    g = np.arange(6.0)
    h = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    m = np.array([9.0, 11.0, 12.0, 8.0, 9.5, 10.0])
    v = I._sweep_verdict(g, h, m)
    assert v["n"] == 6
    assert v["matched_wins"] == 3          # indices 0, 3, 4
    assert v["win_fraction"] == pytest.approx(0.5)
    # Index 5 is a tie, so it belongs in neither the wins nor the loss bands.
    assert v["ties"] == 1
    assert v["loss_ranges"] == [(1.0, 2.0)]
    assert v["worst_loss"] == pytest.approx(-2.0)
    assert v["worst_loss_cents"] == pytest.approx(2.0)


def test_sweep_verdict_reports_a_clean_sweep_as_such():
    v = I._sweep_verdict(np.array([1.0, 2.0]), np.array([5.0, 5.0]),
                         np.array([4.0, 3.0]))
    assert v["win_fraction"] == 1.0
    assert v["loss_ranges"] == []
    assert v["worst_loss"] == 0.0
    assert math.isnan(v["worst_loss_cents"])


def test_two_partials_make_the_timbres_identical_and_the_verdict_says_so():
    """The state at the partials slider's floor, which used to read as a rout.

    The lattice cannot move the fundamental, and with an octave period it
    cannot move the second partial either, so a two-partial matched spectrum
    *is* the harmonic series -- bit-identical, at every generator.  Counted as
    losses those ties made the widget print "harmonic lower over 686.1-719.6 c,
    worst loss 0.000", false in both halves.  Ties are ties.
    """
    g, h, m = I._generator_sweep(5, 2, n_partials=2, resolution=21)
    assert np.array_equal(h, m), "two-partial spectra should coincide exactly"
    v = I._sweep_verdict(g, h, m)
    assert v["matched_wins"] == 0
    assert v["ties"] == v["n"] == 21
    assert v["loss_ranges"] == []
    assert v["worst_loss"] == 0.0
    assert math.isnan(v["worst_loss_cents"])


def test_the_explorer_reports_the_tie_rather_than_a_phantom_loss(capsys):
    """Driving the partials slider to its floor must not print a false claim."""
    ui = I.dissonance_explorer(5, 2, n_partials=8)
    gen, partials, matched, res = _controls(ui)
    res.value = 21
    capsys.readouterr()
    partials.value = partials.min
    printed = capsys.readouterr().out
    assert "identical at every generator" in printed
    assert "harmonic lower over" not in printed
    assert "worst loss" not in printed


def test_sweep_verdict_rejects_ragged_input():
    with pytest.raises(ValueError, match="same shape"):
        I._sweep_verdict(np.zeros(3), np.zeros(3), np.zeros(2))
    with pytest.raises(ValueError, match="empty sweep"):
        I._sweep_verdict(np.zeros(0), np.zeros(0), np.zeros(0))


def test_matched_timbre_wins_most_of_the_range_but_not_all_of_it():
    """The honest version of the Dynamic Tonality claim, measured.

    Milne et al. section 6 argue that matching partials to the tuning lowers
    roughness.  Swept across the whole 5L2s range it does -- for most of it.
    It also loses in narrow bands, chiefly where the harmonic timbre is already
    the matched one because the tuning is near just, and the test pins both
    halves of that so a future change cannot quietly turn it into a clean sweep.
    """
    g, h, m = I._generator_sweep(5, 2, n_partials=8, resolution=81)
    v = I._sweep_verdict(g, h, m)
    assert 0.7 < v["win_fraction"] < 0.95
    assert v["median_reduction_pct"] > 1.0
    assert v["loss_ranges"], "the matched timbre is expected to lose somewhere"
    # Every loss is small next to the wins it is averaged against.
    assert abs(v["worst_loss"]) < 0.2 * float(np.max(h - m))


# --------------------------------------------------------------------------- #
# dissonance_explorer
# --------------------------------------------------------------------------- #
def test_dissonance_explorer_builds_with_the_documented_controls():
    ui = I.dissonance_explorer()
    assert len(ui.children) == 3
    gen, partials, matched, res = _controls(ui)
    assert isinstance(gen, ipywidgets.FloatSlider)
    assert isinstance(partials, ipywidgets.IntSlider)
    assert isinstance(matched, ipywidgets.Checkbox)
    assert isinstance(res, ipywidgets.IntSlider)


def test_dissonance_explorer_bounds_the_generator_to_the_signature():
    lo, hi = T.signature_ranges(5, 2)[1]
    gen = _controls(I.dissonance_explorer())[0]
    assert float(lo) * 1200.0 < gen.min < gen.value < gen.max < float(hi) * 1200.0
    # It opens inside the coherent sub-range, so the initial scale is proper.
    scale = MOSScale(5, 2, gen.value / 1200.0, 2.0, validate=False)
    assert scale.is_proper


def test_dissonance_explorer_responds_to_every_control():
    ui = I.dissonance_explorer(n_partials=6)
    gen, partials, matched, res = _controls(ui)
    gen.value = gen.min
    gen.value = gen.max
    matched.value = False
    partials.value = 4
    res.value = res.min


def test_dissonance_explorer_rejects_a_non_coprime_signature():
    with pytest.raises(ValueError, match="co-prime"):
        I.dissonance_explorer(4, 2)


def test_dissonance_explorer_rejects_an_empty_step_class():
    with pytest.raises(ValueError, match="both step counts must be >= 1"):
        I.dissonance_explorer(5, 0)


@pytest.mark.parametrize("n_partials", [1, 0, -3, 17, 100])
def test_dissonance_explorer_refuses_a_partial_count_it_cannot_show(n_partials):
    """ipywidgets clips a bounded value silently; the explorer must not.

    Left to the slider, ``n_partials=1`` opened on 2 and ``n_partials=100`` on
    16, with nothing said either way, so the timbre on screen was not the one
    asked for.
    """
    with pytest.raises(ValueError, match="partials slider's range"):
        I.dissonance_explorer(5, 2, n_partials=n_partials)


def test_dissonance_explorer_accepts_both_ends_of_the_partials_range():
    lo, hi = I._PARTIALS_RANGE
    for n in (lo, hi):
        assert _controls(I.dissonance_explorer(5, 2, n_partials=n))[1].value == n


# --------------------------------------------------------------------------- #
# Overlap detection -- pure
# --------------------------------------------------------------------------- #
def test_class_overlaps_is_empty_exactly_when_the_scale_is_proper():
    for tuning in (12, 19, 31, 43, "central", "noble"):
        scale = MOSScale.from_signature(5, 2, tuning=tuning)
        assert bool(I._class_overlaps(scale)) == (not MT.is_proper(scale))


def test_class_overlaps_finds_the_pythagorean_breach():
    scale = MOSScale.from_generator(3 / 2, 7)
    overlaps = I._class_overlaps(scale)
    assert [k for k, _ in overlaps] == [3]
    large, small = scale.step_cents
    assert overlaps[0][1] == pytest.approx(large - 2 * small)


def test_every_overlap_is_the_same_size_and_equals_L_minus_2s():
    """Milne et al. section 2, made arithmetic.

    ``max(class k) <= min(class k+1)`` reduces to ``L <= 2s`` for every
    constrained class, so all of them open together and by the same amount.
    """
    for n_large, n_small in ((4, 3), (3, 4), (2, 5), (7, 5)):
        # The improper part of the range sits at whichever end is further from
        # the equalized tuning, and that is not always the upper one.
        lo, hi = T.signature_ranges(n_large, n_small)[1]
        candidates = [float(lo) + f * (float(hi) - float(lo))
                      for f in (0.08, 0.92)]
        scale = max(
            (MOSScale(n_large, n_small, g, 2.0, validate=False)
             for g in candidates),
            key=lambda s: s.hardness,
        )
        assert scale.hardness > 3.0
        overlaps = I._class_overlaps(scale)
        assert len(overlaps) == n_small - 1
        large, small = scale.step_cents
        for _, amount in overlaps:
            assert amount == pytest.approx(large - 2 * small, rel=1e-9)


def test_class_overlaps_accepts_a_raw_scale():
    """Anything metrics accepts -- a ternary scale has no MOS object to pass."""
    scale = MOSScale.from_generator(3 / 2, 7)
    raw = (scale.cents, scale.period_cents)
    assert I._class_overlaps(raw) == I._class_overlaps(scale)

    ternary = TernaryScale.from_barycentric("LLLMMss", 0.52, 0.3, 0.18)
    overlaps = I._class_overlaps((ternary.cents, ternary.period_cents))
    assert bool(overlaps) == (not ternary.is_proper)


def test_overlap_onsets_land_on_the_embedding_edo():
    """Coherence ends where the embedding scale is equally tuned (section 2-3)."""
    _, embedding = T.embedding(5, 2, bright=True)
    onsets = I._overlap_onsets(5, 2)
    assert set(onsets) == {3}
    assert onsets[3] == pytest.approx(float(embedding) * 1200.0, abs=1e-6)


def test_overlap_onsets_tie_across_every_constrained_pair():
    onsets = I._overlap_onsets(4, 3)
    assert set(onsets) == {2, 4}
    assert onsets[2] == pytest.approx(onsets[4], abs=1e-6)
    _, embedding = T.embedding(4, 3, bright=True)
    assert onsets[2] == pytest.approx(float(embedding) * 1200.0, abs=1e-6)


def test_overlap_onsets_are_empty_for_a_single_small_step():
    """With n_small == 1 no class pair is constrained, so none can ever open."""
    assert I._overlap_onsets(2, 1) == {}
    assert I._overlap_onsets(3, 1) == {}


def test_overlap_onsets_agree_with_measured_propriety_either_side():
    onsets = I._overlap_onsets(5, 2)
    boundary = onsets[3] / 1200.0
    assert MT.is_proper(MOSScale(5, 2, boundary - 1e-4, 2.0, validate=False))
    assert not MT.is_proper(MOSScale(5, 2, boundary + 1e-4, 2.0, validate=False))


def test_the_two_propriety_verdicts_agree_except_at_one_small_step():
    """Cross-check of the shortcut against the measurement, as the widget prints.

    They part company only when ``n_small == 1``, which is documented in
    :func:`biotuner.mos.metrics.is_proper` and is why matrix_explorer flags a
    disagreement rather than asserting one cannot happen.
    """
    disagreed = set()
    for n_large in range(1, 8):
        for n_small in range(1, 8):
            if math.gcd(n_large, n_small) != 1 or n_large + n_small < 3:
                continue
            lo, hi = T.signature_ranges(n_large, n_small)[1]
            for frac in (0.1, 0.3, 0.5, 0.7, 0.9):
                g = float(lo) + frac * (float(hi) - float(lo))
                scale = MOSScale(n_large, n_small, g, 2.0, validate=False)
                if scale.is_proper != bool(MT.is_proper(scale)):
                    disagreed.add((n_large, n_small))
    assert all(ns == 1 for _, ns in disagreed), disagreed
    assert disagreed, "the n_small == 1 divergence should still be reachable"


# --------------------------------------------------------------------------- #
# matrix_explorer
# --------------------------------------------------------------------------- #
def test_matrix_explorer_builds_with_the_documented_controls():
    ui = I.matrix_explorer()
    assert len(ui.children) == 3
    gen, mark = _controls(ui)
    assert isinstance(gen, ipywidgets.FloatSlider)
    assert isinstance(mark, ipywidgets.Checkbox)
    assert mark.value is True


def test_matrix_explorer_slider_covers_the_improper_part_of_the_range():
    """The point of the widget: the boundary has to be crossable."""
    gen = _controls(I.matrix_explorer())[0]
    onset = I._overlap_onsets(5, 2)[3]
    assert gen.min < onset < gen.max
    hardest = MOSScale(5, 2, gen.max / 1200.0, 2.0, validate=False)
    assert hardest.hardness > 2.0
    softest = MOSScale(5, 2, gen.min / 1200.0, 2.0, validate=False)
    assert softest.hardness < 2.0


def test_matrix_explorer_sweeps_without_raising():
    ui = I.matrix_explorer()
    gen, mark = _controls(ui)
    for value in np.linspace(gen.min, gen.max, 9):
        gen.value = float(value)
    mark.value = False


def test_matrix_explorer_shades_exactly_the_overlapping_interval():
    """The red band is the claim, so its coordinates are worth pinning.

    Nothing else in this file looks at the patch, so anchoring the rectangle on
    ``min(sizes[k])`` instead of ``min(sizes[k + 1])`` -- drawing the band in
    the wrong place, at the wrong width -- was invisible to the suite.
    """
    from matplotlib.patches import Rectangle

    ui = I.matrix_explorer(5, 2)
    gen = _controls(ui)[0]
    gen.value = gen.max                                  # well past R = 2
    scale = MOSScale(5, 2, gen.value / 1200.0, 2.0, validate=False)
    overlaps = dict(I._class_overlaps(scale))
    assert overlaps, "the far end of the range must be improper"

    right = [ax for ax in plt.gcf().axes
             if ax.get_ylabel().startswith("generic interval class")]
    assert len(right) == 1
    bands = [p for p in right[0].patches if isinstance(p, Rectangle)
             and p.get_width() > 0]
    assert len(bands) == len(overlaps)

    sizes = MT.generic_interval_sizes(scale)
    want = sorted((min(sizes[k + 1]), max(sizes[k])) for k in overlaps)
    got = sorted((p.get_x(), p.get_x() + p.get_width()) for p in bands)
    for (a0, a1), (b0, b1) in zip(want, got):
        assert b0 == pytest.approx(a0)
        assert b1 == pytest.approx(a1)


def test_matrix_explorer_handles_the_propriety_shortcuts_blind_spot():
    ui = I.matrix_explorer(2, 1)
    gen = _controls(ui)[0]
    gen.value = gen.min                 # R > 2, yet measurably proper
    scale = MOSScale(2, 1, gen.value / 1200.0, 2.0, validate=False)
    assert scale.hardness > 2.0
    assert scale.is_proper is False
    assert MT.is_proper(scale) is True


def test_matrix_explorer_rejects_a_non_coprime_signature():
    with pytest.raises(ValueError, match="co-prime"):
        I.matrix_explorer(6, 3)


def test_matrix_explorer_rejects_a_scale_with_one_interval_class():
    with pytest.raises(ValueError, match="no adjacent pair to overlap"):
        I.matrix_explorer(1, 1)


def test_matrix_explorer_accepts_a_pseudo_octave():
    ui = I.matrix_explorer(5, 2, period=3.0)
    gen = _controls(ui)[0]
    lo, hi = T.signature_ranges(5, 2)[1]
    tritave = 1200.0 * math.log2(3.0)
    assert gen.min > float(lo) * tritave
    assert gen.max < float(hi) * tritave


def test_hiding_the_matched_curve_also_hides_its_numbers(capsys):
    """The text panel must not report what the picture is not showing.

    Every line below the timbre summary compares the two timbres, so leaving
    them up while the matched curve is hidden has the figure and the readout
    disagree about what is on screen.
    """
    ui = I.dissonance_explorer(5, 2, n_partials=8)
    gen, partials, matched, res = _controls(ui)
    res.value = 21

    # Toggle off first: assigning a Checkbox its current value fires no change
    # event, so the observer never runs and nothing is printed.
    capsys.readouterr()
    matched.value = False
    hidden = capsys.readouterr().out

    capsys.readouterr()
    matched.value = True
    shown = capsys.readouterr().out
    assert "matched" in shown
    assert "reduction" in shown
    assert "median reduction" in shown

    assert "harmonic" in hidden, "the harmonic total is still on screen"
    assert "reduction" not in hidden
    assert "median reduction" not in hidden
    assert "matched timbre hidden" in hidden


def test_the_harmonic_total_is_the_same_either_way(capsys):
    """Hiding the comparison must not change the number that remains."""
    ui = I.dissonance_explorer(5, 2, n_partials=6)
    gen, partials, matched, res = _controls(ui)
    res.value = 21

    def harmonic_line():
        out = capsys.readouterr().out
        return [l for l in out.splitlines() if "this tuning" in l][-1]

    capsys.readouterr()
    matched.value = False
    off = harmonic_line()
    matched.value = True
    on = harmonic_line()
    value = lambda line: line.split("harmonic")[1].split(",")[0].strip()
    assert value(off) == value(on)
