"""A misspelt trajectory source must fail loudly, not become an empty path.

``mos_trajectory`` deliberately turns a window it cannot derive the source from
into ``None``, so one bad epoch is a gap rather than a crash.  A name no
derivation answers to fails in *every* window, and without a guard that rule
converts a typo into an all-``None`` trajectory -- which is exactly what a
structureless recording looks like.  These tests pin the distinction.
"""

import numpy as np
import pytest

from biotuner.biotuner_object import TUNING_SOURCES, compute_biotuner
from biotuner.mos.derive import mos_trajectory

SF = 1000.0


@pytest.fixture(scope="module")
def drifting_signal():
    """Peaks on a stack of fifths for the first half, elsewhere for the second."""
    rng = np.random.default_rng(7)
    t = np.arange(0, 16.0, 1 / SF)
    x = np.zeros_like(t)
    half = t.size // 2
    for f, a in [(5.0, 1.0), (7.5, 0.8), (11.25, 0.6), (16.875, 0.4)]:
        x[:half] += a * np.sin(2 * np.pi * f * t[:half])
    for f, a in [(6.0, 1.0), (10.0, 0.7), (14.0, 0.5), (22.0, 0.3)]:
        x[half:] += a * np.sin(2 * np.pi * f * t[half:])
    return x + 0.05 * rng.standard_normal(t.size)


TRAJ_KW = dict(window_sec=4.0, step_sec=4.0, peaks_function="fixed",
               precision=0.1, n_peaks=5, max_cardinality=12)


@pytest.mark.parametrize("bad", ["diss_curvee", "PEAKS_RATIOS", "nonsense",
                                 "", None, 42])
def test_a_misspelt_source_raises_instead_of_emptying_the_path(drifting_signal, bad):
    with pytest.raises(ValueError, match="source must be one of"):
        mos_trajectory(drifting_signal, SF, source=bad, **TRAJ_KW)


def test_the_name_is_checked_before_a_single_window_is_analysed(monkeypatch,
                                                                drifting_signal):
    """No peak extraction happens for a name that cannot work."""
    calls = []
    real = compute_biotuner.peaks_extraction

    def spy(self, *a, **k):
        calls.append(1)
        return real(self, *a, **k)

    monkeypatch.setattr(compute_biotuner, "peaks_extraction", spy)
    with pytest.raises(ValueError):
        mos_trajectory(drifting_signal, SF, source="typo", **TRAJ_KW)
    assert calls == []


@pytest.mark.parametrize("source", [s for s in TUNING_SOURCES if s != "mos"])
def test_every_real_source_is_still_accepted(drifting_signal, source):
    """The guard rejects names, not signals: a source that simply cannot be
    derived in any window still comes back as a path of ``None``."""
    traj = mos_trajectory(drifting_signal, SF, source=source, **TRAJ_KW)
    assert len(traj) == 4
    assert all(f is None or f.scale.cardinality >= 4 for f in traj)


@pytest.mark.parametrize("alias,canonical", [
    ("ratios", "peaks_ratios"),
    ("peaks_ratios_cons", "cons_ratios"),
    ("harmonic_entropy", "HE"),
    ("harmonic_fit", "harm_fit_tuning"),
])
def test_the_aliases_get_tuning_accepts_are_accepted_here_too(drifting_signal,
                                                              alias, canonical):
    a = mos_trajectory(drifting_signal, SF, source=alias, **TRAJ_KW)
    b = mos_trajectory(drifting_signal, SF, source=canonical, **TRAJ_KW)
    assert [None if f is None else f.signature for f in a] == \
           [None if f is None else f.signature for f in b]


def test_the_object_method_inherits_the_guard(drifting_signal):
    bt = compute_biotuner(SF, data=drifting_signal, peaks_function="fixed",
                          precision=0.1)
    with pytest.raises(ValueError, match="source must be one of"):
        bt.mos_trajectory(window_sec=4.0, step_sec=4.0, source="diss_curvee",
                          n_peaks=5, max_cardinality=12)
