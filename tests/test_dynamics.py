"""Tests for biotuner.bioelements.dynamics — Phase 1 (state & coherence)."""
import numpy as np
import pytest

from biotuner.bioelements import dynamics as dyn
from biotuner.bioelements import (
    MaterialState, element_state, state_from_signal, coherence, compositional_level,
    material_state, material_order,
)
from biotuner.bioelements.materials import MATERIALS

SF = 250
T = np.arange(int(8 * SF)) / SF


def _synth(comps, lock, seed=0, noise=0.05):
    rng = np.random.default_rng(seed)
    y = np.zeros_like(T)
    for f, a in comps:
        ph = 0.0 if lock else rng.uniform(0, 2 * np.pi)
        y += a * np.sin(2 * np.pi * f * T + ph)
    return y + noise * rng.standard_normal(len(T))


CRYSTAL = _synth([(4.5, 1), (9, .7), (13.5, .55), (18, .42), (22.5, .32), (27, .25)], lock=True)
FLOAT = _synth([(5.2, 1), (8.7, .9), (13.9, .8), (19.3, .7), (24.6, .6)], lock=False, seed=3)
COMPOUND = _synth([(8, 1), (12, .9)], lock=True)
ELEMENT = _synth([(10, 1)], lock=True)


# --------------------------------------------------------------------------- #
# coherence — the float ↔ crystal order parameter
# --------------------------------------------------------------------------- #
def test_coherence_locked_beats_floating():
    assert coherence(CRYSTAL, SF) > 0.8      # phase-locked → crystallised
    assert coherence(FLOAT, SF) < 0.4        # independent → floating
    assert coherence(CRYSTAL, SF) > coherence(FLOAT, SF)


def test_coherence_in_unit_interval():
    for s in (CRYSTAL, FLOAT, COMPOUND, ELEMENT):
        assert 0.0 <= coherence(s, SF) <= 1.0


def test_coherence_return_parts():
    R, parts = coherence(CRYSTAL, SF, return_parts=True)
    assert 0 <= R <= 1
    assert {"freqs", "amps", "resultants", "ref"} <= set(parts)
    assert len(parts["resultants"]) == len(parts["freqs"])


# --------------------------------------------------------------------------- #
# compositional_level — the order ladder, measured
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sig,expected", [
    (CRYSTAL, "structure"),   # coherent + long-range harmonic
    (FLOAT, "mixture"),       # incoherent co-presence
    (COMPOUND, "compound"),   # coherent, few, not long-range
    (ELEMENT, "element"),     # one line
])
def test_compositional_level_labels(sig, expected):
    assert compositional_level(sig, SF)["level"] == expected


def test_compositional_scores_sum_to_one():
    info = compositional_level(FLOAT, SF)
    assert info["scores"].keys() == set(dyn.LEVELS) if isinstance(info["scores"], dict) else True
    assert sum(info["scores"].values()) == pytest.approx(1.0, abs=1e-6)
    assert 0 <= info["R"] <= 1 and 0 <= info["harmonicity"] <= 1


# --------------------------------------------------------------------------- #
# MaterialState — the wavefunction
# --------------------------------------------------------------------------- #
def test_state_is_normalised_superposition():
    st = state_from_signal(FLOAT, SF)
    assert isinstance(st, MaterialState)
    assert st.probabilities.sum() == pytest.approx(1.0, abs=1e-6)   # Born rule
    assert 0.0 <= st.entropy() <= 1.0
    assert len(st.dominant()) == 1


def test_element_state_from_peaks():
    st = element_state([7.83, 14.3, 20.8, 27.3], R=0.5)
    assert st.probabilities.sum() == pytest.approx(1.0, abs=1e-6)
    assert st.coherence == 0.5
    assert len(st.top(5)) <= 5


# --------------------------------------------------------------------------- #
# material_state — resolving spectral degeneracy (order/level/stoichiometry)
# --------------------------------------------------------------------------- #
# a peak set that spectrally resonates with the degenerate materials (their
# affinity is identical within a class, so any a1>0 set exercises the splitter).
DEGEN_PEAKS = np.array([7.83, 14.3, 20.8, 27.3, 33.8, 41.0])


def test_material_state_is_born_normalised():
    df = material_state(DEGEN_PEAKS, R=0.6)
    assert df["prob"].sum() == pytest.approx(1.0, abs=1e-6)
    assert {"material", "prob", "affinity", "order_consistency", "stoich",
            "order_ref", "kind"} <= set(df.columns)
    assert (df["prob"] >= 0).all()
    # ranked descending
    assert list(df["prob"]) == sorted(df["prob"], reverse=True)


def test_material_order_reads_tag_then_class():
    assert material_order(MATERIALS["Diamond"]) == 0.98        # explicit tag
    assert material_order(MATERIALS["Graphite"]) == 0.72       # explicit tag
    # a material with no order tag falls back to its class default
    galena = MATERIALS["Galena"]
    assert "order" not in galena.tags
    assert material_order(galena) == dyn._CLASS_ORDER[galena.material_class]


def _prob(df, name):
    row = df[df.material == name]
    return float(row["prob"].iloc[0]) if len(row) else 0.0


@pytest.mark.parametrize("crystal,amorphous", [
    ("Diamond", "Graphite"),      # both pure carbon — identical spectrum, order tag splits
    ("WaterIce", "Water"),        # crystalline ice vs liquid — identical H2O spectrum
    ("Quartz", "SilicaGlass"),    # crystal vs glass — identical SiO2 spectrum
])
def test_material_state_order_splits_degeneracy(crystal, amorphous):
    """The higher-order (crystalline) member wins when the signal is coherent;
    the lower-order (amorphous/liquid) member wins when it is not — even though
    the two are spectrally identical."""
    hi = material_state(DEGEN_PEAKS, R=0.90)
    lo = material_state(DEGEN_PEAKS, R=0.35)
    assert _prob(hi, crystal) > _prob(hi, amorphous)      # coherent → crystalline
    assert _prob(lo, amorphous) > _prob(lo, crystal)      # incoherent → amorphous


def test_material_state_stoichiometry_matches_peak_ratio():
    """A compound's atom-ratio echoes in the signal's peak-ratio (compounds only)."""
    octave = material_state(np.array([10.0, 20.0]), R=0.55)   # 2:1
    quad = material_state(np.array([10.0, 40.0]), R=0.55)     # 4:1
    w2 = octave[octave.material == "Water"]["stoich"].iloc[0]     # H2O = 2:1
    m2 = octave[octave.material == "Methane"]["stoich"].iloc[0]   # CH4 = 4:1
    w4 = quad[quad.material == "Water"]["stoich"].iloc[0]
    m4 = quad[quad.material == "Methane"]["stoich"].iloc[0]
    assert w2 > m2 and m4 > w4                                # each wins on its own ratio
    # non-compounds are stoichiometrically neutral
    assert octave[octave.material == "Diamond"]["stoich"].iloc[0] == pytest.approx(1.0)


def test_material_state_tuning_factor_fused():
    """a4 (tuning) is present and blended with a1 into the spectral evidence term."""
    df = material_state(DEGEN_PEAKS, R=0.6)
    assert {"affinity", "tuning", "spectral"} <= set(df.columns)
    row = df.iloc[0]
    assert row["spectral"] == pytest.approx(0.7 * row["affinity"] + 0.3 * row["tuning"], abs=1e-9)


def test_material_state_w_tuning_zero_recovers_position():
    df = material_state(DEGEN_PEAKS, R=0.6, w_tuning=0.0)
    assert np.allclose(df["spectral"].values, df["affinity"].values, atol=1e-9)


def test_tuning_is_degenerate_within_a_class():
    """Spectrally-identical twins share one spectrum → identical tuning a4, so the
    tuning term cannot (and must not) resolve within-class — a2 still does."""
    df = material_state(DEGEN_PEAKS, R=0.6)
    d = df[df.material == "Diamond"]["tuning"].iloc[0]
    g = df[df.material == "Graphite"]["tuning"].iloc[0]
    assert d == pytest.approx(g, abs=1e-9)


def test_material_state_from_signal_path():
    df, parts = material_state(sig=CRYSTAL, sf=SF, return_parts=True)
    assert df["prob"].sum() == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= parts["R"] <= 1.0
    assert parts["level"] in dyn.LEVELS
    assert parts["r_is_proxy"] is False          # real coherence, not a proxy
    assert set(parts["level_scores"]) == set(dyn.LEVELS)


def test_material_state_needs_signal_or_peaks():
    with pytest.raises(ValueError):
        material_state()


# --------------------------------------------------------------------------- #
# visualisation renders
# --------------------------------------------------------------------------- #
def test_material_state_portrait_renders():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, info = dyn.plot_material_state(CRYSTAL, SF, title="test")
    assert fig is not None
    assert set(info) >= {"state", "coherence", "level"}
    assert 0 <= info["coherence"] <= 1
    plt.close("all")
