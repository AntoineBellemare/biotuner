"""Tests for the biotuner.bioelements package (units, tables, spectrum,
composition, materials, matching, affinity, bridges) + legacy back-compat."""
import numpy as np
import pandas as pd
import pytest

from biotuner import bioelements as be
from biotuner.bioelements import (
    Composition, Spectrum, element_spectrum, material, MATERIALS,
    coverage_report, match_elements, match_materials, material_tuning,
    load_elements, ELEMENT_CATEGORIES,
)


# --------------------------------------------------------------------------- #
# units
# --------------------------------------------------------------------------- #
def test_unit_roundtrips():
    assert be.hertz_to_angstrom(be.angstrom_to_hertz(5000)) == pytest.approx(5000, rel=1e-9)
    assert be.nm_to_hertz(500) == pytest.approx(be.angstrom_to_hertz(5000), rel=1e-9)


def test_fold_to_optical_in_band():
    w = be.fold_to_optical(10.0, is_hz=True)  # 10 Hz folded to optical
    lo, hi = be.units.OPTICAL_BAND_ANGSTROM
    assert lo <= w <= hi


def test_spectrum_region():
    assert be.spectrum_region(5000) == "Visible light"


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def test_tables_load_and_categories():
    df = load_elements("air")
    assert {"element", "wavelength", "intensity", "type"} <= set(df.columns)
    assert df["intensity"].min() > 0            # non-positive lines dropped
    assert be.element_category("Iron") == "Transition Metals"
    assert len(ELEMENT_CATEGORIES) == 10


def test_unknown_element_raises():
    with pytest.raises(KeyError):
        be.element_table("Unobtainium")


# --------------------------------------------------------------------------- #
# spectrum
# --------------------------------------------------------------------------- #
def test_element_spectrum_normalise_is_unit():
    sp = element_spectrum("Oxygen", top=30, normalise=True)
    assert sp.intensity.sum() == pytest.approx(1.0, rel=1e-9)
    assert len(sp) <= 30


def test_superpose_preserves_labels():
    h = element_spectrum("Hydrogen", top=10, normalise=True)
    o = element_spectrum("Oxygen", top=10, normalise=True)
    comp = be.superpose([h.scaled(2), o.scaled(1)])
    dom = comp.dominant()
    assert dom["Hydrogen"] == pytest.approx(2 / 3, abs=0.02)


# --------------------------------------------------------------------------- #
# composition — the Phase-1 guarantee
# --------------------------------------------------------------------------- #
def test_water_budget_normalised_two_thirds_hydrogen():
    """Budget-normalisation makes stoichiometry, not NIST line count, decide."""
    water = MATERIALS["Water"]
    frac = water.elements()
    assert frac["Hydrogen"] == pytest.approx(2 / 3, abs=0.01)
    assert frac["Oxygen"] == pytest.approx(1 / 3, abs=0.01)
    dom = water.dominant(top=None)
    assert dom["Hydrogen"] == pytest.approx(2 / 3, abs=0.02)


def test_air_matches_reality():
    frac = MATERIALS["Air"].elements()
    assert frac["Nitrogen"] > frac["Oxygen"] > frac["Argon"]
    assert frac["Nitrogen"] == pytest.approx(0.78, abs=0.02)


def test_mass_basis_differs_from_atom():
    atom = MATERIALS["Water"].elements(basis="atom")
    mass = MATERIALS["Water"].elements(basis="mass")
    # by mass O (16) dominates H (1) despite 2:1 atom ratio
    assert mass["Oxygen"] > mass["Hydrogen"]
    assert atom["Hydrogen"] > atom["Oxygen"]


def test_recursion_cloud_is_water_and_air():
    cloud = MATERIALS["Cloud"]
    els = cloud.elements()
    assert "Hydrogen" in els and "Nitrogen" in els   # from water and from air
    assert len(cloud.spectrum()) > 0


def test_composite_spectrum_finite_and_sourced():
    sp = MATERIALS["Wood"].spectrum()
    assert len(sp) > 0
    assert np.all(np.isfinite(sp.wavelength))
    assert np.all(np.isfinite(sp.intensity))
    assert set(sp.dominant()) <= {"Carbon", "Hydrogen", "Oxygen"}


# --------------------------------------------------------------------------- #
# materials registry + coverage gate
# --------------------------------------------------------------------------- #
def test_registry_rejects_untagged_material():
    with pytest.raises(ValueError):
        material("Bogus", {"Oxygen": 1}, kind="compound", material_class="", domain="geosphere")


def test_coverage_report_passes_gate():
    rep = coverage_report(verbose=False)
    assert rep["passed"] is True
    assert rep["n_element_categories"] >= 8
    assert rep["material_classes_missing"] == []
    assert rep["domains_missing"] == []
    # exemplars at every compositional level
    assert set(rep["kinds"]) >= {"element", "compound", "mixture", "structure"}


# --------------------------------------------------------------------------- #
# matching — relative tolerance (the fixed bug)
# --------------------------------------------------------------------------- #
def test_match_elements_returns_hits():
    sig = np.array([7.83, 14.3, 20.8, 27.3, 33.8])
    ranked = match_elements(sig, top=40, tol_cents=60)
    assert len(ranked) > 0
    assert ranked["score"].iloc[0] >= ranked["score"].iloc[-1]   # sorted
    assert ranked["score"].iloc[0] > 0                            # not the 0-match bug


def test_match_materials_ranks():
    sig = np.array([7.83, 14.3, 20.8, 27.3, 33.8])
    ranked = match_materials(sig, tol_cents=60)
    assert list(ranked.columns) == ["material", "affinity", "kind", "archetype"]
    assert (ranked["affinity"].values[:-1] >= ranked["affinity"].values[1:]).all()


def test_match_balance_debiases_sparse_elements():
    """`balance='f1'` neutralises the recall ceiling that lets line-sparse
    elements (few lines → saturate) dominate; it must differ from recall and stay
    in [0,1]. A sparse element that saturates recall should lose ground under f1."""
    from biotuner.bioelements.matching import _match_score
    from biotuner.bioelements import element_spectrum
    sig = np.array([7.83, 14.3, 20.8, 27.3, 33.8])
    rec = match_elements(sig, top=40, tol_cents=60, balance="recall")
    f1 = match_elements(sig, top=40, tol_cents=60, balance="f1")
    assert (f1["score"] >= 0).all() and (f1["score"] <= 1).all()
    assert list(rec["element"].head(5)) != list(f1["element"].head(5))   # reranks
    # a 4-line element (Astatine) saturates recall but its f1 is capped by precision
    ast = element_spectrum("Astatine", top=40, normalise=True)
    r = _match_score(sig, ast, tol_cents=60, band=be.units.OPTICAL_BAND_ANGSTROM, balance="recall")
    f = _match_score(sig, ast, tol_cents=60, band=be.units.OPTICAL_BAND_ANGSTROM, balance="f1")
    assert f <= r + 1e-9


# --------------------------------------------------------------------------- #
# bridges
# --------------------------------------------------------------------------- #
def test_material_tuning_in_octave():
    ratios = material_tuning(MATERIALS["Water"], n_steps=6)
    assert len(ratios) >= 1
    assert all(1.0 <= r <= 2.0 for r in ratios)


def test_material_palette_emission_hexes():
    fire = MATERIALS["Fire"].palette(n=6)
    water = MATERIALS["Water"].palette(n=6)
    assert all(h.startswith("#") and len(h) == 7 for h in fire)
    assert len(fire) >= 1
    # different line sets -> genuinely different emission palettes
    assert set(fire) != set(water)


def test_flame_palettes_iconic_and_distinct():
    from biotuner.bioelements import element_flame_color, material_flame_palette
    assert element_flame_color("Sodium") == "#f4b028"    # iconic amber
    assert element_flame_color("Copper") == "#3fb883"    # iconic green
    water = material_flame_palette(MATERIALS["Water"])
    salt = material_flame_palette(MATERIALS["Halite"])
    assert all(c.startswith("#") and len(c) == 7 for c in water)
    assert set(water) != set(salt)                       # water (H/O) != salt (Na/Cl)


def test_plotting_functions_render():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from biotuner.bioelements import plotting as V
    sig = np.array([7.83, 14.3, 20.8, 27.3, 33.8])
    for fn in (V.plot_periodic_resonance, V.plot_element_resonance,
               V.plot_resonance_by_domain):
        fig, ax = fn(sig)
        assert fig is not None and ax is not None
    fig, _ = V.plot_line_match(sig, "Sulfur")
    assert fig is not None
    plt.close("all")


def test_periodic_layout_places_all_elements():
    from biotuner.bioelements import periodic as P
    from biotuner.bioelements import available_elements
    assert all(P.element_position(e) is not None for e in available_elements("air"))
    r, c = P.element_position("Iron")
    assert 0 <= r < P.N_ROWS and 0 <= c < P.N_COLS


def test_coverage_ten_of_ten():
    rep = coverage_report(verbose=False)
    assert rep["n_element_categories"] == 10
    assert rep["element_categories_missing"] == []


def test_material_geometry_builds_harmonic_input():
    from biotuner.harmonic_input import HarmonicInput
    hi = MATERIALS["Water"].geometry(top=6)
    assert isinstance(hi, HarmonicInput)
    assert hi.peaks is not None and len(hi.peaks) >= 1
    assert hi.metadata.get("material") == "Water"


# --------------------------------------------------------------------------- #
# legacy back-compat
# --------------------------------------------------------------------------- #
def test_legacy_names_present():
    assert be.Angstrom_to_hertz(5000) == pytest.approx(5.9958e14, rel=1e-3)
    assert callable(be.find_matching_spectral_lines)
    assert "Ultraviolet" in be.spectrum_nm


def test_legacy_compute_ratios_df_runs_on_pandas2():
    """The .append() -> pd.concat fix: this used to crash on pandas >= 2.0."""
    df = load_elements("air")
    small = df[df.element == "Hydrogen"].head(6).copy()
    out = be.compute_ratios_df(small, "all", "element")
    assert isinstance(out, pd.DataFrame)
    assert "ratio1" in out.columns and "ratio2" in out.columns
