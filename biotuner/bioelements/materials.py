"""The shipped material dictionary + a tag-enforcing registry + coverage audit.

Materials are :class:`Composition`s tagged on the four coverage axes (architecture
doc §4.5). The registry refuses a material missing a required tag, so the axes
cannot silently rot. :func:`coverage_report` turns "the dictionary spans matter"
into a measured, falsifiable claim: every material class and domain represented,
and ≥ 8 of the 10 NIST element categories exercised.
"""
from __future__ import annotations

from biotuner.bioelements import tables
from biotuner.bioelements.composition import Composition

#: name -> Composition
MATERIALS: dict[str, Composition] = {}

#: Canonical vocabularies for the two hand-set axes (used by the coverage audit).
MATERIAL_CLASSES = (
    "elemental", "molecular", "ionic-salt", "covalent-network", "metallic",
    "organic-polymer", "biomineral-composite", "composite-fluid", "amorphous-glass",
    "gas-mixture", "colloid-suspension", "plasma-energetic", "ice-volatile", "void",
)
DOMAINS = (
    "geosphere", "hydrosphere", "atmosphere", "biosphere", "cosmosphere",
    "technosphere", "energetic-process",
)


def register(comp: Composition) -> Composition:
    """Add a material to the registry, enforcing the required axis tags."""
    for req in ("kind", "material_class", "domain"):
        if not getattr(comp, req, ""):
            raise ValueError(
                f"material {comp.name!r} is missing required tag {req!r}; "
                f"every shipped material must set kind, material_class and domain."
            )
    MATERIALS[comp.name] = comp
    return comp


def material(name, parts=None, *, kind, material_class, domain,
             archetype="", **tags) -> Composition:
    """Build, tag, register and return a material in one call."""
    comp = Composition(name, dict(parts or {}), kind=kind,
                       material_class=material_class, domain=domain,
                       archetype=archetype, tags=tags)
    return register(comp)


# ======================================================================= #
# Elements (leaves)
# ======================================================================= #
Iron = material("Iron", kind="element", material_class="metallic",
                domain="geosphere", archetype="metal", formula="Fe")
Vacuum = material("Vacuum", kind="element", material_class="void",
                  domain="cosmosphere", archetype="ether",
                  note="the empty/ether archetype; reads the vacuum line table")

# ======================================================================= #
# Compounds (fixed stoichiometry, atom counts)
# ======================================================================= #
Water = material("Water", {"Hydrogen": 2, "Oxygen": 1}, kind="compound",
                 material_class="molecular", domain="hydrosphere",
                 archetype="water", formula="H2O")
CarbonDioxide = material("CarbonDioxide", {"Carbon": 1, "Oxygen": 2}, kind="compound",
                         material_class="molecular", domain="atmosphere", formula="CO2")
Halite = material("Halite", {"Sodium": 1, "Chlorine": 1}, kind="compound",
                  material_class="ionic-salt", domain="geosphere", formula="NaCl")
Quartz = material("Quartz", {"Silicon": 1, "Oxygen": 2}, kind="compound",
                  material_class="covalent-network", domain="geosphere",
                  archetype="earth", formula="SiO2")
Cellulose = material("Cellulose", {"Carbon": 6, "Hydrogen": 10, "Oxygen": 5},
                     kind="compound", material_class="organic-polymer",
                     domain="biosphere", formula="(C6H10O5)n")
Lignin = material("Lignin", {"Carbon": 9, "Hydrogen": 10, "Oxygen": 2},
                  kind="compound", material_class="organic-polymer",
                  domain="biosphere", formula="~C9H10O2")
Bone = material("Bone", {"Calcium": 5, "Phosphorus": 3, "Oxygen": 13, "Hydrogen": 1},
                kind="compound", material_class="biomineral-composite",
                domain="biosphere", formula="Ca5(PO4)3OH")
Chlorophyll = material("Chlorophyll",
                       {"Magnesium": 1, "Carbon": 55, "Hydrogen": 72, "Nitrogen": 4, "Oxygen": 5},
                       kind="compound", material_class="organic-polymer",
                       domain="biosphere", formula="C55H72MgN4O5")
SilicaGlass = material("SilicaGlass", {"Silicon": 1, "Oxygen": 2}, kind="compound",
                       material_class="amorphous-glass", domain="technosphere",
                       formula="SiO2 (amorphous)")
WaterIce = material("WaterIce", {"Hydrogen": 2, "Oxygen": 1}, kind="compound",
                    material_class="ice-volatile", domain="hydrosphere",
                    formula="H2O (solid)", state="ice")
EuropiumPhosphor = material("EuropiumPhosphor", {"Europium": 2, "Oxygen": 3},
                            kind="compound", material_class="ionic-salt",
                            domain="technosphere", formula="Eu2O3",
                            note="rare-earth red phosphor (lanthanide)")
UraniumGlass = material("UraniumGlass", {"Uranium": 1, "Silicon": 8, "Oxygen": 18},
                        kind="compound", material_class="amorphous-glass",
                        domain="technosphere", formula="~UO2·8SiO2",
                        note="uranyl-doped silica ('vaseline') glass (actinide)")

# ======================================================================= #
# Mixtures (proportioned blends of elements/compounds)
# ======================================================================= #
Air = material("Air", {"Nitrogen": 78.0, "Oxygen": 21.0, "Argon": 0.93, CarbonDioxide: 0.04},
               kind="mixture", material_class="gas-mixture", domain="atmosphere",
               archetype="air", basis="mole-fraction")
Wood = material("Wood", {Cellulose: 50.0, Lignin: 25.0, Water: 20.0}, kind="mixture",
                material_class="organic-polymer", domain="biosphere",
                archetype="wood", basis="dry-mass-approx")
SoftTissue = material("SoftTissue",
                      {"Hydrogen": 63.0, "Oxygen": 26.0, "Carbon": 9.0, "Nitrogen": 1.4},
                      kind="mixture", material_class="composite-fluid", domain="biosphere",
                      basis="atomic-percent (ICRP-style)")
Blood = material("Blood", {SoftTissue: 99.0, Iron: 0.5, "Sodium": 0.3, "Chlorine": 0.3},
                 kind="mixture", material_class="composite-fluid", domain="biosphere",
                 note="soft-tissue base with the Fe of haemoglobin + saline ions")
Bronze = material("Bronze", {"Copper": 88.0, "Tin": 12.0}, kind="mixture",
                  material_class="metallic", domain="technosphere", archetype="metal")

# ======================================================================= #
# Structures (recursive: materials of materials)
# ======================================================================= #
Cloud = material("Cloud", {Water: 5.0, Air: 95.0}, kind="structure",
                 material_class="colloid-suspension", domain="atmosphere",
                 note="water droplets suspended in air — the recursive case")
Fire = material("Fire", {"Carbon": 1.0, "Hydrogen": 2.0, "Oxygen": 2.0, "Sodium": 0.1},
                kind="structure", material_class="plasma-energetic",
                domain="energetic-process", archetype="fire",
                note="hot C/H/O plasma; Na D-lines dominate the visible glow")
Lightning = material("Lightning", {"Nitrogen": 78.0, "Oxygen": 21.0}, kind="structure",
                     material_class="plasma-energetic", domain="atmosphere",
                     note="ionised air")
StellarPlasma = material("StellarPlasma", {"Hydrogen": 92.0, "Helium": 8.0}, kind="structure",
                         material_class="plasma-energetic", domain="cosmosphere",
                         note="cosmic abundance H/He")

# ======================================================================= #
# Breadth pack — many more compounds & mixtures across the axes
# ======================================================================= #
# --- minerals / geosphere ---------------------------------------------- #
Calcite = material("Calcite", {"Calcium": 1, "Carbon": 1, "Oxygen": 3}, kind="compound",
                   material_class="ionic-salt", domain="geosphere", formula="CaCO3")
Hematite = material("Hematite", {"Iron": 2, "Oxygen": 3}, kind="compound",
                    material_class="ionic-salt", domain="geosphere", formula="Fe2O3")
Magnetite = material("Magnetite", {"Iron": 3, "Oxygen": 4}, kind="compound",
                     material_class="ionic-salt", domain="geosphere", formula="Fe3O4")
Pyrite = material("Pyrite", {"Iron": 1, "Sulfur": 2}, kind="compound",
                  material_class="ionic-salt", domain="geosphere", formula="FeS2")
Corundum = material("Corundum", {"Aluminum": 2, "Oxygen": 3}, kind="compound",
                    material_class="covalent-network", domain="geosphere", formula="Al2O3")
Feldspar = material("Feldspar", {"Potassium": 1, "Aluminum": 1, "Silicon": 3, "Oxygen": 8},
                    kind="compound", material_class="covalent-network", domain="geosphere",
                    formula="KAlSi3O8")
Gypsum = material("Gypsum", {"Calcium": 1, "Sulfur": 1, "Oxygen": 6, "Hydrogen": 4},
                  kind="compound", material_class="ionic-salt", domain="geosphere",
                  formula="CaSO4.2H2O")
Rutile = material("Rutile", {"Titanium": 1, "Oxygen": 2}, kind="compound",
                  material_class="covalent-network", domain="geosphere", formula="TiO2")
Fluorite = material("Fluorite", {"Calcium": 1, "Fluorine": 2}, kind="compound",
                    material_class="ionic-salt", domain="geosphere", formula="CaF2")
Galena = material("Galena", {"Lead": 1, "Sulfur": 1}, kind="compound",
                  material_class="ionic-salt", domain="geosphere", formula="PbS")
Kaolinite = material("Kaolinite", {"Aluminum": 2, "Silicon": 2, "Oxygen": 9, "Hydrogen": 4},
                     kind="compound", material_class="covalent-network", domain="geosphere",
                     formula="Al2Si2O5(OH)4", note="clay mineral")

# --- biomolecules / biosphere ------------------------------------------ #
Glucose = material("Glucose", {"Carbon": 6, "Hydrogen": 12, "Oxygen": 6}, kind="compound",
                   material_class="organic-polymer", domain="biosphere", formula="C6H12O6")
Glycine = material("Glycine", {"Carbon": 2, "Hydrogen": 5, "Nitrogen": 1, "Oxygen": 2},
                   kind="compound", material_class="organic-polymer", domain="biosphere",
                   formula="C2H5NO2", note="simplest amino acid")
Urea = material("Urea", {"Carbon": 1, "Hydrogen": 4, "Nitrogen": 2, "Oxygen": 1},
                kind="compound", material_class="organic-polymer", domain="biosphere",
                formula="CH4N2O")
DNA = material("DNA", {"Carbon": 10, "Hydrogen": 13, "Nitrogen": 4, "Oxygen": 7, "Phosphorus": 1},
               kind="mixture", material_class="organic-polymer", domain="biosphere",
               note="mean nucleotide composition")
ATP = material("ATP", {"Carbon": 10, "Hydrogen": 16, "Nitrogen": 5, "Oxygen": 13, "Phosphorus": 3},
               kind="compound", material_class="organic-polymer", domain="biosphere",
               formula="C10H16N5O13P3")
Melanin = material("Melanin", {"Carbon": 18, "Hydrogen": 10, "Nitrogen": 2, "Oxygen": 4},
                   kind="mixture", material_class="organic-polymer", domain="biosphere",
                   note="eumelanin monomer approx")
Collagen = material("Collagen", {"Carbon": 30, "Hydrogen": 45, "Nitrogen": 9, "Oxygen": 12},
                    kind="mixture", material_class="organic-polymer", domain="biosphere",
                    note="mean residue approx")

# --- gases / atmosphere ------------------------------------------------- #
Methane = material("Methane", {"Carbon": 1, "Hydrogen": 4}, kind="compound",
                   material_class="molecular", domain="atmosphere", formula="CH4")
Ammonia = material("Ammonia", {"Nitrogen": 1, "Hydrogen": 3}, kind="compound",
                   material_class="molecular", domain="atmosphere", formula="NH3")
Ozone = material("Ozone", {"Oxygen": 3}, kind="compound",
                 material_class="molecular", domain="atmosphere", formula="O3")
CarbonMonoxide = material("CarbonMonoxide", {"Carbon": 1, "Oxygen": 1}, kind="compound",
                          material_class="molecular", domain="atmosphere", formula="CO")
HydrogenSulfide = material("HydrogenSulfide", {"Hydrogen": 2, "Sulfur": 1}, kind="compound",
                           material_class="molecular", domain="atmosphere", formula="H2S")

# --- alloys / technosphere --------------------------------------------- #
Steel = material("Steel", {"Iron": 98.0, "Carbon": 2.0}, kind="mixture",
                 material_class="metallic", domain="technosphere", archetype="metal")
Brass = material("Brass", {"Copper": 65.0, "Zinc": 35.0}, kind="mixture",
                 material_class="metallic", domain="technosphere", archetype="metal")
StainlessSteel = material("StainlessSteel", {"Iron": 70.0, "Chromium": 18.0, "Nickel": 8.0, "Carbon": 1.0},
                          kind="mixture", material_class="metallic", domain="technosphere",
                          archetype="metal")
Solder = material("Solder", {"Tin": 60.0, "Lead": 40.0}, kind="mixture",
                  material_class="metallic", domain="technosphere")

# --- rocks & waters (mixtures/structures) ------------------------------ #
Granite = material("Granite", {Quartz: 30.0, Feldspar: 60.0, "Aluminum": 5.0, "Iron": 5.0},
                   kind="mixture", material_class="composite-fluid", domain="geosphere",
                   note="felsic rock: quartz + feldspar + accessory minerals")
Seawater = material("Seawater", {Water: 96.5, Halite: 3.0, "Magnesium": 0.4, "Calcium": 0.1},
                    kind="mixture", material_class="composite-fluid", domain="hydrosphere")
Soil = material("Soil", {Quartz: 45.0, Water: 25.0, Cellulose: 10.0, "Aluminum": 10.0, "Iron": 10.0},
                kind="mixture", material_class="composite-fluid", domain="geosphere")

# --- cosmic ------------------------------------------------------------- #
InterstellarDust = material("InterstellarDust",
                            {"Silicon": 1.0, "Oxygen": 2.0, "Carbon": 1.0, "Iron": 0.5, "Magnesium": 0.5},
                            kind="mixture", material_class="composite-fluid",
                            domain="cosmosphere", note="silicate + carbonaceous grains")
CometIce = material("CometIce", {WaterIce: 80.0, CarbonDioxide: 15.0, Ammonia: 5.0},
                    kind="structure", material_class="ice-volatile", domain="cosmosphere")


# ======================================================================= #
# Full periodic breadth — every NIST element as a first-class material
# ======================================================================= #
_ELEMENT_DOMAIN_BY_CATEGORY = {
    "Noble Gases": "atmosphere",
    "Nonmetals": "biosphere",
    "Halogens": "geosphere",
}
_ELEMENT_DOMAIN_OVERRIDE = {
    "Hydrogen": "cosmosphere", "Helium": "cosmosphere",
    "Nitrogen": "atmosphere", "Oxygen": "atmosphere", "Carbon": "biosphere",
}


def _register_all_elements(table: str = "air") -> int:
    """Promote every NIST element to a first-class element-material.

    Already-registered elements (e.g. the curated ``Iron``) are left untouched.
    Domain is inferred from the element's periodic category; material_class is
    ``'elemental'`` (the metal/nonmetal/gas nature is carried by the derived
    element-category axis, not duplicated here). Returns the count added.
    """
    added = 0
    for el in tables.available_elements(table):
        if el in MATERIALS:
            continue
        cat = tables.element_category(el, table)
        domain = _ELEMENT_DOMAIN_OVERRIDE.get(el) or _ELEMENT_DOMAIN_BY_CATEGORY.get(cat, "geosphere")
        material(el, kind="element", material_class="elemental", domain=domain, category=cat)
        added += 1
    return added


_register_all_elements()


# ======================================================================= #
# Coverage audit — the build gate
# ======================================================================= #
def coverage_report(materials: dict | None = None, *, table: str = "air",
                    verbose: bool = True) -> dict:
    """Audit the dictionary across the four coverage axes.

    Returns a dict with the per-axis fill and a ``passed`` flag that is True iff
    every material class and domain is represented **and** ≥ 8 of the 10 NIST
    element categories are exercised.
    """
    mats = MATERIALS if materials is None else materials
    comps = list(mats.values())

    kinds = {}
    classes = set()
    domains = set()
    all_elements = set()
    for c in comps:
        kinds[c.kind] = kinds.get(c.kind, 0) + 1
        classes.add(c.material_class)
        domains.add(c.domain)
        all_elements.update(c.elements().keys())

    element_cats = tables.category_coverage(all_elements, table)

    class_missing = [c for c in MATERIAL_CLASSES if c not in classes]
    domain_missing = [d for d in DOMAINS if d not in domains]
    cat_missing = [c for c in tables.ELEMENT_CATEGORIES if c not in element_cats]

    passed = (not class_missing) and (not domain_missing) and (len(element_cats) >= 8)

    report = {
        "n_materials": len(comps),
        "kinds": kinds,
        "material_classes_filled": sorted(classes),
        "material_classes_missing": class_missing,
        "domains_filled": sorted(domains),
        "domains_missing": domain_missing,
        "element_categories": sorted(element_cats),
        "element_categories_missing": cat_missing,
        "n_element_categories": len(element_cats),
        "passed": passed,
    }

    if verbose:
        print(f"bioelements coverage — {len(comps)} materials")
        print(f"  kinds:            {kinds}")
        print(f"  classes  {len(classes)}/{len(MATERIAL_CLASSES)}: "
              f"missing {class_missing or 'none'}")
        print(f"  domains  {len(domains)}/{len(DOMAINS)}: "
              f"missing {domain_missing or 'none'}")
        print(f"  element categories {len(element_cats)}/10 "
              f"(gate >=8): {'PASS' if len(element_cats) >= 8 else 'FAIL'}")
        print(f"      have:    {sorted(element_cats)}")
        print(f"      missing: {cat_missing or 'none'}")
        print(f"  OVERALL: {'PASS' if passed else 'FAIL'}")

    return report
