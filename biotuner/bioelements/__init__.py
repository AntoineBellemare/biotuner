"""biotuner.bioelements — biosignals ↔ the periodic table ↔ materials.

Every atom emits at a characteristic set of wavelengths; ``bioelements`` octave-folds
a biosignal's peaks into the optical band and asks which elements' lines they land on.
The expansion lifts this from *atoms* to *materials*: a :class:`Composition` is a
weighted bag of elements (or of other compositions), so water, air, wood, a cloud —
any composition of matter — has its own composite spectrum, and therefore its own
chord, tuning, palette, and biosignal affinity.

Quick start
-----------
>>> from biotuner.bioelements import MATERIALS, coverage_report, match_materials
>>> coverage_report()                        # audit the dictionary across 4 axes
>>> water = MATERIALS["Water"]
>>> water.elements()                         # {'Hydrogen': 0.667, 'Oxygen': 0.333}
>>> water.tuning()                           # a material scale
>>> match_materials([7.83, 14.3, 20.8])      # rank a signal against every material

Layers
------
- :mod:`~biotuner.bioelements.units`        Å ↔ Hz ↔ nm ↔ eV, EM regions, octave-fold
- :mod:`~biotuner.bioelements.tables`       the NIST line tables, in-package, by name
- :mod:`~biotuner.bioelements.spectrum`     :class:`Spectrum` + budget-normalisation
- :mod:`~biotuner.bioelements.composition`  :class:`Composition` + recursive superposition
- :mod:`~biotuner.bioelements.materials`    the tagged material dictionary + registry
- :mod:`~biotuner.bioelements.matching`     relative-tolerance biosignal ↔ element matcher
- :mod:`~biotuner.bioelements.affinity`     biosignal → material affinity
- :mod:`~biotuner.bioelements.bridges`      material → tuning / chord / palette
- :mod:`~biotuner.bioelements.legacy`       the original flat API, frozen

Back-compat
-----------
Every name from the old flat ``bioelements.py`` is re-exported, so
``from biotuner import bioelements as be; be.find_matching_spectral_lines(...)`` and
``be.Angstrom_to_hertz(...)`` keep working (now from :mod:`~biotuner.bioelements.legacy`).
"""

# --- submodules ---------------------------------------------------------- #
from biotuner.bioelements import (
    units, tables, spectrum, composition, materials, matching, affinity, bridges,
    periodic, plotting, dynamics, tuning_match, legacy,
)

# --- new API ------------------------------------------------------------- #
from biotuner.bioelements.units import (
    angstrom_to_hertz, hertz_to_angstrom, nm_to_hertz, hertz_to_nm,
    hertz_to_volt, spectrum_region, fold_to_optical, SPECTRUM_NM,
)
from biotuner.bioelements.tables import (
    load_elements, element_table, available_elements, element_category,
    category_coverage, ELEMENT_CATEGORIES,
)
from biotuner.bioelements.spectrum import Spectrum, element_spectrum, superpose
from biotuner.bioelements.composition import Composition, ATOMIC_MASS
from biotuner.bioelements.materials import (
    Composition as _Composition, material, register, MATERIALS, coverage_report,
    MATERIAL_CLASSES, DOMAINS,
)
from biotuner.bioelements.matching import cents, match_lines, match_elements
from biotuner.bioelements.affinity import material_affinity, match_materials
from biotuner.bioelements.bridges import (
    material_tuning, material_chord, material_palette, material_biocolors_palette,
    material_geometry, element_flame_color, material_flame_palette,
)
# --- dynamics (Phase 1: state & coherence) ------------------------------- #
from biotuner.bioelements.dynamics import (
    MaterialState, element_state, state_from_signal, coherence,
    compositional_level, material_state, material_order, plot_material_state,
)
# --- tuning matching (ratio-structure, transposition-invariant) ---------- #
from biotuner.bioelements.tuning_match import (
    tuning_vector, spectrum_tuning, element_tuning, material_tuning_vector, tuning_cosine,
    match_tuning, match_elements_by_tuning, match_materials_by_tuning,
)

# --- legacy surface (back-compat) --------------------------------------- #
from biotuner.bioelements.legacy import (
    Angstrom_to_hertz, find_matching_spectral_lines, plot_type_proportions,
    compute_ratios_df, spectrum_nm, spectrum_Angstrom, spectrum_hertz, spectrum_volt,
)

__all__ = [
    # submodules
    "units", "tables", "spectrum", "composition", "materials", "matching",
    "affinity", "bridges", "periodic", "legacy",
    # data + spectra
    "load_elements", "element_table", "available_elements", "element_category",
    "category_coverage", "ELEMENT_CATEGORIES",
    "Spectrum", "element_spectrum", "superpose",
    # model + dictionary
    "Composition", "ATOMIC_MASS", "material", "register", "MATERIALS",
    "coverage_report", "MATERIAL_CLASSES", "DOMAINS",
    # matching + affordances
    "cents", "match_lines", "match_elements", "material_affinity", "match_materials",
    "material_tuning", "material_chord", "material_palette", "material_biocolors_palette",
    "material_geometry",
    # units
    "angstrom_to_hertz", "hertz_to_angstrom", "nm_to_hertz", "hertz_to_nm",
    "hertz_to_volt", "spectrum_region", "fold_to_optical", "SPECTRUM_NM",
    # dynamics (state & coherence)
    "MaterialState", "element_state", "state_from_signal", "coherence",
    "compositional_level", "material_state", "material_order", "plot_material_state",
    # tuning matching (ratio structure)
    "tuning_vector", "spectrum_tuning", "element_tuning", "material_tuning_vector",
    "tuning_cosine", "match_tuning", "match_elements_by_tuning", "match_materials_by_tuning",
    # legacy back-compat
    "Angstrom_to_hertz", "find_matching_spectral_lines", "plot_type_proportions",
    "compute_ratios_df", "spectrum_nm", "spectrum_Angstrom", "spectrum_hertz",
    "spectrum_volt",
]
