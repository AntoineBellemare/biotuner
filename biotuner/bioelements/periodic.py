"""Periodic-table geometry — element name -> (row, col) on the standard table.

Used to render a biosignal's element resonance *as the periodic table itself*.
Lanthanides and actinides are placed in the two detached f-block rows (rows 7-8).
"""
from __future__ import annotations

# Z-ordered symbols and NIST-style full names, elements 1..99.
SYMBOLS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al",
    "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm",
    "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
    "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es",
]
NAMES = [
    "Hydrogen", "Helium", "Lithium", "Beryllium", "Boron", "Carbon", "Nitrogen",
    "Oxygen", "Fluorine", "Neon", "Sodium", "Magnesium", "Aluminum", "Silicon",
    "Phosphorus", "Sulfur", "Chlorine", "Argon", "Potassium", "Calcium",
    "Scandium", "Titanium", "Vanadium", "Chromium", "Manganese", "Iron",
    "Cobalt", "Nickel", "Copper", "Zinc", "Gallium", "Germanium", "Arsenic",
    "Selenium", "Bromine", "Krypton", "Rubidium", "Strontium", "Yttrium",
    "Zirconium", "Niobium", "Molybdenum", "Technetium", "Ruthenium", "Rhodium",
    "Palladium", "Silver", "Cadmium", "Indium", "Tin", "Antimony", "Tellurium",
    "Iodine", "Xenon", "Cesium", "Barium", "Lanthanum", "Cerium",
    "Praseodymium", "Neodymium", "Promethium", "Samarium", "Europium",
    "Gadolinium", "Terbium", "Dysprosium", "Holmium", "Erbium", "Thulium",
    "Ytterbium", "Lutetium", "Hafnium", "Tantalum", "Tungsten", "Rhenium",
    "Osmium", "Iridium", "Platinum", "Gold", "Mercury", "Thallium", "Lead",
    "Bismuth", "Polonium", "Astatine", "Radon", "Francium", "Radium",
    "Actinium", "Thorium", "Protactinium", "Uranium", "Neptunium", "Plutonium",
    "Americium", "Curium", "Berkelium", "Californium", "Einsteinium",
]

# The standard 18-wide layout; None = empty cell. La/Ac sit in the main body,
# Ce-Lu and Th-Es fill the two f-block rows underneath.
_GRID = [
    ["H",  None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, "He"],
    ["Li", "Be", None, None, None, None, None, None, None, None, None, None, "B",  "C",  "N",  "O",  "F",  "Ne"],
    ["Na", "Mg", None, None, None, None, None, None, None, None, None, None, "Al", "Si", "P",  "S",  "Cl", "Ar"],
    ["K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr"],
    ["Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe"],
    ["Cs", "Ba", "La", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"],
    ["Fr", "Ra", "Ac", None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
    [None, None, None, "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", None],
    [None, None, None, "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", None, None, None, None, None],
]

_SYM_POS = {sym: (r, c) for r, row in enumerate(_GRID) for c, sym in enumerate(row) if sym}
_NAME_TO_SYM = dict(zip(NAMES, SYMBOLS))
SYMBOL = _NAME_TO_SYM
Z_OF = {name: i + 1 for i, name in enumerate(NAMES)}
N_ROWS = len(_GRID)
N_COLS = 18


# Characteristic flame / discharge colours — the culturally-known emission colours
# (softened to read rich rather than neon). Elements not listed fall back to their
# strongest visible NIST line. Sources: standard flame-test + discharge-tube colours.
FLAME_COLORS = {
    "Hydrogen": "#e05a8a", "Helium": "#f3e2a0", "Lithium": "#d6294a",
    "Beryllium": "#cfe8d8", "Boron": "#6cba54", "Carbon": "#e0863c",
    "Nitrogen": "#6f66c8", "Oxygen": "#79aee0", "Fluorine": "#e4e69a",
    "Neon": "#f0552f", "Sodium": "#f4b028", "Magnesium": "#eef0f0",
    "Aluminum": "#d3d6e0", "Silicon": "#b8c0cc", "Phosphorus": "#8ecab0",
    "Sulfur": "#5a7fb0", "Chlorine": "#a6d472", "Argon": "#9a80dc",
    "Potassium": "#c07de0", "Calcium": "#f2703a", "Scandium": "#d0a0c0",
    "Titanium": "#cfd4dc", "Vanadium": "#9fbf6f", "Chromium": "#c8d060",
    "Manganese": "#b6c23a", "Iron": "#e0a24a", "Cobalt": "#8fb0d0",
    "Nickel": "#c8ccd0", "Copper": "#3fb883", "Zinc": "#7fc9b0",
    "Gallium": "#b0c0d0", "Arsenic": "#6f96cc", "Selenium": "#5f7fd0",
    "Bromine": "#c07a4a", "Krypton": "#dfe6ee", "Rubidium": "#c24a8f",
    "Strontium": "#e03c46", "Barium": "#8fc040", "Silver": "#dfeee0",
    "Cadmium": "#c0b0d0", "Indium": "#8a6fd0", "Tin": "#c0c8d0",
    "Antimony": "#a8c078", "Iodine": "#8a52be", "Xenon": "#bcd0f0",
    "Cesium": "#7a6ae0", "Gold": "#f0cf50", "Mercury": "#a898dc",
    "Lead": "#aec2dc", "Uranium": "#7fbe3f", "Europium": "#e04858",
    "Cerium": "#dcc888", "Lanthanum": "#cbb0d8", "Thorium": "#b8cc90",
}


def element_position(name: str):
    """(row, col) of an element on the periodic table, or ``None`` if unknown."""
    sym = _NAME_TO_SYM.get(name)
    return _SYM_POS.get(sym) if sym else None


def symbol(name: str) -> str:
    """Chemical symbol for a full element name (empty string if unknown)."""
    return _NAME_TO_SYM.get(name, "")
