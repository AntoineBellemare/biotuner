"""The :class:`Composition` model — an element, or a weighted bag of sub-compositions.

One dataclass and one recursive operation ("superpose the component spectra,
weighted, each budget-normalised first") carry every level of the matter hierarchy:

    element  ->  compound  ->  mixture  ->  structure

An element is a leaf (``parts`` empty). A compound weights elements by
stoichiometry. A mixture weights compounds/elements by proportion. A structure
weights *materials* (its keys are themselves ``Composition``s), so composition is
recursive — ``cloud = Composition("Cloud", {water: 0.05, air: 0.95})``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from biotuner.bioelements import tables
from biotuner.bioelements.spectrum import Spectrum, element_spectrum, superpose

#: Standard atomic weights (u) for mass-basis stoichiometry. Covers the elements
#: the shipped dictionary uses; missing elements fall back to atom basis.
ATOMIC_MASS = {
    "Hydrogen": 1.008, "Helium": 4.003, "Lithium": 6.94, "Beryllium": 9.012,
    "Boron": 10.81, "Carbon": 12.011, "Nitrogen": 14.007, "Oxygen": 15.999,
    "Fluorine": 18.998, "Neon": 20.180, "Sodium": 22.990, "Magnesium": 24.305,
    "Aluminum": 26.982, "Aluminium": 26.982, "Silicon": 28.085, "Phosphorus": 30.974,
    "Sulfur": 32.06, "Chlorine": 35.45, "Argon": 39.948, "Potassium": 39.098,
    "Calcium": 40.078, "Titanium": 47.867, "Chromium": 51.996, "Manganese": 54.938,
    "Iron": 55.845, "Cobalt": 58.933, "Nickel": 58.693, "Copper": 63.546,
    "Zinc": 65.38, "Tin": 118.71, "Iodine": 126.90, "Gold": 196.97,
    "Mercury": 200.59, "Lead": 207.2, "Uranium": 238.03,
}


@dataclass(eq=False)   # identity-based hash so a Composition can be a dict key (structures)
class Composition:
    """An element or a weighted composition of materials.

    Every shipped material also records where it sits on the coverage axes
    (§4.5 of the architecture doc) as tags, so the dictionary is auditable.
    """
    name: str
    parts: dict = field(default_factory=dict)   # component (str | Composition) -> weight
    kind: str = "element"                        # AXIS 1  element|compound|mixture|structure
    material_class: str = ""                     # AXIS 2  molecular|ionic-salt|metallic|…
    domain: str = ""                             # AXIS 3  geosphere|hydrosphere|atmosphere|…
    archetype: str = ""                          # earth|water|air|fire|wood|metal|ether|""
    tags: dict = field(default_factory=dict)     # free-form (state, formula, source, citation)

    def __repr__(self):
        if not self.parts:
            return f"Composition(element {self.name!r})"
        return f"Composition({self.name!r}, kind={self.kind!r}, {len(self.parts)} parts)"

    # --- resolution ------------------------------------------------------- #
    @staticmethod
    def _resolve(key) -> "Composition":
        """A part key -> a Composition: pass Compositions through; a string is a
        registered material if known, else an element leaf."""
        if isinstance(key, Composition):
            return key
        from biotuner.bioelements.materials import MATERIALS  # lazy (avoids cycle)
        if key in MATERIALS:
            return MATERIALS[key]
        return Composition(str(key), {})

    def is_element(self) -> bool:
        return not self.parts

    # --- the core operation ---------------------------------------------- #
    def spectrum(self, *, table: str = "air", top: int | None = 40,
                 basis: str = "atom") -> Spectrum:
        """Composite spectrum = Σ wᵢ · normalise(spectrum(partᵢ)), recursively.

        Budget-normalising each part before weighting is the calibration that
        makes stoichiometry — not NIST line count — decide dominance.

        ``basis`` : ``'atom'`` (weights are atom counts, the default) or
        ``'mass'`` (each element's weight is multiplied by its atomic mass, so
        "how much mass resonates" rather than "how many atoms").
        """
        if self.is_element():
            try:
                return element_spectrum(self.name, table=table, top=top, normalise=True)
            except KeyError:
                # an element name with no NIST lines (e.g. the Vacuum/ether archetype)
                # is a void: an empty spectrum, not an error.
                return Spectrum(np.array([]), np.array([]),
                                np.array([], dtype=object), name=self.name)

        parts = []
        for key, w in self.parts.items():
            comp = self._resolve(key)
            weight = float(w)
            if comp.is_element():
                if basis == "mass":
                    weight *= ATOMIC_MASS.get(comp.name, 1.0)
                sp = element_spectrum(comp.name, table=table, top=top, normalise=True)
            else:
                sp = comp.spectrum(table=table, top=top, basis=basis).normalise()
            parts.append(sp.scaled(weight))

        out = superpose(parts)
        out.name = self.name
        return out

    # --- audit helpers ---------------------------------------------------- #
    def elements(self, *, basis: str = "atom") -> dict:
        """Flatten to leaf elements with their normalised effective fractions.

        This is the atom- (or mass-) fraction audit: for water it returns
        ``{'Hydrogen': 0.667, 'Oxygen': 0.333}`` under the atom basis.
        """
        if self.is_element():
            return {self.name: 1.0}
        acc: dict[str, float] = {}
        for key, w in self.parts.items():
            comp = self._resolve(key)
            sub = comp.elements(basis=basis)
            subtot = sum(sub.values()) or 1.0
            for e, ew in sub.items():
                weight = float(w) * (ew / subtot)
                if basis == "mass" and comp.is_element():
                    weight *= ATOMIC_MASS.get(e, 1.0)
                acc[e] = acc.get(e, 0.0) + weight
        tot = sum(acc.values()) or 1.0
        return dict(sorted(((e, v / tot) for e, v in acc.items()),
                           key=lambda kv: kv[1], reverse=True))

    def dominant(self, *, table: str = "air", top: int | None = 40,
                 basis: str = "atom") -> dict:
        """Fraction of composite spectral intensity per source element."""
        return self.spectrum(table=table, top=top, basis=basis).dominant()

    def category_coverage(self, *, table: str = "air") -> set:
        """The set of NIST element categories this material's elements span."""
        return tables.category_coverage(self.elements().keys(), table)

    # --- affordances (thin bridges; see bridges.py) ----------------------- #
    def tuning(self, **kw):
        from biotuner.bioelements import bridges
        return bridges.material_tuning(self, **kw)

    def palette(self, **kw):
        from biotuner.bioelements import bridges
        return bridges.material_palette(self, **kw)

    def chord(self, **kw):
        from biotuner.bioelements import bridges
        return bridges.material_chord(self, **kw)

    def geometry(self, **kw):
        from biotuner.bioelements import bridges
        return bridges.material_geometry(self, **kw)

    def affinity(self, peaks_hz, **kw):
        from biotuner.bioelements import affinity as _aff
        return _aff.material_affinity(peaks_hz, self, **kw)
