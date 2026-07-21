"""Access the NIST emission-line tables shipped inside the package.

Two tables are available, named as in NIST: ``"air"`` (air-wavelength lines, the
terrestrial default) and ``"vacuum"`` (vacuum-wavelength lines, for astrophysical
work / the ``ether`` archetype). Both are loaded lazily and cached.

Columns: ``element, wavelength (Å), intensity, persistence, type, spectrum_region``.
``type`` is the element's periodic category (Transition Metals, Noble Gases, …) and
drives the element-category coverage axis.
"""
from __future__ import annotations

from functools import lru_cache
from importlib import resources

import numpy as np
import pandas as pd

_TABLES = {"air": "air_elements.csv.gz", "vacuum": "vacuum_elements.csv.gz"}

#: The 10 element categories present in the NIST ``type`` column.
ELEMENT_CATEGORIES = (
    "Alkali Metals", "Alkaline Earth Metals", "Transition Metals",
    "Post-transition Metals", "Metalloids", "Nonmetals", "Halogens",
    "Noble Gases", "Lanthanides", "Actinides",
)


@lru_cache(maxsize=2)
def load_elements(table: str = "air") -> pd.DataFrame:
    """Load (and cache) a NIST line table by name (``"air"`` or ``"vacuum"``)."""
    if table not in _TABLES:
        raise ValueError(f"unknown table {table!r}; choose from {sorted(_TABLES)}")
    # Reference the parent package (a real package with __init__.py) and navigate
    # into data/. On Python 3.9, resources.files() of the data/ *namespace* package
    # (no __init__.py) crashes because spec.origin is None; the parent package has a
    # valid origin. joinpath is chained (single-arg) for 3.9 compatibility.
    data = resources.files("biotuner.bioelements").joinpath("data")
    with data.joinpath(_TABLES[table]).open("rb") as fh:
        df = pd.read_csv(fh, compression="gzip")
    # normalise dtypes: wavelength/intensity numeric, drop unusable rows
    df["wavelength"] = pd.to_numeric(df["wavelength"], errors="coerce")
    df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")
    df = df.dropna(subset=["wavelength", "intensity"])
    df = df[df["intensity"] > 0].reset_index(drop=True)
    return df


@lru_cache(maxsize=64)
def element_table(element: str, table: str = "air") -> pd.DataFrame:
    """The line sub-table for one element, sorted by descending intensity."""
    df = load_elements(table)
    sub = df[df["element"] == element]
    if sub.empty:
        raise KeyError(
            f"element {element!r} not found in the {table!r} table; "
            f"use full names, e.g. 'Hydrogen', 'Oxygen', 'Iron'."
        )
    return sub.sort_values("intensity", ascending=False).reset_index(drop=True)


def available_elements(table: str = "air") -> list[str]:
    """All element names present in a table."""
    return sorted(load_elements(table)["element"].unique().tolist())


def element_category(element: str, table: str = "air") -> str:
    """The periodic category (NIST ``type``) of an element."""
    return str(element_table(element, table)["type"].iloc[0])


def category_coverage(elements, table: str = "air") -> set[str]:
    """The set of element categories spanned by a collection of elements."""
    cats = set()
    for e in set(elements):
        try:
            cats.add(element_category(e, table))
        except KeyError:
            continue
    return cats
