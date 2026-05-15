"""
Medium abstraction and pipeline composition for :mod:`biotuner.harmonic_geometry.media`.

A :class:`Medium` is an operator that turns a forcing — a chord
(:class:`HarmonicInput`) or a pre-computed wave field
(:class:`GeometryData`) — into a :class:`GeometryData` response. Media are
organized into five families by the *kind* of operator they apply:

- ``eigenmode``     bounded standing-wave eigenproblem
- ``wave_field``   open-medium coherent superposition
- ``parametric``   instability under a 2f drive (Mathieu / Floquet)
- ``transport``    passive redistribution on a pre-existing wave field
- ``morphogenetic`` pattern growth via phase-field / reaction-diffusion

Composition rules
-----------------
Every medium implements ``respond(forcing, **overrides) -> GeometryData``.
The ``family`` attribute encodes the legal forcing types:

- ``eigenmode`` / ``wave_field`` / ``parametric`` consume :class:`HarmonicInput`.
- ``transport`` consumes :class:`GeometryData`; when given a
  :class:`HarmonicInput`, the medium auto-wraps with its documented default
  source (e.g. :class:`Granular` defaults to :class:`RigidPlate`).
- ``morphogenetic`` consumes either — the field becomes a forcing term,
  the chord shapes growth-anisotropy / RD parameters.

Pipelines chain media: ``Pipeline(RigidPlate(...), Granular(...))(chord)``
or equivalently the explicit functional form
``Granular(...).respond(RigidPlate(...).respond(chord))``.

Notes
-----
This module deliberately keeps the :class:`Medium` base lightweight: it is
an :class:`abc.ABC` with one abstract method and one optional default
source hook. Concrete media live in family-named subpackages (``eigenmode``,
``wave_field``, ``parametric``, ``transport``, ``morphogenetic``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Optional, Union

from biotuner.harmonic_geometry.geometry_data import GeometryData
from biotuner.harmonic_geometry.inputs import HarmonicInput

Family = Literal[
    "eigenmode", "wave_field", "parametric", "transport", "morphogenetic"
]

Forcing = Union[HarmonicInput, GeometryData]


# =========================================================== Domain types ==


@dataclass(frozen=True)
class Domain:
    """Abstract spatial domain. Subclassed by concrete shapes."""


@dataclass(frozen=True)
class Rectangular(Domain):
    """Rectangular plate / region of size ``(Lx, Ly)``."""

    Lx: float = 1.0
    Ly: float = 1.0


@dataclass(frozen=True)
class Circular(Domain):
    """Disk of radius ``R``."""

    R: float = 1.0


@dataclass(frozen=True)
class PolygonDomain(Domain):
    """Regular polygon with ``n_sides`` and circumradius ``radius``."""

    n_sides: int = 6
    radius: float = 1.0


@dataclass(frozen=True)
class Box3D(Domain):
    """Rectangular 3-D box ``(Lx, Ly, Lz)``."""

    Lx: float = 1.0
    Ly: float = 1.0
    Lz: float = 1.0


@dataclass(frozen=True)
class Sphere(Domain):
    """Unit-or-other sphere of radius ``R``."""

    R: float = 1.0


# ================================================================ Medium ==


class Medium(ABC):
    """Abstract base for every operator in :mod:`.media`.

    Subclasses set the class attribute ``family`` and implement
    :meth:`respond`. ``family`` controls which forcing types are valid;
    media that need a wave field will auto-wrap an incoming
    :class:`HarmonicInput` through :meth:`default_source` when called
    without a prior stage.
    """

    family: ClassVar[Family]

    # ------------------------------------------------------------- contract

    @abstractmethod
    def respond(self, forcing: Forcing, **overrides: Any) -> GeometryData:
        """Apply this medium to ``forcing`` and return the response field.

        Parameters
        ----------
        forcing : HarmonicInput or GeometryData
            For ``eigenmode`` / ``wave_field`` / ``parametric`` media:
            must be a :class:`HarmonicInput`.
            For ``transport`` / ``morphogenetic`` media: may be either —
            a :class:`HarmonicInput` is auto-wrapped through
            :meth:`default_source`.
        **overrides
            Per-call parameter overrides (subclass-specific).

        Returns
        -------
        GeometryData
        """

    # ------------------------------------------------------- default source

    def default_source(self) -> Optional["Medium"]:
        """Return the default upstream medium for auto-wrapping.

        ``transport`` and ``morphogenetic`` media override this to declare
        which wave-field stage they consume when a :class:`HarmonicInput`
        is passed directly. Returning ``None`` (the default) means the
        medium accepts :class:`HarmonicInput` directly and does not need
        a source.
        """
        return None

    # --------------------------------------------------------- shorthand

    def __call__(self, forcing: Forcing, **overrides: Any) -> GeometryData:
        return self.respond(forcing, **overrides)

    # ------------------------------------------------------- helpers

    def _resolve_field(self, forcing: Forcing) -> GeometryData:
        """Auto-wrap a :class:`HarmonicInput` through :meth:`default_source`.

        Used by transport / morphogenetic ``respond`` implementations to
        accept either a :class:`HarmonicInput` (auto-wrapped) or an
        upstream :class:`GeometryData`.
        """
        if isinstance(forcing, GeometryData):
            return forcing
        if isinstance(forcing, HarmonicInput):
            src = self.default_source()
            if src is None:
                raise TypeError(
                    f"{type(self).__name__} requires a GeometryData wave-field "
                    "input and does not declare a default_source(). Pre-compute "
                    "a field with an eigenmode/wave_field/parametric medium "
                    "first, then pass it in."
                )
            return src.respond(forcing)
        raise TypeError(
            f"forcing must be HarmonicInput or GeometryData; got "
            f"{type(forcing).__name__}."
        )


# ============================================================== Pipeline ==


class Pipeline:
    """Linear chain of media.

    ``Pipeline(A, B, C)(chord)`` is equivalent to
    ``C.respond(B.respond(A.respond(chord)))``.

    The pipeline does not enforce family compatibility between stages —
    that is the responsibility of each medium's :meth:`Medium.respond`,
    which raises if its input is the wrong type.
    """

    def __init__(self, *stages: Medium) -> None:
        if not stages:
            raise ValueError("Pipeline requires at least one stage.")
        for s in stages:
            if not isinstance(s, Medium):
                raise TypeError(
                    f"Pipeline stages must be Medium instances; got "
                    f"{type(s).__name__}."
                )
        self.stages = tuple(stages)

    def respond(self, forcing: Forcing) -> GeometryData:
        out: Any = forcing
        for s in self.stages:
            out = s.respond(out)
        return out

    def __call__(self, forcing: Forcing) -> GeometryData:
        return self.respond(forcing)

    def __repr__(self) -> str:
        names = " → ".join(type(s).__name__ for s in self.stages)
        return f"Pipeline({names})"
