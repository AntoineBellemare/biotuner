"""The :class:`MOSScale` object -- one well-formed scale, fully specified.

An MOS scale is pinned down by three things: how many large and small steps it
has, where its generator sits inside the valid range for that signature, and
what the period is.  Everything else -- degrees, cents, step sizes, landmarks,
propriety, modes, inverse, embedding family -- follows, and is exposed here as
derived properties rather than as parallel lists.

The abstract structure (``5L 2s``) and the concrete tuning (generator = 702
cents) are deliberately kept as separate coordinates, because that is exactly
the two-part selection the scale labyrinth affords: "the scale labyrinth allows
a musician to choose, simultaneously, a scale structure (number of small and
large steps) and its tuning (the sizes of its period and generator)"
(Milne et al., 2011, §1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

from biotuner.mos import theory as T
from biotuner.mos.theory import (
    Landmarks,
    PERIOD_CENTS,
    generator_fraction,
    mediant,
    noble_mediant,
)

if TYPE_CHECKING:  # pragma: no cover
    from biotuner.mos.modes import Mode

__all__ = ["MOSScale", "mos", "mos_family"]

#: Accepted spellings for :meth:`MOSScale.from_signature`'s ``tuning``.
TuningSpec = Union[str, float, int, Fraction, None]


def _resolve_tuning(
    n_large: int, n_small: int, tuning: TuningSpec, bright: bool
) -> float:
    """Turn a tuning spec into a concrete generator fraction inside the range."""
    lo, hi = T.signature_ranges(n_large, n_small)[1 if bright else 0]
    lm = T.mos_landmarks(n_large, n_small, bright=bright)

    if tuning is None or tuning == "noble":
        # The φ-weighted mediant: the point furthest from every simple
        # rational, so the scale never collapses into an equal temperament.
        return noble_mediant(lo, hi)
    if isinstance(tuning, str):
        if tuning == "equalized":
            return float(lm.equalized)
        if tuning == "central":
            # Middle of the coherent sub-range: safely proper, never degenerate.
            c_lo, c_hi = T.coherence_range(n_large, n_small, bright=bright)
            return (float(c_lo) + float(c_hi)) / 2.0
        if tuning == "middle":
            return (float(lo) + float(hi)) / 2.0
        raise ValueError(
            f"tuning must be a generator fraction, an EDO integer, a Fraction, "
            f"or one of 'noble'/'central'/'middle'/'equalized'; got {tuning!r}"
        )
    if isinstance(tuning, Fraction):
        return float(tuning)
    if isinstance(tuning, int) and not isinstance(tuning, bool):
        # An EDO: pick the step count whose n/EDO lands inside the range.
        #
        # The range is closed at both ends, but only one end is habitable. At
        # the equalized landmark L == s: the scale flattens to an equal
        # temperament but still has all its notes, and that is frequently the
        # scale being asked for -- 5L7s in 12-EDO *is* the chromatic scale, and
        # it sits exactly there. At the other end a step size reaches zero and
        # the notes either side of it collide, so the object would claim a
        # signature it does not have: 4L3s in 12-EDO would come back with four
        # distinct pitches out of seven and infinite hardness.
        collapsed = {lm.small_vanishes, lm.large_vanishes}
        cands = [
            Fraction(k, tuning)
            for k in range(1, tuning)
            if lo <= Fraction(k, tuning) <= hi and Fraction(k, tuning) not in collapsed
        ]
        if not cands:
            degenerate = [
                Fraction(k, tuning)
                for k in range(1, tuning)
                if Fraction(k, tuning) in collapsed
                and lo <= Fraction(k, tuning) <= hi
            ]
            extra = (
                f"; its only candidate {degenerate[0]} is the landmark where a "
                f"step size vanishes, which would collapse the scale"
                if degenerate else ""
            )
            raise ValueError(
                f"{tuning}-EDO contains no generator inside the valid range "
                f"({lo}, {hi}) of {n_large}L{n_small}s{extra}"
            )
        # Closest to the middle of the coherent region.
        c_lo, c_hi = T.coherence_range(n_large, n_small, bright=bright)
        target = (float(c_lo) + float(c_hi)) / 2.0
        return float(min(cands, key=lambda f: abs(float(f) - target)))
    return float(tuning)


@dataclass(frozen=True)
class MOSScale:
    """A moment-of-symmetry (well-formed) scale.

    Parameters
    ----------
    n_large, n_small : int
        Counts of large and small steps per period.  Always co-prime for a
        genuine MOS (Milne et al. §2, "Co-prime step numbers").
    generator : float
        The generator as a *fraction of the period*, in ``(0, 1)``.  Use
        :meth:`from_generator` to build one from a frequency ratio.
    period : float, default 2.0
        The period as a frequency ratio.  ``2.0`` is the octave; anything else
        is a pseudo-octave, which the labyrinth supports natively.
    validate : bool, default True
        Check that ``generator`` really does produce ``n_large L, n_small s``.
        Turn off only when constructing degenerate or deliberately mistuned
        scales.

    Examples
    --------
    The diatonic scale in Pythagorean tuning:

    >>> d = MOSScale.from_generator(3 / 2, 7)
    >>> d.signature
    '5L2s'
    >>> d.word
    'LLLsLLs'
    >>> [round(c, 1) for c in d.cents]
    [0.0, 203.9, 407.8, 611.7, 702.0, 905.9, 1109.8]

    Pythagorean tuning sits *outside* the diatonic's coherent range of
    ``4/7 .. 7/12`` (685.7 .. 700 cents), so it is improper -- its major third
    is wider than its diminished fourth:

    >>> round(d.hardness, 3), d.is_proper
    (2.26, False)

    Flatten the fifth into meantone and coherence returns:

    >>> m = MOSScale.from_signature(5, 2, tuning=31)
    >>> round(m.generator_cents, 2), round(m.hardness, 3), m.is_proper
    (696.77, 1.667, True)
    """

    n_large: int
    n_small: int
    generator: float
    period: float = 2.0
    validate: bool = field(default=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.n_large < 1 or self.n_small < 1:
            raise ValueError(
                f"both step counts must be >= 1, got "
                f"{self.n_large}L {self.n_small}s"
            )
        if math.gcd(self.n_large, self.n_small) != 1:
            raise ValueError(
                f"an MOS signature must be co-prime (Milne et al. §2); "
                f"{self.n_large}L{self.n_small}s has "
                f"gcd = {math.gcd(self.n_large, self.n_small)}"
            )
        if not 0.0 < self.generator < 1.0:
            raise ValueError(
                f"generator must be a period fraction in (0, 1), got "
                f"{self.generator!r}; use MOSScale.from_generator() for a "
                "frequency ratio"
            )
        if self.period <= 1.0:
            raise ValueError(f"period ratio must exceed 1, got {self.period!r}")
        if self.validate:
            got = T.mos_signature(self.generator, self.cardinality)
            if got != (self.n_large, self.n_small):
                lo, hi = self.tuning_range
                raise ValueError(
                    f"generator fraction {self.generator:.6f} produces "
                    f"{got[0]}L{got[1]}s at {self.cardinality} notes, not "
                    f"{self.n_large}L{self.n_small}s. Valid range for "
                    f"{self.n_large}L{self.n_small}s is ({lo}, {hi}) "
                    f"= ({float(lo):.6f}, {float(hi):.6f})."
                )

    # ---------------------------------------------------------------- #
    # Constructors
    # ---------------------------------------------------------------- #
    @classmethod
    def from_generator(
        cls, generator: float, cardinality: int, period: float = 2.0
    ) -> "MOSScale":
        """Build from a generator *frequency ratio* and a note count.

        Parameters
        ----------
        generator : float
            Frequency ratio, e.g. ``3/2``.  Reduced into the period.
        cardinality : int
            Must be one of the generator's MOS cardinalities; a helpful error
            lists the valid ones if it is not.
        period : float, default 2.0

        Examples
        --------
        >>> MOSScale.from_generator(3 / 2, 12).signature
        '5L7s'
        """
        g = generator_fraction(generator, period)
        return cls.from_fraction(g, cardinality, period)

    @classmethod
    def from_fraction(
        cls, g: float, cardinality: int, period: float = 2.0
    ) -> "MOSScale":
        """Build from a generator *period fraction* and a note count."""
        g = float(g) % 1.0
        n_large, n_small = T.mos_signature(g, cardinality)
        return cls(n_large, n_small, g, period, validate=False)

    @classmethod
    def from_signature(
        cls,
        n_large: int,
        n_small: int,
        tuning: TuningSpec = None,
        period: float = 2.0,
        bright: bool = True,
    ) -> "MOSScale":
        """Build from an abstract signature, choosing a tuning inside its range.

        Parameters
        ----------
        n_large, n_small : int
        tuning : float, int, Fraction or str, optional
            Where to sit inside the valid generator range:

            - ``None`` / ``'noble'`` (default) -- the φ-weighted mediant, the
              generator furthest from every equal temperament.
            - ``'central'`` -- middle of the *coherent* sub-range; always proper.
            - ``'middle'`` -- middle of the full valid range.
            - ``'equalized'`` -- the equal temperament where L and s coincide
              (degenerate, but sometimes what you want).
            - an ``int`` -- read the generator off that EDO.
            - a ``float`` -- an explicit generator fraction.
            - a ``Fraction`` -- an explicit rational generator.
        period : float, default 2.0
        bright : bool, default True
            Take the generator above ``1/2`` (the fifth rather than the fourth).

        Examples
        --------
        Stacking always starts from the root, so the scale as built is the
        brightest mode -- Lydian, not Ionian.  Use :meth:`mode` to rotate.

        >>> [round(c, 3) for c in MOSScale.from_signature(5, 2, tuning=12).cents]
        [0.0, 200.0, 400.0, 600.0, 700.0, 900.0, 1100.0]
        >>> round(MOSScale.from_signature(5, 2, tuning=31).generator_cents, 2)
        696.77
        """
        g = _resolve_tuning(n_large, n_small, tuning, bright)
        return cls(n_large, n_small, g, period, validate=False)

    @classmethod
    def from_edo(
        cls, edo: int, steps: int, cardinality: int, period: float = 2.0
    ) -> "MOSScale":
        """Build from a generator of ``steps`` degrees of ``edo``-EDO.

        Examples
        --------
        >>> MOSScale.from_edo(31, 18, 7).signature      # 31-EDO meantone diatonic
        '5L2s'
        """
        return cls.from_fraction(Fraction(steps, edo), cardinality, period)

    # ---------------------------------------------------------------- #
    # Identity
    # ---------------------------------------------------------------- #
    @property
    def cardinality(self) -> int:
        """Total notes per period."""
        return self.n_large + self.n_small

    @property
    def signature(self) -> str:
        """Compact signature, e.g. ``'5L2s'``."""
        return f"{self.n_large}L{self.n_small}s"

    @property
    def is_bright(self) -> bool:
        """True when the generator lies above half the period."""
        return self.generator > 0.5

    @property
    def is_degenerate(self) -> bool:
        """True when large and small steps have collapsed to the same size."""
        large, small = T.step_sizes(self.generator, self.cardinality)
        return math.isclose(large, small, rel_tol=1e-9, abs_tol=1e-12)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        per = "octave" if math.isclose(self.period, 2.0) else f"{self.period:.4g}"
        return (
            f"MOSScale({self.signature}, generator={self.generator_cents:.2f}c, "
            f"period={per}, R={self.hardness:.3f}"
            f"{', proper' if self.is_proper else ', improper'})"
        )

    # ---------------------------------------------------------------- #
    # Tuning
    # ---------------------------------------------------------------- #
    @property
    def period_cents(self) -> float:
        """Size of the period in cents (1200 for an octave)."""
        return PERIOD_CENTS * math.log2(self.period)

    @property
    def generator_ratio(self) -> float:
        """The generator as a frequency ratio."""
        return self.period**self.generator

    @property
    def generator_cents(self) -> float:
        """The generator in cents."""
        return self.generator * self.period_cents

    @property
    def degrees(self) -> List[float]:
        """Scale degrees as period fractions in ``[0, 1)``, ascending."""
        return T.degrees_from_generator(self.generator, self.cardinality)

    @property
    def ratios(self) -> List[float]:
        """Scale degrees as frequency ratios in ``[1, period)``, ascending."""
        return [self.period**d for d in self.degrees]

    @property
    def cents(self) -> List[float]:
        """Scale degrees in cents, ascending, starting at 0."""
        pc = self.period_cents
        return [d * pc for d in self.degrees]

    @property
    def word(self) -> str:
        """Step pattern of the scale as tuned, e.g. ``'LLLsLLs'``.

        Read off the actual stacked scale, so it is the rotation rooted on the
        generator chain's origin -- not necessarily the Christoffel word (see
        :func:`~biotuner.mos.theory.mos_word`), and not necessarily the
        brightest mode either.  The chain origin is the brightest mode only
        when stacking darkens; see
        :func:`~biotuner.mos.modes.stacking_brightens`.  For the brightest
        mode's pattern, use ``scale.mode(0).word``.
        """
        return T.word_from_generator(self.generator, self.cardinality)

    @property
    def step_cents(self) -> Tuple[float, float]:
        """``(large, small)`` step sizes in cents."""
        large, small = T.step_sizes(self.generator, self.cardinality)
        pc = self.period_cents
        return large * pc, small * pc

    @property
    def hardness(self) -> float:
        """Blackwood's ``R = L / s`` -- how uneven the two steps are.

        ``1`` at the equalized tuning, ``2`` at the embedding EDO (the edge of
        propriety), and unbounded as the small step vanishes.
        """
        large, small = T.step_sizes(self.generator, self.cardinality)
        if small <= 0:
            return float("inf")
        return large / small

    # ---------------------------------------------------------------- #
    # Position in the labyrinth
    # ---------------------------------------------------------------- #
    @property
    def landmarks(self) -> Landmarks:
        """The three landmark equal temperaments bounding this MOS pair."""
        return T.mos_landmarks(self.n_large, self.n_small, bright=self.is_bright)

    @property
    def tuning_range(self) -> Tuple[Fraction, Fraction]:
        """Generator fractions between which the scale keeps its identity."""
        return T.signature_ranges(self.n_large, self.n_small)[
            1 if self.is_bright else 0
        ]

    @property
    def coherence_range(self) -> Tuple[Fraction, Fraction]:
        """Generator fractions over which the scale is coherent (``R < 2``)."""
        return T.coherence_range(self.n_large, self.n_small, bright=self.is_bright)

    @property
    def is_proper(self) -> bool:
        """True when the scale is coherent: every generic interval well-ordered.

        Equivalent to ``hardness <= 2`` for a well-formed scale (Milne et al.
        §2); :func:`biotuner.mos.metrics.is_proper` verifies it directly from
        the interval matrix instead, and the two agree.
        """
        return self.hardness <= 2.0 + 1e-12

    @property
    def sb_node(self) -> Optional[T.SBNode]:
        """This scale's node on the Stern-Brocot path of its generator."""
        return T.sb_node_at(self.generator, self.cardinality)

    @property
    def edo(self) -> Optional[int]:
        """The EDO this scale *is*, when the generator is exactly rational.

        ``None`` for a generic tuning.  The denominator cap matters: allowed to
        run to a million, ``limit_denominator`` approximates an irrational
        generator to within 1e-12 and every scale looks like some absurd equal
        division.  Ten thousand keeps genuine EDOs exact while leaving an
        irrational generator visibly irrational.
        """
        f = Fraction(self.generator).limit_denominator(10**4)
        if abs(float(f) - self.generator) < 1e-13:
            return f.denominator
        return None

    # ---------------------------------------------------------------- #
    # Relatives
    # ---------------------------------------------------------------- #
    @property
    def inverse(self) -> "MOSScale":
        """The scale with large and small steps swapped (Milne et al. §2).

        Realised at the mirror-image generator across the equalized landmark,
        keeping the same distance from it -- so the diatonic's inverse is an
        anti-diatonic just as far from 7-EDO as the original.

        Examples
        --------
        >>> MOSScale.from_generator(3 / 2, 7).inverse.signature
        '2L5s'
        """
        eq = float(self.landmarks.equalized)
        g = 2.0 * eq - self.generator
        return MOSScale(self.n_small, self.n_large, g, self.period, validate=False)

    @property
    def parent(self) -> Optional["MOSScale"]:
        """The next *smaller* MOS in this generator's family, or ``None``."""
        cards = T.mos_cardinalities(
            self.generator, self.cardinality, include_trivial=True
        )
        smaller = [c for c in cards if c < self.cardinality]
        if not smaller:
            return None
        return MOSScale.from_fraction(self.generator, smaller[-1], self.period)

    def child(self, steps: int = 1) -> Optional["MOSScale"]:
        """The ``steps``-th *larger* MOS in this generator's family."""
        cards = T.mos_cardinalities(
            self.generator, self.cardinality * 8 + 16, include_trivial=True
        )
        bigger = [c for c in cards if c > self.cardinality]
        if len(bigger) < steps:
            return None
        return MOSScale.from_fraction(self.generator, bigger[steps - 1], self.period)

    @property
    def embedding(self) -> Tuple[int, Fraction]:
        """``(cardinality, tuning)`` of the lowest-cardinality embedding scale.

        Examples
        --------
        >>> MOSScale.from_generator(3 / 2, 7).embedding
        (12, Fraction(7, 12))
        """
        return T.embedding(self.n_large, self.n_small, bright=self.is_bright)

    def family(self, max_cardinality: int = 53) -> List["MOSScale"]:
        """Every MOS this generator produces, smallest first.

        Examples
        --------
        >>> [s.signature for s in MOSScale.from_generator(3 / 2, 7).family(17)]
        ['2L1s', '2L3s', '5L2s', '5L7s', '12L5s']
        """
        return mos_family(
            self.generator_ratio, max_cardinality=max_cardinality, period=self.period
        )

    def retune(self, tuning: TuningSpec) -> "MOSScale":
        """Same structure, different generator inside the valid range.

        Examples
        --------
        >>> round(MOSScale.from_generator(3 / 2, 7).retune(19).generator_cents, 4)
        694.7368
        """
        g = _resolve_tuning(self.n_large, self.n_small, tuning, self.is_bright)
        return MOSScale(
            self.n_large, self.n_small, g, self.period, validate=False
        )

    # ---------------------------------------------------------------- #
    # Modes
    # ---------------------------------------------------------------- #
    def mode(self, index: int = 0) -> "Mode":
        """The ``index``-th mode, ordered brightest (0) to darkest."""
        from biotuner.mos.modes import Mode

        return Mode(self, index)

    def modes(self) -> List["Mode"]:
        """All ``cardinality`` modes, brightest first."""
        return [self.mode(i) for i in range(self.cardinality)]

    # ---------------------------------------------------------------- #
    # Interop
    # ---------------------------------------------------------------- #
    def to_scala(self, name: Optional[str] = None, write: bool = False) -> str:
        """Render as a Scala ``.scl`` file body.

        Uses biotuner's own :func:`~biotuner.biotuner_utils.create_SCL`, so the
        output matches every other tuning the toolbox exports.
        """
        from biotuner.biotuner_utils import create_SCL

        scale = list(self.ratios) + [self.period]
        return create_SCL(scale, name or self.signature, write=write)

    def to_dict(self) -> Dict[str, object]:
        """Flat, JSON-friendly summary -- handy for tables and provenance."""
        lm = self.landmarks
        lo, hi = self.tuning_range
        c_lo, c_hi = self.coherence_range
        large_c, small_c = self.step_cents
        return {
            "signature": self.signature,
            "n_large": self.n_large,
            "n_small": self.n_small,
            "cardinality": self.cardinality,
            "generator": self.generator,
            "generator_cents": self.generator_cents,
            "generator_ratio": self.generator_ratio,
            "period": self.period,
            "period_cents": self.period_cents,
            "word": self.word,
            "step_large_cents": large_c,
            "step_small_cents": small_c,
            "hardness": self.hardness,
            "is_proper": self.is_proper,
            "is_degenerate": self.is_degenerate,
            "tuning_range": (str(lo), str(hi)),
            "coherence_range": (str(c_lo), str(c_hi)),
            "equalized_edo": lm.equalized_edo,
            "small_vanishes_edo": lm.small_vanishes_edo,
            "large_vanishes_edo": lm.large_vanishes_edo,
            "embedding_cardinality": self.embedding[0],
        }

    def summary(self) -> str:
        """Multi-line human-readable description."""
        lm = self.landmarks
        lo, hi = self.tuning_range
        c_lo, c_hi = self.coherence_range
        large_c, small_c = self.step_cents
        emb_n, emb_t = self.embedding
        pc = self.period_cents
        lines = [
            f"{self.signature}  ({self.cardinality} notes)   {self.word}",
            f"  generator      {self.generator_cents:8.3f} c  "
            f"(ratio {self.generator_ratio:.6f}, g = {self.generator:.6f})",
            f"  period         {pc:8.3f} c  (ratio {self.period:.6f})",
            f"  steps          L = {large_c:.3f} c,  s = {small_c:.3f} c,  "
            f"R = {self.hardness:.4f}",
            f"  valid range    {lo} .. {hi}   "
            f"({float(lo) * pc:.1f} .. {float(hi) * pc:.1f} c)",
            f"  coherent over  {c_lo} .. {c_hi}   "
            f"-> {'proper' if self.is_proper else 'IMPROPER at this tuning'}",
            f"  landmarks      equalized {lm.equalized} = {lm.equalized_edo}-EDO,  "
            f"s->0 at {lm.small_vanishes} = {lm.small_vanishes_edo}-EDO,  "
            f"L->0 at {lm.large_vanishes} = {lm.large_vanishes_edo}-EDO",
            f"  inverse        {self.inverse.signature}",
            f"  embedded in    {emb_n} notes (equal at {emb_t})",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Module-level conveniences
# --------------------------------------------------------------------------- #
def mos(
    generator: float, cardinality: int, period: float = 2.0
) -> MOSScale:
    """Shorthand for :meth:`MOSScale.from_generator`.

    Examples
    --------
    >>> mos(3 / 2, 7).signature
    '5L2s'
    """
    return MOSScale.from_generator(generator, cardinality, period)


def mos_family(
    generator: float,
    max_cardinality: int = 53,
    period: float = 2.0,
    min_cardinality: int = 3,
) -> List[MOSScale]:
    """Every MOS scale a generator produces, smallest first.

    This is the corrected replacement for
    :func:`biotuner.scale_construction.find_MOS`: exact rather than
    brute-forced, honouring ``period`` throughout, and returning scale objects
    instead of a dict of parallel lists.

    Parameters
    ----------
    generator : float
        Generator as a frequency ratio.
    max_cardinality : int, default 53
    period : float, default 2.0
    min_cardinality : int, default 3
        Skip the musically empty 2-note MOS by default.

    Examples
    --------
    >>> [s.signature for s in mos_family(3 / 2, 12)]
    ['2L1s', '2L3s', '5L2s', '5L7s']
    >>> [s.cardinality for s in mos_family(2 ** (316 / 1200), 19)]
    [3, 4, 7, 11, 15, 19]
    """
    g = generator_fraction(generator, period)
    return [
        MOSScale(n_large, n_small, g, period, validate=False)
        for card, n_large, n_small in T.mos_series(
            g, max_cardinality=max_cardinality, include_trivial=True
        )
        if card >= min_cardinality
    ]
