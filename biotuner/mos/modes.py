"""Modes of a well-formed scale, and the ℤ² lattice they live in.

A scale presupposes periodicity, so all its rotations are the same scale.  A
*mode* is what you get by choosing a fundamental domain for that period -- a
finalis.  Milne et al. (2011) §4 describe the modal universe of a well-formed
scale as "freely generated from two basic and commuting transformations and
… therefore isomorphic to the free commutative group ℤ² of rank 2":

``σ`` (:meth:`Mode.rotate`)
    *Common origin.*  Keeps the pitch collection, moves the finalis up one
    scale degree.  C-Ionian → D-Dorian.

``τ`` (:meth:`Mode.brighten`)
    *Common finalis.*  Keeps the finalis, moves the origin one generator
    sharpwards.  C-Ionian → C-Lydian.

Adjacent modes in the brightness order are *parsimonious*: they differ by a
single tone, displaced by the augmented prime (the chroma, ``L - s``).  That
claim is checked directly in the test suite.

The lattice itself (Milne et al. Fig. 7) is exposed by
:meth:`Mode.lattice_coords`, which places each scale degree at integer
coordinates in the (generator, period) basis.  The zig-zag those points trace
out is the mode's fundamental frame; ``σ`` and ``τ`` are its vertical and
horizontal shifts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

from biotuner.mos.theory import PERIOD_CENTS, christoffel_word

if TYPE_CHECKING:  # pragma: no cover
    from biotuner.mos.scale import MOSScale

__all__ = [
    "Mode",
    "DIATONIC_MODE_NAMES",
    "mode_names",
    "wf_number",
    "chain_order",
    "mode_lattice",
    "christoffel_mode",
    "parsimony_chain",
]

#: The seven church modes, brightest first -- the order ``τ`` walks.
DIATONIC_MODE_NAMES: Tuple[str, ...] = (
    "Lydian",
    "Ionian",
    "Mixolydian",
    "Dorian",
    "Aeolian",
    "Phrygian",
    "Locrian",
)

#: Signatures for which conventional mode names exist.  Everything else gets
#: a generic ``"mode k of 4L3s"`` label rather than an invented name.
_NAMED_SIGNATURES: Dict[Tuple[int, int], Tuple[str, ...]] = {
    (5, 2): DIATONIC_MODE_NAMES,
}


def chain_order(g: float, cardinality: int) -> List[int]:
    """Sorted-degree index of each generator-chain position.

    ``chain_order(g, N)[i]`` is where the ``i``-th stacked generator lands once
    the scale is sorted.  The inverse mapping -- which chain position each
    sorted degree came from -- is what :func:`wf_number` reads.

    Examples
    --------
    Stacking fifths gives C G D A E B F♯; sorted, those are degrees
    0 4 1 5 2 6 3:

    >>> chain_order(math.log2(3 / 2), 7)
    [0, 4, 1, 5, 2, 6, 3]
    """
    pitches = [(i * g) % 1.0 for i in range(cardinality)]
    order = sorted(range(cardinality), key=lambda i: pitches[i])
    rank = [0] * cardinality
    for sorted_pos, chain_pos in enumerate(order):
        rank[chain_pos] = sorted_pos
    return rank


def wf_number(g: float, cardinality: int) -> int:
    """Carey's ``g`` in ``WF(N, g)`` -- generator order → scale step order.

    The factor that converts generator order into scale step order, mod ``N``
    (Carey 1998; quoted in Milne et al. §1).  Equivalently: taking one step up
    the scale advances you ``wf_number`` places along the generator chain.

    Examples
    --------
    The diatonic scale and its inverse belong to ``WF(7, 2)``, the chromatic
    scale to ``WF(12, 7)``:

    >>> wf_number(math.log2(3 / 2), 7)
    2
    >>> wf_number(math.log2(3 / 2), 12)
    7
    """
    pitches = [(i * g) % 1.0 for i in range(cardinality)]
    order = sorted(range(cardinality), key=lambda i: pitches[i])
    return order[1] % cardinality


def mode_names(n_large: int, n_small: int) -> Optional[Tuple[str, ...]]:
    """Conventional mode names for a signature, brightest first, or ``None``."""
    return _NAMED_SIGNATURES.get((n_large, n_small))


def stacking_brightens(scale: "MOSScale") -> bool:
    """True when stacking the generator upward yields successively *brighter* modes.

    A generator and its complement build the same scale but walk its modes in
    opposite directions, and which of the two a given ``MOSScale`` holds is not
    determined by whether it exceeds half the period -- the two coincide for
    ``5L2s`` and disagree for its inverse ``2L5s``.  So the direction is
    measured rather than assumed: compare the brightness of the mode on the
    root with the mode one generator up.

    Everything that indexes modes by brightness goes through this, which is
    what keeps ``mode(0)`` the brightest mode for *every* signature.

    Examples
    --------
    >>> from biotuner.mos.scale import MOSScale
    >>> stacking_brightens(MOSScale.from_signature(5, 2, tuning=12))
    False
    >>> stacking_brightens(MOSScale.from_signature(2, 5, tuning='central'))
    True
    """
    degrees = scale.degrees
    root = scale.generator % 1.0
    at_root = sum(degrees)
    at_generator = sum((d - root) % 1.0 for d in degrees)
    return at_generator > at_root


@dataclass(frozen=True)
class Mode:
    """One mode of a :class:`~biotuner.mos.scale.MOSScale`.

    Parameters
    ----------
    scale : MOSScale
    index : int
        Brightness rank, ``0`` = brightest.  Mode ``k`` is rooted on the
        ``k``-th note of the generator chain, so each increment darkens the
        scale by one generator.

    Examples
    --------
    >>> from biotuner.mos.scale import MOSScale
    >>> d = MOSScale.from_signature(5, 2, tuning=12)
    >>> [m.name for m in d.modes()]
    ['Lydian', 'Ionian', 'Mixolydian', 'Dorian', 'Aeolian', 'Phrygian', 'Locrian']
    >>> d.mode(1).word
    'LLsLLLs'
    >>> [round(c) for c in d.mode(1).cents]
    [0, 200, 400, 500, 700, 900, 1100]
    """

    scale: "MOSScale"
    index: int

    def __post_init__(self) -> None:
        n = self.scale.cardinality
        if not 0 <= self.index < n:
            raise ValueError(
                f"mode index must lie in [0, {n}) for a {n}-note scale, "
                f"got {self.index}"
            )

    # ---------------------------------------------------------------- #
    # Identity
    # ---------------------------------------------------------------- #
    @property
    def cardinality(self) -> int:
        return self.scale.cardinality

    @property
    def name(self) -> str:
        """Conventional name where one exists, else ``'mode k of 4L3s'``."""
        names = mode_names(self.scale.n_large, self.scale.n_small)
        if names is not None and self.index < len(names):
            return names[self.index]
        return f"mode {self.index} of {self.scale.signature}"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Mode({self.name}, {self.word})"

    # ---------------------------------------------------------------- #
    # Pitches
    # ---------------------------------------------------------------- #
    @property
    def chain_position(self) -> int:
        """How many generators up the chain this mode's finalis sits.

        Brightness indices run brightest-first, but the generator chain runs
        that way only when stacking *darkens* (see :func:`stacking_brightens`);
        otherwise the chain is walked in reverse.
        """
        if stacking_brightens(self.scale):
            return self.cardinality - 1 - self.index
        return self.index

    @classmethod
    def _from_chain(cls, scale: "MOSScale", chain: int) -> "Mode":
        """Build the mode whose finalis is ``chain`` generators up."""
        n = scale.cardinality
        chain %= n
        index = (n - 1 - chain) if stacking_brightens(scale) else chain
        return cls(scale, index % n)

    @property
    def root_degree(self) -> float:
        """Position of this mode's finalis within the parent scale, in ``[0, 1)``."""
        return (self.chain_position * self.scale.generator) % 1.0

    @property
    def degrees(self) -> List[float]:
        """Degrees as period fractions in ``[0, 1)``, rooted at 0."""
        root = self.root_degree
        return sorted((d - root) % 1.0 for d in self.scale.degrees)

    @property
    def ratios(self) -> List[float]:
        """Degrees as frequency ratios in ``[1, period)``."""
        return [self.scale.period**d for d in self.degrees]

    @property
    def cents(self) -> List[float]:
        pc = self.scale.period_cents
        return [d * pc for d in self.degrees]

    @property
    def word(self) -> str:
        """Step pattern of this mode, e.g. ``'LLsLLLs'`` for Ionian."""
        degs = self.degrees + [1.0]
        steps = [degs[i + 1] - degs[i] for i in range(self.cardinality)]
        lo, hi = min(steps), max(steps)
        if math.isclose(lo, hi, rel_tol=1e-9, abs_tol=1e-12):
            return "L" * self.cardinality
        mid = (lo + hi) / 2.0
        return "".join("L" if s > mid else "s" for s in steps)

    @property
    def brightness(self) -> float:
        """Sum of the mode's degrees -- higher is brighter.

        Strictly decreasing in :attr:`index`, which is what makes the index a
        brightness rank.
        """
        return float(sum(self.degrees))

    @property
    def chroma(self) -> float:
        """The augmented prime ``L - s`` in cents.

        The interval by which a single tone moves when stepping between
        adjacent modes (Milne et al. §4).
        """
        large, small = self.scale.step_cents
        return large - small

    # ---------------------------------------------------------------- #
    # The two transformations
    # ---------------------------------------------------------------- #
    def brighten(self, k: int = 1) -> "Mode":
        """``τ^-k``: same finalis, origin ``k`` generators sharpwards.

        Chromatic transposition.  Brightening by one is the paper's
        "transformation of a C-Ionian mode into the common finalis mode
        C-Lydian".

        Examples
        --------
        >>> from biotuner.mos.scale import MOSScale
        >>> MOSScale.from_signature(5, 2, tuning=12).mode(1).brighten().name
        'Lydian'
        """
        return Mode(self.scale, (self.index - k) % self.cardinality)

    def darken(self, k: int = 1) -> "Mode":
        """``τ^k``: same finalis, origin ``k`` generators flatwards."""
        return self.brighten(-k)

    def rotate(self, k: int = 1) -> "Mode":
        """``σ^k``: same pitch collection, finalis ``k`` scale steps higher.

        Diatonic transposition.  Rotating C-Ionian by one gives D-Dorian --
        the same white keys, a new home.

        Examples
        --------
        >>> from biotuner.mos.scale import MOSScale
        >>> MOSScale.from_signature(5, 2, tuning=12).mode(1).rotate().name
        'Dorian'
        """
        # One scale step advances a fixed number of places along the generator
        # chain -- Carey's g -- so the shift is applied in chain space.
        wf = wf_number(self.scale.generator, self.cardinality)
        return Mode._from_chain(self.scale, self.chain_position + k * wf)

    # ---------------------------------------------------------------- #
    # The ℤ² lattice (Milne et al. Fig. 7)
    # ---------------------------------------------------------------- #
    def lattice_coords(self) -> List[Tuple[int, int]]:
        """``(width, height)`` lattice coordinates of each degree.

        Every pitch reachable by stacking is ``height`` periods plus ``width``
        generators away from the *finalis*.  Listing them in generator-chain
        order traces the zig-zag that Milne et al. Fig. 7 calls the mode's
        *fundamental frame*: the width axis is spanned by the augmented prime,
        the height axis by the pseudo-octave.

        Widths run negative for the notes below the finalis in the chain, which
        is exactly what distinguishes one mode's frame from another's -- Lydian
        sits entirely above its finalis, Locrian entirely below.

        Examples
        --------
        >>> from biotuner.mos.scale import MOSScale
        >>> d = MOSScale.from_signature(5, 2, tuning=12)
        >>> d.mode(0).lattice_coords()          # Lydian: all above
        [(0, 0), (1, 0), (2, -1), (3, -1), (4, -2), (5, -2), (6, -3)]
        >>> d.mode(6).lattice_coords()          # Locrian: all below
        [(-6, 4), (-5, 3), (-4, 3), (-3, 2), (-2, 2), (-1, 1), (0, 0)]
        """
        g = self.scale.generator
        chain = self.chain_position
        out = []
        for j in range(self.cardinality):
            # Offset of chain position j from this mode's own finalis.
            width = j - chain
            out.append((width, -math.floor(width * g)))
        return out

    # ---------------------------------------------------------------- #
    # Parsimony
    # ---------------------------------------------------------------- #
    def differences(
        self, other: "Mode", tol: float = 1e-6
    ) -> List[Tuple[int, float, float]]:
        """Scale steps at which this mode and ``other`` disagree.

        Both modes have the same number of degrees, so they are compared step
        by step: entry ``(k, mine, theirs)`` means the ``k``-th degree sits at
        ``mine`` cents here and ``theirs`` cents in ``other``.  Pairing by step
        rather than by proximity matters -- when a tone moves by the chroma it
        lands exactly halfway between its neighbours, and a nearest-value match
        would tie and break arbitrarily on floating-point noise.

        For adjacent modes in the brightness order the result has length 1 and
        the two values differ by exactly the chroma: the parsimony property of
        Milne et al. §4.

        Parameters
        ----------
        other : Mode
        tol : float, default 1e-6
            Cents below which two degrees count as the same tone.
        """
        mine, theirs = self.cents, other.cents
        return [
            (k, a, b)
            for k, (a, b) in enumerate(zip(mine, theirs))
            if abs(a - b) > tol
        ]

    # ---------------------------------------------------------------- #
    # Interop
    # ---------------------------------------------------------------- #
    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "index": self.index,
            "signature": self.scale.signature,
            "word": self.word,
            "cents": self.cents,
            "ratios": self.ratios,
            "brightness": self.brightness,
            "chroma_cents": self.chroma,
        }


def christoffel_mode(scale: "MOSScale") -> Mode:
    """The mode whose step pattern is the signature's Christoffel word.

    Of the ``N`` modes, this is the one that matters for Fourier Scratching.
    Milne et al. §5 claim that a coherent well-formed scale is "played in
    generic scalar order by the first partial play state" -- ``n`` evenly
    spaced fingers striking a keyboard whose keys are as wide as the steps
    above their tones, each key caught exactly once.

    That is true, but it is a property of a *mode*, not of the scale: the
    Christoffel word is by construction the floor-quantisation of the equal
    division of the period, so it is precisely the rotation whose key
    boundaries interleave with evenly spaced fingers.  In any other mode two
    fingers share a key and another key is missed.  Coherence is what makes
    that mode exist; the mode is what makes the claim hold.

    Raises
    ------
    ValueError
        If the scale is degenerate, where every mode has the same all-``L``
        word and the Christoffel word does not single one out.

    Examples
    --------
    For the diatonic that mode is Locrian, not the brightest mode:

    >>> from biotuner.mos.scale import MOSScale
    >>> m = christoffel_mode(MOSScale.from_signature(5, 2, tuning=12))
    >>> m.name, m.word
    ('Locrian', 'sLLsLLL')
    """
    word = christoffel_word(scale.n_large, scale.n_small)
    for mode in scale.modes():
        if mode.word == word:
            return mode
    raise ValueError(
        f"{scale.signature} at {scale.generator_cents:.3f} c has no mode with "
        f"the Christoffel word {word!r} (modes: "
        f"{sorted({m.word for m in scale.modes()})}). This happens for a "
        f"degenerate tuning, where the two step sizes coincide."
    )


def mode_lattice(
    scale: "MOSScale", width: int = 3, height: int = 3, base: int = 0
) -> List[List[Mode]]:
    """A ``height × width`` patch of the modal ℤ² lattice.

    Row ``j``, column ``i`` holds ``σ^j τ^i`` applied to mode ``base``: moving
    right darkens by one generator (chromatic transposition), moving down
    advances the finalis by one scale step (diatonic transposition).  The two
    transformations commute, so the patch reads the same either way -- which
    the test suite verifies.

    Examples
    --------
    >>> from biotuner.mos.scale import MOSScale
    >>> grid = mode_lattice(MOSScale.from_signature(5, 2, tuning=12), 3, 2)
    >>> [[m.name for m in row] for row in grid]
    [['Lydian', 'Ionian', 'Mixolydian'], ['Mixolydian', 'Dorian', 'Aeolian']]
    """
    out = []
    for j in range(height):
        row = []
        for i in range(width):
            row.append(Mode(scale, base).brighten(-i).rotate(j))
        out.append(row)
    return out


def parsimony_chain(
    scale: "MOSScale",
) -> List[Tuple[Mode, Mode, List[Tuple[int, float, float]]]]:
    """Walk the brightness order, reporting what moves at each step.

    Each entry is ``(brighter, darker, moved)``, where ``moved`` comes from
    :meth:`Mode.differences`.  For a well-formed scale every step moves exactly
    one tone, by exactly the chroma -- and which tone moves is different every
    time, so the whole modal universe is reachable one note at a time.

    Examples
    --------
    >>> from biotuner.mos.scale import MOSScale
    >>> chain = parsimony_chain(MOSScale.from_signature(5, 2, tuning=12))
    >>> all(len(moved) == 1 for _, _, moved in chain)
    True
    >>> [round(a - b) for _, _, moved in chain for _, a, b in moved]
    [100, 100, 100, 100, 100, 100]
    >>> [k for _, _, moved in chain for k, _, _ in moved]
    [3, 6, 2, 5, 1, 4]
    """
    modes = scale.modes()
    return [
        (modes[i], modes[i + 1], modes[i].differences(modes[i + 1]))
        for i in range(len(modes) - 1)
    ]
