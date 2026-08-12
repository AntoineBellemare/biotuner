"""Named rank-2 temperaments, with every generator derived from its comma.

Milne et al. §4 overlay the interactive scale labyrinth with named regular
temperaments -- meantone, srutal, magic, hanson, ... -- drawn as radial lines
whose angle is that temperament's optimal generator/period ratio.  This module
supplies those angles.

Nothing here is a remembered table of cents.  A temperament is entered as the
comma it tempers out (81/80 for meantone) and the rest is computed:

* the **mapping** -- how many periods and generators each prime is worth -- is a
  *saturated* integer basis of the comma's annihilator lattice, put into Hermite
  normal form so that it is canonical and not merely correct;
* the **generator** is the Tenney-weighted least-squares fit to the just primes,
  with the period held exactly pure.

Because the numbers are derived, the module checks itself: ``mapping @ monzo``
is exactly ``(0, 0)`` in integer arithmetic for every named comma, and
``periods_per_octave`` is read off the mapping rather than declared.

Saturation matters
------------------
The annihilator of a comma is obtained here by an iterative extended-Euclid on
the monzo's entries, each step a unimodular ``2 x 2`` row operation on an
accumulating matrix ``U``.  When ``U @ m == (gcd, 0, ..., 0)``, rows ``1..`` of
``U`` span *exactly* ``{v : v . m == 0}`` -- every integer val that tempers the
comma out, not just a finite-index sublattice of them.  A rational nullspace
with denominators cleared afterwards can silently land on such a sublattice
(e.g. give ``2`` periods per octave where the temperament really has ``1``), and
then every downstream cent value is wrong in a way that still looks plausible.

Tuning convention
-----------------
Three choices are free in a rank-2 mapping and are pinned down as follows.

1. *Hermite normal form.*  Pivots are positive, the second row is cleared out of
   the octave column (``mapping[1][0] == 0``, so the period is a pure fraction
   of the octave), and the first row is reduced modulo the second's pivot.  For
   meantone this yields the familiar ``[[1, 0, -4], [0, 1, 4]]`` -- prime 5 is
   four generators up -- available as :attr:`Rank2Temperament.hermite_mapping`.

2. *Sign.*  ``mapping[1]`` and its negation describe the same temperament with
   generators running in opposite directions.  The sign is fixed by requiring
   the unreduced generator to be a **positive** interval.  This is what makes
   porcupine come out as ~163 c rather than ~-163 c.

3. *Octave reduction.*  The generator is then folded into ``[0, period)`` by
   adding multiples of ``mapping[1]`` to ``mapping[0]``, which leaves
   ``periods_per_octave`` untouched.  Meantone's twelfth becomes its fifth and
   the stored mapping becomes ``[[1, 1, 0], [0, 1, 4]]``, so that
   ``prime_errors`` is consistent with the generator actually reported.

Octave-locked
-------------
Every quantity here is measured in octaves and then scaled by
``theory.PERIOD_CENTS``: the period is ``1 / periods_per_octave`` octaves, the
period ratio is ``2 ** (1 / periods_per_octave)``, and each prime's error is
measured against ``log2(p)``.  That is only coherent when the first prime *is*
2, so :class:`Rank2Temperament` refuses any other basis.  A no-twos subgroup
such as ``(3, 5, 7)`` would otherwise report a 1200-cent period for an equave
of 3, a 702-cent error on its own equave, and a generator with no meaning --
all without raising.  Supporting one would mean replacing those octave
constants with the equave, not relabelling them.

A generator ``g`` and its complement ``period - g`` build precisely the same
scales (Milne et al. §4), so steps 2--3 are cosmetic: they choose which of two
mirror-image radial lines gets drawn.  Published tables sometimes choose the
other one: father and bug, among others, are normally quoted as this
convention's *complement*.  Hence :attr:`Rank2Temperament.complement_cents`,
and hence :func:`nearest_temperaments` matching against both.

Which optimum
-------------
The period is held exactly pure at ``1 / periods_per_octave`` octaves and the
generator minimises the Tenney-weighted squared error of the remaining primes.
That is the *constrained* Tenney-Euclidean tuning (CTE).  It is not the same as
the POTE tuning quoted on the Xenharmonic wiki, which optimises period and
generator together and *then* rescales the result to a pure octave; the two
agree to a fraction of a cent for accurate temperaments and diverge by several
cents for wildly inaccurate ones.  Both are available --
:attr:`Rank2Temperament.generator_cents` is CTE,
:attr:`Rank2Temperament.pote_generator_cents` is POTE.

References
----------
Milne, A.J., Carlé, M., Sethares, W.A., Noll, T., Holland, S. (2011).
Scratching the Scale Labyrinth. In *Mathematics and Computation in Music*,
LNAI 6726, 180--195.

Smith, G.W. / Breed, G.  Regular temperament theory: vals, monzos, Hermite
normal form and Tenney-Euclidean tunings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

from biotuner.mos.theory import PERIOD_CENTS

if TYPE_CHECKING:  # pragma: no cover
    from biotuner.mos.scale import MOSScale

__all__ = [
    "PRIMES_5",
    "PRIMES_7",
    "monzo",
    "saturated_annihilator",
    "hermite_normal_form",
    "Rank2Temperament",
    "rank2_from_comma",
    "TEMPERAMENTS",
    "temperament",
    "all_temperaments",
    "nearest_temperaments",
]

#: The 5-limit primes -- the plane the classical comma names live in.
PRIMES_5: Tuple[int, ...] = (2, 3, 5)

#: The 7-limit primes.  A rank-2 temperament here needs *two* commas; pass the
#: second through ``rank2_from_comma(..., extra_commas=[...])``.
PRIMES_7: Tuple[int, ...] = (2, 3, 5, 7)

RatioLike = Union[int, Fraction, str]


# --------------------------------------------------------------------------- #
# Prime bases
# --------------------------------------------------------------------------- #
def _is_prime(n: int) -> bool:
    """Trial division; the bases in play here are tiny."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    f = 3
    while f * f <= n:
        if n % f == 0:
            return False
        f += 2
    return True


def _check_prime_basis(primes: Sequence[int]) -> Tuple[int, ...]:
    """Validate a prime basis and return it as a tuple of ``int``.

    A monzo is only well defined over a strictly increasing sequence of genuine
    primes.  Over ``(2, 3, 3)`` the factorisation loop puts every factor of 3
    into the first slot and reports the second as zero; over ``(2, 4)`` the
    exponents are not unique at all.  Either way the resulting "temperament"
    is self-consistent nonsense, so both are rejected here rather than later.
    """
    ps = tuple(int(p) for p in primes)
    if not ps:
        raise ValueError("the prime basis is empty; at least one prime is needed")
    for i, p in enumerate(ps):
        if not _is_prime(p):
            raise ValueError(
                f"the prime basis {ps} contains {p}, which is not prime; "
                "a monzo over a composite basis is not unique"
            )
        if i and p <= ps[i - 1]:
            raise ValueError(
                f"the prime basis {ps} must be strictly increasing; "
                f"{p} does not exceed {ps[i - 1]}"
            )
    return ps


# --------------------------------------------------------------------------- #
# Monzos
# --------------------------------------------------------------------------- #
def monzo(ratio: RatioLike, primes: Sequence[int] = PRIMES_5) -> Tuple[int, ...]:
    """Prime-exponent vector of a rational interval.

    Parameters
    ----------
    ratio : int, Fraction or str
        The interval, exactly.  Floats are refused: ``81 / 80`` as a float is
        not ``Fraction(81, 80)``, and the resulting monzo would be nonsense
        rather than an error.
    primes : sequence of int, default :data:`PRIMES_5`
        The prime basis: strictly increasing, and every entry genuinely prime.
        Every prime factor of ``ratio`` must appear in it.  Both conditions are
        checked, because a monzo over a repeated or composite basis is not
        unique -- ``(2, 3, 3)`` would silently send all of prime 3 into the
        first of the two slots and report the second as untouched.

    Returns
    -------
    tuple of int
        ``m`` such that ``ratio == prod(p ** e for p, e in zip(primes, m))``.

    Raises
    ------
    ValueError
        If ``ratio`` has a prime factor outside ``primes``, or if ``primes`` is
        not a strictly increasing sequence of primes.

    Examples
    --------
    >>> monzo(Fraction(81, 80))
    (-4, 4, -1)
    >>> monzo(Fraction(3, 2))
    (-1, 1, 0)
    >>> monzo(Fraction(64, 63), PRIMES_7)
    (6, -2, 0, -1)
    >>> monzo(Fraction(7, 5))
    Traceback (most recent call last):
        ...
    ValueError: 7/5 is not 5-limit over primes (2, 3, 5): 7/1 is left over

    A basis that is not strictly increasing primes is refused rather than
    silently producing a non-unique monzo:

    >>> monzo(Fraction(9, 8), (2, 3, 3))
    Traceback (most recent call last):
        ...
    ValueError: the prime basis (2, 3, 3) must be strictly increasing; 3 does not exceed 3
    >>> monzo(Fraction(4, 1), (2, 4))
    Traceback (most recent call last):
        ...
    ValueError: the prime basis (2, 4) contains 4, which is not prime; a monzo over a composite basis is not unique
    """
    if isinstance(ratio, float):
        raise TypeError(
            f"monzo() needs an exact ratio, got the float {ratio!r}; "
            "write Fraction(81, 80) rather than 81 / 80"
        )
    fr = Fraction(ratio)
    if fr <= 0:
        raise ValueError(f"a ratio must be positive, got {ratio!r}")
    primes = _check_prime_basis(primes)
    num, den = fr.numerator, fr.denominator
    exponents: List[int] = []
    for p in primes:
        e = 0
        while num % p == 0:
            num //= p
            e += 1
        while den % p == 0:
            den //= p
            e -= 1
        exponents.append(e)
    if num != 1 or den != 1:
        raise ValueError(
            f"{fr} is not {primes[-1]}-limit over primes {tuple(primes)}: "
            f"{num}/{den} is left over"
        )
    return tuple(exponents)


def _is_primitive(m: Sequence[int]) -> bool:
    """True when the entries of ``m`` share no common factor (and ``m != 0``)."""
    g = 0
    for e in m:
        g = math.gcd(g, abs(int(e)))
    return g == 1


# --------------------------------------------------------------------------- #
# Integer linear algebra
# --------------------------------------------------------------------------- #
def _ext_gcd(x: int, y: int) -> Tuple[int, int, int]:
    """``(g, s, t)`` with ``g == s * x + t * y`` and ``g == gcd(|x|, |y|) >= 0``."""
    old_r, r = x, y
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    if old_r < 0:
        old_r, old_s, old_t = -old_r, -old_s, -old_t
    return old_r, old_s, old_t


def _unimodular_reduce(values: Sequence[int]) -> List[List[int]]:
    """Unimodular ``U`` with ``U @ values == (gcd, 0, ..., 0)``.

    Built as a product of ``2 x 2`` row operations of determinant ``+1``: each
    step replaces rows ``(0, i)`` by ``(s*r0 + t*ri, -(v_i/g)*r0 + (v_0/g)*ri)``
    where ``g = s*v_0 + t*v_i``.  Determinant ``s*v_0/g + t*v_i/g == 1``, so the
    whole product is unimodular and rows ``1..`` of ``U`` are a *basis* of the
    annihilator of ``values``, not merely a spanning set of a sublattice.
    """
    n = len(values)
    U = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    v = [int(x) for x in values]
    for i in range(1, n):
        if v[i] == 0:
            continue
        if v[0] == 0:
            # Rotation, not a swap: a swap has determinant -1.
            U[0], U[i] = U[i], [-x for x in U[0]]
            v[0], v[i] = v[i], 0
            continue
        g, s, t = _ext_gcd(v[0], v[i])
        p, q = -(v[i] // g), v[0] // g
        top = [s * U[0][k] + t * U[i][k] for k in range(n)]
        bottom = [p * U[0][k] + q * U[i][k] for k in range(n)]
        U[0], U[i] = top, bottom
        v[0], v[i] = g, 0
    return U


def saturated_annihilator(
    monzos: Sequence[Sequence[int]],
) -> List[List[int]]:
    """A saturated integer basis of ``{v : v . m == 0 for every m}``.

    "Saturated" means the returned rows generate *every* integer vector
    orthogonal to the given monzos, not a finite-index sublattice of them.  That
    is the whole point: a sublattice basis still satisfies ``v . m == 0`` and
    still looks like a mapping, but it multiplies the apparent number of periods
    per octave and silently corrupts every tuning derived from it.

    Parameters
    ----------
    monzos : sequence of int vectors
        All of the same length, and linearly independent.

    Returns
    -------
    list of list of int
        ``len(monzos[0]) - len(monzos)`` rows.

    Examples
    --------
    The rows annihilate the comma exactly, in integer arithmetic:

    >>> basis = saturated_annihilator([monzo(Fraction(81, 80))])
    >>> len(basis)
    2
    >>> [sum(x * y for x, y in zip(row, monzo(Fraction(81, 80)))) for row in basis]
    [0, 0]

    And the lattice they span is the canonical meantone mapping:

    >>> hermite_normal_form(basis)
    [[1, 0, -4], [0, 1, 4]]
    """
    ms = [tuple(int(x) for x in m) for m in monzos]
    if not ms:
        raise ValueError("saturated_annihilator() needs at least one monzo")
    n = len(ms[0])
    for m in ms:
        if len(m) != n:
            raise ValueError(
                f"all monzos must have the same length; got lengths "
                f"{sorted({len(x) for x in ms})}"
            )
    # Rows of ``basis`` are coordinates in the original dual space; they start
    # as all of Z^n and lose one dimension per comma.
    basis: List[List[int]] = [
        [1 if i == j else 0 for j in range(n)] for i in range(n)
    ]
    for m in ms:
        values = [sum(bi * mi for bi, mi in zip(row, m)) for row in basis]
        if all(v == 0 for v in values):
            raise ValueError(
                f"monzo {m} is already tempered out by the commas before it, so "
                "it removes no further dimension; the commas are dependent"
            )
        U = _unimodular_reduce(values)
        basis = [
            [
                sum(U[i][j] * basis[j][k] for j in range(len(basis)))
                for k in range(n)
            ]
            for i in range(1, len(basis))
        ]
    return basis


def hermite_normal_form(rows: Sequence[Sequence[int]]) -> List[List[int]]:
    """Row-style Hermite normal form over the integers.

    Echelon shape, every pivot positive, every entry *above* a pivot reduced
    into ``[0, pivot)``, zero rows dropped.  Two bases of the same lattice have
    the same Hermite normal form, which is what makes a mapping comparable with
    a published one.

    Examples
    --------
    >>> hermite_normal_form([[-1, -1, 0], [0, 1, 4]])
    [[1, 0, -4], [0, 1, 4]]

    Rank deficiency shows up as a shorter result rather than as an error:

    >>> hermite_normal_form([[2, 4], [3, 6]])
    [[1, 2]]
    """
    m = [[int(x) for x in r] for r in rows]
    if not m:
        return []
    n = len(m[0])
    for r in m:
        if len(r) != n:
            raise ValueError(
                f"all rows must have the same length; got lengths "
                f"{sorted({len(r) for r in m})}"
            )
    pivots: List[Tuple[int, int]] = []
    pivot = 0
    for col in range(n):
        if pivot >= len(m):
            break
        for r in range(pivot + 1, len(m)):
            if m[r][col] == 0:
                continue
            if m[pivot][col] == 0:
                m[pivot], m[r] = m[r], [-x for x in m[pivot]]
                continue
            g, s, t = _ext_gcd(m[pivot][col], m[r][col])
            u, v = -(m[r][col] // g), m[pivot][col] // g
            top = [s * m[pivot][k] + t * m[r][k] for k in range(n)]
            bottom = [u * m[pivot][k] + v * m[r][k] for k in range(n)]
            m[pivot], m[r] = top, bottom
        if m[pivot][col] == 0:
            continue
        if m[pivot][col] < 0:
            m[pivot] = [-x for x in m[pivot]]
        pivots.append((pivot, col))
        pivot += 1
    # Reduce above the pivots, left to right: a later pivot's row is zero in all
    # earlier pivot columns, so this never undoes an earlier reduction.
    for row_i, col in pivots:
        p = m[row_i][col]
        for r in range(row_i):
            q = m[r][col] // p  # floor division leaves the remainder in [0, p)
            if q:
                m[r] = [m[r][k] - q * m[row_i][k] for k in range(n)]
    return [r for r in m if any(r)]


# --------------------------------------------------------------------------- #
# Tunings
# --------------------------------------------------------------------------- #
def _tenney_weights(primes: Sequence[int]) -> List[Tuple[float, float]]:
    """``(log2(p), 1 / log2(p) ** 2)`` per prime.

    Weighting the squared *absolute* error by ``1 / log2(p) ** 2`` is the same
    as minimising the squared *relative* error, which is what Tenney weighting
    means: a cent of error on prime 7 is cheaper than a cent on prime 3 because
    7 is a longer interval.
    """
    out = []
    for p in primes:
        l = math.log2(p)
        out.append((l, 1.0 / (l * l)))
    return out


def _cte_generator(
    mapping: Sequence[Sequence[int]], primes: Sequence[int]
) -> float:
    """Generator in octaves for a pure period of ``1 / mapping[0][0]`` octaves.

    Closed form of the one-parameter Tenney-weighted normal equation

    .. math::
        G = \\frac{\\sum_i w_i b_i (l_i - a_i P)}{\\sum_i w_i b_i^2},
        \\qquad w_i = 1 / l_i^2,\\; l_i = \\log_2 p_i .
    """
    a, b = mapping[0], mapping[1]
    period = 1.0 / a[0]
    num = 0.0
    den = 0.0
    for ai, bi, (l, w) in zip(a, b, _tenney_weights(primes)):
        num += w * bi * (l - ai * period)
        den += w * bi * bi
    if den == 0.0:
        raise ValueError(
            f"the generator row {tuple(b)} is all zeros, so no generator can be "
            "fitted; the mapping is rank 1, not rank 2"
        )
    return num / den


def _te_tuning(
    mapping: Sequence[Sequence[int]], primes: Sequence[int]
) -> Tuple[float, float]:
    """``(period, generator)`` in octaves, both free -- the unconstrained TE fit.

    Two-parameter Tenney-weighted least squares.  The octave comes out slightly
    tempered; :attr:`Rank2Temperament.pote_generator_cents` rescales it pure.
    """
    a, b = mapping[0], mapping[1]
    saa = sab = sbb = sal = sbl = 0.0
    for ai, bi, (l, w) in zip(a, b, _tenney_weights(primes)):
        saa += w * ai * ai
        sab += w * ai * bi
        sbb += w * bi * bi
        sal += w * ai * l
        sbl += w * bi * l
    det = saa * sbb - sab * sab
    if det == 0.0:
        raise ValueError(
            f"mapping rows {tuple(a)} and {tuple(b)} are Tenney-collinear, so "
            "period and generator cannot be fitted independently"
        )
    return (sbb * sal - sab * sbl) / det, (saa * sbl - sab * sal) / det


# --------------------------------------------------------------------------- #
# The temperament object
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rank2Temperament:
    """A rank-2 regular temperament: a period, a generator, and a prime mapping.

    Constructing one canonicalises the mapping (Hermite normal form, then the
    sign and octave-reduction conventions described in the module docstring), so
    any basis of the right lattice gives the same object.  The commas are
    re-checked against the canonical rows in exact integer arithmetic.

    Parameters
    ----------
    name : str
    comma : Fraction
        The comma this temperament tempers out.
    primes : tuple of int
        Prime basis, e.g. :data:`PRIMES_5`.  Strictly increasing, and it must
        start at 2 -- see the *Octave-locked* note in the module docstring.
    mapping : 2 x len(primes) integers
        Row 0 counts periods per prime, row 1 counts generators per prime.  Any
        basis of the annihilator lattice will do; it is canonicalised.
    extra_commas : tuple of Fraction, optional
        Further commas, needed once the prime limit exceeds 5 (a rank-2
        temperament in an ``n``-prime basis tempers out ``n - 2`` of them).

    Examples
    --------
    >>> mt = temperament("meantone")
    >>> mt.mapping
    ((1, 1, 0), (0, 1, 4))
    >>> mt.periods_per_octave, round(mt.period_cents, 1)
    (1, 1200.0)
    >>> round(mt.generator_cents, 3)
    697.214

    Prime 3 is one generator up, prime 5 is four -- the syntonic comma is gone:

    >>> mt.hermite_mapping
    ((1, 0, -4), (0, 1, 4))
    >>> [round(e, 3) for e in mt.prime_errors]
    [0.0, -4.741, 2.544]

    The classical POTE figure, for comparison with published tables:

    >>> round(mt.pote_generator_cents, 3)
    696.239
    """

    name: str
    comma: Fraction
    primes: Tuple[int, ...]
    mapping: Tuple[Tuple[int, ...], Tuple[int, ...]]
    extra_commas: Tuple[Fraction, ...] = ()

    # ------------------------------------------------------------------ #
    # Canonicalisation
    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        primes = _check_prime_basis(self.primes)
        if primes[0] != 2:
            raise ValueError(
                f"the prime basis must start at 2, got {primes}: every period "
                "and generator in this class is measured in octaves "
                "(PERIOD_CENTS / periods_per_octave, 2 ** (1 / "
                "periods_per_octave), log2 of each prime), so a basis whose "
                "equave is not 2 would report a 1200-cent period, a "
                f"nonzero error on prime {primes[0]} and a meaningless "
                "generator rather than raising"
            )
        object.__setattr__(self, "primes", primes)
        object.__setattr__(self, "comma", Fraction(self.comma))
        object.__setattr__(
            self, "extra_commas", tuple(Fraction(c) for c in self.extra_commas)
        )

        rows = [list(r) for r in self.mapping]
        if len(rows) != 2:
            raise ValueError(
                f"a rank-2 mapping needs exactly 2 rows, got {len(rows)}"
            )
        for r in rows:
            if len(r) != len(primes):
                raise ValueError(
                    f"every mapping row needs one entry per prime "
                    f"({len(primes)} for {primes}), got a row of length {len(r)}"
                )
        hnf = hermite_normal_form(rows)
        if len(hnf) != 2:
            raise ValueError(
                f"mapping rows {tuple(rows[0])} and {tuple(rows[1])} are "
                f"linearly dependent, so they span rank {len(hnf)}, not rank 2"
            )
        if hnf[0][0] < 1:
            raise ValueError(
                f"the octave column of the mapping is empty ({hnf[0][0]} periods "
                "per octave); prime 2 cannot be tempered out"
            )
        for c in (self.comma,) + self.extra_commas:
            m = monzo(c, primes)
            for row in hnf:
                dot = sum(x * y for x, y in zip(row, m))
                if dot != 0:
                    raise ValueError(
                        f"val {tuple(row)} does not temper out {c} "
                        f"(monzo {m}): the dot product is {dot}, not 0"
                    )

        a, b = list(hnf[0]), list(hnf[1])
        gen = _cte_generator((a, b), primes)
        if gen < 0.0:
            # Same temperament, generator running the other way.
            b = [-x for x in b]
            gen = -gen
        period = 1.0 / a[0]
        for _ in range(64):
            if 0.0 <= gen < period:
                break
            k = int(math.floor(gen / period))
            a = [ai + k * bi for ai, bi in zip(a, b)]
            gen = _cte_generator((a, b), primes)
        else:  # pragma: no cover - unreachable for finite generators
            raise AssertionError(
                f"could not fold the generator {gen} into [0, {period}) for "
                f"{self.name}"
            )
        object.__setattr__(self, "mapping", (tuple(a), tuple(b)))

    # ------------------------------------------------------------------ #
    # Structure
    # ------------------------------------------------------------------ #
    @property
    def commas(self) -> Tuple[Fraction, ...]:
        """Every comma tempered out, primary first."""
        return (self.comma,) + self.extra_commas

    @property
    def periods_per_octave(self) -> int:
        """How many equal periods the octave is cut into.

        Read off ``mapping[0][0]``: prime 2 is worth exactly that many periods
        and no generators, which is what makes the period a pure fraction of a
        pure octave.
        """
        return self.mapping[0][0]

    @property
    def hermite_mapping(self) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        """The mapping before the sign and octave-reduction conventions.

        Same lattice, canonical Hermite form -- this is the shape published
        tables usually print.
        """
        hnf = hermite_normal_form([list(r) for r in self.mapping])
        return (tuple(hnf[0]), tuple(hnf[1]))

    # ------------------------------------------------------------------ #
    # Tuning
    # ------------------------------------------------------------------ #
    @property
    def period_cents(self) -> float:
        """Period size in cents; the octave is exactly pure by construction."""
        return PERIOD_CENTS / self.periods_per_octave

    @property
    def period_ratio(self) -> float:
        """Period as a frequency ratio, ``2 ** (1 / periods_per_octave)``."""
        return 2.0 ** (1.0 / self.periods_per_octave)

    @property
    def generator_octaves(self) -> float:
        """Generator in octaves -- the raw quantity everything else scales."""
        return _cte_generator(self.mapping, self.primes)

    @property
    def generator_cents(self) -> float:
        """Generator in cents, pure-period Tenney-weighted least squares (CTE).

        Lies in ``[0, period_cents)``.  See the module docstring on how this
        differs from the POTE figures in published tables.
        """
        return PERIOD_CENTS * self.generator_octaves

    @property
    def generator_ratio(self) -> float:
        """Generator as a frequency ratio."""
        return 2.0**self.generator_octaves

    @property
    def complement_cents(self) -> float:
        """``period_cents - generator_cents``.

        Builds exactly the same scales as :attr:`generator_cents` (Milne et al.
        §4), and is the figure some tables quote instead.
        """
        return self.period_cents - self.generator_cents

    def generator_fraction(self) -> float:
        """Generator as a fraction of the *period*, in ``(0, 1)``.

        This is the labyrinth coordinate: feed it to
        :func:`biotuner.mos.theory.sb_walk` or
        :meth:`biotuner.mos.scale.MOSScale.from_fraction` directly.

        Examples
        --------
        >>> round(temperament("meantone").generator_fraction(), 6)
        0.581012
        >>> round(temperament("srutal").generator_fraction(), 6)
        0.175227
        """
        return self.generator_octaves * self.periods_per_octave

    @property
    def pote_generator_cents(self) -> float:
        """Generator under the classical POTE tuning, for comparison.

        Period and generator are fitted together (so the octave comes out
        tempered), then everything is rescaled until the octave is pure.  This
        is the number the Xenharmonic wiki lists; :attr:`generator_cents`
        instead holds the octave pure *during* the fit.
        """
        period_te, gen_te = _te_tuning(self.mapping, self.primes)
        octave = self.periods_per_octave * period_te
        gen = gen_te / octave
        period = 1.0 / self.periods_per_octave
        gen -= math.floor(gen / period) * period
        return PERIOD_CENTS * gen

    @property
    def prime_errors(self) -> Tuple[float, ...]:
        """Cents by which each prime is mistuned, in ``primes`` order.

        Positive means sharp.  The first entry is exactly ``0.0``: the octave is
        pure by construction, which is the "PO" in POTE.

        Examples
        --------
        >>> [round(e, 3) for e in temperament("schismatic").prime_errors]
        [0.0, -0.236, -0.063]
        """
        a, b = self.mapping
        period = 1.0 / a[0]
        gen = self.generator_octaves
        return tuple(
            PERIOD_CENTS * (ai * period + bi * gen - math.log2(p))
            for ai, bi, p in zip(a, b, self.primes)
        )

    @property
    def max_error(self) -> float:
        """Largest absolute prime error, in cents."""
        return max(abs(e) for e in self.prime_errors)

    @property
    def rms_error(self) -> float:
        """Root-mean-square prime error in cents, over all primes.

        Prime 2 is included and contributes exactly zero, so this is a little
        lower than an RMS over the tempered primes alone; it is comparable
        across temperaments of the same prime limit, which is what it is for.
        """
        errs = self.prime_errors
        return math.sqrt(sum(e * e for e in errs) / len(errs))

    # ------------------------------------------------------------------ #
    # Scales
    # ------------------------------------------------------------------ #
    def mos_family(self, max_cardinality: int = 32) -> List["MOSScale"]:
        """Every MOS this temperament's optimal generator produces, per period.

        Cardinalities count notes *per period*, so a temperament with several
        periods per octave produces that many times as many notes per octave.

        Returns
        -------
        list of :class:`~biotuner.mos.scale.MOSScale`

        Examples
        --------
        Meantone walks up to the diatonic and on into chromatic territory:

        >>> [s.signature for s in temperament("meantone").mos_family(12)]
        ['2L1s', '2L3s', '5L2s', '7L5s']

        Hanson's minor-third generator gives the Kleismic series instead:

        >>> [s.cardinality for s in temperament("hanson").mos_family(19)]
        [3, 4, 7, 11, 15, 19]
        """
        from biotuner.mos.scale import mos_family as _mos_family

        period = self.period_ratio
        return _mos_family(
            period**self.generator_fraction(),
            max_cardinality=max_cardinality,
            period=period,
        )

    def supports(self, scale: "MOSScale", tol_cents: float = 5.0) -> bool:
        """Whether ``scale`` is this temperament at (near enough) its optimum.

        Both the period and the generator must match within ``tol_cents``.  The
        generator is compared against :attr:`complement_cents` too, since a
        generator and its complement build the same scale.

        Parameters
        ----------
        scale : MOSScale
        tol_cents : float, default 5.0

        Examples
        --------
        31-EDO's diatonic is meantone; 12-EDO's is close enough; Pythagorean
        tuning is not:

        >>> from biotuner.mos.scale import MOSScale
        >>> mt = temperament("meantone")
        >>> mt.supports(MOSScale.from_signature(5, 2, tuning=31))
        True
        >>> mt.supports(MOSScale.from_signature(5, 2, tuning=12))
        True

        Pythagorean tuning is 4.7 cents sharp of the meantone optimum, so it
        squeaks inside the default tolerance and fails a tighter one:

        >>> mt.supports(MOSScale.from_generator(3 / 2, 7))
        True
        >>> mt.supports(MOSScale.from_generator(3 / 2, 7), tol_cents=2.0)
        False
        """
        if abs(scale.period_cents - self.period_cents) > tol_cents:
            return False
        gen = scale.generator_cents
        distance = min(
            abs(gen - self.generator_cents), abs(gen - self.complement_cents)
        )
        return distance <= tol_cents

    # ------------------------------------------------------------------ #
    # Interop
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Rank2Temperament({self.name}, comma={self.comma}, "
            f"period={self.period_cents:.2f}c x{self.periods_per_octave}, "
            f"generator={self.generator_cents:.3f}c, "
            f"max_error={self.max_error:.3f}c)"
        )

    def to_dict(self) -> Dict[str, object]:
        """Flat, JSON-friendly summary -- one row of a temperament table."""
        return {
            "name": self.name,
            "comma": str(self.comma),
            "extra_commas": [str(c) for c in self.extra_commas],
            "primes": list(self.primes),
            "mapping": [list(r) for r in self.mapping],
            "hermite_mapping": [list(r) for r in self.hermite_mapping],
            "periods_per_octave": self.periods_per_octave,
            "period_cents": self.period_cents,
            "generator_cents": self.generator_cents,
            "complement_cents": self.complement_cents,
            "pote_generator_cents": self.pote_generator_cents,
            "generator_fraction": self.generator_fraction(),
            "prime_errors": list(self.prime_errors),
            "max_error": self.max_error,
            "rms_error": self.rms_error,
        }

    def summary(self) -> str:
        """Multi-line human-readable description.

        Examples
        --------
        >>> print(temperament("porcupine").summary())
        porcupine   tempers out 250/243
          mapping        <1 2 3|   <0 -3 -5|   (primes 2.3.5)
          period           1200.000 c  x1 per octave
          generator         164.166 c  (0.136805 of the period)
          also             1035.834 c complement,  163.950 c POTE
          prime errors   2: +0.000,  3: +5.547,  5: -7.143 c
          max 7.143 c,  rms 5.222 c
        """
        errs = ",  ".join(
            f"{p}: {e:+.3f}" for p, e in zip(self.primes, self.prime_errors)
        )
        rows = "   ".join(
            "<" + " ".join(str(x) for x in row) + "|" for row in self.mapping
        )
        return "\n".join(
            [
                f"{self.name}   tempers out "
                + ", ".join(str(c) for c in self.commas),
                f"  mapping        {rows}   "
                f"(primes {'.'.join(str(p) for p in self.primes)})",
                f"  period         {self.period_cents:10.3f} c  "
                f"x{self.periods_per_octave} per octave",
                f"  generator      {self.generator_cents:10.3f} c  "
                f"({self.generator_fraction():.6f} of the period)",
                f"  also           {self.complement_cents:10.3f} c complement,"
                f"  {self.pote_generator_cents:.3f} c POTE",
                f"  prime errors   {errs} c",
                f"  max {self.max_error:.3f} c,  rms {self.rms_error:.3f} c",
            ]
        )


# --------------------------------------------------------------------------- #
# Construction from commas
# --------------------------------------------------------------------------- #
def rank2_from_comma(
    comma: RatioLike,
    primes: Sequence[int] = PRIMES_5,
    name: Optional[str] = None,
    extra_commas: Sequence[RatioLike] = (),
) -> Rank2Temperament:
    """Build the rank-2 temperament that tempers out ``comma``.

    Parameters
    ----------
    comma : Fraction
        Must be *primitive*: the entries of its monzo share no common factor.
        ``(81/80) ** 2`` defines the same temperament as ``81/80`` but is not a
        comma, and accepting it would quietly hide the squaring.
    primes : sequence of int, default :data:`PRIMES_5`
        Strictly increasing primes starting at 2; see the *Octave-locked* note
        in the module docstring for why the equave is not free.
    name : str, optional
        Defaults to the comma itself.
    extra_commas : sequence of Fraction, optional
        Required above the 5-limit: ``len(primes) - 2`` commas in total pin a
        rank-2 temperament down.

    Returns
    -------
    Rank2Temperament

    Examples
    --------
    >>> t = rank2_from_comma(Fraction(81, 80))
    >>> t.name, t.mapping, round(t.generator_cents, 3)
    ('81/80', ((1, 1, 0), (0, 1, 4)), 697.214)

    Septimal meantone needs a second comma, and then prime 7 joins in:

    >>> sm = rank2_from_comma(Fraction(81, 80), PRIMES_7, "septimal meantone",
    ...                       extra_commas=[Fraction(126, 125)])
    >>> sm.mapping
    ((1, 1, 0, -3), (0, 1, 4, 10))
    >>> round(sm.generator_cents, 3)
    696.952

    A non-primitive comma is refused rather than silently accepted:

    >>> rank2_from_comma(Fraction(6561, 6400))
    Traceback (most recent call last):
        ...
    ValueError: comma 6561/6400 has monzo (-8, 8, -2), whose entries share the factor 2; use its primitive root 81/80 instead

    So is a no-twos subgroup, whose equave this octave-locked module cannot
    represent:

    >>> rank2_from_comma(Fraction(245, 243), (3, 5, 7))
    Traceback (most recent call last):
        ...
    ValueError: the prime basis must start at 2, got (3, 5, 7): every period and generator in this class is measured in octaves (PERIOD_CENTS / periods_per_octave, 2 ** (1 / periods_per_octave), log2 of each prime), so a basis whose equave is not 2 would report a 1200-cent period, a nonzero error on prime 3 and a meaningless generator rather than raising
    """
    commas = [Fraction(comma)] + [Fraction(c) for c in extra_commas]
    monzos = [monzo(c, primes) for c in commas]
    for c, m in zip(commas, monzos):
        if not any(m):
            raise ValueError(
                f"comma {c} has the all-zero monzo, so it tempers nothing out"
            )
        if not _is_primitive(m):
            g = 0
            for e in m:
                g = math.gcd(g, abs(e))
            root = Fraction(1)
            for p, e in zip(primes, m):
                root *= Fraction(p) ** (e // g)
            raise ValueError(
                f"comma {c} has monzo {m}, whose entries share the factor {g}; "
                f"use its primitive root {root} instead"
            )
    basis = saturated_annihilator(monzos)
    if len(basis) != 2:
        raise ValueError(
            f"{len(commas)} comma(s) over {tuple(primes)} leave a rank-"
            f"{len(basis)} temperament; rank 2 needs exactly "
            f"{len(primes) - 2} comma(s) -- pass the rest as extra_commas"
        )
    return Rank2Temperament(
        name=name if name is not None else str(commas[0]),
        comma=commas[0],
        primes=tuple(int(p) for p in primes),
        mapping=(tuple(basis[0]), tuple(basis[1])),
        extra_commas=tuple(commas[1:]),
    )


# --------------------------------------------------------------------------- #
# The named catalogue
# --------------------------------------------------------------------------- #
#: 5-limit rank-2 temperaments by name, each stored as the comma it tempers out.
#: The mapping, period and generator of each are *computed* from the comma --
#: see :func:`temperament`.
TEMPERAMENTS: Dict[str, Fraction] = {
    "meantone": Fraction(81, 80),
    "augmented": Fraction(128, 125),
    "diminished": Fraction(648, 625),
    "blackwood": Fraction(256, 243),
    "dicot": Fraction(25, 24),
    "srutal": Fraction(2048, 2025),
    "magic": Fraction(3125, 3072),
    "hanson": Fraction(15625, 15552),
    "porcupine": Fraction(250, 243),
    "tetracot": Fraction(20000, 19683),
    "negri": Fraction(16875, 16384),
    "wuerschmidt": Fraction(393216, 390625),
    "schismatic": Fraction(32805, 32768),
    "mavila": Fraction(135, 128),
    "father": Fraction(16, 15),
    "bug": Fraction(27, 25),
    "amity": Fraction(1600000, 1594323),
    "sensipent": Fraction(78732, 78125),
    "ripple": Fraction(6561, 6250),
    "compton": Fraction(531441, 524288),
}


@lru_cache(maxsize=None)
def _build(key: str) -> Rank2Temperament:
    """Cached construction, keyed on the *normalised* name."""
    return rank2_from_comma(TEMPERAMENTS[key], PRIMES_5, key)


def temperament(name: str) -> Rank2Temperament:
    """The named 5-limit temperament, built from its comma and cached.

    Examples
    --------
    >>> temperament("srutal").periods_per_octave
    2
    >>> round(temperament("srutal").period_cents, 1)
    600.0
    >>> temperament("magic").mapping
    ((1, 0, 2), (0, 5, 1))

    The name is normalised before the cache is consulted, so spelling variants
    share one object:

    >>> temperament("Meantone") is temperament("meantone")
    True

    Unknown names list what is available rather than raising a bare KeyError:

    >>> temperament("meantime")
    Traceback (most recent call last):
        ...
    ValueError: unknown temperament 'meantime'; did you mean 'meantone'?
    """
    key = str(name).strip().lower()
    if key not in TEMPERAMENTS:
        close = [k for k in TEMPERAMENTS if k.startswith(key[:4])]
        hint = (
            f"; did you mean {' or '.join(repr(c) for c in close)}?"
            if close
            else f"; known names are {sorted(TEMPERAMENTS)}"
        )
        raise ValueError(f"unknown temperament {name!r}{hint}")
    return _build(key)


def all_temperaments() -> Dict[str, Rank2Temperament]:
    """Every named temperament, built and cached.

    Examples
    --------
    >>> ts = all_temperaments()
    >>> len(ts)
    20
    >>> sorted(n for n, t in ts.items() if t.periods_per_octave > 1)
    ['augmented', 'blackwood', 'compton', 'diminished', 'srutal']
    """
    return {name: temperament(name) for name in TEMPERAMENTS}


def nearest_temperaments(
    generator_cents: float,
    period_cents: float = PERIOD_CENTS,
    n: int = 3,
    max_distance_cents: float = 25.0,
) -> List[Tuple[str, Rank2Temperament, float]]:
    """Named temperaments whose optimal generator sits near a given one.

    This is the labyrinth's reverse lookup: the user lands somewhere on the
    disc, and this reports which radial lines (Milne et al. §4) they are close
    to.  Only temperaments with a matching period are considered -- a 600-cent
    period is a different disc from a 1200-cent one.  Distances are measured to
    the generator *and* to its complement, because those draw mirror-image lines
    that build identical scales.

    Parameters
    ----------
    generator_cents : float
        Reduced into ``[0, period_cents)`` before comparing.
    period_cents : float, default 1200.0
    n : int, default 3
        Most that will be returned.
    max_distance_cents : float, default 25.0

    Returns
    -------
    list of (name, Rank2Temperament, distance_cents)
        Nearest first; ``distance_cents`` is unsigned.

    Examples
    --------
    >>> [(n, round(d, 3)) for n, _, d in nearest_temperaments(696.6)]
    [('meantone', 0.614), ('schismatic', 5.119), ('mavila', 19.455)]

    Nothing named lives near 640 cents:

    >>> nearest_temperaments(640.0)
    []

    Half-octave periods are searched separately:

    >>> [n for n, _, _ in nearest_temperaments(105.0, period_cents=600.0)]
    ['srutal']
    """
    if period_cents <= 0:
        raise ValueError(f"period_cents must be positive, got {period_cents!r}")
    query = generator_cents - math.floor(generator_cents / period_cents) * period_cents
    out: List[Tuple[str, Rank2Temperament, float]] = []
    for name, temp in all_temperaments().items():
        if abs(temp.period_cents - period_cents) > 1e-2:
            continue
        distance = min(
            abs(query - temp.generator_cents), abs(query - temp.complement_cents)
        )
        if distance <= max_distance_cents:
            out.append((name, temp, distance))
    out.sort(key=lambda row: (row[2], row[0]))
    return out[:n]
