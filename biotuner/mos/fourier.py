"""Fourier scratching -- playing a scale by manipulating a DFT.

Milne et al. (2011) §5 close the scale labyrinth with a performance technique.
A virtual robot with ``n`` fingers strikes a *continuous* circular keyboard at a
fixed pulse.  Finger ``k`` is described by one complex number

.. math::  f_k = r_k \\, e^{i t_k}

whose magnitude ``r_k`` is how hard it strikes (loudness) and whose phase
``t_k`` is *where* on the circle it strikes (pitch).  The whole performance
state is therefore a single vector :math:`f \\in \\mathbb{C}^n`.

The point of the technique is that the performer never edits ``f``.  They edit
its discrete Fourier transform and let the inverse transform put the fingers
back on the keyboard -- "scratching" one Fourier coefficient nudges *every*
finger at once, in a way that is coherent rather than arbitrary.  A single
coefficient's phase is a rigid rotation of a whole interval cycle; its
magnitude is that cycle's depth.  That is what makes the gesture playable:
one continuous control, ``n`` coordinated voices.

Two consequences the paper leans on, both checked in ``tests/mos/test_fourier.py``:

* The *elementary* play states -- the paper's "pure partials", here
  :func:`partial` -- are the states whose spectrum is a single unit impulse.
  Partial ``k`` spreads the fingers evenly around the circle in ``k`` turns.
* Changing the number of fingers is a spectral edit too: grow by
  :meth:`PlayState.zero_pad`, shrink by :meth:`PlayState.prune`, which deletes
  "the Fourier coefficients with minimal energy" exactly as §5 prescribes.

The keyboard is *continuous*, so a phase only becomes a note once it is
quantised against a scale.  :func:`keyboard_sectors` does that, following Fig. 8:
key widths are "proportional to the sizes of the step intervals above each
tone", so a scale tone sits at the *lower* edge of its own key and the key
extends up to the next tone.  Quantising is therefore rounding **down** to the
nearest scale tone, not rounding to nearest -- which has a real musical
consequence, documented on :func:`keyboard_sectors` and measured in the tests.

References
----------
Milne, A.J., Carlé, M., Sethares, W.A., Noll, T., Holland, S. (2011).
Scratching the Scale Labyrinth. In *Mathematics and Computation in Music*,
LNAI 6726, 180--195. https://doi.org/10.1007/978-3-642-21590-2_14
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "PlayState",
    "NoteEvent",
    "partial",
    "keyboard_sectors",
    "phase_to_degree",
    "to_events",
    "to_frequencies",
    "scratch_sequence",
]

TWO_PI = 2.0 * math.pi

#: Radians below a key boundary that still count as *on* that boundary.
#:
#: Quantising rounds **down**, so a finger that ought to sit exactly on a scale
#: tone but lands one ulp below it drops a whole scale step.  That is not a
#: hypothetical: ``exp``/``angle`` round-tripping a finger and multiplying a
#: degree by ``2*pi`` are different computations, and for an equal division
#: (a degenerate MOS, ``L == s``, where every finger sits exactly on a tone)
#: they disagree in the last ulp for most cardinalities.  The same thing at the
#: wrap point puts a finger in the *top* key instead of the root's.
#:
#: 1e-12 rad is 2e-10 cents in a 2/1 period -- some 200x the worst observed
#: round-trip error, and far below both audibility and the narrowest key any
#: real scale has.
_PHASE_EPS = 1e-12

#: Attribute names :func:`scratch_sequence` may sweep.
_SCRATCH_ATTRS = ("magnitude", "phase", "scale", "rotate")


# --------------------------------------------------------------------------- #
# The play state
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, eq=False)
class PlayState:
    """The ``n`` fingers of the robot, as one complex vector.

    Parameters
    ----------
    f : array_like
        1-D, coerced to ``complex128`` and copied read-only, so a
        :class:`PlayState` really is immutable rather than merely frozen at the
        attribute level.  ``f[k]`` is finger ``k``: ``abs(f[k])`` its loudness,
        ``angle(f[k])`` its position on the circular keyboard.

    Notes
    -----
    The spectrum convention here is ``fft(f) / n``, so coefficient magnitudes
    are on the same scale as finger magnitudes: :func:`partial` has a unit
    coefficient, not a coefficient of ``n``.  :meth:`from_spectrum` undoes it.

    Examples
    --------
    >>> s = PlayState.from_polar([1.0, 0.5], [0.0, math.pi])
    >>> s.n
    2
    >>> [round(float(m), 6) for m in s.magnitudes]
    [1.0, 0.5]
    >>> [round(float(p), 6) for p in s.phases]
    [0.0, 3.141593]
    """

    f: np.ndarray

    def __post_init__(self) -> None:
        arr = np.array(self.f, dtype=np.complex128, copy=True)
        if arr.ndim != 1:
            raise ValueError(
                f"a play state is a 1-D vector of finger states; got an array "
                f"of shape {arr.shape} ({arr.ndim} dimensions)"
            )
        if arr.size == 0:
            raise ValueError(
                "a play state needs at least one finger; got an empty array"
            )
        arr.setflags(write=False)
        object.__setattr__(self, "f", arr)

    # ---------------------------------------------------------------- #
    # Constructors
    # ---------------------------------------------------------------- #
    @classmethod
    def from_polar(
        cls, magnitudes: Sequence[float], phases: Sequence[float]
    ) -> "PlayState":
        """Build from loudnesses and keyboard positions.

        Parameters
        ----------
        magnitudes, phases : array_like
            Same length.  ``phases`` are radians and need not be reduced.

        Examples
        --------
        >>> PlayState.from_polar([1, 1, 1], [0, TWO_PI / 3, 2 * TWO_PI / 3]).n
        3
        """
        r = np.asarray(magnitudes, dtype=np.float64)
        t = np.asarray(phases, dtype=np.float64)
        if r.shape != t.shape:
            raise ValueError(
                f"magnitudes and phases must have the same shape; got "
                f"{r.shape} and {t.shape}"
            )
        return cls(r * np.exp(1j * t))

    @classmethod
    def from_spectrum(cls, a: Sequence[complex]) -> "PlayState":
        """Resynthesise a play state from its Fourier coefficients.

        Exact inverse of :attr:`spectrum`: since that is ``fft(f) / n``, this is
        ``n * ifft(a)``.  The number of fingers is ``len(a)``, which is what
        makes :meth:`zero_pad` and :meth:`prune` able to change it.

        Examples
        --------
        >>> s = PlayState([1 + 2j, -3j, 0.5])
        >>> bool(np.allclose(PlayState.from_spectrum(s.spectrum).f, s.f))
        True
        """
        arr = np.asarray(a, dtype=np.complex128)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError(
                f"a spectrum is a non-empty 1-D vector of Fourier "
                f"coefficients; got shape {arr.shape}"
            )
        return cls(arr.size * np.fft.ifft(arr))

    # ---------------------------------------------------------------- #
    # Views
    # ---------------------------------------------------------------- #
    @property
    def n(self) -> int:
        """Number of fingers."""
        return int(self.f.size)

    @property
    def magnitudes(self) -> np.ndarray:
        """``|f_k|`` -- how hard each finger strikes."""
        return np.abs(self.f)

    @property
    def phases(self) -> np.ndarray:
        """``arg(f_k)`` reduced to ``[0, 2*pi)`` -- where each finger strikes.

        ``numpy`` returns angles in ``(-pi, pi]``; reducing modulo ``2*pi``
        pushes a hair-negative angle up against ``2*pi``, which is the far side
        of the keyboard from where it belongs.  Anything within
        ``_PHASE_EPS`` of a full turn is therefore snapped back to ``0``.
        """
        ph = np.angle(self.f) % TWO_PI
        ph[TWO_PI - ph <= _PHASE_EPS] = 0.0
        return ph

    @property
    def spectrum(self) -> np.ndarray:
        """Fourier coefficients, ``fft(f) / n`` -- what the performer edits."""
        return np.fft.fft(self.f) / self.n

    @property
    def energy(self) -> float:
        """``sum |f_k|^2``.

        Parseval ties this to the spectrum: ``energy == n * sum |a_p|^2``, which
        is why "delete the minimal-energy coefficients" (:meth:`prune`) is the
        least destructive way to drop a finger.

        Examples
        --------
        >>> s = PlayState([3, 4j])
        >>> round(s.energy, 9)
        25.0
        """
        return float(np.sum(np.abs(self.f) ** 2))

    # ---------------------------------------------------------------- #
    # Comparison
    # ---------------------------------------------------------------- #
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PlayState):
            return NotImplemented
        return self.f.shape == other.f.shape and bool(np.array_equal(self.f, other.f))

    __hash__ = None  # type: ignore[assignment]

    def allclose(self, other: "PlayState", rtol: float = 1e-9, atol: float = 1e-12) -> bool:
        """Numerical equality -- what round-trip checks actually want.

        Examples
        --------
        >>> s = PlayState([1, 1j, -1])
        >>> PlayState.from_spectrum(s.spectrum).allclose(s)
        True
        """
        if self.n != other.n:
            return False
        return bool(np.allclose(self.f, other.f, rtol=rtol, atol=atol))

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"PlayState(n={self.n}, energy={self.energy:.4f})"

    # ---------------------------------------------------------------- #
    # Scratching
    # ---------------------------------------------------------------- #
    def _check_index(self, k: int) -> int:
        if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
            raise TypeError(
                f"Fourier coefficient index must be an integer, got {k!r} "
                f"({type(k).__name__})"
            )
        k = int(k)
        if not -self.n <= k < self.n:
            raise IndexError(
                f"Fourier coefficient index {k} is out of range for a "
                f"{self.n}-finger play state; valid indices are "
                f"{-self.n}..{self.n - 1}"
            )
        return k % self.n

    def scratch(
        self,
        k: int,
        magnitude: Optional[float] = None,
        phase: Optional[float] = None,
        scale: Optional[float] = None,
        rotate: Optional[float] = None,
    ) -> "PlayState":
        """Edit Fourier coefficient ``k`` and resynthesise.

        The performer's single gesture.  Every finger moves, but coherently:
        rotating coefficient ``k`` by ``d`` advances the ``k``-th interval cycle
        by ``d`` without disturbing any other cycle.

        Parameters
        ----------
        k : int
            Coefficient index.  Negative indices count from the end, as for a
            list, so ``-1`` is the highest coefficient.
        magnitude : float, optional
            **Set** ``|a_k|`` to this.  Must be non-negative.
        scale : float, optional
            **Multiply** ``|a_k|`` by this.  A negative factor is allowed and
            flips the coefficient's phase by ``pi``.
        phase : float, optional
            **Set** ``arg(a_k)`` to this, in radians.
        rotate : float, optional
            **Add** this to ``arg(a_k)``, in radians.

        Raises
        ------
        ValueError
            If the absolute and the relative form of the *same* attribute are
            combined (``magnitude`` with ``scale``, or ``phase`` with
            ``rotate``): there is no sensible order in which to apply both.

        Examples
        --------
        Rotating a pure partial's own coefficient just slides every finger
        round the keyboard by the same angle:

        >>> p = partial(4, 1)
        >>> q = p.scratch(1, rotate=math.pi / 2)
        >>> [round(float(x), 6) for x in (q.phases - p.phases) % TWO_PI]
        [1.570796, 1.570796, 1.570796, 1.570796]

        Damping a coefficient to nothing removes that cycle entirely:

        >>> [round(float(abs(z)), 9) for z in p.scratch(1, magnitude=0.0).f]
        [0.0, 0.0, 0.0, 0.0]
        """
        k = self._check_index(k)
        if magnitude is not None and scale is not None:
            raise ValueError(
                "magnitude= sets |a_k| and scale= multiplies it; passing both "
                f"is ambiguous (got magnitude={magnitude!r}, scale={scale!r}). "
                "Use one or the other."
            )
        if phase is not None and rotate is not None:
            raise ValueError(
                "phase= sets arg(a_k) and rotate= adds to it; passing both is "
                f"ambiguous (got phase={phase!r}, rotate={rotate!r}). "
                "Use one or the other."
            )
        a = self.spectrum
        mag = float(np.abs(a[k]))
        ang = float(np.angle(a[k]))
        if magnitude is not None:
            if float(magnitude) < 0.0:
                raise ValueError(
                    f"magnitude= sets an absolute |a_k| and must be "
                    f"non-negative, got {magnitude!r}; for a phase inversion "
                    "use scale=-1 or rotate=math.pi"
                )
            mag = float(magnitude)
        if scale is not None:
            mag *= float(scale)
        if phase is not None:
            ang = float(phase)
        if rotate is not None:
            ang += float(rotate)
        a[k] = mag * np.exp(1j * ang)
        return PlayState.from_spectrum(a)

    # ---------------------------------------------------------------- #
    # Changing the number of fingers
    # ---------------------------------------------------------------- #
    def zero_pad(self, m: int) -> "PlayState":
        """Grow to ``m`` fingers by appending zero Fourier coefficients.

        Milne et al. §5 change dimension "by zero-padding the DFT of the current
        play state".  Padding at the top of the coefficient list keeps every
        existing coefficient at its own index, so a partial stays the same
        partial and any control the performer had mapped to coefficient ``k``
        still points at it: ``partial(n, k).zero_pad(m)`` is ``partial(m, k)``
        up to floating-point round-off (hence :meth:`allclose`, not ``==``, in
        the check below -- the two go through different FFT lengths).

        Examples
        --------
        >>> partial(4, 1).zero_pad(8).allclose(partial(8, 1))
        True
        >>> [round(float(abs(z)), 9) for z in PlayState([1, 0, 0]).zero_pad(5).spectrum]
        [0.333333333, 0.333333333, 0.333333333, 0.0, 0.0]
        """
        m = int(m)
        if m < self.n:
            raise ValueError(
                f"zero_pad only grows a play state; got m={m} for a "
                f"{self.n}-finger state. Use prune() or truncate() to shrink."
            )
        a = np.zeros(m, dtype=np.complex128)
        a[: self.n] = self.spectrum
        return PlayState.from_spectrum(a)

    def _keep(self, keep_count: int) -> "PlayState":
        """Keep the ``keep_count`` highest-energy coefficients, in index order.

        Ties go to the *lower* index.  ``argsort(..., kind='stable')`` would do
        the opposite here -- it puts the lower index earlier in the ascending
        order, i.e. first in line to be dropped -- so the secondary key is
        ``-index``, which drops the higher index of a tied pair first.
        """
        energies = np.abs(self.spectrum) ** 2
        # primary key last: ascending energy, then descending index
        order = np.lexsort((-np.arange(self.n), energies))
        keep = np.sort(order[self.n - keep_count :])
        return PlayState.from_spectrum(self.spectrum[keep])

    def prune(self, m: int) -> "PlayState":
        """Drop the ``m`` lowest-energy Fourier coefficients, leaving ``n - m``.

        The paper's way of *shrinking* the robot's hand: "deleting the Fourier
        coefficients with minimal energy" throws away the least of the sound.
        Survivors keep their relative order but are re-indexed, so coefficient
        identity is *not* preserved -- unlike :meth:`zero_pad`.  Equal-energy
        coefficients are broken toward the *lower* index: the higher one is
        dropped first.

        Examples
        --------
        >>> s = PlayState.from_spectrum([3, 0.1, 2, 0.2])
        >>> q = s.prune(2)
        >>> q.n, [round(float(abs(z)), 9) for z in q.spectrum]
        (2, [3.0, 2.0])

        The tie-break, made visible by two coefficients of equal magnitude but
        different phase -- the ``1`` survives, not the ``1j``:

        >>> [complex(z) for z in PlayState.from_spectrum([1, 1j, 0, 0]).truncate(1).spectrum]
        [(1+0j)]

        A partial's one loud coefficient always survives, so pruning a partial
        just shortens the hand:

        >>> partial(6, 2).prune(2).n
        4
        """
        m = int(m)
        if not 0 <= m < self.n:
            raise ValueError(
                f"can drop between 0 and {self.n - 1} of a {self.n}-finger "
                f"state's coefficients, got m={m}"
            )
        return self._keep(self.n - m)

    def truncate(self, m: int) -> "PlayState":
        """Keep only the ``m`` highest-energy Fourier coefficients.

        The complement of :meth:`prune`: ``truncate(m)`` and ``prune(n - m)``
        are the same edit, and share the same tie-breaking.

        Examples
        --------
        >>> s = PlayState([1, 2, 3, 4])
        >>> s.truncate(2).allclose(s.prune(2))
        True
        """
        m = int(m)
        if not 1 <= m <= self.n:
            raise ValueError(
                f"can keep between 1 and {self.n} of a {self.n}-finger "
                f"state's coefficients, got m={m}"
            )
        return self._keep(m)

    # ---------------------------------------------------------------- #
    # Whole-hand gestures
    # ---------------------------------------------------------------- #
    def rotate_all(self, delta: float) -> "PlayState":
        """Slide every finger ``delta`` radians round the keyboard.

        Transposition on a continuous keyboard.  In the spectrum this is a
        single global factor ``exp(i*delta)`` on *all* coefficients, so it is
        the one gesture that is as simple in either domain.

        Examples
        --------
        >>> p = partial(5, 1).rotate_all(math.pi)
        >>> [round(float(x), 5) for x in p.phases]
        [3.14159, 4.39823, 5.65487, 0.62832, 1.88496]
        """
        return PlayState(self.f * np.exp(1j * float(delta)))

    def interpolate(self, other: "PlayState", t: float) -> "PlayState":
        """Blend toward ``other``, linearly in the Fourier domain.

        ``t = 0`` is ``self``, ``t = 1`` is ``other``; values outside ``[0, 1]``
        extrapolate.  Because the DFT is linear this coincides exactly with
        interpolating the finger vectors -- the point of phrasing it spectrally
        is that the Fourier coefficients are the performer's coordinates, so a
        morph specified there is a morph they can hear themselves making.  For
        a *non*-linear morph (constant-loudness rotation of a cycle, say) use
        :func:`scratch_sequence` on one coefficient instead.

        Examples
        --------
        >>> a, b = partial(4, 1), partial(4, 2)
        >>> mid = a.interpolate(b, 0.5)
        >>> [round(float(abs(z)), 9) for z in mid.spectrum]
        [0.0, 0.5, 0.5, 0.0]
        """
        if self.n != other.n:
            raise ValueError(
                f"interpolation needs matching finger counts; got {self.n} and "
                f"{other.n}. Use zero_pad() to bring the smaller one up first."
            )
        t = float(t)
        return PlayState.from_spectrum(
            (1.0 - t) * self.spectrum + t * other.spectrum
        )


# --------------------------------------------------------------------------- #
# Elementary play states
# --------------------------------------------------------------------------- #
def partial(n: int, k: int) -> PlayState:
    """The ``k``-th elementary play state on ``n`` fingers -- a "pure partial".

    ``f_j = exp(2*pi*i*j*k/n)``: all fingers strike equally hard, and their
    positions wind ``k`` times round the keyboard.  Its spectrum is a unit
    impulse at ``k``, so these are the basis Milne et al. Fig. 8 draws and the
    thing every other play state is a superposition of.

    Note that ``k`` and ``k + n`` give the same state, and that the *set* of
    finger positions is the same for every ``k`` -- only the order in which the
    fingers visit them changes, which is why :func:`to_events` is where ``k``
    starts to matter musically.

    Parameters
    ----------
    n : int
        Number of fingers, ``>= 1``.
    k : int
        Partial index, taken modulo ``n``.

    Returns
    -------
    PlayState

    Examples
    --------
    >>> [round(float(abs(z)), 9) for z in partial(4, 1).spectrum]
    [0.0, 1.0, 0.0, 0.0]
    >>> [round(float(p), 6) for p in partial(4, 3).phases]
    [0.0, 4.712389, 3.141593, 1.570796]
    >>> partial(4, 5) == partial(4, 1)
    True

    Fingers that land back on the root land there *exactly*, because the
    ``j*k mod n`` below is integer arithmetic.  Doing the reduction in floats
    instead leaves ``exp(2*pi*i*j*k/n)`` a hair off 1, and a hair-negative angle
    reads as the *top* key rather than the root -- a whole scale step of error:

    >>> sorted(set(round(float(p), 9) for p in partial(12, 3).phases))
    [0.0, 1.570796327, 3.141592654, 4.71238898]
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"a play state needs at least one finger, got n={n}")
    idx = (np.arange(n) * (int(k) % n)) % n
    return PlayState(np.exp(2j * np.pi * idx / n))


# --------------------------------------------------------------------------- #
# The circular keyboard
# --------------------------------------------------------------------------- #
def _degrees(mode_or_scale: Any) -> List[float]:
    """Pull validated period fractions off an :class:`MOSScale` or :class:`Mode`."""
    degs = getattr(mode_or_scale, "degrees", None)
    if degs is None:
        raise TypeError(
            f"expected an MOSScale or a Mode -- anything exposing a .degrees "
            f"sequence of period fractions -- got "
            f"{type(mode_or_scale).__name__}"
        )
    d = [float(x) for x in degs]
    if not d:
        raise ValueError("the scale has no degrees; expected at least one")
    if abs(d[0]) > 1e-12:
        raise ValueError(
            f"scale degrees must start at 0 (the root sits at angle 0 on the "
            f"keyboard); got degrees[0] = {d[0]!r}"
        )
    for i in range(len(d) - 1):
        if not d[i] < d[i + 1]:
            raise ValueError(
                f"scale degrees must be strictly ascending period fractions in "
                f"[0, 1); degrees[{i}] = {d[i]!r} is not below "
                f"degrees[{i + 1}] = {d[i + 1]!r}"
            )
    if d[-1] >= 1.0:
        raise ValueError(
            f"scale degrees must be period fractions in [0, 1); got "
            f"degrees[-1] = {d[-1]!r}"
        )
    return d


def keyboard_sectors(mode_or_scale: Any) -> List[Tuple[float, float]]:
    """The angular key each scale degree owns, as ``[start, end)`` in radians.

    Milne et al. Fig. 8 gives the circular keyboard "key widths which are
    proportional to the sizes of the step intervals above each tone".  So a tone
    sits at the *bottom* edge of its own key and the key runs up to the next
    tone: degree ``i`` owns ``[2*pi*d_i, 2*pi*d_{i+1})``, and the last degree's
    key closes the circle at ``2*pi``.  Keys are therefore unequal -- wide above
    a large step, narrow above a small one -- and they tile ``[0, 2*pi)`` exactly.

    The asymmetry is load-bearing.  Because a tone sits on its key's *lower*
    edge, :func:`phase_to_degree` rounds a phase **down** to the nearest scale
    tone rather than to the nearest tone in either direction.  Whether a set of
    evenly spaced fingers then lands one-per-key depends on the *mode*, not just
    on the tuning: it happens exactly in the mode whose step pattern is the
    Christoffel word ``theory.christoffel_word(n_large, n_small)``, since that
    word is by construction the floor-quantisation of the equal division.  See
    :func:`to_events` for what that means in practice.

    Parameters
    ----------
    mode_or_scale : MOSScale or Mode
        Anything exposing ``.degrees`` as ascending period fractions in
        ``[0, 1)`` starting at ``0``.

    Returns
    -------
    list of (float, float)
        One ``(start, end)`` pair per degree, in ascending pitch order.

    Examples
    --------
    Diatonic in 12-EDO: five wide keys of a whole tone, two narrow of a
    semitone, in the order of the scale's own word ``LLLsLLs``.

    >>> from biotuner.mos.scale import MOSScale
    >>> secs = keyboard_sectors(MOSScale.from_signature(5, 2, tuning=12))
    >>> [round(hi - lo, 4) for lo, hi in secs]
    [1.0472, 1.0472, 1.0472, 0.5236, 1.0472, 1.0472, 0.5236]
    >>> round(secs[0][0], 9), abs(secs[-1][1] - TWO_PI) < 1e-12
    (0.0, True)
    """
    d = _degrees(mode_or_scale)
    edges = [TWO_PI * x for x in d] + [TWO_PI]
    return [(edges[i], edges[i + 1]) for i in range(len(d))]


def phase_to_degree(phase: float, mode_or_scale: Any) -> int:
    """Which key a finger's phase lands on.

    Consistent with :func:`keyboard_sectors` by construction: the phase is
    reduced into ``[0, 2*pi)`` and the key whose half-open sector contains it is
    returned, so a phase sitting exactly on a boundary belongs to the key
    *above* it.

    Boundaries are matched to within :data:`_PHASE_EPS` rather than exactly.
    Rounding down turns "one ulp short of a boundary" into a whole scale step of
    error, and a finger *is* one ulp short of its tone whenever the scale is an
    equal division -- ``exp``/``angle`` and ``2*pi*d`` are different
    computations of the same angle.  Without the tolerance,
    ``to_events(partial(3, 1), MOSScale.from_signature(1, 2, tuning=12))``
    returns ``[0, 1, 1]``.

    Parameters
    ----------
    phase : float
        Radians; any value, reduced modulo ``2*pi``.
    mode_or_scale : MOSScale or Mode

    Returns
    -------
    int
        Degree index into ``mode_or_scale.degrees``.

    Examples
    --------
    Probing the middle of each 12-EDO semitone shows the rounding-down: the
    semitone above a scale tone still reads as that tone.

    >>> from biotuner.mos.scale import MOSScale
    >>> d = MOSScale.from_signature(5, 2, tuning=12)
    >>> [phase_to_degree(TWO_PI * (x + 0.5) / 12, d) for x in range(12)]
    [0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6]

    A hair below the root wraps into the top key, not the root's own -- but
    "a hair" means audibly so, not one ulp:

    >>> phase_to_degree(0.0, d), phase_to_degree(-1e-6, d)
    (0, 6)
    >>> phase_to_degree(-1e-15, d)
    0

    An exact equal division quantises to the identity, ulps notwithstanding:

    >>> edo = MOSScale.from_signature(1, 2, tuning=12)   # 3-EDO, L == s
    >>> [phase_to_degree(p, edo) for p in partial(3, 1).phases]
    [0, 1, 2]
    """
    d = _degrees(mode_or_scale)
    p = float(phase) % TWO_PI
    # A hair below zero reduces to a hair below 2*pi (or, once rounded, to
    # exactly 2*pi): both are the root, not the top key.
    if p >= TWO_PI - _PHASE_EPS:
        return 0
    edges = np.asarray(d, dtype=np.float64) * TWO_PI
    i = int(np.searchsorted(edges, p, side="right")) - 1
    # ...and a hair below any *other* boundary is on that boundary.
    if i + 1 < edges.size and edges[i + 1] - p <= _PHASE_EPS:
        i += 1
    return i


# --------------------------------------------------------------------------- #
# Rendering a play state as notes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NoteEvent:
    """One finger's strike, resolved against a scale.

    Attributes
    ----------
    index : int
        Which finger.  Fingers strike in index order, so this is also the
        event's position in the pulse.
    degree : int
        Scale degree the finger's phase quantised to.
    ratio : float
        That degree as a frequency ratio against the root.
    cents : float
        That degree in cents above the root.
    loudness : float
        ``|f_index|``.
    phase : float
        The finger's raw position in ``[0, 2*pi)``, before quantisation --
        kept because the keyboard is continuous and the residual is audible
        information the degree alone throws away.
    """

    index: int
    degree: int
    ratio: float
    cents: float
    loudness: float
    phase: float


def to_events(state: PlayState, mode_or_scale: Any) -> List[NoteEvent]:
    """Resolve a play state into one note per finger, in striking order.

    The finger count and the scale cardinality are independent -- a 5-finger
    robot can play a 12-note scale -- but they coincide in the case Milne et al.
    Fig. 8 illustrates, and that is where the paper's claim about partials
    lives.

    The claim, and what is actually true
    ------------------------------------
    Fig. 8 shows the first partial sweeping every tone of the scale exactly
    once, in ascending scalar order, and higher partials with ``k`` coprime to
    ``n`` generating complete generic interval cycles.  Under the Fig. 8 key
    layout (:func:`keyboard_sectors`) that is a statement about a *mode*, not
    about a scale:

    * It holds exactly when the mode's step pattern is the Christoffel word
      ``theory.christoffel_word(n_large, n_small)`` -- for ``5L2s`` that is
      ``sLLsLLL`` (Locrian), for ``2L3s`` it is ``ssLsL``.  Propriety is
      sufficient but far from necessary: sweeping the whole coherent generator
      range -- endpoints ``R = 1`` and ``R = 2`` included -- of every signature
      up to 12 notes, the Christoffel mode never fails, and ``5L2s`` keeps
      working out past ``R = 6``.  There is no sharper threshold to quote,
      because the cut-off depends on the signature and not on ``R`` alone --
      ``5L2s`` still works above ``R = 6`` where ``4L5s`` has already broken at
      ``R = 3.25``.
    * In any other mode it fails, and the failure is not subtle: in the
      brightest mode of 12-EDO ``5L2s`` the first partial yields degrees
      ``[0, 0, 1, 2, 3, 4, 5]`` -- the root is struck twice and the leading tone
      never.  Finger 0 sits at phase 0, which is the *bottom* edge of key 0, and
      key 0 is a whole tone wide while the fingers are only ``1/7`` of a turn
      apart, so key 0 catches two of them.

    Both branches are asserted in ``tests/mos/test_fourier.py``.

    Parameters
    ----------
    state : PlayState
    mode_or_scale : MOSScale or Mode
        Uses ``.ratios`` and ``.cents`` when present.  If only one is present
        the other is derived from it (``cents = 1200*log2(ratio)``), so the two
        fields of a :class:`NoteEvent` always agree.  If neither is, both come
        from ``.degrees`` against ``.period`` if the object has one and a 2/1
        period otherwise.

    Returns
    -------
    list of NoteEvent

    Examples
    --------
    >>> from biotuner.mos.scale import MOSScale
    >>> locrian = MOSScale.from_signature(5, 2, tuning=12).mode(6)
    >>> locrian.word
    'sLLsLLL'
    >>> ev = to_events(partial(7, 1), locrian)
    >>> [e.degree for e in ev]
    [0, 1, 2, 3, 4, 5, 6]
    >>> [round(e.cents) for e in ev]
    [0, 100, 300, 500, 600, 800, 1000]

    The third partial reorders the same seven tones into a cycle of thirds:

    >>> [e.degree for e in to_events(partial(7, 3), locrian)]
    [0, 3, 6, 2, 5, 1, 4]

    A non-octave period is carried through, because ``.ratios``/``.cents`` come
    from the scale rather than from a hardcoded 2/1:

    >>> tritave = MOSScale.from_signature(5, 2, tuning=12, period=3.0).mode(6)
    >>> [round(e.cents, 2) for e in to_events(partial(7, 1), tritave)]
    [0.0, 158.5, 475.49, 792.48, 950.98, 1267.97, 1584.96]
    """
    d = _degrees(mode_or_scale)
    ratios = getattr(mode_or_scale, "ratios", None)
    cents = getattr(mode_or_scale, "cents", None)
    ratios = None if ratios is None else [float(x) for x in ratios]
    cents = None if cents is None else [float(x) for x in cents]
    if ratios is None and cents is None:
        # No pitch data at all: fall back to the degrees against whatever
        # period the object declares, or a 2/1 if it declares none.
        period = float(getattr(mode_or_scale, "period", 2.0))
        if not period > 1.0:
            raise ValueError(
                f"a period must be a ratio above 1/1; got period={period!r}"
            )
        ratios = [period**x for x in d]
        cents = [1200.0 * math.log2(period) * x for x in d]
    elif cents is None:
        if any(not r > 0.0 for r in ratios):
            raise ValueError(f"scale ratios must be positive; got {ratios!r}")
        cents = [1200.0 * math.log2(r) for r in ratios]
    elif ratios is None:
        ratios = [2.0 ** (c / 1200.0) for c in cents]
    if len(ratios) != len(d) or len(cents) != len(d):
        raise ValueError(
            f"scale exposes {len(d)} degrees but {len(ratios)} ratios and "
            f"{len(cents)} cents; these must agree"
        )
    mags = state.magnitudes
    phases = state.phases
    out: List[NoteEvent] = []
    for i in range(state.n):
        p = float(phases[i])
        deg = phase_to_degree(p, mode_or_scale)
        out.append(
            NoteEvent(
                index=i,
                degree=deg,
                ratio=ratios[deg],
                cents=cents[deg],
                loudness=float(mags[i]),
                phase=p,
            )
        )
    return out


def to_frequencies(events: Iterable[NoteEvent], fund: float = 250.0) -> List[float]:
    """Sound the events against a fundamental, in hertz.

    Parameters
    ----------
    events : iterable of NoteEvent
    fund : float, default 250.0
        Frequency of the scale root.

    Examples
    --------
    >>> from biotuner.mos.scale import MOSScale
    >>> ev = to_events(partial(5, 1), MOSScale.from_signature(2, 3, tuning=12))
    >>> [round(f, 3) for f in to_frequencies(ev, fund=200.0)]
    [200.0, 224.492, 251.984, 299.661, 336.359]
    """
    if not float(fund) > 0.0:  # not `<= 0`: that lets NaN through
        raise ValueError(f"fundamental frequency must be positive, got {fund!r}")
    return [float(fund) * e.ratio for e in events]


def scratch_sequence(
    state: PlayState,
    k: int,
    values: Iterable[float],
    attr: str = "phase",
) -> List[PlayState]:
    """A trajectory of play states: ``scratch(k, **{attr: v})`` for each ``v``.

    Every state is scratched from the *same* base ``state``, not from its
    predecessor, so ``values`` is a path through the control's absolute range
    and the trajectory is reproducible from any frame -- what an animation or a
    scrubbable UI needs.  With ``attr='rotate'`` that means passing already
    cumulative offsets (``np.linspace(0, 2*pi, 60)``), not per-frame deltas.

    Parameters
    ----------
    state : PlayState
    k : int
        Coefficient to scratch.
    values : iterable of float
    attr : {'phase', 'magnitude', 'scale', 'rotate'}, default 'phase'

    Returns
    -------
    list of PlayState

    Examples
    --------
    Sweeping one coefficient's phase through a full turn returns to the start:

    >>> p = partial(4, 1)
    >>> traj = scratch_sequence(p, 1, [0.0, math.pi, TWO_PI])
    >>> len(traj), traj[0].allclose(traj[-1])
    (3, True)
    >>> [round(float(x), 6) for x in traj[1].phases]
    [3.141593, 4.712389, 0.0, 1.570796]
    """
    if attr not in _SCRATCH_ATTRS:
        raise ValueError(
            f"attr must be one of {_SCRATCH_ATTRS}, got {attr!r}"
        )
    return [state.scratch(k, **{attr: float(v)}) for v in values]
