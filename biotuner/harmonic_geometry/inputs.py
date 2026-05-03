"""
Input dataclasses for :mod:`biotuner.harmonic_geometry`.

:class:`HarmonicInput` is the unified per-frame harmonic descriptor consumed
by every geometry-producing function. :class:`HarmonicSequence` is a
time-resolved list of :class:`HarmonicInput` frames, used for animation /
morphing pipelines and naturally paired with the output of
``biotuner.transitional_harmony`` or ``biotuner.harmonic_sequence``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from biotuner.harmonic_geometry._utils import (
    coerce_ratios,
    normalize_amplitudes,
    ratios_to_floats,
)

RatioLike = Union[Fraction, int, float, tuple]


@dataclass
class HarmonicInput:
    """Unified harmonic input.

    At least one of ``ratios`` or ``peaks`` must be provided. All list-typed
    optional fields, if given, must have lengths matching the number of
    components (``len(ratios)`` or ``len(peaks)``).

    Parameters
    ----------
    ratios : list of Fraction or float, optional
        Frequency ratios. Coerced to :class:`~fractions.Fraction` when a
        rational approximation within :data:`DEFAULT_MAX_DENOMINATOR` exists.
    peaks : list of float, optional
        Peak frequencies in Hz.
    amplitudes : list of float, optional
        Linear (not dB) amplitudes per component. Defaults to uniform.
    phases : list of float, optional
        Phase per component, in radians. Defaults to zeros.
    damping : list of float, optional
        Decay rate (1/s) per component. Defaults to zeros.
    base_freq : float, default=1.0
        Reference frequency in Hz. Used when ``ratios`` are given without
        ``peaks`` to recover absolute frequencies.
    equave : float, default=2.0
        Equave width: ``2.0`` for octaves, ``3.0`` for tritaves, etc. Must
        be greater than 1.
    metadata : dict
        Free-form annotations preserved across constructors and
        validation.

    Notes
    -----
    Validation is invoked by :meth:`__post_init__`; constructing an
    inconsistent :class:`HarmonicInput` raises :class:`ValueError`.
    """

    ratios: Optional[List[Union[Fraction, float]]] = None
    peaks: Optional[List[float]] = None
    amplitudes: Optional[List[float]] = None
    phases: Optional[List[float]] = None
    damping: Optional[List[float]] = None
    base_freq: float = 1.0
    equave: float = 2.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Tolerance (relative) when checking peaks vs ratios * base_freq.
    _CONSISTENCY_RTOL: ClassVar[float] = 1e-3

    def __post_init__(self) -> None:
        # Coerce ratios eagerly so all downstream code sees a uniform type.
        if self.ratios is not None:
            self.ratios = coerce_ratios(self.ratios)
        if self.peaks is not None:
            self.peaks = [float(p) for p in self.peaks]
        if self.amplitudes is not None:
            self.amplitudes = [float(a) for a in self.amplitudes]
        if self.phases is not None:
            self.phases = [float(p) for p in self.phases]
        if self.damping is not None:
            self.damping = [float(d) for d in self.damping]
        self.validate()

    # ------------------------------------------------------------- properties

    def n_components(self) -> int:
        """Return the number of harmonic components.

        Resolved from ``ratios`` first, then ``peaks``. At least one of the
        two is guaranteed to be present after :meth:`validate`.
        """
        if self.ratios is not None:
            return len(self.ratios)
        if self.peaks is not None:
            return len(self.peaks)
        return 0  # unreachable — validate() rejects this case

    # -------------------------------------------------------------- accessors

    def to_peaks(self) -> np.ndarray:
        """Return absolute peak frequencies in Hz as a 1-D ``float64`` array.

        If ``peaks`` is set, those values are returned directly. Otherwise
        peaks are reconstructed as ``base_freq * ratios``.
        """
        if self.peaks is not None:
            return np.asarray(self.peaks, dtype=np.float64)
        return self.base_freq * ratios_to_floats(self.ratios)

    def to_ratios(self) -> List[Union[Fraction, float]]:
        """Return ratios.

        If ``ratios`` is set, those values are returned. Otherwise ratios are
        derived as ``peaks / base_freq`` and coerced to :class:`Fraction`.
        """
        if self.ratios is not None:
            return list(self.ratios)
        return coerce_ratios([p / self.base_freq for p in self.peaks])

    def normalized_amplitudes(self) -> np.ndarray:
        """Return amplitudes scaled to sum to 1.

        If ``amplitudes`` is ``None``, returns a uniform distribution over
        the components.
        """
        if self.amplitudes is None:
            n = self.n_components()
            if n == 0:
                return np.array([], dtype=np.float64)
            return np.full(n, 1.0 / n, dtype=np.float64)
        return normalize_amplitudes(self.amplitudes)

    # -------------------------------------------------------------- validate

    def validate(self) -> None:
        """Raise :class:`ValueError` if the input is internally inconsistent.

        Checked invariants:

        * at least one of ``ratios`` / ``peaks`` is given,
        * ``equave > 1`` and ``base_freq > 0``,
        * all list-typed fields have matching lengths,
        * all amplitudes are non-negative,
        * all peaks are positive,
        * if both ``ratios`` and ``peaks`` are given, they agree up to
          ``base_freq`` within a small relative tolerance.
        """
        if self.ratios is None and self.peaks is None:
            raise ValueError(
                "HarmonicInput requires at least one of `ratios` or `peaks`."
            )
        if not (self.equave > 1.0):
            raise ValueError(f"equave must be > 1, got {self.equave!r}.")
        if not (self.base_freq > 0.0):
            raise ValueError(f"base_freq must be > 0, got {self.base_freq!r}.")

        n = self.n_components()

        for name, seq in (
            ("ratios", self.ratios),
            ("peaks", self.peaks),
            ("amplitudes", self.amplitudes),
            ("phases", self.phases),
            ("damping", self.damping),
        ):
            if seq is None:
                continue
            if len(seq) != n:
                raise ValueError(
                    f"{name!r} has length {len(seq)} but expected {n} "
                    "(must match n_components)."
                )

        if self.amplitudes is not None and any(a < 0 for a in self.amplitudes):
            raise ValueError("All amplitudes must be non-negative.")

        if self.peaks is not None and any(p <= 0 for p in self.peaks):
            raise ValueError("All peaks must be strictly positive.")

        if self.ratios is not None and self.peaks is not None:
            ratios_f = ratios_to_floats(self.ratios)
            peaks_f = np.asarray(self.peaks, dtype=np.float64)
            implied_peaks = self.base_freq * ratios_f
            if not np.allclose(
                implied_peaks, peaks_f, rtol=self._CONSISTENCY_RTOL, atol=0.0
            ):
                raise ValueError(
                    "ratios and peaks are inconsistent given base_freq="
                    f"{self.base_freq}. "
                    f"base_freq * ratios = {implied_peaks.tolist()}, "
                    f"but peaks = {peaks_f.tolist()}."
                )

    # ------------------------------------------------------------ constructors

    @classmethod
    def from_ratios(
        cls,
        ratios: Iterable[RatioLike],
        base_freq: float = 1.0,
        amplitudes: Optional[Sequence[float]] = None,
        phases: Optional[Sequence[float]] = None,
        damping: Optional[Sequence[float]] = None,
        equave: float = 2.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "HarmonicInput":
        """Build from a sequence of ratios."""
        return cls(
            ratios=list(ratios),
            base_freq=base_freq,
            amplitudes=list(amplitudes) if amplitudes is not None else None,
            phases=list(phases) if phases is not None else None,
            damping=list(damping) if damping is not None else None,
            equave=equave,
            metadata=dict(metadata) if metadata else {},
        )

    @classmethod
    def from_peaks(
        cls,
        peaks: Iterable[float],
        base_freq: Optional[float] = None,
        amplitudes: Optional[Sequence[float]] = None,
        phases: Optional[Sequence[float]] = None,
        damping: Optional[Sequence[float]] = None,
        equave: float = 2.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "HarmonicInput":
        """Build from a sequence of peak frequencies in Hz.

        If ``base_freq`` is ``None`` it is set to the smallest peak so that
        the implied ratio of the lowest component is exactly 1.
        """
        peaks_list = [float(p) for p in peaks]
        if not peaks_list:
            raise ValueError("`peaks` must be non-empty.")
        if base_freq is None:
            base_freq = float(min(peaks_list))
        return cls(
            peaks=peaks_list,
            base_freq=float(base_freq),
            amplitudes=list(amplitudes) if amplitudes is not None else None,
            phases=list(phases) if phases is not None else None,
            damping=list(damping) if damping is not None else None,
            equave=equave,
            metadata=dict(metadata) if metadata else {},
        )

    @classmethod
    def from_biotuner(cls, bt: Any, equave: float = 2.0) -> "HarmonicInput":
        """Build from a fitted :class:`compute_biotuner` (a.k.a. BiotunerObject) instance.

        Pulls ``peaks`` (required), ``amps``, and ``peaks_ratios`` when
        available. Phase / damping are not currently exposed by biotuner and
        are left at their defaults.

        Parameters
        ----------
        bt : compute_biotuner
            A biotuner object on which ``peaks_extraction`` has been called.
        equave : float, default=2.0

        Raises
        ------
        AttributeError
            If ``bt`` lacks a ``peaks`` attribute.
        ValueError
            If ``bt.peaks`` is empty.
        """
        if not hasattr(bt, "peaks"):
            raise AttributeError(
                "Object passed to from_biotuner has no `peaks` attribute. "
                "Did you call peaks_extraction()?"
            )
        peaks = list(np.asarray(bt.peaks).ravel())
        if len(peaks) == 0:
            raise ValueError("Biotuner object has no peaks (empty array).")

        amps_attr = getattr(bt, "amps", None)
        amplitudes: Optional[List[float]] = None
        if amps_attr is not None:
            amps_arr = np.asarray(amps_attr).ravel()
            if amps_arr.size == len(peaks):
                amplitudes = [float(a) for a in amps_arr]

        ratios_attr = getattr(bt, "peaks_ratios", None)
        ratios: Optional[List[Union[Fraction, float]]] = None
        # peaks_ratios is computed from sorted unique pairs and may not be
        # 1:1 aligned with peaks. Only use it when lengths match; otherwise
        # let HarmonicInput derive ratios from peaks via base_freq.
        if ratios_attr is not None:
            ratios_arr = np.asarray(ratios_attr).ravel()
            if ratios_arr.size == len(peaks):
                ratios = coerce_ratios([float(r) for r in ratios_arr])

        base_freq = float(min(peaks))
        metadata = {"source": "compute_biotuner"}

        return cls(
            ratios=ratios,
            peaks=peaks if ratios is None else None,  # avoid consistency conflict
            amplitudes=amplitudes,
            base_freq=base_freq,
            equave=equave,
            metadata=metadata,
        )


@dataclass
class HarmonicSequence:
    """Time-resolved sequence of :class:`HarmonicInput` frames.

    Pairs naturally with the output of ``biotuner.transitional_harmony`` and
    ``biotuner.harmonic_sequence``: each window's peaks become a frame here,
    and downstream geometry functions can be applied frame-by-frame via
    ``transformations.geometry_sequence``.

    Parameters
    ----------
    frames : list of HarmonicInput
        At least one frame is required.
    times : ndarray, optional
        Time of each frame in seconds. If ``None``, frames are assumed to
        be uniformly spaced at unit intervals.
    """

    frames: List[HarmonicInput]
    times: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if not isinstance(self.frames, list):
            self.frames = list(self.frames)
        if len(self.frames) == 0:
            raise ValueError("HarmonicSequence requires at least one frame.")
        if self.times is not None:
            self.times = np.asarray(self.times, dtype=np.float64)
            if self.times.ndim != 1:
                raise ValueError("times must be a 1-D array.")
            if self.times.shape[0] != len(self.frames):
                raise ValueError(
                    f"times has length {self.times.shape[0]} but there are "
                    f"{len(self.frames)} frames."
                )
            if not np.all(np.diff(self.times) >= 0):
                raise ValueError("times must be non-decreasing.")

    # -------------------------------------------------------------- accessors

    def n_frames(self) -> int:
        """Number of frames in the sequence."""
        return len(self.frames)

    def _resolve_times(self) -> np.ndarray:
        if self.times is not None:
            return self.times
        return np.arange(len(self.frames), dtype=np.float64)

    def at(self, t: float) -> HarmonicInput:
        """Return the frame nearest to time ``t``."""
        times = self._resolve_times()
        idx = int(np.argmin(np.abs(times - float(t))))
        return self.frames[idx]

    def interpolate(self, t: float, mode: str = "log") -> HarmonicInput:
        """Return a :class:`HarmonicInput` interpolated to time ``t``.

        Currently supports two-frame interpolation between the bracketing
        frames. ``mode`` selects the space in which ratios / peaks are
        blended:

        * ``'log'`` (default) — logarithmic, musically correct,
        * ``'linear'`` — straight linear interpolation,
        * ``'nearest'`` — return the nearest frame (no blending).

        Frames must have equal ``n_components``; if they do not, this
        raises :class:`ValueError`. Richer interpolation (mismatched
        component counts, phase wrapping, etc.) is the job of
        ``transformations.interpolate_input`` in Phase 6.
        """
        if mode not in {"log", "linear", "nearest"}:
            raise ValueError(
                f"mode must be 'log', 'linear', or 'nearest'; got {mode!r}."
            )
        times = self._resolve_times()
        t = float(t)

        if mode == "nearest" or len(self.frames) == 1:
            return self.at(t)
        if t <= times[0]:
            return self.frames[0]
        if t >= times[-1]:
            return self.frames[-1]

        # Bracket: largest i such that times[i] <= t < times[i + 1].
        i = int(np.searchsorted(times, t, side="right") - 1)
        i = max(0, min(i, len(self.frames) - 2))
        t0, t1 = float(times[i]), float(times[i + 1])
        if t1 == t0:
            return self.frames[i]
        alpha = (t - t0) / (t1 - t0)

        a, b = self.frames[i], self.frames[i + 1]
        if a.n_components() != b.n_components():
            raise ValueError(
                "Bracketing frames have different n_components "
                f"({a.n_components()} vs {b.n_components()}); component-count "
                "mismatch handling lives in transformations.interpolate_input."
            )

        peaks_a = a.to_peaks()
        peaks_b = b.to_peaks()
        if mode == "log":
            blended_peaks = np.exp(
                (1 - alpha) * np.log(peaks_a) + alpha * np.log(peaks_b)
            )
        else:  # linear
            blended_peaks = (1 - alpha) * peaks_a + alpha * peaks_b

        amps_a = a.normalized_amplitudes()
        amps_b = b.normalized_amplitudes()
        amps = (1 - alpha) * amps_a + alpha * amps_b

        # Phases interpolate linearly without wrapping; renderers that care
        # about wrapping should rewrap on read.
        phases_a = (
            np.asarray(a.phases, dtype=np.float64)
            if a.phases is not None
            else np.zeros(a.n_components())
        )
        phases_b = (
            np.asarray(b.phases, dtype=np.float64)
            if b.phases is not None
            else np.zeros(b.n_components())
        )
        phases = (1 - alpha) * phases_a + alpha * phases_b

        return HarmonicInput.from_peaks(
            blended_peaks.tolist(),
            base_freq=float(min(blended_peaks)),
            amplitudes=amps.tolist(),
            phases=phases.tolist(),
            equave=a.equave,
            metadata={"interpolated_from": (i, i + 1), "alpha": alpha, "mode": mode},
        )

    # ------------------------------------------------------------ constructors

    @classmethod
    def from_biotuner_list(
        cls,
        bt_list: Sequence[Any],
        times: Optional[Sequence[float]] = None,
        equave: float = 2.0,
    ) -> "HarmonicSequence":
        """Build from a sequence of fitted ``compute_biotuner`` objects.

        Each biotuner object becomes one frame. Objects with no peaks are
        skipped; if every object is empty, :class:`ValueError` is raised.

        Parameters
        ----------
        bt_list : sequence of compute_biotuner
            Fitted biotuner objects (``peaks_extraction`` already called).
        times : sequence of float, optional
            One time per *kept* frame. If ``None``, frames are uniformly
            spaced. If provided, must match the number of non-empty
            biotuner objects.
        equave : float, default=2.0
        """
        frames: List[HarmonicInput] = []
        kept_indices: List[int] = []
        for i, bt in enumerate(bt_list):
            peaks = getattr(bt, "peaks", None)
            if peaks is None or len(np.asarray(peaks).ravel()) == 0:
                continue
            frames.append(HarmonicInput.from_biotuner(bt, equave=equave))
            kept_indices.append(i)

        if not frames:
            raise ValueError(
                "No biotuner objects in bt_list yielded any peaks; cannot "
                "build a HarmonicSequence."
            )

        times_arr: Optional[np.ndarray] = None
        if times is not None:
            times_arr = np.asarray(list(times), dtype=np.float64)
            if times_arr.shape[0] != len(frames):
                raise ValueError(
                    f"times has length {times_arr.shape[0]} but {len(frames)} "
                    "non-empty frames were collected."
                )
        return cls(frames=frames, times=times_arr)

    @classmethod
    def from_biotuner_group(
        cls,
        btg: Any,
        times: Optional[Sequence[float]] = None,
        equave: float = 2.0,
    ) -> "HarmonicSequence":
        """Build from a :class:`biotuner.biotuner_group.BiotunerGroup` instance.

        Uses ``btg.objects`` (the per-series ``compute_biotuner`` instances).
        ``BiotunerGroup`` must have been constructed with
        ``store_objects=True`` (the default).

        Parameters
        ----------
        btg : BiotunerGroup
            A group whose ``compute_peaks`` has been run.
        times : sequence of float, optional
            One time per non-empty frame; see :meth:`from_biotuner_list`.
        equave : float, default=2.0

        Raises
        ------
        AttributeError
            If ``btg`` has no ``objects`` attribute, or it is ``None``
            (e.g. ``store_objects=False`` was used).
        """
        objects = getattr(btg, "objects", None)
        if objects is None:
            raise AttributeError(
                "BiotunerGroup has no `objects` available. Did you construct "
                "it with store_objects=True and run compute_peaks()?"
            )
        return cls.from_biotuner_list(objects, times=times, equave=equave)
