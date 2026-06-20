"""
Canonical "tuning + acoustic context" descriptor for the biotuner toolbox.

:class:`HarmonicInput` is the unified per-frame harmonic descriptor consumed
by every module that wants a tuning to work from — geometry routines,
the timbre adapter, the engine backend, downstream exporters. Originally
this lived inside :mod:`biotuner.harmonic_geometry.inputs` because geometry
was its only consumer; promoting it to the top level keeps import graphs
clean (modules that aren't doing geometry don't have to depend on
``harmonic_geometry`` just to read the descriptor format).

:class:`HarmonicSequence` is a time-resolved list of :class:`HarmonicInput`
frames, used for animation / morphing pipelines and naturally paired with
the output of ``biotuner.transitional_harmony`` or
``biotuner.harmonic_sequence``.

:data:`SCALE_ATTRS` is the canonical vocabulary mapping HarmonicInput-side
scale labels to ``compute_biotuner`` attribute names — used by
:meth:`HarmonicInput.from_biotuner` to walk scale priorities and populate
:attr:`HarmonicInput.ratios_alternates`.

Backward compatibility: ``from biotuner.harmonic_geometry.inputs import
HarmonicInput`` still works (the old location is now a thin re-export of
this module).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, ClassVar, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

# NOTE: helpers from biotuner.harmonic_geometry._utils are imported LAZILY
# inside the methods that need them — see __post_init__, to_peaks, etc.
# Doing it eagerly here would force importing biotuner.harmonic_geometry,
# whose __init__.py re-imports HarmonicInput from this module via the
# back-compat shim biotuner/harmonic_geometry/inputs.py, creating a
# circular import on first load. Lazy = no cycle.

RatioLike = Union[Fraction, int, float, tuple]


# ---------------------------------------------------------------------------
# Canonical scale vocabulary.
# ---------------------------------------------------------------------------
#
# Mapping of HarmonicInput-side scale labels → biotuner attribute names. The
# left column is what gets stored in :attr:`HarmonicInput.ratios_source` and
# what keys the :attr:`HarmonicInput.ratios_alternates` dict; the right
# column is what's actually looked up on a ``compute_biotuner`` instance.
#
# The order is also the default ``scale_priority`` used by
# :meth:`HarmonicInput.from_biotuner` — first non-empty wins. To add a new
# scale type, add one row here and every downstream consumer can opt into
# it immediately.
SCALE_ATTRS: List[tuple] = [
    # (HarmonicInput key, biotuner attribute name)
    ("peaks_ratios_cons",          "peaks_ratios_cons"),
    ("peaks_ratios",               "peaks_ratios"),
    ("extended_peaks_ratios_cons", "extended_peaks_ratios_cons"),
    ("extended_peaks_ratios",      "extended_peaks_ratios"),
    ("diss_scale",                 "diss_scale"),
    ("HE",                         "HE_scale"),
    ("euler_fokker",               "euler_fokker"),
    ("harm_tuning",                "harm_tuning_scale"),
    ("harm_fit",                   "harm_fit_tuning_scale"),
]

SCALE_KEYS = [k for k, _ in SCALE_ATTRS]
_SCALE_KEY_TO_ATTR = dict(SCALE_ATTRS)


def _get_scale_values(bt: Any, attr_name: str) -> Optional[List[float]]:
    """Return ``bt.<attr_name>`` as a non-empty list of floats, or None.

    Centralised so the (handle-missing / handle-empty / cast-to-float) logic
    is in one place. None means "this scale isn't usable on this bt".
    """
    val = getattr(bt, attr_name, None)
    if val is None:
        return None
    try:
        arr = np.asarray(val, dtype=np.float64).ravel()
    except Exception:
        return None
    if arr.size == 0:
        return None
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        # Ratios must be strictly positive and finite.
        return None
    return [float(r) for r in arr]


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

    # ---------- Tier-A additions ----------------------------------------
    # Per-partial Lorentzian half-widths in Hz. Optional. When present,
    # consumers like Timbre derive decay_times via 1/(π·linewidth).
    linewidths: Optional[List[float]] = None
    # Raw spectrum context. freqs and psd MUST have matching length when
    # either is provided; both can stay None for descriptor-only inputs.
    freqs: Optional[List[float]] = None
    psd: Optional[List[float]] = None
    # Spectrum producer, e.g. 'fft' / 'multitaper' / 'welch'. Free-form
    # string; consumers can match exact values they recognise.
    spectrum_method: Optional[str] = None
    # FOOOF-style 1/f slope from aperiodic component fitting. Maps to
    # Timbre.spectral_tilt.
    aperiodic_exponent: Optional[float] = None
    # Spectral flatness / entropy estimate in [0, 1]. Maps to
    # Timbre.noise_floor.
    spectral_flatness: Optional[float] = None
    # Provenance: which scale source was used for `ratios`. Defaults to
    # "peaks" — the historical implicit behaviour. Set explicitly by
    # `from_biotuner` when the canonical ratios came from a derived
    # scale (e.g. "peaks_ratios_cons", "HE", "euler_fokker").
    ratios_source: str = "peaks"
    # Other available scales from the source biotuner object, keyed by
    # the same vocabulary as `ratios_source` (see SCALE_ATTRS). Lets a
    # downstream tool present a "switch scale" UI without rebuilding the
    # HarmonicInput, or compare scales side-by-side.
    ratios_alternates: Dict[str, List[float]] = field(default_factory=dict)

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
        # Tier-A coercions
        if self.linewidths is not None:
            self.linewidths = [float(l) for l in self.linewidths]
        if self.freqs is not None:
            self.freqs = [float(f) for f in self.freqs]
        if self.psd is not None:
            self.psd = [float(p) for p in self.psd]
        if self.aperiodic_exponent is not None:
            self.aperiodic_exponent = float(self.aperiodic_exponent)
        if self.spectral_flatness is not None:
            self.spectral_flatness = float(self.spectral_flatness)
        # ratios_alternates: ensure values are list[float]; drop empties.
        if self.ratios_alternates:
            cleaned: Dict[str, List[float]] = {}
            for k, v in self.ratios_alternates.items():
                if v is None:
                    continue
                try:
                    arr = [float(x) for x in v]
                except (TypeError, ValueError):
                    continue
                if arr:
                    cleaned[str(k)] = arr
            self.ratios_alternates = cleaned
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
            ("linewidths", self.linewidths),
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

        if self.linewidths is not None and any(l < 0 for l in self.linewidths):
            raise ValueError("All linewidths must be non-negative.")

        # Spectrum: freqs and psd must either both be present or both absent;
        # if present, they must have matching length. NOT tied to n_components.
        if (self.freqs is None) != (self.psd is None):
            raise ValueError(
                "`freqs` and `psd` must be provided together or both omitted."
            )
        if self.freqs is not None and self.psd is not None:
            if len(self.freqs) != len(self.psd):
                raise ValueError(
                    f"`freqs` (len {len(self.freqs)}) and `psd` "
                    f"(len {len(self.psd)}) must have matching length."
                )

        if self.spectral_flatness is not None:
            if not (0.0 <= self.spectral_flatness <= 1.0):
                raise ValueError(
                    f"spectral_flatness must be in [0, 1], got "
                    f"{self.spectral_flatness!r}."
                )

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
    def from_biotuner(
        cls,
        bt: Any,
        equave: float = 2.0,
        *,
        scale_priority: Optional[Sequence[str]] = None,
        include_alternates: bool = True,
        include_spectrum: bool = True,
        include_fooof: bool = True,
    ) -> "HarmonicInput":
        """Build from a fitted :class:`compute_biotuner` instance.

        Two selection modes for the canonical ``ratios``:

        * **Legacy mode** (``scale_priority`` is ``None``, the default) —
          reproduces the historical behaviour exactly: pulls ``peaks``
          (required), ``amps`` when length-aligned, and uses
          ``peaks_ratios`` as the canonical ratios when it's 1:1 aligned
          with peaks. ``ratios_source`` is set to ``"peaks"`` and
          ``ratios_alternates`` is left empty so existing callers see no
          schema change.

        * **Scale-priority mode** (``scale_priority`` provided) — walks
          the priority list in order (using the canonical names from
          :data:`SCALE_KEYS`), picks the first non-empty scale as
          ``ratios``, records its label in ``ratios_source``. When
          ``include_alternates=True`` (default), every other non-empty
          scale found on ``bt`` is stored in ``ratios_alternates`` under
          the same canonical name. Amplitudes / linewidths are populated
          only if their lengths align with the chosen scale.

        Tier-A spectral context is populated regardless of mode:

        * ``freqs`` / ``psd`` / ``spectrum_method`` (when ``include_spectrum``)
        * ``aperiodic_exponent`` / ``linewidths`` (when ``include_fooof``)
        * ``spectral_flatness`` (always, if available)

        Parameters
        ----------
        bt : compute_biotuner
            A fitted biotuner object on which ``peaks_extraction`` has
            been called.
        equave : float, default=2.0
        scale_priority : sequence of str, optional
            Ordered preference for ``ratios``. Values must come from
            :data:`SCALE_KEYS`. Unknown keys raise :class:`ValueError`.
        include_alternates : bool, default=True
            In scale-priority mode, populate ``ratios_alternates`` with
            every other non-empty scale found on ``bt``. Ignored in
            legacy mode.
        include_spectrum : bool, default=True
            Copy ``bt.freqs`` / ``bt.psd`` / ``bt.spectrum_method`` if
            available.
        include_fooof : bool, default=True
            Copy ``bt.aperiodic_exponent`` and a length-aligned
            ``bt.linewidth`` if available.

        Raises
        ------
        AttributeError
            If ``bt`` lacks a ``peaks`` attribute.
        ValueError
            If ``bt.peaks`` is empty, or ``scale_priority`` contains an
            unknown key.
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
        amps_list: Optional[List[float]] = None
        amps_from_db = False
        if amps_attr is not None:
            amps_arr = np.asarray(amps_attr, dtype=np.float64).ravel()
            if amps_arr.size == len(peaks):
                # `amplitudes` is contracted to be linear and non-negative (see
                # the dataclass docstring, validate(), and normalize_amplitudes,
                # which divides by the sum and treats a non-positive sum as
                # "no information"). compute_biotuner stores the spectrum in dB
                # (psd = 10*log10(power); see compute_peaks_ts), so peak
                # amplitudes read off it are routinely negative. A negative
                # value is impossible for a linear magnitude, so we treat the
                # whole array as dB power and convert it back to linear power
                # (10**(dB/10)): this is non-negative, preserves the relative
                # loudness ordering, and normalises sensibly. Non-finite amps
                # are unusable and dropped (downstream falls back to uniform).
                if not np.all(np.isfinite(amps_arr)):
                    warnings.warn(
                        "Dropping bt.amps: contains non-finite values; "
                        "downstream consumers will use uniform amplitudes.",
                        stacklevel=2,
                    )
                elif np.any(amps_arr < 0):
                    amps_list = [float(10.0 ** (a / 10.0)) for a in amps_arr]
                    amps_from_db = True
                else:
                    amps_list = [float(a) for a in amps_arr]

        base_freq = float(min(peaks))
        metadata = {"source": "compute_biotuner"}
        if amps_from_db:
            # Record that amplitudes were converted from biotuner's dB spectrum.
            metadata["amplitudes_scale"] = "linear_from_db"

        # ---------------- Canonical ratios selection ----------------------
        canonical_ratios: Optional[List[Union[Fraction, float]]] = None
        canonical_peaks: Optional[List[float]] = peaks
        ratios_source = "peaks"
        ratios_alternates: Dict[str, List[float]] = {}
        canonical_n = len(peaks)

        if scale_priority is None:
            # Legacy path — identical behaviour to the pre-extension code.
            ratios_attr = getattr(bt, "peaks_ratios", None)
            if ratios_attr is not None:
                ratios_arr = np.asarray(ratios_attr).ravel()
                if ratios_arr.size == len(peaks):
                    canonical_ratios = coerce_ratios(
                        [float(r) for r in ratios_arr]
                    )
                    canonical_peaks = None  # avoid consistency conflict
        else:
            # Scale-priority path — walk SCALE_ATTRS in user order.
            unknown = [k for k in scale_priority if k not in _SCALE_KEY_TO_ATTR]
            if unknown:
                raise ValueError(
                    f"Unknown scale_priority keys: {unknown}. "
                    f"Valid keys: {SCALE_KEYS}."
                )
            chosen_key = None
            chosen_vals = None
            for key in scale_priority:
                attr = _SCALE_KEY_TO_ATTR[key]
                vals = _get_scale_values(bt, attr)
                if vals is not None:
                    chosen_key = key
                    chosen_vals = vals
                    break
            if chosen_vals is not None:
                # Adopt the chosen scale as canonical ratios. If it
                # aligns with peaks 1:1, also clear `peaks` to avoid the
                # base_freq * ratios consistency check (same trick as the
                # legacy path).
                canonical_ratios = coerce_ratios(chosen_vals)
                ratios_source = chosen_key
                canonical_n = len(chosen_vals)
                if len(chosen_vals) == len(peaks):
                    canonical_peaks = None
                else:
                    canonical_peaks = None  # only ratios are canonical
                # Amps only travel if they happen to match the chosen scale.
                if amps_list is not None and len(amps_list) != canonical_n:
                    amps_list = None

            if include_alternates:
                for key, attr in SCALE_ATTRS:
                    if key == ratios_source:
                        continue
                    vals = _get_scale_values(bt, attr)
                    if vals is not None:
                        ratios_alternates[key] = vals

        # ---------------- Tier-A spectral context -------------------------
        freqs_list: Optional[List[float]] = None
        psd_list: Optional[List[float]] = None
        spectrum_method: Optional[str] = None
        if include_spectrum:
            f_attr = getattr(bt, "freqs", None)
            p_attr = getattr(bt, "psd", None)
            if f_attr is not None and p_attr is not None:
                try:
                    f_arr = np.asarray(f_attr, dtype=np.float64).ravel()
                    p_arr = np.asarray(p_attr, dtype=np.float64).ravel()
                    if f_arr.size > 0 and f_arr.size == p_arr.size:
                        freqs_list = f_arr.tolist()
                        psd_list = p_arr.tolist()
                except Exception:
                    pass
            sm_attr = getattr(bt, "spectrum_method", None)
            if sm_attr is not None:
                try:
                    spectrum_method = str(sm_attr)
                except Exception:
                    spectrum_method = None

        # ---------------- FOOOF -------------------------------------------
        aperiodic: Optional[float] = None
        linewidths_list: Optional[List[float]] = None
        if include_fooof:
            ap_attr = getattr(bt, "aperiodic_exponent", None)
            if ap_attr is None:
                params = getattr(bt, "aperiodic_params", None)
                if params is not None:
                    try:
                        if len(params) >= 2:
                            ap_attr = float(params[1])
                    except (TypeError, IndexError):
                        pass
            if ap_attr is not None:
                try:
                    v = float(ap_attr)
                    if np.isfinite(v):
                        aperiodic = v
                except (TypeError, ValueError):
                    pass
            # linewidths: only attach when length matches canonical n
            for cand in ("linewidth", "peaks_linewidth", "fooof_linewidth"):
                lw_attr = getattr(bt, cand, None)
                if lw_attr is None:
                    continue
                try:
                    lw_arr = np.asarray(lw_attr, dtype=np.float64).ravel()
                except Exception:
                    continue
                if lw_arr.size == canonical_n and np.all(np.isfinite(lw_arr)):
                    linewidths_list = lw_arr.tolist()
                    break

        # ---------------- Spectral flatness / noise estimate --------------
        sf_attr = getattr(bt, "spectral_flatness", None)
        if sf_attr is None:
            sf_attr = getattr(bt, "spectral_entropy", None)
        spectral_flatness: Optional[float] = None
        if sf_attr is not None:
            try:
                v = float(sf_attr)
                if np.isfinite(v):
                    spectral_flatness = float(np.clip(v, 0.0, 1.0))
            except (TypeError, ValueError):
                pass

        return cls(
            ratios=canonical_ratios,
            peaks=canonical_peaks,
            amplitudes=amps_list,
            base_freq=base_freq,
            equave=equave,
            metadata=metadata,
            linewidths=linewidths_list,
            freqs=freqs_list,
            psd=psd_list,
            spectrum_method=spectrum_method,
            aperiodic_exponent=aperiodic,
            spectral_flatness=spectral_flatness,
            ratios_source=ratios_source,
            ratios_alternates=ratios_alternates,
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


# ---------------------------------------------------------------------------
# Deferred helper imports.
# ---------------------------------------------------------------------------
#
# Done at the BOTTOM of the file (after the class definitions) to break a
# circular-import trap: doing this at the top would trigger
# ``biotuner.harmonic_geometry.__init__`` to load, which in turn re-imports
# HarmonicInput via the back-compat shim at
# ``biotuner.harmonic_geometry.inputs``. By the time we hit this import the
# class is already defined and registered in this module's globals, so the
# round-trip succeeds. Method bodies look up these names via the module's
# globals at call time, so the order doesn't affect runtime correctness.
from biotuner.harmonic_geometry._utils import (  # noqa: E402
    coerce_ratios,
    normalize_amplitudes,
    ratios_to_floats,
)
