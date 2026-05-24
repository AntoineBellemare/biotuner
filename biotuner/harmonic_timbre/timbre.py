"""Core data structures for ``biotuner.harmonic_timbre``.

This module defines :class:`Modulator`, :class:`Timbre`, and
:class:`TimbreSequence`. They are deliberately richer than a flat
``(partials, amplitudes)`` pair so that Biotuner's analytical depth can
survive into synthesis and structured-format exporters.

Phase 1 implements the v1 fields and methods. The v1.1 fields
(``detuning``, ``am_modulators``, ``fm_modulators``, ``layers``,
``envelopes``, ``rhythm``) are present in the dataclass shape but
their consumers ship in Phase 3.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict, fields
from typing import Any, Iterable

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Modulator
# ---------------------------------------------------------------------------

@dataclass
class Modulator:
    """A single modulation routing (amplitude or frequency).

    Parameters
    ----------
    carrier_idx : int
        Index into ``Timbre.partials_hz`` of the partial being modulated.
    mod_freq : float
        Modulation frequency in Hz.
    depth : float
        ``0..1`` for AM (depth of amplitude modulation); deviation in Hz
        for FM.
    mod_type : str
        ``'AM'`` or ``'FM'``.
    phase : float, default=0.0
        Initial phase of the modulator, radians.
    source : str, default=''
        Provenance tag (e.g. ``'PAC_theta_gamma'``).
    """

    carrier_idx: int
    mod_freq: float
    depth: float
    mod_type: str
    phase: float = 0.0
    source: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Modulator":
        return cls(**d)


# ---------------------------------------------------------------------------
# Timbre
# ---------------------------------------------------------------------------

# Centralized list of array-valued fields, used for save/load and validate.
_ARRAY_FIELDS = (
    "partials_hz",
    "amplitudes",
    "phases",
    "decay_times",
    "bandwidths",
    "detuning",
    "layer_mix",
)

# Fields that, when set, must match ``len(partials_hz)``.
_PER_PARTIAL_FIELDS = ("amplitudes", "phases", "decay_times", "bandwidths", "detuning")


@dataclass
class Timbre:
    """A spectrum specification, optionally matched to a tuning.

    Required fields:
        ``partials_hz``, ``amplitudes``

    All other fields are optional; ``None`` means "not specified, use the
    default in synthesis".
    """

    # --- required ---
    partials_hz: np.ndarray
    amplitudes: np.ndarray

    # --- v1 spectral ---
    phases: np.ndarray | None = None
    decay_times: np.ndarray | None = None
    bandwidths: np.ndarray | None = None
    spectral_tilt: float | None = None
    noise_floor: float | None = None

    # --- v1.1 dynamics (declared now; consumed in Phase 3) ---
    detuning: np.ndarray | None = None
    am_modulators: list[Modulator] = field(default_factory=list)
    fm_modulators: list[Modulator] = field(default_factory=list)
    layers: list["Timbre"] | None = None
    layer_mix: np.ndarray | None = None
    envelopes: dict[str, np.ndarray] = field(default_factory=dict)
    envelope_rate: float = 100.0

    # --- v1 cross-modal sidecar ---
    palette: list[tuple[int, int, int]] = field(default_factory=list)
    elements: list[str] = field(default_factory=list)
    geometry_signature: dict | None = None

    # --- v1.1 rhythm bridge (consumed in Phase 3) ---
    rhythm: Any | None = None

    # --- metadata ---
    base_freq: float = 1.0
    matched_tuning: list | None = None
    matching_method: str = ""
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        # Coerce required arrays to numpy float64
        object.__setattr__(self, "partials_hz", np.asarray(self.partials_hz, dtype=np.float64))
        object.__setattr__(self, "amplitudes", np.asarray(self.amplitudes, dtype=np.float64))
        for fname in ("phases", "decay_times", "bandwidths", "detuning", "layer_mix"):
            v = getattr(self, fname)
            if v is not None:
                object.__setattr__(self, fname, np.asarray(v, dtype=np.float64))

    def n_partials(self) -> int:
        return int(self.partials_hz.shape[0])

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> None:
        n = self.n_partials()
        if n == 0:
            raise ValueError("Timbre has no partials")
        if not np.all(np.isfinite(self.partials_hz)):
            raise ValueError("partials_hz contains non-finite values")
        if not np.all(self.partials_hz > 0):
            raise ValueError("partials_hz must be strictly positive")
        if self.amplitudes.shape[0] != n:
            raise ValueError(
                f"amplitudes length {self.amplitudes.shape[0]} != n_partials {n}"
            )
        if not np.all(np.isfinite(self.amplitudes)):
            raise ValueError("amplitudes contains non-finite values")
        for fname in _PER_PARTIAL_FIELDS:
            if fname == "amplitudes":
                continue
            v = getattr(self, fname)
            if v is None:
                continue
            if v.shape[0] != n:
                raise ValueError(f"{fname} length {v.shape[0]} != n_partials {n}")
        if self.spectral_tilt is not None and not np.isfinite(self.spectral_tilt):
            raise ValueError("spectral_tilt must be finite when set")
        if self.noise_floor is not None and not (0.0 <= self.noise_floor <= 1.0):
            raise ValueError("noise_floor must be in [0, 1] when set")
        if self.layers is not None and self.layer_mix is not None:
            if self.layer_mix.shape[0] != len(self.layers):
                raise ValueError("layer_mix length must match number of layers")
        # cross-check rhythm/timbre tuning when both available (Phase 3 hook)
        if self.rhythm is not None and self.matched_tuning is not None:
            r_ratios = getattr(self.rhythm, "ratios", None)
            if r_ratios is not None and len(r_ratios) > 0:
                a = np.asarray(self.matched_tuning, dtype=np.float64)
                b = np.asarray(r_ratios, dtype=np.float64)
                if a.shape == b.shape and not np.allclose(a, b, rtol=1e-3):
                    logger.warning(
                        "Timbre.matched_tuning and Timbre.rhythm.ratios differ; "
                        "the time-scale projection invariant may be broken"
                    )

    def normalized_amplitudes(self, *, peak: float = 1.0) -> np.ndarray:
        m = float(np.max(np.abs(self.amplitudes))) if self.amplitudes.size else 0.0
        if m <= 0:
            return self.amplitudes.copy()
        return self.amplitudes * (peak / m)

    # ------------------------------------------------------------------
    # HarmonicInput → Timbre adapter
    # ------------------------------------------------------------------
    @classmethod
    def from_harmonic_input(cls, hi: Any, **overrides) -> "Timbre":
        """Build a :class:`Timbre` from a :class:`HarmonicInput` descriptor.

        Direct mapping — Phase-1 fields only:

        ============================  ============================
        ``HarmonicInput`` field        ``Timbre`` field
        ============================  ============================
        ``peaks`` (or ``ratios``)      ``partials_hz``
        ``amplitudes`` (or uniform)    ``amplitudes``
        ``phases``                     ``phases``
        ``linewidths``                 ``decay_times`` (1/π·lw), ``bandwidths``
        ``aperiodic_exponent``         ``spectral_tilt``
        ``spectral_flatness``          ``noise_floor``
        ``ratios``                     ``matched_tuning``
        ``base_freq``                  ``base_freq``
        ``ratios_source``              ``metadata['scale_source']``
        ============================  ============================

        Phase-3 fields (``am_modulators``, ``fm_modulators``, IMF layers,
        etc.) are NOT populated by this method — those need biotuner-
        only data and live on :meth:`Timbre.from_biotuner` via
        ``attach_*`` augmentation. Use this method when you only have a
        ``HarmonicInput`` (e.g. loaded from a saved analysis) and want
        a complete, playable Phase-1 timbre.

        Any keyword argument in ``overrides`` replaces the corresponding
        field after the HI-derived defaults are computed — same shape
        as the :class:`Timbre` constructor, useful for tweaking just
        one or two fields without losing the HI mapping for the rest.
        """
        # Resolve partials. Prefer hi.peaks when present (absolute Hz);
        # else reconstruct from base_freq * ratios via the public accessor.
        partials = np.asarray(hi.to_peaks(), dtype=np.float64)
        n = partials.shape[0]
        if n == 0:
            raise ValueError("HarmonicInput has no components; cannot build Timbre.")

        # Amplitudes: prefer hi.amplitudes when supplied, else uniform.
        if hi.amplitudes is not None:
            amps = np.asarray(hi.amplitudes, dtype=np.float64)
        else:
            amps = np.full(n, 1.0 / n, dtype=np.float64)

        # Phases: forward when present.
        phases = (
            np.asarray(hi.phases, dtype=np.float64)
            if hi.phases is not None
            else None
        )

        # Linewidths → (decay_times, bandwidths). decay = 1 / (π · lw),
        # clipped to a finite ceiling for zero/negative linewidths.
        decay_times = None
        bandwidths = None
        if hi.linewidths is not None:
            lw_arr = np.asarray(hi.linewidths, dtype=np.float64)
            safe = np.where(np.isfinite(lw_arr) & (lw_arr > 0), lw_arr, np.nan)
            decay = 1.0 / (np.pi * safe)
            decay = np.where(np.isfinite(decay), decay, 1e6)
            decay_times = decay
            bandwidths = lw_arr

        # Scalars: forward straight through.
        spectral_tilt = hi.aperiodic_exponent
        noise_floor = hi.spectral_flatness

        # matched_tuning records the ratio set; for HI-derived timbres we
        # always carry hi.to_ratios() (works whether ratios or peaks were
        # the canonical input).
        matched_tuning = [float(r) for r in hi.to_ratios()]

        # Metadata: propagate scale provenance + a few useful HI hints
        # without leaking the full HarmonicInput.
        meta: dict = {
            "scale_source": hi.ratios_source,
            "from_harmonic_input": True,
        }
        if hi.metadata:
            # Don't overwrite scale_source if HI metadata had its own
            # "source" key (legacy compute_biotuner provenance).
            for k, v in hi.metadata.items():
                meta.setdefault(k, v)

        defaults = dict(
            partials_hz=partials,
            amplitudes=amps,
            phases=phases,
            decay_times=decay_times,
            bandwidths=bandwidths,
            spectral_tilt=spectral_tilt,
            noise_floor=noise_floor,
            base_freq=float(hi.base_freq),
            matched_tuning=matched_tuning,
            matching_method="harmonic_input",
            metadata=meta,
        )
        # User overrides take precedence.
        defaults.update(overrides)
        timbre = cls(**defaults)
        timbre.validate()
        return timbre

    # ------------------------------------------------------------------
    # Phase-3 attach helpers: layer bt-only dynamic features onto an
    # already-built Timbre. Each returns a NEW Timbre (immutable-style)
    # whose ``am_modulators`` / ``fm_modulators`` list has been extended
    # with the bt-derived routing. Use after ``from_harmonic_input`` to
    # add modulation that HarmonicInput deliberately doesn't carry.
    #
    # Each method is a no-op (returns ``self``) when the relevant bt
    # attribute is missing or empty — safe to chain unconditionally:
    #
    #     timbre = (
    #         Timbre.from_harmonic_input(hi)
    #               .attach_modulators_from_pac(bt)
    #               .attach_modulators_from_cfc(bt)
    #               .attach_intermodulation_modulators(bt)
    #     )
    # ------------------------------------------------------------------

    def attach_modulators_from_pac(
        self,
        bt: Any,
        *,
        coupling_threshold: float = 0.0,
        max_modulators: int = 16,
    ) -> "Timbre":
        """Append PAC-derived AM modulators to ``am_modulators``.

        Reads ``bt.pac_freqs`` / ``bt.pac_coupling``; for each pair, the
        partial nearest the high-frequency component becomes an AM
        carrier modulated at the low-frequency component's rate.
        Coupling strength sets AM depth (clipped to ``[0, 1]``).

        Returns a new :class:`Timbre`. Returns ``self`` unchanged when
        ``bt`` has no PAC data.

        Parameters
        ----------
        coupling_threshold : float, default=0.0
            Skip PAC pairs whose coupling falls below this.
        max_modulators : int, default=16
            Cap on appended modulators (strongest-coupling first).
        """
        from biotuner.harmonic_timbre.biotuner_mapping import (
            map_pac_to_am_modulators,
        )
        new_mods = map_pac_to_am_modulators(
            bt,
            partials_hz=self.partials_hz,
            coupling_threshold=coupling_threshold,
            max_modulators=max_modulators,
        )
        if not new_mods:
            return self
        return self.with_partials(
            am_modulators=list(self.am_modulators) + list(new_mods),
        )

    def attach_modulators_from_cfc(
        self,
        bt: Any,
        *,
        coupling_threshold: float = 0.0,
        max_modulators: int = 16,
        deviation_scale: float = 1.0,
    ) -> "Timbre":
        """Append CFC-derived FM modulators to ``fm_modulators``.

        Reads ``bt.cfc_freqs`` / ``bt.cfc_coupling`` (falls back to
        ``pac_freqs`` / ``pac_coupling`` when CFC is absent — many
        pipelines only populate PAC). Each pair becomes an FM modulator
        on the partial nearest the high-frequency component, modulated
        at the low-frequency component's rate with deviation scaled by
        coupling strength.

        Returns a new :class:`Timbre`. Returns ``self`` unchanged when
        ``bt`` has no CFC/PAC data.

        Parameters
        ----------
        coupling_threshold : float, default=0.0
        max_modulators : int, default=16
        deviation_scale : float, default=1.0
            ``1.0`` → audible strong FM; ``0.1`` → subtle vibrato.
        """
        from biotuner.harmonic_timbre.biotuner_mapping import (
            map_cfc_to_fm_modulators,
        )
        new_mods = map_cfc_to_fm_modulators(
            bt,
            partials_hz=self.partials_hz,
            coupling_threshold=coupling_threshold,
            max_modulators=max_modulators,
            deviation_scale=deviation_scale,
        )
        if not new_mods:
            return self
        return self.with_partials(
            fm_modulators=list(self.fm_modulators) + list(new_mods),
        )

    def attach_intermodulation_modulators(
        self,
        bt: Any,
        *,
        mode: str = "AM",
        max_modulators: int = 16,
    ) -> "Timbre":
        """Append intermodulation-derived modulators (AM or FM).

        Reads ``bt.endogenous_intermodulations`` — pairs of
        ``(f1, f2)`` frequencies whose sidebands at ``f1 ± f2`` are
        present in the original signal. AM mode recreates those
        sidebands; FM mode reinterprets the pair as carrier +
        modulating frequency.

        Returns a new :class:`Timbre`. Returns ``self`` unchanged when
        no intermodulation data is present.

        Parameters
        ----------
        mode : {'AM', 'FM'}, default='AM'
            Literal sideband interpretation ('AM') or carrier+mod
            interpretation ('FM').
        max_modulators : int, default=16
        """
        if mode not in ("AM", "FM"):
            raise ValueError(f"mode must be 'AM' or 'FM', got {mode!r}")
        from biotuner.harmonic_timbre.biotuner_mapping import (
            map_intermod_to_modulators,
        )
        new_mods = map_intermod_to_modulators(
            bt,
            partials_hz=self.partials_hz,
            mode=mode,
            max_modulators=max_modulators,
        )
        if not new_mods:
            return self
        bucket = "am_modulators" if mode == "AM" else "fm_modulators"
        existing = getattr(self, bucket)
        return self.with_partials(**{bucket: list(existing) + list(new_mods)})

    # ------------------------------------------------------------------
    # Compose every Phase-3 augmentation in one call
    # ------------------------------------------------------------------
    def attach_all_from_biotuner(
        self,
        bt: Any,
        *,
        pac: bool = True,
        cfc: bool = True,
        intermod: bool = True,
        intermod_mode: str = "AM",
    ) -> "Timbre":
        """Apply every available Phase-3 augmentation from ``bt`` in one call.

        Useful as the final step of ``Timbre.from_biotuner``-style flows:
        chain HarmonicInput-derived Phase-1 fields with the bt-only
        modulator attachments. Each individual augmentation is a no-op
        when the corresponding bt data is absent, so this is safe to
        call on any bt.
        """
        out = self
        if pac:
            out = out.attach_modulators_from_pac(bt)
        if cfc:
            out = out.attach_modulators_from_cfc(bt)
        if intermod:
            out = out.attach_intermodulation_modulators(bt, mode=intermod_mode)
        return out

    # ------------------------------------------------------------------
    # Functional update (immutable-style)
    # ------------------------------------------------------------------
    def with_partials(self, **changes) -> "Timbre":
        """Return a copy of this Timbre with the given fields replaced.

        Mirrors ``dataclasses.replace`` but coerces array-typed inputs and
        accepts the convenience name *partials* as an alias for
        *partials_hz*.
        """
        if "partials" in changes and "partials_hz" not in changes:
            changes["partials_hz"] = changes.pop("partials")
        kw: dict = {}
        for f in fields(self):
            if f.name in changes:
                kw[f.name] = changes[f.name]
            else:
                kw[f.name] = getattr(self, f.name)
        # Avoid sharing mutable default containers between original and copy.
        kw["am_modulators"] = list(kw["am_modulators"])
        kw["fm_modulators"] = list(kw["fm_modulators"])
        kw["envelopes"] = dict(kw["envelopes"])
        kw["palette"] = list(kw["palette"])
        kw["elements"] = list(kw["elements"])
        kw["metadata"] = dict(kw["metadata"])
        return Timbre(**kw)

    # ------------------------------------------------------------------
    # Spectral enrichment transforms (wavetable-rich palette)
    # ------------------------------------------------------------------
    # All five return a new Timbre and never mutate self. They rebuild
    # ``partials_hz``, ``amplitudes`` and ``phases`` consistently and
    # apply a bandlimit so single-cycle wavetables stay safe through
    # ~MIDI 84 (partial bin <= ``max_bin``) without per-octave mipmaps.
    # The 0.7-1.2 rolloff sweet spot, integer-ratio constraint on intermod
    # sidebands, and the Schroeder phase formula come from the wavetable
    # design research summarized in this module's design notes.

    def with_intermod_sidebands(
        self,
        bt: Any,
        *,
        depth: float = 0.5,
        integer_ratio_only: bool = True,
        ratio_tolerance: float = 0.05,
        max_bin: int = 512,
    ) -> "Timbre":
        """Add intermodulation sidebands at ``f1 ± f2`` for each pair.

        Reads ``bt.endogenous_intermodulations`` (list of ``(f1, f2)``
        pairs as written by
        :func:`biotuner.peaks_extraction.endogenous_intermodulations`).
        Each pair contributes two static partials at ``f1 + f2`` and
        ``|f1 - f2|`` with amplitude ``depth * min(amp(f1), amp(f2))``.

        This is the *literal sideband interpretation at DC* — the same
        partials you'd get from AM with carrier ``f1`` and modulator
        ``f2``, but precomputed into a single cycle.

        Parameters
        ----------
        bt
            A biotuner-like object with an ``endogenous_intermodulations``
            attribute. If the attribute is missing or empty, ``self`` is
            returned unchanged.
        depth : float, default=0.5
            Amplitude of each sideband relative to the weaker source
            partial.
        integer_ratio_only : bool, default=True
            If True, only emit sidebands when ``f1/f2`` is within
            ``ratio_tolerance`` of an integer ratio (preserves harmonic
            character). Non-integer ratios destroy pitch perception fast.
        ratio_tolerance : float, default=0.05
            Relative tolerance (5%) for the integer-ratio test.
        max_bin : int, default=512
            Drop sidebands whose partial bin exceeds this. Keeps single-
            cycle wavetables alias-free up to ~MIDI 84.
        """
        intermod = getattr(bt, "endogenous_intermodulations", None)
        if intermod is None or len(intermod) == 0:
            return self

        partials = np.asarray(self.partials_hz, dtype=np.float64)
        amps = np.asarray(self.amplitudes, dtype=np.float64)
        phases = _resolve_phases_or_zeros(self)
        base = float(self.base_freq) if self.base_freq > 0 else 1.0
        new_p, new_a, new_ph = [], [], []

        for entry in intermod:
            try:
                f1, f2 = float(entry[0]), float(entry[1])
            except (TypeError, ValueError, IndexError):
                continue
            if f1 <= 0 or f2 <= 0:
                continue
            if integer_ratio_only and not _is_integer_ratio(f1, f2, ratio_tolerance):
                continue
            i1 = int(np.argmin(np.abs(partials - f1)))
            i2 = int(np.argmin(np.abs(partials - f2)))
            a_side = float(depth) * float(min(amps[i1], amps[i2]))
            for f in (f1 + f2, abs(f1 - f2)):
                if f <= 0:
                    continue
                new_p.append(f)
                new_a.append(a_side)
                new_ph.append(0.0)

        if not new_p:
            return self
        merged_p = np.concatenate([partials, np.asarray(new_p)])
        merged_a = np.concatenate([amps, np.asarray(new_a)])
        merged_ph = np.concatenate([phases, np.asarray(new_ph)])
        merged_p, merged_a, merged_ph = _bandlimit(
            merged_p, merged_a, merged_ph, base, max_bin
        )
        return self.with_partials(
            partials_hz=merged_p, amplitudes=merged_a, phases=merged_ph,
        )

    def with_harmonic_stack(
        self,
        n: int = 4,
        *,
        rolloff: float = 0.9,
        max_bin: int = 512,
    ) -> "Timbre":
        """Add ``2·f, 3·f, ..., n·f`` overtones to each existing partial.

        Each overtone ``h`` of source partial ``fk`` is added with
        amplitude ``amps[k] / h**rolloff``. Inharmonic biosignal peaks
        (e.g. EMD-derived) thereby acquire the overtone series of
        natural acoustic sources, replacing flat single-partial-per-IMF
        wavetables with audibly warmer ones.

        Parameters
        ----------
        n : int, default=4
            Number of overtones per source partial. ``n=4`` adds
            ``2f, 3f, 4f, 5f``.
        rolloff : float, default=0.9
            Amplitude exponent. The 0.7-1.2 window is the sweet spot
            in Serum/Vital factory tables; <0.7 buzzes, >1.2 dulls.
        max_bin : int, default=512
            Drop overtones whose partial bin exceeds this.
        """
        if n <= 0:
            return self
        partials = np.asarray(self.partials_hz, dtype=np.float64)
        amps = np.asarray(self.amplitudes, dtype=np.float64)
        phases = _resolve_phases_or_zeros(self)
        base = float(self.base_freq) if self.base_freq > 0 else 1.0

        harmonics = np.arange(2, 2 + int(n), dtype=np.float64)
        # Outer product: (n_partials, n_harmonics)
        new_p = (partials[:, None] * harmonics[None, :]).reshape(-1)
        atten = np.power(harmonics, float(rolloff))
        new_a = (amps[:, None] / atten[None, :]).reshape(-1)
        new_ph = np.zeros_like(new_p)

        merged_p = np.concatenate([partials, new_p])
        merged_a = np.concatenate([amps, new_a])
        merged_ph = np.concatenate([phases, new_ph])
        merged_p, merged_a, merged_ph = _bandlimit(
            merged_p, merged_a, merged_ph, base, max_bin
        )
        return self.with_partials(
            partials_hz=merged_p, amplitudes=merged_a, phases=merged_ph,
        )

    def with_phase_mode(
        self,
        mode: str = "schroeder",
        *,
        seed: int = 0,
    ) -> "Timbre":
        """Replace partial phases according to a named scheme.

        Same magnitude spectrum, different phase: large perceptual
        difference for short/percussive sounds, smaller-but-real for
        sustained tones.

        Parameters
        ----------
        mode : {'cosine', 'schroeder', 'random', 'biosignal'}
            ``'cosine'`` — all phases zero; spike-shaped time waveform,
            crest factor ~12 dB, "clicky/buzzy".
            ``'schroeder'`` — ``φ_k = -π·k(k-1)/N``; near-flat envelope,
            crest factor ~3 dB, "fat" and easier to mix loud (default).
            ``'random'`` — uniform phases; diffuse/noisy, good for pads.
            ``'biosignal'`` — leave existing phases untouched (no-op).
        seed : int, default=0
            Seed for ``'random'``.
        """
        valid = ("cosine", "schroeder", "random", "biosignal")
        if mode not in valid:
            raise ValueError(f"phase mode must be one of {valid}, got {mode!r}")
        if mode == "biosignal":
            return self
        n = self.n_partials()
        if mode == "cosine":
            new_phases = np.zeros(n, dtype=np.float64)
        elif mode == "schroeder":
            k = np.arange(1, n + 1, dtype=np.float64)
            new_phases = (-np.pi * k * (k - 1.0) / float(n)) % (2.0 * np.pi)
        else:  # random
            rng = np.random.default_rng(int(seed))
            new_phases = rng.uniform(0.0, 2.0 * np.pi, size=n)
        return self.with_partials(phases=new_phases)

    def with_formant(
        self,
        center_hz: float = 2000.0,
        *,
        width_hz: float = 800.0,
        gain_db: float = 4.0,
    ) -> "Timbre":
        """Multiplicative spectral envelope bump (formant).

        Boosts amplitudes of partials near ``center_hz`` with a Gaussian
        envelope of standard deviation ``width_hz``. This is the
        presence-band trick used in nearly every "rich" pad/lead
        wavetable; the multiplicative form preserves partial count and
        only reshapes the spectral envelope.

        Parameters
        ----------
        center_hz : float, default=2000.0
            Formant center frequency. The 1500-2500 Hz "presence band"
            is typical for non-vocal richness; vowel formants live at
            500-1000 (F1) and 1500-2500 (F2).
        width_hz : float, default=800.0
            Gaussian standard deviation in Hz. Wider = broader bump.
        gain_db : float, default=4.0
            Peak gain at the formant center.
        """
        if width_hz <= 0:
            raise ValueError("width_hz must be positive")
        partials = np.asarray(self.partials_hz, dtype=np.float64)
        amps = np.asarray(self.amplitudes, dtype=np.float64)
        gain_lin = 10.0 ** (float(gain_db) / 20.0)
        bump = np.exp(-0.5 * ((partials - float(center_hz)) / float(width_hz)) ** 2)
        # Multiplicative envelope: 1.0 baseline outside the bump, gain_lin at center.
        envelope = 1.0 + (gain_lin - 1.0) * bump
        new_amps = amps * envelope
        return self.with_partials(amplitudes=new_amps)

    def with_slight_detune(
        self,
        *,
        percent: float = 1.0,
        n_partials: int = 3,
        seed: int = 0,
    ) -> "Timbre":
        """Detune ``n_partials`` random partials by up to ``±percent``%.

        The Bilbao/Smith stretched-string trick: nudging a few partials
        of an otherwise-harmonic timbre by 0.3-2% adds "organic shimmer"
        without losing pitch perception. Distinct from the full
        inharmonic-series constructors in :mod:`inharmonic` (those
        rebuild every partial; this only nudges a chosen few).

        Parameters
        ----------
        percent : float, default=1.0
            Maximum detuning magnitude in percent (0.5-2 is musical).
        n_partials : int, default=3
            How many partials to perturb (chosen with the seed).
        seed : int, default=0
            RNG seed.
        """
        n_total = self.n_partials()
        if n_total == 0 or percent <= 0 or n_partials <= 0:
            return self
        rng = np.random.default_rng(int(seed))
        n_pick = int(min(n_partials, n_total))
        idx = rng.choice(n_total, size=n_pick, replace=False)
        deltas = rng.uniform(-1.0, 1.0, size=n_pick) * (float(percent) / 100.0)
        new_partials = np.asarray(self.partials_hz, dtype=np.float64).copy()
        new_partials[idx] = new_partials[idx] * (1.0 + deltas)
        return self.with_partials(partials_hz=new_partials)

    # ------------------------------------------------------------------
    # Persistence: JSON (metadata) + .npz (arrays)
    # ------------------------------------------------------------------
    def save(self, path: str) -> str:
        """Save this Timbre as a JSON+NPZ bundle.

        ``path`` is the path stem; two files are written:
            ``<path>.json`` — scalar metadata + modulator records.
            ``<path>.npz``  — all numpy array fields.
        """
        stem = _strip_known_suffixes(path)
        json_path = stem + ".json"
        npz_path = stem + ".npz"

        arrays: dict[str, np.ndarray] = {}
        for fname in _ARRAY_FIELDS:
            v = getattr(self, fname)
            if v is None:
                continue
            arrays[fname] = np.asarray(v)
        env = self.envelopes or {}
        for k, v in env.items():
            arrays[f"envelope__{k}"] = np.asarray(v)

        np.savez(npz_path, **arrays)

        meta = {
            "spectral_tilt": self.spectral_tilt,
            "noise_floor": self.noise_floor,
            "envelope_rate": self.envelope_rate,
            "base_freq": self.base_freq,
            "matched_tuning": list(self.matched_tuning) if self.matched_tuning is not None else None,
            "matching_method": self.matching_method,
            "metadata": dict(self.metadata),
            "palette": [list(c) for c in self.palette],
            "elements": list(self.elements),
            "geometry_signature": self.geometry_signature,
            "am_modulators": [m.to_dict() for m in self.am_modulators],
            "fm_modulators": [m.to_dict() for m in self.fm_modulators],
            "envelope_keys": list(env.keys()),
            "has_layers": self.layers is not None,
            "_npz": os.path.basename(npz_path),
        }
        with open(json_path, "w", encoding="utf-8") as fp:
            json.dump(meta, fp, indent=2, default=_json_default)

        if self.layers:
            layers_dir = stem + "_layers"
            os.makedirs(layers_dir, exist_ok=True)
            for i, layer in enumerate(self.layers):
                layer.save(os.path.join(layers_dir, f"layer_{i:02d}"))

        return json_path

    @classmethod
    def load(cls, path: str) -> "Timbre":
        """Load a Timbre from a JSON+NPZ bundle saved by :meth:`save`."""
        stem = _strip_known_suffixes(path)
        json_path = stem + ".json"
        npz_path = stem + ".npz"
        with open(json_path, "r", encoding="utf-8") as fp:
            meta = json.load(fp)

        with np.load(npz_path) as npz:
            arrays = {k: npz[k] for k in npz.files}

        envelopes: dict[str, np.ndarray] = {}
        for k in list(arrays.keys()):
            if k.startswith("envelope__"):
                envelopes[k[len("envelope__"):]] = arrays.pop(k)

        kwargs: dict = {
            "partials_hz": arrays["partials_hz"],
            "amplitudes": arrays["amplitudes"],
            "phases": arrays.get("phases"),
            "decay_times": arrays.get("decay_times"),
            "bandwidths": arrays.get("bandwidths"),
            "detuning": arrays.get("detuning"),
            "layer_mix": arrays.get("layer_mix"),
            "spectral_tilt": meta.get("spectral_tilt"),
            "noise_floor": meta.get("noise_floor"),
            "envelopes": envelopes,
            "envelope_rate": meta.get("envelope_rate", 100.0),
            "palette": [tuple(c) for c in meta.get("palette", [])],
            "elements": list(meta.get("elements", [])),
            "geometry_signature": meta.get("geometry_signature"),
            "am_modulators": [Modulator.from_dict(d) for d in meta.get("am_modulators", [])],
            "fm_modulators": [Modulator.from_dict(d) for d in meta.get("fm_modulators", [])],
            "base_freq": meta.get("base_freq", 1.0),
            "matched_tuning": meta.get("matched_tuning"),
            "matching_method": meta.get("matching_method", ""),
            "metadata": dict(meta.get("metadata", {})),
        }

        if meta.get("has_layers"):
            layers_dir = stem + "_layers"
            layers: list[Timbre] = []
            i = 0
            while True:
                layer_stem = os.path.join(layers_dir, f"layer_{i:02d}")
                if not os.path.exists(layer_stem + ".json"):
                    break
                layers.append(cls.load(layer_stem))
                i += 1
            kwargs["layers"] = layers

        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Synthesis entry point
    # ------------------------------------------------------------------
    def synthesize(
        self,
        samplerate: int = 48000,
        duration: float = 1.0,
        base_freq: float | None = None,
        *,
        include_modulators: bool = True,
        include_layers: bool = True,
        normalize: bool = True,
    ) -> np.ndarray:
        """Render to a 1D float32 numpy audio buffer.

        Selects the simplest renderer that covers the populated fields.
        This is the entry point Goofy Pipe nodes call into for live
        rendering.
        """
        # local import to avoid a top-level cycle with synthesis.py
        from biotuner.harmonic_timbre import synthesis as _syn

        has_modulators = include_modulators and (self.am_modulators or self.fm_modulators)
        has_envelope = (
            self.decay_times is not None
            or self.spectral_tilt is not None
            or self.noise_floor is not None
        )
        has_band = self.bandwidths is not None

        if has_modulators:
            return _syn.render_modulated(
                self,
                samplerate=samplerate,
                duration=duration,
                base_freq=base_freq,
                normalize=normalize,
            )
        if has_band:
            return _syn.render_band_limited(
                self,
                samplerate=samplerate,
                duration=duration,
                base_freq=base_freq,
                normalize=normalize,
            )
        if has_envelope:
            return _syn.render_with_envelope(
                self,
                samplerate=samplerate,
                duration=duration,
                base_freq=base_freq,
                normalize=normalize,
            )
        return _syn.render_additive(
            self,
            samplerate=samplerate,
            duration=duration,
            base_freq=base_freq,
            normalize=normalize,
        )


# ---------------------------------------------------------------------------
# TimbreSequence
# ---------------------------------------------------------------------------

@dataclass
class TimbreSequence:
    """A time-resolved sequence of :class:`Timbre` frames.

    Phase 1 ships construction, indexing, and ``synthesize`` (which
    delegates to ``synthesis.render_sequence`` if available, else a simple
    crossfade). Sequence-source constructors (transitional harmony,
    Markov walks, evolving rhythms) ship in Phase 3.
    """

    frames: list[Timbre]
    times: np.ndarray | None = None  # seconds; uniform spacing if None

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("TimbreSequence requires at least one frame")
        if self.times is not None:
            self.times = np.asarray(self.times, dtype=np.float64)
            if self.times.shape[0] != len(self.frames):
                raise ValueError("times length must equal number of frames")

    def n_frames(self) -> int:
        return len(self.frames)

    def at(self, t: float) -> Timbre:
        """Return the nearest frame at time ``t`` (seconds)."""
        if self.times is None:
            # uniform; assume frames span [0, 1]
            n = self.n_frames()
            idx = int(np.clip(round(t * (n - 1)), 0, n - 1))
            return self.frames[idx]
        idx = int(np.argmin(np.abs(self.times - t)))
        return self.frames[idx]

    def synthesize(
        self,
        samplerate: int = 48000,
        crossfade: float = 0.05,
        frame_duration: float = 1.0,
        normalize: bool = True,
    ) -> np.ndarray:
        """Render the sequence to a 1D float32 buffer with crossfaded frames."""
        n = self.n_frames()
        if n == 1:
            return self.frames[0].synthesize(
                samplerate=samplerate, duration=frame_duration, normalize=normalize
            )
        frame_samples = int(round(frame_duration * samplerate))
        cf_samples = int(round(crossfade * samplerate))
        cf_samples = max(0, min(cf_samples, frame_samples - 1))
        total = frame_samples * n - cf_samples * (n - 1)
        out = np.zeros(total, dtype=np.float32)
        write_idx = 0
        for i, frame in enumerate(self.frames):
            buf = frame.synthesize(
                samplerate=samplerate, duration=frame_duration, normalize=False
            )
            buf = buf[:frame_samples].astype(np.float32, copy=False)
            if i == 0:
                out[write_idx : write_idx + frame_samples] = buf
                write_idx += frame_samples - cf_samples
            else:
                if cf_samples > 0:
                    fade = np.linspace(0.0, 1.0, cf_samples, dtype=np.float32)
                    head = buf[:cf_samples] * fade
                    out[write_idx : write_idx + cf_samples] *= 1.0 - fade
                    out[write_idx : write_idx + cf_samples] += head
                    out[write_idx + cf_samples : write_idx + frame_samples] = buf[cf_samples:]
                else:
                    out[write_idx : write_idx + frame_samples] = buf
                write_idx += frame_samples - cf_samples
        if normalize:
            m = float(np.max(np.abs(out))) if out.size else 0.0
            if m > 0:
                out = out / m
        return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_phases_or_zeros(timbre: "Timbre") -> np.ndarray:
    """Return ``timbre.phases`` if set, otherwise an array of zeros."""
    if timbre.phases is None:
        return np.zeros(timbre.n_partials(), dtype=np.float64)
    return np.asarray(timbre.phases, dtype=np.float64)


def _is_integer_ratio(f1: float, f2: float, tolerance: float) -> bool:
    """True if ``f1/f2`` (or its reciprocal) is within ``tolerance`` of an int.

    Used by :meth:`Timbre.with_intermod_sidebands` to keep sideband
    products harmonic when ``integer_ratio_only=True``.
    """
    if f1 <= 0 or f2 <= 0:
        return False
    r = f1 / f2 if f1 >= f2 else f2 / f1
    nearest = round(r)
    if nearest <= 0:
        return False
    return abs(r - nearest) / nearest <= tolerance


def _bandlimit(
    partials: np.ndarray,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    base_freq: float,
    max_bin: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop partials whose bin index ``partial/base_freq`` exceeds ``max_bin``.

    Single-cycle wavetables of length 2048 have 1024 useful bins. Capping
    at ``max_bin=512`` keeps the table alias-free up to roughly MIDI 84
    without per-octave mipmapping.
    """
    if max_bin <= 0 or base_freq <= 0:
        return partials, amplitudes, phases
    keep = (partials / base_freq) <= float(max_bin)
    keep &= partials > 0
    return partials[keep], amplitudes[keep], phases[keep]


def _strip_known_suffixes(path: str) -> str:
    for suffix in (".json", ".npz"):
        if path.endswith(suffix):
            return path[: -len(suffix)]
    return path


def _json_default(obj):
    """Fall-through encoder for json.dump with numpy/array-y values."""
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
