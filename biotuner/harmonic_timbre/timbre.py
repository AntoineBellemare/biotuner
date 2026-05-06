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
