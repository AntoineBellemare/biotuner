"""Engine adapter for biotuner.harmonic_timbre.

Wraps the HarmonicInput → Timbre construction pipeline and the
exporters in a request/response shape the frontend can consume. The
heavy lifting lives in biotuner itself; this service only:

  1. Builds a HarmonicInput from the request payload (analysis snapshot
     + user design choices).
  2. Constructs a Timbre via Timbre.from_harmonic_input.
  3. Attaches PAC/CFC/intermod modulators when the request supplies the
     source data (and the user hasn't disabled them).
  4. Serialises the result into a dict suitable for JSON transport —
     including modulator routing info so the frontend can render the
     bottom-row routing matrix.

Exports run through the same Timbre object; format-specific endpoints
(``/api/timbre/export/{format}``) call back into this service for
construction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence
from types import SimpleNamespace

import numpy as np

from biotuner.harmonic_input import HarmonicInput
from biotuner.harmonic_timbre import Timbre
from biotuner.harmonic_timbre.timbre import Modulator


# ---------------------------------------------------------------------------
# Build a "duck-typed bt" from request payload
# ---------------------------------------------------------------------------
#
# The HarmonicInput.from_biotuner factory + the Timbre.attach_* methods
# both consume a fitted compute_biotuner instance. Rebuilding the full bt
# server-side from the request would be expensive (we'd need the raw
# signal); much lighter to construct a SimpleNamespace with the exact
# attributes those methods read. Any field that's absent on the request
# is simply not set on the namespace, and the downstream code's
# getattr(bt, name, None) fallbacks Just Work.
def _build_pseudo_bt(req: Dict[str, Any]) -> SimpleNamespace:
    """Pack a request payload into a duck-typed bt for the biotuner API."""
    ns = SimpleNamespace()
    # Core spectral content
    peaks = req.get("peaks") or []
    ns.peaks = np.asarray(peaks, dtype=np.float64) if peaks else np.array([])
    if req.get("amps"):
        ns.amps = np.asarray(req["amps"], dtype=np.float64)
    if req.get("phases"):
        ns.phases = np.asarray(req["phases"], dtype=np.float64)
    if req.get("linewidths"):
        ns.linewidth = np.asarray(req["linewidths"], dtype=np.float64)
    # FOOOF + noise estimate
    if req.get("aperiodic_exponent") is not None:
        ns.aperiodic_exponent = float(req["aperiodic_exponent"])
    if req.get("spectral_flatness") is not None:
        ns.spectral_flatness = float(req["spectral_flatness"])
    # Scale variants — passed as a dict keyed by SCALE_KEYS vocabulary
    for key, attr in [
        ("peaks_ratios_cons",          "peaks_ratios_cons"),
        ("peaks_ratios",               "peaks_ratios"),
        ("extended_peaks_ratios",      "extended_peaks_ratios"),
        ("extended_peaks_ratios_cons", "extended_peaks_ratios_cons"),
        ("diss_scale",                 "diss_scale"),
        ("HE",                         "HE_scale"),
        ("euler_fokker",               "euler_fokker"),
        ("harm_tuning",                "harm_tuning_scale"),
        ("harm_fit",                   "harm_fit_tuning_scale"),
    ]:
        vals = (req.get("scales") or {}).get(key)
        if vals:
            setattr(ns, attr, np.asarray(vals, dtype=np.float64))
    # Modulator source data
    if req.get("pac_freqs") and req.get("pac_coupling"):
        ns.pac_freqs = [tuple(p) for p in req["pac_freqs"]]
        ns.pac_coupling = np.asarray(req["pac_coupling"], dtype=np.float64)
    if req.get("cfc_freqs") and req.get("cfc_coupling"):
        ns.cfc_freqs = [tuple(p) for p in req["cfc_freqs"]]
        ns.cfc_coupling = np.asarray(req["cfc_coupling"], dtype=np.float64)
    if req.get("intermodulations"):
        ns.endogenous_intermodulations = [tuple(p) for p in req["intermodulations"]]
    return ns


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _modulator_to_dict(m: Modulator, mod_id: str, enabled: bool = True) -> Dict[str, Any]:
    """Serialise a Modulator for the frontend routing-matrix viz."""
    return {
        "id": mod_id,
        "type": m.mod_type,                    # 'AM' | 'FM'
        "carrier_idx": int(m.carrier_idx),
        "mod_freq": float(m.mod_freq),
        "depth": float(m.depth),
        "phase": float(m.phase),
        "source": m.source,
        "enabled": bool(enabled),
    }


def _arr_or_none(v) -> Optional[List[float]]:
    """Convert an ndarray-or-None to list-or-None for JSON transport."""
    if v is None:
        return None
    return [float(x) for x in np.asarray(v).ravel()]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_timbre(req: Dict[str, Any]) -> Dict[str, Any]:
    """Build a Timbre from an analysis snapshot + design choices.

    Returns the timbre's serialisable state plus modulator routing info
    so the frontend can render the spectrum + routing matrix without a
    second round trip.
    """
    pseudo_bt = _build_pseudo_bt(req)
    if pseudo_bt.peaks.size == 0:
        raise ValueError("`peaks` is required to build a timbre.")

    # 1) HarmonicInput — use the new scale_priority pathway when the
    # request specifies one; otherwise legacy peaks-based selection.
    scale_priority = req.get("scale_priority")
    hi = HarmonicInput.from_biotuner(
        pseudo_bt,
        scale_priority=scale_priority,
        include_alternates=True,
    )

    # 2) Timbre — Phase-1 mapping from HarmonicInput. Voicing overrides
    # (e.g., user-edited noise_floor / spectral_tilt sliders) get passed
    # straight through.
    voicing = req.get("voicing") or {}
    overrides: Dict[str, Any] = {}
    for key in ("spectral_tilt", "noise_floor"):
        if key in voicing and voicing[key] is not None:
            overrides[key] = voicing[key]
    timbre = Timbre.from_harmonic_input(hi, **overrides)

    # 3) Attach modulators. We always run the attach pass so the
    # frontend sees what's available; user-side disable happens in JS
    # at synth time (the `enabled` flag in the response controls it).
    timbre = timbre.attach_all_from_biotuner(pseudo_bt)

    # 4) Determine which modulators are user-enabled. The request may
    # supply a {"pac_0": false, "cfc_2": true, ...} dict; default ON.
    enabled = req.get("enabled_modulators") or {}

    am_dicts: List[Dict[str, Any]] = []
    for i, m in enumerate(timbre.am_modulators):
        # Stable id based on source string + index; the frontend uses
        # this for the per-modulator toggle state across re-renders.
        mod_id = f"am_{i}_{m.source}".replace(" ", "")
        am_dicts.append(_modulator_to_dict(m, mod_id, enabled.get(mod_id, True)))
    fm_dicts: List[Dict[str, Any]] = []
    for i, m in enumerate(timbre.fm_modulators):
        mod_id = f"fm_{i}_{m.source}".replace(" ", "")
        fm_dicts.append(_modulator_to_dict(m, mod_id, enabled.get(mod_id, True)))

    return {
        # ---- Timbre core ----
        "partials_hz":   _arr_or_none(timbre.partials_hz),
        "amplitudes":    _arr_or_none(timbre.amplitudes),
        "phases":        _arr_or_none(timbre.phases) or [0.0] * timbre.n_partials(),
        "decay_times":   _arr_or_none(timbre.decay_times),
        "bandwidths":    _arr_or_none(timbre.bandwidths),
        "spectral_tilt": float(timbre.spectral_tilt) if timbre.spectral_tilt is not None else None,
        "noise_floor":   float(timbre.noise_floor) if timbre.noise_floor is not None else None,
        "matched_tuning": [float(r) for r in (timbre.matched_tuning or [])],
        "base_freq":     float(timbre.base_freq),
        # ---- Modulator routings (for the viz) ----
        "am_modulators": am_dicts,
        "fm_modulators": fm_dicts,
        # ---- Provenance ----
        "scale_source":     hi.ratios_source,
        "scale_alternates": list(hi.ratios_alternates.keys()),
        "matching_method":  timbre.matching_method or "harmonic_input",
        # ---- Echo back the metadata so the UI can show provenance chips
        "metadata": dict(timbre.metadata),
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
#
# Polymorphic dispatch — one endpoint per format would multiply
# boilerplate. Each branch builds a Timbre via compute_timbre's same
# pipeline, then calls the relevant biotuner exporter. The frontend
# downloads the resulting file via FileResponse.

def _build_timbre_from_request(req: Dict[str, Any]) -> Timbre:
    """Reconstruct a Timbre (with attached modulators) from the request.

    Same pipeline as compute_timbre but returns the Timbre object
    itself for exporters to consume directly.
    """
    pseudo_bt = _build_pseudo_bt(req)
    if pseudo_bt.peaks.size == 0:
        raise ValueError("`peaks` is required to build a timbre.")
    hi = HarmonicInput.from_biotuner(
        pseudo_bt,
        scale_priority=req.get("scale_priority"),
        include_alternates=True,
    )
    voicing = req.get("voicing") or {}
    overrides: Dict[str, Any] = {}
    for key in ("spectral_tilt", "noise_floor"):
        if key in voicing and voicing[key] is not None:
            overrides[key] = voicing[key]
    timbre = Timbre.from_harmonic_input(hi, **overrides)
    timbre = timbre.attach_all_from_biotuner(pseudo_bt)
    # Drop user-disabled modulators before export so what they hear is
    # what they get.
    enabled = req.get("enabled_modulators") or {}
    def _keep(mods, prefix):
        if not enabled:
            return mods
        out = []
        for i, m in enumerate(mods):
            mod_id = f"{prefix}_{i}_{m.source}".replace(" ", "")
            if enabled.get(mod_id, True):
                out.append(m)
        return out
    am = _keep(timbre.am_modulators, "am")
    fm = _keep(timbre.fm_modulators, "fm")
    return timbre.with_partials(am_modulators=am, fm_modulators=fm)


def export_to_format(format: str, req: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    """Render the request as a file in ``format``. Returns a path map.

    Output is always written under ``out_dir`` so the route handler can
    pick the primary file and stream it back. Each exporter returns a
    dict of paths; we forward that so the caller knows what's available
    (some exporters write multiple files — e.g. SFZ + samples + .scl).
    """
    from biotuner.harmonic_timbre.exporters import (
        to_vital,
        to_wavetable,
        export_sfz,
        export_surge_bundle,
        export_csound,
        export_supercollider,
        export_tuning_files,
        export_wav_pack,
    )

    timbre = _build_timbre_from_request(req)
    cfg = req.get("export_config") or {}
    fmt = format.lower()

    if fmt == "vital":
        # Default Vital style is spectral; user can pick a variant via cfg.
        style = cfg.get("style", "spectral")
        out_path = f"{out_dir}/timbre.vital"
        if style == "spectral":
            res = to_vital.to_vital_spectral(timbre, out_path)
        elif style == "inharmonic":
            res = to_vital.to_vital_inharmonic(timbre, out_path)
        elif style == "wavetable_morph":
            res = to_vital.to_vital_wavetable_morph(
                timbre, out_path,
                n_frames=int(cfg.get("n_frames", 32)),
                evolution=cfg.get("evolution", "tilt"),
            )
        elif style == "fm":
            res = to_vital.to_vital_fm(timbre, out_path)
        elif style == "simple":
            res = to_vital.to_vital_simple(timbre, out_path)
        elif style == "pluck":
            res = to_vital.to_vital_pluck(timbre, out_path)
        else:
            raise ValueError(f"Unknown vital style: {style!r}")
        return res

    if fmt == "sfz":
        return export_sfz(timbre, out_dir, bundle_name=cfg.get("name", "timbre"))

    if fmt == "surge":
        return export_surge_bundle(timbre, out_dir, bundle_name=cfg.get("name", "timbre"))

    if fmt == "wavetable":
        out_path = f"{out_dir}/timbre.wavetable.wav"
        return to_wavetable.export_wavetable(
            timbre, out_path,
            n_frames=int(cfg.get("n_frames", 32)),
            synth_target=cfg.get("synth_target", "vital"),
            evolution=cfg.get("evolution", "tilt"),
        )

    if fmt == "csound":
        out_path = f"{out_dir}/timbre.csd"
        return export_csound(
            timbre, out_path,
            base_freq=float(cfg.get("base_freq", timbre.base_freq or 220.0)),
            demo_pattern=cfg.get("demo_pattern", "scale"),
        )

    if fmt == "supercollider":
        out_path = f"{out_dir}/timbre.scd"
        return export_supercollider(
            timbre, out_path,
            base_freq=float(cfg.get("base_freq", timbre.base_freq or 220.0)),
        )

    if fmt == "tuning":
        return export_tuning_files(
            timbre, out_dir, bundle_name=cfg.get("name", "timbre"),
        )

    if fmt == "wav":
        return export_wav_pack(
            timbre, out_dir,
            bundle_name=cfg.get("name", "timbre"),
            duration=float(cfg.get("duration", 2.0)),
            samplerate=int(cfg.get("samplerate", 44100)),
        )

    raise ValueError(f"Unknown export format: {format!r}")
