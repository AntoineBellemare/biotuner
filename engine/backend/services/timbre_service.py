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
from biotuner.harmonic_timbre.matching import match_timbre, _METHOD_TO_FUNC


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

    # 2) Build the base Timbre. Two paths:
    #    - 'harmonic_input' (default) — partials = peaks, direct.
    #    - any biotuner matching method (consonance_weighted, sethares,
    #      harmonic_entropy, hybrid, direct) — defers to match_timbre,
    #      which derives partials from the chosen scale's ratios via the
    #      method's specific algorithm. This is what the user clicks
    #      "Consonance-weighted" or "Hybrid" in the Voice picker for.
    matching_method = (req.get("matching_method") or "harmonic_input").lower()
    voicing = req.get("voicing") or {}
    voicing_overrides: Dict[str, Any] = {}
    for key in ("spectral_tilt", "noise_floor"):
        if key in voicing and voicing[key] is not None:
            voicing_overrides[key] = voicing[key]

    if matching_method == "harmonic_input":
        timbre = Timbre.from_harmonic_input(hi, **voicing_overrides)
    elif matching_method in _METHOD_TO_FUNC:
        try:
            timbre = match_timbre(
                hi.to_ratios(),
                method=matching_method,
                base_freq=hi.base_freq,
            )
        except ValueError as e:
            # Some methods (notably 'hybrid' = consonance + entropy)
            # require their component matchers to produce the same
            # n_partials; certain ratio sets violate that. Fall back
            # to consonance_weighted with a clear provenance note so
            # the user sees what happened instead of a 500.
            from biotuner.harmonic_timbre.matching import match_consonance_weighted
            timbre = match_consonance_weighted(
                hi.to_ratios(), base_freq=hi.base_freq,
            )
            timbre.metadata = dict(timbre.metadata or {})
            timbre.metadata["matching_fallback"] = (
                f"{matching_method} failed ({e}); used consonance_weighted"
            )
            matching_method = "consonance_weighted"
        if voicing_overrides:
            timbre = timbre.with_partials(**voicing_overrides)
        timbre.matched_tuning = [float(r) for r in hi.to_ratios()]
        timbre.matching_method = matching_method
        timbre.metadata = dict(timbre.metadata or {})
        timbre.metadata.setdefault("scale_source", hi.ratios_source)
    else:
        raise ValueError(
            f"Unknown matching method {matching_method!r}. "
            f"Known: ['harmonic_input', " + ", ".join(repr(k) for k in _METHOD_TO_FUNC) + "]"
        )

    # 3) Attach modulators. We always run the attach pass so the
    # frontend sees what's available; user-side disable happens in JS
    # at synth time (the `enabled` flag in the response controls it).
    timbre = timbre.attach_all_from_biotuner(pseudo_bt)

    # 4) Optional partial-spectrum enrichment. These ADD partials
    # (unlike modulators which wobble existing ones) — they're the
    # "complexify my voice" knobs. Both routines are no-ops on bts
    # without their respective source data.
    enrich = req.get("enrichment") or {}
    intermod_cfg = enrich.get("intermod")
    if intermod_cfg and intermod_cfg.get("enabled"):
        timbre = timbre.with_intermod_sidebands(
            pseudo_bt,
            depth=float(intermod_cfg.get("depth", 0.5)),
            integer_ratio_only=bool(intermod_cfg.get("integer_ratio_only", True)),
        )
    stack_cfg = enrich.get("harmonic_stack")
    if stack_cfg and stack_cfg.get("enabled"):
        timbre = timbre.with_harmonic_stack(
            n=int(stack_cfg.get("n", 4)),
            rolloff=float(stack_cfg.get("rolloff", 0.9)),
        )

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
    """Reconstruct a Timbre (with attached modulators + enrichment) from
    the request.

    Same pipeline as compute_timbre but returns the Timbre object
    itself for exporters to consume directly. Kept in sync with
    compute_timbre — when one changes, the other must too.
    """
    pseudo_bt = _build_pseudo_bt(req)
    if pseudo_bt.peaks.size == 0:
        raise ValueError("`peaks` is required to build a timbre.")
    hi = HarmonicInput.from_biotuner(
        pseudo_bt,
        scale_priority=req.get("scale_priority"),
        include_alternates=True,
    )
    matching_method = (req.get("matching_method") or "harmonic_input").lower()
    voicing = req.get("voicing") or {}
    voicing_overrides: Dict[str, Any] = {}
    for key in ("spectral_tilt", "noise_floor"):
        if key in voicing and voicing[key] is not None:
            voicing_overrides[key] = voicing[key]
    if matching_method == "harmonic_input":
        timbre = Timbre.from_harmonic_input(hi, **voicing_overrides)
    elif matching_method in _METHOD_TO_FUNC:
        timbre = match_timbre(
            hi.to_ratios(),
            method=matching_method,
            base_freq=hi.base_freq,
        )
        if voicing_overrides:
            timbre = timbre.with_partials(**voicing_overrides)
        timbre.matched_tuning = [float(r) for r in hi.to_ratios()]
        timbre.matching_method = matching_method
    else:
        raise ValueError(f"Unknown matching method {matching_method!r}")
    timbre = timbre.attach_all_from_biotuner(pseudo_bt)
    enrich = req.get("enrichment") or {}
    intermod_cfg = enrich.get("intermod")
    if intermod_cfg and intermod_cfg.get("enabled"):
        timbre = timbre.with_intermod_sidebands(
            pseudo_bt,
            depth=float(intermod_cfg.get("depth", 0.5)),
            integer_ratio_only=bool(intermod_cfg.get("integer_ratio_only", True)),
        )
    stack_cfg = enrich.get("harmonic_stack")
    if stack_cfg and stack_cfg.get("enabled"):
        timbre = timbre.with_harmonic_stack(
            n=int(stack_cfg.get("n", 4)),
            rolloff=float(stack_cfg.get("rolloff", 0.9)),
        )
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


def compute_intermods_on_demand(
    peaks: List[float],
    amps: Optional[List[float]] = None,
    *,
    order: int = 3,
    min_IMs: int = 2,
    max_freq: float = 100.0,
) -> List[List[float]]:
    """Detect endogenous intermodulation pairs from a peak list.

    Wraps biotuner.peaks_extraction.endogenous_intermodulations so the
    Timbre tab can populate the "Intermod sidebands" enrichment toggle
    on demand. The analyze endpoint doesn't compute this attribute by
    default, so without this helper the checkbox would never enable.

    Returns a list of ``[f1, f2]`` pairs whose pairwise IMs satisfied
    the ``min_IMs`` threshold. Same shape that
    ``Timbre.with_intermod_sidebands`` expects to read off the bt's
    ``endogenous_intermodulations`` attribute.
    """
    from biotuner.peaks_extraction import endogenous_intermodulations

    if not peaks or len(peaks) < 2:
        return []
    if amps is None or len(amps) != len(peaks):
        # Uniform amps when missing or mismatched — biotuner's logic
        # only uses them to populate the metadata dict; pair selection
        # is purely frequency-driven.
        amps = [1.0] * len(peaks)
    try:
        _eims, imcs_all, _n = endogenous_intermodulations(
            list(peaks), list(amps),
            order=int(order),
            min_IMs=int(min_IMs),
            max_freq=float(max_freq),
        )
    except Exception as e:
        raise ValueError(f"intermod detection failed: {e}")
    pairs = imcs_all.get("peaks") or []
    out: List[List[float]] = []
    for pair in pairs:
        try:
            f1, f2 = float(pair[0]), float(pair[1])
            if f1 > 0 and f2 > 0:
                out.append([f1, f2])
        except (TypeError, ValueError, IndexError):
            continue
    return out


def compute_scale_on_demand(
    scale_name: str,
    peaks: List[float],
    *,
    sf: float = 1000.0,
    params: Optional[Dict[str, Any]] = None,
) -> List[float]:
    """Compute one of biotuner's derived scales from a peak list.

    Lets the Timbre tab populate any of the Scale-Source dropdown
    options on demand, mirroring the existing
    /api/timbre/extended-ratios pattern. Each branch constructs a
    fresh compute_biotuner, sets bt.peaks, and calls the relevant
    scale-construction method.

    Scales requiring derived state (diss_curve / HE) chain through
    peaks_extension first, since biotuner's scale extraction reads
    from bt.extended_peaks.

    Parameters
    ----------
    scale_name : str
        One of: 'peaks_ratios_cons', 'extended_peaks_ratios',
        'extended_peaks_ratios_cons', 'diss_scale', 'HE',
        'euler_fokker', 'harm_tuning', 'harm_fit'.
    peaks : list of float
        Peak frequencies in Hz from the analysis.
    sf : float
        Sampling rate. Most scales don't use it, but compute_biotuner
        requires it at construction.
    params : dict, optional
        Scale-specific overrides (cons_limit, n_harm, etc.).

    Returns
    -------
    list of float
        The computed scale ratios, with duplicates removed and sorted.
    """
    from biotuner.biotuner_object import compute_biotuner

    if not peaks or len(peaks) < 2:
        raise ValueError(
            f"compute_scale_on_demand: need at least 2 peaks (got {len(peaks)})"
        )
    cfg = params or {}
    bt = compute_biotuner(sf=float(sf))
    bt.peaks = list(peaks)
    # Dummy spectrum so internals that call peaks_to_amps don't crash.
    bt.freqs = np.linspace(0.1, sf / 2, 1024)
    bt.psd = np.full(1024, 1e-12, dtype=np.float64)
    # Always compute the peaks_extension first — most derived scales
    # depend on extended_peaks. Cheap; safe to skip-on-failure.
    try:
        bt.peaks_extension(
            method=cfg.get("extension_method", "harmonic_fit"),
            n_harm=int(cfg.get("n_harm", 10)),
            cons_limit=float(cfg.get("cons_limit", 0.05)),
            ratios_extension=True,
            scale_cons_limit=float(cfg.get("scale_cons_limit", 0.1)),
            harm_function="mult",
        )
    except Exception:
        # Continue — some scales work without extension.
        pass

    def _drain(arr) -> List[float]:
        a = np.asarray(arr, dtype=np.float64).ravel()
        a = a[np.isfinite(a) & (a > 0)]
        return sorted(set(float(x) for x in a))

    name = scale_name.lower()
    if name == "peaks_ratios_cons":
        # Already computed by peaks_extension above when ratios_extension=True.
        attr = getattr(bt, "extended_peaks_ratios_cons", None)
        if attr is None or len(attr) == 0:
            # Fall back: compute consonance filter over peaks_ratios.
            from biotuner.metrics import dyad_similarity
            ratios = sorted(set(
                p1 / p2 for i, p1 in enumerate(peaks) for p2 in peaks[i + 1:]
                if p2 > 0 and p1 / p2 > 1
            ))
            cons_thresh = float(cfg.get("scale_cons_limit", 0.1))
            attr = [r for r in ratios if dyad_similarity(r) >= cons_thresh]
        return _drain(attr)

    if name == "extended_peaks_ratios":
        return _drain(getattr(bt, "extended_peaks_ratios", []))
    if name == "extended_peaks_ratios_cons":
        return _drain(getattr(bt, "extended_peaks_ratios_cons", []))

    if name == "diss_scale":
        bt.compute_diss_curve(
            plot=False,
            input_type=cfg.get("input_type", "extended_peaks"),
            euler_comp=False,
            denom=int(cfg.get("max_denominator", 100)),
            max_ratio=float(cfg.get("max_ratio", 2.0)),
            n_tet_grid=int(cfg.get("n_tet_grid", 12)),
        )
        return _drain(getattr(bt, "diss_scale", []))

    if name in ("he", "harmonic_entropy", "he_scale"):
        bt.compute_harmonic_entropy(
            input_type=cfg.get("input_type", "extended_peaks"),
            res=float(cfg.get("res", 0.001)),
            spread=float(cfg.get("spread", 0.01)),
        )
        return _drain(getattr(bt, "HE_scale", []))

    if name == "euler_fokker":
        scale = bt.euler_fokker_scale(
            method=cfg.get("method", "peaks"),
            octave=int(cfg.get("octave", 2)),
        )
        return _drain(scale)

    if name in ("harm_tuning", "harmonic_tuning"):
        # bt.harmonic_tuning requires a list of harmonic positions.
        # When peaks were extracted via 'harmonic_recurrence' it auto-
        # populates self.all_harmonics; here we bypass that pipeline,
        # so derive a default list directly from the peaks: each
        # peak's index relative to the fundamental, rounded.
        list_harmonics = cfg.get("list_harmonics")
        if not list_harmonics:
            f0 = min(peaks)
            list_harmonics = sorted(set(
                max(1, int(round(p / f0))) for p in peaks if f0 > 0
            ))
            if not list_harmonics:
                list_harmonics = [1, 2, 3, 4, 5]
        scale = bt.harmonic_tuning(
            list_harmonics=list_harmonics,
            octave=int(cfg.get("octave", 2)),
            min_ratio=float(cfg.get("min_ratio", 1.0)),
            max_ratio=float(cfg.get("max_ratio", 2.0)),
        )
        return _drain(scale)

    if name in ("harm_fit", "harm_fit_tuning", "harmonic_fit_tuning"):
        scale = bt.harmonic_fit_tuning(
            n_harm=int(cfg.get("n_harm", 128)),
            bounds=float(cfg.get("bounds", 0.1)),
            n_common_harms=int(cfg.get("n_common_harms", 2)),
        )
        return _drain(scale)

    raise ValueError(
        f"Unknown scale '{scale_name}'. Known: peaks_ratios_cons, "
        "extended_peaks_ratios, extended_peaks_ratios_cons, diss_scale, "
        "HE, euler_fokker, harm_tuning, harm_fit"
    )


def compute_band_timbres(
    signal: List[float],
    sf: float,
    band_edges: List[float],
    *,
    n_peaks_per_band: int = 4,
) -> Dict[str, Any]:
    """Bandpass a signal into N consecutive frequency ranges and
    return one Timbre per band (peaks extracted from each).

    ``band_edges`` has length N+1 defining N bands — e.g.
    ``[4, 8, 13, 30, 100]`` produces four bands (delta, theta, alpha,
    beta-ish for an EEG signal).
    """
    from biotuner.harmonic_timbre.biotuner_mapping import timbres_from_bands

    sig = np.asarray(signal, dtype=np.float64).ravel()
    if sig.size < 100:
        raise ValueError(
            f"compute_band_timbres: signal too short ({sig.size} samples); "
            "need at least 100"
        )
    if len(band_edges) < 2:
        raise ValueError(
            "compute_band_timbres: band_edges must contain at least 2 Hz "
            "values (low, high)"
        )
    try:
        timbres = timbres_from_bands(
            sig, sf=float(sf), band_edges=list(band_edges),
            n_peaks_per_band=int(n_peaks_per_band),
        )
    except Exception as e:
        raise ValueError(f"band extraction failed: {e}")
    return {
        "bands":            [_timbre_to_dict(t) for t in timbres],
        "n_bands_requested": len(band_edges) - 1,
        "n_bands_kept":      len(timbres),
        "band_edges":        [float(e) for e in band_edges],
    }


def compute_imf_timbres(
    signal: List[float],
    sf: float,
    *,
    n_imfs: int = 5,
    n_peaks_per_imf: int = 4,
    method: str = "EMD",
) -> Dict[str, Any]:
    """Decompose a signal into IMFs and return one Timbre per IMF.

    Wraps biotuner's EMD + the new ``timbres_from_imfs`` adapter into
    a JSON-friendly response shape. Each entry is a Timbre dict
    (partials_hz, amplitudes, base_freq) that the wavetable endpoint
    can feed straight back into the ``imf_morph`` evolution.
    """
    from biotuner.peaks_extraction import EMD_eeg
    from biotuner.harmonic_timbre.biotuner_mapping import timbres_from_imfs

    sig = np.asarray(signal, dtype=np.float64).ravel()
    if sig.size < 100:
        raise ValueError(
            f"compute_imf_timbres: signal too short ({sig.size} samples); "
            "need at least 100 to run EMD"
        )
    try:
        imfs = EMD_eeg(sig, method=method, nIMFs=int(n_imfs), graph=False)
    except Exception as e:
        raise ValueError(f"EMD failed: {e}")
    timbres = timbres_from_imfs(
        list(imfs), sf=float(sf), n_peaks_per_imf=int(n_peaks_per_imf),
    )
    return {
        "imfs": [_timbre_to_dict(t) for t in timbres],
        "n_imfs_requested": int(n_imfs),
        "n_imfs_kept":      len(timbres),
        "method":           method,
    }


def _timbre_to_dict(timbre) -> Dict[str, Any]:
    """Compact JSON form of a Timbre for transport between endpoints."""
    return {
        "partials_hz": _arr_or_none(timbre.partials_hz) or [],
        "amplitudes":  _arr_or_none(timbre.amplitudes) or [],
        "base_freq":   float(timbre.base_freq),
    }


def compute_extended_ratios(req: Dict[str, Any]) -> Dict[str, Any]:
    """Run biotuner's peaks_extension over a peak list and return the
    resulting extended ratios.

    Useful when the original /api/analyze run didn't compute extension
    (e.g. the user picked a tuning method that doesn't require it).
    The Timbre tab calls this to populate the "Extended raw ratios" /
    "Extended consonant ratios" scale options without forcing a
    re-analysis of the whole signal.

    Implementation note: peaks_extension internally calls peaks_to_amps,
    which needs ``self.freqs`` and ``self.psd``. We don't have those
    here (the request only carries peaks), so we plug in dummy zeros
    of the right length. The amp computation degrades gracefully —
    extended ratios only depend on extended_peaks, not their amps.
    """
    from biotuner.biotuner_object import compute_biotuner

    peaks = req.get("peaks") or []
    if len(peaks) < 2:
        raise ValueError(
            "compute_extended_ratios requires at least 2 peaks "
            f"(got {len(peaks)})"
        )
    sf = float(req.get("sf") or 1000.0)
    n_harm        = int(req.get("n_harm", 10))
    method        = req.get("extension_method", "harmonic_fit")
    cons_limit    = float(req.get("cons_limit", 0.05))
    scale_cons    = float(req.get("scale_cons_limit", 0.1))

    bt = compute_biotuner(sf=sf)
    bt.peaks = list(peaks)
    # Dummy spectrum so peaks_to_amps doesn't crash. The values don't
    # matter for ratio extraction; biotuner uses them only to look up
    # amplitudes at extended_peak frequencies, which we discard.
    bt.freqs = np.linspace(0.1, sf / 2, 1024)
    bt.psd = np.full(1024, 1e-12, dtype=np.float64)
    try:
        bt.peaks_extension(
            method=method,
            n_harm=n_harm,
            cons_limit=cons_limit,
            ratios_extension=True,
            scale_cons_limit=scale_cons,
            harm_function="mult",
        )
    except Exception as e:
        raise ValueError(f"peaks_extension failed: {e}")

    def _list(attr):
        v = getattr(bt, attr, None)
        if v is None:
            return []
        arr = np.asarray(v).ravel()
        return [float(x) for x in arr if np.isfinite(x) and x > 0]

    return {
        "extended_peaks":             _list("extended_peaks"),
        "extended_peaks_ratios":      _list("extended_peaks_ratios"),
        "extended_peaks_ratios_cons": _list("extended_peaks_ratios_cons"),
        "n_harm": n_harm,
        "method": method,
    }


def compute_wavetable(req: Dict[str, Any]) -> Dict[str, Any]:
    """Build a multi-frame wavetable from the same Timbre pipeline as
    /api/timbre/compute, returning the raw frame samples as JSON so
    the frontend can render them as waveforms.

    Mirrors the per-evolution-mode helpers in
    biotuner.harmonic_timbre.exporters.to_wavetable so the rendered
    .wav export (when the user clicks Download .wavetable) is the
    exact same data the on-screen Wavetable Studio shows.

    Request shape:
      {
        ...TimbreComputeRequest fields...,
        'wavetable_config': {
          'n_frames': 32,           # 1, 16, 32, 64, 128
          'evolution': 'tilt',      # see _EVOLUTIONS
          'table_size': 512,        # samples per frame — defaults 512
          'tilt_range': [0, 2.5],
          'phase_range': [0, 6.283],
        }
      }
    """
    from biotuner.harmonic_timbre.exporters.to_wavetable import (
        _frame_with_tilt,
        _frame_with_active_partials,
        _frame_with_amp_morph,
        _frame_with_phase_sweep,
        _frame_with_intermod_sidebands,
        _frame_with_harmonic_stack,
        _frame_with_formant,
        _frame_with_wavefolding,
        _frame_with_fm_baked,
        _frame_with_noise_to_structure,
        _frame_with_timbre_morph,
        _frame_composite,
        WavetableLayer,
    )
    from biotuner.harmonic_timbre.timbre import Timbre as _Timbre
    from biotuner.harmonic_timbre.synthesis import render_wavetable_cycle

    timbre = _build_timbre_from_request(req)
    pseudo_bt = _build_pseudo_bt(req)

    cfg = req.get("wavetable_config") or {}
    n_frames     = int(cfg.get("n_frames", 32))
    evolution    = cfg.get("evolution", "tilt")
    table_size   = int(cfg.get("table_size", 512))
    tilt_range   = tuple(cfg.get("tilt_range",  [0.0, 2.5]))
    phase_range  = tuple(cfg.get("phase_range", [0.0, 2.0 * np.pi]))
    intermod_range = tuple(cfg.get("intermod_depth_range", [0.0, 0.6]))
    stack_range  = tuple(cfg.get("harmonic_stack_range", [0, 4]))

    n_frames = max(1, min(256, n_frames))
    table_size = max(64, min(2048, table_size))

    if n_frames == 1 or evolution == "none":
        frames_np = [render_wavetable_cycle(timbre, table_size=table_size)]
    elif evolution == "tilt":
        tilts = np.linspace(tilt_range[0], tilt_range[1], n_frames)
        frames_np = [_frame_with_tilt(timbre, float(t), table_size=table_size) for t in tilts]
    elif evolution == "harmonic_buildup":
        n = timbre.n_partials()
        ks = np.linspace(1, n, n_frames).astype(int)
        frames_np = [_frame_with_active_partials(timbre, int(k), table_size=table_size) for k in ks]
    elif evolution == "amp_morph":
        # NB: the helper takes a numpy.random.Generator, not a seed —
        # construct one with the user's seed (or 0) so morphs are
        # reproducible across re-runs.
        rng = np.random.default_rng(int(cfg.get("seed", 0)))
        ts = np.linspace(0, 1, n_frames)
        frames_np = [
            _frame_with_amp_morph(timbre, float(t), table_size=table_size, rng=rng)
            for t in ts
        ]
    elif evolution == "phase_sweep":
        phs = np.linspace(phase_range[0], phase_range[1], n_frames)
        frames_np = [_frame_with_phase_sweep(timbre, float(p), table_size=table_size) for p in phs]
    elif evolution == "intermod_buildup":
        depths = np.linspace(intermod_range[0], intermod_range[1], n_frames)
        frames_np = [
            _frame_with_intermod_sidebands(timbre, pseudo_bt, float(d), table_size=table_size)
            for d in depths
        ]
    elif evolution == "harmonic_stack":
        ns = np.linspace(stack_range[0], stack_range[1], n_frames).astype(int)
        rolloff = float(cfg.get("harmonic_stack_rolloff", 0.9))
        frames_np = [
            _frame_with_harmonic_stack(timbre, int(n), rolloff=rolloff, table_size=table_size)
            for n in ns
        ]
    elif evolution == "formant_sweep":
        fr = tuple(cfg.get("formant_center_range", [1000.0, 3000.0]))
        fw = float(cfg.get("formant_width_hz", 800.0))
        fg = float(cfg.get("formant_gain_db", 4.0))
        centers = np.linspace(fr[0], fr[1], n_frames)
        frames_np = [
            _frame_with_formant(timbre, float(c), fw, fg, table_size=table_size)
            for c in centers
        ]
    elif evolution == "wavefolding":
        # Buchla-style sin folder. fold_amount sweeps 0→4 across
        # frames by default; cycle-mode rendering matches export.
        fr = tuple(cfg.get("fold_range", [0.0, 4.0]))
        drive = float(cfg.get("output_drive", 1.0))
        folds = np.linspace(fr[0], fr[1], n_frames)
        frames_np = [
            _frame_with_wavefolding(timbre, float(f), table_size=table_size,
                                    output_drive=drive)
            for f in folds
        ]
    elif evolution == "fm_baked":
        # FM written into the per-frame cycle. cm_ratio defaults to
        # 2.0 (bell character); target_partial_idx defaults to 0
        # (modulate the fundamental); set to -1 to modulate all
        # partials for a denser sound.
        fr = tuple(cfg.get("fm_index_range", [0.0, 3.0]))
        cm = float(cfg.get("cm_ratio", 2.0))
        target = int(cfg.get("target_partial_idx", 0))
        indices = np.linspace(fr[0], fr[1], n_frames)
        frames_np = [
            _frame_with_fm_baked(timbre, float(b), table_size=table_size,
                                 cm_ratio=cm, target_partial_idx=target)
            for b in indices
        ]
    elif evolution == "noise_to_structure":
        # FOOOF-style decomposition made audible: frame 0 = 1/f^k
        # noise (using the timbre's spectral_tilt or override), frame
        # N = clean structured render. Seed for reproducibility.
        exponent = cfg.get("noise_exponent")
        seed = int(cfg.get("seed", 0))
        alphas = np.linspace(0.0, 1.0, n_frames)
        frames_np = [
            _frame_with_noise_to_structure(
                timbre, float(a), table_size=table_size,
                exponent=(float(exponent) if exponent is not None else None),
                seed=seed,
            )
            for a in alphas
        ]
    elif evolution in ("imf_morph", "band_morph"):
        # Walk through a sequence of Timbres. ``timbre_sequence`` is
        # a list of dicts (partials_hz / amplitudes / base_freq) the
        # frontend got from /api/timbre/imfs or /api/timbre/bands and
        # cached locally — passing them in lets the wavetable endpoint
        # avoid re-running EMD / bandpass on every parameter change.
        raw_seq = cfg.get("timbre_sequence") or []
        if not raw_seq:
            raise ValueError(
                f"evolution={evolution!r} requires 'timbre_sequence' in "
                "wavetable_config (a non-empty list of Timbre dicts)"
            )
        seq = [
            _Timbre(
                partials_hz=np.asarray(t["partials_hz"], dtype=np.float64),
                amplitudes=np.asarray(t["amplitudes"], dtype=np.float64),
                base_freq=float(t.get("base_freq", 1.0)),
            )
            for t in raw_seq
        ]
        blend = cfg.get("blend_mode", "linear_walk")
        sigma = float(cfg.get("gaussian_sigma", 0.5))
        frames_np = [
            _frame_with_timbre_morph(
                seq, i, n_frames, table_size=table_size,
                blend_mode=blend, gaussian_sigma=sigma,
            )
            for i in range(n_frames)
        ]
    elif evolution == "composite":
        # Multi-axis composite. ``layers`` is a list of dicts coming
        # from the frontend; each dict describes one axis (evolution +
        # weight curve + range + params). Coerce to the WavetableLayer
        # dataclass so the helper's validation runs.
        raw_layers = cfg.get("layers") or []
        if not raw_layers:
            raise ValueError(
                "evolution='composite' requires a non-empty 'layers' "
                "list in wavetable_config"
            )
        layers = [WavetableLayer(**lc) for lc in raw_layers]
        seed = int(cfg.get("seed", 0))
        frames_np = [
            _frame_composite(
                timbre, layers, i, n_frames,
                table_size=table_size, bt=pseudo_bt, seed=seed,
            )
            for i in range(n_frames)
        ]
    else:
        raise ValueError(f"Unknown wavetable evolution: {evolution!r}")

    # Normalise each frame to [-1, 1] for consistent visualisation,
    # then convert to list-of-lists for JSON. Keeps payload reasonable
    # for typical settings: 32 frames × 512 samples × 4 bytes ≈ 64 KB.
    frames_out = []
    for f in frames_np:
        arr = np.asarray(f, dtype=np.float32)
        peak = float(np.max(np.abs(arr)) or 1.0)
        frames_out.append([float(x / peak) for x in arr])

    return {
        "frames": frames_out,
        "n_frames": len(frames_out),
        "table_size": int(table_size),
        "evolution": evolution,
        # Tell the user what changed across frames (for tooltips).
        "evolution_label": {
            "tilt": "Spectral tilt 0 → 2.5",
            "harmonic_buildup": "Partials fade in one by one",
            "amp_morph": "Random → matched amplitudes",
            "phase_sweep": "Partial phases rotate 0 → 2π",
            "intermod_buildup": "Intermod sidebands fade in",
            "harmonic_stack": "Overtone stack 2f → nf fades in",
            "formant_sweep": "Formant center sweeps low → high",
            "wavefolding": "Sin-folder 0 → 4 (odd-harmonic enrichment)",
            "fm_baked": "Audio-rate FM index 0 → 3 (Bessel sidebands)",
            "composite": "Multi-axis: chained layer evolutions",
            "noise_to_structure": "Noise → structure: 1/f^k → clean Timbre",
            "imf_morph":          "IMF morph: walk through EMD modes high → low",
            "band_morph":         "Band morph: walk through frequency bands",
            "none": "Static (1 cycle)",
        }.get(evolution, evolution),
    }


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
