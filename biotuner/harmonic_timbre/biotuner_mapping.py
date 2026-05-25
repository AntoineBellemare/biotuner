"""biotuner.harmonic_timbre.biotuner_mapping — compute_biotuner -> Timbre fields.

Module type: Functions

The mapping layer that turns a fitted ``compute_biotuner`` instance into
a :class:`~biotuner.harmonic_timbre.Timbre`. Each Biotuner output has a
default mapping to a Timbre field, all opt-in via a ``use`` set.

Phase 1 covers v1 fields only:
    - partials (from peaks via the resolved ``scale``)
    - amplitudes (raw or consonance-weighted)
    - phases (Hilbert phases when available)
    - decay_times / bandwidths (from FOOOF linewidth when available)
    - spectral_tilt (from FOOOF aperiodic exponent when available)
    - noise_floor (from spectral flatness when available)
    - consonance weighting (always available; uses biotuner.metrics)

v1.1 fields (detuning, modulators, IMF layers, sequence sources) ship
in Phase 3 and are simply not requested here.

A field included in ``use`` but unavailable on the biotuner object is
logged at WARNING level and skipped silently — never an error.
"""

from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

from biotuner.harmonic_timbre._utils import (
    _coerce_ratios,
    normalize_amplitudes,
    resolve_scale,
)
from biotuner.harmonic_timbre.matching import match_timbre, _METHOD_TO_FUNC
from biotuner.harmonic_timbre.timbre import Timbre

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mapping sets
# ---------------------------------------------------------------------------

DEFAULT_MAPPING_V1: frozenset[str] = frozenset({
    "partials",
    "amplitudes",
    "phases",
    "decay_times",
    "bandwidths",
    "spectral_tilt",
    "noise_floor",
    "consonance_weighting",
})

ALL_MAPPINGS: frozenset[str] = DEFAULT_MAPPING_V1 | frozenset({
    # v1.1 fields — declared but the helpers raise NotImplementedError until Phase 3
    "detuning",
    "am_modulators",
    "fm_modulators",
    "layers_from_imf",
    "subharmonics",
    "peaks_extension",
    "harmonicity_weighting",
})


def _resolve_use(use) -> frozenset[str]:
    if use == "default":
        return DEFAULT_MAPPING_V1
    if use == "all":
        return ALL_MAPPINGS
    if isinstance(use, str):
        return frozenset({use})
    return frozenset(use)


# ---------------------------------------------------------------------------
# Per-mapping helpers (v1)
# ---------------------------------------------------------------------------

def map_peaks_to_partials(bt) -> np.ndarray | None:
    """Return absolute partial frequencies from biotuner peaks."""
    peaks = getattr(bt, "peaks", None)
    if peaks is None or len(peaks) == 0:
        return None
    return np.asarray(peaks, dtype=np.float64)


def map_amps_to_amplitudes(bt) -> np.ndarray | None:
    """Return per-peak amplitudes from biotuner."""
    amps = getattr(bt, "amps", None)
    if amps is None or len(amps) == 0:
        return None
    return np.asarray(amps, dtype=np.float64)


def map_phases(bt) -> np.ndarray | None:
    """Return per-peak phases (radians) — looks for ``bt.phases``.

    Only HilbertHuang-based pipelines populate this. Returns None
    otherwise (caller defaults to zero phases).
    """
    phases = getattr(bt, "phases", None)
    if phases is None:
        return None
    arr = np.asarray(phases, dtype=np.float64)
    if arr.size == 0:
        return None
    return arr


def map_linewidth_to_decay(bt) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (decay_times, bandwidths) per partial from FOOOF linewidth.

    Decay-time conversion: ``decay = 1 / (π · linewidth)`` (Lorentzian
    half-width relation; see
    :func:`biotuner.harmonic_geometry.harmonograph.derive_damping_from_linewidth`).
    Returns (None, None) if linewidth is not on the biotuner object.
    """
    lw = getattr(bt, "linewidth", None)
    if lw is None:
        # also try peaks_linewidth or fooof_linewidth
        for alt in ("peaks_linewidth", "fooof_linewidth", "bandwidths"):
            lw = getattr(bt, alt, None)
            if lw is not None:
                break
    if lw is None:
        return None, None
    arr = np.asarray(lw, dtype=np.float64)
    if arr.size == 0:
        return None, None
    # protect against 0 / negative linewidths
    safe = np.where(np.isfinite(arr) & (arr > 0), arr, np.nan)
    decay = 1.0 / (np.pi * safe)
    decay = np.where(np.isfinite(decay), decay, 1e6)  # "very long decay"
    return decay, arr


def map_aperiodic_to_tilt(bt) -> float | None:
    """Return FOOOF aperiodic exponent as the spectral tilt (1/f exponent).

    Looks for ``bt.aperiodic_exponent`` first, then falls back to
    ``bt.aperiodic_params[1]`` (FOOOF stores ``[offset, exponent]``).
    """
    exp = getattr(bt, "aperiodic_exponent", None)
    if exp is None:
        params = getattr(bt, "aperiodic_params", None)
        if params is None or len(params) < 2:
            return None
        try:
            exp = float(params[1])
        except Exception:
            return None
    try:
        v = float(exp)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def map_flatness_to_noise(bt) -> float | None:
    """Return spectral flatness or entropy clipped to [0, 1] as noise_floor.

    Tries ``bt.spectral_flatness``, then ``bt.spectral_entropy``.
    """
    val = getattr(bt, "spectral_flatness", None)
    if val is None:
        val = getattr(bt, "spectral_entropy", None)
    if val is None:
        return None
    try:
        v = float(val)
    except Exception:
        return None
    if not np.isfinite(v):
        return None
    return float(np.clip(v, 0.0, 1.0))


def map_harmonicity_weights(bt, *, metric: str = "dyad_similarity") -> np.ndarray | None:
    """Return per-peak amplitude weights from a harmonicity metric.

    Computes ``metric`` against each peak ratio relative to the lowest peak,
    and normalizes to a 0..1 weight vector. Useful for weighting the raw
    Biotuner amps by the consonance of each peak.
    """
    peaks = getattr(bt, "peaks", None)
    if peaks is None or len(peaks) == 0:
        return None

    from fractions import Fraction
    from biotuner.metrics import (
        dyad_similarity,
        compute_consonance,
        tenneyHeight,
        euler,
    )

    arr = np.asarray(peaks, dtype=np.float64)
    f0 = float(arr.min())
    if f0 <= 0:
        return None
    weights = []
    for p in arr:
        r = float(p) / f0
        if r <= 0:
            weights.append(0.0)
            continue
        try:
            if metric == "dyad_similarity":
                w = dyad_similarity(r)
            elif metric == "consonance":
                w = compute_consonance(r)
            elif metric == "tenney":
                w = 1.0 / max(tenneyHeight([1.0, r], avg=True), 1e-6)
            elif metric == "euler":
                frac = Fraction(r).limit_denominator(1000)
                w = 1.0 / max(euler(frac.numerator, frac.denominator), 1)
            else:
                raise ValueError(f"unknown metric {metric!r}")
            weights.append(float(w))
        except Exception:
            weights.append(0.0)
    out = np.asarray(weights, dtype=np.float64)
    if not np.all(np.isfinite(out)) or out.max() <= 0:
        return None
    return normalize_amplitudes(out)


def map_consonance_priors(bt) -> np.ndarray | None:
    """Alias for :func:`map_harmonicity_weights` with the default metric.

    Named differently to express its role: an *amplitude prior* derived
    from the consonance of the peaks. Used by ``timbre_from_biotuner``
    when ``'consonance_weighting'`` is requested.
    """
    return map_harmonicity_weights(bt, metric="dyad_similarity")


# ---------------------------------------------------------------------------
# v1.1 mappings — AM/FM modulators from biosignal-derived couplings
# ---------------------------------------------------------------------------

from biotuner.harmonic_timbre.timbre import Modulator


def _nearest_partial_idx(partials_hz: np.ndarray, target_hz: float) -> int:
    """Return the index of the partial closest to ``target_hz``."""
    return int(np.argmin(np.abs(np.asarray(partials_hz) - float(target_hz))))


def map_pac_to_am_modulators(
    bt,
    *,
    partials_hz: np.ndarray | None = None,
    coupling_threshold: float = 0.0,
    max_modulators: int = 16,
) -> list[Modulator]:
    """Phase-Amplitude Coupling → AM modulators.

    Each PAC entry says: "low-frequency oscillation at ``f_low`` modulates
    the amplitude of a high-frequency oscillation at ``f_high`` with
    coupling strength ``c``." This is exactly amplitude modulation:

        carrier      = the partial nearest f_high in the timbre
        mod_freq     = f_low                (rate of the amplitude swing)
        depth        = clip(c, 0..1)        (how strongly the AM swings)

    Reads ``bt.pac_freqs`` (list of (low, high) Hz pairs) and
    ``bt.pac_coupling`` (parallel coupling-strength values). Returns an
    empty list if neither attribute is populated.

    Parameters
    ----------
    partials_hz : array, optional
        The partial frequencies that AM ``carrier_idx`` will index into.
        If None, falls back to ``bt.peaks``.
    coupling_threshold : float, default=0.0
        Skip PAC pairs whose coupling is below this threshold.
    max_modulators : int, default=16
        Cap the returned list length (sorted by coupling strength, strongest first).
    """
    pac_freqs = getattr(bt, "pac_freqs", None)
    pac_coupling = getattr(bt, "pac_coupling", None)
    if pac_freqs is None or pac_coupling is None:
        return []
    if len(pac_freqs) == 0:
        return []
    if partials_hz is None:
        peaks = getattr(bt, "peaks", None)
        if peaks is None or len(peaks) == 0:
            return []
        partials_hz = np.asarray(peaks, dtype=np.float64)
    partials_hz = np.asarray(partials_hz, dtype=np.float64)

    # Normalize coupling to a unit-ish range. PAC outputs vary by metric;
    # we clip to [0, 1] for AM depth.
    coup_arr = np.asarray(pac_coupling, dtype=np.float64).flatten()

    pairs: list[tuple[float, Modulator]] = []
    for (low_high, c) in zip(pac_freqs, coup_arr):
        try:
            low, high = float(low_high[0]), float(low_high[1])
        except (TypeError, ValueError, IndexError):
            continue
        c = float(c)
        if not np.isfinite(c) or c <= coupling_threshold:
            continue
        depth = float(np.clip(c, 0.0, 1.0))
        idx = _nearest_partial_idx(partials_hz, high)
        m = Modulator(
            carrier_idx=idx,
            mod_freq=low,
            depth=depth,
            mod_type="AM",
            source=f"PAC[{low:.2f}->{high:.2f}]",
        )
        pairs.append((c, m))

    pairs.sort(key=lambda x: -x[0])
    return [m for _, m in pairs[:max_modulators]]


def map_cfc_to_fm_modulators(
    bt,
    *,
    partials_hz: np.ndarray | None = None,
    coupling_threshold: float = 0.0,
    max_modulators: int = 16,
    deviation_scale: float = 1.0,
) -> list[Modulator]:
    """Cross-Frequency Coupling → FM modulators.

    Reads ``bt.cfc_freqs`` (list of (low, high) Hz pairs) and
    ``bt.cfc_coupling`` (parallel coupling values). Each CFC pair becomes
    an FM modulator on the partial nearest the high frequency:

        carrier        = nearest partial to f_high
        mod_freq       = f_low
        depth (Hz dev) = coupling * deviation_scale * f_high
                         (so coupling=1 yields ≈ f_high deviation —
                          a strong, audibly modulated FM sound)

    If ``bt.cfc_freqs`` and ``bt.cfc_coupling`` aren't present, falls back
    to ``bt.pac_freqs``/``bt.pac_coupling`` (PAC and CFC overlap in
    practice; some pipelines only populate PAC).

    Parameters
    ----------
    deviation_scale : float, default=1.0
        Multiplier on the FM depth. ``1.0`` produces audibly strong FM;
        ``0.1`` produces subtle vibrato-like FM.
    """
    cfc_freqs = getattr(bt, "cfc_freqs", None) or getattr(bt, "pac_freqs", None)
    cfc_coupling = getattr(bt, "cfc_coupling", None) or getattr(bt, "pac_coupling", None)
    if cfc_freqs is None or cfc_coupling is None or len(cfc_freqs) == 0:
        return []
    if partials_hz is None:
        peaks = getattr(bt, "peaks", None)
        if peaks is None or len(peaks) == 0:
            return []
        partials_hz = np.asarray(peaks, dtype=np.float64)
    partials_hz = np.asarray(partials_hz, dtype=np.float64)
    coup_arr = np.asarray(cfc_coupling, dtype=np.float64).flatten()

    pairs: list[tuple[float, Modulator]] = []
    for (low_high, c) in zip(cfc_freqs, coup_arr):
        try:
            low, high = float(low_high[0]), float(low_high[1])
        except (TypeError, ValueError, IndexError):
            continue
        c = float(c)
        if not np.isfinite(c) or c <= coupling_threshold:
            continue
        depth_hz = float(np.clip(c, 0.0, 1.0)) * deviation_scale * high
        idx = _nearest_partial_idx(partials_hz, high)
        m = Modulator(
            carrier_idx=idx,
            mod_freq=low,
            depth=depth_hz,
            mod_type="FM",
            source=f"CFC[{low:.2f}->{high:.2f}]",
        )
        pairs.append((c, m))

    pairs.sort(key=lambda x: -x[0])
    return [m for _, m in pairs[:max_modulators]]


def map_intermod_to_modulators(
    bt,
    *,
    partials_hz: np.ndarray | None = None,
    mode: str = "AM",
    max_modulators: int = 16,
) -> list[Modulator]:
    """Endogenous intermodulation peaks → AM (or FM) modulators.

    Sidebands at f₁ ± f₂ are exactly what AM with carrier f₁ and
    modulator f₂ produces. This helper reverse-engineers that: each
    intermodulation entry ``(f1, f2)`` becomes a Modulator with
    ``carrier_idx`` = nearest partial to ``f1`` and ``mod_freq = f2``.

    Reads ``bt.endogenous_intermodulations`` (list of (f1, f2) pairs as
    written by :func:`biotuner.peaks_extraction.endogenous_intermodulations`)
    if available; otherwise returns an empty list.

    Parameters
    ----------
    mode : {'AM', 'FM'}
        Whether to produce AM or FM modulators. AM is the literal
        sideband interpretation; FM is the alternative reading where
        the same pair is treated as carrier and modulating frequency.
    """
    if mode not in ("AM", "FM"):
        raise ValueError(f"mode must be 'AM' or 'FM', got {mode!r}")
    intermod = getattr(bt, "endogenous_intermodulations", None)
    if intermod is None or len(intermod) == 0:
        return []
    if partials_hz is None:
        peaks = getattr(bt, "peaks", None)
        if peaks is None or len(peaks) == 0:
            return []
        partials_hz = np.asarray(peaks, dtype=np.float64)
    partials_hz = np.asarray(partials_hz, dtype=np.float64)

    out: list[Modulator] = []
    for entry in intermod:
        try:
            f1, f2 = float(entry[0]), float(entry[1])
        except (TypeError, ValueError, IndexError):
            continue
        if f1 <= 0 or f2 <= 0:
            continue
        idx = _nearest_partial_idx(partials_hz, f1)
        if mode == "AM":
            depth = 0.5
        else:
            depth = f1 * 0.1   # 10% deviation by default
        out.append(Modulator(
            carrier_idx=idx,
            mod_freq=f2,
            depth=depth,
            mod_type=mode,
            source=f"intermod[{f1:.2f}_{f2:.2f}]",
        ))
        if len(out) >= max_modulators:
            break
    return out


def map_imfs_to_layers(bt, **kw):  # pragma: no cover — Phase 3 (layered timbres)
    raise NotImplementedError(
        "map_imfs_to_layers ships with the layered-timbre rendering work."
    )


def map_higuchi_to_detuning(bt):  # pragma: no cover — Phase 3
    raise NotImplementedError(
        "map_higuchi_to_detuning ships with the per-partial detuning rendering work."
    )


# ---------------------------------------------------------------------------
# Top-level constructors
# ---------------------------------------------------------------------------

def timbre_from_ratios(
    ratios,
    *,
    matching_method: str = "consonance_weighted",
    base_freq: float = 1.0,
    **matching_kwargs,
) -> Timbre:
    """Build a Timbre from raw ratios — no compute_biotuner instance needed.

    Equivalent to ``match_timbre(ratios, method=matching_method, ...)`` plus
    a provenance tag. Composers using only ``scale_construction`` (or
    hand-authored scales) get full module functionality through this.
    """
    timbre = match_timbre(
        ratios,
        method=matching_method,
        base_freq=base_freq,
        **matching_kwargs,
    )
    timbre.metadata.setdefault("scale_source", "raw")
    return timbre


def timbre_from_biotuner(
    bt,
    *,
    scale: str | Iterable[float] | None = "peaks_ratios_cons",
    use="default",
    matching_method: str = "consonance_weighted",
    base_freq: float | None = None,
    scale_kwargs: dict | None = None,
    **matching_kwargs,
) -> Timbre:
    """Build a Timbre from any ratio set produced by a fitted ``compute_biotuner``.

    ``scale`` selects which Biotuner ratio attribute (or method-backed scale)
    to anchor the Timbre to — peaks, dissonance-curve minima, harmonic
    tunings, Euler-Fokker, harmonic-entropy minima, etc. See
    :data:`~biotuner.harmonic_timbre._utils.SCALE_SOURCES`.

    The chosen scale is recorded in
    ``Timbre.metadata['scale_source']`` and the resolved ratios are
    recorded as ``Timbre.matched_tuning``. Pass an iterable of ratios
    directly for ``scale`` to skip Biotuner ratio selection entirely; in
    that case ``bt`` is used only for secondary parameters (phases,
    FOOOF, PAC, IMFs, …) selected via ``use``.

    Parameters
    ----------
    bt : compute_biotuner
        Fitted biotuner instance.
    scale : str | iterable | None
        Scale source; see SCALE_SOURCES. ``None`` walks the default fallback
        chain (peaks_ratios_cons → peaks_ratios → ...).
    use : str | iterable of str
        Which secondary fields to pull. ``'default'`` = v1 set. ``'all'`` =
        v1 + v1.1; v1.1 fields will warn if requested in Phase 1.
    matching_method : str
        Forwarded to :func:`~biotuner.harmonic_timbre.matching.match_timbre`.
    base_freq : float, optional
        Reference frequency. If None, inferred from biotuner peaks
        (``min(bt.peaks)`` if available, else 1.0).
    scale_kwargs : dict, optional
        Forwarded to ``resolve_scale``.
    **matching_kwargs
        Forwarded to the chosen matching function.

    Returns
    -------
    Timbre
        With ``metadata['biotuner_fields_used']`` listing every successfully-
        applied mapping, and ``metadata['scale_source']`` recording origin.
    """
    use_set = _resolve_use(use)
    if matching_method not in _METHOD_TO_FUNC:
        raise ValueError(
            f"timbre_from_biotuner: unknown matching method {matching_method!r}"
        )

    # Resolve the ratio set
    ratios, provenance = resolve_scale(
        scale, bt=bt, call_kwargs=scale_kwargs,
    )
    ratios = list(ratios)

    # Infer base_freq from biotuner peaks if not given
    if base_freq is None:
        peaks = getattr(bt, "peaks", None)
        if peaks is not None and len(peaks) > 0:
            base_freq = float(np.min(np.asarray(peaks, dtype=np.float64)))
        else:
            base_freq = 1.0

    # Build the base timbre via the chosen matching method
    base_timbre = match_timbre(
        ratios,
        method=matching_method,
        base_freq=base_freq,
        **matching_kwargs,
    )

    # Now overlay biosignal-derived secondary parameters from `use`
    fields_used: list[str] = []
    overrides: dict = {}

    if "amplitudes" in use_set:
        bt_amps = map_amps_to_amplitudes(bt)
        if bt_amps is not None and bt_amps.size == base_timbre.n_partials():
            # only override when shapes line up — otherwise the matched-method
            # amplitudes already encode consonance weighting and are preferred
            overrides["amplitudes"] = normalize_amplitudes(bt_amps)
            fields_used.append("amplitudes")

    if "consonance_weighting" in use_set:
        if matching_method == "consonance_weighted":
            # the matching method itself already encodes consonance weighting;
            # no separate amplitude override needed.
            fields_used.append("consonance_weighting")
        else:
            # try to weight raw amps by per-peak consonance priors. Only
            # applies when the timbre's partials line up 1:1 with bt.peaks
            # (e.g. matching_method='direct' with len(ratios) == len(peaks)).
            weights = map_consonance_priors(bt)
            amps = overrides.get("amplitudes", base_timbre.amplitudes)
            if weights is not None and weights.size == amps.size:
                combined = normalize_amplitudes(amps * weights)
                overrides["amplitudes"] = combined
                fields_used.append("consonance_weighting")
            elif weights is not None:
                logger.debug(
                    "consonance_weighting: weights shape %s != amps shape %s; "
                    "skipping silently",
                    weights.shape, amps.shape,
                )

    if "phases" in use_set:
        bt_phases = map_phases(bt)
        if bt_phases is not None and bt_phases.size == base_timbre.n_partials():
            overrides["phases"] = bt_phases
            fields_used.append("phases")
        elif bt_phases is None:
            logger.debug("phases not available on biotuner instance — skipped")

    if "decay_times" in use_set or "bandwidths" in use_set:
        decay, bw = map_linewidth_to_decay(bt)
        if "decay_times" in use_set and decay is not None and decay.size == base_timbre.n_partials():
            overrides["decay_times"] = decay
            fields_used.append("decay_times")
        if "bandwidths" in use_set and bw is not None and bw.size == base_timbre.n_partials():
            overrides["bandwidths"] = bw
            fields_used.append("bandwidths")

    if "spectral_tilt" in use_set:
        tilt = map_aperiodic_to_tilt(bt)
        if tilt is not None:
            overrides["spectral_tilt"] = tilt
            fields_used.append("spectral_tilt")

    if "noise_floor" in use_set:
        nf = map_flatness_to_noise(bt)
        if nf is not None:
            overrides["noise_floor"] = nf
            fields_used.append("noise_floor")

    # Warn about v1.1 requests in Phase 1
    v11 = use_set - DEFAULT_MAPPING_V1
    for name in v11:
        logger.warning(
            "use=%r requested but it is a v1.1 mapping (Phase 3); ignored.",
            name,
        )

    # Apply overrides via with_partials (immutable-style)
    timbre = base_timbre.with_partials(**overrides) if overrides else base_timbre
    # Update metadata directly (with_partials preserves metadata dict)
    timbre.metadata["scale_source"] = provenance
    timbre.metadata["biotuner_fields_used"] = fields_used
    timbre.metadata["use"] = sorted(use_set)
    timbre.matched_tuning = list(ratios)
    timbre.validate()
    return timbre


# ---------------------------------------------------------------------------
# Signal → Timbre-sequence adapters (feed _frame_with_timbre_morph)
# ---------------------------------------------------------------------------
#
# Each function returns a list of Timbres whose spectral content
# corresponds to a frequency slice of the input signal — either
# data-driven (one Timbre per EMD intrinsic mode function) or
# theory-driven (one Timbre per bandpassed Hz range). The lists feed
# the ``imf_morph`` and ``band_morph`` wavetable evolutions in
# ``to_wavetable``.

def _peaks_from_spectrum(
    signal: np.ndarray,
    sf: float,
    *,
    n_peaks: int = 6,
    min_freq: float = 0.5,
    max_freq: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return up to ``n_peaks`` (freq, magnitude) pairs from a signal's
    Welch spectrum, sorted by magnitude descending. Used by both the
    IMF and band adapters below; consolidates the spectrum-to-peaks
    logic so the two paths stay consistent.
    """
    from scipy.signal import welch, find_peaks
    if max_freq is None:
        max_freq = sf / 2.0
    nperseg = min(len(signal), max(256, int(sf * 2)))
    f, p = welch(signal, sf, nperseg=nperseg)
    mask = (f >= float(min_freq)) & (f <= float(max_freq))
    f = f[mask]; p = p[mask]
    if f.size == 0 or p.size == 0 or float(p.max()) <= 0:
        return np.array([]), np.array([])
    peaks_idx, _ = find_peaks(p, height=float(p.max()) * 0.05)
    if peaks_idx.size == 0:
        peaks_idx = np.array([int(np.argmax(p))])
    order = np.argsort(-p[peaks_idx])[:int(n_peaks)]
    sel = peaks_idx[order]
    return f[sel].astype(np.float64), p[sel].astype(np.float64)


def timbres_from_imfs(
    imfs,
    sf: float,
    *,
    n_peaks_per_imf: int = 4,
    drop_empty: bool = True,
) -> list:
    """Build one Timbre per EMD intrinsic mode function.

    For each IMF, runs a Welch PSD + peak finder to extract up to
    ``n_peaks_per_imf`` partial frequencies and amplitudes. Returns a
    list ordered from highest-frequency IMF (typically index 0 in
    EMD output) to lowest, matching the natural EMD ordering
    convention used by :func:`biotuner.peaks_extraction.EMD_eeg`.

    Empty IMFs (silent or constant) are dropped when ``drop_empty=True``.
    """
    out = []
    for imf in imfs:
        arr = np.asarray(imf, dtype=np.float64)
        if arr.size < 4 or np.all(arr == arr[0]):
            continue
        freqs, mags = _peaks_from_spectrum(
            arr, sf, n_peaks=n_peaks_per_imf,
            min_freq=0.5, max_freq=sf / 2.0,
        )
        if freqs.size == 0:
            if not drop_empty:
                continue
            continue
        amps = mags / max(float(mags.max()), 1e-9)
        out.append(Timbre(
            partials_hz=freqs,
            amplitudes=amps,
            base_freq=float(freqs.min()) if freqs.size else 1.0,
        ))
    return out


def timbres_from_bands(
    signal,
    sf: float,
    band_edges,
    *,
    n_peaks_per_band: int = 4,
    drop_empty: bool = True,
) -> list:
    """Build one Timbre per consecutive frequency band of the signal.

    ``band_edges`` is a list of N+1 Hz values defining N bands:
    edges ``[4, 8, 13, 30, 100]`` produces 4 bands (4-8, 8-13, 13-30,
    30-100). The signal is bandpassed into each range and its peaks
    extracted into a Timbre.

    Returns one Timbre per band, in the same order as the edges.
    Empty bands (no spectral energy in range) are dropped when
    ``drop_empty=True``.
    """
    from scipy.signal import butter, sosfiltfilt
    edges = list(band_edges)
    if len(edges) < 2:
        raise ValueError(
            "timbres_from_bands: band_edges must contain at least 2 values "
            "(low, high) — got %r" % (edges,)
        )
    out = []
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if lo <= 0 or hi <= lo or hi >= sf / 2.0:
            continue
        sos = butter(4, [lo, hi], btype="bandpass", fs=sf, output="sos")
        try:
            band = sosfiltfilt(sos, signal)
        except Exception:
            continue
        freqs, mags = _peaks_from_spectrum(
            band, sf, n_peaks=n_peaks_per_band, min_freq=lo, max_freq=hi,
        )
        if freqs.size == 0:
            continue
        amps = mags / max(float(mags.max()), 1e-9)
        out.append(Timbre(
            partials_hz=freqs,
            amplitudes=amps,
            base_freq=float(freqs.min()) if freqs.size else 1.0,
        ))
    return out
