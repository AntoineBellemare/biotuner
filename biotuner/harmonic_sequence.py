"""
biotuner/harmonic_sequence.py
==============================
Temporal harmonic-structure learning from successive biotuner outputs.

Six complementary approaches build on the outputs of compute_biotuner to
learn or model the way harmonic content evolves over time:

    1. HarmonicMarkov          – Discrete-state Markov chain / HMM
    2. WassersteinTrajectory   – Optimal-transport trajectory in interval space
    3. HarmonicDMD             – Linear dynamical modes of metric vectors
    4. HarmonicLatentSpace     – Smooth manifold of harmonic states (PCA)
    6. HarmonicTopology        – Topological shape of harmonic trajectories (TDA)
    7. HarmonicGrammar         – N-gram / symbolic sequence model

All approaches are exposed through the orchestrating class
``HarmonicSequenceAnalyzer``, which extracts representations from a list
of ``compute_biotuner`` objects and feeds them to each model.

Quick start
-----------
>>> from biotuner.harmonic_sequence import HarmonicSequenceAnalyzer
>>> analyzer = HarmonicSequenceAnalyzer.from_biotuner_list(
...     bt_list, tuning="peaks_ratios"
... )
>>> analyzer.fit_all()
>>> print(analyzer.summary())

Module-level encoding functions (``encode_histograms``, ``encode_scalar_metrics``,
``encode_ji_matrix``) can also be imported and used standalone.

Optional dependencies
---------------------
- scikit-learn  : KMeans, PCA, MDS, StandardScaler – required for
                  HarmonicMarkov, HarmonicLatentSpace, WassersteinTrajectory.embed(),
                  and HarmonicDMD.fit(use_histograms=True).
- hmmlearn      : Gaussian HMM backend for HarmonicMarkov (``use_hmm=True``).
- ripser        : Vietoris-Rips persistent homology for HarmonicTopology;
                  gracefully falls back to H0-only via scipy linkage.
"""
from __future__ import annotations

import warnings
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.cluster.hierarchy import linkage as _sp_linkage
from scipy.stats import wasserstein_distance as _sp_wasserstein

from biotuner.metrics import ratios2harmsim
from biotuner.dictionaries import interval_names as _interval_names_dict

# ── optional deps ─────────────────────────────────────────────────────────────
try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.manifold import MDS
    from sklearn.preprocessing import StandardScaler

    _SKLEARN = True
except ImportError:
    _SKLEARN = False
    warnings.warn(
        "scikit-learn not available. Install with: pip install scikit-learn. "
        "HarmonicMarkov, HarmonicLatentSpace, and WassersteinTrajectory.embed() "
        "will be unavailable.",
        ImportWarning,
        stacklevel=2,
    )

try:
    import hmmlearn.hmm as _hmmlearn_hmm

    _HMMLEARN = True
except ImportError:
    _HMMLEARN = False

try:
    from ripser import ripser as _ripser_fn

    _RIPSER = True
except ImportError:
    _RIPSER = False


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

TUNING_ATTRS: Dict[str, str] = {
    "peaks_ratios":          "peaks_ratios",
    "peaks_ratios_cons":     "peaks_ratios_cons",
    "diss_scale":            "diss_scale",
    "HE_scale":              "HE_scale",
    "euler_fokker":          "euler_fokker",
    "harm_tuning":           "harm_tuning_scale",
    "harm_fit_tuning":       "harm_fit_tuning_scale",
    "extended_peaks_ratios": "extended_peaks_ratios",
}
"""Maps user-facing tuning names to attribute names on compute_biotuner."""

# Ordered feature keys extracted from biotuner objects
_PEAKS_METRIC_KEYS: Tuple[str, ...] = (
    "cons", "tenney", "harmsim", "harm_fit",
)
_SCALE_METRIC_KEYS: Tuple[str, ...] = (
    "diss_euler", "dissonance", "diss_harm_sim", "diss_n_steps",
    "HE", "HE_n_steps", "HE_harm_sim",
)

# Build JI catalog: list of (name, cents) from the biotuner dictionaries
_JI_CATALOG: List[Tuple[str, float]] = [
    (name, float(vals["Cents"]))
    for name, vals in _interval_names_dict.items()
]
_JI_CATALOG_CENTS: np.ndarray = np.array([c for _, c in _JI_CATALOG])


# ─────────────────────────────────────────────────────────────────────────────
# I.  SHARED ENCODING LAYER  (stateless, importable separately)
# ─────────────────────────────────────────────────────────────────────────────

def extract_tuning(bt_obj: Any, tuning: str = "peaks_ratios") -> List[float]:
    """Extract a named tuning from a fitted ``compute_biotuner`` object.

    Parameters
    ----------
    bt_obj : compute_biotuner
        A biotuner object after ``peaks_extraction()`` (and optionally
        ``compute_diss_curve()``, ``compute_harmonic_entropy()``, etc.).
    tuning : str
        One of the keys in :data:`TUNING_ATTRS`.

    Returns
    -------
    list of float
        Positive finite ratios, or an empty list if the attribute is absent.
    """
    attr = TUNING_ATTRS.get(tuning, tuning)
    val = getattr(bt_obj, attr, None)
    if val is None:
        return []
    out = []
    for r in val:
        try:
            v = float(r)
            if np.isfinite(v) and v > 0:
                out.append(v)
        except (TypeError, ValueError):
            pass
    return out


def _ratios_to_hist(
    ratios: List[float],
    n_bins: int,
    min_cents: float,
    max_cents: float,
) -> np.ndarray:
    """Normalised cents histogram for a single ratio set (internal helper)."""
    if not ratios:
        return np.zeros(n_bins)
    cents = []
    for r in ratios:
        if r > 0:
            try:
                c = 1200.0 * np.log2(r)
                if np.isfinite(c):
                    cents.append(c)
            except (ValueError, ZeroDivisionError):
                pass
    if not cents:
        return np.zeros(n_bins)
    bins = np.linspace(min_cents, max_cents, n_bins + 1)
    hist, _ = np.histogram(cents, bins=bins)
    hist = hist.astype(float)
    s = hist.sum()
    return hist / s if s > 0 else hist


def encode_histograms(
    ratios_list: List[List[float]],
    n_bins: int = 240,
    min_cents: float = 0.0,
    max_cents: float = 1200.0,
) -> np.ndarray:
    """Convert a sequence of ratio sets to a cents-histogram matrix.

    Parameters
    ----------
    ratios_list : list of list of float, length T
    n_bins : int, default=240
        Number of bins (5 cents/bin at default range).
    min_cents, max_cents : float
        Histogram range in cents.

    Returns
    -------
    X : ndarray, shape (T, n_bins)
        Rows normalised to sum to 1 (zero-rows for empty ratio sets).
    """
    return np.stack(
        [_ratios_to_hist(r, n_bins, min_cents, max_cents) for r in ratios_list],
        axis=0,
    )


def _safe_float(val: Any) -> float:
    """Safely coerce a metric value (possibly a NaN string) to float."""
    if isinstance(val, str):
        return np.nan
    try:
        v = float(val)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def encode_scalar_metrics(bt_list: List[Any]) -> np.ndarray:
    """Build a scalar feature matrix from a list of ``compute_biotuner`` objects.

    Extracts and concatenates:

    - ``peaks_metrics``: cons, tenney, harmsim, harm_fit, subharm_tension
    - ``scale_metrics``: diss_euler, dissonance, diss_harm_sim, diss_n_steps,
      HE, HE_n_steps, HE_harm_sim
    - Mean peak frequency and peak count.

    Parameters
    ----------
    bt_list : list of compute_biotuner

    Returns
    -------
    X : ndarray, shape (T, D)
        Contains NaN wherever an attribute was absent or not computed.
    """
    rows = []
    for bt in bt_list:
        row = []
        pm = getattr(bt, "peaks_metrics", {}) or {}
        for k in _PEAKS_METRIC_KEYS:
            row.append(_safe_float(pm.get(k, np.nan)))
        row.append(_safe_float(pm.get("subharm_tension", np.nan)))
        sm = getattr(bt, "scale_metrics", {}) or {}
        for k in _SCALE_METRIC_KEYS:
            row.append(_safe_float(sm.get(k, np.nan)))
        peaks = getattr(bt, "peaks", None)
        if peaks is not None and len(peaks) > 0:
            row.extend([float(np.mean(peaks)), float(len(peaks))])
        else:
            row.extend([np.nan, np.nan])
        rows.append(row)
    return np.array(rows, dtype=float)


# Module-level cache for the freq-only dyad-similarity matrix.
# Key: (n_freqs, fmin_int, fmax_int, precision_int, metric, n_harms, delta_lim, min_notes).
# Value: the F×F dyad-similarity matrix.  This matrix depends ONLY on the
# frequency axis, so it can be reused across every window in a session and
# across calls with matching parameters.
_DYAD_MATRIX_CACHE: Dict[Tuple, np.ndarray] = {}


def clear_harmonicity_cache() -> None:
    """Empty the module-level dyad-matrix cache (forces recomputation)."""
    _DYAD_MATRIX_CACHE.clear()


def _dyad_matrix_cache_key(
    freqs: np.ndarray,
    metric: str,
    n_harms: int,
    delta_lim: float,
    min_notes: int,
) -> Tuple:
    """Build a hashable key from analysis parameters.

    The frequency array is summarised by its length plus its first/last/step,
    which uniquely determines a uniform grid.
    """
    n = len(freqs)
    f0 = float(freqs[0]) if n else 0.0
    fN = float(freqs[-1]) if n else 0.0
    return (
        n,
        round(f0, 6),
        round(fN, 6),
        metric,
        int(n_harms),
        float(delta_lim),
        int(min_notes),
    )


def _get_dyad_similarity_matrix(
    freqs: np.ndarray,
    metric: str = "harmsim",
    n_harms: int = 10,
    delta_lim: float = 20.0,
    min_notes: int = 2,
) -> np.ndarray:
    """Get the freq-only dyad-similarity matrix, computing once and caching."""
    from biotuner.harmonic_spectrum import harmonicity_matrices

    key = _dyad_matrix_cache_key(freqs, metric, n_harms, delta_lim, min_notes)
    M = _DYAD_MATRIX_CACHE.get(key)
    if M is None:
        M = harmonicity_matrices(
            freqs, metric=metric, n_harms=n_harms,
            delta_lim=delta_lim, min_notes=min_notes,
        )
        _DYAD_MATRIX_CACHE[key] = M
    return M


def _harmonicity_from_psd(
    psd_clean: np.ndarray,
    dyad_matrix: np.ndarray,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised replacement for compute_harmonic_power.

    Replaces the F²-element Python double loop with two NumPy outer-products,
    producing identical numerical output.

    Returns
    -------
    H_values : ndarray, shape (F,)   per-frequency harmonicity
    H_matrix : ndarray, shape (F, F) power-weighted dyad-similarity matrix
    """
    psd = np.asarray(psd_clean, dtype=float)
    total = float(psd.sum())
    if total <= 0:
        F = len(psd)
        return np.zeros(F), np.zeros((F, F))
    PP = np.outer(psd, psd)
    np.fill_diagonal(PP, 0.0)              # exclude i == j
    H_matrix = (dyad_matrix * PP) / total
    weighted_sum = (dyad_matrix * PP).sum(axis=1)
    H_values = weighted_sum / (2.0 * total) if normalize else weighted_sum
    return H_values, H_matrix


def _compute_harmonicity_from_signal(
    signal: np.ndarray,
    *,
    fs: float,
    fmin: float,
    fmax: float,
    precision_hz: float,
    metric: str,
    n_harms: int,
    smoothness: int,
    smoothness_harm: int,
    normalize: bool,
    power_law_remove: bool,
    delta_lim: float = 20.0,
    min_notes: int = 2,
    return_matrix: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Fast harmonicity computation for one window — no DataFrame, no phase.

    Compared to :func:`biotuner.harmonic_spectrum.compute_global_harmonicity`,
    this skips:

      - phase / PLV / resonance computations (we don't use them),
      - peak-finding (3 calls), spectral-flatness/entropy/Higuchi metrics,
      - DataFrame construction,
      - per-window dyad-similarity matrix recomputation (cached).

    Returns
    -------
    H_values : ndarray, shape (F,)
    H_matrix : ndarray, shape (F, F) or None
    freqs    : ndarray, shape (F,)
    """
    from scipy.ndimage import gaussian_filter
    from biotuner.biotuner_utils import (
        apply_power_law_remove,
        compute_frequency_and_psd,
    )

    freqs, psd = compute_frequency_and_psd(
        np.asarray(signal, dtype=float),
        precision_hz,
        smoothness=smoothness,
        fs=fs,
        noverlap=1,
        fmin=fmin,
        fmax=fmax,
    )
    psd_clean = apply_power_law_remove(freqs, psd, power_law_remove)
    psd_min = np.min(psd_clean)
    psd_max = np.max(psd_clean)
    rng = psd_max - psd_min
    if rng > 0:
        psd_clean = (psd_clean - psd_min) / rng

    dyad = _get_dyad_similarity_matrix(
        freqs, metric=metric, n_harms=n_harms,
        delta_lim=delta_lim, min_notes=min_notes,
    )

    H_values, H_matrix = _harmonicity_from_psd(
        psd_clean, dyad, normalize=normalize
    )
    # Match compute_global_harmonicity: gaussian_filter is applied
    # unconditionally (default sigma=1 in the original is NOT a no-op).
    H_values = gaussian_filter(H_values, smoothness_harm)
    return H_values, (H_matrix if return_matrix else None), freqs


def _bt_cache_key(
    fmin: float, fmax: float, precision_hz: float,
    metric: str, n_harms: int, smoothness: int, smoothness_harm: int,
    normalize: bool, power_law_remove: bool, fs: float,
) -> Tuple:
    """Per-bt-object cache key — only re-uses cache when all params match."""
    return (
        round(fmin, 6), round(fmax, 6), round(precision_hz, 6),
        metric, int(n_harms), int(smoothness), int(smoothness_harm),
        bool(normalize), bool(power_law_remove), round(fs, 6),
    )


def _encode_harmonicity_core(
    bt_list: List[Any],
    *,
    return_matrix: bool,
    fmin: float = 1.0,
    fmax: float = 30.0,
    precision_hz: float = 0.5,
    metric: str = "harmsim",
    n_harms: int = 10,
    smoothness: int = 1,
    smoothness_harm: int = 1,
    normalize: bool = True,
    power_law_remove: bool = False,
    fs: Optional[float] = None,
    cache: bool = True,
    progress: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Shared fast path for both encoders.  Caches results on bt objects."""
    H_rows: List[np.ndarray] = []
    M_stack: List[np.ndarray] = []
    freqs_ref: Optional[np.ndarray] = None

    for i, bt in enumerate(bt_list):
        if progress:
            print(f"  [harmonicity] window {i + 1}/{len(bt_list)}")
        data = getattr(bt, "data", None)
        if data is None:
            raise ValueError(
                f"bt_list[{i}] has no .data attribute. "
                "Harmonicity encoders need the raw signal."
            )
        sf_i = float(fs if fs is not None else getattr(bt, "sf", 1000.0))
        key = _bt_cache_key(
            fmin, fmax, precision_hz, metric, n_harms,
            smoothness, smoothness_harm, normalize, power_law_remove, sf_i,
        )

        cached = getattr(bt, "_harm_cache", None) if cache else None
        hit = (cached is not None and cached.get("key") == key
               and (not return_matrix or cached.get("matrix") is not None))
        if hit:
            H_vec = cached["values"]
            M = cached.get("matrix")
            freqs = cached["freqs"]
        else:
            H_vec, M, freqs = _compute_harmonicity_from_signal(
                np.asarray(data, dtype=float),
                fs=sf_i,
                fmin=fmin, fmax=fmax,
                precision_hz=precision_hz,
                metric=metric,
                n_harms=n_harms,
                smoothness=smoothness,
                smoothness_harm=smoothness_harm,
                normalize=normalize,
                power_law_remove=power_law_remove,
                return_matrix=return_matrix,
            )
            if cache:
                bt._harm_cache = {
                    "key": key, "values": H_vec, "matrix": M, "freqs": freqs,
                }

        H_rows.append(H_vec)
        if return_matrix:
            M_stack.append(M)
        if freqs_ref is None:
            freqs_ref = freqs

    H = np.stack(H_rows, axis=0)
    M_out = np.stack(M_stack, axis=0) if return_matrix else None
    return H, M_out, (freqs_ref if freqs_ref is not None else np.array([]))


def encode_harmonicity_spectrum(
    bt_list: List[Any],
    *,
    fmin: float = 1.0,
    fmax: float = 30.0,
    precision_hz: float = 0.5,
    metric: str = "harmsim",
    n_harms: int = 10,
    smoothness: int = 1,
    smoothness_harm: int = 1,
    normalize: bool = True,
    power_law_remove: bool = False,
    fs: Optional[float] = None,
    cache: bool = True,
    progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the 1-D harmonicity spectrum per window (fast path).

    For each ``compute_biotuner`` object, computes the per-frequency
    power-weighted dyad-similarity values.  This is much faster than the
    full :func:`compute_global_harmonicity` because it:

      - reuses the freq-only dyad-similarity matrix across all windows
        (it depends only on the frequency axis),
      - skips phase, PLV, resonance, peak-finding, entropy, and DataFrame
        overhead (none of which we need here),
      - caches results on each bt object under ``bt._harm_cache``.

    Parameters
    ----------
    bt_list : list of compute_biotuner
        Each ``bt`` must have ``bt.data`` (the raw signal) and ``bt.sf``.
    fmin, fmax : float          analysis range in Hz.
    precision_hz : float        frequency resolution; F = (fmax-fmin)/precision_hz.
    metric : str                'harmsim' or 'subharm_tension'.
    n_harms : int               number of harmonics in dyad similarity.
    smoothness, smoothness_harm : int   PSD / harmonicity-curve smoothing.
    normalize : bool            normalise by total power.
    power_law_remove : bool     remove 1/f trend before computation.
    fs : float, optional        override sampling frequency.
    cache : bool, default=True  reuse ``bt._harm_cache`` when present.
    progress : bool             print one line per window.

    Returns
    -------
    H_spec : ndarray, shape (T, F)
    freqs  : ndarray, shape (F,)
    """
    H, _M, freqs = _encode_harmonicity_core(
        bt_list,
        return_matrix=False,
        fmin=fmin, fmax=fmax, precision_hz=precision_hz,
        metric=metric, n_harms=n_harms,
        smoothness=smoothness, smoothness_harm=smoothness_harm,
        normalize=normalize, power_law_remove=power_law_remove,
        fs=fs, cache=cache, progress=progress,
    )
    return H, freqs


def encode_harmonicity_matrices(
    bt_list: List[Any],
    *,
    fmin: float = 1.0,
    fmax: float = 30.0,
    precision_hz: float = 0.5,
    metric: str = "harmsim",
    n_harms: int = 10,
    smoothness: int = 1,
    normalize: bool = True,
    power_law_remove: bool = False,
    fs: Optional[float] = None,
    cache: bool = True,
    progress: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the 2-D harmonicity matrix per window (fast path).

    Like :func:`encode_harmonicity_spectrum` but returns the F×F pairwise
    matrix per window — strictly more information.  Shares the same
    cache and the same vectorised fast path.

    Returns
    -------
    M : ndarray, shape (T, F, F)
    freqs : ndarray, shape (F,)
    """
    _H, M, freqs = _encode_harmonicity_core(
        bt_list,
        return_matrix=True,
        fmin=fmin, fmax=fmax, precision_hz=precision_hz,
        metric=metric, n_harms=n_harms,
        smoothness=smoothness, smoothness_harm=1,
        normalize=normalize, power_law_remove=power_law_remove,
        fs=fs, cache=cache, progress=progress,
    )
    return M, freqs


def encode_ji_matrix(
    ratios_list: List[List[float]],
    tolerance_cents: float = 30.0,
) -> Tuple[np.ndarray, List[str]]:
    """Binary-encode ratio sets as JI interval presence/absence vectors.

    Each row is a binary vector over the ``interval_names`` dictionary:
    element *j* is 1 if any ratio in that timepoint's set falls within
    ``tolerance_cents`` of interval *j*.

    Parameters
    ----------
    ratios_list : list of list of float, length T
    tolerance_cents : float, default=30
        Match radius in cents.

    Returns
    -------
    X : ndarray, shape (T, n_intervals)  int8
    interval_labels : list of str  (column names)
    """
    labels = [name for name, _ in _JI_CATALOG]
    n = len(labels)
    X = np.zeros((len(ratios_list), n), dtype=np.int8)
    catalog = _JI_CATALOG_CENTS  # (J,) — cached
    for t, ratios in enumerate(ratios_list):
        if not ratios:
            continue
        arr = np.asarray([r for r in ratios if r > 0], dtype=float)
        if arr.size == 0:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            cents = 1200.0 * np.log2(arr)
        cents = cents[np.isfinite(cents)]
        if cents.size == 0:
            continue
        # Vectorised matching:  |catalog[None, :] - cents[:, None]| < tol
        diff = np.abs(catalog[None, :] - cents[:, None])    # (R, J)
        X[t] = (diff < tolerance_cents).any(axis=0).astype(np.int8)
    return X, labels


def _chord_label(ratios: List[float], tolerance_cents: float = 30.0) -> frozenset:
    """Map a ratio set to a frozenset of JI interval names (internal helper)."""
    labels: set = set()
    for r in ratios:
        if r <= 0:
            continue
        try:
            c = 1200.0 * np.log2(r)
        except (ValueError, ZeroDivisionError):
            continue
        if not np.isfinite(c):
            continue
        dists = np.abs(_JI_CATALOG_CENTS - c)
        best = int(np.argmin(dists))
        if dists[best] < tolerance_cents:
            labels.add(_JI_CATALOG[best][0])
        else:
            labels.add(f"{int(round(c / 10.0)) * 10}\u00a2")
    return frozenset(labels)


def _impute_nan_cols(X: np.ndarray) -> np.ndarray:
    """Replace NaN values with column means (or 0 if an entire column is NaN)."""
    X = X.copy().astype(float)
    for j in range(X.shape[1]):
        col = X[:, j]
        mask = np.isnan(col)
        if mask.all():
            X[:, j] = 0.0
        elif mask.any():
            X[mask, j] = float(np.nanmean(col))
    return X


def _uniform_hist(n_bins: int) -> np.ndarray:
    """Return a uniform histogram (fallback for empty distributions in W1)."""
    h = np.ones(n_bins, dtype=float)
    return h / h.sum()


# ─────────────────────────────────────────────────────────────────────────────
# I-bis.  DECODING / RENDERING LAYER  (histogram → ratios → .scl / MIDI)
# ─────────────────────────────────────────────────────────────────────────────

def histogram_to_ratios(
    h: np.ndarray,
    n_peaks: Optional[int] = None,
    *,
    prominence: float = 0.0,
    min_cents: float = 0.0,
    max_cents: float = 1200.0,
    min_distance_cents: float = 25.0,
    include_unison: bool = True,
    include_octave: bool = True,
) -> List[float]:
    """Decode a cents histogram back to a list of frequency ratios.

    Inverts :func:`encode_histograms` at the per-frame level by locating peaks
    in ``h`` and mapping their bin centres back to ratios via
    ``r = 2 ** (cents / 1200)``.

    The same function works for **observed** histograms (one row of
    ``analyzer.histograms``) and for **synthesised** histograms produced by
    Wasserstein barycenters, latent-space decode, DMD reconstruction, or
    Markov-state centroids.

    Parameters
    ----------
    h : ndarray, shape (n_bins,)
        Normalised cents histogram.  Sum should be 1; all-zero rows yield
        an empty (or unison/octave-only) ratio list.
    n_peaks : int or None, default=None
        If given, return at most this many peaks (top-amplitude).  When
        ``None`` all local maxima above ``prominence`` are returned.
    prominence : float, default=0.0
        Minimum peak prominence (passed to ``scipy.signal.find_peaks``).
        Useful to filter out noise in dense histograms (e.g. decoded from
        PCA latent space).
    min_cents, max_cents : float
        Histogram extent in cents (must match the encoder).
    min_distance_cents : float, default=25.0
        Minimum cents separation between detected peaks.  Prevents adjacent
        bins from both registering as peaks.
    include_unison : bool, default=True
        Prepend ``1.0`` to the output (Scala convention; required for .scl).
    include_octave : bool, default=True
        Append ``2.0`` to the output (period of repetition).

    Returns
    -------
    ratios : list of float
        Sorted ascending, finite, positive.  ``[1.0]`` when ``h`` is all-zero
        and ``include_unison=True``.

    Examples
    --------
    >>> # Round-trip: encode then decode a known tuning
    >>> from biotuner.harmonic_sequence import (
    ...     encode_histograms, histogram_to_ratios,
    ... )
    >>> h = encode_histograms([[1.5, 1.25, 1.333]])[0]
    >>> histogram_to_ratios(h, include_unison=False, include_octave=False)
    [1.2497..., 1.3325..., 1.4983...]
    """
    from scipy.signal import find_peaks

    h = np.asarray(h, dtype=float).ravel()
    n_bins = len(h)
    bin_width_cents = (max_cents - min_cents) / n_bins
    bin_centers_cents = min_cents + (np.arange(n_bins) + 0.5) * bin_width_cents

    out: List[float] = []
    if h.sum() > 0:
        # Pad with zeros so peaks at index 0 or n-1 are detected.
        padded = np.concatenate([[0.0], h, [0.0]])
        distance = max(1, int(round(min_distance_cents / bin_width_cents)))
        peak_idx, _ = find_peaks(padded, prominence=prominence, distance=distance)
        peak_idx = peak_idx - 1  # undo padding offset

        if n_peaks is not None and len(peak_idx) > n_peaks:
            top = np.argsort(h[peak_idx])[-n_peaks:]
            peak_idx = np.sort(peak_idx[top])

        for idx in peak_idx:
            r = float(2.0 ** (bin_centers_cents[idx] / 1200.0))
            if np.isfinite(r) and r > 0:
                out.append(r)

    out.sort()
    if include_unison and (not out or out[0] > 1.0 + 1e-9):
        out.insert(0, 1.0)
    if include_octave and (not out or out[-1] < 2.0 - 1e-9):
        out.append(2.0)
    return out


def histogram_to_scl(
    h: np.ndarray,
    name: str = "biotuner_histogram",
    *,
    n_peaks: Optional[int] = None,
    prominence: float = 0.0,
    write: bool = False,
    **kwargs: Any,
) -> str:
    """Render a cents histogram as a Scala (.scl) tuning-file string.

    Wraps :func:`histogram_to_ratios` and :func:`biotuner.biotuner_utils.create_SCL`.
    Pass ``write=True`` to also save ``<name>.scl`` to disk.

    Parameters
    ----------
    h : ndarray, shape (n_bins,)
    name : str
        Used both as the in-file scale title and (when ``write=True``) the
        filename stem.
    n_peaks, prominence : passed to :func:`histogram_to_ratios`.
    write : bool, default=False
        When ``True``, write ``<name>.scl`` next to the working directory.
    **kwargs
        Forwarded to :func:`histogram_to_ratios` (e.g. ``min_distance_cents``).

    Returns
    -------
    scl_str : str
        Scala-formatted text.
    """
    from biotuner.biotuner_utils import create_SCL

    ratios = histogram_to_ratios(
        h,
        n_peaks=n_peaks,
        prominence=prominence,
        include_unison=True,
        include_octave=True,
        **kwargs,
    )
    return create_SCL(ratios, name, write=write)


def histograms_to_midi(
    H: np.ndarray,
    filename: str = "harmonic_sequence",
    *,
    base_freq: float = 220.0,
    duration_beats: float = 1.0,
    velocity: int = 80,
    n_peaks: Optional[int] = 5,
    prominence: float = 0.0,
    microtonal: bool = True,
    subdivision: int = 1,
    skip_empty: bool = True,
    **kwargs: Any,
) -> Any:
    """Render a sequence of histograms as a microtonal MIDI file.

    Each row of ``H`` becomes one chord: peaks of the histogram are decoded to
    ratios, multiplied by ``base_freq`` to get absolute frequencies, then sent
    to :func:`biotuner.biotuner_utils.create_midi` (one channel per voice with
    pitch bends for microtonal precision).

    The function works equally well on:
        - ``analyzer.histograms`` — the recorded biosignal trajectory
        - ``analyzer.wasserstein.interpolate_pair(t1, t2, n)`` — a glissando
        - ``analyzer.latent.decode(Z_path)`` — a synthesised path through
          harmonic space
        - ``analyzer.markov._km.cluster_centers_`` — prototype tunings

    Parameters
    ----------
    H : ndarray, shape (T, n_bins) or (n_bins,)
        Sequence of histograms.  A single 1-D histogram is treated as one
        chord.  Rows do not need to be normalised.
    filename : str, default='harmonic_sequence'
        Output filename stem; ``.mid`` is appended by ``create_midi``.
    base_freq : float, default=220.0
        Frequency that the unison ratio (1.0) maps to (Hz).  220 Hz = A3.
    duration_beats : float, default=1.0
        Note duration per chord.  Pass a list/array of length T for variable
        durations (e.g. modulated by Wasserstein flux).
    velocity : int, default=80
        MIDI velocity (1-127) for every note.
    n_peaks : int or None, default=5
        Maximum peaks decoded per histogram.  ``None`` keeps all peaks.
    prominence : float, default=0.0
        Minimum peak prominence (helps for smooth/decoded histograms).
    microtonal : bool, default=True
        Emit pitch-bend messages so non-12-TET ratios play in tune.
    subdivision : int, default=1
        Beat subdivisions (passed to ``create_midi``).
    skip_empty : bool, default=True
        Drop windows whose histogram has no detected peaks.  When ``False``
        an empty chord is inserted (silent rest of ``duration_beats``).
    **kwargs
        Forwarded to :func:`histogram_to_ratios`.

    Returns
    -------
    mid : mido.MidiFile
        The saved MIDI file object.

    Notes
    -----
    With ``microtonal=True``, each voice occupies its own MIDI channel,
    capping the chord size at 16 notes.  ``n_peaks=5`` is a safe default.
    """
    from biotuner.biotuner_utils import create_midi

    H = np.asarray(H, dtype=float)
    if H.ndim == 1:
        H = H[np.newaxis, :]
    T = H.shape[0]

    # Allow a scalar or an array of durations.
    if np.isscalar(duration_beats):
        durations_arr = [float(duration_beats)] * T
    else:
        durations_arr = list(np.asarray(duration_beats, dtype=float).ravel())
        if len(durations_arr) != T:
            raise ValueError(
                f"duration_beats has length {len(durations_arr)} but H has T={T}."
            )

    chords: List[List[float]] = []
    durations: List[float] = []
    for h_row, dur in zip(H, durations_arr):
        ratios = histogram_to_ratios(
            h_row,
            n_peaks=n_peaks,
            prominence=prominence,
            include_unison=False,
            include_octave=False,
            **kwargs,
        )
        if not ratios:
            if skip_empty:
                continue
            chords.append([])
            durations.append(dur)
            continue
        chords.append([base_freq * r for r in ratios])
        durations.append(dur)

    if not chords:
        raise ValueError(
            "No non-empty histograms to render. "
            "Either pass denser histograms or set skip_empty=False."
        )

    velocities = [[int(velocity)] * len(c) for c in chords]
    return create_midi(
        chords,
        durations,
        velocities=velocities,
        subdivision=subdivision,
        microtonal=microtonal,
        filename=filename,
    )


# ─────────────────────────────────────────────────────────────────────────────
# II.  APPROACH 1 – MARKOV CHAIN / HMM
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_n_states(
    X: np.ndarray,
    k_range: Tuple[int, int] = (2, 10),
    random_state: int = 42,
) -> Tuple[int, Dict[int, float]]:
    """Select the number of Markov states via silhouette score.

    For each K in ``k_range``, K-means is fitted and the mean silhouette
    coefficient is computed.  The K with the highest score is returned.

    Parameters
    ----------
    X : ndarray, shape (T, n_features)
        Histogram (or other feature) matrix.
    k_range : (int, int), default=(2, 10)
        Inclusive range of candidate K values.
    random_state : int, default=42

    Returns
    -------
    best_k : int
    scores : dict  {k: silhouette_score}

    Notes
    -----
    Requires scikit-learn.  Silhouette ranges from -1 (poor) to +1 (ideal);
    values above ~0.5 indicate well-separated clusters.
    """
    if not _SKLEARN:
        raise ImportError(
            "scikit-learn is required for find_optimal_n_states. "
            "Install with: pip install scikit-learn"
        )
    from sklearn.metrics import silhouette_score  # late import (optional dep)

    T = X.shape[0]
    k_min = max(k_range[0], 2)
    k_max = min(k_range[1], T - 1)
    if k_min > k_max:
        return k_min, {}

    scores: Dict[int, float] = {}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X)
        unique_labels = np.unique(km.labels_)
        if len(unique_labels) < 2:
            continue  # silhouette undefined for a single cluster
        scores[k] = float(silhouette_score(X, km.labels_))

    if not scores:
        return k_min, scores
    best_k = max(scores, key=scores.__getitem__)
    return best_k, scores


class HarmonicMarkov:
    """Discrete-state Markov chain (and optional HMM) over harmonic states.

    Each timepoint is assigned to one of ``n_states`` clusters defined by
    KMeans on the cents-histogram representation.  A Markov chain of
    arbitrary ``order`` is estimated from the state sequence.

    Parameters
    ----------
    n_states : int or ``'auto'``, default=5
        Number of discrete harmonic states.  When ``'auto'``, the optimal K
        is selected automatically via silhouette score over the range given
        by ``auto_k_range``.
    order : int, default=1
        Markov chain memory.  ``order=1`` is a standard first-order chain
        (next state depends only on the current state).  ``order=2`` conditions
        on the last two states, etc.  Higher orders can capture longer harmonic
        patterns but require more data (at least ``n_states**order`` transitions
        are needed for reliable estimates).
    auto_k_range : (int, int), default=(2, 10)
        Range of K to search when ``n_states='auto'``.
    use_hmm : bool, default=False
        Fit a Gaussian HMM (requires hmmlearn).
    random_state : int, default=42
    """

    def __init__(
        self,
        n_states: int = 5,
        order: int = 1,
        auto_k_range: Tuple[int, int] = (2, 10),
        use_hmm: bool = False,
        random_state: int = 42,
    ):
        if not _SKLEARN:
            raise ImportError(
                "scikit-learn is required for HarmonicMarkov. "
                "Install with: pip install scikit-learn"
            )
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        self.n_states = n_states          # may be 'auto'
        self.order = order
        self.auto_k_range = auto_k_range
        self.use_hmm = use_hmm
        self.random_state = random_state
        self._km: Optional[KMeans] = None
        self._hmm: Any = None
        self._labels: Optional[np.ndarray] = None
        self._trans: Optional[np.ndarray] = None          # K×K first-order matrix
        self._high_order_trans: Optional[Dict[tuple, np.ndarray]] = None
        self._silhouette_scores: Optional[Dict[int, float]] = None

    def fit(self, X: np.ndarray) -> "HarmonicMarkov":
        """Fit the Markov chain on a histogram matrix.

        Parameters
        ----------
        X : ndarray, shape (T, n_bins)

        Returns
        -------
        self
        """
        # ── optimal K selection ────────────────────────────────────────────────
        if self.n_states == "auto":
            best_k, scores = find_optimal_n_states(
                X, k_range=self.auto_k_range,
                random_state=self.random_state,
            )
            self.n_states = best_k
            self._silhouette_scores = scores

        # ── cluster into K harmonic states ─────────────────────────────────────
        self._km = KMeans(
            n_clusters=self.n_states,
            random_state=self.random_state,
            n_init=10,
        ).fit(X)
        self._labels = self._km.labels_

        # ── first-order transition matrix (always computed) ────────────────────
        K = self.n_states
        counts = np.zeros((K, K))
        for a, b in zip(self._labels[:-1], self._labels[1:]):
            counts[a, b] += 1
        row_sums = counts.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self._trans = counts / row_sums

        # ── higher-order transition table ──────────────────────────────────────
        if self.order > 1:
            hist_counts: Dict[tuple, np.ndarray] = {}
            for t in range(len(self._labels) - self.order):
                history = tuple(int(self._labels[t + i]) for i in range(self.order))
                next_s  = int(self._labels[t + self.order])
                if history not in hist_counts:
                    hist_counts[history] = np.zeros(K)
                hist_counts[history][next_s] += 1
            # Normalise each history distribution
            self._high_order_trans = {}
            for hist, cnt in hist_counts.items():
                s = cnt.sum()
                self._high_order_trans[hist] = cnt / s if s > 0 else cnt
        else:
            self._high_order_trans = None

        # ── optional Gaussian HMM ──────────────────────────────────────────────
        if self.use_hmm:
            if not _HMMLEARN:
                warnings.warn(
                    "hmmlearn not available; skipping HMM fit. "
                    "Install with: pip install hmmlearn",
                    ImportWarning,
                    stacklevel=2,
                )
            else:
                self._hmm = _hmmlearn_hmm.GaussianHMM(
                    n_components=self.n_states,
                    covariance_type="diag",
                    n_iter=100,
                    random_state=self.random_state,
                )
                self._hmm.fit(X)
        return self

    # ── properties ─────────────────────────────────────────────────────────────

    @property
    def transition_matrix_(self) -> np.ndarray:
        """Row-normalised (n_states × n_states) first-order transition matrix."""
        self._check_fitted()
        return self._trans

    @property
    def high_order_transition_(self) -> Optional[Dict[tuple, np.ndarray]]:
        """Higher-order transition table.

        ``None`` when ``order=1``.  Otherwise a dict mapping each observed
        history tuple of length ``order`` to a probability distribution
        (ndarray of shape (n_states,)) over the next state.
        """
        self._check_fitted()
        return self._high_order_trans

    @property
    def state_labels_(self) -> np.ndarray:
        """Integer state label for each fitted timepoint."""
        self._check_fitted()
        return self._labels

    @property
    def silhouette_scores_(self) -> Optional[Dict[int, float]]:
        """Silhouette scores per K when ``n_states='auto'``, else ``None``."""
        return self._silhouette_scores

    @property
    def steady_state_(self) -> np.ndarray:
        """Stationary distribution π such that πT = π (first-order)."""
        self._check_fitted()
        A = self._trans.T - np.eye(self.n_states)
        A = np.vstack([A, np.ones(self.n_states)])
        b = np.zeros(self.n_states + 1)
        b[-1] = 1.0
        pi, *_ = np.linalg.lstsq(A, b, rcond=None)
        pi = np.abs(pi)
        return pi / pi.sum()

    @property
    def transition_entropy_(self) -> float:
        """Mean Shannon entropy (bits) of the transition distributions.

        For ``order=1`` this is the mean row entropy of the K×K matrix.
        For higher orders it is the mean entropy over all observed history
        tuples, weighted equally.
        """
        self._check_fitted()
        if self.order == 1 or self._high_order_trans is None:
            ent = 0.0
            for row in self._trans:
                mask = row > 0
                if mask.any():
                    ent += -float(np.sum(row[mask] * np.log2(row[mask])))
            return ent / self.n_states
        else:
            entropies = []
            for dist in self._high_order_trans.values():
                mask = dist > 0
                if mask.any():
                    entropies.append(-float(np.sum(dist[mask] * np.log2(dist[mask]))))
                else:
                    entropies.append(0.0)
            return float(np.mean(entropies)) if entropies else 0.0

    # ── prediction ─────────────────────────────────────────────────────────────

    def predict_next_proba(self, history) -> np.ndarray:
        """Transition probability distribution from the given history.

        Parameters
        ----------
        history : int or tuple of int
            For ``order=1``, a single state index.
            For ``order>1``, a tuple of the last ``order`` state indices
            (oldest first).

        Returns
        -------
        proba : ndarray, shape (n_states,)
            Falls back to the first-order row when the history tuple was not
            observed during training.
        """
        self._check_fitted()
        if self.order == 1:
            return self._trans[int(history)]
        history = tuple(int(s) for s in history)
        if self._high_order_trans and history in self._high_order_trans:
            return self._high_order_trans[history]
        # Unseen history: fall back to first-order from last state
        return self._trans[history[-1]]

    def decode_viterbi(self, X: np.ndarray) -> np.ndarray:
        """Viterbi decoding via HMM; falls back to KMeans if HMM unavailable."""
        self._check_fitted()
        if self._hmm is not None:
            return self._hmm.predict(X)
        return self._km.predict(X)

    def _check_fitted(self) -> None:
        if self._trans is None:
            raise RuntimeError("Call fit() before accessing model attributes.")


# ─────────────────────────────────────────────────────────────────────────────
# III.  APPROACH 2 – WASSERSTEIN TRAJECTORY
# ─────────────────────────────────────────────────────────────────────────────

class WassersteinTrajectory:
    """Optimal-transport trajectory in interval space.

    Treats each timepoint's cents histogram as a probability distribution
    over [0, 1200] and computes the 1-D Wasserstein (Earth Mover's) distance
    between frames.  The pairwise distance matrix supports manifold
    visualisation; consecutive distances measure harmonic flux.

    Parameters
    ----------
    n_bins : int, default=240
        Number of histogram bins (must match the encoder).
    """

    def __init__(self, n_bins: int = 240):
        self.n_bins = n_bins
        self._X: Optional[np.ndarray] = None
        self._D: Optional[np.ndarray] = None
        self._flux: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "WassersteinTrajectory":
        """Compute pairwise Wasserstein distances and per-frame flux.

        Parameters
        ----------
        X : ndarray, shape (T, n_bins)
            Normalised cents histograms (rows should sum to 1).

        Returns
        -------
        self
        """
        self._X = X
        T = X.shape[0]
        bin_centers = np.arange(self.n_bins, dtype=float)

        # Replace all-zero histograms with uniform to avoid W1 = 0 by default
        X_safe = np.array([
            row if row.sum() > 0 else _uniform_hist(self.n_bins)
            for row in X
        ])

        D = np.zeros((T, T))
        for i in range(T):
            for j in range(i + 1, T):
                d = _sp_wasserstein(
                    bin_centers, bin_centers, X_safe[i], X_safe[j]
                )
                D[i, j] = D[j, i] = d
        self._D = D
        self._flux = np.array([D[t, t + 1] for t in range(T - 1)])
        return self

    @property
    def distance_matrix_(self) -> np.ndarray:
        """Pairwise Wasserstein distance matrix (T × T)."""
        self._check_fitted()
        return self._D

    @property
    def flux_(self) -> np.ndarray:
        """Per-frame harmonic velocity (consecutive W1 distances), length T-1."""
        self._check_fitted()
        return self._flux

    def barycenter(
        self,
        h1: np.ndarray,
        h2: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """1-D Wasserstein barycenter between ``h1`` and ``h2`` at weight ``alpha``.

        Interpolates in quantile space: Q_α = (1-α)·Q₁ + α·Q₂, then maps
        back to a histogram via sampling.

        Parameters
        ----------
        h1, h2 : ndarray, shape (n_bins,)  normalised histograms
        alpha : float in [0, 1]

        Returns
        -------
        bary : ndarray, shape (n_bins,)  normalised
        """
        n = len(h1)
        bin_edges = np.arange(n + 1, dtype=float)
        h1 = h1 if h1.sum() > 0 else _uniform_hist(n)
        h2 = h2 if h2.sum() > 0 else _uniform_hist(n)
        F1 = np.r_[0.0, np.cumsum(h1)]
        F2 = np.r_[0.0, np.cumsum(h2)]
        t_vals = np.linspace(0.0, 1.0, 1000)
        Q1 = np.interp(t_vals, F1, bin_edges)
        Q2 = np.interp(t_vals, F2, bin_edges)
        Q_alpha = (1.0 - alpha) * Q1 + alpha * Q2
        hist, _ = np.histogram(Q_alpha, bins=bin_edges)
        hist = hist.astype(float)
        s = hist.sum()
        return hist / s if s > 0 else hist

    def interpolate_pair(
        self,
        t1: int,
        t2: int,
        n_steps: int = 10,
    ) -> List[np.ndarray]:
        """Return ``n_steps`` histograms interpolating between timepoints t1 and t2.

        Parameters
        ----------
        t1, t2 : int  indices into the fitted sequence
        n_steps : int

        Returns
        -------
        list of ndarray, each shape (n_bins,)
        """
        self._check_fitted()
        h1, h2 = self._X[t1], self._X[t2]
        return [self.barycenter(h1, h2, a) for a in np.linspace(0, 1, n_steps)]

    def embed(self, n_components: int = 2, method: str = "mds") -> np.ndarray:
        """Low-dimensional embedding of the Wasserstein distance matrix.

        Parameters
        ----------
        n_components : int, default=2
        method : str, default='mds'
            ``'mds'`` (requires sklearn) or ``'umap'`` (requires umap-learn).

        Returns
        -------
        Z : ndarray, shape (T, n_components)
        """
        self._check_fitted()
        if not _SKLEARN:
            raise ImportError(
                "scikit-learn required for embed(). "
                "Install with: pip install scikit-learn"
            )
        if method == "mds":
            try:
                mds = MDS(
                    n_components=n_components,
                    dissimilarity="precomputed",
                    random_state=42,
                    normalized_stress=False,
                )
            except TypeError:
                # normalized_stress added in sklearn 1.1
                mds = MDS(
                    n_components=n_components,
                    dissimilarity="precomputed",
                    random_state=42,
                )
            return mds.fit_transform(self._D)
        elif method == "umap":
            try:
                import umap  # type: ignore[import]
            except ImportError:
                raise ImportError(
                    "umap-learn required for method='umap'. "
                    "Install with: pip install umap-learn"
                )
            return umap.UMAP(
                n_components=n_components,
                metric="precomputed",
                random_state=42,
            ).fit_transform(self._D)
        else:
            raise ValueError(f"Unknown embed method '{method}'. Use 'mds' or 'umap'.")

    def _check_fitted(self) -> None:
        if self._D is None:
            raise RuntimeError("Call fit() before accessing model attributes.")


# ─────────────────────────────────────────────────────────────────────────────
# IV.  APPROACH 3 – DYNAMIC MODE DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────

class HarmonicDMD:
    """Linear dynamical modes of harmonic metric vectors.

    Applies the exact DMD algorithm to a (T × D) feature matrix.
    Eigenvalues inside the unit circle correspond to decaying transients;
    on the unit circle to oscillatory harmonic cycles; outside to growing
    instabilities.

    Parameters
    ----------
    rank : int or None, default=None
        SVD truncation rank.  ``None`` keeps all singular values above
        a numerical threshold (relative to the largest singular value).
    center : bool, default=True
        Subtract the temporal mean before fitting.
    """

    def __init__(self, rank: Optional[int] = None, center: bool = True):
        self.rank = rank
        self.center = center
        self._eigenvalues: Optional[np.ndarray] = None
        self._modes: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None
        self._X_fit: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "HarmonicDMD":
        """Fit the DMD model.

        Parameters
        ----------
        X : ndarray, shape (T, D)
            Feature matrix.  NaN values are imputed with column means.

        Returns
        -------
        self
        """
        X = _impute_nan_cols(X)
        if self.center:
            self._mean = X.mean(axis=0)
            X = X - self._mean
        else:
            self._mean = np.zeros(X.shape[1])
        self._X_fit = X

        X1 = X[:-1].T   # D × (T-1)
        X2 = X[1:].T    # D × (T-1)

        U, S, Vt = np.linalg.svd(X1, full_matrices=False)
        threshold = 1e-10 * S[0] if len(S) > 0 else 1e-10
        r_auto = int((S > threshold).sum())
        r = max(1, min(self.rank if self.rank is not None else r_auto, len(S)))

        U_r, S_r, Vt_r = U[:, :r], S[:r], Vt[:r, :]
        S_inv = np.diag(1.0 / S_r)

        # Reduced operator A~
        A_tilde = U_r.T @ X2 @ Vt_r.T @ S_inv
        eigenvalues, W = np.linalg.eig(A_tilde)

        # Full-space DMD modes Φ = X2 Vr Sr^{-1} W
        self._modes = X2 @ Vt_r.T @ S_inv @ W   # D × r  complex
        self._eigenvalues = eigenvalues           # r,  complex
        return self

    @property
    def eigenvalues_(self) -> np.ndarray:
        """DMD eigenvalues (complex array, length r)."""
        self._check_fitted()
        return self._eigenvalues

    @property
    def modes_(self) -> np.ndarray:
        """DMD spatial modes (D × r, complex)."""
        self._check_fitted()
        return self._modes

    @property
    def growth_rates_(self) -> np.ndarray:
        """Real part of log(λ): positive → growing, negative → decaying."""
        self._check_fitted()
        return np.real(np.log(self._eigenvalues + 1e-15))

    @property
    def frequencies_(self) -> np.ndarray:
        """Imag part of log(λ), proportional to oscillation frequency."""
        self._check_fitted()
        return np.imag(np.log(self._eigenvalues + 1e-15))

    def oscillatory_modes(
        self, threshold: float = 0.05
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return eigenvalues and indices of modes close to the unit circle.

        Parameters
        ----------
        threshold : float, default=0.05
            Maximum allowed distance from |λ| = 1.

        Returns
        -------
        eigenvalues : complex ndarray
        idx : int ndarray
        """
        self._check_fitted()
        idx = np.where(np.abs(np.abs(self._eigenvalues) - 1.0) < threshold)[0]
        return self._eigenvalues[idx], idx

    def reconstruct(self, n_steps: int) -> np.ndarray:
        """Forward-propagate from the last fitted state for ``n_steps``.

        Uses the DMD mode expansion:
        x(k) = Φ · diag(λ)^k · b₀   where b₀ = Φ⁺ x₀.

        Returns
        -------
        X_pred : ndarray, shape (n_steps, D)  (with mean added back)
        """
        self._check_fitted()
        Phi = self._modes          # D × r
        PhiInv = np.linalg.pinv(Phi)  # r × D
        x0 = self._X_fit[-1]      # D,
        b0 = PhiInv @ x0          # r, initial amplitudes

        result = []
        for k in range(n_steps):
            b_k = b0 * (self._eigenvalues ** k)
            result.append(np.real(Phi @ b_k))
        return np.stack(result, axis=0) + self._mean

    def _check_fitted(self) -> None:
        if self._eigenvalues is None:
            raise RuntimeError("Call fit() before accessing model attributes.")


# ─────────────────────────────────────────────────────────────────────────────
# V.  APPROACH 4 – LATENT HARMONIC SPACE
# ─────────────────────────────────────────────────────────────────────────────

class HarmonicLatentSpace:
    """Smooth low-dimensional manifold of harmonic states via PCA.

    Fits a linear PCA to the cents-histogram matrix, providing:

    - ``encode(X)``  – project histograms into latent coordinates
    - ``decode(Z)``  – reconstruct histograms from latent coordinates
    - ``interpolate(z1, z2)``  – smooth path between two harmonic states

    Parameters
    ----------
    latent_dim : int, default=3
    random_state : int, default=42
    """

    def __init__(self, latent_dim: int = 3, random_state: int = 42):
        if not _SKLEARN:
            raise ImportError(
                "scikit-learn is required for HarmonicLatentSpace. "
                "Install with: pip install scikit-learn"
            )
        self.latent_dim = latent_dim
        self.random_state = random_state
        self._pca: Optional[PCA] = None
        self._scaler: Optional[StandardScaler] = None
        self._Z_fit: Optional[np.ndarray] = None
        self._X_fit: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "HarmonicLatentSpace":
        """Fit PCA to the histogram matrix.

        Parameters
        ----------
        X : ndarray, shape (T, n_bins)

        Returns
        -------
        self
        """
        self._X_fit = X
        self._scaler = StandardScaler(with_std=False)
        X_c = self._scaler.fit_transform(X)
        n_comp = min(self.latent_dim, X.shape[0] - 1, X.shape[1])
        self._pca = PCA(n_components=n_comp, random_state=self.random_state).fit(X_c)
        self._Z_fit = self._pca.transform(X_c)
        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Project histograms into latent space.

        Parameters
        ----------
        X : ndarray, shape (T, n_bins)

        Returns
        -------
        Z : ndarray, shape (T, latent_dim)
        """
        self._check_fitted()
        return self._pca.transform(self._scaler.transform(X))

    def decode(self, Z: np.ndarray) -> np.ndarray:
        """Reconstruct histograms from latent coordinates.

        Parameters
        ----------
        Z : ndarray, shape (T, latent_dim)

        Returns
        -------
        X_rec : ndarray, shape (T, n_bins)  (clipped to non-negative)
        """
        self._check_fitted()
        X_c = self._pca.inverse_transform(Z)
        return np.maximum(self._scaler.inverse_transform(X_c), 0.0)

    def trajectory(self) -> np.ndarray:
        """Latent coordinates of all fitted timepoints, shape (T, latent_dim)."""
        self._check_fitted()
        return self._Z_fit

    def interpolate(
        self,
        z1: np.ndarray,
        z2: np.ndarray,
        n_steps: int = 10,
    ) -> np.ndarray:
        """Linear interpolation between two latent points.

        Parameters
        ----------
        z1, z2 : ndarray, shape (latent_dim,)
        n_steps : int

        Returns
        -------
        Z_path : ndarray, shape (n_steps, latent_dim)
        """
        alphas = np.linspace(0.0, 1.0, n_steps)
        return np.stack([(1 - a) * z1 + a * z2 for a in alphas], axis=0)

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        """Fraction of variance explained by each latent dimension."""
        self._check_fitted()
        return self._pca.explained_variance_ratio_

    @property
    def reconstruction_error_(self) -> float:
        """Mean squared reconstruction error on the fitted data."""
        self._check_fitted()
        X_rec = self.decode(self._Z_fit)
        return float(np.mean((self._X_fit - X_rec) ** 2))

    def _check_fitted(self) -> None:
        if self._pca is None:
            raise RuntimeError("Call fit() before accessing model attributes.")


# ─────────────────────────────────────────────────────────────────────────────
# VI.  APPROACH 6 – TOPOLOGICAL DATA ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

class HarmonicTopology:
    """Topological shape of harmonic scalar trajectories.

    Applies Takens delay embedding to convert a scalar harmonicity time
    series (e.g., ``harmsim``, ``subharm_tension``) into a point cloud,
    then computes persistent homology:

    - **H0** (connected components): harmonic attractor basins
    - **H1** (loops): cyclic / periodic harmonic progressions

    H0 is always available via scipy single-linkage clustering.
    H1 requires ``ripser`` (``pip install ripser``).

    Parameters
    ----------
    embedding_dim : int, default=3
        Takens embedding dimension d.
    delay : int, default=1
        Takens delay τ (in time-steps).
    """

    def __init__(self, embedding_dim: int = 3, delay: int = 1):
        self.embedding_dim = embedding_dim
        self.delay = delay
        self._point_cloud: Optional[np.ndarray] = None
        self._diagrams: Optional[List[np.ndarray]] = None
        self._linkage_matrix: Optional[np.ndarray] = None

    def fit(self, scalar_series: np.ndarray) -> "HarmonicTopology":
        """Embed the scalar series and compute persistent homology.

        Parameters
        ----------
        scalar_series : array-like, shape (T,)
            A 1-D harmonicity time series.  NaN values are linearly
            interpolated.

        Returns
        -------
        self
        """
        series = np.asarray(scalar_series, dtype=float)
        # Linear interpolation of NaNs
        nans = np.isnan(series)
        if nans.any():
            idx = np.arange(len(series))
            series = np.interp(idx, idx[~nans], series[~nans])

        # Takens delay embedding
        d, tau = self.embedding_dim, self.delay
        N = len(series) - (d - 1) * tau
        if N < d:
            raise ValueError(
                f"Series (length {len(series)}) too short for "
                f"embedding_dim={d}, delay={tau}.  "
                f"Need at least {d + (d - 1) * tau} samples."
            )
        cloud = np.empty((N, d))
        for k in range(d):
            cloud[:, k] = series[k * tau: k * tau + N]
        self._point_cloud = cloud

        if _RIPSER:
            result = _ripser_fn(cloud, maxdim=1)
            self._diagrams = result["dgms"]   # [H0 (birth,death), H1 (birth,death)]
        else:
            warnings.warn(
                "ripser not available; computing H0 only via scipy linkage. "
                "Install with: pip install ripser for full persistence.",
                ImportWarning,
                stacklevel=2,
            )
            Z = _sp_linkage(cloud, method="single")
            self._linkage_matrix = Z
            n_pts = len(cloud)
            # H0: each point born at 0, dies when merged (dendrogram heights)
            # Last component lives forever (death=inf)
            merge_heights = Z[:, 2]
            births = np.zeros(n_pts)
            deaths = np.r_[merge_heights, np.inf]
            self._diagrams = [np.column_stack([births, deaths])]
        return self

    @property
    def takens_embedding_(self) -> np.ndarray:
        """Point cloud from Takens delay embedding, shape (N, embedding_dim)."""
        self._check_fitted()
        return self._point_cloud

    @property
    def persistence_diagram_(self) -> List[np.ndarray]:
        """List of persistence diagrams [H0, H1] (H1 only if ripser installed).

        Each element is an ndarray of (birth, death) pairs.
        """
        self._check_fitted()
        return self._diagrams

    @property
    def betti_numbers_(self) -> np.ndarray:
        """Betti numbers β0, β1 evaluated at the median finite filtration value.

        Returns
        -------
        betti : ndarray of int, shape (2,)
        """
        self._check_fitted()
        b = []
        for i, dgm in enumerate(self._diagrams[:2]):
            finite_deaths = dgm[np.isfinite(dgm[:, 1]), 1]
            if len(finite_deaths) == 0:
                b.append(1 if i == 0 else 0)
                continue
            med = float(np.median(finite_deaths))
            alive = int(np.sum((dgm[:, 0] <= med) & (dgm[:, 1] > med)))
            b.append(alive)
        while len(b) < 2:
            b.append(0)
        return np.array(b, dtype=int)

    def session_fingerprint(self) -> np.ndarray:
        """Fixed-length descriptor summarising the persistence diagrams.

        Returns a 6-element vector:
        [mean_H0_pers, max_H0_pers, n_H0_bars,
         mean_H1_pers, max_H1_pers, n_H1_bars]
        """
        self._check_fitted()
        stats: List[float] = []
        for i in range(2):
            if i < len(self._diagrams):
                dgm = self._diagrams[i]
                pers = dgm[:, 1] - dgm[:, 0]
                pers = pers[np.isfinite(pers)]
            else:
                pers = np.array([])
            if len(pers) == 0:
                stats.extend([0.0, 0.0, 0.0])
            else:
                stats.extend([float(np.mean(pers)), float(np.max(pers)), float(len(pers))])
        return np.array(stats)

    def _check_fitted(self) -> None:
        if self._point_cloud is None:
            raise RuntimeError("Call fit() before accessing model attributes.")


# ─────────────────────────────────────────────────────────────────────────────
# VII.  APPROACH 7 – INTERVAL GRAMMAR
# ─────────────────────────────────────────────────────────────────────────────

class HarmonicGrammar:
    """N-gram symbolic sequence model over JI interval-labelled chords.

    Each timepoint's ratio set is mapped to a ``frozenset`` of JI interval
    names (using the biotuner ``interval_names`` dictionary), creating a
    symbolic chord sequence.  An N-gram model captures the conditional
    probability of each chord token given the preceding N-1 tokens.

    Parameters
    ----------
    tolerance_cents : float, default=30.0
        Match radius for JI interval labelling.
    n_gram : int, default=2
        N-gram order (2 = bigram, 3 = trigram, …).
    """

    def __init__(self, tolerance_cents: float = 30.0, n_gram: int = 2):
        self.tolerance_cents = tolerance_cents
        self.n_gram = n_gram
        self._chord_seq: Optional[List[frozenset]] = None
        self._ngram_counts: Optional[Counter] = None
        self._context_counts: Optional[Counter] = None

    def fit(self, ratios_list: List[List[float]]) -> "HarmonicGrammar":
        """Label each ratio set and build the N-gram language model.

        Parameters
        ----------
        ratios_list : list of list of float, length T

        Returns
        -------
        self
        """
        self._chord_seq = [
            _chord_label(r, self.tolerance_cents) for r in ratios_list
        ]
        n = self.n_gram
        self._ngram_counts = Counter()
        self._context_counts = Counter()
        for i in range(len(self._chord_seq) - n + 1):
            gram = tuple(self._chord_seq[i: i + n])
            self._ngram_counts[gram] += 1
            self._context_counts[gram[:-1]] += 1
        return self

    @property
    def chord_sequence_(self) -> List[frozenset]:
        """Symbolic chord sequence (list of frozensets of interval names)."""
        self._check_fitted()
        return self._chord_seq

    @property
    def vocabulary_(self) -> List[frozenset]:
        """Unique chord tokens in order of first occurrence."""
        self._check_fitted()
        seen: List[frozenset] = []
        for chord in self._chord_seq:
            if chord not in seen:
                seen.append(chord)
        return seen

    def transition_proba(self, context: Tuple) -> Dict[frozenset, float]:
        """Conditional probability P(next_chord | context).

        Parameters
        ----------
        context : tuple of frozenset(s)
            The preceding (n_gram - 1) chords.

        Returns
        -------
        dict  mapping frozenset → probability
        """
        self._check_fitted()
        total = self._context_counts.get(context, 0)
        if total == 0:
            return {}
        return {
            gram[-1]: cnt / total
            for gram, cnt in self._ngram_counts.items()
            if gram[:-1] == context
        }

    @property
    def transition_entropy_(self) -> float:
        """Mean Shannon entropy of (n-1)-gram conditional distributions (bits)."""
        self._check_fitted()
        ents = []
        for context, total in self._context_counts.items():
            probs = [
                cnt / total
                for gram, cnt in self._ngram_counts.items()
                if gram[:-1] == context
            ]
            if len(probs) > 1:
                ents.append(-sum(p * np.log2(p) for p in probs if p > 0))
        return float(np.mean(ents)) if ents else 0.0

    def top_ngrams(self, top_k: int = 10) -> List[Tuple[Tuple, int]]:
        """Return the ``top_k`` most frequent N-grams with their counts."""
        self._check_fitted()
        return self._ngram_counts.most_common(top_k)

    def top_motifs(
        self,
        min_length: int = 2,
        max_length: int = 4,
        top_k: int = 10,
    ) -> List[Tuple[Tuple, int]]:
        """Find the most frequent chord motifs of lengths in [min_length, max_length].

        Returns
        -------
        list of (motif_tuple, count) sorted by frequency descending.
        """
        self._check_fitted()
        seq = self._chord_seq
        counts: Counter = Counter()
        for length in range(min_length, max_length + 1):
            for i in range(len(seq) - length + 1):
                counts[tuple(seq[i: i + length])] += 1
        return counts.most_common(top_k)

    @staticmethod
    def levenshtein(
        seq1: List[frozenset],
        seq2: List[frozenset],
    ) -> int:
        """Levenshtein edit distance between two symbolic chord sequences.

        Substitution cost is 1 for non-identical chords (frozenset equality).

        Parameters
        ----------
        seq1, seq2 : list of frozenset

        Returns
        -------
        int
        """
        m, n = len(seq1), len(seq2)
        dp = np.arange(n + 1)
        for i in range(1, m + 1):
            prev = dp.copy()
            dp[0] = i
            for j in range(1, n + 1):
                cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
                dp[j] = int(min(dp[j] + 1, dp[j - 1] + 1, prev[j - 1] + cost))
        return int(dp[n])

    def _check_fitted(self) -> None:
        if self._chord_seq is None:
            raise RuntimeError("Call fit() before accessing model attributes.")


# ─────────────────────────────────────────────────────────────────────────────
# VIII.  ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

class HarmonicSequenceAnalyzer:
    """Orchestrating class for temporal harmonic-structure learning.

    Extracts representations from a list of ``compute_biotuner`` objects
    (or raw ratio lists) and exposes each modelling approach via a
    consistent ``fit_*`` interface with lazy encoding caches.

    Parameters
    ----------
    n_hist_bins : int, default=240
        Histogram resolution for cents encoding (5 cents/bin over one octave).
    tolerance_cents : float, default=30.0
        JI matching tolerance for grammar and JI matrix encoding.

    Examples
    --------
    >>> analyzer = HarmonicSequenceAnalyzer.from_biotuner_list(
    ...     bt_list, tuning="peaks_ratios"
    ... )
    >>> analyzer.fit_all()
    >>> print(analyzer.summary())
    """

    VALID_REPRESENTATIONS = (
        "cents_histogram",
        "harmonicity_spectrum",
        "harmonicity_matrix",
    )

    def __init__(
        self,
        n_hist_bins: int = 240,
        tolerance_cents: float = 30.0,
        representation: str = "cents_histogram",
        representation_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if representation not in self.VALID_REPRESENTATIONS:
            raise ValueError(
                f"Unknown representation '{representation}'. "
                f"Choose from: {self.VALID_REPRESENTATIONS}"
            )
        self.n_hist_bins = n_hist_bins
        self.tolerance_cents = tolerance_cents
        self.representation = representation
        self.representation_kwargs: Dict[str, Any] = (
            dict(representation_kwargs) if representation_kwargs else {}
        )

        self.ratios_list: List[List[float]] = []
        self._bt_list: Optional[List[Any]] = None

        # Lazy encoding caches
        self._histograms: Optional[np.ndarray] = None      # always cents (for bridge)
        self._scalar_metrics: Optional[np.ndarray] = None
        self._features: Optional[np.ndarray] = None        # active rep, (T, D)
        self._harmonicity_matrices: Optional[np.ndarray] = None   # (T, F, F)
        self._features_freqs: Optional[np.ndarray] = None  # (F,) — non-cents reps

        # Fitted models (one per approach)
        self.markov: Optional[HarmonicMarkov] = None
        self.wasserstein: Optional[WassersteinTrajectory] = None
        self.dmd: Optional[HarmonicDMD] = None
        self.latent: Optional[HarmonicLatentSpace] = None
        self.topology: Optional[HarmonicTopology] = None
        self.grammar: Optional[HarmonicGrammar] = None

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_biotuner_list(
        cls,
        bt_list: List[Any],
        tuning: str = "peaks_ratios",
        n_hist_bins: int = 240,
        tolerance_cents: float = 30.0,
        representation: str = "cents_histogram",
        representation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "HarmonicSequenceAnalyzer":
        """Build an analyzer from a list of fitted ``compute_biotuner`` objects.

        Parameters
        ----------
        bt_list : list of compute_biotuner
            Each element should have had at least ``peaks_extraction()``
            called.  Additional attributes (``peaks_metrics``,
            ``scale_metrics``, ``diss_scale``, etc.) are used when present.
        tuning : str, default='peaks_ratios'
            Which tuning attribute to use as the primary ratio source.
            Supported: ``'peaks_ratios'``, ``'peaks_ratios_cons'``,
            ``'diss_scale'``, ``'HE_scale'``, ``'euler_fokker'``,
            ``'harm_tuning'``, ``'harm_fit_tuning'``,
            ``'extended_peaks_ratios'``.
        n_hist_bins : int, default=240
        tolerance_cents : float, default=30.0
        representation : str, default='cents_histogram'
            Feature representation consumed by Markov / Wasserstein / DMD /
            Latent / Topology.  One of:

            - ``'cents_histogram'``       : 240-bin cents distribution (default).
            - ``'harmonicity_spectrum'``  : 1-D power-weighted harmonicity per
              frequency.  Requires raw signal in ``bt.data``.  Preserves
              frequency identity (alpha vs gamma).
            - ``'harmonicity_matrix'``    : 2-D F×F pairwise harmonicity matrix
              flattened to F² per window.  Strictly more information than the
              spectrum (no collapse along the second frequency axis).

            Grammar is always cents-based (it labels ratios via JI dictionary).
            The MIDI/SCL rendering bridge always uses cents histograms,
            regardless of this parameter.
        representation_kwargs : dict, optional
            Forwarded to :func:`encode_harmonicity_spectrum` /
            :func:`encode_harmonicity_matrices`.  Common keys:
            ``fmin``, ``fmax``, ``precision_hz``, ``metric``, ``n_harms``.

        Returns
        -------
        HarmonicSequenceAnalyzer
        """
        if tuning not in TUNING_ATTRS:
            raise ValueError(
                f"Unknown tuning '{tuning}'. Valid names: {list(TUNING_ATTRS)}"
            )
        inst = cls(
            n_hist_bins=n_hist_bins,
            tolerance_cents=tolerance_cents,
            representation=representation,
            representation_kwargs=representation_kwargs,
        )
        inst._bt_list = bt_list
        inst.ratios_list = [extract_tuning(bt, tuning) for bt in bt_list]
        return inst

    @classmethod
    def from_ratios_list(
        cls,
        ratios_list: List[List[float]],
        n_hist_bins: int = 240,
        tolerance_cents: float = 30.0,
    ) -> "HarmonicSequenceAnalyzer":
        """Build an analyzer directly from a list of ratio sets (no bt objects).

        Parameters
        ----------
        ratios_list : list of list of float, length T
        n_hist_bins : int, default=240
        tolerance_cents : float, default=30.0

        Returns
        -------
        HarmonicSequenceAnalyzer
        """
        inst = cls(n_hist_bins=n_hist_bins, tolerance_cents=tolerance_cents)
        inst.ratios_list = ratios_list
        return inst

    # ── Lazy encoding caches ─────────────────────────────────────────────────

    @property
    def histograms(self) -> np.ndarray:
        """Cents-histogram matrix (T × n_hist_bins), computed on first access."""
        if self._histograms is None:
            self._histograms = encode_histograms(
                self.ratios_list, n_bins=self.n_hist_bins
            )
        return self._histograms

    @property
    def scalar_metrics(self) -> np.ndarray:
        """Scalar metric matrix (T × D) from bt objects, or empty array."""
        if self._scalar_metrics is None:
            if self._bt_list is not None:
                self._scalar_metrics = encode_scalar_metrics(self._bt_list)
            else:
                self._scalar_metrics = np.full(
                    (len(self.ratios_list), 0), np.nan
                )
        return self._scalar_metrics

    @property
    def features(self) -> np.ndarray:
        """Active feature matrix (T, D) consumed by the fit methods.

        Determined by ``self.representation``:

        - ``'cents_histogram'``      → :attr:`histograms` (T × n_hist_bins).
        - ``'harmonicity_spectrum'`` → (T × F) harmonicity per frequency.
        - ``'harmonicity_matrix'``   → (T × F²) flattened pairwise matrices.

        Cached on first access.  For the unflattened (T × F × F) tensor of
        the matrix representation, see :attr:`harmonicity_matrices_`.
        """
        if self._features is None:
            if self.representation == "cents_histogram":
                self._features = self.histograms
            elif self.representation == "harmonicity_spectrum":
                if self._bt_list is None:
                    raise RuntimeError(
                        "representation='harmonicity_spectrum' needs "
                        "bt_list with raw signals (use from_biotuner_list)."
                    )
                H, freqs = encode_harmonicity_spectrum(
                    self._bt_list, **self.representation_kwargs
                )
                self._features = H
                self._features_freqs = freqs
            elif self.representation == "harmonicity_matrix":
                if self._bt_list is None:
                    raise RuntimeError(
                        "representation='harmonicity_matrix' needs "
                        "bt_list with raw signals (use from_biotuner_list)."
                    )
                M, freqs = encode_harmonicity_matrices(
                    self._bt_list, **self.representation_kwargs
                )
                self._harmonicity_matrices = M
                self._features = M.reshape(M.shape[0], -1)
                self._features_freqs = freqs
            else:
                raise ValueError(
                    f"Unknown representation '{self.representation}'."
                )
        return self._features

    @property
    def harmonicity_matrices_(self) -> Optional[np.ndarray]:
        """The (T, F, F) tensor when ``representation='harmonicity_matrix'``.

        ``None`` for other representations.
        """
        if self.representation == "harmonicity_matrix":
            _ = self.features                 # force compute
        return self._harmonicity_matrices

    @property
    def features_freqs_(self) -> Optional[np.ndarray]:
        """Frequency axis (Hz) for the harmonicity representations, else ``None``."""
        return self._features_freqs

    # ── fit methods ──────────────────────────────────────────────────────────

    def fit_markov(
        self,
        n_states=5,
        order: int = 1,
        auto_k_range: Tuple[int, int] = (2, 10),
        use_hmm: bool = False,
        random_state: int = 42,
        **kwargs,
    ) -> HarmonicMarkov:
        """Fit a Markov chain (and optionally HMM) on the active feature matrix.

        Consumes :attr:`features` — i.e. cents histograms, the harmonicity
        spectrum, or the flattened harmonicity matrix, depending on
        ``self.representation``.

        Parameters
        ----------
        n_states : int or ``'auto'``, default=5
            Number of states.  Pass ``'auto'`` to select K automatically via
            silhouette score (see :func:`find_optimal_n_states`).
        order : int, default=1
            Markov chain order (memory length).  ``order=1`` is the standard
            first-order chain; higher values condition on the last *order*
            states.
        auto_k_range : (int, int), default=(2, 10)
            Search range when ``n_states='auto'``.
        use_hmm : bool, default=False
        random_state : int, default=42

        Returns
        -------
        HarmonicMarkov  (also stored as ``self.markov``)
        """
        self.markov = HarmonicMarkov(
            n_states=n_states,
            order=order,
            auto_k_range=auto_k_range,
            use_hmm=use_hmm,
            random_state=random_state,
        ).fit(self.features)
        return self.markov

    def fit_wasserstein(self) -> WassersteinTrajectory:
        """Compute pairwise Wasserstein distances on the active features.

        For the cents-histogram representation, distances are over the
        cents axis (interval space).  For the harmonicity representations,
        each row is first L1-normalised so it is treated as a probability
        density over frequency (or pairwise frequency for the matrix mode).

        Returns
        -------
        WassersteinTrajectory  (also stored as ``self.wasserstein``)
        """
        X = self.features
        # Histograms already sum to 1; harmonicity values do not.
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        X_norm = X / row_sums
        self.wasserstein = WassersteinTrajectory(
            n_bins=X_norm.shape[1]
        ).fit(X_norm)
        return self.wasserstein

    def fit_dmd(
        self,
        rank: Optional[int] = None,
        center: bool = True,
        use_histograms: bool = False,
        **kwargs,
    ) -> HarmonicDMD:
        """Fit DMD on the scalar metric (or histogram PCA) feature matrix.

        Parameters
        ----------
        rank : int or None
            SVD truncation rank.
        center : bool, default=True
            Subtract temporal mean before fitting.
        use_histograms : bool, default=False
            If ``True``, apply PCA (10 components) to the histograms first
            and run DMD on the reduced representation.  Requires sklearn.

        Returns
        -------
        HarmonicDMD  (also stored as ``self.dmd``)
        """
        if use_histograms:
            if not _SKLEARN:
                raise ImportError(
                    "scikit-learn required when use_histograms=True."
                )
            F = self.features                             # respects representation
            n_comp = min(10, F.shape[0] - 1, F.shape[1])
            X = PCA(n_components=n_comp).fit_transform(F)
        else:
            X = self.scalar_metrics
        self.dmd = HarmonicDMD(rank=rank, center=center).fit(X)
        return self.dmd

    def fit_latent(
        self,
        latent_dim: int = 3,
        random_state: int = 42,
        **kwargs,
    ) -> HarmonicLatentSpace:
        """Fit a PCA-based latent harmonic space on the histogram encoding.

        Parameters
        ----------
        latent_dim : int, default=3
        random_state : int, default=42

        Returns
        -------
        HarmonicLatentSpace  (also stored as ``self.latent``)
        """
        self.latent = HarmonicLatentSpace(
            latent_dim=latent_dim, random_state=random_state
        ).fit(self.features)
        return self.latent

    def fit_topology(
        self,
        scalar_key: str = "harmsim",
        embedding_dim: int = 3,
        delay: int = 1,
        **kwargs,
    ) -> HarmonicTopology:
        """Fit TDA on a scalar harmonicity time series.

        Parameters
        ----------
        scalar_key : str, default='harmsim'
            Which scalar to use for the Takens embedding.  Options:

            - Any key in ``peaks_metrics``:
              ``'harmsim'``, ``'cons'``, ``'tenney'``, ``'harm_fit'``,
              ``'subharm_tension'``
            - Any key in ``scale_metrics``:
              ``'dissonance'``, ``'HE'``, ``'diss_harm_sim'``, etc.
            - ``'latent_N'`` (e.g. ``'latent_0'``) uses the N-th dimension
              of a previously fitted ``HarmonicLatentSpace`` model.
        embedding_dim : int, default=3
        delay : int, default=1

        Returns
        -------
        HarmonicTopology  (also stored as ``self.topology``)
        """
        series = self._get_scalar_series(scalar_key)
        self.topology = HarmonicTopology(
            embedding_dim=embedding_dim, delay=delay
        ).fit(series)
        return self.topology

    def fit_grammar(
        self,
        n_gram: int = 2,
        **kwargs,
    ) -> HarmonicGrammar:
        """Fit an N-gram interval grammar on the ratio sequence.

        Parameters
        ----------
        n_gram : int, default=2  (bigram)

        Returns
        -------
        HarmonicGrammar  (also stored as ``self.grammar``)
        """
        self.grammar = HarmonicGrammar(
            tolerance_cents=self.tolerance_cents, n_gram=n_gram
        ).fit(self.ratios_list)
        return self.grammar

    def fit_all(
        self,
        n_states=5,
        markov_order: int = 1,
        auto_k_range: Tuple[int, int] = (2, 10),
        latent_dim: int = 3,
        n_gram: int = 2,
        topology_scalar: str = "harmsim",
        **kwargs,
    ) -> "HarmonicSequenceAnalyzer":
        """Run all six modelling approaches, silently skipping any that fail.

        Parameters
        ----------
        n_states : int or ``'auto'``, default=5
            Number of Markov states.  ``'auto'`` triggers automatic K selection
            via silhouette score over ``auto_k_range``.
        markov_order : int, default=1
            Markov chain memory depth.
        auto_k_range : (int, int), default=(2, 10)
            K search range when ``n_states='auto'``.
        latent_dim : int, default=3
            Latent space dimensionality.
        n_gram : int, default=2
            Grammar N-gram order.
        topology_scalar : str, default='harmsim'
            Scalar key for the TDA embedding.

        Returns
        -------
        self
        """
        tasks = [
            ("markov",      self.fit_markov,      {"n_states": n_states,
                                                    "order": markov_order,
                                                    "auto_k_range": auto_k_range}),
            ("wasserstein", self.fit_wasserstein, {}),
            ("dmd",         self.fit_dmd,         {}),
            ("latent",      self.fit_latent,      {"latent_dim": latent_dim}),
            ("topology",    self.fit_topology,    {"scalar_key": topology_scalar}),
            ("grammar",     self.fit_grammar,     {"n_gram": n_gram}),
        ]
        for name, fn, kw in tasks:
            try:
                fn(**kw)
            except Exception as exc:
                warnings.warn(
                    f"[HarmonicSequenceAnalyzer] '{name}' failed: {exc}",
                    stacklevel=2,
                )
        return self

    # ── helpers ───────────────────────────────────────────────────────────────

    def _get_scalar_series(self, key: str) -> np.ndarray:
        """Extract a named scalar time series for TDA or other 1-D analysis.

        Looks up the key in (1) fitted latent dimensions, (2) ``peaks_metrics``
        and ``scale_metrics`` of bt objects, or (3) falls back to computing
        mean harmsim from stored ratios.

        Parameters
        ----------
        key : str

        Returns
        -------
        series : ndarray, shape (T,)
        """
        if key.startswith("latent_"):
            if self.latent is None:
                raise RuntimeError(
                    "Fit HarmonicLatentSpace first (call fit_latent()) "
                    f"before using scalar_key='{key}'."
                )
            dim = int(key.split("_")[1])
            return self.latent.trajectory()[:, dim]

        if key in ("mean_features", "feature_mean"):
            # Per-window mean of the active representation.  For
            # 'harmonicity_spectrum' this is the mean harmonicity across
            # frequencies — a natural scalar harmonicity-over-time signal
            # that preserves the power-weighted information the cents
            # histogram throws away.
            return self.features.mean(axis=1)

        if key in ("max_features", "feature_max"):
            return self.features.max(axis=1)

        if self._bt_list is not None:
            vals = []
            for bt in self._bt_list:
                pm = getattr(bt, "peaks_metrics", {}) or {}
                sm = getattr(bt, "scale_metrics", {}) or {}
                v = pm.get(key, sm.get(key, np.nan))
                vals.append(_safe_float(v))
            series = np.array(vals, dtype=float)
            # Fall back to harmsim from ratios if all NaN
            if not np.all(np.isnan(series)):
                return series

        # Compute harmsim on-the-fly from stored ratios
        if key == "harmsim":
            vals = []
            for ratios in self.ratios_list:
                if len(ratios) >= 2:
                    try:
                        vals.append(float(np.mean(ratios2harmsim(ratios))))
                    except Exception:
                        vals.append(np.nan)
                else:
                    vals.append(np.nan)
            return np.array(vals, dtype=float)

        raise ValueError(
            f"Unknown scalar_key '{key}' and no bt_list to look it up from. "
            f"Use 'harmsim', a peaks_metrics key, a scale_metrics key, "
            f"or 'latent_N' after calling fit_latent()."
        )

    # ── Rendering / export bridge ─────────────────────────────────────────────

    def get_histograms(
        self,
        source: str = "observed",
        *,
        t1: Optional[int] = None,
        t2: Optional[int] = None,
        n_steps: int = 16,
        n_predict: int = 16,
    ) -> np.ndarray:
        """Pull a histogram sequence from any source — observed or generated.

        Single entry point for the rendering bridge.  Returns a ``(T, n_bins)``
        matrix that can be fed to :func:`histograms_to_midi` /
        :func:`histogram_to_scl`.

        Parameters
        ----------
        source : str
            One of:

            - ``'observed'``        : the recorded histograms (one per window)
            - ``'wasserstein_interp'`` : Wasserstein barycenter path between
              ``t1`` and ``t2`` (requires :meth:`fit_wasserstein`)
            - ``'latent_interp'``   : decoded path between latent points at
              ``t1`` and ``t2`` (requires :meth:`fit_latent`)
            - ``'markov_centroids'`` : K prototype histograms (KMeans cluster
              centres; requires :meth:`fit_markov`)
            - ``'markov_sample'``   : sample a state sequence of length
              ``n_steps`` and return the corresponding centroids
            - ``'dmd_predict'``     : forward-extrapolate ``n_predict`` steps
              (requires DMD fitted with ``use_histograms=True``)
        t1, t2 : int, optional
            Window indices for interpolation modes (default: 0 and T-1).
        n_steps : int, default=16
            Length of the interpolation / Markov sample.
        n_predict : int, default=16
            Steps to extrapolate for ``'dmd_predict'``.

        Returns
        -------
        H : ndarray, shape (T_out, n_hist_bins)
        """
        if source == "observed":
            return self.histograms

        # Generative bridge sources require the active representation to be
        # cents — otherwise the fitted models live in spectrum-space and their
        # outputs cannot be decoded back to playable ratios.
        cents_only = {
            "wasserstein_interp", "latent_interp",
            "markov_centroids", "markov_sample", "dmd_predict",
        }
        if source in cents_only and self.representation != "cents_histogram":
            raise RuntimeError(
                f"source='{source}' requires representation='cents_histogram' "
                f"(got '{self.representation}'). The fitted model lives in "
                "spectrum-space; outputs cannot be decoded to playable tunings. "
                "Re-fit a separate analyzer with the default representation, "
                "or use source='observed' to render the recorded histograms."
            )

        T = len(self.ratios_list)
        a = 0 if t1 is None else int(t1)
        b = (T - 1) if t2 is None else int(t2)

        if source == "wasserstein_interp":
            wt = self.wasserstein
            if wt is None:
                raise RuntimeError("Call fit_wasserstein() first.")
            return np.stack(wt.interpolate_pair(a, b, n_steps=n_steps), axis=0)

        if source == "latent_interp":
            ls = self.latent
            if ls is None:
                raise RuntimeError("Call fit_latent() first.")
            Z = ls.trajectory()
            Z_path = ls.interpolate(Z[a], Z[b], n_steps=n_steps)
            return ls.decode(Z_path)

        if source == "markov_centroids":
            mk = self.markov
            if mk is None or mk._km is None:
                raise RuntimeError("Call fit_markov() first.")
            return mk._km.cluster_centers_

        if source == "markov_sample":
            mk = self.markov
            if mk is None or mk._km is None:
                raise RuntimeError("Call fit_markov() first.")
            rng = np.random.default_rng(mk.random_state)
            T_mat = mk.transition_matrix_
            centroids = mk._km.cluster_centers_
            state = int(np.argmax(mk.steady_state_))
            states = [state]
            for _ in range(n_steps - 1):
                row = T_mat[states[-1]]
                if row.sum() <= 0:
                    state = int(rng.integers(0, mk.n_states))
                else:
                    state = int(rng.choice(mk.n_states, p=row / row.sum()))
                states.append(state)
            return centroids[states]

        if source == "dmd_predict":
            if self.dmd is None:
                raise RuntimeError("Call fit_dmd(use_histograms=True) first.")
            X_pred = self.dmd.reconstruct(n_predict)
            # If fitted on histograms-via-PCA we cannot trivially invert; the
            # caller is expected to pass DMD trained directly on histograms.
            return np.maximum(X_pred, 0.0)

        raise ValueError(
            f"Unknown source '{source}'. "
            "Choose from: observed, wasserstein_interp, latent_interp, "
            "markov_centroids, markov_sample, dmd_predict."
        )

    def to_scl(
        self,
        index: int = 0,
        name: Optional[str] = None,
        source: str = "observed",
        *,
        n_peaks: Optional[int] = None,
        prominence: float = 0.0,
        write: bool = False,
        **kwargs: Any,
    ) -> str:
        """Export one histogram of the sequence as a Scala (.scl) tuning.

        Parameters
        ----------
        index : int, default=0
            Row of the histogram source to export.
        name : str, optional
            Scale name (also filename when ``write=True``).  Defaults to
            ``f"biotuner_{source}_{index}"``.
        source : str
            See :meth:`get_histograms`.
        n_peaks, prominence, write
            Forwarded to :func:`histogram_to_scl`.

        Returns
        -------
        str  Scala file contents.
        """
        H = self.get_histograms(source=source, **kwargs)
        if not (0 <= index < len(H)):
            raise IndexError(f"index {index} out of range (T={len(H)}).")
        scale_name = name or f"biotuner_{source}_{index}"
        return histogram_to_scl(
            H[index],
            name=scale_name,
            n_peaks=n_peaks,
            prominence=prominence,
            write=write,
        )

    def to_midi(
        self,
        filename: str = "harmonic_sequence",
        source: str = "observed",
        *,
        base_freq: float = 220.0,
        duration_beats: float = 1.0,
        velocity: int = 80,
        n_peaks: Optional[int] = 5,
        prominence: float = 0.0,
        flux_modulated_durations: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Export a histogram sequence as a microtonal MIDI file.

        Parameters
        ----------
        filename : str
            Output filename stem (``.mid`` is appended).
        source : str
            See :meth:`get_histograms`.
        base_freq : float, default=220.0
            Frequency that ratio 1.0 maps to (Hz).
        duration_beats : float, default=1.0
            Beats per chord.  Ignored when ``flux_modulated_durations=True``.
        velocity : int, default=80
        n_peaks : int or None, default=5
            Maximum simultaneous voices per chord (also the MIDI channel cap
            when ``microtonal=True``).
        prominence : float, default=0.0
        flux_modulated_durations : bool, default=False
            When ``True`` (and Wasserstein is fitted), each chord's duration
            is inversely proportional to local flux: high-flux moments get
            shorter notes, stable moments get longer.  Useful for sonifying
            the harmonic-velocity envelope alongside the tunings themselves.
        **kwargs
            Forwarded to :meth:`get_histograms` (e.g. ``t1``, ``t2``,
            ``n_steps``).

        Returns
        -------
        mido.MidiFile
        """
        H = self.get_histograms(source=source, **kwargs)

        if flux_modulated_durations and self.wasserstein is not None and source == "observed":
            f = self.wasserstein.flux_
            f = np.r_[f, f[-1]]                   # length T (pad final)
            f_norm = f / (f.max() + 1e-9)
            durations = duration_beats * (1.5 - f_norm)   # 0.5 .. 1.5 beats
        else:
            durations = duration_beats

        return histograms_to_midi(
            H,
            filename=filename,
            base_freq=base_freq,
            duration_beats=durations,
            velocity=velocity,
            n_peaks=n_peaks,
            prominence=prominence,
        )

    def summary(self) -> str:
        """Return a concise text summary of all fitted models."""
        rep = self.representation
        feat_dim = self._features.shape[1] if self._features is not None else "—"
        lines = [
            "HarmonicSequenceAnalyzer",
            f"  T={len(self.ratios_list)} timepoints | "
            f"representation={rep} (D={feat_dim}) | "
            f"n_hist_bins={self.n_hist_bins} | "
            f"tolerance_cents={self.tolerance_cents}",
        ]
        if self.markov is not None:
            ss = self.markov.steady_state_
            dominant = int(np.argmax(ss))
            lines.append(
                f"  [Markov]      n_states={self.markov.n_states} | "
                f"dominant_state={dominant} ({ss[dominant]:.2f}) | "
                f"H_trans={self.markov.transition_entropy_:.2f} bits"
            )
        if self.wasserstein is not None:
            f = self.wasserstein.flux_
            lines.append(
                f"  [Wasserstein] mean_flux={np.mean(f):.4f} | "
                f"max_flux={np.max(f):.4f} | "
                f"std_flux={np.std(f):.4f}"
            )
        if self.dmd is not None:
            osc_eig, _ = self.dmd.oscillatory_modes()
            lines.append(
                f"  [DMD]         rank={len(self.dmd.eigenvalues_)} | "
                f"oscillatory_modes={len(osc_eig)}"
            )
        if self.latent is not None:
            evr = self.latent.explained_variance_ratio_
            lines.append(
                f"  [Latent]      dim={self.latent.latent_dim} | "
                f"var_explained={evr.sum():.1%} | "
                f"recon_err={self.latent.reconstruction_error_:.4f}"
            )
        if self.topology is not None:
            beta = self.topology.betti_numbers_
            fp = self.topology.session_fingerprint()
            lines.append(
                f"  [Topology]    beta0={beta[0]}, beta1={beta[1]} | "
                f"mean_H0_pers={fp[0]:.3f} | "
                f"max_H0_pers={fp[1]:.3f}"
            )
        if self.grammar is not None:
            top1 = self.grammar.top_ngrams(1)
            top_str = str(top1[0]) if top1 else "N/A"
            lines.append(
                f"  [Grammar]     vocab={len(self.grammar.vocabulary_)} | "
                f"H_trans={self.grammar.transition_entropy_:.2f} bits | "
                f"top_ngram={top_str}"
            )
        not_fitted = [
            n for n, attr in [
                ("markov", self.markov), ("wasserstein", self.wasserstein),
                ("dmd", self.dmd), ("latent", self.latent),
                ("topology", self.topology), ("grammar", self.grammar),
            ]
            if attr is None
        ]
        if not_fitted:
            lines.append(f"  [Not fitted]  {', '.join(not_fitted)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"HarmonicSequenceAnalyzer("
            f"T={len(self.ratios_list)}, "
            f"n_hist_bins={self.n_hist_bins})"
        )
