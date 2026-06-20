"""biotuner.math_series — match biosignal ratios to mathematical sequences.

Module type: Object

Generate ratio sets from classic integer sequences (Fibonacci, Lucas,
Padovan, Pell, Jacobsthal, Mersenne, Hofstadter-Q, harmonics, subharmonics,
triangular, Farey), score how present each series is in a biosignal's peak
ratios, and derive musical structures (scales and consonance-selected modes)
from the matched subset of a series.

The entry point is the :class:`math_series` pipeline, which accepts **either**
a fitted :class:`~biotuner.biotuner_object.compute_biotuner` instance **or** a
:class:`~biotuner.harmonic_input.HarmonicInput` descriptor, normalises it to a
single list of octave-folded peak ratios, and compares those against each
series. Matching conservatism is controlled by ``maxdenom`` (the maximum
denominator of the rational approximation used to decide whether two ratios
are "the same"): a low ``maxdenom`` collapses nearby ratios onto simple
fractions (lenient matching), a high ``maxdenom`` keeps fine distinctions
(strict matching).
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from biotuner.biotuner_utils import findsubsets, ratio2frac, ratios2cents
from biotuner.harmonic_input import SCALE_KEYS
from biotuner.metrics import compute_consonance, dyad_similarity, metric_denom
from biotuner.plot_config import get_color_palette, set_biotuner_style
from biotuner.scale_construction import create_mode, tuning_reduction

# Series compared by default — mirrors the notebook's normalized bar chart.
DEFAULT_SERIES: List[str] = ["fibonacci", "lucas", "farey", "harmonics"]

# Ratio/pair as carried internally: (folded_ratio_in_[1, octave), (element_1, element_2)).
RatioPair = Tuple[float, Tuple[float, float]]


# ---------------------------------------------------------------------------
# Sequence generators
# ---------------------------------------------------------------------------
def fibonacci(order: int) -> List[int]:
    """Generate the first ``order`` Fibonacci numbers.

    Examples
    --------
    >>> fibonacci(8)
    [0, 1, 1, 2, 3, 5, 8, 13]
    """
    seq = [0, 1]
    while len(seq) < order:
        seq.append(seq[-1] + seq[-2])
    return seq[:order]


def lucas(order: int, seed: Sequence[int] = (2, 1)) -> List[int]:
    """Generate the first ``order`` terms of a Lucas-type sequence.

    Parameters
    ----------
    order : int
        Number of terms to return.
    seed : sequence of int, default=(2, 1)
        The two starting values. ``(2, 1)`` is the classic Lucas sequence.

    Examples
    --------
    >>> lucas(7)
    [2, 1, 3, 4, 7, 11, 18]
    """
    seq = list(seed)
    while len(seq) < order:
        seq.append(seq[-1] + seq[-2])
    return seq[:order]


def padovan(order: int) -> List[int]:
    """Generate the first ``order`` Padovan numbers.

    Examples
    --------
    >>> padovan(7)
    [1, 1, 1, 2, 2, 3, 4]
    """
    seq = [1, 1, 1]
    while len(seq) < order:
        seq.append(seq[-2] + seq[-3])
    return seq[:order]


def pell(order: int) -> List[int]:
    """Generate the first ``order`` Pell numbers.

    Examples
    --------
    >>> pell(6)
    [0, 1, 2, 5, 12, 29]
    """
    seq = [0, 1]
    while len(seq) < order:
        seq.append(2 * seq[-1] + seq[-2])
    return seq[:order]


def jacobsthal(order: int) -> List[int]:
    """Generate the first ``order`` Jacobsthal numbers.

    Examples
    --------
    >>> jacobsthal(6)
    [0, 1, 1, 3, 5, 11]
    """
    seq = [0, 1]
    while len(seq) < order:
        seq.append(seq[-1] + 2 * seq[-2])
    return seq[:order]


def mersenne(order: int) -> List[int]:
    """Generate the first ``order`` Mersenne numbers ``2**i - 1``.

    Examples
    --------
    >>> mersenne(5)
    [0, 1, 3, 7, 15]
    """
    return [2 ** i - 1 for i in range(order)]


def hofstadter_q(order: int) -> List[int]:
    """Generate the first ``order`` terms of the Hofstadter Q sequence.

    Examples
    --------
    >>> hofstadter_q(8)
    [1, 1, 2, 3, 3, 4, 5, 5]
    """
    q = [1, 1]
    for i in range(2, order):
        q.append(q[i - q[i - 1]] + q[i - q[i - 2]])
    return q[:order]


def harmonics(order: int) -> List[int]:
    """Generate the first ``order`` terms of the harmonic series.

    Examples
    --------
    >>> harmonics(5)
    [1, 2, 3, 4, 5]
    """
    return [i for i in range(1, order + 1)]


def subharmonics(order: int) -> List[float]:
    """Generate the first ``order`` terms of the subharmonic series.

    Examples
    --------
    >>> subharmonics(4)
    [1.0, 0.5, 0.3333333333333333, 0.25]
    """
    return [1.0 / i for i in range(1, order + 1)]


def triangular(order: int) -> List[int]:
    """Generate the first ``order`` triangular numbers.

    Examples
    --------
    >>> triangular(5)
    [1, 3, 6, 10, 15]
    """
    return [i * (i + 1) // 2 for i in range(1, order + 1)]


def farey(order: int) -> List[Tuple[int, int]]:
    """Generate the Farey sequence of a given ``order`` as ``(num, den)`` pairs.

    Unlike the integer sequences, each Farey term is already a fraction in
    ``[0, 1]``; the returned ``(num, den)`` pairs are treated directly as
    ratios (``num / den``) by :func:`series_ratio_pairs`.

    Examples
    --------
    >>> farey(4)
    [(0, 1), (1, 4), (1, 3), (1, 2), (2, 3), (3, 4), (1, 1)]
    """
    seq: List[Tuple[int, int]] = []
    a, b, c, d = 0, 1, 1, order
    seq.append((a, b))
    while c <= order:
        k = (order + b) // d
        a, b, c, d = c, d, k * c - a, k * d - b
        seq.append((a, b))
    return seq


#: Mapping of series name -> generator. Add a row here to register a series.
SERIES_FUNCS = {
    "fibonacci": fibonacci,
    "lucas": lucas,
    "padovan": padovan,
    "pell": pell,
    "jacobsthal": jacobsthal,
    "mersenne": mersenne,
    "hofstadter_q": hofstadter_q,
    "harmonics": harmonics,
    "subharmonics": subharmonics,
    "triangular": triangular,
    "farey": farey,
}


# ---------------------------------------------------------------------------
# Ratio helpers
# ---------------------------------------------------------------------------
def _fold_ratio(ratio: float, octave: float = 2.0) -> float:
    """Fold a ratio into the half-open period ``[1, octave)``.

    Mirrors the octave-reduction convention of
    :func:`~biotuner.biotuner_utils.compute_peak_ratios` (divide while
    ``>= octave``, multiply while ``< 1``) so series ratios live in the same
    range as biosignal peak ratios. Non-positive values are returned as-is.
    """
    r = float(ratio)
    if r <= 0:
        return r
    while r >= octave:
        r = r / octave
    while r < 1:
        r = r * octave
    return r


def _frac_key(ratio: float, maxdenom: int) -> Tuple[int, int]:
    """Return the ``(num, den)`` rational approximation used as a match key.

    Two ratios match iff this key is equal. ``maxdenom`` is the conservatism
    knob: lower values merge nearby ratios onto simpler fractions (lenient),
    higher values preserve fine distinctions (strict).
    """
    return tuple(ratio2frac(float(ratio), maxdenom=maxdenom))


def _coerce_ratio_list(values: Any) -> List[float]:
    """Coerce an arbitrary ratio container to a list of finite positive floats."""
    if values is None:
        return []
    arr = np.asarray([float(v) for v in values], dtype=np.float64).ravel()
    return [float(x) for x in arr if np.isfinite(x) and x > 0]


def _cents(ratios: Any) -> np.ndarray:
    """Convert ratio(s) to cents (1200*log2). Accepts a scalar or an array."""
    return 1200.0 * np.log2(np.asarray(ratios, dtype=float))


def _octave_comb_fit(ratios, peaks_hz, fmin, fmax, octaves):
    """Best-fit across-octave frequency grid of a series' ratios to peaks.

    Replicates the series ratios across ``octaves`` and scales by a base in
    ``[1, 2)`` chosen to minimise the mean cents distance from the peaks to the
    nearest grid line. Returns the winning grid (sorted Hz array).
    """
    ratios = np.asarray(ratios, dtype=float)
    peaks_hz = np.asarray(peaks_hz, dtype=float)
    if ratios.size == 0 or peaks_hz.size == 0:
        return np.array([])

    def grid_for(base):
        g = [base * r * (2.0 ** k) for k in octaves for r in ratios]
        return np.array(sorted({round(x, 4) for x in g if fmin <= x <= fmax}))

    peaks_c = _cents(peaks_hz)
    best_grid, best_d = np.array([]), np.inf
    for base in np.linspace(1.0, 2.0, 160, endpoint=False):
        grid = grid_for(base)
        if grid.size == 0:
            continue
        gc = _cents(grid)
        d = float(np.mean([np.min(np.abs(p - gc)) for p in peaks_c]))
        if d < best_d:
            best_grid, best_d = grid, d
    return best_grid


def series_ratio_pairs(
    name: str,
    order: int,
    octave: float = 2.0,
    which: str = "both",
    lucas_seed: Sequence[int] = (2, 1),
) -> List[RatioPair]:
    """Build the octave-folded ratio set of a series, keeping pair provenance.

    Parameters
    ----------
    name : str
        A key of :data:`SERIES_FUNCS`.
    order : int
        Number of terms (or, for ``"farey"``, the Farey order).
    octave : float, default=2.0
        Period the ratios are folded into (``[1, octave)``).
    which : {'both', 'high/low', 'low/high'}, default='both'
        For integer sequences, whether to keep ratios where the first element
        is larger (``high/low``), smaller (``low/high``), or both. Ignored for
        ``"farey"`` (whose terms are already fractions).
    lucas_seed : sequence of int, default=(2, 1)
        Starting pair forwarded to :func:`lucas` when ``name == "lucas"``.

    Returns
    -------
    list of (float, (float, float))
        Each entry is ``(folded_ratio, (element_1, element_2))``. The pair is
        kept unfolded so it can be plotted on the element/element scatter.
    """
    if name not in SERIES_FUNCS:
        raise ValueError(
            f"Unknown series '{name}'. Available: {sorted(SERIES_FUNCS)}"
        )
    func = SERIES_FUNCS[name]
    elements = func(order, lucas_seed) if name == "lucas" else func(order)

    pairs: List[RatioPair] = []
    if elements and all(isinstance(e, tuple) for e in elements):
        # Farey-style: each element is already a (num, den) fraction.
        for a, b in elements:
            if a == 0 or b == 0:
                continue
            pairs.append((_fold_ratio(a / b, octave), (float(a), float(b))))
    else:
        nums = [e for e in elements if e > 0]
        for a in nums:
            for b in nums:
                if a == b:
                    continue
                if which == "high/low" and not a > b:
                    continue
                if which == "low/high" and not a < b:
                    continue
                pairs.append((_fold_ratio(a / b, octave), (float(a), float(b))))
    return pairs


# Consonance functions selectable for mode reduction.
CONSONANCE_FUNCS = {
    "compute_consonance": compute_consonance,
    "dyad_similarity": dyad_similarity,
    "metric_denom": metric_denom,
}

def _series_colors(names: Sequence[str]) -> Dict[str, Any]:
    """Assign a stable biotuner-palette color to each series name."""
    palette = get_color_palette("biotuner_gradient", n_colors=max(len(names), 1))
    return {n: palette[i % len(palette)] for i, n in enumerate(names)}


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class math_series(object):
    """Identify which mathematical series best matches a biosignal's ratios.

    Accepts a fitted :class:`~biotuner.biotuner_object.compute_biotuner`
    instance or a :class:`~biotuner.harmonic_input.HarmonicInput`, extracts a
    list of octave-folded peak ratios, and scores each candidate series by the
    proportion of biosignal ratios it reproduces (under a max-denominator
    rational match). The matched subset of the best series can then be turned
    into a scale (:meth:`series_scale`) or a consonance-selected mode
    (:meth:`series_mode`).

    Parameters
    ----------
    source : compute_biotuner or HarmonicInput, optional
        The biosignal descriptor. Mutually exclusive with ``ratios``.
    ratios : list of float, optional
        Peak ratios supplied directly (mainly for testing). Used when
        ``source`` is None.
    ratios_source : str, default='peaks_ratios'
        Which scale to read off ``source``. For a ``compute_biotuner`` this is
        an attribute name (e.g. ``"peaks_ratios"`` or
        ``"extended_peaks_ratios"``). For a ``HarmonicInput`` it is looked up
        in ``ratios_alternates``, falling back to the canonical ratios.
    series_names : sequence of str, optional
        Series to compare. Defaults to :data:`DEFAULT_SERIES`.
    order : int, default=20
        Number of terms generated per series (Farey order for ``"farey"``).
    maxdenom : int, default=24
        Maximum denominator of the rational match. Lower = more lenient
        (nearby ratios collapse onto simple fractions), higher = stricter.
        Useful discrimination typically lives in roughly ``16``–``48``: below
        that the small fraction grid saturates (every series matches), above
        it full-precision peak ratios rarely snap onto any simple fraction.
    octave : float, default=2.0
        Period the ratios are folded into.
    which : {'both', 'high/low', 'low/high'}, default='both'
        Pair direction passed to :func:`series_ratio_pairs`.
    lucas_seed : sequence of int, default=(2, 1)
        Starting pair for the Lucas series.

    Attributes
    ----------
    ratios : list of float
        The extracted biosignal peak ratios (octave-folded, in ``[1, octave)``).
    series_scores : dict
        Per-series result dict (populated by :meth:`analyze`). Keys per series:
        ``proportion``, ``proportion_normalized``, ``n_matched``, ``n_target``,
        ``n_series_ratios``, ``matched_keys``, ``matched_series_pairs``,
        ``matched_target_ratios``.
    best_series : str or None
        Name of the series with the highest raw proportion.

    Examples
    --------
    >>> ms = math_series(ratios=[1.5, 1.6, 1.25, 1.333], series_names=["fibonacci", "harmonics"], maxdenom=16)
    >>> ms.analyze().best_series in {"fibonacci", "harmonics"}
    True
    """

    def __init__(
        self,
        source: Any = None,
        ratios: Optional[Sequence[float]] = None,
        *,
        ratios_source: str = "peaks_ratios",
        series_names: Optional[Sequence[str]] = None,
        order: int = 20,
        maxdenom: int = 24,
        octave: float = 2.0,
        which: str = "both",
        lucas_seed: Sequence[int] = (2, 1),
    ) -> None:
        self.ratios_source = ratios_source
        self.series_names = list(series_names) if series_names else list(DEFAULT_SERIES)
        self.order = order
        self.maxdenom = maxdenom
        self.octave = octave
        self.which = which
        self.lucas_seed = tuple(lucas_seed)

        unknown = [n for n in self.series_names if n not in SERIES_FUNCS]
        if unknown:
            raise ValueError(
                f"Unknown series {unknown}. Available: {sorted(SERIES_FUNCS)}"
            )

        if source is not None:
            self.ratios = self._extract_target_ratios(source, ratios_source)
        elif ratios is not None:
            self.ratios = [_fold_ratio(r, octave) for r in _coerce_ratio_list(ratios)]
        else:
            raise ValueError("Provide either `source` (biotuner / HarmonicInput) or `ratios`.")

        if not self.ratios:
            raise ValueError(
                f"No usable ratios found for ratios_source='{ratios_source}'. "
                "Did you run peaks_extraction()/peaks_extension()?"
            )

        # Peak frequencies (Hz) when the source carries them — used by the
        # across-octave comb plot. None for ratios-only construction.
        self.peaks_hz: Optional[np.ndarray] = (
            self._extract_peaks_hz(source, ratios_source) if source is not None else None
        )

        self.series_scores: Dict[str, Dict[str, Any]] = {}
        self.best_series: Optional[str] = None
        self._series_pairs: Dict[str, List[RatioPair]] = {}
        self._target_keys: Dict[Tuple[int, int], float] = {}
        self._colors = _series_colors(self.series_names)

    # ------------------------------------------------------------- extraction
    @staticmethod
    def _extract_peaks_hz(source: Any, ratios_source: str) -> Optional[np.ndarray]:
        """Best-effort peak frequencies (Hz) from a biotuner / HarmonicInput."""
        if "extended" in ratios_source:
            val = getattr(source, "extended_peaks", None)
            if val is not None:
                arr = np.asarray(val, dtype=float).ravel()
                if arr.size:
                    return arr
        if hasattr(source, "to_peaks"):  # HarmonicInput
            try:
                arr = np.asarray(source.to_peaks(), dtype=float).ravel()
                if arr.size:
                    return arr
            except Exception:
                pass
        val = getattr(source, "peaks", None)
        if val is not None:
            arr = np.asarray(val, dtype=float).ravel()
            if arr.size:
                return arr
        return None
    @staticmethod
    def _extract_target_ratios(source: Any, ratios_source: str) -> List[float]:
        """Normalise a biotuner object or HarmonicInput to a list of ratios."""
        # HarmonicInput-like: prefer the requested alternate scale, else canonical.
        if hasattr(source, "to_ratios") and hasattr(source, "ratios_alternates"):
            alternates = getattr(source, "ratios_alternates", None) or {}
            if ratios_source in alternates:
                return _coerce_ratio_list(alternates[ratios_source])
            canonical = getattr(source, "ratios_source", None)
            if canonical is not None and canonical != ratios_source:
                warnings.warn(
                    f"HarmonicInput has no '{ratios_source}' scale "
                    f"(carries '{canonical}'); using its canonical ratios. "
                    "Rebuild with scale_priority/include_alternates to select a "
                    "specific scale.",
                    stacklevel=2,
                )
            return _coerce_ratio_list(source.to_ratios())

        # compute_biotuner-like (or duck-typed mock): read the attribute directly.
        direct = getattr(source, ratios_source, None)
        if direct is not None:
            return _coerce_ratio_list(direct)

        # Last resort: lift via the canonical HarmonicInput bridge.
        if hasattr(source, "to_harmonic_input"):
            hi = source.to_harmonic_input(
                scale_priority=[ratios_source] + list(SCALE_KEYS)
            )
            return math_series._extract_target_ratios(hi, ratios_source)

        raise TypeError(
            "Unsupported `source`: expected a compute_biotuner instance, a "
            "HarmonicInput, or an object exposing the requested ratios."
        )

    def _ensure_analyzed(self) -> None:
        if not self.series_scores:
            self.analyze()

    # ---------------------------------------------------------------- analysis
    def analyze(self) -> "math_series":
        """Score every candidate series against the biosignal ratios.

        Populates :attr:`series_scores` and :attr:`best_series`. Returns
        ``self`` so calls can be chained (``math_series(bt).analyze()``).
        """
        self._target_keys = {}
        for r in self.ratios:
            self._target_keys.setdefault(_frac_key(r, self.maxdenom), r)
        target_key_set = set(self._target_keys)
        n_target = len(target_key_set)

        self.series_scores = {}
        self._series_pairs = {}
        for name in self.series_names:
            pairs = series_ratio_pairs(
                name, self.order, octave=self.octave,
                which=self.which, lucas_seed=self.lucas_seed,
            )
            self._series_pairs[name] = pairs

            key_to_pairs: Dict[Tuple[int, int], List[RatioPair]] = {}
            for ratio, ab in pairs:
                key_to_pairs.setdefault(_frac_key(ratio, self.maxdenom), []).append((ratio, ab))

            matched_keys = target_key_set & set(key_to_pairs)
            matched_series_pairs = [p for k in matched_keys for p in key_to_pairs[k]]
            proportion = len(matched_keys) / n_target if n_target else 0.0
            self.series_scores[name] = {
                "proportion": proportion,
                "n_matched": len(matched_keys),
                "n_target": n_target,
                "n_series_ratios": len(key_to_pairs),
                "matched_keys": sorted(matched_keys),
                "matched_series_pairs": matched_series_pairs,
                "matched_target_ratios": sorted(self._target_keys[k] for k in matched_keys),
            }

        total = sum(s["proportion"] for s in self.series_scores.values())
        for s in self.series_scores.values():
            s["proportion_normalized"] = (s["proportion"] / total) if total > 0 else 0.0

        self.best_series = max(
            self.series_scores,
            key=lambda n: self.series_scores[n]["proportion"],
            default=None,
        )
        return self

    def summary(self) -> pd.DataFrame:
        """Return a per-series score table sorted by raw proportion (descending)."""
        self._ensure_analyzed()
        rows = [
            {
                "series": name,
                "proportion": s["proportion"],
                "proportion_normalized": s["proportion_normalized"],
                "n_matched": s["n_matched"],
                "n_target": s["n_target"],
                "n_series_ratios": s["n_series_ratios"],
            }
            for name, s in self.series_scores.items()
        ]
        df = pd.DataFrame(rows).sort_values("proportion", ascending=False).reset_index(drop=True)
        return df

    # --------------------------------------------------------- musical output
    def series_scale(self, name: Optional[str] = None, include_unison: bool = True) -> List[float]:
        """Return the matched subset of a series as a sorted scale in ``[1, octave)``.

        Parameters
        ----------
        name : str, optional
            Series to use. Defaults to :attr:`best_series`.
        include_unison : bool, default=True
            Prepend ``1.0`` (the unison) if absent.

        Returns
        -------
        list of float
            Sorted, de-duplicated ratios of the matched series subset.
        """
        self._ensure_analyzed()
        name = name or self.best_series
        if name not in self.series_scores:
            raise ValueError(f"Series '{name}' was not analyzed.")
        ratios = sorted({round(p[0], 6) for p in self.series_scores[name]["matched_series_pairs"]})
        if include_unison and (not ratios or ratios[0] != 1.0):
            ratios = [1.0] + [r for r in ratios if r != 1.0]
        return ratios

    def series_mode(
        self,
        name: Optional[str] = None,
        n_steps: int = 7,
        method: str = "subset",
        function: Any = compute_consonance,
    ) -> List[float]:
        """Reduce a series scale to an ``n_steps`` consonance-selected mode.

        Wraps the house mode-selection helpers
        :func:`~biotuner.scale_construction.create_mode` (``method="subset"``)
        and :func:`~biotuner.scale_construction.tuning_reduction`
        (``method="pairwise"``).

        Parameters
        ----------
        name : str, optional
            Series to use. Defaults to :attr:`best_series`.
        n_steps : int, default=7
            Number of steps in the reduced mode (capped at the scale size).
        method : {'subset', 'pairwise'}, default='subset'
            Selection strategy. ``"subset"`` searches all step combinations
            (best for small scales); ``"pairwise"`` greedily adds the most
            consonant pairs.
        function : callable, default=compute_consonance
            Consonance metric. One of
            :func:`~biotuner.metrics.compute_consonance`,
            :func:`~biotuner.metrics.dyad_similarity`,
            :func:`~biotuner.metrics.metric_denom` (or pass the string name).

        Returns
        -------
        list of float
            The mode, sorted ascending.
        """
        self._ensure_analyzed()
        if isinstance(function, str):
            if function not in CONSONANCE_FUNCS:
                raise ValueError(
                    f"Unknown consonance function '{function}'. "
                    f"Available: {sorted(CONSONANCE_FUNCS)}"
                )
            function = CONSONANCE_FUNCS[function]

        tuning = self.series_scale(name)
        n_steps = min(n_steps, len(tuning))
        if len(tuning) < 4 or n_steps >= len(tuning):
            # Too small to reduce — the matched subset is already the mode.
            return sorted(tuning)

        if method == "subset" and math.comb(len(tuning), n_steps) > 50_000:
            warnings.warn(
                f"subset search over C({len(tuning)}, {n_steps}) is large; "
                "falling back to method='pairwise'.",
                stacklevel=2,
            )
            method = "pairwise"

        if method == "subset":
            mode = create_mode(tuning, n_steps=n_steps, function=function)
        elif method == "pairwise":
            _, mode, _ = tuning_reduction(tuning, mode_n_steps=n_steps, function=function)
        else:
            raise ValueError("method must be 'subset' or 'pairwise'.")
        return sorted(mode)

    def scale_cents(self, name: Optional[str] = None) -> List[float]:
        """Return :meth:`series_scale` expressed in cents."""
        return ratios2cents(self.series_scale(name))

    # -------------------------------------------------------------- plotting
    def plot_proportions(
        self,
        normalized: bool = True,
        ax: Optional["plt.Axes"] = None,
        plot: bool = True,
        save: bool = False,
        savename: str = "series_proportions",
    ) -> "plt.Figure":
        """Bar chart of the match proportion per series.

        Parameters
        ----------
        normalized : bool, default=True
            Plot proportions normalised to sum to 1 (across the compared
            series) rather than raw proportions.
        ax : matplotlib.axes.Axes, optional
            Draw onto an existing axis (for composing multi-panel figures).
            A new figure is created when omitted.
        plot : bool, default=True
            Call ``plt.show()`` (ignored when ``ax`` is supplied).
        save : bool, default=False
            Save the figure to ``<savename>.png``.
        savename : str, default='series_proportions'
            File stem used when ``save`` is True.

        Returns
        -------
        matplotlib.figure.Figure
        """
        self._ensure_analyzed()
        names = list(self.series_names)
        key = "proportion_normalized" if normalized else "proportion"
        values = [self.series_scores[n][key] for n in names]
        colors = [self._colors[n] for n in names]

        created = ax is None
        if created:
            set_biotuner_style()
            _, ax = plt.subplots(figsize=(6, 3))
        fig = ax.figure
        ax.bar(names, values, color=colors)
        ax.set_xlabel("Series")
        ax.set_ylabel("Proportion (normalized)" if normalized else "Proportion")
        ax.set_title("Proportion of matching ratios with biosignal")
        if created:
            fig.tight_layout()
        if save:
            fig.savefig(f"{savename}.png", dpi=150, bbox_inches="tight")
        if plot and created:
            plt.show()
        return fig

    def plot_ratio_pairs(
        self,
        names: Optional[Sequence[str]] = None,
        ax: Optional["plt.Axes"] = None,
        plot: bool = True,
        save: bool = False,
        savename: str = "series_ratio_pairs",
    ) -> "plt.Figure":
        """Log-log scatter of sequence element pairs, matched pairs highlighted.

        Each series' pairs are drawn as small translucent dots; the pairs whose
        folded ratio matches a biosignal ratio are overdrawn large with a dark
        edge (the figure from the notebook).

        Parameters
        ----------
        names : sequence of str, optional
            Series to draw. Defaults to all compared series.
        ax : matplotlib.axes.Axes, optional
            Draw onto an existing axis (for composing multi-panel figures).
            A new figure is created when omitted.
        plot, save, savename
            As in :meth:`plot_proportions`.

        Returns
        -------
        matplotlib.figure.Figure
        """
        self._ensure_analyzed()
        names = list(names) if names else list(self.series_names)

        created = ax is None
        if created:
            set_biotuner_style()
            _, ax = plt.subplots(figsize=(6, 6))
        fig = ax.figure
        for name in names:
            color = self._colors[name]
            pairs = self._series_pairs.get(name, [])
            if pairs:
                xs = [ab[0] for _, ab in pairs]
                ys = [ab[1] for _, ab in pairs]
                ax.scatter(xs, ys, s=8, alpha=0.25, color=color)
            matched = self.series_scores[name]["matched_series_pairs"]
            if matched:
                mxs = [ab[0] for _, ab in matched]
                mys = [ab[1] for _, ab in matched]
                ax.scatter(mxs, mys, s=80, color=color, edgecolor="k", linewidth=0.6, label=name)
            else:
                ax.scatter([], [], s=80, color=color, edgecolor="k", linewidth=0.6, label=name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Element 1")
        ax.set_ylabel("Element 2")
        ax.set_title("Comparing biosignal ratios to mathematical sequences")
        ax.legend(title="Series", loc="best")
        if created:
            fig.tight_layout()
        if save:
            fig.savefig(f"{savename}.png", dpi=150, bbox_inches="tight")
        if plot and created:
            plt.show()
        return fig

    # --------------------------------------------------- creative visualizations
    def _display_series(self, order: Optional[int] = None):
        """Return ``(order, {name: ratios}, {name: matched-ratio set})`` for plots.

        Uses the analysed data when ``order`` matches the instance; otherwise it
        regenerates each series' ratios at the requested ``order`` (and re-derives
        which of them match the signal) purely for display — handy to thin dense
        lattices without changing the instance's matching settings.
        """
        self._ensure_analyzed()
        order = order or self.order
        if order == self.order:
            pairs_by = self._series_pairs
            matched = {n: {round(r, 5) for r, _ in self.series_scores[n]["matched_series_pairs"]}
                       for n in self.series_names}
        else:
            target_keys = {_frac_key(r, self.maxdenom) for r in self.ratios}
            pairs_by, matched = {}, {}
            for n in self.series_names:
                pairs = series_ratio_pairs(n, order, octave=self.octave,
                                           which=self.which, lucas_seed=self.lucas_seed)
                pairs_by[n] = pairs
                matched[n] = {round(r, 5) for r, _ in pairs
                              if _frac_key(r, self.maxdenom) in target_keys}
        ratios_by = {n: np.array(sorted({round(r, 5) for r, _ in pairs_by[n]}))
                     for n in self.series_names}
        return order, ratios_by, matched

    @staticmethod
    def _new_ax(ax, **kw):
        """Return ``(fig, ax, created)``; apply the biotuner style on new figures."""
        if ax is None:
            set_biotuner_style()
            fig, ax = plt.subplots(**kw)
            return fig, ax, True
        return ax.figure, ax, False

    def _finish(self, fig, created, plot, save, savename):
        if save:
            fig.savefig(f"{savename}.png", dpi=150, bbox_inches="tight")
        if plot and created:
            plt.show()
        return fig

    def plot_octave_wheel(self, order: Optional[int] = None, ax: Optional["plt.Axes"] = None,
                          plot: bool = True, save: bool = False,
                          savename: str = "series_octave_wheel") -> "plt.Figure":
        """Octave wrapped to a circle (angle = cents): a ratio-ring per series,
        the signal peaks as spokes, and a filled dot where a spoke matches.

        ``order`` overrides the instance order for display only (thins the rings).
        """
        order, ratios_by, matched = self._display_series(order)
        created = ax is None
        if created:
            set_biotuner_style()
            fig = plt.figure(figsize=(7.5, 7.5))
            ax = fig.add_subplot(projection="polar")
        else:
            fig = ax.figure
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        for name, rr in zip(self.series_names, np.linspace(1.25, 2.0, len(self.series_names))):
            rats = ratios_by[name]
            if rats.size:
                ax.scatter(2 * np.pi * np.log2(rats), [rr] * len(rats), s=20,
                           color=self._colors[name], alpha=0.5, label=name, zorder=2)
            if matched[name]:
                mt = 2 * np.pi * np.log2(sorted(matched[name]))
                ax.scatter(mt, [rr] * len(mt), s=130, color=self._colors[name],
                           edgecolor="k", linewidth=1.1, zorder=4)
        brain = np.asarray(self.ratios, dtype=float)
        for r in brain:
            ax.plot([2 * np.pi * np.log2(r)] * 2, [0.95, 2.08], color="#1F1F1F", lw=1.4, alpha=0.5, zorder=3)
        ax.scatter(2 * np.pi * np.log2(brain), [2.12] * len(brain), marker="v", s=80,
                   color="#1F1F1F", zorder=5, label="signal peaks")
        ax.set_ylim(0, 2.3)
        ax.set_yticklabels([])
        ax.set_xticks(np.linspace(0, 2 * np.pi, 13)[:-1])
        ax.set_xticklabels([str(c) for c in range(0, 1200, 100)])
        ax.set_title("Octave wheel — series rings + signal-peak spokes (filled = matched)")
        ax.legend(loc="center left", bbox_to_anchor=(1.04, 0.5), fontsize=9)
        return self._finish(fig, created, plot, save, savename)

    def plot_cents_ruler(self, order: Optional[int] = None, ax: Optional["plt.Axes"] = None,
                         plot: bool = True, save: bool = False,
                         savename: str = "series_cents_ruler") -> "plt.Figure":
        """Each series as a lane of ratio-ticks on a 0-1200 cents axis (bold =
        matched); guide lines drop from each signal peak."""
        order, ratios_by, matched = self._display_series(order)
        fig, ax, created = self._new_ax(ax, figsize=(12, 4.8))
        names = self.series_names
        brain_c = _cents(self.ratios)
        for c in brain_c:
            ax.axvline(c, color="#1F1F1F", lw=1, alpha=0.18, zorder=1)
        for i, name in enumerate(names):
            y = len(names) - i
            if ratios_by[name].size:
                ax.vlines(_cents(ratios_by[name]), y - 0.32, y + 0.32,
                          color=self._colors[name], lw=1.3, alpha=0.5, zorder=2)
            if matched[name]:
                ax.vlines(_cents(sorted(matched[name])), y - 0.4, y + 0.4,
                          color=self._colors[name], lw=3, zorder=3)
        ax.scatter(brain_c, [0] * len(brain_c), marker="v", s=100, color="#1F1F1F", zorder=4)
        ax.set_yticks(range(len(names) + 1))
        ax.set_yticklabels(["signal peaks"] + names[::-1])
        ax.set_ylim(-0.7, len(names) + 0.7)
        ax.set_xlim(0, 1200)
        ax.set_xlabel("Cents within the octave")
        ax.set_title("Where the signal peaks land among each series' ratio lattice (bold = matched)")
        if created:
            fig.tight_layout()
        return self._finish(fig, created, plot, save, savename)

    def plot_fit_landscape(self, order: Optional[int] = None, ax: Optional["plt.Axes"] = None,
                           plot: bool = True, save: bool = False,
                           savename: str = "series_fit") -> "plt.Figure":
        """For each signal peak, the cents distance to the nearest ratio of each
        series (lower = the series hugs the spectrum more tightly)."""
        order, ratios_by, _ = self._display_series(order)
        fig, ax, created = self._new_ax(ax, figsize=(8.5, 4.8))
        names = self.series_names
        brain_c = _cents(self.ratios)
        jit = np.linspace(-0.18, 0.18, len(brain_c))
        for i, name in enumerate(names):
            sc = _cents(ratios_by[name])
            if sc.size == 0:
                continue
            nearest = np.array([np.min(np.abs(sc - c)) for c in brain_c])
            ax.scatter(np.full(len(nearest), i) + jit, nearest, s=60,
                       color=self._colors[name], edgecolor="k", linewidth=0.5, zorder=3)
            ax.hlines(nearest.mean(), i - 0.3, i + 0.3, color=self._colors[name], lw=3, zorder=2)
            ax.text(i, nearest.mean() + 4, f"{nearest.mean():.0f}c", ha="center", fontsize=10)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_ylabel("Cents to nearest series ratio")
        ax.set_xlabel("Mathematical series")
        ax.set_ylim(bottom=0)
        ax.set_title("How tightly each series hugs the signal's peak ratios (lower = closer)")
        if created:
            fig.tight_layout()
        return self._finish(fig, created, plot, save, savename)

    def plot_simplicity_bubbles(self, order: Optional[int] = None, ax: Optional["plt.Axes"] = None,
                                plot: bool = True, save: bool = False,
                                savename: str = "series_simplicity_bubbles") -> "plt.Figure":
        """Each series ratio as a bubble sized by simplicity (bigger = smaller
        denominator); outlined bubbles are matched. Shows whether the signal
        peaks land on simple rungs."""
        order, ratios_by, matched = self._display_series(order)
        fig, ax, created = self._new_ax(ax, figsize=(12, 4.8))
        names = self.series_names
        brain_c = _cents(self.ratios)
        for c in brain_c:
            ax.axvline(c, color="#1F1F1F", lw=1, ls="--", alpha=0.22, zorder=1)
        for i, name in enumerate(names):
            y = len(names) - i
            for r in ratios_by[name]:
                _, q = ratio2frac(float(r), maxdenom=self.maxdenom)
                hit = round(float(r), 5) in matched[name]
                ax.scatter(float(_cents(r)), y, s=900.0 / max(q, 1), color=self._colors[name],
                           alpha=0.85 if hit else 0.35,
                           edgecolor="k" if hit else "none", linewidth=1.3 if hit else 0,
                           zorder=3 if hit else 2)
        ax.scatter(brain_c, [0] * len(brain_c), marker="v", s=100, color="#1F1F1F", zorder=4)
        ax.set_yticks(range(len(names) + 1))
        ax.set_yticklabels(["signal peaks"] + names[::-1])
        ax.set_ylim(-0.7, len(names) + 0.7)
        ax.set_xlim(0, 1200)
        ax.set_xlabel("Cents within the octave")
        ax.set_title("Series ratios as simplicity bubbles (bigger = simpler); outlined = matched")
        if created:
            fig.tight_layout()
        return self._finish(fig, created, plot, save, savename)

    def plot_series_comb(self, peaks_hz: Optional[Sequence[float]] = None,
                         fmin: float = 2.0, fmax: float = 45.0, order: Optional[int] = None,
                         ax: Optional["plt.Axes"] = None, plot: bool = True, save: bool = False,
                         savename: str = "series_comb") -> "plt.Figure":
        """Each series as an across-octave frequency comb (scaled to best fit the
        signal); the signal peaks (Hz) snap onto the nearest comb step, and each
        lane is labelled with the mean miss in cents.

        Needs peak frequencies: uses :attr:`peaks_hz` (captured from a
        biotuner / HarmonicInput) or an explicit ``peaks_hz``.
        """
        peaks = peaks_hz if peaks_hz is not None else self.peaks_hz
        if peaks is None:
            raise ValueError(
                "plot_series_comb needs peak frequencies in Hz. Pass peaks_hz=..., "
                "or build math_series from a compute_biotuner / HarmonicInput."
            )
        peaks = np.asarray(peaks, dtype=float).ravel()
        peaks = np.array([f for f in peaks if fmin <= f <= fmax])
        order, ratios_by, _ = self._display_series(order)
        names = self.series_names
        fig, ax, created = self._new_ax(ax, figsize=(12, 0.95 * len(names) + 1.6))
        octaves = range(-3, 7)
        for i, name in enumerate(names):
            y = len(names) - i
            grid = _octave_comb_fit(ratios_by[name], peaks, fmin, fmax, octaves)
            for g in grid:
                ax.plot([g, g], [y - 0.42, y + 0.42], color=self._colors[name], lw=1.0, alpha=0.4)
            if grid.size and peaks.size:
                gc = _cents(grid)
                miss = float(np.mean([np.min(np.abs(p - gc)) for p in _cents(peaks)]))
                ax.text(fmax * 1.02, y, f"{miss:.0f}c", va="center", fontsize=9, color="#333")
        ax.scatter(peaks, [0] * len(peaks), marker="v", s=90, color="#1F1F1F", zorder=4)
        ax.set_xscale("log")
        ax.set_xlim(fmin, fmax * 1.18)
        ax.set_xticks([2, 5, 10, 20, 40])
        ax.set_xticklabels(["2", "5", "10", "20", "40"])
        ax.set_yticks(range(len(names) + 1))
        ax.set_yticklabels(["signal peaks"] + names[::-1])
        ax.set_ylim(-0.7, len(names) + 0.7)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title("Each series as an across-octave comb; signal peaks snapped on (label = mean miss)")
        if created:
            fig.tight_layout()
        return self._finish(fig, created, plot, save, savename)

    def __repr__(self) -> str:
        best = self.best_series if self.series_scores else "not analyzed"
        return (
            f"math_series(n_ratios={len(self.ratios)}, "
            f"series={self.series_names}, maxdenom={self.maxdenom}, "
            f"best_series={best!r})"
        )
