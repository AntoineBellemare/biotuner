"""
Surrogate signal generation and comparison for BiotunerGroup analysis.

Module type: Functions

Surrogates are signal controls that preserve specific statistical properties
(spectrum, amplitude distribution) while destroying others (phase relationships,
nonlinear structure). Comparing real vs. surrogate BiotunerGroup metrics lets
you assess whether observed harmonicity is above chance.

Typical usage
-------------
>>> from biotuner.biotuner_group import BiotunerGroup
>>> from biotuner.surrogates import surrogate_group, plot_surrogate_distributions
>>>
>>> bt = BiotunerGroup(data, sf=1000)
>>> bt.compute_peaks(peaks_function='EMD').compute_metrics()
>>>
>>> surr = surrogate_group(bt, surr_type='AAFT')
>>> surr_pink = surrogate_group(bt, surr_type='pink')
>>>
>>> plot_surrogate_distributions(bt, {'AAFT': surr, 'pink': surr_pink}, metric='harmsim')
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Optional

from biotuner.biotuner_utils import (
    AAFT_surrogates,
    butter_bandpass_filter,
    UnivariateSurrogatesTFT,
    phaseScrambleTS,
)

SURROGATE_TYPES = ('AAFT', 'TFT', 'phase', 'shuffle', 'white', 'pink', 'brown', 'blue')


# ---------------------------------------------------------------------------
# Low-level signal generation
# ---------------------------------------------------------------------------

def generate_surrogate(
    data: np.ndarray,
    surr_type: str = 'pink',
    low_cut: float = 0.5,
    high_cut: float = 150.0,
    sf: int = 1000,
    TFT_freq: int = 5,
) -> np.ndarray:
    """Generate a surrogate signal from a 1D time series.

    Parameters
    ----------
    data : ndarray, shape (n_samples,)
        Original 1D signal.
    surr_type : str, default='pink'
        Surrogate type:

        * ``'AAFT'``    – Amplitude-Adjusted Fourier Transform surrogate.
          Preserves amplitude distribution and power spectrum.
        * ``'TFT'``     – Truncated Fourier Transform surrogate.
          Preserves low-frequency structure below *TFT_freq* Hz.
        * ``'phase'``   – Phase-scrambled surrogate.
          Preserves power spectrum, destroys phase relationships.
        * ``'shuffle'`` – Randomly shuffled samples.
          Destroys all temporal structure.
        * ``'white'``   – White noise (flat spectrum, β=0).
        * ``'pink'``    – Pink noise (1/f spectrum, β=1).
        * ``'brown'``   – Brown noise (1/f² spectrum, β=2).
        * ``'blue'``    – Blue noise (f spectrum, β=−1).
    low_cut : float, default=0.5
        High-pass cutoff frequency (Hz). Applied after generation for all
        types except TFT.
    high_cut : float, default=150.0
        Low-pass cutoff frequency (Hz).
    sf : int, default=1000
        Sampling frequency (Hz).
    TFT_freq : int, default=5
        Corner frequency for TFT surrogates.

    Returns
    -------
    surrogate : ndarray, shape (n_samples,)
        Surrogate signal matching the length of *data*.
    """
    if surr_type not in SURROGATE_TYPES:
        raise ValueError(f"surr_type must be one of {SURROGATE_TYPES}, got '{surr_type}'")

    if surr_type == 'AAFT':
        indexes = np.arange(len(data))
        data_ = AAFT_surrogates(np.stack((data, indexes)))[0]
        return butter_bandpass_filter(data_, low_cut, high_cut, sf, 4)

    if surr_type == 'TFT':
        return UnivariateSurrogatesTFT(data, 1, fc=TFT_freq)

    if surr_type == 'phase':
        scrambled = phaseScrambleTS(data)[:len(data)]
        return butter_bandpass_filter(scrambled, low_cut, high_cut, sf, 4)

    if surr_type == 'shuffle':
        data_ = data.copy()
        np.random.shuffle(data_)
        return butter_bandpass_filter(data_, low_cut, high_cut, sf, 4)

    # Colored noise
    beta_map = {'white': 0, 'pink': 1, 'brown': 2, 'blue': -1}
    try:
        import colorednoise as cn
    except ImportError:
        raise ImportError(
            "The 'colorednoise' package is required for colored noise surrogates.\n"
            "Install it with:  pip install colorednoise"
        )
    data_ = cn.powerlaw_psd_gaussian(beta_map[surr_type], len(data))
    return butter_bandpass_filter(data_, low_cut, high_cut, sf, 4)


def generate_surrogate_data(
    data: np.ndarray,
    surr_type: str = 'pink',
    low_cut: float = 0.5,
    high_cut: float = 150.0,
    sf: int = 1000,
    **kwargs,
) -> np.ndarray:
    """Generate surrogate signals for a 2D or 3D array of time series.

    Parameters
    ----------
    data : ndarray, shape (n, n_samples) or (n, m, n_samples)
        Original signals. Each 1D slice is treated independently.
    surr_type : str, default='pink'
        Surrogate type. See :func:`generate_surrogate`.
    low_cut, high_cut : float
        Bandpass filter cutoffs (Hz).
    sf : int, default=1000
        Sampling frequency (Hz).
    **kwargs :
        Extra keyword arguments forwarded to :func:`generate_surrogate`
        (e.g. ``TFT_freq``).

    Returns
    -------
    surrogate_data : ndarray
        Same shape as *data*.
    """
    if data.ndim not in (2, 3):
        raise ValueError("data must be 2D or 3D")

    data_ = data.copy()
    if data.ndim == 2:
        for i in range(data.shape[0]):
            data_[i] = generate_surrogate(data[i], surr_type, low_cut, high_cut, sf, **kwargs)
    else:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data_[i, j] = generate_surrogate(
                    data[i, j], surr_type, low_cut, high_cut, sf, **kwargs
                )
    return data_


# ---------------------------------------------------------------------------
# BiotunerGroup-level surrogate creation
# ---------------------------------------------------------------------------

def surrogate_group(
    bt_group,
    surr_type: str = 'AAFT',
    low_cut: float = 0.5,
    high_cut: float = 150.0,
    recompute: bool = True,
    n_jobs: int = 1,
    **peaks_kwargs,
):
    """Create a surrogate :class:`~biotuner.biotuner_group.BiotunerGroup`.

    Generates surrogate signals for every time series in *bt_group*, then
    mirrors the same analysis pipeline (peaks, metrics, dissonance curve, …)
    that was already run on the original group.

    Parameters
    ----------
    bt_group : BiotunerGroup
        Reference group. Shape and sampling frequency are preserved.
    surr_type : str, default='AAFT'
        Surrogate type. See :func:`generate_surrogate`.
    low_cut, high_cut : float
        Bandpass filter cutoffs (Hz) applied during surrogate generation.
    recompute : bool, default=True
        If ``True``, automatically re-run the same analysis steps that were
        already computed on *bt_group* (peaks, metrics, diss_curve, HE).
    n_jobs : int, default=1
        Parallel jobs forwarded to the BiotunerGroup pipeline.
    **peaks_kwargs :
        Extra keyword arguments forwarded to
        :meth:`~biotuner.biotuner_group.BiotunerGroup.compute_peaks`
        (e.g. ``min_freq=1``, ``max_freq=60``).

    Returns
    -------
    surr_group : BiotunerGroup
        New BiotunerGroup backed by surrogate data, with the same pipeline
        state as *bt_group* (if *recompute* is ``True``).

    Examples
    --------
    >>> surr = surrogate_group(bt, surr_type='AAFT', min_freq=1, max_freq=60)
    >>> comparison = plot_surrogate_distributions(bt, {'AAFT': surr}, metric='harmsim')
    """
    from biotuner.biotuner_group import BiotunerGroup

    surr_data = generate_surrogate_data(
        bt_group.data, surr_type, low_cut, high_cut, bt_group.sf
    )

    surr_group = BiotunerGroup(
        surr_data,
        sf=bt_group.sf,
        axis_labels=bt_group.axis_labels,
        store_objects=bt_group.store_objects,
        **bt_group.biotuner_kwargs,
    )

    if recompute:
        computed = bt_group._computed_methods
        if 'peaks_extraction' in computed:
            surr_group.compute_peaks(n_jobs=n_jobs, **peaks_kwargs)
        if 'compute_peaks_metrics' in computed:
            surr_group.compute_metrics(n_jobs=n_jobs)
        if 'compute_diss_curve' in computed:
            surr_group.compute_diss_curve(n_jobs=n_jobs)
        if 'compute_harmonic_entropy' in computed:
            surr_group.compute_harmonic_entropy()

    return surr_group


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_surrogate_distributions(
    real_group,
    surrogate_groups: Dict[str, object],
    metric: str = 'harmsim',
    colors: Optional[List[str]] = None,
    figsize: tuple = (11, 6),
    title: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot metric distributions for real data vs. surrogate conditions.

    Computes an independent t-test between the real group and each surrogate
    group, and annotates significant differences (p < 0.05) with an asterisk
    in the legend.

    Parameters
    ----------
    real_group : BiotunerGroup
        Original data group (analysis pipeline already computed).
    surrogate_groups : dict
        ``{label: BiotunerGroup}`` for each surrogate condition.
    metric : str, default='harmsim'
        Metric column to plot (must exist in every group's summary).
    colors : list of str, optional
        One color per condition (real first, then surrogates in dict order).
        Defaults to a preset palette.
    figsize : tuple, default=(11, 6)
        Figure size.
    title : str, optional
        Custom figure title. Defaults to a descriptive auto-title.
    show : bool, default=True
        Call ``plt.show()`` at the end.

    Returns
    -------
    fig : matplotlib.Figure

    Raises
    ------
    ValueError
        If *metric* is not found in any group's summary.
    """
    _default_colors = ['cyan', 'deeppink', 'gold', 'limegreen', 'orange', 'violet', 'red']
    if colors is None:
        colors = _default_colors

    all_groups: Dict[str, object] = {'real': real_group, **surrogate_groups}
    data_map: Dict[str, np.ndarray] = {}

    for label, grp in all_groups.items():
        if grp.results is None:
            grp.summary()
        if metric not in grp.results.columns:
            raise ValueError(
                f"Metric '{metric}' not found in group '{label}'. "
                f"Available: {list(grp.results.select_dtypes(include=[float, int]).columns)}"
            )
        data_map[label] = grp.results[metric].dropna().values

    real_vals = data_map['real']

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')

    legend_labels: List[str] = []
    for (label, vals), color in zip(data_map.items(), colors):
        sns.kdeplot(vals, ax=ax, color=color, linewidth=2.5)
        if label != 'real':
            n = min(len(vals), len(real_vals))
            _, p = stats.ttest_ind(real_vals[:n], vals[:n], nan_policy='omit')
            legend_labels.append(f'{label} *' if p < 0.05 else label)
        else:
            legend_labels.append(label)

    ax.legend(legend_labels, fontsize=12, framealpha=0.85,
              facecolor='#2d2d4e', labelcolor='white')
    ax.set_xlabel(metric, fontsize=13, color='white')
    ax.set_ylabel('Density', fontsize=13, color='white')
    ax.grid(color='white', linestyle='-.', linewidth=0.5, alpha=0.3)
    ax.tick_params(colors='white')

    if title is None:
        title = f'Real vs. surrogate distributions — {metric}'
    ax.set_title(title, fontsize=15, color='white', fontweight='bold')

    plt.tight_layout()
    if show:
        plt.show()
    return fig
