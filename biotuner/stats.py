"""
Statistical comparison functions for BiotunerGroup analyses.

Module type: Functions

These functions accept :class:`~biotuner.biotuner_group.BiotunerGroup` objects
(or their ``.summary()`` DataFrames) and perform group-level statistical tests
on harmonicity metrics.

Typical usage
-------------
>>> from biotuner.biotuner_group import BiotunerGroup
>>> from biotuner.stats import compare_all_metrics, plot_stats_comparison
>>>
>>> bt1 = BiotunerGroup(data_rest, sf=1000).compute_peaks().compute_metrics()
>>> bt2 = BiotunerGroup(data_task, sf=1000).compute_peaks().compute_metrics()
>>>
>>> pvals, tstats, direction = compare_all_metrics(bt1, bt2, data_labels=['rest', 'task'])
>>> plot_stats_comparison(pvals, tstats, direction, data_labels=['rest', 'task'])
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import List, Optional, Tuple, Union

from biotuner.biotuner_utils import calculate_pvalues


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_summary(
    group,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Return a numeric-only summary DataFrame from a BiotunerGroup or DataFrame."""
    from biotuner.biotuner_group import BiotunerGroup

    if isinstance(group, BiotunerGroup):
        if group.results is None:
            group.summary()
        df = group.results
    elif isinstance(group, pd.DataFrame):
        df = group
    else:
        raise TypeError("group must be a BiotunerGroup or pandas DataFrame")

    # Keep only numeric metric columns; drop internal index columns
    _index_cols = {'series_idx'}
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in _index_cols
    ]

    if metrics is not None:
        numeric_cols = [c for c in metrics if c in numeric_cols]

    return df[numeric_cols]


# ---------------------------------------------------------------------------
# Core statistical tests
# ---------------------------------------------------------------------------

def ttest_groups(
    group1,
    group2,
    metrics: Optional[List[str]] = None,
    alternative: str = 'two-sided',
) -> pd.DataFrame:
    """Independent t-tests comparing all metrics between two groups.

    Parameters
    ----------
    group1, group2 : BiotunerGroup or pd.DataFrame
        Groups to compare. DataFrames should have metrics as columns.
    metrics : list of str, optional
        Metrics to include. If ``None``, uses all numeric columns present in
        both groups.
    alternative : str, default='two-sided'
        Hypothesis direction: ``'two-sided'``, ``'less'``, or ``'greater'``.

    Returns
    -------
    results : pd.DataFrame
        Indexed by metric name with columns:

        * ``t_stat``      – t-statistic
        * ``p_value``     – two-sided (or directed) p-value
        * ``mean_group1`` – mean of group 1
        * ``mean_group2`` – mean of group 2
        * ``higher_group``– 1 if group1 mean ≥ group2 mean, else 2

    Examples
    --------
    >>> results = ttest_groups(bt_rest, bt_task)
    >>> significant = results[results['p_value'] < 0.05]
    """
    df1 = _get_summary(group1, metrics)
    df2 = _get_summary(group2, metrics)
    common = [c for c in df1.columns if c in df2.columns]

    rows = []
    for col in common:
        a = df1[col].dropna().values
        b = df2[col].dropna().values
        n = min(len(a), len(b))
        if n < 2:
            continue
        t, p = stats.ttest_ind(a[:n], b[:n], alternative=alternative, nan_policy='omit')
        rows.append({
            'metric': col,
            't_stat': t,
            'p_value': p,
            'mean_group1': float(np.nanmean(a)),
            'mean_group2': float(np.nanmean(b)),
            'higher_group': 1 if np.nanmean(a) >= np.nanmean(b) else 2,
        })

    return pd.DataFrame(rows).set_index('metric') if rows else pd.DataFrame()


def ancova_groups(
    group1,
    group2,
    metric: str,
    covariate: str = 'peak_freq_mean',
    data_labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """ANCOVA comparing two groups on a metric, controlling for peak frequency.

    Requires the ``pingouin`` package (``pip install pingouin``).

    Parameters
    ----------
    group1, group2 : BiotunerGroup or pd.DataFrame
        Groups to compare.
    metric : str
        Dependent variable (outcome metric).
    covariate : str, default='peak_freq_mean'
        Covariate column (typically average peak frequency). Must exist in
        both group summaries.
    data_labels : list of str, optional
        Names for the two groups. Defaults to ``['group1', 'group2']``.

    Returns
    -------
    ancova_result : pd.DataFrame
        Output of ``pingouin.ancova`` with F-statistic and p-value.
    """
    try:
        from pingouin import ancova
    except ImportError:
        raise ImportError(
            "The 'pingouin' package is required for ANCOVA.\n"
            "Install it with:  pip install pingouin"
        )

    if data_labels is None:
        data_labels = ['group1', 'group2']

    df1 = _get_summary(group1)
    df2 = _get_summary(group2)

    for col in (metric, covariate):
        if col not in df1.columns:
            raise ValueError(
                f"Column '{col}' not in group1. "
                f"Available columns: {list(df1.columns)}"
            )
        if col not in df2.columns:
            raise ValueError(
                f"Column '{col}' not in group2. "
                f"Available columns: {list(df2.columns)}"
            )

    sub1 = df1[[metric, covariate]].copy()
    sub2 = df2[[metric, covariate]].copy()
    sub1['group'] = data_labels[0]
    sub2['group'] = data_labels[1]
    combined = pd.concat([sub1, sub2], ignore_index=True).fillna(0)

    return ancova(data=combined, dv=metric, covar=covariate, between='group')


def compare_all_metrics(
    group1,
    group2,
    method: str = 'ttest',
    metrics: Optional[List[str]] = None,
    data_labels: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare all available metrics between two groups.

    Runs a statistical test for every numeric metric column that appears in
    both group summaries.

    Parameters
    ----------
    group1, group2 : BiotunerGroup or pd.DataFrame
        Groups to compare.
    method : str, default='ttest'
        Statistical test:

        * ``'ttest'``  – independent samples t-test (no extra dependencies).
        * ``'ancova'`` – ANCOVA with ``peak_freq_mean`` as covariate.
          Requires ``pingouin``; automatically skips the covariate column itself.
    metrics : list of str, optional
        Subset of metrics to test. If ``None``, tests all numeric columns
        present in both summaries.
    data_labels : list of str, optional
        Names for the two groups. Defaults to ``['group1', 'group2']``.

    Returns
    -------
    p_values : pd.DataFrame
        Column ``p_value``, indexed by metric.
    statistics : pd.DataFrame
        Column ``statistic`` (t or F), indexed by metric.
    direction : pd.DataFrame
        Column ``direction``: 1 if group1 mean ≥ group2, 2 otherwise, 0 if
        indeterminate (NaN values or covariate skip).

    Examples
    --------
    >>> pvals, tstats, direction = compare_all_metrics(
    ...     bt_rest, bt_task, method='ttest', data_labels=['rest', 'task']
    ... )
    >>> plot_stats_comparison(pvals, tstats, direction, data_labels=['rest', 'task'])
    """
    if data_labels is None:
        data_labels = ['group1', 'group2']

    df1 = _get_summary(group1, metrics)
    df2 = _get_summary(group2, metrics)
    common = [c for c in df1.columns if c in df2.columns]

    pvals, tstats, directions = {}, {}, {}

    for col in common:
        a = df1[col].dropna().values
        b = df2[col].dropna().values
        n = min(len(a), len(b))

        if n < 2:
            pvals[col] = np.nan
            tstats[col] = np.nan
            directions[col] = 0
            continue

        if method == 'ttest':
            t, p = stats.ttest_ind(a[:n], b[:n], nan_policy='omit')
            pvals[col] = p
            tstats[col] = t

        elif method == 'ancova':
            covariate = 'peak_freq_mean'
            if col == covariate:
                pvals[col] = np.nan
                tstats[col] = np.nan
                directions[col] = 0
                continue
            try:
                result = ancova_groups(
                    group1, group2, col,
                    covariate=covariate,
                    data_labels=data_labels,
                )
                pvals[col] = float(result['p-unc'].iloc[0])
                tstats[col] = float(result['F'].iloc[0])
            except Exception:
                pvals[col] = np.nan
                tstats[col] = np.nan

        else:
            raise ValueError(f"method must be 'ttest' or 'ancova', got '{method}'")

        directions[col] = 1 if np.nanmean(a) >= np.nanmean(b) else 2

    p_df = pd.DataFrame({'p_value': pvals})
    s_df = pd.DataFrame({'statistic': tstats})
    d_df = pd.DataFrame({'direction': directions})
    return p_df, s_df, d_df


def correlate_metrics_peaks(
    bt_group,
    metrics: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Correlate harmonicity metrics with peak frequency within a group.

    Useful for assessing whether observed differences in a metric are
    confounded by differences in peak frequency.

    Parameters
    ----------
    bt_group : BiotunerGroup or pd.DataFrame
        Group with computed peaks and metrics.
    metrics : list of str, optional
        Columns to include. If ``None``, uses all numeric columns.

    Returns
    -------
    corr_df : pd.DataFrame
        Absolute Pearson correlation with peak frequency, column ``correlation``.
    pval_df : pd.DataFrame
        Corresponding p-values, column ``p_value``.

    Raises
    ------
    ValueError
        If no peak-frequency column is found (``peak_freq_mean``, ``peaks``,
        or ``peak_freq``).
    """
    df = _get_summary(bt_group, metrics).fillna(0)

    corr_matrix = df.corr()
    pval_matrix = calculate_pvalues(df)

    _peak_col_candidates = ('peak_freq_mean', 'peaks', 'peak_freq')
    peak_col = next(
        (c for c in _peak_col_candidates if c in corr_matrix.columns),
        None,
    )
    if peak_col is None:
        raise ValueError(
            "No peak frequency column found in summary. "
            "Run compute_peaks() and summary() first. "
            f"Looked for: {_peak_col_candidates}"
        )

    corr_df = (
        corr_matrix[[peak_col]]
        .abs()
        .rename(columns={peak_col: 'correlation'})
        .drop(index=peak_col, errors='ignore')
    )
    pval_df = (
        pval_matrix[[peak_col]]
        .abs()
        .rename(columns={peak_col: 'p_value'})
        .drop(index=peak_col, errors='ignore')
    )
    return corr_df, pval_df


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_stats_comparison(
    p_values: pd.DataFrame,
    statistics: Optional[pd.DataFrame] = None,
    direction: Optional[pd.DataFrame] = None,
    data_labels: Optional[List[str]] = None,
    method_name: str = '',
    figsize: Tuple[int, int] = (14, 7),
    save_path: Optional[str] = None,
    show: bool = True,
) -> plt.Figure:
    """Plot statistical comparison results as a line chart with significance markers.

    Each metric is shown on the x-axis; the y-axis shows the p-value.
    A dashed red line marks p = 0.05. Significant metrics are annotated with
    triangular markers indicating which group had the higher mean.

    Parameters
    ----------
    p_values : pd.DataFrame
        p-values indexed by metric name (output of :func:`compare_all_metrics`
        or :func:`ttest_groups`). Column ``p_value`` or first column is used.
    statistics : pd.DataFrame, optional
        Test statistics (t or F) indexed by metric. Currently unused in the
        plot but kept for API consistency.
    direction : pd.DataFrame, optional
        Direction DataFrame from :func:`compare_all_metrics`
        (column ``direction``: 1=group1 higher, 2=group2 higher).
    data_labels : list of str, optional
        Group names for the legend. Defaults to ``['Group 1', 'Group 2']``.
    method_name : str, default=''
        Method name appended to the title.
    figsize : tuple, default=(14, 7)
        Figure size.
    save_path : str, optional
        If provided, save figure to this path (300 dpi).
    show : bool, default=True
        Call ``plt.show()`` at the end.

    Returns
    -------
    fig : matplotlib.Figure
    """
    if data_labels is None:
        data_labels = ['Group 1', 'Group 2']

    pvals = (
        p_values['p_value']
        if 'p_value' in p_values.columns
        else p_values.iloc[:, 0]
    )
    metrics = list(pvals.index)
    pval_arr = pvals.values

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(range(len(metrics)), pval_arr, color='steelblue',
            linewidth=2.5, label=method_name or 'p-values', marker='o', markersize=5)
    ax.axhline(y=0.05, color='crimson', linestyle='--', linewidth=1.5, label='p = 0.05')

    if direction is not None:
        dir_col = direction['direction'] if 'direction' in direction.columns else direction.iloc[:, 0]
        dir_vals = dir_col.reindex(metrics).values
        _colors = {1: 'darkred', 2: 'darkorange'}
        _labels_used = {1: False, 2: False}
        for i, (p, d) in enumerate(zip(pval_arr, dir_vals)):
            if not np.isnan(p) and p < 0.05 and d in (1, 2):
                label = None
                if not _labels_used[d]:
                    label = f'{data_labels[d - 1]} higher'
                    _labels_used[d] = True
                ax.scatter(i, 0, color=_colors[d], marker='^', s=400,
                           zorder=5, label=label)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=30, ha='right', fontsize=12)
    ax.set_ylabel('p-value', fontsize=13)
    ax.set_xlabel('Metric', fontsize=13)

    title = f'Statistical comparison — {data_labels[0]} vs. {data_labels[1]}'
    if method_name:
        title += f'\n({method_name})'
    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right', framealpha=0.9)
    ax.grid(alpha=0.25, linewidth=0.8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=300, facecolor='white')
    if show:
        plt.show()
    return fig
