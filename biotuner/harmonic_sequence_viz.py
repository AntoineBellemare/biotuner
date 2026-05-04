"""
biotuner/harmonic_sequence_viz.py
===================================
Visualization utilities for the ``harmonic_sequence`` module.

Module type: Functions

Every function accepts an optional ``ax`` argument so panels can be embedded
in larger figure layouts.  When ``ax`` is ``None`` a new figure is created.

All multi-panel overview / comparison helpers save their output to *figdir*
(default ``"hs_figures"``).

Quick reference
---------------
Per-model panels:
    plot_histogram_heatmap     – cents distribution heatmap over time
    plot_wasserstein_flux      – consecutive W1 distances
    plot_wasserstein_matrix    – symmetric pairwise W1 matrix
    plot_markov_matrix         – annotated Markov transition matrix
    plot_latent_trajectory     – 2-D PCA path coloured by time
    plot_dmd_spectrum          – eigenvalues on the complex plane
    plot_topology_barcode      – persistence barcode (H0, H1)
    plot_grammar_interval_heatmap – JI interval presence timeline

Composite figures:
    plot_scenario_overview     – 6-panel overview for one scenario
    plot_comparison_flux       – Wasserstein flux for all scenarios
    plot_comparison_latent     – latent trajectories for all scenarios
    plot_comparison_summary    – 4-panel bar-chart summary across scenarios
"""
from __future__ import annotations

import os
import warnings
from typing import List, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

# ── style tweaks (non-intrusive) ─────────────────────────────────────────────
_STYLE = {
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 120,
}

SCENARIO_COLORS = [
    "#3498db",  # blue   – stable
    "#e67e22",  # orange – drift
    "#2ecc71",  # green  – periodic
    "#e74c3c",  # red    – transition
    "#9b59b6",  # purple – random walk
]

# JI reference positions in cents (shown as faint guides on heatmaps)
_JI_GUIDE_CENTS = [0, 112, 204, 316, 386, 498, 590, 702, 814, 884, 996, 1088, 1200]


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _req(model, name: str):
    """Raise a clear error if a model was not fitted."""
    if model is None:
        raise RuntimeError(
            f"'{name}' model is not fitted. "
            f"Call analyzer.fit_{name.lower().replace(' ', '_')}() first."
        )
    return model


def _save(fig: plt.Figure, path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved → {path}")


def _colorbar(mappable, ax: plt.Axes, label: str = "") -> None:
    """Add a compact colorbar."""
    fig = ax.get_figure()
    cb = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    if label:
        cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Cents histogram heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_histogram_heatmap(
    analyzer,
    ax: Optional[plt.Axes] = None,
    title: str = "Harmonic Interval Distribution",
    cmap: str = "magma",
    vmax: Optional[float] = None,
    percentile_clip: float = 98.0,
    gamma: float = 0.5,
) -> plt.Axes:
    """Heatmap of normalised cents histograms.

    X-axis = time step, Y-axis = cents [0, 1200].
    Colour encodes density (brighter = more energy).

    Parameters
    ----------
    analyzer : HarmonicSequenceAnalyzer
    ax : Axes, optional
    title : str
    cmap : str, default ``'magma'``
    vmax : float, optional
        Explicit colour-scale ceiling.  When ``None`` (default), ``vmax`` is
        set to the ``percentile_clip``-th percentile of non-zero histogram
        values to avoid a few bright peaks washing out the rest.
    percentile_clip : float, default 98.0
        Percentile (of non-zero values) used to compute ``vmax`` automatically.
    gamma : float, default 0.5
        Power-norm exponent applied before mapping to colours.  Values < 1
        brighten low-density regions (square-root at 0.5).  Set to 1.0 to
        disable.
    """
    from matplotlib.colors import PowerNorm

    X = analyzer.histograms          # (T, n_bins)
    T, n_bins = X.shape
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3))

    nonzero = X[X > 0]
    if vmax is None:
        vmax_ = float(np.percentile(nonzero, percentile_clip)) if len(nonzero) else 1e-9
        vmax_ = max(vmax_, 1e-9)
    else:
        vmax_ = vmax

    norm = PowerNorm(gamma=gamma, vmin=0, vmax=vmax_) if gamma != 1.0 else None

    im = ax.imshow(
        X.T,
        aspect="auto",
        origin="lower",
        extent=[-0.5, T - 0.5, 0, 1200],
        cmap=cmap,
        norm=norm,
        vmin=0 if norm is None else None,
        vmax=vmax_ if norm is None else None,
        interpolation="nearest",
    )
    for c in _JI_GUIDE_CENTS:
        ax.axhline(c, color="white", alpha=0.25, lw=0.7)
    ax.set_xlabel("Time step", fontsize=9)
    ax.set_ylabel("Cents", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
    _colorbar(im, ax, "Density")
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Wasserstein flux
# ─────────────────────────────────────────────────────────────────────────────

def plot_wasserstein_flux(
    analyzer,
    ax: Optional[plt.Axes] = None,
    title: str = "Wasserstein Flux",
    color: str = "#3498db",
    fill: bool = True,
) -> plt.Axes:
    """Line plot of consecutive W₁ distances (harmonic velocity).

    Parameters
    ----------
    analyzer : HarmonicSequenceAnalyzer
    ax : Axes, optional
    title : str
    color : str
    fill : bool  shade area under the flux curve
    """
    wt = _req(analyzer.wasserstein, "wasserstein")
    flux = wt.flux_
    t = np.arange(len(flux))
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3))

    ax.plot(t, flux, "-o", color=color, ms=4, lw=1.8, zorder=3)
    if fill:
        ax.fill_between(t, flux, alpha=0.18, color=color)
    mean_v = float(np.mean(flux))
    ax.axhline(mean_v, color=color, ls="--", lw=1.2, alpha=0.55,
               label=f"Mean = {mean_v:.2f}")
    ax.set_xlabel("Time step", fontsize=9)
    ax.set_ylabel("W₁ distance", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Wasserstein pairwise distance matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_wasserstein_matrix(
    analyzer,
    ax: Optional[plt.Axes] = None,
    title: str = "Pairwise W₁ Distance",
    cmap: str = "YlOrRd",
) -> plt.Axes:
    """Symmetric pairwise Wasserstein distance matrix.

    Parameters
    ----------
    analyzer : HarmonicSequenceAnalyzer
    ax : Axes, optional
    title : str
    cmap : str
    """
    wt = _req(analyzer.wasserstein, "wasserstein")
    D = wt.distance_matrix_
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(D, cmap=cmap, aspect="equal", origin="upper")
    ax.set_xlabel("Time step", fontsize=9)
    ax.set_ylabel("Time step", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
    _colorbar(im, ax, "W₁")
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Markov transition matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_markov_matrix(
    analyzer,
    ax: Optional[plt.Axes] = None,
    title: str = "Markov Transition Matrix",
    cmap: str = "Blues",
    annotate: bool = True,
) -> plt.Axes:
    """Annotated heatmap of the Markov transition probability matrix.

    Parameters
    ----------
    analyzer : HarmonicSequenceAnalyzer
    ax : Axes, optional
    title : str
    cmap : str
    annotate : bool  write probability values inside each cell
    """
    mk = _req(analyzer.markov, "markov")
    T = mk.transition_matrix_
    n = T.shape[0]
    if ax is None:
        _, ax = plt.subplots(figsize=(n + 1, n + 1))

    im = ax.imshow(T, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    if annotate:
        for i in range(n):
            for j in range(n):
                v = T[i, j]
                txt_color = "white" if v > 0.55 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color=txt_color, fontweight="bold")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"s{k}" for k in range(n)], fontsize=8)
    ax.set_yticklabels([f"s{k}" for k in range(n)], fontsize=8)
    ax.set_xlabel("To state", fontsize=9)
    ax.set_ylabel("From state", fontsize=9)
    ax.set_title(title, fontsize=10)
    _colorbar(im, ax, "Prob.")
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Latent PCA trajectory
# ─────────────────────────────────────────────────────────────────────────────

def plot_latent_trajectory(
    analyzer,
    dims: Tuple[int, int] = (0, 1),
    ax: Optional[plt.Axes] = None,
    title: str = "Latent Harmonic Trajectory",
    cmap: str = "plasma",
    arrow_every: int = 4,
) -> plt.Axes:
    """Scatter of latent PCA coordinates, coloured by time step.

    Arrows are drawn every *arrow_every* steps to indicate direction of travel.

    Parameters
    ----------
    analyzer : HarmonicSequenceAnalyzer
    dims : (int, int)  which PC pair to plot
    ax : Axes, optional
    title : str
    cmap : str
    arrow_every : int
    """
    ls = _req(analyzer.latent, "latent")
    Z = ls.trajectory()
    T_steps = Z.shape[0]
    d0, d1 = dims
    evr = ls.explained_variance_ratio_

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    ax.plot(Z[:, d0], Z[:, d1], "-", color="gray", alpha=0.25, lw=1.0, zorder=1)
    sc = ax.scatter(
        Z[:, d0], Z[:, d1],
        c=np.arange(T_steps), cmap=cmap, s=60,
        zorder=3, edgecolors="white", linewidths=0.4,
    )
    ax.scatter(Z[0, d0], Z[0, d1], marker="*", s=220, color="lime",
               zorder=5, label="Start", edgecolors="black", linewidths=0.5)
    ax.scatter(Z[-1, d0], Z[-1, d1], marker="D", s=100, color="tomato",
               zorder=5, label="End", edgecolors="black", linewidths=0.5)

    for i in range(0, T_steps - 1, arrow_every):
        dx = Z[i + 1, d0] - Z[i, d0]
        dy = Z[i + 1, d1] - Z[i, d1]
        ax.annotate("", xy=(Z[i + 1, d0], Z[i + 1, d1]),
                    xytext=(Z[i, d0], Z[i, d1]),
                    arrowprops=dict(arrowstyle="-|>", color="gray",
                                   alpha=0.55, lw=0.8))

    ev0 = evr[d0] if len(evr) > d0 else 0.0
    ev1 = evr[d1] if len(evr) > d1 else 0.0
    ax.set_xlabel(f"PC{d0 + 1} ({ev0:.1%})", fontsize=9)
    ax.set_ylabel(f"PC{d1 + 1} ({ev1:.1%})", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    _colorbar(sc, ax, "Time step")
    ax.tick_params(labelsize=8)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# 6.  DMD eigenvalue spectrum
# ─────────────────────────────────────────────────────────────────────────────

def plot_dmd_spectrum(
    analyzer,
    ax: Optional[plt.Axes] = None,
    title: str = "DMD Eigenvalue Spectrum",
    annotate_osc: bool = True,
) -> plt.Axes:
    """Eigenvalues plotted on the complex plane with the unit circle.

    Colour encodes |λ| (green = near unit circle = sustained oscillation).

    Parameters
    ----------
    analyzer : HarmonicSequenceAnalyzer
    ax : Axes, optional
    title : str
    annotate_osc : bool  label oscillatory modes with their period
    """
    dmd = _req(analyzer.dmd, "dmd")
    eig = dmd.eigenvalues_
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    # Unit circle
    th = np.linspace(0, 2 * np.pi, 300)
    ax.plot(np.cos(th), np.sin(th), "k--", alpha=0.25, lw=1.0)
    ax.axhline(0, color="gray", lw=0.4, alpha=0.3)
    ax.axvline(0, color="gray", lw=0.4, alpha=0.3)

    mods = np.abs(eig)
    vmax = max(float(mods.max()), 1.5)
    sc = ax.scatter(
        eig.real, eig.imag,
        c=mods, cmap="RdYlGn",
        s=80, vmin=0, vmax=vmax,
        edgecolors="black", linewidths=0.5, zorder=4,
    )

    if annotate_osc:
        osc_eig, osc_idx = dmd.oscillatory_modes(threshold=0.1)
        for k, lam in zip(osc_idx, osc_eig):
            freq = float(np.abs(np.imag(np.log(lam + 1e-15)))) / (2 * np.pi)
            if freq > 1e-3:
                period = 1.0 / freq
                ax.annotate(f"T={period:.1f}", xy=(lam.real, lam.imag),
                            fontsize=7, color="navy",
                            xytext=(lam.real + 0.05, lam.imag + 0.05))

    _colorbar(sc, ax, "|λ|")
    ax.set_xlabel("Re(λ)", fontsize=9)
    ax.set_ylabel("Im(λ)", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=8)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Persistence barcode
# ─────────────────────────────────────────────────────────────────────────────

def plot_topology_barcode(
    analyzer,
    ax: Optional[plt.Axes] = None,
    title: str = "Persistence Barcode",
    max_bars: int = 20,
    dim_colors: Tuple[str, str] = ("#3498db", "#e74c3c"),
) -> plt.Axes:
    """Horizontal persistence barcode for H0 (and H1 if available).

    Parameters
    ----------
    analyzer : HarmonicSequenceAnalyzer
    ax : Axes, optional
    title : str
    max_bars : int  maximum bars per homological dimension
    dim_colors : tuple of 2 str  colours for H0 and H1
    """
    topo = _req(analyzer.topology, "topology")
    dgms = topo.persistence_diagram_
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    # Compute a sensible upper bound for infinite bars
    all_finite: List[float] = []
    for dgm in dgms:
        fin = dgm[np.isfinite(dgm[:, 1]), 1]
        all_finite.extend(fin.tolist())
    inf_cap = (max(all_finite) * 1.25) if all_finite else 1.0

    y_offset = 0
    legend_handles = []
    for dim, (dgm, color) in enumerate(zip(dgms[:2], dim_colors)):
        pers = dgm[:, 1] - dgm[:, 0]
        pers_disp = np.where(np.isfinite(pers), pers, inf_cap)
        order = np.argsort(pers_disp)[::-1][:max_bars]
        for rank, idx in enumerate(order):
            b = float(dgm[idx, 0])
            d = min(float(dgm[idx, 1]), inf_cap) if np.isfinite(dgm[idx, 1]) else inf_cap
            ax.hlines(y_offset + rank, b, d, color=color, lw=2.5, alpha=0.85)
            if not np.isfinite(dgm[idx, 1]):
                ax.annotate("∞", xy=(d, y_offset + rank),
                            fontsize=8, va="center", color=color)
        legend_handles.append(Line2D([0], [0], color=color, lw=2.5, label=f"H{dim}"))
        y_offset += len(order) + 1

    ax.axvline(inf_cap / 1.25, color="gray", ls=":", lw=0.8, alpha=0.5,
               label="max finite")
    ax.set_xlabel("Filtration value", fontsize=9)
    ax.set_ylabel("Feature index", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.legend(handles=legend_handles, fontsize=8)
    ax.tick_params(labelsize=8)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# 8.  JI interval grammar heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_grammar_interval_heatmap(
    analyzer,
    ax: Optional[plt.Axes] = None,
    title: str = "JI Interval Presence over Time",
    top_n: int = 15,
    cmap: str = "Blues",
) -> plt.Axes:
    """Binary heatmap of JI interval presence across time.

    The top *top_n* most frequently occurring named intervals are shown.

    Parameters
    ----------
    analyzer : HarmonicSequenceAnalyzer
    ax : Axes, optional
    title : str
    top_n : int
    cmap : str
    """
    from biotuner.harmonic_sequence import encode_ji_matrix
    _req(analyzer.grammar, "grammar")
    X_ji, labels = encode_ji_matrix(
        analyzer.ratios_list, tolerance_cents=analyzer.tolerance_cents
    )
    freq = X_ji.sum(axis=0)
    top_idx = np.argsort(freq)[::-1][:top_n]
    X_top = X_ji[:, top_idx]
    labels_top = [labels[i] for i in top_idx]

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 0.38 * top_n + 1.5))

    im = ax.imshow(
        X_top.T, aspect="auto", origin="lower",
        cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
    )
    ax.set_xlabel("Time step", fontsize=9)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels_top, fontsize=7)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Per-scenario 6-panel overview figure
# ─────────────────────────────────────────────────────────────────────────────

def plot_scenario_overview(
    analyzer,
    scenario_name: str,
    scenario_color: str = "#3498db",
    figdir: str = "hs_figures",
    save: bool = True,
) -> plt.Figure:
    """Six-panel overview figure for one scenario.

    Panels (3 rows × 3 cols, row 0 spans all columns):
        [0, :]  cents histogram heatmap  (full width)
        [1, 0]  Wasserstein flux
        [1, 1]  Wasserstein distance matrix
        [1, 2]  Markov transition matrix
        [2, 0]  latent PCA trajectory
        [2, 1]  DMD eigenvalue spectrum
        [2, 2]  (empty)

    Parameters
    ----------
    analyzer : HarmonicSequenceAnalyzer
    scenario_name : str  used in the title and filename
    scenario_color : str  accent colour for flux / latent plots
    figdir : str  output directory
    save : bool  write the figure to *figdir*

    Returns
    -------
    fig : matplotlib Figure
    """
    with plt.rc_context(_STYLE):
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(
            3, 3, figure=fig, hspace=0.52, wspace=0.42,
            height_ratios=[1.1, 1, 1],
        )
        ax0 = fig.add_subplot(gs[0, :])
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[1, 1])
        ax3 = fig.add_subplot(gs[1, 2])
        ax4 = fig.add_subplot(gs[2, 0])
        ax5 = fig.add_subplot(gs[2, 1])
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis("off")   # reserved / empty

        plot_histogram_heatmap(analyzer, ax=ax0,
                               title="Harmonic Interval Distribution over Time")

        if analyzer.wasserstein is not None:
            plot_wasserstein_flux(analyzer, ax=ax1, color=scenario_color,
                                  title="Wasserstein Flux (Harmonic Velocity)")
            plot_wasserstein_matrix(analyzer, ax=ax2,
                                    title="Pairwise W₁ Distance Matrix")
        else:
            for ax in (ax1, ax2):
                ax.text(0.5, 0.5, "Wasserstein not fitted",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=10, color="gray")

        if analyzer.markov is not None:
            plot_markov_matrix(analyzer, ax=ax3,
                               title="Markov Transition Matrix")
        else:
            ax3.text(0.5, 0.5, "Markov not fitted",
                     transform=ax3.transAxes, ha="center", va="center",
                     fontsize=10, color="gray")

        if analyzer.latent is not None:
            plot_latent_trajectory(analyzer, ax=ax4,
                                   title="Latent PCA Trajectory")
        else:
            ax4.text(0.5, 0.5, "Latent space not fitted",
                     transform=ax4.transAxes, ha="center", va="center",
                     fontsize=10, color="gray")

        if analyzer.dmd is not None:
            plot_dmd_spectrum(analyzer, ax=ax5, title="DMD Eigenvalue Spectrum")
        else:
            ax5.text(0.5, 0.5, "DMD not fitted",
                     transform=ax5.transAxes, ha="center", va="center",
                     fontsize=10, color="gray")

        fig.suptitle(
            f"Harmonic Sequence Analysis  ·  Scenario {scenario_name}",
            fontsize=13, fontweight="bold", y=1.005,
        )

    if save:
        path = os.path.join(figdir, f"fig_scenario_{scenario_name}_overview.pdf")
        _save(fig, path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Cross-scenario Wasserstein flux comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_flux(
    analyzers: List,
    names: List[str],
    colors: Optional[List[str]] = None,
    figdir: str = "hs_figures",
    save: bool = True,
) -> plt.Figure:
    """Subplot grid showing Wasserstein flux for each scenario.

    Parameters
    ----------
    analyzers : list of HarmonicSequenceAnalyzer
    names : list of str  scenario labels
    colors : list of str, optional
    figdir : str
    save : bool

    Returns
    -------
    fig : Figure
    """
    if colors is None:
        colors = SCENARIO_COLORS[: len(analyzers)]
    n = len(analyzers)
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5), sharey=False)
        if n == 1:
            axes = [axes]
        for ax, ana, name, col in zip(axes, analyzers, names, colors):
            if ana.wasserstein is not None:
                flux = ana.wasserstein.flux_
                t = np.arange(len(flux))
                ax.plot(t, flux, "-o", color=col, ms=3.5, lw=1.8)
                ax.fill_between(t, flux, alpha=0.18, color=col)
                ax.axhline(np.mean(flux), color=col, ls="--", lw=1.0, alpha=0.6)
                ax.set_title(name, fontsize=10, fontweight="bold", color=col)
                ax.set_xlabel("Step", fontsize=8)
                ax.set_ylabel("W₁" if ax is axes[0] else "", fontsize=8)
            else:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                        ha="center", va="center", color="gray")
                ax.set_title(name, fontsize=10, color="gray")
            ax.tick_params(labelsize=8)
        fig.suptitle("Wasserstein Flux Comparison Across Scenarios",
                     fontsize=12, fontweight="bold", y=1.02)
        fig.tight_layout()

    if save:
        path = os.path.join(figdir, "fig_comparison_flux.pdf")
        _save(fig, path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 11.  Cross-scenario latent trajectory comparison
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_latent(
    analyzers: List,
    names: List[str],
    colors: Optional[List[str]] = None,
    figdir: str = "hs_figures",
    save: bool = True,
) -> plt.Figure:
    """Latent-space (PC1 vs PC2) trajectories for all scenarios.

    Parameters
    ----------
    analyzers : list of HarmonicSequenceAnalyzer
    names : list of str
    colors : list of str, optional
    figdir : str
    save : bool

    Returns
    -------
    fig : Figure
    """
    if colors is None:
        colors = SCENARIO_COLORS[: len(analyzers)]
    n = len(analyzers)
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 4.0))
        if n == 1:
            axes = [axes]
        for ax, ana, name, col in zip(axes, analyzers, names, colors):
            if ana.latent is not None:
                Z = ana.latent.trajectory()
                T_steps = Z.shape[0]
                norm = Normalize(0, T_steps - 1)
                cmap = plt.get_cmap("plasma")
                ax.plot(Z[:, 0], Z[:, 1], "-", color="gray", alpha=0.22,
                        lw=0.9, zorder=1)
                for i in range(T_steps):
                    ax.scatter(Z[i, 0], Z[i, 1], color=cmap(norm(i)),
                               s=50, zorder=3, edgecolors="none")
                ax.scatter(Z[0, 0], Z[0, 1], marker="*", s=180, color="lime",
                           zorder=5, edgecolors="black", linewidths=0.4)
                ax.scatter(Z[-1, 0], Z[-1, 1], marker="D", s=90, color="tomato",
                           zorder=5, edgecolors="black", linewidths=0.4)
                evr = ana.latent.explained_variance_ratio_
                ev0 = evr[0] if len(evr) > 0 else 0
                ev1 = evr[1] if len(evr) > 1 else 0
                ax.set_xlabel(f"PC1 ({ev0:.1%})", fontsize=8)
                ax.set_ylabel(f"PC2 ({ev1:.1%})" if ax is axes[0] else "", fontsize=8)
                ax.set_title(name, fontsize=10, fontweight="bold", color=col)
            else:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes,
                        ha="center", va="center", color="gray")
                ax.set_title(name, fontsize=10, color="gray")
            ax.tick_params(labelsize=8)
        fig.suptitle("Latent PCA Trajectories Across Scenarios",
                     fontsize=12, fontweight="bold", y=1.02)
        fig.tight_layout()

    if save:
        path = os.path.join(figdir, "fig_comparison_latent.pdf")
        _save(fig, path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 12.  Cross-scenario summary bar charts
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_summary(
    analyzers: List,
    names: List[str],
    colors: Optional[List[str]] = None,
    figdir: str = "hs_figures",
    save: bool = True,
) -> plt.Figure:
    """Four-panel summary bar charts comparing all scenarios.

    Panels:
        TL  Markov transition entropy (bits)
        TR  Mean Wasserstein flux
        BL  PCA explained variance (first 2 PCs)
        BR  Grammar transition entropy (bits)

    Parameters
    ----------
    analyzers : list of HarmonicSequenceAnalyzer
    names : list of str
    colors : list of str, optional
    figdir : str
    save : bool

    Returns
    -------
    fig : Figure
    """
    if colors is None:
        colors = SCENARIO_COLORS[: len(analyzers)]
    n = len(analyzers)

    # Collect stats
    markov_ent = []
    mean_flux = []
    var_exp = []
    grammar_ent = []
    for ana in analyzers:
        markov_ent.append(
            ana.markov.transition_entropy_ if ana.markov is not None else np.nan
        )
        mean_flux.append(
            float(np.mean(ana.wasserstein.flux_)) if ana.wasserstein is not None else np.nan
        )
        var_exp.append(
            float(ana.latent.explained_variance_ratio_[:2].sum())
            if ana.latent is not None else np.nan
        )
        grammar_ent.append(
            ana.grammar.transition_entropy_ if ana.grammar is not None else np.nan
        )

    metrics = [
        (markov_ent,  "Markov Transition Entropy (bits)",  "H_transition"),
        (mean_flux,   "Mean Wasserstein Flux",              "W₁ (mean)"),
        (var_exp,     "PCA Explained Variance (PC1+PC2)",   "Ratio"),
        (grammar_ent, "Grammar Transition Entropy (bits)",  "H_grammar"),
    ]

    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, (vals, title, ylabel) in zip(axes.flat, metrics):
            xs = np.arange(n)
            bars = ax.bar(xs, vals, color=colors, edgecolor="white",
                          linewidth=0.8, width=0.6)
            ax.set_xticks(xs)
            ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(title, fontsize=10, fontweight="bold")
            for bar, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            v + 0.01 * max(abs(x) for x in vals if np.isfinite(x)),
                            f"{v:.2f}", ha="center", va="bottom", fontsize=8)
            ax.tick_params(labelsize=8)
        fig.suptitle("Cross-Scenario Summary Statistics",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()

    if save:
        path = os.path.join(figdir, "fig_comparison_summary.pdf")
        _save(fig, path)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 13.  Topology + grammar overview  (one row per scenario)
# ─────────────────────────────────────────────────────────────────────────────

def plot_tda_grammar_grid(
    analyzers: List,
    names: List[str],
    figdir: str = "hs_figures",
    save: bool = True,
) -> plt.Figure:
    """Grid of persistence barcodes and grammar interval heatmaps.

    Layout: one row per scenario, two columns (barcode | interval heatmap).

    Parameters
    ----------
    analyzers : list of HarmonicSequenceAnalyzer
    names : list of str
    figdir : str
    save : bool

    Returns
    -------
    fig : Figure
    """
    from biotuner.harmonic_sequence import encode_ji_matrix

    n = len(analyzers)
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(n, 2, figsize=(14, 3.5 * n))
        if n == 1:
            axes = axes[np.newaxis, :]

        for row, (ana, name) in enumerate(zip(analyzers, names)):
            ax_tda = axes[row, 0]
            ax_gram = axes[row, 1]

            # TDA barcode
            if ana.topology is not None:
                try:
                    plot_topology_barcode(ana, ax=ax_tda,
                                          title=f"{name} – Persistence Barcode")
                except Exception as e:
                    ax_tda.text(0.5, 0.5, f"TDA failed:\n{e}",
                                transform=ax_tda.transAxes,
                                ha="center", va="center", fontsize=8, color="red")
            else:
                ax_tda.text(0.5, 0.5, "Topology not fitted",
                            transform=ax_tda.transAxes,
                            ha="center", va="center", fontsize=9, color="gray")
                ax_tda.set_title(f"{name} – Persistence Barcode", fontsize=10)

            # Grammar heatmap
            if ana.grammar is not None:
                try:
                    plot_grammar_interval_heatmap(
                        ana, ax=ax_gram, top_n=12,
                        title=f"{name} – JI Interval Presence",
                    )
                except Exception as e:
                    ax_gram.text(0.5, 0.5, f"Grammar failed:\n{e}",
                                 transform=ax_gram.transAxes,
                                 ha="center", va="center", fontsize=8, color="red")
            else:
                ax_gram.text(0.5, 0.5, "Grammar not fitted",
                             transform=ax_gram.transAxes,
                             ha="center", va="center", fontsize=9, color="gray")
                ax_gram.set_title(f"{name} – JI Interval Presence", fontsize=10)

        fig.suptitle("Topological & Grammar Analysis Across Scenarios",
                     fontsize=13, fontweight="bold", y=1.01)
        fig.tight_layout()

    if save:
        path = os.path.join(figdir, "fig_tda_grammar_grid.pdf")
        _save(fig, path)
    return fig
