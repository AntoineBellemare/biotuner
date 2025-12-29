"""
Biotuner Plotting Utilities
============================

This module provides unified plotting functions for all biotuner visualizations,
ensuring consistent styling across different peak extraction methods and analyses.

Author: Biotuner Team
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Optional, Union, List, Tuple, Dict, Any
import warnings

# Import existing viz functions
try:
    from biotuner.vizs import graph_harm_peaks as _legacy_graph_harm_peaks
except ImportError:
    _legacy_graph_harm_peaks = None

from biotuner.plot_config import (
    set_biotuner_style,
    BIOTUNER_COLORS,
    EMD_COLORS,
    BAND_COLORS,
    BAND_NAMES,
    FREQ_BANDS,
    get_emd_colors,
    get_band_colors,
    get_plot_config,
    get_color_palette,
)


# ============================================================================
# Core Plotting Functions
# ============================================================================

def _plot_peak_amplitude_distribution(
    ax: plt.Axes,
    peaks: np.ndarray,
    amps: np.ndarray,
    xmin: float = 1,
    xmax: float = 60,
    color: str = None,
    EIMC_all: Dict = None,
    n_pairs: int = 5,
    show_notes: bool = True,
    a4: float = 440.0,
    harm_peaks_fit: list = None
) -> None:
    """
    Plot peak amplitude distribution as a bar/stem chart with musical note labels.
    
    For EIMC method, colors bars by peak pair to match the main plot.
    For harmonic_recurrence method, colors bars by fundamental to match the main plot.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to plot on
    peaks : np.ndarray
        Array of peak frequencies
    amps : np.ndarray
        Array of peak amplitudes
    xmin, xmax : float
        Frequency limits for x-axis
    color : str, optional
        Bar color. Default: BIOTUNER_COLORS['accent']
    EIMC_all : dict, optional
        EIMC data for coloring bars by peak pair
    n_pairs : int, default=5
        Number of peak pairs to show for EIMC
    show_notes : bool, default=True
        Whether to show musical note names on bars
    a4 : float, default=440.0
        Reference frequency for A4
    harm_peaks_fit : list, optional
        Harmonic peaks fit data for coloring bars by fundamental (harmonic_recurrence method)
    """
    from biotuner.biotuner_utils import freq_to_note, identify_mode
    
    if color is None:
        color = BIOTUNER_COLORS['accent']
    
    # Filter peaks within range
    mask = (peaks >= xmin) & (peaks <= xmax)
    peaks_filtered = peaks[mask]
    amps_filtered = amps[mask]
    
    # Sort by frequency
    sort_idx = np.argsort(peaks_filtered)
    peaks_sorted = peaks_filtered[sort_idx]
    amps_sorted = amps_filtered[sort_idx]
    
    # Don't shift amplitudes - dB values can be negative and that's valid
    # Just track minimum for y-axis scaling
    min_amp = np.min(amps_sorted) if len(amps_sorted) > 0 else 0
    
    # Thin bar width: 6% of each peak's frequency for clean, narrow bars
    # In log scale, this gives consistent perceived width across all methods
    bar_widths = peaks_sorted * 0.06
    
    # Define color palette (same as used in plot_harmonic_peaks and plot_eimc_peaks)
    harm_colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#9D4EDD']
    
    # For harmonic_recurrence, color bars by position in bt.peaks
    # This matches the color assignment in plot_harmonic_peaks
    if harm_peaks_fit is not None:
        # Build color mapping from original peaks array (bt.peaks)
        # Colors are assigned by ORDER in bt.peaks (first peak gets first color, etc.)
        # Colors cycle through palette for peaks > 5
        peak_to_color = {}
        for i, peak_freq in enumerate(peaks):  # Use ALL peaks, not just first 5
            peak_to_color[round(peak_freq, 1)] = harm_colors[i % len(harm_colors)]
        
        # Assign colors to each bar based on the mapping
        bar_colors = []
        for peak in peaks_sorted:
            peak_key = round(peak, 1)
            # Look up exact match first
            if peak_key in peak_to_color:
                bar_colors.append(peak_to_color[peak_key])
            else:
                # Fallback: find closest match within tolerance
                peak_color = color  # Default
                min_dist = float('inf')
                for key, col in peak_to_color.items():
                    dist = abs(peak - key)
                    if dist < min_dist and dist < 0.5:  # 0.5 Hz tolerance
                        min_dist = dist
                        peak_color = col
                bar_colors.append(peak_color)
        
        # Create bars with individual colors (using centralized bar_widths)
        bars = ax.bar(peaks_sorted, amps_sorted, width=bar_widths, alpha=0.7, linewidth=1.5)
        for bar, bar_color in zip(bars, bar_colors):
            bar.set_facecolor(bar_color)
            bar.set_edgecolor(bar_color)
    
    # For EIMC, color bars by which peak pair they belong to
    elif EIMC_all is not None and 'peaks' in EIMC_all:
        pair_colors = harm_colors[:n_pairs]
        
        # Build a mapping from peak frequency to its primary pair color
        # A peak gets the color of the FIRST (highest priority) pair it appears in
        peak_to_color = {}
        for i, peak_pair in enumerate(EIMC_all['peaks'][:n_pairs]):
            if len(peak_pair) >= 2:
                pair_color = pair_colors[i]
                # Assign color to each peak in the pair if not already assigned
                for peak_freq in [peak_pair[0], peak_pair[1]]:
                    # Round to handle floating point precision
                    peak_key = round(peak_freq, 1)
                    if peak_key not in peak_to_color:
                        peak_to_color[peak_key] = pair_color
        
        # Create color array for each peak
        bar_colors = []
        for peak in peaks_sorted:
            peak_key = round(peak, 1)
            # Look up exact match first
            if peak_key in peak_to_color:
                bar_colors.append(peak_to_color[peak_key])
            else:
                # Fallback: find closest match within tolerance
                peak_color = color  # Default
                min_dist = float('inf')
                for key, col in peak_to_color.items():
                    dist = abs(peak - key)
                    if dist < min_dist and dist < 0.5:  # 0.5 Hz tolerance
                        min_dist = dist
                        peak_color = col
                bar_colors.append(peak_color)
        
        # Create all bars at once with a bar container, then set individual colors (using centralized bar_widths)
        bars = ax.bar(peaks_sorted, amps_sorted, width=bar_widths, alpha=0.7, linewidth=1.5)
        for bar, bar_color in zip(bars, bar_colors):
            bar.set_facecolor(bar_color)
            bar.set_edgecolor(bar_color)
        
        # Add subtitle explaining what's shown
        n_matched = sum(1 for c in bar_colors if c != color)
        ax.text(0.02, 0.95, f'Showing {len(peaks_sorted)} most frequent peaks',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=10, style='italic', color='gray')
        ax.text(0.02, 0.88, f'({n_matched} matched to base pairs)',
                transform=ax.transAxes, ha='left', va='top',
                fontsize=9, style='italic', color='gray')
    else:
        # Standard bar chart for non-EIMC methods (using centralized bar_widths)
        bars = ax.bar(peaks_sorted, amps_sorted, width=bar_widths, color=color, alpha=0.7, 
                      edgecolor=color, linewidth=1.5)
    
    # Add musical note labels on top of bars FIRST, before setting limits
    if show_notes and len(peaks_sorted) <= 12:  # Only show if not too many peaks
        max_amp = np.max(amps_sorted)
        
        for peak, amp, bar in zip(peaks_sorted, amps_sorted, bars):
            note, cents = freq_to_note(peak, a4=a4)
            
            # Format label: note name + cents if significant deviation
            if abs(cents) < 10:
                label = note
            else:
                label = f"{note}\n({cents:+.0f}¢)"
            
            # Place label on top of bar
            ax.text(peak, amp, label, 
                   ha='center', va='bottom', fontsize=9,
                   fontweight='bold', color='black',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            alpha=0.7, edgecolor='none'))
    
    # Style - set limits with proper space for labels
    ax.set_xscale('log')  # Log scale for frequency axis
    
    # Auto-zoom x-axis to peak range +/- 5% for better visibility
    if len(peaks_sorted) > 0:
        peak_min = np.min(peaks_sorted)
        peak_max = np.max(peaks_sorted)
        # Calculate range in log space for log scale
        log_range = np.log10(peak_max) - np.log10(peak_min)
        padding = log_range * 0.05  # 5% padding in log space
        x_min_zoomed = 10 ** (np.log10(peak_min) - padding)
        x_max_zoomed = 10 ** (np.log10(peak_max) + padding)
        ax.set_xlim([x_min_zoomed, x_max_zoomed])
    else:
        ax.set_xlim([xmin, xmax])  # Fallback to full range if no peaks
    
    # Calculate y-limits to accommodate both positive and negative dB values
    if len(amps_sorted) > 0:
        max_amp = np.max(amps_sorted)
        # Start from slightly below minimum to show negative values clearly
        y_min = min_amp - abs(max_amp - min_amp) * 0.1  # 10% padding below
        # Add space above for labels
        y_max = max_amp + abs(max_amp - min_amp) * 0.45  # 45% padding above for labels
    else:
        y_min, y_max = 0, 1
    ax.set_ylim([y_min, y_max])
    ax.set_xlabel('Frequency (Hz)', fontsize=14, fontweight='normal')
    ax.set_ylabel('Amplitude', fontsize=14, fontweight='normal')
    ax.set_title('Peak Amplitude Distribution', fontsize=16, fontweight='bold', pad=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Add mode identification at top-right (replaces peak count annotation)
    # Show mode if detected, otherwise show peak count
    if len(peaks_sorted) >= 3:
        mode_name, similarity, root = identify_mode(peaks_sorted, a4=a4)
        if similarity >= 60:  # Raised threshold to match stricter algorithm
            mode_text = f"{mode_name} in {root}\n({similarity:.0f}% | n={len(peaks_sorted)})"
            ax.text(0.98, 0.95, mode_text,
                   transform=ax.transAxes, ha='right', va='top',
                   fontsize=10, style='normal', fontweight='semibold',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', 
                            alpha=0.9, edgecolor='gray', linewidth=0.5))
        else:
            # No confident mode match, just show peak count
            ax.text(0.98, 0.95, f'n = {len(peaks_sorted)} peaks',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        # Too few peaks for mode detection
        ax.text(0.98, 0.95, f'n = {len(peaks_sorted)} peaks',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))



def _plot_peak_ratios_matrix(
    ax: plt.Axes,
    peaks: np.ndarray,
    metric: str = 'harmsim',
    limit: int = 1000
) -> None:
    """
    Plot matrix of peak ratios and their harmonicity.
    
    Parameters
    ----------
    ax : plt.Axes
        Matplotlib axes to plot on
    peaks : np.ndarray
        Array of peak frequencies
    metric : str, default='harmsim'
        Metric to compute: 'harmsim', 'cons', 'tenney', 'denom', 'subharm_tension'
    limit : int, default=1000
        Denominator limit for ratio simplification
    """
    from biotuner.metrics import (
        ratios2harmsim, compute_consonance, tenneyHeight, metric_denom
    )
    from fractions import Fraction
    
    n_peaks = len(peaks)
    
    # Compute pairwise ratios and metrics
    matrix = np.zeros((n_peaks, n_peaks))
    
    for i in range(n_peaks):
        for j in range(n_peaks):
            if i == j:
                matrix[i, j] = np.nan  # Diagonal
            else:
                # Compute ratio
                ratio = peaks[j] / peaks[i]
                
                # Compute metric
                if metric == 'harmsim':
                    # Use dyad_similarity via ratios2harmsim
                    matrix[i, j] = ratios2harmsim([ratio])[0]
                elif metric == 'cons':
                    matrix[i, j] = compute_consonance(ratio, limit=limit)
                elif metric == 'tenney':
                    # Tenney height for single ratio
                    frac = Fraction(ratio).limit_denominator(limit)
                    matrix[i, j] = np.log2(frac.numerator * frac.denominator)
                elif metric == 'denom':
                    matrix[i, j] = metric_denom(ratio)
                elif metric == 'subharm_tension':
                    # Subharmonic tension (simplified version)
                    from biotuner.biotuner_utils import rebound
                    rebounded = rebound(ratio, 1, 2, 2)
                    frac = Fraction(rebounded).limit_denominator(limit)
                    # Higher denominator = more tension
                    matrix[i, j] = frac.denominator
    
    # Create custom colormap: dark burnt orange to turquoise
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['#8B4513', '#CD5C5C', '#F4A460', '#FFD700', '#90EE90', '#48D1CC', '#40E0D0']
    custom_cmap = LinearSegmentedColormap.from_list('orange_to_turquoise', colors_list)
    
    # Determine colormap based on metric
    if metric in ['harmsim', 'cons']:
        # Higher is better (more harmonic/consonant) - custom burnt orange to turquoise
        cmap = custom_cmap
        vmin, vmax = 0, np.nanmax(matrix) if np.nanmax(matrix) > 0 else 1
    else:
        # Higher is worse (more dissonant/complex) - reversed
        cmap = custom_cmap.reversed()
        vmin, vmax = np.nanmin(matrix[matrix > 0]) if np.any(matrix > 0) else 0, np.nanmax(matrix)
    
    # Plot matrix
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Metric-specific labels
    metric_labels = {
        'harmsim': 'Harmonic Similarity (%)',
        'cons': 'Consonance',
        'tenney': 'Tenney Height (log)',
        'denom': 'Denominator',
        'subharm_tension': 'Subharmonic Tension'
    }
    cbar.set_label(metric_labels.get(metric, metric), fontsize=12)
    
    # Set ticks and labels
    peak_labels = [f'{p:.1f} Hz' for p in peaks]
    ax.set_xticks(np.arange(n_peaks))
    ax.set_yticks(np.arange(n_peaks))
    ax.set_xticklabels(peak_labels, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(peak_labels, fontsize=11)
    
    # Add gridlines
    ax.set_xticks(np.arange(n_peaks) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_peaks) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)
    
    # Labels
    ax.set_xlabel('Peak j (Hz)', fontsize=14, fontweight='normal')
    ax.set_ylabel('Peak i (Hz)', fontsize=14, fontweight='normal')
    ax.set_title(f'Peak Ratios Harmonicity Matrix\nMetric: {metric_labels.get(metric, metric)}', 
                fontsize=16, fontweight='bold', pad=15)
    
    # Add text annotations for small matrices
    if n_peaks <= 8:
        for i in range(n_peaks):
            for j in range(n_peaks):
                if not np.isnan(matrix[i, j]):
                    # Get normalized value (0 to 1)
                    norm_val = (matrix[i, j] - vmin) / (vmax - vmin)
                    
                    # Get RGB color from colormap for this value
                    rgba = cmap(norm_val)
                    
                    # Calculate luminance (perceived brightness)
                    # Using ITU-R BT.709 formula
                    luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                    
                    # Use white text on dark backgrounds, black on light
                    text_color = 'white' if luminance < 0.5 else 'black'
                    
                    if metric in ['denom', 'subharm_tension']:
                        text = f'{int(matrix[i, j])}'
                    else:
                        text = f'{matrix[i, j]:.1f}'
                    ax.text(j, i, text, ha='center', va='center', 
                           color=text_color, fontsize=11, fontweight='bold')


def plot_cepstrum_peaks(
    quefrency: np.ndarray,
    cepstrum: np.ndarray,
    peaks: np.ndarray,
    xmin: float = None,
    xmax: float = None,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    color: str = None,
    peak_color: str = None,
    show_bands: bool = False,
    method: str = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot cepstrum with peak markers.
    
    Cepstrum is different from PSD - it shows quefrency (time) vs amplitude,
    revealing periodicity in the frequency spectrum.
    
    Parameters
    ----------
    quefrency : np.ndarray
        Quefrency vector (in seconds)
    cepstrum : np.ndarray
        Cepstrum amplitude values
    peaks : np.ndarray
        Peak frequencies (in Hz) - will be converted to quefrency
    xmin, xmax : float, optional
        Frequency range (Hz) - converted to quefrency range
    title : str, optional
        Custom plot title
    figsize : tuple, optional
        Figure size
    color : str, optional
        Line color for cepstrum
    peak_color : str, optional
        Color for peak markers
    show_bands : bool, optional
        Not used for cepstrum, included for API compatibility
    method : str, optional
        Method name, included for API compatibility
    ax : plt.Axes, optional
        Existing axes
    **kwargs
        Additional plot parameters
    
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    ax : plt.Axes
        Matplotlib axes
    """
    set_biotuner_style()
    
    # Set defaults
    if color is None:
        color = BIOTUNER_COLORS['primary']
    if peak_color is None:
        peak_color = BIOTUNER_COLORS['accent']
    if figsize is None:
        figsize = (12, 7)
    if xmin is None:
        xmin = 1
    if xmax is None:
        xmax = 60
    
    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Plot cepstrum (don't pass kwargs to avoid matplotlib errors)
    ax.plot(quefrency, np.abs(cepstrum), color=color, linewidth=2, label='Cepstrum')
    
    # Convert frequency limits to quefrency (time)
    qmin = 1 / xmax
    qmax = 1 / xmin
    
    # Set limits
    idx_min = np.argmin(np.abs(quefrency - qmin))
    idx_max = np.argmin(np.abs(quefrency - qmax))
    ymin = np.min(np.abs(cepstrum[idx_min:idx_max]))
    ymax = np.max(np.abs(cepstrum[idx_min:idx_max]))
    
    ax.set_xlim([qmin, qmax])
    ax.set_ylim([ymin, ymax * 1.05])
    
    # Mark peaks (convert frequency to quefrency)
    for peak in peaks:
        if xmin <= peak <= xmax:
            peak_quefrency = 1 / peak
            ax.axvline(x=peak_quefrency, color=peak_color, linestyle='--', 
                      alpha=0.7, linewidth=1.5, zorder=10)
            # Annotate with frequency value
            ax.text(peak_quefrency, ymax * 0.95, f'{peak:.1f} Hz',
                   rotation=90, va='top', ha='right', fontsize=10,
                   color=peak_color, fontweight='bold')
    
    # Labels
    ax.set_xlabel('Quefrency (seconds)', fontsize=16, fontweight='normal')
    ax.set_ylabel('Cepstrum Amplitude', fontsize=16, fontweight='normal')
    
    # Title
    if title:
        ax.set_title(title, fontsize=20, fontweight='bold', pad=15)
    else:
        ax.set_title('Cepstrum Analysis', fontsize=20, fontweight='bold', pad=15)
    
    # Add secondary x-axis showing corresponding frequencies
    ax2 = ax.twiny()
    ax2.set_xlim([xmax, xmin])
    ax2.set_xlabel('Corresponding Frequency (Hz)', fontsize=14, fontweight='normal', color='gray')
    ax2.tick_params(axis='x', labelcolor='gray')
    
    plt.tight_layout()
    return fig, ax


def plot_psd_peaks(
    freqs: np.ndarray,
    psd: np.ndarray,
    peaks: np.ndarray,
    xmin: float = 1,
    xmax: float = 60,
    title: Optional[str] = None,
    method: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    color: str = None,
    peak_color: str = None,
    show_bands: bool = False,
    bands: Optional[Dict[str, List[float]]] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Universal function to plot PSD with peak markers.
    
    Parameters
    ----------
    freqs : np.ndarray
        Frequency vector
    psd : np.ndarray
        Power spectral density values
    peaks : np.ndarray
        Peak frequencies to mark
    xmin, xmax : float
        X-axis limits in Hz
    title : str, optional
        Custom plot title
    method : str, optional
        Peak extraction method name (used in default title)
    figsize : tuple, optional
        Figure size (width, height)
    color : str, optional
        Line color for PSD. Default: BIOTUNER_COLORS['primary']
    peak_color : str, optional
        Color for peak markers. Default: BIOTUNER_COLORS['accent']
    show_bands : bool, default=False
        Whether to show frequency band overlays
    bands : dict, optional
        Custom frequency bands. Default: standard EEG bands
    ax : plt.Axes, optional
        Existing axes to plot on
    **kwargs
        Additional arguments passed to plt.plot()
    
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure object
    ax : plt.Axes
        Matplotlib axes object
    
    Examples
    --------
    >>> fig, ax = plot_psd_peaks(freqs, psd, peaks, xmin=1, xmax=60, 
    ...                          method='EMD', show_bands=True)
    >>> plt.show()
    """
    set_biotuner_style()
    
    # Set defaults
    if color is None:
        color = BIOTUNER_COLORS['primary']
    if peak_color is None:
        peak_color = BIOTUNER_COLORS['accent']
    if figsize is None:
        figsize = get_plot_config('psd')['figsize']
    if bands is None:
        bands = FREQ_BANDS
    
    # Create figure if no axes provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Plot PSD
    ax.plot(freqs, psd, color=color, linewidth=2, label='PSD', **kwargs)
    
    # Set limits first
    idx_min = np.argmin(np.abs(freqs - xmin))
    idx_max = np.argmin(np.abs(freqs - xmax))
    ymin = np.min(psd[idx_min:idx_max])
    ymax = np.max(psd[idx_min:idx_max])
    
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax * 1.05])
    
    # Add frequency bands AFTER setting ylim
    if show_bands:
        _add_frequency_bands(ax, bands, xmin, xmax)
    
    # Mark peaks on top of bands
    for peak in peaks:
        if xmin <= peak <= xmax:
            ax.axvline(x=peak, color=peak_color, linestyle='--', 
                      alpha=0.7, linewidth=1.5, zorder=10)
    
    # Labels
    ax.set_xlabel('Frequency (Hz)', fontsize=16, fontweight='normal')
    ax.set_ylabel('Power Spectral Density', fontsize=16, fontweight='normal')
    
    # Title
    if title:
        ax.set_title(title, fontsize=20, fontweight='bold', pad=15)
    elif method:
        ax.set_title(f'Spectral Peaks - {method} Method', fontsize=20, fontweight='bold', pad=15)
    else:
        ax.set_title('Spectral Peaks', fontsize=20, fontweight='bold', pad=15)
    
    plt.tight_layout()
    return fig, ax


def plot_emd_peaks(
    freqs_all: Union[List[np.ndarray], np.ndarray] = None,
    psd_all: Union[List[np.ndarray], np.ndarray] = None,
    peaks: np.ndarray = None,
    raw_data: Optional[np.ndarray] = None,
    sf: float = 1000,
    xmin: float = 1,
    xmax: float = 60,
    figsize: Optional[Tuple[float, float]] = None,
    show_bands: bool = True,
    bands: Optional[Dict[str, List[float]]] = None,
    compare_raw: bool = True,
    colors: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    IMFs: Optional[List[np.ndarray]] = None,
    nperseg: int = None,
    precision: float = 0.5,
    smooth: int = 1,
    use_db: bool = True,
    fill_imfs: bool = False,
    log_scale: bool = False,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot EMD decomposition with peaks.
    
    Parameters
    ----------
    freqs_all : list of np.ndarray, optional
        Frequency vectors for each IMF. If None, will compute from IMFs.
    psd_all : list of np.ndarray, optional
        PSD values for each IMF. If None, will compute from IMFs.
    peaks : np.ndarray
        Detected peak frequencies
    raw_data : np.ndarray, optional
        Raw signal for comparison
    sf : float, default=1000
        Sampling frequency in Hz
    xmin, xmax : float
        Frequency limits
    figsize : tuple, optional
        Figure size
    show_bands : bool, default=True
        Show frequency band overlays
    bands : dict, optional
        Custom frequency bands
    compare_raw : bool, default=True
        Overlay raw signal PSD
    colors : list, optional
        Custom colors for IMFs
    ax : plt.Axes, optional
        Existing axes
    IMFs : list of np.ndarray, optional
        List of intrinsic mode functions. If provided, will compute PSD internally.
    nperseg : int, optional
        Length of each segment for Welch's method. If None, calculated from precision.
    precision : float, default=0.5
        Frequency bin size in Hz. Used to calculate nperseg if not provided.
    smooth : int, default=1
        Smoothing factor for nperseg calculation.
    use_db : bool, default=True
        Convert PSD to decibels (10*log10). Matches other methods.
    fill_imfs : bool, default=False
        Fill area under IMF curves. Disabled by default for cleaner visualization.
    log_scale : bool, default=False
        Use logarithmic scales for both axes. Set to True for traditional EMD visualization,
        False for linear scales to match other methods (FOOOF, fixed, etc.).
    **kwargs
        Additional plot parameters
    
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    ax : plt.Axes
        Matplotlib axes
    
    Examples
    --------
    >>> # Automatic PSD computation from IMFs
    >>> fig, ax = plot_emd_peaks(peaks=peaks, IMFs=bt.IMFs, sf=1000)
    >>> 
    >>> # Manual PSD specification
    >>> fig, ax = plot_emd_peaks(freqs_all, psd_all, peaks, 
    ...                          raw_data=data, sf=1000)
    >>> plt.show()
    """
    import scipy.signal
    
    set_biotuner_style()
    
    # Calculate nperseg from precision if not provided (same as extract_welch_peaks)
    if nperseg is None:
        mult = 1 / precision
        nfft = sf * mult
        nperseg = int(nfft / smooth)
    else:
        # If nperseg is provided, calculate nfft accordingly
        nfft = nperseg * smooth
    
    # Compute PSD from IMFs if not provided
    if IMFs is not None and (freqs_all is None or psd_all is None):
        freqs_all = []
        psd_all = []
        for imf in IMFs:
            freqs, psd = scipy.signal.welch(imf, sf, nfft=nfft, nperseg=nperseg)
            # Convert to dB to match other methods
            if use_db:
                psd = 10.0 * np.log10(np.maximum(psd, 1e-12))
                psd = np.real(psd)
            freqs_all.append(freqs)
            psd_all.append(psd)
    
    # Set defaults
    if figsize is None:
        figsize = get_plot_config('emd')['figsize']
    if bands is None:
        bands = FREQ_BANDS
    if colors is None:
        colors = get_emd_colors(len(psd_all))
    
    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Plot each IMF
    n_imfs = len(psd_all)
    for i, (freqs, psd, color) in enumerate(zip(freqs_all, psd_all, colors)):
        label = f'IMF{n_imfs - i}'
        ax.plot(freqs, psd, color=color, linewidth=1.5, label=label)
        # Optionally fill area under curves
        if fill_imfs:
            ax.fill_between(freqs, psd, alpha=0.3, color=color)
    
    # Compare with raw data if provided
    if compare_raw and raw_data is not None:
        # Use same parameters as IMFs and other methods
        freqs_raw, psd_raw = scipy.signal.welch(raw_data, sf, nfft=nfft, nperseg=nperseg)
        # Convert to dB to match IMF PSDs
        if use_db:
            psd_raw = 10.0 * np.log10(np.maximum(psd_raw, 1e-12))
            psd_raw = np.real(psd_raw)
        ax.plot(freqs_raw, psd_raw, color=BIOTUNER_COLORS['danger'], 
               linestyle='--', linewidth=2, label='Raw Signal', alpha=0.8)
    
    # Mark peaks
    for i, peak in enumerate(peaks):
        if xmin <= peak <= xmax:
            ax.axvline(x=peak, color=colors[min(i, len(colors)-1)], 
                      linestyle='-', linewidth=2, alpha=0.8)
    
    # Add frequency bands
    if show_bands:
        _add_frequency_bands(ax, bands, xmin, xmax, position='top')
    
    # Styling
    ax.set_xlim([xmin, xmax])
    ax.set_xlabel('Frequency (Hz)', fontsize=16, fontweight='normal')
    # Use consistent label with other methods
    ylabel = 'Power Spectral Density (dB)' if use_db else 'Power Spectral Density'
    ax.set_ylabel(ylabel, fontsize=16, fontweight='normal')
    ax.set_title('Empirical Mode Decomposition', fontsize=20, fontweight='bold', pad=15)
    
    # Apply logarithmic scales if requested
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('symlog')
    
    ax.legend(loc='lower left', fontsize=13, framealpha=0.95)
    
    plt.tight_layout()
    return fig, ax


def plot_eimc_peaks(
    freqs: np.ndarray,
    psd: np.ndarray,
    peaks: List[float],
    EIMC_all: Dict,
    xmin: float = 1,
    xmax: float = 60,
    n_pairs: int = 5,
    figsize: Optional[Tuple[float, float]] = None,
    color: str = None,
    show_bands: bool = False,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot EIMC (Endogenous InterModulation Components) with intermodulation markers.
    
    Parameters
    ----------
    freqs : np.ndarray
        Frequency vector
    psd : np.ndarray
        Power spectral density
    peaks : list of float
        Main peak frequencies
    EIMC_all : dict
        Dictionary with keys 'peaks', 'IMs', 'n_IMs', 'amps' containing intermodulation info
    xmin, xmax : float
        Frequency limits
    n_pairs : int, default=5
        Number of peak pairs to display
    figsize : tuple, optional
        Figure size
    color : str, optional
        PSD line color
    show_bands : bool, default=False
        Whether to show frequency bands
    ax : plt.Axes, optional
        Existing axes
    **kwargs
        Additional parameters
    
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    ax : plt.Axes
        Matplotlib axes
    """
    set_biotuner_style()
    
    # Set defaults
    if color is None:
        color = BIOTUNER_COLORS['primary']
    if figsize is None:
        figsize = (12, 7)
    
    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Plot PSD with consistent styling
    idx_max = np.argmin(np.abs(freqs - xmax))
    ax.plot(freqs[:idx_max], psd[:idx_max], color=color, linewidth=2, label='PSD')
    
    # Set Y limits for annotations
    idx_min = np.argmin(np.abs(freqs - xmin))
    ymin = np.min(psd[idx_min:idx_max])
    ymax = np.max(psd[idx_min:idx_max])
    y_step = (ymax - ymin) / 10
    y_positions = [ymax - (y_step * (i + 2)) for i in range(min(n_pairs, len(EIMC_all['peaks'])))]
    
    # Color palette for different peak pairs
    pair_colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#9D4EDD'][:n_pairs]
    
    # Plot intermodulation components for top pairs
    for i, (peak_pair, IMs, n_IMs, pair_color, y_pos) in enumerate(
        zip(EIMC_all['peaks'][:n_pairs], 
            EIMC_all['IMs'][:n_pairs],
            EIMC_all['n_IMs'][:n_pairs],
            pair_colors,
            y_positions)
    ):
        if len(peak_pair) < 2:
            continue
        
        p1, p2 = peak_pair[0], peak_pair[1]
        
        # Mark the two base peaks with solid lines
        if xmin <= p1 <= xmax:
            ax.axvline(x=p1, color=pair_color, linestyle='-', linewidth=2.5, alpha=0.9)
        if xmin <= p2 <= xmax:
            ax.axvline(x=p2, color=pair_color, linestyle='-', linewidth=2.5, alpha=0.9)
        
        # Mark intermodulation components with dotted lines
        for im_freq in IMs:
            if xmin <= im_freq <= xmax:
                ax.axvline(x=im_freq, color=pair_color, linestyle='dotted', 
                          linewidth=2, alpha=0.8)
                
                # Annotate IM with "IM" label
                ax.annotate(
                    'IM',
                    xy=(im_freq, y_pos),
                    xytext=(im_freq + 0.5, y_pos),
                    fontsize=11,
                    fontweight='semibold',
                    bbox=dict(boxstyle='square,pad=0.3', 
                             alpha=0.25, 
                             facecolor=pair_color,
                             edgecolor='none')
                )
        
        # Add legend entry showing the peak pair
        ax.plot([], [], color=pair_color, linestyle='-', linewidth=2.5, 
               label=f'{p1:.1f}↔{p2:.1f} Hz ({n_IMs} IMs)')
    
    # Add frequency bands if requested
    if show_bands:
        _add_frequency_bands(ax, FREQ_BANDS, xmin, xmax)
    
    # Add legend for peak pairs
    if len(EIMC_all['peaks']) > 0:
        ax.legend(loc='upper right', fontsize=11, framealpha=0.95, 
                 title='Peak Pairs', title_fontsize=12)
    
    # Styling with new larger fonts
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax * 1.05])
    ax.set_xlabel('Frequency (Hz)', fontsize=16, fontweight='normal')
    ax.set_ylabel('Power Spectral Density', fontsize=16, fontweight='normal')
    ax.set_title('Spectral Peaks - EIMC', 
                fontsize=20, fontweight='bold', pad=15)
    
    plt.tight_layout()
    return fig, ax


def plot_harmonic_peaks(
    freqs: np.ndarray,
    psd: np.ndarray,
    harm_peaks_fit: List[Tuple],
    xmin: float = 1,
    xmax: float = 60,
    n_peaks: int = 5,
    figsize: Optional[Tuple[float, float]] = None,
    color: str = None,
    show_bands: bool = False,
    ax: Optional[plt.Axes] = None,
    selected_peaks: Optional[np.ndarray] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot harmonic recurrence with fitted harmonics using new unified style.
    
    Parameters
    ----------
    freqs : np.ndarray
        Frequency vector
    psd : np.ndarray
        Power spectral density
    harm_peaks_fit : list of tuples
        Harmonic fit information: [(peak, harmonic_positions, harmonic_freqs), ...]
    xmin, xmax : float
        Frequency limits
    n_peaks : int, default=5
        Number of fundamental peaks to display
    figsize : tuple, optional
        Figure size
    color : str, optional
        PSD line color
    show_bands : bool, default=False
        Whether to show frequency bands
    ax : plt.Axes, optional
        Existing axes
    selected_peaks : np.ndarray, optional
        The actual selected peaks from harmonic_recurrence algorithm (bt.peaks).
        If provided, only these peaks from harm_peaks_fit will be plotted.
    **kwargs
        Additional parameters
    
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    ax : plt.Axes
        Matplotlib axes
    
    Examples
    --------
    >>> fig, ax = plot_harmonic_peaks(freqs, psd, harm_peaks_fit, 
    ...                                xmin=1, xmax=60, n_peaks=5,
    ...                                selected_peaks=bt.peaks)
    >>> plt.show()
    """
    set_biotuner_style()
    
    # Set defaults
    if color is None:
        color = BIOTUNER_COLORS['primary']  # Use primary blue instead of dark
    if figsize is None:
        figsize = (12, 7)  # Standard size
    
    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Plot PSD with consistent styling
    idx_max = np.argmin(np.abs(freqs - xmax))
    ax.plot(freqs[:idx_max], psd[:idx_max], color=color, linewidth=2, label='PSD')
    
    # Set Y limits for annotations
    idx_min = np.argmin(np.abs(freqs - xmin))
    ymin = np.min(psd[idx_min:idx_max])
    ymax = np.max(psd[idx_min:idx_max])
    y_step = (ymax - ymin) / 10
    y_positions = [ymax - (y_step * (i + 2)) for i in range(n_peaks)]
    
    # Color palette for different fundamentals (using biotuner colors)
    base_colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#9D4EDD']
    
    # If selected_peaks provided, use those as the peaks to plot (bt.peaks)
    # and try to find harmonic info for them in harm_peaks_fit
    if selected_peaks is not None:
        # Use ALL selected_peaks (these are THE peaks we want to show)
        # Don't limit by n_peaks when we have explicit selected_peaks
        actual_peaks_to_plot = selected_peaks
        n_peaks_to_show = len(selected_peaks)  # Update n_peaks for y_positions
        
        # Build a lookup dict for harmonic info
        harm_info_dict = {}
        for peak_info in harm_peaks_fit:
            if len(peak_info) >= 3:
                peak_freq = peak_info[0]
                harm_info_dict[peak_freq] = peak_info
        
        # Recalculate y_positions for the actual number of peaks
        y_positions = [ymax - (y_step * (i + 2)) for i in range(n_peaks_to_show)]
        
        # Create peak_info entries for each selected peak
        # Colors assigned by ORDER in selected_peaks (simple and consistent)
        harm_peaks_to_plot = []
        for peak_freq in actual_peaks_to_plot:
            # Try to find matching harmonic info (with 0.5 Hz tolerance)
            matching_info = None
            for fit_freq, fit_info in harm_info_dict.items():
                if abs(peak_freq - fit_freq) < 0.5:
                    matching_info = fit_info
                    break
            
            if matching_info:
                harm_peaks_to_plot.append(matching_info)
            else:
                # No harmonic info for this peak - just mark it without harmonics
                harm_peaks_to_plot.append([peak_freq, [], [peak_freq]])
        
        # Assign colors by position in selected_peaks (consistent everywhere)
        harm_colors = [base_colors[i % len(base_colors)] for i in range(len(harm_peaks_to_plot))]
    else:
        # Use first n_peaks from harm_peaks_fit (old behavior)
        harm_peaks_to_plot = harm_peaks_fit[:n_peaks]
        harm_colors = [base_colors[i % len(base_colors)] for i in range(len(harm_peaks_to_plot))]
    
    # Plot harmonics for each peak with new style
    for i, (peak_info, harm_color, y_pos) in enumerate(zip(harm_peaks_to_plot, 
                                                             harm_colors, 
                                                             y_positions)):
        if len(peak_info) < 3:
            continue
            
        peak = peak_info[0]
        harm_positions = [int(x) for x in peak_info[1]]
        harm_freqs = list(peak_info[2])
        
        # Remove fundamental from harmonics list
        if peak in harm_freqs:
            harm_freqs.remove(peak)
        
        # Mark fundamental with thicker line
        ax.axvline(x=peak, color=harm_color, linestyle='-', linewidth=2.5, alpha=0.9)
        
        # Mark and annotate harmonics
        for h_idx, harm_freq in enumerate(harm_freqs):
            if xmin <= harm_freq <= xmax:
                ax.axvline(x=harm_freq, color=harm_color, linestyle='dotted', 
                          linewidth=2, alpha=0.8)
                
                # Annotate with harmonic number (modern styling)
                if h_idx < len(harm_positions):
                    ax.annotate(
                        str(harm_positions[h_idx]),
                        xy=(harm_freq, y_pos),
                        xytext=(harm_freq + 0.5, y_pos),
                        fontsize=12,
                        fontweight='semibold',
                        bbox=dict(boxstyle='square,pad=0.4', 
                                 alpha=0.25, 
                                 facecolor=harm_color,
                                 edgecolor='none')
                    )
    
    # Add frequency bands if requested
    if show_bands:
        _add_frequency_bands(ax, FREQ_BANDS, xmin, xmax)
    
    # Styling with new larger fonts - consistent with plot_psd_peaks
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax * 1.05])  # Add 5% padding like other methods
    ax.set_xlabel('Frequency (Hz)', fontsize=16, fontweight='normal')
    ax.set_ylabel('Power Spectral Density', fontsize=16, fontweight='normal')  # Consistent label
    ax.set_title('Spectral Peaks - Harmonic Recurrence', 
                fontsize=20, fontweight='bold', pad=15)
    
    plt.tight_layout()
    return fig, ax


def plot_entropy_curve(
    x_ratios: np.ndarray,
    entropy_values: np.ndarray,
    minima: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    octave: float = 2,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = 'Harmonic Entropy',
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot harmonic entropy curve with local minima.
    
    Parameters
    ----------
    x_ratios : np.ndarray
        Frequency ratio values
    entropy_values : np.ndarray
        Harmonic entropy values
    minima : tuple of arrays, optional
        (minima_ratios, minima_entropies)
    octave : float, default=2
        Octave reference value
    figsize : tuple, optional
        Figure size
    title : str
        Plot title
    ax : plt.Axes, optional
        Existing axes
    **kwargs
        Additional parameters
    
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    ax : plt.Axes
        Matplotlib axes
    """
    set_biotuner_style()
    
    if figsize is None:
        figsize = get_plot_config('entropy')['figsize']
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Plot entropy curve
    ax.plot(x_ratios, entropy_values, color=BIOTUNER_COLORS['primary'], 
           linewidth=2, label='Harmonic Entropy')
    
    # Mark minima
    if minima is not None:
        ax.scatter(minima[0], minima[1], color=BIOTUNER_COLORS['accent'], 
                  s=40, zorder=5, label='Local Minima')
    
    # Styling
    ax.set_xlim(1, octave)
    ax.set_xlabel('Frequency Ratio', fontsize=16, fontweight='normal')
    ax.set_ylabel('Harmonic Entropy', fontsize=16, fontweight='normal')
    ax.set_title(title, fontsize=20, fontweight='bold', pad=15)
    ax.legend(fontsize=13)
    
    plt.tight_layout()
    return fig, ax


def plot_dissonance_curve(
    intervals: np.ndarray,
    dissonance: np.ndarray,
    scale_ratios: Optional[np.ndarray] = None,
    n_tet_grid: Optional[int] = None,
    max_ratio: float = 2,
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot dissonance curve with scale steps.
    
    Parameters
    ----------
    intervals : np.ndarray
        Frequency ratio intervals
    dissonance : np.ndarray
        Dissonance values
    scale_ratios : np.ndarray, optional
        Derived scale ratios to mark
    n_tet_grid : int, optional
        N-TET grid to overlay (e.g., 12 for 12-TET)
    max_ratio : float, default=2
        Maximum ratio (octave)
    figsize : tuple, optional
        Figure size
    ax : plt.Axes, optional
        Existing axes
    **kwargs
        Additional parameters
    
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    ax : plt.Axes
        Matplotlib axes
    """
    set_biotuner_style()
    
    if figsize is None:
        figsize = get_plot_config('dissonance')['figsize']
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Plot dissonance curve
    ax.plot(intervals, dissonance, color=BIOTUNER_COLORS['primary'], 
           linewidth=2.5, label='Dissonance')
    
    # Mark scale ratios
    if scale_ratios is not None:
        for ratio in scale_ratios:
            ax.axvline(ratio, color=BIOTUNER_COLORS['secondary'], 
                      linestyle='--', alpha=0.6, linewidth=1)
    
    # Add N-TET grid if requested
    if n_tet_grid is not None:
        from biotuner.biotuner_utils import NTET_ratios
        tet_ratios = NTET_ratios(n_tet_grid, max_ratio=max_ratio)
        for ratio in tet_ratios:
            ax.axvline(ratio, color=BIOTUNER_COLORS['danger'], 
                      linestyle=':', alpha=0.5, linewidth=1)
    
    # Styling
    ax.set_xlim(1, max_ratio)
    ax.set_xlabel('Frequency Ratio', fontsize=16, fontweight='normal')
    ax.set_ylabel('Dissonance', fontsize=16, fontweight='normal')
    ax.set_title('Dissonance Curve', fontsize=20, fontweight='bold', pad=15)
    
    plt.tight_layout()
    return fig, ax


# ============================================================================
# Helper Functions
# ============================================================================

def _add_frequency_bands(
    ax: plt.Axes,
    bands: Dict[str, List[float]],
    xmin: float,
    xmax: float,
    position: str = 'top',
    alpha: float = 0.12
):
    """
    Add frequency band overlays to a plot with a legend.
    
    Parameters
    ----------
    ax : plt.Axes
        Axes to add bands to
    bands : dict
        Dictionary of band names and [min, max] frequencies
    xmin, xmax : float
        Plot x-axis limits
    position : str, default='top'
        Kept for backward compatibility (not used, legend shown automatically)
    alpha : float, default=0.12
        Transparency of band overlays
    """
    # Collect patches for legend
    patches = []
    
    for band_name in BAND_NAMES:
        if band_name not in bands:
            continue
            
        band_range = bands[band_name]
        # Only show bands that are visible in the plot range
        if band_range[1] < xmin or band_range[0] > xmax:
            continue
        
        # Adjust band range to visible area
        visible_min = max(band_range[0], xmin)
        visible_max = min(band_range[1], xmax)
        
        # Skip if band is too narrow to be visible
        if visible_max <= visible_min:
            continue
        
        color = BAND_COLORS[band_name]
        
        # Draw band overlay with subtle styling
        ax.axvspan(visible_min, visible_max, alpha=alpha, 
                  color=color, zorder=0, linewidth=0)
        
        # Create patch for legend
        patch = mpatches.Patch(
            color=color, 
            alpha=0.5, 
            label=f"{band_name.capitalize()} ({band_range[0]}-{band_range[1]} Hz)"
        )
        patches.append(patch)
    
    # Add legend if we have any visible bands
    if patches:
        ax.legend(handles=patches, loc='upper right', 
                 fontsize=12, framealpha=0.95, 
                 title='Frequency Bands', title_fontsize=13)


def create_comparison_plot(
    data_list: List[Tuple[np.ndarray, np.ndarray, str]],
    xmin: float = 1,
    xmax: float = 60,
    figsize: Optional[Tuple[float, float]] = None,
    title: str = 'Method Comparison',
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a comparison plot of multiple methods.
    
    Parameters
    ----------
    data_list : list of tuples
        [(freqs, psd, label), ...] for each method
    xmin, xmax : float
        Frequency limits
    figsize : tuple, optional
        Figure size
    title : str
        Plot title
    **kwargs
        Additional parameters
    
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    ax : plt.Axes
        Matplotlib axes
    
    Examples
    --------
    >>> data_list = [(freqs1, psd1, 'EMD'), (freqs2, psd2, 'FOOOF')]
    >>> fig, ax = create_comparison_plot(data_list, title='Peak Extraction Comparison')
    >>> plt.show()
    """
    set_biotuner_style()
    
    if figsize is None:
        figsize = (12, 6)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = get_color_palette('biotuner_gradient', n_colors=len(data_list))
    
    for (freqs, psd, label), color in zip(data_list, colors):
        ax.plot(freqs, psd, color=color, linewidth=2, label=label, alpha=0.8)
    
    ax.set_xlim([xmin, xmax])
    ax.set_xlabel('Frequency (Hz)', fontsize=16, fontweight='normal')
    ax.set_ylabel('Power Spectral Density', fontsize=16, fontweight='normal')
    ax.set_title(title, fontsize=20, fontweight='bold', pad=15)
    ax.legend(fontsize=13, framealpha=0.95)
    
    plt.tight_layout()
    return fig, ax


# ============================================================================
# Unified Plot Dispatcher
# ============================================================================

def plot_peaks_spectrum(
    method: str = None,
    freqs: Union[np.ndarray, List[np.ndarray]] = None,
    psd: Union[np.ndarray, List[np.ndarray]] = None,
    peaks: np.ndarray = None,
    xmin: float = 1,
    xmax: float = 60,
    show_bands: bool = False,
    bt_object = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot spectral peaks on PSD (spectrum only, no additional panels).
    
    This plots only the power spectral density with detected peaks overlaid.
    
    Parameters
    ----------
    method : str, optional
        Peak extraction method: 'EMD', 'EEMD', 'CEEMDAN', 'FOOOF', 
        'harmonic_recurrence', 'EIMC', 'cepstrum', 'bicoherence', 'PAC',
        'fixed', 'adapt', 'HH1D_max', etc.
        If bt_object is provided, method will be extracted from it.
    freqs : np.ndarray or list of np.ndarray, optional
        Frequency vector(s). Not required if bt_object or IMFs are provided.
    psd : np.ndarray or list of np.ndarray, optional
        PSD data. Not required if bt_object or IMFs are provided.
    peaks : np.ndarray, optional
        Detected peak frequencies. Extracted from bt_object if not provided.
    xmin, xmax : float
        Frequency range
    show_bands : bool, default=False
        Whether to overlay frequency bands
    bt_object : compute_biotuner, optional
        Biotuner object. If provided, attributes will be extracted automatically.
    **kwargs
        Method-specific parameters
    
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    ax : plt.Axes
        Matplotlib axes
    
    Examples
    --------
    >>> # Using biotuner object (recommended)
    >>> bt = compute_biotuner(sf=1000, peaks_function='FOOOF')
    >>> bt.peaks_extraction(data=signal, min_freq=1, max_freq=50)
    >>> fig, ax = plot_peaks_spectrum(bt_object=bt, show_bands=True)
    >>> 
    >>> # Manual mode (backward compatible)
    >>> fig, ax = plot_peaks_spectrum('FOOOF', freqs, psd, peaks, show_bands=True)
    """
    # If biotuner object provided, extract attributes
    if bt_object is not None:
        # Extract method from peaks_function
        if method is None:
            method = bt_object.peaks_function
        
        # Extract peaks
        if peaks is None and hasattr(bt_object, 'peaks'):
            peaks = bt_object.peaks
        
        # Extract freqs and psd (for most methods)
        if freqs is None and hasattr(bt_object, 'freqs'):
            freqs = bt_object.freqs
        if psd is None and hasattr(bt_object, 'psd'):
            psd = bt_object.psd
        
        # Extract method-specific attributes
        # Cepstrum
        if method == 'cepstrum':
            if 'quefrency_vector' not in kwargs and hasattr(bt_object, 'quefrency_vector'):
                kwargs['quefrency_vector'] = bt_object.quefrency_vector
            if 'cepstrum' not in kwargs and hasattr(bt_object, 'cepstrum'):
                kwargs['cepstrum'] = bt_object.cepstrum
        
        # EMD methods
        if method in ['EMD', 'EEMD', 'CEEMDAN', 'EMD_FOOOF']:
            if 'IMFs' not in kwargs and hasattr(bt_object, 'IMFs'):
                kwargs['IMFs'] = bt_object.IMFs
            if 'sf' not in kwargs and hasattr(bt_object, 'sf'):
                kwargs['sf'] = bt_object.sf
        
        # Harmonic recurrence
        if method == 'harmonic_recurrence':
            if 'harm_peaks_fit' not in kwargs and hasattr(bt_object, 'harm_peaks_fit'):
                kwargs['harm_peaks_fit'] = bt_object.harm_peaks_fit
        
        # EIMC
        if method == 'EIMC':
            if 'EIMC_all' not in kwargs and hasattr(bt_object, 'EIMC_all'):
                kwargs['EIMC_all'] = bt_object.EIMC_all
    
    # Ensure method is provided
    if method is None:
        raise ValueError("Either 'method' parameter or 'bt_object' must be provided")
    
    # Ensure peaks are provided
    if peaks is None:
        raise ValueError("Peaks not found. Ensure bt_object has peaks or provide peaks parameter.")
    
    # EMD-based methods
    if method in ['EMD', 'EEMD', 'CEEMDAN', 'EMD_FOOOF']:
        # Check if IMFs are provided directly in kwargs
        if 'IMFs' in kwargs:
            result = plot_emd_peaks(peaks=peaks, xmin=xmin, xmax=xmax, 
                                   show_bands=show_bands, **kwargs)
        else:
            result = plot_emd_peaks(freqs, psd, peaks, xmin=xmin, xmax=xmax, 
                                   show_bands=show_bands, **kwargs)
    
    # EIMC - use specialized visualization for intermodulation components
    elif method == 'EIMC':
        if 'EIMC_all' in kwargs:
            # Use specialized EIMC visualization
            EIMC_all = kwargs.pop('EIMC_all')
            n_pairs = kwargs.pop('n_pairs', 5)
            color = kwargs.pop('color', BIOTUNER_COLORS['dark'])
            
            result = plot_eimc_peaks(
                freqs, psd, peaks, EIMC_all,
                xmin=xmin, xmax=xmax,
                n_pairs=n_pairs,
                show_bands=show_bands,
                color=color,
                **kwargs
            )
        else:
            # Fallback to standard plot if no EIMC data
            result = plot_psd_peaks(freqs, psd, peaks, xmin=xmin, xmax=xmax,
                                   method=method, show_bands=show_bands, **kwargs)
    
    # Harmonic recurrence - use specialized visualization with new style
    elif method == 'harmonic_recurrence':
        if 'harm_peaks_fit' in kwargs:
            # Use specialized harmonic visualization with new styling
            harm_peaks_fit = kwargs.pop('harm_peaks_fit')
            n_peaks = kwargs.pop('n_peaks', 5)
            color = kwargs.pop('color', BIOTUNER_COLORS['dark'])
            
            result = plot_harmonic_peaks(
                freqs, psd, harm_peaks_fit,
                xmin=xmin, xmax=xmax,
                n_peaks=n_peaks,
                show_bands=show_bands,
                color=color,
                selected_peaks=peaks,
                **kwargs
            )
        else:
            # Fallback to standard plot if no harmonic fit data
            result = plot_psd_peaks(freqs, psd, peaks, xmin=xmin, xmax=xmax,
                                   method=method, show_bands=show_bands, **kwargs)
    
    # Cepstrum - use specialized quefrency-based visualization
    elif method == 'cepstrum':
        # For cepstrum, we expect 'quefrency_vector' and 'cepstrum' in kwargs
        if 'quefrency_vector' in kwargs and 'cepstrum' in kwargs:
            quefrency_vector = kwargs.pop('quefrency_vector')
            cepstrum = kwargs.pop('cepstrum')
            result = plot_cepstrum_peaks(quefrency_vector, cepstrum, peaks, 
                                        xmin=xmin, xmax=xmax,
                                        method=method, show_bands=show_bands, **kwargs)
        else:
            # Fallback to standard plot if quefrency data not provided
            result = plot_psd_peaks(freqs, psd, peaks, xmin=xmin, xmax=xmax,
                                   method=method, show_bands=show_bands, **kwargs)
    
    # All other methods (FOOOF, fixed, adapt, EIMC, PAC, bicoherence, HH1D_max, etc.)
    else:
        result = plot_psd_peaks(freqs, psd, peaks, xmin=xmin, xmax=xmax,
                               method=method, show_bands=show_bands, **kwargs)
    
    return result


def plot_peaks_amplitude(
    peaks: np.ndarray,
    amps: np.ndarray = None,
    xmin: float = 1,
    xmax: float = 60,
    bt_object = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot peak amplitude distribution as a bar chart.
    
    Parameters
    ----------
    peaks : np.ndarray
        Peak frequencies
    amps : np.ndarray, optional
        Peak amplitudes. Extracted from bt_object if not provided.
    xmin, xmax : float
        Frequency range
    bt_object : compute_biotuner, optional
        Biotuner object for extracting attributes
    **kwargs
        Additional parameters for _plot_peak_amplitude_distribution
    
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    ax : plt.Axes
        Matplotlib axes
    
    Examples
    --------
    >>> bt = compute_biotuner(sf=1000)
    >>> bt.peaks_extraction(data=signal)
    >>> fig, ax = plot_peaks_amplitude(bt_object=bt)
    """
    # Extract from bt_object if provided
    if bt_object is not None:
        if peaks is None and hasattr(bt_object, 'peaks'):
            peaks = bt_object.peaks
        if amps is None and hasattr(bt_object, 'amps'):
            amps = bt_object.amps
    
    if peaks is None:
        raise ValueError("Peaks must be provided or bt_object must have peaks attribute")
    if amps is None:
        raise ValueError("Amplitudes must be provided or bt_object must have amps attribute")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot amplitude distribution
    _plot_peak_amplitude_distribution(ax, peaks, amps, xmin, xmax, **kwargs)
    
    return fig, ax


def plot_peaks_matrix(
    peaks: np.ndarray,
    metric: str = 'harmsim',
    bt_object = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot peak ratios harmonicity matrix.
    
    Parameters
    ----------
    peaks : np.ndarray
        Peak frequencies
    metric : str, default='harmsim'
        Metric for matrix: 'harmsim', 'cons', 'tenney', 'denom', 'subharm_tension'
    bt_object : compute_biotuner, optional
        Biotuner object for extracting peaks
    **kwargs
        Additional parameters for _plot_peak_ratios_matrix
    
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    ax : plt.Axes
        Matplotlib axes
    
    Examples
    --------
    >>> bt = compute_biotuner(sf=1000)
    >>> bt.peaks_extraction(data=signal)
    >>> fig, ax = plot_peaks_matrix(bt_object=bt, metric='harmsim')
    """
    # Extract from bt_object if provided
    if bt_object is not None and peaks is None:
        if hasattr(bt_object, 'peaks'):
            peaks = bt_object.peaks
    
    if peaks is None:
        raise ValueError("Peaks must be provided or bt_object must have peaks attribute")
    
    if len(peaks) < 2:
        raise ValueError("At least 2 peaks required for matrix visualization")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Plot matrix
    _plot_peak_ratios_matrix(ax, peaks, metric, **kwargs)
    
    return fig, ax


def plot_peaks_summary(
    method: str = None,
    freqs: Union[np.ndarray, List[np.ndarray]] = None,
    psd: Union[np.ndarray, List[np.ndarray]] = None,
    peaks: np.ndarray = None,
    xmin: float = 1,
    xmax: float = 60,
    show_bands: bool = False,
    show_matrix: bool = False,
    matrix_metric: str = 'harmsim',
    bt_object = None,
    **kwargs
) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
    """
    Universal summary plotting function for peak extraction results.
    
    Plots a comprehensive summary with spectrum, amplitude distribution, and optionally harmonicity matrix.
    For individual plots, use plot_peaks_spectrum(), plot_peaks_amplitude(), or plot_peaks_matrix().
    
    Automatically dispatches to the appropriate plotting function based on method.
    Can accept a biotuner object directly for simplified usage.
    
    Parameters
    ----------
    method : str, optional
        Peak extraction method: 'EMD', 'EEMD', 'CEEMDAN', 'FOOOF', 
        'harmonic_recurrence', 'EIMC', 'cepstrum', 'bicoherence', 'PAC',
        'fixed', 'adapt', 'HH1D_max', etc.
        If bt_object is provided, method will be extracted from it.
    freqs : np.ndarray or list of np.ndarray, optional
        Frequency vector(s). Not required if bt_object or IMFs are provided.
    psd : np.ndarray or list of np.ndarray, optional
        PSD data. Not required if bt_object or IMFs are provided.
    peaks : np.ndarray, optional
        Detected peak frequencies. Extracted from bt_object if not provided.
    xmin, xmax : float
        Frequency range
    show_bands : bool, default=False
        Whether to overlay frequency bands
    show_matrix : bool, default=False
        Whether to show peak ratios harmonicity matrix in a second panel
    matrix_metric : str, default='harmsim'
        Metric to use for matrix computation:
        - 'harmsim': Harmonic similarity (higher = more harmonic)
        - 'cons': Consonance (higher = more consonant)
        - 'tenney': Tenney height (higher = more dissonant)
        - 'denom': Denominator complexity (higher = more dissonant)
        - 'subharm_tension': Subharmonic tension
    bt_object : compute_biotuner, optional
        Biotuner object. If provided, attributes will be extracted automatically.
    **kwargs
        Method-specific parameters
    
    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    ax : plt.Axes or np.ndarray
        Matplotlib axes (single or array if show_matrix=True)
    
    Examples
    --------
    >>> # Using biotuner object (recommended)
    >>> bt = compute_biotuner(sf=1000, peaks_function='FOOOF')
    >>> bt.peaks_extraction(data=signal, min_freq=1, max_freq=50)
    >>> fig, ax = plot_peaks(bt_object=bt, show_bands=True, show_matrix=True)
    >>> 
    >>> # Manual mode (backward compatible)
    >>> fig, ax = plot_peaks('FOOOF', freqs, psd, peaks, show_bands=True)
    >>> 
    >>> # With harmonicity matrix
    >>> fig, axes = plot_peaks(bt_object=bt, show_matrix=True, matrix_metric='harmsim')
    >>> plt.show()
    """
    # If biotuner object provided, extract attributes
    if bt_object is not None:
        # Extract method from peaks_function
        if method is None:
            method = bt_object.peaks_function
        
        # Extract peaks
        if peaks is None and hasattr(bt_object, 'peaks'):
            peaks = bt_object.peaks
        
        # Extract freqs and psd (for most methods)
        if freqs is None and hasattr(bt_object, 'freqs'):
            freqs = bt_object.freqs
        if psd is None and hasattr(bt_object, 'psd'):
            psd = bt_object.psd
        
        # Extract method-specific attributes
        # Cepstrum
        if method == 'cepstrum':
            if 'quefrency_vector' not in kwargs and hasattr(bt_object, 'quefrency_vector'):
                kwargs['quefrency_vector'] = bt_object.quefrency_vector
            if 'cepstrum' not in kwargs and hasattr(bt_object, 'cepstrum'):
                kwargs['cepstrum'] = bt_object.cepstrum
        
        # EMD methods
        if method in ['EMD', 'EEMD', 'CEEMDAN', 'EMD_FOOOF']:
            if 'IMFs' not in kwargs and hasattr(bt_object, 'IMFs'):
                kwargs['IMFs'] = bt_object.IMFs
            if 'sf' not in kwargs and hasattr(bt_object, 'sf'):
                kwargs['sf'] = bt_object.sf
        
        # Harmonic recurrence
        if method == 'harmonic_recurrence':
            if 'harm_peaks_fit' not in kwargs and hasattr(bt_object, 'harm_peaks_fit'):
                kwargs['harm_peaks_fit'] = bt_object.harm_peaks_fit
        
        # EIMC
        if method == 'EIMC':
            if 'EIMC_all' not in kwargs and hasattr(bt_object, 'EIMC_all'):
                kwargs['EIMC_all'] = bt_object.EIMC_all
    
    # Ensure method is provided
    if method is None:
        raise ValueError("Either 'method' parameter or 'bt_object' must be provided")
    
    # Ensure peaks are provided
    if peaks is None:
        raise ValueError("Peaks not found. Ensure bt_object has peaks or provide peaks parameter.")
    
    # Initialize saved variables for show_matrix panel
    EIMC_all_saved = None
    n_pairs_saved = None
    color_saved = None
    harm_peaks_fit_saved = None
    n_peaks_saved = None
    quefrency_vector_saved = None
    cepstrum_saved = None
    title_saved = None
    
    # EMD-based methods
    if method in ['EMD', 'EEMD', 'CEEMDAN', 'EMD_FOOOF']:
        # Check if IMFs are provided directly in kwargs
        if 'IMFs' in kwargs:
            result = plot_emd_peaks(peaks=peaks, xmin=xmin, xmax=xmax, 
                                   show_bands=show_bands, **kwargs)
        else:
            result = plot_emd_peaks(freqs, psd, peaks, xmin=xmin, xmax=xmax, 
                                   show_bands=show_bands, **kwargs)
    
    # EIMC - use specialized visualization for intermodulation components
    elif method == 'EIMC':
        if 'EIMC_all' in kwargs:
            # Save EIMC parameters before popping (needed for matrix panel)
            EIMC_all_saved = kwargs.get('EIMC_all')
            n_pairs_saved = kwargs.get('n_pairs', 5)
            color_saved = kwargs.get('color', BIOTUNER_COLORS['dark'])
            
            # Use specialized EIMC visualization
            EIMC_all = kwargs.pop('EIMC_all')
            n_pairs = kwargs.pop('n_pairs', 5)  # Pop to avoid duplicate
            color = kwargs.pop('color', BIOTUNER_COLORS['dark'])  # Pop to avoid duplicate
            
            result = plot_eimc_peaks(
                freqs, psd, peaks, EIMC_all,
                xmin=xmin, xmax=xmax,
                n_pairs=n_pairs,
                show_bands=show_bands,
                color=color,
                **kwargs
            )
        else:
            # Fallback to standard plot if no EIMC data
            EIMC_all_saved = None
            n_pairs_saved = None
            color_saved = None
            result = plot_psd_peaks(freqs, psd, peaks, xmin=xmin, xmax=xmax,
                                   method=method, show_bands=show_bands, **kwargs)
    
    # Harmonic recurrence - use specialized visualization with new style
    elif method == 'harmonic_recurrence':
        if 'harm_peaks_fit' in kwargs:
            # Save harmonic parameters before popping (needed for matrix panel)
            harm_peaks_fit_saved = kwargs.get('harm_peaks_fit')
            n_peaks_saved = kwargs.get('n_peaks', 5)
            color_saved = kwargs.get('color', BIOTUNER_COLORS['dark'])
            
            # Use specialized harmonic visualization with new styling
            harm_peaks_fit = kwargs.pop('harm_peaks_fit')
            n_peaks = kwargs.pop('n_peaks', 5)  # Pop to avoid duplicate
            color = kwargs.pop('color', BIOTUNER_COLORS['dark'])  # Pop to avoid duplicate
            
            result = plot_harmonic_peaks(
                freqs, psd, harm_peaks_fit,
                xmin=xmin, xmax=xmax,
                n_peaks=n_peaks,
                show_bands=show_bands,
                color=color,
                selected_peaks=peaks,  # ← FIX: Pass bt.peaks to filter harm_peaks_fit
                **kwargs
            )
        else:
            # Fallback to standard plot if no harmonic fit data
            harm_peaks_fit_saved = None
            n_peaks_saved = None
            color_saved = None
            result = plot_psd_peaks(freqs, psd, peaks, xmin=xmin, xmax=xmax,
                                   method=method, show_bands=show_bands, **kwargs)
    
    # Cepstrum - use specialized quefrency-based visualization
    elif method == 'cepstrum':
        # For cepstrum, we expect 'quefrency_vector' and 'cepstrum' in kwargs
        if 'quefrency_vector' in kwargs and 'cepstrum' in kwargs:
            # Save for matrix panel before popping
            quefrency_vector_saved = kwargs.get('quefrency_vector')
            cepstrum_saved = kwargs.get('cepstrum')
            title_saved = kwargs.get('title')
            
            quefrency_vector = kwargs.pop('quefrency_vector')
            cepstrum = kwargs.pop('cepstrum')
            result = plot_cepstrum_peaks(quefrency_vector, cepstrum, peaks, 
                                        xmin=xmin, xmax=xmax,
                                        method=method, show_bands=show_bands, **kwargs)
        else:
            # Fallback to standard plot if quefrency data not provided
            quefrency_vector_saved = None
            cepstrum_saved = None
            title_saved = None
            result = plot_psd_peaks(freqs, psd, peaks, xmin=xmin, xmax=xmax,
                                   method=method, show_bands=show_bands, **kwargs)
    
    # All other methods (FOOOF, fixed, adapt, EIMC, PAC, bicoherence, HH1D_max, etc.)
    else:
        result = plot_psd_peaks(freqs, psd, peaks, xmin=xmin, xmax=xmax,
                               method=method, show_bands=show_bands, **kwargs)
    
    # If matrix requested, add second and third panels
    if show_matrix and len(peaks) > 1:
        fig_orig, ax_orig = result
        
        # Get amplitudes from bt_object or kwargs
        amps = None
        if bt_object is not None and hasattr(bt_object, 'amps'):
            amps = bt_object.amps
        elif 'amps' in kwargs:
            amps = kwargs['amps']
        
        # Create new figure with 3-panel layout:
        # Top: wide PSD plot
        # Bottom-left: amplitude distribution
        # Bottom-right: harmonicity matrix
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], width_ratios=[1, 1], 
                             hspace=0.35, wspace=0.3)
        
        # Top panel (spanning both columns) - PSD plot
        ax1 = fig.add_subplot(gs[0, :])
        plt.close(fig_orig)
        
        # Re-plot on the new axis to preserve all elements properly
        if method in ['EMD', 'EEMD', 'CEEMDAN', 'EMD_FOOOF']:
            if 'IMFs' in kwargs:
                plot_emd_peaks(peaks=peaks, xmin=xmin, xmax=xmax, 
                              show_bands=show_bands, ax=ax1, **kwargs)
            else:
                plot_emd_peaks(freqs, psd, peaks, xmin=xmin, xmax=xmax, 
                              show_bands=show_bands, ax=ax1, **kwargs)
        elif method == 'harmonic_recurrence':
            if harm_peaks_fit_saved is not None:
                plot_harmonic_peaks(
                    freqs, psd, harm_peaks_fit_saved,
                    xmin=xmin, xmax=xmax,
                    n_peaks=n_peaks_saved,
                    show_bands=show_bands,
                    color=color_saved,
                    selected_peaks=peaks,  # ← CRITICAL: Pass bt.peaks here too!
                    ax=ax1,
                    **kwargs
                )
            else:
                plot_psd_peaks(freqs, psd, peaks, xmin=xmin, xmax=xmax,
                              method=method, show_bands=show_bands, ax=ax1, **kwargs)
        elif method == 'EIMC':
            if EIMC_all_saved is not None:
                plot_eimc_peaks(
                    freqs, psd, peaks, EIMC_all_saved,
                    xmin=xmin, xmax=xmax,
                    n_pairs=n_pairs_saved,
                    show_bands=show_bands,
                    color=color_saved,
                    ax=ax1,
                    **kwargs
                )
            else:
                plot_psd_peaks(freqs, psd, peaks, xmin=xmin, xmax=xmax,
                              method=method, show_bands=show_bands, ax=ax1, **kwargs)
        elif method == 'cepstrum':
            if quefrency_vector_saved is not None and cepstrum_saved is not None:
                plot_cepstrum_peaks(quefrency_vector_saved, cepstrum_saved, 
                                   peaks, xmin=xmin, xmax=xmax,
                                   title=title_saved, show_bands=show_bands, ax=ax1)
            else:
                # Should not happen if bt_object was used correctly
                if freqs is not None and psd is not None:
                    plot_psd_peaks(freqs, psd, peaks, xmin=xmin, xmax=xmax,
                                  method=method, show_bands=show_bands, ax=ax1)
        else:
            plot_psd_peaks(freqs, psd, peaks, xmin=xmin, xmax=xmax,
                          method=method, show_bands=show_bands, ax=ax1, **kwargs)
        
        # Bottom-left panel - amplitude distribution
        ax2 = fig.add_subplot(gs[1, 0])
        
        # Track which peaks to use for the matrix (should match amplitude distribution)
        matrix_peaks = peaks  # Default for all methods
        
        if amps is not None:
            # For EIMC, show BASE PEAKS from pairs (not the frequent peaks)
            if method == 'EIMC' and EIMC_all_saved is not None:
                # Extract unique base peaks from the top n_pairs
                base_peaks_set = set()
                for peak_pair in EIMC_all_saved['peaks'][:n_pairs_saved]:
                    if len(peak_pair) >= 2:
                        base_peaks_set.add(peak_pair[0])
                        base_peaks_set.add(peak_pair[1])
                
                # Convert to sorted array
                eimc_base_peaks = np.array(sorted(base_peaks_set))
                matrix_peaks = eimc_base_peaks  # Use base peaks for matrix too
                
                # Find amplitudes for these base peaks from the original peaks/amps
                eimc_base_amps = []
                for base_peak in eimc_base_peaks:
                    # Find closest match in peaks array (within 0.5 Hz tolerance)
                    idx = np.argmin(np.abs(peaks - base_peak))
                    if abs(peaks[idx] - base_peak) < 0.5:
                        eimc_base_amps.append(amps[idx])
                    else:
                        # If not found, use PSD value at that frequency
                        freq_idx = np.argmin(np.abs(freqs - base_peak))
                        eimc_base_amps.append(psd[freq_idx])
                
                eimc_base_amps = np.array(eimc_base_amps)
                
                _plot_peak_amplitude_distribution(ax2, eimc_base_peaks, eimc_base_amps, 
                                                 xmin, xmax,
                                                 EIMC_all=EIMC_all_saved,
                                                 n_pairs=n_pairs_saved)
            # For harmonic_recurrence, use bt.peaks (selected peaks by harmonic recurrence)
            # These are the peaks shown as solid lines in the top plot
            elif method == 'harmonic_recurrence' and harm_peaks_fit_saved is not None:
                # Use bt.peaks directly - these are the selected peaks we want to show
                matrix_peaks = peaks  # Use bt.peaks for matrix
                
                # Get amplitudes for these peaks from PSD
                peak_amps = []
                for peak_freq in peaks:
                    freq_idx = np.argmin(np.abs(freqs - peak_freq))
                    peak_amps.append(psd[freq_idx])
                
                _plot_peak_amplitude_distribution(ax2, peaks, 
                                                 np.array(peak_amps), 
                                                 xmin, xmax,
                                                 harm_peaks_fit=harm_peaks_fit_saved)
            else:
                # All other methods: use bt.peaks directly
                _plot_peak_amplitude_distribution(ax2, peaks, amps, xmin, xmax)
        else:
            # Fallback: create dummy amplitudes if not available
            ax2.text(0.5, 0.5, 'Amplitudes not available', 
                    ha='center', va='center', transform=ax2.transAxes,
                    fontsize=14, color='gray')
            ax2.set_xlim([xmin, xmax])
            ax2.set_xlabel('Frequency (Hz)', fontsize=14)
            ax2.set_ylabel('Amplitude', fontsize=14)
            ax2.set_title('Peak Amplitude Distribution', fontsize=16, fontweight='bold')
        
        # Bottom-right panel - matrix
        # Use the same peaks as shown in amplitude distribution for consistency
        ax3 = fig.add_subplot(gs[1, 1])
        _plot_peak_ratios_matrix(ax3, matrix_peaks, matrix_metric)
        
        return fig, np.array([ax1, ax2, ax3])
    
    return result


# Backward compatibility alias
plot_peaks = plot_peaks_summary


# ============================================================================
# Tuning Visualization Functions
# ============================================================================

def plot_tuning_dissonance(
    ratio_vec: np.ndarray,
    diss_curve: np.ndarray,
    diss_scale: List[float] = None,
    intervals: List[Tuple[int, int]] = None,
    n_tet_grid: Optional[int] = None,
    max_ratio: float = 2,
    show_intervals: bool = True,
    show_tet_grid: bool = False,
    denom: int = 1000,
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot dissonance curve with unified biotuner styling.
    
    Parameters
    ----------
    ratio_vec : np.ndarray
        Frequency ratios for x-axis
    diss_curve : np.ndarray
        Dissonance values
    diss_scale : list of float, optional
        Scale derived from dissonance minima
    intervals : list of tuples, optional
        Interval ratios as (numerator, denominator) pairs.
        If None and diss_scale is provided, will compute from ratios.
    n_tet_grid : int, optional
        N-TET reference grid (e.g., 12 for 12-TET)
    max_ratio : float, default=2
        Maximum ratio to display
    show_intervals : bool, default=True
        Show vertical lines at scale intervals
    show_tet_grid : bool, default=False
        Show N-TET reference grid
    denom : int, default=1000
        Denominator limit for fraction conversion
    figsize : tuple, optional
        Figure size
    ax : plt.Axes, optional
        Existing axes
    
    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    from fractions import Fraction
    
    set_biotuner_style()
    
    if figsize is None:
        figsize = (12, 7)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Plot dissonance curve
    ax.plot(ratio_vec, diss_curve, color=BIOTUNER_COLORS['danger'], 
           linewidth=2.5, label='Dissonance')
    
    # Compute intervals from ratios if not provided
    if intervals is None and diss_scale is not None:
        intervals = [(Fraction(r).limit_denominator(denom).numerator,
                     Fraction(r).limit_denominator(denom).denominator)
                    for r in diss_scale]
    
    # Add scale intervals
    if show_intervals and intervals is not None:
        for n, d in intervals:
            ratio = n / d
            if 1.0 <= ratio <= max_ratio:
                ax.axvline(ratio, color=BIOTUNER_COLORS['dark'], 
                          linestyle='--', linewidth=1.5, alpha=0.6)
    
    # Add N-TET grid
    if show_tet_grid and n_tet_grid is not None:
        from biotuner.biotuner_utils import NTET_ratios
        tet_ratios = NTET_ratios(n_tet_grid, max_ratio=max_ratio)
        for ratio in tet_ratios:
            if 1.0 < ratio <= max_ratio:
                ax.axvline(ratio, color=BIOTUNER_COLORS['secondary'], 
                          linestyle=':', linewidth=1, alpha=0.4)
    
    # Styling
    ax.set_xlim([1.0, max_ratio])
    ax.set_xlabel('Frequency Ratio', fontsize=16, fontweight='normal')
    ax.set_ylabel('Sensory Dissonance', fontsize=16, fontweight='normal')
    ax.set_title('Dissonance Curve', fontsize=20, fontweight='bold', pad=15)
    ax.legend(fontsize=13, framealpha=0.95)
    ax.grid(True, alpha=0.25)
    
    # Add x-axis labels for intervals if available
    if show_intervals and intervals is not None:
        ax.set_xticks([n/d for n, d in intervals if 1.0 <= n/d <= max_ratio])
        ax.set_xticklabels([f'{n}/{d}' for n, d in intervals if 1.0 <= n/d <= max_ratio], 
                          fontsize=13, rotation=45)
    
    plt.tight_layout()
    return fig, ax


def plot_tuning_entropy(
    ratio_vec: np.ndarray,
    entropy_curve: np.ndarray,
    entropy_scale: List[float] = None,
    max_ratio: float = 2,
    show_minima: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot harmonic entropy curve with unified biotuner styling.
    
    Parameters
    ----------
    ratio_vec : np.ndarray
        Frequency ratios for x-axis
    entropy_curve : np.ndarray
        Harmonic entropy values
    entropy_scale : list of float, optional
        Scale derived from entropy minima
    max_ratio : float, default=2
        Maximum ratio to display
    show_minima : bool, default=True
        Show markers at local minima
    figsize : tuple, optional
        Figure size
    ax : plt.Axes, optional
        Existing axes
    
    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    set_biotuner_style()
    
    if figsize is None:
        figsize = (12, 7)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Plot entropy curve
    ax.plot(ratio_vec, entropy_curve, color=BIOTUNER_COLORS['primary'], 
           linewidth=2.5, label='Harmonic Entropy')
    
    # Mark local minima
    if show_minima and entropy_scale is not None:
        for ratio in entropy_scale:
            if 1.0 <= ratio <= max_ratio:
                # Find closest index in ratio_vec
                idx = np.argmin(np.abs(ratio_vec - ratio))
                ax.plot(ratio, entropy_curve[idx], 'o', 
                       color=BIOTUNER_COLORS['accent'], 
                       markersize=10, markeredgecolor=BIOTUNER_COLORS['dark'],
                       markeredgewidth=1.5, label='Local Minima' if ratio == entropy_scale[0] else '')
                ax.axvline(ratio, color=BIOTUNER_COLORS['dark'], 
                          linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Styling
    ax.set_xlim([1.0, max_ratio])
    ax.set_xlabel('Frequency Ratio', fontsize=16, fontweight='normal')
    ax.set_ylabel('Harmonic Entropy', fontsize=16, fontweight='normal')
    ax.set_title('Harmonic Entropy Curve', fontsize=20, fontweight='bold', pad=15)
    
    # Only show legend if minima are displayed
    if show_minima and entropy_scale is not None:
        ax.legend(fontsize=13, framealpha=0.95)
    
    ax.grid(True, alpha=0.25)
    
    plt.tight_layout()
    return fig, ax


def plot_tuning_scale(
    tuning: List[float],
    max_denom: int = 100,
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot tuning scale as bar chart showing cents and ratios.
    
    Parameters
    ----------
    tuning : list of float
        Tuning scale as frequency ratios
    max_denom : int, default=100
        Maximum denominator for fraction display
    figsize : tuple, optional
        Figure size
    ax : plt.Axes, optional
        Existing axes to plot on
    
    Returns
    -------
    fig, ax : matplotlib figure and axes
    
    Examples
    --------
    >>> bt = compute_biotuner(sf=1000)
    >>> bt.peaks_extraction(data=signal)
    >>> tuning = bt.harmonic_tuning()
    >>> fig, ax = plot_tuning_scale(tuning)
    """
    from fractions import Fraction
    
    set_biotuner_style()
    
    if figsize is None:
        figsize = (12, 6)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Convert ratios to cents
    cents = [1200 * np.log2(ratio) if ratio > 0 else 0 for ratio in tuning]
    n_notes = len(tuning)
    colors = get_color_palette('biotuner_gradient', n_colors=n_notes)
    
    bars = ax.bar(range(n_notes), cents, color=colors, 
                   edgecolor=BIOTUNER_COLORS['dark'], linewidth=1.5, alpha=0.8)
    
    # Add values on top with fractions
    for i, (cent, ratio) in enumerate(zip(cents, tuning)):
        frac = Fraction(ratio).limit_denominator(max_denom)
        ratio_str = f'{frac.numerator}/{frac.denominator}' if frac.denominator != 1 else str(frac.numerator)
        ax.text(i, cent + 20, f'{cent:.0f}¢\n{ratio_str}', 
                ha='center', va='bottom', fontsize=9, fontweight='semibold')
    
    ax.set_xlabel('Interval Index', fontsize=16, fontweight='normal')
    ax.set_ylabel('Cents', fontsize=16, fontweight='normal')
    ax.set_title('Tuning Scale', fontsize=20, fontweight='bold', pad=15)
    ax.set_xticks(range(n_notes))
    ax.set_xticklabels([f'{i+1}' for i in range(n_notes)], fontsize=13)
    ax.axhline(1200, color=BIOTUNER_COLORS['dark'], linestyle='--', 
               linewidth=1.5, alpha=0.5, label='Octave')
    ax.legend(fontsize=12, framealpha=0.95)
    ax.grid(True, alpha=0.25, axis='y')
    
    plt.tight_layout()
    return fig, ax


def plot_tuning_matrix(
    tuning: List[float],
    metric: str = 'harmsim',
    ratio_type: str = 'all',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    max_denom: int = 100,
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot consonance matrix for tuning scale.
    
    Parameters
    ----------
    tuning : list of float
        Tuning scale as frequency ratios
    metric : str, default='harmsim'
        Consonance metric: 'harmsim', 'cons', 'tenney', 'denom', 'subharm_tension'
    ratio_type : str, default='all'
        Ratio type: 'all', 'pos_harm', 'sub_harm'
    vmin, vmax : float, optional
        Color scale limits
    max_denom : int, default=100
        Maximum denominator for fractions
    figsize : tuple, optional
        Figure size
    ax : plt.Axes, optional
        Existing axes
    
    Returns
    -------
    fig, ax : matplotlib figure and axes
    
    Examples
    --------
    >>> fig, ax = plot_tuning_matrix(tuning, metric='harmsim')
    """
    from biotuner.metrics import dyad_similarity, compute_consonance
    from fractions import Fraction
    from matplotlib.colors import LinearSegmentedColormap
    
    set_biotuner_style()
    
    if figsize is None:
        figsize = (8, 7)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Metric functions
    metric_functions = {
        'harmsim': lambda r: dyad_similarity(float(Fraction(r).limit_denominator(max_denom))),
        'cons': lambda r: compute_consonance(float(Fraction(r).limit_denominator(max_denom))),
        'tenney': lambda r: 1 / (1 + np.log2(Fraction(r).limit_denominator(max_denom).numerator * 
                                              Fraction(r).limit_denominator(max_denom).denominator)),
        'denom': lambda r: 1 / (1 + Fraction(r).limit_denominator(max_denom).denominator),
    }
    
    metric_labels = {
        'harmsim': 'Harmonic Similarity',
        'cons': 'Consonance',
        'tenney': 'Tenney Height (inv.)',
        'denom': 'Denominator (inv.)',
        'subharm_tension': 'Subharm Tension (inv.)'
    }
    
    # Compute consonance matrix
    n_values = len(tuning)
    cons_matrix = np.zeros((n_values, n_values))
    
    if metric in metric_functions:
        metric_fn = metric_functions[metric]
        
        for i in range(n_values):
            for j in range(n_values):
                if i == j:
                    cons_matrix[i, j] = 0
                else:
                    ratio = tuning[j] / tuning[i]
                    
                    if ratio_type == 'all':
                        actual_ratio = ratio if ratio >= 1 else 1/ratio
                        cons_matrix[i, j] = metric_fn(actual_ratio)
                    elif ratio_type == 'pos_harm':
                        if ratio > 1:
                            cons_matrix[i, j] = metric_fn(ratio)
                    elif ratio_type == 'sub_harm':
                        if ratio < 1:
                            cons_matrix[i, j] = metric_fn(1/ratio)
        
        if metric == 'cons':
            cons_matrix = cons_matrix * 100
        elif metric in ['tenney', 'denom']:
            cons_matrix = cons_matrix * 100
    
    # Auto-adjust vmin/vmax
    if vmin is None or vmax is None:
        non_zero = cons_matrix[cons_matrix != 0]
        if len(non_zero) > 0:
            if vmin is None:
                vmin = np.percentile(non_zero, 5)
            if vmax is None:
                vmax = np.percentile(non_zero, 95)
        else:
            vmin = 0
            vmax = 100
    
    # Colormap
    colors_list = ['#8B4513', '#CD5C5C', '#F4A460', '#FFD700', '#90EE90', '#48D1CC', '#40E0D0']
    custom_cmap = LinearSegmentedColormap.from_list('orange_to_turquoise', colors_list)
    cmap = custom_cmap if metric in ['harmsim', 'cons'] else custom_cmap.reversed()
    
    # Plot heatmap
    im = ax.imshow(cons_matrix, cmap=cmap, vmin=vmin, vmax=vmax, 
                    aspect='auto', origin='lower')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(metric_labels.get(metric, 'Metric'), fontsize=14, fontweight='normal')
    cbar.ax.tick_params(labelsize=12)
    
    # Labels
    fraction_labels = []
    for v in tuning:
        frac = Fraction(v).limit_denominator(max_denom)
        if frac.denominator == 1:
            fraction_labels.append(str(frac.numerator))
        else:
            fraction_labels.append(f'{frac.numerator}/{frac.denominator}')
    
    ax.set_xticks(range(n_values))
    ax.set_yticks(range(n_values))
    ax.set_xticklabels(fraction_labels, fontsize=11, rotation=45)
    ax.set_yticklabels(fraction_labels, fontsize=11)
    
    ax.set_xlabel('Tuning Ratio', fontsize=16, fontweight='normal')
    ax.set_ylabel('Tuning Ratio', fontsize=16, fontweight='normal')
    title_suffix = f' ({ratio_type})' if ratio_type != 'all' else ''
    ax.set_title(f'Consonance Matrix{title_suffix}\n{metric_labels.get(metric, metric)}', 
                 fontsize=20, fontweight='bold', pad=15)
    
    plt.tight_layout()
    return fig, ax


def plot_tuning_intervals(
    tuning: List[float],
    max_denom: int = 100,
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot melodic intervals (step sizes) between adjacent notes in tuning.
    
    Parameters
    ----------
    tuning : list of float
        Tuning scale as frequency ratios
    max_denom : int, default=100
        Maximum denominator for fractions
    figsize : tuple, optional
        Figure size
    ax : plt.Axes, optional
        Existing axes
    
    Returns
    -------
    fig, ax : matplotlib figure and axes
    
    Examples
    --------
    >>> fig, ax = plot_tuning_intervals(tuning)
    """
    set_biotuner_style()
    
    if figsize is None:
        figsize = (12, 6)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Compute step sizes (melodic intervals between adjacent notes)
    step_cents = []
    for i in range(len(tuning) - 1):
        step_ratio = tuning[i + 1] / tuning[i]
        step_cents.append(1200 * np.log2(step_ratio))
    
    # Plot step sizes
    step_colors = get_color_palette('biotuner_gradient', n_colors=len(step_cents))
    bars = ax.bar(range(len(step_cents)), step_cents, color=step_colors,
                  edgecolor=BIOTUNER_COLORS['dark'], linewidth=1.5, alpha=0.8)
    
    # Add values on bars
    for i, cent in enumerate(step_cents):
        ax.text(i, cent + 5, f'{cent:.0f}¢', 
               ha='center', va='bottom', fontsize=10, fontweight='semibold')
    
    # Reference lines for common intervals
    ax.axhline(100, color=BIOTUNER_COLORS['secondary'], linestyle=':', 
              linewidth=1.5, alpha=0.5, label='Semitone (100¢)')
    ax.axhline(200, color=BIOTUNER_COLORS['accent'], linestyle=':', 
              linewidth=1.5, alpha=0.5, label='Whole tone (200¢)')
    
    ax.set_xlabel('Step Number', fontsize=16, fontweight='normal')
    ax.set_ylabel('Step Size (cents)', fontsize=16, fontweight='normal')
    ax.set_title('Melodic Intervals (Step Sizes)', fontsize=20, fontweight='bold', pad=15)
    ax.set_xticks(range(len(step_cents)))
    ax.set_xticklabels([f'{i+1}→{i+2}' for i in range(len(step_cents))], 
                       fontsize=11, rotation=45)
    ax.legend(fontsize=12, framealpha=0.95, loc='upper right')
    ax.grid(True, alpha=0.25, axis='y')
    
    plt.tight_layout()
    return fig, ax


def plot_tuning_consonance_profile(
    tuning: List[float],
    metric: str = 'harmsim',
    ratio_type: str = 'all',
    max_denom: int = 100,
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot consonance profile showing distribution of consonance values for each scale degree.
    
    Parameters
    ----------
    tuning : list of float
        Tuning scale as frequency ratios
    metric : str, default='harmsim'
        Consonance metric: 'harmsim', 'cons', 'tenney', 'denom', 'subharm_tension'
    ratio_type : str, default='all'
        Ratio type: 'all', 'pos_harm', 'sub_harm'
    max_denom : int, default=100
        Maximum denominator for fractions
    figsize : tuple, optional
        Figure size
    ax : plt.Axes, optional
        Existing axes
    
    Returns
    -------
    fig, ax : matplotlib figure and axes
    
    Examples
    --------
    >>> fig, ax = plot_tuning_consonance_profile(tuning, metric='harmsim')
    """
    from biotuner.metrics import dyad_similarity, compute_consonance
    from fractions import Fraction
    
    set_biotuner_style()
    
    if figsize is None:
        figsize = (12, 6)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Metric functions
    metric_functions = {
        'harmsim': lambda r: dyad_similarity(float(Fraction(r).limit_denominator(max_denom))),
        'cons': lambda r: compute_consonance(float(Fraction(r).limit_denominator(max_denom))),
        'tenney': lambda r: 1 / (1 + np.log2(Fraction(r).limit_denominator(max_denom).numerator * 
                                              Fraction(r).limit_denominator(max_denom).denominator)),
        'denom': lambda r: 1 / (1 + Fraction(r).limit_denominator(max_denom).denominator),
    }
    
    metric_labels = {
        'harmsim': 'Harmonic Similarity',
        'cons': 'Consonance',
        'tenney': 'Tenney Height (inv.)',
        'denom': 'Denominator (inv.)',
        'subharm_tension': 'Subharm Tension (inv.)'
    }
    
    # Compute consonance matrix
    n_values = len(tuning)
    cons_matrix = np.zeros((n_values, n_values))
    
    if metric in metric_functions:
        metric_fn = metric_functions[metric]
        
        for i in range(n_values):
            for j in range(n_values):
                if i == j:
                    cons_matrix[i, j] = 0
                else:
                    ratio = tuning[j] / tuning[i]
                    
                    if ratio_type == 'all':
                        actual_ratio = ratio if ratio >= 1 else 1/ratio
                        cons_matrix[i, j] = metric_fn(actual_ratio)
                    elif ratio_type == 'pos_harm':
                        if ratio > 1:
                            cons_matrix[i, j] = metric_fn(ratio)
                    elif ratio_type == 'sub_harm':
                        if ratio < 1:
                            cons_matrix[i, j] = metric_fn(1/ratio)
        
        if metric == 'cons':
            cons_matrix = cons_matrix * 100
        elif metric in ['tenney', 'denom']:
            cons_matrix = cons_matrix * 100
    
    # Collect consonance distributions for each scale degree
    consonance_distributions = []
    for i in range(len(tuning)):
        row_values = cons_matrix[i, :]
        # Exclude diagonal (self-comparison) and zeros
        non_diag = [row_values[j] for j in range(len(row_values)) if j != i and row_values[j] != 0]
        if len(non_diag) > 0:
            consonance_distributions.append(non_diag)
        else:
            consonance_distributions.append([0])  # Fallback
    
    # Create violin plot
    positions = list(range(len(consonance_distributions)))
    parts = ax.violinplot(
        consonance_distributions,
        positions=positions,
        widths=0.7,
        showmeans=True,
        showextrema=True
    )
    
    # Color the violin plots with gradient
    profile_colors = list(get_color_palette('biotuner_gradient', n_colors=len(consonance_distributions)))
    # Ensure we have the right number of bodies
    n_bodies = len(parts['bodies'])
    for i in range(min(n_bodies, len(profile_colors))):
        parts['bodies'][i].set_facecolor(profile_colors[i])
        parts['bodies'][i].set_alpha(0.7)
        parts['bodies'][i].set_edgecolor(BIOTUNER_COLORS['dark'])
        parts['bodies'][i].set_linewidth(1.5)
    
    # Style the mean/extrema lines
    for partname in ('cmeans', 'cmaxes', 'cmins', 'cbars'):
        if partname in parts:
            vp = parts[partname]
            vp.set_edgecolor(BIOTUNER_COLORS['dark'])
            vp.set_linewidth(1.5)
    
    # Convert tuning to fraction labels for x-axis
    from fractions import Fraction
    fraction_labels = []
    for v in tuning:
        frac = Fraction(v).limit_denominator(max_denom)
        if frac.denominator == 1:
            fraction_labels.append(str(frac.numerator))
        else:
            fraction_labels.append(f'{frac.numerator}/{frac.denominator}')
    
    ax.set_xlabel('Tuning Ratio', fontsize=16, fontweight='normal')
    ax.set_ylabel(f'{metric_labels.get(metric, "Consonance")} Distribution', fontsize=16, fontweight='normal')
    ax.set_title('Consonance Profile (Violin Plot)', fontsize=20, fontweight='bold', pad=15)
    ax.set_xticks(positions)
    ax.set_xticklabels(fraction_labels, fontsize=11, rotation=45)
    ax.grid(True, alpha=0.25, axis='y')
    
    plt.tight_layout()
    return fig, ax


def plot_tuning_curve(
    bt_object=None,
    curve_type: str = 'dissonance',
    ratio_vec: Optional[np.ndarray] = None,
    curve_data: Optional[np.ndarray] = None,
    scale: Optional[List[float]] = None,
    max_ratio: float = 2.0,
    show_minima: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot source curve (dissonance or harmonic entropy) with local minima.
    
    Parameters
    ----------
    bt_object : compute_biotuner, optional
        Biotuner object to extract curve data from
    curve_type : str, default='dissonance'
        Type of curve: 'dissonance' or 'entropy'
    ratio_vec : np.ndarray, optional
        Ratio vector for x-axis. Extracted from bt_object if not provided.
    curve_data : np.ndarray, optional
        Curve values for y-axis. Extracted from bt_object if not provided.
    scale : list of float, optional
        Scale derived from curve minima. Extracted from bt_object if not provided.
    max_ratio : float, default=2.0
        Maximum ratio to display
    show_minima : bool, default=True
        Show markers at local minima
    figsize : tuple, optional
        Figure size
    ax : plt.Axes, optional
        Existing axes
    
    Returns
    -------
    fig, ax : matplotlib figure and axes
    
    Examples
    --------
    >>> # From biotuner object
    >>> fig, ax = plot_tuning_curve(bt_object=bt, curve_type='dissonance')
    >>> 
    >>> # Manual data
    >>> fig, ax = plot_tuning_curve(
    ...     ratio_vec=ratios, 
    ...     curve_data=diss_curve, 
    ...     scale=diss_scale,
    ...     curve_type='dissonance'
    ... )
    """
    set_biotuner_style()
    
    if figsize is None:
        figsize = (12, 7)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Extract data from bt_object if provided
    if bt_object is not None:
        if curve_type == 'dissonance':
            if ratio_vec is None and hasattr(bt_object, 'ratio_diss'):
                ratio_vec = bt_object.ratio_diss
            if curve_data is None and hasattr(bt_object, 'diss'):
                curve_data = bt_object.diss
            if scale is None and hasattr(bt_object, 'diss_scale'):
                scale = bt_object.diss_scale
        elif curve_type == 'entropy':
            if ratio_vec is None and hasattr(bt_object, 'ratio_HE'):
                ratio_vec = bt_object.ratio_HE
            if curve_data is None and hasattr(bt_object, 'HE'):
                curve_data = bt_object.HE
            if scale is None and hasattr(bt_object, 'HE_scale'):
                scale = bt_object.HE_scale
    
    if ratio_vec is None or curve_data is None:
        raise ValueError("Either bt_object or both ratio_vec and curve_data must be provided")
    
    # Plot curve
    if curve_type == 'dissonance':
        color = BIOTUNER_COLORS['danger']
        label = 'Dissonance'
        ylabel = 'Dissonance'
        title = 'Dissonance Curve'
    else:  # entropy
        color = BIOTUNER_COLORS['primary']
        label = 'Harmonic Entropy'
        ylabel = 'Harmonic Entropy'
        title = 'Harmonic Entropy Curve'
    
    ax.plot(ratio_vec, curve_data, color=color, linewidth=2.5, label=label)
    
    # Mark local minima
    if show_minima and scale is not None:
        for ratio in scale:
            if 1.0 <= ratio <= max_ratio:
                # Find closest index in ratio_vec
                idx = np.argmin(np.abs(ratio_vec - ratio))
                ax.plot(ratio, curve_data[idx], 'o', 
                       color=BIOTUNER_COLORS['accent'], 
                       markersize=10, markeredgecolor=BIOTUNER_COLORS['dark'],
                       markeredgewidth=1.5, label='Local Minima' if ratio == scale[0] else '')
                ax.axvline(ratio, color=BIOTUNER_COLORS['dark'], 
                          linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Styling
    ax.set_xlim([1.0, max_ratio])
    ax.set_xlabel('Frequency Ratio', fontsize=16, fontweight='normal')
    ax.set_ylabel(ylabel, fontsize=16, fontweight='normal')
    ax.set_title(title, fontsize=20, fontweight='bold', pad=15)
    
    # Only show legend if minima are displayed
    if show_minima and scale is not None:
        ax.legend(fontsize=13, framealpha=0.95)
    
    ax.grid(True, alpha=0.25)
    
    plt.tight_layout()
    return fig, ax


def plot_tuning_interval_table(
    tuning: List[float],
    max_denom: int = 100,
    tolerance_cents: float = 5.0,
    max_intervals: Optional[int] = None,
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a table showing known musical intervals matched to the tuning scale.
    
    Parameters
    ----------
    tuning : list of float
        Tuning scale as frequency ratios
    max_denom : int, default=100
        Maximum denominator for fractions
    tolerance_cents : float, default=5.0
        Tolerance in cents for matching intervals
    max_intervals : int, optional
        Maximum number of intervals to display. If None, show all matches.
    figsize : tuple, optional
        Figure size
    ax : plt.Axes, optional
        Existing axes
    
    Returns
    -------
    fig, ax : matplotlib figure and axes
    
    Examples
    --------
    >>> fig, ax = plot_tuning_interval_table(tuning, max_intervals=15)
    """
    from biotuner.dictionaries import interval_catalog
    from fractions import Fraction
    
    set_biotuner_style()
    
    if figsize is None:
        figsize = (12, 8)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Clear the axis
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Find matching known intervals (within tolerance)
    matched_intervals = []
    
    for ratio in tuning:
        ratio_cents = 1200 * np.log2(ratio) if ratio > 0 else 0
        
        # Check against interval catalog
        for interval_name, interval_ratio in interval_catalog:
            try:
                # Convert sympy ratio to float
                catalog_ratio = float(interval_ratio)
                catalog_cents = 1200 * np.log2(catalog_ratio) if catalog_ratio > 0 else 0
                
                # Check if within tolerance
                if abs(ratio_cents - catalog_cents) < tolerance_cents:
                    matched_intervals.append((ratio, interval_name, abs(ratio_cents - catalog_cents)))
                    break  # Only match first interval found
            except:
                continue
    
    # Create table
    if matched_intervals:
        # Sort by deviation (lowest first) and take top N (or all if max_intervals is None)
        sorted_by_deviation = sorted(matched_intervals, key=lambda x: x[2])  # x[2] is deviation
        if max_intervals is not None:
            top_intervals = sorted_by_deviation[:max_intervals]
        else:
            top_intervals = sorted_by_deviation
        
        # Title
        total_matched = len(matched_intervals)
        shown_count = len(top_intervals)
        title = f'Known Intervals (Top {shown_count} of {total_matched} matched)' if max_intervals is not None and total_matched > max_intervals else f'Known Intervals ({total_matched} matched)'
        ax.text(0.50, 0.95, title, 
                transform=ax.transAxes,
                fontsize=18, fontweight='bold',
                ha='center',
                color=BIOTUNER_COLORS['dark'])
        
        # Prepare table data - sorted by ratio for display
        sorted_intervals = sorted(top_intervals, key=lambda x: x[0])
        
        table_data = []
        
        for ratio, name, deviation in sorted_intervals:
            ratio_cents = 1200 * np.log2(ratio)
            # Convert ratio to fraction
            frac = Fraction(ratio).limit_denominator(max_denom)
            ratio_str = f'{frac.numerator}/{frac.denominator}' if frac.denominator != 1 else str(frac.numerator)
            # Don't truncate interval names
            display_name = name
            table_data.append([
                ratio_str,
                f'{ratio_cents:.1f}¢',
                display_name,
                f'±{deviation:.1f}¢'
            ])
        
        # Create table - FULL WIDTH with adaptive sizing
        if table_data:
            col_labels = ['Ratio', 'Cents', 'Interval Name', 'Dev']
            
            # Adaptive row height based on number of rows
            num_rows = len(table_data)
            if num_rows <= 10:
                row_scale = 1.8
                fontsize = 12
            elif num_rows <= 20:
                row_scale = 1.6
                fontsize = 11
            else:
                row_scale = 1.4
                fontsize = 10
            
            # Use most of panel for table
            table_height = 0.84
            table_width = 0.95
            table_left = 0.025
            table_bottom = 0.08
            
            table = ax.table(
                cellText=table_data,
                colLabels=col_labels,
                cellLoc='left',
                loc='center',
                bbox=[table_left, table_bottom, table_width, table_height]
            )
            
            table.auto_set_font_size(False)
            table.set_fontsize(fontsize)
            table.scale(1, row_scale)
            
            # Set column widths for full width layout
            for i in range(len(table_data) + 1):  # +1 for header
                table[(i, 0)].set_width(0.12)  # Ratio
                table[(i, 1)].set_width(0.11)  # Cents
                table[(i, 2)].set_width(0.67)  # Interval Name (use most space)
                table[(i, 3)].set_width(0.10)  # Dev
            
            # Style header
            for i in range(len(col_labels)):
                cell = table[(0, i)]
                cell.set_facecolor(BIOTUNER_COLORS['primary'])
                cell.set_text_props(weight='bold', color='white', fontsize=fontsize+1)
                cell.set_edgecolor(BIOTUNER_COLORS['dark'])
                cell.set_linewidth(2.0)
            
            # Style data rows
            for i in range(1, len(table_data) + 1):
                for j in range(len(col_labels)):
                    cell = table[(i, j)]
                    # Alternate row colors
                    if i % 2 == 0:
                        cell.set_facecolor('#f8f9fa')
                    else:
                        cell.set_facecolor('white')
                    cell.set_edgecolor('#dee2e6')
                    cell.set_linewidth(1.0)
                    
                    # Color code deviation column
                    if j == 3:  # Deviation column
                        dev_val = sorted_intervals[i-1][2]
                        if dev_val < 2:
                            cell.set_text_props(color='#28a745', weight='bold', fontsize=fontsize)
                        elif dev_val < 4:
                            cell.set_text_props(color='#ffc107', weight='bold', fontsize=fontsize)
                        else:
                            cell.set_text_props(fontsize=fontsize)
                    else:
                        cell.set_text_props(fontsize=fontsize)
    else:
        # No matches
        ax.text(0.50, 0.55, 'No matching known intervals\nfound within ±5¢ tolerance', 
                transform=ax.transAxes,
                fontsize=12,
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=1', 
                         facecolor='#f8d7da',
                         edgecolor='#f5c6cb',
                         linewidth=1.5,
                         alpha=0.8))
    
    plt.tight_layout()
    return fig, ax


def plot_tuning_summary(
    tuning: Optional[List[float]] = None,
    peaks: Optional[np.ndarray] = None,
    ratios: Optional[List[float]] = None,
    metric: str = 'harmsim',
    ratio_type: str = 'all',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    panels: int = 4,
    extra_panels: Optional[List[str]] = None,
    show_summary: bool = True,
    show_source_curve: bool = True,
    max_denom: int = 100,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot comprehensive tuning analysis summary with 2, 4, or 5 panels.
    
    For individual plots, use plot_tuning_scale(), plot_tuning_matrix(), plot_tuning_dissonance(), 
    plot_tuning_entropy(), or plot_tuning_harmonic().
    
    Parameters
    ----------
    tuning : list of float, optional
        Tuning scale (e.g., diss_scale, HE_scale, harmonic_tuning).
        If provided, will use this directly for all panels.
    peaks : np.ndarray, optional
        Spectral peaks (for computing ratios if tuning not provided)
    ratios : list of float, optional
        Pre-computed peak ratios (fallback if neither tuning nor peaks provided)
    metric : str, default='harmsim'
        Metric for consonance matrix:
        - 'harmsim': Harmonic similarity (Gill & Purves) - higher = more consonant
        - 'cons': Consonance (a+b)/(a*b) - higher = more consonant
        - 'tenney': Tenney height - lower = more consonant (inverted for display)
        - 'denom': Denominator complexity - lower = more consonant (inverted for display)
        - 'subharm_tension': Subharmonic tension - lower = more consonant (inverted)
    ratio_type : str, default='all'
        Type of ratios to compute in matrix:
        - 'pos_harm': Only positive harmonics (a/b when a>b)
        - 'sub_harm': Only subharmonics (a/b when a<b)
        - 'all': Both positive and subharmonics
    vmin : float, optional
        Minimum value for color scale (auto-adjusted if None)
    vmax : float, optional
        Maximum value for color scale (auto-adjusted if None)
    panels : int, default=4
        Number of panels (2, 4, or 5 with summary):
        - 2: Tuning scale + Consonance matrix
        - 4: Tuning scale + Consonance matrix + Step sizes + Consonance profile
        - 5: All above + Summary panel (if show_summary=True)
    extra_panels : list of str, optional
        Custom extra panels for 4-panel mode:
        - 'step_sizes': Melodic interval sizes between adjacent notes
        - 'consonance_profile': Average consonance of each scale degree
        - 'interval_distribution': Histogram of all interval sizes in tuning
        - 'harmonic_deviation': Deviation from ideal harmonic ratios
        Default: ['step_sizes', 'consonance_profile']
    show_summary : bool, default=True
        If True, adds a 5th panel with overall harmonicity and interval matches
    show_source_curve : bool, default=True
        If True and tuning is 'diss_curve' or 'HE', shows the source curve
        (dissonance or entropy) as a top full-width panel. Requires biotuner
        object passed via bt_object kwarg.
    max_denom : int, default=100
        Maximum denominator for fraction simplification when displaying ratios.
        Converts float ratios to fractions (e.g., 5/4 instead of 1.25).
        Lower values produce simpler fractions. Default 100 works well for most scales.
    figsize : tuple, optional
        Figure size (auto-adjusted based on panels if None)
    
    Returns
    -------
    fig : matplotlib figure
    """
    from biotuner.metrics import tuning_cons_matrix, dyad_similarity, compute_consonance
    from biotuner.biotuner_utils import compute_peak_ratios
    from fractions import Fraction
    
    set_biotuner_style()
    
    # Set default extra panels
    if extra_panels is None:
        extra_panels = ['step_sizes', 'consonance_profile']
    
    # Check if we need to show source curve (for diss_curve or HE tunings)
    bt_object = kwargs.get('bt_object', None)
    tuning_name = kwargs.get('tuning_name', None)  # Name of the tuning being plotted
    needs_source_curve = False
    source_curve_type = None
    
    if show_source_curve and tuning_name in ['diss_curve', 'HE']:
        if bt_object is not None:
            if tuning_name == 'diss_curve' and hasattr(bt_object, 'ratio_diss') and hasattr(bt_object, 'diss'):
                needs_source_curve = True
                source_curve_type = 'dissonance'
            elif tuning_name == 'HE' and hasattr(bt_object, 'ratio_HE') and hasattr(bt_object, 'HE'):
                needs_source_curve = True
                source_curve_type = 'entropy'
    
    # Auto-adjust figsize based on panels and source curve
    if figsize is None:
        if needs_source_curve:
            # Add extra height for source curve at top - INCREASED for better table spacing
            if panels == 2:
                figsize = (16, 14)  # 2 panels + source curve
            elif show_summary and panels >= 4:
                figsize = (16, 30)  # 5 panels + source curve - INCREASED height for more table rows
            else:  # 4 panels
                figsize = (16, 19)  # 4 panels + source curve - more height
        else:
            if panels == 2:
                figsize = (16, 6)
            elif show_summary and panels >= 4:
                figsize = (16, 16)  # Extra height for summary panel
            else:  # 4 panels
                figsize = (16, 13)
    
    # Create figure with appropriate layout
    if needs_source_curve:
        # Layout with source curve at top - INCREASED SPACING to prevent title overlap
        fig = plt.figure(figsize=figsize)
        
        if panels == 2:
            # Top: source curve (full width), Bottom: 1x2 grid
            gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.60, wspace=0.30)
            ax_curve = fig.add_subplot(gs[0, :])  # Top full width
            ax1 = fig.add_subplot(gs[1, 0])
            ax2 = fig.add_subplot(gs[1, 1])
            axes = [ax1, ax2]
        elif show_summary and panels >= 4:
            # Top: source curve, Middle: 2x2 grid, Bottom: summary
            gs = fig.add_gridspec(4, 2, height_ratios=[0.9, 1, 1, 1.3], hspace=0.60, wspace=0.30)
            ax_curve = fig.add_subplot(gs[0, :])  # Top full width
            ax1 = fig.add_subplot(gs[1, 0])
            ax2 = fig.add_subplot(gs[1, 1])
            ax3 = fig.add_subplot(gs[2, 0])
            ax4 = fig.add_subplot(gs[2, 1])
            ax5 = fig.add_subplot(gs[3, :])  # Bottom full width
            axes = [ax1, ax2, ax3, ax4, ax5]
        else:  # 4 panels
            # Top: source curve, Bottom: 2x2 grid
            gs = fig.add_gridspec(3, 2, height_ratios=[0.9, 1, 1], hspace=0.60, wspace=0.30)
            ax_curve = fig.add_subplot(gs[0, :])  # Top full width
            ax1 = fig.add_subplot(gs[1, 0])
            ax2 = fig.add_subplot(gs[1, 1])
            ax3 = fig.add_subplot(gs[2, 0])
            ax4 = fig.add_subplot(gs[2, 1])
            axes = [ax1, ax2, ax3, ax4]
    else:
        # Original layout without source curve
        if panels == 2:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            axes = [ax1, ax2]
        elif show_summary and panels >= 4:
            # 5 panels: 2x2 grid + wide bottom panel
            fig = plt.figure(figsize=figsize)
            gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.7], hspace=0.45, wspace=0.30)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 0])
            ax4 = fig.add_subplot(gs[1, 1])
            ax5 = fig.add_subplot(gs[2, :])  # Wide bottom panel
            axes = [ax1, ax2, ax3, ax4, ax5]
        else:  # 4 panels
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
            axes = [ax1, ax2, ax3, ax4]
            # Add more vertical spacing between rows to prevent overlap
            plt.subplots_adjust(hspace=0.35, wspace=0.25)
    
    # Determine which tuning to use
    if tuning is not None:
        # Use provided tuning scale directly
        scale = tuning
    elif peaks is not None:
        # Compute ratios from peaks
        if ratios is None:
            ratios = compute_peak_ratios(peaks, rebound=True)
        scale = ratios
    elif ratios is not None:
        # Use provided ratios
        scale = ratios
    else:
        raise ValueError("Must provide either 'tuning', 'peaks', or 'ratios'")
    
    # Ensure scale is a list (not a single value)
    if not isinstance(scale, (list, np.ndarray)):
        scale = [scale]
    
    # Convert SymPy symbolic objects to floats (for euler_fokker)
    try:
        import sympy
        scale = [float(x) if hasattr(x, '__float__') else float(sympy.N(x)) if isinstance(x, sympy.Basic) else x 
                 for x in scale]
    except (ImportError, AttributeError):
        # If sympy not available or no symbolic objects, just ensure numeric
        scale = [float(x) for x in scale]
    
    # Reduce title padding when source curve present to prevent overlap
    title_pad = 10 if needs_source_curve else 15
    
    # PANEL 1: Tuning scale - call individual method
    plot_tuning_scale(tuning=scale, max_denom=max_denom, ax=axes[0])
    # Override title padding for consistency
    axes[0].set_title('Tuning Scale', fontsize=20, fontweight='bold', pad=title_pad)
    
    # PANEL 2: Consonance matrix - call individual method
    plot_tuning_matrix(
        tuning=scale, 
        metric=metric, 
        ratio_type=ratio_type,
        vmin=vmin,
        vmax=vmax,
        max_denom=max_denom,
        ax=axes[1]
    )
    # Override title padding for consistency
    title_suffix = f' ({ratio_type})' if ratio_type != 'all' else ''
    metric_labels = {
        'harmsim': 'Harmonic Similarity',
        'cons': 'Consonance',
        'tenney': 'Tenney Height (inv.)',
        'denom': 'Denominator (inv.)',
        'subharm_tension': 'Subharm Tension (inv.)'
    }
    axes[1].set_title(f'Consonance Matrix{title_suffix}\n{metric_labels.get(metric, metric)}', 
                      fontsize=20, fontweight='bold', pad=title_pad)
    
    # Store cons_matrix for summary panel (need to recompute for now)
    # TODO: Refactor to return cons_matrix from plot_tuning_matrix
    from biotuner.metrics import dyad_similarity, compute_consonance
    from fractions import Fraction
    
    metric_functions = {
        'harmsim': lambda r: dyad_similarity(float(Fraction(r).limit_denominator(max_denom))),
        'cons': lambda r: compute_consonance(float(Fraction(r).limit_denominator(max_denom))),
        'tenney': lambda r: 1 / (1 + np.log2(Fraction(r).limit_denominator(max_denom).numerator * 
                                              Fraction(r).limit_denominator(max_denom).denominator)),
        'denom': lambda r: 1 / (1 + Fraction(r).limit_denominator(max_denom).denominator),
    }
    
    # Quick recompute for summary panel
    n_values = len(scale)
    cons_matrix = np.zeros((n_values, n_values))
    
    if metric in metric_functions:
        metric_fn = metric_functions[metric]
        
        # Compute matrix with directional ratios
        # matrix[i,j] represents consonance from scale[i] to scale[j]
        for i in range(n_values):
            for j in range(n_values):
                if i == j:
                    cons_matrix[i, j] = 0  # Diagonal = 0
                else:
                    # Compute directional ratio: from i to j means scale[j]/scale[i]
                    ratio = scale[j] / scale[i]
                    
                    # Filter based on ratio_type
                    if ratio_type == 'all':
                        # All intervals: compute consonance of ratio (or its inverse if <1)
                        # Most metrics expect ratio > 1, so invert if needed
                        actual_ratio = ratio if ratio >= 1 else 1/ratio
                        cons_matrix[i, j] = metric_fn(actual_ratio)
                    elif ratio_type == 'pos_harm':
                        # Only ascending intervals (j > i means ratio > 1)
                        if ratio > 1:
                            cons_matrix[i, j] = metric_fn(ratio)
                    elif ratio_type == 'sub_harm':
                        # Only descending intervals (j < i means ratio < 1)
                        if ratio < 1:
                            cons_matrix[i, j] = metric_fn(1/ratio)  # Invert for metric
        
        # Scale metrics to 0-100 range for consistency
        if metric == 'cons':
            cons_matrix = cons_matrix * 100
        elif metric in ['tenney', 'denom']:
            cons_matrix = cons_matrix * 100
            
    else:
        # Special handling for subharm_tension
        n_values = len(scale)
        cons_matrix = np.zeros((n_values, n_values))
        
        from biotuner.metrics import compute_subharmonic_tension
        
        def _compute_subharm_metric(ratio):
            """Helper to compute inverted subharmonic tension for a single ratio."""
            try:
                fundamental = 100.0
                frac = Fraction(ratio).limit_denominator(max_denom)
                freq1 = fundamental
                freq2 = fundamental * (frac.numerator / frac.denominator)
                chord = np.array([freq1, freq2])
                _, _, tension, _ = compute_subharmonic_tension(
                    chord, n_harmonics=10, delta_lim=5.0, min_notes=2
                )
                if isinstance(tension, str) or len(tension) == 0 or tension[0] == 0:
                    return compute_consonance(ratio) * 100
                tension_val = tension[0]
                inverted = 100 / (1 + np.log1p(tension_val * 10))
                return max(0, min(100, inverted))
            except (ValueError, ZeroDivisionError, Exception):
                return compute_consonance(ratio) * 100
        
        # Compute matrix manually for subharm_tension
        for i in range(n_values):
            for j in range(n_values):
                if i == j:
                    cons_matrix[i, j] = 0
                else:
                    if ratio_type == "pos_harm":
                        if scale[i] > scale[j]:
                            ratio = scale[i] / scale[j]
                            cons_matrix[i, j] = _compute_subharm_metric(ratio)
                    elif ratio_type == "sub_harm":
                        if scale[i] < scale[j]:
                            ratio = scale[i] / scale[j]
                            cons_matrix[i, j] = _compute_subharm_metric(ratio)
                    elif ratio_type == "all":
                        ratio = scale[i] / scale[j]
                        cons_matrix[i, j] = _compute_subharm_metric(ratio)
    
    # Auto-adjust vmin/vmax based on non-zero values
    if vmin is None or vmax is None:
        non_zero = cons_matrix[cons_matrix != 0]
        if len(non_zero) > 0:
            if vmin is None:
                if metric == 'subharm_tension':
                    vmin = np.min(non_zero)
                else:
                    vmin = np.percentile(non_zero, 5)
            if vmax is None:
                if metric == 'subharm_tension':
                    vmax = np.max(non_zero)
                else:
                    vmax = np.percentile(non_zero, 95)
        else:
            vmin = 0
            vmax = 100
    
    # Create custom colormap: dark burnt orange to turquoise (matching plot_peaks)
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['#8B4513', '#CD5C5C', '#F4A460', '#FFD700', '#90EE90', '#48D1CC', '#40E0D0']
    custom_cmap = LinearSegmentedColormap.from_list('orange_to_turquoise', colors_list)
    
    # Determine colormap based on metric (matching plot_peaks logic)
    if metric in ['harmsim', 'cons']:
        # Higher is better (more harmonic/consonant) - custom burnt orange to turquoise
        cmap = custom_cmap
    else:
        # Higher is worse (more dissonant/complex) - reversed
        cmap = custom_cmap.reversed()
    
    # Note: plot_tuning_matrix() already added the colorbar, no need to add another
    # The colorbar is created by the individual method call above
    
    # Add labels with fractions
    n_values = len(scale)
    axes[1].set_xticks(range(n_values))
    axes[1].set_yticks(range(n_values))
    
    # Convert scale values to fractions for labels
    fraction_labels = []
    for v in scale:
        frac = Fraction(v).limit_denominator(max_denom)
        if frac.denominator == 1:
            fraction_labels.append(str(frac.numerator))
        else:
            fraction_labels.append(f'{frac.numerator}/{frac.denominator}')
    
    axes[1].set_xticklabels(fraction_labels, fontsize=11, rotation=45)
    axes[1].set_yticklabels(fraction_labels, fontsize=11)
    
    # Styling
    axes[1].set_xlabel('Tuning Ratio', fontsize=16, fontweight='normal')
    axes[1].set_ylabel('Tuning Ratio', fontsize=16, fontweight='normal')
    title_suffix = f' ({ratio_type})' if ratio_type != 'all' else ''
    axes[1].set_title(f'Consonance Matrix{title_suffix}\n{metric_labels.get(metric, metric)}', 
                 fontsize=20, fontweight='bold', pad=title_pad)
    
    # ===== EXTRA PANELS (if panels >= 4) =====
    if panels >= 4 and len(axes) >= 4:
        # Panel 3: Call individual method for step sizes or interval distribution
        if 'step_sizes' in extra_panels:
            plot_tuning_intervals(tuning=scale, max_denom=max_denom, ax=axes[2])
            axes[2].set_title('Melodic Intervals (Step Sizes)', fontsize=20, fontweight='bold', pad=title_pad)
            
        elif 'interval_distribution' in extra_panels:
            # Compute all intervals in the tuning
            all_intervals_cents = []
            for i in range(len(scale)):
                for j in range(i + 1, len(scale)):
                    interval_ratio = scale[j] / scale[i]
                    all_intervals_cents.append(1200 * np.log2(interval_ratio))
            
            # Plot histogram
            axes[2].hist(all_intervals_cents, bins=20, color=BIOTUNER_COLORS['primary'],
                        edgecolor=BIOTUNER_COLORS['dark'], linewidth=1.5, alpha=0.7)
            
            axes[2].set_xlabel('Interval Size (cents)', fontsize=16, fontweight='normal')
            axes[2].set_ylabel('Frequency', fontsize=16, fontweight='normal')
            axes[2].set_title('Interval Distribution', fontsize=20, fontweight='bold', pad=title_pad)
            axes[2].grid(True, alpha=0.25, axis='y')
        
        # Panel 4: Call individual method for consonance profile or harmonic deviation
        if 'consonance_profile' in extra_panels:
            plot_tuning_consonance_profile(
                tuning=scale, 
                metric=metric,
                ratio_type=ratio_type,
                max_denom=max_denom,
                ax=axes[3]
            )
            axes[3].set_title('Consonance Profile (Violin Plot)', fontsize=20, fontweight='bold', pad=title_pad)
            
        elif 'harmonic_deviation' in extra_panels:
            # Compute deviation from ideal harmonic ratios (1, 2, 3, 4, 5, 6...)
            ideal_harmonics = [1.0 * (i + 1) for i in range(len(scale))]
            deviations_cents = []
            for i, ratio in enumerate(scale):
                # Find closest harmonic
                closest_harm = min(ideal_harmonics, key=lambda h: abs(h - ratio))
                # Compute deviation in cents
                if closest_harm > 0 and ratio > 0:
                    deviation = 1200 * np.log2(ratio / closest_harm)
                    deviations_cents.append(deviation)
                else:
                    deviations_cents.append(0)
            
            # Plot deviations
            dev_colors = [BIOTUNER_COLORS['success'] if abs(d) < 10 else 
                         BIOTUNER_COLORS['warning'] if abs(d) < 30 else 
                         BIOTUNER_COLORS['danger'] for d in deviations_cents]
            
            bars = axes[3].bar(range(len(deviations_cents)), deviations_cents, color=dev_colors,
                              edgecolor=BIOTUNER_COLORS['dark'], linewidth=1.5, alpha=0.8)
            
            # Add zero reference line
            axes[3].axhline(0, color=BIOTUNER_COLORS['dark'], linestyle='-', 
                          linewidth=2, alpha=0.7, label='Perfect harmonic')
            
            # Add values on bars
            for i, dev in enumerate(deviations_cents):
                y_pos = dev + (5 if dev > 0 else -5)
                va = 'bottom' if dev > 0 else 'top'
                axes[3].text(i, y_pos, f'{dev:+.0f}¢', 
                           ha='center', va=va, fontsize=10, fontweight='semibold')
            
            axes[3].set_xlabel('Scale Degree', fontsize=16, fontweight='normal')
            axes[3].set_ylabel('Deviation from Ideal (cents)', fontsize=16, fontweight='normal')
            axes[3].set_title('Harmonic Deviation', fontsize=20, fontweight='bold', pad=title_pad)
            axes[3].set_xticks(range(len(deviations_cents)))
            axes[3].set_xticklabels([f'{i+1}' for i in range(len(deviations_cents))], fontsize=13)
            axes[3].legend(fontsize=12, framealpha=0.95)
            axes[3].grid(True, alpha=0.25, axis='y')
    
    # ===== SUMMARY PANEL (if show_summary and 5 panels) =====
    if show_summary and len(axes) == 5:
        from biotuner.dictionaries import interval_catalog
        from fractions import Fraction
        from matplotlib.patches import Arc, Wedge, Circle
        from matplotlib.table import Table
        
        # Clear the summary axis
        axes[4].axis('off')
        axes[4].set_xlim(0, 1)
        axes[4].set_ylim(0, 1)
        
        # Compute overall harmonicity (average consonance across all pairs)
        non_zero = cons_matrix[cons_matrix != 0]
        if len(non_zero) > 0:
            overall_harmonicity = np.mean(non_zero)
        else:
            overall_harmonicity = 0
        
        # Find matching known intervals (within 5 cents tolerance)
        tolerance_cents = 5
        matched_intervals = []
        
        for ratio in scale:
            ratio_cents = 1200 * np.log2(ratio) if ratio > 0 else 0
            
            # Check against interval catalog
            for interval_name, interval_ratio in interval_catalog:
                try:
                    # Convert sympy ratio to float
                    catalog_ratio = float(interval_ratio)
                    catalog_cents = 1200 * np.log2(catalog_ratio) if catalog_ratio > 0 else 0
                    
                    # Check if within tolerance
                    if abs(ratio_cents - catalog_cents) < tolerance_cents:
                        matched_intervals.append((ratio, interval_name, abs(ratio_cents - catalog_cents)))
                        break  # Only match first interval found
                except:
                    continue
        
        # === CENTERED TABLE ===
        if matched_intervals:
            # Sort by deviation (lowest first) and take top 10
            sorted_by_deviation = sorted(matched_intervals, key=lambda x: x[2])  # x[2] is deviation
            top_intervals = sorted_by_deviation[:10]  # Top 10 closest matches
            
            # Title
            total_matched = len(matched_intervals)
            shown_count = len(top_intervals)
            title = f'Known Intervals (Top {shown_count} of {total_matched} matched)' if total_matched > 10 else f'Known Intervals ({total_matched} matched)'
            axes[4].text(0.50, 0.95, title, 
                        transform=axes[4].transAxes,
                        fontsize=18, fontweight='bold',
                        ha='center',
                        color=BIOTUNER_COLORS['dark'])
            
            # Prepare table data - sorted by ratio for display
            sorted_intervals = sorted(top_intervals, key=lambda x: x[0])
            
            table_data = []
            
            for ratio, name, deviation in sorted_intervals:
                ratio_cents = 1200 * np.log2(ratio)
                # Convert ratio to fraction
                frac = Fraction(ratio).limit_denominator(max_denom)
                ratio_str = f'{frac.numerator}/{frac.denominator}' if frac.denominator != 1 else str(frac.numerator)
                # Don't truncate interval names
                display_name = name
                table_data.append([
                    ratio_str,
                    f'{ratio_cents:.1f}¢',
                    display_name,
                    f'±{deviation:.1f}¢'
                ])
            
            # Create table - FULL WIDTH with adaptive sizing
            if table_data:
                col_labels = ['Ratio', 'Cents', 'Interval Name', 'Dev']
                
                # Adaptive row height based on number of rows
                # Extra spacing when source curve is present (more vertical space available)
                num_rows = len(table_data)
                if num_rows <= 10:
                    row_scale = 4.0 if needs_source_curve else 2.8
                    fontsize = 12
                else:
                    # Maintain good spacing even with many rows
                    row_scale = 4.5 if needs_source_curve else 2.7
                    fontsize = 11
                
                # Use most of panel for table
                table_height = 0.84
                table_width = 0.95
                table_left = 0.025
                table_bottom = 0.08
                
                table = axes[4].table(
                    cellText=table_data,
                    colLabels=col_labels,
                    cellLoc='left',
                    loc='center',
                    bbox=[table_left, table_bottom, table_width, table_height]
                )
                
                table.auto_set_font_size(False)
                table.set_fontsize(fontsize)
                table.scale(1, row_scale)
                
                # Set column widths for full width layout
                for i in range(len(table_data) + 1):  # +1 for header
                    table[(i, 0)].set_width(0.12)  # Ratio
                    table[(i, 1)].set_width(0.11)  # Cents
                    table[(i, 2)].set_width(0.67)  # Interval Name (use most space)
                    table[(i, 3)].set_width(0.10)  # Dev
                
                # Style header
                for i in range(len(col_labels)):
                    cell = table[(0, i)]
                    cell.set_facecolor(BIOTUNER_COLORS['primary'])
                    cell.set_text_props(weight='bold', color='white', fontsize=fontsize+1)
                    cell.set_edgecolor(BIOTUNER_COLORS['dark'])
                    cell.set_linewidth(2.0)
                
                # Style data rows
                for i in range(1, len(table_data) + 1):
                    for j in range(len(col_labels)):
                        cell = table[(i, j)]
                        # Alternate row colors
                        if i % 2 == 0:
                            cell.set_facecolor('#f8f9fa')
                        else:
                            cell.set_facecolor('white')
                        cell.set_edgecolor('#dee2e6')
                        cell.set_linewidth(1.0)
                        
                        # Color code deviation column
                        if j == 3:  # Deviation column
                            dev_val = sorted_intervals[i-1][2]  # Use sorted_intervals, not matched_intervals
                            if dev_val < 2:
                                cell.set_text_props(color='#28a745', weight='bold', fontsize=fontsize)
                            elif dev_val < 4:
                                cell.set_text_props(color='#ffc107', weight='bold', fontsize=fontsize)
                            else:
                                cell.set_text_props(fontsize=fontsize)
                        else:
                            cell.set_text_props(fontsize=fontsize)
        else:
            # No matches
            axes[4].text(0.50, 0.55, 'No matching known intervals\nfound within ±5¢ tolerance', 
                        transform=axes[4].transAxes,
                        fontsize=12,
                        ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=1', 
                                 facecolor='#f8d7da',
                                 edgecolor='#f5c6cb',
                                 linewidth=1.5,
                                 alpha=0.8))
    
    # Plot source curve if needed (before returning)
    if needs_source_curve and source_curve_type is not None:
        if source_curve_type == 'dissonance':
            # Plot dissonance curve
            ax_curve.plot(bt_object.ratio_diss, bt_object.diss, 
                         color=BIOTUNER_COLORS['danger'], linewidth=2.5, label='Dissonance')
            
            # Mark minima from scale
            if hasattr(bt_object, 'diss_scale'):
                for ratio in bt_object.diss_scale:
                    if 1.0 <= ratio <= 2.0:
                        idx = np.argmin(np.abs(bt_object.ratio_diss - ratio))
                        ax_curve.plot(ratio, bt_object.diss[idx], 'o',
                                    color=BIOTUNER_COLORS['accent'], markersize=8,
                                    markeredgecolor=BIOTUNER_COLORS['dark'], markeredgewidth=1.5)
                        ax_curve.axvline(ratio, color=BIOTUNER_COLORS['dark'],
                                       linestyle='--', linewidth=1.5, alpha=0.4)
            
            ax_curve.set_xlim([1.0, 2.0])
            ax_curve.set_xlabel('Frequency Ratio', fontsize=16, fontweight='normal')
            ax_curve.set_ylabel('Sensory Dissonance', fontsize=16, fontweight='normal')
            ax_curve.set_title('Dissonance Curve (Source)', fontsize=20, fontweight='bold', pad=15)
            ax_curve.legend(fontsize=13, framealpha=0.95)
            ax_curve.grid(True, alpha=0.25)
            
        elif source_curve_type == 'entropy':
            # Plot harmonic entropy curve
            ax_curve.plot(bt_object.ratio_HE, bt_object.HE,
                         color=BIOTUNER_COLORS['primary'], linewidth=2.5, label='Harmonic Entropy')
            
            # Mark minima from scale
            if hasattr(bt_object, 'HE_scale'):
                for ratio in bt_object.HE_scale:
                    if 1.0 <= ratio <= 2.0:
                        idx = np.argmin(np.abs(bt_object.ratio_HE - ratio))
                        ax_curve.plot(ratio, bt_object.HE[idx], 'o',
                                    color=BIOTUNER_COLORS['accent'], markersize=8,
                                    markeredgecolor=BIOTUNER_COLORS['dark'], markeredgewidth=1.5)
                        ax_curve.axvline(ratio, color=BIOTUNER_COLORS['dark'],
                                       linestyle='--', linewidth=1.5, alpha=0.4)
            
            ax_curve.set_xlim([1.0, 2.0])
            ax_curve.set_xlabel('Frequency Ratio', fontsize=16, fontweight='normal')
            ax_curve.set_ylabel('Harmonic Entropy', fontsize=16, fontweight='normal')
            ax_curve.set_title('Harmonic Entropy Curve (Source)', fontsize=20, fontweight='bold', pad=15)
            ax_curve.legend(fontsize=13, framealpha=0.95)
            ax_curve.grid(True, alpha=0.25)
    
    # Use tight_layout only for 2-panel mode without source curve; custom spacing already set for others
    if panels == 2 and not needs_source_curve:
        plt.tight_layout()
    
    return fig


# Backward compatibility alias
plot_tuning = plot_tuning_summary


# ============================================================================
# Harmonic Fit Visualization
# ============================================================================

def plot_harmonic_fit_network(
    peaks: np.ndarray,
    amps: np.ndarray,
    multi_harmonics: np.ndarray,
    n_harm: int = 10,
    harm_bounds: float = 0.5,
    function: str = 'mult',
    figsize: tuple = (10, 10),
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot harmonic network showing relationships between peaks.
    
    Creates a circular network where nodes are peaks and edges show
    the number of shared harmonics between peak pairs.
    
    Parameters
    ----------
    peaks : np.ndarray
        Peak frequencies in Hz
    amps : np.ndarray
        Peak amplitudes
    multi_harmonics : np.ndarray
        Pre-computed harmonic series for each peak (n_peaks x n_harm)
    n_harm : int, default=10
        Number of harmonics per peak
    harm_bounds : float, default=0.5
        Frequency threshold (Hz) for considering harmonics as matching
    function : str, default='mult'
        Harmonic function used ('mult' or 'div')
    figsize : tuple, default=(10, 10)
        Figure size
    ax : plt.Axes, optional
        Existing axes
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    """
    set_biotuner_style()
    
    # Determine if standalone or embedded
    is_standalone = ax is None
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Scale font sizes based on usage
    title_size = 28 if is_standalone else 18
    label_size = 16 if is_standalone else 11
    edge_label_size = 14 if is_standalone else 9
    legend_size = 14 if is_standalone else 9
    
    n_peaks = len(peaks)
    
    # Create network layout - circular arrangement
    theta = np.linspace(0, 2*np.pi, n_peaks, endpoint=False)
    node_x = np.cos(theta)
    node_y = np.sin(theta)
    
    # Normalize amplitudes for node sizing
    norm_amps = (amps - amps.min()) / (amps.max() - amps.min() + 1e-10)
    node_sizes = 300 + norm_amps * 1200
    
    # Color palette
    colors = sns.color_palette('husl', n_peaks)
    
    # Draw edges for harmonic relationships
    max_shared = 0
    edge_data = []
    
    # First pass: collect edge data and find max
    for i in range(n_peaks):
        for j in range(i+1, n_peaks):
            harms_i = multi_harmonics[i]
            harms_j = multi_harmonics[j]
            shared = sum(1 for hi in harms_i for hj in harms_j if abs(hi - hj) < harm_bounds)
            
            if shared > 0:
                edge_data.append((i, j, shared))
                max_shared = max(max_shared, shared)
    
    # Second pass: draw edges with variable styling
    for i, j, shared in edge_data:
        strength_ratio = shared / max(max_shared, 1)
        
        if strength_ratio < 0.3:
            linestyle = ':'
            alpha = 0.3
            linewidth = 0.5
            color = '#CCCCCC'
        elif strength_ratio < 0.6:
            linestyle = '--'
            alpha = 0.5
            linewidth = 1.5
            color = '#999999'
        else:
            linestyle = '-'
            alpha = 0.8
            linewidth = 2.5 + strength_ratio * 2
            color = '#444444'
        
        ax.plot([node_x[i], node_x[j]], [node_y[i], node_y[j]], 
                color=color, alpha=alpha, linewidth=linewidth, 
                linestyle=linestyle, zorder=1)
        
        # Add shared count label on edge (only if 2 or more)
        if shared >= 2:
            mid_x = (node_x[i] + node_x[j]) / 2
            mid_y = (node_y[i] + node_y[j]) / 2
            ax.text(mid_x, mid_y, str(shared), 
                    fontsize=edge_label_size, ha='center', va='center', fontweight='bold',
                    bbox=dict(boxstyle='circle,pad=0.2', facecolor='white', 
                             edgecolor=color, alpha=0.95, linewidth=1.5),
                    color=color, zorder=2)
    
    # Draw nodes
    for i in range(n_peaks):
        ax.scatter(node_x[i], node_y[i], s=node_sizes[i], c=[colors[i]], 
                   edgecolors=BIOTUNER_COLORS['dark'], linewidths=2.5, 
                   zorder=3, alpha=0.9)
        # Add frequency labels
        label_offset = 1.25
        ax.text(node_x[i]*label_offset, node_y[i]*label_offset, 
                f'{peaks[i]:.1f} Hz', fontsize=label_size, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                         edgecolor=colors[i], alpha=0.9, linewidth=2))
    
    ax.set_xlim([-1.65, 1.65])
    ax.set_ylim([-1.65, 1.65])
    ax.set_aspect('equal')
    ax.axis('off')
    
    ax.set_title(f'Harmonic Network ({function.upper()}, n={n_harm})', 
                  fontsize=title_size, fontweight='bold', pad=15)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=10, label='Node size = amplitude'),
        Line2D([0], [0], color='gray', linewidth=2, label='Edge # = shared harmonics')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=legend_size, 
              framealpha=0.9, edgecolor='lightgray', fancybox=True)
    
    plt.tight_layout()
    return fig, ax


def plot_harmonic_fit_matrix(
    peaks: np.ndarray,
    multi_harmonics: np.ndarray,
    n_harm: int = 10,
    harm_bounds: float = 0.5,
    figsize: tuple = (8, 7),
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot harmonic connectivity matrix showing shared harmonics between peak pairs.
    
    Parameters
    ----------
    peaks : np.ndarray
        Peak frequencies in Hz
    multi_harmonics : np.ndarray
        Pre-computed harmonic series for each peak (n_peaks x n_harm)
    n_harm : int, default=10
        Number of harmonics per peak
    harm_bounds : float, default=0.5
        Frequency threshold (Hz) for considering harmonics as matching
    figsize : tuple, default=(8, 7)
        Figure size
    ax : plt.Axes, optional
        Existing axes
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    """
    set_biotuner_style()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    n_peaks = len(peaks)
    connectivity_matrix = np.zeros((n_peaks, n_peaks))
    
    # Count shared harmonics between peak pairs
    for i in range(n_peaks):
        for j in range(i, n_peaks):
            if i == j:
                connectivity_matrix[i, j] = np.nan  # Diagonal = NaN (will be black)
            else:
                harms_i = multi_harmonics[i]
                harms_j = multi_harmonics[j]
                shared = sum(1 for hi in harms_i for hj in harms_j if abs(hi - hj) < harm_bounds)
                connectivity_matrix[i, j] = shared
                connectivity_matrix[j, i] = shared
    
    # Colormap (same as tuning/peak matrices)
    from matplotlib.colors import LinearSegmentedColormap
    colors_list = ['#8B4513', '#CD5C5C', '#F4A460', '#FFD700', '#90EE90', '#48D1CC', '#40E0D0']
    custom_cmap = LinearSegmentedColormap.from_list('orange_to_turquoise', colors_list)
    custom_cmap.set_bad(color='black')  # NaN values will be black
    
    # Calculate vmin/vmax excluding diagonal
    off_diagonal = connectivity_matrix[~np.isnan(connectivity_matrix)]
    vmin = 0
    vmax = np.max(off_diagonal) if len(off_diagonal) > 0 else 1
    
    # Plot heatmap
    im = ax.imshow(connectivity_matrix, cmap=custom_cmap, aspect='auto',
                    interpolation='nearest', vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Shared Harmonics', fontsize=14, rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=12)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n_peaks))
    ax.set_yticks(np.arange(n_peaks))
    ax.set_xticklabels([f'{p:.1f}' for p in peaks], fontsize=11)
    ax.set_yticklabels([f'{p:.1f}' for p in peaks], fontsize=11)
    
    # Add grid
    ax.set_xticks(np.arange(n_peaks) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_peaks) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    
    # Annotate cells with values
    for i in range(n_peaks):
        for j in range(n_peaks):
            if i == j:
                # Don't annotate diagonal
                continue
            elif connectivity_matrix[i, j] > 0:
                text_color = 'black' if connectivity_matrix[i, j] > vmax/2 else 'white'
                ax.text(j, i, f'{int(connectivity_matrix[i, j])}',
                        ha='center', va='center', color=text_color, fontsize=10)
    
    ax.set_xlabel('Peak Frequency (Hz)', fontsize=16)
    ax.set_ylabel('Peak Frequency (Hz)', fontsize=16)
    ax.set_title('Harmonic Connectivity Matrix', fontsize=16, fontweight='bold', pad=10)
    
    plt.tight_layout()
    return fig, ax


def plot_harmonic_fit_positions(
    peaks: np.ndarray,
    amps: np.ndarray,
    multi_harmonics: np.ndarray,
    common_harms: list,
    n_harm: int = 10,
    harm_bounds: float = 0.5,
    function: str = 'mult',
    figsize: tuple = (14, 6),
    ax_left: Optional[plt.Axes] = None,
    ax_right: Optional[plt.Axes] = None,
    **kwargs
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot harmonic position analysis in two side-by-side panels.
    
    Left panel: Bipartite network showing which harmonic positions are shared
    Right panel: Histogram of harmonic position distribution
    
    Parameters
    ----------
    peaks : np.ndarray
        Peak frequencies in Hz
    amps : np.ndarray
        Peak amplitudes
    multi_harmonics : np.ndarray
        Pre-computed harmonic series for each peak (n_peaks x n_harm)
    common_harms : list
        List of most commonly used harmonic positions
    n_harm : int, default=10
        Number of harmonics per peak
    harm_bounds : float, default=0.5
        Frequency threshold (Hz) for considering harmonics as matching
    function : str, default='mult'
        Harmonic function used ('mult' or 'div')
    figsize : tuple, default=(14, 6)
        Figure size (only used if axes not provided)
    ax_left : plt.Axes, optional
        Existing axes for left panel
    ax_right : plt.Axes, optional
        Existing axes for right panel
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    axes : np.ndarray
        Array of axes [ax_left, ax_right]
    """
    from collections import Counter
    
    set_biotuner_style()
    
    if ax_left is None or ax_right is None:
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig = ax_left.get_figure()
    n_peaks = len(peaks)
    colors = sns.color_palette('husl', n_peaks)
    
    # ========== LEFT PANEL: Bipartite Network ==========
    
    peak_harmonic_connections = {}
    harmonic_pos_usage = Counter()
    
    for i in range(n_peaks):
        peak_harmonic_connections[i] = Counter()
        for j in range(n_peaks):
            if i != j:
                harms_i = multi_harmonics[i]
                harms_j = multi_harmonics[j]
                for h_i_idx, hi in enumerate(harms_i):
                    for h_j_idx, hj in enumerate(harms_j):
                        if abs(hi - hj) < harm_bounds:
                            h_pos = h_i_idx + 1
                            if h_pos <= 10:
                                peak_harmonic_connections[i][h_pos] += 1
                                harmonic_pos_usage[h_pos] += 1
    
    top_positions = sorted([k for k, v in harmonic_pos_usage.items() if v > 0])[:8]
    
    if top_positions:
        peak_x = 0.15
        harm_x = 0.85
        
        peak_y_positions = np.linspace(0.12, 0.88, n_peaks)
        harm_y_positions = np.linspace(0.12, 0.88, len(top_positions))
        
        harm_pos_to_y = {pos: y for pos, y in zip(top_positions, harm_y_positions)}
        
        # Draw connections
        for peak_idx in range(n_peaks):
            for h_pos in top_positions:
                if h_pos in peak_harmonic_connections[peak_idx]:
                    count = peak_harmonic_connections[peak_idx][h_pos]
                    if count > 0:
                        y1 = peak_y_positions[peak_idx]
                        y2 = harm_pos_to_y[h_pos]
                        
                        alpha = min(0.7, 0.2 + count / 5)
                        linewidth = 0.5 + count / 2
                        
                        ax_left.plot([peak_x + 0.05, harm_x - 0.05], [y1, y2],
                                color=colors[peak_idx], alpha=alpha, 
                                linewidth=linewidth, zorder=1)
        
        # Draw peak nodes
        for i, (peak, y_pos) in enumerate(zip(peaks, peak_y_positions)):
            ax_left.scatter(peak_x, y_pos, s=500, c=[colors[i]], 
                       edgecolors=BIOTUNER_COLORS['dark'], linewidths=2, 
                       zorder=3, alpha=0.9)
            ax_left.text(peak_x - 0.08, y_pos, f'{peak:.1f} Hz', 
                    fontsize=10, ha='right', va='center', fontweight='bold',
                    color=colors[i])
        
        # Draw harmonic position nodes
        for h_pos, y_pos in harm_pos_to_y.items():
            total_connections = harmonic_pos_usage[h_pos]
            node_size = 250 + total_connections * 45
            node_color = BIOTUNER_COLORS['success'] if total_connections >= 3 else BIOTUNER_COLORS['accent']
            
            ax_left.scatter(harm_x, y_pos, s=node_size, c=node_color,
                       edgecolors=BIOTUNER_COLORS['dark'], linewidths=1.8, 
                       marker='s', zorder=3, alpha=0.85)
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(h_pos, 'th')
            ax_left.text(harm_x + 0.08, y_pos, f'{h_pos}{suffix}', 
                    fontsize=11, ha='left', va='center', fontweight='bold',
                    color=BIOTUNER_COLORS['dark'])
            ax_left.text(harm_x, y_pos, str(total_connections), 
                    fontsize=8, ha='center', va='center', 
                    color='white', fontweight='bold', zorder=4)
        
        # Section labels
        ax_left.text(peak_x, 0.97, 'Peaks', fontsize=13, ha='center', va='top',
                fontweight='bold', color=BIOTUNER_COLORS['dark'])
        ax_left.text(harm_x, 0.97, 'Harmonics', fontsize=13, ha='center', va='top',
                fontweight='bold', color=BIOTUNER_COLORS['dark'])
        
        ax_left.set_xlim([0, 1])
        ax_left.set_ylim([0, 1])
        ax_left.axis('off')
        
        # Legend
        legend_text = (
            f"Line thickness ∝ strength\n"
            f"Square size ∝ usage\n"
            f"Green = ≥3 peaks share"
        )
        ax_left.text(0.5, 0.02, legend_text, fontsize=9, ha='center', va='bottom',
                transform=ax_left.transAxes,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', 
                         alpha=0.5, edgecolor=BIOTUNER_COLORS['dark'], linewidth=1.2))
    else:
        ax_left.text(0.5, 0.5, 'No shared harmonic positions',
                ha='center', va='center', fontsize=12, transform=ax_left.transAxes)
        ax_left.axis('off')
    
    ax_left.set_title('Shared Harmonic Positions\n(which harmonics are used)', 
                  fontsize=16, fontweight='bold', pad=10)
    
    # ========== RIGHT PANEL: Position Distribution ==========
    
    # Count all harmonic positions
    all_harm_positions = []
    for i in range(n_peaks):
        for j in range(i+1, n_peaks):
            harms_i = multi_harmonics[i]
            harms_j = multi_harmonics[j]
            for h_i_idx, hi in enumerate(harms_i):
                for h_j_idx, hj in enumerate(harms_j):
                    if abs(hi - hj) < harm_bounds:
                        all_harm_positions.append(h_i_idx + 1)
                        all_harm_positions.append(h_j_idx + 1)
    
    if all_harm_positions:
        position_counts = Counter(all_harm_positions)
        positions = sorted(position_counts.keys())
        counts = [position_counts[p] for p in positions]
        
        positions_arr = np.array(positions)
        counts_arr = np.array(counts)
        norm_counts = counts_arr / counts_arr.max()
        
        bars = ax_right.bar(positions, counts, color=BIOTUNER_COLORS['accent'], 
                      edgecolor=BIOTUNER_COLORS['dark'], linewidth=1.5, alpha=0.8)
        
        for bar, norm_count in zip(bars, norm_counts):
            bar.set_alpha(0.4 + norm_count * 0.6)
        
        # Highlight common harmonics
        if common_harms:
            for ch in common_harms[:5]:
                if ch in positions:
                    idx = positions.index(ch)
                    bars[idx].set_color(BIOTUNER_COLORS['success'])
                    bars[idx].set_edgecolor('gold')
                    bars[idx].set_linewidth(3)
        
        ax_right.set_xlabel('Harmonic Position', fontsize=16, fontweight='normal')
        ax_right.set_ylabel('Match Count', fontsize=16, fontweight='normal')
        ax_right.set_title('Harmonic Position Distribution\n(green = most common)', 
                     fontsize=18, fontweight='bold', pad=10)
        ax_right.grid(True, alpha=0.25, axis='y')
        ax_right.set_xticks(positions)
        
        # Add value labels
        for pos, count in zip(positions, counts):
            ax_right.text(pos, count + max(counts)*0.02, str(count),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Stats box
        stats_text = f"Total matches: {sum(counts)}\n"
        stats_text += f"Unique positions: {len(positions)}\n"
        stats_text += f"Most common: {positions[np.argmax(counts)]} ({max(counts)}×)"
        
        ax_right.text(0.98, 0.97, stats_text, transform=ax_right.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.4, 
                         edgecolor=BIOTUNER_COLORS['dark'], linewidth=1.5))
    else:
        ax_right.text(0.5, 0.5, 'No harmonic matches found', 
                ha='center', va='center', fontsize=12,
                transform=ax_right.transAxes)
        ax_right.set_title('Harmonic Position Distribution', 
                     fontsize=16, fontweight='bold', pad=10)
        ax_right.axis('off')
    
    plt.tight_layout()
    return fig, np.array([ax_left, ax_right])


def plot_harmonic_fit_summary(
    peaks: np.ndarray,
    amps: np.ndarray,
    freqs: np.ndarray,
    psd: np.ndarray,
    harm_fit: list,
    harmonics_pos: list,
    common_harms: list,
    matching_pos: list,
    extended_peaks: Optional[np.ndarray] = None,
    extended_amps: Optional[np.ndarray] = None,
    n_harm: int = 10,
    harm_bounds: float = 0.5,
    function: str = 'mult',
    xmin: float = 1,
    xmax: float = 60,
    show_bands: bool = True,
    figsize: tuple = (16, 12),
    **kwargs
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create comprehensive harmonic fit visualization with 4 panels.
    
    Panels:
    1. Top-left: Harmonic network graph
    2. Top-right: Shared harmonic positions & distribution
    3. Bottom-left: Harmonic connectivity matrix
    4. Bottom-right: Harmonic position distribution
    
    Parameters
    ----------
    peaks : np.ndarray
        Original peak frequencies
    amps : np.ndarray
        Peak amplitudes
    freqs : np.ndarray
        Full frequency array from PSD
    psd : np.ndarray
        Power spectral density values
    harm_fit : list
        Fitted harmonic frequencies
    harmonics_pos : list
        Harmonic positions that matched
    common_harms : list
        Most common harmonic positions
    matching_pos : list
        Detailed matching position information
    extended_peaks : np.ndarray, optional
        Extended peaks from peaks_extension
    extended_amps : np.ndarray, optional
        Extended peak amplitudes
    n_harm : int, default=10
        Number of harmonics computed
    harm_bounds : float, default=0.5
        Harmonic fit boundary threshold
    function : str, default='mult'
        Harmonic function used ('mult', 'div', 'exp')
    xmin, xmax : float
        Frequency range for visualization
    show_bands : bool, default=True
        Whether to show EEG frequency bands
    figsize : tuple, default=(16, 12)
        Figure size
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    axes : np.ndarray
        2x2 array of axes
    """
    from biotuner.peaks_extension import EEG_harmonics_mult, EEG_harmonics_div
    
    set_biotuner_style()
    
    # Create figure with custom grid layout
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1], hspace=0.3, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])  # Top-left (network - taller)
    ax3 = fig.add_subplot(gs[1, 0])  # Bottom-left (matrix)
    
    fig.suptitle(f'Harmonic Fit Analysis ({function.upper()}, n={n_harm})', 
                 fontsize=22, fontweight='bold', y=0.98)
    
    # Generate harmonics for each peak
    if function == 'mult':
        multi_harmonics = EEG_harmonics_mult(peaks, n_harm)
    elif function == 'div':
        multi_harmonics, _ = EEG_harmonics_div(peaks, n_harm, mode='div')
    else:
        multi_harmonics = np.array([[p**h for p in peaks] for h in range(1, n_harm + 1)])
        multi_harmonics = np.moveaxis(multi_harmonics, 0, 1)
    
    # ========================================================================
    # Panel 1 (Top-Left): Harmonic Network Graph
    # ========================================================================
    _, ax1 = plot_harmonic_fit_network(
        peaks=peaks,
        amps=amps,
        multi_harmonics=multi_harmonics,
        n_harm=n_harm,
        harm_bounds=harm_bounds,
        function=function,
        ax=ax1,
        **kwargs
    )
    
    # ========================================================================
    # Panel 2 (Bottom-Left): Harmonic Connectivity Matrix
    # ========================================================================
    _, ax3 = plot_harmonic_fit_matrix(
        peaks=peaks,
        multi_harmonics=multi_harmonics,
        n_harm=n_harm,
        harm_bounds=harm_bounds,
        ax=ax3,
        **kwargs
    )
    
    # ========================================================================
    # Right Side (Panels 2 & 4): Harmonic Positions
    # ========================================================================
    # Use remaining grid space for position plots
    gs_right = gs[0:2, 1].subgridspec(2, 1, hspace=0.3)
    ax2 = fig.add_subplot(gs_right[0])
    ax4 = fig.add_subplot(gs_right[1])
    
    # Plot harmonic positions directly into our axes
    _, _ = plot_harmonic_fit_positions(
        peaks=peaks,
        amps=amps,
        multi_harmonics=multi_harmonics,
        common_harms=common_harms,
        n_harm=n_harm,
        harm_bounds=harm_bounds,
        function=function,
        ax_left=ax2,
        ax_right=ax4,
        **kwargs
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Return axes as 2x2 array for compatibility
    axes = np.array([[ax1, ax2], [ax3, ax4]])
    return fig, axes


# Backward compatibility alias
plot_harmonic_fit = plot_harmonic_fit_summary




def plot_harmonic_position_mappings(
    peaks: np.ndarray,
    amps: np.ndarray,
    multi_harmonics: np.ndarray,
    n_harm: int = 10,
    harm_bounds: float = 0.5,
    function: str = 'mult',
    figsize: tuple = (14, 10),
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot harmonic position mappings between peak pairs in a circular network layout.
    
    Shows which specific harmonic positions (1st, 2nd, 3rd, etc.) of one peak
    match with which harmonic positions of another peak. Edges are labeled with
    the matching harmonic positions (e.g., "2nd↔5th" means the 2nd harmonic of 
    one peak matches the 5th harmonic of another).
    
    Parameters
    ----------
    peaks : np.ndarray
        Peak frequencies in Hz
    amps : np.ndarray
        Peak amplitudes
    multi_harmonics : np.ndarray
        Pre-computed harmonic series for each peak (n_peaks x n_harm)
    n_harm : int, default=10
        Number of harmonics per peak
    harm_bounds : float, default=0.5
        Frequency threshold (Hz) for considering harmonics as matching
    function : str, default='mult'
        Harmonic function used ('mult' or 'div')
    figsize : tuple, default=(14, 10)
        Figure size
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object
    """
    set_biotuner_style()
    
    n_peaks = len(peaks)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Collect specific harmonic position matches between peak pairs
    peak_pair_harmonics = {}
    
    for i in range(n_peaks):
        for j in range(i+1, n_peaks):
            harms_i = multi_harmonics[i]
            harms_j = multi_harmonics[j]
            matches = []
            
            for h_i_idx, hi in enumerate(harms_i):
                for h_j_idx, hj in enumerate(harms_j):
                    if abs(hi - hj) < harm_bounds:
                        h_i_pos = h_i_idx + 1
                        h_j_pos = h_j_idx + 1
                        matches.append((h_i_pos, h_j_pos, (hi + hj) / 2))
            
            if matches:
                peak_pair_harmonics[(i, j)] = matches
    
    if peak_pair_harmonics:
        # Circular layout for peaks
        theta = np.linspace(0, 2*np.pi, n_peaks, endpoint=False)
        node_x = 0.5 + 0.32 * np.cos(theta)
        node_y = 0.5 + 0.38 * np.sin(theta)
        
        # Assign colors to peaks
        cmap = plt.cm.viridis
        colors = [cmap(i / n_peaks) for i in range(n_peaks)]
        
        # Draw edges with harmonic position labels
        for (i, j), matches in peak_pair_harmonics.items():
            x1, y1 = node_x[i], node_y[i]
            x2, y2 = node_x[j], node_y[j]
            
            n_matches = len(matches)
            
            # Style edges by number of matches
            if n_matches >= 3:
                linewidth = 3
                alpha = 0.6
                color = BIOTUNER_COLORS['success']
            elif n_matches >= 2:
                linewidth = 2
                alpha = 0.5
                color = BIOTUNER_COLORS['warning']
            else:
                linewidth = 1.2
                alpha = 0.35
                color = BIOTUNER_COLORS['dark']
            
            ax.plot([x1, x2], [y1, y2], color=color, 
                    linewidth=linewidth, alpha=alpha, zorder=1)
            
            # Add label showing harmonic position matches
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            
            label_lines = []
            for h_i, h_j, freq in matches[:2]:  # Show top 2 matches
                suffix_i = {1: 'st', 2: 'nd', 3: 'rd'}.get(h_i, 'th')
                suffix_j = {1: 'st', 2: 'nd', 3: 'rd'}.get(h_j, 'th')
                label_lines.append(f"{h_i}{suffix_i}↔{h_j}{suffix_j}")
            
            if n_matches > 2:
                label_lines.append(f"+{n_matches-2}")
            
            label_text = '\n'.join(label_lines)
            
            # Offset label perpendicular to edge
            offset_angle = np.arctan2(y2 - y1, x2 - x1) + np.pi/2
            label_offset = 0.025
            label_x = mid_x + label_offset * np.cos(offset_angle)
            label_y = mid_y + label_offset * np.sin(offset_angle)
            
            ax.text(label_x, label_y, label_text, 
                    fontsize=9, ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor=color, alpha=0.9, linewidth=1.5),
                    fontweight='bold', color=color)
        
        # Draw peak nodes
        for i, (peak, x, y, color) in enumerate(zip(peaks, node_x, node_y, colors)):
            norm_amp = (amps[i] - amps.min()) / (amps.max() - amps.min() + 1e-10)
            node_size = 500 + norm_amp * 800
            
            ax.scatter(x, y, s=node_size, c=[color], 
                       edgecolors=BIOTUNER_COLORS['dark'], linewidths=2.5, 
                       zorder=5, alpha=0.95)
            
            # Position label outside the node
            label_angle = theta[i]
            label_offset = 0.14
            label_x = 0.5 + (0.32 + label_offset) * np.cos(label_angle)
            label_y = 0.5 + (0.38 + label_offset) * np.sin(label_angle)
            
            ax.text(label_x, label_y, f'{peak:.1f} Hz', 
                    fontsize=12, ha='center', va='center', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor=color, 
                             edgecolor=BIOTUNER_COLORS['dark'], alpha=0.95, linewidth=2.5),
                    color='white')
        
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color=BIOTUNER_COLORS['success'], linewidth=3,
                   label='≥3 matches'),
            Line2D([0], [0], color=BIOTUNER_COLORS['warning'], linewidth=2,
                   label='2 matches'),
            Line2D([0], [0], color=BIOTUNER_COLORS['dark'], linewidth=1.2,
                   label='1 match')
        ]
        ax.legend(handles=legend_elements, loc='upper right', 
                 fontsize=11, framealpha=0.95, title='Connection Strength',
                 bbox_to_anchor=(1.0, 1.0))
        
        # Add explanation box
        explanation = (
            "Edge labels show which harmonic of peak A ↔ which harmonic of peak B\n"
            "e.g., '2nd↔5th' = 2nd harmonic of one matches 5th of the other"
        )
        ax.text(0.5, 0.02, explanation, transform=ax.transAxes,
               fontsize=10, ha='center', va='bottom',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                        alpha=0.85, edgecolor=BIOTUNER_COLORS['dark'], linewidth=1.5))
    else:
        ax.text(0.5, 0.5, 'No harmonic matches between peaks',
               ha='center', va='center', fontsize=16, transform=ax.transAxes,
               fontweight='bold', color=BIOTUNER_COLORS['dark'])
        ax.axis('off')
    
    ax.set_title(f'Harmonic Position Mappings Between Peak Pairs\n({function.upper()}, n_harm={n_harm}, tolerance=±{harm_bounds} Hz)',
                 fontsize=20, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    return fig, ax


# ============================================================================
# Export public API
# ============================================================================

__all__ = [
    # Peak extraction plots
    'plot_psd_peaks',
    'plot_emd_peaks',
    'plot_harmonic_peaks',
    'plot_peaks',
    # Tuning plots
    'plot_tuning_dissonance',
    'plot_tuning_entropy',
    'plot_tuning_harmonic',
    'plot_tuning',
    'plot_tuning_comparison',
    'plot_tuning_complete',
    # Harmonic analysis plots
    'plot_harmonic_fit',
    'plot_harmonic_position_mappings',
]
