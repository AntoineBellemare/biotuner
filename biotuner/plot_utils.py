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

def plot_peaks(
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
    Universal plotting function for peak extraction results.
    
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


def plot_tuning_harmonic(
    tuning: List[float],
    peaks: Optional[np.ndarray] = None,
    octave: float = 2,
    show_cents: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot harmonic tuning visualization with unified biotuner styling.
    
    Parameters
    ----------
    tuning : list of float
        Scale/tuning as frequency ratios
    peaks : np.ndarray, optional
        Original spectral peaks
    octave : float, default=2
        Octave value
    show_cents : bool, default=True
        Show cent values above notes
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
        figsize = (14, 6)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Convert ratios to cents
    cents = [1200 * np.log2(ratio) for ratio in tuning]
    
    # Create color palette
    n_notes = len(tuning)
    colors = get_color_palette('biotuner_gradient', n_colors=n_notes)
    
    # Plot tuning as bars
    bars = ax.bar(range(n_notes), cents, color=colors, 
                  edgecolor=BIOTUNER_COLORS['dark'], linewidth=1.5,
                  alpha=0.8)
    
    # Add cent values on top if requested
    if show_cents:
        for i, (cent, ratio) in enumerate(zip(cents, tuning)):
            # Convert ratio to fraction
            frac = Fraction(ratio).limit_denominator(1000)
            ratio_str = f'{frac.numerator}/{frac.denominator}' if frac.denominator != 1 else str(frac.numerator)
            ax.text(i, cent + 20, f'{cent:.0f}¢\n{ratio_str}', 
                   ha='center', va='bottom', fontsize=10, fontweight='semibold')
    
    # Styling
    ax.set_xlabel('Scale Degree', fontsize=16, fontweight='normal')
    ax.set_ylabel('Cents', fontsize=16, fontweight='normal')
    ax.set_title('Harmonic Tuning', fontsize=20, fontweight='bold', pad=15)
    ax.set_xticks(range(n_notes))
    ax.set_xticklabels([f'{i+1}' for i in range(n_notes)], fontsize=13)
    ax.axhline(1200, color=BIOTUNER_COLORS['dark'], linestyle='--', 
              linewidth=1.5, alpha=0.5, label='Octave (1200¢)')
    ax.legend(fontsize=13, framealpha=0.95)
    ax.grid(True, alpha=0.25, axis='y')
    
    plt.tight_layout()
    return fig, ax


def plot_tuning(
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
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot comprehensive tuning analysis with 2, 4, or 5 panels (with summary).
    
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
    
    # Auto-adjust figsize based on panels
    if figsize is None:
        if panels == 2:
            figsize = (16, 6)
        elif show_summary and panels >= 4:
            figsize = (16, 16)  # Extra height for summary panel
        else:  # 4 panels
            figsize = (16, 13)
    
    # Create figure with appropriate layout
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
    
    # LEFT PANEL: Tuning visualization
    # Convert ratios to cents
    cents = [1200 * np.log2(ratio) if ratio > 0 else 0 for ratio in scale]
    n_notes = len(scale)
    colors = get_color_palette('biotuner_gradient', n_colors=n_notes)
    
    bars = axes[0].bar(range(n_notes), cents, color=colors, 
                   edgecolor=BIOTUNER_COLORS['dark'], linewidth=1.5, alpha=0.8)
    
    # Add values on top with fractions
    for i, (cent, ratio) in enumerate(zip(cents, scale)):
        # Convert ratio to fraction
        frac = Fraction(ratio).limit_denominator(1000)
        ratio_str = f'{frac.numerator}/{frac.denominator}' if frac.denominator != 1 else str(frac.numerator)
        axes[0].text(i, cent + 20, f'{cent:.0f}¢\n{ratio_str}', 
                ha='center', va='bottom', fontsize=9, fontweight='semibold')
    
    axes[0].set_xlabel('Interval Index', fontsize=16, fontweight='normal')
    axes[0].set_ylabel('Cents', fontsize=16, fontweight='normal')
    axes[0].set_title('Tuning Scale', fontsize=20, fontweight='bold', pad=15)
    axes[0].set_xticks(range(n_notes))
    axes[0].set_xticklabels([f'{i+1}' for i in range(n_notes)], fontsize=13)
    axes[0].axhline(1200, color=BIOTUNER_COLORS['dark'], linestyle='--', 
               linewidth=1.5, alpha=0.5, label='Octave')
    axes[0].legend(fontsize=12, framealpha=0.95)
    axes[0].grid(True, alpha=0.25, axis='y')
    
    # RIGHT PANEL: Consonance matrix using biotuner's tuning_cons_matrix
    # Map metric names to functions
    metric_functions = {
        'harmsim': dyad_similarity,
        'cons': compute_consonance,
        'tenney': lambda r: 1 / (1 + np.log2(Fraction(r).limit_denominator(1000).numerator * 
                                              Fraction(r).limit_denominator(1000).denominator)),
        'denom': lambda r: 1 / (1 + Fraction(r).limit_denominator(1000).denominator),
    }
    
    metric_labels = {
        'harmsim': 'Harmonic Similarity',
        'cons': 'Consonance',
        'tenney': 'Tenney Height (inv.)',
        'denom': 'Denominator (inv.)',
        'subharm_tension': 'Subharm Tension (inv.)'
    }
    
    # Compute consonance matrix manually with proper ratio_type handling
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
                frac = Fraction(ratio).limit_denominator(1000)
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
    
    # Plot heatmap
    im = axes[1].imshow(cons_matrix, cmap=cmap, vmin=vmin, vmax=vmax, 
                    aspect='auto', origin='lower')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1])
    cbar.set_label(metric_labels.get(metric, 'Metric'), fontsize=14, fontweight='normal')
    cbar.ax.tick_params(labelsize=12)
    
    # Add labels with fractions
    n_values = len(scale)
    axes[1].set_xticks(range(n_values))
    axes[1].set_yticks(range(n_values))
    
    # Convert scale values to fractions for labels
    fraction_labels = []
    for v in scale:
        frac = Fraction(v).limit_denominator(1000)
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
                 fontsize=20, fontweight='bold', pad=15)
    
    # ===== EXTRA PANELS (if panels >= 4) =====
    if panels >= 4 and len(axes) >= 4:
        # Panel 3: Step sizes or interval distribution
        if 'step_sizes' in extra_panels:
            # Compute step sizes (melodic intervals between adjacent notes)
            step_cents = []
            for i in range(len(scale) - 1):
                step_ratio = scale[i + 1] / scale[i]
                step_cents.append(1200 * np.log2(step_ratio))
            
            # Plot step sizes
            step_colors = get_color_palette('biotuner_gradient', n_colors=len(step_cents))
            bars = axes[2].bar(range(len(step_cents)), step_cents, color=step_colors,
                              edgecolor=BIOTUNER_COLORS['dark'], linewidth=1.5, alpha=0.8)
            
            # Add values on bars
            for i, cent in enumerate(step_cents):
                axes[2].text(i, cent + 5, f'{cent:.0f}¢', 
                           ha='center', va='bottom', fontsize=10, fontweight='semibold')
            
            # Reference lines for common intervals
            axes[2].axhline(100, color=BIOTUNER_COLORS['secondary'], linestyle=':', 
                          linewidth=1.5, alpha=0.5, label='Semitone (100¢)')
            axes[2].axhline(200, color=BIOTUNER_COLORS['accent'], linestyle=':', 
                          linewidth=1.5, alpha=0.5, label='Whole tone (200¢)')
            
            axes[2].set_xlabel('Step Number', fontsize=16, fontweight='normal')
            axes[2].set_ylabel('Step Size (cents)', fontsize=16, fontweight='normal')
            axes[2].set_title('Melodic Intervals (Step Sizes)', fontsize=20, fontweight='bold', pad=15)
            axes[2].set_xticks(range(len(step_cents)))
            axes[2].set_xticklabels([f'{i+1}→{i+2}' for i in range(len(step_cents))], 
                                   fontsize=11, rotation=45)
            axes[2].legend(fontsize=12, framealpha=0.95, loc='upper right')
            axes[2].grid(True, alpha=0.25, axis='y')
            
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
            axes[2].set_title('Interval Distribution', fontsize=20, fontweight='bold', pad=15)
            axes[2].grid(True, alpha=0.25, axis='y')
        
        # Panel 4: Consonance profile or harmonic deviation
        if 'consonance_profile' in extra_panels:
            # Collect consonance distributions for each scale degree
            consonance_distributions = []
            for i in range(len(scale)):
                row_values = cons_matrix[i, :]
                # Exclude diagonal (self-comparison) and zeros
                non_diag = [row_values[j] for j in range(len(row_values)) if j != i and row_values[j] != 0]
                if len(non_diag) > 0:
                    consonance_distributions.append(non_diag)
                else:
                    consonance_distributions.append([0])  # Fallback
            
            # Create violin plot
            positions = list(range(len(consonance_distributions)))
            parts = axes[3].violinplot(
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
            
            axes[3].set_xlabel('Scale Degree', fontsize=16, fontweight='normal')
            axes[3].set_ylabel(f'{metric_labels.get(metric, "Consonance")} Distribution', fontsize=16, fontweight='normal')
            axes[3].set_title('Consonance Profile (Violin Plot)', fontsize=20, fontweight='bold', pad=15)
            axes[3].set_xticks(positions)
            axes[3].set_xticklabels([f'{i+1}' for i in range(len(consonance_distributions))], fontsize=13)
            axes[3].grid(True, alpha=0.25, axis='y')
            
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
            axes[3].set_title('Harmonic Deviation', fontsize=20, fontweight='bold', pad=15)
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
            # Title
            axes[4].text(0.50, 0.95, f'Known Intervals ({len(matched_intervals)} matched)', 
                        transform=axes[4].transAxes,
                        fontsize=18, fontweight='bold',
                        ha='center',
                        color=BIOTUNER_COLORS['dark'])
            
            # Prepare table data
            sorted_intervals = sorted(matched_intervals, key=lambda x: x[0])
            
            # Limit to 10 rows to fit nicely (more rows since we have more space)
            max_rows = 10
            table_data = []
            
            for ratio, name, deviation in sorted_intervals[:max_rows]:
                ratio_cents = 1200 * np.log2(ratio)
                # Convert ratio to fraction
                frac = Fraction(ratio).limit_denominator(1000)
                ratio_str = f'{frac.numerator}/{frac.denominator}' if frac.denominator != 1 else str(frac.numerator)
                # Don't truncate interval names
                display_name = name
                table_data.append([
                    ratio_str,
                    f'{ratio_cents:.1f}¢',
                    display_name,
                    f'±{deviation:.1f}¢'
                ])
            
            # Create table - FULL WIDTH
            if table_data:
                col_labels = ['Ratio', 'Cents', 'Interval Name', 'Dev']
                
                # Calculate table position and size - FULL WIDTH
                table_height = min(0.82, 0.075 * (len(table_data) + 1))
                table_width = 0.95  # Nearly full width
                table_left = 0.025  # Small left margin
                
                table = axes[4].table(
                    cellText=table_data,
                    colLabels=col_labels,
                    cellLoc='left',
                    loc='center',
                    bbox=[table_left, 0.90 - table_height, table_width, table_height]
                )
                
                table.auto_set_font_size(False)
                table.set_fontsize(14)  # BIGGER font
                table.scale(1, 2.0)  # BIGGER row height
                
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
                    cell.set_text_props(weight='bold', color='white', fontsize=15)  # BIGGER header
                    cell.set_edgecolor(BIOTUNER_COLORS['dark'])
                    cell.set_linewidth(2.0)  # Thicker border
                
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
                            dev_val = matched_intervals[i-1][2]
                            if dev_val < 2:
                                cell.set_text_props(color='#28a745', weight='bold', fontsize=14)
                            elif dev_val < 4:
                                cell.set_text_props(color='#ffc107', weight='bold', fontsize=14)
                            else:
                                cell.set_text_props(fontsize=14)
                        else:
                            cell.set_text_props(fontsize=14)
                
                # Show count if more intervals exist
                if len(matched_intervals) > max_rows:
                    axes[4].text(0.50, 0.05,
                               f'+ {len(matched_intervals) - max_rows} more intervals...',
                               transform=axes[4].transAxes,
                               fontsize=12, style='italic',
                               ha='center',
                               color=BIOTUNER_COLORS['dark'])
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
    
    # Use tight_layout only for 2-panel mode; custom spacing already set for 4/5-panel
    if panels == 2:
        plt.tight_layout()
    
    return fig


def plot_tuning_comparison(
    biotuner_obj,
    panels: List[str] = ['dissonance', 'entropy', 'harmonic', 'ratios'],
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs
) -> plt.Figure:
    """
    Create multi-panel tuning comparison with unified biotuner styling.
    
    Parameters
    ----------
    biotuner_obj : compute_biotuner
        Biotuner object with computed tuning metrics
    panels : list of str
        Panels to include: 'dissonance', 'entropy', 'harmonic', 'ratios'
    figsize : tuple, optional
        Figure size
    
    Returns
    -------
    fig : matplotlib figure
    """
    set_biotuner_style()
    
    if figsize is None:
        figsize = (16, 14)
    
    n_panels = len(panels)
    if n_panels == 4:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        axes = [ax1, ax2, ax3, ax4]
    elif n_panels == 3:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        axes = [ax1, ax2, ax3]
    elif n_panels == 2:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        axes = [ax1, ax2]
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
        axes = [ax1]
    
    # Plot each panel
    panel_idx = 0
    for panel_name in panels:
        if panel_name == 'ratios' and hasattr(biotuner_obj, 'peaks'):
            # Ratios plot creates its own 2-panel figure, skip for multi-panel comparison
            continue
        
        ax = axes[panel_idx]
        panel_idx += 1
        
        if panel_name == 'dissonance' and hasattr(biotuner_obj, 'diss'):
            plot_tuning_dissonance(
                biotuner_obj.ratio_diss,
                biotuner_obj.diss,
                biotuner_obj.diss_scale if hasattr(biotuner_obj, 'diss_scale') else None,
                ax=ax
            )
        elif panel_name == 'entropy' and hasattr(biotuner_obj, 'HE'):
            plot_tuning_entropy(
                biotuner_obj.ratio_HE,
                biotuner_obj.HE,
                biotuner_obj.HE_scale if hasattr(biotuner_obj, 'HE_scale') else None,
                ax=ax
            )
        elif panel_name == 'harmonic' and hasattr(biotuner_obj, 'peaks'):
            harmonic_tuning = biotuner_obj.harmonic_tuning()
            plot_tuning_harmonic(harmonic_tuning, biotuner_obj.peaks, ax=ax)
    
    plt.tight_layout()
    return fig


def plot_tuning_complete(
    biotuner_obj,
    include_psd: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    **kwargs
) -> plt.Figure:
    """
    Create comprehensive tuning visualization panel with all analyses.
    
    Parameters
    ----------
    biotuner_obj : compute_biotuner
        Biotuner object with computed tuning metrics
    include_psd : bool, default=True
        Include PSD with peaks at top
    figsize : tuple, optional
        Figure size
    
    Returns
    -------
    fig : matplotlib figure
    """
    set_biotuner_style()
    
    if figsize is None:
        figsize = (18, 16)
    
    if include_psd:
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Top: PSD with peaks (spanning both columns)
        ax_psd = fig.add_subplot(gs[0, :])
        if hasattr(biotuner_obj, 'freqs') and hasattr(biotuner_obj, 'psd'):
            plot_psd_peaks(
                biotuner_obj.freqs,
                biotuner_obj.psd,
                biotuner_obj.peaks,
                xmin=1, xmax=50,
                title="Power Spectral Density with Detected Peaks",
                show_bands=True,
                ax=ax_psd
            )
        
        # Middle row: Dissonance and Entropy
        ax_diss = fig.add_subplot(gs[1, 0])
        ax_entropy = fig.add_subplot(gs[1, 1])
        
        # Bottom row: Harmonic tuning and Ratios
        ax_harm = fig.add_subplot(gs[2, 0])
        ax_ratios = fig.add_subplot(gs[2, 1])
        
    else:
        fig, ((ax_diss, ax_entropy), (ax_harm, ax_ratios)) = plt.subplots(
            2, 2, figsize=figsize
        )
    
    # Plot dissonance
    if hasattr(biotuner_obj, 'diss'):
        plot_tuning_dissonance(
            biotuner_obj.ratio_diss,
            biotuner_obj.diss,
            biotuner_obj.diss_scale if hasattr(biotuner_obj, 'diss_scale') else None,
            ax=ax_diss
        )
    
    # Plot entropy
    if hasattr(biotuner_obj, 'HE'):
        plot_tuning_entropy(
            biotuner_obj.ratio_HE,
            biotuner_obj.HE,
            biotuner_obj.HE_scale if hasattr(biotuner_obj, 'HE_scale') else None,
            ax=ax_entropy
        )
    
    # Plot harmonic tuning
    if hasattr(biotuner_obj, 'peaks'):
        harmonic_tuning = biotuner_obj.harmonic_tuning()
        plot_tuning_harmonic(harmonic_tuning, biotuner_obj.peaks, ax=ax_harm)
    
    # Note: Ratios panel now creates its own 2-panel figure
    # For complete view, show it separately or modify layout
    ax_ratios.text(0.5, 0.5, 'Use plot_tuning_ratios()\nseparately for\ntuning + matrix view',
                  ha='center', va='center', fontsize=14, transform=ax_ratios.transAxes)
    ax_ratios.set_title('Peaks Ratios Analysis', fontsize=20, fontweight='bold', pad=15)
    ax_ratios.axis('off')
    
    plt.tight_layout()
    return fig


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
    'plot_tuning_ratios',
    'plot_tuning_comparison',
    'plot_tuning_complete',
]
