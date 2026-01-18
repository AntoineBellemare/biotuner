"""
Biotuner Plotting Configuration
================================

This module provides consistent color schemes, styles, and configuration
for all biotuner visualizations.

Author: Biotuner Team
"""

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, Any, Optional

# ============================================================================
# Color Palettes
# ============================================================================

BIOTUNER_COLORS = {
    'primary': '#2E86AB',      # Blue - main signal
    'secondary': '#A23B72',    # Purple - harmonics
    'accent': '#F18F01',       # Orange - peaks/markers
    'success': '#06A77D',      # Green - positive indicators
    'warning': '#D4AF37',      # Gold - warnings
    'danger': '#C73E1D',       # Red - errors/references
    'dark': '#1F1F1F',         # Dark gray - text
    'light': '#E8E9EB',        # Light gray - backgrounds
}

# EMD/IMF color gradients (from deep to light)
EMD_COLORS = ['#0B3954', '#087E8B', '#3AAFA9', '#BFD7EA', '#DEF2F1']

# Frequency band colors (standard EEG bands)
BAND_COLORS = {
    'delta': '#8B4513',    # Brown
    'theta': '#DAA520',    # Goldenrod
    'alpha': '#FFA500',    # Orange
    'beta': '#FFD700',     # Gold
    'gamma': '#F0E68C',    # Khaki
}

BAND_NAMES = ['delta', 'theta', 'alpha', 'beta', 'gamma']

# Standard frequency bands (Hz)
FREQ_BANDS = {
    'delta': [0.5, 4],
    'theta': [4, 8],
    'alpha': [8, 13],
    'beta': [13, 30],
    'gamma': [30, 60],
}

# Alternative color palettes for different use cases
PALETTES = {
    'viridis': 'viridis',
    'plasma': 'plasma',
    'coolwarm': 'coolwarm',
    'spectral': 'Spectral',
    'biotuner_gradient': sns.color_palette([
        BIOTUNER_COLORS['primary'],
        BIOTUNER_COLORS['secondary'],
        BIOTUNER_COLORS['accent'],
        BIOTUNER_COLORS['success'],
        BIOTUNER_COLORS['warning']
    ]),
    'biotuner_matrix': sns.blend_palette([
        '#1A1A2E',  # Dark navy
        '#16213E',  # Navy blue
        '#0F4C75',  # Ocean blue
        '#3282B8',  # Bright blue
        '#BBE1FA',  # Light blue
        '#FFFFFF',  # White (center)
        '#FFE5B4',  # Peach
        '#FFAD60',  # Light orange
        '#F18F01',  # Orange
        '#C73E1D',  # Red-orange
        '#9A031E'   # Deep red
    ], as_cmap=False),
    'biotuner_diverging': sns.blend_palette([
        '#0B3954',  # Deep blue
        '#2E86AB',  # Primary blue
        '#E8E9EB',  # Light neutral
        '#F18F01',  # Orange
        '#C73E1D'   # Red
    ], as_cmap=False),
}

# Create matplotlib colormaps from the biotuner palettes
# Use the same orange-to-turquoise palette as in plot_utils for consistency
BIOTUNER_MATRIX_CMAP = LinearSegmentedColormap.from_list(
    'biotuner_matrix',
    ['#8B4513', '#CD5C5C', '#F4A460', '#FFD700', '#90EE90', '#48D1CC', '#40E0D0'],
    N=256
)

BIOTUNER_DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    'biotuner_diverging',
    ['#0B3954', '#2E86AB', '#E8E9EB', '#F18F01', '#C73E1D'],
    N=256
)

# ============================================================================
# Default Plot Parameters
# ============================================================================

DEFAULT_STYLE = {
    'figure.figsize': (12, 7),
    'figure.dpi': 100,
    'figure.facecolor': 'white',
    
    # Font sizes - increased for better readability
    'font.size': 14,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'axes.labelsize': 16,
    'axes.titlesize': 20,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'legend.title_fontsize': 14,
    
    # Line properties - modern, clean look
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'lines.antialiased': True,
    
    # Grid - subtle and modern
    'grid.alpha': 0.25,
    'grid.linestyle': '-',
    'grid.linewidth': 0.6,
    'grid.color': '#E0E0E0',
    
    # Axes - clean, modern appearance
    'axes.linewidth': 1.8,
    'axes.edgecolor': '#424242',
    'axes.grid': True,
    'axes.axisbelow': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelweight': 'normal',
    'axes.labelcolor': '#333333',
    
    # Legend - modern styling
    'legend.frameon': True,
    'legend.framealpha': 0.95,
    'legend.fancybox': True,
    'legend.shadow': False,
    'legend.edgecolor': '#CCCCCC',
}

# Specific configurations for different plot types
PLOT_CONFIGS = {
    'psd': {
        'figsize': (12, 7),
        'xlabel': 'Frequency (Hz)',
        'ylabel': 'Power Spectral Density',
        'grid': True,
        'title_prefix': 'Power Spectrum - ',
        'band_label_position': 'top',  # Consistent positioning
    },
    'emd': {
        'figsize': (14, 8),
        'xlabel': 'Frequency (Hz)',
        'ylabel': 'Power',
        'grid': True,
        'title': 'Empirical Mode Decomposition',
        'xscale': 'log',
        'yscale': 'symlog',
        'band_label_position': 'top',  # Consistent positioning
    },
    'harmonic': {
        'figsize': (12, 7),
        'xlabel': 'Frequency (Hz)',
        'ylabel': 'Power Spectral Density',
        'grid': True,
        'title': 'Harmonic Recurrence',
        'band_label_position': 'top',  # Consistent positioning
    },
    'entropy': {
        'figsize': (12, 5),
        'xlabel': 'Frequency Ratio',
        'ylabel': 'Harmonic Entropy',
        'grid': True,
        'title': 'Harmonic Entropy Curve',
    },
    'dissonance': {
        'figsize': (14, 7),
        'xlabel': 'Frequency Ratio',
        'ylabel': 'Dissonance',
        'grid': True,
        'title': 'Dissonance Curve',
    },
    'rhythm': {
        'figsize': (9, 9),
        'aspect': 'equal',
        'title': 'Euclidean Rhythms',
    },
    # Group-level plots (BiotunerGroup)
    'group_peaks': {
        'figsize': (14, 7),
        'xlabel': 'Frequency (Hz)',
        'ylabel': 'Power',
        'grid': True,
        'title_prefix': 'Group Peak Spectrum - ',
        'aggregate_color': '#2E86AB',
        'individual_alpha': 0.08,
        'peak_marker_color': '#F18F01',
        'peak_marker_size': 10,
        'band_label_position': 'top',
    },
    'group_metric_dist': {
        'figsize': (11, 7),
        'title_prefix': 'Metric Distribution - ',
        'palette': 'biotuner_gradient',
    },
    'group_metric_matrix': {
        'figsize': None,  # Calculated dynamically
        'cmap': 'biotuner_matrix',
        'title_prefix': 'Metric Matrix - ',
        'linewidths': 0.8,
        'linecolor': 'white',
    },
    'group_tuning_histogram': {
        'figsize': (14, 7),
        'xlabel': 'Interval Ratio',
        'ylabel': 'Frequency',
        'title_prefix': 'Interval Distribution - ',
        'bins': 50,
        'color': '#2E86AB',
        'alpha': 0.75,
    },
    'group_tuning_common': {
        'figsize': (12, 8),
        'xlabel': 'Number of Occurrences',
        'title_prefix': 'Most Common Intervals - ',
        'color': '#2E86AB',
        'bar_alpha': 0.85,
    },
    'group_scale_dist': {
        'figsize': (12, 7),
        'xlabel': 'Scale Size (number of notes)',
        'ylabel': 'Frequency',
        'title_prefix': 'Scale Size Distribution - ',
        'color': '#2E86AB',
        'alpha': 0.75,
        'mean_line_color': '#C73E1D',
    },
    'group_tuning_comparison': {
        'figsize': (14, 8),
        'xlabel': 'Interval Ratio',
        'ylabel': 'Time Series',
        'title_prefix': 'Tuning Comparison - ',
        'line_width': 2.5,
        'alpha': 0.8,
    },
    'group_peak_distribution': {
        'figsize': (14, 6),
        'xlabel': 'Frequency (Hz)',
        'ylabel': 'Count',
        'title': 'Distribution of All Detected Peaks',
        'bins': 50,
        'color': '#88D8E8',
        'edgecolor': 'white',
        'alpha': 0.85,
        'band_line_color': '#C73E1D',
        'band_line_alpha': 0.6,
        'band_text_color': '#C73E1D',
    },
    'group_comparison': {
        'figsize': (11, 7),
        'palette': 'biotuner_gradient',
        'title_prefix': 'Group Comparison - ',
        'point_size': 5,
        'point_alpha': 0.4,
        'point_color': '#1F1F1F',
    },
}

# ============================================================================
# Style Functions
# ============================================================================

def set_biotuner_style(style: str = 'whitegrid'):
    """
    Apply consistent biotuner plotting style globally.
    
    Parameters
    ----------
    style : str, default='whitegrid'
        Seaborn style to use as base. Options: 'whitegrid', 'darkgrid', 
        'white', 'dark', 'ticks'
    
    Examples
    --------
    >>> from biotuner.plot_config import set_biotuner_style
    >>> set_biotuner_style()
    """
    sns.set_style(style)
    plt.rcParams.update(DEFAULT_STYLE)


def reset_style():
    """Reset matplotlib to default style."""
    plt.rcdefaults()


def get_color_palette(name: str = 'biotuner_gradient', n_colors: Optional[int] = None):
    """
    Get a color palette for plotting.
    
    Parameters
    ----------
    name : str, default='biotuner_gradient'
        Name of the palette. Options: 'viridis', 'plasma', 'coolwarm',
        'spectral', 'biotuner_gradient'
    n_colors : int, optional
        Number of colors to return. If None, returns full palette.
    
    Returns
    -------
    palette : list
        List of color codes
    
    Examples
    --------
    >>> colors = get_color_palette('biotuner_gradient', n_colors=5)
    """
    if name in PALETTES:
        if isinstance(PALETTES[name], str):
            return sns.color_palette(PALETTES[name], n_colors=n_colors)
        else:
            palette = PALETTES[name]
            if n_colors:
                # If requesting more colors than available, interpolate
                if n_colors > len(palette):
                    # Use seaborn to blend the palette into more colors
                    return sns.blend_palette(palette, n_colors=n_colors)
                else:
                    return palette[:n_colors]
            return palette
    else:
        # Fallback to seaborn palette
        return sns.color_palette(name, n_colors=n_colors)


def get_emd_colors(n_imfs: int = 5):
    """
    Get colors for EMD/IMF plotting.
    
    Parameters
    ----------
    n_imfs : int, default=5
        Number of IMFs to generate colors for
    
    Returns
    -------
    colors : list
        List of color codes for each IMF
    
    Examples
    --------
    >>> colors = get_emd_colors(n_imfs=5)
    """
    if n_imfs <= len(EMD_COLORS):
        return EMD_COLORS[:n_imfs]
    else:
        # Generate additional colors using gradient
        return sns.color_palette('viridis', n_colors=n_imfs)


def get_band_colors(bands: Optional[list] = None):
    """
    Get colors for frequency bands.
    
    Parameters
    ----------
    bands : list, optional
        List of band names. If None, uses standard EEG bands.
    
    Returns
    -------
    colors : list or dict
        Colors for each band
    
    Examples
    --------
    >>> colors = get_band_colors(['delta', 'theta', 'alpha'])
    """
    if bands is None:
        return BAND_COLORS
    else:
        return [BAND_COLORS.get(band, BIOTUNER_COLORS['primary']) for band in bands]


# ============================================================================
# Plot Configuration Helpers
# ============================================================================

def get_plot_config(plot_type: str) -> Dict[str, Any]:
    """
    Get configuration dictionary for a specific plot type.
    
    Parameters
    ----------
    plot_type : str
        Type of plot: 'psd', 'emd', 'harmonic', 'entropy', 'dissonance', 'rhythm'
    
    Returns
    -------
    config : dict
        Configuration parameters for the plot
    
    Examples
    --------
    >>> config = get_plot_config('psd')
    >>> fig, ax = plt.subplots(figsize=config['figsize'])
    """
    return PLOT_CONFIGS.get(plot_type, PLOT_CONFIGS['psd']).copy()


def update_plot_config(plot_type: str, **kwargs) -> Dict[str, Any]:
    """
    Get plot configuration and update with custom parameters.
    
    Parameters
    ----------
    plot_type : str
        Type of plot
    **kwargs
        Custom parameters to override defaults
    
    Returns
    -------
    config : dict
        Updated configuration
    
    Examples
    --------
    >>> config = update_plot_config('psd', figsize=(12, 8), title='My Custom Title')
    """
    config = get_plot_config(plot_type)
    config.update(kwargs)
    return config


# ============================================================================
# Context Managers
# ============================================================================

class biotuner_style:
    """
    Context manager for temporarily applying biotuner style.
    
    Examples
    --------
    >>> with biotuner_style():
    ...     plt.plot([1, 2, 3], [1, 4, 9])
    ...     plt.show()
    """
    
    def __init__(self, style: str = 'whitegrid'):
        self.style = style
        self.original_params = plt.rcParams.copy()
    
    def __enter__(self):
        set_biotuner_style(self.style)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.rcParams.update(self.original_params)


# ============================================================================
# Export all public symbols
# ============================================================================

__all__ = [
    'BIOTUNER_COLORS',
    'EMD_COLORS',
    'BAND_COLORS',
    'BAND_NAMES',
    'FREQ_BANDS',
    'PALETTES',
    'BIOTUNER_MATRIX_CMAP',
    'BIOTUNER_DIVERGING_CMAP',
    'DEFAULT_STYLE',
    'PLOT_CONFIGS',
    'set_biotuner_style',
    'reset_style',
    'get_color_palette',
    'get_emd_colors',
    'get_band_colors',
    'get_plot_config',
    'update_plot_config',
    'biotuner_style',
]
