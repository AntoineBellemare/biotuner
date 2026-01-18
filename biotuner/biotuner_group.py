"""
BiotunerGroup: Group-level analysis for multiple time series

This module provides a clean, elegant interface for running biotuner analysis
on multiple time series (trials, electrodes, etc.) with automatic aggregation,
comparison, and visualization capabilities.

Author: Biotuner Team
Date: December 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, List, Dict, Optional, Tuple, Callable
from scipy import stats
from collections import defaultdict
import warnings
from joblib import Parallel, delayed

from biotuner.biotuner_object import compute_biotuner
from biotuner.metrics import (
    ratios2harmsim,
    dyad_similarity,
    compute_subharmonic_tension,
    euler,
    tenneyHeight,
)
from biotuner.biotuner_utils import compute_peak_ratios
from biotuner.plot_config import (
    BIOTUNER_COLORS,
    FREQ_BANDS,
    BIOTUNER_MATRIX_CMAP,
    get_plot_config,
    get_color_palette,
)


class BiotunerGroup:
    """
    Group-level biotuner analysis for multiple time series.
    
    This class manages multiple biotuner objects, enabling batch processing,
    aggregation of metrics, group comparisons, and unified visualizations.
    
    Parameters
    ----------
    data : ndarray
        Time series data with shape:
        - 2D: (n_series, n_samples) - e.g., electrodes × timepoints
        - 3D: (n_trials, n_channels, n_samples) - e.g., trials × electrodes × timepoints
    sf : int
        Sampling frequency in Hz
    axis_labels : list of str, optional
        Names for each dimension, e.g., ['trials', 'electrodes']
        If None, uses generic labels ['dim0', 'dim1', ...]
    metadata : dict, optional
        Metadata for grouping/comparison. Keys are column names, values are lists
        with length matching the first dimension of data.
        Example: {'condition': ['rest', 'rest', 'task', 'task'], 'subject': ['S1', 'S2', 'S1', 'S2']}
    store_objects : bool, default=True
        If True, stores all individual biotuner objects. If False, only stores
        results to save memory (individual objects not accessible).
    **biotuner_kwargs : dict
        Default parameters passed to all compute_biotuner objects
        (e.g., peaks_function='EMD', precision=0.1, n_harm=10)
    
    Attributes
    ----------
    objects : list of compute_biotuner
        Individual biotuner objects (if store_objects=True)
    results : pandas.DataFrame
        Summary DataFrame with all computed metrics
    shape : tuple
        Shape of the input data
    n_series : int
        Total number of time series
    
    Examples
    --------
    Basic usage with 2D data:
    
    >>> # Single electrode across trials
    >>> data = np.random.randn(10, 5000)  # 10 trials, 5000 samples
    >>> btg = BiotunerGroup(data, sf=1000)
    >>> btg.compute_peaks(peaks_function='FOOOF', min_freq=1, max_freq=50)
    >>> summary = btg.summary()
    >>> print(summary)
    
    With 3D data and metadata:
    
    >>> # Multiple electrodes across trials
    >>> data = np.random.randn(20, 64, 5000)  # 20 trials, 64 channels, 5000 samples
    >>> metadata = {'condition': ['rest']*10 + ['task']*10}
    >>> btg = BiotunerGroup(data, sf=1000, axis_labels=['trials', 'electrodes'], metadata=metadata)
    >>> btg.compute_peaks(peaks_function='EMD')
    >>> btg.compute_metrics()
    >>> 
    >>> # Compare groups
    >>> comparison = btg.compare_groups('condition', metric='harmsim')
    >>> 
    >>> # Plot
    >>> btg.plot_metric_distribution('harmsim', groupby='condition')
    
    Memory-efficient mode for large datasets:
    
    >>> btg = BiotunerGroup(data, sf=1000, store_objects=False)
    >>> btg.compute_peaks()
    >>> summary = btg.summary()  # Only summary stats kept in memory
    """
    
    def __init__(
        self,
        data: np.ndarray,
        sf: int,
        axis_labels: Optional[List[str]] = None,
        metadata: Optional[Dict[str, List]] = None,
        store_objects: bool = True,
        **biotuner_kwargs
    ):
        # Validate and store data
        self.data = np.asarray(data)
        self.sf = sf
        self.store_objects = store_objects
        self.biotuner_kwargs = biotuner_kwargs
        
        # Determine data shape and structure
        self.shape = self.data.shape
        ndim = len(self.shape)
        
        if ndim < 2:
            raise ValueError("Data must be at least 2D (n_series, n_samples)")
        
        # Handle 2D vs 3D data
        if ndim == 2:
            self.n_series = self.shape[0]
            self.n_samples = self.shape[1]
            self.is_3d = False
            default_labels = ['series']
        elif ndim == 3:
            self.n_series = self.shape[0] * self.shape[1]
            self.n_samples = self.shape[2]
            self.is_3d = True
            default_labels = ['dim0', 'dim1']
        else:
            raise ValueError("Data must be 2D or 3D")
        
        # Set axis labels
        if axis_labels is None:
            self.axis_labels = default_labels
        else:
            if len(axis_labels) != (ndim - 1):
                raise ValueError(f"axis_labels must have {ndim-1} elements for {ndim}D data")
            self.axis_labels = axis_labels
        
        # Initialize metadata tracking
        self._init_metadata(metadata)
        
        # Initialize storage
        self.objects = [] if store_objects else None
        self.results = None
        self._computed_methods = set()
        
    def _init_metadata(self, metadata: Optional[Dict[str, List]]):
        """Initialize metadata DataFrame with indices."""
        if self.is_3d:
            # Create multi-index for 3D data
            idx0, idx1 = np.meshgrid(
                range(self.shape[0]), 
                range(self.shape[1]), 
                indexing='ij'
            )
            self.index_df = pd.DataFrame({
                self.axis_labels[0]: idx0.flatten(),
                self.axis_labels[1]: idx1.flatten(),
                'series_idx': range(self.n_series)
            })
        else:
            # Simple index for 2D data
            self.index_df = pd.DataFrame({
                self.axis_labels[0]: range(self.n_series),
                'series_idx': range(self.n_series)
            })
        
        # Add metadata if provided
        if metadata is not None:
            for key, values in metadata.items():
                if len(values) != self.shape[0]:
                    raise ValueError(f"Metadata '{key}' length {len(values)} doesn't match first data dimension {self.shape[0]}")
                
                # Expand metadata to match flattened structure if 3D
                if self.is_3d:
                    expanded_values = np.repeat(values, self.shape[1])
                    self.index_df[key] = expanded_values
                else:
                    self.index_df[key] = values
    
    def _get_timeseries(self, idx: int) -> np.ndarray:
        """Get a single time series by linear index."""
        if self.is_3d:
            i = idx // self.shape[1]
            j = idx % self.shape[1]
            return self.data[i, j, :]
        else:
            return self.data[idx, :]
    
    def compute_peaks(
        self,
        peaks_function: Optional[str] = None,
        min_freq: float = 1,
        max_freq: float = 60,
        precision: Optional[float] = None,
        n_peaks: int = 5,
        n_jobs: int = 1,
        verbose: bool = False,
        **kwargs
    ) -> 'BiotunerGroup':
        """
        Extract spectral peaks for all time series.
        
        Parameters
        ----------
        peaks_function : str, optional
            Peak extraction method. If None, uses value from biotuner_kwargs or 'EMD'
        min_freq : float, default=1
            Minimum frequency for peak extraction (Hz)
        max_freq : float, default=60
            Maximum frequency for peak extraction (Hz)
        precision : float, optional
            Frequency precision (Hz). If None, uses value from biotuner_kwargs or 0.1
        n_peaks : int, default=5
            Number of peaks to extract
        n_jobs : int, default=1
            Number of parallel jobs. -1 uses all CPUs, 1 for sequential processing.
            Recommended for large datasets (>50 series) or FOOOF peak detection.
        verbose : bool, default=False
            Print progress
        **kwargs : dict
            Additional parameters for peaks_extraction
        
        Returns
        -------
        self : BiotunerGroup
            Returns self for method chaining
        """
        if peaks_function is None:
            peaks_function = self.biotuner_kwargs.get('peaks_function', 'EMD')
        
        if precision is None:
            precision = self.biotuner_kwargs.get('precision', 0.1)
        
        if verbose:
            print(f"Extracting peaks for {self.n_series} time series...")
            if n_jobs != 1:
                print(f"  Using parallel processing with n_jobs={n_jobs}")
        
        # Prepare kwargs
        bt_kwargs = self.biotuner_kwargs.copy()
        bt_kwargs['peaks_function'] = peaks_function
        bt_kwargs['precision'] = precision
        
        # Define function to process a single time series
        def process_single_series(idx):
            ts = self._get_timeseries(idx)
            
            # Create biotuner object
            bt = compute_biotuner(sf=self.sf, **bt_kwargs)
            
            # Extract peaks
            try:
                bt.peaks_extraction(
                    data=ts,
                    min_freq=min_freq,
                    max_freq=max_freq,
                    n_peaks=n_peaks,
                    verbose=False,
                    **kwargs
                )
            except ValueError as e:
                # Handle case where no peaks detected
                if "No peak detected" in str(e):
                    # Create empty peaks/amps
                    bt.peaks = np.array([])
                    bt.amps = np.array([])
                    bt.peaks_ratios = np.array([])
                    bt.peaks_ratios_cons = np.array([])
                else:
                    raise
            
            return bt
        
        # Process all series (parallel or sequential)
        if n_jobs == 1:
            # Sequential processing
            results = []
            for idx in range(self.n_series):
                bt = process_single_series(idx)
                results.append(bt)
                
                if verbose and ((idx + 1) % 10 == 0 or (idx + 1) == self.n_series):
                    print(f"  Processed {idx + 1}/{self.n_series}")
        else:
            # Parallel processing
            results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
                delayed(process_single_series)(idx) for idx in range(self.n_series)
            )
        
        # Store objects if requested
        if self.store_objects:
            self.objects = results
        
        self._computed_methods.add('peaks_extraction')
        
        if verbose:
            print(f"✓ Peaks extracted for all {self.n_series} time series")
        
        return self
    
    def compute_extension(
        self,
        method: str = 'consonant_harmonic_fit',
        n_harm: Optional[int] = None,
        n_jobs: int = 1,
        verbose: bool = False,
        **kwargs
    ) -> 'BiotunerGroup':
        """
        Compute peak extensions for all time series.
        
        Parameters
        ----------
        method : str, default='consonant_harmonic_fit'
            Extension method: 'harmonic_fit', 'consonant', 'multi_consonant', 
            'consonant_harmonic_fit', 'multi_consonant_harmonic_fit'
        n_harm : int, optional
            Number of harmonics. If None, uses value from biotuner_kwargs or 10
        n_jobs : int, default=1
            Number of parallel jobs. -1 uses all CPUs, 1 for sequential processing.
        verbose : bool, default=False
            Print progress
        **kwargs : dict
            Additional parameters for peaks_extension
        
        Returns
        -------
        self : BiotunerGroup
            Returns self for method chaining
        """
        if not self.store_objects:
            raise RuntimeError("Peak extension requires store_objects=True")
        
        if 'peaks_extraction' not in self._computed_methods:
            raise RuntimeError("Must run compute_peaks() before compute_extension()")
        
        if n_harm is None:
            n_harm = self.biotuner_kwargs.get('n_harm', 10)
        
        if verbose:
            print(f"Computing peak extensions for {self.n_series} time series...")
            if n_jobs != 1:
                print(f"  Using parallel processing with n_jobs={n_jobs}")
        
        # Prepare extension kwargs
        ext_kwargs = kwargs.copy()
        ext_kwargs['method'] = method
        ext_kwargs['n_harm'] = n_harm
        
        # Define function to compute extension for a single object
        def compute_single_extension(bt):
            bt.peaks_extension(**ext_kwargs)
            return bt
        
        # Process all objects (parallel or sequential)
        if n_jobs == 1:
            # Sequential processing
            for idx, bt in enumerate(self.objects):
                compute_single_extension(bt)
                
                if verbose and ((idx + 1) % 10 == 0 or (idx + 1) == self.n_series):
                    print(f"  Processed {idx + 1}/{self.n_series}")
        else:
            # Parallel processing
            self.objects = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
                delayed(compute_single_extension)(bt) for bt in self.objects
            )
        
        self._computed_methods.add('peaks_extension')
        
        if verbose:
            print(f"✓ Peak extensions computed for all {self.n_series} time series")
        
        return self
    
    def compute_metrics(
        self,
        n_harm: Optional[int] = None,
        delta_lim: int = 20,
        n_jobs: int = 1,
        verbose: bool = False,
        **kwargs
    ) -> 'BiotunerGroup':
        """
        Compute peak metrics for all time series.
        
        Parameters
        ----------
        n_harm : int, optional
            Number of harmonics. If None, uses value from biotuner_kwargs or 10
        delta_lim : int, default=20
            Delta limit for subharmonic tension calculation
        n_jobs : int, default=1
            Number of parallel jobs. -1 uses all CPUs, 1 for sequential processing.
        verbose : bool, default=False
            Print progress
        **kwargs : dict
            Additional parameters for compute_peaks_metrics
        
        Returns
        -------
        self : BiotunerGroup
            Returns self for method chaining
        """
        if not self.store_objects:
            raise RuntimeError("Metric computation requires store_objects=True")
        
        if 'peaks_extraction' not in self._computed_methods:
            raise RuntimeError("Must run compute_peaks() before compute_metrics()")
        
        if n_harm is None:
            n_harm = self.biotuner_kwargs.get('n_harm', 10)
        
        if verbose:
            print(f"Computing metrics for {self.n_series} time series...")
            if n_jobs != 1:
                print(f"  Using parallel processing with n_jobs={n_jobs}")
        
        # Prepare metrics kwargs
        metrics_kwargs = kwargs.copy()
        metrics_kwargs['n_harm'] = n_harm
        metrics_kwargs['delta_lim'] = delta_lim
        
        # Define function to compute metrics for a single object
        def compute_single_metrics(bt):
            bt.compute_peaks_metrics(**metrics_kwargs)
            return bt
        
        # Process all objects (parallel or sequential)
        if n_jobs == 1:
            # Sequential processing
            for idx, bt in enumerate(self.objects):
                compute_single_metrics(bt)
                
                if verbose and ((idx + 1) % 10 == 0 or (idx + 1) == self.n_series):
                    print(f"  Processed {idx + 1}/{self.n_series}")
        else:
            # Parallel processing
            self.objects = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
                delayed(compute_single_metrics)(bt) for bt in self.objects
            )
        
        self._computed_methods.add('compute_peaks_metrics')
        
        if verbose:
            print(f"✓ Metrics computed for all {self.n_series} time series")
        
        return self
    
    def compute_diss_curve(
        self,
        input_type: str = 'peaks',
        denom: int = 1000,
        max_ratio: float = 2,
        n_jobs: int = 1,
        verbose: bool = False,
        **kwargs
    ) -> 'BiotunerGroup':
        """
        Compute dissonance curves for all time series.
        
        Parameters
        ----------
        input_type : str, default='peaks'
            Input type: 'peaks' or 'extended_peaks'
        denom : int, default=1000
            Denominator for dissonance curve resolution
        max_ratio : float, default=2
            Maximum ratio (typically octave = 2)
        n_jobs : int, default=1
            Number of parallel jobs. -1 uses all CPUs, 1 for sequential processing.
        verbose : bool, default=False
            Print progress
        **kwargs : dict
            Additional parameters for compute_diss_curve
        
        Returns
        -------
        self : BiotunerGroup
            Returns self for method chaining
        """
        if not self.store_objects:
            raise RuntimeError("Dissonance curve computation requires store_objects=True")
        
        if 'peaks_extraction' not in self._computed_methods:
            raise RuntimeError("Must run compute_peaks() before compute_diss_curve()")
        
        if verbose:
            print(f"Computing dissonance curves for {self.n_series} time series...")
            if n_jobs != 1:
                print(f"  Using parallel processing with n_jobs={n_jobs}")
        
        # Prepare diss curve kwargs
        diss_kwargs = kwargs.copy()
        diss_kwargs['input_type'] = input_type
        diss_kwargs['denom'] = denom
        diss_kwargs['max_ratio'] = max_ratio
        diss_kwargs['plot'] = False
        
        # Define function to compute diss curve for a single object
        def compute_single_diss(bt):
            bt.compute_diss_curve(**diss_kwargs)
            return bt
        
        # Process all objects (parallel or sequential)
        if n_jobs == 1:
            # Sequential processing
            for idx, bt in enumerate(self.objects):
                compute_single_diss(bt)
                
                if verbose and ((idx + 1) % 10 == 0 or (idx + 1) == self.n_series):
                    print(f"  Processed {idx + 1}/{self.n_series}")
        else:
            # Parallel processing
            self.objects = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
                delayed(compute_single_diss)(bt) for bt in self.objects
            )
        
        self._computed_methods.add('compute_diss_curve')
        
        if verbose:
            print(f"✓ Dissonance curves computed for all {self.n_series} time series")
        
        return self
    
    def compute_harmonic_entropy(
        self,
        input_type: str = 'peaks',
        res: float = 0.001,
        spread: float = 0.01,
        octave: float = 2,
        verbose: bool = False,
        **kwargs
    ) -> 'BiotunerGroup':
        """
        Compute harmonic entropy for all time series.
        
        Parameters
        ----------
        input_type : str, default='peaks'
            Input type: 'peaks' or 'extended_peaks'
        res : float, default=0.001
            Resolution of ratio steps
        spread : float, default=0.01
            Spread of normal distribution
        octave : float, default=2
            Octave value
        verbose : bool, default=False
            Print progress
        **kwargs : dict
            Additional parameters for compute_harmonic_entropy
        
        Returns
        -------
        self : BiotunerGroup
            Returns self for method chaining
        """
        if not self.store_objects:
            raise RuntimeError("Harmonic entropy computation requires store_objects=True")
        
        if 'peaks_extraction' not in self._computed_methods:
            raise RuntimeError("Must run compute_peaks() before compute_harmonic_entropy()")
        
        if verbose:
            print(f"Computing harmonic entropy for {self.n_series} time series...")
        
        # Prepare HE kwargs
        he_kwargs = kwargs.copy()
        he_kwargs['input_type'] = input_type
        he_kwargs['res'] = res
        he_kwargs['spread'] = spread
        he_kwargs['octave'] = octave
        he_kwargs['plot_entropy'] = False
        he_kwargs['plot_tenney'] = False
        
        for idx, bt in enumerate(self.objects):
            bt.compute_harmonic_entropy(**he_kwargs)
            
            if verbose:
                if (idx + 1) % 10 == 0 or (idx + 1) == self.n_series:
                    print(f"  Processed {idx + 1}/{self.n_series}")
        
        self._computed_methods.add('compute_harmonic_entropy')
        
        if verbose:
            print(f"✓ Harmonic entropy computed for all {self.n_series} time series")
        
        return self
    
    def compute_euler_fokker(
        self,
        method: str = 'peaks',
        octave: float = 2,
        verbose: bool = False
    ) -> 'BiotunerGroup':
        """
        Compute Euler-Fokker scales for all time series.
        
        Parameters
        ----------
        method : str, default='peaks'
            Method: 'peaks' or 'extended_peaks'
        octave : float, default=2
            Period interval
        verbose : bool, default=False
            Print progress
        
        Returns
        -------
        self : BiotunerGroup
            Returns self for method chaining
        """
        if not self.store_objects:
            raise RuntimeError("Euler-Fokker computation requires store_objects=True")
        
        if 'peaks_extraction' not in self._computed_methods:
            raise RuntimeError("Must run compute_peaks() before compute_euler_fokker()")
        
        if verbose:
            print(f"Computing Euler-Fokker scales for {self.n_series} time series...")
        
        for idx, bt in enumerate(self.objects):
            bt.euler_fokker_scale(method=method, octave=octave)
            
            if verbose:
                if (idx + 1) % 10 == 0 or (idx + 1) == self.n_series:
                    print(f"  Processed {idx + 1}/{self.n_series}")
        
        self._computed_methods.add('euler_fokker_scale')
        
        if verbose:
            print(f"✓ Euler-Fokker scales computed for all {self.n_series} time series")
        
        return self
    
    def compute_harmonic_tuning(
        self,
        list_harmonics: Optional[List[int]] = None,
        octave: float = 2,
        min_ratio: float = 1,
        max_ratio: float = 2,
        verbose: bool = False
    ) -> 'BiotunerGroup':
        """
        Compute harmonic tunings for all time series.
        
        Parameters
        ----------
        list_harmonics : list of int, optional
            Harmonic positions for scale construction
        octave : float, default=2
            Period reference
        min_ratio : float, default=1
            Unison value
        max_ratio : float, default=2
            Octave value
        verbose : bool, default=False
            Print progress
        
        Returns
        -------
        self : BiotunerGroup
            Returns self for method chaining
        """
        if not self.store_objects:
            raise RuntimeError("Harmonic tuning computation requires store_objects=True")
        
        if 'peaks_extraction' not in self._computed_methods:
            raise RuntimeError("Must run compute_peaks() before compute_harmonic_tuning()")
        
        if verbose:
            print(f"Computing harmonic tunings for {self.n_series} time series...")
        
        for idx, bt in enumerate(self.objects):
            bt.harmonic_tuning(
                list_harmonics=list_harmonics,
                octave=octave,
                min_ratio=min_ratio,
                max_ratio=max_ratio
            )
            
            if verbose:
                if (idx + 1) % 10 == 0 or (idx + 1) == self.n_series:
                    print(f"  Processed {idx + 1}/{self.n_series}")
        
        self._computed_methods.add('harmonic_tuning')
        
        if verbose:
            print(f"✓ Harmonic tunings computed for all {self.n_series} time series")
        
        return self
    
    def get_attribute(
        self,
        attr_name: str,
        as_array: bool = False,
        missing_value: any = np.nan
    ) -> Union[List, np.ndarray]:
        """
        Get an attribute from all biotuner objects.
        
        Parameters
        ----------
        attr_name : str
            Attribute name (e.g., 'peaks', 'peaks_ratios', 'peaks_metrics')
        as_array : bool, default=False
            If True, attempt to stack results as numpy array.
            Only works if all results have the same shape.
        missing_value : any, default=np.nan
            Value to use if attribute doesn't exist for an object
        
        Returns
        -------
        results : list or ndarray
            List or array of attribute values from all objects
        
        Examples
        --------
        >>> peaks_all = btg.get_attribute('peaks')
        >>> amps_all = btg.get_attribute('amps')
        >>> ratios_all = btg.get_attribute('peaks_ratios')
        """
        if not self.store_objects:
            raise RuntimeError("Cannot access attributes when store_objects=False")
        
        if len(self.objects) == 0:
            raise RuntimeError("No biotuner objects available. Run compute_peaks() first.")
        
        results = []
        for bt in self.objects:
            if hasattr(bt, attr_name):
                results.append(getattr(bt, attr_name))
            else:
                results.append(missing_value)
        
        if as_array:
            try:
                return np.array(results)
            except ValueError:
                warnings.warn(f"Could not convert {attr_name} to array (inconsistent shapes). Returning list.")
                return results
        
        return results
    
    def summary(
        self,
        metrics: Optional[List[str]] = None,
        aggregation: Union[str, List[str]] = 'all',
        include_index: bool = True
    ) -> pd.DataFrame:
        """
        Generate summary DataFrame with aggregated metrics.
        
        Parameters
        ----------
        metrics : list of str, optional
            Which metrics to include. If None, includes all available metrics.
            Examples: ['harmsim', 'cons', 'tenney', 'euler', 'n_peaks']
        aggregation : str or list of str, default='all'
            Aggregation methods to apply:
            - 'mean', 'std', 'median', 'min', 'max', 'sem'
            - 'all': computes all of the above
            - list: specific subset, e.g., ['mean', 'std']
        include_index : bool, default=True
            Include index columns and metadata in output
        
        Returns
        -------
        df : pandas.DataFrame
            Summary DataFrame with one row per time series
        
        Examples
        --------
        >>> # Get all statistics
        >>> summary = btg.summary()
        >>> 
        >>> # Get specific metrics and stats
        >>> summary = btg.summary(metrics=['harmsim', 'cons'], aggregation=['mean', 'std'])
        >>> 
        >>> # Compare across conditions
        >>> summary = btg.summary()
        >>> summary.groupby('condition').mean()
        """
        if not self.store_objects:
            raise RuntimeError("Summary requires store_objects=True")
        
        if len(self.objects) == 0:
            raise RuntimeError("No biotuner objects available. Run compute_peaks() first.")
        
        # Build aggregation functions
        if aggregation == 'all':
            agg_funcs = {
                'mean': np.nanmean,
                'std': np.nanstd,
                'median': np.nanmedian,
                'min': np.nanmin,
                'max': np.nanmax,
                'sem': lambda x: np.nanstd(x) / np.sqrt(np.sum(~np.isnan(x)))
            }
        else:
            if isinstance(aggregation, str):
                aggregation = [aggregation]
            
            agg_map = {
                'mean': np.nanmean,
                'std': np.nanstd,
                'median': np.nanmedian,
                'min': np.nanmin,
                'max': np.nanmax,
                'sem': lambda x: np.nanstd(x) / np.sqrt(np.sum(~np.isnan(x)))
            }
            agg_funcs = {k: agg_map[k] for k in aggregation if k in agg_map}
        
        # Collect data
        rows = []
        
        for idx, bt in enumerate(self.objects):
            row = {}
            
            # Basic peak info
            if hasattr(bt, 'peaks'):
                peaks = bt.peaks
                amps = bt.amps if hasattr(bt, 'amps') else None
                
                row['n_peaks'] = len(peaks)
                
                if len(peaks) > 0:
                    for agg_name, agg_func in agg_funcs.items():
                        row[f'peak_freq_{agg_name}'] = agg_func(peaks)
                        if amps is not None:
                            row[f'peak_amp_{agg_name}'] = agg_func(amps)
            
            # Peaks metrics
            if hasattr(bt, 'peaks_metrics'):
                pm = bt.peaks_metrics
                
                # Select metrics to include
                if metrics is None:
                    metric_keys = [k for k in pm.keys() if not k.startswith('_')]
                else:
                    metric_keys = [k for k in metrics if k in pm]
                
                # Metrics that should be aggregated (return arrays with multiple values per peak)
                array_metrics = ['harm_pos', 'common_harm_pos']
                
                for key in metric_keys:
                    value = pm[key]
                    
                    # Handle scalar values (most metrics: cons, euler, tenney, harmsim, etc.)
                    if np.isscalar(value):
                        row[key] = value
                    # Handle array values
                    elif isinstance(value, (list, np.ndarray)):
                        if len(value) == 0:
                            continue
                        # Check if this metric should be aggregated
                        elif key in array_metrics and len(value) > 1:
                            # Multiple values - apply aggregation functions
                            for agg_name, agg_func in agg_funcs.items():
                                row[f'{key}_{agg_name}'] = agg_func(value)
                        else:
                            # Single value or metric that returns list but shouldn't be aggregated
                            # (e.g., subharm_tension returns [value])
                            row[key] = value[0] if len(value) == 1 else value
                    # Handle dict values
                    elif isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            if np.isscalar(subvalue):
                                row[f'{key}_{subkey}'] = subvalue
            
            # Scale metrics (from dissonance curve, etc.)
            if hasattr(bt, 'scale_metrics') and bt.scale_metrics:
                for key, value in bt.scale_metrics.items():
                    # Convert to scalar if needed
                    if np.isscalar(value):
                        # Check if it's a valid number
                        if pd.notna(value) and not isinstance(value, str):
                            row[f'scale_{key}'] = value
                    elif isinstance(value, (list, np.ndarray)):
                        # Convert to numpy array for easier handling
                        arr = np.asarray(value)
                        # Skip if all NaN or empty
                        if len(arr) == 0 or np.all(np.isnan(arr)):
                            continue
                        # Take first element or mean
                        if len(arr) == 1:
                            row[f'scale_{key}'] = float(arr[0])
                        else:
                            # For arrays with multiple values, store mean
                            row[f'scale_{key}'] = float(np.nanmean(arr))
                    else:
                        # Try to convert to float
                        try:
                            val = float(value)
                            if pd.notna(val):
                                row[f'scale_{key}'] = val
                        except (TypeError, ValueError):
                            # Skip non-numeric values
                            continue
            
            rows.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Add index columns
        if include_index:
            df = pd.concat([self.index_df, df], axis=1)
        
        self.results = df
        return df
    
    def tuning_summary(
        self,
        include_index: bool = True
    ) -> pd.DataFrame:
        """
        Generate summary DataFrame focused on tuning/scale metrics.
        
        This is a convenience method that extracts only scale-related metrics
        from the full summary, making it easier to analyze tuning characteristics
        across time series.
        
        Parameters
        ----------
        include_index : bool, default=True
            Include index columns and metadata in output
        
        Returns
        -------
        df : pandas.DataFrame
            Summary DataFrame with tuning metrics (scale_* columns)
        
        Examples
        --------
        >>> # Compute tunings
        >>> btg.compute_diss_curve().compute_harmonic_entropy()
        >>> 
        >>> # Get tuning metrics summary
        >>> tuning_df = btg.tuning_summary()
        >>> 
        >>> # Compare tuning metrics across conditions
        >>> tuning_df.groupby('condition').mean()
        """
        # Get full summary first
        full_summary = self.summary(include_index=include_index)
        
        # Extract scale columns
        scale_cols = [c for c in full_summary.columns if c.startswith('scale_')]
        
        if len(scale_cols) == 0:
            print("No tuning metrics found. Run compute_diss_curve() or compute_harmonic_entropy() first.")
            return pd.DataFrame()
        
        # Select index columns + scale columns
        if include_index:
            index_cols = list(self.index_df.columns)
            selected_cols = index_cols + scale_cols
        else:
            selected_cols = scale_cols
        
        return full_summary[selected_cols]
    
    def get_tuning_scales(
        self,
        scale_type: str = 'diss_scale'
    ) -> List:
        """
        Extract tuning scales from all biotuner objects.
        
        Parameters
        ----------
        scale_type : str, default='diss_scale'
            Type of scale to extract:
            - 'diss_scale': Dissonance curve scale
            - 'HE_scale': Harmonic entropy scale
            - 'euler_fokker': Euler-Fokker scale
            - 'harm_tuning_scale': Harmonic tuning scale
        
        Returns
        -------
        scales : list
            List of scales (each scale is a list of ratios)
        
        Examples
        --------
        >>> btg.compute_diss_curve()
        >>> scales = btg.get_tuning_scales('diss_scale')
        >>> 
        >>> # Analyze scale sizes
        >>> scale_sizes = [len(s) for s in scales]
        >>> print(f"Mean scale size: {np.mean(scale_sizes):.1f} steps")
        """
        if not self.store_objects:
            raise RuntimeError("Cannot access scales when store_objects=False")
        
        scales = []
        for bt in self.objects:
            if hasattr(bt, scale_type):
                scales.append(getattr(bt, scale_type))
            else:
                scales.append([])
        
        return scales
    
    def compare_groups(
        self,
        groupby: str,
        metric: str = 'harmsim',
        test: str = 'ttest',
        alpha: float = 0.05,
        plot: bool = True
    ) -> pd.DataFrame:
        """
        Compare metrics across metadata groups.
        
        Parameters
        ----------
        groupby : str
            Metadata column to group by
        metric : str, default='harmsim'
            Metric to compare
        test : str, default='ttest'
            Statistical test: 'ttest', 'anova', 'mannwhitneyu', 'kruskal'
        alpha : float, default=0.05
            Significance level
        plot : bool, default=True
            Generate comparison plot
        
        Returns
        -------
        results : pandas.DataFrame
            Comparison results with statistics
        
        Examples
        --------
        >>> comparison = btg.compare_groups('condition', metric='harmsim_mean')
        >>> print(comparison)
        """
        if self.results is None:
            self.summary()
        
        if groupby not in self.results.columns:
            raise ValueError(f"Column '{groupby}' not found in results. Available: {list(self.results.columns)}")
        
        if metric not in self.results.columns:
            raise ValueError(f"Metric '{metric}' not found in results. Available: {list(self.results.columns)}")
        
        # Get groups
        groups = self.results.groupby(groupby)[metric].apply(list)
        group_names = list(groups.index)
        group_values = list(groups.values)
        
        # Run statistical test
        if len(group_names) == 2:
            if test == 'ttest':
                stat, pval = stats.ttest_ind(group_values[0], group_values[1], nan_policy='omit')
                test_name = "Independent t-test"
            elif test == 'mannwhitneyu':
                stat, pval = stats.mannwhitneyu(group_values[0], group_values[1])
                test_name = "Mann-Whitney U test"
            else:
                raise ValueError(f"Test '{test}' not valid for 2 groups. Use 'ttest' or 'mannwhitneyu'")
        else:
            if test in ['ttest', 'anova']:
                stat, pval = stats.f_oneway(*group_values)
                test_name = "One-way ANOVA"
            elif test in ['mannwhitneyu', 'kruskal']:
                stat, pval = stats.kruskal(*group_values)
                test_name = "Kruskal-Wallis test"
            else:
                raise ValueError(f"Unknown test: {test}")
        
        # Build results table
        comparison_data = []
        for name, values in zip(group_names, group_values):
            comparison_data.append({
                'group': name,
                'n': len([v for v in values if not np.isnan(v)]),
                'mean': np.nanmean(values),
                'std': np.nanstd(values),
                'median': np.nanmedian(values),
                'sem': np.nanstd(values) / np.sqrt(np.sum(~np.isnan(values)))
            })
        
        result_df = pd.DataFrame(comparison_data)
        
        # Add statistical test results as metadata (not in rows)
        result_df.attrs['test_name'] = test_name
        result_df.attrs['statistic'] = stat
        result_df.attrs['p-value'] = pval
        result_df.attrs['alpha'] = alpha
        result_df.attrs['significant'] = pval < alpha
        
        # Print test results
        print(f"\n{test_name}")
        print(f"Metric: {metric}")
        print(f"Grouping: {groupby}")
        print(f"Statistic: {stat:.4f}")
        print(f"p-value: {pval:.4f}")
        print(f"Significant: {'Yes' if pval < alpha else 'No'} (α = {alpha})")
        print(f"\n{result_df.to_string()}\n")
        
        # Plot if requested
        if plot:
            config = get_plot_config('group_comparison')
            palette = get_color_palette(config['palette'], n_colors=len(group_names))
            
            fig, ax = plt.subplots(figsize=config['figsize'])
            
            plot_data = self.results[[groupby, metric]].copy()
            
            # Modern styled boxplot
            sns.boxplot(data=plot_data, x=groupby, y=metric, ax=ax, palette=palette,
                       linewidth=2, fliersize=0)  # No outlier markers
            
            # Add semi-transparent strip plot
            sns.stripplot(data=plot_data, x=groupby, y=metric, ax=ax, 
                         color=config['point_color'], alpha=config['point_alpha'], 
                         size=config['point_size'], jitter=0.2)
            
            # Styling
            significance = '***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < alpha else 'ns'
            ax.set_title(f'{config["title_prefix"]}{metric}\n{test_name}: p = {pval:.4f} ({significance})',
                        fontsize=16, fontweight='bold', pad=15)
            ax.set_ylabel(metric, fontsize=13, fontweight='medium')
            ax.set_xlabel(groupby, fontsize=13, fontweight='medium')
            ax.grid(alpha=0.25, linewidth=0.8, axis='y')
            
            plt.tight_layout()
            plt.show()
        
        return result_df
    
    def plot_group_peaks(
        self,
        show_individual: bool = False,
        aggregate: str = 'mean',
        xmin: float = 1,
        xmax: float = 60,
        show_bands: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
        alpha_individual: Optional[float] = None,
        **kwargs
    ):
        """
        Plot aggregated peak spectrum across all time series.
        
        Parameters
        ----------
        show_individual : bool, default=False
            Show individual PSDs in background
        aggregate : str, default='mean'
            Aggregation method: 'mean', 'median'
        xmin : float, default=1
            Minimum frequency (Hz)
        xmax : float, default=60
            Maximum frequency (Hz)
        show_bands : bool, default=True
            Show frequency band labels
        figsize : tuple, optional
            Figure size (default from plot_config)
        alpha_individual : float, optional
            Transparency for individual traces (default from plot_config)
        **kwargs : dict
            Additional plotting parameters
        
        Returns
        -------
        fig, ax : matplotlib figure and axes
        """
        if not self.store_objects:
            raise RuntimeError("Plotting requires store_objects=True")
        
        # Get plot configuration
        config = get_plot_config('group_peaks')
        if figsize is None:
            figsize = config['figsize']
        if alpha_individual is None:
            alpha_individual = config['individual_alpha']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Collect all PSDs
        all_freqs = []
        all_psds = []
        all_peaks = []
        all_amps = []
        
        for bt in self.objects:
            if hasattr(bt, 'freqs') and hasattr(bt, 'psd'):
                all_freqs.append(bt.freqs)
                all_psds.append(bt.psd)
                
                if hasattr(bt, 'peaks'):
                    all_peaks.append(bt.peaks)
                    all_amps.append(bt.amps if hasattr(bt, 'amps') else np.ones_like(bt.peaks))
        
        if len(all_psds) == 0:
            raise RuntimeError("No PSD data available. Ensure peaks were extracted with proper settings.")
        
        # Find common frequency axis
        common_freqs = all_freqs[0]
        
        # Plot individual PSDs if requested
        if show_individual:
            for freqs, psd in zip(all_freqs, all_psds):
                mask = (freqs >= xmin) & (freqs <= xmax)
                ax.plot(freqs[mask], psd[mask], color=BIOTUNER_COLORS['light'], 
                       alpha=alpha_individual, linewidth=1.2, zorder=1)
        
        # Compute and plot aggregate
        if aggregate == 'mean':
            agg_psd = np.nanmean(all_psds, axis=0)
        elif aggregate == 'median':
            agg_psd = np.nanmedian(all_psds, axis=0)
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate}")
        
        mask = (common_freqs >= xmin) & (common_freqs <= xmax)
        ax.plot(common_freqs[mask], agg_psd[mask], color=config['aggregate_color'], 
                linewidth=3, label=f'{aggregate.capitalize()} PSD', zorder=3)
        
        # Mark aggregate peak positions
        if len(all_peaks) > 0:
            # Find most common peaks
            all_peaks_flat = np.concatenate(all_peaks)
            if len(all_peaks_flat) > 0:
                # Create histogram of peak positions
                peak_hist, peak_bins = np.histogram(all_peaks_flat, bins=int((xmax-xmin)/0.5))
                # Find significant peaks
                threshold = len(self.objects) * 0.2  # Peak in at least 20% of series
                sig_bins = peak_bins[:-1][peak_hist >= threshold]
                
                for peak_freq in sig_bins:
                    # Find PSD value at this frequency
                    freq_idx = np.argmin(np.abs(common_freqs - peak_freq))
                    ax.plot(peak_freq, agg_psd[freq_idx], 'o', color=config['peak_marker_color'], 
                           markersize=config['peak_marker_size'], markeredgecolor='white', 
                           markeredgewidth=2, zorder=4)
        
        # Add frequency bands if requested
        if show_bands:
            # Use gradient colors from light to dark
            band_colors = ['#E8F4F8', '#B8E6F0', '#88D8E8', '#58CAE0', '#28BCD8']
            ylim = ax.get_ylim()
            
            for idx, (band_name, (low, high)) in enumerate(FREQ_BANDS.items()):
                if low >= xmin and low <= xmax:
                    ax.axvspan(low, min(high, xmax), alpha=0.12, color=band_colors[idx % len(band_colors)], 
                             label=band_name.capitalize(), zorder=0)
        
        ax.set_xlabel(config['xlabel'], fontsize=13, fontweight='medium')
        ax.set_ylabel(config['ylabel'], fontsize=13, fontweight='medium')
        ax.set_xlim(xmin, xmax)
        ax.set_title(f'{config["title_prefix"]}n={self.n_series}', fontsize=16, fontweight='bold', pad=15)
        ax.legend(loc='upper right', framealpha=0.95, fontsize=11)
        ax.grid(alpha=0.25, linewidth=0.8)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_peak_distribution(
        self,
        xmin: float = 1,
        xmax: float = 60,
        bins: Optional[int] = None,
        show_bands: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
        **kwargs
    ):
        """
        Plot distribution of all detected peaks across all time series.
        
        Parameters
        ----------
        xmin : float, default=1
            Minimum frequency (Hz)
        xmax : float, default=60
            Maximum frequency (Hz)
        bins : int, optional
            Number of histogram bins (default from plot_config)
        show_bands : bool, default=True
            Show frequency band markers
        figsize : tuple, optional
            Figure size (default from plot_config)
        **kwargs : dict
            Additional plotting parameters
        
        Returns
        -------
        fig, ax : matplotlib figure and axes
        """
        if not self.store_objects:
            raise RuntimeError("Plotting requires store_objects=True")
        
        # Get plot configuration
        config = get_plot_config('group_peak_distribution')
        if figsize is None:
            figsize = config['figsize']
        if bins is None:
            bins = config['bins']
        
        # Extract all peaks
        all_peaks = self.get_attribute('peaks')
        all_peaks_flat = np.concatenate([p for p in all_peaks if len(p) > 0])
        
        if len(all_peaks_flat) == 0:
            print("No peaks detected.")
            return None, None
        
        # Filter to frequency range
        all_peaks_flat = all_peaks_flat[(all_peaks_flat >= xmin) & (all_peaks_flat <= xmax)]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Add frequency bands as shaded regions if requested (before histogram so they're in background)
        if show_bands:
            # Use gradient colors from light to dark (same as plot_group_peaks)
            band_colors = ['#E8F4F8', '#B8E6F0', '#88D8E8', '#58CAE0', '#28BCD8']
            
            for idx, (band_name, (low, high)) in enumerate(FREQ_BANDS.items()):
                if low >= xmin and low <= xmax:
                    ax.axvspan(low, min(high, xmax), alpha=0.12, color=band_colors[idx % len(band_colors)], 
                             label=band_name.capitalize(), zorder=0)
        
        # Plot histogram on top of band regions
        ax.hist(all_peaks_flat, bins=bins, edgecolor=config['edgecolor'], 
               linewidth=1.5, alpha=config['alpha'], color=config['color'], zorder=2)
        
        ax.set_xlabel(config['xlabel'], fontsize=13, fontweight='medium')
        ax.set_ylabel(config['ylabel'], fontsize=13, fontweight='medium')
        ax.set_title(config['title'], fontsize=16, fontweight='bold', pad=15)
        ax.set_xlim(xmin, xmax)
        ax.grid(alpha=0.25, linewidth=0.8, axis='y')
        
        # Add legend if bands are shown
        if show_bands:
            ax.legend(loc='upper right', framealpha=0.95, fontsize=11)
        
        # Add statistics text
        stats_text = f'Total peaks: {len(all_peaks_flat)}\nMean: {np.mean(all_peaks_flat):.2f} Hz'
        ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                        edgecolor=BIOTUNER_COLORS['primary'], alpha=0.9, linewidth=2),
               fontsize=11, fontweight='medium')
        
        plt.tight_layout()
        return fig, ax
    
    def plot_metric_distribution(
        self,
        metric: str,
        groupby: Optional[str] = None,
        kind: str = 'violin',
        figsize: Optional[Tuple[float, float]] = None,
        **kwargs
    ):
        """
        Plot distribution of a metric across time series.
        
        Parameters
        ----------
        metric : str
            Metric to plot (column name from summary DataFrame)
        groupby : str, optional
            Metadata column to group by
        kind : str, default='violin'
            Plot type: 'violin', 'box', 'strip', 'swarm', 'bar'
        figsize : tuple, optional
            Figure size (default from plot_config)
        **kwargs : dict
            Additional plotting parameters passed to seaborn
        
        Returns
        -------
        fig, ax : matplotlib figure and axes
        """
        if self.results is None:
            self.summary()
        
        if metric not in self.results.columns:
            raise ValueError(f"Metric '{metric}' not found. Available: {list(self.results.columns)}")
        
        # Get plot configuration
        config = get_plot_config('group_metric_dist')
        if figsize is None:
            figsize = config['figsize']
        
        # Get color palette
        palette = get_color_palette(config['palette'], n_colors=len(self.results[groupby].unique()) if groupby else 1)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if groupby is not None:
            if groupby not in self.results.columns:
                raise ValueError(f"Column '{groupby}' not found")
            
            if kind == 'violin':
                sns.violinplot(data=self.results, x=groupby, y=metric, ax=ax, palette=palette, **kwargs)
            elif kind == 'box':
                sns.boxplot(data=self.results, x=groupby, y=metric, ax=ax, palette=palette, **kwargs)
            elif kind == 'strip':
                sns.stripplot(data=self.results, x=groupby, y=metric, ax=ax, palette=palette, **kwargs)
            elif kind == 'swarm':
                sns.swarmplot(data=self.results, x=groupby, y=metric, ax=ax, palette=palette, **kwargs)
            elif kind == 'bar':
                sns.barplot(data=self.results, x=groupby, y=metric, ax=ax, palette=palette, **kwargs)
            
            ax.set_title(f'{config["title_prefix"]}{metric} by {groupby}', fontsize=16, fontweight='bold', pad=15)
            ax.set_xlabel(groupby, fontsize=13, fontweight='medium')
            ax.set_ylabel(metric, fontsize=13, fontweight='medium')
        else:
            if kind in ['violin', 'box']:
                if kind == 'violin':
                    sns.violinplot(data=self.results[[metric]], ax=ax, color=palette[0], **kwargs)
                else:
                    sns.boxplot(data=self.results[[metric]], ax=ax, color=palette[0], **kwargs)
            else:
                ax.hist(self.results[metric].dropna(), bins=30, edgecolor='black', 
                       color=palette[0], alpha=0.8, **kwargs)
                ax.set_xlabel(metric, fontsize=13, fontweight='medium')
                ax.set_ylabel('Count', fontsize=13, fontweight='medium')
            
            ax.set_title(f'{config["title_prefix"]}{metric} (n={self.n_series})', 
                        fontsize=16, fontweight='bold', pad=15)
        
        ax.grid(alpha=0.25, linewidth=0.8, axis='y')
        
        plt.tight_layout()
        return fig, ax
    
    def plot_metric_matrix(
        self,
        metric: str = 'harmsim_mean',
        cmap: Optional[str] = None,
        figsize: Optional[Tuple[float, float]] = None,
        **kwargs
    ):
        """
        Plot metric as heatmap (for 3D data: trials × electrodes).
        
        Parameters
        ----------
        metric : str, default='harmsim_mean'
            Metric to plot
        cmap : str, optional
            Colormap (default from plot_config)
        figsize : tuple, optional
            Figure size (calculated dynamically if not provided)
        **kwargs : dict
            Additional parameters for sns.heatmap
        
        Returns
        -------
        fig, ax : matplotlib figure and axes
        """
        if not self.is_3d:
            raise RuntimeError("Matrix plot only available for 3D data (trials × electrodes × timepoints)")
        
        if self.results is None:
            self.summary()
        
        if metric not in self.results.columns:
            raise ValueError(f"Metric '{metric}' not found")
        
        # Get plot configuration
        config = get_plot_config('group_metric_matrix')
        if cmap is None:
            # Use custom biotuner matrix colormap
            cmap = BIOTUNER_MATRIX_CMAP
        
        # Reshape to matrix
        matrix = self.results.pivot(
            index=self.axis_labels[0],
            columns=self.axis_labels[1],
            values=metric
        )
        
        if figsize is None:
            figsize = (max(10, matrix.shape[1] * 0.4), max(7, matrix.shape[0] * 0.35))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Merge cbar_kws if provided in kwargs
        default_cbar_kws = {'label': metric, 'shrink': 0.8}
        if 'cbar_kws' in kwargs:
            default_cbar_kws.update(kwargs.pop('cbar_kws'))
        
        sns.heatmap(matrix, cmap=cmap, ax=ax, cbar_kws=default_cbar_kws, 
                   linewidths=config.get('linewidths', 0.8), 
                   linecolor=config.get('linecolor', 'white'), **kwargs)
        
        ax.set_title(f'{config["title_prefix"]}{metric}', fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel(self.axis_labels[1], fontsize=13, fontweight='medium')
        ax.set_ylabel(self.axis_labels[0], fontsize=13, fontweight='medium')
        
        plt.tight_layout()
        return fig, ax
    
    def plot_interval_histogram(
        self,
        scale_type: str = 'diss_scale',
        max_denom: int = 100,
        bins: Optional[int] = None,
        figsize: Optional[Tuple[float, float]] = None,
        show_common: bool = True,
        groupby: Optional[str] = None
    ):
        """
        Plot histogram of all intervals across time series to identify most recurrent intervals.
        
        Parameters
        ----------
        scale_type : str, default='diss_scale'
            Type of scale: 'diss_scale', 'HE_scale', 'euler_fokker', 'harm_tuning_scale'
        max_denom : int, default=100
            Maximum denominator for fraction simplification
        bins : int, default=50
            Number of bins for histogram
        figsize : tuple, default=(14, 6)
            Figure size
        show_common : bool, default=True
            Annotate most common intervals
        groupby : str, optional
            Metadata column to color by groups
        
        Returns
        -------
        fig, ax : matplotlib figure and axes
        
        Examples
        --------
        >>> btg.compute_diss_curve()
        >>> fig, ax = btg.plot_interval_histogram('diss_scale')
        >>> plt.show()
        """
        from fractions import Fraction
        
        if not self.store_objects:
            raise RuntimeError("Cannot access scales when store_objects=False")
        
        # Collect all intervals
        all_intervals = []
        colors = []
        
        for idx, bt in enumerate(self.objects):
            if hasattr(bt, scale_type):
                scale = getattr(bt, scale_type)
                if len(scale) > 0:
                    all_intervals.extend(scale)
                    if groupby is not None and self.index_df is not None:
                        group_val = self.index_df.iloc[idx][groupby]
                        colors.extend([group_val] * len(scale))
        
        if len(all_intervals) == 0:
            print(f"No {scale_type} data found. Compute scales first.")
            return None, None
        
        all_intervals = np.array(all_intervals)
        
        # Get plot configuration
        config = get_plot_config('group_tuning_histogram')
        if figsize is None:
            figsize = config['figsize']
        if bins is None:
            bins = config['bins']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if groupby is not None and len(colors) > 0:
            # Grouped histogram
            import pandas as pd
            df = pd.DataFrame({'interval': all_intervals, groupby: colors})
            palette = get_color_palette('biotuner_gradient', n_colors=len(df[groupby].unique()))
            for idx, group in enumerate(df[groupby].unique()):
                group_data = df[df[groupby] == group]['interval']
                ax.hist(group_data, bins=bins, alpha=0.65, label=str(group), 
                       edgecolor='white', linewidth=1.2, color=palette[idx])
            ax.legend(fontsize=11, framealpha=0.95)
        else:
            # Simple histogram
            ax.hist(all_intervals, bins=bins, edgecolor='white', linewidth=1.2, 
                   alpha=config['alpha'], color=config['color'])
        
        ax.set_xlabel(config['xlabel'], fontsize=13, fontweight='medium')
        ax.set_ylabel(config['ylabel'], fontsize=13, fontweight='medium')
        ax.set_title(f'{config["title_prefix"]}{self.n_series} Time Series', 
                     fontsize=16, fontweight='bold', pad=15)
        ax.grid(alpha=0.25, linewidth=0.8, axis='y')
        
        # Annotate most common intervals
        if show_common:
            from collections import Counter
            # Round to fractions
            fractions = [Fraction(float(x)).limit_denominator(max_denom) for x in all_intervals]
            counter = Counter(fractions)
            most_common = counter.most_common(5)
            
            y_pos = ax.get_ylim()[1] * 0.9
            text = "Most common intervals:\n"
            for frac, count in most_common:
                text += f"{frac} ({float(frac):.3f}): {count}×\n"
            
            ax.text(0.98, 0.98, text, transform=ax.transAxes,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   fontsize=10, family='monospace')
        
        plt.tight_layout()
        return fig, ax
    
    def plot_scale_size_distribution(
        self,
        scale_type: str = 'diss_scale',
        figsize: Optional[Tuple[float, float]] = None,
        groupby: Optional[str] = None
    ):
        """
        Plot distribution of scale sizes (number of notes) across time series.
        
        Parameters
        ----------
        scale_type : str, default='diss_scale'
            Type of scale to analyze
        figsize : tuple, default=(12, 6)
            Figure size
        groupby : str, optional
            Metadata column to group by
        
        Returns
        -------
        fig, ax : matplotlib figure and axes
        """
        if not self.store_objects:
            raise RuntimeError("Cannot access scales when store_objects=False")
        
        # Collect scale sizes
        scale_sizes = []
        groups = []
        
        for idx, bt in enumerate(self.objects):
            if hasattr(bt, scale_type):
                scale = getattr(bt, scale_type)
                scale_sizes.append(len(scale))
                if groupby is not None and self.index_df is not None:
                    groups.append(self.index_df.iloc[idx][groupby])
        
        if len(scale_sizes) == 0:
            print(f"No {scale_type} data found.")
            return None, None
        
        # Get plot configuration
        config = get_plot_config('group_scale_dist')
        if figsize is None:
            figsize = config['figsize']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if groupby is not None and len(groups) > 0:
            import pandas as pd
            df = pd.DataFrame({'size': scale_sizes, groupby: groups})
            palette = get_color_palette('biotuner_gradient', n_colors=len(df[groupby].unique()))
            sns.boxplot(data=df, x=groupby, y='size', ax=ax, palette=palette)
            ax.set_ylabel(config['xlabel'], fontsize=13, fontweight='medium')
            ax.set_xlabel(groupby, fontsize=13, fontweight='medium')
            ax.set_title(f'{config["title_prefix"]}by {groupby}', fontsize=16, fontweight='bold', pad=15)
        else:
            ax.hist(scale_sizes, bins=20, edgecolor='white', linewidth=1.2, 
                   alpha=config['alpha'], color=config['color'])
            ax.set_xlabel(config['xlabel'], fontsize=13, fontweight='medium')
            ax.set_ylabel(config['ylabel'], fontsize=13, fontweight='medium')
            ax.set_title(f'{config["title_prefix"]}(n={self.n_series})', 
                        fontsize=16, fontweight='bold', pad=15)
            ax.axvline(np.mean(scale_sizes), color=config['mean_line_color'], 
                      linestyle='--', linewidth=2.5, alpha=0.8,
                      label=f'Mean: {np.mean(scale_sizes):.1f}')
            ax.legend(fontsize=11, framealpha=0.95)
        
        ax.grid(alpha=0.25, linewidth=0.8, axis='y')
        plt.tight_layout()
        return fig, ax
    
    def plot_common_intervals(
        self,
        scale_type: str = 'diss_scale',
        top_n: int = 15,
        max_denom: int = 100,
        figsize: Optional[Tuple[float, float]] = None,
        min_occurrence: int = 2
    ):
        """
        Plot bar chart of most common intervals across all time series.
        
        Parameters
        ----------
        scale_type : str, default='diss_scale'
            Type of scale to analyze
        top_n : int, default=15
            Number of top intervals to show
        max_denom : int, default=100
            Maximum denominator for fractions
        figsize : tuple, default=(12, 8)
            Figure size
        min_occurrence : int, default=2
            Minimum number of occurrences to include
        
        Returns
        -------
        fig, ax : matplotlib figure and axes
        """
        from fractions import Fraction
        from collections import Counter
        
        if not self.store_objects:
            raise RuntimeError("Cannot access scales when store_objects=False")
        
        # Collect all intervals
        all_intervals = []
        for bt in self.objects:
            if hasattr(bt, scale_type):
                scale = getattr(bt, scale_type)
                all_intervals.extend(scale)
        
        if len(all_intervals) == 0:
            print(f"No {scale_type} data found.")
            return None, None
        
        # Convert to fractions and count
        fractions = [Fraction(float(x)).limit_denominator(max_denom) for x in all_intervals]
        counter = Counter(fractions)
        
        # Filter by minimum occurrence
        filtered = {k: v for k, v in counter.items() if v >= min_occurrence}
        most_common = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        if len(most_common) == 0:
            print(f"No intervals with at least {min_occurrence} occurrences found.")
            return None, None
        
        # Get plot configuration
        config = get_plot_config('group_tuning_common')
        if figsize is None:
            figsize = config['figsize']
        
        # Prepare data
        labels = [f"{frac}\n({float(frac):.3f})" for frac, _ in most_common]
        counts = [count for _, count in most_common]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create gradient colors for bars
        colors = get_color_palette('biotuner_gradient', n_colors=len(labels))
        bars = ax.barh(range(len(labels)), counts, color=colors, 
                      edgecolor='white', linewidth=1.5, alpha=config['bar_alpha'])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=11, family='monospace')
        ax.set_xlabel(config['xlabel'], fontsize=13, fontweight='medium')
        ax.set_title(f'{config["title_prefix"]}{scale_type}',
                    fontsize=16, fontweight='bold', pad=15)
        ax.grid(alpha=0.25, linewidth=0.8, axis='x')
        
        # Add percentage labels
        total = sum(counts)
        for i, (bar, count) in enumerate(zip(bars, counts)):
            percentage = (count / total) * 100
            ax.text(count, i, f'  {count} ({percentage:.1f}%)', 
                   va='center', fontsize=10, fontweight='medium')
        
        plt.tight_layout()
        return fig, ax
    
    def plot_tuning_comparison(
        self,
        scale_type: str = 'diss_scale',
        indices: Optional[List[int]] = None,
        max_denom: int = 100,
        figsize: Optional[Tuple[float, float]] = None
    ):
        """
        Compare tuning scales from multiple time series side by side.
        
        Parameters
        ----------
        scale_type : str, default='diss_scale'
            Type of scale to compare
        indices : list of int, optional
            Which time series to compare (default: first 5)
        max_denom : int, default=100
            Maximum denominator for fractions
        figsize : tuple, default=(14, 8)
            Figure size
        
        Returns
        -------
        fig, ax : matplotlib figure and axes
        """
        from fractions import Fraction
        
        if not self.store_objects:
            raise RuntimeError("Cannot access scales when store_objects=False")
        
        if indices is None:
            indices = list(range(self.n_series))
        
        # Get plot configuration
        config = get_plot_config('group_tuning_comparison')
        if figsize is None:
            figsize = config['figsize']
        
        fig, ax = plt.subplots(figsize=figsize)
        
        y_offset = 0
        colors = get_color_palette('biotuner_gradient', n_colors=len(indices))
        
        for i, idx in enumerate(indices):
            bt = self.objects[idx]
            if hasattr(bt, scale_type):
                scale = getattr(bt, scale_type)
                if len(scale) > 0:
                    # Convert to fractions
                    fracs = [Fraction(float(x)).limit_denominator(max_denom) 
                            for x in scale]
                    
                    # Plot as horizontal lines
                    for frac in fracs:
                        ax.plot([float(frac), float(frac)], [y_offset, y_offset + 0.8],
                               color=colors[i], linewidth=config['line_width'], 
                               alpha=config['alpha'], solid_capstyle='round')
                    
                    # Label - just the series number
                    label = f"{idx+1}"
                    
                    ax.text(0.02, y_offset + 0.4, label, transform=ax.get_yaxis_transform(),
                           fontsize=14, va='center', fontweight='bold')
                    
                    y_offset += 1
        
        ax.set_xlabel(config['xlabel'], fontsize=15, fontweight='bold')
        ax.set_ylabel(config['ylabel'], fontsize=15, fontweight='bold')
        ax.set_title(f'{config["title_prefix"]}{scale_type}',
                    fontsize=18, fontweight='bold', pad=15)
        ax.set_ylim(-0.5, y_offset)
        ax.set_yticks([])
        ax.grid(alpha=0.25, linewidth=0.8, axis='x')
        
        plt.tight_layout()
        return fig, ax
    
    def __repr__(self):
        """String representation."""
        status = []
        status.append(f"BiotunerGroup(n_series={self.n_series}, sf={self.sf} Hz)")
        status.append(f"  Data shape: {self.shape}")
        status.append(f"  Dimensions: {', '.join(self.axis_labels)}")
        
        if self._computed_methods:
            status.append(f"  Computed: {', '.join(sorted(self._computed_methods))}")
        else:
            status.append("  Status: No computations yet")
        
        if hasattr(self, 'index_df'):
            metadata_cols = [c for c in self.index_df.columns if c not in self.axis_labels + ['series_idx']]
            if metadata_cols:
                status.append(f"  Metadata: {', '.join(metadata_cols)}")
        
        return '\n'.join(status)
    
    def __len__(self):
        """Return number of time series."""
        return self.n_series
