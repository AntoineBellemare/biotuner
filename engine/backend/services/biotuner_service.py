"""
Biotuner Analysis Service
Wraps biotuner library functionality for API use
"""

import numpy as np
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path

# Add parent biotuner package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from biotuner.biotuner_object import compute_biotuner
from biotuner.scale_construction import tuning_reduction
from biotuner.biotuner_utils import scale2frac
from biotuner.metrics import dyad_similarity


class BiotunerService:
    """Service for biotuner harmonic analysis"""
    
    def __init__(self):
        self.method_mapping = {
            "EMD": "EMD",
            "fixed": "fixed",
            "harmonic_recurrence": "harmonic_recurrence",
            "EIMC": "EIMC",
            "FOOOF": "FOOOF",
        }
    
    def analyze(
        self,
        data: np.ndarray,
        sf: float,
        method: str = "harmonic_recurrence",
        n_peaks: int = 5,
        precision: float = 1.0,
        max_freq: float = 100,
        tuning_method: str = "peaks_ratios",
        max_denominator: int = 100,
        n_harm: int = 10,
    ) -> Dict[str, Any]:
        """
        Perform biotuner harmonic analysis
        
        Parameters
        ----------
        data : np.ndarray
            Time series data
        sf : float
            Sampling frequency
        method : str
            Peak extraction method
        n_peaks : int
            Number of peaks to extract
        precision : float
            Frequency precision
        max_freq : float
            Maximum frequency for analysis
            
        Returns
        -------
        dict : Analysis results including peaks, tuning, harmonics
        """
        try:
            # Map method name
            bt_method = self.method_mapping.get(method, "harmonic_recurrence")
            
            # Run biotuner analysis
            bt = compute_biotuner(
                sf=sf,
                peaks_function=bt_method,
                precision=precision,
            )
            
            # Compute for the data - n_peaks goes here
            bt.peaks_extraction(
                data,
                n_peaks=n_peaks,
                max_freq=max_freq,
                ratios_extension=True,
                min_harms=2,
            )
            
            # Extract results - handle both numpy arrays and lists
            if hasattr(bt, 'peaks'):
                peaks = bt.peaks.tolist() if hasattr(bt.peaks, 'tolist') else list(bt.peaks)
            else:
                peaks = []
                
            if hasattr(bt, 'powers'):
                powers = bt.powers.tolist() if hasattr(bt.powers, 'tolist') else list(bt.powers)
            else:
                powers = []
            
            # Check if peaks were found
            if len(peaks) == 0:
                if method in ['harmonic_recurrence', 'EIMC']:
                    raise ValueError(
                        "No peaks detected. Try: (1) Reduce precision (try 0.1 Hz instead of 1 Hz), "
                        "(2) Increase max frequency, or (3) Try a different peak extraction method like 'EMD' or 'fixed'."
                    )
                elif method == 'EMD':
                    raise ValueError(
                        "No peaks detected with EMD method. Try: (1) Change to 'Harmonic Recurrence' method, "
                        "(2) Increase the signal duration, or (3) Check if your signal has enough variation."
                    )
                else:
                    raise ValueError(
                        "No peaks detected. Try: (1) Change peak extraction method, "
                        "(2) Adjust precision or max frequency parameters."
                    )
            
            if len(peaks) < 2:
                raise ValueError(
                    f"Only {len(peaks)} peak detected - need at least 2 for tuning analysis. "
                    "Try: (1) Reduce precision to find more peaks, (2) Increase max frequency, "
                    "(3) Use a longer signal segment, or (4) Try a different method."
                )
            
            # Compute tuning based on method
            tuning = []
            if len(peaks) > 0:
                if tuning_method == 'peaks_ratios':
                    # Use peaks_ratios attribute from biotuner
                    if hasattr(bt, 'peaks_ratios'):
                        pr = bt.peaks_ratios
                        tuning = pr.tolist() if hasattr(pr, 'tolist') else list(pr)
                    else:
                        # Fallback: Convert peaks to ratios relative to fundamental
                        fundamental = peaks[0] if peaks[0] > 0 else 1.0
                        tuning = [p / fundamental for p in peaks]
                        
                elif tuning_method == 'harmonic_fit':
                    # Compute harmonic fit tuning
                    tuning = bt.harmonic_fit_tuning(n_harm=n_harm, bounds=0.1, n_common_harms=50)
                    tuning = tuning.tolist() if hasattr(tuning, 'tolist') else list(tuning)
                    
                elif tuning_method == 'diss_curve':
                    # Compute dissonance curve tuning
                    bt.peaks_extension(
                        method='harmonic_fit',
                        harm_function='mult',
                        n_harm=20,
                        cons_limit=0.05,
                        ratios_extension=True,
                        scale_cons_limit=0.1
                    )
                    bt.compute_diss_curve(
                        plot=False,
                        input_type='extended_peaks',
                        euler_comp=False,
                        denom=max_denominator,
                        max_ratio=2,
                        n_tet_grid=12
                    )
                    if hasattr(bt, 'diss_scale'):
                        ds = bt.diss_scale
                        tuning = ds.tolist() if hasattr(ds, 'tolist') else list(ds)
                    else:
                        pr = bt.peaks_ratios
                        tuning = pr.tolist() if hasattr(pr, 'tolist') else list(pr)
                else:
                    # Default to peaks ratios
                    if hasattr(bt, 'peaks_ratios'):
                        pr = bt.peaks_ratios
                        tuning = pr.tolist() if hasattr(pr, 'tolist') else list(pr)
                    else:
                        fundamental = peaks[0] if peaks[0] > 0 else 1.0
                        tuning = [p / fundamental for p in peaks]
                
                # Round and remove duplicates
                tuning = np.round(np.unique(tuning), 5).tolist()
                
                # Convert to fractions with max denominator
                tuning_fracs, _, _ = scale2frac(tuning, max_denominator)
                # Convert fractions back to floats for JSON serialization
                tuning = [float(t) for t in np.unique(tuning_fracs)]
            
            # Compute harmonics metrics
            harmonics_info = None
            if hasattr(bt, 'harmonics'):
                harmonics_info = {
                    'harmonics': bt.harmonics.tolist() if hasattr(bt.harmonics, 'tolist') else list(bt.harmonics),
                    'harmonic_fit': getattr(bt, 'harmonic_fit', None)
                }
            
            # Compute additional metrics
            metrics = {}
            if hasattr(bt, 'cons_matrix'):
                metrics['consonance'] = float(np.mean(bt.cons_matrix))
            if hasattr(bt, 'tenney'):
                metrics['tenney_height'] = float(np.mean(bt.tenney))
            
            # Calculate tuning consonance
            if len(tuning) > 1:
                from biotuner.metrics import dyad_similarity
                tuning_consonance = []
                for i in range(len(tuning)):
                    for j in range(len(tuning)):
                        if tuning[i] != tuning[j]:
                            entry = tuning[i] / tuning[j]
                            tuning_consonance.append(dyad_similarity(entry))
                if tuning_consonance:
                    metrics['consonance'] = float(np.mean(tuning_consonance))
            
            return {
                'peaks': peaks,
                'powers': powers,
                'tuning': tuning,
                'harmonics': harmonics_info,
                'metrics': metrics,
                'method': bt_method,
                'tuning_method': tuning_method,
                'max_denominator': max_denominator,
                'n_harm': n_harm,
                'precision': precision,
                'max_freq': max_freq,
                'n_peaks': len(peaks)
            }
        
        except Exception as e:
            print(f"Error in biotuner analysis: {str(e)}")
            raise
    
    def reduce_tuning(
        self,
        tuning: List[float],
        n_steps: int = 12,
        max_ratio: float = 2.0
    ) -> List[float]:
        """
        Apply tuning reduction to generate a scale
        
        Parameters
        ----------
        tuning : list
            Original tuning ratios
        n_steps : int
            Number of steps in reduced scale
        max_ratio : float
            Maximum ratio (octave equivalence)
            
        Returns
        -------
        list : Reduced tuning scale
        """
        try:
            # Ensure tuning has at least 4 notes for reduction
            if len(tuning) < 4:
                print(f"Warning: Tuning has only {len(tuning)} notes, need at least 4 for reduction")
                return tuning
            
            # If requested steps >= current tuning size, return original
            if n_steps >= len(tuning):
                return tuning
            
            # Convert to numpy array if needed
            tuning_array = np.array(tuning)
            
            # Apply reduction with consonance function
            tuning_consonance, reduced_scale, mode_consonance = tuning_reduction(
                tuning=tuning_array,
                mode_n_steps=n_steps,
                function=dyad_similarity,  # Use dyad_similarity for consonance
                rounding=4,
                ratio_type="pos_harm"
            )
            
            # Convert back to floats and sort
            reduced_floats = sorted([float(r) for r in reduced_scale])
            
            # Return both the scale and consonance metrics
            return {
                'reduced_tuning': reduced_floats,
                'original_consonance': float(tuning_consonance),
                'reduced_consonance': float(mode_consonance)
            }
        
        except Exception as e:
            print(f"Error in tuning reduction: {str(e)}")
            # Fallback: return first n_steps
            return {
                'reduced_tuning': sorted(tuning)[:n_steps],
                'original_consonance': 0,
                'reduced_consonance': 0
            }
