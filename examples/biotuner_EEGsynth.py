import biotuner
from biotuner.biotuner_functions import scale_consonance, scale_to_metrics
from biotuner.biotuner_object import *

sf = 240
data = []
'''['peaks_ratios', 'diss_curve', 'diss_curve_extended']'''
scale_type = 'diss_curve'

biotuning = biotuner(sf, peaks_function = 'EEMD', precision = 0.5, n_harm = 10, scale_cons_limit = 0.1) # Initialize biotuner object
biotuning.peaks_extraction(data)
biotuning.peaks_extension(method = 'harmonic_fit', harm_function = 'mult', cons_limit = 0.1, 
                          ratios_extension = True, harm_bounds = 0.5)


if scale_type == 'diss_curve':
    biotuning.compute_diss_curve(plot = False, input_type = 'peaks', denom = 50, max_ratio = 2, n_tet_grid = 12)
    scale = biotuning.diss_scale
if scale_type == 'peaks_ratios':
    scale = biotuning.peaks_ratios
if scale_type == 'diss_curve_extended':
    biotuning.compute_diss_curve(plot = False, input_type = 'extended_peaks', denom = 50, max_ratio = 2, n_tet_grid = 12)
    scale = biotuning.diss_scale
    
scale_cons = scale_consonance(scale, dyad_similarity)

'''scale and scale_cons would be send via OSC'''