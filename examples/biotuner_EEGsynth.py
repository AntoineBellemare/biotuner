import biotuner
from biotuner.biotuner_functions import scale_consonance, scale_to_metrics
from biotuner.biotuner_object import *

sf = 240
data = []
'''['peaks_ratios', 'diss_curve', 'diss_curve_extended']'''
scale_type = 'diss_curve'

biotuning = biotuner(sf, peaks_function = 'EEMD', precision = 0.5, n_harm = 10, scale_cons_limit = 0.1) # Initialize biotuner object
biotuning.peaks_extraction(data)
peaks = biotuning.peaks

biotuning.peaks_extension(method = 'harmonic_fit', harm_function = 'mult', cons_limit = 0.1, 
                          ratios_extension = True, harm_bounds = 0.5)


if scale_type == 'diss_curve':
    biotuning.compute_diss_curve(plot = False, input_type = 'peaks', denom = 50, max_ratio = 2, n_tet_grid = 12)
    scale = biotuning.diss_scale
    
if scale_type == 'peaks_ratios':
    scale = 1+biotuning.peaks_ratios
    
if scale_type == 'extended_peaks_ratios_cons':
    biotuning.peaks_extension(method = 'consonant_harmonic_fit', harm_function = 'mult',  n_harm = 20, 
                          cons_limit = 0.1, ratios_extension = True, scale_cons_limit = 0.235) 
    scale = biotuning.extended_peaks_ratios_cons
    
if scale_type == 'diss_curve_extended':
    biotuning.compute_diss_curve(plot = False, input_type = 'extended_peaks', denom = 50, max_ratio = 2, n_tet_grid = 12)
    scale = biotuning.diss_scale
    

scale_cons = scale_consonance(scale, dyad_similarity, rounding = 4)
scale_metrics, _ = scale_to_metrics(scale)
scale_metrics = scale_metrics.values
scale_ordered = sort_scale_by_consonance(scale)
_, mode_3, _ = scale_reduction(scale, 3, dyad_similarity, rounding = 4)
_, mode_4, _ = scale_reduction(scale, 4, dyad_similarity, rounding = 4)
_, mode_5, _ = scale_reduction(scale, 5, dyad_similarity, rounding = 4)
_, mode_6, _ = scale_reduction(scale, 6, dyad_similarity, rounding = 4)
_, mode_7, _ = scale_reduction(scale, 7, dyad_similarity, rounding = 4)

euclid_final, cons = consonant_euclid(scale_ordered, n_steps_down = 2, limit_denom = 8, 
                                      limit_cons = 0.1, limit_denom_final = 16)
euclid_short = [euclid_long_to_short(p) for p in euclid_final]
