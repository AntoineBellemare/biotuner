import numpy as np
import math
import scipy
from fractions import Fraction
import itertools
from biotuner_utils import *
from biotuner_offline import *
import matplotlib.pyplot as plt
from numpy import array, zeros, ones, arange, log2, sqrt, diff, concatenate
import pytuning
from math import gcd
from numpy import array, zeros, ones, arange, log2, sqrt, diff, concatenate
from scipy.stats import norm
from scipy.signal import argrelextrema, detrend
import scipy.signal as ss

'''EXTENDED PEAKS from expansions
'''

    
def EEG_harmonics_mult(peaks, n_harmonics, n_oct_up = 0):
    """
    Natural harmonics

    This function takes a list of frequency peaks as input and computes the desired number of harmonics
    with the formula: x, 2x, 3x ..., nx

    peaks: List (float)
        Peaks represent local maximum in a spectrum
    n_harmonics: int
        Number of harmonics to compute
    n_oct_up: int
        Defaults to 0. Corresponds to the number of octave the peaks are shifted 
    
    Returns
    -------
    multi_harmonics: array
        (n_peaks, n_harmonics + 1)
    """
    n_harmonics = n_harmonics + 2
    multi_harmonics = []
    multi_harmonics_rebound = []
    for p in peaks:
        multi_harmonics_r = []
        multi_harm_temp = []
        harmonics = []
        p = p * (2**n_oct_up)
        i = 1
        harm_temp = p
        while i < n_harmonics:
            harm_temp = p * i
            harmonics.append(harm_temp)
            i+=1
        multi_harmonics.append(harmonics)
    multi_harmonics = np.array(multi_harmonics)

    return multi_harmonics


def EEG_harmonics_div(peaks, n_harmonics, n_oct_up = 0, mode = 'div'):
    """
    Natural sub-harmonics

    This function takes a list of frequency peaks as input and computes the desired number of harmonics
    with using division: 

    peaks: List (float)
        Peaks represent local maximum in a spectrum
    n_harmonics: int
        Number of harmonics to compute
    n_oct_up: int
        Defaults to 0. Corresponds to the number of octave the peaks are shifted 
    mode: str
        Defaults to 'div'.
        'div': x, x/2, x/3 ..., x/n
        'div_add': x, (x+x/2), (x+x/3), ... (x+x/n)
        'div_sub': x, (x-x/2), (x-x/3), ... (x-x/n)
    Returns
    -------
    div_harmonics: array
        (n_peaks, n_harmonics + 1)
    div_harmonics_bounded: array
        (n_peaks, n_harmonics + 1)
    """
    n_harmonics = n_harmonics + 2
    div_harmonics = []
    for p in peaks:
        harmonics = []
        p = p * (2**n_oct_up)
        i = 1
        harm_temp = p
        while i < n_harmonics:
            if mode == 'div':
                harm_temp = (p/i)
            if mode == 'div_add':
                harm_temp = p + (p/i)
            if mode == 'div_sub':
                harm_temp = p - (p/i)
            harmonics.append(harm_temp)
            i+=1
        div_harmonics.append(harmonics)
    div_harmonics = np.array(div_harmonics)
    div_harmonics_bounded = div_harmonics.copy()
    #Rebound the result between 1 and 2
    for i in range(len(div_harmonics_bounded)):
        for j in range(len(div_harmonics_bounded[i])):
            div_harmonics_bounded[i][j] = rebound(div_harmonics_bounded[i][j])
    return div_harmonics, div_harmonics_bounded


def harmonic_fit(peaks, n_harm = 10, bounds = 1, function = 'mult', div_mode = 'div'):
    """
    This function computes harmonics of a list of peaks and compares the lists of harmonics pairwise to find fitting
    between the harmonic series

    peaks: List (float)
        Peaks represent local maximum in a spectrum
    n_harm: int
        Number of harmonics to compute
    bounds: int
        Minimum distance (in Hz) between two frequencies to consider a fit
    function: str
        Defaults to 'mult'.
        'mult' will use natural harmonics
        'div' will use natural sub-harmonics
    div_mode: str
        Defaults to 'div'. See EEG_harmonics_div function.
    """
    from itertools import combinations
    peak_bands = []
    for i in range(len(peaks)):
        peak_bands.append(i)
    if function == 'mult':
        multi_harmonics = EEG_harmonics_mult(peaks, n_harm)
    elif function == 'div':
        multi_harmonics, x = EEG_harmonics_div(peaks, n_harm, mode = div_mode)
    elif function == 'exp':
        multi_harmonics = []
        increments = []
        for h in range(n_harm+1):
            h += 1 
            multi_harmonics.append([i**h for i in peaks])
        multi_harmonics = np.array(multi_harmonics)
        multi_harmonics = np.moveaxis(multi_harmonics, 0, 1)
    #print(np.array(multi_harmonics).shape)
    list_peaks = list(combinations(peak_bands,2))
    #print(list_peaks)
    harm_temp = []
    harm_list1 = []
    harm_list2 = []
    for i in range(len(list_peaks)):
        harms, b, c, d, e = compareLists(multi_harmonics[list_peaks[i][0]], multi_harmonics[list_peaks[i][1]], bounds)
        harm_temp.append(harms)
        harm_list1.append(d)
        harm_list2.append(e)
    harm_fit = np.array(harm_temp).squeeze()

    if len(peak_bands) > 2:
        harm_fit = list(itertools.chain.from_iterable(harm_fit))
        harm_fit = [round(num, 3) for num in harm_fit]
        harm_fit = list(dict.fromkeys(harm_fit))
        harm_fit = list(set(harm_fit))
    return harm_fit, harm_list1, harm_list2

'''EXTENDED PEAKS from restrictions
'''

def consonance_peaks (peaks, limit):
    """
    This function computes consonance (for a given ratio a/b, when a < 2b, consonance corresponds to (a+b)/(a*b)) between peaks

    peaks: List (float)
        Peaks represent local maximum in a spectrum
    limit: float
        minimum consonance value to keep associated pairs of peaks
        
        Comparisons with familiar ratios:
        Unison-frequency ratio 1:1 yields a value of 2
        Octave-frequency ratio 2:1 yields a value of 1.5
        Perfect 5th-frequency ratio 3:2 yields a value of 0.833
        Perfect 4th-frequency ratio 4:3 yields a value of 0.583
        Major 6th-frequency ratio 5:3 yields a value of 0.533
        Major 3rd-frequency ratio 5:4 yields a value of 0.45
        Minor 3rd-frequency ratio 5:6 yields a value of 0.366
        Minor 6th-frequency ratio 5:8 yields a value of 0.325
        Major 2nd-frequency ratio 8:9 yields a value of 0.236
        Major 7th-frequency ratio 8:15 yields a value of 0.192
        Minor 7th-frequency ratio 9:16 yields a value of 0.174
        Minor 2nd-frequency ratio 15:16 yields a value of 0.129
        
    Returns
    -------
    
    consonance: List (float)
        consonance scores for each pairs of consonant peaks
    cons_pairs: List of lists (float)
        list of lists of each pairs of consonant peaks
    cons_peaks: List (float)
        list of consonant peaks (no doublons)
    cons_tot: float
        averaged consonance value for each pairs of peaks
        
    """
    from fractions import Fraction
    consonance_ = []
    peaks2keep = []
    peaks_consonance = []
    cons_tot = []
    for p1 in peaks:
        for p2 in peaks:
            peaks2keep_temp = []
            p2x = p2
            p1x = p1
            if p1x > p2x:
                while p1x > p2x:
                    p1x = p1x/2
            if p1x < p2x:
                while p2x > p1x:
                    p2x = p2x/2
            if p1x < 0.1:
                p1x = 0.06
            if p2x < 0.1:
                p2x = 0.06  #random  number to avoid division by 0
            ratio = Fraction(p2x/p1x).limit_denominator(1000)
            cons_ = (ratio.numerator + ratio.denominator)/(ratio.numerator * ratio.denominator)
            if cons_ < 1 :
                cons_tot.append(cons_)
            if cons_ > 1 or cons_ < limit:
                cons_ = None
                cons_ = None
                p2x = None
                p1x = None
            if p2x != None:
                peaks2keep_temp.extend([p2, p1])
            consonance_.append(cons_)
            peaks2keep.append(peaks2keep_temp)
        #cons_pairs = np.array(peaks2keep)
        cons_pairs = [x for x in peaks2keep if x]
        #consonance = np.array(consonance_)
        consonance = [i for i in consonance_ if i]
        cons_peaks = list(itertools.chain(*cons_pairs))
        cons_peaks = [np.round(c, 2) for c in cons_peaks]
        cons_peaks = list(set(cons_peaks))  
        #consonance = list(set(consonance))
    return consonance, cons_pairs, cons_peaks, np.average(cons_tot)


def multi_consonance(cons_pairs, n_freqs = 5):
    """
    Function that keeps the frequencies that are the most consonant with others
    Takes pairs of frequencies that are consonant (output of the 'compute consonance' function)

    cons_pairs: List of lists (float)
        list of lists of each pairs of consonant peaks
    n_freqs: int
        maximum number of consonant freqs to keep
    
    Returns
    -------
    
    freqs_related: List (float)
        peaks that are consonant with at least two other peaks, starting with the peak that is 
        consonant with the maximum number of other peaks
    """
    freqs_dup = list(itertools.chain(*cons_pairs))
    pairs_temp = list(itertools.chain.from_iterable(cons_pairs))
    freqs_nodup = list(dict.fromkeys(pairs_temp))
    f_count = []
    for f in freqs_nodup:
        f_count.append(freqs_dup.count(f))
    freqs_related = [x for _,x in sorted(zip(f_count,freqs_nodup))][-(n_freqs):][::-1]
    return freqs_related




def consonant_ratios (peaks, limit, sub = False, input_type = 'peaks'):
    """
    Function that computes integer ratios from peaks with higher consonance
    Needs at least two pairs of values
    
    peaks: List (float)
        Peaks represent local maximum in a spectrum
    limit: float
        minimum consonance value to keep associated pairs of peaks
    sub: boolean
        Defaults to False
        When set to True, include ratios a/b when a < b.
    
    Returns
    -------
    
    cons_ratios: List (float)
        list of consonant ratios
    consonance: List (float)
        list of associated consonance values
    """
    from fractions import Fraction
    consonance_ = []
    ratios2keep = []
    if input_type == 'peaks':
        ratios = compute_peak_ratios(peaks, sub = sub)
    if input_type == 'ratios':
        ratios = peaks
    for ratio in ratios:
        frac = Fraction(ratio).limit_denominator(1000)
        cons_ = (frac.numerator + frac.denominator)/(frac.numerator * frac.denominator)
        if cons_ > limit :
            consonance_.append(cons_)
            ratios2keep.append(ratio)
    #print(ratios2keep)
    
    ratios2keep = np.array(np.round(ratios2keep, 3))
    cons_ratios = np.sort(list(set(ratios2keep)))
    #cons_ratios = np.array(ratios2keep)
    #ratios = []
    #ratios = [ratios.append(x) for x in ratios2keep if x not in ratios]
    consonance = np.array(consonance_)
    consonance = [i for i in consonance if i]
    return cons_ratios, consonance


def timepoint_consonance (data, method = 'cons', limit = 0.2, min_notes = 3):
    """
    ## Function that keeps moments of consonance from multiple time series of peak frequencies

    data: List of lists (float)
        Axis 0 represents moments in time
        Axis 1 represents the sets of frequencies
    method: str
        Defaults to 'cons'
        'cons' will compute pairwise consonance between frequency peaks in the form of (a+b)/(a*b)
        'euler' will compute Euler's gradus suavitatis
    limit: float
        limit of consonance under which the set of frequencies are not retained
        When method = 'cons'
             --> See consonance_peaks method's doc to refer consonance values to common intervals 
        When method = 'euler'
             --> Major (4:5:6) = 9
                 Minor (10:12:15) = 9
                 Major 7th (8:10:12:15) = 10
                 Minor 7th (10:12:15:18) = 11
                 Diminish (20:24:29) = 38 
    min_notes: int
        minimum number of consonant frequencies in the chords. Only relevant when method is set to 'cons'. 
    Returns
    -------
    
    chords: List of lists (float)
        Axis 0 represents moments in time
        Axis 1 represents the sets of consonant frequencies
    positions: List (int)
        positions on Axis 0
    """
    
    data = np.moveaxis(data, 0, 1)
    print('NAN', np.argwhere(np.isnan(data)))
    out = []
    positions = []
    for count, peaks in enumerate(data):
        if method == 'cons':
            cons, b, peaks_cons, d = consonance_peaks(peaks, limit)
            out.append(peaks_cons)
            if len(list(set(peaks_cons))) >= min_notes:
                positions.append(count)
        if method == 'euler':
            peaks_ =  [int(np.round(p, 2)*100) for p in peaks]
            #print(peaks_)
            eul = euler(*peaks_)
            #print(eul)
            if eul < limit:
                out.append(list(peaks))
                positions.append(count)
    out = [x for x in out if x != []]
    #if method == 'cons':
    out = list(out for out,_ in itertools.groupby(out))
    chords = [x for x in out if len(x)>=min_notes]
    return chords, positions

'''SCALE CONSTRUCTION#
    ####################################   N-TET (one ratio)  ##############################################################

'''

def oct_subdiv(ratio, octave_limit = 0.01365 ,octave = 2 ,n = 5):
    '''
    N-TET tuning from Generator Interval
    This function uses a generator interval to suggest numbers of steps to divide the octave, 
    so the given interval will be approximately present (octave_limit) in the steps of the N-TET tuning.
    
    ratio: float
        ratio that corresponds to the generator_interval
        e.g.: by giving the fifth (3/2) as generator interval, this function will suggest to subdivide the octave in 12, 53, ...
    octave_limit: float 
        Defaults to 0.01365 (Pythagorean comma)
        approximation of the octave corresponding to the acceptable distance between the ratio of the generator interval after 
        multiple iterations and the octave value.
    octave: int
        Defaults to 2
        value of the octave
    n: int
        Defaults to 5
        number of suggested octave subdivisions
        
    Returns
    -------
    
    Octdiv: List (int)
        list of N-TET tunings corresponding to dividing the octave in equal steps
    Octvalue: List (float)
        list of the approximations of the octave for each N-TET tuning
    '''
    Octdiv, Octvalue, i = [], [], 1
    ratios = []
    while len(Octdiv) < n:
        ratio_mult = (ratio**i)
        while ratio_mult > octave:
            ratio_mult = ratio_mult/octave

        rescale_ratio = ratio_mult - round(ratio_mult)
        ratios.append(ratio_mult)
        i+=1
        if -octave_limit < rescale_ratio < octave_limit:
            Octdiv.append(i-1)
            Octvalue.append(ratio_mult)
        else:
            continue
    return Octdiv, Octvalue



def compare_oct_div(Octdiv = 12, Octdiv2 = 53, bounds = 0.005, octave = 2):
    '''
    Function that compare steps for two N-TET tunings and return matching ratios and corresponding degrees
    
    Octdiv: int
        Defaults to 12.
        first N-TET tuning number of steps
    Octdiv2: int
        Defaults to 53.
        second N-TET tuning number of steps
    bounds: float 
        Defaults to 0.005
        Maximum distance between 1 ratio of Octdiv and 1 ratio of Octdiv2 to consider a match
    octave: int
        Defaults to 2
        value of the octave
        
    Returns
    -------
    
    avg_ratios: List (float)
        list of ratios corresponding to the shared steps in the two N-TET tunings
    shared_steps: List of tuples
        the two elements of each tuple corresponds to the scale steps sharing the same interval in the two N-TET tunings
    '''
    ListOctdiv = []
    ListOctdiv2 = []
    OctdivSum = 1
    OctdivSum2 = 1
    i = 1
    i2 = 1
    while OctdivSum < octave:
        OctdivSum =(nth_root(octave, Octdiv))**i
        i+=1
        ListOctdiv.append(OctdivSum)
    while OctdivSum2 < octave:
        OctdivSum2 =(nth_root(octave, Octdiv2))**i2
        i2+=1
        ListOctdiv2.append(OctdivSum2)
    shared_steps = []
    avg_ratios = []
    for i, n in enumerate(ListOctdiv):
        for j, harm in enumerate(ListOctdiv2):
            if harm-bounds < n < harm+bounds:
                shared_steps.append((i+1, j+1))
                avg_ratios.append((n+harm)/2)
    return avg_ratios, shared_steps




#Output1: octave subdivisions
#Output2: ratios that led to Output1

def multi_oct_subdiv (peaks, max_sub = 100, octave_limit = 1.01365, octave = 2, n_scales = 10):
    '''
    This function uses the most consonant peaks ratios as input of oct_subdiv function. Each consonant ratio
    leads to a list of possible octave subdivisions. These lists are compared and optimal octave subdivisions are
    determined.
    
    peaks: List (float)
        Peaks represent local maximum in a spectrum
    max_sub: int
        Defaults to 100.
        Maximum number of intervals in N-TET tuning suggestions.
    octave_limit: float 
        Defaults to 1.01365 (Pythagorean comma).
        Approximation of the octave corresponding to the acceptable distance between the ratio of the generator interval after 
        multiple iterations and the octave value.
    octave: int
        Defaults to 2.
        value of the octave
    n_scales: int
        Defaults to 10.
        Number of N-TET tunings to compute for each generator interval (ratio).
        
    Returns
    -------
    
    multi_oct_div: List (int)
        List of octave subdivisions that fit with multiple generator intervals.
    ratios: List (float)
        list of the generator intervals for which at least 1 N-TET tuning match with another generator interval.  
    '''
    import itertools
    from collections import Counter
    a, b, pairs, cons = consonance_peaks(peaks, 0.01)
    ratios, cons = consonant_ratios(peaks, 0.01)
    list_oct_div = []
    for i in range(len(ratios)):
        list_temp, no= oct_subdiv(ratios[i], octave_limit, octave, n_scales)
        list_oct_div.append(list_temp)
    counts = Counter(list(itertools.chain(*list_oct_div)))
    oct_div_temp = []
    for k, v in counts.items():
        if v > 1:
            oct_div_temp.append(k)
    oct_div_temp = np.sort(oct_div_temp)
    multi_oct_div = []
    for i in range(len(oct_div_temp)):
        if oct_div_temp[i] < max_sub:
            multi_oct_div.append(oct_div_temp[i])
    return multi_oct_div, ratios


'''
    ########################################   PEAKS METRICS    ############################################################
'''

#Consonance#
#Input: peaks
def euler(*numbers):
    """
    Euler's "gradus suavitatis" (degree of sweetness) function
    Return the "degree of sweetness" of a musical interval or chord expressed
    as a ratio of frequencies a:b:c, according to Euler's formula
    Greater values indicate more dissonance
    
    numbers: List (int)
        frequencies
    """
    factors = prime_factors(lcm(*reduced_form(*numbers)))
    return 1 + sum(p - 1 for p in factors)

#Input: peaks
def tenneyHeight(peaks, avg = True):
    """
    Tenney Height is a measure of inharmonicity calculated on two frequencies (a/b) reduced in their simplest form. 
    It can also be called the log product complexity of a given interval.
    
    peaks: List (float)
        frequencies
    avg: Boolean
        Default to True
        When set to True, all tenney heights are averaged
    """
    pairs = getPairs(peaks)
    pairs
    tenney = []
    for p in pairs:
        try:
            frac = Fraction(p[0]/p[1]).limit_denominator(1000)
        except ZeroDivisionError:
            p[1] = 0.01
            frac = Fraction(p[0]/p[1]).limit_denominator(1000)
        x = frac.numerator
        y = frac.denominator
        tenney.append(log2(x*y))
    if avg == True:
        tenney = np.average(tenney)
    return tenney


def peaks_to_metrics (peaks, n_harm = 10):
    '''
    This function computes different metrics on peak frequencies.
    
    peaks: List (float)
        Peaks represent local maximum in a spectrum
    n_harm: int
        Number of harmonics to compute for 'harm_fit' metric
        
    Returns
    -------
    
    metrics: dict (float)
        Dictionary of values associated to metrics names
    metrics_list: List (float)
        list of peaks metrics values in the order: 'cons', 'euler', 'tenney', 'harm_fit'  
    '''
    peaks = list(peaks)
    metrics = {'cons' : 0, 'euler' : 0, 'tenney': 0, 'harm_fit': 0}
    harm_fit, harm_pos1, harm_pos2 = harmonic_fit(peaks, n_harm = n_harm)
    metrics['harm_pos1'] = harm_pos1
    metrics['harm_pos2'] = harm_pos2
    
    metrics['harm_fit'] = len(harm_fit)

    a, b, c, metrics['cons'] = consonance_peaks (peaks, 0.1)
    peaks_highfreq  = [int(p*1000) for p in peaks]
    
    metrics['euler'] = euler(*peaks_highfreq)
    metrics['tenney'] = tenneyHeight(peaks_highfreq) 
    metrics_list = []
    for value in metrics.values():
        metrics_list.append(value)
    return metrics, metrics_list

'''SCALE METRICS'''
'''Metric of harmonic similarity represents the degree of similarity between a scale and the natural harmonic series ###
   Implemented from Gill and Purves (2009)'''

def dyad_similarity(f1, f2):
    '''
    This function computes the similarity between a dyad of frequencies and the natural harmonic series 
    f1: float
        first frequency
    f2: float
        second frequency
    '''
    frac = Fraction(f1/f2).limit_denominator(1000)
    x = frac.numerator
    y = frac.denominator
    z = ((x+y-1)/(x*y))*100
    return z

#Input: ratios (list of floats) 
def ratios2harmsim (ratios):
    '''
    This function computes the similarity for each ratio of a list
    ratios: List (float)
        list of frequency ratios (forming a scale)
        
    Returns
    ---------
    similarity: List (float)
        list of percentage of similarity for each ratios
    '''
    fracs = []
    for r in ratios:
        fracs.append(Fraction(r).limit_denominator(1000))
    sims = []
    for f in fracs:
        sims.append(dyad_similarity(f.numerator, f.denominator))
    similarity = np.array(sims)
    return similarity


def PyTuning_metrics(scale, maxdenom):
    '''
    This function computes the scale metrics of the PyTuning library (https://pytuning.readthedocs.io/en/0.7.2/metrics.html)
    Smaller values are more consonant
    
    scale: List (float)
        List of ratios corresponding to scale steps
    maxdenom: int
        Maximum value of the denominator for each step's fraction
    
    '''
    scale_frac, num, denom = scale2frac(scale, maxdenom)
    metrics = pytuning.metrics.all_metrics(scale_frac)
    sum_p_q = metrics['sum_p_q']
    sum_distinct_intervals = metrics['sum_distinct_intervals']
    metric_3 = metrics['metric_3']
    sum_p_q_for_all_intervals = metrics['sum_p_q_for_all_intervals']
    sum_q_for_all_intervals = metrics['sum_q_for_all_intervals']
    return sum_p_q, sum_distinct_intervals, metric_3, sum_p_q_for_all_intervals, sum_q_for_all_intervals

def scale_to_metrics(scale):
    '''
    This function computes the scale metrics of the PyTuning library and other scale metrics
    
    scale: List (float)
        List of ratios corresponding to scale steps
        
    Returns
    ----------
    
    scale_metrics: dictionary
        keys correspond to metrics names
    scale_metrics_list: List (float)
        List of values corresponding to all computed metrics (in the same order as dictionary)
    '''
    scale_frac, num, denom = scale2frac(scale, maxdenom=1000)
    scale_metrics = pytuning.metrics.all_metrics(scale_frac)
    scale_metrics['harm_sim'] = np.round(np.average(ratios2harmsim(scale)), 2)
    scale_metrics_list = []
    for value in scale_metrics.values():
        scale_metrics_list.append(value)
    return scale_metrics, scale_metrics_list


#Dissonance
def dissmeasure(fvec, amp, model='min'):
    """
    Given a list of partials in fvec, with amplitudes in amp, this routine
    calculates the dissonance by summing the roughness of every sine pair
    based on a model of Plomp-Levelt's roughness curve.
    The older model (model='product') was based on the product of the two
    amplitudes, but the newer model (model='min') is based on the minimum
    of the two amplitudes, since this matches the beat frequency amplitude.
    """
    # Sort by frequency
    sort_idx = np.argsort(fvec)
    am_sorted = np.asarray(amp)[sort_idx]
    fr_sorted = np.asarray(fvec)[sort_idx]

    # Used to stretch dissonance curve for different freqs:
    Dstar = 0.24  # Point of maximum dissonance
    S1 = 0.0207
    S2 = 18.96

    C1 = 5
    C2 = -5

    # Plomp-Levelt roughness curve:
    A1 = -3.51
    A2 = -5.75

    # Generate all combinations of frequency components
    idx = np.transpose(np.triu_indices(len(fr_sorted), 1))
    fr_pairs = fr_sorted[idx]
    am_pairs = am_sorted[idx]

    Fmin = fr_pairs[:, 0]
    S = Dstar / (S1 * Fmin + S2)
    Fdif = fr_pairs[:, 1] - fr_pairs[:, 0]

    if model == 'min':
        a = np.amin(am_pairs, axis=1)
    elif model == 'product':
        a = np.prod(am_pairs, axis=1)  # Older model
    else:
        raise ValueError('model should be "min" or "product"')
    SFdif = S * Fdif
    D = np.sum(a * (C1 * np.exp(A1 * SFdif) + C2 * np.exp(A2 * SFdif)))

    return D

#Input: peaks and amplitudes
def diss_curve (freqs, amps, denom=1000, max_ratio=2, euler_comp = True, method = 'min', plot = True, n_tet_grid = None):
    '''
    This function computes the dissonance curve and related metrics for a given set of frequencies (freqs) and amplitudes (amps)
    
    freqs: List (float)
        list of frequencies associated with spectral peaks
    amps: List (float)
        list of amplitudes associated with freqs (must be same lenght)
    denom: int
        Defaults to 1000.
        Highest value for the denominator of each interval
    max_ratio: int
        Defaults to 2.
        Value of the maximum ratio
        Set to 2 for a span of 1 octave
        Set to 4 for a span of 2 octaves
        Set to 8 for a span of 3 octaves
        Set to 2**n for a span of n octaves
    euler: Boolean
        Defaults to True
        When set to True, compute the Euler Gradus Suavitatis for the derived scale
    method: str
        Defaults to 'min'
        Can be set to 'min' or 'product'. Refer to dissmeasure function for more information.
    plot: boolean
        Defaults to True
        When set to True, a plot of the dissonance curve will be generated
    n_tet_grid: int
        Defaults to None
        When an integer is given, dotted lines will be add to the plot a steps of the given N-TET scale 
    
    Returns
    -------

    intervals: List of tuples
        Each tuple corresponds to the numerator and the denominator of each scale step ratio
    ratios: List (float)
        list of ratios that constitute the scale
    euler_score: int
        value of consonance of the scale
    diss: float
        value of averaged dissonance of the total curve
    dyad_sims: List (float)
        list of dyad similarities for each ratio of the scale
        
    '''
    from numpy import array, linspace, empty, concatenate
    from scipy.signal import argrelextrema
    from fractions import Fraction
    freqs = np.array(freqs)
    r_low = 1
    alpharange = max_ratio
    method = method
    n = 1000
    diss = empty(n)
    a = concatenate((amps, amps))
    for i, alpha in enumerate(linspace(r_low, alpharange, n)):
        f = concatenate((freqs, alpha*freqs))
        d = dissmeasure(f, a, method)
        diss[i] = d
    diss_minima = argrelextrema(diss, np.less)
    
    
    intervals = []
    for d in range(len(diss_minima[0])):        
        frac = Fraction(diss_minima[0][d]/(n/(max_ratio-1))+1).limit_denominator(denom)
        frac = (frac.numerator, frac.denominator)
        intervals.append(frac)
    intervals.append((2, 1))
    ratios = [i[0]/i[1] for i in intervals]
    ratios_sim = [np.round(r, 2) for r in ratios] #round ratios for similarity measures of harmonic series
    #print(ratios_sim)
    dyad_sims = ratios2harmsim(ratios[:-1]) # compute dyads similarities with natural harmonic series
    dyad_sims
    a = 1
    ratios_euler = [a]+ratios    
    ratios_euler = [int(round(num, 2)*1000) for num in ratios]
    print(ratios_euler)
    euler_score = None
    if euler_comp == True:
        euler_score = euler(*ratios_euler)
        
        euler_score = euler_score/len(diss_minima)
    else:
        euler_score = 'NaN'
    
    if plot == True:
        plt.figure(figsize=(14, 6))
        plt.plot(linspace(r_low, alpharange, len(diss)), diss)
        plt.xscale('linear')
        plt.xlim(r_low, alpharange)
        try:
            plt.text(1.9, 1.5, 'Euler = '+str(int(euler_score)), horizontalalignment = 'center',
        verticalalignment='center', fontsize = 16)
        except:
            pass
        for n, d in intervals:
            plt.axvline(n/d, color='silver')
        # Plot N-TET grid
        if n_tet_grid != None:
            n_tet = NTET_ratios(n_tet_grid, max_ratio = max_ratio)
        for n in n_tet :
            plt.axvline(n, color='red', linestyle = '--')
        # Plot scale ticks
        plt.minorticks_off()
        plt.xticks([n/d for n, d in intervals],
                   ['{}/{}'.format(n, d) for n, d in intervals], fontsize = 13)
        plt.yticks(fontsize = 13)
        plt.tight_layout()
        plt.show()
    return intervals, ratios, euler_score, np.average(diss), dyad_sims

'''Harmonic Entropy'''

def compute_harmonic_entropy_domain_integral(ratios, ratio_interval, spread=0.01, min_tol=1e-15):
    
    # The first step is to pre-sort the ratios to speed up computation
    ind = np.argsort(ratios)
    weight_ratios = ratios[ind]
    
    centers = (weight_ratios[:-1] + weight_ratios[1:]) / 2
    
    ratio_interval = array(ratio_interval)
    N = len(ratio_interval)
    HE = zeros(N)
    for i, x in enumerate(ratio_interval):
        P = diff(concatenate(([0], norm.cdf(log2(centers), loc=log2(x), scale=spread), [1])))
        ind = P > min_tol
        HE[i] = -np.sum(P[ind] * log2(P[ind]))
    
    return weight_ratios, HE

def compute_harmonic_entropy_simple_weights(numerators, denominators, ratio_interval, spread=0.01, min_tol=1e-15):
    
    # The first step is to pre-sort the ratios to speed up computation
    ratios = numerators / denominators
    ind = np.argsort(ratios)
    numerators = numerators[ind]
    denominators = denominators[ind]
    weight_ratios = ratios[ind]
    
    ratio_interval = array(ratio_interval)
    N = len(ratio_interval)
    HE = zeros(N)
    for i, x in enumerate(ratio_interval):
        P = norm.pdf(log2(weight_ratios), loc=log2(x), scale=spread) / sqrt(numerators * denominators)
        ind = P > min_tol
        P = P[ind]
        P /= np.sum(P)
        HE[i] = -np.sum(P * log2(P))
    
    return weight_ratios, HE


def harmonic_entropy (ratios, res = 0.001, spread = 0.01, plot_entropy = True, plot_tenney = False, octave = 2):
    '''
    Harmonic entropy is a measure of the uncertainty in pitch perception, and it provides a physical correlate of tonalness, 
    one aspect of the psychoacoustic concept of dissonance (Sethares). High tonalness corresponds to low entropy and low tonalness
    corresponds to high entropy.
    
    ratios: List (float)
        ratios between each pairs of frequency peaks
    res: float
        Defaults to 0.001
        resolution of the ratio steps
    spread: float
        Default to 0.01
    plot_entropy: boolean
        Defaults to True
        When set to True, plot the harmonic entropy curve
    plot_tenney: boolean
        Defaults to False
        When set to True, plot the tenney heights (y-axis) across ratios (x-axis)
    octave: int
        Defaults to 2
        Value of the maximum interval ratio
        
    Returns
    ----------
    HE_minima: List (float)
        List of ratios corresponding to minima of the harmonic entropy curve
    HE: float
        Value of the averaged harmonic entropy
        
    '''
    fracs, numerators, denominators = scale2frac(ratios)
    ratios = numerators / denominators
    #print(ratios)
    #ratios = np.interp(ratios, (ratios.min(), ratios.max()), (1, 10))
    bendetti_heights = numerators * denominators
    tenney_heights = log2(bendetti_heights)

    ind = np.argsort(tenney_heights)  # first, sort by Tenney height to make things more efficient
    bendetti_heights = bendetti_heights[ind]
    tenney_heights = tenney_heights[ind]
    numerators = numerators[ind]
    denominators = denominators[ind]
    #ratios = ratios[ind]
    if plot_tenney == True:
        fig = plt.figure(figsize=(10, 4), dpi=150)
        ax = fig.add_subplot(111)
        # ax.scatter(ratios, 2**tenney_heights, s=1)
        ax.scatter(ratios, tenney_heights, s=1, alpha=.2)
        # ax.scatter(ratios[:200], tenney_heights[:200], s=1, color='r')
        plt.show()
    
    # Next, we need to ensure a distance `d` between adjacent ratios
    M = len(bendetti_heights)

    delta = 0.00001
    indices = ones(M, dtype=bool)

    for i in range(M - 2):
        ind = abs(ratios[i + 1:] - ratios[i]) > delta
        indices[i + 1:] = indices[i + 1:] * ind

    bendetti_heights = bendetti_heights[indices]
    tenney_heights = tenney_heights[indices]
    numerators = numerators[indices]
    denominators = denominators[indices]
    ratios = ratios[indices]
    M = len(tenney_heights)
    #print(M)
    #print('hello')
    x_ratios = arange(1, octave, res)
    _, HE = compute_harmonic_entropy_domain_integral(ratios, x_ratios, spread=spread)
    #_, HE = compute_harmonic_entropy_simple_weights(numerators, denominators, x_ratios, spread=0.01)
    ind = argrelextrema(HE, np.less)
    HE_minima = (x_ratios[ind], HE[ind])
    if plot_entropy == True:  
        fig = plt.figure(figsize=(10, 4), dpi=150)
        ax = fig.add_subplot(111)
        # ax.plot(weight_ratios, log2(pdf))
        ax.plot(x_ratios, HE)
        # ax.plot(x_ratios, HE_simple)
        ax.scatter(HE_minima[0], HE_minima[1], color='k', s=4)
        ax.set_xlim(1, octave)
        plt.show()
    return HE_minima, np.average(HE)


###PEAKS###
import emd
from PyEMD import EMD, EEMD
from scipy.signal import butter, lfilter
import colorednoise as cn

#PEAKS FUNCTIONS
#HH1D_weightAVG (Hilbert-Huang 1D): takes the average of all the instantaneous frequencies weighted by power
#HH1D_max: takes the frequency bin that has the maximum power value 

def compute_peaks_ts (data, peaks_function = 'EMD', FREQ_BANDS = None, precision = 0.25, sf = 1000, max_freq = 80):
    alphaband = [[7, 12]]
    try:
        if FREQ_BANDS == None:
            FREQ_BANDS = [[2, 3.55], [3.55, 7.15], [7.15, 14.3], [14.3, 28.55], [28.55, 49.4]]
    except:
        pass
    if peaks_function == 'EEMD':
        IMFs = EMD_eeg(data)[1:6]
    if peaks_function == 'EMD':
        data = np.interp(data, (data.min(), data.max()), (0, +1))
        IMFs = emd.sift.sift(data)
        #IMFs = emd.sift.ensemble_sift(data)
        IMFs = np.moveaxis(IMFs, 0, 1)[1:6]
    try:
        peaks_temp = []
        amps_temp = []
        for imf in range(len(IMFs)):
            p, a = compute_peak(IMFs[imf], precision = precision, average = 'median')
            #print(p)
            peaks_temp.append(p)
            
            amps_temp.append(a)
            
        peaks_temp = np.flip(peaks_temp)
        amps_temp = np.flip(amps_temp)
    except:
        pass
    if peaks_function == 'HH1D_max':
        IMFs = EMD_eeg(data)
        IMFs = np.moveaxis(IMFs, 0, 1)
        IP, IF, IA = emd.spectra.frequency_transform(IMFs[:, 1:6], sf, 'nht')
        precision_hh = precision*2
        low = 1
        high = max_freq
        steps = int((high-low)/precision_hh)
        edges, bins = emd.spectra.define_hist_bins(low, high, steps, 'log')
        
        # Compute the 1d Hilbert-Huang transform (power over carrier frequency)
        spec = emd.spectra.hilberthuang_1d(IF, IA, edges)
        spec = np.moveaxis(spec, 0, 1)
        peaks_temp = []
        amps_temp = []
        for e, i in enumerate(spec):
            max_power = np.argmax(i)
            peaks_temp.append(bins[max_power])
            amps_temp.append(spec[e][max_power])
        peaks_temp = np.flip(peaks_temp)
        amps_temp = np.flip(amps_temp)
    #if peaks_function == 'HH1D_weightAVG':
        
    if peaks_function == 'adapt':
        p, a = compute_peaks_raw(data, alphaband, precision = precision, average = 'median')
        FREQ_BANDS = alpha2bands(p)
        peaks_temp, amps_temp = compute_peaks_raw(data, FREQ_BANDS, precision = precision, average = 'median')
    if peaks_function == 'fixed':
        peaks_temp, amps_temp = compute_peaks_raw(data, FREQ_BANDS, precision = precision, average = 'median')
    peaks = np.array(peaks_temp)
    amps = np.array(amps_temp)
    return peaks, amps



def extract_all_peaks (data, sf, precision, max_freq = None):
    if max_freq == None:
        max_freq = sf/2
    mult = 1/precision
    nperseg = sf*mult
    nfft = nperseg
    freqs, psd = scipy.signal.welch(data, sf, nfft = nfft, nperseg = nperseg, average = 'median')
    psd = 10. * np.log10(psd)
    indexes = ss.find_peaks(psd, height=None, threshold=None, distance=10, prominence=None, width=2, wlen=None, rel_height=0.5, plateau_size=None)
    peaks = []
    amps = []
    for i in indexes[0]:
        peaks.append(freqs[i])
        amps.append(psd[i])
    peaks = np.around(np.array(peaks), 5)
    peaks = list(peaks)
    peaks = [p for p in peaks if p<=max_freq]
    return peaks, amps

def harmonic_peaks_fit (peaks, amps, min_freq = 0.5, max_freq = 30, min_harms = 2):
    n_total = []
    harm_ = []
    harm_peaks = []
    max_n = []
    max_peaks = []
    max_amps = []
    harmonics = []
    harmonic_peaks = []
    harm_peaks_fit = []
    for p, a in zip(peaks, amps):
        n = 0
        harm_temp = []
        harm_peaks_temp = []
        if p < max_freq and p > min_freq:
            
            for p2 in peaks:
                if p2 == p:
                    ratio = 0.1 #arbitrary value to set ratio value to non integer
                if p2 > p:
                    ratio = p2/p
                    harm = ratio
                if p2 < p:
                    ratio = p/p2    
                    harm = -ratio
                if ratio.is_integer():
                    n += 1

                    harm_temp.append(harm)

                    harm_peaks_temp.append(p)
                    harm_peaks_temp.append(p2)
        n_total.append(n)
        harm_.append(harm_temp)
        harm_peaks.append(list(set(harm_peaks_temp)))
        if n >= min_harms:
            max_n.append(n)
            max_peaks.append(p)
            max_amps.append(a)
            #print(harm_peaks)
            harmonics.append(harm_temp)
            harmonic_peaks.append(harm_peaks)
            harm_peaks_fit.append([p, harm_temp, list(set(harm_peaks_temp))])
    max_n = np.array(max_n)
    max_peaks = np.array(max_peaks)
    max_amps = np.array(max_amps)
    harmonics = np.array(harmonics)
    harmonic_peaks = np.array(harmonic_peaks)
    #harm_peaks_fit = np.array(harm_peaks_fit)

    #max_indexes = np.argsort(n_total)[-10:]
    
    return max_n, max_peaks, max_amps, harmonics, harmonic_peaks, harm_peaks_fit


def cepstrum(signal, sample_freq, plot_cepstrum = False):
    windowed_signal = signal
    dt = 1/sample_freq
    freq_vector = np.fft.rfftfreq(len(windowed_signal), d=dt)
    X = np.fft.rfft(windowed_signal)
    log_X = np.log(np.abs(X))

    cepstrum = np.fft.rfft(log_X)
    cepstrum = smooth(cepstrum, 10)
    df = freq_vector[1] - freq_vector[0]
    quefrency_vector = np.fft.rfftfreq(log_X.size, df)
    quefrency_vector = smooth(quefrency_vector, 10)

    if plot_cepstrum == True:
        fig, ax = plt.subplots()
        ax.plot(freq_vector, log_X)
        ax.set_xlabel('frequency (Hz)')
        ax.set_title('Fourier spectrum')
        ax.set_xlim(0, 50)
        fig, ax = plt.subplots()
        ax.plot(quefrency_vector, np.abs(cepstrum))
        ax.set_xlabel('quefrency (s)')
        ax.set_title('cepstrum')
        ax.set_xlim(0.02, 0.5)
        ax.set_ylim(0, 200)
    return cepstrum, quefrency_vector


def cepstral_peaks (cepstrum, quefrency_vector, max_time, min_time):

    indexes = ss.find_peaks(cepstrum, height=None, threshold=None, distance=None, prominence=None, width=3, wlen=None, rel_height=0.5,         plateau_size=None)
    #print(indexes[0])
    peaks = []
    amps = []
    for i in indexes[0]:
        if quefrency_vector[i] < max_time and quefrency_vector[i] > min_time:
            amps.append(np.abs(cepstrum)[i])
            peaks.append(quefrency_vector[i])
    peaks = np.around(np.array(peaks), 3)
    peaks = list(peaks)
    #peaks = [p for p in peaks if p<=max_freq]
    peaks = [1/p for p in peaks]
    return peaks, amps
'''OLD sanity_check_code'''
#--------------------------------------------------------------------------------------------------------------------------------------

def extract_peaks_EEGvsNOISE (epochs, conditions, peaks_function = 'EMD', HH=False, run = '0', sub = '0', precision = 0.25, alphaband = [[7, 13]]):
    par = ['Image_on_par_high', 'Image_on_par_mid', 'Image_on_par_low']
    high = ['Image_on_par_high', 'Image_on_nopar_high', 'early_high', 'late_high', 'nopar_high']
    mid = ['Image_on_par_mid', 'Image_on_nopar_mid', 'early_mid', 'late_mid', 'nopar_mid']
    low = ['Image_on_par_low', 'Image_on_nopar_low', 'early_low', 'late_low', 'nopar_low']
    nopar = ['Image_on_nopar_high', 'Image_on_nopar_mid', 'Image_on_nopar_low']
    early = ['early_high', 'early_low', 'early_mid']
    late = ['late_high','late_low', 'late_mid']
    par_rt = early+late
    nopar_rt = ['nopar_high', 'nopar_low', 'nopar_mid']
    epochs_data = epochs.get_data()
    for c in conditions:
        if c == 'high':
            epochs_data = epochs['Image_on_par_high', 'Image_on_nopar_high', 'early_high', 'late_high', 'nopar_high'].get_data()
        if c == 'mid':
            epochs_data = epochs['Image_on_par_mid', 'Image_on_nopar_mid', 'early_mid', 'late_mid', 'nopar_mid'].get_data()
        if c == 'low':
            epochs_data = epochs['Image_on_par_low', 'Image_on_nopar_low', 'early_low', 'late_low', 'nopar_low'].get_data()
        if c == 'par':
            epochs_data = epochs[par].get_data()
        if c == 'par_rt':
            epochs_data = epochs[par_rt].get_data()
        if c == 'nopar':
            epochs_data = epochs[nopar].get_data()
        if c == 'nopar_rt':
            epochs_data = epochs[nopar_rt].get_data()
        if c == 'early':
            epochs_data = epochs[early].get_data()
        if c == 'late':
            epochs_data = epochs[late].get_data()
        if c == 'pink':
            epochs_data = epochs.get_data()
            beta = 1
        if c == 'white':
            epochs_data = epochs.get_data()
            beta = 0
        if c == 'brown':
            epochs_data = epochs.get_data()
            beta = 2
        if c == 'blue':
            epochs_data = epochs.get_data()
            beta = -1
        peaks_tot = []
        amps_tot = []
        ## variables for Hilbert-Huang
        inds_tot = []
        euler_tot = []
        e_tot_ts = []
        IF_avg_tot = []
        for trial in range(len(epochs_data)):
            peaks_trial = []
            amps_trial = []
            ## variables for Hilbert-Huang
            inds_trial = []
            euler_trial = []
            e_tot_trial = []
            IF_avg_trial = []
            for ch in range(len(epochs_data[0])):
                #if :
                if c == 'eeg' or c =='high' or c =='mid' or c =='low' or c == 'par' or c == 'par_rt' or c =='nopar' or c =='nopar_rt' or c =='late' or c =='early':
                    data = epochs_data[trial][ch]
                if c == 'AAFT':
                    data = epochs_data[trial][ch]
                    indexes = [x for x in range(len(data))]
                    data = np.stack((data, indexes))
                    data = AAFT_surrogates(Surrogates, data)
                    data = data[0]
                    data = butter_bandpass_filter(data, 0.5, 150, 1000, 4)
                if c == 'phase':
                    phase_temp = epochs_data[trial][ch]
                    data = phaseScrambleTS(phase_temp)
                    data = butter_bandpass_filter(data, 0.5, 150, 1000, 4)
                if c == 'shuffle':
                    data = epochs_data[trial][ch]
                    np.random.shuffle(data)
                if c == 'white' or c == 'pink' or c == 'brown' or c == 'blue':
                    #print('noise data')
                    data = cn.powerlaw_psd_gaussian(beta, len(epochs_data[0][0]))
                    data = butter_bandpass_filter(data, 0.5, 150, 1000, 4)
                if peaks_function == 'adapt':
                    p, a = compute_peaks_raw(data, alphaband, precision = precision, average = 'median')
                    print(p)
                    FREQ_BANDS = alpha2bands(p)
                    print(FREQ_BANDS)
                    peaks_temp, amps_temp = compute_peaks_raw(data, FREQ_BANDS, precision = precision, average = 'median')
                if peaks_function == 'fixed':
                    peaks_temp, amps_temp = compute_peaks_raw(data, FREQ_BANDS, precision = precision, average = 'median')
                if peaks_function == 'EMD':
                    #print('hello')
                    if HH == False:
                        IMFs = EMD_eeg(data)[1:6]
                        peaks_temp = []
                        amps_temp = []
                        peaks_avg_temp = []
                        for imf in range(len(IMFs)):
                            p, a = compute_peak(IMFs[imf], precision = precision, average = 'median')
                            #print(p)
                            peaks_temp.append(p)
                            amps_temp.append(a)
                    if HH == True:
                        imf = emd.sift.sift(data)
                        IP, IF, IA = emd.spectra.frequency_transform(imf[:, 1:6], 1000, 'nht')
                        IF_avg_trial.append(IF)
                        e_good, cons_ind, e_tot = HH_cons(IF, euler_tresh = 30, mult = 10)
                        inds_temp = len(cons_ind)
                        euler_temp = np.average(e_good)
                
                if HH == False:
                    peaks_trial.append(peaks_temp)
                    amps_trial.append(amps_temp)
                if HH == True:
                    inds_trial.append(inds_temp)
                    euler_trial.append(euler_temp)
                    e_tot_trial.append(e_tot)
            if HH == False:
                peaks_tot.append(peaks_trial)
                amps_tot.append(amps_trial)
            if HH == True:
                inds_tot.append(inds_trial)
                euler_tot.append(euler_trial)
                e_tot_ts.append(e_tot_trial)
                IF_avg_tot.append(IF_avg_trial)
        if HH == False:
            peaks = np.array(peaks_tot)
            amps = np.array(amps_tot)
            np.save('s{}_peaks_RT_run{}_{}_{}'.format(sub, str(run), peaks_function, c), peaks)
            np.save('s{}_amps_RT_run{}_{}_{}'.format(sub, str(run), peaks_function, c), amps)
        if HH == True:
            inds = np.array(inds_tot)
            eul = np.array(euler_tot)
            eul_ts = np.array(e_tot_ts)
            np.save('s{}_n_cons_RT_run{}_{}_{}'.format(sub, str(run), peaks_function, c), inds)
            np.save('s{}_eulerGood_RT_run{}_{}_{}'.format(sub, str(run), peaks_function, c), eul)
            np.save('s{}_eulerTS_RT_run{}_{}_{}'.format(sub, str(run), peaks_function, c), eul_ts)
            np.save('s{}_IFavg_RT_run{}_{}_{}'.format(sub, str(run), peaks_function, c), IF_avg_tot)
            
            
            
def peaks2diss (freqs_amps, cond_name, ref=0, adjust = True, method = 'EMD', s = 0, run = 0, save = True):
    try:
        if ref.any() == None:
            ref = freqs_amps[0]
    except:
        pass
    a = adjust
    peaks = freqs_amps[0]
    peaks_diss = freqs_amps[0]
    amps = freqs_amps[1]
    ratio = np.round(np.average(ref)/np.average(peaks), 2)
    if a == True:
        peaks_diss = peaks_diss*ratio
    euler_tot = []
    tenney_tot = []
    diss_tot = []
    euler_diss_tot = []
    ratios_tot = []
    harm_sim_diss_tot = []
    harm_sim_tot = []
    harm_fit_tot = []
    for t in range(len(peaks)):

        euler_temp = []
        tenney_temp = []
        diss_temp = []
        euler_diss_temp = []
        ratios_temp = []
        harm_sim_diss_temp = []
        harm_sim_temp = []
        harm_fit_temp = []
        for ch in range(len(peaks[t])):
            #if a == True:
            #    ratio = np.round(np.average(ref[0][ch])/np.average(peaks[t][ch]), 2)
            #    peaks_diss[t][ch] = np.array(peaks[t][ch])*ratio
            f_temp = [int(np.round(p*128)) for p in peaks_diss[t][ch]]
            a_temp = [(float(i)-min(amps[t][ch]))/(max(amps[t][ch])-min(amps[t][ch])) for i in amps[t][ch]]
            a_temp = np.interp(a_temp, (np.array(a_temp).min(), np.array(a_temp).max()), (0.3, 0.7))
            ints_t, rats_t, euler_diss_t, diss_t, harm_sim_d = diss_curve_noplot(f_temp,a_temp, denom=1000, max_ratio=2, consonance = True)
            
            e_temp = [int(x*100) for x in peaks[t][ch]]
            #if pairwise == True:
                #pairs = getPairs(peaks[t][ch])
            e_temp = euler(*e_temp)
            tenney_t = tenneyHeight(list(peaks[t][ch]))
            print('euler', e_temp)
            na, harm_ratios = compute_peak_ratios(peaks[t][ch], sub=False)
            harm_fit, harm_pos1, harm_pos2 = harmonic_fit(peaks[t][ch], 50, 0.1)
            harm_sim = ratios2harmsim(harm_ratios)
            if a == True:
                euler_temp.append(e_temp*ratio)
                tenney_temp.append(tenney_t*ratio)
            if a == False:
                euler_temp.append(e_temp)
                tenney_temp.append(tenney_t)
            euler_diss_temp.append(euler_diss_t)
            ratios_temp.append(len(rats_t))
            diss_temp.append(diss_t)
            harm_sim_diss_temp.append(np.nanmean(harm_sim_d))
            harm_sim_temp.append(np.average(harm_sim))
            harm_fit_temp.append(len(harm_fit))
        euler_tot.append(euler_temp)
        tenney_tot.append(tenney_temp)
        diss_tot.append(diss_temp)
        ratios_tot.append(ratios_temp)
        euler_diss_tot.append(euler_diss_temp)
        harm_sim_diss_tot.append(harm_sim_diss_temp)
        harm_sim_tot.append(harm_sim_temp)
        harm_fit_tot.append(harm_fit_temp)
    euler_final = np.array(euler_tot)
    tenney_final = np.array(tenney_tot)
    diss_final = np.array(diss_tot)
    euler_diss_final = np.array(euler_diss_tot)
    ratios_final = np.array(ratios_tot)
    harm_sim_diss_final = np.array(harm_sim_diss_tot)
    harm_sim_final = np.array(harm_sim_tot)
    harm_fit_final = np.array(harm_fit_tot)
    if save == True:
        np.save('s{}_euler_RT_run{}_{}_{}_adapt-{}'.format(str(s), str(run), str(cond_name), method,str(a)), euler_final)
        np.save('s{}_tenney_RT_run{}_{}_{}_adapt-{}'.format(str(s), str(run), str(cond_name), method,str(a)), tenney_final)
        np.save('s{}_diss_RT_run{}_{}_{}_adapt-{}'.format(str(s), str(run), str(cond_name), method,str(a)), diss_final)
        np.save('s{}_diss_euler_RT_run{}_{}_{}_adapt-{}'.format(str(s), str(run), str(cond_name),method, str(a)), euler_diss_final)
        np.save('s{}_Nratios_RT_run{}_{}_{}_adapt-{}'.format(str(s), str(run), str(cond_name), method, str(a)), ratios_final)
        np.save('s{}_HarmSimDiss_RT_run{}_{}_{}_adapt-{}'.format(str(s), str(run), str(cond_name), method, str(a)), harm_sim_diss_final)
        np.save('s{}_HarmSim_RT_run{}_{}_{}_adapt-{}'.format(str(s), str(run), str(cond_name), method, str(a)), harm_sim_final)
        np.save('s{}_HarmFit_RT_run{}_{}_{}_adapt-{}'.format(str(s), str(run), str(cond_name), method, str(a)), harm_fit_final)
    return euler_final, tenney_final, diss_final, euler_diss_final, ratios_final, harm_sim_diss_final, harm_sim_final






def extract_peaks_sleep (epochs_data, condition, sf = 1000, peaks_function = 'EMD', HH=False, run = '0', sub = '0', precision = 0.25, alphaband = [[7, 13]]):
    c = condition
    if c == 'pink':
        beta = 1
    if c == 'white':
        beta = 0
    if c == 'brown':
        beta = 2
    if c == 'blue':
        beta = -1
    peaks_tot = []
    amps_tot = []
    ## variables for Hilbert-Huang
    inds_tot = []
    euler_tot = []
    e_tot_ts = []
    IF_avg_tot = []
    for trial in range(len(epochs_data)):
        peaks_trial = []
        amps_trial = []
        ## variables for Hilbert-Huang
        inds_trial = []
        euler_trial = []
        e_tot_trial = []
        IF_avg_trial = []
        for ch in range(len(epochs_data[0])):
            #if :
            if c == 'eeg' or c =='high' or c =='mid' or c =='low' or c == 'par' or c =='nopar' or c =='nopar_rt' or c =='late' or c =='early' or c=='wake' or c=='stage1' or c=='stage2' or c=='stage3' or c=='stage4' or c=='dream':
                data = epochs_data[trial][ch]
                data = data.astype('float64')
                print(data.shape)
            if c == 'AAFT':
                data = epochs_data[trial][ch]
                indexes = [x for x in range(len(data))]
                data = np.stack((data, indexes))
                data = AAFT_surrogates(Surrogates, data)
                data = data[0]
                data = butter_bandpass_filter(data, 0.5, 150, sf, 4)
            if c == 'phase':
                phase_temp = epochs_data[trial][ch]
                data = phaseScrambleTS(phase_temp)
                data = butter_bandpass_filter(data, 0.5, 150, sf, 4)
            if c == 'shuffle':
                data = epochs_data[trial][ch]
                np.random.shuffle(data)
            if c == 'white' or c == 'pink' or c == 'brown' or c == 'blue':
                #print('noise data')
                data = cn.powerlaw_psd_gaussian(beta, len(epochs_data[0][0]))
                data = butter_bandpass_filter(data, 0.5, 150, sf, 4)
            if peaks_function == 'adapt':
                p, a = compute_peaks_raw(data, alphaband, sf = sf, precision = precision, average = 'median')
                print(p)
                FREQ_BANDS = alpha2bands(p)
                print(FREQ_BANDS)
                peaks_temp, amps_temp = compute_peaks_raw(data, FREQ_BANDS, sf = sf, precision = precision, average = 'median')
            if peaks_function == 'fixed':
                peaks_temp, amps_temp = compute_peaks_raw(data, FREQ_BANDS, sf = sf, precision = precision, average = 'median')
            if peaks_function == 'EMD':
                #print('hello')
                if HH == False:
                    IMFs = EMD_eeg(data)[0:5]
                    print(IMFs.shape)
                    peaks_temp = []
                    amps_temp = []
                    peaks_avg_temp = []
                    for imf in range(len(IMFs)):
                        p, a = compute_peak(IMFs[imf], sf = sf, precision = precision, average = 'median')
                        p = p.tolist()
                        a = a.tolist()
                        peaks_temp.append(p)
                        print(peaks_temp)
                        amps_temp.append(a)
                if HH == True:
                    imf = emd.sift.sift(data)
                    IP, IF, IA = emd.spectra.frequency_transform(imf[:, 1:6], 1000, 'nht')
                    IF_avg_trial.append(IF)
                    e_good, cons_ind, e_tot = HH_cons(IF, euler_tresh = 30, mult = 10)
                    inds_temp = len(cons_ind)
                    euler_temp = np.average(e_good)

            if HH == False:

                peaks_trial.append(peaks_temp)

                amps_trial.append(amps_temp)
            if HH == True:
                inds_trial.append(inds_temp)
                euler_trial.append(euler_temp)
                e_tot_trial.append(e_tot)
        if HH == False:
            peaks_tot.append(peaks_trial)
            amps_tot.append(amps_trial)
        if HH == True:
            inds_tot.append(inds_trial)
            euler_tot.append(euler_trial)
            e_tot_ts.append(e_tot_trial)
            IF_avg_tot.append(IF_avg_trial)
    if HH == False:
        peaks = np.array(peaks_tot)
        amps = np.array(amps_tot)
        np.save('s{}_peaks_RT_run{}_{}_{}'.format(sub, str(run), peaks_function, c), peaks)
        np.save('s{}_amps_RT_run{}_{}_{}'.format(sub, str(run), peaks_function, c), amps)
    if HH == True:
        inds = np.array(inds_tot)
        eul = np.array(euler_tot)
        eul_ts = np.array(e_tot_ts)
        np.save('s{}_n_cons_RT_run{}_{}_{}'.format(sub, str(run), peaks_function, c), inds)
        np.save('s{}_eulerGood_RT_run{}_{}_{}'.format(sub, str(run), peaks_function, c), eul)
        np.save('s{}_eulerTS_RT_run{}_{}_{}'.format(sub, str(run), peaks_function, c), eul_ts)
        np.save('s{}_IFavg_RT_run{}_{}_{}'.format(sub, str(run), peaks_function, c), IF_avg_tot)


