import numpy as np
import math
import scipy
from fractions import Fraction
import itertools
import biotuner
from biotuner.biotuner_utils import *
import matplotlib.pyplot as plt
from numpy import array, zeros, ones, arange, log2, sqrt, diff, concatenate
import pytuning
from math import gcd
from numpy import array, zeros, ones, arange, log2, sqrt, diff, concatenate
from scipy.stats import norm
from scipy.signal import argrelextrema, detrend
import scipy.signal as ss
from pytuning import create_euler_fokker_scale
from collections import Counter
from functools import reduce
from pytuning.utilities import normalize_interval
from pactools import Comodulogram, REFERENCES


'''---------------------------------------------------------Extended peaks-------------------------------------------------------------'''

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


def harmonic_fit(peaks, n_harm = 10, bounds = 1, function = 'mult', div_mode = 'div', n_common_harms = 5):
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
        
    Returns
    -------
    
    
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
    harm_list = []
    harmonics = []
    for i in range(len(list_peaks)):
        harms, _, _, d, e, harm_list = compareLists(multi_harmonics[list_peaks[i][0]], multi_harmonics[list_peaks[i][1]], bounds)
        harm_temp.append(harms)
        harm_list1.append(d)
        harm_list2.append(e)
        harmonics.append(harm_list)
    harm_fit = np.array(harm_temp).squeeze()
    harmonics = reduce(lambda x, y: x+y, harmonics)
    most_common_harmonics= [h for h, h_count in Counter(harmonics).most_common(n_common_harms) if h_count > 1]
    harmonics = list(np.sort(list(set(harmonics))))
    if len(peak_bands) > 2:
        harm_fit = list(itertools.chain.from_iterable(harm_fit))
        harm_fit = [round(num, 3) for num in harm_fit]
        harm_fit = list(dict.fromkeys(harm_fit))
        harm_fit = list(set(harm_fit))
    return harm_fit, harm_list1, harm_list2, harmonics, most_common_harmonics

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




def consonant_ratios (peaks, limit, sub = False, input_type = 'peaks', metric = 'cons'):
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
        if metric == 'cons':
            cons_ = (frac.numerator + frac.denominator)/(frac.numerator * frac.denominator)
        if metric == 'harmsim':
            cons_ = dyad_similarity(ratio)
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
    #print('NAN', np.argwhere(np.isnan(data)))
    out = []
    positions = []
    for count, peaks in enumerate(data):
        peaks = [x for x in peaks if x >= 0]
        if method == 'cons':
            cons, b, peaks_cons, d = consonance_peaks(peaks, limit)
            #print(peaks_cons)
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


'''
    ##################################################   PEAKS METRICS    ############################################################
'''

#Consonance#
#Input: peaks


def consonance (ratio, limit = 1000):
    ''' Compute metric of consonance from a single ratio of frequency
    
    ratio: float
    limit: int
        Defaults to 1000
        Maximum value of the denominator of the fraction representing the ratio  
    '''
    ratio = Fraction(float(ratio)).limit_denominator(limit)
    cons = (ratio.numerator + ratio.denominator)/(ratio.numerator * ratio.denominator)
    return cons

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


def metric_denom(ratio):
    '''Function that computes the denominator of the normalized ratio
    ratio: float
    '''
    
    ratio = sp.Rational(ratio).limit_denominator(10000)
    normalized_degree = normalize_interval(ratio)
    y = int(sp.fraction(normalized_degree)[1])
    return y


'''SCALE METRICS'''
'''Metric of harmonic similarity represents the degree of similarity between a scale and the natural harmonic series ###
   Implemented from Gill and Purves (2009)'''

def dyad_similarity(ratio):
    '''
    This function computes the similarity between a dyad of frequencies and the natural harmonic series 
    ratio: float
        frequency ratio
    '''
    frac = Fraction(float(ratio)).limit_denominator(1000)
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
        sims.append(dyad_similarity(f.numerator/f.denominator))
    similarity = np.array(sims)
    return similarity

def scale_cons_matrix (scale, function):
    '''
    This function gives a metric of a scale corresponding to the averaged metric for each pairs of ratios (matrix)
    scale: List (float)
    function: function
        possible functions: dyad_similarity
        consonance
        metric_denom
        
    '''
    metric_values = []
    mode_values = []
    for index1 in range(len(scale)):
        for index2 in range(len(scale)):
            if scale[index1] > scale[index2]:  #not include the diagonale in the computation of the avg. consonance
                entry = scale[index1]/scale[index2]
                mode_values.append([scale[index1], scale[index2]])
                metric_values.append(function(entry))
    return np.average(metric_values)

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
    scale_metrics['matrix_harm_sim'] = scale_cons_matrix(scale, dyad_similarity)
    scale_metrics['matrix_cons'] = scale_cons_matrix(scale, consonance)
    scale_metrics_list = []
    for value in scale_metrics.values():
        scale_metrics_list.append(value)
    return scale_metrics, scale_metrics_list


def scale_consonance (scale, function, rounding = 4):
    '''
    Function that gives the average consonance of each scale interval
    scale: List (float)
        scale to reduce
    function: function
        function used to compute the consonance between pairs of ratios
        Choose between: consonance, dyad_similarity, metric_denom
    '''
    metric_values = []
    mode_values = []
    for index1 in range(len(scale)):
        metric_value = []
        for index2 in range(len(scale)):          
            entry = scale[index1]/scale[index2]
            mode_values.append([scale[index1], scale[index2]])
            metric_value.append(function(entry))
        metric_values.append(np.average(metric_value))    
    return metric_values

'''
    ################################################   SCALE CONSTRUCTION  ##############################################################

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

def multi_oct_subdiv (peaks, max_sub = 100, octave_limit = 1.01365, octave = 2, n_scales = 10, cons_limit = 0.1):
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
    #a, b, pairs, cons = consonance_peaks(peaks, cons_limit)
    ratios, cons = consonant_ratios(peaks, cons_limit)
    list_oct_div = []
    for i in range(len(ratios)):
        list_temp, _ = oct_subdiv(ratios[i], octave_limit, octave, n_scales)
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

def harmonic_tuning (list_harmonics, octave = 2, min_ratio = 1, max_ratio = 2):
    '''
    Function that computes a tuning based on a list of harmonic positions
    
    list_harmonics: List (int)
        harmonic positions to use in the scale construction
    octave: int
    min_ratio: float
    max_ratio: float
    '''
    ratios = []
    for i in list_harmonics:
        ratios.append(rebound(1*i, min_ratio, max_ratio, octave))
    ratios = list(set(ratios))
    ratios = list(np.sort(np.array(ratios)))
    return ratios

def euler_fokker_scale(intervals, n = 1):
    '''
    Function that takes as input a series of intervals and derives a Euler Fokker Genera scale
    intervals: List (float)
    n: int
        Defaults to 1
        number of times the interval is used in the scale generation
    '''
    multiplicities = [n for x in intervals]
    scale = create_euler_fokker_scale(intervals, multiplicities)
    return scale
    
def generator_interval_tuning (interval = 3/2, steps = 12, octave = 2):
    '''
    Function that takes a generator interval and derives a tuning based on its stacking.
    interval: float
        Generator interval
    steps: int
        Defaults to 12 (12-TET for interval 3/2)
        Number of steps in the scale
    octave: int
        Defaults to 2
        Value of the octave
    '''
    scale = []
    for s in range(steps):
        s += 1
        degree = interval**s
        while degree > octave:
            degree = degree/octave
        scale.append(degree)
    return sorted(scale)


#function that takes two ratios a input (boundaries of )
#The mediant corresponds to the interval where small and large steps are equal.

def tuning_range_to_MOS (frac1, frac2, octave = 2, max_denom_in = 100, max_denom_out = 100):
    gen1 = octave**(frac1)
    gen2 = octave**(frac2)
    a = Fraction(frac1).limit_denominator(max_denom_in).numerator
    b = Fraction(frac1).limit_denominator(max_denom_in).denominator
    c = Fraction(frac2).limit_denominator(max_denom_in).numerator
    d = Fraction(frac2).limit_denominator(max_denom_in).denominator
    print(a, b, c, d)
    mediant = (a+c)/(b+d)
    mediant_frac = sp.Rational((a+c)/(b+d)).limit_denominator(max_denom_out)
    gen_interval = octave**(mediant)
    gen_interval_frac = sp.Rational(octave**(mediant)).limit_denominator(max_denom_out)
    MOS_signature = [d, b]
    invert_MOS_signature = [b, d]
    
    
    return mediant, mediant_frac, gen_interval, gen_interval_frac, MOS_signature, invert_MOS_signature


#def tuning_embedding ()

def stern_brocot_to_generator_interval (ratio, octave = 2):
    gen_interval = octave**(ratio)
    return gen_interval


def gen_interval_to_stern_brocot (gen):
    root_ratio = log2(gen)
    return root_ratio


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
    #print(ratios_euler)
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


'''Scale reduction'''

def scale_reduction (scale, mode_n_steps, function, rounding = 4):
    '''
    Function that reduces the number of steps in a scale according to the consonance between pairs of ratios
    scale: List (float)
        scale to reduce
    mode_n_steps: int
        number of steps of the reduced scale
    function: function
        function used to compute the consonance between pairs of ratios
        Choose between: consonance, dyad_similarity, metric_denom
    '''
    metric_values = []
    mode_values = []
    for index1 in range(len(scale)):
        for index2 in range(len(scale)):
            if scale[index1] > scale[index2]:  #not include the diagonale in the computation of the avg. consonance
                entry = scale[index1]/scale[index2]
                #print(entry_value, scale[index1], scale[index2])
                mode_values.append([scale[index1], scale[index2]])
                #if function == metric_denom:
                #    metric_values.append(int(function(sp.Rational(entry).limit_denominator(1000))))
                #else:
                metric_values.append(function(entry))
    if function == metric_denom:
        cons_ratios = [x for _, x in sorted(zip(metric_values, mode_values))]
    else:
        cons_ratios = [x for _, x in sorted(zip(metric_values, mode_values))][::-1]
    i = 0
    mode_ = []
    mode_out = []
    while len(mode_out) < mode_n_steps: 
        cons_temp = cons_ratios[i]  
        mode_.append(cons_temp)
        mode_out_temp = [item for sublist in mode_ for item in sublist]
        mode_out_temp = [np.round(x, rounding) for x in mode_out_temp]
        mode_out = sorted(set(mode_out_temp), key = mode_out_temp.index)[0:mode_n_steps]
        i +=1
    mode_metric = []
    for index1 in range(len(mode_out)):
        for index2 in range(len(mode_out)):
            if mode_out[index1] > mode_out[index2]:
                entry = mode_out[index1]/mode_out[index2]
                #if function == metric_denom:
                #    mode_metric.append(int(function(sp.Rational(entry).limit_denominator(1000))))
                #else:
                mode_metric.append(function(entry))
    return np.average(metric_values), mode_out, np.average(mode_metric)


'''------------------------------------------------------Peaks extraction--------------------------------------------------------------'''
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

def harmonic_peaks_fit (peaks, amps, min_freq = 0.5, max_freq = 30, min_harms = 2, harm_limit = 128):
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
                    if harm <= harm_limit:
                        n += 1
                        harm_temp.append(harm)
                        if p not in harm_peaks_temp:
                            harm_peaks_temp.append(p)
                        if p2 not in harm_peaks_temp:
                                harm_peaks_temp.append(p2)
        n_total.append(n)
        harm_.append(harm_temp)
        harm_peaks.append(harm_peaks_temp)
        if n >= min_harms:
            max_n.append(n)
            max_peaks.append(p)
            max_amps.append(a)
            #print(harm_temp)
            harmonics.append(harm_temp)
            harmonic_peaks.append(harm_peaks)
            harm_peaks_fit.append([p, harm_temp, harm_peaks_temp])
    for i in range(len(harm_peaks_fit)):
        harm_peaks_fit[i][2] = sorted(harm_peaks_fit[i][2])
    max_n = np.array(max_n)
    max_peaks = np.array(max_peaks)
    max_amps = np.array(max_amps)
    harmonics = np.array(harmonics)
    #print(harmonics.shape)
    harmonic_peaks = np.array(harmonic_peaks)
    #harm_peaks_fit = np.array(harm_peaks_fit)

    #max_indexes = np.argsort(n_total)[-10:]
    
    return max_n, max_peaks, max_amps, harmonics, harmonic_peaks, harm_peaks_fit


def cepstrum(signal, sample_freq, plot_cepstrum = False, min_freq=1.5, max_freq=80):
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
        ax.set_xlim(0, max_freq)
        fig, ax = plt.subplots()
        ax.plot(quefrency_vector, np.abs(cepstrum))
        ax.set_xlabel('quefrency (s)')
        ax.set_title('cepstrum')
        ax.set_xlim(1/max_freq, 1/min_freq)
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


'''--------------------------------------------------Phase-Amplitude Coupling-----------------------------------------------------------'''


def pac_frequencies(ts, sf, method = 'duprelatour', n_values = 10, drive_precision = 0.05, max_drive_freq = 6, min_drive_freq = 3
                   , sig_precision = 1, max_sig_freq = 50, min_sig_freq = 8, 
                   low_fq_width = 0.5, high_fq_width = 1, plot = False):
    
    drive_steps = int(((max_drive_freq-min_drive_freq)/drive_precision)+1)
    low_fq_range = np.linspace(min_drive_freq, max_drive_freq, drive_steps)   
    sig_steps = int(((max_sig_freq-min_sig_freq)/sig_precision)+1)
    high_fq_range = np.linspace(min_sig_freq, max_sig_freq, sig_steps)
    
    estimator = Comodulogram(fs=sf, low_fq_range=low_fq_range,
                             low_fq_width=low_fq_width, high_fq_width = high_fq_width, 
                             high_fq_range = high_fq_range, method=method,
                             progress_bar=False)
    estimator.fit(ts)
    indexes = top_n_indexes(estimator.comod_, n_values)[::-1]
    pac_freqs = []
    for i in indexes:
        pac_freqs.append([low_fq_range[i[0]], high_fq_range[i[1]]])
    
    if plot == True:
        estimator.plot(titles=[REFERENCES[method]])
    return pac_freqs



def pac_most_frequent(pac_freqs, n):
    drive_freqs = [x[0] for x in pac_freqs]
    signal_freqs = [x[1]for x in pac_freqs]

    drive_dict = {k: v for k, v in sorted(Counter(drive_freqs).items(), key=lambda item: item[1])}
    max_drive = list(drive_dict)[::-1][0:n]

    signal_dict = {k: v for k, v in sorted(Counter(signal_freqs).items(), key=lambda item: item[1])}
    max_signal = list(signal_dict)[::-1][0:n]
    
    return [max_signal, max_drive]

def pac_mode(pac_freqs, n, function = dyad_similarity):
    _, mode, _ = scale_reduction(scale_from_pairs(pac_freqs), mode_n_steps = n, function = function)
    return sorted(mode)


'''--------------------------------------------------------Biorhythms-----------------------------------------------------------------'''


def scale2euclid(scale, max_denom = 10, mode = 'normal'):
    euclid_patterns = []
    frac, num, denom = scale2frac(scale, maxdenom = max_denom)
    if mode == 'normal':
        for n, d in zip(num, denom):
            if d <= max_denom:
                try:
                    euclid_patterns.append(bjorklund(n, d))
                except:
                    pass
    if mode == 'full':
        for d, n in zip(num, denom):
            if d <= max_denom:
                steps = d*n
                try:
                    euclid_patterns.append(bjorklund(steps, d))
                    euclid_patterns.append(bjorklund(steps, n))
                except:
                    pass
    return euclid_patterns

def invert_ratio(ratio, n_steps_down, limit_denom = 64):
    inverted_ratio = 1/(ratio)
    i = 2
    if n_steps_down >= 1:
        while i <= n_steps_down:

            inverted_ratio = inverted_ratio/ratio
            i+=1

    
    frac = sp.Rational(inverted_ratio).limit_denominator(limit_denom)
    return frac, inverted_ratio

def binome2euclid(binome, n_steps_down = 1, limit_denom = 64):
    euclid_patterns = []
    fracs = []
    new_binome = []
    new_frac1, b1 = invert_ratio(binome[0], n_steps_down, limit_denom = limit_denom)
    new_frac2, b2 = invert_ratio(binome[1], n_steps_down, limit_denom = limit_denom)
    new_binome.append(b1)
    new_binome.append(b2)
    frac, num, denom = scale2frac(new_binome, limit_denom)
    if denom[0] != denom[1]:
        new_denom = denom[0]*denom[1]
        #print('denom', new_denom)
        #print('num1', num[0]*denom[1])
        #print('num2', num[1]*denom[0])
        try:
            euclid_patterns.append(bjorklund(new_denom, num[0]*denom[1]))
            euclid_patterns.append(bjorklund(new_denom, num[1]*denom[0]))
        except:
            pass
        
    else:
        new_denom = denom[0]
        
        try:
            euclid_patterns.append(bjorklund(new_denom, num[0]))
            euclid_patterns.append(bjorklund(new_denom, num[1]))
        except:
            pass
        
    return euclid_patterns, [new_frac1, new_frac2], [[num[0]*denom[1], new_denom], [num[1]*denom[0], new_denom]]


def consonant_euclid (scale, n_steps_down, limit_denom, limit_cons, limit_denom_final):
    
    pairs = getPairs(scale)
    new_steps = []
    euclid_final = []
    for p in pairs:
        euclid, fracs, new_ratios = binome2euclid(p, n_steps_down, limit_denom)
        #print('new_ratios', new_ratios)
        new_steps.append(new_ratios[0][1])
    pairs_steps = getPairs(new_steps)
    cons_steps = []
    for steps in pairs_steps:   
        #print(steps)
        try:
            steps1 = Fraction(steps[0]/steps[1]).limit_denominator(steps[1]).numerator
            steps2 = Fraction(steps[0]/steps[1]).limit_denominator(steps[1]).denominator
            #print(steps1, steps2)
            cons = (steps1 + steps2)/(steps1 * steps2)
            if cons >= limit_cons and steps[0] <= limit_denom_final and steps[1] <= limit_denom_final:
                cons_steps.append(steps[0])
                cons_steps.append(steps[1])
        except:
            continue
    for p in pairs:
        euclid, fracs, new_ratios = binome2euclid(p, n_steps_down, limit_denom)
        if new_ratios[0][1] in cons_steps:
            
            try:
                euclid_final.append(euclid[0])
                euclid_final.append(euclid[1]) #exception for when only one euclid has been computed (when limit_denom is very low, chances to have a division by zero)
            except:
                pass
    euclid_final = sorted(euclid_final)
    euclid_final = [euclid_final[i] for i in range(len(euclid_final)) if i == 0 or euclid_final[i] != euclid_final[i-1]]
    euclid_final = [i for i in euclid_final if len(Counter(i).keys()) != 1]
    return euclid_final, cons_steps

def interval_vector(euclid):
    indexes = [index+1 for index, char in enumerate(euclid) if char == 1]
    length = len(euclid)+1
    vector = [t - s for s, t in zip(indexes, indexes[1:])]
    vector = vector+[length-indexes[-1]]
    return vector

def bjorklund(steps, pulses):
    steps = int(steps)
    pulses = int(pulses)
    if pulses > steps:
        raise ValueError    
    pattern = []    
    counts = []
    remainders = []
    divisor = steps - pulses
    remainders.append(pulses)
    level = 0
    while True:
        counts.append(divisor // remainders[level])
        remainders.append(divisor % remainders[level])
        divisor = remainders[level]
        level = level + 1
        if remainders[level] <= 1:
            break
    counts.append(divisor)
    
    def build(level):
        if level == -1:
            pattern.append(0)
        elif level == -2:
            pattern.append(1)         
        else:
            for i in range(0, counts[level]):
                build(level - 1)
            if remainders[level] != 0:
                build(level - 2)
    
    build(level)
    i = pattern.index(1)
    pattern = pattern[i:] + pattern[0:i]
    return pattern

def interval_vec_to_string (interval_vectors):
    strings = []
    for i in interval_vectors:
        strings.append('E('+str(len(i))+','+str(sum(i))+')')
    return strings

def euclid_string_to_referent (strings, dict_rhythms):
    referent = []
    for s in strings:
        if s in dict_rhythms.keys():
            referent.append(dict_rhythms[s])
        else:
            referent.append('None')
    return referent

def euclid_long_to_short(pattern):
    steps = len(pattern)
    hits = pattern.count(1)
    return [hits, steps]