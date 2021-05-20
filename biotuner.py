import numpy as np
import math
from fractions import Fraction
import itertools
from biotuner_utils import *
from biotuner_offline import *
import matplotlib.pyplot as plt
from numpy import array, zeros, ones, arange, log2, sqrt, diff, concatenate


#EXTENDED PEAKS from expansions#

    #This function takes a list of frequency peaks as input and computes the desired number of harmonics
#with the formula: x + 2x + 3x ... + nx
def EEG_harmonics_mult(peaks, n_harmonics, n_oct_up = 0):
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

#This function takes a list of frequency peaks as input and computes the desired number of harmonics
#with the formula: x + x/2 + x/3 ... + x/n
def EEG_harmonics_div(peaks, n_harmonics, n_oct_up = 0):
    n_harmonics = n_harmonics + 2
    multi_harmonics = []
    multi_harmonics_sub = []
    for p in peaks:

        harmonics = []
        harmonics_sub = []
        p = p * (2**n_oct_up)
        i = 2
        harm_temp = p
        harm_temp_sub = p
        while i < n_harmonics:
            harm_temp = harm_temp + (p/i)
            harm_temp_sub = abs(harm_temp_sub - (p/i))
            harmonics.append(harm_temp)
            harmonics_sub.append(harm_temp_sub)
            i+=1
        multi_harmonics.append(harmonics)
        multi_harmonics_sub.append(harmonics_sub)
    multi_harmonics = np.array(multi_harmonics)
    multi_harmonics_bounded = multi_harmonics.copy()
    multi_harmonics_sub = np.array(multi_harmonics_sub)
    multi_harmonics_sub_bounded = multi_harmonics_sub.copy()
    #Rebound the result between 1 and 2
    for i in range(len(multi_harmonics_bounded)):
        for j in range(len(multi_harmonics_bounded[0])):
            multi_harmonics_bounded[i][j] = rebound(multi_harmonics_bounded[i][j])
            multi_harmonics_sub_bounded[i][j] = rebound(multi_harmonics_sub_bounded[i][j])
    return multi_harmonics, multi_harmonics_bounded, multi_harmonics_sub, multi_harmonics_sub_bounded

#This function computes harmonics of a list of peaks and compares the lists of harmonics pairwise to find fitting
#between the harmonic series
def harmonic_fit(peaks, n_harm = 10, bounds = 1, function = 'mult'):
    from itertools import combinations
    peak_bands = []
    for i in range(len(peaks)):
        peak_bands.append(i)
    if function == 'mult':
        multi_harmonics = EEG_harmonics_mult(peaks, n_harm)
    elif function == 'div':
        multi_harmonics, x, y, z = EEG_harmonics_div(peaks, n_harm)
    #print(multi_harmonics)
    list_peaks = list(combinations(peak_bands,2))
    #print(list_peaks)
    harm_temp = []
    for i in range(len(list_peaks)):
        harms, b, c = compareLists(multi_harmonics[list_peaks[i][0]], multi_harmonics[list_peaks[i][1]], bounds)
        harm_temp.append(harms)
    harm_fit = np.array(harm_temp).squeeze()

    if len(peak_bands) > 2:
        harm_fit = list(itertools.chain.from_iterable(harm_fit))
        harm_fit = [round(num, 3) for num in harm_fit]
        harm_fit = list(dict.fromkeys(harm_fit))
    return harm_fit

#EXTENDED PEAKS from restrictions#

def consonance_peaks (peaks, limit):
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
        cons_pairs = np.array(peaks2keep)
        cons_pairs = [x for x in cons_pairs if x]
        consonance = np.array(consonance_)
        consonance = [i for i in consonance if i]
        cons_peaks = list(itertools.chain(*cons_pairs))
        cons_peaks = list(set(cons_peaks))
        #consonance = list(set(consonance))
    return consonance, cons_pairs, cons_peaks, np.average(cons_tot)

## Function that keeps the frequencies that are the most consonant with others
##Takes pairs of frequencies that are consonant (output of the 'compute consonance' function)
def multi_consonance(cons_pairs, n_freqs = 5):
    freqs_dup = list(itertools.chain(*cons_pairs))
    pairs_temp = list(itertools.chain.from_iterable(cons_pairs))
    freqs_nodup = list(dict.fromkeys(pairs_temp))
    f_count = []
    for f in freqs_nodup:
        f_count.append(freqs_dup.count(f))
    freqs_related = [x for _,x in sorted(zip(f_count,freqs_nodup))][-(n_freqs):][::-1]
    return freqs_related


# Function that computes integer ratios from peaks with higher consonance
# Needs at least two pairs of values
'''
def consonant_ (cons_peaks):
    from fractions import Fraction
    cons_integer = []
    for i in range(len(cons_peaks)):
        a = cons_peaks[i][0]
        b = cons_peaks[i][1]
        if a > b:
            while a > (b):
                a = a/2
        if a < b:
            while b > (a):
                b = b/2
        cons_temp = Fraction(a/b).limit_denominator(1000)
        num = cons_temp.numerator
        denom = cons_temp.denominator
        frac_list = [num, denom]
        frac = tuple(frac_list)
        cons_integer.append(frac)
    consonant_integers = np.array(cons_integer).squeeze()
    cons_ratios = []
    for j in range(len(consonant_integers)):
        cons_ratios.append((consonant_integers[j][0])/(consonant_integers[j][1]))
    consonant_ratios = np.array(cons_ratios)
    cons_rat = []
    for i in consonant_ratios:
        if i not in cons_rat:
            cons_rat.append(i)
    try:
        cons_rat.remove(1.0)
    except (ValueError, TypeError, NameError):
        pass
    return consonant_integers, cons_rat
'''
def consonant_ratios (peaks, limit, sub = False):
    from fractions import Fraction
    consonance_ = []
    ratios2keep = []
    a, ratios = compute_peak_ratios(peaks, sub = sub)
    for ratio in ratios:
        frac = Fraction(ratio).limit_denominator(1000)
        cons_ = (frac.numerator + frac.denominator)/(frac.numerator * frac.denominator)
        if cons_ > limit :
            consonance_.append(cons_)
            ratios2keep.append(ratio)
    #print(ratios2keep)
    
    ratios2keep = np.array(np.round(ratios2keep, 3))
    ratios2keep = list(set(ratios2keep))
    cons_ratios = np.array(ratios2keep)
    #ratios = []
    #ratios = [ratios.append(x) for x in ratios2keep if x not in ratios]
    consonance = np.array(consonance_)
    consonance = [i for i in consonance if i]
    return cons_ratios, consonance
#SCALE CONSTRUCTION#
    ####################################   N-TET (one ratio)  ##############################################################


    #Oct_subdiv
#Argument 1 : a ratio in the form of a float or a fraction
#Argument 2 : bounds between which the octave should fall
#Argument 3 : value of the octave
#Argument 4 : number of octave subdivisions

def oct_subdiv(ratio,bounds,octave,n):
    Octdiv, Octvalue, i = [], [], 1
    ratios = []
    while len(Octdiv) < n:
        ratio_mult = (ratio**i)
        while ratio_mult > octave:
            ratio_mult = ratio_mult/octave

        rescale_ratio = ratio_mult - round(ratio_mult)
        ratios.append(ratio_mult)
        i+=1
        if -bounds < rescale_ratio < bounds:
            Octdiv.append(i-1)
            Octvalue.append(ratio_mult)
        else:
            continue
    return Octdiv, Octvalue, ratios



def compare_oct_div(Octdiv = 53, Octdiv2 = 12, bounds = 0.01, octave = 2):
    ListOctdiv = []
    ListOctdiv2 = []
    OctdivSum = 1
    OctdivSum2 = 1
    i = 1
    i2 = 1
    Matching_harmonics = []
    #HARMONIC_RATIOS = [1, 1.0595, 1.1225, 1.1892, 1.2599, 1.3348, 1.4142, 1.4983, 1.5874, 1.6818, 1.7818, 1.8897]
    while OctdivSum < octave:
        OctdivSum =(nth_root(octave, Octdiv))**i
        i+=1
        ListOctdiv.append(OctdivSum)
    #print(ListOctdiv)

    while OctdivSum2 < octave:
        OctdivSum2 =(nth_root(octave, Octdiv2))**i2
        i2+=1
        ListOctdiv2.append(OctdivSum2)
    #print(ListOctdiv2)
    for i, n in enumerate(ListOctdiv):
        for j, harm in enumerate(ListOctdiv2):
            if harm-bounds < n < harm+bounds:
                Matching_harmonics.append([n, i+1, harm, j+1])
    Matching_harmonics = np.array(Matching_harmonics)
    return Matching_harmonics


##Here we use the most consonant peaks ratios as input of oct_subdiv function. Each consonant ratio
##leads to a list of possible octave subdivisions. These lists are compared and optimal octave subdivisions are
##determined.

#Output1: octave subdivisions
#Output2: ratios that led to Output1

def multi_oct_subdiv (peaks, max_sub, octave_limit):
    import itertools
    from collections import Counter
    a, b, pairs, cons = consonance_peaks(peaks, 0.01)
    c, ratios = consonant_ratios(b)

    list_oct_div = []
    for i in range(len(ratios)):
        list_temp, no, no2 = oct_subdiv(ratios[i], octave_limit, 2, 30)
        list_oct_div.append(list_temp)


    counts = Counter(list(itertools.chain(*list_oct_div)))
    oct_div_temp = []
    for k, v in counts.items():
        if v > 1:
            oct_div_temp.append(k)
    oct_div_temp = np.sort(oct_div_temp)
    oct_div_final = []
    for i in range(len(oct_div_temp)):
        if oct_div_temp[i] < max_sub:
            oct_div_final.append(oct_div_temp[i])
    return oct_div_final, ratios



    ########################################   METRICS    ############################################################

#Consonance#
#Input: peaks
def euler(*numbers):
    """
    Euler's "gradus suavitatis" (degree of sweetness) function
    Return the "degree of sweetness" of a musical interval or chord expressed
    as a ratio of frequencies a:b:c, according to Euler's formula
    Greater values indicate more dissonance
    """
    factors = prime_factors(lcm(*reduced_form(*numbers)))
    return 1 + sum(p - 1 for p in factors)

#Input: peaks
def tenneyHeight(peaks):
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
    return np.average(tenney)


def peaks_to_metrics (peaks, n_harm = 10):
    peaks = list(peaks)
    metrics = {'cons' : 0, 'euler' : 0, 'tenney': 0, 'harm_fit': 0}   
    metrics['harm_fit'] = len(harmonic_fit(peaks, n_harm = n_harm))
    a, b, c, metrics['cons'] = consonance_peaks (peaks, 0.1)
    metrics['euler'] = euler(*peaks)
    metrics['tenney'] = tenneyHeight(peaks)
    metrics_list = []
    for value in metrics.values():
        metrics_list.append(value)
    return metrics, metrics_list

'''SCALE METRICS'''
'''Metric of harmonic similarity represents the degree of similarity between a scale and the natural harmonic series ###
   Implemented from Gill and Purves (2009)'''

def dyad_similarity(f1, f2):
    frac = Fraction(f1/f2).limit_denominator(1000)
    x = frac.numerator
    y = frac.denominator
    z = ((x+y-1)/(x*y))*100
    return z

#Input: ratios (list of floats) 
def ratios2harmsim (ratios):
    dyads = getPairs(ratios)
    sims = []
    for d in dyads:
        sims.append(dyad_similarity(d[0], d[1]))
    similarity = np.array(sims)
    return similarity

'''Metrics from PyTuning library (https://pytuning.readthedocs.io/en/0.7.2/metrics.html)
   Smaller values are more consonant'''

def PyTuning_metrics(scale, maxdenom):
    scale_frac, num, denom = scale2frac(scale, maxdenom)
    metrics = pytuning.metrics.all_metrics(scale_frac)
    sum_p_q = metrics['sum_p_q']
    sum_distinct_intervals = metrics['sum_distinct_intervals']
    metric_3 = metrics['metric_3']
    sum_p_q_for_all_intervals = metrics['sum_p_q_for_all_intervals']
    sum_q_for_all_intervals = metrics['sum_q_for_all_intervals']
    return sum_p_q, sum_distinct_intervals, metric_3, sum_p_q_for_all_intervals, sum_q_for_all_intervals

def scale_to_metrics(scale):
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

'''
def diss_curve (freqs, amps, denom=1000, max_ratio=2, method = 'min'):
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
    plt.figure(figsize=(14, 6))
    plt.plot(linspace(r_low, alpharange, len(diss)), diss)
    plt.xscale('log')
    plt.xlim(r_low, alpharange)

    plt.xlabel('frequency ratio')
    plt.ylabel('sensory dissonance')


    diss_minima = argrelextrema(diss, np.less)
    print(diss_minima)
    intervals = []
    for d in range(len(diss_minima[0])):
        
        frac = Fraction(diss_minima[0][d]/(n/(max_ratio-1))+1).limit_denominator(denom)
        frac = (frac.numerator, frac.denominator)
        intervals.append(frac)

    
    intervals.append((2, 1))
    ratios = [i[0]/i[1] for i in intervals]
    a = 1
    ratios_euler = [a]+ratios
    ratios_euler = [int(round(num, 2)*1000) for num in ratios]
    euler_score = euler(*ratios_euler)
    plt.text(1.9, 1.5, 'Euler = '+str(int(euler_score)), horizontalalignment = 'center',
      verticalalignment='center', fontsize = 16)
    for n, d in intervals:
        plt.axvline(n/d, color='silver')

    #plt.yticks([])
    plt.minorticks_off()
    plt.xticks([n/d for n, d in intervals],
               ['{}/{}'.format(n, d) for n, d in intervals], fontsize = 13)
    plt.yticks(fontsize = 13)
    plt.tight_layout()
    plt.show()
    return intervals, ratios, euler_score/len(diss_minima), np.average(diss)
'''
#Input: peaks and amplitudes
def diss_curve (freqs, amps, denom=1000, max_ratio=2, consonance = True, method = 'min', plot = True):
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
    #print(diss_minima)
    intervals = []
    for d in range(len(diss_minima[0])):        
        frac = Fraction(diss_minima[0][d]/(n/(max_ratio-1))+1).limit_denominator(denom)
        frac = (frac.numerator, frac.denominator)
        intervals.append(frac)
    intervals.append((2, 1))
    ratios = [i[0]/i[1] for i in intervals]
    ratios_sim = [np.round(r, 2) for r in ratios] #round ratios for similarity measures of harmonic series
    dyad_sims = ratios2harmsim(ratios[:-1]) # compute dyads similarities with natural harmonic series
    dyad_sims
    a = 1
    ratios_euler = [a]+ratios
    ratios_euler = [int(round(num, 2)*1000) for num in ratios]
    euler_score = None
    if consonance == True:
        euler_score = euler(*ratios_euler)
        euler_score = euler_score/len(diss_minima)
    else:
        euler_score = 'NaN'
    #print(euler_score)
    if plot == True:
        plt.figure(figsize=(14, 6))
        plt.plot(linspace(r_low, alpharange, len(diss)), diss)
        plt.xscale('log')
        plt.xlim(r_low, alpharange)
        plt.text(1.9, 1.5, 'Euler = '+str(int(euler_score)), horizontalalignment = 'center',
        verticalalignment='center', fontsize = 16)
        for n, d in intervals:
            plt.axvline(n/d, color='silver')

        #plt.yticks([])
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
    ind = numpy.argsort(ratios)
    weight_ratios = ratios[ind]
    
    centers = (weight_ratios[:-1] + weight_ratios[1:]) / 2
    
    ratio_interval = array(ratio_interval)
    N = len(ratio_interval)
    HE = zeros(N)
    for i, x in enumerate(ratio_interval):
        P = diff(concatenate(([0], norm.cdf(log2(centers), loc=log2(x), scale=spread), [1])))
        ind = P > min_tol
        HE[i] = -numpy.sum(P[ind] * log2(P[ind]))
    
    return weight_ratios, HE

def compute_harmonic_entropy_simple_weights(numerators, denominators, ratio_interval, spread=0.01, min_tol=1e-15):
    
    # The first step is to pre-sort the ratios to speed up computation
    ratios = numerators / denominators
    ind = numpy.argsort(ratios)
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
        P /= numpy.sum(P)
        HE[i] = -numpy.sum(P * log2(P))
    
    return weight_ratios, HE


from math import gcd
import numpy
from numpy import array, zeros, ones, arange, log2, sqrt, diff, concatenate
from scipy.stats import norm
from scipy.signal import argrelextrema, detrend

def harmonic_entropy (ratios, res = 0.001, spread = 0.01, plot_entropy = True, plot_tenney = False):
    fracs, numerators, denominators = scale2frac(ratios)
    ratios = numerators / denominators
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
    #print(M)

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
    
    x_ratios = arange(1, 2, res)
    _, HE = compute_harmonic_entropy_domain_integral(ratios, x_ratios, spread=spread)
    #_, HE = compute_harmonic_entropy_simple_weights(numerators, denominators, x_ratios, spread=0.01)
    ind = argrelextrema(HE, numpy.less)
    HE_minima = (x_ratios[ind], HE[ind])
    if plot_entropy == True:  
        fig = plt.figure(figsize=(10, 4), dpi=150)
        ax = fig.add_subplot(111)
        # ax.plot(weight_ratios, log2(pdf))
        ax.plot(x_ratios, HE)
        # ax.plot(x_ratios, HE_simple)
        ax.scatter(HE_minima[0], HE_minima[1], color='k', s=4)
        ax.set_xlim(1, 2)
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
        amps_temp = np.flip(amps_temps)
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


'''BIOTUNER 2D'''


def surrogate_signal(data, surr_type = 'pink', low_cut = 0.5, high_cut = 150, sf = 1000):
    if surr_type == 'AAFT':
        indexes = [x for x in range(len(data))]
        data = np.stack((data, indexes))
        data = AAFT_surrogates(Surrogates, data)
        data = data[0]
        data = butter_bandpass_filter(data, low_cut, high_cut, sf, 4)
    if surr_type == 'phase':
        len_data = len(data)
        data = phaseScrambleTS(data)
        data = butter_bandpass_filter(data[0:len_data], low_cut, high_cut, sf, 4)
    if surr_type == 'shuffle':
        np.random.shuffle(data)
    if surr_type == 'white':
        beta = 0 
    if surr_type == 'pink':
        beta  = 1
    if surr_type == 'brown':
        beta = 2
    if surr_type == 'blue':
        beta  = -1
    if surr_type == 'white' or surr_type == 'pink' or surr_type == 'brown' or surr_type == 'blue':
        data = cn.powerlaw_psd_gaussian(beta, len(data))
        data = butter_bandpass_filter(data, low_cut, high_cut, sf, 4)
    return data
    
def surrogate_signal_matrices(data, surr_type = 'pink', low_cut = 0.5, high_cut = 150, sf = 1000):
    if np.ndim(data) == 2:
        for i in range(len(data)):
            data[i] = surrogate_signal(data[i], surr_type = surr_type, low_cut = low_cut, high_cut = high_cut, sf = sf)
    if np.ndim(data) == 3:
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j] = surrogate_signal(data[i][j], surr_type = surr_type, low_cut = low_cut, high_cut = high_cut, sf = sf)
    return data
def compute_peaks_matrices (data, peaks_function = 'EMD', precision = 0.25, sf = 1000, max_freq = 80, save = True, suffix = 'default'):
    if np.ndim(data) == 2:
        peaks_tot = []
        amps_tot = []
        for i in range(len(data)):
            peaks, amps = compute_peaks_ts(data[i], peaks_function = peaks_function, FREQ_BANDS = None, precision = precision, sf = sf, max_freq = max_freq) 
            peaks_tot.append(peaks)
            amps_tot.append(amps)
        peaks_total = np.array(peaks_tot)
        amps_total = np.array(amps_tot)
        if save == True:
            np.save('peaks_{}_{}'.format(peaks_function, suffix), peaks_total)
            np.save('amps_{}_{}'.format(peaks_function, suffix), amps_total)
    if np.ndim(data) == 3:
        peaks_tot = []
        amps_tot = []
        for i in range(len(data)):
            peaks_temp = []
            amps_temp = []
            for j in range(len(data[i])):
                peaks, amps = compute_peaks_ts(data[i][j], peaks_function = peaks_function, FREQ_BANDS = None, precision = precision, sf = sf, max_freq = max_freq)
                peaks_temp.append(peaks)
                amps_temp.append(amps)
            peaks_tot.append(peaks_temp)
            amps_tot.append(amps_temp)
        peaks_total = np.array(peaks_tot)
        amps_total = np.array(amps_tot)
        if save == True:
            np.save('peaks_{}_{}'.format(peaks_function, suffix), peaks_total)
            np.save('amps_{}_{}'.format(peaks_function, suffix), amps_total)
    return peaks_total, amps_total

def compute_peaks_surrogates (data, conditions, peaks_function = 'EMD', precision = 0.25, sf = 1000, max_freq = 80, low_cut = 0.5, high_cut = 150, save = False):
    peaks_tot = []
    amps_tot = []
    for e, c in enumerate(conditions):
        print('Condition (', e+1, 'of', len(conditions), '):', c)
        if c == 'og_data':
            peaks, amps = compute_peaks_matrices(data, peaks_function, precision, sf, max_freq)
        if c != 'og_data':
            data = surrogate_signal_matrices(data, surr_type = c, low_cut = low_cut, high_cut = high_cut, sf = sf)
            peaks, amps = compute_peaks_matrices(data, peaks_function, precision, sf, max_freq, save = False, suffix = 'default')
        peaks_tot.append(peaks)
        amps_tot.append(amps)
    peaks_total = np.array(peaks_tot)
    amps_total = np.array(amps_tot)
        
    return peaks_total, amps_total

def peaks_to_metrics_matrices (peaks):
    cons = []
    euler = []
    tenney = []
    harm_fit = []
    if np.ndim(peaks) == 3:
        for i in range(len(peaks)):
            cons_temp = []
            euler_temp = []
            tenney_temp = []
            harm_fit_temp = []
            for j in range(len(peaks[i])):
                metrics, metrics_list = peaks_to_metrics(peaks[i][j], 10)
                cons_temp.append(metrics_list[0])
                euler_temp.append(metrics_list[1])
                tenney_temp.append(metrics_list[2])
                harm_fit_temp.append(metrics_list[3])
            cons.append(cons_temp)
            euler.append(euler_temp)
            tenney.append(tenney_temp)
            harm_fit.append(harm_fit_temp)
    return np.array([cons, euler, tenney, harm_fit])


'''OLD sanity_check_code'''

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
            harm_fit = harmonic_fit(peaks[t][ch], 50, 0.1)
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


