import numpy as np
from fractions import Fraction
import pytuning
from pytuning.utilities import normalize_interval
import biotuner.biotuner_utils
from numpy import log2
import sympy as sp
import biotuner.peaks_extension
import itertools
import seaborn as sbn
import matplotlib.pyplot as plt
from itertools import combinations

'''PEAKS METRICS'''


def compute_consonance(ratio, limit=1000):
    '''
    Compute metric of consonance from a single ratio of frequencies
    in the form (a+b)/(a*b)

    Parameters
    ----------
    ratio: float
    limit: int
        Defaults to 1000
        Maximum value of the denominator of the fraction representing the ratio

    Returns
    -------
    cons : float
        consonance value
    '''
    ratio = Fraction(float(ratio)).limit_denominator(limit)
    cons = (ratio.numerator + ratio.denominator)/(ratio.numerator * ratio.denominator)
    return cons


def euler(*numbers):
    """Euler's "gradus suavitatis" (degree of sweetness) function
    Return the "degree of sweetness" of a musical interval or chord expressed
    as a ratio of frequencies a:b:c, according to Euler's formula
    Greater values indicate more dissonance.

    Parameters
    ----------
    *numbers : List or Array of int
        Frequencies

    Returns
    -------
    int
        Euler Gradus Suavitatis.

    """
    factors = biotuner.biotuner_utils.prime_factors(biotuner.biotuner_utils.lcm(*biotuner.biotuner_utils.reduced_form(*numbers)))
    return 1 + sum(p - 1 for p in factors)


def tenneyHeight(peaks, avg=True):
    """
    Tenney Height is a measure of inharmonicity calculated on
    two frequencies (a/b) reduced in their simplest form.
    It can also be called the log product complexity of a given interval.

    Parameters
    ----------
    peaks: List (float)
        frequencies
    avg: Boolean
        Default to True
        When set to True, all tenney heights are averaged

    Returns
    -------
    tenney : float
        Tenney Height
    """
    pairs = biotuner.biotuner_utils.getPairs(peaks)
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
    if avg is True:
        tenney = np.average(tenney)
    return tenney


def metric_denom(ratio):
    '''Function that computes the denominator of the normalized ratio

    Parameters
    ----------
    ratio: float

    Returns
    -------
    y : float
        denominator of the normalized ratio
    '''
    ratio = sp.Rational(ratio).limit_denominator(10000)
    normalized_degree = normalize_interval(ratio)
    y = int(sp.fraction(normalized_degree)[1])
    return y


def dyad_similarity(ratio):
    '''
    This function computes the similarity between a dyad of frequencies
    and the natural harmonic series.

    Parameters
    ----------
    ratio: float
        frequency ratio

    Returns
    -------
    z : float
        dyad similarity
    '''
    frac = Fraction(float(ratio)).limit_denominator(1000)
    x = frac.numerator
    y = frac.denominator
    z = ((x+y-1)/(x*y))*100
    return z


'''TUNING METRICS'''


def ratios2harmsim(ratios):
    '''
    Metric of harmonic similarity represents the degree of similarity
    between a tuning and the natural harmonic series.
    Implemented from Gill and Purves (2009)

    Parameters
    ----------
    ratios: List (float)
        list of frequency ratios (forming a tuning)

    Returns
    -------
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


def tuning_cons_matrix(tuning, function, ratio_type='pos_harm'):
    '''
    This function gives a tuning metric corresponding to the averaged metric
    for each pairs of ratios

    Parameters
    ----------
    tuning: List (float)
    function: function
        {'dyad_similarity', 'consonance', 'metric_denom'}
    ratio_type: str
        Default to 'pos_harm'
        choice:
        -'pos_harm':a/b when a>b
        -'sub_harm':a/b when a<b
        -'all': pos_harm + sub_harm

    Returns
    -------
    metric_values: List
        list of the size of input
    metric_avg: float
        metric value averaged across all steps
    '''
    metric_values = []
    metric_values_per_step = []
    for index1 in range(len(tuning)):
        for index2 in range(len(tuning)):
            metric_values_temp = []
            if tuning[index1] != tuning[index2]:  # not include the diagonale
                if ratio_type == 'pos_harm':
                    if tuning[index1] > tuning[index2]:
                        entry = tuning[index1]/tuning[index2]
                        metric_values.append(function(entry))
                        metric_values_temp.append(function(entry))
                if ratio_type == 'sub_harm':
                    if tuning[index1] < tuning[index2]:
                        entry = tuning[index1]/tuning[index2]
                        metric_values.append(function(entry))
                        metric_values_temp.append(function(entry))
                if ratio_type == 'all':
                    entry = tuning[index1]/tuning[index2]
                    metric_values.append(function(entry))
                    metric_values_temp.append(function(entry))
        metric_values_per_step.append(np.average(metric_values_temp))
    metric_avg = np.average(metric_values)
    return metric_values, metric_avg


def tuning_to_metrics(tuning, maxdenom=1000):
    '''
    This function computes the tuning metrics of the PyTuning library
    (https://pytuning.readthedocs.io/en/0.7.2/metrics.html)
    and other tuning metrics

    Parameters
    ----------
    tuning: List (float)
        List of ratios corresponding to tuning steps

    Returns
    ----------
    tuning_metrics: dictionary
        keys correspond to metrics names
    tuning_metrics_list: List (float)
        List of values corresponding to all computed metrics
        (in the same order as dictionary)
    '''
    tuning_frac, num, denom = biotuner.biotuner_utils.scale2frac(tuning, maxdenom=maxdenom)
    tuning_metrics = pytuning.metrics.all_metrics(tuning_frac)
    tuning_metrics['harm_sim'] = np.round(np.average(ratios2harmsim(tuning)), 2)
    _, tuning_metrics['matrix_harm_sim'] = tuning_cons_matrix(tuning, dyad_similarity)
    _, tuning_metrics['matrix_cons'] = tuning_cons_matrix(tuning, compute_consonance)
    _, tuning_metrics['matrix_denom'] = tuning_cons_matrix(tuning, metric_denom)
    return tuning_metrics


def timepoint_consonance(data, method='cons', limit=0.2, min_notes=3,
                         graph=False):

    """
    Function that keeps moments of consonance
    from multiple time series of peak frequencies

    Parameters
    ----------
    data: List of lists (float)
        Axis 0 represents moments in time
        Axis 1 represents the sets of frequencies
    method: str
        Defaults to 'cons'
        'cons': will compute pairwise consonance between
               frequency peaks in the form of (a+b)/(a*b)
        'euler': will compute Euler's gradus suavitatis
    limit: float
        limit of consonance under which the set of frequencies are not retained
        When method = 'cons'
             --> See consonance_peaks method's doc to refer to
                 consonance values to common intervals
        When method = 'euler'
             --> Major (4:5:6) = 9
                 Minor (10:12:15) = 9
                 Major 7th (8:10:12:15) = 10
                 Minor 7th (10:12:15:18) = 11
                 Diminish (20:24:29) = 38
    min_notes: int
        minimum number of consonant frequencies in the chords.
        Only relevant when method is set to 'cons'.

    Returns
    -------
    chords: List of lists (float)
        Axis 0 represents moments in time
        Axis 1 represents the sets of consonant frequencies
    positions: List (int)
        positions on Axis 0
    """

    data = np.moveaxis(data, 0, 1)
    out = []
    positions = []
    for count, peaks in enumerate(data):
        peaks = [x for x in peaks if x >= 0]
        if method == 'cons':
            cons, b, peaks_cons, d = biotuner.peaks_extension.consonance_peaks(peaks, limit)
            out.append(peaks_cons)
            if len(list(set(peaks_cons))) >= min_notes:
                positions.append(count)
        if method == 'euler':
            peaks_ = [int(np.round(p, 2)*100) for p in peaks]
            eul = euler(*peaks_)
            if eul < limit:
                out.append(list(peaks))
                positions.append(count)
    out = [x for x in out if x != []]
    out = list(out for out, _ in itertools.groupby(out))
    chords = [x for x in out if len(x) >= min_notes]
    chords = [e[::-1] for e in chords]
    if graph is True:
        ax = sbn.lineplot(data=data[10:-10, :], dashes=False)
        ax.set(xlabel='Time Windows', ylabel=method)
        ax.set_yscale('log')
        plt.legend(scatterpoints=1, frameon=True, labelspacing=1,
                   title='EMDs', loc='best',
                   labels=['EMD1', 'EMD2', 'EMD3', 'EMD4', 'EMD5', 'EMD6'])
        for xc in positions:
            plt.axvline(x=xc, c='black', linestyle='dotted')
        plt.show()
    return chords, positions

def compute_subharmonics_5notes(chord, n_harmonics, delta_lim, c=2.1):
    subharms = []
    subharms_tot = []
    delta_t = []
    common_subs = []
    for i in chord:
        s_ = []
        for j in range(1, n_harmonics+1):
            s_.append(1000/(i/j))
        subharms.append(s_)
    
    combi = np.array(list(itertools.product(subharms[0],subharms[1],subharms[2], subharms[3], subharms[4])))
    print(combi.shape)
    for group in range(len(combi)):
        triplets = list(combinations(combi[group], 3))
        for t in triplets:
            s1 = t[0]
            s2 = t[1]
            s3 = t[2]
            if np.abs(s1-s2) < delta_lim and np.abs(s1-s3) < delta_lim and np.abs(s2-s3) < delta_lim:
                delta_t_ = np.abs(np.min([s1-s2, s1-s3, s2-s3]))
                common_subs_ = np.mean([s1, s2, s3])
                if delta_t_ not in delta_t:
                    delta_t.append(delta_t_)
                if common_subs_ not in common_subs:
                    common_subs.append(common_subs_)
    delta_temp = []
    harm_temp = []
    overall_temp = []
    subharm_tension = []
    c=c
    if len(delta_t) > 0:
        try:
            for i in range(len(delta_t)):
                delta_norm = delta_t[i]/common_subs[i]
                delta_temp.append(delta_norm)
                harm_temp.append(1/delta_norm)
                overall_temp.append((1/common_subs[i])*(delta_t[i]))
            try:
                print(overall_temp)
                #print((1/sum(overall_temp))**(1/c))
                subharm_tension.append(((sum(overall_temp))/len(delta_t)))
            except ZeroDivisionError:
                subharm_tension.append(None)
        except IndexError:
            subharm_tension = 'NaN'
    if len(delta_t) == 0:
        subharm_tension = 'NaN'
    subharm_tension
    return common_subs, delta_t, subharm_tension, harm_temp
