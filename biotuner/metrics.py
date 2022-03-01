import numpy as np
from fractions import Fraction
import pytuning
from pytuning.utilities import normalize_interval
from biotuner_utils import reduced_form, lcm, prime_factors, getPairs
from numpy import log2
import sympy as sp
from biotuner_utils import scale2frac


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
    factors = prime_factors(lcm(*reduced_form(*numbers)))
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
        choice:
        -dyad_similarity
        -consonance
        -metric_denom
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
    for index1 in range(len()):
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
    tuning_frac, num, denom = scale2frac(tuning, maxdenom=maxdenom)
    tuning_metrics = pytuning.metrics.all_metrics(tuning_frac)
    tuning_metrics['harm_sim'] = np.round(np.average(ratios2harmsim(tuning)), 2)
    tuning_metrics['matrix_harm_sim'] = tuning_cons_matrix(tuning, dyad_similarity)
    tuning_metrics['matrix_cons'] = tuning_cons_matrix(tuning, compute_consonance)
    tuning_metrics['matrix_denom'] = tuning_cons_matrix(tuning, metric_denom)
    return tuning_metrics
