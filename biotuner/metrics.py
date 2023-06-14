import numpy as np
from fractions import Fraction
import pytuning
import biotuner
from pytuning.utilities import normalize_interval
from numpy import log2
import sympy as sp
from biotuner.biotuner_utils import scale2frac, Print3Smallest, compute_peak_ratios, string_to_list
import itertools
import seaborn as sbn
import matplotlib.pyplot as plt
from itertools import combinations

"""PEAKS METRICS"""

def compute_consonance(ratio, limit=1000):
    """
    Compute metric of consonance from a single ratio of frequencies in the form (a+b)/(a*b).

    Parameters
    ----------
    ratio : float
        The ratio of frequencies.
    limit : int, optional (default=1000)
        The maximum value of the denominator of the fraction representing the ratio.

    Returns
    -------
    cons : float
        The consonance value.
        
    Examples
    --------
    >>> compute_consonance(3/2, limit=1000)
    0.8333333333333334
    >>> compute_consonance(16/9, limit=1000)
    0.1736111111111111
    """
    ratio = Fraction(float(ratio)).limit_denominator(limit)
    a = ratio.numerator + ratio.denominator
    b = ratio.numerator * ratio.denominator
    cons = a / b
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
    euler : int
        Euler Gradus Suavitatis.

    Examples
    --------
    >>> peaks = [3, 7, 13, 19]
    >>> euler(*peaks)
    39
    >>> peaks = [3, 9, 11, 27]
    >>> euler(*peaks)
    17
    """
    factors = biotuner.biotuner_utils.prime_factors(
        biotuner.biotuner_utils.lcm(*biotuner.biotuner_utils.reduced_form(*numbers))
    )
    return 1 + sum(p - 1 for p in factors)


def tenneyHeight(peaks, avg=True):
    """
    Tenney Height is a measure of inharmonicity calculated on
    two frequencies (a/b) reduced in their simplest form.
    It can also be called the log product complexity of a given interval.
    This function computes the Tenney Height pairwise across a list of peaks.
    Higher values represents higher dissonance.

    Parameters
    ----------
    peaks : List (float)
        frequencies
    avg : bool (default=True)
        When set to True, all tenney heights are averaged.

    Returns
    -------
    tenney : float
        Tenney Height
        
    Examples
    --------
    >>> peaks = [3, 7, 13, 19]
    >>> tenneyHeight(peaks)
    6.170342327181719
    >>> peaks = [3, 9, 11, 27]
    >>> tenneyHeight(peaks)
    4.371319977187242
    
    """
    pairs = biotuner.biotuner_utils.getPairs(peaks)
    pairs
    tenney = []
    for p in pairs:
        try:
            frac = Fraction(p[0] / p[1]).limit_denominator(1000)
        except ZeroDivisionError:
            p[1] = 0.01
            frac = Fraction(p[0] / p[1]).limit_denominator(1000)
        x = frac.numerator
        y = frac.denominator
        tenney.append(log2(x * y))
    if avg is True:
        tenney = np.average(tenney)
    return tenney


def metric_denom(ratio):
    """
    Function that computes the denominator of the normalized ratio.
    Higher value represents higher dissoance.

    Parameters
    ----------
    ratio : float
        frequency ratio

    Returns
    -------
    y : float
        denominator of the normalized ratio
        
    Examples
    --------
    >>> metric_denom(1.50)
    2
    >>> metric_denom(1.51)
    100
    """
    ratio = sp.Rational(ratio).limit_denominator(10000)
    normalized_degree = normalize_interval(ratio)
    y = int(sp.fraction(normalized_degree)[1])
    return y


def dyad_similarity(ratio):
    """
    This function computes the similarity between a dyad of frequencies
    and the natural harmonic series. Higher value represents higher harmonicity.
    Implemented from Gill and Purves (2009)

    Parameters
    ----------
    ratio : float
        frequency ratio

    Returns
    -------
    z : float
        dyad similarity
        
    Examples
    --------
    >>> dyad_similarity(3/2)
    66.66666666666666
    >>> dyad_similarity(16/9)
    16.666666666666664
    """
    frac = Fraction(float(ratio)).limit_denominator(1000)
    x = frac.numerator
    y = frac.denominator
    z = ((x + y - 1) / (x * y)) * 100
    return z

def peaks_to_harmsim(peaks):
    # check if peaks is string
    if isinstance(peaks, str):
        peaks = string_to_list(peaks)
    # check if list is empty
    if len(peaks) == 0:
        return np.nan
    else:
        ratios = compute_peak_ratios(peaks)
        return ratios2harmsim(ratios)
    
"""TUNING METRICS"""


def ratios2harmsim(ratios):
    """
    Metric of harmonic similarity represents the degree of similarity
    between a tuning and the natural harmonic series.
    This function uses the dyad_similarity function on a set of ratios.
    Implemented from Gill and Purves (2009)

    Parameters
    ----------
    ratios : List (float)
        list of frequency ratios (forming a tuning)

    Returns
    -------
    similarity : List (float)
        list of percentage of similarity for each ratios
        
    Examples
    --------
    >>> ratios = [3/2, 4/3, 16/9]
    >>> ratios2harmsim(ratios)
    array([66.66666667, 50.        , 16.66666667])
    
    References
    ----------
    Gill, K. Z., & Purves, D. (2009). A biological rationale for musical
    consonance. Proceedings of the National Academy of Sciences, 106(29),
    12174-12179.
    """
    fracs = []
    for r in ratios:
        fracs.append(Fraction(r).limit_denominator(1000))
    sims = []
    for f in fracs:
        sims.append(dyad_similarity(f.numerator / f.denominator))
    similarity = np.array(sims)
    return similarity


def tuning_cons_matrix(tuning, function, ratio_type="pos_harm"):
    """
    This function gives a tuning metric corresponding to the averaged metric
    for each pairs of ratios

    Parameters
    ----------
    tuning : List (float)
        List of tuning steps (classically between 1 (unison) and 2 (octave))
    function : function
        {'dyad_similarity', 'compute_consonance', 'metric_denom'}
    ratio_type : str (default='pos_harm')
        choice:
            - 'pos_harm' : a/b when a>b\n
            - 'sub_harm' : a/b when a<b\n
            - 'all': pos_harm + sub_harm\n

    Returns
    -------
    metric_values : List
        list of the size of input corresponding to the averaged harmonicity between a step
        and all other steps.
    metric_avg : float
        metric value averaged across all steps
        
    Examples
    --------
    >>> tuning = [1, 1.13, 1.25, 1.33, 1.5, 1.67, 1.75, 1.89]
    >>> function = dyad_similarity
    >>> tuning_cons_matrix(tuning, function, ratio_type="pos_harm")
    ([nan,
    1.8761061946902655,
    14.517994100294985,
    8.079064918934504,
    15.143364606205779,
    10.567105885817467,
    12.66150677394233,
    10.398481038234042],
    10.398481038234042)
    
    >>> tuning = [1, 1.13, 1.25, 1.33, 1.5, 1.67, 1.75, 1.89]
    >>> function = compute_consonance
    >>> tuning_cons_matrix(tuning, function, ratio_type="pos_harm")
    ([nan,
    0.018849557522123892,
    0.1618997050147493,
    0.08918417725730254,
    0.17648067513917537,
    0.1223854594889003,
    0.1428532247343695,
    0.11630349967694496],
    0.11630349967694496)
    """
    metric_values = []
    metric_values_per_step = []
    for index1 in range(len(tuning)):
        for index2 in range(len(tuning)):
            if tuning[index1] != tuning[index2]:  # not include the diagonale
                if ratio_type == "pos_harm":
                    if tuning[index1] > tuning[index2]:
                        entry = tuning[index1] / tuning[index2]
                        metric_values.append(function(entry))
                elif ratio_type == "sub_harm":
                    if tuning[index1] < tuning[index2]:
                        entry = tuning[index1] / tuning[index2]
                        metric_values.append(function(entry))
                elif ratio_type == "all":
                    entry = tuning[index1] / tuning[index2]
                    metric_values.append(function(entry))
        metric_values_per_step.append(np.average(metric_values))
    metric_avg = np.average(metric_values)
    return metric_values_per_step, metric_avg


def tuning_to_metrics(tuning, maxdenom=1000):
    """
    This function computes the tuning metrics of the PyTuning library
    (https://pytuning.readthedocs.io/en/0.7.2/metrics.html)
    and other tuning metrics

    Parameters
    ----------
    tuning : List (float)
        List of ratios corresponding to tuning steps
    maxdenom : int (default=1000)
        Maximum denominator of the fraction representing each tuning step.

    Returns
    ----------
    tuning_metrics : dictionary
        keys correspond to metrics names
    tuning_metrics_list : List (float)
        List of values corresponding to all computed metrics
        (in the same order as dictionary)
        
    Examples
    --------
    >>> tuning = [1, 1.13, 1.25, 1.33, 1.5, 1.67, 1.75, 1.89]
    >>> tuning_to_metrics(tuning, maxdenom=1000)
    {'sum_p_q': 1029,
    'sum_distinct_intervals': 56,
    'metric_3': 20.6720768749949,
    'sum_p_q_for_all_intervals': 10794,
    'sum_q_for_all_intervals': 5244,
    'harm_sim': 31.14,
    'matrix_harm_sim': 10.398481038234042,
    'matrix_cons': 0.11630349967694496,
    'matrix_denom': 83.64285714285714}
    """
    tuning_frac, num, denom = scale2frac(tuning, maxdenom=maxdenom)
    tuning_metrics = pytuning.metrics.all_metrics(tuning_frac)
    tuning_metrics["harm_sim"] = np.round(np.average(ratios2harmsim(tuning)), 2)
    _, tuning_metrics["matrix_harm_sim"] = tuning_cons_matrix(tuning,
                                                              dyad_similarity)
    _, tuning_metrics["matrix_cons"] = tuning_cons_matrix(tuning,
                                                          compute_consonance)
    _, tuning_metrics["matrix_denom"] = tuning_cons_matrix(tuning,
                                                           metric_denom)
    return tuning_metrics


def timepoint_consonance(data,
                         method="cons",
                         limit=0.2,
                         min_notes=3,
                         graph=False):

    """
    Function that keeps moments of consonance
    from multiple time series of peak frequencies.
    Can be used with the :func:`biotuner.biotuner_object.compute_spectromorph` function
    to compute timepoint consonance across the time series of spectral centroïd (or other
    spectromorphological metrics) derived from each Intrinsic Mode Function (IMF) using
    Empirical Mode Decomposition. This function can also be used of the instantaneous frequencies
    associated with each IMF.

    Parameters
    ----------
    data : List of lists (float)
        Axis 0 represents moments in time
        Axis 1 represents the sets of frequencies
    method : str, default='cons'
            'cons': 
                will compute pairwise consonance between
                frequency peaks in the form of (a+b)/(a*b)
            'euler': 
                will compute Euler's gradus suavitatis
            'harmsim':
                will compute harmonic similarity using dyad_similarity function.
    limit : float, default=0.2
        limit of consonance under which the set of frequencies are not retained
            
            When method = 'cons'
                --> See :func:`biotuner.metrics.compute_consonance` to refer to consonance values to common intervals
            When method = 'euler'
                Major (4:5:6) = 9\n
                Minor (10:12:15) = 9\n
                Major 7th (8:10:12:15) = 10\n
                Minor 7th (10:12:15:18) = 11\n
                Diminish (20:24:29) = 38\n
    min_notes : int, default=3
        Minimal number of consonant frequencies in the chords.\n
        Only relevant when method is set to 'cons'.

    Returns
    -------
    chords : List of lists (float)
        Axis 0 represents moments in time
        Axis 1 represents the sets of consonant frequencies
    positions : List (int)
        positions on Axis 0
        
    Examples
    --------
    >>> 'USING RANDOM DATA'
    >>> # Set the number of time series and the number of samples per time series
    >>> n_time_series = 5
    >>> n_samples = 1000
    >>> 
    >>> # Define the frequency range
    >>> min_freq = 1
    >>> max_freq = 10
    >>> precision = 0.1
    >>> 
    >>> # Generate the random time series
    >>> rand_integers = np.random.randint(min_freq * 10 / precision, max_freq * 10 / precision, size=(n_time_series, n_samples))
    >>> time_series = rand_integers / 10
    >>> 
    >>> tc, _ = timepoint_consonance(time_series,
                                     method="cons",
                                     limit=0.2,
                                     min_notes=3,
                                     graph=False)
    >>> tc
    [[23.1, 84.7, 74.8, 81.6],
    [30.0, 75.0, 42.0],
    [10.0, 24.0, 32.5],
    [93.6, 83.2, 33.8],
    [67.6, 91.0, 18.2],
    [72.8, 11.7, 49.5, 89.1],
    [95.4, 53.0, 52.0, 64.0],
    [37.0, 44.4, 37.8, 69.3],
    [19.8, 50.4, 32.4],
    [21.5, 43.0, 17.2],
    [15.0, 54.0, 29.0, 92.8],
    [55.9, 30.1, 99.0, 40.5],
    [55.8, 21.7, 99.2],
    [30.5, 22.4, 12.6, 91.5]]
    
    >>> 'USING THE BIOTUNER OBJECT'
    >>> from biotuner.biotuner_object import compute_biotuner
    >>> # Load data
    >>> data = np.load('data_examples/EEG_pareidolia/parei_occi_L.npy')
    >>> # Keep a single time series.
    >>> data = data[0]
    >>> # Initialize biotuner object
    >>> bt = compute_biotuner(sf=1200, peaks_function='EMD')
    >>> # Extract spectral peaks
    >>> bt.peaks_extraction(data)
    >>> # Compute timepoint consonance using Spectral Centroid on IMFs.
    >>> bt.compute_spectromorph(
            method="SpectralCentroid",
            overlap=1,
            comp_chords=True,
            min_notes=4,
            cons_limit=0.2,
            cons_chord_method="cons",
            graph=False,
        )
    >>> bt.spectro_chords
    [[15.47, 6.63, 31.68, 2.42],
    [5.72, 12.71, 26.64, 2.59],
    [15.31, 6.38, 2.53, 64.77],
    [67.59, 2.7, 33.79, 16.8]]
    
    """
    def process_peaks(method, peaks, limit, min_notes):
        if method == "cons":
            cons, _, peaks_cons, _ = consonance_peaks(peaks, limit)
            if len(set(peaks_cons)) >= min_notes:
                return peaks_cons
        elif method == 'harmsim':
            ratios = compute_peak_ratios(peaks, rebound=False)
            if np.mean(ratios2harmsim(ratios)) > limit:
                return peaks
        elif method == "euler":
            peaks_ = [int(round(p, 2) * 100) for p in peaks]
            eul = euler(*peaks_)
            if eul < limit:
                return list(peaks)
        return []

    # Generate labels using list comprehension
    labels = [f"EMD{i + 1}" for i in range(len(data))]

    data = np.round(data, 2)
    data = np.moveaxis(data, 0, 1)
    out = []
    positions = []
    for count, peaks in enumerate(data):
        peaks = [x for x in peaks if x >= 0.1]
        if len(peaks) == 0:
            result = []
        else:
            result = process_peaks(method, peaks, limit, min_notes)
            out.append(result)
        if result:
            positions.append(count)

    out = [x for x in out if x]
    out = list(out for out, _ in itertools.groupby(out))
    chords = [x for x in out if len(x) >= min_notes]
    chords = [e[::-1] for e in chords]

    if graph:
        fig, ax = plt.subplots()
        for i, label in enumerate(labels):
            ax.plot(data[:, i], label=label)

        ax.set_xlabel("Time Windows")
        ax.set_yscale("log")
        plt.legend(
            scatterpoints=1,
            frameon=True,
            labelspacing=1,
            title="EMDs",
            loc="best",
            labels=labels,
        )
        for xc in positions:
            plt.axvline(x=xc, c="black", linestyle="dotted")
        plt.show()

    return chords, positions


def compute_subharmonics(chord, n_harmonics, delta_lim):
    """
    Compute subharmonics of a chord and find the common subharmonics within a given delta limit.

    Parameters
    ----------
    chord : list of int
        A list of integers representing the chord.
    n_harmonics : int
        The number of harmonics to compute.
    delta_lim : float
        The limit of delta between two subharmonics to be considered common.
        This value is in milliseconds (ms).

    Returns
    -------
    subharms : list of list of float
        A list of lists of subharmonics for each element in the chord.
    common_subs : list of list of float
        A list of lists of common subharmonics within the delta limit.
    delta_t : list of float
        A list of delta values for the common subharmonics.
        
    Examples
    --------
    >>> chord = [3, 5, 7]
    >>> subharms, common_subharms, delta_t = compute_subharmonics(chord, 5, 20)
    >>> print('subharms', subharms)
    >>> print('common_subharms', common_subharms)
    >>> print('delta_t', delta_t)
    subharms [[333.3333333333333, 666.6666666666666, 1000.0, 1333.3333333333333, 1666.6666666666667],
    [200.0, 400.0, 600.0, 800.0, 1000.0],
    [142.85714285714286, 285.7142857142857, 428.57142857142856, 571.4285714285714, 714.2857142857143]] 
    common_subharms [[1000.0, 1000.0]] 
    delta_t [0.0]
    """
    subharms = []
    subharms_tot = []
    delta_t = []
    common_subs = []
    for i in chord:
        s_ = []
        for j in range(1, n_harmonics+1):
            s_.append(1000/(i/j))
        subharms.append(s_)
    for i in range(len(chord)):
        for j in range(i+1, len(chord)):
            for s1 in subharms[i]:
                for s2 in subharms[j]:
                    if np.abs(s1-s2) < delta_lim:
                        delta_t.append(np.abs(np.min([s1-s2])))
                        common_subs.append([s1, s2])
    return subharms, common_subs, delta_t


def compute_subharmonic_tension(chord, n_harmonics, delta_lim, min_notes=2):
    """
    Computes the subharmonic tension for a set of frequencies,
    based on the common subharmonics of a minimum of 2 or 3 frequencies.
    This metric has been adapted from Chan et al. (2019).

    Parameters
    ----------
    chord : numpy array, shape (n,)
        Array containing the frequencies to compute subharmonic tension on.
    n_harmonics : int
        Number of subharmonics to compute for each frequency.
    delta_lim : float
        Maximal distance between subharmonics of different frequencies
        to consider them as common subharmonics.
    min_notes : int, {2, 3}, default=2
        Minimal number of notes to consider common subharmonics.

    Returns
    -------
    common_subs : numpy array, shape (m,)
        Array containing the common subharmonics.
    delta_t : numpy array, shape (m,)
        Array containing the subharmonic distances.
    subharm_tension : float or str
        The subharmonic tension value, calculated as the average of the product
        of the subharmonic distance and the subharmonic frequency over all
        subharmonic pairs. Returns "NaN" if no valid subharmonic pairs are found.
    harm_temp : numpy array, shape (m,)
        Array containing the subharmonic harmonic values.

    Examples
    --------
    >>> chord = [3, 5, 7]
    >>> _, _, subharm_tension, _ = compute_subharmonic_tension(chord, 5, 20, min_notes=2)
    >>> subharm_tension
    [0.0]
    >>> chord = [31, 51, 71]
    >>> _, _, subharm_tension, _ = compute_subharmonic_tension(chord, 5, 20, min_notes=2)
    >>> subharm_tension
    [0.23539483720429924]
    
    Notes
    -----
    The subharmonic tension is a measure of the perceived stability of a musical chord.
    The subharmonic tension is calculated as the average of the product of the
    subharmonic distance and the subharmonic frequency over all subharmonic pairs.
    A subharmonic is a frequency that is an integer divisor of another frequency.
    Common subharmonics are defined as subharmonics that are shared by at least
    `min_notes` notes in the chord.
    
    References
    ----------
    Chan, P. Y., Dong, M., & Li, H. (2019). The science of harmony:A psychophysical basis
    for perceptual tensions and resolutions in music. Research, 2019.
    """

    subharms = [np.array([1000 / (i / j) for j in range(1, n_harmonics + 1)]) for i in chord]
    combi = np.array(list(itertools.product(*subharms)))
    delta_t = []
    common_subs = []
    for group in range(len(combi)):
        for subharmonic_combination in combinations(combi[group], min_notes):
            if all(np.abs(np.diff(subharmonic_combination)) < delta_lim):
                delta_t.append(np.min(np.abs(np.diff(subharmonic_combination))))
                common_subs.append(np.mean(subharmonic_combination))
    delta_temp = []
    harm_temp = []
    overall_temp = []
    subharm_tension = []
    if len(delta_t) > 0:
        try:
            for i in range(len(delta_t)):
                delta_norm = delta_t[i] / common_subs[i]
                delta_temp.append(delta_norm)
                harm_temp.append(1 / delta_norm)
                overall_temp.append((1 / common_subs[i]) * (delta_t[i]))
            try:
                subharm_tension.append(((sum(overall_temp)) / len(delta_t)))
            except ZeroDivisionError:
                subharm_tension.append("NaN")
        except IndexError:
            subharm_tension = "NaN"
    else:
        subharm_tension = "NaN"
    return common_subs, delta_t, subharm_tension, harm_temp


def compute_subharmonics_2lists(list1, list2, n_harmonics, delta_lim, c=2.1):
    """
    Compute the subharmonic tension (Chan et al., 2019) for pairs of frequencies
    from two different lists, based on the common subharmonics of each pair.

    Parameters
    ----------
    list1 : list of floats
        Values of the first set of frequencies to compute subharmonic tension on.
    list2 : list of floats
        Values of the second set of frequencies to compute subharmonic tension on.
    n_harmonics : int
        Number of subharmonics to compute for each frequency.
    delta_lim : float
        Maximal distance between subharmonics of different frequencies
        to consider them as common subharmonics.
    c : float, default=2.1
        Constant parameter for computing subharmonic tension.

    Returns
    -------
    tuple
        common_subs : list of lists (floats)
            List of common subharmonics found for each pair of frequencies.
        delta_t : list of lists (floats)
            List of the smallest differences between common subharmonics found
            for each pair of frequencies.
        sub_tension_final : float
            The overall subharmonic tension value computed by averaging across
            all pairs of frequencies.
        harm_temp : list of floats
            List of harmonic tensions for each subharmonic tension computed.
        pair_melody : list of floats
            List containing the pair of frequency with the lowest subharmonic tension.
            Could be used in transitional harmony to derive subharmonic melodies.
            
    Examples
    --------
    >>> list1 = [5, 9]
    >>> list2 = [13, 20, 7]
    >>> n_harms = 5
    >>> delta_lim = 30
    >>> _, _, sub_tension, _, pair_melody = compute_subharmonics_2lists(list1, list2, n_harms, delta_lim, c=2.1)
    >>> subtension, pair_melody
    (0.05213398154647142,(5, 20))
    """
    list_ = [list1, list2]
    combinations = [p for p in itertools.product(*list_)]
    sub_tension_final = []
    #  Compute subharmonic tension for each pairs of peaks
    #  that belong to different lists.
    subharm_pairs = []
    common_subs_tot = []
    delta_t_tot = []
    for pair in combinations:
        subharms = []
        delta_t = []
        common_subs = []
        #  Compute subharmonics for each fundamental frequency
        for i in pair:
            s_ = []
            for j in range(1, n_harmonics+1):
                s_.append(1000/(i/j))
            subharms.append(s_)
        #  Create pairwise combinations of subharmonics from the two lists.
        combi = np.array(list(itertools.product(subharms[0], subharms[1])))
        #  Iterate through all pairs to identify common subharmonics
        for t in combi:
            s1 = t[0]
            s2 = t[1]
            if np.abs(s1-s2) < delta_lim:
                delta_t_ = np.abs(s1-s2)
                common_subs_ = np.mean([s1, s2])
                if delta_t_ not in delta_t and common_subs_ not in common_subs:
                    delta_t.append(delta_t_)
                    common_subs.append(common_subs_)
        delta_temp = []
        harm_temp = []
        overall_temp = []
        subharm_tension = []
        c = c
        if len(delta_t) > 0:
            subharm_pairs.append(pair)
            #  Iterate through common subharmonics to compute
            #  subarmonic tension associated to a single pair of peaks.
            for i in range(len(delta_t)):
                delta_norm = delta_t[i]/common_subs[i]
                delta_temp.append(delta_norm)
                harm_temp.append(1/delta_norm)
                overall_temp.append((1/common_subs[i])*(delta_t[i]))
            subharm_tension.append(((sum(overall_temp))/len(delta_t)))
            #print('subharm_tension', np.average(subharm_tension))
            sub_tension_final.append(np.average(subharm_tension))
            common_subs_tot.append(common_subs)
            delta_t_tot.append(delta_t)
    #  Compute the overall subharmonic tension by averaging across
    #  pairs of frequencies.
    #mins, low_sub_idx = Print3Smallest(sub_tension_final)
    low_sub_idx = sub_tension_final.index(np.min(sub_tension_final))
    pair_melody = subharm_pairs[low_sub_idx]
    sub_tension_final = np.average(sub_tension_final)
    return common_subs_tot, delta_t_tot, sub_tension_final, harm_temp, pair_melody

def consonant_ratios(data,
                     limit,
                     sub=False,
                     input_type="peaks",
                     metric="cons",
                     set_rebound=True):
    """
    Function that computes integer ratios from peaks with higher consonance

    Parameters
    ----------
    data : List (float)
        Data can whether be frequency values or frequency ratios
    limit : float
        minimum consonance value to keep associated pairs of peaks
    sub : boolean, default=False
        When set to True, include ratios a/b when a < b.
    input_type : str, default='peaks'
        Choose between:
        - 'peaks'
        - 'ratios'
    metric : str, default='cons'
        Choose between:
        - 'cons'
        - 'harmsim'
    set_rebound : bool, default=True
        Defines if the ratios are rebounded between 1 and 2.
        Only valid when input_type = 'peaks'.


    Returns
    -------
    cons_ratios : List (float)
        list of consonant ratios
    consonance : List (float)
        list of associated consonance values
        
    Examples
    --------
    >>> ratios = [1, 1.25, 1.34, 1.5, 1.67, 1.86]
    >>> cons_ratios, metrics = consonant_ratios(ratios,
    >>>                                         0.2,
    >>>                                         sub=False,
    >>>                                         input_type="ratios",
    >>>                                         metric="cons")
    >>> cons_ratios, metrics
    (array([1.  , 1.25, 1.5 ]), [2.0, 0.45, 0.8333333333333334])
    
    >>> freqs = [3, 4.5, 11, 17]
    >>> ratios, metrics = consonant_ratios(freqs,
    >>>                                     0.2,
    >>>                                     sub=False,
    >>>                                     input_type="ratios",
    >>>                                     metric="cons")
    >>> ratios, metrics
    (array([1.222, 1.5  , 1.833]),
     [0.20202020202020202, 0.8333333333333334, 0.25757575757575757])
    """
    consonance_ = []
    ratios2keep = []
    if input_type == "peaks":
        ratios = compute_peak_ratios(data, sub=sub, rebound=set_rebound)
        #print(ratios)
    if input_type == "ratios":
        ratios = data
    for ratio in ratios:
        if metric == "cons":
            cons_ = compute_consonance(ratio)
        if metric == "harmsim":
            cons_ = dyad_similarity(ratio)
        if cons_ > limit:
            consonance_.append(cons_)
            ratios2keep.append(ratio)
    ratios2keep = np.array(np.round(ratios2keep, 3))
    cons_ratios = np.sort(list(set(ratios2keep)))
    consonance = np.array(consonance_)
    consonance = [i for i in consonance if i]
    return cons_ratios, consonance


def consonance_peaks(peaks, limit, limit_pairs=True):
    """
    This function computes consonance (for a given ratio a/b, when a < 2b),
    consonance corresponds to (a+b)/(a*b)) between peaks.

    Parameters
    ----------
    peaks : list of floats
        Peaks represent local maximum in a spectrum.
    limit : float
        Minimum consonance value to keep associated pairs of peaks.
        
            Comparisons with familiar ratios:  
            Unison-frequency ratio 1:1 yields a value of 2\n
            Octave-frequency ratio 2:1 yields a value of 1.5\n
            Perfect 5th-frequency ratio 3:2 yields a value of 0.833\n
            Perfect 4th-frequency ratio 4:3 yields a value of 0.583\n  
            Major 6th-frequency ratio 5:3 yields a value of 0.533\n  
            Major 3rd-frequency ratio 5:4 yields a value of 0.45\n  
            Minor 3rd-frequency ratio 5:6 yields a value of 0.366\n  
            Minor 6th-frequency ratio 5:8 yields a value of 0.325\n  
            Major 2nd-frequency ratio 8:9 yields a value of 0.236\n  
            Major 7th-frequency ratio 8:15 yields a value of 0.192\n  
            Minor 7th-frequency ratio 9:16 yields a value of 0.174\n  
            Minor 2nd-frequency ratio 15:16 yields a value of 0.129\n
            
    limit_pairs : bool, default=True
        Whether to compute consonance only for ratios where a > b.
        If False, also use ratios where a < b by dividing b iteratively.

    Returns
    -------
    consonance : list of floats
        Consonance scores for each pair of consonant peaks.
    cons_pairs : list of lists of floats
        List of lists of each pair of consonant peaks.
    cons_peaks : list of floats
        List of consonant peaks (no duplicates).
    cons_tot : float
        Averaged consonance value for each pair of peaks.

    Examples
    --------
    >>> peaks = [3, 9, 11, 21]
    >>> consonance, cons_pairs, cons_peaks, cons_tot = consonance_peaks(peaks, limit=0.5, limit_pairs=True)
    >>> consonance, cons_pairs, cons_peaks, cons_tot
    ([1.3333333333333333, 1.1428571428571428],
     [[3, 9], [3, 21]],
     [9, 3, 21],
     0.31024531024531027)
    """
    consonance_ = []
    peaks2keep = []
    cons_tot = []
    for p1, p2 in itertools.permutations(peaks, 2):
        if limit_pairs:
            if p1 <= p2:
                continue
        else:
            while p2 >= p1:
                p2 /= 2
            if p1 < 0.1 or p2 < 0.1:
                continue
        cons_ = compute_consonance(p2 / p1)
        if cons_ < 1:
            cons_tot.append(cons_)
        if cons_ < limit or cons_ == 2:
            cons_ = None
            p2 = None
            p1 = None
        if p2 is not None:
            peaks2keep.append([p2, p1])
        consonance_.append(cons_)
    cons_pairs = [x for x in peaks2keep if x]
    consonance = [i for i in consonance_ if i]
    cons_peaks = list(itertools.chain(*cons_pairs))
    cons_peaks = [np.round(c, 2) for c in cons_peaks]
    cons_peaks = list(set(cons_peaks))
    return consonance, cons_pairs, cons_peaks, np.average(cons_tot)


def spectral_flatness(harmonicity_values):
    """
    Calculate the spectral flatness of a signal.

    Parameters
    ----------
    harmonicity_values : ndarray
        Harmonicity values of the signal.

    Returns
    -------
    float
        Spectral flatness of the signal.
    """
    geometric_mean = np.exp(np.mean(np.log(harmonicity_values + np.finfo(float).eps)))
    arithmetic_mean = np.mean(harmonicity_values)
    return geometric_mean / arithmetic_mean


def spectral_entropy(harmonicity_values):
    """
    Calculate the spectral entropy of a signal.

    Parameters
    ----------
    harmonicity_values : ndarray
        Harmonicity values of the signal.

    Returns
    -------
    float
        Spectral entropy of the signal.
    """
    normalized_harmonicity = harmonicity_values / np.sum(harmonicity_values)
    entropy = -np.sum(normalized_harmonicity * np.log2(normalized_harmonicity + np.finfo(float).eps))
    return entropy

def spectral_spread(freqs, psd):
    """
    Calculate the spectral spread of a signal.

    Parameters
    ----------
    freqs : ndarray
        Frequencies for which the PSD is computed.
    psd : ndarray
        Power spectral density of the signal.

    Returns
    -------
    float
        Spectral spread of the signal.
    """
    # Normalize PSD
    psd_norm = psd / np.sum(psd)

    # Calculate spectral centroid
    spectral_centroid = np.sum(freqs * psd_norm)

    # Calculate spectral spread
    spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd_norm))

    return spectral_spread

def higuchi_fd(data, kmax):
    """ Compute Higuchi Fractal Dimension of a time series. kmax is max. delay (should be < len(data)/3) """
    L = []
    x = []
    N = len(data)
    for k in range(1,kmax):
        Lk = 0
        for m in range(0,k):
            Lmk = 0
            for i in range(1,int((N-m)/k)):
                Lmk += abs(data[m+i*k] - data[m+i*k-k])
            Lmk = Lmk*(N - 1)/(((N - m)/ k)* k)
            Lk += Lmk
        L.append(np.log(Lk/(m+1)))
        x.append([np.log(1.0/ k), 1])
    (p, _, _, _) = np.linalg.lstsq(x, L, rcond=None)  # Use rcond=None to silence future warning
    return p[0]