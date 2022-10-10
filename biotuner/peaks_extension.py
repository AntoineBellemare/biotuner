
import numpy as np
import itertools
from biotuner.biotuner_utils import rebound, compareLists
from collections import Counter
from functools import reduce
import biotuner.metrics

'''EXTENDED PEAKS from expansions
   (finding new peaks based on harmonic structure)
'''


def EEG_harmonics_mult(peaks, n_harmonics, n_oct_up=0):
    """
    Natural n_harmonics

    This function takes a list of frequency peaks as input
    and computes the desired number of harmonics
    with the formula: x, 2x, 3x ..., nx.

    Parameters
    ----------
    peaks : List (float)
        Peaks represent local maximum in a spectrum.
    n_harmonics : int
        Number of harmonics to compute.
    n_oct_up : int
        Defaults to 0. Corresponds to the number of octave
        the peaks are shifted.

    Returns
    -------
    multi_harmonics : array
        (n_peaks, n_harmonics + 1)
    """

    n_harmonics = n_harmonics + 2
    multi_harmonics = []
    for p in peaks:
        harmonics = []
        p = p * (2**n_oct_up)
        i = 1
        harm_temp = p
        while i < n_harmonics:
            harm_temp = p * i
            harmonics.append(harm_temp)
            i += 1
        multi_harmonics.append(harmonics)
    multi_harmonics = np.array(multi_harmonics)

    return multi_harmonics


def EEG_harmonics_div(peaks, n_harmonics, n_oct_up=0, mode='div'):
    """
    Natural sub-harmonics

    This function takes a list of frequency peaks as input and computes
    the desired number of harmonics with using division:

    Parameters
    ----------
    peaks : List (float)
        Peaks represent local maximum in a spectrum
    n_harmonics : int
        Number of harmonics to compute
    n_oct_up : int
        Defaults to 0. The number of octave the peaks are shifted
    mode : str
        Defaults to 'div'.
        'div': x, x/2, x/3 ..., x/n
        'div_add': x, (x+x/2), (x+x/3), ... (x+x/n)
        'div_sub': x, (x-x/2), (x-x/3), ... (x-x/n)

    Returns
    -------
    div_harmonics : array
        (n_peaks, n_harmonics + 1)
        Harmonic values in hertz.
    div_harmonics_bounded : array
        (n_peaks, n_harmonics + 1)
        Harmonic values bounded between unison (1) and octave (2).
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
            i += 1
        div_harmonics.append(harmonics)
    div_harmonics = np.array(div_harmonics)
    div_harm_bound = div_harmonics.copy()
    # Rebound the result between 1 and 2
    for i in range(len(div_harm_bound)):
        for j in range(len(div_harm_bound[i])):
            div_harm_bound[i][j] = rebound(div_harm_bound[i][j])
    return div_harmonics, div_harm_bound


def harmonic_fit(peaks, n_harm=10, bounds=1, function='mult',
                 div_mode='div', n_common_harms=5):
    """
    This function computes harmonics of a list of peaks and compares the lists
    of harmonics pairwise to find fitting between the harmonic series.

    Parameters
    ----------
    peaks : List (float)
        Spectral peaks represent local maximum in a spectrum
    n_harm : int
        Number of harmonics to compute
    bounds : int
        Minimum distance (in Hz) between two frequencies to consider a fit
    function : str
        Defaults to 'mult'.
        'mult' will use natural harmonics
        'div' will use natural sub-harmonics
    div_mode : str
        Defaults to 'div'. See EEG_harmonics_div function.
    n_common_harms : int
        minimum number of times the harmonic is found
        to be sent to most_common_harmonics output.

    Returns
    -------
    harm_fit : List
        Frequencies of the harmonics that match
    harmonics_pos : List
        Positions of the harmonics that match
    most_common_harmonics : List
        harmonics that are at least present for ''n_common_harms'' times
    matching_positions : List of lists
        Each sublist corresponds to an harmonic fit,
        the first number is the frequency and the two
        others are harmonic positions.

    """
    from itertools import combinations
    peak_bands = []
    for i in range(len(peaks)):
        peak_bands.append(i)
    if function == 'mult':
        multi_harmonics = EEG_harmonics_mult(peaks, n_harm)
    elif function == 'div':
        multi_harmonics, x = EEG_harmonics_div(peaks, n_harm, mode=div_mode)
    elif function == 'exp':
        multi_harmonics = []
        for h in range(n_harm+1):
            h += 1
            multi_harmonics.append([i**h for i in peaks])
        multi_harmonics = np.array(multi_harmonics)
        multi_harmonics = np.moveaxis(multi_harmonics, 0, 1)
    list_peaks = list(combinations(peak_bands, 2))
    harm_temp = []
    matching_positions = []
    harmonics_pos = []
    for i in range(len(list_peaks)):
        (harms,
         harm_pos,
         matching_pos,
         _) = compareLists(multi_harmonics[list_peaks[i][0]],
                           multi_harmonics[list_peaks[i][1]],
                           bounds)
        harm_temp.append(harms)
        harmonics_pos.append(harm_pos)
        if len(matching_pos) > 0:
            for j in matching_pos:
                matching_positions.append(j)
    matching_positions = [list(i) for i in matching_positions]
    harm_fit = np.array(harm_temp, dtype=object).squeeze()
    harmonics_pos = reduce(lambda x, y: x+y, harmonics_pos)
    most_common_harmonics = [h for h, h_count in Counter(harmonics_pos)
                             .most_common(n_common_harms) if h_count > 1]
    harmonics_pos = list(np.sort(list(set(harmonics_pos))))
    if len(peak_bands) > 2:
        harm_fit = list(itertools.chain.from_iterable(harm_fit))
        harm_fit = [round(num, 3) for num in harm_fit]
        harm_fit = list(dict.fromkeys(harm_fit))
        harm_fit = list(set(harm_fit))
    return harm_fit, harmonics_pos, most_common_harmonics, matching_positions


'''EXTENDED PEAKS from restrictions'''


def consonance_peaks(peaks, limit):
    """
    This function computes consonance (for a given ratio a/b, when a < 2b),
    consonance corresponds to (a+b)/(a*b)) between peaks

    Parameters
    ----------
    peaks : List (float)
        Peaks represent local maximum in a spectrum
    limit : float
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
    consonance : List (float)
        consonance scores for each pairs of consonant peaks
    cons_pairs : List of lists (float)
        list of lists of each pairs of consonant peaks
    cons_peaks : List (float)
        list of consonant peaks (no doublons)
    cons_tot : float
        averaged consonance value for each pairs of peaks

    """
    consonance_ = []
    peaks2keep = []
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
                p2x = 0.06  # random  number to avoid division by 0
            cons_ = biotuner.metrics.compute_consonance(p2x/p1x)
            if cons_ < 1:
                cons_tot.append(cons_)
            if cons_ < limit or cons_ == 2:
                cons_ = None
                cons_ = None
                p2x = None
                p1x = None
            if p2x is not None:
                peaks2keep_temp.extend([p2, p1])
            consonance_.append(cons_)
            peaks2keep.append(peaks2keep_temp)
        cons_pairs = [x for x in peaks2keep if x]
        consonance = [i for i in consonance_ if i]
        cons_peaks = list(itertools.chain(*cons_pairs))
        cons_peaks = [np.round(c, 2) for c in cons_peaks]
        cons_peaks = list(set(cons_peaks))
    return consonance, cons_pairs, cons_peaks, np.average(cons_tot)


def multi_consonance(cons_pairs, n_freqs=5):
    """
    Function that keeps the frequencies that are the most consonant with others
    Takes pairs of frequencies that are consonant
    (output of the 'compute consonance' function)

    Parameters
    ----------
    cons_pairs : List of lists (float)
        list of lists of each pairs of consonant peaks
    n_freqs : int
        maximum number of consonant freqs to keep

    Returns
    -------
    freqs_related : List (float)
        peaks that are consonant with at least two other peaks,
        starting with the peak that is consonant with the maximum
        number of other peaks
    """
    freqs_dup = list(itertools.chain(*cons_pairs))
    pairs_temp = list(itertools.chain.from_iterable(cons_pairs))
    freqs_nodup = list(dict.fromkeys(pairs_temp))
    f_count = []
    for f in freqs_nodup:
        f_count.append(freqs_dup.count(f))
    freqs_related = [x for _, x in sorted(zip(
                                              f_count,
                                              freqs_nodup))][-(n_freqs):][::-1]
    return freqs_related


def consonant_ratios(data, limit, sub=False, input_type='peaks',
                     metric='cons'):
    """
    Function that computes integer ratios from peaks with higher consonance

    Parameters
    ----------
    data : List (float)
        Data can whether be frequency values or frequency ratios
    limit : float
        minimum consonance value to keep associated pairs of peaks
    sub : boolean
        Defaults to False
        When set to True, include ratios a/b when a < b.
    input_type : str
        Defaults to 'peaks'.
        Choose between 'peaks' and 'ratios'.
    metric : str
        Defaults to 'cons'.
        Choose between 'cons' and 'harmsim'.

    Returns
    -------
    cons_ratios : List (float)
        list of consonant ratios
    consonance : List (float)
        list of associated consonance values
    """
    consonance_ = []
    ratios2keep = []
    if input_type == 'peaks':
        ratios = biotuner.biotuner_utils.compute_peak_ratios(data, sub=sub)
    if input_type == 'ratios':
        ratios = data
    for ratio in ratios:
        if metric == 'cons':
            cons_ = biotuner.metrics.compute_consonance(ratio)
        if metric == 'harmsim':
            cons_ = biotuner.metrics.dyad_similarity(ratio)
        if cons_ > limit:
            consonance_.append(cons_)
            ratios2keep.append(ratio)
    ratios2keep = np.array(np.round(ratios2keep, 3))
    cons_ratios = np.sort(list(set(ratios2keep)))
    consonance = np.array(consonance_)
    consonance = [i for i in consonance if i]
    return cons_ratios, consonance
