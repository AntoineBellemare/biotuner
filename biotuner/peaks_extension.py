import numpy as np
import itertools
from biotuner.biotuner_utils import rebound, compareLists
from collections import Counter
from functools import reduce
import biotuner.metrics

"""EXTENDED PEAKS from expansions
   (finding new peaks based on harmonic structure)
"""


def EEG_harmonics_mult (peaks, n_harmonics, n_oct_up=0):
    """
    Computes the harmonics of a list of frequency peaks.
    Given a list of frequency peaks, this function computes the desired
    number of harmonics for each peak. The harmonics are calculated
    using the formula x, 2x, 3x ..., nx, where x is the frequency of the
    peak and n_harmonics is the number of harmonics to compute.

    Parameters
    ----------
    peaks : list of float
        The frequency peaks, represented as local maxima in a spectrum.
    n_harmonics : int
        The number of harmonics to compute for each peak.
    n_oct_up : int, optional
        The number of octaves by which to shift the peaks before computing
        the harmonics. Defaults to 0.

    Returns
    -------
    multi_harmonics : numpy.ndarray
        An array of shape (n_peaks, n_harmonics+1), where n_peaks is the
        number of frequency peaks and each row represents the computed
        harmonics for the corresponding peak.

    Examples
    --------
    >>> peaks = [10.0, 20.0, 30.0]
    >>> n_harmonics = 3
    >>> n_oct_up = 1
    >>> EEG_harmonics_mult(peaks, n_harmonics, n_oct_up)
    array([[ 20.,  40.,  60.,  80.],
        [ 40.,  80., 120., 160.],
        [ 60., 120., 180., 240.]])
    """

    n_harmonics = n_harmonics + 2
    multi_harmonics = []
    for p in peaks:
        harmonics = []
        p = p * (2**n_oct_up)
        for i in range(1, n_harmonics):
            harmonics.append(p * i)
        multi_harmonics.append(harmonics)
    multi_harmonics = np.array(multi_harmonics)

    return multi_harmonics


def EEG_harmonics_div (peaks, n_harmonics, n_oct_up=0, mode="div"):
    """
    Computes the sub-harmonics of a list of frequency peaks using division.
    Given a list of frequency peaks, this function computes the desired
    number of sub-harmonics for each peak using division. The sub-harmonics
    are calculated using the formulas x, x/2, x/3 ..., x/n or x, (x+x/2),
    (x+x/3), ... (x+x/n), depending on the specified mode, where x is the
    frequency of the peak and n_harmonics is the number of sub-harmonics to
    compute.

    Parameters
    ----------
    peaks : list of float
        The frequency peaks, represented as local maxima in a spectrum.
    n_harmonics : int
        The number of sub-harmonics to compute for each peak.
    n_oct_up : int, optional
        The number of octaves by which to shift the peaks before computing
        the sub-harmonics. Defaults to 0.
    mode : str, optional
        The mode to use for computing the sub-harmonics. Possible values are
        'div' for x, x/2, x/3 ..., x/n and 'div_add' for x, (x+x/2), (x+x/3),
        ... (x+x/n). Defaults to 'div'.

    Returns
    -------
    div_harmonics : numpy.ndarray
        An array of shape (n_peaks, n_harmonics+1), where n_peaks is the number
        of frequency peaks and each row represents the computed sub-harmonics
        for the corresponding peak, in Hz.
    div_harmonics_bounded : numpy.ndarray
        An array of shape (n_peaks, n_harmonics+1), where n_peaks is the number
        of frequency peaks and each row represents the computed sub-harmonics
        for the corresponding peak, bounded between unison (1) and octave (2),
        in Hz.

    Examples
    --------
    >>> peaks = [10.0, 20.0, 30.0]
    >>> n_harmonics = 3
    >>> n_oct_up = 1
    >>> mode = 'div'
    >>> EEG_harmonics_div(peaks, n_harmonics, n_oct_up, mode)
    (array([[10.        ,  5.        ,  3.33333333,  2.5       ],
            [20.        , 10.        ,  6.66666667,  5.        ],
            [30.        , 15.        , 10.        ,  7.5       ]]),
    array([[1.        , 1.        , 1.        , 1.        ],
            [2.        , 2.        , 1.5       , 1.25      ],
            [2.        , 2.        , 1.5       , 1.25      ]]))
    """
    n_harmonics = n_harmonics + 2
    div_harmonics = []
    for p in peaks:
        harmonics = []
        p = p * (2**n_oct_up)
        harm_temp = p
        for i in range(1, n_harmonics):
            if mode == "div":
                harm_temp = p / i
            if mode == "div_add":
                harm_temp = p + (p / i)
            harmonics.append(harm_temp)
        div_harmonics.append(harmonics)
    div_harmonics = np.array(div_harmonics)
    div_harm_bound = div_harmonics.copy()
    # Rebound the result between 1 and 2
    for i in range(len(div_harm_bound)):
        for j in range(len(div_harm_bound[i])):
            div_harm_bound[i][j] = rebound(div_harm_bound[i][j])
    return div_harmonics, div_harm_bound


def harmonic_fit (peaks,
                 n_harm=10,
                 bounds=1,
                 function="mult",
                 div_mode="div",
                 n_common_harms=5):
    """
    Compute harmonics of a list of peaks and compare the lists of
    harmonics pairwise to find fitting between the harmonic series.

    Parameters
    ----------
    peaks : list of float
        Spectral peaks representing local maximum in a spectrum
    n_harm : int, optional
        Number of harmonics to compute. Default is 10.
    bounds : int, optional
        Minimum distance (in Hz) between two frequencies to consider a fit.
        Default is 1.
    function : str, optional
        Type of harmonic function to use. Default is 'mult'.
        Possible values are 'mult', 'div', and 'exp'.
        'mult' will use natural harmonics.
        'div' will use natural sub-harmonics.
        'exp' will use exponentials.
    div_mode : str, optional
        Mode of the natural sub-harmonic function when function='div'.
        Default is 'div'. See EEG_harmonics_div function.
    n_common_harms : int, optional
        Minimum number of times the harmonic is found to be sent to
        the most_common_harmonics output. Default is 5.

    Returns
    -------
    harm_fit : list
        Frequencies of the harmonics that match.
    harmonics_pos : list
        Positions of the harmonics that match.
    most_common_harmonics : list
        Harmonics that are present at least 'n_common_harms' times.
    matching_positions : list of lists
        Each sublist corresponds to an harmonic fit, the first number
        is the frequency and the two others are harmonic positions.
    """
    from itertools import combinations

    peak_bands = []
    for i in range(len(peaks)):
        peak_bands.append(i)
    if function == "mult":
        multi_harmonics = EEG_harmonics_mult(peaks, n_harm)
    elif function == "div":
        multi_harmonics, x = EEG_harmonics_div(peaks, n_harm, mode=div_mode)
    elif function == "exp":
        multi_harmonics = []
        for h in range(n_harm + 1):
            h += 1
            multi_harmonics.append([i**h for i in peaks])
        multi_harmonics = np.array(multi_harmonics)
        multi_harmonics = np.moveaxis(multi_harmonics, 0, 1)
    list_peaks = list(combinations(peak_bands, 2))
    harm_temp = []
    matching_positions = []
    harmonics_pos = []
    for i in range(len(list_peaks)):
        harms, harm_pos, matching_pos, _ = compareLists(
                                            multi_harmonics[list_peaks[i][0]],
                                            multi_harmonics[list_peaks[i][1]],
                                            bounds
                                        )
        harm_temp.append(harms)
        harmonics_pos.append(harm_pos)
        if len(matching_pos) > 0:
            for j in matching_pos:
                matching_positions.append(j)
    matching_positions = [list(i) for i in matching_positions]
    harm_fit = np.array(harm_temp, dtype=object).squeeze()
    harmonics_pos = reduce(lambda x, y: x + y, harmonics_pos)
    most_common_harmonics = [
        h
        for h, h_count in Counter(harmonics_pos).most_common(n_common_harms)
        if h_count > 1
    ]
    harmonics_pos = list(np.sort(list(set(harmonics_pos))))
    if len(peak_bands) > 2:
        harm_fit = list(itertools.chain.from_iterable(harm_fit))
        harm_fit = [round(num, 3) for num in harm_fit]
        harm_fit = list(dict.fromkeys(harm_fit))
        harm_fit = list(set(harm_fit))
    return harm_fit, harmonics_pos, most_common_harmonics, matching_positions


"""EXTENDED PEAKS from restrictions"""


def consonance_peaks (peaks, limit):
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
                    p1x = p1x / 2
            if p1x < p2x:
                while p2x > p1x:
                    p2x = p2x / 2
            if p1x < 0.1:
                p1x = 0.06
            if p2x < 0.1:
                p2x = 0.06  # random  number to avoid division by 0
            cons_ = biotuner.metrics.compute_consonance(p2x / p1x)
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


def multi_consonance (cons_pairs, n_freqs=5):
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
    freqs_related = [x for _, x in sorted(zip(f_count,
                                              freqs_nodup))][-(n_freqs):][::-1]
    return freqs_related



