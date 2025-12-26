import numpy as np
import itertools
from biotuner.biotuner_utils import rebound, compareLists
from collections import Counter
from functools import reduce
import itertools
import numpy as np


"""EXTENDED PEAKS from expansions
   (finding new peaks based on harmonic structure)
"""


def EEG_harmonics_mult(peaks, n_harmonics, n_oct_up=0):
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
    n_oct_up : int, default=0
        The number of octaves by which to shift the peaks before computing
        the harmonics.

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


def EEG_harmonics_div(peaks, n_harmonics, n_oct_up=0, mode="div"):
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
    n_oct_up : int, default=0
        The number of octaves by which to shift the peaks before computing
        the sub-harmonics.
    mode : str, default='div'
        The mode to use for computing the sub-harmonics. Possible values are
        'div' for x, x/2, x/3 ..., x/n and 'div_add' for x, (x+x/2), (x+x/3),
        ... (x+x/n).

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


def harmonic_fit(
    peaks, n_harm=10, bounds=1, function="mult", div_mode="div", n_common_harms=5
):
    """
    Compute harmonics of a list of peaks and compare the lists of
    harmonics pairwise to find fitting between the harmonic series.

    Parameters
    ----------
    peaks : list of float
        Spectral peaks representing local maximum in a spectrum
    n_harm : int, default=10
        Number of harmonics to compute.
    bounds : int, default=1
        Minimum distance (in Hz) between two frequencies to consider a fit.
    function : str, default='mult'
        Type of harmonic function to use.
        Possible values are:

        - 'mult' will use natural harmonics.
        - 'div' will use natural sub-harmonics.
        - 'exp' will use exponentials.
    div_mode : str, default='div'
        Mode of the natural sub-harmonic function when function='div'.
        See EEG_harmonics_div function.
    n_common_harms : int, default=5
        Minimum number of times a harmonic position must appear across
        different peak pairs to be included in most_common_harmonics output.
        Acts as a threshold filter (not a limit on number of results).

    Returns
    -------
    harm_fit : list
        Frequencies of the harmonics that match.
    harmonics_pos : list
        Positions of the harmonics that match.
    most_common_harmonics : list
        Harmonic positions that appear at least n_common_harms times across
        peak pairs, sorted by frequency of occurrence (most common first).
    matching_positions : list of lists
        Each sublist corresponds to an harmonic fit, the first number
        is the frequency and the two others are harmonic positions.

    Examples
    --------
    >>> from biotuner.peaks_extension import harmonic_fit
    >>> peaks = [3, 9, 12]
    >>> harm_fit, harmonics_pos, _, _ = harmonic_fit(peaks, n_harm=5, bounds=0.1, function="mult")
    >>> print(harm_fit)
    >>> harm_fit, harmonics_pos, _, _ = harmonic_fit(peaks, n_harm=10, bounds=0.1, function="div")
    >>> print(harm_fit)
    [9.0, 18.0, 12.0, 36.0]
    [0.784, 1.5, 1.045, 3.0, 1.0, 1.757, 1.31, 1.243, 1.162, 1.108]
    """
    from itertools import combinations

    peak_bands = []
    for i in range(len(peaks)):
        peak_bands.append(i)
    if function == "mult":
        multi_harmonics = EEG_harmonics_mult(peaks, n_harm)
    elif function == "div":
        multi_harmonics, _ = EEG_harmonics_div(peaks, n_harm, mode=div_mode)
    elif function == "exp":
        multi_harmonics = np.array(
            [[i**h for i in peaks] for h in range(1, n_harm + 1)]
        )
        multi_harmonics = np.moveaxis(multi_harmonics, 0, 1)

    # Compare harmonics pairwise
    list_peaks = list(combinations(peak_bands, 2))
    harm_temp = []
    matching_positions = []
    harmonics_pos = []

    for i in range(len(list_peaks)):
        harms, harm_pos, matching_pos, _ = compareLists(
            multi_harmonics[list_peaks[i][0]], multi_harmonics[list_peaks[i][1]], bounds
        )
        harm_temp.append(harms)
        harmonics_pos.append(harm_pos)
        matching_positions.extend(matching_pos)

    # Flatten harmonics positions
    if harmonics_pos:
        harmonics_pos = reduce(lambda x, y: x + y, harmonics_pos)
    else:
        harmonics_pos = []

    # Compute most common harmonics
    # Filter harmonics that appear at least n_common_harms times (as per docstring)
    # Then take all such harmonics (not just top N)
    harmonic_counts = Counter(harmonics_pos)
    most_common_harmonics = [
        h for h, h_count in harmonic_counts.items() 
        if h_count >= n_common_harms
    ]
    # Sort by frequency (most common first), then by harmonic position
    most_common_harmonics = sorted(
        most_common_harmonics, 
        key=lambda h: (harmonic_counts[h], h), 
        reverse=True
    )
    harmonics_pos = sorted(set(harmonics_pos))

    # Prepare harm_fit
    harm_fit = np.array(harm_temp, dtype=object).squeeze()

    if len(peak_bands) > 2:
        try:
            if isinstance(harm_fit, (list, np.ndarray)):
                harm_fit = list(itertools.chain.from_iterable(harm_fit))
            harm_fit = [round(num, 3) for num in harm_fit]
            harm_fit = list(dict.fromkeys(harm_fit))  # Remove duplicates
            harm_fit = list(set(harm_fit))  # Final deduplication
        except TypeError:
            print("No common harmonics found, setting harm_fit to empty")
            harm_fit = []

    return harm_fit, harmonics_pos, most_common_harmonics, matching_positions


"""EXTENDED PEAKS from restrictions"""


def multi_consonance(cons_pairs, n_freqs=5):
    """
    Function that keeps the frequencies that are the most consonant with others
    Takes pairs of frequencies that are consonant as input
    (output of the 'compute consonance' function).

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
    freqs_related = [x for _, x in sorted(zip(f_count, freqs_nodup))][-(n_freqs):][::-1]
    return freqs_related
