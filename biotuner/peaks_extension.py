"""biotuner.peaks_extension — extend a peak set with harmonics / consonant fits.

Module type: Functions
"""

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

    # Prepare harm_fit. ``harm_temp`` is a list of per-pair lists of common
    # harmonics. Flatten into a single list of harmonic frequencies, regardless
    # of how many peak pairs were compared. The historical
    # ``np.array(...).squeeze()`` collapsed length-1 ``harm_temp`` to a 0-d
    # numpy array, which broke ``len(harm_fit)`` callers downstream.
    try:
        harm_fit = list(itertools.chain.from_iterable(harm_temp))
    except TypeError:
        # Defensive: if compareLists returned a non-iterable for some pair,
        # fall back to the squeeze approach for that one element.
        flat = []
        for item in harm_temp:
            try:
                flat.extend(list(item))
            except TypeError:
                flat.append(item)
        harm_fit = flat

    if len(peak_bands) > 2:
        try:
            harm_fit = [round(num, 3) for num in harm_fit]
            harm_fit = list(dict.fromkeys(harm_fit))  # Remove duplicates (preserve order)
            harm_fit = list(set(harm_fit))  # Final deduplication
        except TypeError:
            print("No common harmonics found, setting harm_fit to empty")
            harm_fit = []

    return harm_fit, harmonics_pos, most_common_harmonics, matching_positions


"""EXTENDED PEAKS from intermodulation (sum / difference tones)"""


def intermodulation_spectrum(
    peaks,
    amplitudes=None,
    max_order=2,
    sum_diff=("sum", "diff"),
    drop_negative=True,
    min_freq=0.0,
    drop_originals=False,
    dedupe=True,
    dedupe_tol=1e-6,
):
    """Compute intermodulation distortion (IMD) products of a peak set.

    For each ordered pair ``(i, j)`` of distinct peaks and integer
    coefficients ``(m, n)`` with ``m + n = order``, ``m >= 1``,
    ``n >= 1``, generate the IMD products::

        m * f_i + n * f_j   (sum-type product)
        m * f_i - n * f_j   (difference-type product)

    with amplitudes ``(a_i ** m) * (a_j ** n)`` (the standard
    power-series nonlinearity model: a memoryless polynomial of order
    ``order`` produces order-``order`` IMD products with amplitudes
    proportional to the product of input amplitudes raised to the
    matching coefficients).

    These products are the *Tartini* (combination) tones perceived in
    a chord: the brain's auditory system generates them via cochlear
    nonlinearity, and many chord-resolution effects are explained by
    where the IMD products fall on the consonance lattice. They are
    also the spectral primitive for n-limit tuning theory: the
    ``n``-limit of a chord is, equivalently, the largest prime in any
    of its IMD products.

    Parameters
    ----------
    peaks : sequence of float
        Input peak frequencies in Hz.
    amplitudes : sequence of float, optional
        Per-peak amplitudes. Defaults to uniform ``1 / N``.
    max_order : int, default=2
        Maximum IMD order ``m + n`` to compute. ``2`` gives the
        textbook ``f_i ± f_j`` set; ``3`` adds ``2 f_i ± f_j`` and
        ``f_i ± 2 f_j``; higher orders proliferate quickly.
    sum_diff : tuple of {'sum', 'diff'}, default=('sum', 'diff')
        Which IMD families to include.
    drop_negative : bool, default=True
        Discard products with ``f < 0`` (sign-flipped difference terms).
    min_freq : float, default=0.0
        Discard products below this frequency (Hz).
    drop_originals : bool, default=False
        Exclude the original input peaks from the output. ``False`` (the
        default) prepends the originals so the output is ``originals +
        IMD products``.
    dedupe : bool, default=True
        If True, products that land on the same frequency (within
        ``dedupe_tol``) are merged: their amplitudes are summed.
    dedupe_tol : float, default=1e-6
        Frequency tolerance for the dedupe pass.

    Returns
    -------
    peaks_out : numpy.ndarray
        Output peak frequencies sorted ascending.
    amps_out : numpy.ndarray
        Corresponding amplitudes.
    sources : list
        Records of how each output peak was generated. Each record is a
        list of ``(m, n, i, j, sign)`` tuples (one tuple per
        contribution; multiple after dedupe). Originals (when included)
        carry the sentinel record ``(0, 0, i, i, '0')``.

    Examples
    --------
    >>> import numpy as np
    >>> peaks, amps, _ = intermodulation_spectrum(
    ...     [100.0, 150.0], amplitudes=[1.0, 1.0], max_order=2,
    ... )
    >>> sorted(peaks.tolist())
    [50.0, 100.0, 150.0, 250.0]

    Notes
    -----
    Used by ``biotuner.harmonic_geometry`` to enrich a chord with its
    IMD-derived peaks before rendering. Composition pattern::

        from biotuner.peaks_extension import intermodulation_spectrum
        from biotuner.harmonic_geometry import (
            HarmonicInput, quasicrystal_field_2d,
        )
        chord = HarmonicInput(peaks=[100.0, 125.0, 150.0])
        imd_peaks, imd_amps, _ = intermodulation_spectrum(
            chord.peaks, chord.amplitudes,
        )
        rich = HarmonicInput(peaks=imd_peaks, amplitudes=imd_amps)
        field = quasicrystal_field_2d(rich)
    """
    if max_order < 2:
        raise ValueError(f"max_order must be >= 2, got {max_order!r}.")

    invalid_types = set(sum_diff) - {"sum", "diff"}
    if invalid_types:
        raise ValueError(
            f"sum_diff entries must be 'sum' or 'diff', got {sum_diff!r}."
        )
    if dedupe_tol < 0:
        raise ValueError(f"dedupe_tol must be >= 0, got {dedupe_tol!r}.")

    peaks_in = np.asarray(peaks, dtype=np.float64)
    n = peaks_in.shape[0]
    if n == 0:
        return np.empty(0), np.empty(0), []
    if amplitudes is None:
        amps_in = np.full(n, 1.0 / max(n, 1), dtype=np.float64)
    else:
        amps_in = np.asarray(amplitudes, dtype=np.float64)
        if amps_in.shape[0] != n:
            raise ValueError(
                f"amplitudes has length {amps_in.shape[0]} but {n} peaks were given."
            )

    out_peaks = []
    out_amps = []
    out_sources = []

    if not drop_originals:
        for i in range(n):
            out_peaks.append(float(peaks_in[i]))
            out_amps.append(float(amps_in[i]))
            out_sources.append((0, 0, i, i, "0"))

    if n >= 2:
        for order in range(2, max_order + 1):
            for m in range(1, order):
                k = order - m
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            continue
                        fi = peaks_in[i]
                        fj = peaks_in[j]
                        amp_ij = (amps_in[i] ** m) * (amps_in[j] ** k)
                        if "sum" in sum_diff:
                            f = m * fi + k * fj
                            if f >= min_freq and (
                                not drop_negative or f >= 0
                            ):
                                out_peaks.append(float(f))
                                out_amps.append(float(amp_ij))
                                out_sources.append((m, k, i, j, "+"))
                        if "diff" in sum_diff:
                            f = m * fi - k * fj
                            if (not drop_negative or f >= 0) and f >= min_freq:
                                out_peaks.append(float(f))
                                out_amps.append(float(amp_ij))
                                out_sources.append((m, k, i, j, "-"))

    # Sort everything by frequency, then optionally dedupe.
    arr_p = np.asarray(out_peaks, dtype=np.float64)
    arr_a = np.asarray(out_amps, dtype=np.float64)
    order_idx = np.argsort(arr_p)
    arr_p = arr_p[order_idx]
    arr_a = arr_a[order_idx]
    sorted_sources = [out_sources[i] for i in order_idx.tolist()]

    if dedupe and arr_p.size > 0:
        merged_peaks = []
        merged_amps = []
        merged_sources = []
        cur_p = arr_p[0]
        cur_a = arr_a[0]
        cur_src = [sorted_sources[0]]
        for k_idx in range(1, arr_p.size):
            if abs(arr_p[k_idx] - cur_p) <= dedupe_tol:
                # Merge.
                cur_a = cur_a + arr_a[k_idx]
                cur_src.append(sorted_sources[k_idx])
            else:
                merged_peaks.append(cur_p)
                merged_amps.append(cur_a)
                merged_sources.append(cur_src)
                cur_p = arr_p[k_idx]
                cur_a = arr_a[k_idx]
                cur_src = [sorted_sources[k_idx]]
        merged_peaks.append(cur_p)
        merged_amps.append(cur_a)
        merged_sources.append(cur_src)
        return (
            np.asarray(merged_peaks, dtype=np.float64),
            np.asarray(merged_amps, dtype=np.float64),
            merged_sources,
        )

    return arr_p, arr_a, [[s] for s in sorted_sources]


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
