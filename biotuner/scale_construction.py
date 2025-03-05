#!bin/bash
import numpy as np
import matplotlib.pyplot as plt
import sys
from biotuner.biotuner_utils import (
    nth_root,
    rebound,
    NTET_ratios,
    scale2frac,
    findsubsets,
    scale_from_pairs,
    ratio2frac,
)
from biotuner.metrics import (
    ratios2harmsim,
    euler,
    dyad_similarity,
    metric_denom,
    tuning_cons_matrix,
    consonant_ratios,
    tuning_to_metrics,
)

import itertools
from collections import Counter
from numpy import linspace, empty, concatenate, log2
from scipy.signal import argrelextrema
from fractions import Fraction
from scipy.stats import norm
import contfrac
from math import log2
import math
import sympy as sp
import itertools
import operator
from functools import reduce


sys.setrecursionlimit(120000)


def oct_subdiv(ratio, octave_limit=0.01365, octave=2, n=5):
    """
    N-TET tuning from Generator Interval.

    This function uses a generator interval to suggest numbers of steps to divide the octave.

    Parameters
    ----------
    ratio : float
        Ratio that corresponds to the generator_interval.
        For example, by giving the fifth (3/2) as generator interval,
        this function will suggest to subdivide the octave in 12, 53, etc.
    octave_limit : float, default=0.01365
        Approximation of the octave corresponding to the acceptable distance
        between the ratio of the generator interval after multiple iterations
        and the octave value.
        The default value of 0.01365 corresponds to the Pythagorean comma.
    octave : int, default=2
        Value of the octave.
    n : int, default=5
        Number of suggested octave subdivisions.

    Returns
    -------
    Octdiv : List of int
        List of N-TET tunings according to the generator interval.
    Octvalue : List of float
        List of the approximations of the octave for each N-TET tuning.

    Examples
    --------
    >>> oct_subdiv(3/2, n=3)
    ([12, 53, 106], [1.0136432647705078, 1.0020903140410862, 1.0041849974949628])
    """
    Octdiv, Octvalue, i = [], [], 1
    ratios = []
    while len(Octdiv) < n:
        ratio_mult = ratio**i
        while ratio_mult > octave:
            ratio_mult = ratio_mult / octave

        rescale_ratio = ratio_mult - round(ratio_mult)
        ratios.append(ratio_mult)
        i += 1
        if -octave_limit < rescale_ratio < octave_limit:
            Octdiv.append(i - 1)
            Octvalue.append(ratio_mult)
        else:
            continue
    return Octdiv, Octvalue


def compare_oct_div(Octdiv=12, Octdiv2=53, bounds=0.005, octave=2):
    """
    Function that compares steps for two N-TET tunings
    and returns matching ratios and corresponding degrees

    Parameters
    ----------
    Octdiv : int, default=12
        First N-TET tuning number of steps.
    Octdiv2 : int, default=53
        Second N-TET tuning number of steps.
    bounds : float, default=0.005
        Maximum distance between one ratio of Octdiv and one ratio of Octdiv2
        to consider a match.
    octave : int, default=2
        Value of the octave

    Returns
    -------
    avg_ratios : numpy.ndarray
        List of ratios corresponding to the shared steps in the two N-TET tunings
    shared_steps : List of tuples
        The two elements of each tuple corresponds to the tuning steps
        sharing the same interval in the two N-TET tunings

    Examples
    --------
    >>> ratios, shared_steps = compare_oct_div(Octdiv=12, Octdiv2=53, bounds=0.005, octave=2)
    >>> ratios, shared_steps
    ([1.124, 1.187, 1.334, 1.499, 1.78, 2.0],
    [(2, 9), (3, 13), (5, 22), (7, 31), (10, 44), (12, 53)])
    """
    ListOctdiv = []
    ListOctdiv2 = []
    OctdivSum = 1
    OctdivSum2 = 1
    i = 1
    i2 = 1
    while OctdivSum < octave:
        OctdivSum = (nth_root(octave, Octdiv)) ** i
        i += 1
        ListOctdiv.append(OctdivSum)
    while OctdivSum2 < octave:
        OctdivSum2 = (nth_root(octave, Octdiv2)) ** i2
        i2 += 1
        ListOctdiv2.append(OctdivSum2)
    shared_steps = []
    avg_ratios = []
    for i, n in enumerate(ListOctdiv):
        for j, harm in enumerate(ListOctdiv2):
            if harm - bounds < n < harm + bounds:
                shared_steps.append((i + 1, j + 1))
                avg_ratios.append((n + harm) / 2)
    avg_ratios = [round(x, 3) for x in avg_ratios]
    return avg_ratios, shared_steps


def multi_oct_subdiv(
    peaks, max_sub=100, octave_limit=0.01365, octave=2, n_scales=10, cons_limit=0.1
):
    """
    Determine optimal octave subdivisions based on consonant peaks ratios.

    This function takes the most consonant peaks ratios and uses them as input for
    the oct_subdiv function. Each consonant ratio generates a list of possible
    octave subdivisions. The function then compares these lists and identifies
    optimal octave subdivisions that are common across multiple generator intervals.

    Parameters
    ----------
    peaks : List of float
        Peaks represent local maximum in a spectrum.
    max_sub : int, default=100
        Maximum number of intervals in N-TET tuning suggestions.
    octave_limit : float, default=0.01365
        Approximation of the octave corresponding to the acceptable distance
        between the ratio of the generator interval after
        multiple iterations and the octave value.
    octave : int, default=2
        value of the octave
    n_scales : int, default=10
        Number of N-TET tunings to compute for each generator interval (ratio).
    cons_limit : float, default=0.1
        Limit for the consonance of the peaks ratios.

    Returns
    -------
    multi_oct_div : List of int
        List of octave subdivisions that fit with multiple generator intervals.
    ratios : List of float
        List of the generator intervals for which at least 1 N-TET tuning
        matches with another generator interval.

    Examples
    --------
    >>> peaks = [2, 3, 9]
    >>> oct_divs, x = multi_oct_subdiv(peaks, max_sub=100)
    >>> oct_divs, x
    ([53], array([1.125, 1.5  ]))
    """
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
    if len(multi_oct_div) == 0:  # Gracefully handle empty results
        return [], ratios
    return multi_oct_div, ratios


def harmonic_tuning(list_harmonics, octave=2, min_ratio=1, max_ratio=2):
    """
    Generates a tuning based on a list of harmonic positions.

    Parameters
    ----------
    list_harmonics : List of int
        harmonic positions to use in the scale construction
    octave : int
        value of the period reference
    min_ratio : float, default=1
        Value of the unison.
    max_ratio : float, default=2
        Value of the octave.

    Returns
    -------
    ratios : List of float
        Generated tuning.

    Examples
    --------
    >>> list_harmonics = [3, 5, 7, 9]
    >>> harmonic_tuning(list_harmonics, octave=2, min_ratio=1, max_ratio=2)
    [1.125, 1.25, 1.5, 1.75]
    """
    list_harmonics = np.abs(list_harmonics)
    ratios = []
    for i in list_harmonics:
        ratios.append(rebound(1 * i, min_ratio, max_ratio, octave))
    ratios = list(set(ratios))
    ratios = list(np.sort(np.array(ratios)))
    return ratios


def euler_fokker_scale(intervals, n=1, octave=2, normalize=True):
    """
    Generates a tuning based on a set of prime factors in the Euler-Fokker Genera.

    Parameters
    ----------
    intervals : List of int
        Prime factors to use in the scale construction.
    n : int, default=1
        The multiplicity of each factor, controlling how many times each is used.
    octave : int, default=2
        Value of the period reference.
    normalize : bool, default=True
        If True, normalizes the scale to fit within the octave.

    Returns
    -------
    scale : List of sympy.Integer or sympy.Rational
        Generated tuning.

    Examples
    --------
    >>> intervals = [3, 5, 7]
    >>> euler_fokker_scale(intervals, n=1, octave=2, normalize=True)
    [1, 35/32, 5/4, 21/16, 3/2, 105/64, 7/4, 15/8, 2]
    """
    multiplicities = [n for x in intervals]  # Each factor is used once.
    output = []
    for index in range(len(intervals)):
        output = output + [intervals[index]] * multiplicities[index]
        output = [sp.Integer(x) for x in output]

    potential = list(
        itertools.chain(
            *[
                [x for x in itertools.combinations(output, r)]
                for r in range(1, len(output) + 1)
            ]
        )
    )
    output = []
    for item in potential:
        if len(item) == 0:
            output = output + [item[0]]
        else:
            output = output + [reduce(operator.__mul__, item)]

    if normalize:
        output = (
            [sp.Integer(1)]
            + [rebound(x, octave=octave) for x in output]
            + [sp.Integer(octave)]
        )
    else:
        output = [sp.Integer(1)] + [x for x in output] + [sp.Integer(octave)]

    output = sorted(set(output))
    return output


def generator_interval_tuning(interval=3 / 2, steps=12, octave=2, harmonic_min=0):
    """
    Function that takes a generator interval and
    derives a tuning based on its stacking.

    Parameters
    ----------
    interval : float
        Generator interval
    steps : int, default=12
        Number of steps in the scale
        When default, 12-TET for interval 3/2
    octave : int
        Defaults to 2
        Value of the octave

    Returns
    -------
    tuning : List of float
        Generated tuning.

    Examples
    --------
    >>> tuning = generator_interval_tuning(interval=3/2, steps=12, octave=2, harmonic_min=0)
    >>> tuning
    [1.0,
    1.06787109375,
    1.125,
    1.20135498046875,
    1.265625,
    1.3515243530273438,
    1.423828125,
    1.5,
    1.601806640625,
    1.6875,
    1.802032470703125,
    1.8984375]
    """
    tuning = []
    for s in range(steps):
        degree = interval**harmonic_min
        while degree > octave:
            degree = degree / octave
        while degree < octave / 2:
            degree = degree * octave
        tuning.append(degree)
        harmonic_min += 1
    tuning = sorted(list(set(tuning)))
    return tuning


def convergents(interval):
    """
    Return the convergents of the log2 of a ratio.
    The second value represents the number of steps to divide the octave
    while the first value represents the number of octaves up before
    the stacked ratio arrives approximately to the octave value.
    For example, the output of the interval 1.5 will includes [7, 12],
    which means that to approximate the fifth (1.5) in a NTET-tuning, you can
    divide the octave in 12, while stacking 12 fifth will lead to the 7th octave up.

    Parameters
    ----------
    interval : float
        Interval to find convergent.

    Returns
    -------
    convergents : List of lists
        Each sublist corresponds to a pair of convergents.

    Examples
    --------
    >>> convergents(3/2)
    [(0, 1),
    (1, 1),
    (1, 2),
    (3, 5),
    (7, 12),
    (24, 41),
    (31, 53),
    (179, 306),
    (389, 665),
    (9126, 15601),
    (18641, 31867)]
    """
    value = np.log2(interval)
    convergents = list(contfrac.convergents(value))
    return convergents


# Dissonance curves


def dissmeasure(fvec, amp, model="min"):
    """
    Given a list of partials (peak frequencies) in fvec, with amplitudes in amp,
    this routine calculates the dissonance by summing the roughness of
    every sine pair based on a model of Plomp-Levelt's roughness curve.
    The older model (model='product') was based on the product of the two
    amplitudes, but the newer model (model='min') is based on the minimum
    of the two amplitudes, since this matches the beat frequency amplitude.

    Parameters
    ----------
    fvec : List
        List of frequency values
    amp : List
        List of amplitude values
    model : str, default='min'
        Description of parameter `model`.

    Returns
    -------
    D : float
        Dissonance value
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

    if model == "min":
        a = np.amin(am_pairs, axis=1)
    elif model == "product":
        a = np.prod(am_pairs, axis=1)  # Older model
    else:
        raise ValueError('model should be "min" or "product"')
    SFdif = S * Fdif
    D = np.sum(a * (C1 * np.exp(A1 * SFdif) + C2 * np.exp(A2 * SFdif)))

    return D


def diss_curve(
    freqs,
    amps,
    denom=1000,
    max_ratio=2,
    euler_comp=True,
    method="min",
    plot=True,
    n_tet_grid=None,
):
    """
    This function computes the dissonance curve and related metrics for
    a given set of frequencies (freqs) and amplitudes (amps).

    Parameters
    ----------
    freqs : List (float)
        list of frequencies associated with spectral peaks
    amps : List (float)
        list of amplitudes associated with freqs (must be same lenght)
    denom : int, default=1000
        Highest value for the denominator of each interval
    max_ratio : int, default=2
        Value of the maximum ratio
        Set to 2 for a span of 1 octave
        Set to 4 for a span of 2 octaves
        Set to 8 for a span of 3 octaves
        Set to 2**n for a span of n octaves
    euler : bool, default=True
        When set to True, compute the Euler Gradus Suavitatis
        for the derived scale.
    method : str, default='min'
        Refer to dissmeasure function for more information.

        - 'min'
        - 'product'

    plot : bool, default=True
        Plot the dissonance curve.
    n_tet_grid : int, default=None
        When an integer is given, dotted lines will be add to the plot
        at steps of the given N-TET scale

    Returns
    -------
    diss : list of floats
        list of dissonance values for all the intervals
    intervals : List of tuples
        Each tuple corresponds to the numerator and the denominator
        of each scale step ratio
    ratios : List of floats
        list of ratios that constitute the scale
    euler_score : int
        value of consonance of the scale
    diss : float
        value of averaged dissonance of the total curve
    dyad_sims : List of floats
        list of dyad similarities for each ratio of the scale

    """
    freqs = np.array(freqs)
    r_low = 1
    alpharange = max_ratio
    method = method
    n = 1000
    diss = empty(n)
    a = concatenate((amps, amps))
    for i, alpha in enumerate(linspace(r_low, alpharange, n)):
        f = concatenate((freqs, alpha * freqs))
        d = dissmeasure(f, a, method)
        diss[i] = d
    diss_minima = argrelextrema(diss, np.less)
    intervals = []
    for d in range(len(diss_minima[0])):
        frac = Fraction(
            diss_minima[0][d] / (n / (max_ratio - 1)) + 1
        ).limit_denominator(denom)
        frac = (frac.numerator, frac.denominator)
        intervals.append(frac)
    intervals.append((2, 1))
    ratios = [i[0] / i[1] for i in intervals]
    dyad_sims = ratios2harmsim(ratios[:-1])
    a = 1
    ratios_euler = [a] + ratios
    ratios_euler = [int(round(num, 2) * 1000) for num in ratios]
    euler_score = None
    if euler_comp is True:
        euler_score = euler(*ratios_euler)

        euler_score = euler_score / len(diss_minima)
    else:
        euler_score = "NaN"

    if plot is True:
        plt.figure(figsize=(14, 6), facecolor="white")
        plt.plot(linspace(r_low, alpharange, len(diss)), diss)
        plt.xscale("linear")
        plt.xlim(r_low, alpharange)
        try:
            plt.text(
                1.9,
                1.5,
                "Euler = " + str(int(euler_score)),
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=16,
            )
        except:
            pass
        for n, d in intervals:
            plt.axvline(n / d, color="silver")
        # Plot N-TET grid
        if n_tet_grid is not None:
            n_tet = NTET_ratios(n_tet_grid, max_ratio=max_ratio)
            for n in n_tet:
                plt.axvline(n, color="red", linestyle="--")
        # Plot scale ticks
        plt.minorticks_off()
        plt.xticks(
            [n / d for n, d in intervals],
            ["{}/{}".format(n, d) for n, d in intervals],
            fontsize=13,
        )
        plt.xlabel("Frequency ratio", fontsize=14)
        plt.ylabel("Dissonance", fontsize=14)

        plt.yticks(fontsize=13)
        plt.tight_layout()
        plt.show()
    return diss, intervals, ratios, euler_score, np.average(diss), dyad_sims


"""Harmonic Entropy"""


def compute_harmonic_entropy_domain_integral(
    ratios, ratio_interval, spread=0.01, min_tol=1e-15
):
    """
    Computes the harmonic entropy of a list of frequency ratios for a given set of possible intervals.

    Parameters
    ----------
    ratios : List of floats
        List of frequency ratios.
    ratio_interval : List of floats
        List of possible intervals to consider.
    spread : float, default=0.01
        Controls the width of the Gaussian kernel used to smooth the probability density function of the ratios.
    min_tol : float, default=1e-15
        The smallest tolerance value for considering the probability density function.

    Returns
    -------
    weight_ratios : ndarray
        Sorted ratios.
    HE : ndarray
        Harmonic entropy values for each interval in `ratio_interval`.

    Notes
    -----
    Harmonic entropy is a measure of the deviation of a set of frequency ratios from the idealized harmonics
    (integer multiples of a fundamental frequency) and is defined as:

    HE = - sum_i(p_i * log2(p_i))

    where p_i is the probability of a given ratio in a smoothed probability density function.

    The `ratio_interval` defines a range of possible intervals to consider. The algorithm computes the harmonic entropy
    of each possible interval in `ratio_interval` and returns an array of HE values, one for each interval.

    """
    # The first step is to pre-sort the ratios to speed up computation
    ind = np.argsort(ratios)
    weight_ratios = ratios[ind]

    centers = (weight_ratios[:-1] + weight_ratios[1:]) / 2

    ratio_interval = np.array(ratio_interval)
    N = len(ratio_interval)
    HE = np.zeros(N)
    for i, x in enumerate(ratio_interval):
        P = np.diff(
            concatenate(
                ([0], norm.cdf(np.log2(centers), loc=np.log2(x), scale=spread), [1])
            )
        )
        ind = P > min_tol
        HE[i] = -np.sum(P[ind] * np.log2(P[ind]))

    return weight_ratios, HE


def compute_harmonic_entropy_simple_weights(
    numerators, denominators, ratio_interval, spread=0.01, min_tol=1e-15
):
    """
    Compute the harmonic entropy of a set of ratios using simple weights.

    Parameters
    ----------
    numerators : array-like
        Numerators of the ratios.
    denominators : array-like
        Denominators of the ratios.
    ratio_interval : array-like
        Interval to compute the harmonic entropy over.
    spread : float, default=0.01
        Spread of the normal distribution used to compute the weights.
    min_tol : float, default=1e-15
        Minimum tolerance for the weights.

    Returns
    -------
    weight_ratios : ndarray
        Sorted weight ratios.
    HE : ndarray
        Harmonic entropy.
    """
    # The first step is to pre-sort the ratios to speed up computation
    ratios = numerators / denominators
    ind = np.argsort(ratios)
    numerators = numerators[ind]
    denominators = denominators[ind]
    weight_ratios = ratios[ind]

    ratio_interval = np.array(ratio_interval)
    N = len(ratio_interval)
    HE = np.zeros(N)
    for i, x in enumerate(ratio_interval):
        P = norm.pdf(np.log2(weight_ratios), loc=np.log2(x), scale=spread) / np.sqrt(
            numerators * denominators
        )
        ind = P > min_tol
        P = P[ind]
        P /= np.sum(P)
        HE[i] = -np.sum(P * np.log2(P))

    return weight_ratios, HE


def harmonic_entropy(
    ratios, res=0.001, spread=0.01, plot_entropy=True, plot_tenney=False, octave=2
):
    """
    Harmonic entropy is a measure of the uncertainty in pitch perception,
    and it provides a physical correlate of tonalness,one aspect of the
    psychoacoustic concept of dissonance (Sethares). High tonalness corresponds
    to low entropy and low tonalness corresponds to high entropy.

    Parameters
    ----------
    ratios : List of floats
        Ratios between each pairs of frequency peaks.
    res : float, default=0.001
        Resolution of the ratio steps.
    spread : float, default=0.01
        Spread of the normal distribution used to compute the weights.
    plot_entropy : bool, default=True
        When set to True, plot the harmonic entropy curve.
    plot_tenney : bool, default=False
        When set to True, plot the tenney heights (y-axis)
        across ratios (x-axis).
    octave : int, default=2
        Value of reference period.

    Returns
    ----------
    HE_minima : List of floats
        List of ratios corresponding to minima of the harmonic entropy curve
    HE_avg : float
        Value of the averaged harmonic entropy
    HE : List of floats
        List of harmonic entropy values for each ratio.

    """
    fracs, numerators, denominators = scale2frac(ratios)
    ratios = numerators / denominators
    bendetti_heights = numerators * denominators

    tenney_heights = np.log2(bendetti_heights)

    ind = np.argsort(tenney_heights)  # sort by Tenney height
    bendetti_heights = bendetti_heights[ind]
    tenney_heights = tenney_heights[ind]
    numerators = numerators[ind]
    denominators = denominators[ind]
    if plot_tenney is True:
        fig = plt.figure(figsize=(10, 4), dpi=150)
        ax = fig.add_subplot(111)
        # ax.scatter(ratios, 2**tenney_heights, s=1)
        ax.scatter(ratios, tenney_heights, s=1, alpha=0.2)
        # ax.scatter(ratios[:200], tenney_heights[:200], s=1, color='r')
        plt.show()

    # Next, we need to ensure a distance `d` between adjacent ratios
    M = len(bendetti_heights)
    delta = 0.00001
    indices = np.ones(M, dtype=bool)
    for i in range(M - 2):
        ind = abs(ratios[i + 1 :] - ratios[i]) > delta
        indices[i + 1 :] = indices[i + 1 :] * ind
    bendetti_heights = bendetti_heights[indices]
    tenney_heights = tenney_heights[indices]
    numerators = numerators[indices]
    denominators = denominators[indices]
    ratios = ratios[indices]
    M = len(tenney_heights)
    x_ratios = np.arange(1, octave, res)
    _, HE = compute_harmonic_entropy_domain_integral(ratios, x_ratios, spread=spread)
    # HE = compute_harmonic_entropy_simple_weights(numerators,
    #                                                denominators,
    #                                                x_ratios, spread=0.01)
    ind = argrelextrema(HE, np.less)
    HE_minima = (x_ratios[ind], HE[ind])
    if plot_entropy is True:
        fig = plt.figure(figsize=(10, 4), dpi=150)
        ax = fig.add_subplot(111)
        ax.plot(x_ratios, HE)
        ax.scatter(HE_minima[0], HE_minima[1], color="k", s=4)
        ax.set_xlim(1, octave)
        plt.xlabel("Frequency ratio")
        plt.ylabel("Harmonic entropy")
        plt.show()
    return HE_minima, np.average(HE), HE


"""Scale reduction"""


def tuning_reduction(tuning, mode_n_steps, function, rounding=4, ratio_type="pos_harm"):
    """
    Function that reduces the number of steps in a scale according
    to the consonance between pairs of ratios.

    Parameters
    ----------
    tuning : List (float)
        scale to reduce
    mode_n_steps : int
        number of steps of the reduced scale
    function : function, default=compute_consonance
        function used to compute the consonance between pairs of ratios
        Choose between:

        - :func:`compute_consonance <biotuner.metrics.compute_consonance>`
        - :func:`dyad_similarity <biotuner.metrics.dyad_similarity>`
        - :func:`metric_denom <biotuner.metrics.metric_denom>`
    rounding : int
        maximum number of decimals for each step
    ratio_type : str, default='pos_harm'
        Choose between:

        - 'pos_harm':a/b when a>b
        - 'sub_harm':a/b when a<b
        - 'all': pos_harm + sub_harm

    Returns
    -------
    tuning_consonance : float
        Consonance value of the input tuning.
    mode_out : List of floats
        List of mode intervals.
    mode_consonance : float
        Consonance value of the output mode.

    Examples
    --------
    >>> tuning = [1, 1.21, 1.31, 1.45, 1.5, 1.7, 1.875]
    >>> harm_tuning, mode, harm_mode = tuning_reduction(tuning, mode_n_steps=5, function=dyad_similarity, rounding=4, ratio_type="pos_harm")
    >>> print('Tuning harmonicity: ', harm_tuning, '\nMode: ', mode, '\nMode harmonicity: ', harm_mode)
    Tuning harmonicity:  9.267212965965944
    Mode:  [1.5, 1, 1.875, 1.7, 1.45]
    Mode harmonicity:  17.9500338066261
    """
    # scale must be at least 4 notes
    tuning = sorted(tuning)
    if len(tuning) < 4:
        raise ValueError("The scale must have at least 4 notes")
    tuning_values = []
    mode_values = []
    for index1 in range(len(tuning)):
        for index2 in range(len(tuning)):
            if tuning[index1] != tuning[index2]:  # not include the diagonale
                if ratio_type == "pos_harm":
                    if tuning[index1] > tuning[index2]:
                        entry = tuning[index1] / tuning[index2]
                        mode_values.append([tuning[index1], tuning[index2]])
                        tuning_values.append(function(entry))
                if ratio_type == "sub_harm":
                    if tuning[index1] < tuning[index2]:
                        entry = tuning[index1] / tuning[index2]
                        mode_values.append([tuning[index1], tuning[index2]])
                        tuning_values.append(function(entry))
                if ratio_type == "all":
                    entry = tuning[index1] / tuning[index2]
                    mode_values.append([tuning[index1], tuning[index2]])
                    tuning_values.append(function(entry))
    if function == metric_denom:
        cons_ratios = [x for _, x in sorted(zip(tuning_values, mode_values))]
    else:
        cons_ratios = [x for _, x in sorted(zip(tuning_values, mode_values))][::-1]
    i = 0
    mode_ = []
    mode_out = []
    while len(mode_out) < mode_n_steps:
        cons_temp = cons_ratios[i]
        mode_.append(cons_temp)
        mode_out_temp = [item for sublist in mode_ for item in sublist]
        mode_out_temp = [np.round(x, rounding) for x in mode_out_temp]
        mode_out = sorted(set(mode_out_temp), key=mode_out_temp.index)[0:mode_n_steps]
        i += 1
    mode_metric = []
    for index1 in range(len(mode_out)):
        for index2 in range(len(mode_out)):
            if mode_out[index1] > mode_out[index2]:
                entry = mode_out[index1] / mode_out[index2]
                mode_metric.append(function(entry))
    tuning_consonance = np.average(tuning_values)
    mode_consonance = np.average(mode_metric)
    return tuning_consonance, mode_out, mode_consonance


def create_mode(tuning, n_steps, function):
    """Create a mode from a tuning based on the consonance of
       subsets of tuning steps.

    Parameters
    ----------
    tuning : List of floats
        scale to reduce
    n_steps : int
        number of steps of the reduced scale
    function : function, default=compute_consonance
        function used to compute the consonance between pairs of ratios
        Choose between:

        - :func:`compute_consonance <biotuner.metrics.compute_consonance>`
        - :func:`dyad_similarity <biotuner.metrics.dyad_similarity>`
        - :func:`metric_denom <biotuner.metrics.metric_denom>`

    Returns
    -------
    mode : List of floats
        Reduced tuning.

    Examples
    --------
    >>> tuning = [1, 1.21, 1.31, 1.45, 1.5, 1.7, 1.875]
    >>> create_mode(tuning, n_steps=5, function=dyad_similarity)
    [1, 1.45, 1.5, 1.7, 1.875]
    """
    sets = list(findsubsets(tuning, n_steps))
    metric_values = []
    for s in sets:
        _, met, _ = tuning_cons_matrix(s, function)
        metric_values.append(met)
    idx = np.argmax(metric_values)
    mode = list(sets[idx])
    return mode


def pac_mode(pac_freqs, n, function=dyad_similarity, method="subset"):
    """
    Compute the pac mode of a set of frequency pairs.

    Parameters
    ----------
    pac_freqs : List of tuples
        List of frequency pairs (f1, f2) representing phase-amplitude coupling.
    n : int
        Number of steps in the tuning system.
    function : function, default=dyad_similarity
        A function that takes two frequencies as input and returns a similarity score.
    method : str, default='subset'
        The method used to compute the pac mode.
        Possible values:

        - 'pairwise'
        - 'subset'

    Returns
    -------
    List
        The pac mode as a list of frequencies.
    """
    if method == "pairwise":
        _, mode, _ = tuning_reduction(
            scale_from_pairs(pac_freqs), mode_n_steps=n, function=function
        )
    if method == "subset":
        mode = create_mode(scale_from_pairs(pac_freqs), n_steps=n, function=function)
    return sorted(mode)


"""--------------------------MOMENTS OF SYMMETRY---------------------------"""

import sympy as sp


def tuning_range_to_MOS(frac1, frac2, octave=2, max_denom_in=100, max_denom_out=100):
    """
    Compute the Moment of Symmetry (MOS) signature for a range of ratios defined by two input fractions,
    and compute the generative interval for that range.

    The MOS signature of a ratio is a tuple of integers representing the number of equally spaced intervals
    that can fit into an octave when starting from the ratio, going in one direction, and repeating the
    interval until the octave is filled. For example, the MOS signature of an octave is (1,0) because there
    is only one interval that fits into an octave when starting from the ratio of 1:1 and going up.
    The MOS signature of a perfect fifth is (0,1) because there are no smaller intervals that fit into an
    octave when starting from the ratio of 3:2 and going up, but there is one larger interval that fits,
    which is the octave above the perfect fifth.

    The generative interval is the interval that corresponds to the mediant of the two input fractions.
    The mediant is the fraction that lies between the two input fractions and corresponds to the interval
    where small and large steps are equal.

    Parameters
    ----------
    frac1 : str or float
        First ratio as a string or float.
    frac2 : str or float
        Second ratio as a string or float.
    octave : float, default=2
        The ratio of an octave.
    max_denom_in : int, default=100
        Maximum denominator to use when converting the input fractions to rational numbers.
    max_denom_out : int, default=100
        Maximum denominator to use when approximating the generative interval as a rational number.

    Returns
    -------
    tuple
        A tuple containing:
        - the mediant as a float,
        - the mediant as a fraction with a denominator not greater than `max_denom_out`,
        - the generative interval as a float,
        - the generative interval as a fraction with a denominator not greater than `max_denom_out`,
        - the MOS signature of the generative interval as a tuple of integers,
        - the MOS signature of the inverse of the generative interval as a tuple of integers.
    """
    a = Fraction(frac1).limit_denominator(max_denom_in).numerator
    b = Fraction(frac1).limit_denominator(max_denom_in).denominator
    c = Fraction(frac2).limit_denominator(max_denom_in).numerator
    d = Fraction(frac2).limit_denominator(max_denom_in).denominator
    # print(a, b, c, d)
    mediant = (a + c) / (b + d)
    mediant_frac = sp.Rational((a + c) / (b + d)).limit_denominator(max_denom_out)
    gen_interval = octave ** (mediant)
    gen_interval_frac = sp.Rational(octave ** (mediant)).limit_denominator(
        max_denom_out
    )
    MOS_signature = [d, b]
    invert_MOS_signature = [b, d]
    return (
        mediant,
        mediant_frac,
        gen_interval,
        gen_interval_frac,
        MOS_signature,
        invert_MOS_signature,
    )


def stern_brocot_to_generator_interval(ratio, octave=2):
    """
    Converts a fraction in the stern-brocot tree to
    a generator interval for moment of symmetry tunings

    Parameters
    ----------
    ratio : float
        stern-brocot ratio
    octave : float, default=2
        Reference period.

    Returns
    -------
    gen_interval : float
        Generator interval

    """
    gen_interval = octave ** (ratio)
    return gen_interval


def gen_interval_to_stern_brocot(gen):
    """
    Convert a generator interval to fraction in the stern-brocot tree.

    Parameters
    ----------
    gen : float
        Generator interval.

    Returns
    -------
    root_ratio : float
        Fraction in the stern-brocot tree.

    """
    root_ratio = log2(gen)
    return root_ratio


def horogram_tree_steps(ratio1, ratio2, steps=10, limit=1000):
    ratios_list = [ratio1, ratio2]
    s = 0
    while s < steps:
        ratio3 = horogram_tree(ratio1, ratio2, limit)
        ratios_list.append(ratio3)
        ratio1 = ratio2
        ratio2 = ratio3
        s += 1
    frac_list = [ratio2frac(x) for x in ratios_list]
    return frac_list, ratios_list


def horogram_tree(ratio1, ratio2, limit):
    """
    Compute the next step of the horogram tree.

    Parameters
    ----------
    ratio1 : float
        First ratio input.
    ratio2 : float
        Second ratio input.
    limit : int
        Limit for the denominator of the fraction.

    Returns
    -------
    next_step : float
        Next step of the horogram tree.
    """
    a = Fraction(ratio1).limit_denominator(limit).numerator
    b = Fraction(ratio1).limit_denominator(limit).denominator
    c = Fraction(ratio2).limit_denominator(limit).numerator
    d = Fraction(ratio2).limit_denominator(limit).denominator
    next_step = (a + c) / (b + d)
    return next_step


def phi_convergent_point(ratio1, ratio2):
    """
    Compute the phi convergent point of two ratios.

    Parameters
    ----------
    ratio1 : float
        First ratio input.
    ratio2 : float
        Second ratio input.

    Returns
    -------
    convergent_point : float
        Phi convergent point of the two ratios.
    """
    Phi = (1 + 5**0.5) / 2
    a = Fraction(ratio1).limit_denominator(1000).numerator
    b = Fraction(ratio1).limit_denominator(1000).denominator
    c = Fraction(ratio2).limit_denominator(1000).numerator
    d = Fraction(ratio2).limit_denominator(1000).denominator
    convergent_point = (c * Phi + a) / (d * Phi + b)
    return convergent_point


def Stern_Brocot(n, a=0, b=1, c=1, d=1):
    """
    Compute the Stern-Brocot tree of a given depth.

    Parameters
    ----------
    n : int
        Depth of the tree.
    a, b, c, d : int
        Initial values for the Stern-Brocot recursion. Default is a=0, b=1, c=1, d=1.

    Returns
    -------
    list
        List of fractions in the Stern-Brocot tree.
    """
    if a + b + c + d > n:
        return 0
    x = Stern_Brocot(n, a + c, b + d, c, d)
    y = Stern_Brocot(n, a, b, a + c, b + d)
    if x == 0:
        if y == 0:
            return [a + c, b + d]
        else:
            return [a + c] + [b + d] + y
    else:
        if y == 0:
            return [a + c] + [b + d] + x
        else:
            return [a + c] + [b + d] + x + y


def generator_interval_tuning(interval=3 / 2, steps=12, octave=2, harmonic_min=0):
    """
    Function that takes a generator interval and derives a tuning based on its stacking.
    interval: float
        Generator interval
    steps: int, default=12
        Number of steps in the scale.
        When set to 12 --> 12-TET for interval 3/2
    octave: int, default=2
        Value of the octave
    """
    scale = []
    for s in range(steps):
        degree = interval**harmonic_min
        while degree > octave:
            degree = degree / octave
        while degree < octave / 2:
            degree = degree * octave
        scale.append(degree)
        harmonic_min += 1
    return sorted(scale), scale


def interval_exponents(interval, n_steps):
    list_intervals = []
    for n in range(n_steps):
        n += 1
        list_intervals.append(interval**n)
    return list_intervals


def interval_to_radian(interval):
    degree = 360 * (log2(interval))
    # print(degree)
    return math.radians(degree), degree


def tuning_to_radians(interval, n_steps):
    _, tuning = generator_interval_tuning(
        interval=interval, steps=n_steps, octave=2, harmonic_min=1
    )
    radians = []
    degrees = []
    # print(tuning)
    for step in tuning:
        rad, deg = interval_to_radian(step)
        radians.append(rad)
        degrees.append(deg)

    return radians, degrees


def tuning_MOS_info(interval=3 / 2, steps=12, octave=2):
    tuning, _ = generator_interval_tuning(
        interval=interval, steps=steps, octave=octave, harmonic_min=0
    )
    tuning = tuning + [2]
    tuning = np.round(tuning, 10)
    tuning = np.sort(list(set(tuning)))
    intervals = []
    intervals_frac = []
    for i in range(len(tuning)):
        try:
            interval_ = np.round(
                (1200 * log2(tuning[i + 1]) - 1200 * log2(tuning[i])), 3
            )
            intervals.append(interval_)
            intervals_frac.append(
                Fraction(tuning[i + 1] - tuning[i]).limit_denominator(100)
            )

        except:
            pass
    # print(distance)
    # print(intervals_frac)
    distances = list(Counter(intervals).keys())  # equals to list(set(words))
    steps = list(Counter(intervals).values())
    sL = [steps for _, steps in sorted(zip(distances, steps))]

    if len((set(intervals))) == 1:
        # print('Large and small steps are equal')
        Large = sL[0]
        small = sL[0]
    else:
        Large = sL[1]
        small = sL[0]
    # print(sL)
    return len(set(intervals)), Large, small, tuning, sorted(distances)[::-1]


def find_MOS(interval, max_steps=53, octave=2):
    steps = 2
    MOS = {
        "steps": [],
        "sig": [],
        "tuning": [],
        "distances": [],
        "distances_frac": [],
        "NTET": [],
        "harmsim": [],
        "matrix_harmsim": [],
        "stern_brocot_fracs": [],
    }
    while steps < max_steps:
        steps += 1
        n_gaps, L, s, tuning, distances = tuning_MOS_info(interval, steps, octave)
        if n_gaps == 2 or n_gaps == 1:
            stern = Fraction(log2(interval)).limit_denominator(steps)
            stern = [stern.numerator, stern.denominator]
            MOS["stern_brocot_fracs"].append(stern)
            MOS["steps"].append(steps)
            MOS["sig"].append([L, s])
            MOS["tuning"].append(tuning)
            MOS["distances"].append(distances)
            if n_gaps == 2:
                MOS["NTET"].append(False)
            if n_gaps == 1:
                MOS["NTET"].append(True)
            # MOS['distances_frac'].append(distances_frac)
    MOS_metrics = []
    MOS_harmsim = []
    for tuning in MOS["tuning"]:
        dict_metrics = tuning_to_metrics(tuning)
        MOS_metrics.append(dict_metrics)
        MOS["harmsim"].append(dict_metrics["harm_sim"])
        MOS["matrix_harmsim"].append(dict_metrics["matrix_harm_sim"])
    return MOS


def MOS_metric_harmonic_mean(MOS_dict, metric="harmsim"):
    total_steps = np.sum(MOS_dict["steps"])
    harm_mean = []
    for step, harmsim in zip(MOS_dict["steps"], MOS_dict[metric]):
        harm_mean_ = step * harmsim
        harm_mean.append(harm_mean_)
    harm_mean = np.sum(harm_mean) / total_steps
    return harm_mean


def generator_to_stern_brocot_fractions(gen, limit):
    stern_fraction = []
    for i in range(1, limit):
        stern = Fraction(log2(gen)).limit_denominator(i)
        stern = [stern.numerator, stern.denominator]
        stern_fraction.append(stern)
    stern_fraction = sorted(list(set(tuple(row) for row in stern_fraction)))
    return stern_fraction


def measure_symmetry(generator_interval, max_steps=20, octave=2):
    """
    Measure the maximum deviation in symmetry for a given generator interval.

    This function calculates the MOS scales for the given generator interval and determines
    the maximum deviation in symmetry for the scales.

    Parameters
    ----------
    generator_interval : int or float
        The generator interval for which MOS scales will be calculated.
    max_steps : int, default=20
        The maximum number of steps to consider for each MOS scale calculation.
    octave : int, default=2
        The octave size for which the MOS scales will be calculated.

    Returns
    -------
    float
        The maximum deviation in symmetry for the given generator interval.

    Examples
    --------
    >>> generator_interval = 3/2
    >>> measure_symmetry(generator_interval)
    """
    MOS = find_MOS(generator_interval, max_steps=max_steps, octave=octave)
    deviations = []
    for sig in MOS["sig"]:
        deviations.append([abs(s - np.mean(sig)) for s in sig])
    avg_deviations = []
    for i in range(max([len(sig) for sig in MOS["sig"]])):
        deviations_i = [d[i] for d in deviations if i < len(d)]
        if deviations_i:
            avg_deviations.append(np.mean(deviations_i))
    norm_deviations = [d / len(MOS["sig"]) for d in avg_deviations]
    max_deviation = max(norm_deviations)
    return max_deviation
