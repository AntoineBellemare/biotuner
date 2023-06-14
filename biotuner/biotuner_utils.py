#!bin/bash
import numpy as np
import matplotlib.pyplot as plt
import pytuning
import pyACA
from pytuning import *
import scipy.signal
import sympy as sp
import functools
import contfrac
import itertools
import operator
import sys
import bottleneck as bn
import pandas as pd
from scipy.stats import pearsonr
from fractions import Fraction
from functools import reduce
import math
from collections import Counter
from scipy.fftpack import rfft, irfft
from scipy.signal import detrend, welch, find_peaks, stft
from scipy.optimize import curve_fit
from pytuning.tuning_tables import create_scala_tuning
import os
os.environ['SDL_AUDIODRIVER'] = 'directsound'
try:
    from pyunicorn.timeseries.surrogates import *
    from pyunicorn.timeseries import RecurrenceNetwork
except ModuleNotFoundError:
    pass
try:
    from scipy.signal import butter, lfilter
except ModuleNotFoundError:
    pass
sys.setrecursionlimit(120000)
sample_rate = 44100

"""My utility functions.

This module contains a collection of utility functions for working with lists,
numbers, and strings.

Proportion Functions
-----------------
These functions are useful for manipulating proportions:

- `compute_peak_ratios(peaks, rebound=True, octave=2, sub=False)`: Flatten a nested list into a single list.
- `unique(lists)`: Remove duplicates from a list while preserving order.
- `split(lists, n)`: Split a list into n roughly equal parts.

Mathematical Functions
----------------------
These functions perform various mathematical operations:

- `add(x, y)`: Add two numbers together.
- `multiply(x, y)`: Multiply two numbers together.
- `divide(x, y)`: Divide one number by another.

"""


'''Proportion functions'''

def compute_peak_ratios(peaks, rebound=True, octave=2, sub=False):
    """This function calculates all the ratios
       (with the possibility to bound them between 1 and 2)
       derived from input peaks.

    Parameters
    ----------
    peaks : List (float)
        Peaks represent local maximum in a spectrum
    rebound : boolean
        Defaults to True. False will output unbounded ratios
    octave : int
        Arbitrary wanted number of octaves. Defaults to 2.
    sub : boolean
        Defaults to False. True will include ratios below the unison (1)

    Returns
    -------
    ratios_final: List
        List of frequency ratios.
    """
    # Iterating through successive peaks
    ratios = []
    peak_ratios_final = []
    for p1 in peaks:
        for p2 in peaks:

            # If a peak of value '0' is present, we skip this ratio computation
            if p1 < 0.1:
                p1 = 0.1
            try:
                ratio_temp = p2 / p1
            except:
                pass

            # When sub is set to False, only ratios with numerators
            # higher than denominators are consider
            if sub is False:
                if ratio_temp < 1:
                    ratio_temp = None
            # Ratios of 1 correspond to a peak divided by itself
            # and are therefore not considered
            if ratio_temp == 1:
                ratio_temp = None
            ratios.append(ratio_temp)

        # I imagine there's a reason why you reinit with array
        peak_ratios = np.array(ratios)
        peak_ratios = [i for i in peak_ratios if i]  # dealing with NaNs ?
        peak_ratios = sorted(list(set(peak_ratios)))
        ratios_final = peak_ratios.copy()

    # If rebound is given, all the ratios are constrained
    # between the unison and the octave
    if rebound is True:
        for peak in peak_ratios:

            # will divide the ratio by the octave until it reaches
            # a value under the octave.
            if peak > octave:
                while peak > octave:
                    peak = peak / octave
                peak_ratios_final.append(peak)
            # will multiply the ratio by the octave until it reaches
            # a value over the unison (1).
            if peak < octave:
                while peak < 1:
                    peak = peak * octave
                peak_ratios_final.append(peak)
        # Preparing output
        peak_ratios_final = np.array(peak_ratios_final)
        peak_ratios_final = [i for i in peak_ratios_final if i]
        ratios_final = sorted(list(set(peak_ratios_final)))
    return ratios_final


def scale2frac(scale, maxdenom=1000):
    """Transforms a scale provided as a list of floats into a list of fractions.

    Parameters
    ----------
    scale : List
        List of floats, typically between 1 and 2, representing scale steps.
    maxdenom : int
        Maximum denominator of the output fractions.

    Returns
    -------
    scale_frac : List
        List of fractions representing scale steps.
    num : array
        Numerators of all fractions.
    den : array
        Denominators of all fractions.

    """
    num = []
    den = []
    scale_frac = []
    for step in scale:
        frac = Fraction(step).limit_denominator(maxdenom)
        frac_ = sp.Rational(step).limit_denominator(maxdenom)
        num.append(frac.numerator)
        den.append(frac.denominator)
        scale_frac.append(frac_)
    return scale_frac, np.array(num), np.array(den)


def ratio2frac(ratio, maxdenom=1000):
    """Trasform a ratio expressed as a float in to a fraction.

    Parameters
    ----------
    ratio : float
        Frequency ratio, typically between 1 and 2.
    maxdenom : type
        Maximum denominator of the output fractions.

    Returns
    -------
    frac : List
        List of 2 int (numerator and denominator).

    """
    frac = Fraction(ratio).limit_denominator(maxdenom)
    num = frac.numerator
    den = frac.denominator
    frac = [num, den]
    return frac


def frac2scale(scale):
    """Transforms a scale expressed in fractions
       into a set of float ratios.

    Parameters
    ----------
    scale : List
        List of fractions.

    Returns
    -------
    scale_ratio : List
        Original scale expressed in float ratios.

    """
    scale_ratio = []
    for step in scale:
        scale_ratio.append(float(step))
    return scale_ratio


def chords_to_ratios(chords, harm_limit=2, spread=True):
    """Transforms series of frequencies expressed in float
       into series of integer ratios.

    Parameters
    ----------
    chords : List of lists
        Each sublist corresponds to a series of frequencies
        expressed in float.
    harm_limit : int
        Defaults to 2
        Maximum harmonic of the lower note below which the higher note
        should fall.
    spread : Boolean
        Defaults to True.
        When set to True, the harm_limit is applied to the previous note.
        When set to False, the harm_limit is applied to the first note.
        When harm_limit == 2, setting the spread to False will contain the
        chords within the span of one octave.

    Returns
    -------
    type
        Description of returned object.

    """
    chords_ratios = []
    chords_ratios_bounded = []
    for chord in chords:
        chord = sorted(chord)
        if harm_limit is not None:
            # allow each note to be within the defined
            # harm_limit of the previous note
            if spread is True:
                for note in range(len(chord)):
                    while chord[note] > chord[note - 1] * harm_limit:
                        chord[note] = chord[note] / 2
            # allow each note to be within the defined
            # harm_limit of the first note
            if spread is False:
                for note in range(len(chord)):
                    while chord[note] > chord[0] * harm_limit:
                        chord[note] = chord[note] / 2
        chord = sorted([np.round(n, 1) for n in chord])
        chord = [int(n * 10) for n in chord]
        # remove duplicates
        chord = list(dict.fromkeys(chord))
        gcd_chord = 2  # arbitrary number that is higher than 1
        while gcd_chord > 1:
            gcd_chord = gcd(*chord)
            if gcd_chord > 1:
                chord = [int(note / gcd_chord) for note in chord]
        chord_bounded = [c / chord[0] for c in chord]
        chords_ratios_bounded.append(chord_bounded)
        chords_ratios.append(chord)
    return chords_ratios, chords_ratios_bounded


def NTET_steps(octave, step, NTET):
    """This function computes the ratio associated with a specific step
       of a N-TET scale.

    Parameters
    ----------
    octave : int
        value of the octave
    step : int
        value of the step
    NTET : int
        number of steps in the N-TET scale

    Returns
    -------
    answer : float
        Step value.

    """

    answer = octave ** (step / NTET)
    return answer


def NTET_ratios(n_steps, max_ratio, octave=2):
    """Generate a series of ratios dividing the octave in N equal steps.

    Parameters
    ----------
    n_steps : int
        Number of steps dividing the octave.
    max_ratio : type
        Description of parameter `max_ratio`.
    octave : float
        Defaults to 2.
        Value of the octave (period).

    Returns
    -------
    steps_out : List
        List of steps in decimal numbers.

    """
    steps = []
    for s in range(n_steps):
        steps.append(octave ** (s / n_steps))
    steps_out = []
    for j in range(max_ratio - 1):
        steps_out.append([i + j for i in steps])
    steps_out = sum(steps_out, [])
    return steps_out


def scale_from_pairs(pairs):
    """Transforms each pairs of frequencies to ratios.

    Parameters
    ----------
    pairs : List of lists
        Each sublist is a pair of frequencies.

    Returns
    -------
    scale : List
        Scale steps.

    """
    scale = [rebound((x[1] / x[0])) for x in pairs]
    return scale


def ratios_harmonics(ratios, n_harms=1):
    """Computes the harmonics of ratios in the form of r*n when
       r is the ratio expressed as a float and n is the integer
       representing the harmonic position.

    Parameters
    ----------
    ratios : List
        List of ratios expressed as float.
    n_harms : int
        Number of harmonics to compute.

    Returns
    -------
    ratios_harms : List
        List of original ratios and their harmonics.

    """
    ratios_harms = []
    for h in range(n_harms):
        h += 1
        ratios_harms.append([i * h for i in ratios])
    ratios_harms = [i for sublist in ratios_harms for i in sublist]
    return ratios_harms


def ratios_increments(ratios, n_inc=1):
    """Computes the harmonics of ratios in the form of r**n when
       r is the ratio expressed as a float and n is the integer
       representing the increment (e.g. stacking ratios).

    Parameters
    ----------
    ratios : type
        Description of parameter `ratios`.
    n_inc : type
        Description of parameter `n_inc`.

    Returns
    -------
    ratios_harms : List
        List of original ratios and their harmonics.

    """
    ratios_harms = []
    for h in range(n_inc):
        h += 1
        ratios_harms.append([i**h for i in ratios])
    ratios_harms = [i for sublist in ratios_harms for i in sublist]
    ratios_harms = list(set(ratios_harms))
    return ratios_harms

def ratios2cents (ratios):
    """_summary_

    Parameters
    ----------
    ratios : List (float)
        List of frequency ratios

    Returns
    -------
    cents : List (float)
        List of corresponding cents
    """    
    cents = []
    for r in ratios:
        c = 1200 * np.log2 (r)
        cents.append(c)
    return cents


"""---------------------------LIST MANIPULATION--------------------------"""


def flatten(t):
    """Flattens a list of lists.

    Parameters
    ----------
    t : List of lists

    Returns
    -------
    List
        Flattened list.

    """
    return [item for sublist in t for item in sublist]


def findsubsets(S, m):
    """Find all subsets of m elements in a list.

    Parameters
    ----------
    S : List
        List to find subsets in.
    m : int
        Number of elements in each subset.

    Returns
    -------
    List of lists
        Each sublist represents a subset of the original list.

    """
    return list(set(itertools.combinations(S, m)))


def pairs_most_frequent(pairs, n):
    """Finds the most frequent values in a list of lists of pairs of numbers.

    Parameters
    ----------
    pairs : List of lists
        Each sublist should be a pair of values.
    n : int
        Number of most frequent pairs to keep.

    Returns
    -------
    List of lists
        First sublist corresponds to most frequent first values.
        Second sublist corresponds to most frequent second values.

    """
    drive_freqs = [x[0] for x in pairs]
    signal_freqs = [x[1] for x in pairs]
    drive_dict = {
        k: v for k, v in sorted(Counter(drive_freqs).items(),
                                key=lambda item: item[1])
    }
    max_drive = list(drive_dict)[::-1][0:n]
    signal_dict = {
        k: v for k, v in sorted(Counter(signal_freqs).items(),
                                key=lambda item: item[1])
    }
    max_signal = list(signal_dict)[::-1][0:n]
    return [max_signal, max_drive]


def rebound(x, low=1, high=2, octave=2):
    """Rescale a value within given range.

    Parameters
    ----------
    x : float
        Value to rebound.
    low : int
        Lower bound. Defaults to 1.
    high : int
        Higher bound. Defaults to 2.
    octave : int
        Value of an octave.

    Returns
    -------
    x : float
        Input value rescaled between low and high values.

    """

    while x >= high:
        x = x / octave
    while x <= low:
        x = x * octave
    return x


def rebound_list(x_list, low=1, high=2, octave=2):
    """Rescale a list within given range.

    Parameters
    ----------
    x : list
        List to rebound.
    low : int
        Lower bound. Defaults to 1.
    high : int
        Higher bound. Defaults to 2.
    octave : int
        Value of an octave.

    Returns
    -------
    x : list
        Rescaled list between low and high values.

    """
    return [rebound(x, low, high, octave) for x in x_list]


def sum_list(list):
    """Compute the sum of a list.

    Parameters
    ----------
    list : list
        List of values to sum.

    Returns
    -------
    sum : float
        Sum of the list.

    """
    sum = 0
    for x in list:
        sum += x
    return sum


def compareLists(list1, list2, bounds):
    """Find elements that are closest than bounds value from
       two lists.

    Parameters
    ----------
    list1 : list
        First list.
    list2 : list
        Second list.
    bounds : float
        Maximum value between two elements to assume equivalence.

    Returns
    -------
    matching : array
        Elements that match between the two list (average value of the two
        close elements).
    positions : list
        All indexes of the selected elements combined in one list.
    matching_pos : array (n_matching, 3)
        For each matching, a list of the value and the positions
        in list1 and list2.

    """
    matching = []
    matching_pos = []
    positions = []
    for i, l1 in enumerate(list1):
        for j, l2 in enumerate(list2):
            if l2 - bounds < l1 < l2 + bounds:
                matching.append((l1 + l2) / 2)
                matching_pos.append([(l1 + l2) / 2, i + 1, j + 1])
                positions.append(i + 1)
                positions.append(j + 1)
    matching = np.array(matching)
    matching_pos = np.array(matching_pos)
    ratios_temp = []
    for i in range(len(matching_pos)):
        if matching_pos[i][1] > matching_pos[i][2]:
            ratios_temp.append(matching_pos[i][1] / matching_pos[i][2])
        else:
            ratios_temp.append(matching_pos[i][2] / matching_pos[i][1])
    matching_pos_ratios = np.array(ratios_temp)
    return matching, list(set(positions)), matching_pos, matching_pos_ratios


def Print3Smallest(arr):
    MAX=np.max(arr)
    firstmin = MAX
    secmin = MAX
    thirdmin = MAX
    for i in range(0, len(arr)):

        # Check if current element
        # is less than firstmin,
        # then update first,second
        # and third

        if arr[i] < firstmin:
            thirdmin = secmin
            secmin = firstmin
            firstmin = arr[i]

        # Check if current element is
        # less than secmin then update
        # second and third
        elif arr[i] < secmin:
            thirdmin = secmin
            secmin = arr[i]

        # Check if current element is
        # less than,then update third
        elif arr[i] < thirdmin:
            thirdmin = arr[i]
    idx = [arr.index(firstmin), arr.index(secmin), arr.index(thirdmin)]
    mins = [firstmin, secmin, thirdmin]
    return mins, idx


def top_n_indexes(arr, n):
    """Find the index pairs of maximum values in a 2d array.

    Parameters
    ----------
    arr : ndarray(i, j)
        2d array.
    n : int
        Number of index pairs to return.

    Returns
    -------
    indexes : List of lists
        Each sublist contains the 2 indexes associated with higest values
        of the input array.

    """
    idx = bn.argpartition(arr, arr.size - n, axis=None)[-n:]
    width = arr.shape[1]
    indexes = [divmod(i, width) for i in idx]
    return indexes


def getPairs(peaks):
    """Extract all possible pairs of elements from a list.

    Parameters
    ----------
    peaks : List
        List of values.

    Returns
    -------
    out : List of lists
        Each sublist corresponds to a pair of value.

    """
    peaks_ = peaks.copy()
    out = []
    for i in range(len(peaks_) - 1):
        a = peaks_.pop(i - i)
        for j in peaks_:
            out.append([a, j])
    return out


"""--------------------SIMPLE MATHEMATICAL FUNCTIONS-----------------------"""


def nth_root(num, root):
    """This function computes the nth root of a number.

    Parameters
    ----------
    num : int
        value of the octave
    root : int
        number of steps in the N-TET scale

    Returns
    -------
    answer : float
        nth root.

    """
    answer = num ** (1 / root)
    return answer


def gcd(*numbers):
    """Return the greatest common divisor of the given integers
    The result will have the same sign as the last number given (so that
    when the last number is divided by the result, the result comes out
    positive).

    Parameters
    ----------
    *numbers : List
        List of numbers.

    Returns
    -------
    a : float
        Greated common divisor of the input numbers.

    """

    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    return reduce(gcd, numbers)


def reduced_form(*numbers):
    """Return a tuple of numbers which is the reduced form of the input,
       which is a list of integers. Ex: [4, 8, 12, 80] -> [1, 2, 3, 20]

    Parameters
    ----------
    *numbers : list (float)
        List of numbers to reduce.

    Returns
    -------
    reduce : list (int)
        Reduced form of the input.

    """
    reduce = tuple(int(a // gcd(*numbers)) for a in numbers)
    return reduce


def lcm(*numbers):
    """Return the least common multiple of the given integers.

    Parameters
    ----------
    *numbers : list (float)
        List of numbers.

    Returns
    -------
    lcm_ : int
        Least common multiple of the input numbers.

    """
    def lcm(a, b):
        if a == b == 0:
            return 0
        return (a * b) // gcd(a, b)
    lcm_ = reduce(lcm, numbers)
    return lcm_


def prime_factors(n):
    """Return a list of the prime factors of the integer n.
       Don't use this for big numbers; it's a dumb brute-force method.

    Parameters
    ----------
    n : int
        Integer to factorize.

    Returns
    -------
    factors : list
        List of prime factors of n.

    """
    factors = []
    lastresult = n
    while lastresult > 1:
        c = 2
        while lastresult % c > 0:
            c += 1
        factors.append(c)
        lastresult /= c
    return factors


def prime_factor(n):
    """Find the prime number in a list of n numbers.

    Parameters
    ----------
    n : type
        Description of parameter `n`.

    Returns
    -------
    type
        Description of returned object.

    """
    prime = []
    for i in n:
        flag = 0
        if i == 1:
            flag = 1
        for j in range(2, i):
            if i % j == 0:
                flag = 1
                break
        if flag == 0:
            prime.append(i)
    return prime


def contFrac(x, k):
    """Compute the continuous fraction of a value x for k steps.

    Parameters
    ----------
    x : float
        Value to decompose.
    k : int
        Number of steps to go through.

    Returns
    -------
    cf : list
        List of denominators (1/cf[0] + 1/cf[1] ... +1/cf[k]).

    """
    cf = []
    q = math.floor(x)
    cf.append(q)
    x = x - q
    i = 0
    while x != 0 and i < k:
        q = math.floor(1 / x)
        cf.append(q)
        x = 1 / x - q
        i = i + 1
    return cf

def compute_IMs(f1, f2, n):
    """
    InterModulation components: sum or subtraction of any non-zero integer
    multiple of the input frequencies.

    Parameters
    ----------
    f1 : float
        Frequency 1.
    f2 : float
        Frequency 2.
    n : int
        Order of the intermodulation component.

    Returns
    -------
    IMs : List
        List of all intermodulation components.
    order : List
        Order associated with IMs.

    Examples
    --------
    >>> f1 = 3
    >>> f2 = 12
    >>> n = 2
    >>> IMs, order = compute_IMs(f1, f2, n)
    >>> IMs, order
    ([9, 15, 21, 27, 6, 18, 30],
    [(1, 1), (1, 1), (1, 2), (1, 2), (2, 1), (2, 1), (2, 2)])
    """
    IMs = []
    orders = []
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            #print(j)
            #print(f1 * j + f2 * i)
            IM_add = f1 * j + f2 * i
            if IM_add not in IMs:
                IMs.append(IM_add)
                orders.append((j, i))
            IM_sub = np.abs(f1 * j - f2 * i)
            if IM_sub not in IMs:
                IMs.append(IM_sub)
                orders.append((j, i))
    IMs = [x for _, x in sorted(zip(orders, IMs))]
    orders = sorted(orders)
    return IMs, orders

"""------------------------------SURROGATES--------------------------------"""


def phaseScrambleTS(ts):
    """Returns a TS: original TS power is preserved; TS phase is shuffled.

    Parameters
    ----------
    ts : 1d array (numDataPoints, )
        Oriinal time series.

    Returns
    -------
    tsr : array (numDataPoints)
        Phase scrabled time series.

    """
    fs = rfft(ts)
    # rfft returns real and imaginary components
    # in adjacent elements of a real array
    pow_fs = fs[1:-1:2] ** 2 + fs[2::2] ** 2
    phase_fs = np.arctan2(fs[2::2], fs[1:-1:2])
    phase_fsr = phase_fs.copy()
    np.random.shuffle(phase_fsr)
    # use broadcasting and ravel to interleave
    # the real and imaginary components.
    # The first and last elements in the fourier array don't have any
    # phase information, and thus don't change
    fsrp = np.sqrt(pow_fs[:, np.newaxis]) * np.c_[np.cos(phase_fsr),
                                                  np.sin(phase_fsr)]
    fsrp = np.r_[fs[0], fsrp.ravel(), fs[-1]]
    tsr = irfft(fsrp)
    return tsr


def AAFT_surrogates(original_data):
    """Return surrogates using the amplitude adjusted Fourier transform
       method.

    Parameters
    ----------
    original_data : 2D array (data, index)
        The original time series.

    Returns
    -------
    rescaled_data : 2D array (surrogate, index)
    """
    #  Create sorted Gaussian reference series
    gaussian = np.random.randn(original_data.shape[0], original_data.shape[1])
    gaussian.sort(axis=1)

    #  Rescale data to Gaussian distribution
    ranks = original_data.argsort(axis=1).argsort(axis=1)
    rescaled_data = np.zeros(original_data.shape)

    for i in range(original_data.shape[0]):
        rescaled_data[i, :] = gaussian[i, ranks[i, :]]

    #  Phase randomize rescaled data
    phase_randomized_data = correlated_noise_surrogates(rescaled_data)

    #  Rescale back to amplitude distribution of original data
    sorted_original = original_data.copy()
    sorted_original.sort(axis=1)

    ranks = phase_randomized_data.argsort(axis=1).argsort(axis=1)

    for i in range(original_data.shape[0]):
        rescaled_data[i, :] = sorted_original[i, ranks[i, :]]

    return rescaled_data


def correlated_noise_surrogates(original_data):
    """
    Return Fourier surrogates.

    Generate surrogates by Fourier transforming the :attr:`original_data`
    time series (assumed to be real valued), randomizing the phases and
    then applying an inverse Fourier transform. Correlated noise surrogates
    share their power spectrum and autocorrelation function with the
    original_data time series.

    .. note::
       The amplitudes are not adjusted here, i.e., the
       individual amplitude distributions are not conserved!
    """
    #  Calculate FFT of original_data time series
    #  The FFT of the original_data data has to be calculated only once,
    #  so it is stored in self._original_data_fft.
    surrogates = np.fft.rfft(original_data, axis=1)
    #  Get shapes
    (N, n_time) = original_data.shape
    len_phase = surrogates.shape[1]

    #  Generate random phases uniformly distributed in the
    #  interval [0, 2*Pi]
    phases = np.random.uniform(low=0, high=2 * np.pi, size=(N, len_phase))

    #  Add random phases uniformly distributed in the interval [0, 2*Pi]
    surrogates *= np.exp(1j * phases)

    #  Calculate IFFT and take the real part, the remaining imaginary part
    #  is due to numerical errors.
    return np.ascontiguousarray(np.real(np.fft.irfft(surrogates,
                                                     n=n_time,
                                                     axis=1)))


""""""


def UnivariateSurrogatesTFT(data_f, MaxIter=1, fc=5):
    """Compute Truncated Fourier Transform of the input data.
       from: https://github.com/narayanps/NolinearTimeSeriesAnalysis/blob/
       master/SurrogateModule.py

    Parameters
    ----------
    data_f : array
        Original signal.
    MaxIter : int
        Defaults to 1.
        Maximum number of iterations.
    fc : int
        Defaults to 5.
        Description of parameter `fc`.

    Returns
    -------
    xsur : array
        Surrogate of the original signal.

    """

    xs = data_f.copy()
    xs.sort()  # sorted amplitude stored
    # amplitude of fourier transform of orig
    pwx = np.abs(np.fft.fft(data_f))
    phi = np.angle(np.fft.fft(data_f))
    Len = phi.shape[0]
    # data_f.shape=(-1,1)
    # random permutation as starting point
    xsur = np.random.permutation(data_f)
    # xsur.shape = (1,-1)
    Fc = np.round(fc * data_f.shape[0])
    for i in range(MaxIter):
        phi_surr = np.angle(np.fft.fft(xsur))
        # print(phi_surr.shape)
        # print(phi.shape)
        phi_surr[1:Fc] = phi[1:Fc]
        phi_surr[Len - Fc + 1: Len] = phi[Len - Fc + 1: Len]
        phi_surr[0] = 0.0
        new_len = int(Len / 2)

        phi_surr[new_len] = 0.0

        fftsurx = pwx * np.exp(1j * phi_surr)
        xoutb = np.real(np.fft.ifft(fftsurx))
        ranks = xoutb.argsort(axis=0)
        xsur[ranks] = xs
    return xsur


"""----------------------SPECTROMORPHOLOGY FUNCIONS------------------------"""


def computeFeatureCl_new(
    afAudioData,
    cFeatureName,
    f_s,
    window=4000,
    overlap=1
     ):
    """Calculate spectromorphological metrics on time series.

    Parameters
    ----------
    afAudioData : array (numDataPoints, )
        Input signal.
    cFeatureName : str
        {'SpectralCentroid', 'SpectralCrestFactor', 'SpectralDecrease',
         'SpectralFlatness', 'SpectralFlux', 'SpectralKurtosis',
         'SpectralMfccs', 'SpectralPitchChroma', 'SpectralRolloff',
         'SpectralSkewness', 'SpectralSlope', 'SpectralSpread',
         'SpectralTonalPowerRatio', 'TimeAcfCoeff', 'TimeMaxAcf',
         'TimePeakEnvelope', 'TimeRms', 'TimeStd', 'TimeZeroCrossingRate'}
    f_s : int
        Sampling frequency.
    window : int
        Length of the moving window in samples.
    overlap : int
        Overlap between each moving window in samples.

    Returns
    -------
    v : array
        Vector of the spectromorphological metric.
    t : array
        Timestamps.
    """
    [v, t] = pyACA.computeFeature(
                   cFeatureName,
                   afAudioData,
                   f_s,
                   None,
                   window,
                   overlap)
    return (v, t)


def EMD_to_spectromorph(
    IMFs,
    sf,
    method="SpectralCentroid",
    window=None,
    overlap=1,
    in_cut=None,
    out_cut=None,
):
    """Calculate spectromorphological metrics on intrinsic mode functions
       derived using Empirical Mode Decomposition.

    Parameters
    ----------
    IMFs : array (nIMFs, numDataPoints)
        Input data.
    sf : int
        Sampling frequency of the original signal.
    method : str
        Defaults to 'SpectralCentroid'.
        {'SpectralCentroid', 'SpectralCrestFactor', 'SpectralDecrease',
         'SpectralFlatness', 'SpectralFlux', 'SpectralKurtosis',
         'SpectralMfccs', 'SpectralPitchChroma', 'SpectralRolloff',
         'SpectralSkewness', 'SpectralSlope', 'SpectralSpread',
         'SpectralTonalPowerRatio', 'TimeAcfCoeff', 'TimeMaxAcf',
         'TimePeakEnvelope', 'TimeRms', 'TimeStd', 'TimeZeroCrossingRate'}
    window : int
        Length of the moving window in samples.
    overlap : int
        Overlap between each moving window in samples.
    in_cut : int
        Number of samples to remove at the beginning.
    out_cut : type
        Number of samples to remove at the end.

    Returns
    -------
    spectro_IMF : (nIMFs, numSpectroPoints)
        Spectromorphological metric for each IMF.

    """
    # remove 0.1 second at the beginning and the end of the signal.
    if in_cut is None:
        in_cut = int(sf / 10)
    if out_cut is None:
        out_cut = int(len(IMFs[0]) - (sf / 10))
    if window is None:
        window = int(sf / 2)
    spectro_IMF = []
    for e in IMFs:
        f, t = computeFeatureCl_new(e, method, sf, window, overlap)
        try:
            spectro_IMF.append(f[0][in_cut:out_cut])
        except:
            spectro_IMF.append(f[in_cut:out_cut])
    spectro_IMF = np.array(spectro_IMF)
    return spectro_IMF


"""-------------------GENERATE AUDIO / SIGNAL PROCESSING--------------------"""


def generate_signal(
    sf, time_end, freqs, amps, show=False, theta=0, color="blue"
     ):
    time = np.arange(0, time_end, 1 / sf)
    sine_tot = []
    for i in range(len(freqs)):
        sinewave = amps[i] * np.sin(2 * np.pi * freqs[i] * time + theta)
        sine_tot.append(sinewave)
    sine_tot = sum_list(sine_tot)
    if show is True:
        ax = plt.gca()
        ax.set_facecolor("xkcd:black")
        plt.plot(time, sine_tot, color=color)
    return sine_tot



def smooth(x, window_len=11, window="hanning"):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: dimension of the smoothing window; should be an odd integer
        window: type of window
                {'flat', 'hanning', 'hamming', 'bartlett', 'blackman'}
                flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    NOTE: length(output) != length(input), to correct this:
          return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    if window == "flat":  # moving average
        w = np.ones(window_len, "d")
    else:
        w = eval("np." + window + "(window_len)")

    y = np.convolve(w / w.sum(), s, mode="valid")
    return y


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def major_triad(hz):
    return make_chord(hz, [4, 5, 6])

pygame_lib = None

def make_chord(hz, length=1):
    sampling=44100  # this is your sampling rate
    t = np.linspace(0, length, int(sampling * length), False)
    chord = np.sin(hz * t * 2 * np.pi)
    chord *= 4096  # Adjust volume
    # Expand chord to 2D array for stereo compatibility
    chord_stereo = np.vstack((chord, chord)).T
    # Ensure the array is C-contiguous
    chord_stereo = np.ascontiguousarray(chord_stereo)
    return chord_stereo.astype(np.int16)

def play_chord(chord):
    global pygame_lib
    if pygame_lib is None:
        import pygame
        pygame_lib = pygame
    pygame_lib.mixer.init(frequency=44100, size=-16, channels=2)  # Initialize as stereo
    sound = pygame_lib.sndarray.make_sound(chord)
    sound.play()
    
    
def listen_chords(chords, mult=1, duration=1):
    global pygame_lib
    if pygame_lib is None:
        import pygame
        pygame_lib = pygame   
    pygame_lib.init()
    pygame_lib.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
    for c in chords:
        hz = c[0] * mult  # The fundamental frequency
        # The remaining entries in the list are ratios
        ratios = [i / c[0] for i in c]
        
        # Create a chord based on the ratios and fundamental frequency
        chord = np.sum([make_chord(hz * r, duration) for r in ratios], axis=0)
        
        # Ensure the array is C-contiguous
        chord = np.ascontiguousarray(chord)
        
        # Play the chord
        play_chord(chord)
        
        # Wait for the duration of the chord before playing the next one
        pygame_lib.time.wait(int(duration * 1000))


def listen_scale(scale, fund, duration=1):
    global pygame_lib
    if pygame_lib is None:
        import pygame
        pygame_lib = pygame   
    pygame_lib.init()
    pygame_lib.mixer.init(frequency=44100, size=-16, channels=1)
    
    # Add 1 at the beginning of the scale to include the fundamental frequency
    scale = [1] + scale
    
    for s in scale:
        freq = fund * s
        
        # Create a note based on the frequency
        note = make_chord(freq, duration)
        
        # Ensure the array is C-contiguous
        note = np.ascontiguousarray(note)
        
        # Play the note
        play_chord(note)
        
        # Wait for the duration of the note before playing the next one
        pygame_lib.time.wait(int(duration * 1000))


        

def frequency_to_note(frequency):
    # define constants that control the algorithm
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'] # these are the 12 notes in each octave
    OCTAVE_MULTIPLIER = 2 # going up an octave multiplies by 2
    KNOWN_NOTE_NAME, KNOWN_NOTE_OCTAVE, KNOWN_NOTE_FREQUENCY = ('A', 4, 440) # A4 = 440 Hz

    # calculate the distance to the known note
    # since notes are spread evenly, going up a note will multiply by a constant
    # so we can use log to know how many times a frequency was multiplied to get from the known note to our note
    # this will give a positive integer value for notes higher than the known note, and a negative value for notes lower than it (and zero for the same note)
    note_multiplier = OCTAVE_MULTIPLIER**(1/len(NOTES))
    frequency_relative_to_known_note = frequency / KNOWN_NOTE_FREQUENCY
    distance_from_known_note = math.log(frequency_relative_to_known_note, note_multiplier)

    # round to make up for floating point inaccuracies
    distance_from_known_note = round(distance_from_known_note)

    # using the distance in notes and the octave and name of the known note,
    # we can calculate the octave and name of our note
    # NOTE: the "absolute index" doesn't have any actual meaning, since it doesn't care what its zero point is. it is just useful for calculation
    known_note_index_in_octave = NOTES.index(KNOWN_NOTE_NAME)
    known_note_absolute_index = KNOWN_NOTE_OCTAVE * len(NOTES) + known_note_index_in_octave
    note_absolute_index = known_note_absolute_index + distance_from_known_note
    note_octave, note_index_in_octave = note_absolute_index // len(NOTES), note_absolute_index % len(NOTES)
    note_name = NOTES[note_index_in_octave]
    return (note_name, note_octave)

"""------------------------------MIDI-----------------------------------"""


import mido
from mido import Message, MidiFile, MidiTrack
import math

def create_midi(chords, durations, microtonal=True, filename='example'):
    """
    Creates a MIDI file from a given set of chords and durations.
    Args:
        chords (list): List of chords, where each chord is a list of frequencies.
        durations (list): List of durations (in beats) for each chord.
        microtonal (bool): Indicates whether to include microtonal pitch bends (default: True).
        filename (str): Name of the output MIDI file (default: 'example').
    Returns:
        mid (MidiFile): The created MIDI file object.
    """
    # Create a new MIDI file
    mid = MidiFile()

    # Set the tempo
    track = MidiTrack()
    track.append(Message('control_change', control=81, value=120))
    mid.tracks.append(track)

    def frequency_to_midi(chords):
        midi_chords = []
        midi_pitchbends = []
        for chord in chords:
            midi_notes = []
            pitchbends = []
            for frequency in chord:
                # Convert frequency to MIDI note
                midi_note = 69 + 12*math.log2(frequency/440)
                rounded_midi_note = int(midi_note)
                # Check the limits of the MIDI note
                if 0 < rounded_midi_note < 127:
                    rounded_midi_frequency = 440 * 2**((rounded_midi_note - 69)/12)
                    #pitch_bend = int((frequency-rounded_midi_frequency)*8192/100)
                    pitch_bend = int((midi_note - rounded_midi_note) * (8192))
                    midi_notes.append(rounded_midi_note)
                    pitchbends.append(pitch_bend)
                else:
                    continue
            midi_chords.append(midi_notes)
            midi_pitchbends.append(pitchbends)
        return midi_chords, midi_pitchbends

    
    midi_chords, pitchbends = frequency_to_midi(chords)

    # Find the maximum number of notes in any chord
    max_notes = max(len(chord) for chord in midi_chords)

    # Create a fixed number of tracks equal to the maximum number of notes in any chord
    tracks = [MidiTrack() for _ in range(max_notes)]
    for track in tracks:
        mid.tracks.append(track)

    # Iterate through the chords and durations
    current_time = 0
    for chord, duration, pitchbend in zip(midi_chords, durations, pitchbends):
        for i, (note, pb) in enumerate(zip(chord, pitchbend)):
            track = tracks[i]

            # Add a pitch bend message for each note in the chord
            if microtonal is True:
                #print('pitchweel', pb)
                track.append(Message('pitchwheel', pitch=pb, channel=i, time=current_time))

            track.append(Message('note_on', note=note, velocity=64, channel=i, time=current_time))
            track.append(Message('note_off', note=note, velocity=64, channel=i, time=current_time+(duration*480)))
        current_time = current_time + (duration * 480)

    # Save the MIDI file
    mid.save(str(filename)+'.mid')
    return mid


"""-----------------------------OTHERS----------------------------------"""


def create_SCL(scale, fname):
    """Save a scale to .scl file.

    Parameters
    ----------
    scale : list
        List of scale steps.
    fname : str
        Name of the saved file.

    """
    table = create_scala_tuning(scale, fname)
    outF = open(fname + ".scl", "w")
    outF.writelines(table)
    outF.close()
    return


def scale_interval_names(scale, reduce=False):
    """Gives the name of intervals in a scale based on PyTuning dictionary.

    Parameters
    ----------
    scale : list
        List of scale steps either in float or fraction form.
    reduce : boolean
        Defaults to False.
        When set to True, output only the steps that match a key
        in the dictionary.

    Returns
    -------
    interval_names : list of lists
        Each sublist contains the scale step and the corresponding
        interval names.

    """
    try:
        type = scale[0].dtype == "float64"
        if type is True:
            scale, _, _ = scale2frac(scale)
    except:
        pass
    interval_names = []
    for step in scale:
        name = pytuning.utilities.ratio_to_name(step)
        if reduce is True and name is not None:
            interval_names.append([step, name])
        if reduce is False:
            interval_names.append([step, name])
    return interval_names


def calculate_pvalues(df, method='pearson'):
    """Calculate the correlation between each column of a DataFrame.

    Parameters
    ----------
    df : DataFrame
        DataFrame of values to compute correlation on.
    method : str
        Defaults to pearson.
        {'pearson', 'spearman'}

    Returns
    -------
    type
        Description of returned object.

    """
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how="outer")
    for r in df.columns:
        for c in df.columns:
            if method == 'pearson':
                pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
            if method == 'spearman':
                pvalues[r][c] = round(spearmanr(df[r], df[c])[1], 4)
    return pvalues


def peaks_to_amps(peaks, freqs, amps, sf):
    """Find the amplitudes of spectral peaks.

    Parameters
    ----------
    peaks : list
        Spectral peaks in Hertz.
    freqs : list
        All centers of frequency bins.
    amps : list
        All amplitudes associated with frequency bins.
    sf : int
        Sampling frequency.

    Returns
    -------
    amps_out : list
        Amplitudes of spectral peaks.

    """
    bin_size = np.round((sf / 2) / len(freqs), 3)
    amps_out = []
    for p in peaks:
        index = int(p / bin_size)
        amp = amps[index]
        amps_out.append(amp)
    return amps_out


def alpha2bands(a):
    """Derive frequency bands for M/EEG analysis based on the alpha peak.
       Boundaries of adjacent frequency bands are derived based on the
       golden ratio, which optimizes the uncoupling of phases between
       frequencies. (see: Klimesch, 2018)

    Parameters
    ----------
    a : float
        Alpha peak in Hertz.

    Returns
    -------
    FREQ_BANDS : List of lists
        Each sublist contains the boundaries of each frequency band.

    """
    np.float(a)
    center_freqs = [a / 4, a / 2, a, a * 2, a * 4]
    FREQ_BANDS = []
    for f in center_freqs:
        down = np.round((f / 2) * 1.618, 1)
        up = np.round((f * 2) * 0.618, 1)
        band = [down, up]
        FREQ_BANDS.append(band)
    return FREQ_BANDS


def chunk_ts(data, sf, overlap=10, precision=1):
    """Divide a time series into overlapping chunks and provide the indexes.

    Parameters
    ----------
    data : ndarray(1d)
        Time series.
    sf : int
        Sampling frequency.
    overlap : int
        Defaults to 10.
        Proportion of overlap in percentage.
    precision : float
        Defaults to 1.
        Precision in Hertz.
        The precision here determines the length of data
        required to extract the spectral information at that precise frequency.


    Returns
    -------
    type
        Description of returned object.

    """
    overlap = 100/overlap
    nsec = len(data)/sf
    chunk_size = int((1/precision)*sf)
    overlap_samp = int((chunk_size)/overlap)
    i = 0
    pairs = []
    while i < len(data):
        pairs.append((i, i+chunk_size))
        i = i+chunk_size-overlap_samp
    return pairs


def string_to_list(string):
    # Remove the brackets and extra spaces, then split the string into elements
    str_list = string.strip('[]').split()

    # Convert the elements to floats
    float_list = [float(i) for i in str_list]
    return float_list

def safe_max(lst):
    if isinstance(lst, (list, np.ndarray)):
        return np.max(lst) if len(lst) > 0 else np.nan
    elif isinstance(lst, float):
        return lst  # return the float itself
    else:
        return np.nan  # or return any other value you see fit for non-list and non-float entries
    
def safe_mean(lst):
    if isinstance(lst, str):
        lst = string_to_list(lst)
    if isinstance(lst, (list, np.ndarray)):
        return np.mean(lst) if len(lst) > 0 else np.nan
    elif isinstance(lst, float):
        return lst  # return the float itself if only one number is present
    else:
        return np.nan  # or return any other value you see fit for non-list and non-float entries
    
def compute_frequency_and_psd(signal, precision_hz, smoothness, fs, noverlap, fmin=None, fmax=None):
    """
    Compute the frequencies and power spectral density (PSD) of a signal using Welch method.

    Parameters
    ----------
    signal : ndarray
        Input signal.
    precision_hz : float
        Precision in Hz for the frequencies.
    smoothness : float
        Smoothing factor for the PSD.
    fs : float
        Sampling frequency of the signal.
    noverlap : int
        Number of points to overlap between segments.
    fmin : float, optional
        Minimum frequency to compute.
    fmax : float, optional
        Maximum frequency to compute.

    Returns
    -------
    freqs : ndarray
        Frequencies for which the PSD is computed.
    psd : ndarray
        Power spectral density of the signal.
    """
    nperseg = int(fs / precision_hz)
    freqs, psd = welch(signal, fs, nperseg=int(nperseg/smoothness), nfft=nperseg, noverlap=noverlap)
    if fmin is not None or fmax is not None:
        mask = np.ones(freqs.shape, dtype=bool)
        if fmin is not None:
            mask &= (freqs >= fmin)
        if fmax is not None:
            mask &= (freqs <= fmax)
        freqs = freqs[mask]
        psd = psd[mask]
    return freqs, psd


def power_law(x, a, b):
    """
    Define a power law function.

    Parameters
    ----------
    x : ndarray
        Independent variable.
    a : float
        Scaling factor.
    b : float
        Power.

    Returns
    -------
    ndarray
        Values after applying power law function.
    """
    return a * (x ** b)


def apply_power_law_remove(freqs, psd, power_law_remove):
    """
    Apply or not a power law to remove it from PSD.

    Parameters
    ----------
    freqs : ndarray
        Frequencies for which the PSD is computed.
    psd : ndarray
        Power spectral density of the signal.
    power_law_remove : bool
        If True, apply the power law. Otherwise, return the input PSD.

    Returns
    -------
    ndarray
        PSD after potential power law removal.
    """
    if power_law_remove:
        popt, _ = curve_fit(power_law, freqs, psd, maxfev=5000)
        return psd - power_law(freqs, *popt)
    else:
        return psd
    
    
def __get_norm(norm):
    ''''''
    if norm == 0 or norm is None:
        return None, None
    else:
        try:
            norm1, norm2 = norm
        except TypeError:
            norm1 = norm2 = norm
        return norm1, norm2


def __freq_ind(freq, f0):
    try:
        return [np.argmin(np.abs(freq - f)) for f in f0]
    except TypeError:
        return np.argmin(np.abs(freq - f0))


def __product_other_freqs(spec, indices, synthetic=(), t=None):
    p1 = np.prod(
        [
            amplitude * np.exp(2j * np.pi * freq * t + phase)
            for (freq, amplitude, phase) in synthetic
        ],
        axis=0,
    )
    p2 = np.prod(spec[:, indices[len(synthetic):]], axis=1)
    return p1 * p2


def functools_reduce(a):
    return functools.reduce(operator.concat, a)
