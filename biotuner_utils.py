#!bin/bash
"""Collection of operations on timeseries."""
import numpy as np
from PyEMD import EMD, EEMD
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import seaborn as sbn
from scipy import stats


def compute_peak_ratios(peaks, rebound=True, octave=2, sub=True):
    """
    Ratios between peaks.

    This function calculates the 15 ratios (with the possibility to bound them
    between 1 and 2) derived from the 6 peaks.

    peaks: List (type?)
        peaks represent blahblah.
    rebound: boolean
        Defaults to True. False will blahblah.
    ocatve: int
        Arbitrary wanted number of octaves. Defaults to 2.
    sub: boolean
        Defaults to True. blahblah.
    """
    # What are we doing here ? iterating through successive peaks
    ratios = []
    peak_ratios_rebound = []
    for p1 in peaks:
        for p2 in peaks:

            # Which problem do you avoid here ?
            try:
                ratio_temp = p2/p1
            except ZeroDivisionError:
                ratio_temp = p2/0.01

            # comment what initialization you do in 1 line
            if sub is False:
                if ratio_temp < 1:
                    ratio_temp = None
            # comment what initialization you do in 1 line
            if ratio_temp == 1:
                ratio_temp = None
            ratios.append(ratio_temp)

        # I imagine there's a reason why you reinit with array
        peak_ratios = np.array(ratios)
        peak_ratios = [i for i in peak_ratios if i]  # dealing with NaNs ?
        peak_ratios = list(set(peak_ratios))

    # If rebound is given, blahblah
    if rebound is True:
        for peak in peak_ratios:
            # will do this if a given peak ratio is over specified octave
            if peak > octave:
                while peak > octave:
                    peak = peak/octave
                peak_ratios_rebound.append(peak)
            # will do this if a given peak ratio is under specified octave
            if peak < octave:
                while peak < 1:
                    peak = peak*octave
                peak_ratios_rebound.append(peak)

    # Preparing outputs
    peak_ratios_rebound = np.array(peak_ratios_rebound)
    peak_ratios_rebound = [i for i in peak_ratios_rebound if i]
    peak_ratios = sorted(peak_ratios)
    peak_ratios_rebound = sorted(list(set(peak_ratios_rebound)))

    return peak_ratios, peak_ratios_rebound


def rebound(x, low=1, high=2, octave=2):
    """
    Recalculates x based on given octave bounds.

    x: int
        represents a peak value
    ...
    """
    while x > high:
        x = x/octave
    while x < low:
        x = x*octave
    return x

def nth_root (num, root):
    answer = num**(1/root)
    return answer


#Function that compares lists (i.e. peak harmonics)
def compareLists(list1, list2, bounds):
    matching = []
    matching_pos = []
    for i, l1 in enumerate(list1):
        for j, l2 in enumerate(list2):
            if l2-bounds < l1 < l2+bounds:
                matching.append((l1+l2)/2)
                matching_pos.append([(l1+l2)/2, i+1, j+1])
    matching = np.array(matching)
    matching_pos = np.array(matching_pos)
    ratios_temp = []
    for i in range(len(matching_pos)):
        if matching_pos[i][1]>matching_pos[i][2]:
            ratios_temp.append(matching_pos[i][1]/matching_pos[i][2])
            #print(matching_pos[i][0])
        else:
            ratios_temp.append(matching_pos[i][2]/matching_pos[i][1])
    matching_pos_ratios = np.array(ratios_temp)
    return matching, matching_pos, matching_pos_ratios

def create_SCL(scale, fname):
    #Output SCL files
    from pytuning.scales.pythagorean import create_pythagorean_scale
    from pytuning.tuning_tables import create_scala_tuning
    #scale = create_pythagorean_scale()
    table = create_scala_tuning(scale,fname)
    outF = open(fname+'.scl', "w")
    outF.writelines(table)
    outF.close()
    return

from fractions import Fraction
from math import log
from functools import reduce

def gcd(*numbers):
    """
    Return the greatest common divisor of the given integers
    The result will have the same sign as the last number given (so that
    when the last number is divided by the result, the result comes out
    positive).
    """
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    return reduce(gcd, numbers)
def reduced_form(*numbers):
    """
    Return a tuple of numbers which is the reduced form of the input,
    which is a list of integers
    """
    return tuple(int(a // gcd(*numbers)) for a in numbers)
def lcm(*numbers):
    """
    Return the least common multiple of the given integers
    """
    def lcm(a, b):
        if a == b == 0:
            return 0
        return (a * b) // gcd(a, b)

    # LCM(a, b, c, d) = LCM(a, LCM(b, LCM(c, d)))
    return reduce(lcm, numbers)

def prime_factors(n):
    """
    Return a list of the prime factors of the integer n.
    Don't use this for big numbers; it's a dumb brute-force method.
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


def EMD_eeg (eeg_data):
    s = np.interp(eeg_data, (eeg_data.min(), eeg_data.max()), (0, +1))
    eemd = EEMD()
    t = np.linspace(0, 1, len(eeg_data))
    # Say we want detect extrema using parabolic method
    emd = eemd.EMD
    emd.extrema_detection="parabol"
    eIMFs = EMD().emd(s,t)
    #eIMFs = eemd.eemd(S, t)
    nIMFs = eIMFs.shape[0]
    return eIMFs


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

from scipy.fftpack import rfft, irfft

def phaseScrambleTS(ts):
    """Returns a TS: original TS power is preserved; TS phase is shuffled."""
    fs = rfft(ts)
    # rfft returns real and imaginary components in adjacent elements of a real array
    pow_fs = fs[1:-1:2]**2 + fs[2::2]**2
    phase_fs = np.arctan2(fs[2::2], fs[1:-1:2])
    phase_fsr = phase_fs.copy()
    np.random.shuffle(phase_fsr)
    # use broadcasting and ravel to interleave the real and imaginary components.
    # The first and last elements in the fourier array don't have any phase information, and thus don't change
    fsrp = np.sqrt(pow_fs[:, np.newaxis]) * np.c_[np.cos(phase_fsr), np.sin(phase_fsr)]
    fsrp = np.r_[fs[0], fsrp.ravel(), fs[-1]]
    tsr = irfft(fsrp)
    return tsr


##Surrogate data testing using Pyunicorn
try:
    from pyunicorn.timeseries.surrogates import *
    from pyunicorn.timeseries import RecurrenceNetwork
except:
    pass
#Surrogates.silence_level = 1
def AAFT_surrogates(self, original_data):
    """
    Return surrogates using the amplitude adjusted Fourier transform
    method.

    Reference: [Schreiber2000]_

    :type original_data: 2D array [index, time]
    :arg original_data: The original time series.
    :rtype: 2D array [index, time]
    :return: The surrogate time series.
    """
    #  Create sorted Gaussian reference series
    gaussian = random.randn(original_data.shape[0], original_data.shape[1])
    gaussian.sort(axis=1)

    #  Rescale data to Gaussian distribution
    ranks = original_data.argsort(axis=1).argsort(axis=1)
    rescaled_data = np.zeros(original_data.shape)

    for i in range(original_data.shape[0]):
        rescaled_data[i, :] = gaussian[i, ranks[i, :]]

    #  Phase randomize rescaled data
    phase_randomized_data = \
        correlated_noise_surrogates(self, rescaled_data)

    #  Rescale back to amplitude distribution of original data
    sorted_original = original_data.copy()
    sorted_original.sort(axis=1)

    ranks = phase_randomized_data.argsort(axis=1).argsort(axis=1)

    for i in range(original_data.shape[0]):
        rescaled_data[i, :] = sorted_original[i, ranks[i, :]]

    return rescaled_data

def correlated_noise_surrogates(self, original_data):
    """
    Return Fourier surrogates.

    Generate surrogates by Fourier transforming the :attr:`original_data`
    time series (assumed to be real valued), randomizing the phases and
    then applying an inverse Fourier transform. Correlated noise surrogates
    share their power spectrum and autocorrelation function with the
    original_data time series.

    The Fast Fourier transforms of all time series are cached to facilitate
    a faster generation of several surrogates for each time series. Hence,
    :meth:`clear_cache` has to be called before generating surrogates from
    a different set of time series!

    .. note::
       The amplitudes are not adjusted here, i.e., the
       individual amplitude distributions are not conserved!

    **Examples:**

    The power spectrum is conserved up to small numerical deviations:

    >>> ts = Surrogates.SmallTestData().original_data
    >>> surrogates = Surrogates.\
            SmallTestData().correlated_noise_surrogates(ts)
    >>> all(r(np.abs(np.fft.fft(ts,         axis=1))[0,1:10]) == \
            r(np.abs(np.fft.fft(surrogates, axis=1))[0,1:10]))
    True

    However, the time series amplitude distributions differ:

    >>> all(np.histogram(ts[0,:])[0] == np.histogram(surrogates[0,:])[0])
    False

    :type original_data: 2D array [index, time]
    :arg original_data: The original time series.
    :rtype: 2D array [index, time]
    :return: The surrogate time series.
    """
    if self.silence_level <= 1:
        print("Generating correlated noise surrogates...")

    #  Calculate FFT of original_data time series
    #  The FFT of the original_data data has to be calculated only once,
    #  so it is stored in self._original_data_fft.
    #if self._fft_cached:
    #    surrogates = self._original_data_fft
    #else:
    surrogates = np.fft.rfft(original_data, axis=1)
    self._original_data_fft = surrogates
    self._fft_cached = True

    #  Get shapes
    (N, n_time) = original_data.shape
    len_phase = surrogates.shape[1]

    #  Generate random phases uniformly distributed in the
    #  interval [0, 2*Pi]
    phases = random.uniform(low=0, high=2 * np.pi, size=(N, len_phase))

    #  Add random phases uniformly distributed in the interval [0, 2*Pi]
    surrogates *= np.exp(1j * phases)

    #  Calculate IFFT and take the real part, the remaining imaginary part
    #  is due to numerical errors.
    return np.ascontiguousarray(np.real(np.fft.irfft(surrogates, n=n_time,
                                                     axis=1)))

## This function takes instantaneous frequencies (IF) from Hilbert-Huang Transform (HH) as an array in the form of [time, freqs]
## and outputs the average euler consonance for the whole time series (euler_tot) and the number of moments frequencies had consonance
## below euler threshold (euler_tresh)
def HH_cons (IF, euler_tresh = 100, mult = 10):
    euler_tot = []
    cons_ind = []
    euler_good = []
    for t in range(len(IF)):
        freqs = [int(np.round(x*mult)) for x in IF[t]]

        for i in freqs:
            if i <= 0:
                #print('negative')
                #print(IF[t])
                freqs = [1 for x in freqs]
        if mult > 10:
            if freqs[-1] > 10*mult or freqs[-2] > 15*mult or freqs[-3] > 20*mult:
                freqs = [1 for x in freqs]
                print('non-expected high freq detected')
        #print(freqs)
        e_temp = euler(*freqs)
        if e_temp > 1 and e_temp < euler_tresh:
            cons_ind.append(t)
        if e_temp > 1:
            euler_good.append(e_temp)
        euler_tot.append(e_temp)
    return euler_good, cons_ind, euler_tot


import secrets

def graph_dist(dist, metric = 'diss', ref = None, dimensions = [0, 1], labs = ['eeg', 'phase', 'AAFT', 'pink', 'white'], subject = '0', run = '0', adapt = 'False'):
    #if ref == None:
    #    ref = dist[0]
    if metric == 'diss':
        m = 'Dissonance (From Sethares (2005))'
    if metric == 'euler':
        m = 'Consonance (Euler <Gradus Suavitatis>)'
    if metric == 'diss_euler':
        m = 'Consonance (Euler <Gradus Suavitatis>) of dissonant minima'
    if metric == 'Nratios':
        m = 'Number of dissonant minima'
    if metric == 'tenney':
        m = 'Tenney Height'
    if metric == 'HarmSim':
        m = 'Harmonic similarity of peaks'
    if metric == 'HarmSimDiss':
        m = 'Harmonic similarity of scale'
    if metric == 'HarmFit':
        m = 'Harmonic fitness between peaks'


    plt.rcParams['axes.facecolor'] = 'black'
    fig = plt.figure(figsize=(14,10))
    colors = ['cyan', 'goldenrod', 'yellow', 'deeppink', 'white']

    xcoords = []


    for dim in dimensions:
        labs = labs
        if dim == 0:
            dimension = 'channels'
        if dim == 1:
            dimension = 'trials'


        for d, color, enum in zip(dist, colors, range(len(dist))):
            #d = d[~np.isnan(d)]
            print(d.shape)

            if dimensions == [0]:

                sbn.distplot(d, color =color)
                secure_random = secrets.SystemRandom()
                if len(d) < len(ref):
                    ref = secure_random.sample(list(ref), len(d))
                if len(ref) < len(d):
                    d = secure_random.sample(list(d), len(ref))
                t, p = stats.ttest_rel(d, ref)
            if dimensions != [0]:
                sbn.distplot(np.nanmean(d, dim), color =color)
                secure_random = secrets.SystemRandom()
                if len(d) < len(ref):
                    ref = secure_random.sample(list(ref), len(d))
                if len(ref) < len(d):
                    d = secure_random.sample(list(d), len(ref))
                t, p = stats.ttest_rel(np.nanmean(ref, dim), np.nanmean(d, dim))
                print(p)
            if p < 0.05:
                labs[enum] = labs[enum]+' *'
                #xcoords.append(np.average(d))

                #for xc in xcoords:
                #    plt.axvline(x=xc, c='white')


        fig.legend(labels=[labs[0], labs[1]],
                   loc = [0.66, 0.68], fontsize = 16, facecolor = 'white')
        plt.xlabel(m, fontsize = '16')
        plt.ylabel('Proportion of samples', fontsize = '16')
        #plt.xlim([0.25, 0.7])
        plt.grid(color='white', linestyle='-.', linewidth=0.7)
        plt.suptitle('Comparing ' + m+ ' \nfor EEG, surrogate data, and pink noise signals across ' + dimension, fontsize = '22')
        fig.savefig('{}_distribution_s{}-bloc{}_EMD_adapt-{}_{}.png'.format(metric, subject, run, adapt, dimension), dpi=300)
        plt.clf()


def getPairs(peaks):
    out = []
    for i in range(len(peaks)-1):
        a = peaks.pop(i-i)
        for j in peaks:
            out.append([a, j])
    return out



def scale2frac (scale, maxdenom = 1000):
    num = []
    den = []
    scale_frac = []
    for step in scale:
        frac = Fraction(step).limit_denominator(maxdenom)
        num.append(frac.numerator)
        den.append(frac.denominator)
        scale_frac.append(frac)
    return scale_frac, array(num), array(den)
