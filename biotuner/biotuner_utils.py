#!bin/bash
"""Collection of operations on timeseries."""
import numpy as np
from PyEMD import EMD, EEMD
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import seaborn as sbn
from scipy import stats
import pygame, pygame.sndarray
import pytuning
from pytuning import *
import scipy.signal
import sympy as sp
import functools
import itertools
import operator
import sys
import bottleneck as bn
sys.setrecursionlimit(120000)



def compute_peak_ratios(peaks, rebound=True, octave=2, sub=False):
    """
    Ratios between peaks.

    This function calculates all the ratios (with the possibility to bound them
    between 1 and 2) derived from input peaks.

    peaks: List (float)
        Peaks represent local maximum in a spectrum
    rebound: boolean
        Defaults to True. False will output unbounded ratios
    octave: int
        Arbitrary wanted number of octaves. Defaults to 2.
    sub: boolean
        Defaults to False. True will include ratios below the unison (1)
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
                ratio_temp = p2/p1
            except:
                pass
                
            # When sub is set to False, only ratios with numerators higher than denominators are consider
            if sub is False:
                if ratio_temp < 1:
                    ratio_temp = None
            # Ratios of 1 correspond to a peak divided by itself and are therefore not considered
            if ratio_temp == 1:
                ratio_temp = None
            ratios.append(ratio_temp)

        # I imagine there's a reason why you reinit with array
        peak_ratios = np.array(ratios)
        peak_ratios = [i for i in peak_ratios if i]  # dealing with NaNs ?
        peak_ratios = sorted(list(set(peak_ratios)))
        ratios_final = peak_ratios.copy()
    
    # If rebound is given, all the ratios are constrained between the unison and the octave
    if rebound is True:
        for peak in peak_ratios:

            # will divide the ratio by the octave until it reaches a value under the octave
            if peak > octave:
                while peak > octave:
                    peak = peak/octave
                peak_ratios_final.append(peak)
            # will multiply the ratio by the octave until it reaches a value over the unison (1)
            if peak < octave:
                while peak < 1:
                    peak = peak*octave
                peak_ratios_final.append(peak)
        # Preparing output
        peak_ratios_final = np.array(peak_ratios_final)
        peak_ratios_final = [i for i in peak_ratios_final if i]
        ratios_final = sorted(list(set(peak_ratios_final)))
    

    return ratios_final


def rebound(x, low=1, high=2, octave=2):
    """
    Rescale x within given octave bounds.

    x: int
        represents a peak value
    low: int
        Lower bound. Defaults to 1. 
    high: int
        Higher bound. Defaults to 2.
    octave: int
        Value of an octave
    """
    while x >= high:
        x = x/octave
    while x <= low:
        x = x*octave
    return x

def rebound_list(x_list, low=1, high=2, octave=2):
    return [rebound(x, low, high, octave) for x in x_list] 

def nth_root (num, root):
    '''
    This function computes the ratio associated with one step of a N-TET scale
    
    num: int
        value of the octave
    root: int
        number of steps in the N-TET scale
    '''
    answer = num**(1/root)
    return answer

def NTET_steps (octave, step, NTET):
    '''
    This function computes the ratio associated with a specific step of a N-TET scale
    
    octave: int
        value of the octave
    step: int
        value of the step
    NTET: int
        number of steps in the N-TET scale
    '''
    answer = octave**(step/NTET)
    return answer

#Function that compares lists (i.e. peak harmonics)
def compareLists(list1, list2, bounds):
    matching = []
    matching_pos = []
    matching_pos1 = []
    matching_pos2 = []
    positions = []
    for i, l1 in enumerate(list1):
        for j, l2 in enumerate(list2):
            if l2-bounds < l1 < l2+bounds:
                matching.append((l1+l2)/2)
                matching_pos.append([(l1+l2)/2, i+1, j+1])
                matching_pos1.append([list1[0], i+1])
                matching_pos2.append([list2[0], j+1])
                positions.append(i+1)
                positions.append(j+1)
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
    return matching, matching_pos, matching_pos_ratios, matching_pos1, matching_pos2, positions

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

def gcd_n(*args):
    i = 1
    x = args[0]

    while i < len(args):
        x = gcd(x, args[i])
        i += 1
    return x

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

def top_n_indexes(arr, n):
    idx = bn.argpartition(arr, arr.size-n, axis=None)[-n:]
    width = arr.shape[1]
    return [divmod(i, width) for i in idx]

def scale_from_pairs(pairs):   
    return[rebound((x[1]/x[0])) for x in pairs]

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

def getPairs(peaks):
    peaks_ = peaks.copy()
    out = []
    for i in range(len(peaks_)-1):
        a = peaks_.pop(i-i)
        for j in peaks_:
            out.append([a, j])
    return out

def functools_reduce(a):
    return functools.reduce(operator.concat, a)

def scale2frac (scale, maxdenom = 1000):
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

def sort_scale_by_consonance(scale):
    cons_tot = []
    for step in scale:
        frac = Fraction(step).limit_denominator(1000)
        cons = (frac.numerator + frac.denominator)/(frac.numerator * frac.denominator)
        cons_tot.append(cons)
    sorted_scale = list(np.flip([x for _,x in sorted(zip(cons_tot,scale))]))
    return sorted_scale

def ratio2frac (ratio, maxdenom = 1000):
    
    frac = Fraction(ratio).limit_denominator(maxdenom)
    num = frac.numerator
    den = frac.denominator
    return [num, den]

def frac2scale (scale):
    scale_ratio = []
    for step in scale:
        scale_ratio.append(float(step))
    return scale_ratio

def peaks_to_amps (peaks, freqs, amps, sf):
    bin_size = np.round((sf/2)/len(freqs), 3)
    amps_out = []
    #print(amps.shape)
    for p in peaks:
        #print(p)
        index = int(p/bin_size)
        amp = amps[index]
        amps_out.append(amp)
    return amps_out

def chords_to_ratios(chords, harm_limit = 2, spread = True):
    chords_ratios = []
    chords_ratios_bounded = []
    for chord in chords:
        chord = sorted(chord)
        if harm_limit != None:
            if spread == True: #will allow each note to be within the defined harm_limit of the previous note 
                for note in range(len(chord)):
                    while chord[note] > chord[note-1]*harm_limit:
                        chord[note] = chord[note]/2
            if spread == False: #will allow each note to be within the defined harm_limit of the first note
                for note in range(len(chord)):
                    while chord[note] > chord[0]*2:
                        chord[note] = chord[note]/2
        chord = sorted([np.round(n, 1) for n in chord])
        chord = [int(n*10) for n in chord]
        gcd_chord = 2 #arbitrary number that is higher than 1
        while gcd_chord > 1:
            gcd_chord = gcd_n(*chord)
            if gcd_chord > 1:
                chord = [int(note/gcd_chord) for note in chord]
        chord_bounded = [c/chord[0] for c in chord]
        chords_ratios_bounded.append(chord_bounded)
        chords_ratios.append(chord)
    return chords_ratios, chords_ratios_bounded

def NTET_ratios (n_steps, max_ratio):
    steps = []
    for s in range(n_steps):
        steps.append(2**(s/n_steps))
    steps_out = []
    for j in range(max_ratio-1):
        steps_out.append([i+j for i in steps])
    steps_out = sum(steps_out, [])
    return steps_out

def bjorklund(steps, pulses):
    '''From https://github.com/brianhouse/bjorklund'''
    steps = int(steps)
    pulses = int(pulses)
    if pulses > steps:
        raise ValueError    
    pattern = []    
    counts = []
    remainders = []
    divisor = steps - pulses
    remainders.append(pulses)
    level = 0
    while True:
        counts.append(divisor // remainders[level])
        remainders.append(divisor % remainders[level])
        divisor = remainders[level]
        level = level + 1
        if remainders[level] <= 1:
            break
    counts.append(divisor)
    
    def build(level):
        if level == -1:
            pattern.append(0)
        elif level == -2:
            pattern.append(1)         
        else:
            for i in range(0, counts[level]):
                build(level - 1)
            if remainders[level] != 0:
                build(level - 2)
    
    build(level)
    i = pattern.index(1)
    pattern = pattern[i:] + pattern[0:i]
    return pattern


'''Spectromorphology functions'''

import pyACA

def computeFeatureCl_new(afAudioData, cFeatureName, f_s, window=4000, overlap=1):

    # compute feature
    [v, t] = pyACA.computeFeature(cFeatureName, afAudioData, f_s, None, window, overlap)

    return (v, t)

def EMD_to_spectromorph (IMFs,  sf, method = "SpectralCentroid", window = None, overlap = 1, in_cut =None, out_cut = None):
    if in_cut == None:
        in_cut = int(sf/2)
    if out_cut == None:
        out_cut = int(len(IMFs[0])-(sf/2))
    if window == None:
        window = int(sf/2)
    spectro_IMF = []
    for e in IMFs:
        f, t = computeFeatureCl_new(e, method, sf, window, overlap)
        #[f,t] = pyACA.computePitch('TimeAcf', e, 1000, afWindow=None, iBlockLength=1000, iHopLength=200)
        #df[i] = np.squeeze(f)
        if method == 'SpectralCentroid':
            spectro_IMF.append(f[in_cut:out_cut])
        if method == 'SpectralFlux':
            spectro_IMF.append(f[in_cut:out_cut])
    spectro_IMF = np.array(spectro_IMF)
    return spectro_IMF


def butter_bandpass(lowcut,highcut,fs,order=8):
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq

    b,a = butter(order, [low, high], btype='band')
    return b,a

def butter_bandpass_filter(data,lowcut,highcut,fs,order=8):
    b,a = butter_bandpass(lowcut,highcut,fs,order=order)
    return lfilter(b,a,data) 


def ratios_harmonics (ratios, n_harms = 1):
    ratios_harms = []
    for h in range(n_harms):
        h += 1 
        ratios_harms.append([i*h for i in ratios])
    ratios_harms = [i for sublist in ratios_harms for i in sublist]
    return ratios_harms

def ratios_increments (ratios, n_inc = 1):
    ratios_harms = []
    for h in range(n_inc):
        h += 1 
        ratios_harms.append([i**h for i in ratios])
    ratios_harms = [i for sublist in ratios_harms for i in sublist]
    ratios_harms = list(set(ratios_harms))
    return ratios_harms


def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    #if x.ndim != 1:
        #raise ValueError, "smooth only accepts 1 dimension arrays."

    #if x.size < window_len:
        #raise ValueError, "Input vector needs to be bigger than window size."


    #if window_len<3:
        #return x


    #if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        #raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y



sample_rate = 44100
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
   
def sine_wave(hz, peak, n_samples=sample_rate):
    """Compute N samples of a sine wave with given frequency and peak amplitude.
       Defaults to one second.
    """
    length = sample_rate / float(hz)
    omega = np.pi * 2 / length
    xvalues = np.arange(int(length)) * omega
    onecycle = peak * np.sin(xvalues)
    return np.resize(onecycle, (n_samples,)).astype(np.int16)

def square_wave(hz, peak, duty_cycle=.5, n_samples=sample_rate):
    """Compute N samples of a sine wave with given frequency and peak amplitude.
       Defaults to one second.
    """
    t = np.linspace(0, 1, 500 * 440/hz, endpoint=False)
    wave = scipy.signal.square(2 * np.pi * 5 * t, duty=duty_cycle)
    wave = np.resize(wave, (n_samples,))
    return (peak / 2 * wave.astype(np.int16))

waveform=sine_wave
def make_chord(hz, ratios):
    """Make a chord based on a list of frequency ratios."""
    sampling = 2048
    chord = waveform(hz, sampling)
    for r in ratios[1:]:
        chord = sum([chord, sine_wave(hz * r / ratios[0], sampling)])
    return chord

def major_triad(hz):
    return make_chord(hz, [4, 5, 6])

def listen_scale (scale, fund, length):
    print('Scale:', scale)
    scale = [1]+scale
    for s in scale:
        freq = fund*s
        print(freq)
        note = make_chord(freq, [1])
        note = np.ascontiguousarray(np.vstack([note,note]).T)
        sound = pygame.sndarray.make_sound(note)
        sound.play(loops=0, maxtime=0, fade_ms=0)
        pygame.time.wait(int(sound.get_length() * length))
        
def listen_chords (chords, mult = 10, length = 500):
    #chords = np.around(chords, 3)
    print('Chords:', chords)

    for c in chords:
        c = [i*mult for i in c]
        chord = make_chord(c[0], c[1:])
        chord = np.ascontiguousarray(np.vstack([chord,chord]).T)
        sound = pygame.sndarray.make_sound(chord)
        sound.play(loops=0, maxtime=0, fade_ms=0)
        pygame.time.wait(int(sound.get_length() * length))
        
def horogram_tree_steps (ratio1, ratio2, steps = 10, limit = 1000):
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
        

def horogram_tree (ratio1, ratio2, limit):
    a = Fraction(ratio1).limit_denominator(limit).numerator
    b = Fraction(ratio1).limit_denominator(limit).denominator
    c = Fraction(ratio2).limit_denominator(limit).numerator
    d = Fraction(ratio2).limit_denominator(limit).denominator
    next_step = (a+c)/(b+d)
    return next_step
        
def phi_convergent_point (ratio1, ratio2):
    Phi = (1 + 5 ** 0.5) / 2
    a = Fraction(ratio1).limit_denominator(1000).numerator
    b = Fraction(ratio1).limit_denominator(1000).denominator
    c = Fraction(ratio2).limit_denominator(1000).numerator
    d = Fraction(ratio2).limit_denominator(1000).denominator
    convergent_point = (c*Phi+a)/(d*Phi+b)
    return convergent_point

def Stern_Brocot(n,a=0,b=1,c=1,d=1):
    if(a+b+c+d>n):
        return 0
    x=Stern_Brocot(n,a+c,b+d,c,d)
    y=Stern_Brocot(n,a,b,a+c,b+d)
    if(x==0):
        if(y==0):
            return [a+c,b+d]
        else:
            return [a+c]+[b+d]+y
    else:
        if(y==0):
            return [a+c]+[b+d]+x
        else:
            return [a+c]+[b+d]+x+y
        
def scale_interval_names(scale, reduce = False):
    try:
        type = scale[0].dtype == 'float64'
        if type == True:
            scale, _, _ = scale2frac(scale)   
    except:
        pass
    interval_names = []
    for step in scale:
        name = pytuning.utilities.ratio_to_name(step)
        if reduce == True and name != None:
            interval_names.append([step, name])
        if reduce == False:
            interval_names.append([step, name])
    return interval_names

''' Continued fractions '''

import math

def contFrac(x, k):
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

def convergents (interval):
    value = log2(interval)
    convergents = list(contfrac.convergents(value))
    return convergents
