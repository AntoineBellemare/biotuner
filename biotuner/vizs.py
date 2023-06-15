from numpy import sin, pi, linspace
from pylab import plot, subplot
from matplotlib.pyplot import figure
import matplotlib.colors as mcolors
from biotuner.biotuner_utils import scale2frac, NTET_ratios, compute_IMs
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
from math import log2
import plotly.graph_objs as go
import ipywidgets as widgets
from IPython.display import display
from IPython.display import display, clear_output
from collections import defaultdict
import math
from biotuner.biotuner_utils import sum_list, compute_peak_ratios
from biotuner.metrics import ratios2harmsim
#from biotuner.peaks_extraction import compute_IMs


def lissajous_curves(tuning):
    """
    Plot Lissajous curves for given tuning ratios.
    Parameters
    ----------
    tuning: List[float]
        List of tuning ratios for which Lissajous curves should be plotted
    Returns
    -------
    None
    
    Examples
    --------
    .. plot::
    >>> tuning = [1, 1.25, 1.33, 1.5, 1.7, 1.875]
    >>> lissajous_curves(tuning)
    """
    fracs, num, denom = scale2frac(tuning)
    figure(figsize=(64, 40), dpi=80)
    a = num  # plotting the curves for
    b = denom  # different values of a/b
    delta = pi/2
    t = linspace(-pi, pi, 300)
    colors = list(mcolors.TABLEAU_COLORS.values())*3
    for i, c in zip(range(len(a)), colors):
        x = sin(a[i] * t + delta)
        y = sin(b[i] * t)
        if len(a) % 2 == 0:
            subplot(int(len(a)/2), int(len(a)/2), i+1)
        else:
            subplot(int(len(a)/2), int(len(a)/2)+1, i+1)
        plot(x, y, c)
    print(fracs)


def graph_psd_peaks(freqs, psd, peaks, xmin, xmax, color='deeppink',
                    method=None):
    #psd = np.interp(psd, (psd.min(), psd.max()), (0, 0.005))
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(freqs, psd, color=color)
    plt.xlim([xmin, xmax])
    idx1 = list(freqs).index(xmin)
    idx2 = list(freqs).index(xmax)
    ymin = int(np.min(psd[idx1:idx2]))
    ymax = int(np.max(psd[idx1:idx2]))
    plt.ylim([ymin, ymax])
    plt.xlabel('Frequency (Hertz)', size=14)
    plt.ylabel('PSD [V**2/Hz]', size=14)
    if method is not None:
        plt.title('Spectral peaks positions using ' + method
                  + ' method', size=18)
    if method is None:
        plt.title('Spectral peaks positions', size=18)
    for xc in peaks:
        plt.axvline(x=xc, c='black', linestyle='dotted')
    plt.tight_layout()
    plt.show()




def plot_polycoherence(freq1, freq2, bicoh):
    """
    Plot polycoherence (i.e. return values of polycoherence with dim=2)
    """
    df1 = freq1[1] - freq1[0]
    df2 = freq2[1] - freq2[0]
    freq1 = np.append(freq1, freq1[-1] + df1) - 0.5 * df1
    freq2 = np.append(freq2, freq2[-1] + df2) - 0.5 * df2
    plt.figure()
    plt.pcolormesh(freq2, freq1, np.abs(bicoh))
    plt.xlabel('freq (Hz)')
    plt.ylabel('freq (Hz)')
    plt.colorbar()
    return plt


import matplotlib.pyplot as plt
import scipy.signal

def graphEMD_welch(freqs_all, psd_all, peaks, raw_data, FREQ_BANDS,
                   sf, nfft, nperseg, noverlap, min_freq=1,
                   max_freq=60, precision=0.5):
    plt.rcParams["figure.figsize"] = (8, 5)
    color_line = ['aqua', 'darkturquoise', 'darkcyan', 'darkslategrey', 'black']

    for i, (freqs, psd) in enumerate(zip(freqs_all, psd_all)):
        plt.fill_between(freqs, psd, color=color_line[i], alpha=0.2)

    mult = 1/precision
    nperseg = sf*mult
    nfft = nperseg
    freqs_full, psd_full = scipy.signal.welch(raw_data, sf,
                                              nfft=nfft,
                                              nperseg=nperseg,
                                              noverlap=noverlap)

    band_names = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    color_bg = ['darkgoldenrod', 'goldenrod', 'orange', 'gold', 'khaki']
    alpha = [0.6, 0.63, 0.66, 0.69, 0.72]

    # Fill areas for frequency bands and place band names
    for (band, name, color, a) in zip(FREQ_BANDS, band_names, color_bg, alpha):
        plt.axvspan(band[0], band[1], ymin=0, alpha=a, color=color, ec='black')
        plt.text(np.sqrt(band[0]*band[1]), -5, name, horizontalalignment='center', size=15)

    plt.xlim([min_freq, max_freq])
    # use adaptive ylim based on psd_all (list of psd) and psd_full (raw data)
      
    plt.ylim([np.min(psd_all[0]), np.max(psd_full)+((np.max(psd_full)-np.min(psd_full)))/3])
    
    plt.title('PSD of Empirical Mode Decomposition', size=28)
    plt.xlabel('Frequency', size=15)
    plt.ylabel('Power', size=15)
    plt.tick_params(axis='both', which='major', labelsize=12, length=6, width=4)
    plt.tick_params(axis='both', which='minor', labelsize=10, length=6, width=4)
    plt.xscale('log')

    plt.plot(freqs_full, psd_full, color='deeppink', linestyle='dashed', label='raw data')

    xposition = peaks
    labels = ['EMD1', 'EMD2', 'EMD3', 'EMD4', 'EMD5']
    for p, n, band in zip(peaks, range(len(labels)), FREQ_BANDS):
        if p > band[0] and p <= band[1]:
            labels[n] = labels[n]+'*'
    for xc, c, l in zip(xposition, color_line, labels):
        plt.axvline(x=xc, label='{} = {}'.format(l, xc), c=c)

    plt.legend(loc='lower left', fontsize=16)
    plt.tight_layout()  # Make sure everything fits without overlap
    plt.show()  # Display the plot



def graph_harm_peaks(freqs, psd, harm_peaks_fit, xmin, xmax, color='black',
                     save=False, figname='test', n_peaks=5):
    """
    This function plots the power spectral density of a signal,
    and the position of the peaks that are harmonically related.
    
    Parameters
    ----------
    freqs : array
        The frequencies of the signal 
    psd : array
        The power spectral density of the signal
    harm_peaks_fit : array
        The positions of the peaks that are harmonically related
    xmin : int or float
        The minimum frequency to be plotted
    xmax : int or float
        The maximum frequency to be plotted
    color : str
        The color of the plotted PSD, default is black
    save : bool
        Whether to save the figure or not
    figname : str
        The name of the file if the figure is saved
    n_peaks : int
        The number of peaks to be plotted
        
    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(freqs[:np.where(freqs >= xmax)[0][0]], psd[:np.where(freqs >= xmax)[0][0]], color=color)
    idx_min = list(freqs).index(xmin)
    idx_max = list(freqs).index(xmax)
    ymin = np.min(psd[idx_min:idx_max])
    ymax = np.max(psd[idx_min:idx_max])
    plt.ylim([ymin, ymax])
    plt.xlabel('Frequency (Hertz)', size=14)
    plt.ylabel('PSD [V**2/Hz]', size=14)
    plt.title('Spectral peaks positions using Harmonic Recurrence', size=18)
    color_list = ['blue', 'red', 'orange', 'turquoise', 'purple', 'green'][:n_peaks+1]
    y_steps = (ymax-ymin)/10
    y_list = [ymax-(y_steps*(i+2)) for i in range(n_peaks)]

    for peak_info, color_harm, ys in zip(harm_peaks_fit[:n_peaks], color_list[:n_peaks], y_list[:n_peaks]):
        peak = peak_info[0]
        harm_pos = [int(x) for x in peak_info[1]]
        harm_freq = peak_info[2]
        harm_freq.remove(peak)


        plt.axvline(x=peak, c=color_harm, linestyle='-')

        for e, harm in enumerate(harm_freq):
            if harm < xmax:
                plt.axvline(x=harm, c=color_harm, linestyle='dotted')
                ax.annotate(str(harm_pos[e]), (harm, ys),
                                bbox=dict(boxstyle="square", alpha=0.2,color=color_harm),
                                xytext=(harm+0.5, ys), fontsize=12)
    ax.set_xlim([xmin, xmax])
    ax.set_aspect('auto')
    plt.tight_layout()
    if save is True:
        plt.savefig(figname, dpi=300)
    else:
        plt.show()

import matplotlib.colors as mcolors




##Multiple PSD##
from PyEMD import EMD, EEMD
import numpy as np
import operator
import scipy
from random import seed


def EMD_PSD_graph(eeg_data, IMFs, peaks_EMD, spectro='Euler', bands = None, xmin=1, xmax=70,
                  compare=True, name = '', sf=1000, nfft=4096, nperseg=1024, noverlap=512,
                  freqs_all=None, psd_all=None, max_freq=80, precision=0.5):

    if bands is None:
        bands = [[0, 3], [3, 7], [7, 12], [12, 30], [30, 70]]
    percentage_fit = []
    if max_freq is None:
        max_freq = sf / 2
    if nperseg is None:
        mult = 1 / precision
        nperseg = sf * mult
        nfft = nperseg
    #freqs_all = []
    #psd_all = []
    '''for e in range(len(IMFs)):
        freqs, psd = scipy.signal.welch(IMFs[e], sf, nfft=nfft, nperseg=nperseg, noverlap=noverlap)
        freqs_all.append(freqs)
        psd_all.append(psd)
    freqs_all = np.array(freqs_all)
    psd_all = np.array(psd_all)'''
    # Stackplot with X, Y, colors value

    color_line = ['aqua', 'darkturquoise', 'darkcyan', 'darkslategrey', 'black']
    #color_line = ['darkgoldenrod', 'goldenrod', 'orange', 'gold', 'khaki']
    color_bg = ['darkgoldenrod', 'goldenrod', 'orange', 'gold', 'khaki']
    #color_bg = ['aqua', 'darkturquoise', 'darkcyan', 'darkslategrey', 'black']
    idx_min = list(freqs_all[0]).index(xmin)
    idx_max = list(freqs_all[0]).index(xmax)
    #for i in range(len(psd_all)):
#        psd_all[i] = np.interp(psd_all[i], (psd_all[i][idx_min:idx_max].min(), psd_all[i][idx_min:idx_max].max()), (0, 1))


    fig, ax = plt.subplots(figsize=(12, 6))
    #print(len(freqs_all))
    ax.plot(freqs_all[0], psd_all[4], color = color_line[0])
    ax.plot(freqs_all[0], psd_all[3], color = color_line[1])
    ax.plot(freqs_all[0], psd_all[2], color = color_line[2])
    ax.plot(freqs_all[0], psd_all[1], color = color_line[3])
    ax.plot(freqs_all[0], psd_all[0], color = color_line[4])
    ax.fill_between(freqs_all[0], psd_all[0], 0, color=color_line[0], alpha=.7)
    ax.fill_between(freqs_all[1], psd_all[1], 0, color=color_line[1], alpha=.7)
    ax.fill_between(freqs_all[2], psd_all[2], 0, color=color_line[2], alpha=.7)
    ax.fill_between(freqs_all[3], psd_all[3], 0, color=color_line[3], alpha=.7)
    ax.fill_between(freqs_all[4], psd_all[4], 0, color=color_line[4], alpha=.7)


        #color_bg = ['paleturquoise', 'aqua', 'darkturquoise','darkcyan' , 'darkslategrey']
    alpha = [0.6, 0.63, 0.66, 0.69, 0.72]
    shadow = 0.9

    ymin = np.min(psd_all[0][idx_min:idx_max])
    ymax = np.max(psd_all[-1][idx_min:idx_max])
    #print('YMINMAX', ymin, ymax)
    if compare is True:
        freqs_full, psd_full = scipy.signal.welch(eeg_data, sf, nfft = nfft, nperseg = nperseg, noverlap=noverlap)
        psd_full = np.interp(psd_full, (psd_full[idx_min:idx_max].min(), psd_full[idx_min:idx_max].max()), (ymin, ymax))
        ax.plot(freqs_all[0], psd_full, color = 'deeppink', linestyle='dashed')
    plt.text(1.7, 0.0028, 'delta', horizontalalignment = 'center', fontsize=12)
    plt.text(4.5, 0.0028, 'theta', horizontalalignment = 'center',fontsize=12)
    plt.text(9, 0.0028, 'alpha', horizontalalignment = 'center',fontsize=12)
    plt.text(19, 0.0028, 'beta', horizontalalignment = 'center',fontsize=12)
    plt.text(47, 0.0028, 'gamma', horizontalalignment = 'center',fontsize=12)
    plt.xlim([xmin, xmax])

    #plt.ylim([0.000001, 0.006])
    plt.ylim([ymin, ymax+((ymax-ymin)/3)])
    #plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.4f}'))
    plt.title('PSD of Empirical Mode Decomposition 1 to 5', fontsize=14)
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.yscale('symlog')
    plt.xscale('log')

    xposition = peaks_EMD

    labels = ['IMF5', 'IMF4', 'IMF3', 'IMF2', 'IMF1']
    for p, n, band in zip(peaks_EMD, range(len(labels)), bands):
        #print(band[0])
        #print(band[1])
        #print(p)

        if p > band[0] and p<= band[1]:
            #print('got it')
            labels[n] = labels[n]+'*'

    print(labels)
    for xc,c, l in zip(xposition,color_line, labels):
        plt.axvline(x=xc, label='{} = {}'.format(l, xc), c=c)
    plt.axvspan(0, 3, ymin = shadow, alpha=alpha[0], color=color_bg[0], ec ='black')
    plt.axvspan(3, 7, ymin = shadow, alpha=alpha[1], color=color_bg[1],ec ='black')
    plt.axvspan(7, 12, ymin = shadow, alpha=alpha[2], color=color_bg[2], ec ='black')
    plt.axvspan(12, 30, ymin = shadow, alpha=alpha[3], color=color_bg[3], ec ='black')
    plt.axvspan(30, 70, ymin = shadow, alpha=alpha[4], color=color_bg[4], ec ='black')

    plt.legend(loc='lower left')
    plt.show()
    #plt.savefig('PSD_EMD_multi_{}_{}_{}'.format(plot_type, input_data, name), dpi=300)
    plt.clf()
    #percentage_fit.append((i/(len(epochs.ch_names)*5))*100)
    #percentage_fit = np.array(percentage_fit)

    return percentage_fit

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

def EMD_PSD_graph(eeg_data, IMFs, peaks_EMD, spectro='Euler', bands=None, xmin=1, xmax=70,
                  compare=True, name='', sf=1000, nfft=4096, nperseg=1024, noverlap=512,
                  freqs_all=None, psd_all=None, max_freq=80, precision=0.5):

    if bands is None:
        bands = [[0, 3], [3, 7], [7, 12], [12, 30], [30, 70]]
    percentage_fit = []
    if max_freq is None:
        max_freq = sf / 2
    if nperseg is None:
        mult = 1 / precision
        nperseg = sf * mult
        nfft = nperseg

    idx_min = list(freqs_all[0]).index(xmin)
    idx_max = list(freqs_all[0]).index(xmax)

    fig, ax = plt.subplots(figsize=(12, 6))

    color_line = ['aqua', 'darkturquoise', 'darkcyan', 'darkslategrey', 'black']
    color_bg = ['darkgoldenrod', 'goldenrod', 'orange', 'gold', 'khaki']

    for i in range(len(IMFs)):
        ax.plot(freqs_all[i], psd_all[i], color=color_line[i])
        ax.fill_between(freqs_all[i], psd_all[i], 0, color=color_line[i], alpha=.7)

    for i, color in enumerate(color_line):
        ax.plot(freqs_all[i], psd_all[i], color=color)

    ymin = np.min(psd_all[0][idx_min:idx_max])
    ymax = np.max(psd_all[-1][idx_min:idx_max])

    if compare:
        freqs_full, psd_full = scipy.signal.welch(eeg_data, sf, nfft=nfft, nperseg=nperseg, noverlap=noverlap)
        psd_full = np.interp(psd_full, (psd_full[idx_min:idx_max].min(), psd_full[idx_min:idx_max].max()), (ymin, ymax))
        ax.plot(freqs_all[0], psd_full, color='deeppink', linestyle='dashed')

    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax + ((ymax - ymin) / 3)])

    plt.title('PSD of Empirical Mode Decomposition 1 to 5', fontsize=14)
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    plt.yscale('symlog')
    plt.xscale('log')

    xposition = peaks_EMD
    labels = ['IMF5', 'IMF4', 'IMF3', 'IMF2', 'IMF1']
    for p, n, band in zip(peaks_EMD, range(len(labels)), bands):
        if p > band[0] and p <= band[1]:
            labels[n] = labels[n] + '*'

    for xc, c, l in zip(xposition, color_line, labels):
        plt.axvline(x=xc, label='{} = {}'.format(l, xc), c=c)

    for i, band_name in enumerate(['delta', 'theta', 'alpha', 'beta', 'gamma']):
        plt.text(np.mean(bands[i]), ymax + (0.015 * (ymax - ymin)), band_name, horizontalalignment='center', fontsize=12)

    for i, band in enumerate(bands[:-1]):
        plt.axvspan(band[1], bands[i + 1][0], ymin=0, ymax=1, alpha=0.15, color='black')

    plt.legend(loc='lower left')
    plt.show()

    return percentage_fit

import matplotlib.pyplot as plt
import numpy as np

def visualize_rhythms(pulses_steps, offsets=None, plot_size=6, 
                      tolerance=0.1):
    """
    Visualize multiple Euclidean rhythms.
    
    Parameters
    ----------
    pulses_steps : list of tuple
        A list of tuple, where each tuple represent the number of pulses and steps of a rhythm.
    offsets : list of int, optional
        A list of offsets for each rhythm in pulses_steps.
    plot_size : int, optional
        The size of the plot.
    tolerance : float, optional
        The tolerance for considering two pulses to be in the same rhythm.
    
    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(plot_size, plot_size))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    pulses_positions = []
    for i, (pulses, steps) in enumerate(pulses_steps):
        offset = offsets[i] if offsets else 0
        rhythm = euclidean_rhythm(pulses, steps, offset)
        angles = np.linspace(0, 2*np.pi, steps, endpoint=False)
        radius = (i+1) * 0.15
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        ax.scatter(x, y, s = 100, color = colors[i%len(colors)], alpha = 0.5)
        pulse_pos = []
        for j, value in enumerate(rhythm):
            if value == 1:
                ax.scatter(x[j], y[j], s = 230, color = colors[i%len(colors)], alpha = 1)
                pulse_pos.append((x[j], y[j], np.arctan2(y[j], x[j])))
        pulses_positions.append(pulse_pos)
    for i in range(len(pulses_positions)):
        for j in range(i+1, len(pulses_positions)):
            for pulse1 in pulses_positions[i]:
                for pulse2 in pulses_positions[j]:
                    if abs(pulse1[2]-pulse2[2])<tolerance:
                        ax.plot([0, pulse1[0],pulse2[0]], [0, pulse1[1],pulse2[1]], 'k-', lw=2)
    ax.set_aspect("equal")
    ax.set_xlim(-(np.max(x))-0.1, np.max(x)+0.1)
    ax.set_ylim(-(np.max(y))-0.1, np.max(y)+0.1)
    plt.show()
    
def euclidean_rhythm(pulses, steps, offset=0):
    """
    Generate a Euclidean rhythm.
    Args:
        pulses (int): The number of pulses in the rhythm.
        steps (int): The number of steps in the rhythm.
        offset (int): An offset for the rhythm in pulses.
    Returns:
        List[int]: A binary list representing the rhythm, where 1 indicates a pulse and 0 indicates no pulse.
    """
    rhythm = [0] * steps
    for i in range(pulses):
        rhythm[(i * steps // pulses + offset) % steps] = 1
    return rhythm

import math

def find_optimal_offsets(pulses_steps):
    """
    Finds the optimal offset values for a set of Euclidean rhythms
    Args:
        pulses_steps (List[Tuple[int,int]]): A list of tuple, where each tuple
        represent the number of pulses and steps of a rhythm.
    Returns:
        List[int]: A list of offset values for the rhythms in pulses_steps
    """
    offsets = []
    for i, (pulses, steps) in enumerate(pulses_steps):
        lcm = pulses * steps // math.gcd(pulses, steps)
        #offset = (lcm // pulses - 1) * steps % pulses
        offset = (steps - pulses * (steps // pulses)) % steps
        offsets.append(offset)
    return offsets

from biotuner.metrics import dyad_similarity
from biotuner.biotuner_utils import gcd

# This function will allow to simply visualize the harmonic similarity of any pair of frequency.
from ipywidgets import interact, IntSlider

def viz_harmsim(x, y, savefig=False, savename='test', n_fund=10):
    """
    This function allow to simply visualize the harmonic similarity of any pair of frequency.

    Parameters
    ----------
    x : int
        Numerator frequency.
    y : int
        Denominator frequency.
    savefig : bool, optional
        If True, the figure will be saved in the current working directory.
    savename : str, optional
        The name of the file to save the figure as.
    n_fund : int, optional
        Number of fundamental frequencies to plot.

    Returns
    -------
    None

    """
    # Compute harmonic similarity
    HS = dyad_similarity(x/y)
    print('Harmonic similarity : {}'.format(str(np.round(HS, 2))))

    # Find fundamental frequency
    fund = gcd(x, y)
    print('Fundamental frequency : {}'.format(str(fund)))

    # Compute the harmonic series of the fundamental and the two frequencies
    harm_series = [fund*x for x in range(1, 70)]
    x_series = [x*a for a in range(1, 100)]
    y_series = [y*a for a in range(1, 100)]

    # Initialize the figure
    plt.figure(figsize = (7, 4))

    # plot all the harmonics with different colors
    for h in harm_series:
        plt.axvline(x = h, color = 'black',linewidth=3, label = 'Fund harmonic series')
    for xs in x_series:
        plt.axvline(x = xs, color = 'deeppink', linewidth=3, label = 'Numerator ({}Hz)'.format(x), ymin=0.4, ymax=0.8)
    for ys in y_series:
        plt.axvline(x = ys, color = 'darkorange', linewidth=3, label = 'Denominator ({}Hz)'.format(y), ymin=0, ymax=0.4)

    plt.xlim(0, fund*n_fund)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    plt.title('Harmonic similarity = {}%'.format(np.round(HS, 2)), fontsize=18)
    plt.xlabel('Frequency (Hz)', fontsize=14)
    if savefig == True:
        plt.savefig('{}.png'.format(savename), dpi=300)

# Create interactive sliders for x and y
'''interact(viz_harmsim,
         x=IntSlider(min=1, max=100, step=1, value=20),
         y=IntSlider(min=1, max=100, step=1, value=30),
         savefig=False,
         savename='test',
         n_fund=IntSlider(min=1, max=100, step=1, value=10))'''

        

def viz_interharm_conc(x:int, y:int, n_harms:int, savefig:bool=False):
    """
    Derive the harmonic series of each frequency and find common harmonics between the two. 
    Plot the harmonics and shared inter-harmonics on a graph. 
    
    Parameters:
    x (int): Numerator frequency.
    y (int): Denominator frequency.
    n_harms (int): Number of harmonics to consider.
    savefig (bool): If True, save the graph to a file. Default is False.
    
    Returns:
    None
    """
    # derive the harmonic series of each frequency
    x_series = [x*a for a in range(1, n_harms)]
    y_series = [y*a for a in range(1, n_harms)]

    # find common harmonics
    commons = list(set(x_series).intersection(y_series))
    print(commons)

    # Initialize figure
    plt.figure(figsize = (7, 4))

    # plot harmonics and the shared inter-harmonics
    for x_ in x_series:
        plt.axvline(x = x_, color = 'deeppink', linewidth=2, label = 'Numerator ({})'.format(str(x)), ymin=0, ymax=0.33)
    for y_ in y_series:
        plt.axvline(x = y_, color = 'darkorange', linewidth=2, label = 'Denominator ({})'.format(str(y)), ymin=0.33, ymax=0.66)
    for z in commons:
        plt.axvline(x = z, color = 'black', linewidth=4, label = 'Shared inter-harmonics', ymin=0.66, ymax=1)
        
    # plot parameters and label handling
    plt.xlim(0, np.max(y_series))
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    plt.title('Inter-harmonic concordance \nfor the ratio {}/{} with {} harmonics'.format(str(x), str(y), str(n_harms)), fontsize=18)
    plt.xlabel('Frequency (Hz)', fontsize=14)
    plt.xticks(range(0, np.max(y_series), 5))
    plt.show()
    if savefig is True:
        plt.savefig('interharmonic_concordance_{}-{}.png'.format(str(x), str(y)), dpi=300)
        
        
    
from biotuner.metrics import euler
from biotuner.biotuner_utils import lcm, prime_factors
def viz_euler(chord: list, savefig = False) -> None:
    """
    Visualize the least common multiple of a chord using a line plot.
    
    Parameters:
    chord (list): A list of integers representing the chord.
    savefig (bool, optional): Save the figure as an image file, default to False.
    
    Returns:
    None
    """
    x = chord[0]
    y = chord[1]
    z = chord[2]

    # create series by multiplying with range of 1 to 100
    x_series = [x*a for a in range(1, 100)]
    y_series = [y*a for a in range(1, 100)]
    z_series = [z*a for a in range(1, 100)]

    # calculate the least common multiple of the integers in the chord list
    lcm_ = lcm(*chord) 
    factors_ = prime_factors(lcm_)
    print('Euler Gradus Suavitatis = {}'.format(euler(*chord)))
    print('Prime factors are {}'.format(str(factors_)))
    # create a figure with size of (10,5)
    plt.figure(figsize = (10, 5)) 

    # plot the three series
    for x_ in x_series:
        plt.axvline(x = x_, color = 'deeppink', linewidth=1.5, label = str(chord[0]), ymin=0, ymax=0.33)
        
    for y_ in y_series:
        plt.axvline(x = y_, color = 'darkorange', linewidth=1.5, label = str(chord[1]), ymin=0.33, ymax=0.66)
        
    for z_ in z_series:
        plt.axvline(x = z_, color = 'darkblue', linewidth=1.5, label = str(chord[2]), ymin=0.66, ymax=1)

    # plot a black vertical line for the least common multiple of the integers in the chord list 
    plt.axvline(x = lcm_, color = 'black', linewidth=5, label = 'Least common \nmultiple : '+str(lcm_), ymax=1)

    plt.xlim(0, lcm_+5) # set x-axis limit
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    plt.title('Least common multiple of chord {}:{}:{}'.format(str(x), str(y), str(z)), fontsize=18)
    plt.xlabel('Frequency (Hz)', fontsize=14)
    plt.show()
    if savefig is True:
        plt.savefig('euler-{}-{}-{}.png'.format(str(x), str(y), str(z)), dpi=300) # save the figure as an image file


from biotuner.scale_construction import find_MOS, tuning_to_radians
def plot_labyrinth(generator_intervals, max_steps=53, octave=2):
    """
    Plot a labyrinth of Moment of Symmetry (MOS) scales given a list of generator intervals.

    This function calculates MOS scales for each generator interval and plots them on a polar
    coordinate system. Each generator interval is represented by a different color.

    Parameters
    ----------
    generator_intervals : list of int or float
        A list of generator intervals for which MOS scales will be calculated.
    max_steps : int, optional, default: 53
        The maximum number of steps to consider for each MOS scale calculation.
    octave : int, optional, default: 2
        The octave size for which the MOS scales will be calculated.

    Returns
    -------
    None

    Examples
    --------
    >>> generator_intervals = [4/3, 3/2, 9/5]
    >>> plot_labyrinth(generator_intervals)
    """
    # Calculate MOS scales for each generator interval
    MOS_by_generator = {}
    for interval in generator_intervals:
        MOS_by_generator[interval] = find_MOS(interval, max_steps=max_steps, octave=octave)

    # Set up plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Define color cycle
    color_cycle = plt.cm.Set1(np.linspace(0, 1, 10))

    # Plot MOS scales for each generator interval
    col = 0
    scale_steps = []
    for i, interval in enumerate(generator_intervals):
        print(i)
        MOS = MOS_by_generator[interval]
        for j in range(len(MOS['tuning'])):
            radii, angles = tuning_to_radians(interval, MOS['steps'][j])
            signature = tuple(MOS['sig'][j])
            radius = MOS['sig'][j].index(max(signature)) + 1
            color = color_cycle[i]
            ax.plot(angles, np.repeat(radius, len(angles)), color=color, alpha=0.5)
            scale_steps.append(MOS['steps'][j])
            col += 1

    # Set title and axis labels
    ax.set_title('Labyrinth of Moment of Symmetry Scales', fontsize=16)
    ax.set_rlabel_position(22.5)
    ax.set_rticks(range(1, max([MOS_by_generator[interval]['sig'][-1][0] for interval in generator_intervals]) + 1))
    ax.set_rlim(0, max([MOS_by_generator[interval]['sig'][-1][0] for interval in generator_intervals]) + 1)
    ax.set_xlabel('Generator Interval', labelpad=15, fontsize=12)

    # Add legends
    legend_handles_generator = []
    legend_handles_steps = []
    for i, interval in enumerate(generator_intervals):
        legend_handles_generator.append(plt.Line2D([], [], color=color_cycle[i], label=str(interval)))
    #for i in range(len(scale_steps)):
    #    legend_handles_steps.append(plt.Line2D([], [], color=color_cycle[i], label=str(scale_steps[i])))

    legend1 = ax.legend(handles=legend_handles_generator, title='Generator Interval', loc=(1.1, 0.1), fontsize=12)
    ax.add_artist(legend1)
    #ax.legend(handles=legend_handles_steps, title='MOS Scale Steps', loc=(1.1, 0.4), fontsize=12)

    plt.xlim(-3.14, 3.14)
    plt.ylim(-3.14, 3.14)

    plt.show()
    
    
from IPython.display import display
import ipywidgets as widgets
from scipy.fft import rfft, rfftfreq



def interactive_signal_with_sidebands(sample_rate=44100):
    def generate_signal_with_sidebands(sf, time_end, freqs, amps, sidebands, sb_amp, phases, im_order):
        time = np.arange(0, time_end, 1 / sf)
        sine_tot = []
        mod_freq = min(freqs)
        carrier_freq = max(freqs)
        mod_amp = amps[freqs.index(mod_freq)]

        for idx, freq in enumerate(freqs):
            if freq == mod_freq:
                sinewave = mod_amp * np.sin(2 * np.pi * freq * time + phases[idx])
            elif freq == carrier_freq:
                sinewave = np.sin(2 * np.pi * freq * time + phases[idx])
            else:
                sinewave = amps[freqs.index(freq)] * np.sin(2 * np.pi * freq * time + phases[idx])
            sine_tot.append(sinewave)

        # Add sidebands
        for sb in range(sidebands):
            sb_freq = carrier_freq + (sb+1) * mod_freq
            sb_wave = sb_amp * np.sin(2 * np.pi * sb_freq * time + phases[0])  # Modify this line according to the desired phase coupling
            sine_tot.append(sb_wave)
        # Calculate intermodulation components
        if im_order > 0:
            im_pairs = [(freqs[0], freqs[1]), (freqs[0], freqs[2]), (freqs[1], freqs[2])]
            for pair in im_pairs:
                im_freqs, im_orders = compute_IMs(pair[0], pair[1], im_order)
                for idx, im_freq in enumerate(im_freqs):
                    im_sinewave = sb_amp * np.sin(2 * np.pi * im_freq * time + phases[0])
                    sine_tot.append(im_sinewave)

        sine_tot = sum_list(sine_tot)
        # Calculate intermodulation components


        return sine_tot

    def update_plot(freq1, freq2, freq3, amp1, amp2, amp3, sidebands, sb_amp, phase1, phase2, phase3, im_order):

        phases = [phase1, phase2, phase3]
        time_end = 1
        sf = sample_rate

        freqs = [freq1, freq2, freq3]
        amps = [amp1, amp2, amp3]

        signal = generate_signal_with_sidebands(sf, time_end, freqs, amps, sidebands, sb_amp, phases, im_order)

        # Compute FFT
        yf = rfft(signal)
        xf = rfftfreq(len(signal), 1/sf)

        # Find top 5 frequencies with highest amplitude
        top_freqs_idx = np.argsort(-np.abs(yf))[:5]
        top_freqs = xf[top_freqs_idx]
        top_amps = np.abs(yf[top_freqs_idx])

        # Compute harmonic similarity of frequencies
        ratios_freqs = compute_peak_ratios(freqs, rebound=True, octave=2, sub=False)
        harm_sim_freqs = np.mean(ratios2harmsim(ratios_freqs))

        # Compute harmonic similarity of signal peaks
        ratios_signal = compute_peak_ratios(top_freqs, rebound=True, octave=2, sub=False)
        harm_sim_signal = np.mean(ratios2harmsim(ratios_signal))

        # Clear the plot and plot the updated signal
        plt.clf()
        plt.plot(np.arange(len(signal)) / float(sf), signal, color='deeppink', label=f'Gen freqs: {harm_sim_freqs:.2f}\nSignal peaks: {harm_sim_signal:.2f}')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('Generated signal')
        plt.legend(loc='upper right', title='Harmonic similarity', fontsize=10)
        plt.show()

        # Print the frequency and amplitude values chosen by the user
        #print(f'Frequency values: {freqs}')
        print(f'Retrieved freq values: {top_freqs}')
        #print(f'Harmonic similarity of gen freqs: {harm_sim_freqs}')
        #print(f'Harmonic similarity of signal peaks: {harm_sim_signal}')

    phase1_slider = widgets.FloatSlider(min=0, max=2*np.pi, value=0, step=np.pi/16, description='Phase 1:')
    phase2_slider = widgets.FloatSlider(min=0, max=2*np.pi, value=0, step=np.pi/16, description='Phase 2:')
    phase3_slider = widgets.FloatSlider(min=0, max=2*np.pi, value=0, step=np.pi/16, description='Phase 3:')

    freq1_slider = widgets.FloatSlider(min=1, max=20, value=2, step=0.5, description='Freq 1:')
    freq2_slider = widgets.FloatSlider(min=1, max=50, value=5, step=0.5, description='Freq 2:')
    freq3_slider = widgets.FloatSlider(min=1, max=100, value=10, step=0.5, description='Freq 3:')
    amp1_slider = widgets.FloatSlider(min=0, max=1, value=0.3, step=0.05, description='Amp 1:')
    amp2_slider = widgets.FloatSlider(min=0, max=1, value=0.5, step=0.05, description='Amp 2:')
    amp3_slider = widgets.FloatSlider(min=0, max=1, value=0.2, step=0.05, description='Amp 3:')
    sidebands_slider = widgets.IntSlider(min=0, max=10, value=0, step=1, description='Sidebands:')
    sb_amp_slider = widgets.FloatSlider(min=0, max=1, value=0.1, step=0.05, description='Side. Amp:')
    im_order_slider = widgets.IntSlider(min=0, max=6, value=0, step=1, description='IM Order:')

    out = widgets.interactive(update_plot,
                            freq1=freq1_slider,
                            freq2=freq2_slider,
                            freq3=freq3_slider,
                            amp1=amp1_slider,
                            amp2=amp2_slider,
                            amp3=amp3_slider,
                            sidebands=sidebands_slider,
                            sb_amp=sb_amp_slider,
                            phase1=phase1_slider,
                            phase2=phase2_slider,
                            phase3=phase3_slider,
                            im_order=im_order_slider)


    display_box = widgets.VBox([out])
    return display_box


import warnings
from biotuner.biotuner_utils import listen_scale
import ipywidgets as widgets
from IPython.display import display

def MOS_interactive():
    def plot_MOS_labyrinth(generator_intervals, max_steps=20):
        MOS_by_generator = {}
        for interval in generator_intervals:
            MOS_by_generator[interval] = find_MOS(interval, max_steps=max_steps)

        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={'projection': 'polar'})
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        color_cycle = plt.cm.Set2(np.linspace(0, 1, len(generator_intervals)))

        angle_dict = defaultdict(set)
        shared_angles = set()

        for i, interval in enumerate(generator_intervals):
            MOS = MOS_by_generator[interval]
            max_scale_steps = max(MOS['steps'])
            
            for j in range(len(MOS['tuning'])):
                tuning = MOS['tuning'][j]
                steps = MOS['steps'][j]

                radians, _ = tuning_to_radians(interval, steps)

                for angle in radians:
                    scale_info = (interval, steps)
                    if angle not in angle_dict:
                        angle_dict[angle] = set()
                    angle_dict[angle].add(scale_info)

                ax.plot(radians, np.arange(1, steps + 1), 'o-', markersize=5, linewidth=1.5, color=color_cycle[i],
                        label=f'{interval:.2f} ({steps} steps)' if steps == max_scale_steps else None)

        for angle, scale_info_set in angle_dict.items():
            intervals = set([scale_info[0] for scale_info in scale_info_set])
            if len(intervals) > 1:
                shared_angles.add(angle)

        for angle in shared_angles:
            ax.plot([angle, angle], [0, max_steps + 1], 'black', linewidth=1, linestyle='--')

        ax.set_title('Moment of Symmetry scales for different generator intervals', fontsize=16)
        ax.set_rlabel_position(22.5)
        ax.set_rticks(np.arange(1, max_steps + 1, 1))
        ax.set_rlim(0, max_steps + 1)
        ax.set_ylim(0, max_steps + 1)  # Center the labyrinth

        handles, labels = ax.get_legend_handles_labels()
        new_labels = []
        for label in labels:
            interval, steps = label.split('(')
            interval = float(interval)
            new_label = f'{interval:.2f} ({steps.rstrip(")")})'
            if interval in generator_intervals:
                new_labels.append(new_label)
                
        ax.legend(handles, new_labels, title='Generator Interval (steps)', fontsize=10, loc='best')
        plt.show()


    def play_tuning(button):
        fund = 100  # Set the fundamental frequency (e.g., A4 = 440Hz)
        length = 500  # Set the duration for each note in milliseconds
        active_intervals = [interval.value for interval in interval_widgets]
        active_intervals[0]
        print(active_intervals[0])
        MOS = find_MOS(active_intervals[0], max_steps=max_steps_slider.value)

        highest_steps_scale = MOS["tuning"][-1]
        
        if highest_steps_scale is not None:
            listen_scale(highest_steps_scale, fund, length)
        else:
            print("No MOS found for the given generator intervals.")

    play_button = widgets.Button(description="Play Tuning", button_style="success", layout=widgets.Layout(width="50%"))
    play_button.on_click(play_tuning)

    warnings.filterwarnings('ignore')  # Suppress warnings

    def interactive_plot(interval_1, interval_2, interval_3, interval_4, interval_5, max_steps):
        generator_intervals = [interval_1, interval_2, interval_3, interval_4, interval_5]
        active_intervals = [toggle.value for toggle in toggle_widgets]
        active_generator_intervals = [interval for i, interval in enumerate(generator_intervals) if active_intervals[i]]
        plot_MOS_labyrinth(active_generator_intervals, max_steps)

    def create_interval_widget(value):
        return widgets.FloatSlider(min=1, max=2, step=0.01, value=value, description='', layout=widgets.Layout(width='50%'))

    def create_toggle_widget(description):
        return widgets.ToggleButton(value=True, description=description, button_style='info', layout=widgets.Layout(width='50%'))

    def update_plot(change):
        active_intervals = [toggle.value for toggle in toggle_widgets]
        active_generator_intervals = [interval for i, interval in enumerate(intervals) if active_intervals[i]]
        plot_MOS_labyrinth(active_generator_intervals, max_steps_slider.value)
        
    intervals = [1.25, 1.25, 1.25, 1.25, 1.25]
    interval_widgets = [create_interval_widget(value) for value in intervals]
    toggle_widgets = [create_toggle_widget(f"Interval {i+1}") for i in range(len(intervals))]
    max_steps_slider = widgets.IntSlider(min=5, max=50, step=1, value=20, description='Max Steps:', layout=widgets.Layout(width='50%'))

    interact_kwargs = {"interval_" + str(i + 1): interval_widgets[i] for i in range(len(intervals))}
    interact_kwargs["max_steps"] = max_steps_slider

    ui = widgets.VBox([widgets.HBox([toggle_widgets[i], interval_widgets[i]]) for i in range(len(intervals))] + [max_steps_slider, play_button])
    out = widgets.interactive_output(interactive_plot, interact_kwargs)
    display(ui, out)
    update_plot(None)

 
def visualize_rhythms_interactive():
    def visualize_rhythms(pulses_steps, plot_size=600, offsets=None, tolerance=0.1):
        """
        Visualize multiple Euclidean rhythms.
        """
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']
        pulses_positions = []
        fig = go.FigureWidget()
        for i, (pulses, steps) in enumerate(pulses_steps):
            plt.clf()
            offset = offsets[i] if offsets else 0
            rhythm = euclidean_rhythm(pulses, steps, offset)
            angles = np.linspace(0, 2*np.pi, steps, endpoint=False)
            radius = (i+1) * 0.15
            x = radius * np.cos(angles)
            y = radius * np.sin(angles)
            fig.add_trace(go.Scatter(x=x, y=y, mode='markers', marker=dict(size=10, color=colors[i % len(colors)], opacity=0.5)))

            pulse_pos = []

            for j, value in enumerate(rhythm):
                if value == 1:
                    fig.add_trace(go.Scatter(x=[x[j]], y=[y[j]], mode='markers', marker=dict(size=15, color=colors[i % len(colors)], opacity=1)))
                    pulse_pos.append((x[j], y[j], np.arctan2(y[j], x[j])))

            pulses_positions.append(pulse_pos)

        for i in range(len(pulses_positions)):
            for j in range(i+1, len(pulses_positions)):
                for pulse1 in pulses_positions[i]:
                    for pulse2 in pulses_positions[j]:
                        if abs(pulse1[2] - pulse2[2]) < tolerance:
                            fig.add_shape(type="line", x0=0, y0=0, x1=pulse1[0], y1=pulse1[1], line=dict(color="black", width=2))
                            fig.add_shape(type="line", x0=pulse1[0], y0=pulse1[1], x1=pulse2[0], y1=pulse2[1], line=dict(color="black", width=2))

        fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
        fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
        fig.update_layout(width=plot_size, height=plot_size)
        fig.show()
    # Add an output widget to display the plot
    plot_output = widgets.Output()

    def on_value_change(change):
        pulses_steps = [(pulses_sliders[i].value, steps_sliders[i].value) for i in range(len(pulses_sliders))]
        
        with plot_output:
            clear_output(wait=True)
            visualize_rhythms(pulses_steps)

    num_rhythms = 3

    pulses_sliders = [widgets.IntSlider(min=1, max=20, value=5, description=f'Pulses {i + 1}:') for i in range(num_rhythms)]
    steps_sliders = [widgets.IntSlider(min=1, max=20, value=8, description=f'Steps {i + 1}:') for i in range(num_rhythms)]

    # Observe changes in the sliders and update the plot accordingly
    for slider in pulses_sliders + steps_sliders:
        slider.observe(on_value_change, names='value')

    ui = widgets.VBox([widgets.HBox([pulses_sliders[i], steps_sliders[i]]) for i in range(num_rhythms)])
    display(ui, plot_output)  # Display the output widget along with the UI
    on_value_change(None)
    return



