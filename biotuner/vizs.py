from numpy import sin, pi, linspace
from pylab import plot, subplot
from matplotlib.pyplot import figure
import matplotlib.colors as mcolors
from biotuner.biotuner_utils import scale2frac, NTET_ratios
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal


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
    fig, ax = plt.subplots(figsize=(12, 6))
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
        plt.title('Power Spectrum Density and peaks positions using ' + method
                  + ' method', size=18)
    if method is None:
        plt.title('Power Spectrum Density and peaks positions', size=18)
    for xc in peaks:
        plt.axvline(x=xc, c='black', linestyle='dotted')




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


def graphEMD_welch(freqs_all, psd_all, peaks, raw_data, FREQ_BANDS,
                   sf, nfft, nperseg, noverlap, min_freq=1,
                   max_freq=60, precision=0.5):
    plt.rcParams["figure.figsize"] = (13, 8)
    color_line = ['aqua', 'darkturquoise', 'darkcyan', 'darkslategrey', 'black']
    for i in range(len(freqs_all)):
        plt.plot(freqs_all[i], psd_all[i], color=color_line[i])

    mult = 1/precision
    nperseg = sf*mult
    nfft = nperseg
    freqs_full, psd_full = scipy.signal.welch(raw_data, sf,
                                              nfft=nfft,
                                              nperseg=nperseg,
                                              noverlap=noverlap)
    # psd_full = np.interp(psd_full, (psd_full.min(),
#                    psd_full.max()), (0, 0.005))
    plt.text(1.7, -5, 'delta', horizontalalignment='center',
             size=15)
    plt.text(4.5, -5, 'theta', horizontalalignment='center',
             size=15)
    plt.text(9, -5, 'alpha', horizontalalignment='center',
             size=15)
    plt.text(19, -5, 'beta', horizontalalignment='center',
             size=15)
    plt.text(47, -5, 'gamma', horizontalalignment='center',
             size=15)
    plt.xlim([min_freq, max_freq])
    plt.ylim([-75, 0])
    plt.title('PSD of Empirical Mode Decomposition', size=28)
    plt.xlabel('Frequency', size=15)
    plt.ylabel('Power', size=15)
    plt.tick_params(axis='both', which='major', labelsize=12,
                    length=6, width=4)
    plt.tick_params(axis='both', which='minor', labelsize=10,
                    length=6, width=4)
    # plt.yscale('log')
    plt.xscale('log')

    psd_full = np.interp(psd_full, (psd_full.min(), psd_full.max()),
                         (-35, 0))
    plt.plot(freqs_full, psd_full,
             color='deeppink', linestyle='dashed',
             label='raw data')

    alpha = [0.6, 0.63, 0.66, 0.69, 0.72]
    shadow = 0.9
    color_bg = ['darkgoldenrod', 'goldenrod', 'orange',
                'gold', 'khaki']
    xposition = peaks
    labels = ['EMD1', 'EMD2', 'EMD3', 'EMD4', 'EMD5']
    for p, n, band in zip(peaks, range(len(labels)), FREQ_BANDS):
        if p > band[0] and p <= band[1]:
            labels[n] = labels[n]+'*'
    for xc, c, l in zip(xposition, color_line, labels):
        plt.axvline(x=xc, label='{} = {}'.format(l, xc), c=c)
    plt.axvspan(0, 3, ymin=shadow, alpha=alpha[0],
                color=color_bg[0], ec='black')
    plt.axvspan(3, 7, ymin=shadow, alpha=alpha[1],
                color=color_bg[1], ec='black')
    plt.axvspan(7, 12, ymin=shadow, alpha=alpha[2],
                color=color_bg[2], ec='black')
    plt.axvspan(12, 30, ymin=shadow, alpha=alpha[3],
                color=color_bg[3], ec='black')
    plt.axvspan(30, 70, ymin=shadow, alpha=alpha[4],
                color=color_bg[4], ec='black')
    plt.legend(loc='lower left', fontsize=16)


def graph_harm_peaks(freqs, psd, harm_peaks_fit, xmin, xmax, color='black',
                     method=None, save=False, figname='test'):
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
    method : str
        The method used to find the harmonically related peaks
    save : bool
        Whether to save the figure or not
    figname : str
        The name of the file if the figure is saved
        
    Returns
    -------
    None
    """
    #psd = np.interp(psd, (psd.min(), psd.max()), (0, 0.005))
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(freqs, psd, color=color)
    #peak_power = [psd]
    plt.xlim([xmin, xmax])
    # plt.xscale('log')
    idx_min = list(freqs).index(xmin)
    idx_max = list(freqs).index(xmax)
    ymin = np.min(psd[idx_min:idx_max])
    ymax = np.max(psd[idx_min:idx_max])
    plt.ylim([ymin, ymax])
    plt.xlabel('Frequency (Hertz)', size=14)
    plt.ylabel('PSD [V**2/Hz]', size=14)
    if method is not None:
        plt.title('Power Spectrum Density and peaks positions using ' + method
                  + ' method', size=18)
    if method is None:
        plt.title('Power Spectrum Density and peaks positions', size=18)
    color_list = ['blue', 'red', 'orange', 'turquoise', 'purple']
    y_steps = (ymax-ymin)/10
    y_list = [ymax-(y_steps*2), ymax-(y_steps*3), ymax-(y_steps*4), ymax-(y_steps*5)]
    npeaks = len(harm_peaks_fit)
    #print('npeaks', npeaks)
    for peak_info, color_harm, ys in zip(harm_peaks_fit, color_list[0:npeaks], y_list[0:npeaks]):
        peak = peak_info[0]
        harm_pos = [int(x) for x in peak_info[1]]

        harm_freq = peak_info[2]
        print(harm_freq, peak)
        try:
            harm_freq.remove(peak)
        except ValueError:
            pass

        plt.axvline(x=peak, c=color_harm, linestyle='-')

        for e, harm in enumerate(harm_freq):
            plt.axvline(x=harm, c=color_harm, linestyle='dotted')
            #print(harm, harm_pos[e])
            ax.annotate(str(harm_pos[e]), (harm, ys),
                            bbox=dict(boxstyle="square", alpha=0.2,color=color_harm),
                            xytext=(harm+0.5, ys), fontsize=12)
    if save is True:
        plt.savefig(figname, dpi=300)


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
        pulses_steps (List[Tuple[int,int]]): A list of tuple, where each tuple represent the number of pulses and steps of a rhythm.
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
def viz_harmsim(x, y, savefig=False, savename='test', n_fund=10):
    """
    This function will allow to simply visualize the harmonic similarity of any pair of frequency.

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
    harm_series = [fund*x for x in range(1, 50)]
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

    plt.xlim(50, fund*n_fund)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='upper right')
    plt.title('Harmonic similarity for the ratio 3/2 = {}%'.format(np.round(HS, 2)), fontsize=18)
    plt.xlabel('Frequency (Hz)', fontsize=14)
    plt.show()
    if savefig == True:
        plt.savefig('{}.png'.format(savename), dpi=300)
        

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
