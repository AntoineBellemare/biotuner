from numpy import sin, pi, linspace
from pylab import plot, subplot
from matplotlib.pyplot import figure
import matplotlib.colors as mcolors
from biotuner.biotuner_utils import scale2frac, NTET_ratios
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal


def lissajous_curves(tuning):
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
