from numpy import sin, pi, linspace
from pylab import plot, subplot
from matplotlib.pyplot import figure
import matplotlib.colors as mcolors
from biotuner.biotuner_utils import scale2frac
import matplotlib.pyplot as plt
import numpy as np


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
    # plt.xscale('log')
    plt.ylim([-150, -100])
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
                   max_freq=60):
    plt.rcParams["figure.figsize"] = (13, 8)
    color_line = ['aqua', 'darkturquoise', 'darkcyan', 'darkslategrey', 'black']
    for i in range(len(freqs_all)):
        plt.plot(freqs_all[i], psd_all[i], color=color_line[i])

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
    #plt.yscale('log')
    plt.xscale('log')

    plt.plot(freqs_full, psd_full,
             color='deeppink', linestyle='dashed')

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
