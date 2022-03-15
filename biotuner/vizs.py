from numpy import sin, pi, linspace
from pylab import plot, subplot
from matplotlib.pyplot import figure
import matplotlib.colors as mcolors
from biotuner.biotuner_utils import scale2frac
import matplotlib.pyplot as plt

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


def graph_psd_peaks(freqs, psd, peaks, xmin, xmax):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(freqs, psd, color='deeppink')
    plt.xlim([xmin, xmax])
    plt.ylim([-150, -100])
    plt.xlabel('Frequency (Hertz)', size=14)
    plt.ylabel('PSD [V**2/Hz]', size=14)
    plt.title('Power Spectrum Density using Welch Method', size=18)
    for xc in peaks:
        plt.axvline(x=xc, c='black', linestyle='dotted')
