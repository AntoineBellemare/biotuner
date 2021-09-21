import numpy as np
from PyEMD import EMD, EEMD
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import seaborn as sbn
from scipy import stats
from numpy import array, zeros, ones, arange, log2, sqrt, diff, concatenate
import secrets
from biotuner import *
from biotuner_utils import *


'''BIOTUNER 2D'''


def surrogate_signal(data, surr_type = 'pink', low_cut = 0.5, high_cut = 150, sf = 1000):
    if surr_type == 'AAFT':
        indexes = [x for x in range(len(data))]
        data = np.stack((data, indexes))
        data = AAFT_surrogates(Surrogates, data)
        data = data[0]
        data = butter_bandpass_filter(data, low_cut, high_cut, sf, 4)
    if surr_type == 'phase':
        len_data = len(data)
        data = phaseScrambleTS(data)
        data = butter_bandpass_filter(data[0:len_data], low_cut, high_cut, sf, 4)
    if surr_type == 'shuffle':
        np.random.shuffle(data)
    if surr_type == 'white':
        beta = 0 
    if surr_type == 'pink':
        beta  = 1
    if surr_type == 'brown':
        beta = 2
    if surr_type == 'blue':
        beta  = -1
    if surr_type == 'white' or surr_type == 'pink' or surr_type == 'brown' or surr_type == 'blue':
        data = cn.powerlaw_psd_gaussian(beta, len(data))
        data = butter_bandpass_filter(data, low_cut, high_cut, sf, 4)
    return data
    
def surrogate_signal_matrices(data, surr_type = 'pink', low_cut = 0.5, high_cut = 150, sf = 1000):
    if np.ndim(data) == 2:
        for i in range(len(data)):
            data[i] = surrogate_signal(data[i], surr_type = surr_type, low_cut = low_cut, high_cut = high_cut, sf = sf)
    if np.ndim(data) == 3:
        for i in range(len(data)):
            for j in range(len(data[i])):
                data[i][j] = surrogate_signal(data[i][j], surr_type = surr_type, low_cut = low_cut, high_cut = high_cut, sf = sf)
    return data

def compute_peaks_matrices (data, peaks_function = 'EMD', precision = 0.25, sf = 1000, max_freq = 80, save = True, suffix = 'default'):
    if np.ndim(data) == 2:
        peaks_tot = []
        amps_tot = []
        for i in range(len(data)):
            peaks, amps = compute_peaks_ts(data[i], peaks_function = peaks_function, FREQ_BANDS = None, precision = precision, sf = sf, max_freq = max_freq) 
            peaks_tot.append(peaks)
            amps_tot.append(amps)
        peaks_total = np.array(peaks_tot)
        amps_total = np.array(amps_tot)
        if save == True:
            np.save('peaks_{}_{}'.format(peaks_function, suffix), peaks_total)
            np.save('amps_{}_{}'.format(peaks_function, suffix), amps_total)
    if np.ndim(data) == 3:
        peaks_tot = []
        amps_tot = []
        for i in range(len(data)):
            peaks_temp = []
            amps_temp = []
            for j in range(len(data[i])):
                peaks, amps = compute_peaks_ts(data[i][j], peaks_function = peaks_function, FREQ_BANDS = None, precision = precision, sf = sf, max_freq = max_freq)
                peaks_temp.append(peaks)
                amps_temp.append(amps)
            peaks_tot.append(peaks_temp)
            amps_tot.append(amps_temp)
        peaks_total = np.array(peaks_tot)
        amps_total = np.array(amps_tot)
        if save == True:
            np.save('peaks_{}_{}'.format(peaks_function, suffix), peaks_total)
            np.save('amps_{}_{}'.format(peaks_function, suffix), amps_total)
    return peaks_total, amps_total

def compute_peaks_surrogates (data, conditions, peaks_function = 'EMD', precision = 0.25, sf = 1000, max_freq = 80, low_cut = 0.5, high_cut = 150, save = False):
    peaks_tot = []
    amps_tot = []
    for e, c in enumerate(conditions):
        print('Condition (', e+1, 'of', len(conditions), '):', c)
        if c == 'og_data':
            peaks, amps = compute_peaks_matrices(data, peaks_function, precision, sf, max_freq)
        if c != 'og_data':
            data = surrogate_signal_matrices(data, surr_type = c, low_cut = low_cut, high_cut = high_cut, sf = sf)
            peaks, amps = compute_peaks_matrices(data, peaks_function, precision, sf, max_freq, save = False, suffix = 'default')
        peaks_tot.append(peaks)
        amps_tot.append(amps)
    peaks_total = np.array(peaks_tot)
    amps_total = np.array(amps_tot)
        
    return peaks_total, amps_total

def peaks_to_metrics_matrices (peaks, n_harm = 10):
    cons = []
    euler = []
    tenney = []
    harm_fit = []
    metrics_dict = {}
    if np.ndim(peaks) == 3:
        for i in range(len(peaks)):
            cons_temp = []
            euler_temp = []
            tenney_temp = []
            harm_fit_temp = []
            for j in range(len(peaks[i])):
                
                metrics, metrics_list = peaks_to_metrics(peaks[i][j], n_harm)
                cons_temp.append(metrics_list[0])
                euler_temp.append(metrics_list[1])
                tenney_temp.append(metrics_list[2])
                harm_fit_temp.append(metrics_list[3])
            cons.append(cons_temp)
            euler.append(euler_temp)
            tenney.append(tenney_temp)
            harm_fit.append(harm_fit_temp)
        metrics_dict['cons'] = cons
        metrics_dict['euler'] = euler
        metrics_dict['tenney'] = tenney
        metrics_dict['harm_fit'] = harm_fit
    return np.array([cons, euler, tenney, harm_fit]), metrics_dict


def graph_dist(dist, metric = 'diss', ref = None, dimensions = [0, 1], labs = ['eeg', 'phase', 'AAFT', 'pink', 'white'], savefolder = '\\', subject = '0', run = '0', adapt = 'False'):
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
    if metric == 'harm_fit':
        m = 'Harmonic fitness between peaks'
    if metric == 'cons':
        m = 'Averaged consonance of all paired peaks ratios'
        

    plt.rcParams['axes.facecolor'] = 'black'
    fig = plt.figure(figsize=(14,10))
    colors = ['cyan', 'deeppink', 'white', 'yellow', 'orange'] 
    
    xcoords = []
    
    
    for dim in dimensions:
        labs = labs
        if dim == 0:
            dimension = 'channels'
        if dim == 1:
            dimension = 'trials'
        
        
        for d, color, enum in zip(dist, colors, range(len(dist))):
            #d = d[~np.isnan(d)]
            #print(d.shape)
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

        if len(labs) == 2:
            fig.legend(labels=[labs[0], labs[1]], 
                   loc = [0.66, 0.68], fontsize = 16, facecolor = 'white')
        if len(labs) == 3:
            fig.legend(labels=[labs[0], labs[1], labs[2]], 
                   loc = [0.66, 0.68], fontsize = 16, facecolor = 'white')
        if len(labs) == 4:
            fig.legend(labels=[labs[0], labs[1], labs[2], labs[3]], 
                   loc = [0.66, 0.68], fontsize = 16, facecolor = 'white')                    
        plt.xlabel(m, fontsize = '16')
        plt.ylabel('Proportion of samples', fontsize = '16')
        #plt.xlim([0.25, 0.7])
        plt.grid(color='white', linestyle='-.', linewidth=0.7)
        plt.suptitle('Comparing ' + m+ ' \nfor EEG, surrogate data, and pink noise signals across ' + dimension, fontsize = '22')
        fig.savefig(savefolder+'{}_distribution_s{}-bloc{}_EMD_adapt-{}_{}.png'.format(metric, subject, run, adapt, dimension), dpi=300)
        plt.clf()