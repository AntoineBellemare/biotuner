import numpy as np
from PyEMD import EMD, EEMD
from scipy.signal import butter, lfilter
import matplotlib.pyplot as plt
import seaborn as sbn
from scipy import stats
from numpy import array, zeros, ones, arange, log2, sqrt, diff, concatenate
import secrets
import biotuner
from biotuner.biotuner_functions import *
from biotuner.biotuner_utils import *
from biotuner.biotuner_object import *
#from biotuner_object import *


'''BIOTUNER 2D'''


def surrogate_signal(data, surr_type = 'pink', low_cut = 0.5, high_cut = 150, sf = 1000, TFT_freq = 5):
    if surr_type == 'AAFT':
        indexes = [x for x in range(len(data))]
        data_ = np.stack((data, indexes))
        data_ = AAFT_surrogates(Surrogates, data_)
        data_ = data_[0]
        data_ = butter_bandpass_filter(data_, low_cut, high_cut, sf, 4)
    if surr_type == 'TFT':
        data_ = UnivariateSurrogatesTFT(data,1,fc=TFT_freq)
    if surr_type == 'phase':
        len_data = len(data)
        data_ = phaseScrambleTS(data)
        data_ = butter_bandpass_filter(data_[0:len_data], low_cut, high_cut, sf, 4)
    if surr_type == 'shuffle':
        data_ = data.copy()
        np.random.shuffle(data_)
        data_ = butter_bandpass_filter(data_, low_cut, high_cut, sf, 4)
    if surr_type == 'white':
        beta = 0 
    if surr_type == 'pink':
        beta  = 1
    if surr_type == 'brown':
        beta = 2
    if surr_type == 'blue':
        beta  = -1
    if surr_type == 'white' or surr_type == 'pink' or surr_type == 'brown' or surr_type == 'blue':
        data_ = cn.powerlaw_psd_gaussian(beta, len(data))
        data_ = butter_bandpass_filter(data_, low_cut, high_cut, sf, 4)
    return data_
    
def surrogate_signal_matrices(data, surr_type = 'pink', low_cut = 0.5, high_cut = 150, sf = 1000):
    data_ = data.copy()
    if np.ndim(data) == 2:
        for i in range(len(data)):
            data_[i] = surrogate_signal(data[i], surr_type = surr_type, low_cut = low_cut, high_cut = high_cut, sf = sf)
    if np.ndim(data) == 3:
        for i in range(len(data)):
            for j in range(len(data[i])):
                data_[i][j] = surrogate_signal(data[i][j], surr_type = surr_type, low_cut = low_cut, high_cut = high_cut, sf = sf)
    return data_

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

def graph_surrogates(data, sf, conditions = ['eeg', 'pink'], metric_to_graph = 'harmsim', peaks_function = 'adapt', 
                     precision = 0.5, savefolder = None, tag = '-', low_cut = 0.5, high_cut = 150, colors = None, 
                     display = False, save = True, n_harmonic_peaks = 5, min_harms = 2):
    peaks_avg_tot = []
    metric_tot = []
    for c in conditions:
        if c != 'eeg':
            data_ = surrogate_signal_matrices(data, surr_type = c, low_cut = low_cut, high_cut = high_cut, sf = sf)
        else:
            data_ = butter_bandpass_filter(data, low_cut, high_cut, sf, 4)
        peaks_avg = []
        metric = []
        for t in range(len(data_)):
            _data_ = data_[t][:]
            biotuning = biotuner(sf, peaks_function = peaks_function, precision = precision, n_harm = 10,
                            ratios_n_harms = 10, ratios_inc_fit = False, ratios_inc = False) # Initialize biotuner object
            biotuning.peaks_extraction(_data_, ratios_extension = False, max_freq = 60,min_harms = min_harms)
            if peaks_function == 'harmonic_peaks':
                biotuning.peaks = [x for _, x in sorted(zip(biotuning.amps, biotuning.peaks))][::-1][0:n_harmonic_peaks]
                biotuning.amps = sorted(biotuning.amps)[::-1][0:n_harmonic_peaks]
            #print(biotuning.peaks)
            biotuning.compute_peaks_metrics()
            peaks_avg.append(np.average(biotuning.peaks))
            try:
                metric.append(biotuning.peaks_metrics[metric_to_graph])
            except:
                if metric_to_graph == 'sum_p_q' or metric_to_graph =='sum_distinct_intervals' or metric_to_graph == 'metric_3' 'sum_p_q_for_all_intervals' or metric_to_graph =='sum_q_for_all_intervals' or metric_to_graph == 'matrix_harm_sim' or metric_to_graph =='matrix_cons':
                    scale_metrics, _ = scale_to_metrics(biotuning.peaks_ratios)
                    #print(scale_metrics[metric_to_graph])
                    metric.append(float(scale_metrics[metric_to_graph]))
                if metric_to_graph == 'dissonance' or metric_to_graph == 'diss_n_steps' or metric_to_graph == 'diss_harm_sim':
                    biotuning.compute_diss_curve(plot = False, input_type = 'peaks', denom = 100, max_ratio = 2, n_tet_grid = 12)
                    metric.append(biotuning.scale_metrics[metric_to_graph])
        metric_tot.append(metric)
        peaks_avg_tot.append(np.average(peaks_avg))
        #print(run)
    print(peaks_function, ' peaks freqs ', peaks_avg_tot)
    graph_dist(metric_tot, metric = metric_to_graph, ref = metric_tot[0], dimensions = [0], labs = conditions, savefolder = savefolder,         subject = '2', tag = tag, adapt = 'False', peaks_function = peaks_function, colors = colors, display = display, save = save)
    
    

def graph_dist(dist, metric = 'diss', ref = None, dimensions = [0, 1], labs = ['eeg', 'phase', 'AAFT', 'pink', 'white'], savefolder = '\\', subject = '0', tag = '0', adapt = 'False', peaks_function = 'EEMD', colors = None, display = False, save = True):
    #print(len(dist), len(dist[0]), len(dist[1]), len(dist[2]), len(dist[3]))
    #if ref == None:
    #    ref = dist[0]
    if metric == 'dissonance':
        m = 'Dissonance (From Sethares (2005))'
    if metric == 'euler':
        m = 'Consonance (Euler <Gradus Suavitatis>)'
    if metric == 'diss_euler':
        m = 'Consonance (Euler <Gradus Suavitatis>) of dissonant minima'
    if metric == 'diss_n_steps':
        m = 'Number of dissonant minima'
    if metric == 'diss_harm_sim':
        m = 'Harmonic similarity of scale derived from dissonance curve'
    if metric == 'tenney':
        m = 'Tenney Height'
    if metric == 'harmsim':
        m = 'Harmonic similarity of peaks'
    if metric == 'diss_harm_sim':
        m = 'Harmonic similarity of diss scale'
    if metric == 'harm_fit':
        m = 'Harmonic fitness between peaks'
    if metric == 'cons':
        m = 'Averaged consonance of all paired peaks ratios'
    if metric == 'n_harmonic_peaks':
        m = 'Number of harmonic peaks'
    if metric == 'matrix_harm_sim':
        m = 'Harmonic similarity of peaks ratios intervals'
    if metric == 'matrix_cons':
        m = 'Consonance of peaks ratios intervals'
    if metric == 'metric_3':
        m = 'Pytuning consonance metric'
    if metric == 'sum_distinct_intervals':
        m = 'Sum of distinct intervals' 
    if metric == 'sum_p_q_for_all_intervals':
        m = 'Sum of num and denom for all intervals' 
    if metric == 'sum_q_for_all_intervals':
        m = 'Sum of denom for all intervals' 
        
        
        
        

    plt.rcParams['axes.facecolor'] = 'black'
    if display == True:
        fig = plt.figure(figsize=(11,7))
    else:
        fig = plt.figure(figsize=(14,10))
        
    if colors == None:
        colors = ['cyan', 'deeppink', 'white', 'yellow', 'blue', 'orange', 'red'] 
    
    xcoords = []
    
    
    for dim in dimensions:
        labs = labs
        if dim == 0:
            dimension = 'trials'
        if dim == 1:
            dimension = 'channels'
        
        
        for d, color, enum in zip(dist, colors, range(len(dist))):
            #d = d[~np.isnan(d)]
            
            d = [x for x in d if str(x) != 'nan']
            ref = [x for x in ref if str(x) != 'nan']
            if dimensions == [0]:
                sbn.distplot(d, color =color)
                secure_random = secrets.SystemRandom()
                if len(d) < len(ref):
                    ref = secure_random.sample(list(ref), len(d))
                if len(ref) < len(d):
                    d = secure_random.sample(list(d), len(ref))
                #print('d', d)
                #print('ref', ref)
                t, p = stats.ttest_rel(d, ref)
                #print('p value', p)
            if dimensions != [0]:
                sbn.distplot(np.nanmean(d, dim), color =color)
                secure_random = secrets.SystemRandom()
                if len(d) < len(ref):
                    ref = secure_random.sample(list(ref), len(d))
                if len(ref) < len(d):
                    d = secure_random.sample(list(d), len(ref))
                t, p = stats.ttest_rel(np.nanmean(ref, dim), np.nanmean(d, dim))
                
            if p < 0.05:
                labs[enum] = labs[enum]+' *'
                #xcoords.append(np.average(d))

                #for xc in xcoords:
                #    plt.axvline(x=xc, c='white')

        if len(labs) == 2:
            fig.legend(labels=[labs[0], labs[1]], 
                   loc = [0.69, 0.65], fontsize = 15, facecolor = 'white')
        if len(labs) == 3:
            fig.legend(labels=[labs[0], labs[1], labs[2]], 
                   loc = [0.69, 0.65], fontsize = 15, facecolor = 'white')
        if len(labs) == 4:
            fig.legend(labels=[labs[0], labs[1], labs[2], labs[3]], 
                   loc = [0.69, 0.65], fontsize = 15, facecolor = 'white')  
        if len(labs) == 5:
            fig.legend(labels=[labs[0], labs[1], labs[2], labs[3], labs[4]], 
                   loc = [0.69, 0.63], fontsize = 15, facecolor = 'white') 
        if len(labs) == 6:
            fig.legend(labels=[labs[0], labs[1], labs[2], labs[3], labs[4], labs[5]], 
                   loc = [0.69, 0.62], fontsize = 15, facecolor = 'white') 
        plt.xlabel(m, fontsize = '16')
        plt.ylabel('Proportion of samples', fontsize = '16')
        #plt.xlim([0.25, 0.7])
        plt.grid(color='white', linestyle='-.', linewidth=0.7)
        plt.suptitle('Comparing ' + m+ ' \nfor EEG, surrogate data, and noise signals across ' + dimension, fontsize = '20')
        
        if save == True:
            fig.savefig(savefolder+'{}_distribution_s{}-bloc{}_{}_{}.png'.format(metric, subject, tag, dimension, peaks_function), dpi=300)
            plt.clf()
        if display == True:
            plt.rcParams["figure.figsize"] = (5,3)
            plt.show()
        
        
def diss_curve_multi (freqs, amps, denom=10, max_ratio=2, bound = 0.1):
    from numpy import array, linspace, empty, concatenate
    from scipy.signal import argrelextrema
    from fractions import Fraction
    plt.figure(figsize=(18, 8))
    diss_minima_tot = []
    for fr, am in zip(freqs, amps):
        freqs = np.array([x*128 for x in fr])
        am = np.interp(am, (np.array(am).min(), np.array(am).max()), (0.3, 0.7))
        r_low = 1
        alpharange = max_ratio
        method = 'min'

        n = 1000
        diss = empty(n)
        a = concatenate((am, am))
        for i, alpha in enumerate(linspace(r_low, alpharange, n)):
            f = concatenate((freqs, alpha*freqs))
            d = dissmeasure(f, a, method)
            diss[i] = d

        
        plt.plot(linspace(r_low, alpharange, len(diss)), diss)
        plt.xscale('log')
        plt.xlim(r_low, alpharange)

        plt.xlabel('frequency ratio')
        plt.ylabel('sensory dissonance')


        diss_minima = argrelextrema(diss, np.less)
        diss_minima_tot.append(list(diss_minima[0]))
        #print(diss_minima)
    
    diss_tot = [item for sublist in diss_minima_tot for item in sublist]
    diss_tot.sort()
    new_minima = []
    
    for i in range(len(diss_tot)-1):
        if (diss_tot[i+1] - diss_tot[i]) < bound:
            new_minima.append((diss_tot[i]+diss_tot[i+1])/2)
    #print(new_minima)
    intervals = []
    for d in range(len(new_minima)):
        #print(new_minima[d])
        frac = Fraction(new_minima[d]/(n/(max_ratio-1))+1).limit_denominator(denom)
        print(frac)
        frac = (frac.numerator, frac.denominator)
        intervals.append(frac)
    
    #intervals = [(123, 100), (147, 100), (159, 100), (9, 5), (2, 1)]
    intervals.append((2, 1))
    #print(intervals)
    for n, d in intervals:
        plt.axvline(n/d, color='silver')
    plt.axvline(1.001, linewidth=1, color='black')
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.axhline(linewidth=1, color='black')
    
    plt.xscale('linear')
    plt.minorticks_off()
    plt.xticks([n/d for n, d in intervals],
               ['{}/{}'.format(n, d) for n, d in intervals], fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.tight_layout()
    plt.show()
    return diss_minima_tot

def graph_conditions(data, sf, conditions, metric_to_graph, peaks_function, precision, savefolder, run, FREQ_BANDS = None):
    def remove_zeros(list_):
        for x in range(len(list_)):
            if list_[x] == 0.:
                list_[x] = 0.1 
        return list_
    peaks_avg_tot = []
    metric_tot = []
    for cond in range(len(data)):
        data_ = data[cond]
        peaks_avg = []
        metric = []
        for t in range(len(data_)):
            _data_ = data_[t][:]
            biotuning = biotuner(sf, peaks_function = peaks_function, precision = precision, n_harm = 10,
                             ratios_n_harms = 10, ratios_inc_fit = False, ratios_inc = False) # Initialize biotuner object
            print('data', _data_.shape)
            biotuning.peaks_extraction(np.array(_data_,dtype='float64'), ratios_extension = True, max_freq = 50, min_harms = 2,
                                      FREQ_BANDS = FREQ_BANDS)
            biotuning.peaks = remove_zeros(biotuning.peaks)[0:10]
            print('peaks', biotuning.peaks)
            biotuning.compute_peaks_metrics()
            peaks_avg.append(np.average(biotuning.peaks))
            #print(biotuning.peaks_metrics)
            metric.append(biotuning.peaks_metrics[metric_to_graph])
        metric_tot.append(metric)
        peaks_avg_tot.append(np.average(peaks_avg))
        #print(run)

    graph_dist(metric_tot, metric = metric_to_graph, ref = metric_tot[0], dimensions = [0], 
               labs = conditions, savefolder = savefolder, subject = '2', run = run, adapt = 'False', 
               peak_function = peaks_function)