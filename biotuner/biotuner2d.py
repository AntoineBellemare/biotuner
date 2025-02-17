import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from scipy import stats
import secrets
import biotuner
from biotuner.biotuner_object import *
from biotuner.metrics import *
from biotuner.biotuner_utils import AAFT_surrogates, butter_bandpass_filter, UnivariateSurrogatesTFT, phaseScrambleTS
from biotuner.biotuner_object import compute_biotuner
import pandas as pd
from scipy import signal

# from biotuner_object import *


"""BIOTUNER 2D"""


def biotuner_mne(epochs, bt_dict, savefile=False, savename=None):
    """
    Add biotuner metrics to an MNE epochs object and optionally save metrics to a CSV file.

    Parameters
    ----------
    epochs : mne.Epochs object
        The epochs to compute biotuner metrics for.
    bt_dict : dict
        A dictionary containing the parameters to compute biotuner metrics. See the biotuner_mne function
        documentation for details on the required keys and values.
    savefile : bool (default=False)
        If True, save the biotuner metrics to a CSV file.
    savename : str (default=None)
        The filename to save the CSV file. If None, use the original epochs filename with '_biotuner.csv'
        appended to the end.

    Returns
    -------
    df : pd.DataFrame
        A dictionary containing the biotuner metrics for all trials and electrodes.
        Original metadata are included if present in the epochs file.
    """

    # Create an empty list to store the metrics for each epoch
    n_trials, n_electrodes, n_datapoints = epochs.get_data().shape
    metrics_list = []

    data = epochs.get_data()  # Get the time series data for the epochs
    for j in range(n_trials):
        for k in range(n_electrodes):
            ts = data[j, k, :]
            metrics = fit_biotuner(ts, bt_dict)
            metric_dict = {}
            for metric_name, metric_value in metrics.items():
                metric_dict[metric_name] = metric_value
            metric_dict["trial"] = j
            metric_dict["electrode"] = k
            metrics_list.append(metric_dict)

    # Save metrics to CSV
    if savefile:
        if savename is None:
            savename = epochs.filename[:-4] + "_biotuner"
        df = pd.DataFrame(metrics_list)

        # Add metadata from epochs object to the dataframe
        if "metadata" in epochs.info:
            metadata = epochs.metadata
            for key in metadata.keys():
                df[key] = np.tile(metadata[key], n_electrodes)

            df = df[["trial", "electrode"] + list(bt_dict.keys()) + list(metadata.keys())]
        else:
            df = df[["trial", "electrode"] + list(bt_dict.keys())]
        df.to_csv(savename + ".csv", index=False)

    return df


def surrogate_signal(data, surr_type="pink", low_cut=0.5, high_cut=150, sf=1000, TFT_freq=5):
    """Generate surrogate signal.

    Parameters
    ----------
    data : array (numDataPoints, )
        Original signal.
    surr_type : str
        Defaults to 'pink'.
        Type of surrogate.
        {'brown', 'pink', 'white', 'blue',
         'shuffle', 'phase', 'AAFT', 'TFT'}
    low_cut : float
        Defaults to 0.5
        Value in Hertz for the high-pass filter.
    high_cut : float
        Defaults to 150
        Value in Hertz for the low-pass filter.
    sf : int
        Defaults to 1000
        Sampling frequency.
    TFT_freq : int
        XXX.

    Returns
    -------
    data_ : array (numDataPoints, )
        Surrogate signal.

    """
    if surr_type == "AAFT":
        indexes = [x for x in range(len(data))]
        data_ = np.stack((data, indexes))
        data_ = AAFT_surrogates(data_)
        data_ = data_[0]
        data_ = butter_bandpass_filter(data_, low_cut, high_cut, sf, 4)
    if surr_type == "TFT":
        data_ = UnivariateSurrogatesTFT(data, 1, fc=TFT_freq)
    if surr_type == "phase":
        len_data = len(data)
        data_ = phaseScrambleTS(data)
        data_ = butter_bandpass_filter(data_[0:len_data], low_cut, high_cut, sf, 4)
    if surr_type == "shuffle":
        data_ = data.copy()
        np.random.shuffle(data_)
        data_ = butter_bandpass_filter(data_, low_cut, high_cut, sf, 4)
    if surr_type == "white":
        beta = 0
    if surr_type == "pink":
        beta = 1
    if surr_type == "brown":
        beta = 2
    if surr_type == "blue":
        beta = -1
    if surr_type == "white" or surr_type == "pink" or surr_type == "brown" or surr_type == "blue":
        try:
            import colorednoise as cn
        except ImportError:
            raise ImportError(
                "The 'colorednoise' package is required for this function. Install it with:\n\n"
                "    pip install colorednoise\n"
            )
        data_ = cn.powerlaw_psd_gaussian(beta, len(data))
        data_ = butter_bandpass_filter(data_, low_cut, high_cut, sf, 4)
    return data_


def surrogate_signal_matrices(data, surr_type="pink", low_cut=0.5, high_cut=150, sf=1000):
    """Short summary.

    Parameters
    ----------
    data : array (n, [m], numDataPoints)
        Original signals in the form of a 2D or 3D array
    surr_type : str
        Defaults to 'pink'.
        Type of surrogate.
        {'brown', 'pink', 'white', 'blue',
         'shuffle', 'phase', 'AAFT', 'TFT'}
    low_cut : float
        Defaults to 0.5
        Value in Hertz for the high-pass filter.
    high_cut : float
        Defaults to 150
        Value in Hertz for the low-pass filter.
    sf : int
        Defaults to 1000
        Sampling frequency.

    Returns
    -------
    type
        Description of returned object.

    """
    data_ = data.copy()
    if np.ndim(data) == 2:
        for i in range(len(data)):
            data_[i] = surrogate_signal(data[i], surr_type=surr_type, low_cut=low_cut, high_cut=high_cut, sf=sf)
    if np.ndim(data) == 3:
        for i in range(len(data)):
            for j in range(len(data[i])):
                data_[i][j] = surrogate_signal(
                    data[i][j],
                    surr_type=surr_type,
                    low_cut=low_cut,
                    high_cut=high_cut,
                    sf=sf,
                )
    return data_


def compute_peaks_matrices(
    data,
    peaks_function="EEMD",
    precision=0.5,
    sf=1000,
    max_freq=80,
    n_peaks=5,
    save=False,
    suffix="test",
):
    """Extract spectral peaks for a 2D or 3D array of time series.

    Parameters
    ----------
    data : array (n, [m], numDataPoints)
        Input signals.
    peaks_function : str
        Defaults to EEMD.
        Function to extract spectral peaks.
        {'fixed', 'adapt', 'FOOOF',
         'EMD', 'EEMD', 'EEMD_FOOOF',
         'HH1D_max', 'SSA', 'cepstrum',
         'Bicoherence', 'PAC', 'EIMC',
         'harmonic_recurrence'}
    precision : float
        Defaults to 0.5
        Precision in Hertz of the frequency peak.
    sf : int
        Defaults to 1000.
        Sampling frequency.
    max_freq : float
        Defaults to 80.
        Maximum frequency for a spectral peak.
    n_peaks : int
        Defaults to 5.
        Maximum number of peaks to extract.
    save : Boolean
        Defaults to False.
        When set to True, peaks and amplitudes are saved as np arrays.
    suffix : str
        Defaults to 'test'
        Set a suffix for the end of the file to save.

    Returns
    -------
    peaks_total : array
        1D or 2D array of spectral peaks.
    amps_total : array
        1D or 2D array of amplitudes of spectral peaks.

    """
    if np.ndim(data) == 2:
        peaks_tot = []
        amps_tot = []
        for i in range(len(data)):
            peaks, amps = compute_peaks_ts(
                data[i],
                peaks_function=peaks_function,
                FREQ_BANDS=None,
                precision=precision,
                sf=sf,
                max_freq=max_freq,
                n_peaks=n_peaks,
            )
            peaks_tot.append(peaks)
            amps_tot.append(amps)
        peaks_total = np.array(peaks_tot)
        amps_total = np.array(amps_tot)
        if save is True:
            np.save("peaks_{}_{}".format(peaks_function, suffix), peaks_total)
            np.save("amps_{}_{}".format(peaks_function, suffix), amps_total)
    if np.ndim(data) == 3:
        peaks_tot = []
        amps_tot = []
        for i in range(len(data)):
            peaks_temp = []
            amps_temp = []
            for j in range(len(data[i])):
                peaks, amps = compute_peaks_ts(
                    data[i][j],
                    peaks_function=peaks_function,
                    FREQ_BANDS=None,
                    precision=precision,
                    sf=sf,
                    max_freq=max_freq,
                )
                peaks_temp.append(peaks)
                amps_temp.append(amps)
            peaks_tot.append(peaks_temp)
            amps_tot.append(amps_temp)
        peaks_total = np.array(peaks_tot)
        amps_total = np.array(amps_tot)
        if save is True:
            np.save("peaks_{}_{}".format(peaks_function, suffix), peaks_total)
            np.save("amps_{}_{}".format(peaks_function, suffix), amps_total)
    return peaks_total, amps_total


def compute_peaks_surrogates(
    data,
    conditions,
    peaks_function="EMD",
    precision=0.25,
    sf=1000,
    max_freq=80,
    low_cut=0.5,
    high_cut=150,
    save=False,
):
    peaks_tot = []
    amps_tot = []
    for e, c in enumerate(conditions):
        print("Condition (", e + 1, "of", len(conditions), "):", c)
        if c == "og_data":
            peaks, amps = compute_peaks_matrices(data, peaks_function, precision, sf, max_freq)
        if c != "og_data":
            data = surrogate_signal_matrices(data, surr_type=c, low_cut=low_cut, high_cut=high_cut, sf=sf)
            peaks, amps = compute_peaks_matrices(
                data,
                peaks_function,
                precision,
                sf,
                max_freq,
                save=False,
                suffix="default",
            )
        peaks_tot.append(peaks)
        amps_tot.append(amps)
    peaks_total = np.array(peaks_tot)
    amps_total = np.array(amps_tot)

    return peaks_total, amps_total


def peaks_to_metrics_matrices(peaks, n_harm=10):
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
        metrics_dict["cons"] = cons
        metrics_dict["euler"] = euler
        metrics_dict["tenney"] = tenney
        metrics_dict["harm_fit"] = harm_fit
    return np.array([cons, euler, tenney, harm_fit]), metrics_dict


def graph_dist(
    dist,
    metric="diss",
    ref=None,
    dimensions=[0, 1],
    labs=["eeg", "phase", "AAFT", "pink", "white"],
    savefolder="\\",
    subject="0",
    tag="0",
    adapt="False",
    peaks_function="EEMD",
    colors=None,
    display=False,
    save=True,
    title=None,
):
    # print(len(dist), len(dist[0]), len(dist[1]), len(dist[2]), len(dist[3]))
    # if ref == None:
    #    ref = dist[0]
    if metric == "dissonance":
        m = "Dissonance (From Sethares (2005))"
    if metric == "euler":
        m = "Consonance (Euler <Gradus Suavitatis>)"
    if metric == "diss_euler":
        m = "Consonance (Euler <Gradus Suavitatis>) of dissonant minima"
    if metric == "diss_n_steps":
        m = "Number of dissonant minima"
    if metric == "diss_harm_sim":
        m = "Harmonic similarity of scale derived from dissonance curve"
    if metric == "tenney":
        m = "Tenney Height"
    if metric == "harmsim":
        m = "Harmonic similarity of peaks"
    if metric == "diss_harm_sim":
        m = "Harmonic similarity of diss scale"
    if metric == "harm_fit":
        m = "Harmonic fitness between peaks"
    if metric == "cons":
        m = "Averaged consonance of all paired peaks ratios"
    if metric == "n_harmonic_peaks":
        m = "Number of harmonic peaks"
    if metric == "matrix_harm_sim":
        m = "Harmonic similarity of peaks ratios intervals"
    if metric == "matrix_cons":
        m = "Consonance of peaks ratios intervals"
    if metric == "metric_3":
        m = "Pytuning consonance metric"
    if metric == "sum_distinct_intervals":
        m = "Sum of distinct intervals"
    if metric == "sum_p_q_for_all_intervals":
        m = "Sum of num and denom for all intervals"
    if metric == "sum_q_for_all_intervals":
        m = "Sum of denom for all intervals"
    if metric == "n_spectro_chords":
        m = "Number of spectral chords"
    if metric == "n_IF_chords":
        m = "Number of instantaneous frequencies chords"
    if metric == "peaks":
        m = "Average peaks frequency"

    plt.rcParams["axes.facecolor"] = "black"
    if display is True:
        fig = plt.figure(figsize=(11, 7))
    else:
        fig = plt.figure(figsize=(14, 10))

    if colors is None:
        colors = ["cyan", "deeppink", "white", "yellow", "blue", "orange", "red"]

    xcoords = []

    for dim in dimensions:
        labs = labs
        if dim == 0:
            dimension = "trials"
        if dim == 1:
            dimension = "channels"

        for d, color, enum in zip(dist, colors, range(len(dist))):
            # d = d[~np.isnan(d)]

            d = [x for x in d if str(x) != "nan"]
            ref = [x for x in ref if str(x) != "nan"]
            if dimensions == [0]:
                sbn.distplot(d, color=color)
                # DEPRECATED, WILL USE THIS: sbn.displot(biotuning.peaks, color ='red',kind="kde", fill=True)
                secure_random = secrets.SystemRandom()
                if len(d) < len(ref):
                    ref = secure_random.sample(list(ref), len(d))
                if len(ref) < len(d):
                    d = secure_random.sample(list(d), len(ref))
                # print('d', d)
                # print('ref', ref)
                t, p = stats.ttest_rel(d, ref)
                # print('p value', p)
            if dimensions != [0]:
                sbn.distplot(np.nanmean(d, dim), color=color)
                secure_random = secrets.SystemRandom()
                if len(d) < len(ref):
                    ref = secure_random.sample(list(ref), len(d))
                if len(ref) < len(d):
                    d = secure_random.sample(list(d), len(ref))
                t, p = stats.ttest_rel(np.nanmean(ref, dim), np.nanmean(d, dim))

            if p < 0.05:
                labs[enum] = labs[enum] + " *"
                # xcoords.append(np.average(d))

                # for xc in xcoords:
                #    plt.axvline(x=xc, c='white')

        if len(labs) == 2:
            fig.legend(
                labels=[labs[0], labs[1]],
                loc=[0.69, 0.65],
                fontsize=15,
                facecolor="white",
            )
        if len(labs) == 3:
            fig.legend(
                labels=[labs[0], labs[1], labs[2]],
                loc=[0.69, 0.65],
                fontsize=15,
                facecolor="white",
            )
        if len(labs) == 4:
            fig.legend(
                labels=[labs[0], labs[1], labs[2], labs[3]],
                loc=[0.69, 0.65],
                fontsize=15,
                facecolor="white",
            )
        if len(labs) == 5:
            fig.legend(
                labels=[labs[0], labs[1], labs[2], labs[3], labs[4]],
                loc=[0.69, 0.63],
                fontsize=15,
                facecolor="white",
            )
        if len(labs) == 6:
            fig.legend(
                labels=[labs[0], labs[1], labs[2], labs[3], labs[4], labs[5]],
                loc=[0.69, 0.62],
                fontsize=15,
                facecolor="white",
            )
        plt.xlabel(m, fontsize="16")
        plt.ylabel("Proportion of samples", fontsize="16")
        # plt.xlim([0.25, 0.7])
        plt.grid(color="white", linestyle="-.", linewidth=0.7)
        # if 'pink' in labs or 'brown' in labs or 'white' in labs or 'blue' in labs:
        if title == None:
            plt.suptitle(
                "Comparing " + m + " \nfor EEG, surrogate data, and noise signals across " + dimension,
                fontsize="20",
            )
        else:
            plt.suptitle(title, fontsize="20")
        if save == True:
            fig.savefig(
                savefolder + "{}_distribution_s{}-bloc{}_{}_{}.png".format(metric, subject, tag, dimension, peaks_function),
                dpi=300,
            )
            plt.clf()
        if display == True:
            plt.rcParams["figure.figsize"] = (5, 3)
            plt.show()


from biotuner.biotuner_utils import NTET_ratios
from biotuner.scale_construction import dissmeasure


def diss_curve_multi(freqs, amps, denom=10, max_ratio=2, bound=0.1, n_tet_grid=None, data_type="Electrodes", labels=None):
    from numpy import array, linspace, empty, concatenate
    from scipy.signal import argrelextrema
    from fractions import Fraction

    plt.figure(figsize=(18, 8))
    diss_minima_tot = []
    for fr, am, lab in zip(freqs, amps, labels):
        freqs = np.array([x * 128 for x in fr])
        am = np.interp(am, (np.array(am).min(), np.array(am).max()), (0.3, 0.7))
        r_low = 1
        alpharange = max_ratio
        method = "min"

        n = 1000
        diss = empty(n)
        a = concatenate((am, am))
        for i, alpha in enumerate(linspace(r_low, alpharange, n)):
            f = concatenate((freqs, alpha * freqs))
            d = dissmeasure(f, a, method)
            diss[i] = d

        plt.plot(linspace(r_low, alpharange, len(diss)), diss, label=lab)
        plt.xscale("log")
        plt.xlim(r_low, alpharange)

        plt.xlabel("frequency ratio")
        plt.ylabel("sensory dissonance")

        diss_minima = argrelextrema(diss, np.less)
        diss_minima_tot.append(list(diss_minima[0]))
        # print(diss_minima)

    diss_tot = [item for sublist in diss_minima_tot for item in sublist]
    diss_tot.sort()
    new_minima = []

    for i in range(len(diss_tot) - 1):
        if (diss_tot[i + 1] - diss_tot[i]) < bound:
            new_minima.append((diss_tot[i] + diss_tot[i + 1]) / 2)
    # print(new_minima)
    intervals = []
    for d in range(len(new_minima)):
        # print(new_minima[d])
        frac = Fraction(new_minima[d] / (n / (max_ratio - 1)) + 1).limit_denominator(denom)
        print(frac)
        frac = (frac.numerator, frac.denominator)
        intervals.append(frac)

    # intervals = [(123, 100), (147, 100), (159, 100), (9, 5), (2, 1)]
    intervals.append((2, 1))
    # print(intervals)
    for n, d in intervals:
        plt.axvline(n / d, color="silver")
    plt.axvline(1.001, linewidth=1, color="black")
    ax = plt.gca()
    ax.set_facecolor("white")
    ax.axhline(linewidth=1, color="black")

    plt.xscale("linear")
    plt.minorticks_off()
    plt.xticks([n / d for n, d in intervals], ["{}/{}".format(n, d) for n, d in intervals], fontsize=14)
    plt.yticks(fontsize=14)
    if n_tet_grid is not None:
        n_tet = NTET_ratios(n_tet_grid, max_ratio=max_ratio)
        for n in n_tet:
            plt.axvline(n, color="red", linestyle="--")
    plt.tight_layout()
    leg = plt.legend(fontsize=12, title="Electrode")
    leg.set_title(data_type, prop={"size": 18})
    # plt.legend()
    plt.savefig("diss_curve_multi_electrodes_1.png", dpi=300)
    # plt.show()
    return diss_minima_tot


def graph_conditions(
    data,
    sf,
    conditions=["eeg", "pink"],
    metric_to_graph="harmsim",
    peaks_function="adapt",
    precision=0.5,
    savefolder=None,
    tag="-",
    low_cut=0.5,
    high_cut=150,
    colors=None,
    display=False,
    save=True,
    n_harmonic_peaks=5,
    min_harms=2,
    FREQ_BANDS=None,
    min_notes=3,
    cons_limit=0.1,
    max_freq=60,
    title=None,
):
    def remove_zeros(list_):
        for x in range(len(list_)):
            if list_[x] == 0.0:
                list_[x] = 0.1
        return list_

    peaks_avg_tot = []
    metric_tot = []
    for cond in range(len(data)):
        print(conditions[cond])
        data_ = data[cond]
        peaks_avg = []
        metric = []
        for t in range(len(data_)):
            try:
                _data_ = data_[t][:]
                biotuning = biotuner(
                    sf,
                    peaks_function=peaks_function,
                    precision=precision,
                    n_harm=10,
                    ratios_n_harms=10,
                    ratios_inc_fit=False,
                    ratios_inc=False,
                )  # Initialize biotuner object
                # print(_data_.shape)
                biotuning.peaks_extraction(
                    _data_,
                    ratios_extension=False,
                    max_freq=max_freq,
                    min_harms=min_harms,
                    FREQ_BANDS=FREQ_BANDS,
                )
                if peaks_function == "harmonic_peaks":
                    biotuning.peaks = [x for _, x in sorted(zip(biotuning.amps, biotuning.peaks))][::-1][0:n_harmonic_peaks]
                    biotuning.amps = sorted(biotuning.amps)[::-1][0:n_harmonic_peaks]
                # print(biotuning.peaks)
                biotuning.compute_peaks_metrics()
                peaks_avg.append(np.average(biotuning.peaks))
                try:
                    metric.append(biotuning.peaks_metrics[metric_to_graph])
                except:
                    if metric_to_graph == "peaks":
                        metric.append(np.average(biotuning.peaks))

                    if (
                        metric_to_graph == "sum_p_q"
                        or metric_to_graph == "sum_distinct_intervals"
                        or metric_to_graph == "metric_3" "sum_p_q_for_all_intervals"
                        or metric_to_graph == "sum_q_for_all_intervals"
                        or metric_to_graph == "matrix_harm_sim"
                        or metric_to_graph == "matrix_cons"
                    ):
                        scale_metrics, _ = scale_to_metrics(biotuning.peaks_ratios)
                        # print(scale_metrics[metric_to_graph])
                        metric.append(float(scale_metrics[metric_to_graph]))
                    if (
                        metric_to_graph == "dissonance"
                        or metric_to_graph == "diss_n_steps"
                        or metric_to_graph == "diss_harm_sim"
                    ):
                        biotuning.compute_diss_curve(
                            plot=False,
                            input_type="peaks",
                            denom=100,
                            max_ratio=2,
                            n_tet_grid=12,
                        )
                        metric.append(biotuning.scale_metrics[metric_to_graph])
                    if metric_to_graph == "n_spectro_chords":
                        biotuning.compute_spectromorph(
                            comp_chords=True,
                            method="SpectralCentroid",
                            min_notes=min_notes,
                            cons_limit=cons_limit,
                            cons_chord_method="cons",
                            window=500,
                            overlap=1,
                            graph=False,
                        )
                        metric.append(len(biotuning.spectro_chords))
                    if metric_to_graph == "n_IF_chords":
                        IF = np.round(biotuning.IF, 2)
                        chords, positions = timepoint_consonance(IF, method="cons", limit=cons_limit, min_notes=min_notes)
                        metric.append(len(chords))
            except:
                pass
        metric_tot.append(metric)
        peaks_avg_tot.append(np.average(peaks_avg))
        # print(run)
    print(peaks_function, " peaks freqs ", peaks_avg_tot)
    graph_dist(
        metric_tot,
        metric=metric_to_graph,
        ref=metric_tot[0],
        dimensions=[0],
        labs=conditions,
        savefolder=savefolder,
        subject="2",
        tag=tag,
        adapt="False",
        peaks_function=peaks_function,
        colors=colors,
        display=display,
        save=save,
        title=title,
    )


def compare_metrics(
    data,
    sf,
    peaks_function="adapt",
    precision=0.5,
    savefolder=None,
    tag="-",
    low_cut=0.5,
    high_cut=150,
    display=False,
    save=True,
    n_harmonic_peaks=5,
    min_harms=2,
    FREQ_BANDS=None,
    min_notes=3,
    cons_limit=0.1,
    max_freq=60,
    chords_multiple_metrics=True,
    add_cons=0.3,
    add_notes=2,
    chords_metrics=True,
    window=500,
):
    df = pd.DataFrame()

    peaks = []
    sum_p_q = []
    sum_distinct_intervals = []
    matrix_harm_sim = []
    matrix_cons = []
    sum_q_for_all_intervals = []
    dissonance = []
    diss_n_steps = []
    harmsim = []
    cons = []
    tenney = []
    harm_fit = []
    if peaks_function == "EEMD" or peaks_function == "EMD" or peaks_function == "HH1D_max":
        if chords_metrics == True:
            n_spec_chords = []
            if chords_multiple_metrics == True:
                n_spec_chords_cons = []
                n_spec_chords_cons_notes = []
    for t in range(len(data)):

        try:
            _data_ = data[t]
            # print(_data_)
            biotuning = compute_biotuner(
                sf,
                peaks_function=peaks_function,
                precision=precision,
                n_harm=10,
                ratios_n_harms=10,
                ratios_inc_fit=False,
                ratios_inc=False,
            )  # Initialize biotuner object
            biotuning.peaks_extraction(
                _data_,
                ratios_extension=False,
                max_freq=max_freq,
                min_harms=min_harms,
                FREQ_BANDS=FREQ_BANDS,
            )
            if peaks_function == "harmonic_peaks":
                biotuning.peaks = [x for _, x in sorted(zip(biotuning.amps, biotuning.peaks))][::-1][0:n_harmonic_peaks]
                biotuning.amps = sorted(biotuning.amps)[::-1][0:n_harmonic_peaks]
            # print('b', biotuning.peaks, biotuning.amps)
            biotuning.compute_peaks_metrics()
            # peaks_avg.append(np.average(biotuning.peaks))
            scale_metrics, _ = scale_to_metrics(biotuning.peaks_ratios)
            biotuning.compute_diss_curve(plot=False, input_type="peaks", denom=100, max_ratio=2, n_tet_grid=12)
            if peaks_function == "EEMD" or peaks_function == "EMD":
                if chords_metrics == True:
                    biotuning.compute_spectromorph(
                        comp_chords=True,
                        method="SpectralCentroid",
                        min_notes=min_notes,
                        cons_limit=cons_limit,
                        cons_chord_method="cons",
                        window=window,
                        overlap=1,
                        graph=False,
                    )

                    n_spec_chords.append(len(biotuning.spectro_chords))
                    if chords_multiple_metrics == True:

                        biotuning.compute_spectromorph(
                            comp_chords=True,
                            method="SpectralCentroid",
                            min_notes=min_notes,
                            cons_limit=cons_limit + add_cons,
                            cons_chord_method="cons",
                            window=window,
                            overlap=1,
                            graph=False,
                        )

                        n_spec_chords_cons.append(len(biotuning.spectro_chords))
                        biotuning.compute_spectromorph(
                            comp_chords=True,
                            method="SpectralCentroid",
                            min_notes=min_notes + add_notes,
                            cons_limit=cons_limit + add_cons,
                            cons_chord_method="cons",
                            window=window,
                            overlap=1,
                            graph=False,
                        )
                        n_spec_chords_cons_notes.append(len(biotuning.spectro_chords))
            if peaks_function == "HH1D_max":
                if chords_metrics == True:
                    chords, positions = timepoint_consonance(
                        biotuning.IF,
                        method="cons",
                        limit=cons_limit,
                        min_notes=min_notes,
                    )
                    n_spec_chords.append(len(chords))
                    if chords_multiple_metrics == True:
                        chords2, positions2 = timepoint_consonance(
                            biotuning.IF,
                            method="cons",
                            limit=cons_limit + add_cons,
                            min_notes=min_notes,
                        )
                        n_spec_chords_cons.append(len(chords2))
                        chords3, positions3 = timepoint_consonance(
                            biotuning.IF,
                            method="cons",
                            limit=cons_limit + add_cons,
                            min_notes=min_notes + add_notes,
                        )
                        n_spec_chords_cons_notes.append(len(chords3))
            # print(peaks)
            # print(biotuning.scale_metrics)
            peaks.append(np.average(biotuning.peaks))
            sum_p_q.append(float(scale_metrics["sum_p_q"]))
            sum_distinct_intervals.append(float(scale_metrics["sum_distinct_intervals"]))
            matrix_harm_sim.append(float(scale_metrics["matrix_harm_sim"]))
            matrix_cons.append(float(scale_metrics["matrix_cons"]))
            sum_q_for_all_intervals.append(float(scale_metrics["sum_q_for_all_intervals"]))
            dissonance.append(biotuning.scale_metrics["dissonance"])
            diss_n_steps.append(biotuning.scale_metrics["diss_n_steps"])
            cons.append(biotuning.peaks_metrics["cons"])
            harm_fit.append(biotuning.peaks_metrics["harm_fit"])
            harmsim.append(biotuning.peaks_metrics["harmsim"])
            tenney.append(biotuning.peaks_metrics["tenney"])
        except:
            pass
    # print(peaks)
    df["peaks"] = peaks
    df["cons"] = cons
    df["harmsim"] = harmsim
    df["harm_fit"] = harm_fit
    df["tenney"] = tenney
    df["diss_n_steps"] = diss_n_steps
    df["dissonance"] = dissonance
    df["matrix_cons"] = matrix_cons
    df["matrix_harm_sim"] = matrix_harm_sim
    df["sum_q_for_all_intervals"] = sum_q_for_all_intervals
    df["sum_distinct_intervals"] = sum_distinct_intervals
    df["sum_p_q"] = sum_p_q
    if peaks_function == "EEMD" or peaks_function == "EMD" or peaks_function == "HH1D_max":
        if chords_metrics == True:
            df["spectro_chords"] = n_spec_chords
            if chords_multiple_metrics == True:
                df["spectro_chords_cons"] = n_spec_chords_cons
                df["spectro_chords_cons+"] = n_spec_chords_cons_notes

    return df


def compare_corr_metrics_peaks(
    data,
    sf,
    peaks_functions=["fixed", "EMD"],
    precision=0.5,
    FREQ_BANDS=None,
    chords_multiple_metrics=False,
    min_notes=3,
    cons_limit=0.1,
    chords_metrics=True,
    window=500,
    save=False,
    fname="harmonicity_metrics_",
):
    df_corr_total = pd.DataFrame()
    df_metrics_total = []
    for i in range(len(peaks_functions)):
        # print(peaks_functions[i])
        df_metrics = compare_metrics(
            data,
            sf,
            peaks_function=peaks_functions[i],
            precision=precision,
            FREQ_BANDS=FREQ_BANDS,
            chords_multiple_metrics=chords_multiple_metrics,
            chords_metrics=chords_metrics,
            window=window,
        )
        # print('df_metrics', df_metrics)
        df_metrics = df_metrics.fillna(0)
        df_p = calculate_pvalues(df_metrics)
        # print(df_p)
        df_p = df_p.rename({"peaks": peaks_functions[i]}, axis=0)
        df_peaks_p_ = abs(df_p.iloc[0])

        df_corr = df_metrics.corr()
        df_corr = df_corr.rename({"peaks": peaks_functions[i]}, axis=0)
        df_peaks_corr_ = abs(df_corr.iloc[0])

        if i == 0:
            df_peaks_corr = df_peaks_corr_
            df_peaks_p = df_peaks_p_
        else:
            df_peaks_corr = pd.concat([df_peaks_corr, df_peaks_corr_], axis=1, ignore_index=False)
            df_peaks_p = pd.concat([df_peaks_p, df_peaks_p_], axis=1, ignore_index=False)
        df_metrics_total.append(df_metrics)
    df_metrics_total = pd.concat(df_metrics_total, keys=peaks_functions)
    df_metrics_total = df_metrics_total.reset_index(level=1, drop=True)
    if save == True:
        df_metrics_save = df_metrics_total.reset_index()
        df_metrics_save.rename(columns={"level_0": "method"}, inplace=True)
        # df_metrics_save = df_metrics_save.drop('level_1', 1)
        df_metrics_save.to_csv(fname + ".csv", index=False)
        df_peaks_corr_save = df_peaks_corr.reset_index()
        df_peaks_corr_save.to_csv(fname + "_peaks_corr.csv", index=False)
        df_peaks_p_save = df_peaks_p.reset_index()
        df_peaks_p_save.to_csv(fname + "_peaks_p.csv", index=False)
    return df_peaks_corr, df_peaks_p, df_metrics_total


def stats_all_metrics_all_functions(data1, data2, peaks_functions, data_types, stat_method="ANCOVA", plot=False):

    metrics = list(data1.columns)
    ttest_all = pd.DataFrame()
    stat_values_all = pd.DataFrame()
    avg_all = pd.DataFrame()
    for function in peaks_functions:
        metrics_val = []
        stat_val = []
        avg = []
        for metric in metrics:
            if stat_method == "t-test":
                a = data1.loc[function][metric]
                b = data2.loc[function][metric]
                a = [x for x in a if str(x) != "nan"]
                b = [x for x in b if str(x) != "nan"]
                if len(a) > len(b):
                    a = a[0 : len(b)]
                if len(b) > len(a):
                    b = b[0 : len(a)]
                t, p = stats.ttest_rel(a, b)
                metrics_val.append(p)
                stat_val.append(t)
            if stat_method == "ANCOVA":
                if metric == "peaks":
                    metrics_val.append(0)
                    stat_val.append(0)
                else:
                    # print('data1', data1)
                    # print('data2', data2)
                    anc = ancova_biotuner2d(data1, data2, function, metric, data_types, plot=False)
                    metrics_val.append(anc["p-unc"][0])
                    stat_val.append(anc["F"][0])
            avg_1 = np.nanmean(data1.loc[function, metric])
            avg_2 = np.nanmean(data2.loc[function, metric])
            if avg_1 >= avg_2:
                avg.append(1)
            if avg_2 > avg_1:
                avg.append(2)
            if len(avg) < len(metrics_val):  # deal when there are 'NaN' values for methods that doesn't compute spectral chords
                avg += [0] * (len(metrics_val) - len(avg))
        ttest_all[function] = metrics_val
        stat_values_all[function] = stat_val
        avg_all[function] = avg
    ttest_all = ttest_all.set_axis(metrics, axis="index")
    stat_values_all = stat_values_all.set_axis(metrics, axis="index")
    avg_all = avg_all.set_axis(metrics, axis="index")
    return ttest_all, stat_values_all, avg_all


def plot_ttest_all_metrics(
    ttest_all,
    peaks_function,
    labels=["EEG", "ECG"],
    peaks_corr1=None,
    peaks_corr2=None,
    color="darkred",
    save=False,
    avg_all=None,
    savename=None,
    fname="_",
):

    fig, ax = plt.subplots(figsize=(15, 10))
    plt.setp(ax.get_xticklabels(), rotation=25, horizontalalignment="right")
    plt.title(
        "Results of ANCOVA comparing " + labels[0] + " and " + labels[1] + " signals using " + peaks_function,
        fontsize=22,
    )
    plt.xlabel("Harmonicity metrics", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel("p value", fontsize=16)
    plt.axhline(y=0.05, color="r", linestyle="--")
    plt.plot(ttest_all[peaks_function], color=color, label=peaks_function, linewidth=3)
    # if peaks_corr1 != None and peaks_corr2 != None:
    x_position1 = list(range(len(peaks_corr1[peaks_function])))
    # x_position1 = [x+1 for x in x_position1]
    x_position2 = list(range(len(peaks_corr2[peaks_function])))
    # x_position2 = [x+1 for x in x_position2]
    plt.scatter(
        x_position1,
        peaks_corr1[peaks_function],
        color="darkred",
        label=labels[0] + " corr with peaks",
        s=150,
    )
    plt.scatter(
        x_position2,
        peaks_corr2[peaks_function],
        color="darkorange",
        label=labels[1] + " corr with peaks",
        s=150,
    )
    k = 0
    j = 0
    for i in range(len(ttest_all[peaks_function])):
        if ttest_all[peaks_function][i] < 0.05:

            if avg_all[peaks_function][i] == 1:
                if k == 0:
                    plt.scatter(
                        i,
                        0,
                        color="darkred",
                        marker="^",
                        s=500,
                        label=labels[0] + " higher value",
                    )
                    k += 1

                plt.scatter(i, 0, color="darkred", marker="^", s=500)
            if avg_all[peaks_function][i] == 2:
                if j == 0:
                    plt.scatter(
                        i,
                        0,
                        color="darkorange",
                        marker="^",
                        s=500,
                        label=labels[1] + " higher value",
                    )
                    j += 1
                plt.scatter(i, 0, color="darkorange", marker="^", s=500)

    plt.grid()
    plt.legend(loc="upper right", fontsize=14)
    plt.show
    if save == True:
        if savename == None:
            plt.savefig(
                "ANCOVA_" + labels[0] + "_" + labels[1] + "_" + peaks_function + "_" + fname + ".jpg",
                dpi=300,
                facecolor="w",
            )
        else:
            plt.savefig(savename + ".jpg", dpi=300, facecolor="w")


def ttest_all_metrics(data1, data2, function_name):
    list_metrics = list(data1.columns)
    ttest_all = pd.DataFrame()
    metrics_ = []
    for metric in list_metrics:
        a = data1[metric]
        b = data2[metric]
        a = [x for x in a if str(x) != "nan"]
        b = [x for x in b if str(x) != "nan"]
        if len(a) > len(b):
            a = a[0 : len(b)]
        if len(b) > len(a):
            b = b[0 : len(a)]
        t, p = stats.ttest_rel(a, b)
        metrics_.append(p)
    ttest_all[function_name] = metrics_
    ttest_all = ttest_all.set_axis(list_metrics, axis="index")
    return ttest_all


def combine_dims(a, start=0, count=2):
    s = a.shape
    return np.reshape(a, s[:start] + (-1,) + s[start + count :])


def slice_data(data, sf, window=1):
    if np.ndim(data) == 1:
        window_len = int(window * sf)
        n_windows = int(len(data) / window_len)

        data_sliced = []
        for i in range(n_windows):
            start_idx = i * window_len
            stop_idx = start_idx + window_len
            data_sliced.append(data[start_idx:stop_idx])
    if np.ndim(data) == 2:
        window_len = int(window * sf)
        n_windows = int(len(data[0]) / window_len)
        data_sliced = []
        for i in range(len(data)):
            data_sliced_temp = []
            for j in range(n_windows):
                start_idx = j * window_len
                stop_idx = start_idx + window_len
                data_sliced_temp.append(data[i][start_idx:stop_idx])
            data_sliced.append(data_sliced_temp)
        # data_sliced = combine_dims(np.array(data_sliced), 0, 2)
    return np.array(data_sliced)


def resample_2d(data, sf, target_sf):
    sf_ratio = sf / target_sf
    data_crop = []
    for i in range(len(data)):
        resampled = signal.resample(data[i], int(len(data[i]) / sf_ratio))
        data_crop.append(resampled)
    return np.array(data_crop)


def equate_dimensions(data1_, data2_):
    if len(data1_) > len(data2_):
        data1 = data1_[0 : len(data2_)]
        data2 = data2_
    else:
        data2 = data2_[0 : len(data1_)]
        data1 = data1_
    if len(data1[0]) > len(data2[0]):
        data1_new = []
        for i in range(len(data1)):
            data1_new.append(data1[i][0 : len(data2[0])])
        data1 = np.array(data1_new)
    else:
        data2_new = []
        for i in range(len(data2)):
            data2_new.append(data2[i][0 : len(data1[0])])
        data2 = np.array(data2_new)
    return data1, data2


def slice_data(data, sf, window=1):
    if np.ndim(data) == 1:
        window_len = int(window * sf)
        n_windows = int(len(data) / window_len)

        data_sliced = []
        for i in range(n_windows):
            start_idx = i * window_len
            stop_idx = start_idx + window_len
            data_sliced.append(data[start_idx:stop_idx])
    if np.ndim(data) == 2:
        window_len = int(window * sf)
        n_windows = int(len(data[0]) / window_len)
        data_sliced = []
        for i in range(len(data)):
            data_sliced_temp = []
            for j in range(n_windows):
                start_idx = j * window_len
                stop_idx = start_idx + window_len
                data_sliced_temp.append(data[i][start_idx:stop_idx])
            data_sliced.append(data_sliced_temp)
        data_sliced = combine_dims(np.array(data_sliced), 0, 2)
    return np.array(data_sliced)


def ancova_biotuner2d(df1, df2, method, metric, data_types, plot=False):
    df_tot = pd.concat([df1.loc[method], df2.loc[method]], keys=data_types).reset_index()
    df_tot.rename(columns={"level_0": "data_type"}, inplace=True)
    df_tot = df_tot.fillna(0)
    if plot == True:
        sbn.distplot(df1.loc[method, metric])
        sbn.distplot(df2.loc[method, metric])
    # print('df_tot', df_tot, 'metric', metric)
    try:
        from pingouin import ancova
    except ImportError:
        raise ImportError(
            "The 'pingouin' package is required for this function. Install it with:\n\n" "    pip install pingouin\n"
        )
    return ancova(data=df_tot, dv=metric, covar="peaks", between="data_type")


def generate_ecg_dataset(
    duration,
    sf,
    n_trials,
    noise_amp=0.2,
    noise_frequency=[5, 10, 50, 70],
    artifacts_amp=0.2,
    mode="ecg",
):
    ecg_sim = []
    n_trials = n_trials
    n = 0
    while n < n_trials:
        ecg = nk.ecg_simulate(duration=duration, sampling_rate=sf)
        if mode == "noise":
            ecg = np.zeros(len(ecg))
        ecg = nk.signal_distort(
            ecg,
            sampling_rate=1000,
            noise_amplitude=noise_amp,
            noise_frequency=noise_frequency,
            artifacts_amplitude=artifacts_amp,
            artifacts_frequency=50,
        )
        ecg_sim.append(ecg)
        n += 1
    return np.array(ecg_sim)


def bt2d_plot_all_metrics(
    data1,
    data2,
    sf,
    peaks_functions=["fixed", "EEMD"],
    precision=0.5,
    chords_metrics=False,
    chords_multiple_metrics=False,
    data_types=["data1", "data2"],
    FREQ_BANDS=None,
    save=True,
    fname="",
    stat_method="ANCOVA",
):
    peaks_corr1, peaks_p1, metrics1 = compare_corr_metrics_peaks(
        data1,
        sf,
        peaks_functions=peaks_functions,
        precision=precision,
        FREQ_BANDS=FREQ_BANDS,
        chords_metrics=chords_metrics,
        save=save,
        chords_multiple_metrics=chords_multiple_metrics,
        fname=data_types[0] + "_metrics_" + str(precision) + "_" + fname,
    )
    peaks_corr2, peaks_p2, metrics2 = compare_corr_metrics_peaks(
        data2,
        sf,
        peaks_functions=peaks_functions,
        precision=precision,
        FREQ_BANDS=FREQ_BANDS,
        chords_metrics=chords_metrics,
        save=save,
        chords_multiple_metrics=chords_multiple_metrics,
        fname=data_types[1] + "_metrics_" + str(precision) + "_" + fname,
    )
    plt.rcParams["axes.facecolor"] = "white"
    stats_all, vals, avg_all = stats_all_metrics_all_functions(
        metrics1,
        metrics2,
        peaks_functions,
        data_types=data_types,
        stat_method=stat_method,
    )
    colors = [
        "darkcyan",
        "darkred",
        "goldenrod",
        "deeppink",
        "darkgreen",
        "black",
        "darkturquoise",
        "darkblue",
        "orange",
    ]
    for function, color in zip(peaks_functions, colors):
        plot_ttest_all_metrics(
            stats_all,
            function,
            labels=data_types,
            peaks_corr1=peaks_p1,
            peaks_corr2=peaks_p2,
            color=color,
            save=save,
            avg_all=avg_all,
            fname=fname,
        )

    return stats_all, avg_all


def graph_surrogates(
    data,
    sf,
    conditions=["eeg", "pink"],
    metric_to_graph="harmsim",
    peaks_function="adapt",
    precision=0.5,
    savefolder=None,
    tag="-",
    low_cut=0.5,
    high_cut=150,
    colors=None,
    display=False,
    save=True,
    n_harmonic_peaks=5,
    min_harms=2,
    min_notes=3,
    cons_limit=0.1,
    max_freq=60,
    FREQ_BANDS=None,
    title=None,
):
    peaks_avg_tot = []
    metric_tot = []
    for c in conditions:
        if c != "eeg":
            data_ = surrogate_signal_matrices(data, surr_type=c, low_cut=low_cut, high_cut=high_cut, sf=sf)
        else:
            data_ = butter_bandpass_filter(data, low_cut, high_cut, sf, 4)
        peaks_avg = []
        metric = []
        for t in range(len(data_)):
            _data_ = data_[t][:]
            biotuning = biotuner(
                sf,
                peaks_function=peaks_function,
                precision=precision,
                n_harm=10,
                ratios_n_harms=10,
                ratios_inc_fit=False,
                ratios_inc=False,
            )  # Initialize biotuner object
            biotuning.peaks_extraction(
                _data_,
                ratios_extension=False,
                max_freq=max_freq,
                min_harms=min_harms,
                n_peaks_FOOOF=5,
            )
            if peaks_function == "harmonic_peaks":
                biotuning.peaks = [x for _, x in sorted(zip(biotuning.amps, biotuning.peaks))][::-1][0:n_harmonic_peaks]
                biotuning.amps = sorted(biotuning.amps)[::-1][0:n_harmonic_peaks]
            # print(biotuning.peaks)
            biotuning.compute_peaks_metrics()
            peaks_avg.append(np.average(biotuning.peaks))
            try:
                metric.append(biotuning.peaks_metrics[metric_to_graph])
            except:
                if (
                    metric_to_graph == "sum_p_q"
                    or metric_to_graph == "sum_distinct_intervals"
                    or metric_to_graph == "metric_3" "sum_p_q_for_all_intervals"
                    or metric_to_graph == "sum_q_for_all_intervals"
                    or metric_to_graph == "matrix_harm_sim"
                    or metric_to_graph == "matrix_cons"
                ):
                    scale_metrics, _ = scale_to_metrics(biotuning.peaks_ratios)
                    # print(scale_metrics[metric_to_graph])
                    metric.append(float(scale_metrics[metric_to_graph]))
                if metric_to_graph == "dissonance" or metric_to_graph == "diss_n_steps" or metric_to_graph == "diss_harm_sim":
                    biotuning.compute_diss_curve(
                        plot=False,
                        input_type="peaks",
                        denom=100,
                        max_ratio=2,
                        n_tet_grid=12,
                    )
                    metric.append(biotuning.scale_metrics[metric_to_graph])
                if metric_to_graph == "n_spectro_chords":
                    biotuning.compute_spectromorph(
                        comp_chords=True,
                        method="SpectralCentroid",
                        min_notes=min_notes,
                        cons_limit=cons_limit,
                        cons_chord_method="cons",
                        window=500,
                        overlap=1,
                        graph=False,
                    )
                    metric.append(len(biotuning.spectro_chords))
                if metric_to_graph == "n_IF_chords":
                    chords, positions = timepoint_consonance(
                        biotuning.IF,
                        method="cons",
                        limit=cons_limit,
                        min_notes=min_notes,
                    )
                    metric.append(len(chords))
        metric_tot.append(metric)
        peaks_avg_tot.append(np.average(peaks_avg))
        # print(run)
    print(peaks_function, " peaks freqs ", peaks_avg_tot)
    graph_dist(
        metric_tot,
        metric=metric_to_graph,
        ref=metric_tot[0],
        dimensions=[0],
        labs=conditions,
        savefolder=savefolder,
        subject="2",
        tag=tag,
        adapt="False",
        peaks_function=peaks_function,
        colors=colors,
        display=display,
        save=save,
    )
