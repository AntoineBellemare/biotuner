import numpy as np
import emd
from PyEMD import EMD, EEMD
import matplotlib.pyplot as plt
import scipy.signal
from fooof import FOOOF
import sys
from biotuner.biotuner_utils import smooth, top_n_indexes, __get_norm
from biotuner.biotuner_utils import __product_other_freqs, __freq_ind
from biotuner.vizs import plot_polycoherence
from pactools import Comodulogram
from scipy.fftpack import next_fast_len
from scipy.signal import spectrogram
from pyts.decomposition import SingularSpectrumAnalysis

sys.setrecursionlimit(120000)


"""SIGNAL DECOMPOSITION METHODS
   Take single time series as input
   and output multiple time series as output.
   These functions can be used to compute peaks
   on each sub-signal,"""


def EMD_eeg(data, method="EMD", graph=False, extrema_detection="simple", nIMFs=5):
    """
    The Empirical Mode Decomposition is a data-adaptive multiresolution
    technique to decompose a signal into physically meaningful components,
    the Intrinsic Mode Functions (IMFs) It works like a dyadic filter bank.
    Hence, a log2 structure characterize the relation between successive IMFs.

    Parameters
    ----------
    data : array (numDataPoints,)
        Single time series.
    method : str
        Defaults to 'EMD'.
        Type of Empirical Mode Decomposition.
        Choice between:
                       'EMD'
                       'EEMD'
                       'EMD_fast'
                       'EEMD_fast'
    graph : Boolean
        Defaults to False
        Defines if graph is created
    extrema_detection : str
        Defaults to 'simple'
        Choice between: 'simple' and 'parabol'

    Returns
    -------
    eIMFs : array (nIMFS, numDataPoints)
        Returns an array of n Intrinsic Mode Functions
        by the initial number of data points.

    """
    s = np.interp(data, (data.min(), data.max()), (0, +1))
    t = np.linspace(0, 1, len(data))
    if method == "EMD":
        eIMFs = EMD(extrema_detection=extrema_detection).emd(s, t)
    if method == "EEMD":
        eIMFs = EEMD(extrema_detection=extrema_detection).eemd(s, t)
    if method == "EMD_fast":
        eIMFs = emd.sift.sift(data)
        eIMFs = np.moveaxis(eIMFs, 0, 1)
    if method == "EEMD_fast":
        eIMFs = emd.sift.ensemble_sift(data)
        eIMFs = np.moveaxis(eIMFs, 0, 1)
    if graph is True:
        t = np.linspace(0, len(data), len(data))
        nIMFs = nIMFs
        plt.figure(figsize=(12, 9))
        plt.subplot(nIMFs + 1, 1, 1)
        plt.plot(t, data, "r")
        for n in range(nIMFs):
            plt.subplot(nIMFs + 1, 1, n + 2)
            plt.plot(t, eIMFs[n], "darkcyan")
            plt.ylabel("eIMF %i" % (n + 1))
            plt.locator_params(axis="y", nbins=5)

        plt.xlabel("Time [samples]")
        plt.tight_layout()
        plt.savefig("eemd_example", dpi=120)
        plt.show()
    return eIMFs


def SSA_EEG(data, n_components=3, graph=False):
    ssa = SingularSpectrumAnalysis(window_size=20, groups=None)
    X = (data, (range(len(data))))
    X_ssa = ssa.fit_transform(X)
    if graph is True:
        plt.figure(figsize=(16, 6))
        ax1 = plt.subplot(121)
        ax1.plot(X[0], label="Original", color="darkblue")
        ax1.legend(loc="best", fontsize=14)
        plt.xlabel("Samples", size=14)
        plt.ylabel("Amplitude", size=14)
        ax2 = plt.subplot(122)
        color_list = ["darkgoldenrod", "red", "darkcyan", "indigo", "magenta"]
        for i, c in zip(range(len(X_ssa[0][0:n_components])), color_list):
            ax2.plot(X_ssa[0, i], "--", label="SSA {0}".format(i + 1), color=c)
            plt.xlabel("Samples", size=14)
            plt.ylabel("Amplitude", size=14)
        ax2.legend(loc="best", fontsize=14)
        plt.suptitle("Singular Spectrum Analysis", fontsize=20)
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        plt.show()
    return X_ssa


"""PEAKS EXTRACTION METHODS
   Take time series as input
   and output list of spectral peaks"""


def extract_welch_peaks(
    data,
    sf,
    precision=0.5,
    min_freq=1,
    max_freq=None,
    FREQ_BANDS=None,
    average="median",
    noverlap=None,
    nperseg=None,
    nfft=None,
    find_peaks_method="maxima",
    width=2,
    rel_height=0.7,
    prominence=1,
    out_type="all",
    extended_returns=True,
):
    """
    Extract frequency peaks using Welch's method
    for periodograms computation.

    Parameters
    ----------
    data : array (numDataPoints,)
        Single time series.
    sf : int
        Sampling frequency.
    precision : float
        Defaults to 0.5.
        Size of a frequency bin in Hertz.
        Defaults to 0.5Hz
    min_freq : float
        Defaults to 1
        Minimum frequency to consider when out_type='all'.
    max_freq : float
        Maximum frequency to consider when out_type='all'.
    FREQ_BANDS : List of lists
        Each sublist contains the
        minimum and maximum values for each frequency band.
    average : str
        Defaults to 'median'.
        {'mean', 'median'}
        Method to use when averaging periodograms. Defaults to median’.
    nperseg : int
        Length of each segment.
    nfft : int
        Length of the FFT used, if a zero padded FFT is desired.
        If None, the FFT length is nperseg.
        Defaults to None.
    noverlap : int
        Number of points to overlap between segments.
        If None, noverlap = nperseg // 2. Defaults to None.
    find_peaks_method : str
        Defaults to 'maxima'.
        {'maxima', 'wavelet'}
    width : int
        Defaults to 2
        Required width of peaks in samples.
    rel_height : float
        Defaults to 0.7
        Chooses the relative height at which the peak width is measured as a
        percentage of its prominence. 1.0 calculates the width of the peak at
        its lowest contour line while 0.5 evaluates at half the
        prominence height.
    prominence : type
        Defaults to 1
        Required prominence of peaks.
    out_type : str
        Defaults to 'all'.
        {'single', 'bands', 'all'}
        Defines how many peaks are outputed.
    extended_returns : Boolean
        Defaults to True.
        Defines if psd and frequency bins values are outputed along
        the peaks and amplitudes.

    Returns
    -------
    peak : List(float)
        Frequency value.
    amp : List(float)
        Amplitude value.
    freqs : Array
        Frequency bins
    psd : Array
        Power spectrum density of each frequency bin

    """

    if max_freq is None:
        max_freq = sf / 2
    if nperseg is None:
        mult = 1 / precision
        nperseg = sf * mult
        nfft = nperseg
    freqs, psd = scipy.signal.welch(
                 data,
                 sf,
                 nfft=nfft,
                 nperseg=nperseg,
                 average=average,
                 noverlap=noverlap
                 )
    psd = 10.0 * np.log10(psd)
    if out_type == "all":
        if find_peaks_method == "maxima":
            indexes, _ = scipy.signal.find_peaks(
                psd,
                height=None,
                threshold=None,
                distance=2,
                prominence=prominence,
                width=width,
                wlen=None,
                rel_height=rel_height,
                plateau_size=None,
            )
        if find_peaks_method == "wavelet":
            indexes = scipy.signal.find_peaks_cwt(
                psd, widths=[1, max_freq], min_length=0.5
            )
        peaks = []
        amps = []
        for i in indexes:
            peaks.append(freqs[i])
            amps.append(psd[i])
    if out_type == "single":
        index_max = np.argmax(np.array(psd))
        peaks = freqs[index_max]
        peaks = np.around(peaks, 5)
        amps = psd[index_max]
    if out_type == "bands":
        peaks = []
        amps = []
        idx_max = []
        for minf, maxf in FREQ_BANDS:
            bin_size = (sf / 2) / len(freqs)
            min_index = int(minf / bin_size)
            max_index = int(maxf / bin_size)
            index_max = np.argmax(np.array(psd[min_index:max_index]))
            idx_max.append(index_max)
            peaks.append(freqs[min_index + index_max])
            amps.append(psd[min_index + index_max])
        print("Index_max: all zeros indicate 1/f trend", idx_max)
    if out_type != "single":
        peaks = np.around(np.array(peaks), 5)
        peaks = list(peaks)
        peaks = [p for p in peaks if p <= max_freq]
        peaks = [p for p in peaks if p >= min_freq]
    if extended_returns is True:
        return peaks, amps, freqs, psd
    return peaks, amps


def compute_FOOOF(
    data,
    sf,
    precision=0.1,
    max_freq=80,
    noverlap=None,
    nperseg=None,
    nfft=None,
    n_peaks=5,
    extended_returns=False,
    graph=False,
):
    """FOOOF conceives of a model of the power spectrum as a combination of
       two distinct functional processes:
       - An aperiodic component, reflecting 1/f like characteristics
       - A variable number of periodic components (putative oscillations),
         as peaks rising above the aperiodic component.

    Parameters
    ----------
    data : array (numDataPoints,)
        Single time series.
    sf : int
        Sampling frequency.
    precision : float
        Defaults to 0.1
        Size of a frequency bin in Hertz.
        Defaults to 0.5Hz
    max_freq : float
        Defaults to 80
        Maximum frequency to consider as a peak.
    noverlap : int
        Number of points to overlap between segments.
        If None, noverlap = nperseg // 2. Defaults to None.
    nperseg : int
        Length of each segment.
    nfft : int
        Length of the FFT used, if a zero padded FFT is desired.
        If None, the FFT length is nperseg.
        Defaults to None.
    n_peaks : int
        Defaults to 5
        Maximum number of peaks. If FOOOF finds higher number of peaks,
        the peaks with highest amplitude will be retained.
    extended_returns : Boolean
        Defaults to False.
        Defines if psd and frequency bins values are outputed along
        the peaks and amplitudes.
    graph : Boolean
        Defaults to False.
        Defines if a graph is generated.

    Returns
    -------
    peaks : List(float)
        Frequency values.
    amps : List(float)
        Amplitude values.
    freqs : Array
        Frequency bins
    psd : Array
        Power spectrum density of each frequency bin

    """

    if nperseg is None:
        mult = 1 / precision
        nfft = sf * mult
        nperseg = nfft
        noverlap = nperseg // 10
    freqs, psd = scipy.signal.welch(
                 data,
                 sf,
                 nfft=nfft,
                 nperseg=nperseg,
                 noverlap=noverlap
                 )
    fm = FOOOF(
         peak_width_limits=[precision * 2, 3],
         max_n_peaks=50,
         min_peak_height=0.2
         )
    freq_range = [(sf / len(data)) * 2, max_freq]
    fm.fit(freqs, psd, freq_range)
    if graph is True:
        fm.report(freqs, psd, freq_range)
    peaks_temp = []
    amps_temp = []
    for p in range(len(fm.peak_params_)):
        try:
            peaks_temp.append(fm.peak_params_[p][0])
            amps_temp.append(fm.peak_params_[p][1])
        except:
            print("no peaks were found")
            pass
    peaks_temp.append(
        [x for _, x in sorted(zip(amps_temp,
                                  peaks_temp))][::-1][0:n_peaks]
                                  )
    amps = sorted(amps_temp)[::-1][0:n_peaks]
    peaks = [np.round(p, 2) for p in peaks_temp]
    if extended_returns is True:
        return peaks, amps, freqs, psd
    return peaks, amps


"""HARMONIC-PEAKS EXTRACTION METHODS
   Take time series as input
   and output list of harmonic peaks"""


def HilbertHuang1D(
    data,
    sf,
    graph=False,
    nIMFs=5,
    min_freq=1,
    max_freq=80,
    precision=0.1,
    bin_spread="log",
):
    """The Hilbert-Huang transform provides a description of how the energy
       or power within a signal is distributed across frequency.
       The distributions are based on the instantaneous frequency and
       amplitude of a signal.

    Parameters
    ----------
    data : array (numDataPoints,)
        Single time series.
    sf : int
        Sampling frequency.
    graph : Boolean
        Defaults to False.
        Defines if a graph is generated.
    nIMFs : int
        Defaults to 5
        Number of intrinsic mode functions (IMFs) to keep when
        Empirical Mode Decomposition (EMD) is computed.
    min_freq : float
        Defaults to 1
        Minimum frequency to consider.
    max_freq : float
        Defaults to 80
        Maximum frequency to consider.
    precision : float
        Defaults to 0.1
        Value in Hertz corresponding to the minimal step between two
        frequency bins.
    bin_spread : str
        Defaults to 'log'.
        {'linear','log'}

    Returns
    -------
    IF : array (numDataPoints,nIMFs)
        instantaneous frequencies associated with each IMF.
    peaks : List(float)
        Frequency values.
    amps : List(float)
        Amplitude values.
    spec : array (nIMFs, nbins)
        Power associated with all bins for each IMF
    bins : array (nIMFs, nbins)
        Frequency bins for each IMF

    """
    IMFs = EMD_eeg(data, method="EMD")
    IMFs = np.moveaxis(IMFs, 0, 1)
    IP, IF, IA = emd.spectra.frequency_transform(IMFs[:, 1 : nIMFs + 1], sf, "nht")
    low = min_freq
    high = max_freq
    range_hh = int(high - low)
    steps = int(range_hh / precision)
    bin_size = range_hh / steps
    edges, bins = emd.spectra.define_hist_bins(
        low - (bin_size / 2), high - (bin_size / 2), steps, bin_spread
    )
    # Compute the 1d Hilbert-Huang transform (power over carrier frequency)
    freqs = []
    spec = []
    for IMF in range(len(IF[0])):

        freqs_, spec_ = emd.spectra.hilberthuang(IF[:, IMF], IA[:, IMF], edges)
        freqs.append(freqs_)
        spec.append(spec_)
    peaks_temp = []
    amps_temp = []
    for e, i in enumerate(spec):
        max_power = np.argmax(i)
        peaks_temp.append(bins[max_power])
        amps_temp.append(spec[e][max_power])
    peaks_temp = np.flip(peaks_temp)
    amps_temp = np.flip(amps_temp)
    peaks = [np.round(p, 2) for p in peaks_temp]
    amps = [np.round(a, 2) for a in amps_temp]
    bins_ = []
    for i in range(len(spec)):
        bins_.append(bins)
    if graph is True:
        plt.figure(figsize=(8, 4))
        for plot in range(len(spec)):
            plt.plot(bins_[plot], spec[plot])
        plt.xlim(min_freq, max_freq)
        plt.xscale("log")
        plt.xlabel("Frequency (Hz)")
        plt.title("IA-weighted\nHilbert-Huang Transform")
        plt.legend(["IMF-1", "IMF-2", "IMF-3", "IMF-4", "IMF-5", "IMF-6", "IMF-7"])
    return IF, peaks, amps, np.array(spec), np.array(bins_)


def cepstrum(signal, sf, plot_cepstrum=False, min_freq=1.5, max_freq=80):
    """The cepstrum is the result of computing the
       inverse Fourier transform (IFT) of the logarithm of
       the estimated signal spectrum. The method is a tool for
       investigating periodic structures in frequency spectra.

    Parameters
    ----------
    signal : array (numDataPoints,)
        Single time series.
    sf : int
        Sampling frequency.
    plot_cepstrum : Boolean
        Defaults to False.
        Determines wether a plot is generated.
    min_freq : float
        Defaults to 1.5
        Minimum frequency to consider.
    max_freq : float
        Defaults to 80
        Maximum frequency to consider.

    Returns
    -------
    cepstrum : array (nbins,)
        Power of the cepstrum for each quefrency.
    quefrency_vector : array(nbins,)
        Values of each quefrency bins.

    """
    windowed_signal = signal
    dt = 1 / sf
    freq_vector = np.fft.rfftfreq(len(windowed_signal), d=dt)
    X = np.fft.rfft(windowed_signal)
    log_X = np.log(np.abs(X))

    cepstrum = np.fft.rfft(log_X)
    cepstrum = smooth(cepstrum, 10)
    df = freq_vector[1] - freq_vector[0]
    quefrency_vector = np.fft.rfftfreq(log_X.size, df)
    quefrency_vector = smooth(quefrency_vector, 10)

    if plot_cepstrum is True:
        fig, ax = plt.subplots()
        ax.plot(freq_vector, log_X)
        ax.set_xlabel("frequency (Hz)")
        ax.set_title("Fourier spectrum")
        ax.set_xlim(0, max_freq)
        fig, ax = plt.subplots()
        ax.plot(quefrency_vector, np.abs(cepstrum))
        ax.set_xlabel("quefrency (s)")
        ax.set_title("cepstrum")
        ax.set_xlim(1 / max_freq, 1 / min_freq)
        ax.set_ylim(0, 200)
    return cepstrum, quefrency_vector


def cepstral_peaks(cepstrum, quefrency_vector, max_time, min_time):
    """This function extract cepstral peaks based on the cepstrum function.

    Parameters
    ----------
    cepstrum : array
        Values of cepstrum power across all quefrency bins.
    quefrency_vector : array
        Values of all the quefrency bins.
    max_time : float
        Maximum value of the quefrency to keep in ms.
    min_time : float
        Minimum value of the quefrency to keep in ms.

    Returns
    -------
    peaks : List(float)
        Quefrency values.
    amps : List(float)
        Amplitude values.

    """

    indexes = scipy.signal.find_peaks(
        cepstrum,
        height=None,
        threshold=None,
        distance=None,
        prominence=None,
        width=3,
        wlen=None,
        rel_height=0.5,
        plateau_size=None,
    )
    peaks = []
    amps = []
    for i in indexes[0]:
        if quefrency_vector[i] < max_time and quefrency_vector[i] > min_time:
            amps.append(np.abs(cepstrum)[i])
            peaks.append(quefrency_vector[i])
    peaks = np.around(np.array(peaks), 3)
    peaks = list(peaks)
    peaks = [1 / p for p in peaks]
    return peaks, amps


def pac_frequencies(
    ts,
    sf,
    method="duprelatour",
    n_values=10,
    drive_precision=0.05,
    max_drive_freq=6,
    min_drive_freq=3,
    sig_precision=1,
    max_sig_freq=50,
    min_sig_freq=8,
    low_fq_width=0.5,
    high_fq_width=1,
    plot=False,
):
    """A function to compute the comodulogram for phase-amplitude coupling
       and extract the pairs of peaks with maximum coupling value.

    Parameters
    ----------
    ts : array (numDataPoints,)
        Single time series.
    sf : int
        Sampling frequency.
    method : str
        {'ozkurt', 'canolty', 'tort', 'penny', 'vanwijk', 'duprelatour',
         'colgin','sigl', 'bispectrum'}
    n_values : int
        Defaults to 10
        Number of pairs of drive and modulated frequencies to keep.
    drive_precision : float
        Defaults to 0.05
        Value (hertz) of one frequency bin of the phase signal.
    max_drive_freq : float
        Defaults to 6
        Minimum value (hertz) of the phase signal.
    min_drive_freq : float
        Defaults to 3
        Maximum value (hertz) of the phase signal.
    sig_precision : float
        Defaults to 1
        Value (hertz) of one frequency bin of the amplitude signal.
    max_sig_freq : float
        Defaults to 50
        Maximum value (hertz) of the amplitude signal.
    min_sig_freq : float
        Defaults to 8
        Minimum value (hertz) of the amplitude signal.
    low_fq_width : float
        Defaults to 0.5
        Bandwidth of the band-pass filter (phase signal).
    high_fq_width : float
        Defaults to 1
        Bandwidth of the band-pass filter (amplitude signal).
    plot : Boolean
        Defaults to False.
        Determines if a plot of the comodulogram is created.

    Returns
    -------
    pac_freqs : List of lists
        Each sublist correspond to pairs of frequencies for the
        phase and amplitude signals with maximal coupling value.
    pac_coupling : List
        Coupling values associated with each pairs of phase and amplitude
        frequencies.

    """

    drive_steps = int(((max_drive_freq - min_drive_freq) / drive_precision) + 1)
    low_fq_range = np.linspace(min_drive_freq, max_drive_freq, drive_steps)
    sig_steps = int(((max_sig_freq - min_sig_freq) / sig_precision) + 1)
    high_fq_range = np.linspace(min_sig_freq, max_sig_freq, sig_steps)

    estimator = Comodulogram(
        fs=sf,
        low_fq_range=low_fq_range,
        low_fq_width=low_fq_width,
        high_fq_width=high_fq_width,
        high_fq_range=high_fq_range,
        method=method,
        progress_bar=False,
    )
    estimator.fit(ts)
    indexes = top_n_indexes(estimator.comod_, n_values)[::-1]
    pac_freqs = []
    pac_coupling = []
    for i in indexes:
        pac_freqs.append([low_fq_range[i[0]], high_fq_range[i[1]]])
        pac_coupling.append(estimator.comod_[i[0]][i[1]])
    if plot is True:
        estimator.plot()
    return pac_freqs, pac_coupling


def _polycoherence_2d(
    data, fs, *ofreqs, norm=2, flim1=None, flim2=None, synthetic=(), **kwargs
):
    """
    Polycoherence between freqs and their sum as a function of f1 and f2
    """
    norm1, norm2 = __get_norm(norm)
    freq, t, spec = spectrogram(data, fs=fs, mode="complex", **kwargs)
    spec = np.require(spec, "complex64")
    spec = np.transpose(spec, [1, 0])  # transpose (f, t) -> (t, f)
    if flim1 is None:
        flim1 = (0, (np.max(freq) - np.sum(ofreqs)) / 2)
    if flim2 is None:
        flim2 = (0, (np.max(freq) - np.sum(ofreqs)) / 2)
    ind1 = np.arange(*np.searchsorted(freq, flim1))
    ind2 = np.arange(*np.searchsorted(freq, flim2))
    ind3 = __freq_ind(freq, ofreqs)
    otemp = __product_other_freqs(spec, ind3, synthetic, t)[:, None, None]
    sumind = ind1[:, None] + ind2[None, :] + sum(ind3)
    temp = spec[:, ind1, None] * spec[:, None, ind2] * otemp
    if norm is not None:
        temp2 = np.mean(np.abs(temp) ** norm1, axis=0)
    temp *= np.conjugate(spec[:, sumind])
    coh = np.mean(temp, axis=0)
    del temp
    if norm is not None:
        coh = np.abs(coh, out=coh)
        coh **= 2
        temp2 *= np.mean(np.abs(spec[:, sumind]) ** norm2, axis=0)
        coh /= temp2
        coh **= 0.5
    return freq[ind1], freq[ind2], coh


def polycoherence(data, *args, dim=2, **kwargs):
    """
    FROM: https://github.com/trichter/polycoherence
    Polycoherence between frequencies and their sum frequency
    Polycoherence as a function of two frequencies.
    |<prod(spec(fi)) * conj(spec(sum(fi)))>| ** n0 /
        <|prod(spec(fi))|> ** n1 * <|spec(sum(fi))|> ** n2
    i ... 1 - N: N=2 bicoherence, N>2 polycoherence
    < > ... averaging
    | | ... absolute value
    data: 1d data
    fs: sampling rate
    ofreqs: further positional arguments are fixed frequencies
    dim:
        2 - 2D polycoherence as a function of f1 and f2, ofreqs are additional
            fixed frequencies (default)
        1 - 1D polycoherence as a function of f1, at least one fixed frequency
            (ofreq) is expected
        'sum' - 1D polycoherence with fixed frequency sum. The first argument
            after fs is the frequency sum. Other fixed frequencies possible.
        0 - polycoherence for fixed frequencies
    norm:
        2 - return polycoherence, n0 = n1 = n2 = 2 (default)
        0 - return polyspectrum, <prod(spec(fi)) * conj(spec(sum(fi)))>
        tuple (n1, n2): general case with n0=2
    synthetic:
        used for synthetic signal for some frequencies,
        list of 3-item tuples (freq, amplitude, phase), freq must coincide
        with the first fixed frequencies (ofreq, except for dim='sum')
    flim1, flim2: for 2D case, frequency limits can be set
    **kwargs: are passed to scipy.signal.spectrogram. Important are the
        parameters nperseg, noverlap, nfft.
    """
    N = len(data)
    kwargs.setdefault("nperseg", N // 20)
    kwargs.setdefault("nfft", next_fast_len(N // 10))
    f = _polycoherence_2d
    return f(data, *args, **kwargs)


def polyspectrum_frequencies(
    data,
    sf,
    precision,
    n_values=10,
    nperseg=None,
    noverlap=None,
    method="bicoherence",
    flim1=(2, 50),
    flim2=(2, 50),
    graph=False,
):
    """Short summary.

    Parameters
    ----------
    data : type
        Description of parameter `data`.
    sf : type
        Description of parameter `sf`.
    precision : type
        Description of parameter `precision`.
    n_values : type
        Description of parameter `n_values`.
    nperseg : type
        Description of parameter `nperseg`.
    noverlap : type
        Description of parameter `noverlap`.
    method : type
        Description of parameter `method`.
    flim1 : type
        Description of parameter `flim1`.
    flim2 : type
        Description of parameter `flim2`.
    graph : type
        Description of parameter `graph`.

    Returns
    -------
    type
        Description of returned object.

    """
    if method == "bispectrum":
        norm = 0
    if method == "bicoherence":
        norm = 2
    if nperseg is None:
        nperseg = sf
    if noverlap is None:
        noverlap = sf // 10
    nfft = int(sf * (1 / precision))
    kw = dict(nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    freq1, freq2, bispec = polycoherence(
        data, 1000, norm=norm, **kw, flim1=flim1, flim2=flim2, dim=2
    )
    for i in range(len(bispec)):
        for j in range(len(bispec[i])):
            if str(np.real(bispec[i][j])) == "inf":
                bispec[i][j] = 0
    indexes = top_n_indexes(np.real(bispec), n_values)[::-1]
    poly_freqs = []
    poly_amps = []
    for i in indexes:
        poly_freqs.append([freq1[i[0]], freq2[i[1]]])
        poly_amps.append([abs(bispec[i[0], i[1]])])
    if graph is True:
        plot_polycoherence(freq1, freq2, bispec)
    return poly_freqs, poly_amps


"""HARMONIC PEAKS SELECTION
   Take list of all peaks as input
   and output selected harmonic peaks"""


def harmonic_recurrence(
    peaks, amps, min_freq=1, max_freq=30, min_harms=2, harm_limit=128
):
    """The harmonic series of each peak is compared to all other peaks
       to identify those that have the highest recurrence in the spectrum.
       Hence, selected peaks are the ones that have a maximum number of
       harmonics that match other peaks in the spectrum.

    Parameters
    ----------
    peaks : List
        List of spectral peaks.
    amps : type
        Description of parameter `amps`.
    min_freq : float
        Default to 1.
        Minimum peak of frequency to consider.
    max_freq : float
        Default to 30.
        Maximum peak of frequency to consider.
    min_harms : int
        Default to 2.
        Minimum number of harmonic recurrence to keep a peak.
    harm_limit : int
        Default to 128.
        Higher harmonic to consider.

    Returns
    -------
    type
        Description of returned object.

    """
    n_total = []
    harm_ = []
    harm_peaks = []
    max_n = []
    max_peaks = []
    max_amps = []
    harmonics = []
    harmonic_peaks = []
    harm_peaks_fit = []
    for p, a in zip(peaks, amps):
        n = 0
        harm_temp = []
        harm_peaks_temp = []
        if p < max_freq and p > min_freq:

            for p2 in peaks:
                if p2 == p:
                    ratio = 0.1  # arbitrary value to set ratio  to non integer
                if p2 > p:
                    ratio = p2 / p
                    harm = ratio
                if p2 < p:
                    ratio = p / p2
                    harm = -ratio
                if ratio.is_integer():
                    if harm <= harm_limit:
                        n += 1
                        harm_temp.append(harm)
                        if p not in harm_peaks_temp:
                            harm_peaks_temp.append(p)
                        if p2 not in harm_peaks_temp:
                            harm_peaks_temp.append(p2)
        n_total.append(n)
        harm_.append(harm_temp)
        harm_peaks.append(harm_peaks_temp)
        if n >= min_harms:
            max_n.append(n)
            max_peaks.append(p)
            max_amps.append(a)
            harmonics.append(harm_temp)
            harmonic_peaks.append(harm_peaks)
            harm_peaks_fit.append([p, harm_temp, harm_peaks_temp])
    for i in range(len(harm_peaks_fit)):
        harm_peaks_fit[i][2] = sorted(harm_peaks_fit[i][2])
    max_n = np.array(max_n)
    max_peaks = np.array(max_peaks)
    max_amps = np.array(max_amps)
    harmonics = np.array(harmonics)
    harmonic_peaks = np.array(harmonic_peaks)
    return max_n, max_peaks, max_amps, harmonics, harmonic_peaks, harm_peaks_fit


def compute_IMs(f1, f2, n):
    """InterModulation components: sum or subtraction of any non-zero integer
       multiple of the input frequencies.

    Parameters
    ----------
    f1 : float
        Frequency 1.
    f2 : float
        Frequency 2.
    n : int
        Order of the intermodulation component.

    Returns
    -------
    IMs : List
        List of all intermodulation components.
    order : List
        Order associated with IMs.

    """
    IMs = []
    orders = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            IMs.append(f1 * j + f2 * i)
            orders.append(j)
            IMs.append(np.abs(f1 * j - f2 * i))
            orders.append(j)
    IMs = [x for _, x in sorted(zip(orders, IMs))]
    orders = sorted(orders)
    return IMs, orders


def endogenous_intermodulations(peaks, amps, order=3, min_IMs=2, max_freq=100):
    """
    This function computes the intermodulation components (IMCs) for each pairs
    of peaks and compare the IMCs with peaks values. If a pair of peaks has
    a number of IMCs equals to or higher than parameter min_IMs, these peaks
    and the associated IMCs are stored in IMs_all dictionary.
    """
    EIMs = []
    IMCs_all = {"IMs": [], "peaks": [], "n_IMs": [], "orders": [], "amps": []}
    for p, a in zip(peaks, amps):
        IMs_temp = []
        orders_temp = []
        for p2, a2 in zip(peaks, amps):
            if p2 > p:
                if p < max_freq and p2 < max_freq:
                    IMs, orders_ = compute_IMs(p, p2, order)
                    IMs_temp.append(IMs)
                    orders_temp.append(orders_)
                    IMs_all_ = list(set(IMs) & set(peaks))
                    if len(IMs_all_) > min_IMs - 1:
                        IMCs_all["IMs"].append(IMs_all_)
                        IMCs_all["peaks"].append([p, p2])
                        IMCs_all["amps"].append([a, a2])
                        IMCs_all["n_IMs"].append(len(IMs_all_))
                        IMCs_all["orders"].append(orders_temp)
        IMs_temp = [item for sublist in IMs_temp for item in sublist]
        orders_temp = [item for sublist in orders_temp for item in sublist]
        EIMs_temp = list(set(IMs_temp) & set(peaks))
        EIMs.append(EIMs_temp)
        n_IM_peaks = len(IMCs_all["IMs"])
    return EIMs, IMCs_all, n_IM_peaks


def compute_sidebands(carrier, modulator, order=2):
    """Short summary.

    Parameters
    ----------
    carrier : type
        Description of parameter `carrier`.
    modulator : type
        Description of parameter `modulator`.
    order : type
        Description of parameter `order`.

    Returns
    -------
    type
        Description of returned object.

    """
    sidebands = []
    i = 1
    while i <= order:
        if carrier - modulator * i > 0:
            sidebands.append(np.round(carrier - modulator * i, 3))
            sidebands.append(np.round(carrier + modulator * i, 3))
        i += 1
    return np.sort(sidebands)


# https://github.com/voicesauce/opensauce-python/blob/master/opensauce/shrp.py

"""
EXPERIMENTAL / work-in-progress
# from https://homepage.univie.ac.at/christian.herbst/python/dsp_util_8py_source.html#l03455
def detectSubharmonics(signal, fs, timeStep, fMin, fMax, voicingThreshold = 0.3,
    tolerancePercent = 5, maxOctaveCost = 0.25, maxOctaveJumpCost = 0.3,
    minOctaveCost = 0, minOctaveJumpCost = 0):


    #signal /= numpy.nanmax(numpy.absolute(signal)) * 1.5
    n = len(signal)
    duration = float(n) / float(fs)
    arrMetaData = [
     {
         'octaveCost': maxOctaveCost,
         'octaveJumpCost': maxOctaveJumpCost,
     },
     {
         'octaveCost': minOctaveCost,
         'octaveJumpCost': minOctaveJumpCost,
     },
    ]
    arrResults = []
    for idx, metaData in enumerate(arrMetaData):
        octaveCost = metaData['octaveCost']
        octaveJumpCost = metaData['octaveJumpCost']
        # print fs, timeStep, fMin, fMax, voicingThreshold, octaveCost, \
        #       octaveJumpCost
        arrT, arrFo = praatUtil.calculateF0OfSignal(signal, fs,
                                                    readProgress=timeStep,
                                                    acFreqMin = fMin, fMax = fMax,
                                                    voicingThreshold = voicingThreshold,
                                                    octaveCost = octaveCost, octaveJumpCost = octaveJumpCost)
        arrResults.append([arrT, arrFo])
    numDataPoints = duration / float(timeStep)
    arrT = numpy.arange(0, duration, timeStep)
    n = len(arrT)
    arrFo = [numpy.ones(n) * numpy.nan, numpy.ones(n) * numpy.nan]
    for i in range(2):
        arrIdx = getCommonTimeOffsets(arrT, arrResults[i][0], timeStep)
        for j, tmp in enumerate(arrIdx):
            idx1, idx2 = tmp
            arrFo[i][idx1] = arrResults[i][1][idx2]

    # detect subharmonic portions
    arrPeriod = numpy.ones(n) * numpy.nan
    arrFoFinal = numpy.ones(n) * numpy.nan
    subharmonicsFound = False
    for i, t in enumerate(arrT):
        f1 = arrFo[0][i]
        f2 = arrFo[1][i]
        if not numpy.isnan(f1) and not numpy.isnan(f2):
            if f1 > 0 and f2 > 0:

                # check if data points match
                diff = abs(f1 - f2)
                mean = (f1 + f2) / 2.0
                diffPercent = diff * 100.0 / float(mean)
                if diffPercent <= tolerancePercent:
                    arrFoFinal[i] = mean
                    arrPeriod[i] = 1.0

                else:

# check if its subharmonic
                    factor = f1 / float(f2)
                    for base in range(2, 9):
                        boundary1 = base + base * (1 - (100 + tolerancePercent) / 100.0)
                        boundary2 = base + base * (1 - (100 - tolerancePercent) / 100.0)

                    if factor >= boundary1 and factor <= boundary2:
                        #print "\t\t", i, base, factor, boundary1, boundary2
                        # it's a subharmonic data point
                        arrFoFinal[i] = f1
                        #if f2 > f1:
                        arrPeriod[i] = float(base)
                        subharmonicsFound = True
                        break

    if 1 == 2 and subharmonicsFound:
     ms = 8
     plt.plot(arrT, arrFo[0], '+', markersize = ms * 1.8, color='green', alpha=0.8)
     plt.plot(arrT, arrFo[1], 'x', markersize = ms* 1.8, color='yellow', alpha=0.4)
     plt.plot(arrT, arrFoFinal, 'D', markersize = ms, color='violet', alpha=0.8)
     for i, t in enumerate(arrT):
         p = arrPeriod[i]
         if p > 1:
             plt.plot(t, arrFoFinal[i] / float(p), 'o', markersize = ms, color='red', alpha=1)
     plt.grid()
     plt.show()
     exit(1)

    return arrT, arrFoFinal, arrPeriod, arrFo
"""
