from fooof import FOOOF
import scipy.signal
from pytuning import create_euler_fokker_scale
import matplotlib.pyplot as plt
from itertools import combinations

from biotuner.peaks_extraction import (
    HilbertHuang1D,
    harmonic_recurrence,
    cepstrum,
    cepstral_peaks,
    EMD_eeg,
)
import numpy as np
from biotuner.peaks_extraction import (
    extract_welch_peaks,
    compute_FOOOF,
    polyspectrum_frequencies,
    pac_frequencies,
    endogenous_intermodulations,
    polycoherence,
)

from biotuner.biotuner_utils import (
    flatten,
    pairs_most_frequent,
    compute_peak_ratios,
    alpha2bands,
    rebound,
    prime_factor,
    peaks_to_amps,
    EMD_to_spectromorph,
    ratios_harmonics,
    ratios_increments,
    make_chord,
    scale_from_pairs,
)

from biotuner.metrics import (
    euler,
    tenneyHeight,
    timepoint_consonance,
    ratios2harmsim,
    compute_subharmonic_tension,
    dyad_similarity,
    consonant_ratios,
    tuning_to_metrics,
    consonance_peaks
)

from biotuner.peaks_extension import (
    harmonic_fit,
    multi_consonance,
)
from biotuner.scale_construction import (
    diss_curve,
    harmonic_entropy,
    harmonic_tuning
)
from biotuner.vizs import graph_psd_peaks, graphEMD_welch, graph_harm_peaks, EMD_PSD_graph
import seaborn as sbn


class compute_biotuner(object):
    """
    Class used to derive peaks information, musical tunings, rhythms
    and related harmonicity metrics from time series (EEG, ECG, EMG,
    gravitational waves, noise, ...)

    Basic usage
    -----------
    >>> biotuning = compute_biotuner(sf = 1000)
    >>> biotuning.peaks_extraction(data)
    >>> biotuning.peaks_extension()
    >>> biotuning.peaks_metrics()
    
    Parameters
    ----------
    sf: int
        Sampling frequency (in Hz)
    data : array (numDataPoints,)
        Time series to analyze.
    peaks_function: str, default='EMD'
        Defines the method to use for peak extraction.

        'NON-HARMONIC PEAK EXTRACTIONS'
            'fixed' : 
                    Power Spectrum Density (PSD) estimated using Welch's method
                    on fixed frequency bands. Peaks correspond to frequency bins
                    with the highest power.
            'adapt' : 
                    PSD estimated using Welch's method on each frequency band
                    derived from the alpha peak position. Peaks correspond to
                    frequency bins with the highest power.
            'FOOOF' : 
                    PSD is estimated with Welch's method. 'FOOOF' is applied to
                    remove the aperiodic component and find physiologically
                    relevant spectral peaks.

        'SIGNAL DECOMPOSITION BASED PEAK EXTRACTION'
            'EMD': 
                Intrinsic Mode Functions (IMFs) are derived with Empirical
                Mode Decomposition (EMD) PSD is computed on each IMF using
                Welch. Peaks correspond to frequency bins with the highest power.
            'EEMD' : 
                    Intrinsic Mode Functions (IMFs) are derived with Ensemble
                    Empirical Mode Decomposition (EEMD). PSD is computed on each
                    IMF using Welch. Peaks correspond to frequency bins with the
                    highest power.
            'CEEMDAN' : Intrinsic Mode Functions (IMFs) are derived with Complex
                        Ensemble Empirical Mode Decomposition with Adaptive
                        Noise (CEEMDAN). PSD is computed on each IMF using Welch.
            'EMD_FOOOF' : 
                        Intrinsic Mode Functions (IMFs) are derived with
                        Ensemble Empirical Mode Decomposition (EEMD). PSD is
                        computed on each IMF with Welch's method. 'FOOOF' is
                        applied to remove the aperiodic component and find
                        physiologically relevant spectral peaks.
            'HH1D_max' : 
                        Maximum values of the 1D Hilbert-Huang transform on each
                        IMF using EMD.
            'HH1D_FOOOF' :
                        Hilbert-Huang transform on each IMF with Welch's method
                        'FOOOF' is applied to remove the aperiodic component and
                        find physiologically relevant spectral peaks.
            'SSA' : 
                    Singular Spectrum Analysis. The name "singular spectrum
                    analysis" relates to the spectrum of eigenvalues in a singular
                    value decomposition of a covariance matrix.

        'SECOND-ORDER STATISTICAL PEAK EXTRACTION'
            'cepstrum': 
                        Peak frequencies of the cepstrum (inverse Fourier transform
                        (IFT) of the logarithm of the estimated signal spectrum).
            'HPS (to come)' : 
                    Harmonic Product Spectrum (HPS) corresponds to the product of
                    the spectral power at each harmonic. Peaks correspond to the
                    frequency bins with the highest value of the HPS.
            'Harmonic_salience (to come)' :
                                A measure of harmonic salience computed by taking
                                the sum of the squared cosine of the angle between
                                a harmonic and its neighbors.

        'CROSS-FREQUENCY COUPLING BASED PEAK EXTRACTION'
            'Bicoherence' : 
                            Corresponds to the normalized cross-bispectrum.
                            It is a third-order moment in the frequency domain.
                            It is a measure of phase-amplitude coupling.
            'PAC' : 
                    Phase-amplitude coupling. A measure of phase-amplitude
                    coupling between low-frequency phase and high-frequency
                    amplitude. 
                
        'PEAK SELECTION BASED ON HARMONIC PROPERTIES'
            'EIMC' :
                    Endogenous InterModulation Components (EIMC)
                    correspond to spectral peaks that are sums or differences
                    of other peaks harmonics (f1+f2, f1+2f2, f1-f2, f1-2f2...).
                    PSD is estimated with Welch's method. All peaks are extracted.
            'harmonic_recurrence' :
                            PSD is estimated with Welch's method.
                            All peaks are extracted. Peaks for which
                            other peaks are their harmonics are kept.
        
        'PEAKS EXTRACTION PARAMETERS'
    
    precision: float, default=0.1
        Precision of the peaks (in Hz).
        When HH1D_max is used, bins are in log scale by default.
    compute_sub_ratios: bool, default=False
        When set to True, include ratios < 1 in peaks_ratios attribute.
    scale_cons_limit: float, default=0.1
        The minimal value of consonance needed for a peaks ratio to be 
        included in the peaks_ratios_cons attribute.

        'EXTENDED PEAKS PARAMETERS'
    
    n_harm: int, default=10
        Set the number of harmonics to compute in harmonic_fit function.
    harm_function: str, default='mult'
        - 'mult' : Computes harmonics from iterative multiplication
                   (x, 2x, 3x...) 
        - 'div': Computes harmonics from iterative division (x, x/2, x/3...).
    extension_method: str, default='consonant_harmonic_fit'
        
        - 'consonant_harmonic_fit' : computes the best-fit of harmonic peaks
           according to consonance intervals (eg. octaves, fifths).
        - 'all_harmonic_fit' : computes the best-fit of harmonic peaks without
           considering consonance intervals.
        
        'RATIOS EXTENSION PARAMETERS'
    
    ratios_n_harms: int, default=5
        The number of harmonics used to compute extended peaks ratios.
    ratios_harms: bool, default=False
        When set to True, harmonics (x*1, x*2, x*3...,x*n) of specified
        ratios will be computed.
    ratios_inc: bool, default=True
        When set to True, exponentials (x**1, x**2, x**3,...x**n) of
        specified ratios will be computed.
    ratios_inc_fit: bool, Default=False
        When set to True, a fit between exponentials
        (x**1, x**2, x**3,...x**n) of specified ratios will be computed.
    """
    pygame_lib = None
    def __init__(
        self,
        sf,
        data=None,
        peaks_function="EMD",
        precision=0.1,
        compute_sub_ratios=False,
        n_harm=10,
        harm_function="mult",
        extension_method="consonant_harmonic_fit",
        ratios_n_harms=5,
        ratios_harms=False,
        ratios_inc=True,
        ratios_inc_fit=False,
        scale_cons_limit=0.1,
    ):

        #Initializing data
        if type(data) is not None:
            self.data = data
        self.sf = sf
        #Initializing arguments for peak extraction
        self.peaks_function = peaks_function
        self.precision = precision
        self.compute_sub_ratios = compute_sub_ratios
        #Initializing arguments for peaks metrics
        self.n_harm = n_harm
        self.harm_function = harm_function
        self.extension_method = extension_method
        #Initializing dictionary for scales metrics
        self.scale_metrics = {}
        self.scale_cons_limit = scale_cons_limit
        #Initializing arguments for ratios extension
        self.ratios_n_harms = ratios_n_harms
        self.ratios_harms = ratios_harms
        self.ratios_inc = ratios_inc
        self.ratios_inc_fit = ratios_inc_fit

    def peaks_extraction(
        self,
        data,
        peaks_function=None,
        FREQ_BANDS=None,
        precision=None,
        sf=None,
        min_freq=1,
        max_freq=60,
        min_harms=2,
        compute_sub_ratios=False,
        ratios_extension=False,
        ratios_n_harms=None,
        scale_cons_limit=None,
        octave=2,
        harm_limit=128,
        n_peaks=5,
        nIMFs=5,
        graph=False,
        nperseg=None,
        nfft=None,
        noverlap=None,
        max_harm_freq=None,
        EIMC_order=3,
        min_IMs=2,
        smooth_fft=1,
        verbose=False,
        keep_first_IMF=False
    ):
        """
        The peaks_extraction method is central to the use of the Biotuner.
        It uses a time series as input and extract spectral peaks based on
        a variety of methods. See peaks_function parameter description
        in __init__ function for more details.

        Parameters
        ----------
        data: array (numDataPoints,)
            biosignal to analyse
        peaks_function: str
            refer to __init__
        FREQ_BANDS: List of lists of float
            Each list within the list of lists sets the lower and
            upper limit of a frequency band
        precision: float, default=0.1 -> __init__
            Precision of the peaks (in Hz)
            When HH1D_max is used, bins are in log scale.
        sf: int
            Sampling frequency (in Hertz).
        min_freq: float, default=1
            minimum frequency value to be considered as a peak
            Used with 'harmonic_recurrence' and 'HH1D_max' peaks functions
        max_freq: float, default=60
            maximum frequency value to be considered as a peak
            Used with 'harmonic_recurrence' and 'HH1D_max' peaks functions
        min_harms: int, default=2
            minimum number of harmonics to consider a peak frequency using
            the 'harmonic_recurrence' function.
        compute_sub_ratios: Boolean, default=False
            If set to True, will include peaks ratios (x/y) when x < y
        ratios_extension: Boolean, default=False
            When set to True, peaks_ratios harmonics and
            increments are computed.
        ratios_n_harms: int, default=5 -> __init__
            number of harmonics or increments to use in ratios_extension method
        scale_cons_limit: float, default=0.1
            minimal value of consonance to be reach for a peaks ratio
            to be included in the peaks_ratios_cons attribute.
        octave: float, default=2
            value of the octave
        harm_limit: int, default=128
            Maximum harmonic position for 'harmonic_recurrence' method.
        n_peaks: int, default=5
            Number of peaks when using 'FOOOF' and 'cepstrum',
            and 'harmonic_recurrence' functions.
            Peaks are chosen based on their amplitude.
        nIMFs: int, default=5
            Number of intrinsic mode functions to keep when using
            'EEMD' or 'EMD' peaks function.
        graph: boolean, default=False
            When set to True, a graph will accompanies the peak extraction
            method (except for 'fixed' and 'adapt').
        nperseg : int, default=None
            Length of each segment.
            If None, nperseg = nfft/smooth
        nfft : int, default=None
            Length of the FFT used, if a zero padded FFT is desired.
            If None, nfft = sf/(1/precision)
        noverlap : int, default=None
            Number of points to overlap between segments.
            If None, noverlap = nperseg // 2.
        max_harm_freq : int, default=None
            Maximum frequency value of the find peaks function
            when harmonic_recurrence or EIMC peaks extraction method is used.
        EIMC_order : int, default=3
            Maximum order of the Intermodulation Components.
        min_IMs : int, default=2
            Minimal number of Intermodulation Components to select the
            associated pair of peaks.
        smooth_fft : int, default=1
            Number used to divide nfft to derive nperseg.
        verbose : boolean, default=True
            When set to True, number of detected peaks will be displayed.
        keep_first_IMF : boolean, default=False
            When set to True, the first IMF is kept.

        Attributes
        ----------
        self.peaks: List (float)
            List of frequency peaks
        self.amps: List (float)
            List of peaks amplitude
        self.peaks_ratios: List (float)
            List of ratios between all pairs of peaks
        self.peaks_ratios_cons: List (float)
            List of consonant peaks ratios
        'If ratios_extension = True'
        self.peaks_ratios_harm: List (float)
            List of peaks ratios and their harmonics
        self.peaks_ratios_inc: List (float)
            List of peaks ratios and their increments (ratios**n)
        self.peaks_ratios_inc_bound: List (float)
            List of peaks ratios and their increments (ratios**n)
            bound within one octave
        self.peaks_ratios_inc_fit: List (float)
            List of peaks ratios and their congruent increments (ratios**n)
        """

        self.data = data
        if sf is None:
            sf = self.sf
        if precision is None:
            precision = self.precision
        if peaks_function is None:
            peaks_function = self.peaks_function
        if compute_sub_ratios is None:
            compute_sub_ratios = self.compute_sub_ratios
        if scale_cons_limit is None:
            scale_cons_limit = self.scale_cons_limit
        if ratios_n_harms is None:
            ratios_n_harms = self.ratios_n_harms
        self.octave = octave
        self.nIMFs = nIMFs
        self.compute_sub_ratios = compute_sub_ratios
        peaks, amps = self.compute_peaks_ts(
            data,
            peaks_function=peaks_function,
            FREQ_BANDS=FREQ_BANDS,
            precision=precision,
            sf=sf,
            min_freq=min_freq,
            max_freq=max_freq,
            min_harms=min_harms,
            harm_limit=harm_limit,
            n_peaks=n_peaks,
            graph=graph,
            nfft=nfft,
            nperseg=nperseg,
            noverlap=noverlap,
            max_harm_freq=max_harm_freq,
            EIMC_order=EIMC_order,
            min_IMs=min_IMs,
            smooth_fft=smooth_fft,
            keep_first_IMF=keep_first_IMF
        )
        if verbose is True:
            print('Number of peaks : {}'.format(len(peaks)))
        if len(peaks) == 0:
            print('No peak detected')
        self.peaks = peaks
        self.amps = amps
        #print("Number of peaks: ", len(peaks))
        self.peaks_ratios = compute_peak_ratios(
            self.peaks, rebound=True, octave=octave, sub=compute_sub_ratios
        )
        self.peaks_ratios_cons, b = consonant_ratios(self.peaks,
                                                     limit=scale_cons_limit)
        if ratios_extension is True:
            a, b, c = self.ratios_extension(
                self.peaks_ratios, ratios_n_harms=ratios_n_harms
            )
            if a is not None:
                self.peaks_ratios_harms = a
            if b is not None:
                self.peaks_ratios_inc = b
                bound_ = [rebound(x, low=1, high=octave, octave=octave) for x in b]
                self.peaks_ratios_inc_bound = bound_
            if c is not None:
                self.peaks_ratios_inc_fit = c

    def peaks_extension(
        self,
        peaks=None,
        n_harm=None,
        method="harmonic_fit",
        harm_function="mult",
        div_mode="add",
        cons_limit=0.1,
        ratios_extension=False,
        harm_bounds=0.1,
        scale_cons_limit=None,
    ):
        """
        This method is used to extend a set of frequencies based on the
        harmonic congruence of specific elements (extend). It can also
        restrict a set of frequencies based on the consonance level of
        specific peak frequencies.

        Parameters
        ----------
        peaks : List of float
            List of frequency peaks.
        n_harm: int, default=10
            Set the number of harmonics to compute in harmonic_fit function
        method: str, default='harmonic_fit'
        
            - 'harmonic_fit'
            - 'consonant'
            - 'multi_consonant',
            - 'consonant_harmonic_fit'
            - 'multi_consonant_harmonic_fit'
            
        harm_function: str, default='mult'
            
            - 'mult' : Computes harmonics from iterative multiplication (x, 2x, 3x, ...nx)
            - 'div' : Computes harmonics from iterative division (x, x/2, x/3, ...x/n)

        div_mode : strm default='add'
            Defines the way the harmonics are computed when harm_function is 'div'
            
            - 'div': x, x/2, x/3 ..., x/n
            - 'div_add': x, (x+x/2), (x+x/3), ... (x+x/n)
            - 'div_sub': x, (x-x/2), (x-x/3), ... (x-x/n)
            
        cons_limit : float
            Defines the minimal consonance level used in the method.
        ratios_extension : Boolean, default=False
            If is True, ratios_extensions are computed accordingly to what
            was defined in __init__.
        harm_bounds : float, default=0.1
            Maximal distance in Hertz between two frequencies to consider
            them as equivalent.
        scale_cons_limit : float, default=None
            Minimal value of consonance to be reach for a peaks ratio
            to be included in the extended_peaks_ratios_cons attribute.
            When None, scale_cons_limit = 0.1, as defined in __init__

        Returns
        -------
        self.extended_peaks: List (float)
            List of extended peaks frequencies.
        self.extended_amps: List (float)
            List of extended peaks amplitudes.
        self.extended_peaks_ratios : List (float)
            List of pairwise extended peaks ratios.
        self.extended_peaks_ratios_cons : List (float)
            List of pairwise extended peaks ratios when more consonant than
            scale_limit_cons parameter.

        Attributes
        ----------
        'If ratios_extension is True'
        self.peaks_ratios_harm: List (float)
            List of extended peaks ratios and their harmonics
        self.peaks_ratios_inc: List (float)
            List of extended peaks ratios and their increments (ratios**n)
        self.peaks_ratios_inc_fit: List (float)
            List of extended peaks ratios and
            their congruent increments (ratios**n).
        """
        if peaks is None:
            peaks = self.peaks
        if n_harm is None:
            n_harm = self.n_harm
        if method is None:
            method = self.extension_method
        if scale_cons_limit is None:
            scale_cons_limit = self.scale_cons_limit
        if method == "harmonic_fit":
            extended_peaks, harmonics, _, _ = harmonic_fit(
                peaks,
                n_harm,
                function=harm_function,
                div_mode=div_mode,
                bounds=harm_bounds,
            )
            self.extended_peaks = np.sort(list(peaks) + list(set(extended_peaks)))
        if method == "consonant":
            consonance, cons_pairs, cons_peaks, cons_metric = consonance_peaks(
                peaks, limit=cons_limit
            )
            self.extended_peaks = np.sort(np.round(cons_peaks, 3))
        if method == "multi_consonant":
            consonance, cons_pairs, cons_peaks, cons_metric = consonance_peaks(
                peaks, limit=cons_limit
            )
            extended_temp = multi_consonance(cons_pairs, n_freqs=10)
            self.extended_peaks = np.sort(np.round(extended_temp, 3))
        if method == "consonant_harmonic_fit":
            extended_peaks, harmonics, _, _ = harmonic_fit(
                peaks,
                n_harm,
                function=harm_function,
                div_mode=div_mode,
                bounds=harm_bounds,
            )
            consonance, cons_pairs, cons_peaks, cons_metric = consonance_peaks(
                peaks, limit=cons_limit
            )
            self.extended_peaks = np.sort(np.round(cons_peaks, 3))
        if method == "multi_consonant_harmonic_fit":
            extended_peaks, harmonics, _, _ = harmonic_fit(
                peaks,
                n_harm,
                function=harm_function,
                div_mode=div_mode,
                bounds=harm_bounds,
            )
            consonance, cons_pairs, cons_peaks, cons_metric = consonance_peaks(
                peaks, limit=cons_limit
            )
            extended_temp = multi_consonance(cons_pairs, n_freqs=10)
            self.extended_peaks = np.sort(np.round(extended_temp, 3))
        self.extended_peaks = [i for i in self.extended_peaks if i < self.sf / 2]
        self.extended_amps = peaks_to_amps(
            self.extended_peaks, self.freqs, self.psd, self.sf
        )
        print("Number of extended peaks : ", len(self.extended_peaks))
        if len(self.extended_peaks) > 0:
            ext_peaks_rat = compute_peak_ratios(self.extended_peaks, rebound=True)
            if ratios_extension is True:
                a, b, c = self.ratios_extension(ext_peaks_rat)
                if a is not None:
                    self.extended_peaks_ratios_harms = a
                if b is not None:
                    self.extended_peaks_ratios_inc = b
                if c is not None:
                    self.extended_peaks_ratios_inc_fit = c
            ext_peaks_rat = [np.round(r, 2) for r in ext_peaks_rat]
            self.extended_peaks_ratios = list(set(ext_peaks_rat))
            self.extended_peaks_ratios_cons, b = consonant_ratios(
                self.extended_peaks, scale_cons_limit, sub=False
            )
        return self.extended_peaks, self.extended_amps, self.extended_peaks_ratios

    def ratios_extension(self, ratios, ratio_fit_bounds=0.001, ratios_n_harms=None):
        """
        This method takes a series of ratios as input and returns the
        harmonics (ratio*2, ratio*3, ..., ratio*n) or the increments
        (ratio**2, ratio**3, ..., ratio**n).

        Parameters
        ----------
        ratios : List (float)
            List of frequency ratios.
        ratio_fit_bounds : float, default=0.001
            Minimal distance between two ratios to consider a fit
            for the harmonic_fit function.
        ratios_n_harms : int, default=None
            Number of harmonics or increments to compute.
            When None, the number of harmonics or increments
            is set to the value of the attribute ratios_n_harms.
        Returns
        -------
        ratios_harms_ : List (float)
            List of ratios harmonics.
        ratios_inc_ : List (float)
            List of ratios increments.
        ratios_inc_fit_ : List (float)
            List of ratios increments that fit.
        """
        if ratios_n_harms is None:
            ratios_n_harms = self.ratios_n_harms
        if self.ratios_harms is True:
            ratios_harms_ = ratios_harmonics(ratios, ratios_n_harms)
        else:
            ratios_harms_ = None
        if self.ratios_inc is True:
            ratios_inc_ = ratios_increments(ratios, ratios_n_harms)
        else:
            ratios_inc_ = None
        if self.ratios_inc_fit is True:
            ratios_inc_fit_, ratios_inc_fit_pos, _, _ = harmonic_fit(
                ratios, ratios_n_harms, function="exp", bounds=ratio_fit_bounds
            )
        else:
            ratios_inc_fit_ = None
        return ratios_harms_, ratios_inc_, ratios_inc_fit_

    def compute_spectromorph(
        self,
        IMFs=None,
        sf=None,
        method="SpectralCentroid",
        window=None,
        overlap=1,
        comp_chords=False,
        min_notes=3,
        cons_limit=0.2,
        cons_chord_method="cons",
        graph=False,
    ):
        """
        This method computes spectromorphological metrics on
        each Intrinsic Mode Function (IMF).

        Parameters
        ----------
        IMFs : array (nIMFs, numDataPoints)
            Intrinsic Mode Functions.
            When set to 'None', the IMFs are computed in the method.
        sf : int
            Sampling frequency.
        method : str, default='SpectralCentroid'
            Spectromorphological metric to compute.
        
             - 'SpectralCentroid',
             - 'SpectralCrestFactor',
             - 'SpectralDecrease',
             - 'SpectralFlatness',
             - 'SpectralFlux',
             - 'SpectralKurtosis',
             - 'SpectralMfccs',
             - 'SpectralPitchChroma',
             - 'SpectralRolloff',
             - 'SpectralSkewness',
             - 'SpectralSlope',
             - 'SpectralSpread',
             - 'SpectralTonalPowerRatio',
             - 'TimeAcfCoeff',
             - 'TimeMaxAcf',
             - 'TimePeakEnvelope',
             - 'TimeRms',
             - 'TimeStd',
             - 'TimeZeroCrossingRate',}
        window : int
            Window size in samples.
        overlap : int
            Value of the overlap between successive windows.
        comp_chords : Boolean, default=False
            When set to True, consonant spectral chords are computed.
        min_notes : int, default=3
            Minimum number of consonant values to store a spectral chord.
        cons_limit : float, default=0.2
            Minimal value of consonance.
        cons_chord_method : str, default='cons'
            Metrics to use for consonance computation.
            
            - :func:`cons <biotuner.metrics.compute_consonance>`
            - :func:`harmsim <biotuner.metrics.dyad_similarity>`
            - :func:`euler <biotuner.metrics.euler>`
            
        graph : Boolean, default=False
            Defines if graph is plotted.

        Attributes
        ----------
        self.spectro_EMD : array (numDataPoints, nIMFs)
            Spectromorphological metric vector for each IMF.
        self.spectro_chords : List of lists (float)
            Each sublist corresponds to a list of consonant
            spectromorphological metrics.
        """
        if IMFs is None:
            if self.peaks_function == "EEMD" or self.peaks_function == "EMD":
                IMFs = self.IMFs
            else:
                IMFs = EMD_eeg(self.data, method="EMD")[1:6]
                self.IMFs = IMFs
        if sf is None:
            sf = self.sf
        if window is None:
            window = int(sf / 2)

        spectro_EMD = EMD_to_spectromorph(
            IMFs, sf, method=method, window=window, overlap=overlap
        )
        if comp_chords is True:
            self.spectro_chords, spectro_chord_pos = timepoint_consonance(
                spectro_EMD,
                method=cons_chord_method,
                limit=cons_limit,
                min_notes=min_notes,
                graph=graph,
            )
        self.spectro_EMD = spectro_EMD
        if graph is True:
            plt.clf()
            if comp_chords is False:
                data = np.moveaxis(self.spectro_EMD, 0, 1)
                ax = sbn.lineplot(data=data[10:-10, :], dashes=False)
                ax.set(xlabel="Time Windows", ylabel=method)
                if method == "SpectralCentroid":
                    ax.set_yscale("log")
                plt.legend(
                    scatterpoints=1,
                    frameon=True,
                    labelspacing=1,
                    title="EMDs",
                    loc="best",
                    labels=["EMD1", "EMD2", "EMD3", "EMD4", "EMD5"],
                )
                plt.show()

    def compute_peaks_metrics(self, n_harm=None, harm_bounds=0.5, delta_lim=20):
        """
        This function computes consonance metrics on peaks attribute.

        Parameters
        ----------
        n_harm : int
            Set the number of harmonics to compute in harmonic_fit function
        harm_bounds : float, default=0.5
            Maximal distance in Hertz between two frequencies to consider
            them as equivalent.

        Attributes
        ----------
        self.peaks_metrics : dict
            Dictionary with keys corresponding to the different metrics.
            
            - 'cons'
            - 'euler'
            - 'tenney'
            - 'harm_fit'
            - 'harmsim'
            - 'n_harmonic_recurrence',
            - 'n_harmonic_recurrence_ratio'
            - 'harm_pos'
            - 'common_harm_pos'

        """
        if n_harm is None:
            n_harm = self.n_harm

        peaks = list(self.peaks)
        peaks_ratios = compute_peak_ratios(
            peaks, rebound=True, octave=self.octave, sub=self.compute_sub_ratios
        )
        #print('PEAKS RATIOS COMPUTED')
        metrics = {"cons": 0, "euler": 0, "tenney": 0, "harm_fit": 0, "harmsim": 0}
        #try:
        harm_fit, harm_pos, common_harm_pos, _ = harmonic_fit(
            peaks, n_harm=n_harm, bounds=harm_bounds
        )
        #print('HARMONIC FIT COMPUTED')
        metrics["harm_pos"] = harm_pos
        metrics["common_harm_pos"] = common_harm_pos
        metrics["harm_fit"] = len(harm_fit)
        #except:
        #    pass
        a, b, c, metrics["cons"] = consonance_peaks(peaks, 0.1)
        #print('CONSONANCE COMPUTED')
        peaks_euler = [int(round(num, 2) * 1000) for num in peaks]

        spf = self.peaks_function
        '''if spf == "fixed" or spf == "adapt" or spf == "EMD" or spf == "EEMD":
            try:
                metrics["euler"] = euler(*peaks_euler)
            except:
                pass'''
        metrics["tenney"] = tenneyHeight(peaks)
        metrics["harmsim"] = np.average(ratios2harmsim(peaks_ratios))
        #print('HARMSIM COMPUTED')
        _, _, subharm, _ = compute_subharmonic_tension(peaks[0:5],
                                                       n_harm,
                                                       delta_lim,
                                                       min_notes=3)
        metrics["subharm_tension"] = subharm
        #print('SUBHARM COMPUTED')
        if spf == "harmonic_recurrence":
            metrics["n_harmonic_recurrence"] = self.n_harmonic_recurrence
        self.peaks_metrics = metrics

    """Methods to compute scales from whether peaks or extended peaks"""

    def compute_diss_curve(
        self,
        input_type="peaks",
        denom=1000,
        max_ratio=2,
        euler_comp=False,
        method="min",
        plot=False,
        n_tet_grid=12,
        scale_cons_limit=None,
    ):
        """
        Compute dissonance curve based on peak frequencies.

        Parameters
        ----------
        input_type : str, default='peaks'
            Defines whether peaks or extended_peaks are used.
            
            - 'peaks'
            - 'extended_peaks'
            
        denom : int, default=1000
            Maximal value of the denominator when computing frequency ratios.
        max_ratio : float, default=2
            Value of the maximal frequency ratio to use when computing
            the dissonance curve. When set to 2, the curve spans one octave.
            When set to 4, the curve spans two octaves.
        euler_comp : Boolean, default=False
            Defines if euler consonance is computed. Can be computationally
            expensive when the number of local minima is high.
        method : str, default='min'
            Refer to dissmeasure function in scale_construction.py
            for more information.
            
            - 'min'
            - 'product'
            
        plot : Boolean, default=False
            When set to True, dissonance curve is plotted.
        n_tet_grid : int, default=12
            Defines which N-TET tuning is indicated, as a reference,
            in red in the dissonance curve plot.
        scale_cons_limit : float, default=None
            Minimal value of consonance to be reach for a peaks ratio
            to be included in the self.diss_scale_cons attribute.
            When set to None, the value of self.scale_cons_limit is used.

        Attributes
        ----------
        self.diss_scale : List (float)
            List of frequency ratios corresponding to local minima.
        self.diss_scale_cons : List (float)
            List of frequency ratios corresponding to consonant local minima.
        self.scale_metrics : dict
            Add 4 metrics related to the dissonance curve tuning:
            
            - 'diss_euler'
            - 'dissonance'
            - 'diss_harm_sim'
            - 'diss_n_steps'
            

        """
        if input_type == "peaks":
            peaks = self.peaks
            amps = self.amps
        if input_type == "extended_peaks":
            peaks = self.extended_peaks
            amps = self.extended_amps
        if scale_cons_limit is None:
            scale_cons_limit = self.scale_cons_limit

        peaks = [p * 128 for p in peaks]
        amps = np.interp(amps, (np.array(amps).min(), np.array(amps).max()), (0.2, 0.8))

        intervals, self.diss_scale, euler_diss, diss, harm_sim_diss = diss_curve(
            peaks,
            amps,
            denom=denom,
            max_ratio=max_ratio,
            euler_comp=euler_comp,
            method=method,
            plot=plot,
            n_tet_grid=n_tet_grid,
        )
        self.diss_scale_cons, b = consonant_ratios(
            self.diss_scale, scale_cons_limit, sub=False, input_type="ratios"
        )
        self.scale_metrics["diss_euler"] = euler_diss
        self.scale_metrics["dissonance"] = diss
        self.scale_metrics["diss_harm_sim"] = np.average(harm_sim_diss)
        self.scale_metrics["diss_n_steps"] = len(self.diss_scale)

    def compute_harmonic_entropy(
        self,
        input_type="peaks",
        res=0.001,
        spread=0.01,
        plot_entropy=True,
        plot_tenney=False,
        octave=2,
        rebound=True,
        sub=False,
        scale_cons_limit=None,
    ):
        """
        Computes the harmonic entropy from a series of spectral peaks.
        Harmonic entropy has been introduced by Paul Elrich
        [http://www.tonalsoft.com/enc/e/erlich/harmonic-entropy_with-
        commentary.aspx]

        Parameters
        ----------
        input_type : str, default='peaks'
            Defines whether peaks or extended_peaks are used.
            
            - 'peaks'
            - 'extended_peaks'
            
        res : float, default=0.001
            Resolution of the ratio steps.
        spread : float, default=0.01
            Spread of the normal distribution used to compute the weights.
        plot_entropy : Boolean, default=True
            When set to True, plot the harmonic entropy curve.
        plot_tenney : Boolean, default=False
            When set to True, plot the tenney heights (y-axis)
            across ratios (x-axis).
        octave : int, default=2
            Value of the octave.
        rebound : Boolean, default=True
            When set to True, peaks ratios are bounded within the octave.
        sub : Boolean, default=False
            When set to True, will include ratios below the unison (1)
        scale_cons_limit : type, default=None
            Minimal value of consonance to be reach for a peaks ratio
            to be included in the self.diss_scale_cons attribute.
            When set to None, the value of self.scale_cons_limit is used.

        Attributes
        ----------
        self.HE_scale : List (float)
            List of frequency ratios corresponding to local minima.
        self.HE_scale_cons : List (float)
            List of frequency ratios corresponding to consonant local minima.
        self.scale_metrics : dict
            Four metrics related to the dissonance curve tuning:
                
            - 'HE'
            - 'HE_n_steps'
            - 'HE_harm_sim'
            

        """
        if input_type == "peaks":
            ratios = compute_peak_ratios(self.peaks, rebound=rebound, sub=sub)
        if input_type == "extended_peaks":
            ratios = compute_peak_ratios(self.extended_peaks, rebound=rebound, sub=sub)
        if input_type == "extended_ratios_harms":
            ratios = self.extended_peaks_ratios_harms
        if input_type == "extended_ratios_inc":
            ratios = self.extended_peaks_ratios_inc
        if input_type == "extended_ratios_inc_fit":
            ratios = self.extended_peaks_ratios_inc_fit
        if scale_cons_limit is None:
            scale_cons_limit = self.scale_cons_limit

        HE_scale, HE = harmonic_entropy(
            ratios,
            res=res,
            spread=spread,
            plot_entropy=plot_entropy,
            plot_tenney=plot_tenney,
            octave=octave,
        )
        self.HE_scale = HE_scale[0]
        self.HE_scale_cons, b = consonant_ratios(
            self.HE_scale, scale_cons_limit, sub=False, input_type="ratios"
        )
        self.scale_metrics["HE"] = HE
        self.scale_metrics["HE_n_steps"] = len(self.HE_scale)
        HE_harm_sim = np.average(ratios2harmsim(list(self.HE_scale)))
        self.scale_metrics["HE_harm_sim"] = HE_harm_sim

    def euler_fokker_scale(self, method="peaks", octave=2):
        """
        Create a scale in the Euler-Fokker Genera. which is a
        musical scale in just intonation whose pitches can be
        expressed as products of some of the members of some multiset
        of generating primer numbers.

        Parameters
        ----------
        method : str, default='peaks'
            Defines which set of frequencies are used.
            
            - 'peaks'
            - 'extended_peaks'
            
        octave : float, default=2
            Value of period interval.

        Returns
        -------
        self.euler_fokker : List (float)
            Euler-Fokker genera.

        """
        if method == "peaks":
            intervals = self.peaks
        if method == "extended_peaks":
            intervals = self.extended_peaks
        intervals = prime_factor([int(x) for x in intervals])
        multiplicities = [1 for x in intervals]  # Each factor is used once.
        scale = create_euler_fokker_scale(intervals, multiplicities, octave=octave)
        self.euler_fokker = scale
        return scale

    def harmonic_tuning(self, list_harmonics, octave=2, min_ratio=1, max_ratio=2):
        """
        Generates a tuning based on a list of harmonic positions.

        Parameters
        ----------
        list_harmonics: List of int
            harmonic positions to use in the scale construction
        octave: int, default=2
            value of the period reference
        min_ratio: float, default=1
            Value of the unison.
        max_ratio: float, default=2
            Value of the octave.

        Returns
        -------
        ratios : List (float)
            Generated tuning.

        """
        ratios = []
        for i in list_harmonics:
            ratios.append(rebound(1 * i, min_ratio, max_ratio, octave))
        ratios = list(set(ratios))
        ratios = list(np.sort(np.array(ratios)))
        self.harmonic_tuning = ratios
        return ratios

    def harmonic_fit_tuning(self, n_harm=128, bounds=0.1, n_common_harms=2):
        """
        Extracts the common harmonics of spectral peaks and compute
        the associated harmonic tuning.

        Parameters
        ----------
        n_harm : int
            Number of harmonics to consider in the harmonic fit`.
        bounds : float, default=0.1
            Maximal distance in Hertz between two frequencies to consider
            them as equivalent.
        n_common_harms : int, default=2
            minimum number of times the harmonic is found
            to be sent to most_common_harmonics output.

        Returns
        -------
        self.harmonic_fit_tuning : List (float)
            Generated tuning

        """

        _, harmonics, common_h, _ = harmonic_fit(
            self.peaks, n_harm=n_harm, bounds=bounds, n_common_harms=n_common_harms
        )
        self.harmonic_fit_tuning = harmonic_tuning(common_h)
        return self.harmonic_fit_tuning

    def pac(
        self,
        sf=None,
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
        """
        Computes the phase-amplitude coupling and returns to pairs of
        frequencies that have highest coupling value.

        Parameters
        ----------
        sf : int
            Sampling frequency in hertz.
        method : str, default='duprelatour'
            Choice of method for PAC calculation.
                STANDARD_PAC_METRICS:
                    
                    - 'ozkurt'
                    - 'canolty'
                    - 'tort'
                    - 'penny'
                    - 'vanwijk'
                    
                DAR_BASED_PAC_METRICS:
                    
                    - 'duprelatour'
                    
                COHERENCE_PAC_METRICS:
                    
                    - 'jiang'
                    - 'colgin'
                    
                BICOHERENCE_PAC_METRICS:
                    
                    - 'sigl'
                    - 'nagashima'
                    - 'hagihira'
                    - 'bispectrum'
                
        n_values : int, default=10
            Number of pairs of frequencies to return.
        drive_precision : float, default=0.05
            Step-size between each phase signal bins.
        max_drive_freq : float, default=6
            Maximum value of the phase signal in hertz.
        min_drive_freq : float, default=3
            Minimum value of the phase signal in hertz.
        sig_precision : float, default=1
            Step-size between each amplitude signal bins.
        max_sig_freq : float, default=50
            Maximum value of the amplitude signal in hertz.
        min_sig_freq : float, default=8
            Minimum value of the amplitude signal in hertz.
        low_fq_width : float, default=0.5
            Bandwidth of the band-pass filter (phase signal)
        high_fq_width : float, default=1
            Bandwidth of the band-pass filter (amplitude signal)
        plot : Boolean, default=False
            When set to True, a plot of the comodulogram is generated.

        Returns
        -------
        self.pac_freqs : List of lists (float)
            Pairs of frequencies with highest coupling value.
        self.pac_coupling : List (float)
            Values of coupling for each pair in self.pac_freqs
        """
        if sf is None:
            sf = self.sf
        freqs, pac_coupling = pac_frequencies(
            self.data,
            sf,
            method=method,
            n_values=n_values,
            drive_precision=drive_precision,
            max_drive_freq=max_drive_freq,
            min_drive_freq=min_drive_freq,
            sig_precision=sig_precision,
            max_sig_freq=max_sig_freq,
            min_sig_freq=min_sig_freq,
            low_fq_width=low_fq_width,
            high_fq_width=high_fq_width,
            plot=plot,
        )
        self.pac_freqs = freqs
        self.pac_coupling = pac_coupling
        return self.pac_freqs, self.pac_coupling

    def compute_peaks_ts(
        self,
        data,
        peaks_function="EMD",
        FREQ_BANDS=None,
        precision=0.5,
        sf=None,
        min_freq=1,
        max_freq=80,
        min_harms=2,
        harm_limit=128,
        n_peaks=5,
        nIMFs=None,
        graph=False,
        noverlap=None,
        average="median",
        nfft=None,
        nperseg=None,
        max_harm_freq=None,
        EIMC_order=3,
        min_IMs=2,
        smooth_fft=1,
        keep_first_IMF=False
    ):
        """
        Extract peak frequencies. This method is called by the
        peaks_extraction method.

        Parameters
        ----------
        data: array (numDataPoints,)
            Niosignal to analyse
        peaks_function: str
            Refer to __init__
        FREQ_BANDS: List of lists of floats
            Each list within the list of lists sets the lower and
            upper limit of a frequency band
        precision: float, default=0.5
            Precision of the peaks (in Hz)
            When HH1D_max is used, bins are in log scale.
        sf : int, default=None
            Sampling frequency in hertz.
        min_freq: float, default=1
            Minimum frequency value to be considered as a peak
            Used with 'harmonic_recurrence' and 'HH1D_max' peaks functions
        max_freq: float, default=80
            Maximum frequency value to be considered as a peak
            Used with 'harmonic_recurrence' and 'HH1D_max' peaks functions
        min_harms : int, default=2
            Minimum number of harmonics to be considered for peaks extraction
        harm_limit: int, default=128
            Maximum harmonic position for 'harmonic_recurrence' method.
        n_peaks: int, default=5
            Number of peaks when using 'FOOOF' and 'cepstrum',
            and 'harmonic_recurrence' functions.
            Peaks are chosen based on their amplitude.
        nIMFs: int, default=None
            number of intrinsic mode functions to keep when using
            'EEMD' or 'EMD' peaks function.
        graph: boolean, default=False
            When set to True, a graph will accompanies the peak extraction
            method (except for 'fixed' and 'adapt').
        noverlap : int, default=None
            Number of samples overlap between each fft window.
            When set to None, equals sf//10.
        average : str, default='median'
            Method to use when averaging periodograms.
            
            - 'mean': average periodograms
            - 'median': median periodograms 
            
        max_harm_freq : int, default=None
            Maximum frequency value of the find peaks function
            when harmonic_recurrence or EIMC peaks extraction method is used.
        EIMC_order : int, default=3
            Maximum order of the Intermodulation Components.
        min_IMs : int, default=2
            Minimal number of Intermodulation Components to select the
            associated pair of peaks.
        smooth_fft : int, default=1
            Number used to divide nfft to derive nperseg.
            When set to 1, nperseg = nfft.
        keep_first_IMF : boolean, default=False
            When set to True, the first IMF is kept.

        Returns
        -------
        peaks : List (float)
            List of peaks frequencies.
        amps : List (float)
            List of amplitudes associated with peaks frequencies.
        
        Attributes
        ----------
        self.freqs : array
            Vector representing the frequency bins of
            the Power Spectrum Density (PSD)
        self.psd : array
            Vector representing the PSD values for each frequency bin.
        self.IMFs : array (nIMFs, numDataPoints)
            Intrinsic mode functions resulting from decomposing the signal
            with Empirical Mode Decomposition.
        self.IF : array (nIMFs, numDataPoints)
            instantaneous frequencies for each IMF.
            Only when 'HH1D_max' is used as peaks extraction method.
        self.all_harmonics : List (int)
            List of all harmonic positions when
            harmonic_recurrence method is used.
        self.FREQ_BANDS : List of lists (float)
            List of frequency bands.
        """
        alphaband = [[7, 12]]
        if sf is None:
            sf = self.sf
        if nIMFs is None:
            nIMFs = self.nIMFs
        if FREQ_BANDS is None:
            FREQ_BANDS = [
                [2, 3.55],
                [3.55, 7.15],
                [7.15, 14.3],
                [14.3, 28.55],
                [28.55, 49.4],
            ]
        if FREQ_BANDS is not None:
            FREQ_BANDS = FREQ_BANDS
        if max_harm_freq is None:
            max_harm_freq = sf / 2
        self.FREQ_BANDS = FREQ_BANDS
        if peaks_function == "adapt":
            p, a = extract_welch_peaks(
                data,
                sf=sf,
                FREQ_BANDS=alphaband,
                out_type="bands",
                precision=precision,
                average=average,
                extended_returns=False,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                smooth=smooth_fft
            )
            FREQ_BANDS = alpha2bands(p[0])
            self.FREQ_BANDS = FREQ_BANDS
            print('Adaptive frequency bands: ', FREQ_BANDS)
            peaks_temp, amps_temp, self.freqs, self.psd = extract_welch_peaks(
                data,
                sf=sf,
                FREQ_BANDS=FREQ_BANDS,
                out_type="bands",
                precision=precision,
                average=average,
                extended_returns=True,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                smooth=smooth_fft
            )
            if graph is True:
                graph_psd_peaks(
                    self.freqs,
                    self.psd,
                    peaks_temp,
                    xmin=min_freq,
                    xmax=max_freq,
                    color="darkblue",
                    method=peaks_function,
                )
        if peaks_function == "fixed":
            peaks_temp, amps_temp, self.freqs, self.psd = extract_welch_peaks(
                data,
                sf=sf,
                FREQ_BANDS=FREQ_BANDS,
                out_type="bands",
                precision=precision,
                average=average,
                extended_returns=True,
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                smooth=smooth_fft
            )
            if graph is True:
                graph_psd_peaks(
                    self.freqs,
                    self.psd,
                    peaks_temp,
                    xmin=min_freq,
                    xmax=max_freq,
                    color="darkred",
                    method=peaks_function
                )
        if peaks_function == "FOOOF":
            peaks_temp, amps_temp, self.freqs, self.psd = compute_FOOOF(
                data,
                sf,
                precision=precision,
                max_freq=max_freq,
                noverlap=None,
                n_peaks=n_peaks,
                extended_returns=True,
                graph=graph,
            )

        if (
            peaks_function == "EMD"
            or peaks_function == "EEMD"
            or peaks_function == "EMD_fast"
            or peaks_function == "EEMD_fast"
        ):
            IMFs = EMD_eeg(
                data,
                method=peaks_function,
                graph=graph,
                extrema_detection="simple",
                nIMFs=nIMFs,
            )
            if keep_first_IMF is True:
                self.IMFs = IMFs[0: nIMFs + 1]
                IMFs = IMFs[0: nIMFs + 1]
            if keep_first_IMF is False:
                self.IMFs = IMFs[1: nIMFs + 1]
                IMFs = IMFs[1: nIMFs + 1]
            try:
                peaks_temp = []
                amps_temp = []
                freqs_all = []
                psd_all = []
                for imf in range(len(IMFs)):
                    p, a, freqs, psd = extract_welch_peaks(
                        IMFs[imf],
                        sf=sf,
                        precision=precision,
                        average=average,
                        extended_returns=True,
                        out_type="single",
                        nperseg=nperseg,
                        noverlap=noverlap,
                        nfft=nfft,
                        smooth=smooth_fft
                    )
                    #self.freqs = freqs
                    freqs_all.append(freqs)
                    psd_all.append(psd)
                    peaks_temp.append(p)
                    amps_temp.append(a)
                peaks_temp = np.flip(peaks_temp)
                amps_temp = np.flip(amps_temp)
                peaks_temp = peaks_temp[-n_peaks:]
                _, _, self.freqs, self.psd = extract_welch_peaks(
                                                                    data,
                                                                    sf=sf,
                                                                    FREQ_BANDS=FREQ_BANDS,
                                                                    out_type="bands",
                                                                    precision=precision,
                                                                    average=average,
                                                                    extended_returns=True,
                                                                    nperseg=nperseg,
                                                                    noverlap=noverlap,
                                                                    nfft=nfft,
                                                                    smooth=smooth_fft
                                                                )
            except:
                pass
            if graph is True:
                '''graphEMD_welch(
                    freqs_all,
                    psd_all,
                    peaks=peaks_temp,
                    raw_data=self.data,
                    FREQ_BANDS=FREQ_BANDS,
                    sf=sf,
                    nfft=nfft,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    min_freq=min_freq,
                    max_freq=max_freq,
                )'''
                EMD_PSD_graph(self.data, self.IMFs, peaks_temp, spectro='Euler', bands = None, xmin=min_freq, xmax=max_freq,
                                  compare = True, name = '', nfft=nfft, nperseg=nperseg, noverlap=noverlap, sf=self.sf,
                                  freqs_all=freqs_all, psd_all=psd_all, max_freq=max_freq, precision=precision)
                #EMD_PSD_graph(peaks_temp, IMFs, freqs_all, psd_all, spectro='Euler', bands=None, xmin=1, xmax=70, plot_type = 'line',
                #              compare=True, input_data='EEG', name='',sf=self.sf,
                #              raw_data=self.data, precision=precision, noverlap=noverlap, save=False)

        if peaks_function == "EMD_FOOOF":
            nfft = sf / precision
            nperseg = sf / precision
            IMFs = EMD_eeg(
                data, method="EMD_fast", graph=graph, extrema_detection="simple"
            )[1: nIMFs + 1]
            self.IMFs = IMFs
            peaks_temp = []
            amps_temp = []
            for imf in IMFs:
                freqs1, psd = scipy.signal.welch(
                    imf, sf, nfft=nfft, nperseg=nperseg, noverlap=noverlap
                )
                self.freqs = freqs1
                self.psd = psd
                fm = FOOOF(
                    peak_width_limits=[precision * 2, 3],
                    max_n_peaks=50,
                    min_peak_height=0.2,
                )
                freq_range = [(sf / len(data)) * 2, max_freq]
                fm.fit(freqs1, psd, freq_range)
                if graph is True:
                    fm.report(freqs1, psd, freq_range)
                peaks_temp_EMD = fm.peak_params_[:, 0]
                amps_temp_EMD = fm.peak_params_[:, 1]
                try:
                    # Select the peak with highest amplitude.
                    peaks_temp.append(
                        [x for _, x in sorted(zip(amps_temp_EMD, peaks_temp_EMD))][
                            ::-1
                        ][0:1]
                    )
                    amps_temp.append(sorted(amps_temp_EMD)[::-1][0:1])
                except:
                    print("No peaks detected")
            peaks_temp = [np.round(p, 2) for p in peaks_temp]
            peaks_temp = [item for sublist in peaks_temp for item in sublist]
            amps_temp = [item for sublist in amps_temp for item in sublist]

        if peaks_function == "HH1D_max":
            if smooth_fft == 1:
                smooth_sigma = None
            IF, peaks_temp, amps_temp, HH_spec, HH_bins = HilbertHuang1D(
                data,
                sf,
                graph=graph,
                nIMFs=nIMFs,
                min_freq=min_freq,
                max_freq=max_freq,
                precision=precision,
                bin_spread="log",
                smooth_sigma=smooth_sigma
            )
            self.IF = IF
        # if peaks_function == 'HH1D_weightAVG':
        # if peaks_function == 'HH1D_FOOOF':
        if peaks_function == "bicoherence":
            freqs, amps = polyspectrum_frequencies(
                data,
                sf,
                precision,
                n_values=n_peaks,
                nperseg=nperseg,
                noverlap=noverlap,
                method=peaks_function,
                flim1=(min_freq, max_freq),
                flim2=(min_freq, max_freq),
                graph=graph,
            )
            common_freqs = flatten(pairs_most_frequent(freqs, n_peaks))
            peaks_temp = list(np.sort(list(set(common_freqs))))
            peaks_temp = [p for p in peaks_temp if p < max_freq][0:n_peaks]
            '''amp_idx = []
            for i in peaks_temp:
                amp_idx.append(flatten(freqs).index(i))
            amps_temp = np.array(flatten(amps))[amp_idx]
            amps_temp = list(amps_temp)
            # Select the n peaks with highest amplitude.
            peaks_temp = [x for _, x in sorted(zip(amps_temp, peaks_temp))][::-1][0:n_peaks]
            amps_temp = sorted(amps_temp)[::-1][0:n_peaks]'''
            amps_temp = 'NaN'
        if peaks_function == "harmonic_recurrence":
            p, a, self.freqs, self.psd = extract_welch_peaks(
                data,
                sf,
                precision=precision,
                max_freq=max_harm_freq,
                extended_returns=True,
                out_type="all",
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                min_freq=min_freq,
                smooth=smooth_fft
            )

            (
                max_n,
                peaks_temp,
                amps_temp,
                harms,
                harm_peaks,
                harm_peaks_fit,
            ) = harmonic_recurrence(
                p, a, min_freq, max_freq, min_harms=min_harms, harm_limit=harm_limit
            )
            try:
                list_harmonics = np.concatenate(harms)
                list_harmonics = list(set(abs(np.array(list_harmonics))))
                list_harmonics = [h for h in list_harmonics if h <= harm_limit]
                list_harmonics = np.sort(list_harmonics)
                self.all_harmonics = list_harmonics
                self.harm_peaks_fit = harm_peaks_fit
                self.n_harmonic_recurrence = len(harm_peaks_fit)
                # Select the n peaks with maximum number of harmonic recurrence.
                peaks_temp = [x for _, x in sorted(zip(max_n, peaks_temp))][::-1][
                    0:n_peaks
                ]
                amps_temp = [x for _, x in sorted(zip(max_n, amps_temp))][::-1][
                    0:n_peaks
                ]
                #amps_temp = sorted(amps_temp)[::-1][0:n_peaks]
                if graph is True:
                    graph_harm_peaks(self.freqs, self.psd,
                                     harm_peaks_fit, min_freq,
                                     max_freq, color='black',
                                     method=peaks_function, save=False,
                                     figname='test')
            except ValueError:
                print('No peaks were detected. Consider increasing precision or number of harmonics')

        if peaks_function == "EIMC":
            p, a, self.freqs, self.psd = extract_welch_peaks(
                data,
                sf,
                precision=precision,
                max_freq=max_harm_freq,
                extended_returns=True,
                out_type="all",
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                min_freq=min_freq,
                smooth=smooth_fft
            )
            IMC, self.EIMC_all, n = endogenous_intermodulations(
                p, a, order=EIMC_order, min_IMs=min_IMs
            )
            IMC_freq = pairs_most_frequent(self.EIMC_all["peaks"], n_peaks)
            common_freqs = flatten(IMC_freq)
            peaks_temp = list(np.sort(list(set(common_freqs))))
            peaks_temp = [p for p in peaks_temp if p < max_freq]
            amp_idx = []
            for i in peaks_temp:
                amp_idx.append(flatten(self.EIMC_all["peaks"]).index(i))
            amps_temp = np.array(flatten(self.EIMC_all["amps"]))[amp_idx]
            amps_temp = list(amps_temp)
            peaks_temp = [x for _, x in sorted(zip(amps_temp, peaks_temp))][::-1][
                0:n_peaks
            ]

            amps_temp = sorted(amps_temp)[::-1][0:n_peaks]
            if graph is True:
                graph_psd_peaks(
                    self.freqs,
                    self.psd,
                    peaks_temp,
                    xmin=min_freq,
                    xmax=max_freq,
                    color="darkgoldenrod",
                    method=peaks_function,
                )
        if peaks_function == "PAC":
            freqs, amps = self.pac(
                sf=sf,
                method="duprelatour",
                n_values=n_peaks,
                drive_precision=precision,
                max_drive_freq=max_freq / 2,
                min_drive_freq=min_freq,
                sig_precision=precision * 2,
                max_sig_freq=max_freq,
                min_sig_freq=min_freq * 2,
                low_fq_width=0.5,
                high_fq_width=1,
                plot=graph,
            )
            common_freqs = flatten(pairs_most_frequent(freqs, n_peaks))
            peaks_temp = list(np.sort(list(set(common_freqs))))
            peaks_temp = [p for p in peaks_temp if p < max_freq][0:n_peaks]
            '''amp_idx = []
            for i in peaks_temp:
                amp_idx.append(flatten(freqs).index(i))
            amps_temp = np.array(amps)[amp_idx]
            amps_temp = list(amps_temp)
            peaks_temp = [x for _, x in sorted(zip(amps_temp, peaks_temp))][::-1][
                0:n_peaks
            ]'''

            amps_temp = 'NaN'
        if peaks_function == "cepstrum":
            cepstrum_, quefrency_vector = cepstrum(
                self.data,
                self.sf,
                min_freq=min_freq,
                max_freq=max_freq,
                plot_cepstrum=graph,
            )
            peaks_temp_, amps_temp_ = cepstral_peaks(
                cepstrum_, quefrency_vector, 1 / min_freq, 1 / max_freq
            )
            peaks_temp_ = list(np.flip(peaks_temp_))
            peaks_temp = [np.round(p, 2) for p in peaks_temp_]
            amps_temp_ = list(np.flip(amps_temp_))
            peaks_temp = [x for _, x in sorted(zip(amps_temp_, peaks_temp))][::-1][
                0:n_peaks
            ]
            amps_temp = sorted(amps_temp_)[::-1][0:n_peaks]
        # print(peaks_temp)
        peaks_temp = [0 + precision if x == 0 else x for x in peaks_temp]
        peaks = np.array(peaks_temp)
        peaks = np.around(peaks, 3)
        amps = np.array(amps_temp)
        return peaks, amps

    def compute_resonance(self, harm_thresh=30, PPC_thresh=0.6, smooth_fft=2,
                          harmonicity_metric='harmsim', delta_lim=50):
        """Compute resonances between pairs of frequency peaks in the data.

        Parameters
        ----------
        harm_thresh : int, default=30
            The minimum harmonic similarity between a peak pair required to be considered a resonance.
            Must be a positive integer.
        PPC_thresh : float, default=0.6
            The minimum bispectral power correlation required for a peak pair to be considered a resonance.
            Must be a float between 0 and 1.
        smooth_fft : int, default=2
            The number of times to smooth the data using a Hamming window before computing the FFT.
            Must be a positive integer. When smooth_fft=1, nperseg=nfft.
        harmonicity_metric : str, default='harmsim'
            The metric to use for computing the harmonic similarity between a pair of peaks.
            Choose between:
            
            - 'harmsim'
            - 'subharm_tension'
            
        delta_lim : int, default=50
            The maximum number of subharmonic intervals to consider when using the 'subharm_tension' metric.
            Must be a positive integer.

        Returns
        -------
        Tuple[float, List[Tuple[float, float]], List[float], List[float]]
            A tuple containing the following elements:
            
            - **resonance**: a float representing the mean weighted bicorrelation coefficient across all harmonic pairs that meet the specified criteria for harmonicity and PPC
            - **resonant_freqs**: a list of tuples, where each tuple contains two floats representing the frequencies of a pair of resonant harmonics that meet the specified criteria for harmonicity and PPC
            - **harm_all**: a list of floats representing the harmonic similarity metric between all possible harmonic pairs
            - **bicor_all**: a list of floats representing the bicorrelation coefficient between all possible harmonic pairs

        """        
        if self.peaks_function != 'EMD' and self.peaks_function != 'EMD_fast' and self.peaks_function != 'harmonic_recurrence' and self.peaks_function != 'FOOOF':
            print('Peaks extraction function {} is not compatible with resonance metrics'.format(self.peaks_function))
        if len(self.peaks) < 1:
            print('No peaks in the biotuner object. Please use peaks_extraction method first')
        if self.precision is not None:
            mult = 1 / self.precision
            nfft = int(self.sf * mult)
            nperseg = int(nfft/smooth_fft)
        max_peak = np.max(self.peaks)
        freq1, freq2, bispec = polycoherence(
                self.data, self.sf, norm=2, flim1=[1, max_peak+self.precision], flim2=[1, max_peak+self.precision], dim=2,
                nperseg=nperseg, nfft=nfft)

        pairs = list(combinations(self.peaks, 2))

        harm = []
        bicor = []
        weighted_bicor = []
        resonant_freqs = []
        harm_all = []
        bicor_all = []
        for pair in pairs:
            if pair[0] > pair[1]:
                ratio = pair[0]/pair[1]
            if pair[0] <= pair[1]:
                ratio = pair[1]/pair[0]
            if harmonicity_metric == 'harmsim':
                harm_ = dyad_similarity(ratio)
            if harmonicity_metric == 'subharm_tension':
                _, _, harm_, _ = compute_subharmonic_tension(pair,
                                                             self.n_harm,
                                                             delta_lim=delta_lim,
                                                             min_notes=2)
                harm_ = 1 - harm_
            # Determine the number of decimal places you want to consider
            n_decimals = 1 

            # Convert freq1 to a list of rounded values
            freq1_rounded = [round(f, n_decimals) for f in freq1]

            # Round the pair values
            pair_rounded = (round(pair[0], n_decimals), round(pair[1], n_decimals))

            # Now use the rounded values to find the indices
            idx1 = freq1_rounded.index(pair_rounded[0])
            idx2 = freq1_rounded.index(pair_rounded[1])
            #idx1 = (np.abs(np.array(freq1) - pair[0])).argmin()
            #idx2 = (np.abs(np.array(freq1) - pair[1])).argmin()
            bicor_ = np.real(bispec[idx1][idx2])

            if bicor_ < 1:
                harm_all.append(harm_)
                bicor_all.append(bicor_)
            if harm_ > harm_thresh:
                if bicor_ < 1:
                    bicor.append(bicor_)
                    harm.append(harm_)
                    if harmonicity_metric == 'harmsim':
                        weighted_bicor.append((harm_/100)*bicor_)
                    if harmonicity_metric == 'subharm_tension':
                        weighted_bicor.append((harm_)*bicor_)
                    if bicor_ > PPC_thresh:
                        resonant_freqs.append((pair[0], pair[1]))
        # resonance = np.corrcoef(harm_sim, bicor)[0][1]
        resonance_ = np.mean(weighted_bicor)
        self.resonance = resonance_
        self.resonant_freqs = resonant_freqs
        scale = scale_from_pairs(resonant_freqs)
        self.res_tuning = np.sort(list(set(scale)))
        self.PPC_bicor = np.mean(bicor_all)

        return resonance_, resonant_freqs, harm_all, bicor_all


    """Listening methods"""

    def listen_scale(self, scale, fund=250, length=500):
        """
        Play a scale of notes using pygame.

        Parameters
        ----------
        scale : str or np.ndarray
            The scale to play. 
            If `scale` is a string, it can be one of:
            
            - 'peaks': the scale is set to the biotuner object's `peaks_ratios` attribute
            - 'diss': the scale is set to the biotuner object's `diss_scale` attribute
            - 'HE': the scale is set to the biotuner object's `HE_scale` attribute
            
            If `scale` is a numpy array, it should be an array of scale
            ratios.
        fund : float, default=250
            The fundamental frequency of the scale.
        length : float, default=500
            The length of each note in milliseconds.

        Returns
        -------
        None
        """
        if self.pygame_lib is None:
            import pygame
            self.pygame_lib = pygame
        if scale == "peaks":
            scale = self.peaks_ratios
        if scale == "diss":
            try:
                scale = self.diss_scale
            except:
                print("No Dissonance Curve scale available")
                pass
        if scale == "HE":
            try:
                scale = list(self.HE_scale)
            except:
                print("No Harmonic Entropy scale available")
                pass
        scale = np.around(scale, 3)
        print("Scale:", scale)
        scale = list(scale)
        scale = [1] + scale
        for s in scale:
            freq = fund * s
            note = make_chord(freq, [1])
            note = np.ascontiguousarray(np.vstack([note, note]).T)
            sound = pygame.sndarray.make_sound(note)
            sound.play(loops=0, maxtime=0, fade_ms=0)
            pygame.time.wait(int(sound.get_length() * length))

    """Generic method to fit all Biotuner methods"""

    def fit_all(
        self, data, compute_diss=True, compute_HE=True, compute_peaks_extension=True
    ):
        """
        Fit biotuning metrics to input data using various optional computations.

        Parameters
        ----------
        data : array, shape (n_samples,)
            A single time series of EEG data.
        compute_diss : bool, optional, default=True
            If True, compute the dissonance curve.
        compute_HE : bool, optional, default=True
            If True, compute the harmonic entropy.
        compute_peaks_extension : bool, optional, default=True
            If True, compute the peaks extension using the multi-consonant harmonic fit method.

        Returns
        -------
        biotuning : Biotuning object
            The fitted biotuning object containing the computed metrics.
            
        """
        biotuning = compute_biotuner(
            self.sf,
            peaks_function=self.peaks_function,
            precision=self.precision,
            n_harm=self.n_harm,
        )
        biotuning.peaks_extraction(data)
        biotuning.compute_peaks_metrics()
        if compute_diss is True:
            biotuning.compute_diss_curve(input_type="peaks", plot=False)
        if compute_peaks_extension is True:
            biotuning.peaks_extension(
                method="multi_consonant_harmonic_fit",
                harm_function="mult",
                cons_limit=0.01,
            )
        if compute_HE is True:
            biotuning.compute_harmonic_entropy(
                input_type="extended_peaks", plot_entropy=False
            )
        return biotuning

    def info(self, metrics=False, scales=False, whatever=False):
        if metrics is True:
            print("METRICS")
            print(vars(self))

        else:
            print(vars(self))
        return


def fit_biotuner(ts, bt_dict):
    """
    Compute biotuner metrics on a single time series using a specified set of parameters.

    Parameters
    ----------
    ts : array, shape (n_samples,)
        A single time series of EEG data.
    bt_dict : dict
        A dictionary containing the parameters to compute biotuner metrics. See the compute_biotuner and
        tuning_to_metrics functions documentation for details on the required keys and values.

    Returns
    -------
    bt_dict : dict
        The modified input dictionary with the computed biotuner metrics added.
    """
    # Create the biotuner object
    biotuning = compute_biotuner(sf=bt_dict['sf'], peaks_function=bt_dict['peaks_function'],
                                 precision=bt_dict['precision'], n_harm=10,
                                 ratios_n_harms=5, ratios_inc_fit=False, ratios_inc=False)
    
    # Extract the peaks from the time series
    biotuning.peaks_extraction(ts, FREQ_BANDS=None, ratios_extension=False, max_freq=bt_dict['fmax'],
                               n_peaks=bt_dict['n_peaks'], min_freq=bt_dict['fmin'],
                               graph=False, min_harms=2, nIMFs=5)
    
    # Compute the peaks metrics and resonance
    biotuning.compute_peaks_metrics(n_harm=10, delta_lim=bt_dict['delta_lim'])
    biotuning.compute_resonance(harm_thresh=bt_dict['harm_thresh'], smooth_fft=2,
                                harmonicity_metric='harmsim', delta_lim=bt_dict['delta_lim'])
    
    # Convert the peaks ratios and peaks metrics to a dictionary of biotuner metrics
    bt_metrics = tuning_to_metrics(biotuning.peaks_ratios)
    bt_metrics.update(biotuning.peaks_metrics)
    
    # Remove unneeded metrics from the dictionary
    del bt_metrics['harm_pos']
    del bt_metrics['euler']
    del bt_metrics['common_harm_pos']
    
    # Add additional metadata to the dictionary
    peaks_euler = [int(round(num, 2) * 1000) for num in biotuning.peaks]
    bt_dict['peaks_avg'] = np.mean(biotuning.peaks)
    bt_dict['n_peaks'] = len(biotuning.peaks)
    bt_dict['euler'] = euler(*peaks_euler)
    bt_dict['resonance'] = biotuning.resonance
    bt_dict['PPC_bicor'] = biotuning.PPC_bicor
    
    # Update the bt_dict dictionary with the computed biotuner metrics
    bt_dict.update(bt_metrics)
    
    # Return the modified bt_dict dictionary
    return bt_dict