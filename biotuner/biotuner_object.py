from fooof import FOOOF
import scipy.signal
from pytuning import create_euler_fokker_scale
import matplotlib.pyplot as plt
from biotuner.peaks_extraction import (HilbertHuang1D,
                                       harmonic_recurrence,
                                       cepstrum, cepstral_peaks,
                                       EMD_eeg)
import numpy as np
from biotuner.peaks_extraction import (extract_welch_peaks, compute_FOOOF,
                                       polyspectrum_frequencies,
                                       pac_frequencies,
                                       endogenous_intermodulations)
from biotuner.biotuner_utils import (flatten, pairs_most_frequent,
                                     compute_peak_ratios, alpha2bands,
                                     rebound, prime_factor, peaks_to_amps,
                                     EMD_to_spectromorph, ratios_harmonics,
                                     ratios_increments, make_chord)
from biotuner.metrics import (euler, tenneyHeight, timepoint_consonance,
                              ratios2harmsim, compute_subharmonics_5notes)
from biotuner.peaks_extension import (consonant_ratios, harmonic_fit,
                                      consonance_peaks, multi_consonance)
from biotuner.scale_construction import (diss_curve, harmonic_entropy,
                                         harmonic_tuning)
from biotuner.vizs import graph_psd_peaks, graphEMD_welch
import seaborn as sbn
import pygame


class compute_biotuner(object):
    '''
    Class used to derive peaks information, musical tunings, rhythms
    and related harmonicity metrics from time series
    (EEG, ECG, EMG, gravitational waves, noise, ...)

    Example of use:
    biotuning = biotuner(sf = 1000)
    biotuning.peaks_extraction(data)
    biotuning.peaks_extension()
    biotuning.peaks_metrics()

    Methods
    -------
    peaks_extraction(data)
        Extract spectral peaks from time series
    peaks_extension(peaks)
        Extend or restrict a set of spectral peaks
    '''

    def __init__(self, sf, data=None, peaks_function='EMD', precision=0.1,
                 compute_sub_ratios=False, n_harm=10, harm_function='mult',
                 extension_method='consonant_harmonic_fit', ratios_n_harms=5,
                 ratios_harms=False, ratios_inc=True, ratios_inc_fit=False,
                 scale_cons_limit=0.1):
        '''
        Parameters
        ----------
        sf: int
            sampling frequency (in Hz)
        data : array(numDataPoints,)
            Time series to analyse.

        ///// PEAKS EXTRACTION PARAMETERS /////
        peaks_function: str
            Defaults to 'EMD'.
            Defines the method to use for peak extraction
            ##### NON-HARMONIC PEAK EXTRACTIONS #####
            'fixed' : Power Spectrum Density (PSD) estimated using
                      Welch's method on fixed frequency bands.
                      Peaks correspond to frequency bins with
                      the highest power.
            'adapt' : PSD estimated using Welch's method on each
                      frequency bands derived from the alpha peak position.
                      Peaks correspond to frequency bins with
                      the highest power.
            'FOOOF' : PSD is estimated with Welch's method.
                      'FOOOF' is applied to remove the aperiodic
                      component and find physiologically relevant
                      spectral peaks.
            ##### SIGNAL DECOMPOSITION BASED PEAK EXTRACTION #####
            'EMD': Intrinsic Mode Functions (IMFs) are derived with
                   Empirical Mode Decomposition (EMD)
                   PSD is computed on each IMF using Welch. Peaks correspond
                   to frequency bins with the highest power.
            'EEMD' : Intrinsic Mode Functions (IMFs) are derived with
                     Ensemble Empirical Mode Decomposition (EEMD).
                     PSD is computed on each IMF using Welch. Peaks correspond
                     to frequency bins with the highest power.
            'EEMD_FOOOF' : Intrinsic Mode Functions (IMFs) are derived with
                           Ensemble Empirical Mode Decomposition (EEMD).
                           PSD is computed on each IMF with Welch's method.
                           'FOOOF' is applied to remove the aperiodic component
                           and find physiologically relevant spectral peaks.
            'HH1D_max' : Maximum values of the 1D Hilbert-Huang transform
                         on each IMF using EMD.
            'HH1D_avg' : Weighted average values of the 1D Hilbert-Huang
                         transform on each IMF using EMD.
            'HH1D_FOOOF' :
            'SSA' : Singular Spectrum Analysis.
                    The name "singular spectrum analysis" relates to the
                    spectrum of eigenvalues in a singular value decomposition
                    of a covariance matrix.
            ##### SECOND-ORDER STATISTICAL PEAK EXTRACTION #####
            'cepstrum': Peak frequencies of the cepstrum
                        (inverse Fourier transform (IFT) of the logarithm
                        of the estimated signal spectrum).
            'HPS' : Harmonic Product Spectrum (HPS) corresponds to
            'Harmonic_salience' :
            ##### CROSS-FREQUENCY COUPLING BASED PEAK EXTRACTION #####
            'Bicoherence' : Corresponds to the normalised cross-bispectrum.
                            Third-order moment in the frequency domain.
                            Measure of phase-amplitude coupling.
            'PAC' : Measure of phase-amplitude coupling between
                    low-freq phase and high-freq amplitude.
            ##### PEAK SELECTION BASED ON HARMONIC PROPERTIES #####
            'EIMC' : PSD is estimated with Welch's method.
                     All peaks are extracted.
                     Endogenous InterModulation Components (EIMC)
                     correspond to spectral peaks that are sums or differences
                     of other peaks harmonics (f1+f2, f1+2f2, f1-f2, f1-2f2...)
            'Harmonic_peaks' : PSD is estimated with Welch's method.
                               All peaks are extracted. Peaks for which
                               other peaks are their harmonics are kept.
            'Harmonic_symmetry' :

        precision: float
            Defaults to 0.1
            precision of the peaks (in Hz)
            When HH1D_max is used, bins are in log scale.
        compute_sub_ratios: str
            Default to False
            When set to True, include ratios < 1 in peaks_ratios attribute
        scale_cons_limit : float
            Defaults to 0.1
            minimal value of consonance to be reach for a peaks ratio
            to be included in the peaks_ratios_cons attribute.

        ///// EXTENDED PEAKS PARAMETERS /////
        n_harm: int
            Defaults to 10.
            Set the number of harmonics to compute in harmonic_fit function
        harm_function: str
            {'mult' or 'div'}
            Defaults to 'mult'
            Computes harmonics from iterative multiplication (x, 2x, 3x, ...nx)
            or division (x, x/2, x/3, ...x/n).
        extension_method: str
            {'harmonic_fit', 'consonant', 'multi_consonant',
            'consonant_harmonic_fit', 'multi_consonant_harmonic_fit'}

        ///// RATIOS EXTENSION PARAMETERS /////
        ratios_n_harms: int
            Defaults to 5.
            Defines to number of harmonics or exponents for extended ratios
        ratios_harms: boolean
            Defaults to False.
            When set to True, harmonics (x*1, x*2, x*3...,x*n) of specified
            ratios will be computed.
        ratios_inc: boolean
            Defaults to True.
            When set to True, exponentials (x**1, x**2, x**3,...x**n) of
            specified ratios will be computed.
        ratios_inc_fit: boolean
            Defaults to False.
            When set to True, a fit between exponentials
            (x**1, x**2, x**3,...x**n) of specified ratios will be computed.

        '''
        '''Initializing data'''
        if type(data) is not None:
            self.data = data
        self.sf = sf
        '''Initializing arguments for peak extraction'''
        self.peaks_function = peaks_function
        self.precision = precision
        self.compute_sub_ratios = compute_sub_ratios
        '''Initializing arguments for peaks metrics'''
        self.n_harm = n_harm
        self.harm_function = harm_function
        self.extension_method = extension_method
        '''Initializing dictionary for scales metrics'''
        self.scale_metrics = {}
        self.scale_cons_limit = scale_cons_limit
        '''Initializing arguments for ratios extension'''
        self.ratios_n_harms = ratios_n_harms
        self.ratios_harms = ratios_harms
        self.ratios_inc = ratios_inc
        self.ratios_inc_fit = ratios_inc_fit

    def peaks_extraction(self, data, peaks_function=None, FREQ_BANDS=None,
                         precision=None, sf=None, min_freq=1, max_freq=60,
                         min_harms=2, compute_sub_ratios=False,
                         ratios_extension=False, ratios_n_harms=None,
                         scale_cons_limit=None, octave=2, harm_limit=128,
                         n_peaks=5, nIMFs=5, graph=False, nfft=None,
                         noverlap=None, nperseg=None, max_harm_freq=None,
                         EIMC_order=3, min_IMs=2):
        '''
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
        compute_sub_ratios: Boolean
            If set to True, will include peaks ratios (x/y) when x < y
        FREQ_BANDS: List of lists (float)
            Each list within the list of lists sets the lower and
            upper limit of a frequency band
        precision: float
            Defaults to None
            precision of the peaks (in Hz)
            When HH1D_max is used, bins are in log scale.
        min_freq: float
            Defaults to 1
            minimum frequency value to be considered as a peak
            Used with 'harmonic_peaks' and 'HH1D_max' peaks functions
        max_freq: float
            Defaults to 60
            maximum frequency value to be considered as a peak
            Used with 'harmonic_peaks' and 'HH1D_max' peaks functions
        min_harms: int
            Defaults to 2
            minimum number of harmonics to consider a peak frequency using
            the 'harmonic_peaks' function
        ratios_extension: Boolean
            Defaults to False
            When set to True, peaks_ratios harmonics and
            increments are computed
        ratios_n_harms: int
            Defaults to None
            number of harmonics or increments to use in ratios_extension method
        scale_cons_limit: float
            Defaults to 0.1 (__init__)
            minimal value of consonance to be reach for a peaks ratio
            to be included in the peaks_ratios_cons attribute.
        octave: float
            Defaults to 2
            value of the octave
        harm_limit: int
            Defaults to 128
            maximum harmonic position for 'harmonic_peaks' method.
        n_peaks: int
            Defaults to 5
            number of peaks when using 'FOOOF' and 'cepstrum',
            and 'harmonic_peaks' functions.
            Peaks are chosen based on their amplitude.
        nIMFs: int
            Defaults to 5
            number of intrinsic mode functions to keep when using
            'EEMD' or 'EMD' peaks function.
        graph: boolean
            Defaults to False
            when set to True, a graph will accompanies the peak extraction
            method (except for 'fixed' and 'adapt').
        max_harm_freq : int
            Maximum frequency value of the find peaks function
            when harmonic_peaks or EIMC peaks extraction method is used.
        EIMC_order : int
            Maximum order of the Intermodulation Components.
        min_IMs : int
            Minimal number of Intermodulation Components to select the
            associated pair of peaks.

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
        ----------If ratios_extension = True:----------
        self.peaks_ratios_harm: List (float)
            List of peaks ratios and their harmonics
        self.peaks_ratios_inc: List (float)
            List of peaks ratios and their increments (ratios**n)
        self.peaks_ratios_inc_bound: List (float)
            List of peaks ratios and their increments (ratios**n)
            bound within one octave
        self.peaks_ratios_inc_fit: List (float)
            List of peaks ratios and their congruent increments (ratios**n)
        '''

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
        peaks, amps = self.compute_peaks_ts(data,
                                            peaks_function=peaks_function,
                                            FREQ_BANDS=FREQ_BANDS,
                                            precision=precision, sf=sf,
                                            min_freq=min_freq,
                                            max_freq=max_freq,
                                            min_harms=min_harms,
                                            harm_limit=harm_limit,
                                            n_peaks=n_peaks, graph=graph,
                                            nfft=nfft, nperseg=nperseg,
                                            noverlap=noverlap,
                                            max_harm_freq=max_harm_freq,
                                            EIMC_order=EIMC_order,
                                            min_IMs=min_IMs)

        self.peaks = peaks
        self.amps = amps
        print('Number of peaks: ', len(peaks))
        self.peaks_ratios = compute_peak_ratios(self.peaks, rebound=True,
                                                octave=octave,
                                                sub=compute_sub_ratios)
        self.peaks_ratios_cons, b = consonant_ratios(self.peaks,
                                                     limit=scale_cons_limit)
        if ratios_extension is True:
            a, b, c = self.ratios_extension(self.peaks_ratios,
                                            ratios_n_harms=ratios_n_harms)
            if a is not None:
                self.peaks_ratios_harms = a
            if b is not None:
                self.peaks_ratios_inc = b
                bound_ = [rebound(x, low=1, high=octave, octave=octave)
                          for x in b]
                self.peaks_ratios_inc_bound = bound_
            if c is not None:
                self.peaks_ratios_inc_fit = c

    def peaks_extension(self, peaks=None, n_harm=None, method=None,
                        harm_function='mult', div_mode='add',
                        cons_limit=0.1, ratios_extension=False,
                        harm_bounds=0.1, scale_cons_limit=None,
                        ):
        """
        This method is used to extend a set of frequencies based on the
        harmonic congruence of specific elements (extend). It can also
        restrict a set of frequencies based on the consonance level of
        specific peak frequencies.

        Parameters
        ----------
        peaks : List (float)
            List of frequency peaks.
        n_harm: int
            Defaults to 10.
            Set the number of harmonics to compute in harmonic_fit function
        method: str
            {'harmonic_fit', 'consonant', 'multi_consonant',
            'consonant_harmonic_fit', 'multi_consonant_harmonic_fit'}
        harm_function: str
            {'mult' or 'div'}
            Defaults to 'mult'
            Computes harmonics from iterative multiplication (x, 2x, 3x, ...nx)
            or division (x, x/2, x/3, ...x/n).
        div_mode : str
            {'div', 'div_add', 'div_sub'}
            'div': x, x/2, x/3 ..., x/n
            'div_add': x, (x+x/2), (x+x/3), ... (x+x/n)
            'div_sub': x, (x-x/2), (x-x/3), ... (x-x/n)
        cons_limit : type
            Description of parameter `cons_limit`.
        ratios_extension : Boolean
            Defaults to False.
            If is True, ratios_extensions are computed accordingly to what
            was defined in __init__.
        harm_bounds : float
            Defaults to 0.1
            Maximal distance in Hertz between two frequencies to consider
            them as equivalent.
        scale_cons_limit : float
            Defaults to 0.1 (__init__)
            Minimal value of consonance to be reach for a peaks ratio
            to be included in the extended_peaks_ratios_cons attribute.

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

        Other Attributes
        ----------------
        ----------If ratios_extension is True:----------
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
        if method == 'harmonic_fit':
            (extended_peaks,
             harmonics,
             _, _) = harmonic_fit(peaks, n_harm,
                                  function=harm_function,
                                  div_mode=div_mode,
                                  bounds=harm_bounds)
            self.extended_peaks = np.sort(list(self.peaks)
                                          + list(set(extended_peaks)))
        if method == 'consonant':
            (consonance,
             cons_pairs,
             cons_peaks,
             cons_metric) = consonance_peaks(peaks, limit=cons_limit)
            self.extended_peaks = np.sort(np.round(cons_peaks, 3))
        if method == 'multi_consonant':
            (consonance,
             cons_pairs,
             cons_peaks,
             cons_metric) = consonance_peaks(peaks, limit=cons_limit)
            extended_temp = multi_consonance(cons_pairs, n_freqs=10)
            self.extended_peaks = np.sort(np.round(extended_temp, 3))
        if method == 'consonant_harmonic_fit':
            (extended_peaks,
             harmonics,
             _, _) = harmonic_fit(peaks, n_harm,
                                  function=harm_function,
                                  div_mode=div_mode,
                                  bounds=harm_bounds)
            (consonance,
             cons_pairs,
             cons_peaks,
             cons_metric) = consonance_peaks(extended_peaks, limit=cons_limit)
            self.extended_peaks = np.sort(np.round(cons_peaks, 3))
        if method == 'multi_consonant_harmonic_fit':
            (extended_peaks,
             harmonics, _, _) = harmonic_fit(peaks, n_harm,
                                             function=harm_function,
                                             div_mode=div_mode,
                                             bounds=harm_bounds)
            (consonance,
             cons_pairs,
             cons_peaks,
             cons_metric) = consonance_peaks(extended_peaks, limit=cons_limit)
            extended_temp = multi_consonance(cons_pairs, n_freqs=10)
            self.extended_peaks = np.sort(np.round(extended_temp, 3))
        self.extended_peaks = [i for i in self.extended_peaks if i < self.sf/2]
        self.extended_amps = peaks_to_amps(self.extended_peaks, self.freqs,
                                           self.psd, self.sf)
        print('Number of extended peaks : ', len(self.extended_peaks))
        if len(self.extended_peaks) > 0:
            ext_peaks_rat = compute_peak_ratios(self.extended_peaks,
                                                rebound=True)
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
            (self.extended_peaks_ratios_cons,
             b) = consonant_ratios(
                                   self.extended_peaks,
                                   scale_cons_limit, sub=False)
        return (self.extended_peaks,
                self.extended_amps,
                self.extended_peaks_ratios)

    def ratios_extension(self, ratios, ratio_fit_bounds=0.001,
                         ratios_n_harms=None):
        """
        This method takes a series of ratios as input and returns the
        harmonics (ratio*2, ratio*3, ..., ratio*n) or the increments
        (ratio**2, ratio**3, ..., ratio**n).

        Parameters
        ----------
        ratios : List (float)
            List of frequency ratios.
        ratio_fit_bounds : float
            Defaults to 0.001.
            Minimal distance between two ratios to consider a fit
            for the harmonic_fit function.
        ratios_n_harms : int
            Number of harmonics or increments to compute.

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
            (ratios_inc_fit_,
             ratios_inc_fit_pos,
             _, _) = harmonic_fit(ratios,
                                  ratios_n_harms,
                                  function='exp',
                                  bounds=ratio_fit_bounds)
        else:
            ratios_inc_fit_ = None
        return ratios_harms_, ratios_inc_, ratios_inc_fit_

    def compute_spectromorph(self, IMFs=None, sf=None,
                             method='SpectralCentroid', window=None,
                             overlap=1, comp_chords=False, min_notes=3,
                             cons_limit=0.2, cons_chord_method='cons',
                             graph=False):
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
        method : str
            {'SpectralCentroid',
             'SpectralCrestFactor',
             'SpectralDecrease',
             'SpectralFlatness',
             'SpectralFlux',
             'SpectralKurtosis',
             'SpectralMfccs',
             'SpectralPitchChroma',
             'SpectralRolloff',
             'SpectralSkewness',
             'SpectralSlope',
             'SpectralSpread',
             'SpectralTonalPowerRatio',
             'TimeAcfCoeff',
             'TimeMaxAcf',
             'TimePeakEnvelope',
             'TimeRms',
             'TimeStd',
             'TimeZeroCrossingRate',}
            Defaults to 'SpectralCentroid'
            Spectromorphological metric to compute.
        window : int
            Window size.
        overlap : int
            Value of the overlap between successive windows.
        comp_chords : Boolean
            Defaults to False.
            When set to True, consonant spectral chords are computed.
        min_notes : int
            Defaults to 3.
            Minimum number of consonant values to store a spectral chord.
        cons_limit : float
            Minimal value of consonance.
        cons_chord_method : str
            {'consonance', 'euler'}
            Defaults. to 'consonance'.
            Metrics to use for consonance computation.
        graph : Boolean
            Defaults to False.
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
            if self.peaks_function == 'EEMD' or self.peaks_function == 'EMD':
                IMFs = self.IMFs
            else:
                IMFs = EMD_eeg(self.data, method='EMD')[1:6]
                self.IMFs = IMFs
        if sf is None:
            sf = self.sf
        if window is None:
            window = int(sf/2)

        spectro_EMD = EMD_to_spectromorph(IMFs, sf, method=method,
                                          window=window, overlap=overlap)
        if comp_chords is True:
            (self.spectro_chords,
             spectro_chord_pos) = timepoint_consonance(spectro_EMD,
                                                       method=cons_chord_method,
                                                       limit=cons_limit,
                                                       min_notes=min_notes,
                                                       graph=graph)
        self.spectro_EMD = spectro_EMD
        if graph is True:
            plt.clf()
            if comp_chords is False:
                data = np.moveaxis(self.spectro_EMD, 0, 1)
                ax = sbn.lineplot(data=data[10:-10, :], dashes=False)
                ax.set(xlabel='Time Windows', ylabel=method)
                if method == 'SpectralCentroid':
                    ax.set_yscale('log')
                plt.legend(scatterpoints=1, frameon=True, labelspacing=1,
                           title='EMDs', loc='best',
                           labels=['EMD1', 'EMD2', 'EMD3', 'EMD4', 'EMD5'])
                plt.show()

    def compute_peaks_metrics(self,
                              n_harm=None,
                              harm_bounds=0.5,
                              delta_lim=20):
        """
        This function computes consonance metrics on peaks attribute.

        Parameters
        ----------
        n_harm : int
            Set the number of harmonics to compute in harmonic_fit function
        harm_bounds : type
            Defaults to 0.5
            Maximal distance in Hertz between two frequencies to consider
            them as equivalent.

        Attributes
        -------
        self.peaks_metrics : dict
            Dictionary with keys corresponding to the different metrics.
            {'cons', 'euler', 'tenney', 'harm_fit', 'harmsim'}

        """
        if n_harm is None:
            n_harm = self.n_harm

        peaks = list(self.peaks)
        peaks_ratios = compute_peak_ratios(peaks, rebound=True,
                                           octave=self.octave,
                                           sub=self.compute_sub_ratios)
        metrics = {'cons': 0, 'euler': 0, 'tenney': 0,
                   'harm_fit': 0, 'harmsim': 0}
        try:
            (harm_fit,
             harm_pos,
             common_harm_pos, _) = harmonic_fit(peaks,
                                                n_harm=n_harm,
                                                bounds=harm_bounds)
            metrics['harm_pos'] = harm_pos
            metrics['common_harm_pos'] = common_harm_pos
            metrics['harm_fit'] = len(harm_fit)
        except:
            pass
        a, b, c, metrics['cons'] = consonance_peaks(peaks, 0.1)
        peaks_euler = [int(round(num, 2)*1000) for num in peaks]

        spf = self.peaks_function
        if spf == 'fixed' or spf == 'adapt' or spf == 'EMD' or spf == 'EEMD':
            try:
                metrics['euler'] = euler(*peaks_euler)
            except:
                pass
        metrics['tenney'] = tenneyHeight(peaks)
        metrics['harmsim'] = np.average(ratios2harmsim(peaks_ratios))
        _, _, subharm, _ = compute_subharmonics_5notes(peaks, n_harm,
                                                       delta_lim, c=2.1)
        metrics['subharm_tension'] = subharm
        if spf == 'harmonic_peaks':
            metrics['n_harmonic_peaks'] = self.n_harmonic_peaks
        self.peaks_metrics = metrics

    '''Methods to compute scales from whether peaks or extended peaks'''

    def compute_diss_curve(self, input_type='peaks', denom=1000, max_ratio=2,
                           euler_comp=False, method='min',
                           plot=False, n_tet_grid=12, scale_cons_limit=None):
        """
        Compute dissonance curve based on peak frequencies.

        Parameters
        ----------
        input_type : str
            ['peaks', 'extended_peaks']
            Defines whether peaks or extended_peaks are used.
        denom : int
            Maximal value of the denominator when computing frequency ratios.
        max_ratio : float
            Value of the maximal frequency ratio to use when computing
            the dissonance curve. When set to 2, the curve spans one octave.
            When set to 4, the curve spans two octaves.
        euler_comp : Boolean
            Defaults to False.
            Defines if euler consonance is computed. Can be computationally
            expensive when the number of local minima is high.
        method : str
            {'min', 'product'}
            Defaults to 'min'.
            Refer to dissmeasure function in scale_construction.py
            for more information.
        plot : Boolean
            Defaults to False
            When set to True, dissonance curve is plotted.
        n_tet_grid : int
            Defines which N-TET tuning is indicated, as a reference,
            in red in the dissonance curve plot.
        scale_cons_limit : float
            Defaults to 0.1 (__init__)
            Minimal value of consonance to be reach for a peaks ratio
            to be included in the self.diss_scale_cons attribute.

        Attributes
        -------
        self.diss_scale : List (float)
            List of frequency ratios corresponding to local minima.
        self.diss_scale_cons : List (float)
            List of frequency ratios corresponding to consonant local minima.
        self.scale_metrics : dict
            {'diss_euler', 'dissonance', 'diss_harm_sim', diss_n_steps}
            Add 4 metrics related to the dissonance curve tuning

        """
        if input_type == 'peaks':
            peaks = self.peaks
            amps = self.amps
        if input_type == 'extended_peaks':
            peaks = self.extended_peaks
            amps = self.extended_amps
        if scale_cons_limit is None:
            scale_cons_limit = self.scale_cons_limit

        peaks = [p*128 for p in peaks]
        amps = np.interp(amps,
                         (np.array(amps).min(), np.array(amps).max()),
                         (0.2, 0.8))

        (intervals,
         self.diss_scale,
         euler_diss,
         diss,
         harm_sim_diss) = diss_curve(peaks, amps, denom=denom,
                                     max_ratio=max_ratio,
                                     euler_comp=euler_comp,
                                     method=method, plot=plot,
                                     n_tet_grid=n_tet_grid)
        self.diss_scale_cons, b = consonant_ratios(self.diss_scale,
                                                   scale_cons_limit,
                                                   sub=False,
                                                   input_type='ratios')
        self.scale_metrics['diss_euler'] = euler_diss
        self.scale_metrics['dissonance'] = diss
        self.scale_metrics['diss_harm_sim'] = np.average(harm_sim_diss)
        self.scale_metrics['diss_n_steps'] = len(self.diss_scale)

    def compute_harmonic_entropy(self, input_type='peaks', res=0.001,
                                 spread=0.01, plot_entropy=True,
                                 plot_tenney=False, octave=2,
                                 rebound=True, sub=False,
                                 scale_cons_limit=None):
        """
        Computes the harmonic entropy from a series of spectral peaks.
        Harmonic entropy has been introduced by Paul Elrich
        [http://www.tonalsoft.com/enc/e/erlich/harmonic-entropy_with-
        commentary.aspx]

        Parameters
        ----------
        input_type : str
            ['peaks', 'extended_peaks']
            Defines whether peaks or extended_peaks are used.
        res : float
            Defaults to 0.001
            resolution of the ratio steps.
        spread : float
            Description of parameter `spread`.
        plot_entropy : Boolean
            Defaults to True.
            When set to True, plot the harmonic entropy curve.
        plot_tenney : Boolean
            Defaults to False.
            When set to True, plot the tenney heights (y-axis)
            across ratios (x-axis).
        octave : int
            Defaults to 2.
            Value of the octave.
        rebound : Boolean
            Defaults to True.
            When set to True, peaks ratios are bounded within the octave.
        sub : Boolean
            Defaults to False.
            When set to True, will include ratios below the unison (1)
        scale_cons_limit : type
            Defaults to 0.1 (__init__)
            Minimal value of consonance to be reach for a peaks ratio
            to be included in the self.diss_scale_cons attribute.

        Attributes
        -------
        self.HE_scale : List (float)
            List of frequency ratios corresponding to local minima.
        self.HE_scale_cons : List (float)
            List of frequency ratios corresponding to consonant local minima.
        self.scale_metrics : dict
            {'HE', 'HE_n_steps', 'HE_harm_sim'}
            Add 4 metrics related to the dissonance curve tuning

        """
        if input_type == 'peaks':
            ratios = compute_peak_ratios(self.peaks, rebound=rebound, sub=sub)
        if input_type == 'extended_peaks':
            ratios = compute_peak_ratios(self.extended_peaks, rebound=rebound,
                                         sub=sub)
        if input_type == 'extended_ratios_harms':
            ratios = self.extended_peaks_ratios_harms
        if input_type == 'extended_ratios_inc':
            ratios = self.extended_peaks_ratios_inc
        if input_type == 'extended_ratios_inc_fit':
            ratios = self.extended_peaks_ratios_inc_fit
        if scale_cons_limit is None:
            scale_cons_limit = self.scale_cons_limit

        HE_scale, HE = harmonic_entropy(ratios, res=res, spread=spread,
                                        plot_entropy=plot_entropy,
                                        plot_tenney=plot_tenney, octave=octave)
        self.HE_scale = HE_scale[0]
        self.HE_scale_cons, b = consonant_ratios(self.HE_scale,
                                                 scale_cons_limit, sub=False,
                                                 input_type='ratios')
        self.scale_metrics['HE'] = HE
        self.scale_metrics['HE_n_steps'] = len(self.HE_scale)
        HE_harm_sim = np.average(ratios2harmsim(list(self.HE_scale)))
        self.scale_metrics['HE_harm_sim'] = HE_harm_sim

    def euler_fokker_scale(self, method='peaks', octave=2):
        """
        Create a scale in the Euler-Fokker Genera. which is a
        musical scale in just intonation whose pitches can be
        expressed as products of some of the members of some multiset
        of generating primer numbers.

        Parameters
        ----------
        method : str
            {'peaks', 'extended_peaks'}
            Defaults to 'peaks'.
            Defines which set of frequencies are used.
        octave : float
            Value of period interval.

        Returns
        -------
        self.euler_fokker : List (float)
            Euler-Fokker genera.

        """
        if method == 'peaks':
            intervals = self.peaks
        if method == 'extended_peaks':
            intervals = self.extended_peaks
        intervals = prime_factor([int(x) for x in intervals])
        multiplicities = [1 for x in intervals]  # Each factor is used once.
        scale = create_euler_fokker_scale(intervals,
                                          multiplicities,
                                          octave=octave)
        self.euler_fokker = scale
        return scale

    def harmonic_tuning(self, list_harmonics, octave=2, min_ratio=1,
                        max_ratio=2):
        """
        Generates a tuning based on a list of harmonic positions.

        Parameters
        ----------
        list_harmonics: List (int)
            harmonic positions to use in the scale construction
        octave: int
            value of the period reference
        min_ratio: float
            Defaults to 1.
            Value of the unison.
        max_ratio: float
            Defaults to 2.
            Value of the octave.

        Returns
        -------
        ratios : List (float)
            Generated tuning.

        """
        ratios = []
        for i in list_harmonics:
            ratios.append(rebound(1*i, min_ratio, max_ratio, octave))
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
        bounds : float
            Defaults to 0.1
            Maximal distance in Hertz between two frequencies to consider
            them as equivalent.
        n_common_harms : int
            minimum number of times the harmonic is found
            to be sent to most_common_harmonics output.

        Returns
        -------
        self.harmonic_fit_tuning : List (float)
            Generated tuning

        """

        _, harmonics, common_h, _ = harmonic_fit(self.peaks,
                                                 n_harm=n_harm,
                                                 bounds=bounds,
                                                 n_common_harms=n_common_harms)
        self.harmonic_fit_tuning = harmonic_tuning(common_h)
        return self.harmonic_fit_tuning

    def pac(self, sf=None, method='duprelatour', n_values=10,
            drive_precision=0.05, max_drive_freq=6, min_drive_freq=3,
            sig_precision=1, max_sig_freq=50, min_sig_freq=8,
            low_fq_width=0.5, high_fq_width=1, plot=False):
        """
        Computes the phase-amplitude coupling and returns to pairs of
        frequencies that have highest coupling value.

        Parameters
        ----------
        sf : int
            Sampling frequency in hertz.
        method : str
            Defaults to 'duprelatour'.
            Choice of method for PAC calculation.
            STANDARD_PAC_METRICS = ['ozkurt', 'canolty', 'tort', 'penny',
                                    'vanwijk']
            DAR_BASED_PAC_METRICS = ['duprelatour']
            COHERENCE_PAC_METRICS = ['jiang', 'colgin']
            BICOHERENCE_PAC_METRICS = ['sigl', 'nagashima', 'hagihira',
                                       'bispectrum']
        n_values : int
            Defaults to 10.
            Number of pairs of frequencies to return.
        drive_precision : float
            Step-size between each phase signal bins.
        max_drive_freq : float
            Maximum value of the phase signal in hertz.
        min_drive_freq : float
            Minimum value of the phase signal in hertz.
        sig_precision : float
            Step-size between each amplitude signal bins.
        max_sig_freq : float
            Maximum value of the amplitude signal in hertz.
        min_sig_freq : float
            Minimum value of the amplitude signal in hertz.
        low_fq_width : float
            Bandwidth of the band-pass filter (phase signal)
        high_fq_width : float
            Bandwidth of the band-pass filter (amplitude signal)
        plot : Boolean
            Defaults to false.
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
        freqs, pac_coupling = pac_frequencies(self.data, sf, method=method,
                                              n_values=n_values,
                                              drive_precision=drive_precision,
                                              max_drive_freq=max_drive_freq,
                                              min_drive_freq=min_drive_freq,
                                              sig_precision=sig_precision,
                                              max_sig_freq=max_sig_freq,
                                              min_sig_freq=min_sig_freq,
                                              low_fq_width=low_fq_width,
                                              high_fq_width=high_fq_width,
                                              plot=plot)
        self.pac_freqs = freqs
        self.pac_coupling = pac_coupling
        return self.pac_freqs, self.pac_coupling

    def compute_peaks_ts(self, data, peaks_function='EMD', FREQ_BANDS=None,
                         precision=0.25, sf=None, min_freq=1, max_freq=80,
                         min_harms=2, harm_limit=128, n_peaks=5, nIMFs=None,
                         graph=False, noverlap=None, average='median',
                         nfft=None, nperseg=None, max_harm_freq=None,
                         EIMC_order=3, min_IMs=2):
        """
        Extract peak frequencies. This method is called by the
        peaks_extraction method.

        Parameters
        ----------
        data: array (numDataPoints,)
            biosignal to analyse
        peaks_function: str
            refer to __init__
        FREQ_BANDS: List of lists (float)
            Each list within the list of lists sets the lower and
            upper limit of a frequency band
        precision: float
            Defaults to None
            precision of the peaks (in Hz)
            When HH1D_max is used, bins are in log scale.
        sf : type
            Description of parameter `sf`.
        min_freq: float
            Defaults to 1
            minimum frequency value to be considered as a peak
            Used with 'harmonic_peaks' and 'HH1D_max' peaks functions
        max_freq: float
            Defaults to 60
            maximum frequency value to be considered as a peak
            Used with 'harmonic_peaks' and 'HH1D_max' peaks functions
        min_harms : type
            Description of parameter `min_harms`.
        harm_limit: int
            Defaults to 128
            maximum harmonic position for 'harmonic_peaks' method.
        n_peaks: int
            Defaults to 5
            number of peaks when using 'FOOOF' and 'cepstrum',
            and 'harmonic_peaks' functions.
            Peaks are chosen based on their amplitude.
        nIMFs: int
            Defaults to 5
            number of intrinsic mode functions to keep when using
            'EEMD' or 'EMD' peaks function.
        graph: boolean
            Defaults to False
            when set to True, a graph will accompanies the peak extraction
            method (except for 'fixed' and 'adapt').
        noverlap : int
            Defaults to None.
            Number of samples overlap between each fft window.
            When set to None, equals sf//10.
        average : str
            Defaults to 'median'.
            {'mean', 'median'}
            Method to use when averaging periodograms.
        max_harm_freq : int
            Maximum frequency value of the find peaks function
            when harmonic_peaks or EIMC peaks extraction method is used.
        EIMC_order : int
            Maximum order of the Intermodulation Components.
        min_IMs : int
            Minimal number of Intermodulation Components to select the
            associated pair of peaks.

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
            harmonic_peaks method is used.

        Returns
        -------
        peaks : List (float)
            List of peaks frequencies.
        amps : List (float)
            List of amplitudes associated with peaks frequencies.
        """
        alphaband = [[7, 12]]
        if sf is None:
            sf = self.sf
        if nIMFs is None:
            nIMFs = self.nIMFs
        if FREQ_BANDS is None:
            FREQ_BANDS = [[2, 3.55], [3.55, 7.15], [7.15, 14.3],
                          [14.3, 28.55], [28.55, 49.4]]
        if FREQ_BANDS is not None:
            FREQ_BANDS = FREQ_BANDS
        if max_harm_freq is None:
            max_harm_freq = sf/2

        if peaks_function == 'adapt':
            p, a = extract_welch_peaks(data, sf=sf, FREQ_BANDS=alphaband,
                                       out_type='bands', precision=precision,
                                       average=average, extended_returns=False,
                                       nperseg=nperseg, noverlap=noverlap,
                                       nfft=nfft)
            FREQ_BANDS = alpha2bands(p[0])
            (peaks_temp,
             amps_temp,
             self.freqs,
             self.psd) = extract_welch_peaks(data, sf=sf,
                                             FREQ_BANDS=FREQ_BANDS,
                                             out_type='bands',
                                             precision=precision,
                                             average=average,
                                             extended_returns=True,
                                             nperseg=nperseg,
                                             noverlap=noverlap,
                                             nfft=nfft)
            if graph is True:
                graph_psd_peaks(self.freqs, self.psd, peaks_temp,
                                xmin=min_freq, xmax=max_freq,
                                color='darkblue', method=peaks_function)
        if peaks_function == 'fixed':
            (peaks_temp,
             amps_temp,
             self.freqs,
             self.psd) = extract_welch_peaks(data, sf=sf,
                                             FREQ_BANDS=FREQ_BANDS,
                                             out_type='bands',
                                             precision=precision,
                                             average=average,
                                             extended_returns=True,
                                             nperseg=nperseg,
                                             noverlap=noverlap,
                                             nfft=nfft)
            if graph is True:
                graph_psd_peaks(self.freqs, self.psd, peaks_temp,
                                xmin=min_freq, xmax=max_freq,
                                color='darkred', method=peaks_function,
                                precision=precision)
        if peaks_function == 'FOOOF':
            (peaks_temp,
             amps_temp,
             self.freqs,
             self.psd) = compute_FOOOF(data, sf, precision=precision,
                                       max_freq=max_freq, noverlap=None,
                                       n_peaks=n_peaks, extended_returns=True,
                                       graph=graph)

        if (
            peaks_function == 'EMD' or
            peaks_function == 'EEMD' or
            peaks_function == 'EMD_fast' or
            peaks_function == 'EEMD_fast'
             ):
            IMFs = EMD_eeg(data, method=peaks_function, graph=graph,
                           extrema_detection='simple')
            self.IMFs = IMFs[1:nIMFs+1]
            IMFs = IMFs[1:nIMFs+1]
            try:
                peaks_temp = []
                amps_temp = []
                freqs_all = []
                psd_all = []
                for imf in range(len(IMFs)):
                    (p,
                     a,
                     freqs,
                     psd) = extract_welch_peaks(IMFs[imf], sf=sf,
                                                precision=precision,
                                                average=average,
                                                extended_returns=True,
                                                out_type='single',
                                                nperseg=nperseg,
                                                noverlap=noverlap,
                                                nfft=nfft)

                    freqs_all.append(freqs)
                    psd_all.append(psd)
                    peaks_temp.append(p)
                    amps_temp.append(a)
                peaks_temp = np.flip(peaks_temp)
                amps_temp = np.flip(amps_temp)
            except:
                pass
            if graph is True:
                graphEMD_welch(freqs_all, psd_all, peaks=peaks_temp,
                               raw_data=self.data,
                               FREQ_BANDS=FREQ_BANDS,
                               sf=sf, nfft=nfft, nperseg=nperseg,
                               noverlap=noverlap, min_freq=min_freq,
                               max_freq=max_freq)

        if peaks_function == 'EEMD_FOOOF':
            nfft = sf/precision
            nperseg = sf/precision
            IMFs = EMD_eeg(data, method=peaks_function, graph=graph,
                           extrema_detection='simple')[1:nIMFs+1]
            self.IMFs = IMFs
            peaks_temp = []
            amps_temp = []
            for imf in IMFs:
                freqs1, psd = scipy.signal.welch(imf, sf, nfft=nfft,
                                                 nperseg=nperseg,
                                                 noverlap=noverlap)
                self.freqs = freqs1
                self.psd = psd
                fm = FOOOF(peak_width_limits=[precision*2, 3], max_n_peaks=50,
                           min_peak_height=0.2)
                freq_range = [(sf/len(data))*2, max_freq]
                fm.fit(freqs1, psd, freq_range)
                if graph is True:
                    fm.report(freqs1, psd, freq_range)
                peaks_temp_EMD = fm.peak_params_[:, 0]
                amps_temp_EMD = fm.peak_params_[:, 1]
                try:
                    # Select the peak with highest amplitude.
                    peaks_temp.append([x for _, x in
                                       sorted(zip(amps_temp_EMD,
                                                  peaks_temp_EMD))][::-1][0:1])
                    amps_temp.append(sorted(amps_temp_EMD)[::-1][0:1])
                except:
                    print('No peaks detected')
            peaks_temp = [np.round(p, 2) for p in peaks_temp]
            peaks_temp = [item for sublist in peaks_temp for item in sublist]
            amps_temp = [item for sublist in amps_temp for item in sublist]

        if peaks_function == 'HH1D_max':
            (IF,
             peaks_temp,
             amps_temp,
             HH_spec,
             HH_bins) = HilbertHuang1D(data, sf, graph=graph,
                                       nIMFs=nIMFs,
                                       min_freq=min_freq,
                                       max_freq=max_freq,
                                       precision=precision,
                                       bin_spread='log')
            self.IF = IF
        # if peaks_function == 'HH1D_weightAVG':
        # if peaks_function == 'HH1D_FOOOF':
        if peaks_function == 'bicoherence':
            freqs, amps = polyspectrum_frequencies(data, sf, precision,
                                                   n_values=n_peaks,
                                                   nperseg=nperseg,
                                                   noverlap=noverlap,
                                                   method=peaks_function,
                                                   flim1=(min_freq, max_freq),
                                                   flim2=(min_freq, max_freq),
                                                   graph=graph)
            common_freqs = flatten(pairs_most_frequent(freqs, n_peaks))
            peaks_temp = list(np.sort(list(set(common_freqs))))
            peaks_temp = [p for p in peaks_temp if p < max_freq]
            amp_idx = []
            for i in peaks_temp:
                amp_idx.append(flatten(freqs).index(i))
            amps_temp = np.array(flatten(amps))[amp_idx]
            amps_temp = list(amps_temp)
            # Select the n peaks with highest amplitude.
            peaks_temp.append([x for _, x in
                               sorted(zip(amps_temp,
                                          peaks_temp))][::-1][0:n_peaks])
            amps_temp = sorted(amps_temp)[::-1][0:n_peaks]
        if peaks_function == 'harmonic_peaks':
            (p,
             a,
             self.freqs,
             self.psd) = extract_welch_peaks(data, sf,
                                             precision=precision,
                                             max_freq=max_harm_freq,
                                             extended_returns=True,
                                             out_type='all',
                                             nperseg=nperseg,
                                             noverlap=noverlap,
                                             nfft=nfft, min_freq=min_freq)
            (max_n,
             peaks_temp,
             amps_temp,
             harms,
             harm_peaks,
             harm_peaks_fit) = harmonic_recurrence(p, a, min_freq,
                                                   max_freq,
                                                   min_harms=min_harms,
                                                   harm_limit=harm_limit)
            list_harmonics = np.concatenate(harms)
            list_harmonics = list(set(abs(np.array(list_harmonics))))
            list_harmonics = [h for h in list_harmonics if h <= harm_limit]
            list_harmonics = np.sort(list_harmonics)
            self.all_harmonics = list_harmonics
            self.harm_peaks_fit = harm_peaks_fit
            # Select the n peaks with highest amplitude.
            peaks_temp.append([x for _, x in
                               sorted(zip(amps_temp,
                                          peaks_temp))][::-1][0:n_peaks])
            amps_temp = sorted(amps_temp)[::-1][0:n_peaks]
            if graph is True:
                graph_psd_peaks(self.freqs, self.psd, peaks_temp,
                                xmin=min_freq, xmax=max_freq,
                                color='lightseagreen', method=peaks_function)
        if peaks_function == 'EIMC':
            (p,
             a,
             self.freqs,
             self.psd) = extract_welch_peaks(data, sf,
                                             precision=precision,
                                             max_freq=max_harm_freq,
                                             extended_returns=True,
                                             out_type='all',
                                             nperseg=nperseg,
                                             noverlap=noverlap,
                                             nfft=nfft, min_freq=min_freq)
            (IMC,
             self.EIMC_all,
             n) = endogenous_intermodulations(p, a,
                                              order=EIMC_order,
                                              min_IMs=min_IMs)
            IMC_freq = pairs_most_frequent(self.EIMC_all['peaks'], n_peaks)
            common_freqs = flatten(IMC_freq)
            peaks_temp = list(np.sort(list(set(common_freqs))))
            peaks_temp = [p for p in peaks_temp if p < max_freq]
            amp_idx = []
            for i in peaks_temp:
                amp_idx.append(flatten(self.EIMC_all['peaks']).index(i))
            amps_temp = np.array(flatten(self.EIMC_all['amps']))[amp_idx]
            amps_temp = list(amps_temp)
            peaks_temp.append([x for _, x in
                               sorted(zip(amps_temp,
                                          peaks_temp))][::-1][0:n_peaks])
            amps_temp = sorted(amps_temp)[::-1][0:n_peaks]
            if graph is True:
                graph_psd_peaks(self.freqs, self.psd, peaks_temp,
                                xmin=min_freq, xmax=max_freq,
                                color='darkgoldenrod', method=peaks_function)
        if peaks_function == 'PAC':
            freqs, amps = self.pac(sf=sf, method='duprelatour',
                                   n_values=n_peaks,
                                   drive_precision=precision,
                                   max_drive_freq=max_freq/2,
                                   min_drive_freq=min_freq,
                                   sig_precision=precision*2,
                                   max_sig_freq=max_freq,
                                   min_sig_freq=min_freq*2,
                                   low_fq_width=0.5, high_fq_width=1,
                                   plot=graph)
            common_freqs = flatten(pairs_most_frequent(freqs, n_peaks))
            peaks_temp = list(np.sort(list(set(common_freqs))))
            peaks_temp = [p for p in peaks_temp if p < max_freq]
            amp_idx = []
            for i in peaks_temp:
                amp_idx.append(flatten(freqs).index(i))
            amps_temp = np.array(flatten(amps))[amp_idx]
            amps_temp = list(amps_temp)
            peaks_temp.append([x for _, x in
                               sorted(zip(amps_temp,
                                          peaks_temp))][::-1][0:n_peaks])
            amps_temp = sorted(amps_temp)[::-1][0:n_peaks]
        if peaks_function == 'cepstrum':
            cepstrum_, quefrency_vector = cepstrum(self.data, self.sf,
                                                   min_freq=min_freq,
                                                   max_freq=max_freq,
                                                   plot_cepstrum=graph)
            peaks_temp_, amps_temp_ = cepstral_peaks(cepstrum_,
                                                     quefrency_vector,
                                                     1/min_freq,
                                                     1/max_freq)
            peaks_temp_ = list(np.flip(peaks_temp_))
            peaks_temp_ = [np.round(p, 2) for p in peaks_temp_]
            amps_temp_ = list(np.flip(amps_temp_))
            peaks_temp.append([x for _, x in
                               sorted(zip(amps_temp,
                                          peaks_temp))][::-1][0:n_peaks])
            amps_temp = sorted(amps_temp_)[::-1][0:n_peaks]

        peaks_temp = [0+precision if x == 0 else x for x in peaks_temp]
        peaks = np.array(peaks_temp)
        peaks = np.around(peaks, 3)
        amps = np.array(amps_temp)
        return peaks, amps

    '''Listening methods'''

    def listen_scale(self, scale, fund=250, length=500):
        if scale == 'peaks':
            scale = self.peaks_ratios
        if scale == 'diss':
            try:
                scale = self.diss_scale
            except:
                print('No Dissonance Curve scale available')
                pass
        if scale == 'HE':
            try:
                scale = list(self.HE_scale)
            except:
                print('No Harmonic Entropy scale available')
                pass
        scale = np.around(scale, 3)
        print('Scale:', scale)
        scale = list(scale)
        scale = [1]+scale
        for s in scale:
            freq = fund*s
            note = make_chord(freq, [1])
            note = np.ascontiguousarray(np.vstack([note, note]).T)
            sound = pygame.sndarray.make_sound(note)
            sound.play(loops=0, maxtime=0, fade_ms=0)
            pygame.time.wait(int(sound.get_length() * length))

    '''Generic method to fit all Biotuner methods'''

    def fit_all(self, data, compute_diss=True, compute_HE=True,
                compute_peaks_extension=True):
        biotuning = compute_biotuner(self.sf,
                                     peaks_function=self.peaks_function,
                                     precision=self.precision,
                                     n_harm=self.n_harm)
        biotuning.peaks_extraction(data)
        biotuning.compute_peaks_metrics()
        if compute_diss is True:
            biotuning.compute_diss_curve(input_type='peaks', plot=False)
        if compute_peaks_extension is True:
            biotuning.peaks_extension(method='multi_consonant_harmonic_fit',
                                      harm_function='mult', cons_limit=0.01)
        if compute_HE is True:
            biotuning.compute_harmonic_entropy(input_type='extended_peaks',
                                               plot_entropy=False)
        return biotuning

    def info(self, metrics=False, scales=False, whatever=False):
        if metrics is True:
            print('METRICS')
            print(vars(self))

        else:
            print(vars(self))
        return
