from biotuner.biotuner_object import compute_biotuner
from biotuner.biotuner_utils import chunk_ts
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from biotuner.metrics import compute_subharmonics_2lists
import numpy as np

class transitional_harmony(object):
    """
    Class used to compute subharmonic progressions
    in successive windows of time.
    """

    def __init__(
        self,
        sf=None,
        data=None,
        peaks_function="EMD",
        precision=0.1,
        n_harm=10,
        max_harm_freq=150,
        harm_function="mult",
        min_freq=2,
        max_freq=80,
        n_peaks=5,
        n_trans_harm=10,
        mode="win_overlap",
        overlap=10,
        FREQ_BANDS=None,
    ):
        """
        Parameters
        ----------
        sf: int
            sampling frequency (in Hz)
        data : array (numDataPoints,)
            Time series to analyse.
        peaks_function: str, default='EMD'
            See compute_biotuner class for details.
        precision: float, default=0.1
            Precision of the peaks (in Hz)
            When HH1D_max is used, bins are in log scale.
            Also used to divide the time series in successive windows.
            ChunkSizes = (1/precision)*sf
        n_harm: int, default=10
            Set the number of harmonics to compute in harmonic_fit function and peaks_extraction function
        max_harm_freq: int, default=150
            Set the maximum frequency to consider in harmonic_fit function and peaks_extraction function
        harm_function: str, default='mult'
            Computes harmonics from iterative multiplication (x, 2x, 3x, ...nx)
            or division (x, x/2, x/3, ...x/n).
            choice:
                - 'mult' : iterative multiplication\n
                - 'div' : iterative division\n
        min_freq: int, default=2
            Minimum frequency to consider (in Hz)
        max_freq: int, default=80
            Maximum frequency to consider (in Hz)
        n_peaks: int, default=5
            Number of peaks to extract in the peaks_extraction function.
        mode : str, default='win_overlap'
            The method used to chunk the time series.
            choice:
                - 'win_overlap' : divide the time series in successive windows\n
                - 'IF' : uses the instantaneous frequency using Hilbert-Huang transform\n
        overlap : int, default=10
            The overlap between successive windows, in number of samples.
        FREQ_BANDS : list of list of int
            The frequency bands used in the 'fixed' peaks_function.

        """
        if type(data) is not None:
            self.data = data
        self.sf = sf
        self.peaks_function = peaks_function
        self.precision = precision
        self.n_harm = n_harm
        self.n_trans_harm = n_trans_harm
        self.harm_function = harm_function
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.n_peaks = n_peaks
        self.mode = mode
        self.overlap = overlap
        self.max_harm_freq = max_harm_freq
        self.FREQ_BANDS = FREQ_BANDS

    def compute_trans_harmony(self, mode='win_overlap', overlap=10, delta_lim=20,
                             graph=False, save=False, savename='_', n_elements=2,
                             keep_first_IMF=False):
        
        
        """
        Computes the transitional subharmonic tension between successive windows of a time series.

        Parameters
        ----------
        mode : str, default='win_overlap
            The method used to chunk the time series.
            choice:
                - 'win_overlap' : divide the time series in successive windows\n
                - 'IF' : uses the instantaneous frequency using Hilbert-Huang transform\n
        overlap : int, default=10
            The overlap between successive windows, in percentage of the window size.
        delta_lim : int, default=20
            The maximum time difference allowed between the closest common subharmonic
            in two consecutive windows, in milliseconds.
        graph : bool, default=False
            Whether to plot the transitional subharmonic tension as a function of time.
        save : bool, default=False
            Whether to save the plot as a PNG file.
        savename : str, default='_'
            The filename of the saved plot, without the extension.
        keep_first_IMF : bool, default=False
            Whether to keep the first IMF in the EMD decomposition.
        n_elements : int, default=2
            TODO
        Returns
        -------
        trans_subharm : list of float
            The list of transitional subharmonic tension values for each pair of windows.
        time_vec_final : list of float
            The corresponding list of time values, corresponding to the middle of each pair of windows.
        subharm_melody : list of tuple of int
            The list of pairs of indices of the closest common subharmonic in each pair of windows.

        Notes
        -----
        This method chunks the time series using the specified mode and overlap parameters,
        and for each pair of consecutive windows, it extracts the subharmonic of spectral peaks.
        It then computes the closest common subharmonics between
        the two sets of peaks, and determines the transitional subharmonic tension.
        The resulting values are stored
        in the trans_subharm and time_vec_final lists, and optionally plotted and saved.

        """
        # Initialize biotuner object
        self.mode = mode
        self.overlap = overlap
        if mode == 'win_overlap':
            #print('hello')
            data = self.data
            pairs = chunk_ts(data, sf=self.sf, overlap=overlap, precision=self.precision)
            #print(pairs)
            peaks = []
            time_vec = []
            for pair in pairs:
                #try:
                #print(pair)
                data_ = data[pair[0]:pair[1]]
                biotuning = compute_biotuner(self.sf, peaks_function=self.peaks_function,
                                            precision=self.precision, n_harm=self.n_harm)
                biotuning.peaks_extraction(data_, min_freq=self.min_freq,
                                        max_freq=self.max_freq, max_harm_freq=self.max_harm_freq,
                                        n_peaks=self.n_peaks, noverlap=None,
                                        nperseg=None, nfft=None, smooth_fft=1, FREQ_BANDS=self.FREQ_BANDS,
                                        keep_first_IMF=keep_first_IMF)
                #print(biotuning.peaks)
                #print(biotuning.peaks)
                peaks.append(biotuning.peaks)
                time_vec.append(((pair[0]+pair[1])/2)/self.sf)
                #except UnboundLocalError:
                #    time_vec.append(((pair[0]+pair[1])/2)/self.sf)
                #    peaks.append([])
            trans_subharm = []
            i=1
            time_vec_final = []
            subharm_melody = []
            while i < len(peaks):
                list1 = peaks[i-1]
                list2 = peaks[i]
                try:
                    common_subs, delta_t, sub_tension_final, harm_temp, pairs_melody = compute_subharmonics_2lists(list1,
                                                                                                    list2,
                                                                                                    self.n_trans_harm,
                                                                                                    delta_lim=delta_lim,
                                                                                                    c=2.1)
                except:
                    sub_tension_final = np.nan
                    pairs_melody = np.nan
                trans_subharm.append(sub_tension_final)
                subharm_melody.append(pairs_melody)
                self.trans_subharm = trans_subharm
                time_vec_final.append((time_vec[i]+time_vec[i-1])/2)
                i=i+1


                self.time_vec = time_vec_final

        if mode == 'IF':
            bt = compute_biotuner(self.sf, peaks_function='HH1D_max',
                                                precision=self.precision, n_harm=self.n_harm)
            bt.peaks_extraction(self.data, min_freq=self.min_freq,
                                    max_freq=self.max_freq, max_harm_freq=150,
                                    n_peaks=self.n_peaks, noverlap=None,
                                    nperseg=None, nfft=None, smooth_fft=1)
            IFs = np.round(bt.IF, 2)
            trans_subharm = []
            subharm_melody = []
            for i in range(len(IFs)-1):
                list1 = IFs[i]
                list2 = IFs[i+1]    
                print('list1', list1)
                print('list2', list2)
                if all(v == 0 for v in list1) or all(v == 0 for v in list2) or any(v<0 for v in list1) or any(v<0 for v in list2):
                    print("One of the lists is only zeros, skipping computation...")
                    continue  # Skip to the next iteration
                a, b, c, d, pairs_melody = compute_subharmonics_2lists(list1, list2, n_harmonics=10, delta_lim=delta_lim, c=2.1)
                trans_subharm.append(c)
                subharm_melody.append(pairs_melody)
                
            time_vec_final = list(range(0, len(trans_subharm), 1))
            
        if graph is True:
            plt.clf()
            figure(figsize=(8, 5), dpi=300)

            plt.plot(time_vec_final, trans_subharm, color='black', label=str(delta_lim)+'ms')


            plt.legend(title='Maximum distance between \ncommon subharmonics')
            
            plt.ylabel('Transitional subharmonic tension')
            if mode == 'win_overlap':
                plt.xlim(0, len(self.data)/self.sf)
                plt.xlabel('Time (sec)')
            if mode == 'IF':
                plt.xlabel('Timepoints')
            if save is True:
                plt.savefig('Transitional_subharm_{}_delta_{}_overlap_{}.png'.format(mode, str(delta_lim), overlap, savename), dpi=300)
        return trans_subharm, time_vec_final, subharm_melody

    def compare_deltas(self, deltas, save=False, savename='_'):
        """
        Compares the transitional subharmonic tension for multiple maximum delta limits.

        Parameters
        ----------
        deltas : list of int
            The list of maximum time differences allowed between the closest
            common subharmonic in consecutive windows, in milliseconds.
        save : bool, default=False
            Whether to save the plot as a PNG file.
        savename : str, default='_'
            The filename of the saved plot, without the extension.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure showing the plotted data.

        Notes
        -----
        This method computes the transitional subharmonic tension for each maximum
        delta limit in the deltas list, using the compute_trans_harmony method.
        It then plots the resulting values as a function of time, using a different
        color for each delta value, and optionally saves the plot.

        """
        colors = ['darkorange', 'darkred', 'darkblue', 'darkcyan', 'goldenrod']
        
        plt.clf()
        fig = plt.figure(figsize=(8, 5), dpi=300)
        
        for d, c in zip(deltas, colors):
            sub, tvec, _ = self.compute_trans_harmony(mode=self.mode, overlap=self.overlap, delta_lim=d, graph=False, savename='_')
            plt.plot(tvec, sub, color=c, label=str(d)+'ms')

        plt.legend(title='Maximum distance between \ncommon subharmonics')
        plt.xlabel('Time (sec)')
        plt.ylabel('Transitional subharmonic tension')
        plt.xlim(0, len(self.data)/self.sf)
        
        if save is True:
            plt.savefig('Transitional_subharm_{}_delta_{}_overlap_{}.png'.format(self.mode, str((deltas[0], deltas[-1])), self.overlap, savename), dpi=300)
        
        return fig
    #def compute_trans_EMD():
    #    return
