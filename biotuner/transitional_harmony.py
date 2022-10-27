from biotuner.biotuner_object import compute_biotuner
#from biotuner.biotuner_utils import chunk_ts
from matplotlib.pyplot import figure


class transitional_harmony(object):
    """
    Class used to compute harmonic progressions
    in successive windows of time.
    """

    def __init__(
        self,
        sf=None,
        data=None,
        peaks_function="EMD",
        precision=1,
        n_harm=10,
        harm_function="mult",
        min_freq=2,
        max_freq=80,
        n_peaks=5,
        n_trans_harm=10
    ):
        """
        Parameters
        ----------
        sf: int
            sampling frequency (in Hz)
        data : array(numDataPoints,)
            Time series to analyse.
        peaks_function: str
            Defaults to 'EMD'.
            See compute_biotuner class for details.
        precision: float
            Defaults to 0.1
            precision of the peaks (in Hz)
            When HH1D_max is used, bins are in log scale.
        n_harm: int
            Defaults to 10.
            Set the number of harmonics to compute in harmonic_fit function
        harm_function: str
            {'mult' or 'div'}
            Defaults to 'mult'
            Computes harmonics from iterative multiplication (x, 2x, 3x, ...nx)
            or division (x, x/2, x/3, ...x/n).

        """
        """Initializing data"""
        if type(data) is not None:
            self.data = data
        self.sf = sf
        """Initializing arguments for peak extraction"""
        self.peaks_function = peaks_function
        self.precision = precision
        self.n_harm = n_harm
        self.n_trans_harm = n_trans_harm
        self.harm_function = harm_function
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.n_peaks = n_peaks

    def compute_trans_harmony(self, mode='win_overlap', overlap=50, delta_lim=20):
        # Initialize biotuner object

        if mode == 'win_overlap':
            data = self.data
            pairs = chunk_ts(data, sf=self.sf, overlap=overlap, precision=self.precision)
            peaks = []
            time_vec = []
            for pair in pairs:
                data_ = data[pair[0]:pair[1]]
                biotuning = compute_biotuner(self.sf, peaks_function=self.peaks_function,
                                             precision=self.precision, n_harm=self.n_harm)
                biotuning.peaks_extraction(data_, min_freq=self.min_freq,
                                           max_freq=self.max_freq, max_harm_freq=150,
                                           n_peaks=self.n_peaks, noverlap=None,
                                           nperseg=None, nfft=None)
                peaks.append(biotuning.peaks)
                time_vec.append(((pair[0]+pair[1])/2)/self.sf)
            trans_subharm = []
            i=1
            time_vec_final = []
            while i < len(peaks):
                list1 = peaks[i-1]
                list2 = peaks[i]
                common_subs, delta_t, sub_tension_final, harm_temp = compute_subharmonics_2lists(list1,
                                                                                                 list2,
                                                                                                 self.n_trans_harm,
                                                                                                 delta_lim=delta_lim,
                                                                                                 c=2.1)
                trans_subharm.append(sub_tension_final)
                self.trans_subharm = trans_subharm
                time_vec_final.append((time_vec[i]+time_vec[i-1])/2)
                i=i+1


                self.time_vec = time_vec_final
        return trans_subharm, time_vec_final



        def EMD_trans_harmony():
            a = a
            return
