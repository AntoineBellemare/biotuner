import numpy as np
from scipy.signal import detrend, welch, find_peaks, stft
from scipy.optimize import curve_fit
from biotuner.metrics import dyad_similarity, compute_subharmonic_tension, ratios2harmsim
from biotuner.biotuner_utils import compute_peak_ratios
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import pandas as pd
from scipy.signal import peak_prominences
from biotuner.metrics import spectral_flatness, spectral_entropy, spectral_spread, higuchi_fd, peaks_to_harmsim
from biotuner.biotuner_utils import safe_mean, safe_max, apply_power_law_remove, compute_frequency_and_psd
import seaborn as sns
from scipy.stats import pearsonr, chi2
from sklearn.preprocessing import MinMaxScaler
import matplotlib.lines as mlines
from scipy.stats import ttest_ind



def compute_phase_matrix(signal, precision_hz, fs, noverlap, smoothness):
    '''
    Compute the phase matrix of a signal using the Short-Time Fourier Transform (STFT).
    
    Parameters
    ----------
    signal : ndarray
        Input signal.
    precision_hz : float
        Frequency precision.
    fs : int
        The sampling frequency of the signal.
    noverlap : int
        The number of points of overlap between blocks.
    smoothness : float
        A parameter for smoothing the signal.
    
    Returns
    -------
    ndarray
        The phase matrix of the signal.
    '''
    nperseg = int(fs / precision_hz)
    _, _, Zxx = stft(signal, fs, nperseg=int(nperseg/smoothness), noverlap=noverlap)
    return np.angle(Zxx)


def compute_dyad_similarities_and_phase_coupling_matrix(freqs, phase_matrix, metric='harmsim', n_harms=5, delta_lim=150, min_notes=2):
    '''
    Compute dyad similarities and the phase coupling matrix of frequencies.
    The phase coupling metric is the Phase Locking Value.

    Parameters
    ----------
    freqs : ndarray
        Array of frequencies.
    phase_matrix : ndarray
        The phase matrix of the signal.
    metric : str, optional
        The metric to compute dyad similarity. Default is 'harmsim'.
    n_harms : int, optional
        The number of harmonics. Default is 5.
    delta_lim : int, optional
        The delta limit. Default is 150.
    min_notes : int, optional
        The minimum number of notes. Default is 2.

    Returns
    -------
    tuple of ndarrays
        The harmonicity and the phase coupling matrix.
    '''
    harmonicity = np.zeros((len(freqs), len(freqs)))
    phase_coupling_matrix = np.zeros((len(freqs), len(freqs)))

    for i, f1 in enumerate(freqs):
        for j, f2 in enumerate(freqs):
            if f2 != 0:
                if metric == 'harmsim':
                    harmonicity[i, j] = dyad_similarity(f1 / f2)
                if metric == 'subharm_tension':
                    _, _, subharm, _ = compute_subharmonic_tension([f1, f2], n_harmonics=n_harms, delta_lim=delta_lim, min_notes=min_notes)
                    harmonicity[i, j] = 1-subharm[0]

                phase_diff = np.abs(phase_matrix[i] - phase_matrix[j])
                phase_coupling_matrix[i, j] = np.abs(np.mean(np.exp(1j * phase_diff), axis=-1))
    return harmonicity, phase_coupling_matrix


def compute_harmonicity_values_and_phase_coupling_values(freqs, dyad_similarities, phase_coupling_matrix, psd_clean, psd_mode=None,
                                                         harm_phase_norm=True):
    '''
    Compute harmonicity values and phase coupling values for each frequency.

    Parameters
    ----------
    freqs : ndarray
        Array of frequencies.
    dyad_similarities : ndarray
        Dyad similarities matrix.
    phase_coupling_matrix : ndarray
        The phase coupling matrix.
    psd_clean : ndarray
        The cleaned Power Spectral Density (PSD).
    phase_matrix : ndarray
        The phase matrix of the signal.
    harm_phase_norm : bool, default=True
        If True, normalize the harmonicity and phase coupling values
        by dividing by the total power.

    Returns
    -------
    tuple of ndarrays
        The harmonicity values and the phase coupling values.
    '''
    harmonicity_values = np.zeros(len(freqs))
    phase_coupling_values = np.zeros(len(freqs))
    total_power = np.sum(psd_clean)

    for i in range(len(freqs)):
        weighted_sum_harmonicity = 0
        weighted_sum_phase_coupling = 0
        for j in range(len(freqs)):
            if i != j:
                weighted_sum_harmonicity += dyad_similarities[i, j] * (psd_clean[i] * psd_clean[j])
                if psd_mode == 'weighted':
                    weighted_sum_phase_coupling += phase_coupling_matrix[i, j] * (psd_clean[i] * psd_clean[j])
                else:
                    weighted_sum_phase_coupling += phase_coupling_matrix[i, j]
                
        if harm_phase_norm is True:
            harmonicity_values[i] = weighted_sum_harmonicity / (2 * total_power)
            phase_coupling_values[i] = weighted_sum_phase_coupling / (2 * total_power)
        if harm_phase_norm is False:        
            harmonicity_values[i] = weighted_sum_harmonicity
            phase_coupling_values[i] = weighted_sum_phase_coupling
    ##print(phase_coupling_values)
    return harmonicity_values, phase_coupling_values


def compute_resonance_values(harmonicity_values, phase_coupling_values):
    """
    Compute resonance values from harmonicity and phase coupling values.

    Parameters
    ----------
    harmonicity_values : ndarray
        Harmonicity values for each frequency.
    phase_coupling_values : ndarray
        Phase coupling values for each frequency.

    Returns
    -------
    ndarray
        Resonance values for each frequency.
    """
    # Normalize the harmonicity and phase coupling values
    normalized_harmonicity_values = (harmonicity_values - np.min(harmonicity_values)) / (np.max(harmonicity_values) - np.min(harmonicity_values))
    normalized_phase_coupling_values = (phase_coupling_values - np.min(phase_coupling_values)) / (np.max(phase_coupling_values) - np.min(phase_coupling_values))
        
    
    # Compute the resonance values
    resonance_values = normalized_harmonicity_values * normalized_phase_coupling_values
    #resonance_values = harmonicity_values * phase_coupling_values

    return resonance_values

def find_spectral_peaks(values, freqs, n_peaks, prominence_threshold=0.5):
    """
    Identify the prominent spectral peaks in a frequency spectrum.

    This function uses the peak prominence to select the most notable peaks,
    and returns their frequencies and indices. Prominence is a measure of how 
    much a peak stands out due to its intrinsic height and its location relative
    to other peaks.

    Parameters
    ----------
    values : array_like
        1-D array of values for the frequency spectrum.

    freqs : array_like
        1-D array of frequencies corresponding to the values in 'values'.

    n_peaks : int
        The number of top prominent peaks to return.

    prominence_threshold : float, default=0.5
        The minimum prominence a peak must have to be considered notable. 
        Peaks with a prominence less than this value will be ignored.

    Returns
    -------
    peak_frequencies : ndarray
        Frequencies of the 'n_peaks' most prominent peaks.

    prominent_peaks : ndarray
        Indices in 'values' and 'freqs' of the 'n_peaks' most prominent peaks.

    See Also
    --------
    scipy.signal.find_peaks
    scipy.signal.peak_prominences

    Examples
    --------
    >>> values = np.array([0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0])
    >>> freqs = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110])
    >>> find_spectral_peaks(values, freqs, n_peaks=3)
    (array([60, 40, 80]), array([5, 3, 7]))
    """
    # Find harmonicity peaks
    peaks, _ = find_peaks(values)
    prominences = peak_prominences(values, peaks)[0]
    
    # Filter peaks based on prominence threshold
    filtered_peak_indices = np.where(prominences > prominence_threshold)[0]
    
    # Select the top n_peaks from the filtered list
    sorted_peak_indices = prominences[filtered_peak_indices].argsort()[-n_peaks:][::-1]
    prominent_peaks = peaks[filtered_peak_indices[sorted_peak_indices]]
    peak_frequencies = freqs[prominent_peaks]

    return peak_frequencies, prominent_peaks


def harmonic_entropy(freqs, harmonicity_values, phase_coupling_values, resonance_values):
    """
    Compute spectral features and Higuchi Fractal Dimension of Harmonicity, Phase Coupling, and Resonance spectra.

    This function calculates several spectral properties: flatness, entropy, spread, and Higuchi Fractal Dimension 
    for three input spectra: Harmonicity, Phase Coupling, and Resonance. Results are returned as a pandas DataFrame.

    Parameters
    ----------
    freqs : array_like
        1-D array of frequencies common for all the spectra.

    harmonicity_values : array_like
        1-D array of spectral values for the Harmonicity spectrum.

    phase_coupling_values : array_like
        1-D array of spectral values for the Phase Coupling spectrum.

    resonance_values : array_like
        1-D array of spectral values for the Resonance spectrum.

    Returns
    -------
    harmonic_complexity : DataFrame
        A pandas DataFrame with spectral flatness, entropy, spread, and Higuchi Fractal Dimension
        for each of the Harmonicity, Phase Coupling, and Resonance spectra.

    See Also
    --------
    scipy.stats.mstats.gmean : Used to compute spectral flatness.
    scipy.stats.entropy : Used to compute spectral entropy.
    scipy.integrate.simps : Used to compute spectral spread.
    nolds.hfd : Used to compute Higuchi Fractal Dimension
    """

    # Measure entropies of the harmonic, phase coupling, and resonance spectra
    SpecFlat_harmonicity = spectral_flatness(harmonicity_values)
    SpecFlat_phase_coupling = spectral_flatness(phase_coupling_values)
    SpecFlat_resonance = spectral_flatness(resonance_values)
    
    SpecEnt_harmonicity = spectral_entropy(harmonicity_values)
    SpecEnt_phase_coupling = spectral_entropy(phase_coupling_values)
    SpecEnt_resonance = spectral_entropy(resonance_values)
    
    # Compute spectral spread
    SpecSpread_harmonicity = spectral_spread(freqs, harmonicity_values)
    SpecSpread_phase_coupling = spectral_spread(freqs, phase_coupling_values)
    SpecSpread_resonance = spectral_spread(freqs, resonance_values)
    
    # Compute higushi fractal dimension
    HFD_harmonicity = higuchi_fd(harmonicity_values, kmax=10)
    HFD_phase_coupling = higuchi_fd(phase_coupling_values, kmax=10)
    HFD_resonance = higuchi_fd(resonance_values, kmax=10)
    
    # Generate a dataframe with the spectral flatness and entropy values
    harmonic_complexity = pd.DataFrame({'Spectral Flatness': [SpecFlat_harmonicity, SpecFlat_phase_coupling, SpecFlat_resonance],
                          'Spectral Entropy': [SpecEnt_harmonicity, SpecEnt_phase_coupling, SpecEnt_resonance],
                          'Spectral Spread': [SpecSpread_harmonicity, SpecSpread_phase_coupling, SpecSpread_resonance],
                          'Higuchi Fractal Dimension': [HFD_harmonicity, HFD_phase_coupling, HFD_resonance]},
                            index=['Harmonicity', 'Phase Coupling', 'Resonance'], )

    return harmonic_complexity


def compute_global_harmonicity(signal, precision_hz, fmin=1, fmax=30, noverlap=1, fs=1000, power_law_remove=False,
                               n_peaks=5, metric='harmsim', n_harms=10, delta_lim=20, min_notes=2, plot=False, smoothness=1,
                               smoothness_harm=1, save=False, savename='', phase_mode=None, harm_phase_norm=True):
    """
    Compute global harmonicity, phase coupling, and resonance characteristics of a signal.

    This function computes the Power Spectral Density (PSD) of the signal, applies power law removal if required,
    calculates the phase matrix, computes dyad similarities and phase couplings, calculates harmonicity,
    phase coupling and resonance values, identifies spectral peaks, and returns a dataframe summarizing these metrics.

    Parameters
    ----------
    signal : array_like
        1-D input signal.
    precision_hz : float
        Frequency precision for computing spectra.
    fmin : float, default=1
        Minimum frequency to consider in the spectral analysis.
    fmax : float, default=30
        Maximum frequency to consider in the spectral analysis.
    noverlap : int, default=1
        Number of points of overlap between segments for PSD computation.
    fs : int, default=1000
        Sampling frequency.
    power_law_remove : bool, default=False
        If True, applies power law removal to the PSD.
    n_peaks : int, default=5
        Number of spectral peaks to identify.
    metric : str, default='harmsim'
        Method for computing dyad similarity.
        Options are:
            'harmsim' : Harmonic similarity
            'subharm_tension' : Subharmonic tension
    n_harms : int, default=10
        Number of harmonics to consider in dyad similarity computation.
    delta_lim : float, default=0.1
        Limit in ms used when metric is 'subharm_tension'.
    min_notes : int, default=2
        Minimum number of notes for dyad similarity computation.
    plot : bool, default=False
        If True, plots the resulting spectra.
    smoothness : int, default=1
        Smoothing factor applied to the PSD before computing spectra.
    smoothness_harm : int, default=1
        Smoothing factor applied to harmonicity values.
    save : bool, default=False
        If True, saves the plot as a .png file.
    savename : str, default=''
        Name for the saved plot file.
    phase_mode : str, default=None
        Method for weighting phase coupling computation. Options are 'weighted' and 'None'.
    harm_phase_norm : bool, default=True
        If True, normalize the harmonicity and phase coupling values
        by dividing by the total power.

    Returns
    -------
    df : DataFrame
        A DataFrame containing computed harmonicity, phase coupling, and resonance values, spectral flatness,
        entropy, Higushi Fractal Dimension, and spectral spread for each of the three spectra (harmonicity,
        phase coupling, resonance). Also includes average values and maximum values for these metrics, peak frequencies
        for each spectrum, and 'harmsim' values for peak frequencies.

        """
    # Perform initial operations and get cleaned PSD
    freqs, psd = compute_frequency_and_psd(signal, precision_hz, smoothness, fs, noverlap, fmin=fmin, fmax=fmax)
    psd_clean = apply_power_law_remove(freqs, psd, power_law_remove)
    psd_min = np.min(psd_clean)
    psd_max = np.max(psd_clean)
    psd_clean = (psd_clean - psd_min) / (psd_max - psd_min)
    
    # Get phase matrix
    phase_matrix = compute_phase_matrix(signal, precision_hz, fs, noverlap, smoothness)

    # Compute dyad similarities and phase couplings
    dyad_similarities, phase_coupling_matrix = compute_dyad_similarities_and_phase_coupling_matrix(freqs, phase_matrix, metric, n_harms, delta_lim, min_notes=min_notes)
    
    # Compute harmonicity and phase coupling values
    harmonicity_values, phase_coupling_values = compute_harmonicity_values_and_phase_coupling_values(freqs, dyad_similarities, phase_coupling_matrix, psd_clean, phase_mode,
                                                                                                     harm_phase_norm=harm_phase_norm)
    # apply smoothing to harmonicity values
    #print(phase_coupling_values)
    harmonicity_values = gaussian_filter(harmonicity_values, smoothness_harm)
    phase_coupling_values = gaussian_filter(phase_coupling_values, smoothness_harm)
    
    # Compute resonance values
    resonance_values = compute_resonance_values(harmonicity_values, phase_coupling_values)

    # Find peaks in the spectra
    harmonicity_peak_frequencies, harm_peak_idx = find_spectral_peaks(harmonicity_values, freqs, n_peaks, prominence_threshold=0.5)
    phase_peak_frequencies, phase_peak_idx = find_spectral_peaks(phase_coupling_values, freqs, n_peaks, prominence_threshold=0.0001)
    resonance_peak_frequencies, res_peak_idx = find_spectral_peaks(resonance_values, freqs, n_peaks, prominence_threshold=0.00001)

    # Compute spectral flatness and entropy values
    harmonic_complexity = harmonic_entropy(freqs, harmonicity_values, phase_coupling_values, resonance_values)

    df = pd.DataFrame({
        'harmonicity': [harmonicity_values],
        'phase_coupling': [phase_coupling_values],
        'resonance': [resonance_values],
        'harm_spectral_flatness': [harmonic_complexity['Spectral Flatness']['Harmonicity']],
        'harm_spectral_entropy': [harmonic_complexity['Spectral Entropy']['Harmonicity']],
        'harm_higuchi': [harmonic_complexity['Higuchi Fractal Dimension']['Harmonicity']],
        'harm_spectral_spread': [harmonic_complexity['Spectral Spread']['Harmonicity']],
        'phase_spectral_flatness': [harmonic_complexity['Spectral Flatness']['Phase Coupling']],
        'phase_spectral_entropy': [harmonic_complexity['Spectral Entropy']['Phase Coupling']],
        'phase_higuchi': [harmonic_complexity['Higuchi Fractal Dimension']['Phase Coupling']],
        'phase_spectral_spread': [harmonic_complexity['Spectral Spread']['Phase Coupling']],
        'res_spectral_flatness': [harmonic_complexity['Spectral Flatness']['Resonance']],
        'res_spectral_entropy': [harmonic_complexity['Spectral Entropy']['Resonance']],
        'res_higuchi': [harmonic_complexity['Higuchi Fractal Dimension']['Resonance']],
        'res_spectral_spread': [harmonic_complexity['Spectral Spread']['Resonance']],
        'harmonicity_peak_frequencies': [harmonicity_peak_frequencies],
        'phase_peak_frequencies': [phase_peak_frequencies],
        'resonance_peak_frequencies': [resonance_peak_frequencies],
    })
    
    df['harmonicity_avg'] = df['harmonicity'].apply(np.mean)
    df['phase_coupling_avg'] = df['phase_coupling'].apply(np.mean)
    df['resonance_avg'] = df['resonance'].apply(np.mean)
    
    df['harmonicity_peaks_avg'] = df['harmonicity_peak_frequencies'].apply(safe_mean)
    df['phase_peaks_avg'] = df['phase_peak_frequencies'].apply(safe_mean)
    df['res_peaks_avg'] = df['resonance_peak_frequencies'].apply(safe_mean)

    df['resonance_max'] = df['resonance'].apply(np.max)

    #save df
    df['precision'] = precision_hz
    df['fmin'] = fmin
    df['fmax'] = fmax
    df['phase_weighting'] = phase_mode
    df['smooth_fft'] = smoothness
    df['smooth_harm'] = smoothness_harm
    df['fs'] = fs
    
    #calculate harmonic similarity between peaks
    df['phase_harmsim'] = df['phase_peak_frequencies'].apply(peaks_to_harmsim)
    df['harm_harmsim'] = df['harmonicity_peak_frequencies'].apply(peaks_to_harmsim)
    df['res_harmsim'] = df['resonance_peak_frequencies'].apply(peaks_to_harmsim)
    
    df['harm_harmsim_avg'] = df['harm_harmsim'].apply(safe_mean)
    df['phase_harmsim_avg'] = df['phase_harmsim'].apply(safe_mean)
    df['res_harmsim_avg'] = df['res_harmsim'].apply(safe_mean)
    
    df['harm_harmsim_max'] = df['harm_harmsim'].apply(safe_max)
    df['phase_harmsim_max'] = df['phase_harmsim'].apply(safe_max)
    df['res_harmsim_max'] = df['res_harmsim'].apply(safe_max)
    # Plot results if required
    if plot:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12))

        ax1.plot(freqs, 10 * np.log10(psd), color='black')
        ax1.set_title('Spectrum')
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Power (dB)')
        ax1.grid()

        ax2.plot(freqs, harmonicity_values, color='darkblue')
        ax2.plot(freqs[harm_peak_idx], harmonicity_values[harm_peak_idx], 'ro')
        ax2.set_title('Harmonic Spectrum')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Harmonicity')
        # add vertical lines for the peaks in the spectrum
        for peak in harmonicity_peak_frequencies:
            ax2.axvline(peak, color='darkblue', linestyle='--')
        
        # add text for the spectral flatness and entropy values on the toop right corner
        ax2.text(0.85, 0.9, 'Spectral Flatness: ' + str(round(harmonic_complexity['Spectral Flatness']['Harmonicity'], 2)), transform=ax2.transAxes)
        ax2.text(0.85, 0.80, 'Spectral Entropy: ' + str(round(harmonic_complexity['Spectral Entropy']['Harmonicity'], 2)), transform=ax2.transAxes)
        
        ax2.grid()
        
        ax3.plot(freqs, phase_coupling_values, color='darkviolet')
        ax3.plot(freqs[phase_peak_idx], phase_coupling_values[phase_peak_idx], 'ro')
        ax3.set_title('Phase Coupling Spectrum')
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Phase Coupling')
        # add vertical lines for the peaks in the spectrum
        for peak in phase_peak_frequencies:
            ax3.axvline(peak, color='darkviolet', linestyle='--')
            
        # add text for the spectral flatness and entropy values on the toop right corner
        ax3.text(0.85, 0.9, 'Spectral Flatness: ' + str(round(harmonic_complexity['Spectral Flatness']['Phase Coupling'], 2)), transform=ax3.transAxes)
        ax3.text(0.85, 0.80, 'Spectral Entropy: ' + str(round(harmonic_complexity['Spectral Entropy']['Phase Coupling'], 2)), transform=ax3.transAxes)
            
        ax3.grid()
        
        ax4.plot(freqs, resonance_values, color='darkred')
        ax4.plot(freqs[res_peak_idx], resonance_values[res_peak_idx], 'ro')
        ax4.set_title('Resonance Spectrum')
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('Resonance')
        # add vertical lines for the peaks in the spectrum
        for peak in resonance_peak_frequencies:
            ax4.axvline(peak, color='darkred', linestyle='--')
        # add text for the spectral flatness and entropy values on the toop right corner
        ax4.text(0.85, 0.9, 'Spectral Flatness: ' + str(round(harmonic_complexity['Spectral Flatness']['Resonance'], 2)), transform=ax4.transAxes)
        ax4.text(0.85, 0.80, 'Spectral Entropy: ' + str(round(harmonic_complexity['Spectral Entropy']['Resonance'], 2)), transform=ax4.transAxes)
            
        ax4.grid()

        plt.tight_layout()
        if save is True:
            plt.savefig('Spectra' + savename + '.png')
        
        plt.show()


    return df


def harmonic_spectrum_plot_trial_corr(df_all, df_all_rnd, label1='Brain Signals', label2='Random Signals'):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
    
    corrs = []
    ps = []
    reg_lines = []
    scaler = MinMaxScaler()
    for i in range(len(df_all)):

        harm_values = df_all[df_all['trial'] == i]['harmonicity'][0]
        phase_coupling_values = df_all[df_all['trial'] == i]['phase_coupling'][0]
        # Scale the data
        harm_values = scaler.fit_transform(harm_values.reshape(-1, 1))
        phase_coupling_values = scaler.fit_transform(phase_coupling_values.reshape(-1, 1))

        # Flatten the arrays
        harm_values = harm_values.flatten()
        phase_coupling_values = phase_coupling_values.flatten()
        
        corr, p = pearsonr(harm_values, phase_coupling_values)
        corrs.append(corr)
        ps.append(p)
        z = np.polyfit(harm_values, phase_coupling_values, 1)
        reg_lines.append(z)
        ax1.plot(np.sort(harm_values), np.poly1d(z)(np.sort(harm_values)), color='darkblue', linestyle='--', alpha=0.5)
        ax1.set_title(label1)
        ax1.set_xlabel('Mean Harmonicity across freqs')
        ax1.set_ylabel('Mean Phase-Coupling across freqs')
    ax1.text(0.95, 0.95, f'r [{np.round(np.min(corrs), 2)}, {np.round(np.max(corrs), 2)}]', ha='right', va='top', transform=ax1.transAxes, fontsize=10, fontweight='bold')

    corrs_rnd = []
    ps_rnd = []
    reg_lines_rnd = []
    
    for i in range(len(df_all_rnd)):
        harm_values_rnd = df_all_rnd[df_all_rnd['trial'] == i]['harmonicity'][0]
        phase_coupling_values_rnd = df_all_rnd[df_all_rnd['trial'] == i]['phase_coupling'][0]
        # Scale the data
        harm_values_rnd = scaler.fit_transform(harm_values_rnd.reshape(-1, 1))
        phase_coupling_values_rnd = scaler.fit_transform(phase_coupling_values_rnd.reshape(-1, 1))

        # Flatten the arrays
        harm_values_rnd = harm_values_rnd.flatten()
        phase_coupling_values_rnd = phase_coupling_values_rnd.flatten()
        corr_rnd, p_rnd = pearsonr(harm_values_rnd, phase_coupling_values_rnd)
        corrs_rnd.append(corr_rnd)
        ps_rnd.append(p_rnd)
        z_rnd = np.polyfit(harm_values_rnd, phase_coupling_values_rnd, 1)
        reg_lines_rnd.append(z_rnd)
        ax2.plot(np.sort(harm_values_rnd), np.poly1d(z_rnd)(np.sort(harm_values_rnd)), color='darkblue', linestyle='--', alpha=0.5)
        ax2.set_title(label2)
        ax2.set_xlabel('Mean Harmonicity across freqs')
        ax2.set_ylabel('Mean Phase-Coupling across freqs')
    ax2.text(0.95, 0.95, f'r [{np.round(np.min(corrs_rnd), 2)}, {np.round(np.max(corrs_rnd), 2)}]', ha='right', va='top', transform=ax2.transAxes, fontsize=10, fontweight='bold')

    # Calculate the 5th and 95th percentiles for correlation values
    corr_5th = np.percentile(corrs, 5)
    corr_95th = np.percentile(corrs, 95)
    corr_rnd_5th = np.percentile(corrs_rnd, 5)
    corr_rnd_95th = np.percentile(corrs_rnd, 95)

    # Calculate the t-statistics and p-value for the two-sample t-test
    t_stat, p_val = ttest_ind(corrs, corrs_rnd)
    sns.distplot(corrs, ax=ax3, label=label1, color='blue')
    ax3.axvline(corr_5th, color='blue', linestyle='--')
    ax3.axvline(corr_95th, color='blue', linestyle='--')
    sns.distplot(corrs_rnd, ax=ax3, label=label2, color='red')
    ax3.axvline(corr_rnd_5th, color='red', linestyle='--')
    ax3.axvline(corr_rnd_95th, color='red', linestyle='--')
    ax3.set_title('Distribution of correlation values')
    ax3.set_xlabel('Correlation (r)')
    ax3.set_ylabel('Density')
    ax3.legend()

    # Add text with t-test result to the plot
    ax3.text(0.95, 0.05, f't={np.round(t_stat, 2)}, p={np.round(p_val, 4)}', ha='right', va='top', transform=ax3.transAxes, fontsize=10, fontweight='bold')

    fig.tight_layout()
    plt.show()




def harmonic_spectrum_plot_freq_corr(df1, df2, mean_phase_coupling=False, label1='Brain Signals', label2='Random Signals', fmin=2, fmax=30, xlim=None):
    
    n = len(df1)  # Assuming df1 and df2 have the same number of rows

    # Calculate the critical r-values
    alpha = 0.05
    r_critical = np.sqrt(chi2.ppf(1 - alpha, df=1) / n)
    r_positive = r_critical
    r_negative = -r_critical

    # Define frequency range and bins
    freqs = np.linspace(fmin, fmax, len(df1['harmonicity'][0].tolist()[0]))

    # Initialize arrays to hold the correlation values for each frequency bin
    corrs1 = np.zeros(len(freqs))
    corrs2 = np.zeros(len(freqs))

    # Initialize arrays to hold the mean harmonicity values for each frequency bin
    mean_harmonicity1 = np.zeros(len(freqs))
    mean_harmonicity2 = np.zeros(len(freqs))

    mean_phase_coupling1 = np.zeros(len(freqs))
    mean_phase_coupling2 = np.zeros(len(freqs))

    # Compute the correlation for each frequency bin
    for i in range(len(freqs)):
        harm_values1 = [row[i] for row in df1['harmonicity']]
        mean_harmonicity1[i] = np.mean(harm_values1)
        phase_coupling_values1 = [row[i] for row in df1['phase_coupling']]
        corrs1[i] = np.corrcoef(harm_values1, phase_coupling_values1)[0,1]
        
        if mean_phase_coupling:
            mean_phase_coupling1[i] = np.mean(phase_coupling_values1)

        harm_values2 = [row[i] for row in df2['harmonicity']]
        mean_harmonicity2[i] = np.mean(harm_values2)
        phase_coupling_values2 = [row[i] for row in df2['phase_coupling']]
        corrs2[i] = np.corrcoef(harm_values2, phase_coupling_values2)[0,1]

        if mean_phase_coupling:
            mean_phase_coupling2[i] = np.mean(phase_coupling_values2)

    scaler = MinMaxScaler()

    mean_harmonicity1_scaled = scaler.fit_transform(np.array(mean_harmonicity1).reshape(-1,1)).flatten()
    mean_harmonicity2_scaled = scaler.fit_transform(np.array(mean_harmonicity2).reshape(-1,1)).flatten()

    if mean_phase_coupling:
        mean_phase_coupling1_scaled = scaler.fit_transform(np.array(mean_phase_coupling1).reshape(-1,1)).flatten()
        mean_phase_coupling2_scaled = scaler.fit_transform(np.array(mean_phase_coupling2).reshape(-1,1)).flatten()

    plt.figure(figsize=(12.5,4.5))

    # Calculate the min and max for the correlations
    corrs_min = min(corrs1.min(), corrs2.min())-0.05
    corrs_max = max(corrs1.max(), corrs2.max())+0.05

    # Calculate the min and max for the mean harmonicity
    harm_min = min(mean_harmonicity1_scaled.min(), mean_harmonicity2_scaled.min())-0.05
    harm_max = max(mean_harmonicity1_scaled.max(), mean_harmonicity2_scaled.max())+0.05

    # Calculate the min and max for the mean phase-coupling (if flag is true)
    if mean_phase_coupling:
        phase_min = min(mean_phase_coupling1_scaled.min(), mean_phase_coupling2_scaled.min())-0.05
        phase_max = max(mean_phase_coupling1_scaled.max(), mean_phase_coupling2_scaled.max())+0.05

    plt.subplot(1, 2, 1)
    ax1 = plt.gca()
    line1, = ax1.plot(freqs, corrs1, color='black', label='Correlation (Harm x Phase)')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Correlation')
    ax1.set_ylim(corrs_min, corrs_max)
    ax1.axhline(r_positive, color='k', linestyle='--', label='p=0.05')
    ax1.axhline(r_negative, color='k', linestyle='--')

    ax2 = ax1.twinx()
    line2, = ax2.plot(freqs, mean_harmonicity1_scaled, color='mediumblue', label='Mean Harmonicity')
    
    if mean_phase_coupling:
        line3, = ax2.plot(freqs, mean_phase_coupling1_scaled, color='deeppink', label='Mean Phase-Coupling')

    ax2.set_ylabel('Normalized measures')
    ax2.set_ylim(0, 1)  # Since both measures are normalized

    if xlim is not None:
        ax1.set_xlim(xlim[0], xlim[1])
        ax2.set_xlim(xlim[0], xlim[1])

    lines = [line1, line2]
    if mean_phase_coupling:
        lines.append(line3)
    ax1.legend(lines, [l.get_label() for l in lines], loc='upper right')
    ax1.set_title(label1)

    plt.subplot(1, 2, 2)
    ax1 = plt.gca()
    line1, = ax1.plot(freqs, corrs2, color='black', label='Correlation (Harm x Phase)')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Correlation')
    ax1.set_ylim(corrs_min, corrs_max)
    ax1.axhline(r_positive, color='k', linestyle='--', label='p=0.05')
    ax1.axhline(r_negative, color='k', linestyle='--')

    ax2 = ax1.twinx()
    line2, = ax2.plot(freqs, mean_harmonicity2_scaled, color='mediumblue', label='Mean Harmonicity')
    
    if mean_phase_coupling:
        line3, = ax2.plot(freqs, mean_phase_coupling2_scaled, color='deeppink', label='Mean Phase-Coupling')

    ax2.set_ylabel('Normalized measures')
    ax2.set_ylim(0, 1)  # Since both measures are normalized

    if xlim is not None:
        ax1.set_xlim(xlim[0], xlim[1])
        ax2.set_xlim(xlim[0], xlim[1])

    lines = [line1, line2]
    if mean_phase_coupling:
        lines.append(line3)
    ax1.legend(lines, [l.get_label() for l in lines], loc='upper right')
    ax1.set_title(label2)

    plt.tight_layout()
    plt.show()


def harmonic_spectrum_plot_avg_corr(df1, df2, label1='Brain Signals', label2='Random Signals'):
    harm_values = [np.mean(row) for row in df1['harmonicity']]
    phase_coupling_values = [np.mean(row) for row in df1['phase_coupling']]

    harm_values_rnd = [np.mean(row) for row in df2['harmonicity']]
    phase_coupling_values_rnd = [np.mean(row) for row in df2['phase_coupling']]

    plt.figure(figsize=(8,4))

    # plot the correlation between harmonicity and phase coupling for Brain signals
    plt.subplot(1, 2, 1)
    plt.scatter(harm_values, phase_coupling_values, color='darkblue', alpha=0.5)
    plt.xlabel('Averaged Harmonicity')
    plt.ylabel('Averaged Phase Coupling')
    z = np.polyfit(harm_values, phase_coupling_values, 1)
    plt.plot(np.sort(harm_values), np.poly1d(z)(np.sort(harm_values)), "r--")
    corr, p = pearsonr(harm_values, phase_coupling_values)
    print(f'{label1} - correlation: ', corr, 'p-value: ', p)
    plt.title(label1)
    vmin_x = min(harm_values + harm_values_rnd)
    vmax_x = max(harm_values + harm_values_rnd)
    vmin_y = min(phase_coupling_values + phase_coupling_values_rnd)
    vmax_y = max(phase_coupling_values + phase_coupling_values_rnd)
    
    plt.xlim(vmin_x-(vmin_x/100), vmax_x+(vmax_x/100))
    plt.ylim(vmin_y-(vmin_y/100), vmax_y+(vmax_y/100))

    # plot the correlation between harmonicity and phase coupling for Random signals
    plt.subplot(1, 2, 2)
    plt.scatter(harm_values_rnd, phase_coupling_values_rnd, color='darkblue', alpha=0.5)
    plt.xlabel('Averaged Harmonicity')
    plt.ylabel('Averaged Phase Coupling')
    z_rnd = np.polyfit(harm_values_rnd, phase_coupling_values_rnd, 1)
    plt.plot(np.sort(harm_values_rnd), np.poly1d(z_rnd)(np.sort(harm_values_rnd)), "r--")
    corr_rnd, p_rnd = pearsonr(harm_values_rnd, phase_coupling_values_rnd)
    print(f'{label2} - correlation: ', corr_rnd, 'p-value: ', p_rnd)
    plt.title(label2)
    
    plt.xlim(vmin_x-(vmin_x/100), vmax_x+(vmax_x/100))
    plt.ylim(vmin_y-(vmin_y/100), vmax_y+(vmax_y/100))

    plt.tight_layout()
    plt.show()
