"""biotuner.harmonic_spectrum — harmonicity-field computations from PSDs.

Module type: Functions

This module is the H-only entry point of biotuner's spectral resonance machinery.
The full H × PC = R pipeline (formerly the legacy ``compute_global_harmonicity``)
now lives in :mod:`biotuner.resonance`. See :func:`biotuner.resonance.compute_resonance`
for the full strategy-registry-based pipeline with surrogate normalization, soft
Arnold-tongue gating, additional harmonic kernels, and higher-order coupling.

Public entry points:
    compute_harmonic_spectrum    — H(f) spectrum + rich complexity summary
    harmonicity_matrices         — N×N similarity matrix (legacy helper)
    compute_harmonic_power       — probability-weighted reduction of S to H(f)
    find_spectral_peaks          — peak detector used across the resonance package
    harmonic_entropy             — complexity DataFrame for H / PC / R together
    get_harmonic_ratio           — best (n, m) within tolerance for a freq pair
    count_theoretical_harmonic_partners — bandwidth-correction helper
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences
from scipy.ndimage import gaussian_filter

from biotuner.metrics import (
    dyad_similarity,
    compute_subharmonic_tension,
    spectral_flatness,
    spectral_entropy,
    spectral_spread,
    higuchi_fd,
)
from biotuner.biotuner_utils import (
    apply_power_law_remove,
    compute_frequency_and_psd,
)


def compute_harmonic_spectrum(
    signal,
    precision_hz,
    *,
    fmin=1,
    fmax=30,
    fs=1000,
    noverlap=1,
    smoothness=1,
    smoothness_harm=1,
    power_law_remove=True,
    harmonic_kernel="harmsim",
    harmonic_kernel_params=None,
    n_peaks=5,
    normalize=True,
    bandwidth_correction=False,
    psd_normalization="minmax_prob",
    legacy_self_pair_subtract=True,
):
    """Compute the harmonicity spectrum H(f) and its complexity summary.

    This is the narrow, H-only entry point — replaces the harmonicity portion
    of the legacy ``compute_global_harmonicity``. For the full H/PC/R triple
    use :func:`biotuner.resonance.compute_resonance`.

    Under the hood this dispatches the harmonic kernel from the
    :mod:`biotuner.resonance` registry, so all kernels added in later phases
    (sethares, stolzenburg, harmonic_entropy, hopf, lorentzian) become available
    automatically by name.

    Parameters
    ----------
    signal : 1-D array
    precision_hz : float
    fmin, fmax : float
    fs : int, sampling frequency
    noverlap : int, STFT overlap
    smoothness : float, divides STFT nperseg
    smoothness_harm : float, Gaussian smoothing on H before summary
    power_law_remove : bool, FOOOF aperiodic removal
    harmonic_kernel : str, name in :data:`biotuner.resonance.registry.HARMONIC_KERNELS`
        ``'harmsim'`` (default) | ``'subharm_tension'`` (legacy) | future kernels.
    harmonic_kernel_params : dict, passed to the kernel
    n_peaks : int, peaks returned by the summary
    normalize : bool, legacy normalize flag for the per-bin reducer
    bandwidth_correction : bool, legacy bandwidth correction
    psd_normalization : 'minmax_prob' (legacy) | 'prob' | 'none'
    legacy_self_pair_subtract : bool, reproduce legacy self-pair-subtract reducer

    Returns
    -------
    freqs : ndarray (n_freqs,)
    H_values : ndarray (n_freqs,)  -- the harmonicity spectrum
    harmonicity_matrix : ndarray (n_freqs, n_freqs)  -- the kernel similarity matrix
    summary : dict
        spectrum_complexity output: flatness, entropy, spread, higuchi, peaks,
        peak_indices, avg, max, peaks_avg, peak_harmsim, peak_harmsim_avg,
        peak_harmsim_max.

    Examples
    --------
    >>> freqs, H, M, summary = compute_harmonic_spectrum(signal, 0.5, fmin=2, fmax=30)
    >>> summary['flatness'], summary['entropy'], summary['peaks']
    """
    from biotuner.resonance.registry import HARMONIC_KERNELS
    from biotuner.resonance.coupling import reduce_matrix_to_spectrum
    from biotuner.metrics import spectrum_complexity

    if harmonic_kernel_params is None:
        harmonic_kernel_params = {}

    freqs, psd = compute_frequency_and_psd(
        signal, precision_hz, smoothness=smoothness, fs=fs, noverlap=noverlap, fmin=fmin, fmax=fmax
    )
    psd_clean = apply_power_law_remove(freqs, psd, power_law_remove)

    if psd_normalization == "minmax_prob":
        psd_min = np.min(psd_clean)
        psd_max = np.max(psd_clean)
        psd_clean = (psd_clean - psd_min) / (psd_max - psd_min)
        psd_prob = psd_clean / np.sum(psd_clean)
    elif psd_normalization == "prob":
        psd_prob = psd_clean / np.sum(psd_clean)
    else:
        psd_prob = psd_clean.copy()

    if harmonic_kernel not in HARMONIC_KERNELS:
        raise ValueError(
            f"Unknown harmonic_kernel {harmonic_kernel!r}. "
            f"Available: {list(HARMONIC_KERNELS)}"
        )
    kernel_fn = HARMONIC_KERNELS[harmonic_kernel]
    harmonicity_matrix = kernel_fn(freqs, freqs, **harmonic_kernel_params)

    H_values = reduce_matrix_to_spectrum(
        harmonicity_matrix,
        psd_prob,
        normalize=normalize,
        legacy_self_pair_subtract=legacy_self_pair_subtract,
    )

    if bandwidth_correction:
        max_possible_partners = int(fmax / fmin) - 1
        for i in range(len(freqs)):
            n_partners = int(fmax / freqs[i]) - 1
            if n_partners > 0:
                H_values[i] *= max_possible_partners / n_partners

    if smoothness_harm > 0:
        H_values = gaussian_filter(H_values, smoothness_harm)

    summary = spectrum_complexity(H_values, freqs, n_peaks=n_peaks, prominence_threshold=0.5)
    return freqs, H_values, harmonicity_matrix, summary


def harmonicity_matrices(freqs, metric='harmsim', n_harms=5, delta_lim=150, min_notes=2):
    '''
    Compute harmonicity matrix of frequencies.

    Parameters
    ----------
    freqs : ndarray
        Array of frequencies.
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
    ndarray
        The harmonicity matrix.

    See Also
    --------
    biotuner.resonance.kernels_harmonic.kernel_harmsim : new strategy-registry version
        with the same numerics, plus support for additional kernels (sethares,
        stolzenburg, harmonic_entropy, hopf, lorentzian) in later phases.
    '''
    harmonicity = np.zeros((len(freqs), len(freqs)))

    for i, f1 in enumerate(freqs):
        for j, f2 in enumerate(freqs):
            if f2 != 0:
                if metric == 'harmsim':
                    harmonicity[i, j] = dyad_similarity(f1 / f2)
                if metric == 'subharm_tension':
                    _, _, subharm, _ = compute_subharmonic_tension([f1, f2], n_harmonics=n_harms, delta_lim=delta_lim, min_notes=min_notes)
                    harmonicity[i, j] = 1-subharm[0]
    return harmonicity


def get_harmonic_ratio(freq_i, freq_j, max_n=3, max_m=3, tolerance=0.05):
    """
    Identify if two frequencies are at an n:m harmonic ratio.

    Parameters
    ----------
    freq_i : float
        First frequency.
    freq_j : float
        Second frequency.
    max_n : int, default=3
        Maximum value for n in n:m ratio.
    max_m : int, default=3
        Maximum value for m in n:m ratio.
    tolerance : float, default=0.05
        Relative tolerance for ratio matching (5%).

    Returns
    -------
    tuple or None
        (n, m) if frequencies are harmonically related, None otherwise.

    See Also
    --------
    biotuner.resonance.kernels_ratio.binary_nm_kernel : vectorized version
        operating on freq arrays (returns W, N, M matrices).
    """
    if freq_i == 0:
        return None
    ratio = freq_j / freq_i
    best_match = None
    min_error = float('inf')
    for n in range(1, max_n + 1):
        for m in range(1, max_m + 1):
            expected_ratio = m / n
            error = abs(ratio - expected_ratio) / expected_ratio
            if error < tolerance and error < min_error:
                min_error = error
                best_match = (n, m)
    return best_match


def count_theoretical_harmonic_partners(freq, fmin, fmax, max_ratio=5):
    """
    Count theoretical maximum number of harmonic partners within bandwidth.

    This function counts how many frequencies at simple n:m ratios (up to max_ratio)
    would fall within the analysis bandwidth [fmin, fmax] for a given frequency.
    Lower frequencies have more potential partners, creating systematic bias.

    Parameters
    ----------
    freq : float
        The frequency to analyze.
    fmin : float
        Minimum frequency of analysis bandwidth.
    fmax : float
        Maximum frequency of analysis bandwidth.
    max_ratio : int, default=5
        Maximum value for n and m in n:m ratios to consider.

    Returns
    -------
    int
        Number of theoretical harmonic partners within bandwidth.
    """
    n_partners = 0
    for n in range(1, max_ratio + 1):
        for m in range(1, max_ratio + 1):
            if n == m:
                continue
            harmonic_freq = freq * (m / n)
            if fmin <= harmonic_freq <= fmax:
                n_partners += 1
    return n_partners


def compute_harmonic_power(freqs, dyad_similarities, psd_clean, normalize=True,
                          bandwidth_correction=False, fmin=1, fmax=30):
    '''
    Compute harmonicity as probability-weighted average of dyad similarities.

    The harmonicity of frequency i represents the expected harmonic similarity
    with other frequencies, weighted by the joint probability of observing
    each frequency pair. This formulation is scale-invariant and not sensitive
    to spectral shape or total power distribution.

    Parameters
    ----------
    freqs : ndarray
        Array of frequencies.
    dyad_similarities : ndarray
        Dyad similarities matrix.
    psd_clean : ndarray
        The cleaned Power Spectral Density (PSD).
    normalize : bool, default=True
        If True, compute harmonicity as weighted average (recommended).
        If False, compute as sum of weighted harmonic contributions.
    bandwidth_correction : bool, default=False
        If True, apply correction for bandwidth bias where lower frequencies
        have more potential harmonic partners within the analysis range.
    fmin : float, default=1
        Minimum frequency of analysis bandwidth (used for bandwidth correction).
    fmax : float, default=30
        Maximum frequency of analysis bandwidth (used for bandwidth correction).

    Returns
    -------
    tuple of ndarrays
        The harmonicity values and the harmonicity matrix.

    See Also
    --------
    biotuner.resonance.coupling.reduce_matrix_to_spectrum : strategy-registry
        version supporting both the legacy self-pair-subtract reduction and the
        clean off-diagonal-mask reduction recommended for new code.

    Notes
    -----
    The weighting uses psd_prob[i] * psd_prob[j] which requires both
    frequencies to have high power for high harmonicity contribution.
    '''
    psd_prob = psd_clean / np.sum(psd_clean)
    harmonicity_values = np.zeros(len(freqs))
    harmonicity_matrix = np.zeros((len(freqs), len(freqs)))
    for i in range(len(freqs)):
        for j in range(len(freqs)):
            if i != j:
                harmonicity_matrix[i, j] = dyad_similarities[i, j] * psd_prob[i] * psd_prob[j]
        if normalize is True:
            harmonicity_values[i] = psd_prob[i] * np.sum(
                dyad_similarities[i, :] * psd_prob
            ) - dyad_similarities[i, i] * psd_prob[i] ** 2  # exclude self-pair
        else:
            harmonicity_values[i] = np.sum(harmonicity_matrix[i, :])
    if bandwidth_correction:
        max_possible_partners = int(fmax / fmin) - 1
        for i in range(len(freqs)):
            n_partners = int(fmax / freqs[i]) - 1
            if n_partners > 0:
                correction = max_possible_partners / n_partners
                harmonicity_values[i] = harmonicity_values[i] * correction
    return harmonicity_values, harmonicity_matrix


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
    peaks, _ = find_peaks(values)
    prominences = peak_prominences(values, peaks)[0]
    filtered_peak_indices = np.where(prominences > prominence_threshold)[0]
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
    biotuner.metrics.spectrum_complexity : single-spectrum version returning a dict;
        used by both this function and the resonance orchestrator.
    """
    SpecFlat_harmonicity = spectral_flatness(harmonicity_values)
    SpecFlat_phase_coupling = spectral_flatness(phase_coupling_values)
    SpecFlat_resonance = spectral_flatness(resonance_values)

    SpecEnt_harmonicity = spectral_entropy(harmonicity_values)
    SpecEnt_phase_coupling = spectral_entropy(phase_coupling_values)
    SpecEnt_resonance = spectral_entropy(resonance_values)

    SpecSpread_harmonicity = spectral_spread(freqs, harmonicity_values)
    SpecSpread_phase_coupling = spectral_spread(freqs, phase_coupling_values)
    SpecSpread_resonance = spectral_spread(freqs, resonance_values)

    HFD_harmonicity = higuchi_fd(harmonicity_values, kmax=10)
    HFD_phase_coupling = higuchi_fd(phase_coupling_values, kmax=10)
    HFD_resonance = higuchi_fd(resonance_values, kmax=10)

    harmonic_complexity = pd.DataFrame(
        {
            'Spectral Flatness': [SpecFlat_harmonicity, SpecFlat_phase_coupling, SpecFlat_resonance],
            'Spectral Entropy': [SpecEnt_harmonicity, SpecEnt_phase_coupling, SpecEnt_resonance],
            'Spectral Spread': [SpecSpread_harmonicity, SpecSpread_phase_coupling, SpecSpread_resonance],
            'Higuchi Fractal Dimension': [HFD_harmonicity, HFD_phase_coupling, HFD_resonance],
        },
        index=['Harmonicity', 'Phase Coupling', 'Resonance'],
    )
    return harmonic_complexity
