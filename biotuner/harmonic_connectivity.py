"""biotuner.harmonic_connectivity — cross-channel coupling, cross-frequency, and connectivity matrices.

Module type: Object

This module is the **cross-channel** counterpart to
:mod:`biotuner.harmonic_spectrum` (single-signal H) and
:mod:`biotuner.resonance` (single-signal R = H · PC). Everything that compares
TWO-OR-MORE signals lives here.

Quick start
-----------

**Peak-based** connectivity (between extracted peak lists, one scalar per
electrode pair) — uses the legacy peak-extraction pipeline:

::

    from biotuner.harmonic_connectivity import harmonic_connectivity
    hc = harmonic_connectivity(sf=1000, data=data_array, peaks_function="EMD",
                                precision=0.5, min_freq=2, max_freq=30, n_peaks=5)
    H_mat = hc.compute_harm_connectivity(metric="harmsim")       # legacy H
    PC_mat = hc.compute_peak_phase_coupling_connectivity(coupling_metric="nm_plv")
    R_mat = hc.compute_peak_resonance_connectivity(combine="product")

**Spectrum-based** cross-channel resonance (per-frequency H/PC/R between two
signals, with the same swappable kernels/metrics/combine rules as
``biotuner.resonance``):

::

    from biotuner.harmonic_connectivity import compute_cross_resonance
    from biotuner.resonance import ResonanceConfig

    cross = compute_cross_resonance(sig1, sig2, sf=1000)
    # cross.resonance_spectrum["1to2"]  — asymmetric (sig1 at i, sig2 at j)
    # cross.resonance_spectrum["2to1"]  — transposed
    # cross.resonance_spectrum["all"]   — symmetrized average
    # cross.factors["H"][...], cross.factors["PC"][...]
    # cross.summaries["H"/"PC"/"R"]     — complexity dict per spectrum

**Connectivity matrices** (loop ``compute_cross_resonance`` over all electrode
pairs):

::

    M = hc.compute_cross_resonance_connectivity(
        factor="R", flavor="all", aggregate="peak_to_median",
    )

**Statistical inference** via surrogate-z-scoring (the principled way to
separate true cross-channel phase coupling from broadband-power artifacts):

::

    obs, z, p = hc.compute_cross_resonance_connectivity_zscore(
        surrogate_kind="iaaft", n_surrogates=200,
    )

Sister modules
--------------
- :mod:`biotuner.harmonic_spectrum` — single-signal H spectrum.
- :mod:`biotuner.resonance` — single-signal H × PC = R pipeline, including the
  registries this module dispatches against
  (``PAIRWISE_COUPLING_METRICS``, ``HARMONIC_KERNELS``,
  ``RATIO_KERNELS``, ``COMBINE_RULES``).
"""

__all__ = [
    # Class — peak-based connectivity + new spectrum-based methods
    "harmonic_connectivity",
    # Spectrum-based cross-channel orchestrator
    "compute_cross_resonance",
    "CrossResonanceResult",
    # Legacy shim (delegates to compute_cross_resonance internally)
    "compute_cross_spectrum_harmonicity",
    # Standalone coupling utilities (kept for backward compat)
    "wPLI_crossfreq",
    "wPLI_multiband",
    "cross_frequency_rrci",
    "n_m_phase_locking",
    "rhythmic_ratio_coupling_imaginary",
    "compute_rhythmic_ratio",
    "compute_mutual_information",
    "MI_spectral",
    # IMF utilities (different abstraction layer)
    "HilbertHuang1D_nopeaks",
    "EMD_time_resolved_harmonicity",
    "temporal_correlation_fdr",
]

import numpy as np
from biotuner.biotuner_object import compute_biotuner
from biotuner.metrics import ratios2harmsim, compute_subharmonics_2lists, euler, dyad_similarity
from biotuner.biotuner_utils import rebound_list, butter_bandpass_filter
from biotuner.peaks_extension import harmonic_fit
from biotuner.transitional_harmony import transitional_harmony
from scipy.signal import hilbert
from scipy.stats import zscore, t
import itertools
import seaborn as sbn
import matplotlib.pyplot as plt
from fractions import Fraction
import mne
import numpy.ma as ma
# statsmodels and mne.viz are imported lazily inside the functions that need
# them (temporal_correlation_fdr and plot_conn_matrix respectively) so the
# module can be imported in environments where these optional deps aren't
# installed — useful for users who only need the core peak-based / spectrum
# pipelines.
import itertools
import pandas as pd
from scipy.signal import hilbert, coherence, welch
from scipy.stats import pearsonr
import seaborn as sns
from biotuner.metrics import (
    dyad_similarity,
    compute_subharmonic_tension,
    ratios2harmsim,
    peaks_to_harmsim,
)
from biotuner.harmonic_spectrum import (
    find_spectral_peaks,
    harmonic_entropy,
)
from biotuner.resonance.registry import (
    PAIRWISE_COUPLING_METRICS,
    COUPLING_INPUT_TYPE,
    RATIO_KERNELS,
    HARMONIC_KERNELS,
    COMBINE_RULES,
)
from biotuner.resonance.orchestrator import ResonanceConfig
from biotuner.metrics import spectrum_complexity
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from scipy.ndimage import gaussian_filter as _gaussian_filter
from biotuner.biotuner_utils import (
    safe_mean,
    safe_max,
    apply_power_law_remove,
    compute_frequency_and_psd,
)
from scipy.signal import stft
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_filter1d
import emd
from biotuner.peaks_extraction import EMD_eeg
 

class harmonic_connectivity(object):
    """
    Class used to compute harmonicity metrics
    between pairs of sensors.
    """

    def __init__(
        self,
        sf=None,
        data=None,
        peaks_function="EMD",
        precision=0.1,
        n_harm=10,
        harm_function="mult",
        min_freq=2,
        max_freq=80,
        n_peaks=5,
    ):
        """
        Parameters
        ----------
        sf: int
            sampling frequency (in Hz)
        data : 2D array (elec, numDataPoints)
            Electrodes x Time series to analyse.
        peaks_function: str, default='EMD'
            See compute_biotuner class for details.
        precision: float, default=0.1
            Precision of the peaks (in Hz)
            When HH1D_max is used, bins are in log scale.
        n_harm: int, default=10
            Set the number of harmonics to compute in harmonic_fit function
        harm_function: str, default='mult'
            Computes harmonics from iterative multiplication (x, 2x, 3x, ...nx)
            or division (x, x/2, x/3, ...x/n).
            Choose between 'mult' and 'div'
        min_freq: float, default=2
            Minimum frequency (in Hz) to consider for peak extraction.
        max_freq: float, default=80
            Maximum frequency (in Hz) to consider for peak extraction.
        n_peaks: int, default=5
            Number of peaks to extract per frequency band.

        """
        if type(data) is not None:
            self.data = data
        self.sf = sf
        # Initializing arguments for peak extraction
        self.peaks_function = peaks_function
        self.precision = precision
        self.n_harm = n_harm
        self.harm_function = harm_function
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.n_peaks = n_peaks

    def compute_harm_connectivity(
        self,
        metric="harmsim",
        delta_lim=20,
        save=False,
        savename="_",
        graph=True,
        FREQ_BANDS=None,
        max_denom_rrci=16,
    ):
        """
        Computes the harmonic connectivity matrix between electrodes.

        Parameters
        ----------
        metric : str, optional
            The metric to use for computing harmonic connectivity. Default is 'harmsim'.
            
            Possible values are:

             - 'harmsim': 
                computes the harmonic similarity between each pair of peaks from the two electrodes.
                It calculates the ratio between each pair of peaks and computes the mean harmonic similarity.

             - 'euler':
                computes the Euler's Gradus Suavitatis on the concatenated peaks of the two electrodes.

             - 'harm_fit':
                computes the number of common harmonics between each pair of peaks from the two electrodes.
                It evaluates the harmonic fit between each peak pair and counts the number of common harmonics.

             - 'subharm_tension':
                computes the tension between subharmonics of two electrodes.
                It evaluates the tension between subharmonics of the two electrodes by comparing the subharmonics and their ratios.

             - 'RRCi':
                computes the Rhythmic Ratio Coupling with Imaginary Component (RRCi) metric between each pair of
                peaks from the two electrodes, using a bandwidth of 2 Hz and a max_denom of 16. This metric calculates the
                imaginary part of the complex phase differences between two filtered signals.

             - 'wPLI_crossfreq':
                computes the weighted Phase Lag Index (wPLI) for cross-frequency coupling between each pair
                of peaks from the two electrodes. The wPLI measures the phase synchronization between two signals, with a value
                close to 0 indicating no synchronization and a value close to 1 indicating perfect synchronization.

             - 'wPLI_multiband':
                computes the weighted Phase Lag Index (wPLI) for multiple frequency bands between the two electrodes.
                It calculates wPLI for each frequency band and returns an array of wPLI values for the defined frequency bands.

        delta_lim : int, optional
            The delta limit for the subharmonic tension metric. Default is 20.

        save : bool, optional
            Whether to save the connectivity matrix. Default is False.

        savename : str, optional
            The name to use when saving the connectivity matrix. Default is '_'.

        graph : bool, optional
            Whether to display a heatmap of the connectivity matrix. Default is True.
            
        FREQ_BANDS : list, optional
            The frequency bands to use for the computation of the wPLI_multiband metric. Default is None.
            If None, the following frequency bands will be used: [2, 3.55], [3.55, 7.15], [7.15, 14.3], [14.3, 28.55], [28.55, 49.4].
            
        max_denom_rrci : int, optional
            The maximum denominator to use for the computation of the RRCi metric. Default is 16.

        Returns
        -------
        conn_matrix : numpy.ndarray
            The harmonic connectivity matrix between electrodes.
        """

        # Initialize biotuner object
        self.metric = metric
        data = self.data
        list_idx = list(range(len(data)))
        pairs = list(itertools.product(list_idx, list_idx))
        harm_conn_matrix = []
        if FREQ_BANDS is None:
            FREQ_BANDS = [
                [2, 3.55],
                [3.55, 7.15],
                [7.15, 14.3],
                [14.3, 28.55],
                [28.55, 49.4],
            ]
        if metric == "wPLI_multiband":
            harm_conn_matrix = np.zeros((len(FREQ_BANDS), len(data), len(data)))
        for i, pair in enumerate(pairs):
            data1 = data[pair[0]]
            data2 = data[pair[1]]
            # if i % (len(pairs) // 10) == 0:
            percentage_complete = int(i / len(pairs) * 100)
            #print(f"{percentage_complete}% complete")
            bt1 = compute_biotuner(
                self.sf,
                peaks_function=self.peaks_function,
                precision=self.precision,
                n_harm=self.n_harm,
            )
            bt1.peaks_extraction(
                data1,
                min_freq=self.min_freq,
                max_freq=self.max_freq,
                max_harm_freq=150,
                n_peaks=self.n_peaks,
                noverlap=None,
                nperseg=None,
                nfft=None,
                smooth_fft=1,
                FREQ_BANDS=FREQ_BANDS,
            )
            list1 = bt1.peaks
            bt2 = compute_biotuner(
                self.sf,
                peaks_function=self.peaks_function,
                precision=self.precision,
                n_harm=self.n_harm,
            )
            bt2.peaks_extraction(
                data2,
                min_freq=self.min_freq,
                max_freq=self.max_freq,
                max_harm_freq=150,
                n_peaks=self.n_peaks,
                noverlap=None,
                nperseg=None,
                nfft=None,
                smooth_fft=1,
                FREQ_BANDS=FREQ_BANDS,
            )

            list2 = bt2.peaks
            if metric == "subharm_tension":
                (
                    common_subs,
                    delta_t,
                    sub_tension_final,
                    harm_temp,
                    pairs_melody,
                ) = compute_subharmonics_2lists(
                    list1, list2, self.n_harm, delta_lim=delta_lim, c=2.1
                )
                #print(sub_tension_final)
                harm_conn_matrix.append(sub_tension_final)
            # compute the harmonic similarity between each pair of peaks from the two electrodes.
            if metric == "harmsim":
                harm_pairs = list(itertools.product(list1, list2))
                ratios = []
                for p in harm_pairs:
                    if p[0] > p[1]:
                        ratios.append(p[0] / p[1])
                    if p[1] > p[0]:
                        ratios.append(p[1] / p[0])
                ratios = rebound_list(ratios)
                harm_conn_matrix.append(np.mean(ratios2harmsim(ratios)))
            if metric == "RRCi":
                # Legacy path: mne FIR bandpass + Fraction.limit_denominator(max_denom_rrci)
                # ratio detection + |Im(<exp(i*(n*φ_i - m*φ_j))>)| coupling.
                # For a registry-based alternative with similar semantics, see
                # :meth:`compute_peak_phase_coupling_connectivity(coupling_metric='nm_rrci')`,
                # which uses butter bandpass + the resonance binary_nm_kernel
                # (max_nm=3 by default). Numerical results differ slightly due
                # to the different filter/ratio backends; this branch is kept
                # for backward compatibility with existing analyses.
                rrci_values = []
                for peak1 in list1:
                    for peak2 in list2:
                        rrci_value = cross_frequency_rrci(
                            data1, data2, self.sf, peak1, peak2, 2, max_denom_rrci
                        )
                        rrci_values.append(rrci_value)
                harm_conn_matrix.append(np.mean(rrci_values))
            if metric == "euler":
                list_all = list1 + list2
                list_all = [int(x * 10) for x in list_all]
                harm_conn_matrix.append(euler(*list_all))

            # to do
            """if metric == 'PPC_bicor':
                list_all = list1 + list2
                list_all = [int(x*10) for x in list_all]
                harm_conn_matrix.append(euler(*list_all))"""

            if metric == "harm_fit":
                harm_pairs = list(itertools.product(list1, list2))
                harm_fit = []
                for p in harm_pairs:
                    a, b, c, d = harmonic_fit(
                        p,
                        n_harm=self.n_harm,
                        bounds=0.5,
                        function="mult",
                        div_mode="div",
                        n_common_harms=2,
                    )
                    harm_fit.append(len(a))
                harm_conn_matrix.append(np.sum(harm_fit))

            if metric == "wPLI_crossfreq":
                # Legacy path: butter bandpass (bandwidth=1) + 1:1 PLV-like
                # |<exp(i(φ_i − φ_j))>| (despite the "wPLI" name, the legacy
                # implementation does NOT use the imaginary-part weighting that
                # defines wPLI in Vinck 2011). For the actual wPLI formula on
                # complex STFT/analytic signals, use
                # :meth:`compute_peak_phase_coupling_connectivity(coupling_metric='nm_wpli_complex')`.
                # This branch is kept for backward compatibility.
                wPLI_values = []
                for peak1 in list1:
                    for peak2 in list2:
                        wPLI_value = wPLI_crossfreq(data1, data2, peak1, peak2, self.sf)
                        wPLI_values.append(wPLI_value)
                harm_conn_matrix.append(np.mean(wPLI_values))

            if metric == "wPLI_multiband":
                wPLI_values = wPLI_multiband(data1, data2, FREQ_BANDS, self.sf)
                for idx, value in enumerate(wPLI_values):
                    harm_conn_matrix[idx][pair[0]][pair[1]] = value

            if metric == "MI":
                bandwidth = 1
                MI_values = []
                for peak1 in list1:
                    for peak2 in list2:
                        # Filter the original signals using the frequency bands
                        filtered_signal1 = butter_bandpass_filter(
                            data1, peak1 - bandwidth / 2, peak1 + bandwidth / 2, self.sf
                        )
                        filtered_signal2 = butter_bandpass_filter(
                            data2, peak2 - bandwidth / 2, peak2 + bandwidth / 2, self.sf
                        )

                        # Compute the instantaneous phase of each signal using the Hilbert transform
                        analytic_signal1 = hilbert(zscore(filtered_signal1))
                        analytic_signal2 = hilbert(zscore(filtered_signal2))
                        phase1 = np.angle(analytic_signal1)
                        phase2 = np.angle(analytic_signal2)

                        # Compute Mutual Information
                        MI_value = compute_mutual_information(phase1, phase2)
                        MI_values.append(MI_value)

                        harm_conn_matrix.append(np.mean(MI_values))

            if metric == "MI_spectral":
                # Create the pairs of peaks
                peak_pairs = list(itertools.product(list1, list2))

                # Compute the average MI value for the pairs of peaks
                avg_mi = MI_spectral(
                    data1,
                    data2,
                    self.sf,
                    self.min_freq,
                    self.max_freq,
                    self.precision,
                    peak_pairs,
                )
                harm_conn_matrix.append(avg_mi)

        if metric == "wPLI_multiband":
            matrix = harm_conn_matrix
        else:
            matrix = np.empty(shape=(len(data), len(data)))
            for e, p in enumerate(pairs):
                matrix[p[0]][p[1]] = harm_conn_matrix[e]
        # conn_matrix = matrix.astype('float')
        if graph is True:
            sbn.heatmap(matrix)
            # Add title and axis names
            plt.title(f"Harmonic connectivity matrix ({metric})")
            plt.show()
        self.conn_matrix = matrix
        return matrix


    # ------------------------------------------------------------------
    # Peak-based connectivity via the biotuner.resonance registry
    # ------------------------------------------------------------------

    def _extract_peaks_for_pair(self, data1, data2, FREQ_BANDS=None):
        """Extract peak lists for two signals using the class's peak settings.

        Returns (peaks1, peaks2). Used by the peak-based connectivity methods
        below to avoid duplicating the peak-extraction boilerplate.
        """
        if FREQ_BANDS is None:
            FREQ_BANDS = [
                [2, 3.55], [3.55, 7.15], [7.15, 14.3],
                [14.3, 28.55], [28.55, 49.4],
            ]
        bt1 = compute_biotuner(
            self.sf, peaks_function=self.peaks_function,
            precision=self.precision, n_harm=self.n_harm,
        )
        bt1.peaks_extraction(
            data1, min_freq=self.min_freq, max_freq=self.max_freq,
            max_harm_freq=150, n_peaks=self.n_peaks,
            noverlap=None, nperseg=None, nfft=None, smooth_fft=1,
            FREQ_BANDS=FREQ_BANDS,
        )
        bt2 = compute_biotuner(
            self.sf, peaks_function=self.peaks_function,
            precision=self.precision, n_harm=self.n_harm,
        )
        bt2.peaks_extraction(
            data2, min_freq=self.min_freq, max_freq=self.max_freq,
            max_harm_freq=150, n_peaks=self.n_peaks,
            noverlap=None, nperseg=None, nfft=None, smooth_fft=1,
            FREQ_BANDS=FREQ_BANDS,
        )
        return list(bt1.peaks), list(bt2.peaks)

    def _peak_pair_coupling(
        self, data1, data2, peak1, peak2,
        coupling_metric, coupling_metric_params,
        ratio_kernel_fn, ratio_kernel_params, bandwidth,
    ):
        """Compute scalar pairwise coupling between two signals at given peak
        frequencies. Bandpass-filters each signal in a `bandwidth`-Hz window
        around its peak, Hilbert-transforms to get analytic signal, then
        dispatches to the chosen pairwise metric.
        """
        # Resolve (n, m) for this peak pair via the ratio kernel
        W, N_mat, M_mat = ratio_kernel_fn(
            np.array([peak1]), np.array([peak2]), **ratio_kernel_params
        )
        if W[0, 0] <= 0:
            return 0.0
        n, m = int(N_mat[0, 0]), int(M_mat[0, 0])

        # Bandpass + Hilbert at each peak
        filtered1 = butter_bandpass_filter(
            data1, peak1 - bandwidth / 2, peak1 + bandwidth / 2, self.sf,
        )
        filtered2 = butter_bandpass_filter(
            data2, peak2 - bandwidth / 2, peak2 + bandwidth / 2, self.sf,
        )
        analytic1 = hilbert(zscore(filtered1))
        analytic2 = hilbert(zscore(filtered2))

        # Dispatch on input type — phase metrics take np.angle, analytic
        # metrics take the complex signal directly
        metric_fn = PAIRWISE_COUPLING_METRICS[coupling_metric]
        input_type = COUPLING_INPUT_TYPE[coupling_metric]
        if input_type == "phase":
            arg1, arg2 = np.angle(analytic1), np.angle(analytic2)
        else:  # 'analytic'
            arg1, arg2 = analytic1, analytic2
        value = metric_fn(arg1, arg2, n, m, **coupling_metric_params)
        # Weight by ratio-kernel membership (binary: 1.0; soft kernels: <=1)
        return float(W[0, 0]) * float(value)

    def compute_peak_phase_coupling_connectivity(
        self,
        coupling_metric="nm_plv",
        coupling_metric_params=None,
        ratio_kernel="binary",
        ratio_kernel_params=None,
        bandwidth=1.0,
        aggregate="mean",
        FREQ_BANDS=None,
        graph=True,
        save=False,
        savename="_",
    ):
        """Peak-based phase-coupling connectivity matrix.

        For each electrode pair: extract peaks from both signals, for every
        peak pair (f_i, f_j) determine the n:m harmonic ratio via the chosen
        ratio kernel, bandpass-filter both signals at their peak frequencies,
        Hilbert-transform to obtain analytic signals, and compute the chosen
        pairwise coupling metric. Aggregate over peak pairs to a scalar.

        Parameters
        ----------
        coupling_metric : str, default='nm_plv'
            Name of any registered pairwise coupling metric. Options include:
            ``'nm_plv'``, ``'nm_pli'``, ``'nm_wpli'``, ``'nm_rrci'``,
            ``'nm_plv_canonical'``, ``'nm_wpli_complex'``. See
            :mod:`biotuner.resonance.coupling` for definitions.
        coupling_metric_params : dict, optional
            Extra kwargs passed to the metric function.
        ratio_kernel : str, default='binary'
            Name of registered ratio kernel. ``'binary'`` is the legacy n:m gate.
        ratio_kernel_params : dict, optional
            Kwargs for the ratio kernel; defaults to
            ``{'max_nm': 3, 'tolerance': 0.05, 'fallback_to_1_1': True}``.
        bandwidth : float, default=1.0
            Hz width of the bandpass filter applied around each peak.
        aggregate : {'mean', 'max', 'sum'}, default='mean'
            How to reduce the per-peak-pair scalars to one number per electrode pair.
        FREQ_BANDS : list of [low, high], optional
            Override the default frequency-band partition used by peak extraction.
        graph : bool, default=True
            If True, displays a heatmap of the result.
        save, savename : passthrough to plotting.

        Returns
        -------
        matrix : ndarray (n_elec, n_elec)
            Pairwise scalar coupling values.
        """
        if coupling_metric not in PAIRWISE_COUPLING_METRICS:
            raise ValueError(
                f"Unknown coupling_metric {coupling_metric!r}. "
                f"Available: {sorted(PAIRWISE_COUPLING_METRICS)}"
            )
        if ratio_kernel not in RATIO_KERNELS:
            raise ValueError(
                f"Unknown ratio_kernel {ratio_kernel!r}. "
                f"Available: {sorted(RATIO_KERNELS)}"
            )
        if aggregate not in {"mean", "max", "sum"}:
            raise ValueError(f"aggregate must be mean/max/sum, got {aggregate!r}")
        if coupling_metric_params is None:
            coupling_metric_params = {}
        if ratio_kernel_params is None:
            ratio_kernel_params = {"max_nm": 3, "tolerance": 0.05, "fallback_to_1_1": True}

        ratio_fn = RATIO_KERNELS[ratio_kernel]
        agg_fn = {"mean": np.mean, "max": np.max, "sum": np.sum}[aggregate]

        data = self.data
        n_elec = len(data)
        pairs = list(itertools.product(range(n_elec), range(n_elec)))
        matrix = np.zeros((n_elec, n_elec), dtype=np.float64)

        for (i, j) in pairs:
            data1, data2 = data[i], data[j]
            peaks1, peaks2 = self._extract_peaks_for_pair(data1, data2, FREQ_BANDS=FREQ_BANDS)
            if not peaks1 or not peaks2:
                continue
            values = []
            for p1 in peaks1:
                for p2 in peaks2:
                    values.append(self._peak_pair_coupling(
                        data1, data2, p1, p2,
                        coupling_metric, coupling_metric_params,
                        ratio_fn, ratio_kernel_params, bandwidth,
                    ))
            matrix[i, j] = float(agg_fn(values)) if values else 0.0

        if graph:
            sbn.heatmap(matrix)
            plt.title(f"Peak phase-coupling connectivity ({coupling_metric}, agg={aggregate})")
            if save:
                plt.savefig(f"peak_phase_coupling_{savename}.png")
            plt.show()
        self.peak_phase_coupling_matrix = matrix
        return matrix

    def compute_peak_resonance_connectivity(
        self,
        harm_metric="harmsim",
        coupling_metric="nm_plv",
        combine="product",
        coupling_metric_params=None,
        ratio_kernel="binary",
        ratio_kernel_params=None,
        bandwidth=1.0,
        coupling_aggregate="mean",
        delta_lim=20,
        FREQ_BANDS=None,
        graph=True,
        save=False,
        savename="_",
    ):
        """Peak-based resonance connectivity: H × PC per electrode pair.

        Combines a peak-based harmonicity scalar (from :meth:`compute_harm_connectivity`)
        and a peak-based phase-coupling scalar (from
        :meth:`compute_peak_phase_coupling_connectivity`) via a registered
        combine rule. Default ``combine='product'`` gives the standard H · PC
        resonance, analogous to the per-frequency R(f) = H(f) · PC(f) in the
        single-channel framework.

        Parameters
        ----------
        harm_metric : str, default='harmsim'
            Harmonic-similarity metric for the H scalar
            (see :meth:`compute_harm_connectivity`).
        coupling_metric : str, default='nm_plv'
            Phase-coupling metric for the PC scalar
            (see :meth:`compute_peak_phase_coupling_connectivity`).
        combine : str, default='product'
            Name in :data:`biotuner.resonance.registry.COMBINE_RULES`.
            Options: ``'product'`` (legacy), ``'geomean'``, ``'harmmean'``,
            ``'min'``, ``'weighted_log'``.

        Other parameters are forwarded to the two underlying methods.

        Returns
        -------
        matrix : ndarray (n_elec, n_elec)
        """
        if combine not in COMBINE_RULES:
            raise ValueError(
                f"Unknown combine rule {combine!r}. Available: {sorted(COMBINE_RULES)}"
            )

        # Reuse existing H scalar (turn off its own plot)
        H_matrix = self.compute_harm_connectivity(
            metric=harm_metric, delta_lim=delta_lim,
            save=False, savename=savename, graph=False,
            FREQ_BANDS=FREQ_BANDS,
        )
        # Many H metrics (e.g., harmsim) return values on a 0-100 scale;
        # normalize to [0, 1] before combining so the product is meaningful
        # against PC ∈ [0, 1]. NaN can arise from the legacy
        # compute_harm_connectivity for self-pairs when peak-extraction yields
        # too few peaks for any cross-peak ratio (mean of empty list).
        # Replace NaN with 0 so the combine rule doesn't propagate it.
        H_norm = np.asarray(H_matrix, dtype=np.float64)
        H_norm = np.nan_to_num(H_norm, nan=0.0, posinf=0.0, neginf=0.0)
        H_max = np.max(np.abs(H_norm))
        if H_max > 0:
            H_norm = H_norm / H_max

        PC_matrix = self.compute_peak_phase_coupling_connectivity(
            coupling_metric=coupling_metric,
            coupling_metric_params=coupling_metric_params,
            ratio_kernel=ratio_kernel,
            ratio_kernel_params=ratio_kernel_params,
            bandwidth=bandwidth, aggregate=coupling_aggregate,
            FREQ_BANDS=FREQ_BANDS, graph=False,
        )
        PC_norm = np.asarray(PC_matrix, dtype=np.float64)

        combine_fn = COMBINE_RULES[combine]
        R_matrix = combine_fn([H_norm, PC_norm])

        if graph:
            sbn.heatmap(R_matrix)
            plt.title(f"Peak resonance connectivity ({harm_metric} × {coupling_metric}, combine={combine})")
            if save:
                plt.savefig(f"peak_resonance_{savename}.png")
            plt.show()
        self.peak_resonance_matrix = R_matrix
        return R_matrix


    # ------------------------------------------------------------------
    # Layer C — spectrum-based cross-resonance connectivity matrix
    # ------------------------------------------------------------------

    def compute_cross_resonance_connectivity(
        self,
        config=None,
        factor="R",
        flavor="all",
        aggregate="peak_to_median",
        graph=True,
        save=False,
        savename="_",
        store_full_results=False,
    ):
        """Cross-resonance connectivity matrix: loop compute_cross_resonance
        over every electrode pair and reduce each result to a scalar.

        Parameters
        ----------
        config : ResonanceConfig, optional
            Forwarded to :func:`compute_cross_resonance`. Defaults to the
            refined cross-channel config (joint PC + n:m kernel).
        factor : {'H', 'PC', 'R'}, default 'R'
            Which factor's spectrum to reduce to a scalar.
        flavor : {'1to2', '2to1', 'all'}, default 'all'
            Which of the 3 reducer flavors to use.
        aggregate : str, default ``'peak_to_median'``
            How to reduce the per-bin spectrum to a single number per electrode
            pair. Simple aggregates inherit the joint-probability reducer's
            broadband-power bias (any channel with broadband content ranks
            high); normalized aggregates measure peak structure ABOVE the
            spectrum's noise floor:

              ``'max'`` — peak value. Caveat: rewards broadband-power channels
                (e.g. pink noise) because the joint-probability reducer gives
                large weighted sums everywhere the partner channel has power.
              ``'mean'`` — average. Dilutes peaks; same broadband bias.
              ``'sum'`` — total area. Scales with bandwidth.
              ``'peak'`` — value at the prominence-detected peak frequency.
                Mildly better but still inherits H's broadband bias.
              ``'peak_to_median'`` — ``log10(max / median)``. Log-scale
                peak-to-floor ratio; robust to broadband channels. **DEFAULT.**
              ``'peak_over_median'`` — ``(max - median) / (max + median)``.
                DEPRECATED — saturates at 1.0 for any spectrum with a sharp
                peak relative to its baseline; prefer ``peak_to_median``.
              ``'spectral_concentration'`` — fraction of energy in the top-3
                bins. Sharp peaks → high; diffuse spectra → low.
              ``'peak_z'`` — z-score of the dominant peak relative to the
                off-peak distribution (all detected peaks masked out, so
                multi-peak spectra are NOT penalized). Often inflates
                multi-peak spectra; use ``peak_to_median`` for the cleanest
                broadband-vs-focal discrimination.

        graph : bool
            Heatmap if True.
        store_full_results : bool, default False
            If True, also stores ``self.cross_resonance_results`` as a list of
            (i, j, CrossResonanceResult) for downstream graph analysis. Costly
            for large electrode counts (stores n_elec² ResonanceResults).

        Returns
        -------
        matrix : ndarray (n_elec, n_elec)
            The scalar connectivity matrix.

        Notes
        -----
        There is no single scalar aggregate of R(f) that cleanly separates
        true phase-coupling from broadband-power overlap — H's joint p1·p2
        reducer rewards any partner power overlap, so a focal signal paired
        with broadband noise will produce sharp peaks in R(f) at the focal
        channel's frequency. For paper-quality discrimination of true
        cross-channel phase coupling vs broadband artifact, use
        :meth:`compute_cross_resonance_connectivity_zscore` with
        ``surrogate_kind='iaaft'`` — IAAFT surrogates preserve per-channel PSD
        while destroying cross-channel phase, so a high z-score signals
        genuine phase coupling above the broadband null.

        The default ``aggregate='peak_to_median'`` was chosen empirically on
        the 6-channel validation dataset as the cleanest scalar discriminator:
        locked alpha pair > 1:2 harmonic pair > drifting alpha > pink-noise
        pair, in the expected order. ``peak_z`` has a tendency to inflate
        multi-peak spectra and is offered as an alternative.
        """
        valid_factors = {"H", "PC", "R"}
        valid_flavors = {"1to2", "2to1", "all"}
        valid_aggs = {
            "max", "mean", "sum", "peak",
            "peak_to_median", "peak_over_median",
            "spectral_concentration", "peak_z",
        }
        if factor not in valid_factors:
            raise ValueError(f"factor must be one of {valid_factors}, got {factor!r}")
        if flavor not in valid_flavors:
            raise ValueError(f"flavor must be one of {valid_flavors}, got {flavor!r}")
        if aggregate not in valid_aggs:
            raise ValueError(f"aggregate must be one of {valid_aggs}, got {aggregate!r}")

        data = self.data
        n_elec = len(data)
        matrix = np.zeros((n_elec, n_elec), dtype=np.float64)
        full_results = [] if store_full_results else None

        for i in range(n_elec):
            for j in range(n_elec):
                if i == j:
                    matrix[i, j] = np.nan  # self-pair undefined for cross
                    continue
                result = compute_cross_resonance(data[i], data[j], sf=self.sf, config=config)
                # Resolve the spectrum to summarize
                if factor == "R":
                    spec = result.resonance_spectrum[flavor]
                else:
                    spec = result.factors[factor][flavor]
                # Reduce to scalar
                matrix[i, j] = _scalar_aggregate(spec, result, factor, aggregate)

                if store_full_results:
                    full_results.append((i, j, result))

        if graph:
            sbn.heatmap(matrix)
            plt.title(f"Cross-resonance connectivity (factor={factor}[{flavor}], agg={aggregate})")
            if save:
                plt.savefig(f"cross_resonance_conn_{savename}.png")
            plt.show()
        self.cross_resonance_matrix = matrix
        if store_full_results:
            self.cross_resonance_results = full_results
        return matrix

    def compute_cross_resonance_connectivity_zscore(
        self,
        config=None,
        factor="R",
        flavor="all",
        aggregate="max",
        surrogate_kind="iaaft",
        n_surrogates=50,
        rng_seed=42,
        graph=True,
    ):
        """Surrogate-normalized z-scored cross-resonance connectivity matrix.

        For each surrogate, the multichannel data is replaced by independently
        surrogated copies of each channel, the n_elec × n_elec connectivity
        matrix is computed, and the surrogate mean/std is collected per cell.
        The observed matrix is then z-scored against the surrogate distribution.

        Parameters
        ----------
        surrogate_kind : {'phase_randomize', 'iaaft', 'time_shuffle'}
            Which surrogate generator to use (see :mod:`biotuner.resonance.nulls`).
            Default ``'iaaft'`` — tightest cross-channel null per Fig 30.
        n_surrogates : int, default 50
            Number of surrogate matrices.
        Other args forwarded to :meth:`compute_cross_resonance_connectivity`.

        Returns
        -------
        observed : ndarray (n_elec, n_elec)
        z_matrix : ndarray (n_elec, n_elec)
        p_matrix : ndarray (n_elec, n_elec)   one-sided empirical p-value
        """
        from biotuner.resonance.nulls import CROSS_SURROGATE_GENERATORS

        if surrogate_kind not in CROSS_SURROGATE_GENERATORS:
            raise ValueError(
                f"surrogate_kind must be one of {sorted(CROSS_SURROGATE_GENERATORS)}, "
                f"got {surrogate_kind!r}"
            )
        gen = CROSS_SURROGATE_GENERATORS[surrogate_kind]

        # Observed matrix
        observed = self.compute_cross_resonance_connectivity(
            config=config, factor=factor, flavor=flavor, aggregate=aggregate, graph=False,
        )

        # Surrogate matrices
        master_rng = np.random.default_rng(rng_seed)
        n_elec = len(self.data)
        surr_stack = np.empty((n_surrogates, n_elec, n_elec), dtype=np.float64)
        orig_data = self.data
        try:
            for k in range(n_surrogates):
                surr_data = np.empty_like(orig_data)
                for c in range(n_elec):
                    surr_data[c] = gen(
                        orig_data[c],
                        np.random.default_rng(master_rng.integers(0, 2**31)),
                    )
                self.data = surr_data
                surr_stack[k] = self.compute_cross_resonance_connectivity(
                    config=config, factor=factor, flavor=flavor, aggregate=aggregate, graph=False,
                )
        finally:
            self.data = orig_data

        mu = np.nanmean(surr_stack, axis=0)
        sd = np.nanstd(surr_stack, axis=0) + 1e-12
        z_matrix = (observed - mu) / sd
        # Empirical one-sided p-value (with NaN-safe sum)
        p_matrix = (np.nansum(surr_stack >= observed[None, :, :], axis=0) + 1) / (n_surrogates + 1)

        if graph:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
            sbn.heatmap(observed, ax=axes[0]); axes[0].set_title(f"Observed ({factor}[{flavor}])")
            sbn.heatmap(z_matrix, ax=axes[1], cmap="coolwarm", center=0); axes[1].set_title(f"z-score ({surrogate_kind})")
            sbn.heatmap(p_matrix, ax=axes[2], cmap="viridis_r"); axes[2].set_title("Empirical p")
            plt.tight_layout()
            plt.show()

        self.cross_resonance_z_matrix = z_matrix
        self.cross_resonance_p_matrix = p_matrix
        return observed, z_matrix, p_matrix


    def compute_IMF_correlation(self, nIMFs=5, freq_range=(1, 60), precision=0.5, delta_lim=50):
        """
        Compute the correlation, coherence, and peak frequency between each pair of IMFs for each pair of electrodes.
        
        Returns
        -------
        df : pd.DataFrame
            DataFrame with columns: elec1, elec2, imf1, imf2, pearson, coherence, and peak_freq.
        """
        
        num_electrodes, _ = self.data.shape
        results = []

        # Pre-compute IMFs for each electrode
        all_IMFs = []
        peak_frequencies = []  # Store peak frequencies for each IMF

        for elec in range(num_electrodes):
            IMFs = EMD_eeg(self.data[elec, :], method="EMD")
            IMFs = IMFs[:nIMFs]
            all_IMFs.append(IMFs)

            # Compute peak frequencies for each IMF once
            elec_peak_freqs = []
            for imf in IMFs:
                f_welch, Pxx = welch(imf, self.sf, nperseg=int(self.sf/precision))
                peak_freq = f_welch[np.argmax(Pxx)]
                elec_peak_freqs.append(peak_freq)
            peak_frequencies.append(elec_peak_freqs)

        print('shape all IMFS', np.array(all_IMFs).shape)
        
        # Iterate over pairs of electrodes
        for elec1 in range(num_electrodes):
            for elec2 in range(num_electrodes):
                num_IMFs_elec1 = len(all_IMFs[elec1])
                num_IMFs_elec2 = len(all_IMFs[elec2])
                
                # Compute correlation and coherence between IMFs of the two electrodes
                for imf1 in range(num_IMFs_elec1):
                    for imf2 in range(num_IMFs_elec2):
                        corr, _ = pearsonr(all_IMFs[elec1][imf1], all_IMFs[elec2][imf2])
                        f, Cxy = coherence(all_IMFs[elec1][imf1], all_IMFs[elec2][imf2], fs=self.sf)
                        
                        # Filter for desired frequency range
                        filtered_coherence = Cxy[(f >= freq_range[0]) & (f <= freq_range[1])]
                        mean_coherence = np.mean(filtered_coherence)
                        
                        # Retrieve the precomputed peak frequency for the IMF
                        peak_freq1 = peak_frequencies[elec1][imf1]
                        peak_freq2 = peak_frequencies[elec2][imf2]

                        peak1 = peak_frequencies[elec1][imf1]
                        peak2 = peak_frequencies[elec2][imf2]
                        if peak1 >= peak2:
                            harmsim = dyad_similarity(peak1/peak2)
                        if peak2 > peak1:
                            harmsim = dyad_similarity(peak2/peak1)
                            
                        # compute subharmonic tension
                        _, _, subharm_tension, _ = compute_subharmonic_tension([peak1, peak2], n_harmonics=10,
                                                                               delta_lim=delta_lim, min_notes=2)
                        results.append({
                            "elec1": elec1,
                            "elec2": elec2,
                            "imf1": imf1,
                            "imf2": imf2,
                            "pearson": corr,
                            "coherence": mean_coherence,
                            "peak_freq1": peak_freq1,
                            "peak_freq2": peak_freq2,
                            'harmsim': harmsim,
                            'subharm_tension': subharm_tension
                        })

        df = pd.DataFrame(results)
        return df

    def compute_time_resolved_harm_connectivity(
        self, sf, nIMFs, metric="harmsim", delta_lim=50
    ):
        """
        Computes the time-resolved harmonic connectivity matrix between electrodes,
        which is a harmonic connectivity matrix for each intrinsic mode function (IMF),
        and each time point.

        Parameters
        ----------
        data : numpy.ndarray
            Input data with shape (num_electrodes, numDataPoints)
        sf : int
            Sampling frequency
        nIMFs : int
            Number of intrinsic mode functions (IMFs) to consider.
        metric : str, optional
            The metric to use for computing harmonic connectivity. Default is 'harmsim'.
        delta_lim : int, optional
            The delta limit for the subharmonic tension metric. Default is 20.

        Returns
        -------
        connectivity_matrices : numpy.ndarray
            Time-resolved harmonic connectivity matrices with shape (IMFs, numDataPoints, electrodes, electrodes).
            
        Notes
        -----
        !!! This method is very computationally expensive and can take a long time to run. !!!
        """
        data = self.data
        num_electrodes, numDataPoints = data.shape
        connectivity_matrices = np.zeros(
            (nIMFs, numDataPoints, num_electrodes, num_electrodes)
        )
        harmonicity_cache = {}

        for imf in range(nIMFs):
            for t in range(numDataPoints):
                for i in range(num_electrodes):
                    for j in range(i + 1, num_electrodes):
                        pair_key = (i, j)
                        if pair_key not in harmonicity_cache:
                            time_series1 = data[i, :]
                            time_series2 = data[j, :]
                            harmonicity_cache[pair_key] = EMD_time_resolved_harmonicity(
                                time_series1,
                                time_series2,
                                sf,
                                nIMFs=nIMFs,
                                method=metric,
                            )

                        harmonicity = harmonicity_cache[pair_key]
                        connectivity_matrices[imf, t, i, j] = harmonicity[t, imf]
                        connectivity_matrices[imf, t, j, i] = harmonicity[t, imf]

        return connectivity_matrices

    def transitional_connectivity(
        self,
        data=None,
        sf=None,
        mode="win_overlap",
        overlap=10,
        delta_lim=20,
        graph=False,
        n_trans_harm=3,
    ):
        """
        This function calculates the transitional connectivity among electrodes, utilizing concepts of transitional harmony and temporal correlation.
        It does so by employing a two-step approach:

        - Transitional Harmony Calculation: For every electrode, the transitional harmony is determined first.
                                            This measure reflects the patterns or sequences of signal transitions over time
                                            that each electrode experiences.

        - Temporal Correlation with FDR Correction: After obtaining transitional harmonies, the function evaluates the temporal correlation
                                                    between these harmonies for each pair of electrodes. Temporal correlation quantifies
                                                    how similar the timing and pattern of transitional harmonies are between each pair of electrodes.

        The entire process takes into account multiple comparisons, utilizing False Discovery Rate (FDR) correction
        to reduce the likelihood of false positives. This correction is crucial in maintaining the validity and robustness of the results,
        especially when dealing with a large number of comparisons, as is the case with numerous electrodes.

        Parameters
        ----------
        data : numpy.ndarray
            Multichannel EEG data with shape (n_electrodes, n_timepoints).
        sf : float
            Sampling frequency of the EEG data in Hz.
        mode : str, optional, default='win_overlap'
            The mode to compute the transitional harmony.
            'win_overlap' computes the transitional harmony using a sliding window with overlap.
        overlap : int, optional, default=10
            The percentage of overlap between consecutive windows when computing
            the transitional harmony. Default is 10.
        delta_lim : int, optional, default=20
            The minimum delta value for the computation of
            subharmonic tension, in ms. Default is 20.
        graph : bool, optional, default=False
            If True, it will plot the graph of the transitional harmony. Default is False.
        n_trans_harm : int, optional, default=3
            Number of transitional harmonics to compute. Default is 3.

        Returns
        -------
        conn_mat : numpy.ndarray
            Connectivity matrix of shape (n_electrodes, n_electrodes) representing
            the transitional connectivity between electrodes.
        pval_mat : numpy.ndarray
            P-value matrix of shape (n_electrodes, n_electrodes) with FDR-corrected
            p-values for the computed connectivity values.
        """
        if sf is None:
            sf = self.sf
        if data is None:
            data = self.data
        trans_subharm_tot = []
        n_electrodes = data.shape[0]
        for elec in range(n_electrodes):
            th = transitional_harmony(
                sf=self.sf,
                data=data[elec, :],
                peaks_function=self.peaks_function,
                precision=self.precision,
                n_harm=self.n_harm,
                harm_function="mult",
                min_freq=self.min_freq,
                max_freq=self.max_freq,
                n_peaks=self.n_peaks,
                n_trans_harm=n_trans_harm,
            )
            trans_subharm, time_vec_final, pairs_melody = th.compute_trans_harmony(
                mode="win_overlap", overlap=overlap, delta_lim=delta_lim, graph=graph
            )
            trans_subharm_tot.append(trans_subharm)
        subharm = np.array(trans_subharm_tot)
        conn_mat, pval_mat = temporal_correlation_fdr(subharm)
        return conn_mat, pval_mat, subharm

    def plot_conn_matrix(self, conn_matrix=None, node_names=None, n_lines=50):
        """
        Plots a connectivity matrix in a circle plot.

        Parameters
        ----------
        conn_matrix : ndarray, optional
            The connectivity matrix to plot. If None, uses the object's own attribute `conn_matrix`. 
            If `conn_matrix` is still None, a ValueError will be raised.
        node_names : list, optional
            The labels for the nodes. If None, a range object will be converted to a list of string values for node names.
        n_lines : int, default=50
            The number of lines to draw in the plot. Default is 50.

        Returns
        -------
        fig : matplotlib.figure.Figure
            A matplotlib figure object.

        Raises
        ------
        ValueError
            Raised when no connectivity matrix is found.

        Notes
        -----
        This method uses the function `plot_connectivity_circle` to plot the connectivity matrix.
        """

        if conn_matrix is None:
            conn_matrix = self.conn_matrix
            # Raise error if conn_matrix is still None
            if conn_matrix is None:
                raise ValueError("No connectivity matrix found.")
        if node_names is None:
            node_names = range(0, len(conn_matrix), 1)
            node_names = [str(x) for x in node_names]
        # Lazy imports — mne.viz and mne_connectivity are optional deps used
        # only by this plotting helper.
        from mne.viz import circular_layout  # noqa: F401 (used by callers via mne)
        from mne_connectivity.viz import plot_connectivity_circle
        fig = plot_connectivity_circle(
            conn_matrix,
            node_names=node_names,
            n_lines=n_lines,
            fontsize_names=24,
            show=False,
            vmin=0.0,
        )

    def compute_harmonic_spectrum_connectivity(
        self,
        sf=None,
        data=None,
        precision=0.5,
        fmin=None,
        fmax=None,
        noverlap=1,
        power_law_remove=False,
        n_peaks=5,
        metric="harmsim",
        n_harms=10,
        delta_lim=0.1,
        min_notes=2,
        plot=False,
        smoothness=1,
        smoothness_harm=1,
        phase_mode=None,
        save_fig=False,
        savename="harmonic_spectrum_connectivity.png",
    ):
        """
        Computes the harmonic spectrum connectivity between pairs of electrodes. For more details on the harmonic spectrum, 
        see the function :func:`compute_cross_spectrum_harmonicity`.

        Parameters
        ----------
        sf : float, optional
            Sampling frequency of the data. If not provided, uses the object's own attribute `sf`.
        data : ndarray, optional
            2D data array of shape (n_electrodes, n_datapoints). If not provided, uses the object's own attribute `data`.
        precision : float, default=0.5
            Precision of the frequency axis of the cross-spectrum.
        fmin : float, optional
            Minimum frequency for computation. If not provided, uses the smallest possible frequency (i.e., zero).
        fmax : float, optional
            Maximum frequency for computation. If not provided, uses the Nyquist frequency.
        noverlap : int, default=1
            Argument passed to the :func:`scipy.signal.stft` function.
        power_law_remove : bool, default=False
            If True, removes the power-law noise from the cross-spectrum.
        n_peaks : int, default=5
            Number of peaks to derive from the harmonic spectrum.
        metric : str, default='harmsim'
            Name of the metric to be used in the computation of harmonicity.
            Choose between:
            
            - 'harmsim'
            - 'subharm_tension'
        n_harms : int, default=10
            Number of harmonics to consider in the computation.
        delta_lim : float, default=0.1
            Threshold for the delta limit used in subharmonic tension computation.
        min_notes : int, default=2
            Minimum number of notes required for a harmonic pattern to be considered valid,
            when subharmonic tension is used as the metric.
        plot : bool, default=False
            If True, plots the results.
        smoothness : int, default=1
            Smoothness factor of the power spectrum.
            When smoothness=1, no smoothing is applied.
        smoothness_harm : int, default=1
            Smoothness factor of the harmonic spectrum.
            When smoothness_harm=1, no smoothing is applied.
        phase_mode : str, optional
            If set to 'weighted', the phase coupling is weighted by the power of associated frequencies.
        save_fig : bool, default=False
            If True, saves the resulting plot as a .png file.
        savename : str, default='harmonic_spectrum_connectivity.png'
            Name of the .png file to save if `save_fig` is True.

        Returns
        -------
        output : pandas.DataFrame
            DataFrame containing the results of the harmonic spectrum connectivity computation.

        Notes
        -----
        The harmonic spectrum connectivity is computed between each pair of electrodes. This is achieved
        by computing the cross-spectrum harmonicity for each pair of electrodes, and storing the results
        in a DataFrame. The DataFrame includes indices of the electrode pairs, alongside the results of
        the harmonicity computation.
        """

        if sf is None:
            sf = self.sf
        if data is None:
            data = self.data
        electrodes, datapoints = data.shape
        dfs = []  # Will store the generated DataFrames

        for i in range(electrodes):
            for j in range(electrodes):
                if i != j:
                    signal1 = data[i]
                    signal2 = data[j]
                    df = compute_cross_spectrum_harmonicity(
                        signal1,
                        signal2,
                        precision_hz=precision,
                        fmin=fmin,
                        fmax=fmax,
                        noverlap=noverlap,
                        fs=sf,
                        power_law_remove=power_law_remove,
                        n_peaks=n_peaks,
                        metric=metric,
                        n_harms=n_harms,
                        delta_lim=delta_lim,
                        min_notes=min_notes,
                        plot=plot,
                        smoothness=smoothness,
                        smoothness_harm=smoothness_harm,
                        phase_mode=phase_mode,
                        save_fig=save_fig,
                        save_name=savename,
                    )
                    df["elec1"] = i  # Add electrode indices to DataFrame
                    df["elec2"] = j
                    dfs.append(df)

        output = pd.concat(dfs, ignore_index=True)  # Concatenate all the dataframes
        self.harmonic_spectrum_connectivity = output
        return output

    def get_harm_spectrum_metric_matrix(self, metric):
        """
        Method to retrieve a matrix of harmonic spectrum metric values between pairs of electrodes.

        Parameters
        ----------
        metric : str
            The specific metric from the harmonic spectrum connectivity data to create the matrix from.

        Returns
        -------
        matrix : pandas.DataFrame or None
            A pivot table with 'elec1' and 'elec2' as indices and the provided 'metric' as values. 
            If 'harmonic_spectrum_connectivity' is None or 'metric' is not found in 'harmonic_spectrum_connectivity',
            this method will print an error message and return None.

        Notes
        -----
        This method checks if 'harmonic_spectrum_connectivity' exists in the object.
        If it does, it further checks if 'metric' exists in the DataFrame's columns. If both checks pass,
        a pivot table is created using 'elec1' and 'elec2' as indices and 'metric' as values. 
        """

        # Check if self.harmonic_spectrum_connectivity exists
        if self.harmonic_spectrum_connectivity is None:
            print("Error: No harmonic_spectrum_connectivity found.")
            return None

        # Check if the metric exists in the DataFrame
        if metric not in self.harmonic_spectrum_connectivity.columns:
            print(f"Error: {metric} not found in harmonic_spectrum_connectivity.")
            return None

        # Create a pivot table with 'elec1' and 'elec2' as indices and 'metric' as values
        matrix = self.harmonic_spectrum_connectivity.pivot(
            index="elec1", columns="elec2", values=metric
        )

        return matrix
    
    def plot_IMF_correlation_matrix(self, df, elec1, elec2, variable='pearson', savepath=None):
        """
        Plot a heatmap of correlations between IMFs for specified electrodes.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the correlation data with columns: elec1, elec2, imf1, imf2, and pearson.
        elec1 : int
            The first electrode of interest.
        elec2 : int
            The second electrode of interest.
        
        Returns
        -------
        None
        """
        
        # Filter the dataframe for the specified electrodes
        subset = df[(df['elec1'] == elec1) & (df['elec2'] == elec2)]
        
        # Find the number of unique IMFs
        num_IMFs = max(len(subset['imf1'].unique()), len(subset['imf2'].unique()))
        
        # Create an empty matrix for correlations
        corr_matrix = np.zeros((num_IMFs, num_IMFs))
        
        # Fill the matrix with correlation values
        for _, row in subset.iterrows():
            corr_matrix[int(row['imf1']), int(row['imf2'])] = row[variable]

        # Determine the colorbar range
        max_corr = subset[variable].abs().max()
        vmin, vmax = -max_corr, max_corr
        
        # Plot the heatmap
        plt.figure(figsize=(5, 4))
        sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, vmin=vmin, vmax=vmax)
        plt.title(f'Correlation between IMFs for Electrode {elec1} and Electrode {elec2}')
        plt.xlabel(f'IMFs of Electrode {elec2}')
        plt.ylabel(f'IMFs of Electrode {elec1}')
        if savepath is not None:
            plt.savefig(savepath)
        plt.show()


def wPLI_crossfreq(signal1, signal2, peak1, peak2, sf):
    """
    Computes the weighted phase lag index (wPLI) between two signals in specified frequency bands, centered around
    provided peak frequencies.

    Parameters
    ----------
    signal1 : ndarray
        First signal in time series.
    signal2 : ndarray
        Second signal in time series.
    peak1 : float
        Center of the frequency band for the first signal.
    peak2 : float
        Center of the frequency band for the second signal.
    sf : float
        Sampling frequency.

    Returns
    -------
    wPLI : float
        Weighted phase lag index between two signals in specified frequency bands.

    Examples
    --------
    >>> signal1 = np.random.normal(0, 1, 5000)
    >>> signal2 = np.random.normal(0, 1, 5000)
    >>> peak1 = 10.0
    >>> peak2 = 20.0
    >>> sf = 100.0
    >>> wPLI = wPLI_crossfreq(signal1, signal2, peak1, peak2, sf)
    """
    # Define a band around each peak
    bandwidth = 1  # You can adjust the bandwidth as needed

    # Filter the original signals using the frequency bands
    filtered_signal1 = butter_bandpass_filter(
        signal1, peak1 - bandwidth / 2, peak1 + bandwidth / 2, sf
    )
    filtered_signal2 = butter_bandpass_filter(
        signal2, peak2 - bandwidth / 2, peak2 + bandwidth / 2, sf
    )

    # Compute the wPLI between the filtered signals
    n_samples = len(filtered_signal1)
    analytic_signal1 = hilbert(zscore(filtered_signal1))
    analytic_signal2 = hilbert(zscore(filtered_signal2))

    phase_diff = np.angle(analytic_signal1) - np.angle(analytic_signal2)
    wPLI = np.abs(np.mean(np.exp(1j * phase_diff)))

    return wPLI


def wPLI_multiband(signal1, signal2, freq_bands, sf):
    """
    Computes the weighted phase lag index (wPLI) between two signals for multiple frequency bands.

    Parameters
    ----------
    signal1 : ndarray
        First signal in time series.
    signal2 : ndarray
        Second signal in time series.
    freq_bands : list of tuple
        List of frequency bands. Each band is represented as a tuple (lowcut, highcut).
    sf : float
        Sampling frequency.

    Returns
    -------
    wPLI_values : list of float
        List of wPLI values for each frequency band.

    Examples
    --------
    >>> signal1 = np.random.normal(0, 1, 5000)
    >>> signal2 = np.random.normal(0, 1, 5000)
    >>> freq_bands = [(8, 12), (13, 30), (30, 70)]
    >>> sf = 100.0
    >>> wPLI_values = wPLI_multiband(signal1, signal2, freq_bands, sf)
    """
    wPLI_values = []

    for band in freq_bands:
        lowcut, highcut = band

        # Filter the original signals using the frequency bands
        filtered_signal1 = butter_bandpass_filter(signal1, lowcut, highcut, sf)
        filtered_signal2 = butter_bandpass_filter(signal2, lowcut, highcut, sf)

        # Compute the wPLI between the filtered signals
        n_samples = len(filtered_signal1)
        analytic_signal1 = hilbert(zscore(filtered_signal1))
        analytic_signal2 = hilbert(zscore(filtered_signal2))

        phase_diff = np.angle(analytic_signal1) - np.angle(analytic_signal2)
        wPLI = np.abs(np.mean(np.exp(1j * phase_diff)))
        print(wPLI)
        wPLI_values.append(wPLI)

    return wPLI_values

def n_m_phase_locking(signal1, signal2, n, m, sf):
    """
    Calculate n:m phase locking between two signals.
    
    :param signal1: First time series.
    :param signal2: Second time series.
    :param n: Multiplicative factor for the first signal.
    :param m: Multiplicative factor for the second signal.
    :param sf: Sampling frequency of the signals.
    :return: Mean resultant length (Rn:m) indicating the phase locking value.
    """
    # Calculate the instantaneous phase for each signal using the Hilbert transform
    phase1 = np.angle(hilbert(signal1))
    phase2 = np.angle(hilbert(signal2))
    
    # Calculate the n:m phase difference
    phase_diff_nm = n * phase1 - m * phase2
    
    # Calculate the unitary vectors whose angle is the instantaneous phase difference
    unit_vectors = np.exp(1j * phase_diff_nm)
    
    # Compute the length of the mean vector (mean resultant length)
    Rn_m = np.abs(np.sum(unit_vectors)) / len(unit_vectors)
    
    return Rn_m


def cross_frequency_rrci(
    signal1, signal2, sfreq, freq_peak1, freq_peak2, bandwidth=1, max_denom=50
):
    """
    Computes the Rhythmic Ratio Coupling Index (RRCI) between two signals for cross-frequency.

    The function first calculates the rhythmic ratio between two peak frequencies. It then filters
    the two input signals around a frequency band centered on their respective peak frequencies.
    Finally, it calculates the rhythmic ratio coupling index (RRCI) for these filtered signals.
    The RRCI is a measure of how much the rhythms of the two signals, in terms of their phase information,
    are coupled across different frequencies. In other words, it provides a measure of phase-to-phase
    coupling across these frequencies.

    Parameters
    ----------
    signal1 : ndarray
        First signal in time series.
    signal2 : ndarray
        Second signal in time series.
    sfreq : float
        Sampling frequency.
    freq_peak1 : float
        Peak frequency for the first signal.
    freq_peak2 : float
        Peak frequency for the second signal.
    bandwidth : float, default=1
        Frequency bandwidth for filtering the signals.
    max_denom : int, default=50
        The maximum denominator for the rhythmic ratio.

    Returns
    -------
    rrci : float
        Rhythmic Ratio Coupling Index between the two signals.

    Examples
    --------
    >>> signal1 = np.random.normal(0, 1, 5000)
    >>> signal2 = np.random.normal(0, 1, 5000)
    >>> sfreq = 100.0
    >>> rrci = cross_frequency_rrci(signal1, signal2, sfreq, 10, 20, 2, 4)
    """
    freq_band1 = (freq_peak1 - bandwidth / 2, freq_peak1 + bandwidth / 2)
    freq_band2 = (freq_peak2 - bandwidth / 2, freq_peak2 + bandwidth / 2)

    rhythmic_ratio = compute_rhythmic_ratio(freq_peak1, freq_peak2, max_denom)

    rrci = rhythmic_ratio_coupling_imaginary(
        signal1, signal2, rhythmic_ratio, sfreq, freq_band1, freq_band2
    )
    return rrci


def compute_rhythmic_ratio(freq1, freq2, max_denom):
    ratio = Fraction(freq1 / freq2).limit_denominator(max_denom)
    return ratio.numerator, ratio.denominator


def rhythmic_ratio_coupling_imaginary(
    signal1, signal2, rhythmic_ratio, sfreq, freq_band1, freq_band2
):
    """
    Computes the Imaginary part of Rhythmic Ratio Coupling between two signals.
    From the paper : "On cross-frequency phase-phase coupling between theta and gamma oscillations in the hippocampus"

    Parameters
    ----------
    signal1 : ndarray
        First signal in time series.
    signal2 : ndarray
        Second signal in time series.
    rhythmic_ratio : tuple
        Rhythmic ratio (numerator, denominator).
    sfreq : float
        Sampling frequency.
    freq_band1 : tuple
        Frequency band for the first signal (lowcut, highcut).
    freq_band2 : tuple
        Frequency band for the second signal (lowcut, highcut).

    Returns
    -------
    imaginary_part : float
        Imaginary part of the Rhythmic Ratio Coupling.
    """
    l_freq1, h_freq1 = freq_band1
    l_freq2, h_freq2 = freq_band2

    # Filter the signals
    filtered_signal1 = mne.filter.filter_data(
        signal1, sfreq, l_freq1, h_freq1, method="fir", verbose=False
    )
    filtered_signal2 = mne.filter.filter_data(
        signal2, sfreq, l_freq2, h_freq2, method="fir", verbose=False
    )

    # Compute the Hilbert transform
    analytic_signal1 = hilbert(filtered_signal1)
    analytic_signal2 = hilbert(filtered_signal2)

    # Extract instantaneous phases
    phase1 = np.angle(analytic_signal1)
    phase2 = np.angle(analytic_signal2)

    # Calculate the complex phase differences
    n, m = rhythmic_ratio
    phase_diff = n * phase1 - m * phase2

    # Compute the mean of the complex exponential of the phase differences
    mean_exp_phase_diff = np.mean(np.exp(1j * phase_diff))

    # Extract the imaginary part
    imaginary_part = np.imag(mean_exp_phase_diff)

    return np.abs(imaginary_part)


def HilbertHuang1D_nopeaks(
    data,
    sf,
    graph=False,
    nIMFs=5,
    min_freq=1,
    max_freq=45,
    precision=0.5,
    bin_spread="log",
    smooth_sigma=None,
):
    """
    The Hilbert-Huang transform provides a description of how the energy
    or power within a signal is distributed across frequency.
    The distributions are based on the instantaneous frequency and
    amplitude of a signal.

    Parameters
    ----------
    data : array (numDataPoints,)
        Single time series.
    sf : int
        Sampling frequency.
    graph : bool, default=False
        Defines if a graph is generated.
    nIMFs : int, default=5
        Number of intrinsic mode functions (IMFs) to keep when
        Empirical Mode Decomposition (EMD) is computed.
    min_freq : float, default=1
        Minimum frequency to consider.
    max_freq : float, default=80
        Maximum frequency to consider.
    precision : float, default=0.1
        Value in Hertz corresponding to the minimal step between two
        frequency bins.
    bin_spread : str, default='log'

        - 'linear'
        - 'log'
    smooth_sigma : float, default=None
        Sigma value for gaussian smoothing.
    Returns
    -------
    IF : array (numDataPoints,nIMFs)
        instantaneous frequencies associated with each IMF.
    IP : array (numDataPoints,nIMFs)
        instantaneous power associated with each IMF.
    IA : array (numDataPoints,nIMFs)
        instantaneous amplitude associated with each IMF.
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
        if smooth_sigma is not None:
            spec_ = gaussian_filter1d(spec_, smooth_sigma)
        freqs.append(freqs_)
        spec.append(spec_)
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

    return IF, IP, IA, np.array(spec), np.array(bins_)


def EMD_time_resolved_harmonicity(
    time_series1, time_series2, sf, nIMFs=5, method="harmsim"
):
    """
    Computes the harmonicity between the instantaneous frequencies (IF) for each
    point in time between all pairs of corresponding intrinsic mode functions (IMFs).

    Parameters
    ----------
    time_series1 : array (numDataPoints,)
        First time series.
    time_series2 : array (numDataPoints,)
        Second time series.
    sf : int
        Sampling frequency.
    nIMFs : int, default=5
        Number of intrinsic mode functions (IMFs) to consider.

    Returns
    -------
    harmonicity : array (numDataPoints, nIMFs)
        Harmonicity values for each pair of corresponding IMFs.
    """

    # Compute the Hilbert-Huang transform for each time series
    IF1, _, _, _, _ = HilbertHuang1D_nopeaks(time_series1, sf, nIMFs=nIMFs)
    IF2, _, _, _, _ = HilbertHuang1D_nopeaks(time_series2, sf, nIMFs=nIMFs)
    #print(IF1.shape)
    # Compute the harmonicity between the instantaneous frequencies of corresponding IMFs
    harmonicity = np.zeros((IF1.shape[0], nIMFs))

    for i in range(len(IF1)):
        for imf in range(nIMFs):
            if method == "harmsim":
                try:
                    if IF1[i, imf] > IF2[i, imf]:
                        harmonicity[i, imf] = dyad_similarity(IF1[i, imf] / IF2[i, imf])
                    if IF1[i, imf] < IF2[i, imf]:
                        harmonicity[i, imf] = dyad_similarity(IF2[i, imf] / IF1[i, imf])
                except ZeroDivisionError:
                    harmonicity[i, imf] = 0
            elif method == "subharm_tension":
                _, _, subharm_tension, _ = compute_subharmonic_tension(
                    [IF1[i, imf], IF2[i, imf]],
                    n_harmonics=10,
                    delta_lim=100,
                    min_notes=2,
                )
                #print(subharm_tension)
                if subharm_tenion != "NaN":
                    harmonicity[i, imf] = subharm_tension[0]
                else:
                    pass
    return harmonicity

def temporal_correlation_fdr(data):
    """
    Compute the temporal correlation for each pair of electrodes and output a connectivity matrix
    and a matrix of FDR-corrected p-values. This function is used in the computation of the
    transitional harmony connectivity.

    Parameters
    ----------
    data : array-like
        An array of shape (electrodes, samples) containing the electrode recordings.

    Returns
    -------
    connectivity_matrix : ndarray
        A connectivity matrix of shape (electrodes, electrodes) with the temporal correlation for each pair of electrodes.
    fdr_corrected_pvals : ndarray
        A matrix of FDR-corrected p-values of shape (electrodes, electrodes).

    Notes
    -----
    This function calculates the temporal correlation for each pair of electrodes, creating a connectivity matrix.
    Simultaneously, it calculates a matrix of p-values and corrects for multiple comparisons using the False Discovery Rate (FDR) method.
    """

    if not isinstance(data, np.ndarray):
        data = np.array(data)

    num_electrodes = data.shape[0]

    connectivity_matrix = np.zeros((num_electrodes, num_electrodes))
    pvals_matrix = np.zeros((num_electrodes, num_electrodes))

    for i in range(num_electrodes):
        for j in range(num_electrodes):
            data_i_masked = ma.masked_array(data[i], mask=np.isnan(data[i]))
            data_j_masked = ma.masked_array(data[j], mask=np.isnan(data[j]))

            corr = ma.corrcoef(data_i_masked, data_j_masked, allow_masked=True)[0, 1]
            n = np.sum(
                ~np.isnan(data[i]) & ~np.isnan(data[j])
            )  # Calculate number of non-NaN data points
            df = n - 2  # Degrees of freedom
            t_val = corr * np.sqrt(df / (1 - corr**2))  # Calculate t-value
            pval = 2 * (1 - t.cdf(abs(t_val), df))  # Calculate two-tailed p-value

            connectivity_matrix[i, j], pvals_matrix[i, j] = corr, pval

    # Lazy import — statsmodels is an optional dep used only here.
    from statsmodels.stats.multitest import multipletests
    pvals = pvals_matrix.flatten()
    fdr_corrected_pvals = multipletests(pvals, method="fdr_bh")[1]
    fdr_corrected_pvals = fdr_corrected_pvals.reshape((num_electrodes, num_electrodes))

    return connectivity_matrix, fdr_corrected_pvals


def _scalar_aggregate(spec: np.ndarray, result, factor: str, aggregate: str) -> float:
    """Reduce a per-frequency spectrum to a single scalar via the chosen
    aggregate. See :meth:`harmonic_connectivity.compute_cross_resonance_connectivity`
    for the documented options.

    Module-level so it can be reused by the surrogate-z helper and is unit-testable.
    """
    spec = np.asarray(spec, dtype=np.float64)
    if spec.size == 0:
        return 0.0
    if aggregate == "max":
        return float(np.max(spec))
    if aggregate == "mean":
        return float(np.mean(spec))
    if aggregate == "sum":
        return float(np.sum(spec))
    if aggregate == "peak":
        peak_freqs = result.peaks.get(factor, np.array([]))
        if len(peak_freqs) == 0:
            return float(np.max(spec))
        idx = int(np.argmin(np.abs(result.freqs - peak_freqs[0])))
        return float(spec[idx])
    if aggregate == "peak_to_median":
        # Log-scale peak-to-median ratio. Robust to broadband-noise channels.
        # Replaces the older 'peak_over_median' which saturated at 1.0 for any
        # spectrum with a sharp peak (the bounded (max-med)/(max+med) formula
        # hit its ceiling for essentially all real-world signals).
        mx = float(np.max(spec))
        med = float(max(np.median(spec), 1e-15))
        if mx < 1e-15:
            return 0.0
        return float(np.log10(mx / med))
    if aggregate == "peak_over_median":
        # Deprecated — kept for backward compat. Saturates at 1.0 for sharp
        # peaks; prefer 'peak_to_median' or 'peak_z'.
        mx = float(np.max(spec))
        med = float(np.median(spec))
        if mx + med < 1e-15:
            return 0.0
        return (mx - med) / (mx + med)
    if aggregate == "spectral_concentration":
        # Fraction of total energy in the top-3 bins; sharp peaks → high,
        # diffuse spectra → low.
        total = float(np.sum(spec))
        if total < 1e-15:
            return 0.0
        k = min(3, spec.size)
        top_k = np.sort(spec)[-k:]
        return float(np.sum(top_k) / total)
    if aggregate == "peak_z":
        # z-score of the dominant peak relative to the OFF-peak distribution.
        # Crucially, ALL prominence-detected peaks (and their immediate
        # neighbors) are masked from the off-peak distribution so that
        # multi-peak spectra (e.g. a 1:2 harmonic pair with peaks at both
        # 10 and 20 Hz) are not penalized — their secondary peak doesn't
        # inflate the off-peak std.
        peak_freqs = np.atleast_1d(result.peaks.get(factor, np.array([])))
        if peak_freqs.size == 0:
            peak_idx = int(np.argmax(spec))
            peak_idxs = [peak_idx]
        else:
            peak_idxs = [int(np.argmin(np.abs(result.freqs - pf))) for pf in peak_freqs]
            peak_idx = peak_idxs[0]
        mask = np.ones(spec.size, dtype=bool)
        for pidx in peak_idxs:
            lo = max(0, pidx - 1)
            hi = min(spec.size, pidx + 2)
            mask[lo:hi] = False
        off = spec[mask]
        if off.size < 2:
            return 0.0
        sd = float(np.std(off))
        if sd < 1e-15:
            return 0.0
        return (float(spec[peak_idx]) - float(np.mean(off))) / sd
    # Defensive — shouldn't reach (caller validates)
    raise ValueError(f"unknown aggregate {aggregate!r}")


# ===========================================================================
# Cross-channel resonance orchestrator (Layer B of the resonance refactor).
# Lives in harmonic_connectivity (not in biotuner.resonance) because it
# operates on TWO signals — connectivity territory by the architecture
# decision documented in CONNECTIVITY_PROPOSAL.md. Reuses primitives
# (kernels, coupling metrics, combine rules, complexity helper) from
# biotuner.resonance via the registry.
# ===========================================================================


@dataclass
class CrossResonanceResult:
    """Output of :func:`compute_cross_resonance`. Mirrors
    :class:`biotuner.resonance.ResonanceResult` but with three reducer flavors
    per factor (asymmetric 1→2, asymmetric 2→1, symmetrized 'all'), reflecting
    the directionality of cross-channel weighting:

      ``'1to2'`` — H[i] = (p1[i] * Σⱼ p2[j] * S[i,j]) / (2T)  (signal1 at i, signal2 at j)
      ``'2to1'`` — H[i] = (p2[i] * Σⱼ p1[j] * S[i,j]) / (2T)  (transposed)
      ``'all'``  — H[i] = (H_1to2[i] + H_2to1[i]) / 2          (symmetrized)

    where ``T = sum(psd1_clean) + sum(psd2_clean)`` (legacy normalization
    preserved for bit-exact reproduction of compute_cross_spectrum_harmonicity).
    """

    freqs: np.ndarray
    resonance_spectrum: Dict[str, np.ndarray] = field(default_factory=dict)  # {'1to2','2to1','all'}
    factors: Dict[str, Dict[str, np.ndarray]] = field(default_factory=dict)  # {'H': {...}, 'PC': {...}}
    summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)       # nested per spectrum
    peaks: Dict[str, np.ndarray] = field(default_factory=dict)               # {'H','PC','R'} → freqs of 'all' flavor
    config: Optional[ResonanceConfig] = None
    intermediates: Optional[Dict[str, Any]] = None

    def _get_matrix(self, key: str, label: str) -> np.ndarray:
        if self.intermediates is None or key not in self.intermediates:
            raise AttributeError(
                f"{label} matrix not available — re-run compute_cross_resonance with "
                f"ResonanceConfig(return_intermediates=True) to populate "
                f"result.intermediates['{key}']."
            )
        return self.intermediates[key]

    @property
    def harmonicity_matrix(self) -> np.ndarray:
        """The N×N cross-channel harmonic-similarity matrix ``S[i, j]``.

        Symmetric in (i, j). Requires ``ResonanceConfig(return_intermediates=True)``.
        """
        return self._get_matrix("harmonicity_matrix", "Harmonicity")

    @property
    def phase_coupling_matrix(self) -> np.ndarray:
        """The N×N cross-channel phase-coupling matrix ``Φ[i, j]``.

        Each cell is ``W[i, j] · metric(channel-1 at i, channel-2 at j, n, m)``.
        NOT symmetric — ``Φ[i, j]`` and ``Φ[j, i]`` correspond to different
        directional weightings. Requires
        ``ResonanceConfig(return_intermediates=True)``.
        """
        return self._get_matrix("phase_coupling_matrix", "Phase-coupling")


def _cross_reduce_3flavors(
    matrix: np.ndarray,
    psd1: np.ndarray,
    psd2: np.ndarray,
    *,
    normalize: str = "joint_2T",
):
    """Reduce an N×N cross-channel matrix into three per-bin spectra.

    Parameters
    ----------
    matrix : (N, N)
        Either a harmonic-similarity matrix S[i,j] (symmetric in i,j) or a
        cross-spectrum phase-coupling matrix Φ[i,j] (NOT symmetric — Φ[i,j]
        uses Zxx1[i] x conj(Zxx2[j])).
    psd1, psd2 : (N,)
        Per-channel min-max-rescaled PSDs.
    normalize : 'joint_2T' (legacy H normalization) | 'count' (legacy PC
        no-phase-mode) | 'joint_2T_count' (legacy PC weighted phase-mode).

    Returns
    -------
    v1, v2, v_all : (N,)
    """
    n = matrix.shape[0]
    mask = ~np.eye(n, dtype=bool)
    M_off = np.where(mask, matrix, 0.0)
    M_off_T = np.where(mask, matrix.T, 0.0)

    if normalize == "joint_2T":
        # H-style normalization: divide by (2 * (sum(p1) + sum(p2)))
        T = (np.sum(psd1) + np.sum(psd2)) * 2.0
        # v1[i] = (psd1[i] / T) * Σ_{j!=i} M[i,j] * psd2[j]
        v1 = (psd1 * (M_off @ psd2)) / T
        v2 = (psd2 * (M_off @ psd1)) / T
        # v_all = ((M[i,j] p1[i] p2[j]) + (M[j,i] p1[j] p2[i])) / 2 / T
        # M[j,i] @ row i requires M_off_T column
        v_all = (
            (psd1 * (M_off @ psd2)) + (psd2 * (M_off_T @ psd1))
        ) / (2.0 * T)
    elif normalize == "count":
        # PC-style unweighted: divide by count = n - 1 (off-diagonal entries)
        count = n - 1
        v1 = M_off.sum(axis=1) / count
        v2 = M_off_T.sum(axis=1) / count
        v_all = ((M_off + M_off_T) / 2.0).sum(axis=1) / count
    elif normalize == "joint_2T_count":
        # PC-style WEIGHTED: divide by (2 * T) like H, but with the cross
        # matrix instead of S. (phase_mode='weighted' branch in legacy)
        T = (np.sum(psd1) + np.sum(psd2)) * 2.0
        v1 = (psd1 * (M_off @ psd2)) / T
        v2 = (psd2 * (M_off @ psd1)) / T
        v_all = (
            (psd1 * (M_off @ psd2)) + (psd2 * (M_off_T @ psd1))
        ) / (2.0 * T)
    else:
        raise ValueError(f"unknown normalize={normalize!r}")
    return v1, v2, v_all


def compute_cross_resonance(
    signal1: np.ndarray,
    signal2: np.ndarray,
    sf: float,
    config: Optional[ResonanceConfig] = None,
) -> CrossResonanceResult:
    """Cross-channel resonance spectrum H × PC = R between two signals.

    The cross-channel analog of :func:`biotuner.resonance.compute_resonance`,
    producing three reducer flavors per factor (1→2 asymmetric, 2→1 asymmetric,
    symmetrized 'all') and a corresponding resonance spectrum for each. With
    the default ``ResonanceConfig`` this reproduces the legacy
    :func:`compute_cross_spectrum_harmonicity` numerics within ``atol=1e-5``
    on the snapshot regression set.

    Parameters
    ----------
    signal1, signal2 : 1-D arrays
        Time-domain signals to compare.
    sf : float
        Sampling frequency (Hz).
    config : :class:`ResonanceConfig`, optional
        If None, the legacy-default config is used (matches the historical
        ``compute_cross_spectrum_harmonicity`` behavior).

    Returns
    -------
    :class:`CrossResonanceResult`
    """
    if config is None:
        # Sensible recommended-defaults config:
        # - cross_pc_reducer='joint'      (frequency-localized PC; recommended)
        # - cross_use_ratio_kernel=True   (true n:m phase coupling via binary_nm)
        # - ratio_kernel='binary'
        # These come from ResonanceConfig field defaults — see orchestrator.py
        # for the rationale. To exactly reproduce legacy
        # compute_cross_spectrum_harmonicity numerics, pass a config with
        # cross_pc_reducer='count' and cross_use_ratio_kernel=False (this is
        # what the shim compute_cross_spectrum_harmonicity does internally).
        config = ResonanceConfig(
            harmonic_kernel="harmsim",
            harmonic_kernel_params={"n_harms": 10, "delta_lim": 0.1, "min_notes": 2},
            phase_estimator="stft",
            coupling_metric="nm_wpli_complex",   # complex-coefficient wPLI on STFT
            gaussian_smooth_sigma=1.0,
            combine="product",
            precision_hz=1.0,
            fmin=1.0, fmax=30.0, noverlap=1,
            smoothness=1.0,
            n_peaks=5,
            remove_aperiodic=False,              # legacy cross-spectrum default
            # cross_pc_reducer and cross_use_ratio_kernel inherit the new
            # joint+n:m defaults from ResonanceConfig field defaults.
            ratio_kernel="binary",
            ratio_kernel_params={"max_nm": 3, "tolerance": 0.05, "fallback_to_1_1": True},
        )

    nperseg = int(sf / config.precision_hz)

    # PSD per channel (same as single-channel pipeline)
    freqs, psd1 = compute_frequency_and_psd(
        signal1, config.precision_hz, smoothness=config.smoothness,
        fs=sf, noverlap=config.noverlap, fmin=config.fmin, fmax=config.fmax,
    )
    _, psd2 = compute_frequency_and_psd(
        signal2, config.precision_hz, smoothness=config.smoothness,
        fs=sf, noverlap=config.noverlap, fmin=config.fmin, fmax=config.fmax,
    )
    psd1_clean = apply_power_law_remove(freqs, psd1, config.remove_aperiodic)
    psd2_clean = apply_power_law_remove(freqs, psd2, config.remove_aperiodic)

    # Min-max rescale each PSD to [0, 1] (legacy cross-spectrum behavior;
    # the equivalent of psd_normalization='minmax_only' — no division by sum)
    p1_min, p1_max = np.min(psd1_clean), np.max(psd1_clean)
    psd1_clean = (psd1_clean - p1_min) / (p1_max - p1_min)
    p2_min, p2_max = np.min(psd2_clean), np.max(psd2_clean)
    psd2_clean = (psd2_clean - p2_min) / (p2_max - p2_min)

    # STFT per channel — gives complex coefficients used by nm_wpli_complex
    _, _, Zxx1 = stft(signal1, sf, nperseg=int(nperseg / config.smoothness), noverlap=config.noverlap)
    _, _, Zxx2 = stft(signal2, sf, nperseg=int(nperseg / config.smoothness), noverlap=config.noverlap)

    n_freqs = len(freqs)

    # Harmonic similarity matrix S[i,j]
    kernel_fn = HARMONIC_KERNELS[config.harmonic_kernel]
    S = kernel_fn(freqs, freqs, **config.harmonic_kernel_params)
    # Legacy applies S only when freqs[j] != 0; the kernel_harmsim_legacy
    # already does this. Match the legacy "skip f=0" by zeroing those entries.
    for j, f in enumerate(freqs):
        if f == 0:
            S[:, j] = 0.0

    # Phase coupling matrix Φ[i,j] via the chosen pairwise metric.
    # Default: nm_wpli_complex with (n=1, m=1) — matches legacy cross-spectrum wPLI.
    # Refinement B: if cross_use_ratio_kernel, dispatch through the ratio kernel
    # (binary_nm or others) to determine (n, m) for each freq pair and compute
    # true n:m phase coupling instead of always 1:1.
    #
    # Metric-input dispatch (bug fix): registry tags each metric with 'phase'
    # (real phase angles in [-π, π]) or 'analytic' (complex Zxx). Phase-input
    # metrics applied to raw Zxx blow up numerically — they compute
    # ``n·x - m·y`` which is bounded for real phases but unbounded for
    # complex Zxx, especially with large (n, m) from the fraction kernel.
    # Convert to phase angles when the metric expects them.
    metric_fn = PAIRWISE_COUPLING_METRICS[config.coupling_metric]
    input_type = COUPLING_INPUT_TYPE.get(config.coupling_metric, "phase")
    if input_type == "phase":
        arg1_full = np.angle(Zxx1)
        arg2_full = np.angle(Zxx2)
    else:
        arg1_full = Zxx1
        arg2_full = Zxx2
    Phi = np.zeros((n_freqs, n_freqs), dtype=np.float64)
    if config.cross_use_ratio_kernel:
        ratio_fn = RATIO_KERNELS[config.ratio_kernel]
        W, N_mat, M_mat = ratio_fn(freqs, freqs, **config.ratio_kernel_params)
        for i in range(n_freqs):
            for j in range(n_freqs):
                if freqs[j] != 0 and W[i, j] > 0:
                    n, m = int(N_mat[i, j]), int(M_mat[i, j])
                    Phi[i, j] = float(W[i, j]) * metric_fn(arg1_full[i], arg2_full[j], n, m)
    else:
        for i in range(n_freqs):
            for j in range(n_freqs):
                if freqs[j] != 0:
                    Phi[i, j] = metric_fn(arg1_full[i], arg2_full[j], 1, 1)

    # Reduce to 3-flavor H and PC. H always uses joint-probability weighting
    # (legacy behavior). PC reducer is configurable via config.cross_pc_reducer:
    #   'count' (legacy default)  — uniform average over freq pairs
    #   'joint' (Refinement A)     — joint p1[i]*p2[j] weighting (matches H)
    #   'joint_2T_count' (legacy phase_mode='weighted')
    H1, H2, H_all = _cross_reduce_3flavors(S, psd1_clean, psd2_clean, normalize="joint_2T")
    pc_reducer = config.cross_pc_reducer
    if config.phase_estimator_params.get("phase_mode") == "weighted":
        pc_reducer = "joint_2T_count"
    if pc_reducer == "joint":
        # Match H's reducer for frequency-localized PC
        PC1, PC2, PC_all = _cross_reduce_3flavors(Phi, psd1_clean, psd2_clean, normalize="joint_2T")
    else:
        PC1, PC2, PC_all = _cross_reduce_3flavors(Phi, psd1_clean, psd2_clean, normalize=pc_reducer)

    # Gaussian smoothing on each spectrum
    sig = config.gaussian_smooth_sigma
    if sig > 0:
        H1 = _gaussian_filter(H1, sigma=sig)
        H2 = _gaussian_filter(H2, sigma=sig)
        H_all = _gaussian_filter(H_all, sigma=sig)
        PC1 = _gaussian_filter(PC1, sigma=sig)
        PC2 = _gaussian_filter(PC2, sigma=sig)
        PC_all = _gaussian_filter(PC_all, sigma=sig)

    # Combine to R per flavor
    combine_fn = COMBINE_RULES[config.combine]
    R1 = combine_fn([H1, PC1])
    R2 = combine_fn([H2, PC2])
    R_all = combine_fn([H_all, PC_all])

    # Peaks on the 'all' flavor (matches legacy DataFrame columns)
    H_peaks, _ = find_spectral_peaks(H_all, freqs, config.n_peaks, prominence_threshold=0.1)
    PC_peaks, _ = find_spectral_peaks(PC_all, freqs, config.n_peaks, prominence_threshold=0.0001)
    R_peaks, _ = find_spectral_peaks(R_all, freqs, config.n_peaks, prominence_threshold=0.01)

    # Complexity per spectrum (uses 'all' flavor)
    summaries = {
        "H": spectrum_complexity(H_all, freqs, n_peaks=config.n_peaks, prominence_threshold=0.1),
        "PC": spectrum_complexity(PC_all, freqs, n_peaks=config.n_peaks, prominence_threshold=0.0001),
        "R": spectrum_complexity(R_all, freqs, n_peaks=config.n_peaks, prominence_threshold=0.01),
    }

    result = CrossResonanceResult(
        freqs=freqs,
        resonance_spectrum={"1to2": R1, "2to1": R2, "all": R_all},
        factors={
            "H": {"1to2": H1, "2to1": H2, "all": H_all},
            "PC": {"1to2": PC1, "2to1": PC2, "all": PC_all},
        },
        summaries=summaries,
        peaks={"H": H_peaks, "PC": PC_peaks, "R": R_peaks},
        config=config,
    )
    if config.return_intermediates:
        result.intermediates = {
            "psd1_clean": psd1_clean, "psd2_clean": psd2_clean,
            "harmonicity_matrix": S, "phase_coupling_matrix": Phi,
            "Zxx1": Zxx1, "Zxx2": Zxx2,
        }
    return result


def compute_cross_spectrum_harmonicity(
    signal1,
    signal2,
    precision_hz,
    fmin=None,
    fmax=None,
    noverlap=1,
    fs=44100,
    power_law_remove=False,
    n_peaks=5,
    metric="harmsim",
    n_harms=10,
    delta_lim=0.1,
    min_notes=2,
    plot=False,
    smoothness=1,
    smoothness_harm=1,
    phase_mode=None,
    save_fig=False,
    save_name="harmonic_spectrum_connectivity.png",
):
    
    """
    Compute the cross-spectrum harmonicity between two signals. This function is useful for analyzing
    the interaction between frequency components of two different signals.

    Parameters
    ----------
    signal1, signal2 : array_like
        Input signals.
    precision_hz : float
        The precision in Hz for the short time Fourier transform.
    fmin, fmax : float or None
        Minimum and maximum frequency for computing the power spectral density.
    noverlap : int
        Number of points to overlap between segments for computing the power spectral density.
    fs : int
        The sampling frequency of the signals.
    power_law_remove : bool
        If True, power-law noise is removed from the power spectral densities of the signals.
    n_peaks : int
        The number of peaks to identify in the spectra.
    metric : {"harmsim", "subharm_tension"}
        The metric to use for computing the harmonicity between the signals.
    n_harms : int
        Number of harmonics to consider for computing subharmonic tension.
    delta_lim : float
        Delta limit for computing subharmonic tension.
    min_notes : int
        Minimum number of notes for computing subharmonic tension.
    plot : bool
        If True, the function will plot the cross harmonic and phase coupling spectrum.
    smoothness : int
        Smoothness of the Fourier transform. Higher values result in a smoother transform.
    smoothness_harm : int
        Smoothness of the harmonic spectrum. Higher values result in a smoother spectrum.
    phase_mode : {None, "weighted"}
        If "weighted", the phase coupling is computed as a weighted sum. If None, it is computed as a simple average.
    save_fig : bool
        If True, the plot is saved as a png file.
    save_name : str
        Name of the png file if save_fig is True.

    Returns
    -------
    DataFrame
        A pandas DataFrame with columns representing various computed metrics of the signals, such as spectral flatness, spectral entropy, etc.
        The DataFrame also contains columns for peak frequencies in the spectra and their harmonic similarities.
    """
    
    # Delegates to compute_cross_resonance (the new strategy-registry-based
    # cross-channel orchestrator) and repackages the output as the historical
    # DataFrame format. Bit-exact reproduction is asserted by
    # tests/resonance/test_cross_snapshot_regression.py.
    #
    # IMPORTANT: this shim explicitly OPTS OUT of the new
    # cross_pc_reducer='joint' and cross_use_ratio_kernel=True defaults
    # (set in ResonanceConfig) because the legacy compute_cross_spectrum_harmonicity
    # used 'count' reducer + 1:1 PC. The legacy formula is preserved here for
    # paper reproducibility; new analyses should call compute_cross_resonance
    # directly and accept the (better) registry defaults.
    cfg = ResonanceConfig(
        precision_hz=precision_hz,
        fmin=fmin if fmin is not None else 1.0,
        fmax=fmax if fmax is not None else 30.0,
        noverlap=noverlap,
        smoothness=smoothness,
        n_peaks=n_peaks,
        remove_aperiodic=power_law_remove,
        harmonic_kernel=metric,
        harmonic_kernel_params={"n_harms": n_harms, "delta_lim": delta_lim, "min_notes": min_notes},
        phase_estimator="stft",
        # phase_mode='weighted' switches the PC reducer to the joint_2T_count
        # normalization (used by compute_cross_resonance via this kwarg)
        phase_estimator_params={"phase_mode": phase_mode} if phase_mode else {},
        coupling_metric="nm_wpli_complex",
        gaussian_smooth_sigma=smoothness_harm,
        combine="product",
        # LEGACY OVERRIDES — keep snapshot regression bit-exact.
        cross_pc_reducer="count",
        cross_use_ratio_kernel=False,
    )
    result = compute_cross_resonance(signal1, signal2, sf=fs, config=cfg)

    freqs = result.freqs
    # Recompute PSDs once for plotting (compute_cross_resonance doesn't expose
    # the raw PSDs in its public result; only the cleaned/rescaled versions
    # via intermediates).
    _, psd1 = compute_frequency_and_psd(
        signal1, precision_hz, smoothness, fs, noverlap, fmin=fmin, fmax=fmax,
    )
    _, psd2 = compute_frequency_and_psd(
        signal2, precision_hz, smoothness, fs, noverlap, fmin=fmin, fmax=fmax,
    )

    harmonicity_values1 = result.factors["H"]["1to2"]
    harmonicity_values2 = result.factors["H"]["2to1"]
    harmonicity_values_all = result.factors["H"]["all"]
    phase_coupling_values1 = result.factors["PC"]["1to2"]
    phase_coupling_values2 = result.factors["PC"]["2to1"]
    phase_coupling_values_all = result.factors["PC"]["all"]
    normalized_combined_metric = result.resonance_spectrum["all"]

    # Find peaks in the spectra (legacy thresholds — same as compute_cross_resonance)
    harmonicity_peak_frequencies, harm_peak_idx = find_spectral_peaks(
        harmonicity_values_all, freqs, n_peaks, prominence_threshold=0.1
    )
    phase_peak_frequencies, phase_peak_idx = find_spectral_peaks(
        phase_coupling_values_all, freqs, n_peaks, prominence_threshold=0.0001
    )
    resonance_peak_frequencies, res_peak_idx = find_spectral_peaks(
        normalized_combined_metric, freqs, n_peaks, prominence_threshold=0.01
    )

    # Compute spectral flatness and entropy values (legacy DataFrame format)
    harmonic_complexity = harmonic_entropy(
        freqs,
        harmonicity_values_all,
        phase_coupling_values_all,
        normalized_combined_metric,
    )

    # create dataframe with relevant values
    df = pd.DataFrame(
        {
            "harmonicity": [harmonicity_values_all],
            "phase_coupling": [phase_coupling_values_all],
            "resonance": [normalized_combined_metric],
            "harm_spectral_flatness": [
                harmonic_complexity["Spectral Flatness"]["Harmonicity"]
            ],
            "harm_spectral_entropy": [
                harmonic_complexity["Spectral Entropy"]["Harmonicity"]
            ],
            "harm_higuchi": [
                harmonic_complexity["Higuchi Fractal Dimension"]["Harmonicity"]
            ],
            "harm_spectral_spread": [
                harmonic_complexity["Spectral Spread"]["Harmonicity"]
            ],
            "phase_spectral_flatness": [
                harmonic_complexity["Spectral Flatness"]["Phase Coupling"]
            ],
            "phase_spectral_entropy": [
                harmonic_complexity["Spectral Entropy"]["Phase Coupling"]
            ],
            "phase_higuchi": [
                harmonic_complexity["Higuchi Fractal Dimension"]["Phase Coupling"]
            ],
            "phase_spectral_spread": [
                harmonic_complexity["Spectral Spread"]["Phase Coupling"]
            ],
            "res_spectral_flatness": [
                harmonic_complexity["Spectral Flatness"]["Resonance"]
            ],
            "res_spectral_entropy": [
                harmonic_complexity["Spectral Entropy"]["Resonance"]
            ],
            "res_higuchi": [
                harmonic_complexity["Higuchi Fractal Dimension"]["Resonance"]
            ],
            "res_spectral_spread": [
                harmonic_complexity["Spectral Spread"]["Resonance"]
            ],
            "harmonicity_peak_frequencies": [harmonicity_peak_frequencies],
            "phase_peak_frequencies": [phase_peak_frequencies],
            "resonance_peak_frequencies": [resonance_peak_frequencies],
        }
    )

    df["harmonicity_avg"] = df["harmonicity"].apply(np.mean)
    df["phase_coupling_avg"] = df["phase_coupling"].apply(np.mean)
    df["resonance_avg"] = df["resonance"].apply(np.mean)

    df["harmonicity_peaks_avg"] = df["harmonicity_peak_frequencies"].apply(safe_mean)
    df["phase_peaks_avg"] = df["phase_peak_frequencies"].apply(safe_mean)
    df["res_peaks_avg"] = df["resonance_peak_frequencies"].apply(safe_mean)

    df["resonance_max"] = df["resonance"].apply(np.max)

    # save df
    df["precision"] = precision_hz
    df["fmin"] = fmin
    df["fmax"] = fmax
    df["phase_weighting"] = phase_mode
    df["smooth_fft"] = smoothness
    df["smooth_harm"] = smoothness_harm
    df["fs"] = fs

    # calculate harmonic similarity between peaks
    df["phase_harmsim"] = df["phase_peak_frequencies"].apply(peaks_to_harmsim)
    df["harm_harmsim"] = df["harmonicity_peak_frequencies"].apply(peaks_to_harmsim)
    df["res_harmsim"] = df["resonance_peak_frequencies"].apply(peaks_to_harmsim)

    df["harm_harmsim_avg"] = df["harm_harmsim"].apply(safe_mean)
    df["phase_harmsim_avg"] = df["phase_harmsim"].apply(safe_mean)
    df["res_harmsim_avg"] = df["res_harmsim"].apply(safe_mean)

    df["harm_harmsim_max"] = df["harm_harmsim"].apply(safe_max)
    df["phase_harmsim_max"] = df["phase_harmsim"].apply(safe_max)
    df["res_harmsim_max"] = df["res_harmsim"].apply(safe_max)

    if plot is True:
        fig, (ax1, ax2, ax4) = plt.subplots(nrows=3, figsize=(14, 10))

        ax1.plot(freqs, 10 * np.log10(psd1), color="darkred", label="Spectrum 1")
        # ax1.scatter(freqs[peaks_psd1], 10 * np.log10(psd1[peaks_psd1]), color='red', marker='o', s=50)  # Add red dots on detected peaks
        ax1.plot(freqs, 10 * np.log10(psd2), color="darkgoldenrod", label="Spectrum 2")
        # ax1.scatter(freqs[peaks_psd2], 10 * np.log10(psd2[peaks_psd2]), color='red', marker='o', s=50)  # Add red dots on detected peaks
        ax1.set_title("Spectra 1 and 2")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Power (dB)")
        ax1.legend()
        ax1.grid()

        ax2.plot(
            freqs,
            harmonicity_values1,
            color="mediumaquamarine",
            alpha=1,
            linestyle="dashed",
        )
        ax2.plot(
            freqs, harmonicity_values2, color="turquoise", alpha=1, linestyle="dashed"
        )
        ax2.plot(
            freqs,
            harmonicity_values_all,
            color="darkblue",
            label="Cross Harmonic Spectrum",
            linestyle="solid",
        )
        ax2.plot(
            freqs[harm_peak_idx],
            harmonicity_values_all[harm_peak_idx],
            "ro",
            color="darkblue",
        )
        ax2.set_title("Cross Harmonic and Phase Coupling Spectrum")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Harmonicity")
        ax2.legend()
        ax2.grid()
        for peak in harmonicity_peak_frequencies:
            ax2.axvline(peak, color="darkblue", linestyle="--")

        ax3 = ax2.twinx()
        ax3.plot(
            freqs, phase_coupling_values1, color="violet", alpha=1, linestyle="dashed"
        )
        ax3.plot(
            freqs,
            phase_coupling_values2,
            color="mediumorchid",
            alpha=1,
            linestyle="dashed",
        )
        ax3.plot(
            freqs,
            phase_coupling_values_all,
            color="indigo",
            label="Cross Phase Coupling Spectrum",
            linestyle="solid",
        )
        ax3.plot(
            freqs[phase_peak_idx],
            phase_coupling_values_all[phase_peak_idx],
            "ro",
            color="indigo",
        )
        ax3.set_ylabel("Phase Coupling")
        for peak in phase_peak_frequencies:
            ax3.axvline(peak, color="darkviolet", linestyle="--")

        ax4.plot(freqs, normalized_combined_metric, color="deeppink", label="Resonance")
        ax4.set_title("Resonance")
        ax4.set_xlabel("Frequency (Hz)")
        ax4.set_ylabel("Combined Metric")
        ax4.plot(
            freqs[res_peak_idx],
            normalized_combined_metric[res_peak_idx],
            "ro",
            color="deeppink",
        )
        # ax4.legend()
        ax4.grid()
        for peak in resonance_peak_frequencies:
            ax4.axvline(peak, color="black", linestyle="--")

        # Add the legend for the second y-axis (phase_coupling) plot
        lines, labels = ax2.get_legend_handles_labels()
        lines2, labels2 = ax3.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="upper right")

        plt.tight_layout()
        if save_fig is True:
            plt.savefig(f"{savename}.png", dpi=300)
        plt.show()

    return df

"""
def compute_mutual_information(phase1, phase2, num_bins=10):
    # Discretize phase values into bins
    discretized_phase1 = np.digitize(
        phase1, bins=np.linspace(-np.pi, np.pi, num_bins + 1)
    )
    discretized_phase2 = np.digitize(
        phase2, bins=np.linspace(-np.pi, np.pi, num_bins + 1)
    )

    # Compute the joint probability distribution of the discretized phase values
    joint_prob, _, _ = np.histogram2d(
        discretized_phase1, discretized_phase2, bins=num_bins
    )

    # Normalize the joint probability distribution
    joint_prob = joint_prob / np.sum(joint_prob)

    # Compute Mutual Information
    MI = mutual_info_score(None, None, contingency=joint_prob)

    return MI


def MI_spectral(
    signal1, signal2, sf, min_freq, max_freq, precision, peak_pairs, wavelet="cmor"
):
    # Define the scales for the CWT based on the desired frequency precision
    scales = np.arange(min_freq, max_freq + precision, precision)
    scales = (sf / (2 * np.pi * precision)) * np.divide(1, scales)

    # Compute the Continuous Wavelet Transform of the signals
    cwt_signal1 = pywt.cwt(signal1, scales, wavelet)[0]
    cwt_signal2 = pywt.cwt(signal2, scales, wavelet)[0]

    # Extract the phase values from the CWT coefficients
    phase_signal1 = np.angle(cwt_signal1)
    phase_signal2 = np.angle(cwt_signal2)

    # Compute the Mutual Information between the phase values in the time-frequency domain
    mi_matrix = np.zeros((len(scales), len(scales)))
    for i in range(len(scales)):
        for j in range(len(scales)):
            mi_matrix[i, j] = mutual_info_score(
                phase_signal1[i, :], phase_signal2[j, :]
            )

    # Extract the MI values corresponding to the pairs of peaks
    mi_values = []
    for pair in peak_pairs:
        scale1 = int((sf / (2 * np.pi * precision)) * np.divide(1, pair[0]))
        scale2 = int((sf / (2 * np.pi * precision)) * np.divide(1, pair[1]))
        mi_values.append(mi_matrix[scale1, scale2])

    # Calculate the average MI value for the pairs of peaks
    avg_mi = np.mean(mi_values)

    return avg_mi"""

"""    def compute_harmonicity_metric_for_IMFs(data, sf, metric='harmsim', delta_lim=20, nIMFs=5, FREQ_BANDS=None):
        # Apply EMD to each channel
        emd_processor = EMD()
        IMFs = [emd_processor(channel)[:nIMFs] for channel in data]

        # Initialize variables
        list_idx = list(range(len(IMFs)))
        pairs = list(itertools.product(list_idx, list_idx))
        harm_conn_matrix = []

        if FREQ_BANDS is None:
            FREQ_BANDS = [
                [2, 3.55],
                [3.55, 7.15],
                [7.15, 14.3],
                [14.3, 28.55],
                [28.55, 49.4],
            ]

        if metric == 'wPLI_multiband':
            harm_conn_matrix = np.zeros((len(FREQ_BANDS), len(IMFs), len(IMFs)))

        # Compute the desired harmonicity metric for each corresponding IMF for each pair of channels
        for pair in pairs:
            imfs1 = IMFs[pair[0]]
            imfs2 = IMFs[pair[1]]

            for i in range(nIMFs):
                for j in range(nIMFs):
                    # Compute the desired harmonicity metric for the pair of IMFs
                    # [Replace this line with the appropriate code to compute the desired harmonicity metric for the pair of IMFs]

        # [Return the appropriate result]"""
