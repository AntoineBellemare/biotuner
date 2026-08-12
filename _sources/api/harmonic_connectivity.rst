Harmonic connectivity
=====================

Cross-channel coupling, cross-frequency, and connectivity matrices. Everything
that compares two-or-more signals lives here.

Quick start
-----------

**Peak-based** connectivity (between extracted peak lists, one scalar per
electrode pair) — uses the legacy peak-extraction pipeline:

::

   from biotuner.harmonic_connectivity import harmonic_connectivity

   hc = harmonic_connectivity(sf=1000, data=data_array, peaks_function="EMD",
                              precision=0.5, min_freq=2, max_freq=30, n_peaks=5)
   H_mat  = hc.compute_harm_connectivity(metric="harmsim")           # legacy H
   PC_mat = hc.compute_peak_phase_coupling_connectivity(coupling_metric="nm_plv")
   R_mat  = hc.compute_peak_resonance_connectivity(combine="product")

**Spectrum-based** cross-channel resonance (per-frequency H/PC/R between two
signals, with the same swappable kernels/metrics/combine rules as
:mod:`biotuner.resonance`):

::

   from biotuner.harmonic_connectivity import compute_cross_resonance

   cross = compute_cross_resonance(sig1, sig2, sf=1000)
   # cross.resonance_spectrum["1to2"]   — asymmetric (sig1 at i, sig2 at j)
   # cross.resonance_spectrum["2to1"]   — transposed
   # cross.resonance_spectrum["all"]    — symmetrized average
   # cross.factors["H"][...], cross.factors["PC"][...]
   # cross.summaries["H"/"PC"/"R"]      — complexity dict per spectrum

**Connectivity matrices** (loop ``compute_cross_resonance`` over all
electrode pairs):

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

* :mod:`biotuner.harmonic_spectrum` — single-signal H spectrum.
* :mod:`biotuner.resonance` — single-signal H × PC = R pipeline, including the
  registries this module dispatches against (``PAIRWISE_COUPLING_METRICS``,
  ``HARMONIC_KERNELS``, ``RATIO_KERNELS``, ``COMBINE_RULES``).

Public API
----------

The class:

* :class:`harmonic_connectivity` — peak-based + new spectrum-based methods:

  * ``.compute_harm_connectivity(metric=...)`` — legacy peak-based H
  * ``.compute_peak_phase_coupling_connectivity(coupling_metric=...)`` — registry-based PC
  * ``.compute_peak_resonance_connectivity(combine=...)`` — H × PC peak scalar
  * ``.compute_cross_resonance_connectivity(factor=, flavor=, aggregate=)`` — spectrum-based matrix
  * ``.compute_cross_resonance_connectivity_zscore(surrogate_kind=, n_surrogates=)`` — surrogate-normalized inference
  * ``.compute_harmonic_spectrum_connectivity(...)`` — legacy cross-spectrum DataFrame

Spectrum-based cross-channel orchestrator:

* :func:`compute_cross_resonance` — single-call cross-channel R(f) on a signal pair
* :class:`CrossResonanceResult` — output dataclass with 3 reducer flavors
  (``'1to2'`` / ``'2to1'`` / ``'all'``) per factor. The full N×N matrices
  are accessible as ``result.harmonicity_matrix`` and
  ``result.phase_coupling_matrix`` when the config has
  ``return_intermediates=True``.

Legacy shim:

* :func:`compute_cross_spectrum_harmonicity` — delegates to ``compute_cross_resonance``
  with explicit legacy preset; bit-exact reproduction guaranteed for paper workflows.

Standalone coupling utilities (kept for backward compat):

* :func:`wPLI_crossfreq`, :func:`wPLI_multiband`
* :func:`cross_frequency_rrci`, :func:`rhythmic_ratio_coupling_imaginary`, :func:`compute_rhythmic_ratio`
* :func:`n_m_phase_locking`
* :func:`compute_mutual_information`, :func:`MI_spectral`

IMF-based utilities (different abstraction layer):

* :func:`HilbertHuang1D_nopeaks`, :func:`EMD_time_resolved_harmonicity`
* :func:`temporal_correlation_fdr`

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   biotuner.harmonic_connectivity

.. automodule:: biotuner.harmonic_connectivity
   :members:
