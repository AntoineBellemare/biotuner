Resonance
=========

Modular, strategy-registry-based resonance framework. Builds the per-frequency
**H × PC = R** spectrum from a single signal via swappable harmonic kernels,
ratio gates, phase estimators, pairwise coupling metrics, and combine rules —
with surrogate normalization and a separate path for higher-order coupling
metrics (Phase 3).

Quick start
-----------

::

   from biotuner.resonance import compute_resonance, ResonanceConfig

   # Recommended defaults
   result = compute_resonance(signal, sf=1000)
   # result.factors["H"], result.factors["PC"]
   # result.resonance_spectrum     — H · PC
   # result.summaries["H"/"PC"/"R"] — complexity dict per spectrum
   # result.peaks                   — prominence-detected peak freqs

   # To reproduce legacy compute_global_harmonicity bit-exactly:
   cfg = ResonanceConfig(psd_normalization="minmax_prob")
   result = compute_resonance(signal, sf=1000, config=cfg)

**Discovery:** see what kernels / coupling metrics / combine rules are
available by calling :func:`list_strategies`. All names returned are valid
for the corresponding :class:`ResonanceConfig` field.

Sister modules
--------------

* :mod:`biotuner.harmonic_spectrum` — narrow H-only entry point
  (:func:`biotuner.harmonic_spectrum.compute_harmonic_spectrum`).
* :mod:`biotuner.harmonic_connectivity` — cross-channel APIs:
  ``compute_cross_resonance`` for two signals,
  ``harmonic_connectivity(...).compute_cross_resonance_connectivity()`` for
  N-channel matrices, and ``...connectivity_zscore()`` for surrogate-normalized
  statistical inference.

Public API
----------

Main entry points:

* :func:`compute_resonance` — main entry point
* :class:`ResonanceConfig` — all swappable knobs (see ``return_intermediates``
  to keep the full N×N matrices on the result)
* :class:`ResonanceResult` — output dataclass; see "Result views" below
* :class:`HigherOrderResult` — for Phase 3 higher-order metrics
* :func:`with_surrogate_null` — surrogate-z-scored variant of ``compute_resonance``.
  Default ``surr_type='IAAFT'`` (iterated AAFT: preserves PSD + amplitude
  distribution); also accepts ``'phase_randomize'``, ``'time_shuffle'``, and the
  ``generate_surrogate`` types (``'AAFT'``, ``'phase'``, ``'shuffle'``, colored).
  Populates **per-factor** z-scores — see ``factor_z`` below.

Result views (on :class:`ResonanceResult`):

* ``result.factors["H" | "PC"]`` and ``result.resonance_spectrum`` —
  reduced 1-D spectra, length ``n_freqs``
* ``result.factor_z["H" | "PC" | "R"]`` — per-frequency surrogate z-scores
  (populated by :func:`with_surrogate_null`), with matching
  ``result.factor_surrogate_mean`` / ``result.factor_surrogate_std``. Use
  ``factor_z["PC"]`` for phase-coupling inference: R is harmonicity-dominated
  (PSD-driven), so ``factor_z["R"]`` is largely blind to phase coupling under a
  PSD-preserving null. ``result.resonance_spectrum_z`` mirrors ``factor_z["R"]``.
* ``result.harmonicity_matrix`` and ``result.phase_coupling_matrix`` —
  full N×N matrices ``S[i, j]`` and ``Φ[i, j]`` (need
  ``ResonanceConfig(return_intermediates=True)``)
* ``result.summaries["H" | "PC" | "R"]`` — scalar metrics per spectrum:
  ``avg``, ``max``, ``flatness``, ``entropy``, ``spread``, ``higuchi``,
  ``peaks``, ``peaks_avg``, ``peak_harmsim``, ``peak_harmsim_avg``,
  ``peak_harmsim_max`` (one-to-one with legacy
  ``compute_global_harmonicity`` columns)
* ``result.peaks["H" | "PC" | "R"]`` — peak frequencies per spectrum

Discovery + plotting helpers:

* :func:`list_strategies` — print/return the registered strategies across all 8 registries
* :func:`results_to_dataframe` — pack multiple results into a wide
  pandas DataFrame; flattens ``summaries`` into legacy-named columns
  (``harm_flatness``, ``phase_peaks_avg``, ``res_peak_harmsim_max``, …)
  by default; pass ``flatten_summaries=False`` for the narrow 4-column form
* ``harmonic_spectrum_plot_avg_corr``, ``harmonic_spectrum_plot_trial_corr``,
  ``harmonic_spectrum_plot_freq_corr`` — comparison plots over collections of results

Internals
---------

Strategy catalogs live in ``biotuner.resonance.registry``. To add a new kernel,
metric, or combine rule, call the appropriate ``register_*`` function from
your own module; it'll be discoverable by name via :class:`ResonanceConfig`
and listed by :func:`list_strategies`.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   biotuner.resonance

.. automodule:: biotuner.resonance
   :members:

.. automodule:: biotuner.resonance.orchestrator
   :members:

.. automodule:: biotuner.resonance.kernels_harmonic
   :members:

.. automodule:: biotuner.resonance.kernels_ratio
   :members:

.. automodule:: biotuner.resonance.phase_estimators
   :members:

.. automodule:: biotuner.resonance.coupling
   :members:

.. automodule:: biotuner.resonance.combine
   :members:

.. automodule:: biotuner.resonance.nulls
   :members:

.. automodule:: biotuner.resonance.registry
   :members:

.. automodule:: biotuner.resonance.plots
   :members:
