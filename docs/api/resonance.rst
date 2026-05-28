Resonance
=========

Modular, strategy-registry-based resonance framework. Builds per-frequency
H × PC = R spectra from a signal via swappable harmonic kernels, ratio gates,
phase estimators, pairwise coupling metrics, and combine rules — with
surrogate normalization and a separate path for higher-order coupling
metrics (Phase 3).

Public entry points:

* :func:`biotuner.resonance.compute_resonance`
* :class:`biotuner.resonance.ResonanceConfig`
* :class:`biotuner.resonance.ResonanceResult`
* :class:`biotuner.resonance.HigherOrderResult`
* :func:`biotuner.resonance.with_surrogate_null`

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
